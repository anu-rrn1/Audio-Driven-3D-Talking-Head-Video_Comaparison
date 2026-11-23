"""FLAME head model in PyTorch: shape and expression blendshapes, pose correctives and
linear blend skinning.

Model files (free registration at https://flame.is.tue.mpg.de) go in data/flame/:
    generic_model.pkl                 FLAME 2020/2023 (read without installing chumpy)
    FLAME_masks.pkl                   optional vertex lists per region ("lips", "face", ...)
    mediapipe_landmark_embedding.npz  optional MediaPipe -> FLAME barycentric landmarks
A pre-converted flame_model.npz with the same keys is also accepted.

Joints: 0 global, 1 neck, 2 jaw, 3 left eye, 4 right eye; pose is [5 * 3] axis-angle.
"""

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import ROOT
from ..geometry import mesh as M

N_FLAME_VERTICES = 5023
N_HEAD_VERTICES = 3931  # vertices before the eyeballs
N_IDENTITY_COMPONENTS = 300  # shapedirs = 300 identity + 100 expression components


# Reading FLAME pickles without chumpy


class _ChumpyStub:
    """Stands in for chumpy.Ch while unpickling and keeps the raw array."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})

    def __array__(self, dtype=None):
        a = np.asarray(self.__dict__.get("x", self.__dict__.get("a")))
        return a.astype(dtype) if dtype else a


def _install_chumpy_stub():
    if "chumpy" in sys.modules:
        return

    def module_getattr(attr):
        if attr.startswith("__"):  # __file__, __path__ etc. must stay missing
            raise AttributeError(attr)
        return _ChumpyStub

    for name in [
        "chumpy",
        "chumpy.ch",
        "chumpy.ch_ops",
        "chumpy.reordering",
        "chumpy.utils",
        "chumpy.logic",
    ]:
        module = types.ModuleType(name)
        module.__getattr__ = module_getattr
        sys.modules[name] = module


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)  # J_regressor is sparse


def load_flame_arrays(path):
    path = Path(path)
    if path.suffix == ".npz":
        data = dict(np.load(path, allow_pickle=True))
    else:
        _install_chumpy_stub()
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    keys = ["v_template", "shapedirs", "posedirs", "J_regressor", "kintree_table", "weights", "f"]
    return {k: _dense(data[k]) for k in keys}


def rodrigues(r):
    """Axis-angle [..., 3] -> rotation matrix [..., 3, 3]."""
    angle = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = r / angle
    kx, ky, kz = k.unbind(-1)
    z = torch.zeros_like(kx)
    K = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], -1).reshape(*r.shape[:-1], 3, 3)
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


class FLAME(nn.Module):
    def __init__(self, model_path, n_shape=100, n_expr=50, masks_path=None, lmk_path=None):
        super().__init__()
        arrays = load_flame_arrays(model_path)
        shapedirs = arrays["shapedirs"].astype(np.float32)  # [V, 3, 400]
        n_id = N_IDENTITY_COMPONENTS
        self.n_shape, self.n_expr = n_shape, n_expr
        self.register_buffer("v_template", torch.tensor(arrays["v_template"], dtype=torch.float32))
        self.register_buffer("shapedirs", torch.tensor(shapedirs[:, :, :n_shape]))
        self.register_buffer("exprdirs", torch.tensor(shapedirs[:, :, n_id : n_id + n_expr]))
        posedirs = arrays["posedirs"].astype(np.float32)  # [V, 3, 36]
        self.register_buffer(
            "posedirs", torch.tensor(posedirs.reshape(-1, posedirs.shape[-1]).T)
        )  # [36, V*3]
        self.register_buffer(
            "J_regressor", torch.tensor(arrays["J_regressor"], dtype=torch.float32)
        )
        self.register_buffer("lbs_weights", torch.tensor(arrays["weights"], dtype=torch.float32))
        parents = arrays["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        self.parents = parents.tolist()
        self.faces = arrays["f"].astype(np.int64)
        self.masks = self._load_masks(masks_path)
        self.lmk = self._load_landmarks(lmk_path)

    @staticmethod
    def _load_masks(path):
        if path is None or not Path(path).exists():
            return {}
        with open(path, "rb") as f:
            masks = pickle.load(f, encoding="latin1")
        return {k: np.asarray(v, dtype=np.int64) for k, v in masks.items()}

    @staticmethod
    def _load_landmarks(path):
        if path is None or not Path(path).exists():
            return None
        data = np.load(path, allow_pickle=True)
        face_idx = data["lmk_face_idx"].astype(np.int64)
        bary = data["lmk_b_coords"].astype(np.float32)
        if "landmark_indices" in data:
            mp_idx = data["landmark_indices"].astype(np.int64)
        else:
            mp_idx = np.arange(len(face_idx))
        return dict(face_idx=face_idx, bary=bary, mp_idx=mp_idx)

    # Regions

    def region(self, name):
        """Vertex ids of a named region. Without FLAME_masks.pkl, "lips" and "face" are
        derived from the template mesh."""
        if name in self.masks:
            return self.masks[name]
        if name == "lips":
            self.masks["lips"] = self._derive_lips()
            return self.masks["lips"]
        if name == "face":
            self.masks["face"] = self._derive_face()
            return self.masks["face"]
        raise KeyError(f"region '{name}' needs FLAME_masks.pkl (have: {sorted(self.masks)})")

    def _template_loops(self):
        n = N_FLAME_VERTICES
        V = self.v_template[:n].cpu().numpy().astype(np.float64)
        F = self.faces[(self.faces < n).all(1)]
        return V, F, [loop for loop in M.boundary_loops(F) if len(loop) >= 8]

    def _derive_lips(self, hops=3):
        """Lip region = mouth-hole boundary loop grown by `hops` edge hops.

        FLAME 2020 has two boundary loops: the neck (largest extent) and the mouth. After
        dropping the neck loop, the mouth is the most central remaining loop.
        """
        V, F, loops = self._template_loops()
        if len(loops) < 2:
            raise RuntimeError(
                "could not find the mouth boundary loop in the FLAME template; "
                "provide FLAME_masks.pkl"
            )
        loops = sorted(loops, key=lambda loop: np.ptp(V[loop], 0).max())[:-1]
        mouth = min(loops, key=lambda loop: abs(V[loop, 0].mean()) + 0.01 * V[loop, 1].mean())
        idx, _ = M.k_ring(F, len(V), mouth, hops)
        return idx

    def _derive_face(self):
        """Front of the head (vertices with z above the 40th percentile); used for plots."""
        V = self.v_template[:N_FLAME_VERTICES].cpu().numpy()
        return np.where(V[:, 2] > np.percentile(V[:, 2], 40))[0]

    # Geometry

    def shaped(self, beta, offsets=None):
        """Identity mesh in the rest pose: template + shape blendshapes (+ free offsets)."""
        V = self.v_template + torch.einsum("vcs,...s->...vc", self.shapedirs, beta)
        return V if offsets is None else V + offsets

    def forward(self, beta, expr=None, pose=None, offsets=None):
        """beta [B, n_shape], expr [B, n_expr], pose [B, 15] -> vertices [B, V, 3],
        joints [B, 5, 3]."""
        B = beta.shape[0]
        expr = beta.new_zeros(B, self.n_expr) if expr is None else expr
        pose = beta.new_zeros(B, 15) if pose is None else pose
        v = self.shaped(beta, offsets) + torch.einsum("vce,be->bvc", self.exprdirs, expr)
        J = torch.einsum("jv,bvc->bjc", self.J_regressor, v)
        R = rodrigues(pose.reshape(B, -1, 3))  # [B, 5, 3, 3]
        eye = torch.eye(3, device=beta.device)
        pose_features = (R[:, 1:] - eye).reshape(B, -1)  # [B, 36]
        v = v + (pose_features @ self.posedirs).reshape(B, -1, 3)
        return self._skin(v, J, R)

    def _skin(self, v, J, R):
        B, n_joints = J.shape[:2]
        G = []
        for j in range(n_joints):
            p = self.parents[j]
            t = J[:, j] if p < 0 else J[:, j] - J[:, p]
            T = torch.zeros(B, 4, 4, device=v.device)
            T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = R[:, j], t, 1.0
            G.append(T if p < 0 else G[p] @ T)
        G = torch.stack(G, 1)  # [B, J, 4, 4] world transforms
        J_h = torch.cat([J, torch.zeros(B, n_joints, 1, device=v.device)], -1)[..., None]
        # Remove the rest-pose joint offset so that G maps rest positions to posed positions.
        G = G - torch.cat([torch.zeros(B, n_joints, 4, 3, device=v.device), G @ J_h], -1)
        T_v = torch.einsum("vj,bjxy->bvxy", self.lbs_weights, G)
        v_h = torch.cat([v, torch.ones_like(v[..., :1])], -1)
        posed = (T_v @ v_h[..., None])[..., :3, 0]
        J_posed = G[:, :, :3, 3] + (G[:, :, :3, :3] @ J[..., None])[..., 0]
        return posed, J_posed

    def landmarks(self, V):
        """Barycentric landmarks on the mesh surface: V [B, V, 3] -> [B, L, 3]."""
        if self.lmk is None:
            raise RuntimeError("no landmark embedding loaded")
        tri = torch.as_tensor(self.faces[self.lmk["face_idx"]], device=V.device)
        bary = torch.as_tensor(self.lmk["bary"], device=V.device)
        return (V[:, tri] * bary[None, ..., None]).sum(2)

    def jaw_pose(self, jaw):
        """Full [B, 15] pose vector with only the jaw rotation [B, 3] set."""
        B = jaw.shape[0]
        return torch.cat([jaw.new_zeros(B, 6), jaw, jaw.new_zeros(B, 6)], 1)

    # Extra rigged geometry (mouth cavity and teeth)

    def add_proxy(self, P_rest, weights, faces):
        """Append vertices that follow only the skeleton (no blendshapes).

        P_rest [m, 3] is given in the rest pose of the identity being animated, so call this
        after the identity is fixed. weights [m, 5] are LBS weights, faces index the extended
        vertex array.
        """
        m = len(P_rest)
        dev = self.v_template.device

        def pad(t, *shape):
            return torch.cat([t, torch.zeros(*shape, device=dev)], 0)

        self.v_template = torch.cat(
            [self.v_template, torch.as_tensor(P_rest, dtype=torch.float32, device=dev)], 0
        )
        self.shapedirs = pad(self.shapedirs, m, 3, self.n_shape)
        self.exprdirs = pad(self.exprdirs, m, 3, self.n_expr)
        pd = self.posedirs.T.reshape(-1, 3, self.posedirs.shape[0])
        self.posedirs = pad(pd, m, 3, pd.shape[-1]).reshape(-1, pd.shape[-1]).T.contiguous()
        self.J_regressor = torch.cat(
            [self.J_regressor, torch.zeros(self.J_regressor.shape[0], m, device=dev)], 1
        )
        self.lbs_weights = torch.cat(
            [self.lbs_weights, torch.as_tensor(weights, dtype=torch.float32, device=dev)], 0
        )
        self.faces = np.concatenate([self.faces, np.asarray(faces, dtype=np.int64)], 0)

    # Identity fitting

    def fit_beta(self, V_target, reg=1e-3):
        """Closed-form least-squares beta for a FLAME-topology mesh in FLAME's frame.

        Returns beta and the residual per-vertex offsets (detail beyond the shape space).
        """
        A = self.shapedirs.reshape(-1, self.n_shape)
        r = (torch.as_tensor(V_target, dtype=torch.float32) - self.v_template).reshape(-1)
        beta = torch.linalg.solve(A.T @ A + reg * torch.eye(self.n_shape), A.T @ r)
        offsets = torch.as_tensor(V_target, dtype=torch.float32) - self.shaped(beta)
        return beta, offsets


def load_flame(cfg, device="cpu", n_expr=None):
    """Build the FLAME model from the files in cfg["flame"]["dir"]."""
    flame_dir = ROOT / cfg["flame"]["dir"]
    candidates = ["generic_model.pkl", "flame2023.pkl", "flame_model.npz"]
    model = next((flame_dir / n for n in candidates if (flame_dir / n).exists()), None)
    if model is None:
        raise FileNotFoundError(
            f"FLAME model not found in {flame_dir}. Register at https://flame.is.tue.mpg.de, "
            "download FLAME 2020 and put generic_model.pkl there (see README)."
        )
    return FLAME(
        model,
        cfg["flame"]["n_shape"],
        n_expr or cfg["flame"]["n_expr"],
        flame_dir / "FLAME_masks.pkl",
        flame_dir / "mediapipe_landmark_embedding.npz",
    ).to(device)
