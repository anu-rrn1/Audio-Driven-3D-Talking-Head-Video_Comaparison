"""Step 3: lip point cloud. Refined mouth mesh -> lip point cloud, and back.

Mesh -> points: dense area-weighted samples on the refined mouth region plus the inner-lip
ring, reduced by farthest-point sampling. Every point is a barycentric combination of refined
vertices, which are linear in the FLAME vertices, so the cloud is one sparse operator
P = B @ V. B is built on the template and is identity-independent: the same B gives the lip
cloud of every VOCASET speaker and frame and of the user identity.

Points -> mesh: Laplacian editing (differential coordinates preserved, points as soft
handles) turns a lip cloud back into a deformed FLAME mesh. It is validated here on known
FLAME expression and jaw motions of the identity.

Outputs: data/identity/{lip_binding.npz, lip_cloud.npz}, outputs/flame_pipeline/03_lip_cloud/
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from tinypt import visualization as viz
from tinypt.config import FLAME_OUTPUTS, IDENTITY_DIR, ensure_dir, load_config
from tinypt.flame import load_flame
from tinypt.geometry import mesh as M
from tinypt.geometry import mouth as MO
from tinypt.geometry import point_cloud as PC
from tinypt.io_utils import save_json

N_DEMO_FRAMES = 48


def build_binding(V_template, S, F_ref, ref_tri, loop, cfg, rng):
    """Lip-cloud operator B [n_points, n_vertices]: the inner-lip ring is always kept and the
    rest is filled by farthest-point sampling of dense samples on the refined mouth region."""
    n = len(V_template)
    V_ref_template = S @ V_template
    tri, bary = M.sample_surface(
        V_ref_template, F_ref, np.where(ref_tri)[0], rng, cfg["n_candidates"]
    )
    B_samples = PC.binding_matrix(tri, bary, F_ref, None, S)
    B_ring = sp.coo_matrix(
        (np.ones(len(loop)), (np.arange(len(loop)), loop)), shape=(len(loop), n)
    ).tocsr()
    B_candidates = sp.vstack([B_ring, B_samples]).tocsr()
    candidates = B_candidates @ V_template
    keep = PC.farthest_point_sampling(candidates, cfg["n_points"], start=np.arange(len(loop)))
    return B_candidates[keep], candidates


def demo_motion(flame, identity, n_frames):
    """Smooth synthetic expression + jaw sequence of the identity. Returns the full motion
    and the jaw-only (rig) motion, both [T, n, 3]."""
    beta, offsets = torch.tensor(identity["beta"]), torch.tensor(identity["offsets"])
    g = torch.Generator().manual_seed(0)
    tt = torch.linspace(0, 2 * np.pi, n_frames)
    expr = torch.zeros(n_frames, flame.n_expr)
    expr[:, :10] = (
        1.2
        * torch.randn(10, generator=g)[None]
        * torch.sin(tt[:, None] * torch.arange(1, 11)[None] / 3)
    )
    pose = torch.zeros(n_frames, 15)
    pose[:, 6] = 0.12 * (0.5 - 0.5 * torch.cos(2 * tt))  # jaw opening
    with torch.no_grad():
        bb = beta[None].expand(n_frames, -1)
        oo = offsets[None].expand(n_frames, -1, -1)
        V_seq = flame(bb, expr, pose, oo)[0].numpy().astype(np.float64)
        V_jaw = flame(bb, None, pose, oo)[0].numpy().astype(np.float64)
    return V_seq, V_jaw


def main():
    cfg = load_config()
    cloud_cfg = cfg["lip_cloud"]
    out = ensure_dir(FLAME_OUTPUTS / "03_lip_cloud")
    flame = load_flame(cfg)
    identity = np.load(IDENTITY_DIR / "identity.npz")
    S = sp.load_npz(IDENTITY_DIR / "refine_S.npz")
    F, loop = identity["faces"], identity["inner_loop"]
    V_id = identity["V_id"].astype(np.float64)
    n = len(V_id)
    rng = np.random.default_rng(0)

    # 1. Identity-independent binding B, built deterministically on the FLAME template.
    V_template = flame.v_template.numpy().astype(np.float64)
    B, candidates = build_binding(
        V_template, S, identity["F_ref"], identity["ref_tri"], loop, cloud_cfg, rng
    )
    sp.save_npz(IDENTITY_DIR / "lip_binding.npz", B)

    # 2. Lip cloud of the identity and point features.
    P0 = B @ V_id
    P0n, c0, s0 = PC.normalize_cloud(P0)
    normals = PC.estimate_normals(P0n, k=10, towards=np.array([0.0, 0.0, 1.0]))  # camera at +z
    knn = PC.knn_indices(P0n, cloud_cfg["knn"])
    is_ring = np.zeros(len(P0), bool)
    is_ring[: len(loop)] = True
    np.savez(
        IDENTITY_DIR / "lip_cloud.npz",
        P0=P0,
        P0n=P0n,
        c0=c0,
        s0=s0,
        normals=normals,
        knn=knn,
        n_ring=len(loop),
        up_pts=np.where(np.isin(loop, identity["up"]))[0],
        lo_pts=np.where(np.isin(loop, identity["lo"]))[0],
    )

    # 3. Points -> mesh by Laplacian editing, validated on known motions of this identity.
    region = identity["region_idx"]
    in_region = np.zeros(n, bool)
    in_region[region] = True
    deformer = M.LaplacianDeformer(
        V_id, F, B, w=cloud_cfg["deform_w"], fixed=np.where(~in_region)[0]
    )
    V_seq, V_jaw = demo_motion(flame, identity, N_DEMO_FRAMES)
    P_seq = np.einsum("pv,tvc->tpc", B.toarray(), V_seq)  # ground-truth dynamic lip cloud

    err_laplacian, err_rig_only, chamfer, point_to_surface = [], [], [], []
    for k in range(N_DEMO_FRAMES):
        V_rec = deformer(P_seq[k], V_fixed=V_jaw[k])  # outside the mouth: follow the jaw rig
        err_laplacian.append(np.linalg.norm(V_rec[region] - V_seq[k, region], axis=1).mean())
        err_rig_only.append(np.linalg.norm(V_jaw[k, region] - V_seq[k, region], axis=1).mean())
        chamfer.append(PC.chamfer(B @ V_rec, P_seq[k]))
        if k % 8 == 0:
            point_to_surface.append(M.point_to_surface(P_seq[k], V_rec, F, 50000).mean())
    openness = MO.openness(
        V_seq, int(identity["up_mid"]), int(identity["lo_mid"]), float(identity["mouth_w"])
    )
    k_open = int(np.argmax(openness))
    V_open_rec = deformer(P_seq[k_open], V_fixed=V_jaw[k_open])

    D = P_seq - P0[None]
    stats = dict(
        n_points=len(P0),
        n_inner_lip_ring_points=len(loop),
        cloud_radius_mm=1000 * s0,
        binding_nnz_per_point=float(B.nnz / B.shape[0]),
        lap_region_vertex_err_mm=1000 * float(np.mean(err_laplacian)),
        jaw_rig_only_baseline_err_mm=1000 * float(np.mean(err_rig_only)),
        chamfer_reconstructed_vs_target_mm=1000 * float(np.mean(chamfer)),
        point_to_surface_mm=1000 * float(np.mean(point_to_surface)),
        max_point_displacement_mm=1000 * float(np.linalg.norm(D, axis=-1).max()),
        note="points->mesh evaluated on synthetic FLAME expression + jaw motions of this identity",
    )
    save_json(stats, out / "lip_cloud_stats.json")
    print(stats)

    # 4. Figures (image-style coordinates).
    I = viz.flame_to_img
    ring_flag = np.r_[np.ones(len(loop)), np.zeros(len(candidates) - len(loop))]
    viz.points_views(
        I(candidates),
        ring_flag,
        out / "01_candidates.png",
        f"{len(candidates)} candidate samples (yellow = inner-lip ring)",
        size=3,
    )
    viz.points_views(
        I(P0),
        is_ring.astype(float),
        out / "02_fps_lip_cloud.png",
        f"FPS lip cloud, N={len(P0)}",
        size=10,
    )
    edges = np.array([(i, knn[i, j]) for i in range(len(P0)) for j in range(4)])
    viz.points_views(
        I(P0),
        P0[:, 1],
        out / "03_knn_graph.png",
        f"k-NN graph (4 of {cloud_cfg['knn']} drawn)",
        size=6,
        edges=edges,
    )
    viz.points_views(
        I(P0), P0[:, 1], out / "04_pca_normals.png", "PCA normals", size=6, normals=I(normals)
    )
    magnitude = np.linalg.norm(D, axis=-1)
    viz.points_views(
        I(P_seq[k_open]),
        magnitude[k_open] * 1000,
        out / "05_open_mouth_cloud.png",
        "open-mouth cloud (colour = displacement, mm)",
        size=10,
        cmap="turbo",
        extra=I(P0),
    )
    err_open = np.linalg.norm(V_open_rec - V_seq[k_open], axis=1) * 1000
    region_faces = np.where(identity["region_tri"])[0]
    viz.mesh_views(
        I(V_open_rec),
        F[region_faces],
        err_open,
        out / "06_points_to_mesh_error.png",
        "Laplacian points->mesh: vertex error (mm)",
        cmap="magma",
    )

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(np.array(err_rig_only) * 1000, label="jaw rig only (no lip deformation)")
    ax.plot(np.array(err_laplacian) * 1000, label="Laplacian editing from lip cloud")
    ax.set_xlabel("frame")
    ax.set_ylabel("mouth-region vertex error (mm)")
    ax.legend()
    ax.grid(alpha=0.3)
    viz.save_figure(fig, out / "07_points_to_mesh_curve.png")

    frames = np.arange(0, N_DEMO_FRAMES, 2)
    viz.plotly_points_animation(
        I(P_seq[frames]),
        magnitude[frames] * 1000,
        [str(i) for i in frames],
        out / "interactive_dynamic_lip_cloud.html",
    )
    print(
        "saved", IDENTITY_DIR / "lip_cloud.npz", IDENTITY_DIR / "lip_binding.npz", "figures in", out
    )


if __name__ == "__main__":
    main()
