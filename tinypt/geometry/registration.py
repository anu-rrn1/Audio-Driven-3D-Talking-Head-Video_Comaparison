"""Registration of the FLAME template to a user-provided face mesh of any topology.

1. Similarity alignment: Umeyama on 3D landmarks if given, otherwise centroid / scale
   matching followed by trimmed rigid ICP.
2. Non-rigid ICP: closest-point correspondences filtered by distance and normal agreement,
   with a cotangent-Laplacian stiffness term annealed from stiff to loose. Vertices without
   a valid match (back of the head, neck, eyeballs) follow the smooth deformation.
3. Projection onto FLAME: shape coefficients beta plus per-vertex offsets, so the identity
   is reproduced exactly while keeping FLAME's rig.
"""

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from . import mesh as M


def _surface_samples(V, F, n, seed=0):
    """Dense surface samples plus the vertices, with their normals."""
    rng = np.random.default_rng(seed)
    tri, bary = M.sample_surface(V, F, np.arange(len(F)), rng, n)
    fn, _ = M.face_normals_areas(V, F)
    points = np.concatenate([(V[F[tri]] * bary[..., None]).sum(1), V])
    normals = np.concatenate([fn[tri], M.vertex_normals(V, F)])
    return points, normals


def _matches(X, X_normals, tree, target_normals, max_dist, min_cos):
    dist, j = tree.query(X)
    ok = (dist < max_dist) & ((X_normals * target_normals[j]).sum(1) > min_cos)
    return j, ok, dist


def similarity_init(Vs, Fs, Vt, Ft, lmk_src=None, lmk_tgt=None, iters=30, seed=0):
    """Similarity (s, R, t) mapping the source (template) onto the target mesh.

    Without landmarks the template must face roughly the same direction as the target.
    """
    if lmk_src is not None:
        return M.umeyama(lmk_src, lmk_tgt)
    cs, ct = Vs.mean(0), Vt.mean(0)
    ss = np.sqrt(((Vs - cs) ** 2).sum(1).mean())
    st = np.sqrt(((Vt - ct) ** 2).sum(1).mean())
    s, R, t = st / ss, np.eye(3), ct - st / ss * cs
    T, _ = _surface_samples(Vt, Ft, 50000, seed)
    tree = cKDTree(T)
    for _ in range(iters):
        X = M.apply_sim(Vs, s, R, t)
        dist, j = tree.query(X)
        keep = dist < np.percentile(dist, 70)  # trimmed: ignore template parts the scan lacks
        s, R, t = M.umeyama(Vs[keep], T[j[keep]])
    return s, R, t


def nonrigid_icp(
    V,
    F,
    Vt,
    Ft,
    stiffness=(50, 20, 8, 3, 1),
    inner=4,
    max_dist=None,
    min_cos=0.5,
    lmk_idx=None,
    lmk_tgt=None,
    lmk_w=5.0,
    log=print,
):
    """Deform the template (V, F), already similarity-aligned, onto the target (Vt, Ft).

    Each step solves  min_X  sum_i c_i ||X_i - T(i)||^2 + a^2 ||L (X - V)||^2
                             + lmk_w^2 ||X[lmk] - lmk_tgt||^2
    for the current correspondences T(i) and stiffness a.
    """
    n = len(V)
    L = M.cotangent_laplacian(V, F)
    T, T_normals = _surface_samples(Vt, Ft, 100000)
    tree = cKDTree(T)
    if max_dist is None:
        max_dist = 0.05 * np.ptp(Vt, 0).max()
    X = V.copy()
    delta = L @ V
    history = []
    for a in stiffness:
        for _ in range(inner):
            j, ok, dist = _matches(X, M.vertex_normals(X, F), tree, T_normals, max_dist, min_cos)
            A = [a * L, sp.diags(ok.astype(float))]
            b = [a * delta, ok[:, None] * T[j]]
            if lmk_idx is not None:
                A.append(
                    sp.coo_matrix(
                        (np.full(len(lmk_idx), lmk_w), (np.arange(len(lmk_idx)), lmk_idx)),
                        shape=(len(lmk_idx), n),
                    )
                )
                b.append(lmk_w * lmk_tgt)
            A = sp.vstack(A).tocsc()
            lu = splu((A.T @ A + 1e-9 * sp.eye(n)).tocsc())
            rhs = A.T @ np.concatenate(b)
            X = np.stack([lu.solve(rhs[:, c]) for c in range(3)], 1)
        history.append(
            dict(stiffness=a, matched_frac=float(ok.mean()), mean_dist=float(dist[ok].mean()))
        )
        log(f"  NICP stiffness {a:>5}: matched {ok.mean():.0%}, mean dist {dist[ok].mean():.4f}")
    return X, history


def register_identity(flame, Vt, Ft, lmk_tgt=None, cfg=None, log=print):
    """Register FLAME to a user mesh.

    Returns a dict with beta, per-vertex offsets, the registered mesh in FLAME's frame
    (V_id), the template-to-scan similarity and fit metrics.
    """
    cfg = cfg or {}
    Vs = flame.v_template.cpu().numpy().astype(np.float64)
    Fs = flame.faces
    lmk_src = lmk_idx = None
    if lmk_tgt is not None:  # 3D landmarks on the scan, in FLAME's landmark order
        lmk_src = flame.landmarks(torch.tensor(Vs, dtype=torch.float32)[None])[0].numpy()
        lmk_idx = Fs[flame.lmk["face_idx"], np.argmax(flame.lmk["bary"], 1)]

    s, R, t = similarity_init(Vs, Fs, Vt, Ft, lmk_src, lmk_tgt)
    X0 = M.apply_sim(Vs, s, R, t)
    X, history = nonrigid_icp(
        X0,
        Fs,
        Vt,
        Ft,
        stiffness=tuple(cfg.get("stiffness", (50, 20, 8, 3, 1))),
        inner=cfg.get("inner", 4),
        min_cos=cfg.get("min_cos", 0.5),
        lmk_idx=lmk_idx,
        lmk_tgt=lmk_tgt,
        log=log,
    )
    V_flame_frame = M.invert_sim(X, s, R, t)
    beta, offsets = flame.fit_beta(V_flame_frame)

    d_fit = M.point_to_surface(Vt, M.apply_sim(V_flame_frame, s, R, t), Fs)
    d_beta = M.point_to_surface(Vt, M.apply_sim(flame.shaped(beta).numpy(), s, R, t), Fs)
    scale = np.ptp(Vt, 0).max()
    metrics = dict(
        scan_to_registered_mean=float(d_fit.mean() / scale),
        scan_to_registered_p95=float(np.percentile(d_fit, 95) / scale),
        scan_to_beta_only_mean=float(d_beta.mean() / scale),
        offset_rms_m=float(offsets.pow(2).sum(-1).mean().sqrt()),
        note="distances are relative to the scan's bounding-box size",
        nicp=history,
    )
    return dict(
        beta=beta.numpy(),
        offsets=offsets.numpy(),
        V_id=V_flame_frame,
        sim=(s, R, t),
        metrics=metrics,
    )
