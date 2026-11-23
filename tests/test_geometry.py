"""Unit checks for the geometry and FLAME core.

Runs without the licensed FLAME files by using a toy model stored in FLAME's file format.

    python tests/test_geometry.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from tinypt.flame.model import FLAME, rodrigues
from tinypt.geometry import mesh as M
from tinypt.geometry import mouth as MO
from tinypt.geometry import point_cloud as PC
from tinypt.geometry.registration import nonrigid_icp

rng = np.random.default_rng(0)


def grid(n=20, hole=None):
    """Planar n x n grid in the xy plane (z = 0), optionally with a rectangular hole."""
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            if hole and hole[0] <= i < hole[1] and hole[2] <= j < hole[3]:
                continue
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            F += [[a, b, d], [a, d, c]]
    return V, np.array(F)


def uv_sphere(nu=24, nv=16, r=1.0):
    V = [[0, 0, r]]
    for i in range(1, nv):
        th = np.pi * i / nv
        for j in range(nu):
            ph = 2 * np.pi * j / nu
            V.append([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)])
    V.append([0, 0, -r])
    V = np.array(V)
    F = []
    for j in range(nu):
        F.append([0, 1 + j, 1 + (j + 1) % nu])
    for i in range(nv - 2):
        for j in range(nu):
            a, b = 1 + i * nu + j, 1 + i * nu + (j + 1) % nu
            F += [[a, a + nu, b], [b, a + nu, b + nu]]
    last = len(V) - 1
    for j in range(nu):
        F.append([last, 1 + (nv - 2) * nu + (j + 1) % nu, 1 + (nv - 2) * nu + j])
    return V, np.array(F)


def toy_flame(tmp):
    V, F = uv_sphere()
    n = len(V)
    W = rng.random((n, 5))
    W /= W.sum(1, keepdims=True)
    Jr = rng.random((5, n))
    Jr /= Jr.sum(1, keepdims=True)
    path = Path(tmp) / "toy_flame.npz"
    np.savez(
        path,
        v_template=V,
        shapedirs=0.01 * rng.standard_normal((n, 3, 400)),
        posedirs=0.001 * rng.standard_normal((n, 3, 36)),
        J_regressor=Jr,
        kintree_table=np.array([[4294967295, 0, 1, 1, 1], [0, 1, 2, 3, 4]]),
        weights=W,
        f=F,
    )
    return FLAME(path, n_shape=20, n_expr=10)


def check(name, cond, info=""):
    print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")
    if not cond:
        check.failed += 1


check.failed = 0


with tempfile.TemporaryDirectory() as tmp:
    fl = toy_flame(tmp)
    # FLAME layer
    V0, _ = fl(torch.zeros(1, 20))
    check("FLAME zero params == template", torch.allclose(V0[0], fl.v_template, atol=1e-6))
    r = torch.tensor([[0.3, -0.2, 0.5]])
    R = rodrigues(r)[0]
    check(
        "rodrigues is a rotation",
        torch.allclose(R @ R.T, torch.eye(3), atol=1e-6) and abs(torch.det(R) - 1) < 1e-6,
    )
    pose = torch.zeros(1, 15)
    pose[0, :3] = r
    Vp, J = fl(torch.zeros(1, 20), pose=pose)
    J0 = (fl.J_regressor @ fl.v_template)[0]
    expect = (fl.v_template - J0) @ R.T + J0
    check(
        "global rotation = rigid rotation about root joint",
        torch.allclose(Vp[0], expect, atol=1e-5),
    )
    beta = torch.randn(20)
    Vb = fl.shaped(beta).numpy() + 1e-4 * rng.standard_normal((len(fl.v_template), 3))
    b_fit, off = fl.fit_beta(Vb, reg=1e-8)
    check(
        "fit_beta recovers beta",
        torch.allclose(b_fit, beta, atol=1e-2),
        f"(max err {(b_fit - beta).abs().max():.2e})",
    )
    n0 = len(fl.v_template)
    P = np.array([[0, 0, 2.0], [0.1, 0, 2.0], [0, 0.1, 2.0]])
    fl.add_proxy(P, np.eye(5)[[2, 2, 2]], [[n0, n0 + 1, n0 + 2]])
    pose = torch.zeros(1, 15)
    pose[0, 6] = 0.4
    Vp, J = fl(torch.zeros(1, 20), pose=pose)
    d0 = (Vp[0, n0:] - J[0, 2]).norm(dim=1)
    d1 = (torch.tensor(P, dtype=torch.float32) - (fl.J_regressor @ fl.v_template)[2]).norm(dim=1)
    check("proxy vertices move rigidly with their joint", torch.allclose(d0, d1, atol=1e-5))

# topology
V, F = grid(20, hole=(8, 11, 6, 13))
loops = M.boundary_loops(F)
check("grid with hole has 2 boundary loops", len(loops) == 2)
hole_loop = min(loops, key=len)
ring, hops = M.k_ring(F, len(V), hole_loop, 2)
check("k_ring grows the seed set", len(ring) > len(hole_loop) and hops.max() == 2)

# red-green refinement
tri_mask = np.isin(F, ring).all(1)
S, F2 = M.refine_region(V, F, tri_mask)
V2 = S @ V
E = np.sort(np.concatenate([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]]), 1)
_, cnt = np.unique(E, axis=0, return_counts=True)
check("refinement: S rows are affine (sum to 1)", np.allclose(np.asarray(S.sum(1)).ravel(), 1))
check("refinement: manifold (every edge in <= 2 faces)", cnt.max() <= 2)
check("refinement: no new boundary (no T-junctions)", len(M.boundary_loops(F2)) == len(loops))
check(
    "refinement: planar area preserved",
    np.isclose(M.face_normals_areas(V2, F2)[1].sum(), M.face_normals_areas(V, F)[1].sum()),
)
check("refinement: more faces in the region", len(F2) > len(F) + 3 * tri_mask.sum())

# Laplacian editing
# Non-planar rest mesh, so that the cotangent weights vary.
Vw = V.copy()
Vw[:, 2] = 0.05 * np.sin(2 * V[:, 0]) * np.cos(V[:, 1])
tri, bary = M.sample_surface(Vw, F, np.arange(len(F)), rng, 300)
B = PC.binding_matrix(tri, bary, F, len(V))
dfm = M.LaplacianDeformer(Vw, F, B, w=10)
check("Laplacian editing: rest targets -> rest mesh", np.abs(dfm(B @ Vw) - Vw).max() < 1e-6)
bump = Vw.copy()
bump[:, 2] += 0.2 * np.exp(-4 * (V[:, 0] ** 2 + V[:, 1] ** 2))
rec = dfm(B @ bump)
err, base = np.abs(rec - bump).mean(), np.abs(Vw - bump).mean()
# Soft handles and preserved rest differential coordinates: a large bump is recovered
# mostly, not exactly.
check(
    "Laplacian editing: recovers smooth bump from surface points",
    err < 0.25 * base,
    f"(err {err:.4f} vs {base:.4f})",
)

# FPS with forced seeds, binding via a refinement operator
Pc = rng.random((500, 3))
idx = PC.farthest_point_sampling(Pc, 50, start=np.arange(5))
check(
    "FPS keeps forced seeds and is unique", (idx[:5] == np.arange(5)).all() and len(set(idx)) == 50
)
tri2, bary2 = M.sample_surface(V2, F2, np.arange(len(F2)), rng, 100)
B2 = PC.binding_matrix(tri2, bary2, F2, None, S)
check(
    "binding through S == binding on refined mesh",
    np.allclose(B2 @ V, (V2[F2[tri2]] * bary2[..., None]).sum(1)),
)

# non-rigid ICP: sphere -> ellipsoid with a different tessellation
Vs, Fs = uv_sphere(24, 16)
Vt, Ft = uv_sphere(40, 30)
Vt = Vt * np.array([1.2, 0.9, 1.1])
X, hist = nonrigid_icp(Vs, Fs, Vt, Ft, stiffness=(20, 5, 1), inner=3, log=lambda *a: None)
r = np.linalg.norm(X / np.array([1.2, 0.9, 1.1]), axis=1)
check(
    "non-rigid ICP fits the ellipsoid",
    np.abs(r - 1).mean() < 0.02,
    f"(mean |r-1| {np.abs(r - 1).mean():.4f})",
)

# mouth structure on a plane with a 'mouth' hole (z towards +camera)
loop = MO.inner_lip_loop(V, F, ring)
sp_ = MO.split_upper_lower(V, loop)
check("inner-lip loop = hole boundary", set(loop) == set(hole_loop))
check("upper lip above lower lip", V[sp_["up"], 1].mean() > V[sp_["lo"], 1].mean())
Pp, Wp, Fp, part = MO.build_proxies(V, loop, sp_, len(V))
Vall = np.concatenate([V, Pp])
check("proxy weights sum to 1", np.allclose(Wp.sum(1), 1))
check("cavity lies behind the face (z < 0)", Pp[part == "cavity", 2].max() < 1e-9)
teeth = Fp[np.isin(Fp.min(1), np.where(part != "cavity")[0] + len(V))]
fn, _ = M.face_normals_areas(Vall, teeth)
check("teeth face the camera", (fn[:, 2] > 0).all())

print("\nall checks passed" if check.failed == 0 else f"\n{check.failed} check(s) FAILED")
sys.exit(check.failed > 0)
