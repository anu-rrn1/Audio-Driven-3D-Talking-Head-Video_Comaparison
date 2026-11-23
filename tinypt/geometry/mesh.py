"""Triangle-mesh processing: differential geometry, alignment, sampling, refinement and
Laplacian editing.

Conventions: V is an [n, 3] float array of vertices, F an [m, 3] int array of triangles.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

# Topology


def edges_of(F):
    """Directed half-edges (01, 12, 20) of every triangle, stacked as [3m, 2]."""
    return np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])


def adjacency(F, n):
    """Binary vertex adjacency matrix [n, n]."""
    I = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    J = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def boundary_vertices(F, n):
    E = np.sort(edges_of(F), axis=1)
    unique, count = np.unique(E, axis=0, return_counts=True)
    is_boundary = np.zeros(n, bool)
    is_boundary[unique[count == 1].ravel()] = True
    return is_boundary


def boundary_loops(F):
    """Ordered boundary loops (arrays of vertex ids) of a manifold triangle mesh."""
    E = edges_of(F)
    _, inverse, count = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    boundary_edges = E[count[inverse.ravel()] == 1]
    next_vertex = {int(a): int(b) for a, b in boundary_edges}
    loops, seen = [], set()
    for start in next_vertex:
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen and v in next_vertex:
            seen.add(v)
            loop.append(v)
            v = next_vertex[v]
        loops.append(np.array(loop))
    return loops


def k_ring(F, n, seeds, k):
    """Vertices within k edge hops of `seeds` (breadth-first search) and their hop distance."""
    A = adjacency(F, n)
    dist = np.full(n, -1)
    frontier = np.unique(seeds)
    dist[frontier] = 0
    for hop in range(1, k + 1):
        neighbours = np.unique(A[frontier].indices)
        frontier = neighbours[dist[neighbours] < 0]
        dist[frontier] = hop
    idx = np.where(dist >= 0)[0]
    return idx, dist[idx]


def orient_faces_towards_camera(V, F):
    """Flip triangles whose normal points away from a camera looking along +z."""
    normals = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    F = F.copy()
    flip = normals[:, 2] > 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


# Differential geometry


def face_normals_areas(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / (a[:, None] + 1e-12), 0.5 * a


def vertex_normals(V, F):
    """Area-weighted vertex normals."""
    fn, fa = face_normals_areas(V, F)
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn * fa[:, None])
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def cotangent_laplacian(V, F):
    """Cotangent Laplacian L = D - W (positive semi-definite convention)."""
    n = len(V)
    I, J, W = [], [], []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:  # angle at k, opposite edge (i, j)
        u = V[F[:, i]] - V[F[:, k]]
        v = V[F[:, j]] - V[F[:, k]]
        cot = (u * v).sum(1) / (np.linalg.norm(np.cross(u, v), axis=1) + 1e-12)
        I += [F[:, i], F[:, j]]
        J += [F[:, j], F[:, i]]
        W += [cot / 2, cot / 2]
    Wm = sp.coo_matrix(
        (np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n)
    ).tocsr()
    return sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm


def mean_curvature(V, F):
    """Discrete mean curvature |H| = 0.5 * ||M^-1 L V|| (lumped mass matrix M).

    Boundary vertices have no valid estimate and are set to the median.
    """
    _, fa = face_normals_areas(V, F)
    mass = np.zeros(len(V))
    for k in range(3):
        np.add.at(mass, F[:, k], fa / 3)
    HN = (cotangent_laplacian(V, F) @ V) / (mass[:, None] + 1e-12)
    H = 0.5 * np.linalg.norm(HN, axis=1)
    boundary = boundary_vertices(F, len(V))
    H[boundary] = np.median(H[~boundary])
    return H


def taubin_smooth(V, F, iters=10, lam=0.5, mu=-0.53):
    """Taubin (lambda | mu) smoothing: Laplacian smoothing without shrinkage."""
    A = adjacency(F, len(V))
    degree = np.asarray(A.sum(1)).ravel()[:, None]
    V = V.copy()
    for _ in range(iters):
        V += lam * ((A @ V) / degree - V)
        V += mu * ((A @ V) / degree - V)
    return V


def edge_stats(V, F):
    E = edges_of(F)
    length = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return dict(
        n_vertices=int(len(V)),
        n_faces=int(len(F)),
        edge_len_mean=float(length.mean()),
        edge_len_std=float(length.std()),
        surface_area=float(face_normals_areas(V, F)[1].sum()),
    )


# Similarity alignment


def umeyama(src, dst, with_scale=True):
    """Similarity transform (s, R, t) minimising ||s R src + t - dst||^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    s = (sig * d).sum() / ((S**2).sum() / len(src)) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def apply_sim(X, s, R, t):
    return s * X @ R.T + t


def invert_sim(X, s, R, t):
    return ((X - t) / s) @ R


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# Sampling and I/O


def sample_surface(V, F, tri_ids, rng, n):
    """n area-weighted samples on the triangles `tri_ids`: (triangle id, barycentrics)."""
    _, fa = face_normals_areas(V, F)
    p = fa[tri_ids] / fa[tri_ids].sum()
    tri = rng.choice(tri_ids, size=n, p=p)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return tri, np.stack([1 - u - v, u, v], 1)


def point_to_surface(P, V, F, n_samples=200000, seed=0):
    """Approximate distance from points P to the surface (V, F) using dense surface samples."""
    rng = np.random.default_rng(seed)
    tri, bary = sample_surface(V, F, np.arange(len(F)), rng, n_samples)
    samples = (V[F[tri]] * bary[..., None]).sum(1)
    dist, _ = cKDTree(np.concatenate([samples, V])).query(P)
    return dist


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def save_obj(path, V, F):
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in F:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def load_mesh(path):
    """Load a triangle mesh. OBJ is parsed directly (polygons are fan-triangulated);
    other formats (PLY, STL, ...) require trimesh."""
    path = str(path)
    if path.lower().endswith(".obj"):
        V, F = [], []
        with open(path) as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] == "v":
                    V.append([float(x) for x in tokens[1:4]])
                elif tokens[0] == "f":
                    ids = [int(x.split("/")[0]) - 1 for x in tokens[1:]]
                    F += [[ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1)]
        return np.array(V), np.array(F, dtype=np.int64)
    import trimesh

    mesh = trimesh.load(path, process=False, force="mesh")
    return np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)


# Adaptive refinement


def refine_region(V, F, tri_mask):
    """Conforming red-green refinement of the triangles in `tri_mask`.

    Red triangles are split 1->4 and their neighbours 1->2, so no T-junctions appear.
    New edge vertices use the Loop edge rule (3/8, 3/8, 1/8, 1/8) on interior edges and the
    midpoint on boundary edges; existing vertices do not move.

    Returns (S, F_new) where S is a sparse [n_new, n] operator with V_new = S @ V. Because
    the refinement is linear, the refined mesh follows any deformation of the original mesh.
    """
    n = len(V)
    E = edges_of(F)
    opposite = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])  # vertex opposite each half-edge
    unique, inverse = np.unique(np.sort(E, 1), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    nf = len(F)
    edge_of = inverse.reshape(3, nf).T  # [nf, 3] edge ids of (01, 12, 20)
    split = np.zeros(len(unique), bool)
    split[edge_of[tri_mask].ravel()] = True

    # Green closure: a triangle with two or more split edges is refined fully (red).
    red = tri_mask.copy()
    while True:
        n_split = split[edge_of].sum(1)
        grow = (~red) & (n_split >= 2)
        if not grow.any():
            break
        red |= grow
        split[edge_of[grow].ravel()] = True

    opposites = [[] for _ in range(len(unique))]
    for e, o in zip(inverse, opposite):
        opposites[e].append(o)
    rows, cols, vals = list(range(n)), list(range(n)), [1.0] * n
    new_id = -np.ones(len(unique), int)
    for r, e in enumerate(np.where(split)[0], start=n):
        new_id[e] = r
        a, b = unique[e]
        if len(opposites[e]) == 2:
            weights = [(a, 3 / 8), (b, 3 / 8), (opposites[e][0], 1 / 8), (opposites[e][1], 1 / 8)]
        else:
            weights = [(a, 0.5), (b, 0.5)]
        for c, w in weights:
            rows.append(r)
            cols.append(int(c))
            vals.append(w)
    S = sp.coo_matrix((vals, (rows, cols)), shape=(n + int(split.sum()), n)).tocsr()

    F_new = []
    for t in range(nf):
        v0, v1, v2 = F[t]
        m01, m12, m20 = (new_id[edge_of[t, k]] for k in range(3))
        if red[t]:
            F_new += [[v0, m01, m20], [m01, v1, m12], [m20, m12, v2], [m01, m12, m20]]
        elif m01 >= 0:
            F_new += [[v0, m01, v2], [m01, v1, v2]]
        elif m12 >= 0:
            F_new += [[v0, v1, m12], [v0, m12, v2]]
        elif m20 >= 0:
            F_new += [[v0, v1, m20], [m20, v1, v2]]
        else:
            F_new.append([v0, v1, v2])
    return S, np.array(F_new, dtype=np.int64)


# Laplacian editing


class LaplacianDeformer:
    """Deformation that preserves differential coordinates:

        argmin_V'  ||L (V' - V)||^2 + w^2 ||H V' - targets||^2 + wf^2 ||V'[fixed] - V[fixed]||^2

    H is a list of handle vertex ids or a sparse [m, n] matrix of barycentric rows, so the
    targets can be arbitrary surface points (e.g. a predicted lip point cloud). The normal
    equations depend only on the rest mesh and the handles, so they are factorised once.
    """

    def __init__(self, V, F, handles, handle_w=None, w=10.0, fixed=None, fixed_w=100.0):
        n = len(V)
        L = cotangent_laplacian(V, F)
        self.V = V
        if not sp.issparse(handles):
            handles = sp.coo_matrix(
                (np.ones(len(handles)), (np.arange(len(handles)), handles)), shape=(len(handles), n)
            )
        m = handles.shape[0]
        hw = np.ones(m) if handle_w is None else np.asarray(handle_w, float)
        blocks = [L, sp.diags(w * hw) @ handles]

        self.fixed = np.array([], int) if fixed is None else np.asarray(fixed)
        unused = np.setdiff1d(np.arange(n), F)  # vertices in no face are otherwise free
        self.fixed = np.union1d(self.fixed, unused).astype(int)
        if len(self.fixed):
            blocks.append(
                sp.coo_matrix(
                    (np.full(len(self.fixed), fixed_w), (np.arange(len(self.fixed)), self.fixed)),
                    shape=(len(self.fixed), n),
                )
            )

        self.A = sp.vstack(blocks).tocsc()
        self.lu = splu((self.A.T @ self.A + 1e-10 * sp.eye(n)).tocsc())
        self.delta = L @ V
        self.hw, self.w, self.fixed_w = hw, w, fixed_w

    def __call__(self, targets, V_fixed=None):
        """Solve for the deformed mesh. `V_fixed` (full-size) gives the positions of the fixed
        vertices, e.g. the jaw-posed rig; the rest mesh is used by default."""
        rhs = [self.delta, self.w * self.hw[:, None] * targets]
        if len(self.fixed):
            rhs.append(self.fixed_w * (self.V if V_fixed is None else V_fixed)[self.fixed])
        b = self.A.T @ np.concatenate(rhs)
        return np.stack([self.lu.solve(b[:, c]) for c in range(3)], 1)
