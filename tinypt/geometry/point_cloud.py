"""Point-cloud processing: sampling, neighbourhoods, normals, distances and mesh binding."""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def farthest_point_sampling(P, m, start=0):
    """Indices of m points chosen farthest-first.

    `start` is one index or a list of indices that are always kept; the remaining points
    are added greedily.
    """
    init = np.atleast_1d(start).astype(np.int64)
    idx = np.zeros(m, dtype=np.int64)
    idx[: len(init)] = init
    dist = np.full(len(P), np.inf)
    for i in init[:-1]:
        dist = np.minimum(dist, ((P - P[i]) ** 2).sum(1))
    for i in range(len(init), m):
        dist = np.minimum(dist, ((P - P[idx[i - 1]]) ** 2).sum(1))
        idx[i] = dist.argmax()
    return idx


def knn_indices(P, k):
    """Indices of the k nearest neighbours of every point, excluding the point itself."""
    _, idx = cKDTree(P).query(P, k=k + 1)
    return idx[:, 1:]


def estimate_normals(P, k=10, towards=np.array([0.0, 0.0, -1.0])):
    """PCA normals of k-NN neighbourhoods, oriented towards `towards` (the camera)."""
    idx = knn_indices(P, k)
    N = np.zeros_like(P)
    for i in range(len(P)):
        Q = P[idx[i]] - P[idx[i]].mean(0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        n = Vt[-1]
        N[i] = n if n @ towards >= 0 else -n
    return N


def chamfer(A, B):
    """Symmetric Chamfer distance (mean nearest-neighbour L2 distance, both directions)."""
    da, _ = cKDTree(B).query(A)
    db, _ = cKDTree(A).query(B)
    return float(0.5 * (da.mean() + db.mean()))


def normalize_cloud(P):
    """Centre and scale to unit RMS radius. Returns (normalised points, centre, scale)."""
    c = P.mean(0)
    s = np.sqrt(((P - c) ** 2).sum(1).mean())
    return (P - c) / s, c, s


def binding_matrix(tri, bary, F, n_vertices, S=None):
    """Sparse operator B with points = B @ V, each point a barycentric combination of a triangle.

    If F belongs to a refined mesh V_ref = S @ V, pass S: the result then maps the original
    vertices V directly to the points.
    """
    rows = np.repeat(np.arange(len(tri)), 3)
    n_cols = S.shape[0] if S is not None else n_vertices
    B = sp.coo_matrix((bary.ravel(), (rows, F[tri].ravel())), shape=(len(tri), n_cols)).tocsr()
    return (B @ S).tocsr() if S is not None else B
