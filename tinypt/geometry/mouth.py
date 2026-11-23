"""Mouth structure on a FLAME-topology mesh (y up, z towards the camera, metres).

- Inner-lip ring: the boundary loop of the mouth hole, found topologically.
- Upper / lower inner lip: split by the line through the mouth corners.
- Proxy geometry: a mouth cavity and upper / lower teeth strips, skinned to the FLAME
  skeleton (upper teeth follow the head, lower teeth the jaw) as in GaussianAvatars, so
  the open mouth has surfaces for Gaussians to live on.
"""

import numpy as np

from . import mesh as M

NECK, JAW = 1, 2  # FLAME joint indices


def inner_lip_loop(V, F, lips_idx):
    """Boundary loop of the mesh closest to the centroid of the lip region."""
    centre = V[lips_idx].mean(0)
    loops = [loop for loop in M.boundary_loops(F) if len(loop) >= 8]
    dist = [np.linalg.norm(V[loop].mean(0) - centre) for loop in loops]
    best = int(np.argmin(dist))
    if dist[best] > 0.5 * np.ptp(V[lips_idx], 0).max():
        raise RuntimeError(
            "no mouth-hole boundary loop near the lips; is this a FLAME-topology mesh?"
        )
    return loops[best]


def split_upper_lower(V, loop):
    """Upper / lower inner-lip vertices, the two mouth corners and the mid-line vertex pair
    used to measure mouth opening."""
    P = V[loop]
    i_left, i_right = int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0]))
    a, b = P[i_left], P[i_right]
    t = (P[:, 0] - a[0]) / (b[0] - a[0] + 1e-12)
    line_y = a[1] + t * (b[1] - a[1])
    not_corner = (np.arange(len(loop)) != i_left) & (np.arange(len(loop)) != i_right)
    up = loop[(P[:, 1] >= line_y) & not_corner]
    lo = loop[(P[:, 1] < line_y) & not_corner]
    cx = 0.5 * (a[0] + b[0])
    up_mid = int(up[np.argmin(np.abs(V[up, 0] - cx))])
    lo_mid = int(lo[np.argmin(np.abs(V[lo, 0] - cx))])
    return dict(
        up=up, lo=lo, corners=np.array([loop[i_left], loop[i_right]]), up_mid=up_mid, lo_mid=lo_mid
    )


def openness(V, up_mid, lo_mid, scale):
    """Vertical inner-lip gap divided by `scale`. V has shape [..., n_vertices, 3]."""
    return (V[..., up_mid, 1] - V[..., lo_mid, 1]) / scale


def _arch_strip(cx, y_bottom, y_top, cz, rx, rz, depth, n=16):
    """Curved vertical strip shaped like a dental arch: 2 rows x n columns, facing +z."""
    theta = np.linspace(0.08 * np.pi, 0.92 * np.pi, n)
    x = cx + rx * np.cos(theta)
    z = cz - depth - rz * (1 - np.sin(theta))
    P = np.concatenate(
        [np.stack([x, np.full(n, y_top), z], 1), np.stack([x, np.full(n, y_bottom), z], 1)]
    )
    F = []
    for i in range(n - 1):
        a, b, c, d = i, i + 1, n + i, n + i + 1
        F += [[a, c, b], [b, c, d]]
    return P, np.array(F)


def build_proxies(V, loop, split, n_base, cfg=None):
    """Mouth cavity (a bag behind the inner-lip ring) plus upper / lower teeth strips.

    Returns rest positions [m, 3], LBS weights [m, 5], faces indexing the concatenation
    [base vertices; proxy vertices], and a part label per proxy vertex.
    """
    cfg = cfg or {}
    ring = V[loop]
    centre = ring.mean(0)
    width = np.ptp(ring[:, 0])
    depth = cfg.get("cavity_depth", 0.6) * width
    lower = np.isin(loop, split["lo"]).astype(float)  # 1: follows the jaw, 0: follows the head

    P, weights, F, part = [], [], [], []
    prev = loop  # the first ring is the inner lip itself
    next_id = n_base
    for lam in (0.3, 0.55, 0.8):
        ring_k = (1 - lam) * ring + lam * centre + np.array([0, 0, -lam * depth])
        ids = np.arange(next_id, next_id + len(loop))
        next_id += len(loop)
        P.append(ring_k)
        zeros = np.zeros(len(loop))
        weights.append(np.stack([zeros, 1 - lower, lower, zeros, zeros], 1))
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            F += [[prev[i], ids[i], prev[j]], [prev[j], ids[i], ids[j]]]
        prev = ids
    P.append(np.array([[centre[0], centre[1], centre[2] - depth]]))
    weights.append(np.array([[0, 0.5, 0.5, 0, 0]]))
    for i in range(len(loop)):
        F.append([prev[i], next_id, prev[(i + 1) % len(loop)]])
    next_id += 1
    part += ["cavity"] * (next_id - n_base)

    rx, rz, d0 = 0.42 * width, 0.35 * width, cfg.get("teeth_depth", 0.12) * width
    teeth = [
        ("upper_teeth", centre[1] - 0.01 * width, centre[1] + 0.16 * width, NECK),
        ("lower_teeth", centre[1] - 0.16 * width, centre[1] - 0.02 * width, JAW),
    ]
    for name, y0, y1, joint in teeth:
        Pt, Ft = _arch_strip(centre[0], y0, y1, centre[2], rx, rz, d0)
        P.append(Pt)
        w = np.zeros((len(Pt), 5))
        w[:, joint] = 1
        weights.append(w)
        F += (Ft + next_id).tolist()
        next_id += len(Pt)
        part += [name] * len(Pt)
    P, weights, F = np.concatenate(P), np.concatenate(weights), np.array(F, dtype=np.int64)

    # Cavity faces point into the bag (towards the opening), teeth faces towards +z.
    V_all = np.concatenate([V, P])
    fn, _ = M.face_normals_areas(V_all, F)
    face_centres = V_all[F].mean(1)
    is_cavity = np.array([part[i - n_base] == "cavity" if i >= n_base else True for i in F.max(1)])
    target = np.where(
        is_cavity[:, None],
        (centre + np.array([0, 0, -0.3 * depth])) - face_centres,
        np.array([0, 0, 1.0]),
    )
    flip = (fn * target).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return P, weights, F, np.array(part)
