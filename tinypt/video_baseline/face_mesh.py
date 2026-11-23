"""MediaPipe FaceMesh topology (468 vertices) and lip landmarks."""

from collections import defaultdict

import numpy as np

# Outer and inner lip contour, hard-coded so that only the extraction step needs mediapipe.
# fmt: off
LIP_IDX = np.array(sorted([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]))
# fmt: on
UPPER_INNER_LIP, LOWER_INNER_LIP = 13, 14  # centre of the inner lips


def faces_from_mediapipe():
    """Triangles of the 468-vertex mesh, i.e. the 3-cliques of FACEMESH_TESSELATION."""
    import mediapipe as mp

    edges = mp.solutions.face_mesh.FACEMESH_TESSELATION
    neighbours = defaultdict(set)
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    triangles = set()
    for a, b in edges:
        for c in neighbours[a] & neighbours[b]:
            triangles.add(tuple(sorted((a, b, c))))
    return np.array(sorted(triangles), dtype=np.int64)


def openness(V, face_height):
    """Inner-lip gap divided by the face height (image coordinates, y down). V [..., 468, 3]."""
    return (V[..., LOWER_INNER_LIP, 1] - V[..., UPPER_INNER_LIP, 1]) / face_height
