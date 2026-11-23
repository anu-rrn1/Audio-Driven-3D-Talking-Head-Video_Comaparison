"""MediaPipe FaceMesh landmarks and a self-calibrated MediaPipe -> FLAME surface embedding.

The embedding assigns every MediaPipe landmark a fixed point on the FLAME surface
(triangle + barycentric coordinates), so 2D detections can be compared with projected mesh
points. Instead of using a third-party embedding, it is calibrated here: synthetic FLAME
heads (random identity, expression, pose and procedural albedo) are rendered, MediaPipe is
run on the renders, and every detection is back-projected onto the mesh through the
rasterizer's triangle-id buffer. A landmark is a reliable correspondence only if it lands
on the same surface point in every view; contour (jawline) landmarks slide over the surface
as the head turns, show a large spread, and are rejected.
"""

import numpy as np
import torch

from ..rendering import rasterizer as R
from .model import N_HEAD_VERTICES

N_MEDIAPIPE_LANDMARKS = 478  # 468 face mesh points + 10 iris points
EYE_OUTER = (33, 263)  # outer eye corners, used for the inter-ocular distance

_face_mesh = None


def detect(img):
    """Landmarks of one face in a uint8 RGB image: [478, 2] pixel coordinates (pixel i
    spans [i, i + 1]), or None if no face is found."""
    global _face_mesh
    import mediapipe as mp

    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.3,
        )
    H, W = img.shape[:2]
    result = _face_mesh.process(np.ascontiguousarray(img[..., :3]))
    if not result.multi_face_landmarks:
        return None
    return np.array(
        [[p.x * W, p.y * H] for p in result.multi_face_landmarks[0].landmark], np.float64
    )


def mediapipe_lip_ids():
    import mediapipe as mp

    return np.unique([i for edge in mp.solutions.face_mesh.FACEMESH_LIPS for i in edge])


def _barycentric(p, tri_uv):
    a, b, c = tri_uv
    T = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
    l1, l2 = np.linalg.solve(T, p - a)
    w = np.clip(np.array([1 - l1 - l2, l1, l2]), 0, 1)
    return w / w.sum()


def calibrate(
    flame,
    n_ids=4,
    yaws=(-35, -20, -8, 0, 8, 20, 35),
    pitches=(-12, 0, 12),
    size=512,
    seed=100,
    log=print,
):
    """Calibrate the embedding from synthetic renders (see the module docstring).

    Returns dict(face_idx, bary, spread_mm, valid_frac, n_views): per landmark the medoid
    surface point, the median distance of all observations to it, and the detection rate.
    """
    rng = np.random.default_rng(seed)
    F = flame.faces
    V_tmp = flame.v_template.numpy().astype(np.float64)
    head_face = (F < N_HEAD_VERTICES).all(1)  # not on the eyeballs
    observations = [[] for _ in range(N_MEDIAPIPE_LANDMARKS)]  # (face, bary) per landmark
    n_views = 0
    for i in range(n_ids):
        beta = torch.zeros(1, flame.n_shape)
        beta[0, :20] = torch.tensor(rng.normal(0, 1, 20), dtype=torch.float32)
        albedo = R.procedural_albedo(V_tmp, F, flame.region("lips"), rng)
        for yaw in yaws:
            for pitch in pitches:
                expr = torch.zeros(1, flame.n_expr)
                pose = torch.zeros(1, 15)
                if rng.random() < 0.5:  # mild expressions and open mouths in half the views
                    expr[0, :10] = torch.tensor(rng.normal(0, 0.6, 10), dtype=torch.float32)
                    pose[0, 6] = float(rng.uniform(0, 0.2))
                with torch.no_grad():
                    V = flame(beta, expr, pose)[0][0].numpy().astype(np.float64)
                rvec = np.radians(
                    [pitch + rng.normal(0, 2), yaw + rng.normal(0, 2), rng.normal(0, 3)]
                )
                t = [rng.normal(0, 0.01), rng.normal(0.02, 0.01), rng.uniform(0.6, 0.9)]
                f = rng.uniform(1.6, 2.6) * size
                img, fid, _, _ = R.render(
                    V,
                    F,
                    rvec,
                    t,
                    f,
                    size,
                    size,
                    albedo,
                    bg=R.background(size, size, rng),
                    light=(rng.normal(0, 0.4), rng.uniform(0.2, 0.7), 1.0),
                )
                lm = detect(img)
                n_views += 1
                if lm is None:
                    continue
                uv, _ = R.project(R.to_camera(V, rvec, t), f, size / 2, size / 2)
                front = R.face_area(uv, F) < 0  # back faces (mouth interior) cannot match
                px = np.clip(np.floor(lm).astype(int), 0, size - 1)
                for k in range(N_MEDIAPIPE_LANDMARKS):
                    fi = fid[px[k, 1], px[k, 0]]
                    if fi >= 0 and head_face[fi] and front[fi]:
                        observations[k].append((fi, _barycentric(lm[k], uv[F[fi]])))
        log(f"  landmark calibration: identity {i + 1}/{n_ids} done ({n_views} renders)")

    n = N_MEDIAPIPE_LANDMARKS
    face_idx = np.zeros(n, np.int64)
    bary = np.zeros((n, 3), np.float32)
    spread = np.full(n, np.inf)
    valid = np.zeros(n)
    for k in range(n):
        if not observations[k]:
            continue
        P = np.array([(V_tmp[F[fi]] * b[:, None]).sum(0) for fi, b in observations[k]])
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        medoid = int(np.argmin(np.median(D, 1)))
        face_idx[k], bary[k] = observations[k][medoid]
        spread[k] = 1000 * np.median(D[medoid])
        valid[k] = len(observations[k]) / n_views
    return dict(face_idx=face_idx, bary=bary, spread_mm=spread, valid_frac=valid, n_views=n_views)


def select(cal, max_spread_mm=2.0, min_valid=0.8, drop_iris=True):
    """Ids of landmarks that are stable on the surface and detected in most views."""
    ok = (cal["spread_mm"] < max_spread_mm) & (cal["valid_frac"] >= min_valid)
    if drop_iris:
        ok[468:] = False  # iris points depend on gaze
    return np.where(ok)[0]
