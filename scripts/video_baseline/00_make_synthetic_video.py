"""Step 0 (optional): synthetic stand-in for a real video, for smoke-testing the baseline.

Builds a cartoon face (frames, 468 3D landmarks with a Delaunay triangulation, audio) whose
mouth opening follows the loudness envelope of the audio. Writes the same files as step 1
to data/processed/; a real run of step 1 overwrites them.
"""

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay

from tinypt.config import PROCESSED_DIR, ensure_dir
from tinypt.io_utils import save_json
from tinypt.video_baseline.face_mesh import LIP_IDX

FPS, N_FRAMES, SIZE, SR = 25, 400, 128, 16000
CX, CY, RX, RY, MOUTH_Y = 64, 62, 30, 40, 88  # face ellipse and mouth height (pixels)

LIP_GROUPS = dict(
    up_out=[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    lo_out=[146, 91, 181, 84, 17, 314, 405, 321, 375],
    up_in=[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    lo_in=[95, 88, 178, 87, 14, 317, 402, 318, 324],
)


def synthesize_audio(rng):
    """Syllable-like loudness envelope and a harmonic waveform. Returns (wave, envelope at
    the video frame rate)."""
    t = np.arange(int(N_FRAMES / FPS * SR)) / SR
    envelope = np.zeros_like(t)
    for c in np.cumsum(rng.uniform(0.18, 0.5, 60)):
        envelope += rng.uniform(0.4, 1.0) * np.exp(-0.5 * ((t - c) / 0.07) ** 2)
    envelope = np.clip(envelope, 0, 1)
    vibrato = 1 + 0.05 * np.sin(2 * np.pi * 0.7 * t)
    wave = sum(np.sin(2 * np.pi * k * 120 * t * vibrato) / k for k in range(1, 12))
    audio = (0.6 * envelope * wave / np.abs(wave).max() * 32767).astype(np.int16)
    per_frame = gaussian_filter1d(np.interp(np.arange(N_FRAMES) / FPS, t, envelope), 1.0)
    return audio, per_frame


def lip_contours(gap):
    """2D lip contours for a mouth opening of `gap` pixels."""
    contours = {}
    xs = np.linspace(-1, 1, 11)
    for name, half_width, offset, upper in [
        ("up_out", 11, 3.0, True),
        ("up_in", 8, 0.5, True),
        ("lo_out", 11, 3.0, False),
        ("lo_in", 8, 0.5, False),
    ]:
        x = xs if upper else xs[1:-1]
        h = np.sqrt(1 - x**2)
        y = MOUTH_Y - offset - 0.3 * gap * h if upper else MOUTH_Y + offset + 0.7 * gap * h
        contours[name] = np.stack([CX + half_width * x, y], 1)
    return contours


def neutral_layout(rng):
    """Lip landmarks on their contours; the other landmarks spread over the face ellipse."""
    lip_set = set(sum(LIP_GROUPS.values(), []))
    assert lip_set == set(LIP_IDX.tolist())
    others = [i for i in range(468) if i not in lip_set]
    pts = []
    while len(pts) < len(others):
        p = rng.uniform([CX - RX, CY - RY], [CX + RX, CY + RY])
        in_face = ((p[0] - CX) / RX) ** 2 + ((p[1] - CY) / RY) ** 2 < 1
        outside_mouth = np.hypot((p[0] - 64) / 15, (p[1] - MOUTH_Y) / 9) > 1.0
        if in_face and outside_mouth and all(np.hypot(*(p - q)) > 2.0 for q in pts):
            pts.append(p)
    base = np.zeros((468, 2))
    base[others] = np.array(pts)
    for k, v in lip_contours(0).items():
        base[LIP_GROUPS[k]] = v
    return base, np.array(others)


def frame_landmarks(base, others, gap):
    """3D landmarks for one frame: lips open by `gap`, the chin follows, depth from a dome."""
    P = base.copy()
    for k, v in lip_contours(gap).items():
        P[LIP_GROUPS[k]] = v
    x, y = base[others, 0], base[others, 1]
    below_mouth = y > MOUTH_Y + 3
    dy = np.where(
        below_mouth,
        0.7 * gap * np.exp(-(((y - MOUTH_Y) / 14) ** 2)) * np.exp(-(((x - CX) / 18) ** 2)),
        0,
    )
    P[others, 1] += dy
    r2 = ((P[:, 0] - CX) / RX) ** 2 + ((P[:, 1] - CY) / RY) ** 2
    z = -8 * np.clip(1 - r2, 0, 1)
    return np.concatenate([P, z[:, None]], 1)


def draw_face(P):
    """Cartoon face (BGR) with the mouth drawn from the landmarks."""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:] = (70, 100, 130)
    cv2.ellipse(img, (CX, CY - 6), (RX + 5, RY + 8), 0, 180, 360, (40, 30, 30), -1)
    cv2.ellipse(img, (CX, CY), (RX, RY), 0, 0, 360, (150, 180, 225), -1)
    for side in (-1, 1):
        cv2.circle(img, (CX + side * 11, CY - 14), 5, (255, 255, 255), -1)
        cv2.circle(img, (CX + side * 11, CY - 14), 2, (30, 30, 30), -1)
        cv2.line(img, (CX + side * 6, CY - 24), (CX + side * 17, CY - 24), (40, 40, 60), 2)
    cv2.line(img, (CX, CY - 6), (CX - 3, CY + 12), (120, 150, 200), 1)
    inner = np.concatenate([P[LIP_GROUPS["up_in"], :2], P[LIP_GROUPS["lo_in"], :2][::-1]])
    outer = np.concatenate([P[LIP_GROUPS["up_out"], :2], P[LIP_GROUPS["lo_out"], :2][::-1]])
    cv2.polylines(img, [outer.astype(np.int32)], True, (80, 80, 200), 2)
    cv2.fillPoly(img, [inner.astype(np.int32)], (30, 20, 100))
    return img


def main():
    ensure_dir(PROCESSED_DIR)
    rng = np.random.default_rng(0)
    audio, loudness = synthesize_audio(rng)
    wavfile.write(PROCESSED_DIR / "audio.wav", SR, audio)
    gap = 9.0 * loudness  # mouth opening in pixels

    base, others = neutral_layout(rng)
    faces = Delaunay(base).simplices.astype(np.int64)
    landmarks = np.stack([frame_landmarks(base, others, g) for g in gap])
    landmarks += rng.normal(0, 0.03, (N_FRAMES, 468, 3))
    frames = np.stack([draw_face(p) for p in landmarks])[..., ::-1].copy()  # BGR -> RGB

    np.save(PROCESSED_DIR / "frames.npy", frames)
    np.save(PROCESSED_DIR / "landmarks.npy", landmarks.astype(np.float32))
    np.save(PROCESSED_DIR / "faces.npy", faces)
    save_json(
        dict(fps=FPS, T=N_FRAMES, img_size=SIZE, source="synthetic"), PROCESSED_DIR / "meta.json"
    )
    print("synthetic demo data written to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
