"""Step 1: video -> cropped frames, audio and 3D face-mesh landmarks (MediaPipe FaceMesh).

Outputs: data/processed/{frames.npy, landmarks.npy, faces.npy, audio.wav, meta.json},
outputs/video_baseline/01_data/landmarks_overlay.png
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from tinypt import visualization as viz
from tinypt.config import BASELINE_OUTPUTS, PROCESSED_DIR, ROOT, ensure_dir, load_config
from tinypt.io_utils import ffmpeg_exe, save_json
from tinypt.video_baseline.face_mesh import faces_from_mediapipe


def iter_frames(path, target_fps, max_seconds):
    """Yield (index, BGR frame) resampled to target_fps, for at most max_seconds."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_out = int(min(n_src / src_fps, max_seconds) * target_fps)
    wanted = [min(n_src - 1, int(round(k * src_fps / target_fps))) for k in range(n_out)]
    k = i = 0
    while k < n_out:
        ok, frame = cap.read()
        if not ok:
            break
        while k < n_out and wanted[k] == i:
            yield k, frame
            k += 1
        i += 1
    cap.release()


def track_landmarks(video, fps, max_seconds):
    """FaceMesh landmarks [T, 468, 3] in full-resolution pixels; gaps are interpolated."""
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarks = []
    for _, frame in tqdm(iter_frames(video, fps, max_seconds), desc="FaceMesh"):
        H, W = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            landmarks.append(
                np.array(
                    [[p.x * W, p.y * H, p.z * W] for p in result.multi_face_landmarks[0].landmark]
                )
            )
        else:
            landmarks.append(np.full((468, 3), np.nan))
    landmarks = np.stack(landmarks)
    T = len(landmarks)
    missing = np.isnan(landmarks[:, 0, 0])
    print(f"{T} frames, {missing.sum()} without a detected face")
    assert missing.sum() < 0.5 * T, (
        "face not detected in most frames; use a frontal, well-lit video"
    )
    if missing.any():
        good = np.where(~missing)[0]
        flat = landmarks.reshape(T, -1)
        for j in range(flat.shape[1]):
            flat[missing, j] = np.interp(np.where(missing)[0], good, flat[good, j])
        landmarks = flat.reshape(T, 468, 3)
    return landmarks


def crop_transform(landmarks, crop_scale, size):
    """Fixed square crop around the face over the whole video."""
    x, y = landmarks[..., 0], landmarks[..., 1]
    bbox = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], 1)
    centre = np.stack([(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2], 1)
    cx, cy = np.median(centre, 0)
    side = (
        np.percentile(np.maximum(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]), 95) * crop_scale
    )
    x0, y0 = cx - side / 2, cy - side / 2
    scale = size / side
    return x0, y0, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", default=None)
    args = parser.parse_args()
    cfg = load_config()
    data_cfg = cfg["data"]
    video = Path(args.video or ROOT / data_cfg["video"])
    out = ensure_dir(BASELINE_OUTPUTS / "01_data")
    ensure_dir(PROCESSED_DIR)
    assert video.exists(), f"put your video at {video}"

    # Pass 1: landmarks on full-resolution frames (nothing else is kept in memory).
    landmarks = track_landmarks(video, data_cfg["fps"], data_cfg["max_seconds"])
    T = len(landmarks)
    S = data_cfg["img_size"]
    x0, y0, scale = crop_transform(landmarks, data_cfg["crop_scale"], S)
    affine = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])

    # Pass 2: crop and resize the frames.
    frames = []
    for k, frame in tqdm(
        iter_frames(video, data_cfg["fps"], data_cfg["max_seconds"]), desc="crop", total=T
    ):
        if k >= T:
            break
        crop = cv2.warpAffine(
            frame, affine, (S, S), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
        )
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    frames = np.stack(frames)
    T = min(T, len(frames))
    frames, landmarks = frames[:T], landmarks[:T]
    lm = landmarks.copy()
    lm[..., 0] = (landmarks[..., 0] - x0) * scale
    lm[..., 1] = (landmarks[..., 1] - y0) * scale
    lm[..., 2] = landmarks[..., 2] * scale
    lm = gaussian_filter1d(lm, sigma=1.0, axis=0)  # light temporal de-jitter

    np.save(PROCESSED_DIR / "frames.npy", frames)
    np.save(PROCESSED_DIR / "landmarks.npy", lm.astype(np.float32))
    np.save(PROCESSED_DIR / "faces.npy", faces_from_mediapipe())
    save_json(
        dict(fps=data_cfg["fps"], T=T, img_size=S, source=str(video)), PROCESSED_DIR / "meta.json"
    )
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(cfg["audio"]["sr"]),
            "-t",
            str(T / data_cfg["fps"]),
            str(PROCESSED_DIR / "audio.wav"),
        ],
        check=True,
    )

    viz.landmark_overlay(
        frames, lm, np.linspace(0, T - 1, 6).astype(int), out / "landmarks_overlay.png"
    )
    print("saved data/processed/*, figures in", out)


if __name__ == "__main__":
    main()
