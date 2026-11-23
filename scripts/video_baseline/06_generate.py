"""Step 6: generation. Audio -> lip point-cloud deformation -> deformed mesh -> Gaussian
splatting -> video.

Without --audio the held-out tail of the video is used, so the result is compared with
the ground truth (comparison video and metrics). With --audio any new speech drives the
head at the reference pose.

Outputs: outputs/video_baseline/06_generation/{heldout | custom_<name>}/

Usage:
    python scripts/video_baseline/06_generate.py
    python scripts/video_baseline/06_generate.py --audio new.wav
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from skimage.metrics import structural_similarity
from tqdm import tqdm

from tinypt import visualization as viz
from tinypt.audio.features import context_windows, video_frame_features
from tinypt.config import BASELINE_OUTPUTS, CHECKPOINT_DIR, PROCESSED_DIR, ensure_dir, load_config
from tinypt.geometry import mesh as M
from tinypt.io_utils import cut_audio, load_json, save_gif, save_json, write_video_with_audio
from tinypt.metrics import psnr
from tinypt.rendering.splatting import render
from tinypt.video_baseline.face_mesh import LIP_IDX, openness
from tinypt.video_baseline.lip_deform_net import LipDeformNet
from tinypt.video_baseline.rigged_gaussians import load_rigged_gaussians

PANEL_SIZE = 256
MOUTH_CROP = 22  # half size of the mouth crop used for the mouth metrics (pixels)


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def predict_displacements(ckpt, wav, fps, sr, n_frames, start, max_frames, smooth_sigma, points):
    """Per-point lip displacements [n, N, 3] predicted from audio."""
    model = LipDeformNet(ckpt["audio_dim"], 6, ckpt["dim"], ckpt["L"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    feats = video_frame_features(wav, fps, ckpt["feature"], sr, n_frames)
    windows = context_windows((feats - ckpt["mu"]) / ckpt["sd"], ckpt["context"]).astype(np.float32)
    n = min(max_frames, len(windows) - start)
    windows = windows[start : start + n]
    pts = torch.tensor(np.concatenate([points["P0n"], points["normals"]], 1), dtype=torch.float32)
    knn = torch.tensor(points["knn"]).long()
    with torch.no_grad():
        P = (
            torch.cat(
                [model(torch.tensor(windows[i : i + 128]), pts, knn) for i in range(0, n, 128)]
            ).numpy()
            * ckpt["d_scale"]
        )
    return gaussian_filter1d(P, smooth_sigma, axis=0, mode="nearest")


def mouth_scores(img, gt, lm):
    """PSNR / SSIM of the full frame and of a crop around the mouth."""
    c = lm[LIP_IDX, :2].mean(0).astype(int)
    y0, x0 = max(0, c[1] - MOUTH_CROP), max(0, c[0] - MOUTH_CROP)
    crop = (slice(y0, c[1] + MOUTH_CROP), slice(x0, c[0] + MOUTH_CROP))
    return [
        psnr(img, gt),
        structural_similarity(img, gt, channel_axis=2, data_range=1.0),
        psnr(img[crop], gt[crop]),
        structural_similarity(img[crop], gt[crop], channel_axis=2, data_range=1.0, win_size=7),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audio", default=None)
    parser.add_argument("--max_frames", type=int, default=250)
    args = parser.parse_args()
    cfg = load_config()
    custom = args.audio is not None
    subdir = f"custom_{Path(args.audio).stem}" if custom else "heldout"
    out = ensure_dir(BASELINE_OUTPUTS / "06_generation" / subdir)
    meta = load_json(PROCESSED_DIR / "meta.json")
    frames = np.load(PROCESSED_DIR / "frames.npy")
    lm = np.load(PROCESSED_DIR / "landmarks.npy").astype(np.float32)
    mesh = np.load(PROCESSED_DIR / "mesh_data.npz")
    points = np.load(PROCESSED_DIR / "points.npz")
    V_can, F, region, region_w, region_tri = (
        mesh["V_can"],
        mesh["faces"],
        mesh["region_idx"],
        mesh["region_w"],
        mesh["region_tri"],
    )
    face_height, open_scale = float(mesh["face_h"]), float(mesh["open_scale"])
    T, fps, H = meta["T"], meta["fps"], frames.shape[1]
    n_train = int(T * (1 - cfg["data"]["val_ratio"]))
    K, P0 = int(points["K"]), points["P0"]

    gauss_ckpt = torch.load(CHECKPOINT_DIR / "gaussians.pt", weights_only=False)
    gaussians = load_rigged_gaussians(gauss_ckpt, V_can, F, region_tri)
    plate = gauss_ckpt["plate"]

    # Audio -> predicted point displacements.
    wav = Path(args.audio) if custom else PROCESSED_DIR / "audio.wav"
    start = 0 if custom else n_train
    P = predict_displacements(
        torch.load(CHECKPOINT_DIR / "audio2lip.pt", weights_only=False),
        wav,
        fps,
        cfg["audio"]["sr"],
        None if custom else T,
        start,
        args.max_frames,
        cfg["train"]["smooth_sigma"],
        points,
    )
    n = len(P)
    magnitude = np.linalg.norm(P, axis=-1)

    # Point displacements -> deformed mesh, head held at the reference pose.
    V_seq = np.repeat(V_can[None], n, 0).astype(np.float32)
    V_seq[:, region] += (P[:, :K] * region_w[None, :, None]).astype(np.float32)
    open_seq = openness(V_seq, face_height) / open_scale

    def render_mesh(V, o):
        with torch.no_grad():
            pos, cov, color, opacity = gaussians.params(
                torch.tensor(V, dtype=torch.float32), float(o)
            )
            img = render(pos, cov, color, opacity, H, H, plate, cfg["gaussians"]["tile"])
        return img.numpy(), pos.numpy(), color.numpy()

    reference_img = frames[int(mesh["ref_ids"][0])]
    talking_head, pipeline = [], []
    for k in tqdm(range(n), desc="render (static head pose)"):
        img, pos, color = render_mesh(V_seq[k], open_seq[k])
        img = to_uint8(img)
        talking_head.append(cv2.resize(img, (384, 384), interpolation=cv2.INTER_CUBIC))
        pipeline.append(
            viz.pipeline_frame(
                [
                    ("1 reference frame", reference_img),
                    (
                        "2 deformed 3D mesh",
                        viz.draw_mesh_panel(H, V_seq[k], F, region_tri, PANEL_SIZE),
                    ),
                    (
                        "3 lip point cloud",
                        viz.draw_points_panel(
                            H, P0 + P[k], P0, magnitude[k] / 6.0 * 4.0, P0.mean(0), PANEL_SIZE
                        ),
                    ),
                    (
                        "4 3D Gaussians (centres)",
                        viz.draw_gaussian_panel(H, pos, color, PANEL_SIZE),
                    ),
                    ("5 neural render", img),
                ],
                PANEL_SIZE,
            )
        )

    segment_wav = out / "_seg.wav"
    cut_audio(wav, start / fps, n / fps, segment_wav)
    for name, video in [("talking_head", talking_head), ("pipeline_panels", pipeline)]:
        write_video_with_audio(video, segment_wav, out / f"{name}.mp4", fps)
    small = [cv2.resize(p, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for p in pipeline]
    save_gif(small, out / "pipeline_panels.gif", fps=12, step=2)
    storyboard = np.concatenate([pipeline[i] for i in np.linspace(0, n - 1, 6).astype(int)], axis=0)
    cv2.imwrite(str(out / "storyboard.png"), storyboard[..., ::-1])
    np.savez_compressed(out / "predicted_lip_points.npz", points=P0[None] + P, fps=fps)

    if not custom:
        evaluate_heldout(
            out,
            mesh,
            frames,
            lm,
            V_can,
            V_seq,
            open_seq,
            n_train,
            n,
            fps,
            open_scale,
            segment_wav,
            render_mesh,
        )
    print("done ->", out)


def evaluate_heldout(
    out,
    mesh,
    frames,
    lm,
    V_can,
    V_seq,
    open_seq,
    n_train,
    n,
    fps,
    open_scale,
    segment_wav,
    render_mesh,
):
    """Comparison video at the ground-truth head pose, and image metrics against an upper
    bound (tracked mesh) and a closed-mouth baseline."""
    ids = np.arange(n_train, n_train + n)
    pose_s, pose_R, pose_t = mesh["pose_s"], mesh["pose_R"], mesh["pose_t"]
    ours, comparison = [], []
    for k, t in enumerate(tqdm(ids, desc="render (ground-truth head pose)")):
        V_posed = M.invert_sim(V_seq[k], pose_s[t], pose_R[t], pose_t[t]).astype(np.float32)
        img, _, _ = render_mesh(V_posed, open_seq[k])
        ours.append(img)
        comparison.append(
            viz.pipeline_frame(
                [("ground truth", frames[t]), ("ours: audio-driven lips", to_uint8(img))], 320
            )
        )
    write_video_with_audio(comparison, segment_wav, out / "comparison_gt_vs_ours.mp4", fps)

    scores = {k: [] for k in ["gt_mesh", "ours_audio", "neutral_lips"]}
    for k in np.linspace(0, n - 1, min(40, n)).astype(int):
        t = ids[k]
        gt = frames[t] / 255.0
        img_gt_mesh, _, _ = render_mesh(lm[t], mesh["open_series"][t] / open_scale)
        V_neutral = M.invert_sim(V_can, pose_s[t], pose_R[t], pose_t[t]).astype(np.float32)
        img_neutral, _, _ = render_mesh(V_neutral, 0.0)
        scores["gt_mesh"].append(mouth_scores(img_gt_mesh, gt, lm[t]))
        scores["ours_audio"].append(mouth_scores(ours[k], gt, lm[t]))
        scores["neutral_lips"].append(mouth_scores(img_neutral, gt, lm[t]))
    names = ["psnr_full", "ssim_full", "psnr_mouth", "ssim_mouth"]
    metrics = {
        key: {name: float(np.mean([r[i] for r in rows])) for i, name in enumerate(names)}
        for key, rows in scores.items()
    }
    metrics["note"] = (
        "gt_mesh = upper bound (tracked GT mesh); ours_audio = lips predicted from "
        "audio only, GT head pose; neutral_lips = closed-mouth baseline (no audio)"
    )
    save_json(metrics, out / "metrics.json")
    print(json.dumps(metrics, indent=1))
    viz.plot_curves(
        [
            ("ground truth", mesh["open_series"][ids] / open_scale, dict(color="k")),
            ("predicted from audio", open_seq, dict(color="tab:red")),
        ],
        out / "mouth_opening_curve.png",
        "Mouth opening on held-out audio",
        ylabel="opening / p95",
        fps=fps,
    )


if __name__ == "__main__":
    main()
