"""Step 7: generation. Audio -> 3D lip motion -> rigged Gaussian avatar -> talking-head video.

    16 kHz audio -> wav2vec 2.0 + Audio2Motion (step 4b, v1, neutral style) + articulation
        gain (step 4c) -> PCA coefficients at 30 fps
    -> per-vertex displacements (VOCASET is in the FLAME frame, so they apply directly to
       the identity mesh)
    -> jaw angle per frame, found by matching the chin to a FLAME jaw sweep; it drives the
       jaw-rigged mouth cavity and teeth
    -> Gaussian avatar (step 6) rendered every frame -> mp4 with the audio

Subtitles are transcribed from the audio itself (tinypt/subtitles.py), burned in with the
current word highlighted, and written as .srt files.

The default audio is three unseen VOCASET test-speaker sentences joined with short pauses,
so the video can show the avatar driven by the recorded 3D motion next to the avatar
driven by audio alone, and the lip error can be measured. Each sentence goes through the
model separately, as it was trained on single sentences.

Outputs: outputs/flame_pipeline/07_video/

Usage:
    python scripts/flame_pipeline/07_generate_video.py [--seq 170809_00138:2,7,1]
        [--wav my_clip.wav] [--yaw 12] [--res 256]
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile

from tinypt import subtitles
from tinypt.audio.audio2motion import MOTION_FPS, apply_gain, load_audio2motion, predict_from_audio
from tinypt.audio.vocaset import AUDIO_SR, MotionBasis, iter_sequences
from tinypt.config import (
    FLAME_OUTPUTS,
    IDENTITY_DIR,
    ROOT,
    ensure_dir,
    load_config,
    use_all_cpu_threads,
)
from tinypt.flame import load_flame
from tinypt.flame.model import N_FLAME_VERTICES
from tinypt.io_utils import load_json, save_json, write_video_with_audio
from tinypt.rendering.gaussian_avatar import GaussianAvatar

SAMPLES_PER_FRAME = AUDIO_SR // MOTION_FPS
JAW_SWEEP = np.linspace(-0.05, 0.4, 91)  # jaw angles (rad) searched per frame
JAW_JOINT = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--seq", default="170809_00138:2,7,1", help="VOCASET speaker:sentence[,sentence...]"
    )
    parser.add_argument("--gap", type=float, default=0.3, help="pause between sentences (s)")
    parser.add_argument("--no_subtitles", action="store_true")
    parser.add_argument("--wav", default=None, help="any speech clip instead (no ground truth)")
    parser.add_argument(
        "--ckpt", default="outputs/flame_pipeline/04b_audio2motion_v1_better_lip_error/audio2motion.pt"
    )
    parser.add_argument(
        "--gain",
        default="auto",
        help="articulation gain; 'auto' reads step 4c's gain.json, 1 = off",
    )
    parser.add_argument("--data", default="data/vocaset_motion/vocaset_motion.npz")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--yaw", type=float, default=12.0)
    parser.add_argument("--max_seconds", type=float, default=12.0, help="limit for --wav")
    parser.add_argument("--out", default="07_video")
    return parser.parse_args()


class Logger:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        print(f"[{time.time() - self.t0:6.0f}s] {msg}", flush=True)


def load_segments(args, data):
    """Audio segments, each with its ground-truth coefficients (30 fps) or None."""
    if args.wav:
        import librosa

        x, _ = librosa.load(args.wav, sr=AUDIO_SR, mono=True)
        return [(x[: int(args.max_seconds * AUDIO_SR)], None)], Path(args.wav).stem
    speaker, sentences = args.seq.split(":")
    segments, seq = [], None
    for sentence in sentences.split(","):
        seq = next(iter_sequences(data, speaker=speaker, sentences=[int(sentence)]))
        segments.append((seq["audio"], seq["coeffs"]))
    return segments, f"{seq['spk']} sentences {sentences} ({seq['split']} speaker)"


def resolve_gain(arg):
    if arg != "auto":
        return float(arg)
    gain_file = FLAME_OUTPUTS / "04c_articulation" / "gain.json"
    return float(load_json(gain_file)["gain"]) if gain_file.exists() else 1.0


def predict_motion(model, feature, segments, c_mean, gain, gap_frames):
    """Coefficients for the joined segments; pauses blend from one pose to the next.

    Returns audio, predicted coefficients, ground truth (or None), a speech mask per frame
    and the (start, end) time of every segment.
    """
    audio, preds, gts, speech, spans = [], [], [], [], []
    t = 0.0
    for k, (xa, gt) in enumerate(segments):
        n = int(round(len(xa) / SAMPLES_PER_FRAME))
        if gt is not None:
            n = min(n, len(gt))
            gt = gt[:n]
        xa = np.pad(xa, (0, max(0, n * SAMPLES_PER_FRAME - len(xa))))[: n * SAMPLES_PER_FRAME]
        pred = apply_gain(predict_from_audio(model, [xa], [n], feature)[0], c_mean, gain)
        if k > 0 and gap_frames > 0:
            w = np.linspace(0, 1, gap_frames + 2)[1:-1, None]
            preds.append((1 - w) * preds[-1][-1:] + w * pred[:1])
            if gt is not None:
                gts.append((1 - w) * gts[-1][-1:] + w * gt[:1])
            audio.append(np.zeros(gap_frames * SAMPLES_PER_FRAME, np.float32))
            speech.append(np.zeros(gap_frames, bool))
            t += gap_frames / MOTION_FPS
        preds.append(pred)
        audio.append(xa)
        speech.append(np.ones(n, bool))
        if gt is not None:
            gts.append(gt)
        spans.append((t, t + n / MOTION_FPS))
        t += n / MOTION_FPS
    return (
        np.concatenate(audio),
        np.concatenate(preds),
        np.concatenate(gts) if gts else None,
        np.concatenate(speech),
        spans,
    )


def lip_error_metrics(motion, data, D_pred, D_gt, speech, pred_c, c_mean, gain):
    """LVE on the speech frames for the prediction, the prediction without gain, and the
    dataset mean pose."""
    lips = motion.lips

    def lve(D):
        return np.linalg.norm(D[:, lips] - D_gt[speech][:, lips], axis=-1).max(-1) * 1000

    mean_pose = np.repeat(data["coeffs"][::10].mean(0, keepdims=True), int(speech.sum()), 0)
    D_raw = motion.displacements(c_mean + (pred_c - c_mean) / gain)  # undo the gain
    return dict(
        lve_mm_without_gain=float(lve(D_raw[speech]).mean()),
        lve_mm=float(lve(D_pred[speech]).mean()),
        lve_mean_pose_mm=float(lve(motion.displacements(mean_pose)).mean()),
        note="LVE = max lip-vertex error per frame (speech frames only), vs PCA-128 "
        "reconstruction of the recorded scan",
    )


class AvatarRig:
    """Identity mesh + displacements per frame, with the mouth proxies posed by the jaw angle
    that best explains the chin motion."""

    def __init__(self, flame, identity):
        n_proxy = len(identity["proxy_P"])
        flame.add_proxy(identity["proxy_P"], identity["proxy_W"], identity["proxy_F"])
        beta = torch.tensor(identity["beta"], dtype=torch.float32)[None]
        offsets = torch.cat(
            [torch.tensor(identity["offsets"], dtype=torch.float32), torch.zeros(n_proxy, 3)]
        )
        jaw = torch.zeros(len(JAW_SWEEP), 3)
        jaw[:, 0] = torch.tensor(JAW_SWEEP, dtype=torch.float32)
        with torch.no_grad():
            self.sweep = flame(beta.expand(len(JAW_SWEEP), -1), None, flame.jaw_pose(jaw), offsets)[
                0
            ].numpy()  # [A, n + n_proxy, 3]
        self.V_id = self.sweep[np.argmin(np.abs(JAW_SWEEP))]
        n = N_FLAME_VERTICES
        self.chin = np.where(flame.lbs_weights[:n, JAW_JOINT].numpy() > 0.7)[0]
        self.D_sweep = self.sweep[:, self.chin] - self.V_id[self.chin]

    def meshes(self, D):
        """Displacements [T, 5023, 3] -> avatar meshes [T, n + n_proxy, 3], jaw angles [T]."""
        n = N_FLAME_VERTICES
        err = ((D[:, None, self.chin] - self.D_sweep[None]) ** 2).sum((-1, -2))  # [T, A]
        best = np.argmin(err, 1)
        V = np.repeat(self.V_id[None], len(D), 0)
        V[:, :n] += D
        V[:, n:] = self.sweep[best, n:]
        return torch.tensor(V, dtype=torch.float32), JAW_SWEEP[best]


def load_avatar():
    ckpt = torch.load(IDENTITY_DIR / "gaussian_avatar.pt", map_location="cpu", weights_only=False)
    init = {("vertex_albedo" if k == "albedo" else k): v for k, v in ckpt["init"].items()}
    avatar = GaussianAvatar(**init)
    avatar.load_state_dict(ckpt["state"])
    return avatar.eval(), ckpt["lighting"][ckpt["train"]].mean(0)


def lip_gap_strip(open_pred, open_gt, width, path):
    """Pre-rendered lip-gap plot (cursor added per frame). Returns the image and the x range
    of the plot area."""
    fig = plt.figure(figsize=(width / 100, 1.1), dpi=100)
    ax = fig.add_axes([0.07, 0.22, 0.91, 0.72])
    t = np.arange(len(open_pred)) / MOTION_FPS
    if open_gt is not None:
        ax.plot(t, open_gt, c="tab:blue", lw=1.2, label="recorded 3D")
    ax.plot(t, open_pred, c="tab:orange", lw=1.2, label="from audio")
    ax.set_xlim(0, t[-1])
    ax.tick_params(labelsize=6)
    ax.set_ylabel("lip gap mm", fontsize=6)
    ax.legend(fontsize=6, loc="upper right", ncol=2, frameon=False)
    fig.canvas.draw()
    strip = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return strip, int(0.07 * width), int(0.98 * width)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    log = Logger()
    use_all_cpu_threads()
    out = ensure_dir(FLAME_OUTPUTS / args.out)

    # 1. Audio (and ground-truth motion for VOCASET sentences).
    data = np.load(ROOT / args.data)
    motion = MotionBasis(data)
    segments, name = load_segments(args, data)

    # 2. Audio -> PCA coefficients per segment, with the articulation gain.
    model, ckpt = load_audio2motion(ROOT / args.ckpt)
    c_mean = ckpt["model"]["c_mean"].numpy()
    gain = resolve_gain(args.gain)
    log(f"articulation gain {gain} (c' = c_mean + g * (c - c_mean), calibrated in step 4c)")
    x, pred_c, gt_c, speech, spans = predict_motion(
        model,
        ckpt["args"].get("feature", "wav2vec2"),
        segments,
        c_mean,
        gain,
        int(round(args.gap * MOTION_FPS)),
    )
    n_frames = len(pred_c)
    log(
        f"audio: {name}, {len(x) / AUDIO_SR:.1f}s -> {n_frames} frames at {MOTION_FPS} fps "
        f"({len(segments)} segment(s))"
    )
    D_pred = motion.displacements(pred_c)
    D_gt = motion.displacements(gt_c) if gt_c is not None else None
    log("lip motion predicted from audio")

    lines = []
    if not args.no_subtitles:
        lines = subtitles.group_lines(subtitles.transcribe(x), spans)
        log("subtitles: " + " / ".join(subtitles.line_text(words) for *_, words in lines))

    metrics = dict(
        audio=name,
        seconds=len(x) / AUDIO_SR,
        frames=n_frames,
        model=args.ckpt,
        articulation_gain=gain,
    )
    if D_gt is not None:
        metrics.update(lip_error_metrics(motion, data, D_pred, D_gt, speech, pred_c, c_mean, gain))
        log(
            f"lip vertex error on this clip: {metrics['lve_mm']:.2f} mm "
            f"(mean pose {metrics['lve_mean_pose_mm']:.2f} mm)"
        )

    # 3. Identity mesh + motion, jaw angle for the mouth proxies.
    identity = np.load(IDENTITY_DIR / "identity.npz")
    rig = AvatarRig(load_flame(load_config()), identity)
    V_pred, jaw_pred = rig.meshes(D_pred)
    V_gt = rig.meshes(D_gt)[0] if D_gt is not None else None

    # 4. Render the Gaussian avatar.
    avatar, lighting = load_avatar()
    S = args.res
    cam = dict(
        rvec=torch.tensor(np.radians([3, args.yaw, 0]), dtype=torch.float32),
        t=torch.tensor([0.0, 0.025, 0.62]),
        f=2.2 * S,
        cx=S / 2,
        cy=S / 2,
    )
    bg = torch.full((S, S, 3), 0.95)
    up, lo = int(identity["up_mid"]), int(identity["lo_mid"])

    def lip_gap_mm(V):
        return np.linalg.norm(V[:, up] - V[:, lo], axis=-1) * 1000

    def render(V):
        return (avatar.render(V, cam, lighting, S, S, bg).clamp(0, 1).numpy() * 255).astype(
            np.uint8
        )

    open_pred = lip_gap_mm(V_pred.numpy())
    open_gt = lip_gap_mm(V_gt.numpy()) if V_gt is not None else None
    strip_width = 2 * S if V_gt is not None else S
    strip, x0, x1 = lip_gap_strip(open_pred, open_gt, strip_width, out / "01_lip_opening.png")

    comparison, solo, t_render = [], [], time.time()
    with torch.no_grad():
        for k in range(n_frames):
            pred_img = render(V_pred[k])
            row = [label(pred_img, "audio -> 3D motion -> avatar")]
            if V_gt is not None:
                row = [label(render(V_gt[k]), "recorded 3D motion (reference)")] + row
            strip_k = strip.copy()
            cursor = int(x0 + (x1 - x0) * k / max(n_frames - 1, 1))
            strip_k[:, max(cursor - 1, 0) : cursor + 1] = (200, 30, 30)
            t = k / MOTION_FPS
            if lines:
                band = subtitles.render_band(lines, t, strip_width, 0.5)
            else:
                band = np.zeros((0, strip_width, 3), np.uint8)
            comparison.append(
                np.concatenate([np.concatenate(row, 1), band, strip_k[:, :strip_width]], 0)
            )
            solo.append(
                np.concatenate([pred_img, subtitles.render_band(lines, t, S, 0.4)], 0)
                if lines
                else pred_img
            )
            if k % 30 == 0:
                log(
                    f"rendered {k}/{n_frames} frames "
                    f"({(time.time() - t_render) / (k + 1):.2f}s/frame)"
                )

    if lines:
        srt = subtitles.to_srt(lines)
        for stem in ["comparison", "talking_head"]:
            (out / f"{stem}.srt").write_text(srt, encoding="utf-8")
        metrics["subtitles"] = [subtitles.line_text(words) for *_, words in lines]
    wav = out / "audio.wav"
    n_samples = int(n_frames / MOTION_FPS * AUDIO_SR)
    wavfile.write(wav, AUDIO_SR, (np.clip(x[:n_samples], -1, 1) * 32767).astype(np.int16))
    for frames, stem in [(comparison, "comparison"), (solo, "talking_head")]:
        write_video_with_audio(frames, wav, out / f"{stem}.mp4", fps=MOTION_FPS)
    keyframes = np.linspace(0, n_frames - 1, 6).astype(int)
    cv2.imwrite(
        str(out / "02_keyframes.png"),
        cv2.cvtColor(np.concatenate([comparison[k] for k in keyframes], 1), cv2.COLOR_RGB2BGR),
    )

    lip_gap = dict(pred_mean=float(open_pred.mean()), pred_std=float(open_pred.std()))
    if open_gt is not None:
        lip_gap.update(
            gt_mean=float(open_gt.mean()),
            gt_std=float(open_gt.std()),
            corr=float(np.corrcoef(open_pred, open_gt)[0, 1]),
        )
    renders_per_frame = 2 if V_gt is not None else 1
    metrics.update(
        jaw_angle_range_rad=[float(jaw_pred.min()), float(jaw_pred.max())],
        lip_gap_mm=lip_gap,
        render_seconds_per_frame=(time.time() - t_render) / n_frames / renders_per_frame,
        total_seconds=time.time() - log.t0,
    )
    save_json(metrics, out / "metrics.json")
    log(f"videos -> {out / 'talking_head.mp4'} and {out / 'comparison.mp4'}")
    print(metrics)


if __name__ == "__main__":
    main()
