"""Step 4b: train the audio-to-motion network on VOCASET (GPU recommended).

    16 kHz audio -> frozen wav2vec 2.0 (50 Hz, 768-d) -> resampled to 30 fps
        -> Audio2Motion Transformer (tinypt/audio/audio2motion.py) -> 128 PCA coefficients
        -> displacements = mean + coeffs @ basis, added to any FLAME identity

Loss (in mm^2): full-face vertex MSE (equal to the coefficient MSE, the basis is
orthonormal) + lip-vertex MSE + velocity MSE + motion-spread loss. Style: one token per
training speaker and a neutral token, used with probability 0.15 during training and
always for unseen speakers.

Two configurations were trained (outputs/flame_pipeline/04b_audio2motion_v1_better_lip_error and 04b_audio2motion_v2_better_jaw_correlation):
    v1: --epochs 150 --dropout 0.1 --wd 0.01 --w_amp 0 --speeds 1.0 --time_mask 0
        (overfits after about 25 epochs and produces about half the real jaw motion)
    v2: the defaults below: stronger regularisation, speed perturbation x0.9 / x1.1 with
        the motion time-stretched to match, SpecAugment-style time masking, and a loss
        matching the per-sequence temporal std of every PCA mode

Evaluation on the two unseen test speakers at 30 fps: lip vertex error (LVE, max L2 over
lip vertices per frame, averaged) against zero-motion and mean-pose baselines, and the
correlation of PCA mode 1 (jaw opening). Errors are measured against the PCA-128
reconstruction of the scans, which itself adds 0.81 mm (step 4a).

Usage:
    python scripts/flame_pipeline/04b_train_audio2motion.py              # full training
    python scripts/flame_pipeline/04b_train_audio2motion.py --feature mfcc --epochs 3 --quick
"""

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from tinypt import visualization as viz
from tinypt.audio.audio2motion import MOTION_FPS, Audio2Motion
from tinypt.audio.features import extract_features, resample_features
from tinypt.audio.vocaset import iter_sequences
from tinypt.config import DATA_DIR, FLAME_OUTPUTS

N_VERTICES = 5023
PCA_FLOOR_LVE_MM = 0.81  # representation error of PCA-128 on the test speakers (step 4a)
QUICK_SEQUENCES = {"train": 24, "val": 6, "test": 6}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(DATA_DIR / "vocaset_motion" / "vocaset_motion.npz"))
    parser.add_argument("--out", default=str(FLAME_OUTPUTS / "04b_audio2motion_v2_better_jaw_correlation"))
    parser.add_argument("--feature", default="wav2vec2", choices=["wav2vec2", "mfcc"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--w_amp", type=float, default=1.0, help="motion-spread loss weight")
    parser.add_argument(
        "--speeds", default="1.0,0.9,1.1", help="speed-perturbation factors, starting with 1.0"
    )
    parser.add_argument(
        "--time_mask", type=int, default=2, help="masked feature spans per training sequence"
    )
    parser.add_argument("--neutral_p", type=float, default=0.15)
    parser.add_argument("--quick", action="store_true", help="few sequences, for a smoke test")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# Data


def change_speed(x, coeffs, speed):
    """Play the audio `speed` times faster (tempo and pitch, like Kaldi's speed perturbation)
    and time-stretch the motion to match."""
    if speed == 1.0:
        return x, coeffs
    xs = np.interp(np.arange(0, len(x) - 1, speed), np.arange(len(x)), x).astype(np.float32)
    n = max(2, int(round(len(coeffs) / speed)))
    t_new = np.linspace(0, len(coeffs) - 1, n)
    cs = np.stack(
        [np.interp(t_new, np.arange(len(coeffs)), coeffs[:, k]) for k in range(coeffs.shape[1])], 1
    )
    return xs, cs.astype(np.float32)


def take_quick_subset(items):
    """The first few sequences of each split, in dataset order."""
    count = {k: 0 for k in QUICK_SEQUENCES}
    kept = []
    for it in items:
        count[it["split"]] += 1
        if count[it["split"]] <= QUICK_SEQUENCES[it["split"]]:
            kept.append(it)
    return kept


def load_dataset(path, feature, device, cache_dir, quick=False, speeds=(1.0,)):
    """Sequences with audio features at 30 fps. Each item keeps a list of variants
    (features, coeffs), one per speed; extra speeds are added only for training sequences.
    Features are cached in `cache_dir`."""
    data = np.load(path)
    subjects, split = list(data["subjects"]), list(data["split"])
    train_speakers = [str(s) for s, p in zip(subjects, split) if p == "train"]
    items = list(iter_sequences(data))
    if quick:
        items = take_quick_subset(items)
    for it in items:
        it["var"] = []
    for speed in speeds:
        group = [it for it in items if speed == 1.0 or it["split"] == "train"]
        pairs = [change_speed(it["audio"], it["coeffs"], speed) for it in group]
        cache = Path(cache_dir) / f"features_{feature}_x{speed:g}{'_quick' if quick else ''}.npz"
        if cache.exists():
            cached = np.load(cache)
            feats = [cached[f"f{i}"] for i in range(len(group))]
            print(f"audio features (speed {speed:g}) loaded from {cache}")
        else:
            t = time.time()
            raw, rate = extract_features([p[0] for p in pairs], feature, device)
            feats = [resample_features(f, rate, len(p[1]), MOTION_FPS) for f, p in zip(raw, pairs)]
            np.savez(cache, **{f"f{i}": f for i, f in enumerate(feats)})
            print(
                f"audio features ({feature}, speed {speed:g}) computed in "
                f"{time.time() - t:.0f}s -> {cache}"
            )
        for it, f, p in zip(group, feats, pairs):
            it["var"].append((f, p[1]))
    for it in items:
        it["feat"], it["coeffs"] = it["var"][0]  # speed 1.0, used for evaluation
        del it["audio"]
    basis = dict(
        basis=torch.from_numpy(data["basis"]),
        mean=torch.from_numpy(data["mean"]),
        lips=data["lips"],
        explained=float(data["explained_variance"][: data["basis"].shape[0]].sum()),
    )
    return items, train_speakers, basis


def batches(items, batch_size, shuffle, rng):
    order = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for b in range(0, len(order), batch_size):
        yield [items[j] for j in order[b : b + batch_size]]


def collate(group, train_speakers, device, neutral_p=0.0, rng=None):
    """Pad a group of sequences. Returns features, targets, padding mask and style ids."""
    T = max(len(it["coeffs"]) for it in group)
    B, C, K = len(group), group[0]["feat"].shape[1], group[0]["coeffs"].shape[1]
    feat, target = torch.zeros(B, T, C), torch.zeros(B, T, K)
    pad = torch.ones(B, T, dtype=torch.bool)
    style = torch.full((B,), len(train_speakers), dtype=torch.long)  # neutral by default
    for i, it in enumerate(group):
        n = len(it["coeffs"])
        feat[i, :n] = torch.from_numpy(it["feat"])
        target[i, :n] = torch.from_numpy(it["coeffs"])
        pad[i, :n] = False
        # Training speakers get their own style token, except with probability neutral_p.
        if it["spk"] in train_speakers and not (rng is not None and rng.random() < neutral_p):
            style[i] = train_speakers.index(it["spk"])
    return feat.to(device), target.to(device), pad.to(device), style.to(device)


def time_mask(feat, pad, rng, spans=2, max_frac=0.1):
    """SpecAugment-style masking: zero `spans` random time spans (each at most `max_frac`
    of the sequence) per sample."""
    for i in range(feat.shape[0]):
        n = int((~pad[i]).sum())
        for _ in range(spans):
            w = int(rng.integers(0, max(1, int(n * max_frac)) + 1))
            s = int(rng.integers(0, max(1, n - w)))
            feat[i, s : s + w] = 0
    return feat


# Losses and metrics


class LipGeometry:
    """Maps PCA coefficients to lip-vertex displacements."""

    def __init__(self, basis, device):
        B = basis["basis"].to(device)  # [K, D]
        lips = basis["lips"]
        idx = torch.as_tensor(np.stack([3 * lips, 3 * lips + 1, 3 * lips + 2], 1).reshape(-1))
        self.basis_lips = B[:, idx.to(device)]
        self.mean_lips = basis["mean"].to(device)[idx.to(device)]
        self.n_lips = len(lips)

    def lips(self, c):
        return (c @ self.basis_lips + self.mean_lips).reshape(*c.shape[:-1], self.n_lips, 3)


def sequence_std(x, valid):
    """Temporal std of each coefficient over the valid frames: [B, T, K] -> [B, K]."""
    n = valid.sum(1).clamp_min(1)
    m = (x * valid).sum(1) / n
    return (((x - m[:, None]) ** 2 * valid).sum(1) / n + 1e-12).sqrt()


def loss_fn(pred, target, pad, geo, w_lip=1.0, w_vel=0.5, w_amp=0.0):
    """All terms are mean squared per-vertex errors in mm^2. The basis rows are orthonormal,
    so ||c_pred - c_gt||^2 is the summed squared vertex error.

    The motion-spread term matches each sequence's temporal std per PCA mode: plain MSE
    rewards hedging towards the mean pose, which shows up as under-articulated lips.
    """
    valid = (~pad).float()[..., None]
    n = valid.sum()
    l_face = ((((pred - target) * 1000) ** 2) * valid).sum() / n / N_VERTICES
    lip_err = ((geo.lips(pred) - geo.lips(target)) * 1000) ** 2
    l_lip = (lip_err.sum(-1).mean(-1, keepdim=True) * valid).sum() / n
    pair_valid = valid[:, 1:] * valid[:, :-1]
    vel_err = (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
    l_vel = (((vel_err * 1000) ** 2) * pair_valid).sum() / (pair_valid.sum() + 1e-8) / N_VERTICES
    std_err = (sequence_std(pred, valid) - sequence_std(target, valid)) * 1000
    l_amp = (std_err**2).sum(-1).mean() / N_VERTICES
    total = l_face + w_lip * l_lip + w_vel * l_vel + w_amp * l_amp
    return total, dict(face=l_face.item(), lip=l_lip.item(), vel=l_vel.item(), amp=l_amp.item())


def max_lip_error_mm(geo, c, lips_target):
    return ((geo.lips(c) - lips_target).norm(dim=-1).max(-1).values * 1000).cpu().numpy()


@torch.no_grad()
def evaluate(model, items, train_speakers, geo, device, train_mean_c):
    """LVE of the model (neutral style) and of the zero-motion / mean-pose baselines,
    plus correlation and amplitude ratio of the jaw mode (PCA mode 1)."""
    model.eval()
    lve, lve_zero, lve_mean, jaw_r, jaw_amp, per_sequence = [], [], [], [], [], []
    for it in items:
        feat, target, pad, style = collate([it], train_speakers, device)
        style[:] = len(train_speakers)
        pred = model(feat, style, pad)[0]
        lips_target = geo.lips(target[0])
        e = max_lip_error_mm(geo, pred, lips_target)
        lve.append(e)
        lve_zero.append((lips_target.norm(dim=-1).max(-1).values * 1000).cpu().numpy())
        lve_mean.append(max_lip_error_mm(geo, train_mean_c.expand_as(target[0]), lips_target))
        p0, t0 = pred[:, 0].cpu().numpy(), target[0, :, 0].cpu().numpy()
        if p0.std() > 1e-9 and t0.std() > 1e-9:
            jaw_r.append(np.corrcoef(p0, t0)[0, 1])
            jaw_amp.append(p0.std() / t0.std())
        per_sequence.append(
            dict(spk=it["spk"], sent=it["sent"], lve=float(e.mean()), pred_jaw=p0, gt_jaw=t0)
        )
    metrics = dict(
        lve_mm=float(np.concatenate(lve).mean()),
        lve_zero_mm=float(np.concatenate(lve_zero).mean()),
        lve_meanpose_mm=float(np.concatenate(lve_mean).mean()),
        jaw_mode_corr=float(np.mean(jaw_r)) if jaw_r else 0.0,
        jaw_amplitude_ratio=float(np.mean(jaw_amp)) if jaw_amp else 0.0,
        frames=int(sum(len(x) for x in lve)),
    )
    return metrics, per_sequence


# Training


def train(
    model,
    train_items,
    val_items,
    train_speakers,
    geo,
    args,
    device,
    rng,
    out,
    train_mean_c,
    ckpt_extra,
):
    """Train with AdamW + one-cycle LR; keep the checkpoint with the lowest validation LVE."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(train_items) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    history, best, t0 = [], (1e9, -1), time.time()
    for epoch in range(args.epochs):
        model.train()
        parts_log = []
        for group in batches(train_items, args.bs, True, rng):
            # Pick one speed variant per sequence.
            group = [
                dict(it, feat=v[0], coeffs=v[1])
                for it in group
                for v in [it["var"][rng.integers(len(it["var"]))]]
            ]
            feat, target, pad, style = collate(group, train_speakers, device, args.neutral_p, rng)
            if args.time_mask:
                feat = time_mask(feat, pad, rng, args.time_mask)
            feat = feat + 0.01 * torch.randn_like(feat) * feat.std()  # light feature noise
            loss, parts = loss_fn(model(feat, style, pad), target, pad, geo, w_amp=args.w_amp)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            parts_log.append(parts)

        val, _ = evaluate(model, val_items, train_speakers, geo, device, train_mean_c)
        row = dict(
            epoch=epoch + 1,
            **{f"train_{k}": float(np.mean([p[k] for p in parts_log])) for k in parts_log[0]},
            val_lve_mm=val["lve_mm"],
        )
        history.append(row)
        if val["lve_mm"] < best[0]:
            best = (val["lve_mm"], epoch + 1)
            torch.save(
                dict(model=model.state_dict(), args=vars(args), **ckpt_extra),
                out / "audio2motion.pt",
            )
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 15) == 0 or epoch + 1 == args.epochs:
            print(
                f"ep {epoch + 1:4d}  train lip {row['train_lip']:.3f}  "
                f"val LVE {val['lve_mm']:.3f} mm  jaw amp {val['jaw_amplitude_ratio']:.2f}  "
                f"(best {best[0]:.3f} @ {best[1]})  {time.time() - t0:.0f}s",
                flush=True,
            )
    return history, best[1], time.time() - t0


def plot_results(out, history, val, test, per_sequence):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    epochs = [h["epoch"] for h in history]
    ax[0].plot(epochs, [h["train_lip"] for h in history], label="train lip MSE (mm$^2$)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(epochs, [h["val_lve_mm"] for h in history], label="val LVE (mm)")
    ax[1].axhline(val["lve_zero_mm"], ls=":", c="gray", label="zero motion")
    ax[1].axhline(val["lve_meanpose_mm"], ls="--", c="gray", label="mean pose")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    median = sorted(per_sequence, key=lambda r: r["lve"])[len(per_sequence) // 2]
    t = np.arange(len(median["gt_jaw"])) / MOTION_FPS
    ax[2].plot(t, median["gt_jaw"], label="ground truth")
    ax[2].plot(t, median["pred_jaw"], label="predicted from audio")
    ax[2].set_title(
        f"jaw mode, median test sequence ({median['spk'][-12:]} "
        f"s{median['sent']:02d}, LVE {median['lve']:.2f} mm)"
    )
    ax[2].set_xlabel("time (s)")
    ax[2].legend()
    viz.save_figure(fig, out / "01_training_and_jaw.png", dpi=110)

    names = ["zero motion", "mean pose", "ours (audio)"]
    values = [test["lve_zero_mm"], test["lve_meanpose_mm"], test["lve_mm"]]
    fig = plt.figure(figsize=(6, 4))
    plt.bar(names, values, color=["#bbb", "#999", "#2a7"])
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.ylabel("lip vertex error, unseen speakers (mm)")
    viz.save_figure(fig, out / "02_test_lve.png", dpi=110)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {device}, feature {args.feature}")

    speeds = [float(s) for s in args.speeds.split(",")]
    assert speeds[0] == 1.0, "--speeds must start with 1.0"
    items, train_speakers, basis = load_dataset(
        args.data, args.feature, device, out, args.quick, speeds
    )
    splits = {s: [it for it in items if it["split"] == s] for s in ("train", "val", "test")}
    n_train_frames = sum(len(it["coeffs"]) for it in splits["train"])
    print(
        f"sequences: train {len(splits['train'])}, val {len(splits['val'])}, "
        f"test {len(splits['test'])}  ({n_train_frames} train frames @ {MOTION_FPS} fps)"
    )
    geo = LipGeometry(basis, device)
    all_train = np.concatenate([it["coeffs"] for it in splits["train"]])
    train_mean_c = torch.from_numpy(all_train.mean(0)).to(device)

    in_dim, k = splits["train"][0]["feat"].shape[1], splits["train"][0]["coeffs"].shape[1]
    model = Audio2Motion(
        in_dim, k, len(train_speakers), args.d, args.layers, dropout=args.dropout
    ).to(device)
    model.c_mean.copy_(torch.from_numpy(all_train.mean(0)))
    model.c_std.copy_(torch.from_numpy(all_train.std(0) + 1e-6))
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    ckpt_extra = dict(in_dim=in_dim, train_spk=train_speakers, fps=MOTION_FPS, k=k)
    history, best_epoch, seconds = train(
        model,
        splits["train"],
        splits["val"],
        train_speakers,
        geo,
        args,
        device,
        rng,
        out,
        train_mean_c,
        ckpt_extra,
    )

    model.load_state_dict(torch.load(out / "audio2motion.pt", map_location=device)["model"])
    test, per_sequence = evaluate(model, splits["test"], train_speakers, geo, device, train_mean_c)
    val, _ = evaluate(model, splits["val"], train_speakers, geo, device, train_mean_c)
    metrics = dict(
        test=test,
        val=val,
        best_epoch=best_epoch,
        representation_floor_lve_mm=PCA_FLOOR_LVE_MM,
        pca_explained_variance=basis["explained"],
        feature=args.feature,
        epochs=args.epochs,
        train_seconds=seconds,
        args=vars(args),
        note="LVE vs PCA-128 reconstruction of the scans; unseen test speakers, neutral style",
    )
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ("test", "val", "best_epoch")}, indent=2))
    plot_results(out, history, val, test, per_sequence)
    print("saved ->", out)


if __name__ == "__main__":
    main()
