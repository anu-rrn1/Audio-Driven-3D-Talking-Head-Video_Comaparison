"""Step 4c: articulation calibration of the audio-to-motion model (no retraining).

The step-4b model under-articulates: the predicted lip opening moves about half as much as
the recorded one, because a regressor trained with L2-type losses hedges towards the
average mouth when the audio is ambiguous. A post-hoc gain around the training mean pose
c_mean (model.c_mean) compensates:

    c' = c_mean + g * (c_pred - c_mean)

g is chosen on the two validation speakers so that the temporal std of the predicted lip
gap matches the recorded one, then evaluated once on the two unseen test speakers. The
gain minimising the mean lip-vertex error on validation is reported as well; the gap
between the two gains is the hedging effect.

Outputs: outputs/flame_pipeline/04c_articulation/{gain.json, metrics.json, 01_gain_curve.png}
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from tinypt import visualization as viz
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
from tinypt.io_utils import save_json

GAINS = np.round(np.arange(0.8, 2.61, 0.1), 2)
DEFAULT_CKPT = "outputs/flame_pipeline/04b_audio2motion_v1_better_lip_error/audio2motion.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--data", default="data/vocaset_motion/vocaset_motion.npz")
    parser.add_argument("--out", default="04c_articulation")
    return parser.parse_args()


def load_split_predictions(model, data, feature):
    """Ground truth and (uncalibrated) predictions for the val and test sequences."""
    sequences = {"val": [], "test": []}
    for seq in iter_sequences(data, splits=("val", "test")):
        n = min(len(seq["coeffs"]), int(round(len(seq["audio"]) / AUDIO_SR * MOTION_FPS)))
        sequences[seq["split"]].append(
            dict(spk=seq["spk"], audio=seq["audio"], gt=seq["coeffs"][:n], n=n)
        )
    print(
        {k: len(v) for k, v in sequences.items()},
        f"sequences; extracting {feature} features",
        flush=True,
    )
    for items in sequences.values():
        preds = predict_from_audio(
            model, [it["audio"] for it in items], [it["n"] for it in items], feature
        )
        for it, pred in zip(items, preds):
            it["pred"] = pred
    return sequences


def lip_stats(items, gain, c_mean, motion, template, up, lo):
    """Lip metrics over a split for one gain: LVE (max lip error), MVE (mean lip error),
    and lip-gap std ratio, correlation and mean absolute error."""
    lve, mve, std_ratio, corr, mae = [], [], [], [], []
    lips = motion.lips
    for it in items:
        D_pred = motion.displacements(apply_gain(it["pred"], c_mean, gain))
        D_gt = motion.displacements(it["gt"])
        err = np.linalg.norm(D_pred[:, lips] - D_gt[:, lips], axis=-1) * 1000
        lve.append(err.max(-1).mean())
        mve.append(err.mean())
        gap_pred = np.linalg.norm((template + D_pred)[:, up] - (template + D_pred)[:, lo], axis=-1)
        gap_gt = np.linalg.norm((template + D_gt)[:, up] - (template + D_gt)[:, lo], axis=-1)
        gap_pred, gap_gt = gap_pred * 1000, gap_gt * 1000
        std_ratio.append(gap_pred.std() / max(gap_gt.std(), 1e-6))
        corr.append(np.corrcoef(gap_pred, gap_gt)[0, 1])
        mae.append(np.abs(gap_pred - gap_gt).mean())
    return dict(
        lve_mm=float(np.mean(lve)),
        mve_mm=float(np.mean(mve)),
        lipgap_std_ratio=float(np.mean(std_ratio)),
        lipgap_corr=float(np.nanmean(corr)),
        lipgap_mae_mm=float(np.mean(mae)),
    )


def plot_gain_curves(curve, g_amp, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for split, ls in [("val", "--"), ("test", "-")]:
        ax[0].plot(GAINS, [s["lipgap_std_ratio"] for s in curve[split]], ls, label=split)
        ax[1].plot(GAINS, [s["lve_mm"] for s in curve[split]], ls, label=f"{split} LVE (max)")
        ax[1].plot(GAINS, [s["mve_mm"] for s in curve[split]], ls, label=f"{split} mean lip err")
        ax[2].plot(GAINS, [s["lipgap_mae_mm"] for s in curve[split]], ls, label=split)
    ax[0].axhline(1, c="k", lw=0.6)
    ax[0].set_title("lip-gap motion range (pred / recorded)")
    ax[1].set_title("lip vertex error (mm)")
    ax[2].set_title("lip-gap error (mm)")
    for a in ax:
        a.axvline(g_amp, c="tab:red", lw=0.8)
        a.set_xlabel("articulation gain g")
        a.legend(fontsize=7)
    fig.suptitle(f"gain chosen on val speakers (range match): g = {g_amp}", fontsize=10)
    viz.save_figure(fig, path)


def main():
    args = parse_args()
    out = ensure_dir(FLAME_OUTPUTS / args.out)
    use_all_cpu_threads()

    data = np.load(ROOT / args.data)
    motion = MotionBasis(data)
    identity = np.load(IDENTITY_DIR / "identity.npz")
    up, lo = int(identity["up_mid"]), int(identity["lo_mid"])
    template = load_flame(load_config()).v_template.numpy()[:N_FLAME_VERTICES]

    model, ckpt = load_audio2motion(ROOT / args.ckpt)
    c_mean = ckpt["model"]["c_mean"].numpy()
    sequences = load_split_predictions(model, data, ckpt["args"].get("feature", "wav2vec2"))

    def stats(split, gain):
        return lip_stats(sequences[split], gain, c_mean, motion, template, up, lo)

    curve = {split: [stats(split, g) for g in GAINS] for split in sequences}
    val_ratio = np.array([s["lipgap_std_ratio"] for s in curve["val"]])
    g_amp = round(float(np.interp(1.0, val_ratio, GAINS)), 2)  # the ratio grows with g
    g_mve = float(GAINS[np.argmin([s["mve_mm"] for s in curve["val"]])])

    metrics = dict(
        c_bar="training mean of PCA coefficients (model.c_mean)",
        chosen_on="val speakers",
        g_amplitude_match=g_amp,
        g_min_mean_error=g_mve,
        test=dict(
            g_1=stats("test", 1.0),
            g_amplitude_match=stats("test", g_amp),
            g_min_mean_error=stats("test", g_mve),
        ),
        val=dict(g_1=stats("val", 1.0), g_amplitude_match=stats("val", g_amp)),
        n_sequences={k: len(v) for k, v in sequences.items()},
    )
    save_json(metrics, out / "metrics.json")
    save_json(
        dict(
            gain=g_amp,
            c_bar_from=args.ckpt,
            note="c' = c_mean + gain * (c_pred - c_mean); fitted on val speakers",
        ),
        out / "gain.json",
    )
    plot_gain_curves(curve, g_amp, out / "01_gain_curve.png")
    print(
        json.dumps(
            {k: metrics[k] for k in ["g_amplitude_match", "g_min_mean_error", "test"]}, indent=1
        )
    )


if __name__ == "__main__":
    main()
