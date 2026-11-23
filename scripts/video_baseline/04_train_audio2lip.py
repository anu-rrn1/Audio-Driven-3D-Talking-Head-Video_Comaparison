"""Step 4: audio-conditioned lip deformation network (tinypt/video_baseline/lip_deform_net.py).

Trained on the first (1 - val_ratio) of the video and validated on the held-out tail,
against zero-motion and mean-pose baselines.

Outputs: checkpoints/audio2lip.pt, outputs/video_baseline/04_audio2lip/
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from tinypt import visualization as viz
from tinypt.audio.features import context_windows, video_frame_features
from tinypt.config import (
    BASELINE_OUTPUTS,
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    ensure_dir,
    load_config,
    seed_everything,
)
from tinypt.io_utils import load_json, save_json
from tinypt.video_baseline.face_mesh import LOWER_INNER_LIP, UPPER_INNER_LIP
from tinypt.video_baseline.lip_deform_net import LipDeformNet

BATCH_EVAL = 128


class LipData:
    """Audio windows, normalised targets and the canonical lip cloud of the video."""

    def __init__(self, cfg):
        meta = load_json(PROCESSED_DIR / "meta.json")
        self.mesh = np.load(PROCESSED_DIR / "mesh_data.npz")
        points = np.load(PROCESSED_DIR / "points.npz")
        self.T, self.fps = meta["T"], meta["fps"]
        self.n_train = int(self.T * (1 - cfg["data"]["val_ratio"]))
        self.context = cfg["audio"]["context"]
        feats = video_frame_features(
            PROCESSED_DIR / "audio.wav",
            self.fps,
            cfg["audio"]["feature"],
            cfg["audio"]["sr"],
            self.T,
        )
        self.audio_dim = feats.shape[1]
        self.mu, self.sd = feats[: self.n_train].mean(0), feats[: self.n_train].std(0) + 1e-6
        windows = context_windows((feats - self.mu) / self.sd, self.context).astype(np.float32)
        self.Y = points["D_pts"].astype(np.float32)
        self.d_scale = float(self.Y[: self.n_train].std())
        self.X_t, self.Y_t = torch.tensor(windows), torch.tensor(self.Y / self.d_scale)
        self.pts = torch.tensor(
            np.concatenate([points["P0n"], points["normals"]], 1), dtype=torch.float32
        )
        self.knn = torch.tensor(points["knn"]).long()
        self.K = int(points["K"])
        self.P0 = points["P0"]
        region = self.mesh["region_idx"]
        self.lo_local = int(np.where(region == LOWER_INNER_LIP)[0][0])
        self.up_local = int(np.where(region == UPPER_INNER_LIP)[0][0])
        self.val_idx = np.arange(self.n_train, self.T)

    def lve(self, P, idx):
        """Mean lip-vertex error (pixels) over the K mesh-vertex points."""
        return float(np.linalg.norm(P[:, : self.K] - self.Y[idx][:, : self.K], axis=-1).mean())

    def openness(self, P):
        V = self.mesh["V_can"]
        gap = (V[LOWER_INNER_LIP, 1] + P[:, self.lo_local, 1]) - (
            V[UPPER_INNER_LIP, 1] + P[:, self.up_local, 1]
        )
        return gap / float(self.mesh["face_h"])


def predict(model, data, idx, smooth_sigma=None):
    model.eval()
    with torch.no_grad():
        P = (
            torch.cat(
                [
                    model(data.X_t[torch.as_tensor(idx[i : i + BATCH_EVAL])], data.pts, data.knn)
                    for i in range(0, len(idx), BATCH_EVAL)
                ]
            ).numpy()
            * data.d_scale
        )
    if smooth_sigma is None:
        return P
    return gaussian_filter1d(P, smooth_sigma, axis=0, mode="nearest")


def main():
    cfg = load_config()
    seed_everything(0)
    out = ensure_dir(BASELINE_OUTPUTS / "04_audio2lip")
    ensure_dir(CHECKPOINT_DIR)
    train_cfg = cfg["train"]
    data = LipData(cfg)
    n_train, val_idx, ctx = data.n_train, data.val_idx, data.context
    print(
        f"T={data.T} train={n_train} val={data.T - n_train} audio_dim={data.audio_dim} "
        f"d_scale={data.d_scale:.3f}px"
    )

    model = LipDeformNet(data.audio_dim, 6, train_cfg["dim"], 2 * ctx + 1)
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters:", n_params)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_cfg["epochs"])

    zero_lve = data.lve(np.zeros_like(data.Y[val_idx]), val_idx)
    mean_pose = np.repeat(data.Y[:n_train].mean(0, keepdims=True), len(val_idx), 0)
    mean_lve = data.lve(mean_pose, val_idx)
    history, best, best_epoch, t0 = [], 1e9, -1, time.time()
    for epoch in range(train_cfg["epochs"]):
        model.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, train_cfg["batch_size"]):
            b = perm[i : i + train_cfg["batch_size"]]
            x = data.X_t[b] + train_cfg["noise"] * torch.randn_like(data.X_t[b])
            loss = F.l1_loss(model(x, data.pts, data.knn), data.Y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        val_lve = data.lve(predict(model, data, val_idx), val_idx) if len(val_idx) else float("nan")
        history.append((total / n_train, val_lve))
        if val_lve < best:
            best, best_epoch = val_lve, epoch
            torch.save(
                dict(
                    state=model.state_dict(),
                    audio_dim=data.audio_dim,
                    dim=train_cfg["dim"],
                    L=2 * ctx + 1,
                    d_scale=data.d_scale,
                    mu=data.mu,
                    sd=data.sd,
                    feature=cfg["audio"]["feature"],
                    context=ctx,
                    epoch=epoch,
                ),
                CHECKPOINT_DIR / "audio2lip.pt",
            )
        if epoch % 5 == 0 or epoch == train_cfg["epochs"] - 1:
            print(
                f"ep {epoch:3d}  train L1(norm) {total / n_train:.4f}  val LVE {val_lve:.3f}px  "
                f"(zero-motion baseline {zero_lve:.3f}px)  {time.time() - t0:.0f}s"
            )

    # Final evaluation with the best checkpoint.
    ckpt = torch.load(CHECKPOINT_DIR / "audio2lip.pt", weights_only=False)
    model.load_state_dict(ckpt["state"])
    P_val = predict(model, data, val_idx, train_cfg["smooth_sigma"])
    open_scale = float(data.mesh["open_scale"])
    open_pred = data.openness(P_val[:, : data.K]) / open_scale
    open_gt = data.mesh["open_series"][val_idx] / open_scale
    corr = float(np.corrcoef(open_pred, open_gt)[0, 1])
    metrics = dict(
        best_epoch=best_epoch,
        val_LVE_px=data.lve(P_val, val_idx),
        baseline_zero_motion_LVE_px=zero_lve,
        baseline_mean_pose_LVE_px=mean_lve,
        mouth_opening_pearson_r=corr,
        n_params=n_params,
    )
    save_json(metrics, out / "metrics.json")
    print(metrics)

    # Figures.
    h = np.array(history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(h[:, 0])
    ax[0].set_title("train L1 (normalised)")
    ax[0].set_xlabel("epoch")
    ax[0].grid(alpha=0.3)
    ax[1].plot(h[:, 1], label="model")
    ax[1].axhline(zero_lve, c="r", ls="--", label="zero-motion baseline")
    ax[1].axhline(mean_lve, c="orange", ls="--", label="mean-pose baseline")
    ax[1].set_title("validation lip-vertex error (px)")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    viz.save_figure(fig, out / "01_training_curves.png")

    viz.plot_curves(
        [
            ("ground truth (from video mesh)", open_gt, dict(color="k")),
            ("predicted from audio only", open_pred, dict(color="tab:red")),
        ],
        out / "02_mouth_opening_val.png",
        f"Held-out segment: mouth opening, Pearson r = {corr:.2f}",
        ylabel="opening / p95",
        fps=data.fps,
    )

    P0, K = data.P0, data.K
    shown = np.linspace(0, len(val_idx) - 1, 6).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(17, 3.2))
    for ax, k in zip(axes, shown):
        gt = P0[:K] + data.Y[val_idx[k]][:K]
        pred = P0[:K] + P_val[k][:K]
        ax.scatter(P0[:K, 0], P0[:K, 1], s=6, c="lightgray", label="canonical")
        ax.scatter(gt[:, 0], gt[:, 1], s=8, c="tab:green", label="ground truth")
        ax.scatter(pred[:, 0], pred[:, 1], s=8, c="tab:red", marker="x", label="predicted")
        ax.set_ylim(P0[:, 1].max() + 12, P0[:, 1].min() - 8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"val frame {val_idx[k]}", fontsize=8)
    axes[0].legend(fontsize=6)
    fig.suptitle("Predicted vs ground-truth lip vertices (held-out audio)")
    viz.save_figure(fig, out / "03_pred_vs_gt_points.png")

    # Which audio frames does each lip point attend to?
    with torch.no_grad():
        middle = data.X_t[val_idx[len(val_idx) // 2]][None]
        _, attention = model(middle, data.pts, data.knn, return_attn=True)
    order = np.argsort(P0[:, 1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(
        attention[0].numpy()[order], aspect="auto", cmap="magma", extent=[-ctx, ctx, len(order), 0]
    )
    ax.set_xlabel("audio frame offset (0 = current frame)")
    ax.set_ylabel("lip points sorted top -> bottom")
    ax.set_title("Audio-point cross-attention (fusion module)")
    fig.colorbar(im)
    viz.save_figure(fig, out / "04_fusion_attention.png")


if __name__ == "__main__":
    main()
