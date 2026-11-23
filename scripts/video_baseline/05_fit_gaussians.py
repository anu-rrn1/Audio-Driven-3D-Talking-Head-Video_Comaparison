"""Step 5: mesh-rigged 3D Gaussian splatting fitted to the video frames.

Gaussians are initialised on the canonical mesh with colours from one reference frame and
optimised photometrically (L1 + SSIM) against training frames, each posed with that frame's
tracked mesh. Evaluated on held-out frames.

Outputs: checkpoints/gaussians.pt, outputs/video_baseline/05_gaussians/
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from tinypt import visualization as viz
from tinypt.config import (
    BASELINE_OUTPUTS,
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    ensure_dir,
    load_config,
    seed_everything,
)
from tinypt.io_utils import save_json
from tinypt.metrics import psnr, ssim_torch
from tinypt.rendering.splatting import render
from tinypt.video_baseline.rigged_gaussians import RiggedGaussians


def main():
    cfg = load_config()
    seed_everything(0)
    gauss_cfg = cfg["gaussians"]
    out = ensure_dir(BASELINE_OUTPUTS / "05_gaussians")
    frames = np.load(PROCESSED_DIR / "frames.npy")
    lm = np.load(PROCESSED_DIR / "landmarks.npy").astype(np.float32)
    mesh = np.load(PROCESSED_DIR / "mesh_data.npz")
    V_can, F, region_tri = mesh["V_can"], mesh["faces"], mesh["region_tri"]
    openness = mesh["open_series"] / float(mesh["open_scale"])
    T, H, W = len(frames), frames.shape[1], frames.shape[2]
    n_train = int(T * (1 - cfg["data"]["val_ratio"]))
    plate = torch.tensor(np.median(frames, 0) / 255.0, dtype=torch.float32)  # static background
    reference = frames[int(mesh["ref_ids"][0])] / 255.0

    def reference_color(p):
        x = np.clip(np.round(p[:, 0] - 0.5).astype(int), 0, W - 1)
        y = np.clip(np.round(p[:, 1] - 0.5).astype(int), 0, H - 1)
        return reference[y, x]

    model = RiggedGaussians(
        V_can, F, gauss_cfg["n_gauss"], region_tri, reference_color, gauss_cfg["mouth_frac"]
    )
    opt = torch.optim.Adam(
        [
            dict(params=[model.c0], lr=0.05),
            dict(params=[model.c1], lr=0.03),
            dict(params=[model.opacity_logit], lr=0.03),
            dict(params=[model.log_scale], lr=0.01),
            dict(params=[model.offset], lr=0.05),
        ]
    )

    def render_frame(V, o):
        pos, cov, color, opacity = model.params(torch.tensor(V, dtype=torch.float32), float(o))
        return render(pos, cov, color, opacity, H, W, plate, gauss_cfg["tile"])

    def evaluate(ids):
        psnrs, ssims = [], []
        with torch.no_grad():
            for t in ids:
                img = render_frame(lm[t], openness[t]).numpy()
                gt = frames[t] / 255.0
                psnrs.append(psnr(img, gt))
                ssims.append(structural_similarity(img, gt, channel_axis=2, data_range=1.0))
        return float(np.mean(psnrs)), float(np.mean(ssims))

    val_ids = np.linspace(n_train, T - 1, min(30, T - n_train)).astype(int)
    t_vis = int(np.argmax(openness[:n_train]))  # training frame with the widest mouth
    snapshot_iters = sorted({0, 20, 60, 150, gauss_cfg["iters"] - 1})
    snapshots = {}
    init_psnr, init_ssim = evaluate(val_ids)
    print(f"before optimisation: val PSNR {init_psnr:.2f} SSIM {init_ssim:.3f}")
    losses, t0 = [], time.time()
    for it in range(gauss_cfg["iters"]):
        if it in snapshot_iters:
            with torch.no_grad():
                snapshots[it] = render_frame(lm[t_vis], openness[t_vis]).numpy()
        t = int(np.random.randint(0, n_train))
        img = render_frame(lm[t], openness[t])
        gt = torch.tensor(frames[t] / 255.0, dtype=torch.float32)
        loss = (
            0.8 * (img - gt).abs().mean()
            + 0.2 * (1 - ssim_torch(img, gt))
            + 1e-3 * model.offset.pow(2).mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if it % 25 == 0:
            print(f"it {it:4d} loss {np.mean(losses[-25:]):.4f}  {time.time() - t0:.0f}s")

    final_psnr, final_ssim = evaluate(val_ids)
    metrics = dict(
        val_psnr_init=init_psnr,
        val_ssim_init=init_ssim,
        val_psnr_fitted=final_psnr,
        val_ssim_fitted=final_ssim,
        n_gaussians=gauss_cfg["n_gauss"],
        iterations=gauss_cfg["iters"],
        note="held-out frames rendered with their tracked GT mesh",
    )
    save_json(metrics, out / "metrics.json")
    print(metrics)
    torch.save(
        dict(
            state=model.state_dict(),
            n_gauss=gauss_cfg["n_gauss"],
            mouth_frac=gauss_cfg["mouth_frac"],
            plate=plate,
        ),
        CHECKPOINT_DIR / "gaussians.pt",
    )

    # Figures.
    progress = [(frames[t_vis], "ground truth")] + [
        (np.clip(snapshots[i] * 255, 0, 255).astype(np.uint8), f"iteration {i}")
        for i in sorted(snapshots)
    ]
    viz.image_grid(
        [img for img, _ in progress],
        [title for _, title in progress],
        out / "01_optimisation_progress.png",
        ncols=len(progress),
        suptitle="Gaussian splatting optimisation (training frame)",
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.convolve(losses, np.ones(10) / 10, mode="valid"))
    ax.set_xlabel("iteration")
    ax.set_ylabel("L1 + SSIM loss")
    ax.grid(alpha=0.3)
    viz.save_figure(fig, out / "02_loss.png")

    shown = np.linspace(n_train, T - 1, 6).astype(int)
    gts, renders, diffs = [], [], []
    with torch.no_grad():
        for t in shown:
            img = render_frame(lm[t], openness[t]).numpy()
            gts.append(frames[t])
            renders.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
            diffs.append(
                (np.clip(np.abs(img - frames[t] / 255.0) * 3, 0, 1) * 255).astype(np.uint8)
            )
    titles = [f"GT {t}" for t in shown] + [f"render {t}" for t in shown] + ["|diff| x3"] * 6
    viz.image_grid(
        gts + renders + diffs,
        titles,
        out / "03_heldout_gt_vs_render.png",
        ncols=6,
        suptitle=f"Held-out frames: PSNR {final_psnr:.1f} dB, SSIM {final_ssim:.3f}",
    )
    with torch.no_grad():
        pos, _, color, opacity = model.params(torch.tensor(V_can, dtype=torch.float32), 0.0)
    viz.gaussian_cloud_views(
        pos.numpy(), color.numpy(), opacity.numpy(), out / "04_canonical_gaussians_3d.png"
    )


if __name__ == "__main__":
    main()
