"""Step 5: photo fitting. Identity mesh + 1-5 reference photos -> cameras, expressions, colour.

The identity geometry is fixed by the 3D mesh (step 2). The photos supply what the mesh
cannot: per photo the head rotation and translation, 50 FLAME expression parameters and the
jaw pose (from 2D landmarks), and a shared per-vertex colour.

1. Landmark correspondences: MediaPipe FaceMesh -> FLAME surface through a self-calibrated
   embedding (tinypt/flame/landmarks.py), weighted by 1 / spread^2.
2. PnP initialisation (EPnP + Levenberg-Marquardt) on the neutral identity, then joint Adam
   optimisation of rotation, translation, expression and jaw with a robust (Charbonnier)
   reprojection loss and Gaussian priors.
3. Texture: (a) weighted colour average with per-photo exposure gains; (b) inverse
   rendering into a shared albedo and per-photo SH lighting.
4. Evaluation: landmark error, leave-one-photo-out novel-view PSNR, and, for synthetic
   inputs, rotation / translation / lip-motion / albedo error against ground truth.

Outputs: data/identity/photo_fit.npz, outputs/flame_pipeline/05_photo_fit/

Usage:
    python scripts/flame_pipeline/05_fit_photos.py [--photos data/photos/synthetic] [--focal 2.0]
"""

import argparse
import time
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from tinypt import visualization as viz
from tinypt.config import (
    FLAME_OUTPUTS,
    IDENTITY_DIR,
    ROOT,
    ensure_dir,
    load_config,
    seed_everything,
)
from tinypt.flame import landmarks as LM
from tinypt.flame import load_flame, rodrigues
from tinypt.flame.model import N_FLAME_VERTICES, N_HEAD_VERTICES
from tinypt.geometry import mesh as M
from tinypt.io_utils import save_json
from tinypt.metrics import masked_psnr
from tinypt.rendering import rasterizer as R

MAX_PHOTOS = 5
COVERED = 1e-3  # minimum fused weight for a vertex to count as seen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--photos", default="data/photos/synthetic")
    parser.add_argument(
        "--focal", type=float, default=2.0, help="assumed focal length in units of the image size"
    )
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--max_spread_mm", type=float, default=3.0)
    parser.add_argument("--w_expr", type=float, default=2e-2)
    parser.add_argument("--w_jaw", type=float, default=1e-2)
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="recompute the MediaPipe -> FLAME landmark embedding",
    )
    return parser.parse_args()


@dataclass
class Photos:
    imgs: list
    landmarks: list
    names: list
    sizes: np.ndarray  # [P, 2] (H, W)
    focal: np.ndarray
    cxy: np.ndarray
    iod: np.ndarray  # inter-ocular distance in pixels

    def __len__(self):
        return len(self.imgs)


def load_photos(photo_dir, focal_factor):
    files = sorted(p for p in photo_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[
        :MAX_PHOTOS
    ]
    imgs, landmarks, names = [], [], []
    for path in files:
        img = np.array(Image.open(path).convert("RGB"))
        lm = LM.detect(img)
        if lm is None:
            print(f"No face detected in {path.name}; skipped.")
            continue
        imgs.append(img)
        landmarks.append(lm)
        names.append(path.name)
    assert imgs, f"no usable photos in {photo_dir}"
    sizes = np.array([img.shape[:2] for img in imgs])
    iod = np.array([np.linalg.norm(lm[LM.EYE_OUTER[0]] - lm[LM.EYE_OUTER[1]]) for lm in landmarks])
    return Photos(
        imgs,
        landmarks,
        names,
        sizes,
        focal_factor * sizes.max(1).astype(np.float64),
        sizes[:, ::-1] / 2.0,
        iod,
    )


def landmark_embedding(flame, cfg, args):
    """Load (or calibrate) the MediaPipe -> FLAME embedding and keep the stable landmarks."""
    path = ROOT / cfg["flame"]["dir"] / "mp_embedding_selfcalib.npz"
    if args.recalibrate or not path.exists():
        print("calibrating the MediaPipe -> FLAME landmark embedding (~2-3 min, cached)")
        np.savez(path, **LM.calibrate(flame))
    cal = dict(np.load(path))
    selected = LM.select(cal, args.max_spread_mm)
    weights = 1.0 / np.maximum(cal["spread_mm"][selected], 1.0) ** 2
    weights = torch.tensor(weights, dtype=torch.float32)
    flame.lmk = dict(
        face_idx=cal["face_idx"][selected], bary=cal["bary"][selected], mp_idx=selected
    )
    print(
        f"landmarks: {len(selected)} of 468 MediaPipe points kept "
        f"(surface spread < {args.max_spread_mm} mm)"
    )
    return cal, selected, weights / weights.mean()


class LandmarkFitter:
    """Per-photo head pose, expression and jaw from 2D landmarks, identity fixed."""

    def __init__(self, flame, beta, offsets, photos, selected, weights):
        self.flame, self.beta, self.offsets = flame, beta, offsets
        self.P = len(photos)
        self.target = torch.tensor(
            np.array([lm[selected] for lm in photos.landmarks]), dtype=torch.float32
        )
        self.weights = weights
        self.f = torch.tensor(photos.focal, dtype=torch.float32)[:, None, None]
        self.c = torch.tensor(photos.cxy, dtype=torch.float32)[:, None]
        self.iod = torch.tensor(photos.iod, dtype=torch.float32)[:, None, None]
        self.C = torch.tensor(R.C_FLIP, dtype=torch.float32)

    def forward(self, rvec, trans, expr, jaw):
        """Posed meshes [P, n, 3] and their camera-space vertices."""
        V, _ = self.flame(
            self.beta.expand(self.P, -1), expr, self.flame.jaw_pose(jaw), self.offsets
        )
        Rm = self.C @ rodrigues(rvec)
        Xc = torch.einsum("pij,pvj->pvi", Rm, V) + trans[:, None]
        return V, Xc

    def loss(self, rvec, trans, expr, jaw):
        """Robust landmark loss (residuals in % of the inter-ocular distance)."""
        _, Xc = self.forward(rvec, trans, expr, jaw)
        L = self.flame.landmarks(Xc)
        uv = self.f * L[..., :2] / L[..., 2:3] + self.c
        r = ((uv - self.target) / self.iod * 100).pow(2).sum(-1)
        return (torch.sqrt(r + 1.0) * self.weights).mean(), uv

    def pnp_init(self, photos, selected):
        """Rigid pose of the neutral identity per photo (EPnP, refined iteratively)."""
        with torch.no_grad():
            V0 = self.flame(self.beta, offsets=self.offsets)[0]
            L0 = self.flame.landmarks(V0)[0].numpy().astype(np.float64)
        rvecs, trans = [], []
        for i in range(self.P):
            f, (cx, cy) = photos.focal[i], photos.cxy[i]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            pts = photos.landmarks[i][selected]
            _, rc, tc = cv2.solvePnP(L0, pts, K, None, flags=cv2.SOLVEPNP_EPNP)
            _, rc, tc = cv2.solvePnP(
                L0, pts, K, None, rc, tc, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
            )
            # OpenCV gives X_cam = Rc X + t; our convention is X_cam = C R X + t.
            rvecs.append(cv2.Rodrigues(R.C_FLIP @ cv2.Rodrigues(rc)[0])[0].ravel())
            trans.append(tc.ravel())
        return np.array(rvecs), np.array(trans)

    def fit(self, rvec0, trans0, iters, w_expr, w_jaw):
        """Adam: pose only for the first 20% of iterations, then pose, expression and jaw."""
        rvec = torch.tensor(rvec0, dtype=torch.float32, requires_grad=True)
        trans = torch.tensor(trans0, dtype=torch.float32, requires_grad=True)
        expr = torch.zeros(self.P, self.flame.n_expr, requires_grad=True)
        jaw = torch.zeros(self.P, 3, requires_grad=True)
        opt = torch.optim.Adam(
            [
                dict(params=[rvec, jaw], lr=0.01),
                dict(params=[trans], lr=0.002),
                dict(params=[expr], lr=0.05),
            ]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        history = []
        for it in range(iters):
            data_loss, _ = self.loss(rvec, trans, expr, jaw)
            prior = w_expr * expr.pow(2).sum(1).mean() + w_jaw * jaw.pow(2).sum(1).mean() * 100
            opt.zero_grad()
            (data_loss + prior).backward()
            if it < iters // 5:
                expr.grad.zero_()
                jaw.grad.zero_()
            opt.step()
            sched.step()
            history.append(data_loss.item())
            if it % 100 == 0 or it == iters - 1:
                print(
                    f"  it {it:4d}  landmark loss {float(data_loss):.3f}  "
                    f"|expr| {float(expr.norm(dim=1).mean()):.2f}  "
                    f"jaw {float(jaw[:, 0].mean()):.3f} rad"
                )
        return rvec.detach(), trans.detach(), expr.detach(), jaw.detach(), history


class Texture:
    """Per-vertex colours seen in each photo, fused into a texture."""

    def __init__(self, photos, Xc_fit, F):
        self.F, self.photos = F, photos
        samples = [
            R.sample_vertex_colors(
                photos.imgs[i].astype(np.float32) / 255,
                Xc_fit[i],
                F,
                photos.focal[i],
                *photos.cxy[i],
            )
            for i in range(len(photos))
        ]
        self.cols = np.array([s[0] for s in samples])
        self.wts = np.array([s[1] for s in samples])
        self.Ys = np.array([s[2] for s in samples])
        self.rasters = [s[3] for s in samples]
        self.head_face = (F < N_HEAD_VERTICES).all(1)

    def fuse(self, mask, gains=None):
        """Weighted colour average over the photos in `mask`; also returns the weight sum."""
        w = self.wts[mask]
        c = self.cols[mask] * (1 if gains is None else gains[mask][:, None])
        return (c * w[..., None]).sum(0) / np.maximum(w.sum(0), 1e-6)[:, None], w.sum(0)

    def exposure_gains(self, iters=5):
        P = len(self.photos)
        gains = np.ones((P, 3))
        for _ in range(iters):
            ref, _ = self.fuse(np.ones(P, bool), gains)
            gains = np.array([fit_gain(self.cols[i], self.wts[i], ref) for i in range(P)])
            gains /= np.exp(np.log(gains).mean(0))  # fix the global scale
        return gains

    def render_colors(self, i, colors, coverage):
        """Render per-vertex colours into photo i; mask = face pixels with coverage."""
        fid, bary, _ = self.rasters[i]
        img = R.interpolate(colors, self.F, fid, bary)
        mask = fid >= 0
        mask[mask] = self.head_face[fid[mask]] & (coverage[self.F[fid[mask]]] > COVERED).all(1)
        return img, mask

    def psnr(self, img, i, mask):
        return masked_psnr(np.clip(img, 0, 1), self.photos.imgs[i] / 255.0, mask)


def fit_gain(c_photo, w, c_ref):
    """Per-channel exposure / white-balance gain mapping c_photo onto c_ref (least squares)."""
    return (w[:, None] * c_photo * c_ref).sum(0) / np.maximum(
        (w[:, None] * c_photo**2).sum(0), 1e-9
    )


def leave_one_out(tex, gains):
    """Texture from the other photos rendered into each held-out camera. The held-out photo
    only gets its exposure (3 numbers) or SH lighting (27 numbers) fitted."""
    P = len(tex.photos)
    results = []
    if P < 2:
        return results
    for i in range(P):
        others = np.arange(P) != i
        colors, coverage = tex.fuse(others, gains)
        img_baked, mask = tex.render_colors(i, colors, coverage)
        target = tex.photos.imgs[i][mask].reshape(-1, 3) / 255.0
        g_i = fit_gain(img_baked[mask].reshape(-1, 3), np.ones(mask.sum()), target)
        img_baked = np.clip(img_baked * g_i, 0, 1)
        albedo_i, _, _ = R.albedo_and_lighting(tex.cols[others], tex.Ys[others], tex.wts[others])
        L_i = R.fit_sh(tex.cols[i], albedo_i, tex.Ys[i], tex.wts[i] * (coverage > COVERED))
        img_sh, _ = tex.render_colors(i, np.clip(albedo_i * (tex.Ys[i] @ L_i), 0, 1), coverage)
        n_face = max((tex.rasters[i][0] >= 0).sum(), 1)
        results.append(
            dict(
                photo=tex.photos.names[i],
                psnr=tex.psnr(img_sh, i, mask),
                psnr_baked=tex.psnr(img_baked, i, mask),
                covered_face_pixels=float(mask.sum() / n_face),
                img=img_sh,
                img_b=img_baked,
                mask=mask,
            )
        )
    return results


def ground_truth_metrics(
    gt, flame, photos, beta, offsets, rvec, trans, V_fit, albedo, baked, coverage, focal_factor
):
    """Errors against the synthetic ground truth of step 1."""
    gidx = [list(gt["files"]).index(name) for name in photos.names]
    P = len(photos)
    gt_beta = torch.tensor(gt["beta"], dtype=torch.float32)[None]
    # Step 2 placed the identity in its own frame: V_mesh = s Ru V_id + tu (Umeyama to the
    # FLAME template). Express the ground-truth cameras in that frame before comparing:
    # X_cam = C R_gt (s Ru V_id + tu) + t_gt.
    with torch.no_grad():
        V_gt_neutral = flame(gt_beta)[0].numpy()
    su, Ru, tu = M.umeyama(
        flame.v_template.numpy().astype(np.float64), V_gt_neutral[0].astype(np.float64)
    )
    rot_err, trans_err = [], []
    for i, j in enumerate(gidx):
        R_gt = R.rotmat(gt["rvec"][j])
        rot_err.append(M.rotation_angle_deg(R.rotmat(rvec[i].numpy()).T @ R_gt @ Ru))
        trans_err.append(
            1000 * np.linalg.norm(trans[i].numpy() - (R.C_FLIP @ R_gt @ tu + gt["t"][j]))
        )

    lips = flame.region("lips")
    with torch.no_grad():
        jaw_gt = torch.tensor(gt["jaw"][gidx], dtype=torch.float32)
        V_gt = flame(
            gt_beta.expand(P, -1),
            torch.tensor(gt["expr"][gidx], dtype=torch.float32),
            flame.jaw_pose(jaw_gt),
        )[0].numpy()
        V_neutral = flame(beta, offsets=offsets)[0].numpy()
    # Expression only: displacement from each identity's own neutral mesh.
    d_fit, d_gt = V_fit - V_neutral, V_gt - V_gt_neutral
    focal_gt = float(gt["f"][0]) / gt["size"]
    metrics = dict(
        rotation_error_deg=dict(per_photo=rot_err, mean=float(np.mean(rot_err))),
        step2_alignment=dict(
            scale=float(su), rot_deg=M.rotation_angle_deg(Ru), t_mm=(1000 * tu).tolist()
        ),
        translation_error_mm=dict(
            per_photo=trans_err,
            mean=float(np.mean(trans_err)),
            note=f"photos rendered with focal {focal_gt:.1f}x size, fitter assumes "
            f"{focal_factor}x: depth absorbs the focal mismatch",
        ),
        lip_motion_error_mm=float(
            1000 * np.linalg.norm(d_fit[:, lips] - d_gt[:, lips], axis=-1).mean()
        ),
        lip_motion_error_mm_neutral_baseline=float(
            1000 * np.linalg.norm(d_gt[:, lips], axis=-1).mean()
        ),
        jaw_open_gt_rad=gt["jaw"][gidx, 0].tolist(),
    )
    covered = (coverage > COVERED) & (np.arange(N_FLAME_VERTICES) < N_HEAD_VERTICES)
    for key, estimate in [("albedo_sh", albedo), ("baked", baked)]:  # up to an RGB gain
        g = fit_gain(estimate[covered], np.ones(covered.sum()), gt["albedo"][covered])
        e = np.clip(estimate[covered] * g, 0, 1) - gt["albedo"][covered]
        metrics[f"albedo_psnr_{key}"] = float(10 * np.log10(1 / (e**2).mean()))
    return metrics


def plot_landmark_fit(out, flame, photos, target, uv_fit, err_px, V_fit, rvec, trans, jaw):
    P, F = len(photos), flame.faces
    fig, ax = plt.subplots(2, P, figsize=(3.6 * P, 7.4), squeeze=False)
    for i in range(P):
        H, W = photos.sizes[i]
        img = photos.imgs[i]
        ax[0, i].imshow(img)
        ax[0, i].scatter(*target[i].numpy().T, s=4, c="lime", label="MediaPipe")
        ax[0, i].scatter(*uv_fit[i].numpy().T, s=4, c="red", label="fitted mesh")
        err = err_px[i].mean()
        ax[0, i].set_title(
            f"{photos.names[i]}\nlandmark err {err:.1f}px ({100 * err / photos.iod[i]:.1f}% IOD)",
            fontsize=8,
        )
        shaded, *_ = R.render(
            V_fit[i],
            F,
            rvec[i].numpy(),
            trans[i].numpy(),
            photos.focal[i],
            H,
            W,
            np.full((N_FLAME_VERTICES, 3), 0.8),
            bg=img / 255.0,
        )
        ax[1, i].imshow((0.45 * img + 0.55 * shaded).astype(np.uint8))
        ax[1, i].set_title(f"fitted mesh overlay, jaw {float(jaw[i, 0]):.3f} rad", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    ax[0, 0].legend(loc="lower left", fontsize=7, markerscale=3)
    fig.suptitle(
        "Step 5: identity mesh fitted to reference photos "
        "(geometry fixed, pose + expression + jaw fitted)"
    )
    viz.save_figure(fig, out / "01_landmark_fit.png", dpi=110)


def plot_texture(out, flame, beta, offsets, albedo, baked, coverage, P):
    F, S = flame.faces, 320
    with torch.no_grad():
        V_neutral = flame(beta, offsets=offsets)[0][0].numpy().astype(np.float64)
    yaws = [-60, -30, 0, 30, 60]
    coverage_rgb = plt.cm.magma(np.clip(coverage / max(coverage.max(), 1e-6), 0, 1))[:, :3]
    rows = [
        (np.clip(albedo, 0, 1), "albedo (SH de-lit)"),
        (baked, "baked colour average"),
        (coverage_rgb, "coverage (sum of weights)"),
    ]
    fig, ax = plt.subplots(3, len(yaws), figsize=(3 * len(yaws), 9.3))
    for k, yaw in enumerate(yaws):
        for row, (colors, title) in enumerate(rows):
            Xc = R.to_camera(V_neutral, np.radians([0, yaw, 0]), [0, 0.02, 0.7])
            uv, z = R.project(Xc, 2.2 * S, S / 2, S / 2)
            fid, bary, _ = R.rasterize(uv, z, F, S, S)
            img = R.interpolate(colors, F, fid, bary)
            img[fid < 0] = 1.0
            ax[row, k].imshow(np.clip(img, 0, 1))
            ax[row, k].axis("off")
            ax[row, k].set_title(f"{title}, yaw {yaw}", fontsize=8)
    fig.suptitle(f"Per-vertex colour from {P} photos, neutral pose, novel views (shown unlit)")
    viz.save_figure(fig, out / "02_fused_texture.png", dpi=110)


def plot_leave_one_out(out, photos, loo):
    P = len(photos)
    fig, ax = plt.subplots(4, P, figsize=(3.2 * P, 12.8), squeeze=False)
    for i, r in enumerate(loo):
        img = photos.imgs[i] / 255.0
        m = r["mask"][..., None]
        panels = [
            (img, f"held-out {r['photo']}"),
            (np.where(m, r["img_b"], img * 0.35), f"baked colours: {r['psnr_baked']:.1f} dB"),
            (np.where(m, r["img"], img * 0.35), f"albedo x SH light: {r['psnr']:.1f} dB"),
            (np.where(m, np.abs(r["img"] - img) * 3, 0), "|difference| x3 (albedo x SH)"),
        ]
        for row, (x, title) in enumerate(panels):
            ax[row, i].imshow(np.clip(x, 0, 1))
            ax[row, i].axis("off")
            ax[row, i].set_title(title, fontsize=8)
    fig.suptitle(
        "Leave-one-photo-out: texture from the remaining photos rendered into the held-out camera"
    )
    viz.save_figure(fig, out / "03_leave_one_out.png", dpi=110)


def plot_embedding(out, flame, cal, selected):
    V, F = flame.v_template.numpy(), flame.faces
    P = (V[F[cal["face_idx"]]] * cal["bary"][..., None]).sum(1)[:468]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(V[:N_HEAD_VERTICES:3, 0], V[:N_HEAD_VERTICES:3, 1], s=1, c="0.85")
    sc = ax.scatter(
        P[:, 0], P[:, 1], c=np.minimum(cal["spread_mm"][:468], 6), s=12, cmap="RdYlGn_r"
    )
    ax.scatter(
        P[selected, 0], P[selected, 1], s=40, facecolors="none", edgecolors="k", linewidths=0.4
    )
    plt.colorbar(sc, label="surface spread across views (mm)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.09, 0.09)
    ax.set_ylim(-0.085, 0.07)
    ax.set_title(
        f"Self-calibrated MediaPipe -> FLAME embedding ({int(cal['n_views'])} renders)\n"
        f"circled = used ({len(selected)}); red contour points slide over the surface "
        "and are rejected",
        fontsize=9,
    )
    viz.save_figure(fig, out / "04_landmark_embedding.png", dpi=110)


def plot_loss(out, history, iters):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(history)
    ax.axvline(iters // 5, ls="--", c="gray")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("robust landmark loss")
    ax.text(iters // 5, max(history), " + expression, jaw", fontsize=8, va="top")
    viz.save_figure(fig, out / "05_fit_loss.png", dpi=110)


def main():
    args = parse_args()
    t_start = time.time()
    seed_everything(0)
    cfg = load_config()
    out = ensure_dir(FLAME_OUTPUTS / "05_photo_fit")
    flame = load_flame(cfg)
    F = flame.faces
    identity = np.load(IDENTITY_DIR / "identity.npz")
    beta = torch.tensor(identity["beta"], dtype=torch.float32)[None]
    offsets = torch.tensor(identity["offsets"], dtype=torch.float32)

    # 1-2. Landmark embedding, photos and detections.
    cal, selected, lmk_weights = landmark_embedding(flame, cfg, args)
    photo_dir = ROOT / args.photos
    photos = load_photos(photo_dir, args.focal)
    P = len(photos)
    print(f"{P} photos: {photos.names}")

    # 3-4. PnP initialisation and joint optimisation.
    fitter = LandmarkFitter(flame, beta, offsets, photos, selected, lmk_weights)
    rvec0, trans0 = fitter.pnp_init(photos, selected)
    rvec, trans, expr, jaw, history = fitter.fit(rvec0, trans0, args.iters, args.w_expr, args.w_jaw)
    with torch.no_grad():
        V_fit, Xc_fit = fitter.forward(rvec, trans, expr, jaw)
        _, uv_fit = fitter.loss(rvec, trans, expr, jaw)
        _, uv_init = fitter.loss(
            torch.tensor(rvec0, dtype=torch.float32),
            torch.tensor(trans0, dtype=torch.float32),
            torch.zeros_like(expr),
            torch.zeros_like(jaw),
        )
    V_fit, Xc_fit = V_fit.numpy().astype(np.float64), Xc_fit.numpy().astype(np.float64)
    err_px = np.linalg.norm(uv_fit.numpy() - fitter.target.numpy(), axis=-1)  # [P, L]
    err_init_px = np.linalg.norm(uv_init.numpy() - fitter.target.numpy(), axis=-1)
    is_lip = np.isin(selected, LM.mediapipe_lip_ids())

    # 5. Texture: baked average with exposure gains, and albedo x SH lighting.
    tex = Texture(photos, Xc_fit, F)
    gains = tex.exposure_gains()
    baked, coverage = tex.fuse(np.ones(P, bool), gains)
    baked = np.clip(baked, 0, 1)
    albedo, lighting, _ = R.albedo_and_lighting(tex.cols, tex.Ys, tex.wts)
    loo = leave_one_out(tex, gains)
    self_psnr = []
    for i in range(P):
        img, mask = tex.render_colors(
            i, np.clip(albedo * (tex.Ys[i] @ lighting[i]), 0, 1), coverage
        )
        self_psnr.append(tex.psnr(img, i, mask))

    # 6. Metrics.
    loo_sh = [r["psnr"] for r in loo]
    loo_baked = [r["psnr_baked"] for r in loo]
    metrics = dict(
        photos=photos.names,
        n_landmarks=int(len(selected)),
        focal_assumed=args.focal,
        landmark_error_px=dict(
            per_photo=err_px.mean(1).tolist(),
            mean=float(err_px.mean()),
            pnp_init_mean=float(err_init_px.mean()),
        ),
        landmark_nme_percent_iod=float((err_px / photos.iod[:, None]).mean() * 100),
        lip_landmark_error_px=float(err_px[:, is_lip].mean()),
        lip_landmark_error_px_pnp_init=float(err_init_px[:, is_lip].mean()),
        jaw_open_rad=jaw[:, 0].tolist(),
        expr_norm=expr.norm(dim=1).tolist(),
        texture=dict(
            vertices_covered=float((coverage > COVERED)[:N_HEAD_VERTICES].mean()),
            self_view_psnr=self_psnr,
        ),
        leave_one_out_psnr=dict(
            albedo_sh=loo_sh,
            albedo_sh_mean=float(np.mean(loo_sh)) if loo else None,
            baked=loo_baked,
            baked_mean=float(np.mean(loo_baked)) if loo else None,
            note="texture from the OTHER photos rendered into the held-out camera (face pixels "
            "with coverage). albedo_sh: albedo x SH lighting, 27 lighting params fitted "
            "on the held-out photo; baked: weighted colour average, 3 exposure gains "
            "fitted on the held-out photo",
        ),
    )
    gt_path = photo_dir / "ground_truth.npz"
    if gt_path.exists():
        metrics["ground_truth"] = ground_truth_metrics(
            np.load(gt_path),
            flame,
            photos,
            beta,
            offsets,
            rvec,
            trans,
            V_fit,
            albedo,
            baked,
            coverage,
            args.focal,
        )
    metrics["seconds"] = time.time() - t_start
    save_json(metrics, out / "metrics.json")
    np.savez(
        IDENTITY_DIR / "photo_fit.npz",
        files=np.array(photos.names),
        sizes=photos.sizes,
        focal=photos.focal,
        cxy=photos.cxy,
        rvec=rvec.numpy(),
        t=trans.numpy(),
        expr=expr.numpy(),
        jaw=jaw.numpy(),
        landmarks_2d=np.array(photos.landmarks),
        lmk_sel=selected,
        albedo=albedo.astype(np.float32),
        sh_lighting=lighting,
        vertex_color_baked=baked.astype(np.float32),
        photo_gains=gains,
        vertex_weight=coverage.astype(np.float32),
        photo_vertex_weight=tex.wts.astype(np.float32),
        photos_dir=str(args.photos),
    )
    print({k: v for k, v in metrics.items() if k != "photos"})

    # 7. Figures.
    plot_landmark_fit(out, flame, photos, fitter.target, uv_fit, err_px, V_fit, rvec, trans, jaw)
    plot_texture(out, flame, beta, offsets, albedo, baked, coverage, P)
    if loo:
        plot_leave_one_out(out, photos, loo)
    plot_embedding(out, flame, cal, selected)
    plot_loss(out, history, args.iters)
    print(
        f"saved {IDENTITY_DIR / 'photo_fit.npz'} and figures in {out}  "
        f"({time.time() - t_start:.0f}s)"
    )


if __name__ == "__main__":
    main()
