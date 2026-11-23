"""Step 6: few-shot Gaussian avatar. Identity mesh + photo fit -> relightable, rigged 3DGS head.

Initialisation: Gaussians bound to the FLAME + mouth-proxy triangles, denser where the
photos see the head; albedo from inverse rendering (albedo x SH light) solved on the
training photos only.

Training: photometric loss (L1 + SSIM inside the head region; foreground from MediaPipe
selfie segmentation) on the training photos. Optimised: Gaussian albedo, offset, scale and
opacity, the per-photo SH lighting, and small per-photo corrections of the step-5 head pose,
expression and jaw.

Evaluation: a held-out photo rendered from its step-5 landmark camera, with only its
lighting (27 SH numbers) fitted. Baseline: the step-5 mesh texture (albedo x SH) under the
same protocol and mask.

Outputs: data/identity/gaussian_avatar.pt (animatable with any FLAME expression / jaw),
outputs/flame_pipeline/06_gaussian_avatar/

Usage:
    python scripts/flame_pipeline/06_gaussian_avatar.py [--holdout photo_03.png | none]
        [--iters 400] [--res 256] [--n_gauss 15000]
"""

import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from tinypt import visualization as viz
from tinypt.config import (
    FLAME_OUTPUTS,
    IDENTITY_DIR,
    ROOT,
    ensure_dir,
    load_config,
    seed_everything,
)
from tinypt.flame import load_flame
from tinypt.flame.model import N_FLAME_VERTICES, N_HEAD_VERTICES
from tinypt.geometry import mesh as M
from tinypt.io_utils import save_json
from tinypt.metrics import ssim_torch
from tinypt.rendering import rasterizer as R
from tinypt.rendering.gaussian_avatar import GaussianAvatar

CAVITY_ALBEDO = [0.22, 0.07, 0.07]  # dark red
TEETH_ALBEDO = [0.88, 0.85, 0.78]  # off-white
LOG_EVERY = 25


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--holdout",
        default="photo_03.png",
        help="photo kept out of training for evaluation, or 'none'",
    )
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--n_gauss", type=int, default=15000)
    parser.add_argument("--no_pose_refine", action="store_true")
    parser.add_argument("--out", default="06_gaussian_avatar")
    return parser.parse_args()


class Logger:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        print(f"[{time.time() - self.t0:6.0f}s] {msg}", flush=True)


def load_photos(photo_fit, res):
    """Photos resized to the training resolution, foreground masks and scaled cameras."""
    import mediapipe as mp

    names = [str(x) for x in photo_fit["files"]]
    photo_dir = ROOT / str(photo_fit["photos_dir"])
    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    imgs, foregrounds, cams = [], [], []
    for i, name in enumerate(names):
        full = np.array(Image.open(photo_dir / name).convert("RGB"))
        H0, W0 = full.shape[:2]
        scale = res / max(H0, W0)
        H, W = int(round(H0 * scale)), int(round(W0 * scale))
        imgs.append(cv2.resize(full, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
        mask = segmenter.process(full).segmentation_mask.astype(np.float32)
        foregrounds.append(cv2.resize(mask, (W, H)))
        cams.append(
            dict(
                f=float(photo_fit["focal"][i] * scale),
                cx=float(photo_fit["cxy"][i][0] * scale),
                cy=float(photo_fit["cxy"][i][1] * scale),
                H=H,
                W=W,
            )
        )
    return names, photo_dir, imgs, foregrounds, cams


class Scene:
    """FLAME identity with mouth proxies, posed per photo from the step-5 fit plus optional
    learned corrections `deltas` (rot, t, expr, jaw)."""

    def __init__(self, flame, beta, offsets, photo_fit, cams):
        self.flame, self.beta, self.offsets = flame, beta, offsets
        self.photo_fit, self.cams = photo_fit, cams
        self.deltas = None

    def posed(self, i, use_deltas=True):
        pf, d = self.photo_fit, self.deltas if use_deltas else None
        expr = torch.tensor(pf["expr"][i], dtype=torch.float32)[None]
        jaw = torch.tensor(pf["jaw"][i], dtype=torch.float32)[None]
        if d is not None:
            expr, jaw = expr + d["expr"][i : i + 1], jaw + d["jaw"][i : i + 1]
        return self.flame(self.beta, expr, self.flame.jaw_pose(jaw), self.offsets)[0][0]

    def camera(self, i, use_deltas=True):
        pf, d = self.photo_fit, self.deltas if use_deltas else None
        rvec = torch.tensor(pf["rvec"][i], dtype=torch.float32)
        t = torch.tensor(pf["t"][i], dtype=torch.float32)
        if d is not None:
            rvec, t = rvec + d["rot"][i], t + d["t"][i]
        c = self.cams[i]
        return dict(rvec=rvec, t=t, f=c["f"], cx=c["cx"], cy=c["cy"])


def initial_appearance(scene, imgs, train, hold, part, F0, n_faces_total):
    """Albedo and SH lighting from the training photos at the step-5 fit, the face sampling
    weights, and the vertex samples of every photo."""
    pf, cams = scene.photo_fit, scene.cams
    samples = []
    with torch.no_grad():
        for i in range(len(imgs)):
            V = scene.posed(i).numpy().astype(np.float64)[:N_FLAME_VERTICES]
            Xc = R.to_camera(V, pf["rvec"][i], pf["t"][i])
            samples.append(
                R.sample_vertex_colors(imgs[i], Xc, F0, cams[i]["f"], cams[i]["cx"], cams[i]["cy"])
            )
    cols = np.array([s[0] for s in samples])
    wts = np.array([s[1] for s in samples])
    Ys = np.array([s[2] for s in samples])
    albedo_v, L_train, _ = R.albedo_and_lighting(cols[train], Ys[train], wts[train])
    coverage = wts[train].sum(0)
    lighting = np.zeros((len(imgs), 9, 3))
    lighting[train] = L_train
    for i in hold:  # held-out lighting: 27 numbers, least squares on its own colours
        lighting[i] = R.fit_sh(cols[i], albedo_v, Ys[i], wts[i] * (coverage > 1e-3))

    albedo_proxy = np.where((part == "cavity")[:, None], CAVITY_ALBEDO, TEETH_ALBEDO)
    fallback = np.median(albedo_v[coverage > 1e-3], 0)
    albedo_head = np.where(coverage[:, None] > 1e-3, albedo_v, fallback)
    albedo_init = np.clip(np.concatenate([albedo_head, albedo_proxy]), 0.02, 0.98)
    seen = np.clip(coverage / max(coverage.max(), 1e-6), 0, 1)[F0].max(1)
    face_weight = np.concatenate([0.05 + seen, np.ones(n_faces_total - len(F0))])
    return dict(
        samples=samples,
        Ys=Ys,
        albedo_v=albedo_v,
        coverage=coverage,
        lighting=lighting,
        albedo_init=albedo_init,
        face_weight=face_weight,
    )


def freeze_mouth_proxies(avatar, n_base_faces):
    """Mouth-proxy Gaussians keep their initial albedo and are opaque. The photos (mouth
    nearly closed) never show the cavity or teeth, so fitting them would only learn noise
    that becomes visible when the mouth opens."""
    is_proxy = avatar.tri >= n_base_faces
    avatar.opacity_logit.data[is_proxy] = 4.0
    keep = (~is_proxy).float()
    for param in [avatar.albedo_logit, avatar.offset, avatar.log_scale, avatar.opacity_logit]:
        param.register_hook(lambda g: g * keep.view(-1, *[1] * (g.dim() - 1)))
    return is_proxy


def loss_regions(scene, imgs, foregrounds):
    """Dilated silhouette of the fitted mesh per photo, and foreground-only targets."""
    pf, cams, F = scene.photo_fit, scene.cams, scene.flame.faces
    regions, targets = [], []
    with torch.no_grad():
        for i in range(len(imgs)):
            V = scene.posed(i).numpy().astype(np.float64)
            c = cams[i]
            uv, z = R.project(R.to_camera(V, pf["rvec"][i], pf["t"][i]), c["f"], c["cx"], c["cy"])
            fid, _, _ = R.rasterize(uv, z, F, c["H"], c["W"], cull=False)
            silhouette = (fid >= 0).astype(np.uint8)
            regions.append(
                torch.tensor(cv2.dilate(silhouette, np.ones((9, 9), np.uint8)), dtype=torch.float32)
            )
            targets.append(
                torch.tensor(imgs[i] * (foregrounds[i] > 0.5)[..., None], dtype=torch.float32)
            )
    fg = [torch.tensor((f > 0.5).astype(np.float32))[..., None] for f in foregrounds]
    return regions, targets, fg


def psnr_ssim(a, b, mask):
    """PSNR on the masked pixels and SSIM of images that differ only inside the mask."""
    a_masked = np.where(mask[..., None], a, b)
    return (
        float(10 * np.log10(1 / ((a[mask] - b[mask]) ** 2).mean())),
        float(structural_similarity(a_masked, b, channel_axis=2, data_range=1.0)),
    )


def main():
    args = parse_args()
    log = Logger()
    seed_everything(0)
    cfg = load_config()
    out = ensure_dir(FLAME_OUTPUTS / args.out)
    flame = load_flame(cfg)
    identity = np.load(IDENTITY_DIR / "identity.npz")
    photo_fit = np.load(IDENTITY_DIR / "photo_fit.npz")
    F0 = flame.faces.copy()
    n_proxy = len(identity["proxy_P"])
    flame.add_proxy(identity["proxy_P"], identity["proxy_W"], identity["proxy_F"])
    F = flame.faces
    beta = torch.tensor(identity["beta"], dtype=torch.float32)[None]
    offsets = torch.cat(
        [torch.tensor(identity["offsets"], dtype=torch.float32), torch.zeros(n_proxy, 3)]
    )
    head_face = (F0 < N_HEAD_VERTICES).all(1)

    names, photo_dir, imgs, foregrounds, cams = load_photos(photo_fit, args.res)
    P = len(names)
    hold = [i for i, name in enumerate(names) if name == args.holdout]
    train = [i for i in range(P) if i not in hold]
    assert train, "need at least one training photo"
    log(
        f"photos {names}; training on {[names[i] for i in train]}, "
        f"held out {[names[i] for i in hold]}"
    )

    scene = Scene(flame, beta, offsets, photo_fit, cams)
    init = initial_appearance(scene, imgs, train, hold, identity["proxy_part"], F0, len(F))
    coverage = init["coverage"]
    with torch.no_grad():
        V_rest = flame(beta, offsets=offsets)[0][0].numpy().astype(np.float64)
    avatar = GaussianAvatar(V_rest, F, init["face_weight"], init["albedo_init"], args.n_gauss)
    log(
        f"{args.n_gauss} Gaussians on {len(F)} triangles ({len(F) - len(F0)} mouth-proxy triangles)"
    )
    is_proxy = freeze_mouth_proxies(avatar, len(F0))
    # Albedo prior: Gaussians on faces the training photos barely see stay close to the
    # inverse-rendered albedo.
    seen_weight = torch.tensor(np.clip(init["face_weight"] - 0.05, 0, 1), dtype=torch.float32)
    seen_weight = seen_weight[avatar.tri] * (~is_proxy).float()
    albedo0 = torch.sigmoid(avatar.albedo_logit.detach().clone())

    regions, targets, fg = loss_regions(scene, imgs, foregrounds)

    # Optimisation.
    deltas = dict(
        rot=torch.zeros(P, 3, requires_grad=True),
        t=torch.zeros(P, 3, requires_grad=True),
        expr=torch.zeros(P, flame.n_expr, requires_grad=True),
        jaw=torch.zeros(P, 3, requires_grad=True),
    )
    scene.deltas = deltas
    lighting = torch.tensor(init["lighting"], dtype=torch.float32, requires_grad=True)
    groups = [
        dict(params=[avatar.albedo_logit], lr=0.02),
        dict(params=[avatar.offset], lr=2e-4),
        dict(params=[avatar.log_scale], lr=0.01),
        dict(params=[avatar.opacity_logit], lr=0.05),
        dict(params=[lighting], lr=0.003),
    ]
    if not args.no_pose_refine:
        groups += [
            dict(params=[deltas["rot"], deltas["jaw"]], lr=1e-3),
            dict(params=[deltas["t"]], lr=1e-4),
            dict(params=[deltas["expr"]], lr=0.01),
        ]
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=0.0)

    def render_photo(i, use_deltas=True, L=None, bg=None):
        c = cams[i]
        bg = None if bg is None else bg.expand(c["H"], c["W"], 3)
        return avatar.render(
            scene.posed(i, use_deltas),
            scene.camera(i, use_deltas),
            lighting[i] if L is None else L,
            c["H"],
            c["W"],
            bg,
        )

    def photo_loss(img, i, bg):
        """L1 + SSIM inside the head region. The target is the photo foreground composited
        on the random background colour bg, so Gaussians outside the person must become
        transparent rather than taking the background colour."""
        w = regions[i][..., None]
        target = targets[i] + (1 - fg[i]) * bg
        l1 = ((img - target).abs() * w).sum() / (w.sum() * 3)
        return 0.8 * l1 + 0.2 * (1 - ssim_torch(img * w, target * w))

    snapshot_photo = train[0]
    snapshots = {}
    with torch.no_grad():
        snapshots[0] = render_photo(snapshot_photo).numpy()
    history, rng = [], np.random.default_rng(0)
    for it in range(args.iters):
        i = int(rng.choice(train))
        bg = torch.tensor(rng.random(3), dtype=torch.float32)
        img = render_photo(i, bg=bg)
        loss = photo_loss(img, i, bg)
        reg = 0.01 * (avatar.offset / 0.003).pow(2).mean()
        albedo_drift = (torch.sigmoid(avatar.albedo_logit) - albedo0) ** 2
        reg = reg + 0.5 * ((1 - seen_weight)[:, None] * albedo_drift).mean()
        reg = (
            reg
            + 10 * deltas["rot"][i].pow(2).sum()
            + 1e3 * deltas["t"][i].pow(2).sum()
            + 1e-3 * deltas["expr"][i].pow(2).sum()
            + deltas["jaw"][i].pow(2).sum()
        )
        opt.zero_grad()
        (loss + reg).backward()
        opt.step()
        sched.step()
        history.append(loss.item())
        if it % LOG_EVERY == 0 or it == args.iters - 1:
            with torch.no_grad():
                inside = (regions[i][..., None] > 0).expand_as(img)
                mse = ((img - targets[i] - (1 - fg[i]) * bg) ** 2)[inside].mean().item()
            log(
                f"it {it:4d}/{args.iters}  loss {np.mean(history[-LOG_EVERY:]):.4f}  "
                f"train-photo PSNR (head region) {10 * np.log10(1 / mse):.2f} dB"
            )
    with torch.no_grad():
        snapshots[args.iters] = render_photo(snapshot_photo).numpy()

    # Evaluation on the held-out photo.
    metrics = dict(
        n_gauss=args.n_gauss,
        iters=args.iters,
        res=args.res,
        train=[names[i] for i in train],
        holdout=[names[i] for i in hold],
        final_train_loss=float(np.mean(history[-LOG_EVERY:])),
    )
    evals = []
    for i in hold:
        fid, bary, _ = init["samples"][i][3]
        mask = fid >= 0
        mask[mask] = head_face[fid[mask]] & (coverage[F0[fid[mask]]] > 1e-3).all(1)
        mask &= foregrounds[i] > 0.5
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        mesh_colors = np.clip(init["albedo_v"] * (init["Ys"][i] @ init["lighting"][i]), 0, 1)
        baseline = np.clip(R.interpolate(mesh_colors, F0, fid, bary), 0, 1)
        # Avatar frozen, camera / expression from the step-5 landmark fit; only the 27
        # lighting numbers are fitted.
        L_hold = torch.tensor(init["lighting"][i], dtype=torch.float32, requires_grad=True)
        opt_light = torch.optim.Adam([L_hold], lr=0.01)
        mask_t = torch.tensor(mask, dtype=torch.float32)[..., None]
        for _ in range(40):
            img = render_photo(i, use_deltas=False, L=L_hold)
            light_loss = ((img - torch.tensor(imgs[i])).abs() * mask_t).sum() / mask_t.sum()
            opt_light.zero_grad()
            light_loss.backward()
            opt_light.step()
        with torch.no_grad():
            rendered = render_photo(i, use_deltas=False, L=L_hold).clamp(0, 1).numpy()
        psnr_mesh, ssim_mesh = psnr_ssim(baseline, imgs[i], mask)
        psnr_gs, ssim_gs = psnr_ssim(rendered, imgs[i], mask)
        evals.append(dict(i=i, base=baseline, gs=rendered, m=mask))
        metrics[f"holdout_{names[i]}"] = dict(
            mesh_albedo_sh_psnr=psnr_mesh,
            mesh_albedo_sh_ssim=ssim_mesh,
            gaussian_psnr=psnr_gs,
            gaussian_ssim=ssim_gs,
            eval_pixels=int(mask.sum()),
            note="camera/expression from step-5 landmarks, only SH lighting fitted; face pixels "
            "covered by training photos",
        )
        log(
            f"held-out {names[i]}: mesh albedo x SH {psnr_mesh:.2f} dB / SSIM {ssim_mesh:.3f}  |  "
            f"Gaussian avatar {psnr_gs:.2f} dB / SSIM {ssim_gs:.3f}"
        )

    gt_path = photo_dir / "ground_truth.npz"
    if gt_path.exists() and not args.no_pose_refine:  # did photometric refinement fix the pose?
        gt = np.load(gt_path)
        with torch.no_grad():
            V_gt = flame(torch.tensor(gt["beta"], dtype=torch.float32)[None])[0][
                0, :N_FLAME_VERTICES
            ]
        _, Ru, _ = M.umeyama(
            flame.v_template[:N_FLAME_VERTICES].numpy().astype(np.float64),
            V_gt.numpy().astype(np.float64),
        )
        gt_index = {str(name): k for k, name in enumerate(gt["files"])}

        def rotation_error(rvec, j):
            return M.rotation_angle_deg(R.rotmat(rvec).T @ R.rotmat(gt["rvec"][j]) @ Ru)

        metrics["pose_refinement_rotation_error_deg"] = {
            names[i]: dict(
                step5_landmarks=rotation_error(photo_fit["rvec"][i], gt_index[names[i]]),
                after_photometric=rotation_error(
                    photo_fit["rvec"][i] + deltas["rot"][i].detach().numpy(), gt_index[names[i]]
                ),
            )
            for i in train
        }
    metrics["seconds"] = time.time() - log.t0
    save_json(metrics, out / "metrics.json")
    torch.save(
        dict(
            state=avatar.state_dict(),
            init=dict(
                V_rest=V_rest,
                faces=F,
                face_weight=init["face_weight"],
                albedo=init["albedo_init"],
                n_gauss=args.n_gauss,
            ),
            lighting=lighting.detach(),
            deltas={k: v.detach() for k, v in deltas.items()},
            files=names,
            train=train,
            res=args.res,
            n_proxy=n_proxy,
        ),
        IDENTITY_DIR / "gaussian_avatar.pt",
    )
    log(f"saved {IDENTITY_DIR / 'gaussian_avatar.pt'}")

    with torch.no_grad():
        plot_training_fit(
            out, imgs, names, train, snapshots, snapshot_photo, render_photo, args.iters
        )
        if evals:
            plot_holdout(out, imgs, names, evals, metrics)
        plot_novel_views_and_rig(
            out,
            avatar,
            flame,
            beta,
            offsets,
            lighting[train].mean(0),
            photo_fit,
            names,
            train,
            history,
            args.res,
        )
    log(f"figures in {out}")
    print({k: v for k, v in metrics.items() if k not in ("train", "holdout")})


def plot_training_fit(out, imgs, names, train, snapshots, snapshot_photo, render_photo, iters):
    fig, ax = plt.subplots(3, len(train), figsize=(3.2 * len(train), 9.6), squeeze=False)
    for k, i in enumerate(train):
        final = render_photo(i).clamp(0, 1).numpy()
        is_snapshot = i == snapshot_photo
        ax[0, k].imshow(imgs[i])
        ax[0, k].set_title(f"training photo {names[i]}", fontsize=8)
        ax[1, k].imshow(np.clip(snapshots[0], 0, 1) if is_snapshot else np.zeros_like(final))
        ax[1, k].set_title("initialisation (step-5 albedo)" if is_snapshot else "", fontsize=8)
        ax[2, k].imshow(final)
        ax[2, k].set_title(f"after {iters} iterations", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(
        "Gaussian avatar fitted to the training photos (black = outside the segmented head)"
    )
    viz.save_figure(fig, out / "01_training_fit.png", dpi=110)


def plot_holdout(out, imgs, names, evals, metrics):
    fig, ax = plt.subplots(len(evals), 5, figsize=(16, 3.4 * len(evals)), squeeze=False)
    for r, e in enumerate(evals):
        img, m = imgs[e["i"]], e["m"][..., None]
        hm = metrics[f"holdout_{names[e['i']]}"]
        panels = [
            (img, f"held-out {names[e['i']]}"),
            (
                np.where(m, e["base"], img * 0.3),
                f"mesh albedo x SH (step 5): {hm['mesh_albedo_sh_psnr']:.1f} dB",
            ),
            (np.where(m, e["gs"], img * 0.3), f"Gaussian avatar: {hm['gaussian_psnr']:.1f} dB"),
            (np.where(m, np.abs(e["base"] - img) * 4, 0), "|error| x4, mesh"),
            (np.where(m, np.abs(e["gs"] - img) * 4, 0), "|error| x4, Gaussians"),
        ]
        for c, (x, title) in enumerate(panels):
            ax[r, c].imshow(np.clip(x, 0, 1))
            ax[r, c].set_title(title, fontsize=8)
            ax[r, c].axis("off")
    fig.suptitle("Novel view = a photo never used for training (camera from landmarks only)")
    viz.save_figure(fig, out / "02_holdout.png", dpi=110)


def plot_novel_views_and_rig(
    out, avatar, flame, beta, offsets, L_mean, photo_fit, names, train, history, S
):
    def camera(yaw):
        return dict(
            rvec=torch.tensor(np.radians([0, yaw, 0]), dtype=torch.float32),
            t=torch.tensor([0, 0.02, 0.7]),
            f=2.2 * S,
            cx=S / 2,
            cy=S / 2,
        )

    white = torch.ones(S, S, 3)
    V_neutral = flame(beta, offsets=offsets)[0][0]
    yaws = [-60, -30, 0, 30, 60]
    views = [
        avatar.render(V_neutral, camera(y), L_mean, S, S, white).clamp(0, 1).numpy() for y in yaws
    ]
    rig = []
    for jaw_open, expr_from in [(0.0, None), (0.12, None), (0.25, None), (0.08, train[-1])]:
        if expr_from is None:
            expr = torch.zeros(1, flame.n_expr)
        else:
            expr = torch.tensor(photo_fit["expr"][expr_from], dtype=torch.float32)[None]
        pose = torch.zeros(1, 15)
        pose[0, 6] = jaw_open
        V = flame(beta, expr, pose, offsets)[0][0]
        rig.append(avatar.render(V, camera(15), L_mean, S, S, white).clamp(0, 1).numpy())

    fig, ax = plt.subplots(2, 5, figsize=(15, 6.4))
    for k, (view, yaw) in enumerate(zip(views, yaws)):
        ax[0, k].imshow(view)
        ax[0, k].set_title(f"novel view, yaw {yaw}", fontsize=8)
    rig_titles = ["jaw 0", "jaw 0.12 rad", "jaw 0.25 rad", f"expression of {names[train[-1]]}"]
    for k, (view, title) in enumerate(zip(rig, rig_titles)):
        ax[1, k].imshow(view)
        ax[1, k].set_title(f"rigged: {title}", fontsize=8)
    ax[1, 4].plot(np.convolve(history, np.ones(10) / 10, mode="valid"))
    ax[1, 4].set_title("training loss", fontsize=8)
    ax[1, 4].set_yscale("log")
    for a in list(ax[0]) + list(ax[1, :4]):
        a.axis("off")
    fig.suptitle(
        "Gaussian avatar: neutral pose under averaged training light; the mesh rig "
        "drives mouth, jaw and expression"
    )
    viz.save_figure(fig, out / "03_novel_views_and_rig.png", dpi=110)


if __name__ == "__main__":
    main()
