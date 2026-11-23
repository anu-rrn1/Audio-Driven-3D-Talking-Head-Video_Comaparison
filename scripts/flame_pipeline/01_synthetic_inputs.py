"""Step 1: synthetic test inputs with ground truth.

Creates a person who does not exist from a random FLAME shape:
    data/identity/face.obj                   the 3D face mesh (input of step 2)
    data/photos/synthetic/photo_*.png        reference photos with random head pose,
                                             expression, jaw opening, lighting and focal
                                             length (input of step 5)
    data/photos/synthetic/ground_truth.npz   cameras, expressions and albedo, used only to
                                             score later steps

Mesh and photos show the same identity, as the pipeline assumes. Real photos of a
consenting person go through the same scripts.

Usage:
    python scripts/flame_pipeline/01_synthetic_inputs.py [--seed 0] [--n_photos 4] [--size 512]
"""

import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from tinypt.config import ROOT, ensure_dir, load_config
from tinypt.flame import load_flame
from tinypt.geometry import mesh as M
from tinypt.rendering import rasterizer as R

VIEWS = [(0, 0), (-28, 6), (26, -5), (12, 14), (-14, -12)]  # (yaw, pitch) in degrees
FOCAL_FACTOR = 2.2  # deliberately different from the fitter's default of 2.0 x image size


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_photos", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mesh_out", default="data/identity/face.obj")
    parser.add_argument("--photos_out", default="data/photos/synthetic")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    flame = load_flame(load_config())
    F = flame.faces
    V_template = flame.v_template.numpy().astype(np.float64)
    S = args.size

    beta = torch.zeros(1, flame.n_shape)
    beta[0, :20] = torch.tensor(rng.normal(0, 1.0, 20), dtype=torch.float32)
    with torch.no_grad():
        V_neutral = flame(beta)[0][0].numpy().astype(np.float64)
    albedo = R.procedural_albedo(V_template, F, flame.region("lips"), rng)
    mesh_out = ROOT / args.mesh_out
    ensure_dir(mesh_out.parent)
    M.save_obj(mesh_out, V_neutral, F)

    photo_dir = ensure_dir(ROOT / args.photos_out)
    for old in photo_dir.glob("photo_*.png"):
        old.unlink()
    truth = dict(rvec=[], t=[], f=[], expr=[], jaw=[], light=[], files=[])
    for i in range(args.n_photos):
        yaw, pitch = VIEWS[i % len(VIEWS)]
        expr = torch.zeros(1, flame.n_expr)
        pose = torch.zeros(1, 15)
        if i > 0:  # photo 0 is neutral, the others are expressive
            expr[0, :10] = torch.tensor(rng.normal(0, 0.8, 10), dtype=torch.float32)
            pose[0, 6] = float(rng.uniform(0.02, 0.15))  # jaw opening (rad)
        with torch.no_grad():
            V = flame(beta, expr, pose)[0][0].numpy().astype(np.float64)
        rvec = np.radians([pitch + rng.normal(0, 2), yaw + rng.normal(0, 2), rng.normal(0, 4)])
        t = np.array([rng.normal(0, 0.008), rng.normal(0.02, 0.008), rng.uniform(0.65, 0.8)])
        f = FOCAL_FACTOR * S
        light = np.array([rng.normal(0, 0.4), rng.uniform(0.2, 0.7), 1.0])

        # Render at 2x and downsample (anti-aliased edges), then add camera blur and noise.
        img, *_ = R.render(
            V,
            F,
            rvec,
            t,
            2 * f,
            2 * S,
            2 * S,
            albedo,
            light=light,
            bg=R.background(2 * S, 2 * S, rng),
        )
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (3, 3), 0.5).astype(np.float64) + rng.normal(0, 1.5, img.shape)
        name = f"photo_{i:02d}.png"
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(photo_dir / name)

        values = dict(
            rvec=rvec,
            t=t,
            f=f,
            expr=expr[0].numpy(),
            jaw=pose[0, 6:9].numpy(),
            light=light,
            files=name,
        )
        for key, value in values.items():
            truth[key].append(value)

    np.savez(
        photo_dir / "ground_truth.npz",
        beta=beta[0].numpy(),
        albedo=albedo,
        size=S,
        **{k: np.array(v) for k, v in truth.items()},
    )
    print(
        f"synthetic identity (seed {args.seed}): mesh -> {mesh_out}, "
        f"{args.n_photos} photos -> {photo_dir}"
    )


if __name__ == "__main__":
    main()
