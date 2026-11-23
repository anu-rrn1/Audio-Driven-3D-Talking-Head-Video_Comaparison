"""Step 2: identity geometry. User 3D face mesh -> rigged FLAME identity + mouth structure.

    user mesh (any topology) --similarity + non-rigid ICP--> FLAME topology
        --projection--> beta + Taubin-smoothed per-vertex offsets
    -> normals and cotangent mean curvature
    -> mouth region: FLAME lip region grown by a k-ring breadth-first search
    -> red-green refinement of the mouth region as a sparse operator (follows the FLAME rig)
    -> mouth-cavity and teeth proxy geometry skinned to the neck / jaw joints

Outputs: data/identity/{identity.npz, refine_S.npz}, outputs/flame_pipeline/02_identity/

Usage:
    python scripts/flame_pipeline/02_identity_geometry.py --mesh my_face.obj [--landmarks lmk.txt]
    python scripts/flame_pipeline/02_identity_geometry.py --beta id.npz   # FLAME shape given
    python scripts/flame_pipeline/02_identity_geometry.py                 # mesh from the config
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from tinypt import visualization as viz
from tinypt.config import FLAME_OUTPUTS, IDENTITY_DIR, ROOT, ensure_dir, load_config
from tinypt.flame import load_flame
from tinypt.geometry import mesh as M
from tinypt.geometry import mouth as MO
from tinypt.geometry.registration import register_identity
from tinypt.io_utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mesh", default=None, help="identity mesh (OBJ, or PLY/STL with trimesh)")
    parser.add_argument("--beta", default=None, help="npz with FLAME shape parameters 'beta'")
    parser.add_argument("--landmarks", default=None, help="[L, 3] landmarks on the mesh (txt/npy)")
    return parser.parse_args()


def load_landmarks(path):
    if not path:
        return None
    return np.loadtxt(path) if str(path).endswith(".txt") else np.load(path)


def fit_identity(flame, args, cfg):
    """Return (beta, offsets, metrics, scan, source description). `scan` is None unless a
    mesh of a different topology was registered."""
    n = len(flame.v_template)
    mesh_path = Path(args.mesh) if args.mesh else ROOT / cfg["mesh"]
    if args.beta:
        beta = torch.tensor(np.load(args.beta)["beta"][: flame.n_shape], dtype=torch.float32)
        beta = torch.nn.functional.pad(beta, (0, flame.n_shape - len(beta)))
        return beta, torch.zeros(n, 3), {}, None, f"beta file {args.beta}"

    if not mesh_path.exists():
        print(f"No identity mesh at {mesh_path}; using the mean FLAME head as a demo identity.")
        return torch.zeros(flame.n_shape), torch.zeros(n, 3), {}, None, "mean FLAME (demo)"

    Vs, Fs = M.load_mesh(mesh_path)
    print(f"identity mesh: {mesh_path}  ({len(Vs)} vertices, {len(Fs)} faces)")
    if len(Vs) == n:  # already in FLAME topology: similarity alignment + projection only
        V_template = flame.v_template.numpy().astype(np.float64)
        s, R, t = M.umeyama(V_template, Vs)
        beta, offsets = flame.fit_beta(M.invert_sim(Vs, s, R, t))
        return beta, offsets, {}, None, "FLAME-topology mesh"

    landmarks = load_landmarks(args.landmarks or cfg.get("landmarks"))
    reg = register_identity(flame, Vs, Fs, landmarks, cfg["registration"])
    beta, offsets = torch.tensor(reg["beta"]), torch.tensor(reg["offsets"])
    return (
        beta,
        offsets,
        {"registration": reg["metrics"]},
        (Vs, Fs, reg),
        "registered scan (non-rigid ICP)",
    )


def mouth_region(flame, V_id, F, ring_hops):
    """Inner-lip loop, upper/lower split, and the mouth region with a smooth falloff weight."""
    n = len(V_id)
    lips = flame.region("lips")
    loop = MO.inner_lip_loop(V_id, F, lips)
    split = MO.split_upper_lower(V_id, loop)
    seeds = np.union1d(lips, loop)
    region_idx, hops = M.k_ring(F, n, seeds, ring_hops)
    hops = np.where(np.isin(region_idx, seeds), 0, hops)
    region_w = M.smoothstep(1 - hops / (ring_hops + 1))
    in_region = np.zeros(n, bool)
    in_region[region_idx] = True
    return loop, split, region_idx, region_w, in_region


def refine_mouth(V_id, F, in_region, enabled):
    """Red-green refinement of the mouth triangles. Returns S, refined faces and the mask of
    refined faces built only from mouth-region vertices."""
    region_tri = in_region[F].all(1)
    if enabled:
        S, F_ref = M.refine_region(V_id, F, region_tri)
    else:
        S, F_ref = sp.eye(len(V_id), format="csr"), F
    built_from_region = np.asarray((S[:, ~in_region] != 0).sum(1)).ravel() == 0
    ref_tri = built_from_region[F_ref].all(1)
    stats = dict(
        vertices_before=len(V_id),
        vertices_after=int(S.shape[0]),
        faces_before=len(F),
        faces_after=len(F_ref),
        region_faces_before=int(region_tri.sum()),
        region_faces_after=int(ref_tri.sum()),
        boundary_loops_before=len(M.boundary_loops(F)),
        boundary_loops_after=len(M.boundary_loops(F_ref)),
    )
    return S, F_ref, ref_tri, stats


def save_figures(
    out,
    flame,
    V_id,
    F,
    curvature,
    region_idx,
    region_w,
    region_tri,
    loop,
    V_ref,
    F_ref,
    ref_tri,
    proxy_P,
    proxy_F,
    part,
    scan,
):
    n = len(V_id)
    Vi = viz.flame_to_img(V_id)
    face_tri = np.where(np.isin(F, flame.region("face")).all(1))[0]
    viz.mesh_views(
        Vi,
        F[face_tri],
        curvature,
        out / "01_identity_curvature.png",
        "identity mesh (mean curvature)",
    )
    weight = np.zeros(n)
    weight[region_idx] = region_w
    viz.mesh_views(
        Vi,
        F[face_tri],
        weight,
        out / "02_mouth_region.png",
        "mouth region (falloff weight)",
        vmax_pct=100,
        highlight=loop,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = [
        (Vi, F, region_tri, f"original ({region_tri.sum()} mouth faces)"),
        (
            viz.flame_to_img(V_ref),
            F_ref,
            ref_tri,
            f"red-green refined ({ref_tri.sum()} mouth faces)",
        ),
    ]
    for ax, (VV, FF, mask, title) in zip(axes, panels):
        shown = mask | np.isin(FF, region_idx).any(1)
        viz.wireframe_2d(ax, VV, FF[shown], mask[shown], lw=0.4)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Adaptive refinement of the mouth region (conforming, no T-junctions)")
    viz.save_figure(fig, out / "03_mouth_refinement.png")

    V_all = viz.flame_to_img(np.concatenate([V_id, proxy_P]))
    part_value = np.select([part == "cavity", part == "upper_teeth"], [0.5, 1.0], 0.8)
    colors = np.concatenate([np.zeros(n), part_value])
    viz.mesh_views(
        V_all,
        np.concatenate([F[region_tri], proxy_F]),
        colors,
        out / "04_mouth_proxies.png",
        "lips + cavity + teeth proxies",
        vmax_pct=100,
    )

    if scan is not None:
        Vs, Fs, reg = scan
        s, R, t = reg["sim"]
        dist = M.point_to_surface(Vs, M.apply_sim(V_id, s, R, t), F)
        subset = np.random.default_rng(0).choice(len(Fs), min(len(Fs), 20000), replace=False)
        viz.mesh_views(
            viz.flame_to_img(Vs),
            Fs[subset],
            dist,
            out / "05_scan_registration_error.png",
            "scan -> registered FLAME distance",
            cmap="magma",
        )
    viz.plotly_mesh(
        Vi,
        F[face_tri],
        curvature,
        out / "interactive_identity.html",
        "Identity mesh (rotate me)",
        highlight=region_idx,
    )


def main():
    args = parse_args()
    cfg = load_config()
    identity_cfg, mouth_cfg = cfg["identity"], cfg["mouth"]
    out = ensure_dir(FLAME_OUTPUTS / "02_identity")
    ensure_dir(IDENTITY_DIR)
    flame = load_flame(cfg)
    F = flame.faces
    n = len(flame.v_template)

    # 1. Identity: user mesh -> beta + offsets in FLAME topology.
    beta, offsets, metrics, scan, source = fit_identity(flame, args, identity_cfg)
    offsets_raw = offsets.numpy().astype(np.float64)
    if np.abs(offsets_raw).max() > 0:  # smoothing removes scan noise from the offset field
        offsets_np = M.taubin_smooth(offsets_raw, F, iters=identity_cfg["offset_smooth_iters"])
    else:
        offsets_np = offsets_raw
    offsets = torch.tensor(offsets_np, dtype=torch.float32)
    V_id = flame.shaped(beta, offsets).numpy().astype(np.float64)

    # 2. Differential geometry.
    curvature = M.mean_curvature(V_id, F)

    def rms_mm(x):
        return float(1000 * np.sqrt((x**2).sum(1).mean()))

    metrics.update(
        source=source,
        mesh=M.edge_stats(V_id, F),
        mean_curvature=float(np.median(curvature)),
        beta_norm=float(beta.norm()),
        offset_rms_mm=rms_mm(offsets_np),
        offset_rms_before_smoothing_mm=rms_mm(offsets_raw),
    )

    # 3. Mouth structure.
    loop, split, region_idx, region_w, in_region = mouth_region(
        flame, V_id, F, mouth_cfg["ring_hops"]
    )
    region_tri = in_region[F].all(1)
    mouth_width = float(np.ptp(V_id[loop, 0]))
    open_rest = float(MO.openness(V_id, split["up_mid"], split["lo_mid"], mouth_width))

    # 4. Adaptive refinement of the mouth region (V_ref = S @ V).
    S, F_ref, ref_tri, metrics["refinement"] = refine_mouth(V_id, F, in_region, mouth_cfg["refine"])
    V_ref = S @ V_id

    # 5. Mouth-cavity and teeth proxies.
    proxy_P, proxy_W, proxy_F, part = MO.build_proxies(V_id, loop, split, n, mouth_cfg)
    metrics["proxies"] = {p: int((part == p).sum()) for p in np.unique(part)}

    np.savez(
        IDENTITY_DIR / "identity.npz",
        beta=beta.numpy(),
        offsets=offsets_np.astype(np.float32),
        V_id=V_id.astype(np.float32),
        faces=F,
        region_idx=region_idx,
        region_w=region_w,
        region_tri=region_tri,
        inner_loop=loop,
        up=split["up"],
        lo=split["lo"],
        corners=split["corners"],
        up_mid=split["up_mid"],
        lo_mid=split["lo_mid"],
        mouth_w=mouth_width,
        open_rest=open_rest,
        F_ref=F_ref,
        ref_tri=ref_tri,
        proxy_P=proxy_P.astype(np.float32),
        proxy_W=proxy_W.astype(np.float32),
        proxy_F=proxy_F,
        proxy_part=part,
        curvature=curvature,
    )
    sp.save_npz(IDENTITY_DIR / "refine_S.npz", S.tocsr())
    M.save_obj(out / "identity_flame.obj", V_id, F)
    M.save_obj(out / "identity_refined.obj", V_ref, F_ref)
    M.save_obj(
        out / "identity_with_mouth_proxies.obj",
        np.concatenate([V_id, proxy_P]),
        np.concatenate([F, proxy_F]),
    )
    save_json(metrics, out / "identity_metrics.json")
    print({k: v for k, v in metrics.items() if k != "registration"})

    save_figures(
        out,
        flame,
        V_id,
        F,
        curvature,
        region_idx,
        region_w,
        region_tri,
        loop,
        V_ref,
        F_ref,
        ref_tri,
        proxy_P,
        proxy_F,
        part,
        scan,
    )
    print("saved", IDENTITY_DIR / "identity.npz", "and figures in", out)


if __name__ == "__main__":
    main()
