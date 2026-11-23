"""Step 2: 3D mesh processing of the tracked face mesh.

    reference frames (still head, closed mouth) -> canonical mesh (similarity alignment,
    averaging, Taubin smoothing) -> normals and cotangent mean curvature -> mouth region
    -> per-frame head-pose removal (Umeyama) -> lip displacement targets

Outputs: data/processed/mesh_data.npz, outputs/video_baseline/02_mesh/
"""

import matplotlib.pyplot as plt
import numpy as np

from tinypt import visualization as viz
from tinypt.config import BASELINE_OUTPUTS, PROCESSED_DIR, ensure_dir, load_config
from tinypt.geometry import mesh as M
from tinypt.io_utils import load_json, save_json
from tinypt.video_baseline.face_mesh import LIP_IDX, LOWER_INNER_LIP, UPPER_INNER_LIP, openness


def rigid_part(V):
    """Vertices clearly above the mouth (forehead, eyes, nose): the rigid part of the face."""
    face_height = np.ptp(V[:, 1])
    mouth_y = V[LIP_IDX, 1].mean()
    return np.where(V[:, 1] < mouth_y - 0.05 * face_height)[0]


def reference_frames(lm, n_refs, min_separation=10):
    """Frames ranked by combined rank of head-pose deviation and mouth opening."""
    mean = lm.mean(0)
    stable = rigid_part(mean)
    pose_dev = np.sqrt(((lm[:, stable] - mean[stable]) ** 2).sum(-1).mean(1))
    open_raw = openness(lm, np.ptp(mean[:, 1]))

    def rank(x):
        return np.argsort(np.argsort(x)) / len(x)

    refs = []
    for i in np.argsort(rank(pose_dev) + rank(open_raw)):
        if all(abs(i - j) >= min_separation for j in refs):
            refs.append(int(i))
        if len(refs) == n_refs:
            break
    return refs, stable


def mouth_region(V, k):
    """The k vertices nearest to the lip centroid, with a smooth falloff weight."""
    dist = np.linalg.norm(V - V[LIP_IDX].mean(0), axis=1)
    region = np.sort(np.argsort(dist)[:k])
    d_max = dist[region].max()
    weight = M.smoothstep((d_max - dist[region]) / (0.35 * d_max))
    return region, weight


def remove_head_pose(lm, V_can, stable):
    """Similarity-align every frame to the canonical mesh on the rigid vertices."""
    T = len(lm)
    aligned = np.zeros_like(lm)
    s, R, t = np.zeros(T), np.zeros((T, 3, 3)), np.zeros((T, 3))
    for i in range(T):
        s[i], R[i], t[i] = M.umeyama(lm[i, stable], V_can[stable])
        aligned[i] = M.apply_sim(lm[i], s[i], R[i], t[i])
    return aligned, s, R, t


def main():
    cfg = load_config()
    mesh_cfg = cfg["mesh"]
    out = ensure_dir(BASELINE_OUTPUTS / "02_mesh")
    lm = np.load(PROCESSED_DIR / "landmarks.npy").astype(np.float64)  # [T, 468, 3] crop pixels
    frames = np.load(PROCESSED_DIR / "frames.npy")
    F0 = np.load(PROCESSED_DIR / "faces.npy")

    # 1. Reference frames: still head and closed mouth.
    refs, stable0 = reference_frames(lm, mesh_cfg["n_reference_frames"])
    print("reference frames:", refs)

    # 2. Canonical mesh: align the reference meshes to the first one, average, smooth.
    aligned_refs = []
    for i in refs:
        s, R, t = M.umeyama(lm[i, stable0], lm[refs[0], stable0])
        aligned_refs.append(M.apply_sim(lm[i], s, R, t))
    V_raw = np.mean(aligned_refs, 0)
    F = M.orient_faces_towards_camera(V_raw, F0)
    V_can = M.taubin_smooth(V_raw, F, iters=mesh_cfg["taubin_iters"])
    curv_raw, curv_can = M.mean_curvature(V_raw, F), M.mean_curvature(V_can, F)
    stats = dict(
        raw=M.edge_stats(V_raw, F),
        smoothed=M.edge_stats(V_can, F),
        mean_curvature_raw=float(curv_raw.mean()),
        mean_curvature_smoothed=float(curv_can.mean()),
        reference_frames=refs,
    )

    # 3. Mouth region with falloff weights.
    region, region_w = mouth_region(V_can, mesh_cfg["region_k"])
    in_region = np.zeros(len(V_can), bool)
    in_region[region] = True
    region_tri = in_region[F].sum(1) >= 2
    assert UPPER_INNER_LIP in region and LOWER_INNER_LIP in region
    face_height = np.ptp(V_can[:, 1])
    stable = rigid_part(V_can)

    # 4. Remove the head pose of every frame -> lip motion in canonical space.
    aligned, pose_s, pose_R, pose_t = remove_head_pose(lm, V_can, stable)
    D_vert = aligned[:, region] - V_can[region]  # [T, K, 3] training targets
    open_series = openness(aligned, face_height)
    open_scale = float(max(np.percentile(open_series, 95), 1e-3))
    residual = np.linalg.norm(aligned[:, stable] - V_can[stable], axis=-1).mean()
    stats.update(
        region_vertices=int(len(region)),
        rigid_alignment_residual_px=float(residual),
        lip_displacement_rms_px=float(np.sqrt((D_vert**2).sum(-1).mean())),
        open_scale=open_scale,
    )

    np.savez(
        PROCESSED_DIR / "mesh_data.npz",
        V_can=V_can,
        V_raw=V_raw,
        faces=F,
        region_idx=region,
        region_w=region_w,
        region_tri=region_tri,
        stable_idx=stable,
        aligned=aligned.astype(np.float32),
        D_vert=D_vert.astype(np.float32),
        pose_s=pose_s,
        pose_R=pose_R,
        pose_t=pose_t,
        open_series=open_series,
        open_scale=open_scale,
        face_h=face_height,
        ref_ids=np.array(refs),
        curvature=curv_can,
    )
    M.save_obj(out / "canonical_mesh.obj", V_can, F)
    save_json(stats, out / "mesh_stats.json")

    # 5. Figures.
    viz.landmark_overlay(frames, lm, refs, out / "01_reference_frames.png")
    viz.mesh_views(
        V_raw, F, curv_raw, out / "02_mesh_raw_curvature.png", "raw mean-ref mesh (curvature)"
    )
    viz.mesh_views(
        V_can, F, curv_can, out / "03_mesh_smoothed_curvature.png", "Taubin-smoothed canonical mesh"
    )
    weight = np.zeros(len(V_can))
    weight[region] = region_w
    viz.mesh_views(
        V_can,
        F,
        weight,
        out / "04_mouth_region.png",
        "mouth region (colour = falloff weight)",
        vmax_pct=100,
    )

    order = np.argsort(open_series)
    picks = [order[len(order) // 20], order[len(order) // 2], order[-len(order) // 20]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, i, name in zip(axes, picks, ["mouth ~closed", "mouth mid", "mouth wide"]):
        viz.wireframe_2d(ax, V_can, F, lw=0.5, color="#bbbbbb")
        viz.wireframe_2d(ax, aligned[i], F, region_tri)
        ax.set_title(
            f"frame {i}: {name}\n(grey = canonical, black/orange = pose-removed frame mesh)",
            fontsize=8,
        )
    viz.save_figure(fig, out / "05_mesh_deformation.png")

    viz.plot_curves(
        [("mouth opening (normalised)", open_series, dict(color="tab:blue"))],
        out / "06_mouth_opening_timeline.png",
        "Mouth opening extracted from the 3D mesh over the video",
        ylabel="gap / face height",
        fps=load_json(PROCESSED_DIR / "meta.json")["fps"],
    )
    viz.plotly_mesh(
        V_can,
        F,
        curv_can,
        out / "interactive_canonical_mesh.html",
        "Canonical mesh (rotate me)",
        highlight=region,
    )
    print(stats)


if __name__ == "__main__":
    main()
