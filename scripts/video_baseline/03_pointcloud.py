"""Step 3: point-cloud processing of the mouth region.

    mouth-region mesh -> lip point cloud (region vertices + area-weighted surface samples,
    each bound to a triangle) -> normalisation, PCA normals, k-NN graph, farthest-point
    sampling, Chamfer distance -> per-frame point displacement targets

Outputs: data/processed/points.npz, outputs/video_baseline/03_points/
"""

import matplotlib.pyplot as plt
import numpy as np

from tinypt import visualization as viz
from tinypt.config import BASELINE_OUTPUTS, PROCESSED_DIR, ensure_dir, load_config
from tinypt.geometry import mesh as M
from tinypt.geometry import point_cloud as PC
from tinypt.io_utils import save_json

N_FPS_DEMO = 48


def bind_points(V, F, region, n_points, rng):
    """Lip cloud bound to the mesh: every point is a barycentric combination of region
    vertices. Returns (local vertex ids [N, 3], weights [N, 3]); the first K points are the
    region vertices themselves."""
    K = len(region)
    in_region = np.zeros(len(V), bool)
    in_region[region] = True
    region_faces = np.where(in_region[F].all(1))[0]
    if len(region_faces) == 0:
        region_faces = np.where(in_region[F].sum(1) >= 2)[0]
    tri, bary = M.sample_surface(V, F, region_faces, rng, n_points - K)
    local = -np.ones(len(V), int)
    local[region] = np.arange(K)
    local_tri = local[F[tri]]
    valid = (local_tri >= 0).all(1)  # samples on triangles with all vertices in the region
    bind_idx = np.concatenate([np.stack([np.arange(K)] * 3, 1), local_tri[valid]])
    bind_w = np.concatenate([np.tile([1.0, 0, 0], (K, 1)), bary[valid]])
    return bind_idx, bind_w


def main():
    cfg = load_config()
    out = ensure_dir(BASELINE_OUTPUTS / "03_points")
    mesh = np.load(PROCESSED_DIR / "mesh_data.npz")
    V, F, region, D_vert = mesh["V_can"], mesh["faces"], mesh["region_idx"], mesh["D_vert"]
    K, T = len(region), len(D_vert)
    rng = np.random.default_rng(0)

    # 1. Mesh -> point cloud bound to the mesh, so it deforms with it.
    bind_idx, bind_w = bind_points(V, F, region, cfg["points"]["n_points"], rng)
    N = len(bind_idx)
    P0 = (V[region][bind_idx] * bind_w[..., None]).sum(1)  # canonical lip cloud [N, 3]

    # 2. Point-cloud features.
    P0n, c0, s0 = PC.normalize_cloud(P0)
    normals = PC.estimate_normals(P0n, k=10)
    knn = PC.knn_indices(P0n, cfg["points"]["knn"])
    fps_idx = PC.farthest_point_sampling(P0n, N_FPS_DEMO)

    # 3. Dynamic targets: displacement of every point in every frame.
    D_pts = (D_vert[:, bind_idx] * bind_w[None, :, :, None]).sum(2)  # [T, N, 3]
    np.savez(
        PROCESSED_DIR / "points.npz",
        P0=P0,
        P0n=P0n,
        c0=c0,
        s0=s0,
        normals=normals,
        knn=knn,
        bind_idx=bind_idx,
        bind_w=bind_w,
        D_pts=D_pts.astype(np.float32),
        K=K,
    )

    openness = mesh["open_series"]
    i_open, i_closed = int(openness.argmax()), int(openness.argmin())
    stats = dict(
        n_points=int(N),
        n_mesh_vertex_points=int(K),
        n_surface_samples=int(N - K),
        cloud_radius_px=float(s0),
        chamfer_closed_vs_open_px=PC.chamfer(P0 + D_pts[i_closed], P0 + D_pts[i_open]),
        chamfer_canonical_vs_open_px=PC.chamfer(P0, P0 + D_pts[i_open]),
        max_point_displacement_px=float(np.linalg.norm(D_pts, axis=-1).max()),
    )
    save_json(stats, out / "point_stats.json")
    print(stats)

    # 4. Figures.
    viz.points_views(
        P0,
        np.where(np.arange(N) < K, 0.0, 1.0),
        out / "01_lip_point_cloud.png",
        "lip cloud (purple=mesh vertices, yellow=surface samples)",
        size=14,
    )
    edges = np.array([(i, knn[i, j]) for i in range(N) for j in range(4)])
    viz.points_views(
        P0,
        P0[:, 1],
        out / "02_knn_graph.png",
        "k-NN graph (4 of 12 neighbours drawn)",
        size=8,
        edges=edges,
    )
    viz.points_views(
        P0, P0[:, 1], out / "03_pca_normals.png", "PCA normals", size=8, normals=normals
    )
    is_fps = np.zeros(N)
    is_fps[fps_idx] = 1
    viz.points_views(
        P0,
        is_fps,
        out / "04_farthest_point_sampling.png",
        f"FPS: {N_FPS_DEMO} of {N} points (yellow)",
        size=12,
    )

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    viz.wireframe_2d(ax[0], V, F, np.isin(F, region).sum(1) >= 2)
    ax[0].scatter(P0[:, 0], P0[:, 1], s=5, c="tab:blue")
    ax[0].set_title("mesh (orange = mouth region triangles) + bound point cloud", fontsize=9)
    for i in [i_closed, i_open]:
        pts = P0 + D_pts[i]
        ax[1].scatter(pts[:, 0], pts[:, 1], s=7, label=f"frame {i} (open={openness[i]:.2f})")
    ax[1].set_ylim(ax[1].get_ylim()[::-1])
    ax[1].set_aspect("equal")
    ax[1].legend(fontsize=7)
    ax[1].set_title("same point cloud, closed vs open mouth", fontsize=9)
    viz.save_figure(fig, out / "05_mesh_to_points_and_motion.png")

    magnitude = np.linalg.norm(D_pts, axis=-1)
    order = np.argsort(openness)
    picks = [order[int(q * (T - 1))] for q in (0.05, 0.3, 0.55, 0.8, 0.95, 1.0)]
    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    for ax, i in zip(axes, picks):
        pts = P0 + D_pts[i]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=magnitude[i],
            s=8,
            cmap="turbo",
            vmin=0,
            vmax=max(magnitude.max() * 0.8, 1e-3),
        )
        ax.set_ylim(P0[:, 1].max() + 12, P0[:, 1].min() - 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"frame {i}", fontsize=8)
    fig.suptitle("Dynamic lip point cloud (colour = displacement from canonical, px)")
    viz.save_figure(fig, out / "06_dynamic_point_cloud.png")
    viz.points_views(
        P0 + D_pts[i_open],
        magnitude[i_open],
        out / "07_open_mouth_3d.png",
        "open-mouth point cloud",
        size=12,
        cmap="turbo",
        extra=P0,
    )

    shown = np.linspace(0, min(T, 300) - 1, 60).astype(int)
    viz.plotly_points_animation(
        np.stack([P0 + D_pts[i] for i in shown]),
        np.stack([magnitude[i] for i in shown]),
        [str(i) for i in shown],
        out / "interactive_dynamic_points.html",
    )


if __name__ == "__main__":
    main()
