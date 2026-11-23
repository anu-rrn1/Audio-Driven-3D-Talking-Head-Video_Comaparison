"""Plotting helpers: Matplotlib (static PNG), Plotly (interactive HTML), OpenCV (video panels).

3D plots use image-style coordinates (x right, y down, z away from the camera).
"""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

VIEWS = [("front", 5, -90), ("3/4 view", 15, -55), ("side", 5, 0)]


def save_figure(fig, path, dpi=130):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def to_view(X):
    """Image coordinates -> Matplotlib 3D display coordinates (x, depth, up)."""
    return np.stack([X[..., 0], X[..., 2], -X[..., 1]], -1)


def flame_to_img(X):
    """FLAME / OpenGL coordinates (y up, z towards the camera) -> image-style coordinates."""
    return X * np.array([1.0, -1.0, -1.0])


def _axis_3d(fig, pos, X, title, elev, azim):
    ax = fig.add_subplot(*pos, projection="3d")
    Xv = to_view(X)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(np.maximum(np.ptp(Xv, 0), 1e-3))
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    return ax, Xv


def mesh_views(V, F, scalar, path, title="", cmap="viridis", highlight=None, vmax_pct=95):
    """Three views of a mesh with faces coloured by a per-vertex scalar; `highlight` vertex
    ids are drawn in red."""
    fig = plt.figure(figsize=(12, 4.5))
    values = np.asarray(scalar, float)
    vmax = np.percentile(values, vmax_pct) + 1e-9
    face_colors = plt.get_cmap(cmap)(np.clip(values[F].mean(1) / vmax, 0, 1))
    normals = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    facing = np.abs(normals[:, 2]) / (np.linalg.norm(normals, axis=1) + 1e-9)
    face_colors[:, :3] *= (0.55 + 0.45 * facing)[:, None]  # simple shading
    for i, (name, elev, azim) in enumerate(VIEWS):
        ax, Xv = _axis_3d(fig, (1, 3, i + 1), V, f"{title} - {name}", elev, azim)
        ax.add_collection3d(
            Poly3DCollection(
                Xv[F], facecolors=face_colors, edgecolors=(0, 0, 0, 0.25), linewidths=0.3
            )
        )
        ax.auto_scale_xyz(Xv[:, 0], Xv[:, 1], Xv[:, 2])
        if highlight is not None:
            ax.scatter(*Xv[highlight].T, c="red", s=6, depthshade=False)
    save_figure(fig, path)


def points_views(
    P, color, path, title="", size=10, cmap="viridis", edges=None, normals=None, extra=None
):
    """Three views of a point cloud. edges: [E, 2] index pairs; normals: [N, 3] drawn as
    short lines; extra: a second cloud drawn faintly for reference."""
    fig = plt.figure(figsize=(12, 4.5))
    for i, (name, elev, azim) in enumerate(VIEWS):
        ax, Xv = _axis_3d(fig, (1, 3, i + 1), P, f"{title} - {name}", elev, azim)
        if edges is not None:
            for a, b in Xv[edges]:
                ax.plot(*zip(a, b), c="gray", lw=0.3, alpha=0.5)
        if normals is not None:
            Nv = to_view(normals) * 0.12 * np.ptp(Xv, 0).max()
            for p, n in zip(Xv, Nv):
                ax.plot(*zip(p, p + n), c="tab:red", lw=0.6)
        ax.scatter(*Xv.T, c=color, s=size, cmap=cmap, depthshade=False)
        if extra is not None:
            ax.scatter(*to_view(extra).T, c="k", s=size * 0.3, alpha=0.4, depthshade=False)
    save_figure(fig, path)


def gaussian_cloud_views(pos, rgb, opacity, path, title="Canonical Gaussian head"):
    fig = plt.figure(figsize=(12, 4.5))
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.clip(opacity, 0.2, 1)[:, None]], 1)
    for i, (name, elev, azim) in enumerate(VIEWS):
        ax, Xv = _axis_3d(fig, (1, 3, i + 1), pos, f"{title} - {name}", elev, azim)
        ax.scatter(*Xv.T, c=rgba, s=5, depthshade=False)
    save_figure(fig, path)


def image_grid(imgs, titles, path, ncols=6, figsize_per=2.2, suptitle=None):
    nrows = int(np.ceil(len(imgs) / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize_per, nrows * figsize_per * 1.05), squeeze=False
    )
    for ax in axs.ravel():
        ax.axis("off")
    for ax, img, title in zip(axs.ravel(), imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    save_figure(fig, path, dpi=120)


def landmark_overlay(frames, landmarks, ids, path):
    imgs = []
    for i in ids:
        img = frames[i].copy()
        for x, y, _ in landmarks[i]:
            cv2.circle(img, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)
        imgs.append(cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST))
    image_grid(
        imgs,
        [f"frame {i}" for i in ids],
        path,
        ncols=len(ids),
        suptitle="Cropped frames + 3D face-mesh landmarks",
    )


def _triangle_edges(V, F):
    segments = V[F][:, :, :2]
    return np.concatenate([segments[:, [0, 1]], segments[:, [1, 2]], segments[:, [2, 0]]])


def wireframe_2d(ax, V, F, region_mask=None, lw=0.4, color="k"):
    """2D wireframe in image coordinates; triangles in `region_mask` are drawn in orange."""
    ax.add_collection(LineCollection(_triangle_edges(V, F), colors=color, linewidths=lw))
    if region_mask is not None:
        ax.add_collection(
            LineCollection(
                _triangle_edges(V, F[region_mask]), colors="tab:orange", linewidths=lw * 2
            )
        )
    U = V[np.unique(F)]
    pad = 0.03 * np.ptp(U[:, :2], 0).max()
    ax.set_xlim(U[:, 0].min() - pad, U[:, 0].max() + pad)
    ax.set_ylim(U[:, 1].max() + pad, U[:, 1].min() - pad)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_curves(series, path, title="", xlabel="frame", ylabel="", fps=None):
    """series: list of (label, values, plot kwargs). With fps the x axis is in seconds."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for name, y, kwargs in series:
        ax.plot(np.arange(len(y)) / (fps or 1), y, label=name, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("time (s)" if fps else xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, path)


# Interactive HTML


def plotly_mesh(V, F, scalar, path, title="", highlight=None):
    import plotly.graph_objects as go

    Xv = to_view(V)
    fig = go.Figure(
        go.Mesh3d(
            x=Xv[:, 0],
            y=Xv[:, 1],
            z=Xv[:, 2],
            i=F[:, 0],
            j=F[:, 1],
            k=F[:, 2],
            intensity=scalar,
            colorscale="Viridis",
            showscale=True,
            flatshading=True,
        )
    )
    if highlight is not None:
        fig.add_trace(
            go.Scatter3d(
                x=Xv[highlight, 0],
                y=Xv[highlight, 1],
                z=Xv[highlight, 2],
                mode="markers",
                marker=dict(size=3, color="red"),
                name="mouth region",
            )
        )
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False
        ),
    )
    fig.write_html(str(path), include_plotlyjs=True)


def plotly_points_animation(P_seq, magnitude, labels, path, title="Dynamic 3D lip point cloud"):
    """Animated point cloud with a frame slider. P_seq [T, N, 3], magnitude [T, N]."""
    import plotly.graph_objects as go

    Xs = to_view(P_seq)
    cmax = float(np.percentile(magnitude, 99)) + 1e-9

    def trace(i):
        return go.Scatter3d(
            x=Xs[i, :, 0],
            y=Xs[i, :, 1],
            z=Xs[i, :, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=magnitude[i],
                colorscale="Turbo",
                cmin=0,
                cmax=cmax,
                colorbar=dict(title="|disp| px"),
            ),
        )

    frames = [go.Frame(data=[trace(i)], name=str(i)) for i in range(len(Xs))]
    fig = go.Figure(data=[trace(0)], frames=frames)
    slider_steps = [
        dict(
            method="animate",
            label=labels[i],
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
        )
        for i in range(len(Xs))
    ]
    play = dict(
        label="Play",
        method="animate",
        args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)],
    )
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=0, y=-2.2, z=0.3)),
        ),
        sliders=[dict(steps=slider_steps)],
        updatemenus=[dict(type="buttons", buttons=[play])],
    )
    fig.write_html(str(path), include_plotlyjs=True)


# Video panels (OpenCV)


def _titled_panel(img, title, S):
    interpolation = cv2.INTER_CUBIC if img.shape[0] < S else cv2.INTER_AREA
    panel = cv2.resize(img, (S, S), interpolation=interpolation)
    cv2.rectangle(panel, (0, 0), (S, 22), (0, 0, 0), -1)
    cv2.putText(
        panel, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return panel


def pipeline_frame(panels, S=256):
    """Concatenate (title, HxWx3 uint8 RGB) panels horizontally."""
    return np.concatenate([_titled_panel(img, title, S) for title, img in panels], axis=1)


def draw_mesh_panel(img_size, V, F, region_tri, S):
    k = S / img_size
    canvas = np.full((S, S, 3), 20, np.uint8)
    tri = (V[:, :2] * k).astype(np.int32)[F]
    cv2.polylines(canvas, list(tri[~region_tri]), True, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.polylines(canvas, list(tri[region_tri]), True, (255, 160, 40), 1, cv2.LINE_AA)
    return canvas


def draw_points_panel(img_size, P, P_can, magnitude, center, S, zoom=2.4):
    """Zoomed mouth point cloud coloured by displacement magnitude."""
    k = S / img_size * zoom
    canvas = np.full((S, S, 3), 20, np.uint8)
    offset = np.array([S / 2, S / 2]) - center[:2] * k
    for p in (P_can[:, :2] * k + offset).astype(int):
        cv2.circle(canvas, tuple(p), 1, (90, 90, 90), -1)
    levels = np.clip(magnitude / 4.0 * 255, 0, 255).astype(np.uint8)[:, None]
    colors = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)[:, 0]
    for p, c in zip((P[:, :2] * k + offset).astype(int), colors):
        cv2.circle(canvas, tuple(p), 3, tuple(int(x) for x in c[::-1]), -1, cv2.LINE_AA)
    return canvas


def draw_gaussian_panel(img_size, pos, rgb, S):
    """Gaussian centres as 2x2 dots, far to near."""
    k = S / img_size
    canvas = np.full((S, S, 3), 20, np.uint8)
    order = np.argsort(-pos[:, 2])
    xy = (pos[order, :2] * k).astype(int)
    colors = (np.clip(rgb[order], 0, 1) * 255).astype(np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            x = np.clip(xy[:, 0] + dx, 0, S - 1)
            y = np.clip(xy[:, 1] + dy, 0, S - 1)
            canvas[y, x] = colors
    return canvas
