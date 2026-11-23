"""Pinhole camera, z-buffer triangle rasterizer (NumPy, CPU), inverse rendering of albedo and
spherical-harmonics lighting, and a procedural face albedo for synthetic test data.

Camera convention (OpenCV): X_cam = C @ R @ X_flame + t with C = diag(1, -1, -1), because
FLAME has y up and the face looking along +z while OpenCV has y down and the camera looking
along +z. Projection: u = f * x / z + cx, v = f * y / z + cy. R is the head rotation
(axis-angle `rvec`), t the head position in camera coordinates (t_z = distance in metres).
"""

import cv2
import numpy as np

from ..flame.model import N_FLAME_VERTICES, N_HEAD_VERTICES

C_FLIP = np.diag([1.0, -1.0, -1.0])


# Camera


def rotmat(rvec):
    return cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]


def to_camera(X, rvec, t):
    return X @ (C_FLIP @ rotmat(rvec)).T + np.asarray(t, np.float64)


def project(Xc, f, cx, cy):
    z = Xc[:, 2]
    return np.stack([f * Xc[:, 0] / z + cx, f * Xc[:, 1] / z + cy], 1), z


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for i in range(3):
        np.add.at(vn, F[:, i], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def face_area(uv, F):
    """Signed 2D area per face; negative = front facing (FLAME faces are counter-clockwise
    seen from outside)."""
    x, y = uv[F, 0], uv[F, 1]
    return (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])


# Rasterization


def rasterize(uv, z, F, H, W, cull=True):
    """Z-buffer rasterization.

    Returns the face id per pixel [H, W] (-1 = background), barycentrics [H, W, 3] and
    depth [H, W]. With cull=False back faces are kept; seen through an open mouth they are
    the mouth interior.
    """
    fid = np.full((H, W), -1, np.int64)
    bary = np.zeros((H, W, 3), np.float32)
    zbuf = np.full((H, W), np.inf)
    x, y, zz = uv[F, 0], uv[F, 1], z[F]  # [nF, 3]
    area = face_area(uv, F)
    x0 = np.maximum(np.floor(x.min(1) - 0.5), 0).astype(int)
    x1 = np.minimum(np.ceil(x.max(1) - 0.5), W - 1).astype(int)
    y0 = np.maximum(np.floor(y.min(1) - 0.5), 0).astype(int)
    y1 = np.minimum(np.ceil(y.max(1) - 0.5), H - 1).astype(int)
    drawable = ((area < 0) if cull else (area != 0)) & (x0 <= x1) & (y0 <= y1) & (zz.min(1) > 0)
    for i in np.where(drawable)[0]:
        gx, gy = np.meshgrid(np.arange(x0[i], x1[i] + 1) + 0.5, np.arange(y0[i], y1[i] + 1) + 0.5)
        X, Y = x[i], y[i]
        w0 = ((X[1] - gx) * (Y[2] - gy) - (X[2] - gx) * (Y[1] - gy)) / area[i]
        w1 = ((X[2] - gx) * (Y[0] - gy) - (X[0] - gx) * (Y[2] - gy)) / area[i]
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        depth = w0 * zz[i, 0] + w1 * zz[i, 1] + w2 * zz[i, 2]
        rows, cols = slice(y0[i], y1[i] + 1), slice(x0[i], x1[i] + 1)
        zbuf_tile = zbuf[rows, cols]
        inside &= depth < zbuf_tile
        if not inside.any():
            continue
        zbuf_tile[inside] = depth[inside]
        fid[rows, cols][inside] = i
        bary[rows, cols][inside] = np.stack([w0[inside], w1[inside], w2[inside]], 1)
    return fid, bary, zbuf


def interpolate(attr, F, fid, bary):
    """Per-vertex attribute [n, c] -> image [H, W, c] (zero on the background)."""
    out = np.zeros(fid.shape + (attr.shape[1],), np.float32)
    covered = fid >= 0
    out[covered] = (attr[F[fid[covered]]] * bary[covered][..., None]).sum(1)
    return out


def render(
    V,
    F,
    rvec,
    t,
    f,
    H,
    W,
    colors,
    light=(-0.3, 0.45, 1.0),
    ambient=0.35,
    bg=None,
    mouth_color=(0.13, 0.05, 0.05),
):
    """Lambertian render of a per-vertex coloured mesh with a shadow-mapped key light.

    `light` is the direction the light travels in camera coordinates. Back faces (the inside
    of the head, visible through the open mouth) are drawn as a dark mouth interior.
    Returns the uint8 image, face ids (back faces included), barycentrics and depth.
    """
    Xc = to_camera(V, rvec, t)
    uv, z = project(Xc, f, W / 2, H / 2)
    fid, bary, zbuf = rasterize(uv, z, F, H, W, cull=False)
    N = vertex_normals(Xc, F)
    L = -np.asarray(light, np.float64)
    L /= np.linalg.norm(L)
    n_img = interpolate(N, F, fid, bary)
    n_img /= np.linalg.norm(n_img, axis=-1, keepdims=True) + 1e-9
    lit = visibility(Xc, F, L, res=768)[0].astype(np.float64)
    lit_img = np.clip(interpolate(lit[:, None], F, fid, bary)[..., 0], 0, 1)
    shade = ambient + (1 - ambient) * np.clip(n_img @ L, 0, 1) * lit_img
    img = interpolate(np.asarray(colors, np.float64), F, fid, bary) * shade[..., None]
    background_img = np.full((H, W, 3), 0.9) if bg is None else bg
    is_back = np.zeros(fid.shape, bool)
    is_back[fid >= 0] = face_area(uv, F)[fid[fid >= 0]] > 0
    img = np.where(is_back[..., None], np.asarray(mouth_color), img)
    img = np.where((fid >= 0)[..., None], img, background_img)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), fid, bary, zbuf


def background(H, W, rng):
    """Random vertical colour gradient."""
    c0, c1 = rng.uniform(0.35, 0.95, 3), rng.uniform(0.35, 0.95, 3)
    g = np.linspace(0, 1, H)[:, None, None]
    return np.broadcast_to(c0 * (1 - g) + c1 * g, (H, W, 3)).copy()


# Visibility and ambient occlusion


def fibonacci_dirs(n):
    """n roughly uniform unit directions on the sphere."""
    k = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * k / n), np.pi * (1 + 5**0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def visibility(V, F, dirs, res=512, tol=1e-3, normal_offset=2e-3):
    """[D, n] bool: is each vertex visible from far away along each unit direction?

    Uses an orthographic depth map per direction (shadow mapping). Lookups are offset along
    the vertex normal to avoid self-shadowing artefacts.
    """
    c = V.mean(0)
    out = []
    scale = (res - 8) / (2 * np.linalg.norm(V - c, axis=1).max())
    Q = V - c + normal_offset * vertex_normals(V, F)
    for d in np.atleast_2d(dirs):
        a = np.cross(d, [0, 1, 0] if abs(d[1]) < 0.9 else [1, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        X = V - c
        uv = np.stack([X @ a * scale + res / 2, X @ b * scale + res / 2], 1)
        z = 1.0 - (X @ d)  # positive; smaller = closer to the light
        _, _, zbuf = rasterize(uv, z, F, res, res, cull=False)
        px = np.clip(
            np.floor(np.stack([Q @ a, Q @ b], 1) * scale + res / 2).astype(int), 0, res - 1
        )
        out.append(1.0 - (Q @ d) <= zbuf[px[:, 1], px[:, 0]] + tol)
    return np.array(out)


def ambient_occlusion(V, F, n_dirs=128, res=512, tol=1.5e-3):
    """Per-vertex ambient occlusion in [0, 1] (1 = unoccluded): cosine-weighted fraction of
    the normal hemisphere from which the vertex is visible."""
    N = vertex_normals(V, F)
    dirs = fibonacci_dirs(n_dirs)
    vis = visibility(V, F, dirs, res, tol)
    cos = np.clip(N @ dirs.T, 0, None)  # [n, D]
    return np.clip((vis.T * cos).sum(1) / np.maximum(cos.sum(1), 1e-9), 0, 1)


# Procedural albedo for synthetic identities


def procedural_albedo(V_tmp, F, lips_idx, rng):
    """Per-vertex albedo painted on the FLAME template: skin with low-frequency variation,
    lips, eyebrows, eyelids, sclera / iris / pupil, baked ambient occlusion and nostrils.

    Defined on the template, so it transfers to any FLAME identity and expression.
    """
    n = len(V_tmp)
    head = np.arange(N_HEAD_VERTICES)
    tone = rng.uniform(0.75, 1.05)
    col = np.tile(np.array([0.82, 0.63, 0.52]) * tone, (n, 1))
    for _ in range(6):  # low-frequency skin variation
        k = rng.normal(0, 25, 3)
        col *= 1 + 0.03 * np.sin(V_tmp @ k + rng.uniform(0, 6.3))[:, None]

    eyes = [np.arange(N_HEAD_VERTICES, 4477), np.arange(4477, N_FLAME_VERTICES)]
    for e in eyes:  # cheek blush below each eye
        c = V_tmp[e].mean(0)
        d = np.linalg.norm((V_tmp - (c + [0.004 * np.sign(c[0]), -0.035, 0.0]))[:, :2], axis=1)
        col *= 1 + (np.array([0.06, -0.04, -0.03]) * np.exp(-((d / 0.018) ** 2))[:, None])

    lips = np.zeros(n)
    lips[lips_idx] = 1.0
    lip_col = np.array([0.70, 0.36, 0.37]) * tone
    col = col * (1 - lips[:, None]) + lip_col * lips[:, None]

    brow_col = np.array([0.22, 0.15, 0.11]) * rng.uniform(0.7, 1.3)
    for e in eyes:
        c = V_tmp[e].mean(0)
        dx = (V_tmp[head, 0] - c[0]) * np.sign(c[0])  # positive towards the temple
        arch = (
            c[1]
            + 0.019
            + 0.004 * np.cos(np.clip(dx / 0.025, -1.5, 1.5) * 1.2)
            - 0.006 * (dx > 0.012) * (dx - 0.012) / 0.01
        )
        w = (
            np.exp(-(((V_tmp[head, 1] - arch) / 0.0035) ** 4))
            * ((dx > -0.017) & (dx < 0.027))
            * (V_tmp[head, 2] > c[2] - 0.004)
        )
        col[head] = col[head] * (1 - w[:, None]) + brow_col * w[:, None]

    iris_col = rng.choice(
        [np.array([0.30, 0.19, 0.10]), np.array([0.25, 0.35, 0.45]), np.array([0.25, 0.30, 0.18])]
    )
    lash_col = np.array([0.10, 0.07, 0.05])
    for e in eyes:
        c = V_tmp[e].mean(0)
        d = V_tmp[e] - c
        r_eye = np.median(np.linalg.norm(d, axis=1))
        ang = np.degrees(np.arccos(np.clip(d[:, 2] / np.linalg.norm(d, axis=1), -1, 1)))
        upper_lid_shadow = 1 - 0.35 * np.clip(d[:, 1] / r_eye, 0, 1)
        sclera = np.array([0.86, 0.82, 0.78]) * upper_lid_shadow[:, None]
        limbus = np.clip((ang - 20) / 6, 0, 1)[:, None]  # iris darkens towards its rim
        iris = iris_col * (1 - 0.45 * limbus) * (1 + 0.25 * np.clip((15 - ang) / 15, 0, 1))[:, None]
        col[e] = np.where(ang[:, None] < 9, 0.03, np.where(ang[:, None] < 27, iris, sclera))
        # Eyelid margin (skin hugging the eyeball): lash line above, soft shadow below.
        dist_head = np.linalg.norm(V_tmp[head] - c, axis=1)
        rim = np.clip(1 - (dist_head - r_eye) / 0.0035, 0, 1) * (V_tmp[head, 2] > c[2] - 0.004)
        upper = V_tmp[head, 1] > c[1] - 0.002
        w = np.where(upper, 0.85 * rim, 0.35 * rim)[:, None]
        target = np.where(upper[:, None], lash_col, col[head] * 0.7)
        col[head] = col[head] * (1 - w) + target * w

    # The rasterizer casts no shadows from the key light onto concave regions, so ambient
    # occlusion is baked into the albedo (nostrils, mouth corners, eye sockets, ears).
    ao = ambient_occlusion(V_tmp, F)
    col *= (0.3 + 0.7 * ao)[:, None]

    # FLAME nostrils are shallow dimples; darken the recessed vertices below the nose tip.
    tip = head[np.argmax(V_tmp[head, 2] * (np.abs(V_tmp[head, 0]) < 0.01))]
    d = V_tmp[head] - V_tmp[tip]
    box = (np.abs(d[:, 0]) < 0.02) & (d[:, 1] < 0) & (d[:, 1] > -0.02) & (d[:, 2] > -0.035)
    w = box * np.clip((0.75 - ao[head]) / 0.3, 0, 1)
    col[head] *= (1 - 0.75 * w)[:, None]
    return np.clip(col, 0, 1)


# Inverse rendering: albedo and per-photo spherical-harmonics lighting


def sh_basis(N):
    """Order-2 real spherical harmonics (9 terms, constants folded into the coefficients)
    of unit normals [n, 3]."""
    x, y, z = N[:, 0], N[:, 1], N[:, 2]
    return np.stack([np.ones_like(x), y, z, x, x * y, y * z, 3 * z**2 - 1, x * z, x**2 - y**2], 1)


def fit_sh(colors, albedo, Y, w, reg=1e-3):
    """Lighting L [9, 3] with colors ~= albedo * (Y @ L), weighted least squares per channel."""
    L = np.zeros((9, 3))
    for ch in range(3):
        A = Y * albedo[:, ch : ch + 1]
        Aw = A * w[:, None]
        L[:, ch] = np.linalg.solve(A.T @ Aw + reg * np.eye(9), Aw.T @ colors[:, ch])
    return L


def albedo_and_lighting(cols, Ys, wts, iters=15, reg=1e-3):
    """Alternating least squares for a shared albedo and per-photo SH lighting.

    cols [P, n, 3] observed vertex colours, Ys [P, n, 9] SH bases of the camera-frame
    normals, wts [P, n] visibility and view-angle weights. The albedo / lighting scale
    ambiguity is fixed by setting the mean shading of the first photo to 1.
    Returns albedo [n, 3], lighting [P, 9, 3] and shading [P, n, 3].
    """
    P = len(cols)
    albedo = (cols * wts[..., None]).sum(0) / np.maximum(wts.sum(0), 1e-6)[:, None]
    for _ in range(iters):
        Ls = np.array([fit_sh(cols[i], albedo, Ys[i], wts[i], reg) for i in range(P)])
        shade = np.einsum("pnk,pkc->pnc", Ys, Ls)
        num = (wts[..., None] * cols * shade).sum(0)
        den = (wts[..., None] * shade**2).sum(0)
        albedo = np.where(den > 1e-6, num / np.maximum(den, 1e-6), albedo)
        k = np.average(shade[0], axis=0, weights=np.maximum(wts[0], 1e-9))
        albedo, Ls = albedo * k, Ls / k
    shade = np.einsum("pnk,pkc->pnc", Ys, Ls)
    return np.clip(albedo, 0, 1.5), Ls, shade


def sample_vertex_colors(img, Xc, F, f, cx, cy, n_head=N_HEAD_VERTICES, depth_tol=0.002):
    """Colour of every vertex in one photo, with its weight and the SH basis of its normal.

    The weight is visibility (z-buffer test) times a falloff that drops grazing views,
    which would otherwise bleed background colour in at the silhouette.
    img is float [H, W, 3] in [0, 1]; Xc are the vertices in camera coordinates.
    Returns (colours, weights, SH basis, (face ids, barycentrics, uv)).
    """
    H, W = img.shape[:2]
    uv, z = project(Xc, f, cx, cy)
    fid, bary, zbuf = rasterize(uv, z, F, H, W)
    px = np.clip(np.floor(uv).astype(int), [0, 0], [W - 1, H - 1])
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    visible = inside & (z <= zbuf[px[:, 1], px[:, 0]] + depth_tol)
    N = vertex_normals(Xc, F)
    cos_view = np.clip((N * -Xc / np.linalg.norm(Xc, axis=1, keepdims=True)).sum(1), 0, 1)
    w = visible * np.clip((cos_view - 0.25) / 0.75, 0, 1) ** 2
    w[n_head:] *= 0.3  # eyeballs: gaze is not modelled
    colors = cv2.remap(
        img.astype(np.float32),
        (uv[:, 0] - 0.5).astype(np.float32)[None],
        (uv[:, 1] - 0.5).astype(np.float32)[None],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )[0]
    return colors, w, sh_basis(N), (fid, bary, uv)
