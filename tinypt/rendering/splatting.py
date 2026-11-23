"""Differentiable tile-based alpha compositing of projected 2D Gaussians (PyTorch, CPU)."""

import math

import torch


def _composite_tile(pix, pos, inv00, inv01, inv11, color, opacity, bg):
    """Front-to-back compositing of depth-sorted Gaussians over the pixels `pix` [P, 2]."""
    dx = pix[None, :, 0] - pos[:, None, 0]
    dy = pix[None, :, 1] - pos[:, None, 1]
    power = -0.5 * (
        inv00[:, None] * dx * dx + 2 * inv01[:, None] * dx * dy + inv11[:, None] * dy * dy
    )
    alpha = (opacity[:, None] * torch.exp(power)).clamp(max=0.99)
    alpha = torch.where(alpha >= 1.0 / 255.0, alpha, torch.zeros_like(alpha))
    log_one_minus = torch.log1p(-alpha)
    cumulative = torch.cumsum(log_one_minus, dim=0)
    transmittance = torch.exp(cumulative - log_one_minus)  # before each Gaussian
    out = (alpha * transmittance).T @ color  # [P, 3]
    return out + torch.exp(cumulative[-1])[:, None] * bg


def render(pos, cov, color, opacity, H, W, bg, tile=16):
    """Render Gaussians over the background `bg` [H, W, 3].

    pos [G, 3] holds pixel coordinates and depth, cov [G, 2, 2] the 2D covariances.
    Gaussians are sorted by depth; each tile only processes the Gaussians whose 3.5-sigma
    footprint overlaps it, as in the 3DGS rasterizer. Differentiable in all inputs.
    """
    order = torch.argsort(pos[:, 2])  # near (small z) first
    pos, cov, color, opacity = pos[order], cov[order], color[order], opacity[order]
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = (a * c - b**2).clamp_min(1e-6)
    inv00, inv01, inv11 = c / det, -b / det, a / det
    with torch.no_grad():
        max_eig = 0.5 * (a + c) + torch.sqrt((0.5 * (a - c)) ** 2 + b**2)
        radius = 3.5 * torch.sqrt(max_eig)
        px, py = pos[:, 0].detach(), pos[:, 1].detach()
        x0 = int(max(0, math.floor((px - radius).min().item())))
        x1 = int(min(W, math.ceil((px + radius).max().item())))
        y0 = int(max(0, math.floor((py - radius).min().item())))
        y1 = int(min(H, math.ceil((py + radius).max().item())))
    if x1 <= x0 or y1 <= y0:
        return bg

    rows = []
    for ty in range(y0, y1, tile):
        ty1 = min(ty + tile, y1)
        row = []
        for tx in range(x0, x1, tile):
            tx1 = min(tx + tile, x1)
            with torch.no_grad():
                overlaps = (
                    (px > tx - radius)
                    & (px < tx1 + radius)
                    & (py > ty - radius)
                    & (py < ty1 + radius)
                )
                idx = overlaps.nonzero().squeeze(1)
            bg_tile = bg[ty:ty1, tx:tx1]
            if len(idx) == 0:
                row.append(bg_tile)
                continue
            ys, xs = torch.meshgrid(
                torch.arange(ty, ty1) + 0.5, torch.arange(tx, tx1) + 0.5, indexing="ij"
            )
            pix = torch.stack([xs, ys], -1).reshape(-1, 2).float()
            rgb = _composite_tile(
                pix,
                pos[idx],
                inv00[idx],
                inv01[idx],
                inv11[idx],
                color[idx],
                opacity[idx],
                bg_tile.reshape(-1, 3),
            )
            row.append(rgb.reshape(ty1 - ty, tx1 - tx, 3))
        rows.append(torch.cat(row, dim=1))
    img = bg.clone()
    img[y0:y1, x0:x1] = torch.cat(rows, dim=0)
    return img
