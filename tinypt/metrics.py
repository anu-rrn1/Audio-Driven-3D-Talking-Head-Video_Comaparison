"""Image quality metrics."""

import numpy as np
import torch
import torch.nn.functional as F


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR of two images in [0, 1]."""
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse < 1e-12 else float(10 * np.log10(1.0 / mse))


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(10 * np.log10(1 / ((a[mask] - b[mask]) ** 2).mean()))


def ssim_torch(a: torch.Tensor, b: torch.Tensor, win: int = 7) -> torch.Tensor:
    """Differentiable SSIM with a uniform window, for [H, W, 3] tensors in [0, 1]."""
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    pad = win // 2
    mu_a = F.avg_pool2d(a, win, 1, pad)
    mu_b = F.avg_pool2d(b, win, 1, pad)
    var_a = F.avg_pool2d(a * a, win, 1, pad) - mu_a**2
    var_b = F.avg_pool2d(b * b, win, 1, pad) - mu_b**2
    cov_ab = F.avg_pool2d(a * b, win, 1, pad) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return (num / den).mean()
