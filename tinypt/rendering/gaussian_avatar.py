"""Mesh-rigged 3D Gaussian head avatar with a perspective camera and relightable appearance.

Geometry: every Gaussian is bound to one triangle of the FLAME + mouth-proxy mesh by a
barycentric anchor, plus an offset, rotation and scale expressed in the triangle's local
frame (GaussianAvatars-style rigging). The avatar therefore follows any FLAME expression
or jaw pose, including the audio-driven motion of the generation step.

Projection: 3D covariance -> camera frame -> EWA splatting, Sigma_2D = J Sigma_cam J^T with
J the Jacobian of the perspective projection.

Appearance: colour = albedo * SH shading of the triangle normal in the camera frame, with
per-photo lighting L [9, 3]. Albedo is shared across photos and lighting is not, so the
avatar can be rendered under any lighting.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..flame.model import rodrigues
from .splatting import render as splat

C_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
# Offset bound in metres (in-plane, in-plane, along the normal). Unbounded offsets let
# Gaussians that few photos constrain drift off the surface.
MAX_OFFSET = [2e-3, 2e-3, 1e-3]
MAX_SCALE = 0.01  # metres


def logit(x):
    x = np.clip(x, 0.02, 0.98)
    return np.log(x / (1 - x))


def sh_basis_torch(N):
    x, y, z = N[:, 0], N[:, 1], N[:, 2]
    return torch.stack(
        [torch.ones_like(x), y, z, x, x * y, y * z, 3 * z**2 - 1, x * z, x**2 - y**2], 1
    )


class GaussianAvatar(nn.Module):
    def __init__(self, V_rest, faces, face_weight, vertex_albedo, n_gauss, seed=0):
        """V_rest [n, 3] rest mesh, face_weight [F] sampling weight per face (0 = no
        Gaussians), vertex_albedo [n, 3] initial colours."""
        super().__init__()
        rng = np.random.default_rng(seed)
        e1 = V_rest[faces[:, 1]] - V_rest[faces[:, 0]]
        e2 = V_rest[faces[:, 2]] - V_rest[faces[:, 0]]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        p = area * face_weight
        tri = rng.choice(len(faces), n_gauss, p=p / p.sum())
        u, v = rng.random(n_gauss), rng.random(n_gauss)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        bary = np.stack([1 - u - v, u, v], 1)
        albedo = (vertex_albedo[faces[tri]] * bary[..., None]).sum(1)
        size_rest = 0.5 * (np.linalg.norm(e1, axis=1) + np.linalg.norm(e2, axis=1))[tri]
        # Local mean spacing between Gaussians: denser sampling -> smaller initial scale.
        density = p[tri] / area[tri].clip(1e-12)
        spacing = np.sqrt((area * (face_weight > 0)).sum() / n_gauss * density.mean() / density)

        self.n = n_gauss
        self.register_buffer("tri", torch.tensor(tri, dtype=torch.long))
        self.register_buffer("bary", torch.tensor(bary, dtype=torch.float32))
        self.register_buffer("faces", torch.tensor(faces, dtype=torch.long))
        self.register_buffer("size_rest", torch.tensor(size_rest, dtype=torch.float32)[:, None])
        self.register_buffer("max_offset", torch.tensor(MAX_OFFSET))
        self.offset = nn.Parameter(torch.zeros(n_gauss, 3))  # raw value, bounded in offsets()
        log_scale = np.log(spacing)[:, None] + np.log([0.6, 0.6, 0.15])[None]
        self.log_scale = nn.Parameter(torch.tensor(log_scale, dtype=torch.float32))
        self.opacity_logit = nn.Parameter(torch.full((n_gauss,), 2.0))
        self.albedo_logit = nn.Parameter(torch.tensor(logit(albedo), dtype=torch.float32))

    def offsets(self):
        """Bounded offset in the local frame (metres at rest): max * tanh(raw / max)."""
        return self.max_offset * torch.tanh(self.offset / self.max_offset)

    def local_frames(self, V):
        """Anchor point, local frame (columns t1, t2, n) and size of each Gaussian's triangle."""
        f = self.faces[self.tri]
        v0, v1, v2 = V[f[:, 0]], V[f[:, 1]], V[f[:, 2]]
        e1, e2 = v1 - v0, v2 - v0
        n = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=1e-12)
        t1 = F.normalize(e1, dim=-1, eps=1e-12)
        t2 = torch.cross(n, t1, dim=-1)
        R = torch.stack([t1, t2, n], -1)
        size = (0.5 * (e1.norm(dim=-1) + e2.norm(dim=-1)))[:, None]
        anchor = (self.bary[:, :, None] * torch.stack([v0, v1, v2], 1)).sum(1)
        return anchor, R, size

    def project(self, V, rvec, t, f, cx, cy, L):
        """Project the Gaussians of the posed mesh V [n, 3] into a camera.

        Returns pixel positions with depth [G, 3], 2D covariances, shaded colours, opacities
        and albedo.
        """
        anchor, R, size = self.local_frames(V)
        stretch = size / self.size_rest  # follows the triangle's deformation
        pos = anchor + (R @ (self.offsets() * stretch)[..., None])[..., 0]
        scale = torch.exp(self.log_scale.clamp(max=math.log(MAX_SCALE))) * stretch
        R_cam = C_FLIP @ rodrigues(rvec[None])[0]
        Xc = pos @ R_cam.T + t
        M = R_cam @ R  # local frame -> camera frame
        cov_cam = (M * (scale**2)[:, None, :]) @ M.transpose(1, 2)
        x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2].clamp_min(1e-3)
        J = torch.zeros(len(z), 2, 3)
        J[:, 0, 0], J[:, 0, 2] = f / z, -f * x / z**2
        J[:, 1, 1], J[:, 1, 2] = f / z, -f * y / z**2
        cov_2d = J @ cov_cam @ J.transpose(1, 2) + 0.3 * torch.eye(2)  # EWA low-pass filter
        uv = torch.stack([f * x / z + cx, f * y / z + cy, z], 1)
        normal_cam = M[:, :, 2]
        shade = (sh_basis_torch(normal_cam) @ L).clamp_min(0.0)
        albedo = torch.sigmoid(self.albedo_logit)
        return uv, cov_2d, albedo * shade, torch.sigmoid(self.opacity_logit), albedo

    def render(self, V, cam, L, H, W, bg=None):
        uv, cov_2d, color, opacity, _ = self.project(
            V, cam["rvec"], cam["t"], cam["f"], cam["cx"], cam["cy"], L
        )
        bg = torch.zeros(H, W, 3) if bg is None else bg
        return splat(uv, cov_2d, color, opacity, H, W, bg)
