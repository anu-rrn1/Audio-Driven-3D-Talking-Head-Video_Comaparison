"""Mesh-rigged 3D Gaussians for an orthographic camera (pixel units).

Every Gaussian is bound to one triangle by barycentric coordinates and moves and rotates
with the triangle's local frame. Colour = base colour + mouth openness * delta colour, so
the mouth interior can change appearance as the lips open.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry.mesh import face_normals_areas
from ..rendering.gaussian_avatar import logit


class RiggedGaussians(nn.Module):
    def __init__(self, V_can, faces, n_gauss, region_tri_mask, rgb_fn, mouth_frac=0.25, seed=0):
        """`mouth_frac` of the Gaussians are placed on the mouth-region triangles.
        rgb_fn maps positions [N, 3] to initial colours [N, 3] in [0, 1]."""
        super().__init__()
        rng = np.random.default_rng(seed)
        _, area = face_normals_areas(V_can, faces)
        n_mouth = int(n_gauss * mouth_frac)
        all_tri = np.arange(len(faces))
        region_tri = np.where(region_tri_mask)[0]
        tri = np.concatenate(
            [
                rng.choice(all_tri, n_gauss - n_mouth, p=area / area.sum()),
                rng.choice(region_tri, n_mouth, p=area[region_tri] / area[region_tri].sum()),
            ]
        )
        u, v = rng.random(n_gauss), rng.random(n_gauss)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        bary = np.stack([1 - u - v, u, v], 1)
        pos0 = (V_can[faces[tri]] * bary[..., None]).sum(1)

        self.n = n_gauss
        self.register_buffer("tri", torch.tensor(tri, dtype=torch.long))
        self.register_buffer("bary", torch.tensor(bary, dtype=torch.float32))
        self.register_buffer("faces", torch.tensor(faces, dtype=torch.long))
        self.offset = nn.Parameter(torch.zeros(n_gauss, 3))  # local frame, pixels
        log_scale = torch.tensor([math.log(0.45), math.log(0.45), math.log(0.12)])
        self.log_scale = nn.Parameter(log_scale[None].repeat(n_gauss, 1))  # x triangle size
        self.opacity_logit = nn.Parameter(torch.full((n_gauss,), 1.0))
        self.c0 = nn.Parameter(torch.tensor(logit(rgb_fn(pos0)), dtype=torch.float32))
        self.c1 = nn.Parameter(torch.zeros(n_gauss, 3))  # openness-dependent colour

    def params(self, V, openness):
        """V [Nv, 3] mesh vertices in pixels -> position [G, 3], 2D covariance [G, 2, 2],
        colour [G, 3], opacity [G]."""
        f = self.faces[self.tri]
        v0, v1, v2 = V[f[:, 0]], V[f[:, 1]], V[f[:, 2]]
        b = self.bary
        pos = b[:, :1] * v0 + b[:, 1:2] * v1 + b[:, 2:3] * v2
        e1, e2 = v1 - v0, v2 - v0
        n = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=1e-8)
        t1 = F.normalize(e1, dim=-1, eps=1e-8)
        t2 = torch.cross(n, t1, dim=-1)
        R = torch.stack([t1, t2, n], dim=-1)  # columns = local axes
        size = 0.5 * (e1.norm(dim=-1) + e2.norm(dim=-1))[:, None]
        pos = pos + (R @ self.offset[..., None]).squeeze(-1)
        scale = torch.exp(self.log_scale.clamp(-3.0, 0.7)) * size.clamp_min(0.5)
        R2 = R[:, :2, :]  # orthographic projection
        cov = (R2 * (scale**2)[:, None, :]) @ R2.transpose(1, 2)
        cov = cov + 0.3 * torch.eye(2)  # EWA low-pass filter
        color = torch.sigmoid(self.c0 + openness * self.c1)
        return pos, cov, color, torch.sigmoid(self.opacity_logit)


def load_rigged_gaussians(ckpt, V_can, faces, region_tri):
    """Rebuild the model from a checkpoint; the colour initialiser is overwritten by the
    stored parameters."""
    model = RiggedGaussians(
        V_can,
        faces,
        ckpt["n_gauss"],
        region_tri,
        lambda p: np.full((len(p), 3), 0.5),
        ckpt["mouth_frac"],
    )
    model.load_state_dict(ckpt["state"])
    return model.eval()
