"""Audio-conditioned lip point-cloud deformation network (about 150k parameters).

    canonical lip cloud -> EdgeConv point encoder -----\
                                                        cross-attention -> per-point 3D offset
    audio window        -> conv + BiGRU audio encoder -/
"""

import torch
import torch.nn as nn


class EdgeConv(nn.Module):
    """Local point feature: max_j MLP([x_i, x_j - x_i]) over the k nearest neighbours."""

    def __init__(self, cin, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * cin, cout), nn.GELU(), nn.Linear(cout, cout))

    def forward(self, x, idx):  # x [N, C], idx [N, k]
        xj = x[idx]
        xi = x[:, None].expand_as(xj)
        return self.mlp(torch.cat([xi, xj - xi], -1)).max(1).values


class PointEncoder(nn.Module):
    """Two EdgeConv layers (local geometry) and a global max-pool (shape context)."""

    def __init__(self, cin=6, dim=64):
        super().__init__()
        self.e1, self.e2 = EdgeConv(cin, dim), EdgeConv(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.out = nn.Linear(2 * dim, dim)

    def forward(self, x, idx):
        h = self.e2(self.e1(x, idx), idx)
        g = self.g(h).max(0).values
        return self.out(torch.cat([h, g.expand_as(h)], -1))


class AudioEncoder(nn.Module):
    """Temporal encoder over a window of L per-frame audio features."""

    def __init__(self, in_dim, dim, L):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.pos = nn.Parameter(torch.zeros(1, L, dim))

    def forward(self, x):  # [B, L, D]
        h = self.inp(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        h = h + self.pos
        return h, h[:, h.shape[1] // 2]  # sequence and centre-frame summary


class LipDeformNet(nn.Module):
    """Lip points x audio window -> per-point 3D displacement. Points are the attention
    queries, audio frames the keys and values."""

    def __init__(self, audio_dim, pt_in=6, dim=64, L=13, heads=4):
        super().__init__()
        self.pt_enc = PointEncoder(pt_in, dim)
        self.aud_enc = AudioEncoder(audio_dim, dim, L)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.glob = nn.Linear(dim, dim)
        self.head = nn.Sequential(
            nn.Linear(2 * dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3)
        )

    def forward(self, audio, pts, knn, return_attn=False):
        B, N = audio.shape[0], pts.shape[0]
        q = self.pt_enc(pts, knn)[None].expand(B, -1, -1)  # [B, N, dim]
        audio_seq, audio_summary = self.aud_enc(audio)  # [B, L, dim], [B, dim]
        attended, weights = self.attn(
            q, audio_seq, audio_seq, need_weights=return_attn, average_attn_weights=True
        )
        h = self.norm(q + attended)
        h = torch.cat([h, self.glob(audio_summary)[:, None].expand(-1, N, -1)], -1)
        out = self.head(h)
        return (out, weights) if return_attn else out
