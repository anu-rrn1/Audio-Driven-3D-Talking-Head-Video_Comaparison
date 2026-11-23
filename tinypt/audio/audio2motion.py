"""Audio-to-motion network: speech features -> per-frame PCA coefficients of FLAME vertex
displacements (see tinypt.audio.vocaset for the motion representation).

    audio features [B, T, C] -> linear + 1D conv (local co-articulation context)
        -> + speaker style token + sinusoidal positions
        -> non-causal Transformer encoder -> PCA coefficients [B, T, K]

One learned style token per training speaker plus a "neutral" token (the last index),
which is used for unseen speakers and for new audio.
"""

import math

import torch
import torch.nn as nn

from .features import extract_features, resample_features

MOTION_FPS = 30


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]]


class Audio2Motion(nn.Module):
    def __init__(self, in_dim, k=128, n_styles=8, d=256, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.n_styles = n_styles
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Conv1d(d, d, 5, padding=2)
        self.style = nn.Embedding(n_styles + 1, d)
        self.pos = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, k)
        nn.init.zeros_(self.out.weight)  # start at the mean pose
        nn.init.zeros_(self.out.bias)
        # PCA coefficients differ in scale by orders of magnitude: the network predicts
        # standardised values that are rescaled with these training statistics.
        self.register_buffer("c_mean", torch.zeros(k))
        self.register_buffer("c_std", torch.ones(k))

    @property
    def neutral_style(self):
        return self.n_styles

    def forward(self, feat, style, pad_mask=None):
        h = self.inp(feat)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.pos(h + self.style(style)[:, None])
        return self.c_mean + self.c_std * self.out(self.enc(h, src_key_padding_mask=pad_mask))


def load_audio2motion(path, device="cpu"):
    """Load a checkpoint written by the training script. Returns (model in eval mode, ckpt)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = Audio2Motion(
        ckpt["in_dim"],
        ckpt["k"],
        len(ckpt["train_spk"]),
        args.get("d", 256),
        args.get("layers", 4),
        dropout=args.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


@torch.no_grad()
def predict_coefficients(model, feats):
    """PCA coefficients [T, K] for features already resampled to MOTION_FPS, neutral style."""
    x = torch.tensor(feats)[None]
    return model(x, torch.tensor([model.neutral_style]))[0].numpy()


def predict_from_audio(model, audio_list, n_frames_list, feature="wav2vec2"):
    """Predict coefficients for several 16 kHz waveforms (features are extracted in one pass)."""
    raw, rate = extract_features(audio_list, feature)
    return [
        predict_coefficients(model, resample_features(r, rate, n, MOTION_FPS))
        for r, n in zip(raw, n_frames_list)
    ]


def apply_gain(coeffs, c_mean, gain):
    """Articulation gain around the training mean pose: c' = c_mean + gain * (c - c_mean)."""
    return c_mean + gain * (coeffs - c_mean)
