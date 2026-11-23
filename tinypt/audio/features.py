"""Audio feature extraction: MFCC (librosa) or wav2vec 2.0 (transformers)."""

import numpy as np

SAMPLE_RATE = 16000
WAV2VEC2_MODEL = "facebook/wav2vec2-base-960h"
WAV2VEC2_RATE = 50.0  # output frames per second
MFCC_RATE = 100.0


def wav2vec2_features(audio_list, device="cpu", name=WAV2VEC2_MODEL):
    """Frozen wav2vec 2.0 hidden states ([T, 768] at 50 Hz) for a list of 16 kHz waveforms."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
    model = Wav2Vec2Model.from_pretrained(name).to(device).eval()
    out = []
    with torch.no_grad():
        for x in audio_list:
            values = extractor(x, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
            out.append(model(values.to(device)).last_hidden_state[0].float().cpu().numpy())
    return out, WAV2VEC2_RATE


def mfcc_features(audio_list):
    """40 MFCCs + deltas ([T, 80] at 100 Hz) for a list of 16 kHz waveforms."""
    import librosa

    out = []
    for x in audio_list:
        mfcc = librosa.feature.mfcc(y=x, sr=SAMPLE_RATE, n_mfcc=40, hop_length=160, n_fft=400)
        delta = librosa.feature.delta(mfcc)
        out.append(np.concatenate([mfcc, delta]).T.astype(np.float32))
    return out, MFCC_RATE


def extract_features(audio_list, kind, device="cpu"):
    if kind == "wav2vec2":
        return wav2vec2_features(audio_list, device)
    return mfcc_features(audio_list)


def resample_features(feat, rate, n_frames, fps):
    """Linearly interpolate a [T, C] feature sequence sampled at `rate` onto n_frames at fps."""
    t_src = np.arange(len(feat)) / rate
    t_dst = np.arange(n_frames) / fps
    idx = np.clip(np.searchsorted(t_src, t_dst) - 1, 0, len(feat) - 2)
    w = np.clip((t_dst - t_src[idx]) * rate, 0, 1)[:, None]
    return ((1 - w) * feat[idx] + w * feat[idx + 1]).astype(np.float32)


def video_frame_features(wav_path, fps=25, kind="mfcc", sr=SAMPLE_RATE, n_frames=None):
    """Per-video-frame features of a wav file (used by the video baseline).

    mfcc: 26 MFCCs + deltas + delta-deltas + log energy (79-d), one window per frame.
    wav2vec2: 768-d hidden states interpolated to the frame rate.
    """
    import librosa

    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    y = y / (np.sqrt((y**2).mean()) + 1e-8) * 0.1  # RMS-normalise loudness
    hop = int(round(sr / fps))
    n_out = 1 + len(y) // hop
    if kind == "mfcc":
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=40, fmax=8000
        )
        log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=26)
        d1 = librosa.feature.delta(mfcc, width=5, mode="nearest")
        d2 = librosa.feature.delta(mfcc, order=2, width=5, mode="nearest")
        energy = librosa.power_to_db(
            librosa.feature.rms(y=y, frame_length=1024, hop_length=hop) ** 2 + 1e-10
        )
        feats = np.concatenate([mfcc, d1, d2, energy], 0).T
    elif kind == "wav2vec2":
        import torch
        from scipy.interpolate import interp1d
        from transformers import Wav2Vec2Model

        model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL).eval()
        x = torch.tensor((y - y.mean()) / (y.std() + 1e-7), dtype=torch.float32)[None]
        with torch.no_grad():
            hidden = model(x).last_hidden_state[0].numpy()
        t_in = np.arange(len(hidden)) * 0.02 + 0.0125
        t_out = np.arange(n_out) / fps
        feats = interp1d(
            t_in, hidden, axis=0, bounds_error=False, fill_value=(hidden[0], hidden[-1])
        )(t_out)
    else:
        raise ValueError(f"unknown audio feature: {kind}")
    feats = feats.astype(np.float32)
    if n_frames is None:
        return feats
    if len(feats) >= n_frames:
        return feats[:n_frames]
    return np.concatenate([feats, np.repeat(feats[-1:], n_frames - len(feats), 0)], 0)


def context_windows(feats, context):
    """[T, D] -> [T, 2 * context + 1, D] sliding windows with edge padding."""
    padded = np.pad(feats, ((context, context), (0, 0)), mode="edge")
    idx = np.arange(len(feats))[:, None] + np.arange(2 * context + 1)[None]
    return padded[idx]
