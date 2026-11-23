"""Access to the compact VOCASET motion dataset written by the motion-basis step.

Motion is stored as per-vertex displacements from each speaker's neutral template,
compressed with a PCA basis learned on the training speakers:

    V_frame ~= template + mean + coeffs @ basis      (basis: K x 15069, 15069 = 5023 x 3)

The displacements live on FLAME vertices, so they can drive any FLAME-topology identity.
"""

import numpy as np

AUDIO_SR = 16000
DATASET_FPS = 60

# FaceFormer / CodeTalker speaker split.
SPLITS = {
    "train": [
        "FaceTalk_170728_03272_TA",
        "FaceTalk_170904_00128_TA",
        "FaceTalk_170725_00137_TA",
        "FaceTalk_170915_00223_TA",
        "FaceTalk_170811_03274_TA",
        "FaceTalk_170913_03279_TA",
        "FaceTalk_170904_03276_TA",
        "FaceTalk_170912_03278_TA",
    ],
    "val": ["FaceTalk_170811_03275_TA", "FaceTalk_170908_03277_TA"],
    "test": ["FaceTalk_170809_00138_TA", "FaceTalk_170731_00024_TA"],
}


class MotionBasis:
    """PCA motion basis: coefficients [T, K] <-> FLAME vertex displacements [T, 5023, 3]."""

    def __init__(self, data):
        self.basis = data["basis"]
        self.mean = data["mean"]
        self.lips = data["lips"]

    def displacements(self, coeffs):
        return (coeffs @ self.basis + self.mean).reshape(len(coeffs), -1, 3)


def iter_sequences(data, splits=None, speaker=None, sentences=None):
    """Yield the sequences that have audio, as dicts with keys spk, split, sent, coeffs
    (at 30 fps, every second dataset frame) and audio (float32 in [-1, 1], 16 kHz).

    Optional filters: split names, a speaker-name substring and sentence numbers.
    """
    subjects = [str(s) for s in data["subjects"]]
    split_of = [str(s) for s in data["split"]]
    coeffs, frames, starts = data["coeffs"], data["seq_frames"], data["seq_start"]
    audio, offsets = data["audio"], data["audio_offsets"]
    for i, (si, sent, has_audio) in enumerate(data["seq"]):
        if not has_audio:
            continue
        if splits is not None and split_of[si] not in splits:
            continue
        if speaker is not None and speaker not in subjects[si]:
            continue
        if sentences is not None and int(sent) not in sentences:
            continue
        yield dict(
            spk=subjects[si],
            split=split_of[si],
            sent=int(sent),
            coeffs=coeffs[frames[starts[i] : starts[i + 1]]][::2],
            audio=audio[offsets[i] : offsets[i + 1]].astype(np.float32) / 32768.0,
        )
