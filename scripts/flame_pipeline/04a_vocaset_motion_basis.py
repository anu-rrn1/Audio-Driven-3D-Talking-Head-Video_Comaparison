"""Step 4a: VOCASET -> compact motion dataset with a PCA displacement basis.

VOCASET stores raw vertex positions in FLAME topology (123k frames x 5023 x 3, float64,
about 14 GB). Motion is represented as per-vertex displacements from each speaker's neutral
template, compressed with a PCA basis learned only on the training speakers:

    V_frame ~= template + mean + coeffs @ basis      (basis: K x 15069, K = 128)

FLAME's 100 expression blendshapes were measured as an alternative: on the unseen test
speakers they leave a 3.3 mm mean worst-lip-vertex error (inner-lip roll and press lie
outside their span), while PCA-128 leaves about 0.8 mm.

    pass 1: stream the file, take every n-th training frame -> randomized SVD -> basis
    pass 2: stream again, encode every frame, record the reconstruction error (and the
            FLAME-100 least-squares error on the test speakers for comparison)

Outputs: data/vocaset_motion/vocaset_motion.npz, outputs/flame_pipeline/04a_motion_basis/

Usage:
    python scripts/flame_pipeline/04a_vocaset_motion_basis.py [--smoke]
"""

import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import resample_poly

from tinypt import visualization as viz
from tinypt.audio.vocaset import AUDIO_SR, DATASET_FPS, SPLITS
from tinypt.config import DATA_DIR, FLAME_OUTPUTS, ensure_dir, load_config, seed_everything
from tinypt.flame import load_flame
from tinypt.io_utils import save_json

N_FLAME_EXPRESSIONS = 100


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vocaset", default=str(DATA_DIR / "vocaset"))
    parser.add_argument("--k", type=int, default=128, help="number of PCA components")
    parser.add_argument(
        "--sample_step", type=int, default=4, help="every n-th training frame enters the SVD"
    )
    parser.add_argument("--chunk", type=int, default=2048, help="frames read per chunk")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="quick check on every 30th chunk, written to smoke_test.npz",
    )
    parser.add_argument("--out", default=str(DATA_DIR / "vocaset_motion" / "vocaset_motion.npz"))
    return parser.parse_args()


class Logger:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, *msg):
        print(f"[{time.time() - self.t0:6.0f}s]", *msg, flush=True)


class VocasetIndex:
    """Speakers, sequences and per-frame speaker ids of the raw VOCASET files."""

    def __init__(self, vocaset_dir):
        self.dir = Path(vocaset_dir)
        self.verts = np.load(self.dir / "data_verts.npy", mmap_mode="r")  # [N, 5023, 3]
        self.n_frames, self.dim = self.verts.shape[0], self.verts.shape[1] * 3
        templates = self._load_pickle("templates.pkl")
        sequence_index = self._load_pickle("subj_seq_to_idx.pkl")
        self.raw_audio = self._load_pickle("raw_audio_fixed.pkl")
        self.subjects = sorted(sequence_index)
        assert sorted(sum(SPLITS.values(), [])) == self.subjects, "unexpected speaker list"
        self.split_of = {s: k for k, speakers in SPLITS.items() for s in speakers}

        # Speaker of every frame, and the sequence table (speaker, sentence, frame ids).
        self.subject_of = np.full(self.n_frames, -1, np.int64)
        self.sequences = []
        for si, subject in enumerate(self.subjects):
            for sentence in sorted(sequence_index[subject]):
                frames = sequence_index[subject][sentence]
                idx = np.array([frames[k] for k in sorted(frames)], np.int64)
                self.subject_of[idx] = si
                self.sequences.append((subject, sentence, idx))
        assert (self.subject_of >= 0).all(), "frames without a speaker"
        self.templates = np.stack(
            [np.asarray(templates[s], np.float32).reshape(-1) for s in self.subjects]
        )  # [n_speakers, D]
        names = np.array(self.subjects)[self.subject_of]
        self.is_train = np.isin(names, SPLITS["train"])
        self.is_test = np.isin(names, SPLITS["test"])

    def _load_pickle(self, name):
        with open(self.dir / name, "rb") as f:
            return pickle.load(f, encoding="latin1")


def select_chunks(index, chunk, smoke):
    """Start frames of the chunks to process. In smoke mode every 30th chunk is used, plus
    the first chunk of every speaker."""
    starts = set(range(0, index.n_frames, chunk)[:: 30 if smoke else 1])
    if smoke:
        starts |= {
            int(np.argmax(index.subject_of == i)) // chunk * chunk
            for i in range(len(index.subjects))
        }
    return sorted(starts)


def resample_audio(index):
    """22 kHz int16 audio -> 16 kHz int16, concatenated over sequences.

    Five VOCASET sequences have motion but no audio; they are kept with has_audio = 0.
    Returns (audio, offsets, rows) with rows = (speaker id, sentence number, has_audio).
    """
    clips, offsets, rows = [], [0], []
    for subject, sentence, _ in index.sequences:
        has_audio = sentence in index.raw_audio.get(subject, {})
        if has_audio:
            entry = index.raw_audio[subject][sentence]
            x = np.asarray(entry["audio"], np.float32)
            sr = int(entry["sample_rate"])
            g = np.gcd(sr, AUDIO_SR)
            x = resample_poly(x, AUDIO_SR // g, sr // g)
            clips.append(np.clip(x, -32768, 32767).astype(np.int16))
        else:
            clips.append(np.zeros(0, np.int16))
        offsets.append(offsets[-1] + len(clips[-1]))
        rows.append(
            (index.subjects.index(subject), int(sentence.replace("sentence", "")), int(has_audio))
        )
    return np.concatenate(clips), np.array(offsets, np.int64), rows


def stream(index, chunks, chunk):
    """Yield (start, end, displacements [n, D] float32) in file order."""
    for a in chunks:
        b = min(index.n_frames, a + chunk)
        X = np.asarray(index.verts[a:b], dtype=np.float32).reshape(b - a, index.dim)
        yield a, b, X - index.templates[index.subject_of[a:b]]


def fit_basis(index, chunks, args, selected, cache, log):
    """Pass 1: PCA basis of the selected training frames (cached)."""
    if cache.exists():
        c = np.load(cache)
        log(f"pass 1 skipped: basis loaded from {cache.name}")
        return c["mean"], c["basis"], c["S"], c["expl"]
    X_sel = np.empty((selected.sum(), index.dim), np.float32)
    p = 0
    for a, b, X in stream(index, chunks, args.chunk):
        m = selected[a:b]
        X_sel[p : p + m.sum()] = X[m]
        p += m.sum()
        if a // args.chunk % 10 == 0:
            log(f"pass 1: {b}/{index.n_frames} frames read")
    mean = X_sel.mean(0)
    X_sel -= mean
    log(f"randomized SVD on {X_sel.shape[0]} x {index.dim} training displacements")
    _, S, Vh = torch.svd_lowrank(torch.from_numpy(X_sel), q=args.k + 32, niter=6)
    basis = Vh[:, : args.k].T.contiguous().numpy()  # [K, D], orthonormal rows
    total_var = float((X_sel.astype(np.float64) ** 2).sum())
    explained = (S.numpy().astype(np.float64) ** 2) / total_var
    np.savez(cache, mean=mean, basis=basis, S=S.numpy(), expl=explained)
    log(f"basis cached -> {cache.name}")
    return mean, basis, S.numpy(), explained


def encode(index, chunks, args, mean, basis, lips, flame_projector, is_test, log):
    """Pass 2: PCA coefficients of every frame plus reconstruction errors (metres)."""
    N = index.n_frames
    coeffs = np.zeros((N, args.k), np.float32)
    err = dict(
        mean=np.zeros(N, np.float32),
        lip=np.zeros(N, np.float32),
        zero_lip=np.zeros(N, np.float32),
        flame_mean=[],
        flame_lip=[],
    )
    for a, b, X in stream(index, chunks, args.chunk):
        C = (X - mean) @ basis.T
        coeffs[a:b] = C
        E = np.linalg.norm((C @ basis + mean - X).reshape(b - a, -1, 3), axis=2)
        err["mean"][a:b], err["lip"][a:b] = E.mean(1), E[:, lips].max(1)
        err["zero_lip"][a:b] = np.linalg.norm(X.reshape(b - a, -1, 3), axis=2)[:, lips].max(1)
        test = is_test[a:b]
        if test.any():
            Xt = X[test]
            Ef = np.linalg.norm((Xt @ flame_projector.T - Xt).reshape(test.sum(), -1, 3), axis=2)
            err["flame_mean"].append(Ef.mean(1))
            err["flame_lip"].append(Ef[:, lips].max(1))
        if a // args.chunk % 10 == 0:
            log(f"pass 2: {b}/{N} frames encoded")
    err["flame_mean"] = np.concatenate(err["flame_mean"])
    err["flame_lip"] = np.concatenate(err["flame_lip"])
    return coeffs, err


def plot_basis(out, explained, k, err, is_test, S, n_svd, basis, templates):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.arange(1, len(explained) + 1), np.cumsum(explained) * 100)
    ax[0].axvline(k, ls="--", c="gray")
    ax[0].set_xlabel("PCA components")
    ax[0].set_ylabel("cumulative explained variance (%)")
    ax[0].set_title("motion basis (training speakers)")
    bins = np.linspace(0, 15, 76)
    ax[1].hist(err["zero_lip"][is_test] * 1000, bins, alpha=0.5, label="no motion (template)")
    ax[1].hist(err["flame_lip"] * 1000, bins, alpha=0.5, label="FLAME 100 expr (LS)")
    ax[1].hist(err["lip"][is_test] * 1000, bins, alpha=0.5, label=f"PCA {k}")
    ax[1].set_xlabel("max lip-vertex error per frame (mm), unseen test speakers")
    ax[1].legend()
    viz.save_figure(fig, out / "01_basis_and_error.png", dpi=110)

    # First four modes: displacement magnitude at +3 sigma on the mean template, front view.
    sigma = S[:4] / np.sqrt(n_svd)
    T_mean = templates.mean(0).reshape(-1, 3)
    front = T_mean[:, 2] > np.percentile(T_mean[:, 2], 40)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for i, ax in enumerate(axes):
        D = (3 * sigma[i] * basis[i]).reshape(-1, 3)
        P = T_mean + D
        sc = ax.scatter(
            P[front, 0], P[front, 1], c=np.linalg.norm(D[front], axis=1) * 1000, s=2, cmap="viridis"
        )
        ax.set_title(f"mode {i + 1} (+3$\\sigma$), {explained[i] * 100:.1f}% var")
        ax.set_aspect("equal")
        ax.axis("off")
        plt.colorbar(sc, ax=ax, label="mm", shrink=0.8)
    viz.save_figure(fig, out / "02_pca_modes.png", dpi=110)


def main():
    args = parse_args()
    if args.smoke:
        args.out = str(Path(args.out).with_name("smoke_test.npz"))
    seed_everything(0)
    cfg = load_config()
    out_fig = ensure_dir(FLAME_OUTPUTS / "04a_motion_basis")
    ensure_dir(Path(args.out).parent)
    log = Logger()

    index = VocasetIndex(args.vocaset)
    chunks = select_chunks(index, args.chunk, args.smoke)
    processed = np.zeros(index.n_frames, bool)
    for a in chunks:
        processed[a : a + args.chunk] = True
    is_train, is_test = index.is_train & processed, index.is_test & processed
    log(
        f"VOCASET {index.n_frames} frames, {len(index.sequences)} sequences; "
        f"train frames {is_train.sum()}, test frames {is_test.sum()}"
    )

    audio, audio_offsets, rows = resample_audio(index)
    log(f"audio resampled; {sum(1 - r[2] for r in rows)} sequences without audio")

    selected = np.zeros(index.n_frames, bool)
    selected[np.where(is_train)[0][:: args.sample_step]] = True
    suffix = "_smoke" if args.smoke else ""
    cache = Path(args.out).parent / f"_basis_cache_k{args.k}_s{args.sample_step}{suffix}.npz"
    mean, basis, S, explained = fit_basis(index, chunks, args, selected, cache, log)

    flame = load_flame(cfg, n_expr=N_FLAME_EXPRESSIONS)
    lips = np.asarray(flame.region("lips"))
    E = flame.exprdirs.reshape(-1, N_FLAME_EXPRESSIONS).numpy().astype(np.float64)
    flame_projector = (E @ np.linalg.pinv(E)).astype(np.float32)  # onto the expression span
    coeffs, err = encode(index, chunks, args, mean, basis, lips, flame_projector, is_test, log)

    seq_frames = np.concatenate([idx for _, _, idx in index.sequences])
    seq_start = np.cumsum([0] + [len(idx) for _, _, idx in index.sequences])
    # Frames of sequence i: coeffs[seq_frames[seq_start[i]:seq_start[i + 1]]].
    np.savez_compressed(
        args.out,
        coeffs=coeffs,
        basis=basis.astype(np.float32),
        mean=mean.astype(np.float32),
        templates=index.templates,
        subjects=np.array(index.subjects),
        split=np.array([index.split_of[s] for s in index.subjects]),
        seq=np.array(rows, np.int64),
        seq_frames=seq_frames,
        seq_start=seq_start,
        audio=audio,
        audio_offsets=audio_offsets,
        audio_sr=AUDIO_SR,
        fps=DATASET_FPS,
        lips=lips,
        explained_variance=explained.astype(np.float32),
    )
    log(f"saved {args.out} ({Path(args.out).stat().st_size / 2**20:.1f} MB)")

    def mm(x):
        return float(np.mean(x) * 1000)

    def p95(x):
        return float(np.percentile(x, 95) * 1000)

    stats = {
        "frames": int(index.n_frames),
        "sequences": len(index.sequences),
        "k": args.k,
        "svd_frames": int(selected.sum()),
        "explained_variance_k": float(explained[: args.k].sum()),
        "test_speakers": {
            "zero_motion": {
                "lip_max_mm": mm(err["zero_lip"][is_test]),
                "lip_max_p95_mm": p95(err["zero_lip"][is_test]),
            },
            "flame_expr100_ls": {
                "vertex_mm": mm(err["flame_mean"]),
                "lip_max_mm": mm(err["flame_lip"]),
                "lip_max_p95_mm": p95(err["flame_lip"]),
            },
            f"pca{args.k}": {
                "vertex_mm": mm(err["mean"][is_test]),
                "lip_max_mm": mm(err["lip"][is_test]),
                "lip_max_p95_mm": p95(err["lip"][is_test]),
            },
        },
        "train_speakers_pca": {
            "vertex_mm": mm(err["mean"][is_train]),
            "lip_max_mm": mm(err["lip"][is_train]),
        },
        "seconds": time.time() - log.t0,
    }
    save_json(stats, out_fig / "stats.json")
    log(stats)
    plot_basis(out_fig, explained, args.k, err, is_test, S, selected.sum(), basis, index.templates)
    log("figures ->", out_fig)


if __name__ == "__main__":
    main()
