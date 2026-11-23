"""Project paths, configuration loading and runtime setup."""

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"

DATA_DIR = ROOT / "data"
IDENTITY_DIR = DATA_DIR / "identity"  # intermediate files of the FLAME pipeline
PROCESSED_DIR = DATA_DIR / "processed"  # intermediate files of the video baseline
CHECKPOINT_DIR = ROOT / "checkpoints"  # model weights of the video baseline

OUTPUT_DIR = ROOT / "outputs"
FLAME_OUTPUTS = OUTPUT_DIR / "flame_pipeline"
BASELINE_OUTPUTS = OUTPUT_DIR / "video_baseline"


def load_config(path=None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy and (if installed) PyTorch; use all CPU cores for PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))


def use_all_cpu_threads() -> None:
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))


def get_device(cfg=None):
    import torch

    choice = (cfg or {}).get("device", "auto")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)
