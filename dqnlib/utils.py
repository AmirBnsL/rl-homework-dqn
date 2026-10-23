"""Small helpers shared by every other module (seeding, schedules, atomic saves)."""
from __future__ import annotations

import json
import os
import random
import tempfile

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator we use (Python, NumPy, PyTorch, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def epsilon_at(step: int, eps_start: float = 1.0, eps_end: float = 0.05,
               decay_steps: int = 10_000) -> float:
    """Linear epsilon schedule: eps_start at step 0, eps_end after `decay_steps`, then constant."""
    fraction = min(1.0, step / decay_steps)
    return float(eps_start + fraction * (eps_end - eps_start))


def moving_average(values, window: int) -> np.ndarray:
    """Centred-free trailing moving average, same convention as pandas' rolling(min_periods=1)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    out = np.empty_like(values)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    for i in range(len(values)):
        lo = max(0, i + 1 - window)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


def atomic_savez(path: str, **arrays) -> None:
    """Write an .npz without leaving a half-written file behind if the run is killed."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_save_json(path: str, payload: dict) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def atomic_save_torch(path: str, state_dict: dict) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".pth")
    os.close(fd)
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
