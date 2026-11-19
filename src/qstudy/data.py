"""CIFAR-10 loading, the fixed train/val/test split, and tensor-based batching.

The whole dataset fits in memory (50k x 32x32x3 uint8 = 147 MB), so batches are
sliced from tensors instead of going through DataLoader workers (slow to spawn on
Windows). Training augmentation (pad-4 random crop + horizontal flip, the usual
CIFAR recipe) runs on the GPU from a seeded generator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .env import REPO_ROOT

DATA_ROOT = REPO_ROOT / "data"
SPLIT_FILE = REPO_ROOT / "splits" / "cifar10_split.json"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def load_raw(root: Path = DATA_ROOT):
    """Return (train_x, train_y, test_x, test_y); x is uint8 NHWC, y is int64."""
    from torchvision.datasets import CIFAR10
    tr = CIFAR10(str(root), train=True, download=False)
    te = CIFAR10(str(root), train=False, download=False)
    return (torch.from_numpy(tr.data), torch.tensor(tr.targets, dtype=torch.long),
            torch.from_numpy(te.data), torch.tensor(te.targets, dtype=torch.long))


def make_split(train_y: torch.Tensor, val_per_class: int = 500, seed: int = 0) -> dict:
    """Stratified hold-out: `val_per_class` images of each class become validation."""
    rng = np.random.default_rng(seed)
    y = train_y.numpy()
    val = []
    for c in range(10):
        idx = np.flatnonzero(y == c)
        val.extend(rng.choice(idx, size=val_per_class, replace=False).tolist())
    val = sorted(val)
    train = sorted(set(range(len(y))) - set(val))
    digest = hashlib.sha256(json.dumps(val).encode()).hexdigest()
    return {"seed": seed, "val_per_class": val_per_class, "n_train": len(train),
            "n_val": len(val), "n_test": 10000, "val_sha256": digest,
            "val_indices": val}


def load_split() -> dict:
    s = json.loads(SPLIT_FILE.read_text())
    digest = hashlib.sha256(json.dumps(s["val_indices"]).encode()).hexdigest()
    assert digest == s["val_sha256"], "split file corrupted"
    return s


def get_splits():
    """Return dict name -> (x uint8 NHWC, y) for 'train', 'val', 'test'."""
    tx, ty, sx, sy = load_raw()
    s = load_split()
    val = torch.tensor(s["val_indices"])
    mask = torch.ones(len(ty), dtype=torch.bool)
    mask[val] = False
    tr = torch.nonzero(mask).squeeze(1)
    return {"train": (tx[tr], ty[tr]), "val": (tx[val], ty[val]), "test": (sx, sy)}


def calibration_indices(n_train: int, n: int = 1024, seed: int = 0) -> torch.Tensor:
    """Fixed random subset of the *training* split used for PTQ calibration."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_train, generator=g)[:n]


def normalize(x_uint8_nhwc: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC -> float32 NCHW normalized with the CIFAR-10 mean/std."""
    x = x_uint8_nhwc.permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x_uint8_nhwc: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Pad-4 random crop + random horizontal flip, then normalize (all on x's device)."""
    b = x_uint8_nhwc.shape[0]
    dev = x_uint8_nhwc.device
    x = x_uint8_nhwc.permute(0, 3, 1, 2).float().div_(255)
    x = F.pad(x, (4, 4, 4, 4))  # zero (black) padding, like torchvision RandomCrop
    oy = torch.randint(0, 9, (b,), generator=gen, device=dev)
    ox = torch.randint(0, 9, (b,), generator=gen, device=dev)
    ar = torch.arange(32, device=dev)
    rows = (oy[:, None] + ar)[:, :, None]          # b,32,1
    cols = (ox[:, None] + ar)[:, None, :]          # b,1,32
    flip = torch.rand(b, generator=gen, device=dev) < 0.5
    cols = torch.where(flip[:, None, None], cols.flip(-1), cols)
    bi = torch.arange(b, device=dev)[:, None, None]
    x = x.permute(0, 2, 3, 1)[bi, rows, cols]      # b,32,32,3
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(CIFAR_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


def batches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    """Sequential (unshuffled) normalized batches; x stays on its device."""
    for i in range(0, len(y), batch_size):
        yield normalize(x[i:i + batch_size]), y[i:i + batch_size]
