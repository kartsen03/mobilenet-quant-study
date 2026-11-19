"""Accuracy, on-disk size and latency measurement."""
from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .data import normalize


@torch.inference_mode()
def predict(model, x_uint8: torch.Tensor, *, device: str = "cpu", batch_size: int = 500,
            channels_last: bool = False, return_logits: bool = False):
    """Argmax predictions (and optionally logits) for uint8 NHWC images."""
    was_training = model.training
    model.eval()
    preds, logits = [], []
    mf = torch.channels_last if channels_last else torch.contiguous_format
    for i in range(0, len(x_uint8), batch_size):
        xb = normalize(x_uint8[i:i + batch_size].to(device)).contiguous(memory_format=mf)
        out = model(xb).float()
        preds.append(out.argmax(1).cpu())
        if return_logits:
            logits.append(out.cpu())
    model.train(was_training)
    p = torch.cat(preds).numpy()
    return (p, torch.cat(logits).numpy()) if return_logits else p


def accuracy(pred: np.ndarray, y) -> float:
    y = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    return float((pred == y).mean())


def saved_size(state_dict: dict, path: Path) -> int:
    """Bytes on disk of `torch.save(state_dict)`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return os.path.getsize(path)


def time_calls(fn, *, warmup_s: float = 1.0, runs: int = 200) -> list[float]:
    """Run `fn()` for `warmup_s` seconds, then time `runs` calls one by one (ms)."""
    t_end = time.perf_counter() + warmup_s
    while time.perf_counter() < t_end:
        fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out


def summarize_ms(samples: list[float]) -> dict:
    s = sorted(samples)
    q = statistics.quantiles(s, n=20)  # 5% steps
    return {"median_ms": round(statistics.median(s), 4), "p05_ms": round(q[0], 4),
            "p95_ms": round(q[-1], 4), "n": len(s)}
