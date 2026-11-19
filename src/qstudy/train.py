"""Training loop shared by FP32 training and quantization-aware fine-tuning."""
from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn as nn

from .data import augment
from .evaluate import predict


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model: nn.Module, wd: float):
    """Weight decay on conv/linear weights only, not on BN parameters or biases."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def train(model: nn.Module, train_xy, val_xy, *, epochs: int, lr: float, wd: float,
          batch_size: int, seed: int, device: str, warmup_epochs: float = 0.0,
          amp_dtype: torch.dtype | None = None, channels_last: bool = False,
          on_epoch_start: Callable[[nn.Module, int], None] | None = None,
          before_eval: Callable[[nn.Module], None] | None = None,
          log: Callable[[dict], None] = print) -> list[dict]:
    """SGD + Nesterov, linear warmup then cosine decay to zero, per-step schedule.

    `train_xy`/`val_xy` are (uint8 NHWC, labels) tensors already on `device`.
    Returns one record per epoch (loss, train/val accuracy, lr, seconds).
    """
    tx, ty = train_xy
    n = len(ty)
    steps_per_epoch = math.ceil(n / batch_size)
    total = epochs * steps_per_epoch
    warm = int(warmup_epochs * steps_per_epoch)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    opt = torch.optim.SGD(param_groups(model, wd), lr=lr, momentum=0.9, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator(device=device).manual_seed(seed)
    mf = torch.channels_last if channels_last else torch.contiguous_format
    history = []
    for epoch in range(epochs):
        model.train()
        if on_epoch_start is not None:
            on_epoch_start(model, epoch)
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=gen, device=device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            xb = augment(tx[idx], gen).contiguous(memory_format=mf)
            yb = ty[idx]
            with torch.autocast("cuda", dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                out = model(xb)
                loss = loss_fn(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach() * len(yb)
            correct += (out.argmax(1) == yb).sum()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        if before_eval is not None:
            before_eval(model)
        val_pred = predict(model, val_xy[0], device=device, channels_last=channels_last)
        rec = {"epoch": epoch + 1,
               "train_loss": round(loss_sum.item() / n, 5),
               "train_acc": round(correct.item() / n, 5),
               "val_acc": round(float((val_pred == val_xy[1].cpu().numpy()).mean()), 5),
               "lr_end": round(sched.get_last_lr()[0], 6),
               "seconds": round(secs, 2)}
        history.append(rec)
        log(rec)
    return history
