"""Load every model variant by name from checkpoints/ (for latency and export)."""
from __future__ import annotations

import torch

from . import quant, results
from .model import MobileNetV2Cifar


def fp32(fused: bool = False) -> torch.nn.Module:
    m = MobileNetV2Cifar()
    m.load_state_dict(torch.load(results.CKPT / "fp32.pt", weights_only=True))
    m.eval()
    return m.fuse(qat=False) if fused else m


def int8(name: str) -> torch.nn.Module:
    """'ptq', 'qat', or a mixed-precision model 'mixed_<k>' (FP32 layers listed in
    results/mixed.json)."""
    float_layers = ()
    if name.startswith("mixed"):
        float_layers = results.read("mixed")["models"][name]["float_layers"]
    m = quant.int8_skeleton(float_layers=float_layers)
    m.load_state_dict(torch.load(results.CKPT / f"{name}.pt", weights_only=False))
    return m.eval()


def load_torch(name: str) -> torch.nn.Module:
    if name == "fp32":
        return fp32()
    if name == "fp32_fused":
        return fp32(fused=True)
    return int8(name)
