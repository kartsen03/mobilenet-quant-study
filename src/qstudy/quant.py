"""Eager-mode INT8 quantization helpers (torch.ao.quantization, oneDNN engine).

torch.ao.quantization is deprecated in favour of torchao, but in PyTorch 2.14 it
is still the API that runs real INT8 kernels on the CPU in eager mode (the torchao
PT2E flow needs torch.compile/Inductor to lower to INT8 kernels).
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import torch
import torch.ao.quantization as tq
import torch.nn as nn
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import (HistogramObserver, MinMaxObserver,
                                            PerChannelMinMaxObserver)

from .data import normalize
from .model import MobileNetV2Cifar, layer_names

ENGINE = "onednn"  # the only quantized engine in the Windows build of PyTorch 2.14.1

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.ao")
warnings.filterwarnings("ignore", message=r".*quantize_per_tensor.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*quantized tensor creation functions.*")


def set_engine() -> None:
    torch.backends.quantized.engine = ENGINE


def make_qconfig(weight: str = "per_channel", act: str = "histogram") -> QConfig:
    """INT8 config. ('per_channel', 'histogram') equals get_default_qconfig('onednn'):
    unsigned 8-bit activations with an affine scale/zero-point chosen by a histogram
    search, signed 8-bit symmetric weights with one scale per output channel."""
    act_obs = {"histogram": HistogramObserver, "minmax": MinMaxObserver}[act]
    if weight == "per_channel":
        w_obs = PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                                   qscheme=torch.per_channel_symmetric)
    else:
        w_obs = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    return QConfig(activation=act_obs.with_args(reduce_range=False), weight=w_obs)


def get_module(model: nn.Module, name: str) -> nn.Module:
    m = model
    for part in name.split("."):
        m = getattr(m, part)
    return m


def set_module(model: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, leaf = name.rpartition(".")
    setattr(get_module(model, parent) if parent else model, leaf, new)


class FloatIsland(nn.Module):
    """Keeps one layer in FP32 inside an otherwise INT8 network:
    dequantize -> FP32 layer -> requantize (with its own calibrated scale)."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.dequant = tq.DeQuantStub()
        self.layer = layer
        self.layer.qconfig = None  # stays float through prepare/convert
        self.quant = tq.QuantStub()

    def forward(self, x):
        return self.quant(self.layer(self.dequant(x)))


class QuantIsland(nn.Module):
    """The opposite: one INT8 layer inside an otherwise FP32 network
    (quantize -> INT8 layer -> dequantize). Used for per-layer sensitivity."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.layer = layer
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.layer(self.quant(x)))


def prepare_ptq(fp32: MobileNetV2Cifar, qconfig: QConfig, *, float_layers=(),
                only_layer: str | None = None) -> nn.Module:
    """Copy, fuse (eval mode: BN folded into conv), attach observers.

    float_layers: layer names to keep in FP32 (mixed precision).
    only_layer:   quantize just this one layer, everything else FP32.
    """
    set_engine()
    m = copy.deepcopy(fp32).cpu().eval()
    m.fuse(qat=False)
    if only_layer is not None:
        m.qconfig = None
        island = QuantIsland(get_module(m, only_layer))
        island.qconfig = qconfig
        set_module(m, only_layer, island)
    else:
        m.qconfig = qconfig
        for name in float_layers:
            set_module(m, name, FloatIsland(get_module(m, name)))
    tq.prepare(m, inplace=True)
    return m


@torch.inference_mode()
def calibrate(m: nn.Module, x_uint8: torch.Tensor, batch_size: int = 256) -> None:
    for i in range(0, len(x_uint8), batch_size):
        m(normalize(x_uint8[i:i + batch_size]))


def convert(m: nn.Module) -> nn.Module:
    set_engine()
    return tq.convert(m.eval(), inplace=False)


def ptq(fp32, calib_x, qconfig, **kw) -> nn.Module:
    m = prepare_ptq(fp32, qconfig, **kw)
    calibrate(m, calib_x)
    return convert(m)


def int8_skeleton(qconfig: QConfig | None = None, float_layers=()) -> nn.Module:
    """An uncalibrated converted model with the right structure, for load_state_dict."""
    m = prepare_ptq(MobileNetV2Cifar(), qconfig or make_qconfig(), float_layers=float_layers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m(torch.randn(2, 3, 32, 32))  # give observers a range so convert() succeeds
        return convert(m)


# --- weight statistics (why depthwise is hard) ---------------------------------

def int8_weight_sqnr(w: torch.Tensor, per_channel: bool) -> tuple[float, int]:
    """SQNR (dB) of symmetric INT8 weight quantization, and how many output channels
    are rounded entirely to zero."""
    w = w.detach().float()
    flat = w.flatten(1)
    amax = flat.abs().amax(1, keepdim=True) if per_channel else flat.abs().max().expand(len(flat), 1)
    scale = (amax / 127.5).clamp_min(1e-12)
    q = torch.clamp(torch.round(flat / scale), -128, 127) * scale
    err = (flat - q).pow(2).sum()
    sqnr = 10 * torch.log10(flat.pow(2).sum() / err.clamp_min(1e-30)).item()
    zeroed = int(((q == 0).all(1) & (flat != 0).any(1)).sum())
    return sqnr, zeroed


def weight_stats(fp32: MobileNetV2Cifar) -> list[dict]:
    """Per fused (BN-folded) layer: channel-range spread and INT8 weight SQNR."""
    m = copy.deepcopy(fp32).cpu().eval().fuse(qat=False)
    rows = []
    for name in layer_names(m):
        mod = get_module(m, name)
        conv = mod
        while isinstance(conv, nn.Sequential):  # ConvBNReLU -> fused ConvReLU2d -> Conv2d
            conv = conv[0]
        w = conv.weight
        amax = w.detach().flatten(1).abs().amax(1).numpy()
        pt_sqnr, pt_zero = int8_weight_sqnr(w, per_channel=False)
        pc_sqnr, _ = int8_weight_sqnr(w, per_channel=True)
        rows.append({
            "layer": name, "out_channels": int(w.shape[0]),
            "weights_per_channel": int(w[0].numel()),
            "range_spread_max_over_median": float(amax.max() / np.median(amax)),
            "sqnr_db_per_tensor": round(pt_sqnr, 2), "sqnr_db_per_channel": round(pc_sqnr, 2),
            "channels_zeroed_per_tensor": pt_zero,
        })
    return rows
