"""MobileNetV2 adapted for 32x32 CIFAR inputs, written to be quantizable in eager mode.

Changes from the ImageNet MobileNetV2 (Sandler et al., 2018):
- stem conv has stride 1 (not 2) and the second stage has stride 1 (not 2), so a
  32x32 image is downsampled 8x to 4x4 instead of 32x to 1x1;
- ReLU instead of ReLU6, so PyTorch can fuse Conv+BN+ReLU (the same substitution
  torchvision's quantizable MobileNetV2 makes); the network is trained with ReLU
  from the start, so FP32 and INT8 models compute the same function;
- residual adds go through FloatFunctional so they become quantized adds;
- QuantStub/DeQuantStub mark where the INT8 region starts and ends.

Every conv (and the final linear) is a named "layer" so per-layer sensitivity
results read like `blocks.5.dw`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat

# (expansion t, output channels c, repeats n, first stride s)
CIFAR_CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
             (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k, stride=1, groups=1, relu=True):
        layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if relu:
            layers.append(nn.ReLU())
        super().__init__(*layers)

    def fuse(self, qat: bool):
        names = ["0", "1", "2"] if len(self) == 3 else ["0", "1"]
        (fuse_modules_qat if qat else fuse_modules)(self, [names], inplace=True)


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, t):
        super().__init__()
        hidden = cin * t
        self.use_res = stride == 1 and cin == cout
        self.expand = ConvBNReLU(cin, hidden, 1) if t != 1 else None
        self.dw = ConvBNReLU(hidden, hidden, 3, stride, groups=hidden)
        self.project = ConvBNReLU(hidden, cout, 1, relu=False)
        self.skip_add = FloatFunctional()

    def forward(self, x):
        h = self.expand(x) if self.expand is not None else x
        h = self.project(self.dw(h))
        return self.skip_add.add(x, h) if self.use_res else h


class MobileNetV2Cifar(nn.Module):
    def __init__(self, num_classes=10, width=32, last=1280, cfg=CIFAR_CFG):
        super().__init__()
        self.quant = QuantStub()
        self.stem = ConvBNReLU(3, width, 3, stride=1)
        blocks, cin = [], width
        for t, c, n, s in cfg:
            for i in range(n):
                blocks.append(InvertedResidual(cin, c, s if i == 0 else 1, t))
                cin = c
        self.blocks = nn.Sequential(*blocks)
        self.head = ConvBNReLU(cin, last, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last, num_classes)
        self.dequant = DeQuantStub()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.head(self.blocks(self.stem(x)))
        x = torch.flatten(self.pool(x), 1)
        return self.dequant(self.fc(x))

    def fuse(self, qat: bool = False):
        """Fuse every Conv+BN(+ReLU) in place. Eval mode folds BN into the conv;
        QAT mode keeps BN as part of a ConvBnReLU2d that is folded at convert time."""
        for m in self.modules():
            if isinstance(m, ConvBNReLU):
                m.fuse(qat)
        return self


def layer_names(model: MobileNetV2Cifar) -> list[str]:
    """Quantizable compute layers in forward order (stem, block convs, head, fc)."""
    names = ["stem"]
    for i, b in enumerate(model.blocks):
        if b.expand is not None:
            names.append(f"blocks.{i}.expand")
        names += [f"blocks.{i}.dw", f"blocks.{i}.project"]
    return names + ["head", "fc"]


def layer_kind(name: str) -> str:
    for k in ("expand", "dw", "project"):
        if name.endswith("." + k):
            return k
    return name  # stem / head / fc
