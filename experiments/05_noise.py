"""Step 5: weight-noise robustness — a SIMULATION, not real analog hardware.

Analog compute-in-memory chips store weights as device conductances; programming
and read noise perturb every weight. As a simple stand-in we add Gaussian noise to
every conv/linear weight:  w_noisy = w + N(0, (eta * max|w_layer|)^2),
i.e. noise relative to each layer's weight range (one conductance range per
layer). Activations stay in float, so only the weights differ between models:

  fp32 : BN-folded FP32 weights
  ptq  : dequantized INT8 weights of the PTQ model
  qat  : dequantized INT8 weights of the QAT model

For each noise level, 10 independent noise draws; test accuracy mean and std.
Not modelled: conductance drift, IR drop, ADC/DAC quantization, non-linearity,
noise on activations, per-device variation that differs from Gaussian.
"""
import argparse

import _path  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from qstudy import data, evaluate, quant, results, zoo
from qstudy.model import layer_names

p = argparse.ArgumentParser()
p.add_argument("--etas", type=float, nargs="+",
               default=[0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1])
p.add_argument("--draws", type=int, default=10)
args = p.parse_args()

dev = "cuda"
te_x, te_y = data.get_splits()["test"]
te_x_dev = te_x.to(dev)


def float_layer(model, name):
    m = quant.get_module(model, name)
    while isinstance(m, nn.Sequential):
        m = m[0]
    return m


def from_int8(int8_name):
    """FP32 fused model whose weights/biases are the INT8 model's dequantized ones."""
    q = zoo.int8(int8_name)
    f = zoo.fp32(fused=True)
    for name in layer_names(f):
        qm = quant.get_module(q, name)
        qm = qm[0] if isinstance(qm, nn.Sequential) else qm
        fm = float_layer(f, name)
        fm.weight.data = qm.weight().dequantize().float()
        fm.bias.data = qm.bias().detach().float()
    return f


models = {"fp32": zoo.fp32(fused=True), "ptq": from_int8("ptq"), "qat": from_int8("qat")}
out = {"etas": args.etas, "draws": args.draws, "models": {}}
for mname, base in models.items():
    base = base.to(dev).eval()
    clean = [p.detach().clone() for p in (float_layer(base, n).weight for n in layer_names(base))]
    rows = []
    for eta in args.etas:
        accs = []
        for d in range(args.draws if eta > 0 else 1):
            g = torch.Generator(device=dev).manual_seed(1000 * d + 7)
            for n, w0 in zip(layer_names(base), clean):
                w = float_layer(base, n).weight
                sigma = eta * w0.abs().max()
                w.data = w0 + sigma * torch.randn(w0.shape, generator=g, device=dev)
            accs.append(evaluate.accuracy(evaluate.predict(base, te_x_dev, device=dev), te_y))
        rows.append({"eta": eta, "mean_acc": float(np.mean(accs)),
                     "std_acc": float(np.std(accs)), "min_acc": float(np.min(accs)),
                     "accs": accs})
        print(mname, rows[-1]["eta"], round(rows[-1]["mean_acc"], 4), round(rows[-1]["std_acc"], 4), flush=True)
    for n, w0 in zip(layer_names(base), clean):
        float_layer(base, n).weight.data = w0
    out["models"][mname] = rows

out["note"] = __doc__.strip()
print("wrote", results.write("noise", out))
