"""Step 4: where is accuracy lost? Per-layer sensitivity, then mixed precision.

Per-layer: quantize ONE layer to INT8 (its input, weights and output; everything
else FP32), calibrate as in step 2, and measure on the *validation* split:
  - accuracy drop vs FP32,
  - logit SQNR = 10·log10(Σ fp32_logits² / Σ (int8_logits − fp32_logits)²).
Single-layer accuracy drops on 5,000 images are mostly smaller than the noise
(one image = 0.02 points), so layers are ranked by logit SQNR (lower = more
sensitive). This rule was fixed before looking at the data.

Mixed precision: keep the k most sensitive layers in FP32 (dequantize -> FP32
layer -> requantize), everything else INT8 via the same PTQ recipe. Selection uses
validation only; test accuracy is reported for each k. k=3 is the headline model
(chosen in advance; the full curve is reported).
"""
import argparse
import time

import _path  # noqa: F401
import numpy as np
import torch

from qstudy import data, evaluate, quant, results
from qstudy.model import MobileNetV2Cifar, layer_kind, layer_names

p = argparse.ArgumentParser()
p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
p.add_argument("--headline-k", type=int, default=3)
args = p.parse_args()

quant.set_engine()
splits = data.get_splits()
tr_x, _ = splits["train"]
va_x, va_y = splits["val"]
te_x, te_y = splits["test"]
calib = tr_x[data.calibration_indices(len(tr_x), 1024, 0)]
qcfg = quant.make_qconfig()

fp32 = MobileNetV2Cifar()
fp32.load_state_dict(torch.load(results.CKPT / "fp32.pt", weights_only=True))
fp32.eval()
fp32_val_pred, fp32_val_logits = evaluate.predict(fp32, va_x, return_logits=True)
fp32_val_acc = evaluate.accuracy(fp32_val_pred, va_y)
fp32_test_acc = results.read("fp32")["test_acc"]


def sqnr(ref, x):
    return float(10 * np.log10((ref ** 2).sum() / max(((x - ref) ** 2).sum(), 1e-30)))


rows = []
t0 = time.perf_counter()
for name in layer_names(fp32):
    q = quant.ptq(fp32, calib, qcfg, only_layer=name)
    pred, logits = evaluate.predict(q, va_x, return_logits=True)
    acc = evaluate.accuracy(pred, va_y)
    row = {"layer": name, "kind": layer_kind(name), "val_acc": acc,
           "val_acc_drop_pts": round(100 * (fp32_val_acc - acc), 3),
           "val_pred_changed": int((pred != fp32_val_pred).sum()),
           "logit_sqnr_db": round(sqnr(fp32_val_logits, logits), 2)}
    rows.append(row)
    print(row, flush=True)
per_layer_minutes = (time.perf_counter() - t0) / 60
ranked = [r["layer"] for r in sorted(rows, key=lambda r: r["logit_sqnr_db"])]

# Full INT8 reference on val for the same metrics
full = quant.ptq(fp32, calib, qcfg)
fp, fl = evaluate.predict(full, va_x, return_logits=True)
full_int8 = {"val_acc": evaluate.accuracy(fp, va_y), "logit_sqnr_db": round(sqnr(fp32_val_logits, fl), 2)}

models = {}
curve = [{"k": 0, "val_acc": full_int8["val_acc"],
          "test_acc": results.read("ptq")["test_acc"],
          "size_bytes": results.read("ptq")["size_bytes"], "float_layers": []}]
for k in args.ks:
    float_layers = ranked[:k]
    q = quant.ptq(fp32, calib, qcfg, float_layers=float_layers)
    pv = evaluate.predict(q, va_x)
    pt = evaluate.predict(q, te_x)
    name = f"mixed_{k}"
    size = evaluate.saved_size(q.state_dict(), results.CKPT / f"{name}.pt")
    entry = {"k": k, "float_layers": float_layers, "val_acc": evaluate.accuracy(pv, va_y),
             "test_acc": evaluate.accuracy(pt, te_y), "size_bytes": size}
    models[name] = entry
    curve.append(entry)
    print(entry, flush=True)
    if k == args.headline_k:
        np.save(results.RESULTS / "preds" / "mixed_test.npy", pt.astype(np.uint8))

out = {
    "config": {"engine": quant.ENGINE, "calib": "1024 train images, seed 0",
               "ranking": "ascending validation logit SQNR (single layer INT8)",
               "headline": f"mixed_{args.headline_k}"},
    "fp32_val_acc": fp32_val_acc, "fp32_test_acc": fp32_test_acc,
    "full_int8_val": full_int8, "per_layer_minutes": round(per_layer_minutes, 1),
    "per_layer": rows, "ranked": ranked, "curve": curve,
}
print("wrote", results.write("sensitivity", out))
print("wrote", results.write("mixed", {"headline": f"mixed_{args.headline_k}",
                                       "models": models}))
