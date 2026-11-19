"""Step 2: static INT8 post-training quantization.

Main model: Conv+BN+ReLU fused, PyTorch's default oneDNN qconfig (per-channel
symmetric INT8 weights, histogram-calibrated UINT8 activations), calibrated on
1,024 images from the *training* split (calibration seed 0).

Ablation (all on the same FP32 checkpoint): per-tensor vs per-channel weights x
min/max vs histogram activation calibration, each with three calibration subsets,
plus per-layer weight statistics that show why depthwise convs are hard.
"""
import argparse
import itertools
import time

import _path  # noqa: F401
import numpy as np
import torch

from qstudy import data, evaluate, quant, results
from qstudy.model import MobileNetV2Cifar

p = argparse.ArgumentParser()
p.add_argument("--calib-size", type=int, default=1024)
p.add_argument("--calib-seeds", type=int, nargs="+", default=[0, 1, 2])
args = p.parse_args()

torch.manual_seed(0)
quant.set_engine()
splits = data.get_splits()
tr_x, _ = splits["train"]
va_x, va_y = splits["val"]
te_x, te_y = splits["test"]

fp32 = MobileNetV2Cifar()
fp32.load_state_dict(torch.load(results.CKPT / "fp32.pt", weights_only=True))
fp32.eval()
fp32_res = results.read("fp32")
fp32_test = np.load(results.RESULTS / "preds" / "fp32_test.npy")

ablation = []
main = None
for (w, a), seed in itertools.product(
        [("per_tensor", "minmax"), ("per_tensor", "histogram"),
         ("per_channel", "minmax"), ("per_channel", "histogram")], args.calib_seeds):
    calib_idx = data.calibration_indices(len(tr_x), args.calib_size, seed)
    t0 = time.perf_counter()
    q = quant.ptq(fp32, tr_x[calib_idx], quant.make_qconfig(w, a))
    secs = time.perf_counter() - t0
    pv = evaluate.predict(q, va_x)
    pt = evaluate.predict(q, te_x)
    row = {"weights": w, "activations": a, "calib_seed": seed,
           "val_acc": evaluate.accuracy(pv, va_y), "test_acc": evaluate.accuracy(pt, te_y),
           "test_pred_changed_vs_fp32": int((pt != fp32_test).sum()),
           "prepare_calibrate_convert_s": round(secs, 1)}
    ablation.append(row)
    print(row)
    if (w, a, seed) == ("per_channel", "histogram", 0):
        main, main_test, main_val = q, pt, pv

size = evaluate.saved_size(main.state_dict(), results.CKPT / "ptq.pt")
np.save(results.RESULTS / "preds" / "ptq_test.npy", main_test.astype(np.uint8))
np.save(results.RESULTS / "preds" / "ptq_val.npy", main_val.astype(np.uint8))

# Sanity check: the saved INT8 state dict reloads into a fresh skeleton bit-exactly.
reloaded = quant.int8_skeleton()
reloaded.load_state_dict(torch.load(results.CKPT / "ptq.pt", weights_only=False))
x = data.normalize(te_x[:64])
with torch.inference_mode():
    reload_exact = bool(torch.equal(main(x), reloaded(x)))

out = {
    "config": {"engine": quant.ENGINE, "calib_size": args.calib_size,
               "calib_source": "training split, eval transforms (no augmentation)",
               "main": "per_channel weights + histogram activations, calib seed 0"},
    "test_acc": evaluate.accuracy(main_test, te_y),
    "val_acc": evaluate.accuracy(main_val, va_y),
    "fp32_test_acc": fp32_res["test_acc"],
    "test_pred_changed_vs_fp32": int((main_test != fp32_test).sum()),
    "size_bytes": size,
    "fp32_size_bytes": fp32_res["size_bytes"],
    "reload_bit_exact": reload_exact,
    "ablation": ablation,
    "weight_stats": quant.weight_stats(fp32),
    "model_repr": str(main)[:4000],
}
print({k: out[k] for k in ("test_acc", "val_acc", "fp32_test_acc", "size_bytes",
                           "reload_bit_exact")})
print("wrote", results.write("ptq", out))
