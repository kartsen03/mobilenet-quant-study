"""Step 6b: where does the INT8 PyTorch-vs-ORT difference come from?

Runs the PTQ QDQ model in ORT at three graph-optimization levels on 2,000 test
images. At ORT_DISABLE_ALL nothing is fused: each QuantizeLinear/DequantizeLinear
pair runs as written around a *float* Conv, i.e. the ONNX reference semantics. If
that already differs from PyTorch as much as the fully fused INT8 graph does, the
gap comes from PyTorch's oneDNN INT8 kernels (int32 bias, requantization rounding)
vs the QDQ reference math, not from ORT's fused kernels.
"""
import _path  # noqa: F401
import numpy as np
import onnxruntime as ort
import torch

from qstudy import data, quant, results, zoo

quant.set_engine()
X = data.normalize(data.get_splits()["test"][0][:2000])
q = zoo.int8("ptq")
with torch.inference_mode():
    ref = q(X).numpy()
step = float(q.fc.scale)
outs, rows = {}, {}
for lvl in ("ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_ALL"):
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, lvl)
    so.log_severity_level = 3
    s = ort.InferenceSession(str(results.CKPT / "ptq.onnx"), so, providers=["CPUExecutionProvider"])
    got = np.concatenate([s.run(None, {"x": X[i:i + 1].numpy()})[0] for i in range(len(X))])
    outs[lvl] = got
    d = np.abs(got - ref)
    rows[lvl] = {"max_diff_steps_vs_torch": round(float(d.max() / step), 2),
                 "mean_diff_steps_vs_torch": round(float(d.mean() / step), 4),
                 "frac_logits_exact": round(float((d < 1e-6).mean()), 4),
                 "pred_mismatch_vs_torch": int((got.argmax(1) != ref.argmax(1)).sum())}
    print(lvl, rows[lvl], flush=True)
d = np.abs(outs["ORT_ENABLE_ALL"] - outs["ORT_DISABLE_ALL"])
fused_vs_ref = {"max_diff_steps": round(float(d.max() / step), 2),
                "pred_mismatch": int((outs["ORT_ENABLE_ALL"].argmax(1)
                                      != outs["ORT_DISABLE_ALL"].argmax(1)).sum())}
print("ORT fused INT8 vs ORT QDQ reference:", fused_vs_ref)
print("wrote", results.write("onnx_diagnosis", {
    "note": __doc__.strip(), "n_images": len(X), "output_quant_step": step,
    "by_level": rows, "ort_fused_vs_ort_reference": fused_vs_ref}))
