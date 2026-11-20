"""Step 6: export to ONNX and verify ONNX Runtime (CPU) against PyTorch.

- FP32: torch.onnx.export with the dynamo exporter (opset 18), from the trained
  (unfused) model; ORT folds BN itself.
- INT8 (PTQ, QAT, mixed precision): the TorchScript exporter (dynamo=False, opset
  17) writes the eager INT8 model as QDQ ONNX (QuantizeLinear/DequantizeLinear
  around float ops, carrying PyTorch's exact scales and zero-points). With full
  graph optimization ORT fuses those patterns into QLinearConv/QLinearAdd/QGemm,
  i.e. real INT8 kernels (MLAS), not float emulation.

Verification on all 10,000 test images (batch 1):
- FP32 tolerance: max |logit diff| <= 1e-4 (float reassociation only).
- INT8 tolerance: max |logit diff| <= 5 output quantization steps (fc output
  scale), and predictions may differ on at most 0.1% of images. PyTorch (oneDNN)
  and ORT (MLAS) round intermediate requantizations independently, so bit-exact
  equality is not expected. The threshold was set after a 256-image probe of the
  PTQ model (observed max 0.63 logits ~ 3 steps).
"""
import os

import _path  # noqa: F401
import numpy as np
import onnx
import onnxruntime as ort
import torch

from qstudy import data, evaluate, quant, results, zoo

os.environ.setdefault("PYTHONUTF8", "1")
quant.set_engine()
te_x, te_y = data.get_splits()["test"]
X = data.normalize(te_x)
mixed_name = results.read("mixed")["headline"]
names = ["fp32", "ptq", "qat", mixed_name]

out = {"note": __doc__.strip(), "ort_providers": ["CPUExecutionProvider"], "models": {}}
for name in names:
    model = zoo.load_torch(name)
    path = results.CKPT / f"{name}.onnx"
    if name == "fp32":
        torch.onnx.export(model, (X[:1],), str(path), dynamo=True, opset_version=18,
                          external_data=False, input_names=["x"], output_names=["logits"])
    else:
        torch.onnx.export(model, (X[:1],), str(path), dynamo=False, opset_version=17,
                          input_names=["x"], output_names=["logits"])
    onnx.checker.check_model(str(path))

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = torch.cat([model(X[i:i + 500]) for i in range(0, len(X), 500)]).numpy()
    got = np.concatenate([sess.run(None, {"x": X[i:i + 1].numpy()})[0] for i in range(len(X))])
    diff = np.abs(got - ref)
    pred = got.argmax(1)
    row = {"onnx_bytes": os.path.getsize(path),
           "ort_test_acc": evaluate.accuracy(pred, te_y),
           "torch_test_acc": evaluate.accuracy(ref.argmax(1), te_y),
           "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
           "pred_mismatch": int((pred != ref.argmax(1)).sum())}
    if name == "fp32":
        row["tolerance"] = "max_abs_diff <= 1e-4"
        row["pass"] = row["max_abs_diff"] <= 1e-4
    else:
        step = float(model.fc.scale if hasattr(model.fc, "scale") else model.fc.quant.scale)
        row["output_quant_step"] = step
        row["max_diff_in_steps"] = round(row["max_abs_diff"] / step, 2)
        row["tolerance"] = "max_abs_diff <= 5 output steps and pred_mismatch <= 10"
        row["pass"] = row["max_abs_diff"] <= 5 * step and row["pred_mismatch"] <= 10
    # Which kernels ORT actually runs after optimization
    so2 = ort.SessionOptions()
    so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    opt_path = results.CKPT / f"{name}.ort_opt.onnx"
    so2.optimized_model_filepath = str(opt_path)
    ort.InferenceSession(str(path), so2, providers=["CPUExecutionProvider"])
    ops = {}
    for n in onnx.load(str(opt_path)).graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    row["ort_optimized_ops"] = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
    out["models"][name] = row
    print(name, {k: v for k, v in row.items() if k != "ort_optimized_ops"}, flush=True)

print("wrote", results.write("onnx", out))
