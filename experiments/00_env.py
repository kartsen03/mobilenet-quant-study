"""Step 0: record the environment and fix the CIFAR-10 train/validation/test split.

The split is created once (stratified, seed 0: 45,000 train / 5,000 val from the
official training set; the official 10,000 test images are the test set) and
committed as splits/cifar10_split.json. Later runs verify its checksum.
"""
import json

import _path  # noqa: F401
import torch

from qstudy import data, env, results

if not data.SPLIT_FILE.exists():
    _, ty, _, _ = data.load_raw()
    split = data.make_split(ty, val_per_class=500, seed=0)
    data.SPLIT_FILE.parent.mkdir(exist_ok=True)
    data.SPLIT_FILE.write_text(json.dumps(split))
    print("wrote", data.SPLIT_FILE)
split = data.load_split()
splits = data.get_splits()
counts = {k: torch.bincount(v[1], minlength=10).tolist() for k, v in splits.items()}

info = env.collect()
cpu_sets = env.cpu_sets()
summary = {
    "split": {k: v for k, v in split.items() if k != "val_indices"},
    "class_counts": counts,
    "cpu_sets": cpu_sets,
    "note": ("PyTorch picks the CPU INT8 kernels from torch.backends.quantized.engine; "
             "this Windows build lists only the engines in quantized_engines."),
}
path = results.write("env", summary)
for k in ("python", "torch", "torch_cuda", "cudnn", "torchvision", "onnx", "onnxruntime",
          "quantized_engines", "quantized_engine", "cpu", "performance_cpus", "gpu",
          "power_plan", "git"):
    print(f"{k:18s} {info[k]}")
print("split", summary["split"])
print("class counts", counts)
print("wrote", path)
