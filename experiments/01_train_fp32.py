"""Step 1: train the FP32 MobileNetV2 baseline on the GPU.

Records test accuracy (GPU and CPU inference), model size on disk, and the
per-epoch training log. CPU latency is measured separately by latency.py so that
all models are timed in the same interleaved run.
"""
import argparse
import time

import _path  # noqa: F401
import numpy as np
import torch

from qstudy import data, evaluate, results, train
from qstudy.model import MobileNetV2Cifar

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="fp32")
args = p.parse_args()

train.seed_everything(args.seed)
torch.backends.cudnn.benchmark = True
dev = "cuda"
splits = data.get_splits()
tr = tuple(t.to(dev) for t in splits["train"])
va = tuple(t.to(dev) for t in splits["val"])
te_x, te_y = splits["test"]

model = MobileNetV2Cifar().to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"params {n_params:,}")

t0 = time.perf_counter()
history = train.train(model, tr, va, epochs=args.epochs, lr=args.lr, wd=args.wd,
                      batch_size=args.batch_size, seed=args.seed, device=dev,
                      warmup_epochs=1, amp_dtype=torch.bfloat16, channels_last=False)
train_minutes = (time.perf_counter() - t0) / 60

model = model.float()
pred_gpu = evaluate.predict(model, te_x, device=dev)
model = model.cpu().eval()
pred_cpu = evaluate.predict(model, te_x, device="cpu")
val_pred_cpu = evaluate.predict(model, splits["val"][0], device="cpu")

ckpt = results.CKPT / f"{args.tag}.pt"
size = evaluate.saved_size(model.state_dict(), ckpt)
fused = MobileNetV2Cifar()
fused.load_state_dict(model.state_dict())
fused.eval().fuse()
size_fused = evaluate.saved_size(fused.state_dict(), results.CKPT / f"{args.tag}_fused.pt")
pred_fused = evaluate.predict(fused, te_x, device="cpu")

(results.RESULTS / "preds").mkdir(parents=True, exist_ok=True)
np.save(results.RESULTS / "preds" / f"{args.tag}_test.npy", pred_cpu.astype(np.uint8))
np.save(results.RESULTS / "preds" / f"{args.tag}_val.npy", val_pred_cpu.astype(np.uint8))

out = {
    "config": vars(args) | {"optimizer": "SGD nesterov m=0.9, cosine, 1 warmup epoch",
                            "precision": "bf16 autocast, NCHW",
                            "augmentation": "pad-4 random crop + horizontal flip (GPU)"},
    "params": n_params,
    "train_minutes": round(train_minutes, 2),
    "history": history,
    "test_acc_gpu": evaluate.accuracy(pred_gpu, te_y),
    "test_acc": evaluate.accuracy(pred_cpu, te_y),
    "val_acc": evaluate.accuracy(val_pred_cpu, splits["val"][1]),
    "test_acc_fused": evaluate.accuracy(pred_fused, te_y),
    "fused_vs_unfused_pred_mismatch": int((pred_fused != pred_cpu).sum()),
    "size_bytes": size,
    "size_bytes_bn_folded": size_fused,
}
print({k: v for k, v in out.items() if k not in ("history", "config")})
print("wrote", results.write(args.tag, out))
