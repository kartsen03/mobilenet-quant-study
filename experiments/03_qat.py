"""Step 3: quantization-aware training (QAT), plus a fine-tuning control.

QAT: start from the FP32 checkpoint, fuse Conv+BN+ReLU in QAT mode (BN stays
live and is folded into the fake-quantized weights on every step), insert fake
quantization with PyTorch's default oneDNN QAT qconfig, fine-tune on the GPU,
freeze BN statistics and then the observers near the end, convert to real INT8.

Control: fine-tune the FP32 model with the identical schedule but *without* fake
quantization, then apply the same PTQ as step 2. If QAT beats this control, the
gain comes from training against quantization noise, not from extra epochs.
"""
import argparse
import copy

import _path  # noqa: F401
import numpy as np
import torch
import torch.ao.quantization as tq
import torch.ao.nn.intrinsic.qat as nniqat

from qstudy import data, evaluate, quant, results, train
from qstudy.model import MobileNetV2Cifar

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=0.005)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--freeze-bn-epoch", type=int, default=3, help="0-based epoch to freeze BN stats")
p.add_argument("--freeze-observer-epoch", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--skip-control", action="store_true")
args = p.parse_args()

train.seed_everything(args.seed)
torch.backends.cudnn.benchmark = True
dev = "cuda"
splits = data.get_splits()
tr = tuple(t.to(dev) for t in splits["train"])
va = tuple(t.to(dev) for t in splits["val"])
tr_x = splits["train"][0]
va_x, va_y = splits["val"]
te_x, te_y = splits["test"]
fp32_test = np.load(results.RESULTS / "preds" / "fp32_test.npy")

fp32 = MobileNetV2Cifar()
fp32.load_state_dict(torch.load(results.CKPT / "fp32.pt", weights_only=True))

# ---- QAT -----------------------------------------------------------------------
quant.set_engine()
qat = copy.deepcopy(fp32).train()
qat.fuse(qat=True)
qat.qconfig = tq.get_default_qat_qconfig(quant.ENGINE)
tq.prepare_qat(qat, inplace=True)
qat.to(dev)


def schedule(model, epoch):
    if epoch >= args.freeze_bn_epoch:
        model.apply(nniqat.freeze_bn_stats)
    # Observers learn activation ranges from training batches only; they are
    # switched off for each validation pass and for good from freeze_observer_epoch.
    model.apply(tq.enable_observer if epoch < args.freeze_observer_epoch
                else tq.disable_observer)


qat_hist = train.train(qat, tr, va, epochs=args.epochs, lr=args.lr, wd=args.wd,
                       batch_size=args.batch_size, seed=args.seed, device=dev,
                       amp_dtype=None, on_epoch_start=schedule,
                       before_eval=lambda m: m.apply(tq.disable_observer))
fake_quant_val = qat_hist[-1]["val_acc"]
qat_int8 = quant.convert(qat.cpu().eval())
pt = evaluate.predict(qat_int8, te_x)
pv = evaluate.predict(qat_int8, va_x)
size = evaluate.saved_size(qat_int8.state_dict(), results.CKPT / "qat.pt")
np.save(results.RESULTS / "preds" / "qat_test.npy", pt.astype(np.uint8))
np.save(results.RESULTS / "preds" / "qat_val.npy", pv.astype(np.uint8))

reloaded = quant.int8_skeleton()
reloaded.load_state_dict(torch.load(results.CKPT / "qat.pt", weights_only=False))
x = data.normalize(te_x[:64])
with torch.inference_mode():
    reload_exact = bool(torch.equal(qat_int8(x), reloaded(x)))

out = {
    "config": vars(args) | {"qconfig": f"get_default_qat_qconfig('{quant.ENGINE}')",
                            "precision": "FP32 (no autocast) with fake quantization"},
    "history": qat_hist,
    "val_acc_fake_quant_gpu": fake_quant_val,
    "test_acc": evaluate.accuracy(pt, te_y),
    "val_acc": evaluate.accuracy(pv, va_y),
    "test_pred_changed_vs_fp32": int((pt != fp32_test).sum()),
    "size_bytes": size,
    "reload_bit_exact": reload_exact,
}
print({k: v for k, v in out.items() if k not in ("history", "config")})

# ---- control: same fine-tuning without fake quant, then PTQ ----------------------
if not args.skip_control:
    ft = copy.deepcopy(fp32).to(dev)
    ft_hist = train.train(ft, tr, va, epochs=args.epochs, lr=args.lr, wd=args.wd,
                          batch_size=args.batch_size, seed=args.seed, device=dev,
                          amp_dtype=None)
    ft = ft.cpu().eval()
    ft_fp32_test = evaluate.accuracy(evaluate.predict(ft, te_x), te_y)
    calib_idx = data.calibration_indices(len(tr_x), 1024, 0)
    ft_ptq = quant.ptq(ft, tr_x[calib_idx], quant.make_qconfig())
    out["control_finetune_then_ptq"] = {
        "history": ft_hist,
        "fp32_test_acc": ft_fp32_test,
        "ptq_test_acc": evaluate.accuracy(evaluate.predict(ft_ptq, te_x), te_y),
        "ptq_val_acc": evaluate.accuracy(evaluate.predict(ft_ptq, va_x), va_y),
    }
    print("control", {k: v for k, v in out["control_finetune_then_ptq"].items()
                      if k != "history"})

print("wrote", results.write("qat", out))
