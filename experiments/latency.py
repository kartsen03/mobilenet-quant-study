"""CPU latency per image (batch 1) for PyTorch and ONNX Runtime models.

Method:
- each thread count runs in its own process, pinned to that many performance
  cores (one logical CPU per P-core, so no SMT sharing and no E-cores);
- models are timed in interleaved rounds (shuffled order each round) so thermal
  drift and background noise hit every model equally;
- each round: 0.5 s warmup, then `runs` individually timed calls;
- reported: median over all rounds (plus p05/p95 and per-round medians).

Usage: python experiments/latency.py --models fp32_fused ptq --tag stage2
       ONNX models are named ort:<file stem in checkpoints/>, e.g. ort:fp32
"""
import argparse
import json
import os
import subprocess
import sys

import _path  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", required=True)
p.add_argument("--threads", type=int, nargs="+", default=[1, 6])
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--runs", type=int, default=100)
p.add_argument("--tag", required=True)
p.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
args = p.parse_args()

if args.child is None:  # parent: one child process per thread count
    from qstudy import env, results
    by_threads = {}
    for t in args.threads:
        cmd = [sys.executable, __file__, "--child", str(t), "--tag", args.tag,
               "--rounds", str(args.rounds), "--runs", str(args.runs), "--models", *args.models]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(r.stderr)
        by_threads[str(t)] = json.loads(r.stdout.strip().splitlines()[-1])
        for m, s in by_threads[str(t)]["models"].items():
            print(f"threads={t:2d} {m:14s} median {s['median_ms']:.3f} ms "
                  f"(p05 {s['p05_ms']:.3f}, p95 {s['p95_ms']:.3f})")
    print("wrote", results.write(f"latency_{args.tag}", {
        "method": __doc__.strip(), "rounds": args.rounds, "runs_per_round": args.runs,
        "batch_size": 1, "by_threads": by_threads}))
    sys.exit(0)

# ---- child: pin first, then import the numeric libraries --------------------------
from qstudy import env  # noqa: E402

T = args.child
cpus = env.performance_cpus()[:T]
env.pin_process(cpus)
os.environ["OMP_NUM_THREADS"] = str(T)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qstudy import data, evaluate, quant, results, zoo  # noqa: E402

torch.set_num_threads(T)
quant.set_engine()
x = data.normalize(data.get_splits()["test"][0][:1])  # one fixed test image

fns = {}
for name in args.models:
    if name.startswith("ort:"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = T
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(results.CKPT / f"{name[4:]}.onnx"), so,
                                    providers=["CPUExecutionProvider"])
        feed = {sess.get_inputs()[0].name: x.numpy()}
        fns[name] = (lambda s=sess, f=feed: s.run(None, f))
    else:
        model = zoo.load_torch(name)
        fns[name] = (lambda m=model: m(x))

rng = np.random.default_rng(0)
samples = {n: [] for n in fns}
round_medians = {n: [] for n in fns}
with torch.inference_mode():
    for _ in range(args.rounds):
        for name in rng.permutation(list(fns)):
            s = evaluate.time_calls(fns[name], warmup_s=0.5, runs=args.runs)
            samples[name] += s
            round_medians[name].append(round(float(np.median(s)), 4))

out = {"threads": T, "pinned_cpus": cpus, "torch_threads": torch.get_num_threads(),
       "models": {n: evaluate.summarize_ms(s) | {"round_medians_ms": round_medians[n]}
                  for n, s in samples.items()}}
print(json.dumps(out))
