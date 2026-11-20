# Concepts, in plain language

Notes written at each checkpoint of the study. Numbers come from `results/*.json`.

## Checkpoint 1: PTQ basics

### Scale and zero-point
An INT8 tensor stores small whole numbers (0..255 for activations, -128..127 for
weights) plus two constants that say what real value each integer stands for:

```
real value  ≈  scale × (q − zero_point)

  real:   -0.5 ............ 0.0 ................................ 2.0
            |                |                                    |
  uint8:    0               51                                   255

  scale      = (2.0 − (−0.5)) / 255  ≈ 0.0098   (size of one integer step)
  zero_point = 51                               (the integer that means exactly 0.0)
```
Anything below −0.5 or above 2.0 is clipped to 0 or 255. Weights here use a
symmetric range (zero_point = 0) with one scale per output channel; activations use
an affine range (any zero_point) with one scale per tensor.

### Calibration
Weights are known ahead of time, but activation ranges depend on the input. So we run
1,024 training images through the FP32 model with *observers* attached that record
each activation tensor's values, then pick scale/zero_point from them.

```
1,024 train images ──► FP32 model + observers ──► range per tensor ──► scale, zero_point
```
A min/max observer uses the extreme values (one outlier stretches the range and wastes
resolution); the histogram observer searches for the clipping range that minimizes
the total rounding + clipping error.

### Layer fusion
```
before:  Conv ─► [tensor] ─► BatchNorm ─► [tensor] ─► ReLU ─► [tensor]
after:   ConvReLU (BN folded into the conv weights and bias) ─► [tensor]

w' = w × γ/σ      b' = β − γ·μ/σ       (σ = sqrt(running_var + ε))
```
In inference BN is just a per-channel multiply-and-add, so it can be absorbed into
the conv. Fusion means one INT8 kernel instead of three ops and two intermediate
tensors that would each need their own quantization (and their own rounding error).
On FP32 alone, folding BN cut CPU latency from 8.28 ms to 7.01 ms (1 thread).

### Why depthwise convs are usually hard to quantize
```
pointwise 1×1 conv (e.g. 96→576): each output sums 96 products → rounding errors average out
depthwise 3×3 conv:               each output sums 9 products from ONE channel → they don't
```
1. Few weights per output channel, so per-channel ranges can differ a lot; after BN
   folding (γ/σ differs per channel) a few channels can be far larger than the rest.
   With one scale for the whole tensor, small channels get only a handful of integer
   levels, or round to zero entirely.
2. Each output depends on just 9 inputs, so errors don't average out.
3. Activations still get one scale per tensor, and depthwise outputs keep each
   channel's own range, so the same mismatch hits the activations.

**What we measured (this model):** the depthwise weights were well-behaved. Widest
channel / median channel = 2.4× (median over the 17 depthwise layers), at most 5.4×.
No channel rounded to zero under per-tensor INT8. Per-channel scales raised depthwise
weight SQNR from a median of 41.3 dB to 47.1 dB. Accuracy didn't move: every
weight/activation observer combination landed at 93.07–93.26% test, within
calibration-seed noise. The classic failure is reported for ImageNet MobileNetV2 with
ReLU6. Our CIFAR model, trained with weight decay on the depthwise weights, never
developed those outlier channels, so INT8 is mild here. Lower bit-widths (or analog
noise) would bring the problem back, which is what steps 4–5 probe.

## Checkpoint 2: QAT and sensitivity

### Why QAT can recover accuracy
PTQ rounds weights and activations *after* training, so the network never got a
chance to adapt to the rounding. QAT inserts "fake quantization" into training:
the forward pass rounds to the INT8 grid exactly as deployment will, while the
backward pass treats the rounding as identity (the straight-through estimator).

```
forward:  w ──► round to INT8 grid ──► conv ──► round output ──► ...   (sees the real error)
backward: gradient flows straight through the rounding as if it weren't there
```
The weights drift toward values that land well on the grid, and the learned
activation ranges settle where clipping and rounding hurt least. BN statistics are
frozen near the end so the folded weights stop moving, then the observers are frozen
so the scales match the final weights.

**What we measured:** QAT reached 93.44% against FP32's 93.31%. The control matters:
the same 5 extra epochs *without* fake quantization, followed by PTQ, give 93.36%.
PTQ lost only 0.21 points to begin with, so there was little for QAT to recover, and
most of its edge is just extra training. QAT matters when PTQ loses a lot: lower
bit-widths, or hardware with coarse ADCs.

### What sensitivity analysis says to protect
Quantize one layer at a time and measure how much the output moves (logit SQNR;
lower = more damage):

```
most sensitive:  stem (38.5 dB) ─ blocks.0.dw (39.2) ─ blocks.6.project (41.5) ─ fc (41.6)
most tolerant:   blocks.15.expand (63.8) ─ blocks.14.expand (60.8) ─ blocks.15.dw (60.6)
```
- **First layers.** The stem sees the raw image, and its error passes through every
  later layer. Keeping the first and last layers at higher precision is the usual
  first move on any accelerator.
- **Narrow projections feeding residual adds** (no ReLU, signed outputs) are next.
- **Late, wide layers** have redundancy to absorb rounding error.

Keeping the top 3 in FP32 cost +0.05 MB and recovered most of PTQ's small loss
(93.25%). On analog hardware the same ranking says which layers to map to
digital/high-precision tiles, or to program with extra verify cycles.

### Why INT8 QAT didn't help against weight noise
INT8 rounding error has σ ≈ 0.23% of each layer's max weight. The simulated analog
noise that hurts is 2–10%, about 9–44× larger. QAT only trained against the small
one. To get noise robustness, inject noise of the target size during training.
