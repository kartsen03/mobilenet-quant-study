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
