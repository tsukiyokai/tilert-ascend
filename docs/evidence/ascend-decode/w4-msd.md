# CANN W4 MSD rounding

Run: 2026-10-16, `a3-187:trt-asc`, CANN 9.2.0 master20260917,
GLM-5.2-W4A8C8. [Runtime](a3-187-runtime-20261016.json).

For frozen INT8 activation `a`, signed INT4 weights `W`, FP32 channel scale
`sW`, token scale `sA` and checkpoint correction `b`, the installed GMM uses:

```python
h = floor_divide(a.int32(), 16)
l = (a.int32() & 15) - 8
H = ((h @ W.int32().T).float() * sW).half().float()
L = ((l @ W.int32().T).float() * sW).half().float()
z = H * 16
z = z + L
z = z + b
y = (z * sA[:, None]).bfloat16()
```

The FP32 additions are separate. Since `16h+l=a−8`, the checkpoint
correction restores the offset. For TP16 down projections, each K128 shard
selects one of the sixteen correction columns; gate/up select one column.
The signed-dot formula remains the ideal arithmetic comparison.

Eight experts selected by the CPU HF layer-3 trace were tested on rank-0
weight shards. All 16 gate/up and down projections match this formula
bitwise at [M1](quantized-msd-input-20261016.json), and all 16 match at
[M4](quantized-msd-m4-input-20261016.json). M4 repeats the same real input
four times to exercise that kernel shape. Ideal signed-dot NRMSE is
0.000483–0.001080. The M1 report retains one FP32 token-scale check failure
of `9.09e−13`; projection checks use the actual device scale.

The independent CPU reader now emulates these rounding boundaries.
[Operator checks](check-weights-msd-20261016.json) pass 186/186, including
M1/M4, rank-0/rank-15 shards, exact MSD projection checks and an analytic
rounding witness: zero INT8 input with external scale 1, `sum(W)=−127`,
`sW=23×2⁻²⁴`, `b=−23368×2⁻²⁴` yields `−2⁻²¹` under MSD and zero under the
ideal dot. Dynamic quantization of an actual zero row supplies scale zero.

## Installed source

Directory: `/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul/op_kernel/`.

| File | Contract | SHA256 |
| --- | --- | --- |
| `grouped_matmul_antiquant_a8w4_msd_pre.h` | Lines 180–207: high/low split and offset | `169c9b2e3f07bca932eb031db21a5a044d018aa201d3ec11cb2f588c09af1616` |
| `grouped_matmul.cpp` | Line 293: FP16 intermediate C | `12e4d86d672e88c4cdda1cb5ea5ab8e1298ac205ebf76c708363930c437d562a` |
| `grouped_matmul_antiquant_a8w4_msd.h` | Lines 356–363, 470–530: scaled partials and recombination | `a8b7c446d5373f5f50d3ee22bdf6631a8468973da420791bb97ccf0e2f351fbb` |

Server diagnostics are in `/workspace/experiments/glm52/`:

| Script | SHA256 |
| --- | --- |
| `quantized-msd-input-20261016.py` | `7cb498d5353a6abd5e3b58ccc52e17261c9d4eda9f24caffac0716dc1146a737` |
| `quantized-msd-m4-input-20261016.py` | `5cdde9e794ad05656239e543b36a9d343dee3ff0f800fc80cb7477acf47c53f6` |

## Dense-layer boundary

A [TP16 layer-1 check](dense-layer1-boundaries-20261016.json) starts from
the same HF hidden input. Input norm and cache latent match exactly;
attention NRMSE is 0.000642, post-attention hidden 0.000140, and FFN output
0.004942. Reconstructing the HF layer from the independent CPU reader
matches its saved output exactly. All numeric boundaries satisfy the 1%
criterion. Sixteen exact single-key diagnostics retain differences below
`1.16e−24`. Full-model parity is evaluated separately.

The [same-input sensitivity check](dense-layer1-sensitivity-20261016.json)
decomposes `F_N(x_N)−F_H(x_H)` into
`[F_N(x_N)−F_H(x_N)] + [F_H(x_N)−F_H(x_H)]`, using the saved post-attention
BF16 rows. The first term is exactly zero; the second accounts for the
entire 0.004942 FFN NRMSE. This closes the dense FFN arithmetic branch for
that input. The saved two-step device trace is reused for the MSD oracle
comparison; its production code has not changed.
