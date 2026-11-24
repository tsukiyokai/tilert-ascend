# M32 late-layer attention on frozen teacher inputs

A3-194 die 0 executes rank-0 attention for layers 66–73 on 32 real teacher
inputs from the validated M1 trajectory. These cover C8 index producer 66
with consumers 67–69, then BF16 producer 70 with consumers 71–73. Checkpoint
projections, native Prolog/indexer and the production shared-FIA binder run
through V-up projection, with O projection and TP reductions excluded.

The [independent CPU audit](audit.json) reads all eight raw output files:

- Shared and per-layer rebuilt FIA match byte for byte for attention and
  pre-O: 16 pairs, 786,432 BF16 values. Saved output storage is disjoint.
- All 19 new cache fields, 307,264 bytes per side, match historical M1 bytes.
- All 256 query selections are causal, unique and set-equal to M1.
  Consumers preserve their producer's ordered indices. Relative to M1,
  four ordered entries differ throughout group 66 and six throughout group 70.
- All output queries are finite and nonzero. The eight Q tensors and new
  latent tensors are distinct, and the two producer selections differ.

FIA versus SFA diagnostics use `||FIA−SFA||₂ / ||SFA||₂`:

| Layer | Attention NRMSE | Pre-O NRMSE | Pre-O maxabs |
| --- | ---: | ---: | ---: |
| 66 | 0.00135224 | 0.00304061 | 0.001953125 |
| 67 | 0.00159552 | 0.00312481 | 0.001953125 |
| 68 | 0.000973759 | 0.00319462 | 0.0009765625 |
| 69 | 0.00139590 | 0.00295330 | 0.0009765625 |
| 70 | 0.00126305 | 0.00302771 | 0.00048828125 |
| 71 | 0.00113825 | 0.00224233 | 0.0009765625 |
| 72 | 0.00152089 | 0.00294491 | 0.001953125 |
| 73 | 0.00131810 | 0.00271208 | 0.0009765625 |

Attention maxabs is 0.0078125 at every layer. [Raw metrics](audit.json)
include cosine, differing-value counts and all 32 per-query NRMSE values.
These differences receive no tolerance or ULP classification.

[Execution](device-result.json) exits 0 with Torch peak allocated/reserved
153,901,056/174,063,616 bytes under a 512 MiB allocator quota. Context memory
is outside that quota. Two die 0 [preflight samples](preflight.json) show
0% AICore and 6519 MiB free on the shared host. There is no timing result.

This checks local attention with independently frozen M1 inputs at each
layer. It does not propagate M32 hidden states through TP16/MoE or check
final teacher labels. Historical M1 pre-O tensors were not saved; the local
numerical reference is current M32/SFA.

## Reproduce the CPU audit

Raw tensors remain under
`a3-194:trt-asc:/workspace/experiments/layer-major-local-attention-20261020/`.
From that container:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 /opt/trt-asc/venv/bin/python \
  /workspace/experiments/layer-major-local-attention-20261020/audit-device-01/audit.py \
  /workspace/experiments/layer-major-local-attention-20261020 \
  /tmp/late-attention-audit.json
```

[Source alignment](source-audit.md), [original probe](probe.py),
[CPU preparation](cpu-result.json), [launch](launch.py) and
[file hashes](manifest.json) bind the inputs and implementation.
