# CANN MLA Prolog arithmetic

On `a3-gs:trt-asc`, a single die executes the real layer-0, logical-rank-0
TP16 shard: four attention heads, W8 Q/KV projections, BF16 latent/RoPE
cache and INT8 normalized queries. The input is a frozen checkpoint
embedding, not a reduced model. This isolates the first arithmetic boundary
without loading the full model or creating an HCCL group.

## First difference

At position 0, ordinary CANN RMSNorm keeps normalized values in FP32
through gamma multiplication and casts once to BF16. HF's two-stage
normalization differs in 1,551 of 6,144 values. The one-cast CPU formula
matches all 6,144 device values. M1 residual AddRmsNorm retains its distinct
BF16 sum and normalization boundaries.

With the device's normalized input frozen on both sides, the previous HF
adapter also rounds Q-A/KV-A projections and LoRA normalized values before
the fused Prolog does. The [same-input comparison](cann-prolog-same-input-20261017.json)
records the following differences.

| Boundary | Previous HF unequal values | FP32 Prolog oracle unequal values |
| --- | --- | --- |
| Input INT8 codes | 0/6,144 | 0/6,144 |
| Query INT8 codes | 232/2,048 | 0/2,048 |
| Query FP32 scale | 1/1 | 0/1 |
| BF16 latent | 121/512 | 0/512 |
| BF16 key RoPE | 0/64 | 0/64 |

The [device run](cann-prolog-device-20261017.json) matches all six exported
groups bitwise: query codes/scale, latent, key RoPE, absorbed NoPE query
and query RoPE. Torch peak allocation is 36,032,000 bytes, with a 1.5 GiB
allocator limit; driver/runtime allocations are outside this counter.

At [position 1](cann-prolog-pos1-device-20261017.json), input norm, query
codes and all four BF16 output groups also match bitwise. Query scale is
one FP32 ULP lower on device: `0x3b80f49a` versus CPU `0x3b80f49b`, relative
error `1.1833e-7`. The strict scale check remains failed in the raw report.
Replacing only that scale leaves all 4,096 BF16 indexer-projection outputs
unchanged in a CPU sensitivity check. The device FP32 normalized query is
not exported, so this observation does not identify a particular RMS/Div
primitive as the cause.

## Reference implementation

[The CPU helper](../../../tilert/models/glm_5_2_ascend/reference_prolog.py)
reads raw checkpoint weights independently of the production loader. W8
dequantization retains FP32 through LoRA RMS and gamma. Query quantization
uses separate `Div(amax,127)` and `Div(127,amax)` operations. NoPE Q-B
rounds to BF16 before absorption; rotary projections retain FP32 through
rotation and round at the output.

`run_hf_reference.py --mla-preprocess prolog_v3` selects this frontend,
keeping the HF decoder, cache, eager attention core and output projections.
The indexer consumes the same normalized-query quantization pair. Its
projection scale order retains the existing W8 reference contract and is
checked separately from the six Prolog outputs. Intermediate layer logits
use ordinary one-cast RMSNorm; the actual final residual norm retains the
M1 AddRmsNorm convention.

[The CPU checker](../../../tests/glm52/check_reference_prolog.py) compares
the helper directly with the saved device tensors; its
[position-0 checks](reference-prolog-cpu-20261017.json) pass 46/46.
Multirow execution and positions 1/8192/8193 also have scalar formula
checks. A separate [frontend check](prolog-frontend-cpu-20261017.json)
passes 62/62 on CPU with real layer-0/3 weights, HF causal masks and cache
updates at positions 0/1. It holds input norm fixed across both calls and
does not replace the full-model trajectory check.
The [full CPU reference](hf-prolog-cold-20261017.json) consumes teacher
tokens `[154820,198]` and passes 1,273/1,273 two-step/reset checks,
including all layer logits, choices and cache rows. Full CANN comparison
and 8K continuation remain the numerical acceptance work.
The next [single-die MLA check](cann-mla.md) consumes these actual device
outputs and passes the sparse attention/value-projection comparison at
positions 0/1.

## Reproduce

Use the unified server environment and a die with room for its runtime and
the bounded allocation. The input manifest, tensor hashes, installed
sources and operator schema are saved in the report.

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  python tests/glm52/check_cann_prolog.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --trace /workspace/experiments/glm52/hf-cold-current-20261017-attempt-01/step-00000.pt \
  --device npu:15 --output /workspace/offline/prolog-replay/device.json
```

`--cpu-only` checks inputs without initializing a device. Changing the
trace to `step-00001.pt` checks the next frozen embedding at position 1,
including nonzero rotary angles and untouched cache prefix/tail. Complete
decode acceptance still requires the plan's layer logits, choices, KV,
reset and 8K independent-trajectory checks before profiling.
