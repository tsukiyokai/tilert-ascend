# GLM-5.2 Ascend decoder

The BF16 functional backend implements incremental MLA, sparse indexing,
index sharing, dense and routed/shared FFNs, cache append, and request reset.
RMSNorm runs through TileLang Ascend on NPU. The other operations use Torch/NPU.
`tilert.load_backend("glm5_2_ascend")` selects this implementation.

The functional model has four layers, hidden size 256, four attention heads,
and four routed experts with top-2 selection. It preserves the first three
dense layers followed by MoE, and the full/full/full/shared index pattern.
It uses seeded BF16 weights. The CANN backbone and MTP implementation below
use the real W4A8C8 checkpoint and TP16. Their full numerical execution and
live HIXL handoff remain acceptance stages in the [A3 plan](../../../docs/plan.md).

## Run on a3-gs

Use the existing `trt-asc` container and `/opt/trt-asc/venv`. The source tree
is `/workspace/src/TileRT-Ascend`; TileLang Ascend is pinned to
`ascendc_pto@1d5daf0385156cbe0c499a16c10935c5190642a4`.

```bash
source /opt/trt-asc/activate.sh
cd /workspace/src/TileRT-Ascend
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 \
  python -m tilert.models.glm_5_2_ascend.smoke --device cpu \
  --output /workspace/experiments/glm52/smoke-cpu.json
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACL_OP_INIT_MODE=1 \
  ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python -m tilert.models.glm_5_2_ascend.smoke --device npu:0 \
  --output /workspace/experiments/glm52/smoke-npu.json
```

## Functional contract

- `DecodeEngine(config, state_dict, device, capacity)` accepts BF16 HF-layout
  weights and allocates fixed-capacity per-layer caches.
- `step(token, trace=True)` consumes one token, appends its KV, and returns
  next-token logits, layer outputs, selected context indices, and expert IDs.
- `reset()` clears valid lengths; the next request overwrites the valid prefix.
- Latent KV and per-token kvB projections are retained. Projection caching
  preserves the original BF16 rounding as context length changes.

The runner compares every step with the CPU HF eager implementation, using
identical weights and input tokens. It checks hidden states, logits, greedy
tokens, index/expert choices, and expanded KV. It also checks exact reset
replay and a different, shorter request. RoPE frequencies remain FP32;
RMS weights and FP32 router correction biases are nontrivial.
Equal scores prefer the lower context position or expert ID. The HF scoring
code is unchanged; a local reference adapter replaces its indexer/router
Top-K selection with stable descending order to apply the same tie rule.
CPU tensors and reset replay must match exactly. NPU tensors use the plan's
NRMSE ≤0.01 and cosine ≥0.999; logits additionally require max-absolute error
≤0.1. Greedy tokens and selection sets must match exactly on both devices.

Each initial request consumes eight prompt tokens and four generated tokens.
Its cache length is 12; five next-token predictions are available, including
the last prediction that has not yet entered the cache. Reports contain
versions, tolerances, individual checks, memory sizes, and RMS call counts.
These runs establish numerical and state correctness; they do not time TPOT.

## Checkpoint operators

`ModelSlimCheckpoint` in `weights.py` reads indexed slices from the real
W4A8C8 files. `expert(layer, expert, rank)` loads 128 intermediate channels
for one TP16 rank and selects that rank's down-projection scale-bias column.
`NpuExpert.from_checkpoint(shard)` builds a native CANN rank-local expert;
it merges gate/up and retains explicit BF16 SiLU/product boundaries.
Its INT8 projection inputs remain unchanged across native calls.

`NativeIndexer` in `ops/indexer.py` owns a 128-token paged index cache.
Post-RoPE inputs use INT8/Hadamard/FP16-scale storage or BF16 storage,
according to the layer's recipe. `inject` installs encoded prefix rows,
`append` writes one new key, and `select` calls the native top-2048 selector.
`append_many`/`select_many` support M=1/2/4 causal rows. `truncate` invalidates
rejected suffixes without clearing storage.
BF16 8K selection passes; INT8 encoding differences remain under diagnosis.

## Native CANN backbone

`NativeDecodeEngine` in `native_engine.py` connects the real W4A8C8
checkpoint to TP16, batch-one execution across all 78 backbone layers.
It uses CANN quantized projections, RMSNorm, RoPE, SwiGLU, sparse indexing,
compressed MLA and expert unpermute, plus HCCL reductions. `NativeMla` in
`ops/mla.py` keeps separate BF16 latent/RoPE pages and reuses the existing
vLLM-Ascend sparse-attention binding. Numerical execution is pending.

Routed experts load on first use for correctness work. `prepare_experts()`
makes all expert weights resident before profiling. The numerical runner
captures layer hidden states and logits, selected positions/experts and
new cache rows; it verifies reset replay. Independent HF reference traces
are required for a passing result. It records no performance measurements.

Use the [CANN operator and execution guide](../../../docs/evidence/ascend-decode/cann-operators.md).
Its reference contract matches the Ascend prefill's low-rank RMS epsilon,
main/indexer RoPE bases and local A8 partitioning.

`reference.py` retains the installed HF model/decoder/expanded-attention/cache
forward and supplies independent CPU quantized-linear, indexer and FFN
adapters. `reference_weights.py` reads raw safetensors with separate W4
unpacking and integer-dot arithmetic. `run_hf_reference.py` records all
layer outputs/logits, routes, index selections and new cache rows, then
checks exact reset replay. The native runner accepts only a completed,
passing reference with matching checkpoint, trace and prefix hashes.
CPU choices for fused rounding and TP reduction still need device calibration.
Full-model numerical execution remains pending.

## Verification and MTP

`NativeDecodeEngine.verify(tokens)` consumes M=1/2/4 positions and returns
batched logits, trace fields and `last_hidden` after the final RMSNorm.
`step(token)` retains the M1 interface. `truncate(P+A)` commits an accepted
prefix. MoE groups repeated experts across token rows and uses the CANN
unpermute operator with the original token/slot weights.
`use_expert_bank=True` selects a device route/count plan and two grouped
W4A8 matmuls per routed layer; the default retains the per-expert baseline.
`prepare_experts()` and `NativeMtp.prepare_experts()` load weights before
timing. Bank mode releases each layer's old expert cache before allocation.

`verify(..., full_logits=False)` returns distributed greedy `token_ids`.
Each rank publishes only its best score/ID, with global vocabulary order
breaking ties. The full-logit path remains the numerical baseline.
`validate_indices=False` keeps CANN-generated selected IDs on device;
the default validates their values on the host. Both paths have a matching
HF checker in `tests/glm52/run_native_verify.py`.

`NativeMtp(backbone)` owns layer 78 and its KV separately. It shares the
backbone embedding, loads its own logits head, and applies `rot.weight`
before hnorm. `forward(tokens, normalized_parent_hidden)` writes parent
positions: row p consumes hidden[p] and token[p+1]. Chained calls can pass
`external_indices` to reuse the first draft step's selected context.

`MtpRoundExecutor` coordinates seed, chained drafts, main verification,
target-conditioned MTP recomputation and accepted-prefix commit. Pending
tokens remain outside main KV. `AscendPDEngine` and the TP16 command loop
provide greedy request/reset control independently of the ROCm profile.
See [offline checks and device execution order](../../../docs/evidence/ascend-decode/offline-development.md)
for CPU evidence and the remaining numerical, profiling and live-PD gaps.
The [capture and request timing guide](../../../docs/evidence/ascend-decode/measurement.md)
binds numerical reports to frozen prompts and provides separate profiler
and producer/client TPOT entries.
Its capture receiver writes original TP16 main P/MTP P-1 prefixes through
HIXL without loading decode weights. The client retains the exact prompt
and prefill first token for subsequent HF/CANN comparisons.

The [operator reports](../../../docs/evidence/ascend-decode/README.md)
record numerical coverage and pending checks. Runners live under
`tests/glm52/`; hardware calibration lives under `tests/a3-calibration/`.
