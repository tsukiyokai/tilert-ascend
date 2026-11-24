# CANN model execution

[CannModel](../../../tilert/models/glm_5_2_ascend/cann_model.py) reuses the
pinned vLLM-Ascend model, weight processing and CANN operators. TileRT
owns prefix injection and the decode loop. `NPUWorker` initializes the
device and TP groups and loads the model; the adapter calls the model
directly, with no vLLM scheduler, serving process or speculative model.

The handoff state for serial decode is
`(main_KV[0:P], position=P, pending=prefill_first_token)`.
The first call consumes that pending token and produces its successor.
All P cache rows remain intact. Replaying the last prompt token would
require the prefill and decode arithmetic to agree at that position;
[the inspected norm and fused prolog paths differ](prefill-norm-prolog-20261017.json).

The adapter binds the existing paged latent/RoPE and INT8/BF16 index
layouts to 78 main attention layers and 21 indexers. It uses the existing
SFA and indexer metadata builders. M1 uses `DecodeOnly`; target widths 2/4
use the production `ChunkedPrefill` continuation path without a speculative
model. For M rows at P, Q is `[M]`, KV is `[P+M]`, slots are P through
P+M-1, and sparse mode 3 limits row i to keys below P+i+1. Logical TP rank
selects the weight shard; physical device selection remains independent.
Reset invalidates cache length. The next request replaces its valid rows.

BF16 TP partials accumulate in FP32 and cast back once to BF16. The
[TP16 comparison](cann-tp-sum-20261017.json) keeps identical local partials:
direct BF16 reduction differs across ranks in 6,015 of 6,144 coordinates
at the first output projection; the FP32 path produces identical BF16
outputs. With this default, all 79 hidden states match across sixteen
ranks in four cold forwards. Cold reset still differs, first at layer 6 for
step 0 and layer 5 for step 1. Call `shutdown()` to restore the TP group's
original reduction and stop the worker. Other input dtypes retain the
original reduction behavior.

The [device entry](../../../tests/glm52/run_cann_model.py) saves logits,
normalized hidden state, new cache rows and source hashes. Acceptance is
complete-prefix identity, one-row position advance, exact reset replay,
then next-token and numerical comparison against the supplied reference.
CPU API checks establish the pinned interfaces; full model execution and
PD serving integration require device acceptance. MTP follows M1 numerical
acceptance and retains its separate target-hidden and draft-cache state.

## Host verification

The adapter invokes the official platform registration before constructing
`EngineArgs`: its dictionary validation otherwise rejects C8 `int8` before
engine-config initialization can register it. The
[constructor check](cann-m1-init-cpu-20261017.json) passes eight cases with
real `EngineArgs` and device/worker allocation doubles. The
[previous constructor](cann-m1-init-negative-20261017.json) rejects the same
configuration in a fresh process. Neither test initializes an NPU.

The [width metadata checks](cann-target-m124-cpu-20261017.json) pass 15
cases, including positions 0, 1, 8192 and the page boundary at 8319 for
M=1/2/4. They exercise the installed SFA and C8/BF16 indexer builders,
capture wrapper arguments, verify shared-index shape and check truncation.
The kernel source's right-aligned causal bounds are recorded separately
from these CPU witnesses; device arithmetic remains unexecuted.

The [installed API probe](cann-adapter-cpu-api-20261017.json) constructs the
real engine configuration, cache bindings, SFA/indexer metadata and forward
context on CPU at positions 0, 1, 8192 and 8193. Torch-NPU remains
uninitialized. The [source check](cann-model-static-20261017.json) records
21 passing syntax and pinned-interface checks. The request, TP, MTP and
connector [regressions](cann-model-host-checks-20261017/check_ascend_pd.json)
pass 120 cases across five reports in that directory. These checks cover
the host interfaces; they do not execute weights or NPU kernels. The
[PD entry check](cann-model-host-checks-20261017/check_cann_pd_entry.json)
also verifies explicit Torch-NPU registration before NPU device parsing.

The [Native-arithmetic HF reference](hf-cold-current-20261017.json)
completes teacher tokens `[154820, 198]` and exact reset replay. Its raw
traces remain at
`/workspace/experiments/glm52/hf-cold-current-20261017-attempt-01`.
The [comparison check](cann-layer-comparison-cpu-20261017.json) accepts the
identical trace and rejects changed or missing intermediate hidden/logits.
For the cold device test, use `--tokens 154820 198 --steps 2 --layer-trace`.

The [single-die Prolog check](mla-prolog.md) isolates the first difference
against that reference. `--mla-preprocess prolog_v3` keeps fused MLA
intermediates in FP32 before the HF attention core. Select the arithmetic
contract matching the executed path when generating its oracle.
The [updated HF run](cann-fused-moe-20261017.json) passes all 1,273
two-step/reset checks with those teacher tokens. Its raw traces are at
`/workspace/offline/hf-fused-moe-20261017/two-step`; use
that directory for the CANN cold test's `--reference`. This is CPU
reference execution; full CANN numerical acceptance remains pending.
The [rank-local MLA check](cann-mla.md) passes 57 device checks using the
two frozen Prolog rows, including HF expanded-KV comparison and exact
cache preservation. It uses one die and no model loading or collectives.

The [router check](../../../tests/glm52/check_cann_router.py) passes 17
single-die checks at each of positions 0 and 1 using frozen layer-3 inputs.
It executes the production gate and top-k with unscaled probabilities.
The pinned ALLGATHER path casts those probabilities to BF16, combines
experts, multiplies the BF16 routed output by 2.5, adds shared output,
then performs TP reduction. `prolog_v3` now retains that scaling order;
the default Native reference retains scaling before combination. The
[report](cann-router-20261017.json) binds both runs and source hashes.

The routed W4 fused path retains FP32 GMM1 and SwiGLU intermediates. Its
A8 quantizer uses `amax/127`, reciprocal scale, then FP16 and INT8
nearest-even conversions. The
[frozen reference check](../../../tests/glm52/check_fused_moe_reference.py)
compares eight experts on rank 13 at two positions: down-projection and
weighted-combine outputs are bitwise equal to CANN in all four checks.
The [evidence](cann-fused-moe-20261017.json) binds the real single-die
capture, independent raw-weight reader and inspected kernel sources.
The full cold comparison still fails: step 0 final-logits NRMSE increases
from 0.1483 to 0.2686, and step 1 first exceeds the threshold in a dense
layer before MoE. Frozen local parity does not close full-model acceptance.

## Frozen 8K continuation

The [TP16 capture](cann-routed-8k-20261017.json) injects the complete HIXL
prefix at position 8192 and consumes teacher tokens `[6500, 382]`.
All ranks predict `[382, 262]`, matching HF. Numerical parity and exact
reset replay fail. Layer-4 capture retains the actual dispatch, both GMM
outputs, inverse row map and combine probabilities.

At position 8192, ranks 0, 12 and 15 first differ between initial execution
and reset at the fused GMM1/SwiGLU/A8 outputs. Their inputs, dispatch A8
codes/scales and expert counts are bitwise equal. All sixteen ranks retain
identical local stages at position 8193. Independent CPU reconstruction
using raw W4 weights identifies these incorrect expert rows:

| Rank | Initial execution | Reset |
| --- | --- | --- |
| 0 | 174 | 174, 221 |
| 12 | 132, 144 | None |
| 15 | 144 | 77, 132, 144, 174, 221 |

Of 48 expert-phase rows, 37 have exact fused A8 codes and scales; the other
11 scales are 10.03–36.80 times the reference. Repeated output can therefore
remain incorrect. All 48 GMM2 outputs match the independent reconstruction
from their actual fused inputs. All 64 captured calls reconstruct finalize
and post-TP output exactly; routing metadata passes 1,088 checks. The open
boundary is the fused GMM1/SwiGLU/A8 output in the original capture.

The [SwiGLU barrier fix](swiglu-barrier.md) closes that frozen local
boundary: all 11 incorrect scales equal the old gate amax/127, bitwise.
A vector barrier after the SwiGLU UB copy gives 832/832 exact expert rows
across rank-0/12/15 replays. The complete vendor overlay preserves original
host libraries and other operators; actual file reads identify the patched
kernel. Full TP16 reset and HF parity still require a new continuation.

The independent [layer-0 indexer replay](indexer-input-quant.md) reproduces
the first HF choice difference on one die. Fixed HF q/k/weights give exact
2,048-choice parity through the same CANN selector. The actual path first
differs at input DynamicQuant: ten half-amax values round to magnitude 63
instead of 64. With those actual A8 inputs, CPU Prolog reproduces all query
codes and its scale differs by one FP32 ULP. The
[native Div calibration](dynamic-quant-div.md) closes this local boundary:
the updated CPU reference reproduces all 6,144 input codes, 2,048 query
codes, final BF16 query and the complete 2,048-choice set. Complete TP16
continuation with the barrier overlay and updated HF reference remains
the full-model acceptance. The
[updated CPU reference](hf-8k-div-calibrated-20261017.json) passes 1,273
two-step/reset checks at 8K. Use
`/workspace/offline/cann-div-reference-20261017/hf-8k` as `--reference`.

The [frozen 8K attention differential](mla-8k-differential.md) closes the
same-input norm/dense branch under CPU merging: rank-ordered FP32
accumulation of device partials followed by BF16 cast is bitwise equal to HF.
The remaining layer-0 hidden-state NRMSE is 0.003889. An isolated CPU oneDNN
intervention changes BF16 attention arithmetic and reduces that hidden-state
NRMSE to 0.003595. Full-model numerical acceptance remains open.

## Device acceptance

After a complete HIXL prefix capture, stop the producer and receiver and
run the TP16 continuation when all ranks have sufficient HBM and
communication resources. Numerical checks may share a host. Frozen tensor
comparisons run on CPU; rank-local operator checks use one die with the
original TP16 shard shapes. Performance measurements require competing
device and communication work to be absent. The capture manifest must
include all 16 rank receipts and the original 8192 prompt token IDs.

The [processed-weight offload probe](weight-offload-20261017.json) reaches
the production prefetch offloader with one real W4 bank. Copying its
processed INT32/NZ parameter into CPU storage fails in Identity/TransData
with an invalid allocator stream. Separate snapshot checks fail at the
same conversion with implicit, early-enabled and disabled expandable
segments. These runs use die 15 and a 1.5 GiB allocator cap; full-model
prefetch has not executed.

The [layer-streamed check](streamed-decode.md) instead reconstructs raw
CPU weights one layer at a time. Its first layer passes 560/560 checks
across the real TP16 group on the shared host, using `AI_CPU` communication
and less than 908 MiB peak Torch allocation per rank. The bounded-RoPE
five-layer path passes 3,568/3,568 execution/reconstruction checks; its
independent HF comparison passes 70/80 and retains the original criteria.

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
cann_overlay=/workspace/offline/cann-workspace-owner-20261017/kernel-overlay/vendors/custom_transformer
cann_vendor=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
ASCEND_CUSTOM_OPP_PATH="$cann_overlay:$cann_vendor" \
  HCCL_BUFFSIZE=200 HCCL_OP_EXPANSION_MODE=AI_CPU \
  HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto \
  OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1 \
PYTHONPATH="$PWD:${PYTHONPATH:-}" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m torch.distributed.run --standalone --nproc-per-node=16 \
  tests/glm52/run_cann_model.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --steps 2 --capacity 9344 --layer-trace \
  --output /workspace/experiments/glm52/cann-m1-8k-choices-20261017
```

Every rank saves both forwards, verifies all injected rows remain exact,
and repeats from reset. `--reference` adds a matching independent HF trace
with the same prefix hashes and teacher tokens. Final logits, normalized
hidden and all 78 new KV rows are covered. `--layer-trace` adds the
79 hidden states, 78 layer logits, 78 sparse-index selections and 75 MoE
route selections used by the HF comparison. The report keeps
`precision_verified=false` until the coverage in [the plan](../../plan.md)
is complete.

[CannTrace](../../../tilert/models/glm_5_2_ascend/cann_trace.py) captures
each preceding layer's merged residual from the next input norm. A decoder
layer's first output alone is an FFN increment, not its completed hidden.
The last hidden and logits come from the original forward. The other 77
states receive one batched final norm/head projection on every TP rank;
their BF16 tensors are compared against the independent HF trace. The
[CPU hook checks](cann-trace-wide-cpu-20261017.json) pass 37 cases for
M=1/2/4 with no NPU initialization. Hooks and auxiliary projections run
only in this diagnostic mode and are excluded from TPOT measurement.

[CannChoices](../../../tilert/models/glm_5_2_ascend/cann_choices.py) copies
the actual SFA input indices and final routed-expert IDs. Each layer's
shared buffer is cloned before reuse. The first three dense layers retain
route `None`. HF comparison requires the same unique, causal top-2048
indices and eight valid expert IDs; short prefixes use their available key
count and ignore `-1` padding. Exact reset also preserves original dtype,
shape and ordering. Saved hashes bind the collector, SFA implementation
and final expert selector. The
[CPU capture checks](cann-choices-cpu-20261017.json) pass 22 cases with
synthetic call targets and real CPU tensors, including cleanup on errors.

## Target width checks

After the M1 continuation check, use
[run_cann_verify.py](../../../tests/glm52/run_cann_verify.py) with
`--width 2` or `4` and `--layer-trace`. It runs fixed teacher candidates
from the same complete prefix, comparing each token and layer with serial
M1. The captured g0 is the first candidate. These calls verify the target;
they do not run a drafter or measure speculative acceptance.

The entry checks same-width exact replay, each future input's effect on
earlier rows, and rollback while an old tail remains unwritten. The dirty
and clean branches consume the same replacement token at M1 from identical
retained KV. Only the clean branch clears the discarded tail, and both
stop below the old high watermark. Prefix contents, actual input arrays,
returned argmax IDs, logits, hidden and all new cache rows are retained.

The [CPU runner checks](cann-choices-runner-cpu-20261017.json) pass 11 cases
with a scalar model fixture, rejecting deliberately noncausal and stale-tail
reads and checking choice ownership across merge, slice and reset. The
[static/CLI checks](cann-verify-static-20261017.json) pass five
cases. Device cross-width agreement and matching independent HF comparison
remain separate acceptance steps; the report retains
`precision_verified=false` and `performance_measured=false`.

## Saved-trace comparison

Generate the HF oracle with the captured `consumed_tokens` and the same
main-prefix files. Then run [the offline comparator](../../../tests/glm52/compare_saved_cann.py)
on CPU; it reads all 16 rank reports and tensor files without loading
model weights. It verifies file hashes, checkpoint/prompt/teacher identity,
exact reset coverage, all 79 hidden states, 78 logits and new KV rows.
Choice-enabled traces also require all routing/index selections and their
three source files; older partial traces retain choice coverage `false`.
Results are written separately from both inputs. The
[host validator checks](check-compare-saved-cann-choices-20261017.json)
pass 39 cases, including changed hidden states on rank 15, missing sources
and identical but invalid selections in both compared traces.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/glm52/compare_saved_cann.py \
  --native /workspace/experiments/glm52/cann-m1-8k-choices-20261017 \
  --reference /workspace/experiments/glm52/hf-cann-m1-8k-choices-20261017 \
  --output /workspace/experiments/glm52/compare-cann-m1-8k-choices-20261017.json
```

For widths 2/4, [the wide comparator](../../../tests/glm52/compare_saved_cann_verify.py)
compares every row directly against its matching HF trace. It checks all
16 ranks, absolute positions, actual model inputs, captured prefix and
teacher identity, numerical tensors and enabled choices. Serial/wide
agreement is a separate result. The
[CPU validator checks](check-compare-saved-cann-verify-20261017.json)
pass 22 cases, including a BF16 counterexample where each comparison passes
1% NRMSE but wide/HF reaches 1.5625% and fails. These fixture checks verify
the validator; real target/MTP numerical acceptance remains pending.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/glm52/compare_saved_cann_verify.py \
  --native /workspace/experiments/glm52/cann-verify-m4-8k-20261017 \
  --reference /workspace/experiments/glm52/hf-cann-verify-m4-8k-20261017 \
  --output /workspace/experiments/glm52/compare-cann-verify-m4-8k-20261017.json
```

## PD serving

The [M1 PD entry](../../../tests/glm52/serve_cann_pd.py) uses the same
adapter with the existing HTTP request lifecycle and HIXL receiver.
Logical rank r uses physical die `(r+1)%16` to avoid HIXL self connections.
Its default memory guard is 0.4 for two resident models; actual combined
HBM use must be checked when starting P+D. The standalone continuation
must pass before this serving experiment and TPOT measurement. Finish the
request and confirm `/decode_status` is idle before stopping the service.

The [allocation audit](pd-capacity-20261017.json) counts two live D cache
allocations at capacity 9344: model cache and HIXL receive storage, each
884,166,656 bytes, totaling 1.647 GiB/rank. The standalone test has one.
The producer's prior successful eager startup reports 27.998 GiB for target
and drafter weights plus a 2 GiB KV budget. Live D's `0.4` setting lowers
its startup free-memory requirement; it does not cap weight or cache
allocations. Measure eager P and standalone D peaks before choosing
simultaneous residency. The frozen-prefix numerical test releases P first.

If a serving experiment also loads D after stopping P, retain g0 until D
is ready before delivering it to the client. A loading gap after g0 has
already been delivered belongs to request TPOT. The capture-only response
delivers g0 for input provenance and supplies no serving TPOT measurement.
