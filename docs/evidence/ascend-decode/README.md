# A3 minimal decode validation

Run: 2026-10-16, `a3-gs:trt-asc`, one `Ascend910_9392` die. Python 3.11.12,
Torch 2.12.0+cpu, torch-npu 2.12.0, Transformers 5.14.1, CANN 9.2.0.
[Source hashes](source.json) match the local and server trees used for the runs.
[TileLang installation](toolchain/installed.json) records the pinned source
and compiler libraries; its small local patch fixes version lookup from a
different working directory.

## Results

| Run | Checks | Maximum NRMSE | Minimum cosine | Maximum logits error |
| --- | --- | --- | --- | --- |
| [CPU](smoke-cpu.json) | 1,729/1,729 | 0 | exact tensors | 0 |
| [A3 NPU](smoke-npu.json) | 1,729/1,729 | 0.008435 | 0.9999645 | 0.0078125 |

The model has four layers, hidden size 256, four attention heads, and four
routed experts with top-2 selection. It uses seeded BF16 weights and the
dense/dense/dense/MoE, full/full/full/shared layer pattern. The runs consume
12 tokens, replay those 12 after reset, then consume a distinct seven-token
request. All greedy tokens, context selections, expert selections, cache
lengths, and reset replay checks pass. CPU tensors and NPU replay are exact;
NPU versus HF uses NRMSE ≤0.01, cosine ≥0.999, and logits max-abs ≤0.1.

HF eager scoring is retained. Equal scores prefer lower context position or
expert ID on both sides; the reference adapter only changes Top-K ordering.
RoPE frequency buffers remain FP32 after casting model weights to BF16.
Nontrivial RMS gamma and FP32 expert correction bias exercise their semantics.

The NPU path launches TileLang RMSNorm 527 times. Other operations use
Torch/NPU. Peak tensor allocation is 22,550,016 bytes, and peak allocator
reservation is 25,165,824 bytes; these counters exclude driver/context memory.
Weights occupy 5,454,864 bytes and explicit KV buffers 188,416 bytes.
The run validates this reduced BF16 model. Full W4A8C8 weights, TP16, HIXL,
MTP, 8K-context parity, and TPOT remain separate acceptance stages.

## Quantized operators and PD

The full [GLM-5.2 vLLM P+D baseline](vllm-baseline-20261017/README.md)
completes 32 serial 8K-to-1K requests after three warmups on a3-gs.
With TP16, MTP3 and four-row decode graph replay, client TPOT is
16.390 ms p50 and 18.295 ms p95, averaging 60.555 tok/s.

The [2026-10-17 CPU contract suite](cpu-contracts-20261017/summary.json)
passes all eight checks in the unified `a3-gs:trt-asc` environment: MTP state,
round and session transitions; token-delivery timing; causal verify layout;
profiling contexts; HF adapter metadata; and the 16-rank connector protocol.
The connector accepts an entire 8192-token prefill with chunking enabled
and rejects a partial chunk. These checks use no NPU execution or HIXL WRITE.

| Component | Evidence | Result |
| --- | --- | --- |
| Real checkpoint loading and projections | [Weight reports](weights/README.md) | CPU packing/slicing 102/102; native NPU projections 172/172 |
| Complete rank-local expert | [Expert chain](weights/expert-chain.json) | 70/70; maximum output NRMSE 0.003695; rank 0/15 partitions run on one die |
| BF16 sparse indexer | [8K selection](indexer.json) | Positive and negative head-weight cases match all 2048 selected indices |
| INT8 sparse indexer | [8K selection](indexer.json) | 419 key values and one query value differ from the CPU encoder; positive-weight top-k intersection 2046/2048; diagnosis pending |
| TileLang packed W4 GEMV | [Compilation](tilelang-w4a8-xor/compile-result.json) | N256/K6144 and N6144/K128 compile; corrected XOR aliasing awaits device verification |
| Ascend receive layout v2 | [CPU source-binding checks](ascend-layout-v2-source-binding.json) | 196 regions/rank; 79 split latent/RoPE pairs, 22 index keys and 16 scales; 8K transfer 775,159,808 B/rank |

The [CANN adapter](cann-model.md) has TP16 M=1/2/4 execution entries and
teacher-forced hidden/logit/cache checks. Diagnostic traces capture actual
index and expert selections; saved wide rows are compared directly with HF.
An independent HF trace
generator retains the installed model/decoder/attention/cache and supplies
CPU quantization adapters. [Metadata checks](hf-reference-metadata.json)
pass 34/34; [raw-byte checks](hf-reference-weights-metadata.json) cover 24
real W4 rows and 144 correction groups. Neither metadata check runs Torch.
The [78-layer HF short reference](hf-full-short-20261016.json) now passes
1,273/1,273 numerical and reset checks for two generated tokens. This is a
CPU reference; complete TP16 NPU parity remains pending. Device validation
uses [a3-187:trt-asc](a3-187-environment-20261016.json), with the original
container image and [unified runtime](a3-187-runtime-20261016.json).
The [TP16 two-token run](native-full-short-vendor-20261016.json) completes
all 78 layers, predicts `[198,59]`, and replays reset exactly. The
[W4 MSD audit](w4-msd.md) restores two FP16 partial products in the CPU
oracle; 32/32 frozen M1/M4 projections match bitwise. The
[A8 audit](a8-rounding.md) restores `Div(127,amax)` and the separately
rounded returned scale; 184 frozen cases match bitwise. Expanded operator
checks pass 189/190, retaining one random M4 INT8-code difference.
The [RoPE audit](rope-rounding.md) preserves FP32 rotary intermediates with
one BF16 output cast; all 42 frozen index keys match the device bitwise.
Same-input layer checks have [38/1,572 failures](same-input-layers-rope-20261016.json),
with all cache fields passing and five route-switch layers remaining.
The corrected [HF reference](hf-full-short-rope-20261016.json) passes
1,273 reset/control checks and predicts `[198,17351]`. The saved native
trace has [437/2,934 failed checks](recompare-short-rope-20261016.json),
including the second greedy token. Step 0's first two decoder outputs match
exactly. Full numerical acceptance and profiling remain pending.
The receive-layout checks use tensor metadata doubles on the server CPU;
live NPU binding and HIXL transfer remain pending.
The [HIXL bridge](hixl-build.json) compiles and links against the copied
CANN 9.2 SDK. Its [prepared 8K configuration](prefix-8k-setup-20261016.json)
records a successful library load on the replica; device initialization,
registration and WRITE remain pending. The
[8K input](prompt-8192-code-review-metadata.json) contains actual runtime
source and is bound to its tokenizer output and source hashes.

INT8 indexer cache differences originate in encoding; the inject/append
comparisons retain those same differences. The BF16 path skips Hadamard and
A8 conversion. It therefore does not validate INT8 encoding. Both modes
check 8192 keys, top-2048, page crossing, reset, and input preservation.

The W4 kernel's XOR destination must have separate UB storage from its
source. CANN 9.2's `xor_membase_impl.h:41–50` writes the destination before
its next source read. The [initial numerical report](tilelang-w4a8-functional.json)
uses aliased storage; the current generated
[gate/up](tilelang-w4a8-xor/n256-k6144.cpp) and
[down](tilelang-w4a8-xor/n6144-k128.cpp) kernels use a separate destination.
The kernel reads packed weights directly into UB and performs no GM
weight expansion.

The original target container received a
[Docker stop](container-state-20261016.json) sent SIGTERM at
2026-10-16 06:52:45 UTC, then SIGKILL after ten seconds.
`OOMKilled=false`; the container exited at 06:52:56 UTC.

The [offline development report](offline-development.md) records the newer
M=1/2/4 CANN/MTP implementations, device MoE candidate, independent MTP
oracle, CPU control checks and revised runtime traffic budget. Those
implementations await full numerical device execution and profiling.

## SFA initialization

Initialize the sparse-attention binding when constructing `NativeMla`,
before projection/indexer kernels populate KV. The recorded
[production-chain differential](mla-bootstrap-20261016.json) uses the same
finite query and single valid KV row: early binding satisfies the single-key
identity; first-use binding violates it. Neither case reads device tensors
on the CPU before the kernel. The fix adds no decode-loop synchronization.
TP16 layer-zero boundary checks pass [128/128](mla-layer0-fixed-20261016.json),
versus [96/128](mla-layer0-boundaries-20261016.json) before initialization moved.
These checks compare local projections and reductions using the same actual
inputs; full-model parity is a separate check.

The [unified activation script](toolchain/runtime-activate-20261016.sh) also
registers the installed vLLM `custom_transformer` vendor before NPU work.
The [two-key production check](sfa-vendor-20261016.json) binds actual library
hashes and compares SFA with explicit NoPE + RoPE scores. Including this
vendor changes SFA NRMSE from 0.340743 to 0.001059; all six checks pass.
Without it, lookup falls through to builtin `libopapi_transformer` and the
recorded probabilities follow NoPE-only scores.

## RMSNorm precision

The full-model CANN baseline preserves HF's BF16 normalized-row boundary
before multiplying gamma. Unit-gamma CANN RMSNorm followed by BF16 gamma
multiplication matches HF exactly for the four recorded layer-zero
input/q/kv/post cases; single fused RMSNorm differs in input/q.
[Comparison](rmsnorm-rounding-20261016.json). The unit weights are allocated
once for widths 6144, 2048 and 512. Future fusion must retain this rounding.

[Ten NPU cases](rmsnorm-npu.json) cover random and zero input at
H=64/256/512/2048/6144; every BF16 result matches the CPU reference exactly.

The initial bare A3 `Rsqrt` produced `0.982421875` where the FP32 reference
was `0.9847186207771301`, a relative error of −0.002332389. The same probe
found zero BF16 cast mismatches. [Probe](debug-rmsnorm.py),
[raw result](debug-rmsnorm.json), and [initial RMS errors](rmsnorm-npu-before-refinement.json)
retain the evidence. The kernel now refines the inverse square root twice
with `y ← y × (1.5 − 0.5 × variance × y²)` before BF16 rounding.

CANN 9.2's `asc/impl/basic_api/dav_c220/kernel_operator_vec_unary_impl.h`
maps `Rsqrt<float>` directly to `vrsqrt`. Its high-precision RMSNorm in
`asc/impl/adv_api/detail/normalization/rmsnorm/rmsnorm_common_impl.h` uses
`Ln → multiply by −0.5 → Exp`. The [Rsqrt API constraints](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0030.html)
also describe the float precision limit.

## Reproduction

Run the commands in the [backend guide](../../../tilert/models/glm_5_2_ascend/README.md).
The full JSON reports contain configuration, versions, tolerances, every
comparison, and token sequences. The standalone [RMS check](check-rmsnorm.py)
runs with the repository on `PYTHONPATH` in the same activated container.
Performance was not timed. The [availability snapshot](availability.json)
records sixteen live vLLM workers, each holding about 58,680 MB as displayed
by `npu-smi`, and the stopped `trt-asc` container. HBM and HCCL calibration
use an exclusive window.
