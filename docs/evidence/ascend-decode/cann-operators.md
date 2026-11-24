# CANN decode path and numerical gates

Target: GLM-5.2-W4A8C8, TP16, batch one, one token per step, CANN 9.2.0.
First complete numerical decode, then collect its profile, then optimize
the measured bottlenecks. Existing operators are the initial implementation.
Custom kernels require a demonstrated capability gap or an unmet budget
that modification of the existing implementation cannot resolve.

## Existing implementations

Shapes below are logical `[input, output]` dimensions on one TP16 rank.
Historical checks use the source hashes in their reports.

| Component | Existing operator | Target shape | Numerical evidence |
| --- | --- | --- | --- |
| W4 expert projections | `npu_dynamic_quant`, `npu_grouped_matmul` | Gate/up `[6144,256]`; down `[128,6144]`; INT32-packed weights and INT64 scale carriers | [32/32 frozen MSD projections](w4-msd.md); [189/190 operator checks](check-weights-msd-a8-20261016.json), one random M4 activation code differs |
| W8 projections | `npu_dynamic_quant`, `npu_quant_matmul` | qA `[6144,2048]`, kvA `[6144,576]`, qB `[2048,1024]`, output `[1024,6144]`, index query `[2048,4096]` | All shapes execute in the TP16 short run; [A8 operation order](a8-rounding.md) calibrated on frozen inputs |
| W8 FFN | Same quantized matmul, `npu_swiglu` | Dense gate/up `[6144,1536]`, down `[768,6144]`; shared gate/up `[6144,256]`, down `[128,6144]` | [Layer-1 same-input FFN](dense-layer1-sensitivity-20261016.json) exact; all dense/shared layers execute |
| Norm and RoPE | `npu_rms_norm`, `npu_interleave_rope`; FP32 LayerNorm | RMS H=512/2048/6144; RoPE D=64; indexer LayerNorm H=128 | [Four native RMS boundaries](rmsnorm-rounding-20261016.json) exact; [42 frozen rotary/index rows](rope-rounding.md) exact with FP32 intermediates |
| Compressed MLA | Two `npu_transpose_batchmatmul` calls and existing `_C_ascend.npu_sparse_flash_attention` binding | Four heads; latent 512, query NoPE 192, RoPE 64, value 256; 8K/top-2048; page 128 | [Two-key check](sfa-vendor-20261016.json) passes 6/6; full 8K parity pending |
| BF16 index selection | `npu_lightning_indexer` | Q `[1,32,128]`, paged K, top-2048 | [8K selection](indexer.json) passes positive/negative weights |
| C8 index selection | BF16 matmul Hadamard, `npu_dynamic_quant`, `npu_quant_lightning_indexer` | INT8 Q/K, FP16 scales and head weights | Encoder differences pending; [arch22 scoring oracle](c8-arch22/README.md) includes both FP16 intermediates; device comparison pending |
| Expert combine | `npu_moe_token_unpermute` | BF16 `[8,6144]`, INT32 row map, BF16 probabilities `[1,8]` | All 75 MoE layers execute; complete layer parity pending |
| TP communication | HCCL through `torch.distributed` | Two FP32 reductions per backbone layer: M1 24 KiB, M4 96 KiB; embedding/logits separately | TP16 short execution and exact reset complete; BF16-wire/FP32-accumulation candidate and timing pending |

The [historical expert report](weights/expert-chain.json) validates explicit
BF16 SiLU and multiplication boundaries. The native decode uses the existing
fused `npu_swiglu`; that report does not validate the fused boundary.
The custom TileLang W4 experiment has no measured advantage and is outside
the selected decode path.

## Runtime and reference contracts

[NativeDecodeEngine](../../../tilert/models/glm_5_2_ascend/native_engine.py)
loads the 78-layer backbone, shards heads/vocabulary/intermediate channels,
and replicates qA, kvA, router and indexer. Routed experts load on first use.
`prepare_experts()` makes all expert weights resident before profiling.
`step` consumes a token and returns its successor logits. `reset` clears
cache lengths. `inject` receives 78 post-norm/post-RoPE MLA prefixes and
encoded index keys, then sets the next write position to the prefix length.

The reference must use the same checkpoint, TP16 local activation quantization,
and the Ascend prefill's numerical conventions:

- Ordinary, final, qA and kvA RMSNorm use BF16 gamma and epsilon `1e-5`.
  HF 5.14.1 defaults qA/kvA to `1e-6`; the independent HF adapter must
  explicitly set them to the Ascend model's `1e-5`.
- Main MLA uses the even/odd-half output of `npu_interleave_rope`.
  The indexer restores alternating coordinates before its BF16 Hadamard.
  BF16 values and coefficients use FP32 rotary intermediates and one final
  BF16 cast; see the [source and frozen-row differential](rope-rounding.md).
- Indexer LayerNorm computes on FP32 input/gamma/beta with epsilon `1e-6`,
  then casts the result to BF16.
- Indexer head weights enter the selector as the raw BF16 projection.
  C8 converts these weights to FP16 before scoring; a common score factor
  must not be moved before that conversion. A
  [stdlib witness](indexer-head-weight-scale-witness.json) shows strict
  score order reversing when `1/64` is moved before FP16 conversion; it
  demonstrates the rounding mechanism, not a sampled checkpoint failure.
- Router logits, correction, selection and probability normalization are
  FP32. Correction affects expert choice only. Probabilities are normalized,
  multiplied by `2.5`, then converted to BF16 for native unpermute.
- Each expert's down input is dynamically quantized over its local 128
  channels. Its scale-bias uses the rank's checkpoint column.

These contracts come from the inspected
[vLLM-Ascend revision](https://github.com/vllm-project/vllm-ascend/tree/f558b58c177916424ae9dcf9bb8e66c0b9031236):
`attention/sfa_v1.py:1238–1248,1335–1372`,
`ops/rotary_embedding.py:166–169`, `attention/indexer.py:208–260`,
`patch/worker/patch_deepseek_v2.py:135–158`,
`ops/fused_moe/token_dispatcher.py:429–434`, and
`quantization/methods/w4a8/w4a8.py:613–654`.
The [RoPE formula](https://raw.githubusercontent.com/Ascend/op-plugin/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_interleave_rope.md)
and [unpermute formula](https://raw.githubusercontent.com/Ascend/op-plugin/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_token_unpermute.md)
define the two native coordinate and combine operations.

## Correctness execution

When `a3-187` is free, use `trt-asc` and the unified
environment:

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 \
  python tests/glm52/run_hf_reference.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --tokens 154820 1 2 3 --generate 4 \
  --source-commit f554fa57d0a69e7a44ac4f3134d0134b4964e848 \
  --output /workspace/experiments/glm52/hf-reference
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACL_OP_INIT_MODE=1 \
  HCCL_NPU_SOCKET_PORT_RANGE=auto \
  OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=16 \
  tests/glm52/run_native_decode.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --tokens 154820 1 2 3 --generate 4 \
  --reference /workspace/experiments/glm52/hf-reference \
  --source-commit f554fa57d0a69e7a44ac4f3134d0134b4964e848 \
  --output /workspace/experiments/glm52/native-decode
```

The [HF adapter](../../../tilert/models/glm_5_2_ascend/reference.py) constructs
the installed `GlmMoeDsaForCausalLM` on meta and retains its model, decoder,
expanded attention and `DynamicCache` forward. Independent CPU adapters read
the raw checkpoint, apply TP16 quantized projections, encode index keys and
combine experts within each rank. The [raw reader](../../../tilert/models/glm_5_2_ascend/reference_weights.py)
uses its own W4 unpacking, the [MSD FP16 partial-product boundaries](w4-msd.md)
and [A8 operation order](a8-rounding.md). W8 uses signed INT32 dots; W4 adds
the checked checkpoint correction after FP32 `16H+L`. Its active-layer cache
is limited to 256 MiB plus eight experts; embedding/head shards are read on
demand.

The [installed HF source inventory](hf-installed-source.json) records version
5.14.1 and the inspected files. [Raw-byte checks](hf-reference-weights-metadata.json)
cover all 256 nibble encodings and 144 correction groups from 24 real weight
rows. [Metadata checks](hf-reference-metadata.json) exercise trace counts,
prefix shapes/dtypes and manifest acceptance. These checks do not execute
Torch or establish numerical parity.

The CPU reference chooses FP32 SwiGLU/product with one BF16 cast, FP32 expert
weighted sums with BF16 probabilities/output, and FP32 rank-order TP sums
with one BF16 cast. The recorded CANN MSD and A8 boundaries are calibrated;
the HCCL reduction tree and full C8 encoder retain separate device checks.
The encoder uses the source-derived A8 formula and keeps exact cache checks.
The [arch22 scoring oracle](c8-arch22/README.md) now includes
FP16 rounding of `weight * query_scale` and `ReLU(INT32(QK) / 1024)`, with
key scale applied after FP32 head reduction. Its source-bound, same-wire
top-k comparison remains exact. The historical 419 key and one query
encoding differences have not been revalidated.

The runner captures all 78 layer outputs and their final-norm/head logits,
routes, context selections and newly appended cache rows. It resets and
replays the same consumed tokens exactly. Without `--reference`, its status
is `needs_reference`, not numerical acceptance. No timing or profiling is
performed. The [short native run](native-full-short-vendor-20261016.json)
and [corrected HF reference](hf-full-short-msd-a8-20261016.json) have executed;
their [independent-rollout comparison](recompare-short-msd-a8-20261016.json)
still fails. The 8K case remains the target numerical gate.

`--reference DIR` reads `manifest.json` identifying
`implementation=transformers.GlmMoeDsaForCausalLM`, `tp_size=16`, matching
`metadata_sha256`, and the independent generator's `source_sha256`.
Generation and exact reset replay must have `status=pass`; installed HF
sources and every trace file have hashes. The manifest must also declare
completed numerical execution and the current C8 scorer hash, with verified
CANN arch22 source identities. The native runner checks trace
hashes and, when supplied, each rank's prefix hash and initial position.
It explicitly records `rms_norm_eps={ordinary: 1e-5, q_a: 1e-5, kv_a: 1e-5}`
and `index_rope_basis=interleaved`. Reference input tokens drive every rank,
so the comparisons remain teacher-forced after a greedy mismatch.
Each `step-NNNNN.pt` has input token, absolute position, full logits,
79 hidden states, 78 layer logits, 78 selection/route entries, and
78 appended-cache dictionaries (`latent`, `rope`, `index`, `index_scale`).
Floating comparisons use NRMSE ≤0.01 and cosine ≥0.999; all logits also
require max-abs ≤0.1 and equal greedy tokens. Selection sets, integer cache
rows, FP16 index scales, and reset replay remain exact.

`--prefix DIR` loads `rank-N.pt` containing matching `metadata_sha256`,
the original `prompt_token_ids` and 78 `layers` dictionaries accepted by
`inject`. The HF generator checks token count against the prefix rows and
records the prompt identity for [request measurement](measurement.md).
It permits an 8K prefill
prefix followed by short teacher-forced decode. A vLLM-produced first token
is consumed at position 8192 when the injected prompt has 8192 rows. Pass
the same directory to both commands. The HF runner checks replicated prefix
contents across all sixteen files and independently expands latent KV through
kvB. This validates decode from a shared frozen prefill state; it does not
validate prefill computation. Reset creates a fresh HF cache and clears C8
scales before reinjection. The [capture entry](measurement.md#capture-original-vllm-prefixes)
saves original HIXL receive rows without decode replay or model weights.
Live transfer/export and 8K numerical acceptance remain pending.

[Runner metadata checks](native-runner-metadata.json) pass 23/23 cases for
position, cache lengths, indexer presence and collection completeness.
[Source checks](native-source-checks.json) cover syntax and style in eleven
files. Both reports use server CPU execution and import no Torch; neither
is a numerical decode result.
