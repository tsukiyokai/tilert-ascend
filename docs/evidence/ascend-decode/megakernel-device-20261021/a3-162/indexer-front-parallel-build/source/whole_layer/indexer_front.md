# Native GLM-5.2 indexer front

`indexer_front_native.h` implements Q/WK projections, key LayerNorm,
interleaved RoPE, and the optional C8 Hadamard/quantization path in one mixed
AIC:AIV 1:2 launch. It starts with the main Prolog's normalized INT8 Q and
FP32 token scales. It returns the inputs consumed by `indexer_native.h`;
cache writes, top-k selection, SFA, TP reduction, and the main Prolog are
outside this entry. The indexer's 32 heads are replicated under TP16.

The CANN 9.2 `dav-2201` library
`indexer_front_build-scale-order/whole_layer_indexer_front.so`, SHA-256
`477d20ecd6cd2ec5e9e9cec20546801615d5fc89a4be7259818a54f23096b62c`, passed
M=1/2/12/32 captured tests in C8 and BF16 modes, with 32 changing queued
inputs per case and each original WQ scale dtype: [FP32](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-fp32-scale-order.json)
and [BF16](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-bf16-scale-order.json).
All retained production boundaries matched the reference bytes across these
512 replays. FP32 norm statistics remain diagnostics. The build manifest
retains the source snapshot, compiler, SDK identity, and library hash.

## Tensor contract

All tensors use contiguous physical ND storage on one NPU. `M` is 1–32.

| Input | Dtype and shape | Meaning |
| --- | --- | --- |
| `hidden` | BF16 `[M,6144]` | Input of replicated WK and head-weight projection |
| `query_norm` | INT8 `[M,2048]` | Main Prolog normalized Q codes |
| `query_norm_scale` | FP32 `[M]` | Main Prolog normalized Q token scales |
| `query_weight` | INT8 `[16,2048,256]` | Logical WQ `[2048,4096]`, split into 256-column tiles |
| `query_weight_scale` | Original FP32 or BF16 `[4096]` | WQ channel scale descriptor; Q projection has no bias |
| `key_weight` | BF16 `[5,6144,32]` | Logical merged WK/weights `[6144,160]` |
| `gamma`, `beta` | FP32 `[128]` | Key LayerNorm affine parameters |
| `cosine`, `sine` | BF16 `[M,64]` | First 32 values select adjacent-pair RoPE64 |
| `hadamard` | BF16 `[128,128]` | Normalized Sylvester matrix, used by C8 |

The C8 outputs are INT8 Q `[M,32,128]`, INT8 K `[M,128]`, FP16 head weights
`[M,32]`, FP16 Q scales `[M,32]`, and FP16 K scales `[M]`. The BF16 branch
returns BF16 Q, K, and head weights, with empty scale tensors.

Each projection rounds to BF16 before its next operation. Key normalization
uses FP32 mean and variance, epsilon `1e-6`, FP32 gamma/beta, and a BF16 output.
RoPE rotates adjacent pairs in the first 64 coordinates and rounds to BF16.
C8 multiplies those BF16 values by the BF16 Hadamard matrix, rounds to BF16,
then applies dynamic INT8 quantization and casts FP32 scales to FP16.

For WQ K=2048, N=4096, the Q epilogue evaluates
`FP32(accumulator) * RN32(token * channel)`, then rounds to BF16. Both original
FP32 and BF16 channel-scale descriptors use this order; the BF16 values are
first widened exactly to FP32. The dtype flag controls loading only.
At M=12/32 the internal FP32 norm statistics in the original tests differ
by up to `2.384185791015625e-7`, while the BF16 normalized key and all later
key outputs match bytes.

The stock reference receives the original scale dtype. The
[original-BF16 probe](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-bf16-scales.json)
passed M=1/2 but failed at M=12: row 5, channel 3323 had one differing BF16
projection, which propagated through Hadamard to one differing INT8 code.
The [CPU reconstruction](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-bf16-scale-witness.json)
matches all 49,152 integer accumulators to device scratch. At the failing
coordinate, `acc=13330`, channel FP32 bits are `0x3b530000`, and token-scale
bits are `0x3c70d5ca`. Channel-first multiplication yields FP32 `0x3f218000`
and BF16 `0x3f22`; multiplying the two scales first yields FP32 `0x3f217fff`
and BF16 `0x3f21`, matching stock. The combined-scale expression matches all
49,152 stock BF16 projection values in this saved case.

The [FP32-scale reconstruction](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-fp32-scale-witness.json)
independently matches all 49,152 integer accumulators for M=12, replay 4.
At row 6, channel 34, `acc=2284`, channel bits are `0x3b439cee`, and token
bits are `0x3d100b78`. Channel-first multiplication yields FP32 `0x3e757fff`
and BF16 `0x3e75`; combined scales yield FP32 `0x3e758000` and BF16
`0x3e76`, matching stock. Combined scales match every stock projection in
this saved case. The immutable original library, SHA-256
`4a24dabb187b204fd08e27880b9acf75aa6be4f6be3f3abfcbd76196f42a0fc1`,
reproduces the same failure in the [extended control](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-fp32-extended-control.json).
Its earlier four-replay tests did not reach this distinguishing input.

The binding preserves original FP32/BF16 scale tensors without changing the
Torch schema. Scratch and tiling offsets are unchanged. The
`indexer_front_build-scale-order` build uses the same accepted expression in
the [indexed composition](attention_indexed.md). This expression is scoped to
the installed CANN 9.2 WQ K=2048/N=4096 path; it does not prescribe the MLA
output projection's order.

The head-weight path requires a vector barrier between its FP32→BF16
projection Cast and the BF16→FP32→FP16 Cast. Publishing the projection to
MTE3 alone does not order the following vector read. The initial library
returned 32 zero weights at M=1 despite preserving the correct projection;
adding only this barrier made that same case and the replay cases pass.
Raw reports retain the [initial failure](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/indexer-front-m1-v2.json),
[M=1 differential](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/indexer-front-m1-v3.json),
and [captured replay differential](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/indexer-front-replay-v3.json).

## Composition and storage

`indexer_front_prepare(hidden, quantized, blocks=0)` allocates scratch, three
Cube tilings, and five outputs. It returns those tensors plus the launch
group count and sync address. `blocks=0` uses the physical AIC group count.
`indexer_front_out` uses preallocated storage and the current stream with
Torch's host task queue flushed before the direct device launch.

Every resident AIC and AIV reaches each phase barrier. Projection, vector
epilogue, Hadamard, and quantization phases publish GM writes, use mixed
`SyncAll<false>`, and reset the pipe after the preceding objects have left
scope. The BF16 branch uses the first two phases. Per-row DMA reuse waits
for MTE3 before MTE2 overwrites its UB source.

Scratch retains projection, normalization, RoPE, and Hadamard intermediates
for differential diagnosis. Regions are 512-byte aligned. The final 16 MiB
is the Matmul system workspace. Total scratch is 17,092,608 bytes at M=1,
17,461,760 bytes at M=12, and 18,414,080 bytes at M=32. These diagnostic
buffers have not been overlaid or included in a full-layer workspace plan.

## Build and compare

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
bash tests/glm52/whole_layer/indexer_front_build.sh
python3 -m unittest discover -s tests/glm52/whole_layer \
  -p indexer_front_test.py -v
python3 tests/glm52/whole_layer/indexer_front_check.py \
  --rope-source indexer_front_reference/rope.py \
  --widths 1 --modes int8 bf16 --no-timing \
  --output logs/indexer-front-m1.json
# Repeat for float32 and bfloat16 original WQ channel scales.
python3 tests/glm52/whole_layer/indexer_front_check.py \
  --rope-source indexer_front_reference/rope.py \
  --widths 1 2 12 32 --modes int8 bf16 --replays 32 \
  --capture --queued-producer --query-scale-dtype bfloat16 --no-timing \
  --output logs/indexer-front-scale-order-bf16.json
```

`CANN_ROOT`/`PYTHON_BIN` take precedence over
`ASCEND_HOME_PATH`/`PYTHON`. Build dependencies are the CANN Matmul API,
Torch/torch_npu headers, and the existing MLA native helper; no host CANN
operator is called from the native entry.

The reference `rope.py` is the original vLLM-Ascend file. The harness extracts
its unmodified `_triton_rope_siso` function and records both source hashes.
The other reference stages are stock QMM, `F.linear`, FP32
`native_layer_norm`, BF16 matmul, and dynamic quantization. It checks raw
bytes at every retained production boundary. FP32 norm statistics are
reported as diagnostics; the BF16 normalized key is the acceptance boundary.
On failure, the `.pt` file retains all inputs and weights, scratch, reference
tensors, and observed tensors. `--capture --queued-producer` adds matching
Torch producers to both arms and checks changing data during graph replay.
Any optional timings cover this diagnostic front entry, including retained
intermediate writes, rather than a complete indexer or layer.
`--query-scale-dtype bfloat16` passes original BF16 channel scales to both
implementations. `--convert-query-scale-to-fp32` reproduces the old adapter's
native-only conversion for diagnosis. Reports retain both actual scale
dtypes and the harness hash; failure dumps retain both tensors.

## Source contract

The source replay uses vLLM-Ascend commit
`8237ee6d690d1399bc6eb48ef5977ac2bd470f3f`:

- `vllm_ascend/attention/indexer.py:244–267`: BF16 Hadamard, dynamic quantization,
  and scale conversion.
- `vllm_ascend/attention/indexer.py:327–355`: merged WK/weights, key LayerNorm,
  RoPE, and optional key quantization.
- `vllm_ascend/attention/indexer.py:454–483`: direct reuse of the main Prolog's
  Q codes/scales, QMM, and RoPE.
- `vllm_ascend/ops/triton/rope.py:327–400`: adjacent-pair RoPE implementation.

vLLM commit `35a49fcfc2295d04fb252c1080f8d7bd9d888e25`,
`vllm/model_executor/layers/layernorm.py:305–321`, supplies the FP32 LayerNorm
with output cast to the original BF16 dtype. The build manifest also hashes
the installed CANN 9.2 `ops_legacy/dynamic/layer_norm_v3.py` reference.

The installed CANN 9.2
`quant_batch_matmul_v3/quant_batch_matmul_v3_pertoken_opt.h`, SHA-256
`1885876467fc58fabd5aba043363cefa2aaa96cc25eb7e36fa57b00ca675cc5f`,
widens BF16 scales at lines 524–531, multiplies channel and token scales at
line 494, converts the INT32 result to FP32 at line 575, and multiplies it
by the combined scale at line 580 before the BF16 cast at line 586. The saved
device witnesses above distinguish this expression from channel-first
dequantization for both original scale dtypes. The installed host tiler
source was unavailable; this contract does not infer other QMM shapes from
the existence of that header.
