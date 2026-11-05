# TP16 shared W8A8 expert

This standalone A3 MIX entry consumes INT8 `[M,6144]` activations and FP32
per-token scales, with `1 <= M <= 32`. It computes the local shared expert
with intermediate width 128, ordinary SwiGLU, no bias and no output gate.
Input quantization, the routed branch and TP reduction are external.

The gate/up projection splits K into six 1024-element pieces and N into
four 64-column pieces. Twenty-four CUBE tasks produce INT32 partials.
Their sum is exact: even the largest possible signed INT8 dot product has
magnitude at most `6144 * 128 * 128 = 100663296`, below the INT32 limit.
AIVs sum the partials before channel scaling and token scaling, then compute
SwiGLU in FP32. The activation is dynamically quantized before the down
projection, which has 24 output tiles of 256 columns. The output epilogue
uses the batched [MLA helper](../mla_tail/mla_tail_native.h), explicitly
combining channel and token scales in FP32 before multiplying the accumulator,
and returns BF16 `[M,6144]`. This is the CANN 9.2 small-K projection's order;
the MLA tail's K=1024 projection retains channel-then-token multiplication.

The checker uses the installed stock CANN chain:
`QuantMatmul(INT32) -> DequantSwigluQuant(mode=0, dynamic) -> QuantMatmul(BF16)`.
This mode's scale is scalar `max(abs(x)) / 127`, followed by a scalar reciprocal
and vector multiplication. Quantization casts FP32 to INT16, FP16 and INT8.
The grouped vector-division implementation has a different rounding contract.
The vLLM custom mode-1 implementation uses vector multiplication by FP32
`1/127`, vector division by that scale, then INT32/FP16/INT8 casts. Build
the `grouped` contract for that arithmetic. Its ordinary-SwiGLU configuration
also differs from the stock SDK's interleaved mode 1.

Weights use dense packed tiles: gate/up `[6,4,1024,64]` and down `[24,128,256]`.
The two arrays hold 2.25 MiB per TP shard. Input staging, INT32 partials,
quantized activations, scales and down products have separate 512-byte-aligned
regions. Four mixed-core barriers order the stages, with explicit pipeline
events for every reused vector buffer. Capacity is padded to 16 or 32 rows;
producers initialize all rows read by CUBE.

The reusable stages live in [shared_w8_native.h](shared_w8_native.h).
Callers supply the stage barriers and keep each buffer alive until its last
consumer. A composition can share a Matmul system workspace when only one
CUBE projection is active at a time; the standalone entry reserves its own.

```bash
CANN_ROOT=/path/to/cann bash tests/glm52/shared_w8/build.sh
python tests/glm52/shared_w8/check_shared.py \
  --widths 1 2 12 32 --blocks 24 --queued-producer --replays 4 \
  --no-timing --output /tmp/shared-w8.json
```

The build retains source snapshots and hashes beside the library. The checker
compares the INT32 gate/up sum, INT8 activation, FP32 scale and final BF16 output
as raw bytes. Inputs include zeros, sign changes and alternating INT8 extremes.
Both captured arms contain identical queued clones when `--queued-producer`
is enabled. Timings use paired device events after these comparisons pass.

On Ascend910_9392, CANN 9.2.0, the scalar-quant variant
`f3027866ab33efd9a9f92379897ec3c74214148e00675a184f9a705000384d13`
passes all four boundaries at M=1,2,12,32 over eight changing graph replays.
Six alternating timing pairs give native/vendor medians of 38.11/36.31 us
at M=1, 40.89/38.98 us at M=12, and 43.06/40.65 us at M=32. Standalone
fusion is slower in these cases. See the [raw device evidence](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/README.md).

The `grouped` variant
`3c6bd9dd66c0d84bf9ff7ffb32aa2094109ba96f75ceac739036eec92379c04a`
passes the same four widths and eight changing replays against the unmodified
vLLM-Ascend SwiGLU kernel from commit
`8237ee6d690d1399bc6eb48ef5977ac2bd470f3f`, between two stock CANN projections.
The reference wrapper sets the fixed shape and tiling explicitly; it does
not validate the deployed custom operator's host dispatch or engine path.
The source kernel SHA256 is
`2e516dc7332449057ea8379eebd61a0b6f2a2917907b7f88d07b6025d3cf72b2`.

```bash
CANN_ROOT=/path/to/cann bash tests/glm52/shared_w8/build.sh \
  tests/glm52/shared_w8/build-grouped grouped
CANN_ROOT=/path/to/cann bash tests/glm52/shared_w8/build_swiglu_reference.sh \
  /path/to/vllm-ascend/csrc/moe/dequant_swiglu_quant
python tests/glm52/shared_w8/check_shared.py \
  --library tests/glm52/shared_w8/build-grouped/shared_w8.so \
  --reference-source-library tests/glm52/shared_w8/build-swiglu-reference/shared_swiglu_reference.so \
  --widths 1 2 12 32 --blocks 24 --queued-producer --replays 8 \
  --no-timing --output /tmp/shared-w8-grouped.json
```
