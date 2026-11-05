# C8 index scoring on CANN 9.2.0 arch22

The installed QuantLightningIndexer rounds two intermediate values to FP16.
The previous ideal FP32 dequantization formula omits both boundaries and
cannot define exact top-k acceptance for this kernel. This changes the CPU
oracle specification; the historical 419 key and one query encoding
differences in [indexer.json](../indexer.json) remain unverified.

## Source arithmetic

For INT8 query/key rows, FP16 wire scales and FP16 head weights:

```text
D[h,k] = INT32 dot(Q[h,:], K[k,:])
W[h]   = FP16(weight[h] * query_scale[h])
S[h,k] = FP16(ReLU(FP32(D[h,k]) / 1024))
score[k] = FP32_MATMUL(W, S)[k] * FP32(key_scale[k])
```

The [cube source](quant_lightning_indexer_service_cube_arch22.h), lines
498–551, specifies INT32 QK, DEQF16/ReLU with `0x3a800000` (1/1024), and
FP32 head reduction. The [vector source](quant_lightning_indexer_service_vector_arch22.h),
lines 291–318 and 380–382, specifies the FP16 weight-scale product and final
key scaling. Both files came from the stopped `a3-gs:trt-asc` container;
[installed-source.json](installed-source.json) records their paths/hashes.
The [upstream golden](upstream-golden.py), lines 265–289, agrees; its distinct
ops-transformer commit and hash are in [source-audit.json](source-audit.json).
Scores retain the common 1/1024 factor; restoring it cannot restore precision.

## Strict cutoff witnesses

Each witness has INT8 Q `[32,128]`, K `[8192,128]`, FP16 scales/weights,
2047 higher-scoring keys, two competing keys and 6143 zero-scoring keys.
Both orderings have a strict cutoff, so tie handling cannot explain the
different selected key. Full sparse inputs, intermediate FP16 bit patterns,
FP32 scores and selected-set hashes are in
[scoring-stdlib.json](scoring-stdlib.json).

| Rounding boundary | Ideal A / B | arch22 A / B | Ideal / arch22 choice |
| --- | --- | --- | --- |
| QK / 1024 to FP16 | 2048.5 / 2049 | 2.00048828125 / 2 | B / A |
| weight × query scale to FP16 | 2080.125 / 2080.0625 | 2.03125 / 2.03131103515625 | A / B |

The first witness uses dots A=(2048,1), B=(2049,0), weights=(1,0.5) and unit
scales. The second uses A=(2048,0,0), B=(0,2080,1), weights=(129/128,1,1/16),
query scales=(129/128,1,1), and unit key scales. Remaining weights are zero.
All active operands/products are normal finite FP16 values.

For head dimension 128, the absolute INT8 dot is at most 2,097,152.
After division by 1024, the largest magnitude is 2048 and the smallest
nonzero positive value is 2^-10. Thus this QK conversion cannot produce
FP16 overflow or subnormal values. The weight-scale product can. The HF
oracle rejects nonfinite values, negative scales, subnormal operands, and
nonzero products outside the normal FP16 range. The operator checker marks
those inputs unaccepted. No saturation or flush behavior is inferred from
IEEE arithmetic. FP32 Cube reduction order still requires device comparison.

## Implemented acceptance rules

- The ideal scorer remains available. Its exact-set comparisons are recorded
  as diagnostics and are excluded from CANN acceptance.
- `check_indexer.py` requires exact top-k sets from both the independent CPU
  encoder plus arch22 scorer and the actual NPU wire inputs plus that scorer.
  Equal scores still prefer lower indices. Encoder and cache checks remain
  exact, and every scoring input must pass the FP16 range check.
- HF reference construction verifies the installed CANN source hashes.
  HF/MTP manifests include the scorer source hash. Native validators reject
  missing/stale scorer hashes and manifests without completed numerical
  execution. MTP layer 78 uses BF16 index keys; its scorer hash records the
  shared reference-module dependency.
- `check_a8_shapes.py` includes 8193-row keys. It preserves separate records
  for BF16 rotation and quantization of the same actual NPU-rotated rows.
  The A8 reference formula is unchanged.

## Executed and pending

On the a3-gs host, standard-library checks passed: six scoring/source tests,
[42 HF metadata checks](hf-reference-metadata.json), and
[25 MTP metadata/source checks](mtp-contract-metadata.json). The eleven
changed Python files parse as Python 3.11 and satisfy the 79-column check.
The exact commands and source hashes are recorded in these reports.

No Torch operation, container startup or NPU call was executed. The new
Torch scorer and all device comparisons remain unexecuted. When the shared
host is free, run the A8 shape probe to separate BF16 rotation from
same-input quantization, then the captured-wire QLI checker, then complete
HF/native layer and logits comparisons. Exact encoding/top-k, existing
floating thresholds, and 100% greedy agreement on the eight 8K inputs remain
required. These CPU results do not change the historical device report or
establish a latency.
