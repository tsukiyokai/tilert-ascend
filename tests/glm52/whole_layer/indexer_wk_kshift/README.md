# M32 indexer WK accumulation order

This candidate replaces only the BF16 WK projection at M32 in the frozen
O-combined indexed-attention kernel. M1 through M31 retain the original
projection. WQ, LayerNorm, RoPE, quantization, selection, main attention and
the O epilogue retain their parent implementations. The shared-index entry
and Python schemas are unchanged.

The [CANN 9.2 build](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/attention-wk-kshift-build-v1/manifest.json)
produces library SHA-256
`20095457af00d9299fcff884d806b03400669adfe3304750ca9ae97983a40ca8`.
Torch 2.10 loads it with the unchanged shared library and verifies all four
schemas without initializing an NPU. The archive retains all 58 frozen
source files, the one-call patch, object identities and build commands.

The [captured-projection intervention](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-kshift-v1-a3-194/summary.json)
matches all 5,120 production BF16 outputs in three repetitions using the
recorded MatMulV2 key-230097 K256 order. Natural K order exactly reproduces
all original FP32 accumulators and the four differing BF16 elements. Both
arms retain one FP32 L0C tile across 48 K128 Mmad operations.

The integrated helper retains packed weights `[5,6144,32]` and FP32 results
`[5,capacity,32]`. Each of twenty M16/N16 tasks transposes eight K16/N16
fractals into L0B, applies the recorded K order and writes its columns with
row stride 32. Physical AIC groups own tasks cyclically; `FIX_M` protects
L0C reuse between tasks. All mixed cores still reach the parent's phase
barriers. The stage uses 8 KiB L1, 4 KiB each L0A/L0B and 1 KiB L0C, with
no additional GM workspace.

`build.py` verifies the parent library and every frozen source, adds the
helper and replaces one WK call. It recompiles only the indexed kernel and
reuses the original Torch 2.10 binding and tiling objects. Its CPU link check
loads the candidate with the unchanged shared-attention library and checks
all four schemas without NPU initialization.

```bash
python3 tests/glm52/whole_layer/indexer_wk_kshift/test_contract.py -v
```

The CPU contract checks every packed-weight element, every output address
for 1–24 physical groups and all ten K traversal sequences.

The actual M32 outer-call gates pass eight changing inputs in both
[BF16](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-wk-kshift-v14-bf16-m32-a3-194/summary.json)
and [C8](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-wk-kshift-v14-int8-m32-a3-194/summary.json).
All 88 output, ordered-top-k and cache comparisons match raw bytes, including
the previously failing sixth input. The last input also passes every
diagnostic indexer boundary: 19 in BF16 and 23 in C8.

The official graph-wrapper gates pass M1/M2/M12/M32 in
[BF16](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-wk-kshift-v15-bf16-all-graph-a3-194/summary.json)
and [C8](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-wk-kshift-v15-int8-all-graph-a3-194/summary.json):
16 captures, 128 graph replays and 704 raw-byte comparisons against the
original eager outer call, with no Python body calls during replay. These
use synthetic weights for one TP16 local shard and exclude TP collectives
and whole-model latency.
