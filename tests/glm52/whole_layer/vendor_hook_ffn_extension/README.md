# Original DecoderLayer FFN comparison

This fixture compares the complete native FFN with actual layer-6 and layer-7
post-attention norms and original MLP operators, including routed GMM1/GMM2
and the shared W8A8 chain. The candidate is frozen library `8430ef17`.
Both arms use the original postprocessed NZ weight allocations after the
host descriptor correction; the fixture verifies their pointers and raw bytes.
Routed scales are decoded from the original zero-extended FP32 bit carriers.

Inputs are explicit post-AR1 BF16 activations and retained residuals. The
single FFN TP reduction is an identity cut that retains its local input.
The original operator chain runs eagerly; the candidate replays a captured
graph with changing inputs and poisoned outputs. Seven raw-byte comparisons
cover normalized activations, retained residuals, router logits, shared and
routed outputs, the combined result and the pre-AR2 input. Parameters are
synthetic, attention does not execute, and no TP sum is performed.

The [M1 device gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-int8-m1-v2-a3-194/summary.json)
passes all fourteen checks across both weight sets on one Ascend910_9392 die,
using CANN 9.2.0 and Torch/Torch-NPU 2.12. All sixteen Gloo ranks complete;
only rank zero initializes the NPU.

The [wider gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-int8-wide-v2-a3-194/failure-summary.json)
passes layer 6 at M1/M2/M12 for eight inputs each, then M32 inputs 0 and 1.
M32 input 2 differs at two shared-output coordinates:

| Coordinate | Native | Original |
| --- | ---: | ---: |
| `[20, 2893]` | -0.06640625 | -0.06689453125 |
| `[20, 5220]` | 0.09716796875 | 0.09765625 |

The other six boundaries are byte-equal, including the routed and combined
outputs. The run stops with 188 of 189 boundary checks passing, before layer
7's wider cases. `decoder_ffn_diagnostic.py` observes the original shared
stages and the existing native payload on this retained failure. Its capture
must reproduce both saved inputs and all seven outputs in each arm.

The [shared-stage diagnostic](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-shared-m32-v1-a3-194/summary.json)
passes that sixteen-tensor reproduction gate. The first difference is in
middle scales at rows 0, 1 and 20, each one FP32 ULP lower in native output.
Quantized inputs, GU INT32 results and middle INT8 codes are byte-equal.
Both integer projections agree with exact CPU integer dots. Substituting
the original scales in the combined-scales epilogue reproduces all 196,608
original BF16 outputs; the original scale-producing arithmetic is the
remaining precision gate.
