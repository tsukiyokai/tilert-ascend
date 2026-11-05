# Shared W8 output rounding counterexample

The captured M=12 failure has identical INT8 middle activations and FP32 token scales. All 73728 native down INT32 accumulators equal an independent INT64 matrix product. The final BF16 outputs differ only at `[11,5240]`.

At that element, `acc=-15243`, `channel=0.0005920213297940791`, and `token=0.6042792201042175`. FP32 `(acc*channel)*token` produces `-5.453125476837158` and rounds to native BF16 `-5.46875`. FP32 `(channel*token)*acc` produces the exact midpoint `-5.453125` and rounds to vendor BF16 `-5.4375`. The two output values differ by `0.03125`.

Across the complete tensor, the channel-first expression matches every native output byte and the scale-first expression matches every vendor output byte. The installed CANN 9.2 `quant_batch_matmul_v3_pertoken_opt.h` implements scale-first multiplication at lines 494/514/530 and accumulator cast/multiply/BF16 rounding at 575/580/586; SHA256 is `1885876467fc58fabd5aba043363cefa2aaa96cc25eb7e36fa57b00ca675cc5f`. The separate `pertoken.h` implementation uses channel-first multiplication. This failure establishes the required output order for the captured shape and runtime; it does not identify all future runtime tiling choices.

[Raw audit](shared-w8-down-order-audit.json) includes all six multiplication permutations, the independent accumulator comparison, input hash and IEEE bit patterns. [Captured tensors](shared-w8-down-diagnostic-m12-r0.pt) preserve both device outputs.
