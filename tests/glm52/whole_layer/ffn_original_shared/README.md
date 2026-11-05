# Original shared-expert activation contract

This candidate changes the frozen complete FFN `8430ef17` to the shared-expert
activation selected by CANN 9.2 on the captured original DecoderLayer call:
`DequantSwigluQuant_int32_high_performance_30013`. It uses scalar `amax / 127`,
the guarded reciprocal, vector multiplication, and the INT16/half/INT8 cast
sequence from that implementation. Gate and up INT32 values use `CAST_NONE`
when converted to FP32. The grouped-path addition of zero to up is removed.

Library `342d9653` passes the [original DecoderLayer FFN comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-original-shared-wide-v1-a3-194/summary.json)
on a3-194 physical die 4: both layer weight sets, M1/M2/M12/M32, eight changing
inputs each, and all 448 raw-byte boundaries. Native FFN graph replay matches
the original eager norm/MLP chain, including the previously failing M32 shared
output. Parameters are synthetic; attention and TP collectives are excluded.

The routed experts, output epilogue, five body joins, final global completion,
408-byte descriptor, 33 public buffers and 116,480-byte maximum UB allocation
are preserved. The candidate has its own Torch namespace and C entry point.

The original DecoderLayer norm/MLP is the numerical oracle. The earlier
complete-FFN checker uses the old shared activation as part of its reference
and cannot validate this change. CPU source-contract checks and compilation
precede single-die tests with the retained failure and changing inputs.

Evidence: [captured shared activation and actual dispatch](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-activation-scale-source-audit/README.md).
