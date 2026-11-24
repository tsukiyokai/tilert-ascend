# QMM output scale order

CANN 9.2.0 ships more than one per-token output epilogue. The scale dtype alone does not determine multiplication order. These files were read from the installed SDK on a3-78; [manifest.json](manifest.json) records their original paths and SHA-256 identities. [line-excerpts.txt](line-excerpts.txt) retains the relevant numbered source lines. No v10 runtime tiling key has been captured.

| Installed source | Arithmetic before the BF16 output cast |
| --- | --- |
| `quant_batch_matmul_v3_pertoken_opt.h:488–531,575–586` | Broadcast token scale, multiply channel scale, multiply the FP32 cast of the INT32 accumulator, then BF16 `CAST_RINT`. BF16 channel scales are first cast to FP32; FP32 scales use the same multiplication order. |
| `quant_batch_matmul_v3_pertoken.h:491–498,518–536` | `AscendDequant` applies the channel scale to the accumulator before the broadcast token scale is multiplied into that result. |
| `quant_batch_matmul_v3_basic_epilogue.h:196–238,126–138,242–248` | The basic epilogue likewise applies channel dequantization before the token scale. |

The installed `quant_batch_matmul_v3.cpp:501–529` selects the regular and optimized templates using `KERNEL_TEMPLATE_TYPE`, with per-token mode enabled. Both templates support BF16 output. `quant_batch_matmul_v3_tiling_key.h:26–29` assigns template values 0/1/2 to TBE/basic/optimized.

The archived host tiling files are from local ops-nn commit `92076b30bf480c98e6d3dedbeec46284f4a6152e`. Five related device headers in that checkout exactly match the installed SDK; this does not establish the identity of the installed host tiling library. `ProcessMSmall` considers decode geometry, workspace size, kernel support and MC2 mode. `IsKCNetDecode` additionally checks per-token/per-channel INT8, batch 1, M ≤ 256, no bias, transpose/layout, K/N alignment and per-core M. Basic tiling has separate NZ decode and large-matrix conditions. The archived host source therefore explains concrete mechanisms for different template choices, without identifying v10's selected template.

The v10 actual-call descriptor is INT8 `[12,1024]` × NZ INT8 `[1024,6144]`, BF16 channel scale `[6144]`, FP32 token scale `[12]`, no bias or offset, and BF16 output. Its captured output matches combined scales; all preceding production boundaries and the INT32 product match the native path. That observation establishes the required arithmetic for this captured call. It does not establish a universal BF16-scale rule or a TorchNPU-version cause.
