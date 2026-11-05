# W4A8 NZ carrier descriptor mismatch

The failed original decoder GMM1 submitted an INT4 NZ weight descriptor whose physical span was four times its allocation. Its retained tiling used `N=1024`; the host allocated output for `N=256`. The storage-preserving correction completes the same original GMM1 and the complete two-layer M1 comparison on a3-194, Torch/TorchNPU 2.12 and CANN 9.2. The original asynchronous path also passes M1. The wider run reaches an independent M12 attention precision difference described below.

The [original synchronized diagnostic](../vendor-hook-decoder212-diagnostic-int8-m1-v1-a3-194/decoder-diagnostic.json) completed the producer fence, recorded 256 valid expert counts summing to eight, submitted the unchanged official GMM1, and failed at its completion fence before `dispose_tensor`. The original weight remained alive. Those observations remove out-of-range routing counts and the later disposal as necessary explanations for this failure.

| Boundary | Recorded value |
| --- | --- |
| Torch carrier | INT32 `[256,6144,32]`, strides `[196608,32,1]`, offset zero |
| Original allocation | 201,326,592 bytes |
| Owner descriptor | INT8 NZ units; base shape `[256,6144,128]` |
| Actual device weight shape | INT4 NZ `[256,4,384,16,256]` |
| Actual tiling | `groupNum=256`, `coreNum=24`, `K=6144`, `N=1024`, `M=8`, `groupListType=1` |
| Host output | INT8 `[8,128]`, 1,024 bytes |

The formal [`DumpData` decoder](decode_exception.py) reads the 4,005-byte protobuf header, extracts tensor `Shape` and `arg_index`, and locates the 352-byte tiling payload at byte 841,677. All input, output, and workspace lengths sum to the exact 84,712,781-byte file. [Decoded metadata](decoded.json) preserves the complete tiling bytes and leaves its last eight bytes uninterpreted. The first 120 bytes use the common `GMMSwigluQuantV2BaseParams` source prefix; `N` is the fourth INT64. The embedded `TCubeTiling` independently contains `N=1024`. Its field list is from the [installed SDK header](source/installed-matmul_tilingdata.h).

`DumpFile` deliberately writes `DT_UNDEFINED` into the protobuf. INT4/NZ identity comes from the retained [kernel JSON](kernel.json), not that default enum. The zero-byte weight dump is not evidence of a zero-byte allocation: its shape and the independently recorded allocation remain available.

The actual frozen [Torch adapter](source/frozen-op_api_common.h) creates the ACL view from `at_tensor.sizes()`, strides, and INT32 dtype at lines 445–450. For internal formats it copies `npu_desc_.storage_sizes_` without changing their units at lines 414–417. It does not substitute `base_sizes_` for the logical view. Therefore the logical carrier width 32 is correct; the stale physical descriptor is the mismatch.

The legitimate INT32 carrier shape is `[256,4,384,16,8]` in INT32 units. Expansion to INT4 gives `[256,4,384,16,64]`, whose byte span is `256*4*384*16*64/2 = 201326592`. Carrying the original INT8 physical last axis 32 into the INT32 descriptor instead yields the observed INT4 last axis 256 and `805306368` bytes. The [current op-api source](source/current-gmm_dsq_base.h), lines 619–647, explicitly multiplies the final storage axis by eight. Its host tiler, lines 258–264, obtains `N` from the NZ physical axes 1 and 4. The pinned vLLM-Ascend host tiler uses the same formula at [lines 214–220](source/pinned-grouped_matmul_swiglu_quant_v2_base_tiling.cpp). This explains `4*256=1024` without assuming a corrupted logical view.

The installed SDK does not ship that op-api implementation or host tiler. Their repository revisions and file hashes are recorded in the [manifest](manifest.json); they are explanatory source, not asserted binary identities. The formal failure descriptor, complete tiling payload, frozen host adapter, and installed kernel identity establish the actual boundary values independently. The pinned tiling header additionally has a later `swigluLimit` field absent from the decoded common layout; the parser does not apply that entire pinned struct to the installed payload.

The isolated correction creates an independent `StorageImpl` with the same data pointer and canonical INT32 descriptor. A deleter closure retains the original Tensor, and the original owner descriptor is unchanged. GU and Down use the exact supported TP16 shapes; their existing allocations are 192 MiB and 96 MiB per layer. Two layers retain 576 MiB total, with zero additional HBM for the aliases. The device fixture hashes every physical byte before and after, checks descriptor identity, owner reference retention and canonical idempotency, then enters the unchanged synchronized original-forward diagnostic. No model arithmetic or official operator body is modified.

The [alias build](../decoder-nz-alias-build-v1/build/manifest.json) uses the actual Torch/TorchNPU 2.12 headers and ABI. The [corrected diagnostic](../vendor-hook-decoder212-storage-int8-m1-v1-a3-194/nz-storage-correction.json) verifies all four GU/Down banks: identical full raw bytes and pointers, independent corrected descriptors, retained original owners, and no new HBM allocation. All four original GMM1 calls complete before their unchanged disposal. The [asynchronous M1 gate](../vendor-hook-decoder212-storage-forward-int8-m1-v2-a3-194/decoder-forward.json) removes the GMM observation hooks and internal diagnostic fences; its 25 byte checks pass. Both arms still execute the original FFN, with explicit local TP cut inputs.

The [wider asynchronous gate](../vendor-hook-decoder212-storage-forward-int8-wide-v2-a3-194/decoder-forward.json) passes M1/M2 for eight inputs each, then records one index-cache code difference and five ordered top-k entries at M12 iteration 0. It has no GMM address failure. The [coordinate audit](../decoder-index-m12-audit-v1/README.md) and separate front-end diagnostic track that numerical boundary; the descriptor correction does not establish acceptance of those attention outputs.

The large exception dump, tensor snapshot and ELF files remain under `/tmp/tilert-megakernel-validation-20261021/vendor-hook-decoder212-diagnostic-int8-m1-v1-raw` and the remote job directory. Their identities are retained in the manifest and the [original raw-artifact list](../vendor-hook-decoder212-diagnostic-int8-m1-v1-a3-194/raw-artifacts.json).
