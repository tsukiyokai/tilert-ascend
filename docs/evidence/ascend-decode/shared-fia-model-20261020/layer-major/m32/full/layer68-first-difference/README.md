# Layer 68 attention attribution contract

The full M32/SFA run passed all 32 frozen labels on 16 ranks with identical final FP32 logits. Cross-configuration internal differences are diagnostics, with no additional acceptance threshold. The reported first hidden difference is layer68/query13: inputs and latent/RoPE agree, `skip_topk=True`, and the same selected IDs differ only by swapping columns1479/1480. The layer58/query16 INT8-key difference is a separate, later query and cannot explain query13 through causal attention.

## Three distinguishable mechanisms

| Mechanism | Discriminating evidence with identical frozen Q/KV |
| --- | --- |
| Selected-ID order changes finite-precision softmax/PV reduction | Swapping the two columns changes query13 at fixed query width. |
| Query width changes SFA tiling or arithmetic | M1 versus M32 changes query13 while index order remains fixed. |
| Query preprocessing or downstream projection/MLP changes with width | The isolated attention 2×2 is identical; full-layer hidden divergence remains outside that isolated result. Q equality must be checked separately, and a later pre-O comparison can isolate V-up. |

Neither mathematical permutation invariance nor agreement of the selected set proves byte invariance of the installed SFA kernel.

## Available inputs and identities

All paths below are inside `a3-194:trt-asc`, under `/workspace/experiments/m32-full-teacher-20261020/`:

| Input | Relative path | SHA256 |
| --- | --- | --- |
| Old M1 rank report | `m1-comparison/old/device-78x32-sfa-01/rank-0.json` | `40d3d349fd70e507c7eaae7f9f0f3ba20c65afbe3d7a22c79d3c709b6eac7c2b` |
| Old M1 layer68 | `m1-comparison/old/device-78x32-sfa-01/rank-0/layer-68.pt` | `bbdf960f91fc44266c24c7fd7d17eae9ff06dcef0305d9834e3478bfdd58515e` |
| New M32 rank report | `sfa-78/rank-0.json` | `7deccb93e6aeb797e3b41344e8dcf614bcd776a8d929f4c28b0fb71fc38e4eba` |
| New M32 layer68 | `sfa-78/rank-0/layer-68.pt` | `18d1ca812d3036d54bd7d6c04f97f0dfd5bfd4585edf0ad1d2cd704d150d37aa` |

Both layer-file hashes were recomputed on CPU and match their original rank reports. Layer68 stores BF16 `input_hidden/input_residual/input_merged[32,6144]`, INT32 `indices[32,2048]`, and BF16 `cache_rows.latent[32,512]` / `rope[32,64]`. It does not store QL/QRoPE. The shared prefix file is `/workspace/offline/prefix8k-allocation-20261017/capture/case-00/main-prefix/rank-0.pt`; both reports bind SHA `e8aa3e0e190c24d171c0ae9da22b63ad7d8928541427e47adf13dbd9c9bdd7f6`. CPU mmap metadata confirms `layers[68]` contains BF16 latent `[8192,512]` and RoPE `[8192,64]`.

Reconstruction helpers already exist in `/workspace/experiments/layer-major-local-attention-20261020/probe.py` (SHA `e32769d7f379cb772caba412c0097ab53fa02af6b52257bc61fe385012a68426`): actual rank0 checkpoint slices at lines38–56, processed prolog owner at60–85, actual fused input norm and prolog at227–229, and Q/QR saving at259–261. Its CLI is at283–291; the existing command wrapper is `launch-device-01.py` in the same directory. These are reconstruction entry points; no 2×2 execution script was created.

## Minimal deferred 2×2

Reconstruct Q once with the actual input norm on saved `(input_hidden,input_residual)` and the current production prolog; freeze BF16 QL `[32,4,512]` and QR `[32,4,64]`. The saved `input_merged` is a residual sum, not the normalized attention input. Restore KV from the 8K prefix plus saved 32 new rows, zero-initialize capacity9344, and verify the recomputed cache rows against the saved rows before using the reconstruction.

Use one immutable KV and Q throughout four eager attention calls: `(M1, original order)`, `(M1, swapped order)`, `(M32, original order)`, `(M32, swapped order)`. Swap only query13 columns1479/1480. For M1 take Q/QR/indices row13, cumulative Q length `[1]`, KV length `[8206]`; for M32 use all32 rows, cumulative Q length `[32]`, KV length `[8224]`. Compare output row0 versus row13, keeping page table, BF16 cache, scale `1/16`, `sparse_block_size=1`, `sparse_mode=3`, `TND/PA_BSND`, and `attention_mode=2`. This preserves query13's causal position8205. Record the four raw outputs and actual launch keys. Four calls can separate order and width effects for this fixed Q/KV; they cannot reproduce full-layer hidden by themselves or identify the old native binary.

The direct production call is `BaseDeviceAdaptor.execute_sparse_flash_attention_process` in `/workspace/src/vllm-ascend-unified/vllm_ascend/device/device_op.py:521–590`, which calls `_C_ascend.npu_sparse_flash_attention` for these BF16 caches at565. The owning SFA boundary is `attention/sfa_v1.py:1637–1682`, before V-up at2109. No TP collective or router is needed at this boundary.

## Independent layer58 key path

`vllm/model_executor/models/deepseek_v2.py:711–719` constructs replicated, unquantized `wk_weights_proj` and FP32-parameter LayerNorm. The actual Ascend path is `attention/indexer.py:320–353`: merged GEMM → first128 columns → mean-centered LayerNorm → interleaved RoPE (`glm_moe_dsa`) → BF16 Hadamard GEMM (`249–260`) → `npu_dynamic_quant` → scale cast toFP16 → scatter (`276–300`). Thus three possible first differences are projection output; norm/RoPE/Hadamard output; or quantization of identical rotated values. Equal stored FP16 scales do not prove equal pre-cast scales or equal quantizer input. A later stage probe must save these boundaries before naming quantization as the cause.

Old and new rank reports have identical indexer, SFA, Deepseek model and Ascend layernorm Python hashes; current files match them. Indexer SHA is `6513151ea1baa0a806abdbecbf776016ab16dd0cf6c75d9c5cd4aa02892b7dd1`, SFA `b5ad39a05b1323ac5048962f137969158298e4f637b5bd3e401cb4c4248ecfa0`. Rotary Python hashes differ between the historical runs, which is another reason to freeze Q before the four-call comparison. This audit ran CPU source/metadata/hash reads only and did not read or modify `m32_serial_audit` outputs.
