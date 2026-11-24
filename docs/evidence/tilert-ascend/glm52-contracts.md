# GLM5.2 operator contracts and Ascend scope

Scope: the 48 schemas in `libtilert_glm52_rocm.so` from TileRT 0.1.6.post3, SHA256 `4f9b74ef16e479042616cb43c0389078bd5bd7a3069f4ef63b7d186d8e1f98de`. [The complete schemas](schemas.tsv) retain optional arguments, defaults, and mutation annotations. Names beginning `glm5_` below belong to this GLM5.2 ROCm backend.

`S` denotes the number of decode/verification rows, `L` the allocated cache capacity, and `H` local attention heads. The release model wrapper uses batch 1, eight devices, hidden size 6144, 78 layers, 256 routed experts, top-8 routing plus one shared expert, and sparse attention top-2048. MTP depth is 0, 1, or 3; `S=num_mtp+1` is 1, 2, or 4. [Model arguments](../../../tilert/models/glm_5_2_rocm/model_args.py#L21), [MTP restriction](../../../tilert/models/glm_5_2_rocm/end2end.py#L159).

Native validation independently enforces eight devices and MTP depth 0/1/3: preparation at `0x76cf40` compares `npes` to 8 at `0x76cfea` and checks `num_mtp` at `0x76d013`/`0x76d01b`. The failure strings are at `0x1f02f` and `0x187a3`; MoE also contains `S must be 1, 2 or 4` at `0x19d29`. TP16 requires native changes. Any optional verification width above four would also require new native specializations; changing Python validation is insufficient. The current design scope keeps verification width M=1/2/4.

## State and mutation

The wrapper allocates 73 positional temporary tensors and retains weight/cache lists for the lifetime of one process-wide engine lease. `glm52_prepare_rank` has plain `Tensor[]` arguments with no mutation annotations, and subsequent decode/control schemas often have no tensor arguments at all. Nevertheless, [the wrapper reads generated tokens and counts from the retained temporary tensors](../../../tilert/models/glm_5_2_rocm/end2end.py#L565). These schemas do not completely describe state mutation to a graph compiler. The Ascend interface must make engine ownership, storage lifetime, effects, and ordering explicit.

Five workspace families use device `int64[8]` tables populated with each peer's `data_ptr()`: attention, FFN, top-1, top-p, and index transfer. [Pointer table creation](../../../tilert/models/glm_5_2_rocm/end2end.py#L327), [peer access and allocation](../../../tilert/models/glm_5_2_rocm/end2end.py#L379). The schema's plain `Tensor? sym` is a pointer table, not an ordinary immutable numerical tensor. Complete device-side flag ordering and atomic semantics are not established by this ABI extraction.

`cleanup()` calls `glm52_show_hands_go_home` before dropping retained tensors and the lease. The wrapper proves that ordering; native completion/join guarantees require further native inspection. `reset()` is a sequence-state operation; this audit does not claim it physically zeroes all KV storage. [Lifetime and cleanup](../../../tilert/models/glm_5_2_rocm/end2end.py#L606).

## All 48 entries

The contract column records facts visible in the shipped wrapper or selected native checks. “Unknown” denotes a contract field not recovered here. Tensor output buffers are provided by the caller unless the schema returns a value. All native device implementations require an Ascend implementation; the classification below identifies which mathematical and orchestration contracts can be retained.

### Engine and control: 14

| Operator | Established contract | Source |
| --- | --- | --- |
| `glm52_prepare_rank` | Rank plus positional weight/temp/cache lists; 73 temp slots; list lengths and every weight slot depend on rank/layer kind. Native list ownership details unknown. | [prepare](../../../tilert/models/glm_5_2_rocm/end2end.py#L361), [parameter packing](../../../tilert/models/glm_5_2_rocm/weight_converter.py#L1362) |
| `glm52_show_hands_prepare_money` | Configure `npes`, layers, capacity, MTP, top-p mode, temperature, top-p; model requires `npes=8`, MTP 0/1/3; prepare each rank first. | [prepare](../../../tilert/models/glm_5_2_rocm/end2end.py#L361), native checks above |
| `glm52_show_hands` | One host `int32[1]` token; token in `[0,154880)`; effects are stored in engine tensors. | [forward](../../../tilert/models/glm_5_2_rocm/end2end.py#L516), native token checks |
| `glm52_show_hands_prefill` | Token plus host `int32[1]` next token; next token may be -1 to select a sampled token. The standalone generator feeds prompt tokens one at a time. | [prefill](../../../tilert/models/glm_5_2_rocm/end2end.py#L544), [generator](../../../tilert/models/glm_5_2_rocm/generator.py) |
| `glm52_show_hands_decode` | No arguments or return; present in native registration, no direct shipped Python call. Exact single-call progress semantics unknown. | [schema inventory](inventory.json) |
| `glm52_show_hands_decode_n` | Integer step count, wrapper range `[1,L]`; completion tokens read from `AR_ACC`. A step count is not the cache capacity. | [decode and buffers](../../../tilert/models/glm_5_2_rocm/end2end.py#L527) |
| `glm52_show_hands_mtp_n` | Integer step count `[1,L]`, returns an integer used as produced-token count; per-step accepted counts are read separately from `AR_NUM`. | [MTP call](../../../tilert/models/glm_5_2_rocm/end2end.py#L558), [generator](../../../tilert/models/glm_5_2_rocm/generator.py) |
| `glm52_seed_draft` | Two integer tokens seed engine draft state; native token bounds apply. Exact writes to every draft slot unknown. | [seed](../../../tilert/models/glm_5_2_rocm/end2end.py#L561) |
| `glm52_show_hands_reset` | No arguments; resets the current sequence through native state. Exact physical cache clearing unknown. | [reset](../../../tilert/models/glm_5_2_rocm/end2end.py#L600) |
| `glm52_show_hands_set_cur_pos` | Set an integer current position in engine state. Complete cache-validity preconditions unknown. | [position](../../../tilert/models/glm_5_2_rocm/end2end.py#L603) |
| `glm52_show_hands_go_home` | Native cleanup entry before releasing retained tensors and process lease. Native thread/device completion details unknown. | [cleanup](../../../tilert/models/glm_5_2_rocm/end2end.py#L606) |
| `glm52_update_sampling` | Boolean top-p mode, finite positive temperature, `0<top_p<=1`. Python generator fixes per-rank candidate top-k to 256. | [sampling](../../../tilert/models/glm_5_2_rocm/end2end.py#L590), [generator](../../../tilert/models/glm_5_2_rocm/generator.py) |
| `glm52_temp_vars_size` | Returns an integer checked against 73 before allocation. | [layout guard](../../../tilert/models/glm_5_2_rocm/end2end.py#L145) |
| `glm52_mtp_seq_len` | Integer query, no direct shipped Python call. Return value for each engine state was not executed. | [schema inventory](inventory.json) |

### Buffer sizing, discovery, and peer access: 8

| Operator | Established contract | Source |
| --- | --- | --- |
| `supported_models` | Returns `str[]`; actual runtime contents unqueried. | [schema inventory](inventory.json) |
| `glm5_enable_peer_access` | Takes device count, returns count enabled; model wrapper requires all eight peers. Native uses HIP peer-access APIs. | [peer setup](../../../tilert/models/glm_5_2_rocm/ops/unprojo_allreduce.py#L78) |
| `glm5_broadcast_xfer_buf_bytes` | `(samples,topk)->int` bytes used for allocation; full native size formula unknown. | [index allocation](../../../tilert/models/glm_5_2_rocm/ops/index_collective.py#L15) |
| `glm5_unprojo_allreduce_sym_bytes` | `samples->int` bytes; allocate a uint8 buffer. Native padding/flag layout unknown. | [symmetric workspace](../../../tilert/models/glm_5_2_rocm/ops/unprojo_allreduce.py#L65) |
| `glm5_moe_sym_bytes` | `samples->int` bytes; full native formula unknown. | [MoE workspace](../../../tilert/models/glm_5_2_rocm/ops/moe_down_allreduce.py) |
| `glm5_top1_allreduce_sym_bytes` | `samples->int` bytes; full native formula unknown. | [top-1 workspace](../../../tilert/models/glm_5_2_rocm/ops/top1_allreduce.py) |
| `glm5_top_p_send_bytes` | Integer byte query; Python uses `SEND_BYTES=2080`, runtime return unqueried. | [top-p constants](../../../tilert/models/glm_5_2_rocm/ops/top_p.py#L6) |
| `glm5_top_p_sym_bytes` | `samples->int` bytes; full native formula unknown. | [top-p workspace](../../../tilert/models/glm_5_2_rocm/ops/top_p.py#L24) |

### Compute, communication, and step tails: 26

| Operator | Established shape/dtype and state contract | Source |
| --- | --- | --- |
| `gemv_w8a16_mfma_op` | BF16 `[S,6144]` to BF16 `[S,2048]`, `S=1/2/4`; packed FP8 bytes, FP32 scales `[16,48]`. Auxiliary standalone wrapper; end-to-end use not established. | [GEMV](../../../tilert/models/misc_rocm/ops/gemv_w8a16_mfma.py#L5) |
| `gemv_w8a16_mfma_cdna4_op` | Same mathematical GEMV dimensions; a different device weight packing. End-to-end use not established. | [CDNA4 GEMV](../../../tilert/models/misc_rocm/ops/gemv_w8a16_mfma_cdna4.py) |
| `gemv_w8a16_m4_cdna4_op` | Same mathematical GEMV dimensions; M4 packing. End-to-end use not established. | [M4 GEMV](../../../tilert/models/misc_rocm/ops/gemv_w8a16_m4_cdna4.py) |
| `rmsnorm_projx_wqkv_w8a16_mfma_op` | BF16 `[S,2048]` to Q `[S,4096]` and KV `[S,1024]`; FP32 gamma/scales, packed FP8 weights, `S=1/2/4`. Auxiliary dimensions differ from GLM5.2 hidden size. | [QKV wrapper](../../../tilert/models/misc_rocm/ops/rmsnorm_projx_wqkv_w8a16_mfma.py#L5) |
| `rmsnorm_raw_op` | BF16 input/output of equal shape, FP32 gamma matching last dimension; native dimension check permits 2048/6144/8192. | [RMSNorm](../../../tilert/models/misc_rocm/ops/rmsnorm_raw.py), native string at `0x18da3` |
| `glm5_llm_preprocess_op` | Embedding/position gather; int32 token IDs/current position, BF16 embedding and `[S,6144]` output, FP32 frequency table and `[S,64]` output. | [preprocess](../../../tilert/models/glm_5_2_rocm/ops/llm_preprocess.py#L35) |
| `glm5_sparse_select_s0123_op` | Fused indexer projections/normalization/RoPE; BF16 intermediate index scores and rotated Q; updates KI cache and int32 exchange workspaces. Optional FP8 KI/Q bytes and FP32 scales must be supplied together. Complete packed-weight dimensions are in the converter; native implementation unknown. | [index front](../../../tilert/models/glm_5_2_rocm/ops/sparse_select_s0123.py), [weight packing](../../../tilert/models/glm_5_2_rocm/weight_converter.py#L1192) |
| `glm5_sparse_index_op` | Q `[S,32*128]`, KI `[B,L,128]`, index scores `[S,32]` BF16; int32 positions; FP32 logits `[S,stride]`, int32 histogram. Stride rounds active length to a multiple of 8. | [index scores](../../../tilert/models/glm_5_2_rocm/ops/sparse_index.py#L21) |
| `glm5_sparse_index_fp8_op` | FP8-e4m3 bytes for Q/KI plus FP32 row scales; index scores remain BF16; logits/histogram contracts as above. This is not INT8 index-cache quantization. | [FP8 index](../../../tilert/models/glm_5_2_rocm/ops/sparse_index.py#L41) |
| `glm5_topk_select_op` | FP32 logits; mutates int32 histogram, tie-pair workspace and `[S,topk]` indices; default topk 2048. Complete native ordering of equal-score indices unknown. | [selection](../../../tilert/models/glm_5_2_rocm/ops/sparse_index.py#L103) |
| `glm5_xfer_send_indices_op` | Contiguous int32 indices, mutable transfer buffer and integer flag; reference wire layout interleaves each index with its flag. | [wire packing](../../../tilert/models/glm_5_2_rocm/ops/flash_sparse_mla.py#L30) |
| `glm5_broadcast_selected_token_ids_op` | Int32 selection plus peer pointer table, rank/count and epoch flag. Remote writes are not described by an ordinary tensor mutation annotation. | [broadcast](../../../tilert/models/glm_5_2_rocm/ops/index_collective.py#L23) |
| `glm5_pure_mla_allreduce_op` | BF16 `[S,6144]` input/output; fused MLA and projection/reduction; Q-nope `[S,H*512]`, Q-PE `[S,H*64]`, BF16 or FP8 cache, FP32 reduction partials, many int32 exchange buffers. Wrapper has `H=8/10`, four-row workspaces and mutable KV/PE caches. | [fused attention](../../../tilert/models/glm_5_2_rocm/ops/pure_mla_allreduce.py#L16) |
| `glm5_pure_mla_moe_layer_op` | Fused attention and MoE, with seven MoE weight tensors and ten MoE scratch tensors; out and MoE x_out must not alias. `moe_kind` chooses existing MoE arms. Complete intra-launch dependency protocol unknown. | [attention/MoE](../../../tilert/models/glm_5_2_rocm/ops/pure_mla_moe_layer.py), native constraints in inventory |
| `glm5_unprojo_allreduce_op` | BF16 projection input `[S,256*H]` to residual-added BF16 `[S,6144]`; packed FP8 W_o, FP32 scales; optional peer table; wrapper heads 8/10/16 and max sequence 4. | [projection/reduction](../../../tilert/models/glm_5_2_rocm/ops/unprojo_allreduce.py#L7) |
| `glm5_padded_allreduce_add_op` | Optional BF16 residual, peer table and flags; writes BF16 `[S,6144]`. Partial-data location and native padding protocol unknown. | [reduction leg](../../../tilert/models/glm_5_2_rocm/ops/index_collective.py#L41) |
| `glm5_rmsnorm_upgate_silu_op` | Dense FFN front: BF16 `[S,6144]` to BF16 `[S,1536]` in TP8; FP32 gamma/scales and packed FP8 weights. | [dense front](../../../tilert/models/glm_5_2_rocm/ops/upgate_silu.py) |
| `glm5_down_allreduce_op` | Dense FFN down projection/reduction: BF16 `[S,1536]` to `[S,6144]`; packed FP8 weights/scales, optional BF16 residual and peer table. | [dense down](../../../tilert/models/glm_5_2_rocm/ops/down_allreduce.py) |
| `glm5_fused_moe_allreduce_v4_op` | `S=1/2/4`; hidden/norm/output BF16 `[S,6144]`, router scores FP32 `[S,256]`, probs FP32 `[S,8]`, indices int32 `[S,8]`, mid BF16 `[S,9,256]`; flags int32 `[512]`, score lines `[S,32,32]`, mid pairs `[S,9,256]`. FP8 weight banks include shared expert. | [MoE buffers](../../../tilert/models/glm_5_2_rocm/ops/fused_moe_allreduce.py#L15) |
| `glm5_fused_moe_allreduce_w8a8_v4_op` | Same exposed MoE buffers; FP8 activation quantization and K128 weight packing are additional semantics. The name W8A8 does not mean Ascend INT8 activation/INT4 weight compatibility. | [W8A8 arm](../../../tilert/models/glm_5_2_rocm/ops/fused_moe_w8a8.py), [quantizers](../../../tilert/models/glm_5_2_rocm/ops/upgate_silu_w8a8.py) |
| `glm5_eh_proj_allreduce_op` | MTP embed/hidden fusion: BF16 `[S,6144]` inputs/output, FP32 gammas; full BF16 W_eh `[6144,12288]`, rank slice width 1536. Optional native token/embedding/frequency arguments are not exercised by this standalone wrapper; complete optional-arm shapes unknown. | [EH projection](../../../tilert/models/glm_5_2_rocm/ops/eh_proj_allreduce.py#L49) |
| `glm5_rmsnorm_head_proj_op` | BF16 `[S,6144]` to FP32 logits `[S,19360]`, optional BF16 norm output; BF16 packed head, FP32 gamma. Native optional top1 workspace, pointer table and draft/last-token commits extend the five-argument wrapper. | [head projection](../../../tilert/models/glm_5_2_rocm/ops/rmsnorm_head_proj.py#L5) |
| `glm5_top1_allreduce_op` | FP32 logits; int32 `[S]` token output; optional peer table and mutable commit_draft/commit_last. Complete optional commit shapes unknown. | [top-1](../../../tilert/models/glm_5_2_rocm/ops/top1_allreduce.py) |
| `glm5_top_p_local_scores_op` | FP32 logits, positive temperature, uint8 send buffer of 2080 bytes per sample; local top-256 candidate set and distributed score metadata. | [local scores](../../../tilert/models/glm_5_2_rocm/ops/top_p.py#L59) |
| `glm5_top_p_sample_op` | Send buffer, int64 seeds/positions, optional peer table and draft tokens; int32 `[S]` token and FP32 `[S]` probability outputs; `verify_seq` participates in draft verification. | [sampling reference and call](../../../tilert/models/glm_5_2_rocm/ops/top_p.py#L74) |
| `glm5_ar_tail_no_mtp_op` | Explicit mutable token ID, position, accepted-token/count and sampling-position buffers; engine allocations are int32 `[1]`, `[1,L+1]` and int64 position arrays. No direct shipped Python call; complete native invocation-specific shapes unknown. | [engine allocations](../../../tilert/models/glm_5_2_rocm/end2end.py#L201), [schema inventory](inventory.json) |

## Cache, quantization, and 8K capacity

The release allocates MLA latent KV as BF16 `[1,L,512]`, with a separate BF16 RoPE plane `[1,L,64]`; the optional FP8 latent format is uint8 `[1,L,528]` including scale bytes. Index KI is BF16 `[1,L,128]` or a flat FP8 byte plane plus FP32 scales. [Cache allocations](../../../tilert/models/glm_5_2_rocm/end2end.py#L295). These optional FP8 formats are independent of a target checkpoint's C8 recipe; an INT8 indexer cache does not imply FP8 MLA KV.

`L=8192` is a capacity choice, while `S=1/2/4` is the released per-step width. The wrapper requires `L>=2048`, validates the checkpoint maximum position, and stops decode before available cache capacity is exhausted. An 8192-token prompt requires additional capacity for generated and speculative tokens. [Capacity guard](../../../tilert/models/glm_5_2_rocm/end2end.py#L379), [checkpoint maximum](../../../tilert/models/glm_5_2_rocm/checkpoint_config.py#L134), [generator](../../../tilert/models/glm_5_2_rocm/generator.py). An 8K model run was not performed by this audit.

The weight converter explicitly requires an FP8/e4m3 checkpoint and repacks weights for ROCm kernels. It is not a loader for the target Ascend W4A8C8 checkpoint. [Checkpoint quantization validation](../../../tilert/models/glm_5_2_rocm/checkpoint_config.py#L174), [FP8 packing](../../../tilert/models/glm_5_2_rocm/weight_converter.py#L1192). Shared-index layers use attention TP8; full-index layers use rank 0 for indexing and the other ranks for MLA. This rank geometry and its 8/10-head packing also require redesign for 16 Ascend dies.

The first three dense layers have dense FFNs, not dense full-context attention. They receive indexer weights on rank 0 and MLA weights on the other ranks. At context length above top-2048, the attention reference requires a sparse selection. [Layer sharding](../../../tilert/models/glm_5_2_rocm/weight_converter.py#L416), [attention reference](../../../tilert/models/glm_5_2_rocm/ops/flash_sparse_mla.py#L52).

## Work decomposition for Ascend batch 1, 8K, W4A8C8

The 48 entries split into 26 device compute/communication/tail operations, 12 stateful engine/control operations, nine queries, and one peer-access setup operation. The nine queries are the six byte-size helpers, `glm52_temp_vars_size`, `glm52_mtp_seq_len`, and `supported_models`; their semantics can be retained while their Ascend values and registrations are implemented anew. Zero native binaries are reusable on Ascend. Full-wheel ownership can therefore be recorded as 48 GLM5.2 names in scope and 127 other names outside this model's implementation denominator.

| Work | Reuse | Rewrite or add |
| --- | --- | --- |
| Model semantics and host interfaces | Tokenizer/generation conventions, layer pattern, routing/sampling equations, acceptance-buffer meaning, explicit lifetime discipline | Ascend backend selection, device APIs, state ownership, effect declarations, graph/stream scheduling, cancellation/completion semantics |
| Parameters and cache | Logical tensor names, dimensions, per-layer dependency order | W4/INT8 checkpoint recipe and exact scale/offset/layout ingestion, TP16 sharding, kernel-native packing, target INT8 index cache; preserve MLA BF16 if that is the actual checkpoint contract |
| Compute | Mathematical RMSNorm, projection, sparse index/top-k, MLA, MoE, head and MTP semantics; existing CANN operators for supported shapes and recipes | Operand layouts and state integration; tiling/fusion changes after profiling; new kernels for missing capabilities or measured gaps |
| Communication | Logical reductions, index broadcast, ownership of persistent workspaces | Ascend peer-address/registration mechanism, topology-aware 16-rank protocol, flag epochs, visibility and memory-order proof, progress and deadlock constraints; GPU pointer-table bytes cannot be reused directly |
| MTP wider than 4 (optional) | The meaning of accepted prefix, token commit, position advancement, and shared draft selection | Wider target verification, longer draft orchestration, expanded scratch/cache reservations, rejection/rollback validity, deterministic RNG/selection handling, and a measured acceptance-aware scheduling policy; outside the current M=1/2/4 scope |
| Performance work | Fused attention/MoE boundaries are candidate decompositions | Measured choice of fusion, persistent execution versus graph replay, rank asymmetry, collective granularity, and quantization overlap; schema presence supplies no latency result |

The four auxiliary MFMA GEMV/QKV entries should remain in the coverage inventory even if the initial GLM5.2 path does not expose standalone replacements for them. An initial model path can use different internal decomposition, provided its declared API coverage and numerical/state contracts are explicit. No native ROCm binary is directly reusable as an Ascend kernel.

The current Ascend source exposes `NativeDecodeEngine`, `NativeMtp` and
`AscendPDEngine`, backed by CANN operators and HCCL. Compatibility registration
under the recovered ROCm `torch.ops.tilert.*` names remains unimplemented.
[Current coverage and evidence](../ascend-decode/offline-development.md)
separate executable control checks from pending TP16 numerical acceptance.
