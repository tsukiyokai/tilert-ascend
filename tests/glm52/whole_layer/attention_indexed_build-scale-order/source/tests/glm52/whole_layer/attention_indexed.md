# Native indexed attention

`attention_indexed_native.h` executes main MLA Prolog V3, indexer Q/K front,
indexer cache write and top-k selection, sparse MLA, and output projection in
one resident mixed AIC:AIV 1:2 launch. The output is the TP16 rank's BF16
attention partial `[M,6144]`, before TP reduction. The entry covers backbone
layers that compute their own indexer. The existing shared-indexer entry is
unchanged; input normalization, TP reduction, and MTP's cache-write-only path
are outside this entry.

The CANN 9.2 `dav-2201` build
`attention_indexed_build/whole_layer_indexed_attention.so` has SHA-256
`ff5525b8a73ad20d48f35625be3c7a3b383d0bd874dee57ba1d41d10ba599a89`.
Its device code and Torch binding compile and link, all four comparison
libraries load together, and the CPU contract check covers 768 geometries.
On A3-162, M=1 eager and M=1/2/12/32 captured tests passed in C8 and BF16
modes, with four changing inputs per case. Every retained production
boundary matched raw bytes; the diagnostic FP32 norm statistics also matched
in these composition tests. See the [eager report](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-m1.json)
and [captured replay report](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-replay.json).

The [paired timing report](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-timing.json)
uses sequence length 8192, 24 groups, one weight bank, and matching queued
input clones in both graphs. Each cell is the median of six paired samples,
in microseconds per attention call before TP reduction:

| Indexer cache | M | Indexed launch | Separate chain | Reduction |
|---|---:|---:|---:|---:|
| C8 | 1 | 232.369 | 265.254 | 12.40% |
| C8 | 12 | 288.416 | 307.314 | 6.15% |
| C8 | 32 | 406.476 | 419.642 | 3.14% |
| BF16 | 1 | 219.661 | 258.942 | 15.17% |
| BF16 | 12 | 274.687 | 307.861 | 10.78% |
| BF16 | 32 | 373.632 | 411.479 | 9.20% |

All six pairs favor the composed launch in each row. These measurements
compare the diagnostic composition with the specific separate chain below;
they do not measure real model weights, TP reduction, a complete layer, or
end-to-end TPOT.

## Inputs and retained values

`indexed_attention_prepare(token_x, pages, table_pages, index_pages,
index_table_pages, quantized, blocks=0)` returns scratch, immutable tiling,
a dictionary of diagnostic byte offsets, the physical launch group count,
and the context's sync address. Default groups use all physical AICs, matching
the selector's tested merge partition and equal-score ordering.

`indexed_attention_out` accepts these groups of tensors on one NPU:

| Group | Contract |
|---|---|
| Hidden and quantized input | BF16 hidden `[M,6144]` after the layer's input normalization; corresponding INT8 codes `[M,6144]` and FP32 scales `[M,1]` |
| Nine main weights | DQ, UQ/QR, UK, DKV/KR, three channel scales, CQ gamma, CKV gamma, in the order used by `attention_host.h` |
| Six indexer weights | Packed INT8 WQ `[16,2048,256]`, original FP32/BF16 scales `[4096]`, packed BF16 WK/head-weight matrix `[5,6144,32]`, FP32 gamma/beta `[128]`, BF16 Hadamard `[128,128]` |
| Three tail weights | BF16 value tiles `[16,512,64]`, INT8 output tiles `[24,1024,256]`, FP32 output channel scales `[6144]` |
| Optional tail bias | INT32 accumulation bias or FP32 output bias `[6144]` |
| Each metadata list | BF16 sine/cosine `[M,64]`, INT64 slots `[M]`, INT32 cumulative query/key lengths `[1]`, INT32 page table `[1,table_pages]` |
| Main caches | BF16 latent `[pages,128,1,512]` and BF16 RoPE `[pages,128,1,64]` |
| Indexer caches | INT8 or BF16 keys `[index_pages,128,1,128]`; C8 also uses FP16 scales `[index_pages,128]`, while BF16 supplies an empty FP16 tensor |

M is 1–32. The three quantized main matrices keep the vendor's physical NZ
format. Other tensors require contiguous physical ND storage. The binding
checks real NPU storage formats, devices, geometry, and writable aliases;
execution uses `stream()` to submit queued Torch producers before the direct
launch. Metadata values remain device data during capture. The caller must
provide matching logical query positions and lengths for main attention and
the indexer, while their physical page tables and slot numbers may differ.

The Prolog phase enables `queryNormFlag` and exports INT8 normalized Q
`[M,2048]` plus FP32 scales `[M]` into a retained prefix. Those exact tensors
feed the indexer WQ projection. The BF16 hidden input remains live through
WK/head-weight projection. Main attention has four local heads; the indexer
has 32 replicated heads. C8 describes the indexer cache; the main MLA cache
in this contract remains BF16.

The front preserves its BF16 projection, normalization, RoPE, and Hadamard
boundaries. C8 Q/K quantization produces FP16 scales and FP16 head weights;
the BF16 branch skips Hadamard and quantization. The tail retains its
shape-specific channel-then-token dequantization order. See
[the front contract](indexer_front.md) and [the tail contract](../mla_tail/README.md).
The reports above use original FP32 WQ scales and four changing replays.
An extended standalone test exposes a WQ dequantization mismatch with that
library's channel-first order. The `attention_indexed_build-scale-order`
candidate uses combined channel/token scales for WQ K=2048/N=4096 with both
original FP32 and BF16 scales; its device acceptance is separate from those
reports. The adapter retains the original WQ descriptor, and the dtype flag
controls loading only. The tail's shape-specific arithmetic is unchanged.

## Synchronization and workspace

Every resident AIC and AIV enters the original phase barriers. Each phase
drains its pipes, publishes GM writes, completes mixed `SyncAll<false>`, and
resets the pipe after vendor objects leave scope. The main query and RoPE
survive until SFA. The indexer writes its new keys before selection; selected
indices preserve their original top-k order and are published before SFA.
All current-row cache writes may finish first because the selector and SFA
retain their causal length/mask contracts. The tail consumes the published
BF16 SFA latent and installs its own Matmul system workspace.

The selector's internal AIV-only barrier uses a different hardware flag from
mixed barriers. The SFA vendor headers declare generic names globally;
`attention_indexed_sfa.h` isolates those declarations from LI/QLI name lookup
after including SDK headers globally. This shim changes declaration scope
only and is used by the composed translation unit.

Scratch contains a retained prefix followed by separate main-attention,
front, selector, and tail arenas. Prolog and SFA share only the main-attention
arena already used by the standalone entry. No other phase arenas overlap.
At 24 groups and an 8K sequence:

| M | C8 scratch bytes | BF16 scratch bytes | Tiling bytes |
|---:|---:|---:|---:|
| 1 | 100,029,952 | 126,600,704 | 40,960 |
| 12 | 108,518,912 | 135,136,256 | 40,960 |
| 32 | 130,138,624 | 156,840,448 | 81,920 |

These numbers exclude weights, caches, external inputs, and the final output.
The diagnostic prefix retains normalized Q, front outputs, ordered top-k,
and SFA latent. Front intermediates and main Q/RoPE also remain available.

## Engine integration contract

[The engine adapter](../../../tilert/models/glm_5_2_ascend/attention_native.py)
provides `load()`, `WorkspacePool(blocks=0)`, and
`install(attention_layer, pool)` after weight loading and cache binding.
Its source contract is vLLM-Ascend
`8237ee6d690d1399bc6eb48ef5977ac2bd470f3f`,
`vllm_ascend/attention/sfa_v1.py` and `attention/indexer.py`, with vLLM
`35a49fcfc2295d04fb252c1080f8d7bd9d888e25`,
`vllm/model_executor/models/deepseek_v2.py`. The integration point is
`AscendSFAImpl.forward`, where normalized hidden states, cache ownership,
metadata, and the caller's output buffer coexist. A completed native output
is already projected: preserve the output linear's original TP reduction
policy and copy into the caller's output; do not project it again.

For the current [TP16 cache layout](../../../tilert/pd_vllm/ascend_layout.py),
backbone layers 0, 1, 2, 70, and 74 use BF16 indexers; layers 6 through 66
at stride four use C8 indexers. The other 57 backbone layers reuse selected
indices through the existing shared-indexer attention entry. Main MLA caches
remain BF16. MTP layer 78 has a separate runtime `skip_topk` policy and must
still write its indexer cache when selection is skipped; this indexed entry
does not implement that path.

The adapter retains these lifetimes:

- Slots are live replay inputs. `sfa_v1._int64_kv_slots` caches its first cast
  by tensor identity. [CannGraph](../../../tilert/models/glm_5_2_ascend/cann_graph.py)
  refreshes `slots_i64` before every execution but currently binds that buffer
  only to SFA metadata. Indexer slots must use an equally refreshed INT64
  buffer. The adapter captures an INT32-to-INT64 copy for each distinct slot
  source and owns a separate destination per source and stream. Capture
  deduplicates only repeated uses of that same source; eager warmup does not
  consume the reserved capture buffers. Main and indexer page tables,
  lengths, and RoPE tensors retain their separate metadata ownership.
- BF16 normalized hidden `[M,6144]` remains live until the indexer WK/head
  projection completes. [NormForward](attention_norm_native.h) currently
  places that tensor in an arena overwritten by full attention. Indexed
  composition must retain it outside that arena; preserving only the INT8
  codes and FP32 scales is insufficient.
- Ordered top-k belongs to the model's persistent cross-layer buffer,
  allocated in `DeepseekV2Model.__init__`. An indexed producer must publish
  its result there before scratch can be reused; a shared-indexer consumer
  reads that buffer through the original index-cache helper. A scratch view
  cannot replace the persistent buffer's ownership.
- Workspace and tiling are keyed by device, stream, exact M, both cache/page
  table geometries, indexer mode, and launch groups. M changes offsets even
  within the same 16/32-row tiling capacity. Reserve all needed entries before
  capture and retain them, packed weights, and outputs for the graph's life.

Main weights come from the loaded implementation's absorbed `weight_dq`,
`weight_dkv_kr`, `weight_uq_qr`, `W_UK_T`, dequantization scales, and norm
parameters. Indexer WQ is the loaded transposed NZ INT8 matrix; WK/head
weights are the replicated BF16 `[160,6144]` merged projection. Pack copies
outside capture and check the actual quantization scheme and dtype. C8 cache
scales bound by the engine have extra singleton dimensions; view the same
physical ND storage as `[pages,128]` without copying or replacing the cache.

The initial supported engine scope is one request, exact M=1–32, TP16 without
context or sequence parallelism, Prolog V3, BF16 main caches, INT8 dynamic
linear inputs, and ungated output projection. Keep prefill outside this range,
other quantization/layout policies, and MTP on their original paths. Input
dynamic quantization and both layer TP reductions retain their original
positions. A complete layer additionally requires the post-attention norm,
FFN, and communication phases; these attention tests do not validate them.

Call `pool.reserve_capture(reference)` under the forward context before
capture, with BF16 `reference[M,6144]`. The direct `run_body` interface returns
the rank partial and ordered top-k; its caller publishes persistent top-k
and performs TP reduction. The pool's `reserve(reference, geometry,
slot_sources)` interface supports the same preparation without a model
context. The indexed library's default uses all physical AIC groups; the
existing shared library defaults to eight on A3. An explicit `blocks=24`
selects the tested 24-group partition for both entries.

For the vendor revision above, `attention/sfa_v1.py:353` derives
`AscendSFABackend` directly from `AttentionBackend`.
`utils.py:1914–1919` therefore returns false for `use_updatable_graph(SFA)`;
`compilation/acl_graph.py:297–300` calls ordinary graph replay.
`AscendSFAImpl.update_graph_params` at `sfa_v1.py:928–937` is empty, and
`acl_graph.py:322–328` skips absent workspace entries. No attention task
handle registration is required on this path. Metadata tensor addresses
remain fixed and their values are refreshed in place. This is a source
check of the pinned vendor revision; a real ACLGraphWrapper device run is
still required for engine acceptance.

The adapter's 23 CPU tests cover weight packing and original WQ scale dtype,
static/runtime guards,
TP finalization, persistent top-k ownership, exact-M/geometry/stream pools,
warmup followed by capture reservation, and independent A→B→A slot sources.
The indexed and shared libraries also load together with their four schemas
while `torch.npu.is_initialized()` remains false. These checks do not replace
device replay through the installed model.

## Adapter memory

The adapter retains the original weights for fallback. Its packed copies
add `78 × (1 + 6) + 21 × (8 + 1.875) = 753.375 MiB` per TP rank:
value/output tiles for all 78 backbone layers, and WQ/WK tiles for the
21 indexed layers. With original BF16 output channel scales, their FP32
copies add 1.828125 MiB; WQ retains its original scale storage. The five
BF16-indexer Hadamard placeholders add 0.15625 MiB.
The total is 755.359375 MiB, assuming the remaining source tensors already
have their required contiguous ND storage. Original main projection weights
and compatible norm/scaling tensors are referenced directly.

For one cache geometry per mode and page tables containing at least four
pages, the three retained pool entries per exact M and stream occupy:

| M | Indexed 24 groups, shared 8 groups | All entries 24 groups |
|---:|---:|---:|
| 1 | 235.5503 MiB | 274.1128 MiB |
| 12 | 254.7642 MiB | 298.3267 MiB |
| 32 | 303.2939 MiB | 359.3564 MiB |

These are CPU-evaluated sizes from the current C++ layout contracts,
including scratch, tiling, and the three output partials. Warmup and capture
on different streams retain separate entries. For example, the default
M=12 pair plus packed weights adds approximately 1264.89 MiB per rank.
Each independent INT32 slot source also needs an INT64 buffer of `8 × M`
bytes per stream. Other visited widths and cache geometries add entries;
the nine supported widths on one stream total 2.25047 GiB of pool storage
with default groups. Cache storage, temporary packing/quantization tensors,
allocator rounding, and graph-pool reservations are excluded. These sizes
are allocation formulas, not measured process HBM.

## Build and compare

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
OPS_TRANSFORMER_ROOT=/workspace/attention_vendor \
  bash tests/glm52/whole_layer/attention_indexed_build.sh
python3 -m unittest discover -s tests/glm52/whole_layer \
  -p attention_indexed_test.py -v
python3 tests/glm52/whole_layer/attention_indexed_check.py \
  --widths 1 --modes int8 bf16 --no-timing \
  --output logs/attention-indexed-m1.json
python3 tests/glm52/whole_layer/attention_indexed_check.py \
  --capture --queued-producer --widths 1 2 12 32 \
  --modes int8 bf16 --no-timing \
  --output logs/attention-indexed-replay.json
```

`CANN_ROOT` and `PYTHON_BIN` override `ASCEND_HOME_PATH` and `PYTHON`.
The build requires the recorded Prolog, SFA, LI, and QLI vendor headers,
the CANN Matmul tiling API, Torch/torch_npu headers, and `dav-2201` bisheng.
It invokes no host CANN operator from the native entry. The build manifest
records native/vendor source hashes, SDK versions, and library identity;
source snapshots beside the library are checked against that manifest.

The reference chain uses stock Prolog with Qnorm export, the separately
validated native front and selector, stock SFA, and the promoted native tail.
Both arms receive the same changing inputs, weights, cache contents, and
metadata. Main/indexer page maps differ and change between replays. Queued
producer mode places matching hidden/code/scale clones inside both graphs.

Acceptance requires raw-byte equality at every retained production boundary,
including ordered top-k, the entire main and indexer caches, latent, and final
projection. Internal FP32 LayerNorm statistics remain diagnostics; the BF16
normalized key is its production boundary. Failure dumps clone each typed
tensor independently to preserve cross-dtype scratch views safely.

Optional paired timings compare this diagnostic composition against that
specific separate chain. They are neither a full stock attention comparison
nor a layer/TPOT measurement. The separate-buffer count excludes stock CANN
internal workspaces and graph allocations.
