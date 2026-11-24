# Native indexed attention

`attention_indexed_native.h` executes main MLA Prolog V3, indexer Q/K front,
indexer cache write and top-k selection, sparse MLA, and output projection in
one resident mixed AIC:AIV 1:2 launch. The output is the TP16 rank's BF16
attention partial `[M,6144]`, before TP reduction. The entry covers backbone
layers that compute their own indexer. The existing shared-indexer entry is
unchanged; input normalization, TP reduction, and MTP's cache-write-only path
are outside this entry.

The accepted CANN 9.2 `dav-2201` build
`attention_indexed_build-scale-order/whole_layer_indexed_attention.so` has
SHA-256 `6588e1dcde9dd843c10479ce1276ae069d89a9e60c16bc296f20ba4afb7c1c34`.
It is the adapter's default library. The actual adapter body passed 128
changing queued replays covering M=1/2/12/32, C8/BF16 indexer caches, and
original FP32/BF16 WQ scales. Its standalone front passed 512 replays against
stock operators. Retained production boundaries match bytes; FP32 LayerNorm
statistics remain diagnostic. The CPU layout check covers 768 geometries.

The [eight-bank timing report](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-scale-order-bank-summary.json)
uses sequence length 8192, 24 groups, and eight equal-valued weight banks
with independent addresses, totaling 280.658 MiB, above the die's 192 MiB L2.
Each graph contains 16 independent attention calls. Each event measures
30 graph replays, with 30 alternating event pairs and no queued input clones.
The table reports median microseconds per attention call before TP reduction:

| Indexer cache | M | Indexed launch | Separate chain | Reduction |
| --- | ---: | ---: | ---: | ---: |
| C8 | 12 | 297.034 | 308.674 | 3.77% |
| C8 | 32 | 402.488 | 423.165 | 4.89% |
| BF16 | 12 | 285.267 | 306.318 | 6.87% |
| BF16 | 32 | 386.310 | 405.884 | 4.82% |

All 30 pairs favor the composed launch in each row. The bank harness checks
retained boundaries only after the final call; it does not establish an
independent numerical gate for every bank. The front and adapter replay
gates above use separate numerical tests. These timings compare the diagnostic
composition with the separate chain below; they do not measure real model
weights, TP reduction, a complete layer, or end-to-end TPOT.

The earlier `ff5525b8a73ad20d48f35625be3c7a3b383d0bd874dee57ba1d41d10ba599a89`
library's [eager](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-m1.json),
[four-replay](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-replay.json),
and [single-bank timing](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-timing.json)
reports remain historical evidence. They precede the WQ scale-order correction.

## Inputs and retained values

`indexed_attention_prepare(token_x, pages, table_pages, index_pages,
index_table_pages, quantized, blocks=0)` returns scratch, immutable tiling,
a dictionary of diagnostic byte offsets, the physical launch group count,
and the context's sync address. Default groups use all physical AICs, matching
the selector's tested merge partition and equal-score ordering.

`indexed_attention_out` accepts these groups of tensors on one NPU:

| Group | Contract |
| --- | --- |
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
The accepted front uses combined channel/token scales for WQ K=2048/N=4096
with both original FP32 and BF16 scales. The adapter retains the original WQ
descriptor, and the dtype flag controls loading only. Saved device witnesses
in the front contract distinguish this order from channel-first multiplication.
The tail's shape-specific arithmetic is unchanged.

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
| ---: | ---: | ---: | ---: |
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
The source audit uses vLLM-Ascend
`8237ee6d690d1399bc6eb48ef5977ac2bd470f3f`,
`vllm_ascend/attention/sfa_v1.py`, `attention/indexer.py`, and `ops/mla.py`,
and the model definitions from vLLM
`35a49fcfc2295d04fb252c1080f8d7bd9d888e25`,
`vllm/model_executor/models/deepseek_v2.py`. These two snapshots are not an
import-compatible runtime pair: the vendor requires
`vllm.v1.attention.ops.pcp`, which is absent from 35a49. The vendor's
`.github/vllm-main-verified.commit` instead names official vLLM
`a98247ab4db686ee03c66d5feb3c761e52a2f8ab`.
That pair passes [actual CPU imports](../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/README.md)
of SFA, the indexer wrapper, graph wrapper and forward context in an existing
Python 3.12 / Torch 2.10 image without NPU initialization. This is separate
from the native device gates recorded under Torch 2.12; layer construction
and the actual vendor hook still require validation.
The integration point is
`AscendSFAImpl.forward`, where normalized hidden states, cache ownership,
metadata, and the caller's output buffer coexist. A completed native output
is already projected: preserve the output linear's original TP reduction
policy and copy into the caller's output; do not project it again.

In this vendor revision, `ops/mla.py:39–74` constructs an `IndexerWrapper`
whose `impl` is the `AscendSFAIndexerBackend`. The native adapter requires
these exact types and reads backend-only RoPE/parallelism flags and
Hadamard buffers from `wrapper.impl`; the wrapper does not forward those
attributes. Its cache property continues to reference the backend's cache.

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

The adapter's 25 CPU tests cover weight packing and original WQ scale dtype,
static/runtime guards,
TP finalization, persistent top-k ownership, exact-M/geometry/stream pools,
warmup followed by capture reservation, and independent A→B→A slot sources.
Three of these cases execute the original 8237 `IndexerWrapper` class and
constructor with real `torch.nn.Module` semantics, including packing from
its backend without extra attribute forwarding. Set `VLLM_ASCEND_SOURCE`
to that checkout for these source-replay tests; they check the source hash
and skip when the checkout is absent. They do not import the full vendor
package or validate its device execution.
The indexed and shared libraries also load together with their four schemas
while `torch.npu.is_initialized()` remains false.

The actual `run_body`, packing, and workspace-pool probe passed M=1/2/12/32
in C8 and BF16 modes with eight changing queued replays per case for original
[FP32 scales](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-adapter-fp32-scale-order.json)
and [BF16 scales](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-adapter-bf16-scale-order.json).
All retained production boundaries, full caches, ordered top-k, and final
outputs matched bytes across these 128 replays. Each case also copies top-k
to persistent storage and feeds it to the shared-indexer entry. Main/indexer
slot sources have independent addresses and page maps; default-stream warmup
precedes zero-warmup capture on another stream. This tests the real adapter
body with synthetic shard weights; `AscendSFAImpl.forward`, ACLGraphWrapper,
and TP reduction remain outside that probe.

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
| ---: | ---: | ---: |
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
