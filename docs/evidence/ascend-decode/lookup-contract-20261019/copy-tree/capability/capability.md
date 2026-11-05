# Fixed-32 copy-tree capability on A3

Source scope: a3-194, CANN 9.2.0 B114 and the current vLLM-Ascend sources.
[Source hashes](source-sha256.json) bind the installed kernel/header snapshots.
No device execution was performed for this audit.

The CPU copy-tree replay reduced 147 to 132 rounds on one old, unaccepted
8192+1024 trajectory. Its 11.36% allowance is an acceptance ratio, not a measured
tree execution budget or model result. Kernel work remains deferred until an
accepted prompt demonstrates useful gain and a measured round-cost budget.

## Existing interface and its limit

`indexer.py:494-496` exposes
`topk_selector(q_li, weights, indexer, metadata, q_li_scale)` after RoPE,
quantization and key-cache writes. The current A3 C8 call is
`npu_quant_lightning_indexer`, TND/PA_BSND, sparse_count=2048, sparse_mode=3
(`device_op.py:488-505`). Its installed signature accepts lengths and a block
table; it accepts no arbitrary ancestor mask and returns indices only
(`aclnn_quant_lightning_indexer.h:27-46`). Nonquantized LI has a values output.

Ordinary topK followed by a sibling mask is incorrect: with K=2, sibling score 100
and legal prefix scores 10/9, postmask retains only 10 and cannot recover 9.

A prefix-only call must exclude ALL new physical slots L..L+31, including the
pending root. It needs key length L and mode 0. Keeping mode 3 instead exposes
only L-31+i prefix keys at query row i (`service_vector:349-360`). The installed
kernel implements mode 0; installed-host tiling/key acceptance was not exercised.
Use separate prefix-length storage, preserving the main seq_lens=L+32.

## Sorted output does not close an exact boundary merge

Installed C8 output is score ordered: tile SortAll/MergeSort at
`service_vector:402-405`, direct extraction at413-421, cross-core MrgSort and
extraction at 588-658. For a COMMON total order, at most 32 tail keys can evict
only the weakest 32 entries of prefix top2048. Therefore a mathematical topK can
be obtained by retaining 2016 entries and selecting 32 from the last 32 plus tail.
This statement does not prove equivalence to the installed kernel's tied output.

MrgSort preserves input order on ties, rather than sorting index payloads
([official CANN9.1 API constraints](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/advanced_api/sort_operations/MrgSort-93.md)). `vector.h:153-169` merges the new tile as src1 before the previous
accumulator; final reduction follows core order. `kernel_arch22:268-337`
changes that grouping with sequence length. A legal 24-core counterexample is
M32/L8192 with all scores zero: prefix-only has 32 basic blocks, while adding 32
tail rows gives 40. For query rows 4..7, the preferred prefix tile can change
from tile 1 (IDs 2048..4095) to tile 0 (IDs 0..2047). More than 32 prefix IDs change.
The bottom32 proposal is consequently not established as exact.

Its rescorer would also need the original arithmetic:

1. INT8 QK accumulates into INT32 (`service_cube:516-529`).
2. Fixpipe applies DEQF16, ReLU and scale 0x3a800000=2^-10 (`535-551`).
3. Weights multiply query scales in FP16 (`service_vector:291-308`).
4. Cube reduces the FP16 products across heads into FP32 (`service_cube:498-512`).
5. FP16 key scale casts to FP32 and multiplies the result (`service_vector:376-382`).

FP32 dequantized QK or a different head reduction is not this operation. No
existing exposed A3 callable supplies the complete narrow scorer and matching
tie semantics. No rescore implementation is included.

## Other required execution contracts

- Separate logical position L+depth[v] from physical slot L+v. Current
  CannModel._metadata writes one arange to both. MLA/indexer already consume
  cos/sin separately from slots, so their projection/cache primitives are reusable.
- Select legal prefix/ancestor IDs BEFORE topK. Existing shared_fia can encode
  those selected physical IDs; its numerical slot threshold alone cannot prove
  ancestry. Producers 0,1,2,6,...,74 refresh; consumers reuse their selection with
  independent Q/KV (`sfa_v1.py:2076-2083`).
- For c emitted outputs, compact exactly c consumed nodes: root plus the first
  c-1 matched children. Leave the last output pending. Snapshot sources before
  overlapping destination writes. One operation over the flat storage can cover
  78 main layers and 21 indexer regions, excluding MTP layer 78.
- Graph replay requires persistent positions, slots, ancestry and prefix-length
  buffers with fresh contents. All 16 ranks must agree on token/parent/depth
  order and the accepted path, including EOS and budget clipping. Current
  flat resolve/truncate and graph metadata checks do not establish these contracts.

The capability gap is exact pre-topK tree selection plus these state contracts.
An attention-mask change alone cannot supply tree execution.
