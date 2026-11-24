# GLM-5.2 indexer contract and native selector

The native stage writes newly encoded keys and calls the real arch22
LightningIndexer device class in one mixed AIC/AIV launch. It supports the
INT8 and BF16 indexer-cache recipes. Q/K projections, LayerNorm, RoPE and
Hadamard encoding are upstream inputs to this stage. On A3 with CANN 9.2.0,
both cache recipes passed ordered-index and complete-cache byte comparisons
at M=1,2,12,32 with four changing graph replays per width and 24 mixed groups.
The fixture contains synthetic wire tensors; full-indexer preprocessing and
model acceptance remain separate checks.

## Model and cache identity

`CannModel.create_config` selects BF16 model tensors,
`attention_config.indexer_kv_dtype="int8"`, `kv_cache_dtype="auto"` and
128-token pages. It receives the checkpoint path from its caller; it has no
default checkpoint. The source contract in `tilert/pd_vllm/ascend_layout.py`
allocates BF16 main latent `[pages,128,1,512]` and RoPE
`[pages,128,1,64]` caches. C8 describes the indexer cache here, not the main
MLA cache.

The 78-layer backbone has 21 full-indexer layers
`0,1,2,6,10,...,74` and 57 shared-indexer layers. The layout's INT8 recipe
contains 16 layers `6,10,...,66`, each with INT8 keys and FP16 scales.
Full-indexer layers `0,1,2,70,74` use BF16 keys without scales. MTP layer 78
has its own BF16 indexer cache and is outside the 78-layer backbone.
The vLLM-Ascend backend derives its per-layer selection from
`quant_description` entries ending in `.indexer.quant_type` or
`.indexer.wq_b.weight`; the global INT8 setting alone does not select every
layer. A restored checkpoint must match this layout before binding caches.

Historical evidence records `/workspace/models/GLM-5.2-w4a8c8` and
`quant_model_description.json` SHA256
`a8242dd24f993b914ce567d6850ac86f044651d098a0b463a83ce63623c51985`.
Those records do not identify a model currently loaded by the component
harness. The current harness uses synthetic wire tensors.

## Required preprocessing boundaries

The local vLLM `deepseek_v2.py::Indexer` replicates all 32 indexer heads on
each TP rank. It does not split them into two heads per rank. The main MLA
path's four local attention heads are a separate geometry.

| Boundary | Tensor contract |
| --- | --- |
| Hidden input | BF16 `[M,6144]`, the same states supplied to Prolog V3 |
| Main Prolog query norm | INT8 `[M,2048]` and FP32 `[M]`, exported with `query_norm_flag=True` |
| Indexer Q projection | W8 logical `[2048,4096]`, per-output FP32 scales; `npu_quant_matmul(..., bias=None, output_dtype=BF16)` |
| Projected Q | BF16 `[M,32,128]` before RoPE |
| Merged K/head-weight projection | BF16 logical weight `[6144,160]`; output BF16 `[M,160]`, split into K `[M,128]` and weights `[M,32]` |
| K normalization | FP32 LayerNorm over 128 values with FP32 gamma/beta `[128]`, epsilon `1e-6`; result rounded back to BF16 |
| Q/K RoPE | GLM interleaved pairs in the first 64 coordinates; BF16 output before subsequent encoding |
| C8 Hadamard | BF16 128-by-128 normalized Hadamard matrix and BF16 matrix-product output |
| C8 dynamic quantization | Independent per-head query and per-key INT8 codes; FP32 dynamic scales explicitly converted to FP16 |
| Selector head weights | FP16 `[M,32]` for C8, BF16 `[M,32]` for BF16 cache |

Q projection consumes the Prolog's exported INT8 codes and FP32 scale
directly. Reconstructing BF16 query-norm values and quantizing again would
introduce an additional boundary. K normalization is LayerNorm, including
mean subtraction and bias; an RMSNorm stage is not equivalent. C8
Hadamard/quantization is absent from the BF16-cache branch. The Ascend
selector call receives the projected head weights directly, without the
additional softmax/head scaling in the upstream GPU Indexer forward.

The Triton RoPE path loads BF16 coordinates and sin/cos into FP32 and stores
BF16 after pair rotation. Exact FMA contraction and the selected fallback
`npu_rotary_mul` path must be bound to the deployed backend before a native
front replaces that path.

Source anchors, relative to the corresponding checkout:

- TileRT: `tilert/models/glm_5_2_ascend/cann_model.py:126`,
  `tilert/pd_vllm/ascend_layout.py:8` and `:92`.
- vLLM: `vllm/model_executor/models/deepseek_v2.py:603` and
  `vllm/model_executor/layers/layernorm.py:305`.
- vLLM-Ascend: `vllm_ascend/ascend_config.py:929` and `:1195`,
  `vllm_ascend/attention/indexer.py:199`, `:244`, `:309`, `:455`,
  `vllm_ascend/device/device_op.py:483`,
  `vllm_ascend/attention/sfa_v1.py:1426` and `:2078`.
- Main Prolog export:
  `attention/mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h:365`
  and `:1386` in ops-transformer.

## Native wire-stage interface

`indexer_prepare(query, pages, table_pages, blocks=0)` returns
`(scratch, blocks, sync_base)`. Zero selects the full physical mixed grid,
`min(AIC, AIV/2)`, matching the stock selector's partition.
Preparation queries the actual AIC/AIV residency bound and refreshes the
current context's sync base. Execution is allocation-free and flushes the
Torch host task queue before the direct launch.

`indexer_out` accepts the following contiguous physical-ND tensors on one
NPU. The INT8/BF16 query dtype selects the kernel instance.

| Argument | Shape and dtype |
| --- | --- |
| query | `[M,32,128]`, INT8 or BF16 |
| new_keys | `[M,128]`, same dtype as query |
| weights | `[M,32]`, FP16 for INT8, BF16 otherwise |
| query_scale | FP16 `[M,32]` for INT8; `None` otherwise |
| new_key_scale | FP16 `[M]` for INT8; `None` otherwise |
| slots | INT64 `[M]`, unique physical cache positions |
| key_cache | `[pages,128,1,128]`, same dtype as query, mutated |
| scale_cache | FP16 `[pages,128]` for INT8, mutated; `None` otherwise |
| query_lengths, key_lengths | INT32 `[1]`, respectively `[M]` and `[prefix+M]` |
| page_table | INT32 `[1,table_pages]`, physical page IDs |
| scratch | UINT8, exactly the size returned by preparation, mutated |
| output | INT32 `[M,1,2048]`, mutated and returned |

The supported geometry is one causal TND request, `1 <= M <= 32`, page size
128 and 32 replicated query heads. Query row `i` may select logical keys
below `prefix+i+1`; unused top-k entries are `-1`. Tree masks and multiple
requests require a separate contract. Slot and page-table values must
address the supplied caches, and query lengths must describe the supplied
rows. The execute binding validates metadata shapes without reading NPU
values back to the CPU during capture. Negative slots are skipped by the
cache writer; the differential harness supplies valid, unique slots.

The native INT8 selector instantiates
`QLIKernel::QLIPreload<QLIType<int8_t,int8_t,int32_t,true,TND,PA_BSND>>`.
The BF16 selector instantiates
`LIKernel::LightningIndexerKernel<LIType<bfloat16_t,bfloat16_t,int32_t,true,TND,PA_BSND,false>>`.
Both are actual `__aicore__` classes from the vendor arch22 headers; there
is no host CANN operator inside the launch.

The C8 arithmetic remains the vendor's
`FP32_MATMUL(FP16(weight*q_scale), FP16(ReLU(INT32(QK)/1024))) * FP32(k_scale)`.
An ideal FP32 score would remove two FP16 rounding boundaries and can
change the selected set. See
[the captured CANN scoring contract](../../../docs/evidence/ascend-decode/c8-arch22/README.md).

Explicit smaller block counts are useful for diagnostics but may change
ordered indices at equal scores. CANN's within-core merge puts the new
S2 chunk before the accumulated list; its cross-core merge visits cores in
ascending order. Stable equal-score sorting therefore depends on partition.
In the M=12, eight-group fixture, row 9 positions 750/751 are
`[7363,4978]` instead of stock `[4978,7363]`. The retained device partial
list stores identical score bits `1049140327` for both keys. The same inputs
with 24 groups match every ordered stock index. Equal selected sets alone
do not establish identical downstream SFA output.

`indexer_scores.py` reads the wire tensors and, when supplied, the saved
native scratch. It accepts retained scores only from records whose live
merge metadata matches the query row, causal length and output offset.
The device report is `indexer-core24.json`; the score fixture and parsed
report are `indexer-m12-score-data-int8-m12-r0.pt` and
`indexer-m12-score-device.json` under the isolated experiment's `a3-84`
validation directory. The tested selector library SHA256 is
`c496d6ff34fd6ad4d37a67530acd0bc79413076c17ecc0efdf859c5aeab345a4`.

## Synchronization and storage

All launched AIC/AIV groups enter a mixed barrier after cache writes and
after selection. `PIPE_ALL` and a DDR barrier precede `SyncAll<false>()`,
and the TPipe arena resets after the barrier. The selector retains its
original Cube/vector handshakes and top-k merge. Its internal AIV-only
barrier uses hardware flag 14; the mixed barrier uses flags 11/12/13.
An early AIC waits on flag 12 while the AIVs finish selection. Builds do
not enable `__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__`.

At 32 query heads, the exact vendor GM views require 205824 bytes per
launched group for C8 and 1312768 bytes per group for BF16. At eight groups
these are 1646592 and 10502144 bytes. The values exclude input caches and
output. They describe allocated workspace, not measured HBM traffic.
Workspace includes all launched groups even when the vendor assigns work
to fewer groups. The cache writer uses a temporary 768-byte UB buffer that
is dead before the selector allocates its UB/L1/L0 arenas. Slot metadata is
copied through MTE2 into UB and fenced before scalar reads, so the new
writer does not depend on scalar-GM cache reuse across graph replays.

## Build and differential

```sh
CANN_ROOT=/opt/ascend/cann-9.2.0 \
OPS_TRANSFORMER_ROOT=/path/to/ops-transformer \
PYTHON_BIN=python3 bash tests/glm52/whole_layer/indexer_build.sh
g++ -std=c++17 -O2 tests/glm52/whole_layer/indexer_contract_test.cpp \
    -o /tmp/indexer-contract-test
/tmp/indexer-contract-test
```

The build uses real CANN bisheng `dav-2201`, installed AscendC and Torch/NPU
headers, and records source/header/library hashes in `provenance.json`.
CANN 9.2 omits legacy `EVENT_ID4/5` aliases; the include scope supplies their
original numeric slots. Vendor device arithmetic is unchanged.

On an authorized idle device, start with:

```sh
python3 tests/glm52/whole_layer/indexer_check.py --widths 1 --replays 4 \
    --no-timing --output /workspace/logs/indexer-m1.json
```

Then compare changing graph replay with `--capture --queued-producer` and
widths `1 2 4 8 12 16 24 31 32`, including several legal block counts.
The harness changes Q/K/scales/weights, sequence length, physical page
mapping and slots; compares every cache byte and every ordered index;
and only records alternating graph event timings after all comparisons
pass. `--queued-producer` places a Torch clone immediately before the
native query consumer without a device synchronization.

## Remaining full-indexer implementation

The preferred route is vendor selector reuse plus a native Q/K front that
preserves the table above. Main Prolog can export the required query-norm
codes/scales; it needs no reconstruction. The full front still needs the
replicated Q projection, merged BF16 K/weight projection, FP32 LayerNorm,
interleaved RoPE, BF16 Hadamard products and exact DynamicQuant.

An existing `IndexerQuantCache` frontend is a second source route, but the
local implementation is under `arch35` and provides no A3 arch22 instance.
A third route is a custom dot/weighted-ReLU/top-k selector; it adds score
rounding, FP32 reduction and tie/order obligations already embodied by the
vendor classes. It is lower priority than composing the existing classes.

This stage has no whole-model accuracy, accepted-token rate or TPOT result.
