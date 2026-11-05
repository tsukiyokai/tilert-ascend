# Shared-indexer attention component

`attention_kernel.cpp` provides one mixed AscendC kernel containing the actual
arch22 Prolog V3 and SparseFlashAttention device implementations. It compiles
with CANN 9.2.0 for `dav-2201`. Its Torch binding compiles and registers against
torch/torch_npu 2.12. There is no model hook or accepted attention latency or
end-to-end TPOT result.

The component covers the 57 shared-indexer layers in the 78-layer GLM-5.2
backbone at TP16. A full-indexer layer must produce its own C8 indexer results
between Prolog and SFA; this entry does not replace that computation or change
the layer selection. Input normalization and the hidden-state dynamic A8
quantization precede this component. Its BF16 latent output feeds the MLA
projection tail. Attention gating, output projection and TP reduction follow.

## Interface

All pointers in `whole_layer_attention_launch` refer to the current device.
`Config` contains fixed shape and tiling metadata, passed by value to the
kernel. `DeviceArgs` and `Forward(TPipe&, Config, DeviceArgs)` are the native
interface for a later whole-layer entry.

| Tensor | Logical shape | Storage and dtype |
|---|---|---|
| token_x / token_scale | `[M,6144]` / `[M,1]` | ND INT8 / FP32 |
| weight_dq | `[6144,2048]` | physical NZ INT8 |
| weight_dkv_kr | `[6144,576]` | physical NZ INT8 |
| weight_uq_qr | `[2048,1024]` | physical NZ INT8 |
| scale_dq / scale_dkv_kr / scale_uq_qr | `[1,2048]` / `[1,576]` / `[1,1024]` | ND FP32 |
| weight_uk | `[4,192,512]` | ND BF16 |
| gamma_cq / gamma_ckv | `[2048]` / `[512]` | ND BF16 |
| sin / cos | `[M,64]` | ND BF16, vendor half-basis rotary convention |
| slots | `[M]` | ND INT64 physical cache-row IDs |
| kv_cache / rope_cache | `[pages,128,1,512]` / `[pages,128,1,64]` | contiguous ND BF16, mutated |
| topk | `[M,1,2048]` | ND INT32 logical token IDs from the preceding indexer |
| query_lengths / kv_lengths | `[1]` / `[1]` | ND INT32 device values |
| page_table | `[1,table_pages]` | ND INT32 physical page IDs |
| output | `[M,4,512]` | ND BF16, mutated |
| scratch | `[Config.bytes]` | ND bytes, mutated |

`M` is in `[1,32]`. The caller must validate physical formats, pointer ranges,
non-overlap, matching devices, finite positive scale/epsilons, valid page and
slot IDs, and `query_lengths[0] == M`. KV length must fit `table_pages * 128`.
No host tensor-value read belongs in captured execution. Changing input,
positions, slots, sequence lengths and top-k values must be read from the live
device buffers on each replay. A captured graph owns its scratch and fixed
shape configuration until its final replay completes.

The host must launch `blocks <= min(physical_AIC, physical_AIV / 2)` under a
mixed 1:2 schedule and set the current context's C2C synchronization base.
`MakeConfig` assumes these checks have already passed; it is not a validating
public binding. All launched AICs and AIVs must enter `Forward`.

`attention_binding.cpp` supplies the validating standalone interface:

```python
scratch, blocks, sync_base = torch.ops.tilert.shared_attention_prepare(
    token_x, pages, table_pages, requested_blocks,
)
torch.ops.tilert.shared_attention_out(
    token_x, token_scale, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
    scale_dq, scale_uq_qr, scale_dkv_kr, gamma_cq, gamma_ckv, sin, cos, slots,
    kv_cache, rope_cache, topk, query_lengths, kv_lengths, page_table,
    scratch, output, scale, epsilon_cq, epsilon_ckv, blocks, sync_base,
)
```

Preparation allocates scratch and refreshes the current context's C2C address.
The execution binding validates shapes, dtypes, physical formats, aliases and
device identity without reading tensor values or allocating NPU memory. It
uses `NPUStream::stream()` to submit queued Torch producers before the direct
AscendC launch; this flushes the host queue without a device synchronization.

## Numerical and synchronization contract

Prolog is instantiated with INT8 hidden and projection weights, BF16 main KV
cache, PA_BSND, RoPE enabled, no dequant/group optimization, and
`queryNormFlag=0`. It retains the vendor's FP32 RMS and query requantization,
token/channel scale order, BF16 query/latent boundaries and cache writes.
The main cache is BF16; C8 in the model's quantization name also refers to the
separate indexer cache and does not authorize changing this cache to INT8.

SFA uses BF16 query/key/value/output, TND query, PA_BSND cache, V-template,
`sparse_mode=3`, `attention_mode=2`, sparse block size 1, and 2048 selected
entries. Both `pre_tokens` and `next_tokens` retain the vendor's INT64_MAX
attribute contract. Right-down causal masking derives from the live query and
KV lengths. There is no flash-decode split-KV specialization in this entry.

Prolog query and query-RoPE outputs occupy separate persistent scratch slices.
Prolog and SFA temporary GM arenas overlap only across a phase boundary.
Each vendor object ends before PIPE_ALL, DDR publication, all-core
`SyncAll<false>` and `TPipe::Reset`. The same boundary follows SFA for the next
layer stage. Prolog keeps its original 192 KiB UB allocation. No vendor UB
overlay or numerical approximation is applied.

This phase composition still requires device validation of cache visibility,
cross-core flag drain, repeated execution and capture replay. The current SFA
specialization assigns query rows to cores and can leave cores idle at small
M; fewer launches alone do not establish a speedup.

## Build and verification

The source dependency is the local CANN `ops-transformer` tree. Required
headers are `attention/mla_prolog/op_kernel/arch22/*`, its parent tiling and
template-key headers, and
`attention/sparse_flash_attention/op_kernel/arch22/*`. The build includes the
CANN AscendC basic/advanced API paths and compiles the native kernel directly;
it does not call a host CANN attention operator.

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
OPS_TRANSFORMER_ROOT=/path/to/ops-transformer \
    bash tests/glm52/whole_layer/attention_build.sh
g++ -std=c++17 -O2 tests/glm52/whole_layer/attention_contract_test.cpp \
    -o /tmp/attention_contract_test
/tmp/attention_contract_test
```

`CANN_ROOT` takes precedence over `ASCEND_HOME_PATH`; `PYTHON_BIN` takes
precedence over `PYTHON`. The output is `attention_build/whole_layer_attention.so`,
its object files and `provenance.json`, recording source/dependency/artifact hashes plus
CANN version files. The source bypasses the unused vendor host tiling-key
registries, which otherwise define conflicting translation-unit globals. It
also supplies the original hardware event numbers 4–7 for CANN 9.2, whose
headers no longer export those symbolic aliases. Neither change alters a
vendor device operation.

`ATTENTION_KERNEL_ONLY=1` builds the native object without Torch. The CPU check
covers 4,608 shape/core/cache configurations and checks that the query/rope
slices survive both phases and all vendor workspace views fit.

`attention_check.py` compares query, query-RoPE, both full caches and the
attention result with the same-input `npu_mla_prolog_v3` then
`npu_sparse_flash_attention` chain. Exact equality is the default. The harness
changes input codes/scales, positions, physical slots, page mappings, selected
logical IDs and KV lengths. Start with eager execution, then test capture:

```bash
python tests/glm52/whole_layer/attention_check.py --widths 1 --replays 1 \
    --no-timing --output /tmp/attention-eager.json
python tests/glm52/whole_layer/attention_check.py --widths 1 2 12 32 \
    --capture --replays 8 --no-timing --output /tmp/attention-replay.json
```

Timing is enabled only with capture, after every numerical check passes, and
uses alternating native/vendor event measurements. It excludes input norm,
hidden-state A8, the indexer, projection tail and TP reduction. Reports bind
results to library and dependency hashes, CANN versions, device and fixture.
`--fixture` accepts CPU tensors named as in the interface table plus BF16
`hidden[M,6144]`, in logical ND layouts; the harness performs weight NZ packing.
