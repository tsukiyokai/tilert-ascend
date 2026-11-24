# Single-launch shared-indexer attention prototype

`attention_full_kernel.cpp` composes the real Prolog V3, cache writes, sparse
MLA and MLA projection tail in one mixed AscendC launch. CANN 9.2.0 compiles
and links it for `dav-2201`; all three standalone/full Torch schemas can load
in one process. The composed kernel has no device correctness or latency
result yet.

The input is the already normalized, dynamically quantized hidden tensor and
the same weights/cache metadata as [attention.md](attention.md). The output is
BF16 `[M,6144]` before TP reduction. It covers ungated TP16 shared-indexer
attention for `1 <= M <= 32`. Input residual/RMS normalization, input A8,
full-layer index generation, attention gating and TP reduction remain outside
this prototype. The 21 full-indexer layers still require their own indexer
between Prolog and SFA.

The numerical chain is unchanged: INT8 projections and FP32 RMS/quantization
inside vendor Prolog, BF16 query and main cache, vendor BF16 SFA latent,
BF16 value-up result, per-row dynamic A8 and W8 output projection. The output
projection retains optional INT32 accumulation bias or FP32 output bias,
channel-scale then token-scale ordering, and final BF16 rounding. In
particular, the tail sets AIV DEQSCALE to one before INT32-to-FP16 conversion.

`mla_tail_native.h` owns the tail's device functions. The independent
`mla_tail_kernel.cpp` still supplies its original standalone launch and
argument interface. `attention_full_native.h` calls the same `Forward`
function after Prolog/SFA have finished and reset their resources. The full
translation unit selects `ASCENDC_CUBE_ONLY` before all vendor/matmul includes
to retain the tail's Matmul backend. Every resident AIC and AIV participates
in phase barriers. No host CANN operator or nested kernel launch appears in
the device chain.

## Workspace lifetime

The SFA latent occupies a dedicated `M * 4 * 512 * 2` byte slice. The remaining
arena first stores Prolog queries and attention workspace; after SFA's
all-core publication/barrier/reset, the entire arena holds the tail's heads,
value-up output, A8 codes/scales, INT32 accumulators and 16 MiB Matmul system
workspace. The latent remains available until the tail finishes. Output,
weights, cache and serialized tail tiling are outside this arena.

Let `L` be the aligned latent size, `A` the attention scratch size and `T` the
tail scratch size. Separate stages request `L+A+T` bytes; this kernel requests
`L+max(A,T)`. At eight mixed groups and at least four logical cache pages:

| M | Full scratch bytes | Separate scratch plus latent bytes |
| --- | ---: | ---: |
| 1 | 20,226,560 | 37,544,960 |
| 12 | 22,943,744 | 40,262,144 |
| 32 | 29,671,424 | 47,531,008 |

Both paths additionally retain an equal tail tiling tensor, 16,384 bytes for
M<=16 or 32,768 bytes for larger M. These are requested tensor extents per
workspace, excluding allocator rounding and common inputs/output. They are
memory capacity figures, not measured HBM traffic. Query, latent and tail
intermediates still cross GM; reuse alone does not establish a bandwidth or
latency reduction.

## Build and compare

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
OPS_TRANSFORMER_ROOT=/path/to/ops-transformer \
    bash tests/glm52/whole_layer/attention_full_build.sh
bash tests/glm52/mla_tail/build_so.sh \
    tests/glm52/mla_tail/build-composable
```

The full library is
`attention_full_build/whole_layer_full_attention.so`; its provenance manifest
includes the tail headers and host tiling source. The standalone attention
library remains separately built with `attention_build.sh`.

```python
scratch, tail_tiling, blocks, sync_base = (
    torch.ops.tilert.shared_full_attention_prepare(
        token_x, pages, table_pages, requested_blocks,
    )
)
torch.ops.tilert.shared_full_attention_out(
    token_x, token_scale, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
    scale_dq, scale_uq_qr, scale_dkv_kr, gamma_cq, gamma_ckv, sin, cos, slots,
    kv_cache, rope_cache, topk, query_lengths, kv_lengths, page_table,
    value_tiles, output_tiles, output_scale, bias,
    scratch, tail_tiling, output, scale, epsilon_cq, epsilon_ckv,
    blocks, sync_base,
)
```

The binding validates metadata and aliases, enforces the current device's
mixed 1:2 residency limit, refreshes C2C state during preparation, and flushes
Torch's host task queue before a direct launch. It allocates no NPU memory or
reads tensor values in captured execution.

`attention_full_check.py` compares the single launch with separate native
attention and tail launches, including the SFA latent, final BF16 output and
both complete caches. It uses the same changing input/position/page/index
metadata for both paths and exact equality by default. First pass the vendor
comparison in `attention_check.py`; agreement between two native compositions
does not independently establish their correctness against CANN.

```bash
python tests/glm52/whole_layer/attention_full_check.py --widths 1 \
    --replays 1 --no-timing --output /tmp/attention-full-eager.json
python tests/glm52/whole_layer/attention_full_check.py --widths 1 2 12 32 \
    --capture --replays 8 --no-timing --output /tmp/attention-full-replay.json
```

The fixture uses the attention fields plus logical BF16
`value_weight[4,512,256]`, INT8 `output_weight[1024,6144]`, FP32
`output_scale[6144]`, and optional bias. The harness packs the two tail weights
into physical ND tiles. Reports include all three library hashes, build
dependencies, requested workspace bytes and separate timing scopes.
