# Residual norm and shared-indexer attention in one native launch

`attention_norm_native.h` composes the real AddRMSNormBias/DynamicQuant
stage with Prolog V3, paged SFA and the MLA projection tail. One mixed
AIC/AIV kernel consumes BF16 hidden and residual inputs and returns BF16
attention output before the TP reduction. The caller supplies the top-k
indices reused by a shared-indexer layer. This is an ungated attention
half; full-indexer preprocessing, TP communication and FFN are separate.

The composed source has compiled with CANN 9.2.0. Its device differential
and timing are pending. Existing standalone attention, norm and MLA
libraries remain separate comparison artifacts.

## Interface and lifetimes

`shared_norm_attention_prepare(input, pages, table_pages, blocks=0)`
returns `(scratch, tail_tiling, blocks, sync_base)`. The input is physical-ND
BF16 `[M,6144]`, `1 <= M <= 32`. Preparation selects at most eight mixed
groups by default, checks the device's actual AIC/AIV residency, and
uploads tail Matmul tiling outside graph capture.

`shared_norm_attention_out` receives input and residual BF16 `[M,6144]`,
norm gamma BF16 `[6144]`, optional norm beta BF16 `[6144]`, and an external
retained-residual BF16 output `[M,6144]`. The remaining weights, metadata,
cache shapes and output bias retain the
[full attention contract](attention_full.md). The epsilon for input norm
is explicit and independent of the Prolog query/KV norm epsilons. Cache,
retained residual, scratch and final output must not alias inputs or each
other. The normal Torch stream flush precedes the direct launch.

The exact stage order is:

1. AddRMSNormBias stores the retained residual and BF16 normalized values.
2. DynamicQuant reads those BF16 values and writes INT8 codes plus FP32
   per-row scales, preserving the standalone stage's rounding boundaries.
3. An all-core publication/barrier/TPipe reset precedes Prolog V3.
4. Prolog V3 writes main latent/RoPE caches and query tensors.
5. SFA consumes the supplied shared top-k indices and writes BF16 latent.
6. Value up-projection, A8 dynamic quantization and output projection write
   the BF16 TP partial result.

The norm stage retains its internal mixed barrier; composition adds
`EndPhase` (`PIPE_ALL`, DDR publication, mixed barrier, then Reset) before
Prolog. The additional barrier is present in this prototype's timing
scope. No CPU operator or nested kernel launch appears between stages.

Scratch begins with the INT8 codes and a 512-byte-aligned FP32 scale
region. These survive Prolog and stay available for differential checks.
Normalized BF16 values and rstd temporarily occupy the later full-attention
arena. They are dead when Prolog starts. The retained residual has an
external lifetime for subsequent layer processing.

If `F(M,B,P)` is the existing full-attention scratch size, the composed
scratch is
`Align512(6144*M) + Align512(4*M) + F(M,B,P)`.
The CPU contract verifies that the full-attention arena also fits the
temporary `Align512(12288*M) + Align512(4*M)` norm allocation for every
supported width and tested geometry. For eight groups and at least four
logical pages:

| M | Full-attention scratch bytes | Composed scratch bytes |
| --- | ---: | ---: |
| 1 | 20226560 | 20233216 |
| 12 | 22943744 | 23017984 |
| 32 | 29671424 | 29868544 |

These are requested tensor sizes. They exclude equal tail tiling, inputs,
outputs and caches, and are not measurements of HBM traffic.

## Build and validation

```sh
CANN_ROOT=/opt/ascend/cann-9.2.0 \
OPS_TRANSFORMER_ROOT=/path/to/ops-transformer \
VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
PYTHON_BIN=python3 bash tests/glm52/whole_layer/attention_norm_build.sh
g++ -std=c++17 -O2 tests/glm52/whole_layer/attention_norm_contract_test.cpp \
    -o /tmp/attention-norm-contract
/tmp/attention-norm-contract
```

The independent output directory is `attention_norm_build/`. Norm vendor
headers come from `csrc/moe/add_rms_norm_bias/op_kernel`; the build also
uses the same Prolog/SFA and MLA tail dependencies as full attention.
`provenance.json` records native, norm vendor, attention vendor and library
hashes. The 3072-case CPU contract checks alignment, disjoint live prefixes
and the temporary arena bound; it does not execute NPU arithmetic.

On an authorized idle device:

```sh
python3 tests/glm52/whole_layer/attention_norm_check.py \
    --widths 1 --replays 4 --no-timing \
    --output /workspace/logs/attention-norm-m1.json
```

The independent comparison is standalone `norm_quant_out` followed by
standalone `shared_full_attention_out` with identical weights, inputs,
cache contents and metadata. Both sides include the same queued input
clone. The harness checks codes, scales, retained residual, whole caches
and final projection. The normalized/rstd temporaries cannot be read after
their arena is reused; the standalone norm harness checks those outputs
separately against its selected vendor baseline.

After the eager differential passes, use `--capture` and sweep changing
inputs, cancelling residuals, positions, top-k order, physical page IDs,
widths and legal block counts. Event timing runs only after every replay
passes. Composition acceptance supplements the standalone vendor
comparisons; matching two native variants alone is not a model oracle.
