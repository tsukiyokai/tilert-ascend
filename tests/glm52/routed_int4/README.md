# TP16 routed INT4 stage

One A3 mixed kernel gathers routed rows, computes W4A8 gate/up, SwiGLU and
requantization, computes W4A8 down, and combines the original top-8 slots.
Routing decisions, shared experts and TP reduction are outside this stage.
The implementation targets CANN 9.2 and `dav-2201`; it is a correctness
candidate, with no device parity or latency result established by host checks.

`routed_experts_prepare(rows, experts=256, blocks=16, weight_layout=0,
from_dispatch=False)` returns `(scratch, sync_base)` outside capture.
Pass both to the matching execution entry on each invocation.
Scratch belongs to the caller and can be shared by sequential layers on one
device and stream. At 256 experts and M=32 it occupies 35030528 bytes
(33.408 MiB). The output aliases `scratch[9]` and is overwritten on reuse.
Keep the tensors, device context and graph alive together. Prepare checks
physical AIC/AIV residency and refreshes the device synchronization address;
execution checks that metadata without device synchronization or allocation.
Weight layout is 0 for ND and 1 for NZ; pass the same layout to execution.
Prepare records weak owners of both tiling tensors, plus M, expert count,
block count, layout and input mode. Execution rejects mismatched preparation metadata
without reading device routing or retaining scratch after its owner releases it.
An application with a reduced stream core allocation must choose a block count
within that allocation.

| Input | Dtype and shape |
| --- | --- |
| rows | INT8 `[M,6144]`, 1 <= M <= 32 |
| gate_up, ND | INT8 `[256,6144,128]` |
| down, ND | INT8 `[256,128,3072]` |
| gate_up, NZ | INT32 logical `[256,6144,32]` |
| down, NZ | INT32 logical `[256,128,768]` |
| gu_bias, gu_scale | FP32 `[256,256]` |
| down_bias, down_scale | FP32 `[256,6144]` |
| token_scale | FP32 `[M]` |
| group_count, group_start | INT32 `[256]` |
| source_row, source_slot | INT32 `[256]` |
| row_weight | FP32 `[256]` |
| output | BF16 `[M,6144]` |

ND weights store nontransposed B[K,N/2], with adjacent output channels in
low/high nibbles. ModelSlim `[E,N/2,K]` needs a transpose of its byte axes.
NZ weights retain the vendor loader's packed INT8 format-29 storage,
`[E,N/64,K/16,16,32]` bytes, exposed through the final INT32 tensor. The
native kernel reads its raw pointer with `MatmulType<GM,NZ,int4b_t,false>`;
both host tilings also specify NZ. For current K and N, the physical byte
offset of B[e,k,n] is
`e*K*N/2 + (((n//64)*(K//16)+k//16)*16+k%16)*32+(n%64)//2`.
The nibble index is `n%2`. All other inputs and scratch have physical ND
storage. The binding checks format, dtype, shape, device, contiguity and
scratch overlap; NZ requires zero storage offset and a complete bank.

The model hook defaults to copied ND weights. To borrow final vendor NZ
weights, set `TILERT_ROUTED_WEIGHT_LAYOUT=nz` and point
`TILERT_ROUTED_LIB_DIR` at `tests/glm52/routed_int4/build-nz`. It retains the
original weight and FP32 bias tensors after vendor conversion. Original
FP32 scales are copied before their vendor UINT64 encoding. Prefill above
32 rows continues through the original vendor method. Borrowing removes
21.09375 GiB of duplicate packed weights across 75 TP16 MoE layers;
the retained FP32 scale copies occupy 468.75 MiB per rank. These are storage
calculations, independent of device latency and whole-model acceptance.

The first M*8 routes form expert-major groups. Counts are nonnegative and at
most M, starts are their exclusive prefix sums, and `(source_row,source_slot)`
is a permutation of `[0,M) x [0,8)`. Top-8 expert IDs must be distinct for each
token. Unused route entries are zero padded and lie outside every group.
The router owns these value invariants; execution never copies them to the
host. To match the vendor BF16 unpermute path, convert probabilities to BF16
and then FP32 before supplying `row_weight`. Counts and scales are reread on
every replay; AIC scalar count reads explicitly invalidate the data cache.

Intermediate storage is compact in route order. An expert with prefix `s`
and `c` routed rows occupies stacked rows `[2*s, 2*(s+c))`, high MSD rows
followed by low rows; its activation scales occupy `[s, s+c)`. Both matmuls
set `SetOrgShape(2*c,N,K)` and `SetSingleShape(2*c,N,K)`. Host tiling covers
the maximum 64 rows, while the device selects the actual extent on each
replay. Nonempty experts are assigned round-robin to mixed-core groups, so
adjacent hot expert IDs can run on different cores. Kernel-wide barriers
still separate stages.

BF16 input quantization remains `npu_moe_init_routing_v2(quant_mode=1)` in
the model adapter. Its scatter index is token-major:
`expanded_row_idx[token * 8 + slot]` gives the sorted row. Selecting
`expanded_row_idx.reshape(M, 8)[:, 0]` restores one INT8 row and scale per
source token for this kernel. This preserves the dispatcher's quantization
instead of substituting standalone `npu_dynamic_quant`; their arithmetic
orders differ in the available CANN source. Restoration and the kernel's
subsequent gather are additional work that must remain in latency comparisons.

The optional `routed_experts_from_dispatch` entry consumes the existing
vendor dispatch outputs directly. Prepare with `from_dispatch=True` and pass
`sorted_rows [M*8,6144]` INT8, `sorted_scales [M*8]` FP32, `inverse [M*8]`
INT32, `counts [256]` INT64 and raw `probabilities [M,8]` FP32, alongside the
same six weight tensors, scratch and launch metadata. The adapter selects
this entry with `TILERT_ROUTED_REUSE_DISPATCH=1`; its library must come from
`build-route`. BF16 inputs use vendor dispatch, while scaled INT8 inputs and
prefill above 32 rows retain the vendor method.

`inverse[token*8+slot]` identifies that original slot's sorted row. Counts
cover all 256 experts, including zeros, and sum to M*8. The source contract
is `moe/moe_init_routing_v3/op_kernel/moe_v3_full_load_dynamic_quant.h:267–279`
in ops-transformer, SHA256
`c76bd820bea4495e3c9db1d8dac733b26f2617e808d4e17cca54cd0101bb0909`:
the copy-out loop tests `curIndex/k_ == row` before scattering to
`expandedRowIdx[curIndex]`. Expert-local row order need not be stable;
the original-slot address determines both probability ownership and the
fixed-order combine. The CPU contract includes reversed rows within every
expert group, empty experts and the competing `[8,M]` interpretation.

The import phase uses 15904 UB bytes per AIV and adds
`5120 + M*(6144+4)` GM bytes, 201856 bytes at M=32. Scratch entries 10–16 hold
restored token rows, token scales, counts, starts, source rows, source slots
and sorted probabilities. AIV0 alone writes metadata, including the
FP32→BF16→FP32 probability boundary. Each AIV restores its own token rows
using inverse slot zero; scalar scales use exact 4-byte DataCopyPad stores
to preserve adjacent tokens. All AICs and AIVs execute PIPE_ALL, DDR publish
and a mixed SyncAll before resetting UB and entering the existing projection
body. The import uses no atomics, new matmul flags or host route reads.
The combined M32 scratch size is 35232384 bytes. The dispatch kernel still
precedes this launch; this entry does not fuse the complete model layer.

The numerical order follows
[reference_weights.py](../../../tilert/models/glm_5_2_ascend/reference_weights.py):
truncate the low 13 FP32 scale bits, round each scaled high/low INT32 dot to
FP16, convert back to FP32, and compute `16*high + low + scale_bias` using
separate additions. `scale_bias` is already scaled; activation scale follows
the addition. Gate/up remain FP32 through SwiGLU. Down results round to BF16
before FP32 probability multiplication and accumulation in original slot
order; the combined output rounds to BF16 once.

Fused GMM1 quantization uses `quantum = amax/127`, then `factor = 1/quantum`,
followed by FP32 multiplication, FP16 rounding and integer nearest-even
rounding. This differs from standalone `npu_dynamic_quant`. The frozen CANN
source used for this contract is
`/tmp/gmm-compact-20261020/production/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h`,
lines 219–251, SHA256
`c8e4840de9dc7d2bc4d92aecee6c22cb7f039bc06844cca1d7ab7339c1337665`;
`grouped_matmul_swiglu_quant_v2_utils.h`, lines 265–272, SHA256
`5b0213cb78202b47d69cd796b46a1a5c0e6eb491ad54941980d19018cb7632ce`.
The latter casts FP32 to FP16 before INT8. The host test distinguishes this
from direct FP32 rounding with `±1.4999` and `±2.5001`. Zero rows produce zero
codes and scales in this implementation. Device comparison must establish
the deployed operator's behavior, including SwiGLU and zero rows.

Build the Torch extension on a machine with the target CANN and torch_npu:

```bash
CANN_ROOT=/opt/ascend/cann-9.2.0 PYTHON_BIN=/path/to/python \
  bash tests/glm52/routed_int4/build_so.sh
```

The default output is `build/routed_int4_torch.so`. `CANN_TARGET_DIR`,
`BISHENG_BIN`, `CXX`, `SITE_PACKAGES` and `TORCH_CXX11_ABI` support explicit
cross-compilation settings; otherwise Torch supplies the C++ ABI. `SANITIZE=1`
builds an instrumented kernel. `build.sh` builds a standalone zero-row and
hot-expert smoke test with the same interface. No build script runs a kernel.
Build the optional NZ-capable library separately to retain the validated ND
artifact; this library contains both layouts and keeps ND as its API default:

```bash
CANN_ROOT=/opt/ascend/cann-9.2.0 \
  bash tests/glm52/routed_int4/build_so.sh tests/glm52/routed_int4/build-nz
```

Build dispatch reuse separately, preserving both earlier libraries:

```bash
CANN_ROOT=/opt/ascend/cann-9.2.0 \
  bash tests/glm52/routed_int4/build_so.sh tests/glm52/routed_int4/build-route
```

Device comparisons on Ascend910_9392, CANN 9.2.0 and torch/torch_npu 2.12.0
matched the stock ND-weight chain numerically at zero tolerance at
M=1,2,4,8,12,16,31,32,
using eight mixed groups and 60 changing-input, changing-route graph replays.
The [raw reports](../../../docs/evidence/ascend-decode/megakernel-device-20261021/README.md)
identify the library and synthetic seed. The deployed vLLM-Ascend NZ-weight
chain, real checkpoint weights and latency remain to be verified.
The optional native NZ path requires its own numerical and replay validation.

The binding uses `NPUStream::stream()` to submit pending Torch tasks before
its direct runtime launch. Disabling this host queue flush allowed the kernel
to read rows/scales before `MoeInitRouting` and `index_select` produced them.
This operation submits the host queue without synchronizing the device.

For the stock CANN comparison, select `--vendor-backend torch-npu
--vendor-layout nd`. The default backend remains the deployed vLLM-Ascend
custom operator with NZ weights. `--no-timing` performs only numerical checks;
`--eager --no-timing` removes graph capture when isolating a failure.
The checker copies GMM1 codes before GMM2's in-place MSD expansion for
intermediate diagnostics, and recaptures the vendor timing graph without
that diagnostic copy.

`--native-layout nz` selects the separate `build-nz` library unless
`--library` overrides it. Compare that native NZ bank against stock ND using
the same logical weights; the checker records both physical formats and
storage sizes. Its comparison banks are separate allocations, whereas the
model adapter borrows the existing vendor bank.

```bash
python tests/glm52/routed_int4/check.py --device npu:0 \
  --native-layout nz --vendor-backend torch-npu --vendor-layout nd \
  --widths 1 --blocks 8 --replays 4 --warmup 1 --no-timing \
  --output /tmp/routed-int4-nz-m1.json
```

Host contract checks require only a C++17 compiler:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  tests/glm52/routed_int4/contract_test.cpp -o /tmp/routed-int4-contract
/tmp/routed-int4-contract
```

When a target-device run is authorized, compare against the deployed fused
GMM1, GMM2 and unpermute on the same die:

```bash
python tests/glm52/routed_int4/check.py --device npu:0 \
  --widths 1 2 8 12 32 --output /tmp/routed-int4-check.json
```

Use `--checkpoint PATH --layer 3 --rank 0` for real TP16 weights, or `--fixture`
for a CPU dictionary containing the six weight/scale/bias tensors above.
Without either, the script generates seeded nonconstant INT4 banks with
consistent scaled corrections. `--vendor-library PATH` loads a specific
vllm-ascend extension; otherwise the installed package enables its custom ops.
Exact equality is the default. Explicit tolerances are recorded in the report.
The comparison starts from BF16 rows on both paths and runs the actual vendor
init-routing quantization. Every replay checks restored input codes and scales,
and records standalone DynamicQuant differences as a separate diagnostic.
Any failed comparison stops timing and reports middle codes, middle scales and
down results separately, all restored to original token/slot order. Graph
replay changes active expert IDs, hot-expert counts, original slots and
activations. Successful cases record every
paired graph-event sample, alternating measurement order, plus input settings,
source/library hashes and runtime identity. Timing includes init-routing on
both paths and source-row restoration on the native path. It excludes route
metadata creation, input copies and weight preparation from both measurements;
it is neither a full dispatch measurement nor model TPOT.

`--native-routing dispatch` selects `build-route` by default and adds a
third captured comparison: the original native chain with its complete
argsort/count/prefix/gather sequence and both source-row index selections.
Changing-route replays compare final outputs, restored input codes/scales,
all counts and prefixes, original-slot ownership and BF16 probabilities.
`full_replacement_timing` records the complete replacement interval from
BF16 vendor dispatch through combine for all three chains. Router/top-k,
shared experts and TP reduction remain outside that interval. Bytewise
metrics compare the original contiguous CPU dtype storage, including signed
zeros; `pass` and `exact` retain their floating-point numerical semantics.

```bash
python tests/glm52/routed_int4/check.py --device npu:0 \
  --native-layout nz --native-routing dispatch \
  --vendor-backend torch-npu --vendor-layout nd \
  --widths 1 12 32 --blocks 16 --replays 4 --warmup 1 --no-timing \
  --output /tmp/routed-int4-dispatch.json
```

`--vendor-backend cann-nz` uses a separate stock CANN reference bridge. It
constructs an ACL INT32 descriptor with logical `[E,K,N/8]` and physical
`[E,N/64,K/16,16,8]` storage over the existing NZ allocation, then calls
`aclnnGroupedMatmulSwigluQuantWeightNzV2` and `aclnnGroupedMatmulWeightNz`.
Encoded scales have ACL UINT64 descriptors; corrections remain FP32. The
bridge neither repacks weights nor mutates Torch storage descriptors, and
contains no device arithmetic. Descriptor lists own their ACL tensors;
Torch owns output/workspace storage on the current capture stream.

This closes the ambiguity of INT8→NZ→INT32 dtype views retaining storage
dimensions in INT8 units. The explicit `weight_dtype` option in the inspected
Torch operator supports floating-point low-bit types, not INT4. The SDK's
INT32-to-INT4 path multiplies the final physical dimension by eight, requiring
the caller's descriptor to use INT32 units first. The SDK A8W4 NZ example
uses the same `[E,N/64,K/16,16,8]` physical shape. This reference is stock
CANN with canonical descriptors; the deployed vLLM custom operator has a
distinct API including `swigluLimit` and still requires its own comparison.

```bash
CANN_ROOT=/opt/ascend/cann-9.2.0 \
  bash tests/glm52/routed_int4/build_nz_reference.sh
python tests/glm52/routed_int4/check.py --device npu:0 \
  --native-layout nz --native-routing dispatch --vendor-backend cann-nz \
  --widths 1 --blocks 8 --replays 4 --warmup 1 --no-timing \
  --output /tmp/routed-int4-canonical-nz.json
```

The bridge defaults to `build-reference/nz_reference.so`; `--vendor-library`
can select a recorded artifact. Reports include its hash, source hashes,
the declared descriptor contract and loaded SDK libraries.

`profile_routed.py` compares the unchanged routed library with a separate
diagnostic compiled from the same kernel body. It records local SYS_CNT
begin, completed and barrier-release timestamps for dispatch restoration,
metadata/split, gate/up matmul, activation, down matmul, route storage and
combine. Completion marks drain PIPE_ALL. The six existing grid barriers
remain; one final diagnostic barrier keeps trace stores out of combine.
Each core owns a 512-byte record. Reports retain raw ticks without assuming
equal clock origins across cores or converting an unverified frequency.

```bash
CANN_ROOT=/opt/ascend/cann-9.2.0 \
  bash tests/glm52/routed_int4/build_profile.sh
python tests/glm52/routed_int4/profile_routed.py --device npu:0 \
  --widths 12 32 --blocks 24 --samples 30 --iterations 30 --timing-pairs 5 \
  --output /tmp/routed-stage-profile.json
```

The script first checks changing input/routes against the retained library,
including raw BF16 output bytes, middle scales and routed BF16 values. Both
captured arms include BF16 vendor dispatch. Separate paired event timings
measure diagnostic overhead; stage ticks are diagnostic, not operator
latency. The report includes the build manifest, library hashes, exact expert
counts and rows assigned to each mixed-core group. Samples and timing reuse
the final replay's routes and weight bank; this is a warm-bank measurement.
`--no-timing` retains numerical checks and stage samples only. The profile
build has a distinct C launcher and does not register Torch operators.

`build_fixpipe.sh` builds an independent FP16-partial variant in
`build-fixpipe`, registered under `tilert_fixpipe`. It keeps the original
`tilert` namespace and earlier libraries available for comparison. Both
projections use `SetQuantVector` with the original FP32 scale bits encoded
in INT64 carriers, matching the vendor UINT64 descriptor. The CUBE output
is FP16; vector recombination loads these separately rounded high and low
partials, then applies the same scaled correction and activation scale.
SwiGLU, quantization, BF16 route outputs and slot-order combine are unchanged.
This variant's scale inputs have shapes `[256,256]` and `[256,6144]`, with
each FP32 bit pattern in the low 32 bits of an INT64 element. Its two partial
scratch tensors are FP16. The binding checks these distinct dtype contracts.

```bash
CANN_ROOT=/opt/ascend/cann-9.2.0 \
  bash tests/glm52/routed_int4/build_fixpipe.sh
python tests/glm52/routed_int4/check.py --device npu:0 \
  --native-layout nz --native-routing dispatch --native-partial fp16 \
  --vendor-backend cann-nz --widths 1 12 32 --blocks 24 \
  --replays 4 --warmup 1 --no-timing \
  --output /tmp/routed-fixpipe-numerics.json
```

For R=8M, H=6144 and J=128, INT32 partials occupy `8R(H+2J)` bytes and
cause `16R(H+2J)` bytes of writes plus reads. FP16 halves both quantities.
At M=32, dispatch scratch decreases from 35232384 to 28678784 bytes, and
partial traffic decreases by 12.5 MiB per call. These counts describe GM
transactions before cache effects. They do not establish a latency benefit;
the separate device comparison must preserve every rounding boundary.

`bench_weight_banks.py --native-partial fp16` measures the FP16 variant against
the canonical NZ reference. `--compare-native` instead times three native
arms: retained `build-route`, a current INT32 `build-control`, and
`build-fixpipe`. Build the control with `build_control.sh`; it uses a separate
Torch namespace, launcher, tiler and device entry. Each arm owns its scratch
and borrows the same immutable NZ banks. Every captured call includes vendor
dispatch. Four changing-route checks compare each arm's last output with the
canonical NZ reference before timing. Six timing pairs cover all three-arm
orders. The independent calls create weight-cache pressure; they do not form
dependent decoder layers. `--fixture` and `--checkpoint` select the same
weight inputs supported by `check.py`.

```bash
python tests/glm52/routed_int4/bench_weight_banks.py --device npu:0 \
  --native-partial fp16 --compare-native --rows 12 --blocks 24 \
  --banks 4 --calls 16 --iterations 10 --timing-pairs 6 \
  --output /tmp/routed-partials-three-arm-m12-bank4.json
```

`build_balanced.sh` builds the independent INT32 `tilert_balanced` variant in
`build-balanced`. `--native-schedule row` selects it in both checkers. Split
and activation assign expert-major route `r` to AIV `r % (2*blocks)`. The
output stage assigns `r` to mixed-core group `r % blocks`, whose two AIVs
retain separate halves of the hidden dimension. This keeps output halves
parallel at M=1 and bounds each worker's row count independently of expert
hotness. CUBE ownership, tilings, compact high/low addresses, arithmetic and
the six mixed-core barriers remain unchanged.

Activation scales use disjoint four-byte writes from aligned UB storage;
workers do not overwrite another route's scale. The INT32 variant keeps the
same scratch shapes and FP32 channel-scale inputs. More workers may load the
same expert's channel scales. A reduction in row imbalance therefore needs
the complete captured-chain comparison, including dispatch and bank reuse.

Combine seeds its FP32 accumulator with original slot 0's weighted route,
then adds slots 1 through 7 in order. If slot 0 has zero probability, the
seed remains positive zero. This preserves CANN's negative-zero output when
all eight weighted routes are negative zero. Initializing with positive zero
and adding slot 0 changes those output bytes. The installed CANN 9.2.0
`moe_token_unpermute/op_kernel/moe_token_unpermute.h` implements these separate
first-slot and subsequent-slot paths in `CalPartOutToken`, `CalFirstToken`
and `CalToken`; its SHA256 is
`d7d7eb7687e2aeab5b2e365909ae983a0522a3bce660adcbc04bba03236878b0`.

```bash
bash tests/glm52/routed_int4/build_balanced.sh
bash tests/glm52/routed_int4/build_control.sh \
  tests/glm52/routed_int4/build-balanced-control
python tests/glm52/routed_int4/check.py --device npu:0 \
  --native-layout nz --native-routing dispatch --native-schedule row \
  --vendor-backend cann-nz --widths 1 12 32 --blocks 24 \
  --replays 4 --warmup 1 --no-timing --output /tmp/routed-balanced-numerics.json
python tests/glm52/routed_int4/bench_weight_banks.py --device npu:0 \
  --native-schedule row --compare-native --rows 32 --blocks 24 \
  --control-library tests/glm52/routed_int4/build-balanced-control/routed_int4_torch.so \
  --banks 4 --calls 16 --iterations 10 --timing-pairs 6 \
  --output /tmp/routed-balanced-three-arm-m32-bank4.json
```

On 2026-10-21, the row schedule passed M=1/12/32 with four changing-input
and changing-route graph replays at 24 blocks. Final BF16 bytes matched the
canonical CANN NZ reference in every case; the raw report retains the
intermediate comparisons and library identities:
[numerical checks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/routed-balanced-numerics.json).
The tested device was Ascend910_9392, with CANN 9.2.0 and torch_npu 2.12.0.

Four distinct NZ banks were shared by all three timing arms. Each graph had
16 independent calls including vendor dispatch; three warmups preceded six
timing samples covering all arm orders, with ten graph replays per sample.
The table gives median microseconds per call. All six samples favored the
row schedule at both widths.

| M | Row schedule | Retained expert schedule | Current expert control |
|---|---:|---:|---:|
| [12](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/routed-balanced-m12-bank4.json) | 176.212 | 181.749 | 180.566 |
| [32](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/routed-balanced-m32-bank4.json) | 292.137 | 334.949 | 335.583 |

The reduction against the current expert control is 2.41% at M=12 and
12.95% at M=32. This measures one routed chain at 24 blocks, before shared
experts and TP reduction. The timing routes contain 81 and 158 active
experts respectively. These independent calls do not measure model TPOT.
The row library SHA256 is
`592ae06565af86fe38c6574e7db8a397c5c04289a5228bf88c3f0401e64f161f`.

`build_balanced_promoted.sh` retains the row schedule but exports the
existing `tilert` ABI in `build-balanced-promoted`. The scheduling macro
`ROUTED_ROW_BALANCED` is independent of `ROUTED_BALANCED_NAMESPACE`, which
only changes exported names. Load the promoted library in a fresh process;
it registers the same schemas as the retained library. The engine adapter
still defaults to 16 blocks. The promoted library
`a70b29c7d713a0f1649ae4cd2f9c97a018beee03c6c569e7ae511b2ac414a52b`
passed M=1/2/12/32 with four changing-input and changing-route replays at
both 16 and 24 blocks through the original ABI:
[16-block checks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/routed-balanced-promoted16.json),
[24-block checks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/routed-balanced-promoted24.json).
The 24-block timing above does not characterize the 16-block configuration.

The slot-zero combine library in `build-balanced-promoted-zero`, SHA256
`34d59546f07bb534a5072ea5fea90c12367c99bf7a4c881490d02b6ec430ace1`,
also passed the original ABI at M=1/2/12/32 and 16/24 blocks with eight
replays. These checks include zero input, alternating extrema, zero slot-0
probability and zero odd-slot probabilities. At zero tolerance, acceptance
requires both numerical equality and equality of the original dtype bytes:
[16-block checks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-balanced-zero-promoted16.json),
[24-block checks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-balanced-zero-promoted24.json).

```bash
bash tests/glm52/routed_int4/build_balanced_promoted.sh
python tests/glm52/routed_int4/check.py --device npu:0 \
  --native-layout nz --native-routing dispatch --native-schedule row \
  --native-api tilert --vendor-backend cann-nz --widths 1 2 12 32 \
  --blocks 16 --replays 4 --warmup 1 --no-timing \
  --output /tmp/routed-balanced-promoted16.json
```

`build_pipeline.sh` builds `build-pipeline/routed_int4_torch.so` with the
independent `tilert_pipeline` namespace and C launcher `routed_shared_launch`.
It combines the row schedule with shared W8 using
`SHARED_W8_GROUPED_QUANT=1`. Shared activations come from a separate DynamicQuant
input; routed activations retain vendor InitRouting quantization. Outputs
remain separate BF16 tensors, preserving the caller's add and scaling order.

`routed_shared_prepare(rows, experts, blocks, weight_layout)` returns the
usual 17 dispatch scratch tensors followed by shared payload, a 1024-byte
tiling tensor, and shared BF16 output. Capacity is 16 for M<=16 and 32
otherwise. Shared payload is 592384 or 1184768 bytes; both branches share
the routed 16 MiB system workspace. The prepared tensors and device context
must outlive every captured replay. A scratch set serves sequential calls
on one stream. Execution validates dtype, shape, storage format, prepared
tiling identity and non-overlap without allocation or device-to-host reads.

`routed_shared_from_dispatch` takes the existing eleven dispatch/weight
inputs, then `shared_args`, scratch, sync_base, experts, blocks and
weight_layout. `shared_args` has six contiguous ND tensors in this order:

| Input | Dtype | Shape |
|---|---|---|
| Shared activation | INT8 | `[M,6144]` |
| Shared token scale | FP32 | `[M]` |
| Gate/up tiles | INT8 | `[24,1024,64]` |
| Gate/up channel scale | FP32 | `[256]` |
| Down tiles | INT8 | `[24,128,256]` |
| Down channel scale | FP32 | `[6144]` |

The returned routed and shared outputs alias scratch items 9 and 19.
Gate/up tiles are ordered by six K partitions, then four N partitions.
Down tiles are ordered by the 24 output-column partitions. All weights and
scales are borrowed from the caller.

The six existing mixed-core barriers publish both branches' preceding
work. Every launched AIC and AIV reaches them in the same order:

| Phase | AIC | AIV | Completion boundary |
|---|---|---|---|
| 0 | Wait | Shared input staging, then dispatch restoration | Barrier 0 and pipe reset |
| 1 | Shared gate/up | Routed metadata and split | Barrier 1 |
| 2 | Routed gate/up | Shared activation and quantization | Barrier 2 |
| 3 | Shared down | Routed activation and quantization | Barrier 3 |
| 4 | Routed down | Shared output dequantization | Barrier 4 |
| 5 | Wait | Routed BF16 route storage | Barrier 5 |
| 6 | Wait | Routed slot-order combine | Kernel completion |

Phase 0 staging buffers leave scope before restoration resets the pipe;
shared resident buffers are initialized after that reset. After phase 0,
only AIC resets its pipe, between ended matrix objects. AIV keeps routed
and shared buffers live through combine. Peak resident UB is 112352 bytes:
81952 routed, 6592 shared, and 23808 for the shared epilogue. No CUBE
projections execute concurrently, so one system workspace suffices.

`check_pipeline.py` compares interleaved execution, separate native stages
and the canonical reference from the same BF16 input. It checks both final
outputs and intermediate boundaries, including independent input quantizers.
Every timing arm includes dispatch and shared input quantization. This
comparison ends at the two independent BF16 outputs, before their add,
output scaling and TP reduction.

The `build-pipeline-zero` library, SHA256
`d311bcffc7b64ccc102537ec8743f9accc8239edd261d93e0cff67a2165e165a`,
passed M=1/2/12/32 at 16 and 24 blocks on Ascend910_9392 with CANN 9.2.0
and torch_npu 2.12.0. All 34 production and intermediate comparisons matched
the reference bytes over eight changing-input and changing-route replays,
including the zero cases and queued producers:
[M=1/2, 16 blocks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-shared-zero-m1-16.json),
[M=12/32, 16 blocks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-shared-zero-wide16.json),
[all widths, 24 blocks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-shared-zero-replay24.json).
At M=32, scratch is 36811392 bytes for interleaved execution and 53588608
bytes for the separate native stages, a difference of one 16 MiB system
workspace.

Timing uses four distinct banks, 16 independent calls per captured graph,
24 blocks and ten graph replays per sample. Six samples cover every arm
order; all six favor interleaved execution at both widths. Dispatch and
the separate shared DynamicQuant are included in each arm. The table gives
median microseconds per call, with both final outputs checked by raw bytes.

| M | Interleaved | Separate native | Canonical reference |
|---|---:|---:|---:|
| [12](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-shared-zero-m12-bank4.json) | 187.165 | 219.243 | 300.438 |
| [32](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-shared-zero-m32-bank4.json) | 312.363 | 347.326 | 427.039 |

Interleaving reduces latency against separate native stages by 14.63% at
M=12 and 10.07% at M=32. The timed routes contain 81 and 158 active experts,
respectively. This operator-chain result does not measure model TPOT.

```bash
bash tests/glm52/routed_int4/build_pipeline.sh
python tests/glm52/routed_int4/check_pipeline.py --device npu:0 \
  --widths 1 2 12 32 --blocks 16 --replays 8 --no-timing \
  --output /tmp/routed-shared-pipeline16.json
```

`build_gu_mdl.sh` builds `build-gu-mdl` in the independent `tilert_mdl`
namespace. Only gate/up uses `CFG_MDL`; down retains `CFG_NORM`, and the
row schedule, integer products, quantization and combine remain unchanged.
The existing GU tiling has base M/N/K=64/256/256, stepKa=24 and stepKb=4.
MDL retains the complete A K dimension in L1 and groups B loads into six
1024-row K panels. The 24 MMAD K steps and the weight byte count remain
unchanged. CANN's default Norm scheduler instead loads one 256-row K panel
per iteration.

The host tiling dump and build manifest verify L1=458752 bytes, L0C=65536
bytes and no Matmul UB allocation. SDK pools reserve 65536 bytes each for
L0A and L0B; their effective double-buffered payloads are 16384 and 65536
bytes. No other code allocates L0 during this projection. Runtime
`SetSingleShape(2*count, ...)` supplies the MMAD M tail in both schedulers;
the tiling's baseM=64 does not force every expert to execute 64 rows.

Library SHA256
`7f48678afcd186f3bb286d7230abfe8c36c9685f49520fb23edc8354e9564a7a`
passed M=1/2/12/32 with eight graph replays at 24 blocks, including zero
input and zero probabilities:
[numerical checks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-gu-mdl-numerics.json).
The four-bank, 16-call, six-order comparison against the retained row
schedule measured 172.010 versus 173.979 us at
[M=12](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-gu-mdl-m12-bank4.json)
and 288.797 versus 289.255 us at
[M=32](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-gu-mdl-m32-bank4.json).
All six M=12 samples favored MDL; one M=32 sample
reversed direction. MDL remains an independent candidate.

```bash
bash tests/glm52/routed_int4/build_gu_mdl.sh
python tests/glm52/routed_int4/check.py --device npu:0 \
  --native-layout nz --native-routing dispatch --native-schedule row \
  --native-cube gu-mdl --vendor-backend cann-nz --widths 1 2 12 32 \
  --blocks 24 --replays 8 --warmup 1 --no-timing \
  --output /tmp/routed-gu-mdl-numerics.json
python tests/glm52/routed_int4/bench_weight_banks.py --device npu:0 \
  --native-schedule row --native-cube gu-mdl --compare-balanced \
  --rows 12 --blocks 24 --banks 4 --calls 16 --iterations 10 --timing-pairs 6 \
  --output /tmp/routed-gu-mdl-m12-bank4.json
```

`build_gu_pad16.sh` builds a separate Norm-scheduler candidate in
`build-gu-pad16`, namespace `tilert_pad16`. For R=8*M routes, GU packed A
and INT32 C each reserve 16*R rows. An expert with route prefix s and count
c starts at row 16*s; high MSD rows occupy [0,c) and low rows [c,2*c).
GU uses M'=ceil(2*c/16)*16. Since M'<=16*c, its padded writes end before
the next expert's allocation. Activation reads only the first 2*c products.
Down and original-slot outputs retain their compact layouts.

Prepare initializes all GU A bytes once. A graph replay may leave older
integer values in padding; matrix rows are independent and those outputs
are never consumed. The variant increases shared scratch by 14 MiB at
M=32. It retains the same L1/L0 limits and maximum matrix M=64. The change
targets CANN 9.2's per-MMAD PIPE_M branch for `(m/16)*(n/16)<10` by using
a valid larger matrix shape; it also increases GU A reads and C writes.
Device comparison must decide their combined cost.

`check.py --native-cube gu-pad16 --native-schedule row --native-routing
dispatch` loads the retained balanced library as an independent reference.
It compares valid packed A, GU INT32 and down INT32 rows in original-slot
order, then checks middle A8, scales, BF16 routes and final outputs against
the canonical CANN chain. Padding bytes are excluded from these comparisons.

Library SHA256
`59c0df20b3c05953657ee900d89df633e6790ca41f3f63f829603d2217f731aa`
passed M=1/2/12/32 with eight graph replays at 24 blocks. All seven
production boundaries and final output bytes matched, including zero input,
signed zeros and zero probabilities:
[numerical checks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-gu-pad16-numerics.json).
The build compiles from its frozen `source/` directory and verifies all
15 source hashes before and after compilation. Its manifest retains SDK
header hashes and both weight-layout tilings.

The four-bank, 16-call comparison used six arm orders and 30 graph replays
per sample. Pad16 was slower in every order: 175.198 versus 174.638 us at
[M=12](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-gu-pad16-m12-bank4.json),
and 299.996 versus 291.783 us at
[M=32](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-gu-pad16-m32-bank4.json).
The comparison library uses the compact row schedule and the same NZ banks.
Pad16 remains a separate numerical and performance experiment.

`build_input_front.sh` builds an independent BF16 input-front candidate in
`build-input-front`, namespace `tilert_front`. It replaces InitRouting and
the native restoration phase with one phase inside the balanced kernel.
GU, SwiGLU quantization, down projection, original-slot stores and combine
retain the compact INT32 implementation. Prepare still takes an INT8
`[M,6144]` sample with `from_dispatch=True`; execution takes BF16 hidden
states, INT32 `[M,8]` expert IDs and FP32 `[M,8]` probabilities. Weights,
the 17 scratch tensors and their ownership rules remain unchanged.

For original route `o=8*token+slot`, let `C[e]` count its selected expert
and `S[e]` be the exclusive prefix of those counts. The front assigns
`p(o)=S[e]+count(j<o with expert[j]=e)` and writes the original token, slot
and `FP32(BF16(probability[o]))` at `p(o)`. All 256 expert counts and starts
are written, including empty experts; unused route metadata is zero.
Each token must select eight distinct IDs in `[0,256)`, which bounds every
expert count by M<=32. The caller supplies this router contract without
host reads during graph execution.

The CANN dispatcher may use a different order within an expert. Rowwise
integer projections and activation quantization commute with that row
permutation. Both high and low MSD rows carry the same permutation, and
each BF16 route is stored back to its original slot before combine.
Consequently expert-row ordering cannot change the slot-ordered sum,
including the slot-0 seed, skipped zero probabilities and signed zeros.
The front retains the existing compact `[2*S[e],2*(S[e]+C[e]))` MSD
allocation and introduces no matrix padding.

Input quantization follows the installed CANN 9.2 source
`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/`
`moe_init_routing_v3/op_kernel/moe_v3_full_load_dynamic_quant.h`,
SHA256 `97d7a1a73709650aa6a88f2369a709ace98779f0591ec3f7f8c7f7b811494020`.
The NO_SCALE branch casts BF16 to FP32, reduces the absolute maximum,
computes scalar `s=amax/127`, then applies vector `Div(x,s)` followed by
FP32→INT32 RINT, INT32→FP16 with dequant scale 1, and FP16→INT8 TRUNC.
Zero rows follow these same operations. Separate buffers avoid source and
destination overlap for both width-changing casts. The vendor already
quantizes each token once before scattering its routes; this candidate
removes that scatter, external dispatch boundary and native restore.

AIV0 owns metadata. AIV `token%(2*blocks)` owns the original-order A8 row
and an exact four-byte scale store. The phase uses 76320 UB bytes per AIV;
all mixed cores finish PIPE_ALL and DDR synchronization, then the existing
mixed barrier before resetting their pipes and entering the retained body.
The library SHA256 is
`0afd051c2f6088adc04744a3cc0a0d910abc68738aae6069c7cabd1fd636a136`.
CANN 9.2 compilation, linking, C++ geometry checks and CPU schema loading
passed. On Ascend910_9392 with CANN 9.2.0, M=1/2/12/32 at 24 blocks passed
eight changing-input and changing-route replays. All 17 comparisons matched
raw bytes, including zero input, zero probabilities and queued producers:
[numerical checks](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-input-front-numerics.json).
This is the routed-only candidate; it has not been combined with the
shared-expert pipeline.

The same-host three-arm comparison uses four distinct NZ banks, 16 calls
per captured graph, 24 blocks and timed route case 3. Each arm has 30
independent event samples, with 30 graph replays per sample; five complete
six-order cycles balance execution order. Every bank passed all eight
numerical cases, and the final timing outputs matched raw bytes.

| M | Active experts | Front p50/p90, us | Retained dispatch p50/p90, us | CANN p50/p90, us |
|---|---:|---:|---:|---:|
| [12](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-input-front-m12-bank4.json) | 81 | 150.891 / 151.821 | 172.972 / 173.123 | 261.284 / 261.418 |
| [32](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-input-front-m32-bank4.json) | 158 | 269.867 / 270.057 | 285.950 / 286.171 | 353.885 / 354.099 |

The front reduces median complete-chain time by 22.080 us at M=12 and
16.083 us at M=32 against retained dispatch. Its sample standard deviation
is 0.435 and 0.315 us, respectively. These results cover the routed operator
chain, with unchanged downstream projection and combine arithmetic; they
do not measure a full model round or TPOT.

The build saves all 17 own-source files before compiling, verifies the
installed quantization source hash, and records SDK hashes, tilings and
compiler identity. `check_input_front.py` compares independent native,
retained balanced and canonical CANN graphs from the same BF16 input.
It checks A8/scales, metadata, valid high/low integer boundaries, route
outputs, final raw bytes and queued producers over eight changing cases.
Every weight bank has a separate numerical graph and complete boundary
checks; the timing graphs execute the requested bank sequence without
diagnostic copies.
Timing includes each complete BF16-input-to-BF16-output chain and uses at
least 30 event samples per arm over complete six-order cycles. It records
the timed route case and counts, independent bank addresses, p50/p90/p99,
standard deviation and post-timing output checks.

```bash
bash tests/glm52/routed_int4/build_input_front.sh
python tests/glm52/routed_int4/check_input_front.py --device npu:0 \
  --widths 1 2 12 32 --blocks 24 --replays 8 --warmup 1 --no-timing \
  --output /tmp/routed-input-front-numerics.json
python tests/glm52/routed_int4/check_input_front.py --device npu:0 \
  --widths 12 --blocks 24 --banks 4 --calls 16 --timed-route 3 \
  --timing-samples 30 --iterations 30 \
  --output /tmp/routed-input-front-m12-bank4.json
```

`build_pipeline_front.sh` composes that front with the retained shared
interleaving in `build-pipeline-front`, namespace `tilert_pipeline_front`.
`routed_shared_from_bf16` accepts the routed BF16 input, choices and
probabilities plus the original six `shared_args`. The shared branch still
receives its own external DynamicQuant codes and scale; its BF16 source
may differ from the routed source. Both BF16 outputs remain separate.

Phase 0 stages shared A8 into its GM payload, then runs the routed front.
Its temporary UB allocations total 82912 bytes. The existing barrier 0
publishes both sets of GM writes before all pipes reset. Subsequent phases
retain the original 112352-byte maximum AIV allocation, 20 scratch tensors,
six mixed barriers and one shared 16 MiB Matmul system workspace. No
LocalTensor from the temporary phase survives the reset.

Library SHA256
`b0a87a13cf07ce653e6e602c4dfad9eadbbe4ace91cf6f65b0ef673efce74016`
compiled and linked with CANN 9.2, and loaded alongside all retained Torch
namespaces on the host. The build records 28 frozen sources, including the
shared and MLA helper dependencies. M=1/2/12/32 passes eight changing replays
at [16 groups](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-numerics16.json)
and [24 groups](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-numerics24.json).
The actual adapter also passes with original
[FP32](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-adapter-fp32.json)
and [BF16](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-adapter-bf16.json)
shared scales at 24 groups, including INT64 choices, separate input branches
and queued producers. All 37 retained boundaries match raw bytes.

`check_pipeline_front.py` compares the new composition, the immutable
`build-pipeline-zero` interleaved control and canonical CANN. Each numerical
bank uses separate graphs and 37 strict byte comparisons, including both
input branches, original-slot routed integer boundaries, shared split-K
products and queued clones. Shared input uses route-case offset 3, so both
timed inputs are random nonzero values at the default routed case 3.
`--engine-adapter` also checks `run_body`, capture reservation and INT64
top-k IDs converted to INT32 before native launch. At M=12 with 24 core
groups, the adapter selects `tilert_pipeline_front_active`; other shapes
use `tilert_pipeline_front`. `--active-library` supplies the additional
library, and the report identifies the selected namespace and scratch count
for every width. The active table adds a 38th raw-byte boundary.

The paired comparison uses four distinct NZ banks, 16 calls per graph,
24 groups, original BF16 shared scales, and 30 independent event samples
per arm. Each event measures 30 graph replays; the six arm orders repeat
five times. Both inputs are nonzero at routed case 3. Shared DynamicQuant
is inside each arm. Every bank passes all eight numerical replay cases
and all 37 boundaries, and the final timed outputs match bytes.

| M | Front + interleaved p50 / p90 (us) | Dispatch + interleaved p50 / p90 (us) | CANN p50 / p90 (us) |
|---|---:|---:|---:|
| [12](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-m12-bank4.json) | 161.111 / 161.173 | 187.218 / 187.372 | 310.011 / 310.129 |
| [32](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-m32-bank4.json) | 285.147 / 285.280 | 306.638 / 306.882 | 411.279 / 411.430 |

The new composition's standard deviation is 0.271/0.346 us for M=12/32.
These times cover independent routed/shared BF16 inputs through two separate
BF16 outputs, excluding TP reduction and the final add. The standalone
routed input-front measurement of 150.891 us at M=12 covers only the routed
branch; the combined path is 161.111 us in this comparison. Neither is a
complete model-layer or TPOT measurement.

```bash
routed_snapshot=docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-build
bash "$routed_snapshot/source/routed_int4/build_pipeline_front.sh" \
  "$PWD/tests/glm52/routed_int4/build-pipeline-front"
bash tests/glm52/routed_int4/build_pipeline_front_active.sh
python tests/glm52/routed_int4/check_pipeline_front.py --device npu:0 \
  --widths 1 2 12 32 --blocks 24 --replays 8 --warmup 1 --no-timing \
  --output /tmp/routed-pipeline-front-numerics.json
python tests/glm52/routed_int4/check_pipeline_front.py --device npu:0 \
  --engine-adapter --widths 1 2 12 32 --blocks 16 --replays 8 --no-timing \
  --output /tmp/routed-pipeline-front-adapter16.json
```
