# Vendor attention integration boundary

[The TorchNPU 2.12 C8 graph gate](vendor-hook-runtime212-layer6-int8-graph-v3-a3-194/acceptance-summary.json)
and [BF16 graph gate](vendor-hook-runtime212-layer6-bf16-graph-v3-a3-194/acceptance-summary.json)
pass the real layer-6 outer attention call through the official
`ACLGraphWrapper` at M=1/2/12/32, with eight changing inputs per width.
Together they execute 16 captures and 128 arm replays, with all 704 complete
output, ordered top-k and cache byte comparisons equal to the eager oracle.
All sixteen Gloo participants exit successfully; only rank 0 initializes an
NPU. The indexed library is `e957650c7def45dff8b918b74c987a8ce44eb0b2cca816e7246cbbff8a52378e`;
the shared library is `64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c`.
This gate uses synthetic weights for one local TP16 attention shard. It does
not execute the decoder FFN or either TP16 reduction. The private runtime
retains the official sources except for the recorded AST-equivalent Python
3.11 syntax adjustment.

## Earlier numerical isolation

[Both C8 and BF16 retained-stage diagnostics](vendor-hook-index-diagnostic-v13-summary.json)
reproduce the complete M32 input-5 failures on a3-194. Their first differing
boundary is the same four of 5,120 WK BF16 projection elements: `[0,29]`,
`[6,153]`, `[19,51]` and `[31,25]`. The `[6,153]` value becomes head weight
`[6,25]`; it explains the row-6 top-k differences without a query-projection
difference. All query stages and both arms' input/weight bytes agree.
Each arm's RoPE matches the FP32 oracle using its own BF16 input. The
[BF16 report](vendor-hook-index-diagnostic-bf16-v13-a3-194/outer-index-diagnostic.json)
and [C8 report](vendor-hook-index-diagnostic-int8-v13-a3-194/outer-index-diagnostic.json)
retain these producer boundaries; both complete index caches differ at two
elements. Final O output and both main caches remain byte-identical, but
the outer-forward gates fail. All fifteen helper ranks retain
`npu_initialized_after=false`; the launchers exit nonzero and helper exit
codes vary after failure cancellation.

The native WK BF16 output equals the RNE cast of its captured FP32 accumulator
at every element. The auxiliary FP64 dot rounds to the vendor result at three
of the four differing coordinates and to the native result at the fourth.
Matching this production projection therefore requires its accumulation and
rounding contract; greater dot-product precision alone is insufficient.

[The captured-WK stock probe](captured-wk-v2-a3-194/summary.json) completes
seven BF16 arms with three repetitions each. All four full-N160 variants
match production on all 5,120 elements; all three five-call N32 variants
match the captured native projection. NZ/ND weights and transpose storage
do not change these groups. The first arm is the original `F.linear` call.
This identifies projection geometry as a reproducible trigger on this input,
without identifying the selected CANN K-reduction tiling. The first probe's
FP32-output request reaches an unsupported PrivateUse1 fallback; no FP32
vendor accumulator observation is inferred from that failed call.

[The independent native geometry probe](captured-wk-native-v1-a3-194/summary.json)
rejects full N160 as a sufficient fix: N32 and N160 both reproduce all 5,120
captured native FP32 accumulators across three repetitions. N160 changes
baseK from 512 to 96, stepKa/stepKb from 1/12 to 64/2 and iteration order from
0 to 1, with no accumulator-byte change. Its BF16 cast retains the same four
production differences. At that stage, the remaining gap was the stock projection's actual
accumulation contract, beyond these native tiling choices.

[The TorchNPU 2.12 FP32 discriminator](captured-wk-fp32-v1-a3-194/summary.json)
completes all 27 cases and preserves every 2.10 BF16 control group. Both full
N160 and split N32 FP32 outputs equal all 5,120 native accumulators in three
repetitions. Casting the full-N160 FP32 result to BF16 retains the same four
differences from direct BF16 output; the split-N32 results agree. This isolates
an output-dtype-dependent result, without proving whether it comes from a
different reduction route or the final output path.

[Installed dispatch, schema and instruction excerpts](captured-wk-dispatch-source-audit/dispatch-schema.json)
identify the BF16 path as MatMulV2 key 230097, with 152-byte tiling, no
workspace and 20/4 cores for N160/N32. FP32 selects MatMulV3 key 65536, with
280-byte tiling and 24 cores. Both BF16 tilings use `k_dim=1`, K256 L1 segments
and K128 Mmad; `datatype_bf16=1` selects the observed `MMAD.bf162f32` instruction.
Full N160 shifts the K-segment order by N-tile index; N32 disables that shift
because its largest core dimension is two. All four differing coordinates
belong to odd N tiles, whose K256 segments run backward from indices
23, 15, 3 and 23 respectively. These are source and binary facts for the
recorded tilings. [The independent K-order intervention](captured-wk-kshift-v1-a3-194/summary.json)
retains one L0C across all 48 Mmad operations. All three natural-order calls
reproduce every captured native FP32 accumulator and the four BF16 differences;
all three shifted-order calls reproduce all 5,120 vendor BF16 outputs. Both
arms' direct BF16 outputs equal their own FP32 RNE casts. The captured
projection's discrepancy is therefore resolved by this K-order change.
The later complete M32 attention and official graph gates above validate
the integrated K-order change.

[Official ACLGraphWrapper replay](vendor-hook-graph-v12-summary.json) passes
C8 and BF16 at M=1/2/12 with eight changing inputs per width, plus a C8 M12
two-input pilot. Fourteen actual graph captures and 100 arm replays produce
552 passing byte comparisons. Each width has distinct nonzero output hashes,
changing GPU sequence lengths and independent page plans. The native reserve
belongs to the actual inner capture stream, which differs from the outer
worker stream. This covers a synthetic TP16 local attention shard before TP
communication. This earlier gate predates the complete M32 acceptance above;
full-model execution remains untested.

[The retained-stage diagnostic on a3-194](vendor-hook-forward-diagnostic-v10-a3-194/outer-forward-diagnostic.json)
reproduces the original two BF16 output differences, including both complete
output hashes. All thirteen upstream checks are byte-identical: main query,
RoPE, normalized query and scale, SFA output, value-up output, O quantization
codes and scales, weights, and both independent INT32 matrix-product oracles.
For all 73,728 output elements, the vendor result equals
`BF16(FP32(acc * FP32(channel_scale * token_scale)))`; the native result equals
`BF16(FP32(FP32(acc * channel_scale) * token_scale))`.
The [two finite witnesses](vendor-hook-o-cpu-witness-v1.json) record the inputs
and every FP32 intermediate bit pattern. At `[0,4423]`, the two final FP32
values are `0x3b637fff` and `0x3b638000`; at `[5,2813]`, they are `0xbace8000`
and `0xbace8001`. Each pair crosses a BF16 rounding boundary.
The [summary](vendor-hook-forward-diagnostic-v10-a3-194-summary.json)
records the exact v9 reproduction and all fifteen helper ranks retaining
`npu_initialized_after=false`. The independent combined-scale O candidate
removes this earlier M12 output failure. Its wider runs expose the WK failure
above.

[The wider C8 outer-forward run](vendor-hook-forward-c8-v9/outer-forward.json)
passes eight changing inputs each at M=1 and M=2, then M=12 input 0.
M=12 input 1 fails at two of 73,728 output BF16 elements; complete top-k,
main caches, indexer keys and scale cache remain byte-identical. The
[CPU failure record](vendor-hook-forward-c8-v9-failure.json) gives both
coordinates and bit patterns, the frozen source identities and the dump hash.
The largest difference is `1.52587890625e-05`. This run stops before M=32.
That original dump contains final outputs and caches; the subsequent diagnostic
above retains the intermediate producer boundaries.

[The same-image CPU check](vendor-hook-diagnostic-cpu-v10.json) independently
regenerates the failed M=12 input, all metadata and all initial cache bytes
exactly without preceding M=1/2 calls. It uses the deployed Torch 2.10
AArch64 image with no NPU devices exposed and leaves NPU initialization false.
Its three diagnostic contracts also pass. The
[frozen stage diagnostic](vendor-hook-diagnostic-v10-manifest.json) retains
actual producer tensors and decodes existing native scratch after the outer
call. Its first admission on a3-162 was refused because all dies had become
occupied. The identical frozen sources, libraries, CANN package and Torch 2.10
image subsequently run on a3-194 with the result above. Retaining references
extends tensor lifetimes; the exact original output hashes still reproduce.

The [captured-O discriminator](vendor-hook-o-probe-v1-manifest.json) preserves
the actual INT8 codes, NZ weight, FP32 token scales, absent bias and BF16 output.
It changes only the channel-scale descriptor between original BF16 and its
exact FP32 widening. This separates scale dtype from arithmetic order without
another full attention call. [Both runtime versions](captured-o-version-scale-summary.json)
match the combined-scale formula and the original vendor output on all 73,728
elements: TorchNPU 2.10 and 2.12, each with both scale dtypes and three alternating
repeats. The captured case supports neither a dtype-specific nor a
2.10-specific arithmetic branch. The first 2.10 container stops with exit 137
and three recorded OOM events before completing any QMM case. Raising only its
memory limit from 4 GiB to 12 GiB permits the unchanged probe to finish.
The earlier TorchNPU 2.12 standalone BF16-scale
probe used different synthetic inputs; its passing finite sample does not
establish a universal channel-then-token contract.

[The real outer-forward gate](vendor-hook-forward-m1-v9/outer-forward.json)
passes C8 M1 with synthetic weights on one TP16 local shard. It executes
`DeepseekV2MLAAttention.forward` through the actual `mla_forward` custom op
and forward context. The original call completes with zero native-body calls;
the installed call completes with exactly one. Final output, complete ordered
top-k, both main caches, indexer keys and the original four-dimensional scale
cache all have zero differing bytes. All sixteen Gloo participants exit zero;
the fifteen helper ranks leave NPU initialization false.
[The summary](vendor-hook-forward-v9-summary.json) records the six boundaries,
library identities and frozen source manifest. This gate covers one eager
input before TP reduction. The later graph and wider eager gates are recorded
above; this retained v9 report covers only that original eager input.

[The single-device TP16 installation report](vendor-hook-install-v6/rank-00.json)
completes the real worker registration, `DeepseekV2MLAAttention` construction,
official weight postprocessing, cache binding, library loading and native
adapter installation. The local shard has four attention heads and 36,761,856
source parameter bytes, initialized synthetically with seed 41021. All sixteen
real Gloo participants exit successfully; the fifteen helper ranks report no
NPU initialization. Rank 0 reports 69,341,184 allocated bytes and a 95,454,208-byte
peak. The [launcher](vendor_hook_run-v6.sh) and
[harness](vendor_hook_device-install-v6.py) retain the executed source. This
gate does not execute attention forward, TP communication or graph replay.

The preceding [v5 run](vendor-hook-install-v5/rank-00.json) reaches official
weight postprocessing but CANN `InitCannKB` cannot write `/root/Ascend` on the
read-only root filesystem. V6 adds only a 128 MiB tmpfs at that exact path;
the root filesystem remains read-only and the Python harness is unchanged.

[The official SFA adapter build](vendor-sfa-torch210-v1/manifest.json)
registers `_C_ascend::npu_sparse_flash_attention` under Torch 2.10 without
building the other vLLM-Ascend operators. It compiles the unmodified 8237
SFA adapter header, its NPU storage helpers and `EXEC_NPU_CMD` implementation;
the registration block is extracted verbatim from `csrc/torch_binding.cpp`.
The manifest records the original source hashes, every compiler dependency
hash, extraction range, commands and image identity. Library SHA256 is
`956e5382e4629d63e4a2e9f8b9381063c6dd70921cf0b32c879e53fbf939fdad`.
The [CPU API resolution check](vendor-sfa-torch210-v1-api.log) resolves both
SFA entry points to CANN 9.2's transformer opapi library with NPU initialization
still false. This is an independent build of the official SFA adapter,
not a complete vLLM-Ascend extension build.

The retained [outer-forward fixture](vendor_hook_forward-v7.py) uses the real
layer call and forward context. Both arms restore all cache bytes and the
complete persistent top-k before each input. The indexer scale cache has the
runner's `[pages,128,1,1]` shape; the native adapter borrows a two-dimensional
view of that same storage. V6 used `[pages,128,1]`, which installation accepts
but does not exercise the vendor selector's `squeeze(2)` shape contract.
The fixture requires zero native-body calls for the original method and one
for the installed method, then compares output, full caches and top-k bytes.
Its device result is separate from V6's installation-only gate.

The [first outer-forward run](vendor-hook-forward-m1-v7/outer-forward.json)
reaches the original indexer's key RoPE and stops because the fixture omitted
the worker's Triton device-property initialization. No native-body call occurs.
The official 8237 worker calls `init_device_properties_triton()` after
distributed initialization (`worker/worker.py:537–541`). The
[v8 harness](vendor_hook_device-forward-v8.py) invokes that same function after
binding the device and records its actual AIC, vector-core and UB properties.
The [source manifest](vendor-hook-forward-v7-v8-manifest.json) identifies this
single Python change; the launcher, forward fixture and three libraries remain
unchanged.

[V8](vendor-hook-forward-m1-v8/outer-forward.json) completes the original
M1 outer attention call. Its installed arm reaches the native binding and
stops at `indexer_sin`'s physical-format check before kernel submission.
The official initializer reports 24 AIC and 48 vector cores. Its driver
returns `max_shared_mem=1`; the vendor initializer therefore selects its
196,608-byte UB fallback.

The matched Torch 2.10 [format probe](vendor-hook-metadata-format-v1/metadata-format.json)
shows that cloning a four-dimensional ND RoPE tensor produces format 0;
viewing it as `[M,64]` retains that format. The actual indexer builder can
also reach this layout: `attention/indexer.py:909–938` allocates four-dimensional
buffers with `torch.empty`, and speculative/drafting paths copy into them.
For M=1/2/12/32, the adapter's existing `_nd` helper changes format 0 to 2
with unchanged storage address and bytes. It also retags the original
storage aliases. All sixteen captured source updates preserve bytes.
The [v9 adapter](attention_native-v9.py) applies this conversion only to the
main and indexer sine/cosine views. Slot source identities, query/key lengths,
page tables and cache owners retain their existing paths. The
[v9 forward fixture](vendor_hook_forward-v9.py) retains its format-0 input
and records metadata formats before and after each real outer call.

The outer-forward report's `forward_context` identity refers to Python's
`contextmanager` wrapper. A separate [CPU identity check](vendor-hook-forward-context-identity.json)
imports the same pinned function and records `inspect.unwrap(set_forward_context)`
as `vllm/forward_context.py`, SHA256
`8dab5a4dee60e46c4e52821667321c9f05c793578af49cddbdb204cf9324c403`.
That check leaves NPU initialization false; the original reports are unchanged.

[The complete CPU registration report](vendor-hook-cpu-worker-arch-v2.json)
executes the official worker patches and custom-op registration, then constructs
the real TP1 `DeepseekV2MLAAttention` → `AscendMultiHeadLatentAttention` →
`AscendSFAImpl` with `IndexerWrapper`. The 206,288,896-byte parameter set stays
on CPU. Its actual `attn_metadata=None` profiling call clears the supplied
output. Weight postprocessing reaches `npu_format_cast` and stops because that
operator has no CPU implementation. NPU initialization stays false; peak RSS
is 1,634,896 KiB. The retained [script](vendor_hook_cpu-worker-arch-v2.py) and
[log](vendor-hook-cpu-worker-arch-v2.log) bind these results to the executed
source. Real attention computation, TP16 postprocessing and adapter execution
require the device stages below.

This CPU invocation uses Triton's supported
`TRITON_ASCEND_ARCH=Ascend910_9392` target setting. In the installed Triton
Ascend, `backends/ascend/driver.py:166–175` reads that validated setting before
the hardware-query fallback; `utils.py:506–535` lists the supported targets.
The source is unchanged. The private
[dependency manifest](transformers-dependency-manifest.json) records
Transformers 5.19.0, Hugging Face Hub 1.31.0 and Tokenizers 0.23.2, satisfying
the pinned vLLM/vendor requirements, together with wheel hashes.

[The CPU constructor report](vendor-hook-cpu-indexer-v1.json) executes the
official `EngineArgs` configuration, real TP1 Gloo initialization, and
`Indexer` → `IndexerWrapper` → `AscendSFAIndexerBackend` constructors. The
actual `AscendLinearMethod` owns `AscendW8A8DynamicLinearMethod`; its WQ is
INT8 `[4096,2048]`, with BF16 `[4096,1]` channel scales before postprocessing.
The wrapper and backend share the same weight modules and cache owner.
The retained [script](vendor_hook_cpu-indexer-v1.py) and
[log](vendor-hook-cpu-indexer-v1.log) record NPU initialization as false before
and after execution, 10,372,096 parameter bytes, and peak RSS 1,415,900 KiB.
This is the replicated indexer constructor before worker patches or weight
postprocessing; its recorded cache mode is BF16.

The earlier CPU invocation without an explicit Triton target reaches a device query:
`adapt_patch()` imports worker Triton patches, FLA calls Triton's
`get_current_target()`, and the Ascend helper fails at `rtGetSocVersion`
without a mounted device. [The raw report](vendor-hook-cpu-constructors-v3.json)
and [log](vendor-hook-cpu-constructors-v3.log) preserve that failure. Main SFA
construction was therefore outside that earlier run's completed scope.

[The Torch 2.10 rebuild manifest](attention-torch210-manifest.json) records
separate bindings linked against the accepted device and tiling objects,
with both operator families loaded in one CPU process. Their library SHA256s are
`5d5e8ddb6527bcf50ee4e90507723d2b37cd66ea5f7a4734f7ce5995b13e46f7`
(indexed) and
`245ae243c779594664ebe80ce91b5f84bc133457d47da56734261bce2347ee0f`
(shared). The object hashes, frozen source hashes, compiler commands, image ID,
and CANN 9.2 installation records are included. This checks binding linkage
and schema registration under Torch 2.10; the numerical device gates below
remain those of the separate Torch 2.12 binaries.

The isolated [group harness](vendor_hook_device-groups2-v1.py) also passed a
[two-process CPU run](vendor-hook-groups2-v1/summary.json): both real Gloo
participants observed the expected rank sum, then destroyed their groups.
Each reported NPU initialization as false and peak RSS about 1.33 GiB.
It creates actual `GroupCoordinator` objects with
`use_device_communicator=False` and registers them in the recorded vLLM
revision's group fields. This is a local-shard harness, not the standard
sixteen-NPU worker setup: TP communication is excluded, and an unexpected
NPU collective cannot fall back to a simulated sum.

The official vLLM `a98247ab4db686ee03c66d5feb3c761e52a2f8ab` source and
vLLM-Ascend `8237ee6d690d1399bc6eb48ef5977ac2bd470f3f` imported the real
SFA, indexer wrapper, graph wrapper, forward-context and quantization modules
in a CPU-only Python 3.12 container. [The successful raw report](vendor-hook-a982-py312-import-v3.json)
records each imported file and SHA256, the real `NPUPlatform`, and
`torch.npu.is_initialized() == False` before and after the imports.
This import report establishes package importability for the listed modules;
the constructor report above records the additional executed subset.

## Source and environment

vLLM-Ascend's `.github/vllm-main-verified.commit` names
[a98247a](https://github.com/vllm-project/vllm/commit/a98247ab4db686ee03c66d5feb3c761e52a2f8ab).
It was fetched directly from `https://github.com/vllm-project/vllm.git`
with `--no-tags --depth=1` into `/tmp/tilert-vllm-a982.git`, separate from
the existing checkout. Its tree is `7f2a24a827bbbb2f211e02031f9d4d0d556d536c`;
the source archive SHA256 is
`f963ae5acbfe96524d10712ccc8d7f8735796f261a29acb03d13a2cccf15e7a5`.
The vendor Python package came from its exact Git archive and its real
`setup.py build_py` / `egg_info` commands with
`SOC_VERSION=Ascend910_9392 COMPILE_CUSTOM_KERNELS=0`. Isolated `PYTHONPATH`
and the generated entry-point metadata select these sources without replacing
installed packages. No vendor source compatibility patches or PCP stubs were
used.

The successful import used a3-78's existing image
`quay.io/ascend/vllm-ascend:nightly-releases-v0.23.0-a3`, ID
`sha256:9afbed90a0f92e228cc1ba13adf3cee2e89e34f1dd4cad1ed8eb5ce8c091b4af`.
Its packages differ from the Torch 2.12 device-validation environment:

| Package | CPU import image |
| --- | --- |
| Python | 3.12.13 |
| Torch | 2.10.0+cpu |
| Torch NPU | 2.10.0.post2 |
| Transformers | 5.5.4 |
| Triton / Triton Ascend | 3.5.0 / 3.2.1 |
| NumPy | 1.26.4 |
| compressed-tensors | 0.17.0 |

Installed vLLM distribution metadata is `0.23.0+empty`; installed vendor
metadata is `0.23.0rc2.dev1+g5ed283a6c`. The raw report's actual import paths
point to the isolated a982/8237 sources. Successful imports do not imply
that this image satisfies every requirement of those revisions; for example,
8237 declares Torch NPU `2.10.0.post4` and Transformers `5.19.0`.

## Reproduction and failures

Run [the retained script](vendor-hook-a982-import.py) with the exact source
roots in `PYTHONPATH` and `VLLM_PLUGINS=ascend`. It rejects any attempt to
initialize the NPU runtime. The successful Docker run used `--rm`, no
device mounts, `--network none`, `--read-only`, two CPUs, a 4 GiB memory cap,
and `--tmpfs /tmp:rw,exec,nosuid,nodev,size=512m`. Source mounts were read-only;
only the task's evidence output directory was writable. Set
`TRITON_CACHE_DIR=/tmp/triton`, `XDG_CACHE_HOME=/tmp/cache`, and
`ASCEND_PROCESS_LOG_PATH=/tmp/ascend-log`. Importing Triton decorators builds
and loads its real CPU `npu_utils.so` in that temporary cache; it does not run
an NPU kernel.
The constructor runs additionally use `--workdir /tmp` and a writable tmpfs
at `/root/ascend/log`; the earlier constructor attempts
[v1](vendor-hook-cpu-constructors-v1.json) and
[v2](vendor-hook-cpu-constructors-v2.json) preserve the corresponding
read-only filesystem failures.

- [Python 3.11 report](vendor-hook-a982-import.json): config, forward context,
  platform, indexer wrapper, graph wrapper, and W8A8 imported. SFA failed at
  a982 `model_loader/sharded_state_loader.py:214`, whose multiline f-string
  requires Python 3.12 syntax. The project's `pyproject.toml:34` nevertheless
  declares `>=3.11,<3.15`; the vendor declares `>=3.10`. The source was left
  unchanged. This run used Python 3.11.12, Torch 2.12.0+cu130 and Torch NPU 2.12.0.
- [Initial Python 3.12 report](vendor-hook-a982-py312-import.json): Triton tried
  to write `/root/.triton` on the read-only filesystem.
- [Temporary-cache report](vendor-hook-a982-py312-import-v2.json): the temporary
  filesystem lacked executable mappings, so loading its CPU helper failed.
- [Final Python 3.12 report](vendor-hook-a982-py312-import-v3.json): assigning
  the cache to an executable tmpfs cleared both filesystem failures; all
  seven requested modules imported. Their source hashes are identical to
  the preceding runs.

The earlier vLLM 35a49 / vendor 8237 combination is a source-audit snapshot,
not a runnable package pair: 35a49 lacks the vendor's imported
`vllm.v1.attention.ops.pcp` module. The separate native-kernel device reports
remain scoped to their recorded Torch 2.12 environment and binaries.

## Single-device bootstrap

[The two bootstrap reports](vendor-hook-install-v1-v2-summary.json) record
sixteen real Gloo participants completing the rank-sum check, then rank 0
failing at `torch.npu.set_device(0)`: `aclInit` returns 507899 and
`rtGetDeviceCount` reports driver error `0x7020010`. Neither run reaches
worker registration, attention construction, weight postprocessing or adapter
installation. The [v1 launcher](vendor_hook_run-v1.sh) and
[v2 launcher](vendor_hook_run-v2.sh) differ only by unsetting
`ASCEND_DOCKER_RUNTIME` before Python; both execute the same
[harness bytes](vendor_hook_device-install-v1-v2.py) and fail at the same
boundary. Their per-rank JSON and stdout are retained. Separate driver logs
were in temporary container filesystems and were not retained.

[The single-process comparison](runtime-namespace-comparison.json) identifies
the runtime blocker. Before Torch import, `drvGetDevNum` returned 87 and
zero devices. [The matching kernel messages](runtime-single-v1/kernel-uda.log)
report `uda_occupy_dev_by_ns:932` rejecting physical device 4 because another
mount namespace owns it. The installed driver 25.5.5's
[source and hashes](runtime-single-v1/driver-source-evidence.json) define 87 as
`DRV_ERROR_RESOURCE_OCCUPIED` and retain ownership while the namespace root
process remains alive. Stopping the task's otherwise idle main container,
then running the identical probe, changed enumeration to return 0 with one
device and allowed `torch.npu.set_device(0)` on `Ascend910_9392`. The Python
source, recorded environment and loaded-library hashes match between runs.
This establishes runtime binding under Torch NPU 2.10, without executing a
vendor layer or native kernel. The probe's requested persistent user-space
driver-log directory remained empty; the matching kernel log is retained.

The next [sixteen-rank bootstrap](vendor-hook-install-v3-summary.json) passed
NPU binding and reached the official worker registration. It failed when
`patch_qwen3_5` imported `fla_npu.ops.ascendc`, which the base image lacked.
[The dependency manifest](fla-dependency-manifest.json) records the exact A3
wheel and SHA256 pinned by vendor 8237's `Dockerfile.a3`, installed with
`--no-deps --target` in the task directory. The complete installed tree hashes
match between a3-78 and a3-162. Its real
[CPU import](fla-import.json) passed the three required exports without NPU
initialization; this import does not validate FLA operator execution.

[The launcher](../../../../../tests/glm52/whole_layer/vendor_hook_run.sh)
runs [the constructor and installation harness](../../../../../tests/glm52/whole_layer/vendor_hook_device.py)
with sixteen real Gloo participants. Only rank 0 imports the worker patches,
binds the NPU, constructs the TP16 shard, fills synthetic parameters, calls
the official weight postprocessor and cache binder, and installs the actual
native adapter. Its output projection uses `reduce_results=False`; no TP
device collective or model forward is included in this first gate. The other
fifteen ranks reject NPU initialization. The controller records each rank's
result and bounds its own child-process lifetime to 600 seconds.

The launcher uses eight CPUs, a 24 GiB memory cap, read-only source and CANN
mounts, temporary writable caches, and exactly one physical NPU. The
two-process CPU report measured approximately 1.33 GiB RSS per rank; multiplying
that value by sixteen gives a conservative 21.3 GiB sum before rank 0's
additional device bootstrap. The source model configuration and weights are
synthetic. In C8 mode, the quantization description explicitly supplies
`indexer.quant_type=INT8_DYNAMIC`: vendor 8237's per-layer filter does not
select C8 from the WQ `W8A8_DYNAMIC` entry alone.

The fixed image was exported from a3-78 with two compression threads and
transferred to a3-162 through two parallel internal SSH connections. All
thirteen [compressed-part hashes](image-9afbed90-SHA256SUMS)
[matched on the destination](checksum-162.log).
[Docker load](load-162.log) retained the exact
[image ID and 17,226,146,599-byte size](loaded-identity-162.txt).
The task's isolated runtime directory contains the pinned source packages and
Torch 2.10 bindings. It does not replace the existing Torch 2.12 containers.
