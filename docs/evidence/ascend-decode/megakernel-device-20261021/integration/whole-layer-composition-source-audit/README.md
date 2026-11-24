# Whole-layer composition contracts

The first useful integration is the real decoder with its two existing TP
boundaries and the validated attention and FFN implementations. A resident
one-launch layer is a second experiment: the existing one-launch FFN was
slightly slower than three launches at matched addresses. Reducing launch
count alone is therefore not the performance hypothesis.

This audit pins vLLM `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`,
vLLM-Ascend `8237ee6d690d1399bc6eb48ef5977ac2bd470f3f`, CANN 9.2,
indexed-attention DSO `20095457af00d9299fcff884d806b03400669adfe3304750ca9ae97983a40ca8`,
shared-attention DSO `dd66e68b9c8f46c4fdee987d7b7bb29a062e3dcae5481237c45bc21544d346b9`,
and callable-FFN DSO `6dad131d70300e6775aa43f46cf034895d00398a1d805f698b42984c61743d89`.
[Source hashes and retained line-numbered excerpts](sources.json) identify
the implementations independently of the working tree.

The indexed candidate passes the actual M32 outer attention gate in
[BF16 mode](../vendor-hook-wk-kshift-v14-bf16-m32-a3-194/summary.json) and
[C8 mode](../vendor-hook-wk-kshift-v14-int8-m32-a3-194/summary.json), including
the formerly failing replay. Those gates cover a TP16 local attention shard
with synthetic weights, before its TP collective. They do not execute the
decoder's norms or FFN.

## Observable layer contract

For a BF16 MoE backbone layer, let `B` denote BF16 rounding and let
`NormAdd` mean the exact vendor AddRMSNorm operation, including its internal
BF16 stores. The input and return are pairs of BF16 `[M,6144]` tensors:

```text
(x0, r0) = NormAdd(hidden, residual, input_gamma)
(q0, s0) = DynamicQuant(x0)
a_rank = Attention(x0, q0, s0, metadata, caches, ordered_topk)
a = TP_AllReduce(a_rank)
(x1, r1) = NormAdd(a, r0, post_attention_gamma)
(routed, shared) = RoutedAndSharedExperts(x1)
z_rank = B(B(2.5 * routed) + shared)
hidden_next = TP_AllReduce(z_rank)
return hidden_next, r1
```

The first TP result precedes the nonlinear post-attention norm. The second
TP result is returned without adding `r1`; that residual is consumed by the
next layer's input norm. The original first-layer `residual=None` branch is
RMSNorm without addition and needs its own entry. These facts follow from
[the decoder, lines 1367–1423](source/decoder.txt) and
[MoERunner, lines 781–795](source/moe_runner.txt). A local-shard result with
either collective omitted is not this recurrence.

The supported composition domain is TP16, EP1, DP1, PCP1, no sequence
parallelism, BF16 activations and `1 <= M <= 32`. There are 256 local routed
experts with intermediate width 128 per TP shard, eight routes per row, and
one shared expert with local intermediate width 128. The three dense prefix
layers require another FFN body; MTP and its runtime top-k switching remain
separate. The fixed GLM-5.2 pattern has frequency four and offset three:
layer 6 is the first MoE layer with a full indexer, and layer 7 is a MoE
consumer of shared indices. An actual decoder fixture must preserve this
pattern rather than reuse the attention-only fixture's layer 0 or all-full
indexer list. [Configuration and pattern source](../../../../../reference/mla-decode-fusion.md)
records the 21 full and 57 shared backbone layers.

| Seam | Values and ownership |
| --- | --- |
| Input norm to indexed attention | BF16 `x0[M,6144]` remains live through WK; INT8 `q0[M,6144]` and FP32 `s0[M,1]` feed Prolog. Keeping only A8 and scale loses the actual WK input. |
| Prolog to indexer front | Exported INT8 normalized query `[M,2048]` and FP32 token scale remain in the indexed arena. Main Q/rope and both main-cache writes must be published first. |
| Indexer to SFA and later shared layers | Ordered INT32 top-k `[M,1,2048]` is used by this SFA, then copied to the layer-group's persistent owner before scratch reuse. A scratch view must not replace that owner. |
| Attention to TP1 | BF16 partial `[M,6144]`; caches retain their original owners. Main and indexer page tables, actual lengths and live slot sources are independent. |
| TP1 to callable FFN | BF16 reduced attention and retained `r0`, plus post-attention gamma. `ffn_callable_device::Forward` already includes this norm; passing normalized `x1` as its input would normalize twice. |
| Router to routed/shared body | FP32 `[M,256]` logits, ordered INT32 `[M,8]` IDs and FP32 probabilities. The routed front rounds probabilities to BF16 and uses its original InitRouting quantization. Shared DynamicQuant remains independent. |
| FFN to TP2 and next layer | Bridge slot 10 is the BF16 scaled-routed-plus-shared result. Bridge slot 1 is retained `r1`; both survive the collective and return. |

The main cache is BF16 `[pages,128,1,512]` plus BF16 RoPE cache
`[pages,128,1,64]`. C8 refers to the indexer key cache: INT8
`[index_pages,128,1,128]` and FP16 scale cache whose runner owner has shape
`[index_pages,128,1,1]`. BF16 indexer mode stores BF16 keys. Cache writes,
negative-slot behavior, actual GPU lengths and ordered top-k are observable
outputs, even when the final projection happens to match.

## Device phase and storage contract

[Indexed Forward](source/indexed.txt) calls Prolog, front, cache/selection,
SFA and tail in that order. Each completed vendor object leaves scope before
its enclosing `EndPhase`. [EndPhase](source/attention_phases.txt) is
`PIPE_ALL`, DDR publication, mixed `SyncAll<false>`, then `TPipe::Reset`.
The caller initializes SoC state and the sync base once; every resident AIC
and AIV participates in every mixed barrier, including cores without a
matrix or row assignment. Stage-specific task counts do not permit skipping
global barriers. The accepted indexer uses all physical AIC groups to
preserve its actual top-k merge order.

[Callable FFN](source/callable.txt) includes norm, gate/shared quantization,
top-k, the six-barrier routed/shared body, explicit publication after body
return, and the BF16 epilogue. Its return has published output and reset the
pipe. `NoTrace::FinishFinal` alone is empty. Its internal barriers remain
necessary when placed after attention; a wrapper does not establish a new
producer/consumer dependency by itself.

Two storage conditions prevent a naive concatenation:

1. [The shared-index norm wrapper](source/norm_shared_overlay.txt) overlays
   normalized BF16 with the full-attention arena after quantization. That
   lifetime is valid for its shared-index path. Indexed WK still reads BF16
   `x0`; an indexed norm entry needs a disjoint normalized prefix until front
   completion. `r0` remains live until post-attention norm, and `r1` through
   the next layer.
2. CANN 9.2 `SetSysWorkspace` sets the reserved pointer only when it is null.
   A later call does not switch workspaces. In the current indexed entry the
   front selects its 16 MiB area; the tail's call cannot retire that area.
   In standalone callable FFN, the bridge selects its 16 MiB area before the
   body. A combined caller should select one adequate 16 MiB system region
   before either component and keep it alive throughout. The explicit GM
   payloads passed to vendor operations remain separate. Any later overlay
   must exclude the selected system region. The SDK also has
   `SetSysWorkspaceForce`, but introducing a switch is a separate change
   requiring drained objects and a device gate. [SDK and disassembly evidence](../ffn-system-workspace-source-audit/summary.json)
   establishes the distinction.

No UB values can cross a reset. The original norm can consume 195,584 bytes
of AIV UB; communication uses another 20 KiB in its own phase. These cannot
be kept resident together in 192 KiB. The routed/shared body retains its
112,352-byte AIV allocation across its internal phases. Object scopes,
hardware events and selected system workspace must outlive their actual
consumers, rather than the textual wrapper call.

At 24 groups, 8K sequence and 64 pages, the existing indexed scratch sizes
are 108,518,912 bytes at M12/C8, 135,136,256 at M12/BF16, 130,138,624 at
M32/C8 and 156,840,448 at M32/BF16. The M32 callable FFN's 32 distinct
buffers total 55,393,664 bytes. These are allocation sizes, not measured HBM
traffic. A BF16 hidden tensor costs `12,288*M` bytes; persistent top-k costs
`8,192*M`. A first composite keeps component arenas disjoint, plus its live
norm outputs and collective inputs/outputs. A later arena overlay may save
capacity after liveness is proved; it has no demonstrated latency benefit.

## Three ranked routes and decisive gates

### 1. Actual decoder integration with the existing TP boundaries

Construct the official `DeepseekV2DecoderLayer` at layer 6, then layer 7,
with the original config, worker registrations, quantization postload and
cache metadata. Reuse the validated attention adapter and the routed/shared
runner seam first. The original decoder owns both norms; the original
attention O projection and MoERunner retain the two HCCL reduction points.
This closes actual class, routing, residual and collective ownership before
another device scheduler is introduced.

For a later complete-FFN seam, the new call begins at the decoder's
post-attention norm and returns both its local combined result and retained
residual. Calling the complete FFN from the current routed callback would
duplicate the norm/router/epilogue. The active-body optimization requires
its explicit INT32 `[264]` list and 21 body buffers; it cannot silently
replace the 20-buffer callable ABI.

The decisive gate compares actual decoder outputs, retained residual,
ordered top-k and full caches after changing hidden/residual values, GPU
lengths and independent page plans. It then checks the two TP outputs with
rank-dependent inputs. A one-NPU local-shard fixture can validate constructors
and the two cut interfaces; it cannot execute the real TP16 recurrence with
CPU-only Gloo peers. The distributed gate needs the actual TP communicator.
Timing then uses the same graph, weights, geometry, routes and addresses.

This route ranks first because it has a measured local improvement and the
smallest unverified scheduling change. The active complete FFN saves paired
medians of 5.659 and 9.549 us across two matched-address arenas, each with
30/30 wins, at M12, 24 groups and 65 active experts. All 19,200 boundary,
348 complete-list and 360 post-event checks pass. These are local FFN
measurements. [Raw summary](../ffn-active-timing-v3-a3-194/summary.json)

### 2. One resident layer with two explicit device TP phases

Call input norm, indexed/shared attention, `AllReduceBf16`, callable FFN and
a second `AllReduceBf16` from one MIX entry, preserving the phase contracts
above. The existing communication function consumes BF16 `[M,6144]` and
has three mixed barriers per reduction. Its two payload planes and two
ready planes use consecutive persistent epochs; the two reductions and
later graph replays must not reuse or reset an in-flight epoch.

The current SHMEM path requires every peer to be MTE/HCCS reachable. It
sums each rank's BF16 input in ascending rank order in FP32 and rounds once.
Two-rank numerical/replay success does not establish equality with HCCL's
TP16 reduction tree. The first new gate therefore exercises two reductions
around a nonlinear norm/FFN dependency, then tests TP16 ordered/HCCL outputs,
changing replay and a delayed rank. In-flight GM window ownership, physical
residency and transport are acceptance conditions. At capacity32/TP16 the
current communication window is 12,585,024 bytes per rank.

This route can remove host/device phase handoffs and alter collective cost;
it leaves SFA and expert matrix work intact. It ranks second because actual
TP16 correctness and collective latency are not yet measured. A matched
HCCL-versus-SHMEM layer comparison must separate communication changes from
launch changes. [Current communication contract](source/communication.txt)

### 3. Pipeline CUBE, Vector and communication tasks

An expert-level pipeline could admit ready down projections while AIV
activates other experts and AIC computes their GU projections. Keep the
existing BF16/INT8/INT4 boundaries, per-expert GM partials and original-slot
combination order. Readiness must be per task, with a schedule in which a
resident producer cannot be blocked by its waiting consumer. The current
whole-phase `Forward` functions do not expose that contract; deleting their
barriers is not an implementation.

A row-wave path across attention TP and norm has additional constraints:
RMSNorm requires the complete reduced 6144-element row, and routed counts
and grouping depend on all submitted rows. Splitting a round changes GMM
batching and can change tiling or arithmetic. The first experiment should
therefore pipeline only already delimited expert tasks, retain all GM
boundaries for comparison, and include changing active experts, hot routes,
zero probabilities and signed zeros. A separate two-rank phase tests early
row publication before any cross-rank overlap is claimed.

This route ranks third in implementation readiness, although it can change
serialized work rather than just launches. The FFN profile exposes sizeable
GU, activation/down and route-store windows, but they include waits and
different workers' maxima cannot be added. Its instrumentation adds roughly
13 us to the complete FFN. That number is neither a removable production
phase nor a correction to every phase measurement.

## Unified TorchNPU 2.12 runtime

The existing Python 3.11/TorchNPU 2.12 image initially fails the official SFA
import at one multiline f-string in `sharded_state_loader.py:214–219`.
[The unchanged-source report](cpu212-original-report.json) records the actual
failure and NPU initialization false. [The compatibility patch](ast-equivalence.patch)
adds four quote characters to that one diagnostic string, in an independent
source copy. Python 3.12 produces identical complete-file ASTs, with line
counts unchanged; comparison of all 7,701 files finds exactly one changed
file. [AST and source identities](ast-equivalence.json)

The isolated copy imports all eight requested real modules under Python
3.11.12, Torch 2.12.0+cu130 and TorchNPU 2.12.0, including SFA, the actual
decoder module, GateLinear, MoERunner and ACLGraphWrapper.
[Import report](cpu212-compatible-report.json) Both original and compatible
CPU runs expose no NPU devices and leave initialization false.

Official worker registration additionally needs Triton's supported
`TRITON_ASCEND_ARCH=Ascend910_9392` setting in a device-free process.
Without it the hardware architecture query fails; no architecture query
was replaced with a stub. With it, actual worker patches/registration,
real TP1 Gloo groups, MLA/SFA/indexer construction and profiling forward
complete. Weight postprocessing then reaches `npu_format_cast`, which has
no CPU implementation. [Unspecified-target report](cpu212-worker-report.json),
[explicit-target report](cpu212-worker-arch-report.json)

Those constructor results concern the attention component at TP1. The next
decoder fixture must include both norms, the MoE factory and original
quantization wrappers at the correct layer index, then execute real
postload and both reduction cut interfaces. Rebinding frozen device objects
to Torch 2.12 validates ABI loading separately from those device gates.
The Python 3.12/TorchNPU 2.10 runtime and its accepted libraries remain intact.
