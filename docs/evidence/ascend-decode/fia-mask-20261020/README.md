# MLA selection masks

## Verified combination

A3-187, die 0, installed CANN 9.2.0, production vendor. Eight eager calls:
SFA and safe-gather/masked-FIA under A/B/reset, plus all-False/no-mask FIA.
No graph or timing run. Torch peak reservation 166 MiB under a 256 MiB quota.
[Device result and library hashes](result.json), [executed source](device-api-check.py).

All-False and absent masks produce bit-identical FIA output. Replacing two
future KV rows with 1000 leaves each path's output unchanged. The combination
first maps invalid gather addresses to initialized slot 0, then masks those
positions; this experiment does not isolate the mask from safe gathering.
FIA/SFA NRMSE is 0.001658; against independent FP64, 0.001926/0.001949.
These are diagnostics, not new model acceptance thresholds.

Server CPU comparison of masked and explicitly filtered FP64 attention has
max_abs 1.943e-16. Safe gathering also excludes NaN future rows. Preserve
original IDs when constructing the mask; changing actual_seq_kvlen to the
number of valid slots is incorrect when valid slots are not a prefix.
[CPU source](cpu-contract.py), [CPU results](cpu-contract.json).

## Meaning of the old width gate

The indexer already selects with sparse_mode=3 and per-query causal lengths.
Its implementation has no width-six causality boundary. The old width-eight
serial/speculative failure remains observed, but no saved selected-ID evidence
establishes future leakage as its cause. Keep the existing gate until the
failure is isolated; the mask API check alone does not remove it.

## Joint queries over one KV cache

M8, four heads, capacity 9344: BSND queries `[1,8,4,512]`, one latent cache
`[1,9344,1,512]`, RoPE dimension 64 and bool selection mask `[1,1,8,9344]`.
Each query's 2048 unique IDs become unmasked positions. Dense QK and PV do
4.5625 times the selected-token arithmetic; the representation avoids eight
gathered KV copies (18,874,368 bytes of buffers, not measured HBM traffic).

Server CPU masked-dense versus gathered FP64 max_abs is 3.608e-16. Changing
finite unselected values preserves output exactly. Duplicate selected IDs
would change multiplicities, so this set representation requires uniqueness.
[CPU contract](joint/result.json), [executed source](joint/cpu.py).

One eager A3-187 die-0 call on the installed production vendor succeeds.
NRMSE is 0.002092 versus saved SFA and 0.001811 versus filtered FP64;
this cross-kernel diagnostic does not establish decoder equivalence.
Observed tiling key: `105000000010122221`. Peak reservation: 74 MiB.
[Device result](joint/device-result.json), [source](joint/device-api.py),
[selected key](joint/selected-tiling.txt).

Installed device source loads K/V tiles before the inner query-row loop.
M8 times four heads fits that shared tile. The CPU host splitter replay uses
S2=8192, not this call's capacity 9344; its split count is not an observed
device tiling. [Source binding](joint/source-manifest.json).

The original cache allocation uses torch.empty and injection only fills
the prefix. A fixed-capacity dense PV path therefore first needs initialized,
finite unused rows: a zero probability multiplied by NaN is still NaN.
Initialization belongs to cache setup. The CPU counterexample produces
16,384 nonfinite output elements from one unselected NaN row; zeroing the
tail restores the original result. The optional shared_fia backend now zeros
its cache storage before binding views; ordinary SFA allocation is unchanged.

## Complete-path timing and attribution

Thirty same-process alternating pairs, ten calls per Graph replay: SFA event
median 96.070 us, joint path 608.663 us; wall medians 110.222/623.799 us.
Each path's A/B/reset output is bit-identical between eager and Graph.
The joint path includes index conversion, bitmap fill/scatter, causal compare,
OR and FIA. This is one synthetic hot-input M8 fixture, not decoder TPOT.
Another workload released its resident weights during this process, so retain
the full paired samples and resource snapshots. [Raw screen](joint/screen/result.json).

A separate ten-call-per-path attribution profile finds bool ScatterElements
on AI_CPU: mean 531.554 us, 86.28% of joint task time. FIA itself is MIX_AIC,
mean 56.962 us; SFA is 86.014 us in that profile. The actual full-cache FIA
uses 19 Cube/38 Vector blocks. [Task accounting](joint/profile/summary.json),
[raw device durations](joint/profile/kernel_details.csv).

Two subsequent single-change screens preserve every A/B/reset output bit:

| Bitmap representation | Joint / paired SFA event median | Actual dispatch |
| --- | --- | --- |
| uint8, then OR with bool | 693.825 / 94.403 us | Scatter becomes Vector, OR becomes AI_CPU. |
| uint8 scatter, bool view before OR | 132.017 / 95.389 us | All mask kernels are Vector; the view shares the same device pointer. |

Each screen uses 30 alternating pairs of ten-call Graph replays; sources,
raw samples, dispatch evidence and hashes are retained with the
[uint8 result](joint/uint8-screen/summary.json) and
[bool-view result](joint/bool-view-screen/summary.json). Eliminating AI_CPU
fallbacks alone does not produce a net win over SFA.

The shared_fia mechanism reuses the mask across layers that already reuse the same
selected IDs. The actual checkpoint uses offset 3/frequency 4: groups are
`[0]`, `[1]`, `[2..5]`, `[6..9]`, through `[74..77]`, or 21 producers for
78 attention calls. Rebuild on each producer and each forward; shared buffer
addresses alone cannot identify the current selection. Keep each layer's Q/KV
independent. [Checkpoint and lifecycle proof](joint/sharing/audit.json).

## Four-layer sharing and integration

One synthetic M8/C9344 group, one mask build and four distinct Q/KV pairs:
30 alternating Graph pairs give SFA/shared event group medians
425.780/364.470 us. Paired ratio median is 0.85445; all 30 favor sharing.
Wall medians are 565.892/504.636 us. These include causal compare, selection
mask and all four attention calls. Peak reserved is 216 MiB. During the
18-second process another workload loaded 46–47 GB on every die; this is a
shared-host screen. [Raw samples](joint/four-layer/result.json),
[summary](joint/four-layer/summary.json), [source](joint/four-layer/screen.py).

Shared versus independently rebuilt FIA matches bitwise in all 12 A/B/reset
layer comparisons; each path's eager/Graph comparisons pass 24/24. This does
not make SFA and FIA bit-equivalent: their arithmetic order differs.

The real SharedKVAttention binder was then checked separately with 28 calls:
12 eager, four capture, 12 replay. All 24 checked layer outputs match saved
rebuilt FIA bitwise across A/B/reset, including changed IDs and causal slots.
Peak reserved is 136 MiB. [Actual binder check](joint/binder/result.json),
[executed source](joint/binder/check_binder.py).

The opt-in --attention-kernel shared_fia mode supports target widths 8 and32;
M1 and MTP retain SFA. Producer selection is refreshed on every forward and
Graph replay. Server CPU lifecycle/fallback checks pass in 0.006 s. Full-model
frozen-teacher and serial/speculative checks, all 21 groups, and request TPOT
remain unmeasured for this mode. Best measured model TPOT remains 11.6198 ms.

The graph acceptance harness now poisons unused shared_fia main MLA rows
with finite 7.0, preserving NaN/-128 for indexer and MTP caches. This tests
future-slot exclusion without violating dense PV's finite-input contract.
Server CPU execution of the actual loop passes 36 field/range checks.
Compiled-only runs now report serial consistency as null rather than claiming
a serial reference they skipped; six report cases pass.
[Harness contract](joint/poison/result.json).

A server CPU fullgraph check retains 78 opaque MLA nodes in one FX graph.
A/B/reset preserve layer order and selection-mask state with the actual
wrapper/kernel bodies and binder. The FIA primitive is a mask observer in
this check; NPU backend compilation remains untested.
[CPU graph contract](joint/compile/result.json).

## M32 candidate

At 32 queries and four heads, folded M=128 still fits one L1 query tile.
Matching-source host SplitCore replay predicts the same 19 main S2 tasks
at C9344; softmax subloops grow fourfold and FD vector groups rise4→16.
Ordinary workspace plus FD grows35.344→66.844MiB before platform workspace.
These are CPU/source predictions, not observed device tiling.
[Source and replay](joint/m32-source/manifest.json).

The server CPU M32 fixture preserves unique indices, finite KV and independent
layer Q/KV. A/B/reset mask checks and stale-ID/threshold counterexamples pass;
FP64 dense-versus-selected error on the last query is2.498e-16.
[CPU contract](joint/m32-cpu/result.json). Whole-round benefit requires
T32/T8 < A32/A8; the frozen unigram trajectory gives A32/A8≈1.551.
Attention speed alone does not price the wider expert union.

Two M32 eager calls on idle A3-187 die2 execute the real SFA and shared binder
with actual keys1158/105000000010122221. Both outputs are finite; mask equals
the CPU contract. Cross-kernel NRMSE is0.002139; query31 versus FP64 is
0.001952/0.001793 for SFA/FIA. Reserved memory peaks124MiB. No Graph or timing
in this API check. [Bounded API evidence](joint/m32-api/result.json).

The subsequent four-layer M32 screen uses the real binder, one shared mask
and four independent Q/KV pairs. Thirty alternating pairs give event group
medians657.170us SFA /416.120us shared; paired ratio median0.63036,30/30 wins.
Wall medians795.195/554.052us. All12 shared/rebuilt and24 eager/Graph A/B/reset
checks match bitwise. Peak reserved272MiB; die2 is idle before and after,
while dies0/1 host other models. There are364 attention calls in total.
Initialization establishes a default-to-capture-stream dependency before
any warmup. [Raw pairs](joint/m32-screen/result.json),
[summary](joint/m32-screen/summary.json), [source](joint/m32-screen/screen.py).

M32 is now an explicit experimental backend option for lookup verification.
Its whole-model teacher, serial equivalence, memory peak and request TPOT
remain unmeasured. M8 andM32 screens ran at different times and on different
dies; their absolute timings do not establish a cross-width cost ratio.
