# One-launch rank-local FFN

The independent `tilert_ffn_composed` library calls the frozen bridge, the
validated callable routed/shared body, and the bridge BF16 epilogue in one
mixed kernel. The attention TP reduction precedes this interface; the FFN TP
reduction consumes its output. Neither collective is implemented here.

The bridge retains BF16 residual and normalized tensors, FP32 gate logits and
top-eight probabilities, INT32 IDs, and independent shared INT8 codes/scales.
The routed input-front computes its own quantization from normalized BF16;
it does not reuse shared codes or scales. The routed and shared branches
retain separate BF16 outputs. The epilogue stores BF16(routed * 2.5), then
stores BF16(scaled_routed + shared).

The frozen bridge gate uses BF16 ND [6144,256] and sixteen N16 tiles. The
[standalone bridge gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge212-v1-summary.json)
compares all FP32 logits against the pinned Torch-NPU 2.12 production call
with original ND NK storage and transpose. Its M1/2/12/32, 16/24-group
results pass all eight bridge boundaries.

## Interface and ownership

`prepare(BF16[M,6144], blocks, weight_layout)` returns `(bridge12, body20, sync)`.
M is 1 through 32; blocks is 16 or 24; routed weight layout is 0 (ND) or 1 (NZ).
The original preparation code checks physical core capacity and generates
immutable tilings. Both preparations must return the same synchronization base.

`forward_out(input, residual, gamma, gate_kn, correction_bias, routed_weights,
shared_weights, bridge12, body20, epsilon, blocks, weight_layout, sync)` mutates
the supplied scratch and returns nothing. `routed_weights` contains GU,
GU correction, GU scale, down, down correction, down scale. `shared_weights`
contains packed GU, GU scale, packed down and down scale. All weight shapes
and physical ND/NZ layouts retain the parents' checks.

All 32 scratch tensors are disjoint. The two 16 MiB system workspaces remain
separate; M32 occupies 55,393,664 bytes in total. All scratch and captured
graphs must remain alive through completion;
reuse is serial on one device stream. Host validation reads no routing values,
allocates no device memory and flushes queued Torch producers using `stream()`.
The shared scale [M] input is a host-only view of the retained bridge [M,1] owner.

Bridge scratch slots remain those in `ffn_bridge/contract.h`: combined output
is slot 10, scaled routed output 11, retained residual 1. Body slots retain
their standalone numbering, including routed output 9 and shared output 19.

## Synchronization

The bridge's three original EndPhase boundaries complete before body setup.
On CANN 9.2 `dav_c220`, `SetSysWorkspace` initializes only a null reserved
pointer. The standalone wrapper selects bridge scratch slot 9 first; the
later call with body slot 8 does not switch it. Both 16 MiB buffers remain in
the ABI, and the selected workspace remains alive across all stages. The
[SDK and disassembly evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-system-workspace-source-audit/summary.json)
records the actual selection semantics. Every mixed core enters the body's original
six barriers, including cores without a matrix or row. Callable body return
does not publish its final combine output: the wrapper explicitly drains all
pipes, issues DDR DSB, enters mixed SyncAll, then resets the pipe. Only then
does the epilogue read both BF16 outputs. Its final EndPhase publishes the
combined output. No bridge or body GM allocation is reused between phases.

`prepare.py` verifies both parent source manifests, extracts the original
bridge statements unchanged, and scopes each binding's host functions in its
own namespace. The direct-launch statements and Torch registrations are
removed from these private host seams; shape, dtype, physical format, tiling
owner and alias checks remain. The composed binding adds cross-parent overlap
checks. Parent source files and libraries are not edited.

## Build and acceptance

`build.sh NEW_OUTPUT BRIDGE_BUILD_V2 BODY_BUILD_V2` creates a frozen source tree,
compiles CANN and the active Torch-NPU binding, and records command, source and
artifact identities. It refuses an existing output directory. CPU source and
ownership checks do not substitute for device validation.

Device acceptance compares all production boundaries against separate bridge,
body and BF16 epilogue calls, then retains the official norm/top-k/gate and
expert references. Use changing inputs/routes at M1/2/12/32 and 16/24 groups,
including zero, cancellation and tied-logit witnesses. TP16 and whole-model
latency remain separate tests.

`check.py --oracle-mode orchestration` compares the combined kernel with the
three frozen native calls. `--oracle-mode both` adds the official norm,
original-NK gate, top-k and independent DynamicQuant, followed by the accepted
body and separate BF16 multiply/add. Both oracles report individual raw-byte
checks, including retained residual, full FP32 logits and initialized shared
and routed intermediates. They use independent arenas and captured queued
input producers. Capacity padding and system workspaces are not arithmetic
outputs.

The [build-v2 evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-194/ffn-composed-build-v2/manifest.json)
records successful CANN 9.2 compilation and Torch-NPU 2.12 CPU loading, with
library SHA `3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655`.
The frozen `harness-v1/check.py` co-loads all native libraries and the private
official oracle without initializing NPU. Four CPU tests cover exact host
validation preservation, bridge statement preservation, source manifests,
and valid shared rows versus padding; a C++ check covers 64 geometries.

The [composed device gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-composed212-v2-summary.json)
passes M1/2/12/32 at 16 and 24 groups with NZ routed weights and BF16 source
shared scales. Across 24 captures, 64 candidate and 128 reference changing-input
replays, all 3,840 raw-byte comparisons pass. The 30 boundaries include retained
residual, gate FP32 logits, route metadata, both expert branches and both BF16
epilogue stores. The production arm uses the official bridge and the accepted
native expert body; it is not an independent end-to-end vendor FFN reference.

## Paired timing

`bench.py` compares the frozen one-launch C ABI with the frozen bridge, body
and epilogue C ABIs. Both arms use exactly the same 32 scratch addresses in
each pair. Two independent arenas are measured: one prepared by the composed
library, one by the separate libraries. Their owning Torch interfaces validate
shapes, tiling ownership and aliasing before capture and after timing. The raw
C ABIs do not repeat host validation inside graph replay.

The default M12 experiment uses 16 equal-valued weight banks with distinct
storage, 16 independent calls per graph, and 30 alternating paired samples per
arena. Each event encloses 30 graph replays; three warmups precede every sample.
Input updates, copies and numerical snapshots stay outside timed graphs and
events. All eight changing fixtures and all banks pass 30 boundary checks
before timing; each timed arm checks its final output immediately, then all
banks are checked again separately. Reports retain raw events, paired savings,
p50/p90/p99, standard deviation, addresses, hashes and actual route counts.
Before each correctness replay and each timed arm's warmup, logical floating
outputs are filled with NaN and integer outputs with legal test values. The
finite reference must be regenerated by execution. Tiling, system workspace,
input tensors and undefined capacity padding are unchanged; initialization and
its synchronization are outside every event window.

The minimum selected matrix payload is 228 MiB across 16 banks: 3 MiB of gate,
9 MiB for eight routed experts, and 2.25 MiB of shared weights per bank. This
counts logical matrix bytes; it does not measure HBM traffic or cache misses.
The full routed weight allocation is 4.5 GiB. TP communication and whole-model
latency are outside this timing scope.

The [timing-v3 harness](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-194/ffn-composed-timing-v3/manifest.json)
pins all three libraries and seven runtime source files. Five CPU tests pass
on x86-64 and AArch64, including C ABI argument recording against the frozen
C++ declarations and rejection of stale finite outputs after an empty replay.
The three libraries co-load on Torch-NPU 2.12 without initializing NPU.
`bench_run.sh device OUTPUT_DIRECTORY` selects the recorded 16-bank experiment;
`preflight` selects only library and source loading.

The [M12 result](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-composed-timing-v3-a3-194/summary.json)
shows no latency benefit from one launch. At 24 groups, 16 banks and 65 active
experts, fused/separate p50 is 236.917/235.734 us in the first arena and
234.607/234.272 us in the second. Median paired savings are -1.155 and
-0.337 us; the fused arm wins 0/30 pairs in each arena. All 19,200 pre/post
boundary byte checks and 360 event-tail checks pass. This validates the callable
composition's dataflow but does not establish a performance improvement.
