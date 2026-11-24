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

The frozen bridge gate uses BF16 ND [6144,256] and sixteen N16 tiles. Its exact
FP32 logits still require comparison against the available pinned production
gate entry with original NK storage and transpose. Passing isolated top-k or
the callable body does not establish this gate contract.

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
separate. All scratch and captured graphs must remain alive through completion;
reuse is serial on one device stream. Host validation reads no routing values,
allocates no device memory and flushes queued Torch producers using `stream()`.
The shared scale [M] input is a host-only view of the retained bridge [M,1] owner.

Bridge scratch slots remain those in `ffn_bridge/contract.h`: combined output
is slot 10, scaled routed output 11, retained residual 1. Body slots retain
their standalone numbering, including routed output 9 and shared output 19.

## Synchronization

The bridge's three original EndPhase boundaries complete before the body
system workspace is selected. Every mixed core enters the body's original
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
