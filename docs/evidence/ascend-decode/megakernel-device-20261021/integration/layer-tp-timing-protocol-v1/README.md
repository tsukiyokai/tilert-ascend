# Two-rank FP32 reduction timing contract

The primary comparison uses the frozen C2 TP-layer entry against three native compute stages separated by the project's FP32 reduction arithmetic. Both arms retain the same layer6/full-indexer and layer7/shared-indexer weights, metadata, caches, compute scratch and four BF16 reduction destinations. The existing BF16 HCCL2 eager and graph gates establish correctness for two controlled shards; they do not measure this FP32 control. No timing device result is included here.

## Reduction source contract

`CannModel._tp_reduce` calls `_tp_sum_original(value.float()).bfloat16()` for BF16 input. Its saved `_all_reduce_out_place` resolves through the pinned `GroupCoordinator` to `NPUCommunicator.all_reduce`. That class inherits `DeviceCommunicatorBase.all_reduce`, which calls `dist.all_reduce(input_, group=self.device_group)` and returns the same tensor. These pinned functions add no clone. The two dtype conversions create the distinct FP32 and BF16 owners. The source excerpts, full-file hashes and deployed-file checks are in [sources.json](sources.json).

The same-address native control allocates four FP32 `[M,6144]` buffers before capture, one for each reduction in the two-layer pair. At each seam it copies the BF16 partial into the FP32 buffer, executes real HCCL2 FP32 SUM, then converts into the existing BF16 reduction destination. The first result feeds the original FFN parent with the separately retained input-norm residual. The second feeds the next layer; its FFN-retained residual remains a separate input. The additional FP32 storage is `16*M*6144` bytes per rank, or 3,145,728 bytes at M32.

The control preserves both conversion tasks and the collective; preallocation removes allocation preparation, not the dtype operations. An event-external comparison must run the literal project expression on the same partials and compare both conversion results and the final BF16 bytes. Record the actual dispatched operator names for literal conversion and `copy_`; equality of values does not prove identical kernels or task counts. If dispatch differs, label the timed arm as the preallocated native FP32 control. It is not the complete project's host or allocation path.

The actual installed Torch 2.12 [all_reduce source](cpu/all_reduce.py) obtains a work object and waits when `async_op=False`. Keep that entry and its ordering; do not remove the wait or add a device synchronization inside capture. The immediate BF16 consumer and full numerical gate must prove that the captured HCCL completion joins the compute stream.

`CannModel` defaults `aiv_collectives=True` and calls `_choose_collective_engine` before communicator creation. That function uses `setdefault` for `HCCL_OP_EXPANSION_MODE=AIV` and `HCCL_DETERMINISTIC=true`; existing process values can override the defaults. This timing launcher explicitly supplies both literal values before Python starts and each worker checks them before importing Torch. Reports preserve those values, optional HCCL algorithm/buffer overrides and the separate Torch deterministic-algorithms flag. The accepted BF16 HCCL2 gates did not pin this environment and do not establish this AIV FP32 baseline or its graph support. The new numerical and capture gates must establish both. See the pinned [selection source](source/project-collective-engine.py).

## Compute and ownership

Use `norm_quant_out` from library `40a9e3c0…`, indexed attention `8deff750…`, shared attention `64108f6f…`, FFN `381febf9…`, and candidate C2 `c2b72e46…`. The standalone norm writes the existing normalized, retained, INT8 and FP32 scale buffers directly. Its `rstd` argument is a `[M,1]` view of the existing `[M]` allocation, with identical data pointer and storage; no new tensor is substituted. The original norm module plus DynamicQuant remains an event-external oracle for its four observable outputs.

Keep two independent arena sets alive and run both arms in each set, so an address placement advantage cannot masquerade as fusion benefit. Both arms share every production tensor address within an arena, including all four BF16 reduction outputs. Only the control's FP32 collective temporaries are private. Retain the two real synthetic layer weight sets, both rank-specific shard sets and all actual parent validators. This is one repeated weight bank, not a full-model or cold-address measurement.

Both arms use the official `ACLGraphWrapper` on the same actual default capture stream. Construct and retain an unentered Torch graph manager before SHMEM window allocation. The window's stream, the runnable's capture stream and replay's current stream must match. All graphs are reset before their owners are released; all graphs are gone before collective window release. Fourteen CPU-only constructor peers remain alive through the timing run and never enter device collectives.

## Correctness before events

Before timing, compare original eager parents, native-norm plus FP32 HCCL parents, the control graph and the C2 graph over all 113 defined boundaries per rank. Use changing hidden/residual values, independent main/indexer page maps, actual lengths and caches in retained owners. Keep rank-distinct attention/FFN partials, rank-equal reduced outputs, ordered two-rank FP32-sum checks, complete active tables and strict byte comparison. Poison floats with NaN and integer outputs with different defined values; set active-table count to zero while leaving its other 263 words nonzero. Parent validation, CPU snapshots, gathers and deliberate rank delays stay outside timing.

Timed graphs omit top-k observation copies and final observation consumers. They retain the production copy from layer6 selected indices into the persistent top-k owner required by layer7. Observations read the corresponding live production views after execution. Removing observation tasks from both arms requires a fresh numerical gate; the previous graph result alone is insufficient.

Repeated execution must be proven idempotent for the fixed timed input. Every pair starts from the same external hidden/residual, not the prior pair's final output. Current-token cache writes must overwrite the same slots from the same producers. After one pair, repeated pairs and repeated graph replays must preserve all 113 numerical boundaries and all live inputs. The communication epoch is excluded from this equality: it must advance four times per candidate pair and is never reset by the host. A final-state check establishes only the retained final-call boundaries; it does not assert that every intermediate repeated call was observed.

## Paired measurement

Start with M12, 24 mixed groups, two arenas, 16 two-layer pairs per graph, 30 graph replays per event and 30 alternating AB/BA samples per arena. Preserve one-pair graph measurements as a separate launch-gap sensitivity check. Before each arm: reset caches and poison outputs outside events, replay three warmups, synchronize locally, then rendezvous the two device ranks through their CPU group. Both ranks must agree on `(arena, sample, arm)` before recording start events. Do not insert deliberate rank delay into timed intervals.

Record start and end events on the window/capture stream. End-event synchronization precedes any CPU gather. For rank `r`, arm `a` and sample `j`, let `d[r,a,j]` be the local event duration divided by the number of pairs. Use `D[a,j] = max(d[0,a,j], d[1,a,j])` and paired saving `D[control,j] - D[candidate,j]`. Retain both raw rank durations and compute distributions from these paired maxima. Do not compare unpaired medians or subtract timestamps from different devices. This measures the slowest rank's event interval, including any remaining rank arrival skew; it is not a globally synchronized wall-clock makespan.

After every event, compare final output, retained residual, both reduced tensors, persistent top-k and active tables against independent expected CPU snapshots. After all samples, poison and recheck every boundary in both arms, including all cache bytes and live-owner identities. Leave the SHMEM window alive across warmups, arm alternation, samples and arenas.

## Comparison priorities

1. The same-address native FP32 control is the primary feasible comparison. It measures the combined effect of whole-layer execution and SHMEM2 versus separate native compute plus HCCL2. It cannot attribute the difference to fusion alone.
2. The literal conversion path and one-pair graph are sensitivity controls. They distinguish dtype dispatch and replay-submission effects from the primary repeated-pair result without changing device mathematics.
3. A separate-native baseline using the same device SHMEM protocol would isolate launch fusion from collective choice. It needs a new callable communication entry and its own epoch, lifetime and numerical gates; no such result is implied by the current comparison.
