# A hand-written one-sided TP reduction: mechanism works, my kernel does not

The gate was a sixteen-rank one-shot push plus ordered local reduction,
measured against HCCL's 63.5 µs. The mechanism is confirmed available and the
numerics are correct; the kernel I wrote is two orders too slow and the fault
is squarely in its reduction stage.
[kernel](../../../../tests/glm52/shmem/tp_push_kernel.cpp),
[harness](../../../../tests/glm52/shmem/main.cpp).

## What works

Built as an example inside the installed SHMEM SDK tree. Each rank publishes
its `[4, 6144]` float partial into slot `my_pe` of every peer's symmetric
window with `aclshmemx_mte_put_nbi`, drains with `aclshmemx_mte_quiet`,
announces with `aclshmemx_signal_op(..., SET, peer)`; every core then spins
`aclshmem_signal_wait_until` on all sixteen local slots and sums in rank
order. No barrier and no collective.

Each rank contributes its own index as the payload, so the sum must be
`pe_size*(pe_size-1)/2` in every element and a wrong order cannot pass. At two
PEs it reports `wrong 0 of 24576, expected 1.0 got 1.0`. **The one-sided
reduction is numerically correct.**

## What does not, and the measurement that found it

| configuration | per reduction |
| --- | --- |
| push and reduce | 543 µs |
| **reduce only, no communication at all** | **557 µs** |
| reduce only, vectorised through UB | 358 µs |

Splitting the kernel into push-only and reduce-only stages was the whole
diagnosis: with communication switched off entirely the cost is unchanged. So
none of the 543 µs is the push — **it is all my reduction loop**.

The first version accumulated with a scalar `for` over elements, so each of
24576 elements paid sixteen separate global reads. Staging the stripe through
UB and using `Add` brought it to 358 µs, which is still absurd for 96 KiB —
roughly 1 µs of traffic at the measured rate. The remaining fault is visible
in the code: it re-copies all sixteen partials into UB every round and places
`PipeBarrier<PIPE_ALL>` inside the loop, draining the whole pipeline on each
of the sixteen adds. The vectorised variant also hangs in stage 0, so its
flag protocol and its UB region are interacting.

Two incidental notes. `LocalTensor` is constructed by assigning
`address_.logicPos`, `address_.bufferAddr` and `address_.dataLen` directly,
as `examples/notifywait/main.cpp:120` does; `SetAddrWithOffset` does not have
a matching overload here. And the seven-argument
`aclshmemx_mte_put_nbi(dst, src, buf, ub_size, elem_size, pe, sync_id)` is the
GM-to-GM form that stages through a UB scratch, which is the one to use when
the source is in global memory — the five-argument form requires the source
already in UB.

## The launch cost was masquerading as the reduction

Every number above divided one kernel launch by the round count. A round
sweep at fixed shape exposed it:

| rounds | reported per reduction | implied total |
| --- | --- | --- |
| 1 | 26454.8 µs | 26.5 ms |
| 5 | 5363.1 | 26.8 |
| 50 | 573.7 | 28.7 |
| 400 | 70.8 | 28.3 |

The product is constant at about 26–28 ms, so the whole figure was a single
launch amortised, and the per-round work was invisible underneath it. The
earlier "557 µs reduce-only" and "543 µs full" were both that same launch.

With the launch amortised over 4000 rounds the per-reduction cost appears:

| configuration | cores | per reduction |
| --- | --- | --- |
| reduce only | 8 | 7.118 µs |
| reduce only | 48 | 23.184 |
| **push and reduce** | **4** | **5.215** |
| push and reduce | 8 | 6.435 |
| push and reduce | 16 | 9.936 |

**About 5 µs for a complete two-rank push plus ordered local reduction of
96 KiB**, against HCCL's 126 µs eager at the same shape. More cores is
strictly worse — 48 costs 3.8× what 8 does — so the reduction is bound by
inter-core coordination, not by arithmetic, and a batch-one reduction wants
few cores.

These timings are provisional: the numerical check does not pass at high
round counts, so they measure a kernel that is not yet correct.

## Two real defects found, and 507015 earned honestly

Rewriting rather than patching surfaced two concrete faults, both mine.

**A MIX kernel body runs on both core types.** A device print of the stripe
computation showed it directly:

```
[AIC Block 0/4] reduce core=0 cores=4 elements=24576 stripe=6144 begin=0
[AIV Block 0/8] reduce core=0 cores=4 elements=24576 stripe=6144 begin=0
[AIV Block 4/8] reduce core=4 cores=4 elements=24576 stripe=6144 begin=24576
```

`GetBlockNum()` returns the launch's 4, but the vector side has
`GetSubBlockNum()` blocks per cube block, so there are 8. Cube and vector were
duplicating stripes 0–3 onto the same output while vector blocks 4–7 fell off
the end — and the cube side, which has no vector units, wrote rubbish. The fix
is `cores = GetBlockNum() * GetSubBlockNum()` with the reduction inside
`if ASCEND_IS_AIV`.

**The UB budget is 192 KiB in total, not per consumer.** I had the put's
staging area at 190 KiB and then placed the reduction's buffers at a 192 KiB
offset, entirely out of bounds. Its stores silently did nothing, which a
sentinel exposed: pre-filling the output with −12345 and counting survivors
reported `untouched 24576 of 24576`, distinguishing "wrote a wrong sum" from
"never wrote at all" in one run where a zero could not. Both regions now come
out of the one budget: reduction 2×32 KiB from 1 KiB, staging 126 KiB from
65 KiB.

With the layout corrected the reduction executes, and the run now deadlocks
with **507015** in the plog. That is the error HyperMPMD's own bindings name
as what an untimed device spin does under interconnect contention, and it is
now ours legitimately rather than by accident — the earlier retraction stands,
since sixteen PEs do form a domain, but a hand-written spin can still earn it.
One item of their recipe is missing here: the host must converge before the
first remotely visible write, and my `aclshmem_barrier_all()` runs after the
warmup launch rather than before it.

## Where the kernel stands, unfinished

Correct at 50 rounds with a single data plane; after adding round-parity
double buffering — so a push cannot overwrite data a previous round is still
reducing — `out` comes back as zero at every round count. The parity is
computed from `round & 1` inside the kernel while the host issues two
launches with `round_base` of 0 and then `warmup`, so the plane the last round
writes is not the plane the host reads. That is a plain indexing error in my
code and needs a rewrite rather than another single-line guess; stopping here
rather than iterating blind.

## Reading the official demo fixed the deadlock

The protocol I had written was inverted from the SDK's, and
`examples/allgather/allgather_kernel.cpp` shows it in three places.

The window's head is the flag region and the payload follows it —
`gva_sync_gm = (int32_t*)gva`, `gva_data_gm = (int32_t*)gva + aivNum *
SYNC_FLAG_INTERVAL` — and the flags are indexed per core, sixteen int32 apart,
not per rank. I had data first, flags last, one slot per rank 32 bytes apart.

The producer pushes to **itself**: `aclshmemx_mte_put_nbi(..., my_rank, ...)`
moves local GM into its own symmetric window and then
`aclshmemx_signal_op(..., my_rank)` raises its own flag. I had every rank
pushing into all fifteen peers.

And the consumer never blocks. It pulls the peer's flag into UB with
`aclshmem_int32_get_nbi(flags_ub2[g], gva_sync_gm + g * SYNC_FLAG_INTERVAL,
1, x)`, tests it, and `continue`s to the next group if it is not ready —
a non-blocking poll over all groups. I used `aclshmem_signal_wait_until`,
which is an untimed spin on one slot, and that is exactly what cann/shmem
PR 897 attributes `aclError=507015` to.

Rewritten to that structure — flags at the head, publish to self, poll peers
with `int32_get_nbi` — **the deadlock is gone**: the push half completes in
**1.814 µs** at two PEs where it previously hung at every size down to 64
elements.

Also applied from upstream before that, and worth keeping: PR 913 widens the
READY slot so adjacent ranks' flags do not share a cache line, adds an
inter-core sync before the receive wait, and puts a timeout on the wait; PR 897
checks `aclshmemx_set_qp_num`'s return and the SDMA QP count against the
device's vector-core count before launching.

## On the fixed library: publish is 0.874 µs

The library's barrier defect
([ABI](../shmem-abi-20261018/README.md)) had to be rebuilt and reinstalled
before any of this measured anything — device-side builds do not track header
dependencies, so the `.cpp` must be touched first and the `.run` reinstalled.

With that in place, at two PEs, blocks 8, `[4, 6144]` float32:

| stage | result |
| --- | --- |
| publish and wait, no reduce | **0.874 µs**, `wrong 0` |
| reduce only | hangs, no plog |
| full | hangs, no plog |

The publish half more than halved from 1.814 µs once the barrier stopped
being a silent no-op. The stall is in the reduction, which an independent
investigation reproduced on the same code — my earlier attribution of the cost
to the reduce half was right about the location and wrong about everything
measured through the broken barrier.

Four diagnosed faults are now fixed and it still hangs, so at least one is not
the cause: the reduction's buffers come from a `TPipe` with
`TBuf<TPosition::VECCALC>` allocated once at launch scope rather than
hand-filled `LocalTensor::address_`; every explicit `SetFlag`/`WaitFlag` pair
in the reduction is a `PipeBarrier<PIPE_ALL>`; UB regions sit above
ACLSHMEM's reserved `[0, 16384)`; and the core index is
`GetBlockIdx() * GetSubBlockNum() + GetSubBlockIdx()`, since `GetBlockIdx()`
alone is the cube block index in a MIX kernel and two sub-blocks would share
one id.

Handed to the investigation that found the ABI defect, which has the working
environment and far more of this SDK read than I do. One structural question
goes with it: `allgather` needs two disjoint core groups because its consumers
read peers' *payloads*, while a reduction only needs peers' *flags* — every
rank's data is already local once published — so the two-group split may be
the wrong shape here.

## Still open: producer and consumer in one kernel

The full path still hangs while the halves pass separately — push 1.814 µs,
reduction 0.869 µs with `untouched 0`, both correct. So the remaining fault is
in how the two cooperate inside one kernel. `allgather` splits the cores into
two disjoint groups, `core_group_num = aivNum / 2`, with the first half
staging into symmetric memory and the second half reading peers; my version
gives core 0 to the producer and the rest to consumers, which is not the same
partition and leaves one producer core feeding fifteen consumers across
sixteen peers. Aligning fully with the two-group structure is the next step,
and it wants doing properly rather than by another single-line edit — four
such edits tonight produced no information.

## Where this leaves the argument

Nothing here contradicts the case for the one-sided route, and one thing
sharpens it: with the push removed the time did not move, which is the same
result as the size sweep and the wire measurement arriving from a third
direction. The push is not the cost. The remaining work is a competent local
reduction, which is ordinary vector-kernel engineering rather than a question
about the platform.

Established today and unaffected: the wire is 0.31 µs within a module and
0.84 µs across one
([wire](../shmem-install-20261018/README.md#sub-microsecond-and-the-topology-appears));
sixteen PEs form a symmetric domain; and the SDK's own `combine_classic`
performs push plus ordered local reduce at our shape, `-pes 16 -bs 4 -h 6144`,
with `[Combine] check passed`. That example reduces only `bs` tokens with a
scalar loop, which is why its loop shape was a poor model for a full
`[4, 6144]` sum and why copying it cost this detour.
