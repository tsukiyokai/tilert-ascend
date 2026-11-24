# The round, accounted for, and three of my claims withdrawn

An exhaustive decomposition of a profiled round from
`raw-rank-0/*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`, with the
host floor measured separately profiler-off on one die.

## What I had wrong

**There is no 49.92 ms profiled round.** The three
`mtp_round.py(74): advance` events last 195654.350, 133920.010 and
133934.610 µs. The profiled steady round is 133.93 ms; 49.92 came from a
profiler-off wall clock. So "operators sum to 42.5 of 49.07, leaving 7.4 ms"
took its numerator from a profiled capture and its denominator from an
unprofiled one. Recounted, the non-operator residual is 10.1–10.4 ms.

**`aclrtSynchronizeStream` at 24.255 ms over 15 calls is not a loss.** For
all thirteen calls over 1000 µs across three rounds, the sync returns while
the device is still busy and the next device task starts 2.164–22.766 µs
later. It drains only the stream holding that one D2H; streams 80, 81 and 13
keep running. The figure measures how long the host blocks while the device
works, which is not time anyone pays.

**4.32 ms/round of the operator table is not operator time.** The
`SuperKernel`/`MIX_AIV` row (count 858, total 35199.677 µs) contains
`sk_1277_…all_reduce…` at 2837.06/6047.15/4183.38 µs per instance, while its
156 same-named siblings average 36.0 µs with a cross-rank minimum of 30.5.
The entry reduction is absorbing rendezvous skew and it is booked as compute.
Real transferable device work is 37.7 ms, not 42.5.

## What is confirmed, independently

Zero overlap is exact, not approximate. Summing device task durations and
taking the union of their intervals give 41721.140/41721.140,
43885.451/43885.451 and 41766.035/41766.035 µs — a difference of 0.000 µs —
and compute against communication also overlaps by 0.000. Across seven
streams no two device tasks share a microsecond. `op_statistic.csv`'s
127541.996 µs total equals the `Overlap Analysis`/`Computing` events and the
`step_trace_time.csv` column, so nothing is double counted.

Rendezvous, not transfer, re-confirmed from the capture itself: for the same
768-byte `allgatherAivKernel`, the fastest rank's `dur` is 9.080 µs and the
slowest 6304.475. A 96 KiB all_reduce ranges 82.828 to 19960.837 µs, 241×.
Stability tracks whether ranks enter in step, not message size.

## The host floor, which is larger than the weight floor

Measured profiler-off on device 7, one process:

| item | ms/round |
| --- | --- |
| AscendCL launch work, 522 launch-class API calls × 10.542 µs | 5.503 |
| the one remaining host token read | 0.335 |
| `MtpRoundExecutor.advance` Python control flow | 0.023 |
| total beyond device-side reach | **5.861** |

That is 85.3% of the 6.87 ms a round may cost at acceptance 3.433, and larger
than the 3.56 ms weight stream. At least 0.404 ms is strictly exposed: the
eager region's 154 tasks occupy 1.220 ms of device time against 1.624 ms of
host dispatch, so the device cannot cover its own launches.

The fix is graph capture of the eager region, which is host-side work. No
kernel, however good, reaches it.

## What device-side work does reach

The 156 in-group reductions cost 5.635 ms/round and have a floor near
1.57 ms at 10 µs per call, so about 4.1 ms is addressable there. The 19
standalone collectives are already within 0.866 ms of the hardware. That is
the size of the prize for the one-sided route — real, and bounded.

## Rank skew

The host enters `advance` with about 1.2 ms of spread; by the first
collective that is 4424–11028 µs, amplified 3.87–8.36× inside a window that
is 98.94–99.60% device-idle. Pre-collective device busy time differs by only
7–9 µs across ranks, so the skew is not computational. A per-rank component
is real but small (η² 0.520 against a permutation null of 0.334, p=0.033);
the largest excursions are single-round events. Exit spread is 21–65 µs
against the arrival spread, a ratio of 0.003–0.015, so the first collective
absorbs it.

## A dead instrument

Under `HCCL_OP_EXPANSION_MODE=AIV`, `communication.json` reports
`Transit Time`, `Wait Time` and `Synchronization Time` all 0.000, with the
whole elapse in `Idle Time`. AIV expansion erases HCCL's internal phase
breakdown, which is why an earlier attempt to name these phases found nothing.
