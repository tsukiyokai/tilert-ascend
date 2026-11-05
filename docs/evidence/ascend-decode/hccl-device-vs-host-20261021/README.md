# A collective costs 10 microseconds on the device; the rest is the host

The repository already held `tests/a3-calibration/hccl_latency.cpp`, a native
single-host HCCL calibration with no results page and no lesson citing it. It
builds with `build_hccl.sh` and runs with `--expansion aiv`. Sixteen devices,
AIV, 100 batches:

| case | bytes | ops a unit | device p50 | host wall | host release |
| --- | --- | --- | --- | --- | --- |
| independent_12kib | 12288 | 1 | **10.74** | 382.94 | 235.27 |
| chain_12kib | 12288 | 156 | 1564.14 | 5054.61 | 1783.35 |
| independent_48kib | 49152 | 1 | **15.21** | 355.22 | 211.14 |
| chain_48kib | 49152 | 156 | 2162.32 | 5564.41 | 2360.76 |

## Which overturns today's attribution

A single collective of 12 KB takes **10.74 us on the device** and 382.94 us of
host wall time. The 183 to 296 us measured through torch is therefore almost
entirely host-side issue and synchronisation, not a device handshake.

This project said the opposite earlier today: that the floor is "neither
bandwidth nor participants nor host dispatch", on the grounds that a kernel's
issue under graph replay is 1.064 us. That reasoning used the dispatch cost of
a *compute* kernel. A collective goes through a different host path, and the
native numbers separate them.

## And the chain shows the host cost amortises

156 collectives in a chain cost 1564.14 us on the device -- 10.0 us each,
matching the single call. But the host wall is 5054.61 for the chain, which is
**32.4 us a call** against 382.94 for one issued alone.

So the host overhead falls twelvefold when calls are issued back to back
without waiting between them.

## Which reverses the direction

The round's 374 us a layer of collective time is about two calls. On the device
that is 20 us. The rest is host path, and the host path is twelve times cheaper
per call when the calls are not serialised by a wait.

So the lever is not fewer calls -- it is **not waiting between them**. Whether
the engine's decode already issues its collectives back to back, or inserts a
synchronisation each time, is the measurement that follows, and it is a reading
of the engine rather than a sweep.

## Correction: the engine's own collectives are 858 us on the device, not 10

From `profile-rope-20261018`'s `kernel_details.csv`, rank 1:

    hcom     64 kernels    54.94 ms    858.36 us each
    other  5544 kernels   179.14 ms     32.31 us each
      hcom_allGather  34 calls   882.45 us each
      hcom_allReduce  30 calls   831.06 us each

Eighty times the native calibration's 10.74 us. So the reversal above -- that
the cost is host-side and amortises when calls are not serialised -- does not
follow for the engine. Its collectives cost that much *on the device*.

## And the spread says there is a straggler, without separating it

The same count of calls on every rank, with very different totals:

    rank  1    64 calls    54.94 ms    858 us each
    rank 14    64 calls   140.18 ms   2190 us each
    rank 12    64 calls   219.14 ms   3424 us each
    spread 54.94 to 219.14 ms, ratio 3.99

A collective's Duration includes time blocked on peers, so a fourfold spread
means arrival times differ widely. But the fastest rank still reports 858 us a
call, and a collective synchronises everyone -- so even the earliest arrival
waits for the last. The spread therefore establishes that a straggler exists
and does **not** isolate what the operation itself costs.

## Where this leaves the 374 us a layer

Unattributed again, honestly. Three readings have now been tried: torch-level
per-call cost (183 to 296 us, which the native tool shows is mostly host), the
native device cost (10.74 us, which the engine does not match), and the
engine's own profile (858 us a call, which includes peer waiting). None of them
is a measurement of the work.

What would separate them is a capture with per-rank arrival timestamps at each
collective, which is what `--arrival` style instrumentation gives and this
profile does not carry. That is the next measurement, and it needs the machine
the arms are queued on.

## Arrival skew is not the explanation either

Sixteen ranks under AIV with determinism, a barrier before each measurement, a
skew step being one 6144x2048 matmul at roughly 30 us:

| case | rank 0 | slowest rank |
| --- | --- | --- |
| aligned, no extra work | 220.59 | 231.51 |
| aligned, four steps each | 252.32 | 290.21 |
| rank 0 idle, others four | 265.80 | 274.79 |
| last rank sixteen, others none | 360.76 | 383.63 |

One rank doing sixteen extra matmuls -- about 480 us of work nobody else does
-- raises the slowest total from 231.51 to 383.63, which is 152 us. So the
collective **absorbs most of the skew** rather than serialising behind it: time
the early arrivals spend waiting is not time added to the round.

Which removes arrival skew as the explanation for the engine's 858 us a call,
and leaves the three incompatible readings standing. What is excluded now:
host-side issue (the engine runs in a graph), the native device cost (80x
smaller), and arrival skew (absorbed).

The remaining candidate is message size: the engine's collectives carry whole
hidden states across sixteen ranks for a MoE all-gather, which is a different
shape from the 12 to 576 KB swept here. That is measurable without the engine
-- the MoE's gather is rows x hidden x world, and at twelve rows that is 2.3 MB
-- and it is the next thing to try rather than another reading of the same
profile.

## Message size is excluded too, which indicts the profile rather than the engine

Sixteen ranks under AIV with determinism:

| rows | sent MB | gathered MB | all_reduce | all_gather |
| --- | --- | --- | --- | --- |
| 12 | 0.15 | 2.36 | 186.96 | 178.52 |
| 48 | 0.59 | 9.44 | 187.09 | 395.88 |
| 96 | 1.18 | 18.87 | 202.50 | 430.76 |
| 192 | 2.36 | 37.75 | 222.63 | 532.83 |
| 384 | 4.72 | 75.50 | 288.82 | 758.60 |

Reaching the profile's 831 us an allReduce would need far more than 384 rows,
and its 882 us an allGather about 450. A decode round has twelve.

So all four explanations are excluded: host-side issue, the native device cost,
arrival skew, and message size.

## Which leaves the profile itself

`profile-rope-20261018` carries `max_tokens: 6, mtp_width: 4` -- a configuration
this project already ruled void for steady-state decode accounting, in the
correction that voided a whole day's device-time budget. Its collectives are
therefore prefill-sized and include the domain build, which is exactly what a
previous page found when the by-type average folded a one-off into a per-round
figure.

So the 858 us was never a decode figure, and two of today's reversals leaned on
it. The honest state: at decode shapes a sixteen-rank collective is 177 to 187
us through torch and 10.74 on the device, the engine's share of that is
unmeasured because no steady-state capture exists, and the 374 us a layer
residual has no attribution.

The queued steady-state capture is not one more experiment among several. It is
the only thing that can close this.
