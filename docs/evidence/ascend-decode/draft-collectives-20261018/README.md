# The draft path pays 46× more wait per reduction than verify

From the raw per-rank analysis of the archived capture,
`/home/shanshan/trt-asc/offline/cann-mtp-profile-20261017-01/analysis/rank-0.json`
on `a3-gs`, which carries fields the published summary aggregates away.

## The exporter's own four-way split

That file gives `activity_union_ms` alongside `compute_union_ms`,
`collective_union_ms` and `gap_in_device_span_ms`, and its `scope` defines
the gap as "No traced kernel/HCCL/AICPU interval". So a stage's device span
decomposes without subtraction. Rank 0, round 2, in ms:

| stage | span | compute | collective | AICPU residual | idle | collective calls |
| --- | --- | --- | --- | --- | --- | --- |
| `chain`@8195 | 8.150 | 2.413 | 3.861 | 0.000 | 1.876 | 4 |
| `chain`@8196 | 7.901 | 2.383 | 3.546 | 0.000 | 1.972 | 4 |
| `verify`@8195 | 48.553 | 40.698 | 5.313 | 1.528 | 1.014 | 158 |
| `conditioned`@8195 | 8.128 | 2.544 | 3.489 | 0.000 | 2.096 | 4 |
| round 2 total | 72.732 | 48.037 | 16.210 | 1.528 | 6.958 | 170 |

Non-compute is 24.695 ms: 65.6% collective envelope, 28.2% genuinely idle
device, 6.2% AICPU outside both. An earlier reading of this stage table
subtracted compute from span and called the remainder non-compute, which
merged three different things; verify's truly idle device time is 1.014 ms,
not the 7.855 ms that subtraction gives.

## Per call, the draft path is the outlier

Each draft stage has exactly four collectives: three `hcom_allReduce_` and
one `hcom_allGather_`. Their individual envelopes in `chain`@8195 are 0.723,
1.248 and 1.656 ms, plus 0.234 ms for the gather, and they escalate. Of the
stage's 3.861 ms union, `hccl_wait_sum_ms` is 3.703 (95.9%) and
`hccl_transit_sum_ms` is 0.024 (0.6%).

Verify runs 157 reductions and one gather in 5.313 ms: wait 4.100, transit
1.213. That is 33.8 µs per call and about 26 µs of wait per call.

So on the same TP16 domain, a reduction inside verify waits 26 µs and a
reduction inside a draft stage waits about 1234 µs — a factor of 46. Across
the round, nine draft reductions spend roughly 11 ms in Notify_Wait while
moving 72 µs of data. That is 15% of the 72.7 ms round and none of it is
work.

The mechanism is consistent with the timings and needs no new assumption.
Verify's 157 reductions are pipelined: every rank reaches reduction k having
just finished layer k, and the 0.26 ms of compute between reductions keeps
sixteen ranks in step. A draft stage has three reductions with about 0.8 ms
of compute between them and begins immediately after a host-side Gloo
broadcast, so the ranks enter the first reduction carrying whatever skew the
broadcast left and pay it three times. The escalation from 0.723 to 1.656 ms
is the signature of one consistently late rank plus accumulated delay. This
also explains the measured 5.78 ms per round that `--draft-graph` earns
([attribution](../ablate-20261018/README.md)): graphing removes the per-op
host launches between those reductions.

## A candidate that this measurement makes worth costing

The three reductions exist because the single MTP layer is TP-sharded like
the other 78. Replicating only the parts whose shard needs a reduction
removes all three from every draft stage, and the cost is now derived rather
than sketched. From the draft weight components in
[runtime-traffic.json](../runtime-traffic.json), the sharded parts are the
attention shards (`q_b_proj` 2,101,248 + `W_UK` 786,432 + `W_UV` 1,048,576 +
`o_proj` 6,303,744 = 10,240,000 B; `q_a_proj`, `kv_a_proj` and the norms are
already replicated), the routed experts at 10,051,584 B and `shared_ffn` at
2,372,096 B. Replicating those three multiplies them by sixteen, adding
339,955,200 B — 324.2 MiB — per rank per draft step, which at 1138 GB/s
([GMM re-measurement](../gmm-l2-20261018/README.md)) is 299 µs per step and
0.90 ms per round. The LM head stays vocab-sharded: its AllGather is the
cheap collective at 0.234 ms, and replicating it would mean reading the whole
154,880 × 6144 BF16 table.

Against that, the upper bound on the saving is the three stages' collective
envelopes less their AllGathers, 10.20 ms per round, so the net upper bound
is 9.30 ms. Taken in full that would move the round from 57.12 to 47.82 ms
and TPOT to 13.93 ms. The extra 324.2 MiB per die is resident, not per step,
and fits the plan's 48 GiB per-die decode budget.

Two reasons the bound is an upper bound. It comes from a capture taken before
`--draft-graph`, and that flag already earns 5.78 ms per round
([attribution](../ablate-20261018/README.md)) most plausibly by shrinking
these same envelopes. And replication changes the arithmetic — a single-rank
accumulate replaces a sixteen-way FP32 reduction — so the trajectory and
acceptance would move and need their own validation.
