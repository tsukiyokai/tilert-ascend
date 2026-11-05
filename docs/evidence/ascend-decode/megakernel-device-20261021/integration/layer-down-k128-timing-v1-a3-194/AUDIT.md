# K128 Down in the local layer

The K128 candidate passes the numerical comparison but does not establish a
consistent timing improvement over N512 in this workload. It is not selected
for the local layer.

Both arms use the same `609c973e` binary, writable addresses and Top-K copy.
Only the two immutable `body.scratch[7]` tiling pointers differ. The control
uses M64/N512/K128; the candidate uses M32/N1024/K128.

| Allocation arena | Control median (us) | Candidate median (us) | Paired saving median (us) | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| 0 | 997.219 | 998.332 | -1.142 | 6/30 |
| 1 | 1001.240 | 1000.403 | 0.743 | 23/30 |

Time is per indexed/shared two-layer pair at M12, 24 mixed groups, on a3-194
physical die 4. One bank holds two synthetic Decoder weight sets. Each graph
contains 16 pairs; each event contains 30 graph replays. Each arena has 30
alternating candidate/control pairs, with three warm replays before each
event. The profiler is disabled. Copies and output poisoning occur outside
the timed events. Both TP reductions are identities.

All 6,660 numerical checks, 600 event-tail checks, 844 complete 200-byte Down
records and 388 complete host-record comparisons pass. The four corrected
NZ weight storages keep identical bytes and addresses with zero extra HBM.
The byte checks include the immediate two-layer successor, all active-expert
table words, changing inputs, and post-timing results.

[audit.py](audit.py) recomputes the results from event totals and checks the
frozen fixture, complete records, numerical boundaries and rank initialization.
The [summary](summary.json) contains exact medians and hashes. The complete
[device report](layer-down-k128-timing-v1/local-chain.json) and per-arena reports
retain p50/p90/p99, standard deviation and every event sample. The
[launch command](command.json), [idle admission samples](admission.jsonl),
telemetry and runtime logs are retained beside them.

This is a one-bank local-layer comparison. It does not measure real checkpoint
routing, a multilayer weight working set, TP16 communication or end-to-end TPOT.
The earlier 16-bank standalone FFN result uses a different workload and baseline.
