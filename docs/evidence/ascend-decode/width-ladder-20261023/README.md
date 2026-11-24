# The width ladder, recovered: 12.41 ms at width 32

Launched on a3-180 (7.150.10.180) on 2026-10-23, one fresh container an arm,
lookup drafter, `sfa` attention, `--target-graph --compile-target
--super-kernel --fp32-head`, 1024 output tokens a request, 3 warmup and 3
measured. The host revoked login while the ladder ran and the records were
written where nobody read them. Login works again on 2026-10-24 and the three
summaries were intact; `summary.json` per arm is beside this file.

| width | rounds a request | accepted a round | TPOT ms | round ms |
| --- | --- | --- | --- | --- |
| 16 | 294 | 3.483 | 17.132 | 59.67 |
| 24 | 322 | 3.180 | 20.108 | 63.95 |
| 32 | 179 | 5.721 | 12.413 | 71.01 |

`accepted = 1024 / rounds`; `round = TPOT x accepted`, and the four phase
timers sum to within 2% of it, so the two readings are the same round.

Two things follow, and they are not the same thing.

Width 32 is the project's best TPOT: 12.413 ms against 17.073 the day before,
27% better. It buys that by raising acceptance 1.64x (3.483 to 5.721) for 19%
more round time. The marginal row costs 0.709 ms between 16 and 32, against
the 0.814 ms the old fit charged and the 0.25 ms measured between 12 and 16.

Acceptance is not monotone in width. Width 24 accepts *less* than 16 (3.180
against 3.483) while paying for eight more rows, so it loses twice and reads
20.108 ms. A wider round offering strictly more candidates cannot accept
fewer unless the tree shape at 24 is worse -- the budget is spent on nodes
whose ancestry the target rejects. That is a host-side property of
`propose_tree(count)` and measurable on a CPU with a deterministic target, so
the next step is the acceptance curve over budgets 8..64 without a device.
