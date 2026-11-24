# A chain step costs at most 2.34 ms, so the verify holds 72% of the round

From the two widths measured on the fixed configuration, with no new run.

| | round | acceptance | chained forwards |
| --- | --- | --- | --- |
| width 4 | 48.50 ms | 3.332 | 3 |
| width 8 | 57.84 ms | 4.507 | 7 |

The round grows 9.34 ms for four more chain steps, so a step costs **at most
2.34 ms** — at most, because the verify also goes from four rows to eight and
that difference is charged to the chain here.

The chain at width 8 is therefore no more than 16.3 ms of 57.8, leaving **at
least 41.5 ms, 72% of the round**, for the verify, the target-conditioned
draft pass and the commit.

Against a weight floor of 3.886 ms for the whole round, that is where the
10.4× on non-weight work lives.
[the budget](../round-clock-20261019/README.md)

## And it says the width is not yet optimal

A token bought by widening costs 9.34 / 1.174 = **7.95 ms**. An average token
costs 12.83. While the marginal token is cheaper than the average one,
widening still pays — the optimum is where they meet.

The earlier reading that put the optimum at 8, with width 16 returning to
14.037 ms, was taken on the configuration whose captured width ran a stale
static-kernel package and emitted three token streams. It is not evidence
about the fixed configuration.
[the cache](../w4a8-race-closed-20261019/README.md)

Widths 12 and 16 need `cann_mtp_graph.WIDTHS` extended and `mtp_state.WIDTHS`
already lists them. The round clock and the alternating design make the
comparison cheap, and the prediction is explicit: the round should grow about
2.34 ms a step while acceptance grows, until the marginal token reaches 12.8.

## What this predicts for the phase decomposition

The phase hook splits a round into seed, chain, verify, conditioned and
`host_tokens`. If it reports a chain near 16 ms and a verify plus commit near
41, the two methods agree and the decomposition is sound. A chain much larger
than 16 would mean the verify's row count, not the chain's depth, explains the
width difference — and would move the target.
