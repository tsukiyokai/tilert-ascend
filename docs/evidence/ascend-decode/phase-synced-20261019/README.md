# The round, decomposed with each phase owning its device time

Phase hook with a device barrier before each timer stops, on the configuration
holding the best deterministic point: width 8, three gmm1 patches, conditioned
graph, cache fix. 1363 rounds. `status=measured`, no rank errors — and the
first zero exit this harness has produced.

| phase | p10 a call | calls | a round | share |
| --- | --- | --- | --- | --- |
| `verify` | 40.080 | 1 | 40.080 | 75.9% |
| `chain` | 1.731 | 6 | 10.386 | 19.7% |
| `conditioned` | 2.080 | 1 | 2.080 | 3.9% |
| `host_tokens` | 0.263 | 1 | 0.263 | 0.5% |
| | | | **52.809** | |

The sum is 95% of this run's round. The barrier costs about 0.5 ms a token —
TPOT reads 12.32 here against 11.77 without it — so the shares are right and
the absolute figures are slightly inflated.

## It confirms the weight-floor inference exactly

Without the barrier the hook read `verify` 2.196 and `conditioned` 42.877. The
weight floor said 2.196 was impossible for a forward whose weight read alone
is 3.886, so it had to be dispatch and the next phase had to be absorbing the
device work. With the barrier: `verify` 40.080, `conditioned` 2.080. Both
halves of that inference were right.
[the inference](../phase-split-20261019/README.md)

## Where the 2 ms has to come from

A round may cost 9.013 ms at width 8's acceptance of 4.507. The verify alone
is 4.4× that, and 10.3× the round's whole weight floor — the same 10.4× the
budget arithmetic gives for non-weight work, now measured on the operator that
carries it.

| | now | needed |
| --- | --- | --- |
| `verify` | 40.080 | ~4 |
| `chain` | 10.386 | ~1 |

## And the chain is a different problem from the verify

A chain step is one MTP layer at one row: 1.731 ms. The verify is the whole
model at eight rows: 40.080 over roughly 80 layers, 0.501 ms a layer. So a
single-row layer costs **3.5× an eight-row one**, which means the chain is
latency bound and the verify is not — widening the chain's rows would cost it
almost nothing, and shortening it saves almost nothing per row.

That splits the target in two. The verify needs its per-layer work to stop
costting 10× the bytes it moves, which is the megakernel argument and is where
the cube-to-vector GM round trip lives. The chain needs its six serial
single-row forwards to stop paying a fixed cost six times — a different
mechanism, and one speculative decoding's structure may allow batching.
