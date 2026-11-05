# Acceptance at every width, from one run's own record

`width-sweep-20261019` bought four acceptance points with four runs. They were
already in the first one.

## The identity

The draft chain is autoregressive and deterministic given its state, so a
narrower width drafts a prefix of what the captured width drafted, and the
target accepts a prefix of that prefix. A round that matched `k` drafts at the
captured width would have matched `min(k, w - 1)` at width `w`:

    acceptance(w) = 1 + mean(min(matched_drafts, w - 1))

The leading 1 is the token the target produces at the anchor position, which
arrives whether or not any draft survives. `matched_drafts` is already one
field per round in `session.accepted_spans` of every `rank-0.json`.

## It reproduces the sweep to 1.1%, with no run

    python3 tests/glm52/accept_curve.py /workspace/experiments/conditioned-20261019/on-{1,2,3}

| width | predicted from the width-8 record | measured in its own arm |
| --- | --- | --- |
| 4 | 3.3700 | 3.3322 |
| 6 | 4.0969 | 4.1250 |
| 8 | 4.5110 | 4.5066 |

The three arms of `conditioned-20261019` give bit-identical histograms, which
is the determinism check the width-8 trajectory gate also reports.

Widths the sweep never ran: 2 → 1.9119, 3 → 2.7181, 5 → 3.8458, 7 → 4.3216.
Against the round floors of `width-sweep-20261019`, width 7 reads 11.95 ms
against width 8's 12.05 — inside the 0.163 ms between-run floor. Width 8
stands, and now it stands without a sweep.

## The histogram is bimodal, which no per-position decay can produce

681 rounds, three measured requests of the width-8 arm:

| matched drafts | rounds | share |
| --- | --- | --- |
| 0 | 60 | 8.81% |
| 1 | 72 | 10.57% |
| 2 | 105 | 15.42% |
| 3 | 120 | 17.62% |
| 4 | 153 | 22.47% |
| 5 | 18 | 2.64% |
| 6 | 24 | 3.52% |
| 7 | 129 | 18.94% |

A chain whose per-position acceptance decays monotonically gives a
monotonically falling histogram. This one peaks at 4, collapses at 5 and 6,
and then puts 18.94% at full acceptance. So this trajectory has two regimes:
rounds that die by the fourth draft, and rounds the draft gets entirely right.

That matters for anything that reasons about the marginal draft position. The
marginal token at width 8 is 0.1894, but it is not the tail of a decay — it is
the 18.94% of rounds that would have matched everything, truncated. Widening
past 8 keeps paying for them: the sweep's own A(12)=4.871 and A(16)=5.356
are that mode extending.

## Scope

One trajectory, one prompt, 227 rounds a request. The published anchors for
this model class are lower — GLM-5's own paper reports accept length 2.76 at
four speculative steps on a private prompt set, and FastMTP measures a
recursively reused MTP head saturating near 1.8 — so 4.507 is a property of
this trajectory as much as of the checkpoint. The 2 ms target divides by it.
