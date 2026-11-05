# Context-copy drafting

A lookup drafter proposes a continuation already present in the prompt or
committed output. The target verifies it causally and commits the matching
prefix plus its correction or bonus token. Width M permits M−1 proposed
tokens and up to M committed tokens; the output budget clips the final round.

## Exact CPU replay, 2026-10-19

One frozen 8192-token prompt and 1024-token output, shared by three identical
measured requests. Lookup searches the longest matching suffix first and the
most recent earlier occurrence second. Only observed continuations are used;
short proposals use the deployed pending-token padding.

| Width | Minimum suffix 2: rounds / accepted per round | Minimum suffix 1: rounds / accepted per round |
| --- | --- | --- |
| 8 | 253 / 4.0435 | 228 / 4.4868 |
| 16 | 198 / 5.1667 | 171 / 5.9825 |
| 32 | 174 / 5.8793 | 147 / 6.9592 |

The maximum suffix remains eight tokens. At width 32, the one-token fallback
reduces rounds by 15.5%. Of its 62 unigram rounds, 27 match at least one
proposed token. A repeated suffix and a correct continuation are separate
metrics: minimum-suffix 2 has 68 rounds without a repeat and 105 without a
correct draft match; minimum-suffix1 has 6 and 60 respectively.

Prompt-only / generation-only / combined width 32 acceptance is
1.8634 / 5.0644 / 5.8793 for minimum-suffix 2 and
2.1180 / 6.0176 / 6.9592 for minimum-suffix1. Including the prompt helps this
input; it is not a lower-bound theorem, because a new longer match can displace
a better shorter one.

[Raw policy summary and input hashes](../lookup-contract-20261019/policy-summary.json)
include all 18 comparisons. These parameters have not been evaluated on held-out
inputs. The earlier 5.885 figure counted a bonus beyond the output budget.

## Implementation and cost

`LookupDrafter` maintains an append-only suffix index. Entries only refer to
occurrences with a known next token, and each key keeps its most recent such
occurrence. The query is independent of context length for the fixed eight-token
maximum. On the server CPU, the indexed implementation preserves every proposal
and acceptance boundary and reduces total lookup work from about 1.03s to
20.6–20.9ms per request, including index construction.
[Paired CPU measurement](../lookup-contract-20261019/README.md)

The device path skips the MTP chain and conditioned pass. Its MTP cache remains
at the injected P−1 rows; only target KV advances. `--lookup-unigram` selects
the one-token fallback for an independent comparison; the default remains two.

The next device result must report actual target round cost, expert union,
accepted count and request TPOT at widths8/16/32. Width 32 may touch up to256
experts, so a 64-expert measurement is not a worst-case bound. Previous
7–8ms TPOT extrapolations are withdrawn pending this measurement.

## Hybrid scope

A merged lookup/MTP policy must advance along its own accepted boundaries.
Recorded MTP proposals at another policy's boundaries do not provide an oracle
at the newly reached positions; predictions after a rejected draft belong to
that rejected context. Two branches also need correct RoPE depths, indexer
visibility, attention visibility and selected-path KV compaction. Returning to
MTP after lookup requires filling the missing conditioned MTP rows from genuine
target hidden states and successor tokens.

## Verified-copy cursor, CPU rejection

A cursor may continue an earlier source only after every real proposal matched
and its observed bonus equals the new pending token. Padding and budget
clipping cannot establish that state. This differs from choosing the most
recent suffix occurrence, but it does not improve the frozen M32 replay:
147 rounds and4704 rows with either policy. M8 stays228 rounds; M16 improves
171→170 only. Keep the deployed policy unchanged.
[CPU source, invariants and raw result](../lookup-contract-20261019/cursor/result.json).
