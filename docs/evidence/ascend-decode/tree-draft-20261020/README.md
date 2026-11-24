# A tree of candidates, and what it is worth

CPU only, no device. `tests/glm52/tree_draft_eval.py` replays one token
stream through three drafters and counts what a target would accept from each.

## The algorithm, and why it is not a list of candidates

Expected tokens accepted a round is `(1 - a^(g+1)) / (1 - a)`, tending to
`1 / (1 - a)`; the measured 7.1 at width 32 puts `a` at 0.86 whose asymptote
is 7.14, so [a longer chain is spent](../five-times-20261020/README.md). Only a
higher `a` helps, and a tree raises it by verifying several continuations in
one pass.

Candidates share prefixes, so the object is a trie and its cost is its node
count, not branches times depth. Under a budget of `B` nodes the choice is an
optimisation: maximise the expected accepted length, which is the sum over
chosen nodes of the probability that the whole path to the node is right,
subject to the chosen set being prefix-closed. A child's probability is its
parent's times a factor no greater than one, so weights fall along every
path, and best-first expansion by that probability is optimal -- the largest
remaining candidate always has its parent already chosen, so the set stays
prefix-closed, and no prefix-closed set of that size sums higher. A priority
queue, `O(B log B)`.

The probabilities come from the context. If the current suffix occurs `m`
times with a token following `c` of them, `c / m` is the maximum-likelihood
estimate, and a suffix automaton over the token stream answers that in
constant time an edge, its suffix links giving the backoff to a shorter and
more frequent context for free. Against the linear drafter's dictionary of
one tuple a suffix a length -- seven hashes of up to eight elements for every
token committed, keeping only the most recent occurrence and so no
frequencies at all -- the automaton is amortised constant time a token, exact
in its counts, and backs off in one hop.

## What it is worth, with the control that matters

The stream is constructed and says so: there is no GLM-5.2 trace on disk, so
the prompt is a real source file from this repository and the output is that
file with a share of its tokens perturbed, which is what an agentic coding
model emits when it edits code it was shown. The edit rate is swept because
the answer is a curve in the copy rate.

The control is the second chain. The tree is allowed to match a suffix of any
length while the engine's drafter caps its tail at 8, so without a chain at
the same cap the comparison would partly measure a longer match rather than a
tree. Tokens accepted a round, averaged over 423 to 5907 rounds:

| edit rate | chain, tail 8 | chain, tail 32 | tree, 32 nodes | tree over long chain |
| --- | --- | --- | --- | --- |
| 0.00 | 29.738 | 32.076 | 32.470 | 1.012 |
| 0.02 | 10.601 | 10.942 | 12.326 | 1.127 |
| 0.05 | 4.967 | 5.026 | 6.500 | 1.293 |
| 0.10 | 2.680 | 2.699 | 3.857 | 1.429 |
| 0.20 | 1.551 | 1.551 | 2.324 | 1.498 |

Verification cost is equal across the three: 32 draft positions through the
model either way.

Two levers, and they pay in opposite regimes. The longer match is worth
29.7 → 32.1 where the copy is near-exact and nothing at all from 0.05 on. The
tree is worth almost nothing on an exact copy -- one chain is already right --
and 1.13 to 1.50 times once the copy is imperfect.

## Where the real workload sits, and what that does to 5x

The engine measures 7.1 accepted at width 32. The long chain reads 10.94 at
edit rate 0.02 and 5.03 at 0.05, so the real stream sits between them, near
0.04. The tree's gain there is 1.2 to 1.3 times, not the 1.75 that `a` rising
to 0.92 would have given.

So the arithmetic of 5x tightens and must be restated: 3x on the round times
1.25x on acceptance is 3.75x, which is 2.9 ms. Still 2.x, with no margin --
where before this measurement the acceptance lever was assumed to carry 1.75
and left 5.25x. The missing margin has to come from the round, and the
largest unattacked piece there is the attention kernel at 5x its byte floor.

## Two mistakes this cost, both of them mine

A floor of two occurrences on the matched suffix. A suffix seen once before is
the whole signal prompt lookup lives on; demanding two pushed the match onto
a shorter suffix whose continuations are many and flat, and a flat
distribution spends the node budget on breadth instead of depth. The tree
accepted 3.9 against the chain's 30.1.

Then a cap on how long a match to trust. A class's `length` in a suffix
automaton is the longest substring it holds, so inside a repeated region it is
the whole region; refusing classes above a length does not shorten the match,
it jumps to a far shorter one and discards the strongest evidence there is.
The tree accepted 6.3 against 30. Only with both removed does best-first
degenerate to a chain when the copy is exact, which is the behaviour the
objective implies.
