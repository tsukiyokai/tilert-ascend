# Sizing the proposal by the last acceptance buys 6 to 9 percent of TPOT

2026-10-21, CPU, 34 source files from this repository, width ceiling 32.

| edit rate | automaton @32 | adaptive | mean width | TPOT @32 | TPOT adaptive |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 10.020 | 8.318 | 12.6 | 7.18 | **6.75** |
| 0.05 | 5.434 | 4.362 | 8.1 | 13.24 | **12.02** |

The adaptive column accepts less and costs less. A round is 45.88 ms plus
0.814 a proposed row, so thirty-two rows to have eight accepted pays for
twenty-four that were never going to hold; twice the last round's acceptance
keeps 83 percent of the acceptance for 39 percent of the width.

## Two signals tried, one wrong

The first attempt sized the proposal by `matched_length` -- how long a suffix
the automaton found a continuation for -- on the reasoning that a long match
means a copy stretch. It measured worse at every edit rate, and the mean width
column said why: the matched length is typically under nine on this corpus, so
a threshold at eight almost never fired. The automaton's advantage does not
come from long matches; it comes from the most frequent continuation of a
short one. That was a wrong account of its own mechanism.

## And one measurement construction that had to be fixed

Scoring the adaptive column inside the existing loop made it worse still,
because every column there shares one advance -- `step` is the best column's
acceptance -- so a column that proposed four rows was advanced ten and its
feedback loop locked at the narrowest width. It needs its own trajectory,
which doubles the evaluator's cost and is why the sweep now runs two edit
rates rather than four.

## Where it sits

Orthogonal to the tree. The tree raises acceptance (14.510 at 0.02) and needs
a per-row attention mask the kernel does not take; this lowers the width's cost
and needs nothing. Both are needed: engine work reaches about 40 ms a round,
and 2.9 ms at 40 ms needs 13.8 accepted, which the tree reaches at a low edit
rate while this makes each round cheaper.

## The factor, and a baseline the change had quietly rewritten

Sweeping the multiplier at an edit rate of 0.05:

    factor 1.5   acceptance 4.031   mean width  6.1   TPOT 12.62
    factor 2     acceptance 4.369   mean width  8.1   TPOT 12.01
    factor 3     acceptance 4.369   mean width 10.9   TPOT 12.54
    factor 4     acceptance 4.369   mean width 13.2   TPOT 12.97

Two is best, and the shape is right: past three the acceptance is capped at
4.369 while the width keeps growing, and at 1.5 the proposal is too short to
hold what the stream offers.

The sweep also exposed a regression of its own making. The adaptive ceiling
was hardcoded inside `propose`, so the fixed-width column that was supposed to
be the baseline fell from 5.393 to 2.896 -- the comparison was against a
contaminated control. It is now a constructor argument, on by default and off
for a caller that needs a fixed row count, such as a graph with static token
buckets.

Final figures at 0.05: fixed width 32 accepts 5.393 for a TPOT of 13.34, and
adaptive accepts 4.351 at a mean width of 8.0 for 12.05 -- 9.7 percent.

## The tree with an adaptive budget is the best configuration measured

Each column on its own trajectory, width ceiling 32, budget ceiling 64:

| edit rate | automaton @32 | TPOT | adaptive chain | TPOT | adaptive tree | nodes | TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 10.063 | 7.15 | 8.385 | 6.71 | 9.288 | 13.1 | **6.09** |
| 0.05 | 5.436 | 13.23 | 4.358 | 12.03 | 4.731 | 7.4 | **10.97** |

15 and 17 percent better than the fixed-width automaton, and better than the
adaptive chain, by the same mechanism twice over: the tree raises what a round
accepts, and sizing the budget by the last acceptance stops paying for
verification rows that were not going to hold.

The fixed-budget tree accepts more -- 14.647 at 0.02 -- and has a worse TPOT,
because 64 verification rows cost 52 ms by the round model. The ceiling is not
a quota, on a tree as on a chain.
