# What the workload's repetition allows

For each position in the engine's own 1024-token output, the longest run
starting there that appears verbatim earlier -- in the 8192-token prompt or in
the output so far.

    a run of at least  1 token:   99.4% of positions
    a run of at least  2 tokens:  93.2%
    a run of at least  4 tokens:  83.8%
    a run of at least  8 tokens:  72.4%
    a run of at least 16 tokens:  57.9%
    a run of at least 32 tokens:  37.2%

An oracle copy drafter -- one that always finds the longest prior run, knowing
where to look and how far it goes -- accepts **9.660** a round at a budget of
sixteen. With the fixed term the measured merges leave, 30.15 ms, that is
**TPOT 4.21 ms**.

    perfect predictor, budget 16      17.000 accepted   2.39 ms
    oracle copy drafter, budget 16     9.660            4.21 ms
    tree, budget 16, branches 4        6.649            6.12 ms
    lookup at the engine's width 12    4.900           10.98 ms

## What this settles about the target

2.9 ms needs about 13 accepted at a sixteen-row budget. The oracle copy drafter
gives 9.660. So **no copy-based drafter reaches the target on this workload**,
however well implemented -- the limit is how much of the output repeats
something already seen, not how well a drafter finds it.

The tree is at 69 percent of the copy oracle. Closing that gap is worth 6.12 ms
to about 4.7, which is real and is the largest drafting lever left, but it does
not reach 2.x.

Reaching 2.x needs prediction rather than recall: a drafter that is right about
the 42 percent of positions where nothing long enough has been seen before.
That is what a trained head does, and the published figures for this
checkpoint's MTP head -- mean accepted length 1.85 at k=1, 2.6 to 3.9 at k=3 to
5 -- are well below both the tree and the oracle, so it is not that head
either.

## Which reframes the goal

With the fixed term at 30.15 ms, 2.9 ms needs 13 accepted and the workload
offers 9.66 by recall. The remaining ways are:

- cut the fixed term below 30.15 -- the boundary-merging sweep over adjacent
  operator pairs was exhaustive, so this needs a different mechanism;
- or accept more per round than recall allows, which needs a predictor better
  than this checkpoint's own MTP head.

Both are open, and this page is the first measurement that says which of the
two the target actually requires.

## The oracle's own budget curve, and what it implies

    budget   accepted   TPOT ms
         4      4.414      7.43
         8      6.872      5.16
        12      8.393      4.54
        16      9.660      4.21
        24     11.011      4.18   <- floor
        32     11.907      4.31
        48     12.962      4.77
        64     13.474      5.37
        96     13.474      6.94

Acceptance saturates at 13.474 by budget 64 and does not move at 96, so that is
how long the stream's copyable runs go. The floor is 4.18 ms at budget 24,
squeezed between saturating acceptance and a row that costs 0.66 ms.

## Which identifies the one path left for copy-based drafting

13.474 accepted is almost exactly what 2.9 ms needs -- but at budget 64, where
42.2 ms of row cost puts TPOT at 5.37. So the binding constraint is not the
workload after all; it is the row.

    row cost 0.66 ms, budget 64   TPOT 5.37 ms
    row cost 0.20 ms, budget 64   TPOT 3.19 ms
    row cost 0.10 ms, budget 64   TPOT 2.72 ms

So 2.x is reachable by recall alone if a verified row costs about 0.1 ms
instead of 0.66. And the row cost has been measured today: it is the graph's,
rising monotonically with the captured width while the same operators are flat
in rows eagerly. A graph pays for rows that carry nothing.

That makes the question concrete and it is not a modelling question: **why does
graph replay charge for empty rows, when the operators do not?** The synthetic
measurement -- eager 568.7 to 554.4 us across a fourfold row range, replay
527.0 to 582.9 -- says the cost is in capture rather than in execution, which
is where to look.
