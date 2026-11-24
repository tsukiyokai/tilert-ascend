# The tree's ceiling on the engine's own stream

Budget and branching swept against the capture case's 8192-token prompt and the
1024 tokens a real run produced, with TPOT from the fixed term the measured
merges leave -- 45.88 - 15.73 = 30.15 ms -- plus 0.66 a verified row.

| budget | branches | accepted | TPOT ms |
| --- | --- | --- | --- |
| 16 | 2 | 6.564 | 6.20 |
| **16** | **4** | **6.649** | **6.12** |
| 24 | 4 | 7.367 | 6.24 |
| 32 | 4 | 7.877 | 6.51 |
| 48 | 8 | 8.393 | 7.37 |
| 64 | 8 | 8.678 | 8.34 |

## Two findings

**The optimum is small.** Acceptance climbs monotonically to 8.678 at budget 64,
and TPOT gets worse past 16, because 0.66 ms a row outruns the acceptance it
buys. The best configuration is a sixteen-node tree with four branches.

**Branching barely matters.** Two, four and eight branches differ by one to two
percent at every budget. On this stream depth beats breadth: what follows an
edit is found by matching a longer context, not by covering the edit with more
candidates. That is consistent with the automaton -- a single path with the
longest match -- being within 10 percent of the tree.

## What this bounds

    fixed term after every measured merge      30.15 ms
    best drafting configuration                6.649 accepted at 16 rows
    TPOT                                       6.12 ms

So with everything measured on both sides, the floor is 6.12 ms. Reaching 2.9
would need acceptance near 13 at a sixteen-row budget, and the sweep shows the
stream does not offer it -- 8.678 is the most any configuration extracts, and
that one costs more in rows than it returns.

This is the second time today the target has been bounded out of the identified
mechanisms, and the first time with measured data on both sides: the engine
side from single-die ratios under identical conditions, the drafting side from
the engine's own output.

## What that leaves

Either a mechanism that cuts the 30.15 ms fixed term further -- and the
boundary-merging sweep that produced the 15.73 was exhaustive over adjacent
operator pairs -- or a drafter that is right far more often than a suffix
automaton on a tree, which is a modelling question rather than an engineering
one.
