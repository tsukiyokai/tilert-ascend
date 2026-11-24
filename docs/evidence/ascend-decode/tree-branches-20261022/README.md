# The tree's branch cap is not the constraint, and the tree is the only drafter still climbing

2026-10-22, CPU, the engine's own two 78-layer streams -- `lever-20261020`
`tail32` and `tail8`, three measured requests each, 1024 output tokens, against
the same 8192-token capture prompt.

`TreeDrafter.propose` already pops by path probability, which is optimal for
expected accepted prefix under a node budget: the objective is additive over
nodes and a prefix's probability dominates its extension's, so the best B
nodes are prefix-closed by construction. The one thing between that and the
optimum was `_continuations`, which truncates each state to `branches`
likeliest tokens -- a node's fifth child could never enter the frontier however
high its path probability.

Accepted tokens a round, sweeping the cap:

| stream | budget | automaton | b=4 | b=8 | b=16 | b=64 | uncapped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tail32 | 12 | 4.995 | 5.505 | 5.505 | 5.505 | 5.505 | 5.505 |
| tail32 | 32 | 5.953 | 6.919 | 6.872 | 6.872 | 6.827 | 6.827 |
| tail32 | 64 | 6.206 | 7.420 | 7.474 | 7.474 | 7.474 | 7.474 |
| tail8 | 12 | 5.044 | 5.476 | 5.476 | 5.476 | 5.476 | 5.476 |
| tail8 | 32 | 6.024 | 7.014 | 6.966 | 6.966 | 6.919 | 6.919 |
| tail8 | 64 | 6.321 | 7.474 | 7.529 | 7.529 | 7.529 | 7.529 |

Lifting the cap moves acceptance by at most 0.11 and at budget 32 it *loses*
0.09: the automaton's empirical continuation distribution is concentrated
enough that the fifth child never matters. So `branches=4` stays, and the
search is as optimal as its probability model.

## What the same table does settle

**The tree beats the automaton on the engine's own output**, by 16 percent at
budget 32 and 20 at 64 -- not only on the constructed corpus where that was
first claimed.

**Widening is spent for chains and not for trees.** Across budgets 12, 32, 64:
lookup 4.571, 5.224, 5.306; automaton 4.995, 5.953, 6.206; tree 5.505, 6.919,
7.420. The chains flatten, the tree is still climbing at 64.

## And what it costs the target

With the round model's measured terms, the shared read's row cost and the pass
budget's eleven passes:

    w32, tree 6.92   (15.97 + 0.458 x 32) / 6.92 = 4.43 ms
    w64, tree 7.47   (15.97 + 0.458 x 64) / 7.47 = 6.06 ms

Against the 10.56 accepted that 2.9 ms needs at width 32. So tree shape does
not close the acceptance leg on these streams, and the copy oracle's 13.474
says the headroom is real but out of reach of a recall drafter here. Both
fixtures this project has are the wrong distribution: a capture-prefix prompt
whose output repeats itself, and a verbatim file copy. The external evidence
for recall on genuine coding edits is 7.552 to above 10, so the missing
measurement is a real agentic edit trace, not another drafter.
