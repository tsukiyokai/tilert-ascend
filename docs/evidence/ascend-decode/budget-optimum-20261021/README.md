# The optimum budget, with the width term measured

2026-10-21, CPU, 34 source files, edit rate 0.02, the drafter's own trajectory
per column.

| budget | automaton | tree | adaptive tree | its nodes |
| --- | --- | --- | --- | --- |
| 4 | 4.157 | 4.152 | 4.150 | 3.7 |
| 8 | 6.359 | 6.346 | 6.034 | 6.1 |
| 12 | 7.978 | 7.953 | 7.155 | 7.8 |
| 16 | 9.221 | 9.178 | 7.830 | 8.9 |
| 24 | 10.918 | 10.854 | 8.606 | 10.3 |
| 32 | 12.117 | 12.043 | 8.932 | 11.2 |

TPOT with the fixed term after the merges, 45.88 - 21.04 = 24.84 ms, and the
measured 0.66 ms per unit of captured width:

    budget 16   (24.84 + 10.56) / 9.221  = 3.84 ms
    budget 24   (24.84 + 15.84) / 10.918 = 3.73 ms
    budget 32   (24.84 + 21.12) / 12.117 = 3.79 ms

The optimum is near 24 at about **3.73 ms**, and it is flat between 16 and 32.

## Two corrections

**The tree has no advantage on this corpus.** At equal budget it matches the
automaton to within half a percent -- 9.178 against 9.221 at 16, 12.043 against
12.117 at 32. The earlier figure of 14.510 accepted came from a budget of 64
with a width ceiling of 32, which is two parameters varied at once; at equal
budget the tree's branching buys nothing here, because the automaton's single
path is usually right when a suffix matches at all.

**The adaptive budget is worse, not better.** It saves nodes and loses more
acceptance: 7.830 against 9.221 at budget 16. Its earlier win was measured
against a fixed width of 32 while it averaged 12.6 -- again two parameters at
once. With the width term now measured per unit of *captured* width, and the
capture fixed by the configuration rather than by what the drafter asks for,
shrinking the request saves nothing and costs acceptance.

So the configuration to run is the plain automaton at a captured width near 24,
and the tree machinery's value is the causality mask rather than the branching.
