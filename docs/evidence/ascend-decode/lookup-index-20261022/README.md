# The lookup index keyed every tail by a tuple, and it should not exist at all

2026-10-22, CPU. `LookupDrafter.commit` built one tuple for every (position,
tail length) pair and used it as a dict key, so indexing an 8192-token prompt
over lengths 2 to 32 constructed 254 thousand tuples whose hashes cost one
operation a token.

## The patch, and exactly what it is worth

A tail is now keyed by a polynomial hash, so extending it by a token is a
multiply and an add rather than a slice and an O(size) hash. A query verifies
its one candidate against the context, so a collision cannot propose anything
wrong.

Paired, median of three, `tracemalloc` on for the memory column, replaying the
engine's `tail32` stream at width 32:

| variant | build | peak | entries | propose p50 | commit p50 |
| --- | --- | --- | --- | --- | --- |
| tuple keys | 65.7 ms | 51.3 MB | 256744 | 8.93 us | 7.42 us |
| rolling hash | 44.8 ms | 34.9 MB | 256744 | 8.82 us | 4.44 us |

1.47x the build, 1.47x the memory, 1.67x the commit, and `propose` unchanged.
Smaller than it looks like it should be: CPython's slice and tuple hash are
both C, so the naive constant was never large, and the insert count is O(n x L)
either way -- what the hash removes is the O(size) copy inside each insert.

**It is not on the TPOT path.** A drafter costs 0.1 to 0.3 microseconds a round
on the host and TPOT spans callbacks, so this is request setup: 21 ms and 16 MB
a rank, 16 ranks a host.

## Equality, round by round

`prove_lookup.py` drives both implementations from the same committed prefix
and compares every proposal: the engine's own two 78-layer streams (three
measured requests each) and a source file at edit rates 0, 0.02 and 0.05, at
all eight widths the engine can capture. **9 streams x 8 widths, 67071 rounds,
0 mismatches.**

## The structure is dominated on every axis measured

| | lookup | automaton |
| --- | --- | --- |
| accepted a round, engine stream, w12 | 4.571 | 4.995 |
| accepted a round, engine stream, w32 | 5.224 | 5.953 |
| accepted a round, engine stream, w64 | 5.306 | 6.206 |
| build, 8192-token prompt | 65.7 ms | 30.0 ms |
| peak memory | 51.3 MB | 3.7 MB |

A suffix automaton answers "the longest tail that occurred before, and its most
recent end" in O(1) amortised after an O(n) build, which is the asymptotic fix
rather than a constant one -- and it accepts 14 percent more at the engine's own
width. So the honest reading of this file is that the optimised structure is a
1.5x patch on one that should be retired; `lookup` stays only because the
recorded 9.3794 ms used it and a control arm has to reproduce it.
