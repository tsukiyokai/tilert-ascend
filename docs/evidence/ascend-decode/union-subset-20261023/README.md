# Choosing rows for their routes saves 19 percent of the union, and closes the last route

2026-10-23, CPU, the engine's own router output -- `teacher-routes.pt`,
[75 windows, 32 rows, top-8] expert IDs.

The MoE floor falls only if a round touches fewer experts, and after residency
was refuted and a narrower dtype found to have no A3 checkpoint, the one route
left was a smaller union a row: draft rows whose routes overlap. This measures
the ceiling of that idea by choosing, per window, the subset of rows with the
smallest union -- greedy on a submodular objective, which is the right
algorithm for it.

| rows kept | union mean | median | min | max |
| --- | --- | --- | --- | --- |
| 32, all of them | **99.36** | 96.0 | 74 | 135 |
| 16, chosen greedily | 55.08 | 53.0 | 42 | 78 |
| 8, chosen greedily | 33.67 | 33.0 | 25 | 43 |

The 99.36 reproduces the recorded figure exactly, so the instrument checks
itself.

## Why that is not enough

The natural union at a given width is already 68.07 at width 16 and 44.35 at
width 8. So choosing rows buys 68.07 to 55.08 -- **19 percent** -- and 44.35 to
33.67 at width 8, 24 percent. Not the fourfold the floor needed.

Priced on today's measured MoE curve, a union of 55 against 68 saves about 1.9
ms a round, and it costs half the rows. Acceptance at 16 rows is 5.989 against
6.966 at 32, and in fact worse than that, because the rows discarded are not
the weakest rows but the ones whose experts do not fit the crowd. The floor
becomes about (17.33 - 1.9 + 5.125) / 5.9, which is **3.5 ms** against 3.75
without choosing at all.

## So this configuration's floor is 3.5 to 3.75 ms

All three routes to the MoE bytes are now closed by measurement: a narrower
expert dtype has no A3 checkpoint, residency is refuted because a grouped
matmul never reads a shard twice, and choosing rows for their routes is worth
19 to 24 percent and is paid for in acceptance.

MoE plus the collective is 22.45 ms a round at width 16 and 25.66 at 32, the
best drafter this project has measured on its own streams accepts 6.966, and
2.9 ms at that acceptance wants a 20.2 ms round. **2.x is not reachable on
W4A8C8 with TP16 on one host** -- not for want of engineering on attention,
routing or kernel count, all of which could go to zero and leave 3.68.

What would move it, and is outside what was measured here: fewer ranks holding
more experts each so a round's union is a smaller share of the bytes read, an
expert dtype narrower than int4, or a checkpoint whose routing is sparser a
token than top-8 of 256.
