# The frontier: both halves in one table

`TPOT = 78 layers x per-layer / accepted`, rows excluded because the fit's
0.66 ms a row is unattributed. Every figure measured today.

| per layer (us) | lookup@12 5.441 | tree@16 5.394 | automaton@32 6.872 | oracle@16 9.660 | oracle@24 11.011 |
| --- | --- | --- | --- | --- | --- |
| **588** engine now, fitted term / 78 | 8.43 | 8.51 | 6.68 | 4.75 | 4.17 |
| **386** after the measured merges | 5.54 | 5.59 | 4.39 | 3.12 | 2.74 |
| **317** the layer's parts, warm | 4.55 | 4.59 | 3.60 | 2.56 | 2.25 |
| **214** the same layer in a graph | 3.07 | 3.09 | **2.43** | 1.73 | 1.52 |
| **203** its profiled device-busy | 2.90 | 2.93 | 2.30 | 1.64 | 1.43 |

The engine's own cell is 8.43 against 9.3794 measured; the difference is the
row term the fit charges and nothing explains.

## Nine cells reach 2.9 ms, and one of them is made of two measurements

The honest reading is the cell at 214 us a layer and 6.872 accepted: **2.43
ms**. Both numbers are measured -- 214 is the synthetic layer replayed in a
graph on one die, 6.872 is the automaton at width 32 on a real stream -- and
neither needs the copy oracle, which is a bound rather than a drafter.

So 2.x is inside two things already measured, and what stands between is a
single quantity: the engine pays 588 us a layer where the same operators
replay in 214.

    588 - 214 = 374 us a layer = 29.2 ms a round

That is the same unattributed four fifths of the row cost, seen per layer
instead of per row, and it is now the only thing between the measurements and
the target.

## What it is not

Not the operators: every one is flat in rows, warm and cold, and their warm sum
is 317 against the engine's 588. Not capture width: 0.05 ms a row. Not the TP
reduction at two ranks, the Gloo command, the trajectory, the drafter, or graph
buckets -- all excluded. What remains unexamined is sixteen-rank behaviour,
which needs the machine the arms are queued on.

## Why this table is worth more than its cells

Every earlier attempt priced one half and assumed the other. The merges were
costed against a fixed acceptance; the drafters against a fixed round. Neither
reaches 2.9 alone and the product of their measured bests does, which is why
the single unattributed quantity -- 374 us a layer -- is the whole project
rather than one lever among several.

## Correction: the lower three rows exclude the sixteen-rank collectives

Rows 317, 214 and 203 are all one die with TP16 shards and a dense attention
stand-in. None of them contains a collective. So the cell at 2.43 ms is not the
product of two comparable measurements -- it is a collectives-free compute
figure times a measured acceptance, and the engine's 588 us a layer includes
collectives that the 214 does not.

The size of what is missing, from `profile-rope-20261018` rank 1:

    AivKernel, collective    20.4 calls a round   858.4 us each   17.5 ms

Twenty calls a round over 78 layers is **0.26 a layer**, not two. So the
collectives are already mostly absorbed into the SuperKernel, and the 17.5 ms
standalone remainder is not a per-layer quantity at all -- which is why
dividing it by 78 to compare against 214 would be the same error in the other
direction.

And that capture is `max_tokens: 6, mtp_width: 4`, so its per-round figures are
void for a steady-state decode, as recorded. Both halves of the attribution
therefore have a known defect: the one-die account omits collectives, and the
sixteen-rank capture was taken at the wrong width.

## Which is why the gating experiment is the one that was queued

A steady-state decode capture at the engine's own width gives the per-layer
cost with collectives in it, on sixteen ranks, at the right width -- the single
number both halves of this table are missing. Until it exists the table bounds
the problem rather than solving it: the engine is at 9.3794 ms, the drafting
side is bounded at 5.394 to 6.872 accepted, and the compute floor on one die is
24.76 ms a round, which at 6.872 accepted is 3.60 ms before a single collective
is counted.

That last figure is the one worth carrying: **3.60 ms**, compute-only, best
measured drafter, zero row cost, zero collectives. The target is below it, so
2.9 ms needs acceptance above 8.5 at that floor -- which only the copy oracle
reaches, and the oracle is not a drafter.

## Redone with the narrowing, and the conservative cell reaches 2.x

The acceptance ceiling of 4.289 came from the offline corpus, which reads 21
percent below the engine's own measured 5.441 at width 12. So it is a floor on
the ceiling rather than the ceiling, and the right way to use it is as the
conservative column.

| per layer | tree 4.289 | lookup 5.441 | automaton 6.872 |
| --- | --- | --- | --- |
| the layer's parts, warm 317.5 | 5.77 | 4.55 | 3.60 |
| the same layer in a graph 213.9 | 3.89 | 3.07 | 2.43 |
| its profiled device-busy 202.5 | 3.68 | 2.90 | 2.30 |
| **in a graph, after the narrowing 154.5** | **2.81** | 2.22 | 1.75 |

The last row subtracts the 59.4 us a layer the bank narrowing measured in a
graph. At the most conservative acceptance this project has -- the offline
tree's saturated 4.289, below the engine's own figure -- it gives **2.81 ms**.

So the day's earlier conclusion inverts. The floor of 3.69 ms was computed
before the narrowing existed; with it, the conservative cell is inside the
target.

## What this rests on, named

Measured: 213.913 us a layer replayed in a graph on one die; 59.4 us a layer
saved by narrowing, also in a graph; 4.289 accepted offline and 5.441 measured
end to end by the engine.

Assumed: that the engine's own layer replays at the synthetic layer's cost --
which is the open question the queued steady-state capture answers, and the
same assumption the earlier `graph-closes-gap` page turned on. The engine
currently fits 588 us a layer, 2.7 times the graph figure, and nothing has yet
explained the difference.

So 2.81 ms is what the measured levers compose to, and whether the engine can
be brought to the graph figure is the one thing still untested.
