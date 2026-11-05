# Acceptance saturates, so the budget optimum was never the binding constraint

Eight real streams, 512 tokens each, the row cost set to zero and the budget
taken far past anything the round can afford. TPOT computed at the one-die
compute floor of 24.76 ms a round.

| budget | tree | ms | chain | ms |
| --- | --- | --- | --- | --- |
| 16 | 3.692 | 6.71 | 3.177 | 7.79 |
| 32 | 3.978 | 6.22 | 3.336 | 7.42 |
| 64 | 4.143 | 5.98 | 3.394 | 7.29 |
| 128 | 4.235 | 5.85 | 3.400 | 7.28 |
| 256 | 4.289 | 5.77 | 3.400 | 7.28 |

The tree converges to 4.289 and the chain to 3.400. Doubling the budget from
128 to 256 buys 1.3 percent and 0.0 percent.

## What this overturns

Every budget optimum measured in this project -- 16, 24, "16 to 24" -- was read
as a trade-off against the row cost: 0.66 ms a proposed row makes a wide tree
pay for rows that never hold. That reading implied acceptance would keep rising
if the rows were free.

It does not. With rows free and the budget at 256 the tree gains 16 percent
over budget 16 and then stops. The constraint is the drafter's own reach, not
the row price.

## Which closes the drafting side arithmetically

At the one-die compute floor, the best drafter at any budget is **5.77 ms**.
The target is 2.9. So acceptance cannot supply it: 2.9 ms at that floor needs
8.54 accepted, twice what the tree converges to, and the copy oracle's 13.474
comes from comparing against the answer rather than from a larger budget.

## The one axis left

    2.9 ms = 78 layers x per-layer / accepted

With accepted at its measured ceiling of 4.289, per-layer must be 159 us. The
layer's parts sum to 317 warm on one die, the same layer replays in a graph at
214, and its profiled device-busy time is 203. So the target needs per-layer
compute at half the measured sum and below the profiled busy time -- which is
not a boundary to merge but a different amount of arithmetic.

That is the whole of what remains, stated as one number: **159 us a layer**.
Boundary merging has been spent (15.73 ms), the collectives are mostly absorbed
already (0.26 standalone calls a layer), and acceptance saturates.
