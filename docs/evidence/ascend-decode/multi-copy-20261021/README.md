# Several candidate copies lose to the tree

The idea: the copy oracle wins because it knows which prior occurrence extends
furthest, and that choice only has to be made if rows are scarce. Rows are flat
from 1 to 256, so propose the k most recent occurrences as k parallel verbatim
runs and let verification pick.

Eight real streams, TPOT at the profiled device-busy floor of 15.83 ms a round
with rows free.

| branches | depth | accepted | rows | ms |
| --- | --- | --- | --- | --- |
| 1 | 16 | 2.807 | 8.6 | 5.64 |
| 2 | 16 | 2.979 | 15.3 | 5.32 |
| 4 | 16 | 3.108 | 24.4 | 5.09 |
| 4 | 32 | 3.225 | 44.6 | 4.91 |
| 8 | 16 | 3.159 | 36.6 | 5.01 |
| 8 | 32 | 3.280 | 69.5 | 4.83 |
| 16 | 16 | 3.216 | 55.3 | 4.92 |

Against the tree's 4.289 on the same kind of stream. Sixteen branches reach
3.216, and the marginal return on branches is nearly nil: one to sixteen moves
acceptance by 15 percent while rows go from 8.6 to 55.3.

## Why it loses, and what that confirms

The tree branches by frequency at **every node**, so it keeps several
candidates alive at every depth. This keeps several candidates alive only at
the root and then commits each branch to one source. The tree's shape is
strictly stronger, which the numbers show rather than assume.

So the oracle's advantage is not the choice of occurrence. Offering sixteen
occurrences at once reaches 3.216 where the oracle reaches 13.474. Its advantage
is knowing the answer, which is the third independent route to the same wall --
after the root fallback (no match was ever missing) and length-ranking (length
looks backward).

## The index bug that made the first run meaningless

The first version indexed only `tail`-length windows and then queried tails
from `tail` down to 2. A shorter key had never been stored, so the descent
found nothing and acceptance read 1.724 to 1.782 -- which would have been
reported as the mechanism failing. Indexing every length from 2 to `tail` moved
it to 2.807 to 3.280. The mechanism still loses, and for a reason the broken
version could not have shown.

## A source pointer loses too, and says why

Part of the oracle's advantage could be continuity rather than foresight: once
a copy run starts, the source position is known and advancing it needs no
decision, where a suffix automaton re-decides every round and can jump to a
different occurrence. Twelve streams, width 16:

    lookup    mean 3.148   min 1.561   max 3.821
    pointer   mean 1.867   min 1.128   max 2.151
    the pointer advanced 118 times and re-matched 273
    the pointer beats lookup on 0 of 12

The counts explain the loss. The pointer breaks more than twice as often as it
survives -- 273 re-matches against 118 advances -- so continuity is not what
this workload offers. What the target accepts diverges from the source
constantly, and each rebuild takes the latest occurrence of a fixed eight-token
tail, which is weaker than the multi-length backoff lookup already does.

So the oracle's advantage is not continuity either. Four independent routes now
end at the same wall -- the root fallback, length-ranking, several candidates at
once, and a source pointer -- and all four say the same thing: it knows the
answer.

## The index bug, again worth recording

The first run read 1.000 accepted with zero pointers ever formed. `_index` kept
one position a key, so the context's own tail was stored there and the query
rejected it as not earlier than itself, with nowhere to fall back to. Indexing
every occurrence moved it to 1.867. Second time today that a broken index read
exactly like a failed mechanism.
