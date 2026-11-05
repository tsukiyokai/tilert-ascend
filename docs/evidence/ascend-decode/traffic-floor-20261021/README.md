# The traffic floor, and the target is inside it once the bank is narrowed

At batch one every operator in a layer is bandwidth-bound -- the grouped matmul
measured 1170 to 1255 GB/s, the projections 11 us each for about 12 MB, the
attention 48 GB/s at 196 us of which 170 was host. So a round's cost has a
physical floor: the bytes it must read, at the rate the die reads them.

A layer a rank, from the checkpoint's own shapes, at the 1200 GB/s the grouped
matmul achieves:

| item | MB | us |
| --- | --- | --- |
| attention weights | 190.2 | 158.5 |
| KV cache, 8K context | 9.4 | 7.8 |
| expert bank, all sixteen | 302.0 | 251.7 |
| expert bank, the two touched | 37.8 | 31.5 |

    full bank    501.6 MB a layer   39.12 GB a round   32.60 ms   TPOT 5.99
    narrowed     237.3 MB a layer   18.51 GB a round   15.43 ms   TPOT 2.84

## Which is the first statement that the target is reachable

**2.84 ms.** Not a projection from a lever's ratio but the time to move the
bytes, at a rate this session measured on this die. The condition is the bank
narrowed to what a round touches and everything else running at bandwidth.

The measured round is 50.73 ms against a narrowed floor of 15.43, so the engine
pays 3.3 times its traffic. That ratio is what the levers and the queued arms
are about.

## And it names the next target

With the bank narrowed, the attention weights are 190.2 MB of the 237.3 a layer
-- 80 percent. Twelve times the KV cache and five times the narrowed bank.

Within them, `kv_b_proj` is `[28672, 512]` in **BF16**, which is 29.4 MB where
every other attention weight is int8. Carrying it at int8 would take the layer
from 237.3 to 222.6 MB, a floor of 14.47 ms and a TPOT of 2.66 -- and whether
that is numerically acceptable is a question for the checkpoint's own
quantisation, not for this page.

The larger question the floor poses: 190 MB of attention weights a layer a rank
at TP16 means 3.0 GB a layer across the node, which is the model's own size
divided by nothing. Reading all of it once a round is what a dense forward
costs, and no amount of speculation changes it -- only accepting more tokens a
round does, which is why 5.441 accepted is the other half of every figure here.

## Correction: the attention weights are a rank's share, 11.9 MB not 190.2

190.2 MB a layer is the whole layer across sixteen ranks -- this session had
already divided it once, recording 0.93 GB a rank over 78 layers. A rank's share
is 11.9 MB a layer.

| item | MB a rank | us |
| --- | --- | --- |
| attention weights | 11.9 | 9.9 |
| KV cache, 8K, replicated | 9.4 | 7.8 |
| expert bank, all sixteen | 302.0 | 251.7 |
| expert bank, the two touched | 37.8 | 31.5 |

    full bank    323.3 MB a layer   21.01 ms a round   TPOT 3.86
    narrowed      59.0 MB a layer    3.84 ms a round   TPOT 0.71

## Which makes the bank the whole story

It is 93 percent of a layer's traffic at full and 64 percent narrowed. The
attention weights are 4 percent and the KV cache 3 percent, so the previous
section's claim that attention dominates and `kv_b_proj` is the next target is
withdrawn entirely -- it rested on the undivided figure.

## And it moves the floor far below the target

A narrowed round's traffic is **3.84 ms**, a TPOT of 0.71. The engine's 50.73 ms
is thirteen times that. So the target is not near the floor, it is well above
it, and what stands between is implementation rather than physics: per-call
costs, collectives, and whatever the arms attribute.

That is a better position than the previous section's 2.84 ms, which made the
target look like a near-miss against physics. It is not. The bytes are cheap;
the round is expensive for other reasons.

## The engine against its own floor, which is the comparison that means something

Comparing a narrowed floor against an unnarrowed engine gave thirteen times,
which is apples to oranges. Against its own floor:

| state | round | floor | ratio | TPOT |
| --- | --- | --- | --- | --- |
| today, full bank | 50.73 | 21.01 | **2.41** | 9.32 |
| narrowed | 33.33 | 3.84 | 8.68 | 6.13 |

So the engine runs at 2.4 times its traffic today. Narrowing removes 17.4 ms of
traffic and leaves the fixed overhead, which is why the ratio worsens to 8.7
while the TPOT improves to 6.13 -- both true at once because the denominator
moved.

## And the fixed overhead alone exceeds the target's whole budget

    fixed overhead      29.72 ms a round, 381 us a layer
    2.9 ms needs        15.78 ms a round
    the overhead is     13.94 ms over that budget by itself

That is the day's sharpest statement. The bytes are 3.84 ms once the bank is
narrowed; the target allows 15.78; and the per-call costs that are neither
bytes nor arithmetic are 29.72. No amount of traffic reduction reaches 2.9 --
the overhead has to go.

Of that 29.72, the collectives are the largest identified part: 156 calls a
round at the 68.5 to 190 us this session measured, which is 10.7 to 29.6 ms. The
rest is per-operator floors -- the grouped matmul's 17 us at two experts, the
FIA's 49.52, the projections' 11 each.

So the levers divide cleanly: the bank narrowing takes traffic, AIV and graph
amortisation take collective overhead, and nothing yet takes the per-operator
floors, which are what a fused kernel a layer would address.

## Narrowing does not make fewer ranks affordable

The thought was that once the bank is narrowed to what a round touches, the
rank count could fall -- buying better FIA utilisation (four heads a rank at
TP16 reads 9.4 MB in 49.52 us, 190 GB/s, where the grouped matmul reaches 1250)
and cheaper collectives.

It does not, because narrowing does not make the bank rank-independent. A rank
owns `256 / ranks` experts and a round's ninety-six picks touch proportionally
more of them:

| ranks | heads | FIA us | touched | bank us | attn MB | layer us | round ms | TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 32 | 135.9 | 40.1 | 631.4 | 95.1 | 897.9 | 103.44 | 19.01 |
| 4 | 16 | 115.5 | 20.0 | 315.7 | 47.5 | 522.1 | 70.70 | 12.99 |
| 8 | 8 | 91.5 | 10.0 | 157.9 | 23.8 | 320.4 | 60.50 | 11.12 |
| 16 | 4 | 87.2 | 5.0 | 78.9 | 11.9 | 227.2 | 63.80 | 11.73 |

TP8 edges TP16 by five percent, 60.50 ms against 63.80, which is inside the
error of pricing collectives at their eager per-call cost when a graph
amortises them. So the trade is flat and TP16 stays right -- the same
conclusion the earlier pricing reached, and it survives the narrowing.

## What the table does show

FIA's utilisation is the worst in the layer: 9.4 MB in 49.52 us is 190 GB/s
against the grouped matmul's 1250. More heads a rank improves it only slowly --
87.2 us at four heads, 135.9 at thirty-two -- so the inefficiency is not
parallelism starvation but something inside the operator.

That is the per-operator floor nothing yet addresses, and it is 50 us of a
narrowed layer's 136 -- 37 percent, 3.9 ms a round. A fused kernel would not
fix it; whatever makes FIA read at 190 GB/s would.
