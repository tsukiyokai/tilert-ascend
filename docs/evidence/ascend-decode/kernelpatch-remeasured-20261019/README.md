# The three gmm1 kernel patches are worth 0.33 ms a round, once the cache is out of the way

The earlier reading of these patches was +3.15 ms a token and could not
attribute anything: both arms shared the static-kernel cache index, whose key
omits the vendor path, so at the captured width both ran whichever arm compiled
first.
[why that comparison was void](../patch-regression-20261019/README.md)

Re-measured with `FIX_ARM=cache-off`, width 8, one tree, each arm's gmm1
object read back from its own recorded search path, 1024 output tokens, three
warmups and three measured requests.

| arm | gmm1 object | round p10 | TPOT |
| --- | --- | --- | --- |
| barrier only | `a9863296` at index 3 | 57.110, 56.875, 56.839 | 13.1049, 12.9081, 12.8858 |
| + three kernel patches | `0a20fe78` at index 3 | 55.749, 55.343, 55.327 | 12.6554, 12.6037, 12.6427 |

**−1.468 ms a round, −0.326 a token.** On TPOT medians, 12.8858 to 12.6427,
−0.243.

## Why this one is believable where today's others were not

The between-run floor on a three-request arm's p10 mean is 0.201 ms, measured
by running the same configuration twice from different directories. This
effect is 7.3× that floor, and the two arms' nearest values are still 1.090 ms
apart — 5.4× the floor. The metadata A/B that had to be withdrawn was 1.39×.
[the floor](../positions-refuted-20261019/README.md)

The arms also run in the direction opposite to the only order effect observed
today: here the second arm is faster, where in the earlier chain the second was
slower. So this is not the order.

## Correctness

Both arms emit the same 1024 tokens, digest `68f2bc184553`, which is width 4's
reference stream. The gain costs nothing in trajectory.

## What it does to the board

New best deterministic TPOT **12.6037 ms**, median 12.6427. A round's p10 is
55.33 ms against a 9.013 ms budget at width 8's acceptance, so 6.1× rather
than 6.3×.

The patches were measured per call at 8.5 µs over 78 calls, about 0.66 ms a
round. The round moved 1.468 — more than twice the per-call prediction, in the
right direction this time. Per-call figures still do not predict rounds; that
has now failed in both directions.
