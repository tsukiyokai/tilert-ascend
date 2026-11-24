# The TP16 reduction is rendezvous, not transfer, at every size

`a3-84` idle, sixteen ranks, eager float32 `all_reduce`, 20 warmups and 50
timed calls on device events.
[baseline](../../../../tests/glm52/bench_reduce_floor.py).

| elements | message | µs | GB/s |
| --- | --- | --- | --- |
| 1 | 4 B | 157.43 | 0.00 |
| 16 | 64 B | 121.63 | 0.00 |
| 256 | 1 KiB | 97.06 | 0.01 |
| 4096 | 16 KiB | 147.45 | 0.11 |
| 24576 | **96 KiB** (our shape) | 129.61 | 0.76 |
| 98304 | 384 KiB | 103.25 | 3.81 |
| 393216 | 1.5 MiB | 106.99 | 14.70 |
| 1572864 | 6 MiB | 109.72 | 57.34 |

Four bytes cost 157 µs and six mebibytes cost 110. Across a 1.5-million-fold
range the cost does not rise — it drifts within noise. **None of this figure
is transfer.** At our shape the apparent 0.76 GB/s is not a bandwidth
measurement at all; it is 96 KiB divided by a rendezvous.

By collective, at the production shape:

| collective | single µs | marginal µs | mode |
| --- | --- | --- | --- |
| `all_reduce` | 188.67 | 126.20 | AIV |
| `reduce_scatter` | 110.52 | 148.16 | AIV |
| `all_gather` | 196.90 | 104.08 | AIV |
| `barrier` | 426.66 | 356.25 | AIV |
| `all_reduce` | 254.35 | 150.10 | AI_CPU |
| `barrier` | 876.23 | 393.27 | AI_CPU |

Spread across ranks is tight, 125.63 to 127.87 µs, so no straggler is hiding
in the mean. A bare barrier costs about three times one `all_reduce`, which
places HCCL's reduction at roughly a barrier's worth of synchronisation plus
its arithmetic.

## Eager against the profile

The production profile reads 63.5 µs per reduction, half of what eager shows
here, which is the familiar gap: the profile is inside an ACL graph while
this is eager. The conclusion is unaffected — whichever number applies, it is
entirely rendezvous, and 63.5 µs for a message whose transfer is free is the
thing to remove.

## What this licenses

The physical content of one reduction, done as a one-shot push: each rank
writes 48 KiB to each of fifteen peers, which is 48 KiB per link if the links
are independent, then reduces 768 KiB locally in a fixed order, which is under
a microsecond of vector work. The irreducible part is one inter-die round
trip for the flags. That is single-digit microseconds against 63.5, so the
headroom is roughly twenty-fold, and unlike the earlier estimate this now
rests on a measurement that the transfer term is zero.

It also closes the argument against cutting the count. 157 reductions is not
the problem; 63.5 µs of rendezvous is. The count-cutting route was tried and
lost ([EP16](../ep-megamoe-20261018/README.md)).

## The next number

A sixteen-rank one-sided write, timed. Everything above says what it should
cost; nothing above measures it, and this project has already hit 507015
attempting exactly that at TP16 with two PEs surviving. Until a push
measures, the twenty-fold is arithmetic.
