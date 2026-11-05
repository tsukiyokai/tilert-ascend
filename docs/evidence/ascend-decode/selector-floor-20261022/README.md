# Any per-row reduction over twelve by 8192 floats costs about 18 microseconds

The indexed attention's selector stage measures 54.84 microseconds and is pure
vector work, which cannot hide under the phase's cube. So its floor decides
whether it is worth replacing, and the floor turns out not to depend on the
algorithm at all.

## The measurement

Twelve rows of 8192 float scores, 393 KB, one die, five repeats of 50
iterations:

| operation | median us |
| --- | ---: |
| `topk(2048, sorted=True)` | 17.93 |
| `topk(2048, sorted=False)` | 17.91 |
| `sort` of all 8192 | 17.86 |
| `(scores > 0).sum(dim=1)` | 21.09 |

A full sort costs what selecting a quarter costs, and the trivial threshold
count costs more than either. So the figure is not the work: 393 KB in 18
microseconds is 22 GB/s, far under what the vector cores can move, and the cost
belongs to the operator call.

## What it gives and what it does not

The selector's 54.84 microseconds is about three such operations, and one would
be 18. That puts 37 microseconds a layer within reach on the 21 layers that
compute an indexer -- layers 0, 1, 2, 70 and 74 with BF16 indexers and 6 through
66 at stride four with C8 -- which is 0.78 milliseconds a round.

The limit on reading further: 18 microseconds is the fixed cost of a **Torch
operator call**, and the native attention is one kernel whose vector stages are
not operator calls. So this prices the selector as a standalone operation and
does not decompose attention's estimated 199 microseconds of in-kernel vector
work. That needs the per-core trace, which cannot be built on a3-194 for want of
the recorded ops-transformer snapshot.

[`probe.py`](probe.py) is the measurement.
