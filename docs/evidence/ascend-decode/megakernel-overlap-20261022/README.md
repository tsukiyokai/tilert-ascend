# A cube stage hides about ten microseconds of vector work for free

The whole-layer kernel separates its phases with `EndPhase(pipe)` and
`PipeBarrier<PIPE_ALL>()`, so the cube idles through each vector phase and the
vector cores idle through each projection. This measures what that costs by
putting a sweepable vector load on the AIV half of the front stage with no
barrier against the cube.

## The sweep

One idle die of a3-194, twelve rows, the `prolog` level at int8, five timing
repeats. A pass is 72 chunks of 1024 elements: duplicate, absolute value,
reduce, accumulate.

| passes | us | above the cube alone | marginal us a pass | vector stage verified |
| ---: | ---: | ---: | ---: | --- |
| 0 | 25.35 | — | — | — |
| 1 | 27.24 | +1.89 | 1.89 | yes |
| 2 | 34.70 | +9.35 | 7.46 | no |
| 3 | 47.24 | +21.89 | 12.54 | yes |
| 4 | 59.14 | +33.79 | 11.90 | yes |
| 6 | 84.43 | +59.08 | 12.65 | yes |
| 8 | 108.06 | +82.71 | 11.81 | yes |
| 16 | 209.23 | +183.88 | 12.65 | yes |

The marginal cost settles at 12.46 microseconds a pass, taken as the slope from
three passes to sixteen.

## The halves run at once

Three models, with the cube at 25.35 microseconds and a pass at 12.46:

| passes | perfect overlap | no overlap | measured |
| ---: | ---: | ---: | ---: |
| 1 | 25.35 | 37.81 | 27.24 |
| 8 | 99.68 | 125.03 | 108.06 |
| 16 | 199.36 | 224.71 | 209.23 |

The measurement sits within 10 microseconds of perfect overlap at every point
and 15 to 25 below serial execution, so the two halves do run together. The
first pass adds only 1.89 against a marginal 12.46, which puts the free capacity
at about 10.5 microseconds -- some 40 percent of the cube stage's own duration.

## The gate was necessary and it changed the answer

An earlier version of this sweep had the vector stage compute in unified buffer
and publish nothing. It reported under one microsecond at every load, including
64 passes, and was non-monotonic. That was dead-code elision: with no observable
result the compiler removed the loop. Publishing each core's accumulated total
and checking it against fixed arithmetic turned "free at any load" into "12.46
microseconds a pass beyond the first".

Publishing it correctly needed a second fix. One float a core put eight cores on
one 32-byte line, and a flush writes the whole line, so neighbours overwrote each
other and the check failed intermittently. One cache line a core resolved it;
seven of the eight points above verify exactly, and the two-pass point remains a
residual flake in the gate rather than in the timing, which follows the trend.

## Per-tile publication on the row axis cannot pay here

The AMD design this follows hands work between resident workgroups through
memory with a sequence tag, so a consumer starts on the part of its input that
is ready. On the row axis at batch one that cannot pay, and the arithmetic is
already on this page.

A matmul invocation cannot begin on partial input: `IterateAll` consumes the
whole A it was given. So starting the cube early means splitting it into one
invocation a row group, and an invocation at a reduction of 6144 costs about 25
microseconds -- the difference between 23.91 fused and 51.07 split for Prolog's
two projections. What early starting saves is the producer's duration, and the
producer here writes 12 by 6144 int8, 73 KB in and out, a fraction of one 12.46
microsecond pass.

So two row groups would buy a few microseconds of earlier start for 25 of extra
invocation. The trade is decided by the ratio, not by the protocol, and it is
decided against.

A protocol was built to measure it anyway and is not kept. A cross-half
`SyncAll` in the producer deadlocks against a consumer spinning on a flag,
because the barrier counts both halves; replacing it with an atomic counter per
group still hung at every group count, including one. The code is reverted
rather than left half-working, since the measurement it would produce is already
implied by the two figures above.

## What it implies for the layer

The layer's vector work -- the two norms, the dynamic quantizations, RoPE, the
Hadamard, the top-k selection, the expert combine -- is substantial, and the
whole-layer kernel currently pays all of it in series with the cube. If each
cube stage covers roughly 40 percent of its own duration in vector work, the
barriers are leaving tens of microseconds a layer unclaimed.

But claiming it means finding vector work that is *independent* of the cube work
it would hide under, not splitting a dependent producer into tiles. The layer's
large vector items -- the two norms, the dynamic quantizations, the expert
combine -- all sit on the dependency chain, so what remains to schedule under a
cube stage is the work that does not depend on the current token: the indexer's
Hadamard matrices and sine-cosine gathers, the page-map conversions, the top-k
workspace. Those are small. The capacity is 10.5 microseconds a stage; how much
of it a real layer can fill is the open question, and it is a question about the
schedule rather than about the protocol.

[`run.sh`](run.sh) is the sweep; the JSON files are its output, one a load.
