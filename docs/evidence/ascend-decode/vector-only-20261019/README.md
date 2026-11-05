# Vector-only expert GEMM: closed at these shapes, by 1.32× not 3.90×

Two independent investigations reached this question and **agree to the digit**
on every shared figure — 8.13 element-ops per W4 weight, 31.398 µs, 25.05
GB/s, 1.63% of the die, 453.05 µs a layer, 3.90× the vendor, 16.3% of nominal
per core, a 71 GB/s DMA floor at 256-byte segments on stride 6144 — and the
3.69 budget reproduces exactly as 1.137e13 ÷ (1540e9 × 2).

## No configuration beats the vendor

| measurement | vector-only | vendor | ratio |
| --- | --- | --- | --- |
| best per call, `(256, 6144)` at `block_k=1024` | 19.529 µs | — | 38% better than the shipped `block_k=256` |
| best anywhere per 786,432 packed bytes, 48 workers | 16.487–17.643 µs | — | 44.6–47.7 GB/s |
| eight `gate_up` experts, best-ever per-expert × 8 | 131.9 µs | 60–63 µs | 2.1× short |
| layer total, best `gate_up` and `down` mixed | 356.9 µs | 116.2 µs | 3.07× |

The closure does not rest on the shipped tiling being poor: even the best
point found anywhere in a 27-point sweep is short.

## The gap, apportioned rather than named

The 61.5× between 31.398 µs and the 0.511 µs stream bound factors as
**2.2 × 3.0 × 6.2 ≈ 41** plus an additive floor:

- **2.2** — arithmetic intensity, 8.13 element-ops per weight against a 3.69
  budget, because `dav-c220-vec` has no int4 and no int8 multiply and every
  quantised weight must be widened and sign-restored first.
- **3.0** — core occupancy, 16 of 48 vector cores, forced by
  `OUTPUT_PAIRS = 8` with N = 256 (`_w4a8_tilelang.py:10,31,42`).
- **6.2** — per-instruction overhead, 18–26 ns per vector instruction from a
  repeats sweep that holds DMA segments constant, against about 4.3 ns of ALU
  work.
- **11.02 µs** — a granularity floor, recovered independently as the repeats
  sweep's R=0 intercept at 11.15.

**Per-instruction overhead is responsible first and arithmetic intensity
last**: fixing the intensity alone buys 2×, not 60×. And `down` is not
instruction-bound at all — 384 workers is eight waves at about 2.5 µs each,
its DMA-only variant is *slower* than the full kernel (27.872 against
25.089 µs), and doubling its arithmetic adds only 6.75 µs.

A correction to an earlier relay of these figures: "about 23 cycles against
4.3 ns of ALU work" mixes units — 23 cycles is 12.4 ns, and that 12.4 came
from a `block_k` sweep which confounds instruction count with segment count.
The clean figure is 18–26 ns.

## At full occupancy the marginal byte cost is the DMA floor

The measurement the first line did not make. Holding occupancy at all 48
workers and sweeping the call size:

| packed bytes per call | µs |
| --- | --- |
| 98,304 | 7.360 |
| 196,608 | 9.251 |
| 393,216 | 12.220 |
| 786,432 | 17.643 |

Linear over 8×, fitting **5.891 µs fixed plus 66.9 GB/s marginal** — within 6%
of the 71.36 GB/s DMA-granularity floor. So at full occupancy the arithmetic
is very nearly hidden under the transfer, and the per-instruction overhead
that dominates the shipped configuration is an occupancy artefact rather than
a property of the unit.

Projected to one fused launch per projection, which is a projection from that
fit and not a measured fused kernel:

| work | vector-only | vendor | ratio |
| --- | --- | --- | --- |
| eight `gate_up` experts, 6,291,456 B | 99.9 µs | 60–63 | 1.6× loss |
| eight `down` experts, 3,145,728 B | 52.9 µs | 53 | **parity** |
| layer total | 152.8 µs | 116.2 | **1.32× loss** |

The crossover is 3.82 MB of packed W4 per fused call at the `gate_up` budget
and 3.15 MB at the `down` budget — and eight `down` experts are 3.146 MB,
sitting precisely on it. Vector-only therefore wins below roughly 3.2–3.8 MB
per fused launch, which concretely means top-4 routing instead of top-8, TP32
instead of TP16, or an expert intermediate half the current size. At
GLM-5.2's actual TP16 top-8 shapes it does not win. No other quantisation was
tested and there is no evidence either way there.

## Why 1.32× rather than 3.90× changes the skeleton

A vector-only *operator* replacing one vendor call loses badly. A vector-only
*resident kernel* replacing the whole family loses only 32% on the arithmetic
— and that is the figure to weigh against what a resident kernel stops paying.

The expert GMM families cost about 11 ms a round, so a 1.32× arithmetic loss
is roughly +3.5 ms. Against it stand the per-call intercepts the census
located: 15.4 ms a round, of which these two families alone contribute 5.9.
[census](../prologue-census-20261019/README.md)

So the trade is losing about 3.5 ms of arithmetic efficiency to recover up to
15.4 ms of intercept, and it is strongly positive on those numbers. **The
route the first refutation closed was replacing an operator; the route this
reopens is replacing all of them at once**, where the intercepts vanish and
the cube sits idle — idle being a waste rather than a cost at batch one.

Two things this is not. The 1.32× rests on an 8× linear extrapolation to a
fused call that nobody has written, and the intercepts are an upper bound on
what fusion recovers rather than a measured saving. Both are now the next
experiments rather than conclusions.
