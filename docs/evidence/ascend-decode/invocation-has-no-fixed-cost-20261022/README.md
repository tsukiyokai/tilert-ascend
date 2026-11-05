# An extra matmul invocation has no fixed cost, so the layer's floor is its reads

This overturns a model built earlier today and, with it, the conclusion that 2
milliseconds is out of reach for kernel work.

## The measurement

Prolog's 2624 columns against a shrinking reduction, twelve rows, one die, five
repeats. The fused arm makes one invocation; the split arm makes two.

| reduction | fused us | split us | the extra invocation |
| ---: | ---: | ---: | ---: |
| 256 | 16.04 | 17.33 | 1.29 |
| 512 | 18.16 | 16.38 | −1.78 |
| 1024 | 15.89 | 18.73 | 2.84 |
| 2048 | 16.91 | 25.69 | 8.78 |
| 4096 | 23.07 | 38.89 | 15.82 |
| 6144 | 27.82 | 53.50 | 25.68 |

Least squares over the six depths:

    fused                 = 15.04 + 1.96 us per 1000 of K
    the extra invocation  = -1.58 + 4.41 us per 1000 of K

The extra invocation's intercept is zero. Its whole cost is proportional to the
reduction: 1.13 microseconds predicted at K=256 against 1.29 measured, 27.1 at
K=6144 against 25.68.

## What was wrong

An earlier fit of invocation cost against K gave `11.6 + 0.000985 K` and was
read as an 11.6 microsecond fixed cost an invocation. It is not. The 15.04 is
the **kernel's** fixed cost -- its launch, the `Matmul::Init`, the tiling load
and the closing barriers -- and it is paid once however many invocations the
kernel makes.

So the cost is `15.04 + (reduction walks) * about 2.1 * K / 1000`, and a walk is
nothing other than streaming that invocation's slice of the weight. The walk is
the read.

That reconciles two readings of the FFN that looked incompatible. Its cube half
spends 71.25 microseconds, which is both 95.6 MB at 1341 GB/s and three or four
experts a core over two phases at roughly 13 microseconds a K=6144 walk and 4.3
a K=2048 walk, for 60.6. They are the same quantity counted twice.

## What it means for the target

The layer reads 116.9 MB a rank -- 11.9 of attention weights, 9.4 of KV cache,
95.6 of touched experts -- which is 73 microseconds at the 1607 GB/s measured on
this die. The budget for 2.9 milliseconds is 235 microseconds a layer. So the
floor leaves a factor of 3.2, and today's 650 microseconds is 8.9 times the
floor.

The earlier conclusion -- that a layer's twelve to fifteen invocations cost
211 to 264 microseconds of fixed overhead and therefore exhaust the budget
before any mechanism applies -- does not hold. There is no such overhead. What
fills the 577 microseconds between the reads and today's layer is waiting, and
the two halves idling against each other is a measured part of it: 67.52
microseconds of cube barrier and 111.74 of vector barrier in the FFN body alone.

[`run.sh`](run.sh) is the sweep; the JSON files are its output, one a depth an
arm.
