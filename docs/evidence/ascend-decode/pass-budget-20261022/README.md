# The pass budget: 48 today, 11 for 2.9 milliseconds

Everything measured today reduces to one model, and the model turns the target
into a count.

## The model

A pass over data costs `max(12 us, bytes / 1607 GB/s)`. The floor is measured
both ways: a Torch reduction over 393 KB takes 11.82 microseconds and an
in-kernel AscendC vector pass over 288 KB takes 12.46, and the increment
survives graph capture. The bandwidth term is measured too: the expert bank's
95.6 MB takes 71.25 microseconds of cube work, which is 243 times the data for 6
times the time, so large passes leave the floor and ride the bus.

A layer reads 116.9 MB a rank -- 11.9 of attention weights, 9.4 of KV cache,
95.6 of the 81 touched experts -- or 72.7 microseconds at 1607 GB/s. So

    layer = 12 us * (small passes) + 72.7 us

## The count

| TPOT ms | round ms | layer us | small passes |
| ---: | ---: | ---: | ---: |
| 9.324 today | 50.73 | 650.4 | 48.1 |
| 8.570 built | 46.63 | 597.8 | 43.8 |
| 6.000 | 32.65 | 418.5 | 28.8 |
| 4.000 | 21.76 | 279.0 | 17.2 |
| **2.900** | 15.78 | 202.3 | **10.8** |
| 2.000 | 10.88 | 139.5 | 5.6 |

So the layer makes about 48 passes over its data and the target allows about
eleven. That is the whole problem, stated in the one unit that today's
measurements agree on.

## Which passes are known

Removing a pass means two operations sharing one load. The hardware constrains
where that is possible: the cube reads from global memory or L1, never from
unified buffer, so a vector-produced intermediate feeding a matmul always costs
a round trip. Vector-to-vector pairs can share a pass; vector-to-cube cannot.

Accounted for, against the 48:

    one reduction walk instead of four, the front stage          3 passes
    both tensor-parallel reductions inside the layer launch      ~1
    the expert skew's rescheduling                               0, it moved passes
    the selector's three passes folded to one                    2, on 21 of 78 layers
    the FFN's remaining vector work hidden under its cube        ~2 equivalent

That is about six of the thirty-seven that have to go. The rest are inside
operators this project cannot restructure -- the vendor's MLA Prolog and sparse
flash attention -- or inside stages whose per-core composition cannot be
measured on a3-194, because the instrumented attention build needs a recorded
ops-transformer snapshot whose `MlaPrologBaseParams` members are absent from
both CANN trees on the host.

## What this rules out and what it leaves

It rules out the levers that do not change the pass count. Fewer kernel launches
is 1.064 microseconds each in a graph. Fewer operator calls is nothing by
itself, since an in-kernel pass costs what a call costs. Rescheduling passes to
overlap halves is bounded by the vector work that exists to hide -- 23.40
microseconds in the FFN -- and the expert loop's imbalance is locked because
dividing it more finely buys another reduction walk.

It leaves one thing: rewriting the vector stages so that consecutive operations
on the same data share a load. That is what "a perfect megakernel" has to mean
here, and the budget says it has to get from 48 passes to 11.

[`budget.txt`](budget.txt) is the arithmetic.
