# The cost is a pass over the data, in a kernel or out of one

This is the session's central technical conclusion, and it says what a
megakernel is for.

## A vector invocation costs 11 to 17 microseconds, and capture does not change it

Twelve rows of 8192 float scores, one die, five repeats of 30 iterations:

| arm | eager us | graph us | us an extra call, eager | graph |
| --- | ---: | ---: | ---: | ---: |
| one `sum(dim=1)` | 11.82 | 12.01 | — | — |
| four sums | 45.22 | 45.38 | 11.13 | 11.12 |
| one `topk(2048)` | 18.59 | 18.07 | — | — |
| four topk | 71.25 | 69.41 | 17.55 | 17.11 |

The increment is identical under capture, so the cost is device time rather than
host dispatch. That control was necessary: this project once attributed 12.62 ms
a round to eager dispatch from a trace that cannot tell dispatch from replay.

## It is per pass, not per call

Three figures have to be read together:

    a Torch sum over 98304 floats                        11.82 us
    a Torch topk over the same                           18.59
    an in-kernel AscendC vector pass over 73728 floats   12.46

The in-kernel pass costs what the operator call costs. So the 11 to 17
microseconds is not a tax on calling an operator; it is what one pass over this
much data with vector instructions costs on this die, wherever the loop lives.

That resolves everything measured today that looked inconsistent:

The front stage's 26.87 microseconds came from one reduction walk instead of
four -- one pass instead of four over the same weights -- not from making one
call instead of four.

The expert skew won only 2.94 because it rescheduled passes without removing
any: the same gate/up and the same activate still each walk their data.

The native path's four kernels a layer tie the vendor's twenty-two at 574
against 575 microseconds a layer, because moving passes inside a kernel does not
remove them.

## So what a megakernel is for

Not fewer launches: in a graph a kernel costs 1.064 microseconds to issue. Not
fewer operator calls as such. A megakernel pays when it does several operations
in **one pass over the data** -- load once, compute several things, store once --
and it pays nothing when it merely puts several passes in one launch.

The cube's side of the same rule is already measured: an extra matmul
invocation's whole cost is one more walk of its reduction, 4.41 microseconds per
thousand of K, and a walk is the weight's read. Concatenating projections along
their output axis wins because it walks once.

[`probe.py`](probe.py) is the measurement with its capture control.
