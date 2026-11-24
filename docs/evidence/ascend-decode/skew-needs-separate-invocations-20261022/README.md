# Skewing pays only where the cube is already divided into invocations

The upper bound on skewed co-scheduling assumed the cube's work can be cut into
pipeline steps. It cannot be cut for free: each step is another matmul
invocation and each invocation walks the reduction again.

## The measurement

The front stage at `prolog`, twelve rows, one die. Zero steps runs the halves
one after the other with a barrier between, which is what a phase boundary
gives; more steps interleave the cube's step `s` with the vector's step `s-1`
inside one barrier region a step.

| steps | us at 2 vector passes | against serial | us at 4 passes | against serial |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 56.18 | — | 79.49 | — |
| 1 | 53.79 | −2.39 | 79.77 | +0.28 |
| 2 | 62.63 | +6.45 | 80.23 | +0.74 |
| 4 | 99.68 | +43.50 | 100.40 | +20.91 |
| 8 | 126.37 | +70.19 | 136.38 | +56.89 |

Monotonically worse with depth. At eight steps the extra 70 microseconds is
seven more walks of a 6144 reduction, which the increment fit prices at 4.41
microseconds per thousand of K. The vector work available to hide is 25
microseconds at two passes. The trade is 70 against 25.

## The criterion it gives

Skewing is worth it exactly when the cube work is **already** divided into
independent invocations, so interleaving adds barriers and no walks.

That is the difference between this result and the one that works. The front
stage is one invocation by design -- that is what earns its 26.87 microseconds --
so cutting it for a pipeline gives back more than the pipeline saves. The FFN's
expert loop is already one invocation an expert, three or four to a core, so
interleaving expert `e`'s activate with expert `e+1`'s gate/up costs only the
barriers. `ROUTED_SHARED_PIPELINE` demonstrates the same thing from the other
side, placing the shared expert's cube work beside the routed vector stages for
32 microseconds.

So the skew belongs in the expert loop and in the MLA tail's head loop, not in
the projections. And the earlier upper bound of 1.4 to 2.0 milliseconds, which
assumed the whole layer could be skewed, holds only for the part of the layer
whose cube work is already divided.

## The other side of the criterion, measured

The split build makes one cube invocation a part -- their weights differ, so the
invocations are needed either way -- and skewing the vector half between them
changes the schedule without changing the invocation count. The `hidden` level,
four parts, twelve rows:

| vector passes | serial us | 2 steps | 4 steps |
| ---: | ---: | ---: | ---: |
| 2 | 105.82 | 99.63 (−6.19) | 111.13 (+5.31) |
| 4 | 130.74 | 111.98 (**−18.76**) | 112.84 (−17.90) |
| 8 | 181.17 | 156.92 (−24.25) | 141.08 (**−40.09**) |

Now it wins, and by more as the vector half grows. The cube product is exact at
every point.

How much of the available overlap it captures: at eight passes the serial arm is
181.17 and perfect overlap would be `max(cube 83, vector 100)` or 100, so 40.09
of the 81.17 available is 49 percent. At four passes it is 18.76 of 47.74, or 39.

So the same kernel and the same vector work loses 70 microseconds when the cube
is one invocation and wins 40 when it is four. That is the criterion from both
sides.

## What it gives the FFN

The routed body spends 71.25 microseconds of cube work and 23.40 of vector in a
136.40 window, so 65.15 is available. At the 40 to 50 percent capture measured
here that is 26 to 33 microseconds a layer, or 2.0 to 2.6 milliseconds a round
-- below the 42 estimated from perfect overlap less barriers, and with a
measurement behind it.

## Scope

The cube's product is exact at every depth, maximum difference zero against an
integer reference. The vector half's witness does not verify on the split arms:
it accumulates atomically across steps and no longer matches the closed-form
total, so the vector side has timing but not a correctness gate here. The
witness verified on the unsplit arms, where the vector work is one range.

[`run.sh`](run.sh) is the sweep; the JSON files are its output.
