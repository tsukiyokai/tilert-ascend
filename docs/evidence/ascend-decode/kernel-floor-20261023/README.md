# The per-kernel floor, rechecked on today's stack: fusion saves 45 percent

2026-10-23, a3-180, one die, no model, `bench_kernel_floor.py`, with a tenant
holding 40781 MB a die -- which is why this was the measurement available.

The whole route to 2.x rests on one claim: a round's cost is its pass count,
not any pass's inner loop, so fusion is the only lever. The numbers behind it
were measured on another host and another CANN. Rechecked here:

| operation | median | min | p90 |
| --- | --- | --- | --- |
| add_32_rows | 11.836 us | 10.751 | 12.401 |
| foreach_add_32_rows | 51.752 | 50.626 | 53.381 |
| cast_round_trip_8_rows | 26.965 | 25.818 | 29.135 |
| scale_then_add_8_rows | 27.635 | 26.238 | 29.788 |
| fused_axpy_8_rows | 15.083 | 13.656 | 15.595 |

A kernel that moves almost nothing costs **11.836 us**, which is the 12 us a
pass the megakernel ledger prices with. Two steps as two kernels cost 27.635;
the same arithmetic as one kernel costs 15.083 -- **45 percent less, and the
difference is 12.55 us, one kernel floor**. So the claim holds on this stack
and with this CANN, and it is the saving per fused pass rather than an
extrapolation.

`foreach_add_32_rows` is 4.4 times `add_32_rows`: the batched API pays more
floor than it saves at this size, so it is not a fusion route.

## The 28.8 us a kernel the fixed term is priced with reads the host

The same 12-element scalar multiply, three ways on the same die:

| timing | 12 elements | 32 x 6144 |
| --- | --- | --- |
| back to back, one sync a batch | **13.931 us** | 14.371 |
| eager event pair | 47.220 | 45.700 |
| host enqueue (`perf_counter` around the call) | 16.696 | 16.450 |

The round model's 28.824 sits between the eager pair and the host enqueue, not
at the in-stream floor, which is 13.931 -- and this project's own lesson is
that an eager event pair can read the host rather than the device.

Sixteen thousand times the payload costs 3.2 percent more time, so at this
scale the cost is the kernel floor and nothing else.

## What that does to the route

The fixed term was decomposed as 25 kernels a layer x 28.8 us x 78 = 56 ms,
which is the whole of a measured 50.73 ms round. At the measured floor it is
25 x 13.9 x 78 = **27.1 ms**, or 53 percent of the round. So the pass budget's
saving is overstated by about a factor of two, and **23.6 ms a round now has
no owner** -- the round's composition has to be re-derived before 48 passes to
11 can be claimed to deliver 2.9 ms.
