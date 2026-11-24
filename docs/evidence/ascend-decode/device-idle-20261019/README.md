# The device is idle 39% of steady state, and the sampler holds two thirds of it

Read from `profile-rope-20261018`'s `trace_view.json`, rank 0, by union of
busy spans rather than by summing per-operator times — which double counts
concurrent tracks and is how the earlier 35 ms overhead figure was built.

Scripts: `sk_members.py`, `event_wait.py`, `device_union.py`, `idle_shape.py`,
`gap_cause.py` in this directory.

## Two things cleared off the board first

`EVENT WAIT` is the largest single item on the device timeline — 404.290 ms
over 599 spans, 675 µs each, more than the SuperKernel groups' whole 103.587
ms. It is **not** a lever: 403.907 ms of it, 99.9%, is overlapped by work on
another track, and only 16 of 599 waits have nothing running at all. A wait on
one stream costs nothing while another computes through it.

`Cast` on the device costs 2.86 µs per span and 4.551 ms over 1590 spans for
the whole profile. The family I spent a day planning to attribute is cheap
where it runs. What was expensive was the host side of it, which the hoist
already took.

## The real split

| | |
| --- | --- |
| profile wall | 618.496 ms |
| device busy, union | 318.201 ms (51.4%) |
| device idle | 300.295 ms (48.6%) |

That figure covers startup: its largest gap is 111.0 ms, which is compilation
or capture and not a decode stall. Windowing to the middle half of the 471
`all_reduce_float` groups gives steady state:

| | |
| --- | --- |
| steady window | 249.141 ms, 236 reductions |
| device busy, union | 151.179 ms (60.7%) |
| device idle | 97.962 ms (39.3%) |

The idle's shape decides what kind of problem it is, and it is unambiguous:
172 gaps of 100 µs or more sum to 97.194 ms — essentially all of it — while
239 gaps under 10 µs sum to 0.479 ms, 0.2%. So at this granularity decode is
**not** launch-rate bound. It stalls for milliseconds at a time.

## What the stalls are

The span that ends at a gap's start names what the device was last told to do.
Ranked by the idle each precedes, within the steady window:

| idle preceded | gaps | each | operator |
| --- | --- | --- | --- |
| 34.938 ms | 36 | 970.5 µs | `aclnnIndex` |
| 13.706 ms | 4 | 3426.5 µs | `aclnnArgMax` |
| 9.870 ms | 31 | 318.4 µs | `aclnnInplaceCopy` |
| 8.200 ms | 26 | 315.4 µs | `aclnnInplaceFillScalar` |
| 2.073 ms | 5 | 414.6 µs | `aclnnMatmul` |
| 1.826 ms | 4 | 456.4 µs | `aclnnIndexSelect` |

Those four families hold 66.714 ms of the 97.962 ms, 68%. And what starts
after a gap is the same list — 36 `aclnnIndex`, 32 `aclnnInplaceCopy`, 26
`aclnnInplaceFillScalar`. A chain of the same operator separated by
millisecond gaps is the signature of a host round trip per call.

None of this is the model forward. It is the sampler and the speculative
decode's bookkeeping: argmax, index, fill, copy.

## Why that is the connection to the AMD binary

Their shipped library has `top_p`, `top1_allreduce` and
`broadcast_selected_token_ids` as device kernels, the last two carrying
`void* const*` peer pointers — their sampler and its cross-rank agreement run
on the device. Ours runs as a chain of host-synchronous `aclnn` calls, and
pays a host round trip per element of that chain.
[their kernel census](../../tilert-ascend/amd-article-reread-20261018.md)

## The caveat that bounds all of it

A profiled run is not a timed run. The profiler adds host-side work, which
starves a device whose feeder is the host — so the 39.3% idle fraction cannot
be transferred to an unprofiled run, and neither can the 66.714 ms. What does
transfer is the ranking and the shape: the stalls are large, they are at the
same four families on both sides, and those families are the sampler.

Settling the magnitude needs the sampler's own clock in an unprofiled run,
which is the next measurement and does not need the whole machine.
