# What the engine's device spends a round on, deduplicated by name

From `profile-rope-20261018`, rank 1. Rounds from the attention call count,
244 over 78 layers: 3.13. Kernels appearing once across three rounds are the
domain build and are excluded.

    5544 repeated kernels, 57.27 ms of device time a round

    SuperKernel                610.9/round    52.7 us    32.21 ms   56.2%
    hcom_allGather              10.9/round   882.5 us     9.59 ms   16.7%
    hcom_allReduce               9.6/round   831.1 us     7.97 ms   13.9%
    FusedInferAttentionScore    78.0/round    37.5 us     2.92 ms    5.1%
    Cast                       533.5/round     2.6 us     1.40 ms    2.4%
    MatMulV3                     7.7/round    61.2 us     0.47 ms    0.8%
    Muls                        75.1/round     4.5 us     0.34 ms    0.6%
    MatMulV2                    60.4/round     5.4 us     0.33 ms    0.6%

Device busy of 57.27 ms against a measured round near 52.8: the vector and cube
halves run concurrently so the durations add past the wall clock.

## The super kernel does not reduce the kernel count

610.9 SuperKernel launches a round over 78 layers is 7.8 a layer, at 52.7 us
each. A synthetic layer of eight operators measured 214 us in a graph on one
die -- 26.8 us an operator. So the engine runs about the same number of kernels
a layer and each costs twice what the same work costs on a single die at full
width.

That factor of two is the shard: at TP16 each kernel does a sixteenth of the
work and still pays its own fixed cost. It is the same observation the fused
projections rested on -- the router's 3 MB running at 158 GB/s where q_a's
24 MB reaches 1386 -- now visible at the level of the whole round.

## Where the round's time is, in one line each

    32.21 ms   610 kernels a round, each a sixteenth of an operator's work
    17.56 ms   20.5 collectives a round at 858 us
     2.92 ms   attention, outside the super kernel
     1.40 ms   533 casts at 2.6 us

The first two are 87 percent. Fewer, larger kernels is what the five boundary
merges do -- 21.04 ms measured on one die -- and the collectives are what the
one-sided push replaces. Both are built; neither has run end to end.

## Wait time, and what it is not

The capture carries a `Wait Time(us)` column per kernel:

    SuperKernel                32.21 ms   wait  1.02 ms     3%
    hcom_allGather              9.59 ms   wait  1.73 ms    18%
    hcom_allReduce              7.97 ms   wait  1.00 ms    13%
    FusedInferAttentionScore    2.92 ms   wait  1.21 ms    41%
    Cast                        1.40 ms   wait 21.47 ms  1532%
    MatMulV3                    0.47 ms   wait  1.50 ms   319%

Cast at 1532 percent settles what the column means: it is not a part of the
kernel's own duration but the time spent queued before it starts, which can
exceed the kernel's length many times over. So it cannot be used to split a
collective into transfer and synchronisation.

What it does say is that the collectives are not where the round waits -- 18
and 13 percent of their own length, against the Cast family's 1532 -- so their
17.56 ms is largely execution.

## Sizing the one-sided push, honestly

The push was measured at 32.488 us against HCCL's 201.995 at sixteen ranks, a
factor of 6.2, on a single hidden-state reduction. The engine's collectives
cost 858 us each, 4.2 times that HCCL figure, so they carry more or synchronise
more than the microbenchmark did.

    if the 6.2x factor holds at the engine's size   17.56 -> 2.8 ms, saves 14.7
    if it decays to 2x                              17.56 -> 8.8 ms, saves 8.8

So the lever is worth 8.8 to 14.7 ms a round, and its upper end exceeds every
other lever measured today combined. Which is why its arm leads the queue; the
range cannot be narrowed from a single die, because the quantity is a
sixteen-rank collective.

## WITHDRAWN: the capture is width 4 over six tokens

`raw/rank-0.json` records `max_tokens: 6`, `mtp_width: 4`, `target_graph:
True` -- no `--super-kernel`. So this capture is a profiling run of six output
tokens at width four, not decode at steady state.

Every figure on this page derived from it is void. The round count came from
dividing 244 attention calls by 78 layers, which assumed all of them were
decode; at six output tokens and an acceptance near three there are about two
rounds, so the 244 includes warmup and prefill and the per-round division is
meaningless.

What this invalidates, across today's pages: the round's device-time
breakdown (SuperKernel 32.21 ms, collectives 17.56 ms, attention 2.92 ms, Cast
1.40 ms), the collective call count of 20.4 a round, the absorbed-versus-
standalone comparison of 30 us against 858, and the conclusion that collectives
are the round's second largest item.

The ratio between an absorbed and a standalone collective may still hold within
the capture, since both were measured in the same run -- but at width four over
six tokens, with warmup included, even that is a comparison between two
populations whose composition is unknown.

A capture at the decode configuration is the prerequisite for any of this, and
it needs the machine.
