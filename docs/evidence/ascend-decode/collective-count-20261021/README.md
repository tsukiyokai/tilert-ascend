# Forty-one collectives a round, at 830 microseconds each

From `profile-rope-20261018`, rank 1, 3.13 rounds by attention call count.

Grouping by the type column gave allGather 21.7 a round at 882.5 us and
allReduce 19.2 at 831.1 -- 35.1 ms of device time. That is wrong: of the 128
collective records, 64 are a repeated `AivKernel` at 858.4 us and every other
one appears exactly once, at 2126 to 9343 us. The singletons are the domain
build, not the decode, and averaging them into a per-round figure doubled it.

    AivKernel, collective    64 calls   20.4 a round   858.4 us each   17.5 ms

Which agrees with the 17.56 ms a round the earlier by-type sum gave before this
page inflated it. Twenty of these a round at 858 us is the largest item in the
decode after the SuperKernel itself, and the name says they are the vector-side
collectives a super kernel wraps -- which is what the one-sided push replaces.

## What this overturns

The one-sided reduce was costed at 6.5 ms a round from an operator comparison:
32.488 us for the push against HCCL's 201.995 at sixteen ranks, times the 156
calls a TP16 decode was assumed to make -- two a layer over 78 layers.

Both numbers were wrong. The engine makes 41 calls a round, not 156, and each
costs 831 us, not 202. So the comparison was between the wrong message size and
the wrong call count, and it happened to land in the right order of magnitude
for the wrong reasons.

## The question this raises

Twenty-two allGathers a round does not correspond to any obvious layer
structure: 78 layers divided by 21.7 is 3.6. And an 830 us collective at TP16
is large -- the one-sided probe moved its payload in 32 us. So either these
carry far more data than a hidden-state reduction, or they are waiting rather
than transferring.

Which makes the next question concrete and machine-free: find the call sites.
If any of the 41 are redundant, that is worth more than every operator lever
measured today put together.

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
