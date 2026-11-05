# A reduction absorbed into a super kernel costs 30 microseconds, not 858

From `profile-rope-20261018`, rank 1, 3.13 rounds. Super kernels whose name
declares a collective:

    1.0/round   3282.3 us   3.15 ms   sk_1277_full_model_start_aiv_all_reduce_float_
    1.0/round     31.1 us   0.03 ms   sk_1413_...
    1.0/round     31.0 us   0.03 ms   sk_1501_...
    1.0/round     29.6 us   0.03 ms   sk_1305_...
    1.0/round     29.5 us   0.03 ms   sk_1529_...
    1.0/round     29.4 us   0.03 ms   sk_1345_...
    absorbed total 3.29 ms a round

Against the standalone collectives: 20.4 calls a round at 858 us, 17.56 ms.

So the engine already absorbs some reductions into super kernels, and an
absorbed one costs about 30 us where a standalone one costs 858 -- a factor of
28. `sk_1277` at 3282 us is the exception and is a single call a round, so it
carries something else.

## Which redirects the largest lever

The one-sided push was the candidate for the 17.56 ms. Two-rank scaling says it
is a latency optimisation: 9.91x at the decode shape's 288 KB, 2.17x at 1.5 MB,
and a loss at 6 MB. At 858 us a call the standalone collectives are well past
that crossover, so the push probably does not help them.

Absorption does, and the engine demonstrates it: 30 us against 858. If the 20.4
standalone calls were absorbed the way six already are, 17.56 ms becomes about
0.6 -- **a saving near 17 ms a round**, larger than every lever measured today
together, and the mechanism is already in the codebase rather than needing a
new primitive.

## What to find out next

Why those six are absorbed and the twenty are not. The names are
`sk_<id>_full_model_start_aiv_all_reduce_float_end_...`, so the compiler emits
an absorbed form when some condition holds; the standalone ones are where it
does not. That condition is the lever, and it is a question about the
compilation pass rather than about an operator -- which makes it answerable by
reading, without the machine.

## A conflict, and a configuration that was never checked

The absorption lever rests on a capture whose round is about 52.8 ms, with
17.56 ms of standalone collectives in it. But `ablate-20261018` measures
`--super-kernel` making the round *longer* -- 16.640 to 18.447 ms, +1.807, with
acceptance and round count unchanged -- and a round of 16.6 ms cannot contain
17.56 ms of collectives.

So the two sets of numbers come from different configurations, and this page
drew a lever from one while taking its refutation from the other. The capture
is `profile-rope-20261018`; the ablation is `ablate-20261018`. Same day, same
machine, rounds three times apart, and the configuration difference was never
established before quoting both.

What survives is the within-capture ratio: in that capture, a collective
absorbed into a super kernel costs about 30 us and a standalone one 858. That
is a real factor of 28 between two forms measured side by side. What does not
survive is the 17 ms: it scales a per-call difference by a call count from one
configuration against a round from another.

## What has to happen before this lever is quoted again

Establish the capture's configuration -- width, drafter, flags -- and whether
`--super-kernel` was on. If it was off, the six absorbed collectives came from
the `enable_static_kernel` path and `--super-kernel` is the switch that would
absorb the rest, in which case the ablation's +1.807 ms is the measurement that
contradicts the lever and needs explaining. If it was on, the twenty standalone
calls are what absorption leaves behind and the lever does not exist.

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
