# The layer costs 210 microseconds, and the 374 us gap was an artefact

A kernel census of `profile-rope-20261018`, rank 1, non-collective kernels:

    690 distinct kernels, 5544 calls, 179.14 ms

    AivKernel                                   64   54.94 ms   858.36 us
    sk_1277_full_model_start_aiv_all_reduce      3    9.85 ms  3282.35 us
    aclnnFusedInferAttentionScoreV4            244    9.14 ms    37.46 us
    aclnnInplaceCopy_CastAiCore_Cast          1590    4.31 ms     2.71 us
    sk_681_full_model_start_static               3    0.65 ms   215.38 us
    sk_1195_full_model_start_static              3    0.64 ms   212.67 us
    sk_1096_full_model_start_static              3    0.64 ms   211.75 us
    sk_799_full_model_start_static               3    0.63 ms   211.24 us
    ... dozens more of the same shape, 209 to 215 us each

## What the names say

`sk_*_full_model_start_static` is the SuperKernel: one per layer, called three
times -- once a measured request -- at **209 to 215 us**. That is the layer,
and it matches the synthetic layer's in-graph 214 us to within two percent.

So a layer costs about 210 us in the engine, not 588.

## Which makes the 374 us gap an artefact of my own arithmetic

The 588 us came from dividing the round model's fitted fixed term, 45.88 ms, by
78 layers. That fit holds everything in a round -- the standalone collectives,
the drafting, the Gloo command, the MTP recomputation, prefill remnants in a
six-token capture -- and attributing all of it to layers put 374 us a layer
where no layer spends it.

Four explanations were measured and excluded for a quantity that did not exist:
host-side issue, the native device cost, arrival skew, and message size. Each
exclusion is sound on its own terms and none of them was the answer, because
the question was wrong.

## The round's actual shape, from this census

    78 SuperKernels at 210 us                16.4 ms
    64 standalone collectives at 858 us      54.9 ms over 3 requests, 18.3 a request
    244 attention calls at 37.46 us           9.1 ms over 3, 3.0 a request
    1590 casts at 2.71 us                     4.3 ms over 3, 1.4 a request

The collectives dominate and they are outside the SuperKernel. That is where
the round's time goes, and the capture's six-token scope means the absolute
figures still need a steady-state run -- but the structure is now right.

## And two thirds of a round is not on the device

The record's own account of the best arm, `lever84-20261020/auto12`, drafter
`lookup`, width 12, `mtp None`, `draft_graph false`:

    operation           calls   total ms   ms each   share
    advance_mtp_n         188    9602.50    51.077    98.9%
    inject                  1      70.12    70.117     0.7%
    reset                   1      18.30    18.303     0.2%
    start_mtp               1      10.64    10.641     0.1%

    advance rounds: 188, p50 50.710 ms, min 47.686, max 68.339

Each `advance_mtp_n` carries `rounds: 1`, so 50.710 ms is one round entire.
Against it, the kernel census puts 78 SuperKernels at 210 us -- **16.4 ms**.

So about 34 ms of a round is outside the SuperKernel. What it is not:

- MTP recomputation. The arm runs `mtp None` with no draft graph, so the draft
  model is absent.
- The standalone collectives. The census counts 64 over three requests, which
  is 21 a request against 188 rounds -- 0.11 a round, not 34 ms of them.

What remains is the host: the drafter, the acceptance match, the Gloo command
broadcast, and whatever the round does between replays. The `operations` list
does not break those out, so the next step is instrumenting inside
`advance_mtp_n` rather than another device capture.

## Which reorders every lever

If two thirds of a round is host-side, the three levers measured today -- AIV's
collectives, the bank narrowing, attention redundancy -- all act on the 16.4 ms
that is on the device. Their combined effect on a 50.7 ms round is bounded by
that third.

The host side has never been measured at all.

## Correction: the capture holds three rounds, not three requests

Each `sk_*_full_model_start_static` is called three times, and a layer's super
kernel runs once a round. So the capture holds **three rounds** -- consistent
with its `max_tokens: 6` -- and the previous section divided its call counts by
a request's 188 rounds instead.

Redone per round:

    78 super kernels at 210 us                        16.4 ms
    21.3 collectives at 858 us (64 over three)        18.3 ms
    81 attention calls at 37.46 us (244 over three)    3.0 ms
    530 casts at 2.71 us (1590 over three)             1.4 ms
    accounted                                        ~39.1 ms

Against a measured round of 50.710 ms, that leaves about 12 ms rather than 34.

## Which puts the collectives back as the largest single item

18.3 ms a round, more than the super kernels themselves. The claim that they
are "0.11 a round, not 34 ms of them" used 188 as the divisor for a capture
containing three rounds.

So the ordering of today's levers reverses again, and this time with the
arithmetic checked: AIV acts on the largest item in the round. At 1.5 to 1.85x
on the collectives it is worth 6 to 8 ms a round by itself.

The remaining 12 ms is the host side the new instrumentation measures, and it
is a quarter of the round rather than two thirds.
