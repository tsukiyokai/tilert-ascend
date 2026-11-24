# What 55.4 ms a round is made of, and what it could be

2026-10-20. Shapes from the checkpoint's own `config.json`; no machine needed.

## The floor

    active parameters a token   39.98 G
      attention 12.87 G   moe 25.48 G   dense 0.68 G   head 0.95 G
    weights a die at W4          1249 MB
    MLA latent at 8K, int8        368 MB
    bytes a round a die          1618 MB
    HBM floor a round              1.011 ms   at 1600 GB/s

Seventy-eight layers, hidden 6144, 256 experts of intermediate 2048 with eight
routed and one shared a token, three dense layers at intermediate 12288, MLA
with `kv_lora_rank` 512 and `v_head_dim` 256 over 64 heads, vocabulary 154880.
TP16, so a die holds a sixteenth of the weights; the latent is not shardable,
so each die reads all of it.

## Where that leaves the target

    measured round               55.424 ms
    above the floor                 54.8x
    a 2.x ms tpot needs          ~15 ms at acceptance 5.441
      which is                      14.8x the floor

So the goal is not against physics. Fifteen milliseconds is still fifteen
times the bytes' own cost; the question is entirely what else the round is
spending.

## The two kernels everyone optimises are a third of it

    per layer, measured:  attention 89.4 us   moe vendor 157.2 us
    all layers:           19.2 ms
    unaccounted:          36.2 ms   -- 65% of the round

Sixty-five percent of the round is neither the attention kernel nor the
grouped matmuls. It is the norms, the router, the residuals, the TP
reductions, and above all the per-kernel issue and synchronisation across
seventy-eight layers.

That reorders the work. My fused INT4 expert kernel replaces 157.2 µs a layer
with 37.2 µs, which takes the accounted part from 19.2 ms to 9.9 ms -- about
nine milliseconds off a fifty-five millisecond round. Real, and not a factor
of four. The factor of four is in the 36.2 ms, which is what a megakernel
addresses and what no amount of kernel-level tuning will.

## What this does not establish

The 36.2 ms is a residual, not a measurement: it is what is left after
subtracting two per-call figures measured in isolation, and those figures were
taken at shapes that may not match what the engine issues -- the vendor MoE
number in particular is for 256 experts with a mostly-zero group list. The
next step is a profile of one round that counts kernels and attributes the
gaps, rather than a subtraction.
