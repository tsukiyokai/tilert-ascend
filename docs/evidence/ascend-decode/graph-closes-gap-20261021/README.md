# A captured layer replays at its device-busy time, and that reopens 2.x

2026-10-21, a3-84, one die, twelve tokens, TP16 shards.

    eager                  315.031 us a layer
    graph replay           213.913 us a layer   ratio 0.679
    device busy, profiled  202.514 us a layer   66 percent of eager

Replay lands 5.6 percent above the device-busy figure, so capture removes
essentially all of the 32 percent that eager spends on issue and
synchronisation.

## Which corrects yesterday's -- and this morning's -- accounting

The layer account summed its parts to 317.476 us and matched the round model's
post-merge fixed term of 24.84 ms over 78 layers to a third of a percent. That
agreement was read as "the round has no unattributed remainder", and the
conclusion drawn was that 2.x ms lies outside the identified mechanisms.

But 317 us is the **eager** figure and the engine decodes inside a graph. The
same layer in a graph is 214 us. So either the engine is already paying 214 and
the round model's fixed term contains something my synthetic layer omits -- the
sixteen-rank reductions, `init_routing`, `token_unpermute`, the gating, the
second grouped matmul -- or the engine is paying the eager figure despite its
`cudagraph_mode: FULL_DECODE_ONLY`.

The arithmetic distinguishes them: 45.88 ms before the merges is 588 us a
layer, which is 2.7 times the graph figure and 1.9 times the eager one. So the
fixed term holds more than my layer, and the right comparison is after the
merges remove exactly those omitted items.

    fixed term after the merges     24.84 ms   318 us a layer
    the same layer in a graph                  214 us a layer
    difference                       8.1 ms    104 us a layer

If the post-merge engine replays its layer at 214 us rather than 318, the
round is 16.7 ms and

    TPOT = (16.7 + 0.66 x 24) / 10.918 = 2.98 ms

which is the target. So 2.x is back inside the identified mechanisms, on one
condition: that the merged engine's layer replays at its device-busy time.

## What to measure next

Whether the engine's graph replay already achieves the device-busy figure. The
round model was fitted from arms that all ran with the graph on, so if their
588 us a layer is replay rather than eager, the 104 us gap is in the items my
synthetic layer omits and the merges are the way to collect it -- which is
what the queued arms measure. If instead the engine is somehow paying eager
cost inside a graph, that is a separate and larger lever.

## Correction: the acceptance this page used is above the measured ceiling

The 2.98 ms above takes 10.918 accepted, which is the copy oracle's figure.
Today's saturation sweep measured what real drafters converge to with the row
cost set to zero and the budget taken to 256: the tree 4.289, the chain 3.400.

At the measured ceiling the same graph figure gives

    round 16.69 ms, rows free          TPOT 3.89 ms
    round 32.53 ms with 24 rows        TPOT 7.58 ms

So this page's conclusion -- that 2.x is back inside the identified mechanisms
-- is void, on an acceptance 2.5 times the ceiling.

What survives is better than what it replaces, because the graph figure is
sourced and the comparison is now the right one:

    2.9 ms at 4.289 accepted needs      159.5 us a layer
      against eager 317.5               -50%
      against the graph figure 213.9    -25%
      remaining                          54.5 us a layer, 4.25 ms a round

Twenty-five percent device-side, not fifty, because capture already removes the
host boundary and the engine has `FULL_DECODE_ONLY`. That is the gap the
within-layer fold has to close, and it is a quarter rather than a half.
