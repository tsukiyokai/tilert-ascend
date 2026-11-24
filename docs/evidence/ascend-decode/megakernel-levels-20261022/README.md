# What the concatenation is worth, level by level

The kernel is parameterised by the level -- its reduction length and the
projections that read the shared activation -- so adding one costs a dictionary
entry. Three levels say the deciding variable is the reduction's depth, and they
correct an earlier reading of this stage.

| level | reduction | columns | parts | rows | fused us | split us | saving us |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| prolog | 6144 | 2624 | 2 | 1 | 24.74, 24.35 | 55.29, 55.21 | 30.55 |
| prolog | 6144 | 2624 | 2 | 12 | 24.86, 25.58 | 56.64, 55.65 | **31.06** |
| prolog | 6144 | 2624 | 2 | 32 | 27.09, 26.12 | 56.88, 57.13 | 30.04 |
| hidden | 6144 | 3040 | 4 | 12 | 40.31, 40.86 | 83.74, 83.76 | 42.90 |
| query_norm | 2048 | 5120 | 2 | 12 | 31.59, 31.59 | 29.43, 29.36 | −2.16 |

Two alternating passes a row count, eight changing inputs each, one die. Every
arm's product digests are identical to its counterpart's in every case.

## Only `prolog` is a grouping the engine can adopt

The engine's first level does not hold four projections of one activation. Read
from the native sources: Prolog takes `token_x` as int8 and projects
`weight_dq` and `weight_dkv_kr` from it, so those two share an activation; the
indexer's key takes `args.hidden` as **bf16**, a different dtype that cannot
join them; the indexer's query and the query up-projection read the query norm,
which is the second level; and the router's gate is read inside
`ffn_bridge::Forward`, after attention entirely.

So `prolog` is the adoptable grouping and its saving is 31.06 microseconds a
layer, or 2.42 milliseconds a round. The `hidden` row keeps all four to show
what the mechanism does with three invocations removed rather than one; it is
not a figure the engine can take.

## Depth decides, not the invocation count

Both two-part levels remove exactly one invocation. At a reduction of 6144 that
is worth 31 microseconds; at 2048 it costs 2. The fused arm at the short depth
is slower than two invocations -- 31.59 against 29.43 -- so a single wide slice
is not simply better. One invocation at depth 6144 costs about 25 microseconds,
which is the same order as the 18.5 measured at the Torch level.

The fused arm is selected at `prolog` and the split arm at `query_norm`.

## The limit on reading these numbers

The operands are bf16 and the product float, which isolates the invocation count
from the quantization contract. Prolog's real activation is int8 with a token
scale and its weights are W4A8 with channel scales, so it reads a fraction of
these bytes. The saving counts invocations rather than bytes, but the engine's
figure needs its own measurement at its own dtypes.

[`run.sh`](run.sh) is the alternating runner; the JSON files are its output, one
a level an arm a row count a pass.
