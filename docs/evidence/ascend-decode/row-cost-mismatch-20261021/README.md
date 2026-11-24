# The round model's 0.814 ms a row is 31 times what the operators cost

2026-10-21, a3-84, one die. A layer's operators -- norm, the fused input
projection, q_b, o_proj, the shared down projection, the dynamic quant and the
grouped matmul -- at four token counts:

     tokens   layer us   marginal us a token
          8    135.190
         12    132.616               -0.644
         16    136.508                0.165
         32    143.263                0.336

Eight microseconds between 8 tokens and 32. Over 78 layers that is
**0.026 ms a row**, against the round model's fitted **0.814**.

The fit is sound -- four widths, residuals of a few hundred microseconds on
values of 52 to 72 ms, and 19.3 ms separating width 8 from width 32 -- so the
cost is real and it is not in the operators. It is in something a single die
cannot reach:

- the sixteen-rank TP reduction, whose bytes do scale with rows (147 KB at
  width 12 against 393 KB at 32), though small-message HCCL is latency-bound
  and should not produce 19 ms; or
- the graph's static token buckets, where a wider width lands in a larger
  bucket and pays for it.

The drafter is excluded: the automaton and the tree each cost 0.1 to 0.3 us a
round on the host, measured on CPU over an 8192-token context.

## Why this decides the target

If the row cost is an implementation artefact rather than physics, the
constraint that has shaped every drafting decision today disappears. A tree's
nodes stop being expensive, and the fixed-budget tree -- 14.510 accepted a
round at an edit rate of 0.02, which the adaptive budget gave up as too costly
-- becomes the right configuration.

    fixed term after the boundary merges   25.58 ms
    64 nodes at 0.026 ms a row              1.66 ms
    round                                  27.25 ms
    accepted                               14.510
    TPOT                                    1.88 ms

Against 3.90 ms for the adaptive tree at the fitted 0.814. So this one number
is worth more than every lever measured today put together, and it is a
question about the engine rather than about an operator.

## The discriminator

Width 12 against width 13. If the cost is the graph's buckets, two widths in
one bucket differ by the operator figure -- about 0.03 ms -- while two widths
across a boundary differ by a bucket's worth. If it is the TP reduction, the
difference is smooth in the width. One pair of arms settles it, and they need
the whole machine.

## Three candidates excluded, and what is left

- **Graph buckets.** The engine sets `cudagraph_capture_sizes: [graph_width]`,
  one size, so every width captures its own graph and there is no step to pay
  at a boundary. Excluded by reading the config.
- **The TP reduction.** Two ranks, hidden 6144, float32: 111.27 us at 8 rows
  and 93.24 at 64, an eightfold range in bytes with a negative slope
  throughout. Not row-sensitive. Sixteen ranks may differ but the direction
  does not support it.
- **Per-row operator cost.** The page this corrects attributed 10.4 us a layer
  a row to scalar and issue pressure. A whole layer measures 0.336 us a row.
  Excluded, 31 times over.

What remains is trajectory coupling. The four widths each ran their own
trajectory: a wider draft accepts more, so each round advances further, the KV
grows faster, and later rounds pay more for attention. `round(width)` then
mixes the width's own cost with the context length its acceptance produces,
and the 0.814 ms a row is mostly the second.

If that is right, widening does not cost what the model says, a tree's nodes
are not expensive, and the fixed-budget tree at 14.510 accepted becomes
reachable -- the 1.88 ms line above.

## The discriminator, corrected

Not adjacent widths -- buckets are excluded, so that tests nothing. Fix the
number of tokens generated rather than the number of rounds, and compare
widths over the same trajectory length. If the round cost is flat in the width
at constant trajectory, the row term is an artefact of the measurement.

## Five candidates excluded; the slope is sound and what it measures is not identified

Added to the three above:

- **Trajectory length.** The arms cap at `--max-tokens 1024`, so every width
  generates the same token count over the same growing context. Width changes
  the round count, not the trajectory.
- **The Gloo command broadcast.** A round's scope is "Gloo, draft chain,
  verify, MTP recompute, commit", and the Gloo part is one
  `broadcast_object_list` a round. Measured over two ranks it is flat in the
  draft size: 305.0 us at width 0, 289.6 at 64.

So the fitted 0.814 ms a row is real -- four widths, 19.3 ms across the range,
residuals of a few hundred microseconds -- and none of the five mechanisms that
could produce it survives measurement. The honest position is that the slope is
sound and what it measures has not been identified, which is why the queued
width arms lead the queue: this one number decides whether a tree's budget is
affordable, and with it whether 2.x ms is reachable.

## A smaller finding on the way past

The Gloo command broadcast costs 305 us an object against 106 us for an
equivalent integer tensor -- threefold on two ranks, and object broadcast
pickles, sends a length, then sends bytes, so sixteen ranks will be worse.
One a round at 55 ms is half a percent on two ranks and more on sixteen. The
fix is to broadcast a fixed-size tensor rather than a Python object.

The other half of that suggestion was already the case: reading the command
construction, the payload is scalars and fixed tuples -- `token`,
`first_token`, `max_tokens`, `width`, `eos_tokens` -- with only `inject`
carrying a dict, once a request. The draft tokens are not broadcast at all;
each rank already computes them. Which is also why the Gloo measurement came
out flat in the width.

## Settled from the records: it is the width's fixed cost, not the row's marginal cost

Each arm's own rounds, grouped by how many drafted tokens matched that round:

    auto12   0 matched  76 rounds  median 50.49 ms
             6 matched   7 rounds  median 50.65 ms
    auto16   0 matched  79 rounds  median 53.30 ms
             6 matched   5 rounds  median 53.90 ms

Within one arm, six more matched drafts cost 0.16 to 0.60 ms -- 0.03 to 0.10 ms
a row, which agrees with the 0.026 measured on one die. Across arms the
medians are 48.2, 50.5 and 53.5 ms at widths 8, 12 and 16: 0.66 ms per unit of
width.

So the slope is real and it is the **width's** cost, paid every round whatever
the draft achieves, not the row's marginal cost. The engine captures one graph
at `graph_width`, so a wider graph is more expensive to replay even when most
of its rows carry nothing.

## What that does to the tree

It reverses the earlier reading. A tree's budget is expensive -- but because the
graph must be captured at that width, not because its nodes cost anything to
verify. A fixed-budget tree of 64 nodes pays width-64 replay every round.

Which points somewhere specific: match the captured width to the tree's typical
node count rather than to its ceiling. The adaptive tree's mean is 13.1 nodes
against a 64 ceiling, so a graph captured at 16 and a tree capped at 16 pays
53.5 ms a round and accepts what a 13-node tree accepts -- while a 64-wide
graph would pay about 85 ms for the same work.

That is also why width 12 won the original sweep, and it means the drafting
work should be measured at a fixed captured width, with the tree shaped to fill
it rather than to exceed it.

## Where the width cost is, and two readings corrected

Attention does not carry it. The gathered FIA call from `fia-mask-20261020`,
which shapes key as `(width, 1, picks, 512)` so every row carries its own
selected positions:

    width  picks        us   us a row
        8   2048   109.057     13.632
        8   1024   106.797     13.350
        8    512   103.898     12.987
       16   2048   107.370      6.711
       16    512   108.196      6.762

Flat in both. Four times fewer picks moves it five percent and twice the width
not at all. So the per-row scoring hypothesis is wrong, and the earlier
rejection of `--index-topk` as a lever was right for a different reason than
it gave.

The graph records show one capture and one entry a width, with
`BatchDescriptor(num_tokens=width)`. The `replays` counter -- 1350, 1128, 1098
-- divided by one request's 225, 188 and 183 rounds gives exactly 6.0 each,
which looked like six replays a round. It is not: three warmup requests and
three measured ones at the same round count is 6 x 225 = 1350. One replay a
round, and the six was two record scopes divided by each other.

## Attention is not excluded after all

The operator measured above is `npu_fused_infer_attention_score_v2` with a
gathered layout. The engine calls `npu_sparse_flash_attention` with
`layout_kv="PA_BSND"` and a `block_table` -- a different operator on a paged
cache, and the one whose row sensitivity four attempts failed to measure (its
tiling function rejects every argument set tried here). So the flatness
measured belongs to the substitute, not to the operator the round pays for.

Attention therefore remains the leading candidate for the width cost: it is
the only operator in a layer whose work is per query row against a large
context, and it is the only one not measured. The paged capture's query shape,
`1,4,512` under TND, is one token -- which is either a draft forward from the
MTP path or a phase other than verify, and resolving which needs the engine.

So the honest position is narrower than the previous paragraph claimed: the
width cost is paid inside graph replay, every operator that could be measured
on one die is flat in the rows, and the one that could not be measured is the
one most likely to carry it.

## Five attempts, and the operator stays unmeasurable on one die

`aclnnSparseFlashAttention` rejects every argument set tried here with
`Execution_Error(EZ1008): Failed to execute tiling function`, across five
combinations: with and without `query_rope`, selection shaped `(1, rows,
picks)` and `(rows, picks)`, per-row lengths and cumulative starts, and the
MLA and plain head widths. The engine runs it, so what is missing is context
this probe does not reproduce -- most likely the page view
`_view_cache_as_operator_pages` builds, or the persistent sinks
`SparseMLAMetadataState` owns.

Recorded as unmeasurable from a single die rather than pursued further: the
queued width arms measure the same quantity end to end, which is where the
answer has to come from anyway.

## Attention is the candidate that fits, from two captures

The engine's sparse attention could not be run on one die -- six argument sets
rejected, five with a tiling failure and the paged `FusedInferAttentionScore`
with a shape error. So its row scaling is taken from two captures instead:

    attention-sparsity-20261020, width 8   85.048 us a call
    profile-rope-20261018, mtp_width 4     37.500 us a call

Two points, 11.9 us a row, which over 78 layers is **0.93 ms a row** against
the round model's fitted 0.66. The same order, and the only candidate measured
today that is.

The caveat is worse than it looked. `attention-sparsity-20261020` does not
name its capture, and the three on the machine are all `mtp_width: 4` while
that page reports 140 `hcom_allGather_` calls against `profile-rope`'s 68. So
its 85.048 us came from a fourth capture of unknown configuration, and the two
points are not two widths of one thing -- they are one measurement and one
number whose provenance is lost.

So attention's row scaling is unmeasured, not weakly measured. What stands is
only that it is the only term large enough to carry the slope.

What it would mean if it holds: the row cost is attention scoring each query
row against its selected positions, which is irreducible work rather than
overhead, and the 0.66 ms a row is physics. Then the copy oracle's floor of
4.18 ms stands as the floor, and 2.x needs either fewer layers of attention or
a shorter selection -- and `--index-topk` was measured flat, 103.898 us at 512
picks against 109.057 at 2048.

The way to settle it is one capture at two widths, which the queued arms
produce as a side effect: `cap12` and `cap24` both carry `--profile-last`.
