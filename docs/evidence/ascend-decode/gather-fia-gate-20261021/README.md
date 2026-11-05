# The width-six gather_fia gate: four places read, cause not found

`cann_attention.GATHER_FIA_MAX_WIDTH = 6` with a comment -- "widths above six
failed serial-equivalence checks" -- and no evidence page behind it. Read today
with no machine:

## What `gathered_fia` does with the width

    width = ql_nope.shape[0]
    indices = topk_indices.reshape(-1)
    latent = kv_cache[0].view(-1, 512).index_select(0, indices)
    key = latent.view(width, 1, 2048, 512)
    ... actual_seq_qlen=[1] * width, actual_seq_kvlen=[2048] * width

Fully parameterised. The width comes from the input's own shape and every
reshape and sequence list is built from it. No constant, no assumption, nothing
that changes at seven.

## What the topk buffer does

`sfa_v1._update_indexcache_topk_indices` writes into
`self.topk_indices_buffer[:num_tokens, :topk_tokens]`, so the buffer is
two-dimensional and sliced by the round's own token count. It imposes no width
limit, and it is the write path rather than the read path `gathered_fia` uses --
which receives `topk_indices` as an argument.

So the non-contiguity hypothesis fails: the indices `gathered_fia` reshapes are
the caller's tensor, not a slice of that buffer.

## Excluded, and what is left

Excluded today: a hardcoded width in `gathered_fia`, a width limit in the topk
buffer's shape, non-contiguity from buffer slicing, and the sentinel-index
suspect closed earlier in this project.

What is left is what the comment describes and nothing records: a
serial-equivalence check that failed at width 8 or above, whose output nobody
kept. The gate is therefore a recorded observation with no located cause, and
the cheapest way forward is to re-run the comparison and read the failure rather
than to keep reading the code -- four readings have now found nothing.

## The comparison, re-run: the boundary is real and the cause is numerical

One die, the gathered path against the same attention computed one query row at
a time, which is what serial equivalence means. Random selections, 2048 rows a
query, bf16:

| width | max abs | max rel | verdict |
| --- | --- | --- | --- |
| 1 | 0 | 0 | exact |
| 2 | 0 | 0 | exact |
| 4 | 0 | 0 | exact |
| 6 | 0 | 0 | exact |
| 8 | 0.000976562 | 0.194073 | differs |
| 12 | 0.00195312 | 0.164032 | differs |
| 16 | 0.000976562 | 0.171661 | differs |

So the boundary is exactly where the constant says, and the failure is one bf16
least-significant unit -- 0.00098 is 2^-10, the gap between neighbouring bf16
values near 1. The relative figure of 0.17 comes from dividing by elements near
zero, not from a large error.

## Which changes what the gate means

Not a logic error. A misindexed gather or a wrong sequence length would
disagree by order-one amounts, not by one unit in the last place. What changes
at eight is the operator's internal blocking: `actual_seq_kvlen=[2048] * width`
batches the queries, and a different width gives a different reduction order
over the same 2048 keys.

So the gate is correct if the criterion is bit-exactness and unnecessary if the
criterion is a tolerance. That is a decision rather than a defect, and it is
now a measured one: four readings of the code found no cause because there is
no error in the code.

## What this leaves for a width above six

The project's own numerics gate is what decides. If it accepts a bf16 ULP on
the attention output -- and the W4A8C8 checkpoint's own quantisation is far
coarser -- then `GATHER_FIA_MAX_WIDTH` can be raised and the gathered path
becomes available at the widths the drafters actually want, 12 to 24.

## And raising the gate would be slower anyway

Attention is 308.97 us of a TP16 layer's compute, the largest item with no
lever, and the gathered path reads only the 2048 rows `topk_indices` selects
against the full 8192. One die:

| width | full 8192 | gathered 2048 | ratio |
| --- | --- | --- | --- |
| 4 | 183.07 | 214.30 | 0.85 |
| 8 | 193.27 | 229.54 | 0.84 |
| 12 | 196.23 | 243.57 | 0.81 |
| 16 | 198.91 | 251.83 | 0.79 |
| 24 | 194.90 | 257.88 | 0.76 |

The gathered path is slower at every width and worsens as the width grows.

The reason is the bytes. Selecting 2048 rows reads them **per query row**, so
the total is `width x 2048 x 576 x 2` -- 28 MB at width 12 -- where the full
cache is 9.4 MB read once and shared by every row. At twelve rows attention is
dominated by that read, so quartering the logit count while quadrupling the read
loses.

## Which closes both threads at once

The width-six gate is not only numerically right, it protects a path that is
already 19 percent slower at width 12. So `--gather-fia-max-width` stays at its
default, and the lever it looked like does not exist.

And attention's 309 us a layer keeps its standing as the compute half's largest
item with nothing yet aimed at it -- now with one more candidate measured and
excluded.
