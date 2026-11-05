# Capture helps the small operators half as much as it helps attention

2026-10-21, a3-84, one die, twelve rows, TP16 shards. Each operator timed on
its own, eager and replayed from its own graph, with the project's own calls
for attention (`cann_attention.shared_kv_fia`'s argument set) and the grouped
matmul (`ops/quantized.py`'s).

| item | eager | graph | ratio |
| --- | --- | --- | --- |
| attention | 240.33 | 81.26 | **0.338** |
| rms_norm | 87.34 | 42.60 | 0.488 |
| dynamic quant | 63.89 | 42.41 | 0.664 |
| o projection | 63.76 | 43.45 | 0.681 |
| q_b projection | 63.05 | 45.57 | 0.723 |
| fused input projection | 59.42 | 60.54 | **1.019** |
| shared down projection | 59.06 | 38.25 | 0.648 |
| the six small, in graph | | 272.82 | |

The grouped matmul's call still rejects its arguments; `npu_grouped_matmul`
wants something `ops/quantized.py` supplies from a loaded layer that this
standalone probe does not reproduce.

## The absolute figures are not usable and the ratios are

Eager sums to 636.87 us against the layer account's 317.476 for the same eight
items. Each operator here is timed alone, so each pays one synchronisation and
one replay launch -- the fixed cost the layer account amortises across a
sequence. So these numbers answer a comparison, not a budget.

## What the comparison settles

The premise behind folding six operators into two was that capture leaves them
expensive. It does: their in-graph total is 272.82 us where attention alone
falls to 81.26, and their mean ratio is 0.70 against attention's 0.338.
Capture removes two thirds of attention's cost and under a third of theirs.

And `fused input projection` at 1.019 gains nothing from capture at all, which
is the sharpest case for folding: an operator whose cost a graph cannot touch
is one whose cost is in the kernel, where only a fold reaches it.

## What it does not settle

Whether the fold can actually take that cost. The ratios say the room exists;
they do not say a kernel can do a norm, a quant and three projections inside
the attention or the grouped matmul without paying the same cost in another
form. That is the next measurement, and it needs a kernel rather than a timing.

## Completed with the grouped matmul, which changes the ranking

The operator's A8W8 quant path wants scales that `ops/quantized.py` takes from
a loaded layer; a standalone probe cannot reproduce them. Two argument sets are
accepted at this shape: INT8 into INT32 without scales, and bf16 by bf16. With
the first:

| item | eager | graph | ratio |
| --- | --- | --- | --- |
| attention | 200.78 | 75.11 | 0.374 |
| grouped matmul | 179.48 | **157.37** | **0.877** |
| fused input projection | 47.30 | 46.22 | 0.977 |
| rms_norm | 46.89 | 32.81 | 0.700 |
| shared down projection | 42.08 | 42.76 | 1.016 |
| q_b projection | 41.05 | 32.82 | 0.800 |
| dynamic quant | 40.64 | 30.78 | 0.757 |
| o projection | 40.20 | 34.68 | 0.863 |

The ratio finding holds -- capture helps the small operators far less than it
helps attention, 0.70 to 1.02 against 0.374 -- but the ranking it implied does
not. **In a graph the largest single item is the grouped matmul at 157.37 us**,
not the six small ones. Capture barely touches it, 0.877.

And the absolute figures drift between runs on the same die: attention 240 then
201, rms_norm 87 then 47. So only ratios are usable, which is what this page
already said, and the drift is the reason.

## Which redirects the search

The six small operators in a graph total 219.87 us against the grouped matmul's
157.37 alone. Both matter, but the grouped matmul is one operator rather than
six, it is nearly immune to capture, and at twelve rows it is flat in the row
count -- 16 to 256 costs 2.8 percent more. An operator that costs 157 us to
process twelve rows and 162 to process two hundred and fifty-six is doing
almost none of its work on the rows it was given.
