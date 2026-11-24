# A tree that is sound, a chain at 14.743 ms, and an oracle that tells them apart

2026-10-23, a3-180, sixteen dies, 78 layers, the 8K capture, 1024 output
tokens. `shared_fia` reads the whole cache behind a mask it builds from the
indexer's selection, so a tree only has to replace that mask's causal half --
the indexer keeps choosing the prefix at the checkpoint's own 2048 picks.

| arm | TPOT | accepted | distinct | in prompt | gate |
| --- | --- | --- | --- | --- | --- |
| `shared_fia` chain, width 8 | **14.743 ms** | 3.693 | 32 | 0.338 | pass |
| `shared_fia` tree, width 32 | 17.044 | 4.263 | 99 | 0.281 | — |
| `shared_fia` chain, width 32 | 29.148 | 2.283 | 188 | 0.368 | pass |
| `sfa` chain, width 12 | 22.135 | 2.590 | 139 | 0.316 | — |
| `sfa` tree, width 32, no positions or compaction | 7.629 | 9.300 | 17 | **0.010** | void |
| `sfa` tree, width 32, compacted, lossy mask | 16.841 | 7.307 | 37 | **0.027** | void |

## Output identity is not an oracle, and the measure that is

Two arms that both pass -- the same kernel, the same drafter, widths 8 and 32
-- diverge at token **40**. Each width blocks the operator's reduction
differently, which `cann_attention` records as 2 to the minus 10, and
speculative verification turns one such difference into a different token.
Nor can a tree match a chain: the accepted path's keys hold the same values at
different slots, so the sum runs in a different order.

Both paths commit the target's own predictions by construction, so both are
greedy-correct and the difference is arithmetic. What separates a sound arm
from a corrupted one is the output's **structure**: the share of its two-token
contexts that occur in the prompt. The two arms already known to be broken
score 0.010 and 0.027; every structurally sound arm scores 0.281 to 0.370 --
an order of magnitude apart, with nothing in between.

By that measure the tree on `shared_fia` at 0.281 is sound, and its divergence
from the chain at token 4 is arithmetic rather than a fault.

## What the numbers say about the target

The best verified whole-machine TPOT is now **14.743 ms**, against the 22.135
the `sfa` chain reads and the 9.3794 the documents cite without a record.

The tree is sound but does not pay here: 4.263 accepted against the width-8
chain's 3.693 is 1.15x, while its round is 72.7 ms against 54.4 -- 1.34x. The
round, not the drafter, is what stands between this and 2.x: TPOT 2.9 at the
tree's acceptance wants a round of 12.4 ms, and the width-8 chain's round is
54.4.
