# The boundary-merging sweep is finished, at 20.3 ms a round

2026-10-21, a3-84, one die, TP16, twelve tokens. Every adjacent-operator
boundary in a layer, priced and either taken or closed.

| merge | measured | a round |
| --- | --- | --- |
| routed experts, four launches into one | 181.172 -> 72.463 us | **8.48 ms** |
| TP reduction by one-sided push | 63.5 x 2 -> 22.3 us | **6.50 ms** |
| four projections off the hidden states | 68.724 -> 37.238 us | **2.46 ms** |
| residual add with the norm | 37.913 -> 19.189 us | **1.46 ms** |
| grouped matmul to CANN's own | 104.929 -> 66.813 us | **1.40 ms** |
| | | **20.30 ms** |

Taking the round from 55.424 to 35.1, and with the adaptive tree's 9.288
accepted over 13.1 nodes, a TPOT near 4.9 ms.

## What was closed rather than taken

- The norm with its quantisation. `npu_add_rms_norm_quant` is 210.996 us
  against 43.532 separate -- 4.8 times slower, -13.06 ms a round.
- `o_proj` with its residual add. The boundary is 6.663 us, 0.52 ms a round,
  and `torch.addmm` is 2.668 us slower than the two calls while differing by
  7.246e-03 relative against the fused projections' 4.058e-04.
- The attention chain generally. `mla_preprocess_only_decode` already covers
  the projections, both norms, the rope and the cache write; sparse attention
  is the vendor's core; nothing between them is both large and collectable.
- Cross-layer anything. Nesting eight layers in one launch costs 6.15x one and
  the per-layer work is 77 percent of it.

## What this means for the target

The mechanism that produced every entry in the table -- merge adjacent
operators, pay one boundary instead of several -- is now exhausted at 20.3 ms.
The round's remaining 35.1 ms is device time inside operators the vendor has
tuned, at a shape where they are 12 to 19 us each regardless of size.

So the last factor of two cannot come from this mechanism. It has to come from
doing less per round: fewer layers evaluated, or more tokens accepted for the
same evaluation. The second is the drafter, already at 9.288 accepted and
measured to 14.5 with a fixed budget. The first is untouched -- and for a model
that copies its input, a copied token may not need all 78 layers.
