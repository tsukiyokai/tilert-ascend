# One norm fusion wins, the other loses by a factor of five

2026-10-21, a3-84, one die, twelve tokens, hidden 6144.

    add then norm, separate      37.913 us
    npu_add_rms_norm             19.189 us   1.46 ms a round
    norm then quant, separate    43.532 us
    npu_add_rms_norm_quant      210.996 us   -13.06 ms a round

## This closes an open question the wrong way round

The cast census recorded that `norm_quant_fusion_pass` is enabled and fires on
a small minority of the pairs it exists for -- `AddRmsNormBias` at 3.3 calls a
round against `RmsNorm`'s 16.6 and `DynamicQuant`'s 10 -- and asked why it
matches so rarely, treating the rarity as a defect to fix.

At the decode shape the fused operator is 4.8 times slower than the two
separate ones. So the pass missing is not a defect; making it match more often
would cost about 13 ms a round. That line of work is closed.

The 210.996 us figure may reflect a slow branch taken for the scale and offset
shapes passed here rather than the operator's best case. It does not matter
for the decision: there is no version of this that yields 13 ms, and the
separate path is already cheap.

## What is worth taking

`npu_add_rms_norm` for the residual-plus-norm pair, twice a layer, 1.46 ms a
round measured at one site. The engine's own compilation pass may already
place some of these; the cast census counted `AddRmsNormBias` at 3.3 calls a
round out of a possible 156, so most sites are unfused.
