# The permutation pair costs half what was recorded, at the decode shape

2026-10-21, a3-84, one die. Twelve tokens, top eight, 256 experts -- 96 sorted
rows of 6144, which is the shape a width-twelve round actually presents.

    vendor init_routing       22.045 us
    vendor token_unpermute    31.181 us   -> pair 53.226 us, 4.15 ms a round
    torch index_select        24.184 us
    torch index_add          131.957 us   -> pair 156.142 us, 12.18 ms a round

## What this corrects

The lever was recorded as 7.1 ms a round, against "init_routing 45.772 +
finalize_routing 55.979" = 101.751 us. At the decode shape the pair is 53.226
us. The earlier figure was measured at some other shape and carried into a
decode budget, so the subtrahend was nearly double.

The ceiling for absorbing the permutation is therefore 4.15 ms a round, not
7.1, and only if the replacement is cheaper than the vendor. The plainest
replacement is 2.93x more expensive: `index_add_` alone costs 131.957 us
because it atomically accumulates 96 x 6144 bfloat16 elements, which is the
whole of what `npu_moe_token_unpermute` does in 31.181.

## Where that leaves the three levers

    MoE single kernel     -16.3 ms   measured, inverted
    routing absorption    <= 4.15 ms ceiling, current implementation negative
    one-sided reduce        6.5 ms   not yet rechecked

Two of the three are withdrawn or reduced. The one-sided reduce's correctness
rested on an exact value match at sixteen ranks -- a criterion that can fail --
so it stands until measured otherwise.
