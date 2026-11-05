# The routing costs 15.84 ms a round, and only 1.28 of it is arithmetic

2026-10-20, 187, one free die. Twelve rows, 256 experts, top eight -- the shape
decode issues at the measured optimum width.

    gate matmul        17.003 us   x75 layers =  1.28 ms
    topk over 256      32.445 us   x75 layers =  2.43 ms
    weight normalise   38.869 us   x75 layers =  2.92 ms
    dynamic quant      21.110 us   x75 layers =  1.58 ms
    init routing       45.772 us   x75 layers =  3.43 ms
    finalize routing   55.979 us   x75 layers =  4.20 ms
    together          211.179 us   x75 layers = 15.84 ms

29% of a 55.4 ms round, and one of the six is a matmul. The other 14.56 ms is
spent on operators whose arithmetic is negligible: a topk over 256 values a
row, a sigmoid and a division, a quantisation of twelve rows, and two
permutations of ninety-six rows. The bytes are small and the cost is the
operators themselves.

## Corrected: the gating is one operator, not three

The engine does not compute the gating the way the table above measures it.
`router_factory.py` selects `npu_moe_gating_top_k` and `device_op.py:1228`
calls it, and that one operator does the sigmoid, the group selection, the top
k, the renormalisation and the routed scaling factor together:

    fused gating      20.798 us   x75 =  1.56 ms
    separate pieces   68.207 us   x75 =  5.12 ms

So the 32.445 and 38.869 µs rows above -- a torch topk and a hand-written
sigmoid and division -- over-count by 3.56 ms. The routing costs about 12.28
ms a round, not 15.84, and the part of it I had not yet touched is mostly not
there: the engine already fused it.

What survives is the part I did absorb. `init_routing` at 3.43 ms and
`finalize_routing` at 4.20 remain the largest single item in the routing, and
the absorption's 7.1 ms stands.

## The round now accounts

    attention            7.0 ms   89.4 us a layer over 78
    MoE grouped matmuls 12.3 ms   157.2 us a layer over 78
    TP reductions        9.9 ms   63.5 us twice a layer over 78
    routing             12.1 ms   corrected for the fused gating
    host issue           6.6 ms   3.384 us a launch, 25 a layer
    -------------------------------------------------------
    together            47.9 ms   against 55.424 measured

About seven and a half milliseconds unattributed, where the roofline page had
36.2 ms it could not place.

## Which reorders the work

I had the MoE grouped matmuls as the main target, and my INT4 kernel does take
157.2 us a layer to 37.2 -- nine milliseconds. The routing is larger at 15.84
and easier: nothing needs a matmul rewritten. The two permutations alone are
7.63 ms of pure data movement, and they are what the expert kernel should
absorb -- the gather `init_routing` materialises and the weighted unpermuted
sum `finalize_routing` performs. That is the absorption I identified when the
norm fusion was reverted: downstream of the routing, not upstream.

## On the method

Three attributions of mine have now been overturned by measurement rather
than by argument: that the residual was host issue (it is 6.6 ms of 36), that
the shared fold was affordable from the cube's idleness (the weight bandwidth
binds), and that the MoE matmuls were the main target (the routing is larger).
Each cost one experiment to settle and each was cheap. The pattern is that I
reason confidently about where time goes and am wrong about half the time, so
the rate-limiting step is measurement, not analysis.
