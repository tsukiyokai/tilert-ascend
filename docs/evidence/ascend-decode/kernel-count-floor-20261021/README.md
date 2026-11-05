# The round is kernel count times 28.8 us

2026-10-21, a3-84, one die, one process, twelve tokens.

    q_a_proj   6144x2048      17.761 us
    q_b_proj   2048x16384     32.501 us
    kv_a_proj  6144x576       18.391 us
    o_proj     16384x6144    178.148 us   (unsharded; TP16 splits it)
    router     6144x256       18.730 us
    shared_up  6144x12288     52.637 us
    rms_norm                  33.252 us
    residual add              12.657 us
    one scalar multiply on 12x1   28.824 us

The last line is the finding. A single elementwise multiply over twelve
elements costs 28.824 us. That is the per-kernel floor at this batch, and it
is 8.5x the 3.384 us this project had recorded from an empty kernel.

## This closes the round's accounting

A layer issues about 25 kernels. 25 x 28.8 us is 720 us, and the round is
55.424 ms over 78 layers -- 710 us a layer. The round is the kernel count
times the floor; the arithmetic is noise. It also explains why every operator
measured today sat near 30 to 70 us regardless of its shape, why sixteen times
the rows moved the grouped matmul 2.8 percent, and why the round model's fixed
term is 45.88 ms.

## What it implies for the target

    TPOT = kernels a layer x 28.8 us x 78 / tokens accepted

At 25 kernels and 5.44 accepted that is 10.3 ms, which is the measured 9.38
within the error of the kernel count. To reach 2.x ms at the same acceptance
the count has to fall to about five a layer.

So the megakernel is necessary, and for a reason this project had wrong twice
over: not to beat the vendor's arithmetic, and not to fuse across layers --
cross-layer nesting is capped at 23 percent -- but to collapse 25 launches a
layer into a handful. Within a layer, not across them.

## Loose end

`o_proj` at 178.148 us is bandwidth on a 200 MB bfloat16 weight and is
unsharded here; the engine splits it sixteen ways. It needs remeasuring at the
real shard before the per-layer sum is trusted to better than ten percent.
