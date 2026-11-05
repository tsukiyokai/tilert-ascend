# What a kernel launch costs, and what that does to my own attribution

2026-10-20, 187, one free die with another tenant on dies 0 to 7.

    blocks 16  repeats  2000   issue only 3.384 us   issue and drain 19.812 us
    blocks 16  repeats 20000   issue only 4.056 us   issue and drain 29.141 us

## The first reading was my harness, not the part

An empty kernel taking no argument and touching no memory measured 64.654 µs
a launch, with issue-only and issue-and-drain identical. That contradicts a
known fact -- mixbrick's fused arm is 10.2 µs a call including its launch, so a
launch cannot cost 64 µs -- and giving the kernel one argument and one store
dropped it to 3.384 µs. A kernel with neither apparently takes a different
path through the runtime than any real one does.

## What it overturns

The round's cost model is `45.88 ms + 0.814 ms a row`, and the roofline page
attributed the 36.2 ms that neither dominant kernel explains to "per-kernel
issue and synchronisation". The issue half of that is now measured: at 3.4 to
4.1 µs a launch, a layer's twenty-five operators over seventy-eight layers
come to 6.6 to 7.9 ms. So host issue accounts for at most a fifth of the
residual, and my attribution was wrong.

The drain column is where the money is, and it grows with the count -- 19.8 µs
at two thousand launches, 29.1 at twenty thousand -- which is the shape of a
queue backing up rather than of a fixed cost. The cost is not in sending a
kernel but in waiting for it: each kernel's device-side prologue and epilogue,
and the synchronisation between it and the next.

## Which is what mixbrick already measured

Fused 10.2 µs against staged 17.0 for the same arithmetic, one boundary fewer.
6.8 µs a boundary, and a layer has on the order of twenty-five: 13 ms over
seventy-eight layers. So the megakernel's value is in removing boundaries, not
in reducing launches, and two independent measurements now say so -- the launch
floor says issue is not the cost, and mixbrick says the boundary is.

## What is still unattributed

About 23 ms of the round is neither issue, nor the two dominant kernels, nor
the boundaries at mixbrick's rate. The TP reductions are a candidate at 2 a
layer and 63.5 µs in-graph, which is 9.9 ms; the routing -- `npu_moe_init_routing`
and the combine, plus the topk over 256 experts -- is unmeasured. Measuring
those is the next step, and it needs one die rather than a machine.
