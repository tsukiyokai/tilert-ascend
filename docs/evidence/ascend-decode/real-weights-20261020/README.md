# The kernel on the checkpoint's own weights

2026-10-20, a3-84, one die.

    read 144 of 144 tensors
    w13 (16, 128, 6144) torch.int8  w2 (16, 3072, 128)
    gu_scale (16, 256, 1)  gu_bias (16, 256, 1)  down_scale (16, 6144, 1)
    out (16, 6144) torch.bfloat16 finite True absmax 0.00036
      nonzero 98141/98304

Sixteen of layer three's experts, read out of the shards, sharded to one
rank's share, repacked the way `install` does, and run. Every shape matches
what `replay_cann_gmm1.py` declares the engine holds -- `w13_weight
(256, 128, 6144)`, `w2_weight (256, 3072, 128)`, `w13_weight_scale
(256, 256, 1)`, `w13_scale_bias (256, 256, 1)`, `w2_weight_scale
(256, 6144, 1)` -- for the sixteen experts this run covers.

## The sharding, which I had backwards

Each rank does not hold a sixteenth of the experts. It holds **all 256** and a
sixteenth of each expert's intermediate axis. The checkpoint's per-expert
`gate_proj.weight` is `(1024, 6144)`, packed along the output axis, so that
projection's N is 2048 -- and the engine's `w13_weight` is `(256, 128, 6144)`,
whose 128 is two projections' 64 bytes each. 2048 / 16 = 128. The intermediate
is 2048 and a rank computes 128 of it.

So `group_counts` being `(256,)` with eight ones is not "eight of the sixteen
experts on this rank"; it is eight of all 256, every one of which this rank
holds a slice of. My earlier reading, 256 experts over 16 ranks, was a guess
that the arithmetic happened to permit.

`down_proj` follows from it: its K is the intermediate, so a rank's share is
2048/16 = 128 outputs, not 1024/16 = 64 bytes. Slicing it like the gate axis
is wrong by a factor of two, and the only thing that catches it is that the
engine's declared shape says 128.

## What this establishes and what it leaves

Establishes: the weight path works on real bytes. The repack preserves codes
(`repack_test.py`, nothing unequal at three shapes), the shapes agree with the
engine's, and the kernel produces finite output with no structural zeros.

Leaves: the output's magnitude is dominated by the real scales, about 1e-4, so
this run is a shape-and-plumbing check rather than a numerics one. The
numerics are settled separately against the vendor operator under a constant
weight, in [vendor-numerics-20261020](../vendor-numerics-20261020/README.md).
