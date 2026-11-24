# 58 percent of the MoE cost is the call, not the bytes -- and 2.x reopens

2026-10-23. The previous entry concluded this configuration bottoms at 3.5 to
3.68 ms because MoE is "bytes at the roofline". That reading was wrong, and the
data refuting it is the same expert sweep it was drawn from.

Fitting the 8-to-256 expert sweep measured today:

    gate_up = 59.37 us + 0.709 us an expert
    down    = 63.31 us + 0.578 us an expert

At the width-16 union of 68 experts that is 16.40 ms a round against a measured
17.33, so the fit holds to 5.4 percent. The decomposition is what matters:

| part | ms a round | kind |
| --- | --- | --- |
| fixed, both calls, 78 layers | **9.57** | dispatch and tiling |
| bytes, 68 experts, both calls | 6.83 | roofline, 1.08 and 0.671 TB/s |
| sum | 16.40 | measured 17.33 |

**Fifty-eight percent of the MoE cost is the call.** The 59 us fixed charge is
4.3 kernel floors at today's measured 13.931 us, which is what a grouped
matmul's dispatch and tiling cost -- the kind of thing fusion removes, not the
kind that is bytes.

## What that does to the floor

With the fixed part gone, MoE plus the collective is 6.83 + 5.125 = 11.96 ms a
round:

| accepted a round | TPOT |
| --- | --- |
| 6.966, the tree at budget 32 on the engine's own streams | **1.72 ms** |
| 5.989, the tree at budget 16 | **2.00** |
| 5.441, the recorded engine figure | **2.20** |

All three land in 2.x. So the floor is not 3.68 ms and the previous entry's
conclusion is retracted: it counted a call's dispatch as a physical byte.

## And it names the arm to run

`--routed-shared-kernel` interleaves the routed INT4 and shared W8A8 experts
into one call, which is exactly the 9.57 ms this entry isolates. It is one rung
of the staged megakernel ladder, it has passed byte-exact gates, and it has
never been measured as TPOT. That is now the measurement with the most behind
it, not the pass count.

## The fixed charge is per call and already whole at one expert

The entry above assumed the 9.57 ms could go to zero. Measured at one, two,
four and eight experts, with the bank count pinned at 4 because the probe's
own rotation asks for more memory than a tenant leaves:

| experts | gate_up | down |
| --- | --- | --- |
| 1 | 66.85 us | 60.74 |
| 2 | 66.10 | 60.28 |
| 4 | 73.52 | 62.28 |
| 8 | 61.71 | 70.60 |

Flat from one expert to eight, so the ~60 us is a call's own tiling and setup,
whole at a single group. It is not launch latency either -- every figure here
is the mean of twenty back-to-back calls.

So fusing `gate_up` and `down` into one call removes **one** of the two
charges, not both: 4.63 ms a round of the 9.57, leaving 4.94.

    floor = bytes 6.83 + collective 5.125 + remaining fixed 4.94 = 16.90 ms

| accepted a round | TPOT |
| --- | --- |
| 6.966, tree at budget 32 | **2.43 ms** |
| 5.989, tree at budget 16 | 2.82 |
| 5.441, recorded | 3.11 |

Still 2.x at the tree's measured acceptance, but the entry above overstated the
removable part twofold, and that is corrected here rather than carried.
