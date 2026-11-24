# The round, re-derived from this session's own measurements: 43.32 of 50.73 ms

2026-10-23. The previous entry showed the fixed term's 28.8 us a kernel reads
the host and the in-stream floor is 13.931, which left 23.6 ms a round with no
owner. This re-derives the round from items measured on one die today rather
than from a kernel count.

| item | us a layer | ms a round | scope |
| --- | --- | --- | --- |
| MoE gate_up, 68 active experts | 111.90 | 8.73 | measured, 1.08 TB/s |
| MoE down, 68 active experts | 99.17 | 7.74 | measured, 0.671 TB/s |
| attention, gather feeding dense FIA at width 12 | 139.36 | 10.87 | upper bound |
| TP collective, 156-call chain over 78 layers | 65.71 | 5.12 | measured, bf16 |
| routing, about ten kernels at the measured floor | 139.31 | 10.87 | floor x count |
| sum | 555.45 | **43.32** | |

Against a measured round of 50.73 ms: **7.41 ms unattributed, 14.6 percent**,
where the kernel-count view explained 27.17 of 50.73 and left 46 percent.

## The lower bound this sets, and why it is the hopeful reading

MoE is bytes at the roofline and the collective is rendezvous, so

    MoE + collective = 21.59 ms a round

is the floor if attention and routing were driven to zero. What that allows:

| accepted a round | TPOT at that floor |
| --- | --- |
| 5.441, the recorded engine figure | 3.97 ms |
| 6.966, the tree at budget 32 on the engine's own streams | **3.10** |
| 7.447, the tree at budget 64 | **2.90** |

So 2.x does not require the fixed term to fall fourfold. It requires removing
attention and routing -- 21.7 ms between them -- and an accepted count above 7.
Both have a measured route this session: the shared read makes attention flat
in width (slope -0.074 ms a row against the per-row path's 0.356), and the tree
reaches 6.966 at budget 32 on the engine's own output.

It also explains why the tree's optimum budget is 16 to 32 rather than 64: the
expert union grows with the width at 2.29 experts a row and each one is a
shard read, so MoE is itself the row cost that the budget sweep found.

## Scope

Every term is one of this session's own single-die measurements except the
50.73 ms round, which is the project's recorded p50 over 188 rounds on sixteen
ranks. Attention is the synthetic `gather_fia` level, an upper bound on the
fused sparse kernel, and routing is a kernel count times the measured floor
rather than a timing of the stage. The 7.41 ms residue is therefore the budget
for both of those approximations, not a new finding.
