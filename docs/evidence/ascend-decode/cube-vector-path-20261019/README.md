# On this part a cube result reaches vector only through GM

Three sources agree, and they settle the premise the whole-layer kernel rests
on. The question matters because GM is an address space rather than a storage
level — a handoff through GM addresses that L2 serves costs an order of
magnitude less than one that reaches HBM — so "must it go through GM" and
"must it go to HBM" are different questions. The answer here is the stronger
one: there is no path that avoids GM at all.

## The vendor's own page says it is an A5 addition

The department wiki's *1.Cube-Vector 直通*
(`wiki.huawei.com/domains/199799/wiki/426428/WIKI2026092012978364`, updated
2026-09-20) opens by scoping itself: 直通 is "A5 上新增的矩阵计算结果的核内传递
能力", a Fixpipe move along an `L0C2UB` path that hands the result to the
vector unit "无需先写入全局内存（GM）". Its own comparison table gives the
traditional path's intermediate traffic as "写 GM + 读 GM", and its worked
example is a matmul followed by a GLU — exactly the epilogue shape a fused
layer needs.

## The chip difference table quantifies it

*昇腾系列芯片差异对比*
(`wiki.huawei.com/domains/199574/wiki/199574/WIKI202502145963626`, updated
2025-06-24) carries a per-core datapath width table. The row for
`L0C (FIXPIPE) → UB` reads: David-121 256, David V120/V100 128, **1971 `/`** —
a slash, meaning the path is absent.

The same document's specification table identifies that column against
hardware: 1971 has 64 GB of memory, 192 MB of L2 and 1.8 T/s of bandwidth. Our
part, `Ascend910_9392`, reports `l2_size` 192 MiB and 64 GiB per die and
measures a marginal read rate of about 1.54 TB/s. Three for three.

The same row's notes add that the quantisation feature is planned for removal
from Fixpipe with dequantisation moving to the vector unit, "（Fixpipe特性在
L0C2UB通路同样适用）" — which is why the path appears throughout the headers
without being available here.

## And the local evidence confirms it exists one generation over

`~/note/proj/ai-infra/vis/refs/davidv100-core-path-evidence.md` records from
the other side: `FIXP: Support L0C to L1/UB/OUT`, and
`AIC.L0C → AIV.UBuffer` one-way. So the path is a hardware fact in the David
generation and absent in 1971.

## Why the conclusion is stronger than "it goes through GM"

The 1971 column also gives `L0C (FIXPIPE) → L1` as 128, so a cube result has
two destinations: OUT, which is GM, or L1. But that column's
`L1 (MTE1) → UB` is a slash as well. So there is no two-hop route either, and
nothing a cube produces can reach the vector unit without landing in GM.

## What the CANN tree says, and why it is not the judge

`asc/impl/tensor_api/atom/cube/copy_l0c2ub.h` exists in CANN 9.2.0 and carries
no architecture gate; neither does `arch/cube/l0c_to_ub/copy.h`, whose
`static_assert`s only check that the destination is UB and the source L0C. A
probe using the legacy `DataCopy` from a `TBuf<CO1>` to a `TBuf<VECCALC>`
compiles for both `dav-c220-cube` and `dav-c220-vec` at exit 0.

That is not evidence of the path. It is the shape this project has repeatedly
been caught by — compiling is not running — and the two passes both accepting
a tensor in the other's local memory says the front end did not check. The
architecture tables above are what settle it.

`cube_vector_combine=split` is the same string in `Ascend910_9391.ini` and
`Ascend950DT_950x.ini`, so that flag does not discriminate either.

## What it means for the layer kernel

AMD's `pure_mla_moe_layer_kernel` puts MLA and MoE in one kernel, which
requires the cube's result to reach the vector unit on chip. Copying that
shape here pays back in full the GM round trip their design exists to avoid —
roughly 637 handoffs a round. The mechanism their central design depends on is
absent on this part, so the right unit here is not theirs.

This also says A5 deserves its own evaluation rather than an inherited
conclusion: there `L0C → UB` is 256 wide and the premise holds.
