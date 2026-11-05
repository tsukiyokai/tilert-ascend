# Choosing the vehicle for a fused kernel, and a retraction of my own benchmark

Four independent investigations on `a3-84`, each required to compile or run
something and to cite `file:line`. The three candidates were judged on
whether they can express the three things a fused decode kernel needs: a
cube/vector handoff without a GM round trip, a collective in the kernel's own
epilogue, and a FRACTAL_NZ weight read at the vendor's rate.

## The verdict: hand-written AscendC

Only AscendC expresses all three today, and a probe already compiles.

| capability | AscendC | TileLang 0.1.4 | PTO-ISA |
| --- | --- | --- | --- |
| cube/vector handoff | yes, `CrossCoreSetFlag<2, PIPE_MTE3>` / `CrossCoreWaitFlag` | yes, on chip, 12/12 verified | **no on-chip path exists on A3** |
| device-side collective | yes, `Hccl<HCCL_SERVER_TYPE_AICPU>::ReduceScatter` | one-sided put/get only, no reduce | yes, `TREDUCE<CollEngine::AIV>`, arbitrary rank subsets |
| FRACTAL_NZ weight read | yes, `MatmulType<GM, CubeFormat::NZ, …>` | **no**, cannot reach `gemm_v0`'s L1 layout | yes, explicit 5-D `GlobalTensor` + `Layout::NZ` |

The AscendC probe `fused_norm_quant_proj_rs.cpp` does all three in one
`KERNEL_TYPE_MIX_AIC_1_2` kernel — AIV norm/quant, `CrossCoreSetFlag`, then
`Matmul<AQuant, BQuantNz, CAcc, …>` with an in-flight
`hccl_.ReduceScatter<false>(…)` — and compiles dual-pass (`dav-c220-cube` and
`-vec`, exit 0) with a real `SET_CROSS_CORE` instruction in `objdump`. The
deployment path has a precedent already installed on this machine:
`dispatch_ffn_combine_w4_a8`, a W4A8 fused collective custom operator in the
vendor package. The one number still missing is whether a hand-written
`Matmul<>` reproduces the vendor's 1117.3 GB/s.

PTO-ISA is eliminated on a physical fact, not a preference: `TMov`'s
`static_assert` (`pto/npu/a2a3/TMov.hpp:165-169`) admits `Mat→Left/Right/
Bias/Scaling`, `Vec→Vec` and `Acc→Mat` — there is no `Acc→Vec` move, so a
cube result reaches vector only through GM. PTO models that hop as a
`TPipe<DIR_C2V>` GM ring FIFO, which is the best A3 allows but is not on
chip. Two more marks against it: `TMatmul` promotes `m == 1` to 16 to avoid
A3's gemv mode (`TMatmul.hpp:68-73`), so decode's four rows issue as sixteen
and use a quarter of the cube; and its `costmodel` is unusable for us —
bf16 and int8 matmul both return `ok=0`, the transfer model predicts
372 827 µs for a 24 MiB load (67.5 MB/s against a measured 1190.77 GB/s),
and its bandwidth constants are per-core (`kGmToL1PeakGiBs=28.61`, so 24
cubes cap at ~687 GiB/s). The install also ships no runnable example: the
first kernel took eleven compile iterations, and an undocumented prologue
(`set_mask_norm(); set_vector_mask(-1,-1); set_atomic_none();`) is mandatory
or the kernel dies with 507035 `vector core exception`. A bit-exact vector
kernel does now run on device 6.

## I retract my TileLang measurement

I reported the first TileLang kernel at 0.18× the vendor, 190–198 GB/s, flat
across twelve tilings, and blamed an ND tile copy. The flatness was the
clue I misread: it was the Cython adapter's ~130 µs per-call host dispatch
floor, so I was timing dispatch, not the kernel. The same kernel measured
three ways: eager 133.45 µs, one `aclgraph` launch 34.49 µs, twenty launches
20.62 µs each, vendor in the same process 15.91 µs. Device-side the kernel
runs at **1141–1203 GB/s, 96–101% of the die's measured peak and 0.98–1.08×
the vendor**. TileLang is not slow; its real gaps are the NZ layout into
`gemm_v0` and the absence of a reduce primitive.

It also does the thing PTO cannot: both legs of the cube/vector handoff
inside one `T.Kernel` with no GM, `T.copy(a_ub, a_l1, tmp=a_ub_nz)` and
`T.copy(accumulator, c_ub)`, twelve cases across M ∈ {16,128}, K ∈ {128,512},
fp16 and bf16, plain and transposed — all correct.

## The adversarial audit reorders the whole programme

The audit was told to attack the premise, and it did. Three corrections.

**The largest recoverable block is a fixed cost I had not isolated.**
`GroupedMatmulSwigluQuantV2` costs 96 µs per call independently of M, the
number of active experts, `group_list` length, K, N and L2 residency —
measured, not inferred. Across the round that is 10–12 ms. This is the same
shape as the clean GMM result (62–63 µs of prologue per call) and it is
larger than either item I had ranked first.

**The collective is 11.29 ms per round and overlaps nothing.** 157
`aiv_all_reduce_float` groups are 9.974 ms, the entry reduction adds
4.356 ms, the `Cast` wrapping them 1.320 ms; overlap with the compute stream
measures 0.000000 ms. Cube and vector both sit on stream 81 and are strictly
serial, with VEC busy 3.834 ms stacked behind CUBE busy 25.981 ms.

**And the arithmetic that bounds the whole programme**: 157 reductions ×
63.5 µs = 9.97 ms already exceeds a 2 ms/token round budget of 6.87 ms. So
the target is unreachable by fusing kernels alone — the *number* of
collectives has to fall. That makes EP (which deletes 78 MoE reductions) and
MTP replication (which deletes 9 draft reductions) structural prerequisites
rather than optimisations.

Two smaller items it pinned: the on-chip handoff is worth less than I
claimed — CUBE→VEC 326 transitions per round at p50 1.00 µs (0.344 ms
total), VEC→CUBE 326 at p50 1.34 µs (0.443 ms), so ceiling ~3.3 ms net, not
the dominant term. And 8.25 ms per round is host serialisation in the eager
MTP tail (`aclrtSynchronizeStream` 24.255 ms over 15 calls,
`broadcast_object_list`, `tolist`) — which no kernel fusion touches, and
which is exactly what the device-resident tokens attack.

It also reproduced the production weight layout single-process, which
validates the retraction recorded earlier: `DT_INT4` FRACTAL_NZ
`[256, 4, 384, 16, 64]` via `int8 → npu_format_cast(w, 29) → view(int32)`,
and **`torch_npu.npu.config.allow_internal_format = True` is required or the
cast silently falls back to ND** — the exact artefact my ND benchmark
measured.
