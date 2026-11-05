# docs/reference — collected source material

Verbatim source material for the GLM-5.2-W4A8C8 / Ascend A3 ultra-low-latency decode work.
Nothing here is original analysis; every item is quoted as published, under its URL, author
and date, with a one- or two-line `## Relevance` note tying it to our measured numbers. Each
file ends with `## Gaps`.

Our measured baseline, for reading the Relevance notes:
decode round 53.05 ms unprofiled; best TPOT 10.8528 ms (was 11.6198); prompt-lookup acceptance ~7.1
tokens/round; ~3008 kernels/round; per-call launch floors AI_VECTOR_CORE 0.70 us,
AI_CORE 2.30 us, MIX_AIV 12.04 us, MIX_AIC 13.58 us (505 MIX_AIC calls/round, 6.5/layer),
giving an 11.5 ms launch floor with 82% on MIX; TP reduce 5.35 ms and 100% serial.
First fused brick measured 2026-10-20: 12.307 us at 8 rows / 4 blocks, against 15.093 for the same work as two
kernels and 14.801 for the vendor pair (`QuantBatchMatmulV3` 11.661 + `DequantSwigluQuant` 3.140) — 17% faster
than the two vendor operators as one kernel, exact against the two-kernel arm, 24 consecutive clean runs.
Per family, ms spent vs ms of bytes it must move: SparseFlashAttention 6.973 / 1.382;
GroupedMatmulSwigluQuantV2 6.793 / 2.272; GroupedMatmul 5.596 / 1.212;
QuantBatchMatmulV3 3.810 / 0.780; MlaPrologV3 2.950 / 2.745; Cast 2.300 / 0.076;
ForeachAddList 1.622 / 0.018.

## Files

| File | Covers |
| --- | --- |
| `megakernel-persistent-decode.md` | Hazy Research Llama-1B megakernel (full text), MPK / Mirage Persistent Kernel, FlashMoE/FlashDMoE, **CANN SuperKernel device-side source**, ThunderMLA, Event Tensor / ETC, plus a cross-system comparison table. **§9-§72 are the in-kernel cross-rank collective**: `cann/shmem`'s `aclshmem_ptr` and `aclshmemx_mte_atomic_add`, MegaMoE's epoch-tagged `SynchronizeRanks()`, the `sqe_type`/`errType` decode from `cann/runtime`, and the A3 `SyncAll` flagId budget |
| `ascend-fusion-mechanics.md` | AscendC kernel programming, MIX core semantics, cross-core sync APIs, FFTS+, buffer/tiling, SuperKernel and graph-mode launch fusion — public CANN docs and GitCode source. **§31 is the A3 cube→vector GM visibility rule** (`DataCacheCleanAndInvalid` + `DataSyncBarrier<DDR>` on both sides; `CrossCoreSetFlag` alone is not a fence). **§42-§48 are the `int4b_t` and intra-core-sync block**: §42 sub-byte addressing from the shipped headers (and the hard negative on `SetGlobalBuffer`'s length), §43 the authoritative `PipeBarrier` vs `SetFlag`/`WaitFlag` rule with the per-core-half `HardEvent` split, §44 the two shipped vendor int4 kernels and the 950PR unzip tax, §45 the public found-and-fixed W4A4 erratum, §46 `GetQueEvt` as the barrier→event conversion table, §47 the `Cast`/`DataCopy` guard asymmetry and `REGIST_CUBE_OBJ`'s half-partition, §48 the ISASI constraint list and API allowlist. **§49-§57 are the performance block** — Fixpipe dtypes, kernel types, the 512-byte same-address rule, split-K, IBShare, constant tiling, the 256-byte inner-axis rule, the GM-traffic formula, and working `AIC_ONLY`/`MIX_AIC_1_0` code. **§58 is the closed AIV sync matrix** — all sixteen legal intra-core combinations on 2201 with `PipeBarrier` on the diagonal only, the fact that `PipeBarrier` cannot name `PIPE_S` at all, that `--cce-auto-sync` is a build-system default a bare `bisheng` line does not get, that `ReduceMax`'s `sharedTmpBuffer` may alias its source on A3, and that a UB→UB `DataCopy` is PIPE_V. **§59 is `RoundMode`** — a `Cast`'s round mode names the destination's representable set, not the integers (`float→half` with `CAST_RINT` gives 0.5 where `float→float` gives 0.0, from the vendor's own two worked examples), plus the sub-byte saturation table and the 1-in-32 INT4 escape that follows when a nibble split is fed a non-integer |
| `ascend-launch-overhead-and-sync.md` | **Internal Wiki only — do not publish.** Measured MIX head-overhead decomposition, the CrossCoreSetFlag counter limit and MlaPrologV3's real CV pipeline, the 0–7 event-id budget, A3 buffer capacities, `blade/runtime` (an existing Ascend megakernel runtime with `aclrtLaunchMega`), an internal plan for an Ascend MPK, heterogeneous workers, Async-MoE scoreboard sync |
| `fused-moe-expert.md` | gate+up → SwiGLU → quant → down fusion, grouped GEMM at tiny M, W4A8 / antiquant layouts, weight streaming. **Part 8 is grouped matmul over 256 experts inside a hand-written kernel** — `group_list_type`'s three modes, per-expert weight addressing, the 128-group TensorList cap. **§B7 is arXiv 2605.13915 "Multi-Scale Dequant"** — one of four things CANN calls MSD, with the M-stacked single-GEMM form as its equation (16); read §B7.-1 first for the four-family taxonomy and the authoritative A8W4 expansion |
| `w4a8-msd-int4-mmad.md` | The A8W4 MSD decomposition as CANN actually ships it: the four MSD families and which kernels hold each, the documented nibble split and `weightAssistMatrix` / `bias` / `yOffset` correction term, and the `Mmad` both-operands-int4 wall |
| `mla-decode-fusion.md` | MlaPrologV3, SparseFlashAttention / DSA, FusedInferAttentionScore, FlashMLA, weight absorption. **§8-§21 are the SparseFlashAttention roofline gap** — `sparse_block_size` flipping `perfMode_` from `V_TEMPLATE` to `C_TEMPLATE` and deleting the `kvMergeGm_` round trip; the reference operator's 98-100 us/call against our 89.4; the A3 controlled A/B showing the gather penalty is **2.48x instruction count, 1.09x burst length**; no gather DMA on 2201; and the 16-outstanding-MTE2-request limit that makes this latency-bound |
| `ascend-decode-latency-baselines.md` | What batch-1 decode TPOT anyone has actually published on Ascend, so our 10.8528 ms has a reference frame — and the ten-commit host-side lever catalogue behind the best of them (vllm-ascend PR #19033, 910_93, c=1 TPOT 8.2→7.6 ms plain and 7.0→5.1 ms with MTP) |
| `low-latency-tp-reduce.md` | one-shot allreduce, symmetric memory, reduction fused into the producing kernel, MC2, LCCL, aclshmem. **Part 7 is the in-kernel collective without a host barrier** — `aclshmem_quiet`'s implementation, epoch/magic tagging, and the A3 AMO limits. **Part 8 is the vendor's MC2 tuning case** — 32.7% measured on MatmulAllReduce, and the three numeric reasons none of it reaches batch=1, plus the AllReduce cost fit with its 14.769 us constant term |
| `speculative-acceptance.md` | prompt lookup, SuffixDecoding, EAGLE/Medusa/tree drafts, MTP, acceptance-length measurements |
| `ascend-custom-op-packaging.md` | **Read off the machines, not published** — how a hand-written AscendC kernel reaches the graph. The vendor package file by file; `binList` / `simplifiedKey` / `staticKey` decoded with their generator sources; `opc` as the whole toolchain (**not** `msopgen`) with the exact invocation; what `op_proto` and `op_api` do and do not need rebuilding; the `ASCEND_CUSTOM_OPP_PATH` first-wins resolver confirmed from plog; and the torch-custom-op channel that needs no packaging at all. **§10 is seven superseded assumptions**, including that a `<<<>>>` kernel cannot reach the graph (it can, and we already did it once) and that a one-operator `.o` swap could carry a kernel that fuses two operators (it cannot) |
| `tilert-rocm-glm52-structure.md` | The TileRT ROCm GLM-5.2 binary, disassembled — kernel inventory and structure, for comparison against what we generate |
| `tilelang-ascend.md` | **Deprioritized** (hand-written AscendC was chosen instead). Kept for the AscendC primitives it documents: pipe flag protocol, real L1/UB address maps, Cube↔Vector handoff, and the tile-ai repo inventory |

## The findings that change the plan

0. **A working, public, hand-written Ascend megakernel already exists, and its grid barrier is
   copy-pasteable.** `huawei-csl/megagdn-pto` fuses seven stages into one `<<< >>>` launch using
   nothing more exotic than `__aicore__` functions calling each other, with
   `SyncAllMegaKernel<false>()` between stages; it is merged into the SGLang NPU ecosystem and
   gives ~15% faster Qwen3.5/3.6 prefill in vllm-ascend. The full A2/A3 and A5 barrier
   implementations are reproduced verbatim in `megakernel-persistent-decode.md` §8.4. Read that
   section first — it is the shortest path from here to compiling code.
1. **Two Ascend megakernel runtimes also exist.** CANN's `SuperKernel`
   (`gitcode.com/cann/graph-autofusion`) is open source, written in AscendC, and its device-side
   interpreter — GM task queues, `TYPE_FUNC`/`TYPE_SYNC`/`TYPE_PRELOAD` dispatch, ICache
   preload, GM notify/wait — is reproduced verbatim in `megakernel-persistent-decode.md` §4,
   with its fusion-boundary enums in §4b. A second, more ambitious one (`blade/runtime`, Host
   API `aclrtLaunchMega`, AICPU scheduler, per-core 128-slot HBM rings, MMIO doorbell) is
   documented internally in `ascend-launch-overhead-and-sync.md` §10 but was not found publicly.
   Deciding whether either is obtainable comes before writing a runtime.
2. **The 13.58 us MIX floor is itemized, and a third of it disappears with static shapes.**
   `ascend-launch-overhead-and-sync.md` §1: ~4 us irreducible hardware, 2.46 us kernel-args
   prefetch, 2.3 us TilingData copy, 1.3 us other — and the TilingData copy falls from 2.845 us
   to 0.038 us when the shape is const. Batch=1 specialization makes every shape const.
3. **The synchronization budget is the binding Ascend constraint.** `CrossCoreSetFlag` flagIds
   are **0-10** on A2/A3 (`ascend-fusion-mechanics.md` §4, §20 — note
   `ascend-launch-overhead-and-sync.md` §4.7 carries a conflicting "0-7" claim that is
   corrected in place; use 0-10). `Matmul` with N objects already consumes `[0, 2N-1]`, plus
   flagId 15 under SuperKernel 1:2, leaving about 2-10 hand-allocatable. Each counter saturates
   at 15 Sets, no two consecutive Sets of the same flagId are allowed, and exceeding it
   deadlocks non-deterministically (a real defect, DTS2025123130366). Three documented ways
   round it: GM counters with `dcci` polling (SuperKernel's `SkMemoryWaitFlag`), a dedicated AIV
   scoreboard tracking only the latest-completing producer, and power-of-two-decreasing expert
   grouping `{8,4,2,1,1}`.
3b. **Four A3-specific hazards to design around from day one.** On 2201 `CrossCoreWaitFlag`
   ignores its `pipe` argument and blocks *all* pipes, so every AIC→AIV handoff is a full-pipe
   barrier on the waiter. Mode 4 (AIC to a single AIV) is 3510-only, and mode 2 is documented
   to leak across MIX blocks at blockDim > 1 and deadlock the producer — with no mode-4 escape
   on A3. `__schedmode__`/batchmode is mandatory, not advisory: a vendor probe measured 3 hangs
   in 49 runs without it and 0 in 240 with, concentrated on small shapes, i.e. our regime. And
   **A3 has no L0C→UB path** (950PR added it), so cube→vector is
   `L0C → Fixpipe → GM → MTE2 → UB`; the measured penalty is small only because the write lands
   in L2 (`aiv_mte2_time` 320.543 → 57.145 us when fused, −82.2%), so the goal is "don't miss
   L2", not "don't touch GM". Conversely `VECOUT → A1 → A2` vector→cube *is* available, and
   `TPosition::CO1` keeps an accumulator in L0C across `Iterate` calls. All in
   `ascend-fusion-mechanics.md`.
3c. **A cross-core flag is a counter, not a fence — this is what breaks `mix_expert_kernel.cpp`.** Nothing in the
   `CrossCoreSetFlag` / `CrossCoreWaitFlag` documentation claims any memory-ordering or cache effect: set increments
   a counter, wait decrements it. On DAV_2201 three independent third-party kernels measured that a GM buffer written
   by one core is *not* reliably visible to another after the flag alone, with the producer's write still in the write
   buffer when the flag arrives — 「flag 已到、写回仍在途」, symptom 「AIV 读到 vTmp 全 0」, and 「加探针即掩盖」, a
   probe masks it. The fix is `DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>` plus
   `DataSyncBarrier<MemDsbT::DDR>()` — producer before the set, consumer after the wait, before the `DataCopy`.
   `PipeBarrier<PIPE_ALL>()` cannot substitute; it orders issue, not retirement. One measured A/B says the producer's
   *barrier* and the consumer's *invalidate* are the load-bearing halves (9/60 → 1/60 failures), and another says
   hardening only the suspected handoff is useless (19/2000 unchanged) while hardening every cross-core buffer removes
   the drift. The ordinary single-operator path gets a whole-DCache flush appended by the Bisheng compiler and another
   from SuperKernel after every sub-kernel; a 78-layer megakernel crosses no operator boundary and must place all of
   them itself. All verbatim in `ascend-fusion-mechanics.md` §31, with the official API pages for both intrinsics.

4. **Fusion is not unconditionally a win at batch=1, and two sources say so with numbers.**
   MegaMoE's own authors report the fused path is *worse* than Dispatch&Combine at small shape
   because pre-synchronization outweighs the overlap (§13.6), and measure ~50% compute inflation
   from AIC/AIV MTE contention when compute and comm overlap on one core. MonoMoE separately
   measured that fusing the TP all-reduce does not pay because it is terminal.
5. **Acceptance is already at the published ceiling; the round time has to come down.**
   `speculative-acceptance.md`: SuffixDecoding's best reported is 7.8 mean accepted tokens/step
   on SWE-Bench against PLD's 3.2, and our measured 7.1 is at that state of the art. vllm-ascend
   also caps `num_speculative_tokens + 1 <= 16` via `npu_fused_infer_attention_score`.

6. **A foreign-core `WaitFlag` is a silent no-op on dav-2201 while its `SetFlag` still fires.**
   `ascend-fusion-mechanics.md` §43.5: `WaitFlagImpl` wraps its `MTE2_V` / `V_MTE2` / `MTE3_V` / `V_MTE3`
   cases in `#ifndef SPLIT_CORE_CUBE`, so on the cube half of a `__mix__` kernel the wait falls through to a
   bare `default: break;` — nothing — while `SetFlagImpl` has no such guard and `SetFlagInternal` screens per
   core only for `__NPU_ARCH__ == 3510`. No `static_assert`, no log, no diagnostic. The operational rule:
   every `SetFlag` / `WaitFlag` / `FetchEventID` naming a vector event must sit inside `if ASCEND_IS_AIV` and
   every cube event inside `if ASCEND_IS_AIC`; an `__aicore__` helper shared by both halves must never name a
   `HardEvent` unguarded. `M_V`, `V_M`, `MTE1_V`, `V_MTE1` must never be named at all on 2201 — their Waits
   are compiled out on *both* halves. The legal set is 13 events on an AIV and 24 on an AIC, not the 35 in
   `HardEvent`. And `PipeBarrier` is the 单流水 primitive, 「阻塞相同流水」 — it is a different instrument from
   `SetFlag`/`WaitFlag`, not a stronger one, so `PIPE_ALL` is not defined to cover a cross-pipe hazard
   (§43.1-§43.2). `PipeBarrier<PIPE_S>()` 「会引发硬件错误」.
7. **`sizeof(int4b_t) == 1` but `SizeOfBits<int4b_t> == 4`, and that gap has already caused one
   found-and-fixed W4A4 defect on this part.** `ascend-fusion-mechanics.md` §42 and §45. Tensor APIs count
   nibbles (`operator[]` does `offset / INT4_TWO`, silently flooring an odd index and *not* adjusting
   `bufferSize_`); raw `__gm__ int4b_t*` arithmetic counts bytes. Within one API family the Level-2
   `DataCopy(gm, ub, count)` counts elements while `Nd2NzParams`'s `dValue`/`srcDValue` count bytes. The GM
   contract is low nibble = even element, row *r* at byte `r*N/2`; K and N must be multiples of 64; and
   `torch.int4` is *unpacked*, one byte per element, so anything sized or indexed from a torch view is 2× out.
   Separately, `SetGlobalBuffer`'s length argument is **inert on NPU** — nothing on the `DataCopy` path reads
   `bufferSize_` — and the Level-0 `DataCopy(GlobalTensor, LocalTensor, DataCopyParams)` **silently emits
   nothing** if the source `TPosition` maps to neither UB nor L1, because that `else` arm is a debug-only check.
8. **The A3 int4 Cube path does not exist on 950PR/950DT, and the sanctioned replacement costs the whole W4
   win.** `ascend-fusion-mechanics.md` §44.3: the vendor's own `matmul_s4` unzips int4x2 to int8 on the vector
   core and **writes the unpacked int8 back to GM** for the Cube to read, so the W4 bandwidth advantage is zero
   at the point of consumption — plus two vector `Cast` passes (int4→half→int8; there is no direct int4↔int8
   intrinsic), 4 bytes of UB per unpacked element held simultaneously, and it runs per kernel call. Any
   layer-fusion design banking on A3 int4 needs its weight path re-derived for 950, not retuned.

9. **A one-sided device-side all-reduce is already a first-party open-source primitive on this part, and the
   kernel type we were blocked on is `MIX_AIC_1_0`.** `megakernel-persistent-decode.md`: `gitcode.com/cann/shmem`
   (CANN SHMEM v1.7.0, 2026/10) gives `aclshmem_ptr(ptr, pe)`, after which a peer rank's HBM is an ordinary
   `__gm__` pointer written with a plain `DataCopyPad` on MTE3 — the vendor's MegaMoE example never calls a put
   in the data path. `aclshmemx_mte_atomic_add` is supported on Ascend910B/910C for
   int8/int16/half/bfloat16/int32/float, 「performed by MTE2 unit」; every other MTE atomic is 950-only. The
   rendezvous is MegaMoE's `SynchronizeRanks()`: a never-reset monotone generation spun on with `>=`, through
   `ReadGmBypassDCache`/`WriteGmBypassDCache`, with Cube excluded by `if constexpr (g_coreType == AIC) return;`.
   And `ascend-fusion-mechanics.md` §50: `KERNEL_TYPE_AIC_ONLY` **is** documented as supported on A3 and ships
   working in three repos; the cube-only-with-hard-sync type is **`KERNEL_TYPE_MIX_AIC_1_0`**
   (「MIX AIC:AIV 1:0（带有硬同步），算子执行时仅会启动AI Core上的Cube核」), and a forced MIX type on a pure-cube
   kernel additionally requires a **non-zero workspace**. With `<<<>>>` only `KERNEL_TASK_TYPE_DEFAULT` applies.
10. **GM requests are coalesced at 512 bytes and SERIALIZED when several cores hit one window — on A3, at
   priority 高.** `ascend-fusion-mechanics.md` §50.4: 「MTE2、MTE3、Scalar等单元访问Global Memory数据时，其地址
   请求会按照512字节粒度对齐后进行处理 … 多个请求会被串行处理」, and 「同地址访问的核数越多时，串行导致的性能
   劣化越严重」. The vendor's own scenario is 「每个核都会完整载入一遍输入矩阵」 — the decode-time weight
   broadcast, i.e. our case — and the fix is free and shipped as code: rotate each core's access order by
   `blockGroupStart + (mBlockIdx + blockIdx) % numBlocks`. Same section: **≥16 KB per single transfer** is where
   bandwidth saturates; the 512-byte *alignment* rule with its 70% penalty is **A2 only**; `SetL2CacheHint`
   matters because eviction must write back to GM before a new line enters, so an unhinted streaming read burns
   bandwidth evicting the weights you wanted resident. And §50.3 is the vendor advising the opposite of
   intuition for our regime: 「对于整体耗时在微秒级别且单核计算量耗时较少的算子，可以通过减少启动核数并增加单核
   计算量的方式来获得性能提升」 — fewer cores, more work each. Head overhead is itemized as 核启动 +
   核取址TLB MISS + 同地址访问冲突 + 变量资源初始化 and 「会随着使用的核数增加而增加」; automatic kernel-type
   detection 「无法区分AIC和AIV的配比，默认按照AIV:AIC为1:2」 and 「可能失效」.
11. **Half the `Cast` family is free on the cube's drain; int4 out of Fixpipe does not exist.**
   `ascend-fusion-mechanics.md` §49: A3 Fixpipe is `float → {int8,uint8,half,bfloat16,float}` and
   `int32 → {int8,uint8,half,int32}`, with `REQ8`/`VREQ8` the scalar and per-output-channel int32→int8 requant
   applied with no Vector involvement (scale is `uint64_t` per channel, in L1, 32-byte aligned). **No product's
   table has an int4 destination** and A3's `QuantMode_t` has no S4 mode — even though the A3 *architecture*
   chapter lists `S322S4` and `FP322S4` as hardware Fixpipe conversions. Hardware capability and API surface
   disagree; until a probe says otherwise, W4 output quant stays a Vector `Cast`.

12. **There are three kernel-type designs, not two, and the launch floor is 4.5x cheaper outside MIX.**
   `ascend-fusion-mechanics.md` §55: `KERNEL_TYPE_AIC_ONLY` is the **global default in Huawei's own
   `swiglu_gated_mlp` kernel** (whose parameters are `x`, `gate_up_weight`, `down_weight`, `y` — our expert
   chain), with twelve per-tiling-key overrides, nine `AIC_ONLY` and three `AIV_ONLY`, and **no MIX branch
   anywhere**. It also ships in `inclusionAI/AReno`'s fused-experts matmul, `xLLM-AI/xllm-ops`'s
   `index_group_matmul`, `sgl-project/sgl-kernel-npu`, five `huawei-csl/pto-kernels` files and `tilesim/vTriton`.
   The copyable cube-only-with-hard-sync template, from `XPU-Forces/mojo_opset`'s grouped GEMM, is four lines
   and does **not** use `__mix__`: `__global__ __aicore__`, then
   `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);` as the first statement, then
   `if ASCEND_IS_AIV { return; }`, then the `TPipe`. At our floors, one MIX_AIC call (13.58 us) replaced by one
   AIC_ONLY (2.30) plus one AIV_ONLY (0.70) is 3.00 against 13.58 — **4.5x on launch cost for the pair**. The
   price: `KERNEL_TASK_TYPE(key, value)` is 「仅支持在自定义算子工程中使用」, so a `<<<>>>` project gets only
   `KERNEL_TASK_TYPE_DEFAULT` — one type per binary, i.e. separate kernels rather than branches of one. Three
   designs: one MIX megakernel (single launch, 13.58 us each); a tiling-key-switched binary (single launch,
   cheap — needs a custom-operator project); split AIC/AIV kernels (works with `<<<>>>` today, more launches,
   each 4.5x cheaper).

13. **The integration path was never missing, and the vendor-package route cannot carry this kernel.**
    `ascend-custom-op-packaging.md` §9-§10. Two things were held and both are wrong. First, a `<<<>>>`
    kernel does reach the captured decode graph — not as a CANN operator, but as a torch custom op on
    `dispatch_key="PrivateUse1"` with a `fake_impl`, which is how vLLM-Ascend ships its own Triton
    kernels (`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope_vnorm.py:473-478`) and how
    **this project already shipped an AscendC kernel into production once**
    (`torch-allreduce-20261018`: `aclshmem_torch.so` + `libtp_push_kernel.so`, in-graph, 22.3 µs
    against HCCL's 63.5). The hook is already in the engine at `cann_model.py:257-260`, where four
    overrides live, installed after `load_model()` and before capture. Second, an `.o` swap inside the
    vendor package *cannot* carry the routed-expert kernel, because that kernel replaces
    `GroupedMatmulSwigluQuantV2` **and** `GroupedMatmul` and keeps the intermediate in UB, while a
    binary override preserves the operator's prototype by construction — so the graph still issues the
    second operator and the GM round trip between them survives. A per-operator override can only
    deliver a win internal to one operator, which is exactly what the four landed
    `patches/cann-gmm-swiglu-v2/` patches did (−1.468 ms a round). Fusing two operators means changing
    the caller. Also retired there: that packaging needs `msopgen` (it needs `opc`, against sources
    that already ship inside the package), that `simplifiedKey`/`staticKey` must be hand-written (both
    are generated, and our own build script asserts `opc` reproduced them), that `op_proto` must be
    rebuilt for a reused prototype (`ge::TensorType::ALL()` on every parameter), and that the existing
    overlay is someone else's precedent (it is the output of our `tests/glm52/build_gmm1_barrier.sh`).

## Corrections to earlier findings in this file

Each of these overturns something stated above, with the newer source winning. Read them before acting on
finding 3.

- **flagId on A3 is 0-15, not 0-10, and each AIC and each AIV has its own 16.** Established twice independently
  (`ascend-fusion-mechanics.md` §38.4 from the current `CrossCoreSetFlag` page; §40 from a separate pass). The range
  only *looks* like 0-10 because `SyncAll<false>` reserves 11/12/13 and `SyncAll<true>` reserves 14. Out-of-range
  ids are silently truncated to the low 4 bits, so an id reaching 16 aliases onto 0 — inside `Matmul`'s `[0, 2N-1]`.
  Counter depth is 15 (two of three sources; the API pages say 16), and overflow is a hard error, not a hang.
- **There is no flag-counter reset primitive.** A counter only falls by executing matching Waits (§40). Mode 2
  AIC→AIV requires exactly M Waits per Set with each AIV owning its own counter, and unpaired use is explicitly
  「未定义行为」 (§38.2). The three published disciplines are drain-at-a-barrier (MegaGDN), a 15-deep credit window
  (`catlass` `CrossCoreSetFlagWithReverse`), and flagId rotation (MegaMoE). The unbounded alternative is a GM
  counter, which touches no flagId at all (§40).
- **`PIPE_S` is an unsupported pipe for every sync mode A3 has — and it is the header default** for
  `CrossCoreWaitFlag` (§38.3). Write the pipe out.
- **Mode 2 only reaches a cube's own two subblocks**: AIC[i] can flag AIV[2i] and AIV[2i+1] and nothing else, so the
  cube's output tile must be the exact union of its two AIVs' input tiles (§37, with the shipped two-line tiling
  recipe in §41.3).
- **The two halves of a MIX kernel number the grid differently.** `GetBlockNum()` returns the AIC count on *both*
  halves, while the AIV's `GetBlockIdx()` is a global AIV index in `[0, 2N)`. Shipped code works around it with
  `aivNum = GetBlockNum() * GetSubBlockNum()` and the comment 「doc say aivNum has already mul subNum, by we test
  are not」 (§39.2). `SetDim` counts AIV in `matmul_fused`, AIC in `rope_matrix` — check per API, never assume.
- **The cube→vector path has no cache-coherence problem on A3, only an ordering one.** On 2201 the three caches are
  L2 (shared, no coherence issue), DCache (per-core, Scalar-only) and ICache (read-only); FIXPIPE→GM and MTE2→UB
  both go through L2 (§35.1). What is needed is `DataSyncBarrier<MemDsbT::DDR>()` before raising the flag, not
  `DataCacheCleanAndInvalid`. Finding 3b's "no L0C→UB path" is confirmed, and the L1 escape is closed too:
  `Fixpipe(L0C→UB)`, `DataCopy(L0C→UB)` and `DataCopyL1ToUB` are all 不支持 on A3 (§35.4).
- **§47's two-writer hazard has a precondition: `REGIST_CUBE_OBJ` *is* the half-partition, and it works by
  `return`.** `ascend-fusion-mechanics.md` §47.4, found by checking the claim against the vendor's own `__mix__`
  examples (one of three ships an unguarded UB→GM `DataCopy` and is nonetheless correct). In MIX mode the
  **cube** half runs the KFC server loop and `return`s inside the macro; under `ASCENDC_CUBE_ONLY` the **vector**
  half's `REGIST_CUBE_OBJ` expands to a bare `return`. So an unguarded UB→GM copy after `REGIST_MATMUL_OBJ` is
  safe, and §47.3 applies only to a hand-partitioned kernel. The flip side is a second, simpler silent failure:
  in an `ASCENDC_CUBE_ONLY` kernel that calls `REGIST_*_OBJ`, **every vector stage after that line never runs on
  any core.**
- **`sqe_type=3, errType=0x20` is not a kernel-type rejection.** Decoded from the open-source runtime
  (`gitcode.com/cann/runtime`): `RT_DAVID_SQE_TYPE_PLACE_HOLDER = 3` while an AIC kernel SQE is 0, AIV 1,
  FUSION/MIX 2; and `RT_STARS_CQE_ERR_TYPE_SW_STATUS = (1ULL << 5U)` = 0x20, rendered by `stars_engine.cc` as
  「sq sw status error」. A PLACE_HOLDER SQE with a software-status error — the inference that AIC-only kernels
  are unsubmittable does not follow from it.
- **Mode 2 is bidirectional, and mode 3 is not in the A3 mode set.** `dav_c220/kernel_operator_sync_impl.h`'s
  `SyncAllImpl<false>` uses mode 2 in *both* directions in one function (`SYNC_AIV_FLAG=12` vector→cube,
  `SYNC_AIC_AIV_FLAG=13` cube→vector). Direction is carried by which core sets and disambiguated by flagId,
  never by the mode; `wait_flag_dev(flagId)` is mode-blind. The real constraint on a reverse mode-2 is pairing
  cardinality: *both* AIVs of an AI Core must set before the AIC's wait clears.
- **`SyncAll` reserves 11-13 only for `SyncAll<false>`.** `isAIVOnly = true` is the *default* and takes flagId
  14 alone (`SYNC_AIV_ONLY_ALL`). 11 = AIC, 12 = AIV, 13 = AIC+AIV, 14 = AIV_ONLY_ALL; the counter is 4-bit per
  (core, flagId) and traps after 15 unbalanced sets — a real risk for a per-layer flag across 78 layers.
  `SoftSyncAllImpl(gmWorkspace, ubWorkspace, usedCores)` costs **zero** flagIds and can barrier a subset.
- **`__schedmode__(1)` is required for cross-core sync, not advisory.** Without it FFTS dispatches a task's
  cores in batches and the barrier count never completes; the signature is a launch that never returns at
  250-360 W and 0% utilization. Stronger than the earlier "3 hangs in 49 runs" probe.
- **Our 3.3x on the MoE expert chain is against the eager path, not the fused one.** Measured A3 + CANN
  9.2.0-beta.2 numbers show SuperKernel's `early_start` alone buying 6.01-15.21%, weighted 9.18%, with no
  hand-written kernel.
- **Not every `PipeBarrier<PIPE_ALL>` is an antipattern.** 「核函数（Kernel）结束前需要开发者手动调用
  PipeBarrier<PIPE_ALL>()」 — the one at kernel exit is mandatory (§48.1). And the hand-allocatable event-id
  range is **0-5**, not 0-7: 6 and 7 are reserved and using them is 「未定义行为」.
- **`GroupedMatmul` is not in `ops-nn`.** It is in `hicann/ops-transformer` under `gmm/`, with the fused MoE
  megakernel in `mc2/mega_moe` including an A3-specific `mega_moe_kernel_a3.hpp` (`fused-moe-expert.md` Part 8).
- **SUPERSEDED — finding 6's 「`PIPE_ALL` is not defined to cover a cross-pipe hazard」, and §43.1-§43.2 with it.**
  That rested on the CANN 8.0.RC2.2 `PipeBarrier` page's 「阻塞相同流水」. The 9.2.0 page
  (`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.md`) says
  「如果不关注流水类别，希望阻塞所有流水，可以传入PIPE_ALL」 and
  「`PipeBarrier<PIPE_ALL>()`会等待所有流水线中所有先前提交的接口完成，这会对性能产生影响」. So `PIPE_ALL` **is** an
  all-pipe wait on the line we run, and the reason to prefer a named pair is cost, not coverage. The claim that
  survives, and is narrower and more useful, is in §58.3: `PipeBarrier` **cannot name `PIPE_S`** (「其中不支持PIPE_S」),
  so a Scalar edge — `MTE2`→`GetValue`, `SetValue`→`DataCopyPad` — is the one edge for which no barrier exists and
  only `MTE2_S` / `S_MTE3` will do. Anything in this repository that converted a `PipeBarrier<PIPE_ALL>` site to
  named events *on correctness grounds* was acting on the superseded reading; the conversions are still right, for
  a different reason, and they do not cover the Scalar edges or the `V_MTE2` edge.
- **There is no compiler auto-sync in `tests/glm52/`.** `key_features.md:135`'s 「异构编译直调（<<<>>>调用）…已默认设置
  `--cce-auto-sync=on`」 is about the *project templates*; the switch is opt-in
  (`ai_core_operator_compilation.md:119`) and the build system appends it (`compile_op.py:1100-1108`) — while the
  TBE path appends `--cce-auto-sync=off` for A3 (`ccec.py:329-337`). Every `build.sh` here calls `bisheng -xcce`
  directly without it, so **every** edge in §58.2's matrix is the author's to write, `PipeBarrier<PIPE_V>()`
  included. §58.6.
- **Finding 12's tiling-key-switched design may not be expressible, and moving into an operator project costs
  `__mix__` and `__schedmode__` rather than buying anything.** `ascend-custom-op-packaging.md` §5.1-§5.2, from the
  installed compiler. The op-project build textually includes your `.cpp` with `#define __global__ inline` and
  generates `auto_gen_<func>_kernel` as the real `cce_kernel`
  (`asc_op_compile_base/asc_op_compiler/compile_op.py:784-830`), so `__mix__(1,2)` lands on a non-kernel function
  and bisheng hard-errors, and `__schedmode__(1)` on an inlined callee is dropped — the framework substitutes
  `intercoreSync`/`crossCoreSync` in the descriptor, which is **unverified** and is the first thing to check on a
  first build, since this file records `__schedmode__(1)` as required. The ratio then comes from hardware via
  `GetTaskRationImpl()` (`dav_c220/kernel_operator_sys_var_impl.h:43-54`). And `kernel_info_infer.py:218-238`
  accepts only `KERNEL_TYPE_MIX_AIC_1_1` or `MIX_AIC_1_2` as the keyed form's second argument, with the first
  required to be a numeric literal (regex at `:110`); `KERNEL_TASK_TYPE_DEFAULT` takes the full enum. If that
  holds, finding 12's nine-`AIC_ONLY`-plus-three-`AIV_ONLY` single binary cannot be built on CANN 9.2.0 and the
  4.5x lever needs separate binaries — design three, not design two. Unresolved: the `swiglu_gated_mlp` reading
  came from a public repository, not this installation, so grep that kernel for whether its `AIC_ONLY` sites use
  the keyed macro or the default one. Also retired: finding 12's "With `<<<>>>` only `KERNEL_TASK_TYPE_DEFAULT`
  applies" — under `<<<>>>` **neither** form applies. `KERNEL_TASK_TYPE*` expands to
  `ENABLE_FEATURE_FOR_COMPILE`, which is empty unless `__CHECK_FEATURE_AT_PRECOMPILE` is defined, and only the
  project driver defines it (`kernel_utils_macros.h:365-387`). `routed_expert_kernel.cpp:330` is a silent no-op
  and the routed kernel's 1:2 comes entirely from `__mix__(1, 2)`.

## Where to look things up

`asc.gitcode.com/api/` is a complete static mirror of the Ascend C API reference: path-addressable, newer than the
`doc_center` copies, and every page carries an explicit 「产品支持情况」 block naming Atlas A3. Start there.
`hiascend.com/document/detail/...` returns an empty body to a fetcher; `hiascend.com/doc_center/source/...` works
but its id space is undiscoverable. CANN 9.2.0 is not published on the portal at all — 900 is the newest text.
Details and the URL shape in `ascend-fusion-mechanics.md` §35.0.

## Reading order if you are about to write the megakernel

0. `megakernel-persistent-decode.md` §8 — MegaGDN. The minimum viable Ascend megakernel, with
   a working grid barrier for A2/A3 and A5, and the fact that licenses the whole approach:
   one `__aicore__` function may call another, so stages compose without an interpreter.
1. `megakernel-persistent-decode.md` §4 and §4b — CANN SuperKernel. An Ascend megakernel
   interpreter also exists, is open source, and its device-side dispatch loop, task
   descriptors, core sync and GM notify/wait are reproduced there verbatim — along with the
   enums that decide what it will and will not fuse.
2. `ascend-launch-overhead-and-sync.md` §10 — `blade/runtime`. A second, more ambitious Ascend
   megakernel runtime exists internally with an AICPU scheduler and per-core HBM rings. Decide
   whether to obtain it before reimplementing it.
3. `ascend-launch-overhead-and-sync.md` §1 — where the 13.58 us actually goes, and why static
   shapes remove 2.8 us of it for free.
4. `ascend-launch-overhead-and-sync.md` §3, §4 and §12.3 — the synchronization budget (8 event
   ids per pipe pair, 15 outstanding sets per flag) and the two ways round it (GM counters,
   scoreboard AIV).
5. `megakernel-persistent-decode.md` §1 and §2 — the two canonical designs, for the paged
   on-chip allocator and the AOT/JIT task-launch distinction.
6. `ascend-fusion-mechanics.md` §37-§41 — the MIX brick, start to finish, learned the hard way. §38 is the
   authoritative `CrossCoreSetFlag` semantics (pairing arity, counter residue, the `PIPE_S` trap). §39 is two
   complete vendor kernels: a shipped multi-block `ASCENDC_CUBE_ONLY` + flag handoff (`rope_matrix`) and the KFC
   route where the AIV drives the cube and waits on a `TQue`. §41 is the per-layer kernel: the Matmul object's
   lifetime across stages, `SyncAll<false>()` as a mid-kernel phase boundary with a shipped three-barrier example,
   and the pairing tiling. §35 is the A3 cache model that makes the handoff rule one instruction.

## If you are debugging, not designing — start here

The `int4b_t` / synchronization / silent-failure block, in the order it was learned:

1. `ascend-fusion-mechanics.md` §47 — `Cast` is `if ASCEND_IS_AIV`-guarded and plain `DataCopy` to GM is not.
   In a `__mix__` kernel, an ungated `Cast → Cast → DataCopy(GM)` function gives the allocation a **second,
   uncoordinated writer** on the cube half. `DataCopyPad` has the guard; `DataCopy` does not. Check this before
   anything else.
2. §42 — sub-byte addressing: `sizeof(int4b_t) == 1` but `SizeOfBits<int4b_t> == 4`; `operator[]` counts
   nibbles and floors an odd index silently; and `SetGlobalBuffer`'s length argument is **inert on NPU**. Read
   §58.14 with it: `GlobalTensor<int4b_t>::operator[]` counts nibbles *correctly*, and is bit-for-bit identical to
   a hand-rolled `(__gm__ int8_t*)base + elements / 2`, so a helper written to work around it is redundant and the
   subscript is never the variable to suspect. When an int4 store looks like it lands only at offset zero, suspect
   the transfer's **length** — an element count where the API wants bytes writes twice as far and spills one span
   past the region, which presents as identical bytes at two offsets one span apart.
2b. §59 — a `Cast`'s `RoundMode` rounds onto the **destination type's** grid, so `float → half` with `CAST_RINT`
   is *not* a rounding to integer (the vendor's two worked examples give 0.5 and 0.0 for the same input and mode).
   An MSD nibble split fed a non-integral value is faithful in its recombination but its low half reaches +8 on
   one thirty-second of the line and **saturates to 7**, costing one code unit, invisibly — and the band that
   bites is `[-0.5, 0)`, i.e. every small negative activation. Measured: 2.95% of a uniform sweep escapes,
   2.08% of recombinations are off by one, and both go to zero on integral input.
3. §45 — the public found-and-fixed W4A4 erratum, whose root cause is exactly that `sizeof`. Carries the GM
   packing contract (low nibble = even element, row *r* at byte `r*N/2`), the K/N-multiple-of-64 rule, the
   elements-vs-bytes split between `DataCopy(count)` and `Nd2NzParams`, and the fact that `torch.int4` is
   **unpacked**.
4. §43 — a foreign-core `WaitFlag` is a silent no-op on 2201 while its `SetFlag` still fires; the legal
   `HardEvent` set is 13 on an AIV and 24 on an AIC; `M_V`/`V_M`/`MTE1_V`/`V_MTE1` must never be named at all.
5. §46 — `GetQueEvt`, the library's own `PipeBarrier → HardEvent` conversion table, for converting barrier
   sites to named pairs. §48 — the mandatory exit barrier, the 0-5 event-id range, the API allowlist that
   applies once you allocate ids yourself, and `InitSocState` for any kernel without a `TPipe`.
6. §58 — the **closed** sixteen-cell AIV sync matrix, which turns "which barrier do I need" into a lookup, plus
   the three things it settles that §43 got partly wrong or did not reach: `PipeBarrier<PIPE_ALL>()` on 9.2.0 **is**
   an all-pipe wait and the objection to it is cost; `PipeBarrier` **cannot name `PIPE_S`**, so an `MTE2`-written
   UB word read by `GetValue` (`MTE2_S`) and a `SetValue`-written word read by `DataCopyPad` (`S_MTE3`) have no
   barrier form at all; and `--cce-auto-sync`, which would insert exactly those Scalar edges, is set by the
   **project templates** and not by the compiler — a hand-rolled `bisheng -xcce` line, which is how every kernel in
   `tests/glm52/` is built, has no auto-sync whatever. §58 also carries the licence to alias `ReduceMax`'s
   `sharedTmpBuffer` onto its source on A3, what that API really writes to `dst` (a 4-byte scalar store, already
   fenced), and the fact that a UB→UB `DataCopy` is PIPE_V rather than an MTE pipe.

The running list of **silent** failure modes on this part, all with sources in the sections above: a
foreign-core `WaitFlag` (§43.5); a `DataCopy` whose source `TPosition` is neither UB nor L1 (§42.4); a `Cast`
on the cube half (§47.1); `ndNum = 0` on a Fixpipe (§49.5); an odd `operator[]` index on a sub-byte type
(§42.2); a misaligned `DataCopy` count, 「In NPU mode, no error is reported. The value is rounded down by 32B」
(§42.5); `load_cbuf_to_cb_s4` dropping `ifTranspose` (§45.6); an out-of-range `HardEvent` indexing one past
`eventPool_` (§43.6); an out-of-range `flagId` truncated to its low 4 bits (see the corrections above); a
`Mutex::Lock`/`Unlock` call, whose bodies are `#if`-gated to 3510/5102 and **empty on 2201** while
`AllocMutexID()` returns 0 — so an `ENABLE_LOCK`-style arm is completely unsynchronized on A3 while looking
synchronized in the source, and `vllm-ascend`'s `csrc/attention/common/op_kernel/init_output.h` is that shape
(§56); and, under `ASCENDC_CUBE_ONLY`, `REGIST_CUBE_OBJ` expanding to a bare `return` on the vector half, after
which no vector stage runs on any core (§47.4). Two more from §58: a `GetValue` of a UB word that `MTE2` has only
just been told to fetch, which needs `MTE2_S` and gets nothing from any `PipeBarrier` and nothing from the
compiler unless `--cce-auto-sync` was typed on the command line (§58.3, §58.6); and a UB buffer loaded twice from
GM with a vector read between the loads, which needs `V_MTE2` before the second load and survives both the
all-`PipeBarrier<PIPE_ALL>` style and its conversion to named events, because every barrier in that idiom sits
*after* each load rather than before the next one (§58.2, Relevance). The keystone fact behind that second one,
and the reason a `SetFlag` between two `DataCopy`s to one buffer is not the serialiser it looks like:
「SetFlag只是设置硬件标志位，**不会阻塞源流水中的下一个指令**」 (`SetFlag_WaitFlag_ISASI.md:42`, `:72`). Two
transfers into one UB buffer are concurrently in flight; after the first flag fires the buffer can already hold
32-byte blocks of the second. Converting a `PipeBarrier<PIPE_ALL>` site to `MTE2_V` pairs *introduces* that
overlap while closing nothing, and a saturated defect metric will not show it. And it is steady state, not a coin
flip: the flag is **one bit** per (`HardEvent`, `eventID`), set to 1 and cleared to 0
(`intra_core_sync_overview.md:61-65`, which calls the pair a 「锁」机制), `WaitFlag` blocks only the *consumer*
pipe, and `CrossCoreWaitFlag`'s `pipe` argument is discarded on 2201 — so a kernel writing only `MTE2_V`,
`V_MTE3` and `MTE3_V` has placed **no stall at all on `PIPE_MTE2`**, which then runs ahead of its consumer. The
cheapest audit that catches it: count the Set/Wait pairs sharing one event id inside one straight-line region.
One is safe; three on `EVENT_ID0` collapse onto one bit. And from §59: a `Cast` to a narrower **float** type with
`RoundMode::CAST_RINT`, written in the belief that it rounds to an integer — it rounds onto the destination's
float grid, and an INT4 nibble split downstream of it then saturates its low half to 7 on one column in
thirty-two with no diagnostic of any kind.

`Mutex`/`Lock`/`Unlock` would otherwise be the right tool for a reused staging buffer — 28 ids against the 6
hand-allocatable event ids, and the vendor's own stated advantage is 「简化反向同步」 — but all five of its API
pages say `不支持` on A3. **`SetFlag`/`WaitFlag` is the only intra-core mechanism on this part**, which is
settled rather than a preference (§56.2).

## If you are closing a roofline gap — the lever table

`ascend-fusion-mechanics.md` §49-§54 and `fused-moe-expert.md` §B8 are the devkit's own
`operator_practice/best_practices` tree, mined for everything bearing on our profile. Every row is a vendor
measurement with code, not advice.

**First, the result that frames the rest** (§54.1): with K not L1-resident, Matmul GM traffic is
`M*N*K*(1/baseM + 1/baseN)` — `baseK` does not appear. At M=1, `M/baseM` is 1, so the weight is loaded exactly
once: **traffic volume is already at its floor and no base-block choice can reduce it.** A batch=1 roofline gap
is therefore traffic *efficiency* or *Scalar*, by arithmetic rather than by profiling.

| lever | measured | where | gate |
| --- | --- | --- | --- |
| AIV ND2NZ instead of on-the-fly, for a non-256B-aligned inner axis | **−39.0%** (149.04 → 90.95 us) | §53.2 | inner axis not 256B aligned **and** data volume large. **`moe_intermediate=1792` is 896 B = 3.5×256, misaligned in int4** |
| pure-Cube mode (`ASCENDC_CUBE_ONLY`) | −37.2% `aic_time`, **−65.6% Scalar** | §B8.6 | no vector logic in the operator; core split changes 48 AIV → 24 AIC blocks |
| split-K when M and N are small | −30.1% (19.60 → 13.70 us) | §51.1 | C to **GM only**, output **zeroed + AtomicAdd**, **no Bias** |
| AIC:AIV 1:2 on a Vector-bound grouped matmul | −29.3% (218.1 → 154.2 us) | §B8.3 | inner loop count **must not be 1** |
| constant tiling (`MatmulApiStaticTiling`) | −25.9% `aic_time`, **−46.5% Scalar** | §53.1 | const `singleCore*` + const `base*` (full) or const `base*` only (partial) |
| AB IBShare + cube-view core split | −17.2% (27.11 → 22.44 us) | §52.2 | all Matmul objects in the operator must match; `IterateAll` to GM only |
| workspace 2 → 4 (credit window depth 4) | −14.5% (154.2 → 131.8 us) | §B8.4 | 4 × baseM × baseN of workspace |
| K-axis staggered access | **−13.2% with tiling unchanged** | §51.2 | MDL template, K **not** L1-resident, multi-core, A or B in GM, K > 4096 |
| large-packet transfer (`depthA1`/`depthB1`) | MTE2 −27%, 2491 → **3406 GB/s** | §54.3 | — |
| L2 split + small tail + 错位分核 | −7.1%, MTE2 −10.7% | §51.3 | from a 99%-MTE2 starting point |
| MTE2 Preload (`doMTE2Preload`) | −7.7% (30.88 → 28.50 us) | §B8.7 | MDL/SpecialMDL, `singleK <= baseK*stepK`, DB on M or N |
| B-only IBShare | −3.55%, **MTE2 −15.46%** | §52.1 | shared operand **fully L1-resident**: `singleCoreN = baseN*stepN`, `singleCoreK = baseK*stepKb` |
| NBuffer33 | −4.0% (34.01 → 32.66 us) | §53.3 | MDL, GM only, **pure-Cube only**, `IterateAll` only, `step* <= 3` |
| `SetL2CacheHint(CACHE_MODE_DISABLE)` on a read-once weight | unquantified | §B8.4, §50.6 | the vendor does it at `numBlocksM == 1`, i.e. our decode condition |
| `REQ8`/`VREQ8` int32→int8 requant on the Fixpipe drain | removes a Vector `Cast` | §49.1-§49.3 | **no int4 destination exists** on any product |

**The Scalar pattern.** Four independent cases say the Cube's *Scalar* pipeline binds at our shapes, not Cube
and not MTE2: 26.27 of 27.11 us (§52.2), 15.02 of 17.85 (§B8.6), 6.32 of 10.62 (§53.1), plus our own 13.58 us
MIX_AIC floor. The one ratio to read is `aic_scalar_time` against `aic_time` — msOpProf column G against
column C.

**Two levers point opposite ways, deliberately.** §54 fills all cores (4 → 20 took 12045 → 2532 us, 4.75x);
§50.3 says launch *fewer* at microsecond scale because head overhead grows with core count
(「对于整体耗时在微秒级别且单核计算量耗时较少的算子，可以通过减少启动核数并增加单核计算量的方式来获得性能提升」).
Both hold in their own regime. Measure per call, not per family.

**Ruled out, with numbers**: MC2 / fused MatmulAllReduce at batch=1 — `low-latency-tp-reduce.md` Part 8. Only
the M axis may be split (HCCL needs contiguity), the vendor's own short-block floor is `c >= 3*128` = 384 rows,
and the 32.7% in their worked case comes from hiding an 803 us matmul behind a 1071 us AllReduce. At M=1 there
is no axis and nothing to hide. Keep two numbers from it anyway: the AllReduce cost fit has a **14.769 us
constant term** below 8 MB (a fixed per-call floor — ~1.15 ms across 78 layers), and **1.15** is the vendor's
empirical allowance for compute/comm memory-bandwidth contention.

## Provenance of the header-derived findings (§42-§56), and how they were version-checked

Everything in `ascend-fusion-mechanics.md` §42-§56, `fused-moe-expert.md` §B7-§B8 and
`low-latency-tp-reduce.md` Part 8 that is attributed to "the shipped AscendC implementation" or "the shipped
devkit" comes from the local clone at **`/home/shanshan/repo/cann/asc-devkit/`**, HEAD
`6a51d0b167dd8f5555456a0ebb783ef0427e366e`, dated **2026-08-21**. The devkit's tags map 1:1 to CANN releases
(its own `CHANGELOG_en.md` carries a 「CANN Version | Version Source Tag」 table: `v9.1.0-beta.2` ↔ CANN
9.1.0-beta.2, and so on), and the clone's HEAD is a master commit **three weeks newer than the local
`v9.2.0-beta.1` tag** (2026-08-03). Upstream `gitcode.com/cann/asc-devkit` also carries `v9.2.0`,
`v9.2.0-beta.2` and `v9.2.0.pre`, which are not in the local clone.

Because we run **CANN 9.2.0**, every load-bearing structural claim was re-checked against the local
`v9.2.0-beta.1` tag with `git show`. All of them hold unchanged:

| claim | file | status at `v9.2.0-beta.1` |
| --- | --- | --- |
| §43 Set/Wait asymmetry, `HardEvent` enums, `EventToIndex`, `QUE_MAX_EVENT`, §46 `GetQueEvt` | `impl/basic_api/kernel_event.h` | **file byte-identical to HEAD** |
| §47.1 every `CastImpl` wrapped in `if ASCEND_IS_AIV` | `impl/basic_api/dav_c220/kernel_operator_vec_vconv_impl.h` | **file byte-identical to HEAD** |
| §47.2 `ASCEND_IS_AIC` guard in the `DataCopyPad` family only, never in plain `DataCopy` | `impl/basic_api/kernel_operator_data_copy_intf_impl.h` | 8 occurrences, all at line 1412+, i.e. all in the Pad region — same structure |
| §42.3 `bufferSize_` read nowhere on the `DataCopy` path | `impl/basic_api/kernel_tensor_impl.h` | the same 4 sites: write, CPU-debug zero, `GetSize`, `ReinterpretCast` rescale |
| §56.2 `Mutex::Lock`/`Unlock` bodies `#if`-gated to 3510/5102 | `include/basic_api/kernel_common.h` | gate identical |
| §43.5 `SPLIT_CORE_CUBE` / `SPLIT_CORE_VEC` definitions | `impl/utils/sys_macros.h` | identical (the file's only diff is five unrelated CPU-debug type aliases) |
| §47.4 `REGIST_CUBE_OBJ` → bare `return` on the vector half under `ASCENDC_CUBE_ONLY` | `impl/adv_api/detail/kfc/kfc_register_obj.h` | identical |

So these are not findings from a stale tree: they are verified on the CANN 9.2.0 line we run. What remains
unverified is the gap between `v9.2.0-beta.1` and the `v9.2.0` release tag, which is not in the local clone —
if any of §42-§56 is ever contradicted on hardware, fetching `v9.2.0` and re-diffing those seven files is the
first thing to do.

## Provenance rule

`ascend-launch-overhead-and-sync.md` is sourced entirely from `wiki.huawei.com` and is marked
internal at the top of the file. Every other file is public material. Keep that boundary when
anything here is copied elsewhere.
