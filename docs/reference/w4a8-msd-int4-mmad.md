# W4A8 matmul without a mixed-width MAD — CANN's MSD decomposition

Source material on the one problem a hand-written W4A8 kernel on dav-2201 cannot avoid: `Mmad`
takes INT4 only when *both* operands are INT4, so an INT8 activation against an INT4 weight has
no single instruction. CANN's answer is a named scheme, MSD, shipped as a three-stage operator
family. Everything below is quoted as published.

> **Read §12 first.** An exact `int32` MSD already exists on A3, as an open draft PR against
> vllm-ascend, and it deletes the offline `8*sum(w)` tensor by computing the correction on-chip as
> one extra matmul row of packed int4 `-8` values (`0x8888`). `MsdC = MatmulType<..., int32_t>`,
> `rawRows = 2 * valid + 1`, 66/66 golden checks, with the exactness bound written out in the
> source: `K <= 16384` keeps every integer intermediate inside `[-2^24, 2^24]` so the fp32 vector
> reconstruction is exact. That is the design to copy, and it supersedes the shape of the problem
> every other section describes.

> **The contradiction worth carrying around:** §1 claims the decomposition is exact
> (「保证结果等价于原始INT8×INT4计算结果」, 「数值与直接INT8×INT4乘法结果一致」), and all four
> shipped CANN implementations declare their matmul output type as `half` — and so do *both* of
> Huawei's own golden references (`atomic = np.float16` in `catlass`, `dtype=torch.float16` in the
> ST). The `×16` is applied after an fp16 rounding, so the high partial product's error is
> amplified 16-fold. The published acceptance tolerance is `2**-10`, i.e. the fp16 resolution.
> The *identity* is exact; the *shipped implementation* is not; the *claim* is prose. See §2.4,
> §11.2, §13.3.

Reading order:

| Want | Go to |
| --- | --- |
| The design to build | §12 (exact int32 + correction row), then §10 (L1-resident A) |
| Huawei's own writeup, and the only expansion of the acronym | §1 |
| The shipped kernel, instruction by instruction | §2 (SwiGLU variant), §3 (plain GroupedMatmul, better engineered), §5 (non-grouped) |
| The official formula and the operator contract | §4 (grouped), §5.2 (non-grouped) |
| What `Mmad`/`Matmul`/`Cast` will and will not accept for int4 on A3 | §11.1 |
| Why "MSD" sends you to the wrong source | §7 |
| Where `8*sum(w)` is computed offline, and the GLM-5 recipe | §13 |
| Ways a hand-written int4 kernel silently computes garbage | §14.3 |
| Accuracy evidence, such as it is | §11.2.3, §13.5 |

---

## 1. Huawei official: 【昇腾热点算子大解密】CANN极致优化GroupedMatmul量化方案，MoE推理Decode性能提升超30%

URL: https://www.hiascend.com/developer/techArticles/20250717-2
Author: 昇腾社区 官方技术文章 (tags 昇腾CANN / 算子开发 / 性能调优)
Date: 发表于 2025/07/17
Fetched: 2026-10-20 (via `curl`; the `developer/techArticles/` path serves full HTML to a
fetcher, unlike `document/detail/`)

This is the only public document that expands the acronym. It does **not** mean
most-significant-digit or 多位十进制:

> 因此，我们将W4A8量化方案与昇腾硬件微架构特性结合优化，创新性地提出MSD（Mixed-precision
> Split-activation Decomposition，混合精度激活拆分分解）方案，将INT8激活拆分为高、低两路INT4，
> 并直接与INT4权重进行计算，用“高位×16+低位”合成，在保证结果等价于原始INT8×INT4计算结果的同时，
> 大幅降低KV Cache和参数存储需求，提升单卡batch size和上下文的支持长度。

### 1.1 Why the split exists at all — the redundant-dequant motivation

> 然而，仅依赖W4A8方案并不能充分释放低比特运算的潜力。推理流程中不同阶段对计算和访存的依赖有所区别，
> Prefill阶段的activation更长，计算更密集；而Decode阶段activation更短，访存压力更大，两者对硬件资源的
> 需求和优化重点不同。面对Decode阶段的访存瓶颈，传统W4A8方案虽然能降低显存占用，但在实际计算时仍需将
> INT4权重还原为INT8后，再与INT8激活做矩阵乘法，有冗余的反量化计算开销。

And on why GroupedMatmul is the carrier:

> 分组矩阵乘法（GroupedMatmul）是一种将多个独立的小规模矩阵乘法（GEMM）任务分组并行执行的优化技术，
> 通过合并计算拓扑相似的算子为单一内核调用，有效降低核启动开销并减少内存访问碎片化。
>
> MoE模型在Decode阶段每个专家处理的激活数量偏少，导致传统的GEMM操作因单个专家计算规模小而难以充分发挥
> 硬件算力。GroupedMatmul天然适配MoE模型的稀疏专家计算，通过将分散的专家任务合并为分组批量进行计算，
> 提升任务规模以实现接近峰值算力的效率；同时，连续存储专家权重可以进一步优化缓存命中率，显著提升计算性能。

> W4A8量化方案是当前主流的大模型低比特量化方案之一，通常结合per-channel、per-group和per-token等多种
> 量化scale，兼顾精度与效率。通过将权重量化为INT4格式、激活量化为INT8格式，可以极大压缩模型参数的存储
> 需求，W4A8方案的权重占用仅为传统FP16方案的四分之一。其显存优化能力，为大模型在有限硬件资源下的高效
> 部署提供了坚实基础。
>
> 下图展示了GroupedMatmul的非对称/对称量化方案。在推理时，INT4权重通过scale和zero point解码，激活则
> 直接以INT8参与整数GEMM计算；所有量化/反量化操作均在整数域内完成，最后一步统一反量化为浮点。

### 1.2 The three scale levels, and the admission that per-group barely helps

> W4A8量化方案采用三层量化scale设计，实现了精细化的量化控制，其数学公式表达如下：
>
> 其中量化scale分别为：
>
> (1) Per-channel scale：针对weight的通道级量化参数，保证不同通道间的精度平衡
>
> (2) Per-group scale：针对weight的分组级量化参数，提供更细粒度的量化控制
>
> (3) Per-token scale：针对activation的token级量化参数，动态适应不同输入的分布特性
>
> 这种多层级的量化策略能够更好地适应不同数据分布的特点，有效降低量化误差。**实践中发现，per-group量化
> 对模型精度的提升较为轻微，所以实践中一般可以不采用per-group量化。**

(bold ours; the sentence is verbatim)

### 1.3 前处理 — the split, and the `8*weight` offset

> **前处理：完成MSD方案拆分过程**
>
> MSD方案的核心是将单一INT8激活拆分为两个INT4分量，这个拆分过程在前处理中完成。假设原始INT8激活为
> x_int8，那么其拆分过程可以表示为：
>
> 其中A₁[int4]和A₂[int4]分别为activation的高4位和低4位的INT4表示。
>
> 当INT8被拆分为两个INT4时，由于INT4的符号位决定了其范围在[-8, 7]，而低4位的最高有效位（即符号扩展）
> 会带来一个偏移量，具体就是8*weight。为了保证拆分的精度，需要将拆分操作变型为如下过程：
>
> 这个多余项在所有INT8拆分为INT4的操作中是固定的，等价于每个权重通道都引入了一个偏置项。在
> GroupedMatmul的具体实现中，通常将这部分多余项放入bias参数中，通过离线计算提前补偿，从而保证推理精度
> 和性能。下图展示了前处理的计算流程：

(The two displayed formulas and the three figures are images, not text, and did not survive the
fetch. The prose pins the arithmetic down completely anyway, and §2 has the executable version.)

### 1.4 MatMul — the two weight scales are folded offline into one `anti-scale`

> **MatMul过程**
>
> **1. Per-channel和per-group scale的融合**
>
> 权重首先以per-channel FP16 scale进行粗量化，得到中间的INT8权重。然后对INT8权重再做per-group INT4
> 量化，得到最终的INT4权重和per-group scale。
>
> 如上图所示，x与weight被拆分后完成INT4矩阵乘。在实际推理时，为了提升效率，per-channel scale和
> per-group scale会离线融合成一个总scale（图中的anti-scale），这样可以在GEMM输出阶段一次性完成浮点还原，
> 减少多次逐步缩放带来的性能损耗。
>
> 融合后的scale一般为total_scale = perchannel_scale * pergroup_scale。这样，矩阵乘法主环节只需用整数
> 计算，最后输出时统一乘以融合后的scale即可。
>
> **2. GEMM的per-group累加实现**
>
> Per-group scale是针对权重的分组量化，每组权重共用一个scale。在GEMM实现中，通常沿着k轴（即输入通道）
> 切分，每g个通道为一组（g为group size），每组使用独立的scale。
>
> 在做矩阵乘法时，需要将输入矩阵X（已分解为INT4）与权重矩阵W（INT4，已分组）分块处理。**对每个group，
> 通过total scale将INT4权重直接解码为FP16。** 具体而言，每进行一组累加，就应用对应的per-group scale。

(bold ours. 「直接解码为FP16」 is the officially documented FP16 intermediate — it is not an
implementation accident. This is the sentence that §2.4's contradiction turns on.)

### 1.5 后处理 — the three steps, in order

> **后处理：合成并反量化输出结果**
>
> W4A8量化后处理主要包括三个步骤：高低位结果合成、bias补偿、反量化（per-token scale）。如下图，matmul
> 的结果在图中表示为C，经过后处理后输出的结果为y。
>
> 下面详细介绍各环节的原理和实现方式：
>
> (1) 高低位结果合成：在W4A8方案中，INT8激活被拆分为高4位和低4位。为还原INT8原始乘法的效果，需要将高4位
> 的结果左移4位（即乘以16），再与低4位结果相加。这样可以保证拆分计算后，数值与直接INT8×INT4乘法结果一致。
>
> (2) BIAS余量补偿：拆分INT8为两个INT4时，由于符号扩展和量化区间的差异，会引入一个额外的常量项
> （8×W×scale）。这个多余项在量化离线阶段提前计算，并作为bias参数加到最终输出上，补偿拆分带来的偏移，
> 保证数值精度。在实际后处理时，直接将该bias向量加到矩阵乘法的结果上即可，无需额外计算，且不会影响推理效率。
>
> (3) Per-token反量化：W4A8方案通常采用per-token scale对激活进行量化，推理后需反量化还原浮点结果。
> Per-token反量化中包含每个token（即每一行激活）独立的量化尺度，能够动态适应不同输入分布，提升精度。
> 这种逐token的反量化方式，能有效降低量化误差，尤其适合大模型推理中的动态输入场景。

### 1.6 Measured result, and the published entry point

> 至此，W4A8量化与MSD方案的所有主流程均在整数域内完成，充分利用昇腾硬件对低比特整数计算的支持，减少主循环
> 的访存和运算瓶颈，为LLM的高效推理提供了关键支撑，极大提升大模型推理的响应速度和服务能力。**基于W4A8
> 方案的DeepSeek R1单机部署实践表明，其最大BatchSize达到了W8A8方案的3.1倍；同时，W4A8+MSD方案在Decode
> 阶段的性能更是达到了W8A8的1.3~1.7倍。**

> GroupedMatmul算子功能通过CANN软件包使能，社区版资源下载地址：
> https://www.hiascend.com/developer/download/community/result?module=cann
>
> 算子接口定义为：
>
> ```
> aclnnStatus aclnnGroupedMatmulV4GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight, const aclTensorList *biasOptional, const aclTensorList *scaleOptional,
> const aclTensorList *offsetOptional, const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional, const aclTensorList *perTokenScaleOptional,
> const aclTensor *groupListOptional, const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional, const aclTensorList *activationQuantOffsetOptional,
> int64_t splitItem, int64_t groupType, int64_t groupListType, int64_t actType, aclTensorList *out, aclTensorList *activationFeatureOutOptional, aclTensorList *dynQuantScaleOutOptional,
> uint64_t *workspaceSize, aclOpExecutor **executor)
>
> aclnnStatus aclnnGroupedMatmulV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
> ```
>
> 详细接口文档可参考昇腾社区API描述：
> https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/aolapi/context/aclnnGroupedMatmulV4.md

### Relevance

Four things we did not have before, all from this one page:

1. **The acronym is not what we assumed.** MSD = "Mixed-precision Split-activation
   Decomposition / 混合精度激活拆分分解". Not "most-significant-digit", not "多位十进制". Correct
   this in our own notes — searching for the wrong expansion is why this page is hard to find.
2. **The `8*sum(w)` fold is Huawei's own published mechanism**, under the name 「BIAS余量补偿」,
   with the offset written as `8*weight` / 「8×W×scale」, 「在量化离线阶段提前计算」 and 「放入
   bias参数中」. Our per-column bias is the vendor's design, not a workaround.
3. **Per-group weight quantisation is officially rated near-worthless** —
   「per-group量化对模型精度的提升较为轻微，所以实践中一般可以不采用per-group量化」. We can set
   `quantGroupNum = 1` (one scale per column over the whole K) and keep the vendor's own accuracy
   story. That collapses the `loopK` loop in §2.2 to a single `Iterate`, which removes the
   fp16 atomic re-accumulation in §2.4 *and* the per-group scale reload. This is the cheapest
   accuracy-and-speed win on the page.
4. **The win is claimed only for Decode, at 1.3-1.7x over W8A8**, i.e. the same regime we are in.
   The 3.1x is batch size, not speed; do not quote it as a latency number.

---

## 2. The shipped kernel source — `ops-transformer/gmm/grouped_matmul_swiglu_quant`

Repository: `https://gitcode.com/cann/ops-transformer` (CANN Open Software License Agreement
Version 2.0; headers carry `Copyright (c) 2025 Huawei Technologies Co., Ltd.`)
Read from a local snapshot at `d6aaa9fe52300b2fc7138742eb71ba265000196b` (2026-10-19). Every
quote below was then re-verified line by line against a second clone whose `origin` is literally
`https://gitcode.com/cann/ops-transformer.git`, at `e027a008be2766fa5cb6198bebbc7b9dcc08310b`
(2026-09-01, 【bug修复】aclnnRainFusionAttentionGetWorkspaceSize 入参传递错误修复). The two are
*not* byte-identical — the newer snapshot has been reformatted (`T *p` → `T* p`, macro
continuation backslashes realigned) and a few lines moved — but every constant, type, instruction
and comment quoted in §2-§4 is present verbatim in the gitcode-origin clone, at the line numbers
checked below. `git ls-remote https://gitcode.com/cann/ops-transformer.git` answers from this
machine (branches `8.5.0`, `8.5.0-beta.1`, `9.0.0`, ...), so the material is public; the
`gitcode.com/.../raw/master/...` URL form, however, serves a JavaScript shell rather than file
content to a fetcher, so quote from a clone, not a fetch.

Verified line numbers in the gitcode-origin clone, for anyone re-checking:
`..._msd_post.h:24` (`DEFAULT_MUL_SCALE`), `:122` (`// 2. high_4bit * 16 + low_4bit`), `:124-128`
(the `Muls`/`Add`); `..._msd_mid.h:187` (`* 2; // 2: int8 has been split in 2 int4`), `:228`
(`// 8, 256, 512 --> 514us`); `..._msd_pre.h:88` (the 7B UB budget), `:144` (`0x0F0F`), `:148`
(`ONE_SIXTEENTH`), `:174` (`CAST_FLOOR`), `:194-195` (`MINUS_EIGHT`);
`grouped_matmul_antiquant_a8w4_msd.h:120` (`// for 8 * weight`), `:368-369`, `:456`
(`RIGHT_MOVE`), `:546`, `:569`; `grouped_matmul_antiquant_a8w4_msd_pre.h:126`, `:130`, `:179`,
`:192` (`// shrink 0~15 to -8~7`); `docs/aclnnGroupedMatmulV4.md:570`, `:659`;
`grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md:131`, `:135`, `:142`,
`:147`, `:153`. `grouped_matmul_swiglu_quant.cpp` is identical up to whitespace in both.

Paths, verbatim:

```
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.cpp
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_pre.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_mid.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_post.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_utils_kernel.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h
gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h
gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_new.h
gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre.h
gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre_nz.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_pre.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_l1_opt.h
```

### 2.0 The dispatch — A8W8 gets `int32_t`, A8W4-MSD gets `half`

`grouped_matmul_swiglu_quant.cpp`, both macros side by side, verbatim:

```cpp
#define GMM_CV_SPLIT_IMP(computeClass, dtypeWeightScale, transA, transB, sync) \
    do { \
        using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, false>; \
        using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, DTYPE_WEIGHT, false>; \
        using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>; \
        using matmulType = MMImplTypeStatic<xType, weightType, yType>; \
        matmulType::MT mm; \
        ...

#define GMM_CV_SPLIT_IMP_A8W4_MSD(computeClass, dtypeWeightScale, transA, transB, sync) \
    do { \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, gmmSwigluBaseParams, gmmSwigluBaseParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, gmmSwiglu, gmmSwiglu_, tiling); \
        using xType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>; \
        using weightType = MatmulType<TPosition::GM, wFormat, int4b_t, false>; \
        using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>; \
        using matmulType = MMImplType<xType, weightType, yType>; \
        matmulType::MT mm; \
        if ASCEND_IS_AIC { \
            mm.SetSubBlockIdx(0); \
            mm.Init(&mmTilingData_); \
        } \
        computeClass<matmulType> op(mm, &gmmSwigluBaseParams_, &gmmSwiglu_, &tPipe); \
        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, y, yScale, user1); \
\
        op.Process(); \
    } while (0)
```

Both operands are `int4b_t` — confirming our finding that `Mmad` on 2201 needs both sides INT4 —
and the destination is `half`, against `int32_t` on the A8W8 path three lines above it.

The kernel entry, with the extra input the MSD path needs and the tiling keys:

```cpp
extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                                  GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
                                                                  GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR user1 = GetUserWorkspace(workspace);
#if defined(GMM_SWIGLU_QUANT_A8W8)
    if (TILING_KEY_IS(0)) { // antiquant msd
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMSwigluCompute, ...);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMSwigluSplitWorkSpaceCompute, ...);
    }
#elif defined(GMM_SWIGLU_QUANT_A8W4_MSD)
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP_A8W4_MSD(GMMSwigluQuantPipelineSchedule, ...);
    }
#endif
}
```

`weightAssistanceMatrix` is a **declared kernel argument**, i.e. the `8*sum(w)` correction is a
first-class operator input, not something computed on device. `KERNEL_TYPE_MIX_AIC_1_2`: one cube,
two vector, which is how pre (AIV) / mid (AIC) / post (AIV) share one launch.

### 2.1 `..._msd_pre.h` — the split, instruction by instruction

The declared tensor set already tells the story — `int4b_t` for both halves, `half` as the
working type, and a 16-bit mask constant:

```cpp
#define BUFFER_NUM_A8W4_PRE 1
constexpr int TWO = 2;
constexpr int EIGHT = 8;
constexpr size_t LEN_128 = 128; // 16bit operator
constexpr int DATA_BLOCK_SIZE_32 = 32;
```

```cpp
    LocalTensor<int8_t> xTensor;
    LocalTensor<half> xHighHalfTensor;
    LocalTensor<float> xHighFloatTensor;
    LocalTensor<half> xLowHalfTensor;
    LocalTensor<half> xLowHalfTensor2;
    LocalTensor<int4b_t> xHighI4Tensor;
    LocalTensor<int4b_t> xLowI4Tensor;
    LocalTensor<int16_t> xLowI16Tensor;
    LocalTensor<int64_t> groupListTensor;
    LocalTensor<float> groupListFTensor;
    LocalTensor<float> xRowSumTensor;
```

The UB budget, verbatim, with the per-K-element cost spelled out:

```cpp
        vK = gmmSwigluBaseParams->K;
        groupNum = static_cast<uint32_t>(gmmSwigluBaseParams->groupNum);
        // M * K * 7B (1B + 0.5B + 0.5B + 2B + 4B) <= UBsize - 256B
        blockDim = GetBlockNum() * GetTaskRation();
```

```cpp
__aicore__ inline void GMMA8W4PreProcess::CustomInitBuffer(TPipe* pipe)
{
    pipe->InitBuffer(vecInQueueX, BUFFER_NUM_A8W4_PRE, vK * sizeof(int8_t));    // K * 1B
    pipe->InitBuffer(vecOutQueueA1, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // K * 0.5B
    pipe->InitBuffer(vecOutQueueA2, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // K * 0.5B
    pipe->InitBuffer(vecOutQueueA3, BUFFER_NUM_A8W4_PRE, vK * sizeof(half));    // K * 2B
    // xLowHalfTensor, xLowHalfTensor2 and xHighFloatTensor share the same buffer
    pipe->InitBuffer(tempBuff, vK * sizeof(float)); // K * 4B
    constexpr int BUFFER_SIZE_256B = 128 * sizeof(int16_t);
    pipe->InitBuffer(vecOutQueue0F, BUFFER_NUM_A8W4_PRE, BUFFER_SIZE_256B); // 256B
}
```

The mask and the magic constant:

```cpp
        Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK); // get rid of high 4 bits in every int8
        PipeBarrier<PIPE_V>();
        const size_t LEN_VK = (vK / 2) / 128;
        const size_t LAST_LEN_VK = (vK % 256) / 2;
        const half ONE_SIXTEENTH = static_cast<half>(0.0625f);
        // groupList仅支持count
```

And the loop body — this is the whole decomposition, and the only arithmetic in it:

```cpp
        for (uint32_t xloop = 0; xloop < curCoreTaskNum; xloop++) {
            uint64_t relStartAddr = (xloop + curCoreStartOffset) * vK;
            uint64_t absStartAddr = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize * vK + relStartAddr;
            // 高四位处理开始
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
            DataCopy(xTensor, xGM[absStartAddr], vK);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 3
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0); // 3
            Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, vK);
            PipeBarrier<PIPE_V>();
            Muls(xHighHalfTensor, xHighHalfTensor, ONE_SIXTEENTH, vK);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2
            Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, vK);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 4
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0); // 4
            DataCopy(yGm[relStartAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), vK / 2);
            // 高四位处理结束

            // 低四位处理开始
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2
            And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), xLowI16Tensor, LEN_128,
                LEN_VK, {1, 1, 1, 8, 8, 0});
            if (LAST_LEN_VK > 0) {
                And(xLowHalfTensor[LEN_VK * LEN_128].ReinterpretCast<int16_t>(),
                    xTensor[LEN_VK * LEN_128 * TWO].ReinterpretCast<int16_t>(), xLowI16Tensor, LAST_LEN_VK, 1,
                    {1, 1, 1, 8, 8, 0});
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
            Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(),
                 AscendC::RoundMode::CAST_NONE, vK);
            PipeBarrier<PIPE_V>();
            const half MINUS_EIGHT = static_cast<half>(-8);
            Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, vK);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
            Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, vK);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);  // 5
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1); // 5
            DataCopy(yGm[relStartAddr + vK / TWO], xLowI4Tensor.ReinterpretCast<int8_t>(), vK / TWO);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
            // 低四位处理结束
        }
```

Decoded, with nothing added:

| | high nibble | low nibble |
| --- | --- | --- |
| extract | `Cast(int8 -> half)`, `Muls(0.0625)`, `Cast(half -> int4b_t, CAST_FLOOR)` | `And(int16, 0x0F0F)`, `Cast(int8 -> half)`, `Adds(-8)`, `Cast(half -> int4b_t, CAST_NONE)` |
| value | `h = floor(x / 16)`, in `[-8, 7]` | `l = (x & 0x0F) - 8`, in `[-8, 7]` |
| GM byte offset within row `r` | `r*vK` | `r*vK + vK/2` |

So the identity the kernel relies on is

```
x_int8 = 16*h + (l + 8)       h = floor(x/16) in [-8,7],  l = (x & 0xF) - 8 in [-8,7]
```

which is exact for every `x` in `[-128, 127]`, including negatives: `floor(x/16)` is an arithmetic
shift, and `x & 0xF` is the unbiased low nibble. Note the split is done **in fp16 arithmetic**
(`Muls` by `0.0625` then `CAST_FLOOR`), not with a shift instruction — every int8 is exactly
representable in fp16, so this particular step is still exact.

### 2.2 `..._msd_mid.h` — one matmul at 2M rows, not two matmuls

This is the line that settles the stacking question. `Process`, verbatim:

```cpp
            int32_t splitValue = (currSplitValue - prevSplitValue) * 2; // 2: int8 has been split in 2 int4
            prevSplitValue = currSplitValue;

            SetMNConfig(splitValue, mnConfig);
            if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                continue;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
```

`mnConfig.m` is the per-expert row count **doubled**, and `K`/`N` are untouched. One
`SetOrgShape(2M, N, K)`. The A-matrix offset advances by the doubled M:

```cpp
template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::UpdateMnConfig(MNConfig& mnConfig)
{
    if constexpr (bT::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset +=
            static_cast<int64_t>(AlignUp<16>(mnConfig.k)) * AlignUp<32>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += static_cast<int64_t>(mnConfig.k) * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += static_cast<int64_t>(mnConfig.m) * mnConfig.k;
    mnConfig.yBaseOffset += static_cast<int64_t>(mnConfig.m) * mnConfig.n;
}
```

The inner compute, including the per-quant-group K loop and the accumulate flag:

```cpp
    mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize); // 8, 256, 512 --> 514us
    GlobalTensor<int4b_t> weightSlice;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
    for (uint32_t loopK = 0; loopK < gmmBaseParams->quantGroupNum; loopK++) {
        mm.SetTensorA(
            xGM[mnConfig.xBaseOffset + mnConfig.mIdx * mnConfig.k * mnConfig.singleM + loopK * quantGroupSize]);
        if constexpr (mmType::BT::format == CubeFormat::NZ) {
            weightSlice = weightGM[weightOffset + loopK * quantGroupSize * 64];
        } else {
            weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmBaseParams->N];
        }
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice);
        mm.SetQuantVector(weightScaleGM[groupIdx * gmmBaseParams->N * gmmBaseParams->quantGroupNum +
                                        loopK * gmmBaseParams->N + tailN]);
        mm.Iterate();
        mm.GetTensorC(mmOutGM[mnConfig.workSpaceOffset], loopK == 0 ? 0 : 1);
    }
```

Note `quantGroupSize = gmmBaseParams->K / gmmBaseParams->quantGroupNum; // 约束为整除关系`, the
`uint64_t` weight-scale tensor fed to `SetQuantVector` (the packed deq-scale format), and the
in-source tiling datum `// 8, 256, 512 --> 514us` — `baseM=8`, `baseN=256`, `quantGroupSize=512`.

### 2.3 `..._msd_post.h` — `16*high + low`, then the auxiliary matrix

The constant, and the recombination, verbatim:

```cpp
#define DOUBLE_BUFFER 2
constexpr float DEFAULT_MUL_SCALE = 16.0f;
```

```cpp
    mmLocal_fp16 = mmOutQueue.DeQue<half>();
    mmLocal_fp32 = mmLocal_fp16.ReinterpretCast<float>();
    const int64_t processNum = 2 * vecConfig.innerLoopNum * gmmSwiglu->tokenLen;
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(processNum * sizeof(half)), 0, 0, 0};
    DataCopyPadExtParams<half> padParams_0{false, 0, 0, 0};
    DataCopyPad(mmLocal_fp16[processNum], mmOutGM[vecConfig.curOffset * DOUBLE_ROW], copyParams_0, padParams_0);
    GmmsqSetWaitFlag<HardEvent::MTE2_V>();

    mmOutQueue.EnQue(mmLocal_fp16);
    mmLocal_fp16 = mmOutQueue.DeQue<half>();
    // 1. fp16 -> fp32
    Cast(mmLocal_fp32, mmLocal_fp16[processNum], RoundMode::CAST_NONE, processNum);
    PipeBarrier<PIPE_V>();
    // 2. high_4bit * 16 + low_4bit
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++) {
        Muls(mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen], mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen],
             DEFAULT_MUL_SCALE, gmmSwiglu->tokenLen);
        PipeBarrier<PIPE_V>();
        Add(mmLocal_fp32[i * gmmSwiglu->tokenLen], mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen],
            mmLocal_fp32[(DOUBLE_ROW * i + 1) * gmmSwiglu->tokenLen], gmmSwiglu->tokenLen);
        PipeBarrier<PIPE_V>();
        vecConfig.curIdx++;
    }
```

With `constexpr int64_t DOUBLE_ROW = 2;` from the utils header, row `2i` is the high half and row
`2i+1` is the low half of logical row `i`. **The M-stacking is interleaved, not blocked**:
`[h_0, l_0, h_1, l_1, ...]`, with stride `2*tokenLen` between logical rows.

The four post stages, in the order they run:

```cpp
__aicore__ inline void GMMA8W4PostProcess::VectorCompute(uint32_t loopIdx, VecConfig& vecConfig,
                                                         WorkSpaceSplitConfig& workspaceSplitConfig)
{
    // 1.辅助矩阵加回
    MergeAuxiliaryMatrix(loopIdx, vecConfig);
    // 2.perToken反量化
    MulPertokenScale(loopIdx, vecConfig, workspaceSplitConfig);
    // 3.Swiglu
    Swiglu(loopIdx, vecConfig);
    // 4.Quant
    Quant(loopIdx, vecConfig);
}
```

The `8*sum(w)` correction is an `(E, N)` fp32 tensor, added before the per-token scale:

```cpp
        weightAuxiliaryMatrixGM.SetGlobalBuffer((__gm__ float*)gmAddrParams.weightAuxiliaryMatrixGM); // E, N
```

```cpp
__aicore__ inline void GMMA8W4PostProcess::MergeAuxiliaryMatrix(uint32_t loopIdx, VecConfig& vecConfig)
{
    // perChanelScale * perTokenScale
    mmLocal_fp32 = mmOutQueue.DeQue<float>();
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    Add(mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen],
        weightAuxiliaryMatrixLocal, gmmSwiglu->tokenLen);
    vecConfig.nextUpadteInterVal--;
    PipeBarrier<PIPE_V>();
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

__aicore__ inline void GMMA8W4PostProcess::MulPertokenScale(uint32_t loopIdx, VecConfig& vecConfig,
                                                            WorkSpaceSplitConfig& workspaceSplitConfig)
{
    float scale = perTokenScaleGM.GetValue(loopIdx + workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx);
    GmmsqSetWaitFlag<HardEvent::S_V>();
    Muls(mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], scale,
         gmmSwiglu->tokenLen);
    PipeBarrier<PIPE_V>();
}
```

The auxiliary matrix is reloaded only on expert change, by walking `groupList`:

```cpp
__aicore__ inline void GMMA8W4PostProcess::UpdateAuxiliaryMatrix(uint32_t loopIdx, VecConfig& vecConfig)
{
    // 更新weightAuxiliaryMatrix
    if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
        int64_t loop = gmmSwiglu->groupListLen - vecConfig.curGroupIdx - 1;
        while (loop--) {
            int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            vecConfig.curGroupIdx++;
            int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            if (nextTemp != curTemp) {
                vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                break;
            }
        }
        ...
```

And the SwiGLU + per-token-max requant that follows, for completeness (`SWIGLU_REDUCE_FACTOR = 2`,
`BISECT = 2`, `QUANT_SCALE_INT8 = 127.0f`, all from the utils header):

```cpp
__aicore__ inline void GMMA8W4PostProcess::Swiglu(uint32_t loopIdx, VecConfig& vecConfig)
{
    // 高阶API swiglu
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local =
        mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen + gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR];
    LocalTensor<float> src1Local = mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen];

    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
```

```cpp
    ReduceMaxTemplate(reduceResLocal, workLocal, mmLocal_fp32[preOffset + gmmSwiglu->tokenLen / BISECT], reduceTmpLocal,
                      static_cast<uint32_t>(halfTokenLen));
    GmmsqSetWaitFlag<HardEvent::V_S>();
    float quantScale = reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
```

The UB arithmetic for post, verbatim, which is the budget to beat:

```cpp
    // 2 * row * n * sizeof(float) + row * n / 2 * sizeof(int8) + alignUp<row, 8> * sizeof(float) + n * sizeof(float) +
    // n / 2 *sizeof(float) + 64 < 191 * 1024
```

### 2.4 The contradiction: §1 says exact, the source says `half`

Lay the two against each other.

§1.5: 「这样可以保证拆分计算后，数值与直接INT8×INT4乘法结果一致」 — numerically identical to a
direct INT8xINT4 multiply. §1 heading sentence: 「在保证结果等价于原始INT8×INT4计算结果的同时」.

§2.0: `using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;`
§2.3: `mmOutGM1.SetGlobalBuffer((__gm__ half*)...)`, `Cast(mmLocal_fp32, mmLocal_fp16[...])`,
`Muls(..., DEFAULT_MUL_SCALE /* 16.0f */)`.

The integer identity `x = 16h + (l+8)` is exact; the *implementation* is not, in three places,
none of which is visible from the article:

1. **The two partial products are rounded to fp16 before they are combined.** `SetQuantVector`
   applies the weight scale in Fixpipe and the result leaves the cube as `half` — 10 bits of
   mantissa. The `*16` in post then multiplies the high partial product's fp16 rounding error by
   16 as well. An error of 1 ulp on the high term becomes 16 ulp of the result. The A8W8 path two
   lines above in the same file keeps `int32_t` and has none of this.
2. **Accumulation across quant groups is an fp16 atomic add in GM.**
   `mm.GetTensorC(mmOutGM[...], loopK == 0 ? 0 : 1)` — the second argument is `enAtomic`, so
   groups `1..quantGroupNum-1` accumulate atomically into a `half` buffer. Within one group
   (`SetSingleShape(curSingleM, curSingleN, quantGroupSize)`, K = 512) the MAC is exact int32;
   across groups it is fp16.
3. **The auxiliary matrix is fp32 added to a value reconstructed from fp16 terms**, so it cannot
   recover the lost bits — it only corrects the mean.

Consequence for us, stated as a hypothesis to measure rather than a claim: **we can be strictly
more accurate than CANN's MSD and no slower**, by keeping the cube output `int32_t` (as the A8W8
macro already does), doing `16*h + l` in int32 — exact, since `|16h+l| <= 16*8*K + 8*K` fits int32
for any K we will ever use — and applying the weight scale, the `8*sum(w)` term and the per-token
scale once, in fp32, in the vector unit. The cost is `int32` instead of `half` out of Fixpipe, i.e.
2x the cube→GM traffic on the mmOut workspace. With `quantGroupNum = 1` (which §1.2 licenses) item
2 disappears entirely and there is a single `Iterate`.

### Relevance

- **Our M-stacking is the vendor's own choice**, confirmed by
  `int32_t splitValue = (currSplitValue - prevSplitValue) * 2; // 2: int8 has been split in 2 int4`
  and a single `SetOrgShape(2M, N, K)`. There is no two-matmul variant in this operator. One
  refinement to copy: CANN interleaves (`[h_0, l_0, h_1, l_1, ...]`, `DOUBLE_ROW = 2`) rather than
  concatenating all highs then all lows. Interleaving keeps the two halves of one token in the same
  `baseM` tile (`baseM = 8` per the in-source comment), so the post stage reads one contiguous
  `2*tokenLen` run per token and the recombine is a `Muls`+`Add` with no gather. A blocked layout
  would put the pair `M` rows apart and, at `baseM = 8`, in different cube tiles.
- **`8*sum(w)` is right, and its sign and position are pinned.** CANN subtracts 8 from the
  activation's low nibble (`Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, vK)`), so the
  correction is `+8*sum_k(w_k * w_scale_k)` per `(expert, n)`, applied **after** the `16h+l`
  recombine and **before** the per-token scale. That ordering is forced: the term carries the
  weight scale but not the activation scale. Our fold is the same quantity; check the order.
- **Name to use in our own code and when asking the vendor**: input
  `weightAssistanceMatrix` / member `weightAuxiliaryMatrixGM` / 「权重辅助矩阵」, `(E, N)` fp32.
- **Tiling datum to start from**: `// 8, 256, 512 --> 514us` — `baseM=8`, `baseN=256`,
  `quantGroupSize=512`. `baseM = 8` means 4 logical tokens per tile after doubling, which at
  batch=1 decode is already the right order for us.
- **The cheapest accuracy win available** is the one in §2.4: keep int32 out of the cube. CANN's
  own A8W8 macro in the same file does it; the MSD macro does not, and §1 does not admit it.

---

## 3. The same decomposition in plain `GroupedMatmul` — `gmm/grouped_matmul/`

Repository: `https://gitcode.com/cann/ops-transformer`, paths
`gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd{,_pre,_pre_nz,_new}.h` and
`gmm/grouped_matmul/op_kernel/grouped_matmul.cpp`. Same clone and commits as §2.

This is the operator the §1 article is actually about (`aclnnGroupedMatmulV4`). It carries the
*second*, better-engineered implementation of the same decomposition, plus the asymmetric-weight
path that §2's SwiGLU variant does not have. Read this one, not §2, if you are writing the kernel.

### 3.1 The type block, and the line that names the correction

`grouped_matmul_antiquant_a8w4_msd.h`, verbatim:

```cpp
#ifdef GMM_ANTI_QUANT_A8W4_MSD
namespace GROUPED_MATMUL {
using namespace matmul;
using namespace AscendC;

using DTYPE_PERTOKEN_SCALE_A8W4MSD = float;
using DTYPE_BIAS_A8W4MSD = float;
using DTYPE_OFFSET_A8W4MSD = float;

#ifdef GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
using DTYPE_Y_A8W4MSD = bfloat16_t;
#else
using DTYPE_Y_A8W4MSD = half;
#endif

using DTYPE_X_DEV_A8W4MSD = int4b_t;
using DTYPE_WEIGHT_DEV_A8W4MSD = int4b_t;
using DTYPE_SCALE_DEV_A8W4MSD = uint64_t;

constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t MM_BASE_BLOCK_OFFSET = 32768; // baseM * baseN = 128 * 256
```

And the member declarations — the comment on `biasGm` is the single most useful line in this file:

```cpp
    GlobalTensor<DTYPE_BIAS_A8W4MSD> biasGm; // for 8 * weight
    GlobalTensor<DTYPE_SCALE_DEV_A8W4MSD> scaleGm;
```

```cpp
    GlobalTensor<DTYPE_OFFSET_A8W4MSD> offsetGm;
    GlobalTensor<float> xRowSumGm;
```

`biasGm` is bound from the operator's **public `bias` argument**:

```cpp
    biasGm.SetGlobalBuffer(GetTensorAddr<DTYPE_BIAS_A8W4MSD>(0, bias));
    scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE_DEV_A8W4MSD>(0, scale));
```

```cpp
        biasGm.SetGlobalBuffer(GetTensorAddr<DTYPE_BIAS_A8W4MSD>(groupIdx, biasTensorPtr));
```

So `aclnnGroupedMatmulV4`'s `biasOptional` is, on the A8W4-MSD tiling path, not a model bias at
all — it is the `8*sum(w)` MSD correction, fp32, one vector of length N per expert. That matches
§1.3's 「通常将这部分多余项放入bias参数中」 exactly, and it means the operator has **no spare slot
for a real bias** on this path.

`DataCopyScale` reads `biasGm` under a variable called `scale`, which is confusing on purpose
(「bias queue」 in the comment gives it away):

```cpp
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM, mmTilingData->baseN * sizeof(DTYPE_BIAS_A8W4MSD)); // bias queue
```

```cpp
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN,
                                                                uint64_t scaleOffset)
{
    // GM拷贝scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<DTYPE_BIAS_A8W4MSD>();
    DataCopyPad(scaleLocal, biasGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);
    scaleInUb = scaleInQueue.DeQue<float>();
}
```

### 3.2 The MatmulConfig — `isPartialOutput`, and the `int32_t` bias slot that is left unused

`grouped_matmul.cpp`, verbatim:

```cpp
#if defined(GMM_ANTI_QUANT_A8W4_MSD)
constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();
constexpr auto GetMmCFG()
{
    auto CFG = CFG_MDL;
    CFG.isPartialOutput = true;
    return CFG;
}
constexpr MatmulConfig A8W4_GMM_CFG_MDL_NEW = GetMmCFG();
#endif
```

```cpp
#define GMM_CV_SPLIT_IMP_A8W4_MSD(computeClass, cfg) \
    do { \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling); \
        if ASCEND_IS_AIV { \
            if (gmmBaseParams_.isA8W4MSDPreNZ == 1) { \
                GMMA8W4PreProcessNZ op1; \
                op1.Init(x, user1, groupList, user1, gmmBaseParams_, &tPipe); \
                op1.Process(); \
            } else { \
                GMMA8W4PreProcess op1; \
                op1.Init(x, x, groupList, user1, gmmBaseParams_, &tPipe); \
                op1.Process(); \
            } \
            tPipe.Reset(); \
            tPipe.Destroy(); \
            tPipe.Init(); \
        } \
        if (gmmBaseParams_.isA8W4MSDPreNZ == 1) { \
            using aT = MatmulType<TPosition::GM, CubeFormat::NZ, DTYPE_X_DEV_A8W4MSD, false>; \
            using bT = MatmulType<TPosition::GM, wFormat, int4b_t, false>; \
            using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>; \
            using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>; \
            using matmulType = MMImplType<aT, bT, cT, biasT, cfg>; \
            ...
        } else { \
            using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4MSD, false>; \
            using bT = MatmulType<TPosition::GM, wFormat, DTYPE_WEIGHT_DEV_A8W4MSD, false>; \
            using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>; \
            using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>; \
            using matmulType = MMImplType<aT, bT, cT, biasT, cfg>; \
            ...
```

Note what this shows: the `Matmul` object **does** have an `int32_t` bias type parameter
(`biasT`), i.e. the hardware/template path for an int32 bias added inside the cube exists — and
the MSD implementation does not use it. The `8*sum(w)` correction is instead applied in fp32 in
the vector unit (§3.4). That is a choice, not a constraint, and it is the choice that costs
accuracy (§2.4), because by the time the correction is applied the partial products have already
been through `half`.

The AIV pre-stage runs inside the same kernel, then tears down and rebuilds the pipe
(`tPipe.Reset(); tPipe.Destroy(); tPipe.Init();`) before the cube stage — a reusable trick for
reclaiming the whole UB between phases of a megakernel.

### 3.3 `*2` again, and the authoritative statement of the A layout

```cpp
        mnConfig.m = static_cast<uint32_t>(m) * 2; // 2: int8 has been split in 2 int4
```

```cpp
    A8W4preM = AlignUp<8>(xRowSumCount) * 2;
```

```cpp
    uint32_t mGlobalOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * curCubeSingleM; // 2: 2 lines int4 to 1 line int8
```

```cpp
    uint64_t perTokenScaleOffset =
        mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; // 2: m方向两行合并为1行
```

Two int4 rows per int8 row, 「m方向两行合并为1行」. The pre writes them adjacent, exactly as in §2.1:

```cpp
        DataCopy(yGm[startAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), vK / 2); // 2: size int8 -> int4
        ...
        DataCopy(yGm[startAddr + vK / TWO], xLowI4Tensor.ReinterpretCast<int8_t>(), vK / TWO);
```

with `uint64_t startAddr = xloop * vK;` and `yGm` typed `int8_t` — so logical row `r` owns bytes
`[r*vK, (r+1)*vK)`, which as `int4b_t` rows of `K` elements is rows `2r` (high) and `2r+1` (low).
**Interleaved M-stacking, confirmed in both operators.**

### 3.4 The de-interleave: a strided `DataCopyPad`, then one `Muls` and one `Add` per tile

This is the part worth copying. The cube output tile has `2*curVecBaseM` rows, interleaved. The
vector stage reads it as two strided planes:

```cpp
            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            if constexpr (mmType::BT::format == CubeFormat::ND) {
                DataCopyPad2DA8W4ND(mmOutLocal, mmOutGm[mmOutOffset + offsetM * 2 * curVecBaseN], curVecBaseM,
                                    curVecBaseN, curVecBaseN * 2); // 2: 2 lines int4 to 1 line int8
            } else {
                DataCopyPad2DA8W4(mmOutLocal, mmOutGm[mmOutOffset + offsetM * 2 * curVecBaseN], curVecBaseM,
                                  curVecBaseN, curVecBaseN * 2); // 2: 2 lines int4 to 1 line int8
            }
            uint32_t targetAddr;
            if constexpr (mmType::BT::format == CubeFormat::ND) {
                alignBaseN = (alignBaseN + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
            }
            targetAddr = curVecBaseM * alignBaseN;
            uint64_t lowBitAddr = mmOutOffset + (offsetM * 2 + 1) * curVecBaseN;
            if constexpr (mmType::BT::format == CubeFormat::ND) {
                DataCopyPad2DA8W4ND(mmOutLocal[targetAddr], mmOutGm[lowBitAddr], curVecBaseM, curVecBaseN,
                                    curVecBaseN * 2); // 2: 2 lines int4 to 1 line int8
            } else {
                DataCopyPad2DA8W4(mmOutLocal[targetAddr], mmOutGm[lowBitAddr], curVecBaseM, curVecBaseN,
                                  curVecBaseN * 2); // 2: 2 lines int4 to 1 line int8
            }
            vecInQueue.EnQue(mmOutLocal);
            ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
```

with the two copy helpers, verbatim — note `srcStride` is `(srcDim0 - dim0)` with
`srcDim0 = 2*curVecBaseN` and `dim0 = curVecBaseN`, i.e. skip one row each time, and note the
reason the ND variant pads each row to an even number of 32B blocks:

```cpp
template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1,
                                         uint32_t dim0, uint32_t srcDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = Ceil(dim0 * sizeof(T), 32) % 2;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4ND(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1,
                                           uint32_t dim0, uint32_t srcDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    params.dstStride = 0;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
    return;
}
```

After which the recombine is two whole-tile instructions, not a per-row loop:

```cpp
    uint32_t computeSize = curVecBaseM * alignBaseN;
    LocalTensor<cT::T> mmOutInUb = vecInQueue.DeQue<cT::T>();
    uint32_t castSize;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        castSize = computeSize + (computeSize + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
    } else {
        castSize = computeSize * 2;
    }
    Cast(buffer1, mmOutInUb, RoundMode::CAST_NONE, castSize);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    const float RIGHT_MOVE = 16.0f; // right move int4 to int8
    Muls(buffer2, buffer1, RIGHT_MOVE, computeSize);
    PipeBarrier<PIPE_V>();
    uint32_t addStartAddr;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        addStartAddr = (computeSize + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
    } else {
        addStartAddr = computeSize;
    }
    Add(buffer3, buffer1[addStartAddr], buffer2, computeSize);
    PipeBarrier<PIPE_V>();
```

Then the `8*sum(w)` add, broadcast along M by `BinaryRepeatParams` with a zero src1 block stride
(`param(1, 1, 1, blkStride, blkStride, 0)`) — no materialised broadcast needed:

```cpp
    uint32_t loop = alignBaseN / 64; // 256B为64个float，alignBaseN需约束为64倍数
    uint8_t blkStride = static_cast<uint8_t>(alignBaseN * sizeof(float) / 32); // 32: 单位32B
    BinaryRepeatParams param(1, 1, 1, blkStride, blkStride, 0);
    uint64_t mask = 64;
    uint64_t last = alignBaseN % 64;
    for (uint32_t i = 0; i < loop; i++) {
        uint32_t offset = i * 64; // 每次64个元素
        Add(buffer2[offset], buffer3[offset], scaleInUb[offset], mask, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    if (unlikely(last > 0)) {
        uint32_t offset = loop * 64;
        Add(buffer2[offset], buffer3[offset], scaleInUb[offset], last, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
```

### 3.5 The asymmetric-weight path — `withOffset`, `xRowSum`, and `offset`

`constexpr uint32_t WITH_OFFSET = 1;` in the pre header. When set, the pre stage additionally
reduces the **original int8 row** (before the `*0.0625`) to a single fp32 and writes it to
workspace:

```cpp
        Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, vK);
        PipeBarrier<PIPE_V>();

        if (withOffset == WITH_OFFSET) {
            Cast(xHighFloatTensor, xHighHalfTensor, AscendC::RoundMode::CAST_NONE, vK);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            ReduceSum(xRowSumTensor, xHighFloatTensor, workTensor, vK);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

            DataCopyExtParams xRowSumParams{1, static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
            DataCopyPad(xRowSumGm[xloop], xRowSumTensor, xRowSumParams);
            AscendC::PipeBarrier<PIPE_V>();
        }

        Muls(xHighHalfTensor, xHighHalfTensor, one_eight, vK);
```

i.e. `xRowSum[m] = sum_k x_int8[m,k]`, fp32, one value per token. The post stage broadcasts it
along N and multiplies by the per-column `offset`:

```cpp
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyAndBrcbOfRowSum(MNConfig& mnConfig, uint32_t curBaseM,
                                                                          uint32_t alignBaseN, uint32_t offsetM)
{
    const uint64_t xRowSumOffset =
        mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; // 2: m方向两行合并为1行
    ...
    const uint32_t xRowSumBroadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t xRowSumBroadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(buffer3, xRowSumLocal, xRowSumBroadCastDst, xRowSumBroadCastSrc, buffer6);
    xRowSumInQueue.FreeTensor(xRowSumLocal);
}
```

```cpp
    if (withOffset == WITH_OFFSET) {
        DataCopyAndBrcbOfRowSum(mnConfig, curVecBaseM, alignBaseN, offsetM);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < loop; i++) {
            uint32_t offset = i * 64; // 每次64个元素
            Mul(buffer4[offset], buffer3[offset], offsetInUb[offset], mask, curVecBaseM, param);
        }
        PipeBarrier<PIPE_V>();
        if (unlikely(last > 0)) {
            uint32_t offset = loop * 64;
            Mul(buffer4[offset], buffer3[offset], offsetInUb[offset], last, curVecBaseM, param);
        }
        PipeBarrier<PIPE_V>();

        Add(buffer5, buffer4, buffer2, computeSize);
        PipeBarrier<PIPE_V>();
    }

    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);
    PipeBarrier<PIPE_V>();

    LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
    if (withOffset == WITH_OFFSET) {
        Mul(buffer4, buffer5, buffer3, computeSize);
    } else {
        Mul(buffer4, buffer2, buffer3, computeSize);
    }
    PipeBarrier<PIPE_V>();
    Cast(yLocalInUb, buffer4, RoundMode::CAST_RINT, computeSize);
    vecOutQueue.EnQue(yLocalInUb);
```

### 3.6 The complete expression, assembled from §3.4 and §3.5

Reading the buffer traffic end to end, with `h`/`l` the two int4 halves, `s` the fused
`total_scale = perchannel_scale * pergroup_scale` applied in Fixpipe by `SetQuantVector`,
`b` = the `bias` input (`// for 8 * weight`), `z` = the `offset` input, and `t` = per-token scale:

```
P_h[m,n] = sum_k h[m,k] * w[k,n] * s[n]          (fp16 out of the cube)
P_l[m,n] = sum_k l[m,k] * w[k,n] * s[n]          (fp16 out of the cube)

y[m,n] = ( 16*P_h[m,n] + P_l[m,n]  +  b[n]  +  xRowSum[m] * z[n] ) * t[m]
```

evaluated in fp32 in the vector unit, then `Cast(..., CAST_RINT)` to `half` or `bfloat16_t`.
The `withOffset == 0` path drops the `xRowSum*z` term.

### Relevance

- **`bias` is the `8*sum(w)` slot, and the comment says so**: `// for 8 * weight`. Our fold is
  the vendor's, named. If we ever call `aclnnGroupedMatmulV4` on the A8W4-MSD path we must put the
  correction in `biasOptional` and we cannot also pass a model bias there.
- **Interleaved M-stacking, de-interleaved on MTE2.** Both CANN operators write
  `[h_0, l_0, h_1, l_1, ...]` and read it back with a `DataCopyPad` whose `srcStride` skips one
  tile row (`srcDim0 = 2*baseN`, `dim0 = baseN`). That turns the recombine into exactly two
  whole-tile vector instructions — `Muls(buffer2, buffer1, 16.0f, computeSize)` and
  `Add(buffer3, buffer1[addStartAddr], buffer2, computeSize)` — with no per-row loop and no
  gather. §2's SwiGLU variant loops per row; this one does not. Copy this one.
- **The asymmetric-weight correction is `xRowSum[m] * offset[n]`, and `xRowSum` is the sum of the
  *original* int8 row**, taken before the `*0.0625`. If our weights are symmetric
  (zero point 0, which is what msmodelslim's W4A8 recipe does per §1.1's 「所有量化/反量化操作均在
  整数域内完成」) we skip the whole `withOffset` branch and the extra `ReduceSum` per row in pre.
  That is one `ReduceSum` + one `BroadCast` + one `Mul` + one `Add` per tile saved.
- **A real int32 bias slot exists in the `Matmul` template and MSD does not use it**
  (`using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;`). If we want the
  exact-int32 variant from §2.4, that slot is where the `8*sum(w)` term — which *is* an integer,
  `8 * sum_k w[k,n]` before scaling — belongs. Folding it there makes the pre-scale accumulator
  exactly equal to `sum_k x_int8[m,k] * w[k,n]`, which is the whole point.
- **`MM_BASE_BLOCK_OFFSET = 32768; // baseM * baseN = 128 * 256`** is this operator's tile, against
  §2's `// 8, 256, 512 --> 514us`. Two different tilings for the same decomposition; at batch=1
  decode the `baseM = 8` one is ours.
- **`tPipe.Reset(); tPipe.Destroy(); tPipe.Init();`** between the AIV pre stage and the cube stage,
  inside one kernel. Directly useful for our megakernel: it is how CANN reclaims the whole UB
  across phases without a launch boundary.

---

## 4. The official operator specification — `aclnnGroupedMatmulSwigluQuantV2` and `aclnnGroupedMatmulV4/V5`

These are the published `aclnn` docs, shipped **inside the open-source repo** rather than only on
the portal, which is why they are quotable in full. Same clone as §2/§3.

```
gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md
gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md
gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantWeightNzV2.md
gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantWeightNzV3.md
gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md
gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md
gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md
```

The portal copies of the same pages, for citation:
`https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/aolapi/context/aclnnGroupedMatmulV4.md`
(given as the reference link at the end of §1's article).

### 4.1 The complete MSD formula, verbatim — `aclnnGroupedMatmulSwigluQuantV2.md`

Section header 「MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指INT8数据类型，4指INT4数据类型）」, under
「Atlas A3系列产品、Atlas A2系列产品」. Verbatim, LaTeX and all:

> - **定义**：
>   * **⋅** 表示矩阵乘法。
>   * **⊙** 表示逐元素乘法。
>   * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
>   * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
>   * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
>   * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
> - **输入**：
>   * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
>   * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
>   * $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
>   * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K\_group\_num是在K轴维度上的分组数，N是输出维度。
>   * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
>   * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
> - **输出**：
>   * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
>   * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
> - **计算过程**
>   - 1.根据groupList[i]确定当前分组的token，$i \in [0,Len(groupList)]$。
>     - 分组逻辑与A8W8相同。
>   - 2.生成辅助矩阵（weightAssistMatrix）的计算过程（请注意weightAssistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
>     - 当为per-channel量化（$w\_scale$为2维）：
>
>       $weightAssistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$
>
>     - 当为per-group量化（$w\_scale$为3维）：
>
>       $weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$
>
>       注：$num\_per\_group = K // K\_group\_num$
>
>   - 3.根据分组确定的入参进行如下计算：
>
>     - 3.1.将左矩阵$\mathbb{Z_8}$，转变为高低位两部分的$\mathbb{Z_4}$
>       $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
>       $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
>     - 3.2.做矩阵乘时，开启per-channel或per-group量化
>       per-channel：
>
>       $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$
>
>       $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$
>
>       per-group：
>
>       $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$
>
>       $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$
>
>     - 3.3.将高低位的矩阵乘结果还原为整体的结果
>
>       $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$
>
>       $C_{i,act}, gate_{i} = split(C_{i})$
>
>       $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$
>
>   - 3.量化输出结果
>
>     $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$
>
>     $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

Compare, from the same file, the 「量化场景A8W8」 block — note that its definitions list is
identical **minus `Z_4`**, and that its formula has no intermediate at all:

> - **定义**：
>   * **⋅** 表示矩阵乘法。
>   * **⊙** 表示逐元素乘法。
>   * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
>   * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
>   * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
> ...
>   - 2.根据分组确定的入参进行如下计算：
>
>     $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

Both blocks define `Z_32` and **neither MSD formula ever uses it**: `C_high` and `C_low` are
stated to be in ℝ after the `⊙ w_scale`, which is the doc quietly conceding §2.4 — the per-half
products leave the integer domain before they are combined. §1.6's 「所有主流程均在整数域内完成」
is therefore an overstatement of its own operator.

The `weightAssistMatrix` parameter entry:

> | weightAssistMatrix | 表示计算矩阵乘时的辅助矩阵，公式中的weightAssistMatrix。<br> • 仅A8W4场景生效，其他场景需传空指针。 |

and in V1, which spells out the non-tensorlist signature and the "offline" caveat again:

> * $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
> - 2.生成辅助矩阵（weightAssistMatrix）的计算过程（请注意weightAssistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
>   $weightAssistMatrix_{i} = 8 × w\_scale × Σ_{k=0}^{K-1} weight[:,k,:]$

> | 权重矩阵，对应公式中的W。 <br> • INT32为适配用途，实际1个INT32会被解释为8个INT4数据。<br> • A8W4场景：shape支持3或5维，分别对应数据格式ND、FRACTAL_NZ。 | INT8、INT4、INT32 |
> | 计算矩阵乘时的辅助矩阵，对应公式中的weightAssistMatrix。<br> 仅A8W4场景生效，A8W8场景需传空指针。 |
> | per-channel非对称反量化的偏移，对应公式中的offset。 |
> | 右矩阵的量化因子，对应公式中的w_scale。 <br> • A8W4场景：shape支持2或3维，数据类型支持UINT64。 |

V2 differs from V1 only in plurality:

> - 参数weight, weightScale, weightAssistMatrix的字段类型变为tensorlist，请根据实际情况选择合适的接口。

### 4.2 `aclnnGroupedMatmulV4.md` — A8W4 is classified 伪量化, and its output can never be INT32

The scenario table, verbatim:

> | 场景名 | x | weight | out | 约束说明 | 计算公式 |
> | --------- | --------- | ---------------- | -------- | -------- | -- |
> | 非量化 | FLOAT32 | FLOAT32 | FLOAT32 | ... |
> | 非量化 | BFLOAT16 | BFLOAT16 | BFLOAT16 | ... |
> | 非量化 | FLOAT16 | FLOAT16 | FLOAT16 | ... |
> | 全量化-A8W8 | INT8 | INT8 | BFLOAT16/FLOAT16/INT32/INT8 | [A8W8场景约束](#a8w8场景约束) | [计算公式](#全量化场景) |
> | 全量化-A4W4 | INT4 | INT4 | BFLOAT16/FLOAT16 | [A4W4场景约束](#a4w4场景约束) | [计算公式](#全量化场景) |
> | 伪量化-A8W4 | INT8 | INT4 | BFLOAT16/FLOAT16 | [A8W4场景约束](#a8w4场景约束) | [计算公式](#a8w4伪量化场景) |
> | 伪量化-A16W8 | BFLOAT16/FLOAT16 | INT8 | BFLOAT16/FLOAT16 | ... |
> | 伪量化-A16W4 | BFLOAT16/FLOAT16 | INT4 | BFLOAT16/FLOAT16 | ... |

The four formulas, side by side, verbatim — this is the cleanest statement of the accuracy
difference anywhere in the material:

> - x为INT8，bias为INT32
>
>   $$ y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i $$
>
> - x为INT8，bias为BFLOAT16
>
>   $$ y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i $$
>
> - x为INT4，无bias
>
>   $$ y_i=x_i\times (weight_i * scale_i) * per\_token\_scale_i $$

> <a id="a8w4伪量化场景"></a>
>
> - x为INT8，weight为INT4（仅支持x、weight、y均为单tensor的场景）。其中$bias$为必选参数，是离线计算的辅助结果，且 $bias_i=8\times weight_i  * scale_i$ ，并沿k轴规约。
>
>   $$ y_i=((x_i - 8) \times weight_i * scale_i+bias_i ) * per\_token\_scale_i $$

A8W8 with an INT32 bias puts the bias **inside** the parenthesis, before `scale` — an integer add
in the accumulator. A8W4-MSD puts it **after** `* scale_i` — a float add. Identical to what §3.4's
buffer traffic does.

Note also how the doc writes the split: `(x_i - 8) × weight_i`, as if one shifted matmul. The
split into two is not mentioned in the `GroupedMatmulV4` doc at all; only the SwiGLU variant's doc
(§4.1) spells out `C_high`/`C_low`. Searching the V4 page for "MSD" finds nothing.

### 4.3 `aclnnGroupedMatmulV4.md` — the A8W4 constraints, verbatim

> **数据类型要求**
>
> | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
> | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | null | null | null | FLOAT | INT64 | null | null | null | BFLOAT16 |
> | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | null | null | FLOAT | INT64 | null | null | null | FLOAT16 |
>
> **约束说明**
>
> 除[公共约束](#公共约束)外，A8W4场景其余约束如下：
> - 仅支持GroupType=0（M轴分组），actType=0
> - 当前仅支持x、weight、out均为长度为1的TensorList
> - x不支持转置、weight不支持转置
> - x仅支持2维Tensor，Shape为（M，K）
> - weight默认支持3维Tensor，Shape为（E，K，N）
> - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[E, N]$。
> - 当weight传入数据类型为INT32时，会将每个INT32视为8个INT4。
> - offset为空时
>   - 该场景下仅支持groupListType为1（算子不会检查groupListType的值，会认为groupListType为1），k要求为quantGroupSize的整数倍，且要求k <= 18432。其中quantGroupSize为k方向上pergroup量化长度，当前支持quantGroupSize=256。
>   - 该场景下要求n为8的整数倍。
>   - 该场景下scale为pergroup与perchannel离线融合后的结果，shape要求为$[E, quantGroupNum, N]$，其中$quantGroupNum=k \div quantGroupSize$。
>   - 该场景下，各个专家处理的token数的预期值大于n/4时，即tuningConfigOptional中第一个值大于n/4时，通常会取得更好的性能，此时显存占用会增加$g\times k \times n$字节（其中g为matmul组数即分组数）。
> - offset不为空时
>   - scale为pergroup与perchannel离线融合后的结果，shape要求为$[E, 1, N]$。
>   - 该场景下offsetOptional不为空。非对称量化offsetOptional为计算过程中离线计算辅助结果，即$antiquantOffset \times scale$，shape要求为$[E, 1, N]$，dtype为FLOAT32。
> - tuningConfigOptional数组第二个元素可置1，以开启A8W4场景中weight的特殊格式模板，以优化算子性能(性能优势的shape范围参考:K >= 2048 && N >= 2048)。需要说明的是，该模板要求weight的shape为（E，N，K）,然后再对其进行ND2NZ转换后作为算子输入。

From `aclnnGroupedMatmulV5.md`, the `tuningConfigOptional` triple is spelled out:

> | tuningConfigOptional（aclIntArray *） | 可选输入 | 第一个数代表各个专家处理的 token数的预期值；第二个数代表 A8W4 可选开启weight先转置的NZ格式；第三个数代表允许额外使用的内存空间。 | 兼容历史版本，用户如不使用该参数，不传（即为 nullptr）即可 | INT64 | - | 3 | - |

> - `[0]`：预期各专家处理的token数，tiling 按此优化，适用场景：A8W4/A8W8/A4W4，x/weight/out 单Tensor。
> - `[1]`：是否开启weight亲和格式（先转置再 NZ），适用场景：A8W4。

> - A8W8、A8W4、A4W4：仅支持 groupType=0，且 x 为单 tensor。
> | x不转置；weight可转置（A8W4/A4W4 不可） |

> - 确定性计算：
>   - aclnnGroupedMatmulV4默认确定性实现。

### Relevance

- **Our `8*sum(w)` fold is the published formula, character for character.**
  `weightAssistMatrix = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]`, `(E, N)`, fp32, offline,
  and 「并非算子内部完成」. Our high/low extraction is also the published formula:
  `X_high_4bits = ⌊X/16⌋`, `X_low_4bits = X & 0x0f - 8`. Nothing in our derivation needs changing.
- **`quantGroupSize = 256` is the only value the public operator accepts** (「当前支持
  quantGroupSize=256」), with `k % 256 == 0`, `k <= 18432`, `n % 8 == 0`. The kernel comment in §2.2
  says `512` for the SwiGLU variant's tiling, so the two operators differ — do not assume one
  number. Also note `quantGroupSize = 256` with our K contradicts §1.2's advice to skip per-group
  entirely: the `offset == null` path *forces* per-group. To get one scale per column you must take
  the `offset != null` branch, where 「scale ... shape要求为$[E, 1, N]$」.
- **The tuning hint is pointed the wrong way for us.** 「各个专家处理的token数的预期值大于n/4时
  ... 通常会取得更好的性能」. At batch=1 decode with top-k routing, tokens per expert is 1-8 and
  `n/4` is in the hundreds. We are two orders of magnitude below the threshold where CANN's own
  tiling pays off — which is the quantitative reason to write our own rather than call theirs.
- **`out` for A8W4 is `BFLOAT16` or `FLOAT16` only, never INT32 or INT8**, while A8W8 offers
  `BFLOAT16/FLOAT16/INT32/INT8`. If we want an int32 result out of a W4A8 GEMM, no public CANN
  operator will give it to us. That is the API-level face of §2.4.
- **`weight` may be passed as INT32** — 「实际1个INT32会被解释为8个INT4数据」 — useful if our
  weight loader has no int4 dtype.
- **Weight transpose is forbidden on A8W4/A4W4** (「weight可转置（A8W4/A4W4 不可）」), and the
  affine-format fast path wants `(E, N, K)` then ND2NZ with `tuningConfig[1] = 1`, advantageous at
  `K >= 2048 && N >= 2048`. Our GLM-5.2 expert shapes are in that range.
- **`aclnnGroupedMatmulV4默认确定性实现`** — determinism is claimed by default, which matters when
  we A/B our kernel against it.

---

## 5. The non-grouped W4A8 — `ops-nn/matmul/quant_batch_matmul_v4` and `aclnnQuantMatmulV5`

Repository: `https://gitcode.com/cann/ops-nn`, local clone at commit
`af66e479728bed0889c32b25c66afa4ece44a460` (2026-06-26).

```
matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_msd.h
matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_msd_tiling.{h,cpp}
matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_tiling_key.h
matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md
matmul/quant_batch_matmul_v4/tests/ut/op_host/test_quant_batch_matmul_v4_a8w4_tiling.cpp
matmul/convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md
```

This matters because **not all our W4A8 GEMMs are MoE**. The attention and dense projections need a
plain, non-grouped W4A8 matmul, and CANN ships MSD for that too, under a different parameter name.

### 5.1 `quant_batch_matmul_v4_msd.h` — identical split, M doubled, `half` again

```cpp
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t UB_BLOCK_SIZE = 32;
constexpr uint32_t INT4_SIZE = 2; // sizeof(int4)的倒数
...
constexpr uint32_t AND_ONE_REPEAT_LENGTH = 128; // and operation max length
constexpr uint32_t DOUBLE_ROWS = 2; // in8 to in4, one rows will change two rows
```

```cpp
    Duplicate(xLowI16Tensor_, static_cast<int16_t>(0x0F0F),AND_ONE_REPEAT_LENGTH);
    const half OneEight = static_cast<half>(0.0625);
    const uint32_t halfKSize = kSize_ / DOUBLE_ROWS;
    ...
    const half MINUS_EIGHT = static_cast<half>(-8);
    for(uint32_t xLoop = coreIdx_; xLoop < mSize_; xLoop += blockDim_) {
        uint64_t offset = xLoop * kSize_;
        DataCopyGmToUb(xTensor_, xGlobal_[offset]);
        Cast(xHighHalfTensor_, xTensor_, AscendC::RoundMode::CAST_NONE, kSize_);
        Muls(xHighHalfTensor2_, xHighHalfTensor_, OneEight, kSize_);
        Cast(xHighI4Tensor_, xHighHalfTensor2_, AscendC::RoundMode::CAST_FLOOR, kSize_);
        DataCopyUbToGm(yGlobal_[offset], xHighI4Tensor_.ReinterpretCast<int8_t>());
            And(xLowHalfTensor_.ReinterpretCast<int16_t>(), xTensor_.ReinterpretCast<int16_t>(), xLowI16Tensor_, AND_ONE_REPEAT_LENGTH, repeatTimes, {1,1,1,8,8,0});
        ...
        Cast(xLowHalfTensor2_.ReinterpretCast<half>(), xLowHalfTensor_.ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, kSize_);
        Adds(xLowHalfTensor_, xLowHalfTensor2_, MINUS_EIGHT, kSize_);
        Cast(xLowI4Tensor_, xLowHalfTensor_.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, kSize_);
        DataCopyUbToGm(yGlobal_[offset + halfKSize], xLowI4Tensor_.ReinterpretCast<int8_t>());
```

Bit-for-bit the same split as §2.1 and §3.3, same `0x0F0F`, same `0.0625`, same `-8`, same
adjacent-write layout. The M doubling:

```cpp
        uint32_t curSingleM = baseM_;
        if (mIdx == blockDimM_ - 1) {
            curSingleM = mSize_ * 2 - mIdx * baseM_;
        }
```

and the recombine, with the comment that states the layout outright:

```cpp
    uint32_t computeSize = curVecBaseM * alignBaseN;
    LocalTensor<half> mmOutInUb = vecInQueue_.DeQue<half>();
    uint32_t castSize= computeSize + ops::CeilDiv(computeSize, HALF_ALIGN)*HALF_ALIGN;
    Cast(buffer1_, mmOutInUb, RoundMode::CAST_NONE, castSize);  // 2: in pre process, x has been splited to [2m,n]
    PipeBarrier<PIPE_V>();
    vecInQueue_.FreeTensor(mmOutInUb);
    const float RIGHT_MOVE = 16.0f;         // right move int4 to int8
    Muls(buffer2_, buffer1_, RIGHT_MOVE, computeSize);
    PipeBarrier<PIPE_V>();
    uint32_t addStartAddr = ops::CeilDiv(computeSize, HALF_ALIGN)*HALF_ALIGN;
    Add(buffer3_, buffer1_[addStartAddr], buffer2_, computeSize);
    PipeBarrier<PIPE_V>();
    uint32_t loop = alignBaseN / 64; // 256B为64个float，alignBaseN需约束为64倍数
    uint8_t blkStride = static_cast<uint8_t>(alignBaseN * sizeof(float) / 32);  //32: 单位32B
    BinaryRepeatParams param(1, 1, 1, blkStride, blkStride, 0);
    uint64_t mask = 64;     // float is 32 bit, mask continous mode [1,64], set all params involved in the computation
    for (uint32_t i = 0; i < loop; i++) {
        uint32_t offset = i * 64; // 每次64个元素
        Add(buffer2_[offset], buffer3_[offset], yOffsetInUb_[offset], mask, curVecBaseM, param);
    }
```

`// 2: in pre process, x has been splited to [2m,n]` — the layout is `[2m, n]`, in CANN's own
words. And the correction vector here is `yOffsetInUb_`, loaded from the operator's **`y_offset`**
argument (`GM_ADDR y_offset` in `Init`) — a third name for the same `8*sum(w)` quantity, after
`weightAssistMatrix` (§2/§4) and `bias` (§3).

The cube/vector handshake is a credit window, which is directly relevant to our flag budget:

```cpp
        if (cubeCount >= tilingData_->parallNum) {
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        ...
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
```

and the per-quant-group loop with `enAtomic` and `enSequentialWrite` both set:

```cpp
        if constexpr (quantType == QuantType::K_C) {
            mmObj_.SetSingleShape(curSingleM, curSingleN, kSize_);
            mmObj_.SetTensorA(x1Global_[x1Offset]);
            auto weightSlice = x2Global_[x2Offset];
            if (blockDimM_ == 1) {
                weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            mmObj_.SetTensorB(weightSlice, bTrans);
            mmObj_.SetQuantVector(x2ScaleGlobal_[nIdx * baseN_]);
            mmObj_.Iterate();
            mmObj_.GetTensorC(mmOutGlobal_[workSpaceOffset], 0, true);
        } else {
            mmObj_.SetSingleShape(curSingleM, curSingleN, groupSize_);
            GlobalTensor<wType> weightSlice;
            for (uint32_t loopK = 0; loopK < groupNum_; loopK++) {
                mmObj_.SetTensorA(x1Global_[x1Offset + static_cast<uint64_t>(loopK) * groupSize_]);
                auto weightSlice = x2Global_[x2Offset + static_cast<uint64_t>(loopK) * groupSize_ * nSize_];
                ...
                mmObj_.SetQuantVector(x2ScaleGlobal_[static_cast<uint64_t>(loopK) * nSize_ + x2Offset]);
                mmObj_.Iterate();
                mmObj_.GetTensorC(mmOutGlobal_[workSpaceOffset], loopK == 0 ? 0 : 1, true);
            }
        }
```

with `GlobalTensor<half> mmOutGlobal_;` — `half` once more.

### 5.2 `aclnnQuantMatmulV5.md` — the published formula and the `yOffset` spec

Verbatim, from the 「K-G量化模式」 formula block under 「Atlas A2 ... Atlas A3 训练系列产品/
Atlas A3 推理系列产品」:

> - x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64/INT64，yOffset为FLOAT32：
>
>   $$ out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale $$
>
> - x1，x2为INT4，x1Scale，x2Scale为FLOAT32，x2Offset为FLOAT16，out为FLOAT16/BFLOAT16（pertoken-pergroup非对称量化）：
>
>   $$ out = x1Scale * x2Scale * (x1 @ x2 - x1 @ x2Offset) $$

The `yOffset` constraint, from 「T-C && T-T && K-C && K-T量化场景约束」:

> - yOffset的约束：shape支持1维（n）。为计算过程中离线计算的辅助结果，值要求为8\*x2\*x2Scale，并在第1维累加。

The dtype table for the W4A8 row (K-C && K-T), verbatim, with the A8W8 rows for contrast:

> | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | bias | yOffset | out |
> | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | INT8 | INT32 | FLOAT32 | UINT64/INT64 | null | null | null | FLOAT32 | FLOAT16/BFLOAT16 |
> | INT8 | INT8 | null | UINT64/INT64 | null | null | null/INT32 | null | FLOAT16 |
> | INT8 | INT8 | null | UINT64/INT64 | null/FLOAT32 | null | null/INT32 | null | INT8 |
> | INT8 | INT8 | null/FLOAT32 | FLOAT32/BFLOAT16 | null | null | null/INT32/BFLOAT16/FLOAT32 | null | BFLOAT16 |
> | INT8 | INT8 | FLOAT32 | FLOAT32 | null | null | null/INT32/FLOAT16/FLOAT32 | null | FLOAT16 |
> | INT4/INT32 | INT4/INT32 | null | UINT64/INT64 | null | null | null/INT32 | null | FLOAT16 |
> | INT8 | INT8 | null | FLOAT32/BFLOAT16 | null | null | null/INT32 | null | INT32 |
> | INT8 | INT8 | FLOAT32 | FLOAT32 | null | null | FLOAT32 | null | BFLOAT16 |
> | INT4/INT32 | INT4/INT32 | FLOAT32 | FLOAT32/BFLOAT16 | null | null | null/INT32/BFLOAT16/FLOAT32 | null | BFLOAT16 |
> | INT4/INT32 | INT4/INT32 | FLOAT32 | FLOAT32 | null | null | null/INT32/FLOAT16/FLOAT32 | null | FLOAT16 |

And the K-G shape table:

> | 量化类型 | x1数据类型 | x2数据类型 | x1Scale数据类型 | x2Scale数据类型 | x1 shape | x2 shape | x1Scale shape | x2Scale shape | x2Offset shape | yOffset shape | [gsM，gsN，gsK] |
> | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | K-G量化 | INT8 | INT32 | FLOAT32 | UINT64/INT64 | (m, k) | (k, ceil(n / 8)) | (m, 1) | (ceil(k / 256), n) | null | (n) | [0, 0, 256] |
> | K-G量化 | INT4 | INT4 | FLOAT32 | FLOAT32 | (m, k) | (n, k) | (m, 1) | (ceil(k / 256), n) | (ceil(k / 256), n) | null | [0, 0, 256] |

> - x1的约束：
>   - 当数据类型为INT8时，k需与256对齐，并小于29576。transposeX1为false。
>   - 当数据类型为INT4时，k需与1024对齐。transposeX1为false。

> - x2的约束：
>   - 数据类型为INT32时，每个INT32数据存放8个INT4数据，
>      - 当前仅支持2维ND格式。
>      - transposeX2为true时维度为：（n，ceil(k / 8)），要求k为8的倍数。
>      - transposeX2为false时维度为：（k，ceil(n / 8)），要求n为8的倍数。
>      - 可使用aclnnConvertWeightToINT4Pack接口完成x2从INT32（1个int32在0~3bit位存储1个int4）到INT32（1个int32存储8个int4）或INT4（1个int4表示1个int4）的数据格式转换，具体参见[aclnnConvertWeightToINT4Pack接口](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)。

### Relevance

- **There is a non-grouped W4A8 MSD, and its correction vector is called `yOffset`**, shape `(n)`,
  fp32, `8*x2*x2Scale` reduced over dim 1. Three names for one quantity across three operators
  (`weightAssistMatrix`, `bias`, `yOffset`); the quantity is ours.
- **The W4A8 path on a plain matmul is `x1 = INT8, x2 = INT32` (8 int4 per int32), K-G quant,
  `gsK = 256`, `k % 256 == 0`, `k < 29576`, `x1Scale` shape `(m, 1)`,
  `x2Scale` shape `(ceil(k/256), n)`.** Note `x2Scale` is `(K_group, N)` — K-major, not N-major.
- **`// 2: in pre process, x has been splited to [2m,n]`** is the clearest statement of the layout
  in any CANN source. Our M-stacking matches.
- **Not one row of the table gives an INT32 `out` with an INT4 operand.** INT32 out exists only
  for `INT8 x INT8`. Combined with §3.2's `biasT = int32_t` on the int4 path — an int32 bias is
  added *inside* the cube for int4 operands, so the L0C accumulator is demonstrably int32 — this
  narrows §2.4's open question precisely: the accumulator is int32, and what no public CANN code
  does is *emit* it as int32 from Fixpipe when the operands are int4. That is a one-afternoon
  experiment for us, not an assumption.
- **`aclnnConvertWeightToINT4Pack`** is the supported way to get from one-int4-per-int32 to
  eight-int4-per-int32 or to native INT4 — relevant to our weight loader.

---

## 6. CANN has a *second*, different thing also called MSD — `WeightQuantBatchMatmulV2`

Repository: `https://gitcode.com/cann/ops-nn`, same clone as §5.

```
matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_multicore.h
matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_group.h
matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_split_k.h
matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_msd_group.cpp
matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_compute_matmul_tiling.cpp
```

This one is **A16W8**, and it decomposes a *float* activation into two int8 digits with radix
~127.5. It is the decomposition that the acronym "multiple significant digit" actually describes.
It is not the A8W4 scheme, and conflating them will send you to the wrong source.

```cpp
namespace WeightQuantBatchMatmulV2Msd {
static constexpr int32_t SYNC_VECTOR_CUBE_FLAG = 1;
static constexpr int32_t SYNC_CUBE_VECTOR_FLAG = 2;
static constexpr float EXPAND_FACTOR_1 = 127.499;
static constexpr float EXPAND_FACTOR_2 = 127.499 * 127.499 * 2;
```

The split, verbatim — row max, first digit, residual, second digit:

```cpp
        DivMC(xF32, xF32, xMax, baseM, kSizeAligned);
        PipeBarrier<PIPE_V>();
        Muls(xAbs, xF32, EXPAND_FACTOR_1, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        Cast(xAbs, xAbs, RoundMode::CAST_ROUND, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        auto a12F16 = a12F16_.Get<half>();
        Cast(a12F16, xAbs, RoundMode::CAST_NONE, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        auto a1S8 = outQueA1A2_.AllocTensor<int8_t>();
        Cast(a1S8, a12F16, RoundMode::CAST_NONE, baseM * kSizeAligned);
        ...
        DataCopyPad(workspaceXS8Global_[(singleM * curBlockIdx_ + offsetM) * kSize_], a1S8, aOutParams);
        outQueA1A2_.FreeTensor(a1S8);

        Muls(xAbs, xAbs, static_cast<float>(1 / EXPAND_FACTOR_1), baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        if constexpr (precisionType == PrecisionType::HIGH_PRECISION) {
            RowSumInplace(xSum, xAbs, baseM, kSizeAligned);
            PipeBarrier<PIPE_V>();
            // RowSumInplace 会修改 xAbs, 需要恢复数据
            Muls(xAbs, xF32, EXPAND_FACTOR_1, baseM * kSizeAligned);
            PipeBarrier<PIPE_V>();
            Cast(xAbs, xAbs, RoundMode::CAST_ROUND, baseM * kSizeAligned);
            PipeBarrier<PIPE_V>();
            Muls(xAbs, xAbs, static_cast<float>(1 / EXPAND_FACTOR_1), baseM * kSizeAligned);
            PipeBarrier<PIPE_V>();
        }
        Sub(xAbs, xF32, xAbs, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        Muls(xAbs, xAbs, EXPAND_FACTOR_2, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        Cast(xAbs, xAbs, RoundMode::CAST_ROUND, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        Cast(a12F16, xAbs, RoundMode::CAST_NONE, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        auto a2S8 = outQueA1A2_.AllocTensor<int8_t>();
        Cast(a2S8, a12F16, RoundMode::CAST_NONE, baseM * kSizeAligned);
```

i.e. `a1 = round(x/xMax * 127.499)`, `a2 = round((x/xMax - a1/127.499) * 127.499²*2)`, so
`x ≈ xMax * (a1/127.499 + a2/(127.499²·2))`. A two-digit radix-127.5 fixed-point representation,
exactly analogous to MSD-A8W4's two-digit radix-16 representation of an int8.

### 6.1 The decisive precision contrast — this MSD keeps `int32` out of the cube

```cpp
    using InputXType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, aTrans>;
    using InputWType = MatmulType<TPosition::GM, weightFormat, int8_t, bTrans>;
    using OutputYType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using InputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    MatmulImpl<InputXType, InputWType, OutputYType, InputBiasType, CFG_MDL> mmObj;
```

```cpp
    GlobalTensor<int8_t> workspaceXS8Global_;
    GlobalTensor<int32_t> workspaceCS32Global_;
```

```cpp
    LocalTensor<int32_t> tnsC1S32_;
    LocalTensor<int32_t> tnsC2S32_;
```

and the recombination, from `int32` partials upward:

```cpp
            tnsC1S32_ = inQueC1_.AllocTensor<int32_t>();
            DataCopyPad2D(tnsC1S32_, workspaceCS32Global_[offsetGm], curBaseM, curCubeSingleCoreN, n);
            inQueC1_.EnQue(tnsC1S32_);
            inQueC1_.DeQue<int32_t>();
            Cast(tnsC1F32_, tnsC1S32_, RoundMode::CAST_NONE, curBaseM * cubeSingleCoreNAligned);
            ...
            tnsC2S32_ = inQueC2_.AllocTensor<int32_t>();
            DataCopyPad2D(tnsC2S32_, workspaceCS32Global_[m * n + offsetGm], curBaseM, curCubeSingleCoreN, n);
            inQueC2_.EnQue(tnsC2S32_);
            inQueC2_.DeQue<int32_t>();
            Cast(tnsC2F32_, tnsC2S32_, RoundMode::CAST_NONE, curBaseM * cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();
            inQueC2_.FreeTensor(tnsC2S32_);
            Muls(tnsC2F32_, tnsC2F32_, static_cast<float>(1 / EXPAND_FACTOR_2), curBaseM * cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();

            Add(tnsC1F32_, tnsC1F32_, tnsC2F32_, curBaseM * cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();

            // c * amax
            MulMC(tnsC1F32_, tnsC1F32_, tnsMax_[offsetM * 8], curBaseM, cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();

            if constexpr (hasAntiQuantOffset) {
                // c * amax + offset
                Add(tnsC1F32_, tnsAntiQuantOffsetMnF32_, tnsC1F32_, curBaseM * cubeSingleCoreNAligned);
                PipeBarrier<PIPE_V>();
            }

            // (c * amax + offset) * scale
            MulMV(tnsC1F32_, tnsC1F32_, tnsAntiQuantScaleF32_, curBaseM, cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();
```

Note also that this MSD stacks the two digits as **two separate `m*n` planes** in workspace
(`workspaceCS32Global_[offsetGm]` and `workspaceCS32Global_[m * n + offsetGm]`), not interleaved —
a third layout choice inside the same codebase.

And one hard constraint, from the host tiling:

```cpp
bool WeightQuantBatchMatmulV2TilingMsdGroup::IsCapable()
{
    OP_LOGI(opName_, "Begin check msd group");
    // MSD pergroup模板不支持确定性
    if (context_->GetDeterministic() == 1) {
        OP_LOGI(opName_, "MSD pergroup do not support deterministic");
        return false;
    }
```

```cpp
        getTilingResult = MsdA16W8CommonTiling(matmulTiling, multiCoreResult, params, aicoreParams);
        ...
            "can not get tiling from SimpleIncreTiling/MsdA16W8CommonTiling. try to get tiling from cache tiling");
...
bool ComputeMatmulTiling::MsdA16W8CommonTiling(...)
            "MsdA16W8CommonTiling only support b is transposed and b is ND, "
...
    // 计算MSD场景transLen，L1Size和L0CSize
    CalcMsdBufferSize(matmulTiling, params);
```

### Relevance

- **Two different decompositions in CANN share the acronym.** A16W8 MSD: float activation → two
  int8 digits, radix 127.499, `int32` out of the cube. A8W4 MSD: int8 activation → two int4
  nibbles, radix 16, `half` out of the cube. The second is ours. When searching, use `a8w4_msd`,
  not `msd`, or you land in `weight_quant_batch_matmul_v2`.
- **This is the strongest evidence for the §2.4 fix.** The sibling MSD operator, written by the
  same team in the same source tree, does exactly what §2.4 proposes: `OutputYType = int32_t`,
  per-digit `int32` planes in workspace, `Cast` to fp32, radix-multiply the high digit, `Add`,
  then apply the scales. Our proposed int32 variant of A8W4-MSD is not novel — it is CANN's own
  pattern, applied to the operator where they did not apply it. The one unproven step is int4
  operands rather than int8 (see §5's Relevance).
- **The two-separate-planes layout** (`workspaceCS32Global_[0]` and `[m*n]`) is the third of three
  layouts CANN uses for the same "two partial products" problem. Combined with §3.4's interleave +
  strided read, the design space is: interleaved-with-strided-read (A8W4, both operators),
  two-planes (A16W8), blocked. We already chose M-stacking; the remaining choice is
  interleave-vs-two-planes, and CANN's own A8W4 choice is interleave.
- **`MSD pergroup do not support deterministic`** — CANN's own MSD pergroup template is excluded
  when determinism is requested. If we need run-to-run bit-exactness (we do, for A/B testing), the
  per-group atomic accumulation in §2.2 is the reason, and `quantGroupNum = 1` removes it.

---

## 7. "MSD" names at least four different decompositions in CANN — a taxonomy, so you search for the right one

This section exists because the acronym cost us search time and will cost the next reader the
same. All four are "split the activation into digits, matmul each digit against the quantized
weight, recombine with a radix". They differ in the activation dtype, the radix, and — decisively
for us — whether the cube output is `int32` or `half`.

| # | Operator / file | activation in | digits | radix | cube out | expansion |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `grouped_matmul*/..._a8w4_msd*`, `quant_batch_matmul_v4_msd.h` | INT8 | 2 x int4 | **16** | `half` | 「Mixed-precision Split-activation Decomposition / 混合精度激活拆分分解」 (§1) |
| 2 | `weight_quant_batch_matmul_v2_msd_*.h` (A16W8) | FP16/BF16 | 2 x int8 | **127.499**, then `127.499²·2` | `int32` | not expanded in source |
| 3 | `grouped_matmul/op_kernel/a16w4_msd/` (A16W4) | FP16/BF16 | 2 x int8 | **127.0**, `254.0` | `int32` | not expanded in source |
| 4 | `sparse_flash_attention_antiquant/..._gqa_msd.h` | FP16/BF16 | 2 x int8 | **127**, with `1024.0/127` and `1024.0` | `int32` | not expanded in source |

**Row 1 is ours.** Rows 2-4 are a different mechanism that happens to share the acronym: they
exist to avoid *dequantizing the weight* for a float activation, not to work around a missing
mixed-width MAD. Searching for bare `msd` in the CANN trees lands you in rows 2-4, which is why
the A8W4 scheme is hard to find.

Evidence for row 3, verbatim from
`gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_vector_service.h`:

```cpp
    float multiFactors[2] = {127.0f, 254.0f};
    ...
    Muls<float, false>(aF32Tensor_, aF32TmpTensor_[defaultOffset], multiFactors, FP32_MAX_MASK_SIZE, mainRepeatK,
```

and from `..._a16w4_msd_cube_service.h`, the on-chip types — int8 L0A/L0B, **int32 L0C**, and an
int32 bias table resident in BT:

```cpp
    LocalTensor<int8_t> s8l0a_;
    LocalTensor<int8_t> s8l0b_;
    ...
    LocalTensor<int32_t> s32l0c_;
    ...
    LocalTensor<int32_t> datoBiasBT_;
```

```cpp
    s8l0a_ = l0aBuffer.Get<int8_t>();
    s8l0b_ = l0bBuffer.Get<int8_t>();
    s32l0c_ = l0cTBuf.Get<int32_t>();
    InitConstValueParams<int32_t> initDatoBiasParams;
    ...
    AscendC::InitConstValue(cFp16L1_.template ReinterpretCast<int32_t>()[1024], initDatoBiasParams);
    ...
    datoBiasBT_ = biasTableTBuf.Get<int32_t>();
    DataCopy(datoBiasBT_, cFp16L1_.template ReinterpretCast<int32_t>()[1024], bias16InBTParams);
```

Evidence for row 4, verbatim from
`experimental/attention/sparse_flash_attention_antiquant/op_kernel/sparse_flash_attention_antiquant_service_vector_gqa_msd.h`:

```cpp
    static constexpr T antiqCoeff1 = 127;
    ...
    static constexpr float scaleC1 = 1024.0 / 127;
    static constexpr float scaleC2 = 1024.0;
```

```cpp
    Muls(srcUb, srcUb, antiqCoeff1, calcSize); // 127 * A
```

### 7.1 arXiv 2605.13915 — "Multi-Scale Dequant", i.e. rows 2-4, not row 1

URL: https://arxiv.org/abs/2605.13915 (PDF: https://arxiv.org/pdf/2605.13915)
Title: "Multi-Scale Dequant: Eliminating Dequantization Bottleneck via Activation Decomposition
for Efficient LLM Inference"
Authors: Lingchao Zheng, Yuwei Fan, Jun Li, Chengqiu Hu, Qichen Liao, Junyi Fan, Rui Shi,
Fangzheng Miao (no affiliations stated on the abstract page)
Submitted: 13 May 2026 (history: "Wed, 13 May 2026 09:49:56 UTC (271 KB)")
Subjects: primary stat.ML, also cs.AI, cs.LG

Short quotes (the abstract page's full text was not reproducible at length through the fetcher):

> presents Multi-Scale Dequant (MSD), a quantization framework

> removes weight/KV dequantization from the GEMM critical path

> decomposes high-precision BF16 activations into multiple low-precision components

each multipliable

> directly with quantized weights via native hardware-accelerated GEMM

> two-pass INT8 decomposition achieves near 16 effective bits

> two-pass MXFP4 decomposition yields near 6.6 effective bits with error bound 1/64 per block

> surpassing single-pass MXFP8(5.24 bits)

> architectures with decoupled compute units

(the only hardware named is "Ascend NPUs"), and on accuracy:

> does not degrade accuracy compared to dequantization baselines

sometimes reaching

> lower L2 error

plus an efficiency claim that KV cache HBM traffic is reduced

> by up to 2.5 times in attention

**This is a paper about rows 2-4, not about row 1.** Its activation is BF16, its components are
INT8 or MXFP4, and its stated purpose is removing weight/KV *dequantization* from the GEMM
critical path. Our `16*high + low` is not reproducing this paper. The two are siblings — "split
the activation, matmul the pieces, recombine" — but the arithmetic, the radix, the operand dtypes
and the problem being solved are all different. The paper's "near 16 effective bits" for two-pass
INT8 is the right way to read rows 2-4: two int8 digits buy you ~16 bits of activation precision.
Row 1's two int4 nibbles buy you exactly 8 bits, because that is all an int8 had.

The paper also does **not** report perplexity or benchmark accuracy — only L2 error against a
dequantization baseline. There is no published accuracy number for the A8W4-MSD nibble split.

### 7.2 Unsigned INT4 on Ascend: `uint4b_t` exists, and not on 2201

Verified first-hand in the AscendC headers. `impl/basic_api/utils/kernel_utils_constants.h`,
verbatim with its surrounding `#if`s:

```cpp
using int4b_t = integer_sub_type<INT4_BIT_NUM, true>;

#if (__NPU_ARCH__ == 5102)
using int2b_t = IntegerSubType<INT2_BIT_NUM, true>;
using int1b_t = IntegerSubType<INT1_BIT_NUM, true>;
using uint1b_t = IntegerSubType<INT1_BIT_NUM, false>;
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113))
using uint4b_t = IntegerSubType<INT4_BIT_NUM, false>;
using uint3b_t = IntegerSubType<INT3_BIT_NUM, false>;
using uint2b_t = IntegerSubType<INT2_BIT_NUM, false>;
#endif
```

`int4b_t` is unconditional; `uint4b_t` is gated to `__NPU_ARCH__ == 3003 || 3113`. We are 2201, so
**there is no unsigned int4 type for us** — the offset-fold is not one option among several, it is
the only option. `uint4b_t` is nonetheless referenced in the tensor implementation, so the type is
real where it is enabled: `impl/basic_api/kernel_tensor_impl.h`

```cpp
    } else if constexpr (IsSameType<PrimType, uint4b_t>::value) {
        ...
            KERNEL_LOG_INTERNAL(KERNEL_ERROR, "The offset for uint4b_t GetPhyAddr should be an even num."));
```

And `int4b_t`'s signedness is not nominal — the conversion operator sign-extends, which is exactly
why `(x & 0x0F) - 8` is needed rather than storing `0..15` directly.
`impl/utils/common_types.h`, verbatim:

```cpp
constexpr uint32_t INT4_BIT_NUM = 4;

template <int bit_num, bool sign = true>
struct integer_sub_type {
    static int const k_bits = bit_num;
    static bool const k_signed = sign;

    using T = typename conditional<k_signed, int8_t, uint8_t>::type;
    using Storage = uint8_t;

    static Storage const mask = Storage(((static_cast<uint64_t>(1)) << static_cast<uint32_t>(k_bits)) - 1);
    Storage storage;
    ...
    __aicore__ inline operator T() const
    {
        if (k_signed && ((storage & Storage(static_cast<uint64_t>(1) << static_cast<uint32_t>(k_bits - 1))) != 0)) {
            // Sign extend
            return T(storage) | ~T(mask);
        }
        return T(storage);
    }
```

For the `Mmad`/`matmul_int4` side of this question — the both-operands-int4 rule, the
「左矩阵A为int4数据类型时，不支持转置」 and 「int4数据类型输入的Matmul对象仅支持Norm、MDL、IBShare
模板」 constraints, and C restricted to int32/half — see `ascend-fusion-mechanics.md` §42.8, which
a parallel pass established from Huawei's own `matmul_int4` README. Not duplicated here.

### Relevance

- **Use `a8w4_msd` as the search key, never bare `msd`.** Four decompositions share the acronym;
  three of them are the wrong one.
- **arXiv 2605.13915 is not our scheme.** If someone cites it as "the MSD paper" for W4A8, that is
  a category error. There is, as of 2026-10-20, no paper on the A8W4 nibble split.
- **No unsigned int4 on 2201, confirmed from the header's own `#if`.** Our `8*sum(w)` fold is
  forced, not chosen. The one thing to keep in mind is that `int4b_t`'s `operator T()`
  sign-extends, so any host-side reference implementation must mask-then-subtract, not cast.
- **The int32 pattern is the majority pattern.** Three of the four MSD families keep `int32` out of
  the cube and apply scales in the vector unit; only the A8W4 one does not (§2.4). Row 3 even
  keeps the int32 bias resident in the bias table (`datoBiasBT_`) rather than adding it in UB,
  which is the natural place for our `8*sum(w)` if we take the int32 route.

---

## 8. How CANN schedules pre/mid/post as one kernel — the MSD pipeline, and the host tiling

Same repo and commits as §2. This section is here because the three-stage structure is the part
of MSD that is *not* about W4A8 at all — it is a worked example of a software-pipelined
multi-stage MIX kernel, which is what our megakernel has to be.

```
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h   (235 lines)
gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/grouped_matmul_swiglu_quant_v2_base_tiling.cpp
gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/grouped_matmul_swiglu_quant_v2_tiling.h
```

### 8.1 The three-stage loop, verbatim

```cpp
template <class mmType>
class GMMSwigluQuantPipelineSchedule {
private:
    typename mmType::MT &mm;
    TPipe *pipe;
    const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParams;
    const GMMSwigluQuantV2 *__restrict gmmSwigluQuantV2;
    // WorkSpaceSplitConfig控制Workspace切割方式的结构体;
    WorkSpaceSplitConfig workspaceSplitConfig;
    WorkSpaceSplitConfig tempWorkspaceSplitConfig;
    ...
    // 前处理GMMA8W4PreProcess类
    GMMA8W4PreProcess preProcess;
    // 中间处理GMMA8W4MidProcess类
    GMMA8W4MidProcess<mmType> midProcess;
    // 后处理GMMA8W4PostProcess类
    GMMA8W4PostProcess postProcess;
```

```cpp
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::Process()
{
    // 1.对每次workspace切分做大循环。
    preProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams);
    midProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams);
    postProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams, gmmSwigluQuantV2);

    // 1.前处理提前下发一次
    preProcess.Process(workspaceSplitConfig, 0, pipe);
    for (int64_t workspaceSplitLoopIdx = 0; workspaceSplitLoopIdx < workspaceSplitConfig.loopCount;
         workspaceSplitLoopIdx++) {
        // 更新workspaceSplitConfig
        UpdateWorkSpaceSplitConfig(workspaceSplitConfig, workspaceSplitLoopIdx);
        if ASCEND_IS_AIV {
            pipe->Reset();
        }

        SyncAll<false>();
        // 2.第n次中处理 && 第n+1次前处理 && 第n-1次后处理 并行
        midProcess.Process(workspaceSplitConfig, workspaceSplitLoopIdx);

        preProcess.Process(workspaceSplitConfig, workspaceSplitLoopIdx + 1, pipe);
        if ASCEND_IS_AIV {
            pipe->Reset();
            SyncAll<true>();
        }
        postProcess.Process(tempWorkspaceSplitConfig, workspaceSplitLoopIdx - 1, pipe);
        // 3.第n-1次后处理需要保留第n次的切分数据
        tempWorkspaceSplitConfig = workspaceSplitConfig;
        // reset
        if ASCEND_IS_AIV {
            pipe->Reset();
        }
        SyncAll<false>();
        // 3.前一次后处理 && 后一次MM 并行
    }
    // reset
    if ASCEND_IS_AIV {
        pipe->Reset();
    }
    SyncAll<false>();
    // // 4.最后一次后处理
    postProcess.Process(workspaceSplitConfig, workspaceSplitConfig.loopCount - 1, pipe);
    if ASCEND_IS_AIV {
        pipe->Destroy();
    }
}
```

Everything load-bearing is in that one function:

- **Three stages, one `<<< >>>`, software-pipelined at depth 1**: iteration `n` runs `mid(n)` on
  the AIC while the AIV runs `pre(n+1)` and then `post(n-1)`. `pre` is peeled before the loop and
  `post` is peeled after it.
- **`SyncAll<false>()` is the only phase barrier**, used three times per iteration (before `mid`,
  after `post`, and once more at loop exit). No `CrossCoreSetFlag` anywhere in this scheduler —
  the per-stage classes guard themselves with `if ASCEND_IS_AIV` / `if ASCEND_IS_AIC`.
- **`SyncAll<true>()` appears exactly once**, on the AIV only, between `pre(n+1)` and `post(n-1)` —
  the two AIV stages that share UB.
- **`pipe->Reset()` is called on the AIV before every stage** and `pipe->Destroy()` at the end.
  Each of `pre`/`post` calls `pipe->InitBuffer(...)` itself from inside its own `Process`, so the
  whole UB is re-partitioned per stage rather than statically divided three ways. This is the
  same trick as §3.2's `tPipe.Reset(); tPipe.Destroy(); tPipe.Init();`, used per-iteration.
- **The workspace is double-buffered by loop parity**, which is what makes the overlap safe —
  from §2.2 and §2.3: `xGM = (workspaceSplitLoopIdx % 2 == 0 ? xGM1 : xGM2);` and
  `mmOutGM = (workspaceSplitLoopIdx % 2 == 0 ? mmOutGM1 : mmOutGM2);`.
- **Each stage no-ops itself out of range** rather than being guarded by the caller:
  `if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount || workspaceSplitLoopIdx < 0) { return; }`,
  which is how the `n+1` and `n-1` indices are allowed to run off the ends.

The loop bound, from `InitWorkSpaceSplitConfig`:

```cpp
    workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, gmmSwigluQuantV2BaseParams->mLimit);
    workspaceSplitConfig.notLastTaskSize = gmmSwigluQuantV2BaseParams->mLimit;
    workspaceSplitConfig.lastLoopTaskSize =
        workspaceSplitConfig.M - (workspaceSplitConfig.loopCount - 1) * gmmSwigluQuantV2BaseParams->mLimit;
```

### 8.2 The host tiling — `mLimit`, the workspace budget, and the A8W4 base block

`grouped_matmul_swiglu_quant_v2_tiling.h`, verbatim:

```cpp
constexpr uint32_t SYS_WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr int64_t USER_WORKSPACE_LIMIT = static_cast<int64_t>(64 * 1024 * 1024);
constexpr int64_t DOUBLE_WORKSPACE_SPLIT = 2;
...
constexpr int64_t DOUBLE_ROW = 2;
...
constexpr int64_t A8W4_BASEM = 128;
constexpr int64_t A8W4_BASEK = 256;
constexpr int64_t A8W4_BASEN = 256;
constexpr int64_t SIZE_OF_HALF_2 = 2;
```

`grouped_matmul_swiglu_quant_v2_base_tiling.cpp`, verbatim — note the three branches and that the
A8W8 one is the only one sized in `int32_t`:

```cpp
    usrWorkspaceLimit_ = USER_WORKSPACE_LIMIT;
    mLimit_ = 0;
    if (isA8W4MSD_) {
        mLimit_ =
            ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) / (k_ * sizeof(int8_t) + DOUBLE_ROW * n_ * SIZE_OF_HALF_2));
    } else if (isA4W4_) {
        mLimit_ = ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) / (n_ * SIZE_OF_HALF_2));
    } else {
        mLimit_ = ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) / INT32_DTYPE_SIZE) / n_;
    }

    OP_CHECK_IF(
        mLimit_ <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "mLimit_ is %ld, must be greater than 0.", mLimit_),
        return ge::GRAPH_FAILED);
    tilingData_.gmmSwigluQuantV2BaseParams.set_mLimit(mLimit_);

    DynamicTilingSingleN(context_, blockDim_, A8W4_BASEM, A8W4_BASEN, A8W4_BASEK);

    if (isA8W4MSD_) {
        int workSpaceMTemp = mLimit_ * DOUBLE_WORKSPACE_SPLIT;
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset1(workSpaceMTemp * k_ * sizeof(int8_t));
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset2(DOUBLE_ROW * workSpaceMTemp * n_ * SIZE_OF_HALF_2);
        workspaceSize_ =
            SYS_WORKSPACE_SIZE +                     // 系统预留16MB
            (workSpaceMTemp * k_ * sizeof(int8_t)) + // 第一阶段 预处理左矩阵 (mLimit_, K) * int8 * 2(double WorkSpace)
            (DOUBLE_ROW * workSpaceMTemp * n_ *
             SIZE_OF_HALF_2); // 第二阶段 矩阵乘结果 (2 * mLimit_, N) * fp16 * 2(double WorkSpace)
    } else if (isA4W4_) {
        int workSpaceMTemp = mLimit_ * DOUBLE_WORKSPACE_SPLIT;
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset1(mLimit_ * n_ * SIZE_OF_HALF_2);
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset2(0);
        workspaceSize_ = SYS_WORKSPACE_SIZE + (workSpaceMTemp * n_ * SIZE_OF_HALF_2);
    } else {
        int workSpaceMTemp = (mLimit_ * DOUBLE_WORKSPACE_SPLIT > m_ ? m_ : mLimit_ * DOUBLE_WORKSPACE_SPLIT);
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset1(0);
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset2(0);
        workspaceSize_ = SYS_WORKSPACE_SIZE + (workSpaceMTemp * n_ * sizeof(int32_t));
    }

    isSplitWorkSpace_ = m_ > mLimit_ * DOUBLE_WORKSPACE_SPLIT;
```

Note the comment 「第一阶段 预处理左矩阵 (mLimit_, K) * int8」: the split A matrix is accounted as
`mLimit × K` *bytes*, i.e. `2 × mLimit` rows of `K` int4 — the M-doubling shows up in the byte
count as "one int8 per output element" and nowhere else. The mm result is explicitly
「(2 * mLimit_, N) * fp16」.

### Relevance

- **This is the closest thing in public CANN to the structure our per-layer megakernel needs**: N
  stages in one launch, `SyncAll<false>()` as the phase boundary, `pipe->Reset()` to re-partition
  UB per stage, parity-double-buffered GM between stages, and each stage self-guarding on
  `ASCEND_IS_AIV`/`ASCEND_IS_AIC` plus an out-of-range early return so the skewed indices are
  free. Zero `CrossCoreSetFlag` in the scheduler — which, given our flag-budget findings in
  `ascend-fusion-mechanics.md` §38-§41, is the cheapest discipline available.
- **`mLimit` for A8W4-MSD is `(32 MB) / (K + 4N)`** bytes-wise, with 64 MB of user workspace split
  in two. For GLM-5.2 expert shapes that is thousands of rows, so at batch=1 decode
  `loopCount == 1` and the whole three-stage pipeline degenerates to `pre; mid; post` with three
  `SyncAll<false>()` barriers and no overlap at all. **CANN's pipeline buys us nothing at our
  shape** — another reason the vendor operator is not tuned for our regime, and a reason to fuse
  across *layers* rather than across M-chunks.
- **`A8W4_BASEM = 128, A8W4_BASEK = 256, A8W4_BASEN = 256`** are the host-side defaults, against
  the `// 8, 256, 512 --> 514us` comment left in the v1 kernel (§2.2). `baseM = 128` on 2M rows is
  64 logical tokens; at batch=1 we want `baseM = 2` and 126/128 of that tile is waste.
- **The two mm-workspace formulas price §2.4 exactly.** A8W8 is `mLimit * n_ * sizeof(int32_t)`
  = 4 B per logical output element; A8W4-MSD is `DOUBLE_ROW * mLimit * n_ * SIZE_OF_HALF_2` = also
  4 B per logical output element, as two fp16 halves. They are already equal. An int32 MSD would
  be `DOUBLE_ROW * mLimit * n_ * 4` = 8 B per logical output, i.e. **exactly 2x the mm workspace
  and 2x the cube→GM and GM→UB traffic on that buffer** — which is the real price of the §2.4
  accuracy fix, and the thing to measure against the 16x error amplification it removes. At
  batch=1 decode `loopCount == 1` and `2M` is single digits, so the absolute bytes are negligible;
  the question is purely whether the extra L2 traffic shows up in the kernel time.

---

## 9. What CANN actually validates for A8W4-MSD — and what it does not

Same repos and commits as §2 and §5. This section is short because the finding is an absence, and
the absence is load-bearing for question 3 (accuracy cost).

### 9.1 A3 is documented as supported

`gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md` gates its sections
with HTML comments naming the product line. The 「MSD场景A8W4」 formula block sits inside:

```
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
```

and the operator's own support list reads, verbatim:

> <!-- npu="950" id1 -->
> - <term>Ascend 950PR/Ascend 950DT</term>：支持
>
> <!-- npu="A3" id2 -->
> - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
>
> <!-- npu="910b" id3 -->
> - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持

So A8W4-MSD is a documented A3 feature, not an A2-only one.

### 9.2 The only published MSD test case list is on A2, and it is tiling-only

`ops-nn/matmul/quant_batch_matmul_v4/tests/ut/op_host/test_quant_batch_matmul_v4_a8w4_tiling.csv`,
the whole file (4 lines including the header):

```
socVersion,caseName,blockDim,tilingResult,tilingKey
Ascend910B4,UT-A8W4-MSD-ND-Testcase-0_16_256_256_0_0_256_ND_ND_INT8_INT4_NULL_FP32_UINT64_NULL_FP32_BF16_20_40,20,GRAPH_SUCCESS,24
Ascend910B4,UT-A8W4-MSD-ND-Testcase-1_16_256_256_0_0_256_ND_ND_INT8_INT4_NULL_FP32_UINT64_NULL_FP32_FP16_20_40,20,GRAPH_SUCCESS,24
Ascend910B4,UT-A8W4-MSD-ND-Testcase-2_16_256_256_0_0_0_ND_ND_INT8_INT4_NULL_FP32_UINT64_NULL_FP32_FP16_20_40,1,GRAPH_SUCCESS,20
```

```cpp
INSTANTIATE_TEST_CASE_P(MMA8W4MSD, TestQuantBatchMatmulV4TilingMsd, testing::ValuesIn(GetParams()));
```

Decoded: `M=16, K=256, N=256`, `[gsM, gsN, gsK] = [0, 0, 256]` for cases 0-1 and
`[0, 0, 0]` for case 2, `x1 = INT8`, `x2 = INT4`, `x1Scale = FP32`, `x2Scale = UINT64`,
`yOffset = FP32`, `out = BF16` then `FP16` then `FP16`. **Tiling key 24 for the per-group (K-G)
path, 20 for the per-channel (K-C) path.** `blockDim` 20 for K-G, 1 for K-C. `socVersion` is
`Ascend910B4` on every row. The assertion is `GRAPH_SUCCESS` and a tiling key — no output values
are compared.

### 9.3 CORRECTED — there *is* a published numeric test, in a newer commit than the one in §9.2

> This subsection originally read "There is no published numeric test for A8W4-MSD at all". That
> was wrong, and it was wrong because the clone it was written from is a month stale. The A8W4 ST
> and its torch golden reference were added between `e027a008` (2026-09-01) and the current
> `ops-transformer` HEAD. The full golden reference, its precision standard, and the decisive thing
> it reveals are in §11.2. Corrected here rather than deleted, because the stale-clone failure mode
> is worth remembering: on `ops-transformer`, re-pull before concluding an absence.

What the stale clone at `e027a008` (2026-09-01) does contain, which is still a true statement about
that commit:

- `ops-transformer/gmm/*/tests/st/` holds exactly
  `aclnnGroupedMatmulV4_NoQuant`, `aclnnGroupedMatmulV4_SplitK`, `aclnnGroupedMatmulV5_A8W8O16`,
  `aclnnGroupedMatmulV5_A8W8O16AXIS`, `aclnnGroupedMatmulV5_A8W8O8`,
  `aclnnGroupedMatmulSwigluQuantV2_A8W8`, `aclnnGroupedMatmulSwigluQuant`,
  `aclnnGroupedMatmulAdd`, `aclnnGroupedMatmulAddV2` — no A8W4 directory.

The newer snapshot (`d6aaa9fe`, 2026-10-19) adds two:
`aclnnGroupedMatmulSwigluQuantV2_A4W4` and `aclnnGroupedMatmulSwigluQuantV2_A8W4`.

Still true at both commits: no `rtol` / `atol` / `EXPECT_NEAR` constant appears in any A8W4 or MSD
test in either repo — the precision standard travels in the ST's JSON config instead (§11.2.3), not
in the test code.

### Relevance

- **The accuracy budget is published after all, and it is set at the fp16 scale.** See §11.2.3:
  `"fp32_error": "2**-10"`, `"fp32_eb": "2**-1"`, under `"type": "high_performance"`. `2^-10` is
  the fp16 mantissa resolution. CANN is not claiming int-exactness for this operator in its own
  test harness, whatever §1 says in prose.
- **Tiling keys to look for in a profile**: 24 = A8W4-MSD per-group (K-G), 20 = A8W4-MSD
  per-channel (K-C) on `QuantBatchMatmulV4`; 2 = A8W4-MSD on `GroupedMatmulSwigluQuant[V2]`
  (§2.0), 4 and 5 = A4W4, 0 and 1 = A8W8.
- **The one published shape is `M=16, K=256, N=256`.** That is a tiling smoke test, not a
  performance point. Nothing in the public test corpus exercises our regime (M of 1-8 per expert,
  K and N in the thousands) or our chip.

---

## 10. The L1-resident variant — load the split A once, loop N against it

Repository: `https://gitcode.com/cann/ops-transformer`, same clone as §2.
Path: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_l1_opt.h`

A fourth A8W4-MSD implementation, in `GroupedMatmulFinalizeRouting`, with one difference from the
other three: it stages the split A matrix in **L1** and feeds `SetTensorA` from there, so A is
read from GM exactly once per M-tile instead of once per (N-tile x quant-group). At our shape A is
tiny and W is enormous, which makes this the right structure for us and the other three wrong.

The L1 allocation — note the `/2` and its comment, and that this is cube-only:

```cpp
    TSCM<TPosition::GM, 1> xL1Queue;
```

```cpp
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::InitMMBuffer()
{
    if ASCEND_IS_AIV {
        return;
    }
    pipe->InitBuffer(xL1Queue, 1, (tiling->matmulTiling.baseM * tiling->k) / 2); // divide size by 2 to get sizeof(int4b_t)
}
```

The GM→L1 load, with ND2NZ done on the fly by `DataCopy`'s `Nd2NzParams` overload — every `d`
dimension halved because the element is int4:

```cpp
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::CopyInA1(uint64_t xOffset, uint32_t row, uint32_t col, LocalTensor<int4b_t> a1Local)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = row;
    nd2nzParams.dValue = col / 2;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = col / 2;
    nd2nzParams.dstNzC0Stride = Ceil(row, 16) * 16;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(a1Local, xGm[xOffset], nd2nzParams);
    xL1Queue.EnQue(a1Local);
    a1Local = xL1Queue.DeQue<int4b_t>();
}
```

And the loop that pays for it — one `CopyInA1` outside, then `baseN` tiles x `quantGroupNum`
`Iterate`s all reading A out of L1:

```cpp
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::MMComputeSingleN(uint32_t groupIdx, MNConfig& mnConfig,
                                                                        uint32_t singleNOffset, uint32_t curSingleN, uint32_t curSingleM)
{
    uint64_t xOffset = (static_cast<uint64_t>(mnConfig.offsetM) + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
    LocalTensor<int4b_t> xL1LocalInt4 = xL1Queue.AllocTensor<int4b_t>();
    CopyInA1(xOffset, curSingleM, tiling->k, xL1LocalInt4);

    uint32_t baseNCurSingleN = Ceil(curSingleN, mnConfig.baseN);

    for (uint32_t baseNIdx = 0; baseNIdx < baseNCurSingleN; baseNIdx++) {
        if (baseCubeCount % mnConfig.baseNEachSingleN == 0) { // 进入新一轮 更新workspace从起点开始
            mnConfig.workSpaceOffset = mnConfig.baseM * mnConfig.singleN * coreIdx;
        }
        if (baseCubeCount >= mnConfig.baseNEachSingleN) {
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        uint64_t weightOffset;
        if constexpr (mmType::BT::format == CubeFormat::NZ) {
            weightOffset = (static_cast<uint64_t>(groupIdx)) * tiling->n * tiling->k + singleNOffset * tiling->k + baseNIdx * mnConfig.baseN * tiling->k;
        } else {
            weightOffset = (static_cast<uint64_t>(groupIdx)) * tiling->n * tiling->k + singleNOffset + baseNIdx * mnConfig.baseN;
        }

        uint32_t curBaseN = mnConfig.baseN;
        if (unlikely(baseNIdx == baseNCurSingleN - 1)) {
            curBaseN = curSingleN - baseNIdx * mnConfig.baseN;
        }
        mm.SetSingleShape(curSingleM, curBaseN, quantGroupSize);
        GlobalTensor<int4b_t> weightSlice;
        for (uint32_t loopK = 0; loopK < tiling->quantGroupNum; loopK++) {
            mm.SetTensorA(xL1LocalInt4[loopK * quantGroupSize]); // Copy x from L1
            if constexpr (mmType::BT::format == CubeFormat::NZ) {
                weightSlice = weightGm[weightOffset + loopK * quantGroupSize * 64];
            } else {
                weightSlice = weightGm[weightOffset + loopK * quantGroupSize * tiling->n];
            }
            if (mnConfig.blockDimM == 1) {
                weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            mm.SetTensorB(weightSlice);
            mm.SetQuantVector(scaleGm[groupIdx * tiling->n * tiling->quantGroupNum + loopK * tiling->n + singleNOffset + baseNIdx * mnConfig.baseN]);
            mm.Iterate();
            mm.GetTensorC(mmOutGm[mnConfig.workSpaceOffset], loopK == 0 ? 0 : 1, true);
        }
        mm.End();
        mnConfig.workSpaceOffset += mnConfig.baseM * mnConfig.baseN;
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
        baseCubeCount++;
    }
    xL1Queue.FreeTensor(xL1LocalInt4);
}
```

Its `cT` is, predictably, `half`:

```cpp
class GMMA8W4MSDL1OptCompute {
public:
    using cT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
```

and it carries the full `withOffset` / `xRowSum` asymmetric path from §3.5, plus a
`VectorAtomicProcess` for the finalize-routing scatter-add.

### Relevance

- **This is the structure to copy for batch=1 decode, not §2's or §3's.** At `2M = 16` and
  `K = 7168` the split A is `16 * 7168 / 2 = 57344 B = 56 KB`, which sits in A3's 512 KB L1 with
  room to spare — so A can be loaded once per expert and every `baseN` tile and every quant group
  reads it from L1. The other three implementations re-issue `SetTensorA(xGM[...])` per
  `(N-tile, quant-group)` and let the hardware/L2 absorb the repetition. At tiny M the A traffic is
  negligible either way in *bytes*, but the per-`Iterate` MTE2 setup is not negligible in a
  2.x ms budget, and holding A in L1 also removes it from contention with the weight stream, which
  is the only thing that should be moving.
- **`DataCopy` with `Nd2NzParams` does GM-ND → L1-NZ for `int4b_t` directly.** Every `d`-axis
  field is halved (`dValue = col/2`, `srcDValue = col/2`) and `dstNzC0Stride = Ceil(row,16)*16`.
  That is the exact incantation; it is not documented anywhere we found, only shown here.
- **`TSCM<TPosition::GM, 1>`** is the queue type that lets a `LocalTensor<int4b_t>` in L1 be passed
  to `SetTensorA`. Worth noting because `TSCM`'s first template argument being `TPosition::GM` is
  counter-intuitive.
- **`mm.End()` is called per N tile**, not per expert — relevant to the `Matmul` object lifetime
  question in `ascend-fusion-mechanics.md` §41.
- **Same credit-window handshake as §5**: `CrossCoreWaitFlag(SYNC_AIV_TO_AIC)` gated on
  `baseCubeCount >= baseNEachSingleN`, then `CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV)` with
  the comment 「2: mode为2, group内同步」. Depth is `baseNEachSingleN`, i.e. the number of N tiles
  per `singleN` — a data-derived credit depth rather than a constant.

---

## 11. The instruction-level contract, and the golden reference that admits the fp16 loss

Two things in this section. §11.1 is the `Mmad` / `Matmul` / `Cast` contract for `int4b_t` on A3 —
which settles whether the §2.4 int32 variant is even legal. §11.2 is CANN's own torch golden
reference for A8W4-MSD, which is the single most important artifact in this file.

### 11.1 `Mmad` on A3 — the dtype table, and one constraint that is only in the source

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html
Page footer: 「版本 master 提交 5f50fc4c6ada 构建于 2026/10/20 11:47:29 包含未提交修改」
Fetched: 2026-10-20

> 头文件路径为： "basic_api/kernel_operator_mm_intf.h" 。
> ```
> template < typename T , typename U , typename S >
> __aicore__ inline void Mmad ( const LocalTensor < T >& dst , const LocalTensor < U >& fm , const LocalTensor < S >& filter , const MmadParams & mmadParams )
> template < typename T , typename U , typename S , typename V >
> __aicore__ inline void Mmad ( const LocalTensor < T >& dst , const LocalTensor < U >& fm , const LocalTensor < S >& filter , const LocalTensor < V >& bias , const MmadParams & mmadParams )
> ```

> 表7 dst、fm、filter支持的精度类型组合（Atlas A2系列产品）（Atlas A3系列产品）（Atlas 200I/500 A2推理产品）
> int8_t | int8_t | int32_t
> half | half | float
> float | float | float
> bfloat16_t | bfloat16_t | float
> int4b_t | int4b_t | int32_t
>
> 表8 dst、fm、filter、bias支持的精度类型组合（Atlas A2系列产品）（Atlas A3系列产品）（Atlas 200I/500 A2推理产品）
> int8_t | int8_t | int32_t | int32_t
> half | half | float | float
> float | float | float | float
> bfloat16_t | bfloat16_t | float | float

**`int4b_t | int4b_t | int32_t` and nothing mixed.** 表8 has no `int4b_t` row at all, so the
bias-overload `Mmad` is unavailable for int4 operands on A2/A3. `uint8_t` appears only in 表9
(310P: `uint8_t|int8_t|int32_t`, `uint8_t|uint8_t|int32_t`) and 表10 (910: `uint8_t|uint8_t|uint32_t`,
`uint8_t|int8_t|int32_t`) — other products, not ours.

A3 operand placement and layout:

> dst | 输出 | …支持的物理存储位置为L0C Buffer（TPosition:CO1）。LocalTensor的起始地址需要按照1024字节对齐。
> fm | 输入 | 左矩阵A…L0A Buffer（TPosition: A2）。LocalTensor的起始地址需要按照512字节对齐。
> filter | 输入 | 右矩阵B…L0B Buffer（TPosition: B2）。LocalTensor的起始地址需要按照512字节对齐。
> bias | 输入 | Bias矩阵…BT Buffer（TPosition:C2）。LocalTensor的起始地址需要按照64字节对齐。

> A | L0A Buffer | M x K | Zz ; B | L0B Buffer | K x N | Zn ; C | L0C Buffer | M x N，可支持使用偏置矩阵Bias进行初始化，维度为1 x N | Nz

(On 950PR/950DT, A is `Nz` rather than `Zz` — a layout difference to watch if code is ported.)

The constraint that matters most for a batch=1 MSD kernel, verbatim:

> 当M、K、N中的任意一个值为0时，表示指令不会执行，该接口将被视为NOP（空操作）。
> 当M = 1时，会默认开启GEMV（General Matrix-Vector Multiplication）功能。在这种情况下，Mmad API从L0A Buffer读取数据时，会以ND格式进行读取，而不会将其视为ZZ或NZ格式。所以此时左矩阵需要直接按照ND格式进行排布…针对Ascend 950PR&950DT系列产品，可以通过设置MmadParams的disableGemv参数为true，将该功能关闭。
> 同步优化说明 … `if (( m / 16 ) * ( n / 16 ) < 10 ) { AscendC :: PipeBarrier < PIPE_M > (); }`
> 注：Mmad指令应避免NaN输入，否则可能会产生执行报错。

`MmadParams`, from https://gitcode.com/cann/asc-devkit/blob/master/include/basic_api/kernel_struct_mm.h
lines 449-481 (HEAD `5f50fc4c6ada`, 2026-10-20), verbatim:

```cpp
    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    bool isBias = false;
    int32_t fmOffset = 0;
    bool enSsparse = false;
    bool enWinogradA = false;
    bool enWinogradB = false;
    uint8_t unitFlag = 0;
    // also mean gemvCtrl in 3101 and 5102 and 5162
    bool kDirectionAlign = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;
#if defined(__NPU_ARCH__) && \
    ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    bool disableGemv = false;
#endif
```

and the field docs:

> m | 左矩阵Height，取值范围：m∈[0, 4095]。默认值为0。
> n | 右矩阵Width，取值范围：n∈[0, 4095]。默认值为0。
> k | 左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095]。默认值为0。
> cmatrixInitVal | 是否开启C矩阵默认初始化清零操作。默认值为 true。• true：C矩阵默认初始化为0；• false：C矩阵不进行默认操作，通过设置cmatrixSource参数进行初始化。
> cmatrixSource | 配置C矩阵初始值是否来源于BT Buffer。默认值为false。…Atlas A3系列产品，支持配置为true/false。…注意：带Bias输入的接口配置该参数无效，会根据bias输入的位置来判断C矩阵初始值是否来源于BT Buffer。
> isBias | 该参数废弃，新开发内容不要使用该参数。…• false：矩阵乘，无需累加初始矩阵，C = A * B。• true：矩阵乘加，需要累加初始矩阵，C += A * B。
> disableGemv | M=1时，该参数用来配置Mmad计算是否开启 GEMV 模式。…该参数仅支持如下型号：Ascend 950PR&950DT系列产品
> unitFlag | …• 0（2'b00）：不开启unitFlag；• 2（2'b10）：开启unitFlag，硬件执行完指令之后，不复位单元标记位；• 3（2'b11）：…复位单元标记位。…该参数仅支持如下型号：Ascend 950PR&950DT系列产品 / Atlas A2系列产品 / Atlas A3系列产品。
> kDirectionAlign | …控制在使用float数据类型时，L0A Buffer和L0B Buffer矩阵在K方向上的对齐方式。• false：默认值，K方向对齐到 ceil(K / 8) * 8 。• true：K方向对齐到 ceil(K/16)*16 。
> fmOffset | 左矩阵offset…注：未使用，兼容旧款产品接口传入，Atlas A2系列产品及往后产品不做处理。
> enSsparse | 开启结构化稀疏特性，默认false；注：未使用…
> enWinogradA / enWinogradB | …注：未使用…

#### The `k % 2 == 0` rule, which is in the source and not on the page

URL: https://gitcode.com/cann/asc-devkit/blob/master/impl/basic_api/dav_c220/kernel_operator_mm_impl.h
(`dav_c220` is the dav-2201 tree; see the arch binding below). Lines 367-394, verbatim:

```cpp
template <typename T, typename U, typename S>
__aicore__ inline void MmadCal(__cc__ T* c, __ca__ U* a, __cb__ S* b, const MmadParams& mmadParams)
{
    if ASCEND_IS_AIC {
        ASCENDC_DEBUG_ASSERT(
            (SupportType<
                Tuple<T, U, S>, Tuple<int32_t, int8_t, int8_t>, Tuple<float, half, half>, Tuple<float, float, float>,
                Tuple<float, bfloat16_t, bfloat16_t>, Tuple<int32_t, int4b_t, int4b_t>>()),
            KERNEL_LOG_INTERNAL(
                KERNEL_ERROR,
                "Failed to check dtype in "
                "Mmad, current api support dtype combination is dst: int32_t, fm: int8_t, filter: int8_t; dst: "
                "float, fm: half, filter: half; dst: float, fm: float, filter: float; dst: float, fm: "
                "bfloat16_t, filter: bfloat16_t; dst: int32_t, fm: int4b_t, filter: int4b_t"));
        bool cmatrixInitVal = mmadParams.cmatrixInitVal && (!mmadParams.isBias);
        if constexpr ((IsSameType<U, int4b_t>::value) && (IsSameType<S, int4b_t>::value)) {
            ASCENDC_DEBUG_ASSERT(
                (mmadParams.k % 2 == 0), KERNEL_LOG_INTERNAL(
                                             KERNEL_ERROR, "MmadParams.k must be an "
                                                           "even number when U and S are int4b_t.\n"));
            mad_s4(
                c, (__ca__ void*)a, (__cb__ void*)b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag,
                mmadParams.kDirectionAlign, mmadParams.cmatrixSource, cmatrixInitVal);
        } else {
            mad(c, a, b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.kDirectionAlign,
                mmadParams.cmatrixSource, cmatrixInitVal);
        }
    }
}
```

The `if constexpr` requiring **both** `U` and `S` to be `int4b_t` before `mad_s4` is reached is the
mechanism behind the rule we had established. The bias overload (`MmadCal(..., uint64_t bias, ...)`,
lines 396-421) declares a `SupportType` list that omits `int4b_t` while still carrying a `mad_s4`
branch underneath — unreachable by contract, so an int4 bias path through `Mmad` is a trap.

Arch binding, so `dav_c220` ↔ 2201 is not folklore.
https://gitcode.com/cann/asc-devkit/blob/master/.agent/skills/asc-npu-arch/references/npu-arch-facts.json:

> `"ascend910b1": { "enum_name": "ASCEND910B1", "npu_arch": "2201", "soc_version": "ASCEND910B", "arch_dir": "ascendc_case_ascend910b1" }`
> `"S4": { "type": "int4b_t", "storage_type": "int4x2_t" }`

https://gitcode.com/cann/asc-devkit/blob/master/cmake/asc/asc_modules/CMakeASCInformation.cmake:

> `elseif(CMAKE_ASC_ARCHITECTURES STREQUAL "dav-2201")` → `set(_SOC_DIR "Ascend910B1")`

#### The microscaling and sparse escapes are both closed on A3

https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/MmadMx.html and
https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/MmadBitMode.html,
both prototypes, both pages:

> Ascend 950PR&950DT系列产品：支持 / Atlas A3系列产品：不支持 / Atlas A2系列产品：不支持 …

`MmadMx`'s dtype table is fp4x2/fp8 with an `fp8_e8m0_t` scale and never admits `int4b_t`.
`mad_sp` is int8-only: `MmadSpCal(__cc__ int32_t* c, __ca__ int8_t* a, __cb__ int8_t* b, ...)`.
So on A3 the MAD family reduces to `mad` and `mad_s4`, confirming our own finding.

And PTO is stricter still — https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TMATMUL.md
(HEAD `a2044ac6`, 2026-10-20):

> Implementation checks (A2A3): Supported `(CType, AType, BType)` triples: `(int32_t, int8_t, int8_t)` / `(float, half, half)` / `(float, float, float)` / `(float, bfloat16_t, bfloat16_t)`
> Accumulator pitch constraint (CPU, A2A3, A5, A6, and Kirin9030): `mad` writes the accumulator's block columns at a pitch of `ceil16(m)` rows, where `m` is the row count the instruction receives, while every reader of that region takes its pitch from the tile's `Rows`. The two must agree

**PTO does not expose int4 on A2A3 at all.** Going through PTO tiles loses the `mad_s4` path
entirely — a hard constraint on the toolchain choice, not just the kernel.

#### The high-level `Matmul` DOES allow `int32_t` out of an int4 x int4 GEMM

URL: https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/Matmul_usage.html
表2 Matmul输入输出数据类型的支持列表, verbatim:

> int8_t | int8_t | int32_t | int32_t/half | …Atlas A3系列产品…
> int4b_t | int4b_t | int32_t | int32_t/half | Atlas A3系列产品 / Atlas A2系列产品
> half | int8_t | float | float | Atlas推理系列产品AI Core

(the `half | int8_t` row is the table's only mixed-dtype row, and it is 310P-only)

> 注意：除fp8_e4m3fn_t/fp8_e5m2_t两种数据类型、B矩阵为int8_t数据类型外，A矩阵和B矩阵数据类型需要一致…A矩阵和B矩阵为int4b_t数据类型时，矩阵内轴的数据个数必须为偶数。例如，A矩阵为int4b_t数据类型且不转置时， singleCoreK 必须是偶数。
> int4b_t/fp4x2_e2m1_t/fp4x2_e1m2_t数据类型的C0_size为64。
> // A矩阵数据类型是int8_t或int4b_t时，C矩阵的数据类型是int32_t。

**This closes §2.4's and §5's open question.** The column order in 表2 is
`A | B | Bias | C | products`, and the `int4b_t` row reads `int4b_t | int4b_t | int32_t |
int32_t/half` on Atlas A3. So `C` may be `int32_t`, and the `int32_t` in the Bias column means the
bias slot for int4 operands is int32 at the `Matmul` level even though raw `Mmad`'s 表8 omits it.
The int32 variant proposed in §2.4 is **in contract for the `Matmul` API on A3** — CANN's own A8W4
kernels simply never instantiate it (all four use `cT = half`; see §2.0, §3.2, §5.1, §10). What is
*not* available is an `aclnn`-level operator that returns int32 for an int4 operand (§5.2), which is
an API-surface limit, not a hardware one.

#### `Cast` to/from `int4b_t` on A3 — `half` only, and count must be even

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/type_conversion/Cast.html
(same mirror build, 2026-10-20)

> 头文件路径为 "basic_api/kernel_operator_vec_vconv_intf.h" 。
> ```
> template < typename T , typename U >
> __aicore__ inline void Cast ( const LocalTensor < T >& dst , const LocalTensor < U >& src , const RoundMode & roundMode , const uint32_t count )
> ```
> ```
> enum class RoundMode {
>   CAST_NONE = 0 , // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
>   CAST_RINT , // rint，四舍六入五成双舍入
>   CAST_FLOOR , // floor，向负无穷舍入
>   CAST_CEIL , // ceil，向正无穷舍入
>   CAST_ROUND , // round，四舍五入舍入
>   CAST_TRUNC , // trunc，向零舍入
>   CAST_ODD , // Von Neumann rounding，最近邻奇数舍入
>   CAST_HYBRID , // hybrid，目前特指输出结果是hifloat8_t数据时，会用到的一种随机舍入
> };
> ```

The A3 rows involving `int4b_t`, verbatim from 表 Atlas A3系列产品支持的数据类型组合情况:

> int4b_t | half | CAST_NONE
> half | int4b_t | CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC/CAST_NONE

`CAST_FLOOR` for `half → int4b_t` is in contract, which is exactly what §2.1's high-nibble
extraction uses. **On A3 `int4b_t` pairs only with `half`** — there is no `int4b_t ↔ bfloat16_t`
and no `int4b_t ↔ int16_t` row; those exist only in the 950PR/950DT table
(`int4b_t | int16_t | CAST_NONE`, `int4b_t | bfloat16_t | CAST_NONE`, `int16_t | int4b_t | CAST_NONE`).
That is why every CANN A8W4 pre-stage routes through `half` and not through `int16_t`, and why the
split cannot be done with an integer shift in the vector unit.

约束说明, verbatim:

> 当dst或src为int4b_t时，由于一个int4b_t只占半个字节，故申请Tensor空间时，只需申请相同数量的int8_t数据空间的一半。host侧目前暂不支持int4b_t，故在申请int4b_t类型的tensor时，应先申请一个类型为int8_t的tensor，再用Reinterpretcast接口转化为int4b_t类型的tensor，接着调用Cast指令。
> 当dst或src为int4b_t时，tensor高维切分计算接口的连续模式的mask与tensor前n个数据计算接口的count必须为偶数；对于tensor高维切分计算接口的逐bit模式，对应同一字节的相邻两个比特位的数值必须一致，即0-1位数值一致，2-3位数值一致，4-5位数值一致，以此类推。
> 操作数地址重叠约束请参考 通用地址重叠约束 。特别地，对于长度较小的数据类型转换为长度较大的数据类型时，地址重叠可能会导致结果错误。

> ```
> AscendC :: LocalTensor < int8_t > dstLocal ;
> dstLocal . SetSize ( outBufferSize_ );
> AscendC :: LocalTensor < AscendC :: int4b_t > dstLocalTmp = dstLocal . ReinterpretCast < AscendC :: int4b_t > ();
> AscendC :: Cast < AscendC :: int4b_t , half > ( dstLocalTmp , srcLocal , AscendC :: RoundMode :: CAST_CEIL , mask , 1 , { 1 , 1 , 2 , 8 });
> ```

The intrinsic, from https://gitcode.com/cann/asc-devkit/blob/master/impl/basic_api/dav_c220/kernel_operator_vec_vconv_impl.h:
`vconv_f162s4(...)` for `half → int4b_t` (line 932), with
`"CAST_ODD from half to int4b_t is not supported on current device\n"` and
`"illegal input cast mode %d, only support CAST_NONE from int4b_t to half on current device\n"`.

#### Built-in types: `int4b_t` is `[-8, 7]`, and `uint4b_t` is not an A3 type

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/data_structures/builtin_data_types.html

> b8 | bool、int4x2_t、int8_t、uint8_t、fp4x2_e2m1_t、fp4x2_e1m2_t、hifloat8_t、fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t。
> 说明：int4x2_t数据类型由两个四位整型数打包为一个8比特存储单元。
> 表2 数据类型简写: S4 | int4b_t
> 数据类型的产品支持情况：
> Atlas A3系列产品：bool、int4b_t、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、double。
> 表4 整数数据类型取值范围: int4b_t | [-8, 7]

No `uint4b_t`, no `uint4x2_t`. The `[-8, 7]` is the official statement of the range that forces the
`(x & 0x0F) - 8` construction.

The gate in current `asc-devkit` master is wider than the copy quoted in §7.2 —
https://gitcode.com/cann/asc-devkit/blob/master/impl/basic_api/utils/kernel_utils_constants.h
lines 704-719:

```cpp
using int4b_t = integer_sub_type<INT4_BIT_NUM, true>;

#if (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 5162)
using int2b_t = IntegerSubType<INT2_BIT_NUM, true>;
using int1b_t = IntegerSubType<INT1_BIT_NUM, true>;
using uint1b_t = IntegerSubType<INT1_BIT_NUM, false>;
#endif

#if defined(__NPU_ARCH__) &&                                                                         \
    (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113 || __NPU_ARCH__ == 5101 || __NPU_ARCH__ == 5161 || \
     __NPU_ARCH__ == 5165 || __NPU_ARCH__ == 5163 || __NPU_ARCH__ == 5162)
using uint4b_t = IntegerSubType<INT4_BIT_NUM, false>;
using uint3b_t = IntegerSubType<INT3_BIT_NUM, false>;
using uint2b_t = IntegerSubType<INT2_BIT_NUM, false>;
#endif
```

`2201` is in neither list. Searching the whole `impl` + `include` tree: `uint4b_t` returns 4 lines,
`int4b_t` returns 846.

Graph and ACL levels agree that unsigned int4 does not exist as a CANN dtype at all.
https://gitcode.com/cann/metadef/blob/master/inc/external/graph/types.h lines 80-125 define
`DT_INT4 = ::C_DT_INT4`, `DT_UINT1`, `DT_INT2`, `DT_UINT2` — and no `DT_UINT4`;
https://gitcode.com/cann/metadef/blob/master/inc/external/graph/c_types.h has
`C_DT_INT4 = 29,            // int4 type` and `C_DT_UINT1 = 30,           // uint1 type`;
https://gitcode.com/cann/ops-nn/blob/master/docs/zh/context/data_type.md's complete 表1 数据类型简写表
lists `ACL_INT4` and `ACL_UINT1` and no `ACL_UINT4`. CANN defines uint1, int2, uint2, int4 and
deliberately not uint4.

One doc inconsistency to not be misled by: `npu-arch-facts.json`'s
`dtype_product_support.atlas_a2_a3` omits `int4b_t` entirely. That is wrong against the Mmad page
表7, the Matmul page 表2, the builtin-types page, and the `dav_c220` implementation. Trust those four.

### 11.2 The golden reference — CANN's own model of the operator rounds to fp16

Path: `gmm/grouped_matmul_swiglu_quant_v2/tests/st/aclnnGroupedMatmulSwigluQuantV2_A8W4/executor_aclnnGroupedMatmulSwigluQuantV2_A8W4.py`
(415 lines; header `Copyright (c) 2026 Huawei Technologies Co., Ltd.`), plus its sibling
`arch22_atk_aclnnGroupedMatmulSwigluQuantV2_A8W4.json`. Present in the `d6aaa9fe` (2026-10-19)
snapshot and at `ops-transformer` HEAD; absent at `e027a008` (2026-09-01) — see §9.3.

#### 11.2.1 The split, which is pairwise interleaved and says so in one line

```python
def x_INT8_to_x_INT4(x: torch.Tensor):
    M, K = x.shape
    x_High_4bit = torch.floor(x.to(torch.float16) // 16).to(torch.int8)
    x_Low_4bit = torch.bitwise_and(x.view(torch.int16), 0x0F0F).view(torch.int8) - 8
    x_Int4 = torch.empty((2 * M, K), dtype=torch.int8)
    x_Int4[::2, :] = x_High_4bit
    x_Int4[1::2, :] = x_Low_4bit
    return x_Int4
```

`x_Int4[::2] = high`, `x_Int4[1::2] = low`, shape `(2*M, K)`. **Interleaved M-stacking, stated as
plainly as it can be stated.** Note also that the reference does the `//16` in `float16` — exact
for every int8, and the same choice the kernel makes (§2.1's `Muls(0.0625)`).

#### 11.2.2 The recombination, and the `fp16` cast baked into the model

```python
def Custom_MM(x: torch.Tensor, weight: torch.Tensor, weightScale: torch.Tensor, m: int):
    """
    ...
    返回:
        MMOut(fp16): MatMul + perGroup或perChannel反量化结果
    """
    # 使用 int32 精度执行矩阵乘法
    k, n = weight.shape
    MMOut = torch.zeros((m, n), dtype=torch.float16)
    # perGroup 场景
    if len(weightScale.shape) == 2 and weightScale.shape[0] != 1:
        K_group = weightScale.shape[0]
        per_group_ele = k // K_group
        x_grouped = x.view(-1, K_group, per_group_ele).transpose(0, 1)
        weight_grouped = weight.view(K_group, per_group_ele, n)
        c_temp = torch.bmm(
            x_grouped.to(torch.int32), weight_grouped.to(torch.int32)
        ).to(torch.float16)
        for k_idx in range(K_group):
            MMOut += (
                c_temp[k_idx] * weightScale[k_idx].view(1, -1).to(torch.float16)
            ).to(torch.float16)
    # perChannel 场景
    elif len(weightScale.shape) == 1 or (
        len(weightScale.shape) == 2 and weightScale.shape[0] == 1
    ):
        c_temp = torch.matmul(x.to(torch.int32), weight.to(torch.int32)).to(
            torch.float32
        )
        MMOut = (c_temp * weightScale.view(1, -1)).to(torch.float16)
    return MMOut.to(torch.float32)
```

```python
            MMOut = Custom_MM(
                x_INT4[int(start_idx) : int(start_idx + tempV)],
                weight[i],
                weightScale[i],
                tempV,
            )
            MM_Num_Concat = (MMOut[::2] * 16 + MMOut[1::2]) + weightAssistanceMatrix[
                i
            ].view(1, -1)

            PerToken_Quant = MM_Num_Concat * perTokenScale[
                start_idx // 2 : (start_idx + tempV) // 2
            ].view(-1, 1)
            # 将结果分成两部分以应用 SwiGLU 激活函数
            SwiGLU, gate = PerToken_Quant.chunk(2, dim=-1)
            temp = SwiGLU * torch.sigmoid(SwiGLU)  # SwiGLU 激活
            temp = temp * gate  # 与门控值进行逐元素相乘
            # 对输出进行量化
            max_value = torch.max(
                torch.abs(temp), dim=-1
            ).values  # 找到最大绝对值以计算缩放因子
            quantScaleOutput_temp = 127 / max_value  # 计算量化缩放因子
            quantOutput[start_idx // 2 : (start_idx + tempV) // 2] = torch.round(
                temp * quantScaleOutput_temp.reshape(tempV // 2, 1)
            ).to(torch.int8)  # 量化为 int8
            quantScaleOutput[start_idx // 2 : (start_idx + tempV) // 2] = (
                max_value / 127
            )  # 反向量化缩放因子以便后续反量化
```

**`MMOut` is declared `dtype=torch.float16`, every per-group partial is `.to(torch.float16)`
twice over, the per-channel result is `.to(torch.float16)`, and only then is `* 16` applied to
the high half.** The golden reference does not compute the exact integer product; it *models the
fp16 waypoint*, including the fact that the term about to be multiplied by 16 is the one that
passed through fp16 first. §1.5's 「数值与直接INT8×INT4乘法结果一致」 is contradicted by Huawei's
own reference implementation of the same operator, not merely by its kernel.

The `*2` loop bookkeeping, matching §2.2's kernel comment:

```python
        tempV = int(
            (currV - preV) * 2
        )  # 计算当前组的 token 数量 “ * 2 ”是因为 1行Int8--> 2行Int4
```

and the assistance matrix, computed exactly as §4.1 specifies:

```python
    repeat_times = K // weightScale.shape[1] if dequantMode == 1 else 1
    expanded_scale = weightScale.view(E, -1, N).repeat_interleave(repeat_times, dim=1)
    weightAssistanceMatrix = (8 * weight * expanded_scale).sum(dim=1)
```

with the input generator, which tells us the ranges CANN validates over:

```python
def gen_input_data(E, M, K, N, KNum_per_group, dequantMode):
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    weight = torch.randint(-5, 5, (E, K, N), dtype=torch.int8)
    if dequantMode == 1:
        assert K % KNum_per_group == 0, (
            "per-channel&&per-group模式下， K必须为KNum_per_group的整数倍"
        )
        weightScale = (
            torch.randint(-2, 2, (E, K // KNum_per_group, N))
            .to(torch.bfloat16)
            .to(torch.float32)
        )
    elif dequantMode == 0:
        weightScale = torch.randint(-2, 2, (E, N)).to(torch.bfloat16).to(torch.float32)
    xScale = 0.1 * torch.randn(M)
```

Note `weight = torch.randint(-5, 5, ...)` — the validated weight range is `[-5, 4]`, not the full
int4 `[-8, 7]`, and `weightScale` is drawn from the integers `{-2, -1, 0, 1}`. The test is not
stressing the numeric range.

#### 11.2.3 The precision standard — `2**-10`, i.e. the fp16 resolution

`arch22_atk_aclnnGroupedMatmulSwigluQuantV2_A8W4.json`, verbatim:

```json
  "standard": {
   "acc": {
    "single_bm": {
     "type": "high_performance",
     "fp32_error": "2**-10",
     "fp32_eb": "2**-1"
    }
   },
   "perf": "not_key"
  },
```

and the shapes it ships with:

```json
   { "name": "x",                  "dtype": "int8", "shape": [1, 64] },
   { "name": "weight",             "dtype": "int8", "shape": [1, 64, 64] },
   { "name": "weightScale",        "dtype": "fp32", "shape": [1, 64] },
   { "name": "weightAssistMatrix", "dtype": "fp32", "shape": [1, 1], "required": false },
   { "name": "bias",               "dtype": "fp32", "shape": [8, 9],  "required": false },
   { "name": "xScale",             "dtype": "fp32", "shape": [1] },
   { "name": "groupList",          "dtype": "int64","shape": [1] },
   { "name": "dequantMode",   "type": "attr", "range_values": 0 },
   { "name": "groupListType", "type": "attr", "range_values": 1 },
```

`"type": "high_performance"` (CANN's accuracy tiers are `high_precision` / `high_performance`),
relative error `2**-10 ≈ 9.77e-4`, error balance `2**-1 = 0.5`. `"perf": "not_key"`.

### Relevance

- **The int32 variant from §2.4 is legal.** `Matmul` 表2 lists
  `int4b_t | int4b_t | int32_t | int32_t/half` for Atlas A3 — C may be `int32_t`, and the Bias
  column is `int32_t`. So we can instantiate `cT = MatmulType<..., int32_t>`, skip
  `SetQuantVector`, do `16*h + l` in int32 (exact), fold `8*sum_k w[k,n]` as an **integer** through
  the bias slot, and apply all scales once in fp32. None of CANN's four A8W4 kernels does this; the
  API permits it; §6.1 shows the sibling MSD family already doing it with int8 operands. This is now
  a design choice with a known price (§8's 2x mm workspace), not an unknown.
- **`MmadParams.k % 2 == 0` is mandatory for `mad_s4` and appears only in the source.** Our
  `quantGroupSize` and any K-tiling must keep every `Mmad`'s K even. The `Matmul`-level restatement
  is 「A矩阵为int4b_t数据类型且不转置时， singleCoreK 必须是偶数」. The same parity rule applies to
  `Cast`'s `count` and to `mask`.
- **`int4b_t`'s C0_size is 64**, against 16/32 for wider types — this changes every NZ shape
  computation. It is why §4's NZ weight shape is `{(E, N/64, K/16, 16, 64)}`.
- **M = 1 silently switches `Mmad` to GEMV mode and reinterprets L0A as ND**, and `disableGemv` is
  950-only. At batch=1 after M-doubling we are at M = 2 or more per `Mmad`, so we stay out of it —
  but a single-token no-speculation path would hit it, and the layout assumption changes under us
  with no error. Worth a guard.
- **PTO cannot express int4 on A2A3.** If any part of the pipeline is routed through PTO tiles, the
  W4A8 GEMM cannot be. That is a toolchain-level constraint.
- **There is no `uint4b_t` on 2201 at any level** — not in AscendC (arch-gated), not in metadef
  (`DT_UINT4` does not exist), not in ACL (`ACL_UINT4` does not exist). The offset-fold is the only
  route, confirmed three ways.
- **Use CANN's golden as a *bug* oracle, not an *accuracy* oracle.** If we build the exact int32
  variant, it will differ from `executor_aclnnGroupedMatmulSwigluQuantV2_A8W4.py` by roughly the
  `2**-10` the JSON allows — and ours will be the more accurate one. For bring-up, port that
  reference verbatim (it is the fastest way to a known-good comparison); for the accuracy claim,
  write a second reference in int64 and compare all three.
- **CANN's own validated weight range is `[-5, 4]` with integer scales in `{-2,-1,0,1}`, at
  `M=1, K=64, N=64`.** Nothing about the published test justifies confidence at full int4 range or
  at our shapes.

---

## 12. THE HEADLINE — an exact int32 MSD already exists, and it deletes the auxiliary matrix

URL: https://github.com/vllm-project/vllm-ascend/pull/17201
Title: "[Feature][MoE] Add A3 MSD GMSQ with direct ND/NZ weights"
Author: `grandolantow`
Created: 2026-09-22. Status as of 2026-10-20: **OPEN (draft)**.

Read §2.4 first, then this. §2.4 proposed, as a hypothesis for us to measure, keeping `int32` out
of the cube so the `16*high + low` recombination is exact. This PR has done it, on A3, with
measured numbers and a 66/66 golden-check pass — and it goes one step further than we had
imagined: it **deletes the offline `8*sum(w)` auxiliary matrix** and synthesizes the correction
on-chip as a single extra matmul row.

### 12.1 `int32` out of the cube, stated in the type aliases

`csrc/gmsq_situ_quant/op_kernel/exact_msd.h`, verbatim:

```cpp
using MsdA = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
                                AscendC::int4b_t, false>;
using MsdB = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
                                AscendC::int4b_t, false>;
using MsdNzB = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::NZ,
                                  AscendC::int4b_t, false>;
using MsdC = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t, false>;
```

```cpp
    conf.basicK = MSD_BASE_K; // same L0 operand bytes as A8W8 K128.
```

`MsdC` is `int32_t`. Against §2.0's `yType = ... half`, §3.2's `cT = ... half`, §5.1's
`GlobalTensor<half> mmOutGlobal_` and §10's `cT = MatmulType<..., half>` — all four CANN
implementations. The `Matmul` 表2 row in §11.1 said this was permitted; this is it instantiated.

### 12.2 The correction as one extra matmul row of packed `-8`

Instead of consuming a precomputed `(E, N)` fp32 matrix, the kernel appends a row whose every
int4 element is `-8`, and lets the cube compute `(-8 * ones) @ W` for it:

```cpp
    __aicore__ inline void PackCorrectionTile(int32_t slot, int32_t valid,
                                            int32_t k, int32_t len)
    {
        auto out = outQueue_.AllocTensor<int8_t>();
        // Two signed INT4 -8 values per byte. cAux = (-8 * ones) @ W.
        AscendC::Duplicate(out.ReinterpretCast<int16_t>(), static_cast<int16_t>(-30584), len / 4);
```

`-30584` as `int16_t` is `0x8888` — four packed signed int4 `-8` values. The row count becomes:

```cpp
    const int32_t rawRows = 2 * valid + 1;
```

**`2M + 1`, not `2M`.** And the reconstruction, verbatim, with the comment that is the single most
useful paragraph of engineering prose in this entire file:

```cpp
    __aicore__ inline void Reconstruct(int32_t slot, int32_t row, int32_t valid,
        int32_t expert, int32_t col, LocalTensor<float> dst)
    {
        const int64_t base = static_cast<int64_t>(slot) * MSD_RAW_ROWS * N_ + col;
        ReadRaw(base + (2 * row + 1) * N_, dst);
        ReadRaw(base + 2 * valid * N_, t_);
        AscendC::Sub(dst, dst, t_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        ReadRaw(base + 2 * row * N_, t_);
        AscendC::Muls(t_, t_, 16.0f, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, dst, t_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        // low-aux is a dot product with unsigned low nibbles (magnitude <=120K).
        // Both 16*high and the final A8W4 dot have magnitude <=1024K.
        // K<=16384 keeps EVERY integer intermediate in [-2^24,2^24].
        // These FP32 operations reconstruct the INT32 result exactly; no
        // partial channel scaling or FP16 cast has occurred.
```

Read the arithmetic carefully, because the sign is the opposite of §3's:

- row `2r` = `high(r) @ W`, row `2r+1` = `low_stored(r) @ W` with `low_stored = (x & 0xF) - 8`,
  row `2*valid` = `(-8 * ones) @ W` = `-8 * sum_k W[k,n]`.
- `dst = P_low - P_aux` = `sum_k ((x&0xF)-8) W + 8 sum_k W` = `sum_k (x & 0xF) * W`.
- `dst += 16 * P_high`, giving `sum_k (16*floor(x/16) + (x&0xF)) W` = `sum_k x W`, exactly.

The correction is the *same* `8*sum(w)` quantity, but computed by the cube at run time against
unscaled int4 weights, so it is an **integer**, so it can be combined with the partial products
before any scale is applied — which is precisely what makes the whole reconstruction exact. The
offline `(E, N)` fp32 matrix cannot do this, because by construction it already carries
`weightScale` (§4.1) and therefore has to be added after the dequant (§3.6).

And the bound argument is spelled out: `|low-aux| <= 120K`, `|16*high|` and the final dot
`<= 1024K`, so for `K <= 16384` every integer intermediate stays inside `[-2^24, 2^24]` — the
range where fp32 represents integers exactly. **That is why fp32 vector ops are allowed to carry
an exact int32 result**: not because fp32 is exact in general, but because the magnitudes are
bounded below fp32's 24-bit integer limit. Clean, checkable, and K-dependent.

The Python side passes an empty list where the auxiliary matrix used to go
(`vllm_ascend/quantization/methods/w4a8/w4a8.py`):

```python
                hidden_states, swiglu_out_scale = gmsq_op(
                    x=hidden_states,
                    weight=gmsq_weights,
                    weight_scale=gmsq_scales,
                    x_scale=pertoken_scale.reshape(-1).to(dtype=torch.float32).contiguous(),
                    group_list=group_list,
                    weight_assist_matrix=[],
```

### 12.3 What the PR claims, verbatim

> Add an A3 fused grouped matmul + SiTU + per-token INT8 quantization operator using exact A8 decomposition and INT4 Matmul. Packed ND and native packed INT8 NZ weights are consumed in their existing storage, without a full W4-to-W8 weight replica. Build/runtime integration is adapted from #16655, which is not yet merged into main, so this PR includes the required sidecar and W4A8 MoE dispatch integration.
>
> - Read ND directly as ND and native NZ directly as NZ. Validate the packed carrier's dtype, physical geometry and strides; retain expert views without cat/copy.
> - Reconstruct the integer dot product before the existing FP16 channel-scaling boundary, then apply SiTU and dynamic INT8 quantization.
> - Parse routing counts on device on each invocation/replay. Normalize int64/int32/float32 group lists without host value reads.
> - Keep graph-visible activation/accumulator scratch stable per device/stream/K/N. Its size is independent of expert count and layer count for a shared stream/K/N.
> - Accept runtime E/K/N subject to implementation limits, rather than profiling-shape whitelists: E=1..128, K a positive multiple of 64 up to 16384, N a positive multiple of 256 subject to the UB budget, and at most 256 padded M blocks. Support K packing chunks and N tails.
> - Build the new native entry only for A3. Preserve the existing split path when disabled, unavailable, or rejected by the framework's metadata guard. GMM2 is unchanged.

> For K=3584/N=6144, the primary scratch is 54.71875 MiB (packed activation + INT32 accumulation). The weight-replica allocation is zero, compared with E*K*N bytes for a fully expanded W8 representation (2.296875 GiB at E=112). These figures exclude outputs, pointer metadata and framework/system allocations.

Measured (E=14 / K=3584 / N=6144, capacity 28672, median NPU Graph Event time per operator call,
50 graph replays per batch, five batches), verbatim:

> | Case | Valid rows | ND (us) | Native NZ (us) | NZ / ND |
> | --- | ---: | ---: | ---: | ---: |
> | 1 | 320 | 235.37 | 167.37 | 0.711 |
> | 2 | 260 | 242.50 | 177.28 | 0.731 |
> | 3 | 320 | 260.27 | 187.17 | 0.719 |
> | 4 | 320 | 294.33 | 206.60 | 0.702 |
> | 5 | 400 | 308.68 | 224.35 | 0.727 |
> | 6 | 500 | 325.34 | 234.87 | 0.722 |
> | 7 | 600 | 341.40 | 252.78 | 0.740 |
> | 8 | 800 | 373.88 | 284.01 | 0.760 |

> Geometric mean NZ/ND: 0.7263.

> 66/66 CPU-golden + eager/graph checks

> Supported A3 W4A8 SiTU workloads use the fused implementation when VLLM_ASCEND_ENABLE_GMSQ_SITU=1 (default).

### 12.4 Its NZ layout note, which is reusable hardware documentation

```cpp
// Native packed INT8 NZ is [N/64,K/16,16,32] BYTES, not the K-block-major
// layout assumed by the old fallback's UnpackBlockNz. For logical W4 (k,n):
//   byte = ((n/64)*(K/16)+k/16)*512 + (k%16)*32 + (n%64)/2.
// Reinterpreting each byte as its low/high INT4 pair gives precisely the
// Matmul INT4 NZ layout [N/64,K/16,16,64]. The byte order does not change.
// Native ACL physical-memory reads and CANN 9.1 CopyNZ2NZImpl confirm this
// equivalence. The ordinary NZ Matmul loader can consume the original weight
// directly; no callback, L1 reorder, UB staging or GM weight copy is needed.
```

Its own nibble split is the same algebra with the signs rearranged so the `Muls` can be fused:
`Muls(high, low, 0.0625f)` + `CAST_FLOOR`, then `Muls(high, high, -16)`, `Add`, `Adds(low, low, -8)`.

### Relevance

This is the most actionable item in the file, on four counts.

1. **§2.4 is settled, not hypothesised.** `MsdC = MatmulType<..., int32_t>` with `int4b_t` operands
   compiles and runs on A3 and passes 66/66 golden checks. We can write the exact variant.
2. **The correction row beats the offline matrix, and the reason is structural.** Appending one row
   of packed `0x8888` costs `1/(2M+1)` of the GEMM — at `2M = 16` that is 5.9%, at `2M = 2` it is
   33% — and in exchange the correction becomes an *integer* available before any scaling, which is
   the only way to make the recombination exact. At batch=1 decode with `2M` small that percentage
   is the worst it ever gets, so for us the trade is: ~1/3 more cube work on a GEMM that is
   weight-bandwidth-bound anyway (the extra row reads no new weight bytes — it reuses the same `W`
   tile already in L1/L0B), against an exact result and one fewer offline tensor to ship. On a
   memory-bound decode GEMM the extra row is close to free. **This is the design we should copy.**
   It also means we do not need `scale_bias` in our checkpoint at all (§13).
3. **The exactness argument is a bound, and the bound is `K <= 16384`.** `|16*high + low| <= 1024K`
   and fp32 is integer-exact to `2^24 = 16.7M`, so `K <= 16384` is safe with a factor of ~1 margin.
   GLM-5.2's expert K is within that. Above it, the fp32 reconstruction stops being exact and the
   recombination must move to int32 arithmetic. Record the bound; do not inherit it blindly.
4. **It is a draft PR, not merged.** Treat the code as a design we verified by reading, not as
   shipped. The measured 0.7263 NZ/ND ratio is its own ND arm, not a comparison against the CANN
   operator, and `valid rows` 260-800 is far above our regime.

One caution on reading it: `VLLM_ASCEND_ENABLE_GMSQ_SITU` and "SiTU" are this PR's naming; the
activation is the same SwiGLU as everywhere else in this file.

---

## 13. The offline side — where `8*sum(w)` is actually computed, the GLM-5 recipe, and the accuracy numbers

### 13.1 msModelSlim: the tensor is called `scale_bias`, and `process_scale` computes it

Upstream: `https://gitee.com/ascend/msit/tree/master/msmodelslim` (Apache-2.0, `msit-8.1.0 TR5版本`)
and `https://gitcode.com/Ascend/msmodelslim`. Read via the mirror
`https://github.com/verylucky01/MindStudio-ModelSlim` at commit
`ad2c44a33fa26ca753465b1a0cbab53123c02310` (Fri Aug 21 17:02:38 2026 +0800).

> Searching msmodelslim for `msd`, `assistance`, `auxiliary`, `辅助矩阵`, `weight_sum`, `sum_w`,
> `bias_correction`, `offset_fold` returns **zero** code hits. The kernel-side vocabulary does not
> exist in the quantizer. The name to search is `scale_bias`.

`msmodelslim/format/common/pack.py` (identical copies at
`msmodelslim/core/quant_service/modelslim_v1/save/utils/pack.py` and
`msmodelslim/pytorch/llm_ptq/llm_ptq_tools/save/complex_quantifier.py` lines 192-211), verbatim
and complete:

```python
def process_scale(name, bias, tp_num):
    """
    Pack int4 weight to int8 weight
    @param name: 输入tensor名
    @param bias: sum 前bias
    @param tp_num: 推理时tp数
    @return: bias, fp32格式gmm算子所需的偏置量
    """
    if any(char in name for char in ['up_proj', 'gate_proj', 'q_proj', 'k_proj', 'v_proj']):
        up_bias = bias
        up_bias = 8 * up_bias.sum(dim=1, keepdim=True)
        bias = up_bias

    elif any(char in name for char in ['down_proj', 'o_proj']):
        pre_shape = bias.shape[0]
        sum_shape = bias.shape[1] // tp_num
        down_bias = bias.reshape(-1, sum_shape)
        down_bias = 8 * down_bias.sum(dim=1, keepdim=True)
        bias = down_bias.reshape(pre_shape, -1)
    return bias
```

The only production call site, `msmodelslim/core/quant_service/modelslim_v1/save/ascendv1.py`
lines 423-438, verbatim:

```python
    def on_w4a8_dynamic(self, prefix: str, module: qir.W4A8DynamicFakeQuantLinear):
        self.update_quant_type("W4A8_DYNAMIC")

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            weight = module.weight.to(torch.float32)
            deq_weight = weight.T * module.weight_scale
            scale_bias = process_scale(prefix, deq_weight.T, 16)
            self.write_tensor(prefix + ".weight", "W4A8_DYNAMIC", w4a8_pack_int4(module.weight.to(torch.int8)))
            self.write_tensor(prefix + ".weight_scale", "W4A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(
                prefix + ".weight_offset", "W4A8_DYNAMIC", torch.zeros_like(weight_scale).to(torch.float32)
            )
            self.write_tensor(prefix + '.scale_bias', "W4A8_DYNAMIC", scale_bias.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "FLOAT", module.bias.to(torch.float32))
```

Pinned answers: the expression is `8 * (weight * weight_scale).sum(dim=1, keepdim=True)`; it
**does** include the per-channel scale (`deq_weight = weight.T * module.weight_scale` is applied
*before* the sum); the dtype is fp32 at write time and the sum runs in fp32 because
`weight = module.weight.to(torch.float32)`; the tensor name is `<prefix>.scale_bias`.
`weight_offset` is written as `torch.zeros_like(weight_scale)` — symmetric weights confirmed in
code, not merely in the docs. **`tp_num` is the literal `16` at the call site**, and only affects
the `down_proj`/`o_proj` branch, where the sum is a row-block sum for TP sharding rather than a
full-K sum.

The two-level (QServe-style) variant,
`msmodelslim/pytorch/llm_ptq/llm_ptq_tools/save/complex_quantifier.py` lines 338-346 — both scale
levels folded into the same correction:

```python
        if self.is_new_version and model_quant_type in [QuantType.W4A8_DYNAMIC]:
            weight = save_quant_weight.reshape(-1, module.cfg.group_size).to(torch.float32)
            second_scale = weight_scale[1].to(torch.bfloat16).reshape(-1, 1)
            first_deq_weight = (weight * second_scale).reshape(ori_shape)
            second_deq_weight = first_deq_weight * weight_scale[0].to(torch.bfloat16)
            bias = second_deq_weight
            scale_bias = process_scale(name, bias, 16)
            save_quant_weight = _w4a8_pack_int4(save_quant_weight)
```

> \# W4A8_DYNAMIC 有两种量化模式，一种分阶段量化（将浮点先量化成int8，然后再量化成 int4），一种直接量化（将浮点直接量化成 int4）

The int4 packing convention, `msmodelslim/format/common/pack.py` lines 25-70 — **even element in
the low nibble, odd element in the high nibble**, with `n dimension should be even` asserted:

```python
    weight = weight.reshape(-1, 2)
    weight0 = weight[:, :1]
    weight1 = weight[:, 1:]

    weight1_4 = torch.bitwise_left_shift(weight1, 4)
    weight2_4 = weight0 & 0b00001111

    weight_add = torch.bitwise_or(weight1_4, weight2_4)
```

and the dtype assertion from `test/ST/llm_ptq_w4a8_per_token/run.py` lines 438-441:

```python
            scale_bias = tensors[f"{name}.scale_bias"]
            assert scale_bias.dtype == torch.float, \
                f"Expected float for {name}.scale_bias, got {scale_bias.dtype}"
```

### 13.2 The framework side: `w13_scale_bias` / `w2_scale_bias`, and per-group is rejected

`https://github.com/vllm-project/vllm-ascend`, `vllm_ascend/quantization/methods/w4a8/w4a8.py`,
main at `8237ee6d690d1399bc6eb48ef5977ac2bd470f3f` (2026-10-20 15:35:33 +0800), lines ~347-359:

```python
        def update_bias_compressed_tensors(weight: torch.Tensor, scale: torch.Tensor):
            group_num, k, n = weight.shape
            scale = scale.transpose(1, 2).contiguous()
            scale = scale.reshape(group_num, -1, n)
            bias = 8 * (weight.to(torch.float32) * scale).sum(axis=1)
            return bias

        w13_bias = update_bias_compressed_tensors(layer.w13_weight.data, layer.w13_weight_scale.data)
        w2_bias = update_bias_compressed_tensors(layer.w2_weight.data, layer.w2_weight_scale.data)
        w13_scale_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
        layer.register_parameter("w13_scale_bias", w13_scale_bias)
        w2_scale_bias = torch.nn.Parameter(w2_bias, requires_grad=False)
        layer.register_parameter("w2_scale_bias", w2_scale_bias)
```

`AscendW4A8DynamicFusedMoEMethod`'s docstring, verbatim — the shapes on both sides of loading:

> ```
>     - Each linear also has ``weight_scale`` and ``weight_offset``:
>       ``torch.float32``, ``[out_features, 1]``.
>     - Each linear also has ``scale_bias``: ``torch.float32``,
>       ``[moe_intermediate_size, 1]`` for ``gate_proj`` and ``up_proj``,
>       and ``[hidden_sizes, 16 // tp_size]`` for ``down_proj``.
> ...
>     - ``w13_scale_bias``: ``torch.float32``,
>       ``[num_experts, 2 * moe_intermediate_size]``.
>     - ``w2_scale_bias``: ``torch.float32``,
>       ``[num_experts, hidden_sizes]``.
>     - ``w13_weight_scale``: ``torch.int64``,
>       ``[num_experts, 2 * moe_intermediate_size]``.
>     - ``w2_weight_scale``: ``torch.int64``,
>       ``[num_experts, 1, hidden_sizes]``.
> ```

The `[hidden_sizes, 16 // tp_size]` is the direct footprint of `process_scale`'s hardcoded
`tp_num = 16`; the same file allocates
`param_dict["w2_scale_bias"] = torch.empty(num_experts, hidden_sizes, 16 // self.tp_size, dtype=torch.float32)`.
And from `__init__`:

```python
        group_size = vllm_config.quant_config.quant_description.get("group_size", 0)
        if group_size > 0:
            raise ValueError(
                "The current weights use Per‑Group quantization, which is no longer supported. Please "
                "switch to Per‑Channel quantized weights."
            )
```

Two more independent restatements of the same formula.
`https://github.com/sgl-project/sgl-kernel-npu`, `tests/python/deepep/test_fused_deep_moe_a3_w4a8_acc.py`:

```python
def compute_bias(
    weight_packed: torch.Tensor, scale_packed: torch.Tensor, pack_dim: int = 1
) -> torch.Tensor:
    """Compute bias = sum(unpacked_weight * scale, dim=0) * 8."""
    if weight_packed.dtype != torch.int32:
        raise TypeError(
            f"compute_bias expects int32 packed weight, got {weight_packed.dtype}"
        )
    w_int4 = unpack_int4(weight_packed, pack_dim).float()
    scale_fp32 = extract_float_from_int64(scale_packed)
    return (w_int4 * scale_fp32).sum(dim=0) * 8.0
```

`https://github.com/vllm-project/afd-plugin`, `tests/npu/test_async_cam_layered_w4a8.py:47` and
`tests/npu/async_cam_runtime.py:69`:

```python
        dequant = values.float() * scale.repeat_interleave(k // groups, dim=1)
        compensation = 8 * dequant.sum(dim=1)
```
```python
        bias = (8 * dequant.sum(1)).to(device)
```

### 13.3 CANN's second official golden — `catlass/examples/35_w4a8_grouped_matmul_msd`

Repository: `https://gitcode.com/cann/catlass`, clone commit
`37cedeca455d0970f1724b0c23ab3d3fa4bdb08d` (Tue Oct 20 15:14:18 2026 +0800). File header
`Copyright (c) 2025 Huawei Technologies Co., Ltd.`, CANN Open Software License Agreement Version 2.0.
The directory contains **only `gen_data.py`** — no `.cpp`, no README (last commit `5f10a480` by
`cann-robot`, "用pre-commit刷新主要文件的格式", PR `!874`). The golden generator is the entire
public specification in that repo. Lines 85-147, verbatim:

```python
    # golden函数
    def msd_cpu(x, weight, bias, groupList, scale, perTokenScale):
        g, k, n = weight.shape
        quantGroupNum = scale.shape[1]
        kGroupSize = k // quantGroupNum
        xSplit = np.split(x, groupList * 2, axis=0)
        perTokenScaleSplit = np.split(perTokenScale, groupList, axis=0)
        weightGroup = weight.reshape(g, quantGroupNum, kGroupSize, n).astype(np.int32)
        mmOut = []
        atomic = np.float16
        for i in range(g):
            xi = xSplit[i].reshape(-1, quantGroupNum, kGroupSize).astype(np.int32)
            mmi = np.zeros([xi.shape[0], n], dtype=atomic)
            for j in range(quantGroupNum):
                # numpy int32矩阵乘法非常慢，此处用float32的矩阵乘法代替
                mm = np.matmul(
                    xi[:, j, :].astype(np.float32),
                    weightGroup[i, j, ...].astype(np.float32),
                )
                mm = mm.astype(np.float32) * scale[i, j].reshape(1, -1)
                mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)
            mmi = mmi.reshape(-1, 2, n).astype(np.float32)
            mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias[i].reshape(1, n)
            mmi = mmi * perTokenScaleSplit[i]
            mmOut.append(mmi)
        golden = np.concatenate(mmOut, axis=0)
        return golden.astype(np.float32)
```

and lines 122-166:

```python
    # 原始模型输入
    groupList = generate_group_list(m, g)
    x = np.random.randint(-100, 100, [m, k]).astype(np.int8)
    weight = np.random.randint(-8, 8, [g, k, n]).astype(np.int8)
    scale = np.random.normal(0, 0.01, [g, 1, n]).astype(np.float32)
    perGroupScale = np.random.normal(0, 12, [g, quantGroupNum, n]).astype(np.float32)
    perTokenScale = np.random.normal(0, 0.01, [m, 1]).astype(np.float32)

    # 计算精度补齐bias
    weightHigh = weight.astype(np.float32).reshape(
        [g, quantGroupNum, -1, n]
    ) * perGroupScale.reshape([g, quantGroupNum, 1, n])
    weightHigh = weightHigh.reshape([g, k, n])
    bias = 8 * (weightHigh * scale).sum(axis=1)

    # 计算per group和per channel的融合量化系数
    scaleFloat32 = scale * perGroupScale
    # 输入x高低位拼接
    xC12 = np.concatenate(
        [x.reshape(m, 1, k) // 16, (x.reshape(m, 1, k) & 0x0F) - 8], axis=1
    ).reshape(2 * m, k)
```

`examples/36_w4a8_matmul_msd/gen_data.py` is the same algorithm without the expert dimension:
`bias = 8 * (weightHigh * scale).sum(axis=0)` and
`mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias.reshape(1, n)`.

**`atomic = np.float16`** — a *second* official Huawei golden reference that bakes the fp16
waypoint in, independently of §11.2's. Huawei's own Chinese name for the correction here is
「精度补齐bias」, not 辅助矩阵 — a fourth name for the same quantity.

### 13.4 The GLM-5 W4A8 recipe, verbatim

`msmodelslim/lab_practice/glm_5/glm_5_w4a8.yaml` — `config_id: glm_5_w4a8`, `score: 90`, verified
for GLM-5 on vLLM-Ascend and SGLang, Atlas A2 and A3 inference:

```yaml
default_w4a8_dynamic: &default_w4a8_dynamic
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int4"
    symmetric: True
    method: "ssz"
```

**No `group_size` key at all — W4A8_DYNAMIC is per-channel only.** Pipeline order is
`quarot` → `flex_awq_ssz` (on `up-down` subgraphs, `ext: {step: 10}`) → `flex_smooth_quant` →
`group` of `linear_quant`. Layer assignment: `*self_attn*` → W8A8 dynamic, excluding
`model.layers.78*`, `*kv_b_proj`, `*wk`, `*weights_proj`; `*mlp*` → W8A8 dynamic, excluding
`*gate` and `*mlp.experts.*`; `*mlp.experts.*` → **W4A8 dynamic**, excluding `model.layers.78.*`.
`flex_awq_ssz` excludes `model.layers.0.*`, `model.layers.1.*`, `model.layers.2.*`,
`*mlp.shared_experts.*`. `lab_practice/deepseek_v3_2/deepseek_w4a8.yaml` is structurally identical
for DeepSeek-V3.2-Exp, additionally pinning `model.layers.61.mlp.experts.*` to W8A8.

The older per-group path, `example/DeepSeek/quant_deepseek_w4a8.py` lines 229-272:

```python
    w4a8_pertoken_config = QuantConfig(
        a_bit=8,
        w_bit=4,
        w_sym=True,
        dev_id=model.device.index,
        dev_type='npu',
        is_lowbit=True,
        mm_tensor=False,
        group_size=256,
        is_dynamic=True,
        do_smooth=False,
        use_sigma=True,
        open_outlier=False,
        disable_names=disable_names,
    )

    # w4a8_dynamic 和 w8a8 & w8a8_dynamic 混合量化配置
    mix_cfg = {
        "model.layers.[012].mlp.gate_proj": "w8a8_dynamic",
        "model.layers.[012].mlp.down_proj": "w8a8_dynamic",
        "model.layers.[012].mlp.up_proj": "w8a8_dynamic",
        "model.layers.[0-9]*.self_attn.*": "w8a8",
        "model.layers.[0-9]*.mlp.shared_experts.*": "w8a8_dynamic",
        "model.layers.[0-9]*.mlp.experts.*": "w4a8_dynamic",
        "mtp_decoder.mlp.shared_experts.*": "w8a8_dynamic",
        "mtp_decoder.mlp.experts.*": "w8a8_dynamic",
    }
```

> \# w4a8 混合量化中 MindIE 要求 description 中的 model_quant_type 为 W8A8_DYNAMIC

And the knowledge-base spec table,
`docs/zh/knowledge_base/quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md`:

> | 量化粒度 | 权重 per-channel；激活 per-token | 权重逐输出通道共享 scale；激活逐 token 共享 scale |
> | 对称性 | 对称 | 仅 scale，无 offset |

with its 使用限制, which names our exact problem as a limitation of the method:

> - 权重精度风险：4bit 权重仅16档，对离群值敏感，需配合离群值抑制或更细分组。
> - **INT4×INT8 混合位宽 GEMM 需要目标硬件算子库支持，否则收益打折。**

> 本模式：优势是权重压到 4bit、访存降至 FP16 的 1/4，权重侧压缩更彻底；劣势是权重仅16档、有精度风险，且权重/激活位宽不一致时算子需混合处理。

### 13.5 The accuracy numbers that actually exist

**Correction to a lead that does not pan out.** The CSDN article
`https://blog.csdn.net/2503_93041010/article/details/150558465` (2025-08-21 10:51:11, 昇腾MindStudio)
is a verbatim repost of `https://www.hiascend.com/developer/techArticles/20250728-1` (2025/07/28),
and **contains no benchmark table**. Its entire accuracy claim is one sentence:

> 使用msModelSlim提供的量化脚本对DeepSeek R1 671B模型进行W4A8混合量化，原始模型BF16权重数据占内存1.3T，量化后模型权重数据仅占内存340GB。同时，对量化后的模型进行了精度验证，测试结果显示在BoolQ、CEval和GSM8K等多个基准数据集上，量化模型相较于原始模型的平均精度损失控制在1%以内，实现了精度和量化效率的良好平衡。

No per-dataset figure, no W8A8-vs-W4A8 side by side, on either page. The same article's recipe
prose is worth having though:

> 路由专家（Routed Expert）采用W4A8动态量化配置，这样看，误差来源主要是权重，因此权重做Qserve两级量化对权重第一级INT8 per-channel量化，第二级INT4 per-group量化，可以通过较小的量化粒度来保持一个较好的量化精度，激活保持per-token对称量化（A8）
>
> 注：per-tensor量化又称为per-layer量化，指对整个linear的weight tensor使用一个scale值进行量化，per-channel量化指对一个channel使用一个scale进行量化，per-group量化指对权重进行分组量化，每组group_size个元素（group size一般为64或128）。

Tables that do exist:

**(a)** `https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/DeepSeek-V4-Pro.md`
§7.1, verbatim:

> | dataset | version | metric | mode | vllm-api-general-chat | note |
> | ----- | ----- | ----- | ----- | ----- | ----- |
> | GPQA | - | accuracy | gen | 91.41 | 0831 official W4A8 weight with DSpark enabled |
> | GPQA | - | accuracy | gen | 89.90 | 2 Atlas 800 A3 (128GB × 8) |
> | GSM8K | - | accuracy | gen | 96.21 | 2 Atlas 800 A3 (128GB × 8) |
> | GSM8K | - | accuracy | gen | 97.54 | 950PR&950DT Products, w4a4c8 |

**(b)** msmodelslim `docs/zh/best_practices/deepseek_v4_pro_w4a8_new_llm_quantization_case.md`
§6.3, verbatim:

> | 数据集 / 任务 | 样本数 | 基线（论文） | 待验（W4A8 量化） | 差值 | 结论 |
> | --- | --- | --- | --- | --- | --- |
> | GPQA | 全量 | 89.1 (Accuracy) | 89.9 (Accuracy) | +0.8 | 量化误差 < 1%，通过 |
>
> 精度结论：DeepSeek-V4-Pro W4A8 混合量化后 GPQA 实测 89.9，相对论文基线 89.1，得分提升 0.8，满足精度验收标准（相对损失 < 1%）。
> 说明：本案例以论文基线作为对比基准，非浮点服务实测值。量化接入案例聚焦模型接入与量化流程验证，不单独拉起浮点推理服务做对比评测。

(Environment: Atlas 800I A3, `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`, CANN-9.0.0, TP16+DP2,
AISBench `--datasets gpqa_gen`. The caveat in the 说明 matters: the baseline is a *paper* number,
not a measured fp run.)

**(c) A regression with real numbers on both sides.**
`https://github.com/vllm-project/vllm-ascend/issues/8179`,
"[Bug]: Qwen3.5-397B-A17B-w4a8权重的精度较原始的全量权重下降超过正常范围", opened 2026-04-13 by
`kaerlong`, CLOSED. Body verbatim:

> 我们在单机16卡A2的场景下，进行Qwen3.5-397B-A17B-w4a8的推理部署，并使用agent的离线验证脚本进行精度验证（验证动作action和鼠标点击位置grounding的正确性）。经过社区量化版本和原始全量版本的验证对比，发现grounding正确性有超出正常的差异，w4a8的grounding准确率为93.34%，而全量为94.86%，下降达1.5%。
> 还有一个问题，加了QuaRot离群值抑制并重新量化后，8卡条件下mtp的采纳率非常的低（小于40%)

`linfeng-yuan` (maintainer), 2026-04-13:
`There's a accuracy regression report for Qwen3.5-397B-A17B-w4a8. Please check that whether it is an expected behaviour. Additionally, user also reports an acceptance rate regression with QuaRot options.`
`LoganJane`, 2026-06-11: `This issue has been resolved in the latest version. Please use the latest release or rc version`. Closed 2026-08-11 as stale.

**(d) Operator-level tolerance**, the closest public proxy to "is the split lossy".
`https://github.com/sgl-project/sgl-kernel-npu/blob/main/tests/python/deepep/test_fused_deep_moe_a3_w4a8_acc.py`:

```python
DIFF_THRESHOLDS = {
    "w4a8": 5e-3,
    "w8a8": 2e-3,
}
```

**(e)** From sglang PR 32601 (MXFP W4A8, *not* the INT4 MSD path), verbatim:

> | Configuration | Accuracy (%) | Δ vs bf16 |
> | --- | ---: | ---: |
> | bf16 (baseline) | 93.33 | — |
> | Online W4A8 | 93.63 | +0.30 pp (+0.32%) |
> | Offline W4A8 | 93.71 | +0.38 pp (+0.41%) |
>
> Note: Single-run GSM8K exact match. Δ = quantized − baseline.

### Relevance

- **We may not need `scale_bias` in our checkpoint at all.** §12's correction-row design computes
  the same quantity on-chip, as an integer, exactly. If we take that route, the only W4A8 tensors
  we need from msModelSlim are `weight` (packed int4), `weight_scale` and a zero `weight_offset`.
  If we take the offline route instead, the tensor to read is `<prefix>.scale_bias`, fp32, and the
  formula is `8 * (weight * weight_scale).sum(dim=1, keepdim=True)`.
- **`process_scale`'s `down_proj` branch is NOT a full-K sum.** It reshapes to
  `bias.shape[1] // tp_num` with `tp_num` hardcoded to 16, sums within each block, and returns
  shape `[out, 16]` — a *per-TP-shard* correction, so each rank adds only its own shard's term. Our
  TP16 matches the hardcoded 16 by luck, not by design. If our TP ever changes, this tensor is
  wrong and will be silently wrong. `w2_scale_bias`'s `[hidden_sizes, 16 // tp_size]` shape is the
  framework compensating for the same hardcode. **This is the most likely source of a subtle
  numerical bug when loading a public W4A8 checkpoint.**
- **Weights are symmetric, per-channel, and `weight_offset` is written as literal zeros** — in
  msModelSlim's code, in the GLM-5 recipe (`symmetric: True`, `scope: "per_channel"`, no
  `group_size`), in the knowledge base (「仅 scale，无 offset」), and vllm-ascend hard-*errors* on
  per-group. So §3.5's whole `withOffset` / `xRowSum` branch and §4's `offset != null` path are
  dead code for us. Skip them.
- **The GLM-5 recipe is published and names our layers**: `*mlp.experts.*` → W4A8 dynamic, with
  `model.layers.78.*` excluded and attention/shared-experts/gate left at W8A8. That tells us
  exactly which GEMMs in GLM-5.2 need the MSD path and which do not — and it is a minority of them.
- **No public source reports perplexity for the MSD split in isolation.** The honest summary of all
  the accuracy evidence: W4A8 end-to-end costs ~0-1.5% on task metrics (GPQA +0.8, grounding -1.5),
  and that is dominated by int4 *weight* quantization, not by the decomposition. The decomposition's
  own cost is the fp16 waypoint, which no public benchmark isolates, and which §12 shows is
  avoidable.
- **Huawei's own `catlass` golden and its ST golden independently bake in `float16`.** Two separate
  reference implementations, two separate teams, same concession. §1's exactness claim is prose.
- **The weight ranges CANN validates over**: `catlass` uses `randint(-8, 8)` for weight (full int4)
  and `randint(-100, 100)` for x, with `scale ~ N(0, 0.01)` and `perGroupScale ~ N(0, 12)` — a
  wider and more honest test than §11.2.2's `randint(-5, 5)`.

---

## 14. The alternatives, the wall as a public issue, and five ways an int4 kernel silently computes garbage

### 14.1 CANN's *other* answer: widen int4 to int8, then `Mmad` normally

Repository: `https://gitcode.com/cann/catlass`, `examples/32_w4a8_matmul/README.md`, verbatim:

> ## 功能介绍
>
> - 提供了W4A8量化模式下的matmul实现
> - A矩阵`int8`类型，B矩阵`int4`类型，转换为`int8`后，经矩阵乘和per-tensor量化为`fp16_t`(`half`)类型输出

「转换为int8后，经矩阵乘」 is the whole alternative in one clause: promote the weight to int8 in
the vector unit, then run an ordinary `int8 x int8 -> int32` `Mmad`. The component is
`include/catlass/gemm/tile/cast_int4_to_int8.hpp`, which dispatches on `CATLASS_ARCH == 2201` to
`catlass/gemm/tile/atlasa2/cast_int4_to_int8.hpp`, and is wired in `examples/32_w4a8_matmul/w4a8.cpp` as

```cpp
using PrologueB = Gemm::Tile::TileCastInt4ToInt8<ArchTag, PrologueSrcType, PrologueDstType, computeLen>;
```

with `using ElementPrologueB = AscendC::int4b_t;`. The CHANGELOG records
`修复[cast_int4_to_int8](.../include/catlass/gemm/tile/cast_int4_to_int8.hpp)组件的Vector同步问题`
and labels example 32 `[INT4类型反量化Matmul算子]`.

So the public CANN tree ships **both** strategies side by side:
`32_w4a8_matmul` = widen-then-Mmad (A stays int8, B is promoted),
`35_w4a8_grouped_matmul_msd` / `36_w4a8_matmul_msd` = split-then-Mmad (both operands become int4).

The trade, stated plainly, is the one §12.3 quantifies: widening needs a full int8 replica of the
weight — `E*K*N` bytes, which PR #17201 prices at `2.296875 GiB at E=112` for K=3584/N=6144 —
and costs a vector pass per weight tile; splitting needs no weight replica at all but doubles the
M of the GEMM. At batch=1 decode, where the weight stream *is* the cost and M is 1-8, splitting
wins by an enormous margin: doubling a tiny M is nearly free, while doubling the weight bytes is
fatal. That asymmetry is the whole reason MSD exists for decode and why §1 frames it as a Decode
optimisation.

### 14.2 The wall, as a public issue, with a Huawei engineer's answer

URL: https://gitcode.com/cann/pto-isa/issues/115
Title: "[Requirement|需求建议]: Missing int4 conversion and mmad instructions"
Author: `jiawei_zhuang`. Opened Feb 6, closed Feb 10.

The report names three missing instructions, each cross-linked to Huawei's own CANN documentation
and to the corresponding `asc-devkit` v8.5.0 implementation headers:

- `mad_s4` — for W4A4 matmul; documented in CCE and available via `AscendC::Mmad`, but
  "Currently not available in `TMATMUL`."
- `vconv_s42f16` — for W4A16 dequantization; available in `AscendC::Cast`, "Currently not
  available in `TCvt`."
- `vconv_f162s4` — for 16-bit → 4-bit quantization; documented in CCE and available in
  `AscendC::Cast`, likewise not exposed in `TCvt`.

`changxianyu` (member), Feb 6, verbatim:

> 您好，mad_s4相关计算已在需求计划中，后续会进行支持。

The reply addresses only `mad_s4` and is silent on the two `vconv` conversions. Status moved
进行中 → 已完成 and the issue was closed Feb 10 by the same maintainer; the reporter then noted
that as of that day's latest commit (`4451a41`) the int4 instructions were still unsupported and
asked whether to reopen.

This is the closest thing to a public acknowledgement of the gap. Note what it is *not*: nobody in
it asks for a mixed int8xint4 MAD. The request is to expose the int4 instructions CANN already has
through PTO. The mixed-width instruction is never asked for, because the `Mmad` dtype tables (§11.1)
make it obvious there is no such thing.

Huawei's own documentation names it as a limitation of the method rather than a hardware gap —
msmodelslim `term_w4a8_dynamic.md`, 使用限制:

> - **INT4×INT8 混合位宽 GEMM 需要目标硬件算子库支持，否则收益打折。**

And a third-party author states it outright —
`https://blog.csdn.net/2501_94607758/article/details/155892168`,
"昇腾Ascend C极致优化实战：实现INT4量化矩阵乘（GEMM）算子，加速大模型推理", 2025-12-13 20:31:35:

> 然而，昇腾官方尚未开放INT4 GEMM算子！本文将手把手教你用Ascend C从零实现一个高性能INT4 GEMM算子，支持：
> * 权重INT4 + 激活FP16混合精度
> * 按通道（per-channel）反量化
> * 分组量化（Group-wise Quantization）
> * 向量化解包与计算融合
>
> 🔑 关键挑战：
> * INT4需打包存储（2个INT4=1字节）
> * 反量化需按通道/分组加载Scale
> * 计算需在FP16/FP32中进行以保精度

(that author's answer is the third option: unpack to FP16 and skip the integer MAD entirely.)

### 14.3 Five ways an int4 AscendC kernel silently computes garbage

URL: https://github.com/tile-ai/tilelang-ascend/pull/1731
Title: "[Fix] gemm_v0: fix W4A4 int4xint4 GEMM producing wrong results (#438)"
Author: `2Sheldonlaw`. Created 2026-09-03. Status: OPEN.
(Note: this is **our own org's repo**, `tile-ai/tilelang-ascend`.)

This is the single most directly applicable item in the file for anyone about to hand-write an
int4 GEMM, because it is a list of bugs that were *found by measurement* rather than reasoning.
Reproduced on `Ascend910 / CANN 9.0.0`, fixing `99.6% mismatch` → `0 mismatch`, with
`输出 -501/627 而正确值是 641/1785`. Verbatim:

> W4A4（int4×int4→int32）GEMM 结果错误的根源：整个 C++ 模板栈都用 `sizeof(AscendC::int4b_t) == 1` 推导分形几何，但 `int4b_t` 是半字节打包类型（每元素 4 bit，每字节装 2 个元素）。32 字节 C0 块的正确元素数是 64 而非 32——于是所有拷贝 stride、repeat 次数、字节计算全部 2× 偏差。

> B 的转置加载被静默丢弃：通用 `LoadData` 路径对 zN→nZ 的 B 操作数加载设置 `ifTranspose=true`，但 s4 专用指令 `load_cbuf_to_cb_s4` 硬编码关闭转置并静默忽略该标志——B 永远得不到 `mad_s4` 期望的 nZ 布局。

> int4 的 transpose_A / transpose_B：不支持——s4 指令没有 L0A 转置模式，且 B 的转置加载要求 zN 源布局。编译期 `static_assert` 给出明确信息。

> 形状约束：int4 GEMM 要求 K 和 N 为 64 的倍数（一个 s4 C0 块）；example 中有断言。

> GM int4 打包约定（消费量化 kernel 输出或喂 int4 输入的接口契约）：每字节两个 nibble，低 nibble = 偶数元素，高 nibble = 奇数元素；(M, N) tensor 的第 r 行起始于字节 `r * N / 2`。与 CANN 自家的 A4W4 ND 格式（`grouped_matmul_swiglu_quant_v2`）一致。

and on the PTO route:

> 附带 bug（pto target）：`target="pto"` 以一句裸的 `Unsupported datatype` 失败，原因是 pto-isa 缺少全部 int4 指令（`mad_s4`、`vconv_f162s4`、`vconv_s42f16`）——见 [cann/pto-isa#115](https://gitcode.com/cann/pto-isa/issues/115)。这是上游限制；本修复把含义不明的报错替换为指向 pto-isa issue 的可行动信息。
>
> pto target：在 pto-isa 添加 `mad_s4` / `vconv_f162s4` / `vconv_s42f16` 之前无法支持 int4（[cann/pto-isa#115](https://gitcode.com/cann/pto-isa/issues/115)，官方确认"在需求计划中"）。

Collected, the five traps:

1. **`sizeof(AscendC::int4b_t) == 1`, and every generic template believes it.** Two int4 per byte
   means every byte count derived from `sizeof` is 2x wrong. §11.1's official statement is the
   same fact from the other side: 「int4b_t/fp4x2_e2m1_t/fp4x2_e1m2_t数据类型的C0_size为64」 —
   64 elements per 32-byte C0 block, not 32.
2. **`load_cbuf_to_cb_s4` hard-codes transpose off and silently ignores `ifTranspose`.** Any B
   transpose you request is dropped without error. §4's 「weight可转置（A8W4/A4W4 不可）」 is the
   doc-level shadow of this.
3. **No L0A transpose mode for s4 at all**, so `transpose_A` is unavailable, matching §4's
   「x不支持转置、weight不支持转置」.
4. **K and N must be multiples of 64** for int4 GEMM — one s4 C0 block. Stricter than §11.1's
   `k % 2 == 0` from `MmadCal` and §4's `n % 8 == 0`; take the strictest, which is 64.
5. **The GM packing convention is low-nibble-even, high-nibble-odd, row `r` at byte `r*N/2`**,
   and it agrees with msModelSlim's packer (§13.1) and with CANN's A4W4 ND format. Get it backwards
   and you get plausible-looking garbage, not a crash.

### 14.4 The same trick on other hardware — and why CANN's `-8` shift is the better choice

URL: https://arxiv.org/abs/2609.27453
Title: "Implementation and Evaluation of BitNet Inference on a CGLA by Signed-Int4 Instructions"
Authors: Takuto Ando, Yasuhiko Nakashima (Nara Institute of Science and Technology)
Submitted: 23 September 2026. `arXiv:2609.27453v1 [cs.AR]`. Accepted at the CSA Workshop, CANDAR 2026.

Abstract, verbatim:

> Each ternary weight occupies one signed 4-bit lane. Each int8 activation is split into two signed-int4 fragments and reconstructed by shift-and-add.

Introduction, verbatim:

> occupies one signed-int4 lane, while an int8 activation is divided into a signed low-nibble residue and an adjusted high fragment. Two multiply–accumulate streams evaluate the fragments. The host combines their partial sums by shift-and-add and corrects positive values from 120 to 127. Although one OP_SMA4 stream carries eight packed signed-int4 products per 32-bit lane, the exact BitNet mapping requires two streams to cover an int8 activation.

Figure 2's annotations, verbatim:

> int8 activation a / a = 16 aₕᵢ + aₗₒ / aₕᵢ = (a − aₗₒ)/16 / adjusted high fragment / aₗₒ = sign -ext. low residue signed int4 ∈[−8, 7] / OP_SMA4 stream 1 Σ w · aₕᵢ / OP_SMA4 stream 2 Σ w · aₗₒ / shift-and-add 16 Σ(w aₕᵢ) + Σ(w aₗₒ) / integer partial sum (exact in stated range)
>
> Exact for −128 ≤ a ≤ 119: both fragments fit signed int4 [−8, 7], so the two int4 streams reconstruct the original int8 activation and its integer dot product exactly; a ∈[120, 127] uses sparse host correction for exact reconstruction.

> One original BitNet product wiai therefore corresponds to two signed-int4 fragment products in the exact path. ... Exact layer projections count two fragment products for each original MAC.

**This is the sharpest available contrast with CANN's design, and it cuts in CANN's favour.** The
CGLA scheme *sign-extends* the low nibble, so `a_lo ∈ [-8, 7]` but `a_hi = (a - a_lo)/16` reaches
`8` for `a ∈ [120, 127]` and overflows int4 — hence their "sparse host correction" and the awkward
「Exact for −128 ≤ a ≤ 119」. CANN takes the *unsigned* low nibble and shifts it by `-8`, which
keeps both fragments inside `[-8, 7]` for **every** int8 input with no exception; the price is
exactly the `8*sum_K(w)` term, which is a constant per output column.

So: **the `8*sum(w)` correction is not overhead bolted onto the decomposition. It is what buys
full-range exactness.** The alternative design that avoids the correction has a hole at
`a ∈ [120, 127]` and needs a host fixup. Worth knowing before anyone proposes "removing" it — and
worth knowing that §12's correction-row design removes the *offline tensor* while keeping the
*term*, which is the only sound way to do it.

The general framing — an error-free transformation,
`AB = (A₁+A₂)(B₁+B₂) = A₁B₁+A₁B₂+A₂B₁+A₂B₂` split by significand bits with a power-of-two
multiplier — is in `https://arxiv.org/pdf/2509.23565`, "Performance and Numerical Aspects of
Decompositional Factorizations with FP64 Floating-Point Emulation in INT8".

And the reason this is an Ascend idiom rather than a GPU one:
`https://arxiv.org/html/2608.10103v1` ("Hand-Written PTX Tensor-Core GEMM Kernels: A
Multi-Precision Study on NVIDIA L4") reports PTX INT4 at 2.9x-4.3x over WMMA via native
`mma.sync.m16n8k64.s4`, but since Ampere INT4 is only 2x INT8 throughput — so a two-term split
merely breaks even. On Ascend the alternative is not a slower int8 path but, for a weight-bandwidth
-bound decode GEMM, no viable path at all.

### Relevance

- **Trap 1 is the one that will bite us.** `sizeof(int4b_t) == 1` with 64 elements per C0 block
  means every stride, repeat count and byte computation in a hand-written int4 kernel is a 2x
  hazard. Our own org already hit it and measured `99.6% mismatch`. Audit every `sizeof` in the
  int4 path before the first run, and note that §10's `CopyInA1` halves every `d`-axis field by
  hand (`dValue = col / 2`, `srcDValue = col / 2`) for exactly this reason.
- **Take K and N multiples of 64, not 2 and 8.** Three sources give three different parity rules
  (§11.1's `k % 2`, §4's `n % 8`, this PR's `K, N % 64`); the last is the one derived from the s4
  C0 block, and PR #17201 independently says `K a positive multiple of 64 up to 16384` and
  `N a positive multiple of 256`.
- **Never request a transpose on an int4 operand.** It is silently dropped, not rejected. Lay the
  weight out correctly in the checkpoint instead (§4's affine format, `(E, N, K)` then ND2NZ, or
  §12.4's byte formula).
- **PTO is off the table for this GEMM** — confirmed twice independently (§11.1's TMATMUL.md and
  pto-isa#115, with Huawei's 「已在需求计划中」 and no delivery as of the closing commit).
- **Do not propose eliminating the `8*sum(w)` term.** §14.4 is the published cautionary tale: the
  design that drops it has a correctness hole at `a ∈ [120, 127]`. §12 is the right way to move it
  on-chip.
- **If we ever consider the widen-to-int8 alternative**: it is `catlass/examples/32_w4a8_matmul`
  with `TileCastInt4ToInt8`, it is real and A3-supported, and its cost is an `E*K*N`-byte int8
  weight replica — 2.3 GiB at E=112/K=3584/N=6144 by PR #17201's accounting. At batch=1 decode that
  is disqualifying; it is listed here so the comparison is on the record, not so we do it.

---

## Gaps

What is not here, and exactly where it was looked for.

### Not found: a published accuracy number for the MSD split in isolation

No source anywhere measures the decomposition's own cost separately from int4 weight
quantization. The nearest proxies, all in §11.2.3, §13.5: the ST's
`"fp32_error": "2**-10"` tolerance under `"type": "high_performance"`;
`sgl-kernel-npu`'s `DIFF_THRESHOLDS = {"w4a8": 5e-3, "w8a8": 2e-3}` (a 2.5x looser threshold for
W4A8 than W8A8, which is suggestive and not a measurement); and end-to-end task numbers
(GPQA 89.9 vs an 89.1 *paper* baseline; Qwen3.5 grounding 93.34% vs 94.86% full). Tried:
`hiascend.com/developer/techArticles/20250717-2` and `.../20250728-1` (no table on either),
`blog.csdn.net/2503_93041010/article/details/150558465` (a verbatim repost of the latter, also no
table — the lead in the brief does not pan out), msmodelslim `docs/zh/best_practices/*`,
vllm-ascend `docs/source/tutorials/models/*`, `gh search issues` across vllm-ascend and sglang.
**Nothing to borrow; generate it ourselves with the two references in §11.2 and §13.3 plus an
int64 one.**

### Not found: no paper on the A8W4 nibble split

arXiv 2605.13915 "Multi-Scale Dequant" (§7.1) is the BF16→INT8/MXFP4 family, not ours. Searched
arXiv for the int8→two-int4 split; the only hit is the CGLA/BitNet paper in §14.4, which is a
different chip and a different (worse) variant. There is no publication on CANN's A8W4 MSD as of
2026-10-20. The hiascend article in §1 is the primary source and the only one that expands the
acronym.

### Not found: any `bbs.huaweicloud.com` or Zhihu discussion of the mixed-width MAD

Four distinct Chinese query formulations produced nothing. The question appears to be asked only
once in public, on GitCode (`gitcode.com/cann/pto-isa/issues/115`, §14.2), and answered only by
the `Mmad` dtype tables. Tried `bbs.huaweicloud.com`, `zhihu.com`, `blog.csdn.net`,
`gh search issues` on vllm-ascend (`msd` → zero results) and
`gh api search/code 'repo:sgl-project/sglang weight_assist_matrix'` (`total_count` 0).

### Not obtainable: the catlass MSD kernel source

`https://gitcode.com/cann/catlass/tree/master/examples/35_w4a8_grouped_matmul_msd` and
`.../36_w4a8_matmul_msd` contain **only `gen_data.py`** — no `.cpp`, no README. The kernel that
consumes those goldens is not published in `catlass`. The MSD kernel sources that *are* public
are the four in §2, §3, §5, §10 (`ops-transformer`, `ops-nn`) plus the vendored and new ones in
`vllm-project/vllm-ascend` `csrc/gmm/` and `csrc/gmsq_situ_quant/`, and
`vllm-project/afd-plugin` `csrc/npu/ascend_kernels/`.

### Open question, now narrow: does `Fixpipe` emit `int32` for int4 operands *with* a quant vector?

§11.1 establishes that `Matmul`'s 表2 permits `int4b_t | int4b_t | int32_t | int32_t/half` on A3,
and §12 shows `MsdC = MatmulType<..., int32_t>` running. What no source states is whether
`SetQuantVector` can coexist with an `int32_t` C — PR #17201 simply does not call `SetQuantVector`
and applies the scale in the vector unit instead. **Assume they are mutually exclusive until
measured**; the §12 design does not need them together.

### Open question: the exactness bound above `K = 16384`

§12.2's argument is `|16*high + low| <= 1024K` against fp32's integer-exact limit `2^24`. That
makes `K <= 16384` safe. PR #17201 encodes the same bound in its accepted shapes
(`K a positive multiple of 64 up to 16384`). For `K > 16384` the fp32 reconstruction stops being
exact and the recombination must be done in int32. Nobody has published what the degradation looks
like past the bound.

### Unresolved conflict: is V1's A8W4-MSD path real?

A parallel pass reported, from `ops-transformer` HEAD's
`gmm/grouped_matmul_swiglu_quant/README.md`, that `GroupedMatmulSwigluQuant` V1 is A8W8-only and
that "there is no `weightAssistMatrix` in V1; the MSD A8W4 path exists only from V2 onward". That
is inconsistent with what §2 quotes from the same operator directory at two different commits:
`grouped_matmul_swiglu_quant.cpp` declares `GM_ADDR weightAssistanceMatrix` as a kernel argument
and a `GMM_SWIGLU_QUANT_A8W4_MSD` branch at `TILING_KEY_IS(2)`; the three
`grouped_matmul_swiglu_quant_a8w4_msd_{pre,mid,post}.h` files exist; and
`docs/aclnnGroupedMatmulSwigluQuant.md` carries its own 「MSD场景A8W4」 section with
`weightAssistMatrix∈R^{E × N}`. Most likely the README lags its own operator. **Treat V1's MSD
path as real but under-documented, and prefer V2 if the choice is free.** Not resolved here
because resolving it needs the README at HEAD, which was not read directly.

### Known stale: the clone this file was mostly written from

§2-§10 are read from `ops-transformer` snapshots at `e027a008` (2026-09-01) and `d6aaa9fe`
(2026-10-19); `ops-nn` at `af66e479` (2026-06-26). HEAD at the time of writing was
`ops-transformer` `d462156c` (2026-10-20 15:23:59 +0800) and `ops-nn` `92076b30`
(2026-10-20 15:37:51 +0800). §9.3 records a case where the staleness produced a wrong conclusion
and how it was caught. **Re-pull before concluding any absence from these repos.**

### Fetch mechanics that cost time, recorded so they do not cost it twice

- `hiascend.com/developer/techArticles/<id>` **works** with `curl` + a browser UA — that is how §1
  was obtained. This contradicts the general rule about `hiascend.com` and is worth remembering.
- `hiascend.com/document/detail/...` returns an empty body to a fetcher, and `curl` on both
  `/document/detail/` and `/doc_center/` is blocked by HuaweiCloudWAF
  (`<meta http-equiv="Server" content="HuaweiCloudWAF">`, "Please Enable JavaScript and Cookie").
  `doc_center` pages can be downloaded as static HTML and table-extracted, which is how §14.2's
  表6 was obtained.
- `asc.gitcode.com/api/` is the reliable route for API pages, and its footer carries the
  `asc-devkit` commit it was built from. `asc.gitcode.com/api/sitemap.xml` and `hashmap.json` are
  404; use `asc.gitcode.com/api/README.html` (3364 hrefs) as the index.
- `gitcode.com/cann/<repo>/raw/master/<path>` returns a JavaScript shell, not file content.
  `raw.gitcode.com` rate-limits with `429 Limited account:unregisteredUser` after a handful of
  requests. `gitcode.com/cann/<repo>/blob/master/<path>` works for browsing. **Clone instead**:
  `git ls-remote https://gitcode.com/cann/ops-transformer.git` answers from this machine, and
  `git clone --filter=blob:none --no-checkout --depth 1` + `git sparse-checkout` is the cheap path.
- `gitcode.com/cann/{cann-ops,cann-ops-adv,ops-base}` are all 404 — the operator libraries were
  renamed to `ops-nn`, `ops-transformer`, `ops-math`, `ops-blas`. The brief's suggested
  `gitee.com/ascend/cann-ops-adv` and `github.com/Ascend/cann-ops-adv` are the wrong targets for
  this material.
- CSDN blocks `curl` and WebFetch with a JS challenge plus HTTP 521; `https://r.jina.ai/<url>`
  gets through.
- `modelslim.readthedocs.io/.../term_w4a8_dynamic/` is 404; the content exists as `.md` in the
  msmodelslim tree.
- `npu-arch-facts.json`'s `dtype_product_support.atlas_a2_a3` omits `int4b_t` and is wrong against
  four other sources (§11.1). Do not trust it.

### Deliberately not duplicated here

The `Mmad` both-operands-int4 rule as established from Huawei's `matmul_int4` README —
「左矩阵A为int4数据类型时，不支持转置」 and
「int4数据类型输入的Matmul对象仅支持Norm、MDL、IBShare模板」 — is in
`ascend-fusion-mechanics.md` §42.8. §11.1 here covers the same rule from the `Mmad` API page and
the `dav_c220` source instead, and adds the `k % 2 == 0` assert; the two are consistent.
