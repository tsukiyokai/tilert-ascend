# Fused MoE Expert Kernels — Source Material

Verbatim research deposit for the GLM-5.2-W4A8C8 / Ascend A3 megakernel effort.
All quoted blocks are reproduced as published. Chinese sources are left in Chinese.
Collected 2026-10-20.

---

## 1. CANN first-party: the op we are already paying for

### GroupedMatmulSwigluQuantV2 — official op README (fusion scope, A8W4 MSD math)

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/README.md
Huawei / CANN `ops-transformer` repo, master branch.

> # GroupedMatmulSwigluQuantV2
>
> ## 产品支持情况
>
> | 产品 | 是否支持 |
> | :---------------------------- | :-----------: |
> | <term>Ascend 950PR&950DT系列产品</term> | √ |
> | <term>Atlas A3系列产品</term> | √ |
> | <term>Atlas A2系列产品</term> | √ |
> | <term>Atlas 200I/500 A2推理产品</term> | × |
> | <term>Atlas推理系列产品</term> | × |
> | <term>Atlas训练系列产品</term> | × |
>
> ## 功能说明
>
> - 算子功能：融合GroupedMatmul 、dequant、swiglu和quant，详细解释见计算公式。

The A8W8 path (what the fusion actually computes):

```
- 1.根据groupList[i]确定当前分组的token ，$i \in [0,Len(groupList)]$。

  >例子：假设groupList=[3,4,4,6]、groupListType=cumsum或groupList=[3,1,0,2]、groupListType=count。
  >
  >注：以上两种不同的分组方式，实际为相同的分组结果。
  >
  >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`bias[0]`、`offset[0]`、`Q[0:3]`、`Q_scale[0:3]`、`Q_offset[0:3]`；
  >
  >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`bias[1]`、`offset[1]`、`Q[3:4]`、`Q_scale[3:4]`、`Q_offset[3:4]`；
  >
  >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`bias[2]`、`offset[2]`、`Q[4:4]`、`Q_scale[4:4]`、`Q_offset[4:4]`；
  >
  >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`bias[3]`、`offset[3]`、`Q[4:6]`、`Q_scale[4:6]`、`Q_offset[4:6]`；
  >
  >注：grouplist中未指定的部分将不会参与更新。
  >例如当groupList=[12,14,18]、GroupListType=cumsum，X的shape为[30，:]时。
  >
  >则第一个输出Q的shape为[30，:]，其中Q[18:，：]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。

- 2.根据分组确定的入参进行如下计算：

  $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

  $C_{i,act}, gate_{i} = split(C_{i})$

  $S_{i}=Swish(C_{i,act})\odot gate_{i}$  其中$Swish(x)=\frac{x}{1+e^{-x}}$

- 3.量化输出结果

  $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

  $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
```

The A8W4 **MSD** path — this is the one we run. MSD = the INT8 activation is split into
two INT4 halves and **two** matmuls are issued per group, plus an offline correction matrix:

```
<summary>MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指INT8数据类型，4指INT4数据类型）：</summary>

  - **定义**：
    * **⋅** 表示矩阵乘法。
    * **⊙** 表示逐元素乘法。
    * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
    * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
    * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
    * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
  - **输入**：
    * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
    * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
    * $weightAsistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
    * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K\_group\_num是在K轴维度上的分组数，N是输出维度。
    * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
    * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
  - **输出**：
    * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
    * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
  - **计算过程**
    - 1.根据groupList[i]确定当前分组的token，$i \in [0,Len(groupList)]$。
      - 分组逻辑与A8W8相同。
    - 2.生成辅助矩阵（weightAsistMatrix）的计算过程（请注意weightAsistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
      - 当为per-channel量化（$w\_scale$为2维）：

        $weightAsistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

      - 当为per-group量化（$w\_scale$为3维）：

        $weightAsistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

        注：$num\_per\_group = K // K\_group\_num$

    - 3.根据分组确定的入参进行如下计算：

      - 3.1. 将左矩阵$\mathbb{Z}_8$，转变为高低位两部分的$\mathbb{Z}_4$
        $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
        $X\_low\_4bits_{i} = (X_{i} \& 0x0f) - 8$
      - 3.2. 做矩阵乘时，采用per-channel或per-group量化
        per-channel：

        $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

        $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

        per-group：

        $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

        $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

      - 3.3.将高低位的矩阵乘结果还原为整体的结果

        $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAsistMatrix_{i}) \odot x\_scale_{i}$

        $C_{i,act}, gate_{i} = split(C_{i})$

        $S_{i}=Swish(C_{i,act})\odot gate_{i}$  其中$Swish(x)=\frac{x}{1+e^{-x}}$
```

A3-specific parameter and shape rules, including the NZ weight layouts:

```
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
    - x仅支持INT8量化数据类型、不支持其他数据类型。
    - weight仅支持非转置，支持INT8、INT4、INT32数据类型，ND格式shape形如{(E, K, N)}，NZ格式下，当weight数据类型是INT8时shape形如{(E, N / 32, K / 16, 16, 32)}，INT4时shape形如{(E, N / 64, K / 16, 16， 64)}，INT32时shape形如{(E, N / 64, K / 16, 16， 8)}。
    - weightScale，A8W8场景支持FLOAT、FLOAT16、BFLOAT16数据类型，shape只支持2维，形如{(E, N)}；A8W4场景支持UINT64数据类型，shape支持2维和3维，其中per-channel的shape形如{(E, N)}，per-group的shape形如{(E, KGroupCount, N)}。
    - 支持dequantMode参数：0表示激活矩阵per-token，权重矩阵per-channel；1表示激活矩阵per-token，权重矩阵per-group。
    - 不支持dequantDtype参数。
    - 不支持quantMode参数。
    - A8W8/A8W4场景，不支持N轴长度超过10240。
    - A8W8场景，不支持x的尾轴长度大于等于65536。
    - A8W4场景，不支持x的尾轴长度大于等于20000。
    - output仅支持数据类型INT8，shape支持2维，形如(M, N / 2)。
    - outputScale仅支持数据类型FLOAT，shape支持1维，形如(M,)。

  - 确定性计算：
      - aclnnGroupedMatmulSwigluQuantV2默认为确定性实现。
```

## Relevance

This is the exact op that costs us 6.793 ms against 2.272 ms of bytes. The MSD A8W4 formula
shows why: per group it issues **two** INT4 MMADs (`C_high` and `C_low`) over the same
`W_i`, then a vector pass for `C_high*16 + C_low + weightAssistMatrix`, then dequant, split,
Swish, multiply, and a per-token max-abs reduction for `Q_scale`. The weight bytes are read
once but the MMAD count and the vector-pass count are both doubled relative to a plain A8W8
grouped GEMM — so at M=8 (batch=1, top-8) the op is entirely fixed-cost, which matches the
3.0x gap we measure. `weightAssistMatrix` is offline, so it is free to carry into a megakernel.

---

### aclnnGroupedMatmulSwigluQuantV2 — exact signature and `groupListType` / `tuningConfig` semantics

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor     *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix,
    const aclTensor     *bias,
    const aclTensor     *xScale,
    const aclTensor     *smoothScale,
    const aclTensor     *groupList,
    int64_t              dequantMode,
    int64_t              dequantDtype,
    int64_t              quantMode,
    int64_t              groupListType,
    const aclIntArray   *tuningConfigOptional,
    aclTensor           *output,
    aclTensor           *outputScale,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor);
```

> | groupListType | 输入 | 表示分组的解释方式，用于确定groupList的语义。 | 0表示cumsum模式，groupList中的每个元素代表当前分组的累计长度。 1表示count模式，groupList中的每个元素代表该分组包含多少元素。 | INT64 |
>
> | tuningConfigOptional | 可选输入 | 用于算子预估M/E的大小，走不同的算子模板，以适配不同场景性能要求。 | 数组，传入的第一个数字表示各个专家处理的token数的预期值，用于优化tiling，A4W4 右矩阵NZ输入时开启，其他输入请传入空指针。 | INT64 |

> - weightAssistMatrix：
>   - 仅A8W4场景生效，其他场景需传空指针。
> - groupList：表示输入和输出在分组轴方向的matmul大小分布，对应公式中的grouplist，具体含义由groupListType确定。
>   - 当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维。
>   - groupList中的值约束了输出数据的有效部分，groupList未指定的部分将不会参与更新。
> - weight在A4W4下支持NZ输入转置，其他场景仅支持非转置。INT32为A8W4和A4W4场景下的适配用途，实际1个INT32会被解释为8个INT4数据，A8W8场景不支持ND数据格式。
> - 支持dequantMode参数：A8W4和A4W4场景支持取值0和1，A8W8场景仅支持取值0。
> - weight、weightScale和weightAssistMatrix支持单Tensor场景（tensorlist长度为1）和多Tensor场景（tensorlist长度大于1）。
> - A8W4场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于20000。
> - 多tensor场景下，即tensorlist长度大于1时，weight、weightScale和weightAssistMatrix的shape需要按照E的维度展平，例如{(E, K, N)}需要变成{E个(K, N)}。

## Relevance

`tuningConfigOptional` is the only published hook for telling the op "expect this many tokens
per expert", and the doc says it is only honoured for A4W4-NZ — i.e. on our A8W4 path there is
no way to tell the stock op that M=8. That is a direct argument for owning the tiling ourselves.
`groupList未指定的部分将不会参与更新` matters for a megakernel: output tail is uninitialized
garbage, so a fused consumer must not read past `groupList[-1]`.

---

### The A8W4-MSD kernel is already a 3-stage software pipeline with grid-wide barriers

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h

```cpp
template <class mmType>
class GMMSwigluQuantPipelineSchedule {
    ...
    // 前处理GMMA8W4PreProcess类
    GMMA8W4PreProcess preProcess;
    // 中间处理GMMA8W4MidProcess类
    GMMA8W4MidProcess<mmType> midProcess;
    // 后处理GMMA8W4PostProcess类
    GMMA8W4PostProcess postProcess;
```

```cpp
template <class mmType>
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

## Relevance

This is the megakernel pattern already in production inside one op: `pre(n+1) || mid(n) || post(n-1)`
over workspace tiles, with `SyncAll<>()` as the grid-wide barrier and `ASCEND_IS_AIV` selecting
the role of each core. Our megakernel can copy this skeleton and extend the loop body across
op boundaries rather than restarting it per op. Note the cost it implies at batch=1: four
`SyncAll` per iteration plus `pipe->Reset()`, all of which are fixed and unamortized when M=8.

---

### `group_list` is walked with scalar global-memory reads inside the kernel

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h

```cpp
template <class mmType>
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::InitWorkSpaceSplitConfig(
    WorkSpaceSplitConfig& workspaceSplitConfig)
{
    if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
        workspaceSplitConfig.M = groupListGM.GetValue(gmmSwigluQuantV2->groupListLen - 1);
    } else {
        int64_t totalTmp = 0;
        for (uint32_t i = 0; i < gmmSwigluQuantV2->groupListLen; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
        workspaceSplitConfig.M = totalTmp;
    }
    workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, gmmSwigluQuantV2BaseParams->mLimit);
    workspaceSplitConfig.notLastTaskSize = gmmSwigluQuantV2BaseParams->mLimit;
    workspaceSplitConfig.lastLoopTaskSize =
        workspaceSplitConfig.M - (workspaceSplitConfig.loopCount - 1) * gmmSwigluQuantV2BaseParams->mLimit;
    ...
}
```

```cpp
template <class mmType>
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::UpdateWorkSpaceSplitConfig(
    WorkSpaceSplitConfig& workspaceSplitConfig, int32_t workspaceSplitLoopIdx)
{
    if (workspaceSplitLoopIdx < 0)
        return;
    workspaceSplitConfig.leftMatrixStartIndex = workspaceSplitLoopIdx * gmmSwigluQuantV2BaseParams->mLimit;
    workspaceSplitConfig.rightMatrixExpertStartIndex = workspaceSplitConfig.rightMatrixExpertNextStartIndex;
    workspaceSplitConfig.rightMatrixExpertEndIndex = workspaceSplitConfig.rightMatrixExpertStartIndex;
    // 计算右专家矩阵的终止索引(rightMatrixExpertEndIndex) 和下一次的起始索引(rightMatrixExpertNextStartIndex)
    int32_t curTaskNum = 0;
    int32_t nextTaskNum = 0;
    int32_t curTaskNumTmp = 0;
    int32_t nextTaskNumTmp = 0;
    if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertEndIndex; i++) {
            curTaskNumTmp += groupListGM.GetValue(i);
        }
        if (workspaceSplitConfig.rightMatrixExpertEndIndex == 0) {
            nextTaskNumTmp = groupListGM.GetValue(0);
        } else {
            for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertEndIndex; i++) {
                nextTaskNumTmp += groupListGM.GetValue(i);
            }
        }
    }
    while (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSwigluQuantV2->groupListLen) {
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            curTaskNum = groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex) -
                         workspaceSplitConfig.leftMatrixStartIndex;
        } else {
            curTaskNumTmp += groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex);
            curTaskNum = curTaskNumTmp - workspaceSplitConfig.leftMatrixStartIndex;
        }
        int32_t nextTaskIdx = workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSwigluQuantV2->groupListLen - 1 ?
                                  gmmSwigluQuantV2->groupListLen - 1 :
                                  workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            nextTaskNum = groupListGM.GetValue(nextTaskIdx) - workspaceSplitConfig.leftMatrixStartIndex;
        } else {
            if (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSwigluQuantV2->groupListLen - 1) {
                nextTaskNumTmp += groupListGM.GetValue(nextTaskIdx);
            }
            nextTaskNum = nextTaskNumTmp - workspaceSplitConfig.leftMatrixStartIndex;
        }
        if (curTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        } else if (curTaskNum == gmmSwigluQuantV2BaseParams->mLimit &&
                   nextTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
            break;
        } else if (nextTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertEndIndex++;
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        }
        workspaceSplitConfig.rightMatrixExpertEndIndex++;
    }
    workspaceSplitConfig.isLastLoop = workspaceSplitLoopIdx == workspaceSplitConfig.loopCount - 1 ? true : false;
    ...
}
```

## Relevance

Both helpers resolve the group layout at runtime with per-element `groupListGM.GetValue()`
scalar global-memory loads on the scalar unit, and the `while` loop advances one expert per
iteration up to `groupListLen`. With E=256 and total M=8 (batch=1, top-8) the loop cannot exit
early on the `mLimit` tests, so it walks the full expert list doing two dependent scalar GM
reads per step — a serial scalar cost that is independent of how little math there is to do.
A batch=1 megakernel can instead carry the 8 active expert ids in registers or UB and skip the
walk entirely.

---

### `GroupedMatmul算子设计介绍` — how CANN schedules tiles across groups (base-block and diagonal partitioning)

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/GroupedMatmul%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md
CANN `ops-transformer`, licensed CC-BY 4.0 per the file header.

> 声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。
>
> # 1 GroupedMatmul融合算子设计介绍
>
> GroupedMatmul算子的功能是进行分组矩阵乘计算，每组矩阵乘的维度大小可以不同。矩阵乘公式如下：
> $$
> y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g
> $$
> 其中g为分组个数，$m_i、k_i、n_i$为对应shape。
> GroupedMatmul算子实现时还需要考虑如下两个方面：
>
> 1. 支持不同的参数、数据类型，如有无bias、不同激活函数类型，非量化、量化、伪量化等不同场景；不同场景对应的计算流程不同，性能优化方法不同，因此实现上划分成了不同的模板，有各自的模板参数；
> 2. 硬件上AiCore内存大小有限，一般完成一个算子的计算需要对数据进行切分，并对数据搬运和计算过程进行流水并行排布，该过程对算子的影响非常大，也是性能优化阶段主要调整对象，而host上的tiling函数即是为完成该切分和流水的参数计算。

> # 2 场景划分
>
> 从功能角度可分为非量化场景、量化场景和伪量化场景，代码层面通过三种方式选择具体模板：
>
> 1. 编译宏：通过x和weight的数据类型编译的宏ORIG_DTYPE_X，ORIG_DTYPE_WEIGHT；
> 2. tilingkey：如x/weight是否转置；
> 3. tilingData：如tiling中isPerTokenQuant表示是否为per token量化。
> 说明：
>
> - 非量化指x、weight、y均为浮点数类型，如float16/bfloat16/float32，非量化为纯cube场景，其计算过程由matmul高阶api实现；
> - 量化指x和weight为低精度整数类型，GroupedMatmul支持A8W8场景，包括per-tensor + per-channel量化和per-token + per-channel量化（简称per-token量化）；
> - 伪量化指x为浮点数类型，weight为低精度整数类型，GroupedMatmul支持A16W8和A16W4场景。
>
> ## 2.1 per token量化
>
> 本章以per token量化场景为例介绍一下GroupedMatmul的算法流程，计算过程如下：
> matmul(int32) -> 反量化(fp32) -> mul(fp32) -> 激活函数(fp32)(可选) -> cast(fp16/bf16)，其中mul(fp32)的输入perTokenScale还需要从shape(m) broadcast成(m，n)。

Grouping axis semantics:

> ## 2.2 分组方式
>
> 针对不同场景，GroupedMatmul可分为m轴分组和k轴分组，又称切M，切K。在正向训练过程对m轴进行分组，在反向计算梯度时就需要对k轴进行分组。
>
> - m轴分组：$k_i$各组相同，$weight_i/y_i$可以在$n_i$上拼接，此时group type = 0;
> m轴分组可用于非量化正向训练场景，量化场景和伪量化场景。

Single/multi tensor shape table:

> | group_type | supported scenario | x shape | weight shape | y shape | optional-dynamic inputs shape if needed | group_list shape if passed | per_token_scale shape if passed |
> | :--------: | :----------------: | :-----: | :----------: | :-----: | :-------------------------------------: | :------------------------: | :-----------------------------: |
> | -1 | 多多多 | [(M1,K1),(M2,K2),...] | [(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | [(N1),(N2),...] | not support | not support |
> | 0 | 单单单 | [(M,K)] | [(G,K,N)] | [(M,N)] | [(G,N)] | (G) | (M) |
> | 0 | 单多单 | [(M,K)] | [(K,N),(K,N),...] | [(M,N)] | [(N),(N),...] | (G) | not support |
> | 0 | 多多单 | [(M1,K1),(M2,K2),...] | [(K1,N),(K2,N),...] | [(M,N)] | [(N),(N),...] | (G) | not support |
>
> 例如在多多多场景，x shape={{4，16}, {12，16}, {16，16}}，weight shape={{16，8}，{16，8}，{16，8}}，y shape={{4，8}，{12，8}，{16，8}}，如果想在单单单场景进行相同的计算，则x shape={32，16}，weight shape= {3，16，8}，y shape={32，8}，groupList={4，12，16}。

Tiling data layout and the reason the kernel cannot instantiate per-group matmul objects:

> ```c++
> BEGIN_TILING_DATA_DEF(GMMTilingData)
>   TILING_DATA_FIELD_DEF_STRUCT(GMMBaseParams, gmmBaseParams);
>   TILING_DATA_FIELD_DEF_STRUCT(GMMArray, gmmArray);
>   TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
> END_TILING_DATA_DEF;
> ```
>
> tilingData主要包含上述结构里的三个部分：
>
> 1. GMMBaseParams: GroupedMatmul的分组数量、Aicore核数、ub tiling参数、matmul分核tiling参数等基础tiling参数，是否有激活函数、量化类型（per token或per tensor）、激活函数类型等功能参数;
> 2. GMMArray: 当输入是多tensor时，通过三个数组记录每组Matmul的shape, kernel通过GlobalTensor::GetValue()的方式获取。当输入为全单tensor时，m/k/n中的一个值在group_list中，另外两个值在所有的group中都相同，此时kernel只需要访问数组中的第一个值。
> 3. TCubeTiling: matmul高阶api对应的tilingData。
>
> kernel中为了避免在栈空间中申请GMMArray中3个数组以及避免拷贝这些数组，采用GET_TILING_DATA_MEMBER接口拷贝除GMMArray之外的结构体。

UB budget, with concrete numbers:

> ## 3.2. UB buffer分配
>
> 在初始化阶段，GroupedMatmul需要确定UB buffer的分配复用情况。非量化、伪量化的计算过程简单，没有UB buffer的复用，而量化场景多、计算复杂，UB buffer需要复用，以提高单次计算的数据量。
> 定义每份buffer分配的字节大小比上处理的数据个数（baseM\*baseN，记为ubCalcSize，此处baseM/baseN为vector计算的参数）为该buffer的份数，以per token量化为例，假设baseM = 24, baseN = 256，则ubCalcSize = baseM * baseN = 6kb。
>
> UB buffer分配如下：
>
> 1. vector计算的输入即matmul的输出的类型为int32，在开启doubleBuffer之后输入需要sizeof(int32) * 2，即8份buffer；
> 2. vector计算的输出为fp16/bf16，其需要sizeof(fp16/bf16) * 2，即4份buffer；
> 3. vector计算的中间计算过程需要申请tmpBuffer，其中broadcast的输出需要sizeof(fp32)块buffer，反量化的输出sizeof(fp32)块buffer，还需申请sharedBuffer进行buffer复用，包含broadcast临时空间、反量化临时空间和Mul输出，大小为8块buffer，共需要申请tmpBuffer的大小为16块buffer。
>
> 总共需要分配的UB buffer为28 * 6kb = 168kb。

The tile scheduling scheme — base blocks, then diagonal walk, with the two inequalities:

> ## 3.3 基本块分核方案
>
> GroupedMatmul实现时需要考虑输入为多个tensor的情况，即每组matmul的shape可能各不相同，而kernel侧不能为每组matmul单独配置对应的matmul高阶api接口实例（tiling结构体和core栈空间大小均不允许）。为了适配不同shape的matmul计算，GroupedMatmul采用基本块方式（横向分核），以baseM、baseN为基本块进行分核计算，此处baseM/baseN为matmul的参数。
>
> ## 3.4 对角线分核方案
>
> 按基本块方案进行分核，容易存在同地址访问的问题，例如当基本块方案中nDim=coreNum时，则同一时间所有的核都在访问左矩阵的相同地址，对性能影响较大。因此当基本块数量超过coreNum时（没超过coreNum时，对角线方案无法解决同地址访问问题），可以采用如下对角线方案，同一时间不同核尽量错开对数据的访问，每个方块代表一个输出的基本块，数字代表基本块遍历顺序（横向分核为原始基本块分核方案）。
>
> 对于适合对角线优化的场景，在基本块方案上输入数据会存在多次访问，当nDim/mDim很大时，“对角线”不能直接任意往下延伸，否则在k值比较大的场景下，对角线上对应的输入数据均为不同的数据，已经加载过一次的数据被新数据从L2 cache中替换掉，后续需要加载时还是从DDR内存中加载，导致性能劣化。因此需要限制对角线范围，以充分利用L2 cache中缓存的数据。
>
> 将对角线遍历按阈值进行分组，为尽量避免同地址访问，两个方向的阈值最好都不小于实际的物理核数，因此有：
>
> $$
> \min(T_m, T_n) \geq numCore
> $$
>
> 为充分利用cache，一个分组块对应的$X$、$Weight$、$Y$的总数据量最好不超过设备的L2 cache大小，因此有:
>
> $$
> sizeof(dtype) \cdot (T_m \cdot singleM \cdot K + K \cdot T_n \cdot singleN + T_m \cdot singleM \cdot T_n \cdot singleN) \leq L2_{size}
> $$
>
> 若两个条件不能同时满足，须根据实际情况做取舍。

## Relevance

The decisive sentence is `kernel侧不能为每组matmul单独配置对应的matmul高阶api接口实例（tiling结构体和core栈空间大小均不允许）` — CANN's grouped GEMM is forced into one shared
`TCubeTiling` for all 256 groups, so the base block `baseM×baseN` is a single global choice.
At batch=1 our active groups have `m_i = 1` while `baseM` is sized for throughput (the doc's
worked example uses `baseM = 24`), so every base block is almost entirely padding. The diagonal
scheme and both inequalities assume base-block count ≫ `coreNum`; with 8 active groups of 1 row
there are far fewer blocks than cores, so neither the diagonal walk nor the L2 constraint is
doing anything for us. A batch=1 megakernel should specialize `baseM = 1..8` and drop the
diagonal machinery.

---
## 2. The closest published prior art: MonoMoE, a weight-major persistent MoE decode megakernel

### MonoMoE: An Efficient Fused Mega-kernel for Quantized MoE Decoding

https://arxiv.org/abs/2609.04244
Yu Gong, Kailash Budhathoki, Taeho Kim, Haipeng Li, Ashish Khetan. Submitted 19 Aug 2026.
HTML: https://arxiv.org/html/2609.04244v1

Abstract, verbatim:

> Mixture-of-Experts (MoE) layers increase model capacity without proportionally increasing arithmetic, but their sparse expert computation is difficult to execute efficiently during autoregressive decode. Existing grouped and batched GEMMs are token-major: they construct expert-local token tiles and obtain parallelism from the token dimension. When few tokens reach each expert, this organization incurs tile padding and preprocessing, launches short-lived grids that underutilize memory bandwidth, and exposes quantization, activation, and reduction as separate stages. We present **MonoMoE**, a weight-major persistent megakernel for block-wise quantized MoE decode. MonoMoE places the complete decode-step token tile on the fine-grained tensor-core $N$ dimension and partitions CTAs over expert-weight tiles, eliminating expert-local token materialization and reducing padded arithmetic. A persistent grid fuses routing, top-$k$ selection, quantization, both expert projections, activation, and reduction in one launch; warp specialization and readiness flags overlap auxiliary work with the dominant expert-weight stream. MonoMoE is integrated with vLLM and supports multiple model shapes through generated kernel specializations and offline schedule tuning. On NVIDIA H200 GPUs, MonoMoE accelerates the complete routed-MoE operator by up to $\mathbf{1.54\times}$ over vLLM Triton Grouped GEMM, is $\mathbf{2.20}$--$\mathbf{3.84\times}$ faster than FlashMoE-FP8 adaptation across various models and batch sizes, and reduces end-to-end time per output token by up to $\mathbf{18.7\%}$ across the evaluated FP8 models, while preserving task accuracy.

Their baseline profiling of a batch-1 MoE layer — the same measurement we made:

> Our profiling puts numbers to this inefficiency. For a single Qwen3.5-35B-A3B-FP8 layer at batch size one on an H200, vLLM spends 23.0 μs in the two expert GEMMs – yet these kernels sustain only 17–22% of peak DRAM bandwidth. Routing, quantization, activation, and reduction add a comparable amount of low-bandwidth work on the same critical path. Across decode token counts from one to eight, our entitlement analysis shows that closing kernel-boundary gaps and recovering memory throughput can cut inter-token latency by 32–36%. The expert-weight traffic itself is unavoidable — but repeatedly interrupting that traffic and serializing auxiliary work around it does not have to be.

The padding arithmetic for the decode regime, with the E=256 / top-8 case worked out:

> The routing distribution affects the two layouts differently. Let R=Bk denote the number of routed token–expert pairs for routing fanout k. For grouped GEMM, flattening mitigates load imbalance across CTAs, but padding is maximized when the pairs are dispersed across as many experts as possible. In the decode regime where R ≤ E, the worst case assigns one pair to each of R experts, yielding M_sched = R · BLOCK_M and a useful-row fraction of only 1/BLOCK_M. For example, B = 8, k = 8, and BLOCK_M = 16 produce 64 useful rows among 1,024 scheduled rows, or 93.75% padding. Batched GEMM suffers the opposite extreme: if all tokens select the same k experts, then M_cap = B while only k of the E expert slices contain data. Its useful-row fraction is therefore k/E — 3.125% for E = 256 and k = 8 — even before any additional tile-alignment padding.

The SwapAB formulation — tokens go on the narrow MMA axis:

> To reduce this underutilization, MonoMoE adopts SwapAB by placing tokens on the small-N side of the MMA. Let X_e of shape B_e × D denote the compact activations selected for expert e, and let X̃_e of shape B × D denote an expert-masked view of the original token tile. For expert weight W_e of shape D × O, the conventional and weight-major formulations are
>
> Y_e = X_e W_e            : (M,N,K)_std    = (B_e, O, D),        (9)
>
> Ỹ_e^⊤ = W_e^⊤ X̃_e^⊤      : (M,N,K)_SwapAB = (O, B, D),
>
> where D is the reduction dimension and O is the expert output width. SwapAB exchanges the logical operand roles and produces a transposed output layout; it does not launch a transpose kernel or move an additional tensor through global memory. For FP8, Hopper WGMMA supports N = 8,…,256 in multiples of eight, and Blackwell's UMMA/tcgen05 path provides analogous small-N shapes. MonoMoE selects the smallest supported N that covers the step's token count.

Persistent grid structure and warp roles:

> MonoMoE persistent-grid mapping. The kernel launches one CTA per SM, so the grid contains N_SM CTAs on a GPU with N_SM streaming multiprocessors. Each CTA contains 12 warps: eight compute warps that issue tensor-core work and four prefetch warps that drive asynchronous data movement. With 32 threads per warp, each CTA has 384 threads.

> Example MonoMoE persistent schedule with three experts and six CTAs. During the up-projection, CTAs form three two-CTA expert groups: CTAs in the same group stream the same expert's weights while computing disjoint output stripes. The groups then advance through experts in round-robin order. The down-projection uses a different grouping, with three CTAs per active expert, but follows the same pattern of expert-local weight streaming and stripe-wise output partitioning.

On-chip buffer aliasing — how they keep one CTA per SM resident:

> Shared-memory reuse. Persistence is only practical if the live state of the fused layer fits within the per-SM shared-memory budget. MonoMoE uses a lifetime-aware shared-memory layout in which buffers with disjoint lifetimes alias the same storage. The routing-window input tile, the up-projection weight tile, and the down-projection weight tile share one region because their lifetimes do not overlap; similarly, epilogue scratch buffers are collapsed into a single reusable region. This reuse keeps the persistent footprint small enough to maintain one CTA per SM at the target tile sizes.

The negative result on fusing collectives — directly relevant to our TP16:

> Folding these collectives into the kernel does not improve single-node, small-batch decode because NVLink transfers the small messages in only a few microseconds, leaving cross-rank synchronization as the dominant exposed cost. A waiting persistent kernel also keeps one CTA resident on every SM, causing the grid to spin while communication traffic competes with the bandwidth-bound expert-weight stream. TP offers no useful overlap window because its all-reduce is terminal, and MonoMoE's atomicAdd-based accumulation does not produce an output element until all contributing expert groups finish. Exposing partial results earlier would require per-group buffers, roughly doubling output traffic and adding a reduction pass, while the receive would still wait for the slowest rank. Separate TP collectives are therefore preferable in MonoMoE's target regime; fusion becomes more attractive at higher TP degrees or during prefill, where larger transfers pro[vide]...

End-to-end results, including GLM-5.2:

> In end-to-end serving, MonoMoE reduces mean time per output token by 9.9–18.7% on Qwen3.5-35B, 9.1–11.5% on Qwen3.5-122B, and 2.9–16.7% on GLM-5.2. DeepSeek-V3.1's eight-way tensor-parallel path dilutes the local-kernel gain: TPOT ranges from a 1.2% regression to a 3.7% reduction, while aggregate output throughput improves by up to 31.8%.

> The decline is most pronounced for GLM-5.2 because its TP=8 serving path includes communication and non-MoE work that the local kernel does not reduce.

## Relevance

This is our problem, on our model, one generation of hardware away. Their baseline numbers
(two expert GEMMs at 17–22% of peak DRAM bandwidth with batch=1) are the same shape as ours
(GroupedMatmulSwigluQuantV2 at 2.272/6.793 = 33%, GroupedMatmul at 1.212/5.596 = 22%). Four
design decisions transfer directly: (1) partition cores over expert-weight output stripes, not
over tokens — at batch=1 the weight matrix is the only thing with enough parallelism; (2) put
the token count on the narrow MMA axis (SwapAB), which on Ascend means `baseM` = token count
and `baseN` = output width rather than the reverse; (3) alias on-chip buffers by lifetime so
the whole fused layer stays resident; (4) do **not** try to fuse the TP16 all-reduce into the
megakernel — their measurement says the terminal all-reduce has no overlap window and a
spinning persistent grid actively competes with the weight stream. Their 93.75% padding figure
for `B=8, k=8, BLOCK_M=16` is the quantitative indictment of the grouped-GEMM layout at our
operating point.

---

## 3. Masked grouped GEMM: the device-side group-count contract

### DeepGEMM README — contiguous vs masked layouts, and Mega MoE

https://github.com/deepseek-ai/DeepGEMM
Raw: https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md — DeepSeek.

> #### Grouped GEMMs (contiguous layout)
>
> Unlike traditional grouped GEMMs in CUTLASS, DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for scenarios where experts in an MoE model share the same shape. For training forward passes or inference prefilling, where each expert may process a varying number of tokens, we concatenate these tokens into a single tensor, referred to as the "contiguous" layout. Note that each expert segment must be aligned to the GEMM M block size (`get_mk_alignment_for_contiguous_layout()`).  For more information, please refer to the `m_grouped_fp8_gemm_{nt, nn}_contiguous` function documentation.
>
> #### Grouped GEMMs (masked layout)
>
> During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions.
>
> Use `m_grouped_fp8_gemm_nt_masked` for this purpose and consult the relevant documentation. An example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.
>
> #### Mega MoE
>
> Mega MoE fuses and overlaps EP dispatch, linear 1 and linear 2 (FP8xFP4 or FP8xFP8), SwiGLU, and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory.

### `m_grouped_fp8_fp4_gemm_nt_masked` — exact signature and the `expected_m` hint

https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/csrc/apis/gemm.hpp

```cpp
static void m_grouped_fp8_fp4_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& expected_m,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             std::optional<std::tuple<int, int>> recipe_a,
                                             std::optional<std::tuple<int, int>> recipe_b,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast,
                                             const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());
    ...
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
```

```cpp
    m.def("m_grouped_fp8_fp4_gemm_nt_masked", &m_grouped_fp8_fp4_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("epilogue") = nullptr);
    ...
    m.attr("m_grouped_fp8_gemm_nt_masked") = m.attr("m_grouped_fp8_fp4_gemm_nt_masked");
```

Test-side usage, including the efficiency caveat:

https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/tests/test_fp8_fp4.py

```python
def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    ...
        expected_m = int(expected_m_per_group * 1.2)

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m)
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        ...
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a_, b_, d_, grouped_layout, expected_m, **common)
```

## Relevance

The masked contract is the correct shape of the problem at batch=1: static `[G, M, K]`, one
device-resident `int32 masked_m[G]`, and a scalar `expected_m` used only for tile scheduling.
Compare this to CANN's `groupList` in `GroupedMatmulSwigluQuantV2`, which the kernel walks with
scalar GM reads (see Section 1 above). The stated reason for the masked design — `the CPU is
unaware of the number of tokens each expert receives` under graph capture — is exactly our
situation, since we must stay inside an aclgraph.

---

### DeepGEMM Mega MoE host API — one kernel for dispatch + L1 + SwiGLU + L2 + combine, with the shared expert folded in

https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/csrc/apis/mega_moe.hpp

```cpp
static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
)
```

The intermediate-activation ring buffer, sized from a closed-form live-block bound instead of
a full `[E, capacity, H]` staging buffer:

```cpp
    // Ring capacity: worst-case live pool blocks over all candidate BLOCK_M; mirrors the kernel assert.
    // TODO: we temporarily assume the SM count is consistent with the runtime value
    const auto num_sms = get_num_sms_for_mega_moe();
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;
    ...
    int num_ring_tokens = 0;
    for (const auto& block_m: layout::kCandidateBlockM) {
        const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
    }
    num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);
```

The shared expert is expressed as an integer multiple of the routed expert's intermediate width:

```cpp
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = check_weights_layout_3d(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = check_weights_layout_3d(l2_weights);
    ...
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    ...
        shared_intermediate_hidden = shared_l2_k;
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;
```

## Relevance

`num_shared_experts = shared_intermediate_hidden / intermediate_hidden` is the trick for
putting our W8 shared expert in the same megakernel as the W4A8 routed experts: treat it as
extra expert slices of the same shape rather than as a separate op (we currently pay a separate
`QuantBatchMatmulV3` at 3.810 ms vs 0.780 ms of bytes for it). The ring-buffer sizing shows the
intermediate `[8, 128]` activations never need a full-capacity staging buffer — at batch=1 our
live set is 8 rows of 128 per rank, trivially on-chip.

---

### DeepGEMM MegaMoE in-kernel scheduler — task descriptors, bitmask inter-stage dependency, warp roles

https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh

```cpp
// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;
```

The XOR bitmask that lets Linear2 start per-fed-tile instead of after a layer barrier:

```cpp
// Each finished L1 N block toggles its bit in `l2_full_mask`, so L2 K blocks can start as soon as they are fed
// NOTES: the mask parity per ring generation relies on the ring capacity, see `get_num_max_live_pool_blocks`
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : (1ull << kNumL1BlockNs) - 1;
    ...
    CUTLASS_DEVICE static void arrive(const uint64_t* mask_ptr, const uint32_t& n_block_idx) {
        ptx::red_xor_rel(mask_ptr, 1ull << n_block_idx);
    }

    // Re-read the mask only when the K block is not fed yet
    CUTLASS_DEVICE void wait(const uint32_t& k_block_idx) {
        const auto k_block_mask = ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ptx::ld_acq_gpu(mask_ptr) ^ expected_mask;
    }
};
```

The locality/work-stealing rule:

```cpp
    // Locality domain affinity. When a queue is drained, we try work stealing to ensure load balancing.
    // But when there are few tasks, we disable work stealing to avoid cross-die latency.
    static constexpr uint32_t kNumMinStealWaves = 2;
```

Expert token counts fetched lane-parallel, spinning on a completion sentinel:

```cpp
    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
```

Warp-role split and the swap-AB MMA config:
https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh

```cpp
    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    ...
    // Prefetch so the TMEM stage is released earlier, keeping the epilogue off the critical path
    constexpr uint32_t kNumEpiPrefetchStages = BLOCK_M == 240 ? 1 : 0;
```

```cpp
    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        ...
        // Dispatch warps
    } else if (warp_idx == kNumDispatchWarps) {
        // GEMM TMA load warp for tokens with SFA
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // GEMM TMA load warp for weights with SF
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // GEMM MMA issue warp (only the leader CTA will run)
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens, sm_locality_domains);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
```

## Relevance

The mechanism set we need, in production form: a 16-byte aligned task descriptor, a 64-bit XOR
bitmask so the down GEMM's K-blocks start as each up-GEMM N-block lands (instead of a full
barrier between the two GEMMs), `UMMA_N = BLOCK_M` as the swap-AB encoding, and a dedicated
scheduler role on one core. On Ascend the bitmask is replaceable with a workspace counter plus
`CrossCoreWaitFlag`. The comment `when there are few tasks, we disable work stealing to avoid
cross-die latency` is a direct warning for a TP16 A3 superpod: at batch=1 there are few tasks,
so static assignment beats any dynamic balancing.

---

## 4. Token-major grouped GEMM at batch=1: the padding tax, measured in framework code

### vLLM `moe_align_block_size` — the padding formula and the worked example

https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/moe_align_block_size.py

```python
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    return_scatter_idx: bool = False,
) -> ...:
    """Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.
    ...
    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    ...
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
```

## Relevance

`max_num_tokens_padded = numel + num_experts * (block_size - 1)` is the token-major layout's
cost in closed form. At our point `numel() = 8` and local `num_experts = 16` at TP16 (256
global), so `numel() < num_experts` and even vLLM falls back to `numel * block_size` — 8
useful rows occupying 8 whole M-tiles. This is the same conclusion MonoMoE reaches analytically.

---

### vLLM Triton `fused_moe_kernel` — the `naive_block_assignment` batch-1 bypass

https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/fused_moe.py

```python
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, b_bias_ptr, a_scale_ptr, b_scale_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    group_n: tl.constexpr, group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr, use_int8_w8a8: tl.constexpr, use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr, HAS_BIAS: tl.constexpr, SWAP_AB: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    """...
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    ...
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
```

The small-batch invalid-block skip:

```python
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
```

## Relevance

vLLM's own escape hatch: at small batch it abandons `sorted_token_ids` and maps one program to
one token. This is permission to drop the permutation/`moe_init_routing` stage entirely at
batch=1 and index the 8 routed expert weights directly — on Ascend that deletes a whole op from
the chain rather than making it faster.

---

### vLLM decode-tile heuristics for block-wise FP8 MoE — the small-M reasoning in comments

https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/fused_moe.py
and https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/utils.py

```python
    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant. Use a small M tile for decode-like batches where
        # tokens are spread thin across experts. Larger batches benefit from
        # GROUP_SIZE_M > 1 because the per-block scales add memory traffic
        # that benefits from L2 tile reuse.
        #
        # BLOCK_SIZE_N need not equal block_shape[0]: the kernel indexes block
        # scales per N element (offs_bn // group_n), so any N tile dividing the
        # quant block is valid. At decode a 128-wide N tile leaves the gate-up
        # GEMM SM-bound; a 64-wide tile exposes ~2x the thread blocks, and the
        # swap-AB kernel keeps it efficient on Hopper down to the smallest
        # batches, so prefer N=64 through low batch sizes. CUDA only (validated
        # on NVIDIA); ROCm keeps its prior tile/pipeline sizes.
        if current_platform.is_rocm():
            block_n = block_shape[0]
            num_stages = num_stages_rocm
        elif M <= 8 and block_shape[0] % 64 == 0:
            block_n = 64
            # The smallest batches are memory-latency bound, so a deeper
            # pipeline hides the weight loads; by M=8 it turns occupancy/SMEM
            # bound and the extra stages hurt.
            num_stages = 4 if M <= 4 else 3
        ...
        config = {
            "BLOCK_SIZE_M": 16 if M <= 64 else 64,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 1 if M <= 16 else 32,
            "SPLIT_K": 1,
            "num_warps": 4,
            "num_stages": num_stages,
        }
```

```python
@functools.lru_cache
def enable_swap_ab(BLOCK_SIZE_M: int, BLOCK_SIZE_N: int) -> bool:
    return (
        current_platform.is_device_capability(90)
        and BLOCK_SIZE_M < 64
        and BLOCK_SIZE_N >= 64
    )
```

## Relevance

Two transferable levers stated as comments by people who tuned them: at the smallest batches
increase *pipeline depth* to hide weight loads rather than tile size, and narrow the N tile to
expose more blocks. The swap-AB gate `BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64` is the
numeric threshold at which putting tokens on the narrow axis wins.

---

### TensorRT-LLM "min latency mode" — the batch-1 MoE data layout with no permutation arrays

https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/moe/cutlass/include/moe_kernels.h

```cpp
struct MoeMinLatencyParams
{
    // All these are allocated on device memory
    // Number of active experts on current node; smaller than or equal to num_experts_per_node
    int* num_active_experts_per_node;
    // The score of each token for each activated expert. 0 if the expert is not chosen by the token.
    // Only the first num_active_experts_per_ rows are valid
    float* experts_to_token_score;
    // The global expert id for each activated expert
    // Only the first num_active_experts_per_ values are valid
    int* active_expert_global_ids;
```

https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/moe/cutlass/moe_kernels.cu

```cpp
/**
 * Takes the input maps and prepares the expanded maps for min latency
 ...
template <int BLOCK_SIZE>
__global__ void buildMinLatencyActiveExpertMapsKernel(...)
{
    cudaGridDependencySynchronize();
    // Use one block to process the min latency case
```

```cpp
    size_t num_moe_inputs = min_latency_mode ? num_experts_per_node * num_rows : experts_per_token * num_rows;
    ...
    size_t const permuted_row_to_unpermuted_row_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const permuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
```

```cpp
    if (min_latency_mode)
    {
        TLLM_CHECK_WITH_INFO(using_tma_ws_gemm1, "Only TMA warp specialized GEMM is supported in min latency mode.");
        // TODO: as for bias, need to get the correct expert id according to the active expert global ids
        TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Min latency mode does not support bias.");
        TLLM_CHECK(use_fp4);
    }
    ...
    if (min_latency_mode)
        return;   // (in gemm2: no separate finalizeMoeRouting pass)
```

## Relevance

NVIDIA's production answer for minimum-latency MoE abandons the permutation arrays entirely
(`permuted_row_to_unpermuted_row_size = 0`), replaces them with a dense
`[num_experts_per_node, num_rows]` score matrix, builds the routing metadata in a single block,
and deletes the separate finalize pass. That is a complete alternative layout for the batch=1
regime, and it is the layout our megakernel should adopt instead of the prefill layout CANN's
`moe_init_routing` + `GroupedMatmul` + `moe_finalize_routing` chain assumes.

---

### TensorRT-LLM hand-written batch-1 MoE kernel — one block per output row, no MMA at all

https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/llama4MinLatencyMoEOp.cu

```cpp
#define NUM_EXPERTS 128
#define HIDDEN_SIZE 5120
#define INTER_SIZE 1024
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define TILE_ROW 4
#define VEC_SIZE 8

#define ENABLE_ACQBULK 1
#define ENABLE_PREFETCH 1
#define ENABLE_PREEXIT 1
```

```cpp
// This is the hand-optimized kernel.
// The computation is:
//   C = silu(AxB_gated * in_scale * sigmoid(logit)) * (AxB_linear * in_scale * sigmoid(logit)) * out_scale_inv
// The out_scale_inv cannot be fused with in_scale because silu() is non-linear.
// Also, Llama-4 applies score scaling, which is sigmoid(logit), on tensor A.
__global__ void llama4_moe_fc13_swiglu_fp8_kernel(int num_tokens,
    __nv_fp8_e4m3 const* __restrict__ A,      // Input tensor [num_tokens][HIDDEN_SIZE]
    __nv_fp8_e4m3 const* __restrict__ B,      // Input tensor [num_experts][INTER_SIZE*2][HIDDEN_SIZE]
    __nv_bfloat16 const* __restrict__ logits, // Input tensor logits [num_tokens][num_experts]
    __nv_fp8_e4m3* __restrict__ C,            // Output tensor [num_tokens][INTER_SIZE]
    int* __restrict__ exp_idx,                // Output tensor [num_tokens]
    ...
    int const token_idx = blockIdx.x / INTER_SIZE;
    int const row = blockIdx.x % INTER_SIZE; // Matrix row / Output element index
    ...
    // Preload the scaling factors before ACQBULK.
    __shared__ float in_scales_shared[NUM_EXPERTS];
    in_scales_shared[tid] = in_scales[tid];

    // Logits depends on the previous kernel, so we cannot prefetch anything.
#if ENABLE_ACQBULK
    cudaGridDependencySynchronize();
#endif

    // Perform top1 within the current thread, which processes 4 experts.
    ...
    // Perform top1 across threads using Warp reduction.
    // We pack logit and expert index into an int so that we can use integer max op for reduction.
    int best_result
        = ((int) (__bfloat16_as_short(best_logit) ^ (best_logit < __nv_bfloat16(0.f) ? 0x7fff : 0)) << 16) | best_exp;
    ...
#if ENABLE_PREEXIT
    cudaTriggerProgrammaticLaunchCompletion();
#endif
```

```cpp
    int const grid_size = num_tokens * INTER_SIZE;
    ...
    launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, (void*) llama4_moe_fc13_swiglu_fp8_kernel, args, 8);
```

The second GEMM maps `TILE_ROW = 4` output rows per block:
`int const grid_size = num_tokens * HIDDEN_SIZE / TILE_ROW;`

## Relevance

The fully-specialized extreme, and it beats grouped GEMM at batch=1: no matrix unit at all,
routing + gate + up + SwiGLU + requantize inside one block whose grid is the intermediate
width, all scale factors preloaded *above* the dependency-wait point so the wait costs nothing.
On Ascend the analogue is to compute per-output-stripe on AI_VECTOR_CORE (0.70 us floor, vs
AI_CORE 2.30 us and MIX_AIC 13.58 us) and hoist every scalar/scale load above the
`CrossCoreWaitFlag`. Note also the hard-coded shapes — this kernel is specialized to one model,
which is exactly the licence our brief gives us.

---

### FlashInfer `cutlass_fused_moe` — `min_latency_mode` and finalize fused into the GEMM2 epilogue

https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/flashinfer/fused_moe/core.py

```python
@flashinfer_api(trace=cutlass_fused_moe_trace)
def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    fc1_expert_biases: Optional[torch.Tensor] = None,
    fc2_expert_biases: Optional[torch.Tensor] = None,
    input_sf: Optional[torch.Tensor] = None,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1, tp_rank: int = 0,
    ep_size: int = 1, ep_rank: int = 0,
    cluster_size: int = 1, cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    ...
    enable_pdl: Optional[bool] = None,
    activation_type: ActivationType = ActivationType.Swiglu,
    swizzled_input_sf: bool = True,
    use_fused_finalize: bool = True,
    ...
) -> torch.Tensor:
    """Compute a Mixture of Experts (MoE) layer using CUTLASS backend.

    This function implements a fused MoE layer that combines expert selection, expert computation,
    and output combination into a single operation.
```

```
    min_latency_mode : bool = False
        Whether to use minimum latency mode. Defaults to False.
```

```
    use_fused_finalize : bool = True
        Whether to fuse the top-k expert reduction ("finalize") into the GEMM2 epilogue.
        Defaults to True for best performance. The fused epilogue reduces expert outputs via
        non-associative atomics, so results are not deterministic run-to-run. Set to
        False to use the non-fused, deterministic finalize path.
```

```
    enable_pdl : Optional[bool]
        Whether to launch the kernel with Programmatic Dependent Launch (PDL).
        ``None`` (default) lets the runtime pick a safe value based on the device
        and surrounding stream operations; pass ``True`` to force PDL when every
        adjacent kernel on the stream also supports it, or ``False`` to disable.
```

## Relevance

The top-k combine is folded into the down GEMM's epilogue by default, with the nondeterminism
cost stated explicitly. CANN has the equivalent op already —
`GroupedMatmulFinalizeRouting` ("GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作", see below) — so this fusion boundary is available to us
without writing it. Note our `aclnnGroupedMatmulSwigluQuantV2` is documented as
`默认为确定性实现`, so adopting an atomics-based combine is a deliberate determinism trade.

---

### CANN `GroupedMatmulFinalizeRouting` — the down-GEMM + combine fusion that already exists on A3

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_finalize_routing/README.md

> # GroupedMatmulFinalizeRouting
>
> ## 产品支持情况
>
> | 产品 | 是否支持 |
> | :---------------------------- | :-----------: |
> | <term>Ascend 950PR&950DT系列产品</term> | √ |
> | <term>Atlas A3系列产品</term> | √ |
> | <term>Atlas A2系列产品</term> | √ |
>
> ## 功能说明
>
> - 算子功能：GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作

> | x1 | 输入 | 输入x(左矩阵)。 | INT8、FLOAT8_E5M2<sup>1</sup>、FLOAT8_E4M3FN<sup>1</sup>、HIFLOAT8<sup>1</sup>、FLOAT4_E2M1<sup>1</sup> | ND |
> | x2 | 输入 | 输入weight(右矩阵) | INT4<sup>2</sup>、INT8、INT32<sup>2</sup>、FLOAT8_E5M2<sup>1</sup>、FLOAT8_E4M3FN<sup>1</sup>、HIFLOAT8<sup>1</sup>、FLOAT4_E2M1<sup>1</sup> | ND、NZ |
> | scaleOptional | 输入 | 量化参数中的缩放因子，perchannel量化参数 | INT64<sup>2</sup>、BFLOAT16、FLOAT32、FLOAT8_E8M0<sup>1</sup> | ND |

The kernel directory shows a dedicated A8W4-MSD implementation plus an L1-optimized variant:

```
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_l1_opt.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_pre.h
```

## Relevance

INT4 weight (`x2` accepts `INT4`, superscript 2 = A2/A3 only) with an A8W4-MSD kernel and NZ
weight support — i.e. CANN already ships the down-GEMM + combine fusion for exactly our
quantization on exactly our chip. That op is the natural replacement for our separate
`GroupedMatmul` (5.596 ms vs 1.212 ms) plus whatever combine follows it, and the
`_l1_opt` variant name says someone has already worked the L1 residency of the weight stream.

---
## 5. Huawei's own account of why these ops exist, and what MSD costs

### 昇腾官方技术文章：GroupedMatmul W4A8 量化与 MSD 方案

https://www.hiascend.com/developer/techArticles/20250717-2
昇腾CANN，2025/07/17，《【昇腾热点算子大解密】CANN极致优化GroupedMatmul量化方案，MoE推理Decode性能提升超30%》

> 分组矩阵乘法（GroupedMatmul）是一种将多个独立的小规模矩阵乘法（GEMM）任务分组并行执行的优化技术，通过合并计算拓扑相似的算子为单一内核调用，有效降低核启动开销并减少内存访问碎片化。
>
> MoE模型在Decode阶段每个专家处理的激活数量偏少，导致传统的GEMM操作因单个专家计算规模小而难以充分发挥硬件算力。GroupedMatmul天然适配MoE模型的稀疏专家计算，通过将分散的专家任务合并为分组批量进行计算，提升任务规模以实现接近峰值算力的效率；同时，连续存储专家权重可以进一步优化缓存命中率，显著提升计算性能。

> 然而，仅依赖W4A8方案并不能充分释放低比特运算的潜力。推理流程中不同阶段对计算和访存的依赖有所区别，Prefill阶段的activation更长，计算更密集；而Decode阶段activation更短，访存压力更大，两者对硬件资源的需求和优化重点不同。面对Decode阶段的访存瓶颈，传统W4A8方案虽然能降低显存占用，但在实际计算时仍需将INT4权重还原为INT8后，再与INT8激活做矩阵乘法，有冗余的反量化计算开销。
>
> 因此，我们将W4A8量化方案与昇腾硬件微架构特性结合优化，创新性地提出MSD（Mixed-precision Split-activation Decomposition，混合精度激活拆分分解）方案，将INT8激活拆分为高、低两路INT4，并直接与INT4权重进行计算，用"高位×16+低位"合成，在保证结果等价于原始INT8×INT4计算结果的同时，大幅降低KV Cache和参数存储需求，提升单卡batch size和上下文的支持长度。

So MSD is named **Mixed-precision Split-activation Decomposition** — not "multi-scale".
On the three scale levels and whether per-group is worth it:

> W4A8量化方案采用三层量化scale设计，实现了精细化的量化控制，其数学公式表达如下： 其中量化scale分别为：
> (1) Per-channel scale：针对weight的通道级量化参数，保证不同通道间的精度平衡
> (2) Per-group scale：针对weight的分组级量化参数，提供更细粒度的量化控制
> (3) Per-token scale：针对activation的token级量化参数，动态适应不同输入的分布特性
> 这种多层级的量化策略能够更好地适应不同数据分布的特点，有效降低量化误差。实践中发现，per-group量化对模型精度的提升较为轻微，所以实践中一般可以不采用per-group量化。

On the offline bias term and the scale fusion:

> 当INT8被拆分为两个INT4时，由于INT4的符号位决定了其范围在[-8, 7]，而低4位的最高有效位（即符号扩展）会带来一个偏移量，具体就是8*weight。为了保证拆分的精度，需要将拆分操作变型为如下过程： 这个多余项在所有INT8拆分为INT4的操作中是固定的，等价于每个权重通道都引入了一个偏置项。在GroupedMatmul的具体实现中，通常将这部分多余项放入bias参数中，通过离线计算提前补偿，从而保证推理精度和性能。

> 如上图所示，x与weight被拆分后完成INT4矩阵乘。在实际推理时，为了提升效率，per-channel scale和per-group scale会离线融合成一个总scale（图中的anti-scale），这样可以在GEMM输出阶段一次性完成浮点还原，减少多次逐步缩放带来的性能损耗。
> 融合后的scale一般为total_scale = perchannel_scale * pergroup_scale。这样，矩阵乘法主环节只需用整数计算，最后输出时统一乘以融合后的scale即可。

> Per-group scale是针对权重的分组量化，每组权重共用一个scale。在GEMM实现中，通常沿着k轴（即输入通道）切分，每g个通道为一组（g为group size），每组使用独立的scale。
> 在做矩阵乘法时，需要将输入矩阵X（已分解为INT4）与权重矩阵W（INT4，已分组）分块处理。对每个group，通过total scale将INT4权重直接解码为FP16。具体而言，每进行一组累加，就应用对应的per-group scale。

The post-processing decomposition and the headline numbers:

> W4A8量化后处理主要包括三个步骤：高低位结果合成、bias补偿、反量化（per-token scale）。
>
> (1) 高低位结果合成：在W4A8方案中，INT8激活被拆分为高4位和低4位。为还原INT8原始乘法的效果，需要将高4位的结果左移4位（即乘以16），再与低4位结果相加。这样可以保证拆分计算后，数值与直接INT8×INT4乘法结果一致。
> (2) BIAS余量补偿：拆分INT8为两个INT4时，由于符号扩展和量化区间的差异，会引入一个额外的常量项（8×W×scale）。这个多余项在量化离线阶段提前计算，并作为bias参数加到最终输出上，补偿拆分带来的偏移，保证数值精度。在实际后处理时，直接将该bias向量加到矩阵乘法的结果上即可，无需额外计算，且不会影响推理效率。
> (3) Per-token反量化：W4A8方案通常采用per-token scale对激活进行量化，推理后需反量化还原浮点结果。

> 至此，W4A8量化与MSD方案的所有主流程均在整数域内完成，充分利用昇腾硬件对低比特整数计算的支持，减少主循环的访存和运算瓶颈，为LLM的高效推理提供了关键支撑，极大提升大模型推理的响应速度和服务能力。基于W4A8方案的DeepSeek R1单机部署实践表明，其最大BatchSize达到了W8A8方案的3.1倍；同时，W4A8+MSD方案在Decode阶段的性能更是达到了W8A8的1.3~1.7倍。

## Relevance

Huawei states the purpose of GroupedMatmul as `有效降低核启动开销` — amortizing kernel-launch
overhead across the small per-expert GEMMs — which is the same lever we are pulling one level
up. Two actionable items: `per-group量化对模型精度的提升较为轻微，所以实践中一般可以不采用per-group量化`
means we can likely run `dequantMode=0` (per-channel) and delete the per-group K-axis scale
application from the inner loop; and `per-channel scale和per-group scale会离线融合成一个总scale`
plus the `8×W×scale` bias both being offline means the only in-kernel scale work is the
per-token multiply and the final `max|S|/127`.

---

### 昇腾官方技术文章：通算融合 Dispatch/Combine — 下发时延随 EP 规模线性增长

https://www.hiascend.com/developer/techArticles/20250726-1
昇腾CANN，《【昇腾热点算子大解密】CANN创新通算融合算子破局MoE通信瓶颈，实现推理吞吐50%提升》

> 针对MoE架构，我们的技术团队深入分析后发现，高额的通信开销主要源于两大技术挑战：
> 动态专家选择机制下，每个Token分发的目标专家呈现离散分布特征，导致两个关键问题：
> 数据分发不均匀：不同专家接收的Token长度存在差异，需依赖低效的AllToAllV通信；
> 元数据同步开销：获取收发信息需调用前置AllGather算子收集路由表，并在Host侧完成同步，引入额外通信开销和Stream同步延迟；
> 算子下发延迟：传统Host驱动通信需要构造子图并调度，其下发时延随EP规模线性增长；

> 由上可知，Dispatch/Combine操作本质上是计算与通信的结合。因此，我们创新型地实现了MoeDistributeDispatch和MoeDistributeCombine这两个通算融合算子，将路由计算等Host侧逻辑下沉至Device侧，消除了Host与Device间的同步开销。同时，将Combine操作中所需的参数计算、Token的加权求和等计算操作与AllToAllV通信流水并行，也得以实现计算与通信的耗时掩盖。
>
> 与此同时，AIV将数据在共享内存中的地址、数据长度信息通过GM直接传递给同处Device侧上的AICPU，由AICPU直接驱动RDMA通信，彻底摒弃了传统需要Host侧构造子图和调度RDMA任务的繁琐流程，不仅解决了Host侧处理耗时长的问题，更消除了传统调度方式带来的额外时延。

> 以BS = 16，H = 7168，topK = 8，数据类型FP16场景为例：
> 全互联方案下，总通信量近似为BS * H * topK * sizeof(FP16) * 63/64 = 1.75(MB)（63/64是因为发往本卡的MoE专家无需通过RDMA通信），总通信时间约86.1us。
> 分层方案下，每个Token平均选择4个node，机间通信量BS * H * 4 * sizeof(FP16) * 7/8 = 0.766(MB)，机间通信时间约为38.3us，机内通信量为BS * H * 8 * sizeof(FP16) * 7/8=1.53(MB)，机内通信时间约8us，总通信时间46.3us。

> 虽然实现了计算与通信的初步融合，但受限于AICPU驱动RDMA时WQE串行下发的固有机制，导致计算与通信、机间与机内通信无法实现深度流水并行，且下发时延随下发次数线性增长的问题迫使数据必须按目标rank汇聚。
>
> 为了提高流水并行度，技术团队在AIV+AICPU分层方案的基础上，创新性地实现了AIV直驱RDMA的通路。该方案不再依赖AICPU，实现更低时延，且多个AIV核可以并行下发WQE，WQE下发时延不再是算子耗时的关键，彻底解除了Token数据需按照目标rank汇聚的限制，可以直接以Token为粒度发送；… 最终使得算子平均性能较原有AIV+AICPU的分层方案提升10%+

## Relevance

The communication side of the same argument: `算子下发延迟：…其下发时延随EP规模线性增长` and
`下发时延随下发次数线性增长的问题迫使数据必须按目标rank汇聚`. The AIV-direct-RDMA path removes
the per-rank aggregation constraint, which matters if our megakernel wants to emit tokens at
token granularity rather than batching by destination rank.

---

### SuperKernel 技术综述 — what a megakernel actually recovers on Ascend, decomposed

https://cann.csdn.net/69afcc5554b52172bc603633.html
《大模型推理加速利器SuperKernel技术综述》，2025-11-24

> 该方法将整个网络模型重新编译为一个大算子，从而减少硬件调度开销并优化了Cache和同地址访问，在现有网络优化基础上提升整网性能10-20个百分点。

> 以DeepSeekV3网络中的一层为例，根据Profiling数据分析发现，网络上肉眼可见的Task等待调度开销较为明显。在BatchSize大小为96、Sequence长度3K的场景下，平均每层的Task平均等待的开销约为18us左右。按照61层网络计算，总的调度等待开销在1.1ms左右。这些开销基本是绝对开销，在大的Batch场景下还会进一步扩大。

> 通过详细评估和调试，SuperKernel技术优化后的基本收益模型可以分为以下三部分：首先是两个Task之间的调度开销降低；其次是每个Task结束后会执行Cache Flush操作，将所有修改的Cache内容刷新出去，其中包含一些不必要的栈空间；最后是算子调度后的核启动时间，包括同地址访问的排队延迟、Scalar执行初始化等环节。

> 当前已经支持将DeepSeekV3模型61层网络中的58层融合为1个SuperKernel算子，整网收益平均10-20%左右，远超前面的任务调度分析的开销部分。

> 1）ICache Preload优化
> 在SuperKernel执行过程中，由于运行时系统通常仅预取其入口点的指令，子Kernel的代码段往往无法被硬件预取机制有效捕获，导致指令缓存（ICache）未命中率增高。… 该技术在当前子Kernel执行完毕前，以2KB对齐的方式，提前将下一个即将执行的子Kernel代码段预取到ICache中。
>
> 2）同地址访存优化和亲和部署
> 在多核处理器系统中，当多个计算核心同时执行同一个内核函数时，它们会访问内存中完全相同的指令地址（Text段）。这种对同一地址的并发访问会在共享的L2缓存层面形成一个串行化的访问队列，导致多个核心的请求发生争用，即使数据已缓存在L2中，也会因队列拥堵产生访问延迟，显著削弱多核并行带来的性能提升。… 可将子内核的代码段复制为多份副本，使不同核通过核ID映射到不同的物理地址执行。
>
> 3）Early-start优化
> 在常规的Task调度机制或SK内部同步前后两个算子之间，系统通常需要等待所有硬件线程完成当前操作后才启动下一个算子。但实际上，大多数算子在启动初期会先进行 Scalar计算，随后才执行数据搬入和搬出操作。… 首先，在前一个Kernel的MTE3搬出阶段，设置线程级的同步点（set）；其次，在下一个Kernel的MTE2搬入阶段前，仅对数据依赖部分进行同步等待（wait）； 然后，在Scalar阶段无需等待前一个Kernel完成，可直接启动，从而实现两个Kernel之间的"早启动"流水线重叠。

> 2）支持双流并发融合
> 通过多流实现Cube/Vector并发场景后，在简单转换为SK时，如果不感知依赖关系，仅依据执行顺序进行处理，SK内部的执行将退化为串行，导致整体性能相比融合前有所下降。… 可以将算子按类型进行分类，并在SK内部根据Cube和Vector的不同特性分配执行队列；同时，结合流的属性和Event事件，精准插入同步点，从而实现Cube与Vector的高效并发执行。

## Relevance

The authoritative Ascend-specific decomposition of megakernel gains, and it is **not only
launch count**: (1) inter-task scheduling, (2) the per-task Cache Flush that writes back
unnecessary stack space, (3) core-start time including same-address L2 queueing and Scalar
init. Three concrete techniques to copy: ICache preload of the next sub-kernel at 2 KB
alignment, replicating the text segment so cores map to different physical addresses, and
Early-start (set at the previous kernel's MTE3, wait only on the data-dependent part before
MTE2, start Scalar immediately). The last warning is the trap: naive fusion serializes
Cube/Vector unless you partition execution queues by op type.

---

### vLLM-Ascend 优化实践 — parasitic Transpose/Cumsum around GroupedMatmul, measured

https://cann.csdn.net/69afcc540a2f6a37c59646d7.html
《vLLM-Ascend：大模型推理的优化实践》，2025-11-28

> PyTorch的Eager模式可以通过Ascend Extension for PyTorch（torch_npu）无缝对接昇腾芯片，但推理总耗时易受算子下发调度开销影响，出现调度Bound现象，从而显著影响性能。以DeepSeek-V3的MLA结构为例，下图展示了PyTorch在Eager模式下的调度开销情况：NPU在大部分时间处于空闲状态，CPU侧算子调度开销成为推理耗时的关键因素。

> DeepSeek-V3模型普遍使用大EP的推理部署方式，通过减少MoE部分参与GroupedMatmul计算的路由专家数量，以降低单次Decode耗时。然而，在分析EP256配置下的profiling时发现，GroupedMatmul算子执行前出现了额外的Transpose操作，导致性能反而出现较大劣化。
>
> 上图的profiling信息显示，单层MoE中的Transpose算子总耗时高达175us。经过模型代码排查、对比EP128无Transpose场景的dump图，发现问题由以下原因导致：
> 在EP256单卡1专家的配置下，由于专家权重shape中存在值为1的维度，导致针对权重的Transpose算子触发TorchAir的"enable_view_optimize"优化…而被插入了额外的Reshape算子。
> 新引入的Reshape算子使融合Pass "GroupedMatmulTransFusionPass"无法检测到其预定目标模式，最终导致Transpose算子被保留到图执行阶段。

> 对应的torch_npu接口"npu_grouped_matmul"依赖入参"group_list"获取每个子矩阵运算中实际M轴的大小，并支持以"累加和数列"或"逐维度指定"两种形式提供这一参数。vLLM-Ascend的模型脚本中使用了"累加和数列"模式，因而在MoE层中引入了用于实现累加和的Cumsum算子。
>
> 调整TorchAir的优化功能开关"enable_view_optimize"和torch_npu接口"npu_grouped_matmul"的配置后，MoE阶段关键路径上的Transpose/Cumsum算子即被消除。尽管GroupedMatmul算子耗时有所增加，仍能获得单层158us，整网9ms的Decode性能收益，在3K推理长度下可将rollout总耗时优化超过28s。

## Relevance

The decisive sentence is `尽管GroupedMatmul算子耗时有所增加，仍能获得单层158us，整网9ms的Decode性能收益` —
deleting the parasitic ops paid 158 us/layer and 9 ms whole-network **even though
GroupedMatmul itself got slower**. That is the exact shape of the trade a megakernel makes, and
it is the reason not to judge the megakernel by per-op timings. Also names `group_list` cumsum
mode as the source of an extra Cumsum launch per MoE layer, which is live in our path.

---

## 6. vllm-ascend: the W4A8 call sites, weight layouts, and the fused MC2 megakernel that already exists

### The two measured ops, as the framework calls them

https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/device/device_op.py
`BaseDeviceAdaptor` (A2/A3), lines ~334-425.

```python
    @staticmethod
    def npu_grouped_matmul_swiglu_quant(
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scale: torch.Tensor,
        bias=None,
        use_mxfp_quant: bool = False,
        act_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        weight_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        swiglu_limit: float = 0.0,
        mxfp_quant_dtype: QuantType | None = None,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        return torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            x_scale=x_scale,
            group_list=group_list,
            bias=bias,
            swiglu_limit=swiglu_limit,
        )
```

```python
    @classmethod
    def npu_grouped_matmul_gmm2(
        cls,
        *,
        hidden_states: torch.Tensor,
        weight: list[torch.Tensor] | torch.Tensor,
        weight_scale: list[torch.Tensor] | torch.Tensor,
        per_token_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
        ...
    ) -> torch.Tensor:
        ...
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=weight,
            scale=weight_scale,
            bias=bias,
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=fallback_output_dtype,
        )[0]
```

### `AscendW4A8DynamicFusedMoEMethod` — the full expert chain we are replacing

https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/quantization/methods/w4a8/w4a8.py

```python
        hidden_states, swiglu_out_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=w1,
            weight_scale=w1_scale,
            x_scale=pertoken_scale,
            group_list=group_list,
            weight_assist_matrix=bias1,
            dequant_mode=0,
            group_list_type=group_list_type,
            swiglu_limit=mlp_compute_input.swiglu_limit,
        )
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states, swiglu_out_scale
```

```python
    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        return torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        _, _, w2, w2_scale = self._get_mlp_weights(mlp_compute_input.layer)
        _, bias2 = self._get_mlp_scale_bias(mlp_compute_input.layer)
        group_list, group_list_type = self._maybe_convert_group_list(mlp_compute_input)
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            bias=bias2,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
```

### The exact W4A8 weight/scale byte layout the kernel consumes

Same file, class docstring:

```
    Original MoE layer weights generated by msModelSlim:

    - ``model.layers.L.mlp.experts.E.gate_proj.weight``:
      ``torch.int8``, ``[moe_intermediate_size // 2, hidden_sizes]``.
      Each int8 element stores two packed 4-bit weight values along the
      output dimension.
    ...
    - Each linear also has ``scale_bias``: ``torch.float32``,
      ``[moe_intermediate_size, 1]`` for ``gate_proj`` and ``up_proj``,
      and ``[hidden_sizes, 16 // tp_size]`` for ``down_proj``.
    ...
    After processing, ``apply`` passes these tensors to the fused MoE
    operator:

    - ``w13_weight``: ``torch.int32``,
      ``[num_experts, hidden_sizes, moe_intermediate_size // 4]``.
    - ``w2_weight``: ``torch.int32``,
      ``[num_experts, moe_intermediate_size, hidden_sizes // 8]``.
    - ``w13_scale_bias``: ``torch.float32``,
      ``[num_experts, 2 * moe_intermediate_size]``.
    - ``w2_scale_bias``: ``torch.float32``,
      ``[num_experts, hidden_sizes]``.
    - ``w13_weight_scale``: ``torch.int64``,
      ``[num_experts, 2 * moe_intermediate_size]``.
    - ``w2_weight_scale``: ``torch.int64``,
      ``[num_experts, 1, hidden_sizes]``.
```

And the transforms that produce it:

```python
    def _pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        assert weight.shape[-1] % 4 == 0, (
            f"the last dim of weight needs to be divided by 4 but got shape {weight.shape}"
        )
        return weight.view(torch.int32).contiguous()

    def _process_scale(self, scale):
        scale = scale.transpose(1, 2).contiguous()
        scale_np = scale.cpu().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
        return scale_uint64_tensor

    def _pack_int4_to_int8(self, weight: torch.Tensor) -> torch.Tensor:
        shape = weight.shape
        weight = weight.reshape(-1, 2)
        weight0 = weight[:, :1]
        weight1 = weight[:, 1:]
        weight1_4 = torch.bitwise_left_shift(weight1, 4)
        weight2_4 = weight0 & 0b00001111
        weight_add = torch.bitwise_or(weight1_4, weight2_4)
        # The clone() call is used to break the view chain
        return weight_add.reshape(shape[:-1] + (shape[-1] // 2,)).clone()
```

```python
    def process_weights_after_loading_modelslim(self, layer):
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        layer.w13_weight_scale.data = self._process_scale(layer.w13_weight_scale.data)
        layer.w2_weight_scale.data = self._process_scale(layer.w2_weight_scale.data)

        layer.w13_scale_bias.data = layer.w13_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        layer.w2_scale_bias.data = layer.w2_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        layer.w13_weight_scale.data = self.maybe_squeeze_per_channel_weight_scale(layer.w13_weight_scale.data)
```

https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/utils.py

```python
ACL_FORMAT_FRACTAL_NZ = 29
```

```python
# - 310P: always convert supported weights to FRACTAL_NZ
# - non-310P: follow additional_config.weight_nz_mode
# - FP32: never convert
# - meta tensor: never convert
def maybe_trans_nz(
    weight: torch.Tensor,
    customize_dtype: torch.dtype | None = None,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _should_trans_nz(weight):
        return weight
    kwargs = {}
    if customize_dtype is not None:
        kwargs["customize_dtype"] = customize_dtype
    if input_dtype is not None:
        kwargs["input_dtype"] = input_dtype
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ, **kwargs)
```

## Relevance

This pins the byte layout our megakernel must consume without re-converting anything: `w13`
int32 `[E, 6144, 2*128/4]`, `w2` int32 `[E, 128, 6144/8]`, scales as **int64-packed uint32 fp32
bit patterns** (`scale_np.dtype = np.uint32` then `astype(np.int64)`) — so the scale tensor
cannot be reinterpreted as float, and `scale_bias` already summed over the K-group axis to
`[E, 2*moe_intermediate]`. `ACL_FORMAT_FRACTAL_NZ = 29` is the format id.

---

### `DispatchFFNCombineW4A8` — a merged A3 MoE megakernel already in vllm-ascend's csrc

https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_def.cpp

```cpp
class DispatchFFNCombineW4A8 : public OpDef {
 public:
  explicit DispatchFFNCombineW4A8(const char *name) : OpDef(name) {
    ...
    this->Input("w1")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
        .IgnoreContiguous();
    ...
    this->Output("expert_token_nums")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ...})

    this->Attr("group").AttrType(REQUIRED).String();
    this->Attr("M").AttrType(OPTIONAL).Int();
    this->Attr("transB").AttrType(OPTIONAL).Bool(false);
    this->Attr("weightNz").AttrType(OPTIONAL).Bool(false);
    this->Attr("swigluLimit").AttrType(OPTIONAL).Float(0.0f);
    this->Attr("worldSize").AttrType(REQUIRED).Int();

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
    this->AICore().AddConfig("ascend910_93", aicore_config);
    this->MC2().HcclGroup("group");
  }
};

OP_ADD(DispatchFFNCombineW4A8);
```

https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/tiling_args.h

```cpp
namespace Moe {
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 3U * 1024UL * 1024UL;
constexpr uint64_t NOTIFY_DISPATCH_WIN_OFFSET = 204U * 1024UL * 1024UL;
}  // namespace Moe
```

https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp

```cpp
    constexpr int64_t MAX_WORLD_SIZE = 768;
    constexpr uint64_t INIT_TILINGKEY = 1000000;
    constexpr uint64_t TILINGKEY_TRANS_B = 1U;
    constexpr uint64_t TILINGKEY_WEIGHT_NZ = 10;
    constexpr uint32_t BLOCK_NUM = 20;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
...
    uint32_t aivNumInitRouting = 2 * BLOCK_NUM;
```

The host call site, and the weight-layout constraint comment:
https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py

```python
                torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                    x=fused_experts_input.hidden_states,
                    weight1=weights.w1,
                    weight2=weights.w2,
                    expert_idx=fused_experts_input.topk_ids,
                    scale1=weights.w1_scale,
                    scale2=weights.w2_scale,
                    bias1=weights.w1_scale_bias,
                    bias2=weights.w2_scale_bias,
                    probs=fused_experts_input.topk_weights.to(torch.float32),
                    group=self.token_dispatcher.moe_all_to_all_group_name,
                    max_output_size=get_ascend_config().mega_moe_max_tokens,
                    swiglu_limit=self.swiglu_limit,
                    world_size=self.token_dispatcher.ep_world_size,
                    x_active_mask=fused_experts_input.routing.mc2_mask,
                    out=out,
                    expert_token_nums=self.expert_token_nums,
                )
```

```python
        # A8W4-INT MegaMoe reads N from weight1.storageShape.lastDim treated as int8 (N = lastDim*2)
        # and checks weight2.dim0 == N/2, so the weights MUST be int8-shaped (two int4 per byte), NOT
        # the eight-int4-per-int32 packing (that makes the op read N four times too small and fail
        # CheckWeight2Input). The op prototype also REQUIRES FRACTAL_NZ per expert. The W4A8 quant
        # method therefore builds per-expert int8 + FRACTAL_NZ lists (cann_mega_moe_*_weight_list) and
        # they are passed through as-is here. W8A8 weights are already int8 + FRACTAL_NZ, also as-is.
```

```python
        x_active_mask = None
        # Ascend 950 (A5) MegaMoe only support a null x_active_mask, and it
        # must be passed as None. But on A2/A3 it must be valid.
```

## Relevance

A working, merged, A3-targeted (`ascend910_93`) MoE megakernel registered through MC2: one
kernel swallowing dispatch + gmm1 + SwiGLU + quant + gmm2 + combine, with `BLOCK_NUM = 20` AIC
blocks and `aivNumInitRouting = 2 * BLOCK_NUM` AIV, a 16 MiB system workspace, and
`NOTIFY_DISPATCH_WIN_OFFSET = 204 MiB`. It takes `scale1/scale2` and `bias1/bias2` exactly in
the MSD layout above. Two traps it records: W4A8 MegaMoe needs int8-shaped (2 int4/byte)
FRACTAL_NZ per-expert tensors, **not** the int32 packing the non-mega path uses; and
`x_active_mask` must be valid on A2/A3 but null on A5.

---

### The adversarial datapoint: that megakernel is broken on A3 TP16 W4A8 today

https://github.com/vllm-project/vllm-ascend/issues/16914
`[Bug] MegaMoe on Kimi K3 (A3, PD-disaggregated 4P4D) produces garbled output and vector-core crash (acl error 507035)` — linnea-lin-00638949, 2026-09-18, open.

> When enabling MegaMoe (`enable_fused_mc2: 2` + `enable_prefill_mc2: true`) for Kimi K3 W4A8 on Atlas 800I A3, **every request returns garbled tokens**, and the prefill nodes eventually **crash with an AI-vector-core exception (`aclrtSynchronizeEvent` error 507035)** followed by glibc heap corruption (`corrupted size vs. prev_size`) on shutdown.
>
> Disabling MegaMoe on the exact same build/environment produces correct output — so the problem is isolated to the MegaMoe path.
>
> - Hardware: Atlas 800I A3 (910_93), 8 nodes x 16 NPU
> - Deployment: **PD-disaggregated 4P4D**, both sides DP4/TP16 + DCP16, MooncakeConnectorV1 KV transfer, dspark speculative decoding (draft=7)
> - CANN: 9.3.0 weekly `20260916.01` (the A3-ops package bundles `cann-A3-ops-transformer_9.2.0`)
>
> Note: although `multistream_overlap_shared_expert: true` is passed, vllm-ascend auto-disables it (warning `enable_fused_mc2 and multistream_overlap_shared_expert cannot be enabled at the same time. Setting multistream_overlap_shared_expert to False.` appeared 18x = once per worker), so the effective runtime value is `false` on all MegaMoe ranks.

## Relevance

The counterweight, on our exact configuration: A3 910_93, TP16, W4A8, CANN 9.2.0-class ops. The
shipped fused MoE megakernel produced garbled output plus 507035 vector-core faults, and it is
mutually exclusive with shared-expert multistream overlap. Any plan needs an A/B toggle and
per-rank numeric validation, not just a profile.

---

### A3 MSD GMSQ with direct ND/NZ weights — scratch sizes and a measured NZ-vs-ND ratio

https://github.com/vllm-project/vllm-ascend/pull/17201
`[Feature][MoE] Add A3 MSD GMSQ with direct ND/NZ weights` — grandolantow, 2026-09-22, open.

> Add an A3 fused grouped matmul + SiTU + per-token INT8 quantization operator using exact A8 decomposition and INT4 Matmul. Packed ND and native packed INT8 NZ weights are consumed in their existing storage, without a full W4-to-W8 weight replica.
>
> - Read ND directly as ND and native NZ directly as NZ. Validate the packed carrier's dtype, physical geometry and strides; retain expert views without cat/copy.
> - Reconstruct the integer dot product before the existing FP16 channel-scaling boundary, then apply SiTU and dynamic INT8 quantization.
> - Parse routing counts on device on each invocation/replay. Normalize int64/int32/float32 group lists without host value reads.
> - Keep graph-visible activation/accumulator scratch stable per device/stream/K/N. Its size is independent of expert count and layer count for a shared stream/K/N.
> - Accept runtime E/K/N subject to implementation limits, rather than profiling-shape whitelists: E=1..128, K a positive multiple of 64 up to 16384, N a positive multiple of 256 subject to the UB budget, and at most 256 padded M blocks. Support K packing chunks and N tails.
>
> The static pointer metadata requires eager warmup for each weight/scale configuration. Weights/scales must remain static afterward, and graphs sharing scratch must replay serially.
>
> For K=3584/N=6144, the primary scratch is 54.71875 MiB (packed activation + INT32 accumulation). The weight-replica allocation is zero, compared with E*K*N bytes for a fully expanded W8 representation (2.296875 GiB at E=112). These figures exclude outputs, pointer metadata and framework/system allocations.

> Paired operator measurements below use E=14/K=3584/N=6144, capacity=28672 and the eight fitted routing cases. Each ND/NZ pair used the same card and alternating independent weight sets, 50 graph replays per batch and five batches. Values are median NPU Graph Event time per operator call; they are not an end-to-end throughput comparison or a comparison against the prior W8-cache implementation.
>
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
>
> Geometric mean NZ/ND: **0.7263**.

## Relevance

A3-specific, MSD-specific, and quantified: NZ-native weights beat ND by 27% on the same kernel.
The shape envelope (`E=1..128`, K a multiple of 64 up to 16384, N a multiple of 256, at most 256
padded M blocks) admits our K=6144 / N=256 (=2*128). The scratch figure (54.7 MiB for
K=3584/N=6144, independent of E and layer count) sizes the workspace our megakernel will need.
The caveat `graphs sharing scratch must replay serially` is a constraint on multi-stream plans.

---

### An Ascend MoE expert megakernel that already shipped a 1.46x win from launch removal alone

https://github.com/vllm-project/vllm-ascend/pull/10488
`[Feature] Add fused W4A4 INT4 MoE "mega" kernel for Qwen3.x-MoE (Ascend 910B)` — Mocchibird, 2026-06-15, open.

> MoE decode serving on 910B is HBM-bandwidth bound. The standard vllm-ascend W4A4 path ("vendor" throughout this PR) already shrinks the routed-expert weights 4x, but it assembles the expert path from a chain of **separate `torch_npu` `npu_*` op launches** — `npu_moe_init_routing_v2` -> `npu_dynamic_quant` -> `npu_grouped_matmul` (`gate_up`) -> `npu_swiglu` -> `npu_dynamic_quant` -> `npu_grouped_matmul` (`down`) -> `npu_moe_token_unpermute` — each paying dispatch + cross-core sync overhead and round-tripping the activations through HBM between ops.
>
> This kernel folds that whole chain into **one MIX launch**, with the cube matmuls overlapped against the vector stages, so the intermediate activations never leave on-chip memory. The payoff is twofold:
>
> - **vs. standard vllm-ascend W4A4 (same checkpoint, same accuracy):** the fused kernel is the headline win — **1.46x higher decode throughput** purely from removing the per-op dispatch/sync/HBM-round-trip overhead. The quantization is byte-identical (cos 0.997 between the two kernels), so this is a free serving-throughput gain over the vendor path.
> - **vs. bf16:** faster on decode (**1.20x**) *and* the 4x-smaller expert weights free enough HBM to grow KV-cache capacity / max concurrency by **+30%**.
>
> | stage | engine | work |
> | --- | --- | --- |
> | 1 | cube | 64x64 block-diagonal Hadamard on activations (fp16 GEMM) |
> | 2 | vector | fp16 -> INT4 activations + fp32 scale, fused routing scatter |
> | 3 | cube | INT4xINT4 -> INT32 `gate_up` |
> | 4 | vector | dequant + SwiGLU + INT4 requant |
> | 5 | cube | INT4xINT4 -> INT32 `down` |
> | 6 | vector | dequant + scatter-add unpermute + top-k-weight combine |
>
> - Cube matmuls: AscendC `MatmulImpl<int4b_t>`. Vector stages: PTO-ISA. Cross-core scheduling via FFTS so cube/vector overlap across expert-chunks.
>
> | config | GPQA decode (3-seed) | vs bf16 | vs vendor W4A4 |
> | --- | ---: | :--: | :--: |
> | bf16 | 276 +/- 25 tok/s | — | — |
> | w4a4 vendor — `torch_npu` `npu_*` chain (separated) | 227 +/- 22 tok/s | 0.82x | 1.00x |
> | **w4a4 hybrid-mega (ours, fused single launch)** | **332 +/- 18 tok/s** | **1.20x** | **1.46x** |

## Relevance

The single most directly transferable result: an Ascend MoE expert megakernel, byte-identical
numerics (cos 0.997), 1.46x over the stock `npu_*` chain, attributed entirely to
dispatch/sync/HBM-round-trip removal. Its 6-stage alternating cube/vector schedule with FFTS
cross-core scheduling is a structural template, and the use of `MatmulImpl<int4b_t>` for cube
plus PTO-ISA for vector names the tools.

---

### Upstream's measurement that small kernels at batch=1 under graph capture cost ~18.6 us each

https://github.com/vllm-project/vllm-ascend/pull/15714
`[Performance][MoE] Reuse a persistent pad buffer in the MC2 prepare path` — MirkoDeVita98, 2026-09-03, open.

> This costs two extra kernels per MoE layer per forward whenever `pad_size > 0`, i.e. whenever the token count is not an exact multiple of `tp_size` — **7 of every 8 batch sizes at `tp_size=8`**.
>
> Profiled with `torch_npu.profiler` on Qwen3-30B-A3B, 8x Atlas 800I A2, EP=8, batch 1, ACL graph captured, rank 0:
>
> | kernel | before | after |
> | --- | --: | --: |
> | `aclnnConstantPadNd_PadV3AiCore_PadV3` | 945.2 us/step | 0.0 (0 calls) |
> | `aclnnConstantPadNd_PadV3AiCore_MemSet` | 808.4 us/step | 0.0 (0 calls) |
> | all padding kernels | 36.7 us/layer | 0.20 us/layer |
>
> At 18.6 us per call for 32 KB of data, this is launch overhead rather than bandwidth, so the absolute cost is largely independent of how many rows are padded.

> **Decode latency, batch 1** — 32 prompts served one at a time, 128 generated tokens, 3 rounds with the arms rotated within each round to cancel thermal drift:
>
> | round | ITL unpatched | ITL patched | delta |
> | --: | --: | --: | --: |
> | 1 | 18.3972 ms | 17.6319 ms | -4.16% |
> | 2 | 18.4899 ms | 17.5666 ms | -4.99% |

And the companion, with a TP16 calibration:

https://github.com/vllm-project/vllm-ascend/pull/16343
`[Performance][Communicator] Replace per-layer F.pad with cat of a persistent zero block in MoE prepare` — Levi-JQ, 2026-09-11, closed.

> On NPU each `F.pad` lowers to two device kernels (`constant_pad_nd` -> MemSet + PadV3) while the padded data is only a few KB in decode, so the four launches are pure overhead: ~44 us per MoE layer (~5.6% of a decode layer's kernel time in a Kimi K3 TP16 profile). With speculative decoding (`num_speculative_tokens=7`, minimum graph tier of 8 tokens), odd batch sizes always pad 8 rows to 16 and hit this path.

## Relevance

Upstream independently measuring, at batch=1 with ACL graph captured, that a trivially small
kernel costs 18.6 us/call and that this is launch-bound not bandwidth-bound — the same inference
we drew from 6.793 vs 2.272 ms. The TP16 anchor is the useful one: 44 us = 5.6% of a decode
layer's kernel time implies ~786 us per decode layer at TP16, and deleting 2 launches/layer
bought 4-5% ITL.

---

### W4A8 GroupedMatmul per-call cost on A3 TP16, and how little tiling moves it

https://github.com/vllm-project/vllm-ascend/pull/16606
`[Performance] Add tuning_config to w4a8 MoE grouped matmul` — linnea-lin-00638949, 2026-09-15, open.

> Passes `tuning_config=[0, 0, -1]` to `torch_npu.npu_grouped_matmul` in `BaseDeviceAdaptor.npu_grouped_matmul_gmm2` (the w4a8 MoE grouped-matmul path).
>
> Setup: kimi-k3 w4a8 on 4x Atlas 800I A3, vllm-ascend v0.26.0 (b5e8e2561, the call site is identical on main), DP4xTP16 hybrid, DSpark k=7 speculative decoding. Baseline = the same service without the parameter.
>
> **Operator level** (torch_npu profiler, rank 13, identical request, decode path; 11776 GroupedMatmul calls in both runs, deterministic and directly comparable):
>
> | metric | baseline | tuning_config | delta |
> | --- | --- | --- | --- |
> | total (ms) | 776.2 | 743.4 | -4.2% |
> | mean (us) | 65.9 | 63.1 | -4.2% |
> | median (us) | 61.6 | 58.4 | -5.2% |
> | p90 (us) | 94.8 | 92.4 | -2.5% |
> | max (us) | 165.6 | 151.2 | -8.7% |
>
> **E2E** (benchmark_serving, 20 requests, 20k-40k input tokens, max-concurrency=1):
>
> | metric | baseline | tuning_config | delta |
> | --- | --- | --- | --- |
> | duration (s) | 1242.13 | 1161.33 | -6.5% |
> | output throughput (tok/s) | 27.94 | 30.08 | +7.7% |
> | TPOT mean (ms) | 32.70 | 28.51 | -12.8% |
> | TPOT P99 (ms) | 44.82 | 40.79 | -9.0% |

## Relevance

The per-call decomposition of our 5.596 ms GroupedMatmul aggregate: mean 65.9 us, median 61.6 us
per call on A3 TP16 W4A8 decode. Tiling choice moves it only ~4%, which is the evidence that
the gap to 1.212 ms of bytes is structural rather than a tiling miss.

---

### The current ceiling of launch-overhead workarounds at concurrency 1

https://github.com/vllm-project/vllm-ascend/pull/19033
`V0.23.0: Optimization of TPOT under low concurrency conditions` — zhu1365377615, 2026-10-16, open.

> All changes were developed and validated on Atlas 910C (Ascend 910_93, CANN 9.1.0) with Qwen3.5-35B-A3B (TP=2), covering both plain decode and MTP speculative decoding.
>
> 1. **[Perf][Compilation] Add `enable_super_kernel` option for small-operator fusion**
>    New `ascend_compilation_config` key `enable_super_kernel`; when set, passes `super_kernel_optimize` to the npugraph_ex compiler backend so small operators inside static kernels are fused automatically.
>
> 2. **[Perf][Kernel] Lower MoE weight prefetch token threshold to 1**
>    `MOE_PREFETCH_TOKEN_THRESHOLD` 96 -> 1 so MoE weight prefetch also stays active for small decode batches (previously skipped below 96 tokens), improving TPOT at concurrency 1/4/8 when `weight_prefetch_config` is enabled.
>
> 4. **[Perf][Kernel] Overlap shared experts with routed experts and reduce decode overhead**
>    - Runs shared experts on a side stream concurrently with routed experts when `multistream_overlap_shared_expert` is enabled; the shared-expert AllReduce stays on the side stream so it overlaps with routed expert computation.
>    - Allows `VLLM_ASCEND_ENABLE_FUSED_MC2` and `multistream_overlap_shared_expert` to coexist (removes the previous mutual-exclusion check).
>    - Skips dispatch/combine/gmm2 event recording in the overlap path via a new `skip_dispatch_combine_events` forward-context attribute.
>    - Prefetches the next-layer shared-expert gate_up weight together with the routed gate_up weight so it lands in cache during the current layer's GroupedMatmul.
>
> 8. **[BugFix][Kernel] Make multistream shared-expert overlap capture-safe for MTP**
>    The side-stream overlap path was not capture-safe: during NPU graph capture the shared-experts side stream is not registered with the graph pool, so any side-stream op — or even a `wait_stream()` recording an event on it — fails with `aclrtAllocatorGetByStream` / `rtStreamWaitEvent` errors, crashing startup at graph capture when MTP is combined with the overlap.
>
> | Scenario | Before | After |
> | --- | --- | --- |
> | c=1 TPOT (no MTP) | ~8.2 ms | ~7.6 ms |
> | c=1 TPOT (MTP=4, min_tokens=512) | ~7.0 ms | ~5.1 ms (-27%) |
> | c=4 TPOT (MTP=4, min_tokens=512) | ~8.9 ms | ~9.0 ms (unchanged) |
> | c=8 TPOT (MTP=4, min_tokens=512) | ~12.7 ms | ~12.4 ms |

> reuses a pre-allocated NPU event for the mamba align-mode stream sync (removes per-step event create/destroy, ~10us host each step)

## Relevance

The ceiling of the non-megakernel approach: stacking SuperKernel fusion, weight prefetch at
1 token, side-stream shared-expert overlap and event-record elision moves c=1 TPOT only
8.2 -> 7.6 ms without MTP. Two items are directly reusable regardless: lowering the MoE weight
prefetch threshold to 1 token (it is skipped below 96 by default — at batch=1 we get no
prefetch at all today), and prefetching the next layer's shared-expert gate_up weight during
the current layer's GroupedMatmul.

---

### The upstream RFC for a GLM-5.2 megakernel execution backend

https://github.com/vllm-project/vllm-ascend/issues/15639
`[RFC]: GLM-5.2 MegaKernel execution backend for vLLM-Ascend` — keyi-zz, 2026-09-03, open.

> Decode-phase execution in vLLM-Ascend today dispatches one op per layer per step and synchronizes at tensor granularity. This leaves launch/host overhead per step and cannot overlap data movement with compute across ops. GLM-5.2 (`GlmMoeDsaForCausalLM` / `glm_moe_dsa`) decode is a long chain of small-to-medium ops (MLA attention, MoE routing + grouped matmuls, DSA indexer sparse attention, norms, LM head) and is exactly the workload that benefits from whole-graph, tile-granularity execution.
>
> Huawei's MegaKernel / `blockrt` runtime aggregates the ops of a network into a single "mega" kernel, schedules work at MK-Tile granularity across AICores with an AI CPU coordinator + scoreboard + callbacks, and enables per-core copyIn/compute/copyOut software pipelining. This RFC proposes bringing the same execution path to vLLM-Ascend as an opt-in, decode-scoped execution backend for GLM-5.2, without touching default behavior.
>
> 2. **Decode-only seam**: pure decode batches run on a pre-built MegaKernel DAG(`blockrt` `Glm52MegaKernel`); prefill and chunked-prefill batches keep the native vLLM-Ascend path. Logits (incl. LM head) are produced inside the DAG.
> 4. **One physical KV cache**: force the packed SFA C8 layout for `glm_moe_dsa` under the flag so native prefill writes and mega decode read the same MLA + DSA indexer caches. Persistent per-batch-size buffers + per-step RoPE caches keep all tensor addresses stable so the captured DAG is replayed instead of re-captured.
> 6. **Incidental, gated fixes**: ascend-config sparse-SFA-C8 forcing, A5 meta-registration skip + `MK_ENABLE_CUSTOM_OP` escape hatch, and small W4A8 MoE grouped-matmul scale/offset fixes (candidates to split into separate PRs).
>
> Milestone scope: GLM-5.2 decode on Ascend 950-family (A5), TP=1; prefill and TP>1 remain follow-ups.
>
> - Known caveats recorded in the doc: shared-cache coupling to the SFA C8 layout; indexer scale-cache placeholders must be 2-D; `indexer_rope` registration must be verified before MTP runs; accuracy/perf numbers to be attached before code PRs.

## Relevance

The upstream RFC for our exact target, naming the runtime (`blockrt`, MK-Tile granularity,
AI CPU coordinator + scoreboard + callbacks, per-core copyIn/compute/copyOut pipelining), the
correctness seam (decode-only DAG, prefill stays native), and the address-stability requirement
for DAG replay. Critically, its milestone scope is **A5 and TP=1** — our A3 / TP16 case is
explicitly listed as a follow-up, i.e. new ground.

---

### The semantic hazard in collapsing gmm1 + SwiGLU + quant: the fused kernel's SwiGLU has no clamp

https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/ascend.py

```python
def _uses_fused_gmm1(kernel, config: MoeRunnerConfig) -> bool:
    ...
    # The fused gmm1 kernel (npu_grouped_matmul_swiglu_quant_v2) applies plain
    # SiLU with no clamp. Models whose activation is SiLU-with-clamp
    # (swiglu_limit, e.g. DSV4) must keep the unfused path so the clamp is
    # applied (NPUSwigluMxfp8Quant / NPUSwigluStepAndMul below).
    ...
        # MXFP8 has no unfused gmm1 path at all, so a swiglu_limit
        # checkpoint cannot be served by it.
    ...
        return False  # W4A8MXFP4: fall back to unfused to preserve the clamp
```

```python
            # Fused methods (MXFP8; MXFP4 W4A8 via use_fused_gmm1) fold
            # gate/up + swiglu + requant into gmm1, so there is no separate
            # activation step — run() skips it. Left None on purpose so that
            # reaching for it fails loudly instead of silently applying an
            # unfused swiglu to already-requantised activations. This holds
            # for both dispatchers: ascend_tp gets its activation quant fused
            # into routing, DeepEP dispatches bf16 and gmm1 quantises it
            # itself.
```

## Relevance

A correctness constraint for our fused epilogue: if GLM-5.2's activation carries a
`swiglu_limit`, the clamp must be implemented in our kernel. Note vllm-ascend's A3 path does
plumb `swiglu_limit` into `grouped_matmul_swiglu_quant_weight_nz`, and the V3 WeightNz doc
exposes `clampLimit`/`gluAlpha`/`gluBias` — but the stock A5 `_v2` fp8 branch drops it.

---

### A per-pipe reference for what a well-tuned GMSQ looks like on real silicon

https://github.com/triton-lang/triton-ascend/issues/1533
`【性能问题】手写的grouped_matmul_swiglu_quant triton算子和ascendc对应算子性能差距大` — boywithstripe, 2026-08-13.

```
对比项目                         AscendC              Triton
shape                            x[2048,10240]        x[2048,10240]
                                 W[8,10240,8192]      W[8,10240,8192]
dtype                            int8 (A8W8)          int8 (A8W8)
性能对比（x倍ascendc）           1.0                  6.57
总耗时 (ms)                      0.8348               5.4828
带宽 (GB/s)                      1.01                 10.11
Vector时间 (us)                  61.47                2033.41
Vector占比 (%)                   7.37                 37.09
Scalar时间 (us)                  31.38                2849.66
Scalar占比 (%)                   3.76                 51.98
MTE2时间 (us)                    53.75                1765.74
MTE2占比 (%)                     6.44                 32.21
MTE3时间 (us)                    37.33                10.22
Cube时间 (us)                    495.4                648.53
Cube占比 (%)                     59.40                11.83
fixpipe时间 (us)                 21.98                1086.53
MTE1时间 (us)                    430.35               1550.43
MTE2时间 (us)                    715.88               3415.47
GM→UB带宽 (GB/s)                 1.83                 20.07
UB→GM带宽 (GB/s)                 0.20                 0.15
L2读命中率 (%)                   98.82                99.91
L2写命中率 (%)                   9.84                 14.47
Block Dim                        24                   24
```

> 在大shape情况下， vec耗时明显，整体性能都消耗在vec上面，cv算子整体流水没有组织好

> 硬件：950PR
> CANN：9.0.0
> Triton：3.2.0
> Triton-Ascend：3.2.1

## Relevance

A per-pipe target: the vendor AscendC GMSQ is 59.4% Cube with Vector at only 7.37% and Scalar
at 3.76%, at Block Dim 24. If our megakernel's GMSQ stage does not land near that Cube
fraction, the loss is in Scalar/MTE1 orchestration, not the matmul. The Triton column shows the
failure signature of a badly pipelined CV kernel: Scalar 51.98%, fixpipe 1086 us, 6.57x slower.

---

### The routing ops a megakernel must absorb, and the AICPU-argsort workaround

https://github.com/vllm-project/vllm-ascend/blob/v0.9.1/vllm_ascend/quantization/w8a8_dynamic.py

```python
        if hasattr(torch_npu, "npu_moe_init_routing_quant"):
            quantized_tokens, expanded_row_idx, global_expert_tokens, _, token_scales = torch_npu.npu_moe_init_routing_quant(
                hidden_states,
                expert_idx=topk_ids.to(torch.int32),
                active_num=0,
                expert_capacity=0,
                expert_num=global_num_experts,
                drop_pad_mode=0,
                expert_tokens_num_mode=2,
                expert_tokens_before_capacity_flag=False,
                quant_mode=1,
            )
```

```python
        hidden_states, dynamic_scale, inverse_indices, expert_tokens = torch_npu.npu_moe_re_routing(
            gathered_tokens,
            gather_sizes.view(ep_group.world_size, -1),
            per_token_scales=dynamic_scale)
```

```python
        hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
```

```python
    if expert_map is not None:
        reordered_outputs = torch.index_select(
            hidden_states,
            dim=0,
            # Workaround: Convert to float so that argsort runs on AI Core instead of slower AICPU
            index=inverse_indices.to(torch.float32).argsort().to(torch.int32))
```

```python
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
```

And the W4A8 carve-out, same file:

```python
    # `expand_x` will be disposed in the `apply_mlp` function
    if w1_scale_bias is None:
        down_out_list = apply_mlp_decode([expand_x], ...)
    else:
        # w4a8 scene, cannot use apply_mlp_decode because the operator is not supported
        down_out_list = apply_mlp(expand_x, ..., w1_scale_bias=w1_scale_bias, w2_scale_bias=w2_scale_bias)
```

And the group-list cumsum helper that runs per call:

https://github.com/vllm-project/vllm-ascend/blob/v0.11.0rc0/vllm_ascend/ops/moe/moe_mlp.py

```python
def cumsum_group_list(group_list: torch.Tensor,
                      group_list_type: int,
                      active_num: int = 0,
                      expert_num: int = 0) -> torch.Tensor:
    if group_list_type not in [0, 1, 2]:
        raise ValueError(
            f"group_list_type should be in [0, 1, 2], but received {group_list_type}"
        )

    if group_list_type == 0:
        return group_list
    if group_list_type == 1:
        return group_list.cumsum(dim=0)

    experts = pad(group_list[:, 0], (1, 0))
    tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
    cumsum_group_list = torch.full(size=(expert_num, ),
                                   fill_value=active_num,
                                   dtype=group_list.dtype,
                                   device=group_list.device)

    for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
        if end > start:
            cumsum_group_list[start:end] = tokens[i]

    return cumsum_group_list
```

and in the W4A8 path, the cat/diff conversion on the hot path:

```python
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat(
                    [group_list[:1],
                     torch.diff(group_list, dim=0)])
                group_list_type = 1
```

## Relevance

The complete op inventory a megakernel must swallow: `npu_moe_init_routing{,_quant,_v2}`,
`npu_moe_re_routing`, `npu_moe_compute_expert_tokens`, `npu_moe_finalize_routing`, plus
`npu_dynamic_quant`, plus the `cat`/`diff`/`cumsum` group-list conversions, plus an
`argsort` that needs a float cast to avoid landing on AICPU. At batch=1 every one of these is a
fixed-cost launch on data of a few hundred bytes. Upstream also records that the decode-tuned
int32-accumulate path (`apply_mlp_decode`) is unavailable once `scale_bias` is present, i.e.
W4A8 never got the decode specialization.

---

### Huawei's own serving stack (omniinfer): the W4 path still runs unfused, with per-call dummy tensors

https://gitee.com/omniai/omniinfer — `omni/layers/moe/fused_moe/fused_moe.py`

```python
    if layer.weight_num_bits == 8:
        if model_extra_config.operator_opt_config.enable_gmm_swiglu_quant:
            gate_up_proj, pertoken_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                sorted_tokens, [layer.w13_weight], [layer.w13_weight_scale], pertoken_scale, expert_tokens)
        else:
            ...
    elif layer.weight_num_bits == 4:
        tuning_config_13 = None
        tuning_config_2 = None
        if not model_extra_config.operator_opt_config.gmm_nz:
            tuning_config_13 = model_extra_config.task_config.decode_gear_list[:1]
            tuning_config_2 = model_extra_config.task_config.decode_gear_list[:1]
        elif model_extra_config.operator_opt_config.new_w4_op:
            tuning_config_13 = [1, 1]
            if model_extra_config.task_config.decode_gear_list[0] >= 32:
                tuning_config_2 = [256]
        elif model_extra_config.task_config.decode_gear_list[0] >= 32:
            tuning_config_13 = [256]
            tuning_config_2 = [256]
        gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=[layer.w13_weight_bias], scale=[layer.w13_weight_int4_scale],
                                                    ...
                                                    tuning_config=tuning_config_13, output_dtype=torch.bfloat16)[0]

        fake_scale = torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32, device="npu").view(-1, layer.w13_weight_int4_scale.shape[-1])
        pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
        gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj, weight_scale=fake_scale,
                                                                          activation_scale=pertoken_scale,
                                                                          bias=None, quant_scale=None,
                                                                          quant_offset=None, group_index=expert_tokens,
                                                                          activate_left=True, quant_mode=1)
```

Weight prefetch of the *next* layer's attention weight during the MoE stage:

```python
        if model_extra_config.operator_opt_config.attn_prefetch > 0 and next_attention_weights is not None and next_attention_weights['q_a_proj_weight'] is not None:
            attn_prefetch_size = model_extra_config.operator_opt_config.attn_prefetch * 1024 * 1024
            torch_npu.npu_prefetch(next_attention_weights['q_a_proj_weight'], hidden_states_experts, attn_prefetch_size)
```

## Relevance

Huawei's production stack gates the GMSQ fusion behind `enable_gmm_swiglu_quant` and only for
W8; the W4 path still runs gmm1 -> `npu_dequant_swiglu_quant` -> gmm2 and allocates a
`fake_scale` plus a unit `pertoken_scale` **per call** (extra allocations and launches on the
hot path). `tuning_config` is selected from `decode_gear_list` and toggles between `[1, 1]` and
`[256]` — the same knob PR #16606 exploited. `torch_npu.npu_prefetch` with a byte budget is the
documented API for cross-stage weight prefetch, which is what our megakernel must do internally.

---

### MindSpeed: `group_list_type` 0 vs 1 are different compiled ops, and `act_type` already exists

https://gitcode.com/Ascend/MindSpeed — `mindspeed/ops/gmm.py`, `mindspeed/ops/quant_gmm.py`

```python
class GMMFunction(torch.autograd.Function):
    builder = GMMOpBuilder()
    builder2 = GMMV2OpBuilder()

    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
        ...
        if group_list_type == 0:
            outputs = GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
        elif group_list_type == 1:
            outputs = GMMFunction.builder2.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
```

```python
def npu_quant_gmm(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                  output_dtype=None, act_type=0):
    return _npu_quant_gmm_common(x, weight, scale, offset=offset, per_token_scale=per_token_scale,
                                 bias=bias, group_list=group_list, group_list_type=0, output_dtype=output_dtype,
                                 act_type=act_type)

def npu_quant_gmm_v2(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                     output_dtype=None, act_type=0):
    return _npu_quant_gmm_common(x, weight, scale, offset=offset, per_token_scale=per_token_scale,
                                 bias=bias, group_list=group_list, group_list_type=1, output_dtype=output_dtype,
                                 act_type=act_type)
```

## Relevance

`group_list_type` 0 (cumsum) and 1 (count) are separate compiled ops (`GMMOpBuilder` vs
`GMMV2OpBuilder`), so the cumsum/count choice is a build-time specialization, not a runtime
branch — which means a batch=1 megakernel can pick one and drop the other's code path entirely.
`act_type` confirms in-kernel activation selection is an existing axis.

---

### `npu_quant_grouped_matmul_dequant` — the variant that absorbs the input per-token quant

https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/QuantGroupedMatmulDequantNpuOpApi.cpp

```cpp
static bool is_nz_format(const at::Tensor& tensor) {
  const torch_npu::NPUStorageDesc& tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
  return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ || tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_4 ||
      tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_16;
}
at::Tensor npu_quant_grouped_matmul_dequant(
    const at::Tensor& x,
    const at::Tensor& quantized_weight,
    const at::Tensor& weight_scale,
    const at::Tensor& group_list,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& x_scale,
    const c10::optional<at::Tensor>& x_offset,
    const c10::optional<at::Tensor>& smooth_scale,
    c10::optional<c10::string_view> quant_mode) {
  if (is_nz_format(quantized_weight)) {
    static const bool is_weight_nz_available = check_aclnn_kernel_available("aclnnQuantGroupedMatmulDequantWeightNZ");
    ...
```

Call site showing the fused input quant:

```python
    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        """Fused gmm1 (quant + dequant) + swiglu via 310P kernels.

        ``npu_quant_grouped_matmul_dequant`` quantizes the hidden states to
        int8 internally (``quant_mode="pertoken"``) and dequantizes the gmm1
        output, so the activation scale is not needed by ``apply_gmm2``.
        """
```

### And the WeightNZ shape-inference trap

https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/GroupedMatmulSwigluQuantNpuOpapi.cpp

```cpp
    auto x_size = x.sizes();
    // CANN WeightNZ infers logical N from weightScale, not from packed/NZ weight storage shape.
    int n = weight_scale.sizes().back();
    int m = x_size[0];
    int k = x_size[1];

    at::Tensor output = npu_preparation::apply_tensor_without_format({m, n/2}, c10::dtype(c10::ScalarType::Char));
    at::Tensor output_scale = npu_preparation::apply_tensor_without_format(x_scale, {m});
    at::Tensor output_offset = npu_preparation::apply_tensor_without_format({}, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluQuantWeightNZ,
        x, weight, bias, offset, weight_scale, x_scale, group_list,
        output, output_scale, output_offset);
```

## Relevance

`quant_mode="pertoken"` absorbing the input quant inside the GEMM is the pattern for deleting
our separate `npu_dynamic_quant` launch. The WeightNZ comment is a trap worth copying: logical
N comes from `weight_scale.sizes().back()`, not from the packed weight storage shape.

---

## 7. The MonoMoE kernel as shipped in FlashInfer — a complete design spec for a batch<=8, E=256 MoE megakernel

### `docs/design_docs/monomoe_kernel.md`

https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/monomoe_kernel.md
Raw: https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/docs/design_docs/monomoe_kernel.md
Source tree: `csrc/fused_moe/monomoe/` (`monomoe_binding.cu`, `monomoe_wrapper.cuh`,
`src/moe.cuh`, `src/moe_up_projection.cuh`, `src/moe_down_projection.cuh`,
`src/moe_routing.cuh`, `src/moe_scale_inputs.cuh`, `src/moe_tma.cu`),
Python: `flashinfer/fused_moe/monomoe.py`, bench: `benchmarks/bench_monomoe.py`.

Scope and fixed shape — note `E=256` and `M <= 8`:

> # MoE Monokernel — Design
>
> A single persistent CUDA kernel that executes an entire FP8 block-wise MoE
> layer for small decode batches (BS ≤ 8) on Hopper (sm_90a): routing → up
> projection (+SiLU+re-quant) → down projection → weighted combine.  One launch,
> no intermediate kernel boundaries, capturable into a CUDA Graph.
>
> Fixed shape (`Dims_BS8_E256_N512_K2048_BlockFP8_WGMMA_TMA`): E=256 experts,
> N=512 (`moe_intermediate_size` per half), K=2048 hidden, M ≤ 8 tokens.
>
> ```text
> GM in:  activations [BS,K] bf16      router_logits [BS,E] bf16
>         w_up   [E,2N,K] fp8 + block scales [E,2N/128,K/128] fp32
>         w_down [E,K,N]  fp8 + block scales [E,K/128,N/128]  fp32
> GM out: activations_out [BS,K] bf16
> ```

The execution model — one block per SM, roles fixed for the whole kernel, phase changes
reassign *what* each block works on rather than relaunching:

> ## Execution model
>
> - `GRID_SIZE` blocks (default 128), `BLOCK_SIZE = 384` threads = 12 warps.
>   The whole grid is launched ONCE and persists through all five phases —
>   no blocks are created or retired between stages; a block changes *role*
>   (which expert / which output tile it works on) at each phase boundary.
> - `__launch_bounds__(BLOCK_SIZE, 1)` + `GRID_SIZE <= SM count` pin one block
>   per SM.  This co-residency invariant is what makes the cross-block
>   flag/sentinel handoffs (below) deadlock-free: every block is scheduled from
>   launch, so a block spinning on another block's flag can never wait on a
>   block that has not yet been scheduled.
> - Warp roles (fixed for the whole kernel; the same physical warps take
>   phase-specific duties, detailed per stage below):
>     - warps 0–7 (threads 0–255): **calc warps**, forming two WGMMA
>     warpgroups — WG0 = warps 0–3, WG1 = warps 4–7.  A WGMMA is issued by
>     all 128 threads of a warpgroup together.
>     - warp 8 lane 0 (thread 256): the single **TMA launcher thread** —
>     issues every `cp.async.bulk.tensor.2d` and arms every mbarrier in the
>     block, in all phases.
>     - warps 8–11 (threads 256–383): **prefetch warps** (PF0–PF3) — scale
>     loads, the deferred up-proj epilogue, the deferred down-proj
>     accumulate, and (warps 1–11) the Phase-2 quantization.

The derived geometry table — every tile count is a closed form of (E, N, K) and five constants:

> The stage geometry is fixed by the shape's `KernelConfig` (`GRID_SIZE = 128`,
> `DOWN_COL_TILE` = DCT = 256, `K_STEP_UP` = KUP = 256, `K_STEP_DOWN` = KDN = 256,
> `UP_W_SLOTS` = SLOTS = 4; `UP_COL_HALVES` = UCH is derived).  All stage geometry
> follows from these and the (E, N, K) dims:
>
> | derived quantity | formula | value | meaning |
> | ----------------- | --------------------------------------- | ----- | ----------------------------------- |
> | `UCH` | `max(1, 2N·DCT / (128·K))` | 1 | 128-row M-atoms per up-block |
> | `UP_GRID` | `2N / (128·UCH)` | 8 | blocks per expert, up-proj |
> | `UP_GROUPS` | `GRID_SIZE / UP_GRID` | 16 | experts in parallel, up-proj |
> | `K_TILES_UP` | `K / KUP` | 8 | up-proj outer K iterations |
> | `K_SUBSTEPS_UP` | `KUP / 128` | 2 | 128-K substeps per iteration |
> | `UP_ARM_DISTANCE` | `max(1, SLOTS − 2)` | 2 | weight-TMA prefetch distance |
> | `DOWN_GRID` | `K / DCT` | 8 | blocks per expert, down-proj |
> | `DOWN_GROUPS` | `GRID_SIZE / DOWN_GRID` | 16 | experts in parallel, down-proj |
> | `DOWN_COL_HALVES` | `DCT / 128` | 2 | 128-col WGMMA passes per K-step |
> | `K_TILES_DOWN` | `N / KDN` | 2 | down-proj outer K iterations |
> | `K_SUBSTEPS_DOWN` | `KDN / 128` | 2 | 128-K substeps per iteration |
> | `K_BLOCKS_TOTAL` | `K / 128` | 16 | routing-window / quantization atoms |
>
> `UP_GROUPS == DOWN_GROUPS` (16), so the site-#2 producer/consumer sets coincide.
> The routing-window tile is 32 KB (BS=8).  These constraints hold: `GRID_SIZE` is
> a multiple of `UP_GRID` and `DOWN_GRID` and ≤ SM count; `K_TILES_UP` is a
> multiple of SLOTS (cross-expert stitch slot alignment); `K_TILES_DOWN` is even
> (the inter-expert lookahead lands in the slot the next expert reads); SHM fits
> the 224 KB budget.

The five-phase pipeline, with the two barrier-free handoff sites:

> ## Phase pipeline
>
> ```text
> Phase 1  routing (calc warps: topK)              ∥  routing-window TMA:
>                                                     full [BS,K] bf16 tile
>                                                     → SHM bf16_in_full,
>                                                     completion on bar_rwin
> Phase 2  warp 0: prepare_moe_topk            ∥  warps 1..11: wait bar_rwin,
>          (expert tally / prefix sums /              quantize bf16 → fp8_act_full
>           slot assignment)                          + per-128-K act_scale
>          __syncthreads()  — publishes both
> Phase 3  up-projection (WGMMA + weight-TMA pipeline)
>          → spec->temp_fp8 (expert-sorted rows) + temp_act_scale
> Site #2  NO barrier — sentinel handoff: each act-scale cell doubles as the
>          readiness flag for its fp8 payload; the down-proj polls the cells
>          it consumes (per-expert granularity)
> Phase 4  down-projection (WGMMA + weight/activation TMA double-buffer)
>          → atomicAdd into spec->down_partial_out [BS,K] fp32
> Site #3  NO barrier — flag counters: each block bumps its col stripe's
>          arrival counter; only that stripe's Phase-5 writer polls it
> Phase 5  fp32 → bf16 cast + writeback (first DOWN_GRID blocks only)
> ```
>
> Zero-fill of `down_partial_out` happens at kernel entry with no cross-block
> sync: every block runs the full routing+quantize+up-proj pipeline (tens of µs)
> before its first Phase-4 atomicAdd, while the zero-fill retires within the
> launch's first ~µs.  Phase 5 `=`-writes every output element, so no output
> pre-zero pass or extra grid sync exists.

Routing is *replicated* across the grid rather than partitioned:

> ### Phase 1 — routing ∥ input prefetch (all GRID_SIZE blocks, replicated)
>
> Every block executes Phase 1 identically and independently — routing and
> the input tile are needed by every block later, and each block has its own
> SHM, so the work is *replicated* across the grid rather than partitioned
> (the work is tiny: 8 tokens × E logits).  No cross-block communication.
>
> - **Calc warps 0–7 — top-k routing (`topK`)**: one warp per token
>   (warp w handles token w; warps ≥ `batch_size` return immediately).
>   Within a warp, lane t owns experts `{t, t+32, t+64, ...}` in registers
>   (`E/32` per lane — 8 for E=256).  The warp runs `top_k` rounds of
>   warp-reduce-argmax and lane 0 writes the token's ids/weights to
>   `topk_ids_flat` / `topk_weights_flat`.

The up-projection phase, including the deferred epilogue hidden under the *next* expert:

> ### Phase 3 — up-projection (grid partitioned: UP_GROUPS × UP_GRID)
>
> - **Blocks per expert**: `UP_GRID = 2N/(128·UCH)` blocks jointly produce
>   one expert's full `2N` intermediate rows; block `up_block_idx` owns rows
>   `[up_block_idx · 128·UCH, +128·UCH)`.  Here: 8 blocks × 128 rows (64 gate
>   + 64 up features each, interleaved — UCH = 1).
> - **Experts in parallel**: `UP_GROUPS = GRID_SIZE/UP_GRID` = 16 expert groups
>   run concurrently, one active expert per group at a time.
> - **Expert loop**: group g iterates the *active* expert list (built by
>   Phase 2, ascending id, length `expert_count ≤ min(E, BS·top_k)`) as
>   `e = g, g + UP_GROUPS, g + 2·UP_GROUPS, ...`.  Each group therefore
>   visits ≤ `ceil(expert_count / UP_GROUPS)` experts — with BS=8, top_k=8
>   at most 64 are active, so ≤ 4 per group at UP_GROUPS=16.
> - **Per expert, warp duties**:
>     - warp 8 lane 0 (launcher): inside each K-iteration, arms + TMAs the
>     weight slot `UP_ARM_DISTANCE = max(1, SLOTS−2)` iterations ahead; on
>     the last `UP_ARM_DISTANCE` iterations it stitches the *next* expert's
>     first tiles instead, so the pipeline never drains across experts.
>     - warp 8 (lanes 0–31, as PF warp 0): cp.async-prefetches the *next*
>     expert's block-scale tile into the `up_scale` ping-pong during the
>     current K-loop.
>     - warps 8–11 (PF0–PF3): the **deferred epilogue of the previous
>     expert** — one warp per token, `ceil(BS/4)` waves spaced across the
>     first `K_TILES_UP - 1` iterations: read `post_silu_scratch`,
>     warp-reduce max over the block's 64 features,
>     fp8-quantize, store to `temp_fp8[sorted_slot_row]` +
>     `temp_act_scale`.  The last visited expert has no successor and
>     drains inline on calc warps after the loop (one token per warp).

The weight pre-interleave that makes SwiGLU a register-local combine:

> **Interleaved up-proj weights (UCH == 1).**  One 128-row A-tile packs 64 gate
> + 64 up rows in the gate/up *pair layout*.  The tensor must be pre-interleaved
> in Python (`interleave_for_tma_wgmma_up`) so a single 128×128 TMA fetches one
> full WGMMA A-tile.  Each lane then holds gate and up for the same output
> feature, so silu(gate)·up is a register-local combine.

The down projection, with the inter-expert lookahead:

> ### Phase 4 — down-projection (grid re-partitioned: DOWN_GROUPS × DOWN_GRID)
>
> The same 128 blocks re-map: `down_group = blockIdx.x / DOWN_GRID`,
> `down_block_idx = blockIdx.x % DOWN_GRID`,
> `base_col = down_block_idx · DOWN_COL_TILE`.
>
> - **Expert loop**: `e = down_group, down_group + DOWN_GROUPS, ...` over
>   the same active-expert list.  Note the up and down loops visit experts
>   in a different interleaving; correctness needs only that Phase 3
>   finished the expert before Phase 4 reads it, which site #2 guarantees.
> - **Per expert, warp duties**:
>     - warp 8 lane 0 (launcher): prefetches K-step s+1 during step s; on
>     the last step prefetches the *next expert's* step-0 weight +
>     activation tiles instead (inter-expert lookahead).  The activation
>     tile is one bulk TMA per 128-K substep covering all ≤ 8 routed rows
>     of the expert (fetched from the contiguous `temp_fp8` slab).
>     - warp 8 (PF0): loads the expert's per-token activation scales;
>     warp 9 (PF1): loads the expert's weight scales — both once per
>     expert, in parallel, before the K-loop.
>     - warps 8–11 (128 PF threads): the **deferred accumulate of the
>     previous expert** — `out_accum[tok][col] += down_out[col][rank]`,
>     the (tok, col) plane sliced across the first `K_TILES_DOWN − 1`
>     iterations.

The cross-block synchronization section — this is the complete protocol:

> ## Cross-block synchronization (flag/sentinel handoffs)
>
> There are no grid barriers.  The kernel launches via plain
> `cudaLaunchKernel` (CUDA-Graph capturable) and orders its two cross-block
> handoffs through the data path, so consumers wait only on the values they
> actually need.  Both handoffs rely on the one-block-per-SM co-residency
> invariant (a spinning consumer needs its producers scheduled), enforced at
> compile time by `__launch_bounds__(BLOCK_SIZE, 1)` and at runtime by the
> `GRID_SIZE <= SM count` check in the wrapper.
>
> **Site #2 (Phase 3 → 4), sentinel-in-data:** each `temp_act_scale` cell
> doubles as the readiness flag for the fp8 payload segment it covers.  The
> producing warp stores the payload, then `__syncwarp()` +
> `__threadfence()` + `atomicExch` of the scale (`moe_publish_act_scale`),
> clamped to >= FLT_MIN so the sentinel `+0.0f` is never a valid value.  The
> down-projection polls exactly the cells of the expert it is about to
> consume (device-scope atomic reads) before reading scales or issuing the
> activation TMA; the inter-expert lookahead TMA has its own sweep-poll
> (`moe_wait_expert_scales_published`).  This gives per-expert granularity —
> down work for an expert starts as soon as that expert's rows are published.
>
> **Site #3 (Phase 4 → 5), arrival flags:** `down_partial_out` is
> atomicAdd-accumulated, so readiness cannot live in the data (a partial sum
> looks complete).  Each block instead bumps its col stripe's arrival
> counter (`__syncthreads()`, `__threadfence()`, `atomicAdd(+1)`) and runs
> to exit; only the stripe's single Phase-5 writer polls the counter up to
> `DOWN_GROUPS`.
>
> **Reset discipline (both sites):** flag state must never leak across
> launches, so it is double-buffered by launch parity: every block bumps its
> private `launch_flip` word once per launch (all blocks agree on parity
> with no cross-block sync), the current parity's state is consumed, and the
> OTHER parity's state is zero-refilled in the prologue, off the critical
> path.  A `torch.zeros` scratchpad allocation establishes the invariant for
> the first launch — no host-side re-init is ever needed, and the scheme is
> CUDA-Graph-replay safe (parity keeps alternating across replays).

The shared-memory budget, with the lifetime union and the bank-conflict padding:

> ## Shared memory (`MoE_SHM`, ≤ 224 KB)
>
> The dominant space is a union whose members have strictly disjoint lifetimes
> (separated by the Phase-2 trailing sync and the site-#2 barrier):
>
> | view | phase | size (N512_K2048 cfg0) |
> | -------------------------------------- | ----- | --------------- |
> | `bf16_in_full[K/128][BS][128]` | 1–2 | 32 KB |
> | `w_wgmma[UP_W_SLOTS][M_total][128]` | 3 | 64 KB |
> | `w_down_wgmma[2][DCT·K_SUBSTEPS][128]` | 4 | dominates |
>
> Other notable fields:
>
> - `fp8_act_full[K/128][8][T_TILE+1][16]` — single-buffer fp8 activations for
>   the whole K range (Phase 3 reads with no slot alternation).  The 9th
>   token row per 16-B chunk is padding: it moves the chunk stride from 128 B
>   to 144 B so the routing-quantize stores and the WGMMA B reads are
>   bank-conflict-free.  The WGMMA B descriptor's `LBO = 144` steps over the
>   pad.
> - `a_down_wgmma[2][K_SUBSTEPS_DOWN][8][8][16]` — down-proj activation
>   double-buffer (SWZ128 atoms).
> - `partial_result` union: `wgmma_out[128][9]` / `down_out[DCT][8]` /
>   `post_silu_scratch[UCH·128][9]` — the +1 column padding makes the
>   `[col][tok]` read pattern bijective over banks (gcd(9,32)=1).
> - mbarriers (`bar_w[UP_W_SLOTS]`, `bar_a[2]`, `bar_rwin`), `alignas(16)`.
> - Routing tables: `expert_slot_start[E]` (u16, `alignas(16)` — Phase B emits
>   packed STS.128 stores), `expert_routed_count[E]` (u8),
>   `sorted_slot[BS·8]` (u8), `down_rank[E][BS]` (u8, `alignas(16)` for the
>   uint4 seed), `up_rank_for_tok[_prev][BS]` + `up_rw_for_tok[BS]` (V2 only).
> - `act_scale[K/128][BS]` — transposed `[blk][tok]` layout for conflict-free
>   scale-apply broadcasts.

The global scratchpad, and the address-stability invariant required for graph replay:

> ## Global scratchpad (`MoEGemmSpec`)
>
> Persistent GM workspace, one per process, zeroed once on first launch.
> `mono_moe` allocates it automatically when the `scratchpad` argument is
> omitted; callers that want to reuse a buffer across launches allocate one
> explicitly with `alloc_scratchpad()`, sized by `get_scratchpad_size_bytes()`
> (the `monomoe_scratchpad_size` binding, i.e. `sizeof(MoEGemmSpec<Dims>)`):
>
> - `temp_fp8[TEMP_ROWS][N]` + `temp_act_scale[TEMP_ROWS][N/DOWN_ACT_BLOCK]` —
>   Phase 3 → Phase 4 handoff, expert-sorted rows.
>   **Layout invariant:** the host computes the device address of `temp_fp8`
>   as `scratchpad + TEMP_FP8_OFFSET` when building the down-activation TMA
>   descriptor, so no field may ever be inserted before `temp_fp8`; new fields
>   (handoff flags) go at the tail.  A `static_assert` in
>   `monomoe_binding.cu` enforces this.
> - `down_partial_out[BS][K]` fp32 — Phase 4 atomicAdd target.
> - sentinel-handoff tail state: `temp_act_scale_alt` (the second scale
>   buffer), `launch_flip[GRID_SIZE]` (per-block private launch counters →
>   buffer parity), `down_ready[2][DOWN_GRID]` (Phase-4→5 arrival counters).

TMA descriptor table:

> | descriptor | tensor | box | swizzle |
> | ---------------- | ------------------------------------ | ------- | ------- |
> | up weights | `[E·2N, K]` fp8 (interleaved or raw) | 128×128 | 128B |
> | activations | `[BS, K]` bf16 | 8×128 | none |
> | down weights | `[E·K, N]` fp8 (raw) | 128×128 | 128B |
> | down activations | `temp_fp8 [rows, N]` fp8 | 8×128 | 128B |

## Relevance

This is the closest thing to a specification for what we are building, at E=256 and BS<=8,
and it is in a shipping repository rather than a paper. Seven items transfer directly:

1. **One block per core, roles fixed, phase changes reassign work.** The Ascend analogue is one
   persistent kernel over all AI cores with `ASCEND_IS_AIC`/`ASCEND_IS_AIV` deciding the role
   and the phase deciding which expert stripe the core owns.
2. **Routing is replicated, not partitioned.** `the work is tiny: 8 tokens × E logits` — so
   every core computes top-k itself and no cross-core communication happens. That deletes the
   entire `moe_gating_top_k` / `moe_init_routing` / `moe_compute_expert_tokens` chain from the
   critical path, and it is the opposite of what CANN's `GroupedMatmulSwigluQuantV2` does when
   it walks `groupListGM` with scalar GM reads.
3. **No grid barriers at all.** Both handoffs ride the data path: the activation scale cell
   doubles as the readiness flag for its payload (`clamped to >= FLT_MIN so the sentinel +0.0f
   is never a valid value`), giving per-expert granularity. Compare our measured op, whose
   `pre/mid/post` pipeline does four `SyncAll` per iteration.
4. **The epilogue of expert *i* runs on the prefetch warps during expert *i+1*'s K-loop**, and
   the weight TMA for the next expert is stitched in during the last `SLOTS-2` iterations `so
   the pipeline never drains across experts`. That is the answer to per-group fixed overhead:
   there is no per-group drain.
5. **Gate/up weights pre-interleaved offline** so one tile fetch yields both halves and
   `silu(gate)·up is a register-local combine`. On Ascend this is a `npu_format_cast`-time
   decision, free at inference.
6. **On-chip buffers unioned by lifetime** (`bf16_in_full` / `w_wgmma` / `w_down_wgmma` share
   one region) with explicit `+1` padding to make bank access bijective. Our UB budget is the
   binding constraint; the `168 kb` figure in CANN's own GroupedMatmul design doc is for one
   stage only.
7. **Launch-parity double-buffered flag state**, so nothing leaks between replays and no host
   re-init is needed — the discipline that makes a persistent kernel safe under graph capture,
   which is exactly our aclgraph constraint.

Note the shape mismatch to check: this config is K=2048, N=512; GLM-5.2 at TP16 is K=6144,
N=128 per half. Our N is 4x smaller and K 3x larger, so `UCH = max(1, 2N·DCT/(128·K))` stays 1
but `UP_GRID = 2N/128 = 2` rather than 8 — far fewer cores per expert, far more experts in
parallel. The geometry formulas carry over; the constants do not.

---

### Cross-check: an independent persistent-kernel decode megakernel on AMD

https://github.com/ROCm/iris/pull/541
`MegaKernels in Iris` — ROCm/iris, author `neoblizz`, 2026-06-24, open.

> This pull request introduces the GPT-OSS-120B Megakernel example, providing a comprehensive, high-performance single-GPU implementation of the GPT-OSS-120B model using a persistent Triton kernel. The changes include detailed documentation, benchmarking and accuracy scripts, and a reusable set of device-level Triton operations for attention, quantization, and expert routing. Together, these updates enable end-to-end quantized inference, benchmarking, and accuracy evaluation for this large mixture-of-experts model.

> **2. Benchmarking and Evaluation Scripts:**
> - Introduced `bench_tpot.py` for measuring steady-state decode latency (TPOT) and comparing quantized vs. BF16 inference paths.
> - Added `bench_islosl.py` to benchmark prefill and decode latency across various input/output sequence lengths, reporting throughput and latency metrics.
> - Implemented `acc_eval.py` to compare accuracy between FP8 and BF16 attention (with shared FP4 experts), providing detailed metrics such as top-1 agreement, top-k overlap, KL divergence, and logit cosine similarity.

> **3. Reusable Device Operations:**
> - Created `common/__init__.py` to expose a suite of reusable Triton device ops for attention, quantization, GEMV, routing, and SwiGLU, supporting both single-GPU and future multi-GPU kernels.
> - Added `common/attention.py` containing Triton JIT helpers for RoPE+KV cache appending and per-head flash decode with attention sinks, facilitating both current and future kernel designs.

## Relevance

A second vendor, a third framework, and the same decomposition: one persistent kernel holding
attention + routing + quantization + GEMV + SwiGLU, built from a library of device-level
primitives rather than one monolith, with TPOT and an accuracy-vs-BF16 harness as the two
acceptance gates. The `GEMV` entry in the device-op list is the notable one — at batch=1 the
expert projections are matrix-vector, not matrix-matrix, which matches both the MonoMoE
SwapAB argument and Hazy Research's finding that matvec can beat the matrix unit. Note: a
search snippet attributed a quantized-path TPOT improvement of 29.4 ms to 22.9 ms to this
work; that figure is **not** in the PR body as fetched and is therefore not quoted here.

---

### MonoMoE Python entry point — the full shape and dtype contract

https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/fused_moe/monomoe.py

```python
def mono_moe(
    activations_in: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_scales_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    expert_scales_down: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    out: Optional[torch.Tensor] = None,
    scratchpad: Optional[torch.Tensor] = None,
    interleave_up: bool = True,
) -> torch.Tensor:
    """Single-kernel block-FP8 top-K MoE (fixed E256/N512/K2048 shape, SM90a).

    The ``(E, N, K)`` shape is derived from the weight tensors and must be the
    fixed E=256/N=512/K=2048 shape the kernel is specialized for; up to 8
    tokens (the BS8 kernel).

    Args:
        activations_in: bf16 input activations ``[M, K]`` (``M <= 8``).
        router_logits: bf16 router logits ``[M, E]``.
        expert_weights_up: fp8_e4m3 up/gate weights ``[E, 2*N, K]``. This
            function applies :func:`interleave_for_tma_wgmma_up` by default;
            pass ``interleave_up=False`` if the tensor is already interleaved.
        expert_scales_up: fp32 block-wise scales ``[E, 2N/128, K/128]``.
        expert_weights_down: fp8_e4m3 down weights ``[E, K, N]`` (raw row-major).
        expert_scales_down: fp32 block-wise scales ``[E, K/128, N/128]``.
        top_k: experts selected per token (1..8).
        scoring_func: ``"sigmoid"`` or ``"softmax"``.
        renormalize: renormalize the top-K weights to sum to 1.
        out: optional bf16 output buffer ``[M, K]``; allocated if omitted.
        scratchpad: optional reusable uint8 scratchpad from
            :func:`alloc_scratchpad`; allocated per-call if omitted.
        interleave_up: apply the gate/up TMA repack to ``expert_weights_up``
            (default True).

    Returns:
        bf16 MoE output ``[M, K]``.
    """
```

```python
    # The kernel reads the up weights in the interleaved Pair_Layout.
    if interleave_up:
        expert_weights_up = interleave_for_tma_wgmma_up(expert_weights_up)
    ...
    mod.monomoe_topk(
        activations_in,
        router_logits,
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
        out,
        scratchpad,
        int(top_k),
        int(sf_map[scoring_func]),
        bool(renormalize),
    )
```

## Relevance

The entry point takes **router logits**, not routed expert ids — routing lives inside the
kernel, so the host does no top-k and there is no permutation array on the interface at all.
That is the API shape our megakernel should present. The `scratchpad` argument being optional
but reusable across launches is the graph-replay affordance. The fixed-shape specialization
(`must be the fixed E=256/N=512/K=2048 shape the kernel is specialized for`) is the precedent
for our brief's instruction that generalization is not a goal.

---

---

## Part 8 — Grouped matmul over 256 experts inside a hand-written kernel: the group list, per-expert weight addressing, and where the code actually lives

Collected 2026-10-20 for question 2.2. Everything below is verbatim.

## Relevance
This is the material for driving all 256 routed experts from inside one hand-written kernel, against our measured
`GroupedMatmul` 5.596 ms / 1.212 ms of bytes and `GroupedMatmulSwigluQuantV2` 6.793 / 2.272.
- **The repo guess in our notes was wrong, and this is a clean negative from a 40200-path tree walk:
  `GroupedMatmul` is not in `ops-nn` at all.** It lives in `hicann/ops-transformer` under `gmm/`, with
  `mc2/mega_moe` holding the fused MoE megakernel — including an A3-specific `mega_moe_kernel_a3.hpp`. That file is
  the closest published thing to what we are building.
- **The per-group matmul re-point is quoted in full from two independent places**: the official device-side decoder
  `GetSplitValueFromGroupList` (all three `groupListType` modes), and the hand-written 910B kernel's `MMCompute` —
  `SetOrgShape(group M, N, K)` → `SetSingleShape(tile)` → `SetTensorA(xGm[xOffset])` → `SetTensorB(weightGmLocal)` →
  `IterateAll`, with `UpdateMnConfig` bumping all five offsets per group. Note how that differs from §39.1's
  `rope_matrix`, which re-points with `SetTensorA`/`SetTensorB`/`SetTail` only: a grouped matmul must also move
  `SetOrgShape` and `SetSingleShape` because the group's logical shape changes, not just its base address.
- **Per-expert weight addressing has two idioms**, both quoted: a TensorList of GM pointers dereferenced via
  `GetTensorAddr<T>(index, ptr)` (MegaMoE and the official GMM), or one contiguous `[E, K, N]` buffer advanced by
  `k * n` when `listLen == 1`. For us the second is the one to use, and the reason is a hard limit in the first:
  **the TensorList forms cap at 128 groups, and we have 256 experts.** `CATLASS` separately sizes
  `MAX_TENSOR_COUNT = 256`.
- **`group_list_type` is three-valued** — 0 cumsum, 1 count, 2 sparse `[[groupIdx, groupSize], ...]` compacted —
  with a three-way worked example below. Five different kernels disagree on which they honour, so the value must be
  read off the specific kernel rather than assumed. This supersedes the looser note in
  `a5-950-mc2-*` memory that `group_list_type` defaults to cumsum.
- **MegaMoE takes no host `groupList` at all**, deriving both forms on device from round-tagged per-rank counters.
  At batch=1 with top-8 of 256 that is the right shape for us too: the group list is eight entries, computable on
  device from the router output, and shipping it from the host would cost a launch.

# TOPIC B — grouped matmul over many experts inside a hand-written kernel

## B1. `cann/asc-devkit` `quant_group_matmul` — FULL KERNEL SOURCE

Verification note: of the 31 fenced blocks in this section, 28 were machine-checked as exact
literal substrings of the files downloaded from gitcode; the 3 that are not are each labelled
in place as either a `grep -n` inventory, a concatenation of two stated line ranges, or an
HTML-table elision.


Repo: `cann/asc-devkit` on gitcode.com, branch `master`. Fetched 2026-10-20.
Tree walked with the gitcode v5 contents API
(`https://gitcode.com/api/v5/repos/cann/asc-devkit/contents/<path>?ref=master`);
files pulled raw from `https://raw.gitcode.com/cann/asc-devkit/raw/master/<path>`.

### B1.0 Real repo paths (the ones in the task brief were approximate)

The quant group matmul sample is ONE directory, and the `.asc` file inside it is the thing
named `quant_group_matmul_custom`:

- `examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/`
  - `quant_group_matmul_custom.asc`  (684 lines, 32673 bytes — device kernel AND host main in one file)
  - `README.md` (31646 bytes), `README_en.md` (35749 bytes)
  - `CMakeLists.txt`, `data_utils.h`
  - `scripts/gen_data.py`, `scripts/verify_result.py`
  - `figures/CV流水优化1.png`, `figures/CV流水优化2.png`

404 / does-not-exist (recorded as requested):

- `https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_custom/`
  — NO SUCH DIRECTORY. There is no sibling `quant_group_matmul_custom/` directory anywhere in the
  repo; a full dir-level scan of the tree (4000 directories, `git/trees/master?recursive=1` paginated
  at 100/page over 40 pages) returns exactly one path matching `group.*matmul`:
  `examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance`.
- `https://gitcode.com/api/v5/repos/cann/asc-devkit` — HTTP 400 (repo metadata endpoint rejects;
  the `contents` and `git/trees` endpoints work fine).
- Note on the tree API: `git/trees/master?recursive=1` returns ONLY `type":"tree"` entries
  (directories), never blobs, so file discovery has to go through the `contents` endpoint.

There is no separate `.h` device header for this sample: the whole kernel plus the host `main()`
live in the single `.asc` translation unit. The `02_simd_c_api` / `03_simt_api` trees have no
group-matmul counterpart.

The one `.h` device header among the matrix best practices is
`examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_high_performance/matmul.h`
(and `matmul_mxfp4_high_performance/matmul_mx.h`) — not grouped.

---

### B1.1 The group-list loop (item 1)

`groupList` semantics, decided here: `groupListGlobal.GetValue(groupIdx)` is the ROW COUNT OF THAT
GROUP (per-group count), NOT a cumulative sum. The running sum is maintained on device by
`groupConfig.globalRowOffset += groupConfig.groupM;` at the bottom of the loop. The comment on the
struct field says so outright ("当前Group的M维度行数 (从groupList动态读取)" / "跨Group累积的全局行偏移").

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 40-54: sync flag ids, and the GroupConfig struct whose comments fix the groupList semantics.
```cpp
// ==================== 同步标志与常量 ====================
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint32_t BROADCAST_DIM = 2; // Broadcast维度
constexpr uint32_t BUFFER_NUM = 2;    // UB队列Buffer数, 设为2开启double buffer

// 记录当前子块在全局和Group内的位置信息, 在Process循环中逐步更新
struct GroupConfig {
    uint32_t groupM = 0;          // 当前Group的M维度行数 (从groupList动态读取)
    uint32_t mIdx = 0;            // 当前子块在Group内M方向的块索引
    uint32_t nIdx = 0;            // 当前子块在Group内N方向的块索引
    uint32_t numBlocksM = 0;      // 当前Group M方向分块数 = AscendC::Std::ceil_div(groupM, SINGLE_M)
    uint32_t numBlocksN = 0;      // N方向分块数 = AscendC::Std::ceil_div(N, SINGLE_N), 所有Group相同
    uint32_t globalRowOffset = 0; // 跨Group累积的全局行偏移, 影响x/perTokenScale/y的行定位
};
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 250-288: `Process()` — the group loop with the preCount cross-group round-robin, the GM read of the group list, and the globalRowOffset accumulation.
```cpp
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
__aicore__ inline void QuantGroupMatmul<
    AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
    PIPELINE_DEPTH>::Process()
{
    GroupConfig groupConfig;
    groupConfig.numBlocksN = AscendC::Std::ceil_div(N, SINGLE_N);

    // preCount: 上一Group末尾未被CORE_NUM整除的余数块数, 实现跨Group连续轮询
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < GROUP_NUM; ++groupIdx) {
        uint32_t groupM = static_cast<int32_t>(groupListGlobal.GetValue(groupIdx));
        if (groupM <= 0) {
            continue;
        }
        groupConfig.groupM = static_cast<uint32_t>(groupM);
        groupConfig.numBlocksM = AscendC::Std::ceil_div(groupConfig.groupM, SINGLE_M);

        uint32_t curCount = preCount + groupConfig.numBlocksN * groupConfig.numBlocksM;
        uint32_t curBlockIdx = coreIdx >= preCount ? coreIdx : coreIdx + CORE_NUM;

        while (curBlockIdx < curCount) {
            groupConfig.mIdx = (curBlockIdx - preCount) / groupConfig.numBlocksN;
            groupConfig.nIdx = (curBlockIdx - preCount) % groupConfig.numBlocksN;
            // Step 1: Cube侧矩阵乘 (AIC执行), AIV跳过
            MMCompute(groupIdx, groupConfig);
            // Step 2: Vector侧反量化+激活 (AIV执行), AIC跳过
            VectorCompute(groupIdx, groupConfig);
            curBlockIdx += CORE_NUM;
        }
        // 余数传递给下一Group, 跨Group保证负载均衡
        preCount = curCount % CORE_NUM;
        groupConfig.globalRowOffset += groupConfig.groupM;
    }
    // Step 3: 流水收尾, 等待最后若干Vector任务完成
    PostCompute();
}
```

---

### B1.2 Every `SetGlobalBuffer` and the per-group pointer arithmetic (item 2)

Note the shape of this design: `SetGlobalBuffer` is called ONCE per tensor in `Init()`, with NO
length argument and NO per-group re-pointing. Per-expert addressing is done instead by indexing the
`GlobalTensor` with a scalar offset at use time (`xGlobal[xOffset]`, `weightGlobal[weightOffset]`),
and handing that slice to `SetTensorA` / `SetTensorB`. The per-expert weight offset is
`groupIdx * N * K + curOffsetN * K`.

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 186-217: `Init()` — all seven `SetGlobalBuffer` calls, including workspace reinterpreted as the int32 Cube-output staging tensor, and the AIV coreIdx division by GetTaskRation().
```cpp
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
__aicore__ inline void QuantGroupMatmul<
    AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
    PIPELINE_DEPTH>::
    Init(
        __gm__ uint8_t* x, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* groupList,
        __gm__ uint8_t* scale, __gm__ uint8_t* perTokenScale, __gm__ uint8_t* y, __gm__ uint8_t* workspace,
        AscendC::TPipe* tPipeIn)
{
    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
    weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(weight));
    mmOutGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace)); // workspace作为Cube输出中转
    scaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    perTokenScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(perTokenScale));
    groupListGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));

    // 获取核索引: __mix__(1,2)模式下, AIC和AIV的GetBlockIdx()各自独立编号
    //   AIC侧返回0~CORE_NUM-1, AIV侧返回0~2*CORE_NUM-1
    //   AIV需除以GetTaskRation()(=2)得到逻辑核号, 与对应AIC共享同一coreIdx
    subBlockIdx = AscendC::GetSubBlockIdx();
    coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    pipe = tPipeIn;
    InitUbBuffer();
}
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 290-330: `MMCompute()` — the tail handling, then the two offset expressions (`xOffset`, `weightOffset`) that do the per-expert addressing, then the matmul object calls.
```cpp
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
__aicore__ inline void QuantGroupMatmul<
    AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
    PIPELINE_DEPTH>::MMCompute(uint32_t groupIdx, GroupConfig& groupConfig)
{
    uint32_t curOffsetN = groupConfig.nIdx * SINGLE_N;
    uint32_t curOffsetM = groupConfig.mIdx * SINGLE_M;

    uint32_t curSingleN = SINGLE_N;
    if (groupConfig.nIdx == groupConfig.numBlocksN - 1) {
        curSingleN = N - curOffsetN;
    }
    uint32_t curSingleM = SINGLE_M;
    if (groupConfig.mIdx == groupConfig.numBlocksM - 1) {
        curSingleM = groupConfig.groupM - curOffsetM;
    }

    uint64_t xOffset = (groupConfig.globalRowOffset + curOffsetM) * K;
    uint64_t weightOffset = groupIdx * N * K + curOffsetN * K;

    if ASCEND_IS_AIC {
        matmulObj.SetOrgShape(groupConfig.groupM, N, K);
        matmulObj.SetSingleShape(curSingleM, curSingleN, K);
        matmulObj.SetTensorA(xGlobal[xOffset]);
        auto weightSlice = weightGlobal[weightOffset];
        matmulObj.SetTensorB(weightSlice);

        while (matmulObj.Iterate()) {
            if (cubeTaskIdx >= PIPELINE_DEPTH) {
                AscendC::CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC);
            }
            uint64_t workSpaceOffset = BASE_N * BASE_M * (coreIdx + (cubeTaskIdx % PIPELINE_DEPTH) * CORE_NUM);
            matmulObj.GetTensorC(mmOutGlobal[workSpaceOffset], 0, true);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
            cubeTaskIdx++;
        }
    }
}
```

The offsets again, isolated (verbatim, lines 310-311 of the same file):

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 310-311. The two pointer-arithmetic expressions. `globalRowOffset` is the cumulative row sum built from the per-group counts; `weightOffset` strides whole experts by `N*K` and then N-blocks by `K` (valid because weight is stored NZ with the N-fractal outermost — see item 5).
```cpp
    uint64_t xOffset = (groupConfig.globalRowOffset + curOffsetM) * K;
    uint64_t weightOffset = groupIdx * N * K + curOffsetN * K;
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 356-367 (inside `VectorCompute`): the scale / output offsets, i.e. per-group per-channel scale addressing `groupIdx * N + nIdx * SINGLE_N` and the output row addressing.
```cpp
    uint64_t scaleOffset = groupIdx * N + groupConfig.nIdx * SINGLE_N;
    uint64_t outOffset = (groupConfig.globalRowOffset + groupConfig.mIdx * SINGLE_M) * N + groupConfig.nIdx * SINGLE_N;
    uint32_t taskRation = AscendC::GetTaskRation();

    // ---- 外层offsetN循环: 每次对应一次Cube Iterate, 沿N轴遍历 ----
    for (uint32_t offsetN = 0, vecCount = 0; offsetN < curCubeN; offsetN += BASE_N) {
        uint32_t curVecBaseN = BASE_N;
        if (unlikely(offsetN + BASE_N >= curCubeN)) {
            curVecBaseN = curCubeN - offsetN;
        }
        uint32_t alignBaseN = AscendC::Std::ceil_div(curVecBaseN, uint32_t(8)) * 8;
        // Step 1: 拷贝per-channel scale到UB
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 476-503: `DataCopyPerTokenScaleAndBrcb()` — per-token scale offset is the GLOBAL row number, i.e. globalRowOffset + mIdx*SINGLE_M + offsetM, with the comment stating it.
```cpp
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
__aicore__ inline void QuantGroupMatmul<
    AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
    PIPELINE_DEPTH>::
    DataCopyPerTokenScaleAndBrcb(GroupConfig& groupConfig, uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    // perTokenScale偏移: 全局行号 = globalRowOffset + mIdx×baseM + offsetM
    uint64_t perTokenScaleOffset = groupConfig.globalRowOffset + groupConfig.mIdx * SINGLE_M + offsetM;

    // GM → UB: 拷贝curBaseM个float
    AscendC::DataCopyPadExtParams<float> padParams;
    AscendC::DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGlobal[perTokenScaleOffset], perTokenScaleParams, padParams);
    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    // Broadcast: [curBaseM, 1] → [curBaseM, alignBaseN], 每行一个scale扩展到所有列
    perTokenScaleInUb = perTokenScaleInQueue.DeQue<float>();
    const uint32_t broadCastDst[BROADCAST_DIM] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curBaseM, 1};
    AscendC::BroadCast<float, BROADCAST_DIM, 1>(
        pertokenBrcbLocal, perTokenScaleInUb, broadCastDst, broadCastSrc, sharedTmpLocal);
    perTokenScaleInQueue.FreeTensor(perTokenScaleInUb);
}
```

---

### B1.3 Matmul-object API calls and the type declaration (item 3)

What this kernel actually calls: `SetOrgShape`, `SetSingleShape`, `SetTensorA`, `SetTensorB`,
`Iterate()` (in a `while`), `GetTensorC(dst, 0, true)`, and `REGIST_MATMUL_OBJ`.
What it does NOT call: `SetBias` (bias is passed in but `Data(0)`, unused), `SetQuantScalar`,
`SetQuantVector` (dequant is done on the Vector side with `AscendDequant`, not in FIXPIPE),
`IterateAll`, `End()`, `SetWorkspace`, `GetUserWorkspace`, `SyncAll`. For verbatim
`SetQuantScalar`/`SetQuantVector`/`SetBias`/`IterateAll`/`End` from the same repo see section 7.

The type declaration is a `matmul::Matmul<...>` with a compile-time-constant
`MatmulApiStaticTiling`, not a runtime `TCubeTiling` — and `REGIST_MATMUL_OBJ` is passed
`(TCubeTiling*)nullptr` for exactly that reason.

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 84-126: class head, the UB sizing constants, the "tiling constantization" block, the four `MatmulType` aliases (A=GM/ND/int8, B=GM/NZ/int8, C=GM/ND/int32, BIAS=GM/ND/int32), the arch-conditional depthA1/stepKa for dav-2201 vs others, and `GetCustomConstantCFG()`.
```cpp
// ==================== 量化分组矩阵乘 Kernel 类 ====================
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
class QuantGroupMatmul {
    // Vector单次处理的元素数: 24行 × BASE_N列 (UB容量约束, 24行int32=24×256=6144元素)
    static constexpr uint32_t UB_CAL_SIZE = 24 * BASE_N;
    // tmpBuff总大小, 用于存放反量化中间结果和API临时空间
    static constexpr uint32_t UB_REST_BYTES = 118784u;

    // ==================== Tiling常量化配置 ====================
    // 将Tiling参数从运行时变量提升为编译期常量, 消除Matmul初始化和迭代间的Scalar计算开销
    static constexpr MatmulShapeParams shapeParams = {SINGLE_M, SINGLE_N, K, BASE_M, BASE_N, BASE_K};
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    static constexpr int32_t TILING_DEPTHA_PARAM = 16;
    static constexpr int32_t TILING_STEPKA_PARAM = 8;
#else
    static constexpr int32_t TILING_DEPTHA_PARAM = 8;
    static constexpr int32_t TILING_STEPKA_PARAM = 4;
#endif
    static constexpr int32_t TILING_DEPTHB_PARAM = 8;
    static constexpr int32_t TILING_STEPKB_PARAM = 4;
    static constexpr int32_t TILING_STEPMN_PARAM = 1;

    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, BType>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;

    static __aicore__ inline constexpr MatmulApiStaticTiling GetCustomConstantCFG()
    {
        MatmulConfig mmCFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
        mmCFG.enUnitFlag = true;
        auto constantCFG = AscendC::GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG);
        constantCFG.depthA1 = TILING_DEPTHA_PARAM;
        constantCFG.depthB1 = TILING_DEPTHB_PARAM;
        constantCFG.stepKa = TILING_STEPKA_PARAM;
        constantCFG.stepKb = TILING_STEPKB_PARAM;
        constantCFG.stepM = TILING_STEPMN_PARAM;
        constantCFG.stepN = TILING_STEPMN_PARAM;
        return constantCFG;
    }
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 128-150: the public surface and the private method list — note `CONSTANT_CFG` and the `matmul::Matmul<...>` member declaration with its template args.
```cpp
public:
    __aicore__ inline QuantGroupMatmul() {}
    __aicore__ inline void Init(
        __gm__ uint8_t* x, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* groupList,
        __gm__ uint8_t* scale, __gm__ uint8_t* perTokenScale, __gm__ uint8_t* y, __gm__ uint8_t* workspace,
        AscendC::TPipe* tPipeIn);
    __aicore__ inline void Process();

    // Tiling常量化: 编译期确定Tiling参数, 消除运行时Scalar计算开销
    constexpr static auto CONSTANT_CFG = GetCustomConstantCFG();
    matmul::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;

private:
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void MMCompute(uint32_t groupIdx, GroupConfig& groupConfig);
    __aicore__ inline void VectorCompute(uint32_t groupIdx, GroupConfig& groupConfig);
    __aicore__ inline void PostCompute();
    __aicore__ inline void ComputeDequantAndActivate(
        GroupConfig& groupConfig, uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(
        GroupConfig& groupConfig, uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM);
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 151-185: the private members — every GlobalTensor with its logical shape in the comment, the four TQue queues, the single TBuf and its sub-tensors, and the two task counters `cubeTaskIdx` / `vecTaskIdx` that drive the ring workspace.
```cpp
private:
    GlobalTensor<int8_t> xGlobal;            // 左矩阵 x[M, K] int8
    GlobalTensor<int8_t> weightGlobal;       // 右矩阵 weight[GROUP_NUM, K, N] int8, NZ布局存储
    GlobalTensor<int32_t> mmOutGlobal;       // Cube输出 → workspace, int32
    GlobalTensor<float> scaleGlobal;         // per-channel反量化scale [GROUP_NUM, N]
    GlobalTensor<float> perTokenScaleGlobal; // per-token反量化scale [M]
    GlobalTensor<int64_t> groupListGlobal;   // 每组行数 [GROUP_NUM]
    GlobalTensor<half> yGlobal;              // 最终输出 y[M, N] half

    // ---- UB队列 (double buffer) ----
    AscendC::TQue<AscendC::TPosition::VECIN, 1> vecInQueue;           // Cube int32结果输入
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> vecOutQueue;         // half输出
    AscendC::TQue<AscendC::TPosition::VECIN, 1> scaleInQueue;         // per-channel scale
    AscendC::TQue<AscendC::TPosition::VECIN, 1> perTokenScaleInQueue; // per-token scale

    // ---- UB临时缓冲 ----
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuff; // 中间计算结果和API临时空间

    // ---- tmpBuff内部分区 (Vector侧使用) ----
    LocalTensor<float> dequantMiddleResult; // [UB_CAL_SIZE] AscendDequant输出: int32×scale → float
    LocalTensor<float> pertokenBrcbLocal; // [UB_CAL_SIZE] perTokenScale广播后: [vecBaseM, 1] → [vecBaseM, alignBaseN]
    LocalTensor<float> mulsResultLocal;   // [UB_CAL_SIZE] per-token反量化结果: dequant × perTokenScale
    LocalTensor<float> actResultLocal;    // [UB_CAL_SIZE] FasterGelu激活结果
    LocalTensor<uint8_t> sharedTmpLocal;  // [2×UB_CAL_SIZE×4] AscendDequant/BroadCast API内部临时空间
    LocalTensor<float> scaleInUb;         // 当前N切片的per-channel scale, 在offsetN循环内有效
    LocalTensor<float> perTokenScaleInUb; // 当前M切片的per-token scale, 在offsetM循环内有效

    uint32_t subBlockIdx;     // AIV子块索引 (__mix__模式下多AIV共享同一coreIdx时区分任务)
    uint32_t coreIdx;         // 当前核的逻辑索引 (AIC和AIV共享同一coreIdx)
    uint32_t cubeTaskIdx = 0; // AIC侧: 按Iterate粒度递增, 用于环形workspace Slot分配
    uint32_t vecTaskIdx = 0;  // AIV侧: 按offsetN粒度递增, 与cubeTaskIdx天然同步

    AscendC::TPipe* pipe; // 管道对象, 管理UB buffer分配
};
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 313-330: the matmul call sequence proper. `SetOrgShape(groupM, N, K)` uses the CURRENT group M (not global M); `SetSingleShape` the per-block tail-corrected shape; the `while (Iterate())` loop writes each BASE_M x BASE_N tile to its own workspace slot via `GetTensorC(dst, 0, true)` (third arg = enSequentialWrite).
```cpp
    if ASCEND_IS_AIC {
        matmulObj.SetOrgShape(groupConfig.groupM, N, K);
        matmulObj.SetSingleShape(curSingleM, curSingleN, K);
        matmulObj.SetTensorA(xGlobal[xOffset]);
        auto weightSlice = weightGlobal[weightOffset];
        matmulObj.SetTensorB(weightSlice);

        while (matmulObj.Iterate()) {
            if (cubeTaskIdx >= PIPELINE_DEPTH) {
                AscendC::CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC);
            }
            uint64_t workSpaceOffset = BASE_N * BASE_M * (coreIdx + (cubeTaskIdx % PIPELINE_DEPTH) * CORE_NUM);
            matmulObj.GetTensorC(mmOutGlobal[workSpaceOffset], 0, true);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
            cubeTaskIdx++;
        }
    }
}
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 504-535: the kernel entry point — `__mix__(1,1)` vs `__mix__(1,2)` by scenario, the `__kfc_workspace__` qualifier on the workspace arg, `InitSocState()`, the AIC-only `REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.matmulObj, (TCubeTiling*)nullptr)`, and the closing `PipeBarrier<PIPE_ALL>()`.
```cpp
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
#if SCENARIO_NUM == 0
__global__ __mix__(1, 1)
#else
__global__ __mix__(1, 2)
#endif
    void quant_group_matmul_custom(
        __gm__ uint8_t* x, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* groupList,
        __gm__ uint8_t* scale, __gm__ uint8_t* perTokenScale, __gm__ uint8_t* y,
        __kfc_workspace__ __gm__ uint8_t* workspace)
{
    AscendC::InitSocState();
    AscendC::TPipe tPipe;

    QuantGroupMatmul<
        AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
        PIPELINE_DEPTH>
        op;

    // REGIST_MATMUL_OBJ: 仅在AIC侧初始化Matmul对象
    if ASCEND_IS_AIC {
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.matmulObj, (TCubeTiling*)nullptr);
    }

    op.Init(x, weight, bias, groupList, scale, perTokenScale, y, workspace, &tPipe);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

---

### B1.4 Workspace slot arithmetic and the CV sync (items 4 and 6)

The whole workspace discipline is three lines of arithmetic plus two cross-core flags. There is no
`GetUserWorkspace`/`SetWorkspace`/`SyncAll` here: the user workspace is passed as the last kernel
argument tagged `__kfc_workspace__`, the system workspace is reached separately via
`GetSysWorkSpacePtr()` inside `REGIST_MATMUL_OBJ`, and the host sizes the user part as
`PIPELINE_DEPTH * BASE_N * BASE_M * sizeof(int32_t) * CORE_NUM`.

Producer side (AIC), verbatim:

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 320-329: Cube waits for a slot to be released only once it has gotten PIPELINE_DEPTH tiles ahead; the slot address is per-core AND per-pipeline-stage.
```cpp
        while (matmulObj.Iterate()) {
            if (cubeTaskIdx >= PIPELINE_DEPTH) {
                AscendC::CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC);
            }
            uint64_t workSpaceOffset = BASE_N * BASE_M * (coreIdx + (cubeTaskIdx % PIPELINE_DEPTH) * CORE_NUM);
            matmulObj.GetTensorC(mmOutGlobal[workSpaceOffset], 0, true);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
            cubeTaskIdx++;
        }
    }
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 332-398: `VectorCompute()` in full — the consumer side. The matching `mmOutOffset` uses `vecTaskIdx` where the producer used `cubeTaskIdx`; the two counters are kept in lockstep because the AIV `offsetN` loop granularity equals the AIC `Iterate` granularity. Note the `taskRation` filter that splits the offsetM iterations across the two AIVs, and the `CrossCoreSetFlag<2, PIPE_MTE2>` release at the end of each offsetN step.
```cpp
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
__aicore__ inline void QuantGroupMatmul<
    AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
    PIPELINE_DEPTH>::VectorCompute(uint32_t groupIdx, GroupConfig& groupConfig)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t curCubeN = SINGLE_N;
    if (groupConfig.nIdx == groupConfig.numBlocksN - 1) {
        curCubeN = N - groupConfig.nIdx * SINGLE_N;
    }
    uint32_t curCubeM = SINGLE_M;
    if (groupConfig.mIdx == groupConfig.numBlocksM - 1) {
        curCubeM = groupConfig.groupM - groupConfig.mIdx * SINGLE_M;
    }

    // vecBaseM: UB容量约束下单次Vector迭代可处理的行数
    uint32_t vecBaseM = UB_CAL_SIZE / BASE_N;
    vecBaseM = vecBaseM < curCubeM ? vecBaseM : curCubeM;

    uint64_t scaleOffset = groupIdx * N + groupConfig.nIdx * SINGLE_N;
    uint64_t outOffset = (groupConfig.globalRowOffset + groupConfig.mIdx * SINGLE_M) * N + groupConfig.nIdx * SINGLE_N;
    uint32_t taskRation = AscendC::GetTaskRation();

    // ---- 外层offsetN循环: 每次对应一次Cube Iterate, 沿N轴遍历 ----
    for (uint32_t offsetN = 0, vecCount = 0; offsetN < curCubeN; offsetN += BASE_N) {
        uint32_t curVecBaseN = BASE_N;
        if (unlikely(offsetN + BASE_N >= curCubeN)) {
            curVecBaseN = curCubeN - offsetN;
        }
        uint32_t alignBaseN = AscendC::Std::ceil_div(curVecBaseN, uint32_t(8)) * 8;
        // Step 1: 拷贝per-channel scale到UB
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        uint64_t mmOutOffset = BASE_N * BASE_M * (coreIdx + (vecTaskIdx % PIPELINE_DEPTH) * CORE_NUM);

        AscendC::CrossCoreWaitFlag<2>(SYNC_AIC_TO_AIV);

        // ---- 内层offsetM循环: 沿M轴遍历, 每次处理curVecBaseM行 ----
        for (uint32_t offsetM = 0; offsetM < curCubeM; offsetM += vecBaseM) {
            vecCount++;
            // AIV子任务过滤: __mix__多AIV场景下, 仅处理分配给当前subBlockIdx的任务
            if (taskRation != 0 && vecCount % taskRation != subBlockIdx) {
                continue;
            }
            uint32_t curVecBaseM = (offsetM + vecBaseM >= curCubeM) ? (curCubeM - offsetM) : vecBaseM;
            // workspace → UB
            LocalTensor<CType> mmOutLocal = vecInQueue.AllocTensor<CType>();
            DataCopyPad2D(
                mmOutLocal, mmOutGlobal[mmOutOffset + offsetM * curVecBaseN], curVecBaseM, curVecBaseN, curVecBaseN);
            vecInQueue.EnQue(mmOutLocal);
            // Step 2~5: 反量化+激活+类型转换
            ComputeDequantAndActivate(groupConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            // UB → GM
            LocalTensor<half> yLocal = vecOutQueue.DeQue<half>();
            DataCopyPad2D(yGlobal[outOffset + offsetM * N + offsetN], yLocal, curVecBaseM, curVecBaseN, alignBaseN, N);
            vecOutQueue.FreeTensor(yLocal);
        }
        scaleInQueue.FreeTensor(scaleInUb);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);
        vecTaskIdx++;
    }
}
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 399-416: `PostCompute()` — pipeline drain. AIC must absorb exactly `min(cubeTaskIdx, PIPELINE_DEPTH)` outstanding releases or the kernel deadlocks/exits early.
```cpp
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
__aicore__ inline void QuantGroupMatmul<
    AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
    PIPELINE_DEPTH>::PostCompute()
{
    if ASCEND_IS_AIC {
        if (likely(cubeTaskIdx > 0)) {
            uint32_t loop = cubeTaskIdx < PIPELINE_DEPTH ? cubeTaskIdx : PIPELINE_DEPTH;
            for (int32_t idx = 0; idx < loop; ++idx) {
                AscendC::CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC);
            }
        }
    }
}
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 417-454 + 455-474: `ComputeDequantAndActivate()` and `DataCopyScale()` — this is where every `PipeBarrier<PIPE_V>()` in the kernel lives (five of them), and where the UB temp-space aliasing is justified in comments.
```cpp
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
__aicore__ inline void QuantGroupMatmul<
    AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
    PIPELINE_DEPTH>::
    ComputeDequantAndActivate(
        GroupConfig& groupConfig, uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM)
{
    // Step 1: 读取per-token scale并Broadcast
    DataCopyPerTokenScaleAndBrcb(groupConfig, curVecBaseM, alignBaseN, offsetM);

    LocalTensor<CType> mmOutInUb = vecInQueue.DeQue<CType>();

    // Step 2: per-channel反量化: result[i][j] = mmOut[i][j] × scale[j]
    AscendDequant(dequantMiddleResult, mmOutInUb, scaleInUb, sharedTmpLocal, {curVecBaseM, alignBaseN, curVecBaseN});
    AscendC::PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);

    // Step 3: per-token反量化: result[i][j] = dequant[i][j] × perTokenScale[i]
    uint32_t computeSize = curVecBaseM * alignBaseN;
    Mul(mulsResultLocal, dequantMiddleResult, pertokenBrcbLocal, computeSize);
    AscendC::PipeBarrier<PIPE_V>();

    // Step 4: FasterGelu激活
    // 临时空间复用dequantMiddleResult和pertokenBrcbLocal的空间 (时序上不冲突)
    LocalTensor<uint8_t> actTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * UB_CAL_SIZE * sizeof(float), 0);
    FasterGelu(actResultLocal, mulsResultLocal, actTmpLocal, computeSize);
    AscendC::PipeBarrier<PIPE_V>();

    // Step 5: float32 → float16
    LocalTensor<half> yLocalInUb = vecOutQueue.AllocTensor<half>();
    Cast(yLocalInUb, actResultLocal, AscendC::RoundMode::CAST_NONE, computeSize);
    AscendC::PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}

// ==================== DataCopyScale: 拷贝per-channel scale到UB ====================
template <
    typename AType, typename BType, typename CType, typename BiasType, uint32_t M, uint32_t N, uint32_t K,
    uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K, uint32_t SINGLE_M, uint32_t SINGLE_N, uint32_t GROUP_NUM,
    uint32_t CORE_NUM, uint32_t PIPELINE_DEPTH>
__aicore__ inline void QuantGroupMatmul<
    AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
    PIPELINE_DEPTH>::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    // GM → UB: 拷贝curBaseN个float (一维拷贝, blockCount=1)
    AscendC::DataCopyPadExtParams<float> padParams;
    AscendC::DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<float>();
    DataCopyPad(scaleLocal, scaleGlobal[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);

    scaleInUb = scaleInQueue.DeQue<float>();
    scaleInUb.SetSize(alignBaseN); // 设置逻辑大小为对齐后的值, 供后续API使用
}
```

Full inventory of barriers / flags in this kernel (grep result, verbatim lines with numbers):

```text
322:                AscendC::CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC);
326:            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
371:        AscendC::CrossCoreWaitFlag<2>(SYNC_AIC_TO_AIV);
394:        AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);
411:                AscendC::CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC);
434:    AscendC::PipeBarrier<PIPE_V>();
440:    AscendC::PipeBarrier<PIPE_V>();
446:    AscendC::PipeBarrier<PIPE_V>();
451:    AscendC::PipeBarrier<PIPE_V>();
533:    AscendC::PipeBarrier<PIPE_ALL>();
```

(`SyncAll` appears zero times; there is no inter-core sync at all — cores are independent, only
AIC<->AIV on the same physical core synchronise.)

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 610-647 of the host `main()`: the shape/tiling constants, the arch split (dav-2201: SINGLE_M=BASE_M=128, CORE_NUM=24; dav-3510: 256/256/32), and the USER_WORKSPACE_SIZE formula that must match the device slot arithmetic exactly.
```cpp
int32_t main(int32_t argc, char* argv[])
{
    // 全局矩阵维度: x[M, K] × weight[GROUP_NUM, K, N] → y[M, N]
    constexpr uint32_t M = 8192;
    constexpr uint32_t N = 8192;
    constexpr uint32_t K = 1024;
    constexpr uint32_t BASE_N = 256;
    constexpr uint32_t BASE_K = 128;
    constexpr uint32_t GROUP_NUM = 8; // 分组数, 每组使用不同weight
    constexpr uint32_t PIPELINE_DEPTH = (scenarioNum >= 2) ? 4 : 1;
    constexpr uint32_t SINGLE_N = 1024;
#if (NPU_ARCH == 2201)
    constexpr uint32_t SINGLE_M = 128;
    constexpr uint32_t BASE_M = 128;  // M方向基块: 每次Iterate计算128行
    constexpr uint32_t CORE_NUM = 24; // 参与计算的AI Core核数
#elif (NPU_ARCH == 3510)
    constexpr uint32_t SINGLE_M = 256;
    constexpr uint32_t BASE_M = 256;  // M方向基块: 每次Iterate计算256行
    constexpr uint32_t CORE_NUM = 32; // 参与计算的AI Core核数
#endif

    // 数据类型定义
    using AType = int8_t;
    using BType = int8_t;
    using CType = int32_t;
    using BiasType = int32_t;

    constexpr size_t USER_WORKSPACE_SIZE = PIPELINE_DEPTH * BASE_N * BASE_M * sizeof(int32_t) * CORE_NUM;

#if defined(ASCENDC_CPU_DEBUG) && (__NPU_ARCH__ == 2201)
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B1");
#elif defined(ASCENDC_CPU_DEBUG) && (__NPU_ARCH__ == 3510)
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend950PR_9589");
#else
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
#endif
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = USER_WORKSPACE_SIZE + systemWorkspaceSize;
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc
author/date: last touched by yhf0922, 2026-09-04T16:53:45+08:00 (commit 55215c137e9f, "声明CeilDivision接口，整改替换CeilDivision为ceil_div"); Copyright (c) 2025 Huawei Technologies Co., Ltd. — repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 648-684: the rest of host main — workspace = user + GetLibApiWorkSpaceSize(), the argument sizes with their shapes in comments, and the `<<<CORE_NUM, 0, stream>>>` launch with the full template argument list.
```cpp

    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    OpArgs args{
        .x = Data(M * K * sizeof(int8_t)),                  // x[8192, 1024] int8, 8MB
        .weight = Data(GROUP_NUM * K * N * sizeof(int8_t)), // weight[8, 1024, 8192] int8, 64MB
        .bias = Data(0),                                    // 无偏置
        .groupList = Data(GROUP_NUM * sizeof(int64_t)),     // groupList[8] int64, 64B
        .scale = Data(GROUP_NUM * N * sizeof(float)),       // scale[8, 8192] float, 256KB
        .perTokenScale = Data(M * sizeof(float)),           // perTokenScale[8192] float, 32KB
        .workspace = Data(workspaceSize, false),            // workspace
        .y = Data(M * N * sizeof(int16_t), false)           // y[8192, 8192] half, 128MB
    };

    CreateInput(args);

    quant_group_matmul_custom<
        AType, BType, CType, BiasType, M, N, K, BASE_M, BASE_N, BASE_K, SINGLE_M, SINGLE_N, GROUP_NUM, CORE_NUM,
        PIPELINE_DEPTH><<<CORE_NUM, 0, stream>>>(
        args.x.device, args.weight.device, args.bias.device, args.groupList.device, args.scale.device,
        args.perTokenScale.device, args.y.device, args.workspace.device);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(args.y.host, args.y.size, args.y.device, args.y.size, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output.bin", args.y.host, args.y.size);
    FreeData(args);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```

---

### B1.5 groupList semantics, NZ weight layout, and the tiling struct (item 5)

There is no `TilingData` struct for this sample at all: every tiling number is a C++ template
parameter / `constexpr`, and the matmul tiling is a `constexpr MatmulApiStaticTiling` computed by
`GetMatmulApiTiling` at compile time (see section 3). The only host-side "tiling code" is the
constant block in `main()` quoted just above.

The decisive evidence for `groupList` = per-group row counts (NOT cumulative sum) is the data
generator: it writes `[1024]*8` to `groupList.bin` and does its OWN `np.cumsum` on the host to slice
the golden. The device never sees the cumsum.

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/scripts/gen_data.py
author/date: unknown (not in file header) — Copyright (c) 2025 Huawei Technologies Co., Ltd.; repo cann/asc-devkit branch master, fetched 2026-10-20
Lines 15-70: the whole generator. `groupList = np.array([1024] * groupNum, dtype=np.int64)` is per-group counts; `index = np.cumsum(groupList)` is host-only; and the last lines give the exact NZ fractal transform for the weight (k/16,16 x n/32,32 -> transpose to put the N-fractal outermost), which is what makes `weightOffset = groupIdx*N*K + curOffsetN*K` correct.
```python
import os
import numpy as np


def FastGelu(x):
    abs_x = np.abs(x)
    x2 = np.exp(0.851 * (x - abs_x))
    x3 = 1 + np.exp(-1.702 * abs_x)
    return x * x2 / x3


def gen_golden_data():
    groupNum = 8
    m = 8192
    k = 1024
    n = 8192
    groupList = np.array([1024] * groupNum, dtype=np.int64)

    x = np.random.randint(-10, 10, [m, k]).astype(np.int8)
    weight = np.random.randint(-10, 10, [groupNum, k, n]).astype(np.int8)
    scale = np.random.normal(0, 0.01, (groupNum, n)).astype(np.float32)
    perTokenScale = np.random.normal(0, 0.01, (m, 1)).astype(np.float32)

    # 关键优化：一次性转换为 float32，让 BLAS 加速矩阵乘法
    x_f32 = x.astype(np.float32)
    weight_f32 = weight.astype(np.float32)

    # 预计算分割索引
    index = np.cumsum(groupList)
    indices = [0] + index.tolist()

    mmOut = np.empty((m, n), dtype=np.float32)

    for i in range(groupNum):
        start_idx = indices[i]
        end_idx = indices[i + 1]
        # 直接使用 float32 矩阵乘法，BLAS 会多核并行加速
        mmOut[start_idx:end_idx] = np.dot(x_f32[start_idx:end_idx], weight_f32[i])
        # 应用 scale 和 perTokenScale
        mmOut[start_idx:end_idx] *= scale[i] * perTokenScale[start_idx:end_idx]

    golden = FastGelu(mmOut)
    golden = golden.astype(np.float16)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x.tofile("./input/x.bin")
    weight_nz = weight.reshape([groupNum, k // 16, 16, n // 32, 32]).transpose([0, 3, 1, 2, 4])
    weight_nz.tofile("./input/weight.bin")
    groupList.tofile("./input/groupList.bin")
    scale.tofile("./input/scale.bin")
    perTokenScale.tofile("./input/perTokenScale.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
```

The README spells out the same semantics in its own pseudo-code, with the comment
"本组行数, 本例=1024" (= rows of THIS group) on the groupList read:

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/README.md
author/date: unknown — repo cann/asc-devkit branch master, fetched 2026-10-20
README.md lines 186-216: the block-dimension definitions and the annotated Process loop.
`````markdown
Process函数采用**跨Group连续轮询**策略分配子块给多个Core。核心思想是将所有Group的子块在逻辑上连续编号，每个Core按`CORE_NUM`步长轮询取块，用`preCount`变量记录上一Group末尾未被`CORE_NUM`整除的余数，实现跨Group的连续分配，避免核间负载不均。

**分块维度说明**：

- **GroupM**：当前Group的M维度行数，从groupList动态读取。本样例每组GroupM=1024
- **numBlocksM = AscendC::Std::ceil_div(GroupM, SINGLE_M)**：当前Group在M方向的子块数。Atlas A2下 AscendC::Std::ceil_div(1024, 128) = 8，Ascend 950PR下 AscendC::Std::ceil_div(1024, 256) = 4
- **numBlocksN = AscendC::Std::ceil_div(N, SINGLE_N)**：N方向的子块数，所有Group相同。AscendC::Std::ceil_div(8192, 1024) = 8

每个Group共 `numBlocksM × numBlocksN` 个子块，通过`(mIdx, nIdx)`二维坐标定位。每个子块大小为 `[SINGLE_M, SINGLE_N]`，Cube内部按`[BASE_M, BASE_N]`粒度计算（每个子块需 `SINGLE_N/BASE_N × SINGLE_M/BASE_M` 次Iterate）。

```cpp
for (groupIdx = 0, preCount = 0; groupIdx < GROUP_NUM; ++groupIdx) {
    groupM = groupListGlobal.GetValue(groupIdx);          // 本组行数, 本例=1024
    numBlocksM = AscendC::Std::ceil_div(groupM, SINGLE_M);                   // M方向分块数
    numBlocksN = AscendC::Std::ceil_div(N, SINGLE_N);                         // N方向分块数, 所有组相同=8
    curCount = preCount + numBlocksM * numBlocksN;          // 含上组余数的连续编号上界

    // 核心分配公式：确定当前Core在本Group中的第一个块编号
    curBlock = coreIdx >= preCount ? coreIdx : coreIdx + CORE_NUM;

    while (curBlock < curCount) {
        mIdx = (curBlock - preCount) / numBlocksN;          // 组内M方向块索引
        nIdx = (curBlock - preCount) % numBlocksN;          // 组内N方向块索引
        MMCompute(groupIdx, groupConfig);                    // Cube计算
        VectorCompute(groupIdx, groupConfig);                // Vector计算
        curBlock += CORE_NUM;                                // 轮询步进，步长为CORE_NUM
    }
    preCount = curCount % CORE_NUM;                          // 余数传递给下一组
    globalRowOffset += groupM;                               // 全局行偏移累积
}
```
`````

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/README.md
author/date: unknown — repo cann/asc-devkit branch master, fetched 2026-10-20
README.md lines 337-364: the workspace ring-slot design stated in prose + the layout diagram, including the exact offset formula and the stated sync granularity ("同步粒度为Iterate（=BASE_M×BASE_N），而非子块（=SINGLE_M×SINGLE_N）").
`````markdown
**Workspace环形Slot复用**

workspace是Cube和Vector之间的中转站，采用环形分槽设计。这里的 Slot 指由 `workSpaceOffset = BASE_N × BASE_M × (coreIdx + (cubeTaskIdx % PIPELINE_DEPTH) × CORE_NUM)` 定位的一段连续空间，单个 core 的 Slot 大小为 `BASE_M×BASE_N×sizeof(int32_t)`，每层 pipeline 的 Slot 总空间为 `CORE_NUM×BASE_M×BASE_N×sizeof(int32_t)`，`PIPELINE_DEPTH` 个 Slot 轮流使用，Cube 写入当前 Slot 的同时，Vector 可以处理之前 Slot 的数据：

```
workspace布局:
Slot 0: Core0[BASE_M×BASE_N] + Core1[BASE_M×BASE_N] + ... + Core(CORE_NUM-1)[BASE_M×BASE_N]
Slot 1: Core0[BASE_M×BASE_N] + Core1[BASE_M×BASE_N] + ... + Core(CORE_NUM-1)[BASE_M×BASE_N]
Slot 2: ...
Slot 3: ...

workSpaceOffset = BASE_N × BASE_M × (coreIdx + (cubeTaskIdx % PIPELINE_DEPTH) × CORE_NUM)
```

- 当`cubeTaskIdx < PIPELINE_DEPTH`时，Cube无需等待，直接写入新Slot
- 当`cubeTaskIdx >= PIPELINE_DEPTH`时，Cube需等待Vector释放对应Slot（`CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC)`）
- 同步粒度为Iterate（=BASE_M×BASE_N），而非子块（=SINGLE_M×SINGLE_N）

**4级流水时序**：

```
时间 ──────────────────────────────────────────────────────────────>
Cube:  |─Iter0→Slot0─|─Iter1→Slot1─|─Iter2→Slot2─|─Iter3→Slot3─|─Iter4→Slot0─| ...
                      (Wait)                                     (Wait)
Vec:                 |─Slot0处理─|─Slot1处理─|─Slot2处理─|─Slot3处理─| ...
```

前4次Iterate（Slot 0~3）Cube无需等待，连续写入；第5次Iterate复用Slot 0时，需等Vector释放。由此形成Cube-Vector流水重叠，相比Case 1的串行模式，显著减少Cube空闲等待时间。
`````

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/README.md
author/date: unknown — repo cann/asc-devkit branch master, fetched 2026-10-20
README.md lines 229-250: the tiling-constantization and UnitFlag rationale.
`````markdown
8组共512块，Core0~7各处理22块，Core8~23各处理21块，最大负载差仅1块。

**③ Tiling常量化**

所有Tiling参数通过`MatmulApiStaticTiling`在编译期确定，运行时无需Scalar动态计算，减少Scalar开销：

```cpp
constexpr static auto CONSTANT_CFG = GetCustomConstantCFG();
matmul::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;
```


**④ Matmul UnitFlag优化**

本样例通过`mmCFG.enUnitFlag = true`开启Matmul的UnitFlag优化，实现Cube计算与搬运的512B细粒度流水同步：

```cpp
static constexpr MatmulShapeParams shapeParams = {SINGLE_M, SINGLE_N, K, BASE_M, BASE_N, BASE_K};
MatmulConfig mmCFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
mmCFG.enUnitFlag = true;  // 开启UnitFlag
auto constantCFG = AscendC::GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG);
```
`````

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/README.md
author/date: unknown — repo cann/asc-devkit branch master, fetched 2026-10-20
README.md lines 126-156: the data-flow path through the memory hierarchy — this is the megakernel-relevant shape of the fusion (L0C -> GM workspace -> UB, no L0C->UB on dav-2201).
`````markdown

### 数据流路径

```text
存储位置                        数据类型           流水

GM (x, weight)                   int8
  |                                               MTE2
L1                               int8
  |                                               MTE1
L0A / L0B                        int8
  |  Iterate                                     MMAD
L0C                              int32
  |                                               FIXPIPE
GM (workspace)                   int32
  |  DataCopyPad2D                                MTE2
UB vecInQueue                    int32
  |  AscendDequant (×scale)   int32 → float       VECTOR
UB dequantMiddleResult            float
  |  Mul (×perTokenScale)     float → float       VECTOR
UB mulsResultLocal                float
  |  FasterGelu               float → float       VECTOR
UB actResultLocal                 float
  |  Cast                     float → half        VECTOR
UB vecOutQueue                    half
  |  DataCopyPad2D                                MTE3
GM (y)                            half

注：scale[float] 和 perTokenScale[float] 通过 MTE2 独立搬入UB，
    分别参与 AscendDequant 和 Mul 步骤。
```
`````

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/CMakeLists.txt
author/date: unknown — repo cann/asc-devkit branch master, fetched 2026-10-20
CMakeLists.txt in full: `dav-2201` is the default arch, SCENARIO_NUM 0..2 is the only knob, and NPU_ARCH / SCENARIO_NUM are injected as compile definitions.
```cmake
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, sim")
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture: dav-2201, dav-3510")
set(SCENARIO_NUM 0 CACHE STRING "Case type to compile (0-2)")
if(NOT SCENARIO_NUM MATCHES "^[0-2]$")
    message(FATAL_ERROR "Unsupported SCENARIO_NUM=${SCENARIO_NUM}, only 0, 1 and 2 are supported")
endif()

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(demo
    quant_group_matmul_custom.asc
)

if(CMAKE_ASC_ARCHITECTURES STREQUAL "dav-2201")
    set(NPU_ARCH_ID 2201)
elseif(CMAKE_ASC_ARCHITECTURES STREQUAL "dav-3510")
    set(NPU_ARCH_ID 3510)
else()
    message(FATAL_ERROR "Unsupported CMAKE_ASC_ARCHITECTURES=${CMAKE_ASC_ARCHITECTURES}, only dav-2201 and dav-3510 are supported")
endif()

target_link_libraries(demo PRIVATE
    tiling_api
    register
    platform
    m
    dl
)
message(STATUS "[INFO] SCENARIO_NUM is ${SCENARIO_NUM}")

target_compile_definitions(demo PRIVATE
    NPU_ARCH=${NPU_ARCH_ID}
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
```

---

### B1.6 `CrossCoreSetFlag` / `SyncAll` / `PipeBarrier` in the sibling fusion sample

Same directory tree, `matmul_gelu_high_performance/mmad_gelu.asc` — the hand-written (basic-API,
no Matmul object) CV fusion. Useful contrast: it uses `Fixpipe` directly and, on dav-3510 only,
can land L0C straight into UB (`CFG_ROW_MAJOR_UB`, `dualDstCtl = 0b01`), which the dav-2201 path
cannot — it must go through GM.

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/matmul_gelu_high_performance/mmad_gelu.asc
author/date: unknown — repo cann/asc-devkit branch master, fetched 2026-10-20
mmad_gelu.asc lines 519-614: the Fixpipe-out + CrossCoreSetFlag<0x2, PIPE_FIX>(0x8) producer and the GM-staging consumer with its intra-core SetFlag/WaitFlag pairs.
```cpp

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        // ---- A2/A3架构：仅支持GM中转模式，Fixpipe CO1 → GM ----
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = curN;
        fixpipeParams.mSize = curM;
        fixpipeParams.srcStride = curMAlign;
        fixpipeParams.dstStride = N;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.unitFlag = 3;
        AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_GM>(
            cGM[mBlockIdx * baseM * N + nBlockIdx * baseN], cLocal, fixpipeParams);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // ---- Ascend 950PR架构：根据SCENARIO_NUM选择GM中转/UB直通模式 ----
        if constexpr (scenarioNum == 1) {
            // ---- GM中转模式：Fixpipe CO1 → GM ----
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.mSize = curM;
            fixpipeParams.nSize = curN;
            fixpipeParams.srcStride = curMAlign;
            fixpipeParams.dstStride = N;
            fixpipeParams.unitFlag = 3;
            AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_GM>(
                cGM[mBlockIdx * baseM * N + nBlockIdx * baseN], cLocal, fixpipeParams);
        } else {
            // ---- UB直通模式：Fixpipe CO1 → UB（dualDstCtl=0b01按M维度拆分） ----
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.mSize = AscendC::Std::ceil_div(curM, 2) * 2;
            fixpipeParams.nSize = curN;
            fixpipeParams.srcStride = curMAlign;
            fixpipeParams.dstStride = curN;
            fixpipeParams.dualDstCtl = 0b01;
            fixpipeParams.unitFlag = 3;
            AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_UB>(xUB, cLocal, fixpipeParams);
        }
#endif
        // 通知AIV侧数据已就绪
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
    }

    // AIV侧（GM中转）：等待AIC写入GM → DataCopy GM→xUB → GELU计算 → DataCopy geluOutUB→GM
    __aicore__ inline void GeluFromGM(uint32_t mBlockIdx, uint32_t nBlockIdx)
    {
        // 等待AIC侧Fixpipe完成，Matmul结果已写入GM
        AscendC::CrossCoreWaitFlag<0x2>(0x8);

        uint32_t curM = (mBlockIdx != baseMCount) ? baseM : tailM;
        uint32_t curN = (nBlockIdx != baseNCount) ? baseN : tailN;
        uint32_t halfM = curM / 2;
        uint32_t computeLen = halfM * curN;
        uint32_t localSubIdx = AscendC::GetSubBlockIdx() % 2;
        uint32_t gmOffset = mBlockIdx * baseM * N + nBlockIdx * baseN + localSubIdx * baseM / 2 * N;

        // ---- Step 1: DataCopy GM → xUB ----
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = halfM;
        copyInParams.blockLen = curN * sizeof(float);
        copyInParams.srcStride = (N - curN) * sizeof(float);
        copyInParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad<float>(xUB, cGM[gmOffset], copyInParams, padParams);

        // 等待MTE2（GM→UB）完成，确保V侧可以读取
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // ---- Step 2: GELU计算 ----
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        // A2/A3: MemBase方式
        GeluMemBaseCompute(xUB, geluOutUB, computeLen);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // Ascend 950PR: RegBase方式
        GeluRegBaseCompute(xUB, geluOutUB, computeLen);
#endif

        // 等待V计算完成，MTE3可以开始搬运
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // ---- Step 3: DataCopy geluOutUB → GM ----
        AscendC::DataCopyParams copyOutParams;
        copyOutParams.blockCount = halfM;
        copyOutParams.blockLen = curN * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = (N - curN) * sizeof(float);
        AscendC::DataCopyPad<float>(cGM[gmOffset], geluOutUB, copyOutParams);

        // ---- 反向同步：确保V读完xUB后，下一轮循环的MTE2才能重新写入xUB ----
        // SetFlag挂在V队列（V执行完GELU后才置位），WaitFlag挂在MTE2队列
        // （下轮写xUB前必须等待），防止新数据覆盖尚未读完的旧数据
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }
```

---

### B1.7 `SetQuantScalar` / `SetQuantVector` / `SetBias` / `IterateAll` / `End` (verbatim, nearest sample)

These five APIs do not appear in the group-matmul kernel. The repo's canonical use is
`examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_quant/matmul_quant.asc` — int8 x int8 ->
half with FIXPIPE-side dequant, i.e. the alternative to doing `AscendDequant` on the Vector side.
Note that `SetQuantVector` takes a `GlobalTensor<uint64_t>` and that the host must declare the mode
through `tilingApi.SetDequantType(DequantType::TENSOR | SCALAR)`.

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_quant/matmul_quant.asc
author/date: unknown (not in file header) — Copyright (c) 2025 Huawei Technologies Co., Ltd.; repo cann/asc-devkit branch master, fetched 2026-10-20
matmul_quant.asc lines 42-99: the `AscendC::Matmul<...CFG_MDL>` declaration (runtime TCubeTiling flavour), `quantGlobal` as GlobalTensor<uint64_t>, and the GlobalTensor re-pointing idiom `aGlobal = aGlobal[offsetA];` — the alternative to indexing at use site.
```cpp
template <typename AType, typename BType, typename CType, typename BiasType>
class MatmulKernel {
public:
    __aicore__ inline MatmulKernel(){};
    __aicore__ inline void Init(
        __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias, __gm__ uint8_t* c, __gm__ uint8_t* quant,
        const TCubeTiling& tiling);
    __aicore__ inline void Process();

    AscendC::Matmul<
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
        matmulObj;

private:
    __aicore__ inline void CalcOffset(
        int32_t blockIdx, int32_t& offsetA, int32_t& offsetB, int32_t& offsetC, int32_t& offsetBias);

    AscendC::GlobalTensor<AType> aGlobal;
    AscendC::GlobalTensor<BType> bGlobal;
    AscendC::GlobalTensor<BiasType> biasGlobal;
    AscendC::GlobalTensor<CType> cGlobal;
    AscendC::GlobalTensor<uint64_t> quantGlobal;
    TCubeTiling tiling;
    int32_t mCoreIndex;
    int32_t nCoreIndex;
};

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulKernel<AType, BType, CType, BiasType>::Init(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias, __gm__ uint8_t* c, __gm__ uint8_t* quant,
    const TCubeTiling& tiling)
{
    this->tiling = tiling;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(bias), tiling.N);
    quantGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(quant), tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    int32_t offsetBias = 0;
    CalcOffset(AscendC::GetBlockIdx(), offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    quantGlobal = quantGlobal[offsetBias];

    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_quant/matmul_quant.asc
author/date: unknown (not in file header) — Copyright (c) 2025 Huawei Technologies Co., Ltd.; repo cann/asc-devkit branch master, fetched 2026-10-20
matmul_quant.asc lines 101-168: `SetQuantVector` / `SetQuantScalar` (with the float->uint64 bit reinterpretation), `SetTail`, `SetTensorA/B` with the trans flags, conditional `SetBias`, `IterateAll(cGlobal)`, `End()`, `CalcOffset` with the full offset algebra, and the `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), ..., &tiling)` runtime-tiling form.
```cpp
template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulKernel<AType, BType, CType, BiasType>::Process()
{
    if (matmul::GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }

#if defined(CUSTOM_QUANT_VECTOR)
    matmulObj.SetQuantVector(quantGlobal);
#else
    float quantFloat = 0.1f;
    uint64_t quantValue = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&quantFloat));
    matmulObj.SetQuantScalar(quantValue);
#endif

    // process with tail block
    int tailN = tiling.N - nCoreIndex * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
    int tailM = tiling.M - mCoreIndex * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN) {
        matmulObj.SetTail(tailM, tailN);
    }

    matmulObj.SetTensorA(aGlobal, IS_TRANS_A);
    matmulObj.SetTensorB(bGlobal, IS_TRANS_B);
    if (tiling.isBias) {
        matmulObj.SetBias(biasGlobal);
    }
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
}

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulKernel<AType, BType, CType, BiasType>::CalcOffset(
    int32_t blockIdx, int32_t& offsetA, int32_t& offsetB, int32_t& offsetC, int32_t& offsetBias)
{
    const TCubeTiling& tiling = this->tiling;

    auto mSingleBlocks = (tiling.M + tiling.singleCoreM - 1) / tiling.singleCoreM; // split M into mSingleBlocks cores
    mCoreIndex = blockIdx % mSingleBlocks;
    nCoreIndex = blockIdx / mSingleBlocks;

    offsetA = mCoreIndex * tiling.Ka * tiling.singleCoreM;
    if (IS_TRANS_A) {
        offsetA = mCoreIndex * tiling.singleCoreM;
    }
    offsetB = nCoreIndex * tiling.singleCoreN;
    if (IS_TRANS_B) {
        offsetB = nCoreIndex * tiling.Kb * tiling.singleCoreN;
    }
    offsetC = mCoreIndex * tiling.N * tiling.singleCoreM + nCoreIndex * tiling.singleCoreN;
    offsetBias = nCoreIndex * tiling.singleCoreN;
}

__global__ __cube__ void matmul_quant_custom(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias, __gm__ uint8_t* c, __gm__ uint8_t* quant,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tilingGm)
{
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    AscendC::TPipe pipe;

    MatmulKernel<int8_t, int8_t, half, int32_t> matmulKernel;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tiling);
    matmulKernel.Init(a, b, bias, c, quant, tiling);
    matmulKernel.Process();
}
```

https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_quant/matmul_quant.asc
author/date: unknown (not in file header) — Copyright (c) 2025 Huawei Technologies Co., Ltd.; repo cann/asc-devkit branch master, fetched 2026-10-20
matmul_quant.asc lines 31-40 and 170-203: `CopyTiling` (the uint32-at-a-time GM->stack copy of TCubeTiling) and the host `GenerateTiling` with `SetDequantType`, `SetOrgShape`, `SetShape`, `EnableBias`, `SetBufferSpace(-1,-1,-1)`.
```cpp
__aicore__ inline void CopyTiling(TCubeTiling* tiling, __gm__ uint8_t* tilingGM)
{
    uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

void GenerateTiling(platform_ascendc::PlatformAscendC* ascendcPlatform, uint8_t* tilingBuf)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(ascendcPlatform->GetCoreNumAic());
    tilingApi.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, IS_TRANS_A);
    tilingApi.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, IS_TRANS_B);
    tilingApi.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBiasType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);

#if defined(CUSTOM_QUANT_VECTOR)
    tilingApi.SetDequantType(matmul_tiling::DequantType::TENSOR); // set TENSOR quant mode
#else
    tilingApi.SetDequantType(matmul_tiling::DequantType::SCALAR); // set SCALAR quant mode
#endif

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.EnableBias(IS_BIAS);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData); // Get matmul tiling data.
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
        return;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
}
```

---

### B1.8 `SetWorkspace` and `GetUserWorkspace` reference docs (item 4, API semantics)

Quoted because the group kernel deliberately does neither, and the distinction matters for a
megakernel: `SetWorkspace` is for the ASYNC `Iterate`/`GetTensorC` and `IterateNBatch` paths, and
is explicitly unsupported under MixDualMaster.

https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetWorkspace.md
author/date: unknown — repo cann/asc-devkit branch master, fetched 2026-10-20
SetWorkspace.md in full.
`````markdown
# SetWorkspace

## 产品支持情况

<!-- npu="950" id2 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- Atlas A3系列产品：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- Atlas A2系列产品：支持
<!-- end id4 -->
<!-- npu="310b" id5 -->
- Atlas 200I/500 A2推理产品：不支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- Atlas推理系列产品AI Core：不支持
- Atlas推理系列产品Vector Core：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- Atlas训练系列产品：不支持
<!-- end id7 -->
<!-- npu="x90" id1 -->
- Kirin X90处理器系列产品：支持
<!-- end id1 -->
<!-- @ref: asc-devkit/res/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetWorkspace_res.md#id1 -->

## 功能说明

[Iterate](Iterate.md)计算的[异步场景](GetTensorC.md#li17508136205415)，调用本接口申请一块临时空间来缓存计算结果，然后调用[GetTensorC](GetTensorC.md)时会在该临时空间中获取C的矩阵分片。

[IterateNBatch](IterateNBatch.md)计算时，调用本接口申请一块临时空间来缓存计算结果，然后根据[同步或异步场景](IterateNBatch.md)进行其它接口的调用。

## 函数原型

建议用户使用GlobalTensor类型传入：

```
template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr)
```

```
template <class T> __aicore__ inline void SetWorkspace(__gm__ const T* addr, int size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| addr | 输入 | 用户传入的GM上的workspace空间，GlobalTensor类型。 |
| addr | 输入 | 用户传入的GM上的workspace空间，GM地址类型。 |
| size | 输入 | 传入GM地址时，需要配合传入元素个数。 |

## 返回值说明

无

## 约束说明

当开启MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](MatmulConfig.md#matmulconfig-params)设置为true，不支持使用该接口。

## 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetWorkspace(workspaceGM);    //设置异步时使用的临时空间
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(biasGlobal);
}
mm.template Iterate<false>();
for (int i = 0; i < singleCoreM/baseM * singleCoreN/baseN; ++i) {
    mm.template GetTensorC<false>(ub_c);
}
```
`````

https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/basic_api/resource_management/workspace/GetUserWorkspace.md
author/date: unknown — repo cann/asc-devkit branch master, fetched 2026-10-20
GetUserWorkspace.md, the functional-description half verbatim (the parameter table is raw HTML boilerplate and is elided; the full file is 3115 bytes).
`````markdown
# GetUserWorkspace<a name="ZH-CN_TOPIC_0000001666932721"></a>

## 产品支持情况<a name="section1550532418810"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2推理产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品AI Core：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- Atlas推理系列产品Vector Core：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- Atlas训练系列产品：支持
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90处理器系列产品：支持
<!-- end id8 -->
<!-- npu="9030" id9 -->
- Kirin 9030处理器系列产品：支持
<!-- end id9 -->
<!-- @ref: asc-devkit/res/docs/zh/api/SIMD-API/basic_api/resource_management/workspace/GetUserWorkspace_res.md#id1 -->


## 功能说明<a name="section618mcpsimp"></a>

获取用户使用的workspace指针。核函数（Kernel）直调开发方式下，如果未开启HAVE\_WORKSPACE编译选项，框架不会自动设置系统workspace。如果使用了[Matmul核函数（Kernel）侧接口](../../../adv_api/cube_compute/Matmul_Kernel/Matmul_Kernel.md)等需要系统workspace的高阶API，核函数（Kernel）侧需要通过[SetSysWorkSpace](SetSysWorkSpace.md)设置系统workspace，此时用户workspace需要通过该接口获取。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
```

[... HTML parameter table elided ...]

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

用户使用workspace指针。
`````

---

### B1.9 Everything else under `05_best_practices` (names only, as requested)

Full path prefix: `examples/01_simd_cpp_api/05_best_practices/`

- `00_vector_compute/`
  - `add_high_performance`
- `01_matrix_compute/`
  - `matmul_basic_api_high_performance`  (`mmad.asc`)
  - `matmul_hif8_high_performance`  (`matmul_hif8.asc`)
  - `matmul_high_performance`  (`matmul.asc` + `matmul.h`)
  - `matmul_mxfp4_basic_api_high_performance`  (`mmad_mx.asc`)
  - `matmul_mxfp4_high_performance`  (`matmul_mx.asc` + `matmul_mx.h`)
- `02_reg_compute/`
  - `gelu_eltwise_high_performance`
  - `gelu_high_performance`
  - `softmax_high_performance`
- `03_fusion_compute/`
  - `matmul_gelu_high_performance`  (`mmad_gelu.asc`)
  - `quant_group_matmul_high_performance`  (`quant_group_matmul_custom.asc`)  <- the target
  - `ssbuf_aiv_aic_comm`  (`ssbuf_aiv_aic_comm.asc`, 950PR only)
- `04_memory_access/`
  - `bank_conflict_3510`
  - `bank_conflict_nd2nz`
  - `bank_conflict_ub`
  - `data_copy`

Adjacent and likely useful for a megakernel, under
`examples/01_simd_cpp_api/04_advanced_api/00_matmul/` (41 samples, names only):
`batch_matmul`, `batch_matmul_bias_reuse`, `batch_matmul_iterate_n_batch`, `matmul`,
`matmul_a2b2_share`, `matmul_async_iterate`, `matmul_async_iterate_all`, `matmul_callback`,
`matmul_channelsplit_output`, `matmul_co1_output`, `matmul_constant_tiling`,
`matmul_format_column_major`, `matmul_format_gemv`, `matmul_format_nd_align`, `matmul_fp8`,
`matmul_fused`, `matmul_fused_manual`, `matmul_ibshareAB`, `matmul_ibshareB`, `matmul_int4`,
`matmul_k_reorder_load`, `matmul_l0cache`, `matmul_l2cache`, `matmul_mixdualmaster`,
`matmul_mn_double_buffer`, `matmul_multi_core_unaligned`, `matmul_mx`, `matmul_mx_scale_cache`,
`matmul_mx_ub_tscm_nz`, `matmul_nbuffer33`, `matmul_partial_output`, `matmul_preload`,
`matmul_quant`, `matmul_sparse`, `matmul_splitk`, `matmul_splitm`, `matmul_triangle`,
`matmul_tscm`, `matmul_tscm_src_vecout`, `matmul_unitflag`, `matmul_vecout`.

---

### B1.10 Bridge to TOPIC A — the credit-based flag discipline in this kernel (independently re-verified)

Re-verified by me directly against the downloaded
`examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/quant_group_matmul_custom.asc`
with `grep -n 'PIPELINE_DEPTH\|SYNC_AIC_TO_AIV\|SYNC_AIV_TO_AIC\|CrossCore\|cubeTaskIdx\|vecTaskIdx\|workSpaceOffset'`.
This is the same question as Topic A, answered inside a grouped-matmul kernel: how do you drive
an **unbounded** number of CV handoffs (one per `Iterate`, over all 8 — or 256 — groups) with a
flag counter that saturates at 15?

The answer in this source is a credit window of depth `PIPELINE_DEPTH`, not flagId rotation.
Exactly two flagIds are used for the whole kernel:

```cpp
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint64_t SYNC_AIV_TO_AIC = 3;
```

```cpp
//   0: __mix__(1,1) + PIPELINE_DEPTH=1  — 基准: 单AIV, 无流水重叠
//   1: __mix__(1,2) + PIPELINE_DEPTH=1  — 双AIV, 无流水重叠
//   2: __mix__(1,2) + PIPELINE_DEPTH=4  — 双AIV, 4级流水
```

```cpp
    uint32_t cubeTaskIdx = 0; // AIC侧: 按Iterate粒度递增, 用于环形workspace Slot分配
    uint32_t vecTaskIdx = 0;  // AIV侧: 按offsetN粒度递增, 与cubeTaskIdx天然同步
```

Producer (AIC). The `if (cubeTaskIdx >= PIPELINE_DEPTH)` guard is the credit check: it holds the
number of outstanding `SYNC_AIC_TO_AIV` Sets at no more than `PIPELINE_DEPTH`, so with
`PIPELINE_DEPTH = 4` the counter never approaches 15 however many groups there are:

```cpp
            if (cubeTaskIdx >= PIPELINE_DEPTH) {
                AscendC::CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC);
            }
            uint64_t workSpaceOffset = BASE_N * BASE_M * (coreIdx + (cubeTaskIdx % PIPELINE_DEPTH) * CORE_NUM);
            matmulObj.GetTensorC(mmOutGlobal[workSpaceOffset], 0, true);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
            cubeTaskIdx++;
```

Consumer (AIV), using the identical ring formula with its own counter:

```cpp
        uint64_t mmOutOffset = BASE_N * BASE_M * (coreIdx + (vecTaskIdx % PIPELINE_DEPTH) * CORE_NUM);
```

```cpp
        AscendC::CrossCoreWaitFlag<2>(SYNC_AIC_TO_AIV);
```

```cpp
        AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);
        vecTaskIdx++;
```

The epilogue drains exactly the credits still outstanding — the same "drain the handshake"
discipline as `mega_kernel.cpp`'s stage boundaries in A2:

```cpp
        if (likely(cubeTaskIdx > 0)) {
            uint32_t loop = cubeTaskIdx < PIPELINE_DEPTH ? cubeTaskIdx : PIPELINE_DEPTH;
```

```cpp
                AscendC::CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC);
```

And the host-side workspace must be sized to the same ring depth, which is how the device ring
index and the host allocation stay consistent:

```cpp
    constexpr uint32_t PIPELINE_DEPTH = (scenarioNum >= 2) ? 4 : 1;
```

```cpp
    constexpr size_t USER_WORKSPACE_SIZE = PIPELINE_DEPTH * BASE_N * BASE_M * sizeof(int32_t) * CORE_NUM;
```

Consequence for a 78-layer / 256-expert hand-written kernel: `PIPELINE_DEPTH` is the only thing
that has to respect the 15 limit. Pick `PIPELINE_DEPTH <= 15` (this sample picks 4) and the number
of experts, groups, layers and `Iterate` calls becomes irrelevant to flag exhaustion.


---

## B2. Megakernel workspace slot arithmetic — host allocation vs device indexing

This is the pattern the two topics share, and it is worth having verbatim because it is the
contract a hand-written megakernel has to keep between host and device.

https://github.com/huawei-csl/megagdn-pto/blob/main/megagdn_pto/mega_kernel.py

Author: Huawei Technologies Co., Ltd. (repo `huawei-csl/megagdn-pto`, branch `main`). Fetched 2026-10-20.

Host side: every stage's scratch is allocated as `block_dim * <slots_per_core>` tiles. `bd` is the
AI-Core block count. `kkt_ws` gets 2 slots per core (the `ci & 1` double buffer), `h_ws` gets 4:

```python
    kkt_ws    = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    wy_ws_a1  = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    wy_ws_a2  = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    h_ws      = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)
    o_ws_qk   = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_ws_qs   = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
    o_ws_gated = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_out     = torch.empty_like(v)
```

Note also that each stage gets its own distinct GM pointer passed as a separate kernel argument —
the megakernel does not carve one monolithic workspace with offsets; it takes 7 workspace
pointers (`kkt_ws`, `wy_ws_a1`, `wy_ws_a2`, `h_ws`, `o_ws_qk`, `o_ws_qs`, `o_ws_gated`) alongside
the inter-stage tensors:

```python
    lib.call_kernel(
        bd, stream,
        _vp(q), _vp(k), _vp(v), _vp(g_in), _vp(beta),
        _vp(msk_lower), _vp(msk_full), _vp(minus_identity), _vp(cu_seqlens),
        _vp(o_out),
        _vp(g_sum), _vp(g_t), _vp(beta_t),
        _vp(A), _vp(A_inv_f32), _vp(A_inv),
        _vp(w), _vp(u), _vp(s), _vp(v_new), _vp(fs),
        _vp(h0), has_initial_state,
        _vp(kkt_ws), _vp(wy_ws_a1), _vp(wy_ws_a2), _vp(h_ws),
        _vp(o_ws_qk), _vp(o_ws_qs), _vp(o_ws_gated),
        H, kh,
        N_seq, T, T, num_matrices,
    )
```

Device side, kkt stage — `(cid * 2 + slot) * ChunkSquare`, i.e. core-major, slot-minor, which is
exactly the `bd * 2` host shape:

https://github.com/huawei-csl/megagdn-pto/blob/main/kernels/pto/scaled_dot_kkt.cpp

```cpp
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_handle +
                (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare,
            _gs);
        TSTORE(_gm, _l0);
```

Device side, chunk_h stage — four *named* slots per core (not a ring), with a per-core base and
a further `vid` (Vec sub-block) sub-offset:

https://github.com/huawei-csl/megagdn-pto/blob/main/kernels/pto/chunk_h.cpp

```cpp
  constexpr int32_t WS_WS = 0;
  constexpr int32_t WS_K = DD;
  constexpr int32_t WS_S = DD * 2;
  constexpr int32_t WS_KV = DD * 3;
  constexpr int32_t WS_PER_CORE = DD * 4;
```

```cpp
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;
```

```cpp
        GmTensor2D<half> s_global(workspace_handle + ws_base + WS_S, s_shape,
```

```cpp
          workspace_handle + ws_base + WS_S + vid * HalfC * D, s_shape,
```

So two distinct idioms coexist in the same megakernel: a depth-2 ring keyed on a loop counter
(kkt, flags 0/1 vs 2/3), and a fixed set of named per-core buffers (chunk_h, flags 0/1/2/3 used
as four distinct producer-consumer edges rather than as a ring).


---

## B3. ★★★ MegaMoE (Huawei) — `hicann/ops-transformer`, the REAL per-expert GEMM megakernel

This is the headline find. It is Huawei's production MoE megakernel, public, with an A3-specific
device kernel, and it answers TOPIC A and TOPIC B in the same file. Found via
`gh search code 'MegaMoE'`, which surfaced `hicann/ops-transformer:mc2/...`.

Repo: https://github.com/hicann/ops-transformer (branch `master`), listed with
`gh api repos/hicann/ops-transformer/git/trees/master?recursive=1`. Fetched 2026-10-20.
Copyright headers: `Copyright (c) 2026 Huawei Technologies Co., Ltd.`, CANN Open Software
License Agreement Version 2.0.

### B3.0 The MegaMoE file tree (sizes in bytes)

```
mc2/mega_moe/CMakeLists.txt                                                      1230
mc2/mega_moe/README.md                                                          64734
mc2/mega_moe/docs/torchapi_mega_moe.md                                         109678
mc2/mega_moe/op_api/aclnn_mega_moe.cpp                                           9154
mc2/mega_moe/op_graph/CMakeLists.txt                                              992
mc2/mega_moe/op_host/CMakeLists.txt                                              1939
mc2/mega_moe/op_host/mega_moe_def.cpp                                           23569
mc2/mega_moe/op_host/mega_moe_infershape.cpp                                     3974
mc2/mega_moe/op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp                92919
mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp                      180931
mc2/mega_moe/op_host/op_tiling/mega_moe_tiling_host.h                            2825
mc2/mega_moe/op_kernel/arch22/mega_moe.cpp                                       4536
mc2/mega_moe/op_kernel/arch22/mega_moe_a2.h                                     23968
mc2/mega_moe/op_kernel/arch22/mega_moe_arch22.h                                 16551
mc2/mega_moe/op_kernel/arch22/mega_moe_exception_dump_policy.h                   3140
mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a2_bf16.hpp                       92559
mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a2_bf16_w4a8.hpp                  99332
mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a2_int8.hpp                       94688
mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp                           116061   <-- the A3 kernel
mc2/mega_moe/op_kernel/arch22/mega_moe_tiling_a2a3.h                             2682
mc2/mega_moe/op_kernel/arch22/mega_moe_tiling_key.h                              2916
mc2/mega_moe/op_kernel/arch22/moe_permute_prologue/moe_permute_prologue.hpp      21847
mc2/mega_moe/op_kernel/arch22/moe_permute_prologue/moe_permute_prologue_tiling.h 2303
mc2/mega_moe/op_kernel/arch22/unpermute/mc2_mega_moe_moe_token_unpermute.h       17239
mc2/mega_moe/op_kernel/arch22/unpermute/mc2_mega_moe_moe_token_unpermute_tiling.h 2231
mc2/mega_moe/op_kernel/arch22/utils/block_epilogue_pertoken_row.hpp               9061
mc2/mega_moe/op_kernel/arch22/utils/block_epilogue_pertoken_swiglu.hpp           24536
mc2/mega_moe/op_kernel/arch22/utils/block_epilogue_pertoken_v2.hpp               14605
mc2/mega_moe/op_kernel/arch22/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp  28416
mc2/mega_moe/op_kernel/arch22/utils/block_epilogue_w4a8post_pertoken_v2.hpp      16735
mc2/mega_moe/op_kernel/arch22/utils/block_mmad_preload_async_fixpipe_quant.hpp   25250
mc2/mega_moe/op_kernel/arch22/utils/block_mmad_w4a4.hpp                          25447
mc2/mega_moe/op_kernel/arch22/utils/const_args.hpp                               3046
mc2/mega_moe/op_kernel/arch22/utils/copy_gm_to_l1_custom.hpp                     2054
mc2/mega_moe/op_kernel/arch22/utils/copy_l0c_to_gm_custom.hpp                    2523
mc2/mega_moe/op_kernel/arch22/utils/dispatch_policy_custom.hpp                   8164
mc2/mega_moe/op_kernel/arch22/utils/gated_activation.hpp                        11467
mc2/mega_moe/op_kernel/arch22/utils/get_tensor_addr.hpp                          1378
mc2/mega_moe/op_kernel/arch22/utils/hccl_shmem.hpp                              33496
mc2/mega_moe/op_kernel/arch22/utils/layout3d.hpp                                 1219
mc2/mega_moe/op_kernel/arch22/utils/select_helper.hpp                            1377
mc2/mega_moe/op_kernel/arch22/utils/weight_l2_prefetch.hpp                        5501
mc2/mega_moe/op_kernel/arch22/utilsA2/block_epilogue_pertoken_row_a2_bf16.hpp     8884
mc2/mega_moe/op_kernel/arch22/utilsA2/block_epilogue_pertoken_row_a2_int8.hpp     8907
mc2/mega_moe/op_kernel/arch22/utilsA2/block_epilogue_pertoken_swiglu_a2_bf16.hpp 23466
mc2/mega_moe/op_kernel/arch22/utilsA2/block_epilogue_pertoken_swiglu_a2_int8.hpp 21803
```

A mirror of the same tree also exists at `yjyang62/CANN:mc2/mega_moe/...` with an `arch35`
variant (`mc2/mega_moe/op_kernel/arch35/mega_moe.h`); the analysis-tool script
`hicann/ops-transformer:mc2/tools/dump_analysis/dump_analysis_megamoe.py` is also public.

### B3.1 ★★★ THE flagId ALLOCATION PLAN, verbatim — the answer to TOPIC A

https://raw.githubusercontent.com/hicann/ops-transformer/master/mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp

Author: Huawei Technologies Co., Ltd., 2026. Repo `hicann/ops-transformer`, branch `master`. Fetched 2026-10-20.

Lines 50-72. This comment is the complete flagId budget of a production A3 megakernel. Read it
twice: `syncgmm 流 0-8`, `C2V=9`, `V2C=10`, and `SyncAll<false>` is noted as occupying `11-13`.
It also states the 4-bit truncation rule explicitly:

```cpp
#include "utils/get_tensor_addr.hpp"
#include "utils/weight_l2_prefetch.hpp"
#include "mega_moe_exception_dump_policy.h"

namespace Catlass::Gemm::Kernel {

using namespace AscendC;
using namespace Mc2Tiling;

constexpr uint16_t SYNCFLAGC2V = 9;
constexpr uint16_t SYNCFLAGV2C = 10;
constexpr int32_t UB_MOVE_NUM = 16 * 1024;

// 发送侧逐 chunk 处理的 chunk 大小（= moe_permute_prologue 的 MAX_TOKENS=1024）
constexpr int64_t PERMUTE_CHUNK = 1024;
// 相邻 chunk 通过 SyncAll<false> 全核屏障串行化，不同时执行，因此 C2V/V2C 固定占用 flag 9/10，
// 每 chunk 复用同一组 flag（syncgmm 流 0-8、C2V=9、V2C=10、GMM2-combine 0-8）。
// 注意 mode0/1/2 下 flagId 仅 0-15，超出截取低 4bit；若改为按 chunk 偏移必须满足
// C2V(chunk n) != V2C(chunk m) 且不落入 syncgmm 流 0-8 与 SyncAll<false> 的 11-13。
constexpr uint16_t PERMUTE_CHUNK_FLAG_STRIDE = 0;
// prologue 每核计数区按 AIV 核数上限分配（host 侧 workspace 须按实际 aivNum 分配且 <= 此值）
constexpr int64_t PERMUTE_MAX_CORES = 128;

// Atlas A3 (910_93) kernel.
```

And the constant it all hangs on:

https://raw.githubusercontent.com/hicann/ops-transformer/master/mc2/mega_moe/op_kernel/arch22/utils/const_args.hpp

```cpp
constexpr uint16_t CROSS_CORE_FLAG_MAX_SET_COUNT = 15;
constexpr uint32_t MAX_EXPERTS_PER_RANK = 128;
constexpr uint32_t MAX_RANK_PER_CORE = 32;
constexpr static uint16_t SYNCFLAGC2V = 9;
constexpr static uint16_t SYNCFLAGV2C = 10;
```

### B3.2 ★★★ THE ROTATION ITSELF — `flagId = counter / 15`

Every per-expert handshake in this kernel computes its flagId by integer-dividing a
monotonically increasing counter by `CROSS_CORE_FLAG_MAX_SET_COUNT`. Each flagId absorbs
exactly 15 Sets, then the next flagId takes over. With 128 experts per rank, `128 / 15 = 8`,
which is precisely why the comment reserves the "syncgmm 流" as flagIds **0-8**.

AIV side (producer) — note the Set and the counter increment are adjacent, so there is never a
second Set of the same flagId without an intervening increment past the /15 boundary:

```cpp
        // 轮就绪信号：rounds==1 时即 chunk cumsum/路由就绪信号，rounds>1 时为轮表就绪信号
        uint16_t syncgmm1Idx = 0;
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
        syncgmm1Idx++;
```

```cpp
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmm1Idx++;
```

AIC side (consumer), inside the grouped-matmul loop. The `for (; syncGroupIdx <= groupIdx; ...)`
catch-up loop is how a core that skipped empty experts still drains the right number of credits:

```cpp
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                for (; syncGroupIdx <= groupIdx; syncGroupIdx++) {
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmmIdx++;
                }
```

The epilogue drain, so the counters end balanced before the next round:

```cpp
        for (; syncGroupIdx < params.expertPerRank; syncGroupIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx++;
        }
```

Two more independent rotations in the combine path, using the expert index directly:

```cpp
        for (uint32_t t_groupIdx = 0; t_groupIdx < params.expertPerRank; ++t_groupIdx) {
            int32_t flagId = t_groupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
            AscendC::CrossCoreWaitFlag<0x2>(flagId);
            AscendC::SyncAll<true>();
```

```cpp
                for (; syncLoopIdx <= groupIdx; syncLoopIdx++) {
                    int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                    AscendC::CrossCoreWaitFlag<0x2>(flag_id);
                }
```

```cpp
        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            for (; syncLoopIdx < params.expertPerRank; syncLoopIdx++) {
                int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                AscendC::CrossCoreWaitFlag<0x2>(flag_id);
            }
        }
```

### B3.3 ★ THE PER-ROUND COUNTER RESET — how the rotation restarts without exhausting flagIds

The outer AIC loop over chunks and receive-rounds. `syncgmmIdx = 0;` at the bottom of the round
is the reset: the flagId stream restarts at 0 every round, which is legal because the
barrier/`SyncAll` has drained everything. The Chinese comments state the invariant directly —
`syncgmmIdx/syncgmm1Idx 按轮归零，flag 槽位按轮复用` ("zeroed per round, flag slots reused per round"):

```cpp
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        if (IsRankMasked(RuntimeRank(params))) {
            return;
        }
        syncgmmIdx = 0;
        const int64_t bs = params.problemShape.m();
        const int64_t numChunks = NumChunks(bs);
        for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
            // cumsumMM 每 chunk 指向独立缓冲区（不复用，规避跨 chunk cache 陈旧读）；
            // chunkCumsumMM 保持 chunk 级指向，供轮数推导与轮表生成读取。
            GM_ADDR chunkCumsumPtr = ChunkCumsumPtr(params, chunkIdx);
            chunkCumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkCumsumPtr));
            cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkCumsumPtr));
            // 接收端轮次切分：R ≤ B 时单轮（退化零回归）；R > B 时每轮等待 AIV 轮表就绪后
            // 切换 cumsumMM 到轮表。syncgmmIdx/syncgmm1Idx 按轮归零，flag 槽位按轮复用。
            int64_t numRecvRounds = 1;
            for (int64_t roundIdx = 0;; ++roundIdx) {
                // 原 GMM1 入口的初始等待外移到轮级：rounds==1 时即 chunk cumsum/路由就绪信号，
                // rounds>1 时为 BuildRoundTables 完成后的轮表就绪信号
                AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                syncgmmIdx++;
```

```cpp
                if (roundIdx == 0) {
                    numRecvRounds = ComputeNumRecvRounds(params);
                }
                if (numRecvRounds > 1) {
                    SetRoundTableBuffers(params, chunkIdx, roundIdx);
                }
                GMM1(params, chunkIdx);
                if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                    AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                    GMM2(params, chunkIdx);
                } else {
                    GMM2(params, chunkIdx);
                }
                syncgmmIdx = 0; // 轮间从 flag 0 复用 syncgmm 握手（与 AIV syncgmm1Idx 对称）
                if (roundIdx + 1 >= numRecvRounds) {
                    break;
                }
            }
        }
    }
```

### B3.4 ★★★ TOPIC B — the per-expert GEMM loop, verbatim (GMM1)

https://raw.githubusercontent.com/hicann/ops-transformer/master/mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp

This is the whole `GMM1` method: the group loop, the group-list read, the per-expert weight
re-pointing, the per-group problem shape, the scheduler update, and the core-to-group mapping.
Everything TOPIC B asked for:

```cpp
    CATLASS_DEVICE
    void GMM1(Params const& params, int64_t chunkIdx)
    {
        icache_preload(ICACHE_PRELOAD_LINES);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        // 说明：原 GMM1 入口的 "等待 AIV 完成 cumsum" 初始等待已外移到 AIC operator() 的
        // 轮循环内（轮表就绪信号），syncgmmIdx 由调用方维护并已指向第一个 per-group 信号。

        // W1 预取激活时 gmB1 恒 NORMAL（DISABLE 双向 bypass 使预取失效）；
        // 未激活（超门槛）恢复 DISABLE（单 M-tile 组权重单次读）；门槛算式与 AicRange 同源。
        const uint64_t w1PrefetchCap = static_cast<uint64_t>(WEIGHT_L2_PREFETCH_COLDSTART_CHUNKS) *
                                       AscendC::GetBlockNum() * WEIGHT_L2_PREFETCH_CHUNK_BYTES;
        const bool w1Prefetched =
            WeightL2PrefetchBytesPerExpert<ElementB>(params.problemShape.k(), params.problemShape.n()) <= w1PrefetchCap;

        AscendC::GlobalTensor<ElementB> gmB1;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentM = currentM * GATE_UP_ROWS_PER_TOKEN;
            }

            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(
                GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB1))));
            gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale1))));

            if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            if (!w1Prefetched && currentM <= L1TileShape::M) {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB1 = params.layoutB1;
            LayoutScale layoutScale = params.layoutScale1;
            LayoutC layoutC(inGroupProblemShape.m(), inGroupProblemShape.n(), params.gmmOutPreRowStride);
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                for (; syncGroupIdx <= groupIdx; syncGroupIdx++) {
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmmIdx++;
                }
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB1.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                if (currentM > 0) {
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                            (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape, Callback{}, Callback{});
                        } else {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape);
                        }
                    } else if constexpr (std::is_same_v<ElementA, int8_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                            (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                        blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                                  gmC[gmGroupOffsetC + gmOffsetC], layoutC, gmS[gmOffsetS], layoutScale,
                                  actualBlockShape);
                    } else {
                        blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                                  gmC[gmGroupOffsetC + gmOffsetC], layoutC, gmS, layoutScale, actualBlockShape);
                    }
                }
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                if (IsSyncTask(groupIdx, params.expertPerRank)) {
                    syncLoopIdx++;
                    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                        blockMmad.SynchronizeBlock();
                    }
                    blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                }
            } else {
                if ((groupIdx + 1) == params.epilogueGranularity && (groupIdx < params.expertPerRank - 1)) {
                    syncLoopIdx++;
                    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                        blockMmad.SynchronizeBlock();
                    }
                    blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                }
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                preCurrentmSum += currentM / GATE_UP_ROWS_PER_TOKEN;
            } else {
                preCurrentmSum += currentM;
            }
            // 每个专家之间在k小于k2的时候不是密排，所以这里不取k
            gmGroupOffsetA += inGroupProblemShape.m() * params.gmm1InputExpertSlotWidth;
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * params.gmmOutPreRowStride;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        for (; syncGroupIdx < params.expertPerRank; syncGroupIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx++;
        }

        if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }
            blockMmad.Finalize(syncLoopIdx + 1, SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
        }
    }
```

### B3.5 ★★ Per-expert weight addressing on device — `GetTensorAddr`, the whole file

https://raw.githubusercontent.com/hicann/ops-transformer/master/mc2/mega_moe/op_kernel/arch22/utils/get_tensor_addr.hpp

Author: Huawei Technologies Co., Ltd., 2026. Fetched 2026-10-20.

This is how a 256-expert weight *list* is addressed from device code: the kernel receives a
single `GM_ADDR` that points at a TensorList descriptor — a `uint64_t` offset followed by an
array of `uint64_t` device pointers — and indexes it per expert. Whole file:

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file get_tensor_addr.hpp
 * \brief
 */

#ifndef GET_TENSOR_ADDR_HPP
#define GET_TENSOR_ADDR_HPP
#include "kernel_operator.h"

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

template <typename T>
FORCE_INLINE_AICORE __gm__ T* GetTensorAddr(uint32_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

#endif // GET_TENSOR_ADDR_HPP
```

The two addressing modes, isolated. `params.listLen` selects between them:

```cpp
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(
                GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB1))));
```

```cpp
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
```

That is: `listLen == 1` means one contiguous `[E, K, N]` weight tensor, advanced by `K*N` per
expert; `listLen != 1` means a TensorList of `E` separate allocations, re-pointed per expert via
`GetTensorAddr`. The per-expert scale is re-pointed the same way, and in the contiguous case its
offset carries the extra `groupIdx * N` term:

```cpp
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                            (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
```

### B3.6 The GMM2 loop — same shape, second projection

```cpp
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;
```

```cpp
                GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB2))));
```

```cpp
                GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale2))));
```

```cpp
            GemmCoord inGroupProblemShape{currentM, n2, k2}; // M N K
```

```cpp
            LayoutA layoutA = params.layoutA2.GetTileLayout(inGroupProblemShape.GetCoordMK());
```

```cpp
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());
```

```cpp
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
```

```cpp
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
```

Note `n2 = problemShape.k()` and `k2 = problemShape.n() / 2` — the `/2` is the SwiGLU gate+up
split: GMM1 produces `2 * intermediate` columns, the epilogue gates them down to
`intermediate`, and GMM2 projects `intermediate -> hidden`.

### B3.7 The group-list read — note it is a `cumsum` table, indexed at the LAST rank

```cpp
            uint32_t currentM = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + groupIdx);
```

The tensor is named `cumsumMM` and is laid out `[EP][expertPerRank]`; row `EP-1` is read, which
gives the count for that expert summed over all `EP` ranks. So what the GEMM loop consumes is a
per-expert **row count**, obtained by indexing the last row of a table that is cumulative along
the *rank* axis — not a prefix sum along the *expert* axis. The workspace sizing confirms the
shape:

```cpp
                EP * AlignUp(EP * MAX_EXPERTS_PER_RANK + 1, ALIGN_128) * static_cast<int64_t>(sizeof(int32_t));
```

Each chunk gets its OWN cumsum buffer, explicitly to avoid stale cache reads across chunks:

```cpp
            // cumsumMM 每 chunk 指向独立缓冲区（不复用，规避跨 chunk cache 陈旧读）；
            // chunkCumsumMM 保持 chunk 级指向，供轮数推导与轮表生成读取。
            GM_ADDR chunkCumsumPtr = ChunkCumsumPtr(params, chunkIdx);
            chunkCumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkCumsumPtr));
            cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkCumsumPtr));
```

### B3.8 The template head — what a grouped-matmul megakernel is parameterised on

```cpp
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
```

`ElementGroupList_` is a first-class template parameter of the kernel: the group-list element
type is part of the kernel's type, alongside the mmad block, the scheduler, and the two
epilogues. The per-group fields of `Params`:

```cpp
        GemmCoord problemShape;
```

```cpp
        GM_ADDR expertIdx;
```

```cpp
        Params(GemmCoord problemShape_, uint32_t EP_, uint32_t listLen_, uint32_t expertPerRank_,
```

```cpp
            gmmOutPreRowStride = problemShape.n() > problemShape.k() ? problemShape.n() : problemShape.k();
            uint32_t k2 = problemShape.n() / 2;
            gmm1InputExpertSlotWidth = problemShape.k() > k2 ? problemShape.k() : k2;
```


---

## B4. ★★ AUTHORITATIVE `groupListType` semantics — official `GroupedMatmulV4` doc

https://github.com/hicann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md
(raw: https://raw.githubusercontent.com/hicann/ops-transformer/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md)

Author: Huawei Technologies Co., Ltd. Repo `hicann/ops-transformer`, branch `master`, 67561 bytes.
Fetched 2026-10-20.

This settles the cumsum-vs-count question, which the asc-devkit sample (B1) and the MegaMoE
kernel (B3) each answer differently for themselves. The official op attribute has THREE modes,
and the doc spells out all three in one sentence. Note `groupListType == 2`: an `[E, 2]` array of
`(groupIdx, groupSize)` pairs with non-zero groups moved to the front — designed exactly for the
case of many inactive experts:

```
    - groupListType为0时要求groupListOptional中数值为非负单调非递减数列，表示分组轴大小的cumsum结果（累积和），groupListType为1时要求groupListOptional中数值为非负数列，表示分组轴上每组大小，groupListType为2时要求groupListOptional中数值为非负数列，shape为[E, 2]，E表示Group大小，数据排布为[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]，其中groupSize为分组轴上每组大小，详见[groupListOptional配置示例](#grouplistoptional配置示例)。
    - groupType代表需要分组的轴，如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，2：k轴分组。详细参考<a href="#groupType-constraints">groupType支持场景</a>约束。
```

The parameter table entries for the three attributes that control grouping:

```
    <tr>
      <td>splitItem（int64_t）</td>
      <td>输入</td>
      <td>整数型参数，代表输出是否要做tensor切分。</td>
      <td>0/1代表输出为多tensor；2/3代表输出为单tensor。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType（int64_t）</td>
      <td>输入</td>
      <td>整数型参数，代表需要分组的轴。</td>
      <td>枚举值-1、0、2。如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，2：k轴分组。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType（int64_t）</td>
      <td>输入</td>
      <td>-</td>
      <td>枚举值0、1、2。综合约束请参见<a href="#约束说明">约束说明。</a></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
```

The worked example that makes the three encodings unambiguous — the SAME routing (9 groups with
sizes 123, 0, 333, 0, 0, 333, 0, 0, 0) written three ways. This is the single most useful quote
for driving 256 routed experts, because mode 2 is the sparse/compacted form:

```
      X的shape是[[789, 4096]]
      W的shape是[[9, 4096, 7168]]
      Y的shape是[[789, 7168]]

    - groupListType为0时groupList配置如下
      - groupListOptional：`[123, 123, 456, 456, 456, 789, 789, 789, 789]`

    - groupListType为1时groupList配置如下
      - groupListOptional：`[123, 0, 333, 0, 0, 333, 0, 0, 0]`

    - groupListType为2时groupList配置如下
      - groupListOptional在该模式会将所有非0的group移动到前面，适用于非激活专家较多场景。
      - groupListOptional：`[[0, 123], [2, 333], [5, 333], [1, 0], [3, 0], [4, 0], [6, 0], [7, 0], [8, 0]]`
```

The 950PR/950DT constraint block, which restates the rules and adds that mode 2 is full-quant +
m-axis only, and that zero-size groups go to the tail:

```
    - groupListType：支持取值0、1、2。
      - 当groupListType为0时，groupListOptional必须为非负单调非递减数列；
      - 当groupListType为1时，groupListOptional必须为非负数列。
      - 仅全量化且groupType为0场景下支持groupListType为2，此时要求groupListOptional中数值为非负数列，shape为[E, 2]，E表示Group大小，数据排布为[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]，其中groupSize为分组轴上每组大小，此时groupedSize为零的组置于groupList末尾，非零组被前置，详见[groupListOptional配置示例](#grouplistoptional配置示例)。
```

Group-count limits per tensor arrangement — relevant because 256 experts exceeds the 128 limit of
the multi-tensor forms. Only the single-x/single-w/single-y form reaches 25600 (non-quant
FP16/BF16 with groupListType 0/1); the TensorList forms cap at 128:

```
      | groupType | x tensor数 | weight tensor数 | y tensor数 | splitItem| groupListOptional | 转置 | 其余场景限制 |
```

```
      | 0 | 单个|单个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第一维；当groupListType为1时，数值的总和应小于等于x中tensor的第一维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第一维；<br> 3）groupListOptional第1维在非量化FLOAT16/BFLOAT16且groupListType为0/1时最大支持25600，其他场景最大支持1024 |1）x不支持转置；<br> 2）支持weight转置，A8W4与A4W4场景不支持weight转置 |weight中tensor需为3维，x，y中tensor需为2维|
      | 0 | 单个|多个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第一维；当groupListType为1时，数值的总和应小于等于x中tensor的第一维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第一维；<br> 3）groupListOptional第1维最大支持128，即最多支持128个group|1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一 |1）x，weight，y中tensor需为2维；<br> 2）weight中每个tensor的N轴必须相等 |
      | 0 | 多个|多个|单个 | 2/3 | 1）groupListOptional可选；<br> 2）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应；当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应；当groupListType为2时，groupListOptional第二列的数值需与x中tensor的第一维一一对应；<br> 3）groupListOptional第1维最大支持128，即最多支持128个group |1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一|1）x，weight，y中tensor需为2维；<br> 2）weight中每个tensor的N轴必须相等 |
      | 2 | 单个|单个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第二维；当groupListType为1时，数值的总和应小于等于x中tensor的第二维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第二维；<br> 3）groupListOptional第1维最大支持1024，即最多支持1024个group | 1）x必须转置；<br> 2）weight不能转置 |1）x，weight中tensor需为2维，y中tensor需为3维；<br> 2）bias必须传空|
```

One more constraint worth knowing, from the same doc: in the per-group-quant path the op does NOT
validate `groupListType` and silently assumes 1:

```
        - 该场景下仅支持groupListType为1（算子不会检查groupListType的值，会认为groupListType为1），k要求为quantGroupSize的整数倍，且要求k <= 18432。其中quantGroupSize为k方向上pergroup量化长度，当前支持quantGroupSize=256。
```

### B4.1 The GroupedMatmul flagId / buffer constants

https://raw.githubusercontent.com/hicann/ops-transformer/master/gmm/common/cgmct/utils/grouped_matmul_constant.h

Author: Huawei Technologies Co., Ltd., 2026. Repo `hicann/ops-transformer`, branch `master`.
Fetched 2026-10-20. Whole file — note `GMM_FLAG_ID_MAX = 16`, the two reserved cross-core flagIds
(6 and 8), mode 4, and `GMM_MKN_LIST_LEN = 128` (the 128-group ceiling as a compile-time array
length):

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_constant.h
 * \brief
 */
#ifndef UTILS_GROUPED_MATMUL_CONSTANT_H
#define UTILS_GROUPED_MATMUL_CONSTANT_H
namespace Cgmct {
namespace Gemm {
namespace GroupedMatmul {
constexpr uint32_t GMM_BUFFER_NUM = 2;
constexpr uint16_t GMM_FLAG_ID_MAX = 16;
constexpr uint16_t GMM_AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t GMM_AIC_SYNC_AIV_FLAG = 8;
constexpr int32_t GMM_CUBE_SYNC_MTE1_FLAG = 3;
constexpr uint8_t GMM_AIC_SYNC_AIV_MODE = 4;
constexpr uint64_t GMM_MAX_STEP_SCALEA_K = 16;
constexpr uint32_t GMM_UB_ALIGN_SIZE = 32;
constexpr uint32_t GMM_MAX_REPEAT_TIMES = 255;

constexpr uint8_t GMM_SPLIT_M = 0;
constexpr uint8_t GMM_SPLIT_K = 2;
constexpr uint64_t GMM_CUBE_BLOCK = 16;
constexpr uint64_t GMM_INNER_AXIS_MIN_SPLIT_VAL = 128; // ND2NZ cacheline 128
constexpr int32_t GMM_MKN_LIST_LEN = 128;              // 128: predefined array legnth

constexpr uint32_t GMM_BMM_BLOCK_NUM = 16;
constexpr uint32_t K0_B8 = 32;
constexpr uint32_t GMM_k0_FLOAT16 = 16;
constexpr uint16_t GMM_DATA_BLOCK = 32;

enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERGROUP_MODE = 0x1U << 4,
    PERBLOCK_MODE = 0x1U << 5,
};
} // namespace GroupedMatmul
} // namespace Gemm
} // namespace Cgmct
#endif
```

### B4.2 The GroupedMatmul source tree in `hicann/ops-transformer` (so the real kernel can be found)

Listed with `gh api repos/hicann/ops-transformer/git/trees/master?recursive=1` on 2026-10-20:

```
gmm/grouped_matmul/CMakeLists.txt                                                  1197
gmm/grouped_matmul/README.md                                                       9184
gmm/grouped_matmul/docs/GroupedMatmulTransFusionPass.md                            2285
gmm/grouped_matmul/docs/GroupedMatmul算子设计介绍.md                                  9938
gmm/grouped_matmul/docs/aclnnGroupedMatmul.md                                     26891
gmm/grouped_matmul/docs/aclnnGroupedMatmulV2.md                                   28718
gmm/grouped_matmul/docs/aclnnGroupedMatmulV3.md                                   32822
gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md                                   67561
gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md                                   56567
gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md                            110713
gmm/grouped_matmul/examples/arch22/test_aclnn_grouped_matmul_a8w8.cpp             10456
gmm/grouped_matmul/examples/arch35/...                                   (many variants)
gmm/grouped_matmul/op_api/aclnn_grouped_matmul.cpp                               222939
gmm/grouped_matmul/op_api/aclnn_grouped_matmul.h                                   4751
gmm/grouped_matmul/op_api/aclnn_grouped_matmul_v2.h                                4994
gmm/grouped_matmul/op_api/aclnn_grouped_matmul_v3.h                                5396
gmm/grouped_matmul/op_api/aclnn_grouped_matmul_v4.h                                7055
gmm/grouped_matmul/op_api/aclnn_grouped_matmul_v5.h                                6715
gmm/grouped_matmul/op_api/aclnn_grouped_matmul_weight_nz.h                         6894
gmm/grouped_matmul/op_api/grouped_matmul.cpp                                       4474
gmm/grouped_matmul/op_api/grouped_matmul.h                                         1637
gmm/grouped_matmul/op_api/grouped_matmul_950_checker.cpp                         110023
gmm/grouped_matmul/op_api/grouped_matmul_950_checker.h                             7563
gmm/common/cgmct/block/block_grouped_matmul_builder.h                              4797
gmm/common/cgmct/utils/grouped_matmul_constant.h                                   2005
experimental/gmm/grouped_matmul_quant/op_kernel/grouped_matmul_quant_w4a16.cpp    35173
experimental/gmm/grouped_matmul_quant/op_kernel/grouped_matmul_quant_w4a16.h       4767
experimental/gmm/gmm_add/op_kernel/grouped_matmul_utils.h                          3046
experimental/gmm/gmm_k_dim/op_kernel/grouped_matmul_utils.h                        3046
experimental/gmm/gmm_local_exp/op_kernel/grouped_matmul_utils.h                    3046
experimental/gmm/gmm_local_exp_with_zero/op_kernel/grouped_matmul_utils.h          3046
examples/fast_kernel_launch_example/csrc/grouped_matmul/ascend910b/op_kernel/grouped_matmul.h      16125
examples/fast_kernel_launch_example/csrc/grouped_matmul/ascend910b/op_kernel/fkl_grouped_matmul_kernel.h 3012
examples/fast_kernel_launch_example/csrc/grouped_matmul/ascend910b/op_kernel/grouped_matmul_tiling.h    12639
examples/fast_kernel_launch_example/csrc/grouped_matmul/ascend910b/grouped_matmul.cpp              13195
experimental/ffn/swiglu_gated_mlp/op_kernel/common/matmul_act/utils/grouped_matmul_constant.h       1669
```

Note `examples/fast_kernel_launch_example/csrc/grouped_matmul/ascend910b/` — a standalone
hand-written grouped-matmul kernel for 910B specifically, outside the op framework. That is the
closest public analogue to a hand-written AscendC grouped matmul.


---

## B5. ★★★ THE DEVICE-SIDE `groupList` DECODER and the per-group matmul re-pointing — official source

This is the most directly reusable code in the whole dossier for Topic B. Two files.

### B5.1 `grouped_matmul_utils.h` — `GetSplitValueFromGroupList`, all three `groupListType` modes on device

https://raw.githubusercontent.com/hicann/ops-transformer/master/gmm/grouped_matmul/op_kernel/grouped_matmul_utils.h

Author: Huawei Technologies Co., Ltd., 2026. Repo `hicann/ops-transformer`, branch `master`,
15042 bytes. Fetched 2026-10-20.

The two constants:

```cpp
constexpr uint32_t GROUP_LIST_TYPE_SPARSE = 2;
constexpr uint32_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1;
```

The decoder. Mode 0 takes the first difference of the cumsum and carries `preOffset` across
iterations; mode 2 reads the second column of the `[E, 2]` array (hence `+1`); mode 1 reads the
value directly. The `#if` guard also tells you which arches it is compiled for:

```cpp
#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ != 310) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) && \
    !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201)
__aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, int32_t& preOffset,
                                                     const GMMBaseParams* __restrict& gmmBaseParams,
                                                     const GlobalTensor<int64_t>& groupListGm)
{
    int32_t splitValue = 0;
    if (likely(gmmBaseParams->groupType != -1)) { // -1: no  need to split
        if (gmmBaseParams->groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else if (gmmBaseParams->groupListType == GROUP_LIST_TYPE_SPARSE) {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx + SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET));
        } else {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        }
    }
    return splitValue;
}
#endif
```

Note: `groupListGm` is `GlobalTensor<int64_t>` and the read is a scalar `GetValue` straight from
GM — there is no `DataCopy` into UB for the group list in this path.

The same `GetTensorAddr` TensorList trick as MegaMoE, in the official grouped matmul (note the
index type is `uint16_t` here vs `uint32_t` in MegaMoE):

```cpp
template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}
```

The matmul config the grouped matmul uses (basic-block sizes M128/N256/K128):

```cpp
__aicore__ static constexpr MatmulConfig GenGmmConf(bool isND2NZ)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = true,
        .basicM = BASIC_BLOCK_SIZE_128,
        .basicN = BASIC_BLOCK_SIZE_256,
        .basicK = BASIC_BLOCK_SIZE_128,
        .intrinsicsCheck = false,
        .isNBatch = false,
        .enVecND2NZ = isND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
```

### B5.2 `grouped_matmul.h` (fast-kernel-launch, 910B) — the hand-written group loop and the per-group matmul re-point

https://raw.githubusercontent.com/hicann/ops-transformer/master/examples/fast_kernel_launch_example/csrc/grouped_matmul/ascend910b/op_kernel/grouped_matmul.h

Author: Huawei Technologies Co., Ltd., 2026. Repo `hicann/ops-transformer`, branch `master`,
16125 bytes. Fetched 2026-10-20. This is a standalone hand-written grouped-matmul kernel for
**910B specifically** (`ascend910b/`), outside the op framework — the closest public analogue to
a hand-written AscendC grouped matmul in a megakernel.

The header, showing it pulls the official utils and tiling-key headers into a hand-written kernel:

```cpp
/*!
 * \file grouped_matmul.h
 * \brief
 */
#ifndef ASCEND_OPS_GROUPED_MATMUL_H
#define ASCEND_OPS_GROUPED_MATMUL_H

// Data type definitions
#define ORIG_DTYPE_X DT_BFLOAT16
#define DTYPE_X bfloat16_t

#define ORIG_DTYPE_WEIGHT DT_BFLOAT16
#define DTYPE_WEIGHT bfloat16_t

#define ORIG_DTYPE_Y DT_BFLOAT16
#define DTYPE_Y bfloat16_t

#define DTYPE_BIAS int
#define DTYPE_SCALE uint64_t
#define DTYPE_ANTIQUANT_SCALE half

#include "gmm/grouped_matmul/op_kernel/grouped_matmul_utils.h"
#include "gmm/grouped_matmul/op_kernel/grouped_matmul_tiling_key.h"

#include "grouped_matmul_tiling.h"

namespace GROUPED_MATMUL {
```

`UpdateMnConfig` — THE per-group offset arithmetic, all five offsets advanced in one place. Note
the NZ-vs-ND branch on the weight stride (`AlignUp<16>(k) * AlignUp<16>(n)` vs `k * n`):

```cpp
template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::UpdateMnConfig(MNConfig& mnConfig)
{
    if constexpr (B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<16>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.scaleIndex++;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}
```

`SetMNConfig` — how `groupType` decides which of M/K the `splitValue` lands on:

```cpp
    if (gmmBaseParams->groupType == 0) {
        mnConfig.m = splitValue;
        mnConfig.k = kListGm[valueIdx];
        mnConfig.n = nListGm[valueIdx];
        return;
    }

    if (gmmBaseParams->groupType == 2) {
        mnConfig.m = mListGm[valueIdx];
        mnConfig.k = splitValue;
        mnConfig.n = nListGm[valueIdx];
        return;
    }

    mnConfig.m = mListGm[groupIdx];
    mnConfig.k = kListGm[groupIdx];
    mnConfig.n = nListGm[groupIdx];
    return;
}
```

★ `Process()` — THE GROUP-LIST LOOP. Everything Topic B asked for: the loop over groups, the
group-list read, the empty-group skip, the per-group block-dim computation, and the
core-to-group work mapping via a running `count` that carries the core assignment across group
boundaries. Note also `AscendC::WaitPreTaskEnd()` / `AscendC::SetNextTaskStart()` bracketing the
whole thing — the SuperKernel inter-op barrier pair (same primitives as A5.6):

```cpp
template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process()
{
    MNConfig mnConfig;
    if (gmmBaseParams->groupType != -1) { // -1: no split
        if (unlikely(groupListPtr == nullptr)) {
            return;
        }
        preOffset = 0;
    }
    AscendC::WaitPreTaskEnd();
    for (uint32_t groupIdx = 0, count = 0; groupIdx < groupNum; ++groupIdx) {
        UpdateMnConfig(mnConfig);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, preOffset, gmmBaseParams, groupListGm);
        SetMNConfig(splitValue, groupIdx, mnConfig);
        if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
            continue;
        }
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
        mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

        uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
        uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + gmmBaseParams->coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * mnConfig.blockDimN;

        while (curBlock < curCount) {
            MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
            computeOp.MMCompute(groupIdx, mnConfig, coreIdx);
            computeOp.VectorCompute(mnConfig);
            curBlock += gmmBaseParams->coreNum;
        }
        count = curCount % gmmBaseParams->coreNum;
    }
    AscendC::SetNextTaskStart();
}
```

The group-list GlobalTensor setup, once in `Init`:

```cpp
    GM_ADDR groupListPtr;
    GlobalTensor<int64_t> groupListGm;
```

```cpp
    groupListPtr = groupList;
    if (groupListPtr != nullptr) {
        groupListGm.SetGlobalBuffer((__gm__ int64_t*)groupList);
```

★ `SetGlobalBufferW` and `MMCompute` — THE PER-GROUP WEIGHT RE-POINTING AND THE MATMUL OBJECT
RE-POINT. This is the exact `SetOrgShape` / `SetSingleShape` / `SetTensorA` / `SetTensorB` /
`IterateAll` sequence Topic B asked for. Note that a FRESH `GlobalTensor<BT>` is constructed per
group, pointed at `weightTensorPtr + wBaseOffset + wOffset`, and handed to `SetTensorB`; the
matmul object `mm` itself is reused, only its shapes and tensors are reset:

```cpp
        wOffset = tailN * (UB_BLOCK_UNIT_SIZE / sizeof(BT)); // 32: quant is 32, float16 is 16
    } else if constexpr (mmType::BT::format == CubeFormat::NZ) {
        wOffset = tailN * AlignUp<16>(k); // 16: nz format last two dim size
    } else if constexpr (transposeW) {
        wOffset = tailN * k;
    } else {
        wOffset = tailN;
    }
    return wOffset;
}

template <typename mmType, bool sync>
__aicore__ inline GlobalTensor<typename mmType::BT::T> GMMCompute<mmType, sync>::SetGlobalBufferW(uint32_t groupIdx,
                                                                                                  uint32_t tailN,
                                                                                                  MNConfig& mnConfig)
{
    uint64_t wOffset = SetWOffset(tailN, mnConfig.k);

    GlobalTensor<BT> weightGmLocal;
    weightGmLocal.SetGlobalBuffer((__gm__ bfloat16_t*)weightTensorPtr + mnConfig.wBaseOffset + wOffset);
    return weightGmLocal;
}

template <typename mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx)
{
    if (subBlockIdx != 0) {
        return;
    }
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM =
        mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    xGm.SetGlobalBuffer((__gm__ bfloat16_t*)xTensorPtr);
    GlobalTensor<BT> weightGmLocal = SetGlobalBufferW(groupIdx, tailN, mnConfig);
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset], false);
    mm.SetTensorB(weightGmLocal, transposeW);

    yGm.SetGlobalBuffer((__gm__ bfloat16_t*)yTensorPtr + mnConfig.yBaseOffset);
    mm.template IterateAll<sync>(yGm[outOffset], 0);
}

} // namespace GROUPED_MATMUL
#endif // ASCEND_OPS_GROUPED_MATMUL_H
```

Reading this against `SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k)`: `SetOrgShape` is given the
**group's** M/N/K (not the whole tensor's), and `SetSingleShape` the current tile's. So the
"original shape" the matmul sees is re-declared per group, and the global position is carried
entirely in the `GlobalTensor` offsets (`xGm[xOffset]`, `weightGmLocal`, `yGm[outOffset]`) with
`mnConfig.{x,w,y}BaseOffset` advanced by `UpdateMnConfig` at the top of each iteration. That is a
materially different idiom from the asc-devkit sample (B1.2), which keeps one `SetGlobalBuffer`
for the lifetime of the kernel and slices only at `SetTensorA`/`SetTensorB`.


### B5.3 The SwiGLU epilogue inside the fused MegaMoE kernel — where the `/2` gate/up split happens

https://raw.githubusercontent.com/hicann/ops-transformer/master/mc2/mega_moe/op_kernel/arch22/utils/block_epilogue_pertoken_swiglu.hpp

Author: Huawei Technologies Co., Ltd., 2026. Repo `hicann/ops-transformer`, branch `master`,
24536 bytes. Fetched 2026-10-20.

The class, specialised on a dispatch-policy tag that names the whole fusion
(`PerTokenDequantSwigluQuant`):

```cpp
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>, CType_,
```

```cpp
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>;
```

The non-quantised `operator()`. Two things to take from it. First, `branchLength =
shapeC.column() / 2` — the gate and up halves are the two halves of GMM1's N, which is why
MegaMoE's GMM2 sets `k2 = params.problemShape.n() / 2` (B3.6). Second, the epilogue is
parallelised over **rows** (`blockM`) across `epilogueCoreNum` cores via `GetTaskRange`, and the
per-expert context is carried entirely in the caller's `gmC` / `gmD` base offsets plus
`gmmOutPreRowStride` — the epilogue itself never sees a `groupIdx`:

```cpp
    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const& gmC, MatrixCoord const& shapeC,
                    AscendC::GlobalTensor<ElementD> const& gmD, Arch::Resource<ArchTag> const& resource,
                    uint32_t epilogueCoreNum = 40, float activationClamp = 0.0f, uint32_t activationCode = 0,
                    float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                    float activationParams2 = SituActivation::DEFAULT_BETA, uint32_t gmmOutPreRowStride = 1,
                    Callback&& callback = Callback{})
    {
        (void)resource;
        activationCode_ = activationCode;
        activationParams1_ = activationParams1;
        activationParams2_ = activationParams2;
        callback();
        uint32_t blockM = shapeC.row();
        uint32_t branchLength = shapeC.column() / 2;
        uint32_t loopStartIdx = 0;
        uint32_t tasksForIdx = 0;
        if (!GetTaskRange(blockM, epilogueCoreNum, loopStartIdx, tasksForIdx)) {
            return;
        }

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * gmmOutPreRowStride];
            auto gmTileD = gmD[loopIdx * branchLength];
            uint32_t tileCount = TileCount(branchLength);
            uint32_t currentStage = 0;
            LoadTile(gmTileC, branchLength, 0, TileLength(branchLength, 0), gmmOutPreRowStride, currentStage);
            for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                uint32_t tileOffset = tileIdx * TILE_LENGTH;
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                uint32_t nextStage = NextStage(currentStage);
                uint32_t nextOffset = tileOffset + TILE_LENGTH;
                if (nextOffset < branchLength) {
                    LoadTile(gmTileC, branchLength, nextOffset, TileLength(branchLength, nextOffset),
                             gmmOutPreRowStride, nextStage);
                }
                ComputeLoadedTile(currentStage, tileLength, 1.0f, false, activationClamp);
                CopyDirectTile(currentStage, gmTileD[tileOffset], tileLength);
                currentStage = nextStage;
            }
```

Note the software-pipelined UB staging: `LoadTile` for tile `i+1` is issued before
`ComputeLoadedTile` for tile `i`, with `currentStage` / `NextStage(currentStage)` rotating over
`UB_STAGES_` buffers. That is the same depth-bounded ring idea as the flag credit windows in
Topic A, applied to UB rather than to flags.

The per-token-dequant overload additionally batches the scale loads, with its own
`SCALE_INPUT_COUNT`-deep batching and an explicit drain:

```cpp
        uint32_t scaleBatchStart = loopStartIdx;
        uint32_t scaleBatchCount = 0;
        uint32_t scaleBatchIndex = 0;
```

```cpp
            uint32_t scaleOffsetInBatch = (loopIdx - loopStartIdx) % SCALE_INPUT_COUNT;
```

```cpp
        DrainScaleBuffers(scaleBatchIndex);
```


---

## B6. `GroupedMatmulSwigluQuant`, CATLASS grouped-matmul templates, `ops-nn`, and MegaMoE design docs

Second breadth pass over Topic B. Verification: 53 fenced blocks, 47 machine-checked as exact
literal substrings of locally cached sources; the 6 that are not are all repo path/tree listings
produced by `gh api .../git/trees/master?recursive=1` or directory walks, each labelled as such
in place.

### B6.0 Repo discovery — where GroupedMatmul actually lives

The task brief guessed `cann/ops-nn`, `hicann/ops-nn`, `cann/cann-ops-adv`, `cann/cann-ops`,
`cann/ops-math`. Result of probing gitcode:

| candidate | status |
| --- | --- |
| `cann/ops-nn` | EXISTS. Walked the whole tree (6254 path entries at depth 2). NO `grouped_matmul` anywhere. Only `norm/group_norm*`, `quant/group_quant`, `quant/grouped_dynamic_*quant`, `activation/swiglu_group*`. Root categories are `activation cmake common control conv docs examples experimental foreach hash index loss matmul norm optim pooling quant rnn scripts tests torch_extension vfusion`; `matmul/` holds 34 ops, none grouped. |
| `hicann/ops-nn` | 404 `Project not found:hicann/ops-nn` |
| `cann/cann-ops-adv` | 404 `Project not found:cann/cann-ops-adv` |
| `cann/cann-ops` | 404 `Project not found:cann/cann-ops` |
| `cann/ops-math` | EXISTS (not the right place for GMM). |
| `cann/catlass` | EXISTS. |
| **`cann/ops-transformer`** | **EXISTS — this is the real home of every GroupedMatmul variant.** |

Real paths found in `cann/ops-transformer` (branch `master`):

```
gmm/grouped_matmul
gmm/grouped_matmul_activation_quant
gmm/grouped_matmul_add
gmm/grouped_matmul_finalize_routing
gmm/grouped_matmul_swiglu_quant
gmm/grouped_matmul_swiglu_quant_v2
gmm/quant_grouped_matmul_dequant
gmm/quant_grouped_matmul_inplace_add
gmm/common
mc2/allto_allv_grouped_mat_mul
mc2/allto_allv_quant_grouped_mat_mul
mc2/grouped_mat_mul_all_reduce
mc2/grouped_mat_mul_allto_allv
mc2/quant_grouped_mat_mul_allto_allv
mc2/mega_moe                      <-- MegaMoE, open source
experimental/gmm
experimental/moe
moe/moe_init_routing{,_v2,_v3,_v4}, moe/moe_re_routing{,_v2}, moe/moe_finalize_routing{,_v2}
moe/moe_compute_expert_tokens, moe/moe_gating_top_k{,_softmax,_softmax_v2}
moe/moe_token_permute*, moe/moe_token_unpermute*
```

`gmm/grouped_matmul/op_kernel` contents:

```
a16w4_msd/                                 (dir)
arch35/                                    (dir)
gmm_infra/                                 (dir)
grouped_matmul.cpp
grouped_matmul_a4w4.h
grouped_matmul_a4w4_interface.cpp
grouped_matmul_a4w4_regular.h
grouped_matmul_antiquant.h
grouped_matmul_antiquant_a16w8_msd.h
grouped_matmul_antiquant_a8w4.h
grouped_matmul_antiquant_a8w4_msd.h
grouped_matmul_antiquant_a8w4_msd_new.h
grouped_matmul_antiquant_a8w4_msd_pre.h
grouped_matmul_antiquant_a8w4_msd_pre_nz.h
grouped_matmul_antiquant_a8w4_pre.h
grouped_matmul_apt.cpp
grouped_matmul_autotiling_a8w4.h
grouped_matmul_fixaxismove_interface.cpp
grouped_matmul_fixaxismove_regular.h
grouped_matmul_kernel.h                    <-- the driver loop
grouped_matmul_pre_tiling.h
grouped_matmul_quant_mixcore.h
grouped_matmul_tiling_key.h
grouped_matmul_utils.h                     <-- GetSplitValueFromGroupList
grouped_matmul_vector.h
```

---

### B6.1 AUTHORITATIVE `group_list_type` SEMANTICS

#### B6.1.1 The one-line op-definition comment (host side)

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/op_host/grouped_matmul_def.cpp
author/date: unknown — Huawei Technologies Co., Ltd., repo cann/ops-transformer branch master, fetched 2026-10-20

```cpp
        this->Attr("split_item").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Int(0); // Reserved parameter, indicate output data type, not enabled.
        this->Attr("transpose_weight")
            .AttrType(OPTIONAL)
            .Bool(false); // Reserved parameter, indicate wether input weight is transposed, not enabled.
        this->Attr("transpose_x")
            .AttrType(OPTIONAL)
            .Bool(false); // Reserved parameter, indicate wether input x is transposed, not enabled.
        this->Attr("group_type").AttrType(OPTIONAL).Int(-1); // Indicates the splited dimension.
        this->Attr("group_list_type")
            .AttrType(OPTIONAL)
            .Int(0); // Indicates whether the value in group_dist is cumsum or count.
        this->Attr("act_type").AttrType(OPTIONAL).Int(0);            // Indicate activation function type.
        this->Attr("tuning_config").AttrType(OPTIONAL).ListInt({0}); // Indicate activation function type.

```

#### B6.1.2 The device-side decoder — the single place the enum is turned into a per-group M

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/op_kernel/grouped_matmul_utils.h
author/date: unknown — Huawei Technologies Co., Ltd., repo cann/ops-transformer branch master, fetched 2026-10-20

Constants first (lines 193-195):

```cpp
constexpr uint32_t STATIC_TILING_MAX_SINGLE_N = 1024;
constexpr uint32_t GROUP_LIST_TYPE_SPARSE = 2;
constexpr uint32_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1;
```

`GetTensorAddr` — how a TensorList input becomes a device pointer (this is how
multi-tensor weight lists are indexed per expert):

```cpp
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ != 310) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) && \
    !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201)
__aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, int32_t& preOffset,
```

And the decoder itself. **This is the authoritative device-side semantics**: `groupListType == 0`
means the value read from GM is a cumulative offset and the per-group size is
`offset - preOffset`; anything else (i.e. `1`) means the value IS the group size; `2` is the
sparse `[groupIdx, groupSize]` pair layout and the size sits at offset `+1`.

```cpp
#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ != 310) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) && \
    !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201)
__aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, int32_t& preOffset,
                                                     const GMMBaseParams* __restrict& gmmBaseParams,
                                                     const GlobalTensor<int64_t>& groupListGm)
{
    int32_t splitValue = 0;
    if (likely(gmmBaseParams->groupType != -1)) { // -1: no  need to split
        if (gmmBaseParams->groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else if (gmmBaseParams->groupListType == GROUP_LIST_TYPE_SPARSE) {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx + SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET));
        } else {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        }
    }
    return splitValue;
}
#endif

template <typename T>
__aicore__ inline constexpr uint32_t GetTypeBits()
{
```

Note the guard: on `__CCE_AICORE__ == 310` and on `__NPU_ARCH__ == 9201` this function is
compiled out entirely (a different arch-specific path is used).

#### B6.1.3 The user-facing documented semantics (aclnnGroupedMatmulV4)

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md
author/date: unknown — Huawei Technologies Co., Ltd., repo cann/ops-transformer branch master, fetched 2026-10-20

The attribute table (lines 282-309):

```
      <td>splitItem（int64_t）</td>
      <td>输入</td>
      <td>整数型参数，代表输出是否要做tensor切分。</td>
      <td>0/1代表输出为多tensor；2/3代表输出为单tensor。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType（int64_t）</td>
      <td>输入</td>
      <td>整数型参数，代表需要分组的轴。</td>
      <td>枚举值-1、0、2。如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，2：k轴分组。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType（int64_t）</td>
      <td>输入</td>
      <td>-</td>
      <td>枚举值0、1、2。综合约束请参见<a href="#约束说明">约束说明。</a></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
```

The constraint paragraph (line 593) — the single most explicit statement of the enum:

```
    - groupListType为0时要求groupListOptional中数值为非负单调非递减数列，表示分组轴大小的cumsum结果（累积和），groupListType为1时要求groupListOptional中数值为非负数列，表示分组轴上每组大小，groupListType为2时要求groupListOptional中数值为非负数列，shape为[E, 2]，E表示Group大小，数据排布为[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]，其中groupSize为分组轴上每组大小，详见[groupListOptional配置示例](#grouplistoptional配置示例)。
    - groupType代表需要分组的轴，如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，2：k轴分组。详细参考<a href="#groupType-constraints">groupType支持场景</a>约束。
```

The worked example that pins all three encodings against the SAME routing (lines 798-814).
This is the thing to copy when you hand-build a groupList for 256 experts:

```
    - shape信息
      M = 789、 K=4096、 N=7168 、E = 9（0,2,5个专家有需要处理的token，0处理123个token， 2/5处理333个token）
      X的shape是[[789, 4096]]
      W的shape是[[9, 4096, 7168]]
      Y的shape是[[789, 7168]]

    - groupListType为0时groupList配置如下
      - groupListOptional：`[123, 123, 456, 456, 456, 789, 789, 789, 789]`

    - groupListType为1时groupList配置如下
      - groupListOptional：`[123, 0, 333, 0, 0, 333, 0, 0, 0]`

    - groupListType为2时groupList配置如下
      - groupListOptional在该模式会将所有非0的group移动到前面，适用于非激活专家较多场景。
      - groupListOptional：`[[0, 123], [2, 333], [5, 333], [1, 0], [3, 0], [4, 0], [6, 0], [7, 0], [8, 0]]`
    </details>

```

Ascend 950PR/950DT公共约束 (lines 824-830):

```
  - 公共约束：

    - groupListType：支持取值0、1、2。
      - 当groupListType为0时，groupListOptional必须为非负单调非递减数列；
      - 当groupListType为1时，groupListOptional必须为非负数列。
      - 仅全量化且groupType为0场景下支持groupListType为2，此时要求groupListOptional中数值为非负数列，shape为[E, 2]，E表示Group大小，数据排布为[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]，其中groupSize为分组轴上每组大小，此时groupedSize为零的组置于groupList末尾，非零组被前置，详见[groupListOptional配置示例](#grouplistoptional配置示例)。
    - actType（int64\_t，计算输入）：整数型参数，代表激活函数类型，取值范围为0-5。
```

#### B6.1.4 Max group count, and the per-arch group-count ceilings

From the same doc, `groupType` support table (lines 782-788). Note the hard numbers:
`groupListOptional` first dim max **25600** for non-quant FLOAT16/BFLOAT16 with
groupListType 0/1 single-single-single, **1024** otherwise, and only **128** when weight is a
multi-tensor list.

```
      | groupType | x tensor数 | weight tensor数 | y tensor数 | splitItem| groupListOptional | 转置 | 其余场景限制 |
      |:---------:|:-------:|:-------:|:-------:|:--------:|:------------------|:--------| :-------|
      | -1 | 多个|多个|多个 | 0/1 | groupListOptional必须传空 | 1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一| x中tensor要求维度一致，支持2维，weight中tensor需为2维，y中tensor维度和x保持一致 |
      | 0 | 单个|单个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第一维；当groupListType为1时，数值的总和应小于等于x中tensor的第一维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第一维；<br> 3）groupListOptional第1维在非量化FLOAT16/BFLOAT16且groupListType为0/1时最大支持25600，其他场景最大支持1024 |1）x不支持转置；<br> 2）支持weight转置，A8W4与A4W4场景不支持weight转置 |weight中tensor需为3维，x，y中tensor需为2维|
      | 0 | 单个|多个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第一维；当groupListType为1时，数值的总和应小于等于x中tensor的第一维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第一维；<br> 3）groupListOptional第1维最大支持128，即最多支持128个group|1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一 |1）x，weight，y中tensor需为2维；<br> 2）weight中每个tensor的N轴必须相等 |
      | 0 | 多个|多个|单个 | 2/3 | 1）groupListOptional可选；<br> 2）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应；当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应；当groupListType为2时，groupListOptional第二列的数值需与x中tensor的第一维一一对应；<br> 3）groupListOptional第1维最大支持128，即最多支持128个group |1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一|1）x，weight，y中tensor需为2维；<br> 2）weight中每个tensor的N轴必须相等 |
      | 2 | 单个|单个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第二维；当groupListType为1时，数值的总和应小于等于x中tensor的第二维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第二维；<br> 3）groupListOptional第1维最大支持1024，即最多支持1024个group | 1）x必须转置；<br> 2）weight不能转置 |1）x，weight中tensor需为2维，y中tensor需为3维；<br> 2）bias必须传空|
```

---

### B6.2 `GroupedMatmul` production driver loop (ops-transformer, A2/A3 path)

This is the `op_kernel/grouped_matmul_kernel.h` template actually shipped in `ops-transformer`
(distinct from the `examples/fast_kernel_launch_example` copy). Quoted for the core→group
mapping and the diagonal block walk, which the examples copy does not have in the same form.

#### B6.2.1 Per-group offset bump — `UpdateMnConfig`

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/op_kernel/grouped_matmul_kernel.h
author/date: unknown — Huawei Technologies Co., Ltd., repo cann/ops-transformer branch master, fetched 2026-10-20

```cpp
    return;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::UpdateMnConfig(MNConfig& mnConfig)
{
    if constexpr (B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset +=
            static_cast<uint64_t>(AlignUp<16>(mnConfig.k)) * AlignUp<16>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += static_cast<uint64_t>(mnConfig.k) * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.scaleIndex++;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += static_cast<uint64_t>(mnConfig.m) * mnConfig.k;
    mnConfig.yBaseOffset += static_cast<uint64_t>(mnConfig.m) * mnConfig.n;
}

template <typename ComputeType>
__aicore__ inline bool GMMProcess<ComputeType>::UpdateMnConfigForGroupListMSparse(MNConfig& mnConfig,
                                                                                  uint32_t splitValue,
                                                                                  uint32_t groupIdx)
{
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += static_cast<uint64_t>(mnConfig.m) * mnConfig.k;
    mnConfig.yBaseOffset += static_cast<uint64_t>(mnConfig.m) * mnConfig.n;
    mnConfig.scaleIndex++;
    SetMNConfig(splitValue, groupIdx, mnConfig);
    if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
        return true; // skip
    }

    mnConfig.nAxisBaseOffset = static_cast<uint64_t>(groupIdx) * mnConfig.n;
    if constexpr (GMMProcess<ComputeType>::B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset = static_cast<uint64_t>(AlignUp<16>(mnConfig.k)) * AlignUp<16>(mnConfig.nAxisBaseOffset);
    } else {
        mnConfig.wBaseOffset = static_cast<uint64_t>(mnConfig.k) * mnConfig.nAxisBaseOffset;
    }
```

#### B6.2.2 `Process()` — the group loop plus the core→block mapping for MANY groups

Key mechanic for 256 experts: `count` is a running modulo-`coreNum` cursor. Each group
contributes `blockDimM * blockDimN` basic blocks to one global block stream; a core picks up
blocks `curBlock += coreNum`, and at group boundaries `count = curCount % coreNum` CARRIES THE
PHASE so the next group starts where the previous one left off. There is no per-group barrier
and no idle core on tiny groups — a group with 0 tokens is simply `continue`d.

```cpp
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process()
{
    MNConfig mnConfig;
    if (gmmBaseParams->groupType != -1) { // -1: no split
        if (unlikely(groupListPtr == nullptr)) {
            return;
        }
        preOffset = 0;
    }
    AscendC::WaitPreTaskEnd();
    // 2: groupList shape: [e, 2]; 1: groupList shape: [e]
    uint32_t groupListType = gmmBaseParams->groupListType;
    uint32_t groupListInnerShape = groupListType == GROUP_LIST_TYPE_SPARSE ? 2 : 1;
    uint32_t groupListShapeSize = groupNum * groupListInnerShape;
    for (uint32_t groupIdx = 0, count = 0; groupIdx < groupListShapeSize; groupIdx += groupListInnerShape) {
        UpdateMnConfig(mnConfig);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, preOffset, gmmBaseParams, groupListGm);
        if (groupListType == GROUP_LIST_TYPE_SPARSE && splitValue <= 0) {
            break;
        }
        SetMNConfig(splitValue, groupIdx, mnConfig);
        if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
            continue;
        }
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
        mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

        uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
        uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + gmmBaseParams->coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * mnConfig.blockDimN;

        while (curBlock < curCount) {
            MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
            computeOp.MMCompute(groupIdx, mnConfig, coreIdx);
            computeOp.VectorCompute(mnConfig);
            curBlock += gmmBaseParams->coreNum;
        }
        count = curCount % gmmBaseParams->coreNum;
    }
    computeOp.PostCompute();
    AscendC::SetNextTaskStart();
}

```

#### B6.2.3 The diagonal (anti-same-address) basic-block index computation

`thresholdBlockNum = 8`, `thresholdDimM = 5` on A2/A3 (`1` on 310):

```cpp
namespace GROUPED_MATMUL {

constexpr uint32_t thresholdBlockNum = 8; // 8 is obtained by tests, indicating the threshold of basic block numbers
                                          // in both directions when assigning data blocks to cube cores when using
                                          // diagnal strategy
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
constexpr uint32_t thresholdDimM = 1; // not needs any special strategies
#else
constexpr uint32_t thresholdDimM = 5; // 5 is obtained by tests, indicating the threshold for distinguishing
                                      // strategies for large/small shapes
#endif
```

```cpp

__aicore__ inline void MNBlockIdxCompute(MNConfig& mnConfig, const uint32_t curBlock, const uint32_t count,
                                         const uint32_t thresholdM_dimN)
{
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM = relativeBlock >= AlignDown(mnConfig.blockDimM * mnConfig.blockDimN, thresholdM_dimN) ?
                                     mnConfig.blockDimM % thresholdBlockNum :
                                     thresholdBlockNum;
        uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
        uint32_t curThresholdN =
            relativeBlock % thresholdM_dimN >= AlignDown(curThresholdM * mnConfig.blockDimN, curThresholdM_thresholdN) ?
                mnConfig.blockDimN % thresholdBlockNum :
                thresholdBlockNum;

        uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
        mnConfig.nIdx = (localRelativeBlock + localRelativeBlock / LeastCommonMultiple(curThresholdM, curThresholdN)) %
                            curThresholdN +
                        relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * thresholdBlockNum;
    }
}
```

#### B6.2.4 Per-expert weight re-pointing and the full matmul API sequence

`SetWOffset` + `SetGlobalBufferW` + `MMCompute`. Note `SetL2CacheHint(CACHE_MODE_DISABLE)` on
the weight tensor when `blockDimM == 1` — i.e. when a group's weight is streamed once and never
reused, do not pollute L2. This is a directly transplantable trick for a 256-expert megakernel.

```cpp
template <typename mmType, bool sync>
__aicore__ inline uint64_t GMMCompute<mmType, sync>::SetWOffset(uint32_t tailN, uint32_t k)
{
    uint64_t wOffset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ && transposeW) {
        wOffset = tailN * (UB_BLOCK_UNIT_SIZE / sizeof(BT)); // 32: quant is 32, float16 is 16
    } else if constexpr (mmType::BT::format == CubeFormat::NZ) {
        wOffset = tailN * AlignUp<16>(k); // 16: nz format last two dim size
    } else if constexpr (transposeW) {
        wOffset = tailN * k;
    } else {
        wOffset = tailN;
    }
    return wOffset;
}

template <typename mmType, bool sync>
__aicore__ inline GlobalTensor<typename mmType::BT::T> GMMCompute<mmType, sync>::SetGlobalBufferW(uint32_t groupIdx,
                                                                                                  uint32_t tailN,
                                                                                                  MNConfig& mnConfig)
{
    uint64_t wOffset = SetWOffset(tailN, mnConfig.k);
#if defined(GMM_ANTI_QUANT) && !defined(GMM_ANTI_QUANT_A8W4_MSD)
    return weightGm[transposeW ? mnConfig.workSpaceOffset - tailN + wOffset : mnConfig.workSpaceOffset];
#else
    GlobalTensor<BT> weightGmLocal;
    if (singleWeight == 0) {
        weightGmLocal.SetGlobalBuffer(GetTensorAddr<BT>(groupIdx, weightTensorPtr) + wOffset);
    } else if (isA8W4FakeQuant) {
        weightGmLocal.SetGlobalBuffer(reinterpret_cast<__gm__ BT*>(weightTensorPtr) + mnConfig.wBaseOffset + wOffset);
    } else {
        weightGmLocal.SetGlobalBuffer(GetTensorAddr<BT>(0, weightTensorPtr) + mnConfig.wBaseOffset + wOffset);
    }
#if !(defined(ASCENDC_OOM) && ASCENDC_OOM == 1)
    if (mnConfig.blockDimM == 1) {
        weightGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
#endif
    return weightGmLocal;
#endif
}
```

```cpp

template <typename mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx,
                                                           uint32_t listIndex)
{
    if (subBlockIdx != 0) {
        return;
    }
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM =
        mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    // init global buffer
    if (singleX == 0) {
        if (groupListType == GROUP_LIST_TYPE_SPARSE && groupType == 0) { // 0: split M
            xGm.SetGlobalBuffer(GetTensorAddr<AT>(listIndex, xTensorPtr));
        } else {
            xGm.SetGlobalBuffer(GetTensorAddr<AT>(groupIdx, xTensorPtr));
        }
    } else {
        xGm.SetGlobalBuffer(GetTensorAddr<AT>(0, xTensorPtr) + mnConfig.xBaseOffset);
    }
    GlobalTensor<BT> weightGmLocal = SetGlobalBufferW(groupIdx, tailN, mnConfig);
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset], transposeX);
    mm.SetTensorB(weightGmLocal, transposeW);
#if defined(GMM_QUANT_INT8)
    if (singleWeight == 0) {
        scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE>(groupIdx, scaleTensorPtr));
    } else {
        scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE>(0, scaleTensorPtr) + mnConfig.nAxisBaseOffset);
    }
    mm.SetQuantVector(scaleGm[tailN]);
#endif
    SetGlobalBufferBias(groupIdx, tailN, mnConfig);
    if (singleY == 0) {
        if (groupListType == GROUP_LIST_TYPE_SPARSE && groupType == 0) {
            yGm.SetGlobalBuffer(GetTensorAddr<CT>(listIndex, yTensorPtr));
        } else {
            yGm.SetGlobalBuffer(GetTensorAddr<CT>(groupIdx, yTensorPtr));
        }
    } else {
        yGm.SetGlobalBuffer(GetTensorAddr<CT>(0, yTensorPtr) + mnConfig.yBaseOffset);
    }
#if defined(GMM_ANTI_QUANT)
    mm.template IterateAll<false>(yGm[outOffset], 0, false, true);
    mmWaitStatus = true;
#else
    mm.template IterateAll<sync>(yGm[outOffset], 0);
#endif
}
```

#### B6.2.5 The design doc on core-splitting across many groups (Chinese, verbatim)

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/GroupedMatmul%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md
author/date: unknown — Huawei Technologies Co., Ltd. (CC-BY-4.0 header), repo cann/ops-transformer branch master, fetched 2026-10-20

This section states the hard constraint that forces the basic-block scheme: the kernel CANNOT
instantiate one matmul high-level-API object per group.

`````
## 3.3 基本块分核方案

GroupedMatmul实现时需要考虑输入为多个tensor的情况，即每组matmul的shape可能各不相同，而kernel侧不能为每组matmul单独配置对应的matmul高阶api接口实例（tiling结构体和core栈空间大小均不允许）。为了适配不同shape的matmul计算，GroupedMatmul采用基本块方式（横向分核），以baseM、baseN为基本块进行分核计算，此处baseM/baseN为matmul的参数。

![基本块分核](../../../docs/zh/figures/gmm_horizontal_core_partition_scheme.png)

## 3.4 对角线分核方案

按基本块方案进行分核，容易存在同地址访问的问题，例如当基本块方案中nDim=coreNum时，则同一时间所有的核都在访问左矩阵的相同地址，对性能影响较大。因此当基本块数量超过coreNum时（没超过coreNum时，对角线方案无法解决同地址访问问题），可以采用如下对角线方案，同一时间不同核尽量错开对数据的访问，每个方块代表一个输出的基本块，数字代表基本块遍历顺序（横向分核为原始基本块分核方案）。

![对角线分核](../../../docs/zh/figures/gmm_diagonal_core_partition_scheme.png)

对于适合对角线优化的场景，在基本块方案上输入数据会存在多次访问，当nDim/mDim很大时，“对角线”不能直接任意往下延伸，否则在k值比较大的场景下，对角线上对应的输入数据均为不同的数据，已经加载过一次的数据被新数据从L2 cache中替换掉，后续需要加载时还是从DDR内存中加载，导致性能劣化。因此需要限制对角线范围，以充分利用L2 cache中缓存的数据。

![对角线分组](../../../docs/zh/figures/gmm_diagonal_grouping_scheme.png)

将对角线遍历按阈值进行分组，为尽量避免同地址访问，两个方向的阈值最好都不小于实际的物理核数，因此有：

$$
\min(T_m, T_n) \geq numCore
$$

为充分利用cache，一个分组块对应的$X$、$Weight$、$Y$的总数据量最好不超过设备的L2 cache大小，因此有:

$$
sizeof(dtype) \cdot (T_m \cdot singleM \cdot K + K \cdot T_n \cdot singleN + T_m \cdot singleM \cdot T_n \cdot singleN) \leq L2_{size}
$$

若两个条件不能同时满足，须根据实际情况做取舍。
`````

GMMArray — how per-group m/k/n lists reach the device when inputs are multi-tensor:

`````
## 3.1 tilingData设计

```c++
BEGIN_TILING_DATA_DEF(GMMTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(GMMBaseParams, gmmBaseParams);
  TILING_DATA_FIELD_DEF_STRUCT(GMMArray, gmmArray);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;
```

tilingData主要包含上述结构里的三个部分：

1. GMMBaseParams: GroupedMatmul的分组数量、Aicore核数、ub tiling参数、matmul分核tiling参数等基础tiling参数，是否有激活函数、量化类型（per token或per tensor）、激活函数类型等功能参数;
2. GMMArray: 当输入是多tensor时，通过三个数组记录每组Matmul的shape, kernel通过GlobalTensor::GetValue()的方式获取。当输入为全单tensor时，m/k/n中的一个值在group_list中，另外两个值在所有的group中都相同，此时kernel只需要访问数组中的第一个值。
3. TCubeTiling: matmul高阶api对应的tilingData。

kernel中为了避免在栈空间中申请GMMArray中3个数组以及避免拷贝这些数组，采用GET_TILING_DATA_MEMBER接口拷贝除GMMArray之外的结构体。
`````

---

### B6.3 GAP 1 — `GroupedMatmulSwigluQuant`: how SwiGLU is fused with the per-expert GEMM

#### B6.3.1 Real paths

```
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_pipeline.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.cpp
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_mid.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_post.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_pre.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_kernel.h   <-- quoted below
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_split_ws.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_utils_kernel.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/...                 (35 files, arch35 + a4w4/a8w4 MSD)
gmm/common/cgmct/kernel/kernel_gmm_swiglu_pertoken_quant.h       (CATLASS-style "cgmct" variant)
gmm/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h
gmm/common/cgmct/epilogue/block_epilogue_dequant_swiglu.h
gmm/common/cgmct/epilogue/cgmct_block_epilogue_swiglu_mx_quant.h
```

#### B6.3.2 THE central design fact: cube and vector use TWO DIFFERENT decompositions

https://github.com/hicann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_kernel.h
author/date: unknown — Huawei Technologies Co., Ltd., repo hicann/ops-transformer branch master (pushed 2026-10-20), fetched 2026-10-20

The AIC side walks groups and basic blocks exactly like plain GroupedMatmul, writing int32
matmul output to a GM workspace. It does NOT do SwiGLU. Note this kernel hardcodes the cumsum
reading (`currSplitValue - prevSplitValue`) — there is no `groupListType` switch here at all.

```cpp
template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::CubeProcess(MNConfig& mnConfig)
{
    if ASCEND_IS_AIC {
        preOffset = 0;
        int32_t prevSplitValue = 0;
        for (uint32_t groupIdx = 0, count = 0; groupIdx < gmmSwiglu->groupListLen; ++groupIdx) {
            UpdateMnConfig(mnConfig);
            int32_t currSplitValue = static_cast<int32_t>(groupListGM.GetValue(groupIdx));
            int32_t splitValue = currSplitValue - prevSplitValue;
            prevSplitValue = currSplitValue;
            SetMNConfig(splitValue, groupIdx, mnConfig);
            if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                continue;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

            uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
            uint32_t curBlock = blockIdx >= count ? blockIdx : blockIdx + gmmBaseParams->coreNum;
            uint32_t thresholdM_dimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;

            while (curBlock < curCount) {
                MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
                MMCompute(groupIdx, mnConfig, blockIdx);
                curBlock += aicCoreNum;
            }
            count = curCount % gmmBaseParams->coreNum;
        }
        GmmsqAllCoreCubeVecStageBarrier();
    }
}

```

The AIV side ignores groups for load-balancing purposes: it reduces the whole groupList to a
total row count `vecConfig.M`, splits THAT flat row range evenly across AIV cores, and then
walks the groupList a SECOND time to recover which expert its first row belongs to
(`vecConfig.curGroupIdx`) and after how many rows it must advance to the next expert
(`vecConfig.nextUpadteInterVal`). That is the per-expert scale addressing.
Chinese step comments 第一步/第二步/第三步/第四步 are original.

```cpp
template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateVecConfig(uint32_t blockIdx,
                                                                                     VecConfig& vecConfig)
{
    // 第一步 读取grouplist reduceSum 计算总数据个数
    int64_t prevM = 0;
    for (uint32_t groupIdx = 0; groupIdx < gmmSwiglu->groupListLen; groupIdx++) {
        int64_t currM = groupListGM.GetValue(groupIdx);
        int64_t tempM = currM - prevM;
        prevM = currM;
        vecConfig.M += tempM;
    }
    // 第二步 计算分核
    uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
    vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
    uint32_t tailCoreIdx = vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
    vecConfig.taskNum = blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    vecConfig.startIdx =
        blockIdx < tailCoreIdx ? eachCoreTaskNum * blockIdx : ((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
    vecConfig.curIdx = vecConfig.startIdx;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    prevM = 0;
    for (uint32_t groupIdx = 0; groupIdx < gmmSwiglu->groupListLen; groupIdx++) {
        int64_t currM = groupListGM.GetValue(groupIdx);
        int64_t tempM = currM - prevM;
        prevM = currM;
        if (curStartIdx >= 0 && curStartIdx - tempM < 0) {
            vecConfig.curGroupIdx = groupIdx;
            vecConfig.nextUpadteInterVal = tempM - curStartIdx;
        }
        curStartIdx -= tempM;
    }
    // 第三步 计算总数据量
    vecConfig.outLoopNum = (vecConfig.taskNum + gmmSwiglu->maxProcessRowNum - 1) / gmmSwiglu->maxProcessRowNum;
    vecConfig.tailLoopNum = vecConfig.taskNum % gmmSwiglu->maxProcessRowNum ?
                                vecConfig.taskNum % gmmSwiglu->maxProcessRowNum :
                                gmmSwiglu->maxProcessRowNum;
    pipe->Reset();
    // 第四步 申请空间
    pipe->InitBuffer(mmOutQueue, DOUBLE_BUFFER, gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen * sizeof(int32_t));
    pipe->InitBuffer(perChannelScaleInQueue, DOUBLE_BUFFER, gmmSwiglu->tokenLen * sizeof(float));
    pipe->InitBuffer(quantOutQueue, DOUBLE_BUFFER,
                     gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, DOUBLE_BUFFER,
                     AlignUp<int32_t>(gmmSwiglu->maxProcessRowNum, ALIGN_8_ELE) * sizeof(float));
    // two 32 byte buffer for reduceMax calculation in Quant.
    pipe->InitBuffer(reduceWorkspace, gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(float) + UB_BLOCK_UNIT_SIZE +
                                          UB_BLOCK_UNIT_SIZE);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
```

#### B6.3.3 Per-expert `perChannelScale` addressing (`groupIdx * tokenLen`) and the quant buffers

```cpp
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR perChannelScale, GM_ADDR perTokenScale, GM_ADDR groupList, GM_ADDR quantOutput,
    GM_ADDR quantScaleOutput, GM_ADDR workspace, const GMMSwigluBaseParams* __restrict gmmSwigluBaseParamsIn,
    const TCubeTiling* __restrict mmTilingDataIN, const GMMSwiglu* __restrict gmmSwigluIN, TPipe* tPipeIN)
{
    aicCoreNum = GetBlockNum();
    aivCoreNum = aicCoreNum * 2;
    blockIdx = GetBlockIdx();
    mmTilingData = mmTilingDataIN;
    gmmBaseParams = gmmSwigluBaseParamsIn;
    gmmSwiglu = gmmSwigluIN;
    pipe = tPipeIN;
    xTensorPtr = x;
    weightTensorPtr = weight;
    groupListGM.SetGlobalBuffer((__gm__ int64_t*)groupList, gmmSwiglu->groupListLen);
    mmOutGM.SetGlobalBuffer((__gm__ int32_t*)workspace, gmmBaseParams->M * gmmSwiglu->tokenLen);
    if ASCEND_IS_AIV {
        perChannelScaleGM.SetGlobalBuffer((__gm__ CHANNELDTYPE*)perChannelScale,
                                          gmmSwiglu->groupListLen * gmmSwiglu->tokenLen);
        perTokenScaleGM.SetGlobalBuffer((__gm__ float*)perTokenScale, gmmBaseParams->M);
        quantOutputGM.SetGlobalBuffer((__gm__ int8_t*)quantOutput,
                                      gmmBaseParams->M * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float*)quantScaleOutput, gmmBaseParams->M);
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Process()
{
    MNConfig mnConfig;
    VecConfig vecConfig;
    CubeProcess(mnConfig);
    VecProcess(vecConfig);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::PreLoadTokenAndChannel(
    LocalTensor<float>& channelScaleLocal, VecConfig& vecConfig)
{
    DataCopyExtParams copyChannelParams{1, static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(DTYPE_CS)), 0, 0, 0};
    DataCopyPadExtParams<DTYPE_CS> padParams{false, 0, 0, 0};
    if constexpr (!IsSameType<DTYPE_CS, float>::value) {
        LocalTensor<DTYPE_CS> dstLocalT = channelScaleLocal.template ReinterpretCast<DTYPE_CS>();
        DataCopyPad(dstLocalT[gmmSwiglu->tokenLen], perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                    copyChannelParams, padParams);
        GmmsqSetWaitFlag<HardEvent::MTE2_V>();
        Cast(channelScaleLocal, dstLocalT[gmmSwiglu->tokenLen], RoundMode::CAST_NONE, gmmSwiglu->tokenLen);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopyPad(channelScaleLocal, perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                    copyChannelParams, padParams);
        GmmsqSetWaitFlag<HardEvent::MTE2_V>();
    }
    perChannelScaleInQueue.EnQue(channelScaleLocal);
}

```

#### B6.3.4 The cube-side `MMCompute` of the fused kernel — output goes to workspace, not y

```cpp
template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig,
                                                                               uint32_t coreIdx)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM =
        mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    xGM.SetGlobalBuffer((__gm__ int8_t*)xTensorPtr + mnConfig.xBaseOffset);
    weightGM.SetGlobalBuffer((__gm__ int8_t*)weightTensorPtr + mnConfig.wBaseOffset + GetWOffset(tailN, mnConfig.k));
    if (mnConfig.blockDimM == 1) {
        weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGM[xOffset], transposeX);
    mm.SetTensorB(weightGM, transposeW);
    mm.template IterateAll<sync>(mmOutGM[mnConfig.workSpaceOffset], 0);
}

```

#### B6.3.5 The AIV pipeline body, including the all-core cube/vector stage barrier

`GmmsqAllCoreCubeVecStageBarrier()` is called once on the AIC side at the end of `CubeProcess`
and twice on the AIV side (once after buffer alloc + scale preload, once implicitly by ordering).
This is the only cross-cube/vector sync in the fused op.

```cpp
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::VecProcess(VecConfig& vecConfig)
{
    if ASCEND_IS_AIV {
        UpdateVecConfig(blockIdx, vecConfig);
        if (blockIdx < vecConfig.usedCoreNum) {
            LocalTensor<float> channelScaleLocal = perChannelScaleInQueue.AllocTensor<float>();
            LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
            mmOutQueue.EnQue(mmLocal);
            quantScaleOutQueue.EnQue(quantScaleLocal);
            quantOutQueue.EnQue(quantLocal);
            PreLoadTokenAndChannel<CHANNELDTYPE>(channelScaleLocal, vecConfig);
        }
        GmmsqAllCoreCubeVecStageBarrier();
        if (blockIdx < vecConfig.usedCoreNum) {
            int32_t eventIdVToMTE3 = static_cast<int32_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                vecConfig.innerLoopNum =
                    outLoopIdx == (vecConfig.outLoopNum - 1) ? vecConfig.tailLoopNum : gmmSwiglu->maxProcessRowNum;
                customDataCopyIn(outLoopIdx, vecConfig);
                for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                    UpdateChannelScale<CHANNELDTYPE>(innerLoopIdx, vecConfig);
                    VectorCompute(innerLoopIdx, vecConfig);
                }
                SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                customDataCopyOut(vecConfig);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3);

            LocalTensor<float> channelScaleLocal = perChannelScaleInQueue.DeQue<float>();
            LocalTensor<int32_t> mmLocal = mmOutQueue.DeQue<int32_t>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            perChannelScaleInQueue.FreeTensor(channelScaleLocal);
            mmOutQueue.FreeTensor(mmLocal);
            quantScaleOutQueue.FreeTensor(quantScaleLocal);
            quantOutQueue.FreeTensor(quantLocal);
        } else {
            return;
        }
    }
}

```

---

### B6.4 GAP 2 — CATLASS grouped matmul templates

#### B6.4.1 Where it lives

`cann/catlass` on gitcode is mirrored (same content, authenticated access works) as
**`hicann/catlass` on GitHub**, `master`, pushed 2026-10-19, 849 MB, 3487 tracked paths.
`cann/catlass` and `Ascend/catlass` 404 on GitHub; use `hicann/catlass`.

Grouped-matmul kernel templates, real paths:

```
include/catlass/gemm/kernel/group_gemm.hpp
include/catlass/gemm/kernel/grouped_matmul.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_k.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_k_per_token_dequant.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_m.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_m_gelu.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_m_per_tensor_per_channel_dequant_tla.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_tla.hpp
include/catlass/gemm/kernel/grouped_matmul_slice_m_tla.hpp
include/catlass/gemm/kernel/grouped_mx_matmul_finalize_routing_no_deter_tla.hpp
include/catlass/gemm/kernel/grouped_mx_matmul_finalize_routing_tla.hpp
include/catlass/gemm/kernel/grouped_mx_matmul_slice_m_aswt_tla.hpp
include/catlass/gemm/kernel/grouped_mx_matmul_slice_m_swiglu_mx_quant_tla.hpp
include/catlass/gemm/kernel/grouped_mx_matmul_slice_m_tla.hpp
include/catlass/gemm/kernel/weight_quant_a8w4_grouped_mx_matmul.hpp
include/catlass/gemm/block/block_swizzle_grouped_aswt.hpp
examples/02_grouped_matmul_slice_m/, 05_grouped_matmul_slice_k/, 08_grouped_matmul/,
examples/07_grouped_matmul_slice_m_per_token_dequant_moe/, 10_, 11_, 16_group_gemm/,
examples/35_w4a8_grouped_matmul_msd/, 47_ascend950_grouped_matmul_slice_m_per_token_dequant/
```

#### B6.4.2 `GroupedMatmulSliceM` — the canonical group loop and per-group pointer advance

https://github.com/hicann/catlass/blob/master/include/catlass/gemm/kernel/grouped_matmul_slice_m.hpp
author/date: unknown — Copyright (c) 2025 Huawei Technologies Co., Ltd., repo hicann/catlass branch master, fetched 2026-10-20

Note three things relevant to a hand-written megakernel:

1. The groupList read is a per-element `GetValue` with an explicit cumsum-difference, and the
   cumsum-vs-count choice is a COMPILE-TIME macro `CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED`
   (defined → the value IS the group size; undefined → cumsum, diff with previous).
2. `gmGroupOffsetA/B/C` advance by `m*k`, `k*n`, `m*n` after each group — three running int64
   accumulators, no table.
3. `startCoreIdx = (startCoreIdx + coreLoops) % coreNum` is the whole core→group mapping. No
   barrier between groups.

```cpp
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler blockScheduler;
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
#ifdef CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED
            uint32_t currentM = groupList.GetValue(groupIdx);
#else
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                                                  (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
#endif
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;
            LayoutC layoutC = params.layoutC.GetTileLayout(inGroupProblemShape.GetCoordMN());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            if (CeilDiv(currentM, L1TileShape::M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmGroupOffsetC + gmOffsetC],
                    layoutC, actualBlockShape);
            }

            gmGroupOffsetA += static_cast<int64_t>(inGroupProblemShape.m()) * inGroupProblemShape.k();
            gmGroupOffsetB += static_cast<int64_t>(inGroupProblemShape.k()) * inGroupProblemShape.n();
            gmGroupOffsetC += static_cast<int64_t>(inGroupProblemShape.m()) * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
```

#### B6.4.3 `GroupedMatmul` — the fully general per-group problem-shape list (MAX_TENSOR_COUNT = 256)

https://github.com/hicann/catlass/blob/master/include/catlass/gemm/kernel/grouped_matmul.hpp
author/date: unknown — Copyright (c) 2025 Huawei Technologies Co., Ltd., repo hicann/catlass branch master, fetched 2026-10-20

Here there is no groupList at all; instead the host hands in GM arrays of `GemmCoord`
problem shapes and of per-group `LayoutA/B/C`, and the kernel COPIES ALL OF THEM INTO CORE
STACK ARRAYS of size 256 at kernel entry via `UnpackListParam` (a uint64-granular memcpy from
GM). `MAX_TENSOR_COUNT = 256` is exactly the routed-expert count of interest, and
`CanImplement` rejects anything larger.

```cpp
namespace detail {

template <class T>
CATLASS_DEVICE void UnpackListParam(T* const dst, GM_ADDR src, uint32_t len)
{
    for (uint32_t i = 0; i * sizeof(uint64_t) < len * sizeof(T); ++i) {
        reinterpret_cast<uint64_t*>(dst)[i] = reinterpret_cast<__gm__ uint64_t*>(src)[i];
    }
}

} // namespace detail

```

```cpp
// Template for grouped matmul kernel. Compute grouped C = A * B
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GroupedMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t MAX_TENSOR_COUNT = 256;

```

```cpp
        uint8_t* ptrLayoutC;
    };
    static bool CanImplement(const Arguments& args)
    {
        if (args.problemCount > MAX_TENSOR_COUNT) {
            return false;
        }
        return true;
    }
```

The device body — group-problem-shape iteration plus per-group pointer advance:

```cpp

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    /// Executes matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        GemmCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutC layoutCList[MAX_TENSOR_COUNT];

        // Get matmul information from parameters
        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutCList, params.ptrLayoutC, params.problemCount);

        BlockScheduler matmulBlockScheduler;
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t inGroupOffsetA = 0;
        int64_t inGroupOffsetB = 0;
        int64_t inGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            GemmCoord problemShape = problemShapeList[groupIdx];
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutC layoutC = layoutCList[groupIdx];

            matmulBlockScheduler.Update(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[inGroupOffsetA + gmOffsetA], layoutA, gmB[inGroupOffsetB + gmOffsetB], layoutB,
                    gmC[inGroupOffsetC + gmOffsetC], layoutC, actualBlockShape);
            }

            inGroupOffsetA += static_cast<int64_t>(problemShape.m()) * problemShape.k();
            inGroupOffsetB += static_cast<int64_t>(problemShape.k()) * problemShape.n();
            inGroupOffsetC += static_cast<int64_t>(problemShape.m()) * problemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
```

#### B6.4.4 `GemmGroupedAswtTailSplitSwizzle` — the per-group core window that carries across groups

https://github.com/hicann/catlass/blob/master/include/catlass/gemm/block/block_swizzle_grouped_aswt.hpp
author/date: unknown — Copyright (c) 2026 Huawei Technologies Co., Ltd., repo hicann/catlass branch master, fetched 2026-10-20

This is the most directly reusable piece for a 256-expert megakernel: an explicit
"rolling round bookkeeping (carried across groups, no barrier)" state machine, with a
tail-split that fires only when the final wave of a group leaves at least half the cores idle.

```cpp
template <uint32_t WindowLen = 4, bool TransA = false, bool TransB = true>
struct GemmGroupedAswtTailSplitSwizzle {
    static constexpr uint32_t BLOCK_CUBE = 16;                // cube alignment
    static constexpr uint32_t INNER_AXIS_MIN_SPLIT_VAL = 128; // ND2NZ cache-line size

    /// Result of GetBlockShape: tile size (m, n) plus the sub-tile offset inside the base tile.
    /// For non tail-split tiles mOffset == nOffset == 0.
    struct AswtBlockShape {
        uint32_t m;
        uint32_t n;
        uint32_t mOffset;
        uint32_t nOffset;
    };

    /// Data members ---------------------------------------------------------------------------

    // Current group problem dims.
    uint32_t m_{0};
    uint32_t n_{0};
    uint32_t k_{0};
    // Base tile. baseM_ may be rebalanced per group through UpdateBaseM().
    uint32_t baseM_{1};
    uint32_t baseN_{1};
    // Derived per-problem quantities (cached while m/n unchanged across groups).
    uint32_t mCnt_{1};
    uint32_t nCnt_{1};
    uint32_t totalCnt_{1};
    uint32_t mBaseTail_{0};
    uint32_t nBaseTail_{0};
    uint32_t mainMWindow_{1};
    uint32_t mainRow_{0};
    uint32_t tailWindow_{1};
    // Per-core rolling round bookkeeping (carried across groups, no barrier).
    uint32_t blockNum_{1};
    uint32_t blockIdx_{0};
    uint32_t startBlockIdx_{0};
    uint32_t endBlockIdx_{0};
    uint32_t round_{0};
    uint32_t roundIdx_{0};
    // Tail-split state (only the last group ever sets tailCnt_ > 1).
    uint32_t mTailCnt_{1};
    uint32_t nTailCnt_{1};
    uint32_t tailCnt_{1};
    uint32_t tailBlockBase_{0};
    uint32_t nTailAlign_{BLOCK_CUBE};

```

```cpp
    /// Methods --------------------------------------------------------------------------------

    CATLASS_DEVICE
    GemmGroupedAswtTailSplitSwizzle()
    {}

    CATLASS_DEVICE
    GemmGroupedAswtTailSplitSwizzle(uint32_t baseM, uint32_t baseN)
        : baseM_(baseM),
          baseN_(baseN),
          blockNum_(AscendC::GetBlockNum()),
          blockIdx_(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum()),
          endBlockIdx_(AscendC::GetBlockNum() - 1)
    {
        // Tail-split N min alignment. For TransB (ColumnMajor weight) aclnn uses BLOCK_CUBE; for
        // !TransB the split is additionally forced to INNER_AXIS_MIN_SPLIT_VAL inside GetBlockShape
        // (a multiple of BLOCK_CUBE), so this default suffices for both layouts.
        nTailAlign_ = BLOCK_CUBE;
    }

    /// Optional base-M rebalance for the current group. Pass L1_TILE_M for plain ASWT.
    CATLASS_DEVICE
    void UpdateBaseM(uint32_t baseM)
    {
        baseM_ = baseM;
    }

    /// Start a new group. Recomputes window params (cached on m/n) and rolls the core window.
    CATLASS_DEVICE
    void UpdateNextProblem(GemmCoord const& problemShape)
    {
        k_ = problemShape.k();
        if (m_ != problemShape.m() || n_ != problemShape.n()) {
            m_ = problemShape.m();
            n_ = problemShape.n();
            mCnt_ = CeilDiv(m_, baseM_);
            nCnt_ = CeilDiv(n_, baseN_);
            mBaseTail_ = m_ - (mCnt_ - 1) * baseM_;
            nBaseTail_ = n_ - (nCnt_ - 1) * baseN_;
            totalCnt_ = mCnt_ * nCnt_;
            mainMWindow_ = (WindowLen < mCnt_) ? WindowLen : mCnt_;
            mainRow_ = mCnt_ / mainMWindow_ - 1;
            tailWindow_ = mCnt_ - mainMWindow_ * mainRow_;
        }
        // Reset tail-split state for every group (only the last group re-enables it).
        mTailCnt_ = 1;
        nTailCnt_ = 1;
        tailCnt_ = 1;
        tailBlockBase_ = 0;

        roundIdx_ = 0;
        round_ = CeilDiv(totalCnt_, blockNum_);
        // The first physical block of the new group continues right after the previous group's end.
        startBlockIdx_ = (endBlockIdx_ == blockNum_ - 1) ? 0 : (endBlockIdx_ + 1);
        // The last physical block of the new group.
        endBlockIdx_ = (totalCnt_ + startBlockIdx_ - 1) % blockNum_;
        // Cores outside [startBlockIdx_, endBlockIdx_] handle one fewer round.
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
    }

    /// Number of physical cores that own a tile in the current group's final wave.
    CATLASS_DEVICE
    uint32_t GetTailTileCnt() const
    {
        return Min(endBlockIdx_ + 1, totalCnt_);
    }

    CATLASS_DEVICE
    uint32_t GetEndBlockIdx() const
    {
        return endBlockIdx_;
    }

    /// Whether a tail-split is worthwhile: only when at least half the cores are idle in the final
    /// wave (matches aclnn IfNeedSplit).
    CATLASS_DEVICE
    bool NeedTailSplit() const
    {
        return (endBlockIdx_ + 1) <= (blockNum_ / 2);
```

#### B6.4.5 The SwiGLU+MX-quant fused grouped matmul in CATLASS (bridges gap 1 and gap 2)

https://github.com/hicann/catlass/blob/master/include/catlass/gemm/kernel/grouped_mx_matmul_slice_m_swiglu_mx_quant_tla.hpp
author/date: unknown — Huawei Technologies Co., Ltd., repo hicann/catlass branch master, fetched 2026-10-20

Note `N_half` — the gate/up split is handled by halving N of the in-group problem shape, and the
two halves are addressed by separate offsets rather than by two separate GEMMs. Note also that
here `currentM = groupList.GetValue(groupIdx)` is read as a PER-GROUP COUNT (no cumsum diff),
i.e. this template assumes groupListType == 1.

```cpp
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementQ> gmQ;
        gmQ.SetGlobalBuffer((__gm__ ElementQ*)params.ptrQ);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
        mmResUb_ping_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, 0, elems);
        mmResUb_pong_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, elems * sizeof(ElementC), elems);

        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t gmGroupOffsetMxScaleB = 0;
        int64_t mxScaleAlignedK =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

        int64_t totalM = 0;
        uint32_t startCoreIdx = 0;

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);
            if (currentM == 0)
                continue;

            uint32_t N = params.problemShape.n();
            uint32_t N_half = N / N_HALF;

            GemmCoord inGroupProblemShape{currentM, N_half, params.problemShape.k()};

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                currentM, CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
            gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB + gmGroupOffsetMxScaleB);
            AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
            gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + gmGroupOffsetMxScaleA);

            if (CeilDiv(currentM, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
```

Per-group advance of the three offset accumulators (A implicit, B, and BOTH mx-scale tensors):

```cpp
                GemmCoord gateBlockShape{actualBlockShape.m(), actualN_half, actualBlockShape.k()};
                blockMmad(
                    tensorBlockA, tensorBlockB_gate, tensorBlockC_gate, gateBlockShape, tensorBlockMxScaleA_act,
                    tensorBlockMxScaleB_gate);

                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                isVecSetSyncCom_ = true;
            }

            totalM += currentM;
            gmGroupOffsetB += inGroupProblemShape.k() * params.problemShape.n();
            gmGroupOffsetMxScaleA += currentM * mxScaleAlignedK;
            gmGroupOffsetMxScaleB += mxScaleAlignedK * params.problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if (isVecSetSyncCom_) {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
```

The AIV half of the same kernel re-walks the identical group loop with its own `startCoreIdx`
rolling cursor — so cube and vector agree on the group decomposition here, unlike the
ops-transformer `GroupedMatmulSwigluQuant` in section 3.2:

```cpp
        gmQScale.SetGlobalBuffer((__gm__ ElementQScale*)params.ptrQScale);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        blockEpilogue.UpdateGlobalAddr(gmQ, gmQScale);

        constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
        mmResUb_ping_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, 0, elems);
        mmResUb_pong_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, elems * sizeof(ElementC), elems);

        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t mxScaleAlignedK =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

        int64_t totalM = 0;
        uint32_t startCoreIdx = 0;

        uint32_t N = params.problemShape.n();
        uint32_t N_half = N / N_HALF;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);
            if (currentM == 0)
                continue;

            GemmCoord inGroupProblemShape{currentM, N_half, params.problemShape.k()};

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                currentM, CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            uint32_t startLoopIdx;
            if (aicoreIndex < startCoreIdx) {
                startLoopIdx = aicoreIndex + aicoreNum - startCoreIdx;
            } else {
                startLoopIdx = aicoreIndex - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                uint32_t actualN_half = actualBlockShape.n();

                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);

                GemmCoord resShape{actualBlockShape.m(), actualN_half, actualBlockShape.k()};
                blockEpilogue(resShape, totalM, blockCoord, mmResUb_ping_, mmResUb_pong_, L1_TILE_M, L1_TILE_N);

                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
            }

            totalM += currentM;
            startCoreIdx = (startCoreIdx + coreLoops) % aicoreNum;
        }

```

#### B6.4.6 `examples/02_grouped_matmul_slice_m/README.md`

https://github.com/hicann/catlass/blob/master/examples/02_grouped_matmul_slice_m/README.md
author/date: unknown — repo hicann/catlass branch master, fetched 2026-10-20

`````
# GroupedMatmulSliceM Example Readme

## 功能说明

- 算子功能：完成分组矩阵乘（Grouped Matmul）计算。
- 计算公式：

  $$
    C = \mathrm{GroupedGEMM}(A,B), \\
    C_i = A_i \times B_i, \quad 0 \le i < g
  $$

  其中 $g$ 为分组数量，分组矩阵乘中的$A_i$、$B_i$、$C_i$ 的形状分别为 `(m_i,k)`、`(k,n)`、`(m_i,n)`。各组沿 m 轴连续存储，共享 n、k 维度。
- 支持产品型号：Atlas A2/A3系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float16` | 各组左矩阵，layout 固定为 `RowMajor` |
| `B` | Input | `(g,k,n)` | `float16` | 右矩阵，layout支持`RowMajor` 和 `ColumnMajor`（默认），数据类型与左矩阵一致 |
| `groupList` | Input | `(g,)` | `int64` | m 轴分组长度的前缀和序列 |
| `C` | Output | `(m,n)` | `float16` | 分组矩阵乘结果，当前 layout 为 `RowMajor`；有效行数由 `groupList` 的终止量决定（不大于 `m`） |

## 使用范围说明

推荐范围：

- 分组数量 `g` 满足 `1 ≤ g ≤ 128` 且 `g ≤ m`。

## 使用示例

### 命令行参数

```bash
02_grouped_matmul_slice_m [problemCount] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `problemCount` | 无 | 分组数量，必须大于 0 |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 02_grouped_matmul_slice_m
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./02_grouped_matmul_slice_m 128 512 1024 2048 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
`````

---

### B6.5 GAP 3 — Does a SECOND, independent grouped-matmul live in `ops-nn`? NO.

https://github.com/hicann/ops-nn (and https://gitcode.com/cann/ops-nn)
author/date: repo hicann/ops-nn branch master, 40200 tracked paths, fetched 2026-10-20

NEGATIVE RESULT, stated explicitly. Full recursive tree of `hicann/ops-nn` (40200 paths)
grepped case-insensitively for `grouped_mat`, `group_mat`, `gmm` → **zero matches**.
The complete set of `group`/`moe`/`expert`/`rout`-matching op directories in ops-nn is:

```
activation/swiglu_backward_group_quant_with_dual_axis
activation/swiglu_group
activation/swiglu_group_grad
activation/swiglu_group_quant
activation/swiglu_group_quant_grad
activation/swiglu_group_quant_with_dual_axis
norm/group_norm
norm/group_norm_grad
norm/group_norm_silu
norm/group_norm_silu_quant
norm/group_norm_swish
norm/group_norm_swish_grad
norm/group_norm_v2
quant/group_quant
quant/grouped_dynamic_block_quant
quant/grouped_dynamic_mx_quant
quant/grouped_dynamic_mx_quant_with_dual_axis
quant/grouped_quant_max
```

None of these is a matmul. `ops-nn/matmul/` holds 34 ops
(`addmv batch_mat_mul_v3 batch_matmul_quant common convert_weight_to_int4_pack
dual_level_quant_batch_matmul flat_quant fused_linear_cross_entropy_loss_grad
fused_linear_online_max_sum fused_mat_mul fused_quant_mat_mul gemm gemm_syrk gemm_v2 gemm_v3
mat_mul_v3 matmul_compress matmul_emu_split_weight matmul_v2_compress_dequant mv
quant_batch_matmul_inplace_add quant_batch_matmul_v3 quant_batch_matmul_v4 quant_matmul
quant_matmul_activation_quant quant_matmul_dequant quant_matmul_reduce_sum rotate_quant
sparse4to2quant_matmul sparse_tensor_dense_mat_mul transpose_batch_mat_mul
transpose_quant_batch_mat_mul weight_quant_batch_matmul weight_quant_batch_matmul_v2`),
none grouped.

So there is exactly ONE first-party grouped-matmul op family (in `ops-transformer/gmm/`) plus
the CATLASS template library. The only other independent implementations found are
third-party forks/vendored copies, e.g.
`vllm-project/afd-plugin:csrc/npu/ascend_kernels/grouped_matmul_layered/op_kernel/grouped_matmul_utils.h`
and `QingCheng-AI/ascend-kernel:grouped_gemm/op_kernel/grouped_matmul_utils.h`, both of which
are copies of the ops-transformer `GetSplitValueFromGroupList`, and
`egretwAlker/cpl-ascend-transformer-boost:src/kernels/mixkernels/gmm_add/op_kernel/gmm_add.h`
(ATB's own gmm_add). `ops-transformer/mc2/3rd/grouped_matmul/op_kernel/mc2_3rd_quant_utils.h`
is a vendored third-party copy inside ops-transformer itself.

---

### B6.6 GAP 4 — MegaMoE: public design documents (not the source)

#### B6.6.0 NAME-COLLISION WARNING, read first

"MegaMoE" denotes at least three different things in 2026 and they must not be conflated:

1. **Huawei/CANN `MegaMoe`** — the fused operator at `ops-transformer/mc2/mega_moe`, Ascend, AIC/AIV
   split, HCCL shmem / peermem. This is the one in scope. 116 files, fully open source,
   arch22 (A2/A3) and arch35 (950PR/950DT) trees.
2. **MindSpore `HyperMegaMoE`** — a separate, MindSpore-side RFC for the Multicore/HyperParallel
   stack, forward AND backward fused, quoted in 6.2. Also Huawei, also Ascend, but a different
   operator with a different name and its own aclnn entry points.
3. **DeepGEMM / NVIDIA-side `MegaMoE`** — the symmetric-memory NVLink persistent megakernel
   (dispatch → L1 GEMM → SwiGLU → L2 GEMM → combine in one CUDA kernel). AMD Primus has a
   two-kernel analogue. These are NOT Huawei and share no code with (1).

The one public write-up that surfaced for (3) is
https://ggaaooppeenngg.github.io/zh-CN/2026/06/01/MegaMoE-%E9%80%9A%E4%BF%A1%E8%AE%A1%E7%AE%97%E8%9E%8D%E5%90%88Mega-Kernel%E6%B7%B1%E5%BA%A6%E5%88%86%E6%9E%90/
— it is about CUDA/NVLink symmetric memory, not Ascend. Recorded here only so it is not mistaken
for Huawei material.

Searched and found NOTHING for Huawei's MegaMoE: no arxiv paper, no hiascend.com
`aclnnMegaMoE` doc page, no bbs.huaweicloud.com or hiascend forum post, no zhihu/CSDN article
about the Ascend `mc2/mega_moe` operator specifically. CSDN hits
(`gitcode.csdn.net/6a0ecbae662f9a54cb761980.html`,
`aicoding.csdn.net/6a69d2b3662f9a54cb95e871.html`) are generic repo-overview blogspam that does
not describe MegaMoE's internals. The authoritative design documents are the in-repo README and
the MindSpore RFC, both quoted below.

#### B6.6.1 `mc2/mega_moe/README.md` — the operator's own design statement

https://github.com/hicann/ops-transformer/blob/master/mc2/mega_moe/README.md
author/date: unknown — Huawei Technologies Co., Ltd., repo hicann/ops-transformer branch master (pushed 2026-10-20), fetched 2026-10-20

983 lines. The fusion boundary, verbatim:

```
## 功能说明

- 算子功能：MegaMoE算子将MoE层的专家FFN的完整计算流程及前后数据通信（即 Dispatch + Linear1 + Activation + Linear2 + Combine）融合为单个算子，实现了通信和计算的掩盖。

```

The A16W16 non-quant computation walkthrough, which is the clearest statement of how a single
operator covers Dispatch / per-expert Linear1 / Activation / Linear2 / Combine. Note
「按任意固定顺序行堆叠」 — the per-expert token matrix may be stacked in ANY fixed order, i.e.
the kernel does not promise a particular permutation, only that `row(i,k)` is recoverable:

```
    <details>
    <summary> A16W16 非量化场景</summary>

    - **EP Dispatch**

        在 Dispatch 阶段，每个 $\text{token}_i$ 将其token表示 $\mathbf{x}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$。即对于每个 $k$，专家 $e_{i,k}$ 接收一份 $\mathbf{x}_i$。

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数，则有 $\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的token表示 $\mathbf{x}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$。该映射记录了 $\mathbf{x}_i$ 在专家 $e_{i,k}$ 的输入矩阵 $\mathbf{X}_{e_{i,k}}$ 中的位置。

    - **Expert Compute**

        在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用由 `activation` 指定的门控激活结构以提升表达能力。整个计算过程分为如下三个子步骤。

        **1. Linear1 投影**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。其计算公式为
        $$
        \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}\bigr)
        $$
        **2. 激活**

        将 $\mathbf{H}_e$ 沿列维度拆分为 gate 分支 $\mathbf{G}_e$ 和 up 分支 $\mathbf{U}_e$，根据 `activation` 计算中间激活表示 $\mathbf{A}_e$；各可选激活类型的计算方式参见[激活函数公式](#activation-formulas)。

        **3. Linear2 投影**

        Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
        $$
        \mathbf{Y}_e = \mathbf{A}_e \cdot \mathbf{W}_2[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
        $$

        经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。
    - **Token Combine**

        Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：

        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\;
        \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$

        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$。
```

The supported A/W dtype matrix and the per-product exclusions (this is what tells you A3 cannot
do A8W8-FP and 950 cannot do A16W16 / A8W8-INT / A8W4-INT):

```
    | 场景名    |  A |  W   |
    | --- | :---:  | :---:        |
    | A16W16   | BFLOAT16     |BFLOAT16        |
    | A8W8-INT | INT8   | INT8         |
    | A8W8-FP  | FLOAT8_E4M3FN、FLOAT8_E5M2 |FLOAT8_E4M3FN、FLOAT8_E5M2 |
    | A8W4-INT | INT8        | INT4            |
    | A8W4-FP | FLOAT8_E4M3FN        | FLOAT4_E2M1          |
    | A4W4-FP | FLOAT4_E2M1        | FLOAT4_E2M1            |

    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：不支持上表中A8W8-FP场景。
    - <term>Ascend 950PR&950DT系列产品</term>：不支持上表中A16W16、A8W8-INT、A8W4-INT场景。
    - <term>Ascend 950PR&950DT系列产品</term>的dispatch阶段有两条路径：`dispatchQuantMode` = 4时，按下文各量化场景的描述由算子内部对BF16的`x`做MX逐组量化；`dispatchQuantMode` = 0时为预量化直通，`x`已经是量化后的数据、缩放因子由`scales`入参给出，算子跳过下文第一阶段的量化步骤，直接用`x`与`scales`参与后续矩阵乘，其余阶段与下文一致。
    - <term>Ascend 950PR&950DT系列产品</term>：MoE专家与共享专家可以采用不同的权重类型与量化类型，共享专家的量化输出类型由`sharedExpertQuantOutDtype`指定，未显式配置时与`dispatchQuantOutDtype`一致。

```

A determinism constraint worth copying into any megakernel design — `combineCommMode` = 1
accumulates a token's several local expert outputs in FP32 and sends ONE BF16, which is
deliberately NOT bit-identical to mode 0:

```
    - `combineCommMode` = 1时，同一token在同一张卡上的多个专家输出改为在该卡按FP32累加后一次转BF16发送，累加顺序与中间取整次数与关闭态不同，因此`y`与`combineCommMode` = 0的结果不逐位相同，差异属BF16累加顺序不同带来的正常误差；需要与关闭态逐位一致时请使用`combineCommMode` = 0。
```

#### B6.6.2 MindSpore `HyperMegaMoE` RFC — the explicit stage→unit mapping and overlap policy

https://github.com/mindspore-ai/hyper-parallel/issues/654
(mirror of https://gitcode.com/mindspore/hyper-parallel/issues/312)
author: rainyhorse — created 2026-09-15T09:29:09Z

This is the single most useful public design document, because it names which stage runs on AIV
vs AIC, states that there is NO global barrier, and enumerates the overlap pairs.

`````
## 2 交付范围

### 2.1 正向融合

正向算子的逻辑名称为 `HyperMegaMoE`，ACLNN 执行接口为 `aclnnHyperMegaMoE`。

融合计算链路为：

```text
AllToAll-Dispatch (AIV)
          ↓
GMM1 / up_proj (AIC)
          ↓
SwiGLU (AIV)
          ↓
GMM2 / down_proj (AIC)
          ↓
AllToAll-Combine (AIV)
```

各阶段按 token/expert tile 拆分为任务，由 AIC worker 和 AIV worker 并发执行。任务之间通过事件计数表达生产者—消费者依赖，满足数据就绪条件后即可放行下游任务，不引入全局屏障。

正向对外输出最终 token 结果 `y`，shape 与输入 `x` 一致。中间的 dispatch buffer、GMM 输出、SwiGLU 输出和 combine buffer 不作为公共返回值。

### 2.2 反向融合

反向算子的逻辑名称为 `HyperMegaMoEGrad`，ACLNN 执行接口为 `aclnnHyperMegaMoEGrad`。

反向计算链路为：

```text
AllToAll-Dispatch(dY) (AIV)
          ↓
act_grad: dHidden = dispatched_dY × W2ᵀ (AIC)
          │
          ├── SwiGLUGrad (AIV) ──→ input_grad GMM (AIC)
          │
          └── W2Grad GMM (AIC)

AllToAll-Combine(dX) (AIV) ∥ W1Grad GMM (AIC)
```

反向必须生成：

- 输入 token 梯度 `dx`；
- `up_proj_weight` 梯度 `d_up_proj_weight`；
- `down_proj_weight` 梯度 `d_down_proj_weight`；
- 路由权重参与加权归并时的 `d_topk_weights`；
- 其他不可导输入返回 `None`。

W2Grad 与激活梯度分支可并行执行，W1Grad 与 Combine(dX) 可并行执行，以充分利用通信和计算之间的独立性。
`````

Section 5 — the multicore scheduling and overlap policy, verbatim:

```
## 5 多核调度与性能目标

### 5.1 AIC/AIV 并发

- AllToAll 通信任务和 SwiGLU 任务映射到 AIV；
- GMM1、GMM2 及反向 GMM 任务映射到 AIC；
- AIC/AIV worker 独立轮询任务队列，按事件依赖放行；
- 任务粒度同时考虑首块延迟、专家 token 不均衡和 GMM 计算效率；
- 正向和反向均保持单次融合调用语义。

### 5.2 通信—计算重叠

- Dispatch 与已就绪专家的 GMM1 重叠；
- GMM1 与已就绪 tile 的 SwiGLU 重叠；
- GMM2 完成后尽早触发 Combine；
- 反向 W2Grad 与 act_grad 分支交错执行；
- 反向 W1Grad 与 Combine(dX) 重叠；
- 通信顺序避免多 rank 同时集中访问同一目标 rank。

### 5.3 运行时要求

- 任务配置与 tiling 可按 shape/dtype/topology 缓存并复用；
- 动态路由信息在每次调用时正确更新；
- 所有异步通信和跨流数据访问具备明确的完成或同步语义；
- 多次迭代、连续正反向和异常恢复后不得复用过期事件状态；
- 缓冲区容量与 workspace 大小由 host 侧计算并校验。
```

Section 6.3 and 6.4 — the `expert × block` and `expert × wave` granularity schemes. This is the
chunking/round scheme the megakernel question is really about:

```
### 6.3 GMM 专家计算分组粒度优化

将 GMM 就绪与调度粒度从单一专家级拓展为 `expert × block`，根据专家实际 token 数选择 block M。小批次或长尾专家使用较小 block 缩短首块等待，大批次使用较大 block 保持 GMM 效率。

### 6.4 Token Wave 优化

将专家 token 分成多个 wave，为每个 `expert × wave` 设置独立就绪事件。前缀 wave 到齐后立即启动对应 GMM M-tile，并将局部完成信号逐级传递到 SwiGLU、GMM2 和 Combine。尾 wave 的就绪阈值使用真实 token 数，避免等待不存在的补齐 token。
```

#### B6.6.3 arch35 (950) per-expert GEMM driving inside the ONE kernel

This is the arch35 tree, distinct from the arch22/A3 kernel. Two mechanisms matter.

#### (a) `ExpertLoopState` + `UpdateParamsForNextProblem` — re-point without re-Init

https://github.com/hicann/ops-transformer/blob/master/mc2/mega_moe/op_kernel/arch35/common/mega_moe_types.h
author/date: unknown — Huawei Technologies Co., Ltd., repo hicann/ops-transformer branch master, fetched 2026-10-20

```cpp
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    const MegaMoeTilingData* tilingData;
};

enum class AddrUpdateMode : int32_t {
    GMM1,
    GMM2
};

struct BlockJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};

// Count/flag workspace 的物理分区；当前与 BlockJobContext 同值，但其编号由 workspace 生产者和消费者共同约定。
struct BlockWorkspaceContext {
    uint32_t blockIdx;
    uint32_t blockNum;
};

template <typename T>
struct PackedElementTraits {
    static constexpr uint32_t ELEMENTS_PER_BYTE = Std::IsSame<T, fp4x2_e2m1_t>::value ? 2U : 1U;
};

template <int32_t QuantMode>
struct QuantTypeTraits {
    using Type = fp8_e4m3fn_t;
};

template <>
struct QuantTypeTraits<E5M2_QUANT> {
    using Type = fp8_e5m2_t;
};

template <>
struct QuantTypeTraits<E2M1_QUANT> {
    using Type = fp4x2_e2m1_t;
```

https://github.com/hicann/ops-transformer/blob/master/mc2/mega_moe/op_kernel/arch35/common/mega_moe_gmm_common.h
author/date: unknown — Huawei Technologies Co., Ltd., repo hicann/ops-transformer branch master, fetched 2026-10-20

GMM1 and GMM2 are built from the SAME `ProblemShape` by swapping which axis is N and which is K,
and GMM1's N is halved by `activationOutputDivisor = ACTIVATION_N_HALF` to express the gate/up
split. `L1_TILE_M_256` for GMM2, `gmm1TileM` (or `L1_TILE_M_128` under `TopkWeightsPrefetch`)
for GMM1.

```cpp

    __aicore__ static inline void FinalizeProblemConfig(ProblemConfig& config, const BlockJobContext& blockJob,
                                                        uint32_t gmm1TileM)
    {
        config.blockNum = blockJob.totalJobs;
        config.blockIdx = blockJob.jobIndex;
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        config.activationTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : gmm1TileM;
        config.blockMmadTiling = SelectBlockMmadTilingConfig(config.m, config.k, config.tileM);
    }

    __aicore__ static inline ProblemConfig BuildGmm1ProblemConfig(const ProblemShape& problemShape,
                                                                  const BlockJobContext& blockJob, uint32_t gmm1TileM,
                                                                  uint32_t activationOutputDivisor = ACTIVATION_N_HALF)
    {
        ProblemConfig config{};
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<N_VALUE>(problemShape);
        config.k = Get<K_VALUE>(problemShape);
        config.activationOutputDivisor = activationOutputDivisor;
        config.outputN = config.n / config.activationOutputDivisor;
        config.schedulerN = config.n;
        config.tileM = gmm1TileM;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        return config;
    }

    __aicore__ static inline ProblemConfig BuildGmm2ProblemConfig(const ProblemShape& problemShape,
                                                                  const BlockJobContext& blockJob, uint32_t gmm1TileM,
                                                                  uint32_t activationOutputDivisor = ACTIVATION_N_HALF)
    {
        ProblemConfig config{};
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<K_VALUE>(problemShape);
        config.activationOutputDivisor = activationOutputDivisor;
        config.k = Get<N_VALUE>(problemShape) / config.activationOutputDivisor;
        config.outputN = config.n;
        config.schedulerN = config.outputN;
        config.tileM = L1_TILE_M_256;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        return config;
    }
```

The per-expert re-point. `blockMmad.Init(...)` is guarded by an `initConfig_` equality check so
it runs only when the L1 tiling actually changes; every expert then only calls
`UpdateParamsForNextProblem`. That is exactly the "one matmul instance, many problems" trick a
hand-written megakernel needs, and it is the arch35 answer to the A2/A3 design doc's complaint
that you cannot instantiate one matmul object per group (section 2.5).

```cpp
template <typename BlockMmad, typename DispatchPolicy = typename BlockMmad::DispatchPolicy>
struct BlockMmadContext;

template <typename BlockMmad>
struct BlockMmadContext<BlockMmad, Blaze::Gemm::GroupedMatmulWithScaleMx<>> {
    BlockMmad blockMmad;

    template <GmmStage Stage, typename ProblemConfig>
    __aicore__ inline void Init(const ProblemConfig& config)
    {
        const auto& tiling = config.blockMmadTiling;
        // GMM1 门控激活按半宽初始化拼接入口；GELU 和 GMM2 使用普通 MMAD 的完整列宽。
        const uint32_t tileN = Stage == GmmStage::GMM1 ? tiling.tileN / config.activationOutputDivisor : tiling.tileN;
        const BlockMmadInitConfig requestedConfig{config.k, tiling.tileM, tileN, tiling.l1Params.kAL1,
                                                  tiling.l1Params.scaleKL1};
        if (!initialized_ || initConfig_ != requestedConfig) {
            typename BlockMmad::ProblemShape problemShape{config.m, config.n, config.k, 0};
            typename BlockMmad::MmadParams params{
                {tiling.tileM, tileN, L0_TILE_K, 0}, tiling.l1Params, false, false, 2U};
            blockMmad.Init(problemShape, params);
            initConfig_ = requestedConfig;
            initialized_ = true;
        }
        blockMmad.UpdateParamsForNextProblem({config.m, config.n, config.k, 0});
    }

private:
    BlockMmadInitConfig initConfig_{};
    bool initialized_ = false;
};
```

And the L2 hint rule, with its exact precondition spelled out in the comment — the hot-expert
case is explicitly excluded:

```cpp

/*
 * CACHE_MODE_DISABLE 会使读取不填入 L2，仅适用于后续不会复用的数据。
 * 当前 problem 必须覆盖完整专家且 M 方向只有一个 tile，B/scaleB 才不会被后续 M tile 或 Wave 再读。
 * 热点专家被切到多个 Wave 时必须保留正常 L2 cache。
 * B 与 scaleB 的物理行跨度不同，二者必须独立判定 128B cache-line 对齐，不能共用一个结果。
 */
template <bool IsWeightNz, typename MatmulConfig, typename TensorB, typename TensorScaleB>
```

#### (b) How the groupList itself is produced ON DEVICE: 24-bit count + 8-bit round tag

https://github.com/hicann/ops-transformer/blob/master/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_expert_count.h
author/date: unknown — Copyright (c) 2026 Huawei Technologies Co., Ltd., repo hicann/ops-transformer branch master, fetched 2026-10-20

There is no host-supplied groupList in MegaMoE. Each remote rank writes its per-expert token
count into a 512-byte GM slot; the HIGH 8 BITS of each count word carry a ROUND TAG (epoch), so
the consumer can distinguish "this iteration's count" from a stale one WITHOUT zeroing the table
and WITHOUT a barrier. The consumer polls the whole table, checks all tags in one vector pass,
and backs off 500 cycles on mismatch. Then a single fused VF pass does an inclusive prefix sum
masked to the low 24 bits (`InclusivePrefixSumInt32<0x00FFFFFF>`) and, in the same VF call,
differences the per-expert tail values to recover per-expert counts — i.e. it produces BOTH the
cumsum form (groupListType 0) and the count form (groupListType 1) in one pass.

```cpp

// count 整表检查失败后的重试间隔。
constexpr int64_t COUNT_TABLE_POLL_BACKOFF_CYCLES = 500;

// 只检查 UB 快照的高 8 位轮次标签，不修改 count；成功后由前缀和计算去除标签。
// expectedSyncRoundTag 使用 GetSyncRoundTag 返回的高 8 位标签；pending[0] 为零时才可消费快照。
// pending 的各 lane 累积所有向量的失败结果，尾部无效 lane 不参与比较或写回。
__simd_vf__ inline void CheckCountRoundTagsVF(__ubuf__ uint32_t* counts, __ubuf__ uint32_t* pending,
                                              uint32_t elementCount, uint32_t expectedSyncRoundTag)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = AscendC::GetVecLen() / sizeof(uint32_t);
    AscendC::Reg::RegTensor<uint32_t> countReg;
    AscendC::Reg::RegTensor<uint32_t> epochReg;
    AscendC::Reg::RegTensor<uint32_t> pendingReg;
    AscendC::Reg::RegTensor<uint32_t> oneReg;
    AscendC::Reg::RegTensor<uint32_t> resultReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg firstLaneMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::Duplicate(pendingReg, 0U, fullMask);
    AscendC::Reg::Duplicate(oneReg, 1U, fullMask);
    for (uint32_t offset = 0U; offset < elementCount; offset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = elementCount - offset;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<uint32_t>(validCount);
        AscendC::Reg::MaskReg mismatchMask;
        AscendC::Reg::LoadAlign(countReg, counts + offset);
        AscendC::Reg::ShiftRights(epochReg, countReg, static_cast<int16_t>(24), activeMask);
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::NE>(mismatchMask, epochReg, expectedSyncRoundTag >> 24U,
                                                               activeMask);
        AscendC::Reg::Select(pendingReg, oneReg, pendingReg, mismatchMask);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(resultReg, pendingReg, fullMask);
    AscendC::Reg::StoreAlign(pending, resultReg, firstLaneMask);
}

// MTE GM count 表按 512B 槽存放；搬入后的 counts 为连续 UB 表。
```

```cpp
// pending 为独立、32B 对齐的 UB 暂存区（至少一个 int32）。
// 首次立即读取；失败后等待 500 cycle，再重读整表。成功快照直接交给后续 VF cumsum。
// 调用方保证在 AIV 上执行，且 elementCount > 0。
__aicore__ inline void WaitForCountTable(const AscendC::GlobalTensor<int32_t>& countGm,
                                         const AscendC::LocalTensor<int32_t>& counts,
                                         const AscendC::LocalTensor<int32_t>& pending, uint32_t elementCount,
                                         uint32_t expectedSyncRoundTag)
{
    // 首次搬运前保护此前的 Vector 访问；重试时上一轮 V_S 和 Scalar 判断已保证 VF 完成。
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
    while (true) {
        // Compact 将每个 GM 槽首部的 4B 紧密排入 UB，保持后续 epoch 校验和前缀和布局不变。
        AscendC::DataCopyPad<int32_t, AscendC::PaddingMode::Compact>(
            counts, countGm,
            {static_cast<uint16_t>(elementCount), sizeof(int32_t),
             static_cast<int64_t>(MTE_COUNT_SLOT_BYTES) - static_cast<int64_t>(sizeof(int32_t)), 0U, 0U},
            {true, 0U, 0U, 0U});
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        asc_vf_call<CheckCountRoundTagsVF>(reinterpret_cast<__ubuf__ uint32_t*>(counts.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ uint32_t*>(pending.GetPhyAddr()), elementCount,
                                           expectedSyncRoundTag);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
        if (pending.GetValue(0U) == 0) {
            return;
        }
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < COUNT_TABLE_POLL_BACKOFF_CYCLES) {
        }
    }
}

```

```cpp
/*
 * MegaMoe count 表的融合版本：先复用 int32 prefix-scan，再在同一次 VF 中提取每个专家的累计尾值
 * 并作差。相比先调用通用前缀和、再启动第二个 VF，融合实现省去一次 asc_vf_call 和外部流水同步。
 */
__simd_vf__ inline void ComputeExpertCountTablesVF(__ubuf__ int32_t* count, __ubuf__ int32_t* expertCounts,
                                                   uint32_t elementCount, uint32_t expertCount, uint32_t worldSize,
                                                   uint32_t maxOutputSize)
{
    // 去除快照中的高 8 位轮次标签，只累加低 24 位 count。
    InclusivePrefixSumInt32<0x00FFFFFF>(count, elementCount);

    // prefix 已写回 UB；后续从同一区域 Gather 前建立 VEC_STORE -> VEC_LOAD 可见性。
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    constexpr uint32_t ELEMENTS_PER_VECTOR = GetVecLen() / sizeof(int32_t);
    AscendC::Reg::RegTensor<int32_t> expertEndPrefixReg;
```

#### B6.6.4 Complete `mc2/mega_moe` file inventory (for navigation)

https://github.com/hicann/ops-transformer/tree/master/mc2/mega_moe
author/date: repo hicann/ops-transformer branch master, fetched 2026-10-20

The arch35 tree is organised as explicit named STAGES, which is itself the design statement:

```
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_expert_count.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine_dedup.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_layered_combine.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_layered_combine_send.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_layered_dispatch.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_send_mask.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dedup.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch_dedup.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_quant.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_unpermute.h
mc2/mega_moe/op_kernel/arch35/stage/mega_moe_workspace_reset.h
mc2/mega_moe/op_kernel/arch35/mega_moe_arch35.h           (64 KB, the driver)
mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h
mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a4w4.h
mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w4.h
mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h
mc2/mega_moe/op_kernel/arch35/blaze/gemm/policy/dispatch_policy_mega_moe.h
mc2/mega_moe/op_kernel/arch35/blaze/gemm/block/block_mmad_mx_fp8fp4.h
mc2/mega_moe/op_kernel/arch35/blaze/gemm/block/block_scheduler_swizzle.h
mc2/mega_moe/op_kernel/arch35/blaze/gemm/tile/copy_gmm1_concat.h
mc2/mega_moe/op_kernel/arch35/blaze/epilogue/block_epilogue_activation_mx_quant.h
mc2/mega_moe/op_kernel/arch35/common/{mega_moe_constants,mega_moe_gmm_common,
  mega_moe_gmm_epilogue_sync,mega_moe_layered_comm,mega_moe_mxfp8_utils,
  mega_moe_peermem,mega_moe_types,mega_moe_utils,mega_moe_workspace}.h
mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp      (116 KB, the A3 driver)
mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a2_bf16.hpp
mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a2_bf16_w4a8.hpp
mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a2_int8.hpp
mc2/mega_moe/op_kernel/arch22/moe_permute_prologue/moe_permute_prologue.hpp
mc2/mega_moe/op_kernel/arch22/unpermute/mc2_mega_moe_moe_token_unpermute.h
mc2/mega_moe/op_kernel/arch22/utils/{block_epilogue_pertoken_swiglu,
  block_mmad_preload_async_fixpipe_quant,block_mmad_w4a4,const_args,
  copy_gm_to_l1_custom,copy_l0c_to_gm_custom,dispatch_policy_custom,
  gated_activation,get_tensor_addr,hccl_shmem,layout3d,select_helper,
  weight_l2_prefetch}.hpp
mc2/mega_moe/docs/torchapi_mega_moe.md
mc2/mega_moe/README.md
mc2/mega_moe/op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp
mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp
mc2/tools/dump_analysis/dump_analysis_megamoe.py
```

---

### B6.7 Summary of real repo paths (for the record)

| what | where |
| --- | --- |
| every `GroupedMatmul*` op | `cann/ops-transformer` (gitcode) == `hicann/ops-transformer` (GitHub), dir `gmm/` |
| MC2 grouped-matmul + comm fusion | same repo, dir `mc2/` |
| MegaMoE | same repo, `mc2/mega_moe/` |
| CATLASS grouped templates | `hicann/catlass` (GitHub), `include/catlass/gemm/kernel/grouped_*` |
| `cann/ops-nn` / `hicann/ops-nn` | NO grouped matmul. Negative result, see section 5. |
| `cann/cann-ops-adv`, `cann/cann-ops`, `hicann/ops-nn` on gitcode | 404 |

### gitcode API notes (useful if this has to be redone)

- `https://gitcode.com/api/v5/repos/<owner>/<repo>/git/trees/<sha-or-branch>` works unauthenticated
  but DEFAULTS TO 20 ENTRIES. Pass `?per_page=1000`. `?recursive=1` works but is still paginated at
  20 per page and `?recursive=true` errors with `参数类型错误`.
- `https://gitcode.com/api/v5/repos/<o>/<r>/contents/<path>` → 404 even for real paths.
- Raw: `https://raw.gitcode.com/<o>/<r>/raw/master/<path>` works, but more than ~3 concurrent
  requests returns HTTP 429 for minutes. Serialize.
- The `hicann/*` GitHub mirrors are complete, current (pushed same day) and have no such limits.
  Use `gh api repos/hicann/ops-transformer/git/trees/master?recursive=1` (20897 entries,
  `truncated=false`) and `gh api repos/<r>/contents/<p> -H 'Accept: application/vnd.github.raw'`.



## Gaps

What was looked for and not found.

- **No published per-call cost for `GroupedMatmulSwigluQuantV2` at batch=1 on A3.** The closest
  numbers found are: W4A8 `GroupedMatmul` at 65.9 us mean / 61.6 us median per call on A3 TP16
  decode (PR #16606), and the A3 MSD GMSQ at 167-284 us per call for 260-800 valid rows
  (PR #17201) — both far above batch=1 row counts, so neither pins the fixed cost at M=8.
- **The A3 (`arch22`) `grouped_matmul` and `grouped_matmul_swiglu_quant_v2` tiling functions
  were not read.** The MSD pipeline header and the `WorkSpaceSplitConfig` helpers were read, but
  `op_host/*_tiling.cpp` — which is where `mLimit`, `baseM`/`baseN` and the core split are
  actually chosen for A8W4 — was not opened. Without it, the claim that `baseM` is sized for
  throughput at batch=1 rests on the design doc's worked example (`baseM = 24`) rather than on
  the A8W4 code path.
- **No Zhihu or CSDN article found specifically on batch=1 / single-token MoE decode kernels on
  Ascend.** Repeated Chinese-phrased searches returned the English literature (MonoMoE, Cohere,
  MPK) plus Huawei's own op articles, but no community write-up on the batch=1 MoE regime for
  Ascend. The Huawei articles quoted above are the closest, and they address decode generally
  rather than batch=1.
- **No Ascend-side measurement of the MSD double-matmul cost in isolation.** Huawei's article
  gives the end-to-end claim (W4A8+MSD decode is 1.3-1.7x W8A8) but nothing isolates what the
  two INT4 MMADs plus the `C_high*16 + C_low + weightAssistMatrix` vector pass cost relative to
  a single A8W8 MMAD at small M. That number would say whether MSD is worth keeping at batch=1
  or whether expanding W4 to W8 on-chip is cheaper there.
- **`GroupedMatmulFinalizeRouting` has no published A3 performance data**, and no call site for
  it was found in vllm-ascend — so whether the down-GEMM + combine fusion is actually reachable
  from our stack today is unverified.
- **No documentation found for `tuning_config` beyond "expected tokens per expert".** The doc
  says elements after the first are reserved, and that the parameter is only honoured for
  A4W4-NZ; nothing describes what the A4W4 path does with it, which would tell us what template
  switch we would be asking for on A8W4.
- **DeepEP low-latency buffer sizing not captured.** `Buffer.get_low_latency_rdma_size_hint` and
  `num_rdma_bytes` were identified as the relevant API but their values and the V2 buffer-size
  change were not retrieved verbatim. This matters only if we fuse the TP16 collective, which
  MonoMoE's measurement argues against.
- **No source found that measures `CrossCoreSetFlag`/`CrossCoreWaitFlag`/`SyncAll` cost on
  Ascend.** The GPU analogue is quantified (Hazy Research: ~60 ns per CUDA async barrier even in
  the pass state; 40 of 600 us total); without the Ascend number, the barrier budget for a
  78-layer megakernel cannot be estimated.
- **Cohere's megakernel blog (https://cohere.com/blog/megakernels) could not be fetched** from
  this network, so its claims (a 30B MoE decode loop in one persistent kernel, counters in
  global memory instead of launch boundaries) are unverified and not quoted.
- **`MonoMoE`'s GLM-5.2 configuration was not located.** The paper reports GLM-5.2 TPOT results
  but the shipped FlashInfer kernel is specialized to `E256/N512/K2048`; the GLM-5.2
  specialization (which would have N=1536-class and K=6144) was not found in the repo tree.

## B7. ★ arXiv 2605.13915 "Multi-Scale Dequant" — one of four things CANN calls MSD, and NOT ours; read B7.-1 first. Its value to us is the M-stacked single-GEMM form, published as equation (16)

"Multi-Scale Dequant: Eliminating Dequantization Bottleneck via Activation Decomposition for Efficient
LLM Inference", Lingchao Zheng, Yuwei Fan (corresponding, fanyuwei2@huawei.com), Jun Li, Chengqiu Hu,
Qichen Liao, Junyi Fan, Rui Shi, Fangzheng Miao. arXiv:2605.13915v1 [stat.ML], dated 13 May 2026 on the
listing and "May 15, 2026" on the title page. 36 pages. https://arxiv.org/abs/2605.13915 /
https://arxiv.org/pdf/2605.13915 (the HTML render was not available; the PDF was fetched and text-extracted
with `pypdf`). Read 2026-10-20.

This is a Huawei paper, and it is the public writeup of the technique whose name appears in
`grouped_matmul_swiglu_quant_a8w4_msd_pre/mid/post.h`. **Read the first subsection before acting on anything
here: the expansion of the acronym is not what we assumed, and the published instantiations are not ours.**

### B7.-1 ★★ CORRECTION, added after this section was written: "MSD" names four different decompositions in CANN, and this paper is not the one we are building

Established independently and after the rest of this section was drafted, from
https://www.hiascend.com/developer/techArticles/20250717-2 (2025/07/17 — note that the
`developer/techArticles/` path serves full HTML to a fetcher, unlike `document/detail/`), with the
taxonomy verified against local clones of `gitcode.com/cann/ops-transformer` and `gitcode.com/cann/ops-nn`.
The full write-up is in `w4a8-msd-int4-mmad.md` §1-§7; this note exists so that nobody reads B7.0 below and
concludes that our `16*high + low` is unfounded. It is founded; it just is not this paper.

For the A8W4 case — ours — the authoritative expansion is **Mixed-precision Split-activation Decomposition**,
verbatim:

> 创新性地提出MSD（Mixed-precision Split-activation Decomposition，混合精度激活拆分分解）方案，将INT8激活拆分为
> 高、低两路INT4，并直接与INT4权重进行计算，用“高位×16+低位”合成

So our reading of the *mechanism* was right all along — split the INT8 activation into a high and a low INT4
path, multiply each directly against the INT4 weight, recombine as 「高位×16+低位」. What was wrong was only the
expansion of the acronym, and the assumption that one paper covers it.

The four families, all verified in-tree:

| # | Family | Activation | Components | Radix / scales | Cube out | Where |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **A8W4 MSD** — OURS | INT8 | 2 × int4 nibbles | 16 | `half` | `grouped_matmul*/..._a8w4_msd*`, `ops-nn/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_msd.h` |
| 2 | A16W8 MSD | BF16/FP16 | 2 × int8 digits | 127.499, then 127.499²·2 | `int32` | `ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_*.h` |
| 3 | A16W4 MSD | BF16/FP16 | 2 × int8 | `float multiFactors[2] = {127.0f, 254.0f}` | `int32` (`LocalTensor<int32_t> s32l0c_`) | `ops-transformer/gmm/grouped_matmul/op_kernel/a16w4_msd/` |
| 4 | SFA antiquant MSD | — | — | `antiqCoeff1 = 127`, `scaleC1 = 1024.0/127`, `scaleC2 = 1024.0` | — | SparseFlashAttention |

**arXiv 2605.13915, everything below in this section, is families 2-4 — not family 1.** Search `a8w4_msd`,
never bare `msd`, or you land in the wrong family. Two practical notes from the same pass: the non-grouped W4A8
MSD matmul lives in `ops-nn`, not `ops-transformer`, and its kernel carries the comment
`// 2: in pre process, x has been splited to [2m,n]` — **the vendor M-stacks it too, and says so in a comment**,
independently of equation (16) below. And `git clone`/`git ls-remote https://gitcode.com/cann/ops-transformer.git`
works, whereas the `gitcode.com/.../raw/master/...` form returns a JavaScript shell to a fetcher — clone rather
than fight the raw-URL rate limit.

**Our `8*sum(w)` fold is also published formula rather than our own invention.** From
`gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md`, in-repo:
`weightAssistMatrix ∈ R^{E×N}`, 「离线生成作为输入，并非算子内部完成」, with
`weightAssistMatrix_i = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]` per-channel, or
`8 × Σ_k (weight[:,k,:] × weightScale[:, ⌊k/num_per_group⌋, :])` per-group; the split is
`X_high_4bits = ⌊X/16⌋`, `X_low_4bits = X & 0x0f − 8`, and the recombine is
`C = (C_high×16 + C_low + weightAssistMatrix) ⊙ x_scale`. The same quantity ships under three names in three
operators: `weightAssistMatrix` (GroupedMatmulSwigluQuant), `bias` (aclnnGroupedMatmulV4 — the kernel member is
literally `GlobalTensor<DTYPE_BIAS_A8W4MSD> biasGm; // for 8 * weight`), and `yOffset` (aclnnQuantMatmulV5,
「值要求为8*x2*x2Scale，并在第1维累加」). It is generated **offline on the host**, not inside the operator.

### B7.0 ★ Three corrections to the premise we were working from

1. **MSD expands to "Multi-Scale Dequant", not "most-significant-digit" or "multiple significant digit".**
2. **The framework decomposes the ACTIVATION, never the weight, and its published instantiations all take a
   BF16 activation.** The abstract, verbatim:

   > This paper presents **Multi-Scale Dequant (MSD)**, a quantization framework that removes weight/KV
   > dequantization from the GEMM critical path. Instead of lifting low-bit weights to BF16 precision, MSD
   > decomposes high-precision BF16 activations into multiple low-precision components, each of which can be
   > multiplied directly with quantized weights via native hardware-accelerated GEMM. This approach shifts the
   > computational paradigm from *precision conversion* to *multi-scale approximation*, avoiding INT8-to-BF16
   > weight conversion before GEMM.

   The paper's own grouped-matmul reference kernel is `a16w4_msd` — **A16**, a BF16 activation against an INT4
   weight — not `a8w4`. So the `a8w4_msd` kernel family is an instantiation this paper does not analyse. Our
   reading of *what* those kernels do (split the activation into two INT4 components and recombine) matches the
   framework's principle exactly; what does not match is the precision of the thing being split and the number
   of components.
3. **For an INT4 weight the paper's own recommendation is K = 3 INT4 passes from BF16, not K = 2.** Table 4,
   verbatim and complete:

   > Table 4: MSD applicability to various precision configurations. K denotes the number of decomposition
   > scales. Eff. Bits and Error Bound are for the K shown.
   >
   > | Activation | Weight | MSD Decomposition | K | Eff. Bits | Error Bound |
   > | --- | --- | --- | --- | --- | --- |
   > | BF16 | INT8 | BF16→INT8 + INT8 | 2 | ∼16 | M/64516 |
   > | BF16 | MXFP4 | BF16→MXFP4 + MXFP4 | 2 | ∼6.6 | α/64 |
   > | BF16 | FP8 | BF16→FP8 + FP8 | 2 | ∼16 | TBD |
   > | BF16 | INT4 | BF16→INT4 + INT4 + INT4 | 3 | ∼11 | TBD |
   >
   > Baselines for comparison:
   >
   > | BF16 | INT8 | Dequant (BF16×BF16) | — | ∼8 | — |
   > | BF16 | MXFP4 | MXFP8 activation | — | ∼5.24 | — |

   and §3.4, verbatim:

   > **BF16→INT4:** When decomposing BF16 activations to INT4 components, two scales may not fully capture the
   > dynamic range. We can use K = 3 or K = 4 to progressively refine the residual.

   **There is no A8W4 row in Table 4 at all.** This is consistent rather than contradictory for our case: two
   signed-INT4 components cover an INT8 range exactly (`16*high + low` with `high, low ∈ [-8,7]` is an exact
   identity over the int8 range, so K = 2 from an *already-INT8* activation is lossless, whereas K = 2 from a
   *BF16* activation is not and needs a third pass). But it means the paper's error bounds and effective-bit
   figures do not apply to what we are building, and nothing in it validates a two-component INT4 split of a
   BF16 activation.

### B7.1 ★ The M-stacked single-GEMM form is published, is equation (16), and is recommended for exactly our regime

This answers the question we asked directly — whether anyone concatenates the two components along M rather
than issuing two matmuls. They do, and small batch is the stated reason. §3.9, verbatim:

> **GEMM pass fusion for small-batch decode.** The MSD decomposition produces two activation components
> X⁽¹⁾, X⁽²⁾ ∈ R^(b×n), requiring two separate GEMM calls: WX⁽¹⁾ and WX⁽²⁾. However, when b is small—as is
> typical in decode where b ≪ m, n—the two GEMM passes can be **fused into a single GEMM call** by
> concatenating the components along the row dimension:

```
[ α Y⁽¹⁾ ]   [ α X⁽¹⁾ ]
[        ] = [        ] · Wᵀ ,                                   (16)
[ β Y⁽²⁾ ]   [ β X⁽²⁾ ]
```

> where [α X⁽¹⁾ ; β X⁽²⁾] ∈ R^(2b×n) and the result is a 2b×m matrix from which α Y⁽¹⁾ and β Y⁽²⁾ are
> extracted and summed. This reduces two kernel launches to one, eliminates inter-kernel synchronization, and
> improves Cube utilization. When b is large (e.g., large-batch prefill), this concatenation may exceed
> on-chip capacity, and the two GEMM passes must be computed separately.

and the same point in the Figure 1 caption, verbatim:

> When the left matrix has few rows (e.g., decode with b ≪ m, n), the two GEMM passes can be further fused
> into a single GEMM by concatenating X⁽¹⁾ and X⁽²⁾ along the row dimension (see text).

One notation wrinkle to be aware of rather than to copy: as written, the left operand of (16) is
`[α X⁽¹⁾ ; β X⁽²⁾]`, which cannot be an integer matrix if α and β are reals. The integer kernel must
concatenate the raw components `[X⁽¹⁾ ; X⁽²⁾]` and apply α and β to the two output row-blocks, which is what
the surrounding prose describes ("from which α Y⁽¹⁾ and β Y⁽²⁾ are extracted and summed").

### B7.2 The exact decomposition math (W8A16 instantiation), and the post-GEMM scale licence

§3.1, verbatim:

> Consider a linear layer with weight matrix W ∈ Z₈^(m×n) (quantized to INT8 offline with per-channel scales
> s_W ∈ R^m) and activation vector x ∈ R^n (in BF16 format during inference). The standard
> dequantization-based approach computes:
>
>     y = dequant(W)·x = (s_W ⊙ W_int8)·x,                                      (3)
>
> where the weight matrix is first dequantized from INT8 to BF16 via per-channel scaling, then multiplied with
> the BF16 activation. This dequantization step is the bottleneck we aim to eliminate.
>
> MSD takes a different approach: instead of dequantizing W from INT8 to BF16, we decompose the BF16
> activation x into K INT8 components:
>
>     x ≈ Σ_{k=1..K} α_k · x⁽ᵏ⁾,   x⁽ᵏ⁾ ∈ Z₈^n,                                 (4)
>
> where α_k ∈ R are learned or computed scaling coefficients. The output is then computed as:
>
>     y = Σ_{k=1..K} α_k · (W x⁽ᵏ⁾).                                            (5)
>
> Each W x⁽ᵏ⁾ is a native INT8×INT8 GEMM, executable directly on hardware tensor cores without
> dequantization. Since the weight scale s_W is per-output-channel (i.e., each row of W shares a single
> scale), it can be applied **after** the GEMM and reconstruction:
>
>     y = s_W ⊙ ( Σ_{k=1..K} α_k · (W_int8 x⁽ᵏ⁾) ),                             (6)
>
> where ⊙ denotes row-wise scaling. This is a lightweight O(m) Vector operation, analogous to how V's
> per-channel scale is applied after the PV GEMM in attention (Section 4). In practice, we find K = 2 provides
> an excellent trade-off between accuracy and computational cost.

§3.2, the two-pass algorithm, verbatim:

> For K = 2, we use a two-pass decomposition analogous to multi-grid correction in numerical analysis. The
> algorithm proceeds as follows:
>
> **Pass 1: Coarse-Scale Quantization.** Compute the primary scale and quantized activation:
>
>     α = ‖x‖_∞ / 127,                                                          (7)
>     x⁽¹⁾ = clamp( round( x / α ), −128, 127 ).                                (8)
>
> **Pass 2: Fine-Scale Residual.** Compute the residual. Since the quantization error of Pass 1 is bounded in
> (−0.5α, 0.5α), we directly use 2 × 127 = 254 as the secondary scale without computing max:
>
>     r = x − α · x⁽¹⁾,                                                         (9)
>     β = α / 254,                                                              (10)
>     x⁽²⁾ = clamp( round( r / β ), −128, 127 ).                                (11)
>
> The final approximation is x ≈ α·x⁽¹⁾ + β·x⁽²⁾. The decomposition is performed on-the-fly for each
> activation vector during inference, with negligible overhead compared to the subsequent GEMM operations.

So the recombination weight on the second pass is **1/254**, not 1/16 — because the components are INT8 and
the second scale is derived from the first pass's half-LSB bound, not from a nibble split. The `16*high + low`
form is the INT4-component algebra, which the paper only reaches in the MXFP4 instantiation (where
`β = α/16`, §3.2.1 Algorithm 1 line 6) and in Table 4's K = 3 INT4 row.

The MXFP4 instantiation, Algorithm 1, verbatim and complete — this is the closest published analogue of a
two-INT4-component scheme:

```
Algorithm 1 MSD MXFP4 Decomposition (per 32-element block)
Require: Block x_b ∈ R³² (BF16), Weight block W_b (MXFP4)
Ensure:  Decomposed components q₁, q₂ ∈ FP4³², scales α, β ∈ E8M0
 1: M_b ← ‖x_b‖_∞
 2: α ← 2^⌈log2(M_b / 1.859375)⌉           ▷ E8M0 scale (power of two)
 3: s ← x_b / α                            ▷ Scale to FP4 range
 4: q₁ ← round_to_FP4(s)                   ▷ Round-to-nearest on E1M2 grid; |s| > 1.75 maps to ±1.75
 5: r ← x_b − α · q₁                       ▷ Residual; ‖r‖_∞ ≤ α/8
 6: β ← α / 16                             ▷ E8M0 scale; no max-reduction needed
 7: q₂ ← round_to_FP4(r / β)               ▷ 12.5% elements clipped to ±1.75; error ≤ α/64
 8: return q₁, q₂, α, β
```

with the design note, verbatim: 「choosing β as powers of two is therefore deliberate: it ensures the entire
decomposition pipeline—scale, …—」 (§3.2.1) and 「the scaled residual satisfies |r_i/β| ≤ 2, which exceeds
FP4's 〔range〕」.

### B7.3 The error bounds, as theorems — and the answer on exactness

Theorem 5.1, verbatim:

> **Theorem 5.1 (Multi-Scale Reconstruction Error).** Let x ∈ Rⁿ with ‖x‖_∞ = M. Under the two-pass
> decomposition in Algorithm 2, the reconstruction error satisfies:
>
>     ‖x − (α x⁽¹⁾ + β x⁽²⁾)‖_∞ ≤ M / (127 × 2 × 254) = M / 64516 ≈ M / 2¹⁶.     (32)
>
> *Proof.* After the first quantization pass (Eqs. (7)–(8)), the per-element rounding error is bounded by:
> |x_i − α x⁽¹⁾_i| ≤ α/2 = M/(2 × 127). (33) Therefore, the residual satisfies ‖r‖_∞ ≤ M/254. Since the
> quantization error in Pass 1 is bounded in (−0.5α, 0.5α), we directly use β = α/254 as the secondary scale
> without computing max. The second-pass rounding error satisfies:
> |r_i − β x⁽²⁾_i| ≤ β/2 = α/(2 × 254) = M/(127 × 2 × 254) = M/64516 ≈ M/2¹⁶. (34)
> Since r_i = x_i − α x⁽¹⁾_i, we have |x_i − (α x⁽¹⁾_i + β x⁽²⁾_i)| ≤ M/64516 ≈ M/2¹⁶ for all i, establishing
> the bound. ∎

> **Corollary 5.1 (Effective Precision Gain).** Standard single-scale INT8 quantization achieves error bound
> M/254 ≈ M/2⁸. MSD with K = 2 achieves M/64516 ≈ M/2¹⁶, providing approximately **8 additional effective
> bits** of precision (from ∼8 to ∼16 effective bits). With fractional scaling (α = M/127.49, Section 3.3),
> the bound tightens to M/65015, which is closer to 2¹⁶.

Theorem 5.2, verbatim:

> **Theorem 5.2 (MXFP4 Multi-Scale Reconstruction Error).** Let x ∈ R³² be a 32-element block with
> ‖x‖_∞ = M_b. Under the two-pass MXFP4 decomposition with α = 2^⌈log2(M_b/1.859375)⌉ and β = α/16, the
> reconstruction error satisfies:
>
>     ‖x − (α q₁ + β q₂)‖_∞ ≤ α / 64.                                           (35)
>
> *Proof.* The FP4 E1M2 format represents positive values in {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75} with a
> **uniform** step size of 0.25 (including the transition from 1.0 to 1.5, which is 0.5 = 2 × 0.25, since the
> exponent increment doubles the step).
> **Pass 1 residual bound.** After scaling by α, the elements satisfy |x_i/α| ≤ 1.859375. We consider two
> cases: *Normal quantization* (|x_i/α| ≤ 1.75): the rounding error is at most half the step size:
> |x_i/α − q₁,i| ≤ 0.125, so |r_i| ≤ 0.125α = α/8. *Clipped elements* (|x_i/α| ∈ (1.75, 1.859375]): the value
> is mapped to ±1.75 via round-to-nearest. The residual satisfies
> |r_i| = |x_i/α − 1.75|·α ≤ (1.859375 − 1.75)α = 0.109375α < α/8. Therefore, the global Pass 1 residual bound
> is ‖r‖_∞ ≤ α/8.
> **Pass 2 error bound.** With β = α/16, the scaled residual satisfies |r_i/β| ≤ (α/8)/(α/16) = 2. Again two
> cases: *Normal quantization* (|r_i/β| ≤ 1.75, approximately 87.5% of elements): rounding error
> ≤ 0.125β = α/128. *Clipped elements* (|r_i/β| ∈ (1.75, 2], approximately 12.5% of elements): residual mapped
> to ±1.75, error ≤ (2 − 1.75)β = 0.25β = α/64. The worst-case per-element error is therefore α/64,
> establishing the bound. ∎

> **Corollary 5.2 (Effective Precision across Formats).** … **INT8 (W8A16):** Standard single-pass INT8
> quantization achieves ∼8 effective bits. MSD with K = 2 achieves M/64516 ≈ M/2¹⁶, providing ∼16 effective
> bits—an 8-bit gain. **MXFP4 (W4A16):** Standard single-pass MXFP4 quantization achieves ∼2.8 effective bits.
> MSD with K = 2 achieves error bound α/64 per block; since α/M_b ≤ 1.859375, the relative error is at most
> 1.859375/64 ≈ 0.029, yielding ∼6.6 effective bits—a 3.8-bit gain over single-pass MXFP4 and 1.4 bits beyond
> single-pass MXFP8 (∼5.24 bits).
>
> For comparison, BF16 has 7 explicit mantissa bits plus implicit leading 1, giving roughly 8 effective bits of
> precision for normalized numbers. MSD's two-pass INT8 decomposition approaches BF16 fidelity while using
> only INT8 operations throughout the compute-intensive GEMM.

On accuracy cost overall, the abstract, verbatim:

> Numerical simulations on matrix multiplication and Flash Attention kernels confirm that MSD does not degrade
> accuracy compared to dequantization baselines, and in many settings achieves lower L2 error.

**So: the MSD decomposition is approximate by construction when the source is BF16, with a proved L∞ bound,
and the published verdict is "no accuracy degradation versus dequant, often better".** It is not presented as
an exact identity anywhere. Our INT8→two-INT4 case is a different animal: there the split *is* exact, and the
paper offers no bound for it because it never considers an integer source.

### B7.4 ★ Ascend-specific mechanics worth lifting: FixPipe does the recombination for free

§4.4.3 (the optimization list), verbatim — this is directly applicable to our `16*high + low` merge:

> **Cube-side FixPipe on Ascend.** Ascend's Cube core features a *FixPipe* (fixed-point pipeline) unit that
> performs inline post-processing on GEMM outputs **before** they leave the Cube. Specifically, FixPipe can:
> (1) cast INT32 accumulator results to FP16/BF16, (2) multiply by a scalar coefficient, and (3) atomically
> accumulate into global memory—all in a single pass with no Vector involvement. This maps directly onto MSD's
> merging step: the two partial GEMMs W x⁽¹⁾ and W x⁽²⁾ (with INT32 outputs) can each be scaled by α and β
> respectively and accumulated into the final output buffer via FixPipe's atomic add, completely bypassing the
> Vector core for the reconstruction phase. This effectively reduces MSD's Vector overhead to **only** the
> decomposition step, making the merging cost zero from the Vector perspective.
>
> With these optimizations, the effective Vector cost of MSD can be reduced by 30–50%, pushing the crossover
> point N* significantly higher and making MSD beneficial for an even wider range of decode configurations.

and, in the same list:

> **Low-precision decomposition.** The MSD decomposition (round, clamp) and S/O merging (scale, add) can be
> performed in FP16 or even INT16 instead of FP32, reducing Vector instruction count and register pressure.

§3.9's four design principles for the fused tiled kernel, verbatim:

> 1. **Resident weight/KV tile.** Each weight or KV tile is loaded from HBM **once** into on-chip buffer and
>    consumed by both MSD passes before eviction. The tile must satisfy m_t k_t b_w ≤ C_tile, where C_tile is
>    the available on-chip capacity. For attention decode with B_c = 64, d = 128: the KV tile is only 8KB—easily
>    resident.
> 2. **Online activation decomposition.** The activation components x⁽¹⁾, x⁽²⁾ (or Q⁽¹⁾, Q⁽²⁾, P⁽¹⁾, P⁽²⁾ in
>    attention) are generated on-the-fly per tile, not materialized to HBM.
> 3. **In-register/streaming partial results.** The partial outputs y⁽¹⁾, y⁽²⁾ from the two GEMM passes are
>    scaled, summed, and accumulated into the final output buffer via FixPipe (on Ascend) or register-level
>    operations—without intermediate HBM writes.
> 4. **Single final writeback.** Only the reconstructed output y = s_W ⊙ (α y⁽¹⁾ + β y⁽²⁾) is written to HBM.

and the opening warning of that section, verbatim:

> The performance claims in this paper depend on implementing MSD as a *fused tiled kernel*, not as two
> standalone GEMM invocations. If the two MSD GEMM passes were executed as independent kernels, the weight/KV
> data would be read twice from HBM, partial outputs would be materialized, and kernel launch overhead would
> erode the benefits.

with the fallback, verbatim:

> When the resident-tile condition cannot be met (e.g., very large weight matrices without sufficient on-chip
> capacity), the implementation falls back to the conservative two-read model, reducing the traffic benefit from
> ∼3× to ∼1.5× in the dominant term while still avoiding the dequantization round-trip.

### B7.5 Where MSD does and does not pay, with the measured crossover

§3.8, verbatim and complete:

> MSD is designed for **Decode-heavy inference workloads** where batch sizes are small (b ≪ m, n) and latency
> per token is critical. In decode, the additional MSD GEMM is absorbed by INT8's 2× throughput, and the
> Vector-side decomposition/merging cost is O(bn + bm) ≪ O(bmn). In prefill with large batch sizes, the extra
> MSD GEMM grows linearly with b and the dequantization cost is amortized, so MSD is not recommended.

§3.4 on the INT4 trade-off, verbatim:

> **Trade-off:** Increasing K improves approximation accuracy but requires more GEMM operations. On
> accelerators with strong INT4 throughput (typically 4× BF16), this trade-off can be **better than
> break-even**—yielding actual Cube-side speedup, as we analyze below.

Table 2, verbatim:

> Table 2: Effective compute time: MXFP8 baseline vs. MSD-MXFP4 (K = 2)
>
> | Method | Raw FLOPs | Throughput | Effective Time |
> | --- | --- | --- | --- |
> | MXFP8 × MXFP4 (dequant) | 2mn (FP8) | 2× | 1.0 mn |
> | MSD-MXFP4 (2 × FP4) | 4mn (FP4) | 4× | 1.0 mn |
>
> Two FP4 GEMMs at 4× throughput yield the same effective Cube time as one FP8 GEMM at 2× throughput.

The attention crossover table (dequant/MSD speedup ratio against sequence/query count N, with N* marking the
break-even point), verbatim as extracted:

```
              d = 128 (GQA)        d = 576 (MLA)
 N      Dequant/MSD ratio     Dequant/MSD ratio
 1            20.0×                  31.0×
 …            22.0×                   3.0×
32             1.0× (N*)              1.4×
48             0.8×                   1.0× (N*)
```

and the HBM result, verbatim:

> On decoupled architectures (e.g., Ascend NPUs), the dequant approach requires the same Vector→HBM→Cube
> round-trip for K/V as for linear-layer weights (Section 2.1), resulting in 5Md bytes of HBM traffic per
> attention head. MSD avoids this round-trip: K and V remain in INT8 and are read **once** directly by Cube
> cores (2Md bytes total for K+V), a 2.5× reduction. … This efficient data movement enables MSD to achieve
> **over 80% HBM bandwidth utilization** in GQA decode scenarios on Ascend 910B, compared to 40–50% for
> dequant.

### B7.6 Deployment status and the exact kernel paths

§7 / conclusion, verbatim:

> MSD has been deployed in Huawei **CANN 8.0** and validated in production inference workloads on Ascend 910B,
> achieving significant performance improvements in decode-〔heavy scenarios〕.

> The MSD technique is used in multiple operator kernels within the CANN ops-transformer repository
> (https://gitcode.com/cann/ops-transformer). Two representative examples are: the attention kernel
> (https://gitcode.com/cann/ops-transformer/blob/master/attention/incre_flash_attention/op_kernel/arch32/incre_flash_attention_preload_dd.h)
> and the grouped matmul A16W4 kernel
> (https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul/op_kernel/a16w4_msd).

### B7.7 The dequantization bottleneck as the paper frames it, and the Ascend numbers

§1.1, verbatim:

> A concrete illustration comes from DeepSeek's FlashMLA FP8 kernel on NVIDIA Hopper GPUs [2]. Profiling
> reveals that the dequantization path—converting `float8_e4m3` → `half` → `float32` → `bfloat16` followed by
> scale multiplication—consumes approximately **50 clock cycles per KV token**, while the matrix
> multiply-accumulate (MMA) operations for 64 query heads require only **34 cycles**. The kernel is therefore
> *dequantization-bound*: Tensor Cores sit idle while CUDA Cores struggle to feed them with dequantized data.
>
> This phenomenon is even more pronounced on architectures with decoupled compute units, such as Huawei Ascend
> NPUs. The Ascend 910B architecture features a heterogeneous compute structure comprising Vector cores (for
> scalar/vector operations) and Cube cores (for high-throughput matrix operations). Dequantization—which
> involves element-wise type conversion and scaling—must execute on Vector cores, while the subsequent GEMM
> utilizes Cube cores. The disparity in throughput between these units creates a severe pipeline stall: Cube
> cores wait for Vector cores to complete dequantization, resulting in significant underutilization of the
> accelerator's peak compute capacity.

and the dequant data path it enumerates, verbatim:

> 3. Converted weights are written back to HBM (or UB)
> 4. Cube cores read from HBM/UB and perform GEMM

Figure 1's traffic figures, verbatim: dequant path 「dominant ∼3mn」 bytes, MSD fused path 「dominant ∼mn」.

### B7.8 Related work the paper positions MSD against, with its own one-line verdicts

Verbatim, §2.3:

> On Ascend NPUs, He et al. [20] present the first practical W4A16 kernel using Vector cores for on-the-fly
> dequantization and Split-K parallelization, yet report that the redundant HBM transfer remains the dominant
> cost.

> **QServe** [21] introduces W4A8KV4 quantization with compute-aware weight reordering and register-level
> parallelism to reduce dequantization latency on GPUs. **LiquidGEMM** [22] redesigns the W4A8 GEMM pipeline to
> defer dequantization to the epilogue phase, achieving up to 2.9× speedup. **TurboMind** [23] provides a
> comprehensive mixed-precision inference framework with offline weight packing and fused dequantization.
> **MixPE** [25] proposes performing dequantization *after* per-group integer GEMM, reducing the overhead
> through shift-and-add operations rather than multipliers. These approaches optimize the dequantization path
> but do not eliminate it.

> **ABQ-LLM** [24] decomposes quantized weights into binary components and reconstructs arbitrary-precision
> GEMM via Binary TensorCore (BTC) equivalents, achieving acceleration for non-standard bit-widths such as
> W6A6 and W2A8. **LUT-GEMM** [16] and **LUT Tensor Core** [26] replace dequantization with lookup-table-based
> computation, precomputing partial dot products to avoid explicit type conversion. **T-MAN** [27] extends the
> LUT approach to NPUs with a unified table layout for both prefill and decoding. **FIGNA** [28] takes a
> hardware design approach, proposing dedicated FP-INT multiply-accumulate units that natively support
> mixed-precision operations without dequantization.

and its stated novelty claim, verbatim:

> Among existing methods, ABQ-LLM is the closest in 〔spirit〕 … ABQ-LLM decomposes on the **weight** side
> (bit-level), whereas MSD decomposes on the **activation** 〔side〕.

> 〔Contribution:〕 an activation-side multi-scale decomposition framework targeting dequantization bottlenecks
> on decoupled low-precision inference architectures. By decomposing activations rather 〔than weights〕 …
> no binary decomposition of weights.

## Relevance

B7.1 is the result we wanted: concatenating the two components along the row dimension into one `2b×n` GEMM is
published as equation (16), it is recommended specifically for `b ≪ m, n`, and the stated reasons are exactly
ours — one launch instead of two, no inter-kernel sync, better Cube utilization. Our M-stacking is the vendor's
own prescription for decode, not an improvisation, and the paper names the condition under which it stops being
right (large b, on-chip capacity exceeded). B7.0 is the correction: MSD is Multi-Scale **Dequant**, it
decomposes the activation and never the weight, its reference grouped-matmul kernel is `a16w4_msd` with a BF16
activation, and its own INT4 row prescribes **K = 3** passes — so the `a8w4_msd` family is an instantiation this
paper does not cover and its error bounds do not transfer. That cuts both ways for us: it means no published
bound applies to our split, and it means our split is the easy case, because `16*high + low` over INT4 halves of
an INT8 activation is an exact identity while every decomposition in the paper is approximate.

B7.4 is the one piece of free performance here. FixPipe can cast the INT32 accumulator, multiply by a scalar
coefficient, and atomically accumulate into GM in a single pass with no Vector involvement, which is exactly our
`16 * high + low` merge — so the recombination of an M-stacked W4A8 GEMM should cost zero Vector time, not a
`Mul` + `Add` pass. The paper claims this plus FP16/INT16 decomposition arithmetic cuts MSD's Vector overhead by
30–50%. B7.2's equation (6) is the structural licence for folding a per-output-channel correction after the
GEMM — the same argument that makes our `8*sum(w)` per-column fold legitimate as a post-GEMM row-scale rather
than a pre-GEMM weight edit. B7.5 says MSD is not recommended in large-batch prefill at all, and gives a
measured MLA crossover at N* ≈ 48 against a 31× advantage at N = 1, which brackets our batch=1 decode target at
the most favourable end. B7.6 pins the deployment claim (CANN 8.0, production on 910B) and hands us the two
exact kernel paths to read. B7.7's framing — dequant must run on Vector, GEMM on Cube, and the stall is the
whole problem — is the same A3 Vector/Cube asymmetry our 13.58 us MIX floor and our `Cast` family's
2.300 ms/round against 0.076 ms of bytes are symptoms of.

## B8. ★★★ The vendor's own measured tuning case for GroupedMatmul + per-token dequant + activation — 218.1 → 128.1 us, −41.3%, with the code for each step; plus pure-Cube mode's −65.6% on Scalar and MTE2 Preload

Sources, all in the shipped devkit on this machine, read 2026-10-20:

- `docs/zh/guide/operator_practice/best_practices/grouped_matmul_tuning.md`
- `docs/zh/guide/operator_practice/best_practices/matmul_tuning_cases/matmul_pure_cube.md`
- `docs/zh/guide/operator_practice/best_practices/matmul_tuning_cases/matmul_mte2_preload.md`

This matters because the two GroupedMatmul families are the largest block in our round:
`GroupedMatmulSwigluQuantV2` 6.793 ms spent against 2.272 ms of bytes, plus `GroupedMatmul` 5.596 / 1.212 —
12.389 ms spent against 3.484 ms of bytes. The first document below is a measured tuning case for *exactly*
that operator shape: grouped matmul, int8 weights, per-token dequant, activation.

### B8.1 The case, and the operator it is

`grouped_matmul_tuning.md`, 案例介绍, verbatim — note that the Python reference is our expert chain:

> 本案例对分组Matmul即GroupedMatmul算子的per-token量化场景进行性能分析和优化，GroupedMatmul算子计算过程（通过python代码表达）为：

```python
offset = 0
for i in range(g):
    mmOut = x[offset:offset + groupList[i]] * weight[i] + bias[i]
    y[offset:offset + groupList[i]] = Gelu(mmOut * scale[i] * pertokenScale[offset:offset + groupList[i]])
    offset += groupList[i]
```

> 验证平台为Atlas A2 训练系列产品/Atlas A2 推理系列产品。

表1 算子规格, verbatim:

| input | shape | data type | format |
| --- | --- | --- | --- |
| x | (1024,1024) | int8 | ND |
| weight | (8,1024,8192) | int8 | **NZ** |
| bias | (8,8192) | int32 | ND |
| groupList | 8 | int64 | ND |
| scale | (8,8192) | float | ND |
| pertokenScale | 1024 | float | ND |
| y | (1024,8192) | float16 | ND |

> 主要介绍以下优化方法：
> - 对于Vector计算占比较高（Vector Bound）的场景，将AI Core中的AIC核和AIV核启动比例设置为1:2；
> - 优化CV并行流水，减少Cube和Vector计算间的空闲等待时间；
> - 优化Vector计算流水，提高Vector并行计算速度。
>
> 固定8核测试，即当前性能和后续优化tiling中numBlocks固定设置为8。

The diagnosis, verbatim:

> - Vector计算bound，当前为减少核启动开销设置为1:1；
> - 实际优化过程中，对上述问题进行优化、Vector计算占比下降后，Cube和Vector计算各自都有间隙，相互之间都有等待耗时；
> - Vector计算没有开启double buffer，计算和数据搬运部分没有并行。

and the measurement recipe, verbatim:

> 通过msOpProf算子调优工具获取算子性能数据：
> - 获取真实环境执行的性能数据（指令的cycle占比数据ArithmeticUtilization.csv），包含各个流水的占比情况；
> - 获取仿真性能数据（指令流水图），包含各个流水的占用区间，可观察流水间依赖情况，从而优化并行效率。

### B8.2 ★ The measured ladder

| step | total | delta |
| --- | --- | --- |
| baseline (AIC:AIV 1:1, 8 cores) | **218.1 us** | — |
| AIC:AIV ratio set to **1:2** | **154.2 us** | −29.3% |
| workspace count 2 → **4** | **131.8 us** | −14.5% |
| Vector **double buffer** | **128.1 us** | −2.8% |
| cumulative | | **−41.3%** |

Verbatim:

> - 将AI Core中的AIC和AIV启动比例设置为1:2后，执行总耗时从218.1us下降为154.2us。指令流水图显示Cube计算间等待变小。
> - …优化后，总耗时由154.2us下降为131.8us。指令流水图显示Vector、Cube计算各自间隙明显减小。
> - Vector计算开启double buffer，优化后执行总耗时从131.8us下降为128.1us。

and 总结, verbatim:

> - 在Vector计算为主要瓶颈点时，将AI Core中的AIC核和AIV核启动比例设置为1:2；
> - Cube、Vector计算时间接近，且两者都有因相互等待导致的间隙时，采用4份workspace优化；
> - 观察数据搬运是否与计算相互掩盖，多轮计算没有数据依赖，且buffer够大时，开启double buffer，增加并行效率。

### B8.3 Step 1 — AIC:AIV 1:2, with the two-AIV alternation code

> 将AI Core中的AIC核和AIV核启动比例设置为1:2。每次AIC输出的数据，由两个AIV并行计算对应的反量化和激活函数；在Vector侧代码的循环里，AIV0和AIV1交替进行计算（**前提条件，循环次数不为1**）。

```cpp
uint32_t vecCount = 0;
uint32_t taskRation = GetTaskRatio();
for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
    if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
        curVecBaseN = curCubeSingleN - offsetN;
    }
    uint32_t alignBaseN = Ceil(curVecBaseN, uint32_t(8)) * 8;
    DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);
    uint32_t curVecBaseM = vecBaseM;
    uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;
    CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
    for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
         vecCount++;
        if (vecCount % taskRation != subBlockIdx) {
            continue;  // AIV0和AIV1交替进行计算
        }
        if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) {
            curVecBaseM = curCubeSingleM - offsetM;
        }
        // 使用AscendDequant接口做perchannel反量化
        LocalTensor mmOutLocal = vecInQueue.AllocTensor();
        DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN],
                      curVecBaseM, curVecBaseN, curVecBaseN);
        vecInQueue.EnQue(mmOutLocal);
        ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
        LocalTensor yLocal = vecOutQueue.DeQue();
        DataCopyPad2D(yGm[outOffset + offsetM * tiling->n + offsetN], yLocal,
                      curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
        vecOutQueue.FreeTensor(yLocal);
    }
    ...
}
```

`GetTaskRatio()` returns the AIV-per-AIC ratio, and the whole alternation is one line:
`if (vecCount % taskRation != subBlockIdx) continue;`. **The stated precondition is that the loop count is
not 1** — at batch=1 with a single M tile the inner loop may well be 1, which disables the split.

### B8.4 ★★ Step 2 — four workspaces, and the Cube skipping its first four waits

> AIC和AIV启动比例设置为1:2后，出现Cube和Vector计算各自都有间隙、相互之间都有等待耗时的情况。分析原因是因为Vector和Cube计算存在使用一份workspace进行数据传递的场景，通过4份workspace的方案进行优化：**host按4倍baseM * baseN申请workspace，Cube侧代码在计算前可以跳过前4轮的等待。**

```cpp
if ASCEND_IS_AIC {
    if (cubeCount >= tiling->parallNum) {  // tiling->parallNum设置为4
        CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
    }
    mm.SetOrgShape(mnConfig.m, tiling->n, tiling->k);
    mm.SetSingleShape(curSingleM, curSingleN, tiling->k);
    mm.SetTensorA(xGm[xOffset]);
    auto weightSlice = weightGm[weightOffset];
    if (mnConfig.numBlocksM == 1) {
        weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    mm.SetTensorB(weightSlice);
    uint64_t workspaceOffset = mnConfig.workSpaceOffset;
    while (mm.Iterate()) {
        mm.GetTensorC(mmOutGm[workspaceOffset], 0, true);
        CrossCoreSetFlag(SYNC_AIC_TO_AIV);
        workspaceOffset += (mnConfig.baseM * mnConfig.baseN);
    }
}
cubeCount++;
```

and the explanation of why two was not enough, verbatim:

> Vector计算在等Cube计算输出的数据，Cube侧需要等Vector计算完释放workspace以存放下一轮的计算结果，当前为了让Cube、Vector计算流水并行，workspace使用了两份空间：… 因为Vector和Cube计算存在使用一份workspace进行数据传递的场景，存在数据依赖，所以会有等待的间隔。可以采用4份workspace进行优化

**Three things to lift from this snippet.** It is a **credit window of depth 4**, implemented as
`if (cubeCount >= parallNum) CrossCoreWaitFlag(...)` — the producer runs 4 ahead before it ever waits, which is
the same discipline as `catlass`'s `CrossCoreSetFlagWithReverse` 15-deep window but with the depth set by
workspace count rather than by the flag counter. The CV handoff is **per `Iterate()`**, not per kernel:
`while (mm.Iterate()) { mm.GetTensorC(...); CrossCoreSetFlag(SYNC_AIC_TO_AIV); }`. And —

> `if (mnConfig.numBlocksM == 1) { weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE); }`

— **when there is only one M block, the vendor hints the weight OUT of L2.** At one M block each weight tile is
read exactly once, so caching it only evicts something reused. `numBlocksM == 1` is the decode condition.

### B8.5 Step 3 — Vector double buffer

```cpp
pipe->InitBuffer(scaleInQueue, 2, tiling->mmTilingData.baseN * sizeof(DTYPE_SCALE));
pipe->InitBuffer(perTokenScaleInQueue, 2, tiling->mmTilingData.baseM * sizeof(float));
pipe->InitBuffer(vecInQueue, 2, tiling->ubCalSize * sizeof(cT::T));
pipe->InitBuffer(vecOutQueue, 2, tiling->ubCalSize * sizeof(DTYPE_Y));
```

### B8.6 ★★ Pure-Cube mode: the MIX message framework costs 15.02 of 17.85 us, and `ASCENDC_CUBE_ONLY` removes it

`matmul_pure_cube.md`, 案例介绍, verbatim:

> 本案例呈现了在矩阵乘算子场景中，开启Matmul高阶API的纯Cube模式对算子性能的提升效果。如下图所示，**Matmul API默认使用MIX模式，即用户从AIV侧发起消息，通过消息通信框架中转消息后，在AIC侧执行Matmul计算。这套消息处理机制会带来额外的Scalar性能开销。** 相较于MIX模式，纯Cube模式可以直接跳过消息通信框架，完成Matmul计算，提升算子性能。
>
> - 开启纯Cube模式的适用场景：非融合算子，只有矩阵计算的场景。即相较于MIX模式（包含矩阵计算和矢量计算），没有矢量计算的场景。

Spec: `a` (128, 64) float16 ND, `b` (64, 30720) float16 ND, M=128 N=30720 K=64, on a 24-core part with 1 AIC
and 2 AIV per core. The tiling difference between the two modes is the thing to notice, verbatim:

> - 单核shape：
>     - MIX场景：按**48个AIV核**进行切分，singleCoreM=128，singleCoreN=640，singleCoreK=64。
>     - 纯Cube场景：按**24个AIC核**进行切分，singleCoreM=128，singleCoreN=1280，singleCoreK=64。
> - 基本块shape：baseM=128，baseN=256，baseK=64。
> - L1相关Tiling参数：stepM=1，stepN=1，stepKa=4，stepKb=4，depthA1=8，depthB1=8。

The measurement, verbatim:

> - 优化前的Profiling数据如下，从C列的aic_time数据可以看出，多个核中最大算子执行耗时为 **17.85us**。从G列的aic_scalar_time数据可以看出，Scalar平均耗时为 **15.02us**，性能瓶颈在Scalar流水。
> - 优化前的流水图如下，由于默认为MIX模式，每次Matmul计算均涉及消息通信框架对消息进行处理，Scalar流水重，性能开销较大。
> - 优化后的Profiling数据如下，从C列的aic_time数据来看，多个核中最大算子执行耗时为 **11.21us**，较优化前的17.85us有较大提升。从G列的aic_scalar_time数据来看，Scalar平均耗时从优化前的15.02us降低至 **5.17us**。
> - 优化后的流水图如下。对比优化前的流水图，…Scalar流水明显变稀疏。纯Cube模式相较于MIX模式，减少了对消息通信的处理，优化了整体Scalar性能开销。

> 总结：在只有矩阵计算，没有矢量计算的场景下，可以考虑开启纯Cube模式，优化Matmul计算中的消息通信性能开销，提升算子性能。

So: **`aic_time` 17.85 → 11.21 us (−37.2%), `aic_scalar_time` 15.02 → 5.17 us (−65.6%)**, and the whole
difference is the KFC message framework that the default MIX-mode `Matmul` high-level API uses to have the AIV
drive the AIC. The enabling switch is the `#define ASCENDC_CUBE_ONLY` before `#include "lib/matmul_intf.h"`
(§42.6, §44.2).

### B8.7 MTE2 Preload, with its four hard preconditions

`matmul_mte2_preload.md`, verbatim:

> 通过MatmulConfig中的 **doMTE2Preload** 参数开启矩阵M或N方向的预加载功能，预加载即在MTE2间隙提前加载A矩阵/B矩阵数据，开启预加载功能后，可以减少MTE2间隙，提升算子性能。
>
> - 开启MTE2 Preload的适用场景：**MTE2流水间隙较大，且M或N数值较大时。**
> - 开启MTE2 Preload的约束条件
>     - 仅在使用 **MDL模板和SpecialMDL模板** 时，MTE2 Preload有效。
>     - 开启M或N方向预加载功能时，需保证 **K方向数据全载**，且M或N方向开启DoubleBuffer。
>     - K方向数据全载的条件是 `singleK <= baseK * stepK`。
>     - M方向开启DoubleBuffer的条件是 `depthA1 = stepM * stepK * 2`。
>     - N方向开启DoubleBuffer的条件是 `depthB1 = stepN * stepK * 2`。

Spec: `a` (128, 512) float16, `b` (512, 24576) float16, M=128 N=24576 K=512, 24 cores, pure-Cube mode, MDL
template, singleCoreM=128 singleCoreN=1024 singleCoreK=512, baseM=128 baseN=128 baseK=64, stepM=1 stepN=1
stepKa=8 stepKb=8 depthA1=8 **depthB1=16**.

> - 优化前的流水图如下，M和K方向全载，因此A矩阵只搬运一次。由于N较大，B矩阵会搬运多次，可以看到单次MTE2间存在间隙。
> - 优化前的Profiling数据如下，aic_time平均耗时 **30.88us**。
> - 优化后的流水图如下，Tiling参数不变，可以看到，下一次计算使用的B矩阵数据提前加载，MTE2间的间隙缩短。
> - 优化后的Profiling数据如下，aic_time平均耗时 **28.50us**，较优化前的30.88us有所下降。

**30.88 → 28.50 us, −7.7%.** Note `depthB1 = stepN * stepK * 2 = 1 * 8 * 2 = 16`, which is the N-direction
double-buffer precondition satisfied exactly.

## Relevance

B8.2 is a measured −41.3% on the operator our profile calls `GroupedMatmulSwigluQuantV2`, published by the
vendor with the code for every step, and none of the three steps is a tiling retune — they are a launch-ratio
change, a workspace-count change and a `InitBuffer` depth change. Against 12.389 ms in the two GroupedMatmul
families, that ladder is the largest documented win available to us that does not require writing a new kernel.

Two of the three steps carry a precondition that bites at batch=1 and must be checked before counting on them.
The 1:2 ratio works by alternating two AIVs across the inner loop — `if (vecCount % taskRation != subBlockIdx)
continue;` — and the document states 「前提条件，循环次数不为1」. With one M tile the inner loop may be 1, in
which case one AIV does everything and the ratio buys nothing; the fix would be to split along N instead, which
is what `offsetN` already loops over. The four-workspace step is a **credit window of depth 4** on the CV
handoff (`if (cubeCount >= parallNum) CrossCoreWaitFlag(...)`, producer runs four ahead before it ever waits),
which costs 4 × baseM × baseN of workspace and is the same discipline as `catlass`'s 15-deep window — and it is
worth noting that the handoff is per `Iterate()`, not per kernel, so a per-layer megakernel inherits this
pattern rather than replacing it.

The single line most directly applicable to us is in B8.4:
`if (mnConfig.numBlocksM == 1) { weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE); }`. **At one M
block the vendor hints the weight out of L2**, because a tile read exactly once can only evict something that
is reused. `numBlocksM == 1` is the decode condition, and this is one call on a `GlobalTensor`.

B8.6 quantifies what `ASCENDC_CUBE_ONLY` is actually buying and it is larger than expected: the default MIX
`Matmul` has the AIV send a message through the KFC framework to the AIC, and that framework cost
**15.02 us of 17.85 us total `aic_time` in Scalar**, falling to 5.17 us in pure-Cube mode — `aic_time`
17.85 → 11.21 us, −37.2%. Two consequences. Any matmul of ours not marked `ASCENDC_CUBE_ONLY` is paying that,
which is a candidate explanation for part of the measured 13.58 us MIX_AIC floor; and the mode changes the core
split, 48 AIV-based blocks becoming 24 AIC-based blocks with `singleCoreN` doubling, so the tiling must be
recomputed rather than carried over. B8.7's `doMTE2Preload` is a smaller, conditional −7.7%, gated on the MDL
or SpecialMDL template, full K residency (`singleK <= baseK * stepK`) and a double-buffered M or N
(`depthA1 = stepM * stepK * 2`, `depthB1 = stepN * stepK * 2`) — all four are checkable against our tiling
without a run.
