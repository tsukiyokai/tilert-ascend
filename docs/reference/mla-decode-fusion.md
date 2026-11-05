# MLA Decode Fusion on Ascend — Source Material

Verbatim research deposit for the GLM-5.2-W4A8C8 / Ascend A3 megakernel effort.
All quoted blocks are reproduced as published. Chinese sources are left in Chinese.
Collected 2026-10-20.

---

## 1. MlaProlog — the one fused op that already works

### `MlaProlog算子设计介绍` — why the op exists, and the CV pipeline pseudo-code

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog/docs/MlaProlog%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md
Huawei / CANN `ops-transformer`, master branch.

Stated motivation — op-count and launch-overhead reduction is the explicit design goal:

> 在此背景下，DeepSeek团队提出Mla算子需求，需要对算法和硬件进行协同优化，通过算子融合，减少算子调用和头开销时间，在保证模型精度的前提下，降低推理延迟、并减少硬件资源消耗，MlaProlog算子正是Mla算子前处理融合部分。

What it fuses:

> 按照Multi-Head Latent Attention定义的计算流程实现，整体计算流程如下：
>
> 1. 输入序列x经过下采样$W^{DQ}$矩阵进行降秩变换，并进行归一化处理得到$c^Q$
>
> 2. 将$W^{UQ}$和$W^{QR}$矩阵进行拼接，实现对$c^Q$的升秩和映射计算，对运算结果再进行split拆分，分别做Qn的归一化处理和Rope位置编码，最终得到Query和Query Rope
>
> 3. 将$W^{DKV}$和$W^{KR}$矩阵进行拼接，对输入序列x实现降秩和映射计算，对运算结果再进行split拆分，分别做Rope位置编码和$C^{KV}$的归一化处理，最终得到KR Cache和KV Cache

The four matmuls and how each is split across cores:

> MlaProlog算子有多个Matmul运算：
>
> 1、MatmulCq：将BS合轴，作为M轴，HeadSizeCq为N轴，HeadSizeX为K轴，核间对HeadSizeCq按照cube核数进行切分，每个核分配一定的子块
>
> 2、MatmulCkvKr：将BS合轴，作为M轴，HeadSizeCkvKr为N轴，HeadSizeX为K轴，核间对HeadSizeCkvKr进行切分，分为64块
>
> 3、MatmulQcQr：将BS合轴，作为M轴，HeadSizeQcQr为N轴，HeadSizeCq为K轴，核间对HeadSizeQcQr按照cube核数进行切分，每个核分配一定的子块
>
> 4、MatmulQn：将BS合轴，作为M轴，HeadSizeCkv为N轴，dimHeadSizeQc为K轴

The CV (cube/vector) pipeline design and the six sync points:

> ## 流水设计
>
> MlaProlog融合算子包含了Vector计算和Cube计算，Vector侧和Cube侧的计算存在依赖关系，因此需要进行CV流水设计，
> 否则C侧和V侧很有可能是串行流水的效果，不能达到并行计算的目的，无法使得融合算子性能达到最优：
>
> 1、MatmulCq计算的时候，对Hcq进行了分核，单核没有计算一个完整的token，所以在RmsNormCq计算前，
> 需要做AIC与AIV之间的同步控制（SYNC_MMCQ_NOMRCQ）
>
> 2、MatmulCkvKr计算的时候，也是对Hcq进行了分核，单核没有计算一个完整的token，所以在RmsNormCkv计算前，也
> 需要做AIC与AIV之间的同步控制（SYNC_MMCKVKR_NOMRCKV）
>
> 3、由于RmsNormCq按照行切分到不同核上，因此在做MatmulQcQr计算的时候，也需要AIV与AIC的同步控制（SYNC_NOMRCQ_MMQCQR）
>
> 4、由于和RopeQr的分核策略不同，因此RopeQr计算前，也需要AIC和AIV之间的同步控制（SYNC_MMQCQR_ROPEQR）
>
> 5、由于MatmulQcQr和MatmulQn的分核策略不同，MatmulQn依赖于MatmulQcQr的输出，因此需要做cube的全核同步（SYNC_ALL_CUBE）
>
> 6、Vector核运算前，需要做vector的全核同步（SYNC_ALL_VECTOR），确保数据流水搬运

The single-core main loop, verbatim — this is the canonical Ascend intra-kernel
cross-unit dependency mechanism:

```c
// 单核计算伪码
void Process() {
    loops = compute_step_this_core(); //计算当前core的循环次数
    // 
    for (i = 0; i < loops; i++) {
        updateCurrentStepSize(i, stepSize, allTokenSize);  //刷新当前轮的计算的M轴大小
  
        if ASCEND_IS_AIC {
      
            MatmulCq(tokenXOffset, weightDqOffset, cqResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMCQ_NORMSCQ_FLG);      //cube与vector同步

            MatmulCkvKr(tokenXOffset, weightDkvKrOffset, ckvKrResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMCKVKR_NORMROPE_FLG);   //cube与vector同步

            CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ_FLG);                    // MatmulQcQr依赖RmsNormCq的输出，需要插入CV核间同步
      
            MatmulQcQr(weightUqQrOffset, qcQrResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMQCQR_ROPEQR_FLG);  //cube与vector同步


            // 由于MatmulQn和MatmulQcQr的分核策略不一样，MatmulQn又依赖MatmulQcQr的输出
            // 需要等所有cube核上的MatmulQcQr执行完后才能启动MatmulQn
            CrossCoreSetFlag<0x0, PIPE_FIX>(SYNC_ALL_CUBE_FLG);
            CrossCoreWaitFlag(SYNC_ALL_CUBE_FLG);
      
            MatmulQn(qcOffset, weightUkOffset, qnResOffset, mmQnLoopTime);  // MatmulQn的结果直接输出到queryOut, qnOffset需要按Batch轴偏移

      }
    
       if ASCEND_IS_AIV {
      
            GetSinCos(tokenIndex);

      CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ_FLG);        // wait MatmulCq
      
      CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);    
      CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);
      
            RmsNormCq(rmsNormCqOffset);
      
      // 由于RmsNormCq和MatmulQcQr的分核策略不一样，需要等所有vector上的RmsNormCq执行完成后才能启动MatmulQcQr
      // 需要所有vector核上的RmsNormCq执行完成后，才发起MatmulQcQr的执行
      CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);
      CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);

      CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_MMCQ_NORMSCQ_FLG);    // 保障MatmulQcQr等RmsNormCq


      CrossCoreWaitFlag(SYNC_MMCKVKR_NORMROPE_FLG);           // wait MatmulCkvKr

      CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);    
      CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);
      RmsNormRopeScatterCkvKr(tokenIndex, rmsNormCkvOffset, ropeKrOffset);

      CrossCoreWaitFlag(SYNC_MMQCQR_ROPEQR_FLG);        // wait MatmulQcQr

      CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);    
      CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);
      RopeQr(ropeQrOffset, ropeQrResOffset);     
        }
    }
}
```

The complete formula set:

> $$
> q^N = RmsNorm(x \cdot W^{DQ}) \cdot W^{UQ} \cdot W^{UK} \tag{a}
> $$
>
> $$
> q^R = ROPE(RmsNorm(x \cdot W^{DQ}) \cdot W^{QR}) \tag{b}
> $$
>
> $$
> k^R = ROPE(x \cdot W^{KR}) \tag{c}
> $$
>
> $$
> c^{KV} = RmsNorm(x \cdot W^{DKV}) \tag{d}
> $$

The weight-concatenation trick (two GEMMs folded into one by horizontal concat):

> 利用矩阵乘法的性质，$W^{DKV}$和$W^{KR}$矩阵可以横向拼接成一个矩阵$[W^{DKV}|W^{KR}]$来计算，该拼接矩阵对应接口文档的$weightDkvKr$参数。
> $$
> c^{KV}k^R = x \cdot [W^{DKV}|W^{KR}] = [x \cdot W^{DKV}|x \cdot W^{KR}] \tag{7}
> $$

> 同样利用矩阵乘法的性质，$W^{UQ}$和$W^{QR}$矩阵可以横向拼接成一个矩阵$[W^{UQ}|W^{QR}]$来计算，该拼接矩阵对应接口文档的$weightDkvKr$参数。
> $$
> q^Cq^R = c_{norm}^Q \cdot [W^{UQ}|W^{QR}] = [c_{norm}^Q \cdot W^{UQ}|c_{norm}^Q \cdot W^{QR}] \tag{11}
> $$

`MatmulQn` is the absorb step — `q^C · W^UK` — done inside the prolog:

> ### MatmulQn
>
> 对$q^C$矩阵乘上Key的上采样矩阵$W^{UK}$得到最终的Query矩阵$q^N$。
> $$
> q^N = q^C \cdot W^{UK} \tag{16}
> $$

RoPE is implemented in the even/odd-split form so cos/sin are contiguous:

> 为了节省$cos$/$sin$的存储空间并简化内部的实现逻辑，考虑$q^R*k^R$矩阵乘最终要进行Reduce操作：同时调整行和列中元素的位置，不影响最后的累加结果。最终内部$ROPE$实现的计算调整成公式（15）的形式，对输入$x$按奇偶位置拆分成两部分，对应的$cos$和$sin$部分的输入是连续的。

KV cache NZ stride rule:

> - 当前按NZ格式存储KVCache时，是将整个Block存储成NZ格式；小块内是以行为主(Row Major)的排布，形状如Z字型；块与块之间是以列为主的排布，形状如N字形。将完整的H(Hidden States)按32B长度的小块进行分块后，在ND格式下连续存储的小块，在NZ格式下需要跳$32B * BlockSize$存储，即需要设置stride为$32B * BlockSize$。

## Relevance

This is our 2.950 ms vs 2.745 ms op — the only one at the bandwidth bound. The pseudo-code
explains why it wins: four GEMMs, two RmsNorms, two RoPEs and the cache scatter live in a
single kernel launch, coupled only by `CrossCoreSetFlag`/`CrossCoreWaitFlag` and two grid-wide
barriers. Those two primitives plus `ASCEND_IS_AIC`/`ASCEND_IS_AIV` are the complete mechanism
set we need to replicate for the rest of the layer. Note also that `MatmulQn` already performs
`q^C · W^UK` in-kernel, so the absorb multiply is already inside the prolog's launch, not a
separate op.

---

### MlaPrologV3 — TilingKey bit layout, including `SPLIT_M_MODE` and `CV_MODE`

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog_v3/docs/MlaPrologV3%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md

> ## TilingKey划分
>
> TilingKey为uint64类型，每个模板参数对应TilingKey中的一到数个二进制位，具体实现如下：
>
> | 二进制位 | 变量名 | 说明 | 参数列表 |
> | ------- | ------ | ---- | ------- |
> | 0-3 | CACHE_MODE | KVCache的存储格式 | 0-ND（BSND/TND），1-PA_BSND，2-PA_NZ，3-PA_BLK_BSND，4-PA_BLK_NZ |
> | 4-5 | SCENARIO | 输入场景 | 0-FP16(预留)，1-BF16，2-量化场景 |
> | 6-11 | QUANT_MODE | 量化场景 | 0-非量化，1-MMQcQr量化，2-MMQcQr量化+KVcache量化，3-MMcqCkvKr量化+MMQcQr量化，4-MMCqCkvkr量化+MMQcQr量化+KVcache量化，5-MMQcQr量化+KVcache pertoken-pergroup量化，6-MMCqCkvkr量化+MMQcQr量化+KVcache pertoken-pergroup量化，7-Mxfp8量化+MMCqCkvkr量化+MMQcQr量化，8-Mxfp8量化+MMCqCkvkr量化+MMQcQr量化+KVcache量化，9-Mxfp8量化+MMCqCkvkr量化+MMQcQr量化+KVcache pertoken-pergroup量化，10-fp8量化+MMcqCkvKr量化+MMQcQr量化，11-fp8量化+MMCqCkvkr量化+MMQcQr量化+KVcache量化，12-hif8量化+MMcqCkvKr量化+MMQcQr量化，13-hif8量化+MMCqCkvkr量化+MMQcQr量化+KVcache量化，14-fp8量化+MMcqCkvKr量化+MMQcQr量化+KVcache pertoken-pergroup量化，15-hif8量化+MMCqCkvkr量化+MMQcQr量化+KVcache pertoken-pergroup量化 |
> | 12 | ENABLE_DEQUANT_OPTIONAL | 反量化开启，不能与ENABLE_GROUP_COMPUTE_OPTIONAL一同使用 | 0-关闭，1-开启 |
> | 13 | ENABLE_GROUP_COMPUTE_OPTIONAL | 量化的算力分组优化，不能与ENABLE_DEQUANT_OPTIONAL一同使用 | 0-关闭，1-开启 |
> | 14-15 | EMPTY_TENSOR_MODE | 空tensor场景，用于输入tensor维度为0的情况 | 0-无空tensor，1-kv_cache/kr_cache为空，2-query为空且不更新cache |
> | 16-17 | ACTUAL_SEQ_LEN_MODE | actualSeqLen开启场景 | 0-关闭，1-开启actualSeqLen |
> | 18-19 | SPLIT_M_MODE | 切M模式 | 0-关闭(切N)，1-开启(切M) |
> | 20 | ENABLE_ROPE | RoPE开关，仅V4可选关闭（doRope=false时直通写出），V1/V2/V3仅编译开启 | 0-关闭(passthrough)，1-开启 |
> | 21-29 | CV_MODE | CV分核模式 | ASCENDC_TPL_MIX_AIC_1_1(6)：1:1模式，ASCENDC_TPL_MIX_AIC_1_2(7)：1:2模式 |

V3's additional outputs beyond V1:

> 4. 在输出Query量化的情况下，会对Query做Rowmax动态量化，最终得到量化后的Query和对应量化参数DequantScaleQNope
>
> 5. 在queryNormFlag=true的情况下，输入$x$乘以$W^{DQ}$进行下采样和RmsNorm后得到压缩结果$c^Q$，作为queryNorm输出
>
> 6. 在queryNormFlag=true且量化场景（weightQuantMode≠0）的情况下，会对$c^Q$做动态量化，最终得到量化后的queryNorm和对应量化参数dequantScaleQNorm

Tiling criteria the op itself states:

> a. 将核心的数量用满，防止部分核闲置。
>
> b. 每一个核心被分配的计算量相对均匀，避免出现某些核计算的数据量过大，其余核空闲的情况。
>
> c. AIC和AIV之间处理的数据量要符合其对应的算力，避免AIC或AIV出现长时间的空闲。

## Relevance

`SPLIT_M_MODE` (bit 18-19) is the directly actionable knob at batch=1: with BS=1 there is no M
to split, so mode 0 (切N) is the only sane setting and any tiling that assumes 切M wastes cores.
`CV_MODE` exposes the AIC:AIV ratio as a compile-time template choice (1:1 vs 1:2) — relevant
because our per-call floors differ by unit (AI_VECTOR_CORE 0.70us vs AI_CORE 2.30us, MIX_AIV
12.04us vs MIX_AIC 13.58us), so which unit owns which stage changes the fixed cost.
`ENABLE_ROPE` passthrough (V4 only) and `EMPTY_TENSOR_MODE` are the hooks for stripping work we
do not need.

---

### aclnnMlaProlog — original signature, and its scheduled deprecation

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog/op_api/aclnn_mla_prolog.h

```cpp
#ifndef ACLNN_MLA_PROLOG
#define ACLNN_MLA_PROLOG
#warning "aclnn_mla_prolog.h is scheduled to be deprecated in December 2026, " \
    "and will be replaced by the aclnn_mla_prolog_v3_weight_nz.h. " \
    "We apologize for any inconvenience caused " \
    "and appreciate your timely migration to the new interface. "
```

```cpp
__attribute__((visibility("default"))) aclnnStatus
aclnnMlaPrologGetWorkspaceSize(const aclTensor* tokenX, const aclTensor* weightDq, const aclTensor* weightUqQr,
                               const aclTensor* weightUk, const aclTensor* weightDkvKr, const aclTensor* rmsnormGammaCq,
                               const aclTensor* rmsnormGammaCkv, const aclTensor* ropeSin, const aclTensor* ropeCos,
                               const aclTensor* cacheIndex, aclTensor* kvCacheRef, aclTensor* krCacheRef,
                               const aclTensor* dequantScaleXOptional, const aclTensor* dequantScaleWDqOptional,
                               const aclTensor* dequantScaleWUqQrOptional, const aclTensor* dequantScaleWDkvKrOptional,
                               const aclTensor* quantScaleCkvOptional, const aclTensor* quantScaleCkrOptional,
                               const aclTensor* smoothScalesCqOptional, double rmsnormEpsilonCq,
                               double rmsnormEpsilonCkv, char* cacheModeOptional, const aclTensor* queryOut,
                               const aclTensor* queryRopeOut, uint64_t* workspaceSize, aclOpExecutor** executor);
```

## Relevance

Confirms the exact tensor set the fused prolog consumes: the four weight matrices
(`weightDq`, `weightUqQr`, `weightUk`, `weightDkvKr`), both RmsNorm gammas, sin/cos,
`cacheIndex`, and in-place `kvCacheRef`/`krCacheRef`. Any megakernel that subsumes the prolog
must carry exactly this set, and the `weight_nz` successor says the weights want FRACTAL_NZ.

---

### aclnnMlaPrologV3WeightNz — the current (non-deprecated) signature

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.h

```cpp
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    const aclTensor* tokenX, const aclTensor* weightDq, const aclTensor* weightUqQr, const aclTensor* weightUk,
    const aclTensor* weightDkvKr, const aclTensor* rmsnormGammaCq, const aclTensor* rmsnormGammaCkv,
    const aclTensor* ropeSin, const aclTensor* ropeCos, aclTensor* kvCacheRef, aclTensor* krCacheRef,
    const aclTensor* cacheIndexOptional, const aclTensor* dequantScaleXOptional,
    const aclTensor* dequantScaleWDqOptional, const aclTensor* dequantScaleWUqQrOptional,
    const aclTensor* dequantScaleWDkvKrOptional, const aclTensor* quantScaleCkvOptional,
    const aclTensor* quantScaleCkrOptional, const aclTensor* smoothScalesCqOptional,
    const aclTensor* actualSeqLenOptional, const aclTensor* kNopeClipAlphaOptional, double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv, char* cacheModeOptional, int64_t weightQuantMode, int64_t kvCacheQuantMode,
    int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize, double qcQrScale,
    double kcScale, const aclTensor* queryOut, const aclTensor* queryRopeOut,
    const aclTensor* dequantScaleQNopeOutOptional, const aclTensor* queryNormOutOptional,
    const aclTensor* dequantScaleQNormOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);
```

---

### aclnnMlaPrologV4WeightNz — full parameter documentation (quant modes, cache modes, `doRope`)

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v4_weight_nz.h

```cpp
/**
 * @brief 计算MlaPrologV4WeightNz所需的workspace大小并创建算子执行器。
 *
 * @param tokenX [IN] 公式中用于计算Query和Key的输入tensor。
 * @param weightDq [IN] 公式中用于计算Query的下采样权重矩阵。
 * @param weightUqQr [IN] 公式中用于计算Query的上采样权重矩阵和位置编码权重矩阵。
 * @param weightUk [IN] 公式中用于计算Key的上采样权重。
 * @param weightDkvKr [IN] 公式中用于计算Key的下采样权重矩阵和位置编码权重矩阵。
 * @param rmsnormGammaCq [IN] 计算c^Q的RmsNorm公式中的γ参数。
 * @param rmsnormGammaCkv [IN] 计算c^KV的RmsNorm公式中的γ参数。
 * @param ropeSin [IN]
 * 用于计算旋转位置编码的正弦参数矩阵。doRope=true时必选，doRope=false时必须与ropeCos同时为空（nullptr）。
 * @param ropeCos [IN]
 * 用于计算旋转位置编码的余弦参数矩阵。doRope=true时必选，doRope=false时必须与ropeSin同时为空（nullptr）。
 * @param kvCacheRef [IN/OUT] 用于cache索引的aclTensor，计算结果原地更新。
 * @param krCacheRef [IN/OUT] 用于key位置编码的cache，计算结果原地更新。
 * @param cacheIndexOptional [IN] 用于存储kvCache和krCache的索引。
 * @param dequantScaleXOptional [IN] tokenX的反量化参数。
 * @param dequantScaleWDqOptional [IN] weightDq的反量化参数。
 * @param dequantScaleWUqQrOptional [IN] 用于MatmulQcQr矩阵乘后反量化操作的perchannel参数。
 * @param dequantScaleWDkvKrOptional [IN] weightDkvKr的反量化参数。
 * @param quantScaleCkvOptional [IN] 用于对kvCache输出数据做量化操作的参数。
 * @param quantScaleCkrOptional [IN] 用于对krCache输出数据做量化操作的参数。
 * @param smoothScalesCqOptional [IN] 用于对RmsNormCq输出做动态量化操作的参数。
 * @param actualSeqLenOptional [IN] 表示每个batch中的序列长度，以前缀和的形式储存。
 * @param kNopeClipAlphaOptional [IN] 表示对kvCache做clip操作时的缩放因子。
 * @param rmsnormEpsilonCq [IN] 计算c^Q的RmsNorm公式中的ε参数。
 * @param rmsnormEpsilonCkv [IN] 计算c^KV的RmsNorm公式中的ε参数。
 * @param cacheModeOptional [IN]
 * 表示kvCache的模式，可选值为"PA_BSND"、"PA_NZ"、"PA_BLK_BSND"、"PA_BLK_NZ"、"BSND"、"TND"。
 * @param weightQuantMode [IN] 表示weightDq、weightUqQr、weightUk、weightDkvKr的量化模式，0表示非量化。
 * @param kvCacheQuantMode [IN]
 * 表示kvCache的量化模式，0表示非量化，1表示pertensor量化，2表示perchannel量化，3表示pertoken-pergroup量化。
 * @param queryQuantMode [IN] 表示query的量化模式，0表示非量化，1表示per-token-head量化。
 * @param ckvkrRepoMode [IN] 表示kvCache和krCache的存储模式，0表示分别存储，1表示合并存储。
 * @param quantScaleRepoMode [IN]
 * 表示量化scale的存储模式，0表示量化scale和数据分别存储，1表示合并存储作为kvCacheRef输出。
 * @param tileSize [IN] 表示pertoken-pergroup量化时每个tile的大小，仅在kvCacheQuantMode为3时有效，默认值为128。
 * @param qcQrScale [IN] 表示Query的尺度矫正系数，用户不特意指定时需要传入1.0。
 * @param kcScale [IN] 表示Key的尺度矫正系数，用户不特意指定时需要传入1.0。
 * @param doRope [IN]
 * 表示是否对queryRopeOut与krCache执行旋转位置编码（RoPE）。默认true：执行RoPE，此时ropeSin/ropeCos必须为非空有效Tensor；false：跳过RoPE并写直通结果，此时ropeSin/ropeCos必须同时为空（nullptr）。
 * @param queryOut [OUT] 公式中Query的输出tensor。
 * @param queryRopeOut [OUT] 公式中Query位置编码的输出tensor。doRope=false时输出直通结果，shape不受影响。
 * @param dequantScaleQNopeOutOptional [OUT] 公式中Query输出的反量化参数。
 * @param queryNormOutOptional [OUT] 公式中tokenX做rmsNorm后的输出tensor。
 * @param dequantScaleQNormOutOptional [OUT] queryNormOutOptional的反量化参数。
 * @param workspaceSize [OUT] 返回需在Device侧申请的workspace大小，单位为字节。
 * @param executor [OUT] 返回包含算子执行流程的执行器。
 * @return aclnnStatus 成功时返回ACLNN_SUCCESS，否则返回对应错误码。
 * @domain aclnn_ops_infer
 */
```

## Relevance

`ckvkrRepoMode=1` (merged kv/kr storage) and `quantScaleRepoMode=1` (scale merged into
`kvCacheRef`) collapse what would otherwise be separate cache tensors and separate scale
writes — fewer discrete writes per decode step, which is the right direction given that our C8
KV cache path is latency- not bandwidth-limited. `queryQuantMode=1` (per-token-head) produces
a quantized query plus `dequantScaleQNopeOut` directly, so the prolog can hand an already
quantized Q to the attention stage without an intervening Cast (we currently spend 2.300 ms on
Cast against 0.076 ms of bytes).

---

## 2. SparseFlashAttention and the selection stage

### aclnnSparseFlashAttention — signature, `attentionMode=2` is MLA-absorb, and the official statement of its weakness

https://gitcode.com/cann/ops-transformer/blob/master/attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md

> ## 功能说明
>
> - 接口功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过"只计算关键部分"大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。
>
> - 计算公式：
>
> $$
> \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
> $$
>
> 其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。

```cpp
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *sparseIndices,
    const aclTensor     *blockTableOptional,
    const aclTensor     *actualSeqLengthsQueryOptional,
    const aclTensor     *actualSeqLengthsKvOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    double              scaleValue,
    int64_t             sparseBlockSizeOptional,
    char                *layoutQueryOptional,
    char                *layoutKvOptional,
    int64_t             sparseMode,
    int64_t             preTokens,
    int64_t             nextTokens,
    int64_t             attentionMode,
    bool                returnSoftmaxLse,
    const aclTensor     *attentionOut,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

Parameter semantics, extracted from the doc's table:

```
sparseIndices（aclTensor）  输入   离散取kvCache的索引。
      不支持空tensor。
      sparseSize为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparseSize大于0。
   INT32  ND
      layoutQueryOptional为BSND时，shape为(B, S1, N2, sparseSize)。
      layoutQueryOptional为TND时，shape为(T1, N2, sparseSize)。

sparseBlockSizeOptional（int64_t）  输入  代表sparse阶段的block大小。
      sparseBlockSizeOptional为1时，为Token-wise稀疏化场景，将每个token视为独立单元，在计算重要性分数时，评估每个查询token与每个键值token之间的独立关联程度。
      sparseBlockSizeOptional为大于1小于等于128时，为Block-wise稀疏化场景，将token序列划分为固定大小的连续块，以块为单位进行重要性评估，块内token共享相同的稀疏化决策。

sparseMode（int64_t）
      sparseMode为0时，代表全部计算。
      sparseMode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。

attentionMode（int64_t）  输入   仅支持传入2，表示MLA-absorb模式。
```

Constraints:

> ## 约束说明
>
> - 确定性计算：aclnnSparseFlashAttention默认确定性实现。
> - 该接口支持推理场景下使用。
> - N2仅支持1。
> - blockSize为一个block的token数，blockSize取值为16的倍数，且最大支持1024。
> - 参数query中的D和key、value的D值相等为512。
> - queryRopeOptional与keyRopeOptional须成对为空指针或成对传入非空tensor：均为空指针时表示不使用rope，仅对512维nope部分做注意力计算；均传入非空tensor时Dr相等为64。不支持空tensor。
> - 参数query、key、value的数据类型必须保持一致。
> - Ascend 950PR&950DT系列产品：
>   - N1支持1~128。
>   - 仅支持sparseBlockSizeOptional为1。
>   - returnSoftmaxLse为True时，支持layoutKvOptional为PA_BSND。
>   - 当layoutKvOptional为PA_BBND时，key、value和keyRopeOptional支持0轴非连续，非连续Tensor的每一维的stride均不能为0。不支持其他非连续。
> - Atlas A3系列产品、Atlas A2系列产品：
>   - N1支持1/2/4/8/16/32/64/128。
>   - sparseBlockSizeOptional支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparseBlockSizeOptional整除blockSize。
>   - key、value和keyRopeOptional不支持非连续。
>   - returnSoftmaxLse为True时，layoutKvOptional不支持PA_BSND。

A3 call-site example from the doc, with the concrete shapes and parameter values:

```cpp
    std::vector<int64_t> queryShape = {1, 2, 1, 512};
    std::vector<int64_t> keyShape = {1, 2, 1, 512};
    std::vector<int64_t> valueShape = {1, 2, 1, 512};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 2};
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 512};
    ...
    int64_t sparseBlockSize = 64;
    ...
    int64_t sparseMode = 3;
    ...
    int64_t attentionMode = 2;
```

## Relevance

Huawei's own one-line summary of this op is the explanation of our 6.973 ms against 1.382 ms of
bytes: `通过"只计算关键部分"大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加`
— sparsity cuts the FLOPs and replaces them with scattered gathers. `attentionMode` is locked
to `2` = MLA-absorb, and `D=512`, `Dr=64`, `N2=1` confirm the op is written against the
absorbed latent layout, so the absorb trick is already assumed, not an optimization left on
the table. `sparseIndices` must be pre-partitioned (`每行有效值均在前半部分`), i.e. the op will
not compact for us — the indexer's output layout is part of the contract.

---

### aclnnLightningIndexer — the selection stage, exact formula and signature

https://gitcode.com/cann/ops-transformer/blob/master/attention/lightning_indexer/docs/aclnnLightningIndexer.md

> ## 功能说明
>
> - 接口功能：`lightning_indexer`基于一系列操作得到每一个token对应的Top-$k$个位置。
>
> - 计算公式：
>
> $$
> Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
> $$
>
> 对于某个token对应的Index Query $Q_{index}\in\R^{g\times d}$，给定上下文Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中$g$为GQA对应的group size，$d$为每一个头的维度，$S_{k}$是上下文的长度。

```cpp
aclnnStatus aclnnLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    char            *layoutQueryOptional,
    char            *layoutKeyOptional,
    int64_t          sparseCount,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    bool             returnValues,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

## Relevance

This is the `index_topk 2048` stage. The formula is cheap in FLOPs — one `Q_index @ K_index^T`
of head dim `d`, a ReLU, a weighted sum over `g`, then a Top-k — but the Top-k over `S_k`
(8K context) is a reduction that does not shrink with batch=1, and its output
`sparseIndicesOut` is the input `sparseIndices` of SFA. Fusing indexer and SFA into one kernel
removes one launch and one full round-trip of the index array through HBM.

---

### The CANN `attention` operator inventory (what exists to fuse against)

https://gitcode.com/cann/ops-transformer/tree/master/attention
Directory listing, `cann/ops-transformer` master, retrieved 2026-10-20.

```
attention_update                       lightning_indexer
block_sparse_attention                 lightning_indexer_v2
bsa_select_block_mask                  lightning_indexer_v2_metadata
compressor                             mixed_quant_flash_attn
dense_lightning_indexer_softmax_lse    mixed_quant_sparse_flash_mla
flash_attention_score                  mla_preprocess
flash_attn                             mla_preprocess_v2
flash_attn_metadata                    mla_prolog
flash_mla                              mla_prolog_v2
flash_mla_metadata                     mla_prolog_v3
fused_infer_attention_score            nsa_compress
gather_pa_kv_cache                     nsa_compress_attention
incre_flash_attention                  nsa_compress_attention_infer
indexer_quant_cache                    nsa_compress_with_cache
key_pool                               nsa_selected_attention
kv_compress_epilog                     nsa_selected_attention_infer
kv_quant_sparse_flash_attention        pool_key_indexer
kv_quant_sparse_flash_attention_v2     prompt_flash_attention
quant_flash_mla                        quant_lightning_indexer
quant_sparse_flash_mla                 scatter_pa_cache
sparse_flash_attention                 scatter_pa_kv_cache
sparse_flash_mla                       scatter_pa_kv_cache_with_k_scale
sparse_flash_mla_softmax_l1_norm       swin_attention_score_quant
```

## Relevance

Two entries matter for us and were not on our radar: `kv_quant_sparse_flash_attention` /
`_v2` (a C8-KV-cache-aware SFA, i.e. the quantized-cache variant of the op we run) and
`kv_compress_epilog`. `quant_lightning_indexer` is the quantized indexer. If any of these
already fuse the dequant of the C8 cache into the attention load, they remove part of our
2.300 ms Cast.

---
### CANN `flash_mla` / `flash_mla_metadata` — a separate metadata kernel that pre-computes the AICore/AIVCore split

https://gitcode.com/cann/ops-transformer/blob/master/attention/flash_mla/docs/torchapi_flash_mla.md

> ## 功能说明
>
> - **接口功能**：
>
>   `flash_mla`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于非量化 MLA 注意力计算。输入`q`和`k_cache`已合并512维nope与64维rope，算子不执行KV压缩或RoPE变换。
>
>   `k_cache`同时提供Key和Value：Key使用全部576维，Value使用前512维，不单独传入`v`、`q_rope`或`k_rope`。
>
>   `flash_mla_metadata`是主算子的元数据生成接口，记录AICore/AIVCore任务切分结果。典型调用流程如下：
>
>   1. 准备Query、分页KV缓存、块表和序列长度。
>   2. 调用`flash_mla_metadata`生成`metadata`。
>   3. 将`metadata`和对应输入传入`flash_mla`。

```python
cann_ops_transformer.flash_mla_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=None,
    layout_q="BSND",
) -> Tensor
```

```python
cann_ops_transformer.flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
) -> (Tensor, Tensor)
```

> | num_heads_q | int | 必选 | Query头数，配套当前主算子时支持64、96 | int32 | - | - |
> | num_heads_kv | int | 必选 | KV头数，配套当前主算子时为1 | int32 | - | - |

Product support (note: A3 is **not** supported):

> - <term>Ascend 950PR&950DT系列产品</term>：支持
> - <term>Atlas A3系列产品</term>：不支持
> - <term>Atlas A2系列产品</term>：不支持

## Relevance

The `metadata` pattern is directly transplantable: a cheap precursor kernel computes the
AICore/AIVCore work split once and hands it to the heavy kernel as a tensor, so the heavy
kernel does no scheduling scalar work. That is the opposite of what
`GroupedMatmulSwigluQuantV2` does (it resolves its own group layout with scalar GM reads in
kernel). For a batch=1 megakernel the equivalent is to bake the whole decode-round schedule
into one metadata/instruction tensor built once per shape, not per step. Note `flash_mla`
itself is 950-only, so on A3 we are on the `sparse_flash_attention` path.

---

## 3. MLA absorb — the canonical statement of the math and its arithmetic-intensity trade

### vLLM `mla_attention.py` — the full absorb derivation, both paths, with dims

https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py
vLLM project, main branch (file retrieved 2026-10-20).

```python
"""# MLA Common Components

This file implements common components for MLA implementations.

First we define:

Sq      as Q sequence length
Skv     as KV sequence length

MLA has two possible ways of computing, a data-movement friendly approach and a
compute friendly approach. We generally want to use the compute friendly
approach for "prefill" (i.e. the ratio Sq / Skv is relatively large, often near
1) and the data-movement friendly approach for "decode" (i.e. the ratio
Sq / Skv is small, often near 0).

NOTE what we deem small and large is currently determined by if it is labelled
prefill or decode by the scheduler, but this is something we should probably
tune.

Main reference: DeepseekV2 paper, and FlashInfer Implementation
(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

Deepseek's MLA attention works the following way:
* Use a single latent vector to represent the per-token entry of the KV cache.
* For decode (i.e. the memory friendly approach) the attention "simulates" a
multi-head attention, while the compute is similar to multi-query attention.

Below is an example of both paths assuming batch size = 1

## More Extent Definitions:

C           Context length, `Skv - Sq`
H           hidden size
N           number of attention heads
Lq          latent dimension for Q              1536 in DSV3
Lkv         latent dimension for K/V            512 in DSV3
P           nope dimension, no rope.            128 in DSV3
R           rope dimension, goes through rope.  64 in DSV3
V           V head dim.                         128 in DSV3

## Vector/Matrix Definitions

h_t         hidden states (input to attention)  shape [Sq, H]
q_c         latent/compressed Q                 shape [Sq, Lq]
q_nope      uncompressed Q (no-rope)            shape [Sq, N, P]
q_pe        uncompressed Q (rope)               shape [Sq, N, R]
kv_c        latent/compressed KV                shape [Skv, Lkv]
k_pe        decoupled k position embeddings     shape [Skv, R]
new_kv_c    new kv_c from current iter          shape [Sq, Lkv]
new_k_pe    new k_pe from current iter          shape [Sq, R]
cache_kv_c  cached k_c from previous iters      shape [C, Lkv]
cache_k_pe  cached k_pe from previous iters     shape [C, R]
W_DQ        project h_t to q_c                  shape [H, Lq]
W_UQ        project q_c to q_nope               shape [Lq, N * P]
W_QR        project q_c to q_pe                 shape [Lq, N * R]
W_DKV       project h_t to kv_c                 shape [H, Lkv]
W_UK        project kv_c to k_nope              shape [Lkv, N, P]
W_KR        project h_t to k_pe                 shape [H, R]
W_UV        project kv_c to v                   shape [Lkv, N, V]
W_O         project v to h_t                    shape [N * V, H]


## Compute Friendly Approach (i.e. "forward_mha"):

q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

// MHA with QK headdim = P + R
//           V headdim = V
//      sdpa_o shape [Sq, N, V]
sdpa_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v
)
return sdpa_o @ W_O

NOTE: in the actual code,
    `kv_b_proj` is [W_UK; W_UV] concatenated per head
    `q_b_proj` is [W_UQ; W_QR] concatenated per head
    `out_proj` is W_O


## Data-Movement Friendly Approach (i.e. "forward_mqa"):

Runtime
q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(-1, N, P)
ql_nope  = einsum("snh,lnh->snl", q_nope, W_UK)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)

// MQA with QK headdim = Lkv + R
//           V headdim = Lkv
//      sdpa_o shape [Sq, N, Lkv]
// NOTE: this is less compute-friendly since Lkv > P
//       but is more data-movement friendly since its MQA vs MHA
sdpa_o = scaled_dot_product_attention(
    torch.cat([ql_nope, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c
)

o = einsum("snl,lnv->snv", sdpa_o.reshape(-1, N, Lkv), W_UV)
return o.view(-1, N * V) @ W_O


## Chunked Prefill

For chunked prefill we want to use the compute friendly algorithm. We are
assuming sufficiently large Sq / Skv ratio, in the future may want to switch to
the data-movement friendly approach if the chunk (i.e. `Sq`) is small.

However, the compute-friendly approach can potentially run out of memory if Skv
is large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`
```

## Relevance

The absorb is `ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)` on the input side and
`o = einsum("snl,lnv->snv", sdpa_o, W_UV)` on the output side — i.e. it does **not** reduce
work, it moves work from the Skv axis to the Sq axis. At batch=1, Sq=1, so both einsums are
rank-1 against a weight matrix: pure weight streaming, zero reuse, and therefore pure overhead
unless fused into a neighbouring kernel. The comment `less compute-friendly since Lkv > P but
more data-movement friendly since its MQA vs MHA` is the exact statement of why decode uses
absorb and why it does not raise arithmetic intensity at batch=1 — it lowers bytes read from
the cache, not FLOPs per byte of weight.

---

### vLLM DeepSeek V4.1 "mega attention" — one kernel for RoPE + sparse attention + inverse RoPE + FP8 cast, writing into the o_proj input buffer

https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py
vLLM project, main branch (file retrieved 2026-10-20).

```python
"""DeepSeek V4.1 attention on FlashMLA's mega-attention kernel (SM100).

One kernel does Q RoPE, sparse attention, the inverse RoPE of the output and
its FP8 cast, and writes straight into the buffer ``wo_a`` consumes. So the
layer declares ``accepts_unnormed_unroped_query`` -- the kernel RoPEs Q itself,
leaving the fused KV insert only the zero-pad to the kernel's head count -- and
its ``_alloc_attn_out`` / ``_o_proj`` pair speaks QuantizedActivation instead
of bf16, since the output needs no rotation or quantization here.

``wq_b`` rows and ``wo_a`` columns are permuted once at load so the surrounding
GEMMs speak the kernel's chunk-interleaved layouts directly. A step's prefill
and decode segments write disjoint token ranges of one output buffer pair, so
a single ``wo_a`` einsum covers the whole step.
"""
```

Fixed kernel conventions and the quantization group:

```python
_HEAD_DIM_V = 512
_ROPE_DIM = 64
# The kernel always emits per-32 scales, whatever num_per_channels says.
_QUANT_GROUP = 32
# Fixed kernel conventions: GPT-J (non-neox) RoPE, TMA-aligned col-major
# scales, round-to-nearest, packed ue8m0 -- i.e. what DeepGEMM's fp8_einsum
# consumes. Q norm is off: the V4.1 layer norms qr before wq_b.
_ROPE_ARGS = (False, _ROPE_DIM)
_SF_ARGS = (_QUANT_GROUP, True, True, True)
```

The measured mega-kernel vs split-KV table, and the warning about measuring in eager mode:

```python
# Decode always goes through the mega kernel: there is no batch size at which
# falling back to the split-KV decode pays, so there is no minimum-token
# threshold and no fallback path to maintain. What decides whether this layer
# is worth selecting at all is TP, not the batch size.
#
# Measured on one GB300 with benchmarks/kernels/benchmark_dsv41_mega_attn.py,
# CUDA graphs on, counting every per-step op each path runs (64-head model,
# topk_swa=128, topk_extra=512, NVFP4 compressed cache), as mega / split-KV
# microseconds:
#
#     s_q                     1      32     128     256     512
#     TP1 (64 live heads)  29/29   31/31   31/40   53/72   96/140
#     TP2 (32 live)        29/29   31/30   35/35   61/63  110/119
#     TP4 (16 live)        29/29   31/29   34/34   59/58  109/109
#     TP8 (8 live)         29/29   31/29   33/33   59/57  107/104
#
# The kernel absorbs the inverse RoPE and the FP8 cast, whose cost scales with
# *live* heads, while its own cost scales with *padded* heads -- always 64,
# since FlashMLA's FP8 decode takes only h_q in {64, 128}. So it wins at TP1
# (up to 1.45x, and TP1 is also where the fused insert skips the Q pad
# entirely) and from TP2 down the two cancel to within a few percent, with a
# fixed ~2 us penalty at small s_q where the padded-head cost dominates.
# Measure before assuming a win at high TP -- and measure under CUDA graphs:
# in eager mode launch overhead swamps both paths and overstates the fused one
# several-fold.
```

The fused op name and the support gate:

```python
def is_flashmla_mega_attn_supported() -> tuple[bool, str | None]:
    """Whether FlashMLA's fused mega-attention kernels are usable here."""
    is_available, maybe_reason = is_flashmla_sparse_supported()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(100):
        return False, "FlashMLA mega attention requires sm_10x GPUs."
    if not hasattr(torch.ops._flashmla_C, "fused_norm_rope_attn_rope_cast_decode"):
        return False, "vllm._flashmla_C was built without the mega attention ops."
    if not hasattr(torch.ops._flashmla_C, "permute_q_b_proj"):
        return False, "vllm._flashmla_C predates the mega attention weight permutes."
    return True, None
```

The output-buffer allocation, showing the scale layout contract with the consuming GEMM:

```python
def alloc_mega_attn_output(
    num_tokens: int,
    n_wv_group: int,
    device: torch.device,
) -> QuantizedActivation:
    """Allocate the output buffer pair the mega-attention kernels write into.

    One pair per forward step: the prefill and decode segments fill disjoint
    token ranges, so a single ``fp8_einsum`` over all ``num_tokens`` can
    consume the result instead of one call per segment.

    The scale buffer is MN-major -- stride 1 along tokens, head-dim stride
    ``ceil4(num_tokens)`` -- which is both what the kernel requires of a
    caller-provided buffer and what DeepGEMM expects for this N.
    """
```

## Relevance

This is the closest published analogue of what we are building, and it carries two load-bearing
lessons. First, the fusion boundary that paid for them is `norm + rope + attn + inverse-rope +
cast` ending in a buffer the next GEMM reads directly — i.e. the win came from deleting the
epilogue round-trips and the layout conversions (`permute_q_b_proj`, `wo_a` column permute done
once at load), not from merging the attention math. That maps onto our 2.300 ms Cast against
0.076 ms of bytes. Second, the measured table says at `s_q = 1` the mega kernel and the split
path are identical (29/29 us at every TP) — on that hardware batch=1 is already at a floor the
fusion does not touch. Our situation differs because our floors are per-call launch floors
(MIX_AIC 13.58 us, MIX_AIV 12.04 us) rather than kernel-internal, which is precisely what a
megakernel removes; but the warning `in eager mode launch overhead swamps both paths and
overstates the fused one several-fold` means any A/B we run must hold graph capture constant.

---

### GLM-5.x MLA dimensions as configured in vLLM's own attention benchmark

https://github.com/vllm-project/vllm/blob/main/benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml
vLLM project, main branch (file retrieved 2026-10-20).

```yaml
mode: mha_vs_mqa
model:
  name: "glm-5.x"
  num_layers: 78
  num_q_heads: 64
  num_kv_heads: 1
  head_dim: 576
  kv_lora_rank: 512
  qk_nope_head_dim: 192
  qk_rope_head_dim: 64
  v_head_dim: 256
  block_size: 128
  max_model_len: 131072
model_parameter_sweep:
  param_name: "num_q_heads"
  values: [64, 32, 16, 8]
  label_format: "{backend}_{value}h"
backends: ["FLASHMLA_SPARSE", "FLASHINFER_MLA_SPARSE"]
device: "cuda:0"
profile_memory: false
sparse_mla_masked_mha_max_seq_len: 131072
sparse_mla_mha_variants: ["masked_mha", "mqa"]
sparse_mla_topk_pattern: "random"
```

## Relevance

Independent confirmation of the MLA shape we are optimizing: 78 layers, 64 Q heads, 1 KV head,
`kv_lora_rank = 512`, `qk_rope_head_dim = 64`, `head_dim = 576` — which is exactly the
`D = 512` / `Dr = 64` / `N2 = 1` contract `aclnnSparseFlashAttention` enforces. Note
`qk_nope_head_dim = 192` and `v_head_dim = 256`, both larger than DeepSeek's 128/128, so the
unabsorbed `W_UK`/`W_UV` are correspondingly larger and the absorb is more attractive here than
in DSV3. The `num_q_heads` sweep `[64, 32, 16, 8]` is the TP sweep: at our TP16 we have 4 live
Q heads per die.

---
## 4. DeepSeek FlashMLA: the metadata pattern, the arithmetic-intensity threshold, and the fused epilogue

### FlashMLA — API, reusable scheduler metadata, reported throughput

https://github.com/deepseek-ai/FlashMLA
Raw: https://raw.githubusercontent.com/deepseek-ai/FlashMLA/main/flash_mla/flash_mla_interface.py — DeepSeek (Jiashi Li, Shengyu Liu, Yuanhang Sun).

```python
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = True,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    enable_batch_invariant: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run sparse attention decode forward.

    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim). bfloat16. `head_dim` must be 512 and
                `num_heads_q` must be 64 or 128.
        ...
        tile_scheduler_metadata: FlashMLASchedMeta, returned by get_mla_metadata. You may reuse the same
                `tile_scheduler_metadata` across different invocations, but only when the tensor shapes and the
                values of topk_length and extra_topk_length remain the same. Note that the values are NOT
                checked at runtime: reusing it with different `topk_length` / `extra_topk_length` values
                silently reuses stale split-KV scheduling metadata.
        num_splits: must be None (kept for compatibility with the old interface; the split counts are
                returned inside `tile_scheduler_metadata`).
        ...
        enable_batch_invariant: bool. If True, the split-KV decoding path is disabled so that results do
                not depend on how the batch is partitioned
    """
```

```python
def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty instance of FlashMLASchedMeta. The actual scheduling metadata will be generated during the first invocation of flash_mla_with_kvcache.
    """
```

README performance:

> It achieves up to 1460 TFlops during prefill and 950 TFlops during decoding on B200 with CUDA 13.3.

Earlier README (commit `ba89a34`):

> The dense MLA decoding kernel achieves up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in computation-bound configuration on H800 SXM5 with CUDA 12.8.

## Relevance

Scheduling metadata computed once and reused across every decoding step and every layer, with
the hazard stated plainly (`stale split-KV scheduling metadata` is silently accepted). That is
the discipline for a replayable megakernel: hoist all split/tile scheduling into a one-time
host computation, keep it in device memory, replay it. Compare with `flash_mla_metadata` in
CANN, which does the same thing for AICore/AIVCore.

### FlashMLA deep dive — the compute/memory ratio formula and the single-accumulator constraint

https://github.com/deepseek-ai/FlashMLA/blob/ba89a34/docs/20250422-new-kernel-deep-dive.md
DeepSeek, 2025-04-22.

> Assume the number of q heads is $h_q$, the number of q tokens per request is $s_q$ (should be 1 if MTP / speculative decoding is disabled), the number of kv tokens per request is $s_k\ (s_k \gg h_q s_q)$, and the head dimensions of K and V are $d_k$ and $d_v$ respectively. The number of FLOPs is roughly $2 (h_q s_q \cdot d_k \cdot s_k + h_q s_q \cdot s_k \cdot d_v) = 2 h_q s_q s_k (d_k+d_v)$, and the memory access volume (in bytes) is $\mathop{\text{sizeof}}(\text{bfloat16}) \times (h_q s_q d_k + s_k d_k + h_q s_q d_v) \approx 2s_k d_k$. Thus, the compute-memory ratio is $h_q s_q \cdot \frac{d_k+d_v}{d_k} \approx 2 h_q s_q$.
>
> An NVIDIA H800 SXM5 GPU has a peak memory bandwidth of 3.35 TB/s and peak FLOPs of 990 TFlops. However, due to throttling (reducing to ~1600 MHz in our case), the practical peak FLOPs drops to ~865 TFlops. Therefore, when $h_qs_q \ge \frac{1}{2} \cdot \frac{865}{3.35} = 128$, the kernel is compute-bound; otherwise, it's memory-bound.

> The output matrix (scaled and accumulated during each mainloop round, similar to FlashAttention's algorithm) must be stored in registers due to WGMMA instruction requirements. Each $64 \times 512$ output matrix occupies 32,768 32-bit registers. With only 65,536 32-bit registers per SM, we can store only one output matrix per SM. This eliminates the possibility of having two output matrices and letting them use CUDA Core and Tensor Core in a interleaved manner.

The "seesaw" schedule across two warpgroups sharing one running max:

> 0. Maintain a running max $m$ (initialized to $-\infty$, shared between the two warpgroups) and output matrices $\vec o_L, \vec o_R$ (initialized to 0).
> 1. [0] Compute $\vec p_0 = \vec q K_0^\intercal / qk\_scale$.
> 2. [1] Compute $\vec p_1 = \vec q K_1^\intercal / qk\_scale$.
> 3. [0] Compute $mp_0 = \max(\vec p_0)$, $m\_new_0 = \max(m, mp_0)$, and $scale_0 = \exp(m\_new_0 - m)$. Update $m \gets m\_new_0$.
> ...
> 11. [0] Update $\vec o_L \gets \vec o_L \cdot scale_1 + \vec p_1 V_{1L}$.

> - Fine-grained TMA copy - GEMM pipelining: For a $64 \times 576$ K block, we launch 9 TMA copies (each moving a $64 \times 64$ block). GEMM operations begin as soon as each TMA copy completes ... improving memory latency tolerance.
> - Cache hints: Using `cute::TMA::CacheHintSm90::EVICT_FIRST` for TMA copies improves L2 cache hit rates, as shown by experiments.

> These optimizations achieve up to 80% Tensor Core utilization (of the throttled theoretical peak) and 3 TB/s memory bandwidth on an H800 SXM5 GPU.
> ...
> - Programmatic Dependent Launch. We use programmatic dependent launch to overlap `splitkv_mla` and `combine` kernels.
> - Tile Scheduler. We implement a tile scheduler to allocate jobs (requests and blocks) to SMs. This ensures a balanced load across SMs.

## Relevance

`ratio ≈ 2·h_q·s_q` is the formula that decides whether MLA decode is overhead-bound at our
operating point. GLM-5.2 has 64 Q heads; at TP16 that is 4 live heads per die, and with `s_q=1`
the ratio is 8 — two orders of magnitude below the 128 threshold at which the kernel would be
compute-bound. So MLA decode at our configuration is deep in the latency/overhead regime, which
is consistent with SparseFlashAttention at 6.973 ms against 1.382 ms of bytes. DeepSeek's reason
for keeping `h_q` at 128 (no attention TP) is the same observation read in the other direction.
The fine-grained `9 x TMA(64x64)` pipelining of a `64 x 576` K block is the pattern for hiding
C8 cache-load latency inside our attention stage.

### FlashMLA's fused norm + RoPE + attn + inverse-RoPE + FP8-cast kernel

https://github.com/deepseek-ai/FlashMLA#fused-norm--rope--attn--rope--cast-kernel
DeepSeek, 2026.09.10 / 2026.09.30 release notes.

> In the DeepSeek-V4.1 release, we also provide a fused kernel that combines Q-norm (only used in V4, not in V4.1), Q-RoPE, core attention, O-RoPE (conjugate) and the cast to FP8 into a single kernel. It removes the extra time spent on these small kernels while keeping the same or even slightly higher TFlops, at the cost of having to permute the Q_b and Wv weights in advance.
>
> In DeepSeek-V4.1 attention, Q (`[hidden_size]`) is first projected to `[q_lora_rank]` (the Q_a projection) and then to `[num_attention_heads, head_dim]` (the Q_b projection). After core attention, the output (`[num_attention_heads, head_dim]`) is reshaped to `[o_groups, num_attention_heads // o_groups * head_dim]`, and each of its rows is projected to `[o_lora_rank]` (the Wv projection), giving an `[o_groups, o_lora_rank]` matrix. That matrix is reshaped to `[o_groups * o_lora_rank]` and finally projected to `[hidden_size]` (the Wo projection). This kernel requires the Q_b and Wv weights to be permuted.

```python
out_fp8, out_sf, lse = fused_norm_rope_attn_rope_cast.decode(
    enable_q_norm, rms_norm_eps,
    token_positions, False, 64, cos_sin_cache,
    n_wv_group, 32, True, True, True,
    q, k_cache, indices_in_kvcache,
    sm_scale=sm_scale, attn_sink=attn_sink, topk_length=topk_length,
    extra_k_cache=extra_k_cache,
    extra_indices_in_kvcache=extra_indices_in_kvcache,
    extra_topk_length=extra_topk_length,
)

# The FP8 output is consumed directly by the Wv projection, using the permuted Wv weight
wv_out = torch.empty((s_q, n_wv_group, wv_proj_out_dim), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (wv_proj_fp8, wv_sf), wv_out, recipe=(1, 1, 32))
```

> We also optimize the performance of the fused norm-RoPE-attn-RoPE-cast kernel under decoding settings by around 10% - 15%.

## Relevance

The concrete answer on fusing o_proj into the attention epilogue: DeepSeek deliberately stops
short. The attention kernel absorbs Q-norm, Q-RoPE, O-RoPE and the FP8 cast — so the output
leaves the kernel already quantized with its scale factors — while the Wv/Wo projection remains
a separate einsum consuming that FP8 directly. The price is an offline permutation of Q_b and
Wv. For us: fuse norm/RoPE/cast into attention and hand the projection a pre-quantized
activation (which is what `MlaPrologV3`'s `queryQuantMode=1` + `dequantScaleQNopeOut` already
does on the input side), rather than pulling the projection GEMM inside.

---

### MLA absorption — DeepSeek-V2 paper, and why RoPE obstructs it

https://arxiv.org/abs/2405.04434 (HTML: https://arxiv.org/html/2405.04434v5), Section 2.1.2 and Appendix C.

> c_t^{KV} = W^{DKV} h_t,                         (9)
> k_t^{C}  = W^{UK} c_t^{KV},                     (10)
> v_t^{C}  = W^{UV} c_t^{KV},                     (11)
> where c_t^{KV} ∈ R^{d_c} is the compressed latent vector for keys and values; d_c (≪ d_h n_h) denotes the KV compression dimension; W^{DKV} ∈ R^{d_c×d} is the down-projection matrix; and W^{UK}, W^{UV} ∈ R^{d_h n_h × d_c} are the up-projection matrices for keys and values, respectively.
> During inference, MLA only needs to cache c_t^{KV}, so its KV cache has only d_c l elements, where l denotes the number of layers.
> In addition, during inference, since W^{UK} can be absorbed into W^{Q}, and W^{UV} can be absorbed into W^{O}, we even do not need to compute keys and values out for attention.

> In this way, W^{UK} cannot be absorbed into W^{Q} any more during inference, since a RoPE matrix related to the currently generating token will lie between W^{Q} and W^{UK} and matrix multiplication does not obey a commutative law. As a result, we must recompute the keys for all the prefix tokens during inference, which will significantly hinder the inference efficiency.

Appendix C:

> During inference, the naive formula needs to recover k_t^{C} and v_t^{C} from c_t^{KV} for attention.
> Fortunately, due to the associative law of matrix multiplication, we can absorb W^{UK} into W^{UQ}, and W^{UV} into W^{O}.
> Therefore, we do not need to compute keys and values out for each query.
> Through this optimization, we avoid the computational overhead for recomputing k_t^{C} and v_t^{C} during inference.

## Relevance

The absorption identity and its RoPE obstruction — the reason decoupled RoPE exists, and the
reason an absorbed decode kernel needs separate `nope` and `pe` code paths (which is exactly
what `aclnnSparseFlashAttention`'s paired `queryRopeOptional`/`keyRopeOptional` with `Dr=64`
encodes, and what `MlaProlog` splits via the concatenated `[W^UQ|W^QR]` weight).

---

### MLA absorption as runnable einsums, with a FLOPs/footprint table showing decode is ~1% of naive

https://github.com/flashinfer-ai/flashinfer/pull/551
flashinfer-ai/flashinfer, author `tsu-bin`, 2024-10-23, "feat: support MLA decode".

> ### The mystery Mat Absorb algorithm
> In the DeepSeekV2 paper, there was no specific formulas for how to do param matrixes absorption, but it just vaguely said that
> > Fortunately, due to the associative law of matrix multiplication, we can absorb 𝑊𝑈𝐾 into 𝑊𝑈𝑄, and 𝑊𝑈𝑉 into 𝑊𝑂
> ...
> Here is my conclusion on this topic, Mat Absorb is only suitable for decode, do not use Mat Absorb for prefill, which means MLA decode and prefill should have different computation graph and different set of params, and Mat Absorb should merge param matrixes offline, materialize the merged param matrixes.

```python
# Now we merge W_UQ and W_UK (absorb W_UK into W_UQ)
# q~q_lora_rank  n~num_heads  d~qk_nope_head_dim  l~kv_lora_rank
self.W_UQ_UK = torch.einsum("q n d, l n d -> q n l", W_UQ, W_UK).flatten(start_dim=1) # [1536, 65536]

# Merge W_UV and W_O (absorb W_UV into W_O)
# l~kv_lora_rank  n~num_heads  d~v_head_dim  h~hidden_size
self.W_UV_O = torch.einsum("l n d, h n d -> n l h", W_UV, W_O).flatten(start_dim=0, end_dim=1) # [65536, 5120]
```

> And after Mat Absorb, MLA become a special 128 heads / (512+64) dim MQA algorithm, please note that the compressed_kv is used as both K and V directly without any projection.

```python
def num_float_ops_mat_absorb(q_len, kv_len):
    return ( q_len*1536*(128*64) + # from c_Q to q_pe, corresponding to W_QR
                q_len*1536*(128*512) + # from c_Q to q_nope, corresponding to W_UQUK
                128 * (q_len*64*kv_len + q_len*512*kv_len + q_len*kv_len*512) + # 128 heads MQA
                q_len * (128*512) * 5120 ) # from MHA output to output hidden states, corresponding to W_UV_O
```

> prefill: num_float_ops mat_absorb vs vanilla ratio  ~ 3.3602009088734754
> prefill: mem_footprint mat_absorb vs vanilla ratio  ~ 2.2874373717252205
> decode: num_float_ops mat_absorb vs vanilla ratio  ~ 0.010941137164898957
> decode: mem_footprint mat_absorb vs vanilla ratio  ~ 1.167867978048944

> So we can conclude from the result above, for decode case Mat Absorb version only use about 1% computation compared to Vanilla version, and the memory footprint is at the same level with Vanilla version, but for prefill case, both computation and memory footprint are much higher than Vanilla version, so there is no reason to use Mat Absorb for prefill, but it's worth a try for decode.

> There is one problem that the 128heads / (512+64)dim Q data is too large to fit into one SM's register file or even smem, which means we can't use only one SM/block to process one Q data, we have to tile the num_qo_heads dimension into gridDim.y, which can cause kv-cache data movement from gmem to smem multiple times, though this is inevitable.

> Tensor-core version implementation ... but in my opinion maybe this can have limited performance improvement, because consider the above analysis, the bottle neck is IO bandwidth not the computation intensity.

## Relevance

The absorption written as runnable einsums with offline materialisation, plus the quantified
verdict: decode uses ~1.1% of the naive FLOPs at the same memory footprint. Two consequences
for us. First, `Mat Absorb should merge param matrixes offline, materialize the merged param
matrixes` — but note CANN goes the other way: `MlaProlog` keeps `W^UK` separate and applies it
in-kernel as `MatmulQn`, which is the right choice when `W_UQ_UK` would be `[1536, 65536]`.
Second, the closing judgement — `the bottle neck is IO bandwidth not the computation
intensity`, so the matrix unit buys little on the absorbed path — argues for keeping the
absorbed-path arithmetic on the vector units, where our per-call floor is 0.70 us rather than
13.58 us.

---

### FlashInfer `BatchMLAPagedAttentionWrapper` — the absorbed-formulation contract and graph-replay metadata

https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/flashinfer/mla/_batch_mla/_wrapper.py

```python
class BatchMLAPagedAttentionWrapper:
    r"""Wrapper for MLA PagedAttention on DeepSeek models.

    This wrapper is intended for decode and incremental prefill with the
    Matrix Absorption formulation of MLA, where the query/key and value/output
    projections are absorbed before attention. For the non-absorbed MLA
    prefill path, use the appropriate prefill wrapper instead.

    The planned-wrapper surface owns FA2, FA3, CUTLASS, cuTile, TRTLLM-GEN,
    XQA, and CuTe DSL planning and execution. Call :meth:`plan` once with
    canonical metadata before invoking :meth:`run`; the plan captures the
    supported input/output contract and the concrete backend's metadata
    representation.
    """
```

```python
    def plan(
        self, *,
        metadata: MLAPlanMetadata,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
        query_layout: Literal["packed", "split"] = "packed",
        kv_cache_layout: Literal["packed", "split"] = "packed",
        lse_mode: Literal["none", "base2", "basee"] = "none",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal["default", "kv-per-tensor", "bmm-scalar", "bmm-tensor"] = "default",
        skip_softmax: bool = False,
        enable_pdl: Optional[bool] = None,
        use_sinks: bool = False,
    ) -> None: ...
```

> TRTLLM-GEN, XQA, and CuTe
> DSL CUDA graph plans require supplied contiguous ``int32`` dense
> metadata on the wrapper device. Retain and update those device tensors
> in place for replay; CPU and CSR-only graph metadata are rejected.
> Query offsets and lengths must remain fixed after graph planning.

## Relevance

The split of `head_dim_ckv` from `head_dim_kpe` in the interface is the API-level fingerprint of
the absorbed decode path, matching CANN's `D=512` / `Dr=64`. `Retain and update those device
tensors in place for replay` plus `Query offsets and lengths must remain fixed after graph
planning` is the same address-stability discipline the vllm-ascend GLM-5.2 MegaKernel RFC
states for DAG replay.

---

## 5. Megakernel mechanism: how a persistent kernel schedules a whole decode step

### ThunderMLA — a megakernel for decode attention driven by an instruction tensor

https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla
Benjamin Spector, Aaryan Singhal, Dan Fu, Chris Ré; Mar 4, 2025.

> ThunderMLA is a completely fused "megakernel" for decode that's 20-35% faster than DeepSeek's FlashMLA on diverse workloads. It turns out some simple scheduling tricks can get you pretty far!

> The core issue lies in the substantial overhead created by:
> Running two separate kernel launches
> Tail effects between kernel executions
> Limited batch sizes when running large reasoning models
> ...
> A batch of 4 prompts, of lengths [4641, 45118, 1730, 1696]
> Generating 4 new tokens (e.g., for a speculator)
> 8-way tensor parallel -- that is, 16 heads per GPU for DeepSeek R1
>
> With 4 new tokens, FlashMLA runs in 52 us on an SXM H100, achieving just 144 TFLOPS and 1199 GB/s (out of the 939 TFLOPS / 3300 GB/s advertised by NVIDIA).
> ...
> on the same workload, ThunderMLA runs in just 41 us, achieving 183 TFLOPs and 1520 GB/s.

> Our solution eliminates these kernel launches by creating a simple virtual instruction set on the GPU. Instead of launching separate kernels, we fuse them into a single "megakernel," and then pass this megakernel its instructions at runtime through an instruction tensor, which in turn can dramatically modify its behavior. Within the megakernel, GPU SMs read from this instruction tensor, decide what work to perform next, and then execute it.

> Our implementation adapts a slot attention kernel and token reduction kernel into this interpreter template with just a few hundred lines of device code, and uses a global tensor as a semaphore to synchronize dependencies across instructions. Once both the two constituent kernels ("partial" and "reduction") of the megakernel for FlashMLA are implemented, the megakernel can be instantiated as interpreter::kernel<config, partial_template, reduction_template>. Altogether, this fused megakernel is just 250 lines of device code in ThunderKittens.

> Key benefits include:
> Vastly simplified pipelining within instructions, across loop boundaries
> Asynchronous fetching of the future instructions, minimizing instruction overhead
> Deep pipelines that hide data fetch latencies within and across instructions

## Relevance

The minimal viable megakernel, and the natural first increment for us: fuse only the split-KV
partial and the reduction, driven by a runtime instruction tensor, 250 lines of device code,
20-35% over a hand-tuned kernel. On Ascend the semaphore is a workspace counter plus
`CrossCoreWaitFlag`; the instruction tensor is a device-resident schedule built once per shape
(exactly what CANN's `flash_mla_metadata` already is).

---

### "Look Ma, No Bubbles!" — the full cost accounting of a decode megakernel

https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
Benjamin Spector*, Jordan Juravsky*, Stuart Sul, Owen Dugan, Dylan Lim, Dan Fu, Simran Arora, Chris Ré; May 27, 2025.

Why kernels stall, verbatim:

> First: GPU kernels are launched with a strict ordering, so that a thread block in one kernel can't start until all thread blocks in previous kernels have completely finished. Consequently, every time we start a kernel, we have to wait for all the straggler thread blocks from the prior one to finish. For example, if a kernel runs 512 thread blocks (like our Llama-1B down projection), but we only have 148 streaming multiprocessors (like on a B200), we end up with 80 empty SM's at the end.
>
> Second, as we've previously highlighted, each kernel launch and teardown incurs costs. In principle, NVIDIA's CUDA graphs can help hide costs, but by our measurements they still leave a lot on the table. For a simple dummy kernel (which dumps a start time, sleeps, and dumps an end time) on an H100, we find that running on a CUDA stream incurs a launch cost of about 2.1 microseconds, and with CUDA graphs the launch cost only decreases to around 1.3 microseconds – time spent with the GPU doing no useful work!
>
> Finally, even after we start the next kernel, we still have to wait to load weights and activations before any compute can start. These latencies leave the GPU sitting idle for thousands of cycles! Ideally, we'd start loading the next weights while the previous computations and stores are happening. NVIDIA has also built a mechanism for this called Programmatic Dependent Launch (PDL), which allows the next kernel to start preparing while the previous kernel is running, but we found it still introduces unnecessary stalls because the PDL synchronization mechanism (cudaGridDependencySynchronize) is very coarse. For example, it means we have to wait for all queries, keys, and values to complete in order to start attention, as opposed to starting heads as soon as they are ready.

> To illustrate the magnitude of the problem: for single-sequence generation in 16-bit precision on a single H100, the memory limit is 3.35TB/s / 2.48GB = ~1350 forward passes per second. But with 7 kernel launches per layer, and 16 layers, even with an optimistic 5 us of stalling per kernel (counting stragglers, kernel launch, and memory latencies), generation would run at just ~770 forward passes per second.

The instruction/interpreter model:

> Our interpreter is designed such that each streaming multiprocessor (SM) within the GPU receives a sequence of instructions (each implemented using the same CUDA template) and executes them. We schedule each SM's instruction sequence ahead of time on the Python side, and notably we can reuse each schedule for hundreds of forward passes!
>
> For our end-to-end Llama forwards pass megakernel, we define the following set of instructions:
> A fused RMS norm & QKV & RoPE instruction.
> An attention computation instruction.
> An attention reduction instruction (for ThunderGQA on long sequences).
> An O-projection + residual instruction.
> A fused RMS norm & up-gate & SiLU instruction.
> A down-projection + residual instruction.
> An RMS norm & language modeling head instruction, for computing the final token logits.

Paged shared memory:

> We accomplish this by paging shared memory. We first divide the first 213kB of shared memory on an H100 into 13 16KiB pages, and use remaining shared memory for special purposes, like storing instruction parameters. To use one of these pages, instructions have to explicitly request and release them from the interpreter. The interpreter automatically passes released pages to the next instruction, allowing them to start issuing memory loads as early as shared memory becomes available.

Synchronization by counters, and the chunked-MLP optimization it enables:

> We accomplish this with a simple counter system. Before the megakernel launches, we initialize an array of counters (i.e. integers) in GPU global memory with a starting value of zero. Whenever an instruction completes, it increments one of these counters. Similarly, whenever a new instruction starts, it must wait for some of these counters to reach a target value, indicating that all of its dependencies have finished.
>
> One optimization this enables is in the big multi-layer perceptrons (MLPs) in Llama-1B. In a naive implementation using PDL, one must await completing the whole hidden state before beginning the down projection matrix multiply. We instead produce and consume the intermediate state in four chunks, each with their own counter. This way, an instruction for the down projection only needs to wait for its input chunk to finish.

Results, and the full residual-cost breakdown:

> Doing this achieves brr – on an H100, we use 78% of memory bandwidth and outperform existing systems by over 1.5x.

> To our knowledge, our H100 megakernel represents the first time anyone has run the forward pass for a 16-bit 1B+ parameter language model in under one millisecond on a GPU. Our B200 implementation pushes this even further to under 680 microseconds per forward pass!
> ...
> On an H100, our megakernel runs almost 2.5x faster than vLLM and over 1.5x faster than SGLang.
> On a B200, the gap with vLLM rises to over 3.5x, and we remain more than 1.5x faster than SGLang, too.

> A breakdown of the runtime of our current B200 forward pass (total runtime 600 microseconds):
> 250 microseconds are spent storing activations, awaiting consistency, and loading them. This is about 20% higher than a simple model would suggest: since each instruction has a dependence on the last one, we need to pay two load latencies (check ready, and then load activations) and two store latencies (store activations, then mark ready) per instruction. Using ~500 nanoseconds latency per load / store, this would impose about 200 microseconds of overhead. (We suspect some of the remaining 50 microseconds comes from time spent processing atomics in global memory.)
> 200 microseconds are spent actually running RMS norm and matrix-vector computations. 95% of this portion is devoted to matrix-vector. On Blackwell, we find that using the tensor cores is marginally helpful for this; on Hopper, we find it better to simply run on the CUDA cores. This difference comes from the fact that both GPUs have relatively similar CUDA core performance, but Blackwell tensor cores are much faster.
> 30 microseconds are spent awaiting weights from global memory (pipelining works!) Of these, 40% are spent in the LM head, which is the best-pipelined part of the whole megakernel due to its homogeneity and huge size.
> 40 microseconds are spent on low-level synchronization overhead across warps. A key issue here is that CUDA's asynchronous barriers are relatively slow, even when they're already in the "pass" state, requiring about 60 nanoseconds each time.
> 80 microseconds are on setup and various other overheads (e.g. passing instruction barriers, marking pages as complete, etc.)

## Relevance

The most transferable budget in the literature, and a caution. After fusing everything,
250/600 us is still activation store + ready-flag + load latency — two load and two store
latencies per instruction — and 40 us is barrier cost at ~60 ns per barrier. For us that means
the win is bounded by instruction count and flag round-trips, not launch count alone, so the
megakernel's instruction granularity is the design variable. Two directly applicable findings:
chunk the intermediate state and give each chunk its own counter so the down projection starts
per chunk (the Ascend analogue of DeepGEMM's L2 bitmask, and of our `pre/mid/post` pipeline);
and on Hopper matvec is better on CUDA cores than tensor cores — the Ascend analogue is
AI_VECTOR_CORE at a 0.70 us floor versus MIX_AIC at 13.58 us.

---

### TP megakernel for Llama-70B — and the finding that fusing o_proj under TP is the wrong move

https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
and https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro — same authors, Sep 28, 2025.

> However, we've made one important change to this formulation. One of our targets for operation overlapping is to overlap the O projection matrix multiplication with the subsequent reduce-scatter. However, the tensor-parallel sharded O matrix is too small for us to effectively hide the reduce-scatter communication cost. To solve this, we instead choose to replicate the O projection matrix across each GPU and run the O projection with data parallelism instead of with tensor parallelism. Alongside this change, we eliminate the post-attention reduce-scatter and replace it with a "distributed transpose" operation after attention that repartitions our data from a tensor-parallel configuration to a data-parallel configuration.
>
> When using 8 GPUs, this reduces the network traffic by a factor of 8, which makes it much easier to hide the cost of communication cost by overlapping it with matrix multiplications. Note that the downside of this approach is it reduces the maximum batch size by about 15%, because replicating the O projection weights consumes an additional 9 GB of memory per GPU.

The TP instruction set:

> RMS norm + all-gather.
> A QKV matrix multiply + RoPE.
> Attention + distributed transpose.
> O-projection matrix multiply + residual.
> Gate matrix multiply + SiLU.
> Up-projection matrix multiply + elementwise multiplication with the output of the gate.
> Down matrix multiply + reduce-scatter + residual.
> RMS norm without the all-gather (for the LM head)
> LM head matrix multiply.

On recomputation versus communication, and on dependency granularity:

> Most of our instructions for low-latency centered around matrix-vector multiplication, rather than the matrix-matrix multiplications we do here. ... To avoid extra trips to global memory, for our low-latency megakernel we frequently recomputed results across the GPU rather than communicating them through memory. This allowed us to fuse operations more aggressively than usual. For example, our QKV matrix-vector product fused the RMS norm into its beginning, saving an instruction boundary, but performing identical RMS computations over a hundred times. When focusing on throughput, this recomputation is not worth it.

> Putting many tokens in our batch requires us to expand our inter-instruction signalling scheme to track data dependencies across tokens. This signalling can vary by instruction -- for some operations (e.g. attention, RMS norms), sometimes synchronization is at the granularity of 128 output rows of a matrix; other times it's at the granularity of an individual attention head for an individual token.

Static per-SM queues versus a global atomic work queue:

> With our low-latency megakernel, each SM was assigned its own queue of instructions that are scheduled in advance. Instead, for our high-throughput megakernel, we create a global work queue -- a single list of instructions that defines all the work that needs to run on the GPU. When an SM needs to fetch a new instruction to run, it atomically increments a global instruction counter that keeps track of the next instruction to be assigned. ... This solution wasn't possible for our low-latency megakernel, because the runtime of each instruction was so fast that the latency of this atomic increment would be prohibitive.

> We perform all communication from our dedicated storer threads, allowing loader and compute threads to move onto future work while inter-GPU communication is performed in the background on the same SM.

> Whereas with instruction pipelining enabled, the extra gap between consecutive matrix multiply stages is just 3.4 microseconds, without the pipeline, this gap jumps to 10.2 microseconds -- meaning this optimization alone reduces runtime by over 7% on these instructions.

## Relevance

Directly answers the TP16 question. Fusing o_proj into the attention epilogue is the wrong move
when o_proj is TP-sharded: they instead replicate o_proj, run it data-parallel, and replace the
post-attention reduce-scatter with a distributed transpose folded into the attention
instruction — 8x less wire traffic. Also decisive for our batch=1 case: a global atomic work
queue is too slow when instructions are short, so use static per-core instruction queues.
And the licence for aggressive fusion at low latency: recomputing RMS norm a hundred times to
save one instruction boundary was the right trade.

---

### MPK — a compiler and runtime that mega-kernelizes the whole model, with the launch-count arithmetic

https://arxiv.org/abs/2512.22219
`MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs` — Xinhao Cheng, Zhihao Zhang, Yu Zhou, Jianan Ji, Jinchen Jiang, Zepeng Zhao, Ziruo Xiao, Zihao Ye, Yingyi Huang, Ruihang Lai, Hongyi Jin, Bohan Hou, Mengdi Wu, Yixin Dong, Anthony Yip, Songting Wang, Wenqin Yang, Xupeng Miao, Tianqi Chen, Zhihao Jia. Submitted 22 Dec 2025, v2 10 Jun 2026. OSDI '26: https://www.usenix.org/conference/osdi26/presentation/cheng

> We introduce Mirage Persistent Kernel (MPK), the first compiler and runtime system that automatically transforms multi-GPU model inference into a single high-performance mega-kernel. MPK introduces an SM-level graph representation that captures data dependencies at the granularity of individual streaming multiprocessors (SMs), enabling cross-operator software pipelining, fine-grained overlap of computation and communication, and other optimizations that are infeasible under the conventional kernel-per-operator execution model. The MPK compiler lowers tensor programs into optimized SM-level task graphs and generates fast CUDA implementations for each task, while the MPK in-kernel parallel runtime executes these tasks within a single persistent mega-kernel using decentralized scheduling across SMs. Together, these components provide end-to-end kernel fusion with minimal developer effort, while preserving the flexibility of existing programming models. Our evaluation shows that MPK significantly outperforms existing kernel-per-operator LLM serving systems, achieving up to 1.7x lower end-to-end inference latency and pushing LLM inference performance close to the limits of the underlying hardware.

Task/event model:

> Each task—shown as a blue (or orange) rectangle—denotes a unit of computation (or communication) executed on a single SM. Each event—shown as a green circle—represents synchronization across tasks. Tasks and events alternate in the graph: every task only has outgoing edges to triggering events ... A task is ready for execution when its dependent events are all [activated] and notifies its triggering event upon completion. An event is activated once it has received notifications from all tasks associated with it.

> For any two events e1 and e2 in a tGraph, successor-set fusion applies if and only if OutTasks(e1)=OutTasks(e2). MPK removes e1 and e2 from the tGraph and introduces a fused event e′ with InTasks(e′)=InTasks(e1)∪InTasks(e2) and OutTasks(e′)=OutTasks(e1).

> Normalization bounds the dependency metadata stored for each task. When every task depends on and triggers at most one event, each task descriptor needs to store only one dependent-event identifier and one triggering-event identifier, rather than variable-length lists.

> MPK addresses this challenge using a breadth-first-search-based algorithm ... to linearize a tGraph. The algorithm assigns contiguous indices to all tasks triggered by the same event in the final task ordering. As a result, the fan-out of an event can be encoded compactly using only the first and last task indices, eliminating the need to store an explicit list of dependent tasks while preserving all dependencies.

Runtime structure and the JIT/AOT launch distinction:

> To support this execution model, MPK partitions a GPU's SMs into workers and schedulers. Each worker runs on one physical SM and maintains an independent task queue. ... Schedulers are organized at warp granularity, with each SM hosting four scheduler warps. Each scheduler maintains an event queue and repeatedly polls for newly activated events, dispatching the corresponding tasks to workers. The allocation of workers and schedulers is fixed at kernel-launch time and matches the GPU's physical SM count, avoiding any dynamic role-switching overhead inside the kernel.

> Under JIT launch (Figure 8a), the worker executing T1 notifies a scheduler, which then dequeues event e, launches T2, and enqueues it into a worker's task queue. The receiving worker must then dequeue T2 before execution. This chain requires two synchronization steps (worker→scheduler and scheduler→worker). By contrast, under AOT launch (Figure 8b), T2 has already been enqueued on a pre-assigned worker, which only needs to wait for event e to be activated. Thus, AOT launch requires only one synchronization step through the event trigger, reducing per-task launch latency.

> Operators with data-dependent durations (e.g., attention) are marked as JIT, and their downstream operators remain JIT until execution reaches a global barrier (i.e., an event that must be triggered by all upstream tasks). Such barriers eliminate accumulated imbalance, making subsequent operators suitable for AOT launch. All remaining operators are labeled AOT to minimize dispatch overhead.

> A task may acquire one or more pages based on its shared-memory footprint and must release the pages when they are no longer needed. Once a task releases any page, it is no longer permitted to acquire additional shared-memory pages, enforcing a monotonic usage pattern that simplifies scheduling.

> Every task is associated with a task description that encodes its input tensors, output tensors, and configuration parameters; in our current implementation, each description occupies 352 bytes

Configuration and measured numbers:

> GPU  # SMs  # Workers  # Schedulers
> A100  108  104  16
> H100  132  128  16
> B200  148  144  16
> ...
> We set the shared-memory page size to 32 KB on all GPUs, which yields 5, 7, and 7 shared-memory pages per SM on A100, H100, and B200, respectively.

> For example, on Qwen3-8B running on an A100 GPU, MPK reduces per-token decoding latency from 14.5 ms, achieved by highly optimized systems such as vLLM and SGLang, to 12.5 ms. This approaches the approximate hardware lower bound of 10 ms, estimated by loading 16 GB of model parameters at 1.6 TB/s memory bandwidth.

> MPK executes the entire model with a single kernel launch, whereas a kernel-per-operator execution of Qwen3-8B issues 293 kernel launches per token. On B200, each launch costs 3.8 μs in eager execution, totaling 1.1 ms per token, and 0.8 μs with CUDA Graphs, totaling 0.2 ms per token. MPK avoids this overhead; its in-kernel scheduler accounts for only 0.28% of total runtime

> in Qwen3-8B, 293 operators are decomposed into 13,867 tasks. ... across the evaluated models, the final tGraphs contain only 1,142–2,366 events, yet these events encode 69,000–162,000 producer–consumer task-pair dependencies. This corresponds to a 37–118x reduction in synchronization events

The MoE case study:

> For Qwen3-30B-A3B at batch size one on SGLang, this preprocessing step accounts for up to 11% of total MoE execution time. In MPK, implementing this step as a separate preprocessing task would also introduce additional scheduling overhead. MPK addresses this issue by replacing the TMA-based gather with an asynchronous token-level copy integrated directly into the data-loading phase of the GEMM tasks.

> Cross-task pipelining reduces task runtime by 1.2–1.3x and even outperforms cuBLAS-based compiled kernels.

## Relevance

The complete reusable scheduling architecture, with the one design decision that matters most
at batch=1: JIT dispatch costs two synchronization hops and is only worth it for
data-dependent operators (attention); everything else should be pre-enqueued AOT so the runtime
cost is one event-flag wait. Also directly implementable as an Ascend instruction table:
fixed-size task descriptors (352 bytes there), exactly one dependent and one triggering event
per task after normalization, and contiguous-range event fan-out so no dependent-task lists are
stored. The MoE note — fold the gather into the GEMM task's data-loading phase rather than
making it a separate task — is the same conclusion as deleting our `moe_init_routing` stage.

---

### ForgeMegakernel and Event Tensor — the achievable MBU band, and dynamism as the hard part

https://arxiv.org/abs/2609.12379
`ForgeMegakernel` — Leshan Li, Zhui Zhu, Xianglong Deng, Yaojian Chen, Qingfeng He, Yuxuan Li, Rong Zhao, Xu Han, Zhiyuan Liu; submitted 11 Sep 2026.

> Auto-regressive model decode is bandwidth-bound, since every weight and key/value-cache byte crosses high-bandwidth memory once per token. A megakernel is an ideal solution, but existing automatic megakernel generation approaches cannot achieve both generalization across models and correctness guarantees.
> We present ForgeMegakernel, which generates a per-model high-performance decode megakernel using coding agents. ForgeMegakernel pairs a universal knowledge base of ten progressive milestones with an independent mid-state test oracle. The milestones provide the megakernel's structural properties: a fine-grained instruction stream for each SM, dependency counters replacing the global synchronization, and a shared-memory buffer pool for workload balance across SMs and greater parallelism. The test oracle derives the mid-states of the megakernel and checks the performance, error and precision during the generation process, guaranteeing a correct and trustworthy forged megakernel.
> We evaluated ForgeMegakernel on 14 representative decoding operations across eight model families spanning 0.6B-13B parameters. The generated megakernels achieved 50.5-85.9% MBU and geometric mean speedups of 1.21x over SGLang 0.5.18 and 1.54x over a megakernel compiler under identical configurations.

https://arxiv.org/abs/2604.13327
`Event Tensor` — Hongyi Jin, Bohan Hou, Guanjie Wang, Ruihang Lai, Jinqi Chen, Zihao Ye, Yaxing Cai, Yixin Dong, Xinhao Cheng, Zhihao Zhang, Yilong Zhao, Yingyi Huang, Lijie Yang, Jinchen Jiang, Gabriele Oliaro, Jianan Ji, Xupeng Miao, Vinod Grover, Todd C. Mowry, Zhihao Jia, Tianqi Chen; submitted 14 Apr 2026, v2 21 Apr 2026.

> Modern GPU workloads, especially large language model (LLM) inference, suffer from kernel launch overheads and coarse synchronization that limit inter-kernel parallelism. Recent megakernel techniques fuse multiple operators into a single persistent kernel to eliminate launch gaps and expose inter-kernel parallelism, but struggle to handle dynamic shapes and data-dependent computation in real workloads. We present Event Tensor, a unified compiler abstraction for dynamic megakernels. Event Tensor encodes dependencies between tiled tasks, and enables first-class support for both shape and data-dependent dynamism. Built atop this abstraction, our Event Tensor Compiler (ETC) applies static and dynamic scheduling transformations to generate high-performance persistent kernels. Evaluations show that ETC achieves state-of-the-art LLM serving latency while significantly reducing system warmup overhead.

## Relevance

Independent convergence on the same three structural primitives — per-SM instruction stream,
dependency counters replacing global synchronization, shared-memory buffer pool — and a
realistic ceiling to calibrate the 2.x ms target: 50.5-85.9% MBU. Event Tensor names the part
that is genuinely hard for us: data-dependent dynamism, i.e. MoE routing and the sparse
attention index, not the fusion itself.

---

## 6. `KvQuantSparseFlashAttention` — the C8-aware SFA, with dequant and the scale packed into the cache slot

### KvQuantSparseFlashAttention — supported on A3, explicitly rewritten for the discrete-access problem

https://gitcode.com/cann/ops-transformer/blob/master/attention/kv_quant_sparse_flash_attention/README.md
CANN `ops-transformer`, master.

> # KvQuantSparseFlashAttention
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
> - API功能：`kv_quant_sparse_flash_attention`在`sparse_flash_attention`的基础上支持了[Per-Token-Head-Tile-128量化]输入。随着大模型上下文长度的增加，Sparse Attention的重要性与日俱增，这一技术通过"只计算关键部分"大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。
>
> - 计算公式：
>
>     $$
>     Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
>     $$
>
>     其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`LightningIndexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。

The statement of what was optimized:

> 本次公布的`kv_quant_sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

The cache layout — note that the dequant scale is packed into the same slot as nope and rope:

> | query | 输入 | attention结构的Q输入，不支持非连续。query由相同数据类型的q_nope和q_rope按D维度拼接得到。layout_query为"BSND"时shape为[B, Q_S, Q_N, Q_D]。layout_query为"TND"时shape为[Q_T, Q_N, Q_D]。其中Q_D值仅支持576，即q_nope+q_rope=512+64；Q_N值支持1/2/4/8/16/32/48/64/128。 | FLOAT16、BFLOAT16 | ND |
>
> | key | 输入 | attention结构的K输入。k_nope、query相同数据类型的k_rope和float32的量化参数按D维度拼接得到。layout_kv为"BSND"时shape为[B, KV_S, KV_N, KV_D]。layout_kv为"TND"时shape为[KV_T, KV_N, KV_D]。layout_kv为"PA_BSND"时shape为[block_num, block_size, KV_N, KV_D]，其中block_num为PageAttention时block总数，block_size为一个block的token数，block_size取值为16的整数倍，最大支持到1024。KV_N仅支持1；KV_D值仅支持656，即nope+rope*2+dequant_scale*4=512+64*2+4*4。TQ4模式（key_quant_mode=3）时改为使用386字节packed槽位。 |

Quant modes:

> | key_quant_mode | ... |
> | value_quant_mode | 代表value的量化模式，传入2代表per_tile量化模式；新增传入3表示TQ4 INT4量化，且要求key_quant_mode同时传入3。 |
>
> | rope_head_dim | 表示MLA架构下的rope_head_dim大小，仅在attention_mode为2时有效，仅支持64。默认值为64。 |

The TQ4 (TurboQuant 4-bit) slot layout:

> ### INT4（TQ4）量化类型
>
> 本算子新增 TurboQuant 4-bit（TQ4）量化类型。当 `key_quant_mode=3` 且 `value_quant_mode=3` 时，key/value 使用 TQ4 packed KV cache；两个量化模式必须同时设置为 3。
>
> TQ4 的每个 KV token 使用一个 386 字节的槽位，布局如下：
>
> - `[0, 256)`：256 字节 packed INT4，每个字节包含两个 4-bit code，对应 512 维 latent/nope 数据；
> - `[256, 384)`：128 字节 BF16 RoPE 数据，对应 64 个 RoPE 元素；
> - `[384, 386)`：2 字节 FP16 token scale。该 scale 是每个 KV token 独立的反量化系数，与属性 `scale_value` 含义不同。
>
> TQ4 使用 16 个 codebook centroid 进行反量化，算子内部通过 byte-LUT 将 packed byte 解码为两个 centroid，再结合 token scale 参与 key/value 计算。TQ4 不需要额外提供 `key_dequant_scale` 和 `value_dequant_scale` 输入，量化参数随 KV cache 槽位存放。
>
> TQ4 当前面向 Ascend 910B 的 MLA-absorb 场景，query 支持 FLOAT16/BFLOAT16，`attention_mode=2`，`rope_head_dim=64`。现有单算子参数集中的 TQ4 用例采用 `layout_kv=PA_BSND`。

Constraints:

> - TQ4 INT4 模式仅在 `key_quant_mode=value_quant_mode=3` 时启用，key/value 的 KV_D 必须为 386；模式 3 不支持与模式 2 混用。
> - TQ4 的 query 头维度为 576（nope 512 + rope 64），`attention_mode=2`，`rope_head_dim=64`，KV_N 仅支持 1。

## Relevance

This is the most actionable item found for our 6.973 ms vs 1.382 ms SparseFlashAttention gap.
Three things. First, it is supported on A3 and it folds the C8 dequant into the attention itself
(`Dequant(K_int8, Scale_K)` inside the softmax argument), which is work we currently pay for in
the 2.300 ms Cast against 0.076 ms of bytes. Second, `KV_D值仅支持656，即
nope+rope*2+dequant_scale*4=512+64*2+4*4` means the per-token dequant scale is stored
*inside* the same cache slot as nope and rope — so each selected token is a single contiguous
656-byte gather rather than a nope read plus a rope read plus a scale read. That is the concrete
mechanism behind `针对离散访存进行了指令缩减及搬运聚合的细致优化`. Third, the TQ4 mode packs a
token into 386 bytes (256 B packed INT4 + 128 B BF16 rope + 2 B FP16 scale) with a 16-centroid
byte-LUT dequant, cutting the gathered bytes per selected token by 41% versus the 656-byte C8
slot — but it is documented as 910B/MLA-absorb only, so A3 applicability needs checking.

---

### SparseFlashAttention kernel: A3 (`arch22`) is a 4-file MLA-specific cube/vector split

https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention/op_kernel

```
attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_common.h
attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h
attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_cube_mla.h
attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h
attention/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp
attention/sparse_flash_attention/op_kernel/sparse_flash_attention_template_tiling_key.h
```

The template parameter space, which is the complete list of specializations the op admits:

https://gitcode.com/cann/ops-transformer/blob/master/attention/sparse_flash_attention/op_kernel/sparse_flash_attention_template_tiling_key.h

```cpp
#define SFA_LAYOUT_BSND 0
#define SFA_LAYOUT_TND 1
#define SFA_LAYOUT_PA_BSND 2

#define ASCENDC_TPL_4_BW 4

#define C_TEMPLATE 0
#define V_TEMPLATE 1

// 模板参数支持的范围定义
ASCENDC_TPL_ARGS_DECL(SparseFlashAttention, // 算子OpType
                      ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1), ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 0, 1),
                      ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND,
                                            SFA_LAYOUT_TND),
                      ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND,
                                            SFA_LAYOUT_TND, SFA_LAYOUT_PA_BSND),
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, C_TEMPLATE,
                                            V_TEMPLATE),
                      ASCENDC_TPL_BOOL_DECL(IS_SPLIT_G, 0, 1), ASCENDC_TPL_BOOL_DECL(HAS_ROPE, 0, 1), );
```

## Relevance

`FLASH_DECODE` is a template boolean, so the split-KV decode path is a compile-time
specialization rather than a runtime branch — the same lever `SPLIT_M_MODE` gives us in
MlaPrologV3. `IS_SPLIT_G` (split over the GQA group) is the knob that matters at TP16, where we
have 4 live Q heads per die. The `C_TEMPLATE` / `V_TEMPLATE` split plus the separate
`service_cube_mla.h` / `service_vector_mla.h` files confirm the op is internally organized as a
cube service and a vector service over a shared kernel driver — the same `ASCEND_IS_AIC` /
`ASCEND_IS_AIV` role structure as MlaProlog, so a megakernel can host both services directly.

---

## 7. `FusedInferAttentionScore` and `IncreFlashAttention` — the general-purpose decode attention ops

### aclnnFusedInferAttentionScoreV5 — one op for both prefill and decode

https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md

> ## 功能说明
>
> - 接口功能：适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。
>
>   相比于FusedInferAttentionScoreV4，本接口新增qStartIdxOptional、kvStartIdxOptional、pseType参数。
>
>   **说明：**
>
>   decode场景下特有KV Cache：KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。

> 本算子中Score函数采用Softmax函数，self-attention计算公式为：
>
> $$
> Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
> $$

> query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax((QK^T) / √d)的计算结果。

## Relevance

`FusedInferAttentionScore` is the unified prefill/decode attention op — it dispatches to the
`PromptFlashAttention` path or the `IncreFlashAttention` path depending on `Q_S`. It is the
general-purpose op and not the one on our critical path (we are on `SparseFlashAttention`
because of the DSA index stage), but it is the reference for what the non-sparse decode
attention contract looks like, and `IncreFlashAttention` is its decode half.

---

## Gaps

What was looked for and not found.

- **No published per-call number for MlaPrologV3/V4 at batch=1 on A3.** The design docs give the
  tiling strategy and the full TilingKey layout but no microsecond figures. The only MLA-adjacent
  per-call A3 TP16 number found anywhere was for W4A8 GroupedMatmul (65.9 us mean, PR #16606),
  not for the prolog.
- **`SparseFlashAttention` `arch22` kernel bodies not read.** The four A3 files were located and
  the template parameter space was read, but `sparse_flash_attention_service_vector_mla.h` and
  `_service_cube_mla.h` were not opened, so *how* the sparse gather is pipelined against the
  cube work on A3 is still unverified. That is the remaining load-bearing unknown behind the
  6.973 vs 1.382 ms gap.
- **No `docs/` entry for `kv_quant_sparse_flash_attention`.** The README (quoted above) gives the
  formula, the 656-byte C8 slot layout and the TQ4 386-byte slot, but there is no
  `aclnnKvQuantSparseFlashAttention.md`, so the exact aclnn signature and parameter order were
  not found. Whether TQ4 is usable on A3 (the README says `TQ4 当前面向 Ascend 910B 的
  MLA-absorb 场景` while the product table marks A3 supported for the op as a whole) is
  unresolved.
- **`MlaPrologV3` kernel source not read**, so the V3 `CrossCoreSetFlag` sequence (as opposed to
  the V1 pseudo-code quoted above) is unverified, including what actually changes under
  `SPLIT_M_MODE=0` vs `1` and under `CV_MODE` 1:1 vs 1:2.
- **No community work found that fuses MLA prolog + attention + o_proj into one kernel on
  Ascend.** The closest is the vllm-ascend GLM-5.2 MegaKernel RFC (#15639) — a plan, scoped to
  A5 and TP=1, with no accuracy or performance numbers attached yet. On the GPU side DeepSeek's
  `fused_norm_rope_attn_rope_cast` deliberately stops at the attention epilogue and leaves the
  projection GEMM outside.
- **No measurement of whether the lightning indexer and SFA can share a kernel.** Both ops exist
  in `cann/ops-transformer`; no published fusion of the two, and no `nsa_*`/`lightning_indexer`
  fused variant was found.
- **No Ascend-published cost for `CrossCoreSetFlag` / `CrossCoreWaitFlag` / `SyncAll`.** The GPU
  analogue is documented (~60 ns per CUDA async barrier even in the pass state, per Hazy
  Research); the Ascend equivalent is the number needed to budget a megakernel's barrier count,
  and it was not found in any source.
- **`blockrt` / `Glm52MegaKernel` could not be located.** Named in the vllm-ascend RFC as
  Huawei's MegaKernel runtime (MK-Tile granularity, AI CPU coordinator + scoreboard +
  callbacks), but no public repository or documentation for it was found.
- **Cohere's megakernel blog (https://cohere.com/blog/megakernels, Sep 2026) could not be
  fetched** from this network (proxy interception), so nothing from it is quoted here.
- **No `aclnnSparseFlashMla` vs `aclnnSparseFlashAttention` comparison found.**
  `attention/sparse_flash_mla` and `attention/quant_sparse_flash_mla` exist with docs, but
  nothing states when the runtime picks one over the other, or whether `sparse_flash_mla` is
  A3-capable (`flash_mla` itself is 950-only).

---

## 8. `SparseFlashAttention` on A3: the host tiling never splits KV or G, and at batch=1 decode it runs on ONE cube core

Clone, not raw-URL fetch (`raw.gitcode.com` answers `429 Limited account:unregisteredUser`):

```
git clone https://gitcode.com/cann/ops-transformer.git
commit 3efc65dad2a97a4fea856c3be319da596a42fb8a
author L_Euler
date   Tue Oct 20 16:03:55 2026 +0800
subject 【BSA】aclnnBlockSparseAttentionV2 文档函数原型 qDequantScale/kDequantScale/vDequantScale 与头文件参数名不一致
```

Read 2026-10-20. All paths below are relative to the repo root; the browsable form is
`https://gitcode.com/cann/ops-transformer/blob/master/<path>`.

### 8.1 `FLASH_DECODE` is a declared template parameter that is NEVER instantiated

`attention/sparse_flash_attention/op_kernel/sparse_flash_attention_template_tiling_key.h` declares
the parameter with both values:

```cpp
ASCENDC_TPL_ARGS_DECL(SparseFlashAttention, // 算子OpType
                      ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1), ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 0, 1),
```

but every one of the eight entries in the `ASCENDC_TPL_SEL` combination list — the list the comment
says is what `GET_TPL_TILING_KEY` validates against — pins it to `0`:

```cpp
// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0), ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 0),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND),
                         ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND, SFA_LAYOUT_PA_BSND),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, C_TEMPLATE),
                         ASCENDC_TPL_BOOL_SEL(IS_SPLIT_G, 0, 1), ASCENDC_TPL_BOOL_SEL(HAS_ROPE, 0, 1), ),
    ...
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0), ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_TND, SFA_LAYOUT_PA_BSND),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, V_TEMPLATE),
                         ASCENDC_TPL_BOOL_SEL(IS_SPLIT_G, 0, 1), ASCENDC_TPL_BOOL_SEL(HAS_ROPE, 0, 1), ), );
```

(Eight `ASCENDC_TPL_ARGS_SEL` entries = {`PAGE_ATTENTION` 0,1} x {`LAYOUT_T` BSND,TND} x
{`TEMPLATE_MODE` C,V}. `FLASH_DECODE` is `0` in all eight; `1` appears nowhere.)

The host side closes the door with a literal. `op_host/sparse_flash_attention_tiling.cpp`,
`SFAMlaTiling::GenTilingKey()`:

```cpp
void SFAMlaTiling::GenTilingKey()
{
    uint32_t inputQType = static_cast<uint32_t>(sfaInfo_->inputQType);
    uint32_t inputKvType = static_cast<uint32_t>(sfaInfo_->inputKvType);
    uint32_t outputType = static_cast<uint32_t>(sfaInfo_->outputType);
    uint32_t layoutQuery = static_cast<uint32_t>(sfaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaInfo_->kvLayout);
    uint32_t pageAttention = 0U;
    if (sfaInfo_->kvLayout == SFALayout::PA_BSND) {
        pageAttention = 1U;
    }

    tilingKey_ = GET_TPL_TILING_KEY(
        0U, pageAttention, layoutQuery, layoutKV, perfMode_ == SFAPerfMode::V_TEMPLATE_MODE,
        static_cast<uint32_t>(sfaInfo_->gSize > 64),
        static_cast<uint32_t>(sfaInfo_->ropeHeadDim != 0)); // N1 > 64时核间切G; rope 用独立 tiling key 避免污染原路径

    OP_LOGI(sfaInfo_->opName, "SFA tilingKey_: %lu.", tilingKey_);
}
```

The first argument — the `FLASH_DECODE` slot — is the literal `0U`.

And the companion host flag is declared `false` and assigned nowhere in the entire operator
directory. `op_host/sparse_flash_attention_tiling.h:342`:

```cpp
    bool splitKVFlag_ = false;
```

`grep -rn 'splitKVFlag_ *=' attention/sparse_flash_attention/` returns exactly one line — that
declaration. Consequently in `FillTilingSplitKVMla()` the split count is always zeroed:

```cpp
// for flash decode
void SFAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);

    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2U * sfaInfo_->n2Size * mBaseSize_ *
                                               headDimAlign_); // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaInfo_->n2Size *
                                                mBaseSize_ * // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}
```

and the `CalcInnerSize` branch that would halve the S2 base block for decode is unreachable:

```cpp
void SFAMlaTiling::CalcInnerSize(uint32_t sfaS2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && sfaInfo_->qLayout != SFALayout::TND) {
        if (sfaS2Size == 256) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128;  // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
        } else if (sfaS2Size > 256 && sfaS2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2;                  // 2:减半
        }
    }

    sInnerLoopTimes_ = (sfaS2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = sfaS2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
```

So `sInnerSize_` is **always 512** on A3: a 512-token S2 base block, the prefill-shaped choice.

### 8.2 The host "split" is not a split: it assigns no work and claims every core

`op_host/sparse_flash_attention_tiling.cpp`:

```cpp
void SFAMlaTiling::SplitBalanced()
{
    CalcInnerSize(sfaInfo_->s2Size);

    InnerSplitParams sfaInnerSplitParams;
    sfaInnerSplitParams.s1GBaseSize = sfaInfo_->gSize;
    sfaInnerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(sfaInnerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(sfaInnerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void SFAMlaTiling::Split()
{
    SplitBalanced();
}
```

`mBaseSize` is set to the whole `gSize`, i.e. one base block covers all Q heads of the group.
`usedCoreNum_ = aicNum_` unconditionally. The real distribution is deferred to the kernel.

Core counts and the MIX ratio, from `GetPlatformInfo()` / `CalcBlockDim()`:

```cpp
    aivNum_ = sfaAscendcPlatform.GetCoreNumAiv();
    aicNum_ = sfaAscendcPlatform.GetCoreNumAic();
```

```cpp
void SFAMlaTiling::InitParams()
{
    if (sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) { // 4:当前支持范围
        perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
    } else {
        perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
    }

    coreNum_ = aicNum_;
```

```cpp
void SFAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(sfaInfo_->opName, "SFA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}
```

```cpp
ge::graphStatus SFAMlaTiling::SetTilingKey(uint64_t tilingKey) const
{
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1); // 1: batchmode模式
    return ge::GRAPH_SUCCESS;
}
```

The AIC core-count ceiling, `op_host/sparse_flash_attention_tiling.h`:

```cpp
const uint32_t SFA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐
```

and the fixed M base:

```cpp
    uint32_t mBaseSize_ = 128;
```

The kernel is declared MIX 1 AIC : 2 AIV, `op_kernel/sparse_flash_attention.cpp`:

```cpp
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

with the matching index arithmetic in `op_kernel/arch22/sparse_flash_attention_kernel_mla.h`:

```cpp
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }
```

The comments name the A3 shape exactly: **cube 0-23, vec 0-47**.

### 8.3 `InitCalcParamsEach` — the base block is (batch x kv-head x q-token), so batch=1 decode = 1 core

`op_kernel/arch22/sparse_flash_attention_kernel_mla.h`, verbatim:

```cpp
template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitCalcParamsEach()
{
    // 计算总的基本块
    uint32_t actBatchS2 = 1;
    uint32_t totalBaseNum = 0;
    uint32_t s1GBaseSize = sfaKernelConstInfo.gSize;
    uint32_t coreNum = GetBlockNum();
    uint32_t currCoreIdx = aiCoreIdx;
    uint32_t actBatchS1 = 1;
    for (uint32_t bIdx = 0; bIdx < sfaKernelConstInfo.batchSize; bIdx++) {
        uint32_t actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        if (actBatchS1 < sfaKernelConstInfo.qSeqSize) {
            sfaKernelConstInfo.needInit = true;
        }
        totalBaseNum += actBatchS1 * actBatchS2;
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    } else {
        usedCoreNum = totalBaseNum;
    }
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }
    // 计算当前核的基本块
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t lastValidBIdx = 0;
    uint32_t lastValidactBatchS1 = 0;
    bool setStart = false;
    targetBaseNum = (currCoreIdx + 1) * avgBaseNum; // 计算当前的目标权重
    uint32_t targetStartBaseNum = targetBaseNum - avgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < sfaKernelConstInfo.batchSize * sfaKernelConstInfo.kvHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / sfaKernelConstInfo.kvHeadNum;
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < actBatchS1; s1GIdx++) {
            accumBaseNum += 1;
            if (!setStart && accumBaseNum >= targetStartBaseNum) {
                sfaKernelConstInfo.bN2Start = bN2Idx;
                sfaKernelConstInfo.gS1Start = s1GIdx;
                setStart = true;
            }
            if (accumBaseNum >= targetBaseNum) {
                // 更新当前核的End分核信息
                sfaKernelConstInfo.bN2End = bN2Idx;
                sfaKernelConstInfo.gS1End = s1GIdx;
                sfaKernelConstInfo.s2End = 0;
                sfaKernelConstInfo.coreStartKVSplitPos = 0;
                if (aiCoreIdx != 0) {
                    GetAxisStartIdx(sfaKernelConstInfo.bN2Start, sfaKernelConstInfo.gS1Start, 0);
                }
                return;
            }
        }
```

Three literals decide everything here:

- `uint32_t actBatchS2 = 1;` — the S2 (KV) axis contributes a factor of exactly one to
  `totalBaseNum`. KV length is not a parallel axis.
- `totalBaseNum += actBatchS1 * actBatchS2;` over `bIdx < batchSize` — so
  `totalBaseNum = sum_b (q_tokens_in_batch_b)`.
- `else { usedCoreNum = totalBaseNum; }` followed by `if (aiCoreIdx >= usedCoreNum) { return; }`
  — when `totalBaseNum <= coreNum`, `usedCoreNum` is *lowered* to `totalBaseNum` and every core
  above that index returns before doing any work.

For MLA, `kvHeadNum == 1`; the per-core work unit is one `(bN2Idx, s1GIdx)` pair, and
`mBaseSize == gSize` means all Q heads of the group live in that single unit. So for
**batch=1, one query token, MLA**: `totalBaseNum = 1` → `usedCoreNum = 1` → cores `1..23` take
the early `return`, and the entire sparse attention for that layer executes on cube core 0 plus
its two paired vector cores.

`IS_SPLIT_G` does not rescue this. From `GenTilingKey()` above it is
`static_cast<uint32_t>(sfaInfo_->gSize > 64)`, with the comment `N1 > 64时核间切G`. Splitting
over heads only switches on above 64 heads *per rank*. At TP16 there is no configuration of any
published GLM/DeepSeek MLA model that leaves more than 64 Q heads on a rank.

### 8.4 The workspace confirms a GM staging round trip for the per-token gather

`op_host/sparse_flash_attention_tiling.cpp`, `GetWorkspaceSize()`, the non-A5 (A3) branch:

```cpp
        workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * sfaMmResElemSize);
        workspaceSize_ +=
            preLoadNum *
            static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * sfaVec1ResElemSize) * sfaKvDtypeRatio);
        workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * sfaBmm2ResElemSize;
        workspaceSize_ +=
            preLoadNum * static_cast<size_t>(static_cast<float>(qPreSizeMla_ * actCoreNum * sfaQPreProcResElemSize) *
                                             sfaKvDtypeRatio);
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * sfaNUpdateElemSize;
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * sfaSoftmaxSumElemSize;
        // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
        //              bufNum  s2Base   D   dRope  sizeOf(half)
        // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
        workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
        // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
        workspaceSize_ +=
            4 * 128 * 4 * (2 * actCoreNum); // 4:缓存有效mte2 size的长度 128:份数  4:512B对齐的长度  2:aiv核数
```

`topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值` — "the top-k block-size-1 case needs extra
space to cache the discretely-gathered values". 4 buffers x 512 S2 slots x 576 dims x 2 B per
core = 2.25 MB per core of HBM workspace, and the gathered KV is written there and read back.
That buffer is `kvMergeGm_` in the kernel. The second line caches "the length of the valid MTE2
size", 4 copies x 128 entries x 4 B, for `2 * actCoreNum` vector cores.

The matching `PRE_LOAD_NUM`:

```cpp
constexpr uint32_t PRE_LOAD_NUM = 2;
```

### 8.5 The fixed per-core budget: 512-row S2 workspace, 64x576 L1 blocks, 128x128 cube tiles

`op_kernel/arch22/sparse_flash_attention_service_cube_mla.h`:

```cpp
    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t N_WORKSPACE_SIZE = 512; // n方向切分

    static constexpr uint32_t L1_BLOCK_SIZE = (64 * (512 + 64) * sizeof(Q_T));
    static constexpr uint32_t L1_BLOCK_OFFSET = 64 * (512 + 64); // 72K的元素个数

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);
```

```cpp
    constexpr uint32_t MERGE_K_PITCH = 576;
```

`M_SPLIT_SIZE = 128` is the cube M tile. At batch=1 decode the real M is `gSize` — 4 to 8 rows at
TP16 — against a 128-row tile. `L1_BLOCK_SIZE = 64 * 576 * 2 = 72 KiB` per L1 block, `L0C_PP_SIZE
= 64 KiB`.

The cube/vector sync event budget, same file:

```cpp
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3}; // mte12复用
    static constexpr uint32_t sfaMte21KvEvents[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};
```

and the cross-core sync IDs that a megakernel would have to reproduce,
`op_kernel/arch22/sparse_flash_attention_common.h`:

```cpp
struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t SFA_SYNC_MODE2 = 2;
    ...
    // CUBE和VEC的核间同步EventID
    uint32_t syncV1NupdateC2 = 0U;
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;
    uint32_t syncC2V1 = 0U;
```

`splitKVNum` is in that struct and documented as the inter-core S2 split count — the field exists,
it is just always fed 0:

```cpp
    uint32_t splitKVNum = 0U;         // S2核间切分的切分份数
    ...
    // FlashDecoding
    uint32_t actualCombineLoopSize = 0U; // FlashDecoding场景, S2在核间切分的最大份数
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;
```

### 8.6 The template mode switch: `sparseBlockSize <= 4` picks `V_TEMPLATE`

```cpp
    if (sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) { // 4:当前支持范围
        perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
    } else {
        perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
    }
```

with the `sparse_block_size` validity constraints from the same file:

```cpp
    OP_CHECK_IF((npuArch_ == NpuArch::DAV_3510 && *opParamInfo_.sparseBlockSize != 1),
                ...
    OP_CHECK_IF((*opParamInfo_.sparseBlockSize <= 0 || *opParamInfo_.sparseBlockSize > 128 ||
                 (static_cast<uint64_t>(*opParamInfo_.sparseBlockSize) &
                  static_cast<uint64_t>(*opParamInfo_.sparseBlockSize - 1L)) != 0UL),
```

and

```cpp
    OP_CHECK_IF(blockSize_ % sparseBlockSize_ > 0,
                ... ") must be divided by sparse_block_size(" + std::to_string(sparseBlockSize_) +
```

So: `sparse_block_size` must be a power of two in `[1, 128]`; the PageAttention `block_size` must
be a multiple of it; on `DAV_3510` (A5/950) it must be exactly 1; and a DSA-style per-token
selection (`sparse_block_size = 1`) lands in `V_TEMPLATE_MODE`.

## Relevance

This is the answer to the question the earlier Gaps entry left open, and it is a *structural*
answer rather than a data-movement one.

`SparseFlashAttention` on A3 has **no decode specialization at all**. The three things that would
give a batch=1 single-query decode its parallelism are each disabled by a constant:

| lever | state on A3 | where |
| --- | --- | --- |
| split-KV across cores (`FLASH_DECODE`) | `GET_TPL_TILING_KEY(0U, ...)` literal; never in `ASCENDC_TPL_SEL`; `splitKVFlag_` assigned nowhere | `GenTilingKey()`, `template_tiling_key.h` |
| split over Q heads (`IS_SPLIT_G`) | only when `gSize > 64`, i.e. >64 heads per rank | `GenTilingKey()` |
| S2 base block size | `sInnerSize_ = 512` always, decode halving branch unreachable | `CalcInnerSize()` |

and the kernel's own distribution counts base blocks as `sum_b q_tokens_b` with
`actBatchS2 = 1`, then *lowers* `usedCoreNum` to that count. At batch=1, Q_S=1, MLA
(`kvHeadNum=1`) that count is **1**. So the op that costs us 6.973 ms — the largest family in the
round — is running on 1 of 24 cube cores while 23 sit in an early `return`.

This reframes our own decomposition. Of the four candidate causes in the brief, this is (c)
*insufficient parallelism across the AI cores at one query token*, and it is not a soft effect: it
is a hard 1/24. It subsumes (d) as well — the 512-row `s2BaseSize` and the 128-row `M_SPLIT_SIZE`
cube tile are prefill-shaped constants applied to a 1-token query, so the first S2 block is also
the tail block, and the cube runs a 128x128 tile to produce `gSize`(=4..8) useful rows.

Two concrete levers follow, both available to a hand-written engine without touching CANN:

1. **Supply the parallelism the op refuses to find.** Because the base block is
   `(batch, kv_head, q_token)` and `usedCoreNum` is just `totalBaseNum`, presenting the same
   decode work as `totalBaseNum >= 24` — splitting the selected KV list into 24 chunks ourselves
   and reducing the partial `(out, lse)` outside, which is exactly what the dead `FLASH_DECODE`
   path would have done with `accumOutSize` / `logSumExpSize` — recovers the cores. The op even
   exposes `softmaxMax` / `softmaxSum` outputs and a `returnSoftmaxLse` attribute to make the
   external combine possible. The alternative framing is to fake `q_tokens` by packing the
   `gSize` heads along S1 in TND layout, which turns `gSize` into `totalBaseNum`.
2. **Stop paying the MIX launch floor per layer.** `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`
   makes every `SparseFlashAttention` call a MIX_AIC launch — our measured 13.58 us floor, 6.5
   MIX_AIC calls per layer, 11.5 ms of launch floor per round. A one-core kernel paying a
   13.58 us MIX launch is the worst possible trade.

And one number to check against ours: 1/24 of the cores predicts a gap far larger than the 5.0x
we measure (6.973 ms spent vs 1.382 ms of bytes). Either the 1.382 ms roofline is computed
against a bandwidth a single core can nearly saturate, or our call is not shaped the way
`InitCalcParamsEach` sees batch=1 — e.g. we pass TND with the heads packed along T, which would
make `totalBaseNum = gSize` and use `gSize` cores rather than 1. Resolving which is the single
highest-value profiling question: read `usedCoreNum` out of the `OP_LOGI` line
`SFA block dim: %u aiv Num: %u aic Num: %u.` in plog, or count active cores in the Level-2
profile. If it really is 1, there is far more than 5.6 ms available here.

---

## 9. The `V_TEMPLATE` gather: a mandatory HBM round trip, 2 tokens per DMA, 32-row UB staging

Same clone and commit as section 8 (`gitcode.com/cann/ops-transformer`,
`3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20). Files:
`attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h`
and `..._service_cube_mla.h`.

### 9.1 In `V_TEMPLATE` the cube never touches the KV cache — it reads a merged GM buffer

`sparse_flash_attention_service_cube_mla.h`, `ComputeMm1NoRope`. The `V_TEMPLATE` branch is a
single contiguous `Nd2NzParams` copy out of `kvMergeGm_`; only the `else` (= `C_TEMPLATE`) branch
walks the top-k list:

```cpp
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                Nd2NzParams sfaCubeCopyParams;
                sfaCubeCopyParams.ndNum = 1;
                sfaCubeCopyParams.nValue = mm1NrNL1Size;
                sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                sfaCubeCopyParams.dstNzC0Stride = mm1NrNL1SizeAlign;
                sfaCubeCopyParams.dstNzNStride = 1;
                sfaCubeCopyParams.srcNdMatrixStride = 0;
                sfaCubeCopyParams.dstNzMatrixStride = 0;
                DataCopy(mm1NrBL1Tensor,
                         kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * MERGE_K_PITCH +
                                    mm1NrKL1 * (sfaCubeConstInfo.headDim >> 1) +
                                    mm1NrNL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                         sfaCubeCopyParams);
            } else {
                while (copyFinishRowCnt < mm1NrNL1Size) {
                    CalcTopKBlockInfo(sfaCubeRunInfo, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt,
                                      idInTopK);
```

`kvMergeGm_` is the HBM workspace that section 8.4 budgets as
`4 * 512 * (512 + 64) * 2 * actCoreNum` with the comment
`topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值`. So in the DSA case
(`sparse_block_size <= 4` → `V_TEMPLATE`) the data path is:

```
KV cache in HBM  --(scattered MTE2, AIV)-->  UB (kvMergUb_)
UB               --(MTE3, AIV)-----------> kvMergeGm_ in HBM     <-- extra write
kvMergeGm_       --(contiguous MTE2, AIC)-> L1                   <-- extra read
```

Every selected KV token is read from HBM once, written back to HBM once, and read from HBM again.
Three HBM transits where a roofline model counts one.

### 9.2 The gather loop: one `topkGm_.GetValue()` scalar GM read per selected token

`sparse_flash_attention_service_vector_mla.h`:

```cpp
template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t& realS2Idx,
                                                            int64_t topkGmBaseOffset, const RunInfo& sfaVecRunInfo)
{
    int64_t topkGmIdx =
        (s2GmOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize) / sfaVecConstInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= sfaVecConstInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(sfaVecConstInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize) %
                                     sfaVecConstInfo.sparseBlockSize);
}
```

and in PageAttention mode a *second* scalar GM read per token, for the block table:

```cpp
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / sfaVecConstInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % sfaVecConstInfo.kvCacheBlockSize;
        realKeyGmOffset = blkTableGm_.GetValue(sfaVecRunInfo.bIdx * sfaVecConstInfo.maxBlockNumPerBatch + blkTableIdx) *
                              static_cast<int64_t>(sfaVecConstInfo.kvCacheBlockSize) *
                              static_cast<int64_t>(sfaVecConstInfo.kvHeadNum) +
                          blkTableOffset;
    } else {
```

At `sparse_block_size = 1` this is `2 * top_k` scalar GM loads per (head-group, layer), each on the
AIV scalar pipe, each in the dependency chain ahead of the `DataCopyPad` that uses its result.

### 9.3 `MergeKv`: the main loop advances TWO tokens at a time and flushes every 32 rows

```cpp
// b s1 k
template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::MergeKv(const RunInfo& sfaVecRunInfo)
{
    int64_t s2ProcessSize = sfaVecRunInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * sfaVecConstInfo.sparseBlockSize);
```

```cpp
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize;
    int64_t s2GmLimit =
        GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * sfaVecConstInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, sfaVecRunInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + sfaVecConstInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, sfaVecRunInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, sfaVecRunInfo);
        if ((mte2Size - mte3Size + 2 * sfaVecConstInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * sfaVecConstInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }
```

The loop stride is `2 * sparseBlockSize` — **two selected tokens per iteration**. The UB staging
buffer holds 32 rows (`mte2Size - mte3Size + 2 * sparseBlockSize > 32` triggers the flush), and
the UB offsets confirm the size: `kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + ...]` is a double-buffered
32x512 tile, `ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + ...]` a 32x64 tile. So at
`sparse_block_size = 1` the pipeline is: 16 loop iterations (32 tokens) of MTE2, then one MTE3
flush, with a two-deep `MTE3_MTE2` ping-pong and an explicit `SetFlag`/`WaitFlag` pair in between.

The GM merge buffer rotates over 4 slots:

```cpp
    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
```

### 9.4 `CopyInKv`: the whole optimization is folding exactly 2 scattered tokens into one `DataCopyPad`

```cpp
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
                 (!PAGE_ATTENTION && (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0)) ||
                 realS2Idx1 + sfaVecConstInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + sfaVecConstInfo.sparseBlockSize >= s2IdLimit)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, sfaVecRunInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, sfaVecRunInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = sfaVecConstInfo.sparseBlockSize * sfaVecConstInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * sfaVecConstInfo.headDim],
                    keyGm_[startGmOffset * sfaVecConstInfo.headDim], intriParams, padParams);

        if constexpr (SFAT::hasRope) {
            intriParams.blockLen = sfaVecConstInfo.sparseBlockSize * sfaVecConstInfo.headDimRope * sizeof(KV_T);
            intriParams.dstStride = 0;
            intriParams.srcStride = keyRopeSrcStride;
            DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * sfaVecConstInfo.headDimRope],
                        keyRopeGm_[startGmOffset * sfaVecConstInfo.headDimRope], intriParams, padParams);
        }
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * sfaVecConstInfo.sparseBlockSize;
    }
```

This is the entirety of the `搬运聚合` ("transfer aggregation") that the
`kv_quant_sparse_flash_attention` README advertises: `blockCount = 2`, `srcStride` = the gap
between the two chosen tokens. Two tokens per MTE2 instruction and no more, because `DataCopyPad`
takes a single uniform `srcStride` and three arbitrary top-k indices do not share one.

`blockLen` per instruction at MLA/DSA shape: `1 * 512 * 2 = 1024` bytes for nope, and a separate
instruction of `1 * 64 * 2 = 128` bytes for rope. With `blockCount = 2`: 2 KiB and 256 B per pair.
The rope copy is a **128-byte** burst.

Note also the `还原成2条搬运指令` ("fall back to 2 transfer instructions") path: whenever the gap
between the two chosen tokens is negative in the index order, or either token is near the sequence
end, the pairing is abandoned and it becomes one instruction per token. Since the top-k index list
is not guaranteed sorted ascending, `keySrcStride < 0` is a live condition.

### 9.5 `CopyOutMrgeResult` — the write-back, 512 B + 128 B per row

```cpp
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = sfaVecConstInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % 4 * 512 * 576 + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * 32 * 512], dataCopyParams);

    if constexpr (SFAT::hasRope) {
        dataCopyParams.blockLen = sfaVecConstInfo.headDimRope * sizeof(KV_T);
        DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % 4 * 512 * 576 + 512 * 512 +
                               (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                    ropeMergUb_[mergeMte3Idx % 2 * 32 * 64], dataCopyParams);
    }
```

with a hard MTE2→MTE3 barrier immediately before it:

```cpp
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
```

`SetFlag` immediately followed by `WaitFlag` on the same event is a full serialization point, not
a pipelined handoff: the MTE3 flush cannot overlap the MTE2 fill of the same buffer.

### 9.6 The zero-fill tail: one `DataCopyPad` per *missing* row

When fewer than `s2ProcessSize` tokens were found (the usual case at decode when top-k exceeds
the real sequence length, or when the index list terminates with `-1`), the remainder of the 512-row
merge buffer is zeroed one row at a time:

```cpp
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                   s2GmOffset * sfaVecConstInfo.headDim],
                        kvMergUb_, dataCopyParams);
        }
        if constexpr (SFAT::hasRope) {
            dataCopyParams.blockLen = sfaVecConstInfo.headDimRope * sizeof(KV_T);
            for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
                DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                       512 * sfaVecConstInfo.headDim + s2GmOffset * sfaVecConstInfo.headDimRope],
                            kvMergUb_, dataCopyParams);
            }
        }
```

`dataCopyParams.blockCount = 1` (set just above), so these are **single-row, 1 KiB (then 128 B)
MTE3 instructions, one per padded row** — and the cube still multiplies against those zero rows
because `mm1NrNL1Size` is derived from `actualSingleProcessSInnerSize`, not from `mte2Size`.

### 9.7 The cube serializes its own M pipe whenever the Mmad tile is small

`sparse_flash_attention_service_cube_mla.h`, inside the Mmad loop:

```cpp
                    MmadParams mm1NrMmadParams;
                    mm1NrMmadParams.m = mm1NrML1SizeAlign;
                    mm1NrMmadParams.n = mm1NrNL1SizeAlign;
                    mm1NrMmadParams.k = mm1NrKL0Size;
                    mm1NrMmadParams.cmatrixInitVal = (mm1NrKL1 == 0 && mm1NrKL0 == 0);
                    mm1NrMmadParams.cmatrixSource = false;
                    mm1NrMmadParams.unitFlag = (mm1NrKL1 == 1 && mm1NrKL0 == (mm1NrKL0Loops - 1)) ? 0b11 : 0b10;
                    Mmad(mm1NrCL0Tensor, mm1NrAL0Tensor, mm1NrBL0Tensor, mm1NrMmadParams);
                    if ((mm1NrMmadParams.m / 16) * (mm1NrMmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
```

with the K split fixed at:

```cpp
    constexpr uint32_t mm1NrKL1Loops = 2;
    constexpr uint32_t mm1NrKL0Size = 128;
    constexpr uint32_t mm1NrKL0Loops = 2;
```

At batch=1 decode the M of the first matmul is `gSize` rounded up to 16. With `N_SPLIT_SIZE = 128`:
`(16/16) * (128/16) = 8 < 10`, so `PipeBarrier<PIPE_M>()` fires on **every single `Mmad`** —
4 Mmads per (M-tile, N-tile) from `mm1NrKL1Loops * mm1NrKL0Loops`. The guard exists to work
around a small-tile hazard, and decode shape sits permanently on the wrong side of it. The cube's
M pipe is fully drained between consecutive 16x128x128 Mmads.

## Relevance

This answers (b) and (d) of the four candidates, quantitatively, and it changes which number we
should be comparing against.

**The roofline denominator is wrong by ~3x for the DSA path.** Our 1.382 ms is the time to move
the bytes the algorithm needs. Section 9.1 shows `V_TEMPLATE` moves each selected KV token across
HBM three times — scattered read into UB, MTE3 write to `kvMergeGm_`, contiguous read back into
L1. So the *op as written* has a floor near 3 x 1.382 = 4.15 ms, not 1.382 ms. Our measured
6.973 ms is 1.68x that floor, not 5.0x. That reframes the opportunity: the 5.6 ms we hoped to
recover by "closing to roofline" is not available inside this operator at all. About 2.8 ms of it
is the staging round trip, and only a hand-written kernel that keeps the gathered KV in UB/L1 and
feeds the cube directly can claim it.

**The per-instruction burst lengths are small and fixed.** Nope: 1024 B x `blockCount=2`. Rope:
128 B x `blockCount=2`. The merge write-back: 1024 B and 128 B per row. The zero-pad tail: single
1024 B and 128 B instructions, one per row, no batching. A 128-byte MTE2/MTE3 burst is well under
any plausible efficient DMA length, and the rope stream alone is `top_k` instructions of 128 B
per head-group per layer.

**The aggregation ceiling is 2.** `CopyInKv` pairs exactly two tokens per `DataCopyPad` because
`DataCopyExtParams` carries one `srcStride`. Three or more arbitrary top-k indices cannot share
an instruction. So at `sparse_block_size = 1` the instruction count is `top_k / 2` MTE2 pairs
plus the same again for rope — and halves to `top_k` whenever the `还原成2条搬运指令` fallback
trips, which it does whenever the index list is not ascending. If our indexer emits unsorted
top-k, sorting it on the host or in the indexer kernel is a free 2x on MTE2 instruction count.

**Two hard serializations per 32 tokens plus one per Mmad.** The `SetFlag`/`WaitFlag` pair on
`MTE2_MTE3` in `CopyOutMrgeResult` (9.5) prevents overlap of gather and flush within a buffer;
the `PipeBarrier<PIPE_M>()` in 9.7 fires on every Mmad at decode M. Neither is visible as bytes.
Both show up as `aic_mte2_time` / `aiv_mte2_time` far below the kernel wall time, which is the
signature to look for in our own Level-2 profile: if `mte2_ratio` is low *and* `aic_mac_ratio` is
low, the time is in barriers and scalar address generation, not in either pipe.

**Three levers, in descending value:**
1. Eliminate the `kvMergeGm_` round trip. This is the ~2.8 ms item and it is only reachable by
   owning the kernel: gather scattered KV straight into L1 (the `C_TEMPLATE` branch already does
   this, via `CopyInMm1BToL1` / `DataCopyPA`) instead of through GM. The reason upstream went
   through GM at `sparse_block_size <= 4` is that the cube cannot issue `top_k` separate L1 loads
   without stalling its own MTE2 — but with the KV gather on the AIV and a UB→L1 path, or with
   more than 2 tokens per instruction, that constraint is ours to re-examine.
2. Sort the top-k index list ascending. Keeps `CopyInKv` on the paired path instead of the
   `还原成2条搬运指令` fallback, halving MTE2 instruction count.
3. Do not pad to 512. `s2BaseSize` is fixed at 512 (section 8.1) and the shortfall is zero-filled
   one row at a time, then multiplied. At seq 8K with a top-k below 512 per S2 block, the padding
   rows are pure waste in MTE3 instructions, in HBM traffic, and in cube Mmad.

---

## 10. `IncreFlashAttention` vs `SparseFlashAttention`: same vendor, same A3, 16 cores vs 1

Same clone and commit (`gitcode.com/cann/ops-transformer`,
`3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20).
`attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp` and
`..._tiling.cpp`.

This is the A/B the brief asked for, and it is in source rather than in a benchmark: the dense
decode attention op on the same chip implements the split-KV flash-decode that
`SparseFlashAttention` leaves stubbed out (section 8.1).

### 10.1 `IFATiling::IsFlashDecode` — the live heuristic

```cpp
bool IFATiling::IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode)
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode = IfaPerfMode::BMM_ALL_BY_VEC;
        if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
            OP_LOGD(ifaContext_->opName, "flash decode has not been implemented when 310P and GQA > 1.");
            return false;
        }
    }
    if (balanceModeFlag_) {
        uint32_t tndFDCoreArrLen = tilingDataMla_.tndSplitCoreParams.get_tndFDCoreArrLen();
        return tndFDCoreArrLen > 0U;
    }
    if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B &&
        perfMode != IfaPerfMode::CUBE_VIEW_MM_MLA && maxActualseq_ <= 1024U) { // 1024, 经验值
        return false;
    }
    constexpr uint32_t maxQNum = 128U;
    if (ropeFlag_ && (qSeqSize_ > 1U || nNumOfQInOneGroup_ > maxQNum)) {
        return false; // 投机推理 和 G > 64 切G待实现
    }

    if (gqaMtpFlag_) {
        return false; // 投机推理FD待实现
    }

    if (CheckCoreOkFlag(coreNum, perfMode)) {
        return true;
    }

    return false;
}
```

```cpp
bool IFATiling::CheckCoreOkFlag(uint32_t coreNum, IfaPerfMode perfMode) const
{
    bool coreOkFlag = false;
    float flashDecodeBNRatio = static_cast<float>(0.4); // 0.4, 经验值
    if (perfMode == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD || perfMode == IfaPerfMode::CUBE_VIEW_MM ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode == IfaPerfMode::CUBE_VIEW_MM_DD) {
        flashDecodeBNRatio = 0.5; // 0.5, 经验值
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) <= flashDecodeBNRatio * coreNum);
    } else if (perfMode == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5; // 0.5, 全V模板可以按0.5切分
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    } else {
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    }

    if (coreOkFlag && (nNumOfQInOneGroup_ == 1U)) {
        OP_LOGD(ifaContext_->opName, "Flash Decode Split kv."); // 非gqa时这里只判断bn是否满足
        return true;
    }

    if (coreOkFlag && (maxActualseq_ >= 2048U)) { // 2048, 在flash decode + gqa时的经验值
        OP_LOGD(ifaContext_->opName, "Flash Decode And GQA Split kv.");
        return true;
    }
```

The trigger condition is stated as a ratio against the core count: flash-decode turns on when
`batch * kv_heads <= 0.5 * coreNum`. The *fewer* the batch-and-head work units, the *more* the op
splits KV. That is the exact inverse of what `SparseFlashAttention`'s `InitCalcParamsEach` does
(section 8.3), which lowers `usedCoreNum` to `batch * q_tokens` and leaves the rest idle.

Also note `IfaPerfMode::CUBE_VIEW_MM_MLA` is in the 0.5-ratio list and that the only MLA-specific
exclusion is speculative decoding (`qSeqSize_ > 1U`) or `G > 128`:

```cpp
enum class IfaPerfMode : uint32_t {
    NORMAL = 0,
    BMM_ALL_BY_VEC,         // 1
    C1_V1,                  // 2
    CUBE_VIEW_MM,           // 3
    CUBE_VIEW_MM_FULL_LOAD, // 4
    CUBE_VIEW_MM_MLA,       // 5
    CUBE_VIEW_MM_DD         // 6
};
```

### 10.2 `IFATiling::SplitBNS` — the split count, and the 512-token-per-core empirical floor

```cpp
ge::graphStatus IFATiling::SplitBNS()
{
    formerCoreNum_ = 0U;
    blockSplitBn2Range_ = 1U;
    tailSplitedBatchRange_ = 1U;

    uint32_t bn = batchSize_ * numKvHeads_;
    kvSplitPart_ = coreNum_ / bn;
    while (((maxActualseq_ / kvSplitPart_) < 512U) && (kvSplitPart_ > 1U)) { // 512, 经验值
        kvSplitPart_--;
    }

    usedCoreNum_ = bn * kvSplitPart_;
    uint32_t computeSeqSize = (seqSize_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        computeSeqSize = Align(computeSeqSize, blockSize_);
    }
    if (inputKvType_ == ge::DT_INT4) {
        computeSeqSize = Align(computeSeqSize, 2U);
    }
    CalcInnerSize(computeSeqSize);
    return ge::GRAPH_SUCCESS;
}
```

and the flag being set, from `incre_flash_attention_tiling.cpp` (three separate call sites plus
one unconditional one):

```cpp
            OP_LOGI(ifaContext_->opName, "aicNum_(use for usedCoreNum_ and kvSplitPart_):%u\n", aicNum_);
            ...
            splitKVFlag_ = true;
            kvSplit_++;
            kvSplitPart_ = aicNum_;
```

```cpp
    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = tilingInfo.maxKvSplitPart;
```

**Worked for our shape.** batch=1, MLA so `numKvHeads_ = 1`, `coreNum_ = 24` AICs, seq 8192:

- `bn = 1 * 1 = 1`
- `kvSplitPart_ = 24 / 1 = 24`
- loop: `8192 / 24 = 341 < 512` → decrement. `8192/23 = 356`, ... `8192/17 = 481 < 512`,
  `8192/16 = 512` → stop. **`kvSplitPart_ = 16`**
- `usedCoreNum_ = 1 * 16 = 16`
- `computeSeqSize = ceil(8192/16) = 512` KV tokens per core

And the gate passes: `CheckCoreOkFlag` with `CUBE_VIEW_MM_MLA` gives `1 <= 0.5 * 24 = 12` → true;
`nNumOfQInOneGroup_ > 1` for MLA so it falls through to `maxActualseq_ >= 2048` → `8192 >= 2048`
→ true. `IsFlashDecode` is not blocked by `ropeFlag_` because `qSeqSize_ == 1` and
`nNumOfQInOneGroup_ <= 128`.

So on the *identical* decode shape: `IncreFlashAttention` uses **16 of 24 cube cores**;
`SparseFlashAttention` uses **1**.

The `512U // 512, 经验值` constant is itself a published number worth having: Huawei's own
empirical floor for the minimum KV chunk worth giving a core in a decode attention split. Below
512 tokens per core they stop splitting. Note this is the same 512 that
`SparseFlashAttention`'s `sInnerSize_ = 512` uses as its *S2 base block*, i.e. the two ops agree
on 512 as the unit — one of them distributes those units across cores, the other does not.

### 10.3 The combine kernel SFA lacks exists for IFA on A3

`attention/incre_flash_attention/op_kernel/arch22/` contains, at 845 lines:

```
service_vector_flashdecode_mla.h
```

with the reduction interface:

```cpp
class ServiceFlashDecode {
...
    __aicore__ inline void InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm,
                                            GlobalTensor<T> accumOutGm, GlobalTensor<OUT_T> attentionOutGm,
    __aicore__ inline void InitParams(const AttentionCommonFlashDecode::ConstInfoFD& constInfo);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void FlashDecode(FDparams& fd);
...
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, int64_t baseOffset, uint32_t cntM);
```

and the per-row rescale primitives it is built from:

```cpp
__aicore__ inline void MatDivsVecFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
__aicore__ inline void RowSubFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
__aicore__ inline void ColMaxFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
__aicore__ inline void ColAddFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
```

`SparseFlashAttention`'s own kernel already allocates exactly the buffers this consumes —
`accumOutSize`, `logSumExpSize`, `lseSumFdGm`, `lseMaxFdGm` (section 8.1 and
`sparse_flash_attention_kernel_mla.h:503-508`) — behind `if constexpr (FLASH_DECODE)`, which is
never true.

The A3 IFA kernel file inventory, for scale:

```
   1264 ifa_service_matmul_full_quant.h
   4619 incre_flash_attention_preload.h
   4119 incre_flash_attention_preload_dd.h
   4324 incre_flash_attention_preload_mla.h
   4286 incre_flash_attention_split_Bbn2s2_Us2.h
   1987 paged_attention_antiquantkv.h
    845 service_vector_flashdecode_mla.h
  21444 total
```

`incre_flash_attention_split_Bbn2s2_Us2.h` — the name is the split axes: B, bn2, s2, with `Us2`
the KV-split variant. `SparseFlashAttention`'s entire `arch22` kernel is 4140 lines across four
files and has no `s2` split template at all.

### 10.4 A5 (`arch35`) has the same hole — this is not an A3-only gap

`attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla_arch35.h`:

```cpp
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitCalcParamsEach()
{
    // 计算总的基本块
    maxS2LoopCnt = 0; // 所有核中最大累计s2Loop
    uint32_t sfaTotalBaseNum = 0;
    uint32_t s1GBaseSize = sfaConstInfo.gSize;
    uint32_t actBatchS2 = 1;
    uint32_t coreNum = GetBlockNum(); // G128时相邻两个cube核处理一个s1，coreNum减半
    uint32_t sfaCurrCoreIdx = aicIdx;
    if constexpr (IS_SPLIT_G) {
        coreNum = coreNum >> 1;
        sfaCurrCoreIdx = sfaCurrCoreIdx >> 1;
    }
    uint32_t actBatchS1 = 1;
    for (uint32_t bIdx = 0; bIdx < sfaConstInfo.bSize; bIdx++) {
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx); // 不切S2，只关注S1
        if (actBatchS1 < sfaConstInfo.s1Size) {
            sfaConstInfo.needInit = true;
        }
        sfaTotalBaseNum += actBatchS1 * actBatchS2;
    }
    uint32_t sfaAvgBaseNum = 1;
    if (sfaTotalBaseNum > coreNum) {
        sfaAvgBaseNum = (sfaTotalBaseNum + coreNum - 1) / coreNum;
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = (sfaTotalBaseNum + sfaAvgBaseNum - 1) / sfaAvgBaseNum << 1;
        }
    } else {
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = sfaTotalBaseNum << 1;
        } else {
            usedCoreNum = sfaTotalBaseNum;
        }
    }
    ...
    if (aicIdx >= usedCoreNum) {
        return;
    }
```

The comment is explicit: `// 不切S2，只关注S1` — "do not split S2, only consider S1". Same
`actBatchS2 = 1`, same `usedCoreNum = sfaTotalBaseNum`, same early `return`. On A5 the only
mitigation is `IS_SPLIT_G`, which doubles the core count — and `GenTilingKey` only sets it when
`gSize > 64`.

One thing A5 does gain is a real top-k length input, so the zero-padding of section 9.6 can be
avoided:

```cpp
    GlobalTensor<int32_t> oriTopkLengthGm;
    bool hasOriTopkLength = false;
```

(present only in `arch35`; `grep -rn 'oriTopkLength' op_kernel/ op_host/` returns these two lines
and nothing in `arch22`.)

### 10.5 The published `SparseFlashAttention` contract, for reference

`attention/sparse_flash_attention/README.md`:

> - API功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

> 其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。
> 本次公布的`sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

The `sparse_indices` contract — note what it does and does *not* require:

> | sparse_indices | 输入 | 离散取kvCache的索引，不支持空tensor和非连续。sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分、无效值均在后半部分，且sparse_size大于0。layout_query为"BSND"时，shape为[B, Q_S, KV_N, sparse_size]。layout_query为"TND"时，shape为[Q_T, KV_N, sparse_size]。 | INT32 | ND |

Valid entries must be contiguous at the front of each row; **ascending order is not required**.

> | sparse_block_size | 可选属性 | 代表sparse阶段的block大小。sparse_block_size为1时，为Token-wise稀疏化场景；sparse_block_size大于1且小于等于128时，为Block-wise稀疏化场景，块内token共享相同的稀疏化决策。默认值为1。 | INT64 | - |

> | attention_mode | 可选属性 | 表示attention的模式，仅支持传入2，表示MLA-absorb模式。 | INT64 | - |

> | return_softmax_lse | 可选属性 | 用于表示是否返回softmax_max和softmax_sum。True表示返回，False表示不返回，默认值为False。 | BOOL | - |

Constraints:

> ## 约束说明
>
> - 该接口支持推理场景下使用。
> - 该接口支持图模式。
> - KV_N仅支持1。
> - block_size为一个block的token数，block_size取值为16的倍数，且最大支持1024。
> - 参数query中的Q_D和key、value的KV_D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
> - 参数query、key、value的数据类型必须保持一致。
> - 当前只支持query_rope和key_rope传入，不支持rope为空。
> - sinks仅支持Ascend 950PR&950DT系列产品。
> - <term>Ascend 950PR&950DT系列产品</term>：
>   - Q_N支持1~128。
>   - sparse_block_size仅支持1。
>   - 仅在layout_kv为PA_BSND时，key、value和key_rope支持0轴非连续。
>   - return_softmax_lse为True时，支持layout_kv为PA_BSND。
> - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
>   - Q_N支持1/2/4/8/16/32/64/128。
>   - 参数sparse_block_size支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparse_block_size整除block_size。
>   - 不支持sinks。
>   - key、value和key_rope不支持非连续。
>   - return_softmax_lse为True时，layout_kv不支持PA_BSND。

`Q_N支持1/2/4/8/16/32/64/128` on A3 — Q heads per rank must be a power of two in that set, so a
TP16 split of a 64- or 128-head MLA model lands on `Q_N` = 4 or 8, far below the `gSize > 64`
threshold that would enable `IS_SPLIT_G`.

## Relevance

This is the strongest single fact in this whole investigation, because it is an A/B with the
confound removed: same vendor, same repository, same A3 `arch22` target, same decode shape,
same 512-token empirical unit — and the dense op splits KV across 16 cores while the sparse op
splits across 1. The 5.0x roofline miss on `SparseFlashAttention` is not a property of sparse
attention, of scattered DMA, or of A3. It is a missing tiling mode in one operator.

Three consequences for our 6.973 ms.

**The fix has a reference implementation to copy.** `SplitBNS` + `service_vector_flashdecode_mla.h`
is 845 lines of A3 AscendC that does exactly the partial-`(out, max, sum)` combine our
hand-written kernel needs. `SparseFlashAttention` already allocates the workspace for it
(`accumOutSize`, `logSumExpSize`) and already exposes `softmax_max` / `softmax_sum` as outputs
under `return_softmax_lse`, so the external-combine route is open without any CANN change: call
SFA `kvSplitPart_` times over disjoint slices of the sparse index list with
`return_softmax_lse=true`, then run our own `ColMax`/`ColAdd`/`MatDivs` rescale. The cost is
`kvSplitPart_` extra MIX_AIC launches per layer against our 13.58 us floor — at 16 slices that is
217 us per layer, which is far worse than the 75 us we currently pay. **So the external-combine
route only pays inside a megakernel**, where the 16 slices are 16 core-groups of one launch
rather than 16 launches. That is a direct argument for the megakernel being the enabling vehicle
rather than an optimization on top.

**`IFATiling`'s 512-token floor bounds how far the split can go.** `while ((maxActualseq_ /
kvSplitPart_) < 512U && kvSplitPart_ > 1U) kvSplitPart_--;` means at seq 8192 the maximum useful
split is 16, not 24. If our DSA top-k is 2048 rather than the full 8192, the same rule gives
`2048/512 = 4` — only 4 cores' worth of useful split. This is the number that decides whether
closing the parallelism gap is worth 16x or 4x, and it turns on the top-k, which is why the top-k
value matters more than any other single model parameter here.

**Our 5.0x is smaller than single-core execution predicts, so one of our premises is wrong.**
Section 8.3 says 1 core; section 9.1 says 3 HBM transits. Together they predict a gap far beyond
5.0x. The reconciliation candidates, in order of likelihood: (i) we pass `layout_query = "TND"`
with the `gSize` heads packed along `Q_T`, which makes `totalBaseNum = gSize` and recovers
`gSize` cores — testable by reading `Q_T` at the call site; (ii) our 1.382 ms roofline denominator
is a per-die or per-core bandwidth rather than aggregate HBM; (iii) the op is being called once
per head rather than once per layer, which would multiply the launch count and show up in our
3008-kernels-per-round figure. Determining which costs one plog read of
`SFA block dim: %u aiv Num: %u aic Num: %u.` and is the highest-value next measurement.

---

## 11. `SparseFlashMla` + `SparseFlashMlaMetadata` — the A3-supported sparse MLA op that DOES split S2, with an AI-CPU balanced scheduler

Same clone and commit (`gitcode.com/cann/ops-transformer`,
`3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20).
`attention/sparse_flash_mla/` and `attention/sparse_flash_mla_metadata/`.

The earlier Gaps entry recorded "nothing states ... whether `sparse_flash_mla` is A3-capable
(`flash_mla` itself is 950-only)". It is A3-capable, and it is the structural answer.

### 11.1 Product support — A3 is supported

`attention/sparse_flash_mla/README.md`:

> # SparseFlashMla
>
> ## 产品支持情况
>
> | 产品 | 是否支持 |
> | :------------------------------------------------------------ | :------: |
> | <term>Ascend 950PR&950DT系列产品</term> | √ |
> | <term>Atlas A3系列产品</term> | √ |
> | <term>Atlas A2系列产品</term> | √ |
> | <term>Atlas 200I/500 A2推理系列产品</term> | × |
> | <term>Atlas推理系列产品</term> | × |
> | <term>Atlas训练系列产品</term> | × |

> - 算子功能：
>
>   `SparseFlashMla`算子旨在完成以下公式描述的Attention计算，支持SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）、HCA（Heavily Compressed Attention）三类Attention计算场景。调用时需要使用`SparseFlashMlaMetadata`生成的任务列表`metadata`。
>
>   典型调用流程如下：
>
>   1. 准备`q`、`ori_kv`、`cmp_kv`、序列长度、`block table`、`sinks`等输入。
>   2. 调用`SparseFlashMlaMetadata`生成`metadata`。
>   3. 调用`SparseFlashMla`，将上一步得到的`metadata`传入主算子。

### 11.2 The metadata op's stated purpose is exactly the missing load balance

`attention/sparse_flash_mla_metadata/README.md`:

> - 算子功能：`SparseFlashMlaMetadata`是`SparseFlashMla`算子的前置算子，用于后续Attention计算生成负载均衡的任务划分方案。本算子不执行实际的Attention计算，而是根据输入参数在AI CPU计算出每个AI Core应处理的Attention计算起止范围，从而最大化计算资源的利用率，避免各Core间负载不均衡的问题。

> | metadata | 输出 | 表示`SparseFlashMla`主算子使用的任务切分结果，shape固定为(2048, )。 | INT32 | ND |

A fixed 2048-int32 (8 KiB) task list, computed on the AI CPU, consumed by the main kernel.

The A3 constraint note, which confirms a per-token sparse index list is the A3 use case:

> - Atlas A3系列产品约束：
>   - SWA稀疏ori_kv场景下，仅支持SWA模板，`has_ori_kv`为true、`has_cmp_kv`为false、`ori_topk`大于0、`ori_mask_mode`为0，`ori_win_left`和`ori_win_right`为非负数，且必须传入`ori_topk_length`。`ori_topk`应与配套主算子`ori_sparse_indices`最后一维K保持一致；`ori_topk_length`表示每个q token和KV head的左对齐有效索引条目数，取值应在[0, K]范围内；Metadata仅使用`ori_topk_length`生成任务切分。配套主算子在PA_BBND场景仍要求传入`seqused_ori_kv`。

> - <term>Atlas A3系列产品</term> ：num_heads_q/num_heads_kv仅支持1、2、4、8、16、32、64、128，不支持seqused_q、cmp_topk_length；SWA稀疏ori_kv场景支持ori_topk_length、ori_topk大于0及ori_mask_mode为0，ori_win_left和ori_win_right支持非负数；其他SWA场景ori_topk为0、ori_mask_mode为4、ori_win_left为127、ori_win_right为0；cmp_topk支持[0, 8192]，cmp_mask_mode仅支持3，SWA不传入cmp_kv，cmp_ratio不参与计算、CSA支持1、2或4、HCA支持128。

Note `Metadata仅使用ori_topk_length生成任务切分` — the split is computed from the *actual*
per-token valid index count, not from a padded worst case. That is precisely what
`SparseFlashAttention` on `arch22` cannot do (section 9.6: it pads to 512 and multiplies the
zeros; `oriTopkLength` exists only in `arch35`, section 10.4).

### 11.3 `BalanceSchedule` — the scheduler, verbatim

`attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp`:

```cpp
bool SparseFlashMlaMetadataCpuKernel::BalanceSchedule(SplitResult& splitRes)
{
    SplitContext splitContext(batchSize_);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = batchSize_ * numHeadsKv_;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;

    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    // 3、存在FD任务，对FD进行负载均衡分配
    if (splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U); // 至少使用1个core
    return true;
}
```

`CalcSplitPlan` walks a `(bN2, s1G, s2)` cursor across cores until the cost is exhausted:

```cpp
void SparseFlashMlaMetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext& splitContext,
                                                    SplitResult& result)
{
    const CostInfo& costInfo = splitContext.costInfo;
    const SplitInfo& splitInfo = splitContext.splitInfo;
    if (aicCoreNum_ == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    assignContext.bN2S2Loop = costInfo.bN2S2LoopOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    // 负载分配
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlocksToCore(splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}
```

and the cursor explicitly advances along S2 *within* an `(s1G)` unit — the axis
`SparseFlashAttention` pins at `actBatchS2 = 1`:

```cpp
    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) { // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }
```

### 11.4 `SplitFD` — the flash-decode combine is itself load-balanced across AIVs

```cpp
void SparseFlashMlaMetadataCpuKernel::SplitFD(SplitResult& splitRes)
{
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 计算当前最大冗余vec核数
    uint32_t emptyVectorNum = aivCoreNum_ - splitRes.numOfFdHead;
    // 计算每个核处理的load
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_; // 向上取整，避免核负载为0
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        // 冗余vec核数为0，此时规约任务无法进行更小的切分，只能1个vec核计算1个规约任务
        if (emptyVectorNum == 0U) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = 0U;
            splitRes.fdRes.fdMNum[curCoreIndex] = splitRes.fdRes.fdMSize[i];
            curCoreIndex++;
            continue;
        }
        // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        curFDVectorNum = std::max(1U, curFDVectorNum);
        // 计算当前归约任务每个核的行数，向上取整，避免行数为0
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize - 1U) / curAveMSize;
        // 需要使用的vec核数与当前剩余可用vec核数取最小
        curFDVectorNum = std::min(curFDVectorNum, emptyVectorNum + 1U); // 1: Fd任务自身带一个核
        // FD负载分配
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
        // 更新冗余vec核数
        emptyVectorNum -= (curFDVectorNum - 1U); // 1: 空余核不包含FD自身的核，要-1
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}
```

### 11.5 `GenMetadata` — the exact wire format of the per-core task list

```cpp
    } else {
        for (size_t i = 0; i < aicCoreNum_; ++i) {
            if (i >= splitRes.usedCoreNum) {
                metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
                continue;
            }
            metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
            // FA START
            metadataPtr->faMetadata[i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metadataPtr->faMetadata[i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metadataPtr->faMetadata[i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
            // FA END
            metadataPtr->faMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metadataPtr->faMetadata[i][FA_M_END_INDEX] = splitRes.gS1End[i];
            metadataPtr->faMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
            // firstFdDataWorkspace
            metadataPtr->faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
        }
    }

    // FD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metadataPtr->fdMetadata[i][FD_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metadataPtr->fdMetadata[i][FD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metadataPtr->fdMetadata[i][FD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}
```

and the op's own host tiling, which — unlike SFA — does not need to compute a split at all,
because the metadata already carries it:

```cpp
    usedCoreNum_ = aicNum;
    tilingData_.baseParams.set_usedCoreNum(usedCoreNum_);
```

The AI CPU kernel registration:

```cpp
namespace {
static const char* kernelType = "SparseFlashMlaMetadata";
REGISTER_CPU_KERNEL(kernelType, SparseFlashMlaMetadataCpuKernel);
} // namespace
```

## Relevance

CANN ships **two** sparse MLA attention operators that both claim A3 support, and they differ on
exactly the axis that governs our batch=1 cost:

| | `SparseFlashAttention` | `SparseFlashMla` |
| --- | --- | --- |
| A3 (`Atlas A3系列`) | √ | √ |
| S2 (KV) split across cores | no — `actBatchS2 = 1`, `FLASH_DECODE` literal `0U` | yes — `FA_S2_START_INDEX` / `FA_S2_END_INDEX` per core |
| flash-decode combine | dead code behind `if constexpr (FLASH_DECODE)` | `SplitFD` + `fdMetadata` per AIV |
| split computed where | in-kernel, `usedCoreNum = batch * q_tokens` | AI CPU, cost-model balanced, 2048-int32 task list |
| uses actual top-k length | no (pads to 512, `oriTopkLength` is `arch35`-only) | yes — `Metadata仅使用ori_topk_length生成任务切分` |
| per-call launch | 1 MIX_AIC kernel | 1 AI CPU kernel + 1 MIX kernel |

So the fix for our 6.973 ms may not require writing a kernel at all: it may be a matter of
calling the other operator. `SparseFlashMla` on A3 is documented for the SWA-sparse-`ori_kv`
case with `ori_topk > 0`, `ori_mask_mode = 0`, and `ori_topk_length` passed, where
`ori_topk` matches the last dim K of `ori_sparse_indices` — which is the shape of a DSA index
list. The caveats to verify before betting on it: A3 is restricted to `仅支持SWA模板`
(SWA template only) with `ori_win_left`/`ori_win_right` non-negative, so whether a pure
top-k-without-window DSA pattern is expressible needs a parameter-level check; and it costs an
extra AI CPU kernel per call, which at 95-ish layers is a second launch per layer against our
already-saturated 3008-kernel budget.

The deeper point for the megakernel is the *shape* of the answer. `SparseFlashMlaMetadata`
computes, on the host-side AI CPU, a per-core `(bN2_start, M_start, S2_start) ->
(bN2_end, M_end, S2_end)` range plus a per-AIV reduction assignment, and hands it to the kernel
as an 8 KiB int32 blob. That is the same pattern as DeepSeek's FlashMLA `get_mla_metadata`
already documented in section 4, and it is the mechanism our megakernel needs anyway: one
persistent kernel reading a precomputed work list rather than a per-op tiling struct. The
`SparseFlashMlaMetadata` field layout (`FA_CORE_ENABLE_INDEX`, `FA_*_START_INDEX`,
`FA_*_END_INDEX`, `FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX`, `FD_WORKSPACE_NUM_INDEX`,
`FD_M_START_INDEX`, `FD_M_NUM_INDEX`) is a ready-made schema to copy, and `SplitFD`'s
`averageLoad = ceil(totalFDLoad / aivCoreNum)` greedy assignment is a ready-made balancer.

---

## 12. The audit: flash-decode is live in every DENSE decode attention op and dead in every SPARSE one, and the balanced splitter is a shared library SFA does not link

Same clone and commit (`gitcode.com/cann/ops-transformer`,
`3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20).

### 12.1 The grep

```
$ cd ops-transformer
$ for f in <each attention op dir mentioning splitKVFlag_>; do \
      echo "$(grep -rn 'splitKVFlag_ *= *true' $f | wc -l)  $f"; done
2  attention/fused_infer_attention_score/op_host
4  attention/incre_flash_attention
0  attention/kv_quant_sparse_flash_attention
0  attention/nsa_selected_attention_infer
0  attention/sparse_flash_attention
```

Every operator that declares the flag also declares `kvSplitPart_`, a `splitKVParams` tiling
struct, `accumOut` / `logSumExp` workspace, and an `if constexpr (FLASH_DECODE)` kernel branch.
In the three sparse ops none of that is ever reached.

`attention/nsa_selected_attention_infer/op_host/nsa_selected_attention_infer_tiling.cpp` even
hard-codes the admission in its log line:

```cpp
// for flash decode
void NsaSelectTiling::FillTilingSplitKV() const
{
    tilingData_->splitKVParams.set_s2(kvSplitPart_);
    int64_t sInnerLoopSize_ = (maxActualseq_ + (kvSplitPart_ - 1)) / kvSplitPart_;
    sInnerLoopSize_ = Align(sInnerLoopSize_, blockSize_);
    OP_LOGD(context_->opName, "PA FlashDecode is not enabled, sInnerLoopSize is %ld, blockSize is %ld", sInnerLoopSize_,
            blockSize_);
    tilingData_->splitKVParams.set_sInnerLoopSize(sInnerLoopSize_);
    if (!splitKVFlag_) {
        tilingData_->splitKVParams.set_s2(0);
    }
}
```

`PA FlashDecode is not enabled` is emitted unconditionally, as a `OP_LOGD` and not as a warning.

### 12.2 `SparseFlashAttention` was forked from the FIA MLA tiling and the splitter was dropped

`attention/fused_infer_attention_score/op_host/arch22/fia_tiling_nonquant_mla.cpp`:

```cpp
void FiaTilingNonQuantMla::CalcInnerSize(uint32_t seqSize)
{
    sInnerSize_ = 512U;
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && fiaInfo_->inputLayout != TilingKeyLayout::TND) {
        if (seqSize == 256U) {
            sInnerSize_ = 128U;
        } else if (seqSize > 256U && seqSize <= sInnerSize_) {
            sInnerSize_ = (sInnerSize_ + 1U) / 2U;
```

Compare `attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp` (quoted in
full in section 8.1): identical function name, identical `sInnerSize_ = 512`, identical comment
text character-for-character, identical `splitKVFlag_ && ... != TND` guard, identical 256/128
constants. The two are the same code. The difference is everything around it.

FIA's `Split()`:

```cpp
void FiaTilingNonQuantMla::Split()
{
    CalcInnerSize(static_cast<uint32_t>(fiaInfo_->s2Size));

    // 构造分核输入参数
    BaseInfo baseInfo{};
    SplitParam splitParam{};
    CreateSplitInput(baseInfo, splitParam);

    // 构造分核输出参数
    SplitResult res{aicNum_, cvRatio_};
    SplitCore(aicNum_, baseInfo, splitParam, res);
    if (res.numOfFdHead > aicNum_ || res.usedCoreNum > aicNum_ || res.maxS2SplitNum > aicNum_ + 1U) {
        OP_LOGE(fiaInfo_->opName, "used_core_num: %u, num_of_fd_head: %u, max_s2_split_num: %u, aic_num: %u",
                res.usedCoreNum, res.numOfFdHead, res.maxS2SplitNum, aicNum_);
    }
    SetSplitOutput(res);
    tilingData_.innerSplitParams.set_mBaseSize(mBaseSize_);
    tilingData_.innerSplitParams.set_s2BaseSize(sInnerSize_);
    tilingData_.fdParams.set_numOfFdHead(res.numOfFdHead);
    tilingData_.fdParams.set_gS1BaseSizeOfFd(mFdBaseSize_);
    usedCoreNum_ = res.usedCoreNum;

    // kvSplitPart_,用于lse out workspace计算
    if (IsFlashDecode()) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = res.maxS2SplitNum;
        tilingData_.fdParams.set_usedVecNumOfFd(res.usedVecNumOfFd);
    }
    CalcMmResSize();
}
```

```cpp
bool FiaTilingNonQuantMla::IsFlashDecode()
{
    uint32_t tndFDCoreArrLen = tilingData_.fdParams.get_numOfFdHead();
    return tndFDCoreArrLen > 0U;
}
```

SFA's `Split()`, for comparison (section 8.2), is three assignments and `usedCoreNum_ = aicNum_`.

### 12.3 `attention/common/op_host/split_core.{h,cpp}` — a reusable balanced splitter with flash-decode, 797 + 279 lines

```cpp
void SplitCore(uint32_t coreNum, const BaseInfo& baseInfo, const SplitParam& param, SplitResult& result)
{
    // 打印baseInfo,param信息
    LogSplitCoreInput(baseInfo, param);
    SplitContext splitContext(baseInfo, param);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        result.usedCoreNum = 1U;
        result.bN2End[0] = baseInfo.bSize * baseInfo.n2Size;
        result.gS1End[0] = 0U;
        result.s2End[0] = 0U;
        return;
    }

    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore =
        static_cast<uint32_t>(std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    result.usedCoreNum = 1U;
    result.maxCost = INT64_MAX;

    SplitResult tmpResult{coreNum, result.vecCubeRatio};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        OP_LOGD("SplitRes", "CoreNum: %u", i);
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }

    // 3、存在FD任务，对FD进行负载均衡分配
    if (result.numOfFdHead > 0U) {
        SplitFD(result);
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U); // 至少使用1个core

    // 打印分核结果信息
    LogSplitCoreResult(result);
}
```

It is a *search*: it tries every core count from `minCore` to `maxCore` and keeps the plan with
the lowest `maxCost`. Note `maxCore = min(coreNum, totalBlockNum)` — and `totalBlockNum` here is
computed over `(batch, n2, s1G, s2)` blocks including S2, so unlike SFA's `totalBaseNum` it is
not capped at 1 for a one-token decode.

The cost model being minimized:

```cpp
int64_t CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2); // 6：M轴系数，10：S2轴系数
}
```

An explicit published cost model for A3 attention: cost = `6 * ceil(M/16) + 10 * ceil(S2/64)`.
The M axis is quantized at 16 and weighted 6; the S2 axis is quantized at 64 and weighted 10.
A one-token-query decode has `M = gSize`, so `ceil(M/16) = 1` and the M term contributes a
constant 6 regardless of head count up to 16 — i.e. the model says heads 1..16 are free, and all
cost is in S2.

The FD reduction balancer:

```cpp
void SplitFD(SplitResult& result)
{
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] * result.fdRes.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += result.fdRes.gS1SplitNumOfFdHead[i];
    }

    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, result.usedCoreNum * result.vecCubeRatio);
```

and the condition under which a cross-core reduction is recorded at all:

```cpp
bool IsNeedRecordFDInfo(const AssignContext& assignContext, const SplitResult& splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
```

The full public API of the splitter, from `split_core.h`:

```cpp
// cache calculation
void CalcBatchCache(uint32_t bIdx, const SplitContext& splitContext, BatchCache& batchCache);
void CalcS1GCache(uint32_t s1GIdx, const SplitContext& splitContext, const BatchCache& batchCache, S1GCache& s1GCache);
void CopyTmpResult(SplitResult& tmpRes, SplitResult& splitRes);
void ClearTmpResult(SplitResult& tmpResult);

// preprocess
void CalcSplitInfo(SplitContext& splitContext);
void CalcBatchCost(uint32_t bIdx, const SplitContext& splitContext, CostInfo& costInfo);
void CalcCostInfo(SplitContext& splitContext);

// assign
void UpdateCursor(const SplitContext& splitContext, AssignContext& assignContext);
void AssignByBatch(const SplitContext& splitContext, AssignContext& assignContext);
void AssignByRow(const SplitContext& splitContext, AssignContext& assignContext);
void AssignByBlock(AssignContext& assignContext);
void ForceAssign(const SplitContext& splitContext, AssignContext& assignContext);

// FD
bool IsNeedRecordFDInfo(const AssignContext& assignContext, const SplitResult& splitRes);
void RecordFDInfo(const SplitContext& splitContext, const AssignContext& assignContext, SplitResult& result);

// main
void SplitFD(SplitResult& result);
void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext& splitContext, SplitResult& result);
void SplitCore(uint32_t coreNum, const BaseInfo& baseInfo, const SplitParam& param, SplitResult& result);
void LogSplitCoreInput(const BaseInfo& baseInfo, const SplitParam& param);
void LogSplitCoreResult(const SplitResult& result);
void LogAssignContext(const char* phase, const AssignContext& assignContext);
```

`AssignByBatch` / `AssignByRow` / `AssignByBlock` / `ForceAssign` is a four-level fallback: split
by batch first, then by `(s1, G)` row, then by block (which is where S2 gets cut), then force.

### 12.4 Who links it

```
$ grep -rln 'split_core.h\|split_core_v2.h' --include=*.cpp --include=*.h attention/
common/op_host/split_core.cpp
common/op_host/split_core.h
common/op_host/split_core_v2.cpp
common/op_host/split_core_v2.h
fused_infer_attention_score/op_host/arch22/fia_tiling_nonquant.cpp
fused_infer_attention_score/op_host/arch22/fia_tiling_nonquant.h
fused_infer_attention_score/op_host/arch22/fia_tiling_nonquant_mla_arch22.h
fused_infer_attention_score/op_host/arch35/fia_tiling_fullquant_gqa.cpp
fused_infer_attention_score/op_host/arch35/fia_tiling_fullquant_gqa.h
fused_infer_attention_score/op_host/arch35/fia_tiling_fullquant_mla.cpp
fused_infer_attention_score/op_host/arch35/fia_tiling_fullquant_mla.h
fused_infer_attention_score/op_host/arch35/fia_tiling_fullquant_mx.cpp
fused_infer_attention_score/op_host/arch35/fia_tiling_fullquant_mx.h
fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp
fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.h
fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp
fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla_arch35.h
```

Exactly one operator family. And `sparse_flash_attention`'s complete include list:

```cpp
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "../op_kernel/sparse_flash_attention_template_tiling_key.h"
#include "sparse_flash_attention_tiling.h"
```

No `split_core.h`.

(`SparseFlashMlaMetadata`, section 11, does not include it either — it reimplements the same
algorithm, with the same function names `CalcSplitInfo` / `CalcCostInfo` / `CalcSplitPlan` /
`SplitFD` / `AssignBlocksToCore`, inside the AI CPU kernel. So there are two independent
implementations of this splitter in the repo and `SparseFlashAttention` uses neither.)

## Relevance

This closes the question. The 5.0x roofline miss on our 6.973 ms `SparseFlashAttention` is not a
hardware property of A3, not an intrinsic cost of scattered KV gather, and not a small-batch law.
It is one operator that was forked from `FusedInferAttentionScore`'s MLA tiling and shipped
without the `SplitCore` call. Everything else — the `splitKVParams` struct, the `accumOut` and
`logSumExp` workspace, the `if constexpr (FLASH_DECODE)` kernel branch, the `softmax_max` /
`softmax_sum` outputs — is still there, wired to constants.

The published cost model `6 * ceil(M/16) + 10 * ceil(S2/64)` is the most useful single artifact
for our own megakernel scheduler, because it is Huawei's own weighting of the two axes on this
chip and it says the S2 axis dominates by 10:6 per quantum, with the M axis flat up to 16 rows.
For a batch=1 TP16 decode with `gSize` in {4, 8} that is: all the cost is in S2, M is free, and
any scheduler that does not cut S2 is leaving the only axis that matters uncut.

Three things this makes concretely actionable against our numbers:

1. **Our 5.6 ms target is reachable but not by closing to the byte roofline.** Section 9 showed
   the op's own floor is ~3x the byte roofline because of the `kvMergeGm_` round trip. Section 12
   shows the parallelism is 1 core. The product of those two is the gap; the fix for the first is
   a kernel that keeps gathered KV in UB/L1, and the fix for the second is an S2 split with an
   lse combine. A megakernel does both at once, and neither in isolation gets most of the win.
2. **`FusedInferAttentionScore` is a measurable baseline we can run today.** It is on A3
   (`arch22`), it has an MLA non-quant path (`fia_tiling_nonquant_mla.cpp`), and it calls
   `SplitCore`. Running our batch=1 seq-8K decode through FIA *without* sparsity and comparing to
   `SparseFlashAttention` *with* sparsity would separate "sparse gather is expensive" from "sparse
   op has no decode tiling". Our measurement cannot currently tell those apart. If dense FIA on
   8192 tokens beats sparse SFA on top-k tokens, the sparsity is costing us rather than saving us
   — and that would be the most important thing to know.
3. **The `OP_LOGD("SplitRes", "CoreNum: %u", i)` and `LogSplitCoreResult` lines give FIA's chosen
   plan for free in plog**, so the core count FIA *would* pick for our shape is observable without
   any instrumentation.

---

## 13. GLM-5.2's attention, verbatim from the config and the tech report: 64 heads, 576-dim latent, `index_topk = 2048`, 78 layers

### 13.1 `zai-org/GLM-5.2` config.json

https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json
zai-org (Z.ai / Zhipu). Repo `createdAt 2026-06-16T07:39:20.000Z`, `lastModified 2026-09-01T11:36:01.000Z`.
Fetched 2026-10-20, HTTP 200, 3732 bytes.

```json
{
  "architectures": [
    "GlmMoeDsaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "dtype": "bfloat16",
  "ep_size": 1,
  "first_k_dense_replace": 3,
  "head_dim": 192,
  "hidden_act": "silu",
  "hidden_size": 6144,
  "index_head_dim": 128,
  "index_n_heads": 32,
  "index_share_for_mtp_iteration": true,
  "index_skip_topk_offset": 3,
  "index_topk": 2048,
  "index_topk_freq": 4,
  "index_topk_pattern": null,
  "indexer_rope_interleave": true,
  "indexer_types": [ ... 78 entries: 21 "full", 57 "shared" ... ],
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "kv_lora_rank": 512,
  "max_position_embeddings": 1048576,
  "mlp_layer_types": [ ... 78 entries: 3 "dense", 75 "sparse" ... ],
  "model_type": "glm_moe_dsa",
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "moe_router_dtype": "float32",
  "n_group": 1,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 78,
  "num_key_value_heads": 64,
  "num_nextn_predict_layers": 1,
  "pretraining_tp": 1,
  "q_lora_rank": 2048,
  "qk_head_dim": 256,
  "qk_nope_head_dim": 192,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-05,
  "rope_interleave": true,
  "rope_parameters": {
    "rope_theta": 8000000,
    "rope_type": "default"
  },
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "tie_word_embeddings": false,
  "topk_group": 1,
  "topk_method": "noaux_tc",
  "transformers_version": "5.12.0",
  "use_cache": true,
  "v_head_dim": 256,
  "vocab_size": 154880
}
```

`indexer_types` decoded: `full` at layer indices
**[0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74]** = **21 full,
57 shared**, matching the derivation in `configuration_glm_moe_dsa.py`:

```python
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]
```

(https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py)

`zai-org/GLM-5` (https://huggingface.co/zai-org/GLM-5/raw/main/config.json, 1384 bytes) is the
same architecture *without* `index_topk_freq` / `indexer_types` — every layer runs its own
indexer. `index_topk` is 2048 in both. `zai-org/GLM-5.3`
(https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json, 29464 bytes) is architecturally
identical to 5.2, confirmed by the Transformers doc
(https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/glm_moe_dsa.md):

> ### GLM-5.3
> GLM-5.3 uses the same base model as GLM-5.2 — every gain comes from post-training.

> ### GLM-5
> GLM-5 ([GlmMoeDsa](https://huggingface.co/papers/2602.15763)) is a 744B-parameter mixture-of-experts model with 40B active parameters per token, using DeepSeek Sparse Attention (DSA) for efficient 200K-token context handling. It was trained entirely on Huawei Ascend chips and matches frontier-level performance on reasoning and long-context benchmarks.

GLM-4.5 and GLM-4.6 are **not** a valid analogue: `"architectures": ["Glm4MoeForCausalLM"]`,
`"num_attention_heads": 96`, `"num_key_value_heads": 8`, `"head_dim": 128`, no `kv_lora_rank`, no
`index_*` field at all — plain GQA-8, dense attention
(https://huggingface.co/zai-org/GLM-4.6/raw/main/config.json, 1007 bytes).

### 13.2 The GLM-5 tech report on the attention design — and on Ascend specifically

https://arxiv.org/abs/2602.15763 / https://arxiv.org/html/2602.15763v2
GLM-5-Team (Aohan Zeng, Xin Lv, Zhenyu Hou, ..., Jie Tang), "GLM-5: from Vibe Coding to Agentic
Engineering". Submitted 17 Feb 2026 (v1), last revised 24 Feb 2026 (v2).

> Multi-latent Attention.
>
> By employing reduced key-value vectors, Multi-latent attention (MLA) [24] matches the effectiveness of Grouped-Query Attention (GQA) but offers superior GPU memory savings and faster processing for long-context sequences.
>
> However, in our experiments with Muon optimizer, we find that MLA with a 576-dimension latent KV-cache cannot match the performance of GQA with 8 query groups (denoted as GQA-8, 2048-dimension KV-cache). To overcome the performance gap, we propose an adaptation to the recipe of Muon optimizer in GLM-4.5.
> In the original recipe, we apply matrix orthogonalization to the up-projection matrices $W^{UQ},W^{UK},W^{UV}$ for multi-head queries, keys, and values. Instead, we split these matrices into smaller matrices for different heads and apply matrix orthogonalization to these independent matrices. The method, denoted as Muon Split, enables projection weights for different attention heads to update at different scales. As shown in Table 1, the method effectively improves the performance of MLA to match that of GQA-8. In practice, we also find that with Muon Split, the scale of attention logits of GLM-5 remains stable during pre-training without any clipping strategy.
>
> Another disadvantage of MLA is its high computational cost during decoding. In decoding, MLA performs a 576-dimensional dot product, higher than the 128-dimensional computation of GQA. While the number of attention heads in DeepSeek-V3 is selected according to the roofline of H800 [60], it is inappropriate for other hardware. Given the Multi-head Attention (MHA) style of MLA during training and prefilling, we increase the head dimension from 192 to 256 and decrease the number of attention heads by 1/3. This keeps the training computation and the number of parameters constant while decreasing the decoding computation. The variant, denoted as MLA-256 in Table 1, matches the performance of MLA under Muon Split.

The architecture table (GLM-4.5 | GLM-5):

> \# Total Parameters | 355B | 744B
> \# Activated Parameters | 32B | 40B
> \# Dense Layers | 3 | 3
> \# MoE Layers | 89 | 75
> \# MTP Layers | 1 | 1
> Hidden Dim | 5120 | 6144
> Dense Intermediate Dim | 12288 | 12288
> MoE Intermediate Dim | 1536 | 2048
> QK Head Dim | 128 | 192
> V Head Dim | 128 | 256
> Q LoRA Dim | – | 2048
> KV LoRA Dim | – | 512
> \# Attention Heads | 96 | 64
> \# Key-Value Heads | 8 | –
> \# Indexer Attn Heads | – | 32
> \# Indexer Head Dim | – | 128
> \# Experts (total) | 160 | 256
> \# Routed Experts | 8 | 8
> \# Shared Experts | 1 | 1
> Vocabulary Size | 151552 | 154880

On top-k, from the RL section:

> We conduct a very large-scale RL training on a model based on the DSA architecture. Compared with MLA, DSA introduces an additional indexer that retrieves the top-k most relevant key-value entries and computes attention sparsely over the retrieved subset.
> The retrieved top-k results are critical for RL stability. [...]
> However, directly adapting this strategy to indexer replay, i.e., storing the indexer's top-k indices at every token position is clearly impractical, since the $k=2048$ used by the indexer is much larger than the $k$ typically used in MoE, and storing all these indices would incur enormous storage costs as well as significant communication overhead between the training engine and the inference engine.

And Section 5, "Adapting GLM-5 to Chinese Chip Infrastructure" — this names our exact kernel set
and our exact machine:

> Mixed-Precision W4A8 quantization.
>
> To fit the 750B parameter GLM-5 model onto a single Atlas 800T A3 machine, we implemented a sophisticated W4A8 mixed-precision quantization strategy. Utilizing the msModelSlim tool, we applied specific precisions to different model components: standard Attention and MLP blocks use W8A8 (INT8), while the MoE experts are compressed to W4A8 (INT4) to drastically reduce memory footprint without significant accuracy loss. Advanced algorithms like QuaRot [2] for outlier suppression and Flex_AWQ_SSZ for scaling calibration were employed to maintain stability in low-bit deployment.
>
> High-Performance fusion kernels.
>
> To overcome the computational bottlenecks of sparse attention on Ascend NPUs, we developed a suite of customized fusion kernels: Lightning Indexer, Sparse Flash Attention, and MLAPO (Multi-head Latent Attention Pre-processing Optimization).
> Lightning Indexer integrates score calculation, ReLU, and TopK operations into a single kernel, allowing the NPU to overlap computation with memory access.
> For the Sparse Flash Attention kernel, we specifically optimized for GLM-5's sparse patterns. This kernel handles the selection of TopK tokens from the KV cache and sparse attention computation in parallel.
> Last, MLAPO fuses 13 small pre-processing operators into one "super operator", utilizing parallel processing between Vector and Cube units to boost end-to-end efficiency.
> [...]
> Through these hardware-level co-optimizations, GLM-5 on a single Chinese node achieves performance comparable to dual-GPU international clusters, while reducing deployment costs in long-sequence scenarios by 50%.

And the DSA adoption statement:

> 2.1.1 Continued Pre-Training with DeepSeek Sparse Attention (DSA)
>
> We use DSA in our training.
> The core philosophy of DSA [9] is to replace the traditional dense $O(L^{2})$ attention—which becomes prohibitively expensive at $128\text{K}$ contexts—with a dynamic, fine-grained selection mechanism. Unlike fixed patterns (like sliding windows), DSA "looks" at the content to decide which tokens are important.
> [...]
> DSA reduces the attention computation by roughly 1.5-2× for long sequences, which is very important for the reasoning-heavy agents we are building, being able to handle 128K contexts at half the GPU cost.

### 13.3 The reference indexer, and the exact top-k clamp

https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
HuggingFace Transformers, main branch, fetched 2026-10-20 (37299 bytes). Doc states: "*This
model was published in HF papers on 2026-02-17 and contributed to Hugging Face Transformers on
2026-02-09.*"

```python
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # GLM-MoE-DSA uses interleaved RoPE in the indexer
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, D]

        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
        if attention_mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
        else:
            index_scores = index_scores + attention_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]
```

and the per-layer dispatch:

```python
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)
```
```python
        # DSA: select this layer's top-k tokens, or reuse the previous full layer's on `"shared"` layers.
```

The vLLM-Ascend production clamp is the same, and states the decode value directly —
https://github.com/vllm-project/vllm-ascend `vllm_ascend/attention/indexer.py` (1111 lines):

```python
    @property
    def topk_output_width(self) -> int:
        return self.topk_tokens

    def get_topk_lengths(self, positions: torch.Tensor) -> torch.Tensor:
        return (positions + 1).clamp(min=0, max=self.topk_tokens)
```
```python
    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA_INDEXER"
```
```python
    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        return [128]
```
```python
        self.n_head: int = vllm_indexer.n_head  # 64
        self.head_dim: int = vllm_indexer.head_dim  # 128
        self.topk_tokens: int = vllm_indexer.topk_tokens  # 2048
        self.q_lora_rank: int = vllm_indexer.q_lora_rank  # 1536
```
```python
        reads the freshly written cache. ``compute_topk=False`` (SFA layers
```

### 13.4 DeepSeek-V3.2's DSA — the published formula and the 2048

`index_topk = 2048` independently in two DeepSeek files:

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json
deepseek-ai, `lastModified 2025-11-18T02:39:54.000Z`, fetched 2026-10-20, 1552 bytes:

```json
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 2048,
  "kv_lora_rank": 512,
  "num_attention_heads": 128,
  "num_hidden_layers": 61,
  "num_key_value_heads": 128,
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
```

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/inference/config_671B_v3.2.json

```json
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048
```

https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/DeepSeek_V3_2.pdf
(526189 bytes, 6 pages). DeepSeek-AI, "DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with
DeepSeek Sparse Attention":

> Prototype of DSA. The prototype of DSA primarily consists of two components: a lightning indexer and a fine-grained token selection mechanism.
>
> The lightning indexer computes the index score 𝐼𝑡,𝑠 between the query token h 𝑡∈R 𝑑 and a preceding tokenh 𝑠∈R 𝑑, determining which tokens to be selected by the query token:
>
> 𝐼𝑡,𝑠 = ∑︁_{𝑗=1}^{𝐻𝐼} 𝑤𝐼_{𝑡,𝑗}·ReLU( q𝐼_{𝑡,𝑗}·k 𝐼_𝑠 ),  (1)
>
> where 𝐻𝐼 denotes the number of indexer heads; q 𝐼_{𝑡,𝑗} ∈R 𝑑𝐼 and 𝑤𝐼_{𝑡,𝑗} ∈R are derived from the query token h 𝑡; and k 𝐼_𝑠∈R 𝑑𝐼 is derived from the preceding token h 𝑠. We choose ReLU as the activation function for throughput consideration. Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable.
>
> Given the index scores {𝐼𝑡,𝑠} for each query token h 𝑡, our fine-grained token selection mechanism retrieves only the key-value entries {c𝑠} corresponding to the top-k index scores. Then, the attention output u 𝑡 is computed by applying the attention mechanism between the query token h 𝑡 and the sparsely selected key-value entries {c 𝑠}:
>
> u𝑡 =Attn( h𝑡, { c𝑠 | 𝐼𝑡,𝑠∈Top-k(𝐼𝑡,:) } ).  (2)

> Instantiate DSA Under MLA. For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA (DeepSeek-AI, 2024) for DeepSeek-V3.2-Exp. At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency (Yuan et al., 2025). Therefore, we implement DSA based on the MQA (Shazeer, 2019) mode of MLA, where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token.

> In this sparse training stage, we use a learning rate of 7.3× 10−6, and select 2048 key-value tokens for each query token. We train both the main model and the indexer for 15000 steps, with each step consisting of 480 sequences of 128K tokens, resulting in a total of 943.7B tokens.

> Inference Costs. DSA reduces the core attention complexity of the main model from O(𝐿2) to O(𝐿𝑘), where 𝑘 (≪𝐿) is the number of selected tokens. Although the lightning indexer still has a complexity of O(𝐿2), it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus. Combined with our optimized implementation, DSA achieves a significant end-to-end speedup in long-context scenarios. Figure 3 presents how token costs of DeepSeek-V3.1-Terminus and DeepSeek-V3.2-Exp vary with the token position in the sequence. These costs are estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour. Note that for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions.

The authors' reference implementation
(https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/inference/model.py, 38639 bytes):

```python
        weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        index_score = fp8_index(q_fp8.contiguous(), weights, self.k_cache[:bsz, :end_pos].contiguous(), self.k_scale_cache[:bsz, :end_pos].contiguous())
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
```

with the separate indexer K cache:

```python
        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn), persistent=False)
        self.register_buffer("k_scale_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, dtype=torch.float32), persistent=False)
```

### 13.5 The 656-byte C8 KV row — the byte constant our roofline turns on

https://raw.githubusercontent.com/vllm-project/vllm-project.github.io/main/_posts/2025-09-29-deepseek-v3-2.md
vLLM Team, 2025-09-29, "DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action".
(The rendered page https://vllm.ai/blog/2025-09-29-deepseek-v3-2 returns HTTP 302 into a
corporate proxy block page from this network; the git source serves the text.)

> Another interesting point is the handling of the FP8 KV cache, which this model supports. For MLA, each token's KV cache is 656 bytes, structured as:
>
> * First 512 bytes: The "quantized NoPE" part, containing 512 `float8_e4m3` values.
> * Next 16 bytes: Scale factors, containing 4 `float32` values. The first `float32` is the scale for the first 128 `float8_e4m3` values, the second for the next 128, and so on.
> * Last 128 bytes: The "RoPE" part, containing 64 `bfloat16` values. This part is not quantized for accuracy.

> For each new query token, it now passes through the indexer to compute the top 2048 tokens to attend to. A query for a token is a tensor of shape `(h, d)`, with `h` being the number of query heads, and `d` being the head dimension. The context of size `n` is a 2D tensor of shape `(n, d)`. The computed logits (relevance scores between the query and the context) are a tensor of shape `(n, h)`. Weighting the logits by the head weights of shape `(h,)`, we get a tensor of shape `(n,)`. We need to produce a `(2048,)` integer tensor of the indices of the top-2048 tokens, with `-1` filled for the rest if there are fewer than 2048 tokens.

> However, for the indexer key cache, it is stored on a per-block basis. This is one of the reasons we only support block size 64 for this model; the other being that FlashMLA is tailored to it as well. The first `block_size * head_dim` entries contain the value, the rest contain the scaling factor:
>
> ```
> x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
> x_fp8[ :, block_size * head_dim :] = scales.view(num_blocks, block_size).view(dtype=torch.uint8)
> ```
>
> In the indexer, the cache for one token is not stored contiguously.

The Ascend kernel README states the same 656 and gives the formula —
https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/kv_quant_sparse_flash_attention_vllm/README.md:

> k_nope、query相同数据类型的k_rope和float32的量化参数按D维度拼接得到。[…] layout_kv为"PA_BSND"时shape为[block_num, block_size, KV_N, KV_D]，其中block_num为PageAttention时block总数，block_size为一个block的token数，block_size取值为16的整数倍，最大支持到1024。KV_N仅支持1；KV_D = 512 + rope_head_dim*2 + 4*4，表示每行拼接数据的字节数。A2/A3 INT8 C8支持528（rope_head_dim=0）或656（rope_head_dim=64）；[…] 4个FLOAT32 scale组成的区域起始偏移分别为512或640字节，偏移从0开始。

> 代表sparse阶段的block大小。sparse_block_size为1时，为Token-wise稀疏化场景；sparse_block_size大于1且小于等于128时，为Block-wise稀疏化场景，块内token共享相同的稀疏化决策。

> - API功能：`kv_quant_sparse_flash_attention_vllm`在`sparse_flash_attention`的基础上支持了[Per-Token-Head-Tile-128量化]输入。随着大模型上下文长度的增加，Sparse Attention的重要性与日俱增，这一技术通过"只计算关键部分"大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

## Relevance

Everything now locks onto the kernel findings of sections 8-12, and it produces one number that
contradicts our measurement.

**The shape, resolved.** GLM-5.2: 64 Q heads, 78 layers, `kv_lora_rank 512` + `qk_rope_head_dim
64` = the 576-dim latent that `SparseFlashAttention`'s `约束说明` requires (`Q_D ... 为512`,
`Dr ... 为64`). At TP16 that is **`Q_N = 64 / 16 = 4` heads per rank** — a legal value in the op's
`Q_N支持1/2/4/8/16/32/64/128` list, and far below the `gSize > 64` threshold that would turn on
`IS_SPLIT_G` (section 8.1). So `IS_SPLIT_G = 0` is confirmed for our configuration, not inferred.
And in the published cost model `6 * ceil(M/16) + 10 * ceil(S2/64)` (section 12.3),
`ceil(4/16) = 1` — the M term is a flat 6 and *all* of the modelled cost is S2, the one axis the
op refuses to split.

**The byte count, resolved.** Top-k is 2048, clamped only by position:
`(positions + 1).clamp(max=2048)`, so at seq 8192 a decode step reads exactly **2048 of 8192 KV
tokens = 25.0%**, every layer. C8 row = 656 B. Per layer per query token:
`2048 x 656 = 1,343,488 B = 1.28 MiB`. Over 78 layers: **104.8 MB per decode step**.

**The contradiction.** `KV_N仅支持1` — the MLA latent is MQA, one KV head, shared across all
query heads of the token. It is therefore **not shardable by tensor parallelism**. At TP16 with
`--data-parallel-size 1`, all 16 ranks read the *same* 104.8 MB. Our stated roofline for
`SparseFlashAttention` is 1.382 ms of bytes "at batch=1 seq 8K TP16". 104.8 MB / 1.382 ms implies
a bandwidth of **75.8 GB/s**. If our roofline uses A3 per-die HBM bandwidth — order 1.2-1.6 TB/s
— then 1.382 ms corresponds to 1.7-2.2 GB of bytes, i.e. **16-21x more than the 104.8 MB that
`index_topk = 2048` actually requires**. That factor is suspiciously close to TP16. Two
possibilities, and they point in opposite directions:

- If our roofline was computed from the **full 8192-token** KV rather than the 2048 selected
  (8192 x 656 x 78 = 419 MB, 4x more) and/or multiplied by a TP factor, then the roofline is
  *overstated* and the true gap is much larger than 5.0x — consistent with sections 8-12
  predicting a 1-core, 3-HBM-transit execution.
- If our roofline already counts only the selected 2048 tokens once, then the implied bandwidth
  is 75.8 GB/s, which is roughly what **one** AI core can pull — and then our measurement is
  independently confirming section 8.3's conclusion that `usedCoreNum = 1`.

Either way the resolution is the same single check, and it is worth doing before any kernel work:
recompute the `SparseFlashAttention` roofline as `2048 x 656 x (SFA calls per round)` against
aggregate per-die HBM bandwidth, with **no** division by TP. If that lands near 65-90 us rather
than 1.382 ms, then the real gap is ~80-100x, not 5.0x, and the 5.6 ms we hoped to recover is a
severe underestimate of what is on the table.

**Two facts that bound the opportunity.** First, index sharing does *not* reduce SFA work: 57 of
78 layers skip the indexer but all 78 still attend over 2048 tokens. Our SFA budget is 78 x 2048,
not 21 x 2048. Second, GLM's own team names `Sparse Flash Attention` as a kernel they wrote for
Ascend and describes it as handling "the selection of TopK tokens from the KV cache and sparse
attention computation **in parallel**" — the gather/compute overlap that section 9.5's
`SetFlag`/`WaitFlag` pair on `MTE2_MTE3` demonstrably does *not* achieve in the public `arch22`
code. Either the shipped `ops-transformer` kernel is not the one they measured, or the overlap
they claim is the one we are not getting.

---

## 14. `FusedSparseAttentionOverlap` — the shipped A3 fix, whose only compiled template IS the GLM-5.2 decode combination

https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/fused_sparse_attention_overlap/README.md
vLLM-Ascend project, CANN custom op. Fetched 2026-10-20 via
`gh api repos/vllm-project/vllm-ascend/contents/csrc/attention/fused_sparse_attention_overlap/README.md`,
228 lines.

### 14.1 Product support and function

> # FusedSparseAttentionOverlap
>
> ## 产品支持情况
>
> | 产品 | 是否支持 |
> | :---------------------------- | :-----------: |
> | <term>Ascend 950PR&950DT 系列产品</term> | √ |
> | <term>Atlas A3 系列产品</term> | √ |
> | <term>Atlas A2 系列产品</term> | √ |
> | <term>Atlas 200I/500 A2 推理产品</term> | × |
> | <term>Atlas 推理系列产品</term> | × |
> | <term>Atlas 训练系列产品</term> | × |
>
> ## 功能说明
>
> - API功能：`fused_sparse_attention_overlap`面向大序列稀疏注意力推理场景，在一个算子中完成Selection KV Cache h2d更新和Sparse Flash Attention计算。算子根据TopK索引及缓存元数据判断KV是否命中Selection KV Cache；未命中的KV从Full KV Cache加载并写入Selection KV Cache，同时通过流水调度将KV搬运与注意力计算重叠，减少独立Gather和Sparse Flash Attention串行执行的开销。
>
> - 计算公式：
>
>     $$
>     \text{softmax}(\text{scaleValue} \cdot (Q_{NoPE}@\tilde{K}_{NoPE}^{T} + Q_{RoPE}@\tilde{K}_{RoPE}^{T}))@\tilde{V}
>     $$
>
>     其中，$\tilde{K}$和$\tilde{V}$由`selectionTopkIndices`指定。命中项从Selection KV Cache读取，未命中项从Full KV Cache读取。算子执行后会原地更新`selectionKRoPE`、`selectionKvCache`、`selectionKvBlockStatus`和`selectionMembershipMap`，供后续Token复用。

### 14.2 THE constraint list — the vendor states the GLM-5.2 decode template

> ## 约束说明
>
> - 该接口仅支持推理场景，并支持图模式。
> - 当前编译的模板仅支持GLM5.2 Decode使用的组合：`FLASH_DECODE=0`、`layoutQuery="TND"`、`layoutKv="PA_BSND"`、`TEMPLATE_MODE=V_TEMPLATE`、`IS_SPLIT_G=0`。
> - 当前支持FLOAT16和BFLOAT16，`query`、Full KV Cache及Selection KV Cache的数据类型必须保持一致。
> - 当前仅支持MLA-absorb形态，NoPE维度D为512，RoPE维度Dr为64；`query`最后一维为576，`attentionOut`最后一维为512。
> - 当前验证的分组头配置为N1=4、N2=1，PageAttention的`block_size`为128。
> - `selectionKvBlockStatus`最后一维至少容纳TopK个Token ID和一个有效数量字段，并按8个INT32元素对齐。
> - 当前`selectionMembershipMap`每行包含16376个Token映射位置、8个控制字段及对齐空间，`membership_stride`为16400。
> - `selectionTopkIndices`中的有效Token ID必须在对应Batch的`fullKvActualSeq`范围内。
> - Selection KV Cache容量不得小于TopK，Full KV Cache和Selection KV Cache的Block Table必须覆盖所有被访问的逻辑块。

### 14.3 The Selection KV Cache state

> | selectionKvCache | 输入/输出（原地更新） | NPU侧Selection KV Cache，采用PageAttention排布，shape为(selection_block_num,selection_block_size,D)。未命中的KV数据会从fullKvCache加载并写入该tensor。 | FLOAT16、BFLOAT16 | ND |

> | selectionKvBlockStatus | 输入/输出（原地更新） | Selection KV Cache的驻留Token状态，shape为(B,N2,status_stride)。前TopK个位置记录驻留Token ID，并使用附加位置记录有效数量。status_stride需要满足算子对齐要求。 | INT32 | ND |

> | selectionMembershipMap | 输入/输出（原地更新） | Selection KV Cache的Token到Slot映射及紧凑搬运计划，shape为(B,N2,membership_stride)。既可由算子内部Planner维护，也可承载框架生成的External Plan。 | INT16 | ND |

> | selectionTopkIndices | 输入 | Indexer输出的TopK Token索引，不支持空tensor。支持三维(T1,N2,TopK)或可展平为该排布的四维输入。 | INT32 | ND |

> | fullKvCache | 输入 | 完整KV Cache，采用PageAttention排布，shape为(full_block_num,block_size,D)。作为Selection KV Cache未命中时的数据源。 | FLOAT16、BFLOAT16 | ND |

> | fullKRoPE | 输入 | 完整K RoPE缓存，采用PageAttention排布，shape为(full_block_num,block_size,Dr)。 | FLOAT16、BFLOAT16 | ND |

> | fullKvBlockTable | 输入 | Full KV Cache的逻辑块到物理块映射表，shape为(B,full_block_count)。 | INT32 | ND |

> | fullKvActualSeq | 输入 | 各Batch的Full KV有效序列长度，shape为(B)。 | INT32 | ND |

> | fullQActualSeq | 输入 | TND排布下Query的有效序列信息，元素个数需要与query第一维对应。Decode场景中每个Query Token的长度为1。 | INT32 | ND |

> | scaleValue | 属性 | Query与Key乘积的缩放系数，通常设置为1/sqrt(D+Dr)。 | FLOAT | - |

> | sparseBlockSize | 属性 | 稀疏选择粒度。当前融合路径仅支持Token-wise稀疏化，取值为1。 | INT64 | - |

> | selectionTopkBlockSize | 属性 | TopK索引的选择粒度，当前仅支持1。 | INT64 | - |

> | layoutQuery | 属性 | query的数据排布格式，当前仅支持"TND"。 | STRING | - |

> | layoutKv | 属性 | Full KV Cache的数据排布格式，当前仅支持PageAttention排布"PA_BSND"。 | STRING | - |

> | sparseMode | 属性 | 稀疏模式。当前Decode融合路径使用3，表示rightDownCausal模式。 | INT64 | - |

> | attentionOut | 输出 | Attention计算结果，shape为(T1,N1,D)。输出不包含Query的RoPE维度。 | FLOAT16、BFLOAT16 | ND |

### 14.4 What vLLM-Ascend actually ships today, for contrast

https://github.com/vllm-project/vllm-ascend `vllm_ascend/attention/sfa_v1.py` (2213 lines):

```python
        result = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=query.contiguous(),
            key=cache,
            value=cache,
            sparse_indices=indices,
            scale_value=scale,
            sparse_block_size=1,
            block_table=metadata.block_table,
            actual_seq_lengths_query=metadata.query_start_loc[1:].to(torch.int32),
            actual_seq_lengths_kv=metadata.seq_lens.to(torch.int32),
            query_rope=None,
            key_rope=None,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )
```

`sparse_block_size=1`, `layout_query="TND"`, `layout_kv="PA_BSND"`, `attention_mode=2`,
`sparse_mode=3`, `return_softmax_lse=False`. This is the plain `SparseFlashAttention` of
sections 8-12, with no Selection KV Cache and no overlap.

And the documented single-node A3 GLM-5.2-W4A8C8 recipe —
https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html
(source `docs/source/tutorials/models/GLM5.2.md` in vllm-project/vllm-ascend), fetched 2026-10-20:

> 5.1.1 Single-node Deployment
>
> Quantized model GLM-5.2-w4a8c8 can be deployed on 1 Atlas 800 A3 (64GB × 16) .

> export HCCL_BUFFSIZE=200
> export HCCL_OP_EXPANSION_MODE="AIV"
> export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
> vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
> --host 0.0.0.0 \
> --port 8000 \
> --api-server-count 1 \
> --data-parallel-size 2 \
> --enable-expert-parallel \
> --tensor-parallel-size 8 \
> --seed 1024 \
> --served-model-name glm-5 \
> --tool-call-parser glm47 \
> --reasoning-parser glm45 \
> --enable-auto-tool-choice \
> --max-num-seqs 12 \
> --max-model-len 135000 \
> --max-num-batched-tokens 8192 \
> --trust-remote-code \
> --gpu-memory-utilization 0.92 \
> --quantization ascend \
> --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
> --attention_config.indexer_kv_dtype int8 \
> --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true,"multistream_overlap_shared_expert":true, "enable_flashcomm1": true, "enable_fused_mc2": 0}' \
> --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'

> Notice:
> The parameters are explained as follows:
>
> For single-node deployment, we recommend using dp1tp16 and turn off expert parallel in low-latency scenarios.

> Sparse SFA C8 | A3 (w8a8c8); long-context prefill | --kv-cache-dtype int8 | Sparse Flash Attention skips unnecessary attention computation of the C8 quantized model, accelerating long-context prefill | Experimental in v0.23.0. On the w8a8c8 weights it can be combined with DCP/context parallelism (as in the reference configs in this document); on the w4a8c8 weights enabling both together has known issues and is not recommended

> Sparse LI C8 | A3 (w8a8c8) | --attention_config.indexer_kv_dtype int8 | Sparse attention optimization reduces computation of the C8 quantized model, improving throughput | Independent of --kv-cache-dtype int8

> --kv-cache-dtype turboquant_4bit_nc: Enables 4-bit TurboQuant quantization for the SFA packed main KV cache, as an alternative to --kv-cache-dtype int8. It requires an A2/A3 device, does not support context parallelism (DCP/PCP), cannot be combined with the C8 cache, and in Prefill-Decode disaggregation the Mooncake option "use_ascend_direct" must be false.

Only the 1M-context configs add context parallelism:

> --prefill-context-parallel-size 1 \
> --decode-context-parallel-size 16 \
> --cp-kv-cache-interleave-size 128 \

### 14.5 Sibling kernels in the same tree

`gh api repos/vllm-project/vllm-ascend/contents/csrc/attention --jq '.[].name'`, 2026-10-20:

```
attn_res_fwd
chunk_kda_fwd
common
compressor
compressor_metadata
fused_lightning_indexer_manage
fused_quant_lightning_indexer_manage
fused_scatter_copy_sparse_flash_attention
fused_sparse_attention_overlap
indexer_compress_epilog_v2
inplace_partial_rotary_mul
k2q_csr
kda_gate_cumsum_vllm
kda_layout_swap12
kv_compress_epilog_v2
kv_quant_sparse_attn_sharedkv
kv_quant_sparse_attn_sharedkv_metadata
kv_quant_sparse_flash_attention_vllm
lightning_indexer_v2
msa_index_score
quant_lightning_indexer_v2
quant_lightning_indexer_v2_metadata
recurrent_gated_delta_rule_v310
rms_norm_dynamic_quant
sparse_attention_score
sparse_attn_sharedkv
sparse_attn_sharedkv_metadata
sparse_flash_attention
sparse_flash_mla
sparse_flash_mla_metadata
```

Note `fused_scatter_copy_sparse_flash_attention`, `sparse_attn_sharedkv` +
`sparse_attn_sharedkv_metadata`, and `kv_quant_sparse_attn_sharedkv` + its metadata — four more
gather/attention fusion variants, two of them metadata-driven in the section-11 style.

## Relevance

This section confirms, from the vendor, every inference sections 8-13 drew from source — and then
hands us the fix.

**The confirmation is line-for-line.** `当前编译的模板仅支持GLM5.2 Decode使用的组合：
FLASH_DECODE=0、layoutQuery="TND"、layoutKv="PA_BSND"、TEMPLATE_MODE=V_TEMPLATE、IS_SPLIT_G=0`
plus `当前验证的分组头配置为N1=4、N2=1，PageAttention的block_size为128`. Every one of these was
predicted above and is now stated as *the* GLM-5.2 decode configuration:

| predicted in | prediction | vendor statement |
| --- | --- | --- |
| 8.1 | `FLASH_DECODE` never 1 | `FLASH_DECODE=0` |
| 8.1, 13 | `IS_SPLIT_G = 0` because `Q_N <= 64` | `IS_SPLIT_G=0`, `N1=4` |
| 8.6, 9.1 | `sparse_block_size = 1` → `V_TEMPLATE` → `kvMergeGm_` round trip | `TEMPLATE_MODE=V_TEMPLATE`, `sparseBlockSize ... 取值为1` |
| 10.5, 13 | 64 heads / TP16 = 4 per rank | `N1=4、N2=1` |
| 13 | 576 = 512 nope + 64 rope | `NoPE维度D为512，RoPE维度Dr为64；query最后一维为576` |

**And it resolves the 5.0x arithmetic.** `layoutQuery="TND"` is the answer to section 10's open
question. In TND, `InitCalcParamsEach` reads `actBatchS1 = GetBalanceActualSeqLengths(...)` as the
Q token count, so `totalBaseNum = T1`, not 1. The shipped recipe runs
`--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'`, so a decode step
carries **4 query tokens** (1 + 3 MTP) and `usedCoreNum = 4`. Not 1, and not 24 — **4 of 24 cube
cores**, with 20 taking the early `return`.

The two factors are of different kinds and must not simply be multiplied. Section 9.1's 3-HBM-transit
staging is a *bytes* claim: it raises the op's own floor to ~3x the algorithmic byte roofline, and
it composes with the roofline additively-in-bytes. Section 8.3's 4-of-24 is a *bandwidth-ceiling*
claim, and 4 cores do not pull 4/24 of HBM — a handful of AI cores can already saturate a large
fraction of a die's bandwidth, so the penalty is well below 6x and is not separable from the
transit count without a measurement. The honest statement is: the op's floor is ~3x roofline from
staging alone, our measured 6.973 ms is ~1.7x that floor, and the residual 1.7x is where the
4-of-24 core limit, the per-Mmad `PipeBarrier<PIPE_M>` of section 9.7, the `MTE2_MTE3`
serialization of section 9.5, and the 13.58 us MIX_AIC launch floor all live. Per call that is
6.973 ms / 78 layers = 89.4 us, of which 13.58 us is launch floor — 15% of the op's cost is
launch before a single byte moves.

The TND read is verified in source —
`attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h`:

```cpp
__aicore__ inline uint64_t SparseFlashAttentionMla<SFAT>::GetBalanceActualSeqLengths(
    GlobalTensor<int32_t>& actualSeqLengths, uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (sfaKernelConstInfo.actualLenDimsQ == 0) {
            return sfaKernelConstInfo.qSeqSize;
        } else if (sfaKernelConstInfo.actualLenDimsQ == 1) {
            return actualSeqLengths.GetValue(0);
        } else {
            return actualSeqLengths.GetValue(bIdx);
        }
    }
}
```

At `bIdx == 0` in TND this returns `actualSeqLengths.GetValue(0)`, the prefix sum at batch 0 —
i.e. the Q token count. `sfa_v1.py` passes exactly that:
`actual_seq_lengths_query=metadata.query_start_loc[1:].to(torch.int32)`.

**The actionable fact.** `FusedSparseAttentionOverlap` is A3-supported, graph-mode capable, and
its only compiled template is exactly our configuration. It attacks the gather cost from a
direction none of sections 8-12 suggested: a persistent **Selection KV Cache** keyed by resident
token ID (`selectionKvBlockStatus`: `前TopK个位置记录驻留Token ID，并使用附加位置记录有效数量`)
plus a token→slot map and compaction plan (`selectionMembershipMap`, 16376 mapping slots,
`membership_stride = 16400`). Because consecutive decode steps' top-2048 sets overlap heavily —
the IndexCache paper measures 70-100% overlap between *adjacent layers*, and temporal overlap
between adjacent *tokens* at fixed layer is the same phenomenon — most of the 2048 rows are
already resident and only the misses are fetched from `fullKvCache`. On top of that it pipelines
the miss gather against the attention compute: `通过流水调度将KV搬运与注意力计算重叠，减少独立
Gather和Sparse Flash Attention串行执行的开销`.

That is the mechanism that removes section 9.1's extra HBM transits *and* section 9.5's
`SetFlag`/`WaitFlag` serialization, in a shipped A3 kernel, without us writing one. It does not
fix the 4-of-24 core problem — `FLASH_DECODE=0` and `IS_SPLIT_G=0` are still in its template
list, so the parallelism gap survives and remains ours to close in the megakernel.

Two caveats worth stating. First, `Selection KV Cache容量不得小于TopK` — at least 2048 token
slots of 576 halves = 2.25 MiB per (batch, n2) of extra HBM, plus the `selectionMembershipMap` at
16400 INT16 = 32 KiB per row. Second, the published recipe uses `--tensor-parallel-size 8
--data-parallel-size 2` rather than TP16, while the prose says `For single-node deployment, we
recommend using dp1tp16 and turn off expert parallel in low-latency scenarios` — so the TP16
low-latency path is recommended but not the one in the reference script, and `N1=4` in the
constraint list corresponds to TP16, not to the TP8 script (which would give `N1=8`).

**One number that bounds what sparsity is buying us at 8K.** `index_topk = 2048` against seq 8192
is 25% of the KV. A dense `FusedInferAttentionScore` decode over all 8192 tokens would read 4x
the bytes but, per section 12, would run `SplitCore` and use up to 16 cores instead of 4, with no
`kvMergeGm_` round trip and no scattered gather. 4x the bytes at 4x the cores and 1/3 the transits
is roughly break-even on paper. At seq 8K the sparsity may be costing us. That A/B is one
experiment and it is the one to run first.

---

## 15. The A3 decode-attention alternatives, side by side: every one is MIX_AIC, so the launch floor is unavoidable outside a megakernel

Same clone and commit (`gitcode.com/cann/ops-transformer`,
`3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20).

### 15.1 Kernel task type, per operator

```
$ for d in <ops>; do grep -rhn 'KERNEL_TASK_TYPE' $d/op_kernel/*.cpp; done
=== fused_infer_attention_score ===
78:        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
=== sparse_flash_mla ===
101:    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
103:    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
=== sparse_flash_attention ===
83:    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
=== kv_quant_sparse_flash_attention ===
132:    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

`sparse_flash_mla`'s two entries are arch-gated, and the gate names the A3 arch macro directly:

```cpp
#if (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
```

`IncreFlashAttention` is the only one that varies the CV ratio *per tiling key* rather than with a
single default — `attention/incre_flash_attention/op_kernel/incre_flash_attention_arch22.h`:

```cpp
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
...
    KERNEL_TASK_TYPE(QF16_KVF16_OUTF16_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
```

`NsaSelectedAttentionInfer` has no `KERNEL_TASK_TYPE` directive in its `op_kernel/` at all.

### 15.2 The comparison

All on `Atlas A3系列产品 √`, for a batch=1 single-query (or MTP-4) MLA decode with a sparse
block list:

| op | sparse index input | S2 split across cores | combine | task type | A3 decode viability |
| --- | --- | --- | --- | --- | --- |
| `SparseFlashAttention` | `sparse_indices` `[T1,KV_N,K]`, `sparse_block_size` 1..128 | **no** (`FLASH_DECODE=0`, `actBatchS2=1`) | dead code | `MIX_AIC_1_2` | this is what we run; 4 of 24 cores at `T1=4` |
| `KvQuantSparseFlashAttention` | same + C8/TQ4 packed slot | **no** (0 `splitKVFlag_` sites) | dead code | `MIX_AIC_1_2` | folds dequant in; same core limit |
| `NsaSelectedAttentionInfer` | NSA selected blocks | **no** (0 sites, logs `PA FlashDecode is not enabled`) | dead code | none declared | same core limit |
| `SparseFlashMla` + `...Metadata` | `ori_sparse_indices` + `ori_topk_length`; A3 SWA template only | **yes** (`FA_S2_START/END_INDEX` per core, AI-CPU balanced) | `SplitFD` + `fdMetadata` per AIV | `MIX_AIC_1_2` on A3 | real split; A3 restricted to `仅支持SWA模板` |
| `FusedInferAttentionScore` | none (dense) | **yes** (`SplitCore` + `IsFlashDecode`) | `numOfFdHead` + `usedVecNumOfFd` | `MIX_AIC_1_2` | dense baseline; 4x the bytes, up to 16 cores |
| `IncreFlashAttention` | none (dense) | **yes** (`IsFlashDecode` + `SplitBNS`) | `service_vector_flashdecode_mla.h` | `MIX_AIC_1_2`, `1_1` for C1V1 keys | dense decode reference |
| `FusedSparseAttentionOverlap` (vllm-ascend) | `selectionTopkIndices` + Selection KV Cache | **no** (`FLASH_DECODE=0` in its template list) | n/a | n/a (vllm-ascend tree) | removes the gather round trip; keeps the core limit |

### 15.3 `SparseAttnSharedkv` — A3-exclusive, metadata-driven, but not our shape

`experimental/attention/sparse_attn_sharedkv/README.md`:

> | 产品                                                         | 是否支持 |
> |<term>Ascend 950PR&950DT系列产品</term>                        | ×  |
> |<term>Atlas A3系列产品</term>        | √  |
> |<term>Atlas A2系列产品</term>        | ×  |

> - API功能：`SparseAttnSharedKV`算子旨在完成以下公式描述的Attention计算，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。

> - 该接口当前支持三种计算场景：场景一，仅传入`ori_kv`时为Sliding Window Attention计算；场景二，传入`ori_kv`及`cmp_kv`时为Sliding Window Attention + Compressed Attention计算；场景三，传入`ori_kv`、`cmp_kv`及`cmp_sparse_indices`时为Sliding Window Attention + Sparse Compressed Attention计算。

> - 当`layout_q`为TND时，功能使用限制如下：
>   - `q`的shape需要为[T1,N1,D]，其中N1仅支持64。
>   - `ori_sparse_indices`的shape需要为[Q\_T, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的token数，K1仅支持512。
>   - `cmp_sparse_indices`的shape需要为[Q\_T, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数，K2仅支持512。

It takes a `metadata` input in the section-11 style (`op_host/sparse_attn_sharedkv_def.cpp:88`
`this->Input("metadata")`, validated against a fixed `METADATA_LIMIT`), and it is the only
attention op in the repo marked A3-only. But `N1仅支持64` means 64 Q heads per rank — TP1, not
TP16 — and `K1仅支持512` means top-512, not top-2048. Wrong shape for us on both axes.

### 15.4 `SparseFlashAttention`'s own test matrix has no single-query decode case

`attention/sparse_flash_attention/tests/pytest/sparse_flash_attention_paramset.py`. Five cases:
`bsnd_basic`, `bsnd_multi_batch`, `pa_bsnd`, `tnd_basic`, `tnd_pa_multi_batch`. Their shapes:

| case | B | S1 | S2 | N1 | K (top-k) | `sparse_block_size` |
| --- | --- | --- | --- | --- | --- | --- |
| `bsnd_basic` | 1 | **5** | 262144 | 8 | 16 | 1 |
| `bsnd_multi_batch` | 32 | **6** | 512 | 64 | 32 | 1 |
| `pa_bsnd` | 2 | **8192** | 1024 | 16 | 2048 | 1 |
| `tnd_basic` | 2 | **4** | 3072 | 8 | 32 | 1 |
| `tnd_pa_multi_batch` | 4 | **5** | 3072 | 64 | 64 | 1 |

with e.g. `bsnd_basic` verbatim:

```python
    "bsnd_basic": {
        "Testcase_Prefix": ["sfa_bsnd_basic"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [1],
        "S1": [5],
        "S2": [262144],
        "N1": [8],
        "N2": [1],
        "D": [512],
        "K": [16],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [0],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [False],
        "actual_seq_q": [[4]],
        "actual_seq_kv": [[4]],
        "range_query": [[-10.0, 100.0]],
        "range_key": [[5.0, 100.0]],
    },
```

**The minimum `S1` in the entire matrix is 4.** Not one case has `S1 = 1`. The only case with
`K = 2048` is `pa_bsnd`, whose `S1` is 8192 — a prefill. The op has no public test for a pure
single-query decode step.

## Relevance

Two conclusions for our 10.8528 ms.

**The 13.58 us MIX_AIC launch floor is not avoidable by picking a different operator.** Every
shipped A3 decode-attention kernel declares `KERNEL_TYPE_MIX_AIC_1_*`. There is no AIV-only
attention decode path, so our 505 MIX_AIC calls per round and the 11.5 ms launch floor they imply
cannot be reduced by operator substitution — only by reducing the *number* of MIX launches. At
6.5 MIX_AIC calls per layer, SFA is one of them; the launch floor alone is 13.58 us of the
89.4 us we pay per SFA call, i.e. **15% of the op's cost before any byte moves**. This is the
clearest argument in this whole document for the megakernel over per-op optimization: the
launch floor is a property of the dispatch, and the only way past it is fewer dispatches.

**The missing decode tiling is explained, not just observed.** Section 12 showed the splitter was
dropped when SFA was forked from FIA. Section 15.4 shows why nobody noticed: the op's test matrix
has no `S1 = 1` case, so the `totalBaseNum = 1` collapse in `InitCalcParamsEach` is never
exercised. The nearest case, `tnd_basic` with `S1 = 4`, happens to produce `usedCoreNum = 4` —
non-degenerate enough to pass, degenerate enough to hide the bug. And it is exactly the shape
vLLM-Ascend ships, because MTP-3 makes `T1 = 4`. Our configuration sits precisely on the only
decode-ish shape the op was validated against, and that shape uses 4 of 24 cores.

The practical consequence: do not expect an upstream fix to arrive on its own, and do not expect
`KvQuantSparseFlashAttention` or `NsaSelectedAttentionInfer` to behave differently — both have
zero `splitKVFlag_ = true` sites. The two routes that change the parallelism are
`SparseFlashMla` + `SparseFlashMlaMetadata` (if the A3 `仅支持SWA模板` restriction can be made to
express a pure top-k pattern) and our own megakernel using the `split_core.h` cost model
`6*ceil(M/16) + 10*ceil(S2/64)` and the `SparseFlashMlaMetadata` task-list schema.

---

## 16. DMA instruction accounting for a token-wise sparse gather on A3: `DataCopyPA` can only cross one page per instruction

Same clone and commit (`gitcode.com/cann/ops-transformer`,
`3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20).
`attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_cube_mla.h`.

### 16.1 `DataCopyPA` — the block-table gather, verbatim

```cpp
/*
    适用PA数据从GM拷贝到L1，支持ND、NZ数据；
    PA的layout分 BNBD（blockNum,N,blockSize,D） BBH（blockNum,blockSize,N*D
    BSH\BSND\TND 为BBH
    shape.copyRowNumAlign 需要16字节对齐，如拷贝k矩阵，一次拷贝128*512，遇到尾块 10*512 需对齐到16*512
*/
template <typename T, SFA_LAYOUT SRC_LAYOUT>
__aicore__ inline void DataCopyPA(LocalTensor<T>& dstTensor,  // l1
                                  GlobalTensor<T>& srcTensor, // gm
                                  GlobalTensor<int32_t>& blockTableGm,
                                  const PAShape& shape,     // blockSize, headNum, headDim
                                  const Position& startPos) // bacthIdx nIdx curSeqIdx
{
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;  // 获取在单个块上超出的行数
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // 获取block table上的索引
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
        // 计算可以拷贝行数
        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt; // 一次只能处理一个Block
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt; // 一个block未拷满
        }
        uint64_t dStride = shape.headDim;
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim; // PA的偏移

        if constexpr (SRC_LAYOUT == SFA_LAYOUT::BSND || SRC_LAYOUT == SFA_LAYOUT::TND) {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) + reaminRowCnt * shape.headDim * shape.headNum +
                      startPos.dIdx;
            dStride = shape.headDim * shape.headNum;
        } else {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + reaminRowCnt * shape.headDim +
                      startPos.dIdx;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}
```

`uint32_t copyRowCnt = shape.blockSize - reaminRowCnt; // 一次只能处理一个Block` — "one
instruction can only handle one Block". Each loop iteration is one `blockTableGm.GetValue()`
scalar GM read plus one `DataCopy` (Nd2Nz).

The underlying move, also verbatim:

```cpp
// 场景：query、queryRope、key、value GM to L1
// GM按ND格式存储
// L1按NZ格式存储
// GM的行、列、列的stride
template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T>& l1Tensor, GlobalTensor<T>& gmTensor, uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams sfaCubeCopyParams;
    sfaCubeCopyParams.ndNum = 1;
    sfaCubeCopyParams.nValue = rowAct; // nd矩阵的行数
    // T为int4场景下，dValue = col / 2，srcDValue = colStride / 2
    sfaCubeCopyParams.dValue = col;          // nd矩阵的列数
    sfaCubeCopyParams.srcDValue = colStride; // 同一nd矩阵相邻行起始地址间的偏移
    sfaCubeCopyParams.dstNzC0Stride = rowAlign;
    sfaCubeCopyParams.dstNzNStride = 1;
    sfaCubeCopyParams.srcNdMatrixStride = 0;
    sfaCubeCopyParams.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, sfaCubeCopyParams);
}
```

and the non-PA variant the `C_TEMPLATE` path uses:

```cpp
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor,
                                                              const uint64_t keyGmBaseOffset,
                                                              uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                              uint32_t nActCopyRowCount, uint32_t headSize)
{
    uint64_t dStride = sfaCubeConstInfo.headDim;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
        dStride = sfaCubeConstInfo.headDim * sfaCubeConstInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
}
```

Note `ndNum = 1` in every one of these: the Nd2Nz path moves a single ND matrix per instruction,
so there is no multi-descriptor batching available to the cube's gather.

### 16.2 The instruction count, per (layer, head-group), at our configuration

GLM-5.2 at TP16 seq 8K: `top_k = 2048`, `sparse_block_size = 1`, PA `block_size = 128`,
`headDim = 512`, `headDimRope = 64`, bf16.

`V_TEMPLATE` (what we run — `TEMPLATE_MODE=V_TEMPLATE` per section 14.2):

| stage | instruction | count | bytes per instruction |
| --- | --- | --- | --- |
| index read | `topkGm_.GetValue` (scalar GM) | 2048 | 4 |
| block table read | `blkTableGm_.GetValue` (scalar GM) | 2048 | 4 |
| nope gather (MTE2) | `DataCopyPad`, `blockCount=2` | 1024 | 2 x 1024 = 2 KiB |
| rope gather (MTE2) | `DataCopyPad`, `blockCount=2` | 1024 | 2 x 128 = 256 B |
| nope flush (MTE3) | `DataCopyPad`, `blockCount<=32` | 64 | <=32 x 1024 = 32 KiB |
| rope flush (MTE3) | `DataCopyPad`, `blockCount<=32` | 64 | <=32 x 128 = 4 KiB |
| cube reload (MTE2) | `DataCopy` Nd2Nz from `kvMergeGm_` | ~8 | contiguous 128 x 256 |

`>= 4096` scalar GM reads and `>= 2176` DMA instructions per layer per head-group, of which 1024
are 256-byte rope bursts. The rope stream alone is half the instruction count for 1/8 of the
bytes.

`C_TEMPLATE` (`sparse_block_size > 4`), for comparison — the cube gathers directly and
`DataCopyPA`'s `一次只能处理一个Block` bound applies. With `sparse_block_size = B` and PA
`block_size = 128`, a selected run of `B` consecutive tokens lies inside one page whenever
`B | 128` and the run is page-aligned, which the op guarantees via
`blockSize_ % sparseBlockSize_ > 0` being rejected (section 8.6). So the count is
`top_k_blocks = 2048 / B` block-table reads and the same number of `DataCopy` instructions,
with `B x 1024` bytes each — and **no** GM staging, no flush, no reload:

| `sparse_block_size` | selected blocks | MTE2 instructions | bytes/instruction (nope) | GM staging |
| --- | --- | --- | --- | --- |
| 1 (`V_TEMPLATE`) | 2048 | 1024 pairs + 1024 rope + flush + reload | 2 KiB / 256 B | **yes, 3 transits** |
| 16 (`C_TEMPLATE`) | 128 | 128 | 16 KiB | no |
| 32 (`C_TEMPLATE`) | 64 | 64 | 32 KiB | no |
| 128 (`C_TEMPLATE`) | 16 | 16 | 128 KiB | no |

### 16.3 The block-wise lever is documented and currently unused

`attention/sparse_flash_attention/README.md`:

> | sparse_block_size | 可选属性 | 代表sparse阶段的block大小。sparse_block_size为1时，为Token-wise稀疏化场景；sparse_block_size大于1且小于等于128时，为Block-wise稀疏化场景，块内token共享相同的稀疏化决策。默认值为1。 | INT64 | - |

> - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
>   - 参数sparse_block_size支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparse_block_size整除block_size。

A3 and A2 accept any power of two in [1,128]; the 950 series accepts only 1
(`sparse_block_size仅支持1`, and the tiling rejects anything else on `DAV_3510`, section 8.6).
vLLM-Ascend ships `sparse_block_size=1` (section 14.4), and
`FusedSparseAttentionOverlap` is explicit that it cannot do otherwise:

> | sparseBlockSize | 属性 | 稀疏选择粒度。当前融合路径仅支持Token-wise稀疏化，取值为1。 | INT64 | - |
> | selectionTopkBlockSize | 属性 | TopK索引的选择粒度，当前仅支持1。 | INT64 | - |

The direction of travel in the model family is toward block granularity. `zai-org/GLM-5.3-Flash`
(https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json, fetched 2026-10-20, 29464
bytes, nested `text_config`) adds indexer-side pooling: `"index_kpool": 4`,
`"index_kpool_compress": true`, `"index_kpool_always_select_tail": true`, with
`"qk_rope_head_dim": 0` and `"mla_use_nope": true` — which by the Ascend kernel's own formula
`KV_D = 512 + rope_head_dim*2 + 4*4` makes the C8 row **528 bytes** instead of 656. Its
`layer_types` is 45 entries = 34 `linear_attention` + 11 `deepseek_sparse_attention`.

## Relevance

This is the quantitative answer to candidate (b), and it identifies a lever we control.

**The per-instruction burst lengths are tiny and the instruction count is enormous.** At our
configuration, one `SparseFlashAttention` call per layer issues `>= 4096` scalar GM reads (the
top-k index plus the block-table entry for each of 2048 tokens, each on the AIV scalar pipe and
each in the dependency chain ahead of the DMA that consumes it) and `>= 2176` DMA instructions,
half of which move 256 bytes. Against a measured 89.4 us per call, 2176 DMA instructions is
41 ns per instruction — which is in the right order for an MTE issue slot, meaning the kernel is
plausibly **instruction-issue bound rather than bandwidth bound**. That is a different failure
mode from "scattered reads miss L2", and it predicts a specific profile signature: `aiv_mte2_time`
high relative to bytes moved, `aic_mte2_time` low, `aic_mac_ratio` low, and the scalar pipe busy.
If our Level-2 profile shows that pattern, candidate (b) is confirmed in its instruction-count
form rather than its burst-efficiency form, and the fix is to reduce instruction count, not to
improve locality.

**Raising `sparse_block_size` is the single cheapest experiment available.** It is an existing
attribute of the op we already call, accepted on A3 for any power of two up to 128, and it does
three things at once: it cuts MTE2 instruction count by `B`, it raises burst length to
`B x 1024` bytes, and — because `sparse_block_size > 4` switches `perfMode_` to `C_TEMPLATE`
(section 8.6) — **it eliminates the `kvMergeGm_` HBM round trip entirely**, replacing it with
`DataCopyPA` straight into L1. That is the ~2.8 ms staging cost of section 9 removed by changing
one integer. At `sparse_block_size = 16`: 128 instructions of 16 KiB instead of 2048 of 1-2 KiB,
and one HBM transit instead of three.

This is verified, not inferred. `MergeKv` — the whole gather-and-stage routine of section 9 — has
exactly one call site, and it is guarded:
`attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h:909-911`:

```cpp
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                ...
                sfaKernelVectorService.MergeKv(extraInfo0);
```

`grep -n 'MergeKv' op_kernel/arch22/` returns the declaration
(`service_vector_mla.h:66`), the definition (`service_vector_mla.h:1028`) and that single guarded
call. Under `C_TEMPLATE` the routine is never entered, `kvMergeGm_` is never written, and the
cube's `else` branch in `ComputeMm1NoRope` (section 9.1) walks the top-k list straight into L1.
The V_TEMPLATE-only validity bookkeeping goes with it — `ElewiseCompute` reads
`v0ValidSizeUb_.GetValue(...)` only under the same guard
(`service_vector_mla.h:442`).

The cost is accuracy, and it is not free: `块内token共享相同的稀疏化决策` means the indexer must
select whole 16-token blocks. GLM-5.2's indexer emits per-token top-2048, so using
`sparse_block_size = 16` requires either pooling the index scores over 16-token blocks before the
top-k (which is exactly `index_kpool` in GLM-5.3-Flash, so the model family considers it
acceptable) or taking `top_k/16` blocks and accepting a different selected set. The accuracy delta
is a measurement we can make cheaply, and the performance upside is the largest single item found
in this investigation that does not require writing a kernel.

And the template is compiled. The sixth `ASCENDC_TPL_ARGS_SEL` entry of section 8.1 is
`FLASH_DECODE 0, PAGE_ATTENTION 1, LAYOUT_T TND, KV_LAYOUT_T {TND, PA_BSND}, TEMPLATE_MODE
C_TEMPLATE, IS_SPLIT_G {0,1}, HAS_ROPE {0,1}` — exactly our layout combination with `C_TEMPLATE`
instead of `V_TEMPLATE`. No rebuild of CANN is required to try it; `sparse_block_size = 8` or
higher selects it through `perfMode_` at tiling time.

**One caveat that constrains the lever.** `sparse_block_size` must divide the PageAttention
`block_size`, which vLLM-Ascend fixes at 128 for the SFA path
(`get_supported_kernel_block_sizes` returns `[128]`, section 13.3), so `B` in {1,2,4,8,16,32,64,128}
all satisfy the divisibility constraint. But the 950 series rejects everything except 1, so a
block-wise path is A3/A2-only and would not forward-port to A5.

---

## 17. Why discrete gather is structurally expensive on A3: there is no SIMT unit, and CANN says so

https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/develop/cross_platform_migration_guide.md
CANN `ops-transformer`, master, commit `3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20.

### 17.1 The SIMT unit is a 950-series addition

> #### SIMT
>
> Ascend 950系列新增了SIMT单元。SIMT在处理非规整离散访问方面相比SIMD有较大优势，适合地址不连续、访存跨度变化大、分支路径不一致的场景（如scatter/gather、索引重排、稀疏更新等）。
>
> 迁移时建议优先识别"访存主导"且"向量化效率低"的算子子流程：若原有SIMD实现存在大量掩码分支、无效lane占比高、或需要复杂地址拼装，可将该部分改写为SIMT路径，通常可降低控制开销并提升有效访存吞吐。
>
> 实践中需重点关注以下事项：一是线程任务切分要与数据稀疏性匹配，避免线程负载极不均衡；二是减少高频随机访存导致的流水空转，尽量在上游完成索引规整与分桶；三是将边界处理与主路径解耦，避免在热点循环中引入过多分支。建议在迁移后同时对比"纯SIMD实现"和"SIMD+SIMT混合实现"，按数据分布选择最优策略，而非固定单一路径。

### 17.2 The documented SIMD-vs-SIMT split, with a measured template threshold

> **以gather_v2算子为例：SIMD与SIMT实现对比**
>
> gather_v2算子根据合轴后的尾轴为单位进行gather，因此模板选择的依据是：尾轴 ≤ 2048时走SIMT模板，尾轴 > 2048时走SIMD模板。因为尾轴小时，需要离散访问多个不连续的小块地址，simt效率高。以下对比两种实现的核心差异：

> SIMT采用线程级并行模型，每个线程独立处理元素：
>
> ```
> // SIMT: 使用线程级并行，无需显式buffer管理
> __simt_vf__ LAUNCH_BOUND(2048) void GatherSimt(...) {
>     for (INDEX_SIZE_T index = Simt::GetThreadIdx();
>          index < currentCoreElements;
>          index += Simt::GetThreadNum()) {  // 线程跳跃式并行
>         // 每个线程独立计算单点索引并访问
>         INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);
>         INDICES_T indicesValue = indices[gatherI];  // 根据单点索引gatherI直接访问GM
>         y[yIndex] = idxOutOfBound ? 0 : x[xIndex];  // 直接写回GM
>     }
> }
> ```

> **2. 访存模式差异**
>
> | 特性 | SIMD实现 | SIMT实现 |
> | ------ | ---------- | ---------- |
> | 数据访问 | 通过DataCopyPad显式搬运到UB | 线程直接通过`__gm__`指针访问GM |
> | Buffer管理 | 需AllocTensor/EnQue/DeQue/FreeTensor | 无需显式buffer，硬件自动管理 |
> | 同步机制 | 显式事件同步（HardEvent::MTE2_V等） | 线程间隐式同步 |
>
> **3. 适用场景差异**
>
> SIMD适合连续访问大块地址的场景，通过向量化指令高效处理连续数据；
>
> SIMT适合离散访存，线程并行处理；

> | SIMT | SIMT引入后可用线程级并行处理分支/不规则计算，但需要适配线程划分、共享内存与同步语义；部分Vector实现可迁移为SIMT版本 |

### 17.3 Regbase, also 950-only

> #### Regbase
>
> Ascend 950系列引入了Regbase编程范式，相比传统的Membase（Vector API）编程，Regbase更接近底层硬件的寄存器操作，提供更精细的向量化控制能力。
>
> **特点**
>
> - 使用`AscendC::Reg`命名空间下的底层API
> - 直接操作寄存器`RegTensor<T>`而非显式管理UB缓冲队列
> - 通过`MaskReg`实现灵活的元素级掩码控制
>
> > 注：`MicroAPI` 为 `Reg` 的历史别名（CANN 头文件中定义 `namespace MicroAPI = Reg;`，仅在部分新架构生效）。新代码统一使用 `AscendC::Reg` 命名空间，本仓已全量完成 MicroAPI→Reg 整改。

This is consistent with the operator tree: `SparseFlashAttention`'s `arch35` directory contains
`op_kernel/arch35/common/util_regbase.h` and `arch22` has no Regbase equivalent.

## Relevance

This is the structural reason candidate (b) is real on A3 and cannot be engineered away inside the
existing programming model, and it reframes the whole investigation's conclusion.

**A3 has no hardware gather.** `Ascend 950系列新增了SIMT单元` — the SIMT unit that is "适合地址
不连续、访存跨度变化大" (suited to non-contiguous addresses and varying access strides) and whose
listed use cases are literally `scatter/gather、索引重排、稀疏更新` is a **950-series addition**.
A3 (`__NPU_ARCH__ == 2201`) is SIMD/Membase only. Therefore on A3 a scattered KV gather *must* be
expressed as what section 16 counted: many explicit `DataCopyPad` instructions with
`显式事件同步（HardEvent::MTE2_V等）`, each carrying per-instruction issue cost, with the
addresses computed by scalar GM reads on the critical path. There is no thread-per-element path.

**The vendor's own template threshold is a calibration point we can use.** `gather_v2` picks SIMT
when the trailing axis is `<= 2048` elements and SIMD when `> 2048`, because `尾轴小时，需要离散
访问多个不连续的小块地址，simt效率高`. Our per-token gather moves 512 bf16 elements (1024 bytes)
per selected token — firmly inside the "small trailing axis, discrete access to many small
non-contiguous addresses" regime where CANN's own operator library judges SIMD to be the wrong
tool. On A3 that is the only tool available.

**So the three candidate causes resolve as follows**, and the ranking is not what the brief's
framing implied:

- (c) **insufficient parallelism** is the largest and the most fixable: 4 of 24 cube cores, caused
  by a dropped `SplitCore` call (sections 8, 10, 12), with two reference implementations in the
  same repo to copy.
- (b) **gather inefficiency** is real and structural, but its mechanism is *instruction count and
  scalar address generation*, not L2 miss or burst length: `>= 4096` scalar GM reads and
  `>= 2176` DMA instructions per layer per head-group (section 16), on a chip with no SIMT unit
  (this section). Its cheapest mitigation is `sparse_block_size > 4`, which both cuts the
  instruction count by `B` and eliminates the `kvMergeGm_` round trip.
- (d) **tail effects from prefill tiling** are present and measurable but second-order:
  `sInnerSize_ = 512` fixed, `M_SPLIT_SIZE = 128` against `M = 4`, zero-padding to 512 rows one
  `DataCopyPad` at a time, and `PipeBarrier<PIPE_M>()` on every Mmad because
  `(16/16)*(128/16) = 8 < 10`.
- (a) **launch/head overhead** is a flat 13.58 us per call = 15% of the op, unavoidable by
  operator substitution because every A3 decode attention kernel is `MIX_AIC` (section 15), and
  removable only by issuing fewer kernels.

**The synthesis.** Our 6.973 ms is not one problem. It is a dropped core-splitter (fixable by
calling a different operator or owning the kernel), an HBM staging round trip (fixable by one
attribute change), a per-instruction gather cost that is architectural on A3 (mitigable by block
granularity, properly fixed only on 950), and a launch floor that only a megakernel removes. Of
these, the two that need no new kernel are: set `sparse_block_size` to 8 or more so
`perfMode_` picks `C_TEMPLATE`, and evaluate `SparseFlashMla` + `SparseFlashMlaMetadata` for a
real S2 split. Both are configuration changes against already-compiled templates.

---

## 18. CONTRADICTION: small-batch sparse decode is a named family on A3, and flash-decode splitting was quantitatively rejected there

This section exists because it contradicts the recommendation of sections 8, 10 and 12. Read it
before acting on them.

### 18.1 The named bottleneck family, and the rejected fix

https://github.com/tile-ai/tilelang-ascend/pull/1741
Nance-Mike, opened 2026-09-04, merged 2026-09-24 by LLMZhangYC. Test platform A3
(Ascend910_9362), CANN 9.1.0.

> **仍慢于 baseline 的 case 族及根因**（逐族定量分析见 `bottleneck_analysis.md`）：
>
> | 族 | 现状 | 根因 |
> | ---- | ---- | ---- |
> | B 族大 shape（cases 1-5、20） | 0.07-0.12x | dense kernel 本身已 4.4× 于 gather 路径，但 bitmap scatter 构建开销（~450ns/写）+ 每 (block,n) ~3.4μs 的 CV 握手串行吞掉大半收益 |
> | MLA 族（cases 9-13） | 0.23-0.33x | CV relay 全口径流量约为 gather 的 7×（ws 暂存 + acc_gm 往返），聚合带宽 ~1TB/s 饱和；M=32 小 tile staging 次因 |
> | 小批量 decode（cases 14、16、18） | 0.45-0.71x | 块数 16-32 ≈ 核数 20，单块 relay 即 makespan，flash-decoding 拆分经定量推演为负收益 |
>
> R7 四项候选（gather 流水化 GM 直读 / chunk 暂存 + 两项分析）已全部经测量/分析否决，operator 级优化空间已收敛，剩余主要杠杆在框架侧。

Per-case results from the same PR:

> | level4/sparse_flash_attention_9  | ✅ | 448.30 | 0.26x | 0.000122 |
> | level4/sparse_flash_attention_13 | ✅ | 447.92 | 0.23x | 0.000061 |
> | level4/sparse_flash_attention_14 | ✅ |  68.04 | 0.71x | 0.000061 |
> | level4/sparse_flash_attention_16 | ✅ | 128.86 | 0.46x | 0.000122 |
> | level4/sparse_flash_attention_18 | ✅ | 130.16 | 0.45x | 0.000061 |

> | 指标 | baseline | R6 最终版 | 变化 |
> | 综合得分 | 53.37 | **66.00** | +12.63 |
> | 平均加速比 | 0.09x | **0.82x** | 9.1x |

Two statements here cut against sections 8-12:

1. `小批量 decode ... 根因 ... 块数 16-32 ≈ 核数 20，单块 relay 即 makespan，flash-decoding 拆分经定量推演为负收益`
   — "block count 16-32 ≈ core count 20, a single block's relay is the makespan, **flash-decoding
   splitting was quantitatively deduced to be net-negative**".
2. `MLA 族 ... CV relay 全口径流量约为 gather 的 7×（ws 暂存 + acc_gm 往返），聚合带宽 ~1TB/s 饱和`
   — the CV relay path moves **7x** the bytes of the gather path, and 1 TB/s aggregate bandwidth
   is already saturated.

### 18.2 The public per-call number for our exact shape

https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/sparse_flash_attention/bench_sfa/README_zh.md
tile-ai/tilelang-ascend, branch `ascendc_pto`, current as of 2026-10-20. Measured on A3.

> ## 📊测试输入
> 输入定义在 bench_sfa.py 中，公共参数如下：
> - T=1
> - B=1
> - Q_N=128
> - KV_N=1
> - D=512
> - D_rope=64
> - sparse_size=2048
> - block_size=128
> - act_kv_s=2560
>
> 三组 shape 只在 KV_S 上不同：
> | 编号 | KV_S | 说明 |
> | shape0 | 2560 | 短序列 |
> | shape1 | 6400 | 中序列 |
> | shape2 | 48000 | 长序列 |
>
> act_kv_s 固定为 2560，sparse_size 固定为 2048，实际参与计算的有效稀疏窗口规模基本不变。
>
> ## 性能结果
> |  | shape0 | shape1 | shape2 | 平均耗时(us) | AscendC / TileLang |
> | AscendC  | 98.000 | 99.000 | 100.000 | 99.000 | 1.00x |
> | TileLang | 109.760 | 109.100 | 108.980 | 109.280 | 0.91x |
>
> *注： AscendC 参考实现来源：* https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/torch_ops_extension/custom_ops/converter/npu_sparse_flash_attention.py

and the optimization ladder at that same B=1 shape:

> |[sparse_flash_attn_pa_baseline.py]            | √ | 64  | × | × | × | × | × | × | 602us |
> |[sparse_flash_attn_pa_developer.py]           | √ | 64  | √ | × | × | × | √ | × | 347us |
> |[sparse_flash_attn_pa.py]                     | √ | 64  | √ | √ | √ | √ | √ | √ | 127us |
> |[sparse_flash_attn_pa_no_cv_pipeline.py]      | √ | 256 | √ | √ | √ | √ | √ | √ | 109us |

### 18.3 The other published batch-1 sparse decode numbers

https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/deepseek_v4/sparse_flash_mla/bench_mark.md
tile-ai/tilelang-ascend, branch `ascendc_pto`, retrieved 2026-10-20.

> | B | 1 | Batch size |
> | N1 | 64 | Query heads |
> | N2 | 1 | KV heads (so 64 query heads share one KV head) |
> | D | 512 | Head dimension |
> | layout | TND | Query layout |
> | K | 512 | Top-k compressed indices (`csa` / `hca`; `swa` uses none) |
> | S (decode) | 1 | One query step against a KV cache of 8193 |
>
> Measurement: `msprof` **device Task Duration** from `op_summary`, first launch dropped
> (cold start) and the rest averaged. Both sides run the same shapes and the same dtype.
>
> | Scenario | AscendC | tileLang | Performance Ratio (AscendC/tileLang) |
> | swa prefill | 1850.954u | 1490.930u | 124.1% |
> | hca prefill | 3005.296u | 3141.853u | 95.7% |
> | csa prefill | 6882.407u | 6022.700u | 114.3% |
> | swa decode | 23.916u | 13.824u | 173.0% |
> | hca decode | 24.007u | 18.644u | 128.8% |
> | csa decode | 38.759u | 31.659u | 122.4% |

### 18.4 The launch-side component, measured and bounded

https://github.com/vllm-project/vllm-ascend/pull/17863
LiPu-jpg, 2026-10-01, state OPEN (draft). Physical A2: Ascend 910B3, 20 AIC / 40 AIV, CANN 9.1.0.

> A2/A3 NoPE SFA assigns one query group per task, but launches and reserves scratch for every hardware core. A one-row TND query on the tested A2 launches 20 AIC / 40 AIV despite having one query group. Bound the launch by query tensor rows and allocate scratch for that launch.

> - Actual canonical host hashes and profiler counts were checked: q1 = 1 AIC / 2 AIV; q32 = 20 AIC / 40 AIV.
> - **Public latency, all 40 frozen-input cases:** B/C/C/B has 18 lower / 22 higher, maximum increase 1.7883%; reverse C/B/B/C has 30 lower / 10 higher, maximum increase 1.2230%. ... Query counts 1/2/4/8 have all 16 cases lower in both orders, reductions 4.6473–7.8103% across those rounds. Count 16 is mixed in forward and lower in reverse
> - **Memory, all 40 cases in separate original / launch-only / combined processes:** eager and graph peak-increase reductions at q1/2/4/8/16 respectively 57,796,096 / 65,135,104 / 58,818,560 / 44,113,920 / 14,704,640 bytes. At q20/21/24/25/32, peak increase is unchanged. ... Single-query reduction is 55.1187 MiB
> - **Final expanded performance, all 364 cases:** 234 lower / 130 higher, maximum increase 3.8763% in a RoPE64 control outside the guard. All 112 boundary cases at q2/4/9/10/11/12/13 are lower; q16/19 retain increases up to 1.6975% inside the bound.

### 18.5 The padded-capacity waste, independently reported

https://github.com/vllm-project/vllm-ascend/pull/17653
ShuZihan, 2026-09-28, state OPEN (draft).

> Sparse attention can iterate over padded capacity even when only a prefix of the selected indices is valid, as in DCP decode. This PR detects that prefix and shortens the computation loop automatically in C8 and non-C8 V-template kernels with `sparse_mode=0` and `sparse_block_size=1`. Empty rows and other paths keep their existing behavior.
> ...
> NPU correctness/graph tests and an A/B benchmark are included. A3 compilation, device tests and model/performance validation are pending; keep this PR as draft.

## Relevance

Three things, two of which revise conclusions above.

**CONFIRMED, exactly: the full die is launched for one query row.** PR #17863's
`A one-row TND query on the tested A2 launches 20 AIC / 40 AIV despite having one query group`
is section 8.2 and 8.3 observed from outside: `usedCoreNum_ = aicNum_` in `SplitBalanced` sets
`blockDim` to the whole die, and then `InitCalcParamsEach` gives work to `totalBaseNum` of them
and early-`return`s the rest. The measured post-fix counts `q1 = 1 AIC / 2 AIV; q32 = 20 AIC /
40 AIV` are precisely `usedCoreNum = min(T1, coreNum)`.

**REVISED: the launch-side component is only 4.6-7.8%.** PR #17863 removes the idle launches and
reports `Query counts 1/2/4/8 have all 16 cases lower in both orders, reductions 4.6473–7.8103%`.
So *not launching* 16-19 dead cores is worth under 8%. That bounds candidate (a) tightly and says
the idle cores are cheap to leave idle — the expensive thing is that the working cores have all
the work. It also hands us a free 55.1 MiB: `Single-query reduction is 55.1187 MiB` of scratch
reserved for cores that do nothing.

**CONTRADICTION, and it is the important one: flash-decode splitting has already been evaluated
for small-batch sparse decode on A3 and rejected.** `小批量 decode ... flash-decoding 拆分经定量
推演为负收益`. Sections 8, 10 and 12 all conclude that the fix is to split S2 across cores and
combine with an lse reduction. PR #1741 says that was quantitatively analysed and found
net-negative. The reconciliation matters, so state it precisely rather than dismissing it:

- Their regime is `块数 16-32 ≈ 核数 20` — they already had 16-32 parallel blocks against 20
  cores, so the machine was already full and splitting further only adds reduction cost. Ours is
  `T1 = 4` blocks against 24 cores (section 14), so we have 20 idle cores and they had none.
  Their negative result is a statement about a saturated machine, not about splitting per se.
- But their second row is a direct warning about the *cost* side of the split:
  `MLA 族 ... CV relay 全口径流量约为 gather 的 7×（ws 暂存 + acc_gm 往返），聚合带宽 ~1TB/s 饱和`.
  A split-KV combine adds exactly those `ws 暂存 + acc_gm 往返` round trips — the `accumOut` and
  `logSumExp` workspace of section 8.1. They measured that path at **7x** the gather path's bytes
  and found 1 TB/s already saturated. Our section 9.1 estimated the existing `kvMergeGm_` staging
  at 3x; a flash-decode combine on top would add more.

So the honest revision is: splitting S2 across the 20 idle cores is not free, and the published
analysis of the closest regime says the combine traffic can exceed the parallelism gain. The
split is only clearly net-positive if the combine stays on-chip — which is possible only inside a
megakernel, where partial `(out, max, sum)` can live in UB/L1 across core groups instead of
round-tripping `acc_gm`. That is now the second independent argument for the megakernel as the
enabling vehicle rather than an optimization, and it arrives from a measurement rather than from
our reasoning.

**CONTRADICTION with our 6.973 ms: we are already at the public state of the art for this
operator.** The reference AscendC `npu_sparse_flash_attention` at **B=1, Q_N=128, KV_N=1, D=512,
D_rope=64, sparse_size=2048, block_size=128** costs **98-100 us per call on A3**. Our measurement
is 6.973 ms over 78 layers = **89.4 us per call** at `Q_N=4`. Two readings, both significant:

- Our per-call cost is *at or slightly below* the published reference despite running 1/32 of the
  query heads. So there is no misuse to find in our call site; 89.4 us is what this operator
  costs for a 2048-token gather, almost independent of head count.
- The published times are 98.000 / 99.000 / 100.000 us for KV_S = 2560 / 6400 / 48000 — **flat
  across an 18.75x change in KV length**, with `act_kv_s` and `sparse_size` held fixed. The cost
  tracks the sparse window, not the sequence. That is the signature of a kernel whose time is set
  by the 2048-token gather and its fixed overheads, and it means our seq-8K figure would not
  improve at seq 2K either.

The corollary is uncomfortable and should be stated plainly: the 5.6 ms we hoped to recover by
"closing `SparseFlashAttention` to roofline" is not recoverable by calling this operator better.
It is recoverable only by not using this operator — block-wise granularity (section 16),
`FusedSparseAttentionOverlap`'s Selection KV Cache (section 14), or our own kernel. The ladder in
18.2 shows what owning the kernel is worth on this exact shape: **602 us → 347 us → 127 us →
109 us**, i.e. 5.5x from baseline to a fully-optimized hand-written version, with the final step
coming from dropping the CV pipeline and widening the tile from 64 to 256.

---

## 19. The mechanism, from the team that wrote the kernel: MTE2 issue-queue depth is 16, and the BIU will not coalesce a source-contiguous / destination-strided gather

https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20260930-ascend-prefill-deep-dive-zh.md
DeepSeek (deepseek-ai/FlashMLA), doc dated 2026-09-30. Target: Ascend 950.

> DeepSeek 于 2026 年 9 月 30 日开源了华为昇腾平台的推理基础设施与核心组件，其中就包括本仓库（deepseek-ai/FlashMLA）中的昇腾平台的稀疏 Attention 的 prefill 与 decoding 算子。在 DeepSeek v4.1 模型的典型工况下，该算子可以在 prefill 时达到 410 TFlops，decoding 时达到 360 TFlops，分别达到了硬件理论性能上限的 95% 与 83%。

### 19.1 Why 1 Cube : 1 Vector is impossible — three named bounds

> 1. **MTE2 Issue Queue Entry bound**。在华为昇腾 950 NPU 上，单个 VECTOR Core 在同一时刻最多只能有 16 个"已发出但还未完成"的 MTE2 拷贝请求。哪怕用上了 gather2 技巧（见下文），一次也只能拷贝 32 个 token，太少了。
>
> 2. **Dequantize bound**。在 decoding 的时候，算子需要读取 FP8 或 FP4 格式的 KV Cache，将其反量化为 bfloat16 格式，再送入 CUBE Core 计算。根据微基准测试（micro-benchmark），使用 1C1V 时，反量化的速度赶不上 CUBE Core 计算的速度，导致反量化成为瓶颈。
>
> 3. **FixPipe bound**。当只使用一个 CUBE Core 和一个 VECTOR Core 的时候，FixPipe 向外搬运的速度只有 128 Byte/cycle/AI Core。此时每个块的 MMA 计算周期为 `MMA_M * B_TOPK * (512+512) / 4096 = B_TOPK * 16`，FixPipe 搬运所需的周期为 `(MMA_M * 512 + MMA_M * B_TOPK) * sizeof(float) / 128 = 1024 + B_TOPK * 2`。... 需要 `B_TOPK * 16 > 1024 + B_TOPK * 2`，得到 `B_TOPK > 73.14`。为了留出足够的余量，`B_TOPK` 需要选取 96 或 128，这一方面使得 UB 空间更加紧张，另一方面也加重了上面提到的 MTE2 Issue Queue Entry Bound 的问题。
>
> 因此，最终我们还是决定为每个 CUBE Core 配备两个 VECTOR Core，以分摊 MTE2 发射、反量化、FixPipe 搬运、softmax 等操作的开销。

### 19.2 The `gather2` trick — identical to what `CopyInKv` does

> 本算子需要从显存中以 token 为粒度读取 KV。倘若我们为每个 KV 都发起一次 MTE2 拷贝请求，那么总拷贝请求数就会过多，超过 MTE2 的 Issue Queue 队列深度（16）。为此，我们使用了一种技巧：设置 MTE2 请求的 `burst_count=2`，`src_stride` 为两个 token 开始地址之差，这样就可以一次性拷贝两个 KV token 了，节约 MTE2 拷贝请求数量并降低发射 overhead。MTE2 拷贝的各个参数使用 SIMD VF 计算得到，避免阻塞 scalar 核心。

### 19.3 Why the in-line ND2NZ shortcut is unavailable, and why the BIU will not coalesce

> KV 在显存中是以 ND 格式存储的，shape 为 `[s_kv, d_qk]`，stride 为 `[d_qk, 1]`。但是，CUBE Core 要求输入的矩阵使用 NZ layout。正常来说，我们会在把数据从显存搬运到 L1 buffer 中时，随路完成 ND2NZ 转换，但我们的 KV 是使用 MTE2 一个一个 token 地"稀疏"的从显存上搬到片上存储上的，无法启用随路的 ND2NZ 转换。
>
> 有一个思路是，在发起 GM -> L1 拷贝时，将一个 token（512 个 bfloat16）视为 32 个 32B 的数据块，这 32 个数据块在显存上是连续的，在 L1 buffer 中则以 `B_TOPK * 32B` 为 stride 排布。但后续实验显示，AI Core 内部的 BIU（负责发送访存指令的单元）只会在**源和目标都连续**的时候才会触发访存合并，而像这里这种"只有源连续，而目的不连续"的情况是无法合并访存的，导致 L2 带宽利用率很低，整体瓶颈位于 L2 带宽。
>
> 所以，唯一的做法就是将 KV 先从显存拷贝到 UB，使用 SIMD VF 将其从 ND Layout 转为 NZ Layout，然后再拷贝到 L1。

### 19.4 The icache ceiling that bounds specialization

> 华为昇腾 NPU 上的 Instruction Cache (icache) 大小相对有限，只有 32K (CUBE Core)、16K (VECTOR Core) 或 8K (SIMD VF)，因此需要注意控制代码的总体积

### 19.5 The same conclusion, independently, from Tencent on 910B2C

https://arxiv.org/abs/2505.14022 (HTML: https://arxiv.org/html/2505.14022v1), section 3
Chenghuan Huang, Zhigeng Xu (WeChat HPC, Tencent), Chong Sun, Chen Li (WeChat Vision, Tencent),
Ziyang Ma (corresponding). arXiv:2505.14022v1 [cs.PF], 20 May 2025. Ascend 910B2C.

> Since the cube core is dedicated to matrix multiplication, we mainly leverage vector cores to accelerate MSDA operations. Specifically, we further explore the capabilities of random memory accesses on vector cores. Results show that only gathering data within UB is natively supported, while gathering or scattering data between UB and GM can only be accomplished by multiple explicit fine-grained data copy operations. However, it is important to note that overlapping memory access latencies through massive thread-level parallelism is infeasible, as the Ascend NPU does not adopt an SIMT (Single Instruction Multiple Thread) design but relies on a limited number of high-performance cube and vector cores. This necessitates top-down random memory access optimizations for compute patterns such as MSDA.

> Table 1: Ascend 910B2C Performance Specifications.
> #Cube 24  #Vec 48  Peak Performance (Cube@FP16) 353  Peak Performance (Vector@FP16) 20
> GM Size (GB) 64  UB Size (KB) 192  GM Bandwidth (GB/s) 1800

> Extensive experiments show that our solution achieves up to 5.9× (forward), 8.9× (backward), and 7.3× (end-to-end training) speedup over the grid sample-based baseline, and 1.9×, 2.4×, and 2.0× acceleration over the latest vendor library, respectively.

### 19.6 The CANN-side recipe for a *dense* decode attention, for contrast

https://arxiv.org/abs/2505.21411 (HTML: https://arxiv.org/html/2505.21411v1), section 4.2.3
Pangu Team, Huawei (pangutech@huawei.com). arXiv:2505.21411v1 [cs.CL], 27 May 2025.

> With increasing concurrency levels and iterative expansion of sequence lengths, the key-value (KV) cache exhibits linear growth in memory footprint, causing the latency of attention operations to rise to 30%-50% of total inference time. Consequently, the attention module emerges as a key bottleneck in MoE inference.
>
> Profiling results reveal that KV vector data transfer accounts for approximately 70% of attention computation time, followed by vector operations and matrix multiplications. Therefore, optimizing both memory access bottlenecks and vector computation constraints within operator pipelines has become a critical challenge in improving the efficiency of attention mechanisms.
>
> To address this challenge, we propose MulAttention, a fused attention operator optimized for Ascend hardware, specifically designed for the decoding stage in LLM inference. Specifically, MulAttention improves memory bandwidth utilization through a large-packet KV transfer strategy, as illustrated in steps (1) and (5) in Figure 5. Leveraging the MTE2 transfer unit, KV vectors are block-loaded from global memory (GM) into the Unified Buffer (UB) of the vector computation unit, where a NZ layout transpose is performed simultaneously.
>
> Furthermore, we propose a dual-loop pipeline with a pingpong scheduler for KV processing. By decoupling operations with distinct compute patterns, specifically Cube and Vector operations, into separate loops, we eliminate pipeline bubbles caused by interleaved execution of Key, softmax, and Value computations... In addition, the use of a ping-pong buffering mechanism allows for the overlapping of KV data prefetching and computation. This overlap effectively hides memory latency and increases the utilization of the MTE2 pipeline to over 89%. Through these optimizations, MulAttention achieves a 4.5× end-to-end attention speedup, and significantly enhances hardware utilization.

> In the decode stage dominated by memory-intensive operations, Pangu Pro MoE maintains low latency within several tens of milliseconds for small batch sizes such as a single batch.

### 19.7 The same recipe, as applied in the public tilelang-ascend SFA kernel

https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/sparse_flash_attention/bench_sfa/sparse_mla_performance_optimization_zh.md
tile-ai/tilelang-ascend, branch `ascendc_pto`, retrieved 2026-10-20.

> ## Sparse KV: 稀疏访存优化
> 在稀疏场景下，KV 数据在 Global Memory 中呈离散分布，但后续的 Cube 矩阵运算需要连续的内存输入，这里采取了以下机制设计并优化：
>
> - **双vector核访存指令发射**：基于A2/A3芯片Cube核和Vector核数量1：2的特性，利用vector核发射访存指令，将指令下发效率提升一倍。
> - **离散 KV Gather 与连续搬出**：利用统一的稀疏索引先将离散的 KV 从 GM 收集到 UB，在这里gather为连续块之后，再一次性搬出到 Workspace 给后续cube核使用。
> - **异步拷贝 (核内手动同步)**：手动设置 `set_flag` / `wait_flag` 进行精确控制，使得离散拷入异步执行，大幅度缩减gather kv整体时间。
> - **搬入搬出 Ping-Pong (Double Buffer)**：对离散kv gather操作使用双buffer机制，当前数据通过 MTE3 写出到 Workspace 的同时，针对后一块数据的 MTE2 读取指令已经异步发起，使得MTE3与MTE2进行流水掩盖。
>
> 本样例最终实际的优化前后流水如下图所示，通过手动设置核内同步，MTE2搬入指令可实现连续下发，大幅提高了MTE2并行度；同时搬入搬出Ping-Pong实现了MTE3和MTE2流水掩盖，进一步减少kv gather耗时。

and on the workspace / L2 coupling:

> 在此模式下，编译器为每个逻辑任务独立执行 `alloc_buffer`、`annotate_address` 等初始化操作，并分配一份独立的 Workspace。当逻辑任务数远大于物理核数时，会产生三个性能问题：**(1)** 初始化操作随任务数冗余执行；**(2)** Workspace 总量随任务数线性膨胀，显存开销巨大；**(3)** 膨胀后的 Workspace 超出 L2 Cache 容量，Cube-Vector 间频繁交互的中间张量被置换到 HBM，遭遇严重的访存瓶颈。

and the tiling arithmetic:

> 以测试输入参数为例（`q_heads=128, kv_heads=1, m_base_size=16`）：
> - **Head group 维度分块**：GQA group size `g = q_heads / kv_heads = 128`。由于 `g > m_base_size`，将 Q head 维度按 `m_base_size=16` 切分，得到 `g_block_num = g / m_base_size = 8` 个子块。每个计算核处理 16 个连续的 Q head 子块。
> ...
> 将 `topk=2048` 按 `n_base_size=256` 切分，得到 `n_block_num = ⌈topk / n_base_size⌉ = 8` 次迭代。

## Relevance

This is the authoritative mechanism for the 5.0x, and it replaces my section 16 inference with a
hardware number.

**The binding constraint is 16 outstanding MTE2 requests per vector core.** `单个 VECTOR Core 在
同一时刻最多只能有 16 个"已发出但还未完成"的 MTE2 拷贝请求`. With `burst_count=2` that is 32
tokens in flight per vector core, and the author's own verdict on that is `太少了` — too few. On
A3 we have 2 vector cores per cube core, so 64 tokens in flight per cube core. Gathering 2048
tokens therefore takes at minimum 32 sequential issue-queue refills per (layer, head-group), each
refill bounded by HBM latency rather than bandwidth. **This is a latency-bound regime, and a
bandwidth roofline cannot describe it.** Our 1.382 ms "of bytes" is the wrong denominator in kind,
not just in magnitude.

**`CopyInKv`'s `blockCount = 2` is not a design choice, it is the gather2 trick.** Section 9.4
observed that `CopyInKv` folds exactly two tokens into one `DataCopyPad` and could not explain why
the ceiling was 2 rather than higher. It is 2 because `DataCopyExtParams` carries one `srcStride`
*and* because the point is to halve the number of entries consumed in a 16-deep issue queue:
`节约 MTE2 拷贝请求数量并降低发射 overhead`. The CANN kernel and DeepSeek's kernel independently
arrived at the same `burst_count=2`.

**The GM→UB→Workspace→L1 staging of section 9.1 is forced, not lazy.** This is the single most
important correction to section 9. I characterized the `kvMergeGm_` round trip as an avoidable
3-transit penalty. DeepSeek's doc says the direct route is *impossible*: a per-token sparse MTE2
cannot enable in-line ND2NZ (`无法启用随路的 ND2NZ 转换`), and the obvious workaround — treat a
token as 32 contiguous 32 B chunks and scatter them into L1 with a `B_TOPK * 32B` stride — fails
because `AI Core 内部的 BIU ... 只会在源和目标都连续的时候才会触发访存合并`, so a
source-contiguous / destination-strided copy is not coalesced and the bottleneck moves to L2
bandwidth. Their conclusion: `唯一的做法就是将 KV 先从显存拷贝到 UB，使用 SIMD VF 将其从 ND
Layout 转为 NZ Layout，然后再拷贝到 L1`. The staging pass is the only correct implementation.
So the ~2.8 ms I attributed to a removable round trip in section 9 is **not removable** by
restructuring the copy; it is removable only by making the gather less discrete (block-wise,
section 16) or less frequent (Selection KV Cache, section 14).

**The realistic ceiling is 83%, not 100%.** `prefill 时达到 410 TFlops，decoding 时达到 360
TFlops，分别达到了硬件理论性能上限的 95% 与 83%` — from the authors of the reference kernel, on
950, with full control of the implementation. Decode is structurally 12 points worse than prefill
and tops out at 83% of peak. A target that assumes 100% of a memory roofline at batch=1 decode is
not reachable on this architecture by anyone.

**Two levers this section does validate.** First, `双vector核访存指令发射` — use both vector cores
of each 1C2V group purely to issue MTE2, doubling the effective issue-queue depth to 32 entries
(`将指令下发效率提升一倍`). Section 9.3 shows the CANN kernel already splits `MergeKv` across
`GetSubBlockIdx()` 0 and 1, so this is already done. Second, `MTE2 拷贝的各个参数使用 SIMD VF
计算得到，避免阻塞 scalar 核心` — compute the DMA descriptors with vector instructions instead of
scalar, so that address generation does not block issue. Section 9.2 shows the CANN kernel does
the *opposite*: `topkGm_.GetValue()` and `blkTableGm_.GetValue()` are scalar GM reads on the
critical path, two per token. That is a concrete, named divergence between the shipped CANN kernel
and DeepSeek's design, and at 2048 tokens it is 4096 scalar GM reads per call that DeepSeek's
kernel does not pay. **This is the most specific actionable gap in the shipped operator found
anywhere in this document.**

**And the L2 warning bears on the megakernel design.** `膨胀后的 Workspace 超出 L2 Cache 容量，
Cube-Vector 间频繁交互的中间张量被置换到 HBM`. A3 has 192 MB L2 per die (section 20). The
`kvMergeGm_` budget from section 8.4 is 2.25 MB per core x 24 = 54 MB, which fits — but a
megakernel that also stages MoE and projection intermediates must keep the total under 192 MB or
the Cube-Vector relay silently falls out of L2, which is exactly the `CV relay 全口径流量约为
gather 的 7×` failure PR #1741 measured in section 18.1.

---

## 20. The measured decomposition: on A3 the scattered-gather penalty is 2.48x from instruction count and only 1.09x from burst length

This is the answer to the structural question. It is a controlled A/B from Huawei's own A3-labelled
example, and it separates the two hypotheses cleanly.

### 20.1 The controlled experiment — same bytes, same burst length, only `blockCount` varies

https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/05_best_practices/04_memory_access/data_copy/README.md
Huawei / CANN `asc-devkit`, HEAD `5f50fc4c6adab4bf891de229f4ff69b9d49e1897`, retrieved 2026-10-20.
"本样例支持的产品及CANN软件版本 | Atlas A3系列产品 | >= CANN 9.0.0".
优化点1, GM→UB via `DataCopyPad`, Tensor=[12288,12288] half, Block Num=48:

| Scenario | 配置 | Task Duration(μs) | aiv_total_cycles | aiv_mte2_time(μs) | aiv_mte2_ratio | 相对基线MTE2性能提升 | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Tile=[1,64] `DataCopyParams={blockCount=1, blockLen=128B, srcStride=24448B}` | 564.84 | 49222116 | 548.161 | 0.989 | 基线 | 小分块搬运 |
| 2 | Tile=[64,64] `DataCopyParams={blockCount=64, blockLen=128B, srcStride=24448B}` | 233.54 | 20167322 | 220.772 | 0.972 | +148.3% | 中等分块搬运 |
| 3 | Tile=[64,1024] `DataCopyParams={blockCount=64, blockLen=2048B, srcStride=22528B}` | 215.82 | 18563430 | 202.657 | 0.969 | +170.5% | 大分块搬运 |

> **MTE2 带宽理论分析**：… Atlas A2/A3 系列按 GM 带宽约 1.8TB/s 估算，理论耗时约 167.77μs。大分块场景下，GM 到 UB 的 `aiv_mte2_time` 为 202.657μs，GM 到 L1 的 `aic_mte2_time` 为 214.519μs，实测相对理论值分别高约 20.8% 和 27.9%。
>
> **优化效果分析**：
> - 增大分块可以显著降低搬运指令发射开销。GM 到 UB 场景中，大分块相对小分块在 Atlas A2/A3 系列上端到端性能提升约 161.7%，在 Ascend 950 系列上提升约 369.0%。
> - 实际配置时建议在片上空间允许的前提下优先增大单次搬运大小，同时保持分块不超过 UB/L1 可用缓存。单次搬运字节数越大、`mLoopCount/nLoopCount` 越少，DataCopy 指令数量和循环控制开销越低。

**Scenarios 1 and 2 use the identical `blockLen = 128B` and move the identical total bytes.** The
only change is `blockCount` 1 → 64. `aiv_mte2_time` falls 548.161 → 220.772 us = **2.48x**. Then
raising `blockLen` 128B → 2048B (2 → 3) buys only a further 220.772 → 202.657 = **1.09x**.

So on A3: the cost of a scattered gather is **2.48x from needing one instruction per block** and
**1.09x from the burst being short**. Burst length is nearly free; instruction count is not.

### 20.2 The remaining measured penalties, same source

优化点2, non-aligned tail:

> | 3 | Tensor=[12288,12288] Tile=[64,1024] blockLen=2048B | 215.82 | 202.657 | 0.969 | 基线 | 对齐大分块 |
> | 4 | Tensor=[12288,12287] Tile=[64,1024] full/tail blockLen=2048B,2046B | 275.3 | 259.047 | 0.973 | **-21.6%** | 非对齐数据大分块 |
>
> GM 到 L1: | 3 | Tensor=[12288,12287] … dValue=256(full),255(tail) | 438.76 | 422.515 | 0.986 | **-47.5%** | 非对齐数据大分块 |
>
> - 非对齐数据会引入额外边界处理，在 Atlas A2/A3 系列上影响更明显，尤其 GM 到 L1 场景端到端性能劣化约 47.5%。
> - 建议在设计矩阵 shape 和切分策略时尽量采用对齐矩阵。按本样例 half 数据类型计算，**Atlas A2/A3 系列建议主搬运维度对应的连续字节数满足 512B 对齐**；Ascend 950 系列建议满足 128B 对齐。

优化点4, multi-core same-address serialization (Tensor=[6144,512], Tile=[128,64], Block Num=48):

> | 7 | 539.42 | 42823246 | 474.717 | 0.984 | 所有核按相同顺序完整载入 |
> | 8 | 328.88 | 28054500 | 307.38 | 0.973 | 所有核错开分片顺序完整载入 |
>
> - 从端到端性能看，Atlas A2/A3 系列的 UB 场景提升约 64.0%

优化点3, L2 and the A3 hit-rate formula:

> A2/A3 的 L2Cache 大小为 192MB，950PR 的 L2Cache 大小为 128MB。
>
> Atlas A2/A3 系列和 Ascend 950 系列的 profiler 输出字段不同，命中率计算方式也不同：Atlas A2/A3 系列使用 `l2cache_hit_ratio = (r0_hit + r1_hit) / (r0_hit + r1_hit + r0_miss_allocate + r1_miss_allocate)`；Ascend 950 系列使用 `l2cache_hit_ratio = hit / (hit + miss + victim)`。
>
> | 5 | Atlas A2/A3 … 整块矩阵按相同路径连续搬 4 次 | 828.06 | 816.05 | r0_hit 212 | r0_miss_allocate 4718595 | r1_hit 217 | r1_miss_allocate 4718592 | **L2Cache命中率 0.005%** |
> | 6 | Atlas A2/A3 … N 方向切 4 份，每份连续搬 4 次 | 365.74 | 354.56 | 3539159 | 1179644 | 3539158 | 1179655 | **75.00%** |
>
> 重复搬运场景在同一数据规模下执行多轮 GM 读取，使用 `msopprof --ai-core=on --aic-metrics=L2Cache` 采集 L2Cache 读命中和 miss allocate 数据。

### 20.3 The A3 hardware constraint table — the burst split, stated for arch 2201

https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/hardware_constraints/npu_arch_2201.html
(source: https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/hardware_constraints/npu_arch_2201.md)
Huawei / CANN `asc-devkit`, 2026-10-20. Page title 「NPU架构版本2201」, tagged
`<!-- npu="A3" --> Atlas A3系列产品`. 表1 硬件约束以及解决方案建议:

> | 内存访问（GM） | 数据搬运会被拆成128B/256B/512B不同长度进行搬运，非对齐会向上取整。 | Tiling尽量让搬运的内轴128B、256B、512B对齐。<br> 具体解决方案可参考[GM地址尽量512B对齐](...)章节。 |
> | 内存访问（GM） | 单次搬运数据长度16KB以上时，可发挥带宽的最佳性能。 | 根据实测经验，单次搬运数据长度16KB以上时，通常能较好地发挥出带宽的最佳性能。因此对于单次搬运，应考虑尽可能的搬运较大的数据块（不同芯片不一样）。 |
> | 内存访问（GM-->L1） | DataCopy中源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）不要超出65535，单位为DataBlock（32字节）。 | 前面一个数据块的尾与后面数据块的头的间隔超出65535时，需要拆分成多条指令来实现。 |
> | 内存访问（GM） | 多核并行同地址访问GM，会被硬件串行化。 | 对相同地址的访问，会被硬件串行化，性能为排队时间，大约下降10%-20%；<br> 多核访问通过错峰访问（调整数据访问顺序和修改切分策略等），使得第一次加载数据到L2 Cache，后续访问性能反而提升。 |
> | 内存访问（L0 Buffer/L1 Buffer/Unified Buffer（UB）等） | 各存储单元的最小访问粒度/地址对齐要求：<br> UB：32Byte对齐。<br> L1 Buffer：32Byte对齐。<br> L0A Buffer/L0B Buffer：512Byte对齐。<br> L0C Buffer：64Byte对齐。<br> BiasTable Buffer：64Byte对齐。<br> Fixpipe Buffer：128Byte对齐。 |
> | Cube | MTE1和MMAD指令队列的深度为32。 | 对应指令队列容易满，会阻塞其他指令下发，引起流水断流。 |

The 512 B request granularity and its serialization:

https://asc.gitcode.com/guide/operator_practice/simd_operator_optimization/memory_access/avoid_same_address_access.html
tagged for `Ascend 950PR&950DT`, `Atlas A3系列产品`, `Atlas A2系列产品`:

> 【描述】MTE2、MTE3、Scalar等单元访问Global Memory数据时，其地址请求会按照512字节粒度对齐后进行处理。当同时访问Global Memory的数据，且地址处于连续的512字节范围内时，由于数据一致性的原因，多个请求会被串行处理，进而影响数据搬运效率。
>
> 当前算子执行机制保证用户核函数（Kernel）入参（包括Workspace/Tiling）的地址512字节对齐，因此开发者只需要根据地址的偏移量即可判断两个地址是否会落入连续的512字节范围内。
>
> 如下图所示，AI Core内的各个核对Global Memory的数据同时发出读写请求，尽管addr0~addr5是多个不同的地址，但因为落在连续的512字节范围内，被视为同一个地址请求，此时这几个数据请求会被串行处理，数据访问效率会降低。同地址访问的影响受同时访问的核数影响，同地址访问的核数越多时，串行导致的性能劣化越严重。

And the cacheline, same mechanism:

https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/basic_architecture.html

> 所有通过搬运单元读写GM的数据都缺省被缓存在L2Cache，以此加快访问速度，提高访问效率。**核外L2Cache以Cache Line为单位加载数据，根据硬件规格不同，Cache Line大小不同（128/256/512Byte等）。**

https://asc.gitcode.com/guide/operator_practice/simd_operator_optimization/memory_access/l2_cache_split.html

> 【描述】假设，AI处理器的L2 Cache大小为192MB，L2 Cache读写混合带宽约为7TB/s，而GM的带宽约为1.6TB/s，两者之间存在较大差距。…若数据无法命中L2 Cache，即需要访问的数据不在L2 Cache内，导致需要去GM上读写，带宽利用效率较低，最终算子搬入或搬出数据变为算子整个运行过程的性能瓶颈。

### 20.4 The bandwidth-vs-block-size curve

https://www.hiascend.com/developer/techArticles/20240906-1?envFlag=1
昇腾CANN 官方技术文章, 2024-09-06, 「Ascend C算子性能优化实用技巧03——搬运优化」

> ## 02 尽量一次搬运较大的数据块
> 搬运不同大小的数据块时，对带宽的利用率（有效带宽/理论带宽）不一样。根据实测经验，单次搬运数据长度16KB以上时，通常能较好地发挥出带宽的最佳性能。因此对于单次搬运，应考虑尽可能的搬运较大的数据块。下图展示了某款AI处理器上实测的不同搬运数据量下带宽的变化图。
>
> 说明：测试数据与处理器型号及HBM频率相关，且实际测试时可能会存在略微抖动，具体带宽数值并不一定和下文的测试数据严格一致。
>
> ## 03 GM地址尽量512B对齐
> 由于AI处理器内部设计约束，从GM向Local Memory搬运数据时，保证GM地址512B对齐可以最高效的发挥出带宽的效率。如下图示例，展示了在512B对齐以及32B对齐情况下单核的带宽效率：搬运同等数据量，带宽差距最大的情况，32B对齐场景只能达到512B对齐场景的70%。
>
> 说明：本性能优化手段仅针对Atlas A2训练系列产品/Atlas 800I A2推理产品生效。
>
> ## 04 高效使用搬运API
> 使用搬运API时，应该尽可能地使用API的srcStride/dstStride/blockLen/blockCount等参数实现连续搬运或者固定间隔搬运，而不是使用for循环，二者效率差距极大。如不这样，图片的每一行为16KB，需要从每一行中搬运前2KB，针对这种场景，使用srcStride/dstStride/blockLen/blockCount等参数可以达到一次搬完的效果，每次搬运 32 KB；如果使用for循环遍历每行，每次仅能搬运 2 KB。

Chart figure URLs (same article; byte-identical copies in the doc repo as
`docs/zh/guide/figures/gm_ub_bw.png`, `ub_gm_bw.png`, `gm_ub_align.png`, `ub_gm_align.png`):
`https://www.hiascend.com/p/resource/202409/103ad6f61a9d4f47b74b2aed38c70f59.png`,
`.../2a74314eb0ac460384a08a2491b8a613.png`, `.../2bbae8909d534f11b560da0fada6de59.png`,
`.../38487e7f1a224bcfa6d56c8eefd64d15.png`, `.../769746be1b844ed593cf6eef434779e2.png`,
`.../d2a0020cc58d449c8c3413183f082609.png`, `.../d79b8ba0e26b4074a2091a47b9b6b459.png`.

Values read off the vendor plot (a reading of the figure, **not** vendor-printed numbers), axes
带宽(GB/s) vs 单次搬运数据量(KB), single core:

- GM→UB: ~1 KB ≈ 1.7 GB/s · 2 KB ≈ 3 · 4 KB ≈ 6.5 · 8 KB ≈ 12.5 · 12 KB ≈ 17 · 16 KB ≈ 29 ·
  24 KB ≈ 29 · 32 KB ≈ 27.5 (dip) · 48 KB ≈ 31 · 64 KB ≈ 33 · 96 KB ≈ 34.5 · 128-160 KB ≈ 33.7 ·
  192 KB ≈ 35.
- UB→GM: 1 KB ≈ 1.3 · 2 KB ≈ 2.8 · 4 KB ≈ 5.5 · 8 KB ≈ 11 · 12 KB ≈ 15 · 16 KB ≈ 28.5 ·
  32 KB ≈ 27.5 · 48 KB ≈ 30 · 64 KB ≈ 33 · 160 KB ≈ 35.2.
- 512B-aligned vs 32B-aligned (GM→UB): at 16 KB, 29 vs 22 GB/s (76%); at 32 KB, 27.5 vs 26;
  converging above 96 KB.

Single-core saturation ≈ 35 GB/s; a 2 KB transfer gets ≈ 3 GB/s, i.e. ~12x off saturation.

### 20.5 `DataCopyPad` / `DataCopyExtParams` on A3 — exact limits

https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.html
Ascend C API reference (asc.gitcode.com, CANN 9.2.0 tree), 2026-10-20.
「DataCopyPad（GMToUB非对齐数据搬运）」, 产品支持情况: `Atlas A3系列产品：支持`

> ```cpp
> template <typename T>
> __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
>                                    const DataCopyExtParams& dataCopyParams,
>                                    const DataCopyPadExtParams<T>& padParams)
> ```
>
> 针对如下型号，DataCopyExtParams结构体参数定义请参考表4：Atlas A3系列产品 / Atlas A2系列产品 / Atlas 200I/500 A2推理产品
>
> **表4 DataCopyExtParams结构体参数定义**
> | blockCount | 搬运的数据块个数，数据类型为uint16_t，取值范围：**blockCount∈[0, 4095]**，默认值为1。 |
> | blockLen | 搬运的每个数据块长度，数据类型为uint32_t，取值范围：blockLen∈[0, …]，单位：1B。注：blockLen必须是sizeof(T)的整数倍，需要注意不要超过UB空间大小。 |
> | srcStride | 源操作数相邻数据块之间的间隔（即前一个数据块**结束地址**与后一个数据块**起始地址**的差值），数据类型为**uint32_t**，取值范围：srcStride∈[0, …]，单位：1B。 |
> | dstStride | 目的操作数相邻数据块之间的间隔…数据类型为uint32_t…单位：dataBlock（32B）。 |
> | rsv | 预留参数，当前需设置为0。 |
>
> 针对Ascend 950PR&950DT系列产品，DataCopyExtParams结构体参数定义请参考表3。
> | blockCount | …取值范围：**blockCount∈[0, 65535]**，默认值为1。 |
> | srcStride | …数据类型为**int64_t**，取值范围：srcStride∈[**-blockLen**, …]，单位：1B。**当srcStride = -blockLen时，表示每次传输的连续数据块均为同一块，即重复搬运第一个数据块。** |
>
> **约束说明**
> - 当DataCopyExtParams结构体参数blockCount、blockLen任意一个值为0时，该接口将被视为NOP（空操作）。该说明针对如下型号生效：Atlas A3系列产品 / Atlas A2系列产品 / Ascend 950PR&950DT系列产品
> - 位于Global Memory的源地址必须1字节对齐，位于UB的目的地址必须32字节对齐。
> - 特别地，针对Ascend 950PR&950DT系列产品，支持Compact模式，该模式支持单个数据块非32字节对齐，将所有数据块合并成一个连续数据块，在该数据块右侧填充数据至32字节对齐。
>
> ```cpp
> enum class PaddingMode : uint8_t {
>     Normal = 0, // 默认模式，与原有数据搬运格式保持一致，每次数据搬运都会补齐至32字节对齐
>     Compact,    // 紧凑模式，允许单次搬运不对齐，统一在整块数据末尾补齐至32字节对齐
> };
> ```

And the plain `DataCopy` gap limits —
https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopy_GMAndUB_highdim_split.html,
「DataCopy（GM与UB高维切分数据搬运）」, `Atlas A3系列产品：支持`:

> **表3 DataCopyParams结构体参数定义**
> | blockCount | 搬运的数据块个数。数据类型为uint16_t，取值范围：blockCount∈[0, 4095]，默认值为1。 |
> | blockLen | 搬运的每个数据块长度。数据类型为uint16_t，取值范围：blockLen∈[0, 65535]，单位：DataBlock（32B）。 |
> | srcGap | 源操作数相邻数据块之间的间隔（即前一个数据块**结束地址**与后一个数据块**起始地址**的差值），数据类型为uint16_t，取值范围：srcGap∈[0, 65535]，单位：DataBlock（32B）。<br>• blockCount = 1时，srcGap无意义，设置为0即可。 |
> | dstGap | …取值范围：dstGap∈[0, 65535]，单位：DataBlock（32B）。 |

### 20.6 There is no hardware gather-DMA from GM on Ascend

`Gather` and `asc_gather_datablock` are UB-only and run on `PIPE_V`, not MTE2.

https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/scatter_gather/Gather.html

> ```cpp
> template <typename T>
> __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint32_t count)
> ```
> Gather接口接受输入张量（src）、地址偏移张量（srcOffset）和基地址（srcBaseAddr），根据基地址和地址偏移量确定输入张量的索引，将输入张量中对应元素收集到结果张量（dst）中。

https://asc.gitcode.com/api/SIMD-API/c_api/vector_compute/vector_gather/asc_gather_datablock.html

> ```cpp
> __aicore__ inline void asc_gather_datablock(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride, uint8_t dst_block_stride, uint8_t repeat)
> ```
> 给定一个输入的矢量和一个地址偏移矢量，根据偏移地址按照DataBlock的粒度将输入矢量收集到目的矢量中。
> ## 流水类型
> PIPE_V
> ## 约束说明
> - src加上src_offset偏移后的地址不能超出Unified Buffer（UB）大小数据的范围。

And the multi-dimensional DMA that would describe a strided gather in one descriptor is 950-only:

https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopy_GMToUB_NDDMA.html

> ## 产品支持情况
> - Ascend 950PR&950DT系列产品：支持
> - **Atlas A3系列产品：不支持**
> - Atlas A2系列产品：不支持
>
> 多维数据搬运接口，相比于基础数据搬运接口，可更加自由配置搬入的维度信息以及对应的Stride。
> ```cpp
> template <typename T, uint8_t dim, const NdDmaConfig& config = kDefaultNdDmaConfig>
> __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const NdDmaParams<T, dim>& params)
> ```
> | loopSrcStride | 表示每个维度内，该源操作数元素与下一个元素间的间隔。单位为元素个数。数据类型为uint64_t，loopSrcStride需在[0, 2^40)。 |

### 20.7 Profiling metric definitions, and the one metric that does NOT exist on A3

https://www.hiascend.com/doc_center/source/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/atlasprofiling_16_0072.html
Huawei, MindStudio 7.0.RC1 命令行用户指南, 「op_summary（AI Core、AI Vector Core和AI CPU算子数据）」.
表2 PipeUtilization; 表10 is the `ai*_` variant, 「下列字段仅 Atlas A2训练系列产品/Atlas 800I A2推理产品 支持」:

> mte2_time(us) — mte2类型指令（DDR->AICORE搬运类指令）耗时，单位us。
> mte2_ratio — mte2类型指令（DDR->AICORE搬运类指令）的cycle数在total cycle数中的占用比。
> mte1_ratio — mte1类型指令（L1->L0A/L0B搬运类指令）的cycle数在total cycle数中的占用比。
> mte3_ratio — mte3类型指令（AICORE->DDR搬运类指令）的cycle数在total cycle数中的占用比。
> mac_time(us) — cube类型指令（矩阵类运算指令）耗时，单位us。
> mac_ratio — cube类型指令（矩阵类运算指令）的cycle数在total cycle数中的占用比。
> icache_miss_rate — icache是为instruction预留的L2 Cache，icache_miss_rate数值高代表AI Core读取指令的效率低。
> **memory_bound** — 用于识别AICore执行算子计算过程是否存在Memory瓶颈，由mte2_ratio/max(mac_ratio, vec_ratio)计算得出。计算结果小于1，表示没有Memory瓶颈；计算结果大于1 则表示AI Core在执行Task过程中大部分时间都在做内存搬运而不是计算，且数值越大Memory瓶颈越严重。
> **cube_utilization(%)** — cube算子利用率，查看cube算子在单位时间内的运算次数是否达到理论上限，越接近于100%则表示越接近理论上限。计算公式：cube_utilization=total_cycles / (freq * core_num * task_duration)。
>
> 表10 字段说明（PipeUtilization）— 下列字段仅 Atlas A2训练系列产品/Atlas 800I A2推理产品 支持
> **ai\*_mte2_time(us)** — mte2类型指令（**片上内存**->AICORE搬运类指令）耗时，单位us。
> **ai\*_mte2_ratio** — mte2类型指令（**片上内存**->AICORE搬运类指令）的cycle数在total cycle数中的占用比。
> ai\*_mac_ratio — cube类型指令（矩阵类运算指令）的cycle数在total cycle数中的占用比。
>
> 表15 字段说明（ResourceConflictRatio）
> **ai\*_vec_bankgroup_cflt_ratio** — vec_bankgroup_stall_cycles类型指令执行cycle数在total cycle数中的占用比。由于vector指令的block stride的值设置不合理，造成bankgroup冲突。
> ai\*_vec_bank_cflt_ratio — vec_bank_stall_cycles类型指令执行cycle数在total cycle数中的占用比。由于vector指令操作数的读写指针地址不合理，造成bank冲突。
> ai\*_vec_resc_cflt_ratio — …当某个计算单元正在执行计算，但算子逻辑仍然往该单元下发指令，就会造成整体的算力没有得到充分应用。
>
> 表16 字段说明（L2Cache）
> ai\*_write_cache_hit — 写cache命中的次数。
> ai\*_r\*_read_cache_hit — 读r*通道cache命中次数。
> ai\*_r\*_read_cache_miss_allocate — 读r*通道cache缺失后重新分配的次数。

`mte2_iq_full_ratio` exists only on the pre-A2 generation —
https://www.hiascend.com/doc_center/source/zh/canncommercial/5046/devtools/auxiliarydevtool/atlasprofiling_16_0119.html,
CANN 商用版 5.0.4.6, 「AI Core性能指标采集项说明」:

> mte1_iq_full_ratio：代表mte1_iq_full_cycles类型指令执行cycle数在所有指令执行cycle数中占比。
> **mte2_iq_full_ratio：代表mte2_iq_full_cycles类型指令执行cycle数在所有指令执行cycle数中占比。**
> mte3_iq_full_ratio：代表mte3_iq_full_cycles类型指令执行cycle数在所有指令执行cycle数中占比。
> cube_iq_full_ratio：代表cube_iq_full_cycles类型指令执行cycle数在所有指令执行cycle数中占比。
> vec_iq_full_ratio：代表vec_iq_full_ratio类型指令执行cycle数在所有指令执行cycle数中占比。
> **iq_full_ratio：代表vec_resc_cflt_ratio，mte1_iq_full_ratio，mte2_iq_full_ratio，mte3_iq_full_ratio，cube_iq_full_ratio，vec_iq_full_ratio类型指令执行cycle数之和在所有指令执行cycle数中占比。**

On CANN 8.0.0 (https://www.hiascend.com/doc_center/source/zh/canncommercial/800/devaids/devtools/profiling/atlasprofiling_16_0069.html,
「ai_core_utilization（AI Core指令占比）」, 表7 ResourceConflictRatio) the table contains only
`vec_bankgroup_cflt_ratio`, `vec_bank_cflt_ratio`, `vec_resc_cflt_ratio` — no `*_iq_full_ratio`.

The tool that can see per-instruction issue gaps —
https://asc.gitcode.com/guide/programming_guide/debug_and_tuning/performance_tuning.html:

> ```
> OPPROF_{timestamp}_XXX
> ├──ArithmeticUtilization.csv  # cube/vector指令cycle占比，建议优化算子逻辑，减少冗余计算指令
> ├──L2Cache.csv                # L2 Cache命中率，影响MTE2，建议合理规划数据搬运逻辑，增加命中率
> ├──Memory.csv                 # Unified Buffer（UB），L1和主存储器读写带宽速率，单位GB/s
> ├──PipeUtilization.csv        # pipe类指令耗时和占比，建议优化数据搬运逻辑，提高带宽利用率
> ├──ResourceConflictRatio.csv  # UB上的bank group、bank conflict和资源冲突率在所有指令中的占比
> ```
> bisheng add_custom.asc -o add_custom --npu-arch=dav-2201
> msopprof ./add_custom
> | msopprof simulator | 适用于仿真性能分析，可协助用户分析算子指令和代码热点问题。 | 指令流水图 / 算子代码热点图 / 内存通路吞吐率波形图 |

### 20.8 The launch floor: our 0.70 us is corroborated exactly; 13.58 us is not

https://hpc101.zjusct.io/lab/Lab3.5-AscendC-Op/
ZJUSCT (浙江大学超算队), HPC101「Lab 3.5：昇腾算子开发与优化」, retrieved 2026-10-20, platform 910B4:

> **从启动开销的角度思考为什么不一定需要启动全部 AIV**
>
> 在 NPU 中，每个核在被启动时都要单独付一次初始化开销（加载配置、建立执行上下文、准备片上资源等），且这部分几乎无法通过流水、搬运或规约优化消除——它只和「启动了多少个核」有关。下面是用一个 body 几乎为空的算子，只改变 `<<<blocks>>>` 下发 AIV 数目，用 `msprof op` 测得的 Task Duration，可以近似认为是启动开销：
>
> 它随启动核数近似线性增长（**约 2 核 0.67 µs → 40 核 2.25 µs**），本身处在微秒量级，在大算子里几乎被 profiling 波动淹没；但当算子整体只有几微秒、单核计算量又很小时就占比可观。因此计算量过小的任务要选择「够用就好」的核数，而不是无脑写满全部 AIV。
>
> **提高搬运效率**
> MTE 搬运存在固定启动成本，单次搬运过小时难以充分利用带宽。
> 若 UB 预算和数据布局允许，可以尝试用一次 `DataCopy` 搬运多行，或用 `blockCount`、`blockLen` 与 stride 参数表达规则的多段搬运，减少逐行发射指令的开销。

The kernel-type cost model —
https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0218.html,
「设置Kernel类型」:

> ## 功能说明
> 用于用户自定义设置kernel类型，**控制算子执行时只启动该类型的核，避免启动不需要工作的核，缩短核启动开销。**
>
> | KERNEL_TYPE_AIV_ONLY | 算子执行时仅启动AI Core上的Vector核：比如用户在host侧设置numBlocks为10，则会启动10个Vector核。 |
> | KERNEL_TYPE_AIC_ONLY | 算子执行时仅启动AI Core上的Cube核：比如用户在host侧设置numBlocks为10，则会启动10个Cube核。 |
> | KERNEL_TYPE_MIX_AIC_1_1 | AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:1，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核和10个Vector核。 |
> | **KERNEL_TYPE_MIX_AIC_1_2** | AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:2，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置numBlocks为10，**则会启动10个Cube核和20个Vector核**。 |

And the closest public number in magnitude to our MIX floor —
https://cann.csdn.net/69afcc5554b52172bc603633.html, 昇腾CANN 官方,
「大模型推理加速利器SuperKernel技术综述」:

> 以DeepSeekV3网络中的一层为例，根据Profiling数据分析发现，网络上肉眼可见的Task等待调度开销较为明显。在BatchSize大小为96、Sequence长度3K的场景下，**平均每层的Task平均等待的开销约为18us左右**。按照61层网络计算，总的调度等待开销在1.1ms左右。这些开销基本是绝对开销，在大的Batch场景下还会进一步扩大。
>
> 通过详细评估和调试，SuperKernel技术优化后的基本收益模型可以分为以下三部分：首先是两个Task之间的调度开销降低；其次是每个Task结束后会执行Cache Flush操作，将所有修改的Cache内容刷新出去，其中包含一些不必要的栈空间；**最后是算子调度后的核启动时间，包括同地址访问的排队延迟、Scalar执行初始化等环节。**
>
> 在多核处理器系统中，当多个计算核心同时执行同一个内核函数时，它们会访问内存中完全相同的指令地址（Text段）。这种对同一地址的并发访问会在共享的L2缓存层面形成一个串行化的访问队列，导致多个核心的请求发生争用，即使数据已缓存在L2中，也会因队列拥堵产生访问延迟，显著削弱多核并行带来的性能提升。
>
> torchair.scope.super_kernel('layers', 'debug-dcci-all=1') # 自动为每个算子结束后插入cache强制刷新，**每算子约2-3us耗时**，仅做排查问题使用
>
> 特定执行模式需要适配：对于设计上原本AIC和AIV配比为1:1的算子，需要对其进行适配，使其能够在AIC和AIV配比为1:2的模式下正确执行，以满足SuperKernel的调度要求。

https://www.hiascend.com/developer/techArticles/20240715-1
昇腾CANN 官方技术文章, 2024-07-15, 「深度解读昇腾CANN模型下沉技术，提升模型调度性能」:

> 比如，大模型场景的增量推理或训练的FineTune阶段，都是计算量较小的场景，此时很容易出现单个算子的Host下发时间比Device上的算子执行时间还长，从而导致Device间歇处于空闲状态。这种现象通常称为Host Bound，这种模型也称为Host Bound模型。
>
> 以Host Bound的LLaMA-7B模型的推理场景为例，…通过Profiling结果可以看出该模型是一个Host Bound模型，采用Host调度方式，Device上计算时间和空隙的时间比例接近1:1，采用模型下沉执行方式，Device空隙大幅减少，并且端到端有18ms的收益，吞吐量整体提升37%。
>
> 以盘古71B增量推理模型为例（模型输入输出总个数约1600个，模型内节点总数约6300个，…），当前模型下沉的头开销约2ms，关于头开销的性能优化在持续进行中。

### 20.9 PageAttention `block_size` on A3, officially

https://www.hiascend.com/doc_center/source/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000787.html
Huawei 昇腾 PyTorch API 参考, 「torch_npu.npu_fused_infer_attention_score」:

> 当Q_S等于1时：… page attention场景：
> 使能必要条件是blocktable存在且有效，同时key、value是按照blocktable中的索引在一片连续内存中排布，支持key、value dtype为FLOAT16、BFLOAT16、INT8，在该场景下key、value的inputLayout参数无效。
> **该场景下，blockSize是用户自定义的参数，该参数的取值会影响page attention的性能。key、value输入类型为FLOAT16、BFLOAT16时需要16对齐，key、value 输入类型为INT8时需要32对齐，推荐使用128。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。**

https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5908032.html
华为云 ModelArts 最佳实践, 「Ascend-vLLM推理常见问题」:

> 问题3：使用离线推理时，性能较差或精度异常
> **解决方法：将block_size大小设置为128**

And the counter-pressure, https://github.com/vllm-project/vllm-ascend/issues/12589:

> The root cause is traced to the default `block-size=128` … With `block_size=128`, a 128K prompt generates 1024 blocks. Due to hash granularity, only fully filled blocks (128 tokens) can be cached. In practice, the last block of each chunked prefill segment is often incomplete, causing hash mismatches. The large block size leads to high fragmentation: different requests produce different block boundaries for the same logical prefix, preventing hash matches.
> … In vLLM-Ascend, the recommended block size for optimal caching is 32 (as noted in internal documentation and recent code changes).

### 20.10 The official bandwidth-utilization method, with worked numbers

https://bbs.huaweicloud.cn/blogs/433080
昇腾CANN, 华为云社区, 2024-08-16, 「基于Ascend C的Matmul算子性能优化最佳实践」,
性能验证平台为Atlas A2训练系列产品/Atlas 800I A2推理产品:

> 从上图Profiling数据来看，**aic_mte2_ratio数值是0.973**，这表明MTE2类型指令的cycle数在total cycle数中的占比过大，这意味着当前性能瓶颈点可能在于MTE2流水。
>
> **开启大包搬运后，指令条数变化、地址对齐等因素会影响性能，按照经验预估，对于MTE2为性能瓶颈的场景，会有20%以上的MTE2性能收益。**
>
> …搬出一次Matmul结果到Global Memory的base块是64 * 64，由于输出格式是ND，数据类型是float，搬出下一次Matmul结果的起始地址需要偏移一个baseN的大小，即64 * 4 = 256字节，导致fixpipe搬出时Global Memory地址非512byte对齐，那么需要设置更优的基本块。
>
> 从下图可以看到，使能这组基本块后，**MTE2耗时（对应aic_mte2_time）从2452us降低到808us，MTE2性能提升3倍。**
>
> 当前带宽利用率为：totalSize / mte2Time = totalCnt * dtype / mte2Time，代入数据计算为 **2491GB/s**。未使能大包搬运的情况下，矩阵从Global Memory搬运到L1一次只搬运1个基本块。通过模板参数使能大包搬运，一次搬运多个基本块，提高MTE2带宽利用率。
>
> 从下图可以看到，使能大包搬运后，**MTE2耗时从808us下降到591us，带宽利用率代入数据计算为3406GB/s，利用率提升36%+，Cube利用率（对应aic_mac_ratio）达到80%+。**

## Relevance

This is the measured decomposition the structural question asked for, and it settles the ranking
of causes with controlled A/B data rather than reasoning.

**The answer to (b): it is instruction count, not burst length.** From 20.1, holding bytes and
`blockLen = 128B` fixed and changing only `blockCount` 1 → 64 is worth **2.48x**; then raising
`blockLen` 128B → 2048B is worth only **1.09x**. Every mechanism in sections 9, 16 and 19 that
reduces instruction count — the `blockCount=2` pairing, the 32-row UB staging, block-wise
`sparse_block_size` — attacks the 2.48x term. Nothing attacks the 1.09x term, and nothing needs
to. This retires the "short DMA burst" hypothesis as a significant contributor and confirms
"MTE2 descriptor overhead" as the real one.

**The full penalty budget for A3, each measured:**

| cause | measured A2/A3 penalty | source |
| --- | --- | --- |
| one instruction per block vs 64 per instruction, same 128 B burst | **2.48x** | 20.1 sc. 1 vs 2 |
| burst 128 B vs 2048 B, same instruction count | **1.09x** | 20.1 sc. 2 vs 3 |
| non-aligned tail, GM→UB | 1.22x | 20.2 优化点2 |
| non-aligned tail, GM→L1 (ND2NZ) | 1.48x | 20.2 优化点2 |
| multi-core same-512 B-address serialization | 1.64x (constraint table says 10-20%) | 20.2 优化点4, 20.3 |
| L2 hit 0.005% vs 75% | 2.26x | 20.2 优化点3 |
| single-core bandwidth at 2 KB vs saturated | ~12x | 20.4 |

Even the *best* case in 20.1 — 64 contiguous 2 KiB blocks per instruction — lands 20.8% (GM→UB)
and 27.9% (GM→L1) above the 1.8 TB/s theoretical time. So a 1.21-1.28x gap is the floor for a
perfectly-tiled dense copy on A3. A roofline target of 1.0x is unreachable in principle.

**Three A3 hardware limits that constrain every fix proposed above.** First, `blockCount ∈
[0, 4095]` on A3 (65535 on 950), so at most 4095 blocks per instruction, and only if one stride
describes them. Second, **`srcStride` is `uint32_t` on A3** while on 950 it is `int64_t` with the
`-blockLen` repeat trick — so on A3 a pair must be emitted with the lower address first, which is
exactly why `CopyInKv`'s `还原成2条搬运指令` fallback fires on an unsorted index list (section
9.4), and it confirms section 9's recommendation to sort the top-k ascending. Third, A3's
`PaddingMode` has no `Compact`: `每次数据搬运都会补齐至32字节对齐` — every block pads
individually, so A3 pays a padding event per gathered row that 950 does not.

**And the hardest limit: there is no gather DMA.** `Gather` and `asc_gather_datablock` take
`__ubuf__` / `LocalTensor` sources and run on `PIPE_V`, not MTE2; the multi-dimensional
`NdDmaParams` DMA is marked `Atlas A3系列产品：不支持`. Combined with section 17's
`Ascend 950系列新增了SIMT单元` and section 19's BIU no-coalesce finding, that is three independent
confirmations that on A3 a scattered GM gather must become N descriptors. The 2.48x is therefore
not an implementation defect — it is the price of the only available expression.

**Our 0.70 us AIV launch floor is independently reproduced; our 13.58 us MIX floor is not.**
ZJUSCT measures `约 2 核 0.67 µs → 40 核 2.25 µs` for an empty kernel on 910B4, against our
0.70 us — the methodology is sound. But that linear model (~0.04 us/core) predicts a few
microseconds for MIX_AIC_1_2, not 13.58. Nothing public corroborates a ~13 us MIX floor. The
documented mechanisms that could produce it, none of them quantified for MIX specifically:
MIX_AIC_1_2 starts 3x the cores at the same `numBlocks` (20.8); ICache is 32 KiB on the cube core
and 16 KiB on the vector core (section 19.4) while a MIX kernel carries both code segments; and
`多核并行同地址访问ICache，会被硬件串行化` with Huawei's own breakdown of 核启动时间 naming
`同地址访问的排队延迟` and `Scalar执行初始化`. The closest public number in magnitude is
`平均每层的Task平均等待的开销约为18us左右` on a DeepSeek-V3 decode layer, `按照61层网络计算，总的
调度等待开销在1.1ms左右` — same order as our 13.58 us, and SuperKernel is the named cure.

**Diagnostically, one metric we wanted does not exist on A3.** `mte2_iq_full_ratio` — the direct
"MTE2 issue queue is full" signal that would confirm the 2.48x term in our own kernel — is a
pre-A2 metric; on A2/A3 `ResourceConflictRatio` is vector-only. So the instruction-issue
bottleneck must be inferred: compare `aic_mte2_time` / `aiv_mte2_time` against
`bytes / 1.8 TB/s`, collect the `L2Cache` group
(`aiv_r0/r1_read_cache_hit` + `_miss_allocate`, with
`l2cache_hit_ratio = (r0_hit + r1_hit) / (r0_hit + r1_hit + r0_miss_allocate + r1_miss_allocate)`
on A3), and use `msopprof simulator` for the 指令流水图 and 内存通路吞吐率波形图, built with
`--npu-arch=dav-2201`. Note also that `memory_bound = mte2_ratio / max(mac_ratio, vec_ratio)`
cannot distinguish "MTE2 busy moving bytes" from "MTE2 busy issuing descriptors" — both read as
`mte2_ratio ≈ 1`, which is precisely our situation.

**One warning about L2.** A3's L2 is 192 MB and the selected KV working set at seq 8K is
104.8 MB (section 13), so it nominally fits — but the `kvMergeGm_` staging adds 2.25 MB x 24 cores
= 54 MB of write traffic through the same cache, and 20.2 優化点3 shows L2 hit rate swinging
2.26x on access-order alone. Our scattered reads are also exposed to the 512 B same-address
serialization of 20.3: at TP16 with every rank reading the *same* MQA latent cache
(`KV_N仅支持1`), cross-core and cross-rank reads landing in one 512 B window are serialized by
hardware, `同地址访问的核数越多时，串行导致的性能劣化越严重`. That is a TP16-specific penalty
that does not appear at TP1 and is not in any byte count.

---

## 21. The vendor's FlashAttention tuning case, and two candidate causes ruled out by inspection

### 21.1 The measured before/after, with the five levers and their yields

/home/shanshan/repo/cann/asc-devkit/docs/zh/guide/operator_practice/best_practices/flash_attention_tuning.md
(CANN `asc-devkit`, local clone, file dated 2026-08-21, 226 lines; published as
https://www.hiascend.com/developer/techArticles/20240607-1 and
https://www.cnblogs.com/huaweiyun/p/18243836, 昇腾CANN, 2024-06-12,
「基于Ascend C的FlashAttention算子性能优化最佳实践」, verified on
Atlas A2 训练系列产品 / Atlas 800I A2 推理产品)

> ## 分析主要瓶颈点
>
> 通过观察分析流水图和Profiling数据，结合优化经验来判断性能瓶颈点。在优化过程中不同阶段可能会出现不同的瓶颈点，需要不断优化以达到最佳性能。
>
> -   根据优化经验，循环间会存在一些不必要的性能开销，循环越多性能可能越差；满足Unified Buffer（UB）最大空间限制的情况下，UB切分的基本块越大，循环越少。算子中通过InitBuffer接口分配UB buffer大小。
>
>     ```
>     pipe->InitBuffer(ubBuffer, 120 * 1024);
>     pipe->InitBuffer(tmpBuffer, 30 * 1024);
>     pipe->InitBuffer(vecClc3, 8 * 1024);
>     ```
>
>     如上代码所示，InitBuffer接口的第二个参数表示buffer占用的大小，所有buffer大小的和即为占用的总空间。这里120 \* 1024 + 30 \* 1024 + 8 \* 1024 = 158KB < UB Size，没有充分利用UB空间。
>
> -   观察如下流水图，绿色为Vector侧的流水，橙色为Cube侧的流水，可以看出两侧的流水都存在大段的空隙，CV之间流水很大程度上未并行，需要考虑CV流水优化。
>
> -   对于上述场景一，causal场景下可能存在核间分布不均匀的情况，如下图经过atten\_mask掩码后，红色部分是算子需要计算的部分，绿色无需计算；如果不按照基本块的个数来分核，按照第一根轴的大小8（行）来分核，假设平均分到9个核上，每个核做ceil\(8 / 9\) = 1行，则第1个核只需做1个基本块，但是第8个核需要做8个基本块的计算，出现明显的负载不均衡。因此需要考虑将红色块均匀分到多个核上计算。
>
> -   场景一的Profiling数据如下，aic\_fixpipe\_ratio占比极高，可能存在FixPipe bound。
>
> -   场景二的Profiling数据如下，mte2\_ratio占比高，可能存在MTE2 bound。

> -   优化点三：每个核负载均衡
>
>     尽量实现每个核的计算量均匀，负载均衡。优化前的分核及每个核的计算量如[图11]所示，按照第一根轴的大小8（行）来分核，平均分到9个核上，每个核计算ceil\(8/9\)=1行，第1个核只计算1个基本块，但是第8个核计算8个基本块。优化后如[图12]所示，红色块总共36个基本块，均分到每个核上，每个核的计算量为4个基本块，性能提升一倍。
>
> -   优化点四： FIXPIPE优化
>
>     从采集的Profiling数据来看，Cube FixPipe占比高达81%，出现了很严重的bound（达到上限）。CAModel工具打印发现存在很多异常的128B搬运，排查代码，发现workspace地址未512B对齐。
>
>     代码实现中使用SetGlobalBuffer接口设置workspace的起始地址，如果起始地址不是按照512B对齐，搬运效率会很低，可以强制[GM地址512Byte对齐]来避免这个情况。下面代码中ADDR\_ALIGN\_SIZE即为512。
>
>     ```
>     // init workspace address
>     syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace);
>     uint64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
>     dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
>     workspaceOffsets = (workspaceOffsets + qPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;  dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
>     ...
>     ```
>
>     修改代码，workspace地址经过512B对齐后，FixPipe时间减半。
>
> -   优化点五：MTE2优化
>
>     结合如下的Profiling数据和流水图，可以看出MTE2 bound，且部分MTE2搬运时间异常。
>
>     将输入数据排布格式从BSH更改为BNSD后，数据搬运连续，不需要跳地址读取数据，搬运效率提升一倍，部分异常搬运时长降低了一半。

> ## 验证优化方案性能收益
>
> -   调整tiling基本块：理论评估Vector切块越大，计算和搬运循环次数越少，同时能够充分利用搬运带宽和Vector算力。基本块大小从\(64, 128\)增大到\(128, 128\)后，性能提升一倍，实测与理论分析一致。
> -   CV流水并行：CV流水掩盖的时间即为提升的性能，符合预期的收益。
> -   核间负载均衡：优化前负载最多的核计算量减少的倍数，即为预期提升的性能；案例中优化前负载最多的核的计算量大小为8块，优化后为4块，实际性能提升一倍，符合预期的收益。
> -   FixPipe优化：从Profiling数据看出FixPipe占比0.8，优化后占比0.55，实测算子性能提升45%，与理论分析一致。
> -   MTE2优化：从Profiling数据看出MTE2占比0.52，优化后占比减少一半，实测算子性能提升30%，与理论分析一致。
>
> ## 总结
>
> 融合算子场景，可参考此优化。

### 21.2 RULED OUT: `SparseFlashAttention`'s workspace offsets are already 512B-aligned

The FixPipe pathology above (`workspace地址未512B对齐` → `很多异常的128B搬运` → FixPipe at 81%)
is a candidate for our gap, so it was checked. It does not apply.
`attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h`:

```cpp
    // workspace 内存排布
    // |Q--|mm1ResGm(存S)|vec1ResGm(存A1,A2)|mm2ResGm(存O)|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T*)(workspace + offset +
                            aiCoreIdx * dbWorkspaceRatio * sfaKernelConstInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * sfaKernelConstInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T*)(workspace + offset +
                       aiCoreIdx * dbWorkspaceRatio * sfaKernelConstInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * sfaKernelConstInfo.mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T*)(workspace + offset +
                            aiCoreIdx * dbWorkspaceRatio * sfaKernelConstInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * sfaKernelConstInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(mm2ResGm.GetPhyAddr(0)));

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // s2  d+rope bufNum
        kvMergeGm_.SetGlobalBuffer((__gm__ KV_T*)(workspace + offset + aiCoreIdx * 512 * 576 * 4 * sizeof(KV_T)));
        offset += GetBlockNum() * 512 * 576 * 4 * sizeof(KV_T);

        kvValidSizeGm_.SetGlobalBuffer(
            (__gm__ int32_t*)(workspace + offset + (aiCoreIdx * 2) * 128 * 4 * sizeof(int32_t)));
    }
```

`grep -n 'ADDR_ALIGN\|AlignUp\|512) \* 512' op_kernel/arch22/*.h` returns nothing — there is no
explicit alignment step. But the alignment holds structurally:

- `mmResUbSize = sInnerSizeAlign_ * Align(sfaCubeMSize, 16U)` where `sInnerSizeAlign_ =
  Align(sInnerSize_, BYTE_BLOCK=32)` (section 8.1). So `mmResUbSize` is a multiple of
  `32 * 16 = 512` elements, and every `offset +=` term is `GetBlockNum() * 2 * (512k) *
  sizeof(T)` — a multiple of 512 bytes for any `sizeof(T) >= 1`.
- `bmm2ResUbSize = headDimAlign_ * Align(sfaCubeMSize, 16U)`, same argument with
  `headDimAlign_ = Align(512, 32) = 512`.
- `kvMergeGm_`'s per-core stride is `512 * 576 * 4 * 2 = 2,359,296` bytes, and its total is
  `GetBlockNum()` times that. Both 512 B multiples.
- `kvValidSizeGm_`'s per-core stride is `2 * 128 * 4 * 4 = 4096` bytes.

At our shape (`sInnerSizeAlign_ = 512`, `Align(gSize=4, 16) = 16`): `mmResUbSize = 8192` elements,
so `mm1ResGm`'s span is `24 * 2 * 8192 * 4 = 1,572,864` B; `vec1ResGm` 786,432 B; `mm2ResGm`
1,572,864 B; `kvMergeGm_` 56,623,104 B. Every boundary is 512 B-aligned. So the
`很多异常的128B搬运` / FixPipe-at-81% mechanism is **not** a contributor here, and the 45% that
fixing it was worth in the vendor case is not available to us.

### 21.3 RULED OUT: the measured family is the attention op, not the indexer

GLM-5.2 runs the lightning indexer on 21 of 78 layers and `SparseFlashAttention` on all 78
(section 13.1). They are separate operators: `attention/lightning_indexer`,
`attention/quant_lightning_indexer`, `attention/lightning_indexer_v2` in `ops-transformer`, and
`csrc/attention/lightning_indexer_v2`, `quant_lightning_indexer_v2`, `msa_index_score` in
vllm-ascend. A per-operator-family profile attributing 6.973 ms to `SparseFlashAttention`
therefore excludes the indexer, and the indexer's cost is a separate line item not covered by
this document.

## Relevance

Two candidate causes eliminated, and one calibration for what the remaining levers are worth.

**The load-balance lever's yield is bounded by the makespan ratio, and the vendor states the
rule.** `优化前负载最多的核计算量减少的倍数，即为预期提升的性能` — the speedup equals the factor
by which the most-loaded core's work shrinks. Their case: 8 blocks → 4 blocks = 2x. Ours: with
`T1 = 4` base blocks on 4 cores (section 14) and 2048 selected tokens split 512-per-S2-block into
4 sequential `sInnerLoopTimes` iterations on each, the most-loaded core does 4 S2 iterations. A
perfect S2 split over 24 cores would make that 1 iteration on 16 cores (bounded by the 512-token
floor of section 10.2, `2048/512 = 4` useful splits per head-group x 4 head-groups = 16). So the
makespan ratio is **4x**, not 24x — and PR #1741's `flash-decoding 拆分经定量推演为负收益`
(section 18.1) says the combine traffic may eat it. A 4x makespan gain against a combine that
PR #1741 measured at 7x the gather bytes is genuinely uncertain, and that uncertainty is the
honest state of the parallelism lever.

**The two cheap diagnostic candidates are dead.** Workspace misalignment (21.2) and
indexer-cost-attributed-to-attention (21.3) are both eliminated by inspection, so neither explains
any part of the 5.0x. That leaves the four causes of section 17 with their measured weights from
section 20, and the ranking stands.

**The `BSH → BNSD` lever has an analogue we should check.** `将输入数据排布格式从BSH更改为BNSD后，
数据搬运连续，不需要跳地址读取数据，搬运效率提升一倍` — worth 30% in the vendor case for a 0.52
`mte2_ratio`. Our KV layout is fixed at `PA_BSND` by the op (`layout_kv ... 当前仅支持
PageAttention排布"PA_BSND"`, section 14.3), so the layout itself is not ours to change. But the
same principle — `不需要跳地址读取数据` — is exactly what `sparse_block_size > 1` buys
(section 16): consecutive selected tokens become one contiguous run instead of a jump per token.
That is the third independent line of evidence for the block-granularity lever.

---

## Gaps (sections 8-21)

What was looked for and not found, with the exact URLs tried.

**Not found: any public statement framing sparse attention as N× off a memory roofline on
Ascend.** `gh search issues --repo vllm-project/vllm-ascend "roofline"` returns 0 results.
`gh search issues "ascend sparse attention performance"` (global) returns 0. The roofline framing
in our measurement has no public counterpart; the public framing is always 离散访存 /
instruction-count / 块数≈核数.

**Not found: `aic_mte2_time` / `aiv_mte2_time` / `cube_utilization` numbers for *sparse* attention
decode.** `gh search issues --repo vllm-project/vllm-ascend "mte2"` returns exactly one result
(#4167, quoted in section 18 context as a dense-FFN report). The closest numbers for *any*
small-batch attention are flash-attention-npu #325 (B=1, dense FIA, `aic_time` 47.70 us,
`aiv_vec` 5.91 us = 11%, `wait_ib` 18.80 us, `icache_miss_rate` 0.138, ~21 us fixed floor, 4 of 20
cores used) — a different operator on a different code base, so not transferable to
`SparseFlashAttention`.

**`mte2_iq_full_ratio` is not collectible on A3**, so the instruction-issue bottleneck that
section 20.1 measures at 2.48x cannot be confirmed directly in our own profile. Confirmed absent
from CANN 8.0.0's `ai_core_utilization` ResourceConflictRatio table
(https://www.hiascend.com/doc_center/source/zh/canncommercial/800/devaids/devtools/profiling/atlasprofiling_16_0069.html);
present only in CANN 5.0.4.6
(https://www.hiascend.com/doc_center/source/zh/canncommercial/5046/devtools/auxiliarydevtool/atlasprofiling_16_0119.html).

**No corroboration anywhere for a ~13 us MIX launch floor.** Our 0.70 us AI_VECTOR_CORE floor is
reproduced exactly (0.67 us at 2 AIV, https://hpc101.zjusct.io/lab/Lab3.5-AscendC-Op/), but the
same linear model predicts single-digit microseconds for MIX_AIC_1_2, not 13.58. The closest
public figure in magnitude is `平均每层的Task平均等待的开销约为18us左右` on a DeepSeek-V3 decode
layer (https://cann.csdn.net/69afcc5554b52172bc603633.html), which is scheduling wait rather than
core-start cost. Treat 13.58 us as a real but unexplained measurement.

**The `bottleneck_analysis.md` and `optimization_log.md` referenced by tilelang-ascend PR #1741 as
the source of its per-family quantitative analysis are not in the repository on any branch.**
`gh search code --repo tile-ai/tilelang-ascend "bottleneck_analysis"` returns zero results. So the
load-bearing claim `flash-decoding 拆分经定量推演为负收益` (section 18.1) is a conclusion whose
derivation is not public. It is the single most consequential unverifiable statement in sections
8-21, because it contradicts the recommendation of sections 8, 10 and 12.

**No A/B measurement of `sparse_block_size = 1` versus `> 4` anywhere.** The attribute is
documented (section 16.3), the `C_TEMPLATE` path that it selects is compiled (section 16), and
`MergeKv` is verified to be skipped under it (section 16) — but no published run compares them,
and no accuracy delta for block-wise selection on GLM-5.2 was found. This is the cheapest
high-value experiment identified and it is entirely unmeasured publicly.

**No measurement of `SparseFlashMla` + `SparseFlashMlaMetadata` against `SparseFlashAttention` on
the same shape.** Both are A3-supported (sections 11.1, 10.5) and differ on the S2 split, but no
A/B exists. Whether the A3 `仅支持SWA模板` restriction (section 11.2) can express a pure top-k DSA
pattern is unresolved.

**`zhuanlan.zhihu.com/p/2074998277100017262`** ("Qwen3.8-27B 在昇腾 950PR 单卡推理测试",
2026-08-23) returns HTTP 403 to `curl` with desktop and mobile UA and to a fetcher; the tardis
mirrors `https://www.zhihu.com/tardis/zm/art/2074998277100017262` and `.../tardis/bd/art/...`
return HTTP 200 with a 1255-byte stub. Search snippets indicate it contains a per-operator msprof
breakdown including `FusedInferAttentionScore` at 15.3% of kernel time with cube utilization 56.4%
flagged as having optimization headroom — the closest public analogue of our own table — but it
could not be read verbatim and is therefore not quoted.

**`hiascend.com/document/detail/...` is unreadable by a fetcher**: returns HTTP 200 with body
`Please Enable JavaScript and Cookie.` Confirmed for `atlasprofiling_16_0069.html`,
`atlas_ascendc_best_practices_10_0041.html`, `atlas_ascendc_best_practices_10_0035.html`. The
`doc_center/source/...` and `developer/techArticles/...` paths do work; the metric definitions in
section 20.7 came from the MindStudio 7.0.RC1 `doc_center` path.

**`asc.gitcode.com/api/` is HTTP 404**; the index is at `https://asc.gitcode.com/api/README.html`
(381 KB link list). `asc.gitcode.com/api`, `/api/index.html`, `/sitemap.xml`,
`/api/sitemap.xml` all 404.

**Other hard blocks:** `https://blog.csdn.net/m0_71340392/article/details/141166456` HTTP 521
(the same content is at `https://bbs.huaweicloud.cn/blogs/433080`, used instead);
`https://blog.csdn.net/julia4scientist/article/details/154120883` HTTP 521 / ETIMEDOUT — the only
source claiming a measured `block_size` 16→128 sweep (~36% throughput, NPU util 91%, Qwen3-8B on
Atlas 800 A2), **unverified, not cited**; `https://juejin.cn/post/7685816258889023526` and
`.../7379147394755018792` are JS challenge pages; `https://vllm.ai/blog/2025-09-29-deepseek-v3-2`
HTTP 302 into a corporate proxy block page at `114.114.114.114:9421` (worked around via the post's
git source on raw.githubusercontent.com); `https://huggingface.co/THUDM/GLM-4.5/raw/main/config.json`
HTTP 401 (canonical org is `zai-org`); `https://www.remio.ai/zh/post/glm-5-3-sparse-attention-cuts-compute-but-hbm-demand-persists-zh`
is a 1.5 MB Wix JS shell with zero extractable prose.

**`bbs.huaweicloud.com` yielded nothing**: no forum thread surfaced in any search on
稀疏注意力 昇腾 性能, DSA 昇腾, 昇腾 离散访存 性能, or 昇腾 稀疏注意力 decode 小batch. The one
usable Huawei-forum source was `bbs.huaweicloud.cn/blogs/433080` (Matmul tuning, found by other
means).

**No MindIE sparse-attention decode material found anywhere** — on gitee, gitcode or hiascend.
Searched 昇腾 MindIE 稀疏注意力 性能 decode 小batch and the gitee equivalents.

**`gitcode.com/liang_yue/cann-ops-adv-SparseDecodeAttention` renders fully but contains no
performance numbers at all** — no latency, no speedup, no utilization. Its `op_eval/` directory is
described as 性能评估 with no published results. Useful only for architectural claims
(IFA extended to sparse; `减少 CPU 同步`; `降低 kernel launch 开销：单次 aclOpExecutor 调用替代两次独立算子调用`).

**vllm-ascend PRs #12807 ("Sfa topk index cache prefetch") and #15642 ("fused sparse attention
overlap decode") have entirely empty PR descriptions** — only a bot summary. No measurements for
the `FusedSparseAttentionOverlap` of section 14, so its Selection-KV-Cache hit rate and its actual
speedup are unquantified publicly.

**`https://github.com/vllm-project/vllm-ascend/issues/11801` (GLM5系列模型性能基线) was never
answered** — triage asked for config and the thread ends with a request to follow up. There is no
public GLM-5 performance baseline on Ascend to compare our 10.8528 ms against.

**Unsourced claim, explicitly flagged:** a search snippet asserted that sparse-attention kernels
show "vector-compute pipe ratio ~9.5%". Two follow-up searches failed to locate the originating
document. The 9.5% figure is **not** sourced and is not relied on above.

**Figure-only data:** `https://github.com/vllm-project/vllm-ascend/issues/7562`'s per-stage
microsecond latencies for the DSA IndexerSelect pipeline are in an attached PNG
(`user-attachments/assets/e43553ab-...`) and not recoverable as text. The bandwidth-vs-block-size
values in section 20.4 are a reading of vendor chart images, not vendor-printed numbers, and are
labelled as such.

**Reproduction note.** Both CANN doc/operator trees are public git and the default clone is
sparse, omitting `docs/` and `examples/`:

```
git clone --depth 1 https://gitcode.com/cann/ops-transformer.git && git -C ops-transformer sparse-checkout disable
git clone --depth 1 https://gitcode.com/cann/asc-devkit.git     && git -C asc-devkit     sparse-checkout disable
```

Heads used here: `ops-transformer` `3efc65dad2a97a4fea856c3be319da596a42fb8a` (2026-10-20 16:03
+0800) for sections 8-17 and 21.2; `asc-devkit` `5f50fc4c6adab4bf891de229f4ff69b9d49e1897`
(2026-10-20) for section 20. A second reader independently cloned `ops-transformer` at
`d7e6a7c6c1450c040049486845336c77f3100406` the same day; the quoted files are identical.
`raw.gitcode.com` refuses unregistered traffic with `429 Limited account:unregisteredUser` and the
`gitcode.com/.../raw/master/...` form returns a JavaScript shell, so cloning is the only reliable
route.
