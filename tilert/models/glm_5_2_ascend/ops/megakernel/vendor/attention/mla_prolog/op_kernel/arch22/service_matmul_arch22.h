

#ifndef SERVICE_MATMUL_ARCH22_H
#define SERVICE_MATMUL_ARCH22_H

#include "mla_prolog_comm_arch22.h"

namespace MlaProlog {

template <typename SrcT>
__aicore__ inline constexpr uint32_t GetC0Num()
{
    if (sizeof(SrcT) == sizeof(int8_t)) {
        return 32;
    } else if (sizeof(SrcT) == sizeof(float)) {
        return 8;
    }
    return 16;
}

constexpr uint8_t UNIT_FLAG_DISABLE = 0;
constexpr uint8_t UNIT_FLAG_SET = 0b11;
constexpr uint8_t UNIT_FLAG_CHECK = 0b10;


template <typename T, typename O_L0C>
struct mmLocalTensors {


    __aicore__ inline void Init(const MMBufParams& bufParam)
    {
        aL0Tensor.SetAddr(bufParam.aL0BufAddr);
        bL0Tensor.SetAddr(bufParam.bL0BufAddr);
        cL0Tensor.SetAddr(bufParam.cL0BufAddr);
        aL1Tensor.SetAddr(bufParam.aL1BufAddr);
        bL1Tensor.SetAddr(bufParam.bL1BufAddr);
    }

    LocalTensor<T> aL0Tensor;
    LocalTensor<T> bL0Tensor;
    LocalTensor<O_L0C> cL0Tensor;
    LocalTensor<T> aL1Tensor;
    LocalTensor<T> bL1Tensor;
};

template <typename T>
__aicore__ inline void CopyNZGmToL1(LocalTensor<T>& l1Tensor, const GlobalTensor<T>& gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcNstride)
{
    DataCopyParams param;
    param.blockCount = CeilDivT(srcD, GetC0Num<T>());
    param.blockLen = srcN;
    param.dstStride = 0;
    param.srcStride = (srcNstride - srcN);
    DataCopy(l1Tensor, gmSrcTensor, param);
}

template <typename T>
__aicore__ inline void CopyNDGmToL1(LocalTensor<T>& l1Tensor, const GlobalTensor<T>& gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.nValue = srcN;
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = CeilDivT(srcN, BLOCK_CUBE_SIZE) * BLOCK_CUBE_SIZE;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.ndNum = 1;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename T, bool preload = false>
__aicore__ inline void LoadL1A(const GlobalTensor<T>& tensorAGm, const uint32_t mInput, const uint32_t kL1StepSize,
                               const uint32_t kInput, MMBufParams& bufParam)
{
    uint32_t bufIter = bufParam.aL1BufIter;
    if constexpr (preload) {
        bufIter += 1;
    }
    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam.aL1BufAddr);
    uint32_t aL1Offset = L1_A_SIZE / sizeof(T);

    AscendC::WaitFlag<HardEvent::MTE1_MTE2>((bufIter & 1u) + A_EVENT0);
    LocalTensor<T> aL1 = aL1Tensor[(bufIter & 1u) * aL1Offset];
    CopyNDGmToL1(aL1, tensorAGm, mInput, kL1StepSize, kInput);
    AscendC::SetFlag<HardEvent::MTE2_MTE1>((bufIter & 1u) + A_EVENT0);
}

template <typename T, DataFormat loadFormat = DataFormat::NZ, bool preload = false>
__aicore__ inline void LoadL1B(const GlobalTensor<T>& tensorBGm, const uint32_t nL1Size, const uint32_t nInput,
                               const uint32_t kL1StepSize, const uint32_t kInput, MMBufParams& bufParam)
{
    uint32_t bufIter = bufParam.bL1BufIter;
    if constexpr (preload) {
        bufIter += 1;
    }
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam.bL1BufAddr);
    uint32_t bL1Offset = L1_B_SIZE / sizeof(T);

    AscendC::WaitFlag<HardEvent::MTE1_MTE2>((bufIter & 1u) + B_EVENT0);
    LocalTensor<T> bL1 = bL1Tensor[(bufIter & 1u) * bL1Offset];
    if constexpr (loadFormat == DataFormat::NZ) {
        CopyNZGmToL1(bL1, tensorBGm, kL1StepSize, nL1Size, kInput);
    } else {
        CopyNDGmToL1(bL1, tensorBGm, kInput, nL1Size, nInput);
    }
    AscendC::SetFlag<HardEvent::MTE2_MTE1>((bufIter & 1u) + B_EVENT0);
}


template <typename T, bool enTranspose = false>
__aicore__ inline void LoadDataL1ToL0(const LocalTensor<T>& l0Tensor, const LocalTensor<T>& l1Tensor,
                                      const uint32_t mSize, const uint32_t kSize, const uint32_t l1StepSize = 0)
{
    constexpr uint32_t FMATRIX_HEIGHT_UNIT = 16;
    LocalTensor<T> srcTensor = l1Tensor;
    LoadData3DParamsV2<T> loadData3DV2;

    if constexpr (enTranspose) {

        loadData3DV2.l1H = l1StepSize / FMATRIX_HEIGHT_UNIT;
        loadData3DV2.l1W = FMATRIX_HEIGHT_UNIT;
        loadData3DV2.enTranspose = true;
    } else {

        loadData3DV2.l1H = mSize / FMATRIX_HEIGHT_UNIT;
        loadData3DV2.l1W = FMATRIX_HEIGHT_UNIT;
    }
    loadData3DV2.channelSize = kSize;
    loadData3DV2.mStartPt = 0;
    loadData3DV2.kStartPt = 0;
    loadData3DV2.mExtension = mSize;
    loadData3DV2.kExtension = kSize;

    LoadData<T>(l0Tensor, srcTensor, loadData3DV2);
}

template <typename T>
__aicore__ inline void LoadDataL1ToL0B(const LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                       const uint32_t kSize, const uint32_t nSize, const uint32_t kL1StepSize = 0)
{
    constexpr uint32_t LOAD_WIDTH = 32;
    constexpr uint32_t STRIDE_HEIGHT = 32;
    constexpr uint32_t DATA_BYTES_PER_REPEAT = 32;

    uint32_t kloops = kSize / (LOAD_WIDTH / sizeof(T));
    LocalTensor<T> srcTensor = bL1Tensor;
    for (uint32_t i = 0; i < kloops; i++) {
        LoadData2dTransposeParams loadData2DParams;
        loadData2DParams.repeatTimes = nSize / (DATA_BYTES_PER_REPEAT / sizeof(T));
        loadData2DParams.startIndex = 0;
        loadData2DParams.srcStride = kL1StepSize / STRIDE_HEIGHT;
        loadData2DParams.dstFracGap = 0;
        loadData2DParams.dstGap = 1;
        LocalTensor<T> tmpSrcTensor;
        tmpSrcTensor = srcTensor[i * STRIDE_HEIGHT * (DATA_BYTES_PER_REPEAT / sizeof(T))];

        LoadDataWithTranspose(bL0Tensor[i * STRIDE_HEIGHT * nSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename T, typename O, typename S>
__aicore__ inline void MatmulL0(MMBufParams& bufParam, const LocalTensor<T>& aL1, const LocalTensor<T>& bL1,
                                const LocalTensor<T>& aL0Tensor, const LocalTensor<T>& bL0Tensor,
                                const LocalTensor<O>& cL0Tensor, const MmadParams& mmadParams,
                                const uint32_t kL1StepSize, const uint32_t kL1Size = 0)
{
    AscendC::WaitFlag<HardEvent::M_MTE1>((bufParam.aL0BufIter & 1u) + L0A_EVENT0);
    LocalTensor<T> aL0 = aL0Tensor[(bufParam.aL0BufIter & 1u) * (L0A_PP_SIZE / sizeof(T))];

    LoadDataL1ToL0<T>(aL0, aL1, mmadParams.m, mmadParams.k);

    AscendC::SetFlag<HardEvent::MTE1_M>((bufParam.aL0BufIter & 1u) + L0A_EVENT0);
    AscendC::WaitFlag<HardEvent::MTE1_M>((bufParam.aL0BufIter & 1u) + L0A_EVENT0);
    AscendC::WaitFlag<HardEvent::M_MTE1>((bufParam.bL0BufIter & 1u) + L0B_EVENT0);

    LocalTensor<T> bL0 = bL0Tensor[(bufParam.bL0BufIter & 1u) * (L0B_PP_SIZE / sizeof(T))];
    if constexpr (std::is_same<T, int8_t>::value) {
        LoadDataL1ToL0B(bL0, bL1, mmadParams.k, mmadParams.n, kL1StepSize);
    } else {
        LoadDataL1ToL0<T, true>(bL0, bL1, mmadParams.k, mmadParams.n, kL1StepSize);
    }

    AscendC::SetFlag<HardEvent::MTE1_M>((bufParam.bL0BufIter & 1u) + L0B_EVENT0);
    AscendC::WaitFlag<HardEvent::MTE1_M>((bufParam.bL0BufIter & 1u) + L0B_EVENT0);
    Mmad(cL0Tensor, aL0, bL0, mmadParams);
    PipeBarrier<PIPE_M>();
    AscendC::SetFlag<HardEvent::M_MTE1>((bufParam.bL0BufIter & 1u) + L0B_EVENT0);
    bufParam.bL0BufIter++;
    AscendC::SetFlag<HardEvent::M_MTE1>((bufParam.aL0BufIter & 1u) + L0A_EVENT0);
    bufParam.aL0BufIter++;
}

template <typename T, typename O, typename O_L0C, bool enUnitFlag = false>
__aicore__ inline void GetTensorC(const GlobalTensor<O>& tensorCGm, const LocalTensor<O_L0C>& cL0, const uint32_t mSize,
                                  const uint32_t nSize, const uint32_t srcStride, const uint32_t dstStride,
                                  MMBufParams& bufParam)
{
    FixpipeParamsV220 fixParams;
    fixParams.srcStride = srcStride;
    fixParams.dstStride = dstStride;
    fixParams.nSize = nSize;
    fixParams.mSize = mSize;
    fixParams.ndNum = 1;
    fixParams.unitFlag = enUnitFlag ? UNIT_FLAG_SET : UNIT_FLAG_DISABLE;
    if constexpr (std::is_same<O_L0C, float>::value && std::is_same<O, bfloat16_t>::value) {
        fixParams.quantPre = QuantMode_t::F322BF16;
    }
    if constexpr (!enUnitFlag) {
        AscendC::SetFlag<HardEvent::M_FIX>((bufParam.cL0BufIter & 1u) + L0C_EVENT0);
        AscendC::WaitFlag<HardEvent::M_FIX>((bufParam.cL0BufIter & 1u) + L0C_EVENT0);
    }
    Fixpipe(tensorCGm, cL0, fixParams);
}


template <typename T, bool isContinuousCopy>
__aicore__ inline void LoadL1BGroupCompute(LocalTensor<T>& bL1, const GlobalTensor<T>& tensorBGm,
                                           uint32_t subNL1SplitSize, const MMParams& para, int64_t nL1, uint32_t kL1,
                                           uint32_t kOffesetUnit)
{
    int64_t tensorBGmOffset = para.k * nL1 * para.baseN + kOffesetUnit * kL1;
    auto tensorBGmForL1 = tensorBGm[tensorBGmOffset];
    if constexpr (isContinuousCopy) {
        CopyNZGmToL1(bL1, tensorBGmForL1, para.kL1StepSize, subNL1SplitSize, para.k);
    } else {

        CopyNZGmToL1(bL1, tensorBGmForL1, para.kL1StepSize, subNL1SplitSize >> 1, para.k);
        LocalTensor<T> b2L1 = bL1[(subNL1SplitSize * para.kL1StepSize) >> 1];
        auto tensorB2GmForL1 = tensorBGmForL1[DIM_HEAD_SIZE_QCQR * para.k];
        CopyNZGmToL1(b2L1, tensorB2GmForL1, para.kL1StepSize, subNL1SplitSize >> 1, para.k);
    }
}


template <typename T, bool hasL1ALoaded, DataFormat bLoadFormat = DataFormat::NZ>
__aicore__ inline void LoadL1AB(const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm, uint32_t kL1,
                                uint32_t kL1Loops, const MMParams& para, uint32_t nL1Offset, uint32_t nL1Size,
                                uint32_t kOffesetUnit, MMBufParams& bufParam)
{
    if (kL1 == 0) {
        if constexpr (!hasL1ALoaded) {
            LoadL1A(tensorAGm[para.kL1StepSize * kL1], para.m, para.kL1StepSize, para.orgKa, bufParam);
        }
        if constexpr (bLoadFormat == DataFormat::NZ) {
            LoadL1B(tensorBGm[para.k * nL1Offset + kOffesetUnit * kL1], nL1Size, para.n, para.kL1StepSize, para.k,
                    bufParam);
        } else {
            LoadL1B<T, DataFormat::ND, false>(tensorBGm[para.kL1StepSize * kL1 * para.n + nL1Offset], nL1Size, para.n,
                                              para.kL1StepSize, para.kL1StepSize, bufParam);
        }
    }

    if (kL1 + 1 < kL1Loops) {
        if constexpr (!hasL1ALoaded) {
            LoadL1A<T, true>(tensorAGm[para.kL1StepSize * (kL1 + 1)], para.m, para.kL1StepSize, para.orgKa, bufParam);
        }
        if constexpr (bLoadFormat == DataFormat::NZ) {
            LoadL1B<T, DataFormat::NZ, true>(tensorBGm[para.k * nL1Offset + kOffesetUnit * (kL1 + 1)], nL1Size, para.n,
                                             para.kL1StepSize, para.k, bufParam);
        } else {
            LoadL1B<T, DataFormat::ND, true>(tensorBGm[para.kL1StepSize * (kL1 + 1) * para.n + nL1Offset], nL1Size,
                                             para.n, para.kL1StepSize, para.kL1StepSize, bufParam);
        }
    }
}


template <typename T, typename O, typename S, typename O_L0C, bool enUnitFlag>
__aicore__ inline void MatmulL1(const LocalTensor<O_L0C>& cL0, const LocalTensor<T>& aL1, const LocalTensor<T>& bL1,
                                const mmLocalTensors<T, O_L0C>& localTensors, MMBufParams& bufParam,
                                const MMParams& para, uint32_t kL1, uint32_t kL1Loops, MmadParams& mmadParams,
                                int64_t& aOffset)
{
    uint32_t bOffset = 0;
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    uint32_t aOffsetUnit = mSize * para.baseK;
    uint32_t bOffsetUnit = GetC0Num<T>() * para.baseK;

    for (int64_t kL0Loops = 0; kL0Loops < para.stepK; kL0Loops++) {
        mmadParams.cmatrixInitVal = ((kL0Loops == 0) && (kL1 == 0));
        if constexpr (enUnitFlag) {
            mmadParams.unitFlag =
                (kL1 == kL1Loops - 1) && (kL0Loops == para.stepK - 1) ? UNIT_FLAG_SET : UNIT_FLAG_CHECK;
        }
        MatmulL0<T, O_L0C, S>(bufParam, aL1[aOffset], bL1[bOffset], localTensors.aL0Tensor, localTensors.bL0Tensor, cL0,
                              mmadParams, para.kL1StepSize, para.k);
        bOffset += bOffsetUnit;
        aOffset += aOffsetUnit;
    }
}


template <typename T, typename O, typename S, bool hasL1ALoaded = false, bool enUnitFlag = false,
          DataFormat bLoadFormat = DataFormat::NZ>
__aicore__ inline void MatmulSplitK(const GlobalTensor<O>& tensorCGm, const GlobalTensor<T>& tensorAGm,
                                    const GlobalTensor<T>& tensorBGm, const MMParams& para, MMBufParams& bufParam,
                                    const uint32_t nL1Offset, const uint32_t nL1Size)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;


    uint32_t kOffesetUnit = para.kL1StepSize * GetC0Num<T>();
    uint32_t kL1Loops = CeilDivT(para.k, para.kL1StepSize);
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);

    mmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    LocalTensor<T> aL1, bL1;

    if constexpr (hasL1ALoaded) {
        aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * L1_A_SIZE / sizeof(T)];
    }

    AscendC::WaitFlag<HardEvent::FIX_M>((bufParam.cL0BufIter & 1u) + L0C_EVENT0);
    LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];

    MmadParams mmadParams = MmadParams(mSize, nL1Size, para.baseK, UNIT_FLAG_DISABLE, false, true);

    int64_t aOffset = 0;
    for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {

        LoadL1AB<T, hasL1ALoaded, bLoadFormat>(tensorAGm, tensorBGm, kL1, kL1Loops, para, nL1Offset, nL1Size,
                                               kOffesetUnit, bufParam);

        AscendC::WaitFlag<HardEvent::MTE2_MTE1>((bufParam.bL1BufIter & 1u) + B_EVENT0);
        bL1 = localTensors.bL1Tensor[(bufParam.bL1BufIter & 1u) * L1_B_SIZE / sizeof(T)];
        if constexpr (!hasL1ALoaded) {
            AscendC::WaitFlag<HardEvent::MTE2_MTE1>((bufParam.aL1BufIter & 1u) + A_EVENT0);
            aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * L1_A_SIZE / sizeof(T)];
            aOffset = 0;
        }


        MatmulL1<T, O, S, O_L0C, enUnitFlag>(cL0, aL1, bL1, localTensors, bufParam, para, kL1, kL1Loops, mmadParams,
                                             aOffset);

        AscendC::SetFlag<HardEvent::MTE1_MTE2>((bufParam.bL1BufIter & 1u) + B_EVENT0);
        bufParam.bL1BufIter++;
        if constexpr (!hasL1ALoaded) {
            AscendC::SetFlag<HardEvent::MTE1_MTE2>((bufParam.aL1BufIter & 1u) + A_EVENT0);
            bufParam.aL1BufIter++;
        }
    }
    GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[nL1Offset], cL0, para.m, nL1Size, mSize, para.orgKc, bufParam);
    AscendC::SetFlag<HardEvent::FIX_M>((bufParam.cL0BufIter & 1u) + L0C_EVENT0);
    bufParam.cL0BufIter++;
}


template <typename T, typename O, typename S, bool isContinuousCopy = true, bool enUnitFlag = false>
__aicore__ inline void MatmulGroupComputeAFullLoad(const GlobalTensor<O>& tensorCGm, const GlobalTensor<T>& tensorAGm,
                                                   const GlobalTensor<T>& tensorBGm, const MMParams& para,
                                                   MMBufParams& bufParam)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

    uint32_t kOffesetUnit = GetC0Num<T>() * para.kL1StepSize;
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    uint32_t nL1loops = CeilDivT(para.n, para.baseN);
    uint32_t kL1Loops = CeilDivT(para.k, para.kL1StepSize);

    mmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    AscendC::WaitFlag<HardEvent::MTE1_MTE2>((bufParam.aL1BufIter & 1u) + A_EVENT0);
    LocalTensor<T> aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * (L1_A_SIZE / sizeof(T))];
    CopyNDGmToL1(aL1, tensorAGm, para.m, para.k, para.k);
    AscendC::SetFlag<HardEvent::MTE2_MTE1>((bufParam.aL1BufIter & 1u) + A_EVENT0);
    AscendC::WaitFlag<HardEvent::MTE2_MTE1>((bufParam.aL1BufIter & 1u) + A_EVENT0);

    uint32_t subNL1SplitSize = para.baseN;

    MmadParams mmadParams = MmadParams(mSize, subNL1SplitSize, para.baseK, UNIT_FLAG_DISABLE, false, true);

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = para.n - para.baseN * (nL1loops - 1);
            mmadParams.n = subNL1SplitSize;
        }

        AscendC::WaitFlag<HardEvent::FIX_M>((bufParam.cL0BufIter & 1u) + L0C_EVENT0);
        LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];

        int64_t aOffset = 0;
        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            AscendC::WaitFlag<HardEvent::MTE1_MTE2>((bufParam.bL1BufIter & 1u) + B_EVENT0);
            LocalTensor<T> bL1 = localTensors.bL1Tensor[(bufParam.bL1BufIter & 1u) * (L1_B_SIZE / sizeof(T))];
            LoadL1BGroupCompute<T, isContinuousCopy>(bL1, tensorBGm, subNL1SplitSize, para, nL1, kL1, kOffesetUnit);
            AscendC::SetFlag<HardEvent::MTE2_MTE1>((bufParam.bL1BufIter & 1u) + B_EVENT0);
            AscendC::WaitFlag<HardEvent::MTE2_MTE1>((bufParam.bL1BufIter & 1u) + B_EVENT0);


            MatmulL1<T, O, S, O_L0C, enUnitFlag>(cL0, aL1, bL1, localTensors, bufParam, para, kL1, kL1Loops, mmadParams,
                                                 aOffset);
            AscendC::SetFlag<HardEvent::MTE1_MTE2>((bufParam.bL1BufIter & 1u) + B_EVENT0);
            bufParam.bL1BufIter++;
        }
        GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[nL1 * para.baseN], cL0, para.m, subNL1SplitSize, mSize,
                                            para.orgKc, bufParam);
        AscendC::SetFlag<HardEvent::FIX_M>((bufParam.cL0BufIter & 1u) + L0C_EVENT0);
        bufParam.cL0BufIter++;
    }
    AscendC::SetFlag<HardEvent::MTE1_MTE2>((bufParam.aL1BufIter & 1u) + A_EVENT0);
    bufParam.aL1BufIter++;
}


template <typename T, typename O, typename S, bool hasL1BLoaded = false, bool enUnitFlag = false>
__aicore__ inline void MatmulFullLoad(const GlobalTensor<O>& tensorCGm, const GlobalTensor<T>& tensorAGm,
                                      const GlobalTensor<T>& tensorBGm, const MMParams& para, MMBufParams& bufParam)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;
    constexpr uint32_t aL1Offset = L1_A_SIZE / sizeof(T);
    constexpr uint32_t bL1Offset = L1_B_SIZE / sizeof(T);
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);

    mmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    LoadL1A(tensorAGm, para.m, para.k, para.orgKa, bufParam);
    AscendC::WaitFlag<HardEvent::MTE2_MTE1>((bufParam.aL1BufIter & 1u) + A_EVENT0);
    LocalTensor<T> aL1 = localTensors.aL1Tensor[aL1Offset * (bufParam.aL1BufIter & 1u)];

    if constexpr (!hasL1BLoaded) {
        LoadL1B<T, DataFormat::ND, false>(tensorBGm, para.n, para.n, para.k, para.k, bufParam);
        AscendC::WaitFlag<HardEvent::MTE2_MTE1>((bufParam.bL1BufIter & 1u) + B_EVENT0);
    }
    LocalTensor<T> bL1 = localTensors.bL1Tensor[bL1Offset * (bufParam.bL1BufIter & 1u)];

    uint32_t nSplitSize = 128;
    uint32_t nloops = CeilDivT(para.n, nSplitSize);
    uint32_t nSplitSizeAct = nSplitSize;

    MmadParams mmadParams = MmadParams(mSize, nSplitSizeAct, para.k, UNIT_FLAG_DISABLE, false, true);

    for (uint32_t n = 0; n < nloops; n++) {
        if (n == nloops - 1) {
            nSplitSizeAct = para.n - nSplitSize * (nloops - 1);
            mmadParams.n = nSplitSizeAct;
        }

        if constexpr (enUnitFlag) {
            mmadParams.unitFlag = UNIT_FLAG_SET;
        }
        AscendC::WaitFlag<HardEvent::FIX_M>((bufParam.cL0BufIter & 1u) + L0C_EVENT0);
        LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];
        MatmulL0<T, O_L0C, S>(bufParam, aL1, bL1[para.k * n * nSplitSize], localTensors.aL0Tensor,
                              localTensors.bL0Tensor, cL0, mmadParams, para.kL1StepSize);
        GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[n * nSplitSize], cL0, para.m, nSplitSizeAct, mSize, para.orgKc,
                                            bufParam);
        AscendC::SetFlag<HardEvent::FIX_M>((bufParam.cL0BufIter & 1u) + L0C_EVENT0);
        bufParam.cL0BufIter++;
    }

    AscendC::SetFlag<HardEvent::MTE1_MTE2>((bufParam.bL1BufIter & 1u) + B_EVENT0);
    bufParam.bL1BufIter++;
    AscendC::SetFlag<HardEvent::MTE1_MTE2>((bufParam.aL1BufIter & 1u) + A_EVENT0);
    bufParam.aL1BufIter++;
}

}

#endif
