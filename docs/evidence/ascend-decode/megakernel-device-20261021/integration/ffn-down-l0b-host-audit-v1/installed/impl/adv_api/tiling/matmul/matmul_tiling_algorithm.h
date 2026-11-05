/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_tiling_algorithm.h
 * \brief
 */

#ifndef IMPL_MATMUL_TILING_MATMUL_TILING_ALGORITHM_H
#define IMPL_MATMUL_TILING_MATMUL_TILING_ALGORITHM_H

#include <cstdint>
#include <string>
#include <algorithm>
#include <climits>
#include <cmath>
#include <array>
#include <vector>
#include "../../../../include/adv_api/matmul/matmul_tiling.h"
#include "../../../../include/adv_api/matmul/matmul_tiling_base.h"

namespace matmul_tiling {
constexpr int32_t DB_ON = 2;
constexpr int32_t DB_OFF = 1;
constexpr int32_t L1_FACTORS_LEN = 6;
constexpr int32_t L0PARAS_COMBO_LEN = 2;
constexpr float MATMUL_EPSILON = 1.0E-4F;
constexpr float MEMORY_EPSILON = 0.05F;

struct MatmulTemplateCfg {
    int32_t l0aDB = DB_ON;
    int32_t l0bDB = DB_ON;
    int32_t l0cDB = DB_OFF;
    int32_t l1DB = DB_ON;
    bool factorSplit = true;
    bool kSplit = false;
};

struct L1StatusPack {
    int32_t kAL1 = 1;
    int32_t kBL1 = 1;
    int32_t mL1 = 1;
    int32_t nL1 = 1;
    int32_t mAL1 = 1;
    int32_t nBL1 = 1;
    int32_t dbAL1 = 1;
    int32_t dbBL1 = 1;
    int32_t aL1Size = 0;
    int32_t bL1Size = 0;
    int32_t aL1Times = 1;
    int32_t bL1Times = 1;
    int32_t allTimes = 1;
    int32_t loadSize = 0;
    int32_t maxMAL1 = 1;
    int32_t maxNBL1 = 1;
    int32_t maxKAL1 = 1;
    int32_t maxKBL1 = 1;
    bool bothFullLoad = false;
    bool aL1FullLoad = false;
    bool bL1FullLoad = false;
    bool aL1KFullLoad = false;
    bool bL1KFullLoad = false;
    int32_t channelWiseTimes = 0;
    void SetStatus(const int32_t (&tmpL1Factors)[L1_FACTORS_LEN])
    {
        this->kAL1 = tmpL1Factors[0];
        this->kBL1 = tmpL1Factors[1];
        this->mAL1 = tmpL1Factors[2];  // 2 means mAL1 factor index
        this->nBL1 = tmpL1Factors[3];  // 3 means nBL1 factor index
        this->dbAL1 = tmpL1Factors[4]; // 4 means dbAL1 factor index
        this->dbBL1 = tmpL1Factors[5]; // 5 means dbBL1 factor index
    }
};

struct L0StatusPack {
    int32_t mL0 = 1;
    int32_t nL0 = 1;
    int32_t kL0 = 1;
    int32_t batchL0 = 1;
    int32_t l0cMultiBatch = 0;
    int32_t dbL0A = 1;
    int32_t dbL0B = 1;
    int32_t dbL0C = 1;
    int32_t dbCub = 1;
    int32_t finalML0 = 0;
    int32_t finalKL0 = 0;
    int32_t finalNL0 = 0;
    int32_t finalLoadSize = INT_MAX;
    float finalL0cUse = 0;
    int32_t finalMul = 0;
    int32_t finalMte1Loop = INT_MAX;
    int32_t finalMte1Cycles = 0;
    int32_t maxMk = 1;
    int32_t maxNk = 1;
    int32_t maxMn = 1;
    int32_t maxAxisIdx = 0;
    int32_t maxAxisNum = 0;
    int32_t maxAxisPnt = 1;
    int32_t maxN = 1;
    int32_t dtypeBias = 0;
    int32_t load2dTimes = 0;
    int32_t l0cUsed = 0;
    bool updateUsingMte1 = false;
    void InitLoadStatus()
    {
        finalML0 = 0;
        finalKL0 = 0;
        finalNL0 = 0;
        finalLoadSize = INT_MAX;
        finalL0cUse = 0;
        finalMul = 0;
        finalMte1Loop = INT_MAX;
        finalMte1Cycles = 0;
        updateUsingMte1 = false;
    }
};

struct CoreStatusPack {
    CoreStatusPack()
    {
        cycle = INT_MAX;
        loadSize = INT_MAX;
        batch = 1;
        batchDim = 1;
        mDim = 1;
        kDim = 1;
        nDim = 1;
    }
    int32_t batch = 1;
    int32_t m = 1;
    int32_t k = 1;
    int32_t n = 1;
    int32_t batchDim = 1;
    int32_t mDim = 1;
    int32_t nDim = 1;
    int32_t kDim = 1;
    int32_t kAl1Factor = 1;
    int32_t kBl1Factor = 1;
    int32_t mSingleCore = 1;
    int32_t nSingleCore = 1;
    int32_t n0Max = 1;
    int64_t cycle = 1;
    int32_t loadSize = 1;
    int32_t aL1FullLoadSize = 0;
    int32_t bL1FullLoadSize = 0;
    int64_t madCycle = 1;
    int64_t repeatLoadSize = INT_MAX;
};

struct SingleCoreStatus {
    L0StatusPack l0Status;
    L1StatusPack l1Status;
    // UbStatus ubStatus for ub status
};

struct L0Factors {
    int32_t finalML0 = 0;
    int32_t finalKL0 = 0;
    int32_t finalNL0 = 0;
    int32_t finalLoadSize = INT_MAX;
    float finalL0cUse = 0;
    int32_t finalMul = 0;
    int32_t finalMte1Loop = INT_MAX;
    int32_t finalMte1Cycles = 0;
};

struct MKNParasCombo {
    int32_t parasCombo[10];
};

struct MatmulRunParas {
    // public:
    int64_t oriShapeM = 1;
    int64_t oriShapeN = 1;
    int64_t oriShapeKa = 1;
    int64_t oriShapeKb = 1;
    int64_t oriShapeAbatch = 1;
    int64_t oriShapeBbatch = 1;
    int32_t dtypeA = 0;
    int32_t dtypeB = 0;
    int32_t dtypeOut = 0;
    int32_t dtypeBias = 0;
    bool transA = false;
    bool transB = false;
    bool formatANd = false;
    bool formatBNd = false;
    bool formatOutNd = false;
    bool biasFlag = false;
    bool hf32Flag = 1;
    // private:
    int32_t batch32 = 1;
    int32_t m32 = 1;
    int32_t k32 = 1;
    int32_t n32 = 1;
    int32_t batch32A = 1;
    int32_t batch32B = 1;
    int32_t mMapped = 1;
    int32_t kMapped = 1;
    int32_t nMapped = 1;
    int32_t batchMapped = 1;
    bool nonFactorK = false;
};

#ifndef ASCC_STRUCT_L1TILINGTYPE
#define ASCC_STRUCT_L1TILINGTYPE
enum class L1TilingType : uint8_t { KAL1_16, KBL1_16, M_AL1, N_BL1 };
#endif

enum class MultiCoreScenario : uint8_t {
    K_FULL_LOAD,
    SPLIT_MN,
    SPLIT_SMALL_MN, // M*N<128*256*corenum*0.8
    OTHERS
};

struct DimCalculator {
    int32_t batch = 1;
    int32_t m = 1;
    int32_t k = 1;
    int32_t n = 1;
    int32_t kNum = 1;
    int32_t kBytes = 1;
    int32_t batchDimFactor = 1;
    int32_t mDimFactor = 1;
    int32_t nDimFactor = 1;
    int32_t kDimFactor = 1;
    int32_t minLoadSize = 1;
    int32_t coreUse = 1;
    int32_t tmpCoreUse = 1;
    int32_t loopNumToL0 = 1;
    int32_t batchIdx = 0;
    int32_t nIdx = 0;
    int32_t batchDimCnt = 0;
    int32_t mDimCnt = 0;
    int32_t nDimCnt = 0;
    int32_t kDimCnt = 0;
    int32_t batchFactorCnt = 0;
    int32_t oriAmatSize = 0;
    int32_t oriBmatSize = 0;
    int32_t amatSize = 0;
    int32_t bmatSize = 0;
    int32_t tmpAmatSize = 0;
    int32_t tmpBmatSize = 0;
    int32_t tmpLoadSize = 0;
    int32_t totalLoadSize = 0;
    int32_t tmpValue = 0;
    int32_t finalValue = 0;

    std::vector<int32_t> batchDimFactors;
    std::vector<int32_t> mDimFactors;
    std::vector<int32_t> nDimFactors;
    std::vector<int32_t> kDimFactors;
    bool initFlag = false;
    bool bigPackage = false;
    DimCalculator()
    {
        batchDimFactors.clear();
        mDimFactors.clear();
        nDimFactors.clear();
        kDimFactors.clear();
        nDimFactor = 1;
        nDimFactor = 1;
        mDimFactor = 1;
        kDimFactor = 1;
        minLoadSize = INT_MAX;
        loopNumToL0 = INT_MAX;
        bigPackage = false;
    }
};

class DimFactor {
public:
    DimFactor() : batch(1), m(1), k(1), n(1), group(1) {}
    DimFactor(int32_t inBatch, int32_t inM, int32_t inK, int32_t inN, int32_t inGroup = 1)
        : batch(inBatch), m(inM), k(inK), n(inN), group(inGroup)
    {}

    int32_t ReduceMul() const { return batch * m * k * n; }

    void Init()
    {
        batch = 1;
        m = 1;
        k = 1;
        n = 1;
        group = 1;
    }

    bool IsValid() const { return group > 0 && batch > 0 && m > 0 && k > 0 && n > 0; }
    int32_t batch = 1;
    int32_t m = 1;
    int32_t k = 1;
    int32_t n = 1;
    int32_t group = 1;
};

// Compute base block
struct ComputeBaseBlock {
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
};

struct ComputeIntensitySmallShape {
    std::pair<int32_t, int32_t> dimFactor;
    float computeCycle;
    float avgIntensity;
    float bandRatio;
    float memoryTraffic;
    ComputeBaseBlock baseBlock;
    bool operator<(const ComputeIntensitySmallShape& rhs) const
    {
        if (std::fabs(rhs.memoryTraffic - memoryTraffic) > MEMORY_EPSILON * memoryTraffic) {
            return memoryTraffic < rhs.memoryTraffic;
        }
        if (std::fabs(computeCycle - rhs.computeCycle) < 0.01f * rhs.computeCycle) {
            return bandRatio > rhs.bandRatio;
        }
        return avgIntensity > rhs.avgIntensity;
    }
};

struct ComputeIntensity {
    std::pair<int32_t, int32_t> dimFactor;
    float computeCycle;
    float avgIntensity;
    float bandRatio;
    bool operator<(const ComputeIntensity& rhs) const
    {
        if (std::fabs(avgIntensity - rhs.avgIntensity) > MATMUL_EPSILON) {
            return avgIntensity > rhs.avgIntensity;
        }
        if (std::fabs(computeCycle - rhs.computeCycle) < 0.01f * rhs.computeCycle) {
            return bandRatio > rhs.bandRatio;
        }
        return computeCycle < rhs.computeCycle;
    }
};

struct MemoryRatios {
    float aMemoryRatio = 1.0f;
    float bMemoryRatio = 1.0f;
    explicit MemoryRatios(float a = 1.0f, float b = 1.0f)
    {
        aMemoryRatio = a;
        bMemoryRatio = b;
    }
};

// record base block compute intensity
struct BaseBlockIntensity {
    int32_t count;
    float computeCycle;
    float memoryTraffic;
};

class MatmulTilingAlgorithm {
public:
    explicit MatmulTilingAlgorithm(MatmulApiTilingBase* tilingIns);
    ~MatmulTilingAlgorithm() { tilingIns_ = nullptr; }
    int64_t Process();
    bool EnableL1BankConflictOptimise() const;

private:
    bool CheckFinalParams(const CoreStatusPack& coreStatus) const;
    bool CheckBaseMN() const;
    int32_t GetBestValue(int32_t base) const;
    int32_t GetC0Size() const;
    int32_t GetIteratorOrder(
        const SingleCoreStatus& singleCoreStatus, const int32_t singleCoreM, const int32_t singleCoreN,
        const int32_t singleCoreK) const;
    void GetL0StatusFromParasCombo(L0StatusPack& l0Status, int32_t* parasCombo) const;
    void GetTwoFactors(int32_t (&res)[2], int32_t base, int32_t dim, int32_t maxNum = 32) const;
    void SetResFactors(L0Factors& resFactors, const L0StatusPack& l0Status) const;
    int32_t GetLoadSize(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status) const;
    void GetFinalMkn(
        SingleCoreStatus& singleCoreStatus, const CoreStatusPack& coreStatus, const int32_t& k0,
        const int32_t& majorDimFactor, const int32_t& minorDimFactor) const;
    bool CheckK0Align(int32_t k0) const;
    void GetL0FactorsCand(
        L0Factors& resFactors, const CoreStatusPack& coreStatus, SingleCoreStatus& singleCoreStatus,
        int32_t* parasCombo, const MatmulRunParas& param) const;
    MKNParasCombo GetParasCombo(const int32_t& index, const MatmulRunParas& param) const;
    void GetL0cDB(
        const L0Factors (&resFactors)[L0PARAS_COMBO_LEN], const CoreStatusPack& coreStatus,
        L0StatusPack& l0Status) const;
    int32_t GetMxCurL1Size(const SingleCoreStatus& singleCoreStatus) const;
    void GetL0Factors(
        const std::string& opType, const MatmulRunParas& param, const CoreStatusPack& coreStatus,
        SingleCoreStatus& singleCoreStatus) const;
    void AdjustSparseL0Factors(SingleCoreStatus& singleCoreStatus) const;
    void AdjustMxL0Factors(SingleCoreStatus& singleCoreStatus) const;
    void UpdateBaseKForMxGemv(int32_t& baseK, SingleCoreStatus& singleCoreStatus) const;
    void AdjustMxL1Factors(SingleCoreStatus& singleCoreStatus) const;
    void FixMxScaleFactorByRange(uint8_t& factor, uint8_t maxFactor) const;
    void FixMxScaleFactorByPosition(
        uint8_t& scaleFactorM, uint8_t& scaleFactorN, uint8_t& scaleFactorKa, uint8_t& scaleFactorKb) const;
    void GetMxScaleSize(int32_t& scaleA1Size, int32_t& scaleB1Size) const;
    void GetMxScaleFactor(const SingleCoreStatus& singleCoreStatus, int32_t& mxTypePara) const;
    void CheckL0DB(SingleCoreStatus& singleCoreStatus, const int32_t baseK) const;
    void CheckL0DB(int32_t baseM, int32_t baseN, int32_t baseK, SingleCoreStatus& singleCoreStatus) const;
    void GetMxUsedL1Size(
        const SingleCoreStatus& singleCoreStatus, int32_t& dataUsedL1Size, int32_t& scaleUsedL1Size,
        int32_t& biasUsedL1Size) const;
    bool AdjustNBuffer33L0Factors(
        const MatmulRunParas& param, const CoreStatusPack& coreStatus, SingleCoreStatus& singleCoreStatus) const;
    bool AdjustNBuffer33L1Factors(const CoreStatusPack& coreStatus, SingleCoreStatus& singleCoreStatus) const;
    bool CheckFixSplitInputs(int32_t singleCoreM) const;
    void CalcBaseShape(
        const SingleCoreStatus& singleCoreStatus, int32_t& baseM, int32_t& baseN, int32_t& baseK,
        int32_t& reduceSize) const;
    bool CheckL0ASize(int32_t singleCoreM, int32_t singleCoreK, int32_t& baseM, int32_t& baseK) const;
    bool CheckL0BSize(int32_t singleCoreN, int32_t singleCoreK, int32_t& baseN, int32_t& baseK) const;
    bool CheckL0CSize(int32_t singleCoreM, int32_t singleCoreN, int32_t& baseM, int32_t& baseN) const;
    int32_t GetNBuffer33L1Size(const SingleCoreStatus& singleCoreStatus) const;
    bool IsNeedAlign(bool isA) const;
    void GetABL1Const(int32_t& aL1Const, int32_t& bL1Const, const L1StatusPack& l1Status) const;
    int32_t GetL1Size(const L1StatusPack& l1Status, const L0StatusPack& l0Status) const;
    int32_t CalL1MaxLen(
        int32_t resL1Size, L1StatusPack& l1Status, const L0StatusPack& l0Status, const int32_t alignValue,
        const L1TilingType axisName) const;
    void GetNearestFactor(const int32_t& base, int32_t& factor, int32_t capValue = INT32_MAX) const;
    void L1StatusAl1FullLoad(
        const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status, int32_t res[][7]) const;
    void L1StatusBl1FullLoad(
        const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status, int32_t res[][7]) const;
    void L1StatusBothFullLoad(
        const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status, int32_t res[][7]) const;
    void NeitherFullLoadDb(
        const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status,
        const int32_t& kbl1Db) const;
    void NeitherFullLoadMN(
        const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status) const;
    void NeitherFullLoadKforNZ(
        const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status) const;
    void NeitherFullLoadKforND(
        const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status,
        const int32_t& kMaxAxis) const;
    void NeitherFullLoadK(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status) const;
    void L1StatusNeitherFullLoad(
        const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status, int32_t res[][7]) const;
    void GetL1Factors(
        const std::string& opType, const MatmulRunParas& param, const CoreStatusPack& coreStatus,
        const L0StatusPack& l0Status, L1StatusPack& l1Status) const;
    bool CheckL1Size(int32_t amat, int32_t bmat, int32_t curBiasL1Size = 0) const;
    void GetUsedSize(
        int32_t& l1Size, int32_t& l0cSize, int32_t& ubSize, int32_t a1LengthCache, int32_t b1LengthCache) const;
    void GetBankConflictSize(int32_t& length, bool isAMatrix) const;
    void GetBankConflictSize(
        const L1StatusPack& l1Status, const L0StatusPack& l0Status, int32_t& length, bool isAMatrix) const;
    int32_t GetAL1UbSize(const L1StatusPack& l1Status, const L0StatusPack& l0Status) const;
    int32_t GetBL1UbSize(const L1StatusPack& l1Status, const L0StatusPack& l0Status) const;
    bool IsUbNd2Nz() const;
    void GetTransLength(int32_t& transLength) const;
    void SetDepthL1CacheUBParams(int32_t& a1LengthCache, int32_t& b1LengthCache) const;
    void GetABL1KAlignValue(int32_t& kaAlignValue, int32_t& kbAlignValue) const;
    bool CheckBaseMNKL1Size(SingleCoreStatus& singleCoreStatus) const;
    void NonFactorMap(const std::string& opType, MatmulRunParas& param, DimCalculator& dimCalRes) const;
    void UpdateDimCalculator(DimCalculator& dimCalRes) const;
    void GetDimsHelper(
        const DimFactor& dimFactor, CoreStatusPack& coreStatus, DimCalculator& dimCalRes, const MatmulRunParas& params);
    void GetDims(
        const std::string& opType, MatmulRunParas& params, CoreStatusPack& coreStatus, DimCalculator& dimCalRes);
    bool PreProcessMiniShape(
        const std::string& opType, CoreStatusPack& coreStatus, MatmulRunParas& params, const int32_t& coreNum,
        bool splitKFlag) const;
    float CalculateBlockCycles(int32_t baseM, int32_t baseN, int32_t baseK) const;
    float CalculateMemoryTraffic(
        int32_t baseM, int32_t baseN, int32_t baseK, float aMemoryRatio, float bMemoryRatio) const;
    bool AlignSingleShape(
        bool needAlign, int32_t orgShape, int32_t factor, int32_t alignSize, int32_t& singleShape) const;
    ComputeIntensity CalcComputeIntensity(
        const MatmulRunParas& params, const ComputeBaseBlock& baseBlock,
        const std::pair<int32_t, int32_t>& factor) const;
    ComputeIntensitySmallShape CalcComputeIntensitySmallShape(
        const MatmulRunParas& params, const std::pair<int32_t, int32_t>& factor, ComputeBaseBlock& baseBlock) const;
    void CalcMultiCoreDims(
        const MatmulRunParas& params, const ComputeBaseBlock& baseBlock, CoreStatusPack& coreStatus,
        DimCalculator& dimCalRes);
    void CalcMultiCoreDimsPost(const MatmulRunParas& params, CoreStatusPack& coreStatus, DimCalculator& dimCalRes);
    void CalcMultiCoreDimsSmallShape(
        const MatmulRunParas& params, ComputeBaseBlock& baseBlock, CoreStatusPack& coreStatus,
        DimCalculator& dimCalRes);
    bool CalcNBuffer33Dims(
        const MatmulRunParas& params, const ComputeBaseBlock& baseBlock, CoreStatusPack& coreStatus) const;
    void UpdateBaseBlock(
        const MatmulRunParas& params, const int32_t sm, const int32_t sn, ComputeBaseBlock& baseBlock) const;
    std::vector<BaseBlockIntensity> CalcTotalCycleMemory(
        const std::pair<int32_t, int32_t>& shapeM, const std::pair<int32_t, int32_t>& shapeN,
        const ComputeBaseBlock& baseBlock, const float memoryRatio,
        const MemoryRatios memoryRatios = MemoryRatios()) const;
    void UpdateMultiCore(
        const std::string& opType, const MatmulRunParas& params, CoreStatusPack& coreStatus,
        const DimCalculator& dimCalRes) const;
    void CalcLoadSize(
        const DimFactor& dimFactor, const CoreStatusPack& coreStatus, DimCalculator& dimCalRes,
        const MatmulRunParas& params) const;
    void FillParam(MatmulRunParas& param);
    bool IsInvalidFactor(int32_t factor) const;
    void AddOptimalFactors(const std::string& opType, const MatmulRunParas& params, DimCalculator& dimCalRes) const;
    int32_t LoopNumFromSingleCoreToL0(const CoreStatusPack& coreStatus, const DimFactor& dimFactor) const;
    void GenDimsMapFactors(const std::string& opType, MatmulRunParas& params, DimCalculator& dimCalRes) const;
    void UpdateBufferSize(const TilingPolicy policy, const CoreStatusPack& coreStatus) const;
    bool UserPolicy(const TilingPolicy policy, const CoreStatusPack& coreStatus, const DimCalculator& dimCalRes) const;
    void PreprocessL0DB();
    void GetL0bAlign(std::vector<int32_t>& factors) const;
    int32_t GetBigPackageCondition(
        const CoreStatusPack& coreStatus, const DimCalculator& dimCalRes, const MatmulRunParas& params) const;
    int UpdateDepthB1(const SingleCoreStatus& singleCoreStatus) const;
    void GetSingleShape(
        const CoreStatusPack& coreStatus, const MatmulRunParas& param, int32_t& singleCoreM, int32_t& singleCoreN,
        int32_t& singleCoreK) const;
    bool CheckSingleShape(int32_t singleCoreM, int32_t singleCoreN, int32_t singleCoreK) const;
    MultiCoreScenario GetMultiCoreScenario(const MatmulRunParas& params) const;
    ComputeBaseBlock GetMultiCoreBasicBlock(const MatmulRunParas& params) const;
    int32_t GetSingleM() const;
    int32_t GetSingleN() const;
    int32_t GetSingleK() const;
    float CalcBaseBlockBandRatio(int32_t mDim, int32_t nDim, const ComputeBaseBlock& baseBlock) const;
    bool DoMultiCoreSplitMNTiling(const MatmulRunParas& params, CoreStatusPack& coreStatus, DimCalculator& dimCalRes);
    void CalcL1Tiling(
        const ComputeBaseBlock& baseBlock, int32_t& depthA1, int32_t& depthB1, int32_t& stepKa, int32_t& stepKb) const;
    void UpdateStepK(const ComputeBaseBlock& baseBlock, int32_t& stepK) const;
    bool NeedOutputAlign(int32_t m, int32_t n, int32_t k) const;
    void UpdateUsedSize() const;
    int64_t AdjustOuterProductL0Factor(SingleCoreStatus& singleCoreStatus) const;
    L0StatusPack GetL0CoreStatus(const ComputeBaseBlock& baseBlock) const;
    L1StatusPack GetL1CoreStatus(
        const ComputeBaseBlock& baseBlock, int32_t depthA1, int32_t depthB1, int32_t stepKa, int32_t stepKb) const;
    void UpdateShapeAndLayout() const;
    void AdjustFloatL1Factor(const SingleCoreStatus& singleCoreStatus) const;
    int64_t UpdateTiling(
        const MatmulRunParas& param, const CoreStatusPack& coreStatus, SingleCoreStatus& singleCoreStatus) const;

    // L1BankConflictOptimise
    int32_t GetC0Size(DataType dataType) const;
    int32_t GetABaseHeightAlign(int32_t baseHeight) const;
    int32_t GetABaseWidthAlign(int32_t baseWidth) const;
    int32_t GetBBaseHeightAlign(int32_t baseHeight) const;
    int32_t GetBBaseWidthAlign(int32_t baseWidth) const;
    int32_t GetScaleABaseHeightAlign() const;
    int32_t GetScaleABaseWidthAlign() const;
    int32_t GetScaleBBaseHeightAlign() const;
    int32_t GetScaleBBaseWidthAlign() const;
    int32_t GetMatrixAByteSize() const;
    int32_t GetMatrixBByteSize() const;
    int32_t GetMatrixScaleAByteSize() const;
    int32_t GetMatrixScaleBByteSize() const;
    void CalABAndScaleABL1Space(
        int32_t matrixByteSize, int32_t cacheNum, int32_t stepSize, uint32_t& curL1UpperHalfAddr,
        uint32_t& curL1LowerHalfAddr) const;
    void SetBaseMNK(const SingleCoreStatus& singleCoreStatus) const;

private:
    MatmulApiTilingBase* tilingIns_ = nullptr;
    bool enableSingleShape_ = false;
    bool splitCoreFlag_ = false;
    int32_t dbL0A_ = DB_ON;
    int32_t dbL0B_ = DB_ON;
    int32_t dbL0C_ = DB_OFF;
    int32_t numOfBlock_ = 24;
};
} // namespace matmul_tiling

#endif // _MATMUL_TILING_ALGORITHM_H
