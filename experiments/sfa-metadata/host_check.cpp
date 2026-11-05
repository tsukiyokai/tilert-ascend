#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#define __aicore__
#define unlikely(value) (value)
using bfloat16_t = uint16_t;
constexpr int V_TEMPLATE = 0;
enum class SFA_LAYOUT { TND, PA_BSND };
template <class A, class B> using IsSameType = std::is_same<A, B>;

void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

namespace AscendC {
enum class HardEvent { MTE3_MTE2, MTE3_V, V_MTE3, S_MTE3, MTE3_S, MTE2_S };
int flags[6][8]{};
bool pendingMetadata = false;
template <HardEvent event> void SetFlag(int id)
{
    Require(++flags[static_cast<int>(event)][id] == 1, "event double set");
}
template <HardEvent event> void WaitFlag(int id)
{
    Require(flags[static_cast<int>(event)][id]-- == 1, "unpaired event wait");
    if constexpr (event == HardEvent::MTE2_S) pendingMetadata = false;
}
}
using AscendC::SetFlag;
using AscendC::WaitFlag;
using AscendC::HardEvent;

int subBlock = 0;
int GetSubBlockIdx() { return subBlock; }
int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

template <class T> struct LocalTensor {
    std::vector<int32_t>* storage = nullptr;
    int64_t offset = 0;
    LocalTensor operator[](int64_t index) const
    {
        return {storage, offset + index};
    }
    int32_t GetValue(int64_t index) const
    {
        Require(!AscendC::pendingMetadata, "UB read before MTE2_S wait");
        Require(offset + index >= 0 && offset + index < storage->size(),
                "UB read out of bounds");
        return storage->at(offset + index);
    }
    void SetValue(int64_t index, int32_t value) const
    {
        Require(offset + index >= 0 && offset + index < storage->size(),
                "UB write out of bounds");
        storage->at(offset + index) = value;
    }
};

struct TBuf {
    std::vector<int32_t> data = std::vector<int32_t>(2048, 0x5a5a5a5a);
    template <class T> LocalTensor<T> Get() { return {&data, 0}; }
};

template <class T> struct GlobalTensor {
    std::vector<int32_t>* storage = nullptr;
    int64_t offset = 0;
    bool sink = false;
    int* reads = nullptr;
    GlobalTensor operator[](int64_t index) const
    {
        return {storage, offset + index, sink, reads};
    }
    int32_t Raw(int64_t index) const
    {
        Require(offset + index >= 0 && offset + index < storage->size(),
                "GM read out of bounds");
        return storage->at(offset + index);
    }
    int32_t GetValue(int64_t index) const
    {
        if (reads) ++*reads;
        return Raw(index);
    }
};

struct DataCopyExtParams {
    uint32_t blockCount = 0, blockLen = 0, srcStride = 0, dstStride = 0;
};
template <class T> struct DataCopyPadExtParams {};

template <class T> void DataCopyPad(
    LocalTensor<T> destination, GlobalTensor<T> source,
    DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    Require(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0,
            "unexpected metadata DMA geometry");
    Require(copy.blockLen % sizeof(int32_t) == 0, "unaligned INT32 payload");
    for (int64_t i = 0; i < copy.blockLen / sizeof(int32_t); ++i)
        destination.SetValue(i, source.Raw(i));
    // DataCopyPad may round its UB destination up to a 32-byte block;
    // padded bytes must fit without reading beyond the GM payload.
    const int64_t count = copy.blockLen / sizeof(int32_t);
    for (int64_t i = count; i < CeilDiv(count, 8) * 8; ++i)
        destination.SetValue(i, 0x3c3c3c3c);
    AscendC::pendingMetadata = true;
}

template <class T> void DataCopyPad(
    GlobalTensor<T> destination, LocalTensor<T> source, DataCopyExtParams copy)
{
    if (destination.sink) return;
    for (int64_t i = 0; i < copy.blockLen / sizeof(int32_t); ++i)
        destination.storage->at(destination.offset + i) = source.GetValue(i);
}
template <class T, class U> void Duplicate(LocalTensor<T>, U, uint32_t) {}

struct ConstInfo {
    int64_t batchSize = 1, headDim = 512, headDimRope = 64, kvHeadNum = 1;
    int64_t kvCacheBlockSize = 128, sparseBlockSize = 1, preLoadNum = 2;
    int64_t s2BaseSize = 512, maxBlockNumPerBatch = 72;
    int64_t sparseBlockCount = 2048, gSize = 4, sparseMode = 3, qSeqSize = 32;
};
struct RunInfo {
    uint64_t curActualSeqLenOri = 8192, actS1Size = 32;
    int64_t bIdx = 0, gS1Idx = 0, n2Idx = 0, s2Idx = 0, loop = 0;
    int64_t tensorBOffset = 0, actualSingleProcessSInnerSize = 512;
};
struct BF16 { using kvType = bfloat16_t; using outputType = bfloat16_t;
              static constexpr bool hasRope = true; };
struct FP16 { using kvType = int16_t; using outputType = int16_t;
              static constexpr bool hasRope = true; };

using Pair = std::tuple<int64_t, int64_t, int64_t, int64_t>;

// These declarations host the actual extracted upstream/candidate methods.
// CopyInKv is observed at its unchanged pair and address boundary; no payload
// arithmetic, device concurrency or latency is simulated by this program.
#define DEFINE_SERVICE(Name) \
template <class SFAT> class Name { \
public: \
    using KV_T = typename SFAT::kvType; \
    using OUT_T = typename SFAT::outputType; \
    static constexpr bool PAGE_ATTENTION = true, FLASH_DECODE = false; \
    static constexpr int TEMPLATE_MODE = V_TEMPLATE; \
    static constexpr SFA_LAYOUT LAYOUT_T = SFA_LAYOUT::TND; \
    static constexpr SFA_LAYOUT KV_LAYOUT_T = SFA_LAYOUT::PA_BSND; \
    static constexpr int MERGE_CACHE_GM_BUF_NUM = 4, SYNC_INPUT_V0BUF_FLAG = 6; \
    ConstInfo sfaVecConstInfo; \
    TBuf v0ValidSizeBuff, kvScratch, ropeScratch; \
    LocalTensor<int32_t> v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>(); \
    LocalTensor<KV_T> kvMergUb_ = kvScratch.Get<KV_T>(); \
    LocalTensor<KV_T> ropeMergUb_ = ropeScratch.Get<KV_T>(); \
    GlobalTensor<int32_t> topkGm_, blkTableGm_, actualSeqLengthsQGm; \
    GlobalTensor<int32_t> kvValidSizeGm_; \
    GlobalTensor<KV_T> kvMergeGm_{nullptr, 0, true}; \
    bool metadataReady_ = false; \
    int64_t metadataBegin_ = 0, metadataEnd_ = 0; \
    LocalTensor<int32_t> metadataTopk_, metadataPages_; \
    std::vector<Pair> pairs; \
    void StageMetadata(const RunInfo&, int64_t, int64_t, int64_t); \
    void GetRealS2Idx(int64_t, int64_t&, int64_t, const RunInfo&); \
    int64_t GetKeyGmOffset(int64_t, const RunInfo&, int64_t); \
    void MergeKv(const RunInfo&); \
    void CopyOutMrgeResult(int64_t, int64_t, int64_t, int64_t, const RunInfo&) {} \
    void CopyInKv(int64_t& count, int64_t, int64_t, int64_t first, \
                  int64_t second, const RunInfo& info) { \
        int64_t limit = info.curActualSeqLenOri; \
        if (sfaVecConstInfo.sparseMode == 3) \
            limit = limit - info.actS1Size + info.gS1Idx / \
                        sfaVecConstInfo.gSize + 1; \
        const int64_t key1 = GetKeyGmOffset(first, info, limit); \
        const int64_t key2 = GetKeyGmOffset(second, info, limit); \
        pairs.emplace_back(first, second, key1, key2); \
        count += ((key1 >= 0) + (key2 >= 0)) * sfaVecConstInfo.sparseBlockSize; \
    } \
};

DEFINE_SERVICE(OriginalService)
DEFINE_SERVICE(StagedService)
#include "extracted.h"

template <class Service> void Bind(
    Service& service, std::vector<int32_t>& topk, std::vector<int32_t>& pages,
    std::vector<int32_t>& valid, int& reads)
{
    service.topkGm_ = {&topk, 0, false, &reads};
    service.blkTableGm_ = {&pages, 0, false, &reads};
    service.kvValidSizeGm_ = {&valid};
}

template <class Traits> int CheckCases()
{
    std::mt19937 random(53941);
    OriginalService<Traits> original;
    StagedService<Traits> candidate;
    std::vector<int32_t> topk(32 * 2048), pages(1025), validA(1024), validB(1024);
    int originalReads = 0, candidateReads = 0, cases = 0;
    Bind(original, topk, pages, validA, originalReads);
    Bind(candidate, topk, pages, validB, candidateReads);
    const int sizes[] = {0, 1, 2, 3, 31, 32, 33, 127, 128, 255, 256, 511, 512};
    for (int iteration = 0; iteration < 60; ++iteration) {
        for (int i = 0; i < topk.size(); ++i) topk[i] = random() % 9216;
        for (int i = 0; i < pages.size(); ++i)
            pages[i] = (pages.size() - i - 1 + iteration * 17) % pages.size();
        // Include page crossings, sentinels and causal/future boundaries.
        topk[iteration % 2048] = -1;
        topk[2047] = iteration % 2 ? 9215 : -1;
        topk[128] = 127; topk[129] = 128;
        ConstInfo config;
        if (iteration % 3 == 1) config.headDimRope = 32;  // Fallback after a hit.
        if (iteration % 5 == 2) config.maxBlockNumPerBatch = 1024;
        if (iteration % 7 == 3) config.maxBlockNumPerBatch = 1025;
        if (iteration % 11 == 4) config.maxBlockNumPerBatch = 1023;
        original.sfaVecConstInfo = candidate.sfaVecConstInfo = config;
        for (int width : {1, 8, 32}) for (int row : {0, width - 1}) {
            for (int split = 0; split < 4; ++split) for (int size : sizes) {
                for (subBlock = 0; subBlock < 2; ++subBlock) {
                    RunInfo info;
                    info.actS1Size = width;
                    info.gS1Idx = row * 4;
                    info.s2Idx = split;
                    info.actualSingleProcessSInnerSize = size;
                    info.loop = cases % 4;
                    original.pairs.clear(); candidate.pairs.clear();
                    original.MergeKv(info); candidate.MergeKv(info);
                    Require(original.pairs == candidate.pairs,
                            "upstream/candidate pair or address mismatch");
                    Require(std::equal(original.v0ValidSizeBuff.data.begin(),
                                       original.v0ValidSizeBuff.data.begin() + 512,
                                       candidate.v0ValidSizeBuff.data.begin()),
                            "metadata overwrote an original UB live region");
                    // Actual Vec1's receive slice and Vec2's broadcast scratch.
                    for (int index = 128; index < 512; ++index) {
                        int32_t value = random();
                        original.v0ValidSizeBuff.data[index] = value;
                        candidate.v0ValidSizeBuff.data[index] = value;
                    }
                    ++cases;
                }
            }
        }
    }
    if constexpr (std::is_same<Traits, BF16>::value)
        Require(candidateReads < originalReads, "metadata path never eliminated GM reads");
    else
        Require(candidateReads == originalReads, "unsupported dtype used metadata path");
    Require(!AscendC::pendingMetadata, "metadata DMA left pending");
    for (auto& direction : AscendC::flags) for (int flag : direction)
        Require(flag == 0, "event leaked between calls");
    std::cout << cases << " cases, scalar GM reads " << originalReads
              << " -> " << candidateReads << '\n';
    return cases;
}

int main()
{
    try {
        CheckCases<BF16>();
        CheckCases<FP16>();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
