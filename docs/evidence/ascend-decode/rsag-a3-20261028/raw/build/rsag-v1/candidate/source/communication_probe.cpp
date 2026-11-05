#define ASCENDC_CUBE_ONLY

#include "communication_probe.h"
#include "communication_device.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace trt_rsag_candidate;

namespace {

using AInput = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BInput = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using COutput = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using NoBias = MatmulType<TPosition::GM, CubeFormat::ND, float>;

__aicore__ inline void ProjectTiles(
    TPipe& pipe, __gm__ bfloat16_t* input, __gm__ bfloat16_t* weight,
    __gm__ float* product, __gm__ uint8_t* tiling, int64_t elements,
    int64_t blocks)
{
    TCubeTiling cube;
    auto* destination = reinterpret_cast<uint32_t*>(&cube);
    auto* source = reinterpret_cast<__gm__ uint32_t*>(tiling);
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
    Matmul<AInput, BInput, COutput, NoBias> projection;
    projection.Init(&cube, &pipe);
    GlobalTensor<bfloat16_t> a, b;
    GlobalTensor<float> c;
    b.SetGlobalBuffer(weight, kCommunicationProbeTile);
    for (int64_t tile = GetBlockIdx();
         tile < elements / kCommunicationProbeTile; tile += blocks) {
        a.SetGlobalBuffer(input + tile * kCommunicationProbeTile,
                          kCommunicationProbeTile);
        c.SetGlobalBuffer(product + tile * kCommunicationProbeTile,
                          kCommunicationProbeTile);
        projection.SetTensorA(a);
        projection.SetTensorB(b, false);
        projection.IterateAll(c);
        projection.End();
    }
}

__aicore__ inline void NarrowProjection(
    CommunicationBuffers& held, __gm__ float* product,
    __gm__ bfloat16_t* partial, int64_t elements, int64_t vectors)
{
    for (int64_t tile = GetBlockIdx();
         tile < elements / kCommunicationChunk; tile += vectors) {
        GlobalTensor<float> source;
        GlobalTensor<bfloat16_t> destination;
        source.SetGlobalBuffer(product + tile * kCommunicationChunk,
                               kCommunicationChunk);
        destination.SetGlobalBuffer(partial + tile * kCommunicationChunk,
                                    kCommunicationChunk);
        DataCopy(held.accumulator, source, kCommunicationChunk);
        CommunicationFence<HardEvent::MTE2_V>();
        Cast(held.narrow, held.accumulator, RoundMode::CAST_RINT,
             kCommunicationChunk);
        CommunicationFence<HardEvent::V_MTE3>();
        DataCopy(destination, held.narrow, kCommunicationChunk);
        CommunicationFence<HardEvent::MTE3_MTE2>();
    }
}

}  // namespace

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void trt_rsag_candidate_probe(
    uint64_t sync_base, GM_ADDR workspace, __gm__ uint8_t* window,
    __gm__ bfloat16_t* partial, __gm__ bfloat16_t* weight,
    __gm__ bfloat16_t* first_sum, __gm__ float* product,
    __gm__ bfloat16_t* next_partial, __gm__ bfloat16_t* output,
    __gm__ uint8_t* tiling, int64_t rows, int64_t capacity,
    int32_t rank, int32_t ranks, int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    SetSysWorkspace(workspace);
    TPipe pipe;
    CommunicationBuffers held;
    if ASCEND_IS_AIV {
        held.Init(pipe);
    }
    AllReduceBf16(held, window, partial, first_sum, rows, capacity,
                  rank, ranks, GetBlockIdx(), 2 * blocks);
    if ASCEND_IS_AIC {
        ProjectTiles(pipe, first_sum, weight, product, tiling,
                      rows * kCommunicationHidden, blocks);
    }
    CommunicationPublish();
    SyncAll<false>();
    if ASCEND_IS_AIV {
        NarrowProjection(held, product, next_partial,
                          rows * kCommunicationHidden, 2 * blocks);
    }
    AllReduceBf16(held, window, next_partial, output, rows, capacity,
                  rank, ranks, GetBlockIdx(), 2 * blocks);
}

namespace trt_rsag_candidate {

void LaunchCommunicationProbe(
    uint32_t blocks, void* stream, uint64_t sync_base, void* workspace,
    void* window, void* partial, void* weight, void* first_sum,
    void* product, void* next_partial, void* output, void* tiling,
    int64_t rows, int64_t capacity, int32_t rank, int32_t ranks)
{
    trt_rsag_candidate_probe<<<blocks, nullptr, stream>>>(
        sync_base, static_cast<uint8_t*>(workspace),
        static_cast<uint8_t*>(window), static_cast<bfloat16_t*>(partial),
        static_cast<bfloat16_t*>(weight), static_cast<bfloat16_t*>(first_sum),
        static_cast<float*>(product), static_cast<bfloat16_t*>(next_partial),
        static_cast<bfloat16_t*>(output), static_cast<uint8_t*>(tiling),
        rows, capacity, rank, ranks, blocks);
}

}  // namespace trt_rsag_candidate
