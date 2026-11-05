/**
 * Prototype INT8 projection followed by peer push and FP32 reduction.
 * IterateAll completes the projection before publishing PROJECT_READY;
 * this implementation does not overlap projection with communication.
 */
#include "fused_proj_reduce_kernel.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "acl/acl.h"
#include "shmem.h"

using namespace AscendC;

constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t PLANES = 2;
constexpr int64_t SIGNAL_SLOT_INT32 = 16;
constexpr int64_t MAX_SLICES = 32;
constexpr int64_t STAGE_UB_BYTES = 32 * 1024;
constexpr int64_t REDUCE_UB_BYTES = 32 * 1024;
constexpr uint16_t QUANT_READY = 4;
constexpr uint16_t PROJECT_READY = 5;

using AQuant = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using BQuantNz = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t, true>;
using CPartial = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasNone = MatmulType<TPosition::GM, CubeFormat::ND, float>;

ACLSHMEM_DEVICE int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct FusedLayout {
    __gm__ float* data;
    int64_t stride;
    int64_t elements;
};

ACLSHMEM_DEVICE FusedLayout FusedPlane(__gm__ uint8_t* window, int rows, int cols, int64_t pe_size, int64_t plane)
{
    FusedLayout layout;
    layout.elements = static_cast<int64_t>(rows) * cols;
    layout.stride = AlignUp(layout.elements * sizeof(float), ALIGN_BYTES) / sizeof(float);
    const int64_t head_int32 = PLANES * MAX_SLICES * SIGNAL_SLOT_INT32;
    layout.data = (__gm__ float*)((__gm__ int32_t*)window + head_int32) + plane * pe_size * layout.stride;
    return layout;
}

/** Push a chunk of the completed projection to each peer. */
ACLSHMEM_DEVICE void PushTile(const FusedLayout& layout, __gm__ float* partial, int64_t begin, int64_t count,
                             int64_t core, int64_t my_pe, int64_t pe_size, __ubuf__ float* stage)
{
    if (core >= pe_size) {
        return;
    }
    const int64_t peer = (my_pe + core) % pe_size;
    aclshmemx_mte_put_nbi(layout.data + my_pe * layout.stride + begin, partial + begin, stage, STAGE_UB_BYTES, count,
                          peer, EVENT_ID0);
}

/** Sum every rank's slot over this core's stripe, in rank order. */
ACLSHMEM_DEVICE void ReduceStripe(const FusedLayout& layout, __gm__ float* out, int64_t core, int64_t cores,
                                 int64_t pe_size, const LocalTensor<float>& acc, const LocalTensor<float>& term)
{
    const int64_t stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(float));
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));

    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        GlobalTensor<float> source;
        source.SetGlobalBuffer(layout.data + begin + offset, count);
        DataCopy(acc, source, count);
        PipeBarrier<PIPE_ALL>();
        for (int64_t pe = 1; pe < pe_size; ++pe) {
            source.SetGlobalBuffer(layout.data + pe * layout.stride + begin + offset, count);
            DataCopy(term, source, count);
            PipeBarrier<PIPE_ALL>();
            Add(acc, acc, term, count);
            PipeBarrier<PIPE_ALL>();
        }
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, acc, count);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void fused_proj_reduce_kernel(
    uint64_t fftsAddr, __gm__ int8_t* activation, __gm__ int8_t* weight_nz, __gm__ float* partial, __gm__ float* out,
    __gm__ uint8_t* window, __gm__ uint8_t* tiling, int rows, int cols, int rounds, int round_base)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    util_set_ffts_config(fftsAddr);

    TPipe pipe;
    TBuf<TPosition::VECCALC> staging;
    TBuf<TPosition::VECCALC> accumulator;
    TBuf<TPosition::VECCALC> addend;
    pipe.InitBuffer(staging, STAGE_UB_BYTES);
    pipe.InitBuffer(accumulator, REDUCE_UB_BYTES);
    pipe.InitBuffer(addend, REDUCE_UB_BYTES);
    __ubuf__ float* stage = (__ubuf__ float*)staging.Get<float>().GetPhyAddr();
    LocalTensor<float> acc = accumulator.Get<float>();
    LocalTensor<float> term = addend.Get<float>();

    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();

    for (int round = 0; round < rounds; ++round) {
        const int64_t index = static_cast<int64_t>(round_base) + round;
        const FusedLayout layout = FusedPlane(window, rows, cols, pe_size, index % PLANES);

        if ASCEND_IS_AIC {
            TCubeTiling cube;
            auto* destination = (uint32_t*)&cube;
            auto* source = (__gm__ uint32_t*)tiling;
            for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t); ++word) {
                destination[word] = source[word];
            }
            Matmul<AQuant, BQuantNz, CPartial, BiasNone> mm;
            mm.Init(&cube, &pipe);
            GlobalTensor<int8_t> a;
            GlobalTensor<int8_t> b;
            GlobalTensor<float> c;
            a.SetGlobalBuffer((__gm__ int8_t*)activation);
            b.SetGlobalBuffer((__gm__ int8_t*)weight_nz);
            c.SetGlobalBuffer((__gm__ float*)partial);
            mm.SetTensorA(a);
            mm.SetTensorB(b, true);
            mm.IterateAll(c);
            mm.End();
            // The partial is in global memory; hand it to the vector side,
            // which owns the one-sided operations.
            CrossCoreSetFlag<0, PIPE_FIX>(PROJECT_READY);
        }

        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(PROJECT_READY);
            // The full projection is ready before this chunk loop.
            const int64_t chunk = STAGE_UB_BYTES / static_cast<int64_t>(sizeof(float));
            for (int64_t begin = 0; begin < layout.elements; begin += chunk) {
                const int64_t count =
                    (layout.elements - begin < chunk) ? layout.elements - begin : chunk;
                PushTile(layout, partial, begin, count, core, my_pe, pe_size, stage);
            }
            aclshmem_quiet();
            aclshmemx_sync_vec_all();
            ReduceStripe(layout, out, core, cores, pe_size, acc, term);
        }
    }
}

void fused_proj_reduce_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, int8_t* activation, int8_t* weight_nz,
                            float* partial, float* out, uint8_t* window, uint8_t* tiling, int rows, int cols,
                            int rounds, int round_base)
{
    fused_proj_reduce_kernel<<<block_dim, nullptr, stream>>>(fftsAddr, activation, weight_nz, partial, out, window,
                                                             tiling, rows, cols, rounds, round_base);
}
