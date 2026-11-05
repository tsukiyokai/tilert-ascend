/**
 * What does a kernel cost before it does anything?
 *
 * Every measurement today says a decode's cost barely depends on what it
 * carries: the verify is the same at six rows and eight, sparse attention the
 * same at one row and eight, and `TransposeBatchMatMul` computes 4.2 MFLOP in
 * 12.08 us on 24 cube blocks with no vector blocks at all. The remaining
 * candidate for that fixed cost is the cross-core rendezvous, but that last
 * operator has none -- so a floor measured without one bounds how much of the
 * fixed cost any fusion could remove.
 *
 * Four shapes of the same loop, chosen so each isolates one term:
 *
 *   0  an empty body. The launch and the block's own loop.
 *   1  one 32 KiB global read per iteration, nothing else.
 *   2  the read plus one vector pass over it.
 *   3  the read, the pass, and a 32 KiB write back.
 *
 * Vector only, because a cross-core flag under `core_ratio(1, 2)` faces two
 * waits per set and deadlocks -- which the first version of this benchmark
 * did.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t TILE_BYTES = 32 * 1024;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void floor_kernel(
    __gm__ float* source, __gm__ float* sink, int mode, int iterations)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> staging;
    pipe.InitBuffer(staging, TILE_BYTES);
    LocalTensor<float> tile = staging.Get<float>();
    const int64_t count = TILE_BYTES / static_cast<int64_t>(sizeof(float));
    const int64_t core = GetBlockIdx();

    GlobalTensor<float> from;
    GlobalTensor<float> to;
    from.SetGlobalBuffer(source + core * count, count);
    to.SetGlobalBuffer(sink + core * count, count);

    for (int step = 0; step < iterations; ++step) {
        if (mode == 0) {
            continue;
        }
        DataCopy(tile, from, count);
        if (mode == 1) {
            continue;
        }
        SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        Muls(tile, tile, (float)1.0009765625, count);
        if (mode == 2) {
            continue;
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(to, tile, count);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void floor_demo(uint32_t block_dim, void* stream, float* source, float* sink,
                int mode, int iterations)
{
    floor_kernel<<<block_dim, nullptr, stream>>>(source, sink, mode, iterations);
}
