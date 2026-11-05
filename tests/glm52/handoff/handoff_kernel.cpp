/**
 * What a dependent round trip through global memory costs on this part.
 *
 * This generation has no `L0C -> UB` path: the chip difference table gives
 * that row as absent for 1971 while David carries 128 to 256 bits, and
 * `L0C -> L1` exists while `L1 -> UB` does not. So a cube result reaches the
 * vector unit only through global memory, roughly 637 times a decode round,
 * and the layer-scale kernel AMD's design rests on assumes the on-chip path.
 *
 * The verify's fixed cost extrapolates to 36.02 ms, 450 us a layer over
 * eighty layers, which would be 57 us a handoff if eight handoffs a layer
 * carried it. This measures the round trip itself.
 *
 * Two modes, same traffic, differing only in whether the read depends on the
 * write:
 *
 *   0  write a tile to global memory, then read that tile back and use it.
 *   1  write a tile, then read a different tile nobody wrote this iteration.
 *
 * Mode 1 is mode 0's control: identical bytes in both directions, no
 * dependency, so the difference is what ordering costs rather than what the
 * transfer costs.
 *
 * Vector only. The cross-core flag protocol is deliberately not used here:
 * under `core_ratio(1, 2)` one AIC block's flag faces two AIV blocks' waits,
 * which deadlocks -- the first version of this file did, and
 * `shmem/fused_proj_reduce_kernel.cpp` carries the same never-run pattern.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t TILE_BYTES = 32 * 1024;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void handoff_kernel(
    __gm__ float* source, __gm__ float* sink, int mode, int iterations)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> staging;
    pipe.InitBuffer(staging, TILE_BYTES);
    LocalTensor<float> tile = staging.Get<float>();
    const int64_t count = TILE_BYTES / static_cast<int64_t>(sizeof(float));
    const int64_t core = GetBlockIdx();

    GlobalTensor<float> other;
    GlobalTensor<float> own;
    other.SetGlobalBuffer(source + core * count, count);
    own.SetGlobalBuffer(sink + core * count, count);

    for (int step = 0; step < iterations; ++step) {
        DataCopy(own, tile, count);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(tile, mode == 0 ? own : other, count);
        SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        Muls(tile, tile, (float)1.0009765625, count);
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }
}

void handoff_demo(uint32_t block_dim, void* stream, float* source, float* sink,
                  int mode, int iterations)
{
    handoff_kernel<<<block_dim, nullptr, stream>>>(source, sink, mode, iterations);
}
