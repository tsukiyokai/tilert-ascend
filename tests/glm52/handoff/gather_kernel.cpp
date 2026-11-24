/**
 * What bandwidth does a scattered 1152-byte gather reach?
 *
 * Sparse attention spends 73.43 us a call above its structural floor while
 * the bytes it needs are worth 12.26, a ratio of 6.07 -- the largest
 * unexplained deviation in the capture, and the only operator for which rows,
 * bytes and structure are all excluded. It costs the same at one row as at
 * eight.
 *
 * What it does is gather 2048 selected slots out of a paged cache of 73 blocks
 * of 128, each slot 512 latent plus 64 rope values in bf16: 1152 bytes. A
 * gather of 2048 scattered 1152-byte rows is not one read of 2.25 MiB, and if
 * the achieved rate at that granularity is a fraction of the contiguous rate
 * then the 6.07 is the access pattern rather than the kernel.
 *
 * Three patterns over the same total bytes:
 *
 *   0  contiguous: read the rows in order.
 *   1  strided: every row, but one page apart, so no two fall in a page.
 *   2  scattered: a fixed permutation, which is what a top-k selection gives.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t SLOT_BYTES = 1152;
constexpr int64_t SLOT_FLOATS = SLOT_BYTES / sizeof(float);
constexpr int64_t SLOTS = 2048;
constexpr int64_t PAGE_SLOTS = 128;
constexpr int64_t STAGE_SLOTS = 8;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void gather_kernel(
    __gm__ float* cache, __gm__ float* out, int mode, int iterations)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> staging;
    pipe.InitBuffer(staging, STAGE_SLOTS * SLOT_BYTES);
    LocalTensor<float> tile = staging.Get<float>();

    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    const int64_t share = SLOTS / cores;

    for (int step = 0; step < iterations; ++step) {
        for (int64_t index = 0; index < share; ++index) {
            const int64_t logical = core * share + index;
            int64_t slot = logical;
            if (mode == 1) {
                slot = (logical * PAGE_SLOTS) % SLOTS;
            } else if (mode == 2) {
                // A fixed permutation with no locality, standing in for a
                // top-k selection: an odd stride over a power-of-two range
                // visits every slot exactly once.
                slot = (logical * 1021 + 37) % SLOTS;
            }
            GlobalTensor<float> from;
            from.SetGlobalBuffer(cache + slot * SLOT_FLOATS, SLOT_FLOATS);
            DataCopy(tile[(index % STAGE_SLOTS) * SLOT_FLOATS], from, SLOT_FLOATS);
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            Muls(tile[(index % STAGE_SLOTS) * SLOT_FLOATS],
                 tile[(index % STAGE_SLOTS) * SLOT_FLOATS],
                 (float)1.0009765625, SLOT_FLOATS);
        }
    }
}

void gather_demo(uint32_t block_dim, void* stream, float* cache, float* out,
                 int mode, int iterations)
{
    gather_kernel<<<block_dim, nullptr, stream>>>(cache, out, mode, iterations);
}
