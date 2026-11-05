/**
 * What does one cube-to-vector rendezvous cost?
 *
 * It is the last unexcluded candidate for the fixed cost a decode pays per
 * kernel. Excluded already: the rows carried (the verify costs the same at
 * six and eight), the global-memory round trip (1.364 us a tile, 0.87 ms a
 * round), and the launch itself, which is 19.30 us a replayed kernel and
 * accounts for the whole 36 ms. So this measurement is a check on that
 * account rather than a search: if the rendezvous is large, the launch figure
 * is double counting something.
 *
 * The flag id is not arbitrary. `kernel_operator_sync_impl.h` for this part
 * fixes the audience in the id -- 11 notifies AIC, 12 AIV, 13 both, 14 every
 * AIV -- and `GetffstMsg` keeps only four bits of it. An id outside that
 * convention produces a message nobody receives, so the wait never returns:
 * two earlier versions of this file hung on 4 and 5. One set of
 * `SYNC_AIV_FLAG` reaches both subblocks of the AIC's pair, so each waiting
 * once is balanced.
 *
 *   0  the rendezvous alone, no payload
 *   1  the rendezvous plus a 32 KiB tile through global memory
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t TILE_BYTES = 32 * 1024;
constexpr uint16_t NOTIFY_AIV = 12;

extern "C" [[bisheng::core_ratio(1, 2)]] __global__ __aicore__ void rendezvous_kernel(
    __gm__ float* sink, int mode, int iterations)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> staging;
    pipe.InitBuffer(staging, TILE_BYTES);
    LocalTensor<float> tile = staging.Get<float>();
    const int64_t count = TILE_BYTES / static_cast<int64_t>(sizeof(float));
    const int64_t core = GetBlockIdx();

    GlobalTensor<float> window;
    window.SetGlobalBuffer(sink + core * count, count);

    for (int step = 0; step < iterations; ++step) {
        if ASCEND_IS_AIC {
            if (mode == 1) {
                DataCopy(window, tile, count);
            }
            CrossCoreSetFlag<0, PIPE_MTE3>(NOTIFY_AIV);
        }
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(NOTIFY_AIV);
            if (mode == 1) {
                DataCopy(tile, window, count);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                Muls(tile, tile, (float)1.0009765625, count);
            }
        }
    }
}

void rendezvous_demo(uint32_t block_dim, void* stream, float* sink, int mode,
                     int iterations)
{
    rendezvous_kernel<<<block_dim, nullptr, stream>>>(sink, mode, iterations);
}
