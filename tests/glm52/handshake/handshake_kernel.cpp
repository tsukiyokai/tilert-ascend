/**
 * Which cross-core handshake survives being load-bearing in both directions?
 *
 * The cube-to-vector direction is settled: mode 2, and one set raises the
 * counter of both of a cube's vector subblocks. The other direction has never
 * carried a dependency here. The shared expert's chain set a flag that way,
 * but the ordering it relied on came from the `SyncAll` after it, so a wrong
 * mode there would not have shown. Making it load-bearing hung the routed
 * experts' first launch with no output at all.
 *
 * The modes are template arguments, so a sweep needs a switch over constants
 * rather than a variable. Each variant runs as its own process under a
 * timeout: finishing means live, the timeout means deadlocked. Running the
 * handshake several times a launch separates a deadlock from a leftover
 * count, which passes once and then races.
 */
#include "handshake_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint16_t FORWARD = 4;
constexpr uint16_t REVERSE = 5;
// Progress is one int32 a core, on its own 32-byte block so a write cannot
// stamp over a neighbour's slot.
constexpr int64_t STRIDE = 8;

/** Mode 2 reaches the vector side; this direction is already proven. */
__aicore__ inline void SetForward()
{
    CrossCoreSetFlag<0x02, PIPE_FIX>(FORWARD);
}

__aicore__ inline void WaitForward()
{
    CrossCoreWaitFlag<0x02, PIPE_MTE2>(FORWARD);
}

template <uint8_t MODE>
__aicore__ inline void SetReverse()
{
    CrossCoreSetFlag<MODE, PIPE_MTE3>(REVERSE);
}

template <uint8_t MODE>
__aicore__ inline void WaitReverse()
{
    CrossCoreWaitFlag<MODE, PIPE_MTE2>(REVERSE);
}

__aicore__ inline void Mark(GlobalTensor<int32_t>& progress, int64_t slot,
                            int32_t reached)
{
    progress.SetValue(slot * STRIDE, reached);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE,
                             DcciDst::CACHELINE_OUT>(progress);
}

/** The cube's side of one round. */
template <uint8_t MODE>
__aicore__ inline void CubeRound(GlobalTensor<int32_t>& progress,
                                 int64_t waits, int32_t round)
{
    SetForward();
    Mark(progress, 0, 100 * round + 1);
    for (int64_t once = 0; once < waits; ++once) {
        WaitReverse<MODE>();
    }
    Mark(progress, 0, 100 * round + 2);
}

/** A vector subblock's side of one round. */
template <uint8_t MODE>
__aicore__ inline void VectorRound(GlobalTensor<int32_t>& progress,
                                   int64_t slot, bool sets, int32_t round)
{
    WaitForward();
    Mark(progress, slot, 100 * round + 1);
    if (sets) {
        SetReverse<MODE>();
    }
    Mark(progress, slot, 100 * round + 2);
}

}  // namespace

extern "C" __global__ __mix__(1, 2) __aicore__ void handshake_probe(
    uint64_t sync_base, __gm__ int32_t* progress_memory, int64_t reverse_mode,
    int64_t pattern, int64_t rounds)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);

    GlobalTensor<int32_t> progress;
    progress.SetGlobalBuffer(progress_memory, 4 * STRIDE);

    // Pattern 0: subblock zero sets, the cube waits once -- what the routed
    // experts did, and what hung. Pattern 1: both set, the cube waits once --
    // the rule this project wrote for itself, split the work never the
    // handshake, which leaves a surplus set. Pattern 2: both set, the cube
    // waits twice -- symmetric, nothing left over.
    const bool both_set = (pattern != 0);
    const int64_t waits = (pattern == 2) ? 2 : 1;

    for (int32_t round = 1; round <= rounds; ++round) {
        if ASCEND_IS_AIC {
            switch (reverse_mode) {
            case 0:
                CubeRound<0x00>(progress, waits, round);
                break;
            case 1:
                CubeRound<0x01>(progress, waits, round);
                break;
            case 2:
                CubeRound<0x02>(progress, waits, round);
                break;
            default:
                CubeRound<0x03>(progress, waits, round);
                break;
            }
        }
        if ASCEND_IS_AIV {
            const int64_t sub = GetSubBlockIdx();
            const int64_t slot = 1 + sub;
            const bool sets = both_set || (sub == 0);
            switch (reverse_mode) {
            case 0:
                VectorRound<0x00>(progress, slot, sets, round);
                break;
            case 1:
                VectorRound<0x01>(progress, slot, sets, round);
                break;
            case 2:
                VectorRound<0x02>(progress, slot, sets, round);
                break;
            default:
                VectorRound<0x03>(progress, slot, sets, round);
                break;
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

void handshake_launch(void* stream, uint64_t sync_base, void* progress,
                      int64_t reverse_mode, int64_t pattern, int64_t rounds)
{
    handshake_probe<<<1, nullptr, stream>>>(sync_base,
                                            (__gm__ int32_t*)progress,
                                            reverse_mode, pattern, rounds);
}
