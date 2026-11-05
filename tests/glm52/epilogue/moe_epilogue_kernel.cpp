/**
 * The MoE epilogue in one kernel instead of three, and both arms to compare.
 *
 * After the routed combine the layer computes, in three separate launches:
 * `Muls` scales the routed output, `ForeachAddList` adds the shared expert's,
 * and `Cast` widens the sum to FP32 for the TP reduction. One capture round
 * puts them at 8.641, 21.403 and 6.860 us a layer -- 36.9 us to move 98 KiB,
 * whose own bandwidth cost is 0.25. Over 75 MoE layers that is 2.77 ms a
 * round, and it is structure: `ForeachAddList` costs the same at one row as
 * at eight.
 *
 * The rounding is preserved rather than improved. `Muls` emits BF16, so the
 * product is rounded before the add; the add emits BF16, so the sum is
 * rounded before the widening; BF16 to FP32 is exact. A fused kernel holding
 * the product in FP32 would be more accurate and not comparable, so this
 * rounds at both of the same two places.
 *
 * Vector only. The three kernels it replaces are all AI_VECTOR_CORE, which
 * reaches 0.72 us on this die where a MIX kernel cannot go below 13.56.
 */
#include "moe_epilogue_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

// dav-c220's `Muls` rejects BF16 -- it takes half, float, int16 or int32
// only -- so every stage computes in FP32 and uses `Cast` to put the two BF16
// roundings back exactly where the three kernels have them. That is also why
// the vendor's `Muls` costs 8.461 us to move 98 KiB: it widens too.
constexpr int64_t TILE_ELEMENTS = 4 * 1024;

namespace {

struct Stripe {
    int64_t begin;
    int64_t span;
};

__aicore__ inline Stripe OwnStripe(int64_t elements)
{
    const int64_t cores = GetBlockNum();
    const int64_t width = (elements + cores - 1) / cores;
    const int64_t begin = GetBlockIdx() * width;
    if (begin >= elements) {
        return Stripe{0, 0};
    }
    return Stripe{begin, (begin + width <= elements) ? width
                                                     : elements - begin};
}

/** One tile's working set: two BF16 inputs and two FP32 widenings. */
struct Buffers {
    TPipe pipe;
    TBuf<TPosition::VECCALC> narrow_buffer;
    TBuf<TPosition::VECCALC> other_buffer;
    TBuf<TPosition::VECCALC> wide_buffer;
    TBuf<TPosition::VECCALC> addend_buffer;
    LocalTensor<bfloat16_t> narrow;
    LocalTensor<bfloat16_t> other;
    LocalTensor<float> wide;
    LocalTensor<float> addend;

    __aicore__ inline explicit Buffers(int64_t elements)
    {
        pipe.InitBuffer(narrow_buffer, elements * sizeof(bfloat16_t));
        pipe.InitBuffer(other_buffer, elements * sizeof(bfloat16_t));
        pipe.InitBuffer(wide_buffer, elements * sizeof(float));
        pipe.InitBuffer(addend_buffer, elements * sizeof(float));
        narrow = narrow_buffer.Get<bfloat16_t>();
        other = other_buffer.Get<bfloat16_t>();
        wide = wide_buffer.Get<float>();
        addend = addend_buffer.Get<float>();
    }
};

/** `narrow *= scale`, rounded to BF16 as the vendor's `Muls` leaves it. */
__aicore__ inline void Scale(Buffers& held, float scale, int64_t count)
{
    Cast(held.wide, held.narrow, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Muls(held.wide, held.wide, scale, count);
    PipeBarrier<PIPE_V>();
    Cast(held.narrow, held.wide, RoundMode::CAST_RINT, count);
}

/** `narrow += other` in BF16, leaving the FP32 widening in `wide`. */
__aicore__ inline void Combine(Buffers& held, int64_t count)
{
    PipeBarrier<PIPE_V>();
    Cast(held.wide, held.narrow, RoundMode::CAST_NONE, count);
    Cast(held.addend, held.other, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Add(held.wide, held.wide, held.addend, count);
    PipeBarrier<PIPE_V>();
    Cast(held.narrow, held.wide, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(held.wide, held.narrow, RoundMode::CAST_NONE, count);
}

}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void moe_epilogue_fused(__gm__ bfloat16_t* routed, __gm__ bfloat16_t* shared,
                        __gm__ float* out, float routed_scale,
                        int64_t elements, int iterations)
{
    const Stripe mine = OwnStripe(elements);
    if (mine.span == 0) {
        return;
    }
    Buffers held(TILE_ELEMENTS);
    GlobalTensor<bfloat16_t> routed_global;
    GlobalTensor<bfloat16_t> shared_global;
    GlobalTensor<float> out_global;
    routed_global.SetGlobalBuffer(routed + mine.begin, mine.span);
    shared_global.SetGlobalBuffer(shared + mine.begin, mine.span);
    out_global.SetGlobalBuffer(out + mine.begin, mine.span);

    for (int step = 0; step < iterations; ++step)
    for (int64_t done = 0; done < mine.span; done += TILE_ELEMENTS) {
        const int64_t count = (mine.span - done < TILE_ELEMENTS)
                                  ? mine.span - done : TILE_ELEMENTS;
        DataCopy(held.narrow, routed_global[done], count);
        DataCopy(held.other, shared_global[done], count);
        PipeBarrier<PIPE_ALL>();
        Scale(held, routed_scale, count);
        Combine(held, count);
        PipeBarrier<PIPE_ALL>();
        DataCopy(out_global[done], held.wide, count);
    }
}

/** One stage of the unfused arm: 0 scales, 1 adds, 2 widens. */
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void moe_epilogue_stage(__gm__ bfloat16_t* routed, __gm__ bfloat16_t* shared,
                        __gm__ float* out, float routed_scale, int stage,
                        int64_t elements, int iterations)
{
    const Stripe mine = OwnStripe(elements);
    if (mine.span == 0) {
        return;
    }
    Buffers held(TILE_ELEMENTS);
    GlobalTensor<bfloat16_t> routed_global;
    GlobalTensor<bfloat16_t> shared_global;
    GlobalTensor<float> out_global;
    routed_global.SetGlobalBuffer(routed + mine.begin, mine.span);
    shared_global.SetGlobalBuffer(shared + mine.begin, mine.span);
    out_global.SetGlobalBuffer(out + mine.begin, mine.span);

    for (int step = 0; step < iterations; ++step)
    for (int64_t done = 0; done < mine.span; done += TILE_ELEMENTS) {
        const int64_t count = (mine.span - done < TILE_ELEMENTS)
                                  ? mine.span - done : TILE_ELEMENTS;
        if (stage == 0) {
            DataCopy(held.narrow, routed_global[done], count);
            PipeBarrier<PIPE_ALL>();
            Scale(held, routed_scale, count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(routed_global[done], held.narrow, count);
        } else if (stage == 1) {
            DataCopy(held.narrow, routed_global[done], count);
            DataCopy(held.other, shared_global[done], count);
            PipeBarrier<PIPE_ALL>();
            Combine(held, count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(routed_global[done], held.narrow, count);
        } else {
            DataCopy(held.narrow, routed_global[done], count);
            PipeBarrier<PIPE_ALL>();
            Cast(held.wide, held.narrow, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(out_global[done], held.wide, count);
        }
    }
}

void moe_epilogue_fused_launch(uint32_t blocks, void* stream, void* routed,
                               void* shared, float* out, float routed_scale,
                               int64_t elements, int iterations)
{
    moe_epilogue_fused<<<blocks, nullptr, stream>>>(
        (__gm__ bfloat16_t*)routed, (__gm__ bfloat16_t*)shared, out,
        routed_scale, elements, iterations);
}

void moe_epilogue_stage_launch(uint32_t blocks, void* stream, void* routed,
                               void* shared, float* out, float routed_scale,
                               int stage, int64_t elements,
                               int iterations)
{
    moe_epilogue_stage<<<blocks, nullptr, stream>>>(
        (__gm__ bfloat16_t*)routed, (__gm__ bfloat16_t*)shared, out,
        routed_scale, stage, elements, iterations);
}
