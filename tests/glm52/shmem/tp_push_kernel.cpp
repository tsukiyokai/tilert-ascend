/**
 * One-sided TP all-reduce: push-to-all, one cross-PE sync, ordered local sum.
 *
 * Every rank pushes its whole partial into the slot it owns in every peer's
 * window, so after one synchronization each rank holds all pe_size partials
 * locally. Each core then sums its own stripe over those slots in rank order,
 * which keeps the result bit-reproducible.
 *
 * There are no flags and no polling. A reduction needs no producer/consumer
 * core split: the split exists in examples/allgather because its consumers must
 * pull peers' payloads through a staging window, whereas here the push already
 * delivered every payload before the sync released anyone.
 *
 * Two planes alternate, which bounds how far a rank may run ahead with a single
 * sync per round: reaching round r+2 requires passing the sync of round r+1,
 * which every peer reaches only after finishing its round-r reduction.
 */
#include "tp_push_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

using namespace AscendC;

constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t PLANES = 8;
// The window head keeps the region the host still reserves for flags; the
// payload planes follow it.
constexpr int64_t SIGNAL_SLOT_INT32 = 16;
constexpr int64_t MAX_SLICES = 32;

// UB comes from a TPipe so the three buffers cannot overlap each other: the
// staging area for the push, and the accumulator and addend for the sum.
constexpr int64_t STAGE_UB_BYTES = 16 * 1024;
constexpr int64_t REDUCE_UB_BYTES = 32 * 1024;

ACLSHMEM_DEVICE int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct TpPushLayout {
    __gm__ float* data;
    int64_t stride;
    int64_t elements;
};

ACLSHMEM_DEVICE TpPushLayout TpPushPlane(__gm__ uint8_t* window, int rows, int hidden, int64_t pe_size, int64_t plane)
{
    TpPushLayout layout;
    layout.elements = static_cast<int64_t>(rows) * hidden;
    layout.stride = AlignUp(layout.elements * sizeof(float), ALIGN_BYTES) / sizeof(float);
    const int64_t head_int32 = PLANES * MAX_SLICES * SIGNAL_SLOT_INT32;
    layout.data = (__gm__ float*)((__gm__ int32_t*)window + head_int32) + plane * pe_size * layout.stride;
    return layout;
}

/** Core `core` sends the whole partial to one peer, so a fan-out of pe_size costs one put per core. */
ACLSHMEM_DEVICE void TpPushPublish(const TpPushLayout& layout, __gm__ float* partial, int64_t core, int64_t my_pe,
                                   int64_t pe_size, __ubuf__ float* stage)
{
    if (core >= pe_size) {
        return;
    }
    // Staggering the start spreads the first writes of different ranks over distinct links.
    const int64_t peer = (my_pe + core) % pe_size;
    aclshmemx_mte_put_nbi(layout.data + my_pe * layout.stride, partial, stage, STAGE_UB_BYTES, layout.elements, peer,
                          EVENT_ID0);
}

/** Sum every rank's slot in rank order over this core's stripe. */
ACLSHMEM_DEVICE void TpPushReduce(const TpPushLayout& layout, __gm__ float* out, int64_t core, int64_t cores,
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

/**
 * Publish by storing straight into the peer's address.
 *
 * aclshmem_ptr maps our symmetric pointer to the same offset in the peer's
 * window, after which a remote write is an ordinary MTE store. This is how the
 * shmem_mega_moe example communicates, and it skips the library's staging
 * protocol and its trailing quiet.
 */
ACLSHMEM_DEVICE void TpPushPublishDirect(const TpPushLayout& layout, __gm__ float* partial, int64_t core,
                                         int64_t my_pe, int64_t pe_size, const LocalTensor<float>& stage_tensor)
{
    if (core >= pe_size) {
        return;
    }
    const int64_t peer = (my_pe + core) % pe_size;
    __gm__ float* remote = (__gm__ float*)aclshmem_ptr((__gm__ void*)layout.data, peer);
    const int64_t chunk = STAGE_UB_BYTES / static_cast<int64_t>(sizeof(float));

    for (int64_t offset = 0; offset < layout.elements; offset += chunk) {
        const int64_t count = (layout.elements - offset < chunk) ? layout.elements - offset : chunk;
        GlobalTensor<float> source;
        source.SetGlobalBuffer(partial + offset, count);
        DataCopy(stage_tensor, source, count);
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(remote + my_pe * layout.stride + offset, count);
        DataCopy(sink, stage_tensor, count);
        PipeBarrier<PIPE_ALL>();
    }
}

/**
 * Publish with the dependency named rather than a whole-pipeline barrier.
 *
 * Two staging buffers and two events: MTE2 fills one while MTE3 drains the
 * other, which is how QuantizeLocalTokens in examples/shmem_mega_moe keeps the
 * load and store units both busy.
 */
ACLSHMEM_DEVICE void TpPushPublishPipelined(const TpPushLayout& layout, __gm__ float* partial, int64_t core,
                                            int64_t my_pe, int64_t pe_size, const LocalTensor<float>& first,
                                            const LocalTensor<float>& second)
{
    if (core >= pe_size) {
        return;
    }
    const int64_t peer = (my_pe + core) % pe_size;
    __gm__ float* remote = (__gm__ float*)aclshmem_ptr((__gm__ void*)layout.data, peer);
    __gm__ float* slot = remote + my_pe * layout.stride;
    const int64_t chunk = STAGE_UB_BYTES / static_cast<int64_t>(sizeof(float));
    const event_t events[2] = {EVENT_ID0, EVENT_ID1};

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[1]);
    int64_t index = 0;
    for (int64_t offset = 0; offset < layout.elements; offset += chunk) {
        const int64_t count = (layout.elements - offset < chunk) ? layout.elements - offset : chunk;
        const LocalTensor<float>& buffer = (index % 2 == 0) ? first : second;
        const event_t event = events[index % 2];
        ++index;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        GlobalTensor<float> source;
        source.SetGlobalBuffer(partial + offset, count);
        DataCopy(buffer, source, count);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);

        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(slot + offset, count);
        DataCopy(sink, buffer, count);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[1]);
}

/**
 * Sum every rank's slot over this core's stripe with the loads pipelined.
 *
 * The addend is double buffered so peer p+1's load is in flight while peer p
 * is being added, and the order of the additions is still ascending rank, so
 * the result stays bit-reproducible.
 */
ACLSHMEM_DEVICE void TpPushReducePipelined(const TpPushLayout& layout, __gm__ float* out, int64_t core, int64_t cores,
                                           int64_t pe_size, const LocalTensor<float>& acc,
                                           const LocalTensor<float>& first, const LocalTensor<float>& second)
{
    const int64_t stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(float));
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));
    const event_t events[2] = {EVENT_ID0, EVENT_ID1};

    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        GlobalTensor<float> source;
        source.SetGlobalBuffer(layout.data + begin + offset, count);
        DataCopy(acc, source, count);
        SetFlag<AscendC::HardEvent::MTE2_V>(events[0]);
        WaitFlag<AscendC::HardEvent::MTE2_V>(events[0]);

        // Prime the pipeline with the first peer; each iteration then issues
        // the next load before waiting on the current one.
        SetFlag<AscendC::HardEvent::V_MTE2>(events[1]);
        source.SetGlobalBuffer(layout.data + layout.stride + begin + offset, count);
        DataCopy(first, source, count);
        SetFlag<AscendC::HardEvent::MTE2_V>(events[0]);

        for (int64_t pe = 1; pe < pe_size; ++pe) {
            const int64_t current = (pe - 1) % 2;
            const int64_t next = pe % 2;
            const LocalTensor<float>& ready = (current == 0) ? first : second;
            const LocalTensor<float>& incoming = (next == 0) ? first : second;

            if (pe + 1 < pe_size) {
                WaitFlag<AscendC::HardEvent::V_MTE2>(events[next]);
                source.SetGlobalBuffer(layout.data + (pe + 1) * layout.stride + begin + offset, count);
                DataCopy(incoming, source, count);
                SetFlag<AscendC::HardEvent::MTE2_V>(events[next]);
            }
            WaitFlag<AscendC::HardEvent::MTE2_V>(events[current]);
            Add(acc, acc, ready, count);
            SetFlag<AscendC::HardEvent::V_MTE2>(events[current]);
        }
        WaitFlag<AscendC::HardEvent::V_MTE2>(events[0]);
        WaitFlag<AscendC::HardEvent::V_MTE2>(events[1]);

        SetFlag<AscendC::HardEvent::V_MTE3>(events[0]);
        WaitFlag<AscendC::HardEvent::V_MTE3>(events[0]);
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, acc, count);
        SetFlag<AscendC::HardEvent::MTE3_V>(events[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(events[0]);
    }
}

ACLSHMEM_DEVICE __gm__ int32_t* TpPushFlags(__gm__ uint8_t* window, int64_t plane)
{
    return (__gm__ int32_t*)window + plane * MAX_SLICES * SIGNAL_SLOT_INT32;
}

/**
 * Tell peer `core` that our slot in its window is complete.
 *
 * The flag must not become visible before the data it describes, so the store
 * pipeline is drained first -- a fence on one queue rather than a barrier over
 * every core and rank.
 */
ACLSHMEM_DEVICE void TpPushSignal(__gm__ uint8_t* window, int64_t core, int64_t my_pe, int64_t pe_size,
                                  int64_t plane, int32_t token)
{
    if (core >= pe_size) {
        return;
    }
    const int64_t peer = (my_pe + core) % pe_size;
    aclshmemx_signal_op(TpPushFlags(window, plane) + my_pe * SIGNAL_SLOT_INT32, token, ACLSHMEM_SIGNAL_SET,
                        (int)peer);
}

/**
 * Sum every rank's slot, each add gated only by its own producer's flag.
 *
 * The additions stay in ascending rank order so the result is bit-reproducible,
 * but the wait for peer p happens immediately before peer p's add rather than
 * after every peer has arrived.
 */
ACLSHMEM_DEVICE void TpPushReduceSignalled(const TpPushLayout& layout, __gm__ float* out, __gm__ uint8_t* window,
                                           int64_t core, int64_t cores, int64_t pe_size, int64_t plane, int32_t token,
                                           const LocalTensor<float>& acc, const LocalTensor<float>& term)
{
    const int64_t stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(float));
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));
    __gm__ int32_t* flags = TpPushFlags(window, plane);

    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        for (int64_t pe = 0; pe < pe_size; ++pe) {
            if (offset == 0) {
                aclshmem_signal_wait_until(flags + pe * SIGNAL_SLOT_INT32, ACLSHMEM_CMP_GE, token);
            }
            GlobalTensor<float> source;
            source.SetGlobalBuffer(layout.data + pe * layout.stride + begin + offset, count);
            if (pe == 0) {
                DataCopy(acc, source, count);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                continue;
            }
            DataCopy(term, source, count);
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            Add(acc, acc, term, count);
            SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, acc, count);
        SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

/**
 * Check one element of this core's stripe and count a mismatch.
 *
 * The position rotates with the round so 400 rounds cover 400 offsets, which
 * the host's single end-of-run comparison cannot do: with two planes it only
 * ever sees the last round of each.
 */
ACLSHMEM_DEVICE void TpPushProbe(__gm__ float* out, __gm__ int32_t* errors, int64_t core, int64_t cores,
                                 int64_t elements, int64_t round, float expected)
{
    const int64_t stripe = AlignUp((elements + cores - 1) / cores, ALIGN_BYTES / sizeof(float));
    const int64_t begin = core * stripe;
    if (begin >= elements) {
        return;
    }
    const int64_t span = (begin + stripe <= elements) ? stripe : elements - begin;
    GlobalTensor<float> probe;
    probe.SetGlobalBuffer(out + begin + round % span, 1);
    dcci_cacheline((__gm__ uint8_t*)(out + begin + round % span));
    if (probe.GetValue(0) != expected) {
        GlobalTensor<int32_t> counter;
        counter.SetGlobalBuffer(errors + core * SIGNAL_SLOT_INT32, 1);
        counter.SetValue(0, counter.GetValue(0) + 1);
        dcci_cacheline((__gm__ uint8_t*)(errors + core * SIGNAL_SLOT_INT32));
    }
}

/**
 * Publish a bf16 partial, pipelined, straight into the peer's window.
 *
 * Half the bytes of the float32 path for the same values, since the sum is
 * accumulated in float inside the reduction rather than on the wire.
 */
ACLSHMEM_DEVICE void TpPushPublishBf16(const TpPushLayout& layout, __gm__ bfloat16_t* partial, int64_t core,
                                       int64_t cores, int64_t my_pe, int64_t pe_size,
                                       const LocalTensor<bfloat16_t>& first,
                                       const LocalTensor<bfloat16_t>& second,
                                       __gm__ bfloat16_t* routed_slice = nullptr)
{
    const int64_t workers = (cores >= pe_size) ? (cores / pe_size) : 1;
    if (core >= pe_size * workers) {
        return;
    }
    const int64_t peer = (my_pe + (core % pe_size)) % pe_size;
    const int64_t worker = core / pe_size;
    const int64_t reduce_stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(bfloat16_t));
    const int64_t stripe = (cores >= pe_size)
                               ? (pe_size * reduce_stripe)
                               : AlignUp((layout.elements + workers - 1) / workers, ALIGN_BYTES / sizeof(bfloat16_t));
    const int64_t begin = worker * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    __gm__ bfloat16_t* remote = (__gm__ bfloat16_t*)aclshmem_ptr((__gm__ void*)layout.data, peer);
    __gm__ bfloat16_t* slot = remote + my_pe * layout.stride * 2 + begin;
    __gm__ bfloat16_t* local_partial = partial + begin;
    const int64_t chunk = STAGE_UB_BYTES / static_cast<int64_t>(sizeof(bfloat16_t));
    const event_t events[2] = {EVENT_ID0, EVENT_ID1};

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[1]);
    int64_t index = 0;
    if (peer != my_pe) {
        for (int64_t offset = 0; offset < span; offset += chunk) {
            const int64_t count = (span - offset < chunk) ? span - offset : chunk;
            const LocalTensor<bfloat16_t>& buffer = (index % 2 == 0) ? first : second;
            const event_t event = events[index % 2];
            ++index;

            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            GlobalTensor<bfloat16_t> source;
            source.SetGlobalBuffer(local_partial + offset, count);
            DataCopy(buffer, source, count);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);

            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            GlobalTensor<bfloat16_t> sink;
            sink.SetGlobalBuffer(slot + offset, count);
            DataCopy(sink, buffer, count);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        }
    }
    if (routed_slice != nullptr && peer != my_pe) {
        const int64_t slice_elems = layout.elements / pe_size;
        const int64_t r_stripe = AlignUp((slice_elems + workers - 1) / workers, ALIGN_BYTES / sizeof(bfloat16_t));
        const int64_t r_local_begin = worker * r_stripe;
        if (r_local_begin < slice_elems) {
            const int64_t r_span = (r_local_begin + r_stripe <= slice_elems) ? r_stripe : (slice_elems - r_local_begin);
            __gm__ bfloat16_t* local_routed = routed_slice + r_local_begin;
            __gm__ bfloat16_t* routed_slot = remote + layout.elements + my_pe * slice_elems + r_local_begin;
            for (int64_t offset = 0; offset < r_span; offset += chunk) {
                const int64_t count = (r_span - offset < chunk) ? r_span - offset : chunk;
                const LocalTensor<bfloat16_t>& buffer = (index % 2 == 0) ? first : second;
                const event_t event = events[index % 2];
                ++index;

                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
                GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(local_routed + offset, count);
                DataCopy(buffer, source, count);
                SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);

                WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
                GlobalTensor<bfloat16_t> sink;
                sink.SetGlobalBuffer(routed_slot + offset, count);
                DataCopy(sink, buffer, count);
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            }
        }
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[1]);
}

/**
 * Sum every rank's bf16 slot in float, in rank order, and store bf16.
 *
 * The widening and narrowing happen in UB around work the reduction already
 * does, so the host no longer needs a Cast on either side.
 */
ACLSHMEM_DEVICE void TpPushReduceBf16(const TpPushLayout& layout, __gm__ bfloat16_t* out, int64_t core, int64_t cores,
                                      int64_t pe_size, const LocalTensor<float>& acc, const LocalTensor<float>& term,
                                      const LocalTensor<bfloat16_t>& narrow, __gm__ bfloat16_t* routed = nullptr,
                                      float scale = 1.0f, __gm__ int32_t* flag_base = nullptr, int32_t wait_count = 0,
                                      __gm__ bfloat16_t* partial = nullptr, int64_t my_pe = -1,
                                      __gm__ int32_t* pool_base = nullptr, __gm__ bfloat16_t* routed_slice = nullptr)
{
    const int64_t stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(bfloat16_t));
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        if (flag_base != nullptr) {
            for (int64_t pe = 0; pe < pe_size; ++pe) {
                aclshmem_signal_wait_until(flag_base + pe * SIGNAL_SLOT_INT32, ACLSHMEM_CMP_GE, wait_count);
            }
        }
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));

    const event_t events[2] = {EVENT_ID0, EVENT_ID1};
    const LocalTensor<bfloat16_t> first = narrow[0];
    const LocalTensor<bfloat16_t> second = narrow[chunk];

    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        SetFlag<AscendC::HardEvent::V_MTE2>(events[0]);
        SetFlag<AscendC::HardEvent::V_MTE2>(events[1]);

        if (offset == 0 && flag_base != nullptr && my_pe != 0) {
            aclshmem_signal_wait_until(flag_base, ACLSHMEM_CMP_GE, wait_count);
        }
        WaitFlag<AscendC::HardEvent::V_MTE2>(events[0]);
        GlobalTensor<bfloat16_t> source;
        source.SetGlobalBuffer((my_pe == 0 && partial != nullptr)
                                   ? (partial + begin + offset)
                                   : ((__gm__ bfloat16_t*)layout.data + begin + offset),
                               count);
        DataCopy(first, source, count);
        SetFlag<AscendC::HardEvent::MTE2_V>(events[0]);

        for (int64_t pe = 0; pe < pe_size; ++pe) {
            const int64_t current = pe % 2;
            const int64_t next = (pe + 1) % 2;
            const LocalTensor<bfloat16_t>& ready = (current == 0) ? first : second;
            const LocalTensor<bfloat16_t>& incoming = (next == 0) ? first : second;

            if (pe + 1 < pe_size) {
                if (offset == 0 && flag_base != nullptr && (pe + 1) != my_pe) {
                    aclshmem_signal_wait_until(flag_base + (pe + 1) * SIGNAL_SLOT_INT32, ACLSHMEM_CMP_GE, wait_count);
                }
                WaitFlag<AscendC::HardEvent::V_MTE2>(events[next]);
                source.SetGlobalBuffer(((pe + 1) == my_pe && partial != nullptr)
                                           ? (partial + begin + offset)
                                           : ((__gm__ bfloat16_t*)layout.data + (pe + 1) * layout.stride * 2 + begin + offset),
                                       count);
                DataCopy(incoming, source, count);
                SetFlag<AscendC::HardEvent::MTE2_V>(events[next]);
            } else if (routed != nullptr) {
                const int64_t slice_elems = layout.elements / pe_size;
                const int64_t r_pe = begin / slice_elems;
                if (offset == 0 && pool_base != nullptr && r_pe != my_pe) {
                    const int64_t workers = (cores >= pe_size) ? (cores / pe_size) : 1;
                    const int64_t r_stripe = AlignUp((slice_elems + workers - 1) / workers, ALIGN_BYTES / sizeof(bfloat16_t));
                    const int64_t r_worker = (begin % slice_elems) / r_stripe;
                    aclshmem_signal_wait_until(pool_base + (r_worker * pe_size + r_pe) * SIGNAL_SLOT_INT32,
                                               ACLSHMEM_CMP_GE, wait_count);
                }
                WaitFlag<AscendC::HardEvent::V_MTE2>(events[next]);
                source.SetGlobalBuffer((r_pe == my_pe && routed_slice != nullptr)
                                           ? (routed_slice + (begin % slice_elems) + offset)
                                           : (routed + begin + offset),
                                       count);
                DataCopy(incoming, source, count);
                SetFlag<AscendC::HardEvent::MTE2_V>(events[next]);
            }
            WaitFlag<AscendC::HardEvent::MTE2_V>(events[current]);
            if (pe == 0) {
                Cast(acc, ready, AscendC::RoundMode::CAST_NONE, count);
                SetFlag<AscendC::HardEvent::V_MTE2>(events[current]);
            } else {
                Cast(term, ready, AscendC::RoundMode::CAST_NONE, count);
                SetFlag<AscendC::HardEvent::V_MTE2>(events[current]);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, term, count);
            }
            PipeBarrier<PIPE_V>();
        }
        const int64_t r_slot = pe_size % 2;
        const LocalTensor<bfloat16_t>& r_buf = (r_slot == 0) ? first : second;
        if (routed != nullptr) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[1 - r_slot]);
            WaitFlag<AscendC::HardEvent::MTE2_V>(events[r_slot]);
            Cast(term, r_buf, AscendC::RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Cast(first, acc, AscendC::RoundMode::CAST_RINT, count);
            if (scale != 1.0f) {
                Muls(term, term, scale, count);
                PipeBarrier<PIPE_V>();
                Cast(second, term, AscendC::RoundMode::CAST_RINT, count);
                PipeBarrier<PIPE_V>();
                Cast(term, second, AscendC::RoundMode::CAST_NONE, count);
            }
            PipeBarrier<PIPE_V>();
            Cast(acc, first, AscendC::RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, term, count);
            PipeBarrier<PIPE_V>();
        } else {
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[0]);
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[1]);
        }
        Cast(first, acc, AscendC::RoundMode::CAST_RINT, count);
        SetFlag<AscendC::HardEvent::V_MTE3>(events[0]);
        WaitFlag<AscendC::HardEvent::V_MTE3>(events[0]);
        GlobalTensor<bfloat16_t> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, first, count);
        SetFlag<AscendC::HardEvent::MTE3_V>(events[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(events[0]);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    }
}

ACLSHMEM_DEVICE void TpPushRsAgBf16(
    const TpPushLayout& layout, __gm__ bfloat16_t* partial, __gm__ bfloat16_t* out,
    int64_t core, int64_t cores, int64_t my_pe, int64_t pe_size,
    const LocalTensor<bfloat16_t>& stage_buf,
    const LocalTensor<float>& stage_second,
    const LocalTensor<float>& acc, const LocalTensor<float>& term,
    const LocalTensor<bfloat16_t>& narrow,
    __gm__ bfloat16_t* routed_full, __gm__ bfloat16_t* routed_slice, float scale,
    int64_t active_elements)
{
    const bool fast_8x48 = (pe_size == 8 && cores == 48);
    const int64_t workers = fast_8x48 ? 6 : (cores / pe_size);
    if (!fast_8x48 && core >= pe_size * workers) {
        return;
    }
    const int64_t worker = fast_8x48 ? (core >> 3) : (core / pe_size);
    const int64_t peer = fast_8x48 ? ((my_pe + (core & 7)) & 7) : ((my_pe + (core % pe_size)) % pe_size);
    const bool one_hop = (fast_8x48 && active_elements <= 24576);
    const int64_t slice_elems =
        (!one_hop && routed_slice == nullptr && routed_full == nullptr && fast_8x48 && (active_elements % 768) == 0)
            ? (active_elements >> 3)
            : (layout.elements / pe_size);

    const int64_t peer_rem = one_hop ? active_elements : (active_elements - peer * slice_elems);
    const int64_t peer_slice_active =
        (peer_rem <= 0) ? 0 : ((peer_rem < slice_elems) ? peer_rem : slice_elems);
    const int64_t peer_w_elems =
        fast_8x48 ? ((peer_slice_active * 0x2aaaaaabLL) >> 32) : (peer_slice_active / workers);
    const int64_t peer_w_off = worker * peer_w_elems;

    const int64_t my_rem = one_hop ? active_elements : (active_elements - my_pe * slice_elems);
    const int64_t my_slice_active =
        (my_rem <= 0) ? 0 : ((my_rem < slice_elems) ? my_rem : slice_elems);
    const int64_t w_elems =
        fast_8x48 ? ((my_slice_active * 0x2aaaaaabLL) >> 32) : (my_slice_active / workers);
    const int64_t w_off = worker * w_elems;

    auto counter_slot = (__gm__ int32_t*)aclshmemi_get_team_sync_counter(1 + core);
    const int32_t count = aclshmemi_load(counter_slot) + 1;
    aclshmemi_store(counter_slot, count);
    if (w_elems == 0 && peer_w_elems == 0) {
        return;
    }

    __gm__ bfloat16_t* local_rs_win = (__gm__ bfloat16_t*)layout.data;
    __gm__ bfloat16_t* local_ag_win = local_rs_win + layout.elements;
    __gm__ bfloat16_t* remote_rs_win =
        (peer != my_pe) ? (__gm__ bfloat16_t*)aclshmem_ptr((__gm__ void*)local_rs_win, peer) : nullptr;
    __gm__ bfloat16_t* remote_ag_win =
        (peer != my_pe) ? (remote_rs_win + layout.elements) : nullptr;

    __gm__ int32_t* pool_base = (__gm__ int32_t*)aclshmemi_get_team_sync_pool(1);
    __gm__ int32_t* flag_rs = pool_base + worker * pe_size * SIGNAL_SLOT_INT32;
    __gm__ int32_t* flag_ag = pool_base + (workers + worker) * pe_size * SIGNAL_SLOT_INT32;

    if (peer != my_pe && peer_w_elems > 0) {
        GlobalTensor<bfloat16_t> src;
        const int64_t src_off = one_hop ? peer_w_off : (peer * slice_elems + peer_w_off);
        src.SetGlobalBuffer(partial + src_off, peer_w_elems);
        DataCopy(stage_buf, src, peer_w_elems);
        if (one_hop && routed_slice != nullptr && my_pe == 0) {
            GlobalTensor<bfloat16_t> r_src;
            r_src.SetGlobalBuffer(routed_slice + peer_w_off, peer_w_elems);
            DataCopy(stage_buf[4096], r_src, peer_w_elems);
        }
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

        GlobalTensor<bfloat16_t> dst;
        dst.SetGlobalBuffer(remote_rs_win + my_pe * slice_elems + peer_w_off, peer_w_elems);
        DataCopy(dst, stage_buf, peer_w_elems);
        if (one_hop && routed_slice != nullptr && my_pe == 0) {
            GlobalTensor<bfloat16_t> r_dst;
            r_dst.SetGlobalBuffer(remote_ag_win + peer_w_off, peer_w_elems);
            DataCopy(r_dst, stage_buf[4096], peer_w_elems);
        }
        SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        aclshmemi_signal_set(flag_rs + my_pe * SIGNAL_SLOT_INT32, (int)peer, count);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if (one_hop && peer != my_pe) {
        return;
    }

    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));
    const event_t events[2] = {EVENT_ID0, EVENT_ID1};
    const LocalTensor<bfloat16_t> first = narrow[0];
    const LocalTensor<bfloat16_t> second = narrow[chunk];
    const int64_t my_global_off = one_hop ? w_off : (my_pe * slice_elems + w_off);
    const bool preload_local = (w_elems > 0) && (w_elems <= (STAGE_UB_BYTES / static_cast<int64_t>(sizeof(float))));
    const bool remote_routed = (one_hop && routed_slice != nullptr && my_pe != 0);

    if (preload_local) {
        GlobalTensor<bfloat16_t> my_src;
        my_src.SetGlobalBuffer(partial + my_global_off, w_elems);
        DataCopy(stage_buf, my_src, w_elems);
        if (!remote_routed && (routed_slice != nullptr || routed_full != nullptr)) {
            GlobalTensor<bfloat16_t> r_src;
            r_src.SetGlobalBuffer((routed_slice != nullptr) ? (routed_slice + w_off) : (routed_full + my_global_off),
                                  w_elems);
            DataCopy(stage_buf[4096], r_src, w_elems);
        }
        SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (!remote_routed && (routed_slice != nullptr || routed_full != nullptr)) {
            Cast(stage_second, stage_buf[4096], AscendC::RoundMode::CAST_NONE, w_elems);
            if (scale != 1.0f) {
                PipeBarrier<PIPE_V>();
                Muls(stage_second, stage_second, scale, w_elems);
                PipeBarrier<PIPE_V>();
                Cast(stage_buf[4096], stage_second, AscendC::RoundMode::CAST_RINT, w_elems);
                PipeBarrier<PIPE_V>();
                Cast(stage_second, stage_buf[4096], AscendC::RoundMode::CAST_NONE, w_elems);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    for (int64_t offset = 0; offset < w_elems; offset += chunk) {
        const int64_t c = (w_elems - offset < chunk) ? (w_elems - offset) : chunk;
        SetFlag<AscendC::HardEvent::V_MTE2>(events[0]);
        SetFlag<AscendC::HardEvent::V_MTE2>(events[1]);

        GlobalTensor<bfloat16_t> source;
        if (!(preload_local && my_pe == 0)) {
            if (offset == 0 && my_pe != 0) {
                aclshmem_signal_wait_until(flag_rs, ACLSHMEM_CMP_GE, count);
            }
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[0]);
            source.SetGlobalBuffer((my_pe == 0) ? (partial + my_global_off + offset)
                                                : (local_rs_win + w_off + offset),
                                   c);
            DataCopy(first, source, c);
            if (remote_routed) {
                GlobalTensor<bfloat16_t> r_src;
                r_src.SetGlobalBuffer(local_ag_win + w_off + offset, c);
                DataCopy(stage_buf[4096], r_src, c);
            }
            SetFlag<AscendC::HardEvent::MTE2_V>(events[0]);
        }

        for (int64_t pe = 0; pe < pe_size; ++pe) {
            const int64_t current = pe % 2;
            const int64_t next = (pe + 1) % 2;
            const LocalTensor<bfloat16_t>& ready = (current == 0) ? first : second;
            const LocalTensor<bfloat16_t>& incoming = (next == 0) ? first : second;

            if (pe + 1 < pe_size) {
                if (!(preload_local && (pe + 1) == my_pe)) {
                    if (offset == 0 && (pe + 1) != my_pe) {
                        aclshmem_signal_wait_until(flag_rs + (pe + 1) * SIGNAL_SLOT_INT32, ACLSHMEM_CMP_GE, count);
                    }
                    WaitFlag<AscendC::HardEvent::V_MTE2>(events[next]);
                    source.SetGlobalBuffer(((pe + 1) == my_pe)
                                               ? (partial + my_global_off + offset)
                                               : (local_rs_win + (pe + 1) * slice_elems + w_off + offset),
                                           c);
                    DataCopy(incoming, source, c);
                    SetFlag<AscendC::HardEvent::MTE2_V>(events[next]);
                }
            } else if (!preload_local && (routed_slice != nullptr || routed_full != nullptr)) {
                WaitFlag<AscendC::HardEvent::V_MTE2>(events[next]);
                source.SetGlobalBuffer((routed_slice != nullptr)
                                           ? (routed_slice + w_off + offset)
                                           : (routed_full + my_global_off + offset),
                                       c);
                DataCopy(incoming, source, c);
                SetFlag<AscendC::HardEvent::MTE2_V>(events[next]);
            }
            if (preload_local && pe == my_pe) {
                if (pe == 0) {
                    Cast(acc, stage_buf, AscendC::RoundMode::CAST_NONE, c);
                } else {
                    Cast(term, stage_buf, AscendC::RoundMode::CAST_NONE, c);
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, term, c);
                }
            } else {
                WaitFlag<AscendC::HardEvent::MTE2_V>(events[current]);
                if (pe == 0) {
                    Cast(acc, ready, AscendC::RoundMode::CAST_NONE, c);
                    SetFlag<AscendC::HardEvent::V_MTE2>(events[current]);
                } else {
                    Cast(term, ready, AscendC::RoundMode::CAST_NONE, c);
                    SetFlag<AscendC::HardEvent::V_MTE2>(events[current]);
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, term, c);
                }
            }
            PipeBarrier<PIPE_V>();
        }
        const int64_t r_slot = pe_size % 2;
        const LocalTensor<bfloat16_t>& r_buf = (r_slot == 0) ? first : second;
        if (preload_local) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[0]);
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[1]);
            if (remote_routed) {
                Cast(stage_second, stage_buf[4096], AscendC::RoundMode::CAST_NONE, c);
                if (scale != 1.0f) {
                    PipeBarrier<PIPE_V>();
                    Muls(stage_second, stage_second, scale, c);
                    PipeBarrier<PIPE_V>();
                    Cast(stage_buf[4096], stage_second, AscendC::RoundMode::CAST_RINT, c);
                    PipeBarrier<PIPE_V>();
                    Cast(stage_second, stage_buf[4096], AscendC::RoundMode::CAST_NONE, c);
                }
                PipeBarrier<PIPE_V>();
            }
            if (routed_slice != nullptr || routed_full != nullptr) {
                Cast(first, acc, AscendC::RoundMode::CAST_RINT, c);
                PipeBarrier<PIPE_V>();
                Cast(acc, first, AscendC::RoundMode::CAST_NONE, c);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, stage_second, c);
                PipeBarrier<PIPE_V>();
            }
        } else if (routed_slice != nullptr || routed_full != nullptr) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[1 - r_slot]);
            WaitFlag<AscendC::HardEvent::MTE2_V>(events[r_slot]);
            Cast(term, r_buf, AscendC::RoundMode::CAST_NONE, c);
            PipeBarrier<PIPE_V>();
            Cast(first, acc, AscendC::RoundMode::CAST_RINT, c);
            if (scale != 1.0f) {
                Muls(term, term, scale, c);
                PipeBarrier<PIPE_V>();
                Cast(second, term, AscendC::RoundMode::CAST_RINT, c);
                PipeBarrier<PIPE_V>();
                Cast(term, second, AscendC::RoundMode::CAST_NONE, c);
            }
            PipeBarrier<PIPE_V>();
            Cast(acc, first, AscendC::RoundMode::CAST_NONE, c);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, term, c);
            PipeBarrier<PIPE_V>();
        } else {
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[0]);
            WaitFlag<AscendC::HardEvent::V_MTE2>(events[1]);
        }
        Cast(first, acc, AscendC::RoundMode::CAST_RINT, c);
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(events[0]);

        GlobalTensor<bfloat16_t> sink;
        if (peer == my_pe) {
            sink.SetGlobalBuffer(out + my_global_off + offset, c);
        } else {
            sink.SetGlobalBuffer(remote_ag_win + my_global_off + offset, c);
        }
        DataCopy(sink, first, c);
        SetFlag<AscendC::HardEvent::MTE3_V>(events[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(events[0]);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    }

    if (peer != my_pe) {
        if (w_elems > 0) {
            SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            aclshmemi_signal_set(flag_ag + my_pe * SIGNAL_SLOT_INT32, (int)peer, count);
        }

        if (peer_w_elems > 0) {
            aclshmem_signal_wait_until(flag_ag + peer * SIGNAL_SLOT_INT32, ACLSHMEM_CMP_GE, count);
            const int64_t peer_global_off = peer * slice_elems + peer_w_off;
            GlobalTensor<bfloat16_t> ag_src;
            ag_src.SetGlobalBuffer(local_ag_win + peer_global_off, peer_w_elems);
            DataCopy(stage_buf, ag_src, peer_w_elems);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

            GlobalTensor<bfloat16_t> ag_dst;
            ag_dst.SetGlobalBuffer(out + peer_global_off, peer_w_elems);
            DataCopy(ag_dst, stage_buf, peer_w_elems);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}

ACLSHMEM_DEVICE void tp_push_once(__gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, int rows,
                                  int hidden, int64_t round, int stage, __ubuf__ float* stage_ub,
                                  __gm__ int32_t* errors, const LocalTensor<float>& stage_tensor,
                                  const LocalTensor<float>& stage_second, const LocalTensor<float>& term_second,
                                  const LocalTensor<float>& acc, const LocalTensor<float>& term,
                                  float scale = 1.0f, const __gm__ int32_t* active_ptr = nullptr)
{
    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    const int64_t plane = (stage == 9) ? 0 : round % PLANES;
    const TpPushLayout layout = TpPushPlane(window, (stage == 14) ? 1 : rows,
                                            (stage == 14) ? 196608 : hidden, pe_size, plane);

    // stage 1 is the push half alone and stage 2 the sum half alone; neither waits on a peer, so
    // isolating a half can expose a cost but can no longer deadlock.
    // stage 3 moves nothing and only synchronises, which measures the floor the
    // 17.5 us push half is sitting on: it decides whether reduce-scatter plus
    // all-gather, eight times fewer bytes for a second sync, can win.
    if (stage == 3) {
        aclshmem_quiet();
        aclshmemx_sync_vec_all();
        return;
    }
    if (stage == 14) {
        int active_rows = rows;
        if (active_ptr != nullptr && rows == 32 && (core < pe_size || core == 0)) {
            dcci((__gm__ uint64_t*)active_ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
            const int a = *active_ptr;
            if (a >= 1 && a <= rows) {
                active_rows = a;
            }
        }
        if (core < pe_size) {
            dcci((__gm__ uint64_t*)partial, ENTIRE_DATA_CACHE, CACHELINE_OUT);
            dcci((__gm__ uint64_t*)errors, ENTIRE_DATA_CACHE, CACHELINE_OUT);
            __ubuf__ float* ub_vals = stage_ub;
            __ubuf__ int32_t* ub_idxs = (__ubuf__ int32_t*)(stage_ub + 64);
            __gm__ int64_t* gm_idxs = (__gm__ int64_t*)errors;
            for (int i = 0; i < active_rows; ++i) {
                ub_vals[i] = partial[i];
                ub_idxs[i] = static_cast<int32_t>(gm_idxs[i]) + hidden;
            }
            SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            const int64_t peer = (my_pe + core) % pe_size;
            __gm__ float* remote =
                (__gm__ float*)aclshmem_ptr(layout.data + my_pe * 128, (int)peer);
            GlobalTensor<float> sink;
            sink.SetGlobalBuffer(remote, 128);
            DataCopy(sink, stage_tensor, 128);
            SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }
        aclshmem_quiet();
        const int64_t workers = (cores >= pe_size) ? (cores / pe_size) : 1;
        __gm__ int32_t* flag_base = nullptr;
        int32_t count = 0;
        if (core < pe_size * workers) {
            auto counter_slot = (__gm__ int32_t*)aclshmemi_get_team_sync_counter(1 + core);
            count = aclshmemi_load(counter_slot) + 1;
            aclshmemi_store(counter_slot, count);
            const int64_t worker = core / pe_size;
            const int64_t peer = (my_pe + (core % pe_size)) % pe_size;
            flag_base = (__gm__ int32_t*)(aclshmemi_get_team_sync_pool(1) + worker * pe_size);
            aclshmemi_signal_set(flag_base + my_pe * SIGNAL_SLOT_INT32, (int)peer, count);
        }
        if (core == 0) {
            for (int64_t pe = 0; pe < pe_size; ++pe) {
                aclshmem_signal_wait_until(flag_base + pe * SIGNAL_SLOT_INT32, ACLSHMEM_CMP_GE, count);
            }
            GlobalTensor<float> source;
            source.SetGlobalBuffer(layout.data, pe_size * 128);
            DataCopy(term, source, pe_size * 128);
            SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            __ubuf__ float* all_vals = (__ubuf__ float*)term.GetPhyAddr();
            __gm__ int64_t* out_idxs = (__gm__ int64_t*)out;
            for (int i = 0; i < active_rows; ++i) {
                float best_val = all_vals[i];
                int32_t best_idx = reinterpret_cast<__ubuf__ int32_t*>(all_vals + 64)[i];
                for (int64_t pe = 1; pe < pe_size; ++pe) {
                    __ubuf__ float* p_vals = all_vals + pe * 128;
                    __ubuf__ int32_t* p_idxs = reinterpret_cast<__ubuf__ int32_t*>(p_vals + 64);
                    const float v = p_vals[i];
                    if (v > best_val) {
                        best_val = v;
                        best_idx = p_idxs[i];
                    }
                }
                out_idxs[i] = static_cast<int64_t>(best_idx);
            }
            for (int i = active_rows; i < rows; ++i) {
                out_idxs[i] = 0;
            }
            dcci((__gm__ uint64_t*)out, ENTIRE_DATA_CACHE, CACHELINE_OUT);
        }
        return;
    }
    // Stages 10 and 11 run one protocol -- pipelined push, one quiet, one
    // cross-PE sync, pipelined ordered sum -- over bf16 and over float32, so
    // the pair prices the element type and nothing else. The reduction's
    // staging tensor is the addend's buffer rather than the push's, so the
    // narrowed slot never shares UB with a store that may still be in flight.
    if (stage == 10 || stage == 13 || stage == 15) {
        const int64_t workers = (cores >= pe_size) ? (cores / pe_size) : 1;
        if (cores >= pe_size &&
            layout.elements % (pe_size * workers * 16) == 0 &&
            (layout.elements / (pe_size * workers)) <= (STAGE_UB_BYTES / static_cast<int64_t>(sizeof(bfloat16_t)))) {
            int64_t active_elements = layout.elements;
            if (active_ptr != nullptr && layout.elements == 196608) {
                dcci((__gm__ uint64_t*)active_ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
                const int a = *active_ptr;
                if (a >= 1 && a <= 32) {
                    active_elements = static_cast<int64_t>(a) * 6144;
                }
            }
            __gm__ bfloat16_t* routed_full = (stage == 13) ? (__gm__ bfloat16_t*)errors : nullptr;
            __gm__ bfloat16_t* routed_slice = (stage == 15) ? (__gm__ bfloat16_t*)errors : nullptr;
            TpPushRsAgBf16(layout, (__gm__ bfloat16_t*)partial, (__gm__ bfloat16_t*)out,
                           core, cores, my_pe, pe_size,
                           stage_tensor.ReinterpretCast<bfloat16_t>(), stage_second, acc, term,
                           term_second.ReinterpretCast<bfloat16_t>(),
                           routed_full, routed_slice, scale, active_elements);
            return;
        }
        __gm__ bfloat16_t* routed_slice = (stage == 15) ? (__gm__ bfloat16_t*)errors : nullptr;
        TpPushPublishBf16(layout, (__gm__ bfloat16_t*)partial, core, cores, my_pe, pe_size,
                          stage_tensor.ReinterpretCast<bfloat16_t>(), stage_second.ReinterpretCast<bfloat16_t>(),
                          routed_slice);
        SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        __gm__ int32_t* pool_base = (__gm__ int32_t*)aclshmemi_get_team_sync_pool(1);
        __gm__ int32_t* flag_base = nullptr;
        int32_t count = 0;
        if (core < pe_size * workers) {
            auto counter_slot = (__gm__ int32_t*)aclshmemi_get_team_sync_counter(1 + core);
            count = aclshmemi_load(counter_slot) + 1;
            aclshmemi_store(counter_slot, count);
            const int64_t worker = core / pe_size;
            const int64_t peer = (my_pe + (core % pe_size)) % pe_size;
            flag_base = pool_base + worker * pe_size * SIGNAL_SLOT_INT32;
            if (peer != my_pe) {
                aclshmemi_signal_set(flag_base + my_pe * SIGNAL_SLOT_INT32, (int)peer, count);
            }
        }
        __gm__ bfloat16_t* routed =
            (stage == 13) ? (__gm__ bfloat16_t*)errors
                          : ((stage == 15) ? ((__gm__ bfloat16_t*)layout.data + layout.elements) : nullptr);
        TpPushReduceBf16(layout, (__gm__ bfloat16_t*)out, core, cores, pe_size, acc, term,
                         term_second.ReinterpretCast<bfloat16_t>(), routed, scale, flag_base, count,
                         (__gm__ bfloat16_t*)partial, my_pe, (stage == 15) ? pool_base : nullptr, routed_slice);
        return;
    }
    if (stage == 11) {
        TpPushPublishPipelined(layout, partial, core, my_pe, pe_size, stage_tensor, stage_second);
        aclshmem_quiet();
        aclshmemx_sync_vec_all();
        TpPushReducePipelined(layout, out, core, cores, pe_size, acc, term, term_second);
        return;
    }
    // Stage 12 drops the sync from stage 10 exactly as stage 7 drops it from
    // stage 11, so the four stages price the element type at both protocol
    // levels. A sum taken without the sync may read a slot its peer has not
    // written, so stages 7 and 12 are timing probes and not reductions.
    if (stage == 12) {
        TpPushPublishBf16(layout, (__gm__ bfloat16_t*)partial, core, cores, my_pe, pe_size,
                          stage_tensor.ReinterpretCast<bfloat16_t>(), stage_second.ReinterpretCast<bfloat16_t>());
        TpPushReduceBf16(layout, (__gm__ bfloat16_t*)out, core, cores, pe_size, acc, term,
                         term_second.ReinterpretCast<bfloat16_t>());
        return;
    }
    if (stage == 4) {
        TpPushPublishDirect(layout, partial, core, my_pe, pe_size, stage_tensor);
    } else if (stage == 5 || stage == 6 || stage == 7 || stage == 8 || stage == 9) {
        TpPushPublishPipelined(layout, partial, core, my_pe, pe_size, stage_tensor, stage_second);
    } else if (stage != 2) {
        TpPushPublish(layout, partial, core, my_pe, pe_size, stage_ub);
        // Drain this core's remote stores before anything claims they landed.
        if (stage >= 4) {
            SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }
        if (stage == 9) {
        TpPushReducePipelined(layout, out, core, cores, pe_size, acc, term, term_second);
        // No peer may begin its next push until every peer has finished
        // reading, which is what makes a single plane safe.
        aclshmemx_sync_vec_all();
    } else if (stage == 8) {
            TpPushSignal(window, core, my_pe, pe_size, round % PLANES, (int32_t)round + 1);
        } else {
            if (stage != 6) {
                aclshmem_quiet();
            }
            aclshmemx_sync_vec_all();
        }
    }
    if (stage == 8) {
        TpPushReduceSignalled(layout, out, window, core, cores, pe_size, round % PLANES, (int32_t)round + 1, acc, term);
        TpPushProbe(out, errors, core, cores, layout.elements, round, (float)(pe_size * (pe_size - 1) / 2));
    } else if (stage == 7) {
        TpPushReducePipelined(layout, out, core, cores, pe_size, acc, term, term_second);
    } else if (stage != 1) {
        TpPushReduce(layout, out, core, cores, pe_size, acc, term);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void tp_push_kernel(
    uint64_t fftsAddr, __gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, __gm__ int32_t* errors,
    int rows, int hidden, int rounds, int round_base, int stage, const __gm__ int32_t* active_ptr)
{
    util_set_ffts_config(fftsAddr);
    // One carve-up of UB for the whole launch; a TPipe inside the round loop would redo it every round.
    TPipe pipe;
    TBuf<TPosition::VECCALC> staging;
    TBuf<TPosition::VECCALC> accumulator;
    TBuf<TPosition::VECCALC> addend;
    pipe.InitBuffer(staging, STAGE_UB_BYTES);
    TBuf<TPosition::VECCALC> staging_second;
    pipe.InitBuffer(staging_second, STAGE_UB_BYTES);
    TBuf<TPosition::VECCALC> addend_second;
    pipe.InitBuffer(addend_second, REDUCE_UB_BYTES);
    pipe.InitBuffer(accumulator, REDUCE_UB_BYTES);
    pipe.InitBuffer(addend, REDUCE_UB_BYTES);
    LocalTensor<float> stage_tensor = staging.Get<float>();
    LocalTensor<float> stage_second = staging_second.Get<float>();
    LocalTensor<float> term_second = addend_second.Get<float>();
    LocalTensor<float> acc = accumulator.Get<float>();
    LocalTensor<float> term = addend.Get<float>();
    __ubuf__ float* stage_ub = (__ubuf__ float*)stage_tensor.GetPhyAddr();

    const int actual_rounds = (stage == 13 || stage == 14 || stage == 15) ? 1 : rounds;
    union {
        int32_t i;
        float f;
    } scale_bits = {rounds};
    const float scale = (stage == 13 || stage == 15) ? scale_bits.f : 1.0f;

    for (int round = 0; round < actual_rounds; ++round) {
        tp_push_once(partial, out, window, rows, hidden, static_cast<int64_t>(round_base) + round, stage, stage_ub,
                     errors, stage_tensor, stage_second, term_second, acc,
                     term, scale, active_ptr);
    }
}

void tp_push_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, float* partial, float* out, uint8_t* window,
                  int32_t* errors, int rows, int hidden, int rounds, int round_base, int stage,
                  const int32_t* active_ptr)
{
    tp_push_kernel<<<block_dim, nullptr, stream>>>(fftsAddr, partial, out, window, errors, rows, hidden, rounds,
                                                   round_base, stage, (const __gm__ int32_t*)active_ptr);
}

void tp_push_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, float* partial, float* out, uint8_t* window,
                  int32_t* errors, int rows, int hidden, int rounds, int round_base, int stage)
{
    tp_push_demo(block_dim, stream, fftsAddr, partial, out, window, errors, rows, hidden, rounds, round_base, stage,
                 nullptr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void indexer_prep_qk_kernel(
    __gm__ bfloat16_t* kw, __gm__ bfloat16_t* q_in, __gm__ bfloat16_t* cos_gm, __gm__ bfloat16_t* sin_gm,
    __gm__ float* ln_weight, __gm__ float* ln_bias, __gm__ void* slot_mapping, __gm__ bfloat16_t* k_cache,
    __gm__ bfloat16_t* qk_or_q_out, __gm__ uint16_t* weights_out,
    int T, int N, int64_t num_slots, float eps, int is_c8, int slot_is_i64,
    const __gm__ int32_t* active_ptr)
{
    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    if (T <= 0 || T > 64) {
        return;
    }

    int active_T = T;
    int64_t fast_t0 = 0;
    int64_t fast_begin_q = 0;
    int64_t fast_q_count = 0;
    if (T == 32 && N == 32 && cores == 48) {
        if (active_ptr != nullptr) {
            dcci((__gm__ uint64_t*)active_ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
            const int a = *active_ptr;
            if (a >= 1 && a <= 32) {
                active_T = a;
            }
        }
        if (core < 16) {
            if (active_T <= 16) {
                const int64_t shift = (active_T <= 2) ? 3 : ((active_T <= 4) ? 2 : ((active_T <= 8) ? 1 : 0));
                fast_t0 = core >> shift;
                if (fast_t0 >= active_T) {
                    return;
                }
                fast_q_count = 32 >> shift;
                fast_begin_q = (fast_t0 << 5) + ((core & ((1LL << shift) - 1)) * fast_q_count);
            } else {
                fast_t0 = core * 2;
                if (fast_t0 >= active_T) {
                    return;
                }
                fast_begin_q = core * 64;
                fast_q_count = (fast_t0 + 1 < active_T) ? 64 : 32;
            }
        } else if (core - 16 >= active_T) {
            return;
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> b_all;
    pipe.InitBuffer(b_all, 188 * 1024);
    LocalTensor<float> f_all = b_all.Get<float>();
    LocalTensor<bfloat16_t> bf_all = f_all.ReinterpretCast<bfloat16_t>();
    LocalTensor<half> h_all = f_all.ReinterpretCast<half>();

    LocalTensor<float> ub_ln_w = f_all[0];
    LocalTensor<float> ub_ln_b = f_all[128];
    LocalTensor<float> ub_kw_f = f_all[256];
    LocalTensor<float> ub_kw_swap_f = f_all[416];
    LocalTensor<float> ub_cos_f = f_all[576];
    LocalTensor<float> ub_sin_f = f_all[4672];
    LocalTensor<float> ub_cos2_f = f_all[8768];
    LocalTensor<float> ub_sin2_f = f_all[12864];
    LocalTensor<float> ub_q_f = f_all[16960];
    LocalTensor<float> ub_q_swap_f = f_all[25152];
    LocalTensor<float> ub_work = f_all[33344];

    LocalTensor<bfloat16_t> ub_kw_bf16 = bf_all[70784];
    LocalTensor<bfloat16_t> ub_kw_swap_bf16 = bf_all[70944];
    LocalTensor<bfloat16_t> ub_k_bf16 = bf_all[71104];
    LocalTensor<half> ub_w_fp16 = h_all[71232];
    LocalTensor<bfloat16_t> ub_cos_bf16 = bf_all[71264];
    LocalTensor<bfloat16_t> ub_sin_bf16 = bf_all[75360];
    LocalTensor<bfloat16_t> ub_q_bf16 = bf_all[79456];
    LocalTensor<bfloat16_t> ub_q_swap_bf16 = bf_all[87648];

    if (T == 32 && N == 32 && cores == 48) {
        if (core < 16) {
            const int64_t t0 = fast_t0;
            const int64_t begin_q = fast_begin_q;
            const int64_t q_count = fast_q_count;
            GlobalTensor<bfloat16_t> g_cos, g_sin, g_qin;
            g_cos.SetGlobalBuffer(cos_gm + t0 * 64, 128);
            g_sin.SetGlobalBuffer(sin_gm + t0 * 64, 128);
            g_qin.SetGlobalBuffer(q_in + begin_q * 128, q_count * 128);
            DataCopy(ub_cos_bf16, g_cos, 128);
            DataCopy(ub_sin_bf16, g_sin, 128);
            DataCopy(ub_q_bf16, g_qin, q_count * 128);

            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            Cast(ub_cos_f, ub_cos_bf16, AscendC::RoundMode::CAST_NONE, 128);
            Cast(ub_sin_f, ub_sin_bf16, AscendC::RoundMode::CAST_NONE, 128);

            LocalTensor<uint32_t> q_u32 = ub_q_bf16.ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> tmp1_u32 = ub_q_f.ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> tmp2_u32 = ub_q_swap_f.ReinterpretCast<uint32_t>();
            ShiftLeft(tmp1_u32, q_u32, (uint32_t)16, q_count * 64);
            ShiftRight(tmp2_u32, q_u32, (uint32_t)16, q_count * 64);
            Muls(ub_sin2_f, ub_sin_f, -1.0f, 128);
            ShiftRight(ub_work.ReinterpretCast<uint32_t>(), ub_cos_f.ReinterpretCast<uint32_t>(), (uint32_t)16, 128);
            PipeBarrier<PIPE_V>();
            Or(ub_q_swap_bf16.ReinterpretCast<uint16_t>(),
               tmp1_u32.ReinterpretCast<uint16_t>(),
               tmp2_u32.ReinterpretCast<uint16_t>(),
               q_count * 128);
            Or(ub_cos_bf16.ReinterpretCast<uint16_t>(),
               ub_cos_f.ReinterpretCast<uint16_t>(),
               ub_work.ReinterpretCast<uint16_t>(),
               256);
            ShiftRight(ub_work.ReinterpretCast<uint32_t>(), ub_sin2_f.ReinterpretCast<uint32_t>(), (uint32_t)16, 128);
            PipeBarrier<PIPE_V>();
            Or(ub_sin_bf16.ReinterpretCast<uint16_t>(),
               ub_sin_f.ReinterpretCast<uint16_t>(),
               ub_work.ReinterpretCast<uint16_t>(),
               256);
            Cast(ub_cos2_f, ub_cos_bf16, AscendC::RoundMode::CAST_NONE, 192);
            Cast(ub_q_f, ub_q_bf16, AscendC::RoundMode::CAST_NONE, q_count * 128);
            Cast(ub_q_swap_f, ub_q_swap_bf16, AscendC::RoundMode::CAST_NONE, q_count * 128);
            PipeBarrier<PIPE_V>();
            Cast(ub_sin2_f, ub_sin_bf16, AscendC::RoundMode::CAST_NONE, 192);
            PipeBarrier<PIPE_V>();

            const int64_t r0 = (q_count <= 32) ? q_count : 32;
            Mul(ub_q_f[0], ub_q_f[0], ub_cos2_f[0], 64, r0,
                AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
            Mul(ub_q_swap_f[0], ub_q_swap_f[0], ub_sin2_f[0], 64, r0,
                AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
            if (q_count == 64) {
                Mul(ub_q_f[32 * 128], ub_q_f[32 * 128], ub_cos2_f[128], 64, 32,
                    AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
                Mul(ub_q_swap_f[32 * 128], ub_q_swap_f[32 * 128], ub_sin2_f[128], 64, 32,
                    AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
            }
            PipeBarrier<PIPE_V>();
            Add(ub_q_f, ub_q_f, ub_q_swap_f, 64, q_count,
                AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            Cast(ub_q_bf16, ub_q_f, AscendC::RoundMode::CAST_RINT, q_count * 128);
            SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            GlobalTensor<bfloat16_t> g_qout;
            __gm__ bfloat16_t* q_dst = is_c8 ? (qk_or_q_out + (T + begin_q) * 128) : (qk_or_q_out + begin_q * 128);
            g_qout.SetGlobalBuffer(q_dst, q_count * 128);
            DataCopy(g_qout, ub_q_bf16, q_count * 128);
            return;
        }

        const int64_t t = core - 16;
        GlobalTensor<bfloat16_t> g_cos, g_sin, g_kw;
        GlobalTensor<float> g_w, g_b;
        g_cos.SetGlobalBuffer(cos_gm + t * 64, 32);
        g_sin.SetGlobalBuffer(sin_gm + t * 64, 32);
        g_w.SetGlobalBuffer(ln_weight, 128);
        g_b.SetGlobalBuffer(ln_bias, 128);
        g_kw.SetGlobalBuffer(kw + t * 160, 160);
        DataCopy(ub_cos_bf16, g_cos, 32);
        DataCopy(ub_sin_bf16, g_sin, 32);
        DataCopy(ub_ln_w, g_w, 128);
        DataCopy(ub_ln_b, g_b, 128);
        DataCopy(ub_kw_bf16, g_kw, 160);
        if (!is_c8) {
            __gm__ void* ptr = slot_is_i64 ? (__gm__ void*)(((__gm__ int64_t*)slot_mapping) + t)
                                           : (__gm__ void*)(((__gm__ int32_t*)slot_mapping) + t);
            dcci((__gm__ uint64_t*)ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }

        SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        Cast(ub_cos_f, ub_cos_bf16, AscendC::RoundMode::CAST_NONE, 32);
        Cast(ub_sin_f, ub_sin_bf16, AscendC::RoundMode::CAST_NONE, 32);
        Cast(ub_kw_f, ub_kw_bf16, AscendC::RoundMode::CAST_NONE, 160);
        PipeBarrier<PIPE_V>();

        Add(ub_kw_swap_f, ub_kw_f, ub_kw_f[64], 64);
        Muls(ub_sin2_f, ub_sin_f, -1.0f, 32);
        ShiftRight(ub_cos2_f.ReinterpretCast<uint32_t>(), ub_cos_f.ReinterpretCast<uint32_t>(), (uint32_t)16, 32);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(ub_kw_swap_f, ub_kw_swap_f, ub_work, 64);
        Or(ub_cos_bf16.ReinterpretCast<uint16_t>(),
           ub_cos_f.ReinterpretCast<uint16_t>(),
           ub_cos2_f.ReinterpretCast<uint16_t>(),
           64);
        ShiftRight(ub_cos2_f.ReinterpretCast<uint32_t>(), ub_sin2_f.ReinterpretCast<uint32_t>(), (uint32_t)16, 32);
        PipeBarrier<PIPE_V>();
        Muls(ub_kw_swap_f, ub_kw_swap_f, -1.0f / 128.0f, 8);
        Or(ub_sin_bf16.ReinterpretCast<uint16_t>(),
           ub_sin_f.ReinterpretCast<uint16_t>(),
           ub_cos2_f.ReinterpretCast<uint16_t>(),
           64);
        Cast(ub_cos2_f, ub_cos_bf16, AscendC::RoundMode::CAST_NONE, 64);
        PipeBarrier<PIPE_V>();
        Brcb(ub_work, ub_kw_swap_f, 1, AscendC::BrcbRepeatParams{1, 8});
        Cast(ub_sin2_f, ub_sin_bf16, AscendC::RoundMode::CAST_NONE, 64);
        PipeBarrier<PIPE_V>();

        Add(ub_kw_f, ub_kw_f, ub_work, 64, 2,
            AscendC::BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Mul(ub_kw_swap_f, ub_kw_f, ub_kw_f, 128);
        PipeBarrier<PIPE_V>();
        Add(ub_kw_swap_f, ub_kw_swap_f, ub_kw_swap_f[64], 64);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(ub_kw_swap_f, ub_kw_swap_f, ub_work, 64);
        PipeBarrier<PIPE_V>();
        Muls(ub_kw_swap_f, ub_kw_swap_f, 1.0f / 128.0f, 8);
        PipeBarrier<PIPE_V>();
        Adds(ub_kw_swap_f, ub_kw_swap_f, eps, 8);
        PipeBarrier<PIPE_V>();
        Sqrt(ub_kw_swap_f, ub_kw_swap_f, 8);
        Duplicate<float>(ub_work, 1.0f, 8);
        PipeBarrier<PIPE_V>();
        Div(ub_kw_swap_f, ub_work, ub_kw_swap_f, 8);
        PipeBarrier<PIPE_V>();
        Brcb(ub_work, ub_kw_swap_f, 1, AscendC::BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();

        Mul(ub_kw_f, ub_kw_f, ub_work, 64, 2,
            AscendC::BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Mul(ub_kw_f, ub_kw_f, ub_ln_w, 128);
        PipeBarrier<PIPE_V>();
        Add(ub_kw_f, ub_kw_f, ub_ln_b, 128);
        PipeBarrier<PIPE_V>();

        Cast(ub_k_bf16, ub_kw_f, AscendC::RoundMode::CAST_RINT, 128);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> k_u32 = ub_k_bf16.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> kt1_u32 = ub_kw_f.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> kt2_u32 = ub_kw_swap_f.ReinterpretCast<uint32_t>();
        ShiftLeft(kt1_u32, k_u32, (uint32_t)16, 32);
        ShiftRight(kt2_u32, k_u32, (uint32_t)16, 32);
        PipeBarrier<PIPE_V>();
        Or(ub_kw_swap_bf16.ReinterpretCast<uint16_t>(),
           kt1_u32.ReinterpretCast<uint16_t>(),
           kt2_u32.ReinterpretCast<uint16_t>(),
           64);
        PipeBarrier<PIPE_V>();

        Cast(ub_kw_f, ub_k_bf16, AscendC::RoundMode::CAST_NONE, 64);
        Cast(ub_kw_swap_f, ub_kw_swap_bf16, AscendC::RoundMode::CAST_NONE, 64);
        PipeBarrier<PIPE_V>();

        Mul(ub_kw_f, ub_kw_f, ub_cos2_f[0], 64);
        Mul(ub_kw_swap_f, ub_kw_swap_f, ub_sin2_f[0], 64);
        PipeBarrier<PIPE_V>();
        Add(ub_kw_f, ub_kw_f, ub_kw_swap_f, 64);
        PipeBarrier<PIPE_V>();
        Cast(ub_k_bf16, ub_kw_f, AscendC::RoundMode::CAST_RINT, 64);
        if (is_c8) {
            Cast(ub_w_fp16, ub_kw_f[128], AscendC::RoundMode::CAST_RINT, 32);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);

        if (is_c8) {
            GlobalTensor<bfloat16_t> g_kout;
            g_kout.SetGlobalBuffer(qk_or_q_out + t * 128, 128);
            DataCopy(g_kout, ub_k_bf16, 128);

            GlobalTensor<half> g_wout;
            g_wout.SetGlobalBuffer((__gm__ half*)(weights_out + t * 32), 32);
            DataCopy(g_wout, ub_w_fp16, 32);
        } else {
            const int64_t slot = slot_is_i64 ? ((__gm__ int64_t*)slot_mapping)[t]
                                             : static_cast<int64_t>(((__gm__ int32_t*)slot_mapping)[t]);
            if (slot >= 0 && slot < num_slots) {
                GlobalTensor<bfloat16_t> g_kcache;
                g_kcache.SetGlobalBuffer(k_cache + slot * 128, 128);
                DataCopy(g_kcache, ub_k_bf16, 128);
            }
            GlobalTensor<bfloat16_t> g_wout;
            g_wout.SetGlobalBuffer((__gm__ bfloat16_t*)(weights_out + t * 32), 32);
            DataCopy(g_wout, ub_kw_bf16[128], 32);
        }
        return;
    }

    const int64_t total_q = static_cast<int64_t>(T) * N;
    int64_t begin_q = 0;
    int64_t q_count = 0;
    if (is_c8 && T == 32 && N == 64 && cores == 48) {
        if (core < 32) {
            begin_q = core * 40;
            q_count = 40;
        } else {
            begin_q = 1280 + (core - 32) * 48;
            q_count = 48;
        }
    } else {
        const int64_t q_stripe = (total_q + cores - 1) / cores;
        begin_q = core * q_stripe;
        q_count = (begin_q < total_q)
                      ? ((begin_q + q_stripe <= total_q) ? q_stripe : (total_q - begin_q))
                      : 0;
    }

    GlobalTensor<bfloat16_t> g_cos, g_sin;
    g_cos.SetGlobalBuffer(cos_gm, T * 64);
    g_sin.SetGlobalBuffer(sin_gm, T * 64);
    DataCopy(ub_cos_bf16, g_cos, T * 64);
    DataCopy(ub_sin_bf16, g_sin, T * 64);

    if (core < T) {
        GlobalTensor<float> g_w, g_b;
        g_w.SetGlobalBuffer(ln_weight, 128);
        g_b.SetGlobalBuffer(ln_bias, 128);
        DataCopy(ub_ln_w, g_w, 128);
        DataCopy(ub_ln_b, g_b, 128);
        if (T <= cores) {
            GlobalTensor<bfloat16_t> g_kw_pre;
            g_kw_pre.SetGlobalBuffer(kw + core * 160, 160);
            DataCopy(ub_kw_bf16, g_kw_pre, 160);
        }
        if (!is_c8) {
            for (int64_t t = core; t < T; t += cores) {
                __gm__ void* ptr = slot_is_i64 ? (__gm__ void*)(((__gm__ int64_t*)slot_mapping) + t)
                                               : (__gm__ void*)(((__gm__ int32_t*)slot_mapping) + t);
                dcci((__gm__ uint64_t*)ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
            }
        }
    }

    if (q_count > 0) {
        GlobalTensor<bfloat16_t> g_qin;
        g_qin.SetGlobalBuffer(q_in + begin_q * 128, q_count * 128);
        DataCopy(ub_q_bf16, g_qin, q_count * 128);
    }

    SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    Cast(ub_cos_f, ub_cos_bf16, AscendC::RoundMode::CAST_NONE, T * 64);
    Cast(ub_sin_f, ub_sin_bf16, AscendC::RoundMode::CAST_NONE, T * 64);

    if (q_count > 0) {
        LocalTensor<uint32_t> q_u32 = ub_q_bf16.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> tmp1_u32 = ub_q_f.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> tmp2_u32 = ub_q_swap_f.ReinterpretCast<uint32_t>();
        ShiftLeft(tmp1_u32, q_u32, (uint32_t)16, q_count * 64);
        ShiftRight(tmp2_u32, q_u32, (uint32_t)16, q_count * 64);
        PipeBarrier<PIPE_V>();
        Or(ub_q_swap_bf16.ReinterpretCast<uint16_t>(),
           tmp1_u32.ReinterpretCast<uint16_t>(),
           tmp2_u32.ReinterpretCast<uint16_t>(),
           q_count * 128);
        PipeBarrier<PIPE_V>();
        Cast(ub_q_f, ub_q_bf16, AscendC::RoundMode::CAST_NONE, q_count * 128);
        Cast(ub_q_swap_f, ub_q_swap_bf16, AscendC::RoundMode::CAST_NONE, q_count * 128);
    }

    SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    __ubuf__ float* cos_f_ptr = (__ubuf__ float*)ub_cos_f.GetPhyAddr();
    __ubuf__ float* sin_f_ptr = (__ubuf__ float*)ub_sin_f.GetPhyAddr();
    __ubuf__ float* cos2_ptr = (__ubuf__ float*)ub_cos2_f.GetPhyAddr();
    __ubuf__ float* sin2_ptr = (__ubuf__ float*)ub_sin2_f.GetPhyAddr();

    int64_t t_min = (q_count > 0) ? (begin_q / N) : T;
    int64_t t_max = (q_count > 0) ? ((begin_q + q_count - 1) / N) : -1;
    for (int64_t t = t_min; t <= t_max; ++t) {
        for (int i = 0; i < 32; ++i) {
            const float c = cos_f_ptr[t * 64 + i];
            const float s = sin_f_ptr[t * 64 + i];
            cos2_ptr[t * 64 + 2 * i] = c;
            cos2_ptr[t * 64 + 2 * i + 1] = c;
            sin2_ptr[t * 64 + 2 * i] = -s;
            sin2_ptr[t * 64 + 2 * i + 1] = s;
        }
    }
    for (int64_t t = core; t < T; t += cores) {
        if (t < t_min || t > t_max) {
            for (int i = 0; i < 32; ++i) {
                const float c = cos_f_ptr[t * 64 + i];
                const float s = sin_f_ptr[t * 64 + i];
                cos2_ptr[t * 64 + 2 * i] = c;
                cos2_ptr[t * 64 + 2 * i + 1] = c;
                sin2_ptr[t * 64 + 2 * i] = -s;
                sin2_ptr[t * 64 + 2 * i + 1] = s;
            }
        }
    }
    SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    if (q_count > 0) {
        int64_t r = 0;
        while (r < q_count) {
            const int64_t t = (begin_q + r) / N;
            const int64_t next_t_q = (t + 1) * N - begin_q;
            const int64_t span = (next_t_q < q_count ? next_t_q : q_count) - r;
            Mul(ub_q_f[r * 128], ub_q_f[r * 128], ub_cos2_f[t * 64], 64, span,
                AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
            Mul(ub_q_swap_f[r * 128], ub_q_swap_f[r * 128], ub_sin2_f[t * 64], 64, span,
                AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
            r += span;
        }
        PipeBarrier<PIPE_V>();
        Add(ub_q_f, ub_q_f, ub_q_swap_f, 64, q_count,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
        PipeBarrier<PIPE_V>();
        Cast(ub_q_bf16, ub_q_f, AscendC::RoundMode::CAST_RINT, q_count * 128);
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<bfloat16_t> g_qout;
        __gm__ bfloat16_t* q_dst = is_c8 ? (qk_or_q_out + (T + begin_q) * 128) : (qk_or_q_out + begin_q * 128);
        g_qout.SetGlobalBuffer(q_dst, q_count * 128);
        DataCopy(g_qout, ub_q_bf16, q_count * 128);
    }

    for (int64_t t = core; t < T; t += cores) {
        if (T > cores || t != core) {
            GlobalTensor<bfloat16_t> g_kw;
            g_kw.SetGlobalBuffer(kw + t * 160, 160);
            DataCopy(ub_kw_bf16, g_kw, 160);
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        }

        Cast(ub_kw_f, ub_kw_bf16, AscendC::RoundMode::CAST_NONE, 160);
        PipeBarrier<PIPE_V>();

        Add(ub_kw_swap_f, ub_kw_f, ub_kw_f[64], 64);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(ub_kw_swap_f, ub_kw_swap_f, ub_work, 64);
        SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        const float mean = ub_kw_swap_f.GetValue(0) * (1.0f / 128.0f);
        SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);

        Adds(ub_kw_f, ub_kw_f, -mean, 128);
        PipeBarrier<PIPE_V>();
        Mul(ub_kw_swap_f, ub_kw_f, ub_kw_f, 128);
        PipeBarrier<PIPE_V>();
        Add(ub_kw_swap_f, ub_kw_swap_f, ub_kw_swap_f[64], 64);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(ub_kw_swap_f, ub_kw_swap_f, ub_work, 64);
        PipeBarrier<PIPE_V>();
        Muls(ub_kw_swap_f, ub_kw_swap_f, 1.0f / 128.0f, 8);
        PipeBarrier<PIPE_V>();
        Adds(ub_kw_swap_f, ub_kw_swap_f, eps, 8);
        PipeBarrier<PIPE_V>();
        Sqrt(ub_kw_swap_f, ub_kw_swap_f, 8);
        Duplicate<float>(ub_work, 1.0f, 8);
        PipeBarrier<PIPE_V>();
        Div(ub_kw_swap_f, ub_work, ub_kw_swap_f, 8);
        SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        const float rstd = ub_kw_swap_f.GetValue(0);
        SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);

        Muls(ub_kw_f, ub_kw_f, rstd, 128);
        PipeBarrier<PIPE_V>();
        Mul(ub_kw_f, ub_kw_f, ub_ln_w, 128);
        PipeBarrier<PIPE_V>();
        Add(ub_kw_f, ub_kw_f, ub_ln_b, 128);
        PipeBarrier<PIPE_V>();

        Cast(ub_k_bf16, ub_kw_f, AscendC::RoundMode::CAST_RINT, 128);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> k_u32 = ub_k_bf16.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> kt1_u32 = ub_kw_f.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> kt2_u32 = ub_kw_swap_f.ReinterpretCast<uint32_t>();
        ShiftLeft(kt1_u32, k_u32, (uint32_t)16, 32);
        ShiftRight(kt2_u32, k_u32, (uint32_t)16, 32);
        PipeBarrier<PIPE_V>();
        Or(ub_kw_swap_bf16.ReinterpretCast<uint16_t>(),
           kt1_u32.ReinterpretCast<uint16_t>(),
           kt2_u32.ReinterpretCast<uint16_t>(),
           64);
        PipeBarrier<PIPE_V>();

        Cast(ub_kw_f, ub_k_bf16, AscendC::RoundMode::CAST_NONE, 64);
        Cast(ub_kw_swap_f, ub_kw_swap_bf16, AscendC::RoundMode::CAST_NONE, 64);
        PipeBarrier<PIPE_V>();

        Mul(ub_kw_f, ub_kw_f, ub_cos2_f[t * 64], 64);
        Mul(ub_kw_swap_f, ub_kw_swap_f, ub_sin2_f[t * 64], 64);
        PipeBarrier<PIPE_V>();
        Add(ub_kw_f, ub_kw_f, ub_kw_swap_f, 64);
        PipeBarrier<PIPE_V>();
        Cast(ub_k_bf16, ub_kw_f, AscendC::RoundMode::CAST_RINT, 64);
        if (is_c8) {
            Cast(ub_w_fp16, ub_kw_f[128], AscendC::RoundMode::CAST_RINT, 32);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);

        if (is_c8) {
            GlobalTensor<bfloat16_t> g_kout;
            g_kout.SetGlobalBuffer(qk_or_q_out + t * 128, 128);
            DataCopy(g_kout, ub_k_bf16, 128);

            GlobalTensor<half> g_wout;
            g_wout.SetGlobalBuffer((__gm__ half*)(weights_out + t * 32), 32);
            DataCopy(g_wout, ub_w_fp16, 32);
        } else {
            const int64_t slot = slot_is_i64 ? ((__gm__ int64_t*)slot_mapping)[t]
                                             : static_cast<int64_t>(((__gm__ int32_t*)slot_mapping)[t]);
            if (slot >= 0 && slot < num_slots) {
                GlobalTensor<bfloat16_t> g_kcache;
                g_kcache.SetGlobalBuffer(k_cache + slot * 128, 128);
                DataCopy(g_kcache, ub_k_bf16, 128);
            }
            GlobalTensor<bfloat16_t> g_wout;
            g_wout.SetGlobalBuffer((__gm__ bfloat16_t*)(weights_out + t * 32), 32);
            DataCopy(g_wout, ub_kw_bf16[128], 32);
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

void launch_indexer_prep_qk(uint32_t block_dim, void* stream, void* kw, void* q_in, void* cos_gm, void* sin_gm,
                            void* ln_weight, void* ln_bias, void* slot_mapping, void* k_cache,
                            void* qk_or_q_out, void* weights_out,
                            int T, int N, int64_t num_slots, float eps, int is_c8, int slot_is_i64,
                            const int32_t* active_ptr)
{
    indexer_prep_qk_kernel<<<block_dim, nullptr, stream>>>(
        (__gm__ bfloat16_t*)kw, (__gm__ bfloat16_t*)q_in, (__gm__ bfloat16_t*)cos_gm, (__gm__ bfloat16_t*)sin_gm,
        (__gm__ float*)ln_weight, (__gm__ float*)ln_bias, (__gm__ void*)slot_mapping, (__gm__ bfloat16_t*)k_cache,
        (__gm__ bfloat16_t*)qk_or_q_out, (__gm__ uint16_t*)weights_out,
        T, N, num_slots, eps, is_c8, slot_is_i64, (const __gm__ int32_t*)active_ptr);
}

__aicore__ inline void fwht_h32_16_8(
    int64_t n64,
    const LocalTensor<float>& dst,
    const LocalTensor<float>& src,
    const LocalTensor<float>& tmp)
{
    Add(tmp[0], src[0], src[32], 32, n64, AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 8});
    Sub(tmp[32], src[0], src[32], 32, n64, AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    for (int64_t rem = 2 * n64, off = 0; rem > 0; rem -= 128, off += 4096) {
        const int64_t rep = (rem <= 128) ? rem : 128;
        Add(src[off], tmp[off], tmp[off + 16], 16, rep, AscendC::BinaryRepeatParams{1, 1, 1, 4, 4, 4});
        Sub(src[off + 16], tmp[off], tmp[off + 16], 16, rep, AscendC::BinaryRepeatParams{1, 1, 1, 4, 4, 4});
    }
    PipeBarrier<PIPE_V>();
    for (int64_t rem = 4 * n64, off = 0; rem > 0; rem -= 128, off += 2048) {
        const int64_t rep = (rem <= 128) ? rem : 128;
        Add(dst[off], src[off], src[off + 8], 8, rep, AscendC::BinaryRepeatParams{1, 1, 1, 2, 2, 2});
        Sub(dst[off + 8], src[off], src[off + 8], 8, rep, AscendC::BinaryRepeatParams{1, 1, 1, 2, 2, 2});
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void init_fwht_trans_idx(const LocalTensor<uint32_t>& ub_trans_idx, int64_t max_n64)
{
    __ubuf__ uint32_t* ti = (__ubuf__ uint32_t*)ub_trans_idx.GetPhyAddr();
    for (uint32_t i = 0; i < 64; ++i) {
        ti[i] = (((i & 7U) << 3) | (i >> 3)) << 2;
    }
    SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
    WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
    LocalTensor<int32_t> idx_i32 = ub_trans_idx.ReinterpretCast<int32_t>();
    for (int64_t step = 1; step < max_n64; step <<= 1) {
        Adds(idx_i32[step * 64], idx_i32[0], static_cast<int32_t>(step * 256), step * 64);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void quantize_rows_c8(
    int64_t rows,
    const LocalTensor<bfloat16_t>& ub_in_bf16,
    const LocalTensor<float>& ub_cast_f,
    const LocalTensor<float>& ub_abs_f,
    const LocalTensor<int8_t>& ub_out_i8,
    const LocalTensor<float>& ub_amax_f,
    const LocalTensor<float>& ub_qscale_f,
    const LocalTensor<float>& ub_brcb_f,
    const LocalTensor<float>& ub_scale_f,
    const LocalTensor<half>& ub_scale_h,
    const LocalTensor<float>& ub_const_127,
    const LocalTensor<float>& ub_const_inv127,
    const LocalTensor<uint32_t>& ub_trans_idx,
    bool do_fwht = false)
{
    Cast(ub_cast_f, ub_in_bf16, AscendC::RoundMode::CAST_NONE, rows * 128);
    PipeBarrier<PIPE_V>();
    if (do_fwht) {
        const int64_t n64 = 2 * rows;
        Muls(ub_cast_f, ub_cast_f, 0.08837890625f, rows * 128);
        PipeBarrier<PIPE_V>();
        for (int64_t rem = 4 * n64, off = 0; rem > 0; rem -= 128, off += 2048) {
            const int64_t rep = (rem <= 128) ? rem : 128;
            Add(ub_abs_f[off], ub_cast_f[off], ub_cast_f[off + 8], 8, rep, AscendC::BinaryRepeatParams{1, 1, 1, 2, 2, 2});
            Sub(ub_abs_f[off + 8], ub_cast_f[off], ub_cast_f[off + 8], 8, rep, AscendC::BinaryRepeatParams{1, 1, 1, 2, 2, 2});
        }
        PipeBarrier<PIPE_V>();
        Gather(ub_cast_f, ub_abs_f, ub_trans_idx, (uint32_t)0, 64, (uint8_t)n64, (uint16_t)8);
        PipeBarrier<PIPE_V>();
        fwht_h32_16_8(n64, ub_abs_f, ub_cast_f, ub_abs_f);
        Gather(ub_cast_f, ub_abs_f, ub_trans_idx, (uint32_t)0, 64, (uint8_t)n64, (uint16_t)8);
        PipeBarrier<PIPE_V>();
        for (int c = 0; c < 8; ++c) {
            if ((c & 1) == 0) {
                Add(ub_abs_f[c * 16], ub_cast_f[0], ub_cast_f[16], 16, rows, AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
            } else {
                Sub(ub_abs_f[c * 16], ub_cast_f[0], ub_cast_f[16], 16, rows, AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
            }
        }
        PipeBarrier<PIPE_V>();
        for (int b = 2; b < 8; ++b) {
            for (int c = 0; c < 8; ++c) {
                int p = b & c;
                p ^= (p >> 2);
                p ^= (p >> 1);
                if ((p & 1) == 0) {
                    Add(ub_abs_f[c * 16], ub_abs_f[c * 16], ub_cast_f[b * 16], 16, rows, AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
                } else {
                    Sub(ub_abs_f[c * 16], ub_abs_f[c * 16], ub_cast_f[b * 16], 16, rows, AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
                }
            }
            PipeBarrier<PIPE_V>();
        }
        Cast(ub_in_bf16, ub_abs_f, AscendC::RoundMode::CAST_RINT, rows * 128);
        PipeBarrier<PIPE_V>();
        Cast(ub_cast_f, ub_in_bf16, AscendC::RoundMode::CAST_NONE, rows * 128);
        PipeBarrier<PIPE_V>();
    }
    Abs(ub_abs_f, ub_cast_f, rows * 128);
    PipeBarrier<PIPE_V>();
    Max(ub_abs_f, ub_abs_f[64], ub_abs_f, 64, rows, AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
    Duplicate<float>(ub_amax_f, 0.0f, 64);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(ub_amax_f, ub_abs_f, 64, rows, 1, 1, 16, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    Div(ub_qscale_f, ub_const_127, ub_amax_f, 64);
    Mul(ub_scale_f, ub_amax_f, ub_const_inv127, 64);
    PipeBarrier<PIPE_V>();
    Brcb(ub_brcb_f, ub_qscale_f, 8, AscendC::BrcbRepeatParams{1, 8});
    Cast(ub_scale_h, ub_scale_f, AscendC::RoundMode::CAST_RINT, 64);
    PipeBarrier<PIPE_V>();
    Mul(ub_abs_f, ub_cast_f, ub_brcb_f, 64, rows, AscendC::BinaryRepeatParams{1, 1, 0, 16, 16, 1});
    Mul(ub_abs_f[64], ub_cast_f[64], ub_brcb_f, 64, rows, AscendC::BinaryRepeatParams{1, 1, 0, 16, 16, 1});
    PipeBarrier<PIPE_V>();
    LocalTensor<int16_t> ub_i16 = ub_abs_f.ReinterpretCast<int16_t>();
    LocalTensor<half> ub_h16 = ub_abs_f.ReinterpretCast<half>();
    Cast(ub_i16, ub_abs_f, AscendC::RoundMode::CAST_RINT, rows * 128);
    PipeBarrier<PIPE_V>();
    Cast(ub_h16, ub_i16, AscendC::RoundMode::CAST_ROUND, rows * 128);
    PipeBarrier<PIPE_V>();
    Cast(ub_out_i8, ub_h16, AscendC::RoundMode::CAST_TRUNC, rows * 128);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void indexer_split_cast_scatter_c8_kernel(
    __gm__ int8_t* qk_int8, __gm__ float* qk_scale, __gm__ void* slot_mapping,
    __gm__ int8_t* k_cache, __gm__ uint16_t* k_scale_cache,
    __gm__ int8_t* q_int8_out, __gm__ half* q_scale_out,
    int T, int N, int64_t num_slots, int slot_is_i64,
    const __gm__ int32_t* active_ptr)
{
    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    if (T <= 0 || T > 64) {
        return;
    }

    int active_T = T;
    if (active_ptr != nullptr && T == 32) {
        dcci((__gm__ uint64_t*)active_ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
        const int a = *active_ptr;
        if (a >= 1 && a <= T) {
            active_T = a;
        }
    }
    const int64_t total_q = static_cast<int64_t>(T) * N;
    const int64_t active_q = static_cast<int64_t>(active_T) * N;
    const int64_t q_cores = (cores > 1) ? (cores - 1) : 1;
    int64_t fast_begin_q = 0;
    int64_t fast_q_count = 0;
    const bool is_bf16_in = (qk_scale == nullptr || (uintptr_t)qk_scale == 1ULL);
    const bool do_fwht = ((uintptr_t)qk_scale == 1ULL);
    if (is_bf16_in && core < q_cores) {
        const int64_t total_q_blocks16 = (active_q + 15) / 16;
        const int64_t q_blocks16_per_core = (total_q_blocks16 + q_cores - 1) / q_cores;
        fast_begin_q = core * q_blocks16_per_core * 16;
        fast_q_count = (fast_begin_q < active_q)
                           ? ((fast_begin_q + q_blocks16_per_core * 16 <= active_q)
                                  ? (q_blocks16_per_core * 16)
                                  : (active_q - fast_begin_q))
                           : 0;
        if (fast_q_count <= 0) {
            return;
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> b_all;
    pipe.InitBuffer(b_all, 128 * 1024);
    LocalTensor<float> f_all = b_all.Get<float>();
    LocalTensor<int8_t> i8_all = f_all.ReinterpretCast<int8_t>();
    LocalTensor<half> h_all = f_all.ReinterpretCast<half>();
    LocalTensor<bfloat16_t> bf_all = f_all.ReinterpretCast<bfloat16_t>();

    if (is_bf16_in) {
        LocalTensor<bfloat16_t> ub_in_bf16 = bf_all[0];
        LocalTensor<float> ub_cast_f = f_all[4096];
        LocalTensor<float> ub_abs_f = f_all[12288];
        LocalTensor<int8_t> ub_out_i8 = i8_all[20480 * 4];
        LocalTensor<float> ub_amax_f = f_all[22528];
        LocalTensor<float> ub_qscale_f = f_all[22592];
        LocalTensor<float> ub_brcb_f = f_all[22656];
        LocalTensor<float> ub_scale_f = f_all[23168];
        LocalTensor<half> ub_scale_h = h_all[23232 * 2];
        LocalTensor<float> ub_const_127 = f_all[23264];
        LocalTensor<float> ub_const_inv127 = f_all[23328];
        LocalTensor<uint32_t> ub_trans_idx = f_all[23424].ReinterpretCast<uint32_t>();

        if (core < q_cores) {
            const int64_t begin_q = fast_begin_q;
            const int64_t q_count = fast_q_count;
            Duplicate<float>(ub_const_127, 127.0f, 64);
            Duplicate<float>(ub_const_inv127, 1.0f / 127.0f, 64);
            PipeBarrier<PIPE_V>();
            if (do_fwht) {
                init_fwht_trans_idx(ub_trans_idx, (q_count <= 32) ? (2 * q_count) : 64);
            }

            for (int64_t q_off = 0; q_off < q_count; q_off += 32) {
                const int64_t rows = (q_off + 32 <= q_count) ? 32 : (q_count - q_off);
                const int64_t cur_q = begin_q + q_off;
                GlobalTensor<bfloat16_t> g_qin;
                g_qin.SetGlobalBuffer(((__gm__ bfloat16_t*)qk_int8) + (T + cur_q) * 128, rows * 128);
                DataCopy(ub_in_bf16, g_qin, rows * 128);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                quantize_rows_c8(
                    rows, ub_in_bf16, ub_cast_f, ub_abs_f, ub_out_i8,
                    ub_amax_f, ub_qscale_f, ub_brcb_f, ub_scale_f,
                    ub_scale_h, ub_const_127, ub_const_inv127, ub_trans_idx, do_fwht);

                SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                GlobalTensor<int8_t> g_qout;
                g_qout.SetGlobalBuffer(q_int8_out + cur_q * 128, rows * 128);
                DataCopy(g_qout, ub_out_i8, rows * 128);
                GlobalTensor<half> g_qsout;
                g_qsout.SetGlobalBuffer(q_scale_out + cur_q, AlignUp(rows, 16));
                DataCopy(g_qsout, ub_scale_h, AlignUp(rows, 16));
                if (q_off + 32 < q_count) {
                    SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
            }
        }

        if (core == cores - 1) {
            GlobalTensor<bfloat16_t> g_kin;
            g_kin.SetGlobalBuffer((__gm__ bfloat16_t*)qk_int8, active_T * 128);
            DataCopy(ub_in_bf16, g_kin, active_T * 128);
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

            Duplicate<float>(ub_const_127, 127.0f, 64);
            Duplicate<float>(ub_const_inv127, 1.0f / 127.0f, 64);
            PipeBarrier<PIPE_V>();
            if (do_fwht) {
                init_fwht_trans_idx(ub_trans_idx, 2 * active_T);
            }

            const int64_t slot_bytes = static_cast<int64_t>(T) * (slot_is_i64 ? 8 : 4);
            for (int64_t off = 0; off < slot_bytes + 64; off += 64) {
                dcci((__gm__ uint64_t*)(((__gm__ uint8_t*)slot_mapping) + off), SINGLE_CACHE_LINE, CACHELINE_OUT);
            }

            int64_t slots[64];
            bool contig = (active_T > 0);
            bool contig_s = (T > 0);
            for (int t = 0; t < T; ++t) {
                const int64_t slot = slot_is_i64 ? ((__gm__ int64_t*)slot_mapping)[t]
                                                 : static_cast<int64_t>(((__gm__ int32_t*)slot_mapping)[t]);
                slots[t] = slot;
                if (t > 0 && slot != slots[0] + t) {
                    contig_s = false;
                    if (t < active_T) {
                        contig = false;
                    }
                }
            }
            if (contig && (slots[0] < 0 || slots[0] + active_T > num_slots)) {
                contig = false;
            }
            if (contig_s && (slots[0] < 0 || slots[0] + T > num_slots)) {
                contig_s = false;
            }

            if (!contig_s) {
                uint64_t prev_cl = ~0ULL;
                for (int t = 0; t < T; ++t) {
                    const int64_t slot = slots[t];
                    if (slot >= 0 && slot < num_slots) {
                        uint64_t cl = ((uint64_t)(k_scale_cache + slot)) >> 6;
                        if (cl != prev_cl) {
                            dcci((__gm__ uint64_t*)(k_scale_cache + slot), SINGLE_CACHE_LINE, CACHELINE_OUT);
                            prev_cl = cl;
                        }
                    }
                }
            }

            if (contig_s && active_T < T) {
                Duplicate(ub_scale_h.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), 64);
                PipeBarrier<PIPE_V>();
            }
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

            quantize_rows_c8(
                active_T, ub_in_bf16, ub_cast_f, ub_abs_f, ub_out_i8,
                ub_amax_f, ub_qscale_f, ub_brcb_f, ub_scale_f,
                ub_scale_h, ub_const_127, ub_const_inv127, ub_trans_idx, do_fwht);

            SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            if (!contig_s) {
                SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            }

            WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            if (contig) {
                GlobalTensor<int8_t> g_kcache;
                g_kcache.SetGlobalBuffer(k_cache + slots[0] * 128, active_T * 128);
                DataCopy(g_kcache, ub_out_i8, active_T * 128);
            } else {
                for (int t = 0; t < active_T; ++t) {
                    const int64_t slot = slots[t];
                    if (slot >= 0 && slot < num_slots) {
                        GlobalTensor<int8_t> g_kcache;
                        g_kcache.SetGlobalBuffer(k_cache + slot * 128, 128);
                        DataCopy(g_kcache, ub_out_i8[t * 128], 128);
                    }
                }
            }

            if (contig_s) {
                GlobalTensor<uint16_t> g_kscale;
                g_kscale.SetGlobalBuffer(k_scale_cache + slots[0], T);
                AscendC::DataCopyExtParams pad_params{1, static_cast<uint32_t>(T * 2), 0, 0, 0};
                DataCopyPad(g_kscale, ub_scale_h.ReinterpretCast<uint16_t>(), pad_params);
            } else {
                WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                __ubuf__ uint16_t* sh = (__ubuf__ uint16_t*)ub_scale_h.GetPhyAddr();
                for (int t = 0; t < active_T; ++t) {
                    const int64_t slot = slots[t];
                    if (slot >= 0 && slot < num_slots) {
                        k_scale_cache[slot] = sh[t];
                    }
                }
                for (int t = active_T; t < T; ++t) {
                    const int64_t slot = slots[t];
                    if (slot >= 0 && slot < num_slots) {
                        k_scale_cache[slot] = 0;
                    }
                }
                uint64_t prev_cl = ~0ULL;
                for (int t = 0; t < T; ++t) {
                    const int64_t slot = slots[t];
                    if (slot >= 0 && slot < num_slots) {
                        uint64_t cl = ((uint64_t)(k_scale_cache + slot)) >> 6;
                        if (cl != prev_cl) {
                            dcci((__gm__ uint64_t*)(k_scale_cache + slot), SINGLE_CACHE_LINE, CACHELINE_OUT);
                            prev_cl = cl;
                        }
                    }
                }
            }
        }
        return;
    }

    LocalTensor<int8_t> ub_q_i8 = i8_all[0];
    LocalTensor<int8_t> ub_k_i8 = i8_all[6144];
    LocalTensor<float> ub_scale_f = f_all[1600];
    LocalTensor<half> ub_scale_h = h_all[7600];
    LocalTensor<half> ub_scale_aligned = h_all[9776];

    const int64_t q_stripe = (total_q + cores - 1) / cores;
    const int64_t begin_q = core * q_stripe;
    const int64_t q_count = (begin_q < total_q)
                                ? ((begin_q + q_stripe <= total_q) ? q_stripe : (total_q - begin_q))
                                : 0;

    if (q_count > 0) {
        GlobalTensor<int8_t> g_qin, g_qout;
        g_qin.SetGlobalBuffer(qk_int8 + (T + begin_q) * 128, q_count * 128);
        g_qout.SetGlobalBuffer(q_int8_out + begin_q * 128, q_count * 128);
        DataCopy(ub_q_i8, g_qin, q_count * 128);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(g_qout, ub_q_i8, q_count * 128);
    }

    for (int64_t t = core; t < T; t += cores) {
        __gm__ void* ptr = slot_is_i64 ? (__gm__ void*)(((__gm__ int64_t*)slot_mapping) + t)
                                       : (__gm__ void*)(((__gm__ int32_t*)slot_mapping) + t);
        dcci((__gm__ uint64_t*)ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }

    for (int64_t t = core; t < T; t += cores) {
        const int64_t slot = slot_is_i64 ? ((__gm__ int64_t*)slot_mapping)[t]
                                         : static_cast<int64_t>(((__gm__ int32_t*)slot_mapping)[t]);
        if (slot >= 0 && slot < num_slots) {
            GlobalTensor<int8_t> g_kin, g_kcache;
            g_kin.SetGlobalBuffer(qk_int8 + t * 128, 128);
            g_kcache.SetGlobalBuffer(k_cache + slot * 128, 128);
            DataCopy(ub_k_i8, g_kin, 128);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            DataCopy(g_kcache, ub_k_i8, 128);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    if (core == 0) {
        for (int t = 0; t < T; ++t) {
            __gm__ void* ptr = slot_is_i64 ? (__gm__ void*)(((__gm__ int64_t*)slot_mapping) + t)
                                           : (__gm__ void*)(((__gm__ int32_t*)slot_mapping) + t);
            dcci((__gm__ uint64_t*)ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        const int64_t total_scales = static_cast<int64_t>(T) * (1 + N);
        const int64_t copy_scales = AlignUp(total_scales, 16);
        GlobalTensor<float> g_sin;
        g_sin.SetGlobalBuffer(qk_scale, copy_scales);
        DataCopy(ub_scale_f, g_sin, copy_scales);
        SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);

        Cast(ub_scale_h, ub_scale_f, AscendC::RoundMode::CAST_RINT, copy_scales);
        if (T % 16 == 0) {
            SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            GlobalTensor<half> g_qsout;
            g_qsout.SetGlobalBuffer(q_scale_out, total_q);
            DataCopy(g_qsout, ub_scale_h[T], AlignUp(total_q, 16));
            SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        } else {
            SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            __ubuf__ uint16_t* sh = (__ubuf__ uint16_t*)ub_scale_h.GetPhyAddr();
            __ubuf__ uint16_t* sa = (__ubuf__ uint16_t*)ub_scale_aligned.GetPhyAddr();
            for (int64_t i = 0; i < total_q; ++i) {
                sa[i] = sh[T + i];
            }
            SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            GlobalTensor<half> g_qsout;
            g_qsout.SetGlobalBuffer(q_scale_out, total_q);
            DataCopy(g_qsout, ub_scale_aligned, AlignUp(total_q, 16));
        }

        __ubuf__ uint16_t* sh = (__ubuf__ uint16_t*)ub_scale_h.GetPhyAddr();
        for (int t = 0; t < T; ++t) {
            const int64_t slot = slot_is_i64 ? ((__gm__ int64_t*)slot_mapping)[t]
                                             : static_cast<int64_t>(((__gm__ int32_t*)slot_mapping)[t]);
            if (slot >= 0 && slot < num_slots) {
                dcci((__gm__ uint64_t*)(k_scale_cache + slot), SINGLE_CACHE_LINE, CACHELINE_OUT);
            }
        }
        for (int t = 0; t < T; ++t) {
            const int64_t slot = slot_is_i64 ? ((__gm__ int64_t*)slot_mapping)[t]
                                             : static_cast<int64_t>(((__gm__ int32_t*)slot_mapping)[t]);
            if (slot >= 0 && slot < num_slots) {
                k_scale_cache[slot] = sh[t];
            }
        }
        for (int t = 0; t < T; ++t) {
            const int64_t slot = slot_is_i64 ? ((__gm__ int64_t*)slot_mapping)[t]
                                             : static_cast<int64_t>(((__gm__ int32_t*)slot_mapping)[t]);
            if (slot >= 0 && slot < num_slots) {
                dcci((__gm__ uint64_t*)(k_scale_cache + slot), SINGLE_CACHE_LINE, CACHELINE_OUT);
            }
        }
    }
}

void launch_indexer_split_cast_scatter_c8(uint32_t block_dim, void* stream, void* qk_int8, void* qk_scale,
                                          void* slot_mapping, void* k_cache, void* k_scale_cache,
                                          void* q_int8_out, void* q_scale_out,
                                          int T, int N, int64_t num_slots, int slot_is_i64,
                                          const int32_t* active_ptr)
{
    indexer_split_cast_scatter_c8_kernel<<<block_dim, nullptr, stream>>>(
        (__gm__ int8_t*)qk_int8, (__gm__ float*)qk_scale, (__gm__ void*)slot_mapping,
        (__gm__ int8_t*)k_cache, (__gm__ uint16_t*)k_scale_cache,
        (__gm__ int8_t*)q_int8_out, (__gm__ half*)q_scale_out,
        T, N, num_slots, slot_is_i64, (const __gm__ int32_t*)active_ptr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void indexer_prep_qk_c8_kernel(
    __gm__ bfloat16_t* kw, __gm__ bfloat16_t* q_in, __gm__ bfloat16_t* cos_gm, __gm__ bfloat16_t* sin_gm,
    __gm__ float* ln_weight, __gm__ float* ln_bias, __gm__ void* slot_mapping,
    __gm__ int8_t* k_cache, __gm__ uint16_t* k_scale_cache,
    __gm__ int8_t* q_int8_out, __gm__ half* q_scale_out, __gm__ half* weights_out,
    int T, int N, int64_t num_slots, float eps, int slot_is_i64,
    const __gm__ int32_t* active_ptr)
{
    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    if (T != 32 || (N != 32 && N != 64) || cores != 48) {
        return;
    }

    int active_T = 32;
    if (active_ptr != nullptr) {
        dcci((__gm__ uint64_t*)active_ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
        const int a = *active_ptr;
        if (a >= 1 && a <= 32) {
            active_T = a;
        }
    }

    int64_t q_t = 0;
    int64_t begin_q = 0;
    int64_t q_count = 0;
    if (core < 46) {
        const int64_t one_core_tokens = (2 * active_T > 46) ? (2 * active_T - 46) : 0;
        if (core < one_core_tokens) {
            q_t = core;
            begin_q = core * N;
            q_count = N;
        } else {
            const int64_t rem = core - one_core_tokens;
            q_t = one_core_tokens + (rem >> 1);
            if (q_t >= active_T) {
                return;
            }
            q_count = N >> 1;
            begin_q = one_core_tokens * N + rem * q_count;
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> b_all;
    pipe.InitBuffer(b_all, 188 * 1024);
    LocalTensor<float> f_all = b_all.Get<float>();
    LocalTensor<int8_t> i8_all = f_all.ReinterpretCast<int8_t>();
    LocalTensor<half> h_all = f_all.ReinterpretCast<half>();
    LocalTensor<bfloat16_t> bf_all = f_all.ReinterpretCast<bfloat16_t>();

    LocalTensor<bfloat16_t> ub_in_bf16 = bf_all[0];
    LocalTensor<bfloat16_t> ub_kw_swap_bf16 = bf_all[8192];
    LocalTensor<float> ub_cast_f = f_all[8192];
    LocalTensor<float> ub_abs_f = f_all[16384];
    LocalTensor<uint32_t> ub_trans_idx = f_all[24576].ReinterpretCast<uint32_t>();
    LocalTensor<int8_t> ub_out_i8 = i8_all[32768 * 4];
    LocalTensor<float> ub_amax_f = f_all[34816];
    LocalTensor<float> ub_qscale_f = f_all[34880];
    LocalTensor<float> ub_brcb_f = f_all[34944];
    LocalTensor<float> ub_scale_f = f_all[35456];
    LocalTensor<half> ub_scale_h = h_all[35520 * 2];
    LocalTensor<float> ub_const_127 = f_all[35552];
    LocalTensor<float> ub_const_inv127 = f_all[35616];
    LocalTensor<float> ub_ln_w = f_all[35680];
    LocalTensor<float> ub_ln_b = f_all[35808];
    LocalTensor<half> ub_w_fp16 = h_all[35936 * 2];
    LocalTensor<float> ub_cos_f = f_all[36448];
    LocalTensor<float> ub_sin_f = f_all[38496];
    LocalTensor<float> ub_cos2_f = f_all[40544];
    LocalTensor<float> ub_sin2_f = f_all[42592];
    LocalTensor<bfloat16_t> ub_cos_bf16 = bf_all[44640 * 2];
    LocalTensor<bfloat16_t> ub_sin_bf16 = bf_all[45664 * 2];

    if (core < 46) {
        GlobalTensor<bfloat16_t> g_cos, g_sin, g_qin;
        g_cos.SetGlobalBuffer(cos_gm + q_t * 64, 32);
        g_sin.SetGlobalBuffer(sin_gm + q_t * 64, 32);
        g_qin.SetGlobalBuffer(q_in + begin_q * 128, q_count * 128);
        DataCopy(ub_cos_bf16, g_cos, 32);
        DataCopy(ub_sin_bf16, g_sin, 32);
        DataCopy(ub_in_bf16, g_qin, q_count * 128);
        SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        Duplicate<float>(ub_const_127, 127.0f, 64);
        Duplicate<float>(ub_const_inv127, 1.0f / 127.0f, 64);
        PipeBarrier<PIPE_V>();
        init_fwht_trans_idx(ub_trans_idx, 2 * q_count);

        WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        Cast(ub_cos_f, ub_cos_bf16, AscendC::RoundMode::CAST_NONE, 32);
        Cast(ub_sin_f, ub_sin_bf16, AscendC::RoundMode::CAST_NONE, 32);
        LocalTensor<uint32_t> q_u32 = ub_in_bf16.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> tmp1_u32 = ub_cast_f.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> tmp2_u32 = ub_abs_f.ReinterpretCast<uint32_t>();
        ShiftLeft(tmp1_u32, q_u32, (uint32_t)16, q_count * 64);
        ShiftRight(tmp2_u32, q_u32, (uint32_t)16, q_count * 64);
        PipeBarrier<PIPE_V>();

        Muls(ub_sin2_f, ub_sin_f, -1.0f, 32);
        ShiftRight(ub_cos2_f.ReinterpretCast<uint32_t>(), ub_cos_f.ReinterpretCast<uint32_t>(), (uint32_t)16, 32);
        Or(ub_kw_swap_bf16.ReinterpretCast<uint16_t>(),
           tmp1_u32.ReinterpretCast<uint16_t>(),
           tmp2_u32.ReinterpretCast<uint16_t>(),
           q_count * 128);
        PipeBarrier<PIPE_V>();

        Or(ub_cos_bf16.ReinterpretCast<uint16_t>(),
           ub_cos_f.ReinterpretCast<uint16_t>(),
           ub_cos2_f.ReinterpretCast<uint16_t>(),
           64);
        ShiftRight(ub_cos2_f.ReinterpretCast<uint32_t>(), ub_sin2_f.ReinterpretCast<uint32_t>(), (uint32_t)16, 32);
        PipeBarrier<PIPE_V>();

        Or(ub_sin_bf16.ReinterpretCast<uint16_t>(),
           ub_sin_f.ReinterpretCast<uint16_t>(),
           ub_cos2_f.ReinterpretCast<uint16_t>(),
           64);
        Cast(ub_cos2_f, ub_cos_bf16, AscendC::RoundMode::CAST_NONE, 64);
        Cast(ub_cast_f, ub_in_bf16, AscendC::RoundMode::CAST_NONE, q_count * 128);
        Cast(ub_abs_f, ub_kw_swap_bf16, AscendC::RoundMode::CAST_NONE, q_count * 128);
        PipeBarrier<PIPE_V>();
        Cast(ub_sin2_f, ub_sin_bf16, AscendC::RoundMode::CAST_NONE, 64);
        PipeBarrier<PIPE_V>();

        Mul(ub_cast_f, ub_cast_f, ub_cos2_f, 64, q_count,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
        Mul(ub_abs_f, ub_abs_f, ub_sin2_f, 64, q_count,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
        PipeBarrier<PIPE_V>();
        Add(ub_cast_f, ub_cast_f, ub_abs_f, 64, q_count,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
        PipeBarrier<PIPE_V>();
        Cast(ub_in_bf16, ub_cast_f, AscendC::RoundMode::CAST_RINT, q_count * 128);
        PipeBarrier<PIPE_V>();

        quantize_rows_c8(
            q_count, ub_in_bf16, ub_cast_f, ub_abs_f, ub_out_i8,
            ub_amax_f, ub_qscale_f, ub_brcb_f, ub_scale_f,
            ub_scale_h, ub_const_127, ub_const_inv127, ub_trans_idx, true);

        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<int8_t> g_qout;
        g_qout.SetGlobalBuffer(q_int8_out + begin_q * 128, q_count * 128);
        DataCopy(g_qout, ub_out_i8, q_count * 128);
        GlobalTensor<half> g_qsout;
        g_qsout.SetGlobalBuffer(q_scale_out + begin_q, q_count);
        DataCopy(g_qsout, ub_scale_h, q_count);
        return;
    }

    const int64_t k_t0 = (core == 46) ? 0 : 16;
    const int64_t k_rows = (T > k_t0) ? ((T - k_t0 < 16) ? (T - k_t0) : 16) : 0;
    if (k_rows <= 0) {
        return;
    }
    const int64_t k_act = (active_T > k_t0) ? ((active_T - k_t0 < 16) ? (active_T - k_t0) : 16) : 0;

    GlobalTensor<bfloat16_t> g_cos, g_sin, g_kw;
    GlobalTensor<float> g_w, g_b;
    g_cos.SetGlobalBuffer(cos_gm + k_t0 * 64, k_rows * 64);
    g_sin.SetGlobalBuffer(sin_gm + k_t0 * 64, k_rows * 64);
    g_w.SetGlobalBuffer(ln_weight, 128);
    g_b.SetGlobalBuffer(ln_bias, 128);
    g_kw.SetGlobalBuffer(kw + k_t0 * 160, k_rows * 160);
    DataCopy(ub_cos_bf16, g_cos, k_rows * 64);
    DataCopy(ub_sin_bf16, g_sin, k_rows * 64);
    DataCopy(ub_ln_w, g_w, 128);
    DataCopy(ub_ln_b, g_b, 128);
    DataCopy(ub_in_bf16, g_kw, k_rows * 160);
    SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

    Duplicate<float>(ub_const_127, 127.0f, 64);
    Duplicate<float>(ub_const_inv127, 1.0f / 127.0f, 64);
    Duplicate<float>(ub_scale_f, 0.0f, 64);
    PipeBarrier<PIPE_V>();
    if (k_act > 0) {
        init_fwht_trans_idx(ub_trans_idx, 2 * k_act);
    }

    WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

    Cast(ub_cos_f, ub_cos_bf16, AscendC::RoundMode::CAST_NONE, k_rows * 64);
    Cast(ub_sin_f, ub_sin_bf16, AscendC::RoundMode::CAST_NONE, k_rows * 64);
    Cast(ub_abs_f, ub_in_bf16, AscendC::RoundMode::CAST_NONE, k_rows * 160);
    PipeBarrier<PIPE_V>();

    Cast(ub_w_fp16, ub_abs_f[128], AscendC::RoundMode::CAST_RINT, 32, k_rows,
         AscendC::UnaryRepeatParams{1, 1, 2, 20});
    SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<half> g_wout;
    g_wout.SetGlobalBuffer(weights_out + k_t0 * 32, k_rows * 32);
    DataCopy(g_wout, ub_w_fp16, k_rows * 32);

    if (k_act > 0) {
        Muls(ub_sin2_f, ub_sin_f, -1.0f, k_act * 64);
        ShiftRight(ub_cos2_f.ReinterpretCast<uint32_t>(), ub_cos_f.ReinterpretCast<uint32_t>(), (uint32_t)16, k_act * 64);
        Add(ub_cast_f, ub_abs_f[0], ub_abs_f[64], 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 8, 20, 20});
        PipeBarrier<PIPE_V>();

        WholeReduceSum<float>(ub_amax_f, ub_cast_f, 64, k_act, 1, 1, 8);
        Or(ub_cos_f.ReinterpretCast<uint16_t>(),
           ub_cos_f.ReinterpretCast<uint16_t>(),
           ub_cos2_f.ReinterpretCast<uint16_t>(),
           k_act * 128);
        ShiftRight(ub_cos2_f.ReinterpretCast<uint32_t>(), ub_sin2_f.ReinterpretCast<uint32_t>(), (uint32_t)16, k_act * 64);
        PipeBarrier<PIPE_V>();

        Muls(ub_amax_f, ub_amax_f, -1.0f / 128.0f, 64);
        Or(ub_sin_f.ReinterpretCast<uint16_t>(),
           ub_sin_f.ReinterpretCast<uint16_t>(),
           ub_cos2_f.ReinterpretCast<uint16_t>(),
           k_act * 128);
        Cast(ub_cos2_f, ub_cos_f.ReinterpretCast<bfloat16_t>(),
             AscendC::RoundMode::CAST_NONE, 64, k_act,
             AscendC::UnaryRepeatParams{1, 1, 8, 8});
        PipeBarrier<PIPE_V>();

        Brcb(ub_brcb_f, ub_amax_f, 8, AscendC::BrcbRepeatParams{1, 8});
        Cast(ub_sin2_f, ub_sin_f.ReinterpretCast<bfloat16_t>(),
             AscendC::RoundMode::CAST_NONE, 64, k_act,
             AscendC::UnaryRepeatParams{1, 1, 8, 8});
        PipeBarrier<PIPE_V>();

        Add(ub_cast_f[0], ub_abs_f[0], ub_brcb_f, 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 0, 16, 20, 1});
        Add(ub_cast_f[64], ub_abs_f[64], ub_brcb_f, 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 0, 16, 20, 1});
        PipeBarrier<PIPE_V>();
        Mul(ub_abs_f, ub_cast_f, ub_cast_f, k_act * 128);
        PipeBarrier<PIPE_V>();
        Add(ub_abs_f, ub_abs_f[0], ub_abs_f[64], 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 8, 16, 16});
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float>(ub_amax_f, ub_abs_f, 64, k_act, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(ub_amax_f, ub_amax_f, 1.0f / 128.0f, 64);
        PipeBarrier<PIPE_V>();
        Adds(ub_amax_f, ub_amax_f, eps, 64);
        PipeBarrier<PIPE_V>();
        Sqrt(ub_amax_f, ub_amax_f, 64);
        Duplicate<float>(ub_qscale_f, 1.0f, 64);
        PipeBarrier<PIPE_V>();
        Div(ub_amax_f, ub_qscale_f, ub_amax_f, 64);
        PipeBarrier<PIPE_V>();
        Brcb(ub_brcb_f, ub_amax_f, 8, AscendC::BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();

        Mul(ub_cast_f[0], ub_cast_f[0], ub_brcb_f, 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        Mul(ub_cast_f[64], ub_cast_f[64], ub_brcb_f, 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        PipeBarrier<PIPE_V>();
        Mul(ub_cast_f[0], ub_cast_f[0], ub_ln_w[0], 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
        Mul(ub_cast_f[64], ub_cast_f[64], ub_ln_w[64], 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
        PipeBarrier<PIPE_V>();
        Add(ub_cast_f[0], ub_cast_f[0], ub_ln_b[0], 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
        Add(ub_cast_f[64], ub_cast_f[64], ub_ln_b[64], 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 0});
        PipeBarrier<PIPE_V>();
        Cast(ub_in_bf16, ub_cast_f, AscendC::RoundMode::CAST_RINT, k_act * 128);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> k_u32 = ub_in_bf16.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> kt1_u32 = ub_cast_f.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> kt2_u32 = ub_abs_f.ReinterpretCast<uint32_t>();
        ShiftLeft(kt1_u32, k_u32, (uint32_t)16, k_act * 64);
        ShiftRight(kt2_u32, k_u32, (uint32_t)16, k_act * 64);
        PipeBarrier<PIPE_V>();
        Or(ub_kw_swap_bf16.ReinterpretCast<uint16_t>(),
           kt1_u32.ReinterpretCast<uint16_t>(),
           kt2_u32.ReinterpretCast<uint16_t>(),
           k_act * 128);
        PipeBarrier<PIPE_V>();
        Cast(ub_cast_f, ub_in_bf16, AscendC::RoundMode::CAST_NONE, k_act * 128);
        Cast(ub_abs_f, ub_kw_swap_bf16, AscendC::RoundMode::CAST_NONE, k_act * 128);
        PipeBarrier<PIPE_V>();
        Mul(ub_cast_f, ub_cast_f, ub_cos2_f, 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 8});
        Mul(ub_abs_f, ub_abs_f, ub_sin2_f, 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 8});
        PipeBarrier<PIPE_V>();
        Add(ub_cast_f, ub_cast_f, ub_abs_f, 64, k_act,
            AscendC::BinaryRepeatParams{1, 1, 1, 16, 16, 16});
        PipeBarrier<PIPE_V>();
        Cast(ub_in_bf16, ub_cast_f, AscendC::RoundMode::CAST_RINT, k_act * 128);
        PipeBarrier<PIPE_V>();

        quantize_rows_c8(
            k_act, ub_in_bf16, ub_cast_f, ub_abs_f, ub_out_i8,
            ub_amax_f, ub_qscale_f, ub_brcb_f, ub_scale_f,
            ub_scale_h, ub_const_127, ub_const_inv127, ub_trans_idx, true);
        PipeBarrier<PIPE_V>();
    } else {
        Cast(ub_scale_h, ub_scale_f, AscendC::RoundMode::CAST_RINT, 64);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<uint16_t> ub_scale_u16 = ub_scale_h.ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> ub_brcb_u16 = ub_brcb_f.ReinterpretCast<uint16_t>();
    Brcb(ub_brcb_u16, ub_scale_u16, 2, AscendC::BrcbRepeatParams{1, 8});
    SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);

    const int64_t slot_bytes = static_cast<int64_t>(T) * (slot_is_i64 ? 8 : 4);
    for (int64_t off = 0; off < slot_bytes + 64; off += 64) {
        dcci((__gm__ uint64_t*)(((__gm__ uint8_t*)slot_mapping) + off), SINGLE_CACHE_LINE, CACHELINE_OUT);
    }

    int64_t slots[16];
    bool contig_k = (k_act > 0);
    bool contig_s = (k_rows > 0);
    for (int r = 0; r < k_rows; ++r) {
        const int64_t t = k_t0 + r;
        const int64_t slot = slot_is_i64 ? ((__gm__ int64_t*)slot_mapping)[t]
                                         : static_cast<int64_t>(((__gm__ int32_t*)slot_mapping)[t]);
        slots[r] = slot;
        if (r > 0 && slot != slots[0] + r) {
            contig_s = false;
            if (r < k_act) {
                contig_k = false;
            }
        }
    }
    if (slots[0] < 0 || slots[0] + k_rows > num_slots) {
        contig_s = false;
        if (k_act > 0 && (slots[0] < 0 || slots[0] + k_act > num_slots)) {
            contig_k = false;
        }
    }

    WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
    if (k_act > 0) {
        if (contig_k) {
            GlobalTensor<int8_t> g_kcache;
            g_kcache.SetGlobalBuffer(k_cache + slots[0] * 128, k_act * 128);
            DataCopy(g_kcache, ub_out_i8, k_act * 128);
        } else {
            for (int r = 0; r < k_act; ++r) {
                const int64_t slot = slots[r];
                if (slot >= 0 && slot < num_slots) {
                    GlobalTensor<int8_t> g_kcache;
                    g_kcache.SetGlobalBuffer(k_cache + slot * 128, 128);
                    DataCopy(g_kcache, ub_out_i8[r * 128], 128);
                }
            }
        }
    }

    if (contig_s) {
        GlobalTensor<uint16_t> g_kscale;
        g_kscale.SetGlobalBuffer(k_scale_cache + slots[0], k_rows);
        AscendC::DataCopyExtParams pad_params{1, static_cast<uint32_t>(k_rows * 2), 0, 0, 0};
        DataCopyPad(g_kscale, ub_scale_u16, pad_params);
    } else {
        AscendC::DataCopyExtParams pad_params{1, 2, 0, 0, 0};
        for (int r = 0; r < k_rows; ++r) {
            const int64_t slot = slots[r];
            if (slot >= 0 && slot < num_slots) {
                GlobalTensor<uint16_t> g_kscale;
                g_kscale.SetGlobalBuffer(k_scale_cache + slot, 16);
                DataCopyPad(g_kscale, ub_brcb_u16[r * 16], pad_params);
            }
        }
    }
}

void launch_indexer_prep_qk_c8(uint32_t block_dim, void* stream, void* kw, void* q_in, void* cos_gm, void* sin_gm,
                               void* ln_weight, void* ln_bias, void* slot_mapping,
                               void* k_cache, void* k_scale_cache,
                               void* q_int8_out, void* q_scale_out, void* weights_out,
                               int T, int N, int64_t num_slots, float eps, int slot_is_i64,
                               const int32_t* active_ptr)
{
    indexer_prep_qk_c8_kernel<<<block_dim, nullptr, stream>>>(
        (__gm__ bfloat16_t*)kw, (__gm__ bfloat16_t*)q_in, (__gm__ bfloat16_t*)cos_gm, (__gm__ bfloat16_t*)sin_gm,
        (__gm__ float*)ln_weight, (__gm__ float*)ln_bias, (__gm__ void*)slot_mapping,
        (__gm__ int8_t*)k_cache, (__gm__ uint16_t*)k_scale_cache,
        (__gm__ int8_t*)q_int8_out, (__gm__ half*)q_scale_out, (__gm__ half*)weights_out,
        T, N, num_slots, eps, slot_is_i64, (const __gm__ int32_t*)active_ptr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void tree_splice_topk_kernel(
    __gm__ int32_t* topk_indices,
    const __gm__ int32_t* prefix_len_ptr,
    const __gm__ int32_t* ancestors,
    __gm__ int32_t* cache_buf,
    int T, int K, int W)
{
    const int64_t core = GetBlockIdx();
    if (core >= T || K != 2048 || W != 32) {
        return;
    }

    dcci((__gm__ uint64_t*)prefix_len_ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
    const int32_t prefix_len = *prefix_len_ptr;

    TPipe pipe;
    TBuf<TPosition::VECCALC> b_all;
    pipe.InitBuffer(b_all, 32 * 1024);
    LocalTensor<int32_t> i32_all = b_all.Get<int32_t>();
    LocalTensor<float> f_all = i32_all.ReinterpretCast<float>();
    LocalTensor<uint32_t> u32_all = i32_all.ReinterpretCast<uint32_t>();
    LocalTensor<uint8_t> u8_all = i32_all.ReinterpretCast<uint8_t>();

    LocalTensor<int32_t> ub_src_i32 = i32_all[0];
    LocalTensor<float> ub_src_f = f_all[2048];
    LocalTensor<int32_t> ub_dst_i32 = i32_all[4096];
    LocalTensor<uint8_t> ub_lt_u8 = u8_all[6144 * 4];
    LocalTensor<uint32_t> ub_lt_u32 = u32_all[6144];

    GlobalTensor<int32_t> g_topk, g_anc;
    g_topk.SetGlobalBuffer(topk_indices + core * 2048, 2048);
    g_anc.SetGlobalBuffer((__gm__ int32_t*)(ancestors + core * 32), 32);
    DataCopy(ub_src_i32, g_topk, 2048);
    SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    ResetMask();
    Cast(ub_src_f, ub_src_i32, AscendC::RoundMode::CAST_NONE, 2048);
    PipeBarrier<PIPE_V>();
    CompareScalar(ub_lt_u8, ub_src_f, static_cast<float>(prefix_len), AscendC::CMPMODE::LT, 2048);
    PipeBarrier<PIPE_V>();
    uint64_t dummy_cnt = 0;
    GatherMask<int32_t, uint32_t>(
        ub_dst_i32, ub_src_i32, ub_lt_u32, true, 2048,
        AscendC::GatherMaskParams{1, 32, 8, 0}, dummy_cnt);
    ResetMask();

    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    DataCopy(ub_dst_i32[2016], g_anc, 32);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(g_topk, ub_dst_i32, 2048);
    if (cache_buf != nullptr) {
        GlobalTensor<int32_t> g_cache;
        g_cache.SetGlobalBuffer(cache_buf + core * 2048, 2048);
        DataCopy(g_cache, ub_dst_i32, 2048);
    }
}

void launch_tree_splice_topk(uint32_t block_dim, void* stream, void* topk_indices,
                             const void* prefix_len_ptr, const void* ancestors,
                             void* cache_buf, int T, int K, int W)
{
    tree_splice_topk_kernel<<<block_dim, nullptr, stream>>>(
        (__gm__ int32_t*)topk_indices,
        (const __gm__ int32_t*)prefix_len_ptr,
        (const __gm__ int32_t*)ancestors,
        (__gm__ int32_t*)cache_buf,
        T, K, W);
}
