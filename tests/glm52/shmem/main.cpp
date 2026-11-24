/**
 * Time the TP16 decode reduction as a one-shot push plus ordered local sum.
 *
 * The baseline it has to beat is HCCL: 63.5 microseconds in-graph and 126
 * eager for this message, neither of which varies with the message size. The
 * wire underneath measures 0.31 microseconds within a module.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "tp_push_kernel.h"
#include "utils.h"

namespace {

constexpr uint64_t HEAP_BYTES = 1024UL * 1024UL * 1024UL;
constexpr int64_t ALIGN_BYTES = 32;
// Must match the kernel: flags occupy the window's head, 16 int32 apart so no
// two ranks share a cache line, and the payload follows them.
constexpr int64_t SIGNAL_SLOT_INT32 = 16;
constexpr int64_t PLANES = 2;
constexpr int64_t MAX_SLICES = 32;

int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

namespace {

// Two ranks cancel exactly; the other fourteen contribute e. In bf16 4096 + e
// is 4096, so the fourteen e's survive only where the accumulation is wider
// than the payload.
constexpr float CANCELLING_MAGNITUDE = 4096.0F;
constexpr int CANCELLING_RANKS = 2;
// e cycles 1, 2, 4 with the round, so a stale output from the previous round
// of the same plane reads as a wrong value rather than as a pass.
constexpr int MAGNITUDE_CYCLE = 3;

enum class Payload {
    AllOnes,
    Cancelling,
    TrailingCancel,
};

float SmallMagnitude(int round)
{
    return static_cast<float>(1 << (round % MAGNITUDE_CYCLE));
}

// Where the negative rank sits decides what the payload can catch. Adjacent to
// the positive one, the accumulator is back at zero before the small terms
// arrive, so a bf16 accumulator would pass; straddling them, it must hold 4096
// while fourteen e's arrive, and at 4096 the bf16 step is 32, so a bf16
// accumulator drops every one of them and ends at 0 instead of 14e.
float Contribution(Payload payload, int pe_id, int pe_size, int round)
{
    if (payload == Payload::AllOnes) {
        return 1.0F;
    }
    const int negative_rank = (payload == Payload::Cancelling) ? 1 : pe_size - 1;
    if (pe_id == 0) {
        return CANCELLING_MAGNITUDE;
    }
    if (pe_id == negative_rank) {
        return -CANCELLING_MAGNITUDE;
    }
    return SmallMagnitude(round);
}

float Expectation(Payload payload, int pe_size, int round)
{
    if (payload == Payload::AllOnes) {
        return static_cast<float>(pe_size);
    }
    return static_cast<float>(pe_size - CANCELLING_RANKS) * SmallMagnitude(round);
}

}  // namespace

int main(int argc, char* argv[])
{
    if (argc < 8) {
        std::printf("usage: %s <pe_size> <pe_id> <ip_port> <rows> <hidden> <warmup> <rounds> [stage] [blocks]"
                    " [payload: 0 all-ones, 1 cancelling, 2 trailing-cancel]\n", argv[0]);
        return 1;
    }
    const int pe_size = std::atoi(argv[1]);
    const int pe_id = std::atoi(argv[2]);
    const char* ip_port = argv[3];
    const int rows = std::atoi(argv[4]);
    const int hidden = std::atoi(argv[5]);
    const int warmup = std::atoi(argv[6]);
    const int rounds = std::atoi(argv[7]);
    const int stage = (argc > 8) ? std::atoi(argv[8]) : 0;
    const Payload payload = (argc > 10) ? static_cast<Payload>(std::atoi(argv[10])) : Payload::Cancelling;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(pe_id));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uid{};  // unused by the default config-store bootstrap
    test_set_attr(pe_id, pe_size, HEAP_BYTES, ip_port, uid, &attributes);
    ACL_CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    const int64_t elements = static_cast<int64_t>(rows) * hidden;
    const int64_t stride = AlignUp(elements * sizeof(float), ALIGN_BYTES) / sizeof(float);
    const size_t flag_bytes = PLANES * MAX_SLICES * SIGNAL_SLOT_INT32 * sizeof(int32_t);
    const size_t window_bytes = flag_bytes + PLANES * pe_size * stride * sizeof(float);
    void* window = aclshmem_malloc(window_bytes);
    if (window == nullptr) {
        std::printf("pe %d: aclshmem_malloc failed\n", pe_id);
        return 1;
    }
    ACL_CHECK(aclrtMemset(window, window_bytes, 0, window_bytes));

    // Every rank contributes its own index, so the sum is the known constant
    // pe_size*(pe_size-1)/2 in every element and a wrong order cannot pass.
    std::vector<float> host(elements, static_cast<float>(pe_id));
    void* partial = nullptr;
    void* out = nullptr;
    ACL_CHECK(aclrtMalloc(&partial, elements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&out, elements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(partial, elements * sizeof(float), host.data(), elements * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    // A sentinel distinguishes "the kernel wrote a wrong sum" from "the kernel
    // never wrote the output at all", which a zero cannot.
    const float SENTINEL = -12345.0F;
    std::vector<float> sentinel(elements, SENTINEL);
    ACL_CHECK(aclrtMemcpy(out, elements * sizeof(float), sentinel.data(), elements * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    void* errors = nullptr;
    ACL_CHECK(aclrtMalloc(&errors, 1024 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(errors, 1024 * sizeof(int32_t), 0, 1024 * sizeof(int32_t)));
    // Stage 10 reduces bf16; the window and the partial hold bf16 at the same
    // element count, so the host writes the narrowed values directly.
    std::vector<uint16_t> narrow(elements);
    auto narrow_of = [](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<uint16_t>(bits >> 16);
    };
    auto widen_of = [](uint16_t half) {
        const uint32_t bits = static_cast<uint32_t>(half) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };

    const uint64_t ffts = util_get_ffts_config();
    // The push needs one core per peer; the reduction wants every core it can
    // get, so the launch is sized for the reduction and the push uses the
    // first pe_size cores of it.
    const uint32_t blocks = (argc > 9) ? static_cast<uint32_t>(std::atoi(argv[9])) : 48U;

    // Every rank's window must be zeroed and visible before any rank makes a
    // remotely visible write, or a receiver can spin on a slot whose peer has
    // not finished initialising it -- which is how an untimed device spin
    // earns aicore 507015 under contention.
    aclshmem_barrier_all();

    // Signal values continue across both launches, so neither needs a reset.
    tp_push_demo(blocks, stream, ffts, (float*)partial, (float*)out, (uint8_t*)window, (int32_t*)errors, rows, hidden, warmup, 0, stage);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    aclshmem_barrier_all();

    if (stage == 10) {
        int64_t wrong_rounds = 0;
        double total = 0.0;
        for (int round = 0; round < rounds; ++round) {
            const float mine = Contribution(payload, pe_id, pe_size, round);
            const float want = Expectation(payload, pe_size, round);
            for (int64_t index = 0; index < elements; ++index) {
                narrow[index] = narrow_of(mine);
            }
            ACL_CHECK(aclrtMemcpy(partial, elements * sizeof(uint16_t), narrow.data(),
                                  elements * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
            aclshmem_barrier_all();
            timespec round_begin;
            timespec round_end;
            clock_gettime(CLOCK_MONOTONIC, &round_begin);
            tp_push_demo(blocks, stream, ffts, (float*)partial, (float*)out, (uint8_t*)window,
                         (int32_t*)errors, rows, hidden, 1, round, stage);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            clock_gettime(CLOCK_MONOTONIC, &round_end);
            total += (round_end.tv_sec - round_begin.tv_sec) * 1e6
                   + (round_end.tv_nsec - round_begin.tv_nsec) / 1e3;
            ACL_CHECK(aclrtMemcpy(narrow.data(), elements * sizeof(uint16_t), out,
                                  elements * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
            if (pe_id == 0 && round == 0) {
                std::vector<uint16_t> peek(64);
                const size_t head = PLANES * MAX_SLICES * SIGNAL_SLOT_INT32 * sizeof(int32_t);
                for (int slot = 0; slot < 3; ++slot) {
                    ACL_CHECK(aclrtMemcpy(peek.data(), 8 * sizeof(uint16_t),
                                          (uint8_t*)window + head + slot * stride * sizeof(float),
                                          8 * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
                    std::printf("  window slot %d:", slot);
                    for (int k = 0; k < 4; ++k) {
                        std::printf(" %.3f", widen_of(peek[k]));
                    }
                    std::printf("\n");
                }
            }
            int64_t bad = 0;
            for (int64_t index = 0; index < elements; ++index) {
                if (widen_of(narrow[index]) != want) {
                    ++bad;
                }
            }
            if (bad != 0) {
                ++wrong_rounds;
                if (pe_id == 0 && wrong_rounds <= 3) {
                    std::printf("  round %d want %.1f got %.1f wrong %ld of %ld\n", round, want,
                                widen_of(narrow[0]), (long)bad, (long)elements);
                }
            }
        }
        // The loop above pays a host memcpy and a stream synchronisation per
        // round, which the amortising stages do not; one launch over every
        // round is what makes this number comparable with theirs.
        aclshmem_barrier_all();
        timespec amortised_begin;
        timespec amortised_end;
        clock_gettime(CLOCK_MONOTONIC, &amortised_begin);
        tp_push_demo(blocks, stream, ffts, (float*)partial, (float*)out, (uint8_t*)window, (int32_t*)errors,
                     rows, hidden, rounds, 0, stage);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        clock_gettime(CLOCK_MONOTONIC, &amortised_end);
        const double amortised_us = (amortised_end.tv_sec - amortised_begin.tv_sec) * 1e6
                                  + (amortised_end.tv_nsec - amortised_begin.tv_nsec) / 1e3;

        if (pe_id == 0) {
            std::printf("RESULT stage 10 per_reduction_us %.3f amortised_us %.3f wrong_rounds %ld of %d\n",
                        total / rounds, amortised_us / rounds, (long)wrong_rounds, rounds);
        }
        aclshmem_free(window);
        aclshmem_finalize();
        ACL_CHECK(aclrtResetDevice(pe_id));
        ACL_CHECK(aclFinalize());
        return wrong_rounds == 0 ? 0 : 1;
    }

    struct timespec begin;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    tp_push_demo(blocks, stream, ffts, (float*)partial, (float*)out, (uint8_t*)window, (int32_t*)errors, rows, hidden, rounds, warmup, stage);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double total_us =
        (end.tv_sec - begin.tv_sec) * 1e6 + (end.tv_nsec - begin.tv_nsec) / 1e3;
    std::vector<float> result(elements, 0.0F);
    ACL_CHECK(aclrtMemcpy(result.data(), elements * sizeof(float), out, elements * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    // Only the synchronised stages 0 and 11 produce a sum; the push-free and
    // sync-free stages leave `out` untouched or racy, so checking them against
    // the sum would report every element wrong.
    const float expected = static_cast<float>(pe_size * (pe_size - 1) / 2);
    const bool reduces = (stage == 0 || stage == 11);
    int64_t wrong = 0;
    if (reduces) {
        for (int64_t i = 0; i < elements; ++i) {
            if (result[i] != expected) {
                ++wrong;
            }
        }
    }
    int64_t untouched = 0;
    for (int64_t i = 0; i < elements; ++i) {
        if (result[i] == SENTINEL) {
            ++untouched;
        }
    }
    std::vector<int32_t> error_counts(1024, 0);
    ACL_CHECK(aclrtMemcpy(error_counts.data(), 1024 * sizeof(int32_t), errors, 1024 * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t probe_failures = 0;
    for (int32_t value : error_counts) {
        probe_failures += value;
    }
    if (pe_id == 0) {
        std::printf("pe_size %d rows %d hidden %d bytes %ld rounds %d blocks %u\n", pe_size, rows,
                    hidden, (long)(elements * sizeof(float)), rounds, blocks);
        std::printf("RESULT stage %d per_reduction_us %.3f probe_failures %ld wrong %ld untouched %ld of %ld expected %.1f got %.1f\n",
                    stage, total_us / rounds, (long)probe_failures, (long)wrong, (long)untouched, (long)elements, expected, result[0]);
    }

    aclshmem_free(window);
    aclshmem_finalize();
    ACL_CHECK(aclrtFree(partial));
    ACL_CHECK(aclrtFree(out));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(pe_id));
    ACL_CHECK(aclFinalize());
    return (reduces && wrong != 0) ? 1 : 0;
}
