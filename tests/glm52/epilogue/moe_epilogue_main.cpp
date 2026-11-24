/**
 * Does one kernel beat three at the MoE epilogue, and give the same bytes?
 *
 * Both arms run in this process on the same inputs, alternating, so the
 * comparison does not cross a run. Correctness is checked twice: the fused
 * arm against a host reference that rounds where the device rounds, and the
 * two arms against each other bit for bit.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "acl/acl.h"
#include "moe_epilogue_kernel.h"

namespace {

constexpr int64_t HIDDEN = 6144;
constexpr uint32_t BLOCK_DIM = 48;

void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", what, status);
        std::exit(1);
    }
}

double Elapsed(const timespec& begin, const timespec& end)
{
    return (end.tv_sec - begin.tv_sec) * 1e6
           + (end.tv_nsec - begin.tv_nsec) / 1e3;
}

/** Round a float to BF16 the way the device does, nearest with ties even. */
uint16_t ToBf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t low = bits & 0xFFFF;
    const uint32_t carry = (low > 0x8000)
                           || (low == 0x8000 && ((bits >> 16) & 1));
    return static_cast<uint16_t>((bits >> 16) + carry);
}

float FromBf16(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

/** The three-stage arithmetic, with both of its roundings. */
float Reference(uint16_t routed, uint16_t shared, float routed_scale)
{
    const uint16_t scaled = ToBf16(FromBf16(routed) * routed_scale);
    return FromBf16(ToBf16(FromBf16(scaled) + FromBf16(shared)));
}

double TimeFused(void* stream, void* routed, void* shared, float* out,
                 float scale, int64_t elements, int repeats, int inner)
{
    timespec begin;
    timespec end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (int step = 0; step < repeats; ++step) {
        moe_epilogue_fused_launch(BLOCK_DIM, stream, routed, shared, out,
                                  scale, elements, inner);
    }
    Check(aclrtSynchronizeStream(stream), "fused");
    clock_gettime(CLOCK_MONOTONIC, &end);
    return Elapsed(begin, end) / repeats;
}

double TimeStaged(void* stream, void* routed, void* shared, float* out,
                  float scale, int64_t elements, int repeats, int inner)
{
    timespec begin;
    timespec end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (int step = 0; step < repeats; ++step) {
        for (int stage = 0; stage < 3; ++stage) {
            moe_epilogue_stage_launch(BLOCK_DIM, stream, routed, shared, out,
                                      scale, stage, elements, inner);
        }
    }
    Check(aclrtSynchronizeStream(stream), "staged");
    clock_gettime(CLOCK_MONOTONIC, &end);
    return Elapsed(begin, end) / repeats;
}

}  // namespace

int main(int argc, char** argv)
{
    const int64_t rows = (argc > 1) ? std::atoll(argv[1]) : 8;
    const int32_t device = (argc > 2) ? std::atoi(argv[2]) : 0;
    const int repeats = (argc > 3) ? std::atoi(argv[3]) : 200;
    const int rounds = (argc > 4) ? std::atoi(argv[4]) : 6;
    const float scale = 2.5F;
    const int64_t elements = rows * HIDDEN;

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::vector<uint16_t> routed_host(elements);
    std::vector<uint16_t> shared_host(elements);
    for (int64_t index = 0; index < elements; ++index) {
        // Values a BF16 cannot hold exactly, so both roundings are exercised.
        routed_host[index] = ToBf16(0.013F * static_cast<float>(index % 701)
                                    - 1.7F);
        shared_host[index] = ToBf16(0.0071F * static_cast<float>(index % 457)
                                     + 0.33F);
    }

    const size_t narrow_bytes = elements * sizeof(uint16_t);
    const size_t wide_bytes = elements * sizeof(float);
    void* routed = nullptr;
    void* shared = nullptr;
    void* scratch = nullptr;
    void* fused_out = nullptr;
    void* staged_out = nullptr;
    Check(aclrtMalloc(&routed, narrow_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "a");
    Check(aclrtMalloc(&shared, narrow_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "b");
    Check(aclrtMalloc(&scratch, narrow_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "c");
    Check(aclrtMalloc(&fused_out, wide_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "d");
    Check(aclrtMalloc(&staged_out, wide_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "e");
    Check(aclrtMemcpy(shared, narrow_bytes, shared_host.data(), narrow_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload shared");

    // Correctness. The staged arm consumes its own copy because it rewrites
    // the routed buffer in place, which is what makes it three kernels.
    Check(aclrtMemcpy(routed, narrow_bytes, routed_host.data(), narrow_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload routed");
    moe_epilogue_fused_launch(BLOCK_DIM, stream, routed, shared,
                              (float*)fused_out, scale, elements, 1);
    Check(aclrtSynchronizeStream(stream), "fused once");
    Check(aclrtMemcpy(scratch, narrow_bytes, routed_host.data(), narrow_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload scratch");
    for (int stage = 0; stage < 3; ++stage) {
        moe_epilogue_stage_launch(BLOCK_DIM, stream, scratch, shared,
                                  (float*)staged_out, scale, stage, elements,
                                  1);
    }
    Check(aclrtSynchronizeStream(stream), "staged once");

    std::vector<float> fused_host(elements);
    std::vector<float> staged_host(elements);
    Check(aclrtMemcpy(fused_host.data(), wide_bytes, fused_out, wide_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "download fused");
    Check(aclrtMemcpy(staged_host.data(), wide_bytes, staged_out, wide_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "download staged");

    int64_t wrong_reference = 0;
    int64_t wrong_arms = 0;
    for (int64_t index = 0; index < elements; ++index) {
        const float want = Reference(routed_host[index], shared_host[index],
                                     scale);
        wrong_reference += (fused_host[index] != want);
        wrong_arms += (fused_host[index] != staged_host[index]);
    }
    std::printf("rows %lld  elements %lld  blocks %u\n", (long long)rows,
                (long long)elements, BLOCK_DIM);
    std::printf("fused against host reference: %lld of %lld differ\n",
                (long long)wrong_reference, (long long)elements);
    std::printf("fused against staged arm:     %lld of %lld differ\n",
                (long long)wrong_arms, (long long)elements);

    // Two instruments. One launch a call gives the issue-bound cost, which is
    // what a stream of separate kernels pays from the host; many passes inside
    // one launch amortize the issue away and leave the device cost, which is
    // what an in-graph replay pays. A scale of one keeps the staged arm, which
    // rewrites its input every pass, from running away to infinity.
    const int inner = 64;
    TimeFused(stream, routed, shared, (float*)fused_out, scale, elements, 20,
              1);
    TimeStaged(stream, scratch, shared, (float*)staged_out, scale, elements,
               20, 1);
    for (int round = 0; round < rounds; ++round) {
        const double fused = TimeFused(stream, routed, shared,
                                       (float*)fused_out, scale, elements,
                                       repeats, 1);
        const double staged = TimeStaged(stream, scratch, shared,
                                         (float*)staged_out, scale, elements,
                                         repeats, 1);
        const double deep_fused = TimeFused(stream, routed, shared,
                                            (float*)fused_out, 1.0F, elements,
                                            4, inner) / inner;
        const double deep_staged = TimeStaged(stream, scratch, shared,
                                              (float*)staged_out, 1.0F,
                                              elements, 4, inner) / inner;
        std::printf("round %d  issue-bound fused %7.3f staged %7.3f %.2fx"
                    "   device fused %6.3f staged %6.3f %.2fx\n", round,
                    fused, staged, staged / fused, deep_fused, deep_staged,
                    deep_staged / deep_fused);
    }

    Check(aclrtFree(routed), "free");
    Check(aclrtFree(shared), "free");
    Check(aclrtFree(scratch), "free");
    Check(aclrtFree(fused_out), "free");
    Check(aclrtFree(staged_out), "free");
    Check(aclrtDestroyStream(stream), "destroy");
    Check(aclrtResetDevice(device), "reset");
    Check(aclFinalize(), "finalize");
    return (wrong_reference == 0 && wrong_arms == 0) ? 0 : 1;
}
