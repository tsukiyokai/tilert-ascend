/**
 * Time one kernel launch, and then each term added to its body.
 *
 * A launch is measured by dividing a long loop by its iterations, which gives
 * the per-iteration floor rather than the launch itself; a second run with
 * one iteration gives the launch. Both are wanted: the model's operators are
 * one launch each, and the fixed cost under test is per operator.
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "acl/acl.h"
#include "gather_kernel.h"

namespace {

constexpr int64_t SLOTS = 2048;
constexpr int64_t SLOT_FLOATS = 1152 / sizeof(float);
constexpr int64_t TILE_FLOATS = SLOTS * SLOT_FLOATS;
constexpr uint32_t BLOCK_DIM = 16;

double Elapsed(const timespec& begin, const timespec& end)
{
    return (end.tv_sec - begin.tv_sec) * 1e6 + (end.tv_nsec - begin.tv_nsec) / 1e3;
}

void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", what, status);
        std::exit(1);
    }
}

double Time(void* stream, float* source, float* sink, int mode, int iterations)
{
    gather_demo(BLOCK_DIM, stream, source, sink, mode, 8);
    Check(aclrtSynchronizeStream(stream), "warmup");
    timespec begin;
    timespec end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    gather_demo(BLOCK_DIM, stream, source, sink, mode, iterations);
    Check(aclrtSynchronizeStream(stream), "measure");
    clock_gettime(CLOCK_MONOTONIC, &end);
    return Elapsed(begin, end);
}

}  // namespace

int main(int argc, char* argv[])
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 2000;

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    const size_t bytes = BLOCK_DIM * TILE_FLOATS * sizeof(float);
    void* source = nullptr;
    void* sink = nullptr;
    Check(aclrtMalloc(&source, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc source");
    Check(aclrtMalloc(&sink, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc sink");
    Check(aclrtMemset(source, bytes, 1, bytes), "memset source");

    const char* labels[] = {"contiguous", "one page apart",
                            "a permutation, no locality"};
    std::printf("blocks %u, %ld slots of %zu B = %zu KiB\n", BLOCK_DIM,
                (long)SLOTS, SLOT_FLOATS * sizeof(float),
                TILE_FLOATS * sizeof(float) / 1024);
    const int modes[] = {0, 1, 2};
    for (int mode : modes) {
        const double many = Time(stream, (float*)source, (float*)sink, mode, iterations);
        const double once = Time(stream, (float*)source, (float*)sink, mode, 1);
        const double each = many / iterations;
        const double gib = (double)TILE_FLOATS * sizeof(float) / (1 << 30);
        std::printf("mode %d  %8.2f us a pass  %7.1f GB/s  %s\n",
                    mode, each, gib / (each / 1e6), labels[mode]);
        std::fflush(stdout);
    }

    Check(aclrtFree(source), "free source");
    Check(aclrtFree(sink), "free sink");
    Check(aclrtDestroyStream(stream), "destroy stream");
    Check(aclrtResetDevice(device), "reset device");
    Check(aclFinalize(), "aclFinalize");
    return 0;
}
