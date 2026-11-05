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
#include "floor_kernel.h"

namespace {

constexpr int64_t TILE_FLOATS = 32 * 1024 / sizeof(float);
constexpr uint32_t BLOCK_DIM = 20;

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
    floor_demo(BLOCK_DIM, stream, source, sink, mode, 8);
    Check(aclrtSynchronizeStream(stream), "warmup");
    timespec begin;
    timespec end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    floor_demo(BLOCK_DIM, stream, source, sink, mode, iterations);
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

    const char* labels[] = {"empty body", "one 32 KiB read",
                            "read + vector pass", "read + pass + write"};
    std::printf("blocks %u, tile %zu KiB\n", BLOCK_DIM,
                TILE_FLOATS * sizeof(float) / 1024);
    const int modes[] = {0, 1, 2, 3};
    for (int mode : modes) {
        const double many = Time(stream, (float*)source, (float*)sink, mode, iterations);
        const double once = Time(stream, (float*)source, (float*)sink, mode, 1);
        std::printf("mode %d  %7.3f us an iteration  %8.3f us for one launch  %s\n",
                    mode, many / iterations, once, labels[mode]);
        std::fflush(stdout);
    }

    Check(aclrtFree(source), "free source");
    Check(aclrtFree(sink), "free sink");
    Check(aclrtDestroyStream(stream), "destroy stream");
    Check(aclrtResetDevice(device), "reset device");
    Check(aclFinalize(), "aclFinalize");
    return 0;
}
