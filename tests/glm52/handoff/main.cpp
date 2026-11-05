/**
 * Time the three handoff modes against each other on one die.
 *
 * mode 0 is the serialised cube-to-vector handoff, mode 1 the same traffic
 * with no ordering, mode 2 the flag with no payload. The three are each
 * other's control: if 0 and 1 agree then either the flag does nothing or the
 * dependency is free, and 2 says which.
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "acl/acl.h"
#include "handoff_kernel.h"

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

    std::printf("tile %zu KiB  blocks %u  iterations %d\n",
                TILE_FLOATS * sizeof(float) / 1024, BLOCK_DIM, iterations);
    const int modes[] = {0, 1};
    for (int mode : modes) {
        handoff_demo(BLOCK_DIM, stream, (float*)source, (float*)sink, mode, 64);
        Check(aclrtSynchronizeStream(stream), "warmup");

        timespec begin;
        timespec end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        handoff_demo(BLOCK_DIM, stream, (float*)source, (float*)sink, mode, iterations);
        Check(aclrtSynchronizeStream(stream), "measure");
        clock_gettime(CLOCK_MONOTONIC, &end);

        const double each = Elapsed(begin, end) / iterations;
        std::printf("mode %d  %8.3f us a round trip  %s\n", mode, each,
                    mode == 0 ? "the read depends on the write"
                              : "same bytes, no dependency");
        std::fflush(stdout);
    }

    Check(aclrtFree(source), "free source");
    Check(aclrtFree(sink), "free sink");
    Check(aclrtDestroyStream(stream), "destroy stream");
    Check(aclrtResetDevice(device), "reset device");
    Check(aclFinalize(), "aclFinalize");
    return 0;
}
