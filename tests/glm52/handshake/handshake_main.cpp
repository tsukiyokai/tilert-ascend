/** One variant a process, so a deadlock is a timeout and not a lost shell. */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "handshake_kernel.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int64_t STRIDE = 8;
constexpr int64_t SLOTS = 4;

void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", what, status);
        std::exit(2);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const int64_t reverse_mode = (argc > 1) ? std::atoll(argv[1]) : 2;
    const int64_t pattern = (argc > 2) ? std::atoll(argv[2]) : 0;
    const int64_t rounds = (argc > 3) ? std::atoll(argv[3]) : 1;
    const int32_t device = (argc > 4) ? std::atoi(argv[4]) : 0;

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    uint64_t sync_base = 0;
    uint32_t sync_length = 0;
    if (rtGetC2cCtrlAddr(&sync_base, &sync_length) != 0) {
        std::printf("rtGetC2cCtrlAddr failed\n");
        return 2;
    }

    void* progress = nullptr;
    Check(aclrtMalloc(&progress, SLOTS * STRIDE * sizeof(int32_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
    const std::vector<int32_t> blank(SLOTS * STRIDE, -1);
    Check(aclrtMemcpy(progress, blank.size() * sizeof(int32_t), blank.data(),
                      blank.size() * sizeof(int32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE), "blank");

    handshake_launch(stream, sync_base, progress, reverse_mode, pattern,
                     rounds);
    Check(aclrtSynchronizeStream(stream), "synchronize");

    std::vector<int32_t> reached(SLOTS * STRIDE);
    Check(aclrtMemcpy(reached.data(), reached.size() * sizeof(int32_t),
                      progress, reached.size() * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST), "download");
    std::printf("reverse mode %lld  pattern %lld  rounds %lld  live  "
                "cube %d  sub0 %d  sub1 %d\n", (long long)reverse_mode,
                (long long)pattern, (long long)rounds, reached[0],
                reached[STRIDE], reached[2 * STRIDE]);

    Check(aclrtDestroyStream(stream), "destroy");
    Check(aclrtResetDevice(device), "reset");
    Check(aclFinalize(), "finalize");
    return 0;
}
