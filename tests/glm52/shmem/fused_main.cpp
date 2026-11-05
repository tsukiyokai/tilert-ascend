/** Prototype driver for INT8 projection and peer reduction. */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "fused_proj_reduce_kernel.h"
#include "utils.h"

namespace {

constexpr int64_t SIGNAL_SLOT_INT32 = 16;
constexpr int64_t MAX_SLICES = 32;
constexpr int64_t PLANES = 2;
constexpr int64_t ALIGN_BYTES = 32;
constexpr uint64_t HEAP_BYTES = 1024UL * 1024UL * 1024UL;
constexpr uint32_t BLOCK_DIM = 16;

int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

double Elapsed(const timespec& begin, const timespec& end)
{
    return (end.tv_sec - begin.tv_sec) * 1e6 + (end.tv_nsec - begin.tv_nsec) / 1e3;
}

}  // namespace

int main(int argc, char* argv[])
{
    if (argc < 8) {
        std::printf("usage: %s <pe_size> <pe_id> <ip_port> <rows> <inner> <cols> <rounds>\n", argv[0]);
        return 1;
    }
    const int pe_size = std::atoi(argv[1]);
    const int pe_id = std::atoi(argv[2]);
    const char* ip_port = argv[3];
    const int rows = std::atoi(argv[4]);
    const int inner = std::atoi(argv[5]);
    const int cols = std::atoi(argv[6]);
    const int rounds = std::atoi(argv[7]);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(pe_id));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uid{};
    test_set_attr(pe_id, pe_size, HEAP_BYTES, ip_port, uid, &attributes);
    ACL_CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    const int64_t elements = static_cast<int64_t>(rows) * cols;
    const int64_t stride = AlignUp(elements * sizeof(float), ALIGN_BYTES) / sizeof(float);
    const size_t head_bytes = PLANES * MAX_SLICES * SIGNAL_SLOT_INT32 * sizeof(int32_t);
    const size_t window_bytes = head_bytes + PLANES * pe_size * stride * sizeof(float);
    void* window = aclshmem_malloc(window_bytes);
    if (window == nullptr) {
        std::printf("pe %d: aclshmem_malloc(%zu) failed\n", pe_id, window_bytes);
        return 1;
    }
    ACL_CHECK(aclrtMemset(window, window_bytes, 0, window_bytes));

    // All-ones inputs give pe_size * inner. Equal contributions cannot
    // distinguish a duplicated peer from a missing peer.
    void* activation = nullptr;
    void* weight = nullptr;
    void* partial = nullptr;
    void* out = nullptr;
    void* tiling = nullptr;
    const size_t activation_bytes = static_cast<size_t>(rows) * inner;
    const size_t weight_bytes = static_cast<size_t>(inner) * cols;
    ACL_CHECK(aclrtMalloc(&activation, activation_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&weight, weight_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&partial, elements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&out, elements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&tiling, 4096, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<int8_t> ones_a(activation_bytes, 1);
    std::vector<int8_t> ones_b(weight_bytes, 1);
    ACL_CHECK(aclrtMemcpy(activation, activation_bytes, ones_a.data(), activation_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(weight, weight_bytes, ones_b.data(), weight_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // A sentinel separates "summed wrongly" from "never written", which a zero
    // cannot: an untouched output and a dropped peer both read as zero.
    const float SENTINEL = -12345.0F;
    std::vector<float> sentinel(elements, SENTINEL);
    ACL_CHECK(aclrtMemcpy(out, elements * sizeof(float), sentinel.data(), elements * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    const uint64_t ffts = util_get_ffts_config();
    aclshmem_barrier_all();

    fused_proj_reduce_demo(BLOCK_DIM, stream, ffts, (int8_t*)activation, (int8_t*)weight, (float*)partial,
                           (float*)out, (uint8_t*)window, (uint8_t*)tiling, rows, cols, 20, 0);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    aclshmem_barrier_all();

    timespec begin;
    timespec end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    fused_proj_reduce_demo(BLOCK_DIM, stream, ffts, (int8_t*)activation, (int8_t*)weight, (float*)partial,
                           (float*)out, (uint8_t*)window, (uint8_t*)tiling, rows, cols, rounds, 20);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    clock_gettime(CLOCK_MONOTONIC, &end);

    std::vector<float> result(elements, 0.0F);
    ACL_CHECK(aclrtMemcpy(result.data(), elements * sizeof(float), out, elements * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    const float expected = static_cast<float>(pe_size) * static_cast<float>(inner);
    int64_t wrong = 0;
    int64_t untouched = 0;
    for (int64_t index = 0; index < elements; ++index) {
        if (result[index] == SENTINEL) {
            ++untouched;
        }
        if (result[index] != expected) {
            ++wrong;
        }
    }
    if (pe_id == 0) {
        std::printf("pe_size %d rows %d inner %d cols %d rounds %d\n", pe_size, rows, inner, cols, rounds);
        std::printf("RESULT per_layer_us %.3f wrong %ld untouched %ld of %ld expected %.1f got %.1f\n",
                    Elapsed(begin, end) / rounds, (long)wrong, (long)untouched, (long)elements, expected,
                    result[0]);
    }

    aclshmem_free(window);
    aclshmem_finalize();
    ACL_CHECK(aclrtFree(activation));
    ACL_CHECK(aclrtFree(weight));
    ACL_CHECK(aclrtFree(partial));
    ACL_CHECK(aclrtFree(out));
    ACL_CHECK(aclrtFree(tiling));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(pe_id));
    ACL_CHECK(aclFinalize());
    return wrong == 0 ? 0 : 1;
}
