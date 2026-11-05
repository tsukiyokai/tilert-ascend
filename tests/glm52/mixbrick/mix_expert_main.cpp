/**
 * Does one MIX kernel beat a matmul kernel plus a vector kernel?
 *
 * Both arms run in this process on the same inputs, alternating, at the same
 * block count, so the comparison is of the cube-to-vector boundary and not of
 * a parallel decomposition. Correctness is against a host reference that
 * computes the INT32 product exactly and rounds to BF16 once, where the
 * device rounds.
 */
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "acl/acl.h"
#include "mix_expert_kernel.h"

// Not in the public headers; the SHMEM tree reaches it the same way.
extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
constexpr size_t TILING_BYTES = 256;
constexpr int64_t HALF_CHANNELS = 128;

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

/** Round a float to IEEE half the way the device's `Cast` does.
 *
 * The kernel quantises float to INT8 through half, which is the Ascend C
 * idiom, so the reference has to round where the device rounds -- twice --
 * or 9% of the codes differ by one and the reference is what is wrong.
 */
float ToHalf(float value)
{
    const float magnitude = std::fabs(value);
    if (magnitude < 6.103515625e-05F) {
        return value;
    }
    int exponent = 0;
    const float fraction = std::frexp(magnitude, &exponent);
    const float quantum = std::ldexp(1.0F, exponent - 11);
    const float rounded = std::nearbyintf(magnitude / quantum) * quantum;
    return (value < 0.0F) ? -rounded : rounded;
}

uint16_t ToBf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t low = bits & 0xFFFF;
    const uint32_t carry = (low > 0x8000)
                           || (low == 0x8000 && ((bits >> 16) & 1));
    return static_cast<uint16_t>((bits >> 16) + carry);
}

}  // namespace

int main(int argc, char** argv)
{
    const int64_t rows = (argc > 1) ? std::atoll(argv[1]) : 8;
    const int32_t device = (argc > 2) ? std::atoi(argv[2]) : 0;
    const int repeats = (argc > 3) ? std::atoi(argv[3]) : 100;
    const int rounds = (argc > 4) ? std::atoi(argv[4]) : 5;
    // 0 both arms, 1 the staged arm alone, 2 the fused arm alone. A hang
    // tells you nothing unless you can run one side of it.
    const int arm = (argc > 5) ? std::atoi(argv[5]) : 0;
    const int stages = (argc > 6) ? std::atoi(argv[6]) : 7;
    const uint32_t blocks = (argc > 7) ? std::atoi(argv[7]) : 4;
    const bool uniform = (argc > 8) && std::atoi(argv[8]) != 0;

    std::vector<uint8_t> tiling_host(TILING_BYTES, 0);
    const int tiling_status = mix_build_tiling(
        static_cast<int>(rows),
        static_cast<int>((stages & 32) ? 2 * HALF_CHANNELS
                                       : HALF_CHANNELS / blocks),
        static_cast<int>(HIDDEN), tiling_host.data());
    std::vector<uint8_t> down_tiling_host(TILING_BYTES, 0);
    const int down_status = mix_build_tiling(
        static_cast<int>(rows), static_cast<int>(HIDDEN / blocks),
        static_cast<int>(HALF), down_tiling_host.data());
    if (tiling_status != 0) {
        std::printf("mix_build_tiling returned %d\n", tiling_status);
        return 1;
    }

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    // The control address exists once a device is bound, not before.
    uint64_t sync_base = 0;
    uint32_t sync_length = 0;
    const int32_t sync_status = rtGetC2cCtrlAddr(&sync_base, &sync_length);
    if (sync_status != 0) {
        std::printf("rtGetC2cCtrlAddr returned %d\n", sync_status);
        return 1;
    }

    std::vector<int8_t> x_host(rows * HIDDEN);
    std::vector<int8_t> weight_host(GATE_UP * HIDDEN);
    std::vector<float> scale_host(GATE_UP);
    std::vector<float> token_host(32, 0.0F);
    for (int64_t index = 0; index < rows * HIDDEN; ++index) {
        x_host[index] = static_cast<int8_t>((index % 37) - 18);
    }
    for (int64_t index = 0; index < GATE_UP * HIDDEN; ++index) {
        weight_host[index] = static_cast<int8_t>((index % 23) - 11);
    }
    for (int64_t index = 0; index < GATE_UP; ++index) {
        scale_host[index] = 0.00031F + 0.0000007F * static_cast<float>(index);
    }
    for (int64_t index = 0; index < rows; ++index) {
        // `uniform` makes every row's scale identical, so a wrong per-row
        // scalar read becomes invisible: it tells a bad scalar read apart
        // from a stale product.
        token_host[index] = uniform ? 0.021F
                                    : 0.021F + 0.003F
                                      * static_cast<float>(index);
    }

    void* x = nullptr;
    void* weight = nullptr;
    void* partial = nullptr;
    void* scale = nullptr;
    void* token = nullptr;
    void* fused_out = nullptr;
    void* staged_out = nullptr;
    void* tiling = nullptr;
    void* down_tiling = nullptr;
    void* middle = nullptr;
    void* down = nullptr;
    void* down_scale = nullptr;
    void* wide = nullptr;
    void* narrow_out = nullptr;
    Check(aclrtMalloc(&x, x_host.size(), ACL_MEM_MALLOC_HUGE_FIRST), "x");
    Check(aclrtMalloc(&weight, weight_host.size(),
                      ACL_MEM_MALLOC_HUGE_FIRST), "weight");
    Check(aclrtMalloc(&partial, rows * GATE_UP * sizeof(int32_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "partial");
    Check(aclrtMalloc(&scale, GATE_UP * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "scale");
    Check(aclrtMalloc(&token, 32 * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "token");
    Check(aclrtMalloc(&fused_out, rows * HALF * sizeof(uint16_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "fused out");
    Check(aclrtMalloc(&staged_out, rows * HALF * sizeof(uint16_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "staged out");
    Check(aclrtMalloc(&tiling, TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST),
          "tiling");
    Check(aclrtMalloc(&down_tiling, TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST),
          "down tiling");
    Check(aclrtMalloc(&middle, rows * HALF, ACL_MEM_MALLOC_HUGE_FIRST),
          "middle");
    Check(aclrtMalloc(&down, HIDDEN * HALF, ACL_MEM_MALLOC_HUGE_FIRST),
          "down");
    Check(aclrtMalloc(&down_scale, HIDDEN * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "down scale");
    Check(aclrtMalloc(&wide, rows * HIDDEN * sizeof(int32_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "wide");
    Check(aclrtMalloc(&narrow_out, rows * HIDDEN * sizeof(uint16_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "narrow out");

    std::vector<int8_t> down_host(HIDDEN * HALF);
    std::vector<float> down_scale_host(HIDDEN);
    for (size_t index = 0; index < down_host.size(); ++index) {
        down_host[index] = static_cast<int8_t>((index % 29) - 14);
    }
    for (int64_t index = 0; index < HIDDEN; ++index) {
        down_scale_host[index] = 0.00047F
                                 + 0.0000003F * static_cast<float>(index);
    }
    void* middle_scale = nullptr;
    void* hidden = nullptr;
    void* maxima = nullptr;
    Check(aclrtMalloc(&middle_scale, 32 * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "middle scale");
    Check(aclrtMalloc(&hidden, rows * HALF * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "hidden");
    Check(aclrtMalloc(&maxima, 32 * 32 * 8 * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "maxima");
    const std::vector<float> zeros(32 * 32 * 8, 0.0F);
    Check(aclrtMemcpy(maxima, 32 * 32 * 8 * sizeof(float), zeros.data(),
                      32 * 32 * 8 * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE), "clear maxima");

    Check(aclrtMemcpy(x, x_host.size(), x_host.data(), x_host.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload x");
    Check(aclrtMemcpy(weight, weight_host.size(), weight_host.data(),
                      weight_host.size(), ACL_MEMCPY_HOST_TO_DEVICE),
          "upload weight");
    Check(aclrtMemcpy(scale, GATE_UP * sizeof(float), scale_host.data(),
                      GATE_UP * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
          "upload scale");
    Check(aclrtMemcpy(token, 32 * sizeof(float), token_host.data(),
                      32 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
          "upload token");
    Check(aclrtMemcpy(tiling, TILING_BYTES, tiling_host.data(), TILING_BYTES,
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload tiling");
    Check(aclrtMemcpy(down_tiling, TILING_BYTES, down_tiling_host.data(),
                      TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE),
          "upload down tiling");
    Check(aclrtMemcpy(down, down_host.size(), down_host.data(),
                      down_host.size(), ACL_MEMCPY_HOST_TO_DEVICE),
          "upload down");
    Check(aclrtMemcpy(down_scale, HIDDEN * sizeof(float),
                      down_scale_host.data(), HIDDEN * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload down scale");

    // A sentinel separates "written wrongly" from "never written", which a
    // zero cannot: an untouched output and a zeroed one read the same, and
    // the allocator hands back the previous process's buffer often enough to
    // make the difference alternate.
    const size_t out_bytes_pre = rows * HALF * sizeof(uint16_t);
    const uint16_t SENTINEL = 0x7777;
    std::vector<uint16_t> sentinel(rows * HALF, SENTINEL);
    Check(aclrtMemcpy(fused_out, out_bytes_pre, sentinel.data(),
                      out_bytes_pre, ACL_MEMCPY_HOST_TO_DEVICE),
          "sentinel fused");
    Check(aclrtMemcpy(staged_out, out_bytes_pre, sentinel.data(),
                      out_bytes_pre, ACL_MEMCPY_HOST_TO_DEVICE),
          "sentinel staged");

    if (arm != 2) {
        std::printf("staged launching\n");
        std::fflush(stdout);
        mix_expert_staged_launch(blocks, stream, sync_base, x, weight,
                                 partial, (float*)scale, (float*)token,
                                 staged_out, tiling, rows, (float*)hidden,
                                 (float*)maxima);
        Check(aclrtSynchronizeStream(stream), "staged once");
        std::printf("staged returned\n");
        std::fflush(stdout);
    }
    if (arm != 1) {
        std::printf("fused launching\n");
        std::fflush(stdout);
        mix_expert_fused_launch(blocks, stream, sync_base, x, weight,
                                partial, (float*)scale, (float*)token,
                                fused_out, tiling, rows, stages, middle, down,
                                (float*)down_scale, wide, narrow_out,
                                down_tiling, (float*)middle_scale,
                                (float*)hidden, (float*)maxima);
        Check(aclrtSynchronizeStream(stream), "fused once");
        std::printf("fused returned\n");
        std::fflush(stdout);
    }
    if (arm != 0) {
        Check(aclrtDestroyStream(stream), "destroy");
        Check(aclrtResetDevice(device), "reset");
        Check(aclFinalize(), "finalize");
        return 0;
    }

    int64_t untouched = 0;
    int64_t untouched_rows = 0;
    std::vector<uint16_t> fused_host(rows * HALF);
    std::vector<uint16_t> staged_host(rows * HALF);
    const size_t out_bytes = rows * HALF * sizeof(uint16_t);
    Check(aclrtMemcpy(fused_host.data(), out_bytes, fused_out, out_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "download fused");
    Check(aclrtMemcpy(staged_host.data(), out_bytes, staged_out, out_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "download staged");

    if (stages & 64) {
        std::vector<int32_t> probe(64, -1);
        Check(aclrtMemcpy(probe.data(), 64 * sizeof(int32_t), partial,
                          64 * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
              "download probe");
        std::printf("launched with %u blocks\n", blocks);
        for (int half = 0; half < 2; ++half) {
            std::printf("  %s GetBlockNum %d, per-block ",
                        half ? "vector" : "cube", probe[half * 32]);
            for (int slot = 1; slot < 20; ++slot) {
                if (probe[half * 32 + slot] >= 0) {
                    std::printf("[%d]=%d ", slot - 1,
                                probe[half * 32 + slot]);
                }
            }
            std::printf("\n");
        }
        return 0;
    }

    // Check the INT32 product itself before the activation reads it: a
    // matmul that lands in the wrong place and a vector stage that reads the
    // wrong place look identical in the output.
    std::vector<int32_t> product_host(rows * GATE_UP);
    Check(aclrtMemcpy(product_host.data(), rows * GATE_UP * sizeof(int32_t),
                      partial, rows * GATE_UP * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST), "download partial");
    int64_t wrong_product = 0;
    int64_t first_wrong_row = -1;
    int64_t first_wrong_column = -1;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < GATE_UP; ++column) {
            int32_t want = 0;
            for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                want += x_host[row * HIDDEN + inner]
                        * weight_host[column * HIDDEN + inner];
            }
            if (product_host[row * GATE_UP + column] != want) {
                if (first_wrong_row < 0) {
                    first_wrong_row = row;
                    first_wrong_column = column;
                }
                ++wrong_product;
            }
        }
    }
    std::printf("INT32 product: %lld of %lld differ, first at row %lld "
                "column %lld\n", (long long)wrong_product,
                (long long)(rows * GATE_UP), (long long)first_wrong_row,
                (long long)first_wrong_column);

    int64_t wrong_reference = 0;
    int64_t wrong_arms = 0;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < HALF; ++column) {
            int32_t gate_product = 0;
            int32_t up_product = 0;
            for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                const int32_t activation = x_host[row * HIDDEN + inner];
                gate_product += activation
                                * weight_host[column * HIDDEN + inner];
                up_product += activation
                              * weight_host[(HALF + column) * HIDDEN + inner];
            }
            const float gate = static_cast<float>(gate_product)
                               * scale_host[column] * token_host[row];
            const float up = static_cast<float>(up_product)
                             * scale_host[HALF + column] * token_host[row];
            const float want = gate / (1.0F + std::exp(-gate)) * up;
            const uint16_t rounded = ToBf16(want);
            const int64_t index = row * HALF + column;
            // SwiGLU is a transcendental, so one unit in the last place of
            // BF16 is agreement; the INT32 product before it is exact.
            const int32_t slack = static_cast<int32_t>(fused_host[index])
                                  - static_cast<int32_t>(rounded);
            const bool off = (slack > 1 || slack < -1);
            if (off && wrong_reference < 6) {
                std::printf("  wrong at row %lld column %lld: got %u "
                            "want %u\n", (long long)row, (long long)column,
                            fused_host[index], rounded);
            }
            wrong_reference += off;
            wrong_arms += (fused_host[index] != staged_host[index]);
        }
    }
    for (int64_t row = 0; row < rows; ++row) {
        int64_t row_untouched = 0;
        for (int64_t column = 0; column < HALF; ++column) {
            row_untouched += (fused_host[row * HALF + column] == SENTINEL);
        }
        untouched += row_untouched;
        untouched_rows += (row_untouched == HALF);
    }
    if (stages & 128) {
        // The whole chain on the host: two projections with a SwiGLU and a
        // static-scale quantisation between them.
        std::vector<int8_t> code(rows * HALF);
        std::vector<float> activation(rows * HALF);
        std::vector<float> row_scale(rows, 0.0F);
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t column = 0; column < HALF; ++column) {
                int32_t gate_product = 0;
                int32_t up_product = 0;
                for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                    const int32_t activation = x_host[row * HIDDEN + inner];
                    gate_product += activation
                                    * weight_host[column * HIDDEN + inner];
                    up_product += activation
                                  * weight_host[(HALF + column) * HIDDEN
                                                + inner];
                }
                const float gate = static_cast<float>(gate_product)
                                   * scale_host[column] * token_host[row];
                const float up = static_cast<float>(up_product)
                                 * scale_host[HALF + column]
                                 * token_host[row];
                const float value = gate / (1.0F + std::exp(-gate)) * up;
                activation[row * HALF + column] = value;
                const float magnitude = std::fabs(value);
                row_scale[row] = (magnitude > row_scale[row]) ? magnitude
                                                              : row_scale[row];
            }
        }
        for (int64_t row = 0; row < rows; ++row) {
            row_scale[row] /= 127.0F;
            for (int64_t column = 0; column < HALF; ++column) {
                const float scaled = activation[row * HALF + column]
                                     / row_scale[row];
                code[row * HALF + column] =
                    static_cast<int8_t>(std::nearbyintf(ToHalf(scaled)));
            }
        }
        std::vector<float> scale_device(32, -1.0F);
        Check(aclrtMemcpy(scale_device.data(), 32 * sizeof(float),
                          middle_scale, 32 * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST), "download scale");
        std::vector<float> maxima_device(32 * 32 * 8, -1.0F);
        Check(aclrtMemcpy(maxima_device.data(), 32 * 32 * 8 * sizeof(float),
                          maxima, 32 * 32 * 8 * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST), "download maxima");
        std::vector<int8_t> middle_device(rows * HALF, 0);
        Check(aclrtMemcpy(middle_device.data(), rows * HALF, middle,
                          rows * HALF, ACL_MEMCPY_DEVICE_TO_HOST),
              "download middle");
        int64_t middle_nonzero = 0;
        for (int8_t one : middle_device) {
            middle_nonzero += (one != 0);
        }
        std::printf("quantised intermediate: %lld of %lld nonzero\n",
                    (long long)middle_nonzero, (long long)(rows * HALF));
        std::printf("row scale, want then got:");
        for (int64_t row = 0; row < (rows < 4 ? rows : 4); ++row) {
            std::printf("  %.6g/%.6g", row_scale[row], scale_device[row]);
        }
        std::printf("\n  block maxima row 0:");
        for (uint32_t block = 0; block < blocks; ++block) {
            std::printf(" %.6g", maxima_device[block * 32 * 8]);
        }
        std::printf("\n");

        std::vector<uint16_t> chain_host(rows * HIDDEN);
        Check(aclrtMemcpy(chain_host.data(),
                          rows * HIDDEN * sizeof(uint16_t), narrow_out,
                          rows * HIDDEN * sizeof(uint16_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), "download chain");
        int64_t wrong_chain = 0;
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t column = 0; column < HIDDEN; ++column) {
                int32_t product = 0;
                for (int64_t inner = 0; inner < HALF; ++inner) {
                    product += code[row * HALF + inner]
                               * down_host[column * HALF + inner];
                }
                const float want = static_cast<float>(product)
                                   * down_scale_host[column]
                                   * row_scale[row];
                const int32_t slack =
                    static_cast<int32_t>(chain_host[row * HIDDEN + column])
                    - static_cast<int32_t>(ToBf16(want));
                wrong_chain += (slack > 1 || slack < -1);
            }
        }
        std::printf("chain against host reference, 1 ulp allowed: %lld of "
                    "%lld\n", (long long)wrong_chain,
                    (long long)(rows * HIDDEN));
    }

    std::printf("rows %lld  blocks %u  tiling %d  untouched %lld of %lld "
                "in %lld whole rows\n", (long long)rows, blocks,
                tiling_status, (long long)untouched,
                (long long)(rows * HALF), (long long)untouched_rows);
    std::printf("fused against host reference, 1 ulp allowed: %lld of %lld\n",
                (long long)wrong_reference, (long long)(rows * HALF));
    std::printf("fused against staged arm, exact:             %lld of %lld\n",
                (long long)wrong_arms, (long long)(rows * HALF));

    for (int round = 0; round < rounds; ++round) {
        timespec begin;
        timespec end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (int step = 0; step < repeats; ++step) {
            mix_expert_fused_launch(blocks, stream, sync_base, x, weight,
                                    partial, (float*)scale, (float*)token,
                                    fused_out, tiling, rows, stages, middle, down,
                                (float*)down_scale, wide, narrow_out,
                                down_tiling, (float*)middle_scale,
                                (float*)hidden, (float*)maxima);
        }
        Check(aclrtSynchronizeStream(stream), "fused timing");
        clock_gettime(CLOCK_MONOTONIC, &end);
        const double fused = Elapsed(begin, end) / repeats;

        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (int step = 0; step < repeats; ++step) {
            mix_expert_staged_launch(blocks, stream, sync_base, x, weight,
                                     partial, (float*)scale, (float*)token,
                                     staged_out, tiling, rows, (float*)hidden,
                                 (float*)maxima);
        }
        Check(aclrtSynchronizeStream(stream), "staged timing");
        clock_gettime(CLOCK_MONOTONIC, &end);
        const double staged = Elapsed(begin, end) / repeats;

        std::printf("round %d  fused %8.3f us  staged %8.3f us  saved "
                    "%7.3f  %.2fx\n", round, fused, staged, staged - fused,
                    staged / fused);
    }

    Check(aclrtDestroyStream(stream), "destroy");
    Check(aclrtResetDevice(device), "reset");
    Check(aclFinalize(), "finalize");
    return (wrong_reference == 0 && wrong_arms == 0) ? 0 : 1;
}
