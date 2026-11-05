/** Inspect installed host tilings; no device or Torch runtime is initialized. */
#include <array>
#include <cstdint>
#include <iostream>
#include <iomanip>

#include "lib/matmul/matmul_tiling.h"

struct Shape {
    const char* name;
    int rows;
    int base_m;
    int base_n;
    int base_k;
};

int main()
{
    constexpr std::array<Shape, 3> cases{{
        {"accepted_m64_n512_k128", 64, 64, 512, 128},
        {"m32_n1024_k64", 64, 32, 1024, 64},
        {"m32_n1024_k128_request", 64, 32, 1024, 128},
    }};
    for (int layout : {0, 1}) {
        for (const auto& shape : cases) {
            matmul_tiling::MatmulApiTiling tiling;
            tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_INT4, false);
            tiling.SetBType(matmul_tiling::TPosition::GM,
                            layout ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_INT4, false);
            tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_INT32);
            tiling.SetShape(shape.rows, 6144, 128);
            tiling.SetOrgShape(shape.rows, 6144, 128);
            tiling.SetBias(false);
            tiling.SetBufferSpace(-1, -1, -1);
            int fixed_status = 0;
            if (shape.base_m >= 0 || shape.base_n >= 0 || shape.base_k >= 0) {
                fixed_status = tiling.SetFixSplit(shape.base_m, shape.base_n, shape.base_k);
            }
            AscendC::tiling::TCubeTiling cube{};
            const int64_t status = tiling.GetTiling(cube);
            std::cout << "{\"case\":\"" << shape.name << "\",\"layout\":" << layout
                      << ",\"shape_m\":" << shape.rows
                      << ",\"fix_status\":" << fixed_status
                      << ",\"status\":" << status
                      << ",\"tiling_bytes\":" << sizeof(cube);
#define FIELD(name) std::cout << ",\"" #name "\":" << cube.name
            FIELD(M); FIELD(N); FIELD(Ka); FIELD(Kb);
            FIELD(singleCoreM); FIELD(singleCoreN); FIELD(singleCoreK);
            FIELD(baseM); FIELD(baseN); FIELD(baseK);
            FIELD(depthA1); FIELD(depthB1);
            FIELD(stepM); FIELD(stepN); FIELD(stepKa); FIELD(stepKb);
            FIELD(iterateOrder); FIELD(shareL1Size); FIELD(shareL0CSize); FIELD(shareUbSize);
            FIELD(dbL0A); FIELD(dbL0B); FIELD(dbL0C);
#undef FIELD
            std::cout << ",\"raw_hex\":\"";
            const auto* bytes = reinterpret_cast<const unsigned char*>(&cube);
            for (size_t i = 0; i < sizeof(cube); ++i) {
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<unsigned>(bytes[i]);
            }
            std::cout << std::dec << "\"}\n";
        }
    }
}
