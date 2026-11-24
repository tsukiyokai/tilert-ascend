#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "lib/matmul/matmul_tiling.h"

using Cube = AscendC::tiling::TCubeTiling;
static_assert(sizeof(Cube) == 200);

void Print(const char* label, int layout, const Cube& cube)
{
    std::cout << "{\"label\":\"" << label << "\",\"layout\":" << layout
              << ",\"raw_hex\":\"";
    const auto* bytes = reinterpret_cast<const uint8_t*>(&cube);
    for (size_t index = 0; index < sizeof(cube); ++index) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<unsigned>(bytes[index]);
    }
    std::cout << std::dec << "\"}\n";
}

int main()
{
    for (int layout : {0, 1}) {
        matmul_tiling::MatmulApiTiling tiling;
        tiling.SetAType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_INT4, false);
        tiling.SetBType(matmul_tiling::TPosition::GM,
                       layout ? matmul_tiling::CubeFormat::NZ
                              : matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_INT4, false);
        tiling.SetCType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_INT32);
        tiling.SetShape(64, 6144, 128);
        tiling.SetOrgShape(64, 6144, 128);
        tiling.SetBias(false);
        tiling.SetBufferSpace(-1, -1, -1);
        if (tiling.SetFixSplit(32, 1024, 64) != 0) {
            return 2;
        }
        Cube cube{};
        if (tiling.GetTiling(cube) != 0) {
            return 3;
        }
        Print("parent", layout, cube);
        cube.baseK = 128;
        cube.depthA1 = 2;
        cube.depthB1 = 6;
        cube.stepKa = 1;
        cube.stepKb = 1;
        cube.dbL0B = 1;
        Print("proposed", layout, cube);
    }
    std::cout << "{\"field_offsets\":{"
              << "\"baseK\":" << offsetof(Cube, baseK)
              << ",\"depthA1\":" << offsetof(Cube, depthA1)
              << ",\"depthB1\":" << offsetof(Cube, depthB1)
              << ",\"stepKa\":" << offsetof(Cube, stepKa)
              << ",\"stepKb\":" << offsetof(Cube, stepKb)
              << ",\"dbL0B\":" << offsetof(Cube, dbL0B)
              << ",\"transLength\":" << offsetof(Cube, transLength)
              << "}}\n";
}
