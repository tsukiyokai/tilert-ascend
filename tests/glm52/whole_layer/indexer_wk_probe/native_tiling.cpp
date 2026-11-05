#include <array>
#include <cstdint>
#include <cstring>

#include "lib/matmul/matmul_tiling.h"

extern "C" int indexer_front_build_tiling(int capacity, void* buffer);

extern "C" int wk_geometry_tiling(int32_t columns, void* buffer, int64_t* fields)
{
    if (columns != 32 && columns != 160) {
        return -1;
    }
    AscendC::tiling::TCubeTiling cube{};
    static_assert(sizeof(cube) <= 512);
    if (columns == 32) {
        std::array<uint8_t, 1536> original{};
        const int status = indexer_front_build_tiling(32, original.data());
        if (status != 0) {
            return status;
        }
        std::memcpy(&cube, original.data() + 512, sizeof(cube));
    } else {
        using namespace matmul_tiling;
        MatmulApiTiling tiling;
        tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
        tiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
        tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        tiling.SetShape(32, columns, 6144);
        tiling.SetOrgShape(32, columns, 6144);
        tiling.SetBias(false);
        tiling.SetBufferSpace(-1, -1, -1);
        const int status = tiling.GetTiling(cube);
        if (status != 0) {
            return status;
        }
    }
    std::memset(buffer, 0, 512);
    std::memcpy(buffer, &cube, sizeof(cube));
    const int64_t values[] = {
        cube.M, cube.N, cube.Ka, cube.Kb, cube.singleCoreM, cube.singleCoreN,
        cube.singleCoreK, cube.baseM, cube.baseN, cube.baseK, cube.depthA1,
        cube.depthB1, cube.stepM, cube.stepN, cube.stepKa, cube.stepKb,
        cube.iterateOrder, cube.shareL1Size, cube.shareL0CSize, cube.shareUbSize,
        cube.dbL0A, cube.dbL0B, cube.dbL0C};
    std::memcpy(fields, values, sizeof(values));
    return 0;
}
