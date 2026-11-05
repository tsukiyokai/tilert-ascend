#include "routed_int4_pipeline.h"

#include <cstring>
#include "lib/matmul/matmul_tiling.h"

namespace {
int BuildSharedProjection(int rows, int columns, int inner, void* destination)
{
    using namespace matmul_tiling;
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    tiling.SetShape(rows, columns, inner);
    tiling.SetOrgShape(rows, columns, inner);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);
    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= shared_w8::kTilingStride);
    const int status = tiling.GetTiling(cube);
    if (status == 0) {
        std::memcpy(destination, &cube, sizeof(cube));
    }
    return status;
}
}  // namespace

extern "C" int routed_shared_build_tiling(int capacity, void* destination)
{
    if ((capacity != 16 && capacity != 32) || destination == nullptr) {
        return -1;
    }
    std::memset(destination, 0, 2 * shared_w8::kTilingStride);
    const int status = BuildSharedProjection(capacity, shared_w8::kGateTile,
        shared_w8::kInnerTile, destination);
    if (status != 0) {
        return status;
    }
    return BuildSharedProjection(capacity, shared_w8::kDownTile,
        shared_w8::kIntermediate,
        static_cast<uint8_t*>(destination) + shared_w8::kTilingStride);
}
