#include <cstring>

#include "lib/matmul/matmul_tiling.h"
#include "contract.h"

extern "C" int ffn_descriptor_bridge_build_tiling(int capacity, void* buffer)
{
    using namespace matmul_tiling;
    if (capacity != 16 && capacity != 32) {
        return -1;
    }
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tiling.SetShape(capacity, ffn_bridge::kGateTile, ffn_bridge::kHidden);
    tiling.SetOrgShape(capacity, ffn_bridge::kExperts, ffn_bridge::kHidden);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);
    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= ffn_bridge::kTilingBytes);
    const int status = tiling.GetTiling(cube);
    if (status == 0) {
        std::memset(buffer, 0, ffn_bridge::kTilingBytes);
        std::memcpy(buffer, &cube, sizeof(cube));
    }
    return status;
}
