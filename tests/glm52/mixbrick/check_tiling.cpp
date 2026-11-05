/**
 * Can a standalone program get a cube tiling, without the operator framework?
 *
 * A hand-written MIX kernel needs a `TCubeTiling` and there is no way around
 * it: the Matmul library reads one in `Init`. The repo's existing fused kernel
 * allocates 4096 bytes for it and never writes them, so its matmul ran on
 * uninitialised tiling -- which is why this check exists before the next
 * kernel is written rather than after.
 *
 * Two traps. `GetTiling(optiling::TCubeTiling&)` drags in the operator
 * registry's `TilingDef`, so take the `AscendC::tiling::TCubeTiling` overload
 * and the plain struct it fills. And the link line is not discoverable from
 * the headers: `tiling_api` needs `register` for those same symbols,
 * `unified_dlog` for `CheckLogLevel` and `c_sec` for `memcpy_s`, and the
 * program must run under `set_env.sh` or `libruntime_common` fails to resolve
 * `ErrorManager`.
 */
#include <cstdint>
#include <cstdio>

#include "lib/matmul/matmul_tiling.h"

namespace {

constexpr int32_t ROWS = 8;
constexpr int32_t GATE_UP = 256;
constexpr int32_t HIDDEN = 6144;

}  // namespace

int main(int argc, char** argv)
{
    const int32_t rows = (argc > 1) ? std::atoi(argv[1]) : ROWS;
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT8, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT8, true);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT32);
    tiling.SetShape(rows, GATE_UP, HIDDEN);
    tiling.SetOrgShape(rows, GATE_UP, HIDDEN);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling cube;
    const int64_t status = tiling.GetTiling(cube);
    std::printf("rows %d  status %lld  struct %zu bytes\n", rows,
                (long long)status, sizeof(cube));
    std::printf("  base  M %d  N %d  K %d\n", cube.baseM, cube.baseN,
                cube.baseK);
    std::printf("  step  M %d  N %d  depthA1 %d  depthB1 %d\n", cube.stepM,
                cube.stepN, cube.depthA1, cube.depthB1);
    std::printf("  single M %d  N %d  K %d\n", cube.singleCoreM,
                cube.singleCoreN, cube.singleCoreK);
    return status == 0 ? 0 : 1;
}
