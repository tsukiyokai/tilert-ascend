#ifndef TILERT_FFN_DOWN_N512_TILING_CONTRACT_H
#define TILERT_FFN_DOWN_N512_TILING_CONTRACT_H

namespace ffn_down_n512 {
template <class Cube>
bool IsExpectedDownTiling(const Cube& cube)
{
    return cube.M == 64 && cube.N == 6144 && cube.Ka == 128 && cube.Kb == 128 &&
        cube.singleCoreM == 64 && cube.singleCoreN == 6144 && cube.singleCoreK == 128 &&
        cube.baseM == 64 && cube.baseN == 512 && cube.baseK == 128 &&
        cube.depthA1 == 1 && cube.depthB1 == 12 &&
        cube.stepM == 1 && cube.stepN == 12 && cube.stepKa == 1 && cube.stepKb == 1 &&
        cube.iterateOrder == 0 && cube.shareL1Size == 397312 &&
        cube.shareL0CSize == 131072 && cube.shareUbSize == 0 &&
        cube.dbL0A == 2 && cube.dbL0B == 2 && cube.dbL0C == 1;
}
}
#endif
