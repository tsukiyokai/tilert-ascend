#ifndef TILERT_FFN_DOWN_OVERLAP_NATIVE_H
#define TILERT_FFN_DOWN_OVERLAP_NATIVE_H

#if defined(ROUTED_FIXPIPE_HALF) || defined(ROUTED_ENABLE_PROFILE)
#error "Down overlap requires the uninstrumented INT32 parent"
#endif

// Included after StoreRoutes, inside the frozen routed-body helper namespace.
template <class MatmulType>
__aicore__ inline void ProjectDownOverlap(
    MatmulType& mm, __gm__ int4b_t* rows, __gm__ int4b_t* weight,
    __gm__ RoutedProduct* product, int64_t columns, int64_t inner,
    int64_t count, int64_t& sequence)
{
    const int64_t matrix_rows = 2 * count;
    GlobalTensor<int4b_t> a, b;
    GlobalTensor<RoutedProduct> c;
    a.SetGlobalBuffer(rows, matrix_rows * inner);
    b.SetGlobalBuffer(weight, inner * columns);
    c.SetGlobalBuffer(product, matrix_rows * columns);
    mm.SetOrgShape(matrix_rows, columns, inner);
    mm.SetSingleShape(matrix_rows, columns, inner);
    mm.SetTensorA(a);
    mm.SetTensorB(b, false);
    while (mm.Iterate()) {
        if (sequence >= ffn_down_overlap::kDownCredits) {
            CrossCoreWaitFlag(ffn_down_overlap::kDownFreeFlag);
        }
        // Preserve the original full-GM row stride and current M/N tile offset.
        mm.GetTensorC(c, 0, false);
        CrossCoreSetFlag<2, PIPE_FIX>(ffn_down_overlap::kDownReadyFlag);
        ++sequence;
    }
    mm.End();
}

__aicore__ inline void DrainDownCredits(int64_t sequence)
{
    const int64_t remaining = sequence < ffn_down_overlap::kDownCredits
        ? sequence : ffn_down_overlap::kDownCredits;
    for (int64_t credit = 0; credit < remaining; ++credit) {
        CrossCoreWaitFlag(ffn_down_overlap::kDownFreeFlag);
    }
}

__aicore__ inline void ConsumeDown(
    Workspace& held, __gm__ RoutedProduct* wide, __gm__ float* correction,
    __gm__ ScaleStorage* scales, __gm__ float* middle_scales,
    __gm__ bfloat16_t* routes, int64_t experts, int64_t tokens, int64_t blocks)
{
    int64_t active = 0;
    int64_t next = GetBlockIdx() / 2;
    const int64_t half = GetSubBlockIdx() * ffn_down_overlap::kDownHalfTile;
    for (int64_t expert = 0; expert < experts; ++expert) {
        const int64_t count = held.count.GetValue(expert);
        if (count == 0) {
            continue;
        }
        if (active++ != next) {
            continue;
        }
        next += blocks;
        const int64_t begin = held.start.GetValue(expert);
        for (int64_t tile = 0; tile < ffn_down_overlap::kDownTilesPerExpert; ++tile) {
            CrossCoreWaitFlag(ffn_down_overlap::kDownReadyFlag);
            const int64_t column = tile * ffn_down_overlap::kDownTile + half;
            StoreRoutes(held, count, begin, wide + 2 * begin * kHidden,
                correction + expert * kHidden, scales + expert * kHidden,
                middle_scales + begin, routes, column,
                column + ffn_down_overlap::kDownHalfTile, tokens);
            // StoreRoutes drains each route store before reusing its UB source.
            CrossCoreSetFlag<2, PIPE_MTE3>(ffn_down_overlap::kDownFreeFlag);
        }
    }
}

#endif
