#ifndef TESTS_GLM52_ROUTED_INT4_ACTIVE_LIST_H
#define TESTS_GLM52_ROUTED_INT4_ACTIVE_LIST_H

#include "routed_int4_contract.h"

#if defined(ROUTED_ACTIVE_LIST) && (defined(ROUTED_FIXPIPE_HALF) \
    || defined(ROUTED_GU_MDL) || defined(ROUTED_GU_PAD16) \
    || defined(ROUTED_DOWN_STRIPED) || defined(ROUTED_GU_SPLIT_K) \
    || defined(ROUTED_ENABLE_PROFILE))
#error "Active-list enumeration requires the independent INT32 Norm path"
#endif
#if defined(ROUTED_ACTIVE_LIST) && (!defined(ROUTED_INPUT_FRONT) \
    || !defined(ROUTED_SHARED_PIPELINE) || !defined(ROUTED_ROW_BALANCED))
#error "Active-list enumeration requires the row-balanced input-front pipeline"
#endif

namespace routed_int4 {

// Scratch[20]: count, seven reserved zero words, then ascending expert IDs.
// Unused IDs are zero. Every replay rewrites the complete 32-byte-aligned span.
constexpr int64_t kActiveHeaderElements = 8;
constexpr int64_t kActiveListElements = kActiveHeaderElements + kMaxExperts;
constexpr size_t kActiveListBytes = kActiveListElements * sizeof(int32_t);
static_assert(kActiveListBytes == 1056);
static_assert(kActiveListBytes % 32 == 0);

#if defined(__CCE__)
#define ROUTED_ACTIVE_INLINE __aicore__ inline
#else
#define ROUTED_ACTIVE_INLINE inline
#endif

template <class Storage>
ROUTED_ACTIVE_INLINE int32_t AppendActiveExpert(
    Storage& list, int32_t size, int64_t expert, int64_t count)
{
    if (count != 0) {
        list.SetValue(kActiveHeaderElements + size,
                      static_cast<int32_t>(expert));
        ++size;
    }
    return size;
}

#undef ROUTED_ACTIVE_INLINE

}  // namespace routed_int4

#endif
