#ifndef TESTS_GLM52_ATTENTION_INDEXED_SHADOW_H
#define TESTS_GLM52_ATTENTION_INDEXED_SHADOW_H

#include "indexer_front_native.h"

namespace whole_layer_indexed {

struct ShadowRows {
    GM_ADDR latent;
    GM_ADDR rope;
    GM_ADDR slots;
    GM_ADDR packed;

    __aicore__ inline void Run(AscendC::TPipe& pipe,
                               const indexer_front::Config& config) const
    {
        using namespace AscendC;
        GlobalTensor<bfloat16_t> ckv, kr, shadow;
        GlobalTensor<int64_t> slot_ids;
        ckv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(latent));
        kr.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rope));
        shadow.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(packed));
        slot_ids.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
        TBuf<TPosition::VECCALC> storage;
        pipe.InitBuffer(storage, 576 * sizeof(bfloat16_t));
        auto row = storage.Get<bfloat16_t>();
        for (int64_t token = GetBlockIdx(); token < config.rows;
             token += 2 * config.blocks) {
            const int64_t slot = slot_ids.GetValue(token);
            if (slot < 0) {
                continue;
            }
            DataCopy(row, ckv[slot * 512], 512);
            DataCopy(row[512], kr[slot * 64], 64);
            indexer_front::device::Fence<HardEvent::MTE2_MTE3>();
            DataCopy(shadow[slot * 576], row, 576);
            indexer_front::device::Fence<HardEvent::MTE3_MTE2>();
        }
    }
};

}  // namespace whole_layer_indexed

#endif
