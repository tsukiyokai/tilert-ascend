#ifndef TILERT_FFN_GU_READY_CONTRACT_H
#define TILERT_FFN_GU_READY_CONTRACT_H

#include <cstdint>

namespace ffn_gu_ready {
constexpr uint16_t kReadyFlag = 8;
constexpr uint16_t kFreeFlag = 9;
constexpr int64_t kCredits = 4;

constexpr bool UseGateOverlap(int64_t active_experts, int64_t blocks)
{
    return active_experts >= blocks;
}
}  // namespace ffn_gu_ready
#endif
