// Compile this separately against each variant's frozen contract.h.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <contract.h>

#if defined(VARIANT_ACTIVE)
namespace contract = ffn_active;
#define VARIANT_LAUNCH ffn_active_launch
#else
namespace contract = ffn_profile;
#define VARIANT_LAUNCH ffn_profile_launch
#endif

namespace {
std::vector<uint64_t> recorded;
uint64_t Address(const void* value) { return reinterpret_cast<uint64_t>(value); }
void Pointers(void** values, size_t count)
{
    for (size_t i = 0; i < count; ++i) recorded.push_back(Address(values[i]));
}
}

extern "C" uint64_t variant_count() { return recorded.size(); }
extern "C" uint64_t variant_value(uint64_t index) { return recorded.at(index); }
extern "C" uint64_t variant_body_count() { return contract::kBodyBuffers; }
extern "C" uint64_t variant_dispatch_size() { return sizeof(routed_int4::DispatchInput); }
extern "C" uint64_t variant_shared_size() { return sizeof(routed_int4::pipeline::Input); }
extern "C" uint64_t variant_shared_offset(int index)
{
    using S = routed_int4::pipeline::Input;
    constexpr size_t fields[] = {offsetof(S, input), offsetof(S, input_scale),
        offsetof(S, gate_weight), offsetof(S, gate_scale), offsetof(S, down_weight),
        offsetof(S, down_scale), offsetof(S, payload), offsetof(S, tiling),
        offsetof(S, output), offsetof(S, capacity), offsetof(S, partial_offset),
        offsetof(S, middle_offset), offsetof(S, scale_offset), offsetof(S, down_offset)};
    return fields[index];
}

extern "C" void VARIANT_LAUNCH(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** weights, void** bridge, void** body, routed_int4::pipeline::Input shared,
    int64_t rows, int64_t blocks, int64_t layout, float epsilon
#if !defined(VARIANT_ACTIVE)
    , void* trace
#endif
    )
{
    recorded = {Address(stream), sync, Address(input), Address(residual),
                Address(gamma), Address(gate), Address(bias)};
    Pointers(weights, contract::kRoutedWeights);
    Pointers(bridge, contract::kBridgeBuffers);
    Pointers(body, contract::kBodyBuffers);
    const uint64_t fields[] = {shared.input, shared.input_scale, shared.gate_weight,
        shared.gate_scale, shared.down_weight, shared.down_scale, shared.payload,
        shared.tiling, shared.output, static_cast<uint64_t>(shared.capacity),
        shared.partial_offset, shared.middle_offset, shared.scale_offset, shared.down_offset};
    recorded.insert(recorded.end(), std::begin(fields), std::end(fields));
    uint32_t epsilon_bits;
    std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon_bits));
    recorded.insert(recorded.end(), {uint64_t(rows), uint64_t(blocks),
                                    uint64_t(layout), epsilon_bits});
#if !defined(VARIANT_ACTIVE)
    recorded.push_back(Address(trace));
#endif
}
