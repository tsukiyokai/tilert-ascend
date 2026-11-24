// Compile each arm separately against its own frozen contract.h.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <contract.h>

#ifdef TOKEN_TERMINAL_CANDIDATE
namespace contract = ffn_token_terminal;
#define FFN_ENTRY ffn_token_terminal_launch
#else
namespace contract = ffn_active_token_join;
#define FFN_ENTRY ffn_active_token_join_launch
#endif

namespace {
std::vector<uint64_t> recorded;
uint64_t Address(const void* value) { return reinterpret_cast<uint64_t>(value); }
void Pointers(void** values, size_t size)
{
    for (size_t i = 0; i < size; ++i) recorded.push_back(Address(values[i]));
}
}

extern "C" uint64_t probe_count() { return recorded.size(); }
extern "C" uint64_t probe_value(uint64_t index) { return recorded.at(index); }
extern "C" uint64_t probe_shared_size()
{
    return sizeof(routed_int4::pipeline::Input);
}
extern "C" uint64_t probe_shared_offset(int index)
{
    using S = routed_int4::pipeline::Input;
    constexpr size_t offsets[] = {
        offsetof(S, input), offsetof(S, input_scale), offsetof(S, gate_weight),
        offsetof(S, gate_scale), offsetof(S, down_weight), offsetof(S, down_scale),
        offsetof(S, payload), offsetof(S, tiling), offsetof(S, output),
        offsetof(S, capacity), offsetof(S, partial_offset), offsetof(S, middle_offset),
        offsetof(S, scale_offset), offsetof(S, down_offset)};
    return offsets[index];
}

extern "C" void FFN_ENTRY(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** weights, void** bridge, void** body, routed_int4::pipeline::Input shared,
    int64_t rows, int64_t blocks, int64_t layout, float epsilon)
{
    recorded = {Address(stream), sync, Address(input), Address(residual),
                Address(gamma), Address(gate), Address(bias)};
    Pointers(weights, contract::kRoutedWeights);
    Pointers(bridge, contract::kBridgeBuffers);
    Pointers(body, contract::kBodyBuffers);
    const uint64_t values[] = {
        shared.input, shared.input_scale, shared.gate_weight, shared.gate_scale,
        shared.down_weight, shared.down_scale, shared.payload, shared.tiling,
        shared.output, static_cast<uint64_t>(shared.capacity), shared.partial_offset,
        shared.middle_offset, shared.scale_offset, shared.down_offset};
    recorded.insert(recorded.end(), std::begin(values), std::end(values));
    uint32_t bits;
    std::memcpy(&bits, &epsilon, sizeof(bits));
    recorded.insert(recorded.end(), {uint64_t(rows), uint64_t(blocks),
                                    uint64_t(layout), bits});
}
