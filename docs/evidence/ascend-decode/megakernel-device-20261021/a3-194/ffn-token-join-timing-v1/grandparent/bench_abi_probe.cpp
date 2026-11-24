// CPU ABI recorder: declarations come from the actual frozen device build.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "contract.h"
#define ffn_composed_unused_body_launch routed_shared_body_launch
#include "../body/source/routed_int4/routed_int4_kernel.h"

namespace {
std::vector<uint64_t> recorded;

uint64_t Address(const void* value) { return reinterpret_cast<uint64_t>(value); }
void Pointers(void** values, size_t size)
{
    for (size_t i = 0; i < size; ++i) recorded.push_back(Address(values[i]));
}
void Shared(routed_int4::pipeline::Input value)
{
    const uint64_t fields[] = {
        value.input, value.input_scale, value.gate_weight, value.gate_scale,
        value.down_weight, value.down_scale, value.payload, value.tiling,
        value.output, static_cast<uint64_t>(value.capacity), value.partial_offset,
        value.middle_offset, value.scale_offset, value.down_offset};
    recorded.insert(recorded.end(), std::begin(fields), std::end(fields));
}
uint32_t Bits(float value)
{
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
}

extern "C" void probe_clear() { recorded.clear(); }
extern "C" uint64_t probe_count() { return recorded.size(); }
extern "C" uint64_t probe_value(uint64_t index) { return recorded.at(index); }
extern "C" uint64_t probe_size(int kind)
{
    return kind == 0 ? sizeof(routed_int4::DispatchInput)
                     : sizeof(routed_int4::pipeline::Input);
}
extern "C" uint64_t probe_offset(int kind, int index)
{
    using D = routed_int4::DispatchInput;
    using S = routed_int4::pipeline::Input;
    constexpr size_t dispatch[] = {offsetof(D, rows), offsetof(D, scales),
        offsetof(D, inverse), offsetof(D, counts), offsetof(D, probabilities),
        offsetof(D, bf16_rows), offsetof(D, expert_ids)};
    constexpr size_t shared[] = {offsetof(S, input), offsetof(S, input_scale),
        offsetof(S, gate_weight), offsetof(S, gate_scale), offsetof(S, down_weight),
        offsetof(S, down_scale), offsetof(S, payload), offsetof(S, tiling),
        offsetof(S, output), offsetof(S, capacity), offsetof(S, partial_offset),
        offsetof(S, middle_offset), offsetof(S, scale_offset), offsetof(S, down_offset)};
    return kind == 0 ? dispatch[index] : shared[index];
}

extern "C" void ffn_composed_launch(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** weights, void** bridge, void** body, routed_int4::pipeline::Input shared,
    int64_t rows, int64_t blocks, int64_t layout, float epsilon)
{
    recorded = {1, Address(stream), sync, Address(input), Address(residual),
                Address(gamma), Address(gate), Address(bias)};
    Pointers(weights, ffn_composed::kRoutedWeights);
    Pointers(bridge, ffn_composed::kBridgeBuffers);
    Pointers(body, ffn_composed::kBodyBuffers);
    Shared(shared);
    recorded.insert(recorded.end(), {uint64_t(rows), uint64_t(blocks),
                                    uint64_t(layout), Bits(epsilon)});
}

extern "C" void ffn_bridge_launch(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** scratch, int64_t rows, int64_t blocks, float epsilon)
{
    recorded.insert(recorded.end(), {2, Address(stream), sync, Address(input),
        Address(residual), Address(gamma), Address(gate), Address(bias)});
    Pointers(scratch, ffn_bridge::kBuffers);
    recorded.insert(recorded.end(), {uint64_t(rows), uint64_t(blocks), Bits(epsilon)});
}

extern "C" void routed_shared_body_launch(uint32_t blocks, void* stream,
    uint64_t sync, void* rows, void* gate_up, float* gu_bias,
    routed_int4::ScaleStorage* gu_scale, void* down, float* down_bias,
    routed_int4::ScaleStorage* down_scale, float* token_scale,
    void* count, void* start, void* source_row, void* source_slot, float* weight,
    void* stacked, void* partial, void* middle, float* middle_scale, void* wide,
    void* routes, void* output, void* gu_tiling, void* down_tiling, void* workspace,
    int64_t experts, int64_t tokens, int64_t layout,
    routed_int4::DispatchInput dispatch, routed_int4::pipeline::Input shared)
{
    recorded.insert(recorded.end(), {3, blocks, Address(stream), sync});
    void* pointers[] = {rows, gate_up, gu_bias, gu_scale, down, down_bias,
        down_scale, token_scale, count, start, source_row, source_slot, weight,
        stacked, partial, middle, middle_scale, wide, routes, output,
        gu_tiling, down_tiling, workspace};
    Pointers(pointers, 23);
    recorded.insert(recorded.end(), {uint64_t(experts), uint64_t(tokens),
        uint64_t(layout), dispatch.rows, dispatch.scales, dispatch.inverse,
        dispatch.counts, dispatch.probabilities, dispatch.bf16_rows, dispatch.expert_ids});
    Shared(shared);
}

extern "C" void ffn_bridge_epilogue_launch(void* stream, uint64_t sync,
    void* routed, void* shared, void** scratch, int64_t rows, int64_t blocks)
{
    recorded.insert(recorded.end(), {4, Address(stream), sync, Address(routed), Address(shared)});
    Pointers(scratch, ffn_bridge::kBuffers);
    recorded.insert(recorded.end(), {uint64_t(rows), uint64_t(blocks)});
}
