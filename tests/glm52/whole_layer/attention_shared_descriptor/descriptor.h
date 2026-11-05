#ifndef TILERT_SHARED_ATTENTION_DESCRIPTOR_H
#define TILERT_SHARED_ATTENTION_DESCRIPTOR_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace shared_attention_arguments {

template <class Address>
struct Attention {
    Address token_x, token_scale, weight_dq, weight_uq_qr, weight_uk;
    Address weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr;
    Address gamma_cq, gamma_ckv, sin, cos, slots;
    Address kv_cache, rope_cache, topk, query_lengths, kv_lengths, page_table;
    Address scratch, output;
};

template <class Address>
struct Full {
    Attention<Address> attention;
    Address value_weight, output_weight, output_scale, bias;
    Address scratch, tail_tiling, output;
};

template <class Address>
inline Full<Address> Capture(
    Address token_x, Address token_scale, Address weight_dq, Address weight_uq_qr,
    Address weight_uk, Address weight_dkv_kr, Address scale_dq, Address scale_uq_qr,
    Address scale_dkv_kr, Address gamma_cq, Address gamma_ckv, Address sin,
    Address cos, Address slots, Address kv_cache, Address rope_cache, Address topk,
    Address query_lengths, Address kv_lengths, Address page_table,
    Address value_weight, Address output_weight, Address output_scale,
    Address bias, Address scratch, Address tail_tiling, Address output,
    uint64_t arena_offset)
{
    return {{token_x, token_scale, weight_dq, weight_uq_qr, weight_uk,
        weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr, gamma_cq, gamma_ckv,
        sin, cos, slots, kv_cache, rope_cache, topk, query_lengths, kv_lengths,
        page_table, scratch + arena_offset, scratch}, value_weight, output_weight,
        output_scale, bias, scratch, tail_tiling, output};
}

using Portable = Full<uint64_t>;
static_assert(std::is_standard_layout<Portable>::value);
static_assert(std::is_trivially_copyable<Portable>::value);
static_assert(sizeof(Attention<uint64_t>) == 176);
static_assert(sizeof(Portable) == 232);
static_assert(offsetof(Portable, value_weight) == 176);
static_assert(offsetof(Portable, scratch) == 208);
static_assert(offsetof(Portable, output) == 224);

}  // namespace shared_attention_arguments

#endif
