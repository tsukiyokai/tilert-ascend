#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "attention_indexed_contract.h"

struct Region {
    uint64_t begin;
    uint64_t size;
};

void Check(uint32_t rows, uint32_t blocks, uint32_t table_pages, bool quantized)
{
    const auto attention = whole_layer_attention::MakeConfig(
        rows, table_pages + 3, table_pages, blocks, .0625F, 1e-5F, 1e-5F);
    const auto c = whole_layer_indexed::MakeConfig(
        attention, table_pages + 6, table_pages + 1, quantized);
    const auto bf16_scale = whole_layer_indexed::MakeConfig(
        attention, table_pages + 6, table_pages + 1, quantized, true);
    assert(!c.front.query_scale_bf16 && bf16_scale.front.query_scale_bf16);
    assert(c.bytes == bf16_scale.bytes);
    assert(c.front_offset == bf16_scale.front_offset);
    assert(c.tail_offset == bf16_scale.tail_offset);
    const uint64_t r = rows;
    const uint64_t encoded_size = quantized ? 1 : 2;
    const std::vector<Region> regions{
        {c.query_norm, r * 2048}, {c.query_norm_scale, r * 4},
        {c.query, r * 4096 * encoded_size},
        {c.key, r * 128 * encoded_size}, {c.weights, r * 32 * 2},
        {c.query_scale, r * 32 * 2}, {c.key_scale, r * 2},
        {c.topk, r * 2048 * 4}, {c.latent, r * 4 * 512 * 2},
        {c.attention_offset, c.attention.bytes},
        {c.front_offset, c.front.bytes},
        {c.indexer_offset, c.indexer.bytes},
        {c.tail_offset, c.tail.bytes},
    };
    for (size_t index = 0; index < regions.size(); ++index) {
        const auto& region = regions[index];
        assert(region.begin % 512 == 0);
        assert(region.begin + region.size <= c.bytes);
        for (size_t other = 0; other < index; ++other) {
            assert(regions[other].begin + regions[other].size <= region.begin);
        }
    }
    assert(c.tail_offset + c.tail.bytes == c.bytes);
    assert(c.front.capacity == (rows <= 16 ? 16 : 32));
    assert(c.indexer.pages != c.attention.pages);
    assert(c.indexer.table_pages != c.attention.table_pages);
    assert(c.indexer.rows == c.front.rows && c.front.rows == c.attention.rows);
    assert(c.indexer.blocks == c.front.blocks && c.front.blocks == blocks);
    assert(c.attention.query_offset + r * 4 * 512 * 2 <= c.attention.rope_offset);
    assert(c.attention.rope_offset + r * 4 * 64 * 2 <= c.attention.work_offset);
    assert(c.front.system + (16ULL << 20) <= c.front.bytes);
    assert(c.tail.system + (16ULL << 20) <= c.tail.bytes);
    assert(indexer_front::kTilingBytes % 512 == 0);
    assert(whole_layer_indexed::kTilingBytes == 2560);
}

int main()
{
    uint64_t cases = 0;
    for (uint32_t rows = 1; rows <= 32; ++rows) {
        for (uint32_t blocks : {1U, 8U, 24U}) {
            for (uint32_t pages : {1U, 3U, 65U, 1025U}) {
                for (bool quantized : {false, true}) {
                    Check(rows, blocks, pages, quantized);
                    ++cases;
                }
            }
        }
    }
    std::cout << "indexed workspace contracts: " << cases << " cases\n";
    for (uint32_t rows : {1U, 12U, 32U}) {
        for (bool quantized : {false, true}) {
            const auto c = whole_layer_indexed::MakeConfig(
                whole_layer_attention::MakeConfig(rows, 68, 65, 24,
                                                   .0625F, 1e-5F, 1e-5F),
                71, 66, quantized);
            std::cout << "M=" << rows << " int8=" << quantized
                      << " scratch=" << c.bytes
                      << " tiling=" << c.front.capacity *
                          whole_layer_indexed::kTilingBytes
                      << " attention=" << c.attention.bytes
                      << " front=" << c.front.bytes
                      << " indexer=" << c.indexer.bytes
                      << " tail=" << c.tail.bytes << '\n';
        }
    }
}
