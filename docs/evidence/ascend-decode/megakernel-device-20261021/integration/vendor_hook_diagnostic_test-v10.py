"""CPU checks for retained-region decoding and independent O-dot analysis."""

import torch

from vendor_hook_diagnostic import compare, decode_native, output_oracles


def test_decode_preserves_tile_order_and_valid_rows():
    rows, capacity = 3, 16
    sources = {
        "attention_query": torch.arange(rows * 4 * 512).reshape(rows, 4, 512).bfloat16(),
        "attention_rope": torch.arange(rows * 4 * 64).reshape(rows, 4, 64).bfloat16(),
        "query_norm": torch.arange(rows * 2048).reshape(rows, 2048).to(torch.int8),
        "query_norm_scale": torch.arange(rows).float(),
        "latent": torch.arange(rows * 4 * 512).reshape(rows, 4, 512).bfloat16(),
        "tail_values": torch.arange(16 * capacity * 64).reshape(16, capacity, 64).float(),
        "tail_quantized": torch.arange(capacity * 1024).reshape(capacity, 1024).to(torch.int8),
        "tail_scales": torch.arange(capacity * 8).reshape(capacity, 8).float(),
        "tail_accumulators": torch.arange(24 * capacity * 256).reshape(24, capacity, 256).int(),
    }
    offsets = {"capacity": capacity}
    parts, size = [], 0
    for name, value in sources.items():
        offsets[name] = size
        parts.append(value.flatten().view(torch.uint8))
        size += value.numel() * value.element_size()
    scratch = torch.cat(parts)
    result = decode_native(torch, scratch, offsets, rows)
    for row in range(rows):
        for column in (0, 63, 64, 255, 256, 1023):
            expected = (column // 64 * capacity + row) * 64 + column % 64
            assert result["value_up_fp32"][row, column].item() == expected
        for column in (0, 255, 256, 4095, 6143):
            expected = (column // 256 * capacity + row) * 256 + column % 256
            assert result["o_accumulators"][row, column].item() == expected
    assert result["o_token_scale"].tolist() == [0, 8, 16]
    assert result["o_codes"].shape == (rows, 1024)
    assert result["value_up"].dtype == torch.bfloat16
    assert torch.equal(result["attention_query"], sources["attention_query"])


def test_integer_oracle_uses_codes_and_weight_values():
    values = {
        "o_codes": torch.tensor([[127, -128, 3], [-7, 5, 11]], dtype=torch.int8),
        "o_weight": torch.tensor([[8, -9], [-6, 7], [5, 4]], dtype=torch.int8),
        "o_weight_scale": torch.tensor([0.003, 0.002], dtype=torch.bfloat16),
        "o_token_scale": torch.tensor([0.25, 0.125]),
    }
    native = dict(values, o_weight_scale=values["o_weight_scale"].float())
    result, bounds = output_oracles(torch, values, native)
    expected = torch.tensor([[1799, -2027], [-31, 142]], dtype=torch.int32)
    assert torch.equal(result["vendor_i32_oracle"], expected)
    assert torch.equal(result["native_i32_oracle"], expected)
    assert bounds["vendor"] == {"minimum": -2027, "maximum": 1799}


def test_comparison_detects_signed_zero_and_rounding_difference():
    native = torch.tensor([0.0, 0.0034637451171875], dtype=torch.bfloat16)
    vendor = torch.tensor([-0.0, 0.00347900390625], dtype=torch.bfloat16)
    result = compare(torch, native, vendor)
    assert result["bitwise_different_bytes"] == 2
    assert [item["coordinate"] for item in result["first_differences"]] == [[0], [1]]
