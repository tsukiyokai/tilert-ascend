"""Expose initialized local-chain outputs without reading reused workspaces."""

import math


def tensor_view(torch, scratch, offset, shape, dtype):
    count = math.prod(shape)
    size = count * torch.empty((), dtype=dtype).element_size()
    if offset < 0 or offset + size > scratch.numel():
        raise ValueError("A logical observation exceeds its scratch owner")
    return scratch[offset:offset + size].view(dtype).reshape(shape)


def attention_values(torch, scratch, offsets, rows, mode):
    """Shared attention retains latent and tail; its earlier arena is dead."""
    capacity = 16 if rows <= 16 else 32
    if mode == "shared":
        align = lambda value: (value + 511) // 512 * 512
        arena = align(rows * 4 * 512 * 2)
        values = align(capacity * 4 * 512 * 2)
        quantized = align(values + capacity * 1024 * 4)
        scales = align(quantized + capacity * 1024)
        accumulators = align(scales + capacity * 8 * 4)
        offsets = {"latent": 0, "tail_values": arena + values,
                   "tail_quantized": arena + quantized,
                   "tail_scales": arena + scales,
                   "tail_accumulators": arena + accumulators}
    shapes = {"latent": ((rows, 4, 512), torch.bfloat16)}
    if mode != "shared":
        shapes.update({
            "attention_query": ((rows, 4, 512), torch.bfloat16),
            "attention_rope": ((rows, 4, 64), torch.bfloat16),
            "query_norm": ((rows, 2048), torch.int8),
            "query_norm_scale": ((rows,), torch.float32),
            "query": ((rows, 32, 128),
                      torch.int8 if mode == "int8" else torch.bfloat16),
            "key": ((rows, 128),
                    torch.int8 if mode == "int8" else torch.bfloat16),
            "weights": ((rows, 32),
                        torch.float16 if mode == "int8" else torch.bfloat16),
            "topk": ((rows, 1, 2048), torch.int32),
        })
        for name, shape in (
                ("query_projection", (rows, 32, 128)),
                ("key_projection", (rows, 160)),
                ("key_normalized", (rows, 128)),
                ("query_rope", (rows, 32, 128)),
                ("key_rope", (rows, 128))):
            shapes["front_" + name] = shape, torch.bfloat16
        if mode == "int8":
            shapes.update({
                "query_scale": ((rows, 32), torch.float16),
                "key_scale": ((rows,), torch.float16),
                "front_query_hadamard": ((rows, 32, 128), torch.float32),
                "front_key_hadamard": ((rows, 128), torch.float32),
            })
    result = {name: tensor_view(torch, scratch, offsets[name], shape, dtype)
              for name, (shape, dtype) in shapes.items()}
    result["tail_values"] = tensor_view(
        torch, scratch, offsets["tail_values"], (16, capacity, 64),
        torch.float32)[:, :rows]
    result["tail_quantized"] = tensor_view(
        torch, scratch, offsets["tail_quantized"], (capacity, 1024),
        torch.int8)[:rows]
    result["tail_scales"] = tensor_view(
        torch, scratch, offsets["tail_scales"], (capacity, 8),
        torch.float32)[:rows]
    result["tail_accumulators"] = tensor_view(
        torch, scratch, offsets["tail_accumulators"], (24, capacity, 256),
        torch.int32)[:, :rows]
    return result


def poison(values, integer_value=1):
    """Invalidate outputs while keeping speculative integer addresses bounded."""
    for name, value in values.items():
        bounded = name.endswith(("ids", "counts", "starts", "source_row",
                                 "source_slot", "topk", "active_list"))
        value.fill_(float("nan") if value.is_floating_point()
                    else 0 if bounded else integer_value)


def strict_checks(torch, actual, expected, metric):
    if actual.keys() != expected.keys():
        raise RuntimeError("The two arms expose different boundary sets")
    checks = {name: metric(torch, actual[name], value)
              for name, value in expected.items()}
    return checks, all(value["pass"] and value["bitwise_equal"]
                       for value in checks.values())
