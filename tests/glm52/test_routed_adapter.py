"""CPU coverage of native weight addressing and top-8 route ownership."""

from types import SimpleNamespace
import sys

import pytest
import torch

from tilert.models.glm_5_2_ascend import routed_expert
from tilert.models.glm_5_2_ascend.reference_weights import unpack_w4


@pytest.mark.parametrize("shape", [(2, 8, 16), (3, 256, 64), (1, 64, 128)])
def test_nontransposed_int4_matmul(shape):
    experts, outputs, inner = shape
    generator = torch.Generator().manual_seed(20261021)
    checkpoint = torch.randint(
        -128, 128, (experts, outputs // 2, inner), dtype=torch.int8,
        generator=generator,
    )
    native = routed_expert._repack(checkpoint, outputs)
    activations = torch.randint(
        -128, 128, (5, inner), dtype=torch.int32, generator=generator,
    )
    for expert in range(experts):
        logical = unpack_w4(checkpoint[expert]).int()
        b = torch.empty(inner, outputs, dtype=torch.int32)
        # Address B as the non-transposed CUBE operand, not a checkpoint.
        for k in range(inner):
            for n in range(outputs):
                byte = int(native[expert, k, n // 2]) & 255
                code = (byte >> (4 * (n % 2))) & 15
                b[k, n] = code if code < 8 else code - 16
        torch.testing.assert_close(activations @ b, activations @ logical.T)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 12, 16, 32])
@pytest.mark.parametrize("routing", ["hot", "disjoint", "overlap"])
def test_routes_preserve_token_slot_and_bf16_probability(tokens, routing):
    ids = torch.arange(tokens * 8).reshape(tokens, 8)
    if routing == "hot":
        ids = ids % 8
    elif routing == "overlap":
        ids = (ids * 31 + ids // 8) % 256
    probabilities = torch.linspace(0.001, 0.999, tokens * 8).reshape(tokens, 8)
    counts, starts, rows, slots, weights = routed_expert.route_metadata(
        ids, probabilities,
    )
    expected = sorted(
        (int(ids[token, slot]), token, slot)
        for token in range(tokens) for slot in range(8)
    )
    offset = 0
    for expert in range(256):
        group = [entry for entry in expected if entry[0] == expert]
        assert int(counts[expert]) == len(group)
        assert int(starts[expert]) == offset
        offset += len(group)
    for packed, (_, token, slot) in enumerate(expected):
        assert (int(rows[packed]), int(slots[packed])) == (token, slot)
        assert weights[packed] == probabilities[token, slot].bfloat16().float()
    assert int(counts.sum()) == tokens * 8


def test_changing_routes_do_not_retain_previous_groups():
    probabilities = torch.ones(32, 8)
    old = routed_expert.route_metadata(
        torch.arange(8).expand(32, -1), probabilities,
    )
    new = routed_expert.route_metadata(
        torch.arange(248, 256).expand(32, -1), probabilities,
    )
    assert old[0][:8].tolist() == [32] * 8
    assert new[0][:8].tolist() == [0] * 8
    assert new[0][-8:].tolist() == [32] * 8


@pytest.mark.parametrize("tokens", [1, 12, 32])
def test_recover_quantized_tokens_from_dispatch_inverse(tokens):
    generator = torch.Generator().manual_seed(20261021)
    rows = torch.randint(-127, 128, (tokens, 64), dtype=torch.int8,
                         generator=generator)
    scales = torch.linspace(0.001, 0.07, tokens)
    expert_ids = torch.stack([
        torch.randperm(256, generator=generator)[:8]
        for _ in range(tokens)
    ])
    # InitRouting emits sorted rows and the inverse of the original
    # token-major [M,8] permutation, including when groups are empty.
    sorted_slots = expert_ids.flatten().argsort(stable=True)
    source_tokens = sorted_slots // 8
    inverse = torch.empty(tokens * 8, dtype=torch.int32)
    inverse[sorted_slots] = torch.arange(tokens * 8, dtype=torch.int32)
    recovered_rows, recovered_scales = routed_expert._restore_token_rows(
        rows[source_tokens], scales[source_tokens], inverse, tokens,
    )
    torch.testing.assert_close(recovered_rows, rows, rtol=0, atol=0)
    torch.testing.assert_close(recovered_scales, scales, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 12, 32])
def test_dispatch_inverse_does_not_require_stable_expert_rows(tokens):
    """An inverse permutation preserves slots even if each group is reversed."""
    ids = (torch.arange(8).expand(tokens, -1)
           + torch.arange(tokens).view(-1, 1)) % 8
    groups = [[original for original, expert in enumerate(ids.flatten())
               if int(expert) == wanted][::-1] for wanted in range(256)]
    order = torch.tensor([original for group in groups for original in group])
    inverse = torch.empty(tokens * 8, dtype=torch.int32)
    inverse[order] = torch.arange(tokens * 8, dtype=torch.int32)
    counts = torch.tensor([len(group) for group in groups], dtype=torch.int64)
    assert counts.shape == (256,) and not bool(counts[8:].any())
    probability = torch.linspace(0.0011, 0.9837, tokens * 8).reshape(tokens, 8)
    probability = probability.bfloat16().float()
    values = (torch.arange(tokens * 8).reshape(tokens, 8) * 0.031 + 0.4).bfloat16()
    sorted_values = values.flatten()[order]
    combined = torch.zeros(tokens)
    expected = torch.zeros(tokens)
    for slot in range(8):
        original = torch.arange(tokens) * 8 + slot
        combined += sorted_values[inverse[original].long()].float() * probability[:, slot]
        expected += values[:, slot].float() * probability[:, slot]
    assert torch.equal(combined.bfloat16(), expected.bfloat16())
    # A [8,M] interpretation would address different original route slots.
    if tokens > 1:
        assert not torch.equal(inverse.reshape(tokens, 8)[:, 0],
                               inverse.reshape(8, tokens)[0])


def test_shared_quant_scheme_uses_the_requested_layers_weights(monkeypatch):
    calls = []
    scheme = SimpleNamespace(apply=lambda *args: "vendor")
    first, second, foreign = object(), object(), object()
    monkeypatch.setattr(
        routed_expert, "_run",
        lambda method, layer, x, probabilities, ids, weights, pool:
        calls.append((layer, weights)) or weights,
    )
    undo_first = routed_expert._install_layer(first, scheme, "first", None)
    undo_second = routed_expert._install_layer(second, scheme, "second", None)
    small = torch.empty(12, 6144)
    assert scheme.apply(first, small, None, None, None, None) == "first"
    assert scheme.apply(second, small, None, None, None, None) == "second"
    assert scheme.apply(foreign, small, None, None, None, None) == "vendor"
    large = torch.empty(33, 6144)
    assert scheme.apply(first, large, None, None, None, None) == "vendor"
    undo_second()
    undo_first()
    assert scheme.apply(first, small, None, None, None, None) == "vendor"
    assert calls == [(first, "first"), (second, "second")]


def test_active_lora_keeps_the_vendor_quant_method(monkeypatch):
    layer = SimpleNamespace(_ascend_moe_lora_context=object())
    scheme = SimpleNamespace(apply=lambda *args: "vendor")
    monkeypatch.setattr(routed_expert, "_run", lambda *args: "native")
    restore = routed_expert._install_layer(layer, scheme, None, None)
    try:
        assert scheme.apply(layer, torch.empty(1, 6144, dtype=torch.bfloat16),
                            None, None, None, None) == "vendor"
    finally:
        restore()


def test_loading_hook_captures_weights_before_nz_conversion(monkeypatch):
    class Scheme:
        def process_weights_after_loading(self, layer):
            layer.w13_weight = torch.empty(256, 1, 1, 1, 1, device="meta")

        def apply(self, *args):
            return "vendor"

    scheme = Scheme()
    layer = SimpleNamespace(
        quant_method=SimpleNamespace(quant_method=scheme),
        activation="silu", moe_config=SimpleNamespace(swiglu_limit=0),
    )
    for name, shape in {
        "w13_weight": (256, 128, 6144),
        "w2_weight": (256, 3072, 128),
        "w13_weight_scale": (256, 256, 1),
        "w2_weight_scale": (256, 6144, 1),
        "w13_scale_bias": (256, 256, 1),
        "w2_scale_bias": (256, 6144, 1),
    }.items():
        dtype = torch.int8 if name.endswith("_weight") else torch.float32
        setattr(layer, name, torch.empty(shape, dtype=dtype, device="meta"))
    seen = []
    monkeypatch.setattr(routed_expert, "_load", lambda: None)
    monkeypatch.setattr(
        routed_expert, "_run",
        lambda method, target, x, probabilities, ids, weights, pool:
        seen.append((method, target, weights)) or "native",
    )
    original = Scheme.process_weights_after_loading
    restore, installed, pool = routed_expert.install_before_conversion(Scheme)
    try:
        scheme.process_weights_after_loading(layer)
        assert layer.w13_weight.ndim == 5
        assert installed == [layer]
        assert scheme.apply(layer, torch.empty(12, 6144), None, None,
                            None, None) == "native"
        assert seen[0][:2] == (scheme, layer)
        assert seen[0][2][0].shape == (256, 6144, 128)
        assert seen[0][2][3].shape == (256, 128, 3072)
        assert pool.experts == 256
        assert pool.layer_weights[id(layer)] is seen[0][2]
    finally:
        restore()
    assert Scheme.process_weights_after_loading is original
    assert pool.layer_weights == {}
    assert scheme.apply(None) == "vendor"


def test_nz_hook_borrows_vendor_owners_and_prepares_matching_layout(monkeypatch):
    """NZ dtype views must keep their owner; encoded scales are a different ABI."""
    class Scheme:
        def process_weights_after_loading(self, layer):
            layer.w13_weight = torch.empty(
                256, 6144, 32, dtype=torch.int32, device="meta",
            )
            layer.w2_weight = torch.empty(
                256, 128, 768, dtype=torch.int32, device="meta",
            )
            for name, columns in (("w13", 256), ("w2", 6144)):
                setattr(layer, name + "_weight_scale", torch.empty(
                    256, columns, dtype=torch.int64, device="meta",
                ))
                setattr(layer, name + "_scale_bias", torch.empty(
                    256, columns, dtype=torch.float32, device="meta",
                ))

        def apply(self, *args):
            return "vendor"

    layer = SimpleNamespace(
        activation="silu", moe_config=SimpleNamespace(swiglu_limit=0),
    )
    for name, shape in {
        "w13_weight": (256, 128, 6144),
        "w2_weight": (256, 3072, 128),
        "w13_weight_scale": (256, 256, 1),
        "w2_weight_scale": (256, 6144, 1),
        "w13_scale_bias": (256, 256, 1),
        "w2_scale_bias": (256, 6144, 1),
    }.items():
        dtype = torch.int8 if name.endswith("_weight") else torch.float32
        setattr(layer, name, torch.empty(shape, dtype=dtype, device="meta"))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        get_npu_format=lambda tensor: 29,
    ))
    monkeypatch.setattr(routed_expert, "_load", lambda: None)
    monkeypatch.setattr(
        routed_expert, "_run",
        lambda method, target, x, probabilities, ids, weights, pool: weights,
    )
    prepared = []
    monkeypatch.setattr(
        torch.ops.tilert, "routed_experts_prepare",
        lambda *args, **kwargs: prepared.append((args, kwargs)) or "scratch",
        raising=False,
    )
    scheme = Scheme()
    restore, _, pool = routed_expert.install_before_conversion(
        Scheme, weight_layout="nz",
    )
    try:
        scheme.process_weights_after_loading(layer)
        weights = scheme.apply(layer, torch.empty(12, 6144), None, None,
                               None, None)
        assert weights[0] is layer.w13_weight
        assert weights[3] is layer.w2_weight
        assert weights[1] is layer.w13_scale_bias
        assert weights[4] is layer.w2_scale_bias
        assert weights[2].dtype == weights[5].dtype == torch.float32
        assert layer.w13_weight_scale.dtype == torch.int64
        pool.reserve_capture(torch.empty(12, 6144, dtype=torch.int8))
        assert prepared[0][1] == {"weight_layout": 1}
        assert next(iter(pool.capture_reserves))[-1] == 1
        assert scheme.apply(layer, torch.empty(33, 6144), None, None,
                            None, None) == "vendor"
    finally:
        restore()
