"""Check shared tile algebra and the routed callback/finalize boundary."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest
import torch

from tilert.models.glm_5_2_ascend import routed_shared


def test_shared_tiles_preserve_full_integer_projections():
    generator = torch.Generator().manual_seed(872)
    gate = torch.randint(-128, 128, (6144, 256), dtype=torch.int8,
                         generator=generator)
    down = torch.randint(-128, 128, (128, 6144), dtype=torch.int8,
                         generator=generator)
    inputs = torch.randint(-128, 128, (3, 6144), generator=generator)
    middle = torch.randint(-128, 128, (3, 128), generator=generator)
    gate_tiles, down_tiles = routed_shared.pack_shared_weights(gate, down)
    actual_gate = torch.zeros(3, 256, dtype=torch.int64)
    for tile, weight in enumerate(gate_tiles):
        inner, columns = divmod(tile, 4)
        actual_gate[:, columns * 64:(columns + 1) * 64] += (
            inputs[:, inner * 1024:(inner + 1) * 1024] @ weight.long()
        )
    actual_down = torch.cat([middle @ weight.long() for weight in down_tiles], 1)
    torch.testing.assert_close(actual_gate, inputs @ gate.long(), rtol=0, atol=0)
    torch.testing.assert_close(actual_down, middle @ down.long(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_shared_scale_contract_matches_cann_supported_types(monkeypatch, dtype):
    class Scheme:
        act_quant_type = torch.int8

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        npu_format_cast=lambda value, layout: value))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic",
        SimpleNamespace(AscendW8A8DynamicLinearMethod=Scheme),
    )

    def projection(shape):
        scale = torch.linspace(0.001, 0.03, shape[1]).to(dtype)
        return SimpleNamespace(
            quant_method=SimpleNamespace(quant_method=Scheme()), tp_size=16,
            weight=torch.empty(*shape, dtype=torch.int8, device="meta"),
            weight_scale=scale, weight_scale_fp32=scale.float(),
        )

    shared = SimpleNamespace(layer=SimpleNamespace(
        gate_up_proj=projection((6144, 256)), down_proj=projection((128, 6144))))
    if dtype == torch.float16:
        with pytest.raises(ValueError, match="FP32 or BF16"):
            routed_shared._shared_weights(shared)
    else:
        weights = routed_shared._shared_weights(shared)
        torch.testing.assert_close(
            weights[1], shared.layer.gate_up_proj.weight_scale.float(),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            weights[3], shared.layer.down_proj.weight_scale.float(),
            rtol=0, atol=0,
        )


@pytest.fixture
def runner(monkeypatch):
    class AllGather:
        pass

    events = []
    context = SimpleNamespace(moe_comm_method=AllGather())
    for name, module in {
        "vllm_ascend.ascend_forward_context": SimpleNamespace(_EXTRA_CTX=context),
        "vllm_ascend.ops.fused_moe.moe_comm_method": SimpleNamespace(
            AllGatherCommImpl=AllGather, FusedExpertsResult=SimpleNamespace),
        "vllm_ascend.ops.fused_moe.moe_mlp": SimpleNamespace(
            maybe_record_event=lambda: events.append("event")),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    class Routed:
        return_with_event = True
        ascend_pertoken_scale = None

        def forward_impl(self, *, hidden_states, router_logits, input_ids,
                         fused_experts=None):
            events.append("prepare_and_select")
            value = fused_experts(
                layer=self, x=hidden_states + 2,
                topk_weights=router_logits, topk_ids=input_ids,
                shared_experts=None, shared_experts_input=None,
            )
            events.append("finalize")
            return value.routed_out + 3, object()

    runner = SimpleNamespace(
        moe_config=SimpleNamespace(tp_size=16, ep_size=1, dp_size=1, pcp_size=1,
                                   is_sequence_parallel=False),
        routed_input_transform=None, routed_output_transform=None,
        ascend_shared_experts=SimpleNamespace(
            weights_replicated=False, situ_activation=None, swiglu_limit=0,
            swiglu_alpha=1, swiglu_beta=0, layer=SimpleNamespace(),
            lora_context=None,
        ),
        routed_experts=Routed(), is_internal_router=True,
        _sequence_parallel_context=nullcontext,
        _compute_router_logits=lambda hidden, logits: logits + hidden[0, 0].float(),
        _forward_impl=lambda *args: "vendor",
        events=events,
    )
    monkeypatch.setattr(routed_shared, "_shared_weights", lambda shared: "shared")
    return runner


def test_callback_uses_prepared_routes_and_preserves_finalize(monkeypatch, runner):
    hidden = torch.ones(2, 6144, dtype=torch.bfloat16)
    shared_input = hidden * 4
    probabilities = torch.full((2, 8), 0.125)
    ids = torch.arange(8).expand(2, -1)
    calls = []

    def body(x, weights, selected, shared, routed_weights, shared_weights, pool):
        calls.append((x, weights, selected, shared, routed_weights,
                      shared_weights, pool))
        return hidden * 6, hidden * 7, torch.zeros(256, dtype=torch.int64)

    monkeypatch.setattr(routed_shared, "run_body", body)
    pool = object()
    restore = routed_shared.install(runner, "routed", pool)
    output = runner._forward_impl(hidden, probabilities, shared_input, ids)
    torch.testing.assert_close(output[0], hidden * 7, rtol=0, atol=0)
    torch.testing.assert_close(output[1], hidden * 9, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][0], hidden + 2, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], probabilities + 4, rtol=0, atol=0)
    assert calls[0][2] is ids and calls[0][3] is shared_input
    assert calls[0][4:] == ("routed", "shared", pool)
    assert runner.events == ["prepare_and_select", "event", "event", "finalize"]
    restore()
    assert runner._forward_impl(hidden, probabilities, shared_input, ids) == "vendor"


@pytest.mark.parametrize("reason", ["prefill", "lora", "quantized", "shared_shape"])
def test_unsupported_calls_preserve_vendor_path(runner, reason):
    routed_shared.install(runner, "routed", object())
    hidden = torch.zeros(33 if reason == "prefill" else 1, 6144,
                         dtype=torch.bfloat16)
    shared = hidden
    if reason == "lora":
        runner.ascend_shared_experts.lora_context = object()
    elif reason == "quantized":
        hidden = hidden.to(torch.int8)
    elif reason == "shared_shape":
        shared = hidden[:, :128]
    assert runner._forward_impl(hidden, None, shared) == "vendor"
    assert runner.events == []


@pytest.mark.parametrize("field,value", [("tp_size", 8), ("ep_size", 16),
                                         ("dp_size", 2), ("pcp_size", 2),
                                         ("is_sequence_parallel", True)])
def test_distributed_layout_is_checked_before_install(runner, field, value):
    setattr(runner.moe_config, field, value)
    with pytest.raises(ValueError, match="TP16"):
        routed_shared.install(runner, "routed", object())


@pytest.mark.parametrize("blocks,width,expected", [
    (24, 12, "tilert_pipeline_front_active"),
    (24, 1, "tilert_pipeline_front"),
    (24, 2, "tilert_pipeline_front"),
    (24, 32, "tilert_pipeline_front"),
    (16, 12, "tilert_pipeline_front"),
])
def test_shape_selection_claims_matching_scratch_and_entry(
        monkeypatch, blocks, width, expected):
    runtime = SimpleNamespace(capturing=False, stream=11)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        current_stream=lambda device: SimpleNamespace(npu_stream=runtime.stream),
        is_current_stream_capturing=lambda: runtime.capturing,
    ), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        npu_dynamic_quant=lambda value: (
            torch.zeros_like(value, dtype=torch.int8),
            torch.ones(value.shape[0])),
    ))
    allocations, launches = [], []

    def register(name):
        namespace = getattr(torch.ops, name)

        def prepare(*args):
            scratch = [object() for _ in range(21 if name.endswith("active") else 20)]
            allocations.append((name, scratch))
            return scratch, 123

        def launch(*args):
            scratch, sync = args[-5:-3]
            assert sync == 123
            assert any(owner == name and value is scratch
                       for owner, value in allocations)
            launches.append((name, scratch))
            return "routed", "shared"

        monkeypatch.setattr(namespace, "routed_shared_prepare", prepare, raising=False)
        monkeypatch.setattr(namespace, "routed_shared_from_bf16", launch, raising=False)

    register("tilert_pipeline_front")
    register("tilert_pipeline_front_active")
    pool = routed_shared.WorkspacePool(blocks=blocks)
    hidden = torch.zeros(width, 6144, dtype=torch.bfloat16)
    pool.reserve_capture(hidden.to(torch.int8))
    assert allocations[0][0] == expected
    reserved = allocations[0][1]
    runtime.stream, runtime.capturing = 23, True
    args = (hidden, torch.zeros(width, 8), torch.zeros(width, 8, dtype=torch.int64),
            hidden, (), (), pool)
    result = routed_shared.run_body(*args)
    assert result == ("routed", "shared", reserved[12])
    assert launches == [(expected, reserved)]
    assert not pool.capture_reserves and len(allocations) == 1
    routed_shared.run_body(*args)
    assert len(allocations) == 1 and launches[-1][1] is reserved
    runtime.stream, runtime.capturing = 11, False
    routed_shared.run_body(*args)
    assert len(allocations) == 2 and launches[-1][1] is not reserved
