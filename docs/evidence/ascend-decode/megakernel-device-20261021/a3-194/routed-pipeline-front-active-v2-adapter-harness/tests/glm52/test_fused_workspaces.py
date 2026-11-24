"""Host allocation lifecycle used before native NPU graph capture."""

from types import SimpleNamespace

import pytest
import torch

from tilert.models.glm_5_2_ascend.mla_tail import WorkspacePool
from tilert.models.glm_5_2_ascend.routed_expert import _WorkspacePool
from tilert.models.glm_5_2_ascend.routed_shared import WorkspacePool as SharedPool


@pytest.mark.parametrize("kind", [
    "moe", "moe_dispatch", "moe_shared", "moe_shared_front", "mla",
])
def test_capture_claims_reserved_storage_without_allocating(monkeypatch, kind):
    runtime = SimpleNamespace(capturing=False, stream=11)
    allocations = []
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        current_stream=lambda device: SimpleNamespace(
            npu_stream=runtime.stream,
        ),
        is_current_stream_capturing=lambda: runtime.capturing,
    ), raising=False)

    def prepare(*args, **kwargs):
        if kind == "moe_dispatch":
            assert kwargs == {"from_dispatch": True}
        result = object()
        allocations.append(result)
        return result

    namespace = torch.ops.tilert
    if kind in ("moe_shared", "moe_shared_front"):
        input_front = kind == "moe_shared_front"
        pool = SharedPool(input_front=input_front)
        opname = "routed_shared_prepare"
        namespace = getattr(torch.ops, pool.namespace_name(12))
        reference = torch.empty(12, 6144, dtype=torch.int8)
        result = pool.get
    elif kind in ("moe", "moe_dispatch"):
        pool = _WorkspacePool(256, 16, from_dispatch=kind == "moe_dispatch")
        opname = "routed_experts_prepare"
        reference = torch.empty(12, 6144, dtype=torch.int8)
        result = pool.get
    else:
        pool = WorkspacePool()
        opname = "mla_tail_prepare"
        reference = torch.empty(12, 4, 512, dtype=torch.bfloat16)
        result = lambda value: pool.get(value)[1]
    monkeypatch.setattr(namespace, opname, prepare, raising=False)
    eager = result(reference)
    pool.reserve_capture(reference)
    reserved = allocations[-1]
    runtime.stream = 23
    runtime.capturing = True
    assert result(reference) is reserved
    assert result(reference) is reserved
    assert len(allocations) == 2
    runtime.stream = 11
    runtime.capturing = False
    assert result(reference) is eager
    runtime.stream = 29
    runtime.capturing = True
    with pytest.raises(RuntimeError, match="Reserve"):
        result(reference)
    assert len(allocations) == 2
