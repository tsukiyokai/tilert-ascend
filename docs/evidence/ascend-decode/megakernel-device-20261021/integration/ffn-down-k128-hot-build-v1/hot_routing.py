"""Observe the original router and check the hot-expert M-tile boundary."""

from contextlib import contextmanager
import hashlib
from types import MethodType


def configure_hot_bias(torch, layer):
    gate_bias = layer.mlp.gate.e_score_correction_bias
    router = layer.mlp.experts.routed_experts.router
    router_bias = router.e_score_correction_bias
    if (gate_bias.dtype != torch.float32 or tuple(gate_bias.shape) != (256,)
            or router_bias.dtype != torch.float32
            or router_bias.data_ptr() != gate_bias.data_ptr()):
        raise RuntimeError("The original gate and router must share one FP32 bias")
    if router.routed_scaling_factor != 1.0:
        raise RuntimeError("This native bridge requires unscaled router probabilities")
    gate_bias.fill_(-4.0)
    gate_bias[:8].fill_(4.0)
    return {
        "address": gate_bias.data_ptr(),
        "dtype": str(gate_bias.dtype),
        "shape": list(gate_bias.shape),
        "sha256": hashlib.sha256(gate_bias.detach().cpu().numpy().tobytes()).hexdigest(),
        "selected_experts": list(range(8)),
        "selected_bias": 4.0,
        "other_bias": -4.0,
        "router_scale": router.routed_scaling_factor,
    }


def check_bias_owner(torch, layer, expected_address):
    gate_bias = layer.mlp.gate.e_score_correction_bias
    router_bias = layer.mlp.experts.routed_experts.router.e_score_correction_bias
    if (gate_bias.data_ptr() != expected_address
            or router_bias.data_ptr() != expected_address):
        raise RuntimeError("The original router changed the shared bias owner")
    expected = torch.full((256,), -4.0, dtype=torch.float32)
    expected[:8] = 4.0
    actual = gate_bias.detach().cpu().contiguous().view(torch.uint8)
    if not torch.equal(actual, expected.view(torch.uint8)):
        raise RuntimeError("The shared hot-routing bias changed")


@contextmanager
def observe_router(torch, layer):
    router = layer.mlp.experts.routed_experts.router
    original = router._select_experts
    had_override = "_select_experts" in vars(router)
    old_override = vars(router).get("_select_experts")
    values = {}

    def select(unused_self, *args, **kwargs):
        if values:
            raise RuntimeError("Expected one original router invocation per FFN")
        result = original(*args, **kwargs)
        if (not isinstance(result, tuple) or len(result) != 2
                or any(not isinstance(value, torch.Tensor) for value in result)):
            raise RuntimeError("The original router must return probabilities and IDs")
        values["route_probabilities"] = result[0].clone()
        values["route_ids"] = result[1].clone()
        return result

    router._select_experts = MethodType(select, router)
    try:
        yield values
        if len(values) != 2:
            raise RuntimeError("The original FFN did not call the observed router")
    finally:
        if had_override:
            router._select_experts = old_override
        else:
            del router._select_experts


def expected_routes(torch, values, rows):
    ids = values["route_ids"].detach().cpu()
    probabilities = values["route_probabilities"].detach().cpu()
    if (tuple(ids.shape) != (rows, 8)
            or ids.dtype not in (torch.int32, torch.int64)
            or tuple(probabilities.shape) != (rows, 8)
            or probabilities.dtype != torch.float32):
        raise RuntimeError("Require original [M,8] integer IDs and FP32 probabilities")
    if not bool(((ids >= 0) & (ids < 256)).all()):
        raise RuntimeError("The original router returned an invalid expert ID")
    counts = torch.bincount(ids.reshape(-1).long(), minlength=256).int()
    active_ids = torch.nonzero(counts, as_tuple=False).reshape(-1).int()
    active = torch.zeros(264, dtype=torch.int32)
    active[0] = active_ids.numel()
    active[8:8 + active_ids.numel()] = active_ids
    expected_ids = torch.arange(8).expand(rows, 8)
    hot = bool(torch.equal(ids.sort(dim=1).values.long(), expected_ids))
    finite = bool(torch.isfinite(probabilities).all())
    metadata = {
        "route_ids": ids.to(torch.int32),
        "route_probabilities": probabilities,
        "route_counts": counts,
        "active_experts": active,
    }
    summary = {
        "original_id_dtype": str(ids.dtype),
        "every_token_selects_first_eight_once": hot,
        "finite_probabilities": finite,
        "active_experts": active_ids.tolist(),
        "counts": counts.tolist(),
        "maximum_count": int(counts.max()),
        "msd_rows_per_active_expert": 2 * rows,
        "down_m_tiles_per_expert": (2 * rows + 31) // 32,
        "pass": hot and finite and active_ids.numel() == 8
        and int(counts.max()) == rows,
    }
    return metadata, summary


def poison_routes(torch, bridge, body):
    bridge[4].fill_(float("nan"))
    bridge[5].copy_(torch.arange(8, dtype=torch.int32,
                                 device=bridge[5].device).expand_as(bridge[5]))
    body[12].zero_()
    body[20].fill_(1)
    body[20][0] = 0
