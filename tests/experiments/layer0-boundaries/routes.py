"""Capture the expert IDs, weights and logits consumed by a real MoE layer."""

import hashlib
import inspect
from functools import wraps
from pathlib import Path
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_routes(layer, samples):
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    routed = layer.mlp.experts.routed_experts
    original = routed._select_experts
    router = routed.router

    @wraps(original)
    def observed(
        hidden_states, router_logits, enable_force_load_balance, input_ids=None
    ):
        result = original(
            hidden_states, router_logits, enable_force_load_balance, input_ids
        )
        weights, ids = result
        router_source = Path(inspect.getfile(type(router))).resolve()
        routed_source = Path(inspect.getfile(type(routed))).resolve()
        samples.append(
            {
                "ids": ids.detach().clone(),
                "weights": weights.detach().clone(),
                "router_logits": router_logits.detach().clone(),
                "metadata": {
                    "router_class": type(router).__module__
                    + "."
                    + type(router).__name__,
                    "top_k": router.top_k,
                    "eplb_state_known": hasattr(router, "eplb_state"),
                    "eplb_state_present": (
                        router.eplb_state is not None
                        if hasattr(router, "eplb_state")
                        else None
                    ),
                    "log2phy_present": routed.log2phy is not None,
                    "mix_placement": getattr(routed, "mix_placement", False),
                    "enable_force_eplb": get_ascend_config().enable_force_eplb,
                    "enable_force_load_balance": enable_force_load_balance,
                    "in_profile_run": getattr(
                        _EXTRA_CTX, "in_profile_run", None
                    ),
                    "source_sha256": {
                        str(router_source): digest(router_source),
                        str(routed_source): digest(routed_source),
                    },
                },
            }
        )
        return result

    replacement = patch.object(routed, "_select_experts", observed)
    replacement.start()
    return replacement.stop
