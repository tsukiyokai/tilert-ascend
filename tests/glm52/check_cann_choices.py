"""Check CANN choice capture with CPU tensors and synthetic call targets.

The fixtures retain the installed SFA and routed-expert method signatures.
They exercise interception, ownership and cleanup, not model arithmetic.
"""

import argparse
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.cann_choices import CannChoices  # noqa: E402


ATTENTION = "_execute_sparse_flash_attention_process"
ROUTER = "_select_experts"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(operation, error=ValueError):
    try:
        operation()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


class Attention:
    def __init__(self):
        self.calls = 0
        self.result = object()
        self.received = None
        self.fail = False
        self.overwrite = False

    def _execute_sparse_flash_attention_process(
        self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
        actual_seq_lengths_query, actual_seq_lengths_key, block_table=None,
    ):
        self.calls += 1
        self.received = topk_indices
        if self.fail:
            raise RuntimeError("Synthetic SFA failure")
        if self.overwrite:
            topk_indices.fill_(-7)
        return self.result


class RoutedExperts:
    def __init__(self, ids):
        self.calls = 0
        self.result = (object(), ids)
        self.fail = False

    def _select_experts(
        self, hidden_states, router_logits, enable_force_load_balance,
        input_ids=None,
    ):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Synthetic router failure")
        return self.result


class Model:
    def __init__(self, width, dtype):
        import torch

        self.width = width
        self.indices = torch.empty((width, 2048), dtype=dtype)
        self.routes = torch.empty((width, 8), dtype=dtype)
        self.model = SimpleNamespace(start_layer=0, end_layer=78, layers=[])
        self.attention = [Attention() for _ in range(78)]

        class RoutedModule(torch.nn.Module, RoutedExperts):
            def __init__(self, ids):
                torch.nn.Module.__init__(self)
                RoutedExperts.__init__(self, ids)

        self.experts = [RoutedModule(self.routes) for _ in range(75)]
        self.original_returns = True
        self.original_inputs = True
        for layer, attention in enumerate(self.attention):
            mlp = SimpleNamespace()
            if layer >= 3:
                mlp.experts = SimpleNamespace(
                    routed_experts=self.experts[layer - 3],
                )
            self.model.layers.append(SimpleNamespace(
                self_attn=SimpleNamespace(mla_attn=SimpleNamespace(
                    mla_attn=SimpleNamespace(impl=attention),
                )), mlp=mlp, use_sequence_parallel_moe=False,
            ))

    def forward(self, *, stop=78, skip_route=None, change=None):
        import torch

        rows = torch.arange(self.width, dtype=self.indices.dtype)[:, None]
        indices = torch.arange(2048, dtype=self.indices.dtype)[None, :]
        routes = torch.arange(8, dtype=self.routes.dtype)[None, :]
        for layer in range(stop):
            self.indices.copy_(indices + rows * 2048 + layer)
            self.indices[:, -1] = -1
            wire = self.indices.unsqueeze(1) if layer % 2 else self.indices
            if change is not None:
                wire = change(layer, wire)
            attention = self.attention[layer]
            if layer % 3 == 0:
                result = getattr(attention, ATTENTION)(
                    None, None, None, wire, None, None, None,
                )
            elif layer % 3 == 1:
                result = getattr(attention, ATTENTION)(
                    ql_nope=None, q_pe=None, kv_cache=None, topk_indices=wire,
                    attn_metadata=None, actual_seq_lengths_query=None,
                    actual_seq_lengths_key=None,
                )
            else:
                result = getattr(attention, ATTENTION)(
                    None, None, kv_cache=None, topk_indices=wire,
                    attn_metadata=None, actual_seq_lengths_query=None,
                    actual_seq_lengths_key=None,
                )
            self.original_returns &= result is attention.result
            self.original_inputs &= attention.received is wire
            if layer >= 3 and layer != skip_route:
                self.routes.copy_((routes + rows * 8 + layer) % 256)
                expert = self.experts[layer - 3]
                if layer % 2:
                    result = expert._select_experts(None, None, False)
                else:
                    result = expert._select_experts(
                        hidden_states=None, router_logits=None,
                        enable_force_load_balance=False, input_ids=None,
                    )
                self.original_returns &= result is expert.result
        self.indices.fill_(-9)
        self.routes.fill_(-9)


def method_state(model):
    return [
        (target, name, name in vars(target), vars(target).get(name))
        for targets, name in ((model.attention, ATTENTION),
                              (model.experts, ROUTER))
        for target in targets
    ]


def restored(before):
    return all((name in vars(target)) == existed
               and (not existed or vars(target)[name] is value)
               for target, name, existed, value in before)


def check_capture(torch, width, dtype):
    model = Model(width, dtype)
    before = method_state(model)
    with CannChoices(model, width) as capture:
        model.forward()
        indices, routes = capture.finish()
        sources = capture.source_metadata()
        rejects(capture.finish, RuntimeError)
    rows = torch.arange(width, dtype=dtype)[:, None]
    for layer, value in enumerate(indices):
        expected = torch.arange(2048, dtype=dtype)[None, :] + rows * 2048
        expected = expected + layer
        expected[:, -1] = -1
        require(torch.equal(value, expected), f"SFA layer {layer}")
        require(value.shape == (width, 2048) and value.dtype == dtype
                and value.device == model.indices.device
                and value.data_ptr() != model.indices.data_ptr(),
                "indices must preserve layout and own storage")
    require(routes[:3] == (None, None, None), "dense layers have no routes")
    for layer, value in enumerate(routes[3:], start=3):
        expected = (torch.arange(8, dtype=dtype)[None, :]
                    + rows * 8 + layer) % 256
        require(torch.equal(value, expected),
                f"MoE layer {layer}")
        require(value.shape == (width, 8) and value.dtype == dtype
                and value.device == model.routes.device
                and value.data_ptr() != model.routes.data_ptr(),
                "routes must preserve layout and own storage")
    require(len(indices) == len(routes) == 78, "complete choices")
    targets = model.attention + model.experts
    require(all(value.calls == 1 for value in targets)
            and model.original_returns and model.original_inputs,
            "each original call and its argument/return identities")
    require(restored(before), "class methods must leave no instance override")
    require(set(sources) == {str(Path(__file__).resolve()),
                            str(ROOT / "tilert/models/glm_5_2_ascend/"
                                "cann_choices.py")}
            and all(hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    == value for path, value in sources.items()),
            "source hashes must cover collector and original methods")
    with patch.object(Path, "read_bytes", side_effect=AssertionError):
        require(capture.source_metadata() == sources,
                "source hashes remain cached after context exit")
    rejects(capture.finish, RuntimeError)


def check_instance_override(torch):
    model = Model(1, torch.int32)
    counts = []
    for target, name in ((model.attention[0], ATTENTION),
                         (model.experts[0], ROUTER)):
        original = getattr(target, name)

        def install(original, target, name):
            @wraps(original)
            def override(*args, **kwargs):
                counts.append(name)
                return original(*args, **kwargs)
            setattr(target, name, override)

        install(original, target, name)
    before = method_state(model)
    with CannChoices(model) as capture:
        model.forward()
        capture.finish()
    require(counts == [ATTENTION, ROUTER] and restored(before),
            "pre-existing instance overrides must run once and survive")


def check_attention_entry(torch):
    model = Model(1, torch.int64)
    model.attention[0].overwrite = True
    with CannChoices(model) as capture:
        model.forward()
        indices, _ = capture.finish()
    require(torch.equal(indices[0][0, :-1], torch.arange(2047))
            and indices[0][0, -1].item() == -1,
            "SFA capture must freeze the input before the original call")


def check_failure(torch, kind):
    model = Model(4, torch.int32)
    before = method_state(model)
    kwargs, error = {}, ValueError
    if kind == "missing_attention":
        kwargs["stop"] = 77
    elif kind == "missing_router":
        kwargs["skip_route"] = 77
    elif kind == "width":
        kwargs["change"] = lambda layer, value: (
            value[:1] if layer == 4 else value
        )
    elif kind == "sfa_dtype":
        kwargs["change"] = lambda layer, value: (
            value.float() if layer == 4 else value
        )
    elif kind == "router_dtype":
        model.experts[0].result = (object(), model.routes.float())
    elif kind == "router_width":
        model.experts[0].result = (object(), model.routes[:1])
    elif kind == "sfa_exception":
        model.attention[20].fail = True
        error = RuntimeError
    elif kind == "router_exception":
        model.experts[10].fail = True
        error = RuntimeError
    elif kind == "enter_exception":
        setattr(model.attention[10], ATTENTION, lambda: None)
        before = method_state(model)

    def run():
        with CannChoices(model, 4) as capture:
            model.forward(**kwargs)
            capture.finish()

    rejects(run, error)
    require(restored(before), f"{kind} must restore every instance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    import torch_npu

    torch.set_num_threads(1)
    initialized_before = torch_npu.npu._initialized
    checks = []

    def record(name, operation):
        result = {"name": name, "passed": False}
        try:
            operation()
            result["passed"] = True
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        checks.append(result)

    for width in (1, 2, 4, 6, 8, 12, 16, 24, 32):
        for dtype in (torch.int32, torch.int64):
            record(f"capture.{width}.{dtype}",
                   lambda: check_capture(torch, width, dtype))
    record("restore_instance_overrides",
           lambda: check_instance_override(torch))
    record("sfa_input_before_call", lambda: check_attention_entry(torch))
    for kind in ("missing_attention", "missing_router", "width", "sfa_dtype",
                 "router_dtype", "router_width", "sfa_exception",
                 "router_exception", "enter_exception"):
        record("reject." + kind, lambda: check_failure(torch, kind))
    for width in (0, 3, True, "4"):
        record(f"reject.width_{width}", lambda: rejects(
            lambda: CannChoices(Model(1, torch.int32), width),
        ))
    initialized_after = torch_npu.npu._initialized
    record("no_npu_initialization", lambda: require(
        not initialized_before and not initialized_after,
        "CPU checks must leave NPU uninitialized",
    ))
    paths = (Path(__file__).resolve(),
             ROOT / "tilert/models/glm_5_2_ascend/cann_choices.py")
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "scope": "Real CPU tensors with synthetic SFA/router call targets; "
                 "no GLM model arithmetic or device acceptance",
        "numerical_model_execution": False, "npu_execution": False,
        "npu_initialized_before": initialized_before,
        "npu_initialized_after": initialized_after,
        "versions": {"torch": str(torch.__version__),
                     "torch_npu": str(torch_npu.__version__)},
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()).hexdigest() for path in paths
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "failed": [item for item in checks
                                 if not item["passed"]]}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
