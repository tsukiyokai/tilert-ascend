"""Check CANN trace hook ownership and ordering with real CPU Torch modules."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from tilert.models.glm_5_2_ascend.cann_trace import CannTrace


class DispatchedNorm(nn.Module):
    """Use the same Module -> forward -> bound-method dispatch as CustomOp."""

    def __init__(self):
        super().__init__()
        self._forward_method = self.forward_oot
        self.mode = "valid"
        self.last_output = None

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_oot(self, value, residual=None):
        if residual is None:
            output = value
        else:
            merged = value + residual
            output = (torch.zeros_like(value), merged)
            if self.mode == "tensor":
                output = merged
            elif self.mode == "one_tuple":
                output = (merged,)
            elif self.mode == "list":
                output = list(output)
            elif self.mode == "non_tensor":
                output = (value, None)
            elif self.mode == "wrong_shape":
                output = (value, merged.expand(2, -1))
            elif self.mode == "wrong_dtype":
                output = (value, merged.float())
        self.last_output = output
        return output


class FixtureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = DispatchedNorm()
        self.use_sequence_parallel_moe = False


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(FixtureLayer() for _ in range(78))
        self.model.start_layer = 0
        self.model.end_layer = 78
        self.model.hidden_size = 4
        self.live = []
        self.unchanged_outputs = True

    def forward(self, stop=78, fail_at=None, width=1):
        rows = torch.arange(width, dtype=torch.bfloat16).unsqueeze(1)
        embedding = rows.expand(width, 4).clone().requires_grad_(True)
        self.live = [embedding]
        residual = self.model.layers[0].input_layernorm(embedding)
        self.unchanged_outputs = residual is embedding
        for layer in range(1, stop):
            if layer == fail_at:
                raise RuntimeError("fixture forward failed")
            norm = self.model.layers[layer].input_layernorm
            output = norm(torch.ones_like(residual), residual)
            self.unchanged_outputs &= output is norm.last_output
            residual = output[1]
            self.live.append(residual)
        return SimpleNamespace(last_hidden=(rows + 78).expand(width, 4))


def hook_keys(model):
    return [
        (tuple(layer.input_layernorm._forward_pre_hooks),
         tuple(layer.input_layernorm._forward_hooks))
        for layer in model.model.layers
    ]


def rejects(name, operation, checks, error=ValueError):
    try:
        operation()
    except error as exc:
        checks.append({"name": name, "passed": True, "error": str(exc)})
    else:
        checks.append({"name": name, "passed": False})


def check_snapshots(checks):
    model = FixtureModel()
    observed = []
    existing = model.model.layers[1].input_layernorm.register_forward_hook(
        lambda module, inputs, output: observed.append(id(output)),
    )
    before = hook_keys(model)
    with CannTrace(model) as trace:
        step = model()
        with torch.no_grad():
            for tensor in model.live:
                tensor.fill_(-1)
        hidden = trace.finish(step)
        step.last_hidden.fill_(-2)
        rejects("second_finish", lambda: trace.finish(step), checks,
                error=RuntimeError)
    checks.extend([
        {"name": "79_ordered_snapshots_and_clone_stability",
         "passed": len(hidden) == 79 and all(
             torch.equal(value, torch.full((1, 4), index,
                                          dtype=torch.bfloat16))
             for index, value in enumerate(hidden))},
        {"name": "snapshots_detached", "passed": all(
            not value.requires_grad and value.grad_fn is None
            for value in hidden)},
        {"name": "hooks_preserve_outputs", "passed": model.unchanged_outputs},
        {"name": "owned_hooks_removed_existing_hook_preserved",
         "passed": hook_keys(model) == before and len(observed) == 1},
    ])
    existing.remove()
    rejects("finish_after_exit", lambda: trace.finish(step), checks,
            error=RuntimeError)


def check_failures(checks):
    def run(model, **kwargs):
        with CannTrace(model) as trace:
            return trace.finish(model(**kwargs))

    model = FixtureModel()
    before = hook_keys(model)
    rejects("missing_last_layer", lambda: run(model, stop=77), checks)
    checks.append({"name": "missing_layer_cleanup",
                   "passed": hook_keys(model) == before})
    for mode in ("tensor", "one_tuple", "list", "non_tensor", "wrong_shape",
                 "wrong_dtype"):
        model = FixtureModel()
        model.model.layers[10].input_layernorm.mode = mode
        before = hook_keys(model)
        rejects(mode, lambda: run(model), checks,
                error=TypeError if mode == "non_tensor" else ValueError)
        checks.append({"name": mode + "_cleanup",
                       "passed": hook_keys(model) == before})
    model = FixtureModel()
    before = hook_keys(model)
    rejects("forward_exception", lambda: run(model, fail_at=40), checks,
            error=RuntimeError)
    checks.append({"name": "forward_exception_cleanup",
                   "passed": hook_keys(model) == before})
    with CannTrace(model) as trace:
        embedding = torch.zeros(1, 4, dtype=torch.bfloat16)
        model.model.layers[0].input_layernorm(embedding)
        rejects("duplicate_hook", lambda: (
            model.model.layers[0].input_layernorm(embedding)), checks)
    for field, value in (("start_layer", 1), ("end_layer", 77)):
        model = FixtureModel()
        setattr(model.model, field, value)
        rejects(field, lambda: CannTrace(model), checks)
    model = FixtureModel()
    model.model.layers[3].use_sequence_parallel_moe = True
    rejects("sequence_parallel", lambda: CannTrace(model), checks)


def check_widths(checks):
    for width in (2, 4):
        model = FixtureModel()
        before = hook_keys(model)
        with CannTrace(model, width) as trace:
            step = model(width=width)
            hidden = trace.finish(step)
            for value in model.live:
                value.detach().fill_(-1)
            step.last_hidden.fill_(-2)
        rows = torch.arange(width, dtype=torch.bfloat16).unsqueeze(1)
        checks.extend([
            {"name": f"width_{width}.79_owned_ordered_rows",
             "passed": len(hidden) == 79 and all(
                 torch.equal(value, (rows + layer).expand(width, 4))
                 for layer, value in enumerate(hidden))},
            {"name": f"width_{width}.hooks_removed",
             "passed": hook_keys(model) == before},
        ])
    for width in (0, 3, True, "4"):
        rejects(f"invalid_width_{width}",
                lambda: CannTrace(FixtureModel(), width), checks)
    model = FixtureModel()
    before = hook_keys(model)

    def mismatch():
        with CannTrace(model, 4):
            model(width=2)

    rejects("width_mismatch", mismatch, checks)
    checks.append({"name": "width_mismatch_cleanup",
                   "passed": hook_keys(model) == before})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch_npu

    torch.set_num_threads(1)
    initialized_before = torch_npu.npu._initialized
    checks = []
    check_snapshots(checks)
    check_failures(checks)
    check_widths(checks)
    initialized_after = torch_npu.npu._initialized
    checks.append({"name": "no_npu_initialization",
                   "passed": not initialized_before and not initialized_after})
    source = Path(__file__).resolve().parents[2] / (
        "tilert/models/glm_5_2_ascend/cann_trace.py"
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU Module hook ordering, snapshots and cleanup; no NPU "
                 "execution or GLM numerical parity",
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "npu_initialized_before": initialized_before,
        "npu_initialized_after": initialized_after,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), source)
        },
        "checks": checks, "passed": all(item["passed"] for item in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(checks),
                      "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
