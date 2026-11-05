"""Compare stateful decode with the CPU Hugging Face GLM-MoE-DSA model."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial, wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from torch.overrides import TorchFunctionMode


PROMPT_LENGTH = 8
DECODE_STEPS = 4
CACHE_CAPACITY = 32
MAX_NRMSE = 1e-2
MIN_COSINE = 0.999
MAX_LOGIT_ERROR = 0.1


class StableTopk(TorchFunctionMode):
    """Keep HF scoring intact and resolve equal scores by ascending index."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in (torch.topk, torch.Tensor.topk):
            return func(*args, **kwargs)
        scores = args[0] if args else kwargs["input"]
        count = args[1] if len(args) > 1 else kwargs["k"]
        dim = kwargs.get("dim", args[2] if len(args) > 2 else -1)
        largest = kwargs.get("largest", args[3] if len(args) > 3 else True)
        indices = scores.argsort(dim=dim, descending=largest, stable=True)
        indices = indices.narrow(dim, 0, count)
        return torch.return_types.topk((scores.gather(dim, indices), indices))


def use_stable_topk(forward):
    @wraps(forward)
    def wrapped(*args, **kwargs):
        with StableTopk():
            return forward(*args, **kwargs)

    return wrapped


@dataclass
class Snapshot:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    indices: tuple[torch.Tensor, ...]
    routes: tuple[torch.Tensor | None, ...]
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    lengths: tuple[int, ...]
    position: int


def cpu_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


class Checks:
    def __init__(self, report: dict[str, Any], exact: bool):
        self.items = report["checks"]
        self.exact = exact

    def equal(self, name: str, actual: Any, expected: Any) -> None:
        self.items.append({
            "name": name,
            "passed": actual == expected,
            "actual": actual,
            "expected": expected,
        })

    def selection(
        self,
        name: str,
        actual: torch.Tensor | None,
        expected: torch.Tensor | None,
        upper_bound: int,
    ) -> None:
        actual_ids = None if actual is None else actual.reshape(-1).tolist()
        expected_ids = (
            None if expected is None else expected.reshape(-1).tolist()
        )
        if actual_ids is None or expected_ids is None:
            self.equal(name, actual_ids, expected_ids)
            return
        valid = all(0 <= item < upper_bound for item in actual_ids)
        unique = len(actual_ids) == len(set(actual_ids))
        self.items.append({
            "name": name,
            "passed": (
                valid and unique and sorted(actual_ids) == sorted(expected_ids)
            ),
            "actual": actual_ids,
            "expected": expected_ids,
            "valid_range": valid,
            "unique": unique,
        })

    def tensor(
        self, name: str, actual: torch.Tensor, expected: torch.Tensor
    ) -> None:
        dtype_matches = actual.dtype == expected.dtype
        actual_dtype = str(actual.dtype)
        expected_dtype = str(expected.dtype)
        actual = cpu_copy(actual).float()
        expected = cpu_copy(expected).float()
        item = {
            "name": name,
            "passed": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": actual_dtype,
            "expected_dtype": expected_dtype,
            "criterion": "exact" if self.exact else "nrmse_and_cosine",
        }
        self.items.append(item)
        if actual.shape != expected.shape:
            return
        finite = bool(torch.isfinite(actual).all())
        finite = finite and bool(torch.isfinite(expected).all())
        item["finite"] = finite
        if not finite:
            return
        difference = (actual - expected).abs()
        rmse = float(difference.square().mean().sqrt())
        reference_norm = float(expected.norm())
        actual_norm = float(actual.norm())
        nrmse = float(difference.norm()) / max(reference_norm, 1e-12)
        if reference_norm == 0 or actual_norm == 0:
            cosine = float(reference_norm == actual_norm)
        else:
            cosine = float((actual * expected).sum()) / (actual_norm * reference_norm)
            cosine = min(1.0, max(-1.0, cosine))
        exact_equal = dtype_matches and bool(torch.equal(actual, expected))
        within_budget = nrmse <= MAX_NRMSE and cosine >= MIN_COSINE
        if name.endswith(".logits"):
            within_budget = within_budget and float(difference.max()) <= MAX_LOGIT_ERROR
        item.update({
            "passed": exact_equal if self.exact else dtype_matches and within_budget,
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "rmse": rmse,
            "normalized_rmse": nrmse,
            "cosine": cosine,
            "exact_equal": exact_equal,
            "reference_abs_max": float(expected.abs().max()),
        })
        worst = int(difference.reshape(-1).argmax())
        item["worst_flat_index"] = worst
        item["actual_at_worst"] = float(actual.reshape(-1)[worst])
        item["expected_at_worst"] = float(expected.reshape(-1)[worst])


class ReferenceTrace:
    def __init__(self, model: Any):
        self.layers = model.model.layers
        self.handles = []
        self.clear()

    def clear(self) -> None:
        count = len(self.layers)
        self.indices = [None] * count
        self.indexer_indices = [None] * count
        self.routes = [None] * count
        self.indexer_calls = [0] * count
        self.router_calls = [0] * count
        self.bias_changed_selection = False

    def __enter__(self) -> ReferenceTrace:
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.register_forward_hook(
                partial(self.record_attention, index)
            ))
            if layer.self_attn.indexer is not None:
                self.handles.append(
                    layer.self_attn.indexer.register_forward_hook(
                        partial(self.record_indexer, index)
                    )
                )
            if hasattr(layer.mlp, "gate"):
                self.handles.append(layer.mlp.gate.register_forward_hook(
                    partial(self.record_router, index)
                ))
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def record_attention(
        self, index: int, _module: Any, _inputs: Any, output: Any
    ) -> None:
        self.indices[index] = cpu_copy(output[2][0, -1])

    def record_indexer(
        self, index: int, _module: Any, _inputs: Any, output: Any
    ) -> None:
        self.indexer_calls[index] += 1
        self.indexer_indices[index] = cpu_copy(output[0, -1])

    def record_router(
        self, index: int, _module: Any, _inputs: Any, output: Any
    ) -> None:
        self.router_calls[index] += 1
        self.routes[index] = cpu_copy(output[2][0])
        unbiased = output[0][0].sigmoid().argsort(descending=True, stable=True)
        unbiased = unbiased[: output[2].shape[-1]]
        if not torch.equal(unbiased.sort().values, output[2][0].sort().values):
            self.bias_changed_selection = True


def reference_snapshot(output: Any, trace: ReferenceTrace) -> Snapshot:
    cache = output.past_key_values
    return Snapshot(
        logits=cpu_copy(output.logits[:, -1]),
        hidden_states=tuple(
            cpu_copy(hidden[:, -1]) for hidden in output.hidden_states
        ),
        indices=tuple(trace.indices),
        routes=tuple(trace.routes),
        keys=tuple(cpu_copy(layer.keys) for layer in cache.layers),
        values=tuple(cpu_copy(layer.values) for layer in cache.layers),
        lengths=tuple(layer.get_seq_length() for layer in cache.layers),
        position=cache.get_seq_length(),
    )


def engine_snapshot(engine: Any, result: Any) -> Snapshot:
    expanded = [
        engine.expanded_cache(index) for index in range(len(engine.cache))
    ]
    return Snapshot(
        logits=cpu_copy(result.logits),
        hidden_states=tuple(cpu_copy(value) for value in result.hidden_states),
        indices=tuple(cpu_copy(value) for value in result.indices),
        routes=tuple(
            None if value is None else cpu_copy(value)
            for value in result.routes
        ),
        keys=tuple(cpu_copy(key) for key, _value in expanded),
        values=tuple(cpu_copy(value) for _key, value in expanded),
        lengths=tuple(int(layer.length) for layer in engine.cache),
        position=int(engine.position),
    )


def compare_snapshots(
    checks: Checks,
    name: str,
    actual: Snapshot,
    expected: Snapshot,
    expert_count: int,
) -> None:
    checks.equal(f"{name}.position", actual.position, expected.position)
    checks.equal(f"{name}.cache_lengths", actual.lengths, expected.lengths)
    checks.equal(
        f"{name}.hidden_count",
        len(actual.hidden_states),
        len(expected.hidden_states),
    )
    for index, (actual_hidden, expected_hidden) in enumerate(zip(
        actual.hidden_states, expected.hidden_states
    )):
        label = "embedding" if index == 0 else f"layer_{index - 1}"
        checks.tensor(f"{name}.hidden.{label}", actual_hidden, expected_hidden)
    for field in ("indices", "routes", "keys", "values"):
        actual_values = getattr(actual, field)
        expected_values = getattr(expected, field)
        checks.equal(
            f"{name}.{field}_count", len(actual_values), len(expected_values)
        )
        for index, (actual_value, expected_value) in enumerate(zip(
            actual_values, expected_values
        )):
            label = f"{name}.layer_{index}.{field}"
            if field in ("indices", "routes"):
                bound = (
                    expected.position if field == "indices" else expert_count
                )
                checks.selection(label, actual_value, expected_value, bound)
            else:
                checks.tensor(label, actual_value, expected_value)
    checks.tensor(f"{name}.logits", actual.logits, expected.logits)
    checks.equal(
        f"{name}.greedy_token",
        int(actual.logits.argmax(-1).item()),
        int(expected.logits.argmax(-1).item()),
    )


def check_reference_trace(
    checks: Checks, name: str, trace: ReferenceTrace, config: Any
) -> None:
    previous_indices = None
    for index, mode in enumerate(config.indexer_types):
        label = f"{name}.reference.layer_{index}"
        checks.equal(
            f"{label}.indexer_calls", trace.indexer_calls[index],
            int(mode == "full"),
        )
        checks.equal(
            f"{label}.router_calls", trace.router_calls[index],
            int(config.mlp_layer_types[index] == "sparse"),
        )
        expected = trace.indexer_indices[index]
        if mode == "shared":
            expected = previous_indices
        checks.selection(
            f"{label}.index_source", trace.indices[index], expected,
            CACHE_CAPACITY,
        )
        previous_indices = trace.indices[index]


def run_sequence(
    report: dict[str, Any],
    checks: Checks,
    model: Any,
    engine: Any,
    prompt: list[int],
    baseline: list[Snapshot] | None = None,
    name: str = "initial",
) -> list[Snapshot]:
    if baseline is not None:
        name = "reset_replay"
    run = {"name": name, "input_tokens": [], "steps": [], "bias_changed_steps": 0}
    report["runs"].append(run)
    snapshots = []
    cache = None
    next_token = prompt[0]
    replay_checks = Checks(report, exact=True)
    with ReferenceTrace(model) as trace, torch.inference_mode():
        for index in range(len(prompt) + DECODE_STEPS):
            token = prompt[index] if index < len(prompt) else next_token
            if baseline is not None:
                token = report["runs"][0]["input_tokens"][index]
            run["input_tokens"].append(token)
            step_name = f"{name}.step_{index}"
            report["stage"] = step_name
            trace.clear()
            output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = output.past_key_values
            run["bias_changed_steps"] += int(trace.bias_changed_selection)
            expected = reference_snapshot(output, trace)
            result = engine.step(token, trace=True)
            actual = engine_snapshot(engine, result)
            snapshots.append(actual)
            checks.equal(
                f"{step_name}.consumed_length", actual.position, index + 1
            )
            checks.equal(
                f"{step_name}.reference_length", expected.position, index + 1
            )
            check_reference_trace(checks, step_name, trace, model.config)
            compare_snapshots(
                checks, step_name, actual, expected,
                model.config.n_routed_experts,
            )
            if baseline is not None:
                compare_snapshots(
                    replay_checks, f"{step_name}.replay", actual,
                    baseline[index],
                    model.config.n_routed_experts,
                )
            next_token = int(expected.logits.argmax(-1).item())
            run["steps"].append({
                "step": index,
                "phase": "prompt" if index < len(prompt) else "decode",
                "input_token": token,
                "consumed_tokens": actual.position,
                "reference_next_token": next_token,
                "engine_next_token": int(actual.logits.argmax(-1).item()),
            })
    run["generated_tokens"] = [
        step["reference_next_token"] for step in run["steps"][len(prompt) - 1:]
    ]
    run["generated_token_count"] = len(run["generated_tokens"])
    run["consumed_tokens"] = len(run["input_tokens"])
    return snapshots


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@torch.no_grad()
def initialize_reference(model: Any) -> dict[str, Any]:
    rotary = model.model.rotary_emb
    frequencies, _scale = rotary.compute_default_rope_parameters(model.config)
    # Casting model weights to BF16 also casts nonpersistent RoPE buffers.
    rotary.inv_freq = frequencies
    rotary.original_inv_freq = frequencies.clone()
    norm_count = 0
    for module in model.modules():
        if type(module).__name__ == "GlmMoeDsaRMSNorm":
            gamma = torch.linspace(0.75, 1.25, module.weight.numel())
            module.weight.copy_(gamma.to(torch.bfloat16))
            norm_count += 1
        if type(module).__name__ in ("GlmMoeDsaIndexer", "GlmMoeDsaTopkRouter"):
            module.forward = use_stable_topk(module.forward)
    routers = [
        layer.mlp.gate for layer in model.model.layers
        if hasattr(layer.mlp, "gate")
    ]
    for router in routers:
        router.e_score_correction_bias = torch.linspace(
            -0.125, 0.125, router.weight.shape[0], dtype=torch.float32
        )
    return {
        "rope_buffers_restored_to_fp32_after_weight_cast": True,
        "rope_inv_freq_dtype": str(rotary.inv_freq.dtype),
        "rms_gamma": "BF16 linspace(0.75, 1.25) per normalization module",
        "rms_gamma_modules": norm_count,
        "router_weights_nonzero": [
            bool(torch.count_nonzero(router.weight)) for router in routers
        ],
        "router_correction_bias": "FP32 linspace(-0.125, 0.125) per routed layer",
    }


def execute(args: argparse.Namespace, report: dict[str, Any]) -> None:
    from transformers import GlmMoeDsaForCausalLM

    from .config import DecodeConfig
    from .engine import DecodeEngine

    if args.device.startswith("npu:"):
        importlib.import_module("torch_npu")
        from tilert import load_backend, tilert_init

        load_backend("glm5_2_ascend")
        torch.npu.set_device(args.device)
        tilert_init()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = DecodeConfig.smoke()
    hf_config = config.to_hf_config()
    hf_config._attn_implementation = "eager"
    hf_config._experts_implementation = "eager"
    report["config"] = hf_config.to_dict()
    report["stage"] = "reference_initialization"
    model = GlmMoeDsaForCausalLM(hf_config).to(
        device="cpu", dtype=torch.bfloat16
    ).eval()
    report["reference_initialization"] = initialize_reference(model)
    report["reference_parameter_dtypes"] = sorted({
        str(parameter.dtype) for parameter in model.parameters()
    })
    report["stage"] = "engine_initialization"
    engine = DecodeEngine(
        config, model.state_dict(), device=args.device, capacity=CACHE_CAPACITY
    )
    report["engine"] = {
        "rms_backend": "TileLang Ascend" if args.device.startswith("npu:") else "CPU reference",
        "weight_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in engine.weights.values()
        ),
        "cache_bytes": sum(
            tensor.numel() * tensor.element_size()
            for layer in engine.cache
            for tensor in (layer.latent, layer.rope, layer.projected, layer.index_keys)
            if tensor is not None
        ),
    }
    checks = Checks(report, exact=args.device == "cpu")
    checks.equal("initial.position", int(engine.position), 0)
    checks.equal(
        "initial.cache_lengths", [int(layer.length) for layer in engine.cache],
        [0] * hf_config.num_hidden_layers,
    )
    prompt = [
        (11 + 17 * index) % hf_config.vocab_size
        for index in range(PROMPT_LENGTH)
    ]
    report["prompt_tokens"] = prompt
    snapshots = run_sequence(report, checks, model, engine, prompt)
    report["stage"] = "reset"
    engine.reset()
    checks.equal("reset.position", int(engine.position), 0)
    checks.equal(
        "reset.cache_lengths", [int(layer.length) for layer in engine.cache],
        [0] * hf_config.num_hidden_layers,
    )
    run_sequence(report, checks, model, engine, prompt, baseline=snapshots)
    engine.reset()
    run_sequence(report, checks, model, engine, list(reversed(prompt[:3])), name="short_request")
    checks.equal(
        "correction_bias_changes_selection",
        any(run["bias_changed_steps"] > 0 for run in report["runs"]),
        True,
    )
    report["engine"]["rms_calls"] = engine.norm_calls
    if args.device.startswith("npu:"):
        report["engine"]["peak_tensor_memory_bytes"] = torch.npu.max_memory_allocated()
        report["engine"]["peak_reserved_memory_bytes"] = torch.npu.max_memory_reserved()
    report["stage"] = "complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu:0"), default="cpu")
    parser.add_argument("--output", "--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stage": "imports",
        "device": args.device,
        "reference_device": "cpu",
        "seed": args.seed,
        "dtype": "bfloat16",
        "tolerances": {
            "cpu_reference": "exact",
            "npu_reference": {
                "max_nrmse": MAX_NRMSE,
                "min_cosine": MIN_COSINE,
                "max_logit_abs_error": MAX_LOGIT_ERROR,
            },
            "reset_replay": "exact",
            "selection_sets": "exact",
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "torch_npu": package_version("torch_npu"),
        },
        "scope": {
            "model": "reduced GLM-5.2 structure with random weights",
            "batch_size": 1,
            "prompt_tokens": PROMPT_LENGTH,
            "decode_steps": DECODE_STEPS,
            "cache_capacity": CACHE_CAPACITY,
            "reference_attention": "eager",
            "reference_experts": "eager",
            "selection_ties": "equal scores prefer lower context position or expert ID",
            "reference_selection_adapter": "HF indexer/router topk uses stable descending argsort",
            "input_schedule": "8 prompt tokens, then 4 HF greedy tokens",
            "comparison": "same-prefix teacher forcing and reset replay",
            "short_request": (
                "reset after 12 consumed tokens, then a distinct 3-token prompt plus 4 decode steps"
            ),
            "reset_replay_tolerance": "exact tensor equality",
            "hidden_states": (
                "embedding and layer outputs; final layer normalized"
            ),
            "checkpoint_loaded": False,
            "mtp": False,
            "distributed": False,
            "performance_measured": False,
        },
        "checks": [],
        "runs": [],
    }
    try:
        execute(args, report)
    except Exception as error:
        report["status"] = "error"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    failed = [item for item in report["checks"] if not item["passed"]]
    if report["status"] != "error":
        report["status"] = "fail" if failed else "pass"
    report["passed"] = report["status"] == "pass"
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = len(failed)
    report["first_failure"] = failed[0] if failed else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "checks_total": report["checks_total"],
        "checks_failed": report["checks_failed"],
        "report": str(args.output.resolve()),
        "first_failure": report["first_failure"],
    }))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
