"""Locate a repeated M32 forward's first difference without timing it."""

import argparse
from functools import partial, wraps
import inspect
import json
import os
from pathlib import Path
import sys
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests/glm52")]

from run_cann_model import capture_mlp, digest, load_capture  # noqa: E402
from frozen_teacher import load_teacher  # noqa: E402
from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint  # noqa: E402
from tilert.pd_vllm.decode_timing import atomic_json  # noqa: E402


class Boundaries:
    """Own device copies; read the CPU only after the whole forward finishes."""

    def __init__(self, target):
        self.target = target
        self.values = {}
        self.handles = []
        self.overrides = []
        self.scope = None
        self.reductions = {}

    def save(self, tag, value):
        import torch

        if isinstance(value, torch.Tensor):
            if tag in self.values:
                raise ValueError("Boundary observed twice: " + tag)
            self.values[tag] = value.detach().clone()
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.save(f"{tag}.{index}", item)
        elif isinstance(value, dict):
            for key, item in value.items():
                self.save(f"{tag}.{key}", item)

    def before(self, tag, module, args, kwargs):
        self.save(tag + ".in", args)
        self.save(tag + ".kwargs", kwargs)

    def after(self, tag, module, args, kwargs, output):
        self.save(tag + ".out", output)

    def module(self, tag, module):
        self.handles.append(module.register_forward_pre_hook(
            partial(self.before, tag), with_kwargs=True))
        self.handles.append(module.register_forward_hook(
            partial(self.after, tag), with_kwargs=True))

    def enter_scope(self, tag, module, args):
        self.scope = tag

    def leave_scope(self, module, args, output):
        self.scope = None

    def replace(self, target, name, replacement):
        previous = vars(target).get(name)
        existed = name in vars(target)
        setattr(target, name, replacement)
        self.overrides.append((target, name, existed, previous))

    def attention(self, tag, original):
        signature = inspect.signature(original)

        @wraps(original)
        def observed(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            for name in ("ql_nope", "q_pe", "topk_indices"):
                self.save(f"{tag}.{name}", arguments[name])
            result = original(*args, **kwargs)
            self.save(tag + ".out", result)
            return result

        return observed

    def reduce(self, original):
        @wraps(original)
        def observed(value):
            scope = self.scope
            if scope is not None:
                index = self.reductions.get(scope, 0)
                self.reductions[scope] = index + 1
                tag = f"{scope}.tp.{index}"
                self.save(tag + ".in", value)
            result = original(value)
            if scope is not None:
                self.save(tag + ".out", result)
            return result

        return observed

    def method(self, tag, original):
        @wraps(original)
        def observed(*args, **kwargs):
            self.save(tag + ".in", args)
            self.save(tag + ".kwargs", kwargs)
            result = original(*args, **kwargs)
            self.save(tag + ".out", result)
            return result

        return observed

    def __enter__(self):
        try:
            for index in (3, 4, 5):
                layer = self.target.model.model.layers[index]
                tag = f"layer.{index}"
                self.module(tag + ".input_norm", layer.input_layernorm)
                self.module(tag + ".post_norm", layer.post_attention_layernorm)
                for name, module in (("attention", layer.self_attn),
                                     ("mlp", layer.mlp)):
                    self.module(tag + "." + name, module)
                    self.handles.append(module.register_forward_pre_hook(
                        partial(self.enter_scope, tag + "." + name)))
                    self.handles.append(module.register_forward_hook(
                        self.leave_scope))
                attention = layer.self_attn.mla_attn.mla_attn.impl
                name = "_execute_sparse_flash_attention_process"
                self.replace(attention, name, self.attention(
                    tag + ".sfa", getattr(attention, name)))
                if index == 3:
                    runner = layer.mlp.experts
                    for owner, name in ((runner, "_forward_impl"),
                                        (runner, "_compute_router_logits"),
                                        (runner.routed_experts, "forward_impl"),
                                        (runner.routed_experts, "_select_experts")):
                        self.replace(owner, name, self.method(
                            tag + ".moe." + name, getattr(owner, name)))
            group = self.target._tp_sum_group
            self.replace(group, "_all_reduce_out_place", self.reduce(
                group._all_reduce_out_place))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def begin(self):
        self.values = {}
        self.reductions = {}
        self.scope = None

    def readback(self, step):
        import torch

        self.save("logits", step.logits)
        self.save("token_ids", step.token_ids)
        torch.npu.synchronize()
        return {tag: value.cpu() for tag, value in self.values.items()}

    def __exit__(self, exc_type, exc_value, error):
        for handle in self.handles:
            handle.remove()
        for target, name, existed, previous in reversed(self.overrides):
            if existed:
                setattr(target, name, previous)
            else:
                delattr(target, name)
        self.handles.clear()
        self.overrides.clear()


def compare(previous, current):
    import torch

    if previous.keys() != current.keys():
        raise ValueError("Repeated forward boundary inventories differ")
    results = []
    for tag, left in previous.items():
        right = current[tag]
        equal = torch.equal(left, right)
        entry = {"tag": tag, "equal": equal, "shape": list(left.shape),
                 "dtype": str(left.dtype),
                 "finite": bool(torch.isfinite(left).all()
                                and torch.isfinite(right).all())}
        if not equal:
            different = left != right
            entry.update(count=int(different.sum()),
                         first_index=different.nonzero()[0].tolist(),
                         max_abs=float((left.float() - right.float()).abs().max()))
        results.append(entry)
    return results


def run(args, report):
    import torch
    import torch_npu  # noqa: F401
    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    rank = report["rank"]
    assert int(os.environ["WORLD_SIZE"]) == 16
    assert int(os.environ["LOCAL_RANK"]) == rank
    torch.set_num_threads(1)
    properties = torch.npu.get_device_properties(rank)
    assert properties.name.upper() in ("ASCEND910_9392", "ASCEND910_9382")
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    prefixes, _, inputs = load_capture(args.capture_case, rank, checkpoint)
    assert inputs["initial_position"] == 8192
    teacher = load_teacher(args.teacher, args.capture_case, 32, args.capacity)
    tokens = teacher["teacher_tokens"]
    target = CannModel(args.checkpoint, f"npu:{rank}", rank,
                       capacity=args.capacity, gpu_memory_utilization=0.85,
                       attention_kernel="sfa", fp32_head=True, graph_width=32,
                       compile_target=False, super_kernel=False)
    report.update(device=str(properties), inputs=inputs, teacher=teacher,
                  source_sha256=target.source_metadata(),
                  checkpoint_sha256=checkpoint)

    def prepare(clear):
        target.reset()
        if clear:
            target.storage.zero_()
        target.inject(prefixes)
        torch.npu.synchronize()
        assert target.position == 8192

    try:
        prepare(True)
        for token in tokens:
            target.step(token)
        torch.npu.synchronize()
        previous = first = None
        report["phases"] = []
        with Boundaries(target) as trace:
            for phase, clear in (("wide", True), ("reset", False), ("clear", True)):
                prepare(clear)
                trace.begin()
                with capture_mlp(target, 3) as stages:
                    step = target.verify(tokens, full_logits=True)
                    trace.save("layer.3.mlp_stages", stages)
                values = trace.readback(step)
                path = args.output / f"rank-{rank}-{phase}.pt"
                torch.save(values, path)
                entry = {"phase": phase, "clear_tail": clear,
                         "artifact_sha256": digest(path),
                         "position": target.position, "boundaries": len(values)}
                if previous is not None:
                    entry["vs_previous"] = compare(previous, values)
                    entry["vs_wide"] = compare(first, values)
                else:
                    first = values
                previous = values
                report["phases"].append(entry)
        target.reset()
        report.update(status="captured", final_position=target.position)
    finally:
        target.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "capture-case", "teacher", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--capacity", type=int, default=9344)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    path = args.output / f"rank-{rank}.json"
    if path.exists() or (args.output / f"rank-{rank}-wide.pt").exists():
        parser.error("Preserve existing captures; use a new output directory")
    report = {"schema": "cann-reset-boundaries-v1", "rank": rank,
              "source_commit": args.source_commit, "status": "error",
              "script_sha256": digest(Path(__file__)),
              "scope": "instrumented eager TP16 M32, layers 3/4/5, one frozen 8K case",
              "precision_verified": False, "performance_measured": False,
              "formal_precision_gate_passed": False}
    try:
        run(args, report)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    atomic_json(path, report)
    print(json.dumps({"rank": rank, "status": report["status"]}))
    return 0 if report["status"] == "captured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
