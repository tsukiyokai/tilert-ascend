"""Compare host-drafted token staging against a Git revision on CPU."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend import cann_model, mtp_round  # noqa: E402


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def baseline_module(name, revision):
    relative = f"tilert/models/glm_5_2_ascend/{name}.py"
    source = git("show", f"{revision}:{relative}")
    qualified = f"tilert.models.glm_5_2_ascend._baseline_{name}"
    module = ModuleType(qualified)
    sys.modules[qualified] = module
    exec(compile(source, f"{revision}:{relative}", "exec"), module.__dict__)
    return module


class AscendingDrafter:
    def __init__(self):
        self.pending = 1

    def propose(self, count):
        return list(range(self.pending + 1, self.pending + count + 1))

    def commit(self, tokens):
        self.pending = tokens[-1]


class CpuTarget:
    def __init__(self, module, capacity):
        self.module = module
        self.device = "cpu"
        self.position = 8192
        self.capacity = capacity

    def verify(self, tokens, full_logits=False):
        values = self.module.device_tokens(tokens, self.device)
        self.position += len(tokens)
        return SimpleNamespace(token_ids=values + 1)

    def host_tokens(self, *groups):
        return self.module.CannModel.host_tokens(self, *groups)

    def truncate(self, length):
        self.position = length


def make_executor(modules, width, rounds):
    model, executor = modules
    target = CpuTarget(model, 8192 + width * rounds + 64)
    result = executor.MtpRoundExecutor(
        target, None, width, width * rounds + 1, drafter=AscendingDrafter(),
    )
    result.start(1, None)
    return result


def measure(modules, width, rounds):
    executor = make_executor(modules, width, rounds)
    start = perf_counter_ns()
    for _ in range(rounds):
        result = executor.advance()
    elapsed = perf_counter_ns() - start
    assert result.commit.accepted_tokens[-1] == width * rounds + 1
    assert executor.target.position == 8192 + width * rounds
    return elapsed / rounds / 1000


def structure(modules, width):
    executor = make_executor(modules, width, 1)
    counts = Counter()

    def count(name, operation):
        def call(*args, **kwargs):
            counts[name] += 1
            return operation(*args, **kwargs)
        return call

    with (
        patch.object(torch, "tensor", count("tensor", torch.tensor)),
        patch.object(torch, "as_tensor", count("as_tensor", torch.as_tensor)),
        patch.object(torch, "stack", count("stack", torch.stack)),
        patch.object(torch, "cat", count("cat", torch.cat)),
    ):
        result = executor.advance()
    return {
        "torch_calls": dict(counts),
        "candidates": list(result.candidates),
        "predictions": list(result.target_predictions),
        "accepted": list(result.commit.accepted_tokens),
    }


def stats(values):
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p90": ordered[round(0.9 * (len(ordered) - 1))],
        "p99": ordered[round(0.99 * (len(ordered) - 1))],
        "std": statistics.pstdev(ordered),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    arms = {
        "baseline": tuple(baseline_module(name, args.baseline_ref)
                          for name in ("cann_model", "mtp_round")),
        "candidate": (cann_model, mtp_round),
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU torch staging and scalar target; no NPU/H2D/TPOT timing",
        "hostname": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "command": sys.argv,
        "baseline_commit": git("rev-parse", args.baseline_ref),
        "candidate_parent": git("rev-parse", "HEAD"),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(cann_model.__file__),
                         Path(mtp_round.__file__),
                         Path(mtp_round.__file__).with_name("mtp_state.py"))
        },
        "warmup_batches": args.warmup, "samples": args.samples,
        "rounds_per_sample": args.rounds,
        "outlier_rule": "none; alternating paired arms, all samples retained",
        "precision_verified": False, "performance_measured": False,
        "results": [],
    }
    for width in args.widths:
        operations = {name: structure(modules, width)
                      for name, modules in arms.items()}
        for field in ("candidates", "predictions", "accepted"):
            assert operations["baseline"][field] == (
                operations["candidate"][field])
        samples = []
        for number in range(-args.warmup, args.samples):
            names = list(arms) if number % 2 else list(reversed(arms))
            pair = {name: measure(arms[name], width, args.rounds)
                    for name in names}
            if number >= 0:
                samples.append(pair)
        item = {
            "width": width, "structure": operations,
            "round_cpu_us": {name: stats([sample[name] for sample in samples])
                             for name in arms},
            "paired_saving_us": stats([
                sample["baseline"] - sample["candidate"] for sample in samples
            ]),
            "samples_us": samples,
        }
        report["results"].append(item)
        print(json.dumps({key: value for key, value in item.items()
                          if key not in ("structure", "samples_us")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
