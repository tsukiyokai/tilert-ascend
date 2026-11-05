"""Pair real A3 token staging in isolated eager-runtime processes."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREFIX = "TRT_RESULT "


def file_identity(path):
    path = Path(path)
    return {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def baseline_module(name, directory):
    path = directory / f"{name}.py"
    qualified = f"tilert.models.glm_5_2_ascend._baseline_{name}"
    module = ModuleType(qualified)
    module.__file__ = str(path)
    sys.modules[qualified] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def respond(value):
    print(PREFIX + json.dumps(value), flush=True)


def worker(args):
    import torch
    import torch_npu  # noqa: F401

    from tilert.models.glm_5_2_ascend import cann_model, mtp_round
    from bench import AscendingDrafter

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device_name = torch.npu.get_device_name(args.device)
    if not any(sku in device_name for sku in ("9392", "9382")):
        raise ValueError(f"Real Ascend A3 required, got {device_name}")
    modules = (cann_model, mtp_round)
    if args.worker == "baseline":
        modules = tuple(baseline_module(name, args.baseline_root)
                        for name in ("cann_model", "mtp_round"))
    model, executor_module = modules

    class Target:
        position = 8192
        capacity = 9344
        device = args.device

        def verify(self, tokens, full_logits=False):
            values = model.device_tokens(tokens, self.device)
            self.position += len(tokens)
            return SimpleNamespace(token_ids=values + 1)

        def host_tokens(self, *groups):
            return model.CannModel.host_tokens(self, *groups)

        def truncate(self, length):
            self.position = length

    pressure = torch.zeros(args.pressure_mib * 1024 * 1024,
                           dtype=torch.uint8, device=args.device)
    stream = torch.npu.Stream(device=args.device)
    respond({"ready": True, "device_name": device_name,
             "pid": os.getpid()})
    with torch.inference_mode(), torch.npu.stream(stream):
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "quit":
                break
            if request["command"] == "identity":
                paths = set()
                for entry in Path("/proc/self/maps").read_text().splitlines():
                    path = entry.split()[-1]
                    if (path.startswith("/") and ".so" in path
                            and ("/Ascend/" in path or "/torch_npu/" in path)
                            and Path(path).is_file()):
                        paths.add(path)
                respond({"packages": {name: version(name)
                                      for name in ("torch", "torch-npu")},
                         "modules": [file_identity(mod.__file__)
                                     for mod in modules],
                         "libraries": [file_identity(path)
                                       for path in sorted(paths)],
                         "cache_root": str(Path.cwd()),
                         "cache_files": [file_identity(path) for path in
                                         sorted(Path.cwd().rglob("*"))
                                         if path.is_file()]})
                continue
            width = request["width"]
            first = 1 + request["generation"] * 113 % (154880 - width - 2)
            target = Target()
            drafter = AscendingDrafter()
            drafter.pending = first
            executor = executor_module.MtpRoundExecutor(
                target, None, width, width + 1, drafter=drafter,
            )
            executor.start(first, None)
            pressure.add_(1)
            torch.npu.synchronize()
            start = perf_counter_ns()
            result = executor.advance()
            torch.npu.synchronize()
            elapsed = perf_counter_ns() - start
            candidates = tuple(range(first, first + width))
            predictions = tuple(range(first + 1, first + width + 1))
            assert result.candidates == candidates
            assert result.target_predictions == predictions
            assert result.commit.accepted_tokens == predictions
            assert target.position == 8192 + width
            assert executor.state.main_length == target.position
            respond({"wall_us": elapsed / 1000,
                     "signature": hashlib.sha256(json.dumps([
                         result.candidates, result.target_predictions,
                         result.commit.accepted_tokens, target.position,
                     ]).encode()).hexdigest()})


def exchange(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    return receive(process)


def receive(process):
    for line in process.stdout:
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX):])
    raise RuntimeError(f"Worker exited with {process.poll()}")


def stats(values):
    values = sorted(values)
    return {"p50": statistics.median(values),
            "p90": values[round(0.9 * (len(values) - 1))],
            "p95": values[round(0.95 * (len(values) - 1))],
            "p99": values[round(0.99 * (len(values) - 1))],
            "std": statistics.pstdev(values)}


def benchmark(args):
    from tilert.pd_vllm.decode_timing import atomic_json

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "A3 eager integer staging and scalar Add target round",
        "hypothesis": "Batch host candidates reduces width32 round p50 "
                      "by at least 100 us against ff2ab6c",
        "hostname": platform.node(), "command": sys.argv,
        "candidate_parent": args.source_commit,
        "baseline_commit": (args.baseline_root / "commit.txt").read_text().strip(),
        "source": file_identity(__file__),
        "shared_state": file_identity(
            ROOT / "tilert/models/glm_5_2_ascend/mtp_state.py"),
        "protocol": {"clock": "synchronized host perf_counter_ns",
                     "profiler": False, "graph_capture": False,
                     "warmup_pairs": args.warmup, "pairs": args.samples,
                     "pressure_mib_per_arm": args.pressure_mib,
                     "pressure": "read/write whole buffer before each timed round",
                     "order": "AB/BA alternating", "outlier_rule": "none",
                     "truth": "all candidates, predictions, commits and lengths exact"},
        "precision_verified": False, "performance_measured": False,
        "workers": {}, "results": [],
    }
    processes = {}
    logs = []
    try:
        for arm in ("baseline", "candidate"):
            directory = args.output.parent / f"cache-{arm}"
            directory.mkdir()
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            for name in ("ASCEND_CACHE_PATH", "XDG_CACHE_HOME",
                         "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                         "VLLM_CACHE_ROOT"):
                cache = directory / name.lower()
                cache.mkdir()
                env[name] = str(cache)
            log = (args.output.parent / f"{arm}.log").open("w")
            logs.append(log)
            process = subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()),
                "--worker", arm, "--baseline-root", str(args.baseline_root),
                "--device", args.device, "--pressure-mib", str(args.pressure_mib),
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, cwd=directory, env=env)
            processes[arm] = process
            report["workers"][arm] = receive(process)
        for width in args.widths:
            samples = []
            for number in range(-args.warmup, args.samples):
                order = ("baseline", "candidate")
                if number % 2:
                    order = order[::-1]
                pair = {arm: exchange(processes[arm], {
                    "command": "measure", "width": width,
                    "generation": number + args.warmup,
                }) for arm in order}
                assert pair["baseline"]["signature"] == (
                    pair["candidate"]["signature"])
                if number >= 0:
                    samples.append({"pair": number, "order": list(order),
                                    **pair})
            values = {arm: [sample[arm]["wall_us"] for sample in samples]
                      for arm in processes}
            savings = [a - b for a, b in
                       zip(values["baseline"], values["candidate"])]
            item = {"width": width, "wall_us": {
                arm: stats(timings) for arm, timings in values.items()},
                "paired_saving_us": stats(savings),
                "wins": sum(value > 0 for value in savings),
                "samples": samples}
            report["results"].append(item)
            atomic_json(args.output, report)
            print(json.dumps({key: value for key, value in item.items()
                              if key != "samples"}), flush=True)
        for arm, process in processes.items():
            report["workers"][arm]["identity"] = exchange(
                process, {"command": "identity"})
        report["status"] = "pass"
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.stdin.write('{"command": "quit"}\n')
                process.stdin.flush()
                process.wait(timeout=30)
        for log in logs:
            log.close()
        atomic_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("baseline", "candidate"))
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--pressure-mib", type=int, default=256)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pressure_mib <= 192:
        parser.error("L2 pressure must exceed 192 MiB")
    if args.worker:
        worker(args)
    else:
        if args.output is None or args.source_commit is None:
            parser.error("--output and --source-commit are required")
        if args.samples < 30:
            parser.error("At least 30 paired samples are required")
        benchmark(args)


if __name__ == "__main__":
    main()
