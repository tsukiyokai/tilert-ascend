"""Capture one precision-gated TP16 decode request, without TPOT timing.

Three unprofiled requests warm the same prefix and production session.
Each rank profiles its executing thread. Raw Level1 CPU/NPU/HCCL data
is retained for separate analysis; HTTP and HIXL are outside this scope.
"""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import threading
import traceback

from precision_gate import digest, validate_precision
from prefix_io import mtp_prefix_layer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.profiling import (  # noqa: E402
    PROFILER_CONTRACT, PROFILER_SOURCES, StageMarkers, TORCH_NPU_COMMIT,
    capture_phase, instrument_engine, reject_msprof_environment, run_phase,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--precision-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--mtp-prefix", type=Path, required=True)
    parser.add_argument("--last-prompt-token", type=int, required=True)
    parser.add_argument("--mtp-width", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--max-tokens", type=int, choices=range(2, 7),
                        default=6)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--expert-bank", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.last_prompt_token < 154880:
        parser.error("--last-prompt-token is outside the vocabulary")
    if args.capacity < 8192 + args.max_tokens - 1:
        parser.error("Cache capacity cannot hold the bounded 8K request")
    return args


def bind_prefixes(args, gate, rank):
    paths = {"main": args.prefix / f"rank-{rank}.pt",
             "mtp": args.mtp_prefix / f"rank-{rank}.pt"}
    hashes = {name: digest(path) for name, path in paths.items()}
    expected = gate["prefix_sha256"][str(rank)]
    for name in ("main", "mtp"):
        if expected[name] is not None and hashes[name] != expected[name]:
            raise ValueError(f"{name} prefix differs from device precision")
    if (gate["last_prompt_token"] is not None
            and args.last_prompt_token != gate["last_prompt_token"]):
        raise ValueError("Prompt replay token differs from device precision")
    return paths, hashes


def load_prefix(paths, gate, token, device):
    import torch

    from tilert.pd_vllm.ascend_cache import RankPrefix

    main = torch.load(paths["main"], map_location="cpu", weights_only=True)
    mtp = torch.load(paths["mtp"], map_location="cpu", weights_only=True)
    if main["metadata_sha256"] != gate["checkpoint_sha256"]:
        raise ValueError("Main prefix checkpoint differs from decode")
    layers = main["layers"]
    if len(layers) != 78:
        raise ValueError("Main prefix requires all 78 layers")
    length = gate["initial_position"]
    draft = mtp_prefix_layer(mtp, gate["checkpoint_sha256"])
    if any(value.shape[0] != length for layer in layers
           for value in layer.values()):
        raise ValueError("Main prefix tensors must contain exactly 8K rows")
    if any(value.shape[0] != length - 1 for value in draft.values()):
        raise ValueError("MTP prefix requires P-1 parent rows")
    return RankPrefix(
        [{name: value[:length - 1].to(device)
          for name, value in layer.items()} for layer in layers],
        {name: value.to(device) for name, value in draft.items()}, token,
        length,
    )


def profiler_identity(torch_npu):
    from torch_npu.version import git_version

    package = Path(torch_npu.__file__).resolve().parent
    actual = {name: digest(package / name) for name in PROFILER_SOURCES}
    if (str(torch_npu.__version__) != "2.12.0"
            or git_version != TORCH_NPU_COMMIT
            or actual != PROFILER_SOURCES):
        raise ValueError("Installed profiler differs from the audited source")
    home = Path(os.environ["ASCEND_HOME_PATH"])
    msprof_hash = digest(home / "tools/profiler/bin/msprof")
    versions = {}
    for name in ("version.cfg", "compiler/version.info",
                 "runtime/version.info"):
        path = home / name
        if path.is_file():
            versions[name] = {"sha256": digest(path),
                              "text": path.read_text()}
    return {"source_sha256": actual, "contract": PROFILER_CONTRACT,
            "cann_home": str(home), "cann_version_files": versions,
            "msprof_sha256": msprof_hash,
            "msprof_matches_cached_package":
                msprof_hash == PROFILER_CONTRACT["msprof_sha256"]}


def make_profiler(directory, rank, *, with_stack=False):
    from torch_npu import profiler

    return profiler.profile(
        activities=[profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=with_stack,
        on_trace_ready=profiler.tensorboard_trace_handler(
            str(directory), worker_name=f"rank-{rank}", analyse_flag=False,
        ),
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=profiler.AiCMetrics.AiCoreNone, mstx=True,
        ),
    )


@contextmanager
def npu_range(name):
    import torch
    from torch_npu.npu import mstx

    with torch.autograd.profiler.record_function(name):
        identifier = mstx.range_start(name)
        try:
            yield
        finally:
            mstx.range_end(identifier)


def trace_inventory(directory):
    files = [path for path in directory.rglob("*") if path.is_file()]
    entries = [{"path": str(path.relative_to(directory)),
                "bytes": path.stat().st_size, "sha256": digest(path)}
               for path in sorted(files)]
    nonempty = [Path(item["path"]).parts for item in entries if item["bytes"]]
    framework = any("FRAMEWORK" in parts for parts in nonempty)
    device = any("data" in parts
                 and any(part.startswith("device_") for part in parts)
                 and any(part.startswith("PROF_") for part in parts)
                 for parts in nonempty)
    if not framework or not device:
        raise RuntimeError("Capture lacks nonempty framework or device data")
    return entries


def run(args, report):
    reject_msprof_environment()
    gate = validate_precision(
        args.precision_dir, args.checkpoint, mtp_width=args.mtp_width,
        expert_backend="bank" if args.expert_bank else "per_expert",
    )
    rank = int(os.environ["RANK"])
    paths, hashes = bind_prefixes(args, gate, rank)
    report.update(precision=gate, prefix_sha256=hashes)

    import torch
    import torch.distributed as distributed
    import torch_npu

    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine
    from tilert.models.glm_5_2_ascend.native_mtp import NativeMtp
    from tilert.models.glm_5_2_ascend.session import PrefixRequest
    from tilert.models.glm_5_2_ascend.tp_runtime import TorchControlChannel

    report["profiler_api"] = profiler_identity(torch_npu)
    packages = {"torch": str(torch.__version__),
                "torch_npu": str(torch_npu.__version__)}
    if packages != gate["packages"]:
        raise ValueError("Runtime packages differ from device precision")
    report["packages"] = packages
    torch.set_num_threads(1)
    device = torch.device(f"npu:{int(os.environ['LOCAL_RANK'])}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    if (distributed.get_world_size() != 16
            or distributed.get_rank() != rank or device.index != rank):
        raise ValueError("Capture requires one local TP16 group")
    report.update(world_size=16, device=str(device),
                  device_name=torch.npu.get_device_name(device))
    target = NativeDecodeEngine(
        args.checkpoint, device, args.capacity,
        use_expert_bank=args.expert_bank,
    )
    draft = NativeMtp(target)
    target.prepare_experts()
    draft.prepare_experts()
    prefix = load_prefix(paths, gate, args.last_prompt_token, device)
    request = PrefixRequest("capture", prefix.seq_len)
    markers = StageMarkers(rank)
    report["markers"] = markers.events
    engine = instrument_engine(target, draft, markers)
    channel = TorchControlChannel()
    phase_args = (engine, rank, channel, lambda request, rank: prefix,
                  request, args.mtp_width, args.max_tokens, markers)
    with torch.inference_mode():
        report["warmup"] = run_phase(*phase_args, requests=3, ignore_eos=True)
        torch.npu.synchronize(device)
        distributed.barrier()
        raw = args.output / f"raw-rank-{rank}"
        raw.mkdir(parents=True, exist_ok=False)
        report["trace_directory"] = str(raw)
        markers.phase = "capture"
        markers.range_factory = npu_range
        report["profiler_thread_id"] = threading.get_ident()
        report["profiler_native_thread_id"] = threading.get_native_id()
        report["capture"] = capture_phase(
            make_profiler(raw, rank), markers, distributed.barrier,
            lambda: run_phase(*phase_args, ignore_eos=True),
        )
        markers.range_factory = nullcontext
    report["trace_files"] = trace_inventory(raw)
    report["profiled"] = True
    report["event_coverage"] = "raw captured; timeline analysis pending"
    if rank == 0:
        reference = report["warmup"]["requests"][0]["tokens"]
        records = report["warmup"]["requests"] + report["capture"]["requests"]
        if any(item["tokens"] != reference for item in records):
            raise RuntimeError("Capture and warmup token sequences differ")
        if len(reference) != args.max_tokens:
            raise RuntimeError("The bounded request did not reach its budget")
    distributed.barrier()
    distributed.destroy_process_group()
    report["status"] = "raw_capture"


def main(argv=None):
    args = parse_arguments(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    sources = [Path(__file__), Path(__file__).with_name("precision_gate.py"),
               Path(__file__).with_name("prefix_io.py"),
               *(ROOT / "tilert/models/glm_5_2_ascend").rglob("*.py"),
               *(ROOT / "tilert/pd_vllm").rglob("*.py")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "rank": rank,
        "hostname": platform.node(), "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "source_commit": args.source_commit,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path)
                          for path in sources},
        "mtp_width": args.mtp_width, "max_tokens": args.max_tokens,
        "last_prompt_token": args.last_prompt_token, "ignore_eos": True,
        "expert_backend": "bank" if args.expert_bank else "per_expert",
        "scope": "TP16 production session/command loop; no HTTP or HIXL",
        "tpot_measured": False, "profiled": False, "status": "error",
    }
    try:
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    path = args.output / f"rank-{rank}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rank": rank, "status": report["status"],
                      "report": str(path)}))
    return int(report["status"] != "raw_capture")


if __name__ == "__main__":
    raise SystemExit(main())
