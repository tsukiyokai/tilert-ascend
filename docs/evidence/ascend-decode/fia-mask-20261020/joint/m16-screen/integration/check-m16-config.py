"""Construct the CANN adapter's real config without initializing an NPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def run(args):
    import torch
    import torch_npu
    from vllm.config import AttentionConfig
    from vllm_ascend.worker import worker as worker_module

    if torch_npu.npu._initialized:
        raise RuntimeError("Run this checker in a fresh CPU process")
    try:
        AttentionConfig(indexer_kv_dtype="int8")
    except ValueError:
        registration_was_needed = True
    else:
        registration_was_needed = False

    specification = importlib.util.spec_from_file_location(
        "cann_init_candidate", args.source,
    )
    candidate = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = candidate
    specification.loader.exec_module(candidate)
    calls = []

    class WorkerDouble:
        def __init__(self, config, **kwargs):
            calls.append(("construct", kwargs))
            self.device = torch.device(f"npu:{kwargs['local_rank']}")
            model = torch.nn.Module()
            self.model_runner = SimpleNamespace(
                get_model=lambda: model, drafter=None,
            )

        def init_device(self):
            calls.append(("init_device", None))

        def load_model(self):
            calls.append(("load_model", None))

    with (
        patch.object(torch.npu, "set_device") as set_device,
        patch.object(worker_module, "NPUWorker", WorkerDouble),
        patch.object(candidate.CannModel, "_bind_cache") as bind,
        patch.object(candidate.CannModel, "_allocate_inputs") as allocate,
        patch.object(candidate.CannModel, "_install_tp_sum") as tp_sum,
    ):
        engine = candidate.CannModel(
            args.checkpoint, "npu:12", rank=7,
            attention_kernel="shared_fia", graph_width=16,
        )
        assert engine.graph_width == 16
        assert engine.attention_kernel == "shared_fia"
        assert engine.config.scheduler_config.max_num_batched_tokens == 16

    checks = [
        {"name": "unregistered_int8_was_rejected",
         "passed": registration_was_needed},
        {"name": "actual_constructor_configured_int8",
         "passed": engine.config.attention_config.indexer_kv_dtype == "int8"},
        {"name": "tp16_without_drafter",
         "passed": engine.config.parallel_config.tensor_parallel_size == 16
         and engine.config.speculative_config is None},
        {"name": "eager_batch_one",
         "passed": engine.config.model_config.enforce_eager
         and engine.config.scheduler_config.max_num_seqs == 1},
        {"name": "logical_rank_independent_of_device",
         "passed": engine.rank == 7 and engine.device.index == 12
         and calls[0][1]["rank"] == 7
         and calls[0][1]["local_rank"] == 12},
        {"name": "worker_lifecycle_and_empty_prefix",
         "passed": [name for name, _ in calls]
         == ["construct", "init_device", "load_model"]
         and engine.position == 0},
        {"name": "device_binding_cache_allocation_boundaries",
         "passed": set_device.call_count == bind.call_count
         == tp_sum.call_count == allocate.call_count == 1},
        {"name": "no_npu_initialization",
         "passed": not torch_npu.npu._initialized},
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Real CannModel constructor and installed EngineArgs config",
        "stubbed": ["NPU set_device", "NPUWorker lifecycle and weights",
                    "cache binding", "input allocation", "TP sum install"],
        "registration_was_needed": registration_was_needed,
        "npu_initialized": bool(torch_npu.npu._initialized),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.source, Path(__file__).resolve())
        },
        "checks": checks, "passed": all(item["passed"] for item in checks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=(
        Path(__file__).resolve().parents[2]
        / "tilert/models/glm_5_2_ascend/cann_model.py"
    ))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"],
                      "registration_was_needed":
                      report["registration_was_needed"],
                      "checks": len(report["checks"]),
                      "npu_initialized": report["npu_initialized"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
