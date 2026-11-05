"""Compare CANN M=1/2/4 verification and rollback with frozen HF traces."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import traceback

from run_native_decode import (
    check_lengths, compare_tensor, compare_trace, digest, layer_logits,
    snapshot, validate_reference_manifest,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--expert-bank", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def row_result(result, row, position, top_k):
    from tilert.models.glm_5_2_ascend.engine import StepResult

    return StepResult(
        result.logits[row:row + 1],
        tuple(value[row:row + 1] for value in result.hidden_states),
        tuple(value[row, :min(position + 1, top_k)]
              for value in result.indices),
        tuple(value[row] if value is not None else None
              for value in result.routes),
        result.last_hidden[row:row + 1],
    )


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu

    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine

    reference = json.loads((args.reference / "manifest.json").read_text())
    validate_reference_manifest(reference, report["metadata_sha256"])
    teacher = reference["teacher_tokens"][:4]
    if len(teacher) != 4:
        raise ValueError("HF reference must contain at least four input rows")
    initial = reference["initial_position"]
    if initial + 4 > args.capacity:
        raise ValueError("Four verification rows exceed cache capacity")
    report.update(reference_manifest=reference, initial_position=initial,
                  teacher_tokens=teacher)
    torch.set_num_threads(1)
    device = f"npu:{int(os.environ['LOCAL_RANK'])}"
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    rank = distributed.get_rank()
    report.update(rank=rank, world_size=distributed.get_world_size(),
                  device=device, torch=torch.__version__,
                  torch_npu=torch_npu.__version__)
    engine = NativeDecodeEngine(
        args.checkpoint, device, args.capacity,
        use_expert_bank=args.expert_bank,
    )
    prefixes = None
    if args.prefix is not None:
        path = args.prefix / f"rank-{rank}.pt"
        if reference["prefix_files_sha256"].get(path.name) != digest(path):
            raise ValueError("HF and native prefix files differ")
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle["metadata_sha256"] != report["metadata_sha256"]:
            raise ValueError("Prefix checkpoint differs from decode")
        prefixes = [{name: value.to(device) for name, value in item.items()}
                    for item in bundle["layers"]]
    elif initial != 0:
        raise ValueError("The nonempty HF prefix needs --prefix")
    traces = {}
    for offset in range(4):
        path = args.reference / f"step-{initial + offset:05d}.pt"
        if reference["trace_sha256"].get(path.name) != digest(path):
            raise ValueError("HF trace hash differs from its manifest")
        traces[offset] = torch.load(
            path, map_location="cpu", weights_only=True
        )
        if traces[offset]["token"] != teacher[offset]:
            raise ValueError("HF trace and manifest input tokens differ")

    def restart(label):
        engine.reset()
        if prefixes is not None:
            engine.inject(prefixes)
        check_lengths(report, label, engine, initial)

    def compare_rows(label, result, offset):
        projected = layer_logits(engine, result)
        greedy = engine.project_tokens(result.last_hidden, normalized=True)
        report["checks"].append({
            "name": label + ".distributed_top1",
            "passed": torch.equal(greedy, result.logits.argmax(-1)),
        })
        for row in range(result.logits.shape[0]):
            step = offset + row
            position = initial + step
            entry = row_result(result, row, position, engine.config.index_topk)
            actual = snapshot(engine, entry, teacher[step], position,
                              tuple(value[row:row + 1] for value in projected))
            compare_trace(f"{label}.row{row}", actual, traces[step],
                          report["checks"],
                          expected_committed_position=engine.position)
            compare_tensor(f"{label}.row{row}.mtp_conditioning",
                           entry.last_hidden.cpu(),
                           traces[step]["hidden_states"][-1],
                           report["checks"])

    with torch.inference_mode():
        for width in (1, 2, 4):
            restart(f"m{width}.initial")
            for offset in range(0, 4, width):
                result = engine.verify(teacher[offset:offset + width],
                                       trace=True)
                check_lengths(report, f"m{width}.{offset}.lengths", engine,
                              initial + offset + width)
                compare_rows(f"m{width}.{offset}", result, offset)

        for accepted in range(5):
            restart(f"rollback{accepted}.initial")
            # Write a rejected branch, invalidate it, then overwrite its rows.
            branch = [token if row < accepted else (token + 1) % 154880
                      for row, token in enumerate(teacher)]
            engine.verify(branch)
            engine.truncate(initial + accepted)
            check_lengths(report, f"rollback{accepted}.commit", engine,
                          initial + accepted)
            for offset in range(accepted, 4):
                result = engine.verify(teacher[offset:offset + 1], trace=True)
                compare_rows(f"rollback{accepted}.{offset}", result, offset)
            check_lengths(report, f"rollback{accepted}.final", engine,
                          initial + 4)

        for layer in engine.layers:
            layer.mla.validate_indices_on_host = False
        restart("device_indices.initial")
        result = engine.verify(teacher, trace=True)
        compare_rows("device_indices", result, 0)
        restart("token_only.initial")
        result = engine.verify(teacher, full_logits=False)
        expected = torch.tensor(
            [int(traces[row]["logits"].argmax(-1).item())
             for row in range(4)],
            dtype=torch.int64, device=device,
        )
        report["checks"].append({
            "name": "token_only.greedy_matches_hf",
            "passed": result.logits is None
            and torch.equal(result.token_ids, expected),
        })
        engine.reset()
        check_lengths(report, "final.reset", engine, 0)
    failed = torch.tensor([int(any(not item["passed"]
                                  for item in report["checks"]))],
                          dtype=torch.int32, device=device)
    distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
    report["any_rank_failed"] = bool(failed.item())
    report["status"] = "fail" if report["any_rank_failed"] else "pass"
    distributed.barrier()
    distributed.destroy_process_group()


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__),
               Path(__file__).with_name("run_native_decode.py")]
    sources.extend((root / "tilert/models/glm_5_2_ascend").rglob("*.py"))
    report = {
        "status": "error",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "git_head": args.source_commit,
        "command": [sys.executable, *sys.argv],
        "implementation": "NativeDecodeEngine.verify",
        "scope": "HF numerical parity and tentative KV rollback; no timing",
        "timed": False, "profiled": False, "checks": [],
        "expert_backend": "bank" if args.expert_bank else "per_expert",
        "checkpoint": str(args.checkpoint),
        "reference": str(args.reference),
        "prefix": str(args.prefix) if args.prefix else None,
        "python": platform.python_version(),
    }
    try:
        report["source_sha256"] = {
            str(path.relative_to(root)): digest(path) for path in sources
        }
        report["metadata_sha256"] = {
            name: digest(args.checkpoint / name) for name in (
                "config.json", "quant_model_description.json",
                "quant_model_weights.safetensors.index.json",
            )
        }
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    rank = int(os.environ.get("RANK", "0"))
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(not item["passed"]
                                  for item in report["checks"])
    output = args.output / f"rank-{rank}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rank": rank, "status": report["status"],
                      "report": str(output)}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
