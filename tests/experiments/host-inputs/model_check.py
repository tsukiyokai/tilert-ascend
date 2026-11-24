"""Check host packets against frozen actual TP16 model outputs; no timing."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "checkpoint",
        "capture-case",
        "teacher",
        "reference",
        "helpers",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [
        str(args.source),
        str(args.source / "tests/glm52"),
        str(args.helpers),
    ]
    if digest(args.helpers / "run.py") != (
        "7e5028edb20bd552b8a0861eb8ff7cca049ac0793126344afd262de8a21ce216"
    ):
        raise ValueError("Frozen capture helper changed")
    spec = importlib.util.spec_from_file_location(
        "width_helpers",
        args.helpers / "run.py",
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import torch
    import torch_npu
    from run_cann_model import check_prefix, load_capture
    from vllm.distributed import get_world_group

    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.models.glm_5_2_ascend.host_inputs import HostInputs
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    rank = int(os.environ["RANK"])
    if (
        int(os.environ["WORLD_SIZE"]) != 16
        or int(os.environ["LOCAL_RANK"]) != rank
    ):
        raise ValueError("Use one actual local TP16 world")
    torch.set_num_threads(1)
    torch_npu.npu.set_device(rank)
    torch.npu.matmul.allow_hf32 = False
    name = torch_npu.npu.get_device_name(rank)
    if not any(sku in name for sku in ("9382", "9392")):
        raise ValueError("Actual A3 required")
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    prefixes, first, inputs = load_capture(args.capture_case, rank, checkpoint)
    if digest(args.reference) != (
        "1f3b3d62119d4e887dff91fbabefd397391d787a04414cbe7601e2a290af76fe"
    ) or digest(args.teacher) != (
        "82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655"
    ):
        raise ValueError("Frozen inputs differ")
    reference = torch.load(
        args.reference, map_location="cpu", weights_only=True
    )
    tokens = json.loads(args.teacher.read_text())["teacher_tokens"]
    if (
        tokens != reference["teacher_tokens"]
        or first != tokens[:1]
        or reference["checkpoint_sha256"] != checkpoint
        or inputs["initial_position"] != 8192
    ):
        raise ValueError("Capture and teacher identities differ")
    engine = CannModel(
        args.checkpoint,
        f"npu:{rank}",
        rank,
        capacity=9344,
        attention_kernel="sfa",
        fp32_head=True,
        graph_width=32,
        compile_target=False,
        super_kernel=False,
    )
    try:
        output = args.output / f"rank-{rank}"
        output.mkdir(parents=True, exist_ok=False)
        checks, files, states = [], {}, {}
        old_buffers, old_length = engine._input_buffers, engine.seq_lens
        packet = HostInputs(engine.device, tuple(old_buffers))
        sources = engine.source_metadata()
        for item in (HostInputs, main, check_prefix, load_capture):
            path = Path(inspect.getfile(item)).resolve()
            sources[str(path)] = digest(path)
        for mode in ("separate", "packet"):
            engine._host_inputs = packet if mode == "packet" else None
            engine._input_buffers = (
                {
                    width: (*packet.views(width)[:3], *values[3:])
                    for width, values in old_buffers.items()
                }
                if mode == "packet"
                else old_buffers
            )
            engine.seq_lens = old_length
            previous_wide = None
            for phase in ("first", "wide", "reset"):
                engine.reset()
                if phase != "reset":
                    engine.storage.zero_()
                engine.inject(prefixes)
                torch.npu.synchronize()
                current = tokens[:1] if phase == "first" else tokens
                result = engine.verify(current, full_logits=True)
                torch.npu.synchronize()
                frame = {
                    "mode": mode,
                    "phase": phase,
                    "position": 8192,
                    "committed_position": engine.position,
                    "outputs": helper.output_view(result),
                    "inputs": helper.cpu(
                        {
                            "ids": engine.input_ids,
                            "positions": engine.positions,
                            "slots": engine.slots,
                            "length": engine.seq_lens,
                            "query": engine.query,
                        }
                    ),
                    "new_cache": helper.cpu(
                        [
                            {
                                key: value[8192 : 8192 + len(current)]
                                for key, value in layer.items()
                            }
                            for layer in engine.cache
                        ]
                    ),
                }
                wanted = {
                    "ids": current,
                    "positions": list(range(8192, 8192 + len(current))),
                    "slots": list(range(8192, 8192 + len(current))),
                    "length": [8192 + len(current)],
                    "query": [0, len(current)],
                }
                checks.append(
                    {
                        "name": mode + "." + phase + ".inputs",
                        "passed": engine.position == 8192 + len(current)
                        and all(
                            value.tolist() == wanted[key]
                            for key, value in frame["inputs"].items()
                        ),
                    }
                )
                control = reference[
                    "serial_first" if phase == "first" else "wide"
                ]
                checks.append(
                    {
                        "name": mode + "." + phase + ".old_outputs",
                        "passed": helper.output_matches(
                            frame["outputs"], control
                        ),
                    }
                )
                label = mode + "-" + phase
                check_prefix(engine, prefixes, checks, label)
                states[label] = helper.state_hash(frame["outputs"])
                if phase == "reset":
                    checks.append(
                        {
                            "name": mode + ".reset_exact",
                            "passed": helper.exact(
                                {
                                    key: frame[key]
                                    for key in (
                                        "outputs",
                                        "inputs",
                                        "new_cache",
                                    )
                                },
                                {
                                    key: previous_wide[key]
                                    for key in (
                                        "outputs",
                                        "inputs",
                                        "new_cache",
                                    )
                                },
                            ),
                        }
                    )
                if phase == "wide":
                    previous_wide = frame
                path = output / (label + ".pt")
                torch.save(frame, path)
                files[path.name] = digest(path)
                if mode == "packet":
                    baseline = torch.load(
                        output / ("separate-" + phase + ".pt"),
                        weights_only=True,
                        map_location="cpu",
                    )
                    checks.append(
                        {
                            "name": mode
                            + "."
                            + phase
                            + ".same_consumer_state",
                            "passed": helper.exact(
                                {
                                    key: frame[key]
                                    for key in (
                                        "outputs",
                                        "inputs",
                                        "new_cache",
                                    )
                                },
                                {
                                    key: baseline[key]
                                    for key in (
                                        "outputs",
                                        "inputs",
                                        "new_cache",
                                    )
                                },
                            ),
                        }
                    )
        report = {
            "rank": rank,
            "status": "pass" if all(x["passed"] for x in checks) else "fail",
            "scope": "One resident TP16 world, input producer substitution only; numerical check",
            "source_sha256": sources,
            "checkpoint_sha256": checkpoint,
            "inputs": inputs,
            "teacher_sha256": digest(args.teacher),
            "reference_sha256": digest(args.reference),
            "files_sha256": files,
            "states": states,
            "checks": checks,
            "formal_precision_gate_passed": False,
            "performance_measured": False,
        }
        path = args.output / f"rank-{rank}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        group = get_world_group()
        gathered = group.cpu_group
        reports = [None] * 16
        torch.distributed.all_gather_object(
            reports,
            {
                "rank": rank,
                "status": report["status"],
                "report_sha256": digest(path),
                "states": states,
            },
            group=gathered,
        )
        agree = all(item["states"] == reports[0]["states"] for item in reports)
        if rank == 0:
            summary = {
                "status": "pass"
                if agree and all(item["status"] == "pass" for item in reports)
                else "fail",
                "ranks": reports,
                "all_rank_global_states_agree": agree,
                "formal_precision_gate_passed": False,
                "performance_measured": False,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "status": summary["status"],
                        "all_rank_global_states_agree": agree,
                    }
                ),
                flush=True,
            )
        group.barrier()
        return (
            0
            if agree and all(item["status"] == "pass" for item in reports)
            else 1
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
