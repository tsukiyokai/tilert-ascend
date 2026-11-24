"""Compare CANN MTP rounds with serial continuation of a v3 capture.

Launch with torchrun --nproc-per-node=16. Main KV contains P rows, MTP KV
contains P-1 rows, and the captured normalized, unrotated parent is at P-1.
The prefill first token is pending: no prompt row is replayed. Logical
outputs include EOS; the final output has not been consumed by the target.
Serial and first speculative runs clear shared KV storage before injection;
the reset run retains speculative tail bytes for the exact repeat check.
This checks serial/speculative consistency and exact reset, not HF parity
or performance. Only one target and one official CANN MTP model are loaded.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import traceback

from prefix_io import mtp_prefix_layer
from run_cann_model import check_prefix, load_capture
from run_native_decode import compare_tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.mtp_round import (  # noqa: E402
    MtpRoundExecutor,
)
from tilert.pd_vllm.ascend_layout import (  # noqa: E402
    PARENT_HIDDEN_REGION, validate_layout, validate_parent_hidden_metadata,
)
from tilert.pd_vllm.decode_timing import atomic_json  # noqa: E402
from tilert.pd_vllm.prefix_capture import (  # noqa: E402
    copy_to_cpu, digest, save_bundle,
)


def token_ids(result):
    """Read a step's ids to the host; serial teacher forcing needs each one."""
    values = result.token_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(int(value) for value in values)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--width", type=int,
                        choices=(2, 4, 6, 8, 12, 16, 24, 32), default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--target-graph", action="store_true",
                        help="Compare graph after uncaptured execution")
    parser.add_argument("--draft-graph", action="store_true",
                        help="Graph shared-index M1 draft chains")
    parser.add_argument("--conditioned-graph", action="store_true",
                        help="Graph the M4 target-conditioned MTP pass")
    parser.add_argument("--draft-projection-tp", action="store_true",
                        help="Shard MTP rot and eh_proj outputs over TP16")
    parser.add_argument("--hoist-metadata", action="store_true",
                        help="Build SFA/indexer metadata once per captured"
                             " width and refresh only its rope slices")
    parser.add_argument("--hoist-forward-context", action="store_true",
                        help="Serve a prebuilt MC2 mask per token width")
    parser.add_argument("--compile-target", action="store_true",
                        help="Compile the target backbone")
    parser.add_argument("--super-kernel", action="store_true",
                        help="Use static kernels and SuperKernel for target")
    parser.add_argument("--attention-kernel",
                        choices=("sfa", "gather_fia", "shared_fia"),
                        default="sfa")
    parser.add_argument("--fp32-head", action="store_true")
    parser.add_argument("--profile-stacks", action="store_true",
                        help="record Python stacks so eager kernels name their caller")
    parser.add_argument("--profile-output", type=Path,
                        help="Capture M4 stages after device checks pass")
    parser.add_argument("--device-offset", type=int, choices=range(16),
                        default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--init-method", default="env://")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.capacity < 128:
        parser.error("Positive output budget and capacity >= 128 required")
    if (args.target_graph or args.draft_graph) and args.max_tokens < 2:
        parser.error("Graph comparison requires a decode round")
    if args.draft_graph and args.width == 2:
        parser.error("--draft-graph requires a width the graph can capture")
    if args.conditioned_graph and not args.draft_graph:
        parser.error("--conditioned-graph requires --draft-graph")
    if args.compile_target and not args.target_graph:
        parser.error("--compile-target requires --target-graph")
    if args.super_kernel and not args.compile_target:
        parser.error("--super-kernel requires --compile-target")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("Memory utilization must be in (0, 1]")
    return args


def validate_v3(manifest, checkpoint):
    if (type(manifest.get("seq_len")) is not int
            or manifest["seq_len"] < 1
            or type(manifest.get("generation")) is not int
            or manifest["generation"] < 1
            or not isinstance(manifest.get("request_id"), str)
            or not manifest["request_id"]):
        raise ValueError("Capture needs a valid request, generation and P")
    runtime = manifest["runtime"]
    layout = runtime["layout"]
    validate_layout(layout)
    if (layout["version"] != 3 or runtime["world_size"] != 16
            or runtime["checkpoint_sha256"] != checkpoint):
        raise ValueError("MTP requires a checkpoint-bound TP16 v3 capture")
    contract = manifest.get(PARENT_HIDDEN_REGION)
    validate_parent_hidden_metadata(contract, manifest["seq_len"])
    if any(type(receipt.get("rank")) is not int
           or receipt.get(PARENT_HIDDEN_REGION) != contract
           for receipt in manifest["ranks"]):
        raise ValueError("Rank receipts disagree on the parent contract")
    sources = runtime.get("source_sha256")
    if not isinstance(sources, dict) or not sources or any(
        not isinstance(name, str) or not name
        or not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for name, value in sources.items()
    ):
        raise ValueError("Capture source SHA256 records are required")
    return contract


def load_inputs(args, rank, checkpoint):
    import torch

    case = args.capture_case
    manifest = json.loads((case / "manifest.json").read_text())
    contract = validate_v3(manifest, checkpoint)
    main, teacher, inputs = load_capture(case, rank, checkpoint)
    receipt = next(item for item in manifest["ranks"]
                   if item["rank"] == rank)
    path = case / "mtp-prefix" / f"rank-{rank}.pt"
    file_hash = digest(path)
    if file_hash != receipt["files"]["mtp"]["sha256"]:
        raise ValueError("MTP prefix differs from its rank receipt")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    draft = mtp_prefix_layer(saved, checkpoint)
    length = inputs["initial_position"]
    if (length + args.max_tokens - 1 > args.capacity
            or length >= args.capacity):
        raise ValueError("Prefix and output budget exceed cache capacity")
    for name, width in (("latent", 512), ("rope", 64), ("index_keys", 128)):
        value = draft[name]
        if (value.shape != (length - 1, width)
                or value.dtype != torch.bfloat16):
            raise ValueError(f"MTP {name} must contain BF16 P-1 rows")
    parent = saved.get("parent_hidden", {})
    identity = {**contract, "request_id": inputs["request_id"],
                "generation": inputs["generation"]}
    if ({key: value for key, value in parent.items() if key != "tensor"}
            != identity):
        raise ValueError("MTP parent identity differs from the capture")
    tensor = parent.get("tensor")
    if (not isinstance(tensor, torch.Tensor)
            or tensor.shape != (1, 6144) or tensor.dtype != torch.bfloat16
            or not torch.isfinite(tensor).all()):
        raise ValueError("MTP parent must be finite BF16 [1,6144]")
    parent_hash = hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes(),
    ).hexdigest()
    inputs.update(
        rank=rank, mtp_prefix_file=str(path), mtp_prefix_sha256=file_hash,
        parent_hidden=identity, parent_hidden_sha256=parent_hash,
        capture_source_sha256=manifest["runtime"]["source_sha256"],
    )
    return main, draft, tensor, teacher[0], inputs


def final_positions(initial, output_count):
    """Counts include EOS; the last output remains pending in the target."""
    if output_count < 1:
        raise ValueError("The prefill token is the first logical output")
    main = initial + output_count - 1
    return (main, initial - 1 if output_count == 1 else main)


def finish_reason(tokens, position, capacity, budget, stops):
    if tokens[-1] in stops:
        return "eos"
    if len(tokens) >= budget:
        return "length"
    if position == capacity:
        return "capacity"
    return None


def serial_run(target, draft, first, parent, budget, stops, observe=None):
    """Teacher-force every MTP row with a serial target parent/prediction."""
    output = [first]
    reason = finish_reason(output, target.position, target.capacity,
                           budget, stops)
    if reason is None:
        draft.forward((first,), parent, full_logits=False)
    while reason is None:
        result = target.verify((output[-1],), full_logits=False)
        prediction = token_ids(result)[0]
        draft.forward((prediction,), result.last_hidden, full_logits=False)
        output.append(prediction)
        reason = finish_reason(output, target.position, target.capacity,
                               budget, stops)
        if observe is not None:
            observe((output, target.position, draft.position, reason))
    return {"tokens": output, "positions": [target.position, draft.position],
            "finish_reason": reason}


def speculative_run(target, draft, first, parent, width, budget, stops,
                    observe=None):
    executor = MtpRoundExecutor(target, draft, width, budget, stops)
    executor.start(first, parent)
    output, rounds = [first], []
    while executor.state.finish_reason is None:
        executed = executor.advance()
        output.extend(executed.commit.accepted_tokens)
        rounds.append(asdict(executed))
        if observe is not None:
            observe((output, target.position, draft.position,
                     executor.state.finish_reason))
    return {
        "tokens": output, "positions": [target.position, draft.position],
        "finish_reason": executor.state.finish_reason, "rounds": rounds,
        "statistics": executor.state.statistics(),
    }


def new_cache(target, draft, initial):
    def rows(layers, begin, end):
        return [{name: copy_to_cpu(value[begin:end])
                 for name, value in layer.items()} for layer in layers]

    return {"main": rows(target.cache, initial, target.position),
            "mtp": rows([draft.cache], initial - 1, draft.position)}


def prepare_phase(target, draft, main_prefix, mtp_prefix, label):
    target.reset()
    draft.reset()
    if target.position != 0 or draft.position != 0:
        raise ValueError("Reset must clear both logical cache lengths")
    cleared = label != "reset"
    if cleared:
        # Layer 78 shares this allocation with the main-model caches.
        target.storage.zero_()
    target.inject(main_prefix)
    draft.inject(mtp_prefix)
    return cleared


def compare_cache(label, actual, expected, checks, *, exact=False):
    import torch

    for model, count in (("main", 78), ("mtp", 1)):
        left, right = actual[model], expected[model]
        checks.append({"name": f"{label}.{model}.layers",
                       "passed": len(left) == len(right) == count})
        for layer, (values, references) in enumerate(zip(left, right)):
            key = f"{label}.{model}.{layer}"
            checks.append({"name": key + ".fields",
                           "passed": values.keys() == references.keys()
                           and {"latent", "rope"} <= values.keys()})
            for name in values.keys() & references.keys():
                value, reference = values[name], references[name]
                same_layout = (value.shape == reference.shape
                               and value.dtype == reference.dtype)
                checks.append({"name": key + "." + name + ".layout",
                               "passed": same_layout})
                if not same_layout:
                    continue
                for row in range(value.shape[0]):
                    compare_tensor(
                        f"{key}.{name}.row.{row}", value[row], reference[row],
                        checks,
                        exact=exact or not value.is_floating_point()
                        or name == "index_scales",
                    )
                    if not exact and value.is_floating_point() and (
                        name != "index_scales"
                    ):
                        item = checks[-1]
                        item.update(
                            diagnostic_only=True,
                            bit_exact=torch.equal(value[row], reference[row]),
                            passed=bool(torch.isfinite(value[row]).all()
                                        and torch.isfinite(
                                            reference[row],
                                        ).all()),
                        )


def source_hashes(target, draft):
    from vllm.model_executor.model_loader.default_loader import (
        DefaultModelLoader,
    )
    from vllm.model_executor.model_loader.utils import (
        process_weights_after_loading,
    )

    paths = {Path(__file__).resolve(),
             *[Path(__file__).with_name(name) for name in
               ("run_cann_model.py", "run_native_decode.py", "prefix_io.py")]}
    for value in (type(draft), type(draft.model), type(draft.model.model),
                  type(draft.model.model.layers["78"]), DefaultModelLoader,
                  process_weights_after_loading, MtpRoundExecutor):
        paths.add(Path(inspect.getfile(inspect.unwrap(value))).resolve())
    paths.add(ROOT / "tilert/models/glm_5_2_ascend/mtp_state.py")
    paths.add(ROOT / "tilert/models/glm_5_2_ascend/cann_graph.py")
    paths.add(ROOT / "tilert/models/glm_5_2_ascend/cann_mtp_graph.py")
    paths.add(ROOT / "tilert/models/glm_5_2_ascend/cann_mtp_projection.py")
    paths.update(ROOT / f"tilert/pd_vllm/{name}.py" for name in
                 ("ascend_layout", "ascend_cache", "prefix_capture"))
    return {**target.source_metadata(),
            **{str(path): digest(path) for path in sorted(paths)}}


def execute(args):
    import torch
    import torch.distributed as distributed
    from vllm.distributed import get_world_group
    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.models.glm_5_2_ascend.cann_mtp import CannMtp
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise ValueError("Launch exactly 16 ranks")
    physical = (rank + args.device_offset) % 16
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    main_prefix, mtp_prefix, saved_parent, first, inputs = load_inputs(
        args, rank, checkpoint,
    )
    config = json.loads((args.checkpoint / "config.json").read_text())
    stops = config["eos_token_id"]
    stops = (stops,) if type(stops) is int else tuple(stops)
    stops = () if args.ignore_eos else stops
    torch.set_num_threads(1)
    target = CannModel(
        args.checkpoint, f"npu:{physical}", rank, args.init_method,
        args.capacity, args.gpu_memory_utilization,
        compile_target=args.compile_target,
        super_kernel=args.super_kernel,
        attention_kernel=args.attention_kernel,
        graph_width=args.width,
        fp32_head=args.fp32_head,
        hoist_metadata=args.hoist_metadata,
        hoist_forward_context=args.hoist_forward_context,
    )
    draft = CannMtp(target, shard_projections=args.draft_projection_tp)
    if not draft.model.do_not_compile:
        raise ValueError("MTP backbone compilation must be disabled")
    source = source_hashes(target, draft)
    group = get_world_group().cpu_group

    def gather(value):
        values = [None] * 16
        distributed.all_gather_object(values, value, group=group)
        return values

    def agree(value):
        if any(other != value for other in gather(value)):
            raise ValueError("Sixteen ranks disagree on inputs or decode")

    agree({"capture": inputs["capture_manifest_sha256"],
           "prompt": inputs["prompt_sha256"], "first": first,
           "parent": inputs["parent_hidden_sha256"], "source": source,
           "checkpoint": checkpoint, "width": args.width,
           "budget": args.max_tokens, "stops": stops})
    initial = inputs["initial_position"]
    checks, runs, paths = [], {}, {}
    directory = args.output / f"rank-{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    expected_cache = None
    # The compiled target is traced at the width of its first forward, so a
    # serial M1 pass would fix it at one row and the M4 phase would then feed
    # four rows into a one-row buffer.
    phases = ["speculative"] if args.compile_target else ["serial",
                                                          "speculative"]
    if args.target_graph or args.draft_graph:
        phases.append("graph")
    phases.append("reset")
    with torch.inference_mode():
        for label in phases:
            cleared = prepare_phase(
                target, draft, main_prefix, mtp_prefix, label,
            )
            if label == "graph":
                for layers, begin, dense_mla in (
                    (target.cache, initial,
                     target.attention_kernel == "shared_fia"),
                    ([draft.cache], initial - 1, False),
                ):
                    for fields in layers:
                        for name, value in fields.items():
                            tail = value[begin:initial + args.max_tokens + 4]
                            # Dense PV requires finite masked rows.
                            poison = (7.0 if dense_mla and name in
                                      ("latent", "rope") else float("nan"))
                            tail.fill_(poison
                                       if value.is_floating_point() else -128)
                if args.target_graph:
                    target.enable_graph(width=args.width)
                if args.draft_graph:
                    draft.enable_graph(
                        conditioned_width=(args.width
                                           if args.conditioned_graph else 0))
            parent = saved_parent.to(target.device)
            predicted = token_ids(target.predict(parent))[0]
            agree((predicted, target.position, draft.position))
            if predicted != first:
                raise ValueError("Captured parent head differs from g0")
            if (target.position, draft.position) != (initial, initial - 1):
                raise ValueError("Injection must retain main P / MTP P-1")
            if label == "serial":
                result = serial_run(
                    target, draft, first, parent, args.max_tokens, stops,
                    observe=agree,
                )
            else:
                result = speculative_run(
                    target, draft, first, parent, args.width,
                    args.max_tokens, stops, observe=agree,
                )
            result["cache_zeroed_before_inject"] = cleared
            runs[label] = result
            checks.append({
                "name": label + ".pending_lengths",
                "passed": tuple(result["positions"]) == final_positions(
                    initial, len(result["tokens"]),
                ),
            })
            check_prefix(target, main_prefix, checks, label + ".main")
            check_prefix(SimpleNamespace(cache=[draft.cache]), [mtp_prefix],
                         checks, label + ".mtp")
            compare_tensor(label + ".parent_unchanged", copy_to_cpu(parent),
                           saved_parent, checks, exact=True)
            cache = new_cache(target, draft, initial)
            path = directory / f"{label}.pt"
            save_bundle({**result, "new_cache": cache}, path)
            paths[label] = {"path": str(path), "sha256": digest(path)}
            if label == phases[0]:
                expected_cache = cache
            else:
                reference = runs[phases[phases.index(label) - 1]]
                checks.append({
                    "name": label + ".tokens_and_state",
                    "passed": all(result[key] == reference[key] for key in
                                  ("tokens", "positions", "finish_reason")),
                })
                compare_cache(label, cache, expected_cache, checks,
                              exact=label == "reset")
                if label != "reset":
                    expected_cache = cache
                else:
                    checks.append({
                        "name": "reset.rounds_exact",
                        "passed": result["rounds"] == reference["rounds"],
                    })
    if target.graph is not None:
        graph = target.graph
        graph.validate_metadata(target._metadata(args.width))
        entries = graph.wrapper.concrete_aclgraph_entries
        checks.extend([
            {"name": "graph.actual_capture",
             "passed": graph.capture_count == 1 and len(entries) == 1
             and all(entry.aclgraph is not None
                     for entry in entries.values())},
            {"name": "graph.all_rounds_replayed",
             "passed": graph.replay_count == sum(
                 len(runs[label]["rounds"]) for label in ("graph", "reset")
             )},
            {"name": "graph.input_addresses",
             "passed": entries[graph.descriptor].input_addresses == [
                 target.input_ids.data_ptr(), target.positions.data_ptr(),
             ]},
        ])
    if args.draft_graph:
        # A round proposes `width` candidates: it starts with the pending
        # token, appends width-1 more, and chains between appends except after
        # the last, so the chain replays width-2 times. The literal 2 that
        # used to stand here was that count at width 4.
        chains_per_round = args.width - 2
        checks.extend([
            {"name": "draft_graph.single_capture",
             "passed": draft.graph.capture_count == 1},
            {"name": "draft_graph.chains_per_round",
             "passed": draft.graph.replay_count == chains_per_round * sum(
                 len(runs[label]["rounds"]) for label in ("graph", "reset")
             )},
        ])
    checks.append({"name": "source_stable",
                   "passed": source_hashes(target, draft) == source})
    passed = all(item["passed"] for item in checks)
    report = {
        "status": "pass" if passed else "fail", "rank": rank,
        "world_size": 16, "physical_device": physical,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "command": [sys.executable, *sys.argv],
        "source_commit": args.source_commit, "source_sha256": source,
        "checkpoint_sha256": checkpoint, "inputs": inputs,
        "width": args.width, "max_tokens": args.max_tokens,
        "target_graph": args.target_graph,
        "draft_graph": args.draft_graph,
        "draft_projection_tp": args.draft_projection_tp,
        "hoist_metadata": args.hoist_metadata,
        "hoist_forward_context": args.hoist_forward_context,
        "compile_target": args.compile_target,
        "super_kernel": args.super_kernel,
        "attention_kernel": args.attention_kernel,
        "fp32_head": args.fp32_head,
        "graph": None if target.graph is None else {
            "width": target.graph.width,
            "captures": target.graph.capture_count,
            "replays": target.graph.replay_count,
            "entries": [{"descriptor": str(key),
                         "has_aclgraph": entry.aclgraph is not None,
                         "input_addresses": entry.input_addresses}
                        for key, entry in target.graph.wrapper
                        .concrete_aclgraph_entries.items()],
        },
        "draft_graph_state": None if draft.graph is None else {
            "captures": draft.graph.capture_count,
            "replays": draft.graph.replay_count,
            "scope": "M1 shared-index chain, including own head",
        },
        "conditioned_graph_state": (
            None if draft.conditioned_graph is None else {
                "captures": draft.conditioned_graph.capture_count,
                "replays": draft.conditioned_graph.replay_count,
                "scope": "M4 target-conditioned pass, including own head",
            }),
        "capacity": args.capacity, "eos_tokens": stops,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "packages": {name: version(name) for name in
                     ("torch", "torch-npu", "vllm", "vllm-ascend")},
        "scope": "CANN continuation consistency across recorded phases",
        "compared_phases": phases,
        "logical_outputs_include_eos": True,
        "final_output_is_pending": True, "prompt_replayed": False,
        "serial_speculative_consistency": passed if "serial" in runs else None,
        "reset_exact": all(item["passed"] for item in checks
                           if item["name"].startswith("reset.")),
        "precision_verified": False, "hf_parity": False,
        "performance_measured": False, "numerical_execution": True,
        "coverage": {
            "greedy_tokens": True, "valid_new_main_mtp_kv": True,
            "frozen_prefix_exact": True, "reset_exact": True,
            "intermediate_hidden_logits": False, "routes_indices": False,
        },
        "runs": runs, "trace_files": paths, "checks": checks,
    }
    path = args.output / f"rank-{rank}.json"
    atomic_json(path, report)
    receipts = gather({"rank": rank, "passed": passed,
                       "report_sha256": digest(path),
                       "tokens": runs["speculative"]["tokens"],
                       "positions": runs["speculative"]["positions"]})
    same = all(item["tokens"] == receipts[0]["tokens"]
               and item["positions"] == receipts[0]["positions"]
               for item in receipts)
    success = same and all(item["passed"] for item in receipts)
    if rank == 0:
        summary = {
            "status": "pass" if success else "fail", "ranks": receipts,
            "all_ranks_agree": same,
            "serial_speculative_consistency": (
                success if "serial" in runs else None
            ),
            "compared_phases": phases,
            "precision_verified": False, "hf_parity": False,
            "performance_measured": False,
        }
        atomic_json(args.output / "summary.json", summary)
        print(json.dumps(summary))
    if success and args.profile_output:
        from profile_cann_decode import capture_validated_mtp
        from tilert.pd_vllm.ascend_cache import FullRankPrefix

        prefix = FullRankPrefix(main_prefix, mtp_prefix,
                                saved_parent.to(target.device), initial)
        capture_validated_mtp(target, draft, prefix, first, rank,
                              args.profile_output, args.width,
                              with_stack=args.profile_stacks)
        atomic_json(args.output / f"profile-binding-rank-{rank}.json", {
            "validation_summary_sha256": digest(args.output / "summary.json"),
            "validation_report_sha256": digest(path),
            "profile_report_sha256": digest(
                args.profile_output / f"rank-{rank}.json",
            ),
            "capture_manifest_sha256": inputs["capture_manifest_sha256"],
        })
    draft.close()
    target.shutdown()
    return 0 if success else 1


def main():
    args = parse_arguments()
    try:
        return execute(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "unknown")
        path = args.output / f"rank-{rank}-error.txt"
        path.write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
