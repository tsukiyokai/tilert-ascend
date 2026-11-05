"""Capture one eager TP16 CANN 8K/32-row case and strictly compare HF."""

import argparse
from datetime import datetime, timezone
from importlib import util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests/glm52")]

from compare_saved_cann_verify import single_row as saved_single_row  # noqa: E402
from frozen_teacher import load_teacher  # noqa: E402
from run_cann_model import (  # noqa: E402
    check_prefix, compare_step, digest, load_capture,
    validate_reference_manifest,
)
from run_cann_verify import capture, check_record  # noqa: E402
from run_hf_reference import validate_trace  # noqa: E402
from run_native_decode import compare_tensor  # noqa: E402
from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint  # noqa: E402
from tilert.pd_vllm.decode_timing import atomic_json  # noqa: E402


WIDTH, INITIAL, LAYERS = 32, 8192, 78


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_identity():
    paths = [Path(__file__), HERE / "README.md", HERE / "check.py",
             ROOT / "experiments/graph-evidence/producer_probe.py"]
    paths.extend(ROOT / "tests/glm52" / name for name in (
        "run_cann_model.py", "run_cann_verify.py", "run_native_decode.py",
        "run_hf_reference.py", "compare_saved_cann_verify.py", "frozen_teacher.py",
        "prefix_io.py"))
    paths.extend(ROOT / "tilert/models/glm_5_2_ascend" / name for name in (
        "cann_model.py", "cann_trace.py", "cann_choices.py"))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def admit(args, rank):
    """Read actual frozen input/reference artifacts before model construction."""
    import torch

    metadata = checkpoint_fingerprint(args.checkpoint)
    prefixes, _, inputs = load_capture(args.capture_case, rank, metadata)
    require(inputs["initial_position"] == INITIAL,
            "This entry requires the captured full 8192-token prefix")
    teacher = load_teacher(args.teacher, args.capture_case, WIDTH, args.capacity)
    receipt = json.loads((args.capture_case / "manifest.json").read_text())
    require(all(type(item.get("rank")) is int for item in receipt["ranks"]),
            "Capture rank IDs must have exact integer types")
    if args.capture_only:
        require(args.reference is None, "Capture-only must not supply a reference")
        return prefixes, teacher, inputs, None, metadata
    require(args.reference is not None, "Strict precision requires an independent HF reference")
    reference = json.loads((args.reference / "manifest.json").read_text())
    validate_reference_manifest(reference, metadata, fp32_head=True)
    require(reference.get("initial_position") == INITIAL
            and reference.get("layer_count") == LAYERS
            and reference.get("mla_preprocess") == "prolog_v3"
            and not reference.get("timed") and not reference.get("profiled"),
            "HF must be a non-profiled 78-layer 8K Prolog-v3 FP32-head reference")
    require(reference.get("teacher_tokens") == teacher["teacher_tokens"],
            "HF and target must consume exactly the same 32 teacher inputs")
    for key in ("prompt_length", "prompt_sha256", "last_prompt_token"):
        require(reference.get(key) == inputs[key], "HF/capture prompt differs: " + key)
    require(reference.get("prefix_files_sha256", {}).get(f"rank-{rank}.pt")
            == inputs["prefix_sha256"], "HF and CANN use different prefix bytes")
    artifacts = {}
    for row, token in enumerate(teacher["teacher_tokens"]):
        name = f"step-{INITIAL + row:05d}.pt"
        path = args.reference / name
        require(reference["trace_sha256"].get(name) == digest(path),
                "HF trace bytes differ: " + name)
        trace = torch.load(path, map_location="cpu", weights_only=True)
        validate_trace(trace, LAYERS, fp32_head=True)
        require(type(trace["token"]) is int and trace["token"] == token
                and type(trace["position"]) is int and trace["position"] == INITIAL + row
                and type(trace["committed_position"]) is int
                and trace["committed_position"] == INITIAL + row + 1,
                "HF row token/position identity differs")
        require(tuple(trace["logits"].shape) == (1, 154880)
                and all(tuple(value.shape) == (1, 154880)
                        for value in trace["layer_logits"]),
                "HF reference must retain the complete vocabulary")
        artifacts[name] = digest(path)
    return prefixes, teacher, inputs, {
        "manifest_sha256": digest(args.reference / "manifest.json"),
        "manifest": reference, "trace_sha256": artifacts,
    }, metadata


def returned_ids(label, actual, checks):
    """Execution check independent of any HF reference."""
    import torch

    ids = actual["token_ids"]
    valid = isinstance(ids, torch.Tensor) and ids.dtype == torch.int64 and ids.shape == (1,)
    checks.append({"name": label + ".returned_ids_layout", "passed": valid})
    if valid:
        compare_tensor(label + ".ids_vs_actual_logits", ids,
                       actual["logits"].argmax(-1), checks, exact=True)
    return valid


def strict_row(label, actual, expected, checks, *, exact=False):
    """Keep the original numerical thresholds and explicitly check return IDs."""
    checks.append({"name": label + ".typed_row_identity", "passed": all(
        type(actual[key]) is int and type(expected[key]) is int
        and actual[key] == expected[key]
        for key in ("token", "position", "committed_position"))})
    compare_step(label, actual, expected, checks, exact=exact)
    if returned_ids(label, actual, checks):
        ids = actual["token_ids"]
        compare_tensor(label + ".ids_vs_reference_logits", ids,
                       expected["logits"].argmax(-1), checks, exact=True)


def verdict(capture_only, execution_passed, hf_passed, reference_present):
    """Capture completion never has permission to certify numerical precision."""
    precision = (not capture_only and reference_present
                 and execution_passed and hf_passed)
    passed = execution_passed if capture_only else precision
    status = (("captured_unverified" if passed else "captured_with_failures")
              if capture_only else ("numerical_pass" if passed else "fail"))
    return {"mode": "capture_only" if capture_only else "strict_hf",
            "status": status, "capture_checks_passed": bool(execution_passed),
            "independent_hf_compared": bool(reference_present and not capture_only),
            "case_precision_passed": bool(precision), "entry_passed": bool(passed),
            "precision_verified": False, "formal_precision_gate_passed": False}


def single_row(wide, row, *, choices=True):
    """Use the existing row adapter; represent one scale as HF's length-1 row.

    CANN's actual flat scale storage and the existing CPU producer's [M,1]
    storage encode the same one scalar per row. Preserve the raw wide tensor;
    only the comparison view is canonicalized, without dtype/value changes.
    """
    result = saved_single_row(wide, row, choices=choices)
    for layer in result["cache_rows"]:
        scale = layer["index_scale"]
        if scale is not None:
            require(scale.numel() == 1, "A cache row must contain exactly one index scale")
            layer["index_scale"] = scale.reshape(1).clone()
    return result


def wide_trace(values, tokens, position, committed_position):
    """Package the existing observer's owned CPU tensors without numerical work."""
    require(committed_position == position + WIDTH, "M32 did not commit exactly 32 rows")
    return {
        "token": list(tokens), "position": position,
        "committed_position": committed_position,
        **{name: values[name] for name in ("logits", "last_hidden", "token_ids")},
        **{field: tuple(values[f"{field}.{layer}"] for layer in range(count))
           for field, count in (("hidden_states", 79), ("layer_logits", 78),
                                ("indices", 78), ("routes", 78))},
        "cache_rows": tuple({name: values[f"cache_rows.{layer}.{name}"]
                             for name in ("latent", "rope", "index", "index_scale")}
                            for layer in range(LAYERS)),
    }


def load_observer():
    path = ROOT / "experiments/graph-evidence/producer_probe.py"
    spec = util.spec_from_file_location("cann_precision_persistent_observer", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PersistentTrace


def mapped_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and ".so" in path and any(
            key in path.lower() for key in (
                "ascend", "hixl", "shmem", "torchair", "vllm", "custom_transformer",
                "libopapi", "libcust", "tilert",
            )
        ) and Path(path).is_file():
            paths.add(path)
    return {path: digest(Path(path)) for path in sorted(paths)}


def run(args, report, rank):
    import torch
    import torch.distributed as dist

    PersistentTrace = load_observer()
    prefixes, teacher, inputs, reference, checkpoint = admit(args, rank)
    report.update(inputs=inputs, teacher=teacher, reference=reference,
                  checkpoint_sha256=checkpoint, stage="inputs_admitted")
    if args.check_inputs_only:
        for other in range(16):
            _, _, identity = load_capture(args.capture_case, other, checkpoint)
            if reference is not None:
                require(reference["manifest"]["prefix_files_sha256"].get(f"rank-{other}.pt")
                        == identity["prefix_sha256"], "HF lacks a matching TP16 prefix")
        report.update(status="inputs_admitted", inputs=inputs, teacher=teacher,
                      reference=reference, checkpoint_sha256=checkpoint,
                      npu_executed=False)
        return True

    from vllm.distributed import get_world_group
    from vllm.config import set_current_vllm_config
    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    require(int(os.environ["WORLD_SIZE"]) == 16
            and int(os.environ["LOCAL_RANK"]) == rank,
            "Use one local sixteen-rank torchrun world")
    torch.set_num_threads(1)
    import torch_npu  # noqa: F401

    properties = torch.npu.get_device_properties(rank)
    require(properties.name.upper() in ("ASCEND910_9392", "ASCEND910_9382"),
            "Numerical device execution is restricted to actual Ascend A3")
    report["stage"] = "model.construct"
    target = CannModel(args.checkpoint, f"npu:{rank}", rank,
                       capacity=args.capacity,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       attention_kernel="sfa", fp32_head=True, graph_width=WIDTH,
                       compile_target=False, super_kernel=False)
    group = get_world_group().cpu_group
    artifacts, checks, states = {}, report["checks"], []
    report.update(trace_sha256=artifacts, states=states)
    directory = args.output / f"rank-{rank}"
    directory.mkdir(exist_ok=False)

    def gather(value):
        values = [None] * 16
        dist.all_gather_object(values, value, group=group)
        return values

    def checked(label, operation):
        report["stage"] = label
        result, error = None, None
        try:
            result = operation()
        except Exception:
            error = traceback.format_exc()
        failures = {index: value for index, value in enumerate(gather(error)) if value}
        if failures:
            raise RuntimeError(f"{label}: {failures}")
        return result

    def save(name, value):
        path = directory / name
        torch.save(value, path)
        artifacts[name] = digest(path)

    def expected(row):
        require(reference is not None, "Capture-only has no HF oracle")
        name = f"step-{INITIAL + row:05d}.pt"
        path = args.reference / name
        require(digest(path) == reference["trace_sha256"][name], "HF artifact changed during run")
        return torch.load(path, map_location="cpu", weights_only=True)

    def prepare(clear):
        target.reset()
        require(target.position == 0, "Reset did not clear target length")
        if clear:
            target.storage.zero_()
        target.inject(prefixes)
        torch.npu.synchronize()
        require(target.position == INITIAL, "Injection did not retain P main rows")

    try:
        require(target.config.model_config.enforce_eager and target.graph is None
                and target.fp32_head and target.attention_kernel == "sfa",
                "Actual target must be eager SFA with FP32 head")
        require("float32" in target.tp_reduction,
                "Keep the production FP32 TP reduction")
        report.update(inputs=inputs, teacher=teacher, reference=reference,
                      checkpoint_sha256=checkpoint,
                      source_sha256={**source_identity(), **target.source_metadata()},
                      packages={name: version(name) for name in
                                ("torch", "torch-npu", "vllm", "vllm-ascend")},
                      device=str(target.device), model_tp_reduction=target.tp_reduction,
                      device_properties=str(properties),
                      execution={"widths": [1, 32], "layers": 78, "fp32_head": True,
                                 "attention_kernel": target.attention_kernel,
                                 "enforce_eager": target.config.model_config.enforce_eager,
                                 "compile_target": False, "super_kernel": False,
                                 "graph": None, "mtp_model_loaded": False,
                                 "capacity": target.capacity},
                      loaded_libraries_sha256=mapped_libraries(), npu_executed=True)
        common = {"checkpoint": checkpoint, "prompt": inputs["prompt_sha256"],
                  "capture": inputs["capture_manifest_sha256"],
                  "teacher": teacher["sha256"], "mode": report["mode"],
                  "reference": None if reference is None else reference["manifest_sha256"],
                  "sources": report["source_sha256"], "packages": report["packages"]}
        require(all(value == common for value in gather(common)),
                "TP16 ranks do not share source/checkpoint/input/reference identities")
        checked("prepare.serial", lambda: prepare(True))
        index_dtype = route_dtype = None
        predictions = []
        for row, token in enumerate(teacher["teacher_tokens"]):
            actual = checked(f"capture.serial.{row}", lambda: capture(target, (token,), True))
            save(f"serial-{row:02d}.pt", actual)
            check_record(f"serial.{row}", actual, (token,), checks)
            selected = single_row(actual, 0, choices=True)
            returned_ids(f"serial.{row}", selected, checks)
            if reference is not None:
                strict_row(f"hf.serial.{row}", selected, expected(row), checks)
            predictions.append(int(selected["token_ids"][0]))
            if row == 0:
                index_dtype, route_dtype = actual["indices"][0].dtype, actual["routes"][3].dtype
        checked("prefix.serial", lambda: check_prefix(target, prefixes, checks, "serial"))
        states.append({"phase": "serial", "position": target.position, "predictions": predictions})

        producer = checked("observer.construct", lambda: PersistentTrace(
            target, index_dtype=index_dtype, route_dtype=route_dtype))
        checked("observer.install", producer.__enter__)
        try:
            for epoch, phase in enumerate(("wide", "reset"), start=1):
                checked("prepare." + phase, lambda: prepare(phase != "reset"))
                checked("begin." + phase, lambda: producer.begin(epoch))
                result = checked("verify." + phase, lambda: target.verify(
                    teacher["teacher_tokens"], full_logits=True))
                torch.npu.synchronize()
                with torch.npu.device(target.device), torch.inference_mode(), (
                    set_current_vllm_config(target.config)
                ):
                    values = checked("readback." + phase, lambda: producer.readback(
                        result, teacher["teacher_tokens"], INITIAL))
                actual = wide_trace(values, teacher["teacher_tokens"], INITIAL, target.position)
                save(phase + ".pt", actual)
                save(phase + "-stamps.pt", producer.seen.detach().cpu().clone())
                for row in range(WIDTH):
                    selected = single_row(actual, row, choices=True)
                    returned_ids(f"{phase}.{row}", selected, checks)
                    if reference is not None:
                        strict_row(f"hf.{phase}.{row}", selected, expected(row), checks)
                    serial = torch.load(directory / f"serial-{row:02d}.pt",
                                        map_location="cpu", weights_only=True)
                    strict_row(f"serial_vs_{phase}.{row}", selected,
                               single_row(serial, 0, choices=True), checks)
                if phase == "reset":
                    previous = torch.load(directory / "wide.pt", map_location="cpu", weights_only=True)
                    for row in range(WIDTH):
                        strict_row(f"reset_exact.{row}", single_row(actual, row, choices=True),
                                   single_row(previous, row, choices=True), checks, exact=True)
                checked("prefix." + phase, lambda: check_prefix(target, prefixes, checks, phase))
                states.append({"phase": phase, "position": target.position,
                               "predictions": actual["token_ids"].tolist(),
                               "epoch": epoch, "stamp_count": producer.seen.numel()})
        finally:
            producer.__exit__(None, None, None)
        target.reset()
        checks.append({"name": "final.reset_position", "passed": target.position == 0})
        execution = [item for item in checks if not item["name"].startswith("hf.")]
        hf = [item for item in checks if item["name"].startswith("hf.")]
        execution_passed = bool(execution) and all(item["passed"] for item in execution)
        hf_passed = bool(hf) and all(item["passed"] for item in hf)
        report.update(trace_sha256=artifacts, states=states, final_position=target.position,
                      capture_complete=True, loaded_libraries_sha256=mapped_libraries(),
                      **verdict(args.capture_only, execution_passed, hf_passed, reference is not None))
        path = args.output / f"rank-{rank}.json"
        atomic_json(path, report)
        receipts = gather({"rank": rank, "report_sha256": digest(path),
                           "execution_passed": execution_passed, "hf_passed": hf_passed,
                           "case_precision_passed": report["case_precision_passed"],
                           "states": states, "final_position": target.position})
        agreement = all(value["states"] == receipts[0]["states"]
                        and value["final_position"] == 0 for value in receipts)
        result = verdict(args.capture_only,
                         agreement and all(value["execution_passed"] for value in receipts),
                         agreement and all(value["hf_passed"] for value in receipts),
                         reference is not None)
        if rank == 0:
            atomic_json(args.output / "summary.json", {
                **result, "capture_complete": True,
                "scope": report["scope"], "ranks": receipts,
                "all_rank_state_agreement": agreement, "performance_measured": False,
                "compiled_graph_verified": False, "super_kernel_verified": False,
                "checkpoint_sha256": checkpoint, "source_sha256": report["source_sha256"],
                "teacher_sha256": teacher["sha256"],
                "reference_manifest_sha256": None if reference is None else reference["manifest_sha256"]})
        return result["entry_passed"]
    finally:
        target.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "capture-case", "teacher", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--check-inputs-only", action="store_true")
    args = parser.parse_args()
    if (args.capture_only and args.reference is not None
            or not args.capture_only and args.reference is None):
        parser.error("Strict mode requires --reference; --capture-only must omit it")
    if args.capacity < INITIAL + WIDTH or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("Capacity must cover 8K+32; memory utilization must be in (0,1]")
    if any(os.environ.get(name) for name in ("PROFILING_MODE", "PROFILER_SAMPLECONFIG")):
        parser.error("Run a numerical capture without profiling")
    rank = int(os.environ.get("RANK", "0"))
    args.output.mkdir(parents=True, exist_ok=True)
    if ((args.output / f"rank-{rank}.json").exists()
            or (args.output / f"rank-{rank}").exists()):
        parser.error("Use a new output directory; existing rank artifacts are preserved")
    report = {"schema": "cann-single-case-eager-v1", "rank": rank, "world_size": 16,
              "created_at": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
              "source_commit": args.source_commit, "command": [sys.executable, *sys.argv],
              "scope": ("one frozen 8K/32-row TP16 CANN eager capture; HF not executed"
                        if args.capture_only else
                        "one frozen 8K/32-row TP16 CANN eager case vs independent HF"),
              "mode": "capture_only" if args.capture_only else "strict_hf",
              "status": "error", "checks": [], "source_sha256": source_identity(),
              "case_precision_passed": False, "precision_verified": False,
              "formal_precision_gate_passed": False, "timed": False, "profiled": False,
              "performance_measured": False, "npu_executed": False,
              "capture_complete": False, "capture_checks_passed": False,
              "independent_hf_compared": False,
              "compiled_graph_verified": False, "super_kernel_verified": False}
    try:
        passed = run(args, report, rank)
    except Exception:
        passed = False
        report.update(status="captured_with_failures" if args.capture_only else "error",
                      case_precision_passed=False, error=traceback.format_exc())
    atomic_json(args.output / f"rank-{rank}.json", report)
    print(json.dumps({"rank": rank, "status": report["status"],
                      "case_precision_passed": report["case_precision_passed"],
                      "entry_passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
