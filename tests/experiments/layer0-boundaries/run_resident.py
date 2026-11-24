"""Capture one real resident TP16 M1 frame and its exact reset."""

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
from importlib import util
from pathlib import Path
from unittest.mock import patch

PREFIX_SUMMARY_SHA = "e433433cbc690b66a0d23775c0a7fb6ef5a0f3524b127a3a40f154d874e8834b"
CACHED_ATTENTION_SHA = "a531c7db7921e9faad64259ee7810d3b3439b4307b116e6f20de64ce5fdf9a43"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prefix_values(directory, rank, checkpoint, inputs):
    import torch

    summary_path = directory / "summary.json"
    summary_sha = digest(summary_path)
    if summary_sha != PREFIX_SUMMARY_SHA:
        raise ValueError("Prefix Value summary differs from the verified preparation")
    summary = json.loads(summary_path.read_text())
    ranks = summary["ranks"]
    if (summary["schema"] != "cpu-prefix-values-v1"
            or summary["status"] != "prepared"
            or summary["layers"] != list(range(78))
            or summary["requested_ranks"] != list(range(16))
            or len(ranks) != 16
            or any(type(item["rank"]) is not int for item in ranks)
            or sorted(item["rank"] for item in ranks) != list(range(16))
            or summary["checkpoint"]["metadata_sha256"] != checkpoint):
        raise ValueError("Prefix Value preparation scope/checkpoint differs")
    saved = next(item for item in ranks if item["rank"] == rank)
    receipt_path = directory / f"rank-{rank}.json"
    receipt_sha = digest(receipt_path)
    if receipt_sha != saved["receipt_sha256"]:
        raise ValueError("Prefix Value rank receipt bytes differ")
    receipt = json.loads(receipt_path.read_text())
    prefix = {key: value for key, value in receipt["prefix"].items()
              if key != "prefix_file"}
    expected = {key: value for key, value in inputs.items() if key != "prefix_file"}
    if (receipt["rank"] != rank or receipt["checkpoint"] != summary["checkpoint"]
            or receipt["shape"] != [78, 4, 8192, 256]
            or receipt["dtype"] != "torch.bfloat16"
            or [item["layer"] for item in receipt["layers"]] != list(range(78))
            or prefix != expected):
        raise ValueError("Prefix Value rank/prefix identity differs")
    path = directory / f"rank-{rank}.pt"
    file_sha = digest(path)
    if file_sha != receipt["file_sha256"] or file_sha != saved["file_sha256"]:
        raise ValueError("Prefix Value bank file bytes differ")
    bank = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (not isinstance(bank, torch.Tensor) or bank.device.type != "cpu"
            or bank.dtype != torch.bfloat16 or tuple(bank.shape) != (78, 4, 8192, 256)):
        raise ValueError("Prefix Value bank must be CPU BF16 [78,4,8192,256]")
    raw = bank.detach().contiguous().view(torch.uint8).reshape(-1).numpy()
    tensor_sha = hashlib.sha256(memoryview(raw)).hexdigest()
    if tensor_sha != receipt["tensor_sha256"] or tensor_sha != saved["tensor_sha256"]:
        raise ValueError("Prefix Value bank tensor bytes differ")
    return bank, {"directory": str(directory), "summary_sha256": summary_sha,
                  "rank_receipt_sha256": receipt_sha, "file_sha256": file_sha,
                  "tensor_sha256": tensor_sha, "prefix_sha256": inputs["prefix_sha256"],
                  "checkpoint_sha256": checkpoint}


def capture_routes(original, install, engine, *args, **kwargs):
    restorers, samples = [], {}
    try:
        for number, layer in enumerate(engine.model.model.layers[3:], start=3):
            samples[number] = []
            restorers.append(install(layer, samples[number]))
        result = original(engine, *args, **kwargs)
    finally:
        for restore in reversed(restorers):
            restore()
        restorers.clear()
    receipts = {}
    for number, rows in samples.items():
        if len(rows) != 1:
            raise ValueError(f"Expected one actual route for layer {number}")
        row = rows[0]
        receipts[number] = {
            "metadata": row["metadata"],
            **{
                name: row[name].cpu().clone()
                for name in ("ids", "weights", "router_logits")
            },
        }
    result["routing_receipts"] = receipts
    return result


def capture_expanded(engine, args, legacy, install, restore, sources,
                     *, candidate="expanded", prefix_directory=None):
    import torch
    from run_native_decode import compare_tensor
    from torch import distributed
    from vllm.distributed import get_world_group

    parent_path = args.output / f"rank-{engine.rank}.json"
    parent = json.loads(parent_path.read_text())
    prefixes, _, inputs = legacy.load_capture(
        args.capture_case, engine.rank, parent["checkpoint_sha256"]
    )
    assert inputs == parent["inputs"]
    reference_path = args.reference / "step-08192.pt"
    manifest = json.loads((args.reference / "manifest.json").read_text())
    assert (
        legacy.digest(reference_path)
        == manifest["trace_sha256"][reference_path.name]
    )
    expected = torch.load(
        reference_path, map_location="cpu", weights_only=True
    )
    bank, bank_identity = None, None
    if prefix_directory is not None:
        error = None
        try:
            bank, bank_identity = load_prefix_values(
                prefix_directory, engine.rank, parent["checkpoint_sha256"], inputs,
            )
        except Exception as failure:  # noqa: BLE001 - All ranks must rendezvous on admission errors.
            error = repr(failure)
        errors = [None] * 16
        distributed.all_gather_object(
            errors, {"rank": engine.rank, "bank_error": error},
            group=get_world_group().cpu_group,
        )
        if any(item["bank_error"] is not None for item in errors):
            raise ValueError(f"Prefix Value admission failed: {errors}")
    output = args.output.parent / candidate
    directory = output / f"rank-{engine.rank}"
    directory.mkdir(parents=True, exist_ok=False)
    for number, layer in enumerate(engine.model.model.layers):
        impl = layer.self_attn.mla_attn.mla_attn.impl
        restore.append(install(impl) if bank is None
                       else install(impl, bank[number].to(engine.device)))

    def prepare():
        engine.reset()
        assert engine.position == 0
        engine.storage.zero_()
        engine.inject(prefixes)
        assert engine.position == 8192

    def capture():
        return legacy.capture_step(
            engine,
            expected["token"],
            8192,
            True,
            mlp_layer=args.trace_mlp_layer,
        )

    prepare()
    actual = capture()
    if bank_identity is not None:
        actual["attention_recipe"]["prefix_values"] = bank_identity
    path = directory / "step-08192.pt"
    with path.open("xb") as stream:
        torch.save(actual, stream)
    checks, hf_checks = [], []
    compare_tensor(
        "ids_vs_actual_logits",
        actual["token_ids"],
        actual["logits"].argmax(-1),
        checks,
        exact=True,
    )
    legacy.check_prefix(engine, prefixes, checks, "decode")
    legacy.compare_step("hf.0", actual, expected, hf_checks)
    prepare()
    repeated = capture()
    if bank_identity is not None:
        repeated["attention_recipe"]["prefix_values"] = bank_identity
    reset_path = directory / "reset-step-08192.pt"
    with reset_path.open("xb") as stream:
        torch.save(repeated, stream)
    legacy.compare_step("reset.0", repeated, actual, checks, exact=True)
    legacy.check_prefix(engine, prefixes, checks, "reset")
    passed = all(row["passed"] for row in checks)
    report = {
        **parent,
        "scope": f"Same resident TP16 model, {candidate} attention M1 and exact reset",
        "status": "captured" if passed else "fail",
        "recipe": candidate,
        "attention_implementation": candidate,
        "prefix_values": bank_identity,
        "parent_control_report_sha256": legacy.digest(parent_path),
        "source_sha256": {
            **engine.source_metadata(),
            **sources,
            **actual["choice_source_sha256"],
        },
        "trace_sha256": {path.name: legacy.digest(path)},
        "reset_trace_sha256": {reset_path.name: legacy.digest(reset_path)},
        "checks": checks,
        "diagnostics": [],
        "hf_strict_checks": hf_checks,
        "hf_strict_checks_passed": all(row["passed"] for row in hf_checks),
        "repeatability_passed": passed,
        "predicted_tokens": actual["token_ids"].tolist(),
        "hf_diagnostics_complete": True,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "recipe_comparison": "Same model and world, restored KV; caches shared; no timing claim",
    }
    report_path = output / f"rank-{engine.rank}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    ranks = [None] * 16
    distributed.all_gather_object(
        ranks,
        {
            "rank": engine.rank,
            "passed": passed,
            "hf_strict_checks_passed": report["hf_strict_checks_passed"],
            "predicted_tokens": report["predicted_tokens"],
            "report_sha256": legacy.digest(report_path),
        },
        group=get_world_group().cpu_group,
    )
    if engine.rank == 0:
        summary = {
            "status": "captured"
            if all(row["passed"] for row in ranks)
            else "fail",
            "ranks": ranks,
            "precision_verified": False,
            "formal_precision_gate_passed": False,
            "performance_measured": False,
            "scope": report["scope"],
            "recipe": candidate,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary))
    return all(row["passed"] for row in ranks)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-inputs-only", action="store_true")
    parser.add_argument("--control-attention", choices=("original", "expanded"),
                        default="original")
    parser.add_argument("--candidate-prefix-values", type=Path)
    parser.add_argument(
        "--recipe", choices=("control", "expanded", "paired"), required=True
    )
    options, remaining = parser.parse_known_args()
    sys.path[:0] = [str(options.source), str(options.source / "tests/glm52")]
    legacy = importlib.import_module("run_cann_model")
    from expanded_attention import install
    from routes import install_routes

    cached_install = None
    if options.candidate_prefix_values is not None:
        if options.recipe != "paired":
            parser.error("Prefix Value candidates require --recipe paired")
        path = Path(__file__).resolve().with_name("cached_attention.py")
        if digest(path) != CACHED_ATTENTION_SHA:
            raise ValueError("Cached attention helper differs from the measured source")
        spec = util.spec_from_file_location("resident_cached_attention", path)
        helper = util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        cached_install = helper.install

    with patch.object(sys, "argv", [sys.argv[0], *remaining]):
        args = legacy.parse_arguments()
    if (
        args.steps != 1
        or not args.layer_trace
        or args.reference is None
        or args.capture_case is None
        or args.tp_reduction != "default"
        or args.device_offset != 0
        or args.trace_mlp_layer != 6
    ):
        parser.error(
            "Requires one captured M1, layer trace, HF and MoE layer6"
        )
    if options.check_inputs_only:
        from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

        rank = int(os.environ.get("RANK", "0"))
        metadata = checkpoint_fingerprint(args.checkpoint)
        prefixes, _, inputs = legacy.load_capture(
            args.capture_case, rank, metadata
        )
        reference = json.loads((args.reference / "manifest.json").read_text())
        legacy.validate_reference_manifest(reference, metadata, fp32_head=True)
        assert len(prefixes) == 78 and inputs["initial_position"] == 8192
        assert (
            reference["prefix_files_sha256"][f"rank-{rank}.pt"]
            == inputs["prefix_sha256"]
        )
        if options.candidate_prefix_values is not None:
            load_prefix_values(options.candidate_prefix_values, rank, metadata, inputs)
        print(
            json.dumps(
                {
                    "inputs_verified": True,
                    "rank": rank,
                    "npu_executed": False,
                    "script_sha256": digest(__file__),
                }
            )
        )
        return 0

    import torch_npu

    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    original_init = CannModel.__init__
    original_source = CannModel.source_metadata
    original_shutdown = CannModel.shutdown
    original_validate = legacy.validate_reference_manifest
    original_capture = legacy.capture_forward
    source_files = [
        Path(inspect.getfile(value)).resolve()
        for value in (
            original_init,
            original_validate,
            original_capture,
            install,
            install_routes,
            main,
        )
    ]
    sources = {str(path): digest(path) for path in source_files}
    if cached_install is not None:
        path = Path(inspect.getfile(cached_install)).resolve()
        sources[str(path)] = digest(path)
    engines, closed, restorers, paired_started = [], set(), {}, set()
    paired_failed = False
    control = ("expanded" if options.recipe == "expanded"
               else options.control_attention)
    candidate = "expanded" if cached_install is None else "cached_prefix"
    phase = {"recipe": "expanded" if options.recipe == "expanded" else "control",
             "implementation": control}
    matmul = torch_npu.npu.matmul
    initial_hf32 = matmul.allow_hf32
    matmul.allow_hf32 = False

    def initialize(engine, *engine_args, **engine_kwargs):
        engine_kwargs["fp32_head"] = True
        original_init(engine, *engine_args, **engine_kwargs)
        engines.append(engine)
        restorers[id(engine)] = []
        if control == "expanded":
            for layer in engine.model.model.layers:
                impl = layer.self_attn.mla_attn.mla_attn.impl
                restorers[id(engine)].append(install(impl))
        recipe = {
            "recipe": options.recipe,
            "control_attention": control,
            "candidate_attention": candidate,
            "layers": 78,
            "steps": 1,
            "fp32_head": engine.fp32_head,
            "graph": engine.graph is not None,
            "matmul_allow_hf32_before_experiment": initial_hf32,
            "matmul_allow_hf32_during_experiment": matmul.allow_hf32,
            "source_sha256": sources,
            "formal_precision_gate_passed": False,
            "performance_measured": False,
        }
        path = args.output / f"recipe-rank-{engine.rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recipe, indent=2) + "\n")

    def source_metadata(engine):
        return {**original_source(engine), **sources}

    def shutdown(engine):
        nonlocal paired_failed

        parent_path = args.output / f"rank-{engine.rank}.json"
        try:
            if (
                options.recipe == "paired"
                and id(engine) not in paired_started
                and parent_path.exists()
            ):
                from torch import distributed
                from vllm.distributed import get_world_group

                paired_started.add(id(engine))
                distributed.barrier(group=get_world_group().cpu_group)
                decision = json.loads((args.output / "summary.json").read_text())
                if decision["status"] in ("diagnostic", "captured"):
                    phase.update(recipe=candidate, implementation=candidate)
                    passed = capture_expanded(
                        engine,
                        args,
                        legacy,
                        install if cached_install is None else cached_install,
                        restorers[id(engine)],
                        sources,
                        candidate=candidate,
                        prefix_directory=options.candidate_prefix_values,
                    )
                    paired_failed = paired_failed or not passed
        finally:
            try:
                for restore in reversed(restorers.pop(id(engine), [])):
                    restore()
            finally:
                try:
                    original_shutdown(engine)
                finally:
                    closed.add(id(engine))

    def validate(reference, metadata):
        return original_validate(reference, metadata, fp32_head=True)

    def capture(engine, *capture_args, **capture_kwargs):
        result = capture_routes(
            original_capture,
            install_routes,
            engine,
            *capture_args,
            **capture_kwargs,
        )
        result["attention_recipe"] = {
            "recipe": phase["recipe"],
            "implementation": phase["implementation"],
            "matmul_allow_hf32": matmul.allow_hf32,
            "helper_sha256": (
                None if phase["implementation"] == "original" else sources[
                    str(Path(inspect.getfile(cached_install if phase["implementation"]
                                             == "cached_prefix" else install)).resolve())
                ]
            ),
        }
        return result

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    try:
        with (
            patch.object(CannModel, "__init__", initialize),
            patch.object(CannModel, "source_metadata", source_metadata),
            patch.object(CannModel, "shutdown", shutdown),
            patch.object(legacy, "validate_reference_manifest", validate),
            patch.object(legacy, "capture_forward", capture),
        ):
            try:
                result = legacy.execute(args)
            finally:
                for engine in engines:
                    if id(engine) not in closed:
                        shutdown(engine)
            return 1 if paired_failed else result
    finally:
        matmul.allow_hf32 = initial_hf32


if __name__ == "__main__":
    raise SystemExit(main())
