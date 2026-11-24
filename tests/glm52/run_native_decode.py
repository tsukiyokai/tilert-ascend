"""Capture TP16 W4A8C8 decode and compare a matching HF reference trace."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_reference_manifest(reference, metadata, *, fp32_head=False):
    from tilert.models.glm_5_2_ascend.ops.indexer import (
        cann_arch22_scoring_contract,
    )

    if reference["implementation"] != "transformers.GlmMoeDsaForCausalLM":
        raise ValueError("Reference must identify the HF implementation")
    if reference.get("status") != "pass":
        raise ValueError("HF trace generation and exact replay must pass")
    if reference.get("numerical_execution") is not True:
        raise ValueError("HF reference requires numerical execution")
    if reference["tp_size"] != 16 or reference["metadata_sha256"] != metadata:
        raise ValueError("HF reference TP layout or checkpoint mismatch")
    if (not reference["source_sha256"]
            or not reference.get("installed_hf_source_sha256")):
        raise ValueError("HF reference needs generator and installed hashes")
    head_mode = reference.get("fp32_head", False)
    if type(head_mode) is not bool or head_mode != fp32_head:
        raise ValueError("HF reference head precision differs from execution")
    head = reference.get("weight_arithmetic", {}).get("lm_head")
    if fp32_head or head is not None:
        expected_head = {
            "input_dtype": "bfloat16", "weight_dtype": "bfloat16",
            "accumulator_dtype": "float32",
            "projection_output_dtype": "float32" if fp32_head else "bfloat16",
            "tensor_parallel_size": 16, "vocabulary_range": [0, 154880],
        }
        if head != expected_head:
            raise ValueError("HF reference head arithmetic contract differs")
    if fp32_head:
        sources = (
            "tests/glm52/run_hf_reference.py",
            "tilert/models/glm_5_2_ascend/reference.py",
            "tilert/models/glm_5_2_ascend/reference_weights.py",
        )
        root = Path(__file__).resolve().parents[2]
        for name in sources:
            expected_hash = digest(root / name)
            if reference["source_sha256"].get(name) != expected_hash:
                raise ValueError("FP32 head reference source differs: " + name)
            if name.startswith("tilert/"):
                loaded = reference.get("loaded_reference_sources", {}).get(
                    name, {},
                )
                if (loaded.get("sha256") != expected_hash
                        or not isinstance(loaded.get("path"), str)
                        or not loaded["path"]):
                    raise ValueError(
                        "FP32 head loaded source differs: " + name
                    )
    expected = cann_arch22_scoring_contract()
    scorer = "tilert/models/glm_5_2_ascend/ops/indexer.py"
    scorer_hash = reference["source_sha256"].get(scorer)
    if scorer_hash != expected["reference_source_sha256"]:
        raise ValueError("HF reference needs the current C8 scorer hash")
    contract = reference.get("weight_arithmetic", {}).get(
        "c8_index_scoring", {}
    )
    if (contract.get("source_verified") is not True
            or contract.get("implementation") != expected["implementation"]
            or contract.get("source_sha256") != expected["source_sha256"]
            or contract.get("reference_source_sha256") != scorer_hash):
        raise ValueError("HF C8 scoring source identity differs")
    if not reference.get("trace_sha256"):
        raise ValueError("HF reference needs trace-file hashes")
    if reference["rms_norm_eps"] != {
        "ordinary": 1e-5, "q_a": 1e-5, "kv_a": 1e-5,
    } or reference["index_rope_basis"] != "interleaved":
        raise ValueError("HF reference must match Ascend prefill numerics")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", nargs="+", type=int, required=True,
        help="Prompt tokens, or tokens consumed after an injected prefix",
    )
    parser.add_argument("--generate", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=9216)
    parser.add_argument("--reference", type=Path,
                        help="HF quantized-reference trace directory")
    parser.add_argument(
        "--prefix", type=Path,
        help="Directory containing rank-N.pt backbone prefixes",
    )
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not args.tokens or args.generate < 1:
        parser.error("A prompt and at least one generated token are required")
    if len(args.tokens) + args.generate - 1 > args.capacity:
        parser.error("Prompt and generation exceed cache capacity")
    return args


def snapshot(engine, result, token, position, layer_logits):
    def cpu(value):
        return value.detach().cpu().clone() if value is not None else None

    rows = []
    for layer in engine.layers:
        entry = {
            "latent": cpu(layer.mla.latent_cache.view(-1, 512)[position]),
            "rope": cpu(layer.mla.rope_cache.view(-1, 64)[position]),
            "index": None, "index_scale": None,
        }
        if layer.indexer is not None:
            entry["index"] = cpu(layer.indexer.keys.view(-1, 128)[position])
            if layer.indexer.scales is not None:
                entry["index_scale"] = cpu(
                    layer.indexer.scales.view(-1)[position:position + 1]
                )
        rows.append(entry)
    return {
        "token": token, "position": position,
        "committed_position": engine.position,
        "logits": cpu(result.logits),
        "hidden_states": tuple(cpu(value) for value in result.hidden_states),
        "indices": tuple(cpu(value) for value in result.indices),
        "routes": tuple(cpu(value) for value in result.routes),
        "layer_logits": tuple(cpu(value) for value in layer_logits),
        "cache_rows": tuple(rows),
    }


def compare_tensor(
    name, actual, expected, checks, *, exact=False, logits=False, max_ulp=None,
):
    import torch
    import torch.nn.functional as functional

    item = {"name": name, "passed": False, "exact": exact}
    checks.append(item)
    if actual is None or expected is None:
        item["passed"] = actual is None and expected is None
        return
    item.update(shape=list(actual.shape), expected_shape=list(expected.shape),
                dtype=str(actual.dtype), expected_dtype=str(expected.dtype))
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return
    left, right = actual.float(), expected.float()
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        return
    norm = right.norm().item()
    error = left - right
    nrmse = error.norm().item() / max(norm, 1e-12)
    cosine = functional.cosine_similarity(
        left.flatten(), right.flatten(), dim=0
    ).item() if norm else float(torch.count_nonzero(left) == 0)
    max_abs = error.abs().max().item()
    item.update(nrmse=nrmse, cosine=cosine, max_abs=max_abs)
    if max_ulp is not None and actual.is_floating_point():
        integer_dtype = {torch.bfloat16: torch.int16,
                         torch.float16: torch.int16,
                         torch.float32: torch.int32}[actual.dtype]
        sign = 1 << (actual.element_size() * 8 - 1)

        def ordered(value):
            bits = value.contiguous().view(integer_dtype).to(torch.int64)
            return torch.where(bits < 0, -sign - bits, bits)

        distance = (ordered(actual) - ordered(expected)).abs()
        observed = int(distance.max().item())
        item.update(max_ulp=observed, allowed_ulp=max_ulp,
                    bit_exact=torch.equal(actual.view(integer_dtype),
                                          expected.view(integer_dtype)))
        item["passed"] = observed <= max_ulp
    elif exact:
        item["passed"] = torch.equal(actual, expected)
    else:
        item["passed"] = nrmse <= 0.01 and cosine >= 0.999
        if logits:
            item["passed"] &= max_abs <= 0.1
            item["passed"] &= torch.equal(
                actual.argmax(-1), expected.argmax(-1)
            )


def compare_trace(
    label, actual, expected, checks, *, exact=False,
    expected_committed_position=None,
):
    if expected_committed_position is None:
        expected_committed_position = expected["position"] + 1
    checks.append({
        "name": label + ".position_and_input",
        "passed": actual["token"] == expected["token"]
        and actual["position"] == expected["position"],
    })
    checks.append({
        "name": label + ".committed_position",
        "passed": actual["committed_position"] == expected_committed_position,
    })
    compare_tensor(label + ".logits", actual["logits"], expected["logits"],
                   checks, exact=exact, logits=True)
    expected_lengths = {"hidden_states": 79, "indices": 78,
                        "routes": 78, "layer_logits": 78}
    for field in expected_lengths:
        left, right = actual[field], expected[field]
        checks.append({
            "name": label + "." + field + ".length",
            "passed": len(left) == len(right) == expected_lengths[field],
        })
        for layer, (value, reference) in enumerate(zip(left, right)):
            if (field in ("indices", "routes") and not exact
                    and value is not None):
                import torch

                if reference is not None:
                    integers = (torch.int32, torch.int64)
                    valid_dtype = (
                        value.dtype in integers and reference.dtype in integers
                    )
                    checks.append({
                        "name": f"{label}.{field}.{layer}.integer_dtype",
                        "passed": valid_dtype,
                        "actual_dtype": str(value.dtype),
                        "reference_dtype": str(reference.dtype),
                    })
                    if not valid_dtype:
                        continue
                    value = value.to(torch.int64).sort().values
                    reference = reference.to(torch.int64).sort().values
            compare_tensor(
                f"{label}.{field}.{layer}", value, reference, checks,
                exact=exact or field in ("indices", "routes"),
                logits=field == "layer_logits",
            )
    checks.append({
        "name": label + ".cache_rows.length",
        "passed": len(actual["cache_rows"]) == 78
        and len(expected["cache_rows"]) == 78,
    })
    for layer, (left, right) in enumerate(zip(
        actual["cache_rows"], expected["cache_rows"],
    )):
        for name in ("latent", "rope", "index", "index_scale"):
            value, reference = left[name], right[name]
            integer = value is not None and not value.is_floating_point()
            compare_tensor(
                f"{label}.cache.{layer}.{name}", value, reference, checks,
                exact=exact or integer or name == "index_scale",
            )


def layer_logits(engine, result):
    projected = [engine.project_logits(hidden)
                 for hidden in result.hidden_states[1:-1]]
    return (*projected, result.logits)


def check_lengths(report, label, engine, expected):
    mla = [layer.mla.length for layer in engine.layers]
    index = [layer.indexer.length if layer.indexer is not None else None
             for layer in engine.layers]
    report["checks"].append({
        "name": label, "expected_position": expected,
        "actual_position": engine.position, "mla_lengths": mla,
        "index_lengths": index,
        "passed": engine.position == expected and len(mla) == 78
        and all(length == expected for length in mla)
        and all((length is not None) == engine.config.has_indexer(number)
                for number, length in enumerate(index))
        and all(length == expected for length in index if length is not None),
    })


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu

    from tilert.models.glm_5_2_ascend.native_engine import NativeDecodeEngine

    if args.reference is not None:
        manifest = args.reference / "manifest.json"
        reference = json.loads(manifest.read_text())
        validate_reference_manifest(reference, report["metadata_sha256"])
        report["reference_manifest"] = reference
        report["reference_manifest_sha256"] = digest(manifest)
        report["reference_trace_sha256"] = {}
    torch.set_num_threads(1)
    device = f"npu:{int(os.environ['LOCAL_RANK'])}"
    torch.npu.set_device(device)
    distributed.init_process_group("hccl")
    rank = distributed.get_rank()
    report.update(
        rank=rank, world_size=distributed.get_world_size(), device=device
    )
    report["versions"].update(
        torch=torch.__version__, torch_npu=torch_npu.__version__
    )
    torch.npu.reset_peak_memory_stats(device)
    engine = NativeDecodeEngine(
        args.checkpoint, device, capacity=args.capacity
    )
    report["config"] = engine.config.__dict__
    report["rms_eps"] = engine.config.rms_norm_eps
    report["weight_policy"] = "routed experts load on first use; no timing"
    prefixes = None
    if args.prefix is not None:
        prefix_path = args.prefix / f"rank-{rank}.pt"
        prefix = torch.load(prefix_path, map_location="cpu", weights_only=True)
        if prefix["metadata_sha256"] != report["metadata_sha256"]:
            raise ValueError("Prefix and decode checkpoint do not match")
        prefixes = [
            {name: value.to(device) for name, value in layer.items()}
            for layer in prefix["layers"]
        ]
        engine.inject(prefixes)
        report["prefix_file_sha256"] = digest(prefix_path)
        if args.reference is not None and (
            reference["prefix_files_sha256"].get(prefix_path.name)
            != report["prefix_file_sha256"]
        ):
            raise ValueError("HF and native prefix artifacts differ")
    initial_position = engine.position
    if initial_position + len(args.tokens) + args.generate - 1 > args.capacity:
        raise ValueError("Injected prefix and decode exceed cache capacity")
    report["initial_position"] = initial_position
    if args.reference is not None and (
        reference["initial_position"] != initial_position
        or len(reference["teacher_tokens"])
        != len(args.tokens) + args.generate - 1
    ):
        raise ValueError("HF and native position or trace length differ")
    report["decoding_mode"] = (
        "teacher_forced_reference" if args.reference else "greedy"
    )
    check_lengths(report, "initial.cache_lengths", engine, initial_position)
    trace_directory = args.output / "trace"
    if rank == 0:
        trace_directory.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    token = args.tokens[0]
    teacher_tokens, generated = [], []
    steps = len(args.tokens) + args.generate - 1
    with torch.inference_mode():
        for offset in range(steps):
            position = initial_position + offset
            if offset < len(args.tokens):
                token = args.tokens[offset]
            reference = None
            if args.reference is not None:
                if rank == 0:
                    reference_path = args.reference / f"step-{position:05d}.pt"
                    reference = torch.load(reference_path, map_location="cpu",
                                           weights_only=True)
                    actual_hash = digest(reference_path)
                    expected_hash = report["reference_manifest"][
                        "trace_sha256"
                    ].get(reference_path.name)
                    report["reference_trace_sha256"][reference_path.name] = (
                        actual_hash
                    )
                    if actual_hash != expected_hash:
                        raise ValueError(
                            "HF trace differs from its manifest: "
                            + reference_path.name
                        )
                    if (offset < len(args.tokens)
                            and reference["token"] != token):
                        raise ValueError("Input tokens differ from HF trace")
                    token = int(reference["token"])
                chosen = torch.tensor(
                    [token], dtype=torch.int64, device=device
                )
                distributed.broadcast(chosen, src=0)
                token = int(chosen.item())
            result = engine.step(token, trace=True)
            projected = layer_logits(engine, result)
            check_lengths(
                report, f"step{position}.cache_lengths", engine, position + 1
            )
            teacher_tokens.append(token)
            token = int(result.logits.argmax(dim=-1).item())
            if offset >= len(args.tokens) - 1:
                generated.append(token)
            if rank == 0:
                actual = snapshot(engine, result, teacher_tokens[-1],
                                  position, projected)
                torch.save(actual, trace_directory / f"step-{position:05d}.pt")
                if args.reference is not None:
                    compare_trace(f"hf.step{position}", actual, reference,
                                  report["checks"])

        engine.reset()
        check_lengths(report, "reset.cache_lengths", engine, 0)
        if prefixes is not None:
            engine.inject(prefixes)
            check_lengths(
                report, "reinjected.cache_lengths", engine, initial_position
            )
        for offset, token in enumerate(teacher_tokens):
            position = initial_position + offset
            result = engine.step(token, trace=True)
            projected = layer_logits(engine, result)
            check_lengths(report, f"reset.step{position}.cache_lengths",
                          engine, position + 1)
            if rank == 0:
                previous = torch.load(
                    trace_directory / f"step-{position:05d}.pt",
                    map_location="cpu", weights_only=True,
                )
                compare_trace(
                    f"reset.step{position}",
                    snapshot(engine, result, token, position, projected),
                    previous, report["checks"], exact=True,
                )
    report["teacher_tokens"] = teacher_tokens
    report["generated_tokens"] = generated
    report["position"] = engine.position
    report["loaded_experts_by_layer"] = [
        len(layer.experts) for layer in engine.layers
    ]
    report["peak_tensor_bytes"] = torch.npu.max_memory_allocated(device)
    report["peak_reserved_bytes"] = torch.npu.max_memory_reserved(device)
    failed = torch.tensor(
        [int(any(not check["passed"] for check in report["checks"]))],
        dtype=torch.int32, device=device,
    )
    distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
    report["any_rank_failed"] = bool(failed.item())
    if rank == 0:
        report["precision"] = (
            "HF reference compared" if args.reference
            else "HF reference pending"
        )
        report["status"] = (
            "fail" if report["any_rank_failed"]
            else "pass" if args.reference else "needs_reference"
        )
    else:
        report["status"] = (
            "fail" if report["any_rank_failed"] else "rank_forward_complete"
        )
    distributed.barrier()
    distributed.destroy_process_group()


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    model = root / "tilert/models/glm_5_2_ascend"
    sources = [Path(__file__), *model.rglob("*.py")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "hostname": platform.node(), "git_head": args.source_commit,
        "command": [sys.executable, *sys.argv],
        "checkpoint": str(args.checkpoint),
        "versions": {"python": platform.python_version()},
        "reference": str(args.reference) if args.reference else None,
        "prefix": str(args.prefix) if args.prefix else None,
        "implementation": "NativeDecodeEngine",
        "timed": False, "profiled": False, "checks": [],
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
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(
        not item["passed"] for item in report["checks"]
    )
    rank = int(os.environ.get("RANK", "0"))
    output = args.output / f"rank-{rank}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "rank": rank, "status": report["status"], "report": str(output),
    }))
    return int(report["status"] not in ("pass", "rank_forward_complete"))


if __name__ == "__main__":
    raise SystemExit(main())
