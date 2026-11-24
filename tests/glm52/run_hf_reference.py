"""Generate CPU HF W4A8C8 traces and check exact request-reset replay."""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import traceback

from prefix_io import prefix_prompt_identity, prompt_identity


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, required=True)
    parser.add_argument("--generate", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=9216)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--mla-preprocess", choices=("native", "prolog_v3"),
                        default="native")
    parser.add_argument("--layers", type=int, choices=range(1, 79), default=78)
    parser.add_argument("--fp32-head", action="store_true",
                        help="Keep BF16 head products in FP32 before output")
    parser.add_argument("--c8-source-directory", type=Path,
                        help="Verify the pinned CANN C8 headers here")
    args = parser.parse_args()
    if args.generate < 1 or args.capacity < 1:
        parser.error("Generation and cache capacity must be positive")
    return args


def validate_trace(trace, layer_count=78, *, fp32_head=False):
    position = trace["position"]
    if trace["committed_position"] != position + 1:
        raise ValueError("HF did not commit exactly one token")
    counts = {"hidden_states": layer_count + 1, "layer_logits": layer_count,
              "indices": layer_count, "routes": layer_count,
              "cache_rows": layer_count, "index_scores": layer_count,
              "attention_weights": layer_count}
    for name, count in counts.items():
        if len(trace[name]) != count:
            raise ValueError(f"Incomplete HF {name}")
    if fp32_head:
        import torch

        if any(value.dtype != torch.float32 for value in (
            trace["logits"], *trace["layer_logits"],
        )):
            raise ValueError("FP32 head requires FP32 final and layer logits")
    for layer, row in enumerate(trace["cache_rows"]):
        full = layer < 3 or (layer - 2) % 4 == 0
        quantized = full and 6 <= layer <= 66
        if row["latent"] is None or row["rope"] is None:
            raise ValueError(f"Missing MLA row at layer {layer}")
        if (row["index"] is not None) != full:
            raise ValueError(f"Wrong index row presence at layer {layer}")
        if (row["index_scale"] is not None) != quantized:
            raise ValueError(f"Wrong index scale presence at layer {layer}")
        if trace["indices"][layer] is None:
            raise ValueError(f"Missing attention selection at layer {layer}")
        if (trace["routes"][layer] is not None) != (layer >= 3):
            raise ValueError(f"Wrong routing presence at layer {layer}")
        scores = trace["index_scores"][layer]
        if (scores is not None) != full or (
            full and tuple(scores.shape) != (position + 1,)
        ):
            raise ValueError(f"Wrong index score shape at layer {layer}")
        if tuple(trace["attention_weights"][layer].shape) != (
            64, position + 1,
        ):
            raise ValueError(f"Wrong attention weight shape at layer {layer}")


def load_prefix(args, report):
    import torch

    if args.prefix is None:
        return None
    report["prefix_files_sha256"] = {}
    reference = None
    for rank in range(16):
        path = args.prefix / f"rank-{rank}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["metadata_sha256"] != report["metadata_sha256"]:
            raise ValueError("Prefix checkpoint metadata differs")
        if len(saved["layers"]) != 78:
            raise ValueError("Prefix must contain all 78 backbone layers")
        identity = prefix_prompt_identity(
            saved, saved["layers"][0]["latent"].shape[0],
        )
        if reference is None:
            reference = saved["layers"]
            report.update(identity)
        else:
            if any(report[name] != value for name, value in identity.items()):
                raise ValueError("Replicated TP prompt IDs differ")
            for left, right in zip(reference, saved["layers"]):
                if left.keys() != right.keys() or any(
                    left[name].dtype != right[name].dtype
                    or not torch.equal(left[name], right[name])
                    for name in left
                ):
                    raise ValueError("Replicated TP prefix contents differ")
        report["prefix_files_sha256"][path.name] = digest(path)
    report["prefix_scope"] = (
        "decode from the same frozen compressed prefix; prefill not checked"
    )
    return reference


def check_lengths(engine, expected, report, label):
    layer_count = len(engine.model.model.layers)
    lengths = [layer.get_seq_length() for layer in engine.cache.layers]
    index_lengths = []
    for cached, layer in zip(engine.cache.layers, engine.model.model.layers):
        keys = cached.indexer_keys
        if layer.self_attn.indexer is None:
            index_lengths.append(None if keys is None else keys.shape[1])
        else:
            index_lengths.append(0 if keys is None else keys.shape[1])
    scales = [
        layer.self_attn.indexer.scales
        for layer in engine.model.model.layers
        if (layer.self_attn.indexer is not None
            and layer.self_attn.indexer.quantized)
    ]
    report["checks"].append({
        "name": label, "expected_position": expected,
        "actual_position": engine.position, "mla_lengths": lengths,
        "index_lengths": index_lengths,
        "passed": len(lengths) == layer_count and engine.position == expected
        and all(length == expected for length in lengths)
        and all(length == (expected if layer < 3 or (layer - 2) % 4 == 0
                           else None)
                for layer, length in enumerate(index_lengths))
        and all((value.numel() if value is not None else 0) == expected
                for value in scales),
    })


def compare_replay(label, trace, previous, checks, layer_count):
    from run_native_decode import compare_tensor, compare_trace

    if layer_count == 78:
        compare_trace(label, trace, previous, checks, exact=True)
    else:
        checks.append({
            "name": label + ".position_and_input",
            "passed": all(trace[key] == previous[key] for key in (
                "token", "position", "committed_position")),
        })
        compare_tensor(label + ".logits", trace["logits"],
                       previous["logits"], checks, exact=True)
        for field in ("hidden_states", "layer_logits", "indices", "routes"):
            for layer, (left, right) in enumerate(zip(
                trace[field], previous[field],
            )):
                compare_tensor(f"{label}.{field}.{layer}", left, right,
                               checks, exact=True)
        for layer, (left, right) in enumerate(zip(
            trace["cache_rows"], previous["cache_rows"],
        )):
            for field in ("latent", "rope", "index", "index_scale"):
                compare_tensor(f"{label}.cache.{layer}.{field}",
                               left[field], right[field], checks, exact=True)
    compare_tensor(label + ".last_merged_hidden", trace["last_merged_hidden"],
                   previous["last_merged_hidden"], checks, exact=True)
    for field in ("index_scores", "attention_weights"):
        for layer, (left, right) in enumerate(zip(
            trace[field], previous[field],
        )):
            compare_tensor(f"{label}.{field}.{layer}", left, right,
                           checks, exact=True)


def bind_reference_sources(engine, report):
    reader = type(engine.weights)
    loaded = {
        "tilert/models/glm_5_2_ascend/reference.py": type(engine),
        "tilert/models/glm_5_2_ascend/reference_weights.py": reader,
    }
    report["loaded_reference_sources"] = {
        name: {"path": str(Path(inspect.getsourcefile(cls)).resolve()),
               "sha256": digest(Path(inspect.getsourcefile(cls)))}
        for name, cls in loaded.items()
    }
    if any(value["sha256"] != report["source_sha256"][name]
           for name, value in report["loaded_reference_sources"].items()):
        raise ValueError(
            "Loaded HF reference source differs from the manifest"
        )


def run_reference(args, report):
    import torch
    import transformers
    from transformers import GlmMoeDsaForCausalLM
    from transformers import cache_utils
    from transformers.models.glm_moe_dsa import configuration_glm_moe_dsa
    from transformers.utils import output_capturing

    from tilert.models.glm_5_2_ascend.reference import HfReference

    torch.set_num_threads(8)
    report["versions"].update(
        torch=torch.__version__, transformers=transformers.__version__
    )
    report["installed_hf_source_sha256"] = {
        str(path): digest(path) for path in (
            Path(inspect.getsourcefile(GlmMoeDsaForCausalLM)),
            Path(configuration_glm_moe_dsa.__file__),
            Path(cache_utils.__file__), Path(output_capturing.__file__),
        )
    }
    engine = HfReference(args.checkpoint, args.mla_preprocess,
                         layer_count=args.layers, fp32_head=args.fp32_head,
                         c8_source_directory=args.c8_source_directory)
    bind_reference_sources(engine, report)
    report["weight_arithmetic"] = engine.weights.contract
    if args.mla_preprocess == "prolog_v3":
        report["expert_combine"] = engine.weights.contract["expert_combine"]
        report["fused_expert_quantization"] = (
            engine.weights.contract["fused_expert_quantization"]
        )
        report["fused_swiglu"] = (
            "Dense/shared: FP32 SiLU/product, BF16 output; "
            "routed: FP32 GMM1/SiLU/product before fused A8 quantization"
        )
    prefixes = load_prefix(args, report)
    if prefixes is not None:
        engine.inject(prefixes)
    initial = engine.position
    steps = len(args.tokens) + args.generate - 1
    if initial + steps > args.capacity:
        raise ValueError("Prefix, input and generation exceed cache capacity")
    report["initial_position"] = initial
    check_lengths(engine, initial, report, "initial.cache_lengths")
    consumed, generated = [], []
    token = args.tokens[0]
    with torch.inference_mode():
        for offset in range(steps):
            if offset < len(args.tokens):
                token = args.tokens[offset]
            trace = engine.step(token)
            validate_trace(trace, args.layers, fp32_head=args.fp32_head)
            check_lengths(engine, initial + offset + 1, report,
                          f"step{initial + offset}.cache_lengths")
            name = f"step-{initial + offset:05d}.pt"
            path = args.output / name
            torch.save(trace, path)
            report["trace_sha256"][name] = digest(path)
            consumed.append(token)
            token = int(trace["logits"].argmax(-1).item())
            if offset >= len(args.tokens) - 1:
                generated.append(token)
        engine.reset()
        check_lengths(engine, 0, report, "reset.cache_lengths")
        if prefixes is not None:
            engine.inject(prefixes)
        check_lengths(engine, initial, report, "reinjected.cache_lengths")
        for offset, token in enumerate(consumed):
            trace = engine.step(token)
            validate_trace(trace, args.layers, fp32_head=args.fp32_head)
            check_lengths(engine, initial + offset + 1, report,
                          f"replay{initial + offset}.cache_lengths")
            previous = torch.load(
                args.output / f"step-{initial + offset:05d}.pt",
                map_location="cpu", weights_only=True,
            )
            validate_trace(previous, args.layers, fp32_head=args.fp32_head)
            compare_replay(f"reset.step{initial + offset}", trace, previous,
                           report["checks"], args.layers)
    report["teacher_tokens"] = consumed
    report["generated_tokens"] = generated
    report["position"] = engine.position
    report["numerical_execution"] = True
    report["status"] = (
        "pass" if all(item["passed"] for item in report["checks"]) else "fail"
    )


def run(args, report):
    import torch

    with torch.backends.mkldnn.flags(
        enabled=False, deterministic=None, allow_tf32=None,
        fp32_precision=None,
    ):
        report["cpu_matmul"] = {
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "scope": "Prefix expansion, HF forward and diagnostic projections",
        }
        run_reference(args, report)


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(__file__).with_name("run_native_decode.py"),
             Path(__file__).with_name("prefix_io.py")]
    files.extend(root / "tilert/models/glm_5_2_ascend" / name for name in (
        "reference.py", "reference_weights.py", "ops/indexer.py",
    ))
    if args.mla_preprocess == "prolog_v3":
        files.append(root / "tilert/models/glm_5_2_ascend/reference_prolog.py")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation": "transformers.GlmMoeDsaForCausalLM", "tp_size": 16,
        "scope": (
            "CPU HF traces with oneDNN disabled and selected MLA arithmetic; "
            "exact reset checks reference repeatability"
        ),
        "status": "running", "hostname": platform.node(),
        "git_head": args.source_commit, "command": [sys.executable, *sys.argv],
        "checkpoint": str(args.checkpoint), "versions": {
            "python": platform.python_version(),
        },
        "prompt_tokens": args.tokens, "requested_generation": args.generate,
        "mla_preprocess": args.mla_preprocess,
        "layer_count": args.layers,
        "fp32_head": args.fp32_head,
        "requested_c8_source_directory": (
            str(args.c8_source_directory) if args.c8_source_directory else None
        ),
        "last_merged_hidden": "Last decoder output before the final RMSNorm",
        "partial_layer_logits": (
            "Use the full-model intermediate norm/head projection formula"
        ),
        "attention_weights": "HF expanded-KV softmax weights; BF16 [64,P]",
        "index_scores": "Masked indexer scores [P]; None on shared layers",
        "rms_norm_eps": {"ordinary": 1e-5, "q_a": 1e-5, "kv_a": 1e-5},
        "index_rope_basis": "interleaved",
        "tie_rule": "stable descending score; ascending expert/position ID",
        "fused_swiglu": "FP32 SiLU and product, one BF16 output cast",
        "expert_combine": "BF16 probabilities; FP32 weighted sum; BF16 output",
        "timed": False, "profiled": False, "trace_sha256": {}, "checks": [],
        "prefix_files_sha256": {},
        **prompt_identity(args.tokens),
    }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["source_sha256"] = {
            str(path.relative_to(root)): digest(path) for path in files
        }
        report["metadata_sha256"] = {
            name: digest(args.checkpoint / name) for name in (
                "config.json", "quant_model_description.json",
                "quant_model_weights.safetensors.index.json",
            )
        }
        run(args, report)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(
        not item["passed"] for item in report["checks"]
    )
    manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(manifest)}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
