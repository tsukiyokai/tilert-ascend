"""Check HF layer arithmetic at saved native hidden states and KV on CPU."""

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import torch
import transformers

from run_native_decode import compare_trace, digest
from tilert.models.glm_5_2_ascend.reference import HfReference


def frozen_prefix(base, history):
    if not base and not history:
        return None
    prefixes = []
    regions = {"latent": "latent", "rope": "rope",
               "index_keys": "index", "index_scales": "index_scale"}
    for layer in range(78):
        entry = {}
        for name, field in regions.items():
            values = ([base[layer][name]]
                      if base and name in base[layer] else [])
            for trace in history:
                row = trace["cache_rows"][layer][field]
                if row is not None:
                    values.append(row.reshape(1) if name == "index_scales"
                                  else row.unsqueeze(0))
            if values:
                entry[name] = torch.cat(values)
        prefixes.append(entry)
    return prefixes


def local_trace(engine, actual, outputs):
    tokens = torch.tensor([[actual["token"]]])
    engine.indices, engine.routes, engine.latents = (
        [None] * 78 for _ in range(3)
    )
    result = engine.model(
        input_ids=tokens, past_key_values=engine.cache, use_cache=True,
        output_hidden_states=False, return_dict=True,
    )
    engine.cache = result.past_key_values
    hidden = [outputs[layer][:, -1] for layer in range(78)]
    normalized = engine.model.model.norm(torch.cat(hidden[:-1]))
    projected = engine.model.lm_head(normalized)
    final_hidden = engine.model.model.norm(hidden[-1])
    final_logits = result.logits[:, -1]
    rows = []
    for number, layer in enumerate(engine.model.model.layers):
        indexer = layer.self_attn.indexer
        rows.append({
            "latent": engine.latents[number],
            "rope": engine.cache.layers[number].keys[0, 0, -1, 192:].clone(),
            "index": None if indexer is None else indexer.last_key.clone(),
            "index_scale": (None if indexer is None or not indexer.quantized
                            else indexer.last_scale.clone()),
        })
    return {
        "token": actual["token"], "position": actual["position"],
        "committed_position": engine.position,
        "logits": final_logits,
        "hidden_states": (
            engine.model.model.embed_tokens(tokens)[:, -1],
            *hidden[:-1], final_hidden,
        ),
        "layer_logits": (*projected.split(1), final_logits),
        "indices": tuple(engine.indices), "routes": tuple(engine.routes),
        "cache_rows": tuple(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report_path = args.native_run / "rank-0.json"
    native = json.loads(report_path.read_text())
    if "error" in native or not native.get("teacher_tokens"):
        raise ValueError("A completed native trace is required")
    engine = HfReference(args.checkpoint)
    if (engine.weights.contract["metadata_sha256"]
            != native["metadata_sha256"]):
        raise ValueError("Checkpoint metadata differs from the native run")
    base = None
    artifacts = {str(report_path): digest(report_path)}
    if native.get("prefix"):
        path = Path(native["prefix"]) / "rank-0.pt"
        if digest(path) != native["prefix_file_sha256"]:
            raise ValueError("Native prefix hash differs")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        base = saved["layers"]
        artifacts[str(path)] = digest(path)

    state, hooks = {}, []
    for number, layer in enumerate(engine.model.model.layers):
        def inject(module, positional, keywords, index=number,
                   shared=layer.self_attn.indexer is None):
            hidden = state["actual"]["hidden_states"][index].unsqueeze(1)
            if shared:
                indices = state["actual"]["indices"][index].to(torch.int64)
                keywords = {**keywords,
                            "prev_topk_indices": indices.reshape(1, 1, -1)}
            if positional:
                return (hidden, *positional[1:]), keywords
            return positional, {**keywords, "hidden_states": hidden}

        def capture(module, positional, output, index=number):
            value = output if isinstance(output, torch.Tensor) else output[0]
            state["outputs"][index] = value.detach().clone()

        hooks.append(layer.register_forward_pre_hook(
            inject, with_kwargs=True,
        ))
        hooks.append(layer.register_forward_hook(capture))

    checks, history = [], []
    initial = native["initial_position"]
    trace_directory = args.output.with_suffix("")
    trace_directory.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        for offset, token in enumerate(native["teacher_tokens"]):
            position = initial + offset
            path = args.native_run / "trace" / f"step-{position:05d}.pt"
            actual = torch.load(path, map_location="cpu", weights_only=True)
            if actual["token"] != token:
                raise ValueError("Native trace input differs from its report")
            artifacts[str(path)] = digest(path)
            engine.reset()
            prefixes = frozen_prefix(base, history)
            if prefixes is not None:
                engine.inject(prefixes)
            if engine.position != position or actual["position"] != position:
                raise ValueError("HF and native input positions differ")
            state.update(actual=actual, outputs={})
            expected = local_trace(engine, actual, state["outputs"])
            if (engine.position != position + 1
                    or len(engine.cache.layers) != 78
                    or any(layer.get_seq_length() != position + 1
                           for layer in engine.cache.layers)):
                raise ValueError("HF did not commit one cache row per layer")
            compare_trace(f"same_input.step{position}", actual, expected,
                          checks)
            expected_path = trace_directory / f"step-{position:05d}.pt"
            torch.save(expected, expected_path)
            artifacts[str(expected_path)] = digest(expected_path)
            history.append(actual)
            print(json.dumps({"position": position,
                              "failed": sum(not x["passed"] for x in checks)}),
                  flush=True)
    for hook in hooks:
        hook.remove()

    source = Path(__file__)
    model = source.parents[2] / "tilert/models/glm_5_2_ascend"
    sources = [source, source.with_name("run_native_decode.py"),
               model / "reference.py", model / "reference_weights.py",
               model / "ops/indexer.py"]
    installed_model = Path(inspect.getsourcefile(type(engine.model)))
    failed = sum(not item["passed"] for item in checks)
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "HF layers at native hidden/KV/shared-index inputs; "
            "independent rollout separate"
        ),
        "weight_arithmetic": engine.weights.contract,
        "native_run": str(args.native_run), "artifact_sha256": artifacts,
        "source_sha256": {str(path): digest(path) for path in sources},
        "installed_hf_source_sha256": {
            str(installed_model): digest(installed_model),
        },
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
        "coverage": {"raw_decoder_outputs": 77,
                     "final_normalized_outputs": 1,
                     "layer_logits": 78, "cache_layers": 78},
        "timed": False, "profiled": False, "checks": checks,
        "checks_total": len(checks), "checks_failed": failed,
        "status": "pass" if not failed else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "status", "checks_total", "checks_failed",
    )}))
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
