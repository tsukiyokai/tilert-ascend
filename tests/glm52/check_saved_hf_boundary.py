"""Replay one frozen HF decoder layer on CPU and save its stage inputs.

This checks local HF replay against an existing cold trace. It neither runs
CANN nor establishes NPU precision or end-to-end model acceptance.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sys
import time

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402

from run_native_decode import (  # noqa: E402
    compare_tensor, digest, validate_reference_manifest,
)
from tilert.models.glm_5_2_ascend.reference import HfReference  # noqa: E402


def npu_initialized():
    module = sys.modules.get("torch_npu")
    return bool(module is not None and module.npu._initialized)


def load_inputs(directory, checkpoint, position):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = {name: digest(checkpoint / name) for name in (
        "config.json", "quant_model_description.json",
        "quant_model_weights.safetensors.index.json",
    )}
    validate_reference_manifest(manifest, metadata)
    if manifest["initial_position"] != 0 or manifest["prefix_files_sha256"]:
        raise ValueError("This diagnostic requires the saved cold HF trace")
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for name in ("reference.py", "reference_weights.py", "ops/indexer.py"):
        relative = "tilert/models/glm_5_2_ascend/" + name
        sources[relative] = digest(root / relative)
    artifacts = {str(manifest_path): digest(manifest_path)}
    traces = []
    for index in range(position + 1):
        path = directory / f"step-{index:05d}.pt"
        artifacts[str(path)] = digest(path)
        if artifacts[str(path)] != manifest["trace_sha256"][path.name]:
            raise ValueError("Frozen trace hash differs: " + path.name)
        trace = torch.load(path, map_location="cpu", weights_only=True)
        if (trace["position"] != index
                or trace["committed_position"] != index + 1
                or trace["token"] != manifest["teacher_tokens"][index]
                or len(trace["hidden_states"]) != 79
                or len(trace["cache_rows"]) != 78):
            raise ValueError("Frozen trace position or coverage differs")
        traces.append(trace)
    return manifest, traces, artifacts, sources


def inject_layer(engine, layer_number, history, stages):
    if not history:
        return
    attention = engine.model.model.layers[layer_number].self_attn
    rows = [trace["cache_rows"][layer_number] for trace in history]
    latent = torch.stack([row["latent"] for row in rows])
    rope = torch.stack([row["rope"] for row in rows])
    stages["cache.before.latent"] = latent.clone()
    stages["cache.before.rope"] = rope.clone()
    expanded = attention.kv_b_proj(latent).reshape(-1, 64, 448)
    key, value = expanded.split((192, 256), dim=-1)
    rope_heads = rope[:, None, :].expand(-1, 64, 64)
    key = torch.cat((key, rope_heads), -1).transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    engine.cache.update(key.contiguous(), value.contiguous(), layer_number)
    if attention.indexer is not None:
        keys = torch.stack([row["index"] for row in rows])
        scales = (torch.cat([row["index_scale"] for row in rows])
                  if attention.indexer.quantized else None)
        attention.indexer.inject(engine.cache, keys, scales)
        stages["cache.before.index"] = keys.clone()
        if scales is not None:
            stages["cache.before.index_scale"] = scales.clone()


@contextmanager
def capture_stages(layer, stages):
    hooks = []

    def recorder(name):
        def capture(module, inputs, output):
            stages[name + ".input"] = inputs[0].detach().clone()
            stages[name + ".output"] = output.detach().clone()
        return capture

    modules = {
        "input_layernorm": layer.input_layernorm,
        "post_attention_layernorm": layer.post_attention_layernorm,
        "mlp": layer.mlp,
        **{name: getattr(layer.self_attn, name) for name in (
            "q_a_proj", "q_a_layernorm", "q_b_proj",
            "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o_proj",
        )},
    }
    for name, module in modules.items():
        hooks.append(module.register_forward_hook(recorder(name)))

    def attention_output(module, inputs, output):
        stages["attention.output"] = output[0].detach().clone()

    hooks.append(layer.self_attn.register_forward_hook(attention_output))
    if hasattr(layer.mlp, "gate"):
        def router_output(module, inputs, output):
            stages["router.input"] = inputs[0].detach().clone()
            for name, value in zip(("logits", "weights", "choices"), output):
                stages["router." + name] = value.detach().clone()

        hooks.append(layer.mlp.gate.register_forward_hook(router_output))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def replay(engine, traces, number):
    saved = traces[-1]
    position = saved["position"]
    layer = engine.model.model.layers[number]
    stages = {}
    inject_layer(engine, number, traces[:-1], stages)
    engine.indices, engine.routes, engine.latents = (
        [None] * 78 for _ in range(3)
    )
    hidden = saved["hidden_states"][number].unsqueeze(1).clone()
    if hidden.shape != (1, 1, 6144) or hidden.dtype != torch.bfloat16:
        raise ValueError("Expected a frozen BF16 hidden row")
    positions = torch.tensor([[position]], dtype=torch.int64)
    cosine, sine = engine.model.model.rotary_emb(
        hidden, position_ids=positions,
    )
    stages.update({"decoder.input": hidden.clone(),
                   "position_ids": positions.clone(),
                   "cos": cosine.clone(), "sin": sine.clone()})
    shared = layer.self_attn.indexer is None
    indices = saved["indices"][number].reshape(1, 1, -1).clone()
    if shared:
        stages["shared_indices.input"] = indices.clone()
    with capture_stages(layer, stages):
        output, selected = layer(
            hidden, attention_mask=None, position_ids=positions,
            past_key_values=engine.cache, use_cache=True,
            position_embeddings=(cosine, sine),
            prev_topk_indices=indices if shared else None,
        )
    stages["decoder.output"] = output.clone()
    stages["indices"] = selected.clone()
    actual = (engine.model.model.norm(output) if number == 77 else output)
    stages["decoder.comparable_output"] = actual.clone()
    checks = []
    compare_tensor("decoder.output", actual[:, -1],
                   saved["hidden_states"][number + 1], checks, exact=True)
    compare_tensor("indices", selected.reshape(-1),
                   saved["indices"][number].reshape(-1), checks, exact=True)
    compare_tensor("routes", engine.routes[number], saved["routes"][number],
                   checks, exact=True)
    indexer = layer.self_attn.indexer
    cache = engine.cache.layers[number]
    rows = {
        "latent": engine.latents[number],
        "rope": cache.keys[0, 0, -1, 192:],
        "index": None if indexer is None else indexer.last_key,
        "index_scale": None if indexer is None else indexer.last_scale,
    }
    for name, value in rows.items():
        compare_tensor("cache." + name, value,
                       saved["cache_rows"][number][name], checks, exact=True)
        if value is not None:
            stages["cache.after." + name] = value.detach().clone()
    checks.append({"name": "cache.length",
                   "passed": cache.get_seq_length() == position + 1})
    return stages, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=(0, 1), default=0)
    parser.add_argument("--layer", type=int, choices=range(78), default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    before = npu_initialized()
    started = time.perf_counter()
    manifest, traces, artifacts, sources = load_inputs(
        args.reference, args.checkpoint, args.step,
    )
    engine = HfReference(args.checkpoint)
    installed = Path(inspect.getsourcefile(type(engine.model)))
    installed_hash = digest(installed)
    if installed_hash not in manifest["installed_hf_source_sha256"].values():
        raise ValueError("Installed HF model source differs")
    stages, checks = replay(engine, traces, args.layer)
    helper = Path(inspect.getsourcefile(digest))
    sources[str(helper)] = digest(helper)
    sources[str(Path(__file__).resolve())] = digest(Path(__file__))
    sources[str(installed)] = installed_hash
    metadata = {
        "layer": args.layer, "position": args.step,
        "token": traces[-1]["token"],
        "checkpoint_sha256": engine.weights.contract["metadata_sha256"],
        "artifact_sha256": artifacts, "source_sha256": sources,
        "frozen_source_sha256": manifest["source_sha256"],
    }
    path = args.output / "stages.pt"
    torch.save({"schema_version": 1, "metadata": metadata,
                "stages": stages}, path)
    after = npu_initialized()
    checks.append({"name": "no_npu_initialization",
                   "passed": not before and not after})
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "One frozen HF layer replay on CPU; no CANN comparison",
        "precision_verified": False, "performance_measured": False,
        "layer": args.layer, "position": args.step,
        "token": traces[-1]["token"], "cpu_threads": 4,
        "cpu_wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": engine.weights.contract["metadata_sha256"],
        "artifact_sha256": {**artifacts, str(path): digest(path)},
        "frozen_source_sha256": manifest["source_sha256"],
        "source_sha256": sources, "torch": str(torch.__version__),
        "stages": {name: {"shape": list(value.shape),
                          "dtype": str(value.dtype)}
                   for name, value in stages.items()},
        "npu_initialized_before": before, "npu_initialized_after": after,
        "checks": checks, "passed": all(item["passed"] for item in checks),
    }
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({key: result[key] for key in
                      ("passed", "layer", "position", "cpu_wall_seconds")}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
