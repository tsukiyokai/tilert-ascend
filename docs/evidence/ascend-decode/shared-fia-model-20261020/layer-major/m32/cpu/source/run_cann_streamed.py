"""Run CANN layers 0 or 0 through 4 with one resident decoder layer.

Use torchrun with sixteen ranks. Compare the captured 8K continuation with
an existing full CANN trace; this is a residency/replay check, not HF parity
or a performance measurement. No processed NZ weight is copied to CPU.
"""

import argparse
from contextlib import contextmanager
import gc
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch
import weakref

from run_cann_model import check_prefix, digest, load_capture
from run_native_decode import compare_tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CAPACITY = 8320
QUOTA_BYTES = 2 * 1024 ** 3


def gather(value):
    import torch.distributed as distributed
    from vllm.distributed import get_world_group

    values = [None] * 16
    distributed.all_gather_object(
        values, value, group=get_world_group().cpu_group,
    )
    return values


def checked(label, operation):
    value, error = None, None
    try:
        value = operation()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    errors = gather(error)
    if any(errors):
        raise RuntimeError(f"{label}: {errors}")
    return value


def baseline(args, rank, metadata, inputs):
    import torch

    report_path = args.baseline / f"rank-{rank}.json"
    report = json.loads(report_path.read_text())
    name = f"step-{inputs['initial_position']:05d}.pt"
    path = args.baseline / f"rank-{rank}" / name
    if (report["rank"] != rank or report["checkpoint_sha256"] != metadata
            or report["inputs"]["prefix_sha256"] != inputs["prefix_sha256"]
            or report["trace_sha256"][name] != digest(path)
            or "float32" not in (report.get("tp_reduction"),
                                 report.get("model_tp_reduction"))):
        raise ValueError("Baseline must bind prefix, rank and FP32 TP sum")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["position"] != 8192
            or saved["token"] != inputs["first_token_id"]):
        raise ValueError("Baseline continuation input differs")
    layers = [{
        "input": saved["hidden_states"][number],
        "merged": saved["hidden_states"][number + 1],
        "layer_logits": saved["layer_logits"][number],
        "cache": saved["cache_rows"][number],
        "indices": saved["indices"][number],
    } for number in range(args.stop_layer + 1)]
    hashes = {str(path): digest(path), str(report_path): digest(report_path)}
    return layers, hashes


def load_scope(engine, checkpoint, prefixes):
    import torch
    from vllm.model_executor.models import utils
    from vllm_ascend.ops.layernorm import AscendRMSNorm

    for model in (engine.model, engine.model.model):
        utils._model_to_pp_missing_layer_names.pop(id(model), None)
    expected = {name for name, _ in engine.model.named_parameters()
                if name.startswith(prefixes)}
    names = sorted(name for name in checkpoint.weight_map
                   if name.startswith(prefixes))
    loaded = engine.model.load_weights(
        (name, checkpoint.tensor(name)) for name in names
    )
    synthesized = set()
    for name in expected - loaded:
        parent, _, field = name.rpartition(".")
        module = engine.model.get_submodule(parent)
        if (isinstance(module, AscendRMSNorm) and field == "bias"
                and name not in checkpoint.weight_map
                and not module.bias_loaded
                and torch.count_nonzero(module.bias).item() == 0):
            synthesized.add(name)
    missing = expected - loaded - synthesized
    if not expected or missing:
        raise ValueError(f"Unloaded parameters: {sorted(missing)}")
    return {"raw_names": names, "loaded": sorted(loaded),
            "synthesized_zero_norm_bias": sorted(synthesized),
            "expected": sorted(expected)}


def make_skeleton(engine):
    import torch
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.model_executor.models import deepseek_v2, utils

    factories = []

    def missing_layers(count, layer_fn, prefix):
        if count != 78 or prefix != "model.layers":
            raise ValueError("Expected the 78-layer main-model factory")
        factories.append(layer_fn)
        return 0, count, torch.nn.ModuleList(
            utils.PPMissingLayer() for _ in range(count)
        )

    with torch.device(engine.device), patch.object(
        deepseek_v2, "make_layers", missing_layers,
    ):
        engine.model = initialize_model(vllm_config=engine.config).eval()
    if len(factories) != 1:
        raise ValueError("Expected exactly one decoder factory")
    engine.runner.model = engine.model
    return factories[0]


def bind_layer_cache(engine, number):
    from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
    from vllm_ascend.core.kv_cache_interface import (
        AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec,
    )
    from vllm_ascend.patch.worker.patch_bind_kv_cache import (
        bind_kv_cache_to_layers,
    )

    caches, engine.builders = {}, []
    kinds = set()
    for name, spec in engine.runner.get_kv_cache_spec().items():
        if not name.startswith(f"model.layers.{number}."):
            raise ValueError(f"Unexpected resident attention: {name}")
        if isinstance(spec, AscendMLAAttentionSpec):
            values, builder = engine.pages[:2], AscendSFAMetadataBuilder
        elif isinstance(spec, AscendSFAIndexerCacheSpec):
            values, builder = engine.pages[2:], AscendSFAIndexerMetadataBuilder
        else:
            raise ValueError(f"Unexpected cache specification: {spec}")
        caches[name] = values
        kinds.add(builder)
        engine.builders.append((
            [name], builder(spec, [name], engine.config, engine.device),
        ))
    expected = {AscendSFAMetadataBuilder}
    if number < 3:
        expected.add(AscendSFAIndexerMetadataBuilder)
    if len(caches) != len(expected) or kinds != expected:
        raise ValueError(f"Layer {number}: cache bindings differ")
    bind_kv_cache_to_layers(
        caches, engine.config.compilation_config.static_forward_context,
    )


@contextmanager
def selections(layer):
    impl = layer.self_attn.mla_attn.mla_attn.impl
    name = "_execute_sparse_flash_attention_process"
    original = getattr(impl, name)
    signature, values = inspect.signature(original), []

    def capture(*args, **kwargs):
        indices = signature.bind(*args, **kwargs).arguments["topk_indices"]
        values.append(indices.detach().clone())
        return original(*args, **kwargs)

    with patch.object(impl, name, capture):
        yield values


def bound_rope_tables(layer):
    import torch
    from vllm_ascend.ops import rotary_embedding as rope

    owners = [module for module in layer.modules()
              if isinstance(module, rope.AscendRotaryEmbedding)]
    if len(owners) != 1:
        raise ValueError("Expected one shared default RoPE object")
    owner, tables = owners[0], {}
    for name, alias in (
        ("cos_sin_cache", "_cos_sin_cache"),
        ("_rope_derived_cos_cache", "_cos_cache"),
        ("_rope_derived_sin_cache", "_sin_cache"),
    ):
        original = getattr(owner, name)
        if (getattr(rope, alias) is not original
                or original.shape[0] < CAPACITY
                or original.shape[1:] != (64,)
                or original.dtype != torch.bfloat16):
            raise ValueError(f"Unexpected shared RoPE table: {name}")
        bounded = (original if original.shape[0] == CAPACITY
                   else original[:CAPACITY].clone())
        if not torch.equal(bounded, original[:CAPACITY]):
            raise ValueError(f"RoPE prefix changed: {name}")
        tables[name] = {"original_rows": original.shape[0],
                        "retained_rows": bounded.shape[0],
                        "prefix_exact": True}
        setattr(owner, name, bounded)
        setattr(rope, alias, bounded)
    return {"max_position_embeddings": owner.max_position_embeddings,
            "base": owner.base, "tables": tables}


def run_layer(
    engine, factory, checkpoint, prefix, hidden, residual, checks, label,
    number,
):
    import torch
    from vllm.model_executor.model_loader.utils import (
        process_weights_after_loading,
    )
    from vllm.model_executor.models.utils import PPMissingLayer
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor

    compilation = engine.config.compilation_config
    old_keys = set(compilation.static_forward_context)
    old_moe = list(compilation.static_all_moe_layers)
    old_eplb = list(VllmEplbAdaptor._registered_moe_layers)
    try:
        def prepare():
            with torch.device(engine.device):
                layer = factory(prefix=f"model.layers.{number}").eval()
            engine.model.model.layers[number] = layer
            rope_tables = bound_rope_tables(layer)
            if any(table["original_rows"] > CAPACITY
                   for table in rope_tables["tables"].values()):
                torch.npu.empty_cache()
            coverage = load_scope(
                engine, checkpoint, (f"model.layers.{number}.",),
            )
            coverage["rope_tables"] = rope_tables
            process_weights_after_loading(
                layer, engine.config.model_config, engine.device,
            )
            bind_layer_cache(engine, number)
            for name, value in prefix.items():
                if value is not None:
                    engine.cache[0][name][:8192].copy_(value)
            return layer, coverage

        layer, coverage = checked(label + ".prepare", prepare)
        references = {"layer": weakref.ref(layer)}
        references.update({
            "parameter." + name: weakref.ref(value)
            for name, value in layer.named_parameters()
        })
        if number >= 3:
            references["experts"] = weakref.ref(
                layer.mlp.experts.routed_experts,
            )
        check_prefix(engine, [prefix], checks, label + ".before")
        metadata = checked(label + ".metadata", engine._metadata)
        raw_hidden = hidden.detach().clone()
        raw_residual = (
            residual.detach().clone() if residual is not None else None
        )
        topk = layer.self_attn.topk_indices_buffer[:1]
        topk_before = topk.detach().clone() if number >= 3 else None
        inputs = []

        def capture_input(module, arguments, output):
            merged = arguments[0] if len(arguments) == 1 else output[1]
            inputs.append(merged.detach().clone())

        reductions, original = [], engine._tp_sum_original

        def reduce(value):
            reductions.append({"dtype": str(value.dtype),
                               "shape": list(value.shape)})
            return original(value)

        with (
            layer.input_layernorm.register_forward_hook(capture_input),
            set_ascend_forward_context(
                metadata, engine.config, num_tokens=1, num_actual_tokens=1,
                model_instance=engine.model,
            ),
            selections(layer) as indices,
            patch.object(engine, "_tp_sum_original", reduce),
        ):
            hidden, residual = layer(engine.positions, hidden, residual)
        merged = torch.ops._C_ascend.npu_add_rms_norm_bias(
            hidden, residual, engine.norm_units, None, 1e-5,
        )[2]
        logits = engine.model.compute_logits(engine.model.model.norm(merged))
        torch.npu.synchronize(engine.device)
        if (len(inputs) != 1 or len(indices) != 1
                or indices[0].numel() != 2048):
            raise ValueError("Expected one actual SFA selection of 2048 keys")
        choices = indices[0].reshape(-1).cpu().clone()
        topk_after = topk.cpu().clone()
        checks.extend([
            {"name": label + ".indices_valid", "passed": bool(
                choices.dtype in (torch.int32, torch.int64)
                and choices.unique().numel() == 2048
                and torch.all((choices >= 0) & (choices <= 8192))
            )},
            {"name": label + ".decoder_reductions", "passed": reductions
             == [{"dtype": "torch.float32", "shape": [1, 6144]}] * 2},
            {"name": label + ".topk_buffer", "passed": torch.equal(
                topk_after.reshape(-1), choices,
            )},
        ])
        if topk_before is not None:
            topk_before = topk_before.cpu()
            checks.append({"name": label + ".shared_topk_unchanged",
                           "passed": torch.equal(topk_before, topk_after)})
        check_prefix(engine, [prefix], checks, label + ".after")
        sources = engine.source_metadata()
        for value in (layer.self_attn.__init__, selections, type(checkpoint),
                      type(layer.self_attn.rotary_emb),
                      type(engine.model.model.norm),
                      type(engine.model.logits_processor),
                      load_capture, compare_tensor, VllmEplbAdaptor,
                      process_weights_after_loading):
            path = Path(inspect.getfile(inspect.unwrap(value))).resolve()
            sources[str(path)] = digest(path)
        actual = {"layer": number, "input": inputs[0].cpu(),
                  "raw_input_hidden": raw_hidden.cpu(),
                  "raw_input_residual": (
                      raw_residual.cpu() if raw_residual is not None else None
                  ), "merged": merged.cpu(), "layer_logits": logits.cpu(),
                  "hidden": hidden.cpu(), "residual": residual.cpu(),
                  "topk_before": topk_before, "topk_after": topk_after,
                  "indices": choices, "cache": {
                      "latent": engine.cache[0]["latent"][8192].cpu(),
                      "rope": engine.cache[0]["rope"][8192].cpu(),
                      "index": (engine.cache[0]["index_keys"][8192].cpu()
                                if number < 3 else None),
                      "index_scale": None,
                  }}
        return (actual, hidden, residual, coverage, reductions, sources,
                references)
    finally:
        torch.npu.synchronize(engine.device)
        engine.model.model.layers[number] = PPMissingLayer()
        engine.builders = []
        for name in set(compilation.static_forward_context) - old_keys:
            del compilation.static_forward_context[name]
        compilation.static_all_moe_layers[:] = old_moe
        VllmEplbAdaptor._registered_moe_layers[:] = old_eplb


def compare(actual, expected, checks, label, *, exact):
    compare_tensor(label + ".input", actual["input"], expected["input"],
                   checks, exact=exact or actual["layer"] == 0)
    compare_tensor(label + ".merged", actual["merged"], expected["merged"],
                   checks, exact=exact)
    compare_tensor(label + ".layer_logits", actual["layer_logits"],
                   expected["layer_logits"], checks, exact=exact, logits=True)
    for name in actual["cache"]:
        compare_tensor(label + ".cache." + name, actual["cache"][name],
                       expected["cache"][name], checks, exact=True)
    left, right = actual["indices"], expected["indices"]
    if not exact:
        left, right = left.long().sort().values, right.long().sort().values
    compare_tensor(label + ".indices", left, right, checks, exact=True)
    if exact:
        for name in ("hidden", "residual", "raw_input_hidden",
                     "raw_input_residual", "topk_before", "topk_after"):
            compare_tensor(label + "." + name, actual[name], expected[name],
                           checks, exact=True)


def execute(args):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import (
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm_ascend.worker.worker import NPUWorker
    from tilert.models.glm_5_2_ascend.cann_model import (
        CannModel, create_config,
    )
    from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16 or not 0 <= rank < 16:
        raise ValueError("Launch one node with sixteen ranks")
    metadata = checkpoint_fingerprint(args.checkpoint)
    prefixes, teacher, inputs = load_capture(args.capture_case, rank, metadata)
    if inputs["initial_position"] != 8192:
        raise ValueError("This check requires an 8192-token captured prefix")
    if len(prefixes) != 78:
        raise ValueError("Capture must contain all 78 main layers")
    prefixes = [{name: value for name, value in entry.items()
                 if value is not None}
                for entry in prefixes[:args.stop_layer + 1]]
    for number, prefix in enumerate(prefixes):
        shapes = {"latent": (8192, 512), "rope": (8192, 64)}
        if number < 3:
            shapes["index_keys"] = (8192, 128)
        if (prefix.keys() != shapes.keys() or any(
            value.shape != shapes[name] or value.dtype != torch.bfloat16
            for name, value in prefix.items()
        )):
            raise ValueError(f"Layer {number}: invalid full-P BF16 cache")
    expected, artifacts = baseline(args, rank, metadata, inputs)
    engine = CannModel.__new__(CannModel)
    engine.rank, engine.device = rank, torch.device(f"npu:{rank}")
    engine.capacity, engine.position = CAPACITY, 8192
    engine.attention_kernel = "sfa"
    engine.fp32_head = False
    engine.checkpoint = str(args.checkpoint.resolve())
    engine._tp_sum_group = None
    torch.npu.set_device(engine.device)
    total = torch.npu.get_device_properties(engine.device).total_memory
    torch.npu.set_per_process_memory_fraction(
        QUOTA_BYTES / total, engine.device,
    )
    engine.config = create_config(args.checkpoint, CAPACITY, 0.02)
    checks, phases = [], {}
    with (set_current_vllm_config(engine.config),
          set_default_torch_dtype(torch.bfloat16), torch.inference_mode()):
        engine.worker = NPUWorker(
            engine.config, local_rank=rank, rank=rank,
            distributed_init_method="env://", is_driver_worker=rank == 0,
        )
        engine.worker.init_device()
        try:
            if engine.worker.device != engine.device:
                raise ValueError("Worker remapped the requested physical NPU")
            engine.runner = engine.worker.model_runner
            engine._install_tp_sum()
            factory = checked("skeleton", lambda: make_skeleton(engine))
            checkpoint = ModelSlimCheckpoint(args.checkpoint)
            coverage = checked("base_weights", lambda: load_scope(
                engine, checkpoint,
                ("model.embed_tokens.", "model.norm.", "lm_head."),
            ))
            checked("base_postprocess", lambda: process_weights_after_loading(
                engine.model, engine.config.model_config, engine.device,
            ))
            engine.layout = {"capacity_tokens": CAPACITY}
            engine.pages = tuple(torch.empty(
                (CAPACITY // 128, 128, 1, width), dtype=torch.bfloat16,
                device=engine.device,
            ) for width in (512, 64, 128))
            engine.cache = [dict(zip(
                ("latent", "rope", "index_keys"),
                (value.view(-1, value.shape[-1]) for value in engine.pages),
            ))]
            engine.norm_units = torch.ones(
                6144, dtype=torch.bfloat16, device=engine.device,
            )
            engine._allocate_inputs()
            sources, first = {}, []
            for phase in ("first", "reload"):
                args.progress = {"phase": phase, "layer": 0}
                engine.input_ids.fill_(teacher[0])
                hidden = engine.model.embed_input_ids(engine.input_ids)
                residual = None
                previous = None
                phases[phase] = []
                for number, prefix in enumerate(prefixes):
                    args.progress = {"phase": phase, "layer": number}
                    label = f"{phase}.layer{number}"
                    torch.npu.reset_peak_memory_stats(engine.device)
                    before = torch.npu.memory_allocated(engine.device)
                    (actual, hidden, residual, loaded, reductions, source,
                     references) = run_layer(
                        engine, factory, checkpoint, prefix, hidden, residual,
                        checks, label, number,
                    )
                    gc.collect()
                    alive = [name for name, ref in references.items()
                             if ref() is not None]
                    checks.append({"name": label + ".owners_released",
                                   "passed": not alive, "alive": alive,
                                   "checked_owners": sorted(references)})
                    torch.npu.empty_cache()
                    phases[phase].append({
                        "layer": number, "load_coverage": loaded,
                        "reductions": reductions, "allocated_before": before,
                        "allocated_after_release": torch.npu.memory_allocated(
                            engine.device),
                        "peak_allocated": torch.npu.max_memory_allocated(
                            engine.device),
                    })
                    sources.update(source)
                    args.output.mkdir(parents=True, exist_ok=True)
                    filename = f"rank-{rank}-{phase}-layer{number}.pt"
                    path = args.output / filename
                    actual.update(position=8192, token=teacher[0])
                    checked(label + ".save", lambda: torch.save(actual, path))
                    artifacts[str(path)] = digest(path)
                    compare(actual, expected[number] if phase == "first"
                            else first[number], checks, label,
                            exact=phase == "reload")
                    if previous is not None:
                        for name in ("hidden", "residual"):
                            compare_tensor(
                                label + ".carried_" + name,
                                actual["raw_input_" + name], previous[name],
                                checks, exact=True,
                            )
                    previous = actual
                    if phase == "first":
                        first.append(actual)

                    def require_release():
                        if alive:
                            raise RuntimeError(
                                f"Resident layer owners: {alive}",
                            )

                    checked(label + ".release", require_release)
                del hidden, residual
            checks.append({
                "name": "phase_end.released_allocations_stable",
                "passed": phases["first"][-1]["allocated_after_release"]
                == phases["reload"][-1]["allocated_after_release"],
            })
            sources[str(Path(__file__).resolve())] = digest(Path(__file__))
            passed = all(item["passed"] for item in checks)
            report = {
                "status": "pass" if passed else "fail",
                "rank": rank, "inputs": inputs, "checkpoint_sha256": metadata,
                "precision_verified": False, "performance_measured": False,
                "scope": "Streamed layers vs full CANN and exact reload",
                "stop_layer": args.stop_layer,
                "layer_logits_projection_rows": 1,
                "baseline_layer_logits_projection_rows": 77,
                "torch_quota_bytes": QUOTA_BYTES,
                "gpu_memory_utilization": 0.02,
                "base_load_coverage": coverage, "phases": phases,
                "source_sha256": sources, "artifact_sha256": artifacts,
                "checks": checks, "command": sys.argv,
                "environment": {name: os.environ.get(name) for name in (
                    "ASCEND_CUSTOM_OPP_PATH", "HCCL_ALGO",
                    "HCCL_OP_EXPANSION_MODE", "PYTORCH_NPU_ALLOC_CONF",
                    "ASCEND_LAUNCH_BLOCKING")},
                "packages": {name: version(name) for name in (
                    "torch", "torch-npu", "vllm", "vllm-ascend")},
            }
            path = args.output / f"rank-{rank}.json"
            checked("report", lambda: path.write_text(
                json.dumps(report, indent=2) + "\n",
            ))
            ranks = gather({"rank": rank, "status": report["status"],
                            "report_sha256": digest(path)})
            passed = all(item["status"] == "pass" for item in ranks)
            if rank == 0:
                (args.output / "summary.json").write_text(json.dumps({
                    "status": "pass" if passed else "fail", "ranks": ranks,
                    "precision_verified": False, "performance_measured": False,
                }, indent=2) + "\n")
            return 0 if passed else 1
        finally:
            engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "capture-case", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stop-layer", type=int, choices=(0, 4), default=0)
    args = parser.parse_args()
    args.progress = {"phase": "setup", "layer": None}
    try:
        return execute(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        error = {**args.progress, "rank": os.environ.get("RANK"),
                 "status": "error", "traceback": traceback.format_exc()}
        (args.output / f"rank-{os.environ.get('RANK')}-error.json").write_text(
            json.dumps(error, indent=2) + "\n",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
