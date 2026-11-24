"""Observe the full indexer inside unchanged asynchronous decoder calls."""

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

from decoder_forward import run_decoder_forward as original_forward
from vendor_hook_diagnostic import compare, cpu_copy, replace_attribute, tensor_identity
from vendor_hook_forward import source_identity
from vendor_hook_index_diagnostic import IndexerTaps, normalize_shapes, rope_reference


@contextmanager
def observe_full_indexer(taps, adapter):
    """Execute shared attention unchanged, without overwriting the full-indexer tap."""
    original = adapter.run_body
    with taps.installed():
        indexed = adapter.run_body

        def dispatch(*args, **kwargs):
            function = original if kwargs["mode"] == "shared" else indexed
            return function(*args, **kwargs)

        with replace_attribute(adapter, "run_body", dispatch):
            yield


def reproduction(torch, parent, current):
    checks = {"hidden": compare(torch, current["hidden"], parent["hidden"])}
    if (parent["residual"] is None) != (current["residual"] is None):
        return {"pass": False, "reason": "Initial residual branch changed"}
    if parent["residual"] is not None:
        checks["residual"] = compare(torch, current["residual"], parent["residual"])
    if len(parent["post_ar1"]) != len(current["post_ar1"]):
        return {"pass": False, "reason": "The TP boundary input count changed"}
    for index, (left, right) in enumerate(zip(current["post_ar1"], parent["post_ar1"])):
        checks[f"post_ar1/{index}"] = compare(torch, left, right)
    for collection in ("metadata", "original", "native"):
        if parent[collection].keys() != current[collection].keys():
            return {"pass": False, "reason": f"The {collection} tensor set changed"}
        for name, expected in parent[collection].items():
            checks[f"{collection}/{name}"] = compare(torch, current[collection][name], expected)
    return {"pass": all(value["bitwise_equal"] for value in checks.values()), "checks": checks}


def save_diagnostic(torch, taps, args, outer_error):
    torch.npu.synchronize()
    vendor = {name: cpu_copy(torch, value) for name, value in taps.vendor.items()}
    shapes = {name: list(value.shape) for name, value in vendor.items()}
    native, offsets, blocks = taps.read_native()
    rows = native["hidden"].shape[0]
    normalize_shapes(vendor, rows)
    normalize_shapes(native, rows)
    vendor["weights"] = vendor["key_projection"][:, 128:].to(
        torch.float16 if args.mode == "int8" else torch.bfloat16).contiguous()
    order = ["hidden", "query_norm", "query_norm_scale", "key_projection",
             "key_normalized", "key_rope", "query_projection", "query_rope"]
    if args.mode == "int8":
        order += ["key_hadamard", "key", "key_scale", "query_hadamard", "query", "query_scale"]
    else:
        order += ["key", "query"]
    order.append("weights")
    checks = {name: compare(torch, native[name], vendor[name]) for name in order}
    for name in ("key_weight", "query_weight", "query_weight_scale", "gamma",
                 "beta", "cosine", "sine", "index_slots"):
        checks[name] = compare(torch, native[name], vendor[name])
    oracles = {}
    oracle_checks = {}
    for arm, values in (("vendor", vendor), ("native", native)):
        for name, source in (("key", "key_normalized"), ("query", "query_projection")):
            output = rope_reference(values[source], values["cosine"], values["sine"])
            label = arm + "_" + name + "_rope_cpu_fp32"
            oracles[label] = output
            oracle_checks[label] = compare(torch, output, values[name + "_rope"])
    dot = vendor["key_hidden"].double() @ vendor["key_weight"].double().t()
    oracles["key_projection_cpu_fp64"] = dot
    for arm, values in (("vendor", vendor), ("native", native)):
        oracle_checks[arm + "_key_projection_cpu_fp64"] = compare(
            torch, dot.bfloat16(), values["key_projection"])
    outer = json.loads((args.output / "decoder-forward.json").read_text())
    case = outer["cases"][-1]
    parent = torch.load(args.index_diagnostic_parent, map_location="cpu", weights_only=True)
    if "failure_dump" in case:
        current = torch.load(case["failure_dump"], map_location="cpu", weights_only=True)
        reproduced = reproduction(torch, parent, current)
    else:
        current = None
        reproduced = {"pass": False, "reason": "The original failure did not recur"}
    reproduced.update(parent_path=str(args.index_diagnostic_parent),
                      parent_sha256=hashlib.sha256(args.index_diagnostic_parent.read_bytes()).hexdigest())
    if current is not None:
        for arm, values, key in (("vendor", vendor, "original"), ("native", native, "native")):
            values["key_cache"] = current[key]["layer6.index_cache"]
            if args.mode == "int8":
                values["scale_cache"] = current[key]["layer6.index_scale"]
            values["topk"] = current[key]["layer6.topk"]
            values["block_table"] = current["metadata"]["index_table"]
            values["key_lengths"] = current["metadata"]["lengths"]
    dump = args.output / "decoder-index-diagnostic.pt"
    torch.save({"vendor": vendor, "native": native, "oracles": oracles,
                "offsets": offsets, "case": case}, dump)
    report = {
        "scope": "Layer6 full-indexer taps inside both unchanged complete DecoderLayer 6/7 calls",
        "script": source_identity(save_diagnostic), "sources_before_taps": taps.sources,
        "perturbation": "Two vendor Q/K device clones preserve pre-RoPE values; other taps retain references. No GMM hooks, added internal synchronization, or timing claim.",
        "shared_layer_execution": "Original layer7 execution and native shared body both retained; only instrumentation is filtered",
        "case": {"rows": case["rows"], "iteration": case["iteration"],
                 "original_decoder_gate_pass": case["pass"]},
        "outer_error": outer_error, "calls": taps.calls, "blocks": blocks,
        "production_boundary_order": order, "checks": checks,
        "first_unequal_boundary_in_recorded_order": next(
            (name for name in order if not checks[name]["bitwise_equal"]), None),
        "original_failure_reproduction": reproduced,
        "auxiliary_cpu_oracle_checks": oracle_checks,
        "cpu_oracle_scope": "FP64 WK dot and FP32 RoPE are auxiliary arithmetic, not the device reduction order",
        "vendor_original_shapes": shapes, "vendor_descriptors": taps.descriptors,
        "vendor_tensors": {name: tensor_identity(torch, value) for name, value in vendor.items()},
        "native_tensors": {name: tensor_identity(torch, value) for name, value in native.items()},
        "dump": {"path": str(dump), "sha256": hashlib.sha256(dump.read_bytes()).hexdigest()},
    }
    (args.output / "decoder-index-diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decoder_index_diagnostic": report}), flush=True)
    return report


def run_decoder_forward(torch, layers, config, state, args):
    import torch_npu
    if args.forward_widths != [12] or args.forward_replays != 1 or args.mode != "int8":
        raise ValueError("This diagnostic reproduces only the frozen C8 M12/r0 case")
    os.environ["TILERT_INDEXED_ATTENTION_LIBRARY"] = str(args.indexed_library)
    os.environ["TILERT_SHARED_ATTENTION_LIBRARY"] = str(args.shared_library)
    sys.path.insert(0, str(args.adapter.resolve().parents[3]))
    adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
    if Path(adapter.__file__).resolve() != args.adapter.resolve():
        raise RuntimeError("The tap imported a different native adapter")
    impl = layers[0].self_attn.mla_attn.mla_attn.impl
    taps = IndexerTaps(torch, torch_npu, impl, adapter)
    try:
        with observe_full_indexer(taps, adapter):
            result = original_forward(torch, layers, config, state, args)
    except Exception:
        outer_error = traceback.format_exc()
        try:
            save_diagnostic(torch, taps, args, outer_error)
        except Exception:
            error = {"outer_error": outer_error, "diagnostic_error": traceback.format_exc()}
            (args.output / "decoder-index-diagnostic-error.json").write_text(
                json.dumps(error, indent=2) + "\n")
            print(json.dumps(error), flush=True)
        raise
    report = save_diagnostic(torch, taps, args, None)
    if not report["original_failure_reproduction"]["pass"]:
        raise RuntimeError("The diagnostic did not reproduce the frozen full-decoder failure")
    return result
