"""Observe the indexer before a frozen decoder graph failure releases its owners."""

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import decoder_graph
from vendor_hook_diagnostic import compare, cpu_copy, replace_attribute, tensor_identity
from vendor_hook_forward import source_identity
from vendor_hook_index_diagnostic import IndexerTaps, normalize_shapes, rope_reference


def reproduction(torch, parent, current, parent_case, current_case):
    checks = {}
    for collection in ("plan", "expected", "actual"):
        if parent[collection].keys() != current[collection].keys():
            return {"pass": False, "reason": f"The {collection} tensor set changed"}
        for name, expected in parent[collection].items():
            checks[f"{collection}/{name}"] = compare(torch, current[collection][name], expected)
    fields = ("rows", "initial_residual", "iteration", "phase", "inputs")
    metadata = {name: parent_case[name] == current_case[name] for name in fields}
    return {"pass": all(value["bitwise_equal"] for value in checks.values())
                    and all(metadata.values()),
            "checks": checks, "case_and_live_producer_witnesses": metadata}


@contextmanager
def observe_full_indexer(taps, adapter):
    original = adapter.run_body
    with taps.installed():
        indexed = adapter.run_body

        def dispatch(*args, **kwargs):
            function = original if kwargs["mode"] == "shared" else indexed
            return function(*args, **kwargs)

        with replace_attribute(adapter, "run_body", dispatch):
            yield


@contextmanager
def observe_failure_saves(torch, callback):
    original = torch.save

    def save(value, destination, *args, **kwargs):
        result = original(value, destination, *args, **kwargs)
        if (isinstance(value, dict) and set(value) == {"plan", "expected", "actual"}
                and Path(destination).name.endswith("-native.pt")):
            callback(value, destination, original)
        return result

    with replace_attribute(torch, "save", save):
        yield


def expected_numerical_failure(error, report):
    return (isinstance(error, RuntimeError)
            and str(error) == "The complete decoder graph differs from its eager oracle"
            and report is not None and report.get("intermediate_evidence_accepted") is True)


def accepted_evidence(reproduced, input_checks, producer_checks):
    return (reproduced["pass"]
            and all(value["bitwise_equal"] for value in input_checks.values())
            and all(value["bitwise_equal"] for value in producer_checks.values()))


def capture_diagnostic(torch, taps, args, failure, inputs, save):
    """Run only after the original comparison and save, before graph teardown."""
    torch.npu.synchronize()
    report_path = args.output / "decoder-graph.json"
    graph_report = json.loads(report_path.read_text())
    case = graph_report["cases"][-1]
    parent = torch.load(args.graph_index_parent, map_location="cpu", weights_only=True)
    parent_report = json.loads(args.graph_index_parent_report.read_text())
    reproduced = reproduction(torch, parent, failure, parent_report["cases"][-1], case)
    original_shapes = {name: list(value.shape) for name, value in taps.vendor.items()}
    descriptors = {name: taps.descriptor(value) for name, value in taps.vendor.items()}
    vendor = {name: cpu_copy(torch, value) for name, value in taps.vendor.items()}
    native, offsets, blocks = taps.read_native()
    rows = native["hidden"].shape[0]
    normalize_shapes(vendor, rows)
    normalize_shapes(native, rows)
    vendor["weights"] = vendor["key_projection"][:, 128:].half().contiguous()
    order = ["hidden", "query_norm", "query_norm_scale", "key_projection",
             "key_normalized", "key_rope", "key_hadamard", "key", "key_scale",
             "query_projection", "query_rope", "query_hadamard", "query",
             "query_scale", "weights"]
    checks = {name: compare(torch, native[name], vendor[name]) for name in order}
    for name in ("key_weight", "query_weight", "query_weight_scale", "gamma",
                 "beta", "cosine", "sine", "index_slots"):
        checks[name] = compare(torch, native[name], vendor[name])
    producers = {name: cpu_copy(torch, value) for name, value in inputs.fields().items()}
    input_checks = {
        "vendor_hidden_matches_original_input_norm": compare(
            torch, vendor["hidden"], failure["expected"]["layer6.input_norm.0"]),
        "native_hidden_matches_replayed_input_norm": compare(
            torch, native["hidden"], failure["actual"]["layer6.input_norm.0"]),
        "vendor_slots_match_current_plan": compare(torch, vendor["index_slots"],
                                                   failure["plan"]["index_slots"]),
        "native_slots_match_current_plan": compare(torch, native["index_slots"],
                                                   failure["plan"]["index_slots"]),
    }
    oracles, oracle_checks, producer_checks = {}, {}, {}
    for arm, values, result_name in (("vendor", vendor, "expected"),
                                     ("native", native, "actual")):
        values["key_cache"] = failure[result_name]["layer6.index_cache"]
        values["scale_cache"] = failure[result_name]["layer6.index_scale"]
        values["topk"] = failure[result_name]["layer6.topk"]
        slots = failure["plan"]["index_slots"].long()
        stored = values["key_cache"].reshape(-1, 128)[slots]
        producer_checks[arm + "_key_matches_written_cache"] = compare(torch, values["key"], stored)
        stored_scale = values["scale_cache"].reshape(-1)[slots]
        producer_checks[arm + "_key_scale_matches_written_cache"] = compare(
            torch, values["key_scale"], stored_scale)
        for name, source in (("key", "key_normalized"), ("query", "query_projection")):
            label = arm + "_" + name + "_rope_cpu_fp32"
            output = rope_reference(values[source], values["cosine"], values["sine"])
            oracles[label] = output
            oracle_checks[label] = compare(torch, output, values[name + "_rope"])
    dot = vendor["key_hidden"].double() @ vendor["key_weight"].double().t()
    oracles["key_projection_cpu_fp64"] = dot
    for arm, values in (("vendor", vendor), ("native", native)):
        oracle_checks[arm + "_key_projection_cpu_fp64"] = compare(
            torch, dot.bfloat16(), values["key_projection"])
    path = args.output / "decoder-graph-index-diagnostic.pt"
    save({"vendor": vendor, "native": native, "producers": producers,
          "oracles": oracles, "offsets": offsets, "case": case,
          "failure": failure}, path)
    result = {
        "scope": "Original eager layer6 indexer versus retained native layer6 graph producers before graph teardown",
        "script": source_identity(capture_diagnostic), "sources_before_taps": taps.sources,
        "original_graph": source_identity(decoder_graph.run_decoder_graph),
        "perturbation": "Two pre-RoPE clones in the original indexer; other taps retain references. Original graph, decoder calls, FFN disposal, comparison and DSO identities are unchanged.",
        "read_point": "After original native-replay failure comparison and saved fixture; before wrapper reset or pool release",
        "graph_captures": graph_report["captures"],
        "original_failure_reproduction": reproduced,
        "parent_failure_sha256": hashlib.sha256(args.graph_index_parent.read_bytes()).hexdigest(),
        "parent_report_sha256": hashlib.sha256(args.graph_index_parent_report.read_bytes()).hexdigest(),
        "calls": taps.calls, "blocks": blocks, "production_boundary_order": order,
        "checks": checks, "live_input_checks": input_checks,
        "live_producer_checks": producer_checks,
        "first_unequal_boundary_in_recorded_order": next(
            (name for name in order if not checks[name]["bitwise_equal"]), None),
        "intermediate_evidence_accepted": accepted_evidence(reproduced, input_checks, producer_checks),
        "vendor_original_shapes": original_shapes, "vendor_device_descriptors": descriptors,
        "vendor_call_descriptors": taps.descriptors,
        "vendor_tensors": {name: tensor_identity(torch, value) for name, value in vendor.items()},
        "native_tensors": {name: tensor_identity(torch, value) for name, value in native.items()},
        "auxiliary_cpu_oracle_checks": oracle_checks,
        "cpu_oracle_scope": "FP64 WK and FP32 RoPE are auxiliary formulas, not observations of device reduction order",
        "dump": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    (args.output / "decoder-graph-index-diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decoder_graph_index_diagnostic": result}), flush=True)
    return result


def run_decoder_graph(torch, layers, config, state, args):
    import torch_npu
    if (args.forward_widths != [1] or args.forward_replays != 3 or args.mode != "int8"
            or args.graph_residual_modes != ["provided"]):
        raise ValueError("This diagnostic requires C8 M1 provided residual and iterations 0..2")
    os.environ["TILERT_INDEXED_ATTENTION_LIBRARY"] = str(args.indexed_library)
    os.environ["TILERT_SHARED_ATTENTION_LIBRARY"] = str(args.shared_library)
    sys.path.insert(0, str(args.adapter.resolve().parents[3]))
    adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
    if Path(adapter.__file__).resolve() != args.adapter.resolve():
        raise RuntimeError("The tap imported a different native adapter")
    impl = layers[0].self_attn.mla_attn.mla_attn.impl
    taps = IndexerTaps(torch, torch_npu, impl, adapter)
    update = decoder_graph.DecoderInputs.update
    observed = {}

    def remember_inputs(inputs, iteration):
        plan = update(inputs, iteration)
        observed["inputs"] = inputs
        return plan

    def saved_failure(value, destination, save):
        if "report" in observed:
            raise RuntimeError("The diagnostic observed more than one failure")
        try:
            observed["report"] = capture_diagnostic(
                torch, taps, args, value, observed["inputs"], save)
        except Exception:
            error = {"diagnostic_error": traceback.format_exc(),
                     "original_failure_saved": str(destination)}
            (args.output / "decoder-graph-index-diagnostic-error.json").write_text(
                json.dumps(error, indent=2) + "\n")
            raise

    try:
        with observe_full_indexer(taps, adapter), \
                replace_attribute(decoder_graph.DecoderInputs, "update", remember_inputs), \
                observe_failure_saves(torch, saved_failure):
            result = decoder_graph.run_decoder_graph(torch, layers, config, state, args)
    except RuntimeError as error:
        report = observed.get("report")
        if not expected_numerical_failure(error, report):
            raise
        report.update(original_graph_pass=False, expected_numerical_failure_handled=True)
        (args.output / "decoder-graph-index-diagnostic.json").write_text(
            json.dumps(report, indent=2) + "\n")
        print(json.dumps({"completed_graph_index_diagnostic": report}), flush=True)
        return report
    if "report" not in observed:
        raise RuntimeError("The frozen graph failure did not recur")
    return result
