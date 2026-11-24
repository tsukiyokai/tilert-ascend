"""Check causal M2/M4 CANN forwards against serial M1 on one TP16 host.

A capture case supplies the full-P main cache and its first token. Explicit
tokens must start with that token; otherwise fixed candidate IDs follow it.
Without a capture case, --tokens supplies a cold-cache teacher sequence.
This checks target execution, not independent HF precision or performance.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import traceback

from run_cann_model import check_prefix, compare_step, digest, load_capture


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE_FIELDS = {
    "latent": "latent", "rope": "rope",
    "index": "index_keys", "index_scale": "index_scales",
}
TENSOR_FIELDS = ("token_ids", "logits", "last_hidden")
TRACE_FIELDS = ("hidden_states", "layer_logits", "indices", "routes")
VOCAB_SIZE = 154880


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-case", type=Path)
    parser.add_argument("--tokens", nargs="+", type=int)
    parser.add_argument("--width", type=int, choices=(2, 4), default=4)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--device-offset", type=int, choices=range(16),
                        default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--init-method", default="env://")
    parser.add_argument("--layer-trace", action="store_true")
    args = parser.parse_args(argv)
    if args.capture_case is None and args.tokens is None:
        parser.error("Provide --capture-case or explicit cold --tokens")
    if args.tokens is not None and (
        len(args.tokens) != args.width
        or any(not 0 <= token < VOCAB_SIZE for token in args.tokens)
    ):
        parser.error("Provide exactly width valid vocabulary IDs")
    if args.capacity < 128 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("Capacity >= 128 and memory utilization in (0, 1]")
    return args


def teacher_tokens(width, explicit, first=None):
    if explicit is None:
        if first is None:
            raise ValueError("Cold execution requires explicit teacher IDs")
        explicit = [first, *(198 + index for index in range(width - 1))]
    tokens = tuple(explicit)
    if (width not in (2, 4) or len(tokens) != width
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE
                   for token in tokens)):
        raise ValueError("Teacher IDs must match width 2 or 4")
    if first is not None and tokens[0] != first:
        raise ValueError("First teacher ID differs from the captured token")
    return tokens


def future_variants(tokens):
    for row in range(1, len(tokens)):
        changed = list(tokens)
        changed[row] = (changed[row] + 1) % VOCAB_SIZE
        yield row, tuple(changed)


def rollback_plan(position, tokens):
    width = len(tokens)
    retained_rows = width // 2 if width == 4 else 0
    retained = position + retained_rows
    return {
        "retained_rows": retained_rows, "retained_position": retained,
        "old_high_watermark": position + width,
        "new_end": retained + 1,
        "replacement": (tokens[retained_rows] + 1) % VOCAB_SIZE,
    }


def cpu_copy(value):
    return value.detach().cpu().clone()


def cache_rows(engine, start, stop):
    return tuple({
        field: cpu_copy(layer[key][start:stop]) if key in layer else None
        for field, key in CACHE_FIELDS.items()
    } for layer in engine.cache)


def snapshot(engine, result, tokens, position):
    return {
        "token": list(tokens), "position": position,
        "committed_position": engine.position,
        "model_inputs": {
            name: cpu_copy(getattr(engine, name)) for name in
            ("input_ids", "positions", "slots", "query_cpu", "seq_lens")
        },
        **{field: cpu_copy(getattr(result, field))
           for field in TENSOR_FIELDS},
        "cache_rows": cache_rows(engine, position, engine.position),
    }


def capture(engine, tokens, layer_trace):
    import torch
    from vllm.config import set_current_vllm_config
    from tilert.models.glm_5_2_ascend.cann_choices import CannChoices
    from tilert.models.glm_5_2_ascend.cann_trace import CannTrace

    position, width = engine.position, len(tokens)
    if not layer_trace:
        return snapshot(engine, engine.verify(tokens), tokens, position)
    with torch.npu.device(engine.device), torch.inference_mode(), (
        set_current_vllm_config(engine.config)
    ):
        with (
            CannTrace(engine.model, width) as trace,
            CannChoices(engine.model, width) as choices,
        ):
            result = engine.verify(tokens)
            hidden = trace.finish(result)
            indices, routes = choices.finish()
        normalized = engine.model.model.norm(torch.cat(hidden[1:-1]))
        projected = engine.model.compute_logits(normalized)
        if projected.shape != (77 * width, result.logits.shape[-1]):
            raise ValueError("Layer projections must retain 77 * width rows")
        layers = (*projected.split(width, dim=0), result.logits)
        saved = snapshot(engine, result, tokens, position)
        saved.update(hidden_states=tuple(map(cpu_copy, hidden)),
                     layer_logits=tuple(map(cpu_copy, layers)),
                     indices=tuple(map(cpu_copy, indices)),
                     routes=tuple(None if value is None else cpu_copy(value)
                                  for value in routes),
                     choice_source_sha256=choices.source_metadata())
        return saved


def merge_serial(items):
    import torch

    def concatenate(values):
        return None if values[0] is None else torch.cat(values).clone()

    result = {
        "token": [item["token"][0] for item in items],
        "position": items[0]["position"],
        "committed_position": items[-1]["committed_position"],
        **{key: concatenate([item[key] for item in items])
           for key in TENSOR_FIELDS},
        "cache_rows": tuple({
            key: concatenate([item["cache_rows"][layer][key]
                              for item in items])
            for key in CACHE_FIELDS
        } for layer in range(78)),
    }
    for key in TRACE_FIELDS:
        if key in items[0]:
            result[key] = tuple(concatenate([item[key][layer]
                                            for item in items])
                                for layer in range(len(items[0][key])))
    return result


def snapshot_rows(saved, start, stop):
    def rows(value):
        return value[start:stop].clone() if value is not None else None

    result = {
        **saved, "token": saved["token"][start:stop],
        "position": saved["position"] + start,
        "committed_position": saved["position"] + stop,
        **{key: rows(saved[key]) for key in TENSOR_FIELDS},
        "cache_rows": tuple({key: rows(value) for key, value in layer.items()}
                            for layer in saved["cache_rows"]),
    }
    for key in TRACE_FIELDS:
        if key in saved:
            result[key] = tuple(map(rows, saved[key]))
    return result


def first_rows(saved, count):
    return snapshot_rows(saved, 0, count)


def check_record(label, saved, ids, checks):
    import torch

    start, width = saved["position"], len(ids)
    end = start + width
    expected = {
        "input_ids": list(ids), "positions": list(range(start, end)),
        "slots": list(range(start, end)), "query_cpu": [0, width],
        "seq_lens": [end],
    }
    for name, values in expected.items():
        actual = saved["model_inputs"][name].tolist()
        checks.append({"name": f"{label}.model_inputs.{name}",
                       "passed": actual == values, "actual": actual,
                       "expected": values})
    checks.append({"name": label + ".committed_position",
                   "passed": saved["committed_position"] == end})
    token_ids, logits = saved["token_ids"], saved["logits"]
    valid = token_ids.shape == (width,) and logits.ndim == 2
    valid = valid and logits.shape[0] == width
    checks.append({"name": label + ".argmax_token_ids",
                   "passed": valid and torch.equal(token_ids,
                                                    logits.argmax(-1))})


def accepted_prefix(prefixes, saved, count):
    import torch

    result = []
    for layer, new in enumerate(saved["cache_rows"]):
        values = {}
        for field, key in CACHE_FIELDS.items():
            if new[field] is None:
                continue
            old = prefixes[layer][key] if prefixes else new[field][:0]
            values[key] = torch.cat((old, new[field][:count])).clone()
        result.append(values)
    return result


def check_tail(label, actual, expected, checks):
    from run_native_decode import compare_tensor

    for layer, (left, right) in enumerate(zip(actual, expected)):
        for field in CACHE_FIELDS:
            compare_tensor(f"{label}.{layer}.{field}", left[field],
                           right[field], checks, exact=True)


def run_checks(engine, prefixes, tokens, directory, layer_trace, checked):
    import torch

    checked("output.prepare", lambda: directory.mkdir(
        parents=True, exist_ok=True,
    ))
    checks, hashes, choice_sources = [], {}, {}

    def prepare(label, prefix):
        def initialize():
            engine.reset()
            engine.storage.zero_()
            if prefix:
                engine.inject(prefix)
            expected = len(prefix[0]["latent"]) if prefix else 0
            checks.append({"name": label + ".initial_position",
                           "passed": engine.position == expected,
                           "actual": engine.position, "expected": expected})
            check_prefix(engine, prefix, checks, label + ".prefix.before")
        checked(label + ".prepare", initialize)

    def record(label, ids):
        def operation():
            result = capture(engine, ids, layer_trace)
            check_record(label, result, ids, checks)
            for name, sha in result.get("choice_source_sha256", {}).items():
                if name in choice_sources and choice_sources[name] != sha:
                    raise ValueError("Choice capture source changed")
                choice_sources[name] = sha
            return result
        return checked(label + ".forward", operation)

    def save(label, value):
        def operation():
            path = directory / (label + ".pt")
            torch.save(value, path)
            hashes[path.name] = digest(path)
        checked(label + ".save", operation)

    def compare(label, actual, expected, exact=False):
        def operation():
            for row in range(len(expected["token"])):
                compare_step(
                    f"{label}.row-{row}",
                    snapshot_rows(actual, row, row + 1),
                    snapshot_rows(expected, row, row + 1),
                    checks, exact=exact,
                )
        checked(label + ".compare", operation)

    def prefix_check(label, prefix):
        checked(label, lambda: check_prefix(engine, prefix, checks, label))

    prepare("serial", prefixes)
    serial_rows = [record(f"serial-{row}", (token,))
                   for row, token in enumerate(tokens)]
    serial = checked("serial.merge", lambda: merge_serial(serial_rows))
    serial["model_inputs"] = [item["model_inputs"] for item in serial_rows]
    save("serial", serial)
    prefix_check("serial.prefix.after", prefixes)
    prepare("wide", prefixes)
    wide = record("wide", tokens)
    save("wide", wide)
    compare("serial_vs_wide", wide, serial)
    prefix_check("wide.prefix.after", prefixes)
    prepare("reset", prefixes)
    repeated = record("reset", tokens)
    save("reset", repeated)
    compare("reset", repeated, wide, exact=True)
    prefix_check("reset.prefix.after", prefixes)

    plan = rollback_plan(wide["position"], tokens)
    accepted = checked("rollback.accepted", lambda: accepted_prefix(
        prefixes, repeated, plan["retained_rows"],
    ))
    checked("rollback.truncate", lambda: engine.truncate(
        plan["retained_position"],
    ))
    prefix_check("rollback.dirty.prefix.before", accepted)
    tail = checked("rollback.old_tail", lambda: cache_rows(
        engine, plan["new_end"], plan["old_high_watermark"],
    ))
    checks.append({
        "name": "rollback.uncovered_tail_nonzero",
        "passed": any(bool(torch.count_nonzero(value))
                      for layer in tail for value in layer.values()
                      if value is not None),
    })
    dirty = record("rollback-dirty", (plan["replacement"],))
    save("rollback-dirty", dirty)
    after = checked("rollback.dirty_tail", lambda: cache_rows(
        engine, plan["new_end"], plan["old_high_watermark"],
    ))
    checked("rollback.tail_unchanged", lambda: check_tail(
        "rollback.tail_unchanged", after, tail, checks,
    ))
    save("rollback-invalid-tail", {"plan": plan, "before": tail,
                                   "after_dirty": after})
    prefix_check("rollback.dirty.prefix.after", accepted)
    prepare("rollback.clean", accepted)
    clean = record("rollback-clean", (plan["replacement"],))
    save("rollback-clean", clean)
    compare("rollback.same_m1", dirty, clean, exact=True)
    zeros = tuple({key: torch.zeros_like(value) if value is not None else None
                   for key, value in layer.items()} for layer in tail)
    clean_tail = checked("rollback.clean_tail", lambda: cache_rows(
        engine, plan["new_end"], plan["old_high_watermark"],
    ))
    checked("rollback.tail_zero", lambda: check_tail(
        "rollback.clean_tail_zero", clean_tail, zeros, checks,
    ))
    prefix_check("rollback.clean.prefix.after", accepted)
    checks.append({"name": "rollback.below_old_high_watermark",
                   "passed": clean["committed_position"]
                   == dirty["committed_position"] == plan["new_end"]
                   < plan["old_high_watermark"]})

    for row, changed in future_variants(tokens):
        label = f"future-{row}"
        prepare(label, prefixes)
        actual = record(label, changed)
        save(label, actual)
        compare(label + ".earlier_rows", first_rows(actual, row),
                first_rows(wide, row))
        prefix_check(label + ".prefix.after", prefixes)
    return checks, hashes, plan, wide["token_ids"].tolist(), choice_sources


def execute(args):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    import torch.distributed as distributed
    from vllm.distributed import get_world_group
    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint
    from tilert.pd_vllm.decode_timing import atomic_json

    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    device = (rank + args.device_offset) % 16
    metadata = checkpoint_fingerprint(args.checkpoint)
    prefixes, first = [], None
    inputs = {"initial_position": 0}
    if args.capture_case:
        prefixes, first_tokens, inputs = load_capture(
            args.capture_case, rank, metadata,
        )
        first = first_tokens[0]
    tokens = teacher_tokens(args.width, args.tokens, first)
    if inputs["initial_position"] + len(tokens) > args.capacity:
        raise ValueError("Prefix and candidate rows exceed capacity")
    inputs["teacher_tokens"] = list(tokens)
    engine = CannModel(args.checkpoint, f"npu:{device}", rank,
                       args.init_method, args.capacity,
                       args.gpu_memory_utilization)
    group = get_world_group().cpu_group

    def gather(value):
        values = [None] * 16
        distributed.all_gather_object(values, value, group=group)
        return values

    def checked(label, operation):
        value, error = None, None
        try:
            value = operation()
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        failures = {number: error for number, error in enumerate(gather(error))
                    if error}
        if failures:
            raise RuntimeError(f"{label} failed: {failures}")
        return value

    identities = gather({"teacher_tokens": list(tokens),
                         "initial_position": inputs["initial_position"],
                         "capture": inputs.get("capture_manifest_sha256")})
    if any(value != identities[0] for value in identities):
        raise ValueError("Sixteen ranks need the same capture and teachers")
    checks, traces, plan, predicted, choice_sources = run_checks(
        engine, prefixes, tokens, args.output / f"rank-{rank}",
        args.layer_trace, checked,
    )
    passed = all(item["passed"] for item in checks)
    source_paths = [
        Path(__file__), Path(__file__).with_name("run_cann_model.py"),
        Path(__file__).with_name("run_native_decode.py"),
        Path(__file__).with_name("prefix_io.py"),
        ROOT / "tilert/models/glm_5_2_ascend/cann_trace.py",
    ]
    report = {
        "status": "pass" if passed else "fail", "rank": rank,
        "physical_device": device, "world_size": 16, "width": len(tokens),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "inputs": inputs,
        "initial_position": inputs["initial_position"],
        "final_position": engine.position, "capacity": args.capacity,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "checkpoint_sha256": metadata, "rollback": plan,
        "engine_settings": {
            "dtype": str(engine.config.model_config.dtype),
            "quantization": engine.config.model_config.quantization,
            "enforce_eager": engine.config.model_config.enforce_eager,
            "kv_cache_dtype": engine.config.cache_config.cache_dtype,
            "block_size": engine.config.cache_config.block_size,
            "tensor_parallel_size": (
                engine.config.parallel_config.tensor_parallel_size
            ),
        },
        "precision_verified": False, "performance_measured": False,
        "independent_reference": False, "draft_model_loaded": False,
        "scope": "Target serial/wide, reset, future inputs and stale tail",
        "coverage": {
            "final_logits": True, "final_normalized_hidden": True,
            "all_layers_new_kv": True, "frozen_prefix_exact": bool(prefixes),
            "reset_exact": True, "rollback_same_m1_exact": True,
            "future_rows": list(range(1, len(tokens))),
            "intermediate_hidden_states": args.layer_trace,
            "intermediate_layer_logits": args.layer_trace,
            "routing_choices": args.layer_trace,
            "index_choices": args.layer_trace, "mtp": False,
        },
        "interpretation": "Cross-width and future-input differences may "
        "include shape-dependent rounding; failures do not identify a mask "
        "error without the corresponding operator boundary.",
        "packages": {name: version(name) for name in
                     ("torch", "torch-npu", "vllm", "vllm-ascend")},
        "source_sha256": {**engine.source_metadata(), **choice_sources,
                          **{str(path.resolve()): digest(path)
                             for path in source_paths}},
        "trace_sha256": traces, "checks": checks,
    }
    path = args.output / f"rank-{rank}.json"
    checked("report", lambda: atomic_json(path, report))
    summaries = gather({"rank": rank, "passed": passed,
                        "predicted_tokens": predicted,
                        "report_sha256": digest(path)})
    same = all(item["predicted_tokens"] == predicted for item in summaries)
    success = same and all(item["passed"] for item in summaries)
    def save_summary():
        if rank != 0:
            return
        summary = {"status": "pass" if success else "fail",
                   "all_ranks_agree": same, "width": len(tokens),
                   "precision_verified": False, "performance_measured": False,
                   "ranks": summaries}
        atomic_json(args.output / "summary.json", summary)
        print(json.dumps(summary))
    checked("summary", save_summary)
    engine.shutdown()
    return 0 if success else 1


def main(argv=None):
    args = parse_arguments(argv)
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
