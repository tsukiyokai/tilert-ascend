"""CPU diagnostics of frozen layer68 original/copy/sorted TP16 observations."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import first_difference as frozen
import torch

MODES = ("original", "copy", "sorted")
PHASES = ("serial", "wide", "reset")
NAMES = {f"{mode}-{phase}.pt" for mode in MODES for phase in PHASES}
ROW, INITIAL, END = 13, 8192, 8206
REFERENCE_SHA = "1f3b3d62119d4e887dff91fbabefd397391d787a04414cbe7601e2a290af76fe"
HELPER_SHA = "30ae41b719eed1adefd4ea53d33180c8058c3fee5c38ba4fd8127b44e4504043"
CID = "5ce93694f7329055c1e898ab2337c84cf18f75010f300f3d7ae464bf0ee60289"
SOURCES = {
    "/width-boundaries-v1/run.py": "7e5028edb20bd552b8a0861eb8ff7cca049ac0793126344afd262de8a21ce216",
    "/tilert/models/glm_5_2_ascend/cann_model.py": "0c59db724306ec82db21f2655879874193f69dd1425872180511a6e4d434994d",
    "/vllm/model_executor/models/deepseek_v2.py": "5cd8c8883b18cac91979781a8b191703781971d8c59cb9823167cd3e0dfc2408",
    "/vllm_ascend/device/device_op.py": "b7062ab6757b52a1ce95844dd2c190b7154e5dbb11cc1b31545a13c1628abbb8",
}
OBSERVED_SOURCES = {
    "/observe.py": "0c2f0ccff671b19a9625759c0cb1e6057dbad549512740fa4a040dff9cd49d5e",
    "/routes.py": "8a0a3d8b863145a86bd2f75a04a114b07f69db1f3565317852aca6ad7bbed334",
    "/tests/glm52/run_cann_model.py": "c5aa6a009f3a8b7d0034ca78b658b600c94f45c27e4d5ef8507e0999eeb9e03e",
    "/vllm_ascend/attention/sfa_v1.py": "b5ad39a05b1323ac5048962f137969158298e4f637b5bd3e401cb4c4248ecfa0",
}
require, digest = frozen.require, frozen.digest


def leaves(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, prefix + "." + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from leaves(item, prefix + "." + str(index))


def tensor_hash(value):
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
    }


def tree_hash(value):
    return {key: tensor_hash(tensor) for key, tensor in leaves(value)}


def exact_tree(left, right):
    return bool(dict(leaves(left))) and tree_hash(left) == tree_hash(right)


def source_check(actual, expected):
    for suffix, sha in expected.items():
        matches = [value for path, value in actual.items() if path.endswith(suffix)]
        require(
            matches and all(value == sha for value in matches),
            "Source differs: " + suffix,
        )


def global_state(frame):
    prefixes = (
        "layer68.input_norm.",
        "layer68.post_norm.",
        "layer69.input_norm.",
        "layer68.attention.call0.out",
        "layer68.mlp.call0.in",
        "layer68.mlp.call0.out",
    )
    return tree_hash(
        {
            "outputs": frame["outputs"],
            "global": {
                key: value
                for key, value in frame["observed"]["tensors"].items()
                if key.startswith(prefixes)
            },
            "routes": frame["observed"]["routes"],
        }
    )


def exit_receipts(capture):
    directory = capture.parents[1] / "receipts"
    result = {"status": "producer_exit_unknown", "capture_fail": True, "identities": {}}
    text_path = directory / "width-boundaries-v1-exit.txt"
    json_path = directory / "width-boundaries-v1-exit.json"
    text_code = None
    if text_path.is_file():
        text_code = int(text_path.read_text().strip())
        result["identities"]["driver"] = {
            "path": str(text_path),
            "sha256": digest(text_path),
        }
        result["driver_exit"] = text_code
    if json_path.is_file():
        receipt = json.loads(json_path.read_text())
        state = receipt["state"]
        require(receipt["id"] == CID, "Exit watcher belongs to a different container")
        require(
            state["Running"] is False and state["Status"] == "exited",
            "Exit watcher does not establish terminal state",
        )
        code = state["ExitCode"]
        require(
            type(code) is int and receipt["docker_wait_exit"] == str(code),
            "Docker wait and terminal ExitCode disagree",
        )
        result.update(
            container_id=CID,
            docker_wait_exit=code,
            terminal_state={
                key: state.get(key)
                for key in ("Status", "Running", "ExitCode", "OOMKilled")
            },
        )
        result["identities"]["watcher"] = {
            "path": str(json_path),
            "sha256": digest(json_path),
        }
        result["status"] = (
            "producer_exited"
            if text_code == code
            else "producer_exit_receipts_incomplete_or_disagree"
        )
        result["capture_fail"] = not (text_code == code == 0)
    return result


def receipts(capture, reference):
    require(digest(Path(frozen.__file__)) == HELPER_SHA, "Frozen metric helper changed")
    frozen.bind(reference, REFERENCE_SHA)
    expected = torch.load(reference, map_location="cpu", mmap=True, weights_only=True)
    summary_path = capture / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows = summary["ranks"]
    require(
        len(rows) == 16
        and all(type(item["rank"]) is int for item in rows)
        and {item["rank"] for item in rows} == set(range(16)),
        "Summary needs sixteen unique ranks",
    )
    reports, identities = (
        [],
        {
            "summary": {"path": str(summary_path), "sha256": digest(summary_path)},
            "reference": {"path": str(reference), "sha256": REFERENCE_SHA},
            "reports": {},
        },
    )
    for item in sorted(rows, key=lambda value: value["rank"]):
        rank = item["rank"]
        report_path = capture / f"rank-{rank}.json"
        identities["reports"][str(rank)] = frozen.bind(
            report_path, item["report_sha256"]
        )
        report = json.loads(report_path.read_text())
        require(
            report["rank"] == rank
            and report["status"] in ("captured", "fail")
            and report["source_commit"] == "8a2d3d89be6570cf69ad31596b142d710f7c1b42"
            and report["reference_sha256"] == REFERENCE_SHA
            and report["teacher_sha256"] == frozen.TEACHER_SHA256
            and report["checkpoint_sha256"] == expected["checkpoint_sha256"]
            and report["inputs"]["initial_position"] == INITIAL
            and set(report["trace_sha256"]) == NAMES,
            "Rank identity, source, teacher, checkpoint or raw inventory differs",
        )
        require(
            report["formal_precision_gate_passed"] is False
            and report["performance_measured"] is False,
            "Producer scope unexpectedly changed",
        )
        source_check(report["source_sha256"], SOURCES)
        if reports:
            require(
                report["source_sha256"] == reports[0]["source_sha256"]
                and report["packages"] == reports[0]["packages"],
                "Ranks disagree on source/runtime",
            )
        reports.append(report)
    return summary, rows, reports, expected, identities


def get(frame, key):
    return frame["observed"]["tensors"].get(key)


def argument(frame, name, index, aliases):
    prefix = name + ".call0"
    value = get(frame, prefix + f".in.{index}")
    if value is None:
        for alias in aliases:
            value = get(frame, prefix + ".kwargs." + alias)
            if value is not None:
                break
    return value


def row_value(value, width, row):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim == 0
        or value.shape[0] != width
    ):
        return None
    return value[row : row + 1]


def length(frame, name):
    prefix = "layer68.sfa.call0." + name
    value = get(frame, prefix)
    if isinstance(value, torch.Tensor):
        require(value.dtype in (torch.int32, torch.int64), "SFA length dtype differs")
        return value.flatten().tolist()
    scalars = frame["observed"]["metadata"]["scalar_values"]
    if type(scalars.get(prefix)) is int:
        return [scalars[prefix]]
    return [scalars[key] for key in sorted(scalars) if key.startswith(prefix + ".")]


def logical_kv(frame, number):
    cache = get(frame, f"layer68.sfa.call0.kv_cache.{number}")
    table = get(frame, "layer68.sfa.call0.block_table")
    size = 512 if number == 0 else 64
    require(
        isinstance(cache, torch.Tensor)
        and cache.ndim == 4
        and tuple(cache.shape[1:]) == (128, 1, size)
        and cache.dtype == torch.bfloat16,
        "Expected frozen BF16 paged MLA cache",
    )
    require(
        isinstance(table, torch.Tensor)
        and table.ndim == 2
        and table.shape[0] == 1
        and table.dtype in (torch.int32, torch.int64)
        and table.shape[1] >= (END + 127) // 128,
        "Insufficient logical block table",
    )
    logical = torch.arange(END)
    pages = table[0].long().index_select(0, logical // 128)
    require(
        bool(((pages >= 0) & (pages < cache.shape[0])).all()),
        "Page table exceeds saved physical cache",
    )
    return cache.reshape(-1, size).index_select(0, pages * 128 + logical % 128)


def indices(frame):
    value = get(frame, "layer68.sfa.call0.topk_indices")
    return value.reshape(value.shape[0], -1)


def source_indices(frame):
    value = frame["observed"]["tensors"].get("intervention.original_indices")
    return indices(frame) if value is None else value.reshape(value.shape[0], -1)


def validate(frame, tokens, phase, mode, rank):
    width = 1 if phase == "serial" else 32
    start = INITIAL + ROW if phase == "serial" else INITIAL
    ids = [tokens[ROW]] if phase == "serial" else tokens
    expected = {
        "input_ids": ids,
        "positions": list(range(start, start + width)),
        "slots": list(range(start, start + width)),
        "query_cpu": [0, width],
        "seq_lens": [start + width],
    }
    require(
        all(
            frame["model_inputs"][key].tolist() == value
            for key, value in expected.items()
        ),
        "Saved actual model inputs differ",
    )
    positions = argument(frame, "layer68.forward", 0, ("positions",))
    require(
        isinstance(positions, torch.Tensor)
        and positions.tolist() == expected["positions"],
        "Layer68 actual positions differ from model input buffers",
    )
    require(
        frame["position"] == start and frame["committed_position"] == start + width,
        "Saved position/commit differs",
    )
    require(
        frame.get("token") == ids[0] if width == 1 else frame.get("tokens") == ids,
        "Saved teacher row identity differs",
    )
    observed, issues = frame["observed"], []
    metadata = observed["metadata"]
    source_check(metadata["source_sha256"], OBSERVED_SOURCES)
    modules = (
        "layer68.forward",
        "layer68.input_norm",
        "layer68.post_norm",
        "layer68.attention",
        "layer68.mlp",
        "layer69.input_norm",
        "layer68.sfa",
    )
    if not all(metadata["call_counts"].get(name) == 1 for name in modules):
        issues.append("module/SFA call coverage differs")
    require(
        set(observed["coverage"])
        == {
            "modules_once",
            "actual_fp32_tp_both_phases",
            "one_actual_router",
            "complete_moe_stages",
        },
        "Unexpected coverage schema",
    )
    if not all(observed["coverage"].values()):
        issues.append("producer observation coverage failed")
    require(
        length(frame, "actual_seq_lengths_query") == [width]
        and length(frame, "actual_seq_lengths_key") == [start + width],
        "Actual SFA lengths differ",
    )
    for tag, layout in metadata["tensor_layouts"].items():
        value = observed["tensors"][tag]
        require(
            value.device.type == "cpu"
            and list(value.shape) == layout["shape"]
            and str(value.dtype) == layout["dtype"],
            "Snapshot layout differs: " + tag,
        )
        if (
            tag.startswith(("layer68.sfa.", "layer68.tp."))
            and ".actual_seq_lengths_" not in tag
        ):
            require(
                layout["device"] == f"npu:{rank}", "Original observation device differs"
            )
    consumed = indices(frame)
    require(
        consumed.dtype in (torch.int32, torch.int64)
        and tuple(consumed.shape) == (width, 2048),
        "Actual SFA index shape differs",
    )
    for row, value in enumerate(consumed):
        require(
            value.unique().numel() == 2048
            and bool(((value >= 0) & (value <= start + row)).all()),
            "Invalid actual causal index selection",
        )
    if mode != "original":
        original = observed["tensors"]["intervention.original_indices"].reshape(
            width, -1
        )
        require(metadata["index_intervention"] == mode, "Intervention label differs")
        expected_indices = original if mode == "copy" else original.sort(-1).values
        require(
            original.dtype == consumed.dtype
            and torch.equal(consumed, expected_indices),
            "Actual consumer indices do not implement the declared intervention",
        )
    else:
        require(
            "intervention.original_indices" not in observed["tensors"],
            "Original arm was transformed",
        )
    require(
        len(observed["routes"]) == 1
        and len(metadata["sfa_calls"]) == 1
        and metadata["sfa_calls"][0]["scale"] == 0.0625,
        "Router/SFA call metadata differs",
    )
    for call in metadata["tp_calls"]:
        require(
            call["phase"] in ("attention", "mlp")
            and call["owner"] == f"layer68.{call['phase']}.call0",
            "TP call phase/owner differs",
        )
        for side in ("in", "out"):
            require(
                get(frame, call["tag"] + "." + side).dtype == torch.float32,
                "Actual TP snapshot is not FP32",
            )
    if {call["phase"] for call in metadata["tp_calls"]} != {"attention", "mlp"}:
        issues.append("one TP phase is unobserved")
    return issues


def metric(left, right, *, choice=None, logits=False):
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return {"available": False}
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "available": False,
            "wide_shape": list(left.shape),
            "serial_shape": list(right.shape),
            "wide_dtype": str(left.dtype),
            "serial_dtype": str(right.dtype),
        }
    result = (
        frozen.choices(left, right, choice, END - 1)
        if choice
        else frozen.compare(left, right, logits=logits)
    )
    result["available"] = True
    return result


def comparison(wide, serial, wide_row=ROW, serial_row=0):
    ww = len(wide["model_inputs"]["input_ids"])
    sw = len(serial["model_inputs"]["input_ids"])
    groups = []

    def add(name, pairs):
        fields = {}
        for key, left, right, choice in pairs:
            fields[key] = metric(left, right, choice=choice)
            if key.startswith("effective_KV.") and fields[key]["available"]:
                a = left.contiguous().view(torch.uint8).reshape(END, -1)
                b = right.contiguous().view(torch.uint8).reshape(END, -1)
                fields[key]["different_logical_positions"] = (
                    (a != b).any(-1).nonzero().flatten().tolist()
                )
        groups.append({"boundary": name, "fields": fields})

    def pair(key, label=None, choice=None):
        return (
            label or key,
            row_value(get(wide, key), ww, wide_row),
            row_value(get(serial, key), sw, serial_row),
            choice,
        )

    for name in ("layer68.input_norm",):
        add(
            "norm68.inputs",
            [
                (
                    name + ".input." + str(index),
                    row_value(argument(wide, name, index, aliases), ww, wide_row),
                    row_value(argument(serial, name, index, aliases), sw, serial_row),
                    None,
                )
                for index, aliases in ((0, ("x", "hidden_states")), (1, ("residual",)))
            ],
        )
        add("norm68.outputs", [pair(name + f".call0.out.{index}") for index in (0, 1)])
    add(
        "SFA.inputs",
        [pair("layer68.sfa.call0." + name) for name in ("ql_nope", "q_pe")]
        + [
            (
                "effective_KV." + str(number),
                logical_kv(wide, number),
                logical_kv(serial, number),
                None,
            )
            for number in (0, 1)
        ]
        + [
            (
                "actual_indices",
                indices(wide)[wide_row : wide_row + 1],
                indices(serial)[serial_row : serial_row + 1],
                "indices",
            ),
            (
                "pre_intervention_indices",
                source_indices(wide)[wide_row : wide_row + 1],
                source_indices(serial)[serial_row : serial_row + 1],
                "indices",
            ),
        ]
        + [
            (
                "W_UV",
                get(wide, "layer68.sfa.call0.W_UV"),
                get(serial, "layer68.sfa.call0.W_UV"),
                None,
            )
        ],
    )
    add("SFA.output", [pair("layer68.sfa.call0.out")])

    def tp(phase):
        left = [
            call
            for call in wide["observed"]["metadata"]["tp_calls"]
            if call["phase"] == phase
        ]
        right = [
            call
            for call in serial["observed"]["metadata"]["tp_calls"]
            if call["phase"] == phase
        ]
        if len(left) != len(right):
            groups.append(
                {
                    "boundary": phase + ".TP",
                    "fields": {
                        "call_alignment": {
                            "available": False,
                            "wide_calls": len(left),
                            "serial_calls": len(right),
                        }
                    },
                }
            )
        else:
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                for side in ("in", "out"):
                    add(
                        f"{phase}.TP.{index}.{side}",
                        [
                            (
                                side,
                                row_value(
                                    get(wide, a["tag"] + "." + side), ww, wide_row
                                ),
                                row_value(
                                    get(serial, b["tag"] + "." + side), sw, serial_row
                                ),
                                None,
                            )
                        ],
                    )

    tp("attention")
    add("attention.output", [pair("layer68.attention.call0.out")])
    name = "layer68.post_norm"
    add("post_norm.inputs", [pair(name + f".call0.in.{index}") for index in (0, 1)])
    add("post_norm.outputs", [pair(name + f".call0.out.{index}") for index in (0, 1)])
    add("MLP.input", [pair("layer68.mlp.call0.in.0")])
    add(
        "MoE.router",
        [
            (
                key,
                row_value(wide["observed"]["routes"][0].get(key), ww, wide_row),
                row_value(serial["observed"]["routes"][0].get(key), sw, serial_row),
                "routes" if key == "ids" else None,
            )
            for key in ("router_logits", "weights", "ids")
        ],
    )
    add(
        "MoE.branches_and_combine",
        [
            (
                key,
                row_value(wide["observed"]["moe_stages"].get(key), ww, wide_row),
                row_value(serial["observed"]["moe_stages"].get(key), sw, serial_row),
                None,
            )
            for key in ("input", "shared", "finalize", "combine")
        ],
    )
    tp("mlp")
    add("MLP.output", [pair("layer68.mlp.call0.out")])
    add(
        "layer68.forward.outputs",
        [pair(f"layer68.forward.call0.out.{index}") for index in (0, 1)],
    )
    add(
        "norm69.inputs",
        [pair(f"layer69.input_norm.call0.in.{index}") for index in (0, 1)],
    )
    add(
        "norm69.outputs",
        [pair(f"layer69.input_norm.call0.out.{index}") for index in (0, 1)],
    )

    def first(key, floating_only=False):
        incomplete = False
        for group in groups:
            for name, value in group["fields"].items():
                if not value.get("available"):
                    incomplete = True
                    continue
                if floating_only and value.get("dtype") not in (
                    "torch.bfloat16",
                    "torch.float16",
                    "torch.float32",
                ):
                    continue
                if not value[key]:
                    return {
                        "boundary": group["boundary"],
                        "field": name,
                        "earlier_unavailable": incomplete,
                        "interpretation": "First recorded group, not unobserved internal operator order",
                    }
        return {"boundary": None, "earlier_unavailable": incomplete}

    dispatch = {
        key: {
            name: tensor_hash(value)
            for name, value in frame["observed"]["moe_stages"].items()
            if isinstance(value, torch.Tensor)
            and name not in ("input", "output", "shared", "combine", "finalize")
        }
        for key, frame in (("wide", wide), ("serial", serial))
    }
    return {
        "groups": groups,
        "first_bit_difference": first("bit_equal"),
        "first_floating_bit_difference": first("bit_equal", True),
        "first_original_threshold_failure": first("original_passed"),
        "h69": metric(
            row_value(get(wide, "layer69.input_norm.call0.out.1"), ww, wide_row),
            row_value(get(serial, "layer69.input_norm.call0.out.1"), sw, serial_row),
        ),
        "expanded_dispatch": {
            "mapping": "Not established; no batch-axis row slicing or causal claim",
            **dispatch,
        },
        "selected_row_outputs": {
            name: metric(
                row_value(wide["outputs"][name], ww, wide_row),
                row_value(serial["outputs"][name], sw, serial_row),
                logits=name == "logits",
            )
            for name in ("logits", "last_hidden", "token_ids")
        },
        "whole_outputs_bit_exact": exact_tree(wide["outputs"], serial["outputs"])
        if ww == sw
        else None,
    }


def original_controls(frame, reference, phase):
    expected = reference["serial_target" if phase == "serial" else "wide"]
    row = 0 if phase == "serial" else ROW
    tensors = frame["observed"]["tensors"]
    return {
        "outputs_bit_exact": exact_tree(
            frame["outputs"], {name: expected[name] for name in frame["outputs"]}
        ),
        "h68": metric(
            get(frame, "layer68.input_norm.call0.out.1")[row : row + 1],
            expected["h68"][row : row + 1],
        ),
        "h69": metric(
            get(frame, "layer69.input_norm.call0.out.1")[row : row + 1],
            expected["h69"][row : row + 1],
        ),
        "indices": metric(
            indices(frame)[row : row + 1],
            expected["indices"][row : row + 1].reshape(1, -1),
            choice="indices",
        ),
        "routes": metric(
            frame["observed"]["routes"][0]["ids"][row : row + 1],
            expected["routes"][row : row + 1].reshape(1, -1),
            choice="routes",
        ),
        "observed_tensor_count": len(tensors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary, rank_rows, reports, reference, identities = receipts(
        args.capture, args.reference
    )
    terminal = exit_receipts(args.capture)
    results, all_checks = [], []
    tokens = reference["teacher_tokens"]
    require(len(tokens) == 32 and tokens[ROW] == 24187, "Reference teacher row differs")
    for report, rank_row in zip(
        reports, sorted(rank_rows, key=lambda row: row["rank"]), strict=True
    ):
        rank = report["rank"]
        frames, raw_ids, issues = {}, {}, []
        required_checks = {
            f"{mode}-{phase}.observation_coverage" for mode in MODES for phase in PHASES
        }
        required_checks.update(f"{mode}.reset_exact" for mode in MODES)
        required_checks.update(
            (
                "original.serial_first",
                "original.serial_predictions",
                "original.serial_target",
                "original.wide",
            )
        )
        for mode in MODES:
            for phase in PHASES:
                for layer in range(78):
                    fields = ["latent", "rope"]
                    if layer < 3 or (layer - 2) % 4 == 0:
                        fields.append("index_keys")
                    if 6 <= layer <= 66 and (layer - 2) % 4 == 0:
                        fields.append("index_scales")
                    required_checks.update(
                        f"{mode}.{phase}.prefix.{layer}.{field}" for field in fields
                    )
        if not required_checks <= {check["name"] for check in report["checks"]}:
            issues.append("producer execution/control checks are incomplete")
        if rank_row["passed"] != (report["status"] == "captured"):
            issues.append("summary/rank execution verdicts disagree")
        for mode in MODES:
            for phase in PHASES:
                label = mode + "-" + phase
                name = label + ".pt"
                path = args.capture / f"rank-{rank}" / name
                raw_ids[name] = frozen.bind(path, report["trace_sha256"][name])
                frame = torch.load(
                    path, map_location="cpu", mmap=True, weights_only=True
                )
                issues.extend(
                    label + ": " + issue
                    for issue in validate(frame, tokens, phase, mode, rank)
                )
                if global_state(frame) != rank_row["states"][mode + "." + phase]:
                    issues.append(label + ": global state receipt differs")
                frames[label] = frame
        resets = {
            mode: {
                "tensor_bit_exact": exact_tree(
                    frames[mode + "-reset"], frames[mode + "-wide"]
                ),
                "metadata_exact": frames[mode + "-reset"]["observed"]["metadata"]
                == frames[mode + "-wide"]["observed"]["metadata"],
            }
            for mode in MODES
        }
        result = {
            "rank": rank,
            "producer_status": report["status"],
            "issues": issues,
            "failed_producer_checks": [
                check for check in report["checks"] if check["passed"] is not True
            ],
            "raw_identities": raw_ids,
            "reset": resets,
            "original_controls": {
                phase: original_controls(frames["original-" + phase], reference, phase)
                for phase in ("serial", "wide")
            },
            "serial_vs_wide": {
                mode: comparison(frames[mode + "-wide"], frames[mode + "-serial"])
                for mode in MODES
            },
            "copy_vs_original": {
                phase: comparison(
                    frames["copy-" + phase],
                    frames["original-" + phase],
                    0 if phase == "serial" else ROW,
                    0 if phase == "serial" else ROW,
                )
                for phase in PHASES
            },
        }
        results.append(result)
        all_checks.append(
            not issues
            and report["status"] == "captured"
            and all(check["passed"] is True for check in report["checks"])
            and all(
                item["tensor_bit_exact"] and item["metadata_exact"]
                for item in resets.values()
            )
        )
        del frames
    agreement = all(row["states"] == rank_rows[0]["states"] for row in rank_rows)
    result = {
        "schema": "width-boundaries-read-v1",
        "scope": "Selected native row13 CPU readback, not precision or performance acceptance",
        "phase_evidence": "Filename/report hashes, actual input buffers and lengths; frozen observer records no epoch stamps",
        "reader_sha256": digest(Path(__file__)),
        "metric_helper_sha256": HELPER_SHA,
        "identities": identities,
        "producer_exit": terminal,
        "capture_fail": terminal["capture_fail"]
        or summary["status"] != "captured"
        or not all(all_checks)
        or summary["all_rank_global_states_agree"] != agreement
        or not agreement,
        "all_rank_global_states_agree_recomputed": agreement,
        "original_summary_agreement": summary["all_rank_global_states_agree"],
        "ranks": results,
        "h69_bit_exact_rank_counts": {
            mode: sum(
                item["serial_vs_wide"][mode]["h69"].get("bit_equal", False)
                for item in results
            )
            for mode in MODES
        },
        "sorting_prediction_all16_h69_bit_exact": all(
            item["serial_vs_wide"]["sorted"]["h69"].get("bit_equal", False)
            for item in results
        ),
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "capture_fail",
                    "h69_bit_exact_rank_counts",
                    "sorting_prediction_all16_h69_bit_exact",
                    "formal_precision_gate_passed",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
