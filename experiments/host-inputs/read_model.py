"""Read the frozen TP16 host-input substitution experiment on CPU only."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

MODES = ("separate", "packet")
PHASES = ("first", "wide", "reset")
INDEX_LAYERS = frozenset((0, 1, 2, *range(6, 78, 4)))
INT8_LAYERS = frozenset(range(6, 67, 4))
FILES = {f"{mode}-{phase}.pt" for mode in MODES for phase in PHASES}
OUTPUTS = ("logits", "last_hidden", "token_ids")
REFERENCE_SHA = "1f3b3d62119d4e887dff91fbabefd397391d787a04414cbe7601e2a290af76fe"
TEACHER_SHA = "82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655"
CID = "cedad49dd6d0ac5ec1f21dba6abe97b8246a77004af0e324d5d134f1660e3243"
SOURCES = {
    "cann_model.py": "7e5a9822426468b3e2b0a65da539da79536aac5d13a01c4d8047e8fd0bce47fd",
    "host_inputs.py": "f2f2286fb43ef4dbdf09b2bff6588be69b57f39de1855f5a3363a2b6d0ef6034",
    "model_check.py": "209d8ba6d9f9b4cb262ed6bf246e18f4397ba78fa2dcf7a591d7a415341fe1cf",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_tensor_file(path, expected):
    require(digest(path) == expected, "Tensor file SHA differs: " + str(path))
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def tensor(value, shape, dtype, label):
    require(
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and tuple(value.shape) == shape
        and value.dtype == dtype,
        "Tensor layout differs: " + label,
    )
    if value.is_floating_point():
        require(bool(torch.isfinite(value).all()), "Nonfinite tensor: " + label)


def raw(value):
    return value.contiguous().reshape(-1).view(torch.uint8)


def exact(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(raw(left), raw(right))
    )


def state_hash(outputs):
    return {
        "." + name: {
            "shape": list(outputs[name].shape),
            "dtype": str(outputs[name].dtype),
            "sha256": hashlib.sha256(
                memoryview(raw(outputs[name]).numpy())
            ).hexdigest(),
        }
        for name in OUTPUTS
    }


def cache_fields(layer):
    fields = {"latent": (512, torch.bfloat16), "rope": (64, torch.bfloat16)}
    if layer in INDEX_LAYERS:
        fields["index_keys"] = (
            128,
            torch.int8 if layer in INT8_LAYERS else torch.bfloat16,
        )
    if layer in INT8_LAYERS:
        fields["index_scales"] = (None, torch.float16)
    return fields


def expected_checks():
    names = set()
    for mode in MODES:
        for phase in PHASES:
            names.update(
                f"{mode}.{phase}.{suffix}" for suffix in ("inputs", "old_outputs")
            )
            names.update(
                f"{mode}-{phase}.prefix.{layer}.{field}"
                for layer in range(78)
                for field in cache_fields(layer)
            )
        names.add(mode + ".reset_exact")
    names.update(f"packet.{phase}.same_consumer_state" for phase in PHASES)
    require(len(names) == 1175, "Reader check inventory differs")
    return names


class Checks:
    def __init__(self):
        self.count = 0
        self.failed = []

    def add(self, label, passed):
        self.count += 1
        if not passed:
            self.failed.append(label)


def terminal(receipt):
    state = receipt["state"]
    require(
        receipt["id"] == CID
        and receipt["docker_wait_exit"] == "0"
        and state["Status"] == "exited"
        and state["Running"] is False
        and type(state["ExitCode"]) is int
        and state["ExitCode"] == 0
        and state["OOMKilled"] is False,
        "Actual TP16 container did not exit cleanly",
    )


def flags(report):
    require(
        report["formal_precision_gate_passed"] is False
        and report["performance_measured"] is False,
        "Producer report exceeds the numerical substitution scope",
    )


def source_identity(report):
    sources = report["source_sha256"]
    require(
        bool(sources) and all(is_sha(value) for value in sources.values()),
        "Source inventory is empty or malformed",
    )
    for name, expected in SOURCES.items():
        found = [value for path, value in sources.items() if Path(path).name == name]
        require(found == [expected], "Frozen source identity differs: " + name)
    return sources


def report_checks(report):
    items = report["checks"]
    names = [item["name"] for item in items]
    require(
        len(names) == len(set(names)) == 1175 and set(names) == expected_checks(),
        "Producer checks are missing, duplicated or renamed",
    )
    require(all(item["passed"] is True for item in items), "A producer check failed")
    require(
        all(item["exact"] is True for item in items if ".prefix." in item["name"]),
        "Prefix assertions are not exact",
    )


def validate_outputs(outputs, width, label):
    require(set(outputs) == set(OUTPUTS), "Output fields differ: " + label)
    for name, shape, dtype in (
        ("logits", (width, 154880), torch.float32),
        ("last_hidden", (width, 6144), torch.bfloat16),
        ("token_ids", (width,), torch.int64),
    ):
        tensor(outputs[name], shape, dtype, label + "." + name)
    require(
        torch.equal(outputs["token_ids"], outputs["logits"].argmax(-1)),
        "Returned IDs differ from actual logits argmax: " + label,
    )


def validate_frame(frame, mode, phase, tokens, label):
    width = 1 if phase == "first" else 32
    require(
        set(frame)
        == {
            "mode",
            "phase",
            "position",
            "committed_position",
            "outputs",
            "inputs",
            "new_cache",
        }
        and frame["mode"] == mode
        and frame["phase"] == phase
        and type(frame["position"]) is int
        and frame["position"] == 8192
        and type(frame["committed_position"]) is int
        and frame["committed_position"] == 8192 + width,
        "Frame mode/phase/position differs: " + label,
    )
    wanted = {
        "ids": (tokens[:width], torch.int32),
        "positions": (list(range(8192, 8192 + width)), torch.int64),
        "slots": (list(range(8192, 8192 + width)), torch.int32),
        "length": ([8192 + width], torch.int32),
        "query": ([0, width], torch.int32),
    }
    require(set(frame["inputs"]) == set(wanted), "Input fields differ: " + label)
    for name, (values, dtype) in wanted.items():
        actual = frame["inputs"][name]
        tensor(actual, (len(values),), dtype, label + ".inputs." + name)
        require(
            actual.tolist() == values,
            "Actual input values differ: " + label + "." + name,
        )
    validate_outputs(frame["outputs"], width, label)
    cache = frame["new_cache"]
    require(
        isinstance(cache, (list, tuple)) and len(cache) == 78,
        "Need exactly 78 new-cache layers: " + label,
    )
    for layer, entry in enumerate(cache):
        fields = cache_fields(layer)
        require(
            set(entry) == set(fields), "Cache fields differ: " + label + f".{layer}"
        )
        for name, (columns, dtype) in fields.items():
            shape = (width,) if columns is None else (width, columns)
            tensor(entry[name], shape, dtype, label + f".cache.{layer}." + name)


def same_frame(checks, label, left, right):
    for group in ("inputs", "outputs"):
        for name in left[group]:
            checks.add(
                label + "." + group + "." + name,
                exact(left[group][name], right[group][name]),
            )
    for layer in range(78):
        for name in left["new_cache"][layer]:
            checks.add(
                label + f".cache.{layer}." + name,
                exact(left["new_cache"][layer][name], right["new_cache"][layer][name]),
            )


def read(args):
    terminal(load_json(args.exit))
    contract = load_json(args.contract)
    flags(contract)
    require(
        contract["source_sha256"] == SOURCES
        and contract["controls_reference_sha256"] == REFERENCE_SHA,
        "Pre-registered source/reference contract differs",
    )
    reference = load_tensor_file(args.reference, REFERENCE_SHA)
    tokens = reference["teacher_tokens"]
    require(
        isinstance(tokens, list)
        and len(tokens) == 32
        and all(type(token) is int and 0 <= token < 154880 for token in tokens),
        "Reference teacher does not contain 32 vocabulary IDs",
    )
    for phase, width in (("serial_first", 1), ("wide", 32)):
        validate_outputs(
            {name: reference[phase][name] for name in OUTPUTS}, width, phase
        )
    summary = load_json(args.capture / "summary.json")
    flags(summary)
    ranks = summary["ranks"]
    require(
        summary["status"] == "pass"
        and summary["all_rank_global_states_agree"] is True
        and len(ranks) == 16
        and {item["rank"] for item in ranks} == set(range(16))
        and all(
            type(item["rank"]) is int and item["status"] == "pass" for item in ranks
        ),
        "Summary lacks a successful complete TP16 world",
    )
    checks = Checks()
    sources, shared_inputs, global_states = None, None, None
    frames_read = 0
    for item in sorted(ranks, key=lambda row: row["rank"]):
        rank = item["rank"]
        path = args.capture / f"rank-{rank}.json"
        require(digest(path) == item["report_sha256"], f"Rank{rank} report SHA differs")
        report = load_json(path)
        flags(report)
        require(
            type(report["rank"]) is int
            and report["rank"] == rank
            and report["status"] == "pass",
            f"Rank{rank} report identity/status differs",
        )
        report_checks(report)
        current_sources = source_identity(report)
        sources = current_sources if sources is None else sources
        require(current_sources == sources, "Ranks loaded different source inventories")
        require(
            report["reference_sha256"] == REFERENCE_SHA
            and report["teacher_sha256"] == TEACHER_SHA
            and report["checkpoint_sha256"] == reference["checkpoint_sha256"],
            f"Rank{rank} checkpoint/teacher/reference differs",
        )
        inputs = report["inputs"]
        require(
            inputs["initial_position"] == inputs["prompt_length"] == 8192
            and inputs["first_token_id"] == tokens[0]
            and all(
                is_sha(inputs[name])
                for name in (
                    "prompt_sha256",
                    "prefix_sha256",
                    "capture_manifest_sha256",
                )
            )
            and Path(inputs["prefix_file"]).name == f"rank-{rank}.pt",
            f"Rank{rank} prefix/teacher position identity differs",
        )
        common = {
            name: value
            for name, value in inputs.items()
            if name not in ("prefix_sha256", "prefix_file")
        }
        shared_inputs = common if shared_inputs is None else shared_inputs
        require(common == shared_inputs, "Ranks use different prefix requests")
        files = report["files_sha256"]
        directory = args.capture / f"rank-{rank}"
        require(
            set(files) == FILES and {p.name for p in directory.glob("*.pt")} == FILES,
            f"Rank{rank} needs exactly six recorded PT files",
        )
        require(
            set(report["states"]) == {name[:-3] for name in FILES},
            f"Rank{rank} state coverage differs",
        )
        frames, states = {}, {}
        for mode in MODES:
            for phase in PHASES:
                label = mode + "-" + phase
                frame = load_tensor_file(
                    directory / (label + ".pt"), files[label + ".pt"]
                )
                validate_frame(frame, mode, phase, tokens, f"rank{rank}." + label)
                frames[label] = frame
                frames_read += 1
                control = reference["serial_first" if phase == "first" else "wide"]
                for name in OUTPUTS:
                    checks.add(
                        f"rank{rank}.{label}.reference.{name}",
                        exact(frame["outputs"][name], control[name]),
                    )
                states[label] = state_hash(frame["outputs"])
                checks.add(
                    f"rank{rank}.{label}.recorded_state",
                    states[label] == report["states"][label],
                )
        checks.add(f"rank{rank}.summary_states", states == item["states"])
        if global_states is None:
            global_states = states
        checks.add(f"rank{rank}.final_output_global_states", states == global_states)
        for mode in MODES:
            same_frame(
                checks,
                f"rank{rank}.{mode}.reset_exact",
                frames[mode + "-wide"],
                frames[mode + "-reset"],
            )
        for phase in PHASES:
            same_frame(
                checks,
                f"rank{rank}.{phase}.packet_vs_separate",
                frames["separate-" + phase],
                frames["packet-" + phase],
            )
    require(frames_read == 96, "Need all 96 PT frames")
    return {
        "status": "verified_component_state" if not checks.failed else "fail",
        "ranks": 16,
        "frames": frames_read,
        "producer_checks": 16 * 1175,
        "producer_prefix_assertions": 16 * 6 * 193,
        "independent_comparisons": checks.count,
        "failed_count": len(checks.failed),
        "failed": checks.failed,
        "all_rank_final_output_states_agree": not any(
            name.endswith("final_output_global_states") for name in checks.failed
        ),
        "source_sha256": SOURCES,
        "identity": {
            name: digest(getattr(args, name))
            for name in ("exit", "contract", "reference")
        }
        | {"summary": digest(args.capture / "summary.json")},
        "evidence_boundary": "All raw inputs/outputs/new cache were read. Prefix bytes were not saved: preservation remains a complete producer assertion. Global states cover final outputs only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "exit", "contract", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {
        "schema": "host-inputs-model-independent-read-v1",
        "scope": "Frozen one-input TP16 eager numerical producer substitution; no HF precision or timing claim",
        "reader_sha256": digest(Path(__file__)),
        "formal_precision_gate_passed": False,
        "performance_measured": False,
    }
    try:
        result.update(read(args))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        result.update(status="fail", failed_count=1, failed=[str(error)])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "failed_count",
                    "formal_precision_gate_passed",
                    "performance_measured",
                )
            }
        )
    )
    return int(result["status"] == "fail")


if __name__ == "__main__":
    raise SystemExit(main())
