"""Host-only validators and scalar schedules for the CANN MTP runner.

Synthetic Torch CPU prefix files exercise serialization and identity gates.
Independent scalar target/draft models exercise the real MtpRoundExecutor.
No checkpoint weights, official model arithmetic, NPU or TP collective runs.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import run_cann_mtp_rounds as runner  # noqa: E402
from check_mtp_round import (  # noqa: E402
    autoregressive, make_executor,
)
from prefix_io import prompt_identity  # noqa: E402
from tilert.pd_vllm.ascend_layout import (  # noqa: E402
    FULL_INDEX_LAYERS, INT8_INDEX_LAYERS, PARENT_HIDDEN_REGION,
    cache_layout, parent_hidden_metadata,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(operation):
    try:
        operation()
    except (ValueError, KeyError):
        return
    raise AssertionError("Invalid capture was accepted")


def check_schedule(width, prefix, budget, mode, eos_position, room):
    serial, prompt, first = make_executor(
        width, prefix, True, mode, budget=budget, room=room,
    )
    oracle, _ = autoregressive(prompt, first, budget, serial.target.capacity)
    stops = () if eos_position is None else (oracle[eos_position],)
    oracle, _ = autoregressive(
        prompt, first, budget, serial.target.capacity, stops,
    )
    expected = runner.serial_run(
        serial.target, serial.draft, first, serial.seed_parent_hidden,
        budget, stops,
    )
    speculative, _, _ = make_executor(
        width, prefix, True, mode, budget=budget, room=room,
    )
    main_prefix = copy.deepcopy(speculative.target.rows)
    draft_prefix = copy.deepcopy(speculative.draft.rows)
    observations = []
    actual = runner.speculative_run(
        speculative.target, speculative.draft, first,
        speculative.seed_parent_hidden, width, budget, stops,
        observe=lambda value: observations.append(copy.deepcopy(value)),
    )
    for key in ("tokens", "positions", "finish_reason"):
        require(actual[key] == expected[key], key)
    require(actual["tokens"] == oracle, "independent greedy oracle")
    positions = runner.final_positions(prefix, len(oracle))
    require(tuple(actual["positions"]) == positions, "pending lengths")
    for left, right, end in zip(
        (serial.target, serial.draft),
        (speculative.target, speculative.draft), positions,
    ):
        require(left.rows[:end] == right.rows[:end], "valid cache parents")
    require(speculative.target.rows[:prefix] == main_prefix, "main prefix")
    require(speculative.draft.rows[:prefix - 1] == draft_prefix,
            "draft prefix")
    require(len(observations) == len(actual["rounds"]), "round observations")
    require(sum(len(item["commit"]["accepted_tokens"])
                for item in actual["rounds"]) == len(oracle) - 1,
            "actual accepted count")
    if len(oracle) == 1:
        require(not speculative.draft.calls and not serial.draft.calls,
                "terminal first token must not seed MTP")
    else:
        require(serial.target.rows[positions[0] - 1][0] == oracle[-2],
                "last pending token must remain outside main KV")


def synthetic_capture(directory, torch):
    checkpoint = {name: "a" * 64 for name in (
        "config.json", "quant_model_description.json",
        "quant_model_weights.safetensors.index.json",
    )}
    prompt = [17, 19]
    identity = prompt_identity(prompt)
    contract = parent_hidden_metadata(len(prompt))

    def layer(number, length):
        values = {"latent": torch.ones((length, 512), dtype=torch.bfloat16),
                  "rope": torch.ones((length, 64), dtype=torch.bfloat16)}
        if number in FULL_INDEX_LAYERS:
            dtype = (torch.int8 if number in INT8_INDEX_LAYERS
                     else torch.bfloat16)
            values["index_keys"] = torch.ones((length, 128), dtype=dtype)
        if number in INT8_INDEX_LAYERS:
            values["index_scales"] = torch.ones(length, dtype=torch.float16)
        return values

    main = {"metadata_sha256": checkpoint, "prompt_token_ids": prompt,
            "layers": [layer(number, 2) for number in range(78)]}
    draft = {
        "metadata_sha256": checkpoint, "layer": layer(78, 1),
        "parent_hidden": {
            **contract, "request_id": "host-fixture", "generation": 1,
            "tensor": torch.ones((1, 6144), dtype=torch.bfloat16),
        },
    }
    receipts = []
    for rank in range(16):
        files = {}
        for role, bundle in (("main", main), ("mtp", draft)):
            path = directory / f"{role}-prefix" / f"rank-{rank}.pt"
            runner.save_bundle(bundle, path)
            files[role] = {"sha256": runner.digest(path)}
        receipts.append({
            "rank": rank, "status": "saved", "request_id": "host-fixture",
            "generation": 1, "seq_len": 2, "mtp_seq_len": 1,
            "prompt_sha256": identity["prompt_sha256"],
            "checkpoint_sha256": checkpoint, "files": files,
            PARENT_HIDDEN_REGION: contract,
        })
    manifest = {
        "status": "capture_complete", "reset_complete": True,
        "request_id": "host-fixture", "generation": 1, "seq_len": 2,
        "prompt_token_ids": prompt, "prompt_sha256": identity["prompt_sha256"],
        "first_token_id": 23, "delivered_tokens": [23], "ranks": receipts,
        PARENT_HIDDEN_REGION: contract,
        "runtime": {
            "layout": cache_layout(128, parent_hidden=True), "world_size": 16,
            "checkpoint_sha256": checkpoint,
            "source_sha256": {"synthetic-host-fixture": "b" * 64},
        },
    }
    runner.atomic_json(directory / "manifest.json", manifest)
    args = SimpleNamespace(capture_case=directory, max_tokens=4, capacity=128,
                           width=4)
    return args, checkpoint, manifest, draft


def check_capture(directory, torch, record):
    args, checkpoint, manifest, draft = synthetic_capture(directory, torch)

    def valid():
        for rank in range(16):
            main, mtp, parent, first, inputs = runner.load_inputs(
                args, rank, checkpoint,
            )
            require(len(main) == 78 and first == 23, "main/g0")
            require(main[0]["latent"].shape[0] == 2
                    and mtp["latent"].shape[0] == 1, "P/P-1")
            require(inputs["rank"] == rank
                    and parent.shape == (1, 6144), "rank/parent")
    record("capture.valid_all_16_ranks", valid)
    for width in (8, 32):
        boundary = SimpleNamespace(**vars(args))
        boundary.width = width
        boundary.capacity = 2 + boundary.max_tokens + width - 2
        record(f"capture.full_width_{width}", lambda: runner.load_inputs(
            boundary, 0, checkpoint,
        ))
        boundary.capacity -= 1
        record(f"capture.reject_short_width_{width}", lambda: rejects(
            lambda: runner.load_inputs(boundary, 0, checkpoint),
        ))
    terminal = SimpleNamespace(**vars(args))
    terminal.max_tokens, terminal.capacity = 1, 3
    record("capture.first_token_only", lambda: runner.load_inputs(
        terminal, 0, checkpoint,
    ))
    mutations = {
        "v2": lambda value: value["runtime"].update(layout=cache_layout(128)),
        "missing_parent": lambda value: value.pop(PARENT_HIDDEN_REGION),
        "rotated_parent": lambda value: value[PARENT_HIDDEN_REGION].update(
            rotated=True),
        "parent_position": lambda value: value[PARENT_HIDDEN_REGION].update(
            position=2),
        "source_missing": lambda value: value["runtime"].pop("source_sha256"),
        "rank_bool": lambda value: value["ranks"][1].update(rank=True),
        "prompt": lambda value: value["prompt_token_ids"].__setitem__(0, 31),
        "receipt_generation": lambda value: value["ranks"][15].update(
            generation=2),
        "mtp_sha": lambda value: value["ranks"][15]["files"]["mtp"].update(
            sha256="c" * 64),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(manifest)
        mutate(changed)
        runner.atomic_json(args.capture_case / "manifest.json", changed)
        record("capture.reject." + name, lambda: rejects(
            lambda: runner.load_inputs(args, 15, checkpoint),
        ))
    runner.atomic_json(args.capture_case / "manifest.json", manifest)
    for name, mutate in {
        "parent_rid": lambda value: value["parent_hidden"].update(
            request_id="another"),
        "parent_nan": lambda value: value["parent_hidden"]["tensor"].fill_(
            float("nan")),
        "mtp_length": lambda value: value["layer"].update(
            latent=torch.zeros((2, 512), dtype=torch.bfloat16)),
    }.items():
        changed = copy.deepcopy(draft)
        mutate(changed)
        path = args.capture_case / "mtp-prefix/rank-15.pt"
        runner.save_bundle(changed, path)
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["ranks"][15]["files"]["mtp"]["sha256"] = (
            runner.digest(path)
        )
        runner.atomic_json(args.capture_case / "manifest.json",
                           changed_manifest)
        record("capture.reject." + name, lambda: rejects(
            lambda: runner.load_inputs(args, 15, checkpoint),
        ))


def check_cache(torch):
    values = {"latent": torch.ones((2, 2), dtype=torch.bfloat16),
              "rope": torch.ones((2, 2), dtype=torch.bfloat16)}
    cache = {"main": [copy.deepcopy(values) for _ in range(78)],
             "mtp": [copy.deepcopy(values)]}
    cache["main"][0].update(
        index_keys=torch.ones((2, 2), dtype=torch.int8),
        index_scales=torch.ones(2, dtype=torch.float16),
    )
    checks = []
    runner.compare_cache("normal", cache, cache, checks)
    require(all(item["passed"] for item in checks), "same cache")
    numeric = [item for item in checks if "exact" in item]
    require(all(item["exact"] == ("index_" in item["name"])
                for item in numeric), "integer/scale exact flags")
    for field in ("latent", "index_keys", "index_scales"):
        wrong = copy.deepcopy(cache)
        wrong["main"][0][field][0] = 0
        checks = []
        runner.compare_cache("mutation", wrong, cache, checks)
        if field == "latent":
            changed = next(item for item in checks
                           if item["name"] == "mutation.main.0.latent.row.0")
            require(changed["diagnostic_only"] and not changed["bit_exact"]
                    and changed["passed"], "finite cross-width diagnostic")
        else:
            require(any(not item["passed"] for item in checks), field)
    wrong = copy.deepcopy(cache)
    for value in (float("nan"), float("inf")):
        wrong["main"][0]["latent"][0] = value
        checks = []
        runner.compare_cache("nonfinite", wrong, cache, checks)
        require(any(not item["passed"] for item in checks), "nonfinite cache")
    wrong["main"][0]["latent"][0] = 0
    checks = []
    runner.compare_cache("reset", wrong, cache, checks, exact=True)
    require(any(not item["passed"] for item in checks), "exact reset mutation")
    checks = []
    runner.compare_cache("reset", cache, cache, checks, exact=True)
    require(all(item["exact"] for item in checks if "exact" in item),
            "reset compares every value exactly")


def check_row_diagnostics(torch):
    reference = torch.ones((15, 512), dtype=torch.bfloat16)
    wrong = reference.clone()
    wrong[14] = 1.03125
    whole = []
    runner.compare_tensor("whole_matrix", wrong, reference, whole)
    entry = {"latent": reference, "rope": reference[:, :64]}
    expected = {"main": [entry] * 78, "mtp": [entry]}
    actual = {"main": [{**entry, "latent": wrong}, *([entry] * 77)],
              "mtp": [entry]}
    checks = []
    runner.compare_cache("row_metric", actual, expected, checks)
    row = next(item for item in checks
               if item["name"] == "row_metric.main.0.latent.row.14")
    require(row["diagnostic_only"] and not row["bit_exact"]
            and row["nrmse"] == 0.03125 and row["passed"],
            "per-row error remains visible without a cross-width gate")
    require(row["nrmse"] > whole[0]["nrmse"], "row error is not diluted")


def check_phase_preparation():
    events = []

    class Storage:
        def zero_(self):
            events.append("zero_shared_storage")

    class Engine:
        def __init__(self, name, storage):
            self.name, self.storage, self.position = name, storage, 99

        def reset(self):
            events.append(self.name + ".reset")
            self.position = 0

        def inject(self, prefix):
            events.append(self.name + ".inject")
            self.position = prefix

    storage = Storage()
    target, draft = Engine("main", storage), Engine("mtp", storage)
    for label, should_clear in (("teacher", True), ("serial", True),
                                ("speculative", True),
                                ("reset", False)):
        events.clear()
        cleared = runner.prepare_phase(target, draft, 2, 1, label)
        expected = ["main.reset", "mtp.reset"]
        if should_clear:
            expected.append("zero_shared_storage")
        expected.extend(("main.inject", "mtp.inject"))
        require(cleared == should_clear and events == expected,
                "fresh serial/speculative, dirty reset, before injection")
        require((target.position, draft.position) == (2, 1), "P/P-1")


def check_teacher(directory, torch):
    prompt = [17, 19]
    teacher = {
        "prompt_sha256": runner.hashlib.sha256(
            (json.dumps(prompt) + "\n").encode(),
        ).hexdigest(),
        "prompt_tokens": 2, "first_token_id": 23,
        "teacher_tokens": list(range(23, 55)),
        "expected_next_tokens": list(range(24, 56)),
    }
    path = directory / "teacher.json"
    runner.atomic_json(path, teacher)
    runner.atomic_json(directory / "manifest.json", {
        "prompt_token_ids": prompt, "seq_len": 2, "first_token_id": 23,
    })
    loaded = runner.load_teacher(path, directory, 32, 34)
    require(loaded["sha256"] == runner.digest(path), "teacher identity")
    for width, capacity in ((32, 33), (6, 34)):
        rejects(lambda: runner.load_teacher(path, directory, width, capacity))
    for field, value in (("prompt_sha256", "0" * 64),
                         ("first_token_id", 22),
                         ("expected_next_tokens", [24, 24, *range(26, 56)])):
        runner.atomic_json(path, {**teacher, field: value})
        rejects(lambda: runner.load_teacher(path, directory, 32, 34))
    runner.atomic_json(path, {
        **teacher, "teacher_tokens": teacher["teacher_tokens"][:8],
        "expected_next_tokens": teacher["expected_next_tokens"][:8],
    })
    rejects(lambda: runner.load_teacher(path, directory, 8, 34))

    class Target:
        def __init__(self, fault):
            self.position, self.consumed, self.fault = 2, [], fault

        def verify(self, ids, full_logits):
            require(full_logits, "teacher diagnostic retains logits")
            self.consumed.extend(ids)
            logits = torch.zeros((len(ids), 64))
            for row, token in enumerate(ids):
                predicted = 0 if self.fault and token == 24 else token + 1
                logits[row, predicted] = 2
            self.position += len(ids)
            return SimpleNamespace(token_ids=logits.argmax(-1), logits=logits,
                                   last_hidden=torch.tensor(ids)[:, None])

    for fault in (False, True):
        for width in (1, 8, 32):
            target, observed = Target(fault), []
            report = runner.teacher_run(target, loaded, width, directory,
                                        observed.append)
            require(target.consumed == teacher["teacher_tokens"],
                    "predictions must never replace frozen inputs")
            require(len(observed) == 32 // width and target.position == 34,
                    "teacher consumes each row once")
            require(report["passed"] is not fault, "teacher label gate")
            require(report["first_difference"] == (1 if fault else None),
                    "first teacher difference")
            if fault:
                saved = torch.load(
                    directory / f"teacher-m{width}-first-difference.pt",
                    weights_only=True,
                )
                require(saved["position"] == 3 and saved["expected"] == 25
                        and saved["prediction"] == 0
                        and saved["last_hidden"].tolist() == [24],
                        "first difference binds logits and consumed input")
                artifact = report["first_difference_file"]
                require(runner.digest(Path(artifact["path"]))
                        == artifact["sha256"], "first-difference file hash")

    class ShortTarget(Target):
        def verify(self, ids, full_logits):
            result = super().verify(ids, full_logits)
            result.token_ids = result.token_ids[:-1]
            return result

    rejects(lambda: runner.teacher_run(ShortTarget(False), loaded, 32,
                                       directory, lambda value: None))

    class WrongLogitsTarget(Target):
        def verify(self, ids, full_logits):
            result = super().verify(ids, full_logits)
            result.logits.zero_()
            return result

    rejects(lambda: runner.teacher_run(WrongLogitsTarget(False), loaded, 32,
                                       directory, lambda value: None))

    from check_cann_choices import Model

    class RoutedTarget(Target):
        def __init__(self, width, shared):
            super().__init__(True)
            self.model = Model(width, torch.int32)
            if shared:
                ids = torch.arange(8).repeat(width, 1)
                for expert in self.model.experts:
                    expert.result = (object(), ids)

        def verify(self, ids, full_logits):
            self.model.forward()
            return super().verify(ids, full_logits)

    for width, shared in ((1, False), (8, False), (32, False), (32, True)):
        target = RoutedTarget(width, shared)
        report = runner.teacher_run(target, loaded, width, directory,
                                    lambda value: None, capture_routes=True)
        artifact = report["routing"]
        saved = torch.load(artifact["path"], weights_only=True)
        require(saved["routes"].shape == (32, 75, 8)
                and saved["input_ids"] == loaded["teacher_tokens"]
                and saved["teacher_sha256"] == loaded["sha256"],
                "route rows bind actual teacher inputs and 75 MoE layers")
        expected = torch.arange(8)[None, None, :].expand(32, 75, 8)
        if not shared:
            expected = (expected
                        + (torch.arange(32) % width)[:, None, None] * 8
                        + torch.arange(3, 78)[None, :, None]) % 256
        require(torch.equal(saved["routes"], expected),
                "route axes preserve input, layer and top-k order")
        expected_active = [[8 if shared else width * 8] * 75] * (32 // width)
        expected_max = [[width if shared else 1] * 75] * (32 // width)
        require(artifact["sha256"] == runner.digest(Path(artifact["path"]))
                and artifact["active_experts"] == expected_active
                and artifact["max_rows_per_expert"] == expected_max,
                "counts distinguish shared and dispersed experts")
        require(not report["passed"] and report["first_difference"] == 1,
                "route diagnostics do not change the teacher label gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    torch.set_num_threads(4)
    checks = []

    def record(name, operation):
        check = {"name": name, "passed": False}
        try:
            operation()
            check["passed"] = True
        except Exception as error:
            check["error"] = f"{type(error).__name__}: {error}"
        checks.append(check)

    for width in (2, 4):
        for prefix in (1, 127, 128, 8192):
            for mode in ("accept", "reject", "mixed"):
                for budget, eos, room in (
                    (1, None, 8), (2, None, 8), (7, None, 8),
                    (7, 0, 8), (7, 2, 8), (7, None, 2),
                ):
                    name = (f"schedule.{width}.{prefix}.{mode}"
                            f".{budget}.{eos}.{room}")
                    record(name, lambda: check_schedule(
                        width, prefix, budget, mode, eos, room,
                    ))
    with tempfile.TemporaryDirectory(prefix="cann-mtp-host-") as temporary:
        check_capture(Path(temporary), torch, record)
    with tempfile.TemporaryDirectory(prefix="teacher-host-") as temporary:
        record("teacher.fixed_inputs_and_first_difference",
               lambda: check_teacher(Path(temporary), torch))
    record("cache.compare_and_mutations", lambda: check_cache(torch))
    record("cache.per_row_diagnostics",
           lambda: check_row_diagnostics(torch))
    record("cache.phase_preparation", check_phase_preparation)
    path = Path(runner.__file__)
    record("runner.ast", lambda: ast.parse(path.read_text()))
    record("runner.cli", lambda: runner.parse_arguments([
        "--checkpoint", "/fixture", "--capture-case", "/fixture/case",
        "--output", "/fixture/out", "--source-commit", "fixture",
    ]))
    record("no_npu_import", lambda: require(
        "torch_npu" not in sys.modules, "NPU runtime was imported",
    ))
    source = [Path(__file__), path,
              *[Path(__file__).with_name(name) for name in (
                  "check_mtp_round.py", "run_cann_model.py",
                  "run_native_decode.py", "prefix_io.py", "frozen_teacher.py",
                  "check_cann_choices.py",
              )],
              ROOT / "tilert/models/glm_5_2_ascend/cann_choices.py",
              ROOT / "tilert/models/glm_5_2_ascend/mtp_round.py",
              ROOT / "tilert/models/glm_5_2_ascend/mtp_state.py",
              ROOT / "tilert/pd_vllm/ascend_layout.py",
              ROOT / "tilert/pd_vllm/prefix_capture.py"]
    report = {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic host validator and independent scalar schedule",
        "numerical_model_execution": False, "npu_execution": False,
        "torch_npu_imported": "torch_npu" in sys.modules,
        "torch_version": str(torch.__version__),
        "source_sha256": {str(file.relative_to(ROOT)): runner.digest(file)
                          for file in source},
        "checks": checks,
    }
    runner.atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "failed": [item for item in checks
                                 if not item["passed"]]}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
