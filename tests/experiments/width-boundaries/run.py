"""Capture layer68 in one resident TP16 world, with index-order controls."""

import argparse
import hashlib
import inspect
import json
import os
import sys
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from observe import observe

REFERENCE_SHA = (
    "1f3b3d62119d4e887dff91fbabefd397391d787a04414cbe7601e2a290af76fe"
)
MODES = ("original", "copy", "sorted")
ROW, INITIAL, LAYER = 13, 8192, 68


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(item) for item in value]
    return value


def tensor_leaves(value, prefix=""):
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, prefix + "." + str(key))
    elif isinstance(value, (tuple, list)):
        for number, item in enumerate(value):
            yield from tensor_leaves(item, prefix + "." + str(number))


def exact(left, right):
    import torch

    a, b = dict(tensor_leaves(left)), dict(tensor_leaves(right))
    return (
        a.keys() == b.keys()
        and bool(a)
        and all(
            a[key].shape == b[key].shape
            and a[key].dtype == b[key].dtype
            and torch.equal(
                a[key].reshape(-1).view(torch.uint8),
                b[key].reshape(-1).view(torch.uint8),
            )
            for key in a
        )
    )


def state_hash(value):
    import torch

    hashes = {}
    for key, tensor in tensor_leaves(value):
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        hashes[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
        }
    return hashes


def output_view(result):
    return cpu(
        {
            name: getattr(result, name)
            for name in ("logits", "last_hidden", "token_ids")
        }
    )


def output_matches(actual, expected):
    return exact(actual, {name: expected[name] for name in actual})


def transformed(impl, mode, captured):
    """Change only the consumer's index ownership/order, out of place."""
    if mode == "original":
        return nullcontext()
    original = impl._execute_sparse_flash_attention_process

    def attention(
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):

        # Frozen positions fit exactly in FP32; float sort supports the same
        # integer ordering on the deployed NPU backend.
        indices = (
            topk_indices.clone()
            if mode == "copy"
            else topk_indices.float()
            .sort(dim=-1)
            .values.to(topk_indices.dtype)
        )
        if captured is not None:
            captured["tensors"]["intervention.original_indices"] = (
                topk_indices.detach().clone()
            )
            captured["metadata"]["index_intervention"] = mode
        return original(
            ql_nope,
            q_pe,
            kv_cache,
            indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table=block_table,
        )

    return patch.object(
        impl, "_execute_sparse_flash_attention_process", attention
    )


def forward(engine, tokens, mode, *, capture):
    import torch

    context = observe(engine) if capture else nullcontext(None)
    impl = engine.model.model.layers[LAYER].self_attn.mla_attn.mla_attn.impl
    initial = engine.position
    with context as captured, transformed(impl, mode, captured):
        result = engine.verify(tuple(tokens), full_logits=True)
    torch.npu.synchronize()
    if captured is not None:
        counts = captured["metadata"].get("call_counts", {})
        modules = ("layer68.forward", "layer68.input_norm", "layer68.post_norm",
                   "layer68.attention", "layer68.mlp", "layer69.input_norm",
                   "layer68.sfa")
        phases = {row["phase"] for row in captured["metadata"]["tp_calls"]}
        stages = {"input", "output", "expanded_row_idx", "finalize_probs_fp32",
                  "finalize", "probs_fp32", "topk_ids", "dispatch_a8",
                  "dispatch_scale", "group_counts", "fused_a8", "fused_scale",
                  "gmm2_group_counts", "down", "combine", "shared"}
        captured["coverage"] = {
            "modules_once": all(counts.get(name) == 1 for name in modules),
            "actual_fp32_tp_both_phases": phases == {"attention", "mlp"},
            "one_actual_router": len(captured["routes"]) == 1,
            "complete_moe_stages": stages <= captured["moe_stages"].keys(),
        }
    outputs = output_view(result)
    if (
        not torch.isfinite(outputs["logits"]).all()
        or not torch.isfinite(outputs["last_hidden"]).all()
        or not torch.equal(outputs["token_ids"], outputs["logits"].argmax(-1))
    ):
        raise ValueError("Actual returned predictions/logits are inconsistent")
    model_inputs = cpu({name: getattr(engine, name) for name in (
        "input_ids", "positions", "slots", "query_cpu", "seq_lens",
    )})
    if (engine.position != initial + len(tokens)
            or model_inputs["input_ids"].tolist() != list(tokens)
            or model_inputs["positions"].tolist()
            != list(range(initial, initial + len(tokens)))
            or model_inputs["slots"].tolist()
            != list(range(initial, initial + len(tokens)))
            or model_inputs["query_cpu"].tolist() != [0, len(tokens)]
            or model_inputs["seq_lens"].tolist() != [initial + len(tokens)]):
        raise ValueError("Actual model inputs/committed position differ")
    return {"outputs": outputs, "model_inputs": model_inputs,
            "observed": cpu(captured)}


def global_state(frame):
    """Global model results only; TP-local partials are intentionally separate."""
    tensors = frame["observed"]["tensors"]
    prefixes = (
        "layer68.input_norm.",
        "layer68.post_norm.",
        "layer69.input_norm.",
        "layer68.attention.call0.out",
        "layer68.mlp.call0.in",
        "layer68.mlp.call0.out",
    )
    return state_hash(
        {
            "outputs": frame["outputs"],
            "global": {
                key: value
                for key, value in tensors.items()
                if key.startswith(prefixes)
            },
            "routes": frame["observed"]["routes"],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "checkpoint",
        "capture-case",
        "teacher",
        "reference",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check-inputs-only", action="store_true")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.source), str(args.source / "tests/glm52")]
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    from frozen_teacher import load_teacher
    from run_cann_model import check_prefix, load_capture

    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    rank = int(os.environ.get("RANK", "0"))
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    prefixes, _, inputs = load_capture(args.capture_case, rank, checkpoint)
    teacher = load_teacher(args.teacher, args.capture_case, 32, 9344)
    if digest(args.reference) != REFERENCE_SHA:
        raise ValueError("Frozen native reference views differ")
    reference = torch.load(
        args.reference, map_location="cpu", weights_only=True
    )
    if (
        reference["checkpoint_sha256"] != checkpoint
        or reference["teacher_tokens"] != teacher["teacher_tokens"]
        or inputs["initial_position"] != INITIAL
    ):
        raise ValueError(
            "Reference, teacher and injected prefix identities differ"
        )
    if args.check_inputs_only:
        print(
            json.dumps(
                {
                    "rank": rank,
                    "inputs_admitted": True,
                    "reference_sha256": REFERENCE_SHA,
                    "npu_executed": False,
                }
            )
        )
        return 0
    import torch_npu
    from torch import distributed
    from vllm.distributed import get_world_group

    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    if (
        int(os.environ["WORLD_SIZE"]) != 16
        or int(os.environ["LOCAL_RANK"]) != rank
    ):
        raise ValueError("One local TP16 torchrun world required")
    torch.set_num_threads(1)
    torch_npu.npu.set_device(rank)
    torch.npu.matmul.allow_hf32 = False
    name = torch_npu.npu.get_device_name(rank)
    if not any(sku in name for sku in ("9392", "9382")):
        raise ValueError("Actual Ascend A3 required")
    engine = CannModel(
        args.checkpoint,
        f"npu:{rank}",
        rank,
        capacity=9344,
        gpu_memory_utilization=0.85,
        attention_kernel="sfa",
        fp32_head=True,
        graph_width=32,
        compile_target=False,
        super_kernel=False,
    )
    directory = args.output / f"rank-{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    checks, artifacts, states = [], {}, {}
    sources = engine.source_metadata()
    for function in (main, observe, check_prefix, load_capture):
        path = Path(inspect.getfile(function)).resolve()
        sources[str(path)] = digest(path)

    def prepare(clear):
        engine.reset()
        if engine.position != 0:
            raise ValueError("Reset did not clear the position")
        if clear:
            engine.storage.zero_()
        engine.inject(prefixes)
        torch.npu.synchronize()
        if engine.position != INITIAL:
            raise ValueError("Prefix injection changed its length")

    def save(label, frame):
        path = directory / (label + ".pt")
        with path.open("xb") as stream:
            torch.save(frame, stream)
        artifacts[path.name] = digest(path)
        checks.append({"name": label + ".observation_coverage",
                       "passed": all(frame["observed"]["coverage"].values())})

    try:
        if (
            engine.graph is not None
            or not engine.fp32_head
            or "float32" not in engine.tp_reduction
        ):
            raise ValueError(
                "Keep eager SFA, FP32 head and production FP32 TP"
            )
        tokens = teacher["teacher_tokens"]
        for mode in MODES:
            prepare(True)
            predictions, history = [], []
            target = None
            for row, token in enumerate(tokens[: ROW + 1]):
                actual = forward(engine, [token], mode, capture=row == ROW)
                predictions.extend(actual["outputs"]["token_ids"].tolist())
                history.append(
                    cpu(
                        {
                            key: value[INITIAL + row : INITIAL + row + 1]
                            for key, value in engine.cache[LAYER].items()
                        }
                    )
                )
                if row == 0 and mode == "original":
                    checks.append(
                        {
                            "name": "original.serial_first",
                            "passed": output_matches(
                                actual["outputs"], reference["serial_first"]
                            ),
                        }
                    )
                if row == ROW:
                    target = actual
            target.update(
                token=tokens[ROW],
                position=INITIAL + ROW,
                committed_position=engine.position,
                history=history,
            )
            save(mode + "-serial", target)
            states[mode + ".serial"] = global_state(target)
            if mode == "original":
                checks.extend(
                    [
                        {
                            "name": "original.serial_predictions",
                            "passed": predictions
                            == reference["serial_predictions"][: ROW + 1],
                        },
                        {
                            "name": "original.serial_target",
                            "passed": output_matches(
                                target["outputs"], reference["serial_target"]
                            ),
                        },
                    ]
                )
            check_prefix(engine, prefixes, checks, mode + ".serial")
            first = None
            for phase in ("wide", "reset"):
                prepare(phase != "reset")
                actual = forward(engine, tokens, mode, capture=True)
                actual.update(
                    tokens=tokens,
                    position=INITIAL,
                    committed_position=engine.position,
                    new_cache=cpu(
                        {
                            key: value[INITIAL : INITIAL + 32]
                            for key, value in engine.cache[LAYER].items()
                        }
                    ),
                )
                save(mode + "-" + phase, actual)
                states[mode + "." + phase] = global_state(actual)
                if phase == "wide":
                    first = actual
                    if mode == "original":
                        checks.append(
                            {
                                "name": "original.wide",
                                "passed": output_matches(
                                    actual["outputs"], reference["wide"]
                                ),
                            }
                        )
                else:
                    checks.append(
                        {
                            "name": mode + ".reset_exact",
                            "passed": exact(actual, first),
                        }
                    )
                check_prefix(engine, prefixes, checks, mode + "." + phase)
            print(
                json.dumps({"rank": rank, "mode": mode, "captured": True}),
                flush=True,
            )
        report = {
            "status": "captured"
            if all(c["passed"] for c in checks)
            else "fail",
            "scope": "Resident A3 TP16 layer68, one8Kinput, M1x14/M32/reset; no timing",
            "rank": rank,
            "checks": checks,
            "trace_sha256": artifacts,
            "source_sha256": sources,
            "source_commit": args.source_commit,
            "checkpoint_sha256": checkpoint,
            "inputs": inputs,
            "teacher_sha256": teacher["sha256"],
            "reference_sha256": REFERENCE_SHA,
            "packages": {
                key: version(key)
                for key in ("torch", "torch-npu", "vllm", "vllm-ascend")
            },
            "formal_precision_gate_passed": False,
            "performance_measured": False,
        }
        report_path = args.output / f"rank-{rank}.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        rows = [None] * 16
        distributed.all_gather_object(
            rows,
            {
                "rank": rank,
                "passed": report["status"] == "captured",
                "report_sha256": digest(report_path),
                "states": states,
            },
            group=get_world_group().cpu_group,
        )
        agreement = all(row["states"] == rows[0]["states"] for row in rows)
        passed = agreement and all(row["passed"] for row in rows)
        if rank == 0:
            summary = {
                "status": "captured" if passed else "fail",
                "ranks": rows,
                "all_rank_global_states_agree": agreement,
                "formal_precision_gate_passed": False,
                "performance_measured": False,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "status": summary["status"],
                        "all_rank_global_states_agree": agreement,
                    }
                ),
                flush=True,
            )
        return 0 if passed else 1
    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
