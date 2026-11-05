"""Build the first differing rank's independent CPU expanded-Value fixture."""

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
HELPER_SHA = "50e2a19d621ee65d0700a20f38284a3ad83357ff090967e2ae755de500020a00"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(
        value.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def tensor(value, shape, dtype, name, *, finite=True):
    require(
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and value.shape == shape
        and value.dtype == dtype,
        "Bad tensor: " + name,
    )
    if finite:
        require(bool(torch.isfinite(value).all()), "Nonfinite: " + name)


def run(args, report):
    torch.set_num_threads(8)
    require(
        file_sha(args.helper) == HELPER_SHA, "Expanded helper source differs"
    )
    prior = json.loads(args.cpu_report.read_text())
    require(
        prior["status"] == "diagnostic_complete"
        and prior["cpu_backend"]["torch"] == torch.__version__
        and prior["cpu_backend"]["mkldnn_enabled"] is False,
        "Prior independent CPU report/environment differs",
    )
    expected_sha = prior["sfa"]["expanded_value"]["bf16_score_probability"][
        "o_proj_input_sha256"
    ]
    reader = ReferenceWeights(args.checkpoint, dense_cache_bytes=1024**3)
    require(
        reader.contract["metadata_sha256"]
        == prior["identity"]["checkpoint_metadata_sha256"],
        "Checkpoint metadata differs",
    )
    for relative, loaded in (
        (
            "tilert/models/glm_5_2_ascend/reference_weights.py",
            ReferenceWeights,
        ),
        ("tests/glm52/check_mla.py", metrics),
    ):
        bound = [
            sha
            for path, sha in prior["source_sha256"].items()
            if Path(path).as_posix().endswith("/" + relative)
        ]
        require(
            len(bound) == 1
            and bound[0]
            == file_sha(args.source_root / relative)
            == file_sha(inspect.getsourcefile(loaded)),
            "CPU source differs: " + relative,
        )
    summary_path = args.captures / "summary.json"
    summary = json.loads(summary_path.read_text())
    require(
        len(summary["ranks"]) == 16
        and sorted(item["rank"] for item in summary["ranks"])
        == list(range(16)),
        "Capture world reports incomplete",
    )
    control_ranks = {item["rank"]: item for item in prior["sfa"]["ranks"]}
    projections, actual_projections, ranks = [], [], []
    first = None
    for rank in range(16):
        raw_path = args.captures / f"rank-{rank}/layer-00.pt"
        receipt_path = args.captures / f"rank-{rank}.json"
        receipt = json.loads(receipt_path.read_text())
        summary_rank = next(
            item for item in summary["ranks"] if item["rank"] == rank
        )
        require(
            file_sha(receipt_path) == summary_rank["report_sha256"],
            "Rank receipt hash differs",
        )
        bound = [
            sha
            for path, sha in receipt["artifact_sha256"].items()
            if Path(path).parts[-2:] == (f"rank-{rank}", "layer-00.pt")
        ]
        require(
            bound == [file_sha(raw_path)], "Rank tensor artifact hash differs"
        )
        require(
            receipt["rank"] == rank
            and receipt["layers"]
            == receipt["steps"]
            == receipt["query_width"]
            == 1,
            "Capture scope differs",
        )
        data = torch.load(raw_path, map_location="cpu", weights_only=True)
        require(
            data["rank"] == rank
            and data["layer"] == 0
            and data["positions"] == [8192]
            and data["checkpoint_sha256"]
            == receipt["checkpoint_sha256"]
            == reader.contract["metadata_sha256"],
            "Capture row/checkpoint differs",
        )
        b, meta = data["boundaries"], data["boundary_metadata"]["sfa"]
        require(
            meta["source_sha256"] == HELPER_SHA and meta["scale"] == 0.0625,
            "Capture did not execute the bound expanded helper",
        )
        layout = {
            "query_nope": ((1, 4, 512), torch.bfloat16),
            "query_rope": ((1, 4, 64), torch.bfloat16),
            "key_cache": ((73, 128, 1, 512), torch.bfloat16),
            "rope_cache": ((73, 128, 1, 64), torch.bfloat16),
            "indices": ((1, 1, 2048), torch.int32),
            "block_table": ((1, 73), torch.int32),
            "query_lengths": ((1,), torch.int32),
            "key_lengths": ((1,), torch.int32),
        }
        inputs = {}
        for name, (shape, dtype) in layout.items():
            value = b["sfa." + name]
            tensor(
                value,
                shape,
                dtype,
                name,
                finite=name not in ("key_cache", "rope_cache"),
            )
            inputs[name] = value
        tensor(
            b["o_proj.input"], (1, 1024), torch.bfloat16, "actual Oproj input"
        )
        tensor(
            b["sfa.output"],
            (1, 4, 256),
            torch.bfloat16,
            "expanded helper output",
        )
        require(
            torch.equal(b["sfa.output"].reshape(1, 1024), b["o_proj.input"]),
            "Expanded helper output does not link to Oproj",
        )
        require(
            inputs["query_lengths"].tolist() == [1]
            and inputs["key_lengths"].tolist() == [8193],
            "Sequence lengths differ",
        )
        indices = inputs["indices"].reshape(-1).long()
        table = inputs["block_table"][0].long()
        require(
            indices.unique().numel() == 2048
            and bool(torch.all((indices >= 0) & (indices < 8193))),
            "Invalid sparse selection",
        )
        require(
            table[:65].unique().numel() == 65
            and bool(torch.all((table[:65] >= 0) & (table[:65] < 73))),
            "Invalid active pages",
        )
        positions = torch.arange(8193)
        physical = table[positions // 128] * 128 + positions % 128
        latent = inputs["key_cache"].view(-1, 512).index_select(0, physical)
        rope = inputs["rope_cache"].view(-1, 64).index_select(0, physical)
        tensor(latent, (8193, 512), torch.bfloat16, "valid logical latent")
        tensor(rope, (8193, 64), torch.bfloat16, "valid logical RoPE")
        control = control_ranks[rank]
        for name in ("query_nope", "query_rope", "indices", "block_table"):
            require(
                tensor_sha(inputs[name]) == control["sha256"][name],
                "Input differs from CPU control: " + name,
            )
        require(
            tensor_sha(latent) == control["logical_latent_sha256"]
            and tensor_sha(rope) == control["logical_rope_sha256"],
            "Logical KV differs from CPU control",
        )
        selected = latent[indices].float()
        score = inputs["query_nope"][0].float() @ selected.T
        score = (
            score + inputs["query_rope"][0].float() @ rope[indices].float().T
        )
        score = (score * 0.0625).bfloat16()
        probability = torch.softmax(score.float(), dim=-1).bfloat16()
        aligned = torch.zeros((4, 8193), dtype=torch.bfloat16)
        aligned[:, indices] = probability
        name = "model.layers.0.self_attn.kv_b_proj"
        require(
            reader.quantization.get(name + ".weight", "FLOAT") == "FLOAT",
            "KV-b must be FLOAT",
        )
        rows = (rank * 1792, (rank + 1) * 1792)
        weight = reader.tensor(
            name + ".weight", rows=rows, dtype=torch.bfloat16
        )
        uv = weight.reshape(4, 448, 512)[:, 192:].transpose(1, 2).contiguous()
        prefix = reader._floating_product(
            name, latent[:8192], rows=rows
        ).reshape(8192, 4, 448)
        current = reader._floating_product(
            name, latent[8192:], rows=rows
        ).reshape(1, 4, 448)
        value_prefix = prefix[:, :, 192:].permute(1, 0, 2).contiguous()
        value_current = current[:, :, 192:].permute(1, 0, 2).contiguous()
        pv = torch.matmul(
            aligned.unsqueeze(1),
            torch.cat((value_prefix, value_current), dim=1),
        ).transpose(0, 1)
        expected = pv.reshape(1, 1024)
        projections.append(expected)
        actual_projections.append(b["o_proj.input"])
        item = {
            "rank": rank,
            "artifact_sha256": bound[0],
            "receipt_sha256": file_sha(receipt_path),
            "cpu_output_sha256": tensor_sha(expected),
            "full_npu_output_sha256": tensor_sha(b["o_proj.input"]),
            "cpu_vs_full_npu": {
                "exact": torch.equal(expected, b["o_proj.input"]),
                **metrics(expected, b["o_proj.input"]),
            },
        }
        ranks.append(item)
        if first is None and not item["cpu_vs_full_npu"]["exact"]:
            first = {
                "inputs": inputs,
                "weights": {"uv": uv},
                "cpu": {
                    "scores_bf16": score,
                    "probability_bf16": probability,
                    "aligned_probability": aligned,
                    "value_prefix": value_prefix,
                    "value_current": value_current,
                    "pv": pv,
                },
                "full_npu": {
                    "pv": b["sfa.output"],
                    "o_proj_input": b["o_proj.input"],
                },
                "rank": rank,
                "capture_inputs_identity": data["inputs"],
                "capture_metadata": meta,
            }
    reconstructed = tensor_sha(torch.cat(projections, dim=-1))
    report.update(
        cpu_report_sha256=file_sha(args.cpu_report),
        helper_sha256=HELPER_SHA,
        capture_summary_sha256=file_sha(summary_path),
        capture_summary_status=summary["status"],
        expected_combined_sha256=expected_sha,
        reconstructed_combined_sha256=reconstructed,
        full_npu_combined_sha256=tensor_sha(
            torch.cat(actual_projections, dim=-1)
        ),
        ranks=ranks,
    )
    require(
        reconstructed == expected_sha,
        "Independent CPU combined SHA does not reproduce prior CPU candidate",
    )
    require(
        first is not None,
        "No differing rank; do not create a synthetic discrepancy fixture",
    )
    require(
        not any(
            name.startswith("transformers.models.") for name in sys.modules
        )
        and "torch_npu" not in sys.modules,
        "CPU builder imported an HF model or NPU backend",
    )
    first["identity"] = {
        "scope": "First differing rank of admitted layer0 expanded helper",
        "rank": first.pop("rank"),
        "helper_sha256": HELPER_SHA,
        "builder_sha256": file_sha(__file__),
        "cpu_report_sha256": file_sha(args.cpu_report),
        "checkpoint_metadata_sha256": reader.contract["metadata_sha256"],
        "reference_weights_sha256": file_sha(
            inspect.getsourcefile(ReferenceWeights)
        ),
        "check_mla_sha256": file_sha(inspect.getsourcefile(metrics)),
        "cpu_torch": str(torch.__version__),
        "cpu_mkldnn_enabled": False,
        "cpu_threads": 8,
        "ranks": ranks,
        "expected_combined_sha256": expected_sha,
        "capture_summary_sha256": file_sha(summary_path),
        "npu_w_uv_directly_observed": False,
    }
    first["identity"]["tensor_sha256"] = {
        section: {name: tensor_sha(value) for name, value in values.items()}
        for section, values in first.items()
        if section in ("inputs", "weights", "cpu", "full_npu")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(first, stream)
    report.update(
        status="fixture_created",
        identity=first["identity"],
        fixture_sha256=file_sha(args.output),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "checkpoint",
        "captures",
        "cpu-report",
        "helper",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(
        not args.output.exists()
        and not args.output.with_suffix(".json").exists(),
        "Preserve existing output",
    )
    sys.path[:0] = [
        str(args.source_root),
        str(args.source_root / "tests/glm52"),
    ]
    global torch, ReferenceWeights, metrics
    import torch
    from check_mla import metrics

    from tilert.models.glm_5_2_ascend.reference_weights import ReferenceWeights

    report = {
        "status": "error",
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "command": [sys.executable, *sys.argv],
    }
    try:
        with (
            torch.inference_mode(),
            torch.backends.mkldnn.flags(
                enabled=False,
                deterministic=None,
                allow_tf32=None,
                fp32_precision=None,
            ),
        ):
            run(args, report)
    except Exception as error:  # noqa: BLE001 - retain a failed diagnostic receipt.
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.output.with_suffix(".json")),
            }
        )
    )
    return int(report["status"] != "fixture_created")


if __name__ == "__main__":
    raise SystemExit(main())
