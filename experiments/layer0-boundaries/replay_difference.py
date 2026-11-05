"""Replay one frozen expanded-attention rank on A3 logical0; no timing."""

import argparse
import hashlib
import inspect
import json
import os
import sys
from importlib.metadata import version
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


def compare(actual, expected):
    require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        "Comparison shape/dtype differs",
    )
    return {
        "exact": torch.equal(actual, expected),
        **metrics(actual, expected),
    }


def run(args, report):
    require(
        file_sha(args.helper) == HELPER_SHA, "Expanded helper source differs"
    )
    receipt_path = args.fixture.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    require(
        receipt["status"] == "fixture_created"
        and receipt["fixture_sha256"] == file_sha(args.fixture),
        "Fixture builder receipt/SHA differs",
    )
    frozen = torch.load(args.fixture, map_location="cpu", weights_only=True)
    require(
        frozen["identity"] == receipt["identity"]
        and frozen["identity"]["helper_sha256"] == HELPER_SHA,
        "Fixture identity differs",
    )
    require(
        file_sha(inspect.getsourcefile(metrics))
        == frozen["identity"]["check_mla_sha256"],
        "Replay metrics source differs from the independent CPU fixture",
    )
    for section, values in frozen["identity"]["tensor_sha256"].items():
        for name, sha in values.items():
            require(
                tensor_sha(frozen[section][name]) == sha,
                "Fixture tensor SHA differs",
            )
    require(
        not any(
            name.startswith("transformers.models.") for name in sys.modules
        ),
        "HF model import forbidden",
    )
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    original_hf32 = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    raw = {}
    try:
        inputs = {
            name: value.to(device) for name, value in frozen["inputs"].items()
        }
        uv = frozen["weights"]["uv"].to(device)
        ql_nope, q_pe = inputs["query_nope"], inputs["query_rope"]
        positions = torch.arange(8193, device=ql_nope.device)
        pages = (
            inputs["block_table"][0].long().index_select(0, positions // 128)
        )
        physical = pages * 128 + positions % 128
        latent = inputs["key_cache"].view(-1, 512).index_select(0, physical)
        rope = inputs["rope_cache"].view(-1, 64).index_select(0, physical)
        indices = inputs["indices"].reshape(-1).long()
        if indices.numel() != 2048:
            raise ValueError("Expected the actual 2048 selected positions")
        keys = latent.index_select(0, indices)
        key_rope = rope.index_select(0, indices)
        scores = ql_nope[0].float() @ keys.float().T
        scores = scores + q_pe[0].float() @ key_rope.float().T
        scores = (scores * 0.0625).to(torch.bfloat16)
        probabilities = torch.softmax(scores.float(), dim=-1).bfloat16()
        aligned = torch.zeros(
            (4, 8193), dtype=torch.bfloat16, device=ql_nope.device
        )
        aligned.scatter_(1, indices.expand(4, -1), probabilities)
        torch.npu.synchronize()
        shared = {
            "scores_bf16": scores.cpu(),
            "probability_bf16": probabilities.cpu(),
            "aligned_probability": aligned.cpu(),
        }
        raw["shared"] = shared
        report["shared_vs_independent_cpu"] = {
            name: compare(value, frozen["cpu"][name])
            for name, value in shared.items()
        }
        report["arms"] = {}
        for arm in ("bf16", "fp32"):
            prefix_input = (
                latent[:8192].unsqueeze(0).expand(4, -1, -1).contiguous()
            )
            current_input = (
                latent[8192:].unsqueeze(0).expand(4, -1, -1).contiguous()
            )
            if arm == "bf16":
                prefix = torch.bmm(prefix_input, uv)
                current = torch.bmm(current_input, uv)
            else:
                prefix = torch.bmm(prefix_input.float(), uv.float()).bfloat16()
                current = torch.bmm(
                    current_input.float(), uv.float()
                ).bfloat16()
            values = torch.cat((prefix, current), dim=1)
            torch.npu.synchronize()
            arm_raw = {
                "value_prefix": prefix.cpu(),
                "value_current": current.cpu(),
                "pv_replays": [],
            }
            hashes = []
            for _ in range(32):
                output = torch.bmm(aligned.unsqueeze(1), values).transpose(
                    0, 1
                )
                torch.npu.synchronize()
                cpu = output.cpu().contiguous()
                arm_raw["pv_replays"].append(cpu)
                hashes.append(tensor_sha(cpu))
            raw[arm] = arm_raw
            checks = {
                name: compare(arm_raw[name], frozen["cpu"][name])
                for name in ("value_prefix", "value_current")
            }
            pv_cpu = [
                compare(value, frozen["cpu"]["pv"])
                for value in arm_raw["pv_replays"]
            ]
            pv_full = [
                compare(value, frozen["full_npu"]["pv"])
                for value in arm_raw["pv_replays"]
            ]
            report["arms"][arm] = {
                "value_vs_independent_cpu": checks,
                "pv_vs_independent_cpu": pv_cpu,
                "pv_vs_original_tp16": pv_full,
                "pv_sha256": hashes,
                "pv_rawbytes_stable": all(sha == hashes[0] for sha in hashes),
                "all_pv_match_original_tp16": all(
                    item["exact"] for item in pv_full
                ),
                "arithmetic_exact": all(
                    item["exact"] for item in checks.values()
                )
                and all(item["exact"] for item in pv_cpu)
                and all(
                    item["exact"]
                    for item in report["shared_vs_independent_cpu"].values()
                ),
            }
    finally:
        torch.npu.matmul.allow_hf32 = original_hf32
    with (args.output_dir / "outputs.pt").open("xb") as stream:
        torch.save(raw, stream)
    report.update(
        identity=frozen["identity"],
        fixture_sha256=file_sha(args.fixture),
        fixture_receipt_sha256=file_sha(receipt_path),
        helper_sha256=HELPER_SHA,
        replay_source_sha256=file_sha(__file__),
        outputs_sha256=file_sha(args.output_dir / "outputs.pt"),
        repetition_scope="32 PV calls per arm; QK/softmax fixed once; V expanded once per arm",
        matmul_allow_hf32_during_replay=False,
        matmul_allow_hf32_restored=original_hf32,
        device=str(device),
        device_name=torch.npu.get_device_name(0),
        packages={name: version(name) for name in ("torch", "torch-npu")},
        visible_devices={
            name: os.environ.get(name)
            for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")
        },
    )
    exact = all(
        arm["arithmetic_exact"] and arm["pv_rawbytes_stable"]
        for arm in report["arms"].values()
    )
    report["arithmetic_exact"] = exact
    report["status"] = (
        "diagnostic_exact" if exact else "captured_with_differences"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "fixture", "helper", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [
        str(args.source_root),
        str(args.source_root / "tests/glm52"),
    ]
    global torch, metrics
    import torch
    import torch_npu  # noqa: F401 - explicit device backend registration.
    from check_mla import metrics

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
    except Exception as error:  # noqa: BLE001 - retain the device failure receipt.
        report["error"] = f"{type(error).__name__}: {error}"
    with (args.output_dir / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.output_dir / "report.json"),
            }
        )
    )
    return int(report["status"] != "diagnostic_exact")


if __name__ == "__main__":
    raise SystemExit(main())
