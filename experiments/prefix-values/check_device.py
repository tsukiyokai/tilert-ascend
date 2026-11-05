"""Replay the real rank2 M1 through original and cached-prefix Value on A3."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

FIXTURE_SHA = (
    "10c8abc1b891f4de0903cec67d33aa5b31e85829a8b44e57c75d4a8829b36f71"
)
ORIGINAL_SHA = (
    "50e2a19d621ee65d0700a20f38284a3ad83357ff090967e2ae755de500020a00"
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def tensor_sha(value):
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "fixture",
        "bank",
        "bank-receipt",
        "original",
        "cached",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    global torch
    import torch
    import torch_npu

    assert digest(args.fixture) == FIXTURE_SHA
    assert digest(args.original) == ORIGINAL_SHA
    receipt = json.loads(args.bank_receipt.read_text())
    assert receipt["rank"] == 2 and receipt["file_sha256"] == digest(args.bank)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    assert (
        receipt["prefix"]["prefix_sha256"]
        == fixture["capture_inputs_identity"]["prefix_sha256"]
    )
    assert (
        receipt["checkpoint"]["metadata_sha256"]
        == fixture["identity"]["checkpoint_metadata_sha256"]
    )
    bank = torch.load(
        args.bank, map_location="cpu", weights_only=True, mmap=True
    )
    assert bank.dtype == torch.bfloat16 and bank.shape == (78, 4, 8192, 256)
    assert torch.equal(bank[0], fixture["cpu"]["value_prefix"])
    original = module(args.original, "expanded_attention_control")
    cached = module(args.cached, "cached_prefix_attention")
    torch_npu.npu.set_device(0)
    device = torch.device("npu:0")
    before_hf32 = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    arms = {}
    try:
        with torch.inference_mode():
            inputs = {
                name: value.to(device)
                for name, value in fixture["inputs"].items()
            }
            before = {
                name: tensor_sha(value.cpu()) for name, value in inputs.items()
            }
            uv = fixture["weights"]["uv"].to(device)
            prefix = bank[0].to(device)
            impl = SimpleNamespace(
                W_UV=uv,
                scale=0.0625,
                _execute_sparse_flash_attention_process=lambda *args, **kwargs: (
                    None
                ),
                _v_up_proj=lambda value: value,
            )
            metadata = SimpleNamespace(block_table=inputs["block_table"])
            for name in ("expanded_control", "cached_prefix"):
                restore = (
                    original.install(impl)
                    if name == "expanded_control"
                    else cached.install(impl, prefix)
                )
                outcomes = []
                try:
                    for _ in range(32):
                        output = impl._execute_sparse_flash_attention_process(
                            inputs["query_nope"],
                            inputs["query_rope"],
                            (inputs["key_cache"], inputs["rope_cache"]),
                            inputs["indices"],
                            metadata,
                            inputs["query_lengths"],
                            inputs["key_lengths"],
                        )
                        value = output.cpu().clone()
                        outcomes.append(value)
                        assert torch.equal(
                            impl._v_up_proj(output).cpu(),
                            value.reshape(1, 1024),
                        )
                finally:
                    restore()
                raw = outcomes[0]
                arms[name] = {
                    "output_sha256": tensor_sha(raw),
                    "whole_pipeline_repetitions": len(outcomes),
                    "stable": all(
                        tensor_sha(raw) == tensor_sha(value)
                        for value in outcomes
                    ),
                    "matches_original_cpu_pv": torch.equal(
                        raw, fixture["cpu"]["pv"]
                    ),
                    "matches_original_device_pv": torch.equal(
                        raw, fixture["full_npu"]["pv"]
                    ),
                    "unequal_to_cpu": int((raw != fixture["cpu"]["pv"]).sum()),
                }
            after = {
                name: tensor_sha(value.cpu()) for name, value in inputs.items()
            }
            prefix_unchanged = tensor_sha(prefix.cpu()) == tensor_sha(bank[0])
    finally:
        torch.npu.matmul.allow_hf32 = before_hf32
    report = {
        "scope": "One actual A3 die; rank2 M1, fixed2048 positions, whole QK/P/Value/PV repeated32; no timing",
        "arms": arms,
        "expected_cpu_pv_sha256": tensor_sha(fixture["cpu"]["pv"]),
        "original_device_pv_sha256": tensor_sha(fixture["full_npu"]["pv"]),
        "inputs_unchanged": before == after,
        "prefix_bank_input_unchanged": prefix_unchanged,
        "fixture_sha256": digest(args.fixture),
        "bank_file_sha256": digest(args.bank),
        "bank_receipt_sha256": digest(args.bank_receipt),
        "original_sha256": digest(args.original),
        "cached_sha256": digest(args.cached),
        "script_sha256": digest(__file__),
        "device_name": torch.npu.get_device_name(0),
        "visible_devices": {
            key: os.environ.get(key)
            for key in ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
        },
        "formal_precision_gate_passed": False,
        "performance_measured": False,
    }
    passed = (
        arms["expanded_control"]["matches_original_device_pv"]
        and arms["cached_prefix"]["matches_original_cpu_pv"]
        and all(arm["stable"] for arm in arms.values())
        and before == after
        and prefix_unchanged
    )
    report["status"] = (
        "diagnostic_exact" if passed else "captured_with_differences"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
