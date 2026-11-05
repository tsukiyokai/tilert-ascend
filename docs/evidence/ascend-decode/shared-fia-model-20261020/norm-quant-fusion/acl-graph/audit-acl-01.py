"""Audit the two completed ACL norm/quant arms using CPU tensors only."""

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch


ROOT = Path(__file__).resolve().parent
NAMES = ("q", "scale", "residual")
INPUT_SHA = "5e59d7dc1f27f545c21ea4f9fb58efb9015ea5fc8b8ca73167c1437d384c6673"
checks = []


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def same(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8),
                            right.contiguous().view(torch.uint8)))


def metrics(left, right):
    return {"bitexact": same(left, right),
            "different": int((left != right).sum()),
            "maxabs": float((left.double() - right.double()).abs().max())}


def check(name, passed):
    checks.append({"name": name, "passed": bool(passed)})


def compiled_config(tree):
    return {node.targets[0].slice.value: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and ast.unparse(node.targets[0].value) == "compile_configs"}


def main():
    torch.set_num_threads(1)
    check("torch_npu_not_imported_before", "torch_npu" not in sys.modules)
    frozen_path = ROOT.parent / "inputs.pt"
    check("frozen_bundle_hash", sha(frozen_path) == INPUT_SHA)
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=True)
    data, reports, identities, configurations = {}, {}, {}, {}
    for arm in ("on", "off"):
        directory = ROOT / f"acl-01-{arm}"
        report = json.loads((directory / "result.json").read_text())
        reports[arm] = report
        identities[arm] = {name: sha(directory / name)
                           for name in ("result.json", "compiled-ir.py",
                                        "outputs.pt")}
        check(f"{arm}.script_hash", report["script_sha256"]
              == sha(ROOT / "check-acl.py"))
        check(f"{arm}.input_hash", report["inputs_sha256"] == INPUT_SHA)
        for filename, field in (("outputs.pt", "outputs_sha256"),
                                ("compiled-ir.py", "compiled_ir_sha256")):
            check(f"{arm}.{filename}.hash", identities[arm][filename]
                  == report["device_result"][field])
        tree = ast.parse((directory / "compiled-ir.py").read_text())
        configurations[arm] = compiled_config(tree)
        counts = Counter(ast.unparse(node.func).removeprefix("torch.ops.")
                         for node in ast.walk(tree)
                         if isinstance(node, ast.Call))
        for name, expected in {
            "npu.npu_add_rms_norm_dynamic_quant.default": int(arm == "on"),
            "_C_ascend.npu_add_rms_norm_bias.default": int(arm == "off"),
            "npu.npu_dynamic_quant.default": int(arm == "off"),
        }.items():
            check(f"{arm}.IR.{name}", counts[name] == expected)
        bundle = torch.load(directory / "outputs.pt", map_location="cpu",
                            weights_only=True)
        data[arm] = bundle
        for name in ("x", "residual", "gamma", "beta"):
            check(f"{arm}.frozen.{name}",
                  same(bundle["inputs"][name], frozen[name]))
        variants = bundle["graph_replays"]
        check(f"{arm}.variant_order", [row["input"] for row in variants]
              == ["original", "changed", "restored"])
        for index, row in enumerate(variants):
            expected_x = -frozen["x"] if index == 1 else frozen["x"]
            expected_r = (frozen["residual"].roll(1, -1) if index == 1
                          else frozen["residual"])
            check(f"{arm}.{index}.input_x", same(row["x"], expected_x))
            check(f"{arm}.{index}.input_r", same(row["residual"], expected_r))
            for field, actual, expected in zip(NAMES, row["graph"],
                                                row["expected"]):
                check(f"{arm}.{index}.expected.{field}",
                      same(actual, expected))
                check(f"{arm}.{index}.repeat.{field}",
                      same(actual, actual[:1].expand_as(actual)))
            for kind in ("expected", "eager"):
                observed = {field: metrics(left, right)
                            for field, left, right in zip(
                                NAMES, row[kind], row["graph"])}
                saved = report["device_result"]["comparisons"][index]
                check(f"{arm}.{index}.{kind}.metrics",
                      observed == saved[f"{kind}_vs_graph"])
        for offset, field in enumerate(NAMES):
            check(f"{arm}.A_restored.{field}",
                  same(variants[0]["graph"][offset],
                       variants[2]["graph"][offset]))
            check(f"{arm}.B_changed.{field}",
                  not same(variants[0]["graph"][offset],
                           variants[1]["graph"][offset]))
            check(f"{arm}.warm_matches_A.{field}",
                  same(bundle["compiled_warm"][offset],
                       variants[0]["graph"][offset]))
    check("compiled_configuration_identical",
          configurations["on"] == configurations["off"])
    on = dict(reports["on"]["validated_ascend_config"])
    off = dict(reports["off"]["validated_ascend_config"])
    check("only_fuse_norm_quant_differs", on.pop("fuse_norm_quant") is True
          and off.pop("fuse_norm_quant") is False and on == off)
    check("source_hashes_identical", reports["on"]["source_sha256"]
          == reports["off"]["source_sha256"])
    differences = []
    for index in range(3):
        left = data["on"]["graph_replays"][index]
        right = data["off"]["graph_replays"][index]
        for offset, field in enumerate(NAMES):
            check(f"cross.{index}.eager.{field}",
                  same(left["eager"][offset], right["eager"][offset]))
            check(f"cross.{index}.off_is_eager.{field}",
                  same(right["graph"][offset], right["eager"][offset]))
        observed = {field: metrics(a, b)
                    for field, a, b in zip(NAMES, left["graph"],
                                          right["graph"])}
        check(f"cross.{index}.residual_exact",
              observed["residual"]["bitexact"])
        check(f"cross.{index}.q_differs_by_one",
              observed["q"]["different"] > 0
              and observed["q"]["maxabs"] == 1)
        check(f"cross.{index}.scale_differs",
              observed["scale"]["different"] == 8)
        differences.append({"input": left["input"], **observed})
    check("torch_npu_not_imported_after", "torch_npu" not in sys.modules)
    result = {"status": "pass" if all(item["passed"] for item in checks)
              else "fail", "checks": checks, "check_count": len(checks),
              "artifacts_sha256": identities, "on_vs_off": differences,
              "compiled_config": configurations["on"],
              "script_sha256": sha(__file__), "scope":
              "CPU audit of saved tensors/IR; no device execution"}
    path = ROOT / "acl-01-independent-cpu-audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
