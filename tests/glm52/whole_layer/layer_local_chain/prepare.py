"""Freeze verified attention and FFN sources for one local layer launch."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ATTENTION_SHA = "8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f"
FFN_SHA = "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"
FFN_NAME = "ffn_original_shared_gu_static.so"
TOOLS = ("contract.h", "kernel.cpp", "prepare.py", "build.py", "host.cpp",
         "host_metadata.h", "host.py", "host_cpu_check.py",
         "contract_test.cpp", "README.md")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Frozen source or library changed: {path}")


def prepare(attention, ffn, output):
    attention_manifest = json.loads((attention / "manifest.json").read_text())
    ffn_manifest = json.loads((ffn / "manifest.json").read_text())
    if (attention_manifest["library_sha256"] != ATTENTION_SHA
            or ffn_manifest["artifacts"][FFN_NAME] != FFN_SHA):
        raise ValueError("Use the verified 8deff750 attention and 381febf9 FFN")
    verify(attention / "whole_layer_indexed_attention.so", ATTENTION_SHA)
    verify(ffn / FFN_NAME, FFN_SHA)
    for relative, expected in attention_manifest["source_hashes"].items():
        verify(attention / "source" / relative, expected)
    for relative, expected in ffn_manifest["sources"].items():
        verify(ffn / relative, expected)
    attention_tail = (attention / "source/tests/glm52/mla_tail/mla_tail_native.h").read_text()
    ffn_tail = (ffn / "body/source/mla_tail/mla_tail_native.h").read_text()
    combined = ("        DequantizeOutputRows<ScaleOrder::CombinedScales>(\n"
                "            pipe, accumulators, output_scale, scales, bias,")
    default = "        DequantizeOutputRows(pipe, accumulators, output_scale, scales, bias,"
    if attention_tail.count(combined) != 1 or attention_tail.replace(combined, default) != ffn_tail:
        raise ValueError("The two frozen MLA-tail definitions differ beyond standalone scale order")
    output.mkdir(parents=True, exist_ok=False)
    for root, prefix, identities in (
            (attention / "source", "attention/source",
             attention_manifest["source_hashes"]),
            (ffn, "ffn", ffn_manifest["sources"])):
        for relative in identities:
            target = output / prefix / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
    own = output / "source"
    own.mkdir()
    for name in TOOLS:
        shutil.copy2(Path(__file__).with_name(name), own / name)
    shutil.copy2(attention / "manifest.json", output / "attention-manifest.json")
    shutil.copy2(attention / "parent-provenance.json",
                 output / "attention-provenance.json")
    shutil.copy2(ffn / "manifest.json", output / "ffn-manifest.json")
    shutil.copy2(ffn / "commands.jsonl", output / "ffn-commands.jsonl")
    report = {
        "scope": "Input AddRMSNorm/quant, attention and complete FFN in one NZ local-shard launch; both TP cuts are identity",
        "attention_library_sha256": ATTENTION_SHA,
        "ffn_library_sha256": FFN_SHA,
        "residual_entry": "A separate BF16 residual is required",
        "input_norm": "The frozen complete FFN's original vendor AddRMSNorm and dynamic quantization",
        "output": "bridge[10] local FFN partial and bridge[1] retained residual",
        "topk": "Indexed output stays in its attention arena; shared input is an explicit retained owner",
        "source_include_order": "Attention MLA tail precedes FFN; only its standalone Forward scale-order instantiation differs",
        "device_execution": False,
        "sources": {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attention", type=Path)
    parser.add_argument("ffn", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = prepare(args.attention.resolve(), args.ffn.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output), "sources": len(result["sources"])}))
