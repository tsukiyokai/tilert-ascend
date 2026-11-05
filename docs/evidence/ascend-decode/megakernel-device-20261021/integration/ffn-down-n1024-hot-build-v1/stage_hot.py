"""Freeze an original-FFN gate with exactly M rows per hot expert."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


PARENT_MANIFEST = "b45a7211c9f88f494366a8c5dda029dadab97f5229d042f886f1442833ebed70"
PARENT_DECODER = "d349a9d22bb8c73d76e113fdd0ea9b5eaad4db78b6984784210f61a8f7e4f1f9"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one frozen seam: {old!r}")
    return source.replace(old, new, 1)


def decoder_source(original):
    source = once(original, "from types import MethodType\n", """from types import MethodType

from hot_routing import (check_bias_owner, configure_hot_bias, expected_routes,
                         observe_router, poison_routes)
""")
    source = once(source, '        "blocks": 24, "tp_geometry": 16,',
                  '        "blocks": args.native_blocks, "tp_geometry": 16,')
    source = once(source, '        "down_tiling_checks": [],\n',
                  '        "hot_routing": "Original bias owner: first8 +4, remaining -4",\n'
                  '        "metadata_checks": ["route_ids", "route_probabilities",\n'
                  '                            "route_counts", "active_experts"],\n'
                  '        "down_tiling_checks": [],\n')
    source = once(source,
                  "                bias = layer.mlp.gate.e_score_correction_bias.detach()\n",
                  """                hot_bias = configure_hot_bias(torch, layer)
                bias = layer.mlp.gate.e_score_correction_bias.detach()
""")
    source = once(source, '                    "epsilon": epsilon, "router": expected,\n',
                  '                    "epsilon": epsilon, "router": expected,\n'
                  '                    "hot_bias": hot_bias,\n')
    source = once(source, "native.prepare(hidden, 24, 1)",
                  "native.prepare(hidden, args.native_blocks, 1)")
    source = once(source, "weights, shared, bridge, body, epsilon, 24, 1, sync)",
                  "weights, shared, bridge, body, epsilon, args.native_blocks, 1, sync)")
    source = once(source,
                  "                                with observe_original(torch, layer, state.get_tp_group()) as values:\n",
                  """                                with (observe_router(torch, layer) as routing,
                                      observe_original(torch, layer, state.get_tp_group()) as values):
""")
    source = once(source,
                  "                            reference = {name: snapshot(torch, value) for name, value in values.items()}\n",
                  """                            check_bias_owner(torch, layer, hot_bias["address"])
                            reference = {name: snapshot(torch, value) for name, value in values.items()}
                            route_reference, hot_case = expected_routes(torch, routing, rows)
                            reference.update(route_reference)
""")
    source = once(source, "                            graph.replay()\n",
                  "                            poison_routes(torch, bridge, body)\n"
                  "                            graph.replay()\n")
    source = once(source,
                  "                            actual.update({\"consumer.\" + name: value\n",
                  """                            actual.update({"route_ids": bridge[5],
                                           "route_probabilities": bridge[4],
                                           "route_counts": body[12],
                                           "active_experts": body[20]})
                            actual.update({"consumer." + name: value
""")
    source = once(source, "passed = len(checks) == 11 and all(",
                  "passed = hot_case[\"pass\"] and len(checks) == 15 and all(")
    source = once(source, '                                    "pass": passed, "checks": checks}',
                  '                                    "pass": passed, "checks": checks,\n'
                  '                                    "hot_route": hot_case}')
    source = once(source, '                                            "reference": reference, "native": actual}, dump)',
                  '                                            "reference": reference, "native": actual,\n'
                  '                                            "bias": bias.detach().cpu()}, dump)')
    before = {node.name: ast.dump(node, include_attributes=False)
              for node in ast.parse(original).body if isinstance(node, ast.FunctionDef)}
    after = {node.name: ast.dump(node, include_attributes=False)
             for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    for name in ("unpack_scale", "shared_weights", "clone_output",
                 "observe_original", "check_down_tiling"):
        if before[name] != after[name]:
            raise ValueError(f"Original numerical function changed: {name}")
    return source


def entry_source(source):
    source = once(source, '    parser.add_argument("--forward-widths", type=int, nargs="+", default=[1])',
                  '    parser.add_argument("--forward-widths", type=int, nargs="+", default=[17, 32])\n'
                  '    parser.add_argument("--native-blocks", type=int, choices=(16, 24), default=24)')
    source = once(source, "width not in (1, 2, 12, 32)", "width not in (17, 32)")
    return once(source, "Use widths 1/2/12/32, 1..32 replays",
                "Use widths 17/32, 1..32 replays")


def stage(parent, output):
    if (digest(parent / "manifest.json") != PARENT_MANIFEST
            or digest(parent / "decoder_ffn.py") != PARENT_DECODER):
        raise ValueError("Use the frozen original-FFN Down-N1024 gate")
    identities = {}
    for line in (parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(parent / name) != expected:
            raise ValueError(f"The frozen parent changed: {name}")
        identities[name] = expected
    output.mkdir(parents=True, exist_ok=False)
    for name in identities:
        if name == "manifest.json":
            continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    (output / "decoder_ffn.py").write_text(decoder_source(
        (parent / "decoder_ffn.py").read_text()))
    (output / "device_ffn.py").write_text(entry_source(
        (parent / "device_ffn.py").read_text()))
    (output / "device_ffn_run.sh").write_text(once(
        (parent / "device_ffn_run.sh").read_text(), parent.name, output.name))
    for name, target in (("routing.py", "hot_routing.py"),
                         ("test_hot.py", "test_hot.py"),
                         ("stage.py", "stage_hot.py")):
        shutil.copy2(Path(__file__).with_name(name), output / target)
    shutil.copy2(parent / "decoder_ffn.py", output / "parent-decoder_ffn.py")
    changed = {"decoder_ffn.py", "device_ffn.py", "device_ffn_run.sh", "manifest.json"}
    for name, expected in identities.items():
        if name not in changed and digest(output / name) != expected:
            raise ValueError(f"An unrelated frozen parent file changed: {name}")
    files = {str(path.relative_to(output)): digest(path)
             for path in output.rglob("*") if path.is_file()}
    manifest = {
        "scope": "Original norm/MLP with shared hot-routing bias; native graph replay",
        "parent_manifest_sha256": PARENT_MANIFEST,
        "library_sha256": json.loads((parent / "manifest.json").read_text())["library_sha256"],
        "widths": [17, 32], "blocks": [16, 24],
        "checks_per_case": 15, "original_math_unchanged": True,
        "changed_input": "Actual shared correction-bias storage: first8 +4, others -4",
        "required_routes": "Each token selects every expert 0..7 exactly once",
        "metadata": "Actual original IDs/probabilities; full INT32 counts[256] and active[264]",
        "fresh_write": "NaN outputs/probabilities; active fill1 then header E=0; counts zero",
        "down_tiling": "All 200 bytes after prepare and each replay",
        "files_sha256": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files["manifest.json"] = digest(output / "manifest.json")
    (output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"manifest_sha256": files["manifest.json"], "files": len(files)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage(args.parent, args.output)


if __name__ == "__main__":
    main()
