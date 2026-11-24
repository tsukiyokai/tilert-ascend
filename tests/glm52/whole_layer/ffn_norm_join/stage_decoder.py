"""Freeze the original FFN comparison with immediate graph consumers."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


PARENT_DECODER = "806866e0b5ba358519202760ae6651aaaebc598ec311816237515326717d39a7"
PARENT_LIBRARY = "8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84"
BUILD_PARENT = "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"
NAMESPACE = "ffn_norm_join"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one frozen decoder seam: {old!r}")
    return source.replace(old, new, 1)


def add_consumers(source):
    source = once(source, "                    owners.append((bridge, body))\n", """                    consumer_sources = {
                        "combined": bridge[10], "routed": body[9],
                        "shared": body[19], "retained": bridge[1],
                    }
                    consumers = {name: torch.empty_like(value)
                                 for name, value in consumer_sources.items()}
                    owners.append((bridge, body, consumers))
""")
    source = once(source, "                                           weights, shared, bridge, body, epsilon, 24, 1, sync)\n", """                                           weights, shared, bridge, body, epsilon, 24, 1, sync)
                        for name, value in consumer_sources.items():
                            consumers[name].copy_(value)
""")
    source = once(source, "                            for value in (bridge[0], bridge[1], bridge[3], bridge[10], body[9], body[19]):\n", """                            for name in consumers:
                                reference["consumer." + name] = reference[name]
                            for value in (bridge[0], bridge[1], bridge[3], bridge[10],
                                          body[9], body[19], *consumers.values()):
""")
    source = once(source, "                            actual = {name: snapshot(torch, value) for name, value in actual.items()}\n", """                            actual.update({"consumer." + name: value
                                           for name, value in consumers.items()})
                            actual = {name: snapshot(torch, value) for name, value in actual.items()}
""")
    source = once(source, "passed = len(checks) == 7", "passed = len(checks) == 11")
    return once(source, '        "layers": [], "cases": [], "pass": False,\n', """        "immediate_graph_consumers": ["combined", "routed", "shared", "retained"],
        "consumer_storage": "Preallocated outside capture and retained by owner",
        "consumer_poison": "All four outputs poisoned before every replay",
        "layers": [], "cases": [], "pass": False,
""")


def check_original_functions(original, changed):
    def functions(source):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)}

    before, after = functions(original), functions(changed)
    for name in ("unpack_scale", "shared_weights", "clone_output", "observe_original"):
        if before[name] != after[name]:
            raise RuntimeError(f"Original FFN reference changed: {name}")


def stage(parent, library, library_sha256, output):
    if digest(parent / "decoder_ffn.py") != PARENT_DECODER:
        raise ValueError("Use the frozen v2 original DecoderLayer FFN gate")
    if digest(library) != library_sha256:
        raise ValueError("The candidate library differs from the supplied SHA256")
    build = json.loads(library.with_name("manifest.json").read_text())
    if (build["parent_library_sha256"] != BUILD_PARENT
            or build["artifacts"][NAMESPACE + ".so"] != library_sha256
            or build["final_outer_end_phases"] != 0
            or build["body_barriers"] != 5):
        raise ValueError("Use the original-shared expert-ready GU build")
    inputs = {}
    for line in (parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(parent / name) != expected:
            raise ValueError(f"The frozen decoder input changed: {name}")
        inputs[name] = expected
    output.mkdir(parents=True, exist_ok=False)
    for name in inputs:
        if name == "manifest.json":
            continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    for name in ("decoder_ffn.py", "cpu_preflight.py"):
        original = (parent / name).read_text()
        changed = original.replace("ffn_active_token_join", NAMESPACE)
        changed = changed.replace(PARENT_LIBRARY, library_sha256)
        if name == "decoder_ffn.py":
            changed = add_consumers(changed)
            check_original_functions(original, changed)
        compile(changed, name, "exec")
        (output / name).write_text(changed)
    launcher = (parent / "device_ffn_run.sh").read_text()
    launcher = once(launcher, parent.name, output.name)
    (output / "device_ffn_run.sh").write_text(launcher)
    shutil.copy2(library, output / (NAMESPACE + ".so"))
    shutil.copy2(__file__, output / "stage_norm_join.py")
    changed_files = {"decoder_ffn.py", "cpu_preflight.py", "device_ffn_run.sh"}
    for name, expected in inputs.items():
        if name not in changed_files | {"manifest.json"}:
            if digest(output / name) != expected:
                raise RuntimeError(f"Unrelated decoder input changed: {name}")
    identities = {
        str(path.relative_to(output)): digest(path)
        for path in output.rglob("*") if path.is_file()}
    manifest = {
        "scope": "Original Decoder norm/MLP and four immediate native graph consumers",
        "parent_frozen_sha256": digest(parent / "frozen-inputs.sha256"),
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "library_sha256": library_sha256,
        "original_math_and_inputs": "Unchanged",
        "native_graph_replay": True, "reference_graph_replay": False,
        "immediate_graph_consumers": ["combined", "routed", "shared", "retained"],
        "checks_per_case": 11,
        "attention_executed": False, "tp_collectives_executed": 0,
        "files_sha256": identities,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    identities["manifest.json"] = digest(output / "manifest.json")
    (output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"files": len(identities), "library_sha256": library_sha256,
                      "manifest_sha256": identities["manifest.json"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage(args.parent, args.library, args.library_sha256, args.output)


if __name__ == "__main__":
    main()
