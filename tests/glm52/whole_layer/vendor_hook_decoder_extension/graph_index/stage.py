"""Freeze an observation-only overlay of the original Decoder graph fixture."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Unexpected frozen seam: {before}")
    return text.replace(before, after, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The frozen graph parent changed: {name}")
        files[name] = expected
    required = {
        "decoder_graph.py": "12bd361bdf9fd8b7890dadafaf701e260eb38124553b8e0a3d477039b2b07016",
        "device_graph.py": "efdc1209c4cf38c9c5f94607770de5d4a3cd276ce1df1f210cd63cd3be3e9e3c",
        "decoder_forward.py": "c802ae9371103bb75c1f8f5df39e3c9b2e49d77fd3a86ef23ad5d58781fcb207",
    }
    for name, expected in required.items():
        if files.get(name) != expected:
            raise RuntimeError(f"The accepted graph source changed: {name}")
    failure_sha = "e84b65134ba32de1168e285ab32ce9773e2011b68c5c2d0879bfbc1b6839905a"
    report_sha = "a47fa508986e46bbbe0ceacfc6a85e7538f381f9d254ff8217d8efb1d05a9010"
    if digest(args.failure) != failure_sha or digest(args.report) != report_sha:
        raise RuntimeError("The original failure evidence changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name == "manifest.json":
            continue
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, destination)
    source = Path(__file__).resolve().parent
    for name, output in (("diagnostic.py", "decoder_graph_index_diagnostic.py"),
                         ("test_contract.py", "test_graph_index_diagnostic.py"),
                         ("cpu_check.py", "graph_index_cpu_check.py"),
                         ("stage.py", "stage_graph_index.py")):
        shutil.copy2(source / name, args.output / output)
    for name, expected in {
            "vendor_hook_index_diagnostic.py": "382aa5cdc23dacf3617bacd4de94c7ba37e273f894863842f5351d9b63bbfbe6",
            "vendor_hook_diagnostic.py": "d797c12d06fa714f9aa4756115020b3ae822d76e8ba17497aeec3b350c5e76b3",
    }.items():
        path = source.parent.parent / name
        if digest(path) != expected:
            raise RuntimeError(f"The retained-boundary tap changed: {name}")
        shutil.copy2(path, args.output / name)
    helper = once((args.parent / "decoder_storage_graph.py").read_text(),
                  "from decoder_graph import run_decoder_graph as original_forward",
                  "from decoder_graph_index_diagnostic import run_decoder_graph as original_forward")
    (args.output / "decoder_storage_graph_index.py").write_text(helper)
    driver = once((args.parent / "device_graph.py").read_text(),
                  "from decoder_storage_graph import run_decoder_forward",
                  "from decoder_storage_graph_index import run_decoder_forward")
    driver = once(driver, 'stage("actual_local_decoder_graph", lambda: (',
                  'stage("actual_decoder_graph_index_diagnostic", lambda: (')
    driver = once(driver, '{"report": str(args.output / "decoder-graph.json")}',
                  '{"report": str(args.output / "decoder-graph-index-diagnostic.json"), '
                  '"original_graph_pass": False}')
    driver = once(driver, "    args = parser.parse_args()\n",
                  '    parser.add_argument("--graph-index-parent", type=Path, required=True)\n'
                  '    parser.add_argument("--graph-index-parent-report", type=Path, required=True)\n'
                  "    args = parser.parse_args()\n"
                  "    if (args.forward_widths != [1] or args.forward_replays != 3\n"
                  '            or args.mode != "int8" or args.graph_residual_modes != ["provided"]):\n'
                  '        parser.error("Use C8 M1, provided residual, and exactly three replays")\n'
                  "    for path, expected in (\n"
                  f"            (args.graph_index_parent, {failure_sha!r}),\n"
                  f"            (args.graph_index_parent_report, {report_sha!r})):\n"
                  "        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:\n"
                  '            parser.error(f"The original graph failure changed: {path}")\n')
    (args.output / "device_graph_index.py").write_text(driver)
    launcher = (args.parent / "device_graph_run.sh").read_text()
    launcher = launcher.replace("vendor-hook-decoder212-graph-v1", args.output.name)
    launcher = launcher.replace("device_graph.py", "device_graph_index.py")
    (args.output / "device_graph_index_run.sh").write_text(launcher)
    identities = {str(path.relative_to(args.output)): digest(path)
                  for path in args.output.rglob("*") if path.is_file()}
    manifest = {
        "scope": "Observation-only layer6 indexer taps around the original Decoder graph fixture",
        "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "parent_frozen_sha256": digest(args.parent / "frozen-inputs.sha256"),
        "unchanged_sources": required,
        "failure_sha256": failure_sha, "failure_report_sha256": report_sha,
        "required_indexed_library_sha256": "5f8868bf96efcfd5a53f17890ce8b619287d4463386a2f020c80546332edfabc",
        "required_shared_library_sha256": "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c",
        "read_point": "Original failed comparison already saved; graph objects and buffers still live",
        "perturbation": "Two vendor pre-RoPE device clones; references for other producer boundaries",
        "gmm_hooks_and_fences": False, "device_execution": False,
        "expected_failure_handling": "Only the exact original numerical RuntimeError after full reproduction and accepted live inputs; original_graph_pass remains false",
        "files_sha256": identities,
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    identities["manifest.json"] = digest(path)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"path": str(args.output), "file_count": len(identities),
                      "manifest_sha256": digest(path),
                      "frozen_inputs_sha256": digest(args.output / "frozen-inputs.sha256")}, indent=2))


if __name__ == "__main__":
    main()
