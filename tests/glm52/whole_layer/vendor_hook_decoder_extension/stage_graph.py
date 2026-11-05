"""Freeze official Decoder graph replay beside the asynchronous local fixture."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {}
    frozen = args.parent / "frozen-inputs.sha256"
    for line in frozen.read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The asynchronous parent changed: {name}")
        files[name] = expected
    required = {
        "decoder_forward.py": "c802ae9371103bb75c1f8f5df39e3c9b2e49d77fd3a86ef23ad5d58781fcb207",
        "decoder_storage_forward.py": "148911466ccc7a7bc3b87e3f3870ad451a95fe3626e6823e1a7c0bb051a779c0",
        "device_forward_storage_async.py": "66237ea5b2683aca1691b5a3cf703bf2113c988730fb48720871132dcac1aa9e",
    }
    for name, expected in required.items():
        if files.get(name) != expected:
            raise RuntimeError(f"The frozen asynchronous entry changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name == "manifest.json":
            continue
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, destination)
    source = Path(__file__).parent
    for name in ("decoder_graph.py", "test_decoder_graph.py", "stage_graph.py"):
        shutil.copy2(source / name, args.output / name)
    helper = once((args.parent / "decoder_storage_forward.py").read_text(),
                  "from decoder_forward import run_decoder_forward as original_forward",
                  "from decoder_graph import run_decoder_graph as original_forward")
    helper = once(helper, "before the unchanged original asynchronous decoder",
                  "before original Decoder calls through the official ACLGraphWrapper")
    (args.output / "decoder_storage_graph.py").write_text(helper)
    driver = once((args.parent / "device_forward_storage_async.py").read_text(),
                  "from decoder_storage_forward import run_decoder_forward",
                  "from decoder_storage_graph import run_decoder_forward")
    driver = once(driver, "    args = parser.parse_args()\n",
                  '    parser.add_argument("--graph-residual-modes", nargs="+",\n'
                  '                        choices=("none", "provided"), default=["none", "provided"])\n'
                  "    args = parser.parse_args()\n"
                  "    if len(set(args.graph_residual_modes)) != len(args.graph_residual_modes):\n"
                  '        parser.error("Each initial residual branch needs one distinct capture")\n')
    driver = once(driver, 'stage("actual_local_decoder_forward", lambda: (',
                  'stage("actual_local_decoder_graph", lambda: (')
    driver = once(driver, 'str(args.output / "decoder-forward.json")',
                  'str(args.output / "decoder-graph.json")')
    (args.output / "device_graph.py").write_text(driver)
    launcher = (args.parent / "device_storage_forward_run.sh").read_text()
    launcher = launcher.replace("vendor-hook-decoder212-storage-forward-v2",
                                "vendor-hook-decoder212-graph-v1")
    launcher = launcher.replace("device_forward_storage_async.py", "device_graph.py")
    (args.output / "device_graph_run.sh").write_text(launcher)
    identities = {str(path.relative_to(args.output)): digest(path)
                  for path in args.output.rglob("*") if path.is_file()}
    report = {
        "scope": "Official ACLGraphWrapper around original DecoderLayer 6/7 with explicit local TP cuts",
        "parent": str(args.parent), "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "parent_frozen_sha256": digest(frozen),
        "original_forward_sha256": required["decoder_forward.py"],
        "official_ffn_and_nz_alias": "Unchanged",
        "gmm_hooks_and_fences": False,
        "capture_observation": "Device-side boundary clones; strong owners; poisoned before every explicit replay",
        "sequence_contract": "Actual GPU lengths change; fixed host upper bound 8192 and 64 page columns",
        "initial_residual_contract": "Separate captures for None and provided BF16 input",
        "tp16_collectives_executed": 0,
        "files_sha256": identities,
    }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    identities["manifest.json"] = digest(manifest)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"path": str(args.output), "file_count": len(identities),
                      "manifest_sha256": digest(manifest),
                      "frozen_inputs_sha256": digest(args.output / "frozen-inputs.sha256")}, indent=2))


if __name__ == "__main__":
    main()
