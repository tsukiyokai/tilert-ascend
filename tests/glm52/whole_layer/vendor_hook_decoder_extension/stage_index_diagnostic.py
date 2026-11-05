"""Freeze a full-decoder indexer tap beside the accepted asynchronous fixture."""

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
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
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
            raise RuntimeError(f"The accepted parent identity changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name == "manifest.json":
            continue
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, destination)
    source = Path(__file__).parent
    for name in ("decoder_index_diagnostic.py", "audit_decoder_index.py",
                 "test_decoder_index_diagnostic.py", "stage_index_diagnostic.py"):
        shutil.copy2(source / name, args.output / name)
    reference = args.runtime / "tests/glm52/whole_layer"
    for name, expected in {
            "vendor_hook_index_diagnostic.py": "382aa5cdc23dacf3617bacd4de94c7ba37e273f894863842f5351d9b63bbfbe6",
            "vendor_hook_diagnostic.py": "d797c12d06fa714f9aa4756115020b3ae822d76e8ba17497aeec3b350c5e76b3",
    }.items():
        if digest(reference / name) != expected:
            raise RuntimeError(f"The accepted retained-boundary tap changed: {name}")
        shutil.copy2(reference / name, args.output / name)
    helper = once((args.parent / "decoder_storage_forward.py").read_text(),
                  "from decoder_forward import run_decoder_forward as original_forward",
                  "from decoder_index_diagnostic import run_decoder_forward as original_forward")
    helper = once(helper, "before the unchanged original asynchronous decoder",
                  "before unchanged original asynchronous decoder calls with full-indexer taps")
    (args.output / "decoder_storage_index_diagnostic.py").write_text(helper)
    driver = once((args.parent / "device_forward_storage_async.py").read_text(),
                  "from decoder_storage_forward import run_decoder_forward",
                  "from decoder_storage_index_diagnostic import run_decoder_forward")
    driver = once(driver, "    args = parser.parse_args()\n",
                  '    parser.add_argument("--index-diagnostic-parent", type=Path, required=True)\n'
                  "    args = parser.parse_args()\n"
                  "    if not args.index_diagnostic_parent.is_file():\n"
                  '        parser.error("The retained full-decoder failure fixture is required")\n'
                  "    if hashlib.sha256(args.index_diagnostic_parent.read_bytes()).hexdigest() != "
                  + repr(digest(args.failure)) + ":\n"
                  '        parser.error("The retained failure identity changed")\n')
    (args.output / "device_forward_index_diagnostic.py").write_text(driver)
    launcher = (args.parent / "device_storage_forward_run.sh").read_text()
    launcher = launcher.replace("vendor-hook-decoder212-storage-forward-v2",
                                "vendor-hook-decoder212-index-diagnostic-v1")
    launcher = launcher.replace("device_forward_storage_async.py",
                                "device_forward_index_diagnostic.py")
    (args.output / "device_index_diagnostic_run.sh").write_text(launcher)
    identities = {str(path.relative_to(args.output)): digest(path)
                  for path in args.output.rglob("*") if path.is_file()}
    report = {
        "scope": "Original asynchronous decoder 6/7 and NZ alias with layer6 full-indexer observation",
        "parent": str(args.parent), "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "parent_frozen_sha256": digest(frozen),
        "original_forward_sha256": required["decoder_forward.py"],
        "failure_path": str(args.failure), "failure_sha256": digest(args.failure),
        "shared_layer": "Execution retained; only the full-indexer instrumentation bypasses shared mode",
        "changes": ["Observation helper import", "Required frozen parent fixture argument",
                    "Two pre-RoPE device clones and retained tensor references"],
        "official_operators_and_dsos": "Unchanged", "gmm_hooks_and_fences": False,
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
