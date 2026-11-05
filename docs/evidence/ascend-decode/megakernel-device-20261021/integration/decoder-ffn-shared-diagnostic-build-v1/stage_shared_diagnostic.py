"""Freeze shared-stage observation beside the original FFN comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from stage import digest, once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The frozen complete-FFN input changed: {name}")
        files[name] = expected
    required = "806866e0b5ba358519202760ae6651aaaebc598ec311816237515326717d39a7"
    if files["decoder_ffn.py"] != required:
        raise RuntimeError("Use the original complete-FFN v2 fixture")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    for name in ("decoder_ffn_diagnostic.py", "stage_shared_diagnostic.py",
                 "test_shared_diagnostic.py"):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)
    helper = once(
        (args.parent / "decoder_storage_ffn.py").read_text(),
        "from decoder_ffn import run_decoder_forward as original_forward",
        "from decoder_ffn_diagnostic import "
        "run_decoder_forward as original_forward")
    (args.output / "decoder_storage_ffn_diagnostic.py").write_text(helper)
    driver = once(
        (args.parent / "device_ffn.py").read_text(),
        "from decoder_storage_ffn import run_decoder_forward",
        "from decoder_storage_ffn_diagnostic import run_decoder_forward")
    driver = once(
        driver, "    args = parser.parse_args()\n",
        '    parser.add_argument("--ffn-shared-parent", '
        'type=Path, required=True)\n    args = parser.parse_args()\n')
    (args.output / "device_ffn_diagnostic.py").write_text(driver)
    launcher = (args.parent / "device_ffn_run.sh").read_text()
    launcher = launcher.replace("vendor-hook-decoder212-ffn-v2",
                                args.output.name)
    launcher = launcher.replace("device_ffn.py", "device_ffn_diagnostic.py")
    (args.output / "device_ffn_diagnostic_run.sh").write_text(launcher)
    identities = {
        str(path.relative_to(args.output)): digest(path)
        for path in args.output.rglob("*") if path.is_file()}
    manifest = {
        "scope": "Observe original shared FFN and existing native payload",
        "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "parent_frozen_sha256": digest(args.parent / "frozen-inputs.sha256"),
        "original_ffn_source_sha256": required,
        "official_operators_and_native_library": "Unchanged",
        "extra_internal_synchronizations": 0, "timing": False,
        "files_sha256": identities,
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    identities["manifest.json"] = digest(path)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"files": len(identities), "manifest": digest(path)}))


if __name__ == "__main__":
    main()
