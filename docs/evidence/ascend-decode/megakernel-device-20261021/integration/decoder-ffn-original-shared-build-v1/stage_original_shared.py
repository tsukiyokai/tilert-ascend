"""Swap only the native complete-FFN library in the original Decoder gate."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PARENT_DECODER = "806866e0b5ba358519202760ae6651aaaebc598ec311816237515326717d39a7"
PARENT_LIBRARY = "8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84"
CANDIDATE_LIBRARY = "342d965315a52ea25c5e664d41a90031ecca36ed0d8109d7a1d80fb6b82f87c7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.parent / "decoder_ffn.py") != PARENT_DECODER:
        raise ValueError("Use the frozen v2 original DecoderLayer FFN gate")
    if digest(args.library) != CANDIDATE_LIBRARY:
        raise ValueError("The original shared-activation candidate changed")
    inputs = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise ValueError(f"The frozen decoder input changed: {name}")
        inputs[name] = expected
    args.output.mkdir(parents=True, exist_ok=False)
    for name in inputs:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    for name in ("decoder_ffn.py", "cpu_preflight.py"):
        source = (args.parent / name).read_text()
        source = source.replace("ffn_active_token_join", "ffn_original_shared")
        source = source.replace(PARENT_LIBRARY, CANDIDATE_LIBRARY)
        (args.output / name).write_text(source)
    launcher = (args.parent / "device_ffn_run.sh").read_text()
    if launcher.count(args.parent.name) != 1:
        raise ValueError("The frozen launcher workspace changed")
    (args.output / "device_ffn_run.sh").write_text(
        launcher.replace(args.parent.name, args.output.name))
    shutil.copy2(args.library, args.output / "ffn_original_shared.so")
    shutil.copy2(__file__, args.output / "stage_original_shared.py")
    changed = {"decoder_ffn.py", "cpu_preflight.py", "device_ffn_run.sh"}
    for name, expected in inputs.items():
        if name not in changed | {"manifest.json"}:
            if digest(args.output / name) != expected:
                raise RuntimeError(f"Unrelated decoder input changed: {name}")
    identities = {
        str(path.relative_to(args.output)): digest(path)
        for path in args.output.rglob("*") if path.is_file()}
    manifest = {
        "scope": "Original Decoder norm/MLP against CANN-compatible shared activation",
        "parent_frozen_sha256": digest(args.parent / "frozen-inputs.sha256"),
        "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "library_sha256": CANDIDATE_LIBRARY,
        "original_math_and_inputs": "Unchanged",
        "native_graph_replay": True, "reference_graph_replay": False,
        "attention_executed": False, "tp_collectives_executed": 0,
        "files_sha256": identities,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    identities["manifest.json"] = digest(args.output / "manifest.json")
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"files": len(identities), "library_sha256": CANDIDATE_LIBRARY,
                      "manifest_sha256": identities["manifest.json"]}))


if __name__ == "__main__":
    main()
