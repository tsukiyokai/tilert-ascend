"""Freeze the actual decoder construction with a complete-FFN comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one source seam: {old}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The frozen decoder input changed: {name}")
        files[name] = expected
    if digest(args.parent / "device_forward_storage_async.py") != (
            "66237ea5b2683aca1691b5a3cf703bf2113c988730fb48720871132dcac1aa9e"):
        raise RuntimeError("Use the accepted asynchronous decoder driver")
    if digest(args.library) != "8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84":
        raise RuntimeError("Use the accepted active-token-join complete FFN")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    source = Path(__file__).parent
    for name in ("stage.py", "decoder_ffn.py", "test_scale.py", "cpu_preflight.py"):
        shutil.copy2(source / name, args.output / name)
    shutil.copy2(args.library, args.output / "ffn_active_token_join.so")
    helper = (args.parent / "decoder_storage_forward.py").read_text()
    helper = once(
        helper,
        "from decoder_forward import run_decoder_forward as original_forward",
        "from decoder_ffn import run_decoder_forward as original_forward")
    helper = once(
        helper, "One-time INT32 NZ carrier descriptors before "
        "the unchanged original asynchronous decoder",
        "One-time INT32 NZ carrier descriptors before "
        "the actual norm/MLP versus native complete-FFN comparison")
    (args.output / "decoder_storage_ffn.py").write_text(helper)
    driver = (args.parent / "device_forward_storage_async.py").read_text()
    driver = once(driver, "from decoder_storage_forward import run_decoder_forward",
                  "from decoder_storage_ffn import run_decoder_forward")
    driver = once(driver, 'stage("actual_local_decoder_forward",',
                  'stage("actual_local_complete_ffn",')
    driver = once(driver, 'str(args.output / "decoder-forward.json")',
                  'str(args.output / "decoder-ffn.json")')
    driver = once(driver, '"full_decoder_forward_executed": True,',
                  '"full_decoder_forward_executed": False, "complete_ffn_executed": True,')
    driver = once(
        driver, '"scope": "Original DecoderLayer.forward with explicit '
        'post-AR1 inputs and local pre-AR2 outputs",',
        '"scope": "Actual DecoderLayer post-attention norm and MLP versus '
        'native complete FFN, with local pre-AR2 outputs",')
    driver = once(
        driver, '"scope": "Two actual decoder layers; explicit local TP cuts; '
        'original FFN in both arms; no TP16 numerical reduction",',
        '"scope": "Two actual decoder weight sets; original post-attention '
        'norm/MLP versus complete native FFN; no TP16 numerical reduction",')
    (args.output / "device_ffn.py").write_text(driver)
    launcher = (args.parent / "device_storage_forward_run.sh").read_text()
    launcher = launcher.replace("vendor-hook-decoder212-storage-forward-v2",
                                args.output.name)
    launcher = launcher.replace("device_forward_storage_async.py", "device_ffn.py")
    (args.output / "device_ffn_run.sh").write_text(launcher)
    identities = {str(path.relative_to(args.output)): digest(path)
                  for path in args.output.rglob("*") if path.is_file()}
    report = {
        "scope": ("Actual DecoderLayer norm/MLP and corrected same-byte "
                  "NZ weights versus the complete native FFN"),
        "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "parent_frozen_sha256": digest(args.parent / "frozen-inputs.sha256"),
        "library_sha256": digest(args.library),
        "original_weight_postprocessing_and_operators": "Unchanged",
        "scale_conversion": "Inverse of the original FP32 bits to zero-extended INT64 encoding",
        "native_graph_replay": True, "reference_graph_replay": False,
        "attention_executed": False, "tp_collectives_executed": 0,
        "files_sha256": identities,
    }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    identities["manifest.json"] = digest(manifest)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"files": len(identities), "manifest_sha256": digest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
