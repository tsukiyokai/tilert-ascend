"""Freeze the accepted Decoder bootstrap with a local-chain component fixture."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


PARENT_SHA = "e787cdc36624aaec311bcfcd340c98055fee1772911338689a5e70796c44125d"
HOST_SHA = "70bfd7393e7ceb2a6921100be6057f0cea3f922c52d6e3df2e02ea398c66e6f9"
CHAIN_SHA = "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f"
FFN_VIEWS_SHA = "d5c1f9f3214a1461d0aff29e166ec781d894c6bd991369bbac27f80baed3c9bb"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one frozen source anchor: {old!r}")
    return text.replace(old, new)


def transform_driver(source):
    source = once(source, "from decoder_storage_full_graph import run_decoder_forward",
                  "from decoder_storage_local_chain import run_decoder_forward")
    source = once(source, 'stage("actual_local_decoder_graph",',
                  'stage("actual_local_component_chain",')
    source = once(source, 'args.output / "decoder-graph.json"',
                  'args.output / "local-chain.json"')
    source = once(source, '"full_decoder_forward_executed": True,',
                  '"full_decoder_forward_executed": False,\n'
                  '                    "local_component_chain_executed": True,')
    source = once(source,
                  '"scope": "Original DecoderLayer.forward with explicit post-AR1 inputs and local pre-AR2 outputs",',
                  '"scope": "Actual weights and metadata; separate native components versus one local layer launch; both TP reductions identity",')
    source = once(source, 'choices=("none", "provided"), default=["none", "provided"]',
                  'choices=("none", "provided"), default=["provided"]')
    source = once(source, '    args = parser.parse_args()\n',
                  '    parser.add_argument("--chain-library", type=Path,\n'
                  '                        default=Path(__file__).with_name("layer_local_chain.so"))\n'
                  '    args = parser.parse_args()\n'
                  '    if args.graph_residual_modes != ["provided"]:\n'
                  '        parser.error("The local-chain ABI requires a provided residual")\n')
    source = once(source, 'args.sfa_library, args.ffn_library):',
                  'args.sfa_library, args.ffn_library, args.chain_library):')
    source = once(source,
                  '"scope": "Two actual decoder layers; explicit local TP cuts; native complete FFN in the native arm; no TP16 numerical reduction",',
                  '"scope": "Actual Decoder weights; separate components versus a local layer launch; no TP reduction",')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--ffn-views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", default="vendor-hook-local-chain-v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    for path, expected in ((args.parent / "manifest.json", PARENT_SHA),
                           (args.library, CHAIN_SHA), (args.host, HOST_SHA),
                           (args.ffn_views, FFN_VIEWS_SHA)):
        if digest(path) != expected:
            raise RuntimeError(f"A frozen dependency changed: {path}")
    parent = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"A parent source changed: {name}")
        parent[name] = expected
    args.output.mkdir(parents=True, exist_ok=False)
    for name in parent:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    for source, name in ((args.host, "local_chain_host.py"),
                         (args.library, "layer_local_chain.so"),
                         (args.ffn_views, "ffn_views.py")):
        shutil.copy2(source, args.output / name)
    for source, name in (("local_chain.py", "local_chain.py"),
                         ("views.py", "local_chain_views.py"),
                         ("test_local_chain.py", "test_local_chain.py"),
                         ("cpu_check.py", "local_chain_cpu_check.py"),
                         ("stage.py", "stage_local_chain.py")):
        shutil.copy2(Path(__file__).with_name(source), args.output / name)
    storage = once((args.parent / "decoder_storage_full_graph.py").read_text(),
                   "from decoder_full_graph import run_decoder_graph as original_forward",
                   "from local_chain import run_local_chain as original_forward")
    storage = once(storage,
                   "Normalize W4A8 carrier descriptors before the original asynchronous forward.",
                   "Normalize W4A8 carrier descriptors before the local component chain.")
    storage = once(storage,
                   "One-time INT32 NZ carrier descriptors before original Decoder calls through the official ACLGraphWrapper",
                   "One-time INT32 NZ carrier descriptors before the eager local component chain")
    (args.output / "decoder_storage_local_chain.py").write_text(storage)
    original = (args.parent / "device_full_graph.py").read_text()
    driver = transform_driver(original)
    compile(driver, "device_local_chain.py", "exec")
    (args.output / "device_local_chain.py").write_text(driver)
    (args.output / "driver.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), driver.splitlines(keepends=True),
        fromfile="parent/device_full_graph.py", tofile="candidate/device_local_chain.py")))
    launcher = (args.parent / "device_full_graph_run.sh").read_text().replace(
        "vendor-hook-decoder212-full-graph-v1", args.deploy_name).replace(
        "device_full_graph.py", "device_local_chain.py")
    (args.output / "device_local_chain_run.sh").write_text(launcher)
    identities = {str(path.relative_to(args.output)): digest(path)
                  for path in args.output.rglob("*") if path.is_file()}
    report = {
        "scope": "Actual Decoder6/7 weight/metadata fixture; one local launch per layer versus separate accepted components",
        "parent_manifest_sha256": PARENT_SHA,
        "deployment_directory": args.deploy_name,
        "chain_library_sha256": CHAIN_SHA, "chain_host_sha256": HOST_SHA,
        "indexed_library_sha256": "8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f",
        "shared_attention_library_sha256": "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c",
        "complete_ffn_library_sha256": "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2",
        "input_norm_reference": "Actual original input norm module; original npu_dynamic_quant",
        "tp_contract": "Two identities; no external post-AR1 input and no TP16 collective",
        "decoder_forward_called": False,
        "input_rstd": "Candidate diagnostic only",
        "full_cache_reset": True, "fresh_write_poison": True,
        "device_execution": False, "timing": False,
        "files_sha256": identities,
    }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    identities["manifest.json"] = digest(manifest)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"directory": str(args.output), "files": len(identities),
                      "manifest_sha256": digest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
