"""Freeze complete native FFNs into the original Decoder graph fixture."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


GRAPH_SHA = "12bd361bdf9fd8b7890dadafaf701e260eb38124553b8e0a3d477039b2b07016"
FFN_SHA = "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"
PACKING_SHA = "806866e0b5ba358519202760ae6651aaaebc598ec311816237515326717d39a7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"The frozen graph seam changed: {before!r}")
    return text.replace(before, after, 1)


def transform_graph(text):
    text = once(text, '    installed = [impl.forward for impl in implementations]\n',
                '    installed = [impl.forward for impl in implementations]\n'
                '    from ffn_host_adapter import LayerFFNAdapter, load as load_ffn\n'
                '    complete_ffn = load_ffn(args.ffn_library)\n'
                '    ffn_adapters = [LayerFFNAdapter(torch, torch_npu, layer, config, complete_ffn)\n'
                '                    for layer in layers]\n')
    text = once(text,
                'original FFN in both arms; native attention in one arm',
                'original attention/FFN versus native attention and complete FFN')
    text = once(text,
                '        "original_ffn_disposal": "Unmodified; no GMM hooks or diagnostic synchronization",\n',
                '        "original_ffn_disposal": "Unmodified in the vendor oracle; the native arm uses one complete FFN",\n'
                '        "ffn_library": {"path": str(args.ffn_library), "sha256": hashlib.sha256(args.ffn_library.read_bytes()).hexdigest()},\n'
                '        "ffn_adapter": source_identity(LayerFFNAdapter),\n'
                '        "ffn_workspace_contract": "Separate layer identity, exact row count and actual stream",\n'
                '        "ffn_packed_weight_pointers": [value.packed_weight_pointers for value in ffn_adapters],\n')
    text = once(text,
                '            impl.forward = forward\n\n    def collect',
                '            impl.forward = forward\n'
                '        for value in ffn_adapters:\n'
                '            if forwards is installed:\n'
                '                value.enable()\n'
                '            else:\n'
                '                value.disable()\n\n    def collect')
    text = once(text, '        pool.captured_slot_sources.clear()\n',
                '        pool.captured_slot_sources.clear()\n'
                '        for value in ffn_adapters:\n'
                '            value.pool.entries.clear()\n'
                '            value.pool.reserves.clear()\n'
                '            value.pool.claims.clear()\n')
    text = once(text,
                '                        before_hits = len(hits)\n'
                '                        counter = compilation_counter.num_cudagraph_captured\n',
                '                        before_hits = len(hits)\n'
                '                        before_ffn_hits = [len(value.hits) for value in ffn_adapters]\n'
                '                        counter = compilation_counter.num_cudagraph_captured\n')
    text = once(text, '                            reserves, slots = {}, {}\n',
                '                            reserves, slots = {}, {}\n'
                '                            ffn_reserves = []\n')
    text = once(text,
                '                                if len(reserves) != 2 or len(slots) != 3:\n',
                '                                ffn_reserves = [value.pool.reserve(inputs.hidden)\n'
                '                                                for value in ffn_adapters]\n'
                '                                if len(reserves) != 2 or len(slots) != 3:\n')
    text = once(text, '                        record.update(\n',
                '                        ffn_claimed = None\n'
                '                        ffn_calls = [value.hits[start:] for value, start\n'
                '                                     in zip(ffn_adapters, before_ffn_hits)]\n'
                '                        expected_ffn = [] if name == "vendor" else ["complete_ffn", "original_final_reduce"]\n'
                '                        if any([call["phase"] for call in calls] != expected_ffn for calls in ffn_calls):\n'
                '                            raise RuntimeError("Capture entered the wrong FFN implementation")\n'
                '                        if name == "native":\n'
                '                            if any(not calls[1]["lookup_same_runner"]\n'
                '                                   or calls[1]["layer_name"] != value.runner.layer_name\n'
                '                                   for value, calls in zip(ffn_adapters, ffn_calls)):\n'
                '                                raise RuntimeError("Capture resolved a different FFN runner")\n'
                '                            ffn_key = (inputs.hidden.device.index, stream, rows)\n'
                '                            ffn_claimed = all(value.pool.entries.get(ffn_key) is reserved\n'
                '                                              and not value.pool.reserves\n'
                '                                              for value, reserved in zip(ffn_adapters, ffn_reserves))\n'
                '                            if not ffn_claimed or ffn_reserves[0] is ffn_reserves[1]:\n'
                '                                raise RuntimeError("Capture did not claim separate layer-owned FFN buffers")\n'
                '                            for value in ffn_adapters:\n'
                '                                value.pending.require_empty()\n'
                '                        record.update(ffn_reserve_claimed=ffn_claimed,\n'
                '                                      ffn_capture_calls=ffn_calls,\n'
                '                                      ffn_reserved_bytes=[value.bytes() for value in ffn_reserves])\n'
                '                        record.update(\n')
    text = once(text,
                '                            before_hits, calls = len(hits), observation.calls\n',
                '                            before_hits, calls = len(hits), observation.calls\n'
                '                            before_ffn_hits = [len(value.hits) for value in ffn_adapters]\n')
    text = once(text,
                '                            if [value.data_ptr() for value in result] != [\n',
                '                            if before_ffn_hits != [len(value.hits) for value in ffn_adapters]:\n'
                '                                raise RuntimeError("Replay re-entered a Python FFN adapter")\n'
                '                            if [value.data_ptr() for value in result] != [\n')
    text = once(text, '        for restore in reversed(restores):\n',
                '        for value in ffn_adapters:\n'
                '            value.restore()\n'
                '        for restore in reversed(restores):\n')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--packing-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", default="vendor-hook-decoder212-full-graph-v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment directory name")
    if digest(args.parent / "decoder_graph.py") != GRAPH_SHA:
        raise ValueError("Use the frozen original-FFN Decoder graph fixture")
    if digest(args.library) != FFN_SHA:
        raise ValueError("Use the validated original-shared complete FFN")
    if digest(args.packing_helper) != PACKING_SHA:
        raise ValueError("Use the frozen actual-FFN scale and weight packing helper")
    sources = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The frozen parent changed: {name}")
        sources[name] = expected
    args.output.mkdir(parents=True, exist_ok=False)
    for name in sources:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    original = (args.parent / "decoder_graph.py").read_text()
    graph = transform_graph(original)
    compile(graph, "decoder_full_graph.py", "exec")
    (args.output / "decoder_full_graph.py").write_text(graph)
    (args.output / "graph.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), graph.splitlines(keepends=True),
        fromfile="parent/decoder_graph.py", tofile="candidate/decoder_full_graph.py")))
    helper = once((args.parent / "decoder_storage_graph.py").read_text(),
                  "from decoder_graph import run_decoder_graph as original_forward",
                  "from decoder_full_graph import run_decoder_graph as original_forward")
    (args.output / "decoder_storage_full_graph.py").write_text(helper)
    driver = once((args.parent / "device_graph.py").read_text(),
                  "from decoder_storage_graph import run_decoder_forward",
                  "from decoder_storage_full_graph import run_decoder_forward")
    driver = once(driver, "    args = parser.parse_args()\n",
                  '    parser.add_argument("--ffn-library", type=Path,\n'
                  '                        default=Path(__file__).with_name("complete_ffn.so"))\n'
                  "    args = parser.parse_args()\n")
    driver = once(driver,
                  "for path in (args.adapter, args.indexed_library, args.shared_library, args.sfa_library):",
                  "for path in (args.adapter, args.indexed_library, args.shared_library, args.sfa_library, args.ffn_library):")
    driver = driver.replace("original FFN in both arms", "native complete FFN in the native arm")
    (args.output / "device_full_graph.py").write_text(driver)
    launcher = (args.parent / "device_graph_run.sh").read_text().replace(
        "vendor-hook-decoder212-graph-v1", args.deploy_name).replace(
        "device_graph.py", "device_full_graph.py")
    (args.output / "device_full_graph_run.sh").write_text(launcher)
    for name in ("ffn_host_adapter.py", "test_ffn_host_adapter.py",
                 "test_ffn_runner_protocol.py", "full_graph_cpu_check.py",
                 "stage_full_graph.py"):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)
    shutil.copy2(args.packing_helper, args.output / "decoder_ffn.py")
    shutil.copy2(args.library, args.output / "complete_ffn.so")
    identities = {str(path.relative_to(args.output)): digest(path)
                  for path in args.output.rglob("*") if path.is_file()}
    report = {
        "scope": "Actual Decoder layers 6/7: original attention/FFN versus native attention and complete FFN",
        "parent_graph_manifest_sha256": digest(args.parent / "manifest.json"),
        "deployment_directory": args.deploy_name,
        "indexed_library_sha256": "8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f",
        "shared_attention_library_sha256": "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c",
        "actual_decoder_class_sha256": "96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d",
        "library_sha256": FFN_SHA,
        "packing_helper_sha256": digest(args.packing_helper),
        "actual_final_reduction": "Original AscendMoERunner._maybe_reduce_final_output and _maybe_add_zero_expert_output",
        "original_moe_lookup": "One original _encode_layer_name, _resolve_layer_name and get_layer_from_name call in that order; returned object must be the same runner",
        "lookup_source_sha256": "20b69fe514bebd6723a1378f89a564c8fec4144c30244101f629c3f32c188183",
        "workspace_key": "Separate layer owner, NPU device, exact M and actual stream",
        "device_execution": False, "tp16_collectives_executed": 0,
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
