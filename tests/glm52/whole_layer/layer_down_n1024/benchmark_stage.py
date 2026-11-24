"""Freeze the 609 two-layer comparison with private N512/N1024 Down tilings."""

import argparse
import difflib
import json
from pathlib import Path
import re
import shutil

from record_layout import measure_layout, probe_source
from stage import digest, once, verify_build, CHAIN_MANIFEST


EAGER_MANIFEST = "228ca66512b1ad477bf2e018260d2c37d3fad6fa46958bebceb3f0a2c86ae0ea"
NATIVE_MANIFEST = "a69b6cae51cb0435b5756599772bc98d22546c9e2531a3d435dd1f4796ca990f"
CONTROL_MANIFEST = "6a8780321d41b5d834e1d268ff08c30b675bf39558367a10134c53e28bebf613"
CONTROL_SHA = "c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032"
TILING_RECORD_SHA = "85ff0e20eaac8487cb12d0fffb1299e02f43a6575a61e2783f64bded1cdebc56"


def verify_fixture(directory, expected):
    if digest(directory / "manifest.json") != expected:
        raise ValueError(f"Use the frozen fixture: {directory}")
    files = {}
    for line in (directory / "frozen-inputs.sha256").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if digest(directory / name) != sha:
            raise ValueError(f"A frozen fixture file changed: {name}")
        files[name] = sha
    return files


def fixture_source(source):
    source = once(source, '        "timing": False,\n',
                  '        "timing": True, "timings": [],\n'
                  '        "timing_scope": "609 plus TopK copy, N512 versus N1024 Down tiling",\n')
    source = once(source, "        for rows in args.forward_widths:\n",
                  "        from layer_pair_timing import run_pair_timing\n"
                  "        if args.forward_widths != [12]:\n"
                  '            raise ValueError("The paired timing fixture requires M12")\n'
                  "        retained_arenas = []\n"
                  "        for arena in range(2):\n"
                  "            rows = 12\n")
    source = once(source, '                "rows": rows, "owners": owners,\n',
                  '                "rows": rows, "arena": arena, "owners": owners,\n')
    source = once(source, '                        "rows": rows, "iteration": iteration, "arm": arm,\n',
                  '                        "rows": rows, "iteration": iteration, "arm": arm, "arena": arena,\n')
    source = once(source, '                record = {"rows": rows, "iteration": iteration,\n',
                  '                record = {"rows": rows, "iteration": iteration, "arena": arena,\n')
    source = once(source, "            del first, second, cases, inputs\n",
                  "            retained_arenas.append((first, second, inputs))\n"
                  '            report["timings"].append(run_pair_timing(\n'
                  "                torch, cases, inputs, topk, caches, bases, arena, args.output))\n"
                  "            persist()\n"
                  "            del first, second, cases, inputs\n")
    compile(source, "local_chain.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--native-parent", type=Path, required=True)
    parser.add_argument("--control-build", type=Path, required=True)
    parser.add_argument("--chain-build", type=Path, required=True)
    parser.add_argument("--tiling-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    frozen = verify_fixture(args.parent, EAGER_MANIFEST)
    verify_fixture(args.native_parent, NATIVE_MANIFEST)
    verify_build(args.control_build, CONTROL_MANIFEST)
    verify_build(args.chain_build, CHAIN_MANIFEST)
    library = args.control_build / "ffn_down_n512.so"
    if digest(library) != CONTROL_SHA or digest(args.tiling_record) != TILING_RECORD_SHA:
        raise ValueError("The control library or installed-host tiling record changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in frozen:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    shutil.copy2(library, args.output / "control_ffn.so")
    for name in ("native_components.py", "native-components.patch", "native_norm.so"):
        shutil.copy2(args.native_parent / name, args.output / name)
    shutil.copy2(args.native_parent / "layer_pair_timing.py",
                 args.output / "parent-layer-pair-timing.py")
    tools = Path(__file__).parent
    for name in ("tiling_packets.py", "record_layout.py", "test_benchmark.py"):
        shutil.copy2(tools / name, args.output / name)
    shutil.copy2(tools / "timing.py", args.output / "layer_pair_timing.py")
    for name in ("benchmark_stage.py", "TIMING.md"):
        shutil.copy2(tools / name, args.output / "n1024_tools" / name)
    for path in args.chain_build.rglob("*.h"):
        if not path.is_relative_to(args.chain_build / "candidate_tools"):
            target = args.output / "record_headers" / path.relative_to(args.chain_build)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    layout = measure_layout(args.output / "record_headers")
    (args.output / "record-layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    (args.output / "record-layout-probe.cpp").write_text(probe_source())
    tilings = json.loads(args.tiling_record.read_text())["arms"]
    expected = {}
    for arm, old, geometry in (("control", "separate", [64, 512, 128]),
                               ("candidate", "fused", [32, 1024, 64])):
        if tilings[old]["layouts"]["0"] != tilings[old]["layouts"]["1"]:
            raise ValueError("The frozen ND and NZ Down records differ")
        expected[arm] = {**tilings[old]["layouts"]["1"], "geometry": geometry,
                         "library_sha256": tilings[old]["library_sha256"]}
    (args.output / "paired-tilings.json").write_text(json.dumps(expected, indent=2) + "\n")
    shutil.copy2(args.tiling_record, args.output / "installed-host-tilings.json")
    original = (args.parent / "local_chain.py").read_text()
    candidate = fixture_source(original)
    (args.output / "local_chain.py").write_text(candidate)
    (args.output / "paired-tiling-fixture.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="eager/local_chain.py", tofile="timing/local_chain.py")))
    (args.output / "paired-timing.patch").write_text("".join(difflib.unified_diff(
        (args.native_parent / "layer_pair_timing.py").read_text().splitlines(keepends=True),
        (args.output / "layer_pair_timing.py").read_text().splitlines(keepends=True),
        fromfile="native-v3/layer_pair_timing.py", tofile="paired-tiling/layer_pair_timing.py")))
    cpu = args.output / "local_chain_cpu_check.py"
    text = once(cpu.read_text(), "    import local_chain\n",
                "    import local_chain\n"
                "    import layer_pair_timing\n"
                "    import native_components\n"
                "    from tiling_packets import load_control\n"
                '    torch.ops.load_library(str(args.decoder / "native_norm.so"))\n'
                "    control_ffn = load_control(torch)\n")
    text = once(text, '                importlib.import_module("test_contract"))]))',
                '                importlib.import_module("test_contract")),\n'
                '            unittest.defaultTestLoader.loadTestsFromModule(\n'
                '                importlib.import_module("test_benchmark"))]))')
    text = once(text, '                                               "tilert_ffn_down_n1024::")))',
                '                                               "tilert_ffn_down_n1024::",\n'
                '                                               "tilert_ffn_down_n512::",\n'
                '                                               "tilert::norm_")))')
    text = once(text, "if len(schemas) != 6", "if len(schemas) != 11")
    text = once(text, "four-library co-load", "six-library co-load")
    text = once(text, '            args.decoder / "layer_local_chain.so")},',
                '            args.decoder / "layer_local_chain.so",\n'
                '            args.decoder / "control_ffn.so", args.decoder / "native_norm.so")},')
    cpu.write_text(text)
    launcher = args.output / "device_local_chain_run.sh"
    launcher.write_text(once(launcher.read_text(), "vendor-hook-layer-down-n1024-v1",
                             args.deploy_name))
    report = json.loads((args.parent / "manifest.json").read_text())
    report.update({
        "scope": "Same-address 609 plus TopK copy with N512 or N1024 Down tiling",
        "parent_fixture_manifest_sha256": EAGER_MANIFEST,
        "native_timing_manifest_sha256": NATIVE_MANIFEST,
        "control_build_manifest_sha256": CONTROL_MANIFEST,
        "control_ffn_library_sha256": CONTROL_SHA,
        "deployment_directory": args.deploy_name,
        "timing": {"rows": 12, "arenas": 2, "weight_banks": 1, "weight_sets": 2,
                   "pair_calls_per_graph": 16, "replays_per_event": 30,
                   "alternating_pairs_per_arena": 30,
                   "unit": "us per indexed/shared two-layer pair"},
        "record_contract": "664B config equal; 928B addresses differ only in uint64 body.scratch[7] at offset624",
        "control_owner_gate": "Original c90.prepare/forward_out keep their full TensorImpl set outside graph capture",
        "native_norm_reference": "Frozen 40a9 norm_quant_out, same addresses; checked against original norm and DynamicQuant",
        "device_execution": False,
        "files_sha256": {str(path.relative_to(args.output)): digest(path)
                         for path in args.output.rglob("*") if path.is_file()},
    })
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    files = {**report["files_sha256"], "manifest.json": digest(manifest)}
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"output": str(args.output), "files": len(files),
                      "manifest_sha256": digest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
