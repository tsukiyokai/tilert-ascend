"""Freeze a same-owner two-layer graph timing comparison at M12."""

import argparse
import difflib
import json
from pathlib import Path
import re
import shutil

from graph_stage import EAGER_SHA, verify
from stage import digest, once


CONTROL_SHA = "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f"
CONTROL_HOST_SHA = "70bfd7393e7ceb2a6921100be6057f0cea3f922c52d6e3df2e02ea398c66e6f9"


def fixture_source(source):
    source = once(source, '        "timing": False,\n',
                  '        "timing": True, "timings": [],\n'
                  '        "timing_scope": "609 plus copy versus direct-owner 6a; N512 in both",\n')
    source = once(source, "        for rows in args.forward_widths:\n",
                  "        from layer_pair_timing import run_pair_timing\n"
                  "        if args.forward_widths != [12]:\n"
                  '            raise ValueError("The paired timing fixture requires M12")\n'
                  "        retained_arenas = []\n"
                  "        for arena in range(2):\n"
                  "            rows = 12\n")
    source = once(source, '                "rows": rows, "owners": owners,\n',
                  '                "rows": rows, "arena": arena, "owners": owners,\n')
    source = once(source,
                  '                        "rows": rows, "iteration": iteration, "arm": arm,\n',
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
    parser.add_argument("--control-library", type=Path, required=True)
    parser.add_argument("--control-host", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    frozen = verify(args.parent, EAGER_SHA)
    if (digest(args.control_library) != CONTROL_SHA
            or digest(args.control_host) != CONTROL_HOST_SHA):
        raise ValueError("Use the accepted 609 local layer and its Python host")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in frozen:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    shutil.copy2(args.control_library, args.output / "control_local_chain.so")
    shutil.copy2(args.control_host, args.output / "local_control_host.py")
    shutil.copy2(Path(__file__).with_name("timing.py"), args.output / "layer_pair_timing.py")
    shutil.copy2(Path(__file__), args.output / "n512_tools/benchmark_stage.py")
    original = (args.parent / "local_chain.py").read_text()
    candidate = fixture_source(original)
    (args.output / "local_chain.py").write_text(candidate)
    (args.output / "layer-timing.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="parent/local_chain.py", tofile="candidate/local_chain.py")))
    cpu = args.output / "local_chain_cpu_check.py"
    text = once(cpu.read_text(), "    import local_chain\n",
                "    import local_chain\n"
                "    import layer_pair_timing\n"
                "    from local_control_host import Library as ControlLibrary\n")
    text = once(text,
                '    chain = Library(args.decoder / "layer_local_chain.so", local_chain.CHAIN_SHA)\n',
                '    chain = Library(args.decoder / "layer_local_chain.so", local_chain.CHAIN_SHA)\n'
                '    control = ControlLibrary(args.decoder / "control_local_chain.so",\n'
                '                             layer_pair_timing.CONTROL_SHA)\n')
    text = once(text, '            args.decoder / "layer_local_chain.so")},\n',
                '            args.decoder / "layer_local_chain.so",\n'
                '            args.decoder / "control_local_chain.so")},\n')
    text = once(text, "four-library co-load", "five-library co-load")
    cpu.write_text(text)
    launcher = args.output / "device_local_chain_run.sh"
    launcher.write_text(once(launcher.read_text(), "vendor-hook-layer-n512-v1",
                             args.deploy_name))
    report = json.loads((args.parent / "manifest.json").read_text())
    report.update({
        "scope": "Same-owner 609-plus-copy versus direct-owner 6a; two N512 local layers",
        "parent_fixture_manifest_sha256": EAGER_SHA,
        "control_library_sha256": CONTROL_SHA,
        "control_host_sha256": CONTROL_HOST_SHA,
        "deployment_directory": args.deploy_name,
        "timing": {"rows": 12, "arenas": 2, "weight_banks": 1,
                   "pair_calls_per_graph": 16, "replays_per_event": 30,
                   "alternating_pairs_per_arena": 30,
                   "unit": "us per indexed/shared two-layer pair"},
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
