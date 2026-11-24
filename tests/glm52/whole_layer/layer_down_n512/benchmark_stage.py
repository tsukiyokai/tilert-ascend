"""Freeze a same-owner two-layer graph timing comparison at M12."""

import argparse
import ast
import difflib
import json
from pathlib import Path
import re
import shutil
import textwrap

from graph_stage import EAGER_SHA, verify
from stage import digest, once


CONTROL_SHA = "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f"
CONTROL_HOST_SHA = "70bfd7393e7ceb2a6921100be6057f0cea3f922c52d6e3df2e02ea398c66e6f9"
NORM_SHA = "40a9e3c04d22355cd51228c1dff7d3dc516d3442253ee3310a6cf95c01182027"


def native_source(source):
    tree = ast.parse(source)
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "LayerCase")
    node = next(node for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == "separate")
    original = textwrap.dedent("".join(source.splitlines(keepends=True)[
        node.lineno - 1:node.end_lineno]))
    prefix = '''    normalized, retained = self.input_norm(self.norm[0], self.norm[1])
    self.norm[3].copy_(normalized)
    self.norm[4].copy_(retained)
    codes, scales = self.torch_npu.npu_dynamic_quant(self.norm[3])
    self.norm[6].copy_(codes)
    self.norm[7].copy_(scales.view(self.rows, 1))
'''
    replacement = '''    self.torch.ops.tilert.norm_quant_out(
        *self.norm[:3], None, self.norm[3], self.norm[4],
        self.norm[5].view(self.rows, 1), *self.norm[6:8],
        self.input_norm.variance_epsilon, BLOCKS, self.sync)
'''
    candidate = once(original, prefix, replacement)
    header = '''"""Same-owner native stages with a separately validated norm/quant launch."""

import hashlib
from pathlib import Path

from local_chain import BLOCKS


def prepare(torch, cases):
    path = Path(__file__).with_name("native_norm.so")
    if hashlib.sha256(path.read_bytes()).hexdigest() != NORM_SHA:
        raise RuntimeError("The frozen norm library changed")
    torch.ops.load_library(str(path))
    for case in cases:
        blocks, sync = torch.ops.tilert.norm_prepare(case.norm[0], BLOCKS)
        if blocks != BLOCKS or sync != case.sync:
            raise RuntimeError("The native norm and layer mixed-core geometry differs")


'''
    complete = header + f"NORM_SHA = {NORM_SHA!r}\n\n\n" + candidate + "\n"
    compile(complete, "native_components.py", "exec")
    return complete, "".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="original-norm/LayerCase.separate", tofile="native-norm/separate"))


def fixture_source(source, comparison="topk-owner"):
    scope = {
        "topk-owner": "609 plus copy versus direct-owner 6a; N512 in both",
        "whole-layer": "Original norm with owner copies plus native attention/FFN versus 609 whole layer",
        "native-stages": "Native norm/attention/FFN versus 609 whole layer; N512 and top-k copy in both",
    }[comparison]
    source = once(source, '        "timing": False,\n',
                  '        "timing": True, "timings": [],\n'
                  f'        "timing_scope": {scope!r},\n')
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
                  "                torch, cases, inputs, topk, caches, bases, arena, args.output,\n"
                  f"                comparison={comparison!r}))\n"
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
    parser.add_argument("--comparison", choices=("topk-owner", "whole-layer", "native-stages"),
                        default="topk-owner")
    parser.add_argument("--norm-library", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    frozen = verify(args.parent, EAGER_SHA)
    if (digest(args.control_library) != CONTROL_SHA
            or digest(args.control_host) != CONTROL_HOST_SHA):
        raise ValueError("Use the accepted 609 local layer and its Python host")
    if args.comparison == "native-stages":
        if args.norm_library is None or digest(args.norm_library) != NORM_SHA:
            raise ValueError("Use the recorded Torch212 native norm library")
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
    if args.comparison == "native-stages":
        native, patch = native_source(original)
        (args.output / "native_components.py").write_text(native)
        (args.output / "native-components.patch").write_text(patch)
        shutil.copy2(args.norm_library, args.output / "native_norm.so")
    candidate = fixture_source(original, args.comparison)
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
    if args.comparison == "native-stages":
        text = once(text, "    import layer_pair_timing\n",
                    "    import layer_pair_timing\n"
                    "    import native_components\n"
                    '    torch.ops.load_library(str(args.decoder / "native_norm.so"))\n')
        text = once(text, '            args.decoder / "control_local_chain.so")},\n',
                    '            args.decoder / "control_local_chain.so",\n'
                    '            args.decoder / "native_norm.so")},\n')
        text = once(text, "five-library co-load", "six-library co-load")
        text = once(text, '                                               "tilert_ffn_down_n512::")))',
                    '                                               "tilert_ffn_down_n512::",\n'
                    '                                               "tilert::norm_")))')
        text = once(text, "if len(schemas) != 6", "if len(schemas) != 9")
    cpu.write_text(text)
    launcher = args.output / "device_local_chain_run.sh"
    launcher.write_text(once(launcher.read_text(), "vendor-hook-layer-n512-v1",
                             args.deploy_name))
    report = json.loads((args.parent / "manifest.json").read_text())
    report.update({
        "scope": "Same-owner two-layer timing with N512",
        "comparison": args.comparison,
        "parent_fixture_manifest_sha256": EAGER_SHA,
        "control_library_sha256": CONTROL_SHA,
        "control_host_sha256": CONTROL_HOST_SHA,
        "norm_library_sha256": NORM_SHA if args.comparison == "native-stages" else None,
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
