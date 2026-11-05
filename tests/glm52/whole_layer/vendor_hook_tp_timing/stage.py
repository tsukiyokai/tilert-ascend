"""Freeze same-address native FP32 HCCL timing beside the accepted TP graph."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


PARENT_SHA = "546f56c9bc0bb47d67d3848da7acf3d64f66409fde53784b11986aad24d27ea7"
NORM_SHA = "40a9e3c04d22355cd51228c1dff7d3dc516d3442253ee3310a6cf95c01182027"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one frozen source anchor: {old!r}")
    return source.replace(old, new)


def timing_source(source):
    start = source.index("            for rows in args.forward_widths:")
    stop = source.index("            torch.npu.synchronize()\n            torch.distributed.barrier", start)
    allocation_end = source.index("                topk_copies =", start)
    allocate = source[start:allocation_end]
    allocate = once(allocate, "for rows in args.forward_widths:", "for arena in range(2):\n                rows = 12")
    block = (
        "            arenas = []\n"
        + allocate
        + "                inputs.lock_owners()\n"
          "                arenas.append((inputs, cases))\n"
          "            private_owners = []\n"
          "            for inputs, cases in arenas:\n"
          "                tensors = [inputs.hidden, inputs.residual]\n"
          "                for case in cases:\n"
          "                    local = case.local\n"
          "                    tensors.extend([*local.norm[3:], local.scratch, local.tiling,\n"
          "                                    local.partial, *local.bridge, *local.body,\n"
          "                                    case.attention_sum, case.output])\n"
          "                private_owners.append({value.untyped_storage().data_ptr()\n"
          "                                       for value in tensors if value.numel()})\n"
          "            if private_owners[0] & private_owners[1]:\n"
          "                raise RuntimeError(\"The two compute arenas share an allocation\")\n"
          "            report[\"shared_between_arenas\"] = [\"caches\", \"persistent_topk\",\n"
          "                                                 \"source_model_weights\", \"ffn_weights\"]\n"
          "            report[\"private_arena_allocations\"] = [sorted(values) for values in private_owners]\n"
          "            for arena, (inputs, cases) in enumerate(arenas):\n"
          "                measured = run_timing(torch, config, layers, inputs, cases, caches,\n"
          "                                      bases, topk, window, stream, args, arena)\n"
          "                report[\"arenas\"].append({\"arena\": arena, \"pass\": measured[\"pass\"],\n"
          "                    \"report\": str(args.output / f\"tp-pair-timing-arena{arena}.json\"),\n"
          "                    \"report_sha256\": digest(args.output / f\"tp-pair-timing-arena{arena}.json\")})\n"
          "                persist()\n"
    )
    value = source[:start] + block + source[stop:]
    value = once(value, "from tp_parent import separate", "from tp_parent import separate\nfrom tp_timing_benchmark import run_timing")
    value = value.replace("run_tp_graph", "run_tp_timing")
    value = once(value, '"Official ACL graph of two TP16-shaped local shards with SHMEM2; eager separate parents plus HCCL2 oracle"',
                 '"Two same-address native FP32 HCCL2 versus C2 SHMEM2 graph arenas; M12, two TP16-shaped shards"')
    value = once(value, '"original_hccl_entry": "torch.distributed.all_reduce(SUM, group=actual_hccl2)",',
                 '"original_hccl_entry": "BF16.float / real HCCL2 FP32 SUM / bfloat16",')
    value = once(value, '"timing": False,', '"timing": True,')
    value = once(value, '"captures": [], "cases": [], "pass": False,', '"arenas": [], "pass": False,')
    value = once(value, '"tp-chain-graph.json"', '"tp-chain-timing.json"')
    value = once(value, '    current_graph = None\n', '')
    value = once(value, '        if current_graph is not None:\n            current_graph.close()\n', '')
    value = once(value, '    rank = args.device_rank\n',
                 '    if args.forward_widths != [12] or args.forward_replays != 8:\n'
                 '        raise ValueError("Use M12 and eight changing timing-gate inputs")\n'
                 '    rank = args.device_rank\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "norm", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deploy-name", default="vendor-hook-tp-timing-v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment directory")
    if digest(args.parent / "manifest.json") != PARENT_SHA or digest(args.norm) != NORM_SHA:
        raise RuntimeError("The accepted graph or native norm parent changed")
    files = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"A parent source changed: {name}")
        files[name] = expected
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name != "manifest.json":
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.parent / name, target)
    directory = Path(__file__).resolve().parent
    for source, target in (("native.py", "tp_timing_native.py"),
                           ("environment.py", "tp_timing_environment.py"),
                           ("benchmark.py", "tp_timing_benchmark.py"),
                           ("stage.py", "stage_tp_timing.py"),
                           ("test_timing.py", "test_tp_timing.py"),
                           ("cpu_check.py", "tp_timing_cpu_check.py")):
        shutil.copy2(directory / source, args.output / target)
    shutil.copy2(args.norm, args.output / "native_norm.so")
    shutil.copytree(args.protocol, args.output / "timing-protocol")
    parent_source = (args.parent / "tp_graph.py").read_text()
    generated = timing_source(parent_source)
    (args.output / "tp_timing_chain.py").write_text(generated)
    (args.output / "timing.patch").write_text("".join(difflib.unified_diff(
        parent_source.splitlines(keepends=True), generated.splitlines(keepends=True),
        fromfile="tp_graph.py", tofile="tp_timing_chain.py")))
    storage = once((args.parent / "decoder_storage_tp_graph.py").read_text(),
                   "from tp_graph import run_tp_graph as original_forward",
                   "from tp_timing_chain import run_tp_timing as original_forward")
    storage = storage.replace("before the two-rank component graph",
                              "before two-rank FP32 HCCL graph timing")
    (args.output / "decoder_storage_tp_timing.py").write_text(storage)
    constructor = (args.parent / "tp_graph_constructor.py").read_text()
    constructor = once(constructor, "from decoder_storage_tp_graph import run_decoder_forward",
                       "from decoder_storage_tp_timing import run_decoder_forward")
    constructor = constructor.replace("actual_two_rank_component_graph", "actual_two_rank_fp32_graph_timing")
    constructor = constructor.replace("tp-chain-graph.json", "tp-chain-timing.json")
    constructor = constructor.replace("two_rank_component_graph_executed", "two_rank_graph_timing_executed")
    (args.output / "tp_timing_constructor.py").write_text(constructor)
    device = once((args.parent / "tp_graph_device.py").read_text(),
                  "from tp_graph_constructor import run_device_rank",
                  "from tp_timing_constructor import run_device_rank")
    device = once(device, "    try:\n        import torch\n", "    try:\n"
                  "        from tp_timing_environment import check_environment\n"
                  "        report[\"collective_environment_before_torch_import\"] = check_environment()\n"
                  "        import torch\n")
    device = once(device, "    args = parser.parse_args()\n", "    args = parser.parse_args()\n"
                  "    if args.forward_widths != [12] or args.forward_replays != 8:\n"
                  "        parser.error(\"Timing requires M12 and eight changing inputs\")\n")
    device = once(device, 'parser.add_argument("--forward-replays", type=int, default=2)',
                  'parser.add_argument("--forward-replays", type=int, default=8)')
    (args.output / "tp_timing_device.py").write_text(device)
    launcher = (args.parent / "device_tp_graph_run.sh").read_text().replace(
        "vendor-hook-tp-graph-v1", args.deploy_name)
    launcher = once(launcher, "python3 /decoder/tp_graph_device.py", "python3 /decoder/tp_timing_device.py")
    launcher = once(launcher, "        GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\\n",
                    "        HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true \\\n"
                    "        GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\\n")
    (args.output / "device_tp_timing_run.sh").write_text(launcher)
    parent = json.loads((args.parent / "manifest.json").read_text())
    files = {str(path.relative_to(args.output)): digest(path)
             for path in args.output.rglob("*") if path.is_file()}
    report = {key: value for key, value in parent.items()
              if key.endswith("_sha256") and key != "parent_manifest_sha256"}
    report.update(parent_manifest_sha256=PARENT_SHA, norm_library_sha256=NORM_SHA,
                  deployment_directory=args.deploy_name,
                  scope="Same-address native FP32 HCCL2 versus C2 SHMEM2 official graph timing",
                  device_execution=False, files_sha256=files)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    files["manifest.json"] = digest(args.output / "manifest.json")
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"directory": str(args.output), "files": len(files),
                      "manifest_sha256": files["manifest.json"]}, indent=2))


if __name__ == "__main__":
    main()
