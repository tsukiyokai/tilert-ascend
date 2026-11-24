"""Prepare a byte-exact comparison with the frozen one-launch parent."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

from prepare import LAYER, replace_once


PARENT_CHECK = "41d909ff3bd2803aa86634b70f7894373311161b299156435775c01028b6d707"


def transform(source):
    start = source.index('    parser.add_argument("--attention-library"')
    end = source.index("    return arguments(parser)", start)
    source = (source[:start]
              + '    parser.add_argument("--control-library", type=Path, required=True)\n'
              + '    parser.set_defaults(no_timing=True, atol=0.0, rtol=0.0)\n'
              + source[end:])
    source = source.replace("torch.ops.tilert.shared_full_attention_prepare(",
                            "torch.ops.tilert_shared_descriptor.shared_full_attention_prepare(")
    source = source.replace("torch.ops.tilert.shared_full_attention_out(",
                            "torch.ops.tilert_shared_descriptor.shared_full_attention_out(")
    source = replace_once(source,
        "    latent = empty((rows, 4, 512), torch.bfloat16)\n", "")
    start = source.index("    attention_scratch, attention_blocks, attention_sync =")
    end = source.index("    core_weights =", start)
    source = (source[:start] +
        "    separate_scratch, separate_tiling, separate_blocks, separate_sync = "
        "torch.ops.tilert.shared_full_attention_prepare(\n"
        "        x, pages, table_pages, blocks,\n"
        "    )\n"
        "    if separate_blocks != blocks:\n"
        '        raise RuntimeError("Both one-launch arms need the same core count")\n'
        + source[end:])
    start = source.index("    def separate():")
    end = source.index("    def update(replay):", start)
    source = (source[:start] + """    def separate():
        return torch.ops.tilert.shared_full_attention_out(
            x, x_scale, *core_weights, sin, cos, slots, separate_cache, separate_rope,
            topk, query_lengths, kv_lengths, page_table,
            value_tiles, output_tiles, output_scale, bias,
            separate_scratch, separate_tiling, separate_output, scale, epsilon, epsilon,
            blocks, separate_sync,
        )

""" + source[end:])
    source = replace_once(source,
        "            update(replay)\n            if args.capture:",
        "            update(replay)\n"
        '            full_output.fill_(float("nan"))\n'
        '            separate_output.fill_(float("nan"))\n'
        "            for scratch in (full_scratch, separate_scratch):\n"
        "                scratch[:rows * 4 * 512 * 2].view(torch.bfloat16).fill_(float('nan'))\n"
        "            if args.capture:")
    source = replace_once(source,
        "        full_latent = full_latent.view(torch.bfloat16).reshape(rows, 4, 512)\n",
        "        full_latent = full_latent.view(torch.bfloat16).reshape(rows, 4, 512)\n"
        "        latent = separate_scratch[:rows * 4 * 512 * 2].cpu()\n"
        "        latent = latent.view(torch.bfloat16).reshape(rows, 4, 512)\n")
    source = replace_once(source,
        'result["pass"] = all(value["pass"] for value in result.values() if isinstance(value, dict))',
        'result["pass"] = all(value["pass"] and value["bitwise_equal"] '
        'for value in result.values() if isinstance(value, dict))')
    source = replace_once(source,
        "    separate_bytes = (attention_scratch.numel() + tail_scratch.numel() +\n"
        "                      latent.numel() * latent.element_size() + tail_tiling.numel())",
        "    separate_bytes = separate_scratch.numel() + separate_tiling.numel()")
    start = source.index('    if report["pass"] and args.capture and not args.no_timing:')
    end = source.index("    return report", start)
    source = source[:start] + source[end:]
    source = source.replace("(args.library, args.attention_library, args.tail_library)",
                            "(args.library, args.control_library)")
    source = source.replace("(args.attention_library, args.tail_library)",
                            "(args.control_library,)")
    return source.replace(
        "one native shared-indexer attention launch versus separate native attention/tail; before TP reduction",
        "shared-attention descriptor versus frozen one-launch parent; before TP reduction")


def prepare(build, output):
    parent = build / "source" / LAYER
    original = (parent / "attention_full_check.py").read_bytes()
    if hashlib.sha256(original).hexdigest() != PARENT_CHECK:
        raise ValueError("Unexpected parent full-attention checker")
    output.mkdir(parents=True, exist_ok=False)
    candidate = transform(original.decode())
    (output / "check.py").write_text(candidate)
    for name in ("attention_check.py", "attention_provenance.py"):
        shutil.copy2(parent / name, output / name)
    shutil.copy2(Path(__file__), output / "prepare_check.py")
    (output / "checker.patch").write_text("".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="parent/check.py", tofile="candidate/check.py")))
    manifest = {
        "parent_checker_sha256": PARENT_CHECK,
        "scope": "Numerical comparisons only; separate parent and candidate arenas",
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(output.iterdir()) if path.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.build, args.output)
