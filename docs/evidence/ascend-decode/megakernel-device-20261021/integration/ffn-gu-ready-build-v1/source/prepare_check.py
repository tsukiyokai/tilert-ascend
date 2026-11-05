"""Check complete active FFN arithmetic and its full active-expert table."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare import replace_once


PARENT_CHECKER = "d5c1f9f3214a1461d0aff29e166ec781d894c6bd991369bbac27f80baed3c9bb"


def prepare_check(parent, output):
    content = parent.read_bytes()
    if hashlib.sha256(content).hexdigest() != PARENT_CHECKER:
        raise ValueError("Use the frozen complete-FFN harness-v1 checker")
    source = content.decode().replace("tilert_ffn_composed", "tilert_ffn_active")
    source = replace_once(source,
        "                        actual = all_values(torch, cb, cr, rows)",
        "                        actual = all_values(torch, cb, cr, rows)\n"
        '                        actual["body.active_experts"] = cr[20]')
    source = replace_once(source,
        "    def check_values(name, actual, expected):\n",
        "    def check_values(name, actual, expected):\n"
        '        counts = expected["body.counts"].cpu()\n'
        "        ids = (counts != 0).nonzero().reshape(-1).to(torch.int32)\n"
        "        active = torch.zeros(264, dtype=torch.int32)\n"
        "        active[0] = ids.numel()\n"
        "        active[8:8 + ids.numel()] = ids\n"
        '        expected = {**expected, "body.active_experts": active}\n')
    output.mkdir(parents=True, exist_ok=False)
    path = output / "check.py"
    path.write_text(source)
    report = {
        "parent_checker_sha256": PARENT_CHECKER,
        "numerical_contract": "Original 30 boundaries plus all 264 active-table INT32 values against reference counts",
        "sources": {"check.py": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare_check(args.parent.resolve(), args.output.resolve())
