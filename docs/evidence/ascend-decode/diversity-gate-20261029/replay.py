"""Replay the same one-ms constant-output fixture through both gates."""

import ast
import gzip
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "tests/glm52"))


def main():
    import bench_pd_decode as bench

    baseline = gzip.decompress(
        (HERE / "baseline-bench_pd_decode.py.gz").read_bytes()
    )
    assert hashlib.sha256(baseline).hexdigest() == (
        "3f83d22eb66129312b4df5396906c1afe530bf683d64eb05459f23f05af9fa17"
    )
    tree = ast.parse(baseline)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "passes_target"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    code = compile(module, "baseline-passes_target", "exec")
    exec(code, namespace)  # noqa: S102 - Only the pinned baseline function.
    with tempfile.TemporaryDirectory(prefix="trt-diversity-replay-") as folder:
        directory = Path(folder)
        with tarfile.open(HERE / "fixture.tar.gz") as archive:
            for member in archive:
                assert (
                    member.isfile() and Path(member.name).name == member.name
                )
                (directory / member.name).write_bytes(
                    archive.extractfile(member).read()
                )
        summary = json.loads((directory / "fixture-summary.json").read_text())
        measured = [
            json.loads((directory / name).read_text())
            for name in summary["requests"]
        ]
        old = namespace["passes_target"](summary)
        new = bench.passes_target(summary, measured, directory)
        assert old is True and new is False
        result = {
            "old_passes_target": old,
            "new_passes_target": new,
            "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
            "current_sha256": hashlib.sha256(
                Path(bench.__file__).read_bytes()
            ).hexdigest(),
            "statistics": bench.output_diversity(
                measured[0]["client_token_ids"]
            ),
            "torch_imported": "torch" in sys.modules,
        }
        assert result["torch_imported"] is False
        print(json.dumps(result))


if __name__ == "__main__":
    main()
