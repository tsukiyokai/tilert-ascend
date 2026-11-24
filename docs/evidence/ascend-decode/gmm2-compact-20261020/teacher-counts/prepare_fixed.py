"""Apply the production temporary-buffer declaration to the paired test ELF."""

import difflib
import hashlib
import json
from pathlib import Path
import shutil

root = Path("/workspace/experiments/gmm2-compact-20261020")
original = root / "retile-paired-build"
out = root / "teacher-counts-screen/fixed-wide-buffer-build"
assert not out.exists()
shutil.copytree(original / "source", out / "source", symlinks=True)
shutil.copytree(original / "overlay", out / "overlay", symlinks=True)
shutil.copyfile(original / "input-param.json", out / "input-param.json")
(out / "kernel").mkdir()
relative = "source/ops_transformer/ascendc/grouped_matmul/op_kernel/"
header = out / relative / "grouped_matmul_antiquant_a8w4_msd.h"
before = header.read_text()
old = "    pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);"
new = """    // The FP32 epilogue owns four ubCalSize regions after the queues.
    const uint32_t tmpBytes = activeGroups.enabled ?
        4 * tiling->ubCalSize * sizeof(float) : tiling->ubRestBytes;
    pipe->InitBuffer(tmpBuff, tmpBytes);"""
production = root / "adaptive-release-build" / relative / header.name
assert new in production.read_text()
assert before.count(old) == 1
after = before.replace(old, new)
header.write_text(after)
diff = difflib.unified_diff(before.splitlines(), after.splitlines(),
                            header.name, header.name)
(out / "buffer-declaration.patch").write_text("\n".join(diff) + "\n")
changed = []
for source in (original / "source").rglob("*"):
    if source.is_file():
        target = out / source.relative_to(original)
        if source.read_bytes() != target.read_bytes():
            changed.append(str(source.relative_to(original)))
assert changed == [relative + header.name]
identity = {"parent_elf_sha256": hashlib.sha256(next(
    (original / "kernel").glob("*.o")).read_bytes()).hexdigest(),
    "changed_files": changed,
    "mid_header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
    "production_mid_sha256": hashlib.sha256(production.read_bytes()).hexdigest(),
    "declared_tmp_bytes": 131072, "wide_ub_end_bytes": 184448,
    "wide_ub_platform_limit_bytes": 196352, "npu_launched": False}
(out / "prepare.json").write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps(identity, indent=2))
