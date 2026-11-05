"""Freeze eight count vectors and the two production ELF identities."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/workspace/experiments/gmm2-compact-20261020")
OUT = ROOT / "guard-release-screen"
OUT.mkdir(exist_ok=True)
sources = [ROOT / name / "counts.json"
           for name in ("teacher-counts-screen", "guard-counts-screen")]
available = {}
for source in sources:
    for case in json.loads(source.read_text())["cases"]:
        available[case["name"]] = case
names = ["teacher-layer-29", "teacher-layer-60", "teacher-layer-21",
         "teacher-layer-17", "active64-hot8-rows17", "active64-hot8-rows25",
         "active32-hot4-rows32", "shared8"]
cases = []
for name in names:
    case = available[name]
    counts = case["counts"]
    assert len(counts) == 256 and sum(counts) == 256
    active, hot, maximum = sum(c > 0 for c in counts), sum(c > 16 for c in counts), max(counts)
    assert all(0 <= c <= 32 for c in counts)
    cases.append({"name": name, "counts": counts, "active": active,
                  "hot": hot, "maximum": maximum,
                  "old_expected_wide": maximum <= 16,
                  "new_expected_wide": maximum <= 16 or (
                      maximum <= 32 and active >= 64 and hot * 8 <= active)})
assert sum(c["old_expected_wide"] for c in cases) == 1
assert sum(c["new_expected_wide"] for c in cases) == 6
fixture = {"scope": "Four reconstructed teacher priors and four bounded synthetic count vectors; activations/weights/scales are seeded synthetic.",
           "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sources}, "cases": cases}
(OUT / "counts.json").write_text(json.dumps(fixture, indent=2) + "\n")
variant = "GroupedMatmul_a0be48d01b162435398ed52e1b6182d4"
identities = {}
for arm, directory in (("A", "adaptive-release-build"), ("B", "guard-release-build")):
    build = ROOT / directory
    binary = build / "overlay/vendors/tilert_gmm2/op_impl/ai_core/tbe/kernel/ascend910_93/grouped_matmul" / (variant + ".o")
    sha = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None
    if arm == "A":
        assert sha == "16e8bea5f250b048da83cbcecac5f0e77f0dd66886a0fc1412144c1d6f47a114"
    else:
        assert sha == "a40750d33709c588d284a37b2419b35e41a4536c70e2a6c44f78765147f8c404"
    if sha is not None:
        assert json.loads(binary.with_suffix(".json").read_text())["sha256"] == sha
    source = build / "source/ops_transformer/ascendc/grouped_matmul/op_kernel"
    identities[arm] = {"binary": str(binary), "elf_sha256": sha,
                       "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (source / n for n in (
                                       "grouped_matmul.cpp", "grouped_matmul_active_groups_a8w4.h",
                                       "grouped_matmul_antiquant_a8w4_msd.h"))}}
(OUT / "identity.json").write_text(json.dumps(identities, indent=2) + "\n")
print(json.dumps({"cases": names, "old_wide": 1, "new_wide": 6,
                  "elfs": {k: v["elf_sha256"] for k, v in identities.items()}}))
