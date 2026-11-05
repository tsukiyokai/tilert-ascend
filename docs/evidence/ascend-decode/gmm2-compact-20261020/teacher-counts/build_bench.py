"""Derive the bounded count replay from the existing paired benchmark."""

import difflib
import hashlib
from pathlib import Path

root = Path("/workspace/experiments/gmm2-compact-20261020")
original = (root / "bench_retile.py").read_text()
assert hashlib.sha256(original.encode()).hexdigest() == (
    "d58aac9d181a015c49fad3dde83502c75587f6e65d6b09408d23338641f8a2df")
text = original.replace(
    "from validate import build_bank",
    "import hashlib\nimport sys\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
    "from validate import build_bank, digest")
text = text.replace("CALLS = 128\nBANKS = 8", "CALLS = 10\nBANKS = 8")
text = text.replace("3584 * 2**20", "2048 * 2**20")
text = text.replace(
    '"cases": []}',
    '"quota_bytes": 2048 * 2**20, "cases": [],\n'
    '              "counts_sha256": hashlib.sha256(Path(__file__).with_name("counts.json").read_bytes()).hexdigest(),\n'
    '              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n'
    '              "validate_sha256": hashlib.sha256((Path(__file__).resolve().parent.parent / "validate.py").read_bytes()).hexdigest()}')
start = text.index('    for width, distribution in ((1, "random")')
end = text.index('        graphs, outputs, buffers = {}, {}, []', start)
replacement = '''    fixture = json.loads(Path(__file__).with_name("counts.json").read_text())
    report["counts_scope"] = fixture["scope"]
    report["data_scope"] = "reconstructed teacher counts; seeded synthetic x/weights/scales"
    for case in fixture["cases"]:
        width, distribution = 32, case["name"]
        rows = 256
        frozen = torch.randint(-127, 128, (rows, 128), dtype=torch.int8,
                               generator=generator).to(args.device)
        token_scale = (torch.rand(rows, generator=generator) * .016 + .004).to(args.device)
        routes = [torch.tensor(case["counts"], dtype=torch.int64)] * CALLS
'''
text = text[:start] + replacement + text[end:]
text = text.replace(
    '"group_counts": [c.tolist() for c in routes],',
    '"group_counts": case["counts"],\n'
    '                  "active": case["active"], "hot": case["hot"],\n'
    '                  "maximum": case["maximum"],\n'
    '                  "compact_tasks": case["compact"]["tasks"],\n'
    '                  "wide_tasks": case["wide"]["tasks"],')
start = text.index('        exact = all(torch.equal(a.cpu(), b.cpu())')
end = text.index('\n        def measure(arm):', start)
text = text[:start] + '''        cpu_outputs = {arm: [value.cpu() for value in values]
                       for arm, values in outputs.items()}
        exact = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(cpu_outputs["base512"],
                                    cpu_outputs["wide1024"]))
        assert exact, (width, distribution)
        assert all(torch.isfinite(v).all().item()
                   for values in cpu_outputs.values() for v in values)
        output_hashes = {arm: [digest(v) for v in values]
                         for arm, values in cpu_outputs.items()}
''' + text[end:]
text = text.replace(
    '"outputs_exact": exact,',
    '"outputs_exact": exact, "output_sha256": output_hashes,\n'
    '                  "x_sha256": digest(frozen),\n'
    '                  "token_scale_sha256": digest(token_scale),')
text = text.replace(
    '        report["peak_reserved_bytes"] = torch.npu.max_memory_reserved()',
    '        report["peak_reserved_bytes"] = torch.npu.max_memory_reserved()\n'
    '        report["peak_allocated_bytes"] = torch.npu.max_memory_allocated()')
compile(text, "bench_counts.py", "exec")
out = root / "teacher-counts-screen"
(out / "bench_counts.py").write_text(text)
diff = difflib.unified_diff(original.splitlines(), text.splitlines(),
                            "bench_retile.py", "bench_counts.py")
(out / "bench-counts.diff").write_text("\n".join(diff) + "\n")
print("bench_sha256", hashlib.sha256(text.encode()).hexdigest())
