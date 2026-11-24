"""Validate variant C ABIs against separate frozen headers without NPU."""

import ast
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


def load_module(path):
    spec = importlib.util.spec_from_file_location("variant_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nested_function(path, name, globals_dict):
    run = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    function = next(node for node in run.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals_dict)
    return globals_dict[name]


class VariantABI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.benchmark_path = Path(os.environ.get(
            "FFN_VARIANT_BENCH",
            "/tmp/tilert-megakernel-validation-20261021/ffn-variant-timing-v3/bench.py"))
        cls.benchmark = load_module(cls.benchmark_path)
        configured = os.environ.get("FFN_BUILD_EVIDENCE")
        if configured:
            evidence = Path(configured)
        else:
            evidence = (Path(__file__).resolve().parents[4]
                        / "docs/evidence/ascend-decode/megakernel-device-20261021/a3-194")
        builds = {
            kind: Path(os.environ.get(f"FFN_{kind.upper()}_BUILD", evidence / name))
            for kind, name in (("base", "ffn-composed-build-v2"),
                               ("active", "ffn-active-build-v1"),
                               ("profile", "ffn-profile-build-v1"))}
        base_probe = Path(os.environ.get(
            "FFN_BASE_ABI_PROBE", evidence / "ffn-composed-timing-v3/bench_abi_probe.cpp"))
        cls.temporary = tempfile.TemporaryDirectory()
        cls.paths = {}
        cls.header_identities = {}
        for kind, build in builds.items():
            manifest = json.loads((build / "manifest.json").read_text())
            for name, expected in manifest["sources"].items():
                actual = hashlib.sha256((build / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Frozen {kind} source changed: {name}")
            cls.header_identities[kind] = manifest["sources"]["source/contract.h"]
            command = json.loads((build / "commands.jsonl").read_text().splitlines()[0])
            macros = [value for value in command["argv"] if value.startswith("-D")]
            output = Path(cls.temporary.name) / (kind + ".so")
            source = base_probe if kind == "base" else Path(__file__).with_name(
                "variant_abi_probe.cpp")
            if kind == "active":
                macros.append("-DVARIANT_ACTIVE=1")
            result = subprocess.run([
                "g++", "-std=c++17", "-shared", "-fPIC", *macros,
                "-I", str(build / "source"), str(source), "-o", str(output)],
                capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            cls.paths[kind] = output

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def calls(self, kind, rows=12, blocks=24, layout=1, launchers=None):
        torch = self.torch
        b = [torch.empty(1, dtype=torch.int64) for _ in range(12)]
        r = [torch.empty(1, dtype=torch.int64) for _ in range(20)]
        bank = {"routed": [torch.empty(1) for _ in range(6)],
                "shared": [torch.empty(1) for _ in range(4)],
                "gamma": torch.empty(1), "gate": torch.empty(1), "bias": torch.empty(1)}
        if launchers is None:
            launchers = self.benchmark.Launchers(
                self.paths["base"], self.paths["base"], self.paths["base"],
                self.paths[kind], kind)
        runtime = SimpleNamespace(empty=torch.empty, int32=torch.int32, int64=torch.int64,
                                  npu=SimpleNamespace(current_stream=lambda:
                                      SimpleNamespace(npu_stream=0x100000007)))
        call = self.benchmark.Calls(runtime, launchers, (b, r, 0x20000000b), bank,
                                    torch.empty(1), torch.empty(1), rows, blocks, layout)
        return call

    def record(self, library, prefix):
        count = getattr(library, prefix + "count")
        value = getattr(library, prefix + "value")
        count.restype = value.restype = ctypes.c_uint64
        return [value(i) for i in range(count())]

    def test_independent_headers_keep_macro_and_struct_layouts(self):
        self.assertEqual(len(set(self.header_identities.values())), 3)
        for kind in ("active", "profile"):
            call = self.calls(kind)
            base = call.launchers.libraries[0]
            variant = call.launchers.libraries[-1]
            self.assertEqual(base.probe_size(0), 56)
            self.assertEqual(base.probe_size(1), 112)
            self.assertEqual(variant.variant_shared_size(), 112)
            self.assertEqual(variant.variant_dispatch_size(), 64 if kind == "active" else 56)
            self.assertEqual(variant.variant_body_count(), 21 if kind == "active" else 20)
            for index, (name, _) in enumerate(self.benchmark.SharedInput._fields_):
                expected = variant.variant_shared_offset(index)
                self.assertEqual(getattr(self.benchmark.SharedInput, name).offset, expected)

    def test_every_common_address_and_variant_tail_survives_c_abi(self):
        for kind in ("active", "profile"):
            for rows in (1, 12, 32):
                for blocks in (16, 24):
                    for layout in (0, 1):
                        with self.subTest(kind=kind, rows=rows, blocks=blocks, layout=layout):
                            call = self.calls(kind, rows, blocks, layout)
                            call.separate()
                            base = self.record(call.launchers.libraries[0], "probe_")[1:]
                            call.fused()
                            candidate = self.record(call.launchers.libraries[-1], "variant_")
                            body_end = 7 + 6 + 12 + 20
                            if kind == "active":
                                self.assertEqual(candidate[body_end], call.extra.data_ptr())
                                candidate = candidate[:body_end] + candidate[body_end + 1:]
                            else:
                                self.assertEqual(candidate[-1], call.extra.data_ptr())
                                candidate = candidate[:-1]
                            self.assertEqual(candidate, base)
                            b, r, _ = call.arena
                            self.assertEqual(base[13:45], [t.data_ptr() for t in (*b, *r)])
                            self.assertNotIn(call.extra.data_ptr(), base[13:45])

    def test_extra_owner_is_per_arena_and_shared_by_banks(self):
        for kind in ("active", "profile"):
            first = self.calls(kind)
            second = self.calls(kind, launchers=first.launchers)
            b, r, sync = first.arena
            again = self.benchmark.Calls(first.torch, first.launchers, (b, r, sync),
                first.bank, first.hidden, first.residual, first.rows, first.blocks, first.layout)
            self.assertIs(again.extra, first.extra)
            self.assertNotEqual(first.extra.data_ptr(), second.extra.data_ptr())
            expected = 264 * 4 if kind == "active" else 3 * first.blocks * 32 * 8
            self.assertEqual(first.extra.nbytes, expected)

    def test_active_padding_omission_is_rejected_after_real_poison(self):
        torch = self.torch
        call = self.calls("active")
        args = SimpleNamespace(rows=12, blocks=24, candidate_kind="active")
        namespace = {
            "args": args, "launchers": call.launchers,
            "checker": SimpleNamespace(all_values=lambda *args: {}),
            "poison_outputs": self.benchmark.poison_outputs,
            "torch": SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))}
        poison = nested_function(self.benchmark_path, "poison", namespace)
        poison(call.arena)
        self.assertEqual(int(call.extra[0]), 0)
        self.assertTrue(bool((call.extra[1:] == 1).all()))
        counts = torch.zeros(256, dtype=torch.int32)
        counts[4] = counts[9] = 1
        call.extra[0] = 2
        call.extra[8:10] = torch.tensor([4, 9], dtype=torch.int32)
        namespace.update(torch=torch, bridge=SimpleNamespace(compare=lambda _, a, b:
            {"bitwise_equal": torch.equal(a, b)}))
        check_extra = nested_function(self.benchmark_path, "check_extra", namespace)
        with self.assertRaisesRegex(RuntimeError, "Active-expert table"):
            check_extra(call.arena, {"body.counts": counts})
        call.extra.zero_()
        call.extra[0] = 2
        call.extra[8:10] = torch.tensor([4, 9], dtype=torch.int32)
        self.assertTrue(check_extra(call.arena, {"body.counts": counts})["bitwise_equal"])


if __name__ == "__main__":
    unittest.main()
