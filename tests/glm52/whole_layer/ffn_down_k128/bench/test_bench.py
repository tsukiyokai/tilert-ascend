"""Verify both active C ABIs and the retained event/numerical gates on CPU."""

import ast
import ctypes
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from prepare_bench import PARENT, digest, transform


LOCAL = Path("/tmp/tilert-megakernel-validation-20261021")
TIMING = Path(os.environ.get(
    "FFN_DOWN_K128_TIMING_PARENT", LOCAL / "ffn-down-n1024-timing-v1/bench.py"))
COMPOSED = Path(os.environ.get(
    "FFN_FROZEN_BUILD", LOCAL / "ffn-composed-build-v2"))
ACTIVE = Path(os.environ.get(
    "FFN_DOWN_K128_BASE_BUILD", LOCAL / "ffn-down-n1024-build-v2"))
CANDIDATE = Path(os.environ.get(
    "FFN_DOWN_K128_BUILD",
    LOCAL / "ffn-down-k128-build-v2"))
BASE_PROBE = Path(os.environ.get(
    "FFN_BASE_ABI_PROBE",
    LOCAL / "ffn-down-n1024-timing-v1/parent/bench_abi_probe.cpp"))


class Address:
    device = "npu:0"

    def __init__(self, value):
        self.value = value

    def data_ptr(self):
        return self.value


class BenchmarkContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if digest(TIMING) != PARENT:
            raise RuntimeError("Frozen original-shared static-GU timing identity differs")
        cls.original = TIMING.read_text()
        cls.source = transform(cls.original)
        cls.temporary = tempfile.TemporaryDirectory()
        directory = Path(cls.temporary.name)
        benchmark = directory / "bench.py"
        benchmark.write_text(cls.source)
        spec = importlib.util.spec_from_file_location("combined_bench", benchmark)
        cls.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.bench)
        cls.libraries = []
        probes = ((COMPOSED, BASE_PROBE, []),
                  (ACTIVE, Path(__file__).with_name("bench_abi_probe.cpp"),
                   ["-DROUTED_ACTIVE_LIST"]),
                  (CANDIDATE, Path(__file__).with_name("bench_abi_probe.cpp"),
                   ["-DROUTED_ACTIVE_LIST", "-DDOWN_K128_CANDIDATE"]))
        for index, (frozen, probe, definitions) in enumerate(probes):
            library = directory / f"probe{index}.so"
            subprocess.run([
                "g++", "-std=c++17", "-shared", "-fPIC",
                "-DROUTED_INPUT_FRONT", "-DROUTED_SHARED_PIPELINE",
                "-DROUTED_ROW_BALANCED",
                *definitions, "-I", str(frozen / "source"), str(probe),
                "-o", str(library)], check=True)
            cls.libraries.append(library)
        cls.launchers = cls.bench.Launchers(
            cls.libraries[0], cls.libraries[0], cls.libraries[0],
            cls.libraries[2], "active", cls.libraries[1])
        cls.probes = [cls.launchers.libraries[index] for index in (3, 4)]
        for probe in cls.probes:
            for name in ("probe_count", "probe_value", "probe_shared_size",
                         "probe_shared_offset"):
                getattr(probe, name).restype = ctypes.c_uint64

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_32_addresses_match_and_each_arm_uses_its_own_down_tiling(self):
        b = [Address(0x100000000 + i * 4096) for i in range(12)]
        r = [Address(0x200000000 + i * 4096) for i in range(20)]
        extra = Address(0xC00000000)
        self.launchers.down_tilings = {
            "fused": Address(0xD00000000), "separate": Address(0xE00000000)}
        bank = {"routed": [Address(0x300000000 + i * 4096) for i in range(6)],
                "shared": [Address(0x400000000 + i * 4096) for i in range(4)],
                "gamma": Address(0x500000000), "gate": Address(0x600000000),
                "bias": Address(0x700000000)}
        hidden, residual = Address(0x800000000), Address(0x900000000)
        stream, sync = 0xA00000001, 0xB00000003

        def empty(shape, *, dtype, device):
            self.assertEqual((shape, dtype, device), ((264,), "int32", "npu:0"))
            return extra

        torch = SimpleNamespace(
            npu=SimpleNamespace(current_stream=lambda:
                                SimpleNamespace(npu_stream=stream)),
            int32="int32", int64="int64", empty=empty)
        for rows in (1, 2, 12, 32):
            for blocks in (16, 24):
                for layout in (0, 1):
                    with self.subTest(rows=rows, blocks=blocks, layout=layout):
                        call = self.bench.Calls(
                            torch, self.launchers, (b, r, sync), bank,
                            hidden, residual, rows, blocks, layout)
                        capacity = 16 if rows <= 16 else 32
                        expected = [stream, sync, hidden.value, residual.value,
                                    bank["gamma"].value, bank["gate"].value,
                                    bank["bias"].value]
                        expected += [x.value for x in bank["routed"] + b + r]
                        expected += [extra.value, b[6].value, b[7].value,
                            *(x.value for x in bank["shared"]),
                            r[17].value, r[18].value, r[19].value, capacity,
                            *(x * capacity for x in (6144, 12288, 12416, 12448)),
                            rows, blocks, layout,
                            struct.unpack("I", struct.pack("f", 1e-5))[0]]
                        for arm, function, probe in (("separate", call.separate, self.probes[0]),
                                                     ("fused", call.fused, self.probes[1])):
                            expected[7 + 6 + 12 + 7] = self.launchers.down_tilings[arm].value
                            function()
                            actual = [probe.probe_value(i)
                                      for i in range(probe.probe_count())]
                            self.assertEqual(actual, expected)

    def test_corrected_torch_reference_validates_the_active_table(self):
        recorded = []
        torch = SimpleNamespace(ops=SimpleNamespace(
            tilert_ffn_down_n1024=SimpleNamespace(
                forward_out=lambda *args: recorded.append(("separate", args))),
            tilert_ffn_down_k128=SimpleNamespace(
                forward_out=lambda *args: recorded.append(("fused", args)))))
        b, r, extra = [object() for _ in range(12)], [object() for _ in range(20)], object()
        bank = {name: object() for name in ("gamma", "gate", "bias", "routed", "shared")}
        call = SimpleNamespace(arena=(b, r, 37), torch=torch, bank=bank,
                               hidden=object(), residual=object(), extra=extra,
                               blocks=24, layout=1)
        for owner in ("fused", "separate"):
            self.bench.Calls.owning_torch(call, owner)
        self.assertEqual([owner for owner, _ in recorded], ["fused", "separate"])
        self.assertEqual(recorded[0][1], recorded[1][1])
        self.assertEqual(recorded[0][1][7], b)
        self.assertEqual(recorded[0][1][8], [*r, extra])
        self.assertEqual(recorded[0][1][9:], (1e-5, 24, 1, 37))

    def test_arena_keeps_the_validator_owned_tilings_and_active_table(self):
        bridge = [Address(0x100000000 + i * 4096) for i in range(12)]
        body = [Address(0x200000000 + i * 4096) for i in range(21)]
        hidden = object()
        calls = []

        def prepare(*args):
            calls.append(args)
            return bridge, body, 37

        torch = SimpleNamespace(ops=SimpleNamespace(
            tilert_ffn_down_n1024=SimpleNamespace(prepare=prepare),
            tilert_ffn_down_k128=SimpleNamespace(prepare=prepare)))
        launchers = SimpleNamespace(extra_by_arena={}, down_tilings={}, down_owners={})
        arena = self.bench.prepare_arena(torch, launchers, hidden, 24, "fused")
        self.assertEqual(calls, [(hidden, 24, 1)])
        self.assertEqual(arena, (bridge, body[:20], 37))
        self.assertIs(launchers.extra_by_arena[bridge[8].data_ptr()], body[20])
        self.assertIs(launchers.down_tilings["fused"], body[7])
        self.assertIs(launchers.down_owners["fused"][1], body)

    def test_actual_shared_pod_layouts(self):
        shared = self.bench.SharedInput
        self.assertEqual(ctypes.sizeof(shared), 112)
        for probe in self.probes:
            self.assertEqual(probe.probe_shared_size(), 112)
            for index, (name, _) in enumerate(shared._fields_):
                self.assertEqual(getattr(shared, name).offset,
                                 probe.probe_shared_offset(index))

    def test_event_window_and_existing_gates_are_preserved(self):
        trees = [ast.parse(value) for value in (self.original, self.source)]
        for name in ("poison_outputs", "update", "capture",
                     "snapshot", "poison", "checked", "check_extra",
                     "sequence", "input_arguments"):
            functions = [[node for node in ast.walk(tree)
                          if isinstance(node, ast.FunctionDef) and node.name == name]
                         for tree in trees]
            self.assertEqual(len(functions[0]), 1)
            self.assertEqual(ast.dump(functions[0][0]), ast.dump(functions[1][0]))
        begin = "                        start, end = torch.npu.Event"
        end = "                        actual_tail ="
        self.assertEqual(
            self.original[self.original.index(begin):self.original.index(end)],
            self.source[self.source.index(begin):self.source.index(end)])

    def test_both_arms_check_the_entire_active_table(self):
        self.assertEqual(self.source.count(
            'result["extra_checks"].append({"arm": arm,'), 3)
        self.assertNotIn('if arm == "fused":', self.source)
        self.assertIn('extra.fill_(1)\n            extra[0].zero_()', self.source)
        self.assertIn('reference = torch.zeros(264, dtype=torch.int32)', self.source)


    def test_full_tiling_bytes_reject_an_erased_or_partially_changed_intervention(self):
        import torch

        launchers = SimpleNamespace(down_tilings={}, down_owners={})
        for arm, expected in self.bench.EXPECTED_DOWN_TILINGS["arms"].items():
            payload = bytes.fromhex(expected["layouts"]["1"]["hex"])
            tensor = torch.tensor(list(payload), dtype=torch.uint8)
            body = [None] * 21
            body[7] = tensor
            launchers.down_tilings[arm] = tensor
            launchers.down_owners[arm] = ([], body, 1)
        records = self.bench.check_down_tilings(torch, launchers)
        self.assertEqual(len(records), 2)
        original = launchers.down_tilings["fused"].clone()
        launchers.down_tilings["fused"].copy_(launchers.down_tilings["separate"])
        with self.assertRaisesRegex(RuntimeError, "host tiler"):
            self.bench.check_down_tilings(torch, launchers)
        launchers.down_tilings["fused"].copy_(original)
        launchers.down_tilings["fused"][199] ^= 1
        with self.assertRaisesRegex(RuntimeError, "host tiler"):
            self.bench.check_down_tilings(torch, launchers)
        launchers.down_tilings["fused"].copy_(original)
        launchers.down_tilings["fused"] = original.clone()
        with self.assertRaisesRegex(RuntimeError, "Tensor owner"):
            self.bench.check_down_tilings(torch, launchers)

    def test_tiling_checks_bracket_captures_and_do_not_modify_replays(self):
        for stage in ("before_capture", "after_capture", "after_timing_capture",
                      "after_timing", "after_postchecks"):
            self.assertEqual(self.source.count(f'"stage": "{stage}"'), 1)
        self.assertNotIn(".set_(", self.source)
        self.assertIn("launchers.down_owners[owner] = (bridge, body, sync)", self.source)
        self.assertIn('for slot in (6, 18):', self.source)
        self.assertNotIn('for slot in (6, 7, 18):', self.source)


if __name__ == "__main__":
    unittest.main()
