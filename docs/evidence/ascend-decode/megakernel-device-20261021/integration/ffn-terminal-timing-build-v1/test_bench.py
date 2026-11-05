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
    "FFN_ACTIVE_TOKEN_JOIN_TIMING_PARENT", LOCAL / "ffn-active-token-join-timing-v1/bench.py"))
COMPOSED = Path(os.environ.get(
    "FFN_FROZEN_BUILD", LOCAL / "ffn-composed-build-v2"))
ACTIVE = Path(os.environ.get(
    "FFN_ACTIVE_TOKEN_JOIN_BASE_BUILD", LOCAL / "ffn-active-token-join-runtime-v1/build-v1"))
CANDIDATE = Path(os.environ.get(
    "FFN_TERMINAL_BUILD",
    LOCAL / "ffn-terminal-runtime-v1/build-v1"))
BASE_PROBE = Path(os.environ.get(
    "FFN_BASE_ABI_PROBE",
    Path(__file__).resolve().parent.parent / "ffn_composed/bench_abi_probe.cpp"))


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
            raise RuntimeError("Frozen active-token-join timing identity differs")
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
                   ["-DROUTED_ACTIVE_LIST", "-DCOMBINED_CANDIDATE"]))
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

    def test_all_33_addresses_and_launch_arguments_match(self):
        b = [Address(0x100000000 + i * 4096) for i in range(12)]
        r = [Address(0x200000000 + i * 4096) for i in range(20)]
        extra = Address(0xC00000000)
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
                        for function, probe in ((call.separate, self.probes[0]),
                                                (call.fused, self.probes[1])):
                            function()
                            actual = [probe.probe_value(i)
                                      for i in range(probe.probe_count())]
                            self.assertEqual(actual, expected)

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
        for name in ("owning_torch", "poison_outputs", "update", "capture",
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


if __name__ == "__main__":
    unittest.main()
