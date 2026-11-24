"""Check the two actual frozen FFN ABIs and retained fresh-write gates on CPU."""

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
ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / "docs/evidence").is_dir():
    ROOT = ROOT.parent
DEFAULT_TIMING = (ROOT / "docs/evidence/ascend-decode/megakernel-device-20261021"
                  / "a3-194/ffn-composed-timing-v3")
TIMING = Path(os.environ.get("FFN_TIMING_PARENT", DEFAULT_TIMING))
BASE = Path(os.environ.get("FFN_FROZEN_BUILD", LOCAL / "ffn-composed-build-v2"))
CANDIDATE = Path(os.environ.get("FFN_DOWN_OVERLAP_BUILD", LOCAL / "ffn-down-overlap-build-v2"))


class Address:
    def __init__(self, value):
        self.value = value

    def data_ptr(self):
        return self.value


def load_benchmark(path):
    spec = importlib.util.spec_from_file_location("down_overlap_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (TIMING / "bench.py", BASE / "source/contract.h",
                     CANDIDATE / "source/contract.h"):
            if not path.is_file():
                raise RuntimeError(f"Required frozen contract is missing: {path}")
        if digest(TIMING / "bench.py") != PARENT:
            raise RuntimeError("Timing parent identity differs")
        cls.temporary = tempfile.TemporaryDirectory()
        temporary = Path(cls.temporary.name)
        source = transform((TIMING / "bench.py").read_text())
        (temporary / "bench.py").write_text(source)
        cls.bench = load_benchmark(temporary / "bench.py")
        cls.libraries = []
        probes = ((BASE, TIMING / "bench_abi_probe.cpp"),
                  (CANDIDATE, Path(__file__).with_name("bench_abi_probe.cpp")))
        for index, (frozen, probe) in enumerate(probes):
            library = temporary / f"probe{index}.so"
            subprocess.run([
                "g++", "-std=c++17", "-shared", "-fPIC", "-DROUTED_INPUT_FRONT",
                "-DROUTED_SHARED_PIPELINE", "-I", str(frozen / "source"),
                str(probe), "-o", str(library)], check=True, capture_output=True, text=True)
            cls.libraries.append(library)
        cls.launchers = cls.bench.Launchers(
            cls.libraries[0], cls.libraries[0], cls.libraries[0], cls.libraries[1])
        cls.probes = [cls.launchers.libraries[index] for index in (0, 3)]
        for probe in cls.probes:
            for name in ("probe_count", "probe_value"):
                getattr(probe, name).restype = ctypes.c_uint64
        cls.probes[0].probe_size.restype = ctypes.c_uint64
        cls.probes[0].probe_offset.restype = ctypes.c_uint64
        cls.probes[1].probe_shared_size.restype = ctypes.c_uint64
        cls.probes[1].probe_shared_offset.restype = ctypes.c_uint64

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_every_argument_matches_both_actual_headers(self):
        b = [Address(0x100000000 + i * 4096) for i in range(12)]
        r = [Address(0x200000000 + i * 4096) for i in range(20)]
        bank = {"routed": [Address(0x300000000 + i * 4096) for i in range(6)],
                "shared": [Address(0x400000000 + i * 4096) for i in range(4)],
                "gamma": Address(0x500000000), "gate": Address(0x600000000),
                "bias": Address(0x700000000)}
        hidden, residual = Address(0x800000000), Address(0x900000000)
        stream, sync = 0xA00000001, 0xB00000003
        torch = SimpleNamespace(npu=SimpleNamespace(
            current_stream=lambda: SimpleNamespace(npu_stream=stream)))
        for rows in (1, 2, 12, 32):
            for blocks in (16, 24):
                for layout in (0, 1):
                    with self.subTest(rows=rows, blocks=blocks, layout=layout):
                        calls = self.bench.Calls(torch, self.launchers, (b, r, sync),
                            bank, hidden, residual, rows, blocks, layout)
                        capacity = 16 if rows <= 16 else 32
                        expected = [1, stream, sync, hidden.value, residual.value,
                            bank["gamma"].value, bank["gate"].value, bank["bias"].value]
                        expected += [x.value for x in bank["routed"] + b + r]
                        expected += [b[6].value, b[7].value,
                            *(x.value for x in bank["shared"]),
                            r[17].value, r[18].value, r[19].value, capacity,
                            *(x * capacity for x in (6144, 12288, 12416, 12448)),
                            rows, blocks, layout, struct.unpack("I", struct.pack("f", 1e-5))[0]]
                        for call, probe in ((calls.separate, self.probes[0]),
                                            (calls.fused, self.probes[1])):
                            probe.probe_clear()
                            call()
                            actual = [probe.probe_value(i) for i in range(probe.probe_count())]
                            self.assertEqual(actual, expected)

    def test_shared_pod_sizes_and_offsets(self):
        shared = self.bench.SharedInput
        self.assertEqual(ctypes.sizeof(shared), 112)
        self.assertEqual(self.probes[0].probe_size(1), 112)
        self.assertEqual(self.probes[1].probe_shared_size(), 112)
        for index, (name, _) in enumerate(shared._fields_):
            self.assertEqual(getattr(shared, name).offset, self.probes[0].probe_offset(1, index))
            self.assertEqual(getattr(shared, name).offset, self.probes[1].probe_shared_offset(index))

    def test_numerical_timing_and_owner_gates_are_unchanged(self):
        original = (TIMING / "bench.py").read_text()
        candidate = transform(original)
        trees = [ast.parse(value) for value in (original, candidate)]
        names = ("poison_outputs", "owning_torch", "update", "capture", "snapshot",
                 "poison", "checked", "input_arguments")
        for name in names:
            functions = [[node for node in ast.walk(tree)
                          if isinstance(node, ast.FunctionDef) and node.name == name]
                         for tree in trees]
            self.assertEqual(len(functions[0]), 1)
            self.assertEqual(ast.dump(functions[0][0]), ast.dump(functions[1][0]))
        start = "            arenas = {"
        end = '        report["pass"] = True'
        self.assertEqual(original[original.index(start):original.index(end)],
                         candidate[candidate.index(start):candidate.index(end)])
        self.assertNotIn("extra_by_arena", candidate)

    def test_empty_replay_is_rejected_without_poisoning_padding(self):
        import torch

        storage = torch.zeros(2, 16)
        logical = storage[:, :12]
        expected = logical.clone()
        counts = torch.ones(256, dtype=torch.int32)
        codes = torch.zeros(2, 128, dtype=torch.int8)
        self.bench.poison_outputs({"bridge.combined": logical,
            "body.counts": counts, "body.shared_middle": codes})
        self.assertFalse(torch.equal(logical, expected))
        self.assertTrue(bool(torch.isnan(logical).all()))
        self.assertTrue(bool((storage[:, 12:] == 0).all()))
        self.assertTrue(bool((counts == 0).all()))
        self.assertTrue(bool((codes == 1).all()))


if __name__ == "__main__":
    unittest.main()
