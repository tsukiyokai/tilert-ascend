"""Check Python C ABI marshalling against frozen C++ declarations, without NPU."""

import ctypes
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from bench import Calls, DispatchInput, Launchers, SharedInput, poison_outputs, summary


class Address:
    def __init__(self, value):
        self.value = value

    def data_ptr(self):
        return self.value


class BenchmarkABI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("FFN_FROZEN_BUILD")
        if configured:
            frozen = Path(configured)
        else:
            evidence = (Path(__file__).resolve().parents[4]
                        / "docs/evidence/ascend-decode/megakernel-device-20261021")
            frozen = evidence / "a3-194/ffn-composed-build-v2"
        if not (frozen / "source/contract.h").is_file():
            raise unittest.SkipTest("Frozen composition headers are unavailable")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.library = Path(cls.temporary.name) / "abi_probe.so"
        subprocess.run([
            "g++", "-std=c++17", "-shared", "-fPIC", "-DROUTED_INPUT_FRONT",
            "-DROUTED_SHARED_PIPELINE", "-I", str(frozen / "source"),
            str(Path(__file__).with_name("bench_abi_probe.cpp")),
            "-o", str(cls.library)], check=True, capture_output=True, text=True)
        cls.launchers = Launchers(cls.library, cls.library, cls.library)
        cls.probe = cls.launchers.libraries[0]
        for name in ("probe_count", "probe_value", "probe_size", "probe_offset"):
            getattr(cls.probe, name).restype = ctypes.c_uint64

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.probe.probe_clear()
        self.bridge = [Address(0x100000 + i * 0x1000) for i in range(12)]
        self.body = [Address(0x200000 + i * 0x1000) for i in range(20)]
        self.bank = {"routed": [Address(0x300000 + i * 0x1000) for i in range(6)],
                     "shared": [Address(0x400000 + i * 0x1000) for i in range(4)],
                     "gamma": Address(0x500000), "gate": Address(0x600000),
                     "bias": Address(0x700000)}
        self.stream, self.sync = 0x100000001, 0x200000003
        torch = SimpleNamespace(npu=SimpleNamespace(
            current_stream=lambda: SimpleNamespace(npu_stream=self.stream)))
        self.calls = Calls(torch, self.launchers, (self.bridge, self.body, self.sync),
                           self.bank, Address(0x800000), Address(0x900000), 12, 24, 1)

    def recorded(self):
        return [self.probe.probe_value(i) for i in range(self.probe.probe_count())]

    def expected_shared(self):
        return [self.bridge[6].value, self.bridge[7].value,
                *(value.value for value in self.bank["shared"]), self.body[17].value,
                self.body[18].value, self.body[19].value, 16,
                98304, 196608, 198656, 199168]

    def test_pod_sizes_and_offsets_match_frozen_headers(self):
        for kind, value in enumerate((DispatchInput, SharedInput)):
            self.assertEqual(ctypes.sizeof(value), self.probe.probe_size(kind))
            for index, (name, _) in enumerate(value._fields_):
                self.assertEqual(getattr(value, name).offset, self.probe.probe_offset(kind, index))

    def test_composed_forwards_every_scratch_and_weight_pointer(self):
        self.calls.fused()
        expected = [1, self.stream, self.sync, 0x800000, 0x900000,
                    0x500000, 0x600000, 0x700000]
        expected += [value.value for value in self.bank["routed"] + self.bridge + self.body]
        expected += self.expected_shared() + [12, 24, 1, struct.unpack("I", struct.pack("f", 1e-5))[0]]
        self.assertEqual(self.recorded(), expected)

    def test_three_calls_keep_exact_output_and_input_aliases(self):
        self.calls.separate()
        expected = [2, self.stream, self.sync, 0x800000, 0x900000,
                    0x500000, 0x600000, 0x700000]
        expected += [value.value for value in self.bridge]
        expected += [12, 24, struct.unpack("I", struct.pack("f", 1e-5))[0]]
        expected += [3, 24, self.stream, self.sync, self.body[10].value]
        expected += [value.value for value in self.bank["routed"]]
        expected += [self.body[index].value for index in
                     (11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 9, 6, 7, 8)]
        expected += [256, 12, 1, 0, 0, 0, 0, self.bridge[4].value,
                     self.bridge[0].value, self.bridge[5].value]
        expected += self.expected_shared()
        expected += [4, self.stream, self.sync, self.body[9].value, self.body[19].value]
        expected += [value.value for value in self.bridge] + [12, 24]
        self.assertEqual(self.recorded(), expected)


class TimingStatistics(unittest.TestCase):
    def test_empty_replay_cannot_reuse_finite_reference(self):
        import torch

        storage = torch.zeros(2, 16, dtype=torch.float32)
        logical = storage[:, :12]
        expected = logical.clone()
        metadata = torch.ones(256, dtype=torch.int32)
        codes = torch.zeros(3, 8, dtype=torch.int8)
        poison_outputs({"bridge.combined": logical, "body.counts": metadata,
                        "body.shared_middle": codes})
        self.assertTrue(bool(torch.isnan(logical).all()))
        self.assertFalse(torch.equal(logical, expected))
        self.assertTrue(bool((storage[:, 12:] == 0).all()))
        self.assertTrue(bool((metadata == 0).all()))
        self.assertTrue(bool((codes == 1).all()))

    def test_independent_event_samples_preserve_signed_savings(self):
        actual = summary([-3.0, -1.0, 1.0, 3.0])
        self.assertEqual(actual["samples"], 4)
        self.assertEqual(actual["p50"], 0)
        self.assertAlmostEqual(actual["p90"], 2.4)
        self.assertAlmostEqual(actual["p99"], 2.94)
        self.assertAlmostEqual(actual["std"], 5 ** .5)


if __name__ == "__main__":
    unittest.main()
