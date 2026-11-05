"""Check the host address mapping and unchanged numerical source."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from prepare import FIELDS, LAYER, bind_full, bind_kernel, prepare, private_names


if "SHARED_DESCRIPTOR_PARENT" in os.environ:
    PARENT = Path(os.environ["SHARED_DESCRIPTOR_PARENT"])
else:
    PARENT = Path(__file__).resolve().parents[4] / (
        "docs/evidence/ascend-decode/megakernel-device-20261021"
        "/integration/runtime212-build-v1/attention")


class DescriptorContract(unittest.TestCase):
    def test_host_launch_contract_and_field_order(self):
        original = (PARENT / "shared/source" / LAYER / "attention_full_kernel.cpp").read_text()
        transformed = bind_kernel(original)
        marker = "void whole_layer_full_attention_launch("
        declaration = original.split(marker, 1)[1].split("\n{", 1)[0]
        self.assertEqual(transformed.split(marker, 1)[1].split("\n{", 1)[0], declaration)
        fields = re.findall(r"void\*\s*(\w+)", declaration)
        self.assertEqual(fields, ["stream"] + FIELDS)
        capture = transformed.split("Capture<GM_ADDR>(", 1)[1].split(");", 1)[0]
        self.assertEqual(re.findall(r"\(GM_ADDR\)(\w+)", capture), FIELDS)
        self.assertEqual(transformed.count("InitSocState();"), 1)
        self.assertEqual(transformed.count("SetSyncBaseAddr(sync_base);"), 1)

    def test_tail_statements_and_phase_boundaries_are_preserved(self):
        original = (PARENT / "shared/source" / LAYER / "attention_full_native.h").read_text()
        transformed = bind_full(original)
        start = "    mla_tail::device::Forward("
        old = original[original.index(start):]
        actual = transformed[transformed.index(start):]
        actual = actual.replace("arguments.attention.scratch", "arena")
        for field in FIELDS[20:]:
            actual = actual.replace("arguments." + field, field)
        self.assertEqual(old, actual)
        self.assertIn("Forward(pipe, config.attention, arguments.attention);", transformed)
        self.assertEqual(transformed.count("EndPhase(pipe);"), 1)

    def test_other_numerical_code_and_host_validation_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, output)
            for name in ("attention_full_binding.cpp", "attention_host.h",
                         "attention_full_contract.h", "attention_contract.h"):
                expected = private_names((PARENT / "shared/source" / LAYER / name).read_text())
                self.assertEqual((output / "source" / LAYER / name).read_text(), expected)
            name = "attention_native.h"
            old = (PARENT / "shared/source" / LAYER / name).read_text()
            new = (output / "source" / LAYER / name).read_text()
            marker = "__aicore__ inline uint32_t Ceil("
            self.assertEqual(new[new.index(marker):], private_names(old[old.index(marker):]))
            for name in ("mla_tail.h", "mla_tail_native.h", "mla_tail_tiling.cpp"):
                relative = "tests/glm52/mla_tail/" + name
                expected = private_names((PARENT / "shared/source" / relative).read_text())
                self.assertEqual((output / "source" / relative).read_text(), expected)

    def test_full_width_addresses_and_actual_arena_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, output)
            binary = Path(directory) / "probe"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-I", str(output / "source" / LAYER),
                str(Path(__file__).with_name("descriptor_probe.cpp")),
                "-o", str(binary),
            ], check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
