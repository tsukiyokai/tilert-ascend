"""Source preservation and actual C++ plan contracts for the private reducer."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import prepare


HERE = Path(__file__).resolve().parent
PARENT = Path(os.environ.get(
    "COMMUNICATION_TREE_PARENT",
    "/tmp/tilert-megakernel-validation-20261021/communication-order-v2"))
CONTROL = Path(os.environ.get("COMMUNICATION_TREE_CONTROL",
                              str(PARENT / "communication_probe.so")))


class TreeContract(unittest.TestCase):
    def test_actual_header_against_round_recurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "contract"
            subprocess.run([
                "c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-fno-fast-math", "-ffp-contract=off",
                str(HERE / "contract_test.cpp"), "-o", str(binary)], check=True)
            report = json.loads(subprocess.check_output([str(binary)], text=True))
        self.assertEqual(report["rank_counts"], 15)
        self.assertEqual(report["random_cases"], 3840)
        self.assertEqual(report["special_cpu_cases"], 2160)
        self.assertEqual(report["max_live_slots"], 5)
        self.assertEqual(report["ub_bytes"], 45056)

    def test_parent_execution_and_host_abi_are_unchanged(self):
        for name in ("communication_probe.cpp", "communication_probe_host.cpp",
                     "communication_host.cpp", "communication_probe.h"):
            parent = (PARENT / name).read_text()
            child = prepare.transform(name, parent)
            restored = child.replace("whole_layer_tree", "whole_layer")
            restored = restored.replace("TESTS_GLM52_COMMUNICATION_TREE_",
                                        "TESTS_GLM52_WHOLE_LAYER_")
            restored = restored.replace("communication_tree_probe", "communication_probe")
            self.assertEqual(restored, parent)
        script = (PARENT / "communication_build.sh").read_text()
        self.assertEqual(prepare.transform("communication_build.sh", script), script)

    def test_window_layout_and_all_phase_barriers_are_unchanged(self):
        name = "communication.h"
        parent = (PARENT / name).read_text()
        child = prepare.transform(name, parent)
        restored = child.replace("whole_layer_tree", "whole_layer").replace(
            "TESTS_GLM52_COMMUNICATION_TREE_", "TESTS_GLM52_WHOLE_LAYER_")
        restored = restored.replace("kCommunicationChunk * 22;", "kCommunicationChunk * 10;")
        self.assertEqual(restored, parent)
        name = "communication_device.h"
        parent = (PARENT / name).read_text()
        child = prepare.transform(name, parent)
        anchor = "// All resident AIC and AIV cores"
        self.assertEqual(child[child.index(anchor):].replace(
            "whole_layer_tree", "whole_layer"), parent[parent.index(anchor):])
        begin = "template <AscendC::HardEvent event>"
        end = "__aicore__ inline void ReduceCommunicationPayload("
        actual = child[child.index(begin):child.index('#include "reduce.inc"')]
        self.assertEqual(actual, parent[parent.index(begin):parent.index(end)])

    def test_reuse_fences_and_phase_local_allocation(self):
        child = prepare.transform("communication_device.h",
                                  (PARENT / "communication_device.h").read_text())
        self.assertEqual(child.count("pipe.InitBuffer("), 2)
        self.assertIn("kCommunicationChunk * 4 * 5", child)
        self.assertNotIn("term_buffer_", child)
        reduce = (HERE / "reduce.inc").read_text()
        self.assertIn("CommunicationFence<AscendC::HardEvent::V_MTE2>();", reduce)
        self.assertIn("CommunicationFence<AscendC::HardEvent::MTE3_MTE2>();", reduce)
        self.assertEqual(reduce.count("AscendC::Add("), 1)
        self.assertEqual(reduce.count("AscendC::DataCopy(buffers.narrow, source"), 1)
        self.assertNotIn("Duplicate(", reduce)
        self.assertNotIn("SetAtomic", reduce)

    def test_parent_pins_and_private_kernel_symbols(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        for name in prepare.PARENT_FILES:
            self.assertEqual(prepare.digest(PARENT / name),
                             manifest["files_sha256"][name])
        self.assertEqual(prepare.digest(CONTROL), prepare.PARENT_LIBRARY_SHA)
        probe = prepare.transform("communication_probe.cpp",
                                  (PARENT / "communication_probe.cpp").read_text())
        self.assertIn("void communication_tree_probe(", probe)
        self.assertIn("communication_tree_probe<<<", probe)
        self.assertIn("__mix__(1, 2) __schedmode__(1)", probe)
        self.assertNotIn("communication_probe<<<", probe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
