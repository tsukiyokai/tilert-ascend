"""Validate the combined argument map against the original active FFN."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from prepare import BODY, load_transform, prepare, rename


class ActiveDescriptorContract(unittest.TestCase):
    parent = Path(os.environ.get("FFN_ACTIVE_DESCRIPTOR_PARENT", str(
        Path(__file__).resolve().parents[4]
        / "docs/evidence/ascend-decode/megakernel-device-20261021"
        / "a3-194/ffn-active-build-v1")))

    def test_numerical_body_and_barriers_are_unchanged(self):
        transform, _ = load_transform()
        original = (self.parent / BODY).read_text()
        candidate, numerical, _ = transform.bind_body(original)
        restored = transform.body_parts(candidate)[2]
        for name, expression in sorted(
                transform.BINDINGS.items(), key=lambda item: -len(item[1])):
            restored = restored.replace(expression, name)
        self.assertEqual(restored, numerical)
        self.assertEqual(restored.count("timing.Barrier("), 5)
        self.assertIn("dispatch_input.active_list", restored)

    def test_fields_select_original_body_arguments(self):
        transform, _ = load_transform()
        declaration = transform.body_parts((self.parent / BODY).read_text())[1]
        declarations = re.sub(r"^#.*$", "", declaration, flags=re.MULTILINE)
        declarations = declarations.split("    TPipe& pipe,", 1)[1].rsplit(")", 1)[0]
        names = [re.findall(r"[A-Za-z_][A-Za-z_0-9]*", item)[-1]
                 for item in declarations.split(",")]
        kernel = (self.parent / "source/kernel.cpp").read_text()
        original = kernel.split("routed_body::Forward<format, true>(pipe,", 1)[1]
        original = original.split(");", 1)[0].split(",")
        self.assertEqual(len(names), len(original))
        for name, value in zip(names, original):
            expression = transform.BINDINGS[name]
            for before, after in (("arguments.weights", "w.values"),
                                  ("arguments.scratch", "body.values"),
                                  ("arguments.rows", "rows"),
                                  ("arguments.blocks", "blocks"),
                                  ("arguments.dispatch", "dispatch"),
                                  ("arguments.shared", "shared")):
                expression = expression.replace(before, after)

            def normalize(text):
                return re.sub(r"\([^)]*\*\)|\(GM_ADDR\)|[\s()]", "", text)

            self.assertEqual(normalize(expression), normalize(value), name)

    def test_host_ownership_and_active_checker_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(self.parent, output)
            for name in ("source/binding.cpp", "source/contract.h",
                         "body/source/routed_int4/host_validate.h",
                         "body/source/routed_int4/routed_int4_front.h",
                         "body/source/routed_int4/routed_int4_dispatch.h",
                         "body/source/routed_int4/routed_int4_active_list.h"):
                self.assertEqual((output / name).read_text(),
                                 rename((self.parent / name).read_text()), name)
            checker = (self.parent / "harness-v1/check.py").read_text()
            self.assertEqual((output / "check.py").read_text(), checker.replace(
                "torch.ops.tilert_ffn_active",
                "torch.ops.tilert_ffn_active_descriptor"))
            original = (self.parent / "source/kernel.cpp").read_text()
            kernel = (output / "source/kernel.cpp").read_text()
            for operation in ("SetSysWorkspace(", "SetSyncBaseAddr(sync);",
                              "ffn_bridge::device::EndPhase(pipe);"):
                self.assertEqual(kernel.count(operation), original.count(operation))
            self.assertEqual(report["descriptor_bytes"], 408)
            self.assertEqual(report["additional_gm_bytes_vs_active_parent"], 0)

    def test_full_width_addresses_and_shared_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(self.parent, output)
            binary = Path(directory) / "probe"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-DROUTED_ROW_BALANCED=1",
                "-DROUTED_SHARED_PIPELINE=1", "-DROUTED_INPUT_FRONT=1",
                "-DROUTED_ACTIVE_LIST=1", "-DSHARED_W8_GROUPED_QUANT=1",
                "-I", str(output / "source"),
                str(Path(__file__).with_name("descriptor_probe.cpp")),
                "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
