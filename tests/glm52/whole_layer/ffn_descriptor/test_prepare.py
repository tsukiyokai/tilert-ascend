"""Check descriptor mapping and statement preservation against frozen sources."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from prepare import BINDINGS, BODY, bind_body, body_parts, prepare


class DescriptorSeam(unittest.TestCase):
    parent = Path(os.environ.get("FFN_DESCRIPTOR_PARENT", str(
        Path(__file__).resolve().parents[4]
        / "docs/evidence/ascend-decode/megakernel-device-20261021"
        / "a3-194/ffn-composed-build-v2")))

    def test_numerical_statements_and_all_barriers_are_retained(self):
        original = (self.parent / BODY).read_text()
        candidate, numerical, _ = bind_body(original)
        actual = body_parts(candidate)[2]
        for name, expression in sorted(BINDINGS.items(), key=lambda item: -len(item[1])):
            actual = actual.replace(expression, name)
        self.assertEqual(actual, numerical)
        self.assertEqual(actual.count("timing.Barrier("), 5)
        for name in ("routed_int4_front.h", "routed_int4_dispatch.h"):
            helper = self.parent / Path(BODY).with_name(name)
            self.assertEqual(helper.read_text().count("timing.Barrier(0);"), 1)
        prefix, _, _, suffix = body_parts(original)
        actual_prefix, _, _, actual_suffix = body_parts(candidate)
        self.assertEqual(actual_prefix.replace(
            '\n#include "../../../source/descriptor.h"', ""), prefix)
        self.assertEqual(actual_suffix, suffix)

    def test_parameter_fields_match_original_call_arguments(self):
        source = (self.parent / BODY).read_text()
        declaration = body_parts(source)[1]
        declarations = re.sub(r"^#.*$", "", declaration, flags=re.MULTILINE)
        declarations = declarations.split("    TPipe& pipe,", 1)[1].rsplit(")", 1)[0]
        names = [re.findall(r"[A-Za-z_][A-Za-z_0-9]*", item)[-1]
                 for item in declarations.split(",")]
        kernel = (self.parent / "source/kernel.cpp").read_text()
        arguments = kernel.split("routed_body::Forward<format, true>(pipe,", 1)[1]
        arguments = arguments.split(");", 1)[0].split(",")
        self.assertEqual(len(arguments), len(names))
        for name, original in zip(names, arguments):
            expression = BINDINGS[name]
            for before, after in (("arguments.weights", "w.values"),
                                  ("arguments.scratch", "body.values"),
                                  ("arguments.rows", "rows"),
                                  ("arguments.blocks", "blocks"),
                                  ("arguments.dispatch", "dispatch"),
                                  ("arguments.shared", "shared")):
                expression = expression.replace(before, after)
            # Compare the address/constant selected; types are checked by CANN.
            normalize = lambda value: re.sub(r"\([^)]*\*\)|\(GM_ADDR\)|[\s()]", "", value)
            self.assertEqual(normalize(expression), normalize(original), name)

    def test_host_validators_and_outer_barriers_do_not_change(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "descriptor"
            report = prepare(self.parent, output)
            for name in ("source/binding.cpp", "source/contract.h",
                         "bridge/source/ffn_bridge/host_validate.h",
                         "body/source/routed_int4/host_validate.h",
                         "body/source/routed_int4/routed_int4_front.h",
                         "body/source/routed_int4/routed_int4_dispatch.h"):
                expected = (self.parent / name).read_text().replace(
                    "ffn_composed", "ffn_descriptor").replace(
                    "FFN_COMPOSED", "FFN_DESCRIPTOR")
                self.assertEqual((output / name).read_text(), expected)
            original = (self.parent / "source/kernel.cpp").read_text()
            kernel = (output / "source/kernel.cpp").read_text()
            for operation in ("InitSocState();", "SetSyncBaseAddr(sync);",
                              "ffn_bridge::device::EndPhase(pipe);"):
                self.assertEqual(kernel.count(operation), original.count(operation))
            self.assertEqual(kernel.count("SetSysWorkspace("), 2)
            self.assertEqual(report["descriptor_bytes"], 392)
            self.assertEqual(report["additional_gm_bytes"], 0)
            for name, expected in report["sources"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)

    def test_portable_cpp_captures_full_addresses_and_shared_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "descriptor"
            prepare(self.parent, output)
            binary = Path(directory) / "probe"
            result = subprocess.run([
                "g++", "-std=c++17", "-O2", "-DROUTED_ROW_BALANCED=1",
                "-DROUTED_SHARED_PIPELINE=1", "-DROUTED_INPUT_FRONT=1",
                "-DSHARED_W8_GROUPED_QUANT=1", "-I", str(output / "source"),
                str(Path(__file__).with_name("descriptor_probe.cpp")), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            subprocess.run([str(binary)], check=True)

    def test_wrong_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            parent.mkdir()
            (parent / "manifest.json").write_text(json.dumps({
                "artifacts": {"ffn_composed.so": "unknown"}}))
            with self.assertRaisesRegex(ValueError, "validated uninstrumented"):
                prepare(parent, Path(directory) / "candidate")


if __name__ == "__main__":
    unittest.main()
