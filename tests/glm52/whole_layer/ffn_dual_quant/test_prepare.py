"""Check both original suffixes, phase seams and the real frozen C ABI."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import audit_contract
from prepare import (BODY, BRIDGE, FRONT, HELPER, KERNEL, NAMESPACE,
                     OLD_NAMESPACE, TRANSFORMS, prepare, quant_block, rename)


PARENT = Path(os.environ.get(
    "FFN_DUAL_QUANT_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-down-n512-build-v2"))
HERE = Path(__file__).resolve().parent


def statements(source):
    source = re.sub(r"//[^\n]*", "", source)
    return re.sub(r"\s+", "", source)


def slice_between(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


class DualQuantContracts(unittest.TestCase):
    def test_two_original_suffixes_are_preserved(self):
        shared = (PARENT / "bridge/source/norm_native.h").read_text()
        helper = (HERE / HELPER).read_text()
        prefix = slice_between(shared, "        DataCopy(bf16_, source, kHidden);",
                               "        Fence<HardEvent::S_V>();")
        self.assertIn(prefix.rstrip(), helper)
        start = shared.index("        Fence<HardEvent::S_V>();",
                             shared.index("const float maximum"))
        end = shared.index("\n    }\n\nprivate:", start)
        self.assertIn(shared[start:end], helper)

        route = (PARENT / FRONT).read_text()
        route = slice_between(route, "            const float quantum =",
                              "\n        }\n    }")
        route = route.replace("scale.GetValue(0)", "maximum")
        route = route.replace("output_gm[token * kHidden]", "routed_destination")
        route = route.replace("scale_gm[token]", "routed_scales")
        names = {"scale": "other_", "work": "work_", "values": "value_",
                 "integer": "routed_integer", "half_output": "routed_half",
                 "output": "codes_", "Fence": "RouteFence"}
        for old, new in names.items():
            route = re.sub(r"\b" + old + r"\b", new, route)
        candidate = slice_between(helper, "        const float quantum =",
                                  "\n        Fence<HardEvent::S_V>();")
        candidate = candidate.replace(
            "        auto routed_half = bf16_.ReinterpretCast<half>();\n", "")
        self.assertEqual(statements(route), statements(candidate))

    def test_only_four_phase_seams_change(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(PARENT, output)
            old = json.loads((PARENT / "manifest.json").read_text())
            for name in old["sources"]:
                path = Path(name)
                if path.suffix not in (".cpp", ".h"):
                    continue
                original = (PARENT / name).read_text()
                expected = TRANSFORMS[name](original) if name in TRANSFORMS else original
                self.assertEqual((output / name).read_text(), rename(expected), name)
            self.assertEqual(report["public_scratch_count"], 33)
            self.assertEqual(report["descriptor_bytes"], 408)
            self.assertEqual(report["body_barriers"], 5)

    def test_front_metadata_and_joins_unchanged(self):
        original = (PARENT / FRONT).read_text()
        changed = TRANSFORMS[FRONT](original)
        block = quant_block(original)
        wrapped = "        if constexpr (!input_quantized) {\n" + "\n".join(
            "    " + line if line else line for line in block.splitlines())
        restored = changed.replace(wrapped + "\n        }", block, 1).replace(
            "template <bool input_quantized = false, class Timing>",
            "template <class Timing>", 1)
        self.assertEqual(original, restored)
        body = TRANSFORMS[BODY]((PARENT / BODY).read_text())
        before = (PARENT / BODY).read_text()
        self.assertEqual(re.findall(r"timing\.Barrier\([^;]+;", body),
                         re.findall(r"timing\.Barrier\([^;]+;", before))
        bridge = TRANSFORMS[BRIDGE]((PARENT / BRIDGE).read_text())
        self.assertEqual(bridge.count("device::EndPhase(pipe);"), 3)
        kernel = TRANSFORMS[KERNEL]((PARENT / KERNEL).read_text())
        self.assertIn("routed_body::Forward<format, true, true>", kernel)
        self.assertEqual(kernel.count("PipeBarrier<PIPE_ALL>();"), 1)

    def test_resource_and_exact_row_ownership(self):
        self.assertEqual(audit_contract.check_row_stores()["all_geometry_cases"], 64)
        self.assertEqual(audit_contract.check_ub()["quant_ub_bytes"], 92192)
        self.assertEqual(audit_contract.check_reduction_tree()["leaves"], 6144)
        witness = audit_contract.check_distinct_suffixes()
        self.assertNotEqual(witness["routed_scale_bits"], witness["shared_scale_bits"])

    def test_independent_real_headers_have_identical_c_abi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            prepare(PARENT, candidate)
            outputs = []
            for index, (tree, namespace) in enumerate(
                    ((PARENT, OLD_NAMESPACE), (candidate, NAMESPACE))):
                binary = root / f"probe-{index}"
                command = [
                    "g++", "-std=c++17", "-DROUTED_ACTIVE_LIST",
                    "-DROUTED_INPUT_FRONT", "-DROUTED_ROW_BALANCED",
                    "-DROUTED_SHARED_PIPELINE", f"-DFFN_NAMESPACE={namespace}",
                    f"-DFFN_LAUNCH={namespace}_launch", "-I", str(tree),
                    str(HERE / "abi_probe.cpp"), "-o", str(binary)]
                subprocess.run(command, check=True, capture_output=True)
                outputs.append(subprocess.check_output([str(binary)], text=True))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(len(outputs[0].splitlines()), 8)
            self.assertTrue(all(len(line) == 816 for line in outputs[0].splitlines()))


if __name__ == "__main__":
    unittest.main()
