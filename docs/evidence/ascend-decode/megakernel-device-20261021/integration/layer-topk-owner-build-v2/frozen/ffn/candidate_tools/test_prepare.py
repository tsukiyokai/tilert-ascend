"""Check common ancestry, original shared arithmetic and the combined C ABI."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from prepare import (BODY, GATE_ALIAS, INCLUDE, NAMESPACE, OLD_ALIAS, SHARED,
                     SHARED_HEADER, digest, prepare, rename, verify_common_body)


PARENT = Path(os.environ.get(
    "FFN_ORIGINAL_SHARED_GU_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-original-shared-terminal-runtime-v1/build-v1"))
GATE = Path(os.environ.get(
    "FFN_ORIGINAL_SHARED_GU_GATE", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-gu-static-build-v2"))


class CombinedGateContract(unittest.TestCase):
    def test_common_body_is_an_exact_reversal(self):
        self.assertEqual(len(verify_common_body(PARENT, GATE)), 64)
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory)
            body = gate / BODY
            body.parent.mkdir(parents=True)
            body.write_text((GATE / BODY).read_text() + "\n// different body\n")
            with self.assertRaisesRegex(ValueError, "identical GU bodies"):
                verify_common_body(PARENT, gate)

    def test_only_gu_alias_and_include_change_in_body(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, GATE, output)
            changed = (output / BODY).read_text()
            restored = changed.replace(
                GATE_ALIAS.replace("ffn_gu_static", NAMESPACE), OLD_ALIAS).replace(INCLUDE, "")
            self.assertEqual(restored, rename((PARENT / BODY).read_text()))
            for fragment in ("mm.SetOrgShape(matrix_rows, columns, inner);",
                             "mm.SetSingleShape(matrix_rows, columns, inner);",
                             "const int64_t matrix_rows = pad_rows ? GateMatrixRows(count) : 2 * count;"):
                self.assertIn(fragment, changed)

    def test_shared_math_and_other_compiled_sources_are_unchanged(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(PARENT, GATE, output)
            self.assertEqual(digest(output / SHARED), SHARED_HEADER)
            self.assertEqual(report["shared_header_sha256"], SHARED_HEADER)
            for name in manifest["sources"]:
                if Path(name).suffix in (".h", ".cpp") and name != BODY:
                    self.assertEqual((output / name).read_text(),
                                     rename((PARENT / name).read_text()), name)
            self.assertFalse((output / "check.py").exists())
            self.assertEqual((output / "parent-check.py").read_bytes(),
                             (PARENT / "check.py").read_bytes())

    def test_static_values_keep_dynamic_rows_and_both_weight_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, GATE, output)
            header = (output / "source/gu_static_tiling.h").read_text()
            expected = (GATE / "source/gu_static_tiling.h").read_text().replace(
                "ffn_gu_static", NAMESPACE).replace("FFN_GU_STATIC", NAMESPACE.upper())
            self.assertEqual(header, expected)
            values = {name: int(value) for name, value in
                      re.findall(r"value\.(\w+) = (-?\d+);", header)}
            self.assertEqual(len(values), 21)
            self.assertNotIn("M", values)
            self.assertNotIn("singleCoreM", values)
            self.assertIn("value.cfg = ::CFG_NORM;", header)
            tilings = [json.loads(line) for line in
                       (GATE / "candidate_tools/gate-tiling.jsonl").read_text().splitlines()]
            gates = [record for record in tilings if record["projection"] == 0]
            self.assertEqual({record["layout"] for record in gates}, {0, 1})
            for record in gates:
                for name, value in values.items():
                    self.assertEqual(record[name], value, (record["layout"], name))

    def test_actual_frozen_descriptor_and_c_abi(self):
        probe = r'''
#include <cstdio>
#include <initializer_list>
#include "source/descriptor.h"
namespace candidate = FFN_NAMESPACE;
using Launch = void (*)(void*, uint64_t, void*, void*, void*, void*, void*,
                       void**, void**, void**, routed_int4::pipeline::Input,
                       int64_t, int64_t, int64_t, float);
static_assert(std::is_same_v<decltype(&FFN_LAUNCH), Launch>);
int main() {
    void* weights[6];
    void* bridge[12];
    void* body[21];
    for (uint64_t i = 0; i < 6; ++i) weights[i] = (void*)(0x1000 + i * 32);
    for (uint64_t i = 0; i < 12; ++i) bridge[i] = (void*)(0x2000 + i * 32);
    for (uint64_t i = 0; i < 21; ++i) body[i] = (void*)(0x3000 + i * 32);
    routed_int4::pipeline::Input shared{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                     32, 11, 12, 13, 14};
    static_assert(sizeof(shared) == 112);
    static_assert(candidate::kBridgeBuffers + candidate::kBodyBuffers == 33);
    for (const int rows : {1, 2, 12, 32}) {
        for (const int blocks : {16, 24}) {
            auto args = candidate::MakeBodyArguments(
                weights, bridge, body, shared, rows, blocks);
            static_assert(sizeof(args) == 408);
            const auto* bytes = reinterpret_cast<const unsigned char*>(&args);
            for (size_t i = 0; i < sizeof(args); ++i) std::printf("%02x", bytes[i]);
            std::puts("");
        }
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidate"
            prepare(PARENT, GATE, output)
            results = []
            for index, (tree, namespace) in enumerate((
                    (PARENT, "ffn_original_shared_terminal"), (output, NAMESPACE))):
                source = root / f"probe{index}.cpp"
                source.write_text(probe.replace("FFN_NAMESPACE", namespace).replace(
                    "FFN_LAUNCH", namespace + "_launch"))
                binary = root / f"probe{index}"
                result = subprocess.run([
                    "g++", "-std=c++17", "-DROUTED_ACTIVE_LIST",
                    "-DROUTED_INPUT_FRONT", "-DROUTED_ROW_BALANCED",
                    "-DROUTED_SHARED_PIPELINE", "-I", str(tree), str(source),
                    "-o", str(binary),
                ], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                results.append(subprocess.check_output([str(binary)]))
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
