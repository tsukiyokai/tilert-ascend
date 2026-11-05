"""Check the frozen GU transformation, dynamic rows and complete public ABI."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from prepare import BODY, INCLUDE, NEW_ALIAS, OLD_ALIAS, bind_body, prepare, rename


PARENT = Path(os.environ.get(
    "FFN_GU_STATIC_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-token-terminal-build-v1"))


class StaticGateContract(unittest.TestCase):
    def test_constants_match_both_retained_weight_layout_tilings(self):
        source = Path(__file__).with_name("gu_static_tiling.h").read_text()
        constants = {name: int(value) for name, value in
                     re.findall(r"value\.(\w+) = (-?\d+);", source)}
        self.assertEqual(len(constants), 21)
        records = [json.loads(line) for line in
                   Path(__file__).with_name("gate-tiling.jsonl")
                   .read_text().splitlines()]
        gates = [record for record in records if record["projection"] == 0]
        self.assertEqual({record["layout"] for record in gates}, {0, 1})
        for record in gates:
            for name, value in constants.items():
                self.assertEqual(value, record[name], (record["layout"], name))
        self.assertNotIn("M", constants)
        self.assertNotIn("singleCoreM", constants)
        self.assertIn("value.cfg = AscendC::CFG_NORM;", source)
        self.assertNotRegex(source, r"value\.cfg\.\w+\s*=")

    def test_only_gate_template_and_its_include_change(self):
        original = (PARENT / BODY).read_text()
        changed = bind_body(original)
        restored = changed.replace(NEW_ALIAS, OLD_ALIAS).replace(INCLUDE, "")
        self.assertEqual(restored, original)
        self.assertIn("mm.SetOrgShape(matrix_rows, columns, inner);", changed)
        self.assertIn("mm.SetSingleShape(matrix_rows, columns, inner);", changed)
        self.assertIn("const int64_t matrix_rows = pad_rows ? GateMatrixRows(count) : 2 * count;",
                      changed)
        self.assertIn("using Projection = Matmul<APacked,", changed)
        for invalid in (original.replace(OLD_ALIAS, ""), original + OLD_ALIAS):
            with self.assertRaises(ValueError):
                bind_body(invalid)

    def test_all_other_device_and_host_sources_are_unchanged(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, output)
            for name in manifest["sources"]:
                if Path(name).suffix not in (".h", ".cpp") or name == BODY:
                    continue
                self.assertEqual((output / name).read_text(),
                                 rename((PARENT / name).read_text()), name)

    def test_checker_preserves_both_oracles_and_immediate_consumers(self):
        original = (PARENT / "check.py").read_text()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, output)
            changed = (output / "check.py").read_text()
            self.assertEqual(ast.dump(ast.parse(changed)),
                             ast.dump(ast.parse(rename(original))))
            self.assertIn("destination.copy_(value)", changed)
            self.assertIn("for value in (*terminal_outputs, *consumed):", changed)

    def test_frozen_descriptor_and_public_c_abi(self):
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
            candidate = root / "candidate"
            prepare(PARENT, candidate)
            outputs = []
            for index, (tree, namespace) in enumerate((
                    (PARENT, "ffn_token_terminal"),
                    (candidate, "ffn_gu_static"))):
                source = root / f"probe{index}.cpp"
                source.write_text(probe.replace("FFN_NAMESPACE", namespace)
                                  .replace("FFN_LAUNCH", namespace + "_launch"))
                binary = root / f"probe{index}"
                compiled = subprocess.run([
                    "g++", "-std=c++17", "-DROUTED_ACTIVE_LIST",
                    "-DROUTED_INPUT_FRONT", "-DROUTED_ROW_BALANCED",
                    "-DROUTED_SHARED_PIPELINE", "-I", str(tree), str(source),
                    "-o", str(binary),
                ], check=False, capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                outputs.append(subprocess.check_output([str(binary)]))
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
