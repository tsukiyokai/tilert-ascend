"""Protect FFN publication boundaries, references and host launch contract."""

import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from prepare import (
    KERNEL, NEW_EXIT, OLD_EXIT, bind_checker, bind_kernel, prepare, rename,
)


PARENT = Path(os.environ.get(
    "FFN_TOKEN_TERMINAL_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-token-split-build-v2"))


class TerminalContract(unittest.TestCase):
    def test_frozen_descriptor_and_public_c_abi(self):
        probe = r'''
#include <cstdio>
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
    auto args = candidate::MakeBodyArguments(weights, bridge, body, shared, 12, 24);
    static_assert(sizeof(args) == 408 && sizeof(shared) == 112);
    static_assert(candidate::kBridgeBuffers + candidate::kBodyBuffers == 33);
    static_assert(routed_int4::pipeline::kRoutedResidentUB +
                  routed_int4::pipeline::kSharedResidentUB +
                  routed_int4::pipeline::kSharedEpilogueUB + 3072 == 116480);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&args);
    for (size_t i = 0; i < sizeof(args); ++i) std::printf("%02x", bytes[i]);
    std::puts("");
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            prepare(PARENT, candidate)
            outputs = []
            for index, (tree, namespace) in enumerate((
                    (PARENT, "ffn_token_split"),
                    (candidate, "ffn_token_terminal"))):
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

    def test_only_final_global_completion_changes(self):
        original = (PARENT / KERNEL).read_text()
        transformed = bind_kernel(original)
        self.assertEqual(transformed.replace(NEW_EXIT, OLD_EXIT), original)
        self.assertIn(NEW_EXIT + "\n}\n", transformed)
        for invalid in (original.replace(OLD_EXIT, ""), original + OLD_EXIT):
            with self.assertRaises(ValueError):
                bind_kernel(invalid)

    def test_callable_math_and_all_host_arguments_are_unchanged(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            prepare(PARENT, output)
            for name in manifest["sources"]:
                if Path(name).suffix not in (".h", ".cpp") or name == KERNEL:
                    continue
                self.assertEqual((output / name).read_text(),
                                 rename((PARENT / name).read_text()), name)
            body = output / "body/source/routed_int4/routed_body_native.h"
            self.assertIn("#ifdef ROUTED_ENABLE_PROFILE\n#error", body.read_text())

    def test_reference_chains_and_queued_input_producers_are_preserved(self):
        original = ast.parse((PARENT / "check.py").read_text())
        generated = bind_checker((PARENT / "check.py").read_text())
        candidate = ast.parse(generated)

        def function(tree, name):
            return next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == name)

        for name in ("separate", "production", "all_values", "body_values"):
            self.assertEqual(ast.dump(function(original, name)),
                             ast.dump(function(candidate, name)), name)
        composed = function(candidate, "composed")
        self.assertEqual(len(composed.body), 2)
        self.assertEqual(ast.unparse(composed.body[0].value.args[0]), "hidden.clone()")
        self.assertEqual(ast.unparse(composed.body[0].value.args[1]), "residual.clone()")
        self.assertEqual(ast.unparse(composed.body[1].body[0]), "destination.copy_(value)")
        self.assertIn("for value in (*terminal_outputs, *consumed):", generated)
        self.assertIn("value.fill_(float('nan'))", generated)


if __name__ == "__main__":
    unittest.main()
