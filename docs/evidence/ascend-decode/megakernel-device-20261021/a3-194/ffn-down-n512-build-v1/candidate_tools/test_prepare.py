"""Check N512 geometry, exact source scope and the frozen FFN ABI."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from prepare import (CHECK_TILING, CONTRACT, INCLUDE, NAMESPACE, OLD_NAMESPACE,
                     SET_SPLIT, TILING, prepare, rename)


PARENT = Path(os.environ.get(
    "FFN_DOWN_N512_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-original-shared-gu-static-build-v1"))


def compile_run(source, root, includes):
    cpp = root / "probe.cpp"
    binary = root / "probe"
    cpp.write_text(source)
    argv = ["g++", "-std=c++17", "-DROUTED_ACTIVE_LIST", "-DROUTED_INPUT_FRONT",
            "-DROUTED_ROW_BALANCED", "-DROUTED_SHARED_PIPELINE"]
    for directory in includes:
        argv.extend(("-I", str(directory)))
    subprocess.run(argv + [str(cpp), "-o", str(binary)], check=True,
                   capture_output=True, text=True)
    return subprocess.check_output([str(binary)])


class DownN512Contract(unittest.TestCase):
    def test_only_host_tiling_changes_after_public_symbol_normalization(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(PARENT, output)
            for name in manifest["sources"]:
                if Path(name).suffix not in (".h", ".cpp"):
                    continue
                changed = (output / name).read_text()
                if name == TILING:
                    changed = changed.replace(INCLUDE, "").replace(
                        SET_SPLIT, "").replace(CHECK_TILING, "")
                self.assertEqual(changed, rename((PARENT / name).read_text()), name)
            self.assertEqual(report["public_scratch_count"], 33)
            self.assertEqual(report["descriptor_bytes"], 408)
            self.assertEqual(report["body_barriers"], 5)
            self.assertEqual(report["final_outer_end_phases"], 0)

    def test_all_tiling_fields_and_resource_overflows_are_rejected(self):
        header = Path(__file__).with_name(CONTRACT).read_text()
        fields = re.findall(r"cube\.(\w+) == (\d+)", header)
        self.assertEqual(len(fields), 23)
        declarations = "\n".join(f"int {name} = {value};" for name, value in fields)
        mutations = "\n".join(
            f"{{ auto changed = original; ++changed.{name}; "
            "if (ffn_down_n512::IsExpectedDownTiling(changed)) return 1; }"
            for name, _ in fields)
        source = ('#include "down_tiling_contract.h"\nstruct Cube {\n' + declarations +
                  '\n};\nint main() { Cube original;\n'
                  'if (!ffn_down_n512::IsExpectedDownTiling(original)) return 2;\n' +
                  mutations + '\n}\n')
        with tempfile.TemporaryDirectory() as directory:
            compile_run(source, Path(directory), [Path(__file__).parent])
        self.assertEqual(64 * 128 // 2 * 2, 8192)
        self.assertEqual(512 * 128 // 2 * 2, 65536)
        self.assertEqual(64 * 512 * 4, 131072)
        self.assertGreater(64 * 1024 * 4, 131072)
        self.assertGreater(512 * 256 // 2 * 2, 65536)

    def test_nz_tiles_and_compact_output_rows_form_exact_partitions(self):
        # Physical NZ is [N/64, K/16, 16, 64] in INT4 elements.
        addresses = []
        for tile in range(12):
            mapped = [(((n // 64) * 8 + k // 16) * 16 + k % 16) * 64 + n % 64
                      for n in range(tile * 512, (tile + 1) * 512)
                      for k in range(128)]
            self.assertEqual(sorted(mapped), list(range(tile * 512 * 128,
                                                        (tile + 1) * 512 * 128)))
            addresses.extend(mapped)
        self.assertEqual(len(set(addresses)), 128 * 6144)
        for count in range(1, 33):
            self.assertEqual((2 * count + 63) // 64, 1)
            intervals = [(row * 6144 + column, row * 6144 + column + 512)
                         for row in range(2 * count) for column in range(0, 6144, 512)]
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], 2 * count * 6144)
            self.assertTrue(all(left[1] == right[0]
                                for left, right in zip(intervals, intervals[1:])))

    def test_descriptor_and_c_abi_use_the_same_33_addresses(self):
        source = r'''
#include <cstdio>
#include <initializer_list>
#include "source/descriptor.h"
namespace candidate = FFN_NAMESPACE;
using Launch = void (*)(void*, uint64_t, void*, void*, void*, void*, void*,
                       void**, void**, void**, routed_int4::pipeline::Input,
                       int64_t, int64_t, int64_t, float);
static_assert(std::is_same_v<decltype(&FFN_LAUNCH), Launch>);
int main() {
    void* weights[6]; void* bridge[12]; void* body[21];
    for (uint64_t i = 0; i < 6; ++i) weights[i] = (void*)(0x1000 + i * 32);
    for (uint64_t i = 0; i < 12; ++i) bridge[i] = (void*)(0x2000 + i * 32);
    for (uint64_t i = 0; i < 21; ++i) body[i] = (void*)(0x3000 + i * 32);
    routed_int4::pipeline::Input shared{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                     32, 11, 12, 13, 14};
    static_assert(sizeof(shared) == 112);
    static_assert(candidate::kBridgeBuffers + candidate::kBodyBuffers == 33);
    for (const int rows : {1, 2, 12, 32}) {
        for (const int blocks : {16, 24}) {
            auto args = candidate::MakeBodyArguments(weights, bridge, body, shared, rows, blocks);
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
            prepare(PARENT, output)
            results = []
            for index, (tree, namespace) in enumerate(((PARENT, OLD_NAMESPACE),
                                                       (output, NAMESPACE))):
                probe_root = root / str(index)
                probe_root.mkdir()
                probe = source.replace("FFN_NAMESPACE", namespace).replace(
                    "FFN_LAUNCH", namespace + "_launch")
                results.append(compile_run(probe, probe_root, [tree]))
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
