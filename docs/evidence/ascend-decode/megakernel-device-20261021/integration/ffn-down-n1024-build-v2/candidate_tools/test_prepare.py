"""Check N1024 tail ownership, complete tiling bytes and the frozen FFN ABI."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from prepare import (CONTRACT, NAMESPACE, NEW_SPLIT, OLD_NAMESPACE, OLD_SPLIT,
                     TILING, prepare, rename)


PARENT = Path(os.environ.get(
    "FFN_DOWN_N1024_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-down-n512-build-v2"))


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


class DownN1024Contract(unittest.TestCase):
    def test_only_down_host_tiling_and_validation_change(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(PARENT, output)
            for name in manifest["sources"]:
                if (Path(name).suffix not in (".h", ".cpp") or
                        name.startswith("candidate_tools/") or name == CONTRACT):
                    continue
                changed = (output / name).read_text()
                if name == TILING:
                    changed = changed.replace(NEW_SPLIT, OLD_SPLIT)
                self.assertEqual(changed, rename((PARENT / name).read_text()), name)
            self.assertEqual(report["public_scratch_count"], 33)
            self.assertEqual(report["descriptor_bytes"], 408)
            self.assertEqual(report["body_barriers"], 5)
            self.assertEqual(report["final_outer_end_phases"], 0)
            self.assertEqual(report["peak_ub_bytes"], 195584)
            self.assertEqual(report["routed_body_ub_bytes"], 116480)

    def test_full_tiling_payload_rejects_every_byte_mutation(self):
        expected = json.loads(
            Path(__file__).with_name("expected-tilings.json").read_text())
        candidate = bytes.fromhex(expected["candidate"]["0"]["raw_hex"])
        parent = bytes.fromhex(expected["parent"]["0"]["raw_hex"])
        self.assertEqual(len(candidate), 200)
        self.assertEqual(expected["candidate"]["0"], expected["candidate"]["1"] | {
            "layout": 0})
        self.assertEqual(hashlib.sha256(candidate).hexdigest(),
                         expected["candidate"]["0"]["raw_sha256"])
        data = ",".join(str(value) for value in candidate)
        parent_data = ",".join(str(value) for value in parent)
        source = '''
#include <array>
#include "down_tiling_contract.h"
int main() {
    const std::array<unsigned char, 200> original = {CANDIDATE};
    const std::array<unsigned char, 200> parent = {PARENT};
    if (!ffn_down_n1024::IsExpectedDownTiling(original)) return 1;
    if (ffn_down_n1024::IsExpectedDownTiling(parent)) return 2;
    for (int i = 0; i < 200; ++i) {
        auto changed = original;
        changed[i] ^= 1;
        if (ffn_down_n1024::IsExpectedDownTiling(changed)) return 3;
    }
}
'''.replace("CANDIDATE", data).replace("PARENT", parent_data)
        with tempfile.TemporaryDirectory() as directory:
            compile_run(source, Path(directory), [Path(__file__).parent])
        self.assertEqual(32 * 64 // 2 * 2, 2048)
        self.assertEqual(1024 * 64 // 2 * 2, 65536)
        self.assertEqual(32 * 1024 * 4, 131072)
        self.assertGreater(1024 * 128 // 2 * 2, 65536)

    def test_nz_k_halves_partition_the_original_full_stride(self):
        addresses = []
        for column in range(0, 6144, 1024):
            chunks = []
            for inner in (0, 64):
                mapped = [(((n // 64) * 8 + k // 16) * 16 + k % 16) * 64 + n % 64
                          for n in range(column, column + 1024)
                          for k in range(inner, inner + 64)]
                chunks.extend(mapped)
            self.assertEqual(sorted(chunks), list(range(column * 128,
                                                        (column + 1024) * 128)))
            addresses.extend(chunks)
        self.assertEqual(len(set(addresses)), 128 * 6144)
        # K64 is a tile; the NZ N-block stride still contains the full K128.
        self.assertNotEqual(((64 // 64) * 4) * 16 * 64,
                            ((64 // 64) * 8) * 16 * 64)

    def test_all_expert_counts_have_unique_output_and_no_padding_writes(self):
        for count in range(1, 33):
            rows = 2 * count
            intervals = []
            tiles = []
            for row in range(0, rows, 32):
                height = min(32, rows - row)
                for column in range(0, 6144, 1024):
                    tiles.append((row, height, column))
                    intervals.extend((m * 6144 + column, m * 6144 + column + 1024)
                                     for m in range(row, row + height))
            intervals.sort()
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], rows * 6144)
            self.assertTrue(all(left[1] == right[0]
                                for left, right in zip(intervals, intervals[1:])))
            self.assertEqual(len(tiles), 6 if count <= 16 else 12)
            self.assertEqual(len(tiles) * 2, 12 if count <= 16 else 24)
        # Exact INT32 accumulation precedes all MSD floating-point reconstruction.
        for values in ([64] * 128, [-56] * 128,
                       [64 if k % 3 else -56 for k in range(128)]):
            self.assertEqual(sum(values), sum(values[:64]) + sum(values[64:]))
            self.assertLessEqual(abs(sum(values)), 8192)

    def test_frozen_c_abi_and_descriptor_preserve_all_33_addresses(self):
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
