"""Check single-buffer Down records, integer sums, ownership and the frozen ABI."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from prepare import (CONTRACT, NAMESPACE, NEW_CHECK, OLD_CHECK, OLD_NAMESPACE,
                     PLAN_INCLUDE, TILING, prepare, rename)


PARENT = Path(os.environ.get(
    "FFN_DOWN_K128_PARENT", "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-down-n1024-build-v2"))


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


class DownK128Contract(unittest.TestCase):
    def test_every_parent_device_consumer_is_unchanged(self):
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
                    changed = changed.replace(rename(NEW_CHECK), rename(OLD_CHECK))
                    changed = changed.replace(PLAN_INCLUDE, "")
                self.assertEqual(changed, rename((PARENT / name).read_text()), name)
            self.assertEqual(report["public_scratch_count"], 33)
            self.assertEqual(report["descriptor_bytes"], 408)
            self.assertEqual(report["body_barriers"], 5)
            self.assertEqual(report["final_outer_end_phases"], 0)
            self.assertEqual(report["peak_ub_bytes"], 195584)
            self.assertEqual(report["routed_body_ub_bytes"], 116480)

    def test_full_records_reject_mutation_and_map_only_six_named_fields(self):
        expected = json.loads(
            Path(__file__).with_name("expected-tilings.json").read_text())
        candidate = bytes.fromhex(expected["candidate"]["0"]["raw_hex"])
        parent = bytes.fromhex(expected["parent"]["0"]["raw_hex"])
        for label in ("parent", "candidate"):
            self.assertEqual(expected[label]["0"], expected[label]["1"] | {"layout": 0})
            payload = bytes.fromhex(expected[label]["0"]["raw_hex"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(),
                             expected[label]["0"]["raw_sha256"])
        data = ",".join(str(value) for value in candidate)
        parent_data = ",".join(str(value) for value in parent)
        source = r'''
#include <array>
#include <cstddef>
#include <cstring>
#include "down_plan.h"
#include "down_tiling_contract.h"
struct Cube {
    int32_t prefix[10], baseK, depthA1, depthB1, gap0[13];
    int32_t stepKa, stepKb, gap1[3], dbL0B, suffix[18];
};
static_assert(sizeof(Cube) == 200);
static_assert(offsetof(Cube, baseK) == 40 && offsetof(Cube, depthA1) == 44);
static_assert(offsetof(Cube, depthB1) == 48 && offsetof(Cube, stepKa) == 104);
static_assert(offsetof(Cube, stepKb) == 108 && offsetof(Cube, dbL0B) == 124);
int main() {
    const std::array<unsigned char, 200> candidate = {CANDIDATE};
    const std::array<unsigned char, 200> parent = {PARENT};
    if (!ffn_down_k128::IsExpectedDownTiling(candidate)) return 1;
    if (!ffn_down_k128::IsParentDownTiling(parent)) return 2;
    if (ffn_down_k128::IsExpectedDownTiling(parent)) return 3;
    if (ffn_down_k128::IsParentDownTiling(candidate)) return 4;
    for (int i = 0; i < 200; ++i) {
        auto changed = candidate; changed[i] ^= 1;
        if (ffn_down_k128::IsExpectedDownTiling(changed)) return 5;
        changed = parent; changed[i] ^= 1;
        if (ffn_down_k128::IsParentDownTiling(changed)) return 6;
    }
    Cube cube;
    std::memcpy(&cube, parent.data(), sizeof(cube));
    ffn_down_k128::ApplyDownPlan(cube, {128, 2, 6, 1, 1, 1});
    if (std::memcmp(&cube, candidate.data(), sizeof(cube))) return 7;
}
'''.replace("CANDIDATE", data).replace("PARENT", parent_data)
        with tempfile.TemporaryDirectory() as directory:
            compile_run(source, Path(directory), [Path(__file__).parent])
        offsets = {40, 44, 48, 104, 108, 124}
        self.assertEqual({i for i in range(0, 200, 4)
                          if parent[i:i + 4] != candidate[i:i + 4]}, offsets)
        self.assertEqual(2 * 32 * 128 // 2 + 6 * 1024 * 128 // 2, 397312)
        self.assertEqual(1024 * 128 // 2, 65536)
        self.assertEqual(32 * 1024 * 4, 131072)
        self.assertEqual((2 - 1) & (1 - 1), 0)

    def test_full_k_operand_matches_two_parent_nz_tiles(self):
        for column in range(0, 6144, 1024):
            def address(n, k):
                return (((n // 64) * 8 + k // 16) * 16 + k % 16) * 64 + n % 64
            single = {address(n, k) for n in range(column, column + 1024)
                      for k in range(128)}
            halves = [{address(n, k) for n in range(column, column + 1024)
                       for k in range(begin, begin + 64)} for begin in (0, 64)]
            self.assertFalse(halves[0] & halves[1])
            self.assertEqual(halves[0] | halves[1], single)
            self.assertEqual(single, set(range(column * 128, (column + 1024) * 128)))

    def test_all_counts_preserve_output_ownership_and_exact_integer_sum(self):
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
        rng = random.Random(729)
        operands = [([-8] * 128, [-8] * 128), ([-8] * 128, [7] * 128)]
        operands += [([rng.randrange(-8, 8) for _ in range(128)],
                      [rng.randrange(-8, 8) for _ in range(128)]) for _ in range(200)]
        for a, b in operands:
            products = [x * y for x, y in zip(a, b)]
            self.assertEqual(sum(products), sum(products[:64]) + sum(products[64:]))
            self.assertLessEqual(sum(abs(x) for x in products), 8192)

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
