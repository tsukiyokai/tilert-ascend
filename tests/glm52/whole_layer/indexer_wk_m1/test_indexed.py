"""Check packed M1 WK operands, exact row ownership and the frozen dispatch."""

from array import array
import os
from pathlib import Path
import unittest

from build_indexed import INCLUDE, NEW_BRANCH, OLD_BRANCH, transform


class IndexedM1Contract(unittest.TestCase):
    def test_packed_operand_matches_nk_microprobe(self):
        packed = array("I", (
            (tile * 32 + column) * 6144 + inner
            for tile in range(5) for inner in range(6144) for column in range(32)))
        for tile in range(10):
            bank, half = divmod(tile, 2)
            for start in range(0, 6144, 128):
                base = bank * 6144 * 32 + start * 32 + half * 16
                l1 = array("I", (packed[base + inner * 32 + column]
                                 for inner in range(128) for column in range(16)))
                l0 = array("I", (l1[fractal * 256 + inner * 16 + column]
                                 for fractal in range(8) for column in range(16)
                                 for inner in range(16)))
                expected = array("I", ((tile * 16 + column) * 6144
                                       + start + fractal * 16 + inner
                                       for fractal in range(8) for column in range(16)
                                       for inner in range(16)))
                self.assertEqual(l0, expected)

    def test_every_valid_packed_output_written_once_and_padding_untouched(self):
        for blocks in range(1, 25):
            for capacity in (16, 32):
                with self.subTest(blocks=blocks, capacity=capacity):
                    writes = [0] * (5 * capacity * 32)
                    for core in range(blocks):
                        for task in range(core, 10, blocks):
                            bank, half = divmod(task, 2)
                            base = bank * capacity * 32 + half * 16
                            for column in range(16):
                                writes[base + column] += 1
                    expected = [int(row == 0) for bank in range(5)
                                for row in range(capacity) for column in range(32)]
                    self.assertEqual(writes, expected)
                    self.assertEqual(sum(writes), 160)

    def test_gemv_a_load_contains_all_consumed_values(self):
        for start in range(0, 6144, 128):
            vector = [None] * 256
            for inner in range(128):
                vector[(inner // 16) * 16 + inner % 16] = start + inner
            self.assertEqual(vector[:128], list(range(start, start + 128)))
            self.assertEqual(vector[128:], [None] * 128)

    def test_only_exact_m1_dispatch_is_inserted(self):
        default = Path("/tmp/tilert-megakernel-validation-20261021/attention-wk-m12-build-v1/")
        default /= "source/tests/glm52/whole_layer/indexer_front_native.h"
        parent = Path(os.environ.get("WK_M1_PARENT_FRONT", default))
        original = parent.read_text()
        result = transform(original)
        restored = result.replace(INCLUDE + '\n#include "indexer_wk_m1_native.h"', INCLUDE)
        restored = restored.replace(NEW_BRANCH, OLD_BRANCH)
        self.assertEqual(restored, original)
        self.assertEqual(result.count("if (c.rows == 1)"), 1)
        for width in (12, 32):
            self.assertEqual(result.count(f"indexer_wk_m{width}::device::Project"), 1)
        with self.assertRaises(RuntimeError):
            transform(result)


if __name__ == "__main__":
    unittest.main()
