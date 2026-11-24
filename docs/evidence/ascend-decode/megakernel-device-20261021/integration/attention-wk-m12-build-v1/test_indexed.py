"""Validate packed WK operands, M12 tail ownership and the exact dispatch seam."""

from array import array
import os
from pathlib import Path
import unittest

from build_indexed import INCLUDE, NEW_BRANCH, OLD_BRANCH, transform


class IndexedM12Contract(unittest.TestCase):
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
                            for row in range(12):
                                for column in range(16):
                                    writes[base + row * 32 + column] += 1
                    expected = [int(row < 12) for bank in range(5)
                                for row in range(capacity) for column in range(32)]
                    self.assertEqual(writes, expected)
                    self.assertEqual(sum(writes), 1920)

    def test_only_new_exact_m12_dispatch_is_inserted(self):
        relative = ("docs/evidence/ascend-decode/megakernel-device-20261021/integration/"
                    "runtime212-build-v1/attention/indexed/source/tests/glm52/whole_layer/"
                    "indexer_front_native.h")
        parent = Path(os.environ.get("WK_M12_PARENT_FRONT", Path(__file__).resolve().parents[4] / relative))
        original = parent.read_text()
        result = transform(original)
        restored = result.replace(INCLUDE + '\n#include "indexer_wk_m12_native.h"', INCLUDE)
        restored = restored.replace(NEW_BRANCH, OLD_BRANCH)
        self.assertEqual(restored, original)
        self.assertEqual(result.count("if (c.rows == 12)"), 1)
        self.assertEqual(result.count("indexer_wk_m32::device::Project"), 1)
        with self.assertRaises(RuntimeError):
            transform(result)


if __name__ == "__main__":
    unittest.main()
