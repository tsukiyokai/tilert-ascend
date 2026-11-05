"""Check the packed-weight transform and complete M32 output ownership."""

from array import array
import unittest


class PackedWkContract(unittest.TestCase):
    def test_every_loaded_b_operand_matches_original_nk(self):
        # Independent logical identifiers make row/column swaps observable.
        packed = array("I", (
            (tile * 32 + column) * 6144 + inner
            for tile in range(5)
            for inner in range(6144)
            for column in range(32)
        ))
        checked = 0
        for column_tile in range(10):
            tile, half = divmod(column_tile, 2)
            for inner_start in range(0, 6144, 128):
                base = tile * 6144 * 32 + inner_start * 32 + half * 16
                # ND2NZ n=128,d=16: eight [K0,N0] fractals.
                l1 = array("I", (
                    packed[base + inner * 32 + column]
                    for inner in range(128) for column in range(16)
                ))
                # LoadData transposes each 16x16 fractal, yielding L0B.
                l0 = array("I", (
                    l1[fractal * 256 + inner * 16 + column]
                    for fractal in range(8)
                    for column in range(16)
                    for inner in range(16)
                ))
                expected = array("I", (
                    (column_tile * 16 + column) * 6144
                    + inner_start + fractal * 16 + inner
                    for fractal in range(8)
                    for column in range(16)
                    for inner in range(16)
                ))
                self.assertEqual(l0, expected)
                checked += len(l0)
        self.assertEqual(checked, 160 * 6144)

    def test_physical_groups_own_every_output_once(self):
        for blocks in range(1, 25):
            writes = [0] * (5 * 32 * 32)
            coordinates = set()
            for core in range(blocks):
                for task in range(core, 20, blocks):
                    column_tile, row_tile = divmod(task, 2)
                    tile, half = divmod(column_tile, 2)
                    base = tile * 32 * 32 + row_tile * 16 * 32 + half * 16
                    for row in range(16):
                        for column in range(16):
                            writes[base + row * 32 + column] += 1
                            coordinates.add((row_tile * 16 + row,
                                             column_tile * 16 + column))
            self.assertEqual(writes, [1] * len(writes))
            self.assertEqual(len(coordinates), 32 * 160)

    def test_recorded_k_order_visits_every_inner_element_once(self):
        starts = (0, 23, 4, 3, 8, 7, 12, 11, 16, 15)
        for column_tile in range(10):
            segments = [
                ((23 - iteration if column_tile % 2 else iteration)
                 + (column_tile // 2) * 4) % 24
                for iteration in range(24)
            ]
            self.assertEqual(segments[0], starts[column_tile])
            visits = [segment * 256 + half * 128 + inner
                      for segment in segments
                      for half in range(2) for inner in range(128)]
            self.assertEqual(sorted(visits), list(range(6144)))


if __name__ == "__main__":
    unittest.main()
