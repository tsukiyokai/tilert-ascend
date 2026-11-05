"""Check M12 task ownership, valid-tail addresses, and K traversal."""

import unittest

from kshift_check import outer_order


class GeometryTests(unittest.TestCase):
    def test_each_valid_output_has_one_owner(self):
        owners = {}
        for core in range(10):
            for row in range(12):
                for lane in range(16):
                    address = row * 160 + core * 16 + lane
                    self.assertNotIn(address, owners)
                    owners[address] = core
        self.assertEqual(set(owners), set(range(12 * 160)))
        self.assertTrue(set(owners).isdisjoint(range(12 * 160, 16 * 160)))

    def test_every_operand_coordinate_is_valid(self):
        for shifted in (False, True):
            for core in range(10):
                inner = [segment * 512 + step * 128 + lane
                         for segment in outer_order(core, shifted)
                         for step in range(4) for lane in range(128)]
                self.assertEqual(sorted(inner), list(range(6144)))
                for row in range(12):
                    self.assertLess(row * 6144 + max(inner), 12 * 6144)
                for column in range(core * 16, (core + 1) * 16):
                    self.assertLess(column * 6144 + max(inner), 160 * 6144)

    def test_recorded_witness_tasks_have_distinct_order(self):
        self.assertEqual(outer_order(5, True),
                         [3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4])
        self.assertEqual(outer_order(7, True),
                         [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6])
        self.assertEqual(outer_order(5, False), list(range(12)))

    def test_nd2nz_operand_layout_matches_l0(self):
        # M0/N0 are the inner 16-element axes; each K128 load has eight fractals.
        a = {(k // 16) * 256 + row * 16 + k % 16: (row, k)
             for row in range(12) for k in range(128)}
        b = {(k // 16) * 256 + column * 16 + k % 16: (column, k)
             for column in range(16) for k in range(128)}
        self.assertEqual(len(a), 12 * 128)
        self.assertEqual(set(b), set(range(16 * 128)))
        padding = {(k // 16) * 256 + row * 16 + k % 16
                   for row in range(12, 16) for k in range(128)}
        self.assertTrue(set(a).isdisjoint(padding))
        self.assertEqual(set(a) | padding, set(range(16 * 128)))


if __name__ == "__main__":
    unittest.main()
