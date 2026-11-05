"""Validate the two-clock trace contract used by the sparse profile gate."""

import copy
import unittest

import decode


class SparseTraceTest(unittest.TestCase):
    def setUp(self):
        self.records = []
        for slot in range(6):
            kind = int(slot >= 2)
            index = slot - 2 if kind else slot
            start = 1000 + 100 * slot
            finish = start + 50 * (slot + 1)
            self.records.append(
                [1, kind, index, 12, 2, 0, 0, 0]
                + [start] * 4 + [finish] * 9 + [0] * 11)

    def test_only_body_setup_is_reported(self):
        samples = decode.read_sparse_trace(self.records, 12, 2)
        report = decode.summarize_sparse([samples], 2)
        self.assertEqual(report["sampled_marks"], [3, 4])
        self.assertEqual(report["clock_reads_per_worker"], 2)
        self.assertEqual(report["max_per_worker_median_body_setup_us"], 6)
        self.assertEqual(report["critical_worker"]["kind"], "aiv")
        self.assertEqual(report["critical_worker"]["index"], 3)
        self.assertNotIn("max_per_worker_median_phase_us", report)

    def test_rejects_monotonic_but_nonrepeated_fields(self):
        records = copy.deepcopy(self.records)
        records[0][8] -= 1
        with self.assertRaisesRegex(ValueError, "timestamp fields"):
            decode.read_sparse_trace(records, 12, 2)

    def test_rejects_unwritten_padding(self):
        for word in (5, 7, 21, 31):
            with self.subTest(word=word):
                records = copy.deepcopy(self.records)
                records[0][word] = 1
                with self.assertRaisesRegex(ValueError, "trace padding"):
                    decode.read_sparse_trace(records, 12, 2)

    def test_rejects_wrong_identity_or_clock_order(self):
        for word, value in ((2, 9), (12, 999)):
            with self.subTest(word=word):
                records = copy.deepcopy(self.records)
                records[0][word] = value
                with self.assertRaises(ValueError):
                    decode.read_sparse_trace(records, 12, 2)


if __name__ == "__main__":
    unittest.main()
