import copy
import unittest

from profile_routed import decode_trace, summarize


def records():
    result = []
    for slot in range(3):
        kind, index = (0, 0) if slot == 0 else (1, slot - 1)
        record = [0] * 64
        record[:5] = [1, kind, index, 12, 1]
        record[8:29] = [slot * 1000 + mark for mark in range(10, 220, 10)]
        result.append(record)
    return result


class ProfileContractTests(unittest.TestCase):
    def test_independent_core_clock_offsets_preserve_local_deltas(self):
        first = decode_trace(records(), 12, 1)
        shifted = records()
        for index, record in enumerate(shifted):
            record[8:29] = [tick + (index + 1) * 7000000 for tick in record[8:29]]
        second = decode_trace(shifted, 12, 1)
        self.assertEqual(summarize([first]), summarize([second]))
        self.assertTrue(first[2]["stages"][1]["performs_stage_work"])
        self.assertFalse(first[2]["stages"][3]["performs_stage_work"])

    def test_rejects_stale_owner_and_backwards_completion(self):
        for slot, word, value in ((1, 3, 1), (2, 2, 0), (0, 9, 9999)):
            with self.subTest(slot=slot, word=word):
                broken = copy.deepcopy(records())
                broken[slot][word] = value
                with self.assertRaises(ValueError):
                    decode_trace(broken, 12, 1)

    def test_row_schedule_includes_both_vector_cores(self):
        expert = decode_trace(records(), 12, 1)
        balanced = decode_trace(records(), 12, 1, row_balanced=True)
        self.assertFalse(expert[2]["stages"][3]["performs_stage_work"])
        self.assertTrue(balanced[2]["stages"][3]["performs_stage_work"])
        self.assertEqual(expert[2]["raw_ticks"], balanced[2]["raw_ticks"])


if __name__ == "__main__":
    unittest.main()
