import copy
import unittest

from profile_stages import decode_trace, summarize


def records():
    result = []
    for slot in range(3):
        kind, core = (0, 0) if slot == 0 else (1, slot - 1)
        record = [0] * 64
        record[:6] = [1, kind, core, 12, 16, 1]
        record[8:23] = [1000 * slot + tick for tick in range(10, 160, 10)]
        result.append(record)
    return result


class TraceContractTests(unittest.TestCase):
    def test_core_offsets_do_not_change_local_deltas(self):
        first = decode_trace(records(), 12, 16, 1)
        shifted = records()
        for record in shifted:
            record[8:23] = [tick + 7000000 for tick in record[8:23]]
        second = decode_trace(shifted, 12, 16, 1)
        self.assertEqual(summarize([first]), summarize([second]))

    def test_rejects_stale_wrong_owner_and_reordered_timestamps(self):
        base = records()
        for slot, word, value in ((1, 3, 1), (2, 2, 0), (0, 9, 9999)):
            with self.subTest(slot=slot, word=word):
                broken = copy.deepcopy(base)
                broken[slot][word] = value
                with self.assertRaises(ValueError):
                    decode_trace(broken, 12, 16, 1)


if __name__ == "__main__":
    unittest.main()
