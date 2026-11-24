"""Check raw-byte layout and pair-order obligations, without device execution."""

import random
import struct
import unittest


LATENT_BYTES = 1024
ROPE_BYTES = 128
ROW_BYTES = LATENT_BYTES + ROPE_BYTES
INT32_MAX = (1 << 31) - 1


def row_bytes(physical):
    # Include signed zero and nonfinite BF16 bit patterns: packing is a copy.
    words = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1]
    words += [(physical * 37 + column * 13) & 0xFFFF
              for column in range(576 - len(words))]
    return struct.pack("<576H", *words)


def original_order(logical, physical, limit):
    stride = (abs(physical[0] - physical[1]) - 1) * LATENT_BYTES
    fallback = (stride < 0 or stride >= INT32_MAX
                or any(row + 1 >= limit for row in logical))
    valid = [row for row in physical if row >= 0]
    return valid if fallback else sorted(valid)


def packed_reads(logical, physical, limit):
    original_stride = (abs(physical[0] - physical[1]) - 1) * LATENT_BYTES
    if (original_stride < 0 or original_stride >= INT32_MAX
            or any(row + 1 >= limit for row in logical)):
        return [row_bytes(row) for row in physical if row >= 0]
    count = sum(row >= 0 for row in physical)
    if not count:
        return []
    packed_stride = original_stride // 512 * 576
    if count == 2 and packed_stride >= INT32_MAX:
        return [row_bytes(row) for row in sorted(physical)]
    first = min(row for row in physical if row >= 0)
    addresses = [first * ROW_BYTES + index * (ROW_BYTES + packed_stride)
                 for index in range(count)]
    return [row_bytes(address // ROW_BYTES) for address in addresses]


def deinterleave(packed, count, destination, ping, plane, row_offset):
    source_base = ping * 32 * ROW_BYTES
    plane_base = plane * 512 * ROW_BYTES
    for column_offset, length, source_gap, target in (
            (0, LATENT_BYTES, 4, plane_base + row_offset * LATENT_BYTES),
            (LATENT_BYTES, ROPE_BYTES, 32,
             plane_base + 512 * LATENT_BYTES + row_offset * ROPE_BYTES)):
        for row in range(count):
            source = source_base + column_offset + row * (length + source_gap * 32)
            start = target + row * length
            destination[start:start + length] = packed[source:source + length]


class PackedShadowContract(unittest.TestCase):
    def test_fast_pair_order_and_causal_tail_differ(self):
        self.assertEqual(original_order((3, 7), (131, 7), 20), [7, 131])
        self.assertEqual(original_order((3, 7), (131, 7), 8), [131, 7])
        self.assertEqual(original_order((3, 3), (131, 131), 20), [131, 131])
        self.assertEqual(original_order((3, 30), (131, -1), 20), [131])

    def test_all_small_pair_states_preserve_bits(self):
        table = [5, 2, 7, 1, 4, 0, 3, 6]
        for limit in (1, 127, 128, 129, 511, 777, 1024):
            candidates = (-1, 0, 1, 126, 127, 128, limit - 1, limit, 900)
            for first in candidates:
                for second in candidates:
                    logical = (first, second)
                    physical = tuple(table[row // 128] * 128 + row % 128
                                     if 0 <= row < limit else -1
                                     for row in logical)
                    expected = [row_bytes(row) for row in
                                original_order(logical, physical, limit)]
                    self.assertEqual(packed_reads(logical, physical, limit),
                                     expected, (logical, physical, limit))

    def test_packed_stride_overflow_keeps_fast_path_order(self):
        logical, physical = (1, 2), (2_000_000, 0)
        self.assertLess((physical[0] - 1) * LATENT_BYTES, INT32_MAX)
        self.assertGreaterEqual((physical[0] - 1) * ROW_BYTES, INT32_MAX)
        self.assertEqual(packed_reads(logical, physical, 10),
                         [row_bytes(0), row_bytes(2_000_000)])

    def test_every_flush_count_plane_and_ping_has_exact_planar_bytes(self):
        for count in range(33):
            for ping in range(2):
                packed = bytearray([0xA5] * (2 * 32 * ROW_BYTES))
                rows = [row_bytes(200 - row) for row in range(count)]
                start = ping * 32 * ROW_BYTES
                packed[start:start + count * ROW_BYTES] = b"".join(rows)
                for plane in range(4):
                    for row_offset in (0, 224, 256, 480):
                        actual = bytearray([0x55] * (4 * 512 * ROW_BYTES))
                        expected = bytearray(actual)
                        deinterleave(packed, count, actual, ping, plane,
                                     row_offset)
                        base = plane * 512 * ROW_BYTES
                        latent = base + row_offset * LATENT_BYTES
                        rope = base + 512 * LATENT_BYTES + row_offset * ROPE_BYTES
                        expected[latent:latent + count * LATENT_BYTES] = (
                            b"".join(row[:LATENT_BYTES] for row in rows))
                        expected[rope:rope + count * ROPE_BYTES] = (
                            b"".join(row[LATENT_BYTES:] for row in rows))
                        self.assertEqual(actual, expected,
                                         (count, ping, plane, row_offset))

    def test_permuted_pages_causal_compaction_and_sentinel(self):
        generator = random.Random(20261021)
        table = [5, 2, 7, 1, 4, 0, 3, 6]
        for causal_limit in (2, 129, 766, 777):
            selected = [generator.randrange(1024) for _ in range(512)]
            selected[198] = -1
            for first, end in ((0, 256), (256, 512)):
                actual, expected = [], []
                for index in range(first, end, 2):
                    logical = tuple(selected[index:index + 2])
                    if logical[0] < 0:
                        break
                    physical = tuple(table[row // 128] * 128 + row % 128
                                     if 0 <= row < causal_limit else -1
                                     for row in logical)
                    actual.extend(packed_reads(logical, physical, causal_limit))
                    expected.extend(row_bytes(row) for row in
                                    original_order(logical, physical, causal_limit))
                self.assertEqual(actual, expected)
                self.assertLessEqual(len(actual), 256)

    def test_ub_budget_and_half_boundaries(self):
        buffers = (73728, 16384, 32768, 4096, 32768, 8192, 6 * 2048, 2048)
        self.assertEqual(sum(buffers), 178 * 1024)
        self.assertLess(sum(buffers), 188 * 1024)
        self.assertEqual(2 * 32 * ROW_BYTES, buffers[0])


if __name__ == "__main__":
    unittest.main()
