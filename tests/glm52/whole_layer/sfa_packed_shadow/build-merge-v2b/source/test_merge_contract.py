"""Check packed merge stores and the unchanged CUBE operand/reduction order."""

from array import array
import unittest

from test_contract import ROW_BYTES, row_bytes


WIDTH = 576
PLANE = 512 * WIDTH


def align16(value):
    return (value + 15) // 16 * 16


def words(physical):
    result = array("H")
    result.frombytes(row_bytes(physical))
    return result


def layouts(processed, plane):
    """Form equivalent caches with independent zero tails in the two AIV spans."""
    planar = array("H", [0xDEAD]) * (4 * PLANE)
    packed = array("H", [0xDEAD]) * (4 * PLANE)
    middle = min(((processed + 1) // 2 + 1) // 2 * 2, processed)
    for begin, end, missing in ((0, middle, 3), (middle, processed, 7)):
        valid_end = max(begin, end - missing)
        for row in range(begin, end):
            value = words(plane * 512 + row) if row < valid_end else array("H", [0]) * WIDTH
            base = plane * PLANE
            planar[base + row * 512:base + (row + 1) * 512] = value[:512]
            rope = base + 512 * 512 + row * 64
            planar[rope:rope + 64] = value[512:]
            packed[base + row * WIDTH:base + (row + 1) * WIDTH] = value
    return planar, packed


def nd_to_nz(destination, source, source_start, destination_start, rows, columns,
             source_stride, row_align):
    """A3 Nd2Nz: C0=16 BF16 elements, C0Stride=row_align, NStride=1."""
    for column in range(0, columns, 16):
        for row in range(rows):
            src = source_start + row * source_stride + column
            dst = destination_start + column * row_align + row * 16
            destination[dst:dst + 16] = source[src:src + 16]


def qk_operand(source, packed, plane, row_begin, count, half):
    aligned = align16(count)
    output = array("H", [0xA55A]) * (aligned * 288)
    base = plane * PLANE
    latent = base + row_begin * (WIDTH if packed else 512) + half * 256
    rope = (base + row_begin * WIDTH + 512 + half * 32 if packed else
            base + 512 * 512 + row_begin * 64 + half * 32)
    latent_destination = 0 if half == 0 else 32 * aligned
    rope_destination = 256 * aligned if half == 0 else 0
    nd_to_nz(output, source, latent, latent_destination, count, 256,
             WIDTH if packed else 512, aligned)
    nd_to_nz(output, source, rope, rope_destination, count, 32,
             WIDTH if packed else 64, aligned)
    return output


def pv_operand(source, packed, plane, row_begin, count, column_begin):
    aligned = align16(count)
    output = array("H", [0xA55A]) * (aligned * 128)
    source_start = plane * PLANE + row_begin * (WIDTH if packed else 512) + column_begin
    nd_to_nz(output, source, source_start, 0, count, 128,
             WIDTH if packed else 512, aligned)
    return output


class PackedMergeContract(unittest.TestCase):
    def test_all_flush_counts_and_two_aiv_halves_stay_within_plane(self):
        for plane in range(4):
            for ping in range(2):
                for begin in (0, 224, 256, 480):
                    for count in range(33):
                        source = bytearray([0x55] * (2 * 32 * ROW_BYTES))
                        records = b"".join(row_bytes(row + 13) for row in range(count))
                        source[ping * 32 * ROW_BYTES:ping * 32 * ROW_BYTES + len(records)] = records
                        destination = bytearray([0xA5] * (4 * PLANE * 2))
                        start = (plane * PLANE + begin * WIDTH) * 2
                        end = start + count * ROW_BYTES
                        destination[start:end] = source[ping * 32 * ROW_BYTES:
                                                        ping * 32 * ROW_BYTES + count * ROW_BYTES]
                        self.assertEqual(destination[start:end], records)
                        self.assertEqual(destination[:start], bytes([0xA5]) * start)
                        self.assertEqual(destination[end:], bytes([0xA5]) * (len(destination) - end))
                        self.assertLessEqual(end, (plane + 1) * PLANE * 2)

    def test_qk_and_pv_l1_bytes_match_for_every_plane_and_tail_residue(self):
        for plane in range(4):
            for processed in (1, 2, 15, 16, 17, 31, 32, 127, 128, 129,
                              255, 256, 257, 511, 512):
                planar, packed = layouts(processed, plane)
                for row in range(0, processed, 128):
                    count = min(128, processed - row)
                    for half in range(2):
                        self.assertEqual(qk_operand(planar, False, plane, row, count, half),
                                         qk_operand(packed, True, plane, row, count, half),
                                         ("QK", plane, processed, row, half))
                    for column in range(0, 512, 128):
                        self.assertEqual(pv_operand(planar, False, plane, row, count, column),
                                         pv_operand(packed, True, plane, row, count, column),
                                         ("PV", plane, processed, row, column))

    def test_qk_halves_preserve_six_96_column_reduction_steps(self):
        packed = array("H", range(WIDTH))
        first = qk_operand(packed, True, 0, 0, 1, 0)
        second = qk_operand(packed, True, 0, 0, 1, 1)

        def original_columns(value):
            return [value[(column // 16) * 16 * 16 + column % 16]
                    for column in range(288)]

        first, second = original_columns(first), original_columns(second)
        self.assertEqual(first, list(range(256)) + list(range(512, 544)))
        self.assertEqual(second, list(range(544, 576)) + list(range(256, 512)))
        steps = [half[start:start + 96] for half in (first, second)
                 for start in range(0, 288, 96)]
        self.assertEqual(steps[2], list(range(192, 256)) + list(range(512, 544)))
        self.assertEqual(steps[3], list(range(544, 576)) + list(range(256, 320)))
        self.assertEqual(len(steps), 6)


if __name__ == "__main__":
    unittest.main()
