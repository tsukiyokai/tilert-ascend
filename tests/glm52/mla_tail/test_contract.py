"""Run with python -m unittest discover -s tests/glm52/mla_tail."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from contract import (
    bf16_round,
    dequantize,
    pack_weights,
    quantize_rows,
    reference,
    resident_blocks,
    tiled_reference,
    workspace_layout,
)


class GeometryTests(unittest.TestCase):
    def test_all_widths_have_complete_disjoint_storage(self):
        for width in (1, 2, 4, 6, 8, 12, 16, 24, 32):
            with self.subTest(width=width):
                capacity = (width + 15) // 16 * 16
                regions = workspace_layout(capacity)
                previous_end = 0
                for region in regions:
                    self.assertGreaterEqual(region.offset, previous_end)
                    self.assertEqual(region.offset % 512, 0)
                    previous_end = region.offset + region.size
                self.assertEqual(regions[0].size, capacity * 4 * 512 * 2)
                self.assertEqual(regions[4].size, capacity * 6144 * 4)

    def test_each_tile_is_owned_once_with_idle_cores(self):
        for blocks in (1, 2, 3, 4, 8, 16, 20, 24):
            for tiles in (16, 24):
                assigned = [tile for core in range(blocks)
                            for tile in range(core, tiles, blocks)]
                self.assertEqual(sorted(assigned), list(range(tiles)))

    def test_mixed_residency_checks_both_core_types(self):
        self.assertEqual(resident_blocks(24, 48), 8)
        self.assertEqual(resident_blocks(24, 12), 6)
        self.assertEqual(resident_blocks(20, 48, 20), 20)
        for requested in (-1, 7, 24):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    resident_blocks(24, 12, requested)

    def test_cpp_layout_matches_cpu_byte_contract(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("a host C++ compiler is required for the layout check")
        here = Path(__file__).resolve().parent
        source = r'''
#include <iostream>
#include "mla_tail.h"
int main() {
    for (int rows : {16, 32}) {
        auto l = mla_tail::MakeWorkspaceLayout(rows);
        std::cout << l.heads << ' ' << l.values << ' ' << l.quantized << ' '
                  << l.token_scales << ' ' << l.accumulators << ' '
                  << l.system << ' ' << l.bytes << '\n';
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="mla-tail-contract-") as tmp:
            path = Path(tmp)
            (path / "layout.cpp").write_text(source)
            subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-I", str(here), str(path / "layout.cpp"),
                 "-o", str(path / "layout")], check=True,
            )
            lines = subprocess.check_output(
                [str(path / "layout")], text=True,
            ).splitlines()
        expected = []
        for rows in (16, 32):
            regions = workspace_layout(rows)
            values = [region.offset for region in regions]
            values.append(regions[-1].offset + regions[-1].size)
            expected.append(values)
        self.assertEqual([list(map(int, line.split())) for line in lines],
                         expected)


class ArithmeticTests(unittest.TestCase):
    def test_bf16_boundary_changes_a_quantization_decision(self):
        values = np.array([[127.0, 0.5018, -0.5018]], dtype=np.float32)
        rounded = bf16_round(values)
        codes, _ = quantize_rows(rounded)
        unrounded, _ = quantize_rows(values)
        self.assertEqual(codes.tolist(), [[127, 0, 0]])
        self.assertEqual(unrounded.tolist(), [[127, 1, -1]])

    def test_a8_nearest_even_and_zero_rows(self):
        values = np.array([
            [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5],
            [0.0] * 7,
        ], dtype=np.float32)
        codes, scale = quantize_rows(values)
        self.assertEqual(codes.tolist(), [
            [127, 0, 2, 2, 0, -2, -2], [0] * 7,
        ])
        np.testing.assert_array_equal(scale, [1.0, 0.0])

    def test_integer_and_float_bias_have_distinct_units(self):
        accumulator = np.array([[10, -10]], dtype=np.int32)
        scale = np.array([0.5, 0.5], dtype=np.float32)
        tokens = np.array([0.25], dtype=np.float32)
        integer = dequantize(accumulator, scale, tokens,
                             np.array([2, 2], dtype=np.int32))
        floating = dequantize(accumulator, scale, tokens,
                              np.array([2, 2], dtype=np.float32))
        np.testing.assert_array_equal(integer, [[1.5, -1.0]])
        np.testing.assert_array_equal(floating, [[3.25, 0.75]])

    def test_tile_pack_preserves_every_weight_coordinate(self):
        uv = np.arange(4 * 512 * 256, dtype=np.int32).reshape(4, 512, 256)
        wo = np.arange(1024 * 6144, dtype=np.int32).reshape(1024, 6144)
        value_tiles, output_tiles = pack_weights(uv, wo)
        restored_uv = value_tiles.reshape(4, 4, 512, 64)
        restored_uv = restored_uv.transpose(0, 2, 1, 3).reshape(4, 512, 256)
        restored_wo = output_tiles.transpose(1, 0, 2).reshape(1024, 6144)
        np.testing.assert_array_equal(restored_uv, uv)
        np.testing.assert_array_equal(restored_wo, wo)

    def test_full_tp16_chain_matches_with_padding_and_idle_cores(self):
        # Exact powers of two avoid CPU BLAS summation-order differences;
        # the test exercises all output columns and the real TP16 geometry.
        rng = np.random.default_rng(829)
        latent = rng.integers(-2, 3, (2, 4, 512)).astype(np.float32) / 8
        uv = rng.integers(-1, 2, (4, 512, 256)).astype(np.float32) / 16
        wo = rng.integers(-8, 8, (1024, 6144), dtype=np.int8)
        scale = rng.integers(1, 5, 6144).astype(np.float32) / 32
        value_tiles, output_tiles = pack_weights(uv, wo)
        expected = reference(latent, uv, wo, scale)
        actual = tiled_reference(latent, value_tiles, output_tiles, scale,
                                  capacity=16, blocks=20)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
