"""CPU checks for the indexed-attention lifetime and diagnostic contracts."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from attention_indexed_check import decode_front, tensor_from_bytes
from attention_indexed_profile import read_trace, summarize
from indexer_front_check import layout


HERE = Path(__file__).resolve().parent


class IndexedContractTest(unittest.TestCase):
    def test_native_regions_are_disjoint_for_all_supported_widths(self):
        with tempfile.TemporaryDirectory(prefix="indexed-attention-") as directory:
            executable = Path(directory) / "contract"
            for defines in ([], ["-DATTENTION_INDEXED_PROFILE"]):
                subprocess.run([
                    "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    *defines, str(HERE / "attention_indexed_contract_test.cpp"),
                    "-o", str(executable),
                ], check=True, capture_output=True, text=True)
                result = subprocess.run([str(executable)], check=True,
                                        capture_output=True, text=True)
                self.assertIn("768 cases", result.stdout)

    def test_profile_rejects_stale_workers_and_uses_only_local_differences(self):
        records = torch.zeros((6, 16), dtype=torch.int64)
        for slot in range(6):
            kind, index = (0, slot) if slot < 2 else (1, slot - 2)
            records[slot, :5] = torch.tensor([1, kind, index, 12, 2])
            records[slot, 8:14] = torch.tensor([0, 50, 150, 300, 500, 750])
            records[slot, 8:14] += (slot + 1) * 1000000
        scratch = records.view(torch.uint8).flatten()
        offsets = {"trace": 0, "trace_words": 16, "trace_marks": 6,
                   "trace_stamp_offset": 8}
        trace = read_trace(torch, scratch, offsets, 12, 2)
        result = summarize([trace, trace], 2)
        self.assertEqual(result["max_per_worker_median_local_window_us"], 15)
        self.assertEqual(result["max_per_worker_median_phase_us"], {
            "prolog": 1, "indexer_front": 2, "indexer": 3, "sfa": 4,
            "projection_tail": 5,
        })
        records[4, 3] = 1
        with self.assertRaisesRegex(ValueError, "identity"):
            read_trace(torch, scratch, offsets, 12, 2)

    def test_typed_diagnostics_preserve_independent_storage_when_saved(self):
        rows = 12
        regions, size = layout(rows)
        scratch = torch.zeros(size, dtype=torch.uint8)
        shapes = {
            "query_projection": (rows, 32, 128),
            "key_projection": (rows, 160), "key_normalized": (rows, 128),
            "query_rope": (rows, 32, 128), "key_rope": (rows, 128),
            "norm_stats": (rows, 2), "query_hadamard": (rows, 32, 128),
            "key_hadamard": (rows, 128),
        }
        for index, (name, shape) in enumerate(shapes.items()):
            dtype = torch.float32 if name == "norm_stats" or (
                "hadamard" in name) else torch.bfloat16
            view = tensor_from_bytes(torch, scratch, regions[name][0], shape,
                                     dtype)
            view.fill_(index + .5)
        decoded = decode_front(
            torch, scratch, {name: start for name, (start, _) in regions.items()},
            rows, True)
        for index, name in enumerate(shapes):
            self.assertEqual(decoded[name].shape, shapes[name])
            self.assertTrue(torch.all(decoded[name] == index + .5))
        saved = {name: value.clone(memory_format=torch.contiguous_format)
                 for name, value in decoded.items()}
        with tempfile.TemporaryDirectory(prefix="indexed-dump-") as directory:
            path = Path(directory) / "diagnostics.pt"
            torch.save(saved, path)
            restored = torch.load(path, weights_only=True)
        for name, value in saved.items():
            self.assertTrue(torch.equal(value.view(torch.uint8),
                                        restored[name].view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
