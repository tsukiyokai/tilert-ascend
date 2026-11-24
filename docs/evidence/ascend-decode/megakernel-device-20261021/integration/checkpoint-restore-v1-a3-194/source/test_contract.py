"""Exercise incorrect content and model-view owner counterexamples on CPU."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from verify import create_view, inspect_file


class CheckpointContract(unittest.TestCase):
    def test_same_size_corruption_requires_the_requested_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            path.write_bytes(b"abcd")
            expected = {"bytes": 4, "sha256": hashlib.sha256(b"abcd").hexdigest()}
            self.assertTrue(inspect_file(path, expected, True)["pass"])
            path.write_bytes(b"abce")
            self.assertTrue(inspect_file(path, expected, False)["pass"])
            self.assertFalse(inspect_file(path, expected, True)["pass"])

    def test_size_failure_is_not_hidden_by_a_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            path.write_bytes(b"abc")
            expected = {"bytes": 4, "sha256": hashlib.sha256(b"abc").hexdigest()}
            self.assertFalse(inspect_file(path, expected, True)["pass"])

    def test_view_preserves_two_owners_and_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "nas-data", root / "owned-data"
            first.write_bytes(b"nas")
            second.write_bytes(b"owned")
            view = root / "view"
            create_view(view, {"config.json": first, "optional/quarot": second})
            self.assertEqual((view / "config.json").read_bytes(), b"nas")
            self.assertEqual((view / "optional/quarot").read_bytes(), b"owned")
            self.assertEqual(first.read_bytes(), b"nas")
            with self.assertRaises(FileExistsError):
                create_view(view, {"config.json": second})


if __name__ == "__main__":
    unittest.main(verbosity=2)
