"""Exercise the transfer endpoint, including interrupted and foreign partials."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import prefix_transfer


class TransferContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "destination"
        self.data = bytes(range(256)) * 257
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.run_endpoint("init")

    def run_endpoint(self, mode, data=None, offset=0, check=True):
        result = subprocess.run(
            [sys.executable, "-c", prefix_transfer.REMOTE, mode,
             str(self.root), "main-prefix/rank-0.pt", str(len(self.data)),
             self.digest, str(offset)], input=data, capture_output=True)
        if check:
            result.check_returncode()
        return result

    def test_resume_validates_full_payload_before_rename(self):
        split = 1001
        failed = self.run_endpoint("write", self.data[:split], check=False)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((self.root / "main-prefix/rank-0.pt").exists())
        state = json.loads(self.run_endpoint("state").stdout)
        self.assertEqual(state, {"complete": False, "offset": split})
        result = json.loads(self.run_endpoint(
            "write", self.data[split:], offset=split).stdout)
        self.assertEqual(result["sha256"], self.digest)
        self.assertEqual(self.run_endpoint("read", offset=split).stdout,
                         self.data[split:])

    def test_wrong_hash_preserves_partial_and_rejects_final(self):
        wrong = self.data[:-1] + bytes([self.data[-1] ^ 1])
        failed = self.run_endpoint("write", wrong, check=False)
        self.assertNotEqual(failed.returncode, 0)
        path = self.root / "main-prefix/rank-0.pt"
        self.assertFalse(path.exists())
        self.assertEqual(path.with_suffix(".pt.partial").read_bytes(), wrong)

    def test_foreign_partial_is_not_resumed(self):
        partial = self.root / "main-prefix/rank-0.pt.partial"
        partial.parent.mkdir()
        partial.write_bytes(self.data[:100])
        self.assertNotEqual(self.run_endpoint("state", check=False).returncode, 0)
        self.assertNotEqual(self.run_endpoint(
            "write", self.data[100:], offset=100, check=False).returncode, 0)
        self.assertEqual(partial.read_bytes(), self.data[:100])

    def test_verified_final_is_immutable_and_rechecked(self):
        self.run_endpoint("write", self.data)
        state = json.loads(self.run_endpoint("state").stdout)
        self.assertEqual(state["sha256"], self.digest)
        self.assertNotEqual(self.run_endpoint(
            "write", self.data, check=False).returncode, 0)
        path = self.root / "main-prefix/rank-0.pt"
        path.write_bytes(bytes(len(self.data)))
        self.assertNotEqual(self.run_endpoint("state", check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
