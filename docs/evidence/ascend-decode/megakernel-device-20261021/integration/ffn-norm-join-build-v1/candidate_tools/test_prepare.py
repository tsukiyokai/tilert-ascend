"""Check the frozen caller publication and isolate the removed norm join."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from prepare import (
    CALLER, NORM, PARENT_LIBRARY, digest, prepare, rename, transform,
)


PARENT = Path(os.environ.get(
    "FFN_NORM_JOIN_PARENT",
    "/tmp/tilert-megakernel-validation-20261021/"
    "ffn-original-shared-gu-static-build-v1"))


class NormJoinContract(unittest.TestCase):
    def test_only_the_final_norm_join_changes(self):
        original = (PARENT / NORM).read_text()
        changed = transform(original)
        restored = changed.replace(
            "}\n\n}\n#endif\n", "    SyncAll<false>();\n}\n\n}\n#endif\n")
        self.assertEqual(restored, original)
        self.assertEqual(original.count("pipe.Reset()"),
                         changed.count("pipe.Reset()"))
        with self.assertRaises(ValueError):
            transform(changed)

    def test_the_caller_retains_the_complete_publication(self):
        caller = (PARENT / CALLER).read_text()
        self.assertEqual(caller.count("device::EndPhase(pipe)"), 3)
        native = (PARENT / "bridge/source/ffn_bridge/native.h").read_text()
        self.assertIn("    PipeBarrier<PIPE_ALL>();\n"
                      "    DataSyncBarrier<MemDsbT::DDR>();\n"
                      "    SyncAll<false>();\n    pipe.Reset();", native)

    def test_all_other_sources_and_the_public_contract_are_unchanged(self):
        manifest = json.loads((PARENT / "manifest.json").read_text())
        self.assertEqual(
            manifest["artifacts"]["ffn_original_shared_gu_static.so"],
            PARENT_LIBRARY)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(PARENT, output)
            for name in manifest["sources"]:
                if (name in (NORM, "parent-manifest.json")
                        or name.startswith("candidate_tools/")):
                    continue
                original = (PARENT / name).read_bytes()
                if Path(name).suffix in (".h", ".cpp"):
                    original = rename(original.decode()).encode()
                self.assertEqual((output / name).read_bytes(), original, name)
            for name, expected in report["sources"].items():
                self.assertEqual(digest(output / name), expected, name)
            self.assertEqual(report["public_scratch_count"], 33)
            self.assertEqual(report["body_barriers"], 5)
            self.assertEqual(report["norm_internal_joins"], 0)


if __name__ == "__main__":
    unittest.main()
