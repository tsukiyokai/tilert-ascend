"""Keep lookup benchmarks on the same longest-tail and recency policy."""

from contextlib import redirect_stdout
from functools import partial
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import bench_lookup as bench
from tilert.models.glm_5_2_ascend import lookup_draft
from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter


class LookupBenchmarkChecks(unittest.TestCase):
    def setUp(self):
        tail = list(range(1, 33))
        self.context = tail + [111] + tail[-8:] + [222] + tail

    def test_longer_tail_disambiguates_the_more_recent_short_match(self):
        self.assertEqual(bench.ReverseScan(self.context).propose(1), [111])
        self.assertEqual(LookupDrafter(self.context).propose(1), [111])
        short = bench.ReverseScan(self.context, longest=8)
        self.assertEqual(short.propose(1), [222])

    def test_replays_compare_the_same_tail_bound_at_every_width(self):
        prompt, tokens = self.context[:-1], [32, 111, 3, 4, 5]
        for longest in (8, 32):
            for width in (2, 8, 32):
                with self.subTest(longest=longest, width=width):
                    baseline = bench.replay(
                        partial(bench.ReverseScan, longest=longest),
                        prompt, tokens, width,
                    )
                    candidate = bench.replay(
                        partial(LookupDrafter, longest=longest),
                        prompt, tokens, width,
                    )
                    self.assertEqual(baseline["signature_sha256"],
                                     candidate["signature_sha256"])

    def test_cli_records_and_passes_the_same_bound_to_both_arms(self):
        for longest in (8, 32):
            with TemporaryDirectory() as directory:
                root = Path(directory)
                record, manifest, output = (root / name for name in (
                    "record.json", "manifest.json", "result.json",
                ))
                record.write_text(json.dumps({"requests": [{
                    "warmup": False, "tokens": [32, 111, 3, 4, 5],
                }]}))
                manifest.write_text(json.dumps({
                    "prompt_token_ids": self.context[:-1],
                }))
                argv = ["bench", "--record", str(record), "--manifest",
                        str(manifest), "--candidate", lookup_draft.__file__]
                argv += ["--output", str(output), "--widths", "32",
                         "--warmup", "0", "--samples", "1", "--longest",
                         str(longest)]
                with (
                    patch.object(sys, "argv", argv),
                    redirect_stdout(io.StringIO()),
                ):
                    bench.main()
                report = json.loads(output.read_text())
                self.assertEqual(report["longest_tail"], longest)
                self.assertTrue(report["results"][0]["proposals_identical"])


if __name__ == "__main__":
    unittest.main()
