"""Check the native address contract against its diagnostic decoder on CPU."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from indexer_front_check import layout


HERE = Path(__file__).resolve().parent


class FrontContractTest(unittest.TestCase):
    def test_all_widths_match_native_layout_and_address_coverage(self):
        with tempfile.TemporaryDirectory(prefix="indexer-front-") as d:
            executable = Path(d) / "contract"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                str(HERE / "indexer_front_contract_test.cpp"),
                "-o", str(executable),
            ], check=True, capture_output=True, text=True)
            result = subprocess.run([str(executable)], check=True,
                                    capture_output=True, text=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 32)
        for line in lines:
            rows, *actual = map(int, line.split())
            regions, total = layout(rows)
            self.assertEqual(actual,
                             [offset for offset, _ in regions.values()]
                             + [total])


if __name__ == "__main__":
    unittest.main()
