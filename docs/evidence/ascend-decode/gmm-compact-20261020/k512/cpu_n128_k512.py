"""CPU resource and exact-integer contract for one GMM1 K retile."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parent
K = 6144
N = 256
FIELDS = {
    "N128_K256": {"baseM": 128, "baseN": 128, "baseK": 256,
                  "stepKa": 4, "stepKb": 4, "depthA1": 8, "depthB1": 8},
    "N128_K512": {"baseM": 128, "baseN": 128, "baseK": 512,
                  "stepKa": 2, "stepKb": 2, "depthA1": 4, "depthB1": 4},
}


def resources(fields):
    m, n, k = (fields[name] for name in ("baseM", "baseN", "baseK"))
    return {
        "L0A_double": m * k,
        "L0B_double": n * k,
        "L0C_single": m * n * 4,
        "L1A": m * k * fields["depthA1"] // 2,
        "L1B": n * k * fields["depthB1"] // 2,
        "A_K_span": k * fields["stepKa"],
        "B_K_span": k * fields["stepKb"],
        "K_iterations": K // k,
        "A_outer": K // (k * fields["stepKa"]),
        "B_outer": K // (k * fields["stepKb"]),
    }


def main():
    api = subprocess.run([str(ROOT / "tiling_nsplit"), "128", "512", "128"],
                         capture_output=True, text=True, check=True)
    values = {line.split("=", 1)[0]: int(line.split("=", 1)[1])
              for line in api.stdout.splitlines() if "=" in line}
    assert values["rc"] == 0 and values["baseK"] == 512
    report = {"scope": "CPU source/resource contract; no NPU execution",
              "actual_API_request": {"baseM": 128, "baseN": 128, "baseK": 512},
              "actual_API_fields": values, "actual_API_stdout": api.stdout,
              "forced_comparison_fields": FIELDS, "resources": {}}
    for arm, fields in FIELDS.items():
        res = resources(fields)
        assert max(res["L0A_double"], res["L0B_double"]) <= 65536
        assert res["L0C_single"] <= 131072
        assert res["L1A"] + res["L1B"] + 2048 <= 458752 < 524288
        assert fields["depthA1"] == 2 * fields["stepKa"]
        assert fields["depthB1"] == 2 * fields["stepKb"]
        assert res["A_K_span"] == res["B_K_span"] == 1024
        blocks = [range(start, start + fields["baseK"])
                  for start in range(0, K, fields["baseK"])]
        assert [offset for block in blocks for offset in block] == list(range(K))
        for tail_n in (0, 128):
            assert tail_n % 64 == 0
            intervals = []
            for nz_column in (tail_n, tail_n + 64):
                for block in blocks:
                    start = nz_column * K + block.start * 64
                    stop = nz_column * K + block.stop * 64
                    assert 0 <= start < stop <= N * K
                    intervals.append((start, stop))
            assert intervals[0][0] == tail_n * K
            assert intervals[-1][1] == (tail_n + 128) * K
            assert all(a[1] == b[0] for a, b in
                       zip(intervals, intervals[1:]))
        report["resources"][arm] = res
    assert report["resources"]["N128_K256"]["L1A"] == report["resources"]["N128_K512"]["L1A"]
    assert report["resources"]["N128_K256"]["L1B"] == report["resources"]["N128_K512"]["L1B"]

    generator = np.random.default_rng(20261020)
    checked = []
    for mode in ("random0", "random1", "all_64", "all_minus56", "all_49"):
        if mode.startswith("random"):
            a = generator.integers(-8, 8, size=(K, 128), dtype=np.int16)
            b = generator.integers(-8, 8, size=(K, 128), dtype=np.int16)
            products = a.astype(np.int64) * b
        else:
            value = {"all_64": 64, "all_minus56": -56, "all_49": 49}[mode]
            products = np.full((K, 128), value, dtype=np.int64)
        accumulated = {}
        for block_k in (256, 512):
            partial = products.reshape(K // block_k, block_k, 128).sum(axis=1)
            accumulated[block_k] = partial.sum(axis=0)
            assert np.abs(partial).max() <= 64 * K < 2**31
        assert np.array_equal(accumulated[256], accumulated[512])
        checked.append({"case": mode, "outputs": 128,
                        "max_abs_int32": int(np.abs(accumulated[256]).max()),
                        "exact": True})
    report.update(integer_checks=checked, all_possible_abs_bound=64 * K,
                  accumulation_contract="Signed INT4 products accumulate in INT32 with no overflow; one unchanged GetTensorC/Fixpipe per full quantGroupSize6144 and output tile. Device check remains required.",
                  address_contract="Retain full OrgShapeN256, NZ tailN*K offsets and both MSD rows; only internal K partitions differ.",
                  status="pass")
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT / "cpu-n128-k512-contract.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
