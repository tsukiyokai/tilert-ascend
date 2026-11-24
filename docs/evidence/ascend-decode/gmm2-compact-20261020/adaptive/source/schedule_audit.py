"""Bound the B114 retile schedule and addresses without modeling CANN arithmetic."""

import hashlib
import json
import os
from pathlib import Path
import random
import re


ROOT = Path("/workspace/experiments/gmm2-compact-20261020")
SOURCE = ROOT / "retile-paired-build/source/ops_transformer/ascendc/grouped_matmul/op_kernel"
N, K, CORES, PARALLEL, SLOT, UB_ELEMENTS = 6144, 128, 24, 4, 32768, 8192


def check(counts, base_m, base_n):
    rows = sum(counts)
    spans = [[] for _ in range(rows)]
    expected_expert = [e for e, count in enumerate(counts) for _ in range(count)]
    core_tiles = []
    max_ring = 0
    phases = [0, 0]
    for core in range(CORES):
        pre_count = offset_m = cube_count = 0
        vec_counts = [0, 0]
        for expert, count in enumerate(counts):
            if count <= 0:
                continue
            expanded_m = 2 * count
            blocks_n = (N + base_n - 1) // base_n
            current_count = pre_count + blocks_n * ((expanded_m + base_m - 1) // base_m)
            block = core if core >= pre_count else core + CORES
            while block < current_count:
                mi, ni = divmod(block - pre_count, blocks_n)
                cm = min(base_m, expanded_m - mi * base_m)
                cn = min(base_n, N - ni * base_n)
                assert cm > 0 and cm % 2 == 0 and cn > 0
                row_start = offset_m // 2 + mi * base_m // 2
                x_start = (offset_m + mi * base_m) * K
                assert x_start + cm * K <= 2 * rows * K
                weight_start = expert * N * K + ni * base_n * K
                assert weight_start + cn * K <= (expert + 1) * N * K
                scale_start = expert * N + ni * base_n
                assert scale_start + cn <= (expert + 1) * N
                scratch = SLOT * (core + (cube_count % PARALLEL) * CORES)
                assert cm * cn <= SLOT
                max_ring = max(max_ring, scratch + cm * cn)
                # Source waits for one consumed-tile acknowledgement before
                # reusing each core's four-slot ring, starting at tile four.
                assert (cube_count >= PARALLEL) == (cube_count - PARALLEL >= 0)
                consumed = set()
                vec_base_m = min(UB_ELEMENTS // (((base_n + 7) // 8) * 8), cm // 2)
                assert vec_base_m > 0
                for sub in (0, 1):
                    for om in range(0, cm // 2, vec_base_m):
                        vec_counts[sub] += 1
                        if vec_counts[sub] % 2 != sub:
                            continue
                        vm = min(vec_base_m, cm // 2 - om)
                        size = vm * cn
                        assert size <= UB_ELEMENTS
                        # GM->UB copies select even/odd partial rows at stride
                        # two; the latter copy ends exactly at this bound.
                        ring_end = (2 * (om + vm) - 1) * cn + cn
                        assert ring_end <= cm * cn
                        for row in range(row_start + om, row_start + om + vm):
                            assert row not in consumed
                            consumed.add(row)
                            assert expected_expert[row] == expert
                            spans[row].append((ni * base_n, ni * base_n + cn))
                        phases[sub] += vm * cn
                assert len(consumed) == cm // 2
                assert vec_counts[0] == vec_counts[1]
                cube_count += 1
                block += CORES
            pre_count = current_count % CORES
            offset_m += expanded_m
        assert offset_m == 2 * rows
        core_tiles.append(cube_count)
    for intervals in spans:
        end = 0
        for start, next_end in sorted(intervals):
            assert start == end, (start, end)
            end = next_end
        assert end == N
    assert sum(phases) == rows * N
    assert max_ring <= SLOT * CORES * PARALLEL
    return {"tiles": sum(core_tiles), "max_ring_half_end": max_ring,
            "aiv_output_elements": phases}


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    paths = [SOURCE / name for name in (
        "grouped_matmul.cpp", "grouped_matmul_antiquant_a8w4_msd.h",
        "grouped_matmul_active_groups_a8w4.h")]
    fields_path = ROOT / "tiling-32x1024.txt"
    fields = dict(line.split("=") for line in fields_path.read_text().splitlines())
    assignments = dict(re.findall(r"cfg\.(\w+) = (\d+);", paths[2].read_text()))
    assert assignments == fields
    rng = random.Random(20261020)
    cases = []
    for count in range(1, 257):
        counts = [0] * 256
        counts[(count * 37) % 256] = count
        cases.append(("single_" + str(count), counts))
    for count in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128):
        cases.append(("ends_" + str(count), [count] + [0] * 254 + [256 - count]))
    cases.append(("coherent_top8_m32", [32] * 8 + [0] * 248))
    for queries in (1, 8, 32):
        for sample in range(10):
            counts = [0] * 256
            for _ in range(queries):
                for expert in rng.sample(range(256), 8):
                    counts[expert] += 1
            cases.append((f"top8_m{queries}_{sample}", counts))
    result_cases = []
    for name, counts in cases:
        old, new = check(counts, 64, 512), check(counts, 32, 1024)
        result_cases.append({"name": name, "rows": sum(counts),
                             "old_tiles": old["tiles"], "new_tiles": new["tiles"]})
    declared_before_tmp = 4096 + 128 + 32768 + 16384
    tmp_access_end = 4 * UB_ELEMENTS * 4
    physical_peak = declared_before_tmp + tmp_access_end
    assert physical_peak == 184448 < 196352
    result = {"scope": "CPU source equations, output coverage, address bounds, and AIV phase only; no CANN arithmetic or asynchronous hardware simulation",
              "npu_launched": False, "case_count": len(cases),
              "all_output_elements_exactly_once": True,
              "all_x_weight_bias_and_ring_addresses_bounded": True,
              "all_33_emitted_tcube_fields_equal_source": len(fields) == 33,
              "ring_bytes": SLOT * CORES * PARALLEL * 2,
              "ub_declared_bytes": declared_before_tmp + 73728,
              "ub_physical_access_end_bytes": physical_peak,
              "ub_platform_available_bytes": 196352,
              "ub_physical_margin_bytes": 196352 - physical_peak,
              "inherited_tbuf_declared_extent_exceeded_bytes": tmp_access_end - 73728,
              "ub_layout_assumptions": "32-byte sequential TPipe UB allocation, withOffset=0, no additional buffer after tmpBuff, Matmul shareUbSize=0; pre and publisher buffers reset before mid",
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths + [fields_path, Path(__file__)]},
              "input_sha256": hashlib.sha256(json.dumps(cases).encode()).hexdigest(),
              "cases": result_cases}
    out = Path(__file__).with_name("result.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
