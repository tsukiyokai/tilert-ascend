"""Check the frozen B114 GMM2 dense/compact tile schedules on one CPU."""

import hashlib
import json
import os
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent
CORES = 24
PARALLEL = 4
BASE_M = 64
BASE_N = 512
K = 128
N = 6144
SLOT_ELEMENTS = 32768


def active_table(counts):
    end = 0
    records = []
    for expert, count in enumerate(counts):
        end += count
        if count:
            records.append((end << 32) | expert)
    return records


def schedule(counts, compact):
    output = [[] for _ in range(CORES)]
    groups = active_table(counts) if compact else counts
    for core in range(CORES):
        pre_count = offset_m = previous_end = cube_count = 0
        for slot, value in enumerate(groups):
            if compact:
                expert = value & 0xffffffff
                end = value >> 32
                count = end - previous_end
                previous_end = end
            else:
                expert, count = slot, value
            if count <= 0:
                continue
            expanded_m = 2 * count
            blocks_m = (expanded_m + BASE_M - 1) // BASE_M
            blocks_n = (N + BASE_N - 1) // BASE_N
            current_count = pre_count + blocks_m * blocks_n
            current_block = core if core >= pre_count else core + CORES
            while current_block < current_count:
                m_index, n_index = divmod(current_block - pre_count, blocks_n)
                single_m = min(BASE_M, expanded_m - m_index * BASE_M)
                single_n = min(BASE_N, N - n_index * BASE_N)
                scratch = SLOT_ELEMENTS * (core + (cube_count % PARALLEL) * CORES)
                output[core].append((
                    expert, expanded_m, offset_m, m_index, n_index,
                    single_m, single_n,
                    (offset_m + m_index * BASE_M) * K,
                    expert * N * K + n_index * BASE_N * K,
                    expert * N + n_index * BASE_N,
                    offset_m // 2 + m_index * BASE_M // 2,
                    scratch, cube_count >= PARALLEL,
                ))
                assert scratch + single_m * single_n <= (
                    SLOT_ELEMENTS * CORES * PARALLEL)
                cube_count += 1
                current_block += CORES
            pre_count = current_count % CORES
            offset_m += expanded_m
        assert offset_m == 2 * sum(counts)
    return output


def cases(rows, rng):
    first = [0] * 256
    first[0] = rows
    yield "first", first
    last = [0] * 256
    last[-1] = rows
    yield "last", last
    ends = [0] * 256
    ends[0], ends[-1] = rows // 2, rows - rows // 2
    yield "ends", ends
    spread = [0] * 256
    for expert in range(rows):
        spread[expert * 256 // rows] += 1
    yield "spread", spread
    for index in range(30):
        counts = [0] * 256
        for _ in range(rows // 8):
            for expert in rng.sample(range(256), 8):
                counts[expert] += 1
        yield "top8_" + str(index), counts


def main():
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    original = ROOT / "original"
    candidate = ROOT / "candidate"
    name = "grouped_matmul_antiquant_a8w4_msd.h"
    marker = "__aicore__ inline void GMMA8W4MSDCompute<mmType>::MMCompute("
    before = (original / name).read_text()
    after = (candidate / name).read_text()
    assert before.split(marker, 1)[1] == after.split(marker, 1)[1]
    ring_bytes = SLOT_ELEMENTS * 2 * CORES * PARALLEL
    allocation_bytes = 117440512
    user_bytes = allocation_bytes - 16 * 2**20
    metadata_bytes = 260 * 8
    assert ring_bytes == 6291456
    assert ring_bytes % 32 == 0
    assert ring_bytes + metadata_bytes <= user_bytes
    rng = random.Random(20261020)
    results = []
    digest = hashlib.sha256()
    total_tiles = 0
    for rows in (8, 64, 256):
        for route, counts in cases(rows, rng):
            assert len(counts) == 256 and sum(counts) == rows
            assert all(value >= 0 for value in counts)
            dense = schedule(counts, False)
            compact = schedule(counts, True)
            assert dense == compact, (rows, route)
            table = active_table(counts)
            assert table[-1] >> 32 == rows
            assert 1 <= len(table) <= min(rows, 256)
            words = ((len(table) + 4) // 4) * 4
            assert words <= 260 and words % 4 == 0
            tiles = sum(map(len, dense))
            total_tiles += tiles
            digest.update(json.dumps(counts).encode())
            results.append({"rows": rows, "route": route, "active": len(table),
                            "tiles": tiles, "schedule_exact": True})
    # Same active count with a different tail tests stale IDs, not just A.
    left = [1] * 8 + [0] * 248
    right = [0] * 248 + [1] * 8
    assert active_table(left) != active_table(right)
    assert len(active_table(left)) == len(active_table(right))
    result = {
        "scope": "CPU scalar schedule and address proof; no CANN arithmetic or NPU emulation",
        "npu_launched": False,
        "case_count": len(results),
        "input_sha256": digest.hexdigest(),
        "total_tiles_per_arm": total_tiles,
        "all_24_core_schedules_exact": True,
        "MMCompute_and_VectorCompute_suffix_unchanged": True,
        "ring_bytes": ring_bytes, "metadata_bytes": metadata_bytes,
        "verified_operator_allocation_bytes": allocation_bytes,
        "system_bytes_excluded": 16 * 2**20,
        "metadata_end_in_user_bytes": ring_bytes + metadata_bytes,
        "user_allocation_bytes": user_bytes,
        "same_A_different_experts_counterexample": True,
        "source": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), original / name, candidate / name,
                         candidate / "grouped_matmul_active_groups_a8w4.h",
                         candidate / "grouped_matmul.cpp"]
        },
        "cases": results,
    }
    (ROOT / "cpu-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"},
                     indent=2))


if __name__ == "__main__":
    main()

