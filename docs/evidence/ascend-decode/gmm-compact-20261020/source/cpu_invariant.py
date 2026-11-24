"""Compare dense and active expert schedules for the production A8W4 chunk."""

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path
import random
import re
import socket
import time


NAME = 'grouped_matmul_swiglu_quant_v2'
PRODUCTION = Path('/workspace/experiments/gmm-fastpath-20261018/prod-kernel')
KERNEL = (PRODUCTION / 'overlay/vendors/custom_transformer/op_impl/ai_core'
          / 'tbe/custom_transformer_impl/ascendc' / NAME)
HOST = (Path('/workspace/src/vllm-ascend-unified/csrc/gmm') / NAME / 'op_host')
SOC = Path('/opt/ascend/cann-9.2.0/aarch64-linux/data/platform_config'
           '/Ascend910_9392.ini')


def read_integer(text, name):
    return int(re.search(r'\b' + name + r'\s*=\s*(\d+)', text).group(1))


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def active_table(counts):
    end = 0
    active = []
    for expert, count in enumerate(counts):
        end += count
        if count > 0:
            active.append((expert, end))
    return active


def validate(counts, rows, limit):
    if len(counts) != 256 or any(count < 0 for count in counts):
        raise ValueError('Exactly 256 nonnegative expert counts are required')
    if sum(counts) != rows or not 0 < rows <= limit:
        raise ValueError('Positive, single-chunk M must equal sum(counts)')


def tile(expert, prefix_rows, count, m_index, n_index, serial, config):
    base_m, base_n, k, n, cores = config
    tail_n = n_index * base_n
    single_m = min(base_m, 2 * count - m_index * base_m)
    single_n = min(base_n, n - tail_n)
    return (serial, serial % cores, expert, 2 * count, m_index, n_index,
            single_m, single_n,
            (2 * prefix_rows + m_index * base_m) * k,
            (2 * prefix_rows + m_index * base_m) * n + tail_n,
            expert * k * n + tail_n * k,
            expert * n + tail_n)


def dense_mid(counts, config):
    base_m, base_n, k, n, cores = config
    by_core = [[] for _ in range(cores)]
    for core in range(cores):
        previous_m = x_base = y_base = prefix = previous_end = 0
        pre_count = absolute_tiles = 0
        for expert, count in enumerate(counts):
            # UpdateMnConfig runs before SetMNConfig, including for zero slots.
            x_base += previous_m * k
            y_base += previous_m * n
            end = previous_end + count
            expanded_m = (end - previous_end) * 2
            previous_end = end
            previous_m = expanded_m
            if expanded_m <= 0:
                continue
            blocks_m = (expanded_m + base_m - 1) // base_m
            blocks_n = (n + base_n - 1) // base_n
            cur_count = pre_count + blocks_m * blocks_n
            cur_block = core if core >= pre_count else core + cores
            while cur_block < cur_count:
                local = cur_block - pre_count
                m_index, n_index = divmod(local, blocks_n)
                record = tile(expert, prefix, count, m_index, n_index,
                              absolute_tiles + local, config)
                # Compare actual incremental X/Y offsets to the closed form.
                assert record[8] == x_base + m_index * k * base_m
                assert record[9] == y_base + m_index * base_m * n + n_index * base_n
                assert record[1] == core
                by_core[core].append(record)
                cur_block += cores
            absolute_tiles += blocks_m * blocks_n
            pre_count = cur_count % cores
            prefix = end
    return by_core


def compact_mid(active, config):
    base_m, base_n, _, n, cores = config
    by_core = [[] for _ in range(cores)]
    previous_end = serial = 0
    for expert, end in active:
        count = end - previous_end
        for m_index in range((2 * count + base_m - 1) // base_m):
            for n_index in range((n + base_n - 1) // base_n):
                record = tile(expert, previous_end, count,
                              m_index, n_index, serial, config)
                by_core[serial % cores].append(record)
                serial += 1
        previous_end = end
    return by_core


def post_partition(rows, vectors):
    each = (rows + vectors - 1) // vectors
    used = min(rows, vectors)
    tail = rows - (each - 1) * used
    return [(each * core if core < tail else (each - 1) * core + tail,
             each if core < tail else each - 1)
            for core in range(vectors)]


def dense_post(counts, rows, vectors, n):
    result = []
    for core, (begin, count) in enumerate(post_partition(rows, vectors)):
        if count == 0:
            continue
        relative = begin
        expert = remaining = None
        for index, size in enumerate(counts):
            if relative >= 0 and relative - size < 0:
                expert, remaining = index, size - relative
            relative -= size
        assert expert is not None
        for row in range(begin, begin + count):
            if remaining == 0:
                # Literal count-table UpdateAuxiliaryMatrix, before a valid row.
                expert += 1
                while counts[expert] == 0:
                    expert += 1
                remaining = counts[expert]
            result.append((core, row, expert, 2 * row * n,
                           row, expert * n, row * (n // 2), row))
            remaining -= 1
    return result


def compact_post(active, rows, vectors, n):
    ends = [end for _, end in active]
    result = []
    for core, (begin, count) in enumerate(post_partition(rows, vectors)):
        # An unused core can have begin == M and upper_bound == active_count.
        if count == 0:
            continue
        slot = bisect_right(ends, begin)
        assert slot < len(active)
        for row in range(begin, begin + count):
            if row == ends[slot]:
                slot += 1
            assert slot < len(active) and row < ends[slot]
            expert = active[slot][0]
            result.append((core, row, expert, 2 * row * n,
                           row, expert * n, row * (n // 2), row))
    return result


def cases(rows):
    def distribute(name, experts):
        counts = [0] * 256
        for row in range(rows):
            counts[experts[row % len(experts)]] += 1
        return name, counts

    result = [distribute('single-' + str(expert), [expert])
              for expert in [0, 127, 255]]
    result.extend([
        distribute('both-ends', [0, 255]),
        distribute('uniform-256', list(range(256))),
        distribute('uniform-spread-8', [0, 32, 64, 96, 128, 160, 192, 255]),
        distribute('empty-first-and-last', list(range(1, 255))),
        distribute('holes', [1, 7, 93, 129, 254]),
    ])
    counts = [0] * 256
    counts[0] = min(rows, 65)
    counts[255] = rows - counts[0]
    result.append(('tile-tail-65', counts))
    for seed in range(12):
        rng = random.Random(20261020 + rows * 100 + seed)
        counts = [0] * 256
        for _ in range(rows):
            counts[rng.randrange(256)] += 1
        result.append(('random-' + str(seed), counts))
    return result


def counterexamples(limit):
    ends = [4, 8]
    boundary = {'ends': ends, 'start': 4,
                'strict_slot': bisect_right(ends, 4),
                'nonstrict_slot': bisect_left(ends, 4)}
    inactive_start, inactive_count = post_partition(8, 48)[8]
    inactive = {'M': 8, 'core': 8, 'start': inactive_start,
                'task_count': inactive_count, 'active_ends': [8],
                'unguarded_slot': bisect_right([8], inactive_start),
                'active_count': 1}
    rejected = []
    for name, rows, counts in [
            ('sum-less-than-M', 8, [7] + [0] * 255),
            ('sum-more-than-M', 8, [9] + [0] * 255),
            ('negative-count', 8, [9, -1] + [0] * 254),
            ('not-single-chunk', limit + 1, [limit + 1] + [0] * 255),
            ('all-zero-empty-chunk', 0, [0] * 256)]:
        try:
            validate(counts, rows, limit)
        except ValueError as error:
            rejected.append({'name': name, 'rejected': True, 'reason': str(error)})
        else:
            raise AssertionError(name)
    return {
        'use-original-expert-id': {'active': [[255, 8]],
                                  'correct_weight_expert': 255,
                                  'wrong-active-slot-as-expert': 0},
        'strict-end-search': boundary,
        'unused-core-must-not-dereference': inactive,
        'end-is-original-row-unit': {'counts': [1, 7], 'row': 1,
                                     'ends': [1, 8], 'correct_slot': 1,
                                     'wrong-doubled-ends': [2, 16],
                                     'wrong_slot': bisect_right([2, 16], 1)},
        'curIdx-is-not-current-vector-row': {
            'start': 0, 'innerLoopNum': 1,
            'curIdx_after_customDataCopyIn': 1,
            'actual_row_before_VectorCompute': 0,
            'ends': [1, 8], 'correct_slot': 0,
            'wrong_slot_if_searching_curIdx': bisect_right([1, 8], 1)},
        'invalid-domain': rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-root', type=Path, default=KERNEL)
    parser.add_argument('--host-root', type=Path, default=HOST)
    parser.add_argument('--soc', type=Path, default=SOC)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('cpu-invariant-result.json'))
    args = parser.parse_args()
    began = time.perf_counter()
    host_header = args.host_root / (NAME + '_tiling.h')
    text = host_header.read_text()
    base_m = read_integer(text, 'A8W4_BASEM')
    base_n = read_integer(text, 'A8W4_BASEN')
    soc = args.soc.read_text()
    cores = read_integer(soc, 'cube_core_cnt')
    vectors = read_integer(soc, 'vector_core_cnt')
    k, n = 6144, 256
    limit = (64 * 1024 * 1024 // 2) // (k + 2 * n * 2)
    config = base_m, base_n, k, n, cores
    result = {'scope': 'CPU schedules and addresses only; no numerical kernel '
                      'execution, device memory publication or runtime claim',
              'hostname': socket.gethostname(), 'parameters': {
                  'E': 256, 'K': k, 'N': n, 'baseM': base_m, 'baseN': base_n,
                  'cube_cores': cores, 'vector_cores': vectors, 'mLimit': limit,
                  'format': 'FRACTAL_NZ', 'isSingleTensor': 1,
                  'groupListType': 1, 'quantGroupNum': 1,
                  'mid_address_units': 'X/weight:int4 elements; Y:half elements; '
                                       'weight_scale:uint64 elements',
                  'post_address_units': 'MM:half; xscale/assist/yscale:float; '
                                        'quantized_output:int8'},
              'cases': [], 'sources': []}
    for rows in [8, 64, 256]:
        for name, counts in cases(rows):
            validate(counts, rows, limit)
            active = active_table(counts)
            dense = dense_mid(counts, config)
            compact = compact_mid(active, config)
            assert dense == compact, ('mid', rows, name)
            dense_rows = dense_post(counts, rows, vectors, n)
            compact_rows = compact_post(active, rows, vectors, n)
            assert dense_rows == compact_rows, ('post', rows, name)
            assert sorted(row[1] for row in dense_rows) == list(range(rows))
            coverage = []
            for per_core in dense:
                for record in per_core:
                    _, _, _, _, m_index, _, single_m, _, _, y, _, _ = record
                    assert 0 <= y < 2 * rows * n
                    coverage.extend(range(y // n, y // n + single_m))
            assert sorted(coverage) == list(range(2 * rows))
            payload = json.dumps([dense, dense_rows], separators=(',', ':')).encode()
            result['cases'].append({
                'M': rows, 'name': name, 'counts': counts, 'active': active,
                'tiles': sum(map(len, dense)), 'post_rows': len(dense_rows),
                'per_cube_tiles': list(map(len, dense)),
                'mid_and_post_match': True,
                'schedule_sha256': hashlib.sha256(payload).hexdigest()})
    sources = [host_header, args.host_root / (NAME + '_base_tiling.cpp'), args.soc]
    sources += [args.kernel_root / (NAME + suffix) for suffix in [
        '_a8w4_msd_mid.h', '_a8w4_msd_post.h', '_a8w4_msd_pre.h',
        '_a8w4_msd_pipeline.h', '_utils.h']]
    result['sources'] = [identity(path) for path in sources]
    result['counterexamples'] = counterexamples(limit)
    result['case_count'] = len(result['cases'])
    result['tiles_checked'] = sum(row['tiles'] for row in result['cases'])
    result['post_rows_checked'] = sum(row['post_rows'] for row in result['cases'])
    result['all_passed'] = True
    result['wall_seconds'] = time.perf_counter() - began
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in [
        'hostname', 'parameters', 'case_count', 'tiles_checked',
        'post_rows_checked', 'all_passed', 'wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
