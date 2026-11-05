"""Scalar transcription of the frozen C_TEMPLATE top-k scan only."""

import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def int32(value):
    value &= MASK32
    return value - (1 << 32) if value >= (1 << 31) else value


class BoundedIndices:
    def __init__(self, indices, valid_count):
        self.indices = indices
        self.valid_count = valid_count
        self.attempts = []

    def get_value(self, index):
        self.attempts.append(index)
        if not 0 <= index < self.valid_count:
            raise IndexError(index)
        return self.indices[index]


def original_first_read(topk, cursor, valid_count):
    return topk.get_value(cursor)


def patched_first_read(topk, cursor, valid_count):
    sparse_indices = -1
    if cursor < valid_count:
        sparse_indices = topk.get_value(cursor)
    return sparse_indices


def calc_sinner_topk_begin(case, state, info, topk, first_read):
    """C++ lines954-1021, with bounded reads and explicit integer casts."""
    threshold = case["threshold"]
    block_size = case["block_size"]
    base_size = case["base_size"]
    threshold_sparse_count = (threshold + block_size - 1) // block_size
    valid_count = min(len(case["indices"]), threshold_sparse_count)
    cursor = state["cursor"]
    offset = state["offset"]

    sparse_indices = first_read(topk, cursor, valid_count)
    if sparse_indices == -1 or cursor == valid_count:
        info["actual_size"] = 0
        info["aligned_size"] = 0
        info["tail"] = 0
        if cursor == 0:
            info["zero_output_calls"] += 1
        return

    sparse_len = 0
    block_begin = (sparse_indices * block_size) & MASK64
    block_end = min((block_begin + block_size) & MASK64, threshold)
    block_len = int32((block_end - block_begin) & MASK64)
    if block_len > int32(offset):
        sparse_len = (sparse_len + block_len - offset) & MASK32

    first_valid_flag = False
    if cursor > 0:
        info["saved_cursor"] = cursor
        info["saved_offset"] = offset
    elif cursor == 0 and sparse_len > 0:
        info["saved_cursor"] = cursor
        info["saved_offset"] = 0
        first_valid_flag = True

    for topk_index in range(cursor + 1, valid_count):
        sparse_indices = topk.get_value(topk_index)
        if sparse_indices == -1:
            state["cursor"] = topk_index
            state["offset"] = 0
            break
        block_begin = (sparse_indices * block_size) & MASK64
        if block_begin >= threshold:
            continue
        if not first_valid_flag and state["cursor"] == 0:
            info["saved_cursor"] = topk_index
            info["saved_offset"] = 0
            first_valid_flag = True
        block_end = min((block_begin + block_size) & MASK64, threshold)
        block_len = (block_end - block_begin) & MASK64
        sparse_len = (sparse_len + block_len) & MASK32
        if sparse_len >= base_size:
            state["cursor"] = topk_index
            state["offset"] = (block_len - (sparse_len - base_size)) & MASK64
            sparse_len = base_size
            break
        if topk_index == valid_count - 1:
            state["cursor"] = valid_count
            state["offset"] = 0

    info["actual_size"] = sparse_len
    info["aligned_size"] = ((sparse_len + 31) // 32) * 32
    info["tail"] = 0 if sparse_len == base_size else sparse_len
    if state["cursor"] == 0 and sparse_len == 0:
        info["zero_output_calls"] += 1


def run_call(case, state, first_read):
    valid_count = min(
        len(case["indices"]),
        (case["threshold"] + case["block_size"] - 1) // case["block_size"],
    )
    topk = BoundedIndices(case["indices"], valid_count)
    # Preexisting RunInfo fields must remain untouched on the same branches.
    info = {
        "saved_cursor": 91,
        "saved_offset": 99,
        "actual_size": 77,
        "aligned_size": 77,
        "tail": 77,
        "zero_output_calls": 0,
    }
    try:
        calc_sinner_topk_begin(case, state, info, topk, first_read)
        status = "defined"
    except IndexError as error:
        status = "out_of_bounds"
        info["rejected_index"] = error.args[0]
    return {"status": status, "state": state, "info": info}, topk.attempts


def make_cases():
    common = {"block_size": 1, "base_size": 512, "cursor": 0, "offset": 0}
    specs = [
        ("count0", [], 0, 0, 0, 2),
        ("count1", [0], 1, 0, 0, 3),
        ("count1_terminal", [0], 1, 1, 0, 2),
        ("count3_partial", [0, 1, 2], 3, 0, 0, 3),
        ("count3_terminal", [0, 1, 2], 3, 3, 0, 2),
        ("count3_middle", [0, 1, 2], 3, 1, 0, 3),
        ("sentinel_first", [-1, -1, -1], 3, 0, 0, 3),
        ("sentinel_middle", [0, -1, -1], 3, 0, 0, 3),
        ("sentinel_last", [0, 1, -1], 3, 0, 0, 3),
        ("future_first", [9, 0, 1], 3, 0, 0, 3),
        ("future_middle", [0, 9, 1], 3, 0, 0, 3),
        ("future_last", [0, 1, 9], 3, 0, 0, 3),
        ("future_only", [9], 1, 0, 0, 3),
        ("exact_tile512", list(range(512)), 512, 0, 0, 3),
        ("partial_tile513", list(range(513)), 513, 0, 0, 4),
    ]
    cases = [
        dict(common, name=name, indices=indices, threshold=threshold,
             cursor=cursor, offset=offset, calls=calls)
        for name, indices, threshold, cursor, offset, calls in specs
    ]
    cases.append(dict(common, name="causal_block_tail", indices=[0, 1, 2],
                      threshold=10, block_size=4, calls=3))
    return cases


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path("/workspace/src/vllm-ascend-unified/csrc/attention/"
                       "sparse_flash_attention/op_kernel")
    header = source_root / "arch22/sparse_flash_attention_kernel_mla.h"
    header_bytes = header.read_bytes()
    assert digest_bytes(header_bytes) == (
        "cec94843ad2144f80e5a504b33f92a682ba8908037435de65da26f5682ce7cdd"
    )
    lines = header_bytes.decode().splitlines(keepends=True)
    fragment = "".join(lines[945:1022])
    (output_dir / "CalcSinnerTopKBegin.original.cpp").write_text(fragment)
    install = Path("/opt/ascend/cann-9.2.0/aarch64-linux/"
                   "ascend_toolkit_install.info")
    api = Path("/opt/ascend/cann-9.2.0/aarch64-linux/ascendc/include/"
               "basic_api/impl/kernel_operator_data_copy_check.h")
    cases = make_cases()
    case_bytes = (json.dumps(cases, sort_keys=True) + "\n").encode()
    (output_dir / "input-cases.json").write_bytes(case_bytes)
    case_results = []
    equal_calls = 0
    prevented_reads = 0
    for case in cases:
        state = {"cursor": case["cursor"], "offset": case["offset"]}
        calls = []
        for call_index in range(case["calls"]):
            before = copy.deepcopy(state)
            original, original_reads = run_call(
                case, copy.deepcopy(before), original_first_read,
            )
            patched, patched_reads = run_call(
                case, copy.deepcopy(before), patched_first_read,
            )
            assert patched["status"] == "defined"
            if original["status"] == "defined":
                assert patched == original
                assert patched_reads == original_reads
                equal_calls += 1
            else:
                assert original_reads == [before["cursor"]]
                assert patched_reads == []
                prevented_reads += 1
            state = patched["state"]
            calls.append({
                "call": call_index,
                "input_cursor": before,
                "original_status": original["status"],
                "patched_output": patched,
                "original_read_attempts": len(original_reads),
                "patched_reads": len(patched_reads),
                "defined_read_trace_equal": (
                    patched_reads == original_reads
                    if original["status"] == "defined" else None
                ),
            })
        case_results.append({"name": case["name"], "calls": calls})
    result = {
        "scope": "C_TEMPLATE scalar scan only; explicit cases, not a device proof.",
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256": digest_bytes(Path(__file__).read_bytes()),
        "header_path": str(header),
        "header_sha256": digest_bytes(header_bytes),
        "fragment_sha256": digest_bytes(fragment.encode()),
        "input_cases_sha256": digest_bytes(case_bytes),
        "patch_sha256": digest_bytes(
            Path("/tmp/c1-index-bounds-20261019.patch").read_bytes()
        ),
        "toolkit_identity": {
            "install_info_path": str(install),
            "install_info_sha256": digest_bytes(install.read_bytes()),
            "install_info_text": install.read_text(),
            "header_path": str(api),
            "header_resolved_path": str(api.resolve()),
            "header_sha256": digest_bytes(api.read_bytes()),
        },
        "totals": {"cases": len(cases), "calls": sum(c["calls"] for c in cases),
                   "original_defined_calls_equal": equal_calls,
                   "original_invalid_reads_prevented": prevented_reads,
                   "patched_invalid_reads": 0, "extra_reads": 0},
        "limitations": [
            "The common scalar body transcribes the source statements; only the first-read function differs, exactly as in the patch.",
            "Nonnegative thresholds and small legal integer inputs only; no claim about arbitrary integer overflow or other kernel reads.",
            "Drain calls invoke this scan only; the separate ring/flag source proof remains required.",
            "future_last preserves the existing non-advancing cursor; this patch does not repair that separate behavior.",
        ],
        "results": case_results,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"totals": result["totals"],
                      "results_sha256": digest_bytes(
                          (output_dir / "results.json").read_bytes()),
                      "source_sha256": result["source_sha256"]}))


if __name__ == "__main__":
    main()
