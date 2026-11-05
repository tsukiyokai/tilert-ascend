"""CPU storage and address checks for a shared GU split-K timing arena."""
import argparse
import ast
from pathlib import Path
import gc
import hashlib
import json
import weakref

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checker", type=Path,
                    default=Path(__file__).with_name("placement_checker.py"))
parser.add_argument("--output", type=Path,
                    default=Path(__file__).with_name("result.json"))
args = parser.parse_args()
torch.set_num_threads(1)

TOP_K = 8
COLUMNS = 256
TILINGS = (6, 7, 18)


def paired_arenas(candidate, control, owner, backing):
    """Return both lists so the caller can retain every backing through replay."""
    if candidate[1] != control[1]:
        raise ValueError("sync_base differs")
    if len(candidate[0]) != 20 or len(control[0]) != 20:
        raise ValueError("unexpected scratch count")
    rows = 2 * TOP_K * candidate[0][9].shape[0]
    expected_full = (3 * rows, COLUMNS)
    expected_first = (rows, COLUMNS)
    if candidate[0][1].shape != expected_full or control[0][1].shape != expected_first:
        raise ValueError("unexpected GU product contract")
    if backing.shape != expected_full or backing.dtype != torch.int32:
        raise ValueError("backing must contain three complete INT32 planes")
    if backing.device != candidate[0][1].device or backing.device != control[0][1].device:
        raise ValueError("backing device differs")
    if not backing.is_contiguous() or backing.storage_offset() != 0:
        raise ValueError("backing must start at contiguous storage origin")
    if backing.stride() != (COLUMNS, 1):
        raise ValueError("backing stride differs")
    if backing.untyped_storage().nbytes() < backing.numel() * backing.element_size():
        raise ValueError("backing is too small")
    outputs = [list(candidate[0]), list(control[0])]
    chosen = candidate[0] if owner == "candidate" else control[0]
    for index, (left, right) in enumerate(zip(*outputs, strict=True)):
        if index == 1:
            continue
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            raise ValueError("other scratch contract differs")
        if index in TILINGS:
            if not torch.equal(left.view(torch.uint8), right.view(torch.uint8)):
                raise ValueError("prepared tiling bytes differ")
        else:
            outputs[0][index] = chosen[index]
            outputs[1][index] = chosen[index]
    outputs[0][1] = backing
    outputs[1][1] = backing.narrow(0, 0, rows)
    first = outputs[1][1]
    assert first.shape == expected_first and first.stride() == (COLUMNS, 1)
    assert first.is_contiguous() and first.storage_offset() == 0
    assert first.data_ptr() == backing.data_ptr()
    assert first.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    for index in TILINGS:
        assert outputs[0][index] is candidate[0][index]
        assert outputs[1][index] is control[0][index]
    return ((outputs[0], candidate[1]), (outputs[1], control[1]))


def fixture(tokens):
    candidate = [torch.empty(32, dtype=torch.uint8) for _ in range(20)]
    control = [torch.empty(32, dtype=torch.uint8) for _ in range(20)]
    for index in TILINGS:
        candidate[index].fill_(index)
        control[index].fill_(index)
    candidate[1] = torch.empty((6 * tokens * TOP_K, COLUMNS), dtype=torch.int32)
    control[1] = torch.empty((2 * tokens * TOP_K, COLUMNS), dtype=torch.int32)
    candidate[9] = torch.empty((tokens, 16), dtype=torch.bfloat16)
    control[9] = torch.empty((tokens, 16), dtype=torch.bfloat16)
    return (candidate, 123456), (control, 123456)


cases = 0
for tokens in (1, 2, 12, 32):
    original_candidate, original_control = fixture(tokens)
    used_backings = []
    for owner in ("candidate", "control"):
        backing = (original_candidate[0][1] if owner == "candidate"
                   else torch.empty_like(original_candidate[0][1]))
        assert all(backing.data_ptr() != prior.data_ptr() for prior in used_backings)
        used_backings.append(backing)
        arenas = paired_arenas(original_candidate, original_control, owner, backing)
        candidate, control = arenas
        rows = 2 * tokens * TOP_K
        backing.fill_(91234567)
        control[0][1].fill_(-37)
        assert torch.all(backing[:rows] == -37)
        assert torch.all(backing[rows:] == 91234567)
        candidate[0][1][:rows].fill_(17)
        candidate[0][1][rows:2 * rows].fill_(-2048)
        candidate[0][1][2 * rows:].fill_(2049)
        assert torch.all(control[0][1] == 17)
        owner_ref = weakref.ref(backing)
        del backing
        gc.collect()
        assert owner_ref() is not None
        assert candidate[0][1].data_ptr() == control[0][1].data_ptr()
        assert original_candidate[0][1].shape[0] == 3 * rows
        assert original_control[0][1].shape[0] == rows
        cases += 1

bounds = 0
for tokens in range(1, 33):
    routes = tokens * TOP_K
    plane = 2 * routes * COLUMNS
    for start in range(routes):
        for count in range(1, min(tokens, routes - start) + 1):
            begin = 2 * start * COLUMNS
            end = 2 * (start + count) * COLUMNS
            assert 0 <= begin < end <= plane
            # Every control high/low row read and the full C write fit in plane0.
            high_last = (2 * start + count - 1) * COLUMNS + COLUMNS - 1
            low_last = (2 * start + 2 * count - 1) * COLUMNS + COLUMNS - 1
            assert begin <= high_last < low_last < end
            # The candidate accesses the same rows shifted by exactly q planes.
            for part in range(3):
                assert part * plane <= part * plane + begin
                assert part * plane + low_last < (part + 1) * plane <= 3 * plane
            bounds += 1

candidate, control = fixture(12)
invalid = [
    torch.empty_like(control[0][1]),
    torch.empty_like(candidate[0][1], dtype=torch.int64),
    torch.empty((candidate[0][1].shape[0], COLUMNS * 2), dtype=torch.int32)[:, ::2],
    torch.empty((candidate[0][1].shape[0] + 1, COLUMNS), dtype=torch.int32)[1:],
]
rejections = 0
for backing in invalid:
    try:
        paired_arenas(candidate, control, "candidate", backing)
    except ValueError:
        rejections += 1
    else:
        raise AssertionError("invalid backing accepted")
assert rejections == len(invalid)

report = {
    "torch": torch.__version__,
    "cpu_storage_cases": cases,
    "all_valid_group_interval_checks": bounds,
    "invalid_backing_rejections": rejections,
    "tiling_tensor_objects_preserved": True,
    "full_and_prefix_data_ptrs_equal": True,
    "two_independent_backings_checked": True,
    "control_write_leaves_extra_planes_unchanged": True,
    "native_read_bounds_proved_from_source_formulas": True,
    "device_format_and_native_execution_checked": False,
}
checker_bytes = args.checker.read_bytes()
checker_sha = hashlib.sha256(checker_bytes).hexdigest()
assert checker_sha == "63d21886fdedd94cfa534c4acc8a23014af0e48f3e7daa38d3e813f0ab6e5aa6"
parsed = ast.parse(checker_bytes, filename=str(args.checker))
helper = next(node for node in parsed.body
              if isinstance(node, ast.FunctionDef)
              and node.name == "shared_timing_arenas")
module = ast.Module(body=[helper], type_ignores=[])
namespace = {}
exec(compile(module, str(args.checker), "exec"), namespace)
actual_helper = namespace["shared_timing_arenas"]
actual_cases = 0
for tokens in (1, 2, 12, 32):
    for owner_planes in (1, 3):
        candidate, control = fixture(tokens)
        owner, recipient = ((candidate, control) if owner_planes == 3
                            else (control, candidate))
        owned, shared = actual_helper(torch, owner, recipient)
        full, prefix = sorted((owned[0][1], shared[0][1]),
                              key=lambda value: value.numel(), reverse=True)
        assert full.data_ptr() == prefix.data_ptr()
        assert full.stride() == prefix.stride() == (COLUMNS, 1)
        assert full.storage_offset() == prefix.storage_offset() == 0
        assert full.untyped_storage().nbytes() >= full.numel() * 4
        full.fill_(42)
        prefix.fill_(-19)
        assert torch.all(full[:prefix.shape[0]] == -19)
        assert torch.all(full[prefix.shape[0]:] == 42)
        for index in TILINGS:
            assert owned[0][index] is owner[0][index]
            assert shared[0][index] is recipient[0][index]
        arenas = {"owner": owned, "recipient": shared}
        timing_arenas = dict(arenas)
        full_ref = weakref.ref(full)
        arenas.update({"owner": owner, "recipient": recipient})
        del full, prefix, owned, shared
        gc.collect()
        assert full_ref() is not None
        assert (timing_arenas["owner"][0][1].data_ptr()
                == timing_arenas["recipient"][0][1].data_ptr())
        actual_cases += 1
report.update({
    "actual_checker_cpu_cases": actual_cases,
    "actual_checker_sha256": checker_sha,
    "actual_checker_restore_keeps_backing_alive": True,
    "contract_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
})
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
