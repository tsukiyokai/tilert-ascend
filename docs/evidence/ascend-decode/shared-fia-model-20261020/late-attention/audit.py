"""Inspect saved late-attention tensors on CPU without a tolerance gate."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bits(value):
    return value.contiguous().view(torch.uint8)


def tensor_sha(value):
    return hashlib.sha256(bits(value).numpy().tobytes()).hexdigest()


def metrics(value, reference):
    a, b = value.double(), reference.double()
    delta = (a - b).norm().item()
    norm_a, norm_b = a.norm().item(), b.norm().item()
    cosine = a.flatten().dot(b.flatten()).item() / (norm_a * norm_b)
    query_error = (a - b).flatten(1).norm(dim=1)
    return {"bitexact": torch.equal(bits(value), bits(reference)),
            "nrmse": delta / norm_b, "maxabs": (a - b).abs().max().item(),
            "cosine": cosine,
            "different_values": int((value != reference).sum()),
            "query_nrmse": (query_error / b.flatten(1).norm(dim=1)).tolist()}


root, output = map(Path, sys.argv[1:])
torch.set_num_threads(1)
report = json.loads((root / "device-01/result.json").read_text())
old_report = json.loads((root / "raw/rank-0.json").read_text())
assert report["status"] == "attention_only_complete"
assert not report["process_group_initialized"]
assert sha(root / "raw/rank-0.json") == report["original_report_sha256"]
for path, expected in report["source_sha256"].items():
    assert sha(path) == expected, path
result = {"status": "raw_tensor_checks_pass",
          "device_report_sha256": sha(root / "device-01/result.json"),
          "auditor_sha256": sha(__file__),
          "source_files_rehashed": len(report["source_sha256"]),
          "layers": [], "artifact_sha256": {}, "cache_pairs": 0,
          "cache_bytes_compared": 0,
          "shared_rebuilt_pairs": 0, "shared_rebuilt_elements": 0}
previous, query_hashes, latent_hashes, producer_sets = None, [], [], []
for number in range(66, 74):
    path = root / f"device-01/layer-{number:02d}.pt"
    old_path = root / f"raw/rank-0/layer-{number:02d}.pt"
    old_sha = sha(old_path)
    expected_sha = next(v for k, v in old_report["artifact_sha256"].items()
                        if k.endswith(f"/rank-0/layer-{number:02d}.pt"))
    assert old_sha == expected_sha
    entry = next(x for x in report["layers"] if x["layer"] == number)
    assert entry["input_sha256"] == old_sha
    result["artifact_sha256"].update({str(path): sha(path),
                                      str(old_path): old_sha})
    data, old = [torch.load(p, map_location="cpu", weights_only=True,
                           mmap=True) for p in (path, old_path)]
    assert old["positions"] == list(range(8192, 8224))
    for name, shape in (("q", (32, 4, 512)), ("qr", (32, 4, 64))):
        assert data[name].shape == shape and data[name].dtype == torch.bfloat16
        assert bool(torch.isfinite(data[name]).all())
    query_hashes.append(tensor_sha(data["q"]))
    latent_hashes.append(tensor_sha(data["new_cache_rows"]["latent"]))
    row = {"layer": number, "cache": {}, "shared_rebuilt_bitexact": {},
           "saved_outputs_disjoint": {},
           "nonzero_every_output_query": True, "differences_vs_sfa": {}}
    for stage, shape in (("attention", (32, 4, 512)), ("pre_o", (32, 1024))):
        values = data[stage]
        for value in values.values():
            assert value.shape == shape and value.dtype == torch.bfloat16
            assert bool(torch.isfinite(value).all())
            assert bool((value.flatten(1).abs().sum(1) > 0).all())
        left, right = values["shared_fia"], values["rebuilt_fia"]
        exact = torch.equal(bits(left), bits(right))
        left_end = left.data_ptr() + left.numel() * left.element_size()
        right_end = right.data_ptr() + right.numel() * right.element_size()
        disjoint = (left_end <= right.data_ptr()
                    or right_end <= left.data_ptr())
        assert exact and disjoint
        row["shared_rebuilt_bitexact"][stage] = exact
        row["saved_outputs_disjoint"][stage] = disjoint
        row["differences_vs_sfa"][stage] = metrics(left, values["sfa"])
        result["shared_rebuilt_pairs"] += 1
        result["shared_rebuilt_elements"] += left.numel()
    assert data["new_cache_rows"].keys() == old["cache_rows"].keys()
    for name, value in data["new_cache_rows"].items():
        expected = old["cache_rows"][name]
        assert value.dtype == expected.dtype
        assert value.numel() == expected.numel()
        value = value.reshape_as(expected)
        assert bool(torch.isfinite(value).all())
        assert torch.equal(bits(value), bits(expected))
        row["cache"][name] = {"bitexact": True, "sha256": tensor_sha(value)}
        result["cache_pairs"] += 1
        result["cache_bytes_compared"] += value.numel() * value.element_size()
    ids = data["indices"].reshape(32, 2048)
    assert ids.dtype == torch.int32
    sorted_ids = ids.sort(1).values
    positions = torch.arange(8192, 8224)[:, None]
    assert bool(((ids >= 0) & (ids <= positions)).all())
    assert bool((sorted_ids.diff() > 0).all())
    assert torch.equal(sorted_ids, old["indices"].sort(1).values)
    producer = number in (66, 70)
    if producer:
        previous = ids.clone()
        producer_sets.append(sorted_ids.clone())
    else:
        assert torch.equal(bits(ids), bits(previous))
    lifecycle = "refreshed" if producer else "reused"
    assert entry["shared_mask_lifecycle"] == lifecycle
    row["indices"] = {
        "causal_unique": True, "same_set_old_M1_every_query": True,
        "ordered_equal_old_M1": torch.equal(bits(ids), bits(old["indices"])),
        "different_ordered_entries": int((ids != old["indices"]).sum()),
        "consumer_ordered_equal_producer": None if producer else True,
    }
    result["layers"].append(row)
result.update(
    distinct_query_tensors=len(set(query_hashes)),
    distinct_new_latent_tensors=len(set(latent_hashes)),
    producer_sets_differ=not torch.equal(*producer_sets),
    peak_allocated_bytes=max(x["peak_allocated"] for x in report["layers"]),
    peak_reserved_bytes=max(x["peak_reserved"] for x in report["layers"]),
    torch_npu_imported="torch_npu" in sys.modules, performance_measured=False,
    scope=("rank0 local attention on original M1 teacher inputs; "
           "no TP16 propagation or teacher-label acceptance"),
)
assert result["distinct_query_tensors"] == 8
assert result["distinct_new_latent_tensors"] == 8
assert not result["torch_npu_imported"]
output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
keys = ("status", "shared_rebuilt_pairs", "cache_pairs",
        "cache_bytes_compared", "peak_allocated_bytes")
print(json.dumps({k: result[k] for k in keys}))
