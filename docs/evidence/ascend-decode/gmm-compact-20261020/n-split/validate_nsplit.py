"""Single-die equivalence of compact GMM1 N256/N128/N64 tiles."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
import torch_npu

sys_path = str(Path(__file__).resolve().parent.parent)
import sys
sys.path.insert(0, sys_path)
import gmm1_inputs
from vllm_ascend.utils import enable_custom_op


def routes(rows):
    counts = torch.zeros(256, dtype=torch.int64)
    counts[255] = rows
    yield "last", counts
    counts = torch.zeros(256, dtype=torch.int64)
    counts[0], counts[255] = rows - 1, 1
    yield "ends", counts
    generator = torch.Generator().manual_seed(rows)
    counts = torch.bincount(torch.randint(256, (rows,), generator=generator),
                            minlength=256)
    yield "random", counts
    counts = torch.zeros(256, dtype=torch.int64)
    counts[torch.arange(rows) * 256 // rows] = 1
    yield "spread", counts


ARMS = ("N256", "N128", "N64")
TAGS = {"N256": 0, "N128": 32, "N64": 64}


def tagged_counts(counts, device, tag):
    backing = torch.empty(272, dtype=torch.int64, device=device)
    offset = ((tag - (backing.data_ptr() & 127)) & 127) // 8
    result = backing[offset:offset + 256]
    assert result.data_ptr() & 96 == tag
    result.copy_(counts)
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:15")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    total = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(1536 * 2**20 / total)
    torch.npu.reset_peak_memory_stats()
    assert enable_custom_op()
    inputs = gmm1_inputs.build(args.device, active=256, banks=1,
                              capture_reference=True)
    op = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2
    report = {"scope": "single-die shared-load correctness only", "cases": []}
    def digest(tensor):
        raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()
    tensors = [inputs.x, inputs.x_scale, *inputs.reference.values()]
    report["input_sha256"] = [digest(tensor) for tensor in tensors]
    start = time.monotonic()
    for rows in (8, 64, 256):
        kwargs = inputs.kwargs()
        kwargs["x"] = inputs.x[:rows].clone()
        kwargs["x_scale"] = inputs.x_scale[:rows].clone()
        initial = torch.zeros(256, dtype=torch.int64)
        initial[-1] = rows
        groups = {arm: tagged_counts(initial, args.device, TAGS[arm]) for arm in ARMS}
        graphs, outputs = {}, {}
        for name in ARMS:
            kwargs["group_list"] = groups[name]
            limit = 0.0
            op(**(kwargs | {"swiglu_limit": limit}))
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                pair = op(**(kwargs | {"swiglu_limit": limit}))
            graphs[name], outputs[name] = graph, pair
            torch.npu.synchronize()
        # One graph and workspace per arm, with group-list contents changing.
        for route, counts in list(routes(rows)) * 2:
            for group in groups.values():
                group.copy_(counts)
            torch.npu.synchronize()
            actual = {}
            for name in ARMS:
                graphs[name].replay()
                torch.npu.synchronize()
                actual[name] = tuple(x.cpu() for x in outputs[name])
            equal = [torch.equal(a, b) for a, b in
                     zip(actual["N256"], actual["N128"])]
            equal += [torch.equal(a, b) for a, b in zip(actual["N256"], actual["N64"])]
            finite = bool(torch.isfinite(actual["N64"][1]).all())
            record = {"rows": rows, "route": route, "active": int((counts > 0).sum()),
                      "N128_exact": equal[:2], "N64_exact": equal[2:],
                      "all_exact": all(equal),
                      "finite": finite,
                      "output_sha256": [digest(t) for t in actual["N64"]]}
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
            if not all(equal) or not finite:
                torch.save(actual | {"counts": counts},
                           args.output.with_suffix(".failure.pt"))
                report["status"] = "fail"
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                raise SystemExit(1)
        del graphs, outputs, kwargs, pair, graph
    report.update(status="pass", elapsed_s=time.monotonic() - start,
                  peak_reserved_bytes=torch.npu.max_memory_reserved(),
                  opp_path=os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    main()
