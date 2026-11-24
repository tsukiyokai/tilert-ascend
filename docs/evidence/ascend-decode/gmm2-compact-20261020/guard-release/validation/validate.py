"""Replay the production NZ W4A8 down projection against CPU MSD arithmetic."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, "/workspace/experiments/gmm-compact-20261020")

import torch
import torch_npu
import gmm1_inputs
from vllm.config import set_current_vllm_config
from tilert.models.glm_5_2_ascend.reference_weights import (
    unpack_w4, w4_msd_product,
)


def digest(tensor):
    raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def build_bank(device):
    generator = torch.Generator().manual_seed(20261020)
    values = torch.randint(-128, 128, (256, 3072, 128),
                           dtype=torch.int8, generator=generator)
    scale = torch.rand((256, 6144, 1), generator=generator) * .015 + .005
    bias = torch.rand((256, 6144, 1), generator=generator) * .03 - .015
    raw = {"w13_weight": torch.zeros(256, 128, 6144, dtype=torch.int8),
           "w13_weight_scale": torch.ones(256, 256, 1),
           "w13_scale_bias": torch.zeros(256, 256, 1),
           "w2_weight": values, "w2_weight_scale": scale,
           "w2_scale_bias": bias}
    bank = torch.nn.Module()
    for name, tensor in raw.items():
        bank.register_parameter(name, torch.nn.Parameter(
            tensor.to(device), requires_grad=False))
    config, method = gmm1_inputs.quant_method()
    with set_current_vllm_config(config):
        method.process_weights_after_loading(bank)
        _, _, weight, packed_scale = method._get_mlp_weights(bank)
        _, assist = method._get_mlp_scale_bias(bank)
    return weight, packed_scale, assist, (values, scale, bias)


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
    if rows == 256:
        counts = torch.zeros(256, dtype=torch.int64)
        counts[:16] = 16
        yield "balanced16", counts.clone()
        counts[0], counts[1] = 17, 15
        yield "border17", counts.clone()
        counts.zero_()
        counts[-16:] = 16
        yield "balanced16_shifted", counts
        root = Path("/workspace/experiments/gmm2-compact-20261020")
        fixtures = json.loads((root / "guard-counts-screen/counts.json").read_text())
        chosen = {
            "active64-hot8-rows17", "active64-hot8-rows25",
            "active64-hot9-rows17", "active32-hot4-rows32",
            "active16-hot8-rows30",
        }
        for case in fixtures["cases"]:
            if case["name"] in chosen:
                yield case["name"], torch.tensor(case["counts"], dtype=torch.int64)
        fixtures = json.loads((root / "teacher-counts-screen/counts.json").read_text())
        for case in fixtures["cases"]:
            if case["name"] == "teacher-layer-60":
                yield case["name"], torch.tensor(case["counts"], dtype=torch.int64)


def reference(raw, counts, x, token_scale):
    values, scale, bias = raw
    output = []
    start = 0
    for expert in counts.nonzero().flatten().tolist():
        stop = start + int(counts[expert])
        integer = unpack_w4(values[expert]).to(torch.int32)
        output.append(w4_msd_product(
            x[start:stop].to(torch.int32), integer,
            scale[expert].flatten(), bias[expert].flatten(),
            token_scale[start:stop, None]))
        start = stop
    return torch.cat(output)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:15")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    capacity = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(1536 * 2**20 / capacity)
    torch.npu.reset_peak_memory_stats()
    weights, scales, bias, raw = build_bank(args.device)
    generator = torch.Generator().manual_seed(42)
    x_cpu = torch.randint(-127, 128, (256, 128), dtype=torch.int8,
                          generator=generator)
    s_cpu = torch.rand(256, generator=generator) * .016 + .004
    report = {"scope": "one-die NZ W4A8 down projection; not decoder TPOT",
              "opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
              "input_sha256": [digest(x) for x in (*raw, x_cpu, s_cpu)],
              "cases": []}
    started = time.monotonic()
    for rows in (8, 16, 17, 64, 256):
        frozen = x_cpu[:rows].to(args.device)
        x = frozen.clone()
        token_scale = s_cpu[:rows].to(args.device)
        groups = torch.zeros(256, dtype=torch.int64, device=args.device)
        groups[-1] = rows

        def call():
            x.copy_(frozen)
            return torch_npu.npu_grouped_matmul(
                x=[x], weight=weights, scale=scales, bias=bias,
                per_token_scale=[token_scale], group_list=groups,
                split_item=2, group_type=0, group_list_type=1,
                output_dtype=torch.bfloat16)[0]

        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = call()
        expected = {name: reference(raw, counts, x_cpu[:rows], s_cpu[:rows])
                    for name, counts in routes(rows)}
        for name, counts in list(routes(rows)) * 2:
            groups.copy_(counts)
            graph.replay()
            torch.npu.synchronize()
            actual = output.cpu()
            ref = expected[name]
            record = {"rows": rows, "route": name,
                      "active": int((counts > 0).sum()),
                      "exact": torch.equal(actual, ref),
                      "unequal": int((actual != ref).sum()),
                      "max_abs": float((actual.float() - ref.float()).abs().max()),
                      "finite": bool(torch.isfinite(actual).all()),
                      "output_sha256": digest(actual)}
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
            if not record["exact"]:
                torch.save({"actual": actual, "expected": ref, "counts": counts},
                           args.output.with_suffix(".failure.pt"))
                break
        if not report["cases"][-1]["exact"]:
            break
        del graph, output
    report.update(status="pass" if all(c["exact"] for c in report["cases"])
                  else "fail", elapsed_s=time.monotonic() - started,
                  peak_reserved_bytes=torch.npu.max_memory_reserved())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
    raise SystemExit(report["status"] != "pass")


if __name__ == "__main__":
    main()
