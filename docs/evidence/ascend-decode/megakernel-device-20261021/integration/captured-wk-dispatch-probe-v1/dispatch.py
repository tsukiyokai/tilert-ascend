#!/usr/bin/env python3
"""Record one captured WK call with installed CANN debug logging enabled."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import check as contract


KINDS = ("full-bf16", "full-fp32", "split-bf16")


def select_case(captured, kind):
    columns = 32 if kind == "split-bf16" else 160
    if kind == "full-fp32":
        expected = captured["native"]["key_accumulators"]
    elif kind == "split-bf16":
        expected = captured["native"]["key_projection"][:, :columns].contiguous()
    else:
        expected = captured["vendor"]["key_projection"]
    return columns, expected


def loaded_libraries():
    names = ("libopapi", "libnnopbase", "liboptiling", "libopmaster",
             "libtorch_npu", "libopskernel", "libascendcl")
    paths = {line.split()[-1]
             for line in Path("/proc/self/maps").read_text().splitlines()
             if "/" in line and any(name in line.split()[-1] for name in names)}
    return [contract.file_identity(path) for path in sorted(paths)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "One original captured WK operator call; dispatch evidence only, no timing or full-layer acceptance",
        "kind": args.kind, "pid": os.getpid(),
        "script": contract.file_identity(__file__),
        "contract_source": contract.file_identity(contract.__file__),
        "capture": contract.file_identity(args.capture),
        "python": sys.version, "platform": platform.platform(),
        "environment": {name: os.environ.get(name) for name in (
            "ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_SLOG_PRINT_TO_STDOUT",
            "ASCEND_PROCESS_LOG_PATH", "ASCEND_DOCKER_RUNTIME")},
        "completed": False, "cpu_only": args.cpu_only,
        "call_count": 0,
        "interpretation": "Output dtype can change the selected kernel or tiling; these observations do not alone isolate Fixpipe.",
    }

    def persist():
        (args.output / "wk-dispatch.json").write_text(
            json.dumps(report, indent=2) + "\n")

    try:
        import torch

        torch.set_num_threads(1)
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        _, witnesses = contract.cpu_contract(torch, captured)
        columns, expected = select_case(captured, args.kind)
        hidden_cpu = captured["vendor"]["key_hidden"]
        weight_cpu = captured["vendor"]["key_weight"][:columns].contiguous()
        report.update(
            torch=torch.__version__, witnesses=witnesses,
            columns=[0, columns],
            input=contract.tensor_identity(torch, hidden_cpu),
            weight=contract.tensor_identity(torch, weight_cpu),
            expected=contract.tensor_identity(torch, expected),
            operator=("torch.mm(hidden, weight.t(), out_dtype=torch.float32)"
                      if args.kind == "full-fp32" else
                      "torch.nn.functional.linear(hidden, weight, bias=None)"))
        persist()
        if args.cpu_only:
            report["completed"] = True
            return
        if os.environ.get("ASCEND_GLOBAL_LOG_LEVEL") != "0":
            raise RuntimeError("This dispatch probe requires command-local DEBUG logging")

        import torch_npu

        report["npu_initialized_before_probe"] = torch.npu.is_initialized()
        torch.npu.set_device(0)
        if torch.npu.device_count() != 1:
            raise RuntimeError("Expose exactly one NPU")
        torch.npu.config.allow_internal_format = True
        hidden = torch_npu.npu_format_cast(hidden_cpu.to("npu"), 2)
        weight = torch_npu.npu_format_cast(weight_cpu.to("npu"), 29)
        if torch_npu.get_npu_format(weight) != 29:
            raise RuntimeError("The original physical NZ weight format was not retained")
        if not contract.compare(torch, weight.cpu(), weight_cpu)["bitwise_equal"]:
            raise RuntimeError("Packing changed WK operand bytes")
        if not contract.compare(torch, hidden.cpu(), hidden_cpu)["bitwise_equal"]:
            raise RuntimeError("Device copy changed hidden bytes")
        operand = weight.t()
        torch.npu.synchronize()
        report.update(
            torch_npu=torch_npu.__version__,
            torch_npu_extension=contract.file_identity(torch_npu._C.__file__),
            device=torch.npu.get_device_name(0),
            device_properties=str(torch.npu.get_device_properties(0)),
            schemas={"linear": str(torch.ops.aten.linear.default._schema),
                     "mm": str(torch.ops.aten.mm.default._schema),
                     "mm_dtype": str(torch.ops.aten.mm.dtype._schema)},
            device_inputs={"hidden": contract.device_descriptor(torch_npu, hidden),
                           "weight": contract.device_descriptor(torch_npu, weight),
                           "mm_right_operand": contract.device_descriptor(torch_npu, operand)},
            call_start_unix_ns=time.time_ns())
        persist()
        print(json.dumps({"marker": "WK_CALL_BEGIN", "kind": args.kind,
                          "unix_ns": report["call_start_unix_ns"]}), flush=True)
        with torch.inference_mode():
            if args.kind == "full-fp32":
                result = torch.mm(hidden, operand, out_dtype=torch.float32)
            else:
                result = torch.nn.functional.linear(hidden, weight, bias=None)
        report["call_count"] = 1
        torch.npu.synchronize()
        report["call_end_unix_ns"] = time.time_ns()
        print(json.dumps({"marker": "WK_CALL_END", "kind": args.kind,
                          "unix_ns": report["call_end_unix_ns"]}), flush=True)
        actual = result.cpu().contiguous()
        report["output"] = contract.tensor_identity(torch, actual)
        report["reproduction"] = contract.compare(torch, actual, expected)
        report["output_descriptor"] = contract.device_descriptor(torch_npu, result)
        report["loaded_libraries"] = loaded_libraries()
        torch.save(actual, args.output / "wk-dispatch-output.pt")
        if not report["reproduction"]["bitwise_equal"]:
            raise RuntimeError("The instrumented call did not reproduce its captured control")
        report["completed"] = True
        print(json.dumps({"completed": True, "kind": args.kind,
                          "output": report["output"]}), flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
