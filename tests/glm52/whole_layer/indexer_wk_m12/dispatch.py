#!/usr/bin/env python3
"""Record the exact captured M12 F.linear call and installed CANN dispatch."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import contract


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
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Captured full Decoder M12 WK only; one original F.linear call, no timing",
        "script": contract.file_identity(__file__),
        "contract_source": contract.file_identity(contract.__file__),
        "capture": contract.file_identity(args.capture),
        "python": sys.version, "platform": platform.platform(),
        "environment": {name: os.environ.get(name) for name in (
            "ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_SLOG_PRINT_TO_STDOUT",
            "ASCEND_PROCESS_LOG_PATH", "ASCEND_DOCKER_RUNTIME")},
        "completed": False, "cpu_only": args.cpu_only, "call_count": 0,
        "operator": "torch.nn.functional.linear(hidden, NZ_weight, bias=None)",
        "cpu_oracle_scope": "Auxiliary FP64 dot; numerical acceptance targets captured vendor BF16 bytes",
    }

    def persist():
        (args.output / "wk-dispatch.json").write_text(
            json.dumps(report, indent=2) + "\n")

    try:
        if report["capture"]["sha256"] != contract.CAPTURE_SHA256:
            raise ValueError("The captured M12 input identity changed")
        import torch

        torch.set_num_threads(1)
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        hidden_cpu = captured["vendor"]["key_hidden"]
        weight_cpu = captured["vendor"]["key_weight"]
        expected = captured["vendor"]["key_projection"]
        report.update(
            torch=torch.__version__, witnesses=contract.cpu_contract(torch, captured),
            input=contract.tensor_identity(torch, hidden_cpu),
            weight=contract.tensor_identity(torch, weight_cpu),
            expected=contract.tensor_identity(torch, expected))
        persist()
        if args.cpu_only:
            report["completed"] = True
            return
        if os.environ.get("ASCEND_GLOBAL_LOG_LEVEL") != "0":
            raise RuntimeError("This probe requires command-local CANN DEBUG logging")
        import torch_npu

        report["npu_initialized_before_probe"] = torch.npu.is_initialized()
        torch.npu.set_device(0)
        if torch.npu.device_count() != 1:
            raise RuntimeError("Expose exactly one NPU")
        torch.npu.config.allow_internal_format = True
        hidden = torch_npu.npu_format_cast(hidden_cpu.to("npu"), 2)
        weight = torch_npu.npu_format_cast(weight_cpu.to("npu"), 29)
        if torch_npu.get_npu_format(weight) != 29:
            raise RuntimeError("The original NZ weight format was not retained")
        for actual, target in ((hidden, hidden_cpu), (weight, weight_cpu)):
            if not contract.compare(torch, actual.cpu(), target)["bitwise_equal"]:
                raise RuntimeError("Device packing changed WK operand bytes")
        operand = weight.t()
        torch.npu.synchronize()
        report.update(
            torch_npu=torch_npu.__version__,
            torch_npu_extension=contract.file_identity(torch_npu._C.__file__),
            device=torch.npu.get_device_name(0),
            device_properties=str(torch.npu.get_device_properties(0)),
            schemas={"linear": str(torch.ops.aten.linear.default._schema)},
            device_inputs={"hidden": contract.device_descriptor(torch_npu, hidden),
                           "weight": contract.device_descriptor(torch_npu, weight),
                           "right_operand": contract.device_descriptor(torch_npu, operand)},
            call_start_unix_ns=time.time_ns())
        persist()
        print(json.dumps({"marker": "WK_M12_CALL_BEGIN",
                          "unix_ns": report["call_start_unix_ns"]}), flush=True)
        with torch.inference_mode():
            result = torch.nn.functional.linear(hidden, weight, bias=None)
        report["call_count"] = 1
        torch.npu.synchronize()
        report["call_end_unix_ns"] = time.time_ns()
        print(json.dumps({"marker": "WK_M12_CALL_END",
                          "unix_ns": report["call_end_unix_ns"]}), flush=True)
        actual = result.cpu().contiguous()
        report["output"] = contract.tensor_identity(torch, actual)
        report["reproduction"] = contract.compare(torch, actual, expected)
        report["captured_native_projection"] = contract.compare(
            torch, actual, captured["native"]["key_projection"])
        report["output_descriptor"] = contract.device_descriptor(torch_npu, result)
        report["loaded_libraries"] = loaded_libraries()
        torch.save(actual, args.output / "wk-dispatch-output.pt")
        if not report["reproduction"]["bitwise_equal"]:
            raise RuntimeError("The original operator did not reproduce captured vendor bytes")
        report["completed"] = True
        print(json.dumps({"completed": True, "output": report["output"]}), flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
