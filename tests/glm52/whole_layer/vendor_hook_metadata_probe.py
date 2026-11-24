#!/usr/bin/env python3
"""Measure RoPE storage conversion with the runtime's original Torch build."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "RoPE format conversion and captured producer only",
              "script": file_identity(__file__), "cases": [], "pass": False}

    def persist():
        (args.output / "metadata-format.json").write_text(
            json.dumps(report, indent=2) + "\n")

    try:
        import torch
        import torch_npu

        torch.npu.set_device(0)
        if torch.npu.device_count() != 1:
            raise RuntimeError("Expose exactly one NPU")
        torch.npu.config.allow_internal_format = True
        sys.path.insert(0, str(args.adapter.resolve().parents[3]))
        adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
        if Path(adapter.__file__).resolve() != args.adapter.resolve():
            raise RuntimeError("The imported adapter differs from the requested source")
        report.update(python=sys.version, torch=torch.__version__,
                      torch_npu=torch_npu.__version__, device=torch.npu.get_device_name(0),
                      adapter=file_identity(args.adapter),
                      torch_npu_extension=file_identity(torch_npu._C.__file__))

        def raw(value):
            return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

        def describe(value):
            return {"shape": list(value.shape), "stride": list(value.stride()),
                    "format": torch_npu.get_npu_format(value),
                    "data_ptr": value.data_ptr(),
                    "storage_ptr": value.untyped_storage().data_ptr(),
                    "storage_nbytes": value.untyped_storage().nbytes(),
                    "logical_bytes_sha256": hashlib.sha256(raw(value)).hexdigest()}

        def values(rows, iteration):
            value = (torch.arange(rows * 64, dtype=torch.float32) % 139 - 69)
            value = (value / (31 + iteration) + iteration / 8).bfloat16()
            value[0] = -0.0
            return value.reshape(rows, 1, 1, 64)

        for rows in (1, 2, 12, 32):
            expected = values(rows, 0)
            source = torch_npu.npu_format_cast(expected.to("npu"), 2)
            cloned = source.clone()
            view = cloned.view(rows, 64)
            # These allocation/copy operations reproduce the official indexer
            # metadata builder's four-dimensional persistent RoPE buffers.
            builder_buffer = torch.empty(32, 1, 1, 64, dtype=torch.bfloat16, device="npu")
            builder = builder_buffer[:rows]
            builder.copy_(source)
            builder_view = builder.view(rows, 64)
            case = {"rows": rows, "source": describe(source),
                    "clone": describe(cloned), "clone_view": describe(view),
                    "builder_view": describe(builder_view), "replays": []}
            normalized = adapter._nd(view)
            normalized_builder = adapter._nd(builder_view)
            case.update(normalized=describe(normalized),
                        normalized_builder=describe(normalized_builder),
                        clone_after=describe(cloned),
                        builder_after=describe(builder),
                        byte_equal=all(raw(value) == raw(expected) for value in (
                            cloned, view, builder_view, normalized, normalized_builder)))
            output = torch.empty(rows, 64, dtype=torch.bfloat16, device="npu")

            def produce():
                converted = adapter._nd(builder_view)
                output.copy_(converted)

            for _ in range(2):
                produce()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                produce()
            for iteration in range(4):
                current = values(rows, iteration + 1)
                builder.copy_(current.to("npu"))
                graph.replay()
                torch.npu.synchronize()
                equal = raw(output) == raw(current)
                case["replays"].append({"iteration": iteration, "byte_equal": equal,
                                        "output": describe(output)})
            case["pass"] = (case["byte_equal"]
                            and case["normalized"]["format"] == 2
                            and case["normalized_builder"]["format"] == 2
                            and all(item["byte_equal"] for item in case["replays"]))
            report["cases"].append(case)
            persist()
            print(json.dumps(case), flush=True)
            if not case["pass"]:
                raise RuntimeError("RoPE normalization changed bytes or lost a captured update")
        report["pass"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
