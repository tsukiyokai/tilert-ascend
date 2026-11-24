"""Calibrate the event-loop controls used by paired component graph timings."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics

import torch
import torch_npu


def elapsed(graph, stream, iterations):
    with torch.npu.stream(stream):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device("npu:0")
    stream = torch.npu.Stream(device="npu:0")
    source = torch.ones(8, device="npu:0")
    destination = torch.empty_like(source)
    stream.wait_stream(torch.npu.current_stream())
    report = {"utc": datetime.now(timezone.utc).isoformat(),
              "host": platform.node(), "torch": torch.__version__,
              "torch_npu": torch_npu.__version__, "unit": "us_per_replay",
              "scope": "raw empty and 32-byte-copy graph controls; no subtraction",
              "cases": []}
    for name in ("empty", "copy_32_bytes"):
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph, stream=stream):
                if name == "copy_32_bytes":
                    destination.copy_(source)
            with torch.npu.stream(stream):
                for _ in range(5):
                    graph.replay()
            stream.synchronize()
            for iterations in (1, 10, 30, 100):
                samples = [elapsed(graph, stream, iterations) for _ in range(7)]
                report["cases"].append({"graph": name, "iterations": iterations,
                                        "samples": samples,
                                        "median": statistics.median(samples)})
        except Exception as error:
            report["cases"].append({"graph": name, "error": str(error)})
    report["copy_correct"] = torch.equal(destination.cpu(), source.cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
