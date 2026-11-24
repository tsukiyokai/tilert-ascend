"""Is a kernel's floor 11.8 us or 28.8?

The round model prices the fixed term at 28.8 us a kernel, from a 12-element
scalar multiply. `bench_kernel_floor` just read 11.836 for a trivial kernel on
the same die. The two cannot both be the in-graph floor, and the project's own
lesson is that an eager event pair can read the host instead of the device.
So: the same 12-element multiply, timed three ways.
"""
import json
import statistics
import time

import torch
import torch_npu

DEVICE = 0
CALLS = 50
WARMUP = 50
BATCHES = 20


def batch_us(call, calls):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(calls):
        call()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / calls / 1e3


def eager_event_us(call):
    start, end = torch.npu.Event(True), torch.npu.Event(True)
    torch.npu.synchronize()
    start.record()
    call()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3


def main():
    torch_npu.npu.set_device(DEVICE)
    x = torch.ones(12, dtype=torch.bfloat16, device=f"npu:{DEVICE}")
    wide = torch.ones(32, 6144, dtype=torch.bfloat16, device=f"npu:{DEVICE}")

    cases = {
        "scalar_mul_12_elements": lambda: x.mul(1.0001),
        "scalar_mul_32x6144": lambda: wide.mul(1.0001),
    }
    report = {}
    for name, call in cases.items():
        for _ in range(WARMUP):
            call()
        batched = sorted(batch_us(call, CALLS) for _ in range(BATCHES))
        singles = sorted(eager_event_us(call) for _ in range(BATCHES))
        host = []
        for _ in range(BATCHES):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            call()
            host.append((time.perf_counter_ns() - start) / 1e3)
        report[name] = {
            "back_to_back_us": statistics.median(batched),
            "eager_event_us": statistics.median(singles),
            "host_enqueue_us": statistics.median(host),
        }
        row = report[name]
        print(f"{name:24} back-to-back {row['back_to_back_us']:7.3f}  "
              f"eager event {row['eager_event_us']:7.3f}  "
              f"host enqueue {row['host_enqueue_us']:7.3f}")
    json.dump(report, open("/out/floor/scalar.json", "w"), indent=2)


main()
