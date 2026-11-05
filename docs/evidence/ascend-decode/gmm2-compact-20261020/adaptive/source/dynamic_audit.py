"""Check adaptive metadata transitions; this does not emulate NPU caches."""

import hashlib
import json
from pathlib import Path


ROOT = Path("/workspace/experiments/gmm2-compact-20261020")
SOURCE = ROOT / "retile-adaptive-build/source/ops_transformer/ascendc/grouped_matmul/op_kernel"


def publish(counts):
    active = end = 0
    wide = True
    values = [0] * 260
    for expert, rows in enumerate(counts):
        assert rows >= 0
        if rows > 0:
            wide = wide and rows <= 16
            end += rows
            active += 1
            values[active] = (end << 32) | expert
    values[0] = (int(wide) << 32) | active
    length = ((active + 4) // 4) * 4
    return values[:length]


def main():
    a = [16] * 16 + [0] * 240
    b = [17, 15] + [16] * 14 + [0] * 240
    c = [0] * 240 + [16] * 16
    d = [0] * 240 + [17, 15] + [16] * 14
    cases = {"A": a, "B": b, "C": c, "D": d,
             "shared8": [32] * 8 + [0] * 248,
             "active256": [1] * 256,
             "one_token": [1] * 8 + [0] * 248}
    order = ["A", "B", "A", "C", "D", "C", "shared8", "active256", "A", "one_token", "B", "A"]
    gm = [0xDEADBEEF] * 260
    transitions = []
    for name in order:
        counts = cases[name]
        written = publish(counts)
        gm[:len(written)] = written
        # Model the contract of publication followed by all-core invalidation.
        for consumer in range(72):
            header = gm[0]
            active = header & 0xFFFFFFFF
            assert active == sum(value > 0 for value in counts)
            assert bool(header >> 32) == (max(counts) <= 16)
            previous = 0
            recovered = [0] * 256
            for record in gm[1:active + 1]:
                expert, end = record & 0xFFFFFFFF, record >> 32
                assert end > previous
                recovered[expert] = end - previous
                previous = end
            assert recovered == counts
            for pointer_bit in (0, 32):
                # Each invocation loads original TCube data before Retile.
                cfg = (64, 512)
                if pointer_bit == 0 and header >> 32:
                    cfg = (32, 1024)
                assert cfg == ((32, 1024) if pointer_bit == 0 and max(counts) <= 16 else (64, 512))
        transitions.append({"state": name, "rows": sum(counts),
                            "active": active, "header": header,
                            "retile": bool(header >> 32)})
    paths = [SOURCE / name for name in (
        "grouped_matmul.cpp", "grouped_matmul_active_groups_a8w4.h",
        "grouped_matmul_antiquant_a8w4_msd.h")]
    result = {"scope": "CPU metadata and fresh-TCube state contract; no hardware synchronization or cache simulation",
              "npu_launched": False, "consumers_per_transition": 72,
              "transition_count": len(order), "transitions": transitions,
              "same_active_count_threshold_crossing": True,
              "same_header_different_experts": publish(a)[0] == publish(c)[0] and publish(a)[1:] != publish(c)[1:],
              "stale_header_counterexample": bool(publish(a)[0] >> 32) != bool(publish(b)[0] >> 32),
              "stale_local_tcube_counterexample": "If cfg survives A->B, B skips Retile and incorrectly retains 32x1024",
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths + [Path(__file__)]},
              "input_sha256": hashlib.sha256(json.dumps(cases).encode()).hexdigest()}
    Path(__file__).with_name("result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
