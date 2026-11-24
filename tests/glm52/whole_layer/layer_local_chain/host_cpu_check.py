"""Check ctypes record marshalling without submitting a device launch."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from host import Library, Prepared


class AddressOwner:
    """A CPU-only address token; it never denotes allocated device storage."""

    def __init__(self, address, device, rows):
        self.address = address
        self.device = device
        self.shape = (rows, 6144)

    def data_ptr(self):
        return self.address


def must_reject(call, message):
    try:
        call()
    except (ValueError, RuntimeError) as error:
        if message not in str(error):
            raise
        return str(error)
    raise AssertionError("The invalid host record was accepted")


def check(library, rows, indexed):
    device = SimpleNamespace(type="npu", index=0)
    groups = []
    address = 0x10000
    for size in (8, 41, 3, 6, 12, 21, 4):
        group = []
        for _ in range(size):
            group.append(AddressOwner(address, device, rows))
            address += 0x10000
        groups.append(group)
    norm, attention, post, routed, bridge, body, shared = groups
    attention[:3] = norm[3], norm[6], norm[7]
    topk = AddressOwner(address, device, rows)
    geometry = [rows, 66, 64, 66, 64, 24, 1, 0, 0]
    scalars = [0.0441941738, 1e-5, 1e-5, 1e-5, 1e-5]

    def prepare():
        return Prepared(library, norm, attention, post, routed, bridge, body,
                        shared, topk, geometry, scalars, 0, indexed)

    prepared = prepare()
    rejected = [must_reject(prepared.launch, "Invalid local layer submission")]
    norm[3].address += 32
    rejected.append(must_reject(prepared.launch, "changed its data address"))
    norm[3].address -= 32
    geometry[1] = 0
    rejected.append(must_reject(prepare, "Invalid local layer metadata"))
    geometry[1] = 66
    attention[0] = AddressOwner(address + 0x10000, device, rows)
    rejected.append(must_reject(prepare, "Invalid local layer metadata"))
    return {"rows": rows, "indexed": indexed, "record_preparation": True,
            "negative_controls": rejected}


def main():
    import torch
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.npu.is_initialized() or list(Path("/dev").glob("davinci[0-9]*")):
        raise RuntimeError("This gate requires no initialized or mapped NPU")
    report = {"scope": "Python ctypes to actual ARM C++ records using address tokens; no device storage or tensor validation",
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "npu_initialized_before": False, "mapped_davinci_devices": [],
              "device_execution": False, "library_sha256": args.sha256,
              "cases": [], "pass": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    library = Library(args.library, args.sha256)
    report["cases"] = [check(library, rows, indexed)
                       for rows in (1, 2, 12, 32) for indexed in (False, True)]
    report["npu_initialized_after"] = torch.npu.is_initialized()
    if report["npu_initialized_after"]:
        raise RuntimeError("The rejected submission initialized the NPU")
    report["pass"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
