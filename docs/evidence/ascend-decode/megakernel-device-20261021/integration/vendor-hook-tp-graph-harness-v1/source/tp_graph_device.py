"""Run two device participants and fourteen CPU constructor-geometry peers."""

import argparse
import copy
import ctypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def device_mapping(torch, args):
    library = ctypes.CDLL("libascendcl.so")
    convert = library.aclrtGetPhyDevIdByLogicDevId
    convert.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    convert.restype = ctypes.c_int
    access = library.aclrtDeviceCanAccessPeer
    access.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32]
    access.restype = ctypes.c_int
    physical = []
    for index in (0, 1):
        value = ctypes.c_int32(-1)
        if convert(index, ctypes.byref(value)) != 0:
            raise RuntimeError("CANN could not resolve the physical device mapping")
        physical.append(value.value)
    if physical != args.physical_devices or torch.npu.current_device() != args.device_rank:
        raise RuntimeError(f"Unexpected logical-to-physical mapping: {physical}")
    peer = ctypes.c_int32()
    if access(ctypes.byref(peer), args.device_rank, 1 - args.device_rank) != 0 or peer.value != 1:
        raise RuntimeError("The two exposed devices do not report direct peer access")
    return None, {"logical_to_physical": physical, "logical_rank": args.device_rank,
                  "can_access_peer": peer.value,
                  "acl_header_sha256": "5ffca8a8cd22258c587a0a1c6c66e8a8605deac00005d9de896be0ab1f7c3769"}


def worker(rank, args):
    log = (args.output / f"rank-{rank:02d}.log").open("w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    report = {"rank": rank, "device_participant": rank < 2, "stages": []}
    state, groups = None, ()

    def stage(name, action):
        item = {"name": name}
        try:
            result, details = action()
            item.update(success=True, **details)
        except Exception:
            item.update(success=False, error=traceback.format_exc())
            report["stages"].append(item)
            print(json.dumps(item), flush=True)
            raise
        report["stages"].append(item)
        print(json.dumps(item), flush=True)
        return result

    try:
        import torch
        import torch_npu
        from vendor_hook_runtime212_device import GROUP_FIELDS, identity
        from tp_groups import initialize_constructor_groups, create_communication_groups

        torch.set_num_threads(1)
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__,
                      python=sys.version, runtime_image=os.environ.get("VENDOR_HOOK_RUNTIME_IMAGE"),
                      npu_initialized_before=torch.npu.is_initialized())
        if rank >= 2:
            def refuse_init(*unused_args, **unused_kwargs):
                raise RuntimeError("A CPU participant may not initialize the NPU")
            torch.npu._lazy_init = refuse_init
            torch_npu._C._npu_init = refuse_init
        from vllm.platforms import current_platform

        stage("platform_registration", lambda: (
            current_platform.pre_register_and_update(), identity(type(current_platform))))
        state, groups, details = initialize_constructor_groups(torch, rank, args)
        report["stages"].append({"name": "actual_tp16_gloo_geometry", "success": True, **details})
        success = True
        if rank < 2:
            from tp_graph_constructor import run_device_rank

            current = copy.copy(args)
            current.device_rank = rank
            current.output = args.output / f"device-rank{rank}"
            current.output.mkdir()
            try:
                report["result"] = run_device_rank(torch, current, state, stage, [False])
            except Exception:
                report["error"] = traceback.format_exc()
                success = False
        else:
            stage("device_group_nonmember", lambda: (
                create_communication_groups(torch, state, rank),
                {"mc2_singletons": [[0], [1]], "hccl_oracle": [0, 1],
                 "device_group_member": False}))
        statuses = [None] * 16
        torch.distributed.all_gather_object(statuses, {"rank": rank, "success": success},
                                             group=state.get_world_group().cpu_group)
        report["all_rank_statuses"] = statuses
        report["success"] = all(item["success"] for item in statuses)
        torch.distributed.barrier(group=state.get_world_group().cpu_group)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if state is not None:
            for field in GROUP_FIELDS:
                setattr(state, field, None)
            for group in groups:
                group.destroy()
            state.destroy_distributed_environment()
        if "torch" in locals():
            report["npu_initialized_after"] = torch.npu.is_initialized()
        report["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / f"rank-{rank:02d}.json").write_text(json.dumps(report, indent=2) + "\n")
    if "error" in report or not report.get("success", False):
        raise SystemExit(1)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "runtime", "adapter", "indexed-library", "shared-library", "sfa-library"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--physical-devices", type=int, nargs=2, required=True)
    parser.add_argument("--shmem-endpoint", default="tcp://127.0.0.1:24571")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--mode", choices=("int8",), default="int8")
    parser.add_argument("--forward-widths", type=int, nargs="+", default=[12])
    parser.add_argument("--forward-replays", type=int, default=2)
    parser.add_argument("--forward-sequence", type=int, default=8192)
    parser.add_argument("--graph-residual-modes", nargs="+", choices=("provided",), default=["provided"])
    for name, filename in (("ffn-library", "complete_ffn.so"),
                           ("chain-library", "layer_local_chain.so"),
                           ("tp-library", "layer_tp_chain.so")):
        parser.add_argument("--" + name, type=Path, default=Path(__file__).with_name(filename))
    args = parser.parse_args()
    if (args.physical_devices[0] >= args.physical_devices[1]
            or args.physical_devices[0] < 0 or args.timeout < 1
            or len(set(args.forward_widths)) != len(args.forward_widths)
            or any(rows not in (1, 2, 12, 32) for rows in args.forward_widths)
            or not 1 <= args.forward_replays <= 32
            or args.forward_sequence != 8192
            or args.graph_residual_modes != ["provided"]):
        parser.error("Use ordered distinct devices, unique M1/2/12/32, 1..32 inputs and sequence8192")
    if not args.shmem_endpoint.startswith("tcp://127.0.0.1:"):
        parser.error("Use a loopback SHMEM endpoint in the isolated container")
    for path in (args.adapter, args.indexed_library, args.shared_library, args.sfa_library,
                 args.ffn_library, args.chain_library, args.tp_library):
        if not path.is_file():
            parser.error(f"Missing frozen input: {path}")
    if not (args.runtime / "tests/glm52/whole_layer/vendor_hook_runtime212_device.py").is_file():
        parser.error("Use the frozen runtime212 directory")
    args.world_size = 16
    return args


def main():
    args = arguments()
    hashes = json.loads(Path(__file__).with_name("shmem-libraries.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((Path("/shmem") / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"The recorded SHMEM library changed: {name}")
    sys.path.insert(0, str(args.runtime / "tests/glm52/whole_layer"))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, args)) for rank in range(16)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + args.timeout
    while any(process.is_alive() for process in processes):
        if (time.monotonic() >= deadline
                or any(process.exitcode not in (None, 0) for process in processes)):
            for process in processes:
                if process.is_alive():
                    process.terminate()
            break
        for process in processes:
            process.join(timeout=0.1)
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    report = {"scope": "Two controlled TP16-shaped shards; real HCCL2 versus SHMEM2 layer launches",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "exitcodes": [process.exitcode for process in processes],
        "success": all(process.exitcode == 0 for process in processes)}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return int(not report["success"])


if __name__ == "__main__":
    raise SystemExit(main())
