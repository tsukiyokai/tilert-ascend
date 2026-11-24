"""Exercise the real Torch212 HCCL nonmember entry inside a CPU Gloo world."""

from datetime import timedelta
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import tempfile
import traceback


def worker(rank, directory):
    import torch
    import torch_npu

    root = Path(directory)
    report = {"rank": rank, "npu_initialized_before": torch.npu.is_initialized()}
    initialized = False
    try:
        torch.set_num_threads(1)
        def refuse(*args, **kwargs):
            raise RuntimeError("The nonmember CPU probe may not initialize an NPU")
        torch.npu._lazy_init = refuse
        torch_npu._C._npu_init = refuse
        torch.distributed.init_process_group(
            "gloo", rank=rank, world_size=3,
            init_method=f"file://{root / 'rendezvous'}", timeout=timedelta(seconds=60))
        initialized = True
        torch.distributed.barrier()
        if rank == 2:
            result = torch.distributed.new_group([0, 1], backend="hccl", pg_options=None)
            if result != torch.distributed.GroupMember.NON_GROUP_MEMBER:
                raise RuntimeError("The real Torch nonmember path created a device group")
            report.update(nonmember_result=result, pg_options=None, actual_entry="torch.distributed.new_group")
        torch.distributed.barrier()
        report["pass"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if initialized:
            torch.distributed.destroy_process_group()
        report["npu_initialized_after"] = torch.npu.is_initialized()
        (root / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report.get("pass") or report["npu_initialized_after"]:
        raise SystemExit(1)


def run():
    import torch
    from torch.distributed import distributed_c10d

    with tempfile.TemporaryDirectory(prefix="tp-nonmember-") as directory:
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=worker, args=(rank, directory)) for rank in range(3)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
        reports = [json.loads((Path(directory) / f"rank-{rank}.json").read_text())
                   for rank in range(3)]
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"The actual CPU nonmember probe failed: {reports}")
    functions = (distributed_c10d.new_group, distributed_c10d._new_group_with_tag,
                 distributed_c10d._new_process_group_helper)
    source_path = Path(inspect.getfile(distributed_c10d))
    return {"scope": "Real three-rank CPU Gloo world; only global rank2 enters the HCCL nonmember path; no HCCL group created",
            "exitcodes": [process.exitcode for process in processes], "ranks": reports,
            "torch": torch.__version__, "source_path": str(source_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "functions": {function.__name__: {
                "line": inspect.getsourcelines(function)[1], "source": inspect.getsource(function)}
                for function in functions}}
