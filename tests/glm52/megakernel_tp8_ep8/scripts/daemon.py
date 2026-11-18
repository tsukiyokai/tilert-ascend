import os, sys, time, traceback

sys.path.append("/opt/ascend/cann-9.2.0/python/site-packages")
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

os.environ["VLLM_PLUGINS"] = "ascend,ascend_model"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
os.environ["HCCL_DETERMINISTIC"] = "true"

import torch
import torch_npu

torch.npu.set_device(rank)
torch.npu.config.allow_internal_format = True

job_dir = os.environ.get("TILERT_JOB_DIR", "/dev/shm")
cmd_file = f"{job_dir}/worker_cmd.txt"
done_file = f"{job_dir}/worker_done.txt"
ack_dir = f"{job_dir}/worker_ack"
os.makedirs(ack_dir, exist_ok=True)

if rank == 0:
    print(f"[Daemon] 8-rank torch_npu ready on NPU 0..7", flush=True)
    with open(f"{job_dir}/daemon_ready.txt", "w") as f:
        f.write("ready\n")

epoch = 0
while True:
    target = f"{epoch}:"
    script_path = None
    while True:
        if os.path.exists(cmd_file):
            try:
                with open(cmd_file, "r") as f:
                    content = f.read().strip()
                if content.startswith(target):
                    script_path = content[len(target):]
                    break
            except Exception:
                pass
        time.sleep(0.01)

    if script_path == "EXIT":
        break

    t0 = time.perf_counter()
    err = None
    try:
        with open(script_path, "r") as sf:
            code = sf.read()
        exec(
            compile(code, script_path, "exec"),
            {"__name__": "__main__", "rank": rank, "world_size": world_size},
        )
    except Exception:
        err = traceback.format_exc()
        print(f"[Rank {rank} Error]\n{err}", flush=True)

    with open(f"{ack_dir}/rank_{rank}_{epoch}", "w") as af:
        af.write("ERR" if err else "OK")

    if rank == 0:
        for r in range(world_size):
            while not os.path.exists(f"{ack_dir}/rank_{r}_{epoch}"):
                time.sleep(0.005)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[Daemon] Finished {script_path} in {dt:.1f} ms", flush=True)
        for r in range(world_size):
            try:
                os.remove(f"{ack_dir}/rank_{r}_{epoch}")
            except OSError:
                pass
        with open(done_file, "w") as df:
            df.write(f"{dt:.1f}\n")
    else:
        while os.path.exists(f"{ack_dir}/rank_{rank}_{epoch}"):
            time.sleep(0.005)

    epoch += 1
