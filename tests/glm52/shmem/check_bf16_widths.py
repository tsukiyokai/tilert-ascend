import json
import os
from pathlib import Path

import torch
import torch_npu

from tilert.models.glm_5_2_ascend.onesided_reduce import OneSidedReduce


rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
assert world == 8
device = torch.device(f"npu:{rank}")
reducer = OneSidedReduce(rank, device, world, "tcp://127.0.0.1:29797")
results = []
try:
    for rows in (1, 4, 8, 32):
        values = torch.arange(rows * 6144, device=device, dtype=torch.float32) % 16
        partial = (values + rank).to(torch.bfloat16).view(rows, 6144)
        output = torch.full_like(partial, float("nan"))
        torch.ops.tilert.onesided_all_reduce_out(partial, output)
        torch.npu.synchronize()
        expected = (values * 8 + 28).to(torch.bfloat16).view_as(output)
        result = {
            "rows": rows, "elements": output.numel(),
            "finite": int(torch.isfinite(output).sum().item()),
            "equal": int((output == expected).sum().item()),
        }
        results.append(result)
    path = Path(os.environ["RESULT_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    (path / f"rank-{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"BF16 rank {rank}: {results}", flush=True)
finally:
    reducer.close()

assert all(result["equal"] == result["elements"] for result in results), results
