"""Inventory which fused matmul-collective operators the A3 package serves."""

import os

import torch
import torch.distributed as distributed
import torch_npu

ROWS = 4
INPUT_PER_RANK = 2048
OUTPUT = 6144


def report(label, call, rank):
    try:
        result = call()
        torch.npu.synchronize()
        shapes = (tuple(result.shape) if torch.is_tensor(result)
                  else [tuple(item.shape) for item in result])
        if rank == 0:
            print(f"  {label:38s} OK    {shapes}")
    except Exception as error:
        head = str(error).split("\n")[0]
        code = ""
        for token in head.replace(",", " ").split():
            if token.isdigit() and len(token) == 6:
                code = f"[{token}] "
        if rank == 0:
            print(f"  {label:38s} 拒绝  {code}{head[-90:]}")


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl", rank=rank, world_size=world)
    group = distributed.distributed_c10d._get_default_group()
    hcom = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    torch.manual_seed(7)
    rows = ROWS * world
    activation = torch.randn(rows, INPUT_PER_RANK, dtype=torch.bfloat16,
                             device=device)
    weight = torch.randn(INPUT_PER_RANK, OUTPUT, dtype=torch.bfloat16,
                         device=device)
    weight_int8 = torch.randint(-8, 8, (INPUT_PER_RANK, OUTPUT),
                                dtype=torch.int8, device=device)
    scale = torch.rand(OUTPUT, dtype=torch.float32, device=device) * 0.01
    shard = activation[:ROWS].contiguous()
    quantized, pertoken = torch_npu.npu_dynamic_quant(activation)

    if rank == 0:
        print(f"world {world}  x1 {tuple(activation.shape)}  "
              f"x2 {tuple(weight.shape)}")
    report("npu_mm_reduce_scatter_base bf16",
           lambda: torch.ops.npu.npu_mm_reduce_scatter_base(
               activation, weight, hcom, world), rank)
    report("npu_quant_mm_reduce_scatter",
           lambda: torch.ops.npu.npu_quant_mm_reduce_scatter(
               quantized, weight_int8, hcom, world, dequant_scale=scale,
               pertoken_scale=pertoken), rank)
    report("npu_all_gather_base_mm bf16",
           lambda: torch.ops.npu.npu_all_gather_base_mm(
               shard, weight, hcom, world), rank)
    report("npu_all_gather_quant_mm",
           lambda: torch.ops.npu.npu_all_gather_quant_mm(
               quantized[:ROWS].contiguous(), weight_int8, scale, hcom,
               world), rank)
    distributed.destroy_process_group()


main()
