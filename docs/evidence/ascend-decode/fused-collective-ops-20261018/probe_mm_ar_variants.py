"""Find which npu_mm_all_reduce_base variants the A3 operator package serves."""

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
        if rank == 0:
            print(f"  {label:34s} OK   {tuple(result.shape)} {result.dtype}")
        return True
    except Exception as error:
        message = str(error).split("\n")[0]
        if rank == 0:
            print(f"  {label:34s} 拒绝 {message[:120]}")
        return False


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl", rank=rank, world_size=world)
    group = distributed.distributed_c10d._get_default_group()
    hcom = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    torch.manual_seed(7)
    activation = torch.randn(ROWS, INPUT_PER_RANK, dtype=torch.bfloat16,
                             device=device)
    weight_bf16 = torch.randn(INPUT_PER_RANK, OUTPUT, dtype=torch.bfloat16,
                              device=device)
    weight_int8 = torch.randint(-8, 8, (INPUT_PER_RANK, OUTPUT),
                                dtype=torch.int8, device=device)
    scale = torch.rand(OUTPUT, dtype=torch.float32, device=device) * 0.01
    residual = torch.randn(ROWS, OUTPUT, dtype=torch.bfloat16, device=device)
    quantized, pertoken = torch_npu.npu_dynamic_quant(activation)
    fused = torch.ops.npu.npu_mm_all_reduce_base

    if rank == 0:
        print(f"world {world}, x1 {tuple(activation.shape)}, "
              f"x2 {tuple(weight_bf16.shape)}")
    report("bf16 × bf16", lambda: fused(activation, weight_bf16, hcom), rank)
    report("bf16 × bf16 + x3 残差",
           lambda: fused(activation, weight_bf16, hcom, x3=residual), rank)
    report("bf16 × int8 antiquant",
           lambda: fused(activation, weight_int8, hcom, antiquant_scale=scale,
                         antiquant_group_size=0), rank)
    report("int8 × int8 pertoken dequant",
           lambda: fused(quantized, weight_int8, hcom, dequant_scale=scale,
                         pertoken_scale=pertoken), rank)
    report("bf16 × bf16 comm_turn=2",
           lambda: fused(activation, weight_bf16, hcom, comm_turn=2), rank)

    distributed.destroy_process_group()


main()
