import os
import sys
import torch
import torch.distributed as dist
import torch_npu
from tilert.models.glm_5_2_ascend import onesided_reduce

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'npu:{rank}')
    torch.npu.set_device(device)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

    reduce = onesided_reduce.OneSidedReduce(rank, device, world_size, 'tcp://127.0.0.1:9487')

    num_tokens = 32
    hidden = 6144
    slice_tokens = num_tokens // world_size  # 4 tokens

    torch.manual_seed(100 + rank)
    partial = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    routed_slice = torch.randn(slice_tokens, hidden, dtype=torch.bfloat16, device=device)
    scale = 2.5

    # In GLM-5.2, residual is replicated across TP ranks
    torch.manual_seed(999)
    residual = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    gamma = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    out_buf = torch.empty_like(partial)

    # Reference calculation:
    # 1. AllReduce shared_output (partial)
    partial_all = partial.clone().float()
    dist.all_reduce(partial_all)
    partial_reduced = partial_all.bfloat16()

    # 2. Gather routed_slice across all ranks
    gathered_slices = [torch.empty_like(routed_slice) for _ in range(world_size)]
    dist.all_gather(gathered_slices, routed_slice)
    routed_full = torch.cat(gathered_slices, dim=0)

    ref_moe = (partial_reduced.float() + (routed_full.float() * scale).bfloat16().float()).bfloat16()
    # Alternatively, without intermediate cast:
    ref_moe_fp32 = (partial_reduced.float() + routed_full.float() * scale).bfloat16()

    ref_res = (ref_moe.float() + residual.float()).bfloat16()

    eps = 1e-5
    ref_var = ref_res.float().pow(2).mean(-1, keepdim=True)
    ref_rstd = torch.rsqrt(ref_var + eps)
    ref_norm = (ref_res.float() * ref_rstd * gamma.float()).bfloat16()

    cann_q, cann_s = torch_npu.npu_dynamic_quant(ref_norm, dst_type=torch.int8)

    norm_out, res_out, q_out, s_out = torch.ops.tilert.onesided_all_reduce_rmsnorm_quant(
        partial, out_buf, routed_slice, scale, residual, gamma, None, eps, True, True
    )
    torch.npu.synchronize()

    if rank == 0:
        print('=== MOE COMBINED CHECK ===')
        diff_out_buf = (out_buf.float() - ref_moe.float()).abs().max().item()
        diff_out_buf2 = (out_buf.float() - ref_moe_fp32.float()).abs().max().item()
        print('Max diff in out_buf vs ref_moe:', diff_out_buf, 'vs ref_moe_fp32:', diff_out_buf2)

        print('=== RESIDUAL CHECK ===')
        diff_res = (res_out.float() - ref_res.float()).abs().max().item()
        print('Max diff in residual:', diff_res)

        print('=== NORM CHECK ===')
        diff_norm = (norm_out.float() - ref_norm.float()).abs().max().item()
        print('Max diff in norm:', diff_norm)

        print('=== QUANT CHECK ===')
        diff_cann = (q_out != cann_q).sum().item()
        print(f'Mismatch count vs CANN npu_dynamic_quant: {diff_cann} / {q_out.numel()} ({(diff_cann/q_out.numel())*100:.4f}%)')
        print('Sample q_out[0, :5]:', q_out[0, :5].tolist())
        print('Sample cann_q[0, :5]:', cann_q[0, :5].tolist())
        print('Sample s_out[:4]:', s_out[:4].tolist())
        print('Sample cann_s[:4]:', cann_s[:4].tolist())

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
