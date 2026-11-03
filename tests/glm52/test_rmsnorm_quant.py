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

    reduce = onesided_reduce.OneSidedReduce(rank, device, world_size, 'tcp://127.0.0.1:9485')

def test_tokens(reduce, rank, device, num_tokens):
    hidden = 6144
    torch.manual_seed(42 + rank)
    partial = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    residual = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    out_buf = torch.empty_like(partial)

    # Reference calculation
    partial_all = partial.clone().float()
    dist.all_reduce(partial_all)
    partial_reduced = partial_all.bfloat16()

    ref_res = (partial_reduced.float() + residual.float()).bfloat16()

    eps = 1e-5
    ref_variance = ref_res.float().pow(2).mean(-1, keepdim=True)
    ref_rstd = torch.rsqrt(ref_variance + eps)
    ref_norm = (ref_res.float() * ref_rstd * gamma.float()).bfloat16()

    ref_amax = ref_norm.float().abs().max(dim=-1).values
    ref_scale = ref_amax / 127.0
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    ref_quant = torch.clamp(torch.round(ref_norm.float() / ref_scale.unsqueeze(-1)), -128, 127).to(torch.int8)

    norm_out, res_out, q_out, s_out = torch.ops.tilert.onesided_all_reduce_rmsnorm_quant(
        partial, out_buf, None, 1.0, residual, gamma, None, eps, True, True
    )
    torch.npu.synchronize()

    cann_q, cann_s = torch_npu.npu_dynamic_quant(norm_out, dst_type=torch.int8)

    if rank == 0:
        print(f'=== TOKENS={num_tokens} RESIDUAL CHECK ===')
        print('res_out nan:', torch.isnan(res_out).sum().item(), 'ref_res nan:', torch.isnan(ref_res).sum().item())
        diff_res = (res_out.float() - ref_res.float()).abs().max().item()
        print('Max diff in residual:', diff_res)

        print(f'=== TOKENS={num_tokens} NORM CHECK ===')
        print('norm_out nan:', torch.isnan(norm_out).sum().item(), 'ref_norm nan:', torch.isnan(ref_norm).sum().item())
        diff_norm = (norm_out.float() - ref_norm.float()).abs().max().item()
        print('Max diff in norm:', diff_norm)

        print(f'=== TOKENS={num_tokens} SCALE CHECK ===')
        print('s_out nan:', torch.isnan(s_out).sum().item(), 'ref_scale nan:', torch.isnan(ref_scale).sum().item())
        print('s_out values:', s_out[:5].tolist())
        diff_scale = (s_out.float() - ref_scale.float()).abs().max().item()
        print('Max diff in scale:', diff_scale)

        print(f'=== TOKENS={num_tokens} QUANT CHECK ===')
        diff_quant = (q_out.float() - ref_quant.float()).abs().max().item()
        print('Max diff in quant (vs ref):', diff_quant)
        diff_cann = (q_out != cann_q).sum().item()
        print(f'Mismatch count vs CANN npu_dynamic_quant: {diff_cann} / {q_out.numel()} ({(diff_cann/q_out.numel())*100:.4f}%)')
        if diff_cann > 0:
            mismatches = (q_out != cann_q).nonzero()
            for idx in mismatches[:10]:
                r, c = idx[0].item(), idx[1].item()
                print(f'  mismatch [{r}, {c}]: q_out={q_out[r, c].item()}, cann={cann_q[r, c].item()}, norm={norm_out[r, c].item()}, s_out={s_out.flatten()[r].item()}, cann_s={cann_s.flatten()[r].item()}')

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'npu:{rank}')
    torch.npu.set_device(device)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

    reduce = onesided_reduce.OneSidedReduce(rank, device, world_size, 'tcp://127.0.0.1:9485')

    test_tokens(reduce, rank, device, 1)
    test_tokens(reduce, rank, device, 32)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
