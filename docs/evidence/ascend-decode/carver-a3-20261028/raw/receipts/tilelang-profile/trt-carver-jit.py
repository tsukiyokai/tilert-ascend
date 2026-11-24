"""Compile one CUBE tile, and run only an explicitly admitted tile on A3."""
import argparse
import hashlib
import json
from pathlib import Path

import tilelang
import tilelang.language as T
import torch
import torch_npu
from tilelang.carver.arch.ascend import Ascend
from tilelang.carver.roller.policy.ascend import AscendDefaultPolicy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--block-m', type=int, required=True)
parser.add_argument('--block-n', type=int, required=True)
parser.add_argument('--compile-only', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.npu.set_device(0)
arch = Ascend()
policy = AscendDefaultPolicy(arch)
bm, bn, bk = args.block_m, args.block_n, 64
admitted = policy._check_l0_usage(bm * bk * 2, bk * bn * 2, bm * bn * 4, 2)
if not args.compile_only and not admitted:
    raise ValueError('Refuse to launch a tile exceeding actual A3 budgets')
configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=configs)
def matmul(M, N, K, block_m, block_n, block_k):
    n_num = N // block_n

    @T.prim_func
    def main(A: T.Tensor((M, K), 'float16'),
             B: T.Tensor((K, N), 'float16'),
             C: T.Tensor((M, N), 'float16')):
        with T.Kernel((M // block_m) * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_l1 = T.alloc_shared((block_m, block_k), 'float16')
            b_l1 = T.alloc_shared((block_k, block_n), 'float16')
            c_l0 = T.alloc_fragment((block_m, block_n), 'float32')
            for k in T.serial(T.ceildiv(K, block_k)):
                T.copy(A[bx * block_m, k * block_k], a_l1)
                T.copy(B[k * block_k, by * block_n], b_l1)
                T.gemm_v0(a_l1, b_l1, c_l0, init=(k == 0))
            T.copy(c_l0, C[bx * block_m, by * block_n])
    return main


report = {'block': [bm, bn, bk], 'admitted': admitted,
          'compile_only': args.compile_only, 'compiled': False,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'device': str(torch.npu.get_device_properties(0)), 'checks': []}
args.output.parent.mkdir(parents=True, exist_ok=True)
try:
    kernel = matmul(256, 256, 256, bm, bn, bk)
    report['compiled'] = True
    code = kernel.get_kernel_source()
    source = args.output.with_suffix('.cpp')
    source.write_text(code)
    report['generated_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    if not args.compile_only:
        for seed in (47, 219, 1031):
            generator = torch.Generator().manual_seed(seed)
            a = (torch.randint(-8, 9, (256, 256), generator=generator).float() / 8).half()
            b = (torch.randint(-8, 9, (256, 256), generator=generator).float() / 8).half()
            expected = (a.float() @ b.float()).half()
            actual = kernel(a.npu(), b.npu())
            torch.npu.synchronize()
            raw = actual.cpu().view(torch.uint8).numpy().tobytes()
            truth = expected.view(torch.uint8).numpy().tobytes()
            report['checks'].append({'seed': seed, 'bitwise_equal': raw == truth,
                                     'actual_sha256': hashlib.sha256(raw).hexdigest(),
                                     'expected_sha256': hashlib.sha256(truth).hexdigest()})
        report['pass'] = all(check['bitwise_equal'] for check in report['checks'])
        if not report['pass']:
            raise RuntimeError('Actual A3 GEMM differs from independent dyadic oracle')
except BaseException as error:
    report['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
