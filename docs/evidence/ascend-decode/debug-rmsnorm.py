import json
import torch, torch_npu, tilelang
from tilelang import language as T
@T.prim_func
def debug_rms(x:T.Tensor((256,), 'bfloat16'), inv:T.Tensor((1,), 'float32'), y:T.Tensor((256,), 'float32'), b:T.Tensor((256,), 'bfloat16')):
 with T.Kernel(1,threads=1,is_npu=True) as block:
  xb=T.alloc_ub((256,), 'bfloat16')
  xf=T.alloc_ub((256,), 'float32')
  sq=T.alloc_ub((256,), 'float32')
  v=T.alloc_ub((1,), 'float32')
  T.copy(x,xb)
  T.tile.cast(xf,xb,'CAST_NONE',256)
  T.tile.mul(sq,xf,xf)
  T.reduce_sum(sq,v,dim=0)
  T.tile.mul(v,v,1/256)
  T.tile.add(v,v,1e-5)
  T.tile.rsqrt(v,v)
  T.copy(v,inv)
  T.tile.mul(xf,xf,v[0])
  T.copy(xf,y)
  T.tile.cast(xb,xf,'CAST_RINT',256)
  T.copy(xb,b)
k=tilelang.compile(debug_rms,out_idx=[],target='ascendc',platform='A3',execution_backend='cython',pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE:True,tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC:True,tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING:False})
torch.manual_seed(20261016); x=torch.randn(256,dtype=torch.bfloat16); inv=torch.empty(1,device='npu'); y=torch.empty(256,device='npu'); b=torch.empty(256,device='npu',dtype=torch.bfloat16)
k(x.npu(),inv,y,b)
inv,y,b=inv.cpu(),y.cpu(),b.cpu(); refinv=torch.rsqrt(x.float().square().mean()+1e-5); ref=x.float()*refinv
report={'inverse':inv.item(),'reference_inverse':refinv.item(),'inverse_relative_error':(inv/refinv-1).item(),'cast_exact_vs_cpu':torch.equal(y.bfloat16(),b),'cast_mismatches':(y.bfloat16()!=b).sum().item(),'max_float_error':(y-ref).abs().max().item()}
print(json.dumps(report),flush=True)
open('/workspace/experiments/glm52/debug-rmsnorm.json','w').write(json.dumps(report,indent=2)+'\n')
