import json
from pathlib import Path
import torch
import torch_npu
from tilert.models.glm_5_2_ascend.ops import rms_norm, rms_norm_reference
torch.set_num_threads(1)
torch.manual_seed(20261016)
torch.npu.set_device(0)
rows = []
for width, eps in [(256,1e-5),(64,1e-6),(512,1e-6),(2048,1e-6),(6144,1e-5)]:
    for zero in (False,True):
        x = torch.zeros(1,width,dtype=torch.bfloat16) if zero else torch.randn(1,width,dtype=torch.bfloat16)
        gamma = torch.linspace(.75,1.25,width).bfloat16().float()
        expected = rms_norm_reference(x,gamma,eps)
        actual = rms_norm(x.npu(),gamma.npu(),eps).cpu()
        difference = actual.float()-expected.float()
        row = {'width':width,'epsilon':eps,'zero_input':zero,'finite':bool(actual.isfinite().all()),'max_abs':difference.abs().max().item(),'nrmse':(difference.norm()/expected.float().norm().clamp_min(1e-12)).item(),'exact':bool(torch.equal(actual,expected))}
        rows.append(row)
        print(json.dumps(row),flush=True)
report = {'device':torch.npu.get_device_name(0),'torch':torch.__version__,'seed':20261016,'results':rows,'passed':all(row['finite'] and row['nrmse']<=0.01 for row in rows),'peak_tensor_memory_bytes':torch.npu.max_memory_allocated()}
Path('/workspace/experiments/glm52/rmsnorm-npu.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['passed'] else 1)
