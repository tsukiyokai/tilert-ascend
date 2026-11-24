from pathlib import Path
import hashlib,inspect,json
import torch,torch_npu
from torch.distributed import distributed_c10d as c10d
report={"torch":torch.__version__,"torch_npu":torch_npu.__version__,"npu_initialized_before":torch.npu.is_initialized(),"mapped_davinci_devices":[str(p) for p in Path("/dev").glob("davinci*")],"sources":{}}
for name,fn in (("all_reduce",c10d.all_reduce),("graph",torch.npu.graph),("Event",torch.npu.Event)):
 target=inspect.unwrap(fn)
 path=Path(inspect.getfile(target))
 lines,start=inspect.getsourcelines(target)
 text=''.join(lines)
 (Path("/out")/(name+".py")).write_text(text)
 report["sources"][name]={"path":str(path),"file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"first_line":start,"source_sha256":hashlib.sha256(text.encode()).hexdigest()}
report["npu_initialized_after"]=torch.npu.is_initialized()
assert not report["npu_initialized_before"] and not report["npu_initialized_after"] and not report["mapped_davinci_devices"]
report["pass"]=True
Path("/out/report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
