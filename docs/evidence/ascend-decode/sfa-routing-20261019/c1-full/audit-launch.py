from pathlib import Path
import hashlib
import json
import re
import sys

output, vendor = Path(sys.argv[1]), Path(sys.argv[2])
expected_key = int(sys.argv[3])
log = Path(str(output) + '.launch.log').read_text(errors='replace')
keys = sorted({int(x) for x in re.findall(r'SFA tilingKey_:\s*(\d+)', log)})
assert keys == [expected_key], keys
selected = list(dict.fromkeys(re.findall(
    r'OpName:\[SparseFlashAttention_0\] Reading json path: ([^\r\n]+)', log)))
assert len(selected) == 1, selected
path = Path(selected[0].strip())
assert path.is_relative_to(vendor), (path, vendor)
descriptor = json.loads(path.read_text())
elf = path.with_suffix('.o')
sha = hashlib.sha256(elf.read_bytes()).hexdigest()
assert descriptor['sha256'] == sha
p = output / 'result.json'
x = json.loads(p.read_text())
x.update(observed_tiling_keys=keys, expected_tiling_key=expected_key,
         tiling_key_verified=True, selected_descriptor_path=str(path),
         selected_descriptor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         selected_device_elf_path=str(elf), selected_device_elf_sha256=sha,
         descriptor_matches_elf=True)
p.write_text(json.dumps(x, indent=2) + '\n')
