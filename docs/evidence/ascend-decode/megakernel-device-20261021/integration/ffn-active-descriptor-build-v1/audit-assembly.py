from pathlib import Path
import hashlib,importlib.util,json,shlex,shutil,struct,subprocess
root=Path('/tmp/tilert-megakernel-validation-20261021')
repo=Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','7.150.10.78']
remote='/home/shanshan/tilert-megakernel-20261021/logs/ffn-assembly-audit-v1'
objdump='/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0/bin/llvm-objdump'
records={}
for name,folder in [('active',root/'ffn-active-build-v1'),('active-descriptor',root/'ffn-active-descriptor-runtime-v1/build-v1')]:
 data=(folder/'kernel.o').read_bytes()
 manifest=json.loads((folder/'manifest.json').read_text())
 assert hashlib.sha256(data).hexdigest()==manifest['artifacts']['kernel.o']
 offset=data.find(b'\x7fELF',1)
 h=struct.unpack_from('<16sHHIQQQIHHHHHH',data,offset)
 assert h[2]==4137,(name,h[2])
 elf=data[offset:offset+h[6]+h[11]*h[12]]
 path=root/('ffn-'+name+'-device.elf');path.write_bytes(elf)
 dest=remote+'/'+name+'.elf'
 subprocess.run(['scp','-o','BatchMode=yes','-o','ClearAllForwardings=yes',str(path),'7.150.10.78:'+dest],check=True)
 command=[objdump,'-d','--demangle',dest]
 text=subprocess.check_output(ssh+[shlex.join(command)])
 disassembly=root/('ffn-'+name+'-device-disassembly.txt');disassembly.write_bytes(text)
 records[name]={'kernel_object_sha256':hashlib.sha256(data).hexdigest(),'device_elf_offset':offset,'device_elf_sha256':hashlib.sha256(elf).hexdigest(),'disassembly_sha256':hashlib.sha256(text).hexdigest(),'disassembler_argv':command}
helper=repo/'tests/glm52/whole_layer/ffn_descriptor/assembly_audit.py'
spec=importlib.util.spec_from_file_location('assembly_audit',helper);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
r=audit.compare(root/'ffn-active-device-disassembly.txt',root/'ffn-active-descriptor-device-disassembly.txt')
r['identities']=records
out=repo/'docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-descriptor-build-v1'
(out/'assembly-audit.json').write_text(json.dumps(r,indent=2)+'\n')
shutil.copy2(helper,out/'assembly_audit.py')
shutil.copy2(Path(__file__),out/'audit-assembly.py')
print(json.dumps(r['candidate_minus_parent'],indent=2))
