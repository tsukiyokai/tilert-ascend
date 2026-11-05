import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path('/tmp/tilert-megakernel-validation-20261021')
repo=Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
name=sys.argv[1]
widths=list(map(int,sys.argv[2].split(',')))
raw=root/(name+'-raw')
p=raw/name
r=json.loads((p/'local-chain.json').read_text())
s=json.loads((p/'summary.json').read_text())
assert s['success'] and s['exitcodes']==[0]*16
assert s['arguments']['forward_widths']==widths
assert r['pass'] and r['widths']==widths and r['mode']=='int8'
assert r['blocks']==24 and r['iterations']==8 and len(r['cases'])==8*len(widths)
assert r['tp_geometry']==16 and r['tp_collectives_executed']==0
assert r['same_addresses'] and r['fresh_output_poison'] and not r['timing']
expected={
 'script':'d45472f81a2ce795f91a507401fef37fb24ef85bf065ff1c93a5ee4f6a96b49c',
 'decoder_class':'96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d',
 'input_norm':'75491119b6b494d6d399a552bf005d57db40bfe2a11d5ae13259c68f824e02b4',
 'attention_packing':'336eaff0de153d32439d4d2b0557522b313a24a7e5bef82b8dc11f1257511c01',
 'ffn_packing':'7d5d3517e1c97ecb61cb77252b653089bb8adad54b19998d7eadbbdb36d5e8c6',
}
for key,h in expected.items():assert r[key]['sha256']==h,key
assert set(r['libraries'].values())=={
 '609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f',
 '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
 '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
 'bbd52e425a1e94bb199e464e3bf6001c4420e8238afe993e8741e38af50b77ef',
}
tiling_count=0
for width in widths:
 prepared=next(v for v in r['prepared'] if v['rows']==width)
 addresses=[owner['body'][7][0] for owner in prepared['owners']]
 records=[v for v in r['down_tiling_checks'] if v['rows']==width]
 assert len(records)==16
 assert [(v['iteration'],v['arm']) for v in records]==[(i,a) for i in range(8) for a in ('separate','combined')]
 for values in [prepared['down_tiling']]+[v['layers'] for v in records]:
  assert len(values)==2 and [v['address'] for v in values]==addresses
  for value in values:
   assert value['pass'] and value['bytes']==200
   assert value['sha256']=='c207230aafb11e2c2f825cd9a2ea70147ee7733046b8b2957593c9c2bc3e8009'
   tiling_count+=1
assert tiling_count==34*len(widths)
count=0
changes={}
for width in widths:
 cases=[c for c in r['cases'] if c['rows']==width]
 assert [c['iteration'] for c in cases]==list(range(8))
 hashes={key:{c['outputs'][key]['sha256'] for c in cases} for key in ('consumer.output','consumer.retained')}
 assert all(len(v)==8 for v in hashes.values())
 hidden={c['inputs']['tensors']['hidden']['sha256'] for c in cases}
 slots={c['inputs']['tensors']['slot_i64:0']['sha256'] for c in cases}
 index={c['inputs']['tensors']['slot_i64:2']['sha256'] for c in cases}
 lengths={json.dumps(c['inputs']['actual_main_lengths']) for c in cases}
 assert len(hidden)==8 and len(slots)==8 and len(index)==8 and len(lengths)==8
 changes[str(width)]={'output_hashes':8,'retained_hashes':8,'hidden_hashes':8,'main_slots_hashes':8,'index_slots_hashes':8,'live_length_sets':8}
 for c in cases:
  assert c['pass'] and len(c['checks'])==109
  for key,q in c['checks'].items():
   assert q['pass'] and q['bitwise_equal'] and q['bitwise_different_bytes']==0,(width,c['iteration'],key)
  for d in c['candidate_diagnostics'].values():assert d['finite']
  assert c['inputs']['tensors']['slot_i64:0']['sha256']!=c['inputs']['tensors']['slot_i64:2']['sha256']
  count+=len(c['checks'])
 prepared=next(v for v in r['prepared'] if v['rows']==width)
 first,second=prepared['owners']
 assert first['attention_mode']=='int8' and second['attention_mode']=='shared'
 assert first['bridge'][10]==second['norm'][0]
 assert first['bridge'][1]==second['norm'][1]
 assert first['bridge'][10][0]!=first['bridge'][1][0]
 assert first['norm'][6][1]==[width,6144] and first['norm'][6][3]=='torch.int8'
 assert first['sync'] and first['sync']==second['sync']
for rank in range(16):
 v=json.loads((p/f'rank-{rank:02d}.json').read_text())
 assert v['torch']=='2.12.0+cu130' and v['torch_npu']=='2.12.0'
 assert not v['npu_initialized_before'] and v['npu_initialized_after']==(rank==0)
 assert v['rank_zero_status']['success'] and all(x['success'] for x in v['stages'])
 if rank==0:
  assert v['result']['local_component_chain_executed'] and not v['result']['full_decoder_forward_executed']
a=json.loads((p/'nz-storage-correction.json').read_text())
assert a['completed'] and a['additional_hbm_bytes']==0 and len(a['banks'])==4
for bank in a['banks']:
 assert bank['raw_bytes_equal'] and bank['raw_sha256_before']==bank['raw_sha256_after']
 assert bank['new_device_allocation_bytes']==0 and bank['before']['data_ptr']==bank['after']['data_ptr']
dest=repo/'docs/evidence/ascend-decode/megakernel-device-20261021/integration'/(name+'-a3-194')
dest.mkdir(exist_ok=False)
for directory in raw.iterdir():shutil.copytree(directory,dest/directory.name)
for suffix in ('command.json','launcher.log','host-memory.json'):
 shutil.copy2(root/(name+'.'+suffix),dest/suffix)
shutil.copy2(Path(__file__),dest/'audit.py')
shutil.copy2(root/'archive-local-fusion-device.py',dest/'archive.py')
shutil.copy2(root/'run-layer-down-n1024-wide-v1.py',dest/'runner.py')
shutil.copy2(root/(name+'-admission.jsonl'),dest/'admission.jsonl')
shutil.copy2(root/'two-die-idle-gate.py',dest/'idle-gate.py')
summary={
 'scope':'Two consecutive local MoE layers: one MIX launch per layer, separate accepted components as reference',
 'host':'a3-194','physical_die':4,'mode':'int8','rows':widths,'blocks':24,
 'cases':len(r['cases']),'raw_byte_checks':count,'pass':True,
 'weight_source':'Synthetic parameters in original Decoder6/7 classes with official postprocessing',
 'norm_reference':'Original input norm and npu_dynamic_quant',
 'reference_libraries':list(r['libraries'].values())[1:],
 'candidate_library_sha256':r['libraries']['/decoder/layer_local_chain.so'],
 'defined_boundaries_per_case':109,'changing_inputs_and_outputs':changes,
 'down_tiling_200_byte_checks':tiling_count,'down_tiling_sha256':'c207230aafb11e2c2f825cd9a2ea70147ee7733046b8b2957593c9c2bc3e8009',
 'down_preparation':'N1024 M32 K64 host record; 609 embedded device arithmetic unchanged',
 'tp_contract':'Both TP reductions are identity; actual local attention output feeds FFN',
 'next_layer_input':'Previous layer combined output and retained residual, same GPU owners',
 'topk_transfer':'Explicit copy to the original persistent top-k owner before layer7',
 'graph_capture':False,'timing':False,'tp16_collectives':0,'checkpoint_loaded':False,
 'report_sha256':hashlib.sha256((p/'local-chain.json').read_bytes()).hexdigest(),
}
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
