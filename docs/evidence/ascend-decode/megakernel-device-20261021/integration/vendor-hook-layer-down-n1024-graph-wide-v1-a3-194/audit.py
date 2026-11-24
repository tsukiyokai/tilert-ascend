import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path('/tmp/tilert-megakernel-validation-20261021')
repo=Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
name=sys.argv[1]
widths=list(map(int,sys.argv[2].split(',')))
iterations=int(sys.argv[3])
raw=root/(name+'-raw')
p=raw/name
r=json.loads((p/'local-chain-graph.json').read_text())
s=json.loads((p/'summary.json').read_text())
assert s['success'] and s['exitcodes']==[0]*16
assert s['arguments']['forward_widths']==widths
assert r['pass'] and r['widths']==widths and r['mode']=='int8'
assert r['blocks']==24 and r['iterations']==iterations and len(r['cases'])==iterations*len(widths)
assert r['tp_geometry']==16 and r['tp_collectives_executed']==0
assert not r['timing'] and not r['original_decoder_forward_called']
assert 'active table one with E=0' in r['poison_contract']
expected={
 'script':'03a71b694cd31a73fdd067e22d1a179a99f702116d9f1b9c75d72689cb439872',
 'component_fixture':'d45472f81a2ce795f91a507401fef37fb24ef85bf065ff1c93a5ee4f6a96b49c',
 'actual_decoder_class':'96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d',
 'graph_wrapper':'8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955',
 'graph_capture':'a6c51c8bf24d676816f0d4450d24520abd6029cd3a1031a2b8d9140eed40407f',
 'forward_context':'b8baf98e6708bf9d79ff8101ce74e94122ce43450033ab82889955676889cf8d',
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
 prepared=next(v for v in r['captures'] if v['rows']==width)
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
 assert [c['iteration'] for c in cases]==list(range(iterations))
 hashes={key:{c[key]['sha256'] for c in cases} for key in ('output','retained')}
 assert all(len(v)==iterations for v in hashes.values())
 hidden={c['inputs']['tensors']['hidden']['sha256'] for c in cases}
 slots={c['inputs']['tensors']['slot_i64:0']['sha256'] for c in cases}
 index={c['inputs']['tensors']['slot_i64:2']['sha256'] for c in cases}
 lengths={json.dumps(c['inputs']['actual_main_lengths']) for c in cases}
 assert len(hidden)==iterations and len(slots)==iterations and len(index)==iterations and len(lengths)==iterations
 changes[str(width)]={'output_hashes':iterations,'retained_hashes':iterations,'hidden_hashes':iterations,'main_slots_hashes':iterations,'index_slots_hashes':iterations,'live_length_sets':iterations}
 for c in cases:
  assert c['pass'] and len(c['checks'])==109
  for key,q in c['checks'].items():
   assert q['pass'] and q['bitwise_equal'] and q['bitwise_different_bytes']==0,(width,c['iteration'],key)
  assert c['explicit_replay']==c['iteration']+1 and c['python_calls_during_replay']==0
  assert c['output']['finite'] and c['retained']['finite']
  assert c['output']['nonzero_elements'] and c['retained']['nonzero_elements']
  assert c['inputs']['tensors']['slot_i64:0']['sha256']!=c['inputs']['tensors']['slot_i64:2']['sha256']
  count+=len(c['checks'])
 prepared=next(v for v in r['captures'] if v['rows']==width)
 assert prepared['runnable_calls']==1 and prepared['capture_counter_delta']==1
 assert prepared['launch_modes']==['int8','shared']
 assert prepared['capture_stream'] not in (prepared['caller_stream'],prepared['outer_capture_stream'])
 assert prepared['workspace_preallocated'] and prepared['observed_boundaries']==109
 assert prepared['graph_type']=='UpdatableGraph'
 for number in ('layer6','layer7'):
  assert len({c['attention'][number]['sha256'] for c in cases})==iterations
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
  assert v['result']['local_component_graph_executed'] and not v['result']['full_decoder_forward_executed']
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
 'scope':'Official ACLGraphWrapper for two consecutive local MoE layers: one MIX launch per layer plus persistent top-k copy and observation consumers',
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
 'graph_capture':True,'actual_captures':len(r['captures']),'explicit_replays':len(r['cases']),'python_calls_per_capture':1,'python_calls_per_replay':0,'timing':False,'tp16_collectives':0,'checkpoint_loaded':False,
 'report_sha256':hashlib.sha256((p/'local-chain-graph.json').read_bytes()).hexdigest(),
}
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
