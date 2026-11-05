"""Verify the completed contracts and the failed original GMM1 device task."""
import hashlib
import json
from pathlib import Path
import shutil

out = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-forward-int8-m1-v1-a3-194')
ranks = [json.loads((out/f'rank-{rank:02d}.json').read_text()) for rank in range(16)]
stages = {stage['name']: stage for stage in ranks[0]['stages']}
assert stages['official_packed_quant_mapping']['success']
assert stages['official_packed_quant_mapping']['packed_modules_mapping']['gate_up_proj'] == ['gate_proj','up_proj']
for layer in [6,7]:
    shared = stages[f'layer{layer}_processed_shared_contract']
    assert shared['success'] and shared['path'] == 'A8_INT_FUSED'
    for key,shape in [('gate',[6144,256]),('down',[128,6144])]:
        assert shared[key]['dtype'] == 'torch.int8'
        assert shared[key]['format'] == 29 and shared[key]['shape'] == shape
    for key in ['gate_scale','down_scale']:
        assert shared[key]['dtype'] == 'torch.bfloat16'
    assert stages[f'layer{layer}_official_weight_postprocessing']['success']
assert stages['layer6_processed_shared_contract']['topk_owner_storage_ptr'] == stages['layer7_processed_shared_contract']['topk_owner_storage_ptr']
assert stages['actual_local_decoder_forward']['success'] is False
assert '507015' in ranks[0]['error']
log_path = out/'driver/debug/plog/plog-222_20261022032834329.log'
log=log_path.read_text()
kernel='GroupedMatmulSwigluQuantV2_ed7aa5e41344a2216aa977694f4ab9b3_2'
assert 'stream_id=47, report_stream_id=47, task_id=144' in log and 'fault kernel_name='+kernel in log
assert 'fixedPCOffset=0x3bd4' in log
report = {
    'pass':False,
    'scope':'Actual original decoder layer 6 forward with TP16 shard dimensions, synthetic weights and local TP boundary cuts; one NPU',
    'shared_w8a8_contract_pass':True,
    'official_weight_postprocessing_layers':[6,7],
    'shared_topk_owner_identity_pass':True,
    'native_arm_executed':False,
    'failed_task':{'kernel':kernel,'stream':47,'task':144,'aic_core':20,'fixed_pc_offset':'0x3bd4','error':507015},
    'established':'Driver identifies the failed routed GMM1 task',
    'unresolved':'The driver task identity does not distinguish invalid group counts, packed-weight descriptors or input lifetime',
    'driver_log_sha256':hashlib.sha256(log_path.read_bytes()).hexdigest(),
    'rank_zero_report_sha256':hashlib.sha256((out/'rank-00.json').read_bytes()).hexdigest(),
    'tp16_collective_validation':False,
}
(out/'failure-summary.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(__file__,out/'audit.py')
print(json.dumps(report,indent=2))
