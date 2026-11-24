"""Audit the failure position and original GMM1 input-count contract."""
import hashlib
import json
from pathlib import Path
import shutil

out=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-diagnostic-int8-m1-v1-a3-194')
p=out/'decoder-diagnostic.json'
r=json.loads(p.read_text())
events=r['events']
names=[event['event'] for event in events]
assert not r['completed'] and r['gmm1_call_count']==1
assert names[-1]=='gmm1_completion_failed'
assert names.index('gmm1_producers_complete') < names.index('gmm1_submit') < names.index('gmm1_submitted') < names.index('gmm1_completion_failed')
assert not any(name.startswith('dispose_') for name in names)
assert '507015' in events[-1]['error']
ready=next(event for event in events if event['event']=='gmm1_inputs_ready')
group=ready['group_list']
assert group['group_list_type']==1 and group['valid'] and group['nonnegative']
assert group['all_row_ends_in_bounds'] and len(group['values'])==256
assert group['rows']==group['sum_counts']==sum(group['values'])==8
assert group['active_experts']==[45,93,100,132,151,190,219,226]
assert all(group['values'][expert]==1 for expert in group['active_experts'])
summary={
    'pass':False,
    'scope':r['scope'],
    'diagnostic_report_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'pre_gmm1_synchronize_pass':True,
    'gmm1_submission_returned':True,
    'post_gmm1_synchronize_error':507015,
    'dispose_executed_before_error':False,
    'group_list':group,
    'interpretation':'This run fails before input disposal and has valid group counts; neither invalid counts nor the later dispose is required to reproduce this failure.',
    'unresolved':'Packed INT4/NZ conversion, actual device address calculations and host workspace ownership require further inspection.',
    'original_forward_sha256':r['original_forward']['sha256'],
    'gmm1_schema':r['gmm1_schema'],
}
(out/'failure-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,out/'audit.py')
print(json.dumps({k:v for k,v in summary.items() if k!='group_list'},indent=2))
