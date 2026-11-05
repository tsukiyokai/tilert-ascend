
import threading
from collections import OrderedDict

import torch
from torch._dynamo.testing import rand_strided

import torch_npu
from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph, AclGraphCacheInfo
from npugraph_ex._acl_concrete_graph.acl_graph_cache_utils import SerializableGraphModule
from npugraph_ex.ops._tagged_event import _npu_create_tagged_event

assert_size_stride = torch._C._dynamo.guards.assert_size_stride

from torch._dynamo.guards import _get_closure_vars
globals().update(_get_closure_vars())
globals().update({"nan": float("nan")})

def forward(*args, node_info=[], is_capturing: bool = False):
    arg0_1, arg1_1, arg2_1, arg3_1 = args

    npu_add_rms_norm_dynamic_quant = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default(arg0_1, arg1_1, arg2_1, beta = arg3_1, epsilon = 1e-05, output_mask = [True, False]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = None
    getitem = npu_add_rms_norm_dynamic_quant[0]
    getitem_2 = npu_add_rms_norm_dynamic_quant[2]
    getitem_3 = npu_add_rms_norm_dynamic_quant[3];  npu_add_rms_norm_dynamic_quant = None
    return (getitem, getitem_3, getitem_2)


compile_configs = {}
compile_configs["force_eager"] = True
compile_configs["reuse_graph_pool_in_same_fx"] = True
compile_configs["clone_output"] = False
compile_configs["capture_limit"] = "64"
compile_configs["clone_input"] = False
compile_configs["deadlock_check"] = False
compile_configs["capture_error_mode"] = "global"
compile_configs["force_recapture"] = False
compile_configs["static_kernel_compile"] = True
compile_configs["_vllm_aclnn_static_kernel_sym_range"] = "[8]"
compile_configs["_vllm_aclnn_static_kernel_sym_index"] = "0"
compile_configs["super_kernel_optimize"] = True
compile_configs["super_kernel_optimize_options"] = "{'dcci_after_kernel_end': ['.*']}"
compile_configs["remove_noop_ops"] = True
compile_configs["pattern_fusion_pass"] = True
compile_configs["frozen_parameter"] = False
compile_configs["sym_input_index"] = "[]"

acl_graph = AclGraph(fx_forward=forward, config=compile_configs)

aclgraph_cache_info = AclGraphCacheInfo(
    pool=None,
    stream=torch.npu.Stream(),
    capture_error_mode="global",
    num_warmup_iters=0,
    fx_graph_name="graph_1",
    user_inputs_mapping=OrderedDict([('arg0_1', 0), ('arg1_1', 1), ('arg2_1', 2), ('arg3_1', 3)]),
    unupdated_sym_input_index=[],
    updated_ops_param={},
    ops_update_rulers={},
    mutated_user_inputs=[],
    tagged_event_names=[],
    parameter_user_inputs=[],
    user_stream_label=set(),
    user_stream_info={},
    userinput_ref_with_output={}
)
acl_graph.load(aclgraph_cache_info)


_is_first_run = True
def kernel(*args, **kwargs):

    global _is_first_run
    if _is_first_run:
        _is_first_run = False

        assert_size_stride(args[0], (8, 6144), (6144, 1))
        assert_size_stride(args[1], (8, 6144), (6144, 1))
        assert_size_stride(args[2], (6144,), (1,))
        assert_size_stride(args[3], (6144,), (1,))
    return acl_graph(*args, **kwargs)

def main():
    arg0_1 = rand_strided((8, 6144),(6144, 1),device ='npu:9',dtype =torch.bfloat16)
    arg1_1 = rand_strided((8, 6144),(6144, 1),device ='npu:9',dtype =torch.bfloat16)
    arg2_1 = rand_strided((6144,),(1,),device ='npu:9',dtype =torch.bfloat16)
    arg3_1 = rand_strided((6144,),(1,),device ='npu:9',dtype =torch.bfloat16)
    return kernel(arg0_1, arg1_1, arg2_1, arg3_1)
