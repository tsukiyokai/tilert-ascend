


def forward(self, x_1, residual_1, gamma_1, beta_1):
    npu_add_rms_norm_bias = torch.ops._C_ascend.npu_add_rms_norm_bias.default(x_1, residual_1, gamma_1, beta_1, 1e-05);  x_1 = residual_1 = gamma_1 = beta_1 = None
    getitem = npu_add_rms_norm_bias[0]
    getitem_2 = npu_add_rms_norm_bias[2];  npu_add_rms_norm_bias = None
    npu_dynamic_quant = torch.ops.npu.npu_dynamic_quant.default(getitem, dst_type = 1);  getitem = None
    getitem_3 = npu_dynamic_quant[0]
    getitem_4 = npu_dynamic_quant[1];  npu_dynamic_quant = None
    return (getitem_3, getitem_4, getitem_2)
    