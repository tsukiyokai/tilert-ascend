


def forward(self, x_1, residual_1, gamma_1, beta_1):
    npu_add_rms_norm_dynamic_quant_default = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default(x_1, residual_1, gamma_1, beta = beta_1, epsilon = 1e-05, output_mask = [True, False]);  x_1 = residual_1 = gamma_1 = beta_1 = None
    getitem_5 = npu_add_rms_norm_dynamic_quant_default[0]
    getitem_6 = npu_add_rms_norm_dynamic_quant_default[2]
    getitem_7 = npu_add_rms_norm_dynamic_quant_default[3];  npu_add_rms_norm_dynamic_quant_default = None
    return (getitem_5, getitem_7, getitem_6)
    