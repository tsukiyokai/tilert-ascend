    TORCH_CHECK(rows >= 1 && rows <= kMaximumRows, "rows must be in [1,32]");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0 &&
                epsilon <= std::numeric_limits<float>::max() &&
                static_cast<float>(epsilon) > 0,
                "epsilon must be finite and positive in FP32");
    CheckTensor(input, input, at::kBFloat16, {rows, kHidden}, "input");
    CheckTensor(residual, input, at::kBFloat16, {rows, kHidden}, "residual");
    CheckTensor(gamma, input, at::kBFloat16, {kHidden}, "gamma");
    CheckTensor(normalized, input, at::kBFloat16, {rows, kHidden}, "normalized");
    CheckTensor(retained, input, at::kBFloat16, {rows, kHidden}, "retained");
    CheckTensor(rstd, input, at::kFloat, {rows, 1}, "rstd");
    std::vector<at::Tensor> read{input, residual, gamma};
    if (beta.has_value()) {
        CheckTensor(*beta, input, at::kBFloat16, {kHidden}, "beta");
        read.push_back(*beta);
    }
    const std::vector<at::Tensor> written{normalized, retained, rstd};
    for (size_t i = 0; i < written.size(); ++i) {
        for (const auto& source : read) {
            at::assert_no_overlap(written[i], source);
        }
        for (size_t j = 0; j < i; ++j) {
            at::assert_no_overlap(written[i], written[j]);
        }
    }
    std::lock_guard<std::mutex> guard(launch_mutex);
    const auto found = launches.find(input.get_device());
    TORCH_CHECK(found != launches.end() && blocks > 0 &&
                blocks <= found->second.maximum_blocks &&
                static_cast<uint64_t>(sync_base) == found->second.sync_base,
                "call norm_prepare for this context before capture");
}

at::Tensor BiasOut(const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
    at::Tensor& normalized, at::Tensor& retained, at::Tensor& rstd,
    double epsilon, int64_t blocks, int64_t sync_base)
{
    const c10::OptionalDeviceGuard device_guard(input.device());
    CheckInputs(input, residual, gamma, beta, normalized, retained, rstd,
                epsilon, blocks, sync_base);
    whole_layer_norm_launch(blocks, c10_npu::getCurrentNPUStream().stream(),
        sync_base, input.data_ptr(), residual.data_ptr(), gamma.data_ptr(),
        beta.has_value() ? beta->data_ptr() : nullptr, normalized.data_ptr(),
        retained.data_ptr(), rstd.data_ptr<float>(), nullptr, nullptr,
        input.size(0), static_cast<float>(epsilon));
    return normalized;
}

at::Tensor QuantOut(const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
    at::Tensor& normalized, at::Tensor& retained, at::Tensor& rstd,
    at::Tensor& quantized, at::Tensor& scales,
    double epsilon, int64_t blocks, int64_t sync_base)
{
    const c10::OptionalDeviceGuard device_guard(input.device());
    CheckInputs(input, residual, gamma, beta, normalized, retained, rstd,
                epsilon, blocks, sync_base);
    CheckTensor(quantized, input, at::kChar, {input.size(0), kHidden}, "quantized");
    CheckTensor(scales, input, at::kFloat, {input.size(0), 1}, "scales");
    for (const auto& tensor : {input, residual, gamma, normalized, retained, rstd}) {
        at::assert_no_overlap(quantized, tensor);
        at::assert_no_overlap(scales, tensor);
    }
    if (beta.has_value()) {
        at::assert_no_overlap(quantized, *beta);
        at::assert_no_overlap(scales, *beta);
    }
    at::assert_no_overlap(quantized, scales);
    whole_layer_norm_launch(blocks, c10_npu::getCurrentNPUStream().stream(),
        sync_base, input.data_ptr(), residual.data_ptr(), gamma.data_ptr(),
        beta.has_value() ? beta->data_ptr() : nullptr, normalized.data_ptr(),
        retained.data_ptr(), rstd.data_ptr<float>(), quantized.data_ptr(),
        scales.data_ptr<float>(), input.size(0), static_cast<float>(epsilon));
    return quantized;
}
}

TORCH_LIBRARY_FRAGMENT(tilert, library)
{
    library.def("norm_prepare(Tensor input, int blocks=0) -> (int, int)");
    library.def("norm_bias_out(Tensor input, Tensor residual, Tensor gamma, "
        "Tensor? beta, Tensor(a!) normalized, Tensor(b!) retained, Tensor(c!) rstd, "
        "float epsilon, int blocks, int sync_base) -> Tensor(a!)");
    library.def("norm_quant_out(Tensor input, Tensor residual, Tensor gamma, "
        "Tensor? beta, Tensor(a!) normalized, Tensor(b!) retained, Tensor(c!) rstd, "
        "Tensor(d!) quantized, Tensor(e!) scales, "
        "float epsilon, int blocks, int sync_base) -> Tensor(d!)");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("norm_prepare", &Prepare);
    library.impl("norm_bias_out", &BiasOut);
    library.impl("norm_quant_out", &QuantOut);
}
