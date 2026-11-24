#include "contract.h"
#include "../bridge/source/ffn_bridge/host_validate.h"
#include "../body/source/routed_int4/host_validate.h"

namespace ffn_composed_host {
using Tensors = std::vector<at::Tensor>;
namespace bridge = ffn_composed_bridge_host;
namespace body = ffn_composed_body_host;

void CheckCrossOverlap(at::TensorList first, at::TensorList second)
{
    for (const auto& left : first) {
        for (const auto& right : second) {
            at::assert_no_overlap(left, right);
        }
    }
}

std::tuple<Tensors, Tensors, int64_t> Prepare(
    const at::Tensor& input, int64_t blocks, int64_t weight_layout)
{
    auto [bridge_scratch, bridge_sync] = bridge::Prepare(input, blocks);
    auto [body_scratch, body_sync] = body::PreparePipeline(
        bridge_scratch[ffn_bridge::kSharedCodes], ffn_bridge::kExperts,
        blocks, weight_layout);
    TORCH_CHECK(bridge_sync == body_sync,
                "bridge and body preparation require one device synchronization base");
    CheckCrossOverlap(bridge_scratch, body_scratch);
    return {bridge_scratch, body_scratch, bridge_sync};
}

void Forward(const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& gamma, const at::Tensor& gate, const at::Tensor& bias,
    at::TensorList routed_weights, at::TensorList shared_weights,
    Tensors bridge_scratch, Tensors body_scratch, double epsilon,
    int64_t blocks, int64_t weight_layout, int64_t sync)
{
    const c10::OptionalDeviceGuard device_guard(input.device());
    TORCH_CHECK(routed_weights.size() == ffn_composed::kRoutedWeights,
                "routed_weights requires GU, GU bias/scale, down, down bias/scale");
    TORCH_CHECK(shared_weights.size() == ffn_composed::kSharedWeights,
                "shared_weights requires GU weight/scale and down weight/scale");
    // Frozen Forward contains all original checks; its launch alone is removed.
    bridge::Forward(input, residual, gamma, gate, bias, bridge_scratch,
                    epsilon, blocks, sync);
    const int64_t rows = input.size(0);
    // This is a CPU metadata view over a retained GM owner. It allocates no
    // device memory and the direct launch copies only its data address.
    const Tensors shared_inputs = {
        bridge_scratch[ffn_bridge::kSharedCodes],
        bridge_scratch[ffn_bridge::kSharedScales].view({rows}),
        shared_weights[0], shared_weights[1], shared_weights[2], shared_weights[3]};
    body::FromBFloat16(bridge_scratch[ffn_bridge::kNormalized],
        routed_weights[0], routed_weights[1], routed_weights[2],
        routed_weights[3], routed_weights[4], routed_weights[5],
        bridge_scratch[ffn_bridge::kIds], bridge_scratch[ffn_bridge::kProbabilities],
        body_scratch, sync, ffn_bridge::kExperts, blocks, weight_layout, shared_inputs);
    CheckCrossOverlap(bridge_scratch, body_scratch);
    CheckCrossOverlap(bridge_scratch, routed_weights);
    CheckCrossOverlap(bridge_scratch, shared_weights);
    CheckCrossOverlap(body_scratch, {input, residual, gamma, gate, bias});

    auto bridge_addresses = bridge::Addresses(bridge_scratch, {});
    std::array<void*, ffn_composed::kBodyBuffers> body_addresses;
    std::array<void*, ffn_composed::kRoutedWeights> weight_addresses;
    for (size_t i = 0; i < body_addresses.size(); ++i) {
        body_addresses[i] = body_scratch[i].data_ptr();
    }
    for (size_t i = 0; i < weight_addresses.size(); ++i) {
        weight_addresses[i] = routed_weights[i].data_ptr();
    }
    const auto shared = body::CheckSharedInputs(shared_inputs, body_scratch,
        body_scratch[routed_int4::kRestoredRows]);
    ffn_composed_launch(c10_npu::getCurrentNPUStream().stream(), sync,
        input.data_ptr(), residual.data_ptr(), gamma.data_ptr(), gate.data_ptr(),
        bias.data_ptr(), weight_addresses.data(), bridge_addresses.data(),
        body_addresses.data(), shared, rows, blocks, weight_layout,
        static_cast<float>(epsilon));
}
}  // namespace ffn_composed_host

TORCH_LIBRARY_FRAGMENT(tilert_ffn_composed, library)
{
    library.def("prepare(Tensor input, int blocks=24, int weight_layout=1) "
                "-> (Tensor[], Tensor[], int)");
    library.def("forward_out(Tensor input, Tensor residual, Tensor gamma, "
                "Tensor gate, Tensor bias, Tensor[] routed_weights, "
                "Tensor[] shared_weights, Tensor(a!)[] bridge_scratch, "
                "Tensor(b!)[] body_scratch, float epsilon, int blocks, "
                "int weight_layout, int sync) -> ()");
}

TORCH_LIBRARY_IMPL(tilert_ffn_composed, PrivateUse1, library)
{
    library.impl("prepare", &ffn_composed_host::Prepare);
    library.impl("forward_out", &ffn_composed_host::Forward);
}
