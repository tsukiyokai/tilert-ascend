#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void w4a8_gemv_row_kernel( GM_ADDR x_handle,  GM_ADDR packed_handle,  GM_ADDR weight_scale_handle,  GM_ADDR token_scale_handle,  GM_ADDR out_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<int8_t> x;
  x.SetGlobalBuffer((__gm__ int8_t*)x_handle);
  AscendC::GlobalTensor<uint8_t> packed;
  packed.SetGlobalBuffer((__gm__ uint8_t*)packed_handle);
  AscendC::GlobalTensor<float> weight_scale;
  weight_scale.SetGlobalBuffer((__gm__ float*)weight_scale_handle);
  AscendC::GlobalTensor<float> token_scale;
  token_scale.SetGlobalBuffer((__gm__ float*)token_scale_handle);
  AscendC::GlobalTensor<bfloat16_t> out;
  out.SetGlobalBuffer((__gm__ bfloat16_t*)out_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto block = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    block = block / 2;
  }
  auto low_sum = ascend_ub.GetWithOffset<float>(8, 0);
  auto high_sum = ascend_ub.GetWithOffset<float>(8, 32);
  auto low_mask = ascend_ub.GetWithOffset<int16_t>(2048, 64);
  auto sign_mask = ascend_ub.GetWithOffset<int16_t>(2048, 4160);
  auto x_integer = ascend_ub.GetWithOffset<int8_t>(256, 8256);
  auto packed_ub = ascend_ub.GetWithOffset<uint8_t>(2048, 8512);
  auto x_half = ascend_ub.GetWithOffset<half>(256, 10560);
  auto x_float = ascend_ub.GetWithOffset<float>(256, 11072);
  auto weight_half = ascend_ub.GetWithOffset<half>(2048, 12096);
  auto weight_integer = ascend_ub.GetWithOffset<int16_t>(2048, 16192);
  auto nibble = ascend_ub.GetWithOffset<int16_t>(2048, 20288);
  auto signed_nibble = ascend_ub.GetWithOffset<int16_t>(2048, 24384);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(4096, 28480);
  auto product = ascend_ub.GetWithOffset<float>(2048, 32576);
  auto partial = ascend_ub.GetWithOffset<float>(8, 40768);
  auto output_float = ascend_ub.GetWithOffset<float>(16, 40800);
  auto scale = ascend_ub.GetWithOffset<float>(16, 40864);
  auto row_scale = ascend_ub.GetWithOffset<float>(1, 40928);
  auto output_bfloat = ascend_ub.GetWithOffset<bfloat16_t>(16, 40960);
  auto vector = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 8);
    AscendC::Duplicate<float, false>(low_sum[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::Duplicate<float, false>(high_sum[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
    AscendC::Duplicate<int16_t, false>(low_mask[0], 15, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::Duplicate<int16_t, false>(sign_mask[0], 8, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    for (int32_t tile = 0; tile < 24; ++tile) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_ub<int8_t, 256>(x_integer[0], x[(tile * 256)], 6144, 1, 256, (signed char)0);
      tl::ascend::copy_gm_to_ub<uint8_t, 256, 8>(packed_ub[0], packed[(((block * 98304) + (vector * 49152)) + (tile * 256))], 6144, 8, 256, (uchar)0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
      AscendC::Cast<half, int8_t, false>(x_half[0], x_integer[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast<float, half, false>(x_float[0], x_half[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
      AscendC::Cast<half, uint8_t, false>(weight_half[0], packed_ub[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast<int16_t, half, false>(weight_integer[0], weight_half[0], AscendC::RoundMode::CAST_RINT, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>(0xffffffffffffffffULL, 0xffffffffffffffffULL);
      AscendC::And<int16_t, false>(nibble[0], weight_integer[0], low_mask[0], AscendC::MASK_PLACEHOLDER, 16, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Xor(signed_nibble[0], nibble[0], sign_mask[0], tmp_ub[0]);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
      AscendC::Cast<half, int16_t, false>(weight_half[0], signed_nibble[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast<float, half, false>(product[0], weight_half[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      {
      AscendC::Adds<float, false>(product[0], product[0], -8.000000e+00f, AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
      }
      for (int32_t pair = 0; pair < 8; ++pair) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Mul<float, false>(product[(pair * 256)], product[(pair * 256)], x_float[0], AscendC::MASK_PLACEHOLDER, 4, AscendC::BinaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 8, 256, -1, 256, true>(partial[0], product[0], tmp_ub.ReinterpretCast<float>()[0]);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
      AscendC::Add<float, false>(low_sum[0], low_sum[0], partial[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
      AscendC::SetVectorMask<uint8_t>(0xffffffffffffffffULL, 0xffffffffffffffffULL);
      {
      AscendC::ShiftRight<int16_t, false>(nibble[0], weight_integer[0], int16_t(4), AscendC::MASK_PLACEHOLDER, 16, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Xor(signed_nibble[0], nibble[0], sign_mask[0], tmp_ub[0]);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
      AscendC::Cast<half, int16_t, false>(weight_half[0], signed_nibble[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast<float, half, false>(product[0], weight_half[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      {
      AscendC::Adds<float, false>(product[0], product[0], -8.000000e+00f, AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
      }
      for (int32_t pair_1 = 0; pair_1 < 8; ++pair_1) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Mul<float, false>(product[(pair_1 * 256)], product[(pair_1 * 256)], x_float[0], AscendC::MASK_PLACEHOLDER, 4, AscendC::BinaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 8, 256, -1, 256, true>(partial[0], product[0], tmp_ub.ReinterpretCast<float>()[0]);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
      AscendC::Add<float, false>(high_sum[0], high_sum[0], partial[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    }
    for (int32_t pair_2 = 0; pair_2 < 8; ++pair_2) {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(4);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(4);
      AscendC::PipeBarrier<PIPE_ALL>();
      output_float.SetValue((pair_2 * 2), low_sum.GetValue(pair_2));
      AscendC::PipeBarrier<PIPE_ALL>();
      output_float.SetValue(((pair_2 * 2) + 1), high_sum.GetValue(pair_2));
    }
    tl::ascend::copy_gm_to_ub<float, 16>(scale[0], weight_scale[((block * 32) + (vector * 16))], 256, 1, 16, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<float, 1>(row_scale[0], token_scale[0], 1, 1, 1, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(6);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(6);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)65535);
    AscendC::Mul<float, false>(output_float[0], output_float[0], scale[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto row_scale_scalar = row_scale.GetValue(0);
    AscendC::Muls<float, false>(output_float[0], output_float[0], row_scale_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 16);
    AscendC::Cast<bfloat16_t, float, false>(output_bfloat[0], output_float[0], AscendC::RoundMode::CAST_RINT, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 16>(out[((block * 32) + (vector * 16))], output_bfloat[0], 256, 1, 16);
  }
}

void w4a8_gemv_row_kernel_tiling() {
}

extern "C" void call(uint8_t* x_handle, uint8_t* packed_handle, uint8_t* weight_scale_handle, uint8_t* token_scale_handle, uint8_t* out_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  w4a8_gemv_row_kernel_tiling();
  w4a8_gemv_row_kernel<<<8, nullptr, stream>>>(x_handle, packed_handle, weight_scale_handle, token_scale_handle, out_handle, fftsAddr);
}
