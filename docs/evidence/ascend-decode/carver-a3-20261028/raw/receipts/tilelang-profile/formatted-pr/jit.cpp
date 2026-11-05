#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR A_handle,  GM_ADDR B_handle,  GM_ADDR C_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> A;
  A.SetGlobalBuffer((__gm__ half*)A_handle);
  AscendC::GlobalTensor<half> B;
  B.SetGlobalBuffer((__gm__ half*)B_handle);
  AscendC::GlobalTensor<half> C;
  C.SetGlobalBuffer((__gm__ half*)C_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto c_l0 = ascend_l0c.GetWithOffset<float>(32768, 0);
  auto a_l1 = ascend_l1.GetWithOffset<half>(8192, 0);
  auto b_l1 = ascend_l1.GetWithOffset<half>(16384, 16384);
  if ASCEND_IS_AIC {
    for (int32_t k = 0; k < 4; ++k) {
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(2);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_l1<half, 128, 64>(a_l1[0], A[((cid * 32768) + (k * 64))], 256, 128, 64, true);
      tl::ascend::copy_gm_to_l1<half, 64, 256>(b_l1[0], B[(k * 16384)], 256, 64, 256, true);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<half, float, 128, 256, 64, false, false, 128>(a_l1[0], b_l1[0], c_l0[0], ascend_l0a, ascend_l0b, (k == 0), 256);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
    tl::ascend::copy_l0c_to_gm<float, half, layout::RowMajor, 128, 256, 0>(C[(cid * 32768)], c_l0[0], 256, 128, 256, 0);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<2, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}
