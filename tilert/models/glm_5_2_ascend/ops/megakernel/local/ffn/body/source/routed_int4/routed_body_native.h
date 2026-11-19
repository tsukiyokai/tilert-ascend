#ifndef TILERT_ROUTED_BODY_NATIVE_H
#define TILERT_ROUTED_BODY_NATIVE_H

#ifdef ROUTED_ENABLE_PROFILE
#error "Token tail join requires the uninstrumented NoTrace body"
#endif


#define ASCENDC_CUBE_ONLY

#include "routed_int4_kernel.h"
#include "../../../source/output_contract.h"
#include "../../../source/split_contract.h"
#include "../../../source/descriptor.h"
#include "routed_int4_dispatch.h"
#include "routed_int4_trace.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../../../source/gu_static_tiling.h"
#ifdef ROUTED_INPUT_FRONT
#include "routed_int4_front.h"
#endif

#ifdef ROUTED_SHARED_PIPELINE
#include "routed_int4_pipeline_native.h"
#ifdef ROUTED_INPUT_FRONT
#define routed_int4_fused ffn_original_shared_gu_static_unused_body_kernel
#else
#define routed_int4_fused routed_shared_fused
#endif
#elif defined(ROUTED_CONTROL)
#define routed_int4_fused routed_int4_control_fused
#elif defined(ROUTED_INPUT_FRONT)
#define routed_int4_fused routed_int4_front_fused
#elif defined(ROUTED_GU_MDL)
#define routed_int4_fused routed_int4_mdl_fused
#elif defined(ROUTED_GU_PAD16)
#define routed_int4_fused routed_int4_pad16_fused
#elif defined(ROUTED_BALANCED_NAMESPACE)
#define routed_int4_fused routed_int4_balanced_fused
#endif

namespace routed_body {

using namespace AscendC;
using namespace routed_int4;

namespace {

constexpr int64_t kSlack = 64;
constexpr int64_t kBlockFloats = 8;
using APacked = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
#ifdef ROUTED_FIXPIPE_HALF
using RoutedProduct = half;
#else
using RoutedProduct = int32_t;
#endif
using CProduct = MatmulType<TPosition::GM, CubeFormat::ND, RoutedProduct>;
using BiasNone = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
template <CubeFormat format>
using Projection = Matmul<APacked,
    MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone>;
#if defined(ROUTED_GU_MDL) || defined(ROUTED_GU_PAD16) || defined(ROUTED_FIXPIPE_HALF)
#error "GU constant tiling preserves the parent INT32 Norm geometry"
#endif
template <CubeFormat format>
using GateProjection = Matmul<APacked, MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone, ffn_original_shared_gu_static::kGateTiling>;
template <CubeFormat format>
using DownProjection = Matmul<APacked, MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone, ffn_original_shared_gu_static::kDownTiling>;
#if GU_WIDE && !defined(K11040_GU_FAST128)
template <CubeFormat format>
using GateProjection16 = Matmul<APacked, MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone, ffn_original_shared_gu_static::kGateTiling16>;
#else
template <CubeFormat format>
using GateProjection16 = Matmul<APacked, MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone, ffn_original_shared_gu_static::kGateTiling16, MatmulCallBackFunc<nullptr, nullptr, nullptr>, ffn_original_shared_gu_static::FastMatmulPolicy>;
#endif
template <CubeFormat format>
using DownProjection16 = Matmul<APacked, MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone, ffn_original_shared_gu_static::kDownTiling16, MatmulCallBackFunc<nullptr, nullptr, nullptr>, ffn_original_shared_gu_static::FastMatmulPolicy>;
template <CubeFormat format>
using GateProjection16_N1024 = Matmul<APacked, MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone, ffn_original_shared_gu_static::kGateTiling16_N1024>;
template <CubeFormat format>
using DownProjection16_N1536 = Matmul<APacked, MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone, ffn_original_shared_gu_static::kDownTiling16_N1536>;


__aicore__ inline __gm__ int4b_t* Nibbles(__gm__ int4b_t* base,
                                         int64_t elements)
{
    return (__gm__ int4b_t*)((__gm__ int8_t*)base + elements / 2);
}

template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

__aicore__ inline void Publish()
{
#ifndef K112722_NO_DDR_EXTRA
    DataSyncBarrier<MemDsbT::DDR>();
#endif
}

struct Workspace {
    TBuf<TPosition::VECCALC> input_buf, value_buf, other_buf, work_buf;
    TBuf<TPosition::VECCALC> half_buf, packed_buf, staging_buf, scale_buf;
    TBuf<TPosition::VECCALC> mask_buf, token_buf, input_scale_buf;
    TBuf<TPosition::VECCALC> maximum_buf;
    TBuf<TPosition::VECCALC> count_buf, start_buf, source_buf, slot_buf;
    TBuf<TPosition::VECCALC> weight_buf, original_weight_buf;
    LocalTensor<int8_t> input;
    LocalTensor<float> value, other, work, scale, token, maximum;
    LocalTensor<float> weight, original_weight, input_scale;
    LocalTensor<half> half_value;
    LocalTensor<int4b_t> packed;
    LocalTensor<int32_t> staging, mask, count, start, source, slot;
#ifdef ROUTED_ACTIVE_LIST
    TBuf<TPosition::VECCALC> active_buf;
    LocalTensor<int32_t> active;
#endif
#if defined(K11220_FAST_TOK_OUT) || defined(K11220_FAST_ACT_P3)
    TBuf<TPosition::VECCALC> corr_cache_buf, tok_out_buf;
    LocalTensor<float> corr_cache;
    LocalTensor<bfloat16_t> tok_out;
#endif
#ifdef K112612_OVERLAP_SH_P4
    TBuf<TPosition::VECCALC> sh_red_acc_buf, sh_red_nar_buf;
    LocalTensor<float> sh_red_acc;
    LocalTensor<bfloat16_t> sh_red_nar;
#endif

    __aicore__ inline void Init(TPipe& pipe)
    {
        constexpr int64_t span = kVectorTile + kSlack;
        pipe.InitBuffer(input_buf, span);
        pipe.InitBuffer(value_buf, span * sizeof(float));
        pipe.InitBuffer(other_buf, span * sizeof(float));
        pipe.InitBuffer(work_buf, span * sizeof(float));
        pipe.InitBuffer(half_buf, span * sizeof(half));
        pipe.InitBuffer(packed_buf, span / 2 + kSlack);
        pipe.InitBuffer(staging_buf, span * sizeof(int32_t));
        pipe.InitBuffer(scale_buf, span * sizeof(float));
        pipe.InitBuffer(mask_buf, span * sizeof(int32_t));
        pipe.InitBuffer(token_buf, kMaxRowsPerExpert * sizeof(float));
        pipe.InitBuffer(input_scale_buf, kMaxTokens * sizeof(float));

        pipe.InitBuffer(maximum_buf,
                        kMaxRowsPerExpert * kBlockFloats * sizeof(float));
        pipe.InitBuffer(count_buf, kLocalExperts * sizeof(int32_t));
        pipe.InitBuffer(start_buf, kLocalExperts * sizeof(int32_t));
        pipe.InitBuffer(source_buf, 2 * kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(slot_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(weight_buf, kMaxRoutes * sizeof(float));
        pipe.InitBuffer(original_weight_buf, kMaxRoutes * sizeof(float));
        input = input_buf.Get<int8_t>();
        value = value_buf.Get<float>();
        other = other_buf.Get<float>();
        work = work_buf.Get<float>();
        half_value = half_buf.Get<half>();
        packed = packed_buf.Get<int4b_t>();
        staging = staging_buf.Get<int32_t>();
        scale = scale_buf.Get<float>();
        mask = mask_buf.Get<int32_t>();
        token = token_buf.Get<float>();
        input_scale = input_scale_buf.Get<float>();

        maximum = maximum_buf.Get<float>();
        count = count_buf.Get<int32_t>();
        start = start_buf.Get<int32_t>();
        source = source_buf.Get<int32_t>();
        slot = slot_buf.Get<int32_t>();
        weight = weight_buf.Get<float>();
        original_weight = original_weight_buf.Get<float>();
#ifdef ROUTED_ACTIVE_LIST
        pipe.InitBuffer(active_buf, kActiveListBytes);
        active = active_buf.Get<int32_t>();
#endif
#if defined(K11220_FAST_TOK_OUT) || defined(K11220_FAST_ACT_P3)
        pipe.InitBuffer(corr_cache_buf, span * sizeof(float));
        pipe.InitBuffer(tok_out_buf, 2 * span * sizeof(bfloat16_t));
        corr_cache = corr_cache_buf.Get<float>();
        tok_out = tok_out_buf.Get<bfloat16_t>();
#endif
#ifdef K112612_OVERLAP_SH_P4
        pipe.InitBuffer(sh_red_acc_buf, 8 * 1024 * sizeof(float));
        pipe.InitBuffer(sh_red_nar_buf, 8 * 1024 * sizeof(bfloat16_t));
        sh_red_acc = sh_red_acc_buf.Get<float>();
        sh_red_nar = sh_red_nar_buf.Get<bfloat16_t>();
#endif
        Duplicate(mask, int32_t(-8192), kVectorTile);
        PipeBarrier<PIPE_V>();
    }
};

__aicore__ inline void LoadTiling(TCubeTiling& cube, __gm__ uint8_t* bytes)
{
    auto* destination = (uint32_t*)&cube;
    auto* source = (__gm__ uint32_t*)bytes;
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
}

template <bool pad_rows = false, class MatmulType>
__aicore__ inline void Project(MatmulType& mm, __gm__ int4b_t* rows,
                               __gm__ int4b_t* weight,
                               __gm__ RoutedProduct* product,
                               __gm__ ScaleStorage* weight_scale,
                               int64_t tile_columns, int64_t org_columns,
                               int64_t inner, int64_t count)
{
    const int64_t matrix_rows = pad_rows ? GateMatrixRows(count) : 2 * count;
    GlobalTensor<int4b_t> a, b;
    GlobalTensor<RoutedProduct> c;
    a.SetGlobalBuffer(rows, matrix_rows * inner);
    b.SetGlobalBuffer(weight, inner * tile_columns);
    c.SetGlobalBuffer(product, matrix_rows * org_columns);
    mm.SetOrgShape(matrix_rows, org_columns, inner);
    mm.SetSingleShape(matrix_rows, tile_columns, inner);
    mm.SetTensorA(a);
    mm.SetTensorB(b, false);
#ifdef ROUTED_FIXPIPE_HALF
    GlobalTensor<uint64_t> scale;
    scale.SetGlobalBuffer(weight_scale, tile_columns);
    mm.SetQuantVector(scale);
#endif
    mm.IterateAll(c);
}

#ifdef K11080_DIRECT_W4A8
template <int64_t kInner, int64_t kNStep, int64_t kKChunk, bool kAvoid128 = false>
struct DirectW4A8Runner16 {
    __aicore__ static inline int32_t ChooseStep(int32_t rem)
    {
        if constexpr (kAvoid128) {
            if (rem == 384 || rem == 192) {
                return 192;
            }
        }
        return (rem < kNStep) ? rem : static_cast<int32_t>(kNStep);
    }
    static constexpr int64_t kKChunks = kInner / kKChunk;
    static constexpr int64_t kSingleANibbles = 16 * kInner;
    static constexpr int64_t kSingleABytes = kSingleANibbles / 2;
    static constexpr int64_t kB1Bytes = (kNStep * kKChunk) / 2;
    static constexpr int64_t kL0BBytes = (256 * kNStep) / 2;
#ifdef K11220_MULTI_M32
    static constexpr int64_t kA1TotalBytes = (kInner == 2048) ? (4 * kSingleABytes) : (2 * kSingleABytes);
    static constexpr int64_t kL0ATotalBytes = (kInner == 6144) ? 49152 : 65536;
    static constexpr int64_t kL0AHalfNibbles = kL0ATotalBytes;
    static constexpr int64_t kL0CElems = 32 * kNStep;
#elif defined(K11220_MULTI_M16)
    static constexpr int64_t kA1TotalBytes = 2 * kSingleABytes;
    static constexpr int64_t kL0ATotalBytes = (kInner == 6144) ? 49152 : 65536;
    static constexpr int64_t kL0AHalfNibbles = kL0ATotalBytes;
    static constexpr int64_t kL0CElems = 32 * kNStep;
#else
    static constexpr int64_t kA1TotalBytes = 2 * kSingleABytes;
    static constexpr int64_t kL0ATotalBytes = (2 * kSingleABytes <= 65536) ? (2 * kSingleABytes) : kSingleABytes;
    static constexpr int64_t kL0AHalfNibbles = kSingleANibbles;
    static constexpr int64_t kL0CElems = 16 * kNStep;
#endif

    TBuf<TPosition::A1> a1_all_buf;
    TBuf<TPosition::B1> b1_buf[2];
    TBuf<TPosition::A2> l0a_all_buf;
    TBuf<TPosition::B2> l0b_buf[2];
    TBuf<TPosition::CO1> l0c_buf[2];

    LocalTensor<int4b_t> a1_all;
    LocalTensor<int4b_t> a1[2];
    LocalTensor<int4b_t> b1[2];
    LocalTensor<int4b_t> l0a_all;
    LocalTensor<int4b_t> l0b[2];
    LocalTensor<int32_t> l0c[2];

    uint32_t a_ping = 0;
    uint32_t b_ping = 0;
    uint32_t l0a_ping = 0;
    uint32_t l0b_ping = 0;
    uint32_t c_ping = 0;
    bool has_pref_b = false;
    uint32_t pref_bi = 0;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(a1_all_buf, kA1TotalBytes);
        a1_all = a1_all_buf.template Get<int4b_t>();
        a1[0] = a1_all[0];
        a1[1] = a1_all[static_cast<uint32_t>(kSingleANibbles)];
        pipe.InitBuffer(b1_buf[0], kB1Bytes);
        pipe.InitBuffer(b1_buf[1], kB1Bytes);
        b1[0] = b1_buf[0].template Get<int4b_t>();
        b1[1] = b1_buf[1].template Get<int4b_t>();
        pipe.InitBuffer(l0a_all_buf, kL0ATotalBytes);
        l0a_all = l0a_all_buf.template Get<int4b_t>();
        pipe.InitBuffer(l0b_buf[0], kL0BBytes);
        pipe.InitBuffer(l0b_buf[1], kL0BBytes);
        l0b[0] = l0b_buf[0].template Get<int4b_t>();
        l0b[1] = l0b_buf[1].template Get<int4b_t>();
        pipe.InitBuffer(l0c_buf[0], kL0CElems * sizeof(int32_t));
        pipe.InitBuffer(l0c_buf[1], kL0CElems * sizeof(int32_t));
        l0c[0] = l0c_buf[0].template Get<int32_t>();
        l0c[1] = l0c_buf[1].template Get<int32_t>();

        a_ping = 0;
        b_ping = 0;
        l0a_ping = 0;
        l0b_ping = 0;
        c_ping = 0;
        has_pref_b = false;
        pref_bi = 0;

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID6);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    }

    __aicore__ inline void End()
    {
        if (has_pref_b) {
            const event_t pe = pref_bi ? EVENT_ID1 : EVENT_ID0;
            WaitFlag<HardEvent::MTE2_MTE1>(pe);
            SetFlag<HardEvent::MTE1_MTE2>(pe);
            has_pref_b = false;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint32_t IssueAMte2(__gm__ int4b_t* rows_ptr, int64_t cnt)
    {
        const int32_t m_rows = static_cast<int32_t>(2 * cnt);
        GlobalTensor<int4b_t> next_a;
        next_a.SetGlobalBuffer(rows_ptr, static_cast<uint64_t>(m_rows) * kInner);
        const uint32_t ai = a_ping++ & 1U;
        const event_t a_evt = ai ? EVENT_ID3 : EVENT_ID2;
        WaitFlag<HardEvent::MTE1_MTE2>(a_evt);
        Nd2NzParams copy_a{};
        copy_a.ndNum = 1;
        copy_a.nValue = static_cast<uint16_t>(m_rows);
        copy_a.dValue = static_cast<uint16_t>(kInner >> 1);
        copy_a.srcDValue = static_cast<uint16_t>(kInner >> 1);
        copy_a.dstNzC0Stride = 16;
        copy_a.dstNzNStride = 1;
        DataCopy(a1[ai], next_a, copy_a);
        SetFlag<HardEvent::MTE2_MTE1>(a_evt);
        return ai;
    }

    __aicore__ inline uint32_t IssueBMte2(const GlobalTensor<int4b_t>& b_gm, int32_t n_off, int32_t cur_n, int32_t kc)
    {
        const uint32_t bi = b_ping++ & 1U;
        const event_t b_evt = bi ? EVENT_ID1 : EVENT_ID0;
        WaitFlag<HardEvent::MTE1_MTE2>(b_evt);
#ifdef K11088_GATE_CONTIG
        if constexpr (kInner == 6144 && kNStep == 256) {
            const uint16_t blocks = static_cast<uint16_t>((cur_n >> 6) * kKChunk);
            const uint64_t off = static_cast<uint64_t>(n_off) * kInner
                + static_cast<uint64_t>(kc) * static_cast<uint64_t>(cur_n) * kKChunk;
            DataCopy(b1[bi], b_gm[off], DataCopyParams{1, blocks, 0, 0});
        } else
#endif
#ifdef K11220_DOWN_CONTIG_256
        if constexpr (kInner == 2048 && kNStep == 256) {
            const uint16_t blocks = static_cast<uint16_t>((cur_n >> 6) * kKChunk);
            const uint64_t off = static_cast<uint64_t>(n_off) * kInner
                + static_cast<uint64_t>(kc) * static_cast<uint64_t>(cur_n) * kKChunk;
            DataCopy(b1[bi], b_gm[off], DataCopyParams{1, blocks, 0, 0});
        } else
#endif
        if constexpr (kKChunk == kInner) {
            const uint16_t blocks = static_cast<uint16_t>((cur_n >> 6) * kInner);
            DataCopy(b1[bi], b_gm[static_cast<uint64_t>(n_off) * kInner], DataCopyParams{1, blocks, 0, 0});
        } else {
            const uint16_t nburst = static_cast<uint16_t>(cur_n >> 6);
            const uint16_t block_len = static_cast<uint16_t>(kKChunk);
            const uint16_t src_stride = static_cast<uint16_t>(kInner - kKChunk);
            const uint64_t off = static_cast<uint64_t>(kc * kKChunk) * 64ULL + static_cast<uint64_t>(n_off) * kInner;
            DataCopy(b1[bi], b_gm[off], DataCopyParams{nburst, block_len, src_stride, 0});
        }
        SetFlag<HardEvent::MTE2_MTE1>(b_evt);
        return bi;
    }

#ifdef K11220_MULTI_M32
    __aicore__ inline void RunMultiM32(
        __gm__ int4b_t* rows, const GlobalTensor<int4b_t>& b_gm, const GlobalTensor<int32_t>& c_gm,
        int64_t tile_columns, int64_t org_columns, int32_t matrix_rows,
        __gm__ int4b_t* next_weight, int64_t next_tile_columns)
    {
        const int32_t m_tiles = (matrix_rows + 15) >> 4;
        const uint16_t m_last = static_cast<uint16_t>(matrix_rows - (m_tiles - 1) * 16);
        GlobalTensor<int4b_t> a_gm;
        a_gm.SetGlobalBuffer(rows, static_cast<uint64_t>(matrix_rows) * kInner);

        if constexpr (kInner == 2048) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            Nd2NzParams copy_a{};
            copy_a.ndNum = 1;
            copy_a.dValue = static_cast<uint16_t>(kInner >> 1);
            copy_a.srcDValue = static_cast<uint16_t>(kInner >> 1);
            copy_a.dstNzC0Stride = 16;
            copy_a.dstNzNStride = 1;
            for (int32_t mt = 0; mt < m_tiles; ++mt) {
                copy_a.nValue = (mt + 1 == m_tiles) ? m_last : 16U;
                DataCopy(a1_all[static_cast<uint32_t>(mt * kSingleANibbles)],
                         a_gm[static_cast<uint64_t>(mt) * kSingleANibbles], copy_a);
            }
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

            uint32_t cur_bi = 0U;
            if (has_pref_b) {
                cur_bi = pref_bi;
                has_pref_b = false;
            } else {
                const int32_t first_n = ChooseStep(static_cast<int32_t>(tile_columns));
                cur_bi = IssueBMte2(b_gm, 0, first_n, 0);
            }
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);
            LoadData2DParams load_a{};
            load_a.repeatTimes = static_cast<uint8_t>(m_tiles * (kInner >> 6));
            load_a.srcStride = 1;
            load_a.ifTranspose = false;
            LoadData(l0a_all, a1_all, load_a);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID5);
            bool waited_l0a = false;

            for (int32_t n_off = 0; n_off < tile_columns;) {
                const int32_t cur_n = ChooseStep(static_cast<int32_t>(tile_columns) - n_off);
                LoadData2dTransposeParams trans(0, static_cast<uint8_t>(cur_n >> 6), static_cast<uint16_t>(kKChunk >> 6), 3, 0, inc);
                const uint32_t dst_step = static_cast<uint32_t>(cur_n * 64);

                for (int32_t kc = 0; kc < kKChunks; ++kc) {
                    int32_t next_kc = kc + 1;
                    int32_t next_n_off = n_off;
                    if (next_kc == kKChunks) {
                        next_kc = 0;
                        next_n_off = n_off + cur_n;
                    }
                    uint32_t next_bi = 0;
                    if (next_n_off < tile_columns) {
                        const int32_t next_n = ChooseStep(static_cast<int32_t>(tile_columns) - next_n_off);
                        next_bi = IssueBMte2(b_gm, next_n_off, next_n, next_kc);
                    }
#ifdef K11220_XTASK_B
                    else if (next_weight != nullptr) {
                        GlobalTensor<int4b_t> nb_gm;
                        nb_gm.SetGlobalBuffer(next_weight, static_cast<uint64_t>(next_tile_columns) * kInner);
                        nb_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                        const int32_t nf_n = ChooseStep(static_cast<int32_t>(next_tile_columns));
                        pref_bi = IssueBMte2(nb_gm, 0, nf_n, 0);
                        has_pref_b = true;
                    }
#endif
                    const event_t b_evt = cur_bi ? EVENT_ID1 : EVENT_ID0;
                    WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
                    for (int32_t k_sub = 0; k_sub < kKChunk; k_sub += 256) {
                        const uint32_t lbi = l0b_ping++ & 1U;
                        const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
                        WaitFlag<HardEvent::M_MTE1>(lb_evt);
                        uint32_t l1b_off = static_cast<uint32_t>(k_sub * 64);
                        uint32_t l0b_off = 0;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            LoadDataWithTranspose(l0b[lbi][l0b_off], b1[cur_bi][l1b_off], trans);
                            l1b_off += 4096U;
                            l0b_off += dst_step;
                        }
                        if (k_sub + 256 == kKChunk) {
                            SetFlag<HardEvent::MTE1_MTE2>(b_evt);
                        }
                        SetFlag<HardEvent::MTE1_M>(lb_evt);
                        WaitFlag<HardEvent::MTE1_M>(lb_evt);
                        if (!waited_l0a) {
                            WaitFlag<HardEvent::MTE1_M>(EVENT_ID5);
                            waited_l0a = true;
                        }
                        const int32_t k_total = kc * kKChunk + k_sub;
                        if (k_total == 0) {
                            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                            WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
                        }
                        MmadParams mad{};
                        mad.m = 16;
                        mad.k = 256;
                        mad.n = static_cast<uint16_t>(cur_n);
                        mad.cmatrixSource = false;
                        mad.cmatrixInitVal = (k_total == 0);
                        Mmad(l0c[0][0], l0a_all[static_cast<uint32_t>(k_total * 16)], l0b[lbi], mad);
                        Mmad(l0c[0][static_cast<uint32_t>(16 * kNStep)],
                             l0a_all[static_cast<uint32_t>(kSingleANibbles + k_total * 16)], l0b[lbi], mad);
                        Mmad(l0c[1][0],
                             l0a_all[static_cast<uint32_t>(2 * kSingleANibbles + k_total * 16)], l0b[lbi], mad);
                        if (m_tiles == 4) {
                            Mmad(l0c[1][static_cast<uint32_t>(16 * kNStep)],
                                 l0a_all[static_cast<uint32_t>(3 * kSingleANibbles + k_total * 16)], l0b[lbi], mad);
                        }
                        const bool is_last_mmad = (n_off + cur_n >= tile_columns) && (kc + 1 == kKChunks) && (k_sub + 256 == kKChunk);
                        if (cur_n < 160 || is_last_mmad) {
                            PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(lb_evt);
                    }
                    cur_bi = next_bi;
                }
                if (n_off + cur_n >= tile_columns) {
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID6);
                }
                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
                FixpipeParamsV220 fix{};
                fix.nSize = static_cast<uint16_t>(cur_n);
                fix.mSize = 16;
                fix.srcStride = 16;
                fix.dstStride = static_cast<uint32_t>(org_columns);
                fix.ndNum = 1;
                fix.quantPre = QuantMode_t::NoQuant;
                Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[n_off], l0c[0][0], fix);
                Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                    c_gm[static_cast<uint64_t>(16 * org_columns + n_off)],
                    l0c[0][static_cast<uint32_t>(16 * kNStep)], fix);
                fix.mSize = (m_tiles == 3) ? m_last : 16U;
                Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                    c_gm[static_cast<uint64_t>(32 * org_columns + n_off)],
                    l0c[1][0], fix);
                if (m_tiles == 4) {
                    fix.mSize = m_last;
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                        c_gm[static_cast<uint64_t>(48 * org_columns + n_off)],
                        l0c[1][static_cast<uint32_t>(16 * kNStep)], fix);
                }
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                SetFlag<HardEvent::FIX_M>(EVENT_ID1);
                n_off += cur_n;
            }
        } else {
            auto issue_a_kc = [&](int32_t kc_idx) __aicore__ -> uint32_t {
                const uint32_t ai = a_ping++ & 1U;
                const event_t a_evt = ai ? EVENT_ID3 : EVENT_ID2;
                WaitFlag<HardEvent::MTE1_MTE2>(a_evt);
                Nd2NzParams copy_a{};
                copy_a.ndNum = 1;
                copy_a.dValue = static_cast<uint16_t>(kKChunk >> 1);
                copy_a.srcDValue = static_cast<uint16_t>(kInner >> 1);
                copy_a.dstNzC0Stride = 16;
                copy_a.dstNzNStride = 1;
                for (int32_t mt = 0; mt < m_tiles; ++mt) {
                    copy_a.nValue = (mt + 1 == m_tiles) ? m_last : 16U;
                    const uint64_t a_gm_off = static_cast<uint64_t>(mt) * kSingleANibbles
                        + static_cast<uint64_t>(kc_idx * kKChunk);
                    DataCopy(a1[ai][static_cast<uint32_t>(mt * 16 * kKChunk)], a_gm[a_gm_off], copy_a);
                }
                SetFlag<HardEvent::MTE2_MTE1>(a_evt);
                return ai;
            };

            uint32_t cur_ai = issue_a_kc(0);
            uint32_t cur_bi = 0U;
            if (has_pref_b) {
                cur_bi = pref_bi;
                has_pref_b = false;
            } else {
                const int32_t first_n = ChooseStep(static_cast<int32_t>(tile_columns));
                cur_bi = IssueBMte2(b_gm, 0, first_n, 0);
            }

            for (int32_t n_off = 0; n_off < tile_columns;) {
                const int32_t cur_n = ChooseStep(static_cast<int32_t>(tile_columns) - n_off);
                LoadData2dTransposeParams trans(0, static_cast<uint8_t>(cur_n >> 6), static_cast<uint16_t>(kKChunk >> 6), 3, 0, inc);
                const uint32_t dst_step = static_cast<uint32_t>(cur_n * 64);

                for (int32_t kc = 0; kc < kKChunks; ++kc) {
                    int32_t next_kc = kc + 1;
                    int32_t next_n_off = n_off;
                    if (next_kc == kKChunks) {
                        next_kc = 0;
                        next_n_off = n_off + cur_n;
                    }
                    const event_t a_evt = cur_ai ? EVENT_ID3 : EVENT_ID2;
                    WaitFlag<HardEvent::MTE2_MTE1>(a_evt);
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);
                    LoadData2DParams load_a{};
                    load_a.repeatTimes = static_cast<uint8_t>(m_tiles * (kKChunk >> 6));
                    load_a.srcStride = 1;
                    load_a.ifTranspose = false;
                    LoadData(l0a_all, a1[cur_ai], load_a);
                    SetFlag<HardEvent::MTE1_MTE2>(a_evt);
                    SetFlag<HardEvent::MTE1_M>(EVENT_ID5);
                    bool waited_kc_l0a = false;

                    uint32_t next_ai = 0U;
                    uint32_t next_bi = 0U;
                    if (next_n_off < tile_columns) {
                        const int32_t next_n = ChooseStep(static_cast<int32_t>(tile_columns) - next_n_off);
                        next_ai = issue_a_kc(next_kc);
                        next_bi = IssueBMte2(b_gm, next_n_off, next_n, next_kc);
                    }
#ifdef K11220_XTASK_B
                    else if (next_weight != nullptr) {
                        GlobalTensor<int4b_t> nb_gm;
                        nb_gm.SetGlobalBuffer(next_weight, static_cast<uint64_t>(next_tile_columns) * kInner);
                        nb_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                        const int32_t nf_n = ChooseStep(static_cast<int32_t>(next_tile_columns));
                        pref_bi = IssueBMte2(nb_gm, 0, nf_n, 0);
                        has_pref_b = true;
                    }
#endif
                    const event_t b_evt = cur_bi ? EVENT_ID1 : EVENT_ID0;
                    WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
                    for (int32_t k_sub = 0; k_sub < kKChunk; k_sub += 256) {
                        const uint32_t lbi = l0b_ping++ & 1U;
                        const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
                        WaitFlag<HardEvent::M_MTE1>(lb_evt);
                        uint32_t l1b_off = static_cast<uint32_t>(k_sub * 64);
                        uint32_t l0b_off = 0;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            LoadDataWithTranspose(l0b[lbi][l0b_off], b1[cur_bi][l1b_off], trans);
                            l1b_off += 4096U;
                            l0b_off += dst_step;
                        }
                        if (k_sub + 256 == kKChunk) {
                            SetFlag<HardEvent::MTE1_MTE2>(b_evt);
                        }
                        SetFlag<HardEvent::MTE1_M>(lb_evt);
                        WaitFlag<HardEvent::MTE1_M>(lb_evt);
                        if (!waited_kc_l0a) {
                            WaitFlag<HardEvent::MTE1_M>(EVENT_ID5);
                            waited_kc_l0a = true;
                        }
                        const int32_t k_total = kc * kKChunk + k_sub;
                        if (k_total == 0) {
                            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                            WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
                        }
                        MmadParams mad{};
                        mad.m = 16;
                        mad.k = 256;
                        mad.n = static_cast<uint16_t>(cur_n);
                        mad.cmatrixSource = false;
                        mad.cmatrixInitVal = (k_total == 0);
                        Mmad(l0c[0][0], l0a_all[static_cast<uint32_t>(k_sub * 16)], l0b[lbi], mad);
                        Mmad(l0c[0][static_cast<uint32_t>(16 * kNStep)],
                             l0a_all[static_cast<uint32_t>(16 * kKChunk + k_sub * 16)], l0b[lbi], mad);
                        Mmad(l0c[1][0],
                             l0a_all[static_cast<uint32_t>(32 * kKChunk + k_sub * 16)], l0b[lbi], mad);
                        if (m_tiles == 4) {
                            Mmad(l0c[1][static_cast<uint32_t>(16 * kNStep)],
                                 l0a_all[static_cast<uint32_t>(48 * kKChunk + k_sub * 16)], l0b[lbi], mad);
                        }
                        const bool end_kc = (k_sub + 256 == kKChunk);
                        if (cur_n < 160 || end_kc) {
                            PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(lb_evt);
                        if (end_kc) {
                            SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
                            SetFlag<HardEvent::M_MTE1>(EVENT_ID6);
                        }
                    }
                    cur_ai = next_ai;
                    cur_bi = next_bi;
                }
                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
                FixpipeParamsV220 fix{};
                fix.nSize = static_cast<uint16_t>(cur_n);
                fix.mSize = 16;
                fix.srcStride = 16;
                fix.dstStride = static_cast<uint32_t>(org_columns);
                fix.ndNum = 1;
                fix.quantPre = QuantMode_t::NoQuant;
                Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[n_off], l0c[0][0], fix);
                Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                    c_gm[static_cast<uint64_t>(16 * org_columns + n_off)],
                    l0c[0][static_cast<uint32_t>(16 * kNStep)], fix);
                fix.mSize = (m_tiles == 3) ? m_last : 16U;
                Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                    c_gm[static_cast<uint64_t>(32 * org_columns + n_off)],
                    l0c[1][0], fix);
                if (m_tiles == 4) {
                    fix.mSize = m_last;
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                        c_gm[static_cast<uint64_t>(48 * org_columns + n_off)],
                        l0c[1][static_cast<uint32_t>(16 * kNStep)], fix);
                }
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                SetFlag<HardEvent::FIX_M>(EVENT_ID1);
                n_off += cur_n;
            }
        }
    }
#endif

    __aicore__ inline void Run(
        __gm__ int4b_t* rows, __gm__ int4b_t* weight, __gm__ int32_t* product,
        int64_t tile_columns, int64_t org_columns, int64_t count,
        __gm__ int4b_t* next_weight = nullptr, int64_t next_tile_columns = 0)
    {
        (void)next_weight;
        (void)next_tile_columns;
        const int32_t matrix_rows = static_cast<int32_t>(2 * count);
        GlobalTensor<int4b_t> b_gm;
        GlobalTensor<int32_t> c_gm;
        b_gm.SetGlobalBuffer(weight, static_cast<uint64_t>(tile_columns) * kInner);
        b_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        c_gm.SetGlobalBuffer(product, static_cast<uint64_t>(matrix_rows) * org_columns);

#ifdef K11220_MULTI_M32
        if (matrix_rows > 32) {
            RunMultiM32(rows, b_gm, c_gm, tile_columns, org_columns, matrix_rows, next_weight, next_tile_columns);
            return;
        }
#endif
#ifdef K11220_MULTI_M16
        if (matrix_rows > 16) {
            GlobalTensor<int4b_t> a_gm;
            a_gm.SetGlobalBuffer(rows, static_cast<uint64_t>(matrix_rows) * kInner);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            Nd2NzParams copy_a{};
            copy_a.ndNum = 1;
            copy_a.nValue = 16;
            copy_a.dValue = static_cast<uint16_t>(kInner >> 1);
            copy_a.srcDValue = static_cast<uint16_t>(kInner >> 1);
            copy_a.dstNzC0Stride = 16;
            copy_a.dstNzNStride = 1;
            DataCopy(a1[0], a_gm[0], copy_a);
            copy_a.nValue = static_cast<uint16_t>(matrix_rows - 16);
            DataCopy(a1[1], a_gm[static_cast<uint64_t>(kSingleANibbles)], copy_a);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

            uint32_t cur_bi = 0U;
            if (has_pref_b) {
                cur_bi = pref_bi;
                has_pref_b = false;
            } else {
                const int32_t first_n = ChooseStep(static_cast<int32_t>(tile_columns));
                cur_bi = IssueBMte2(b_gm, 0, first_n, 0);
            }
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

            if constexpr (kInner == 2048) {
                const uint32_t lai = l0a_ping++ & 1U;
                const event_t la_evt = lai ? EVENT_ID6 : EVENT_ID5;
                LocalTensor<int4b_t> cur_l0a = l0a_all[static_cast<uint32_t>(lai * kL0AHalfNibbles)];
                WaitFlag<HardEvent::M_MTE1>(la_evt);
                LoadData2DParams load_a{};
                load_a.repeatTimes = static_cast<uint8_t>(2 * (kInner >> 6));
                load_a.srcStride = 1;
                load_a.ifTranspose = false;
                LoadData(cur_l0a, a1_all, load_a);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
                SetFlag<HardEvent::MTE1_M>(la_evt);
                bool waited_l0a = false;

                for (int32_t n_off = 0; n_off < tile_columns;) {
                    const int32_t cur_n = ChooseStep(static_cast<int32_t>(tile_columns) - n_off);
                    const uint32_t ci = c_ping++ & 1U;
                    const event_t c_evt = ci ? EVENT_ID1 : EVENT_ID0;
                    LoadData2dTransposeParams trans(0, static_cast<uint8_t>(cur_n >> 6), static_cast<uint16_t>(kKChunk >> 6), 3, 0, inc);
                    const uint32_t dst_step = static_cast<uint32_t>(cur_n * 64);

                    for (int32_t kc = 0; kc < kKChunks; ++kc) {
                        int32_t next_kc = kc + 1;
                        int32_t next_n_off = n_off;
                        if (next_kc == kKChunks) {
                            next_kc = 0;
                            next_n_off = n_off + cur_n;
                        }
                        uint32_t next_bi = 0;
                        if (next_n_off < tile_columns) {
                            const int32_t next_n = ChooseStep(static_cast<int32_t>(tile_columns) - next_n_off);
                            next_bi = IssueBMte2(b_gm, next_n_off, next_n, next_kc);
                        }
#ifdef K11220_XTASK_B
                        else if (next_weight != nullptr) {
                            GlobalTensor<int4b_t> nb_gm;
                            nb_gm.SetGlobalBuffer(next_weight, static_cast<uint64_t>(next_tile_columns) * kInner);
                            nb_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                            const int32_t nf_n = ChooseStep(static_cast<int32_t>(next_tile_columns));
                            pref_bi = IssueBMte2(nb_gm, 0, nf_n, 0);
                            has_pref_b = true;
                        }
#endif
                        const event_t b_evt = cur_bi ? EVENT_ID1 : EVENT_ID0;
                        WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
                        for (int32_t k_sub = 0; k_sub < kKChunk; k_sub += 256) {
                            const uint32_t lbi = l0b_ping++ & 1U;
                            const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
                            WaitFlag<HardEvent::M_MTE1>(lb_evt);
                            uint32_t l1b_off = static_cast<uint32_t>(k_sub * 64);
                            uint32_t l0b_off = 0;
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                LoadDataWithTranspose(l0b[lbi][l0b_off], b1[cur_bi][l1b_off], trans);
                                l1b_off += 4096U;
                                l0b_off += dst_step;
                            }
                            if (k_sub + 256 == kKChunk) {
                                SetFlag<HardEvent::MTE1_MTE2>(b_evt);
                            }
                            SetFlag<HardEvent::MTE1_M>(lb_evt);
                            WaitFlag<HardEvent::MTE1_M>(lb_evt);
                            if (!waited_l0a) {
                                WaitFlag<HardEvent::MTE1_M>(la_evt);
                                waited_l0a = true;
                            }
                            const int32_t k_total = kc * kKChunk + k_sub;
                            if (k_total == 0) {
                                WaitFlag<HardEvent::FIX_M>(c_evt);
                            }
                            MmadParams mad{};
                            mad.m = 16;
                            mad.k = 256;
                            mad.n = static_cast<uint16_t>(cur_n);
                            mad.cmatrixSource = false;
                            mad.cmatrixInitVal = (k_total == 0);
                            Mmad(l0c[ci][0], cur_l0a[static_cast<uint32_t>(k_total * 16)], l0b[lbi], mad);
                            Mmad(l0c[ci][static_cast<uint32_t>(16 * kNStep)],
                                 cur_l0a[static_cast<uint32_t>(kSingleANibbles + k_total * 16)], l0b[lbi], mad);
                            const bool is_last_mmad = (n_off + cur_n >= tile_columns) && (kc + 1 == kKChunks) && (k_sub + 256 == kKChunk);
                            if (cur_n < 160 || is_last_mmad) {
                                PipeBarrier<PIPE_M>();
                            }
                            SetFlag<HardEvent::M_MTE1>(lb_evt);
                        }
                        cur_bi = next_bi;
                    }
                    if (n_off + cur_n >= tile_columns) {
                        SetFlag<HardEvent::M_MTE1>(la_evt);
                    }
                    SetFlag<HardEvent::M_FIX>(c_evt);
                    WaitFlag<HardEvent::M_FIX>(c_evt);
                    FixpipeParamsV220 fix{};
                    fix.nSize = static_cast<uint16_t>(cur_n);
                    fix.mSize = 16;
                    fix.srcStride = 16;
                    fix.dstStride = static_cast<uint32_t>(org_columns);
                    fix.ndNum = 1;
                    fix.quantPre = QuantMode_t::NoQuant;
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[n_off], l0c[ci][0], fix);
                    fix.mSize = static_cast<uint16_t>(matrix_rows - 16);
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                        c_gm[static_cast<uint64_t>(16 * org_columns + n_off)],
                        l0c[ci][static_cast<uint32_t>(16 * kNStep)], fix);
                    SetFlag<HardEvent::FIX_M>(c_evt);
                    n_off += cur_n;
                }
                return;
            } else {
                for (int32_t n_off = 0; n_off < tile_columns;) {
                    const int32_t cur_n = ChooseStep(static_cast<int32_t>(tile_columns) - n_off);
                    const uint32_t ci = c_ping++ & 1U;
                    const event_t c_evt = ci ? EVENT_ID1 : EVENT_ID0;
                    LoadData2dTransposeParams trans(0, static_cast<uint8_t>(cur_n >> 6), static_cast<uint16_t>(kKChunk >> 6), 3, 0, inc);
                    const uint32_t dst_step = static_cast<uint32_t>(cur_n * 64);

                    for (int32_t kc = 0; kc < kKChunks; ++kc) {
                        int32_t next_kc = kc + 1;
                        int32_t next_n_off = n_off;
                        if (next_kc == kKChunks) {
                            next_kc = 0;
                            next_n_off = n_off + cur_n;
                        }
                        uint32_t next_bi = 0;
                        if (next_n_off < tile_columns) {
                            const int32_t next_n = ChooseStep(static_cast<int32_t>(tile_columns) - next_n_off);
                            next_bi = IssueBMte2(b_gm, next_n_off, next_n, next_kc);
                        }
#ifdef K11220_XTASK_B
                        else if (next_weight != nullptr) {
                            GlobalTensor<int4b_t> nb_gm;
                            nb_gm.SetGlobalBuffer(next_weight, static_cast<uint64_t>(next_tile_columns) * kInner);
                            nb_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                            const int32_t nf_n = ChooseStep(static_cast<int32_t>(next_tile_columns));
                            pref_bi = IssueBMte2(nb_gm, 0, nf_n, 0);
                            has_pref_b = true;
                        }
#endif

                        const uint32_t lai = l0a_ping++ & 1U;
                        const event_t la_evt = lai ? EVENT_ID6 : EVENT_ID5;
                        LocalTensor<int4b_t> cur_l0a = l0a_all[static_cast<uint32_t>(lai * kL0AHalfNibbles)];
                        WaitFlag<HardEvent::M_MTE1>(la_evt);
                        LoadData2DParams load_a{};
                        load_a.repeatTimes = static_cast<uint8_t>(kKChunk >> 6);
                        load_a.srcStride = 1;
                        load_a.ifTranspose = false;
                        const uint32_t a1_k_off = static_cast<uint32_t>(kc * kKChunk * 16);
                        LoadData(cur_l0a[0], a1[0][a1_k_off], load_a);
                        LoadData(cur_l0a[static_cast<uint32_t>(16 * kKChunk)], a1[1][a1_k_off], load_a);
                        if (n_off + cur_n >= tile_columns && kc + 1 == kKChunks) {
                            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
                            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
                        }
                        SetFlag<HardEvent::MTE1_M>(la_evt);
                        bool waited_kc_l0a = false;

                        const event_t b_evt = cur_bi ? EVENT_ID1 : EVENT_ID0;
                        WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
                        for (int32_t k_sub = 0; k_sub < kKChunk; k_sub += 256) {
                            const uint32_t lbi = l0b_ping++ & 1U;
                            const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
                            WaitFlag<HardEvent::M_MTE1>(lb_evt);
                            uint32_t l1b_off = static_cast<uint32_t>(k_sub * 64);
                            uint32_t l0b_off = 0;
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                LoadDataWithTranspose(l0b[lbi][l0b_off], b1[cur_bi][l1b_off], trans);
                                l1b_off += 4096U;
                                l0b_off += dst_step;
                            }
                            if (k_sub + 256 == kKChunk) {
                                SetFlag<HardEvent::MTE1_MTE2>(b_evt);
                            }
                            SetFlag<HardEvent::MTE1_M>(lb_evt);
                            WaitFlag<HardEvent::MTE1_M>(lb_evt);
                            if (!waited_kc_l0a) {
                                WaitFlag<HardEvent::MTE1_M>(la_evt);
                                waited_kc_l0a = true;
                            }
                            const int32_t k_total = kc * kKChunk + k_sub;
                            if (k_total == 0) {
                                WaitFlag<HardEvent::FIX_M>(c_evt);
                            }
                            MmadParams mad{};
                            mad.m = 16;
                            mad.k = 256;
                            mad.n = static_cast<uint16_t>(cur_n);
                            mad.cmatrixSource = false;
                            mad.cmatrixInitVal = (k_total == 0);
                            Mmad(l0c[ci][0], cur_l0a[static_cast<uint32_t>(k_sub * 16)], l0b[lbi], mad);
                            Mmad(l0c[ci][static_cast<uint32_t>(16 * kNStep)],
                                 cur_l0a[static_cast<uint32_t>(16 * kKChunk + k_sub * 16)], l0b[lbi], mad);
                            const bool end_kc = (k_sub + 256 == kKChunk);
                            if (cur_n < 160 || end_kc) {
                                PipeBarrier<PIPE_M>();
                            }
                            SetFlag<HardEvent::M_MTE1>(lb_evt);
                            if (end_kc) {
                                SetFlag<HardEvent::M_MTE1>(la_evt);
                            }
                        }
                        cur_bi = next_bi;
                    }
                    SetFlag<HardEvent::M_FIX>(c_evt);
                    WaitFlag<HardEvent::M_FIX>(c_evt);
                    FixpipeParamsV220 fix{};
                    fix.nSize = static_cast<uint16_t>(cur_n);
                    fix.mSize = 16;
                    fix.srcStride = 16;
                    fix.dstStride = static_cast<uint32_t>(org_columns);
                    fix.ndNum = 1;
                    fix.quantPre = QuantMode_t::NoQuant;
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[n_off], l0c[ci][0], fix);
                    fix.mSize = static_cast<uint16_t>(matrix_rows - 16);
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                        c_gm[static_cast<uint64_t>(16 * org_columns + n_off)],
                        l0c[ci][static_cast<uint32_t>(16 * kNStep)], fix);
                    SetFlag<HardEvent::FIX_M>(c_evt);
                    n_off += cur_n;
                }
                return;
            }
        }
#endif

        const uint32_t ai = IssueAMte2(rows, count);
        uint32_t cur_bi = 0U;
        bool pre_kc1 = false;
        uint32_t pre_kc1_bi = 0U;
        if (has_pref_b) {
            cur_bi = pref_bi;
            has_pref_b = false;
#ifdef K112612_DOWN_PIPE2
            if constexpr (kInner == 2048 && kKChunks == 2) {
                const int32_t fn0 = ChooseStep(static_cast<int32_t>(tile_columns));
                pre_kc1_bi = IssueBMte2(b_gm, 0, fn0, 1);
                pre_kc1 = true;
            }
#endif
        } else {
            const int32_t first_n = ChooseStep(static_cast<int32_t>(tile_columns));
            cur_bi = IssueBMte2(b_gm, 0, first_n, 0);
        }
        (void)pre_kc1;
        (void)pre_kc1_bi;
        const event_t a_evt = ai ? EVENT_ID3 : EVENT_ID2;

        uint32_t lai = 0U;
        event_t la_evt = EVENT_ID5;
        if constexpr (kInner == 6144) {
            WaitFlag<HardEvent::MTE2_MTE1>(a_evt);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);
        } else {
            lai = l0a_ping++ & 1U;
            la_evt = lai ? EVENT_ID6 : EVENT_ID5;
            WaitFlag<HardEvent::MTE2_MTE1>(a_evt);
            WaitFlag<HardEvent::M_MTE1>(la_evt);
        }
        LocalTensor<int4b_t> cur_l0a = l0a_all[static_cast<uint32_t>(lai * kL0AHalfNibbles)];
        LoadData2DParams load_a{};
        load_a.repeatTimes = static_cast<uint8_t>(kInner >> 6);
        load_a.srcStride = 1;
        load_a.ifTranspose = false;
        LoadData(cur_l0a, a1[ai], load_a);
        SetFlag<HardEvent::MTE1_MTE2>(a_evt);
        SetFlag<HardEvent::MTE1_M>(la_evt);
        bool waited_l0a = false;

        for (int32_t n_off = 0; n_off < tile_columns;) {
            const int32_t cur_n = ChooseStep(static_cast<int32_t>(tile_columns) - n_off);
            const uint32_t ci = c_ping++ & 1U;
            const event_t c_evt = ci ? EVENT_ID1 : EVENT_ID0;
            LoadData2dTransposeParams trans(0, static_cast<uint8_t>(cur_n >> 6), static_cast<uint16_t>(kKChunk >> 6), 3, 0, inc);

            for (int32_t kc = 0; kc < kKChunks; ++kc) {
                int32_t next_kc = kc + 1;
                int32_t next_n_off = n_off;
                if (next_kc == kKChunks) {
                    next_kc = 0;
                    next_n_off = n_off + cur_n;
                }
                uint32_t next_bi = 0;
#ifdef K112612_DOWN_PIPE2
                if (pre_kc1 && n_off == 0 && kc == 0) {
                    next_bi = pre_kc1_bi;
                    pre_kc1 = false;
                } else
#endif
                if (next_n_off < tile_columns) {
                    const int32_t next_n = ChooseStep(static_cast<int32_t>(tile_columns) - next_n_off);
                    next_bi = IssueBMte2(b_gm, next_n_off, next_n, next_kc);
                }
#ifdef K11220_XTASK_B
                else if (next_weight != nullptr) {
                    GlobalTensor<int4b_t> nb_gm;
                    nb_gm.SetGlobalBuffer(next_weight, static_cast<uint64_t>(next_tile_columns) * kInner);
                    nb_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                    const int32_t nf_n = ChooseStep(static_cast<int32_t>(next_tile_columns));
                    pref_bi = IssueBMte2(nb_gm, 0, nf_n, 0);
                    has_pref_b = true;
                }
#endif

                const event_t b_evt = cur_bi ? EVENT_ID1 : EVENT_ID0;
                WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
                const uint32_t dst_step = static_cast<uint32_t>(cur_n * 64);

                for (int32_t k_sub = 0; k_sub < kKChunk; k_sub += 256) {
                    const uint32_t lbi = l0b_ping++ & 1U;
                    const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
                    WaitFlag<HardEvent::M_MTE1>(lb_evt);
                    uint32_t l1b_off = static_cast<uint32_t>(k_sub * 64);
                    uint32_t l0b_off = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        LoadDataWithTranspose(l0b[lbi][l0b_off], b1[cur_bi][l1b_off], trans);
                        l1b_off += 4096U;
                        l0b_off += dst_step;
                    }
                    if (k_sub + 256 == kKChunk) {
                        SetFlag<HardEvent::MTE1_MTE2>(b_evt);
                    }
                    SetFlag<HardEvent::MTE1_M>(lb_evt);
                    WaitFlag<HardEvent::MTE1_M>(lb_evt);
                    if (!waited_l0a) {
                        WaitFlag<HardEvent::MTE1_M>(la_evt);
                        waited_l0a = true;
                    }
                    const int32_t k_total = kc * kKChunk + k_sub;
                    if (k_total == 0) {
                        WaitFlag<HardEvent::FIX_M>(c_evt);
                    }
                    MmadParams mad{};
                    mad.m = 16;
                    mad.k = 256;
                    mad.n = static_cast<uint16_t>(cur_n);
                    mad.cmatrixSource = false;
                    mad.cmatrixInitVal = (k_total == 0);
                    Mmad(l0c[ci], cur_l0a[static_cast<uint32_t>(k_total * 16)], l0b[lbi], mad);
                    const bool is_last_mmad = (n_off + cur_n >= tile_columns) && (kc + 1 == kKChunks) && (k_sub + 256 == kKChunk);
                    if (cur_n < 160 || is_last_mmad) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(lb_evt);
                }
                cur_bi = next_bi;
            }

            if (n_off + cur_n >= tile_columns) {
                if constexpr (kInner == 6144) {
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID6);
                } else {
                    SetFlag<HardEvent::M_MTE1>(la_evt);
                }
            }
            SetFlag<HardEvent::M_FIX>(c_evt);
            WaitFlag<HardEvent::M_FIX>(c_evt);
            FixpipeParamsV220 fix{};
            fix.nSize = static_cast<uint16_t>(cur_n);
            fix.mSize = static_cast<uint16_t>(matrix_rows);
            fix.srcStride = 16;
            fix.dstStride = static_cast<uint32_t>(org_columns);
            fix.ndNum = 1;
            fix.quantPre = QuantMode_t::NoQuant;
            Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[n_off], l0c[ci], fix);
            SetFlag<HardEvent::FIX_M>(c_evt);
            n_off += cur_n;
        }
    }
};

template <bool pad_rows = false, int64_t kInner, int64_t kNStep, int64_t kKChunk, bool kAvoid128>
__aicore__ inline void Project(DirectW4A8Runner16<kInner, kNStep, kKChunk, kAvoid128>& runner,
                               __gm__ int4b_t* rows,
                               __gm__ int4b_t* weight,
                               __gm__ RoutedProduct* product,
                               __gm__ ScaleStorage* weight_scale,
                               int64_t tile_columns, int64_t org_columns,
                               int64_t inner, int64_t count,
                               __gm__ int4b_t* next_weight = nullptr,
                               int64_t next_tile_columns = 0)
{
    static_assert(!pad_rows, "DirectW4A8Runner16 requires compact rows");
    (void)weight_scale;
    (void)inner;
    runner.Run(rows, weight, product, tile_columns, org_columns, count, next_weight, next_tile_columns);
}
#endif

template <bool pad_rows = false, class MatmulType>
__aicore__ inline void ProjectGroup(MatmulType& mm, __gm__ int4b_t* rows,
                                    __gm__ int4b_t* weight_base,
                                    __gm__ RoutedProduct* product_base,
                                    int64_t base_tile, int64_t sub_tiles,
                                    int64_t tile_columns, int64_t org_columns,
                                    int64_t inner, int64_t count)
{
    const int64_t matrix_rows = pad_rows ? GateMatrixRows(count) : 2 * count;
    GlobalTensor<int4b_t> a, b;
    GlobalTensor<RoutedProduct> c;
    a.SetGlobalBuffer(rows, matrix_rows * inner);
    mm.SetOrgShape(matrix_rows, org_columns, inner);
    mm.SetSingleShape(matrix_rows, tile_columns, inner);
    mm.SetTensorA(a);
    for (int64_t s = 0; s < sub_tiles; ++s) {
        const int64_t col_offset = (base_tile + s) * tile_columns;
        b.SetGlobalBuffer(Nibbles(weight_base, col_offset * inner), inner * tile_columns);
        c.SetGlobalBuffer(product_base + col_offset, matrix_rows * org_columns);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
    }
}

__aicore__ inline void SplitHalves(Workspace& held, int64_t span)
{
    Muls(held.other, held.value, 1.0F / 16.0F, span);
    PipeBarrier<PIPE_V>();
    Floor(held.other, held.other, span);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.other, -16.0F, span);
    PipeBarrier<PIPE_V>();
    Add(held.work, held.work, held.value, span);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, -8.0F, span);
    PipeBarrier<PIPE_V>();
    DataCopy(held.value, held.other, span);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Emit(Workspace& held, LocalTensor<float> values,
                            GlobalTensor<int4b_t>& destination,
                            int64_t offset, int64_t count)
{
    Cast(held.half_value, values, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(held.packed, held.half_value, RoundMode::CAST_RINT, count);
    Fence<HardEvent::V_MTE3>();
    DataCopyPad(destination[offset], held.packed,
                DataCopyExtParams{1, uint32_t(count / 2), 0, 0, 0});
    Fence<HardEvent::MTE3_V>();
}
#ifdef K11090_FAST_PREP_SPLIT
__aicore__ inline void EmitAligned(Workspace& held, LocalTensor<float> values,
                                   GlobalTensor<int8_t>& destination,
                                   int64_t byte_offset, int64_t count)
{
    Cast(held.half_value, values, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(held.packed, held.half_value, RoundMode::CAST_RINT, count);
    Fence<HardEvent::V_MTE3>();
    DataCopy(destination[byte_offset], held.packed.ReinterpretCast<int8_t>(), count >> 1);
    Fence<HardEvent::MTE3_V>();
}
#endif

template <bool balanced = false>
__aicore__ inline void Split(Workspace& held, __gm__ int8_t* rows,
                             __gm__ int4b_t* stacked, int64_t count,
                             int64_t start, int64_t tokens,
                             int64_t first_row = 0, int64_t row_step = 1)
{
    GlobalTensor<int8_t> input;
    GlobalTensor<int4b_t> high, low;
    input.SetGlobalBuffer(rows, tokens * kHidden);
    high.SetGlobalBuffer(stacked, count * kHidden);
    low.SetGlobalBuffer(Nibbles(stacked, count * kHidden), count * kHidden);
    for (int64_t row = balanced ? first_row : 0; row < count;
         row += balanced ? row_step : 1) {
        const int64_t source = held.source.GetValue(start + row);
        for (int64_t column = 0; column < kHidden; column += kVectorTile) {
            Fence<HardEvent::V_MTE2>();
            DataCopy(held.input, input[source * kHidden + column],
                     kVectorTile);
            Fence<HardEvent::MTE2_V>();
            Cast(held.half_value, held.input, RoundMode::CAST_NONE, kVectorTile);
            PipeBarrier<PIPE_V>();
            Cast(held.value, held.half_value, RoundMode::CAST_NONE, kVectorTile);
            PipeBarrier<PIPE_V>();
            SplitHalves(held, kVectorTile);
            Emit(held, held.value, high, row * kHidden + column, kVectorTile);
            Emit(held, held.work, low, row * kHidden + column, kVectorTile);
        }
    }
}

__aicore__ inline void LoadScale(Workspace& held,
                                 GlobalTensor<float>& scales,
                                 int64_t offset, int64_t count)
{
    Fence<HardEvent::V_MTE2>();
    DataCopy(held.scale, scales[offset], count);
    Fence<HardEvent::MTE2_V>();

    auto bits = held.scale.ReinterpretCast<uint16_t>();
    auto mask = held.mask.ReinterpretCast<uint16_t>();
    And(bits, bits, mask, 2 * count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ScaledPartial(Workspace& held,
                                     LocalTensor<float> destination,
                                     GlobalTensor<RoutedProduct>& product,
                                     int64_t offset, int64_t count)
{
    Fence<HardEvent::V_MTE2>();
#ifdef ROUTED_FIXPIPE_HALF
    DataCopy(held.half_value, product[offset], count);
    Fence<HardEvent::MTE2_V>();
    Cast(destination, held.half_value, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
#else
    DataCopy(held.staging, product[offset], count);
    Fence<HardEvent::MTE2_V>();
    Cast(destination, held.staging, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Mul(destination, destination, held.scale, count);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, destination, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(destination, held.half_value, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
#endif
}

__aicore__ inline void Recombine(Workspace& held,
                                 GlobalTensor<RoutedProduct>& product,
                                 GlobalTensor<float>& correction,
                                 int64_t row, int64_t rows, int64_t columns,
                                 int64_t offset, int64_t count,
                                 GlobalTensor<float>* opt_scales = nullptr)
{
    Fence<HardEvent::V_MTE2>();
    if (opt_scales != nullptr) {
        DataCopy(held.scale, (*opt_scales)[offset], count);
    }
    DataCopy(held.staging, product[row * columns + offset], count);
    Fence<HardEvent::MTE2_V>();
    if (opt_scales != nullptr) {
        auto bits = held.scale.ReinterpretCast<uint16_t>();
        auto mask = held.mask.ReinterpretCast<uint16_t>();
        And(bits, bits, mask, 2 * count);
        PipeBarrier<PIPE_V>();
    }
    Cast(held.value, held.staging, RoundMode::CAST_NONE, count);
    Fence<HardEvent::V_MTE2>();
    DataCopy(held.staging, product[(rows + row) * columns + offset], count);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.value, held.scale, count);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, held.value, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(held.value, held.half_value, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Muls(held.value, held.value, 16.0F, count);
    Fence<HardEvent::MTE2_V>();
    Cast(held.work, held.staging, RoundMode::CAST_NONE, count);
    Fence<HardEvent::V_MTE2>();
    auto corr_staging = held.staging.ReinterpretCast<float>();
    DataCopy(corr_staging, correction[offset], count);
    PipeBarrier<PIPE_V>();
    Mul(held.work, held.work, held.scale, count);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, held.work, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(held.work, held.half_value, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.work, count);
    Fence<HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, corr_staging, count);
    PipeBarrier<PIPE_V>();
}

#ifdef K11220_FAST_TOK_OUT
__aicore__ inline void RecombineCached(
    Workspace& held,
    GlobalTensor<RoutedProduct>& product,
    GlobalTensor<float>& correction,
    int64_t row, int64_t rows, int64_t columns,
    int64_t offset, int64_t count,
    GlobalTensor<float>* opt_scales)
{
    auto low_staging = held.other.ReinterpretCast<int32_t>();
    Fence<HardEvent::V_MTE2>();
    if (opt_scales != nullptr) {
        DataCopy(held.scale, (*opt_scales)[offset], count);
        DataCopy(held.corr_cache, correction[offset], count);
    }
    DataCopy(held.staging, product[row * columns + offset], count);
    DataCopy(low_staging, product[(rows + row) * columns + offset], count);
    Fence<HardEvent::MTE2_V>();
    if (opt_scales != nullptr) {
        auto bits = held.scale.ReinterpretCast<uint16_t>();
        auto mask = held.mask.ReinterpretCast<uint16_t>();
        And(bits, bits, mask, 2 * count);
        PipeBarrier<PIPE_V>();
    }
    Cast(held.value, held.staging, RoundMode::CAST_NONE, count);
    Cast(held.work, low_staging, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.value, held.scale, count);
    Mul(held.work, held.work, held.scale, count);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, held.value, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(held.value, held.half_value, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Muls(held.value, held.value, 16.0F, count);
    Cast(held.half_value, held.work, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(held.work, held.half_value, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.work, count);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.corr_cache, count);
    PipeBarrier<PIPE_V>();
}
#endif

__aicore__ inline void SwiGlu(Workspace& held)
{
    DataCopy(held.other, held.value[kIntermediate], kIntermediate);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.value, -1.0F, kIntermediate);
    PipeBarrier<PIPE_V>();
    Exp(held.work, held.work, kIntermediate);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, 1.0F, kIntermediate);
    PipeBarrier<PIPE_V>();
    Div(held.value, held.value, held.work, kIntermediate);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.value, held.other, kIntermediate);
    PipeBarrier<PIPE_V>();
}

template <bool balanced = false>
__aicore__ inline void Activate(
    Workspace& held, int64_t count, int64_t start, __gm__ RoutedProduct* product,
    __gm__ float* correction, __gm__ ScaleStorage* scales,
    __gm__ int4b_t* middle, __gm__ float* middle_scales,
    int64_t first_row = 0, int64_t row_step = 1)
{
    GlobalTensor<RoutedProduct> dots;
    GlobalTensor<float> bias, weight_scales, output_scales;
    GlobalTensor<int4b_t> high, low;
    dots.SetGlobalBuffer(product, 2 * count * kGateUp);
    bias.SetGlobalBuffer(correction, kGateUp);
#ifndef ROUTED_FIXPIPE_HALF
    weight_scales.SetGlobalBuffer(scales, kGateUp);
#endif
    output_scales.SetGlobalBuffer(middle_scales, count);
    high.SetGlobalBuffer(middle, count * kIntermediate);
    low.SetGlobalBuffer(Nibbles(middle, count * kIntermediate),
                        count * kIntermediate);
#ifdef K11090_FAST_PREP_SPLIT
    GlobalTensor<int8_t> high_i8, low_i8;
    high_i8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(middle), (count * kIntermediate) >> 1);
    low_i8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(Nibbles(middle, count * kIntermediate)),
                           (count * kIntermediate) >> 1);
#endif
    for (int64_t row = balanced ? first_row : 0; row < count;
         row += balanced ? row_step : 1) {
        const int64_t source = held.source.GetValue(start + row);
        const float activation_scale = held.input_scale.GetValue(source);
        Fence<HardEvent::S_V>();
        Recombine(held, dots, bias, row, count, kGateUp, 0, kIntermediate, &weight_scales);
        Muls(held.value, held.value, activation_scale, kIntermediate);
        PipeBarrier<PIPE_V>();
        Muls(held.work, held.value, -1.0F, kIntermediate);
        PipeBarrier<PIPE_V>();
        Exp(held.work, held.work, kIntermediate);
        PipeBarrier<PIPE_V>();
        Adds(held.work, held.work, 1.0F, kIntermediate);
        PipeBarrier<PIPE_V>();
        Div(held.other, held.value, held.work, kIntermediate);
        PipeBarrier<PIPE_V>();
        Recombine(held, dots, bias, row, count, kGateUp, kIntermediate, kIntermediate, &weight_scales);
        Muls(held.value, held.value, activation_scale, kIntermediate);
        PipeBarrier<PIPE_V>();
        Mul(held.value, held.other, held.value, kIntermediate);
        PipeBarrier<PIPE_V>();
        Abs(held.work, held.value, kIntermediate);
        PipeBarrier<PIPE_V>();
        auto reduce_work = held.staging.ReinterpretCast<float>();
        ReduceMax(held.maximum[row * kBlockFloats], held.work, reduce_work,
                  kIntermediate, false);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
        const float maximum = held.maximum.GetValue(row * kBlockFloats);
        const float quantum = maximum / 127.0F;
        const float factor = maximum == 0.0F ? 0.0F : 1.0F / quantum;
        Fence<HardEvent::S_V>();
        Muls(held.value, held.value, factor, kIntermediate);
        PipeBarrier<PIPE_V>();
        Mins(held.value, held.value, 127.0F, kIntermediate);
        PipeBarrier<PIPE_V>();
        Maxs(held.value, held.value, -127.0F, kIntermediate);
        PipeBarrier<PIPE_V>();


        Cast(held.half_value, held.value, RoundMode::CAST_RINT, kIntermediate);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.half_value, RoundMode::CAST_NONE, kIntermediate);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.value, RoundMode::CAST_RINT, kIntermediate);
        PipeBarrier<PIPE_V>();
        SplitHalves(held, kIntermediate);
#ifdef K11090_FAST_PREP_SPLIT
        EmitAligned(held, held.value, high_i8, (row * kIntermediate) >> 1, kIntermediate);
        EmitAligned(held, held.work, low_i8, (row * kIntermediate) >> 1, kIntermediate);
#else
        Emit(held, held.value, high, row * kIntermediate, kIntermediate);
        Emit(held, held.work, low, row * kIntermediate, kIntermediate);
#endif
        if constexpr (balanced) {


            held.token.SetValue(0, quantum);
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(output_scales[row], held.token,
                DataCopyExtParams{1, sizeof(float), 0, 0, 0});
            Fence<HardEvent::MTE3_S>();
        } else {
            held.token.SetValue(row, quantum);
        }
    }
    if constexpr (!balanced) {
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(output_scales, held.token,
                    DataCopyExtParams{1, uint32_t(count * sizeof(float)), 0, 0, 0});
        Fence<HardEvent::MTE3_S>();
    }
}

#ifdef K11220_FAST_ACT_P3
__aicore__ inline void ActivateRowFast(
    Workspace& held, int64_t count, int64_t row, int64_t src_row,
    __gm__ RoutedProduct* product, __gm__ float* correction,
    __gm__ ScaleStorage* scales, __gm__ int4b_t* middle,
    __gm__ float* middle_scales, bool is_last)
{
    GlobalTensor<RoutedProduct> dots;
    GlobalTensor<float> bias, weight_scales, output_scales;
    GlobalTensor<int8_t> high_i8, low_i8;
    dots.SetGlobalBuffer(product, 2 * count * kGateUp);
    bias.SetGlobalBuffer(correction, kGateUp);
    weight_scales.SetGlobalBuffer(scales, kGateUp);
    output_scales.SetGlobalBuffer(middle_scales, count);
    high_i8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(middle), (count * kIntermediate) >> 1);
    low_i8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(Nibbles(middle, count * kIntermediate)),
                           (count * kIntermediate) >> 1);
    const float activation_scale = held.input_scale.GetValue(src_row);
    Fence<HardEvent::S_V>();
    auto low_buf = held.tok_out.ReinterpretCast<int32_t>();
    auto low_packed = held.input.ReinterpretCast<int4b_t>();
    auto bits = held.scale.ReinterpretCast<uint16_t>();
    auto mask = held.mask.ReinterpretCast<uint16_t>();

    Fence<HardEvent::V_MTE2>();
    DataCopy(held.scale, weight_scales[0], kIntermediate);
    DataCopy(held.staging, dots[row * kGateUp], kIntermediate);
    DataCopy(low_buf, dots[(count + row) * kGateUp], kIntermediate);
    DataCopy(held.corr_cache, bias[0], kIntermediate);
    Fence<HardEvent::MTE2_V>();

    And(bits, bits, mask, 2 * kIntermediate);
    Cast(held.value, held.staging, RoundMode::CAST_NONE, kIntermediate);
    Cast(held.work, low_buf, RoundMode::CAST_NONE, kIntermediate);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.value, held.scale, kIntermediate);
    Mul(held.work, held.work, held.scale, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, held.value, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.value, held.half_value, RoundMode::CAST_NONE, kIntermediate);
    PipeBarrier<PIPE_V>();
    Muls(held.value, held.value, 16.0F, kIntermediate);
    Cast(held.half_value, held.work, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.work, held.half_value, RoundMode::CAST_NONE, kIntermediate);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.work, kIntermediate);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.corr_cache, kIntermediate);

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    DataCopy(held.scale, weight_scales[kIntermediate], kIntermediate);
    DataCopy(held.staging, dots[row * kGateUp + kIntermediate], kIntermediate);
    DataCopy(low_buf, dots[(count + row) * kGateUp + kIntermediate], kIntermediate);
    DataCopy(held.corr_cache, bias[kIntermediate], kIntermediate);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

    PipeBarrier<PIPE_V>();
    Muls(held.value, held.value, activation_scale, kIntermediate);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.value, -1.0F, kIntermediate);
    PipeBarrier<PIPE_V>();
    Exp(held.work, held.work, kIntermediate);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, 1.0F, kIntermediate);
    PipeBarrier<PIPE_V>();
    Div(held.other, held.value, held.work, kIntermediate);

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    PipeBarrier<PIPE_V>();
    And(bits, bits, mask, 2 * kIntermediate);
    Cast(held.value, held.staging, RoundMode::CAST_NONE, kIntermediate);
    Cast(held.work, low_buf, RoundMode::CAST_NONE, kIntermediate);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.value, held.scale, kIntermediate);
    Mul(held.work, held.work, held.scale, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, held.value, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.value, held.half_value, RoundMode::CAST_NONE, kIntermediate);
    PipeBarrier<PIPE_V>();
    Muls(held.value, held.value, 16.0F, kIntermediate);
    Cast(held.half_value, held.work, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.work, held.half_value, RoundMode::CAST_NONE, kIntermediate);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.work, kIntermediate);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.corr_cache, kIntermediate);
    PipeBarrier<PIPE_V>();
    Muls(held.value, held.value, activation_scale, kIntermediate);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.other, held.value, kIntermediate);
    PipeBarrier<PIPE_V>();
    Abs(held.work, held.value, kIntermediate);
    PipeBarrier<PIPE_V>();
    auto reduce_work = held.staging.ReinterpretCast<float>();
    ReduceMax(held.maximum[row * kBlockFloats], held.work, reduce_work,
              kIntermediate, false);
    PipeBarrier<PIPE_V>();
    Fence<HardEvent::V_S>();
    const float maximum = held.maximum.GetValue(row * kBlockFloats);
    const float quantum = maximum / 127.0F;
    const float factor = maximum == 0.0F ? 0.0F : 1.0F / quantum;
    held.token.SetValue(0, quantum);
    Fence<HardEvent::S_V>();
    Muls(held.value, held.value, factor, kIntermediate);
    PipeBarrier<PIPE_V>();
    Mins(held.value, held.value, 127.0F, kIntermediate);
    PipeBarrier<PIPE_V>();
    Maxs(held.value, held.value, -127.0F, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, held.value, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.value, held.half_value, RoundMode::CAST_NONE, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.value, held.value, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    SplitHalves(held, kIntermediate);
    Cast(held.half_value, held.value, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.packed, held.half_value, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(held.half_value, held.work, RoundMode::CAST_RINT, kIntermediate);
    PipeBarrier<PIPE_V>();
    Cast(low_packed, held.half_value, RoundMode::CAST_RINT, kIntermediate);
    Fence<HardEvent::V_MTE3>();
    Fence<HardEvent::S_MTE3>();
    DataCopy(high_i8[(row * kIntermediate) >> 1], held.packed.ReinterpretCast<int8_t>(), kIntermediate >> 1);
    DataCopy(low_i8[(row * kIntermediate) >> 1], low_packed.ReinterpretCast<int8_t>(), kIntermediate >> 1);
    DataCopyPad(output_scales[row], held.token, DataCopyExtParams{1, sizeof(float), 0, 0, 0});
    if (!is_last) {
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::MTE3_S>();
    }
}
#endif

template <bool balanced = false>
__aicore__ inline void StoreRoutes(
    Workspace& held, int64_t count, int64_t start, __gm__ RoutedProduct* product,
    __gm__ float* correction, __gm__ ScaleStorage* scales,
    __gm__ float* middle_scales, __gm__ bfloat16_t* routes,
    int64_t begin, int64_t end, int64_t tokens,
    int64_t first_row = 0, int64_t row_step = 1)
{
    GlobalTensor<RoutedProduct> dots;
    GlobalTensor<float> bias, weight_scales, activation_scales;
    GlobalTensor<bfloat16_t> output;
    dots.SetGlobalBuffer(product, 2 * count * kHidden);
    bias.SetGlobalBuffer(correction, kHidden);
#ifndef ROUTED_FIXPIPE_HALF
    weight_scales.SetGlobalBuffer(scales, kHidden);
#endif
    activation_scales.SetGlobalBuffer(middle_scales, count);
    output.SetGlobalBuffer(routes, tokens * kTopK * kHidden);
    Fence<HardEvent::S_MTE2>();
    DataCopyPad(held.token, activation_scales,
                DataCopyExtParams{1, uint32_t(count * sizeof(float)), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    Fence<HardEvent::MTE2_S>();
    auto narrowed = held.half_buf.Get<bfloat16_t>();
    for (int64_t column = begin; column < end; column += kVectorTile) {
        const int64_t span = end - column < kVectorTile
            ? end - column : kVectorTile;
#ifndef ROUTED_FIXPIPE_HALF
        LoadScale(held, weight_scales, column, span);
#endif
        for (int64_t row = balanced ? first_row : 0; row < count;
             row += balanced ? row_step : 1) {
            Recombine(held, dots, bias, row, count, kHidden, column, span);
            Muls(held.value, held.value, held.token.GetValue(row), span);
            PipeBarrier<PIPE_V>();
            Cast(narrowed, held.value, RoundMode::CAST_RINT, span);
            const int64_t source = held.source.GetValue(start + row);
            const int64_t slot = held.slot.GetValue(start + row);
            const int64_t offset = (source * kTopK + slot) * kHidden + column;
            Fence<HardEvent::V_MTE3>();
            DataCopy(output[offset], narrowed, span);
            Fence<HardEvent::MTE3_V>();
        }
    }
}

__aicore__ inline void CombineTokens(Workspace& held,
                                      __gm__ bfloat16_t* route_output,
                                      __gm__ bfloat16_t* out,
                                      int64_t tokens, int64_t blocks)
{
    GlobalTensor<bfloat16_t> routes, output;
    routes.SetGlobalBuffer(route_output, tokens * kTopK * kHidden);
    output.SetGlobalBuffer(out, tokens * kHidden);
    auto values = held.half_buf.Get<bfloat16_t>();
    const int64_t tiles_per_token = kHidden / kVectorTile;
    const int64_t vector_cores = 2 * blocks;
    for (int64_t tile = GetBlockIdx(); tile < tokens * tiles_per_token;
         tile += vector_cores) {
        const int64_t token = tile / tiles_per_token;
        const int64_t column = (tile % tiles_per_token) * kVectorTile;
        Duplicate(held.value, 0.0F, kVectorTile);
        PipeBarrier<PIPE_V>();
        for (int64_t slot = 0; slot < kTopK; ++slot) {
            const int64_t original = token * kTopK + slot;
            const float probability = held.original_weight.GetValue(original);

            if (probability == 0.0F) {
                continue;
            }
            Fence<HardEvent::V_MTE2>();
            DataCopy(values, routes[original * kHidden + column], kVectorTile);
            Fence<HardEvent::MTE2_V>();


            auto term = slot == 0 ? held.value : held.other;
            Cast(term, values, RoundMode::CAST_NONE, kVectorTile);
            PipeBarrier<PIPE_V>();
            Muls(term, term, probability, kVectorTile);
            PipeBarrier<PIPE_V>();
            if (slot != 0) {
                Add(held.value, held.value, term, kVectorTile);
                PipeBarrier<PIPE_V>();
            }
        }
        Cast(values, held.value, RoundMode::CAST_RINT, kVectorTile);
        Fence<HardEvent::V_MTE3>();
        DataCopy(output[token * kHidden + column], values, kVectorTile);
        Fence<HardEvent::MTE3_V>();
    }
}

#include "../../../source/output_native.h"
#include "../../../source/split_native.h"

}

template <CubeFormat weight_format, bool from_dispatch>
#ifdef K11240_OUTLINE_PHASES
__aicore__ __attribute__((noinline)) inline void Forward(
#else
__aicore__ inline void Forward(
#endif
    TPipe& pipe, const ffn_original_shared_gu_static::BodyArguments& arguments,
    __gm__ bfloat16_t* scaled_output, __gm__ bfloat16_t* combined_output,
    int32_t passed_expert_offset = -1, bool record_prof = false, bool combine_shared = true,
    GM_ADDR topk_logits = nullptr, GM_ADDR topk_bias = nullptr, GM_ADDR topk_sys = nullptr,
    __gm__ uint8_t* comm_window = nullptr, int64_t comm_capacity = 0, int32_t comm_rank = 0, int32_t comm_ranks = 0,
    uint64_t layer_pw = 0, int64_t* moe_prof = nullptr)
{
    Timing timing;
    int64_t mp0 = 0, mp1 = 0, mp2 = 0, mp3 = 0, mp4 = 0, mp5 = 0;
    if (moe_prof != nullptr) { mp0 = GetSystemCycle(); }
    timing.Begin(0);
#ifdef ROUTED_SHARED_PIPELINE
    static_assert(from_dispatch, "the shared pipeline requires input preparation");
    pipeline::Stages shared(arguments.shared, arguments.rows, arguments.blocks);
#endif
    int32_t expert_offset = passed_expert_offset;
    if (expert_offset < 0) { GlobalTensor<int32_t> m; m.SetGlobalBuffer((__gm__ int32_t*)arguments.scratch[kGateTiling], 128); expert_offset = m.GetValue(120); }
    timing.Begin(1);
    Workspace held;
    uint64_t my_gexp_u64[4] = {0ULL, 0ULL, 0ULL, 0ULL};
    uint8_t* my_gexp = reinterpret_cast<uint8_t*>(my_gexp_u64);
    uint64_t mw_s[6] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
    uint32_t local_origs_s[kMaxRoutes];
    int32_t local_cnt_s = -1;
    uint32_t act_pack_s[kMaxRoutes];
    int32_t act_total_s = -1;
    bool pipe_wave0_s = false;
    int32_t split_act_pos_s = 0;
    int32_t split_local_cnt_s = 0;
    int32_t split_ord6_exp_s = 0;
#if defined(K11180_LITE_MIRROR) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
    const auto* lmeta = reinterpret_cast<const __gm__ whole_layer_tree::LiteMirrorMeta*>(layer_pw);
    const bool has_mirror = (lmeta != nullptr);
    const uint64_t* mw_ptr = has_mirror ? mw_s : nullptr;
    const bool pw_active = has_mirror;
#elif defined(K11220_FAST_MIRROR_V4) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
    const auto* mmeta = reinterpret_cast<const __gm__ whole_layer_tree::LayerMirrorMeta*>(layer_pw);
    const bool has_mirror = (mmeta != nullptr);
    const uint64_t* mw_ptr = has_mirror ? mw_s : nullptr;
    const bool pw_active = has_mirror;
#else
    const bool has_mirror = false;
    const uint64_t* mw_ptr = nullptr;
    const bool pw_active = false;
    (void)layer_pw;
#endif
    uint32_t saved_local_mask = 0U;
    uint32_t comm_base_epoch = 0U;
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
    if ASCEND_IS_AIV {
        if (comm_window != nullptr) {
            comm_base_epoch = whole_layer_tree::ReadCommunicationEpoch(comm_window);
        }
    }
#endif
#ifdef ROUTED_SHARED_PIPELINE
    shared.InitResident(pipe);
    shared.Gate(pipe);
#endif
    GlobalTensor<int32_t> counts, starts;
#ifdef ROUTED_ACTIVE_LIST
    GlobalTensor<int32_t> active_experts;
    active_experts.SetGlobalBuffer((__gm__ int32_t*)arguments.dispatch.active_list,
                                   kActiveListElements);
    int64_t active_count = 0;
#endif
    if ASCEND_IS_AIV {
        held.Init(pipe);
        const int64_t routes = arguments.rows * kTopK;
        auto choice = held.slot;
        auto inverse = held.weight.ReinterpretCast<int32_t>();
        auto cursor = held.staging;
        auto active_tokens = held.staging[2048];
        auto probability = held.value;
        auto narrow = held.half_buf.Get<bfloat16_t>();
        GlobalTensor<int32_t> choices;
        GlobalTensor<float> probabilities, input_scales;
        choices.SetGlobalBuffer((__gm__ int32_t*)arguments.dispatch.expert_ids, routes);
        probabilities.SetGlobalBuffer((__gm__ float*)arguments.dispatch.probabilities, routes);
        input_scales.SetGlobalBuffer(((__gm__ float*)arguments.scratch[kRestoredScale]), arguments.rows);
        auto mmeta_ub = held.other.ReinterpretCast<uint8_t>();
        uint64_t g2e_u64[32];
        int8_t* gexp_to_eidx = reinterpret_cast<int8_t*>(g2e_u64);
        uint64_t own_u64[32];
        int32_t raw_r[kMaxRoutes];
        uint64_t prob_nz[4] = {0ULL, 0ULL, 0ULL, 0ULL};
        (void)mmeta_ub;
        (void)own_u64;
#ifdef K11220_SINGLE_OWN_P1
        const bool fast_no_comb = !combine_shared;
        const bool defer_tid0_prob = (has_mirror && GetBlockIdx() == 0 && fast_no_comb);
        DataCopy(choice, choices, routes);
        if (!defer_tid0_prob) {
            DataCopy(probability, probabilities, routes);
            DataCopyPad(held.input_scale, input_scales,
                        DataCopyExtParams{1, uint32_t(arguments.rows * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
#if defined(K11220_FAST_MIRROR_V4) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (has_mirror && GetBlockIdx() == 0) {
            GlobalTensor<uint32_t> mm_gm;
            mm_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ whole_layer_tree::LayerMirrorMeta*>(mmeta)), 32);
            DataCopy(held.other.ReinterpretCast<uint32_t>(), mm_gm, 32);
        }
#endif
        if (!fast_no_comb) {
            Duplicate(held.original_weight, 0.0F, kMaxRoutes);
            Fence<HardEvent::MTE2_V>();
            Fence<HardEvent::V_S>();
        }
        Fence<HardEvent::MTE2_S>();
        {
            auto choice64 = choice.ReinterpretCast<uint64_t>();
            const int64_t rpairs = routes >> 1;
            for (int64_t p = 0; p < rpairs; ++p) {
                const uint64_t v64 = choice64.GetValue(p);
                raw_r[2 * p] = static_cast<int32_t>(v64 & 0xFFFFFFFFULL);
                raw_r[2 * p + 1] = static_cast<int32_t>(v64 >> 32ULL);
            }
            if ((routes & 1) != 0) {
                raw_r[routes - 1] = choice.GetValue(routes - 1);
            }
        }
        auto unpack_prob_nz = [&]() __aicore__ {
            auto prob64 = probability.ReinterpretCast<uint64_t>();
            const int64_t rpairs = routes >> 1;
            for (int64_t p = 0; p < rpairs; ++p) {
                const uint64_t pv = prob64.GetValue(p);
                const uint32_t i0 = static_cast<uint32_t>(2 * p);
                const uint64_t b0 = ((pv & 0x7FFFFFFFULL) != 0ULL) ? 1ULL : 0ULL;
                const uint64_t b1 = (((pv >> 32ULL) & 0x7FFFFFFFULL) != 0ULL) ? 1ULL : 0ULL;
                prob_nz[i0 >> 6U] |= (b0 << (i0 & 63U)) | (b1 << ((i0 + 1U) & 63U));
            }
            if ((routes & 1) != 0 && probability.GetValue(routes - 1) != 0.0F) {
                const uint32_t ilast = static_cast<uint32_t>(routes - 1);
                prob_nz[ilast >> 6U] |= (1ULL << (ilast & 63U));
            }
        };
        if (!defer_tid0_prob) {
            unpack_prob_nz();
        }
#if defined(K11220_FAST_MIRROR_V4) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (has_mirror) {
            GlobalTensor<uint32_t> plan_gm;
            plan_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(arguments.scratch[kGroupCount]), 96);
            if (GetBlockIdx() == 0) {
                auto mm_u64 = held.other.ReinterpretCast<uint64_t>();
                for (uint32_t k = 0U; k < 6U; ++k) {
                    mw_s[k] = mm_u64.GetValue(8U + k);
                }
                const uint64_t my_ge_pack = mm_u64.GetValue(14U);
                const uint64_t hdr = mm_u64.GetValue(15U);
                const uint32_t num_cands = static_cast<uint32_t>(hdr & 0xFFULL);
                const uint32_t my_mslots = static_cast<uint32_t>((hdr >> 8ULL) & 0xFFULL);
                uint64_t cand_ge_u64[4] = {0ULL, 0ULL, 0ULL, 0ULL};
                uint64_t cand_mask_u64[4] = {0ULL, 0ULL, 0ULL, 0ULL};
                if (num_cands > 0U) {
                    cand_ge_u64[0] = mm_u64.GetValue(0U);
                    cand_ge_u64[1] = mm_u64.GetValue(1U);
                    cand_ge_u64[2] = mm_u64.GetValue(2U);
                    cand_ge_u64[3] = mm_u64.GetValue(3U);
                    cand_mask_u64[0] = mm_u64.GetValue(4U);
                    cand_mask_u64[1] = mm_u64.GetValue(5U);
                    cand_mask_u64[2] = mm_u64.GetValue(6U);
                    cand_mask_u64[3] = mm_u64.GetValue(7U);
                }
                uint64_t cnt_u64_buf[32];
                uint8_t* exp_cnt = reinterpret_cast<uint8_t*>(cnt_u64_buf);
                uint8_t act_ge[32];
                uint8_t act_own[32];
                const uint32_t n_act = whole_layer_tree::AssignExpertOwnersFastV4(
                    raw_r, static_cast<int32_t>(routes), num_cands,
                    cand_ge_u64, cand_mask_u64, exp_cnt, act_ge, act_own);
                for (uint32_t w = 0U; w < 32U; ++w) {
                    g2e_u64[w] = 0xFFFFFFFFFFFFFFFFULL;
                }
                const uint32_t u_rank = static_cast<uint32_t>(comm_rank);
                const uint32_t home_base = u_rank * static_cast<uint32_t>(kLocalExperts);
                uint32_t my_home_active = 0U;
                for (uint32_t le = 0U; le < static_cast<uint32_t>(kLocalExperts); ++le) {
                    if (exp_cnt[home_base | le] > 0U) {
                        my_home_active |= (1U << le);
                    }
                }
                uint8_t mir_ge[8];
                uint8_t mir_lslot[8];
                uint32_t n_mir = 0U;
                for (uint32_t idx = 0U; idx < n_act; ++idx) {
                    const uint32_t ge = act_ge[idx];
                    const uint32_t own = act_own[idx];
                    const uint32_t home = ge >> whole_layer_tree::kExpertShift;
                    if (home == u_rank && own != u_rank) {
                        my_home_active &= ~(1U << (ge & static_cast<uint32_t>(kLocalExperts - 1)));
                    } else if (own == u_rank && home != u_rank) {
                        for (uint32_t ms = 0U; ms < my_mslots; ++ms) {
                            if (((my_ge_pack >> (ms * 8U)) & 0xFFULL) == ge) {
                                mir_ge[n_mir] = static_cast<uint8_t>(ge);
                                mir_lslot[n_mir] = static_cast<uint8_t>(static_cast<uint32_t>(kLocalExperts) + ms);
                                ++n_mir;
                                break;
                            }
                        }
                    }
                }
                int32_t my_ecnt = 0;
                while (my_home_active != 0U) {
                    const uint32_t le = static_cast<uint32_t>(__builtin_ctz(my_home_active));
                    my_home_active &= (my_home_active - 1U);
                    gexp_to_eidx[home_base | le] = static_cast<int8_t>(my_ecnt);
                    my_gexp[my_ecnt++] = static_cast<uint8_t>(le);
                }
                for (uint32_t k = 0U; k < n_mir; ++k) {
                    gexp_to_eidx[mir_ge[k]] = static_cast<int8_t>(my_ecnt);
                    my_gexp[my_ecnt++] = mir_lslot[k];
                }
                const uint32_t u64_per_rank = static_cast<uint32_t>(kLocalExperts) >> 3U;
                for (uint32_t r = 0U; r < static_cast<uint32_t>(comm_ranks); ++r) {
                    const uint64_t v = 0x0101010101010101ULL * r;
                    for (uint32_t u = 0U; u < u64_per_rank; ++u) {
                        own_u64[r * u64_per_rank + u] = v;
                    }
                }
                for (uint32_t idx = 0U; idx < n_act; ++idx) {
                    const uint32_t ge = act_ge[idx];
                    const uint32_t sh = (ge & 7U) << 3U;
                    own_u64[ge >> 3U] = (own_u64[ge >> 3U] & ~(0xFFULL << sh))
                        | (static_cast<uint64_t>(act_own[idx]) << sh);
                }
                Fence<HardEvent::V_S>();
                auto own_ub64 = held.input.ReinterpretCast<uint64_t>();
                for (uint32_t w = 0U; w < 32U; ++w) {
                    own_ub64.SetValue(w, own_u64[w]);
                    mm_u64.SetValue(w, g2e_u64[w]);
                }
                for (uint32_t w = 0U; w < 4U; ++w) {
                    mm_u64.SetValue(32U + w, my_gexp_u64[w]);
                }
                for (uint32_t w = 0U; w < 6U; ++w) {
                    mm_u64.SetValue(36U + w, mw_s[w]);
                }
                Fence<HardEvent::S_MTE3>();
                GlobalTensor<uint32_t> own_dst;
                own_dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(arguments.scratch[kDownTiling]), 64);
                DataCopy(own_dst, own_ub64.ReinterpretCast<uint32_t>(), 64);
                DataCopy(plan_gm, mm_u64.ReinterpretCast<uint32_t>(), 96);
                if (defer_tid0_prob) {
                    DataCopy(probability, probabilities, routes);
                    DataCopyPad(held.input_scale, input_scales,
                                DataCopyExtParams{1, uint32_t(arguments.rows * sizeof(float)), 0, 0, 0},
                                DataCopyPadExtParams<float>{false, 0, 0, 0});
                }
                PipeBarrier<PIPE_ALL>();
                DataSyncBarrier<MemDsbT::DDR>();
            }
            SyncAll<true>();
            if (GetBlockIdx() == 0) {
                if (defer_tid0_prob) {
                    Fence<HardEvent::MTE2_S>();
                    unpack_prob_nz();
                }
            } else {
                Fence<HardEvent::V_MTE2>();
                DataCopy(held.other.ReinterpretCast<uint32_t>(), plan_gm, 96);
                Fence<HardEvent::MTE2_S>();
                auto pl_u64 = held.other.ReinterpretCast<uint64_t>();
                for (uint32_t w = 0U; w < 32U; ++w) {
                    g2e_u64[w] = pl_u64.GetValue(w);
                }
                for (uint32_t w = 0U; w < 4U; ++w) {
                    my_gexp_u64[w] = pl_u64.GetValue(32U + w);
                }
                for (uint32_t w = 0U; w < 6U; ++w) {
                    mw_s[w] = pl_u64.GetValue(36U + w);
                }
            }
        }
#endif
#else
        DataCopy(choice, choices, routes);
        DataCopy(probability, probabilities, routes);
#if defined(K11180_LITE_MIRROR) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (has_mirror) {
            GlobalTensor<uint32_t> mm_gm;
            mm_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ whole_layer_tree::LiteMirrorMeta*>(lmeta)), 16);
            DataCopy(held.other.ReinterpretCast<uint32_t>(), mm_gm, 16);
        }
#elif defined(K11220_FAST_MIRROR_V4) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (has_mirror) {
            GlobalTensor<uint32_t> mm_gm;
#ifdef K11220_FAST_P1_V3
            mm_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ whole_layer_tree::LayerMirrorMeta*>(mmeta)), 64);
            DataCopy(held.other.ReinterpretCast<uint32_t>(), mm_gm, 64);
#else
            mm_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ whole_layer_tree::LayerMirrorMeta*>(mmeta)), 32);
            DataCopy(held.other.ReinterpretCast<uint32_t>(), mm_gm, 32);
#endif
        }
#endif
        DataCopyPad(held.input_scale, input_scales,
                    DataCopyExtParams{1, uint32_t(arguments.rows * sizeof(float)), 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
#if defined(K11220_FAST_P1_V3) && defined(K11220_FAST_TOK_OUT) && defined(K11220_FAST_ACT_P3)
        if (combine_shared) {
            Duplicate(held.original_weight, 0.0F, kMaxRoutes);
            Fence<HardEvent::MTE2_V>();
            Fence<HardEvent::V_S>();
        }
#else
        Duplicate(held.original_weight, 0.0F, kMaxRoutes);
        Fence<HardEvent::MTE2_V>();
        Fence<HardEvent::V_S>();
#endif
        Fence<HardEvent::MTE2_S>();
        {
            auto choice64 = choice.ReinterpretCast<uint64_t>();
            const int64_t rpairs = routes >> 1;
            for (int64_t p = 0; p < rpairs; ++p) {
                const uint64_t v64 = choice64.GetValue(p);
                raw_r[2 * p] = static_cast<int32_t>(v64 & 0xFFFFFFFFULL);
                raw_r[2 * p + 1] = static_cast<int32_t>(v64 >> 32ULL);
            }
            if ((routes & 1) != 0) {
                raw_r[routes - 1] = choice.GetValue(routes - 1);
            }
        }
        const int64_t own_writer = (2 * arguments.blocks >= 48) ? 47 : 0;
        (void)own_writer;
#if defined(K11180_LITE_MIRROR) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        uint8_t* owner_of_exp = reinterpret_cast<uint8_t*>(own_u64);
        if (has_mirror) {
            auto mm_u64 = held.other.ReinterpretCast<uint64_t>();
            for (uint32_t k = 0U; k < 6U; ++k) {
                mw_s[k] = mm_u64.GetValue(k);
            }
            const uint64_t packed_mge = mm_u64.GetValue(6);
            whole_layer_tree::FillLiteOwners(choice, static_cast<int32_t>(routes), packed_mge, owner_of_exp);
            for (uint32_t w = 0U; w < 32U; ++w) {
                g2e_u64[w] = 0xFFFFFFFFFFFFFFFFULL;
            }
            uint64_t my_mask[4] = {0ULL, 0ULL, 0ULL, 0ULL};
            for (int64_t original = 0; original < routes; ++original) {
                const int32_t raw = raw_r[original];
                if (raw < 0) {
                    continue;
                }
                const uint32_t ge = static_cast<uint32_t>(raw) & 255U;
                if (owner_of_exp[ge] == static_cast<uint8_t>(comm_rank)) {
                    my_mask[ge >> 6U] |= (1ULL << (ge & 63U));
                }
            }
            int32_t my_ecnt = 0;
            for (uint32_t w = 0U; w < 4U; ++w) {
                uint64_t m = my_mask[w];
                while (m != 0ULL) {
                    const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(m));
                    const uint32_t ge = (w << 6U) | bit;
                    gexp_to_eidx[ge] = static_cast<int8_t>(my_ecnt);
                    my_gexp[my_ecnt++] = ((ge >> whole_layer_tree::kExpertShift) == static_cast<uint32_t>(comm_rank))
                        ? static_cast<uint8_t>(ge & static_cast<uint32_t>(kLocalExperts - 1))
                        : static_cast<uint8_t>(kLocalExperts);
                    m &= (m - 1ULL);
                }
            }
        }
#elif defined(K11220_FAST_MIRROR_V4) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (has_mirror) {
            auto mm_u64 = held.other.ReinterpretCast<uint64_t>();
            for (uint32_t k = 0U; k < 6U; ++k) {
                mw_s[k] = mm_u64.GetValue(8U + k);
            }
            const uint64_t my_ge_pack = mm_u64.GetValue(14U);
            const uint64_t hdr = mm_u64.GetValue(15U);
            const uint32_t num_cands = static_cast<uint32_t>(hdr & 0xFFULL);
            const uint32_t my_mslots = static_cast<uint32_t>((hdr >> 8ULL) & 0xFFULL);
            uint64_t cnt_u64_buf[32];
            uint8_t* exp_cnt = reinterpret_cast<uint8_t*>(cnt_u64_buf);
#ifdef K112600_MIRROR_64
            uint64_t cand_ge_u64[8] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
            uint64_t cand_mask_u64[8] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
            if (num_cands > 0U) {
                cand_ge_u64[0] = mm_u64.GetValue(0U);
                cand_ge_u64[1] = mm_u64.GetValue(1U);
                cand_ge_u64[2] = mm_u64.GetValue(2U);
                cand_ge_u64[3] = mm_u64.GetValue(3U);
                cand_mask_u64[0] = mm_u64.GetValue(4U);
                cand_mask_u64[1] = mm_u64.GetValue(5U);
                cand_mask_u64[2] = mm_u64.GetValue(6U);
                cand_mask_u64[3] = mm_u64.GetValue(7U);
                if (num_cands > 32U && ((hdr & (1ULL << 17ULL)) != 0ULL)) {
                    cand_ge_u64[4] = mm_u64.GetValue(24U);
                    cand_ge_u64[5] = mm_u64.GetValue(25U);
                    cand_ge_u64[6] = mm_u64.GetValue(26U);
                    cand_ge_u64[7] = mm_u64.GetValue(27U);
                    cand_mask_u64[4] = mm_u64.GetValue(28U);
                    cand_mask_u64[5] = mm_u64.GetValue(29U);
                    cand_mask_u64[6] = mm_u64.GetValue(30U);
                    cand_mask_u64[7] = mm_u64.GetValue(31U);
                } else if (num_cands > 32U && ((hdr & (1ULL << 16ULL)) != 0ULL)) {
                    cand_ge_u64[4] = mm_u64.GetValue(24U);
                    cand_ge_u64[5] = mm_u64.GetValue(25U);
                    cand_mask_u64[4] = mm_u64.GetValue(26U);
                    cand_mask_u64[5] = mm_u64.GetValue(27U);
                }
            }
            uint8_t act_ge[64];
            uint8_t act_own[64];
            const uint32_t n_act = whole_layer_tree::AssignExpertOwnersFastV6(
                raw_r, static_cast<int32_t>(routes), num_cands,
                cand_ge_u64, cand_mask_u64, exp_cnt, act_ge, act_own);
#elif defined(K11220_FAST_P1_V3)
            uint64_t cand_ge_u64[6] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
            uint64_t cand_mask_u64[6] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
            if (num_cands > 0U) {
                cand_ge_u64[0] = mm_u64.GetValue(0U);
                cand_ge_u64[1] = mm_u64.GetValue(1U);
                cand_ge_u64[2] = mm_u64.GetValue(2U);
                cand_ge_u64[3] = mm_u64.GetValue(3U);
                cand_mask_u64[0] = mm_u64.GetValue(4U);
                cand_mask_u64[1] = mm_u64.GetValue(5U);
                cand_mask_u64[2] = mm_u64.GetValue(6U);
                cand_mask_u64[3] = mm_u64.GetValue(7U);
                if (num_cands > 32U && ((hdr & (1ULL << 16ULL)) != 0ULL)) {
                    cand_ge_u64[4] = mm_u64.GetValue(24U);
                    cand_ge_u64[5] = mm_u64.GetValue(25U);
                    cand_mask_u64[4] = mm_u64.GetValue(26U);
                    cand_mask_u64[5] = mm_u64.GetValue(27U);
                }
            }
            uint8_t act_ge[48];
            uint8_t act_own[48];
            const uint32_t n_act = whole_layer_tree::AssignExpertOwnersFastV5(
                raw_r, static_cast<int32_t>(routes), num_cands,
                cand_ge_u64, cand_mask_u64, exp_cnt, act_ge, act_own);
#else
            uint64_t cand_ge_u64[4] = {0ULL, 0ULL, 0ULL, 0ULL};
            uint64_t cand_mask_u64[4] = {0ULL, 0ULL, 0ULL, 0ULL};
            if (num_cands > 0U) {
                cand_ge_u64[0] = mm_u64.GetValue(0U);
                cand_ge_u64[1] = mm_u64.GetValue(1U);
                cand_ge_u64[2] = mm_u64.GetValue(2U);
                cand_ge_u64[3] = mm_u64.GetValue(3U);
                cand_mask_u64[0] = mm_u64.GetValue(4U);
                cand_mask_u64[1] = mm_u64.GetValue(5U);
                cand_mask_u64[2] = mm_u64.GetValue(6U);
                cand_mask_u64[3] = mm_u64.GetValue(7U);
            }
            uint8_t act_ge[32];
            uint8_t act_own[32];
            const uint32_t n_act = whole_layer_tree::AssignExpertOwnersFastV4(
                raw_r, static_cast<int32_t>(routes), num_cands,
                cand_ge_u64, cand_mask_u64, exp_cnt, act_ge, act_own);
#endif
            for (uint32_t w = 0U; w < 32U; ++w) {
                g2e_u64[w] = 0xFFFFFFFFFFFFFFFFULL;
            }
            const uint32_t u_rank = static_cast<uint32_t>(comm_rank);
            const uint32_t home_base = u_rank * static_cast<uint32_t>(kLocalExperts);
            uint32_t my_home_active = 0U;
            for (uint32_t le = 0U; le < static_cast<uint32_t>(kLocalExperts); ++le) {
                if (exp_cnt[home_base | le] > 0U) {
                    my_home_active |= (1U << le);
                }
            }
            uint8_t mir_ge[8];
            uint8_t mir_lslot[8];
            uint32_t n_mir = 0U;
            for (uint32_t idx = 0U; idx < n_act; ++idx) {
                const uint32_t ge = act_ge[idx];
                const uint32_t own = act_own[idx];
                const uint32_t home = ge >> whole_layer_tree::kExpertShift;
                if (home == u_rank && own != u_rank) {
                    my_home_active &= ~(1U << (ge & static_cast<uint32_t>(kLocalExperts - 1)));
                } else if (own == u_rank && home != u_rank) {
                    for (uint32_t ms = 0U; ms < my_mslots; ++ms) {
                        if (((my_ge_pack >> (ms * 8U)) & 0xFFULL) == ge) {
                            mir_ge[n_mir] = static_cast<uint8_t>(ge);
                            mir_lslot[n_mir] = static_cast<uint8_t>(static_cast<uint32_t>(kLocalExperts) + ms);
                            ++n_mir;
                            break;
                        }
                    }
                }
            }
            int32_t my_ecnt = 0;
            while (my_home_active != 0U) {
                const uint32_t le = static_cast<uint32_t>(__builtin_ctz(my_home_active));
                my_home_active &= (my_home_active - 1U);
                gexp_to_eidx[home_base | le] = static_cast<int8_t>(my_ecnt);
                my_gexp[my_ecnt++] = static_cast<uint8_t>(le);
            }
            for (uint32_t k = 0U; k < n_mir; ++k) {
                gexp_to_eidx[mir_ge[k]] = static_cast<int8_t>(my_ecnt);
                my_gexp[my_ecnt++] = mir_lslot[k];
            }
            if (GetBlockIdx() == own_writer) {
                const uint32_t u64_per_rank = static_cast<uint32_t>(kLocalExperts) >> 3U;
                for (uint32_t r = 0U; r < static_cast<uint32_t>(comm_ranks); ++r) {
                    const uint64_t v = 0x0101010101010101ULL * r;
                    for (uint32_t u = 0U; u < u64_per_rank; ++u) {
                        own_u64[r * u64_per_rank + u] = v;
                    }
                }
                for (uint32_t idx = 0U; idx < n_act; ++idx) {
                    const uint32_t ge = act_ge[idx];
                    const uint32_t sh = (ge & 7U) << 3U;
                    own_u64[ge >> 3U] = (own_u64[ge >> 3U] & ~(0xFFULL << sh))
                        | (static_cast<uint64_t>(act_own[idx]) << sh);
                }
            }
        }
#endif
#endif
        int32_t cnt_arr[kLocalExperts] = {};
        int32_t start_arr[kLocalExperts];
        int32_t cur_arr[kLocalExperts];
        int32_t inv_arr[kMaxRoutes];
        uint32_t hit_pack[kMaxRoutes];
        int32_t local_tok_s[kMaxTokens];
        int32_t num_hits = 0;
        for (int64_t original = 0; original < routes; ++original) {
            const int32_t raw_ge = raw_r[original];
            const int32_t expert = pw_active
                ? ((static_cast<uint32_t>(raw_ge) < 256U) ? static_cast<int64_t>(gexp_to_eidx[raw_ge]) : -1)
                : (raw_ge - expert_offset);
            if (static_cast<uint32_t>(expert) < static_cast<uint32_t>(kLocalExperts)) {
                cnt_arr[expert]++;
                hit_pack[num_hits++] = static_cast<uint32_t>(original) | (static_cast<uint32_t>(expert) << 8U);
            }
        }
        int32_t prefix = 0;
        int32_t ord5_exp = -1;
        int32_t ord6_exp = -1;
        int32_t ord6_sb = 0;
#ifdef ROUTED_ACTIVE_LIST
        int32_t act_cnt = 0;
#endif
        for (int64_t expert = 0; expert < kLocalExperts; ++expert) {
            const int32_t c = cnt_arr[expert];
#if defined(K11220_SINGLE_OWN_P1) || (defined(K11220_FAST_P1_V3) && defined(K11220_FAST_TOK_OUT) && defined(K11220_FAST_ACT_P3))
            if (combine_shared)
#endif
            {
                held.start.SetValue(expert, prefix);
                held.count.SetValue(expert, c);
            }
            start_arr[expert] = prefix;
            cur_arr[expert] = prefix;
#ifdef ROUTED_ACTIVE_LIST
            if (c > 0) {
                const uint32_t gexp_pack = pw_active
                    ? (static_cast<uint32_t>(my_gexp[expert]) << 24U)
                    : 0U;
                int32_t rem = c;
                int32_t sb = prefix;
#ifdef K11220_MULTI_M32
                constexpr int32_t kMaxChunkRows = 32;
#elif defined(K11220_MULTI_M16)
                constexpr int32_t kMaxChunkRows = 16;
#else
                constexpr int32_t kMaxChunkRows = 8;
#endif
                do {
                    const int32_t sc = rem > kMaxChunkRows ? kMaxChunkRows : rem;
#if defined(K11220_SINGLE_OWN_P1) || defined(K11220_FAST_P1_V3)
                    if (GetBlockIdx() == 0)
#endif
                    {
                        held.active.SetValue(kActiveHeaderElements + act_cnt,
                            static_cast<int32_t>(static_cast<uint32_t>(expert)
                                | (static_cast<uint32_t>(sb) << 8U)
                                | (static_cast<uint32_t>(sc) << 16U)
                                | gexp_pack));
                    }
                    if (act_cnt == 5) {
                        ord5_exp = static_cast<int32_t>(expert);
                    } else if (act_cnt == 6) {
                        ord6_exp = static_cast<int32_t>(expert);
                        ord6_sb = sb;
                    }
                    ++act_cnt;
                    rem -= kMaxChunkRows;
                    sb += kMaxChunkRows;
                } while (rem > 0);
            }
#endif
            prefix += c;
        }
        uint32_t local_mask = 0;
        int32_t num_local_tokens = 0;
#ifdef K11220_FAST_TOK_OUT
        uint32_t pos_pack[kMaxRoutes];
#endif
        for (int32_t i = 0; i < num_hits; ++i) {
            const uint32_t hp = hit_pack[i];
            const int32_t original = static_cast<int32_t>(hp & 0xFFU);
            const int32_t expert = static_cast<int32_t>(hp >> 8U);
            const int32_t position = cur_arr[expert]++;
            const int32_t src_row = original >> 3;
            const uint32_t bit = 1u << uint32_t(src_row);
            if ((local_mask & bit) == 0) {
                local_mask |= bit;
                local_tok_s[num_local_tokens] = src_row;
#if defined(K11220_SINGLE_OWN_P1) || (defined(K11220_FAST_P1_V3) && defined(K11220_FAST_TOK_OUT) && defined(K11220_FAST_ACT_P3))
                if (combine_shared)
#endif
                {
                    active_tokens.SetValue(num_local_tokens, src_row);
                }
                ++num_local_tokens;
            }
            inv_arr[original] = position;
#ifdef K11220_SINGLE_OWN_P1
            bool p_nonzero = false;
            if (!combine_shared) {
                const uint32_t uorig = static_cast<uint32_t>(original);
                p_nonzero = ((prob_nz[uorig >> 6U] >> (uorig & 63U)) & 1ULL) != 0ULL;
            } else {
                const float prob = probability.GetValue(original);
                p_nonzero = (prob != 0.0F);
                held.source.SetValue(position, src_row);
                held.source.SetValue(kMaxRoutes + position, original);
                inverse.SetValue(original, position);
                held.original_weight.SetValue(original, prob);
            }
#elif defined(K11220_FAST_P1_V3) && defined(K11220_FAST_TOK_OUT) && defined(K11220_FAST_ACT_P3)
            const float prob = probability.GetValue(original);
            const bool p_nonzero = (prob != 0.0F);
            if (combine_shared) {
                held.source.SetValue(position, src_row);
                held.source.SetValue(kMaxRoutes + position, original);
                inverse.SetValue(original, position);
                held.original_weight.SetValue(original, prob);
            }
#else
            held.source.SetValue(position, src_row);
            held.source.SetValue(kMaxRoutes + position, original);
            inverse.SetValue(original, position);
            const float prob = probability.GetValue(original);
            held.original_weight.SetValue(original, prob);
            const bool p_nonzero = (prob != 0.0F);
#endif
#if defined(K11220_FAST_TOK_OUT) || defined(K11220_FAST_ACT_P3)
            const int32_t st = start_arr[expert];
            const int32_t r = position - st;
            const int32_t cnt = cnt_arr[expert];
#endif
#ifdef K11220_FAST_ACT_P3
            act_pack_s[position] = static_cast<uint32_t>(expert)
                | (static_cast<uint32_t>(st) << 6U)
                | (static_cast<uint32_t>(r) << 14U)
                | (static_cast<uint32_t>(cnt - 1) << 19U)
                | (static_cast<uint32_t>(src_row) << 24U);
#endif
#ifdef K11220_FAST_TOK_OUT
            pos_pack[position] = p_nonzero
                ? (static_cast<uint32_t>(original)
                    | (static_cast<uint32_t>(expert) << 8U)
                    | (static_cast<uint32_t>(st) << 14U)
                    | (static_cast<uint32_t>(r) << 22U)
                    | (static_cast<uint32_t>(cnt - 1) << 27U))
                : 0xFFFFFFFFU;
#endif
        }
#ifdef K11220_FAST_ACT_P3
        act_total_s = prefix;
#endif
#ifdef K112600_PIPE_C2V
        pipe_wave0_s = (!combine_shared && act_cnt > 6 && ord5_exp != ord6_exp);
        split_act_pos_s = ord6_sb;
        split_ord6_exp_s = ord6_exp;
#else
        (void)ord5_exp;
        (void)ord6_exp;
        (void)ord6_sb;
#endif
#ifdef K11220_FAST_TOK_OUT
        {
            int32_t lc = 0;
            for (int32_t p = 0; p < prefix; ++p) {
                if (p == ord6_sb) {
                    split_local_cnt_s = lc;
                }
                const uint32_t pk = pos_pack[p];
                if (pk != 0xFFFFFFFFU) {
                    local_origs_s[lc++] = pk;
                }
            }
            local_cnt_s = lc;
        }
#endif
        saved_local_mask = local_mask;
#if defined(K11220_SINGLE_OWN_P1) || (defined(K11220_FAST_P1_V3) && defined(K11220_FAST_TOK_OUT) && defined(K11220_FAST_ACT_P3))
        if (combine_shared)
#endif
        {
            active_tokens.SetValue(32, int32_t(local_mask));
            active_tokens.SetValue(33, num_local_tokens);
        }
#ifdef ROUTED_ACTIVE_LIST
        active_count = act_cnt;
#if defined(K11220_SINGLE_OWN_P1) || defined(K11220_FAST_P1_V3)
        if (GetBlockIdx() == 0) {
            held.active.SetValue(0, act_cnt);
        }
#else
        held.active.SetValue(0, act_cnt);
#endif
#endif
        if (GetBlockIdx() == 0) {
            Fence<HardEvent::S_MTE3>();
#ifdef ROUTED_ACTIVE_LIST
#ifdef K11220_FAST_P1_V3
            const int32_t copy_act_elems = kActiveHeaderElements + ((act_cnt + 7) & ~7);
            DataCopy(active_experts, held.active, copy_act_elems);
#else
            DataCopy(active_experts, held.active, kActiveListElements);
#endif
#endif
            Fence<HardEvent::MTE3_V>();
        }
#if defined(K11220_FAST_P1_V3) && defined(K11220_FAST_MIRROR_V4) && !defined(K11220_SINGLE_OWN_P1) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (GetBlockIdx() == own_writer && has_mirror) {
            auto own_ub64 = held.tok_out.ReinterpretCast<uint64_t>();
            for (uint32_t w = 0U; w < 32U; ++w) {
                own_ub64.SetValue(w, own_u64[w]);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
            GlobalTensor<uint32_t> own_dst;
            own_dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(arguments.scratch[kDownTiling]), 64);
            DataCopy(own_dst, own_ub64.ReinterpretCast<uint32_t>(), 64);
        }
#endif
        PipeBarrier<PIPE_ALL>();
        SplitTokens(held, (__gm__ int8_t*)arguments.scratch[kRestoredRows],
            (__gm__ int4b_t*)arguments.scratch[kStacked],
            (__gm__ int32_t*)arguments.dispatch.expert_ids,
            arguments.rows, arguments.blocks, expert_offset,
            pw_active ? gexp_to_eidx : nullptr,
            raw_r, start_arr, cnt_arr, inv_arr,
            local_tok_s, num_local_tokens);
#if !defined(K11220_FAST_P1_V3) && defined(K11220_FAST_MIRROR_V4) && !defined(K11220_SINGLE_OWN_P1) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (GetBlockIdx() == own_writer && has_mirror) {
            Fence<HardEvent::V_S>();
            auto own_ub64 = held.input.ReinterpretCast<uint64_t>();
            for (uint32_t w = 0U; w < 32U; ++w) {
                own_ub64.SetValue(w, own_u64[w]);
            }
            Fence<HardEvent::S_MTE3>();
            GlobalTensor<uint32_t> own_dst;
            own_dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(arguments.scratch[kDownTiling]), 64);
            DataCopy(own_dst, own_ub64.ReinterpretCast<uint32_t>(), 64);
        }
#endif
        PipeBarrier<PIPE_ALL>();
#ifndef K112612_TRIM_BARRIERS
        Publish();
#endif
    }
#if defined(K11220_FAST_P1_V3) && defined(K11220_FAST_MIRROR_V4) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
    if ASCEND_IS_AIC {
        if (has_mirror) {
            GlobalTensor<uint64_t> mw_gm;
            mw_gm.SetGlobalBuffer(const_cast<__gm__ uint64_t*>(mmeta->mirror_weights), 8);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(mw_gm);
            mw_s[0U] = mw_gm.GetValue(0);
            mw_s[2U] = mw_gm.GetValue(2);
            mw_s[3U] = mw_gm.GetValue(3);
            mw_s[5U] = mw_gm.GetValue(5);
        }
    }
#endif
    timing.Barrier(1);
    if (moe_prof != nullptr) { mp1 = GetSystemCycle(); }
    timing.Begin(2);
    if ASCEND_IS_AIC {
#if defined(K11180_LITE_MIRROR) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (has_mirror) {
            GlobalTensor<uint64_t> mw_gm;
            mw_gm.SetGlobalBuffer(const_cast<__gm__ uint64_t*>(lmeta->mirror_weights), 8);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(mw_gm);
            mw_s[0U] = mw_gm.GetValue(0);
            mw_s[2U] = mw_gm.GetValue(2);
            mw_s[3U] = mw_gm.GetValue(3);
            mw_s[5U] = mw_gm.GetValue(5);
        }
#elif !defined(K11220_FAST_P1_V3) && defined(K11220_FAST_MIRROR_V4) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if (has_mirror) {
            GlobalTensor<uint64_t> mw_gm;
            mw_gm.SetGlobalBuffer(const_cast<__gm__ uint64_t*>(mmeta->mirror_weights), 8);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(mw_gm);
            mw_s[0U] = mw_gm.GetValue(0);
            mw_s[2U] = mw_gm.GetValue(2);
            mw_s[3U] = mw_gm.GetValue(3);
            mw_s[5U] = mw_gm.GetValue(5);
        }
#endif
#ifdef ROUTED_ACTIVE_LIST
#ifdef K11220_FAST_P1_V3
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[0]);
        active_count = active_experts.GetValue(0);
        if (active_count > 8) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[16]);
            if (active_count > 24) {
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[32]);
                if (active_count > 40) {
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[48]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[64]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[80]);
                }
            }
        }
#else
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[0]);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[16]);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[32]);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[48]);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[64]);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(active_experts[80]);
        active_count = active_experts.GetValue(0);
#endif
#ifdef K112600_PIPE_C2V
        pipe_wave0_s = (!combine_shared && active_count > 6
            && ((active_experts.GetValue(kActiveHeaderElements + 5) & 0xFF)
                != (active_experts.GetValue(kActiveHeaderElements + 6) & 0xFF)));
#endif
#endif
    }
    auto gate_w_ptr = [&](uint32_t packed, int64_t expert, int64_t column) __aicore__ -> __gm__ int4b_t* {
        if (pw_active) {
            const uint32_t lslot = (packed >> 24U) & 0xFFU;
            if (lslot < 32U) {
                return Nibbles(reinterpret_cast<__gm__ int4b_t*>(arguments.weights[0]), (static_cast<int64_t>(lslot) * kGateUp + column) * kHidden);
            }
            const int64_t ms = static_cast<int64_t>(lslot - 32U);
            return Nibbles(reinterpret_cast<__gm__ int4b_t*>(mw_s[0U]), (ms * kGateUp + column) * kHidden);
        }
        return Nibbles(reinterpret_cast<__gm__ int4b_t*>(arguments.weights[0]), (expert * kGateUp + column) * kHidden);
    };
    auto gate_s_ptr = [&](uint32_t packed, int64_t expert, int64_t column) __aicore__ -> __gm__ ScaleStorage* {
        if (pw_active) {
            const uint32_t lslot = (packed >> 24U) & 0xFFU;
            if (lslot < 32U) {
                return reinterpret_cast<__gm__ ScaleStorage*>(arguments.weights[2]) + static_cast<int64_t>(lslot) * kGateUp + column;
            }
            const int64_t ms = static_cast<int64_t>(lslot - 32U);
            return reinterpret_cast<__gm__ ScaleStorage*>(mw_s[2U]) + ms * kGateUp + column;
        }
        return reinterpret_cast<__gm__ ScaleStorage*>(arguments.weights[2]) + expert * kGateUp + column;
    };
    auto down_w_ptr = [&](uint32_t packed, int64_t expert, int64_t column) __aicore__ -> __gm__ int4b_t* {
        if (pw_active) {
            const uint32_t lslot = (packed >> 24U) & 0xFFU;
            if (lslot < 32U) {
                return Nibbles(reinterpret_cast<__gm__ int4b_t*>(arguments.weights[3]), (static_cast<int64_t>(lslot) * kHidden + column) * kIntermediate);
            }
            const int64_t ms = static_cast<int64_t>(lslot - 32U);
            return Nibbles(reinterpret_cast<__gm__ int4b_t*>(mw_s[3U]), (ms * kHidden + column) * kIntermediate);
        }
        return Nibbles(reinterpret_cast<__gm__ int4b_t*>(arguments.weights[3]), (expert * kHidden + column) * kIntermediate);
    };
    auto down_s_ptr = [&](uint32_t packed, int64_t expert, int64_t column) __aicore__ -> __gm__ ScaleStorage* {
        if (pw_active) {
            const uint32_t lslot = (packed >> 24U) & 0xFFU;
            if (lslot < 32U) {
                return reinterpret_cast<__gm__ ScaleStorage*>(arguments.weights[5]) + static_cast<int64_t>(lslot) * kHidden + column;
            }
            const int64_t ms = static_cast<int64_t>(lslot - 32U);
            return reinterpret_cast<__gm__ ScaleStorage*>(mw_s[5U]) + ms * kHidden + column;
        }
        return reinterpret_cast<__gm__ ScaleStorage*>(arguments.weights[5]) + expert * kHidden + column;
    };
    auto rem_gate_task = [&](int64_t g_rem6, int64_t rem_base, int64_t tid, int64_t& ordinal, int64_t& column, int64_t& g_tile) __aicore__ {
        ordinal = -1;
        column = 0;
        g_tile = 0;
        if (g_rem6 == 5 && tid < 24) {
            if (tid < 20) {
                const uint32_t q5 = (static_cast<uint32_t>(tid) * 205U) >> 10U;
                ordinal = rem_base + static_cast<int64_t>(q5);
                const int64_t sub = tid - static_cast<int64_t>(q5 * 5U);
                column = (sub == 0) ? 0 : (1024 + (sub - 1) * 768);
                g_tile = (sub == 0) ? 1024 : 768;
            } else {
                ordinal = rem_base + 4;
                column = (tid - 20) * 1024;
                g_tile = 1024;
            }
        } else if (g_rem6 == 4 && tid < 24) {
            const uint32_t q6 = (static_cast<uint32_t>(tid) * 683U) >> 12U;
            ordinal = rem_base + static_cast<int64_t>(q6);
            const int64_t sub = tid - static_cast<int64_t>(q6 * 6U);
            column = (sub < 4) ? (sub * 768) : (3072 + (sub - 4) * 512);
            g_tile = (sub < 4) ? 768 : 512;
        } else if (g_rem6 == 3 && tid < 24) {
            ordinal = rem_base + (tid >> 3);
            column = (tid & 7) * 512;
            g_tile = 512;
        } else if (g_rem6 == 2 && tid < 24) {
            const uint32_t q12 = (static_cast<uint32_t>(tid) * 683U) >> 13U;
            ordinal = rem_base + static_cast<int64_t>(q12);
            const int64_t sub = tid - static_cast<int64_t>(q12 * 12U);
            column = (sub < 4) ? (sub * 512) : (2048 + (sub - 4) * 256);
            g_tile = (sub < 4) ? 512 : 256;
        } else if (g_rem6 == 1 && tid < 16) {
            ordinal = rem_base;
            column = tid * 256;
            g_tile = 256;
        }
    };
    auto run_gate_range = [&](auto tiles_tag, auto& mm, int64_t ord_begin, int64_t ord_count, int64_t g_rem6 = 0, int64_t rem_base = 0, bool sig_wave0 = false, int64_t blk_begin = 0, int64_t blk_count = 24) __aicore__ {
        constexpr int64_t g_tiles = decltype(tiles_tag)::value;
        constexpr int64_t g_shift = (g_tiles == 4) ? 2 : 3;
        constexpr int64_t g_mask = g_tiles - 1;
        constexpr int64_t g_tile = kGateUp / g_tiles;
        const int64_t tid = GetBlockIdx() - blk_begin;
        if (ord_count <= 0 || tid < 0 || tid >= blk_count) return;
        const int64_t total_tasks = ord_count * g_tiles;
        uint32_t cached_pk = 0U;
        bool has_cached_pk = false;
        for (int64_t task = tid; task < total_tasks; task += blk_count) {
            const int64_t ordinal = ord_begin + (task >> g_shift);
            const int64_t column = (task & g_mask) * g_tile;
            const uint32_t packed = has_cached_pk ? cached_pk : static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + ordinal));
            has_cached_pk = false;
            const int64_t expert = packed & 0xFFU;
            const int64_t begin = (packed >> 8) & 0xFFU;
            const int64_t sub_count = (packed >> 16) & 0xFFU;
#if defined(K11080_DIRECT_W4A8) && defined(K11220_XTASK_B)
            __gm__ int4b_t* next_w = nullptr;
            int64_t next_tc = 0;
            const int64_t ntask = task + blk_count;
            if (ntask < total_tasks) {
                const int64_t nord = ord_begin + (ntask >> g_shift);
                const int64_t ncol = (ntask & g_mask) * g_tile;
                const uint32_t npk = (nord == ordinal) ? packed : static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + nord));
                cached_pk = npk;
                has_cached_pk = true;
                next_w = gate_w_ptr(npk, npk & 0xFFU, ncol);
                next_tc = g_tile;
            } else if (g_rem6 > 0 && blk_begin == 0 && blk_count == 24) {
                int64_t rord = -1, rcol = 0, rtile = 0;
                rem_gate_task(g_rem6, rem_base, tid, rord, rcol, rtile);
                if (rord >= 0) {
                    const uint32_t npk = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + rord));
                    next_w = gate_w_ptr(npk, npk & 0xFFU, rcol);
                    next_tc = rtile;
                }
            }
            Project<kPadGateRows>(mm,
                    Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                    gate_w_ptr(packed, expert, column),
                    ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp + column,
                    gate_s_ptr(packed, expert, column),
                    g_tile, kGateUp, kHidden, sub_count, next_w, next_tc);
#else
            (void)g_rem6;
            (void)rem_base;
            Project<kPadGateRows>(mm,
                    Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                    gate_w_ptr(packed, expert, column),
                    ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp + column,
                    gate_s_ptr(packed, expert, column),
                    g_tile, kGateUp, kHidden, sub_count);
#endif
#ifdef K112600_PIPE_C2V
            if (sig_wave0 && task == tid) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(8);
            }
#else
            (void)sig_wave0;
#endif
        }
    };
    auto run_gate_waves = [&](auto& mm, int64_t batch_begin, int64_t batch_count) __aicore__ {
        if (batch_count <= 0) return;
        const int64_t g_full6 = static_cast<int64_t>((static_cast<uint32_t>(batch_count) * 683U) >> 12U) * 6;
        const int64_t g_rem6 = batch_count - g_full6;
        const int64_t rem_base = batch_begin + g_full6;
        run_gate_range(std::integral_constant<int64_t, 4>{}, mm, batch_begin, g_full6, g_rem6, rem_base, (batch_begin == 0 && pipe_wave0_s));
#ifdef K11088_GATE_CONTIG
        if (g_rem6 > 0) {
            int64_t ordinal = -1, column = 0, g_tile = 0;
            rem_gate_task(g_rem6, rem_base, GetBlockIdx(), ordinal, column, g_tile);
            if (ordinal >= 0) {
                const uint32_t packed = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + ordinal));
                const int64_t expert = packed & 0xFFU;
                const int64_t begin = (packed >> 8) & 0xFFU;
                const int64_t sub_count = (packed >> 16) & 0xFFU;
                Project<kPadGateRows>(mm,
                        Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                        gate_w_ptr(packed, expert, column),
                        ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp + column,
                        gate_s_ptr(packed, expert, column),
                        g_tile, kGateUp, kHidden, sub_count);
            }
        }
#else
        if (g_rem6 == 5) {
            const int64_t tid = GetBlockIdx();
            if (tid < 24) {
                int64_t ordinal;
                int64_t column;
                int64_t g_tile;
                if (tid < 20) {
                    const uint32_t q5 = (static_cast<uint32_t>(tid) * 205U) >> 10U;
                    ordinal = rem_base + static_cast<int64_t>(q5);
                    const int64_t sub = tid - static_cast<int64_t>(q5 * 5U);
                    column = (sub < 2) ? (sub * 896) : (sub * 768 + 256);
                    g_tile = (sub < 2) ? 896 : 768;
                } else {
                    ordinal = rem_base + 4;
                    column = (tid - 20) * 1024;
                    g_tile = 1024;
                }
                const uint32_t packed = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + ordinal));
                const int64_t expert = packed & 0xFFU;
                const int64_t begin = (packed >> 8) & 0xFFU;
                const int64_t sub_count = (packed >> 16) & 0xFFU;
                Project<kPadGateRows>(mm,
                        Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                        gate_w_ptr(packed, expert, column),
                        ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp + column,
                        gate_s_ptr(packed, expert, column),
                        g_tile, kGateUp, kHidden, sub_count);
            }
        } else if (g_rem6 == 4) {
            const int64_t tid = GetBlockIdx();
            if (tid < 24) {
                const uint32_t q6 = (static_cast<uint32_t>(tid) * 683U) >> 12U;
                const int64_t ordinal = rem_base + static_cast<int64_t>(q6);
                const int64_t sub = tid - static_cast<int64_t>(q6 * 6U);
                const int64_t column = (sub < 2) ? (sub * 768) : (sub * 640 + 256);
                const int64_t g_tile = (sub < 2) ? 768 : 640;
                const uint32_t packed = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + ordinal));
                const int64_t expert = packed & 0xFFU;
                const int64_t begin = (packed >> 8) & 0xFFU;
                const int64_t sub_count = (packed >> 16) & 0xFFU;
                Project<kPadGateRows>(mm,
                        Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                        gate_w_ptr(packed, expert, column),
                        ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp + column,
                        gate_s_ptr(packed, expert, column),
                        g_tile, kGateUp, kHidden, sub_count);
            }
        } else if (g_rem6 == 3) {
            run_gate_range(std::integral_constant<int64_t, 8>{}, mm, rem_base, 3);
        } else if (g_rem6 == 2) {
            const int64_t tid = GetBlockIdx();
            if (tid < 24) {
                const uint32_t q12 = (static_cast<uint32_t>(tid) * 683U) >> 13U;
                const int64_t ordinal = rem_base + static_cast<int64_t>(q12);
                const int64_t sub = tid - static_cast<int64_t>(q12 * 12U);
                const int64_t column = (sub < 8) ? (sub * 384) : (sub * 256 + 1024);
                const int64_t g_tile = (sub < 8) ? 384 : 256;
                const uint32_t packed = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + ordinal));
                const int64_t expert = packed & 0xFFU;
                const int64_t begin = (packed >> 8) & 0xFFU;
                const int64_t sub_count = (packed >> 16) & 0xFFU;
                Project<kPadGateRows>(mm,
                        Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                        gate_w_ptr(packed, expert, column),
                        ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp + column,
                        gate_s_ptr(packed, expert, column),
                        g_tile, kGateUp, kHidden, sub_count);
            }
        } else if (g_rem6 == 1) {
            const int64_t tid = GetBlockIdx();
            if (tid < 24) {
                const int64_t ordinal = rem_base;
                const int64_t column = (tid < 8) ? (tid * 256) : (tid * 128 + 1024);
                const int64_t g_tile = (tid < 8) ? 256 : 128;
                const uint32_t packed = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + ordinal));
                const int64_t expert = packed & 0xFFU;
                const int64_t begin = (packed >> 8) & 0xFFU;
                const int64_t sub_count = (packed >> 16) & 0xFFU;
                Project<kPadGateRows>(mm,
                        Nibbles(((__gm__ int4b_t*)arguments.scratch[kStacked]), GateStorageRow(begin) * kHidden),
                        gate_w_ptr(packed, expert, column),
                        ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp + column,
                        gate_s_ptr(packed, expert, column),
                        g_tile, kGateUp, kHidden, sub_count);
            }
        }
#endif
    };
    auto rem_down_task = [&](int64_t d_rem6, int64_t rem_base, int64_t tid, int64_t& ordinal, int64_t& column, int64_t& d_tile) __aicore__ {
        ordinal = -1;
        column = 0;
        d_tile = 0;
        if (tid < 0 || tid >= 24) return;
        const uint32_t utid = static_cast<uint32_t>(tid);
        if (d_rem6 == 5) {
            if (tid < 20) {
                const uint32_t q5 = (utid * 205U) >> 10U;
                ordinal = rem_base + static_cast<int64_t>(q5);
                const int64_t sub = tid - static_cast<int64_t>(q5 * 5U);
                column = sub * 1280;
                d_tile = (sub < 4) ? 1280 : 1024;
            } else {
                ordinal = rem_base + 4;
#ifdef K11220_BITMASK_OWN_P1
                column = (tid - 20) * 1280;
                d_tile = 1280;
#else
                column = (tid - 20) * 1536;
                d_tile = 1536;
#endif
            }
        } else if (d_rem6 == 4) {
            const uint32_t q6 = (utid * 683U) >> 12U;
            ordinal = rem_base + static_cast<int64_t>(q6);
            column = (tid - static_cast<int64_t>(q6 * 6U)) * 1024;
            d_tile = 1024;
        } else if (d_rem6 == 3) {
            ordinal = rem_base + (tid >> 3);
            column = (tid & 7) * 768;
            d_tile = 768;
        } else if (d_rem6 == 2) {
            const uint32_t q12 = (utid * 683U) >> 13U;
            ordinal = rem_base + static_cast<int64_t>(q12);
            column = (tid - static_cast<int64_t>(q12 * 12U)) * 512;
            d_tile = 512;
        } else if (d_rem6 == 1) {
            ordinal = rem_base;
            column = tid * 256;
            d_tile = 256;
        }
    };
    auto run_down_range = [&](auto tiles_tag, auto& mm, int64_t ord_begin, int64_t ord_count, int64_t d_rem6 = 0, int64_t rem_base = 0, bool sig_wave0 = false, int64_t blk_begin = 0, int64_t blk_count = 24) __aicore__ {
        constexpr int64_t d_tiles = decltype(tiles_tag)::value;
        constexpr int64_t d_tile = kHidden / d_tiles;
        const int64_t tid = GetBlockIdx() - blk_begin;
        if (ord_count <= 0 || tid < 0 || tid >= blk_count) return;
        const int64_t total_tasks = ord_count * d_tiles;
        uint32_t cached_pk = 0U;
        bool has_cached_pk = false;
        for (int64_t task = tid; task < total_tasks; task += blk_count) {
            const uint32_t utask = static_cast<uint32_t>(task);
            const uint32_t q = utask >> 2U;
            const int64_t ordinal = ord_begin + static_cast<int64_t>(q);
            const int64_t column = static_cast<int64_t>(utask & 3U) * d_tile;
            const uint32_t packed = has_cached_pk ? cached_pk : static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + ordinal));
            has_cached_pk = false;
            const int64_t expert = packed & 0xFFU;
            const int64_t begin = (packed >> 8) & 0xFFU;
            const int64_t sub_count = (packed >> 16) & 0xFFU;
#if defined(K11080_DIRECT_W4A8) && defined(K11220_XTASK_B)
            __gm__ int4b_t* next_w = nullptr;
            int64_t next_tc = 0;
            const int64_t ntask = task + blk_count;
            if (ntask < total_tasks) {
                const uint32_t nutask = static_cast<uint32_t>(ntask);
                const int64_t nord = ord_begin + static_cast<int64_t>(nutask >> 2U);
                const int64_t ncol = static_cast<int64_t>(nutask & 3U) * d_tile;
                const uint32_t npk = (nord == ordinal) ? packed : static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + nord));
                cached_pk = npk;
                has_cached_pk = true;
                next_w = down_w_ptr(npk, npk & 0xFFU, ncol);
                next_tc = d_tile;
            } else if (d_rem6 > 0 && blk_begin == 0 && blk_count == 24) {
                int64_t rord = -1, rcol = 0, rtile = 0;
                rem_down_task(d_rem6, rem_base, tid, rord, rcol, rtile);
                if (rord >= 0) {
                    const uint32_t npk = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + rord));
                    next_w = down_w_ptr(npk, npk & 0xFFU, rcol);
                    next_tc = rtile;
                }
            }
            Project(mm,
                    Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                    down_w_ptr(packed, expert, column),
                    ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * begin * kHidden + column,
                    down_s_ptr(packed, expert, column),
                    d_tile, kHidden, kIntermediate, sub_count, next_w, next_tc);
#else
            (void)d_rem6;
            (void)rem_base;
            Project(mm,
                    Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                    down_w_ptr(packed, expert, column),
                    ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * begin * kHidden + column,
                    down_s_ptr(packed, expert, column),
                    d_tile, kHidden, kIntermediate, sub_count);
#endif
#ifdef K112600_PIPE_C2V
#ifndef K112612_NO_FLAG9
            if (sig_wave0 && task == tid) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(9);
            }
#else
            (void)sig_wave0;
#endif
#else
            (void)sig_wave0;
#endif
        }
    };
    auto run_down_waves = [&](auto& mm, int64_t batch_begin, int64_t batch_count) __aicore__ {
        if (batch_count <= 0) return;
        const int64_t d_full6 = static_cast<int64_t>((static_cast<uint32_t>(batch_count) * 683U) >> 12U) * 6;
        const int64_t d_rem6 = batch_count - d_full6;
        const int64_t rem_base = batch_begin + d_full6;
        run_down_range(std::integral_constant<int64_t, 4>{}, mm, batch_begin, d_full6, d_rem6, rem_base, (batch_begin == 0 && pipe_wave0_s));
        if (d_rem6 > 0) {
            const int64_t tid = GetBlockIdx();
            int64_t ordinal = -1, column = 0, d_tile = 0;
            rem_down_task(d_rem6, rem_base, tid, ordinal, column, d_tile);
            if (ordinal >= 0) {
                const uint32_t packed = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + ordinal));
                const int64_t expert = packed & 0xFFU;
                const int64_t begin = (packed >> 8) & 0xFFU;
                const int64_t sub_count = (packed >> 16) & 0xFFU;
#if defined(K11080_DIRECT_W4A8) && defined(K11220_BITMASK_OWN_P1)
                const uint32_t utid = static_cast<uint32_t>(tid);
                const uint32_t q5 = (utid * 205U) >> 10U;
                const bool has_sub4_tail = (d_rem6 == 5 && tid < 20 && (tid - static_cast<int64_t>(q5 * 5U)) == 4);
                if (has_sub4_tail) {
                    const uint32_t pk4 = static_cast<uint32_t>(active_experts.GetValue(kActiveHeaderElements + rem_base + 4));
                    const int64_t exp4 = pk4 & 0xFFU;
                    const int64_t beg4 = (pk4 >> 8) & 0xFFU;
                    const int64_t sc4 = (pk4 >> 16) & 0xFFU;
                    const int64_t col4 = 5120 + static_cast<int64_t>(q5) * 256;
#ifdef K11220_XTASK_B
                    Project(mm,
                            Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                            down_w_ptr(packed, expert, column),
                            ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * begin * kHidden + column,
                            down_s_ptr(packed, expert, column),
                            d_tile, kHidden, kIntermediate, sub_count,
                            down_w_ptr(pk4, exp4, col4), 256);
#else
                    Project(mm,
                            Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                            down_w_ptr(packed, expert, column),
                            ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * begin * kHidden + column,
                            down_s_ptr(packed, expert, column),
                            d_tile, kHidden, kIntermediate, sub_count);
#endif
                    Project(mm,
                            Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * beg4 * kIntermediate),
                            down_w_ptr(pk4, exp4, col4),
                            ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * beg4 * kHidden + col4,
                            down_s_ptr(pk4, exp4, col4),
                            256, kHidden, kIntermediate, sc4);
                } else {
                    Project(mm,
                            Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                            down_w_ptr(packed, expert, column),
                            ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * begin * kHidden + column,
                            down_s_ptr(packed, expert, column),
                            d_tile, kHidden, kIntermediate, sub_count);
                }
#else
                Project(mm,
                        Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                        down_w_ptr(packed, expert, column),
                        ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]) + 2 * begin * kHidden + column,
                        down_s_ptr(packed, expert, column),
                        d_tile, kHidden, kIntermediate, sub_count);
#endif
            }
        }
    };
    auto run_activate_subrange = [&](int32_t pos_beg, int32_t pos_end) __aicore__ {
#ifdef K11220_FAST_ACT_P3
        if (pos_beg >= pos_end) return;
        const int32_t w_step = static_cast<int32_t>(2 * arguments.blocks);
        for (int32_t pos = pos_beg + static_cast<int32_t>(GetBlockIdx()); pos < pos_end; pos += w_step) {
            const uint32_t apk = act_pack_s[pos];
            const int64_t expert = apk & 0x3FU;
            const int64_t begin = (apk >> 6U) & 0xFFU;
            const int64_t row = (apk >> 14U) & 0x1FU;
            const int64_t count = ((apk >> 19U) & 0x1FU) + 1;
            const int64_t src_row = (apk >> 24U) & 0x1FU;
            __gm__ float* act_corr = ((__gm__ float*)arguments.weights[1]) + expert * kGateUp;
            __gm__ ScaleStorage* act_scale = ((__gm__ ScaleStorage*)arguments.weights[2]) + expert * kGateUp;
            if (pw_active) {
                const uint32_t lslot = my_gexp[expert];
                if (lslot < 32U) {
                    act_corr = ((__gm__ float*)arguments.weights[1]) + static_cast<int64_t>(lslot) * kGateUp;
                    act_scale = ((__gm__ ScaleStorage*)arguments.weights[2]) + static_cast<int64_t>(lslot) * kGateUp;
                } else {
                    const int64_t ms = static_cast<int64_t>(lslot - 32U);
                    act_corr = reinterpret_cast<__gm__ float*>(mw_s[1U]) + ms * kGateUp;
                    act_scale = reinterpret_cast<__gm__ ScaleStorage*>(mw_s[2U]) + ms * kGateUp;
                }
            }
            ActivateRowFast(held, count, row, src_row,
                ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp,
                act_corr, act_scale,
                Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                ((__gm__ float*)arguments.scratch[kMiddleScale]) + begin,
                (pos + w_step >= pos_end));
        }
#ifdef K112612_TRIM_BARRIERS
        PipeBarrier<PIPE_ALL>();
#else
        Publish();
#endif
#else
        (void)pos_beg;
        (void)pos_end;
#endif
    };
    (void)run_activate_subrange;
    auto run_activate_batch = [&](int64_t ord_begin, int64_t ord_end) __aicore__ {
        if (ord_begin >= ord_end) return;
        const int64_t workers = 2 * arguments.blocks;
#ifdef K11220_FAST_ACT_P3
        if (act_total_s >= 0 && ord_begin == 0 && ord_end == active_count) {
#ifdef K112600_PIPE_C2V
            run_activate_subrange(pipe_wave0_s ? split_act_pos_s : 0, act_total_s);
            return;
#endif
            const int32_t w_step = static_cast<int32_t>(workers);
            for (int32_t pos = static_cast<int32_t>(GetBlockIdx()); pos < act_total_s; pos += w_step) {
                const uint32_t apk = act_pack_s[pos];
                const int64_t expert = apk & 0x3FU;
                const int64_t begin = (apk >> 6U) & 0xFFU;
                const int64_t row = (apk >> 14U) & 0x1FU;
                const int64_t count = ((apk >> 19U) & 0x1FU) + 1;
                const int64_t src_row = (apk >> 24U) & 0x1FU;
                __gm__ float* act_corr = ((__gm__ float*)arguments.weights[1]) + expert * kGateUp;
                __gm__ ScaleStorage* act_scale = ((__gm__ ScaleStorage*)arguments.weights[2]) + expert * kGateUp;
                if (pw_active) {
                    const uint32_t lslot = my_gexp[expert];
                    if (lslot < 32U) {
                        act_corr = ((__gm__ float*)arguments.weights[1]) + static_cast<int64_t>(lslot) * kGateUp;
                        act_scale = ((__gm__ ScaleStorage*)arguments.weights[2]) + static_cast<int64_t>(lslot) * kGateUp;
                    } else {
                        const int64_t ms = static_cast<int64_t>(lslot - 32U);
                        act_corr = reinterpret_cast<__gm__ float*>(mw_s[1U]) + ms * kGateUp;
                        act_scale = reinterpret_cast<__gm__ ScaleStorage*>(mw_s[2U]) + ms * kGateUp;
                    }
                }
                ActivateRowFast(held, count, row, src_row,
                    ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp,
                    act_corr, act_scale,
                    Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                    ((__gm__ float*)arguments.scratch[kMiddleScale]) + begin,
                    (pos + w_step >= act_total_s));
            }
#ifdef K112612_TRIM_BARRIERS
            PipeBarrier<PIPE_ALL>();
#else
            Publish();
#endif
            return;
        }
#endif
        int64_t next = GetBlockIdx();
        int64_t prev_expert = -1;
        for (int64_t ordinal = ord_begin; ordinal < ord_end; ++ordinal) {
            const int64_t expert = held.active.GetValue(kActiveHeaderElements + ordinal) & 0xFF;
            if (expert == prev_expert) {
                continue;
            }
            prev_expert = expert;
            const int64_t count = held.count.GetValue(expert);
            const int64_t begin = held.start.GetValue(expert);
            while (next < begin) {
                next += workers;
            }
            if (next >= begin + count) {
                continue;
            }
            __gm__ float* act_corr = ((__gm__ float*)arguments.weights[1]) + expert * kGateUp;
            __gm__ ScaleStorage* act_scale = ((__gm__ ScaleStorage*)arguments.weights[2]) + expert * kGateUp;
            if (pw_active) {
                const uint32_t lslot = my_gexp[expert];
                if (lslot < 32U) {
                    act_corr = ((__gm__ float*)arguments.weights[1]) + static_cast<int64_t>(lslot) * kGateUp;
                    act_scale = ((__gm__ ScaleStorage*)arguments.weights[2]) + static_cast<int64_t>(lslot) * kGateUp;
                } else {
                    const int64_t ms = static_cast<int64_t>(lslot - 32U);
                    act_corr = reinterpret_cast<__gm__ float*>(mw_s[1U]) + ms * kGateUp;
                    act_scale = reinterpret_cast<__gm__ ScaleStorage*>(mw_s[2U]) + ms * kGateUp;
                }
            }
            Activate<true>(held, count, begin,
                ((__gm__ RoutedProduct*)arguments.scratch[kGateProduct]) + GateStorageRow(begin) * kGateUp,
                act_corr, act_scale,
                Nibbles(((__gm__ int4b_t*)arguments.scratch[kMiddle]), 2 * begin * kIntermediate),
                ((__gm__ float*)arguments.scratch[kMiddleScale]) + begin, next - begin, workers);
            do {
                next += workers;
            } while (next < begin + count);
        }
        Publish();
    };
    auto run_shared_store = [&]() __aicore__ {
#ifdef ROUTED_SHARED_PIPELINE
        __gm__ bfloat16_t* sh_dst_base = nullptr;
        bool sh_quad4 = false;
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
        sh_quad4 = whole_layer_tree::IsQuad4Active(comm_capacity, arguments.rows);
        const auto layout = sh_quad4
            ? whole_layer_tree::MakeQuad4Layout(comm_capacity, comm_ranks, arguments.rows)
            : whole_layer_tree::MakeCommunicationLayout(comm_capacity, comm_ranks);
        const uint32_t p_mask = sh_quad4 ? 3U : 1U;
        const uint32_t sh_epoch = comm_base_epoch + 1U;
        if (comm_window != nullptr && !combine_shared) {
#ifdef K11220_RS768_PUSH
            const int64_t sh_rs_qcap = whole_layer_tree::RsQCap(comm_capacity, arguments.rows, sh_quad4);
            sh_dst_base = reinterpret_cast<__gm__ bfloat16_t*>(
                comm_window + layout.payload + (sh_epoch & p_mask) * layout.plane_bytes)
                + static_cast<int64_t>(comm_rank) * (sh_rs_qcap * 768);
#elif defined(K112600_RS768_HIW)
            if (arguments.rows > 16) {
                const int64_t sh_rs_qcap = whole_layer_tree::RsQCap(comm_capacity, arguments.rows, sh_quad4);
                sh_dst_base = reinterpret_cast<__gm__ bfloat16_t*>(
                    comm_window + layout.payload + (sh_epoch & p_mask) * layout.plane_bytes)
                    + static_cast<int64_t>(comm_rank) * (sh_rs_qcap * 768);
            } else {
                sh_dst_base = reinterpret_cast<__gm__ bfloat16_t*>(
                    comm_window + layout.payload + (sh_epoch & p_mask) * layout.plane_bytes
                    + comm_rank * layout.rank_bytes);
            }
#else
            sh_dst_base = reinterpret_cast<__gm__ bfloat16_t*>(
                comm_window + layout.payload + (sh_epoch & p_mask) * layout.plane_bytes
                + comm_rank * layout.rank_bytes);
#endif
        }
#endif
        shared.Store(held, sh_dst_base, comm_rank, comm_ranks, sh_quad4 && (sh_dst_base != nullptr));
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
#if !(defined(K112612_DEFER_SH_SIG) && !defined(K112612_OVERLAP_SH_P4))
        if ASCEND_IS_AIV {
            if (comm_window != nullptr && !combine_shared) {
                AscendC::SyncAll<true>();
                const int32_t off = static_cast<int32_t>(AscendC::GetBlockIdx());
                if (off > 0 && off < comm_ranks) {
                    const int32_t peer = (comm_rank + off) & (comm_ranks - 1);
                    __gm__ int32_t* remote_sig = reinterpret_cast<__gm__ int32_t*>(
                        aclshmem_ptr(whole_layer_tree::CommunicationReady(
                            comm_window, layout, comm_ranks, sh_epoch, comm_rank), peer));
                    *remote_sig = static_cast<int32_t>(sh_epoch);
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_sig));
                }
            }
        }
#endif
#endif
#endif
    };

    int64_t split_ord = 0;
#ifdef K11050_MOE_PIPE2
    if (!combine_shared && active_count > 6) {
        auto pick_split = [&](const auto& act_ten) __aicore__ -> int64_t {
            const uint32_t ac = static_cast<uint32_t>(active_count);
            const uint32_t tot_waves = ((ac + 5U) * 683U) >> 12U;
            int64_t s = (ac >= 12U) ? 12 : 6;
            const int32_t exp_hi = act_ten.GetValue(kActiveHeaderElements + s) & 0xFF;
            while (s > 0 && (act_ten.GetValue(kActiveHeaderElements + s - 1) & 0xFF) == exp_hi) {
                --s;
            }
            if (s <= 0) {
                s = (ac >= 12U) ? 12 : 6;
                while (s < active_count && (act_ten.GetValue(kActiveHeaderElements + s) & 0xFF) == exp_hi) {
                    ++s;
                }
            }
            if (s <= 0 || s >= active_count) {
                return 0;
            }
            const uint32_t su = static_cast<uint32_t>(s);
            const uint32_t w1 = ((su + 5U) * 683U) >> 12U;
            const uint32_t w2 = ((ac - su + 5U) * 683U) >> 12U;
            return (w1 + w2 <= tot_waves) ? s : 0;
        };
        if ASCEND_IS_AIC {
            split_ord = pick_split(active_experts);
        }
        if ASCEND_IS_AIV {
            split_ord = pick_split(held.active);
        }
    }
#endif
    if (split_ord > 0) {
#ifdef ROUTED_SHARED_PIPELINE
        shared.Activate();
#endif
        if ASCEND_IS_AIC {
            pipe.Reset();
#ifdef K11080_DIRECT_W4A8
            {
                DirectW4A8Runner16<kHidden, 256, 1536, true> gate_mm16;
                gate_mm16.Init(pipe);
                run_gate_waves(gate_mm16, 0, split_ord);
                gate_mm16.End();
            }
            Publish();
            SyncAll<false>();
            pipe.Reset();
            {
                DirectW4A8Runner16<kHidden, 256, 1536, true> gate_mm16;
                gate_mm16.Init(pipe);
                run_gate_waves(gate_mm16, split_ord, active_count - split_ord);
                gate_mm16.End();
            }
            Publish();
#else
            TCubeTiling gate_cube = TCubeTiling{};
            gate_cube.M = kMaxStackedRows;
            gate_cube.singleCoreM = kMaxStackedRows;
            gate_cube.singleCoreN = kGateUp;
            GateProjection16<weight_format> gate_mm16;
            gate_mm16.Init(&gate_cube, &pipe);
            run_gate_waves(gate_mm16, 0, split_ord);
            gate_mm16.End();
            Publish();
            SyncAll<false>();
            pipe.Reset();
            GateProjection16<weight_format> gate_mm16_b2;
            gate_mm16_b2.Init(&gate_cube, &pipe);
            run_gate_waves(gate_mm16_b2, split_ord, active_count - split_ord);
            gate_mm16_b2.End();
            Publish();
#endif
        }
#ifdef ROUTED_SHARED_PIPELINE
        shared.Down(pipe);
#endif
        if ASCEND_IS_AIV {
            SyncAll<false>();
            run_activate_batch(0, split_ord);
        }
        timing.Barrier(2);
        if (moe_prof != nullptr) { mp2 = GetSystemCycle(); }
        timing.Begin(3);
        if ASCEND_IS_AIC {
            pipe.Reset();
#ifdef K11080_DIRECT_W4A8
            DirectW4A8Runner16<kIntermediate, 256, 1024, false> down_mm16;
            down_mm16.Init(pipe);
#else
            TCubeTiling down_cube = TCubeTiling{};
            down_cube.M = kMaxStackedRows;
            down_cube.singleCoreM = kMaxStackedRows;
            down_cube.singleCoreN = kHidden;
            DownProjection16<weight_format> down_mm16;
            down_mm16.Init(&down_cube, &pipe);
#endif
            run_down_waves(down_mm16, 0, split_ord);
            down_mm16.End();
            Publish();
        }
        if ASCEND_IS_AIV {
            run_activate_batch(split_ord, active_count);
            PipeBarrier<PIPE_ALL>();
            run_shared_store();
            PipeBarrier<PIPE_ALL>();
        }
        timing.Barrier(3);
        if (moe_prof != nullptr) { mp3 = GetSystemCycle(); }
        timing.Begin(4);
        if ASCEND_IS_AIC {
            pipe.Reset();
#ifdef K11080_DIRECT_W4A8
            DirectW4A8Runner16<kIntermediate, 256, 1024, false> down_mm16;
            down_mm16.Init(pipe);
#else
            TCubeTiling down_cube = TCubeTiling{};
            down_cube.M = kMaxStackedRows;
            down_cube.singleCoreM = kMaxStackedRows;
            down_cube.singleCoreN = kHidden;
            DownProjection16<weight_format> down_mm16;
            down_mm16.Init(&down_cube, &pipe);
#endif
            run_down_waves(down_mm16, split_ord, active_count - split_ord);
            down_mm16.End();
            Publish();
        }
        if ASCEND_IS_AIV {
            PipeBarrier<PIPE_ALL>();
            const int32_t split_exp = held.active.GetValue(kActiveHeaderElements + split_ord) & 0xFF;
            TokenOutputs(pipe, held, ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]), ((__gm__ float*)arguments.weights[4]), ((__gm__ ScaleStorage*)arguments.weights[5]), ((__gm__ float*)arguments.scratch[kMiddleScale]),
                (__gm__ int32_t*)arguments.dispatch.expert_ids, ((__gm__ bfloat16_t*)arguments.scratch[kRouteOutput]), ((__gm__ bfloat16_t*)arguments.scratch[kOutput]),
                (__gm__ bfloat16_t*)arguments.shared.output, scaled_output,
                combined_output, arguments.rows, arguments.blocks, expert_offset, combine_shared, saved_local_mask, comm_window, comm_capacity, comm_rank, comm_ranks, comm_base_epoch, 0, split_exp, my_gexp, mw_ptr);
#ifdef K11120_SKIP_DUP_PUB
            if (combine_shared || comm_window == nullptr) {
                Publish();
            }
#else
            Publish();
#endif
        }
        timing.Barrier(4);
        if (moe_prof != nullptr) { mp4 = GetSystemCycle(); }
        timing.Begin(5);
        if ASCEND_IS_AIV {
            const int32_t split_exp = held.active.GetValue(kActiveHeaderElements + split_ord) & 0xFF;
            TokenOutputs(pipe, held, ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]), ((__gm__ float*)arguments.weights[4]), ((__gm__ ScaleStorage*)arguments.weights[5]), ((__gm__ float*)arguments.scratch[kMiddleScale]),
                (__gm__ int32_t*)arguments.dispatch.expert_ids, ((__gm__ bfloat16_t*)arguments.scratch[kRouteOutput]), ((__gm__ bfloat16_t*)arguments.scratch[kOutput]),
                (__gm__ bfloat16_t*)arguments.shared.output, scaled_output,
                combined_output, arguments.rows, arguments.blocks, expert_offset, combine_shared, saved_local_mask, comm_window, comm_capacity, comm_rank, comm_ranks, comm_base_epoch, split_exp, kLocalExperts, my_gexp, mw_ptr);
#ifdef K11120_SKIP_DUP_PUB
            if (combine_shared || comm_window == nullptr) {
                Publish();
            }
#else
            Publish();
#endif
        }
    } else {
#ifdef ROUTED_SHARED_PIPELINE
        shared.Activate();
#endif
#ifdef K112600_PIPE_C2V
        if ASCEND_IS_AIV {
            if (pipe_wave0_s) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_S>(8);
                AscendC::SyncAll<true>();
                run_activate_subrange(0, split_act_pos_s);
            }
        }
#endif
        if ASCEND_IS_AIC {
            if (active_count > 0) {
                pipe.Reset();
#ifdef K11080_DIRECT_W4A8
                DirectW4A8Runner16<kHidden, 256, 1536, true> gate_mm16;
                gate_mm16.Init(pipe);
#else
                TCubeTiling gate_cube = TCubeTiling{};
                gate_cube.M = kMaxStackedRows;
                gate_cube.singleCoreM = kMaxStackedRows;
                gate_cube.singleCoreN = kGateUp;
                GateProjection16<weight_format> gate_mm16;
                gate_mm16.Init(&gate_cube, &pipe);
#endif
                run_gate_waves(gate_mm16, 0, active_count);
                gate_mm16.End();
#ifndef K112612_TRIM_BARRIERS
                Publish();
#endif
            }
        }
        timing.Barrier(2);
        if (moe_prof != nullptr) { mp2 = GetSystemCycle(); }
        timing.Begin(3);
#ifdef ROUTED_SHARED_PIPELINE
        shared.Down(pipe);
#endif
        if ASCEND_IS_AIV {
            run_activate_batch(0, active_count);
        }
        timing.Barrier(3);
        if (moe_prof != nullptr) { mp3 = GetSystemCycle(); }
        timing.Begin(4);
        run_shared_store();
#if defined(K112612_OVERLAP_SH_P4) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if ASCEND_IS_AIV {
            if (comm_window != nullptr && !combine_shared && scaled_output != nullptr && arguments.rows > 0) {
                const uint32_t tid_sh = static_cast<uint32_t>(GetBlockIdx());
                if (tid_sh < 48U) {
                    const bool sh_q4 = whole_layer_tree::IsQuad4Active(comm_capacity, arguments.rows);
                    const auto sh_lay = sh_q4
                        ? whole_layer_tree::MakeQuad4Layout(comm_capacity, comm_ranks, arguments.rows)
                        : whole_layer_tree::MakeCommunicationLayout(comm_capacity, comm_ranks);
                    const uint32_t sh_pm = sh_q4 ? 3U : 1U;
                    const uint32_t sh_ep = comm_base_epoch + 1U;
                    const int32_t sh_sig = static_cast<int32_t>(sh_ep);
                    const int64_t sh_rstride = sh_lay.rank_bytes / static_cast<int64_t>(sizeof(bfloat16_t));
                    __gm__ bfloat16_t* sh_pl = reinterpret_cast<__gm__ bfloat16_t*>(
                        comm_window + sh_lay.payload + (sh_ep & sh_pm) * sh_lay.plane_bytes);
                    __gm__ bfloat16_t* sh_twin_pl = (arguments.rows > 16 && sh_q4)
                        ? reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(sh_pl, comm_rank ^ 1))
                        : sh_pl;
                    __gm__ bfloat16_t* sh_self_out = (__gm__ bfloat16_t*)arguments.shared.output;
                    const auto sh_plan = whole_layer_tree::GetTreePlan(comm_ranks);
                    uint32_t sh_unr = (comm_ranks >= 32) ? 0xFFFFFFFFU : ((1U << static_cast<uint32_t>(comm_ranks)) - 1U);
                    for (int32_t lf = 0; lf < comm_ranks; ++lf) {
                        if (static_cast<int32_t>(whole_layer_tree::GetTreeStep(sh_plan, lf).rank) == comm_rank) {
                            sh_unr &= ~(1U << lf);
                        }
                    }
                    auto map_sh_wave8 = [&](int64_t w_rows, int64_t& r_tok, int64_t& r_col, int64_t& r_chk) __aicore__ {
                        if (w_rows <= 1) {
                            r_chk = 128; r_tok = 0; r_col = static_cast<int64_t>(tid_sh) * 128;
                        } else if (w_rows == 2) {
                            r_chk = 256; const uint32_t q = (tid_sh * 683U) >> 14U;
                            r_tok = static_cast<int64_t>(q); r_col = static_cast<int64_t>(tid_sh - q * 24U) * 256;
                        } else if (w_rows == 3) {
                            r_chk = 384; r_tok = static_cast<int64_t>(tid_sh >> 4U); r_col = static_cast<int64_t>(tid_sh & 15U) * 384;
                        } else if (w_rows == 4) {
                            r_chk = 512; const uint32_t q = (tid_sh * 683U) >> 13U;
                            r_tok = static_cast<int64_t>(q); r_col = static_cast<int64_t>(tid_sh - q * 12U) * 512;
                        } else if (w_rows <= 6) {
                            r_chk = 768; r_tok = static_cast<int64_t>(tid_sh >> 3U); r_col = static_cast<int64_t>(tid_sh & 7U) * 768;
                        } else {
                            r_chk = 1024; const uint32_t q = (tid_sh * 683U) >> 12U;
                            r_tok = static_cast<int64_t>(q); r_col = static_cast<int64_t>(tid_sh - q * 6U) * 1024;
                        }
                    };
                    const int64_t sh_wave_step = (comm_ranks > 8) ? 4 : 8;
                    for (int64_t row_base = 0; row_base < arguments.rows; row_base += sh_wave_step) {
                        const int64_t w_rows = (arguments.rows - row_base > sh_wave_step) ? sh_wave_step : (arguments.rows - row_base);
                        int64_t rel_t = 0, c_off = 0, chk = 1024;
                        map_sh_wave8(w_rows, rel_t, c_off, chk);
                        if (rel_t >= w_rows) {
                            break;
                        }
                        if (row_base > 0) {
                            Fence<HardEvent::MTE3_MTE2>();
                            Fence<HardEvent::V_MTE2>();
                        }
                        const int64_t sh_off = (row_base + rel_t) * kHidden + c_off;
                        uint32_t pend = sh_unr;
                        for (int32_t lf = 0; lf < comm_ranks; ++lf) {
                            const auto stp = whole_layer_tree::GetTreeStep(sh_plan, lf);
                            if (static_cast<int32_t>(stp.rank) == comm_rank) {
                                GlobalTensor<bfloat16_t> s_gm;
                                if (sh_q4) {
                                    s_gm.SetGlobalBuffer(sh_pl + static_cast<int64_t>(stp.rank) * sh_rstride + sh_off, chk);
                                } else {
                                    s_gm.SetGlobalBuffer(sh_self_out + sh_off, chk);
                                }
                                DataCopy(held.sh_red_nar[lf * chk], s_gm, chk);
                            } else if ((pend & (1U << lf)) == 0U) {
                                GlobalTensor<bfloat16_t> s_gm;
                                __gm__ bfloat16_t* sh_b = (arguments.rows > 16 && sh_q4 && (((static_cast<int32_t>(stp.rank) ^ comm_rank) & 1) != 0))
                                    ? sh_twin_pl : sh_pl;
                                s_gm.SetGlobalBuffer(sh_b + static_cast<int64_t>(stp.rank) * sh_rstride + sh_off, chk);
                                DataCopy(held.sh_red_nar[lf * chk], s_gm, chk);
                            }
                        }
                        while (pend != 0U) {
                            for (int32_t l = 0; l < comm_ranks; ++l) {
                                const int32_t lf = (l + static_cast<int32_t>(tid_sh)) & (comm_ranks - 1);
                                if ((pend & (1U << lf)) == 0U) {
                                    continue;
                                }
                                const auto stp = whole_layer_tree::GetTreeStep(sh_plan, lf);
                                __gm__ volatile int32_t* s_addr =
                                    whole_layer_tree::CommunicationReady(comm_window, sh_lay, comm_ranks, sh_ep, stp.rank);
                                dcci_cacheline((__gm__ uint8_t*)s_addr);
                                if (*s_addr == sh_sig) {
                                    GlobalTensor<bfloat16_t> s_gm;
                                    __gm__ bfloat16_t* sh_b = (arguments.rows > 16 && sh_q4 && (((static_cast<int32_t>(stp.rank) ^ comm_rank) & 1) != 0))
                                        ? sh_twin_pl : sh_pl;
                                    s_gm.SetGlobalBuffer(sh_b + static_cast<int64_t>(stp.rank) * sh_rstride + sh_off, chk);
                                    DataCopy(held.sh_red_nar[lf * chk], s_gm, chk);
                                    pend &= ~(1U << lf);
                                }
                            }
                        }
                        sh_unr = 0U;
                        Fence<HardEvent::MTE2_V>();
                        Cast(held.sh_red_acc[0], held.sh_red_nar[0], RoundMode::CAST_NONE, comm_ranks * chk);
                        PipeBarrier<PIPE_V>();
                        for (int32_t stride = 1; stride < comm_ranks; stride <<= 1) {
                            for (int32_t i = 0; i + stride < comm_ranks; i += (stride << 1)) {
                                Add(held.sh_red_acc[i * chk], held.sh_red_acc[i * chk], held.sh_red_acc[(i + stride) * chk], chk);
                            }
                            PipeBarrier<PIPE_V>();
                        }
                        Cast(held.sh_red_nar[0], held.sh_red_acc[0], RoundMode::CAST_RINT, chk);
                        Fence<HardEvent::V_MTE3>();
                        GlobalTensor<bfloat16_t> d_gm;
                        d_gm.SetGlobalBuffer(scaled_output + sh_off, chk);
                        DataCopy(d_gm, held.sh_red_nar[0], chk);
                    }
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
#endif
#if defined(K112600_PIPE_C2V) && !defined(K112612_NO_FLAG9)
        if ASCEND_IS_AIV {
            if (pipe_wave0_s) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_S>(9);
                AscendC::SyncAll<true>();
                if (split_local_cnt_s > 0) {
                    TokenOutputs(pipe, held, ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]),
                        ((__gm__ float*)arguments.weights[4]), ((__gm__ ScaleStorage*)arguments.weights[5]),
                        ((__gm__ float*)arguments.scratch[kMiddleScale]),
                        (__gm__ int32_t*)arguments.dispatch.expert_ids,
                        ((__gm__ bfloat16_t*)arguments.scratch[kRouteOutput]),
                        ((__gm__ bfloat16_t*)arguments.scratch[kOutput]),
                        (__gm__ bfloat16_t*)arguments.shared.output, scaled_output,
                        combined_output, arguments.rows, arguments.blocks, expert_offset, combine_shared,
                        saved_local_mask, comm_window, comm_capacity, comm_rank, comm_ranks, comm_base_epoch,
                        0, split_ord6_exp_s, my_gexp, mw_ptr, local_origs_s, split_local_cnt_s);
                }
            }
        }
#endif
        if ASCEND_IS_AIC {
            if (active_count > 0) {
                pipe.Reset();
#ifdef K11080_DIRECT_W4A8
                DirectW4A8Runner16<kIntermediate, 256, 1024, false> down_mm16;
                down_mm16.Init(pipe);
#else
                TCubeTiling down_cube = TCubeTiling{};
                down_cube.M = kMaxStackedRows;
                down_cube.singleCoreM = kMaxStackedRows;
                down_cube.singleCoreN = kHidden;
                DownProjection16<weight_format> down_mm16;
                down_mm16.Init(&down_cube, &pipe);
#endif
                run_down_waves(down_mm16, 0, active_count);
                down_mm16.End();
#ifndef K112612_TRIM_BARRIERS
                Publish();
#endif
            }
        }
        timing.Barrier(4);
#if defined(K112612_DEFER_SH_SIG) && !defined(K112612_OVERLAP_SH_P4) && defined(ROUTED_SHARED_PIPELINE) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
        if ASCEND_IS_AIV {
            if (comm_window != nullptr && !combine_shared) {
                const int32_t off = static_cast<int32_t>(AscendC::GetBlockIdx());
                if (off > 0 && off < comm_ranks) {
                    const bool sh_q4_d = whole_layer_tree::IsQuad4Active(comm_capacity, arguments.rows);
                    const auto sh_lay_d = sh_q4_d
                        ? whole_layer_tree::MakeQuad4Layout(comm_capacity, comm_ranks, arguments.rows)
                        : whole_layer_tree::MakeCommunicationLayout(comm_capacity, comm_ranks);
                    const uint32_t sh_ep_d = comm_base_epoch + 1U;
                    const int32_t peer = (comm_rank + off) & (comm_ranks - 1);
                    __gm__ int32_t* remote_sig = reinterpret_cast<__gm__ int32_t*>(
                        aclshmem_ptr(whole_layer_tree::CommunicationReady(
                            comm_window, sh_lay_d, comm_ranks, sh_ep_d, comm_rank), peer));
                    *remote_sig = static_cast<int32_t>(sh_ep_d);
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_sig));
                }
            }
        }
#endif
        if (moe_prof != nullptr) { mp4 = GetSystemCycle(); }
        timing.Begin(5);
        if ASCEND_IS_AIV {
            TokenOutputs(pipe, held, ((__gm__ RoutedProduct*)arguments.scratch[kDownProduct]), ((__gm__ float*)arguments.weights[4]), ((__gm__ ScaleStorage*)arguments.weights[5]), ((__gm__ float*)arguments.scratch[kMiddleScale]),
                (__gm__ int32_t*)arguments.dispatch.expert_ids, ((__gm__ bfloat16_t*)arguments.scratch[kRouteOutput]), ((__gm__ bfloat16_t*)arguments.scratch[kOutput]),
                (__gm__ bfloat16_t*)arguments.shared.output, scaled_output,
#if defined(K112600_PIPE_C2V) && !defined(K112612_NO_FLAG9)
                combined_output, arguments.rows, arguments.blocks, expert_offset, combine_shared, saved_local_mask, comm_window, comm_capacity, comm_rank, comm_ranks, comm_base_epoch, (pipe_wave0_s && split_local_cnt_s > 0) ? split_ord6_exp_s : 0, kLocalExperts, my_gexp, mw_ptr, (pipe_wave0_s && split_local_cnt_s > 0) ? (local_origs_s + split_local_cnt_s) : local_origs_s, (pipe_wave0_s && split_local_cnt_s > 0) ? (local_cnt_s - split_local_cnt_s) : local_cnt_s);
#else
                combined_output, arguments.rows, arguments.blocks, expert_offset, combine_shared, saved_local_mask, comm_window, comm_capacity, comm_rank, comm_ranks, comm_base_epoch, 0, kLocalExperts, my_gexp, mw_ptr, local_origs_s, local_cnt_s);
#endif
#ifdef K11120_SKIP_DUP_PUB
            if (combine_shared || comm_window == nullptr) {
                Publish();
            }
#else
            Publish();
#endif
        }
    }
    timing.Begin(6);
    timing.FinishFinal(6);
    if (moe_prof != nullptr) {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                mp5 = GetSystemCycle();
                moe_prof[0] += (mp1 - mp0);
                moe_prof[1] += (mp2 - mp1);
                moe_prof[2] += (mp3 - mp2);
                moe_prof[3] += (mp4 - mp3);
                moe_prof[4] += (mp5 - mp4);
                const int64_t ac = active_count;
                moe_prof[5] += ac;
                if (ac > moe_prof[6]) { moe_prof[6] = ac; }
                moe_prof[7] += saved_local_mask ? __builtin_popcount(saved_local_mask) : 0;
            }
        }
    }
    if (record_prof) timing.Store(((__gm__ int64_t*)arguments.scratch[kGateTiling]) + 48, arguments.rows, arguments.blocks);
}

}

#endif
