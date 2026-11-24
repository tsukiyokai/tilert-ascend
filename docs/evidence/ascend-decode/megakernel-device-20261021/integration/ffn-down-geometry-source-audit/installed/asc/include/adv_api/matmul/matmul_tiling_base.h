/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_tiling_base.h
 * \brief
 */

#ifndef LIB_MATMUL_MATMUL_TILING_BASE_H
#define LIB_MATMUL_MATMUL_TILING_BASE_H

#include "matmul_tilingdata.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace matmul_tiling {
#if !defined(__NPU_DEVICE__) && !defined(__NPU_HOST__) && !defined(__ASCC_HOST__) && !defined(__ASCC_DEVICE__)
using half = double;
#endif // !defined(__NPU_DEVICE__) && !defined(__NPU_HOST__) && !defined(__ASCC_HOST__) && !defined(__ASCC_DEVICE__)
constexpr int32_t UINT8_BYTES = 1;
constexpr int32_t INT8_BYTES = 1;
constexpr int32_t FP32_BYTES = 4;
constexpr int32_t FP16_BYTES = 2;
constexpr int32_t C0_SIZE = 16;
constexpr int32_t C0_BYTE_SIZE = 32;
constexpr int32_t BITS_PER_BYTE = 8;

enum class DataType : int32_t {
    DT_FLOAT = 0,           // float type
    DT_FLOAT16 = 1,         // fp16 type
    DT_INT8 = 2,            // int8 type
    DT_INT16 = 6,           // int16 type
    DT_UINT16 = 7,          // uint16 type
    DT_UINT8 = 4,           // uint8 type
    DT_INT32 = 3,           // int32 type
    DT_INT64 = 9,           // int64 type
    DT_UINT32 = 8,          // unsigned int32
    DT_UINT64 = 10,         // unsigned int64
    DT_BOOL = 12,           // bool type
    DT_DOUBLE = 11,         // double type
    DT_STRING = 13,         // std::string type
    DT_DUAL_SUB_INT8 = 14,  // dual output int8 type
    DT_DUAL_SUB_UINT8 = 15, // dual output uint8 type
    DT_COMPLEX64 = 16,      // complex64 type
    DT_COMPLEX128 = 17,     // complex128 type
    DT_QINT8 = 18,          // qint8 type
    DT_QINT16 = 19,         // qint16 type
    DT_QINT32 = 20,         // qint32 type
    DT_QUINT8 = 21,         // quint8 type
    DT_QUINT16 = 22,        // quint16 type
    DT_RESOURCE = 23,       // resource type
    DT_STRING_REF = 24,     // std::string ref type
    DT_DUAL = 25,           // dual output type
    DT_VARIANT = 26,        // dt_variant type
    DT_BF16 = 27,           // bf16 type
    DT_UNDEFINED = 28,      // Used to indicate a DataType field has not been set.
    DT_INT4 = 29,           // int4 type
    DT_UINT1 = 30,          // uint1 type
    DT_INT2 = 31,           // int2 type
    DT_UINT2 = 32,          // uint2 type
    DT_BFLOAT16 = 33,       // bf16 type
    DT_HIFLOAT8 = 34,       // hifp8 type
    DT_FLOAT8_E4M3FN = 35,  // fp8_e4m3 type
    DT_FLOAT8_E5M2 = 36,    // fp8_e5m2 type
    DT_FLOAT4_E2M1 = 37,    // fp4_e2m1 type
    DT_FLOAT8_E8M0 = 38,    // fp8_e8m0 type
    DT_FLOAT4_E1M2 = 39,    // fp4_e1m2 type
    DT_MAX = 40,            // Mark the boundaries of data types
};

#if !defined(__NPU_DEVICE__) && !defined(__ASCC_DEVICE__)
const std::map<DataType, uint32_t> DTYPE_BYTE_TAB = {
    {DataType::DT_FLOAT, 4},      {DataType::DT_FLOAT16, 2},       {DataType::DT_INT8, 1},
    {DataType::DT_INT16, 2},      {DataType::DT_UINT16, 2},        {DataType::DT_UINT8, 1},
    {DataType::DT_INT32, 4},      {DataType::DT_INT64, 8},         {DataType::DT_UINT32, 4},
    {DataType::DT_UINT64, 8},     {DataType::DT_BF16, 2},          {DataType::DT_BFLOAT16, 2},
    {DataType::DT_INT4, 1},       {DataType::DT_FLOAT4_E2M1, 1},   {DataType::DT_FLOAT4_E1M2, 1},
    {DataType::DT_HIFLOAT8, 1},   {DataType::DT_FLOAT8_E4M3FN, 1}, {DataType::DT_FLOAT8_E5M2, 1},
    {DataType::DT_FLOAT8_E8M0, 1}};

const std::map<DataType, uint32_t> DTYPE_BIT_TAB = {
    {DataType::DT_FLOAT, 32},     {DataType::DT_FLOAT16, 16},      {DataType::DT_INT8, 8},
    {DataType::DT_INT16, 16},     {DataType::DT_UINT16, 16},       {DataType::DT_UINT8, 8},
    {DataType::DT_INT32, 32},     {DataType::DT_INT64, 64},        {DataType::DT_UINT32, 32},
    {DataType::DT_UINT64, 64},    {DataType::DT_BF16, 16},         {DataType::DT_BFLOAT16, 16},
    {DataType::DT_INT4, 4},       {DataType::DT_FLOAT4_E2M1, 4},   {DataType::DT_FLOAT4_E1M2, 4},
    {DataType::DT_HIFLOAT8, 8},   {DataType::DT_FLOAT8_E4M3FN, 8}, {DataType::DT_FLOAT8_E5M2, 8},
    {DataType::DT_FLOAT8_E8M0, 8}};
#endif // !defined(__NPU_DEVICE__) && !defined(__ASCC_DEVICE__)
/**
 * @enum class TPosition
 * @brief TPosition inherits from int32_t and includes a set of storage positions
 */
enum class TPosition : int32_t {
    GM,            ///< GM position
    A1,            ///< A1 position
    A2,            ///< A2 position
    B1,            ///< B1 position
    B2,            ///< B2 position
    C1,            ///< C1 position
    C2,            ///< C2 position
    CO1,           ///< CO1 position
    CO2,           ///< CO2 position
    VECIN,         ///< Vector input position
    VECOUT,        ///< Vector output position
    VECCALC,       ///< Vector calculation position
    LCM = VECCALC, ///< LCM position (equivalent to VECCALC)
    SPM,           ///< SPM position
    SHM = SPM,     ///< SHM position (equivalent to SPM)
    TSCM,          ///< TSCM position
    MAX,           ///< Maximum position
};
/**
 * @enum class TilingPolicy
 * @brief TilingPolicy inherits from int32_t and includes a set of policys
 */
enum class TilingPolicy : int32_t {
    FIXED_A_TSCM,   ///< Fixed A TSCM policy
    FIXED_B_TSCM,   ///< Fixed B TSCM policy
    FIXED_A_B_TSCM, ///< Fixed A and B TSCM policy
    NO_POLICY       ///< No policy
};
/**
 * @enum class CubeFormat
 * @brief CubeFormat inherits from int32_t and includes a set of cube formats
 */
enum class CubeFormat : int32_t {
    ND = 0,           ///< Undefined format
    NZ,               ///< NZ format
    ZN,               ///< ZN format
    ZZ,               ///< ZZ format
    NN,               ///< NN format
    ND_ALIGN,         ///< ND alignment format
    SCALAR,           ///< Scalar format
    VECTOR,           ///< Vector format
    ROW_MAJOR = ND,   // ND              ///< Row-major format, equivalent to ND
    COLUMN_MAJOR = 8, // DN             ///< Column-major format, equivalent to DN
};
/**
 * @enum class MatrixTraverse
 * @brief MatrixTraverse inherits from int32_t and includes a set of traverse methods
 */
enum class MatrixTraverse : int32_t {
    NOSET = 0,  ///< Traverse method not set
    FIRSTM = 1, ///< Traverse by rows first
    FIRSTN = 2, ///< Traverse by columns first
};
/**
 * @enum class MatrixMadType
 * @brief MatrixMadType inherits from int32_t and includes a set of matrix operation modes
 */
enum class MatrixMadType : int32_t {
    NORMAL = 0, ///< Normal matrix operation mode
    HF32 = 1,   // V220 HF32              ///< High-performance 32-bit floating-point operation mode (V220 HF32)
    MXMODE = 2, // v310 MxMatmulFlag    ///< Matrix multiplication flag mode (V310 MXMODE)
};
/**
 * @enum class DequantType
 * @brief DequantType inherits from int32_t and includes a set of quantification modes
 */
enum class DequantType : int32_t {
    SCALAR = 0, ///< Scalar type, value is 0
    TENSOR = 1, ///< Tensor type, value is 1
};

/**
 * @enum class ScheduleType
 * @brief ScheduleType inherits from int32_t and includes a set of operation types
 */
enum class ScheduleType : int32_t {
    INNER_PRODUCT = 0, ///< Inner product operation type, value is 0
    OUTER_PRODUCT = 1, ///< Outer product operation type, value is 1
    N_BUFFER_33 = 2,   ///< Buffer type, value is 2
};

/**
 * @struct SysTilingTempBufSize
 * @brief System tiling temporary buffer size structure
 *
 * This structure stores the temporary buffer size information required during system tiling.
 */
struct SysTilingTempBufSize {
    int32_t ubSize = 0;
    int32_t l1Size = 0;
    int32_t l0cSize = 0;
};
/**
 * @struct MatTilingType
 * @brief Structure for matrix tiling type configuration
 */
struct MatTilingType {
    /**
     * @brief Matrix position, default is global memory (GM)
     */
    TPosition pos = TPosition::GM;
    /**
     * @brief Matrix format, default is ND format
     */
    CubeFormat type = CubeFormat::ND;
    /**
     * @brief Matrix data type, default is float
     */
    DataType dataType = DataType::DT_FLOAT;
    /**
     * @brief Whether the matrix is transposed, default is false
     */
    bool isTrans = false;
    /**
     * @brief Whether the matrix uses double buffer, default is false
     */
    bool isDB = false;
    /**
     * @brief Whether scale type has been set, default is false
     */
    bool hasSetScaleType = false;
    /**
     * @brief Scale position, default is global memory (GM)
     */
    TPosition scalePos = TPosition::GM;
    /**
     * @brief Scale format, default is ND format
     */
    CubeFormat scaleType = CubeFormat::ND;
    /**
     * @brief Whether scale is transposed, default is false
     */
    bool isScaleTrans = false;
};
/**
 * @struct BufferPool
 * @brief Buffer pool structure for managing buffers of different sizes
 */
struct BufferPool {
    int32_t l1Size;  ///< Size of the L1 buffer
    int32_t l0CSize; ///< Size of the L0C buffer
    int32_t ubSize;  ///< Size of the UB buffer
    int32_t l0ASize; ///< Size of the L0A buffer
    int32_t l0BSize; ///< Size of the L0B buffer
    int32_t btSize;  ///< Size of the BT buffer

    int32_t l1AlignSize;  ///< Aligned size of the L1 buffer
    int32_t l0CAlignSize; ///< Aligned size of the L0C buffer
    int32_t l0AAlignSize; ///< Aligned size of the L0A buffer
    int32_t l0BAlignSize; ///< Aligned size of the L0B buffer
    int32_t ubAlignSize;  ///< Aligned size of the UB buffer
};
/**
 * @struct PlatformInfo
 * @brief A structure that stores platform information.
 */
struct PlatformInfo {
    /**
     * @brief Soc version information.
     */
    platform_ascendc::SocVersion socVersion;
    uint64_t l1Size = 0;  ///< Size of L1 cache, in bytes
    uint64_t l0CSize = 0; ///< Size of L0C cache, in bytes
    uint64_t ubSize = 0;  ///< Size of UB cache, in bytes
    uint64_t l0ASize = 0; ///< Size of L0A cache, in bytes
    uint64_t l0BSize = 0; ///< Size of L0B cache, in bytes
};
/**
 * @struct MatmulConfigParams
 * @brief Matrix multiplication configuration parameters structure
 */
struct MatmulConfigParams {
    /**
     * @brief Matrix multiplication configuration type
     */
    int32_t mmConfigType;
    /**
     * @brief Whether to enable L1 cache
     */
    bool enableL1CacheUB;
    /**
     * @brief Schedule type
     */
    ScheduleType scheduleType;
    /**
     * @brief Matrix traversal method
     */
    MatrixTraverse traverse;
    /**
     * @brief Whether to enable vector ND2NZ
     */
    bool enVecND2NZ;
    /**
     * @brief Constructor
     * @param [in] mmConfigTypeIn Matrix multiplication configuration type, default is 1
     * @param [in] enableL1CacheUBIn Whether to enable L1 cache, default is false
     * @param [in] scheduleTypeIn Schedule type, default is ScheduleType::INNER_PRODUCT
     * @param [in] traverseIn Matrix traversal method, default is MatrixTraverse::NOSET
     * @param [in] enVecND2NZIn Whether to enable vector ND2NZ, default is false
     */
    MatmulConfigParams(
        int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false,
        ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, MatrixTraverse traverseIn = MatrixTraverse::NOSET,
        bool enVecND2NZIn = false)
        : mmConfigType(mmConfigTypeIn),
          enableL1CacheUB(enableL1CacheUBIn),
          scheduleType(scheduleTypeIn),
          traverse(traverseIn),
          enVecND2NZ(enVecND2NZIn)
    {}
};

class MatmulApiTilingBase {
public:
    MatmulApiTilingBase();
    explicit MatmulApiTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform);
    explicit MatmulApiTilingBase(const PlatformInfo& platform);
    virtual ~MatmulApiTilingBase();
    /**
     * @brief Set the A type
     * @param [in] pos: the position, type TPosition
     * @param [in] type: the cube format, type CubeFormat
     * @param [in] dataType: the data type, type DataType
     * @param [in] isTrans: whether to transpose, default is false
     */
    int32_t SetAType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false);
    /**
     * @brief Set the B type
     * @param [in] pos: the position, type TPosition
     * @param [in] type: the cube format, type CubeFormat
     * @param [in] dataType: the data type, type DataType
     * @param [in] isTrans: whether to transpose, default is false
     */
    int32_t SetBType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false);
    /**
     * @brief Set the scale A type
     * @param [in] scalePos: scale position, type TPosition
     * @param [in] scaleType: scale type, type CubeFormat
     * @param [in] isScaleTrans: whether to perform scale transformation, default is false
     */
    int32_t SetScaleAType(TPosition scalePos, CubeFormat scaleType, bool isScaleTrans = false);
    /**
     * @brief Set the scale B type
     * @param [in] scalePos: scale position, type TPosition
     * @param [in] scaleType: scale type, type CubeFormat
     * @param [in] isScaleTrans: whether to perform scale transformation, default is true
     */
    int32_t SetScaleBType(TPosition scalePos, CubeFormat scaleType, bool isScaleTrans = true);
    /**
     * @brief Set the type and data type of a cube
     * @param [in] pos: the position, type TPosition
     * @param [in] type: the cube format, type CubeFormat
     * @param [in] dataType: the data type, type DataType
     */
    int32_t SetCType(TPosition pos, CubeFormat type, DataType dataType);
    /**
     * @brief Set bias type
     * @param [in] pos: the position, type TPosition
     * @param [in] type: the cube format, type CubeFormat
     * @param [in] dataType: the data type, type DataType
     */
    int32_t SetBiasType(TPosition pos, CubeFormat type, DataType dataType);
    /**
     * @brief Set the dequantization type
     * @param [in] dequantType: the dequantization type enumeration value
     * @return Return 0 to indicate successful setting
     */
    int32_t SetDequantType(DequantType dequantType)
    {
        this->deqType = dequantType;
        return 0;
    }
    /**
     * @brief Set the shape of the object
     * @param [in] m: first dimension of the shape
     * @param [in] n: second dimension of the shape
     * @param [in] k: third dimension of the shape
     */
    virtual int32_t SetShape(int32_t m, int32_t n, int32_t k);
    /**
     * @brief Set the original shape dimensions
     * @param [in] orgMIn: the M dimension size of the original shape
     * @param [in] orgNIn: the N dimension size of the original shape
     * @param [in] orgKIn: the K dimension size of the original shape
     */
    int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKIn);
    /**
     * @brief Set the original shape dimensions
     * @param [in] orgMIn: the M dimension size of the original shape
     * @param [in] orgNIn: the N dimension size of the original shape
     * @param [in] orgKaIn: the Ka dimension size of the original shape
     * @param [in] orgKbIn: the Kb dimension size of the original shape
     */
    int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKaIn, int32_t orgKbIn);
    /**
     * @brief Set the layout axis information for matrix A, including B, S, N, G, and D axis
     * @param [in] b: batch dimension (B-axis) size, representing the number of batches
     * @param [in] s: spatial dimension (S-axis) size, representing the number of spatial dimensions
     * @param [in] n: channel dimension (N-axis) size, representing the number of channels
     * @param [in] g: group dimension (G-axis) size, representing the number of groups
     * @param [in] d: dimension (D-axis) size, representing the number of dimensions
     */
    int32_t SetALayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
    /**
     * @brief Set the layout axis information for matrix B, including B, S, N, G, and D axis
     * @param [in] b: batch dimension (B-axis) size, representing the number of batches
     * @param [in] s: spatial dimension (S-axis) size, representing the number of spatial dimensions
     * @param [in] n: channel dimension (N-axis) size, representing the number of channels
     * @param [in] g: group dimension (G-axis) size, representing the number of groups
     * @param [in] d: dimension (D-axis) size, representing the number of dimensions
     */
    int32_t SetBLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
    /**
     * @brief Set the layout axis information for matrix C, including B, S, N, G, and D axis
     * @param [in] b: batch dimension (B-axis) size, representing the number of batches
     * @param [in] s: spatial dimension (S-axis) size, representing the number of spatial dimensions
     * @param [in] n: channel dimension (N-axis) size, representing the number of channels
     * @param [in] g: group dimension (G-axis) size, representing the number of groups
     * @param [in] d: dimension (D-axis) size, representing the number of dimensions
     */
    int32_t SetCLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
    /**
     * @brief Set the batch information for normal processing
     * @param [in] batchA: the value for batch A
     * @param [in] batchB: the value for batch B
     * @param [in] m: the value for parameter m
     * @param [in] n: the value for parameter n
     * @param [in] k: the value for parameter k
     */
    int32_t SetBatchInfoForNormal(int32_t batchA, int32_t batchB, int32_t m, int32_t n, int32_t k);
    /**
     * @brief Set the batch number
     * @param [in] batch: the batch number to set
     */
    int32_t SetBatchNum(int32_t batch);
    /**
     * @brief Enable the bias
     * @param [in] isBiasIn: if true, enable the bias; if false, disable the bias, default is false
     */
    int32_t EnableBias(bool isBiasIn = false);
    /**
     * @brief Set the bias parameter
     * @param [in] isBiasIn: whether to use bias, default is false
     */
    int32_t SetBias(bool isBiasIn = false);
    /**
     * @brief Set fixed split parameters
     * @param [in] baseMIn: initial value for parameter M, default is -1
     * @param [in] baseNIn: initial value for parameter N, default is -1
     * @param [in] baseKIn: initial value for parameter K, default is -1
     * @return Return the result of the setting operation
     */
    int32_t SetFixSplit(int32_t baseMIn = -1, int32_t baseNIn = -1, int32_t baseKIn = -1);
    /**
     * @brief Set the size of buffer spaces
     * @param [in] l1Size: size of L1 buffer in bytes; -1 leaves the current setting unchanged
     * @param [in] l0CSize: size of L0C buffer in bytes; -1 leaves the current setting unchanged
     * @param [in] ubSize: size of UB buffer in bytes; -1 leaves the current setting unchanged
     * @param [in] btSize: size of BT buffer in bytes; -1 leaves the current setting unchanged
     * @return Return 0 if success, -1 if failure
     */
    int32_t SetBufferSpace(int32_t l1Size = -1, int32_t l0CSize = -1, int32_t ubSize = -1, int32_t btSize = -1);
    /**
     * @brief Set the traversal method for the matrix
     * @param [in] traverse: the traversal method to be set
     * @return Return 0 if success
     */
    int32_t SetTraverse(MatrixTraverse traverse); // Set the N direction first for the upper left corner matrix
    /**
     * @brief Set the MAD of the matrix
     * @param [in] madType: the MAD type to set
     */
    int32_t SetMadType(MatrixMadType madType); // Set hf32 mode
    // L0C:  BaseM * baseN = GetTensorC()
    // L1 :  BaseM * BaseK + BaseK*BaseN,  --> [disable temporarily] BaseK/k(1)=k1,  BaseM/m(1)=m1, BaseN/n(1) = n1

    /**
     * @brief Set the split range
     * @param [in] maxBaseM: maximum M value, default is -1
     * @param [in] maxBaseN: maximum N value, default is -1
     * @param [in] maxBaseK: maximum K value, default is -1
     * @param [in] minBaseM: minimum M value, default is -1
     * @param [in] minBaseN: minimum N value, default is -1
     * @param [in] minBaseK: minimum K value, default is -1
     * @return Return the result of the setting
     */
    int32_t SetSplitRange(
        int32_t maxBaseM = -1, int32_t maxBaseN = -1, int32_t maxBaseK = -1, int32_t minBaseM = -1,
        int32_t minBaseN = -1, int32_t minBaseK = -1);
    /**
     * @brief Set the double buffer mode
     * @param [in] a: enable double buffer mode for matrix A
     * @param [in] b: enable double buffer mode for matrix B
     * @param [in] c: enable double buffer mode for matrix C
     * @param [in] bias: enable double buffer mode for bias
     * @param [in] transND2NZ: enable transpose from ND to NZ, default is true
     * @param [in] transNZ2ND: enable transpose from NZ to ND, default is true
     * @return Return 0 if success
     */
    int32_t SetDoubleBuffer(bool a, bool b, bool c, bool bias, bool transND2NZ = true, bool transNZ2ND = true);
    /**
     * @brief Set matrix multiplication configuration parameters
     * @param [in] mmConfigTypeIn: matrix multiplication configuration type, default is 1
     * @param [in] enableL1CacheUBIn: enable L1 cache, default is false
     * @param [in] scheduleTypeIn: schedule type, default is INNER_PRODUCT
     * @param [in] traverseIn: matrix traversal method, default is NOSET
     * @param [in] enVecND2NZIn: enable vector ND2NZ, default is false
     * @note this function is used to set matrix multiplication configuration parameters,
     *       including configuration type, cache enablement, schedule type, traversal method, and vector conversion
     */
    void SetMatmulConfigParams(
        int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false,
        ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, MatrixTraverse traverseIn = MatrixTraverse::NOSET,
        bool enVecND2NZIn = false);
    /**
     * @brief Set matrix multiplication configuration parameters
     * @param [in] configParams: matrix multiplication configuration parameters object
     * @note this function sets matrix multiplication configuration parameters by passing a MatmulConfigParams object
     */
    void SetMatmulConfigParams(const MatmulConfigParams& configParams);
    /**
     * @brief Set the sparse matrix flag
     * @param [in] isSparseIn: input flag for sparse matrix, the matrix is sparse if true
     */
    int32_t SetSparse(bool isSparseIn = false);
    /**
     * @brief Get the base M value
     * @return Return the base M value
     */
    int32_t GetBaseM() const { return baseM; }
    /**
     * @brief Get the base N value
     * @return Return the base N value
     */
    int32_t GetBaseN() const { return baseN; }
    /**
     * @brief Get the base K value
     * @return Return the base K value
     */
    int32_t GetBaseK() const { return baseK; }

    /**
     * @brief Interface to get tiling information
     * @param [in] tiling: reference to store the tiling information
     * @note the tiling of this function is in namespace optiling
     */
    virtual int64_t GetTiling(optiling::TCubeTiling& tiling) = 0;
    /**
     * @brief Interface to get tiling information
     * @param [in] tiling: reference to store the tiling information
     * @note the tiling of this function is in global namespace
     */
    virtual int64_t GetTiling(AscendC::tiling::TCubeTiling& tiling) = 0;

public:
    optiling::TCubeTiling tiling_;

    MatTilingType aType_;
    MatTilingType bType_;
    MatTilingType cType_;
    MatTilingType biasType_;
    bool isBias = false;
    bool isSupportL0c2Out = true;
    int32_t blockDim = 0;
    int32_t orgM = 0;
    int32_t orgN = 0;
    int32_t orgKa = 0;
    int32_t orgKb = 0;

    int32_t aLayoutInfoB = 0;
    int32_t aLayoutInfoS = 0;
    int32_t aLayoutInfoN = 0;
    int32_t aLayoutInfoG = 0;
    int32_t aLayoutInfoD = 0;
    int32_t bLayoutInfoB = 0;
    int32_t bLayoutInfoS = 0;
    int32_t bLayoutInfoN = 0;
    int32_t bLayoutInfoG = 0;
    int32_t bLayoutInfoD = 0;
    int32_t cLayoutInfoB = 0;
    int32_t cLayoutInfoS1 = 0;
    int32_t cLayoutInfoN = 0;
    int32_t cLayoutInfoG = 0;
    int32_t cLayoutInfoS2 = 0;
    int32_t batchNum = 0;

    int32_t singleM = 0;
    int32_t singleN = 0;
    int32_t singleK = 0;

    int32_t singleCoreM = 0;
    int32_t singleCoreN = 0;
    int32_t singleCoreK = 0;

    int32_t baseM = 0;
    int32_t baseN = 0;
    int32_t baseK = 0;
    int32_t batchM = 0;
    int32_t batchN = 0;
    int32_t singleBatchM = 0;
    int32_t singleBatchN = 0;
    int32_t alignSingleM = 1;
    int32_t alignSingleN = 1;
    int32_t alignSingleK = 1;

    struct MnmAdjust {
        int32_t maxBaseM;
        int32_t maxBaseN;
        int32_t maxBaseK;

        int32_t minBaseM;
        int32_t minBaseN;
        int32_t minBaseK;
    } adjust_;

    BufferPool oriBufferPool_;
    BufferPool bufferPool_;
    MatrixTraverse traverse_ = MatrixTraverse::FIRSTM;
    MatrixMadType madType_ = MatrixMadType::NORMAL;
    ScheduleType scheduleType = ScheduleType::INNER_PRODUCT;
    bool transND2NZ_ = false;
    bool transNZ2ND_ = false;
    bool isSparse_ = false;

    int32_t maxSingleM = 0;
    int32_t maxSingleN = 0;
    int32_t maxSingleK = 0;
    int32_t minSingleM = 0;
    int32_t minSingleN = 0;
    int32_t minSingleK = 0;
    DequantType deqType = DequantType::SCALAR;
    bool enableSplitK_ = false;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    int32_t mmConfigType = 1; // 0: Norm; 1: MDL
    bool enableL1CacheUB = false;
    bool enVecND2NZ = false;
    bool isBMNKBmm = false;

protected:
    virtual int64_t Compute() = 0;
    void SetFinalTiling(optiling::TCubeTiling& tiling);
    void SetFinalTiling(AscendC::tiling::TCubeTiling& tiling);
    bool CheckSetParam();
    void PrintTilingData();
    void PrintTilingDataInfo(optiling::TCubeTiling& tiling) const;
    void PrintTilingDataInfo(AscendC::tiling::TCubeTiling& tiling) const;
};
} // namespace matmul_tiling

#endif // LIB_MATMUL_MATMUL_TILING_BASE_H
