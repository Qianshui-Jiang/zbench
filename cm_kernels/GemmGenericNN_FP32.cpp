/*========================== begin_copyright_notice ============================

INTEL CONFIDENTIAL

Copyright (C) 2018-2023 Intel Corporation

This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.

============================= end_copyright_notice ===========================*/

#include <cm/cm.h>
#include <cm/cmtl.h>

#if FP16_UNIT_USED
    #ifndef UNIT_TYPE
    #define UNIT_TYPE half
    #endif

    #define UNIT_VAL_MAX HALF_MAX
    #define UNIT_VAL_MIN -UNIT_VAL_MAX
    #define UNIT_VAL_ONE 1.0h
    #define UNIT_VAL_ZERO 0.0h
    #define TO_UNIT_TYPE(v) convert_half(v)
#else
    #ifndef UNIT_TYPE
    #define UNIT_TYPE float
    #endif

    #define UNIT_VAL_MAX FLT_MAX
    #define UNIT_VAL_MIN -UNIT_VAL_MAX
    #define UNIT_VAL_ONE 1.0f
    #define UNIT_VAL_ZERO 0.0f
    #define TO_UNIT_TYPE(v) (float)(v)
#endif

#define LOG2E  1.4426950408889634f // base 2
#define LOGE2  0.6931471805599453f // base e

template<typename DT, unsigned VS>
_GENX_ inline void activation(vector_ref<DT, VS> accu, DT m, DT n)
{
#if defined ACTIVATION_FUNCTION_LOGISTIC
    accu = UNIT_VAL_ONE / (UNIT_VAL_ONE + cm_exp(-accu * LOG2E));
#elif defined ACTIVATION_FUNCTION_HYPERBOLIC_TAN
    accu = cm_tanh(accu);
#elif defined ACTIVATION_FUNCTION_RELU
    accu = cm_max(UNIT_VAL_ZERO, accu);
#elif defined ACTIVATION_FUNCTION_RELU_NEGATIVE_SLOPE
    const auto f32NegInfinity_asU32 = vector<uint, 1>(0xff800000u);
    const float f32NegInfinity = f32NegInfinity_asU32.format<float>()[0];
    accu = ((m >= -f32NegInfinity) || (m <= f32NegInfinity)) ? (((accu >= UNIT_VAL_ZERO).all()) ? accu : m) : (cm_max(accu, UNIT_VAL_ZERO) + m * cm_min(accu, UNIT_VAL_ZERO));
#elif defined ACTIVATION_FUNCTION_CLAMP
    accu = cm_max(m, cm_min(n), accu);
#elif defined ACTIVATION_FUNCTION_SOFTRELU
    accu = cm_log(UNIT_VAL_ONE + cm_exp(accu * LOG2E)) * LOGE2;
#elif defined ACTIVATION_FUNCTION_ABS
    accu = cm_abs(accu);
#elif defined ACTIVATION_FUNCTION_LINEAR
    accu = m * accu + n;
#elif defined ACTIVATION_FUNCTION_SQUARE
    accu = accu * accu;
#elif defined ACTIVATION_FUNCTION_SQRT
    accu = cm_sqrt(accu);
#elif defined ACTIVATION_FUNCTION_ELU
    accu = ((accu >= UNIT_VAL_ZERO).all()) ? accu : m * (cm_exp(accu * LOG2E) - UNIT_VAL_ONE);
#elif defined ACTIVATION_FUNCTION_HARD_SIGMOID
    accu = cm_fmax(UNIT_VAL_ZERO, cm_min(m) * accu + n, UNIT_VAL_ONE);
#elif defined ACTIVATION_FUNCTION_SCALED_ELU
    accu = ((accu > UNIT_VAL_ZERO).all()) ? n * accu : n *  (m * cm_exp(accu * LOG2E) - m);
#elif defined ACTIVATION_FUNCTION_SCALED_TANH
    accu = m * cm_tanh(n) * accu;
#elif defined ACTIVATION_FUNCTION_SOFTPLUS
    accu = (m > UNIT_VAL_ONE) ? ((cm_log(UNIT_VAL_ONE + cm_exp(accu * m * LOG2E)) * LOGE2) / m) : (cm_log(UNIT_VAL_ONE + cm_exp(accu * LOG2E)) * LOGE2);
#elif defined ACTIVATION_FUNCTION_LEAKY_RELU
    accu = ((accu >= UNIT_VAL_ZERO).all()) ? accu : m * accu;
#elif defined ACTIVATION_FUNCTION_PARAMETERIZED_RELU
    accu = ((accu >= UNIT_VAL_ZERO).all()) ? accu : m * accu;
#elif defined ACTIVATION_FUNCTION_PARAMETRIC_SOFTPLUS
    accu = m * (cm_log(UNIT_VAL_ONE + cm_exp(n * accu * LOG2E)) * LOGE2);
#elif defined ACTIVATION_FUNCTION_SOFTSIGN
    accu = accu / (UNIT_VAL_ONE + cm_abs(accu));
#elif defined ACTIVATION_FUNCTION_THRESHOLDED_RELU
    accu = ((accu >= m).all()) ? input : UNIT_VAL_ZERO;
#else
    accu = accu;
#endif
}

extern "C" _GENX_MAIN_ void intelblas_gemm_buffer_NN(
    SurfaceIndex SrcASurfInd [[type("buffer_t")]],
    SurfaceIndex SrcBSurfInd [[type("buffer_t")]],
#if C_TERM
    SurfaceIndex SrcCSurfInd [[type("buffer_t")]],
#endif
    SurfaceIndex DstSurfInd [[type("buffer_t")]],
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int off0,
    int off1,
    int off2,
    int offd,
    int ldA,
    int ldB,
    int ldC,
    int ldDst,
    int start_index,
    int stride)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);

    const uint32_t input_a_load_size = TILE_K;
    const uint32_t input_b_load_size = TILE_N;
    const uint32_t input_c_load_size = TILE_N;
    const uint32_t output_store_size = TILE_N;

    const uint32_t input_a_base_offset = thread_id_1 * TILE_M * K + off0;
    const uint32_t input_b_base_offset = thread_id_0 * TILE_N + off1;

    uint32_t output_offset = (thread_id_1 * TILE_M * N + thread_id_0 * TILE_N + offd) * sizeof(float);

    matrix<float, TILE_M, TILE_K> input_a; // 8TILE_M x 16TILE_K
    matrix_ref<uint32_t, TILE_M, TILE_K> input_a_packed = input_a.format<uint32_t, TILE_M, TILE_K>();

    uint32_t input_b_offset = input_b_base_offset * sizeof(float);
    matrix<float, TILE_M, TILE_N> accu(0.0f);

//#pragma unroll
    for(uint32_t i = 0; i < K / TILE_K; i++)
    {
#pragma unroll
        for(int m = 0; m < TILE_M; m++)
        {
            const uint32_t input_a_offset = (input_a_base_offset + (m * K + i * TILE_K)) * sizeof(float);
            input_a_packed.row(m) = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(SrcASurfInd, input_a_offset);
        }

#pragma unroll
        for(uint32_t k = 0; k < TILE_K; k++)
        {
            vector<uint32_t, input_b_load_size> input_b_packed = cm_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(SrcBSurfInd, input_b_offset);
            vector_ref<float, TILE_N> input_b = input_b_packed.format<float>();
            input_b_offset += N * sizeof(float);

#pragma unroll
            for(uint32_t j = 0; j < TILE_M; j++)
            {
                accu.select<1, 1, TILE_N, 1>(j, 0) += input_b * vector<float, TILE_N>(input_a.select<1, 1, 1, 1>(j, k).replicate<TILE_N>());
            }
        }
    }

    accu *= alpha;

#if C_TERM && (BETA != 0 )
    if (start_index == 0)
    {
        uint32_t input_c_offset = (thread_id_0 * TILE_N + thread_id_1 * TILE_M * N + off2) * sizeof(float);
#pragma unroll
        for(uint32_t i = 0; i < TILE_M; i++)
        {
            vector<float, TILE_N> input_c = cm_load<float, input_c_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(SrcCSurfInd, input_c_offset);
            accu.select<1, 1, TILE_N, 1>(i, 0).format<float>() += beta * input_c;
            input_c_offset += N * sizeof(float);
        }
    }
#endif

#pragma unroll
    for(uint32_t i = 0; i < TILE_M; i++)
    {
        vector_ref<uint32_t, output_store_size> accu_0_packed = accu.select<1, 1, TILE_N, 1>(i, 0).format<uint32_t>();
        activation<float, output_store_size>(accu_0_packed.format<float>(), NL_M, NL_N);
        cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(DstSurfInd, output_offset, accu_0_packed);
        output_offset += N * sizeof(float);
    }
}
