/**************************************************************************//**
*
* INTEL CONFIDENTIAL
* Copyright 2023
* Intel Corporation All Rights Reserved.
*
* The source code contained or described herein and all documents related to the
* source code ("Material") are owned by Intel Corporation or its suppliers or
* licensors. Title to the Material remains with Intel Corporation or its suppliers
* and licensors. The Material contains trade secrets and proprietary and confidential
* information of Intel or its suppliers and licensors. The Material is protected by
* worldwide copyright and trade secret laws and treaty provisions. No part of the
* Material may be used, copied, reproduced, modified, published, uploaded, posted
* transmitted, distributed, or disclosed in any way without Intel's prior express
* written permission
*
* No license under any patent, copyright, trade secret or other intellectual
* property right is granted to or conferred upon you by disclosure or delivery
* of the Materials, either expressly, by implication, inducement, estoppel
* or otherwise. Any license under such intellectual property rights must be
* express and approved by Intel in writing.
*
* @file D3D12MetaCommandGemmMadLsc.h
*
* @brief Gemm MetaCommand object
*
* Notes:
*
******************************************************************************/
#pragma once

namespace MONZA
{
namespace DL
{
/*
* This kernel is not best in class. It was developed in short timeframe to support some of the Stable Diffusion cases for MTL and DG2 platform. 
* ToDo: optimize this kernel
*/
struct GemmMadLsc
{
    std::uint64_t B      = 0;
    std::uint64_t C      = 0;
    std::uint64_t M      = 0;
    std::uint64_t K      = 0;
    std::uint64_t N      = 0;
    std::uint64_t tile_m = 0;
    std::uint64_t tile_n = 0;
    std::uint64_t tile_k = 0;
    float         alpha  = 0.0f;
    float         beta   = 0.0f;
    bool          use_fp32_accu = false;

    static bool IsSupported( const MetaCommandGemmParam& Param, const ADAPTER_INFO* pAdapter )
    {
        // check device
        if( pAdapter && pAdapter->GfxPlatform.eProductFamily < IGFX_DG2 ) // ToDo: actually it should check for LSC supprt, but for not it will work
        {
            return false;
        }

        // resoruces have to be fp16
        if( Param.Desc.ADesc.DataType != META_COMMAND_TENSOR_DATA_TYPE_FLOAT16 
            || Param.Desc.BDesc.DataType != META_COMMAND_TENSOR_DATA_TYPE_FLOAT16
            || Param.Desc.OutputDesc.DataType != META_COMMAND_TENSOR_DATA_TYPE_FLOAT16
            )
        {
            return false;
        }

        //  check layouts
        if( Param.Desc.ADesc.GetLayout() != META_COMMAND_TENSOR_LAYOUT_NCHW 
            || Param.Desc.BDesc.GetLayout() != META_COMMAND_TENSOR_LAYOUT_NCHW || Param.Desc.OutputDesc.GetLayout() != META_COMMAND_TENSOR_LAYOUT_NCHW )
        {
            return false;
        }

        // check if tensor is packed
        if( !Param.Desc.ADesc.IsPackedTensor() || !Param.Desc.BDesc.IsPackedTensor() || !Param.Desc.OutputDesc.IsPackedTensor() )
        {
            return false;
        }

        // 2d and 3d tensors should be supported as well, but should be validated first
        if( Param.Desc.OutputDesc.DimensionCount != 4)
        {
            return false;
        }

        std::cout << "--------TEST-----------" << std::endl;
        
        // check tensor shapes
        if(  Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_W ] % 16 != 0 || Param.Desc.BDesc.Size[ META_COMMAND_TENSOR_4D_W ] % 32 != 0 )
        // if( Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_H ] % 16 != 0 || Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_W ] % 16 != 0 || Param.Desc.BDesc.Size[ META_COMMAND_TENSOR_4D_W ] % 32 != 0 )
        {
            return false;
        }
                

       if(  Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_H ] % 16 != 0 && Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_H ] != 1 )
        {
            return false;
        }

        // chceck transform param
        if( Param.Desc.Attributes.ATransform != META_COMMAND_MATRIX_TRANSFORM_NONE || Param.Desc.Attributes.BTransform != META_COMMAND_MATRIX_TRANSFORM_NONE )
        {
            return false;
        }

        // no support for CDesc now
        if( !Param.Desc.CDesc.IsNull )
        {
            return false;
        }

        // no supprot for fused activation now
        if( !Param.Desc.Attributes.Activation.IsNull )
        {
            return false;
        }

        // only fp16 and fp32 accumulators allowed
        if( Param.Desc.Attributes.Precision != META_COMMAND_PRECISION_FLOAT16 && Param.Desc.Attributes.Precision != META_COMMAND_PRECISION_FLOAT32 )
        {
            return false;
        }

        return true;
    }

    void Set( const MetaCommandGemmParam& Param )
    {
        if( Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_H ] == 1 )
        {
             tile_m = 1;
         } else {
            tile_m = 16;
        }
        tile_k = 16;
        tile_n = 32;
        B      = Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_N ];
        C      = Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_C ];
        M      = Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_H ];
        K      = Param.Desc.ADesc.Size[ META_COMMAND_TENSOR_4D_W ];
        N      = Param.Desc.BDesc.Size[ META_COMMAND_TENSOR_4D_W ];

        alpha = Param.Desc.Attributes.Alpha;
        beta  = Param.Desc.Attributes.Beta;
        use_fp32_accu = Param.Desc.Attributes.Precision == META_COMMAND_PRECISION_FLOAT32;
    }

    std::string GetBuildOptions()
    {
        // kernel jits
        const std::string pre_jit                = "-D";
        const std::string post_jit               = " ";
        const std::string between_name_and_value = "=";
        std::string       build_options          = "";

        auto add_define = [ & ]( const std::string& name, auto value ) {
            using namespace std;
            const std::string value_str = std::to_string( value );
            build_options += pre_jit + name + between_name_and_value + value_str + post_jit;
        };

        add_define( "SIZE_B", B );
        add_define( "SIZE_C", C );
        add_define( "SIZE_M", M );
        add_define( "SIZE_K", K );
        add_define( "SIZE_N", N );
        add_define( "ALPHA", alpha );
        add_define( "BETA", beta );

        add_define( "TILE_K", tile_k );
        add_define( "TILE_N", tile_n );
        add_define( "TILE_M", tile_m );

        const auto gws = get_gws();
        const auto lws = get_lws();
        add_define( "LWS_SIZE_X", lws[ 0 ] );
        add_define( "LWS_SIZE_Y", lws[ 1 ] );
        add_define( "LWS_SIZE_Z", lws[ 2 ] );
        add_define( "GWS_SIZE_X", gws[ 0 ] );
        add_define( "GWS_SIZE_Y", gws[ 1 ] );
        add_define( "GWS_SIZE_Z", gws[ 2 ] );

        add_define( "ACCU_IS_FP32", use_fp32_accu );
        add_define( "CM_BINDLESS", 1 );
        build_options += "-DDT=half ";
        build_options += "-Qxcm_doubleGRF ";
        return build_options;
    }

    template<typename T>
    void GetShaderObjectInfo( MetaCommandShaderCode& ShaderCode, DispatchParams& DispatchParam )
    {
        const auto gws = get_gws();
        const auto lws = get_lws();        
        DispatchParam.globalTG[ 0 ] = gws[ 0 ] / lws[ 0 ];
        DispatchParam.globalTG[ 1 ] = gws[ 1 ] / lws[ 1 ];
        DispatchParam.globalTG[ 2 ] = gws[ 2 ] / lws[ 2 ];

        ShaderCode.pShaderCode           = g_gemm_nchw_fp16_CM;
        ShaderCode.ShaderCodeSizeInBytes = sizeof( g_gemm_nchw_fp16_CM );
        static UINT64 ShaderCodeHash     = iSTD::Hash( reinterpret_cast<const DWORD*>( ShaderCode.pShaderCode ), ( ShaderCode.ShaderCodeSizeInBytes / sizeof( DWORD ) ) );
        ShaderCode.ShaderCodeHash        = ShaderCodeHash;
        ShaderCode.mLanguage             = MLSS::ShaderType::ShaderType_CM;
        T::MetaCommandInstrumentationFunctionsT::MC_LOG( GFXDBG_NORMAL, DEBUG_METACOMMAND_GEMM, "Gemm Kernel: g_gemm_nchw_fp16_CM " );
    }

private:
    std::array<std::uint32_t, 3> get_gws() const
    {
        if( M == 1 )
        {
             return { 1u, 1024u, 1u };
        } else {
            return { static_cast<uint32_t>( M / tile_m ), static_cast<uint32_t>( N / tile_n ), 1u };
        }
    }

    std::array<std::uint32_t, 3> get_lws() const
    {
        return { 1u, 1u, 1u };
    }
};

} // namespace DL
} // namespace MONZA