import argparse
import math
import time

import torch
import zbench
import numpy as np
import os


def test_rt_igdext():

    def _bench_gemm_nchw_fp16(A, B, C, m, k, n, tile_m,  tile_k, tile_n,
                         tx, ty, tz, gx, gy, gz, iter_nums):

        # Update kernel binary according to macros
        flag = f"  \
                 "

        # Setup inputs and outputs
        print(flag)

        # build_opt = " -DSIZE_B=1 -DSIZE_C=1 -DSIZE_M=1 -DSIZE_K=4096 -DSIZE_N=4096 -DSCALE=1 -DDT=half -DTILE_K=16 -DTILE_N=32 -DTILE_M=1 -DSLICE_K=1 -DACCU_IS_FP32=1 -DFUSE_SOFTMAX=0  -mdump_asm -Qxcm_doubleGRF -mCM_printregusage -DLWS_SIZE_X=1 -DLWS_SIZE_Y=1 -DLWS_SIZE_Z=1"

        ''' original thread group setup
        gws_x = get_M() / cm_params_.tile_m;
        gws_y = get_N() / cm_params_.tile_n;
        gws_z = get_batch() * get_channels() * cm_params_.slice_k;

        assert(gws_x % lws_x);
        assert(gws_y % lws_y);
        assert(gws_z % lws_z);

        const auto thg_x = gws_x / lws_x;
        const auto thg_y = gws_y / lws_y;
        const auto thg_z = gws_z / lws_z;
        '''

        # ACCU_IS_FP32 will have impact on performance
        build_opt = f"-I \" \" -DSIZE_B=1 -DSIZE_C=1 -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                    -DSCALE=1.0000000000 -DDT=half -DTILE_M={tile_m}  -DTILE_K={tile_k}  -DTILE_N={tile_n} \
                    -DSLICE_K=1 -DACCU_IS_FP32=1 -DFUSE_SOFTMAX=0 \
                    -mdump_asm -Qxcm_doubleGRF -mCM_printregusage \
                    -DLWS_SIZE_X=1 -DLWS_SIZE_Y=1 -DLWS_SIZE_Z=1"
        temp_res  = zbench.launch_rt_igdext(cm_file = "./gemm_nchw_fp16.cpp", build_options = build_opt,
                                            A=A, B=B, C=C, 
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_nums)
        print(f'thg_y: {int(n/tile_n)}')
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))

        return temp_res

    m=1
    k=4096
    n=4096

    tile_m=1
    tile_k=32
    tile_n=16

    gx=1
    gy=1024
    gz=1
    
    tx=1
    ty=1
    tz=1
    

    if 1:
        # m_A = np.random.randint(0, 5, [m,k]).astype("float16")
        # m_B = np.random.randint(0, 5, [k,n]).astype("float16")
        m_A = np.random.uniform(0, 1, [m,k]).astype("float16") 
        m_B = np.random.uniform(0, 1, [k,n]).astype("float16") 
    else:
        m_A = np.ones([m,k]).astype("float16")
        m_B = np.ones([k,n]).astype("float16")


    # print(m_A)
    ref_C = np.matmul(m_A, m_B).astype("float16")
    
    # np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    # ref_C = np.genfromtxt('ref_C.csv', delimiter=',').reshape((m,n)).astype("float16")
    # np.savetxt("m_A.csv", m_A, delimiter=",", fmt='%.0f')
    # np.savetxt("m_B.csv", m_B, delimiter=",", fmt='%.0f')

    m_A = m_A.view(np.uint16)
    m_B = m_B.view(np.uint16)
    m_C = np.zeros_like(ref_C)
    print("Preparing data ready! ")


    real_C = _bench_gemm_nchw_fp16(m_A, m_B, m_C, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_nums=int(1e3))

    np.testing.assert_array_equal(real_C, ref_C)
    print("------------PASS-------------")
    # print(real_C)
    # print(ref_C)
    
if __name__ == "__main__":
    # print("---------------")
    test_rt_igdext()
