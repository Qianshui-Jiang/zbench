import argparse
import math
import time

import torch
import zbench
import numpy as np
import os


def test_rt_igdext():
    
    
    
    



    def _bench_gemm_nchw_fp16(A, B, C, m, k, n, tile_m,  tile_k, tile_n,
                         tx, ty, tz, gx, gy, gz, iter_num):
    
        # Update kernel binary according to macros
        flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                -DTILE_M={tile_m}  -DTILE_K={tile_k}  -DTILE_N={tile_n} "

        
        # Setup inputs and outputs
        print(flag)
        build_opt = "-I \" \" -DSIZE_B=1 -DSIZE_C=1 -DSIZE_M=1 -DSIZE_K=512 -DSIZE_N=512 \
                    -DSCALE=1.0000000000 -DDT=half -DTILE_K=16 -DTILE_N=32 -DTILE_M=1 -DSLICE_K=1 \
                    -DACCU_IS_FP32=1 -DFUSE_SOFTMAX=0  -mdump_asm -Qxcm_doubleGRF -mCM_printregusage \
                    -DLWS_SIZE_X=1 -DLWS_SIZE_Y=1 -DLWS_SIZE_Z=1"
        
        # build_opt = " -DSIZE_B=1 -DSIZE_C=1 -DSIZE_M=1 -DSIZE_K=4096 -DSIZE_N=4096 -DSCALE=1 -DDT=half -DTILE_K=16 -DTILE_N=32 -DTILE_M=1 -DSLICE_K=1 -DACCU_IS_FP32=1 -DFUSE_SOFTMAX=0  -mdump_asm -Qxcm_doubleGRF -mCM_printregusage -DLWS_SIZE_X=1 -DLWS_SIZE_Y=1 -DLWS_SIZE_Z=1"
                    
                    
        '''
        assert(gws_x % lws_x);
        assert(gws_y % lws_y);
        assert(gws_z % lws_z);

        const auto thg_x = gws_x / lws_x;
        const auto thg_y = gws_y / lws_y;
        const auto thg_z = gws_z / lws_z;
        '''
        temp_res  = zbench.test_rt_igdext(cm_file = "./gemm_nchw_fp16.cpp", build_options = build_opt,
                                            A=A, B=B, C=C, thg_x=1, thg_y=1024, thg_z=1)
        # temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "vxm_test_fp16.bin"), 
        #                                     os.path.join(source_bin_path, "vxm_test_fp16.spv"), 
        #                                     "vxm_test_fp16",
        #                                     A=A, B=B, m=m,  k=k, n=n,
        #                                     tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)
            
        return temp_res

    m=1
    k=512
    n=512

    
    tile_m=1
    tile_k=8
    tile_n=32

    tx=1
    ty=1
    tz=1
    gx=1
    gy=1024
    gz=1

    # tx=1
    # ty=1
    # tz=1
    # gx=1
    # gy=1
    # gz=1

    if 0:
        m_A = np.random.randint(0, 5, [m,k]).astype("float16")
        m_B = np.random.randint(0, 5, [k,n]).astype("float16")
    else:
        m_A = np.ones([m,k]).astype("float16")
        m_B = np.ones([k,n]).astype("float16")
    # m_A = np.random.uniform(0, 1, [m,k]).astype("float16") 
    # m_B = np.random.uniform(0, 1, [k,n]).astype("float16") 

    # print(m_A)
    ref_C = np.matmul(m_A, m_B).astype("float16")
    # np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    # ref_C = np.genfromtxt('ref_C.csv', delimiter=',').reshape((m,n)).astype("float16")
    np.savetxt("m_A.csv", m_A, delimiter=",", fmt='%.0f')
    np.savetxt("m_B.csv", m_B, delimiter=",", fmt='%.0f')

    m_A = m_A.view(np.uint16)
    m_B = m_B.view(np.uint16)
    m_C = np.zeros_like(ref_C)
    
   
    real_C = _bench_gemm_nchw_fp16(m_A, m_B, m_C, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_num=int(1e3))
    print(real_C)
    
if __name__ == "__main__":
    # print("---------------")
    test_rt_igdext()
