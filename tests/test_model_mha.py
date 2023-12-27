import os
import sys
# sys.path.append(os.path.dirname(__file__) + "./build/Debug")

import numpy as np
np.random.seed(123)
np.set_printoptions(threshold=np.inf)

import zbench
from kernel_build import build_cm_kernels


def test_fp16_gemm():
    m=16*2
    k=16*2
    n=32*2
    # m_A = np.random.uniform(0, 10, [m,k]).astype("float32") 
    # m_B = np.random.uniform(0, 10, [k,n]).astype("float32") 

    m_A = np.random.randint(0, 5, [m,k]).astype("float16")
    m_B = np.random.randint(0, 5, [k,n]).astype("float16")

    # m_A = np.ones([m,k]).astype("float16")
    # m_B = np.ones([k,n]).astype("float16")

    # print(m_A)
    np.savetxt("m_A.csv", m_A, delimiter=",", fmt='%.0f')
    np.savetxt("m_B.csv", m_B, delimiter=",", fmt='%.0f')

    ref_C = np.matmul(m_A, m_B)
    m_A = m_A.view(np.uint16)
    m_B = m_B.view(np.uint16)
    # print(m_A.view(np.float16))
    

    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    
    # print(len(ma_right[0]))
    # print(np_bf162np_float(m_A)[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()
    tx=1
    ty=1
    tz=1
    gx=16
    gy=1
    gz=1
    flag = f"-DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
             -DTILE_M=16  -DTILE_K=16  -DTILE_N=32 \
             -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} \
             -DGWS_SIZE_X={gz} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
             
             
    build_cm_kernels(kernel_file='gemm_nchw_fp16.cpp', define_flag= flag)
      
    temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "gemm_nchw_fp16.bin"), 
                                        os.path.join(source_bin_path, "gemm_nchw_fp16.spv"), 
                                        "gemm_nchw_fp16",
                                        A=m_A, B=m_B, m=m,  k=k, n=n,
                                        tx=tx, ty=tx, gx=gx, gy=gy)
    temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
    
    # print("temp_res:", temp_res)
    np.savetxt("temp_res.csv", temp_res, delimiter=",", fmt='%.0f')
    np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    np.testing.assert_array_equal(ref_C, temp_res)
    
    # print("ref_C:", ref_C)
    
    # print("temp_res:", rec_calc.reshape((m, n))[0])
    # print("np_bf162np_float(m_A):", np_bf162np_float(m_A)[0])


if __name__ == "__main__":
    source_bin_path = r"C:\Users\12900K\Documents\Engneeing_works\dml_workspace\zbench\cm_kernel_output"
    test_fp16_gemm()

          