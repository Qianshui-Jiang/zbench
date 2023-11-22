import os
import sys
sys.path.append(os.path.dirname(__file__) + "./build/Debug")

import numpy as np

import zbench

def np_float2np_bf16(arr):
    """Convert a numpy array of float to a TVM array
    of bf16"""
    orig = arr.view("<u4")
    bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
    nparr = np.right_shift(orig + bias, 16).astype("uint16")
    return nparr


def np_bf162np_float(arr):
    """Convert a numpy array of bf16 (uint16) to a numpy array
    of float"""
    u32 = np.left_shift(arr.astype("uint32"), 16)
    return u32.view("<f4")

def setMatrix(m, n):
    pass
    

if __name__ == "__main__":
    source_bin_path = r"C:\Users\12900K\Documents\Engneeing_works\dml_workspace\zbench\build\Debug"
    # arr1 = np.array([6, 7.5, 8,0, 1])
    # arr2 = np.array([[1,2,3],[4,5,6]])
    # cmt = zbench.test_bind(mode="bench", input="weights", A=16, B=arr1 )
    # print(zbench.add(3, 4))
    # cmt = zbench.test_get_json()
    # print(cmt)
    # print(zbench.test_take_json(cmt))
    
    # print(zbench.run_sgemm(m=1024, niterations= 1, gy=1, gx=4, 
    #                        bin_file=os.path.join(source_bin_path, "sgemm_genx.bin"),
    #                        fn_name = "sgemm_kernel" )
    #       )
    
    # temp_res = zbench.run_bgemm( M = 128, N= 128,K= 128,
    #                         threadWidth= 4, threadHeight= 4,
    #                         groupWidth= 1, groupHeight= 1,
    #                         bin_file= os.path.join(source_bin_path, "bgemm_dpas_genx.bin"), 
    #                         fn_name= "bgemm_dpas"
    #                         )

    m=128
    n=128
    k=128
    np.random.seed(123)
    m_A = np.random.uniform(0, 10, [m,k]).astype("float32") 
    m_B = np.random.uniform(0, 10, [k,n]).astype("float32") 
    ref_C = np.matmul(m_A, m_B)

    m_A = np_float2np_bf16(m_A)
    m_B = np_float2np_bf16(m_B)
    
    
    # m_A = np.random.uniform(0, 10, [m,k]).astype("float16") 
    # m_B = np.random.uniform(0, 10, [k,n]).astype("float16") 
    
    # print(np_bf162np_float(np_float2np_bf16(m_A)))
    # print(m_B)
    # exit()
    # temp_res  = zbench.run_kernel(bin_file= os.path.join(source_bin_path, "bgemm_dpas_genx.bin"), 
    #                             spirv_file =  os.path.join(source_bin_path, "bgemm_dpas_genx.spv"), 
    #                             fn_name= "bgemm_dpas",
    #                             )
    
    temp_res  = zbench.run_kernel(os.path.join(source_bin_path, "bgemm_dpas_genx.bin"), 
                                  os.path.join(source_bin_path, "bgemm_dpas_genx.spv"), 
                                 "bgemm_dpas",
                                A=m_A, B=m_B, m=m, n=n, k=k, tx=4, ty=4, gx=1, gy=1)
    temp_res = np.array(temp_res).reshape((m, n)).astype("uint16")
    temp_res = np_bf162np_float(temp_res)
    
    print("m_A:", np_bf162np_float(m_A[0]))
    print("temp_res:", temp_res[0])
    # print("temp_res:", ref_C[0])
    # print("temp_res:", temp_res[1])
    # print("temp_res:", temp_res[2])
    # print("temp_res:", temp_res[3])
    # print("temp_res:", temp_res.astype('float16'))
          