import os
import sys
sys.path.append(os.path.dirname(__file__) + "./build/Debug")

import numpy as np
np.random.seed(123)
np.set_printoptions(threshold=np.inf)

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
    

def test_bf16_passing():
    m=128
    n=128
    k=128
    # m_A = np.random.uniform(0, 10, [m,k]).astype("float32") 
    # m_B = np.random.uniform(0, 10, [k,n]).astype("float32") 
    m_A = np.random.randint(0, 5, [m,k]).astype("float32") 
    m_B = np.random.randint(0, 5, [k,n]).astype("float32") 

    ref_C = np.matmul(m_A, m_B)
    m_A = np_float2np_bf16(m_A)
    m_B = np_float2np_bf16(m_B)
    # ref_C = np.matmul(np_bf162np_float(m_A), np_bf162np_float(m_B))
    # ref_C = np_float2np_bf16(ref_C)
    # ref_C = np_bf162np_float(ref_C)
    
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    # print(len(ma_right[0]))
    # print(np_bf162np_float(m_A)[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()

    temp_res  = zbench.run_kernel(os.path.join(source_bin_path, "bgemm_dpas_genx.bin"), 
                                  os.path.join(source_bin_path, "bgemm_dpas_genx.spv"), 
                                 "bgemm_dpas",
                                A=m_A, B=m_B, m=m, n=n, k=k, tx=4, ty=4, gx=1, gy=1)
    temp_res = np.array(temp_res).astype("uint16")
    temp_res = np_bf162np_float(temp_res)

    rec_calc = np.zeros_like(temp_res)
    for j in range(m):
        for i in range(n):
            index = (i >> 4) * 16 * m + (i & 0xf) + j * 16
            rec_calc[j * n + i] = temp_res[index]
        
    # print("temp_res:", temp_res[0])
    print("temp_res:", rec_calc.reshape((m, n))[0])
    print("ref_C:", ref_C[0])
    np.testing.assert_array_equal(rec_calc.reshape((m, n)), ref_C)
    # print("np_bf162np_float(m_A):", np_bf162np_float(m_A)[0])


    
def test_fp16_passing():
    m=128
    n=128
    k=128
    m_A = np.random.uniform(0, 10, [m,k]).astype("float16") 
    m_B = np.random.uniform(0, 10, [k,n]).astype("float16") 


    ref_C = np.matmul(m_A.astype("float32") , m_B.astype("float32") )
    
    temp_res  = zbench.run_kernel(os.path.join(source_bin_path, "bgemm_dpas_genx.bin"), 
                                  os.path.join(source_bin_path, "bgemm_dpas_genx.spv"), 
                                 "bgemm_dpas",
                                A=m_A.view(np.uint16), B=m_B.view(np.uint16), m=m, n=n, k=k, tx=4, ty=4, gx=1, gy=1)
    temp_res = np.array(temp_res).reshape((m, n)).astype("uint16")
    
    # print("m_A:", m_A[0])
    print("temp_res:", np_bf162np_float(temp_res[0]))
    # print("temp_res:", temp_res[0].view(np.float16))
    print("temp_res:", ref_C[0].dtype) 
    print("ref_C:", ref_C[0])
    

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
    
    test_bf16_passing()
    # test_fp16_passing()
          