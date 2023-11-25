import argparse
import math
import time

import torch


# from cpp.lltm import LLTM
# from lltm_cpp import test_print
from zbench import test_print
import zbench
import numpy as np
import os

# from cpp.jit import test_print
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

    temp_res  = zbench.run_kernel(os.path.join("cpp/bgemm_dpas_genx.bin"), 
                                  os.path.join('.', "bgemm_dpas_genx.spv"), 
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

if __name__ == "__main__":
    # print("---------------")
    test_print("-----------------")
    test_bf16_passing()
