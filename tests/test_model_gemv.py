import os
import sys
import math
# sys.path.append(os.path.dirname(__file__) + "./build/Debug")

import numpy as np
np.random.seed(123)
np.set_printoptions(threshold=np.inf)

import zbench
from kernel_build import build_cm_kernels

def np_bf162np_float(arr):
    """Convert a numpy array of bf16 (uint16) to a numpy array
    of float"""
    u32 = np.left_shift(arr.astype("uint32"), 16)
    return u32.view("<f4")

def test_fp32_vxm():
    
    def _bench_gemm_nchw_fp16(A, B, m, k, n, tile_m,  tile_k, tile_n,
                            tx, ty, tz, gx, gy, gz, iter_num):
        
        # Update kernel binary according to macros
        flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                -DTILE_M={tile_m}  -DTILE_K={tile_k}  -DTILE_N={tile_n} "
        build_cm_kernels(kernel_file='vxm_test/vxm_test_fp16.cpp', define_flag= flag)
        
        # Setup inputs and outputs
        print(flag)
        temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "vxm_test_fp16.bin"), 
                                            os.path.join(source_bin_path, "vxm_test_fp16.spv"), 
                                            "vxm_test_fp16",
                                            A=A, B=B, m=m,  k=k, n=n,
                                            tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)
            
        return temp_res
        
    scal=128
    # scal=1
    # m=16*scal
    # k=16*scal
    # n=16*scal
    m=1
    k=32*scal # 2080
    n=32*scal # 1040
    
    tx=1
    ty=1
    tz=1
    gx=1
    gy=n//32
    gz=1
    tile_m=1
    tile_k=32
    tile_n=32

    if 0:
        m_A = np.random.randint(0, 5, [m,k]).astype("float16")
        m_B = np.random.randint(0, 5, [k,n]).astype("float16")
    else:
        m_A = np.ones([m,k]).astype("float32")
        m_B = np.ones([k,n]).astype("float32")
    # m_A = np.random.uniform(0, 1, [m,k]).astype("float16") 
    # m_B = np.random.uniform(0, 1, [k,n]).astype("float16") 

    # print(m_A)
    ref_C = np.matmul(m_A, m_B).astype("float16")
    # np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    # ref_C = np.genfromtxt('ref_C.csv', delimiter=',').reshape((m,n)).astype("float16")
    np.savetxt("m_A.csv", m_A, delimiter=",", fmt='%.0f')
    np.savetxt("m_B.csv", m_B, delimiter=",", fmt='%.0f')

    m_A = m_A.view(np.uint32)
    m_B = m_B.view(np.uint32)
    
   
    real_C = _bench_gemm_nchw_fp16(m_A, m_B, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_num=int(10))
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    
    # print(len(ma_right[0]))
    # print(np_bf162np_float(m_A)[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()
    # print("temp_res:", temp_res)
    np.savetxt("real_C.csv", real_C, delimiter=",", fmt='%.0f')
    np.testing.assert_array_equal(real_C, ref_C)
    # np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    print("----PASS----")

    
    # print("ref_C:", ref_C)
    
    # print("temp_res:", rec_calc.reshape((m, n))[0])
    # print("np_bf162np_float(m_A):", np_bf162np_float(m_A)[0])


def test_fp16_vxm():
    
    def _bench_gemm_nchw_fp16(A, B, m, k, n, tile_m,  tile_k, tile_n,
                         tx, ty, tz, gx, gy, gz, iter_num):
    
        # Update kernel binary according to macros
        flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                -DTILE_M={tile_m}  -DTILE_K={tile_k}  -DTILE_N={tile_n} "
        build_cm_kernels(kernel_file='vxm_test/vxm_test_fp16.cpp', define_flag= flag)
        
        # Setup inputs and outputs
        print(flag)
        temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "vxm_test_fp16.bin"), 
                                            os.path.join(source_bin_path, "vxm_test_fp16.spv"), 
                                            "vxm_test_fp16",
                                            A=A, B=B, m=m,  k=k, n=n,
                                            tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)
            
        return temp_res
    # scal=2
    # scal=65
    test_case = 0
    if test_case == 0:
        m=1
        k=4096
        n=4096
    elif test_case == 1:
        m=1
        k=4096
        n=11008
    elif test_case == 2:
        m=1
        k=11008
        n=4096
    elif test_case == 3:
        m=1
        k=4096
        n=32000
    else:
        m=1
        k=32*32
        n=32*32
    
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
    
   
    real_C = _bench_gemm_nchw_fp16(m_A, m_B, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_num=int(1e3))
    exit()
    # print(real_C.astype('float32'))
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    
    # print(len(ma_right[0]))
    # print(np_bf162np_float(m_A)[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()
    # print("temp_res:", temp_res)
    np.savetxt("real_C.csv", real_C.reshape(128,n//128), delimiter=",", fmt='%.0f')
    np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    np.testing.assert_array_equal(real_C, ref_C)
    print("----PASS----")

    
    # print("ref_C:", ref_C)



def test_fp16_vxm_dpas():
    # scal=2
    # scal=65
    def _bench_gemm_nchw_fp16_dpas(A, B, m, k, n, tile_m,  tile_k, tile_n,
                         tx, ty, tz, gx, gy, gz, iter_num):
    
        # Update kernel binary according to macros
        flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                -DTILE_M={tile_m} -DTILE_K={tile_k} -DTILE_N={tile_n} "
        build_cm_kernels(kernel_file='gemm_nchw_dpas.cpp', define_flag=flag)
        
        # Setup inputs and outputs
        print(flag)
        temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "gemm_nchw_dpas.bin"), 
                                            os.path.join(source_bin_path, "gemm_nchw_dpas.spv"), 
                                            "gemm_nchw_dpas",
                                            A=A, B=B, m=m,  k=k, n=n,
                                            tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)
    
        return temp_res

    def _bench_gemm_nchw_fp16_generic(A, B, m, k, n, tile_m,  tile_k, tile_n,
                        tx, ty, tz, gx, gy, gz, iter_num):

        # Update kernel binary according to macros
        flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                -DTILE_M={tile_m} -DTILE_K={tile_k} -DTILE_N={tile_n} "
        build_cm_kernels(kernel_file='gemm_nchw_fp16.cpp', define_flag=flag)
        
        # Setup inputs and outputs
        print(flag)
        temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "gemm_nchw_fp16.bin"), 
                                            os.path.join(source_bin_path, "gemm_nchw_fp16.spv"), 
                                            "gemm_nchw_fp16",
                                            A=A, B=B, m=m,  k=k, n=n,
                                            tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)

        return temp_res
    test_case = 0
    if test_case == 0:
        m=1
        k=4096
        n=4096
    elif test_case == 1:
        m=1
        k=4096
        n=11008
    elif test_case == 2:
        m=1
        k=11008
        n=4096
    elif test_case == 3:
        m=1
        k=4096
        n=32000
    else:
        m=512
        k=512
        n=512
    
    tile_m=16
    tile_k=16
    tile_n=32

    tx=8
    ty=8
    tz=1
    gx=128
    gy=128
    gz=1

    # tx=1
    # ty=1
    # tz=1
    # gx=1
    # gy=1
    # gz=1

    if 1:
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
    
   
    real_C = _bench_gemm_nchw_fp16_dpas(m_A, m_B, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_num=int(1e3))
    # real_C = _bench_gemm_nchw_fp16_generic(m_A, m_B, m, k, n, 
    #                                 tile_m, tile_k,tile_n,
    #                                 tx, ty, tz, gx, gy, gz, iter_num=int(1))
    # print(real_C.astype('float32'))
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    
    # print(len(ma_right[0]))
    # print(np_bf162np_float(m_A)[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()
    # print("temp_res:", temp_res)
    np.savetxt("real_C.csv", real_C.reshape(m, n), delimiter=",", fmt='%.0f')
    np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    np.testing.assert_array_equal(real_C, ref_C)
    print("----PASS----")

    
    # print("ref_C:", ref_C)


def test_fp16_vxm_dpas_trans():
    # scal=65
    def _bench_vxm_nchw_fp16_dpas(A, B, m, k, n, tile_m,  tile_k, tile_n,
                         tx, ty, tz, gx, gy, gz, iter_num):
    
        # Update kernel binary according to macros
        flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                -DTILE_M={tile_m} -DTILE_K={tile_k} -DTILE_N={tile_n} "
        build_cm_kernels(kernel_file='vxm_test/vxm_test_fp16_dpas_dev.cpp', define_flag=flag)
        
        # Setup inputs and outputs
        print(flag)
        temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "vxm_test_fp16_dpas_dev.bin"), 
                                            os.path.join(source_bin_path, "vxm_test_fp16_dpas_dev.spv"), 
                                            "vxm_test_fp16",
                                            A=A, B=B, m=m,  k=k, n=n,
                                            tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)
    
        return temp_res

    # def _bench_vxm_nchw_fp16_dpas(A, B, m, k, n, tile_m,  tile_k, tile_n,
    #                      tx, ty, tz, gx, gy, gz, iter_num):
    
    #     # Update kernel binary according to macros
    #     flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
    #             -DTILE_M={tile_m} -DTILE_K={tile_k} -DTILE_N={tile_n} "
    #     build_cm_kernels(kernel_file='vxm_test/vxm_unit_tests.cpp', define_flag=flag)
        
    #     # Setup inputs and outputs
    #     print(flag)
    #     temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "vxm_unit_tests.bin"), 
    #                                         os.path.join(source_bin_path, "vxm_unit_tests.spv"), 
    #                                         "gemm_nchw_dpas",
    #                                         A=A, B=B, m=m,  k=k, n=n,
    #                                         tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
    #     temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
    #     # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
    #     # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
    #     # temp_res = np_bf162np_float(temp_res)
    
    #     return temp_res

    
    scal=2
    test_case = 1
    if test_case == 0:
        m=1
        k=16
        n=8
        
        tile_m=8
        tile_k=16
        tile_n=8

        tx=1
        ty=1
        tz=1
        gx=1
        gy=1
        gz=1
    elif test_case == 1:
        m=1
        k=4096
        n=4096
        
        tile_m=8
        tile_k=8
        tile_n=32

        tx=1
        ty=1
        tz=1
        gx=1
        gy=1024
        gz=1
        
    elif test_case == 2:
        m=1
        k=4096
        n=11008
    elif test_case == 3:
        m=1
        k=11008
        n=4096
    elif test_case == 4:
        m=1
        k=4096
        n=32000
    else:
        m=1
        k=64*scal
        n=64*scal

    
    # tx=1
    # ty=1
    # tz=1
    # gx=1
    # gy=1
    # gz=1

    if 1:
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
    
   
    real_C = _bench_vxm_nchw_fp16_dpas(m_A, m_B, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_num=int(1e2))
    # print(real_C.astype('float32'))
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    
    # print(len(ma_right[0]))
    # print(np_bf162np_float(m_A)[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()
    # print("temp_res:", temp_res)
    
    np.savetxt("real_C.csv", real_C.reshape(m, n), delimiter=",", fmt='%.0f')
    np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    np.testing.assert_array_equal(real_C, ref_C)
    print("----PASS----")

    
    # print("ref_C:", ref_C)

def test_fp16_slm():
    
    def _build_kernel(A, B, m, k, n, tile_m,  tile_k, tile_n,
                         tx, ty, tz, gx, gy, gz, iter_num):
    
        # Update kernel binary according to macros
        flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                -DTILE_M={tile_m} -DTILE_K={tile_k} -DTILE_N={tile_n} "
        build_cm_kernels(kernel_file='vxm_test/vxm_test_fp16_dpas.cpp', define_flag=flag)
        
        # Setup inputs and outputs
        print(flag)
        temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "vxm_test_fp16_dpas.bin"), 
                                            os.path.join(source_bin_path, "vxm_test_fp16_dpas.spv"), 
                                            "vxm_test_fp16",
                                            A=A, B=B, m=m,  k=k, n=n,
                                            tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)
    
        return temp_res

    # def _bench_vxm_nchw_fp16_dpas(A, B, m, k, n, tile_m,  tile_k, tile_n,
    #                      tx, ty, tz, gx, gy, gz, iter_num):
    
    #     # Update kernel binary according to macros
    #     flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
    #             -DTILE_M={tile_m} -DTILE_K={tile_k} -DTILE_N={tile_n} "
    #     build_cm_kernels(kernel_file='vxm_test/vxm_unit_tests.cpp', define_flag=flag)
        
    #     # Setup inputs and outputs
    #     print(flag)
    #     temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "vxm_unit_tests.bin"), 
    #                                         os.path.join(source_bin_path, "vxm_unit_tests.spv"), 
    #                                         "gemm_nchw_dpas",
    #                                         A=A, B=B, m=m,  k=k, n=n,
    #                                         tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
    #     temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
    #     # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
    #     # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
    #     # temp_res = np_bf162np_float(temp_res)
    
    #     return temp_res

    
    scal=2
    test_case = 0
    if test_case == 0:
        m=1
        k=16
        n=8
        
        tile_m=8
        tile_k=16
        tile_n=8

        tx=1
        ty=1
        tz=1
        gx=1
        gy=1
        gz=1
    elif test_case == 1:
        m=1
        k=4096
        n=4096
        
        tile_m=8
        tile_k=32
        tile_n=8

        tx=1
        ty=32
        tz=1
        gx=1
        gy=64
        gz=1
    elif test_case == 2:
        m=1
        k=4096
        n=11008
    elif test_case == 3:
        m=1
        k=11008
        n=4096
    elif test_case == 4:
        m=1
        k=4096
        n=32000
    else:
        m=1
        k=64*scal
        n=64*scal

    
    if 1:
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
    
   
    real_C = _build_kernel(m_A, m_B, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_num=int(1))
    # print(real_C.astype('float32'))
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    
    # print(len(ma_right[0]))
    # print(np_bf162np_float(m_A)[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()
    # print("temp_res:", temp_res)
    
    np.savetxt("real_C.csv", real_C.reshape(m, n), delimiter=",", fmt='%.0f')
    np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    # np.testing.assert_array_equal(real_C, ref_C)
    print("----PASS----")

    
    # print("ref_C:", ref_C)

def test_fp16_vxm_row_major():
    def _bench_gemm_nchw_fp16(A, B, m, k, n, tile_m,  tile_k, tile_n,
                         tx, ty, tz, gx, gy, gz, iter_num):
    
        # Update kernel binary according to macros
        flag = f" -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} \
                -DTILE_M={tile_m}  -DTILE_K={tile_k}  -DTILE_N={tile_n} "
        build_cm_kernels(kernel_file='vxm_test/vxm_test_fp16_row_maj.cpp', define_flag= flag)
        
        # Setup inputs and outputs
        print(flag)
        temp_res  = zbench.run_gemm_nchw_fp16(os.path.join(source_bin_path, "vxm_test_fp16_row_maj.bin"), 
                                            os.path.join(source_bin_path, "vxm_test_fp16_row_maj.spv"), 
                                            "vxm_test_fp16_row_maj",
                                            A=A, B=B, m=m,  k=k, n=n,
                                            tx=tx, ty=ty, gx=gx, gy=gy, iter_num=iter_num)
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)
            
        return temp_res
    test_case = 1
    if test_case == 0:
        m=1
        k=32
        n=8
        
        tile_m=8
        tile_k=8
        tile_n=4

        tx=1
        ty=1
        tz=1
        gx=1
        gy=int(n/tile_n)
        gz=1
        
    elif test_case == 1:
        m=1
        k=4096
        n=4096
        
        tile_m=1
        tile_k=8
        tile_n=32

        tx=1
        ty=1
        tz=1
        gx=1
        gy=1024
        gz=1

    if 1:
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
    
   
    real_C = _bench_gemm_nchw_fp16(m_A, m_B, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_num=int(1e3))
    # print(real_C.astype('float32'))
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    
    # print(len(ma_right[0]))
    # print(np_bf162np_float(m_A)[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()
    # print("temp_res:", temp_res)
    np.savetxt("real_C.csv", real_C, delimiter=",", fmt='%.0f')
    np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    np.testing.assert_array_equal(real_C, ref_C)
    print("----PASS----")



if __name__ == "__main__":
    source_bin_path = r"C:\Users\12900K\Documents\Engneeing_works\dml_workspace\zbench\cm_kernel_output"
    # test_fp32_vxm()
    test_fp16_vxm()
    # test_fp16_vxm_dpas()
    # test_fp16_vxm_dpas_trans()
    # test_fp16_slm()
    # test_fp16_vxm_row_major()

    exit()
    num = np.float16(2080)
    binary_format = np.binary_repr(num.view('h'))
    print(binary_format)
    print(num)
    # num = np.float32(2049.1)
    # print(num)
    # print(binary_format)
    # num = np.float16(2050)
    # binary_format = np.binary_repr(num.view('h'))
    # print(binary_format)
          