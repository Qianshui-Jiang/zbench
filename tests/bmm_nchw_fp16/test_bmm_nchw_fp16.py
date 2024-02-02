import numpy as np
import zbench


def test_gemm0_bmm():
    if 1:
        m_A = np.random.randint(0, 5, [1, 32, 1, 128]).astype("float16")
        m_B = np.random.randint(0, 5, [1, 32, 256, 128]).astype("float16")
        # m_B = np.random.uniform(0, 1, [k,n]).astype("float16")
        # m_A = np.random.uniform(0, 1, [m,k]).astype("float16")
    else:
        m_A = np.ones([1, 32, 1, 256]).astype("float32")
        m_B = np.ones([1, 32, 256, 128]).astype("float32")



    m_B_trans = np.transpose(m_B, (0, 1, 3, 2))
    ref_C = np.matmul(m_A, m_B_trans).astype("float16")

    m_A = m_A.view(np.uint16)
    m_B = m_B.view(np.uint16)
    calc_C = np.zeros_like(ref_C)

    print(ref_C.shape)


def test_gemm1_bmm():
    if 0:
        m_A = np.random.randint(0, 5, [1, 32, 1, 256]).astype("float16")
        m_B = np.random.randint(0, 5, [1, 32, 256, 128]).astype("float16")
        # m_B = np.random.uniform(0, 1, [k,n]).astype("float16")
        # m_A = np.random.uniform(0, 1, [m,k]).astype("float16")
    else:
        m_A = np.ones([1, 32, 1, 256]).astype("float32")
        m_B = np.ones([1, 32, 256, 128]).astype("float32")



    ref_C = np.matmul(m_A, m_B).astype("float16")

    m_A = m_A.view(np.uint16)
    m_B = m_B.view(np.uint16)
    calc_C = np.zeros_like(ref_C)

    print(ref_C.shape)


def test_rt_igdext_gemm():
    def _build_bench(A, B, C, m, k, n, tile_m,  tile_k, tile_n,
                         tx, ty, tz, gx, gy, gz, iter_num):

        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DSIZE_B=1 -DSIZE_C=1 -DSIZE_M={m} -DSIZE_K={k} -DSIZE_N={n} "
        _define += f"-DTILE_M={tile_m}  -DTILE_K={tile_k}  -DTILE_N={tile_n} -DSCALE=1.0000000000 -DDT=half "
        _define += f"-DSLICE_K=1 -DACCU_IS_FP32=0 -DFUSE_SOFTMAX=0 -DALPHA=3.000000 -DBETA=3.000000 "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "

        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "

        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./bmm_nchw_fp16.cpp", 
                                          build_options = build_opt,
                                          A=A, B=B, C=C,
                                          thg_x=gx, thg_y=gy, thg_z=gz, iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))
        # temp_res = np_bf162np_float(temp_res)

        return temp_res

    m=1
    k=224
    n=64

    tile_m=1
    tile_k=16
    tile_n=16

    tx=1
    ty=1
    tz=1
    
    gx=1
    gy=int(n/tile_n)
    gz=1

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
    ref_C = ref_C * 3 + 3
    # np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    # ref_C = np.genfromtxt('ref_C.csv', delimiter=',').reshape((m,n)).astype("float16")
    # np.savetxt("m_A.csv", m_A, delimiter=",", fmt='%.0f')
    # np.savetxt("m_B.csv", m_B, delimiter=",", fmt='%.0f')

    m_A = m_A.view(np.uint16)
    m_B = m_B.view(np.uint16)
    m_C = np.zeros_like(ref_C)
    print("Preparing data ready! ")


    calc_C = _build_bench(m_A, m_B, m_C, m, k, n, 
                                    tile_m, tile_k,tile_n,
                                    tx, ty, tz, gx, gy, gz, iter_num=1)
    
    print(np.testing.assert_array_equal(calc_C, ref_C))
    print("==>> assert_array_equal: --- PASS ---")
    
    exit()
    print(calc_C)
    print(ref_C)
    # try:
    # except:
    #     print("==>> assert_array_equal: --- FAIL ---")


if __name__ == "__main__":
    test_rt_igdext_gemm()

    # test_gemm0_bmm()
    # test_gemm1_bmm()
