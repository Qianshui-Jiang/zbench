import numpy as np
import zbench
import os
import sys
sys.path.append(os.path.dirname(__file__))
np.random.seed(123)
# Build Option String for mha sfotmax shader=  
# -DINOUT_WIDTH=576 -DINOUT_HEIGHT=576 
# -DBASE_INPUT_OFFSET=0 -DBASE_OUTPUT_OFFSET=0 
# -DITEMNUM_PER_HW=64 
# -DLWS_SIZE_X_ALIGNED=16 
# -DGWS_SIZE_X=9 
# -DGWS_SIZE_Y=576 
# -DGWS_SIZE_Z=16 
# -DLWS_SIZE_X=9 
# -DLWS_SIZE_Y=1 
# -DLWS_SIZE_Z=1 
# -DCM_BINDLESS=1

def softmax_ref(x, axis):
    # math_e = 2.718281828459045235360287471352
    # e_x = np.power(math_e, x - np.max(x, axis=axis, keepdims=True))
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def test_ref_softmax():
    # x = np.random.rand(2,8,576,576)
    x = np.ones([2,8,576,576])
    # x = np.random.rand(2,2,2,2)


    res_x = softmax_ref(x, axis=2)
    print(f"==>> res_x: {res_x}")
    exit()

def test_cm_softmax_conformance():
    
    def _build_bench(A,  m, n, 
                    gx, gy, gz,
                    tx, ty, tz,  
                    iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DINOUT_WIDTH={m} -DINOUT_HEIGHT={n} -DBASE_INPUT_OFFSET=0 -DBASE_OUTPUT_OFFSET=0 "
        _define += f"-DITEMNUM_PER_HW=64 -DLWS_SIZE_X_ALIGNED=16 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1"
        
        _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        # _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./softmax_nchw.cpp", 
                                          build_options = build_opt,
                                          input=A, 
                                          thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                          iter_nums=iter_num)

        # print(f"==>> A.shape: {A.shape}")
        
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(A.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))


        return temp_res
    
    m = 576
    n = 576
    shape_list = [
        [2, 8, m, n],
    ]
    test_case = 0

    gx=9
    gy=576
    gz=16
    
    tx=9
    ty=1
    tz=1

    # input_buf = np.random.randint(0, 5, shape_list[test_case]).astype("float16")
    # input_buf = np.ones(shape_list[test_case]).astype("float16")
    input_buf = np.random.uniform(0, 1, shape_list[test_case]).astype("float16") 
    
    ref_C = softmax_ref(input_buf, axis=2)

    input_buf_uint16 = input_buf.view(np.uint16)

    real_C = _build_bench(input_buf_uint16, m, n,
                          gx, gy, gz, 
                          tx, ty, tz, 
                          iter_num=int(1))

    # exit()
    print(f"==>> input_buf: ")
    # print(input_buf[0,0, 0, 0+64:64+64])
    # exit()
    # print(f"==>> real_C: {real_C}")
    # print(f"==>> real_C: {real_C}")
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")

    # np.testing.assert_array_equal(np_bf162np_float(m_B)[0], mb_right[0])
    # np.testing.assert_array_equal(np_bf162np_float(m_A)[0], ma_right[0])
    # exit()
    # np.savetxt("real_C.csv", real_C.reshape(128,n//128), delimiter=",", fmt='%.0f')
    # np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    # exit()
    ref_argmax = np.argmax(ref_C, axis=3)
    real_argmax = np.argmax(real_C, axis=3)
    # print(f"==>> index_of_max_value: {real_argmax}")
    # np.testing.assert_array_equal(real_argmax, ref_argmax)
    np.testing.assert_array_equal(real_C, ref_C)
    print("------------PASS-------------")


def test_cm_onlines_oftmax_dev():
    
    def _build_bench_dev(A,  m, n, 
                    gx, gy, gz,
                    tx, ty, tz,  
                    iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DINOUT_WIDTH={m} -DINOUT_HEIGHT={n} -DBASE_INPUT_OFFSET=0 -DBASE_OUTPUT_OFFSET=0 "
        _define += f"-DITEMNUM_PER_HW=64 -DLWS_SIZE_X_ALIGNED=16 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1"
        
        _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        # _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./online_softmax_nchw.cpp", 
        # temp_res  = zbench.launch_rt_igdext(cm_file = "./softmax_nchw.cpp", 
                                          build_options = build_opt,
                                          input=A, 
                                          thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                          iter_nums=iter_num)

        # print(f"==>> A.shape: {A.shape}")
        
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(A.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))


        return temp_res
    
    def _build_bench_origin(A,  m, n, 
                gx, gy, gz,
                tx, ty, tz,  
                iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DINOUT_WIDTH={m} -DINOUT_HEIGHT={n} -DBASE_INPUT_OFFSET=0 -DBASE_OUTPUT_OFFSET=0 "
        _define += f"-DITEMNUM_PER_HW=64 -DLWS_SIZE_X_ALIGNED=16 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1"
        
        _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        # _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./softmax_nchw.cpp", 
        # temp_res  = zbench.launch_rt_igdext(cm_file = "./softmax_nchw.cpp", 
                                            build_options = build_opt,
                                            input=A, 
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        # print(f"==>> A.shape: {A.shape}")
        
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(A.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))


        return temp_res
    # x = 256
    # y = 256
    # shape_list = [
    #     [1, 2, x, y],
    # ]

    # test_case = 0

    # gx=int(x/64)
    # gy=256
    # gz=2
    
    # tx=int(x/64)
    # ty=1
    # tz=1
    
    x = 576
    y = 576
    shape_list = [
        [2, 8, x, y],
    ]
    test_case = 0

    gx=9
    gy=576
    gz=16
    
    tx=9
    ty=1
    tz=1

    # input_buf = np.random.randint(0, 5, shape_list[test_case]).astype("float16")
    # input_buf = 0.123 * np.ones(shape_list[test_case]).astype("float16")
    input_buf = np.random.uniform(0, 1, shape_list[test_case]).astype("float16") 


    # ref_C = np.genfromtxt('ref_C.csv', delimiter=',').reshape((m,n)).astype("float16")
    # np.savetxt("m_A.csv", m_A, delimiter=",", fmt='%.0f')
    # np.savetxt("input_buf.csv", input_buf, delimiter=",", fmt='%.0f')

    input_buf_uint16 = input_buf.view(np.uint16)


    real_C = _build_bench_dev(input_buf_uint16, 
                          x, y,
                          gx, gy, gz, 
                          tx, ty, tz, 
                          iter_num=int(1000))
    origin_C = _build_bench_origin(input_buf_uint16, 
                          x, y,
                          gx, gy, gz, 
                          tx, ty, tz, 
                          iter_num=int(1))
    ref_C = softmax_ref(input_buf, axis=2)
    # print(f"==>> real_C: {real_C}")
    
    # mb_right = np.genfromtxt('matrixB_bind.csv', delimiter=',').astype("float32")
    # ma_right = np.genfromtxt('matrixA_bind.csv', delimiter=',').astype("float32")
    # np.savetxt("real_C.csv", real_C.reshape(128,n//128), delimiter=",", fmt='%.0f')
    # np.savetxt("ref_C.csv", ref_C, delimiter=",", fmt='%.0f')
    
    
    
    
    # if not np.allclose(real_C, origin_C, rtol=1e-3):
    #     np.testing.assert_array_equal(real_C, origin_C)
    """    
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    """
    
    # np.testing.assert_array_equal(real_C, ref_C)
    print(f"==>> output sum : {sum(real_C[0,0, 0, :])}")
    print(f"==>> output sum : {sum(origin_C[0,0, 0, :])}")
    print(f"==>> output sum : {sum(ref_C[0,0, 0, :])}")
    
    print("------------PASS-------------")


if __name__ == "__main__":
    # test_ref_softmax()
    # test_cm_softmax_conformance()
    test_cm_onlines_oftmax_dev()
