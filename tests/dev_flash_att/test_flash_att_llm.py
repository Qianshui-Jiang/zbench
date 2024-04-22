import numpy as np
import zbench
import os
import sys
sys.path.append(os.path.dirname(__file__))
np.random.seed(123)
np.set_printoptions(edgeitems=30, linewidth=100000)

def softmax_ref(x, axis):
    # math_e = 2.718281828459045235360287471352
    # e_x = np.power(math_e, x - np.max(x, axis=axis, keepdims=True))
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def small_reference_flash_decoding():
    llama2_q_small_shape = [1, 4, 1, 8]
    llama2_k_small_shape = [1, 4, 16, 8]
    llama2_v_small_shape = [1, 4, 16, 8]
    llama2_output_small_shape = llama2_q_small_shape
    
    
    f = 16  
    t = 16  
    h = 8   

    # q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    # k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    # v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 

    dst_path = "./flash_decoding_json"
    q = np.load(os.path.join(dst_path, "q_tensor_small.npy")).reshape(llama2_q_small_shape)[0, 0, :, :]
    k = np.load(os.path.join(dst_path, "k_tensor_small.npy")).reshape(llama2_k_small_shape)[0, 0, :, :]
    v = np.load(os.path.join(dst_path, "v_tensor_small.npy")).reshape(llama2_v_small_shape)[0, 0, :, :]
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_output.npy")).reshape(llama2_output_small_shape)[0, 0, :, :]
    print(f"==>> q: {q}")
    print(f"==>> k: {k}")
    print(f"==>> v: {v}")
    print(f"==>> output: {output}")
    
    from original_flash_att import flash_attention
    ref_C = flash_attention(q, k, v)
    # print(f"==>> ref_C: {ref_C}")


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
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att.cpp", 
                                          build_options = build_opt,
                                          input=A, 
                                          thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                          iter_nums=iter_num)

        print(f"==>> A.shape: {A.shape}")
        
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
    # print(f"==>> input_buf: ")
    print(input_buf[0,0, 0, 0+64:64+64])
    # exit()
    print(f"==>> real_C: {real_C}")
    print(f"==>> real_C: {real_C}")
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
    print(f"==>> index_of_max_value: {real_argmax}")
    # np.testing.assert_array_equal(real_argmax, ref_argmax)
    np.testing.assert_array_equal(real_C, ref_C)
    # print("------------PASS-------------")


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

        print(f"==>> A.shape: {A.shape}")
        
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

        print(f"==>> A.shape: {A.shape}")
        
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
    print(f"==>> real_C: {real_C}")
    
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
    # print(f"==>> output sum : {sum(real_C[0,0, 0, :])}")
    # print(f"==>> output sum : {sum(origin_C[0,0, 0, :])}")
    # print(f"==>> output sum : {sum(ref_C[0,0, 0, :])}")
    
    # print("------------PASS-------------")


def test_flash_att():

    def _build_bench_dev(q, k, v, output_C,
                        f, t, h,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={f} -DKV_SEQ_LEN={t} -DHEAD_DIM={h} "
        _define += f"-DTILE_Q={2} -DTILE_KV={2} -DTILE_HEAD={h} "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        # _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        print(f"==>> A.shape: {A.shape}")
        
        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))


        return temp_res
    
    
    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    # llama2_q_shape = [1, 32, 1, 2048]
    # llama2_k_shape = [1, 32, 128, 2048]
    # llama2_v_shape = [1, 32, 128, 2048]
    
    f = 16  
    t = 16  
    h = 8   

    q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    print(f"==>> q: {q}")
    print(f"==>> k: {k}")
    print(f"==>> v: {v}")
    
    # q = np.ones((f, h)).astype("float16")
    # k = np.ones((t, h)).astype("float16")
    # v = np.ones((t, h)).astype("float16")
    
    
    input_buf_q = q.view(np.uint16)
    input_buf_k = k.view(np.uint16)
    input_buf_v = v.view(np.uint16)
    
    gx=1
    gy=8
    gz=1
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(q)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                f, t, h,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1))
    
    print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    from original_flash_att import flash_attention
    ref_C = flash_attention(q, k, v)
    # np.testing.assert_array_equal(origin_C, ref_C)
    np.testing.assert_allclose(origin_C, ref_C, atol=1e-3)
    # print("----------------[PASS]----------------")
    
    print(f"==>> ref_C: {ref_C}")


def test_flash_decoding_small_shape():

    def _build_bench_dev(q, k, v, output_C,
                        f, t, h,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={16} -DHEAD_DIM={8} "
        _define += f"-DTILE_Q={1} -DTILE_KV={2} -DTILE_HEAD={8} -DHEAD_SCALE=0.001 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_decoding.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))


        return temp_res
    
    
    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    llama2_q_small_shape = [1, 4, 1, 8]
    llama2_k_small_shape = [1, 4, 16, 8]
    llama2_v_small_shape = [1, 4, 16, 8]
    llama2_output_small_shape = llama2_q_small_shape
    
    f = 16  
    t = 16  
    h = 8   

    # q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    # k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    # v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 

    dst_path = "./flash_decoding_json"
    q = np.load(os.path.join(dst_path, "q_tensor_small.npy")).reshape(llama2_q_small_shape)
    k = np.load(os.path.join(dst_path, "k_tensor_small.npy")).reshape(llama2_k_small_shape)
    v = np.load(os.path.join(dst_path, "v_tensor_small.npy")).reshape(llama2_v_small_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_output.npy")).reshape(llama2_output_small_shape)
    print(f"==>> q: {q}")
    print(f"==>> k: {k}")
    print(f"==>> v: {v}")
    print(f"==>> output: {output}")
    
    # q = np.ones((f, h)).astype("float16")
    # k = np.ones((t, h)).astype("float16")
    # v = np.ones((t, h)).astype("float16")
    
    
    input_buf_q = q.view(np.uint16)
    input_buf_k = k.view(np.uint16)
    input_buf_v = v.view(np.uint16)
    
    gx=4
    gy=1
    gz=1
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                f, t, h,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    exit()
    from original_flash_att import flash_attention
    ref_C = flash_attention(q, k, v)
    # np.testing.assert_array_equal(origin_C, ref_C)
    np.testing.assert_allclose(origin_C, ref_C, atol=1e-3)
    # print("----------------[PASS]----------------")
    print(f"==>> ref_C: {ref_C}")


def test_flash_decoding_llama2_shape():

    def _build_bench_dev(q, k, v, output_C,
                        f, t, h,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={2048} -DHEAD_DIM={128} "
        _define += f"-DTILE_Q={1} -DTILE_KV={8} -DTILE_HEAD={128} -DHEAD_SCALE=0.001 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_decoding.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))


        return temp_res
    
    
    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    llama2_q_shape = [1, 32, 1, 128]
    llama2_k_shape = [1, 32, 2048, 128]
    llama2_v_shape = [1, 32, 2048, 128]
    llama2_output_shape = llama2_q_shape
    
    f = 16  
    t = 16  
    h = 8   

    # q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    # k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    # v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 

    dst_path = "./flash_decoding_json"
    q = np.load(os.path.join(dst_path, "q_tensor.npy")).reshape(llama2_q_shape)
    k = np.load(os.path.join(dst_path, "k_tensor.npy")).reshape(llama2_k_shape)
    v = np.load(os.path.join(dst_path, "v_tensor.npy")).reshape(llama2_v_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_output.npy")).reshape(llama2_output_shape)
    
    q_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_query.npy")).reshape(llama2_q_shape)
    k_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_key.npy")).reshape(llama2_k_shape)
    v_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_value.npy")).reshape(llama2_v_shape)
    np.testing.assert_allclose(q_ref, q)
    np.testing.assert_allclose(k_ref, k)
    np.testing.assert_allclose(v_ref, v)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> k.shape: {k.shape}")
    # print(f"==>> v.shape: {v.shape}")
    
    print(f"==>> k: {k}")
    print(f"==>> v: {v}")
    print(f"==>> output: {output}")
    
    # q = np.ones((f, h)).astype("float16")
    # k = np.ones((t, h)).astype("float16")
    # v = np.ones((t, h)).astype("float16")
    
    input_buf_q = q.view(np.uint16)
    input_buf_k = k.view(np.uint16)
    input_buf_v = v.view(np.uint16)
    
    gx=32
    gy=1
    gz=1
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                f, t, h,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1e2))
    
    print(f"==>> q:\n {q}")
    print(f"==>> origin_C:\n {origin_C}")
    # exit()
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    # print("----------------[PASS]----------------")
    exit()
    from original_flash_att import flash_attention
    ref_C = flash_attention(q, k, v)
    # np.testing.assert_array_equal(origin_C, ref_C)
    print(f"==>> ref_C: {ref_C}")


def test_flash_decoding_small_split_kv():

    def _build_bench_dev(q, k, v, output_C,
                        f, t, h,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={16} -DHEAD_DIM={8} "
        _define += f"-DTILE_Q={1} -DTILE_KV={2} -DTILE_HEAD={8} -DSPLIT_KV={4} -DHEAD_SCALE=0.001 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_decoding_split_kv.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))


        return temp_res
    
    
    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    llama2_q_small_shape = [1, 4, 1, 8]
    llama2_k_small_shape = [1, 4, 16, 8]
    llama2_v_small_shape = [1, 4, 16, 8]
    llama2_output_small_shape = llama2_q_small_shape
    
    f = 16  
    t = 16  
    h = 8   

    # q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    # k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    # v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 

    dst_path = "./flash_decoding_json"
    q = np.load(os.path.join(dst_path, "q_tensor_small.npy")).reshape(llama2_q_small_shape)
    k = np.load(os.path.join(dst_path, "k_tensor_small.npy")).reshape(llama2_k_small_shape)
    v = np.load(os.path.join(dst_path, "v_tensor_small.npy")).reshape(llama2_v_small_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_output.npy")).reshape(llama2_output_small_shape)
    print(f"==>> q: {q}")
    print(f"==>> k: {k}")
    print(f"==>> v: {v}")
    print(f"==>> output: {output}")
    
    # q = np.ones((f, h)).astype("float16")
    # k = np.ones((t, h)).astype("float16")
    # v = np.ones((t, h)).astype("float16")
    
    
    input_buf_q = q.view(np.uint16)
    input_buf_k = k.view(np.uint16)
    input_buf_v = v.view(np.uint16)
    
    gx=4
    gy=1
    gz=4
    
    tx=1
    ty=1
    tz=4
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                f, t, h,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1))
    
    print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    # print("----------------[PASS]----------------")
    exit()
    from original_flash_att import flash_attention
    ref_C = flash_attention(q, k, v)
    # np.testing.assert_array_equal(origin_C, ref_C)
    print(f"==>> ref_C: {ref_C}")


def test_flash_decoding_llama2_split_kv():

    def _build_bench_dev(q, k, v, output_C,
                        f, t, h,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={2048} -DHEAD_DIM={128} "
        _define += f"-DTILE_Q={1} -DTILE_KV={12} -DTILE_HEAD={128} -DSPLIT_KV={12} -DHEAD_SCALE=0.001 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF "
        _define += f" -Qxcm_doubleGRF  -mCM_printregusage"

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_decoding_split_kv.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), # for dispatching
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))


        return temp_res
    
    
    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    llama2_q_shape = [1, 32, 1, 128]
    llama2_k_shape = [1, 32, 2048, 128]
    llama2_v_shape = [1, 32, 2048, 128]
    llama2_output_shape = llama2_q_shape
    
    f = 16  
    t = 16  
    h = 8   

    # q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    # k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    # v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 

    dst_path = "./flash_decoding_json"
    q = np.load(os.path.join(dst_path, "q_tensor.npy")).reshape(llama2_q_shape)
    k = np.load(os.path.join(dst_path, "k_tensor.npy")).reshape(llama2_k_shape)
    v = np.load(os.path.join(dst_path, "v_tensor.npy")).reshape(llama2_v_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_output.npy")).reshape(llama2_output_shape)
    # print(f"==>> q: {q}")
    # print(f"==>> k: {k}")
    # print(f"==>> v: {v}")
    # print(f"==>> output: {output}")
    
    # q = np.ones((f, h)).astype("float16")
    # k = np.ones((t, h)).astype("float16")
    # v = np.ones((t, h)).astype("float16")
    
    
    input_buf_q = q.view(np.uint16)
    input_buf_k = k.view(np.uint16)
    input_buf_v = v.view(np.uint16)
    
    gx=32
    gy=1
    gz=12
    
    tx=1
    ty=1
    tz=12
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                f, t, h,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1e3))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-2)
    print("----------------[PASS]----------------")
    exit()
    from original_flash_att import flash_attention
    ref_C = flash_attention(q, k, v)
    # np.testing.assert_array_equal(origin_C, ref_C)
    print(f"==>> ref_C: {ref_C}")



if __name__ == "__main__":
    # test_cm_softmax_conformance()
    # test_cm_onlines_oftmax_dev()
    # test_flash_att()
    # test_flash_decoding_small_shape()
    # test_flash_decoding_llama2_shape()
    # test_flash_decoding_small_split_kv()
    test_flash_decoding_llama2_split_kv()

