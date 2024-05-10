import numpy as np
import zbench
import os
import sys
sys.path.append(os.path.dirname(__file__))
np.random.seed(123)
np.set_printoptions(edgeitems=30, linewidth=100000)

from dummy_flash_att_test import  flash_attention2, flash_attention, naive_attention

def test_flash_decoding_small_shape_no_kv_tiling():

    def _build_bench_dev(q, k, v, output_C,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={16} -DHEAD_COUNT={4} -DHEAD_DIM={8} "
        _define += f"-DTILE_Q={1} -DTILE_HEAD={8} -DHEAD_SCALE=1 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./shaders/dev_flash_decoding_no_kv_tiling.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)


        return temp_res

    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]

    llama2_q_small_shape = [1, 1, 1,  32]
    llama2_k_small_shape = [1, 1, 16, 32]
    llama2_v_small_shape = [1, 1, 16, 32]
    llama2_output_small_shape = llama2_q_small_shape


    dst_path = "./flash_decoding_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q_tensor_small.npy")).reshape(llama2_q_small_shape)
    k = np.load(os.path.join(dst_path, "k_tensor_small.npy")).reshape(llama2_k_small_shape)
    v = np.load(os.path.join(dst_path, "v_tensor_small.npy")).reshape(llama2_v_small_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_output.npy")).reshape(llama2_output_small_shape)
    
    
    q_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_query.npy")).reshape(llama2_q_small_shape)
    k_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_key.npy")).reshape(llama2_k_small_shape)
    v_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_value.npy")).reshape(llama2_v_small_shape)
    np.testing.assert_allclose(q_ref, q)
    np.testing.assert_allclose(k_ref, k)
    np.testing.assert_allclose(v_ref, v)


    dummy_output = np.zeros_like(output_ref)
    dummy_output[0, 0, 0, 0:8]= flash_attention2(q_ref[0, 0, :,  0:8], k_ref[0, 0, :  ,0:8], v_ref[0, 0,   :,0:8],head_scale=1)
    dummy_output[0, 0, 0, 8:16]= flash_attention2(q_ref[0, 0, :, 8:16], k_ref[0, 0, : ,8:16], v_ref[0, 0,  :,8:16],head_scale=1)
    dummy_output[0, 0, 0, 16:24]= flash_attention2(q_ref[0, 0, :,16:24], k_ref[0, 0, :,16:24], v_ref[0, 0, :,16:24],head_scale=1)
    dummy_output[0, 0, 0, 24:32]= flash_attention2(q_ref[0, 0, :,24:32], k_ref[0, 0, :,24:32], v_ref[0, 0, :,24:32],head_scale=1)
    np.testing.assert_allclose(output_ref, dummy_output, atol=1e-3)
    # q_dummy_input = q

    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> k.shape: {k.shape}")
    # print(f"==>> v.shape: {v.shape}")
    # print(f"==>> output: {output_ref}")

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
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1e3))

    np.testing.assert_allclose(origin_C, dummy_output, atol=1e-3)
    print("----------------[PASS]----------------")

    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")


def test_flash_decoding_small_split_kv_tiling():

    def _build_bench_dev(q, k, v, output_C,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={16} -DHEAD_COUNT={4} -DHEAD_DIM={8} "
        _define += f"-DTILE_Q={1} -DTILE_KV={2} -DTILE_HEAD={8} -DSPLIT_KV={4} -DHEAD_SCALE=1 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./shaders/dev_flash_decoding_split_kv_tiling.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)

        return temp_res

    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    llama2_q_small_shape = [1, 4, 1, 8]
    llama2_k_small_shape = [1, 4, 16, 8]
    llama2_v_small_shape = [1, 4, 16, 8]
    llama2_output_small_shape = llama2_q_small_shape

    # q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    # k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    # v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 

    dst_path = "./flash_decoding_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q_tensor_small.npy")).reshape(llama2_q_small_shape)
    k = np.load(os.path.join(dst_path, "k_tensor_small.npy")).reshape(llama2_k_small_shape)
    v = np.load(os.path.join(dst_path, "v_tensor_small.npy")).reshape(llama2_v_small_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_output.npy")).reshape(llama2_output_small_shape)

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
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1e3))
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")


def test_flash_decoding_llama2_shape_kv_tiling():

    def _build_bench_dev(q, k, v, output_C,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={2048} -DHEAD_COUNT={32} -DHEAD_DIM={128} "

        _define += f"-DTILE_Q={1} -DTILE_KV={8} -DTILE_HEAD={128} -DHEAD_SCALE=0.0883883 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./shaders/dev_flash_decoding_kv_tiling.cpp", 
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
    
    llama2_q_shape = [1, 1, 1, 4096]
    llama2_k_shape = [1, 1, 2048, 4096]
    llama2_v_shape = [1, 1, 2048, 4096]
    llama2_output_shape = llama2_q_shape
    


    dst_path = "./flash_decoding_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q_tensor.npy")).reshape(llama2_q_shape)
    k = np.load(os.path.join(dst_path, "k_tensor.npy")).reshape(llama2_k_shape)
    v = np.load(os.path.join(dst_path, "v_tensor.npy")).reshape(llama2_v_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_output.npy")).reshape(llama2_output_shape)

    # q_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_query.npy")).reshape(llama2_q_shape)
    # k_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_key.npy")).reshape(llama2_k_shape)
    # v_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_value.npy")).reshape(llama2_v_shape)
    # np.testing.assert_allclose(q_ref, q)
    # np.testing.assert_allclose(k_ref, k)
    # np.testing.assert_allclose(v_ref, v)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> k.shape: {k.shape}")
    # print(f"==>> v.shape: {v.shape}")
    
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
    gz=1
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1e3))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-4)
    print("----------------[PASS]----------------")


def test_flash_decoding_llama2_shape_no_kv_tiling():

    def _build_bench_dev(q, k, v, output_C,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={2048} -DHEAD_COUNT={32} -DHEAD_DIM={128} "

        _define += f"-DTILE_Q={1} -DTILE_KV={1} -DTILE_HEAD={128} -DHEAD_SCALE=0.0883883 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "
        _define += f" -Qxcm_doubleGRF "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./shaders/dev_flash_decoding_no_kv_tiling.cpp", 
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
    
    llama2_q_shape = [1, 1, 1, 4096]
    llama2_k_shape = [1, 1, 2048, 4096]
    llama2_v_shape = [1, 1, 2048, 4096]
    llama2_output_shape = llama2_q_shape
    

    dst_path = "./flash_decoding_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q_tensor.npy")).reshape(llama2_q_shape)
    k = np.load(os.path.join(dst_path, "k_tensor.npy")).reshape(llama2_k_shape)
    v = np.load(os.path.join(dst_path, "v_tensor.npy")).reshape(llama2_v_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_output.npy")).reshape(llama2_output_shape)

    # q_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_query.npy")).reshape(llama2_q_shape)
    # k_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_key.npy")).reshape(llama2_k_shape)
    # v_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_value.npy")).reshape(llama2_v_shape)
    # np.testing.assert_allclose(q_ref, q)
    # np.testing.assert_allclose(k_ref, k)
    # np.testing.assert_allclose(v_ref, v)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> k.shape: {k.shape}")
    # print(f"==>> v.shape: {v.shape}")
    
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
    gz=1
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1000))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")


def test_flash_decoding_llama2_split_kv_tiling():

    def _build_bench_dev(q, k, v, output_C,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={2048} -DHEAD_COUNT={32} -DHEAD_DIM={128} "
        _define += f"-DTILE_Q={1} -DTILE_KV={2} -DTILE_HEAD={128} -DSPLIT_KV={8} -DHEAD_SCALE=0.0883883 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        print(f"==>> _define: {_define}")
        # _define += f"-mdump_asm -Qxcm_doubleGRF "
        _define += f" -Qxcm_doubleGRF  -mCM_printregusage"
        # _define += f" -mCM_no_debug -Qunused-arguments"
        _define += f" -Qxcm_jit_option=-noLocalSplit -Qxcm_jit_option=-globalTokenAllocation -Qxcm_jit_option=-enableBCR -Qxcm_jit_option=-SWSBDepReduction  "


        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./shaders/dev_flash_decoding_split_kv_tiling.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), # for dispatching
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)


        return temp_res
    
    
    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    llama2_q_shape = [1, 1, 1, 4096]
    llama2_k_shape = [1, 1, 2048, 4096]
    llama2_v_shape = [1, 1, 2048, 4096]
    llama2_output_shape = llama2_q_shape
    
    # q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    # k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    # v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 

    dst_path = "./flash_decoding_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q_tensor.npy")).reshape(llama2_q_shape)
    k = np.load(os.path.join(dst_path, "k_tensor.npy")).reshape(llama2_k_shape)
    v = np.load(os.path.join(dst_path, "v_tensor.npy")).reshape(llama2_v_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_output.npy")).reshape(llama2_output_shape)
    
    # q_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_query.npy")).reshape(llama2_q_shape)
    # k_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_key.npy")).reshape(llama2_k_shape)
    # v_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_value.npy")).reshape(llama2_v_shape)
    # np.testing.assert_allclose(q_ref, q)
    # np.testing.assert_allclose(k_ref, k)
    # np.testing.assert_allclose(v_ref, v)
    
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
    gz=8
    
    tx=1
    ty=1
    tz=8
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1000))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-4)
    print("----------------[PASS]----------------")


def test_flash_decoding_llama2_split_kv_no_tiling():

    def _build_bench_dev(q, k, v, output_C,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f"-DQ_SEQ_LEN={1} -DKV_SEQ_LEN={2048} -DHEAD_COUNT={32} -DHEAD_DIM={128} "
        _define += f"-DTILE_Q={1} -DTILE_KV={1} -DTILE_HEAD={128} -DSPLIT_KV={8} -DHEAD_SCALE=0.0883883 "
        _define += f"-DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz} "
        _define += f"-DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz} "
        _define += f"-DCM_BINDLESS=1 -DITEMNUM_PER_HW=16 "
        # _define += f"-mdump_asm -Qxcm_doubleGRF "
        _define += f" -Qxcm_doubleGRF  -mCM_printregusage"
        # _define += f" -mCM_no_debug -Qunused-arguments"
        _define += f" -Qxcm_jit_option=-noLocalSplit -Qxcm_jit_option=-globalTokenAllocation -Qxcm_jit_option=-enableBCR -Qxcm_jit_option=-SWSBDepReduction  "


        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./shaders/dev_flash_decoding_split_kv_no_tiling.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), # for dispatching
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)


        return temp_res
    
    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    llama2_q_shape = [1, 1, 1, 4096]
    llama2_k_shape = [1, 1, 2048, 4096]
    llama2_v_shape = [1, 1, 2048, 4096]
    llama2_output_shape = llama2_q_shape
    
    # q = np.random.uniform(0, 1, size=(f, h)).astype("float16") 
    # k = np.random.uniform(0, 1, size=(t, h)).astype("float16") 
    # v = np.random.uniform(0, 1, size=(t, h)).astype("float16") 

    dst_path = "./flash_decoding_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q_tensor.npy")).reshape(llama2_q_shape)
    k = np.load(os.path.join(dst_path, "k_tensor.npy")).reshape(llama2_k_shape)
    v = np.load(os.path.join(dst_path, "v_tensor.npy")).reshape(llama2_v_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_output.npy")).reshape(llama2_output_shape)
    
    # q_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_query.npy")).reshape(llama2_q_shape)
    # k_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_key.npy")).reshape(llama2_k_shape)
    # v_ref = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_value.npy")).reshape(llama2_v_shape)
    # np.testing.assert_allclose(q_ref, q)
    # np.testing.assert_allclose(k_ref, k)
    # np.testing.assert_allclose(v_ref, v)
    
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
    gz=8
    
    tx=1
    ty=1
    tz=8
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, output_C,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1000))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-4)
    print("----------------[PASS]----------------")


if __name__ == "__main__":
    # test_flash_decoding_small_shape_no_kv_tiling()
    # test_flash_decoding_small_split_kv_tiling()

    # test_flash_decoding_llama2_shape_kv_tiling()
    # test_flash_decoding_llama2_shape_no_kv_tiling()
    # test_flash_decoding_llama2_split_kv_tiling()
    test_flash_decoding_llama2_split_kv_no_tiling()