import numpy as np
import zbench
import os
import sys
sys.path.append(os.path.dirname(__file__))
np.random.seed(123)
np.set_printoptions(edgeitems=30, linewidth=100000)


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


def test_flash_att_sd_q_kv_small():

    def _build_bench_dev(q, kv, output_C,
                        batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={8} -DTILE_KV={4} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 -DITEMNUM_PER_HW=16"
        _define += f" -Qxcm_doubleGRF -mCM_printregusage"
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv_tile_q.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 4096, 320]
    sd_kv_shape = [2, 77, 8, 2, 40]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 4096
    kv_seq_len = 77
    head_dim = 40
    head_count = 8

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q_small_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv_small_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv_small_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    # q_ref = q[0, :, 0:8]
    # print(f"==>> q_ref: {q_ref}")
    # # print(f"==>> q_ref.shape: {q_ref.shape}")
    # k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # # print(f"==>> k_ref.shape: {k_ref.shape}")
    # v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:8]
    # # from original_flash_att import flash_attention
    # # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # # # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # # exit()


    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=int(4096/8)  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv,  output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")

def test_flash_att_sd_q_kv64():

    def _build_bench_dev(q, kv, output_C,
                        batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={1} -DTILE_KV={4} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 -DITEMNUM_PER_HW=16"
        _define += f" -Qxcm_doubleGRF "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 64, 1280]
    sd_kv_shape = [2, 77, 8, 2, 160]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 64
    kv_seq_len = 77
    head_count = 8
    head_dim = 160

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q64_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv64_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv64_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    q_ref = q[0, :, 0:head_dim]
    # print(f"==>> q_ref: {q_ref}")
    # print(f"==>> q_ref.shape: {q_ref.shape}")
    k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # print(f"==>> k_ref.shape: {k_ref.shape}")
    v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:head_dim]
    # from original_flash_att import flash_attention
    # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # exit()

    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=64  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv, output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")

def test_flash_att_sd_q_kv64_tile_q():

    def _build_bench_dev(q, kv, output_C,
                        batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={2} -DTILE_KV={4} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 "
        _define += f" -Qxcm_doubleGRF "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv_tile_q.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 64, 1280]
    sd_kv_shape = [2, 77, 8, 2, 160]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 64
    kv_seq_len = 77
    head_count = 8
    head_dim = 160

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q64_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv64_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv64_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    q_ref = q[0, :, 0:head_dim]
    # print(f"==>> q_ref: {q_ref}")
    # print(f"==>> q_ref.shape: {q_ref.shape}")
    k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # print(f"==>> k_ref.shape: {k_ref.shape}")
    v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:head_dim]
    # from original_flash_att import flash_attention
    # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # exit()

    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=int(64/2)  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv, output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    # print(f"==>> output_ref: {output_ref}")
    print("----------------[PASS]----------------")

def test_flash_att_sd_q_kv256():

    def _build_bench_dev(q, kv, output_C,
                        batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={1} -DTILE_KV={4} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 -DITEMNUM_PER_HW=16"
        _define += f" -Qxcm_doubleGRF -mCM_printregusage "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 256, 1280]
    sd_kv_shape = [2, 77, 8, 2, 160]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 256
    kv_seq_len = 77
    head_count = 8
    head_dim = 160

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q256_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv256_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv256_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    q_ref = q[0, :, 0:head_dim]
    # print(f"==>> q_ref: {q_ref}")
    # print(f"==>> q_ref.shape: {q_ref.shape}")
    k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # print(f"==>> k_ref.shape: {k_ref.shape}")
    v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:head_dim]
    # from original_flash_att import flash_attention
    # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # exit()

    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=int(256/1)  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv, output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")

def test_flash_att_sd_q_kv256_tile_q():

    def _build_bench_dev(q, kv, output_C,
                        batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                        gx, gy, gz,
                        tx, ty, tz,  
                        iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={2} -DTILE_KV={3} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 -DITEMNUM_PER_HW=16"
        _define += f" -Qxcm_doubleGRF -mCM_printregusage "
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv_tile_q.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 256, 1280]
    sd_kv_shape = [2, 77, 8, 2, 160]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 256
    kv_seq_len = 77
    head_count = 8
    head_dim = 160

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q256_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv256_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv256_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    q_ref = q[0, :, 0:head_dim]
    # print(f"==>> q_ref: {q_ref}")
    # print(f"==>> q_ref.shape: {q_ref.shape}")
    k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # print(f"==>> k_ref.shape: {k_ref.shape}")
    v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:head_dim]
    # from original_flash_att import flash_attention
    # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # exit()

    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=int(256/2)  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv, output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")

def test_flash_att_sd_q_kv1024():
    
    def _build_bench_dev(q, kv, output_C,
                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                gx, gy, gz,
                tx, ty, tz,  
                iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={1} -DTILE_KV={4} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 -DITEMNUM_PER_HW=16"
        # _define += f" -mCM_printregusage"
        # _define += f" -Qxcm_doubleGRF -mCM_printregusage"
        
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 1024, 640]
    sd_kv_shape = [2, 77, 8, 2, 80]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 1024
    kv_seq_len = 77
    head_dim = 80
    head_count = 8

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q1024_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv1024_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv1024_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    # q_ref = q[0, :, 0:8]
    # print(f"==>> q_ref: {q_ref}")
    # # print(f"==>> q_ref.shape: {q_ref.shape}")
    # k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # # print(f"==>> k_ref.shape: {k_ref.shape}")
    # v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:8]
    # # from original_flash_att import flash_attention
    # # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # # # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # # exit()


    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=int(1024/1)  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv,  output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")
    
def test_flash_att_sd_q_kv1024_tile_q():
    
    def _build_bench_dev(q, kv, output_C,
                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                gx, gy, gz,
                tx, ty, tz,  
                iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={4} -DTILE_KV={7} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 -DITEMNUM_PER_HW=16"
        # _define += f" -mCM_printregusage"
        _define += f" -Qxcm_doubleGRF -mCM_printregusage"
        
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv_tile_q.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 1024, 640]
    sd_kv_shape = [2, 77, 8, 2, 80]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 1024
    kv_seq_len = 77
    head_dim = 80
    head_count = 8

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q1024_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv1024_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv1024_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    # q_ref = q[0, :, 0:8]
    # print(f"==>> q_ref: {q_ref}")
    # # print(f"==>> q_ref.shape: {q_ref.shape}")
    # k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # # print(f"==>> k_ref.shape: {k_ref.shape}")
    # v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:8]
    # # from original_flash_att import flash_attention
    # # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # # # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # # exit()


    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=int(1024/4)  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv,  output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")

def test_flash_att_sd_q_kv4096():

    def _build_bench_dev(q, kv, output_C,
                    batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                    gx, gy, gz,
                    tx, ty, tz,  
                    iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={1} -DTILE_KV={4} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 -DITEMNUM_PER_HW=16"
        # _define += f" -Qxcm_doubleGRF -mCM_printregusage"
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 4096, 320]
    sd_kv_shape = [2, 77, 8, 2, 40]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 4096
    kv_seq_len = 77
    head_dim = 40
    head_count = 8

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q4096_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv4096_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv4096_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    # q_ref = q[0, :, 0:8]
    # print(f"==>> q_ref: {q_ref}")
    # # print(f"==>> q_ref.shape: {q_ref.shape}")
    # k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # # print(f"==>> k_ref.shape: {k_ref.shape}")
    # v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:8]
    # # from original_flash_att import flash_attention
    # # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # # # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # # exit()


    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=int(4096/1)  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv,  output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")
    

def test_flash_att_sd_q_kv4096_tile_q():

    def _build_bench_dev(q, kv, output_C,
                    batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                    gx, gy, gz,
                    tx, ty, tz,  
                    iter_num):
        # ACCU_IS_FP32 will have impact on performance
        _define =  f" -DBATCH_SIZE={batch_size} -DQ_SEQ_LEN={q_seq_len} -DKV_SEQ_LEN={kv_seq_len} -DHEAD_COUNT={head_count} -DHEAD_DIM={head_dim}"
        _define += f" -DTILE_Q={8} -DTILE_KV={4} -DTILE_HEAD={head_dim} -DHEAD_SCALE=0.001"
        _define += f" -DGWS_SIZE_X={gx} -DGWS_SIZE_Y={gy} -DGWS_SIZE_Z={gz}"
        _define += f" -DLWS_SIZE_X={tx} -DLWS_SIZE_Y={ty} -DLWS_SIZE_Z={tz}"
        _define += f" -DCM_BINDLESS=1 -DITEMNUM_PER_HW=16"
        _define += f" -Qxcm_doubleGRF -mCM_printregusage"
        # _define += f"-mdump_asm -Qxcm_doubleGRF -mCM_printregusage "

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./dev_flash_att_q_kv_tile_q.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_kv=kv, output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)


        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(output_C.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res

    # Q Tensor layout:  [batchSize, sequenceLength, hiddenSize]
    # KV Tensor layout: [batchSize, keyValueSequenceLength, headCount, 2, headSize]

    sd_q_shape = [2, 4096, 320]
    sd_kv_shape = [2, 77, 8, 2, 40]
    sd_output_shape = sd_q_shape

    batch_size = 2
    q_seq_len = 4096
    kv_seq_len = 77
    head_dim = 40
    head_count = 8

    dst_path = "./flash_att2_SD_json/tensor_file"
    q = np.load(os.path.join(dst_path, "q4096_np_tensor.npy")).reshape(sd_q_shape)
    kv = np.load(os.path.join(dst_path, "kv4096_np_tensor.npy")).reshape(sd_kv_shape)
    output_ref = np.load(os.path.join(dst_path, "dml_mha_q_kv4096_output.npy")).reshape(sd_output_shape)
    # print(f"==>> q.shape: {q.shape}")
    # print(f"==>> kv.shape: {kv.shape}")
    # print(f"==>> output_ref.shape: {output_ref.shape}")
    # print(f"==>> q: {q}")
    # print(f"==>> kv: {kv}")
    # print(f"==>> output_ref: {output_ref}")
    # q_ref = q[0, :, 0:8]
    # print(f"==>> q_ref: {q_ref}")
    # # print(f"==>> q_ref.shape: {q_ref.shape}")
    # k_ref = kv[0, :, 0, 0, :]
    # print(f"==>> k_ref: {k_ref}")
    # # print(f"==>> k_ref.shape: {k_ref.shape}")
    # v_ref = kv[0, :, 0, 1, :]
    # print(f"==>> v_ref: {v_ref}")
    # # print(f"==>> v_ref.shape: {v_ref.shape}")
    # output_1rank = output_ref[0, :, 0:8]
    # # from original_flash_att import flash_attention
    # # ref_C_1rank = flash_attention(q_ref, k_ref, v_ref)
    # # # print(f"==>> output_1rank: {output_1rank}")
    # print(f"==>> output_1rank.shape: {output_1rank.shape}")
    # np.testing.assert_allclose(ref_C_1rank, output_1rank, atol=1e-3)
    # # exit()


    # input_buf_q = q_ref.view(np.uint16)
    # input_buf_k = k_ref.view(np.uint16)
    # input_buf_v = v_ref.view(np.uint16)
    input_buf_q = q.view(np.uint16)
    input_buf_kv = kv.view(np.uint16)


    gx=2  # Batch size parallel
    gy=int(4096/8)  # Q_SEQ_LEN parallel, TILE_Q items per thread
    gz=8  # HEAD_COUNT parallel
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(output_ref)
    origin_C = _build_bench_dev(input_buf_q, input_buf_kv,  output_C,
                                batch_size, q_seq_len, kv_seq_len, head_count, head_dim,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(100))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C[0, :, 0:8]}")
    # print(f"==>> ref_C: {ref_C}")
    # np.testing.assert_array_equal(origin_C, ref_C_1rank)
    
    # print(f"==>> q: {q[0, :, 8:]}")
    # print(f"==>> k: {kv[:, :, :, :, :]}")
    # print(f"==>> v: {kv[0, :, 0, :, :]}")
    # print(f"==>> output_ref: {output_ref}")
    
    np.testing.assert_allclose(origin_C, output_ref, atol=1e-3)
    print("----------------[PASS]----------------")
    
    
if __name__ == "__main__":
    # test_flash_att()
    # test_flash_att_sd_q_kv_small()
    # test_flash_att_sd_q_kv64()
    # test_flash_att_sd_q_kv64_tile_q()
    
    # test_flash_att_sd_q_kv256()
    # test_flash_att_sd_q_kv256_tile_q()
    
    # test_flash_att_sd_q_kv1024()
    test_flash_att_sd_q_kv1024_tile_q()
    
    # test_flash_att_sd_q_kv4096()
    # test_flash_att_sd_q_kv4096_tile_q()