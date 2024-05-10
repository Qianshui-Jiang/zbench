import numpy as np
import zbench
import os
import sys
sys.path.append(os.path.dirname(__file__))
np.random.seed(123)
np.set_printoptions(edgeitems=30, linewidth=100000)


def test_flash_decoding_mc_shader():

    def _build_bench_dev(q, k, v, 
                        past_seq_len, 
                        output_present_k,
                        output_present_v,
                        output_C,
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
        _define += f" -Qxcm_jit_option=-noLocalSplit -Qxcm_jit_option=-globalTokenAllocation -Qxcm_jit_option=-enableBCR -Qxcm_jit_option=-SWSBDepReduction  "
        

        _include = f"-I . "
        build_opt = _include + _define
        temp_res  = zbench.launch_rt_igdext(cm_file = "./shaders/mha_q_k_v_flash_decoding.cpp", 
                                            build_options = build_opt,
                                            input_q=q, input_k=k, input_v=v, 
                                            past_seq_len = past_seq_len, 
                                            output_present_k = output_present_k,
                                            output_present_v = output_present_v,
                                            output_c = output_C,
                                            thg_x=int(gx/tx), thg_y=int(gy/ty), thg_z=int(gz/tz), 
                                            iter_nums=iter_num)

        temp_res = np.array(temp_res,dtype="uint16").view(np.float16).reshape(q.shape)
        # temp_res = np.array(temp_res,dtype="uint16").view(np.float16)
        # temp_res = np.array(temp_res,dtype="uint32").view(np.float32).reshape((m,n))
        # temp_res = np.array(temp_res,dtype="uint16").reshape((m,n))

        return temp_res
    
    # [batchSize, headCount, keyValueSequenceLength, headSize(headDIM)]
    # [batch_size, seq_len, self.n_heads, self.head_dim]
    
    llama2_q_shape = [1, 1, 1, 4096] # 32 x 128, q seq_len = 1
    llama2_k_shape = [1, 1, 1, 4096] # 32 x 128, kv seq_len = 1
    llama2_v_shape = [1, 1, 1, 4096] # 32 x 128, kv seq_len = 1
    llama2_past_seq_len_shape = [1, 1, 1, 1, 1]
    llama2_output_present_k_shape = [1, 32, 2048, 128]
    llama2_output_present_v_shape = [1, 32, 2048, 128]
    
    
    llama2_output_shape = llama2_q_shape


    q = np.random.uniform(-1, 1, llama2_q_shape).astype("float16")
    k = np.random.uniform(-1, 1, llama2_k_shape).astype("float16")
    v = np.random.uniform(-1, 1, llama2_v_shape).astype("float16")
    

    past_seq_len_value = 1000
    past_seq_len = np.array(past_seq_len_value).reshape(1, 1, 1, 1, 1).astype("uint32")
    
    output_present_k = np.random.uniform(-1, 1, llama2_output_present_k_shape).astype("float16")
    output_present_v = np.random.uniform(-1, 1, llama2_output_present_v_shape).astype("float16")
    
    input_buf_q = q.view(np.uint16)
    input_buf_k = k.view(np.uint16)
    input_buf_v = v.view(np.uint16)
    
    input_buf_past_seq_len = past_seq_len.view(np.uint16)
    input_buf_output_present_k = output_present_k.view(np.uint16)
    input_buf_output_present_v = output_present_v.view(np.uint16)
    
    
    gx=32
    gy=1
    gz=1
    
    tx=1
    ty=1
    tz=1
    output_C = np.zeros_like(input_buf_q)
    origin_C = _build_bench_dev(input_buf_q, input_buf_k, input_buf_v, 
                                input_buf_past_seq_len, 
                                input_buf_output_present_k,
                                input_buf_output_present_v,
                                output_C,
                                gx, gy, gz, 
                                tx, ty, tz, 
                                iter_num=int(1000))
    
    # print(f"==>> q:\n {q}")
    # print(f"==>> origin_C:\n {origin_C}")
    print("----------------[PASS]----------------")



if __name__ == "__main__":
    test_flash_decoding_mc_shader()