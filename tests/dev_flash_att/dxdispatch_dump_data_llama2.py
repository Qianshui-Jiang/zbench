import os
import json

import pandas as pd
import numpy as np
np.random.seed(123)


dxdispatch = r"D:\dml_workspace\zbench\tests\dev_flash_att\DxDispatchRelWithDebInfo\dxdispatch.exe"


def launch_dxdispatch(input_file):
    cmd_launch = f"{dxdispatch} -w 10 -r 10 -i 10  {input_file}"
    print(cmd_launch)
    os.system(cmd_launch)


def gen_temp_json_from(template_json = None):
    
    dst_path = os.path.dirname(template_json)
    print(f"==>> dst_path: {dst_path}")
    with open("dml_gemm.json", 'r') as f:
        data = json.load(f)

    q_shape = [1, 32, 1, 128]
    k_shape = [1, 32, 2048, 128]
    v_shape = [1, 32, 2048, 128]
    output_shape = [1, 32, 1, 128]


    # Setup shapes
    data['dispatchables']['mha']['desc']['QueryTensor']['Sizes'] = q_shape
    data['dispatchables']['mha']['desc']['KeyTensor']['Sizes'] = k_shape
    data['dispatchables']['mha']['desc']['ValueTensor']['Sizes'] = v_shape
    data['dispatchables']['mha']['desc']['OutputTensor']['Sizes'] = output_shape

    with open(os.path.join(dst_path, "tmp_dml_MHA.json"), 'w') as f:
        json.dump(data, f)


def gen_small_input_npy_buffer_flash_decoding(dst_path = None):
    q_shape = [1, 4, 1, 8]
    k_shape = [1, 4, 16, 8]
    v_shape = [1, 4, 16, 8]
    output_shape = [1, 4, 1, 8]

    q_tensor = np.random.uniform(0, 1, q_shape).astype("float16")
    k_tensor = np.random.uniform(0, 1, k_shape).astype("float16")
    v_tensor = np.random.uniform(0, 1, v_shape).astype("float16")

    # q_tensor = np.ones(q_shape).astype("float16")
    # k_tensor = np.ones(k_shape).astype("float16")
    # v_tensor = np.ones(v_shape).astype("float16")

    np.save(os.path.join(dst_path, "q_tensor_small.npy"), q_tensor)
    np.save(os.path.join(dst_path, "k_tensor_small.npy"), k_tensor)
    np.save(os.path.join(dst_path, "v_tensor_small.npy"), v_tensor)
    
    q_tensor = np.load(os.path.join(dst_path, "q_tensor_small.npy"))
    k_tensor = np.load(os.path.join(dst_path, "k_tensor_small.npy"))
    v_tensor = np.load(os.path.join(dst_path, "v_tensor_small.npy"))


def gen_llama2_input_npy_buffer_flash_decoding(dst_path = None):
    q_shape = [1, 32, 1, 128]
    k_shape = [1, 32, 2048, 128]
    v_shape = [1, 32, 2048, 128]
    output_shape = [1, 32, 1, 128]

    q_tensor = np.random.uniform(-1, 1, q_shape).astype("float16")
    k_tensor = np.random.uniform(-1, 1, k_shape).astype("float16")
    v_tensor = np.random.uniform(-1, 1, v_shape).astype("float16")

    # q_tensor = np.ones(q_shape).astype("float16")
    # k_tensor = np.ones(k_shape).astype("float16")
    # v_tensor = np.ones(v_shape).astype("float16")

    np.save(os.path.join(dst_path, "q_tensor.npy"), q_tensor)
    np.save(os.path.join(dst_path, "k_tensor.npy"), k_tensor)
    np.save(os.path.join(dst_path, "v_tensor.npy"), v_tensor)

    # test loading
    q_tensor = np.load(os.path.join(dst_path, "q_tensor.npy"))
    k_tensor = np.load(os.path.join(dst_path, "k_tensor.npy"))
    v_tensor = np.load(os.path.join(dst_path, "v_tensor.npy"))


if __name__ == "__main__":


    # small shape test 
    gen_small_input_npy_buffer_flash_decoding(dst_path = "./flash_decoding_json")
    mha_json = "./flash_decoding_json/dml_mha_q_k_v_small.json"
    launch_dxdispatch(mha_json)

    # LLAMA2 shape test 
    gen_llama2_input_npy_buffer_flash_decoding(dst_path = "./flash_decoding_json")
    mha_json = "./flash_decoding_json/dml_mha_q_k_v2048.json"
    launch_dxdispatch(mha_json)

