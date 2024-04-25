import os
import json

import pandas as pd
import numpy as np
np.random.seed(123)


dxdispatch = r"DxDispatchRelWithDebInfo\dxdispatch.exe"


def launch_dxdispatch(input_file):
    cmd_launch = f"{dxdispatch} -w 10 -r 10 -i 10  {input_file}"
    print(cmd_launch)
    os.system(cmd_launch)



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

    
def gen_q_k_v_input(input_json = None):
    with open(input_json, 'r') as f:
        data = json.load(f)

    q_name  = data['resources']['query']['initialValues']['sourcePath']
    q_shape =  data['dispatchables']['mha']['desc']['QueryTensor']['Sizes']
    print(f"==>> q_name: {q_name},  q_shape: {q_shape}")
    
    k_name  = data['resources']['key']['initialValues']['sourcePath']
    k_shape =  data['dispatchables']['mha']['desc']['KeyTensor']['Sizes']
    print(f"==>> k_name: {k_name},  k_shape: {k_shape}")
    
    v_name  = data['resources']['value']['initialValues']['sourcePath']
    v_shape =  data['dispatchables']['mha']['desc']['ValueTensor']['Sizes']
    print(f"==>> v_name: {v_name},  v_shape: {v_shape}")
    q_tensor = np.random.uniform(-1, 1, q_shape).astype("float16")
    k_tensor = np.random.uniform(-1, 1, k_shape).astype("float16")
    v_tensor = np.random.uniform(-1, 1, v_shape).astype("float16")
    # qkv_tensor = np.ones(q_shape).astype("float16")

    dst_path = os.path.dirname(q_name)
    print(f"==>> dst_path: {dst_path}")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    np.save(q_name, q_tensor)
    np.save(k_name, k_tensor)
    np.save(v_name, v_tensor)
    # q_tensor = np.load(q_name)
    # k_tensor = np.load(k_name)
    # v_tensor = np.load(v_name)

        
        

if __name__ == "__main__":

    # small shape test 
    # mha_json = "./flash_decoding_json/dml_mha_q_k_v_small.json"
    # gen_q_k_v_input(input_json = mha_json)
    # launch_dxdispatch(mha_json)

    # LLAMA2 shape test 
    # mha_json = "./flash_decoding_json/dml_mha_q_k_v2048.json"
    # gen_q_k_v_input(input_json = mha_json)
    # launch_dxdispatch(mha_json)

