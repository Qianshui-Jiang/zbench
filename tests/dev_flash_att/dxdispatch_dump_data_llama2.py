import sys
import os
import re
import json
import time
import shlex
import shutil
import datetime
import argparse

import pandas as pd
import numpy as np
np.random.seed(123)

from functools import reduce
import operator

log_file = True 


# dxdispatch = r"D:\dml_workspace\zbench\tests\dev_flash_att\DxDispatch\dxdispatch.exe"
dxdispatch = r"D:\dml_workspace\zbench\tests\dev_flash_att\DxDispatchRelWithDebInfo\dxdispatch.exe"


def launch_dxdispatch(input_file):
    cmd_launch = f"{dxdispatch} -w 10 -r 10 -i 10  {input_file}"
    print(cmd_launch)
    os.system(cmd_launch)
    # res = os.popen(cmd_launch)
    # print(res.readlines()[:-1])
    # res_log = "dxdispatch.log"
    # date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    # with open(res_log, 'a+') as f:
    #     f.write(f'--------dxdispatch-log-{date_time}------\n ')
    #     for item in res.readlines()[:-1]:
    #         f.write(f'{item}')
    #     f.write('\n')


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
    output_tensor = np.zeros(output_shape)
    output_tensor = np.ones(output_shape) * 0

    # q_tensor = np.ones(q_shape).astype("float16")
    # k_tensor = np.ones(k_shape).astype("float16")
    # v_tensor = np.ones(v_shape).astype("float16")

    np.save(os.path.join(dst_path, "q_tensor_small.npy"), q_tensor)
    np.save(os.path.join(dst_path, "k_tensor_small.npy"), k_tensor)
    np.save(os.path.join(dst_path, "v_tensor_small.npy"), v_tensor)
    np.save(os.path.join(dst_path, "output_tensor_small.npy"), output_tensor)
    
    q_tensor = np.load(os.path.join(dst_path, "q_tensor_small.npy"))
    k_tensor = np.load(os.path.join(dst_path, "k_tensor_small.npy"))
    v_tensor = np.load(os.path.join(dst_path, "v_tensor_small.npy"))
    outout_tensor = np.load(os.path.join(dst_path, "output_tensor_small.npy"))

def validate_q_k_v_small_outout(dst_path = None):
    # Q
    query_tensor_npy = np.load(os.path.join(dst_path, "q_tensor_small.npy"))
    query_tensor_input = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_query.npy"))
    # print(f"==>> query_tensor_npy: {query_tensor_npy}")
    # print(f"==>> query_tensor_input: {query_tensor_input}")
    
    # # K
    key_tensor_npy = np.load(os.path.join(dst_path, "k_tensor_small.npy"))
    key_tensor_input = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_key.npy"))
    # print(f"==>> key_tensor_npy: {key_tensor_npy}")
    # print(f"==>> key_tensor_input: {key_tensor_input}")
    
    # V
    value_tensor_npy = np.load(os.path.join(dst_path, "v_tensor_small.npy"))
    value_tensor_input = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_value.npy"))
    print(f"==>> value_tensor_npy: {value_tensor_npy}")
    print(f"==>> value_tensor_input: {value_tensor_input}")
    print(f"==>> type(value_tensor_input): {value_tensor_input.dtype}")
    
    # outout_tensor = np.load(os.path.join(dst_path, "dml_mha_q_k_v_small_output.npy"))
    # print(f"==>> outout_tensor: {outout_tensor}")


def gen_llama2_input_npy_buffer_flash_decoding(dst_path = None):
    q_shape = [1, 32, 1, 128]
    k_shape = [1, 32, 2048, 128]
    v_shape = [1, 32, 2048, 128]
    output_shape = [1, 32, 1, 128]

    q_tensor = np.random.uniform(0, 1, q_shape).astype("float16")
    k_tensor = np.random.uniform(0, 1, k_shape).astype("float16")
    v_tensor = np.random.uniform(0, 1, v_shape).astype("float16")
    output_tensor = np.zeros(output_shape)
    output_tensor = np.ones(output_shape) * 0

    # q_tensor = np.ones(q_shape).astype("float16")
    # k_tensor = np.ones(k_shape).astype("float16")
    # v_tensor = np.ones(v_shape).astype("float16")

    np.save(os.path.join(dst_path, "q_tensor.npy"), q_tensor)
    np.save(os.path.join(dst_path, "k_tensor.npy"), k_tensor)
    np.save(os.path.join(dst_path, "v_tensor.npy"), v_tensor)
    np.save(os.path.join(dst_path, "output_tensor.npy"), output_tensor)
    
    q_tensor = np.load(os.path.join(dst_path, "q_tensor.npy"))
    k_tensor = np.load(os.path.join(dst_path, "k_tensor.npy"))
    v_tensor = np.load(os.path.join(dst_path, "v_tensor.npy"))
    outout_tensor = np.load(os.path.join(dst_path, "output_tensor.npy"))

def validate_llama2_q_k_v_outout(dst_path = None):
    outout_tensor = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_output.npy"))
    query_tensor_input = np.load(os.path.join(dst_path, "q_tensor.npy"))
    print(f"==>> query_tensor_input: {query_tensor_input}")
    query_tensor = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_query.npy"))
    print(f"==>> query_tensor: {query_tensor}")
    # print(f"==>> outout_tensor.shape: {outout_tensor.shape}")
    # print(f"==>> outout_tensor: {outout_tensor}")


def gen_sd_input_npy_buffer_flash_att2(dst_path = None):
    q_shape = [2,64,1280]
    kv_shape = [2, 77, 8, 2, 160]
    output_shape = [2,64,1280]

    q_tensor = np.random.uniform(0, 1, q_shape).astype("float16")
    kv_tensor = np.random.uniform(0, 1, kv_shape).astype("float16")
    output_tensor = np.zeros_like(output_shape)

    # q_tensor = np.ones(q_shape).astype("float16")
    # kv_tensor = np.ones(kv_shape).astype("float16")

    np.save("q_tensor.npy",q_tensor)
    np.save("kv_tensor.npy",kv_tensor)
    np.save("output_tensor.npy",output_tensor)

    q_tensor = np.load("q_tensor.npy")
    kv_tensor = np.load("kv_tensor.npy")


if __name__ == "__main__":
    # mha_json = "./mha_json/dml_mha_q_kv64.json"
    # mha_json = "./dml_gemm_llama2.json"

    # small shape test 
    # gen_small_input_npy_buffer_flash_decoding(dst_path = "./flash_decoding_json")
    # mha_json = "./flash_decoding_json/dml_mha_q_k_v_small.json"
    # launch_dxdispatch(mha_json)
    # validate_q_k_v_small_outout(dst_path = "./flash_decoding_json")

    # gen_llama2_input_npy_buffer_flash_decoding(dst_path = "./flash_decoding_json")
    mha_json = "./flash_decoding_json/dml_mha_q_k_v2048.json"
    launch_dxdispatch(mha_json)
    # validate_llama2_q_k_v_outout(dst_path = "./flash_decoding_json")

    # gen_sd_input_npy_buffer_flash_att2(dst_path = "./flash_att2_SD_json")
