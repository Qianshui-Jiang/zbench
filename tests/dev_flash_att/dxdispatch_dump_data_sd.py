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


# dxdispatch = r"D:\dml_workspace\zbench\tests\dev_flash_att\DxDispatch\dxdispatch.exe"
dxdispatch = r"D:\dml_workspace\zbench\tests\dev_flash_att\DxDispatchRelWithDebInfo\dxdispatch.exe"


def launch_dxdispatch(input_file, iter_nums = 10):
    cmd_launch = f"{dxdispatch} -w 10 -r 10 -i {iter_nums}  {input_file}"
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


def validate_llama2_q_k_v_outout(dst_path = None):
    outout_tensor = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_output.npy"))
    query_tensor_input = np.load(os.path.join(dst_path, "q_tensor.npy"))
    print(f"==>> query_tensor_input: {query_tensor_input}")
    query_tensor = np.load(os.path.join(dst_path, "dml_mha_q_k_v2048_query.npy"))
    print(f"==>> query_tensor: {query_tensor}")
    # print(f"==>> outout_tensor.shape: {outout_tensor.shape}")
    # print(f"==>> outout_tensor: {outout_tensor}")


def gen_q_kv_input(input_json = None):
    
    dst_path = os.path.dirname(input_json)
    dst_path = os.path.join(dst_path, "tensor_file")
    print(f"==>> dst_path: {dst_path}")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    with open(input_json, 'r') as f:
        data = json.load(f)

    q_shape =  data['dispatchables']['mha']['desc']['QueryTensor']['Sizes']
    q_name  = data['resources']['query']['initialValues']['sourcePath']
    kv_shape = data['dispatchables']['mha']['desc']['StackedKeyValueTensor']['Sizes']
    kv_name  = data['resources']['stackedKeyValue']['initialValues']['sourcePath']
    print(f"==>> q_name: {q_name}")
    print(f"==>> q_shape: {q_shape}")
    print(f"==>> kv_name: {kv_name}")
    print(f"==>> kv_shape: {kv_shape}")

    q_tensor = np.random.uniform(-1, 1, q_shape).astype("float16")
    kv_tensor = np.random.uniform(-1, 1, kv_shape).astype("float16")

    # q_tensor = np.ones(q_shape).astype("float16")
    # kv_tensor = np.ones(kv_shape).astype("float16")

    np.save(q_name, q_tensor)
    np.save(kv_name, kv_tensor)


    q_tensor = np.load(q_name)
    kv_tensor = np.load(kv_name)


def gen_qkv_input(input_json = None):
    
    dst_path = os.path.dirname(input_json)
    dst_path = os.path.join(dst_path, "tensor_file")
    print(f"==>> dst_path: {dst_path}")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    with open(input_json, 'r') as f:
        data = json.load(f)

    qkv_shape =  data['dispatchables']['mha']['desc']['StackedQueryKeyValueTensor']['Sizes']
    qkv_name  = data['resources']['stackedQueryKeyValue']['initialValues']['sourcePath']
    print(f"==>> kv_name: {qkv_shape}")
    print(f"==>> kv_shape: {qkv_name}")

    qkv_tensor = np.random.uniform(-1, 1, qkv_shape).astype("float16")

    # qkv_tensor = np.ones(q_shape).astype("float16")

    np.save(qkv_name, qkv_tensor)
    qkv_tensor = np.load(qkv_name)


if __name__ == "__main__":

    # small shape test 


    # validate_llama2_q_k_v_outout(dst_path = "./flash_decoding_json")

    json_lists = [
        "flash_att2_SD_json\dml_mha_q_kv_small.json",
        "flash_att2_SD_json\dml_mha_q_kv64.json",
        "flash_att2_SD_json\dml_mha_q_kv256.json",
        "flash_att2_SD_json\dml_mha_q_kv1024.json",
        "flash_att2_SD_json\dml_mha_q_kv4096.json",
        "flash_att2_SD_json\dml_mha_qkv_small.json",
        "flash_att2_SD_json\dml_mha_qkv64.json",
        "flash_att2_SD_json\dml_mha_qkv256.json",
        "flash_att2_SD_json\dml_mha_qkv1024.json",
        "flash_att2_SD_json\dml_mha_qkv4096.json",
    ]

    # gen_q_kv_input(input_json = "flash_att2_SD_json\dml_mha_q_kv_small.json",)
    # launch_dxdispatch(input_file="flash_att2_SD_json\dml_mha_q_kv_small.json", iter_nums=10000)

    # gen_q_kv_input(input_json = "flash_att2_SD_json\dml_mha_q_kv_tile.json",)
    # launch_dxdispatch(input_file="flash_att2_SD_json\dml_mha_q_kv_til.json", iter_nums=100)
    
    # gen_q_kv_input(input_json = "flash_att2_SD_json\dml_mha_q_kv64.json",)
    # launch_dxdispatch(input_file="flash_att2_SD_json\dml_mha_q_kv64.json", iter_nums=100)    

    # gen_q_kv_input(input_json = "flash_att2_SD_json\dml_mha_q_kv256.json")
    # launch_dxdispatch(input_file="flash_att2_SD_json\dml_mha_q_kv256.json", iter_nums=100)

    gen_q_kv_input(input_json = "flash_att2_SD_json\dml_mha_q_kv1024.json")
    launch_dxdispatch(input_file="flash_att2_SD_json\dml_mha_q_kv1024.json", iter_nums=100)

    # gen_q_kv_input(input_json = "flash_att2_SD_json\dml_mha_q_kv4096.json")
    # launch_dxdispatch(input_file="flash_att2_SD_json\dml_mha_q_kv4096.json", iter_nums=100)
    exit()

    gen_qkv_input(input_json = json_lists[5])
    launch_dxdispatch(input_file=json_lists[5], iter_nums=1000)
    
    # gen_qkv_input(input_json = json_lists[5])
    # launch_dxdispatch(input_file=json_lists[5], iter_nums=1000)
    
    # gen_qkv_input(input_json = json_lists[6])
    # launch_dxdispatch(input_file=json_lists[6], iter_nums=1000)
    
    # gen_qkv_input(input_json = json_lists[7])
    # launch_dxdispatch(input_file=json_lists[7], iter_nums=1000)
    # gen_qkv_input()

