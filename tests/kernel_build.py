import os
import shutil
cm_root_path = r"C:\Users\12900K\Documents\Engneeing_works\dml_workspace\ComputeSDK_Windows_internal_2023_WW41"
file_root_path = r"C:\Users\12900K\Documents\Engneeing_works\dml_workspace\zbench"
# file_root_path = os.path.dirname(__file__)
# cm_root_path = os.environ['CSDK_IGC']
cm_exe = os.path.join(cm_root_path,'compiler/bin/cmc.exe')
cm_kernel_path =  os.path.join(file_root_path, 'cm_kernels')

cm_kernel_output_path =  os.path.join(file_root_path, 'cm_kernel_output')
if not os.path.exists(cm_kernel_output_path):
    os.mkdir(cm_kernel_output_path)

def build_cm_kernels(kernel_file, define_flag=None):
  device = "DG2"

  kernel = os.path.basename(kernel_file).split('.')[0]
  src_path = os.path.join(cm_kernel_path, kernel_file)
  dst_path = os.path.join(cm_kernel_output_path, f'{kernel}.bin')
  if os.path.exists(dst_path):
    os.remove(dst_path)
  inc_path = os.path.join(file_root_path, 'rt_l0')
  if define_flag is not None:
    cmd = f'{cm_exe} -march={device} {define_flag} -I {inc_path} {src_path} -o {dst_path}'
  else:
    cmd = f'{cm_exe} -march={device} -I {inc_path} {src_path} -o {dst_path}'
  
  res = os.system(cmd)
  

if __name__ == '__main__':
  
    flag = "-DSIZE_M=16 \
            -DSIZE_K=16 \
            -DSIZE_N=16 \
            -DTILE_K=16 \
            -DTILE_N=16 \
            -DTILE_M=16 \
            -DLWS_SIZE_X=1 \
            -DLWS_SIZE_Y=1 \
            -DLWS_SIZE_Z=1 \
            -DGWS_SIZE_X=16 \
            -DGWS_SIZE_Y=1 \
            -DGWS_SIZE_Z=1"
    build_cm_kernels(kernel_file='gemm_nchw_fp16.cpp', define_flag= flag)
    # build_cm_kernels(kernel_file='bgemm_dpas_genx.cpp')