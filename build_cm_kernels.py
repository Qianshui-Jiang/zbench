import os

file_root_path = os.path.dirname(__file__)
cm_root_path = os.environ['CSDK_IGC']
cm_exe = os.path.join(cm_root_path,'compiler/bin/cmc.exe')
cm_kernel_path =  os.path.join(file_root_path, 'cm_kernels')

cm_kernel_output_path =  os.path.join(file_root_path, 'cm_kernel_output')
if not os.path.exists(cm_kernel_output_path):
    os.mkdir(cm_kernel_output_path)

def build_cm_kernels(kernel_file):
  device = "DG2"

  kernel = kernel_file.split('.')[0]
  src_path = os.path.join(cm_kernel_path, kernel_file)
  dst_path = os.path.join(cm_kernel_output_path, f'{kernel}.bin')
  inc_path = os.path.join(file_root_path, 'rt_l0')
  cmd = f'{cm_exe} -march={device} -I {inc_path} {src_path} -o {dst_path}'
  
  res = os.system(cmd)
  

if __name__ == '__main__':
    build_cm_kernels(kernel_file='bgemm_dpas_genx.cpp')