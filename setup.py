import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

__version__ = "0.0.1"

os.environ['DISTUTILS_USE_SDK']='1'
cm_root_path = os.environ['CSDK_IGC']
file_root_path = os.path.dirname(__file__)


LIB_OPENCL = "Intel_OpenCL_ICD64"
LIB_OPENCL_PATH = os.path.join(cm_root_path, 'runtime/opencl/lib')

LIB_LEVEL0 = "ze_loader"
LIB_LEVEL0_PATH = os.path.join(cm_root_path, 'runtime/level_zero/lib')

PYBIND11_JSON_PATH = os.path.join(file_root_path, '3rdparty/pybind11_json/include')
NLOHMANN_JSON =  os.path.join(file_root_path, r'3rdparty/json/include')
RT_L0_PATH = os.path.join(file_root_path,'rt_l0')

# ----include LIB and Path-----
include_path_list = list()
library_path_list = list()
library_list = list()
src_list = list()

include_path_list.append('.')
include_path_list.append(RT_L0_PATH)
include_path_list.append(PYBIND11_JSON_PATH)
include_path_list.append(NLOHMANN_JSON)

include_path_list.append(os.path.join(cm_root_path, 'compiler/include'))
include_path_list.append(os.path.join(cm_root_path, 'compiler/bin'))
include_path_list.append(os.path.join(cm_root_path, 'runtime/level_zero/include '))
include_path_list.append(os.path.join(cm_root_path, 'usr/include '))


# library_path_list.append(LIB_LEVEL0_PATH)
# library_list.append('Intel_OpenCL_ICD64') # openCL runtime
library_path_list.append(LIB_LEVEL0_PATH)
library_list.append('ze_loader') # Level 0 runtime

src_list = ['zbench.cpp', 'rt_l0/l0_launch.cpp', 'rt_l0/l0_launch_model_shader.cpp', 'dev_test.cpp']


# exit()
module= CppExtension(name='zbench', 
                    sources= src_list,
                    include_dirs=include_path_list,
                    library_dirs=library_path_list,
                    libraries= library_list,
                    cxx_std=17,
                    define_macros=[('VERSION_INFO', __version__)])

setup(name='zbench',
    ext_modules=[module],
    cmdclass={'build_ext': BuildExtension})
