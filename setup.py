import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

__version__ = "0.0.1"
os.environ['DISTUTILS_USE_SDK']='1'  # something about oneAPI openCL runtime
file_root_path = os.path.dirname(__file__)

# ----include Path-----
PYBIND11_JSON_PATH = os.path.join(file_root_path, '3rdparty/pybind11_json/include')
NLOHMANN_JSON =  os.path.join(file_root_path, r'3rdparty/json/include')
RT_IGDEXT_PATH = os.path.join(file_root_path,'rt_igdext')

include_path_list = list()
include_path_list.append(PYBIND11_JSON_PATH)
include_path_list.append(NLOHMANN_JSON)
include_path_list.append(RT_IGDEXT_PATH)

include_path_list.append(os.path.join(file_root_path, '3rdparty/d3dx12'))
include_path_list.append(os.path.join(file_root_path, '3rdparty/dml/include'))
include_path_list.append(os.path.join(file_root_path, '3rdparty/dmlx'))
include_path_list.append(os.path.join(file_root_path, '3rdparty/igdext/include'))

include_path_list.append('.')

# ----library Path & List-----
library_path_list = list()
library_list = list()

# dml dmlx igdext libdml
# d3d12 dxgi dxguid d3d12x
library_list.append('d3d12')
library_list.append('dxgi')
library_list.append('dxguid')
library_list.append('Ole32')

library_path_list.append(os.path.join(file_root_path, '3rdparty/dml/bin/x64-win'))
library_list.append('DirectML') 

library_path_list.append(os.path.join(file_root_path, '3rdparty/igdext/lib'))
library_list.append('igdext64')
library_list.append('cfgmgr32')
library_list.append('SetupAPI')
library_list.append('ShLwApi')

# ----source code  List-----
src_list = list()
src_list = ['zbench.cpp', 'dev_test.cpp']
src_list.extend(['rt_igdext/rt_igdext.cpp'])

module= CppExtension(name='zbench', 
                    sources=src_list,
                    include_dirs=include_path_list,
                    library_dirs=library_path_list,
                    libraries=library_list,
                    cxx_std=20,
                    define_macros=[('VERSION_INFO', __version__)])

setup(name='zbench',
    ext_modules=[module],
    cmdclass={'build_ext': BuildExtension})
