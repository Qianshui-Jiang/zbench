zbench:  A simple CM kernel launchpad
==============

## Introduction

- Using dxd12 runtime and intel graphics dev extension to compile and running a CM shader, 
- Using Pybind11 to construct the interface, which means all the IO tensors, build options, shader params are setup from python file.
  - All the IO buffers are in Numpy array format. and direct setup as the kernel input arguments.
  - All the Hyper parameters are setup through C/C++ Macros, which means contained in build options in format like `-Dxxxx=123`
  - the param name and value are flexible, except that buffers should follow the order of your CM kernel input.
  - 3 necessary args should be given in a fixed name: `cm_file`, `build_options`, and `iter_nums`
- Assign the iteration numbers by `iter_nums`, and usually put it in the last arg of  `launch_rt_igdext()`
- For more information, please find out the examples in `zbench/tests`:

## Build from source

1. you need have oneAPI base toolkit installed in your windows machine. after installed, you can find `setvars.bat` in your oneAPI target path, and run following cmd in your terminal shell:

   ```shell
   cmd "/K" '"{Your OneAPI Path}\setvars.bat" && powershell'
   ```

2. create a python environment(here using conda), and activate it: 

   ```shell
   conda create -n kernel_dev python==3.10.13
   conda activate kernel_dev
   pip install -r requirements
   ```

3. clone the repo:

   ```
   git clone
   git submodule init
   git submodule update 
   ```

4. entering zbench folder and build or install it:

   ```shell
   cd zbench
   // direct build & install 
   python setup.py install
   // only build
   python setup.py build
   ```

5. if you want build python wheels:

   ```shell
   python setup.py bdist_wheel
   ```

   the wheel files would be exists in folder `zbench/dist`

## Install already built python wheel directly


1. create a python environment(here using conda), and activate it: 

   ```shell
   conda create -n kernel_dev python==3.10.13
   conda activate kernel_dev
   pip install -r requirements
   ```

2. install the zbench wheel:

   ```shell
   pip install zbench-0.0.0-cp310-cp310-win_amd64.whl
   ```

## Try the example:

- Step 1. exporting reference tensor by dxdispatch

```shell
python dxdispatch_dump_data_llama2.py
```

- Step 2. test MHA with exported ref data

```
python test_flash_att_llm.py
```

## TODOs:

- [ ] Add MHA tests for SD's shape
- [ ] Add online softmax tests.
- [ ] Add GEMV tests.
- [ ] Add GEMM tests.
- [ ] More runtime support. （L0 / OCL）
- [ ] Hardware performance monitor counter support.