CMEMU examples
==============

In order to run cmemu examples make sure, you have set up the environment via developer command
prompt for VS and setenv.bat

*mkdir build*

*cd build*

Run cmake
---------

*cmake ..*

Build tests using cmake
-----------------------

*cmake --build .*
or 
*cmake --build . --target install*

or
---

*msbuild cmemu\_examples.sln*

Run tests
---------

*ctest -C Debug*


1. Bench kernel code for Xe core / both L0 & OCL runtime
2. Add python front end use pybind11, to splite the kernel and data analysis
3. Bench DML kernels for Driver metacommand