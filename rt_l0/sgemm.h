#pragma once

#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py=pybind11;

#include <nlohmann/json.hpp>
namespace nl=nlohmann;

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef uint16_t MType;

int run_sgemm(int m, int niterations, int gx, int gy, 
            const char* bin_file, const char* fn_name);

int run_bgemm(int M, int K, int N, int threadWidth, int threadHeight,
            int groupWidth, int groupHeight,
            const char* bin_file, const char* fn_name);


std::vector<MType> run_kernel(const char* bin_file , const char* spirv_file, const char* fn_name,
              const py::args& args, const py::kwargs& kwargs);