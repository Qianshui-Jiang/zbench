#pragma once

#include <vector>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py=pybind11;

#include <nlohmann/json.hpp>
namespace nl=nlohmann;

typedef uint16_t MType;

std::vector<MType> test_rt_igdext(const std::string &cm_file , const std::string &build_options,
                                    const py::args& args, const py::kwargs& kwargs);