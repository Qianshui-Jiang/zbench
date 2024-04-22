#pragma once

#include <iostream>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>
namespace py = pybind11;

#include <nlohmann/json.hpp>
namespace nl = nlohmann;

int add(int i, int j);
void test_bind(const std::string &mode, const std::string &input,
               const py::args &args, const py::kwargs &kwargs);
nl::json test_set_json(const nl::json &json);
nl::json test_get_json();

