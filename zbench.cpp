#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11_json/pybind11_json.hpp>
namespace py = pybind11;
#include <nlohmann/json.hpp>
namespace nl = nlohmann;



int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(zbench, m)
{

    m.doc() = "pybind11 example plugin";

    m.def("add", &add, "A function which adds two numbers", py::arg("i") = 1, py::arg("j") = 2);
}