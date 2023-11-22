#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/numpy.h>
// #include <pybind11_json/pybind11_json.hpp>
// namespace py = pybind11;
// #include <nlohmann/json.hpp>
// namespace nl = nlohmann;

#include "dev_test.h"
#include "sgemm.h"

PYBIND11_MODULE(zbench, m)
{
    m.doc() = "pybind11 testbed for meta commande and CM kernels";
    // dev function/feature test
    m.def("add", &add, "A function which adds two numbers", py::arg("i") = 1, py::arg("j") = 2);
    m.def("test_bind", &test_bind, py::arg("mode"), py::arg("input"),"test pybind11 for array transfering");
    m.def("test_take_json", &test_take_json, "pass py::object to a C++ function that takes an nlohmann::json");
    m.def("test_get_json", &test_get_json, "return py::object from a C++ function that returns an nlohmann::json");

    // sample bench for sgemm cm kernel
    m.def("run_sgemm", &run_sgemm, "A function which adds two numbers", 
          py::arg("m") = 1024, py::arg("niterations") = 1,
          py::arg("gy") = 1, py::arg("gx") = 4,
          py::arg("bin_file") = "sgemm_genx.bin", 
          py::arg("fn_name") = "sgemm_kernel"
          );

    // // sample bench for DPAS bgemm cm kernel
    m.def("run_bgemm", &run_bgemm, "A function which adds two numbers", 
          py::arg("M") = 128, py::arg("N") = 128,py::arg("K") = 128,
          py::arg("threadWidth") = 4, py::arg("threadHeight") = 4,
          py::arg("groupWidth") = 1, py::arg("groupHeight") = 1,
          py::arg("bin_file") = "bgemm_dpas_genx.bin", 
          py::arg("fn_name") = "bgemm_dpas"
          );

    // code for testing the numpy array binding
    m.def("run_kernel", &run_kernel, "A function which adds two numbers", 
          py::arg("bin_file") = "bgemm_dpas_genx.bin", 
          py::arg("spirv_file") = "bgemm_dpas_genx.spv", 
          py::arg("fn_name") = "bgemm_dpas"
          );

}