#include "dev_test.h"

#include <iostream>
#include <vector>


int add(int i, int j) { return i + j; }

void test_bind(const std::string &mode, const std::string &input,
               const py::args &args, const py::kwargs &kwargs) {
  std::cout << "mode: " << mode << std::endl;
  std::cout << "input: " << input << std::endl;
  if (kwargs) {
    for (auto item : kwargs) {
      if (py::type::of(item.second) == py::type::of(py::int_())) {
        std::cout << "get input type: " << py::type::of(item.second).str()
                  << std::endl; // <class 'int'>
        int arg_in = py::int_(kwargs[item.first]);
      }
      if (py::type::of(item.second) == py::type::of(py::array())) {
        std::cout << "get input type: " << py::type::of(item.second).str()
                  << std::endl; // <class 'numpy.ndarray'>
        auto buf_in =
            py::array_t<float, py::array::c_style | py::array::forcecast>(
                kwargs[item.first]);
        float *m_A = const_cast<float *>(buf_in.data());
        // std::memcpy(m_A, buf_in.data(), buf_in.size() * sizeof(float));
        std::cout << "input buf_in size: " << buf_in.size()
                  << std::endl; // <class 'numpy.ndarray'>
        for (int i = 0; i < 5; i++) {
          std::cout << m_A[i] << std::endl;
        }
      }
    }
  }
}

nl::json test_take_json(const nl::json &j) {
  std::cout << "This function took an nlohmann::json instance as argument: "
            << j << std::endl;
  return j;
}

nl::json test_get_json() {
  nl::json j = {{"value", 1}};
  std::cout << "This function returns an nlohmann::json instance: " << j
            << std::endl;
  return j;
}