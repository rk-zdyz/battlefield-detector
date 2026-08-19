#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pybind_wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(battlefield_core, m) {
    m.doc() = "Integration Developer - pybind11 Zero-Copy Memory Pointer Bridge";

    py::class_<ZeroCopyPybindBridge>(m, "ZeroCopyPybindBridge")
        .def(py::init<float>(), py::arg("threshold") = 0.05f)
        .def("compute_mse_zerocopy", &ZeroCopyPybindBridge::compute_mse_zerocopy);

    py::class_<FastAnomalyCalculus>(m, "FastAnomalyCalculus")
        .def(py::init<float>(), py::arg("mse_threshold") = 0.05f)
        .def("compute_pixelwise_mse", &FastAnomalyCalculus::compute_pixelwise_mse)
        .def("set_threshold", &FastAnomalyCalculus::set_threshold)
        .def("get_threshold", &FastAnomalyCalculus::get_threshold);
}
