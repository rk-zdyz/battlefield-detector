#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "battlefield_core_standalone.hpp"

namespace py = pybind11;

PYBIND11_MODULE(battlefield_core, m) {
    m.doc() = "C++ Zero-Copy NumPy Buffer Anomaly Calculus Engine";

    py::class_<FastAnomalyCalculus>(m, "FastAnomalyCalculus")
        .def(py::init<float>(), py::arg("mse_threshold") = 0.05f)
        .def("compute_pixelwise_mse", &FastAnomalyCalculus::compute_pixelwise_mse)
        .def("set_threshold", &FastAnomalyCalculus::set_threshold)
        .def("get_threshold", &FastAnomalyCalculus::get_threshold);
}
