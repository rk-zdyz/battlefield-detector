#ifndef PYBIND_WRAPPER_HPP
#define PYBIND_WRAPPER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "battlefield_core_standalone.hpp"

namespace py = pybind11;

/**
 * @brief Integration Developer - Zero-Copy Pointer Bridge
 * Connects C++ backend hardware pipeline with Python frontend.
 * Passes spatial anomaly heatmap tensors strictly via zero-copy memory buffer pointers.
 */
class ZeroCopyPybindBridge {
private:
    FastAnomalyCalculus calculus_;

public:
    explicit ZeroCopyPybindBridge(float threshold = 0.05f) : calculus_(threshold) {}

    py::array_t<float> compute_mse_zerocopy(py::array_t<float> raw, py::array_t<float> recon) {
        return calculus_.compute_pixelwise_mse(raw, recon);
    }
};

#endif // PYBIND_WRAPPER_HPP
