#ifndef BATTLEFIELD_CORE_HPP
#define BATTLEFIELD_CORE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace py = pybind11;

/**
 * High-Performance C++ Zero-Copy Anomaly Calculus Engine.
 * Operates directly on py::array_t<float> NumPy buffer views.
 */
class FastAnomalyCalculus {
private:
    float mse_threshold_;

public:
    explicit FastAnomalyCalculus(float mse_threshold = 0.05f) : mse_threshold_(mse_threshold) {}

    py::array_t<float> compute_pixelwise_mse(py::array_t<float> raw_buf, py::array_t<float> recon_buf) {
        py::buffer_info raw_info = raw_buf.request();
        py::buffer_info recon_info = recon_buf.request();

        if (raw_info.size != recon_info.size) {
            throw std::invalid_argument("Input buffer dimensions do not match.");
        }
        if (!raw_info.ptr || !recon_info.ptr) {
            throw std::runtime_error("Null buffer pointer passed to compute_pixelwise_mse.");
        }

        size_t size = raw_info.size;
        auto result = py::array_t<float>(raw_info.shape);
        py::buffer_info res_info = result.request();

        float* raw_ptr = static_cast<float*>(raw_info.ptr);
        float* recon_ptr = static_cast<float*>(recon_info.ptr);
        float* res_ptr = static_cast<float*>(res_info.ptr);

        for (size_t i = 0; i < size; ++i) {
            float diff = raw_ptr[i] - recon_ptr[i];
            res_ptr[i] = diff * diff;
        }

        return result;
    }

    void set_threshold(float th) { mse_threshold_ = th; }
    float get_threshold() const { return mse_threshold_; }
};

#endif // BATTLEFIELD_CORE_HPP
