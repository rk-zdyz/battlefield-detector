#include "camera_stream.hpp"
#include <iostream>

CameraStream::CameraStream(size_t queue_capacity)
    : frame_queue_(queue_capacity), is_running_(false), fps_(30.0) {}

CameraStream::~CameraStream() {
    stop();
}

bool CameraStream::open(const std::string& source) {
    if (source == "0" || source == "webcam") {
        return cap_.open(0);
    }
    return cap_.open(source);
}

void CameraStream::start() {
    if (!cap_.isOpened()) {
        std::cerr << "[!] CameraStream Error: Source not opened." << std::endl;
        return;
    }
    is_running_ = true;
    capture_thread_ = std::thread(&CameraStream::captureLoop, this);
}

void CameraStream::stop() {
    is_running_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (cap_.isOpened()) {
        cap_.release();
    }
}

void CameraStream::captureLoop() {
    cv::Mat frame;
    while (is_running_) {
        if (cap_.read(frame) && !frame.empty()) {
            frame_queue_.push(frame);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool CameraStream::getFrame(cv::Mat& frame_out, int timeout_ms) {
    return frame_queue_.pop(frame_out, timeout_ms);
}
