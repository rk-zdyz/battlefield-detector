#ifndef CAMERA_STREAM_HPP
#define CAMERA_STREAM_HPP

#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>
#include "frame_queue.hpp"

using FrameQueue = IsolatedMemoryQueue<cv::Mat>;

/**
 * @brief Core Systems Lead - CameraStream Hardware Abstraction Layer
 * Handles multithreaded OpenCV camera video ingestion, thread synchronization
 * via std::mutex, and bounded frame queues to guarantee zero-bottleneck ingestion.
 */
class CameraStream {
private:
    cv::VideoCapture cap_;
    FrameQueue frame_queue_;
    std::thread capture_thread_;
    std::atomic<bool> is_running_;
    double fps_;

    void captureLoop();

public:
    explicit CameraStream(size_t queue_capacity = 30);
    ~CameraStream();

    bool open(const std::string& source);
    void start();
    void stop();
    bool getFrame(cv::Mat& frame_out, int timeout_ms = 30);
    
    double getFPS() const { return fps_; }
    size_t getQueueSize() const { return frame_queue_.size(); }
};

#endif // CAMERA_STREAM_HPP
