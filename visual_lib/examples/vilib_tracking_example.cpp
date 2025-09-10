/*
 * ViLib Feature Tracker Example
 * 
 * This example demonstrates how to use vilib's GPU-accelerated feature tracker
 * to process a sequence of images and measure tracking performance.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

// ViLib includes
#include "vilib/feature_tracker/feature_tracker_gpu.h"
#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/feature_detection/harris/harris_gpu.h"
#include "vilib/storage/pyramid_pool.h"
#include "vilib/cuda_common.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"

using namespace vilib;

// Configuration parameters
int FRAME_IMAGE_PYRAMID_LEVELS = 5;
int FEATURE_DETECTOR_CELL_SIZE_WIDTH = 32;
int FEATURE_DETECTOR_CELL_SIZE_HEIGHT = 32;
int FEATURE_DETECTOR_MIN_LEVEL = 0;
int FEATURE_DETECTOR_MAX_LEVEL = 2;
int FEATURE_DETECTOR_HORIZONTAL_BORDER = 8;
int FEATURE_DETECTOR_VERTICAL_BORDER = 8;

// Harris detector parameters
float FEATURE_DETECTOR_HARRIS_K = 0.04f;
float FEATURE_DETECTOR_HARRIS_QUALITY_LEVEL = 0.1f;
conv_filter_border_type FEATURE_DETECTOR_HARRIS_BORDER_TYPE = conv_filter_border_type::BORDER_SKIP;

class ViLibTrackingExample {
public:
    ViLibTrackingExample() : initialized_(false) {}
    
    bool initialize(int image_width, int image_height) {
        if (initialized_) return true;
        
        // Initialize CUDA
        if (!cuda_initialize()) {
            std::cerr << "Failed to initialize CUDA" << std::endl;
            return false;
        }
        
        // Setup feature tracker options
        FeatureTrackerOptions tracker_options;
        tracker_options.reset_before_detection = false;
        tracker_options.use_best_n_features = 500;
        tracker_options.min_tracks_to_detect_new_features = static_cast<size_t>(0.3 * tracker_options.use_best_n_features);
        tracker_options.klt_max_level = 4;
        tracker_options.klt_min_level = 0;
        tracker_options.klt_patch_sizes = {16, 16, 16, 8, 8};
        tracker_options.klt_min_update_squared = 0.0005;
        
        // Create feature tracker (1 camera)
        tracker_ = std::make_shared<FeatureTrackerGPU>(tracker_options, 1);
        
        // Create Harris detector
        auto harris_detector = std::make_shared<HarrisGPU>(
            image_width, image_height,
            32, 32,
            0, 3,
            8, 8,
            conv_filter_border_type::BORDER_SKIP,
             false, 0, 0.01
        );
        
        // Cast to base class for the tracker
        detector_ = std::static_pointer_cast<DetectorBaseGPU>(harris_detector);
        
        // Set detector for the tracker
        tracker_->setDetectorGPU(detector_, 0);
        
        // Initialize pyramid pool
        PyramidPool::init(
            1, // 1 pyramid per pool
            image_width, image_height,
            1, // grayscale (1 byte per pixel)
            FRAME_IMAGE_PYRAMID_LEVELS,
            Subframe::MemoryType::PITCHED_DEVICE_MEMORY
        );
        
        frame_id_ = 0;
        initialized_ = true;
        return true;
    }
    
    void processImage(const cv::Mat& image, bool first_image = false) {
        if (!initialized_) {
            std::cerr << "Tracker not initialized!" << std::endl;
            return;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Convert to grayscale if needed
        cv::Mat gray_image;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        } else {
            gray_image = image;
        }
        
        // Create Frame directly from image (it will handle pyramid creation)
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(
            gray_image,
            0, // camera id
            FRAME_IMAGE_PYRAMID_LEVELS
        );
        
        // Create frame bundle (single camera)
        std::vector<std::shared_ptr<Frame>> framelist;
        framelist.push_back(frame);
        std::shared_ptr<FrameBundle> frame_bundle = std::make_shared<FrameBundle>(framelist);
        
        // Track features
        std::size_t total_tracked_features = 0;
        std::size_t total_detected_features = 0;
        
        auto track_start = std::chrono::high_resolution_clock::now();
        tracker_->track(frame_bundle, total_tracked_features, total_detected_features);
        auto track_end = std::chrono::high_resolution_clock::now();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Calculate timing
        double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double track_time = std::chrono::duration<double, std::milli>(track_end - track_start).count();
        
        // Store statistics
        // don't store stats for the first image if specified
        if (!first_image) {
        frame_times_.push_back(total_time);
        track_times_.push_back(track_time);
        tracked_features_.push_back(total_tracked_features);
        detected_features_.push_back(total_detected_features);
        }
        
        // Print current frame info
        std::cout << "Frame " << frame_id_ << ": "
                  << "Total=" << total_time << "ms, "
                  << "Track=" << track_time << "ms, "
                  << "Tracked=" << total_tracked_features << ", "
                  << "Detected=" << total_detected_features << std::endl;
        
        frame_id_++;
    }
    
    void printStatistics() {
        if (frame_times_.empty()) {
            std::cout << "No frames processed." << std::endl;
            return;
        }
        
        // Calculate averages
        double avg_total_time = 0.0;
        double avg_track_time = 0.0;
        double avg_tracked_features = 0.0;
        double avg_detected_features = 0.0;
        
        for (size_t i = 0; i < frame_times_.size(); ++i) {
            avg_total_time += frame_times_[i];
            avg_track_time += track_times_[i];
            avg_tracked_features += tracked_features_[i];
            avg_detected_features += detected_features_[i];
        }
        
        size_t count = frame_times_.size();
        avg_total_time /= count;
        avg_track_time /= count;
        avg_tracked_features /= count;
        avg_detected_features /= count;
        
        // Calculate min/max
        auto min_max_total = std::minmax_element(frame_times_.begin(), frame_times_.end());
        auto min_max_track = std::minmax_element(track_times_.begin(), track_times_.end());
        
        std::cout << "\n=== PERFORMANCE STATISTICS ===" << std::endl;
        std::cout << "Frames processed: " << count << std::endl;
        std::cout << "\nTiming (ms):" << std::endl;
        std::cout << "  Total processing time - Min: " << *min_max_total.first 
                  << ", Max: " << *min_max_total.second 
                  << ", Avg: " << avg_total_time << std::endl;
        std::cout << "  Track/Detect time    - Min: " << *min_max_track.first 
                  << ", Max: " << *min_max_track.second 
                  << ", Avg: " << avg_track_time << std::endl;
        
        std::cout << "\nFeatures:" << std::endl;
        std::cout << "  Average tracked features: " << avg_tracked_features << std::endl;
        std::cout << "  Average detected features: " << avg_detected_features << std::endl;
        
        std::cout << "\nThroughput:" << std::endl;
        std::cout << "  Average FPS (total): " << 1000.0 / avg_total_time << std::endl;
        std::cout << "  Average FPS (track only): " << 1000.0 / avg_track_time << std::endl;
    }
    
private:
    bool initialized_;
    std::shared_ptr<FeatureTrackerGPU> tracker_;
    std::shared_ptr<DetectorBaseGPU> detector_;

    int frame_id_;
    
    // Statistics
    std::vector<double> frame_times_;
    std::vector<double> track_times_;
    std::vector<size_t> tracked_features_;
    std::vector<size_t> detected_features_;
};

class OpenCVTrackingExample {
public:
    OpenCVTrackingExample() : initialized_(false), max_features_(500), quality_level_(0.01), min_distance_(10.0), block_size_(3), use_harris_detector_(false), k_(0.04) {}
    
    bool initialize(int image_width, int image_height) {
        if (initialized_) return true;
        
        image_width_ = image_width;
        image_height_ = image_height;
        frame_id_ = 0;
        initialized_ = true;
        return true;
    }
    
    void processImage(const cv::Mat& image) {
        if (!initialized_) {
            std::cerr << "OpenCV tracker not initialized!" << std::endl;
            return;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Convert to grayscale if needed
        cv::Mat gray_image;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        } else {
            gray_image = image;
        }
        
        size_t total_tracked_features = 0;
        size_t total_detected_features = 0;
        
        auto track_start = std::chrono::high_resolution_clock::now();
        
        if (frame_id_ == 0) {
            // First frame: detect features
            detectNewFeatures(gray_image);
            total_detected_features = current_points_.size();
            prev_gray_ = gray_image.clone();
        } else {
            // Track features from previous frame
            trackFeatures(gray_image);
            total_tracked_features = current_points_.size();
            
            // Detect new features if we have too few
            if (current_points_.size() < static_cast<size_t>(max_features_ * 0.3)) {
                detectNewFeatures(gray_image);
                total_detected_features = current_points_.size() - total_tracked_features;
            }
            
            prev_gray_ = gray_image.clone();
            prev_points_ = current_points_;
        }
        
        auto track_end = std::chrono::high_resolution_clock::now();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Calculate timing
        double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double track_time = std::chrono::duration<double, std::milli>(track_end - track_start).count();
        
        // Store statistics
        frame_times_.push_back(total_time);
        track_times_.push_back(track_time);
        tracked_features_.push_back(total_tracked_features);
        detected_features_.push_back(total_detected_features);
        
        // Print current frame info
        std::cout << "Frame " << frame_id_ << " (OpenCV): "
                  << "Total=" << total_time << "ms, "
                  << "Track=" << track_time << "ms, "
                  << "Tracked=" << total_tracked_features << ", "
                  << "Detected=" << total_detected_features << std::endl;
        
        frame_id_++;
    }
    
    void printStatistics() {
        if (frame_times_.empty()) {
            std::cout << "No frames processed." << std::endl;
            return;
        }
        
        // Calculate averages
        double avg_total_time = 0.0;
        double avg_track_time = 0.0;
        double avg_tracked_features = 0.0;
        double avg_detected_features = 0.0;
        
        for (size_t i = 0; i < frame_times_.size(); ++i) {
            avg_total_time += frame_times_[i];
            avg_track_time += track_times_[i];
            avg_tracked_features += tracked_features_[i];
            avg_detected_features += detected_features_[i];
        }
        
        size_t count = frame_times_.size();
        avg_total_time /= count;
        avg_track_time /= count;
        avg_tracked_features /= count;
        avg_detected_features /= count;
        
        // Calculate min/max
        auto min_max_total = std::minmax_element(frame_times_.begin(), frame_times_.end());
        auto min_max_track = std::minmax_element(track_times_.begin(), track_times_.end());
        
        std::cout << "\n=== OPENCV PERFORMANCE STATISTICS ===" << std::endl;
        std::cout << "Frames processed: " << count << std::endl;
        std::cout << "\nTiming (ms):" << std::endl;
        std::cout << "  Total processing time - Min: " << *min_max_total.first 
                  << ", Max: " << *min_max_total.second 
                  << ", Avg: " << avg_total_time << std::endl;
        std::cout << "  Track/Detect time    - Min: " << *min_max_track.first 
                  << ", Max: " << *min_max_track.second 
                  << ", Avg: " << avg_track_time << std::endl;
        
        std::cout << "\nFeatures:" << std::endl;
        std::cout << "  Average tracked features: " << avg_tracked_features << std::endl;
        std::cout << "  Average detected features: " << avg_detected_features << std::endl;
        
        std::cout << "\nThroughput:" << std::endl;
        std::cout << "  Average FPS (total): " << 1000.0 / avg_total_time << std::endl;
        std::cout << "  Average FPS (track only): " << 1000.0 / avg_track_time << std::endl;
    }
    
private:
    void detectNewFeatures(const cv::Mat& gray_image) {
        // Create mask to avoid detecting features too close to existing ones
        cv::Mat mask = cv::Mat::ones(gray_image.size(), CV_8UC1) * 255;
        
        if (!current_points_.empty()) {
            for (const auto& pt : current_points_) {
                cv::circle(mask, pt, static_cast<int>(min_distance_), 0, -1);
            }
        }
        
        // Detect good features to track
        std::vector<cv::Point2f> new_points;
        cv::goodFeaturesToTrack(gray_image, new_points, 
                               max_features_ - current_points_.size(),
                               quality_level_, min_distance_, mask, 
                               block_size_, use_harris_detector_, k_);
        
        // Add new points to current points
        current_points_.insert(current_points_.end(), new_points.begin(), new_points.end());
    }
    
    void trackFeatures(const cv::Mat& gray_image) {
        if (prev_points_.empty()) return;
        
        std::vector<uchar> status;
        std::vector<float> error;
        
        // Calculate optical flow using Lucas-Kanade method
        cv::calcOpticalFlowPyrLK(prev_gray_, gray_image, 
                                prev_points_, current_points_,
                                status, error,
                                cv::Size(21, 21), 3,
                                cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));
        
        // Filter out bad tracking results
        std::vector<cv::Point2f> good_points;
        for (size_t i = 0; i < current_points_.size(); ++i) {
            if (status[i] && 
                current_points_[i].x >= 0 && current_points_[i].y >= 0 &&
                current_points_[i].x < image_width_ && current_points_[i].y < image_height_) {
                good_points.push_back(current_points_[i]);
            }
        }
        
        current_points_ = good_points;
    }
    
    bool initialized_;
    int image_width_, image_height_;
    int frame_id_;
    
    // OpenCV tracking parameters
    int max_features_;
    double quality_level_;
    double min_distance_;
    int block_size_;
    bool use_harris_detector_;
    double k_;
    
    // Tracking data
    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_points_;
    std::vector<cv::Point2f> current_points_;
    
    // Statistics
    std::vector<double> frame_times_;
    std::vector<double> track_times_;
    std::vector<size_t> tracked_features_;
    std::vector<size_t> detected_features_;
};

std::vector<std::string> getImageFiles(const std::string& directory) {
    std::vector<std::string> image_files;
    
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        std::cerr << "Directory does not exist: " << directory << std::endl;
        return image_files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        std::string ext = filename.substr(filename.find_last_of("."));
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
            image_files.push_back(directory + "/" + filename);
        }
    }
    closedir(dir);
    
    // Sort files by name
    std::sort(image_files.begin(), image_files.end());
    return image_files;
}

int main(int argc, char* argv[]) {
    std::string image_directory = "test/images/euroc/images/640_480";
    
    if (argc > 1) {
        image_directory = argv[1];
    }
    
    std::cout << "ViLib Feature Tracking Example" << std::endl;
    std::cout << "Looking for images in: " << image_directory << std::endl;
    
    // Get list of image files
    std::vector<std::string> image_files = getImageFiles(image_directory);
    
    if (image_files.empty()) {
        // Try alternative directories
        std::vector<std::string> alt_dirs = {
            "test/images",
            "../test/images",
            "./test/images/scenery/hut"
        };
        
        for (const auto& alt_dir : alt_dirs) {
            image_files = getImageFiles(alt_dir);
            if (!image_files.empty()) {
                std::cout << "Found images in alternative directory: " << alt_dir << std::endl;
                break;
            }
        }
        
        if (image_files.empty()) {
            std::cerr << "No image files found in any directory!" << std::endl;
            return -1;
        }
    }
    
    std::cout << "Found " << image_files.size() << " image files." << std::endl;
    
    // Create both tracker examples
    ViLibTrackingExample vilib_tracker;
    OpenCVTrackingExample opencv_tracker;
    
    // Process first image to get dimensions and initialize trackers
    cv::Mat first_image = cv::imread(image_files[0], cv::IMREAD_COLOR);
    if (first_image.empty()) {
        std::cerr << "Failed to load first image: " << image_files[0] << std::endl;
        return -1;
    }
    
    std::cout << "Image dimensions: " << first_image.cols << "x" << first_image.rows << std::endl;
    
    if (!vilib_tracker.initialize(first_image.cols, first_image.rows)) {
        std::cerr << "Failed to initialize ViLib tracker!" << std::endl;
        return -1;
    }
    
    if (!opencv_tracker.initialize(first_image.cols, first_image.rows)) {
        std::cerr << "Failed to initialize OpenCV tracker!" << std::endl;
        return -1;
    }
    
    std::cout << "Both trackers initialized successfully." << std::endl;
    std::cout << "Processing images with both ViLib GPU and OpenCV CPU trackers..." << std::endl;
    
    size_t max_frames = std::min(image_files.size(), static_cast<size_t>(50));
    
    // Process all images with both trackers
    for (size_t i = 0; i < max_frames; ++i) {
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_files[i] << std::endl;
            continue;
        }
        
        // Process with ViLib tracker
        vilib_tracker.processImage(image);
        
        // Process with OpenCV tracker
        opencv_tracker.processImage(image);
        
        std::cout << "---" << std::endl;
    }
    
    std::cout << "Processing complete!" << std::endl;
    std::cout << "\n" << std::string(60, '=') << std::endl;
    
    // Print final statistics for both trackers
    std::cout << "\nVILIB GPU TRACKER RESULTS:" << std::endl;
    vilib_tracker.printStatistics();
    
    std::cout << "\nOPENCV CPU TRACKER RESULTS:" << std::endl;
    opencv_tracker.printStatistics();
    
    return 0;
}
