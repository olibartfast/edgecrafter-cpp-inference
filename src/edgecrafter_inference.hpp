#pragma once

#include "backends/inference_backend.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <span>
#include <string>
#include <vector>

using edgecrafter::backend::create_backend;
using edgecrafter::backend::InferenceBackend;

enum class TaskType { DETECTION, SEGMENTATION, POSE };

struct Config {
    int resolution{640};
    float threshold{0.5F};
    std::array<float, 3> means{0.485F, 0.456F, 0.406F};
    std::array<float, 3> stds{0.229F, 0.224F, 0.225F};
    TaskType task_type{TaskType::DETECTION};
    float mask_threshold{0.0F};
    int num_keypoints{17};
    float keypoint_threshold{0.3F};
};

struct Result {
    int class_id{};
    float score{};
    std::array<float, 4> box{};
    cv::Mat mask{};
    std::vector<std::array<float, 3>> keypoints{};
};

class EdgeCrafterInference {
  public:
    EdgeCrafterInference(const std::filesystem::path &model_path, const std::filesystem::path &label_file_path,
                         const Config &config = Config{});

    EdgeCrafterInference(std::unique_ptr<InferenceBackend> backend, const std::filesystem::path &label_file_path,
                         const Config &config = Config{});

    std::vector<float> preprocess_image(const std::filesystem::path &image_path, int &orig_h, int &orig_w);
    std::vector<float> preprocess_image(const cv::Mat &bgr_image, int &orig_h, int &orig_w);

    [[nodiscard]] std::vector<Result> infer(std::span<const float> input_data, int orig_h, int orig_w);

    void draw_results(cv::Mat &image, std::span<const Result> results);

    [[nodiscard]] std::optional<std::filesystem::path> save_output_image(const cv::Mat &image,
                                                                         const std::filesystem::path &output_path);

    [[nodiscard]] const std::vector<std::string> &get_labels() const noexcept { return labels_; }
    [[nodiscard]] int get_resolution() const noexcept { return config_.resolution; }

  private:
    void load_labels(const std::filesystem::path &label_file_path);
    [[nodiscard]] size_t output_index_or_throw(std::string_view name) const;
    [[nodiscard]] std::string label_for(int class_id) const;

    std::unique_ptr<InferenceBackend> backend_;
    std::vector<std::string> labels_;
    Config config_;
    std::vector<int64_t> input_shape_;
};
