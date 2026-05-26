#include "edgecrafter_inference.hpp"

#include "processing_utils.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace {

using edgecrafter::backend::Tensor;
using edgecrafter::backend::TensorDataType;

size_t tensor_size(const std::vector<int64_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1},
                           [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
}

template <typename T> std::vector<uint8_t> bytes_from_span(std::span<const T> values) {
    std::vector<uint8_t> bytes(values.size_bytes());
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

template <typename T> std::vector<T> tensor_values_as(const Tensor &tensor, TensorDataType expected_dtype) {
    if (tensor.dtype != expected_dtype) {
        throw std::runtime_error("Tensor '" + tensor.name + "' has dtype " +
                                 edgecrafter::backend::tensor_dtype_name(tensor.dtype) + ", expected " +
                                 edgecrafter::backend::tensor_dtype_name(expected_dtype));
    }
    if (tensor.bytes.size() != tensor_size(tensor.shape) * sizeof(T)) {
        throw std::runtime_error("Tensor byte size mismatch for output: " + tensor.name);
    }

    std::vector<T> values(tensor.bytes.size() / sizeof(T));
    std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    return values;
}

} // namespace

EdgeCrafterInference::EdgeCrafterInference(const std::filesystem::path &model_path,
                                           const std::filesystem::path &label_file_path, const Config &config)
    : backend_(create_backend()), config_(config), input_shape_({1, 3, config.resolution, config.resolution}) {
    input_shape_ = backend_->initialize(model_path, input_shape_);
    if (config_.resolution == 0 && input_shape_.size() == 4) {
        config_.resolution = static_cast<int>(input_shape_[2]);
    }
    load_labels(label_file_path);
}

EdgeCrafterInference::EdgeCrafterInference(std::unique_ptr<InferenceBackend> backend,
                                           const std::filesystem::path &label_file_path, const Config &config)
    : backend_(std::move(backend)), config_(config), input_shape_({1, 3, config.resolution, config.resolution}) {
    load_labels(label_file_path);
}

void EdgeCrafterInference::load_labels(const std::filesystem::path &label_file_path) {
    std::ifstream file(label_file_path);
    if (!file) {
        throw std::runtime_error("Could not open label file: " + label_file_path.string());
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }
    if (labels_.empty()) {
        throw std::runtime_error("No labels found in file: " + label_file_path.string());
    }
}

std::vector<float> EdgeCrafterInference::preprocess_image(const std::filesystem::path &image_path, int &orig_h,
                                                          int &orig_w) {
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not load image: " + image_path.string());
    }
    return preprocess_image(image, orig_h, orig_w);
}

std::vector<float> EdgeCrafterInference::preprocess_image(const cv::Mat &bgr_image, int &orig_h, int &orig_w) {
    if (bgr_image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    orig_h = bgr_image.rows;
    orig_w = bgr_image.cols;

    const auto res = static_cast<size_t>(config_.resolution);
    std::vector<float> input_tensor(3 * res * res);
    edgecrafter::processing::preprocess_frame(bgr_image, input_tensor, config_.resolution, config_.means, config_.stds);
    return input_tensor;
}

std::vector<Result> EdgeCrafterInference::infer(std::span<const float> input_data, int orig_h, int orig_w) {
    std::array<int64_t, 2> orig_target_size{static_cast<int64_t>(orig_w), static_cast<int64_t>(orig_h)};
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor{"images", TensorDataType::Float32, input_shape_, bytes_from_span(input_data)});
    inputs.push_back(
        Tensor{"orig_target_sizes", TensorDataType::Int64, {1, 2}, bytes_from_span<int64_t>(orig_target_size)});

    backend_->run_inference(inputs);
    const auto &outputs = backend_->get_outputs();

    const size_t labels_idx = output_index_or_throw(outputs, "labels");
    const size_t scores_idx = output_index_or_throw(outputs, "scores");

    const auto &labels_tensor = outputs[labels_idx];
    const auto &scores_tensor = outputs[scores_idx];
    const auto &labels_shape = labels_tensor.shape;
    const auto &scores_shape = scores_tensor.shape;
    if (labels_shape.size() != 2 || scores_shape != labels_shape) {
        throw std::runtime_error("Unexpected EdgeCrafter output shapes");
    }

    std::vector<int64_t> labels = tensor_values_as<int64_t>(labels_tensor, TensorDataType::Int64);
    std::vector<float> scores = tensor_values_as<float>(scores_tensor, TensorDataType::Float32);

    std::vector<float> boxes;
    std::vector<int64_t> boxes_shape;
    bool has_boxes = false;
    size_t boxes_idx = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].name == "boxes") {
            boxes_idx = i;
            has_boxes = true;
            break;
        }
    }
    if (has_boxes) {
        const auto &boxes_tensor = outputs[boxes_idx];
        boxes_shape = boxes_tensor.shape;
        if (boxes_shape.size() != 3 || boxes_shape[2] != 4 || boxes_shape[1] != labels_shape[1]) {
            throw std::runtime_error("Unexpected EdgeCrafter boxes output shape");
        }
        boxes = tensor_values_as<float>(boxes_tensor, TensorDataType::Float32);
    }

    std::vector<float> masks;
    std::vector<int64_t> masks_shape;
    bool has_masks = false;
    if (config_.task_type == TaskType::SEGMENTATION) {
        const size_t masks_idx = output_index_or_throw(outputs, "masks");
        const auto &masks_tensor = outputs[masks_idx];
        masks_shape = masks_tensor.shape;
        if (masks_shape.size() != 4 || masks_shape[1] != labels_shape[1]) {
            throw std::runtime_error("Unexpected EdgeCrafter masks output shape");
        }
        masks = tensor_values_as<float>(masks_tensor, TensorDataType::Float32);
        has_masks = true;
    }

    std::vector<float> keypoints;
    std::vector<int64_t> keypoints_shape;
    bool has_keypoints = false;
    if (config_.task_type == TaskType::POSE) {
        const size_t keypoints_idx = output_index_or_throw(outputs, "keypoints");
        const auto &keypoints_tensor = outputs[keypoints_idx];
        keypoints_shape = keypoints_tensor.shape;
        const auto kpt_dim = keypoints_shape.size() == 4 ? keypoints_shape[3] : 0;
        if (keypoints_shape.size() != 4 || keypoints_shape[1] != labels_shape[1] || (kpt_dim != 2 && kpt_dim != 3)) {
            throw std::runtime_error("Unexpected EdgeCrafter keypoints output shape");
        }
        keypoints = tensor_values_as<float>(keypoints_tensor, TensorDataType::Float32);
        has_keypoints = true;
    }

    std::vector<Result> results;
    const auto count = static_cast<size_t>(labels_shape[1]);
    results.reserve(count);
    const auto num_kpts = has_keypoints ? static_cast<size_t>(keypoints_shape[2]) : 0;
    const auto kpt_dim = has_keypoints ? static_cast<int64_t>(keypoints_shape[3]) : int64_t{0};
    for (size_t i = 0; i < count; ++i) {
        const float score = scores[i];
        if (score <= config_.threshold) {
            continue;
        }

        Result result;
        result.class_id = static_cast<int>(labels[i]) + config_.label_offset;
        result.score = score;

        if (has_boxes) {
            const size_t box_offset = i * 4;
            result.box = {boxes[box_offset], boxes[box_offset + 1], boxes[box_offset + 2], boxes[box_offset + 3]};
        }

        if (has_keypoints) {
            result.keypoints.resize(num_kpts);
            const size_t kpt_offset = i * num_kpts * static_cast<size_t>(kpt_dim);
            for (size_t k = 0; k < num_kpts; ++k) {
                const size_t off = kpt_offset + k * static_cast<size_t>(kpt_dim);
                const float conf = kpt_dim == 3 ? keypoints[off + 2] : 1.0F;
                result.keypoints[k] = {keypoints[off], keypoints[off + 1], conf};
            }

            if (!has_boxes) {
                float x_min = std::numeric_limits<float>::max();
                float y_min = std::numeric_limits<float>::max();
                float x_max = 0.0F;
                float y_max = 0.0F;
                bool any = false;
                for (const auto &kp : result.keypoints) {
                    if (kp[2] > config_.keypoint_threshold) {
                        x_min = std::min(x_min, kp[0]);
                        y_min = std::min(y_min, kp[1]);
                        x_max = std::max(x_max, kp[0]);
                        y_max = std::max(y_max, kp[1]);
                        any = true;
                    }
                }
                if (any) {
                    result.box = {x_min, y_min, x_max, y_max};
                }
            }
        }

        if (has_masks) {
            const auto mask_h = static_cast<int>(masks_shape[2]);
            const auto mask_w = static_cast<int>(masks_shape[3]);
            const size_t mask_offset = i * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);
            cv::Mat mask_small(mask_h, mask_w, CV_32F, masks.data() + mask_offset);
            cv::Mat mask_resized;
            cv::resize(mask_small, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
            result.mask = mask_resized > config_.mask_threshold;
        }

        results.push_back(std::move(result));
    }

    return results;
}

size_t EdgeCrafterInference::output_index_or_throw(std::span<const Tensor> outputs, std::string_view name) {
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].name == name) {
            return i;
        }
    }
    throw std::runtime_error("Model output not found: " + std::string(name));
}

std::string EdgeCrafterInference::label_for(int class_id) const {
    if (class_id >= 0 && static_cast<size_t>(class_id) < labels_.size()) {
        return labels_[static_cast<size_t>(class_id)];
    }
    return std::to_string(class_id);
}

void EdgeCrafterInference::draw_results(cv::Mat &image, std::span<const Result> results) {
    constexpr double alpha = 0.45;
    cv::Mat overlay;

    if (config_.task_type == TaskType::SEGMENTATION) {
        overlay = image.clone();
        for (const auto &result : results) {
            const cv::Scalar color = edgecrafter::processing::get_color_for_class(result.class_id);
            if (!result.mask.empty() && result.mask.rows == image.rows && result.mask.cols == image.cols) {
                overlay.setTo(color, result.mask);
            }
        }
        cv::addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image);
    }

    for (const auto &result : results) {
        const cv::Scalar color = edgecrafter::processing::get_color_for_class(result.class_id);
        cv::Point2f p1(result.box[0], result.box[1]);
        cv::Point2f p2(result.box[2], result.box[3]);
        cv::rectangle(image, p1, p2, color, 2);

        const std::string text = label_for(result.class_id) + " " + std::to_string(result.score).substr(0, 4);
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.75, 2, &baseline);
        cv::Point text_org(static_cast<int>(std::max(0.0F, p1.x)), static_cast<int>(std::max(18.0F, p1.y - 4.0F)));
        cv::rectangle(image, cv::Point(text_org.x, text_org.y - text_size.height - baseline - 4),
                      cv::Point(text_org.x + text_size.width + 5, text_org.y + 3), color, cv::FILLED);
        cv::putText(image, text, cv::Point(text_org.x + 3, text_org.y - baseline - 2), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        if (config_.task_type == TaskType::POSE && !result.keypoints.empty()) {
            edgecrafter::processing::draw_keypoints(
                image, result.keypoints, edgecrafter::processing::coco_skeleton_edges(), config_.keypoint_threshold);
        }
    }
}

std::optional<std::filesystem::path> EdgeCrafterInference::save_output_image(const cv::Mat &image,
                                                                             const std::filesystem::path &output_path) {
    if (cv::imwrite(output_path.string(), image)) {
        return output_path;
    }
    return std::nullopt;
}
