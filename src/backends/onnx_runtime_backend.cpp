#ifdef USE_ONNX_RUNTIME

#include "onnx_runtime_backend.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace edgecrafter::backend {

namespace {

size_t tensor_size(const std::vector<int64_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1},
                           [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
}

} // namespace

OnnxRuntimeBackend::OnnxRuntimeBackend()
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "EdgeCrafterInference")),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

std::vector<int64_t> OnnxRuntimeBackend::initialize(const std::filesystem::path &model_path,
                                                    const std::vector<int64_t> &input_shape) {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path.string());
    }

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    const size_t input_count = session_->GetInputCount();
    if (input_count < 2) {
        throw std::runtime_error("EdgeCrafter ONNX export must have inputs: images and orig_target_sizes");
    }

    input_name_strings_.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        Ort::AllocatedStringPtr name_ptr = session_->GetInputNameAllocated(i, allocator_);
        input_name_strings_.emplace_back(name_ptr.get());
    }
    std::transform(input_name_strings_.begin(), input_name_strings_.end(), std::back_inserter(input_names_),
                   [](const std::string &name) { return name.c_str(); });

    std::vector<int64_t> detected_shape = input_shape;
    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
    if ((input_shape[2] == 0 || input_shape[3] == 0) && shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
        detected_shape = shape;
        detected_shape[0] = 1;
        std::cout << "[ONNX Runtime] Auto-detected input size: " << shape[2] << "x" << shape[3] << std::endl;
    }

    const size_t output_count = session_->GetOutputCount();
    output_name_strings_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        output_name_strings_.emplace_back(name_ptr.get());
    }
    std::transform(output_name_strings_.begin(), output_name_strings_.end(), std::back_inserter(output_names_),
                   [](const std::string &name) { return name.c_str(); });

    std::cout << "[ONNX Runtime] Inputs:";
    for (const auto &name : input_name_strings_) {
        std::cout << " " << name;
    }
    std::cout << "\n[ONNX Runtime] Outputs:";
    for (const auto &name : output_name_strings_) {
        std::cout << " " << name;
    }
    std::cout << std::endl;

    return detected_shape;
}

void OnnxRuntimeBackend::run_inference(std::span<const float> image_data, const std::vector<int64_t> &image_shape,
                                       std::span<const int64_t, 2> orig_target_size) {
    if (!session_) {
        throw std::runtime_error("Backend has not been initialized");
    }

    std::array<int64_t, 2> size_shape{1, 2};
    Ort::Value image_tensor =
        Ort::Value::CreateTensor<float>(memory_info_, const_cast<float *>(image_data.data()), image_data.size(),
                                        image_shape.data(), image_shape.size());
    Ort::Value size_tensor =
        Ort::Value::CreateTensor<int64_t>(memory_info_, const_cast<int64_t *>(orig_target_size.data()),
                                          orig_target_size.size(), size_shape.data(), size_shape.size());

    std::array<Ort::Value, 2> input_tensors{std::move(image_tensor), std::move(size_tensor)};
    output_tensors_ = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
                                    input_tensors.size(), output_names_.data(), output_names_.size());
}

size_t OnnxRuntimeBackend::get_output_count() const { return output_name_strings_.size(); }

std::string OnnxRuntimeBackend::get_output_name(size_t output_index) const {
    if (output_index >= output_name_strings_.size()) {
        throw std::out_of_range("Output index out of range");
    }
    return output_name_strings_[output_index];
}

std::vector<int64_t> OnnxRuntimeBackend::get_output_shape(size_t output_index) const {
    if (output_index >= output_tensors_.size()) {
        throw std::out_of_range("Output index out of range");
    }
    return output_tensors_[output_index].GetTensorTypeAndShapeInfo().GetShape();
}

void OnnxRuntimeBackend::get_float_output_data(size_t output_index, float *data, size_t size) const {
    if (output_index >= output_tensors_.size()) {
        throw std::out_of_range("Output index out of range");
    }
    auto shape = get_output_shape(output_index);
    const size_t actual_size = tensor_size(shape);
    if (actual_size != size) {
        throw std::runtime_error("Output tensor size mismatch");
    }
    const float *tensor_data = output_tensors_[output_index].GetTensorData<float>();
    std::copy(tensor_data, tensor_data + size, data);
}

void OnnxRuntimeBackend::get_int64_output_data(size_t output_index, int64_t *data, size_t size) const {
    if (output_index >= output_tensors_.size()) {
        throw std::out_of_range("Output index out of range");
    }
    auto shape = get_output_shape(output_index);
    const size_t actual_size = tensor_size(shape);
    if (actual_size != size) {
        throw std::runtime_error("Output tensor size mismatch");
    }
    const int64_t *tensor_data = output_tensors_[output_index].GetTensorData<int64_t>();
    std::copy(tensor_data, tensor_data + size, data);
}

} // namespace edgecrafter::backend

#endif // USE_ONNX_RUNTIME
