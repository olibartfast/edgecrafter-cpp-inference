#ifdef USE_ONNX_RUNTIME

#include "onnx_runtime_backend.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace edgecrafter::backend {

namespace {

TensorDataType from_onnx_type(ONNXTensorElementDataType type) {
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return TensorDataType::Float32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return TensorDataType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return TensorDataType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return TensorDataType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return TensorDataType::UInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return TensorDataType::Bool;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return TensorDataType::Float16;
    default:
        throw std::runtime_error("Unsupported ONNX tensor element type: " + std::to_string(type));
    }
}

ONNXTensorElementDataType to_onnx_type(TensorDataType dtype) {
    switch (dtype) {
    case TensorDataType::Float16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TensorDataType::Float32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case TensorDataType::Int8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case TensorDataType::Int32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case TensorDataType::Int64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TensorDataType::UInt8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case TensorDataType::Bool:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    throw std::runtime_error("Unsupported tensor data type");
}

void validate_input_tensor(const Tensor &expected, const Tensor &actual) {
    if (actual.name != expected.name) {
        throw std::runtime_error("Input tensor name mismatch. Expected " + expected.name + ", got " + actual.name);
    }
    if (actual.dtype != expected.dtype) {
        throw std::runtime_error("Input tensor dtype mismatch for " + actual.name + ". Expected " +
                                 tensor_dtype_name(expected.dtype) + ", got " + tensor_dtype_name(actual.dtype));
    }
    if (actual.shape != expected.shape) {
        throw std::runtime_error("Input tensor shape mismatch for " + actual.name);
    }
    const size_t expected_bytes = tensor_byte_size(expected.dtype, expected.shape);
    if (actual.bytes.size() != expected_bytes) {
        throw std::runtime_error("Input tensor size mismatch for " + actual.name + ". Expected " +
                                 std::to_string(expected_bytes) + " bytes, got " + std::to_string(actual.bytes.size()));
    }
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

    std::vector<int64_t> detected_shape = input_shape;
    input_metadata_.clear();
    input_name_strings_.clear();
    input_names_.clear();
    input_metadata_.reserve(input_count);
    input_name_strings_.reserve(input_count);

    for (size_t i = 0; i < input_count; ++i) {
        Ort::AllocatedStringPtr name_ptr = session_->GetInputNameAllocated(i, allocator_);
        input_name_strings_.emplace_back(name_ptr.get());

        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        Tensor tensor{
            input_name_strings_.back(), from_onnx_type(tensor_info.GetElementType()), tensor_info.GetShape(), {}};
        if (i == 0 && tensor.shape.size() == input_shape.size()) {
            if ((input_shape[2] == 0 || input_shape[3] == 0) && tensor.shape[2] > 0 && tensor.shape[3] > 0) {
                detected_shape = tensor.shape;
                detected_shape[0] = 1;
                std::cout << "[ONNX Runtime] Auto-detected input size: " << tensor.shape[2] << "x" << tensor.shape[3]
                          << std::endl;
            }
            tensor.shape = detected_shape;
        } else if (i == 1 && tensor.shape.size() == 2) {
            tensor.shape = {1, 2};
        }
        input_metadata_.push_back(std::move(tensor));
    }
    std::transform(input_name_strings_.begin(), input_name_strings_.end(), std::back_inserter(input_names_),
                   [](const std::string &name) { return name.c_str(); });

    const size_t output_count = session_->GetOutputCount();
    output_name_strings_.clear();
    output_names_.clear();
    output_name_strings_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        output_name_strings_.emplace_back(name_ptr.get());
    }
    std::transform(output_name_strings_.begin(), output_name_strings_.end(), std::back_inserter(output_names_),
                   [](const std::string &name) { return name.c_str(); });

    std::cout << "[ONNX Runtime] Inputs:";
    for (const auto &tensor : input_metadata_) {
        std::cout << " " << tensor.name << ":" << tensor_dtype_name(tensor.dtype);
    }
    std::cout << "\n[ONNX Runtime] Outputs:";
    for (const auto &name : output_name_strings_) {
        std::cout << " " << name;
    }
    std::cout << std::endl;

    return detected_shape;
}

void OnnxRuntimeBackend::run_inference(const std::vector<Tensor> &inputs) {
    if (!session_) {
        throw std::runtime_error("Backend has not been initialized");
    }
    if (inputs.size() != input_metadata_.size()) {
        throw std::runtime_error("Input tensor count mismatch. Expected " + std::to_string(input_metadata_.size()) +
                                 ", got " + std::to_string(inputs.size()));
    }

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        validate_input_tensor(input_metadata_[i], inputs[i]);
        const auto &input = inputs[i];
        uint8_t *data = const_cast<uint8_t *>(input.bytes.data());
        const size_t byte_count = input.bytes.size();
        const auto onnx_type = to_onnx_type(input.dtype);
        ort_inputs.emplace_back(Ort::Value::CreateTensor(memory_info_, data, byte_count, input.shape.data(),
                                                         input.shape.size(), onnx_type));
    }

    std::vector<Ort::Value> ort_outputs =
        session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), ort_inputs.data(), ort_inputs.size(),
                      output_names_.data(), output_names_.size());

    outputs_.clear();
    outputs_.reserve(ort_outputs.size());
    for (size_t i = 0; i < ort_outputs.size(); ++i) {
        auto tensor_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
        Tensor output{
            output_name_strings_[i], from_onnx_type(tensor_info.GetElementType()), tensor_info.GetShape(), {}};
        output.bytes.resize(tensor_byte_size(output.dtype, output.shape));
        const void *data = ort_outputs[i].GetTensorRawData();
        std::memcpy(output.bytes.data(), data, output.bytes.size());
        outputs_.push_back(std::move(output));
    }
}

} // namespace edgecrafter::backend

#endif // USE_ONNX_RUNTIME
