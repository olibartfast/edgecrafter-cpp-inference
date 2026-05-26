#ifdef USE_TENSORRT

#include "tensorrt_backend.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace edgecrafter::backend {

namespace {

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

bool has_dynamic_dim(const std::vector<int64_t> &shape) {
    return std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; });
}

} // namespace

void TensorRtBackend::Logger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT] " << msg << '\n';
    }
}

TensorRtBackend::~TensorRtBackend() { release(); }

void TensorRtBackend::release_buffers() noexcept {
    for (auto &binding : bindings_) {
        if (binding.device_buffer != nullptr) {
            cudaFree(binding.device_buffer);
            binding.device_buffer = nullptr;
        }
        binding.byte_size = 0;
    }
}

void TensorRtBackend::release() noexcept {
    release_buffers();
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    delete context_;
    context_ = nullptr;
    delete engine_;
    engine_ = nullptr;
    delete runtime_;
    runtime_ = nullptr;
}

TensorDataType TensorRtBackend::from_trt_type(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return TensorDataType::Float32;
    case nvinfer1::DataType::kHALF:
        return TensorDataType::Float16;
    case nvinfer1::DataType::kINT8:
        return TensorDataType::Int8;
    case nvinfer1::DataType::kINT32:
        return TensorDataType::Int32;
    case nvinfer1::DataType::kINT64:
        return TensorDataType::Int64;
    case nvinfer1::DataType::kUINT8:
        return TensorDataType::UInt8;
    case nvinfer1::DataType::kBOOL:
        return TensorDataType::Bool;
    default:
        throw std::runtime_error("Unsupported TensorRT tensor data type");
    }
}

std::vector<int64_t> TensorRtBackend::shape_from_dims(const nvinfer1::Dims &dims) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(dims.nbDims));
    for (int i = 0; i < dims.nbDims; ++i) {
        shape.push_back(static_cast<int64_t>(dims.d[i]));
    }
    return shape;
}

nvinfer1::Dims TensorRtBackend::dims_from_shape(const std::vector<int64_t> &shape) {
    if (shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS)) {
        throw std::runtime_error("Tensor shape exceeds TensorRT maximum rank");
    }
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        dims.d[i] = static_cast<int32_t>(shape[i]);
    }
    return dims;
}

TensorRtBackend::Binding &TensorRtBackend::binding_for_input(size_t input_index) {
    return bindings_.at(input_binding_indices_.at(input_index));
}

const TensorRtBackend::Binding &TensorRtBackend::binding_for_output(size_t output_index) const {
    return bindings_.at(output_binding_indices_.at(output_index));
}

void TensorRtBackend::validate_input_tensor(const Tensor &tensor, const Binding &binding) {
    if (tensor.name != binding.name) {
        throw std::runtime_error("Input tensor name mismatch. Expected " + binding.name + ", got " + tensor.name);
    }
    if (tensor.dtype != binding.dtype) {
        throw std::runtime_error("Input tensor dtype mismatch for " + tensor.name + ". Expected " +
                                 tensor_dtype_name(binding.dtype) + ", got " + tensor_dtype_name(tensor.dtype));
    }
    if (tensor.shape != binding.shape) {
        throw std::runtime_error("Input tensor shape mismatch for " + tensor.name);
    }
    if (tensor.bytes.size() != binding.byte_size) {
        throw std::runtime_error("Input tensor size mismatch for " + tensor.name + ". Expected " +
                                 std::to_string(binding.byte_size) + " bytes, got " +
                                 std::to_string(tensor.bytes.size()));
    }
}

std::vector<int64_t> TensorRtBackend::initialize(const std::filesystem::path &model_path,
                                                 const std::vector<int64_t> &input_shape) {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("TensorRT engine file does not exist: " + model_path.string());
    }

    std::ifstream engine_file(model_path, std::ios::binary | std::ios::ate);
    if (!engine_file) {
        throw std::runtime_error("Failed to open TensorRT engine file: " + model_path.string());
    }
    const auto file_size = static_cast<size_t>(engine_file.tellg());
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(file_size);
    engine_file.read(engine_data.data(), static_cast<std::streamsize>(engine_data.size()));

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (runtime_ == nullptr) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine_ == nullptr) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    bindings_.clear();
    input_binding_indices_.clear();
    output_binding_indices_.clear();
    const int tensor_count = engine_->getNbIOTensors();
    bindings_.reserve(static_cast<size_t>(tensor_count));
    for (int i = 0; i < tensor_count; ++i) {
        const char *name = engine_->getIOTensorName(i);
        Binding binding{std::string(name),
                        from_trt_type(engine_->getTensorDataType(name)),
                        shape_from_dims(engine_->getTensorShape(name)),
                        engine_->getTensorIOMode(name),
                        nullptr,
                        0};
        const size_t index = bindings_.size();
        if (binding.mode == nvinfer1::TensorIOMode::kINPUT) {
            input_binding_indices_.push_back(index);
        } else {
            output_binding_indices_.push_back(index);
        }
        bindings_.push_back(std::move(binding));
    }

    if (input_binding_indices_.size() < 2) {
        throw std::runtime_error("EdgeCrafter TensorRT engine must have inputs: images and orig_target_sizes");
    }

    std::vector<int64_t> detected_shape = input_shape;
    auto &image_binding = binding_for_input(0);
    if ((input_shape[2] == 0 || input_shape[3] == 0) && image_binding.shape.size() == 4 && image_binding.shape[2] > 0 &&
        image_binding.shape[3] > 0) {
        detected_shape = image_binding.shape;
        detected_shape[0] = 1;
        std::cout << "[TensorRT] Auto-detected input size: " << image_binding.shape[2] << "x" << image_binding.shape[3]
                  << std::endl;
    }
    set_input_shapes(detected_shape);
    allocate_buffers();

    std::cout << "[TensorRT] Inputs:";
    for (size_t index : input_binding_indices_) {
        const auto &binding = bindings_[index];
        std::cout << " " << binding.name << ":" << tensor_dtype_name(binding.dtype);
    }
    std::cout << "\n[TensorRT] Outputs:";
    for (size_t index : output_binding_indices_) {
        std::cout << " " << bindings_[index].name;
    }
    std::cout << std::endl;

    return detected_shape;
}

void TensorRtBackend::set_input_shapes(const std::vector<int64_t> &image_shape) {
    auto &image_binding = binding_for_input(0);
    image_binding.shape = image_shape;
    if (!context_->setInputShape(image_binding.name.c_str(), dims_from_shape(image_binding.shape))) {
        throw std::runtime_error("Failed to set TensorRT input shape for " + image_binding.name);
    }

    auto &size_binding = binding_for_input(1);
    size_binding.shape = {1, 2};
    if (!context_->setInputShape(size_binding.name.c_str(), dims_from_shape(size_binding.shape))) {
        throw std::runtime_error("Failed to set TensorRT input shape for " + size_binding.name);
    }

    refresh_output_shapes();
}

void TensorRtBackend::refresh_output_shapes() {
    for (size_t index : output_binding_indices_) {
        auto &binding = bindings_[index];
        binding.shape = shape_from_dims(context_->getTensorShape(binding.name.c_str()));
        if (has_dynamic_dim(binding.shape)) {
            throw std::runtime_error("TensorRT output has unresolved dynamic shape: " + binding.name);
        }
    }
}

void TensorRtBackend::allocate_buffers() {
    release_buffers();
    for (auto &binding : bindings_) {
        if (has_dynamic_dim(binding.shape)) {
            throw std::runtime_error("TensorRT tensor has unresolved dynamic shape: " + binding.name);
        }
        binding.byte_size = tensor_byte_size(binding.dtype, binding.shape);
        check_cuda(cudaMalloc(&binding.device_buffer, std::max<size_t>(1, binding.byte_size)), "cudaMalloc");
        if (!context_->setTensorAddress(binding.name.c_str(), binding.device_buffer)) {
            throw std::runtime_error("Failed to bind TensorRT tensor address for " + binding.name);
        }
    }
}

void TensorRtBackend::run_inference(const std::vector<Tensor> &inputs) {
    if (context_ == nullptr) {
        throw std::runtime_error("Backend has not been initialized");
    }
    if (inputs.size() != input_binding_indices_.size()) {
        throw std::runtime_error("Input tensor count mismatch. Expected " +
                                 std::to_string(input_binding_indices_.size()) + ", got " +
                                 std::to_string(inputs.size()));
    }

    bool shape_changed = false;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto &binding = binding_for_input(i);
        if (inputs[i].shape != binding.shape) {
            binding.shape = inputs[i].shape;
            if (!context_->setInputShape(binding.name.c_str(), dims_from_shape(binding.shape))) {
                throw std::runtime_error("Failed to set TensorRT input shape for " + binding.name);
            }
            shape_changed = true;
        }
    }
    if (shape_changed) {
        refresh_output_shapes();
        allocate_buffers();
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto &binding = binding_for_input(i);
        validate_input_tensor(inputs[i], binding);
        check_cuda(cudaMemcpyAsync(binding.device_buffer, inputs[i].bytes.data(), inputs[i].bytes.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync input");
    }

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT inference failed");
    }

    outputs_.clear();
    outputs_.reserve(output_binding_indices_.size());
    for (size_t i = 0; i < output_binding_indices_.size(); ++i) {
        const auto &binding = binding_for_output(i);
        Tensor output{binding.name, binding.dtype, binding.shape, std::vector<uint8_t>(binding.byte_size)};
        check_cuda(cudaMemcpyAsync(output.bytes.data(), binding.device_buffer, output.bytes.size(),
                                   cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync output");
        outputs_.push_back(std::move(output));
    }
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

} // namespace edgecrafter::backend

#endif // USE_TENSORRT
