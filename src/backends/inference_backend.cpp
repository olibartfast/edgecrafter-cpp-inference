#include "inference_backend.hpp"

#include <numeric>
#include <stdexcept>

#ifdef USE_ONNX_RUNTIME
#include "onnx_runtime_backend.hpp"
#endif
#ifdef USE_TENSORRT
#include "tensorrt_backend.hpp"
#endif

namespace edgecrafter::backend {

size_t tensor_element_size(TensorDataType dtype) {
    switch (dtype) {
    case TensorDataType::Float16:
        return 2;
    case TensorDataType::Float32:
    case TensorDataType::Int32:
        return 4;
    case TensorDataType::Int64:
        return 8;
    case TensorDataType::Int8:
    case TensorDataType::UInt8:
    case TensorDataType::Bool:
        return 1;
    }
    throw std::runtime_error("Unsupported tensor data type");
}

size_t tensor_element_count(const std::vector<int64_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, [](size_t acc, int64_t dim) {
        if (dim < 0) {
            throw std::runtime_error("Cannot calculate tensor size for dynamic shape");
        }
        return acc * static_cast<size_t>(dim);
    });
}

size_t tensor_byte_size(TensorDataType dtype, const std::vector<int64_t> &shape) {
    return tensor_element_count(shape) * tensor_element_size(dtype);
}

const char *tensor_dtype_name(TensorDataType dtype) noexcept {
    switch (dtype) {
    case TensorDataType::Float16:
        return "float16";
    case TensorDataType::Float32:
        return "float32";
    case TensorDataType::Int8:
        return "int8";
    case TensorDataType::Int32:
        return "int32";
    case TensorDataType::Int64:
        return "int64";
    case TensorDataType::UInt8:
        return "uint8";
    case TensorDataType::Bool:
        return "bool";
    }
    return "unknown";
}

std::unique_ptr<InferenceBackend> create_backend() {
#ifdef USE_TENSORRT
    return std::make_unique<TensorRtBackend>();
#elif defined(USE_ONNX_RUNTIME)
    return std::make_unique<OnnxRuntimeBackend>();
#else
#error "No backend enabled. Build with -DUSE_ONNX_RUNTIME=ON or -DUSE_TENSORRT=ON"
#endif
}

} // namespace edgecrafter::backend
