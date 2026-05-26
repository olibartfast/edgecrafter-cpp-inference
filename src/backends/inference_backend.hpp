#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace edgecrafter::backend {

enum class TensorDataType { Float16, Float32, Int8, Int32, Int64, UInt8, Bool };

struct Tensor {
    std::string name;
    TensorDataType dtype{};
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

[[nodiscard]] size_t tensor_element_size(TensorDataType dtype);
[[nodiscard]] size_t tensor_element_count(const std::vector<int64_t> &shape);
[[nodiscard]] size_t tensor_byte_size(TensorDataType dtype, const std::vector<int64_t> &shape);
[[nodiscard]] const char *tensor_dtype_name(TensorDataType dtype) noexcept;

class InferenceBackend {
  public:
    InferenceBackend() = default;
    virtual ~InferenceBackend() = default;
    InferenceBackend(const InferenceBackend &) = delete;
    InferenceBackend &operator=(const InferenceBackend &) = delete;
    InferenceBackend(InferenceBackend &&) = delete;
    InferenceBackend &operator=(InferenceBackend &&) = delete;

    virtual std::vector<int64_t> initialize(const std::filesystem::path &model_path,
                                            const std::vector<int64_t> &input_shape) = 0;

    virtual void run_inference(const std::vector<Tensor> &inputs) = 0;

    [[nodiscard]] virtual const std::vector<Tensor> &get_outputs() const = 0;

    [[nodiscard]] virtual std::string get_backend_name() const = 0;
};

std::unique_ptr<InferenceBackend> create_backend();

} // namespace edgecrafter::backend
