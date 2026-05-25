#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace edgecrafter::backend {

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

    virtual void run_inference(std::span<const float> image_data, const std::vector<int64_t> &image_shape,
                               std::span<const int64_t, 2> orig_target_size) = 0;

    [[nodiscard]] virtual size_t get_output_count() const = 0;
    [[nodiscard]] virtual std::string get_output_name(size_t output_index) const = 0;
    [[nodiscard]] virtual std::vector<int64_t> get_output_shape(size_t output_index) const = 0;

    virtual void get_float_output_data(size_t output_index, float *data, size_t size) const = 0;
    virtual void get_int64_output_data(size_t output_index, int64_t *data, size_t size) const = 0;

    [[nodiscard]] virtual std::string get_backend_name() const = 0;
};

std::unique_ptr<InferenceBackend> create_backend();

} // namespace edgecrafter::backend
