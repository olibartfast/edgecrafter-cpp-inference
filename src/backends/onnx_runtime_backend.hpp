#pragma once

#ifdef USE_ONNX_RUNTIME

#include "inference_backend.hpp"

#include <memory>
#include <onnxruntime_cxx_api.h>

namespace edgecrafter::backend {

class OnnxRuntimeBackend : public InferenceBackend {
  public:
    OnnxRuntimeBackend();
    ~OnnxRuntimeBackend() override = default;

    std::vector<int64_t> initialize(const std::filesystem::path &model_path,
                                    const std::vector<int64_t> &input_shape) override;

    void run_inference(std::span<const float> image_data, const std::vector<int64_t> &image_shape,
                       std::span<const int64_t, 2> orig_target_size) override;

    [[nodiscard]] size_t get_output_count() const override;
    [[nodiscard]] std::string get_output_name(size_t output_index) const override;
    [[nodiscard]] std::vector<int64_t> get_output_shape(size_t output_index) const override;

    void get_float_output_data(size_t output_index, float *data, size_t size) const override;
    void get_int64_output_data(size_t output_index, int64_t *data, size_t size) const override;

    [[nodiscard]] std::string get_backend_name() const override { return "ONNX Runtime"; }

  private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_;

    std::vector<std::string> input_name_strings_;
    std::vector<const char *> input_names_;
    std::vector<std::string> output_name_strings_;
    std::vector<const char *> output_names_;
    std::vector<Ort::Value> output_tensors_;
};

} // namespace edgecrafter::backend

#endif // USE_ONNX_RUNTIME
