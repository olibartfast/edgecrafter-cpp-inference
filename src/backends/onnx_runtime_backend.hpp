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

    void run_inference(const std::vector<Tensor> &inputs) override;

    [[nodiscard]] const std::vector<Tensor> &get_outputs() const override { return outputs_; }

    [[nodiscard]] std::string get_backend_name() const override { return "ONNX Runtime"; }

  private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_;

    std::vector<Tensor> input_metadata_;
    std::vector<std::string> input_name_strings_;
    std::vector<const char *> input_names_;
    std::vector<std::string> output_name_strings_;
    std::vector<const char *> output_names_;
    std::vector<Tensor> outputs_;
};

} // namespace edgecrafter::backend

#endif // USE_ONNX_RUNTIME
