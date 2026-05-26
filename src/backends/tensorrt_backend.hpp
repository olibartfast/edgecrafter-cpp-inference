#pragma once

#ifdef USE_TENSORRT

#include "inference_backend.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace edgecrafter::backend {

class TensorRtBackend : public InferenceBackend {
  public:
    TensorRtBackend() = default;
    ~TensorRtBackend() override;

    std::vector<int64_t> initialize(const std::filesystem::path &model_path,
                                    const std::vector<int64_t> &input_shape) override;

    void run_inference(const std::vector<Tensor> &inputs) override;

    [[nodiscard]] const std::vector<Tensor> &get_outputs() const override { return outputs_; }

    [[nodiscard]] std::string get_backend_name() const override { return "TensorRT"; }

  private:
    class Logger : public nvinfer1::ILogger {
      public:
        void log(Severity severity, const char *msg) noexcept override;
    };

    struct Binding {
        std::string name;
        TensorDataType dtype{};
        std::vector<int64_t> shape;
        nvinfer1::TensorIOMode mode{};
        void *device_buffer{};
        size_t byte_size{};
    };

    void release() noexcept;
    void release_buffers() noexcept;
    void allocate_buffers();
    void refresh_output_shapes();
    void set_input_shapes(const std::vector<int64_t> &image_shape);

    [[nodiscard]] static TensorDataType from_trt_type(nvinfer1::DataType type);
    [[nodiscard]] static std::vector<int64_t> shape_from_dims(const nvinfer1::Dims &dims);
    [[nodiscard]] static nvinfer1::Dims dims_from_shape(const std::vector<int64_t> &shape);
    [[nodiscard]] Binding &binding_for_input(size_t input_index);
    [[nodiscard]] const Binding &binding_for_output(size_t output_index) const;
    static void validate_input_tensor(const Tensor &tensor, const Binding &binding);

    Logger logger_;
    nvinfer1::IRuntime *runtime_{};
    nvinfer1::ICudaEngine *engine_{};
    nvinfer1::IExecutionContext *context_{};
    cudaStream_t stream_{};
    std::vector<Binding> bindings_;
    std::vector<size_t> input_binding_indices_;
    std::vector<size_t> output_binding_indices_;
    std::vector<Tensor> outputs_;
};

} // namespace edgecrafter::backend

#endif // USE_TENSORRT
