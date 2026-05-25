#include "inference_backend.hpp"

#include <stdexcept>

#ifdef USE_ONNX_RUNTIME
#include "onnx_runtime_backend.hpp"
#endif

namespace edgecrafter::backend {

std::unique_ptr<InferenceBackend> create_backend() {
#ifdef USE_ONNX_RUNTIME
    return std::make_unique<OnnxRuntimeBackend>();
#else
#error "No backend enabled. Build with -DUSE_ONNX_RUNTIME=ON"
#endif
}

} // namespace edgecrafter::backend
