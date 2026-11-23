#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Google Cloud TPU backend powered by OpenXLA PJRT_Api (dlopen(libtpu.so) -> GetPjrtApi()).
// Zero Python, zero JAX, zero vllm-tpu at runtime.

GGML_BACKEND_API ggml_backend_t ggml_backend_cloudtpu_init(void);

GGML_BACKEND_API ggml_backend_t ggml_backend_cloudtpu_init_device(int device_index);

GGML_BACKEND_API bool ggml_backend_is_cloudtpu(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_cloudtpu_is_mock(ggml_backend_t backend);

GGML_BACKEND_API int ggml_backend_cloudtpu_get_device_count(void);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cloudtpu_buffer_type(int device_index);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cloudtpu_reg(void);

#ifdef __cplusplus
}
#endif
