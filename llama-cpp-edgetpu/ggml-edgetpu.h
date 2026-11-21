#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pixel Tensor G5 (buenos_a0) bare-metal EdgeTPU backend powered by libopenbuenos.

GGML_BACKEND_API ggml_backend_t ggml_backend_edgetpu_init(void);

GGML_BACKEND_API bool ggml_backend_is_edgetpu(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_edgetpu_is_mock(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_edgetpu_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_edgetpu_reg(void);

#ifdef __cplusplus
}
#endif
