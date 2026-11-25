// SPDX-License-Identifier: Apache-2.0
// Clean-room C++17 + ARM64 NEON GGUF K-quant / I-quant unpackers and requantizers
// for libopenbuenos (Q8_0, Q4_0, Q4_K, Q6_K, IQ4_NL, IQ4_XS).
#ifndef OPENBUENOS_QUANT_UNPACK_HPP_
#define OPENBUENOS_QUANT_UNPACK_HPP_

#include "openbuenos.h"

#include <cstddef>
#include <cstdint>

namespace ob {

struct WorkerPool;

namespace quant {

// Exact 16-entry non-linear codebook shared by GGUF IQ4_NL and IQ4_XS.
extern const int8_t kValuesIq4Nl[16];

struct RowNormStats {
  double l2 = 0.0;
  uint64_t l1 = 0ULL;
  uint64_t asum = 0ULL;
};

// Returns true if raw uint8_t value is a valid ob_quant_type_t (0..5).
inline bool is_valid_type(uint8_t raw_type) {
  return raw_type <= static_cast<uint8_t>(OB_QUANT_IQ4_XS);
}

// Compute L2, L1, and |sum| norms of an int8 row (NEON + scalar fallback).
RowNormStats compute_row_norm_stats(const int8_t* row, uint32_t len);

// Portable IEEE-754 binary16 -> binary32 conversion (exact for normals, subnormals, +/-0, Inf/NaN).
float fp16_to_fp32_scalar(uint16_t h);

// Decode 6-bit sub-block scale and min for GGUF Q4_K sub-block j in [0..7].
void get_scale_min_k4(uint32_t j, const uint8_t scales[OB_K_SCALE_SIZE],
                      uint8_t* out_sc, uint8_t* out_m);

// Dequantize one full row [0..k_dim) using the active SIMD/NEON fast path.
void dequant_row_f32(ob_quant_type_t type,
                     const void* row_blocks,
                     uint32_t k_dim,
                     float* dst_f32);

// Portable scalar reference dequantizer for one full row [0..k_dim) (exact vs ggml).
void dequant_row_scalar_f32(ob_quant_type_t type,
                            const void* row_blocks,
                            uint32_t k_dim,
                            float* dst_f32);

// Dequantize an arbitrary sub-slice [k0, k0 + tk) of a quantized row (row_idx)
// from a row-major matrix of quantized blocks with full width k_dim.
void dequant_row_slice_f32(ob_quant_type_t type,
                           const void* b_nk,
                           uint32_t k_dim,
                           uint32_t row_idx,
                           uint32_t k0,
                           uint32_t tk,
                           float* dst_f32);

// Extract a single column `abs_k` across rows [n0, n0 + tn) into dst_col[0..tn).
void extract_column_f32(ob_quant_type_t type,
                        const void* b_nk,
                        uint32_t k_dim,
                        uint32_t n0,
                        uint32_t tn,
                        uint32_t abs_k,
                        float* dst_col);

// Compute max |w| across sub-slice [k0, k0 + tk) of row `row_idx`.
float row_max_abs_f32(ob_quant_type_t type,
                      const void* b_nk,
                      uint32_t k_dim,
                      uint32_t row_idx,
                      uint32_t k0,
                      uint32_t tk);

// Exact FP32 dot product between activation row a[0..k_dim) and quantized weight row.
float dot_row_f32(ob_quant_type_t type,
                  const float* a_row,
                  const void* b_row,
                  uint32_t k_dim);

// Exact FP32 tiled matmul accumulation C[m0..m0+tm, n0..n0+tn] += A[:, k0..k0+tk] @ B[:, k0..k0+tk]^T.
void cpu_matmul_f32_tile(ob_quant_type_t type,
                         const float* a_mk,
                         const void* b_nk,
                         float* c_mn,
                         uint32_t k_dim,
                         uint32_t n_dim,
                         uint32_t m0,
                         uint32_t tm,
                         uint32_t k0,
                         uint32_t tk,
                         uint32_t n0,
                         uint32_t tn);

// Unpack and requantize rows [n0, n0 + tn) over [k0, k0 + tk) into symmetric INT8
// rows `out_bq_base[ni * hw_k]` with per-row scales `out_sb_base[ni]`.
//
// When `out_row_offset != nullptr` and `type == OB_QUANT_Q4_K`:
//   Applies exact algebraic midrange centering (`mu = 0.5 * (w_min + w_max)`) so
//   asymmetric Q4_K `dmin` offsets never waste symmetric INT8 dynamic range.
//   Stores `mu` into `out_row_offset[ni]` and returns true if any row has a non-zero offset.
//   For symmetric formats (`Q8_0`, `Q4_0`, `Q6_K`, `IQ4_NL`, `IQ4_XS`), writes 0.0f
//   (when `out_row_offset != nullptr`) and returns false.
bool requant_rows(ob_quant_type_t type,
                  const void* b_nk,
                  uint32_t k_dim,
                  uint32_t n0,
                  uint32_t tn,
                  uint32_t k0,
                  uint32_t tk,
                  uint32_t hw_k,
                  int8_t* out_bq_base,
                  float* out_sb_base,
                  float* out_row_offset,
                  WorkerPool* pool,
                  RowNormStats* out_stats = nullptr);

}  // namespace quant
}  // namespace ob

#endif  // OPENBUENOS_QUANT_UNPACK_HPP_
