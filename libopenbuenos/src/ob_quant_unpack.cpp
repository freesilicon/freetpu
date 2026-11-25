// SPDX-License-Identifier: Apache-2.0
// Clean-room C++17 + ARM64 NEON GGUF K-quant / I-quant unpackers and requantizers
// for libopenbuenos (Q8_0, Q4_0, Q4_K, Q6_K, IQ4_NL, IQ4_XS).

#include "ob_quant_unpack.hpp"
#include "ob_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ob {
namespace quant {

// Non-linear 4-bit quantization codebook from GGUF IQ4_NL / IQ4_XS.
alignas(16) const int8_t kValuesIq4Nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

static_assert(sizeof(ob_block_q8_0_t) == 34u, "ob_block_q8_0_t must be 34 bytes");
static_assert(sizeof(ob_block_q4_0_t) == 18u, "ob_block_q4_0_t must be 18 bytes");
static_assert(sizeof(ob_block_q4_k_t) == 144u, "ob_block_q4_k_t must be 144 bytes");
static_assert(sizeof(ob_block_q6_k_t) == 210u, "ob_block_q6_k_t must be 210 bytes");
static_assert(sizeof(ob_block_iq4_nl_t) == 18u, "ob_block_iq4_nl_t must be 18 bytes");
static_assert(sizeof(ob_block_iq4_xs_t) == 136u, "ob_block_iq4_xs_t must be 136 bytes");

namespace {

inline uint16_t load_u16_unaligned(const void* p) {
  uint16_t v = 0u;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline float fp16_to_fp32_fast(uint16_t h) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
#else
  return fp16_to_fp32_scalar(h);
#endif
}

inline int8_t quant_sym_i8(float x) {
  long q = std::lround(x);
  if (q < -127) q = -127;
  if (q > 127) q = 127;
  return static_cast<int8_t>(q);
}

inline void quant_sym_i8_row(const float* src, int8_t* dst, uint32_t len, float inv_scale) {
  uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t v_inv = vdupq_n_f32(inv_scale);
  const int32x4_t v_min = vdupq_n_s32(-127);
  const int32x4_t v_max = vdupq_n_s32(127);
  for (; ki + 16u <= len; ki += 16u) {
    int32x4_t q0 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + ki), v_inv)), v_min), v_max);
    int32x4_t q1 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + ki + 4u), v_inv)), v_min), v_max);
    int32x4_t q2 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + ki + 8u), v_inv)), v_min), v_max);
    int32x4_t q3 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + ki + 12u), v_inv)), v_min), v_max);
    int16x8_t s01 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
    int16x8_t s23 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
    vst1q_s8(dst + ki, vcombine_s8(vmovn_s16(s01), vmovn_s16(s23)));
  }
#endif
  for (; ki < len; ++ki) {
    dst[ki] = quant_sym_i8(src[ki] * inv_scale);
  }
}

inline void quant_sym_i8_row_centered(const float* src,
                                      int8_t* dst,
                                      uint32_t len,
                                      float mu,
                                      float inv_scale) {
  uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t v_mu = vdupq_n_f32(mu);
  const float32x4_t v_inv = vdupq_n_f32(inv_scale);
  const int32x4_t v_min = vdupq_n_s32(-127);
  const int32x4_t v_max = vdupq_n_s32(127);
  for (; ki + 16u <= len; ki += 16u) {
    float32x4_t x0 = vmulq_f32(vsubq_f32(vld1q_f32(src + ki),       v_mu), v_inv);
    float32x4_t x1 = vmulq_f32(vsubq_f32(vld1q_f32(src + ki + 4u),  v_mu), v_inv);
    float32x4_t x2 = vmulq_f32(vsubq_f32(vld1q_f32(src + ki + 8u),  v_mu), v_inv);
    float32x4_t x3 = vmulq_f32(vsubq_f32(vld1q_f32(src + ki + 12u), v_mu), v_inv);
    int32x4_t q0 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(x0), v_min), v_max);
    int32x4_t q1 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(x1), v_min), v_max);
    int32x4_t q2 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(x2), v_min), v_max);
    int32x4_t q3 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(x3), v_min), v_max);
    int16x8_t s01 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
    int16x8_t s23 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
    vst1q_s8(dst + ki, vcombine_s8(vmovn_s16(s01), vmovn_s16(s23)));
  }
#endif
  for (; ki < len; ++ki) {
    dst[ki] = quant_sym_i8((src[ki] - mu) * inv_scale);
  }
}

// =========================================================================
// Scalar single-block dequantizers (exact match to ggml-quants.c)
// =========================================================================

void dequant_block_q8_0_scalar(const ob_block_q8_0_t& blk, float* dst) {
  const float d = fp16_to_fp32_scalar(load_u16_unaligned(&blk.d));
  for (uint32_t j = 0u; j < OB_QK8_0; ++j) {
    dst[j] = d * static_cast<float>(blk.qs[j]);
  }
}

void dequant_block_q4_0_scalar(const ob_block_q4_0_t& blk, float* dst) {
  const float d = fp16_to_fp32_scalar(load_u16_unaligned(&blk.d));
  constexpr uint32_t kHalf = OB_QK8_0 / 2u;
  for (uint32_t j = 0u; j < kHalf; ++j) {
    const uint8_t byte = blk.qs[j];
    dst[j] = d * static_cast<float>(static_cast<int32_t>(byte & 0x0Fu) - 8);
    dst[j + kHalf] = d * static_cast<float>(static_cast<int32_t>(byte >> 4) - 8);
  }
}

void dequant_block_q4_k_scalar(const ob_block_q4_k_t& blk, float* dst) {
  const float d = fp16_to_fp32_scalar(load_u16_unaligned(&blk.d));
  const float dmin = fp16_to_fp32_scalar(load_u16_unaligned(&blk.dmin));
  const uint8_t* q = blk.qs;
  uint32_t is = 0u;
  for (uint32_t j = 0u; j < OB_QK_K; j += 64u) {
    uint8_t sc1 = 0u, m1_u = 0u, sc2 = 0u, m2_u = 0u;
    get_scale_min_k4(is + 0u, blk.scales, &sc1, &m1_u);
    get_scale_min_k4(is + 1u, blk.scales, &sc2, &m2_u);
    const float d1 = d * static_cast<float>(sc1);
    const float m1 = dmin * static_cast<float>(m1_u);
    const float d2 = d * static_cast<float>(sc2);
    const float m2 = dmin * static_cast<float>(m2_u);
    for (uint32_t l = 0u; l < 32u; ++l) {
      dst[j + l] = d1 * static_cast<float>(q[l] & 0x0Fu) - m1;
    }
    for (uint32_t l = 0u; l < 32u; ++l) {
      dst[j + 32u + l] = d2 * static_cast<float>(q[l] >> 4) - m2;
    }
    q += 32u;
    is += 2u;
  }
}

void dequant_block_q6_k_scalar(const ob_block_q6_k_t& blk, float* dst) {
  const float d = fp16_to_fp32_scalar(load_u16_unaligned(&blk.d));
  const uint8_t* ql = blk.ql;
  const uint8_t* qh = blk.qh;
  const int8_t* sc = blk.scales;
  float* y = dst;
  for (uint32_t n = 0u; n < OB_QK_K; n += 128u) {
    for (uint32_t l = 0u; l < 32u; ++l) {
      const uint32_t is = l / 16u;
      const int8_t q1 = static_cast<int8_t>(
          static_cast<int32_t>((ql[l + 0u] & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32);
      const int8_t q2 = static_cast<int8_t>(
          static_cast<int32_t>((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32);
      const int8_t q3 = static_cast<int8_t>(
          static_cast<int32_t>((ql[l + 0u] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32);
      const int8_t q4 = static_cast<int8_t>(
          static_cast<int32_t>((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32);
      y[l + 0u]  = (d * static_cast<float>(sc[is + 0u])) * static_cast<float>(q1);
      y[l + 32u] = (d * static_cast<float>(sc[is + 2u])) * static_cast<float>(q2);
      y[l + 64u] = (d * static_cast<float>(sc[is + 4u])) * static_cast<float>(q3);
      y[l + 96u] = (d * static_cast<float>(sc[is + 6u])) * static_cast<float>(q4);
    }
    y += 128u;
    ql += 64u;
    qh += 32u;
    sc += 8u;
  }
}

void dequant_block_iq4_nl_scalar(const ob_block_iq4_nl_t& blk, float* dst) {
  const float d = fp16_to_fp32_scalar(load_u16_unaligned(&blk.d));
  constexpr uint32_t kHalf = OB_QK4_NL / 2u;
  for (uint32_t j = 0u; j < kHalf; ++j) {
    const uint8_t byte = blk.qs[j];
    dst[j] = d * static_cast<float>(kValuesIq4Nl[byte & 0x0Fu]);
    dst[j + kHalf] = d * static_cast<float>(kValuesIq4Nl[byte >> 4]);
  }
}

void dequant_block_iq4_xs_scalar(const ob_block_iq4_xs_t& blk, float* dst) {
  const float d = fp16_to_fp32_scalar(load_u16_unaligned(&blk.d));
  const uint16_t scales_h = load_u16_unaligned(&blk.scales_h);
  const uint8_t* qs = blk.qs;
  float* y = dst;
  for (uint32_t ib = 0u; ib < OB_QK_K / 32u; ++ib) {
    const int32_t ls = static_cast<int32_t>(
        ((blk.scales_l[ib / 2u] >> (4u * (ib & 1u))) & 0x0Fu) |
        (((scales_h >> (2u * ib)) & 3u) << 4));
    const float dl = d * static_cast<float>(ls - 32);
    for (uint32_t j = 0u; j < 16u; ++j) {
      const uint8_t byte = qs[j];
      y[j] = dl * static_cast<float>(kValuesIq4Nl[byte & 0x0Fu]);
      y[j + 16u] = dl * static_cast<float>(kValuesIq4Nl[byte >> 4]);
    }
    y += 32u;
    qs += 16u;
  }
}

// =========================================================================
// ARM64 NEON single-block dequantizers (bit-exact vs scalar)
// =========================================================================

#if defined(__ARM_NEON) && defined(__aarch64__)

inline void store_s8x16_scaled(int8x16_t q, float32x4_t vd, float* dst) {
  int16x8_t q16_lo = vmovl_s8(vget_low_s8(q));
  int16x8_t q16_hi = vmovl_high_s8(q);
  vst1q_f32(dst +  0u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo)))));
  vst1q_f32(dst +  4u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(q16_lo))));
  vst1q_f32(dst +  8u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi)))));
  vst1q_f32(dst + 12u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(q16_hi))));
}

inline void store_u8x16_affine(uint8x16_t q, float32x4_t vd, float32x4_t vm, float* dst) {
  uint16x8_t q16_lo = vmovl_u8(vget_low_u8(q));
  uint16x8_t q16_hi = vmovl_high_u8(q);
  // Explicit mul then sub matches scalar `d * q - m` bit-for-bit.
  vst1q_f32(dst +  0u, vsubq_f32(vmulq_f32(vd, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16_lo)))), vm));
  vst1q_f32(dst +  4u, vsubq_f32(vmulq_f32(vd, vcvtq_f32_u32(vmovl_high_u16(q16_lo))), vm));
  vst1q_f32(dst +  8u, vsubq_f32(vmulq_f32(vd, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16_hi)))), vm));
  vst1q_f32(dst + 12u, vsubq_f32(vmulq_f32(vd, vcvtq_f32_u32(vmovl_high_u16(q16_hi))), vm));
}

void dequant_block_q8_0_neon(const ob_block_q8_0_t& blk, float* dst) {
  const float32x4_t vd = vdupq_n_f32(fp16_to_fp32_fast(load_u16_unaligned(&blk.d)));
  store_s8x16_scaled(vld1q_s8(blk.qs), vd, dst);
  store_s8x16_scaled(vld1q_s8(blk.qs + 16u), vd, dst + 16u);
}

void dequant_block_q4_0_neon(const ob_block_q4_0_t& blk, float* dst) {
  const float32x4_t vd = vdupq_n_f32(fp16_to_fp32_fast(load_u16_unaligned(&blk.d)));
  const uint8x16_t raw = vld1q_u8(blk.qs);
  const int8x16_t v_eight = vdupq_n_s8(8);
  const int8x16_t q_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0Fu))), v_eight);
  const int8x16_t q_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v_eight);
  store_s8x16_scaled(q_lo, vd, dst);
  store_s8x16_scaled(q_hi, vd, dst + 16u);
}

void dequant_block_q4_k_neon(const ob_block_q4_k_t& blk, float* dst) {
  const float d = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
  const float dmin = fp16_to_fp32_fast(load_u16_unaligned(&blk.dmin));
  const uint8x16_t v_mask = vdupq_n_u8(0x0Fu);
  const uint8_t* q = blk.qs;
  uint32_t is = 0u;
  for (uint32_t j = 0u; j < OB_QK_K; j += 64u) {
    uint8_t sc1 = 0u, m1_u = 0u, sc2 = 0u, m2_u = 0u;
    get_scale_min_k4(is + 0u, blk.scales, &sc1, &m1_u);
    get_scale_min_k4(is + 1u, blk.scales, &sc2, &m2_u);
    const float32x4_t vd1 = vdupq_n_f32(d * static_cast<float>(sc1));
    const float32x4_t vm1 = vdupq_n_f32(dmin * static_cast<float>(m1_u));
    const float32x4_t vd2 = vdupq_n_f32(d * static_cast<float>(sc2));
    const float32x4_t vm2 = vdupq_n_f32(dmin * static_cast<float>(m2_u));

    const uint8x16_t raw0 = vld1q_u8(q);
    const uint8x16_t raw1 = vld1q_u8(q + 16u);
    store_u8x16_affine(vandq_u8(raw0, v_mask), vd1, vm1, dst + j);
    store_u8x16_affine(vandq_u8(raw1, v_mask), vd1, vm1, dst + j + 16u);
    store_u8x16_affine(vshrq_n_u8(raw0, 4), vd2, vm2, dst + j + 32u);
    store_u8x16_affine(vshrq_n_u8(raw1, 4), vd2, vm2, dst + j + 48u);

    q += 32u;
    is += 2u;
  }
}

void dequant_block_q6_k_neon(const ob_block_q6_k_t& blk, float* dst) {
  const float d = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
  const uint8x16_t v_0f = vdupq_n_u8(0x0Fu);
  const uint8x16_t v_03 = vdupq_n_u8(0x03u);
  const int8x16_t v_32 = vdupq_n_s8(32);

  const uint8_t* ql = blk.ql;
  const uint8_t* qh = blk.qh;
  const int8_t* sc = blk.scales;
  float* y = dst;

  for (uint32_t n = 0u; n < OB_QK_K; n += 128u) {
    for (uint32_t is = 0u; is < 2u; ++is) {
      const uint32_t loff = is * 16u;
      const uint8x16_t ql0 = vld1q_u8(ql + loff);
      const uint8x16_t ql1 = vld1q_u8(ql + 32u + loff);
      const uint8x16_t qhv = vld1q_u8(qh + loff);

      const int8x16_t q1 = vsubq_s8(
          vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, v_0f), vshlq_n_u8(vandq_u8(qhv, v_03), 4))),
          v_32);
      const int8x16_t q2 = vsubq_s8(
          vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, v_0f),
                                       vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), v_03), 4))),
          v_32);
      const int8x16_t q3 = vsubq_s8(
          vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4),
                                       vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), v_03), 4))),
          v_32);
      const int8x16_t q4 = vsubq_s8(
          vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), vshlq_n_u8(vshrq_n_u8(qhv, 6), 4))),
          v_32);

      store_s8x16_scaled(q1, vdupq_n_f32(d * static_cast<float>(sc[is + 0u])), y + loff);
      store_s8x16_scaled(q2, vdupq_n_f32(d * static_cast<float>(sc[is + 2u])), y + loff + 32u);
      store_s8x16_scaled(q3, vdupq_n_f32(d * static_cast<float>(sc[is + 4u])), y + loff + 64u);
      store_s8x16_scaled(q4, vdupq_n_f32(d * static_cast<float>(sc[is + 6u])), y + loff + 96u);
    }
    y += 128u;
    ql += 64u;
    qh += 32u;
    sc += 8u;
  }
}

void dequant_block_iq4_nl_neon(const ob_block_iq4_nl_t& blk, float* dst) {
  const int8x16_t v_lut = vld1q_s8(kValuesIq4Nl);
  const float32x4_t vd = vdupq_n_f32(fp16_to_fp32_fast(load_u16_unaligned(&blk.d)));
  const uint8x16_t raw = vld1q_u8(blk.qs);
  const int8x16_t q_lo = vqtbl1q_s8(v_lut, vandq_u8(raw, vdupq_n_u8(0x0Fu)));
  const int8x16_t q_hi = vqtbl1q_s8(v_lut, vshrq_n_u8(raw, 4));
  store_s8x16_scaled(q_lo, vd, dst);
  store_s8x16_scaled(q_hi, vd, dst + 16u);
}

void dequant_block_iq4_xs_neon(const ob_block_iq4_xs_t& blk, float* dst) {
  const int8x16_t v_lut = vld1q_s8(kValuesIq4Nl);
  const uint8x16_t v_mask = vdupq_n_u8(0x0Fu);
  const float d = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
  const uint16_t scales_h = load_u16_unaligned(&blk.scales_h);
  const uint8_t* qs = blk.qs;
  float* y = dst;
  for (uint32_t ib = 0u; ib < OB_QK_K / 32u; ++ib) {
    const int32_t ls = static_cast<int32_t>(
        ((blk.scales_l[ib / 2u] >> (4u * (ib & 1u))) & 0x0Fu) |
        (((scales_h >> (2u * ib)) & 3u) << 4));
    const float32x4_t vdl = vdupq_n_f32(d * static_cast<float>(ls - 32));
    const uint8x16_t raw = vld1q_u8(qs);
    const int8x16_t q_lo = vqtbl1q_s8(v_lut, vandq_u8(raw, v_mask));
    const int8x16_t q_hi = vqtbl1q_s8(v_lut, vshrq_n_u8(raw, 4));
    store_s8x16_scaled(q_lo, vdl, y);
    store_s8x16_scaled(q_hi, vdl, y + 16u);
    y += 32u;
    qs += 16u;
  }
}

#endif  // __ARM_NEON && __aarch64__

void dequant_block_fast(ob_quant_type_t type, const uint8_t* blk_ptr, float* dst) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  switch (type) {
    case OB_QUANT_Q8_0:
      dequant_block_q8_0_neon(*reinterpret_cast<const ob_block_q8_0_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_Q4_0:
      dequant_block_q4_0_neon(*reinterpret_cast<const ob_block_q4_0_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_Q4_K:
      dequant_block_q4_k_neon(*reinterpret_cast<const ob_block_q4_k_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_Q6_K:
      dequant_block_q6_k_neon(*reinterpret_cast<const ob_block_q6_k_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_IQ4_NL:
      dequant_block_iq4_nl_neon(*reinterpret_cast<const ob_block_iq4_nl_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_IQ4_XS:
      dequant_block_iq4_xs_neon(*reinterpret_cast<const ob_block_iq4_xs_t*>(blk_ptr), dst);
      return;
  }
#else
  switch (type) {
    case OB_QUANT_Q8_0:
      dequant_block_q8_0_scalar(*reinterpret_cast<const ob_block_q8_0_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_Q4_0:
      dequant_block_q4_0_scalar(*reinterpret_cast<const ob_block_q4_0_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_Q4_K:
      dequant_block_q4_k_scalar(*reinterpret_cast<const ob_block_q4_k_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_Q6_K:
      dequant_block_q6_k_scalar(*reinterpret_cast<const ob_block_q6_k_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_IQ4_NL:
      dequant_block_iq4_nl_scalar(*reinterpret_cast<const ob_block_iq4_nl_t*>(blk_ptr), dst);
      return;
    case OB_QUANT_IQ4_XS:
      dequant_block_iq4_xs_scalar(*reinterpret_cast<const ob_block_iq4_xs_t*>(blk_ptr), dst);
      return;
  }
#endif
}

// Extract a single element `rem` (0 <= rem < qk) from one quantized block.
inline float extract_block_elem(ob_quant_type_t type, const uint8_t* blk_ptr, uint32_t rem) {
  switch (type) {
    case OB_QUANT_Q8_0: {
      const auto& blk = *reinterpret_cast<const ob_block_q8_0_t*>(blk_ptr);
      return fp16_to_fp32_fast(load_u16_unaligned(&blk.d)) * static_cast<float>(blk.qs[rem]);
    }
    case OB_QUANT_Q4_0: {
      const auto& blk = *reinterpret_cast<const ob_block_q4_0_t*>(blk_ptr);
      const float d = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
      const uint8_t byte = blk.qs[rem & 15u];
      const int32_t q = static_cast<int32_t>((rem < 16u) ? (byte & 0x0Fu) : (byte >> 4)) - 8;
      return d * static_cast<float>(q);
    }
    case OB_QUANT_Q4_K: {
      const auto& blk = *reinterpret_cast<const ob_block_q4_k_t*>(blk_ptr);
      const float d = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
      const float dmin = fp16_to_fp32_fast(load_u16_unaligned(&blk.dmin));
      const uint32_t g = rem >> 6;
      const uint32_t in64 = rem & 63u;
      const uint32_t sub = (g << 1) + (in64 >= 32u ? 1u : 0u);
      uint8_t sc = 0u, m = 0u;
      get_scale_min_k4(sub, blk.scales, &sc, &m);
      const uint8_t byte = blk.qs[g * 32u + (in64 & 31u)];
      const uint32_t qval = (in64 < 32u) ? (byte & 0x0Fu) : (byte >> 4);
      return (d * static_cast<float>(sc)) * static_cast<float>(qval) -
             (dmin * static_cast<float>(m));
    }
    case OB_QUANT_Q6_K: {
      const auto& blk = *reinterpret_cast<const ob_block_q6_k_t*>(blk_ptr);
      const float d = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
      const uint32_t half = rem >> 7;
      const uint32_t in128 = rem & 127u;
      const uint32_t quad = in128 >> 5;
      const uint32_t l = in128 & 31u;
      const uint32_t is = l >> 4;
      const uint8_t* ql = blk.ql + half * 64u;
      const uint8_t* qh = blk.qh + half * 32u;
      const int8_t* sc = blk.scales + half * 8u;
      uint32_t low4 = 0u;
      if (quad == 0u) low4 = ql[l] & 0x0Fu;
      else if (quad == 1u) low4 = ql[l + 32u] & 0x0Fu;
      else if (quad == 2u) low4 = ql[l] >> 4;
      else low4 = ql[l + 32u] >> 4;
      const uint32_t hi2 = (qh[l] >> (quad * 2u)) & 3u;
      const int32_t q = static_cast<int32_t>(low4 | (hi2 << 4)) - 32;
      return (d * static_cast<float>(sc[is + quad * 2u])) * static_cast<float>(q);
    }
    case OB_QUANT_IQ4_NL: {
      const auto& blk = *reinterpret_cast<const ob_block_iq4_nl_t*>(blk_ptr);
      const float d = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
      const uint8_t byte = blk.qs[rem & 15u];
      const uint32_t idx = (rem < 16u) ? (byte & 0x0Fu) : (byte >> 4);
      return d * static_cast<float>(kValuesIq4Nl[idx]);
    }
    case OB_QUANT_IQ4_XS: {
      const auto& blk = *reinterpret_cast<const ob_block_iq4_xs_t*>(blk_ptr);
      const float d = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
      const uint16_t scales_h = load_u16_unaligned(&blk.scales_h);
      const uint32_t ib = rem >> 5;
      const uint32_t in32 = rem & 31u;
      const int32_t ls = static_cast<int32_t>(
          ((blk.scales_l[ib / 2u] >> (4u * (ib & 1u))) & 0x0Fu) |
          (((scales_h >> (2u * ib)) & 3u) << 4));
      const float dl = d * static_cast<float>(ls - 32);
      const uint8_t byte = blk.qs[ib * 16u + (in32 & 15u)];
      const uint32_t idx = (in32 < 16u) ? (byte & 0x0Fu) : (byte >> 4);
      return dl * static_cast<float>(kValuesIq4Nl[idx]);
    }
  }
  return 0.0f;
}

// =========================================================================
// Fast dot-product helpers for CPU fallback and M=1 SwiGLU outlier path
// =========================================================================

inline float dot_f32_row(const float* a, const float* w, uint32_t len) {
  uint32_t i = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  for (; i + 16u <= len; i += 16u) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i +  0u), vld1q_f32(w + i +  0u));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i +  4u), vld1q_f32(w + i +  4u));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i +  8u), vld1q_f32(w + i +  8u));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12u), vld1q_f32(w + i + 12u));
  }
  for (; i + 4u <= len; i += 4u) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(w + i));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; i < len; ++i) sum += a[i] * w[i];
  return sum;
#else
  double sum = 0.0;
  for (; i < len; ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(w[i]);
  }
  return static_cast<float>(sum);
#endif
}

// Algebraic dmin-factored Q4_K dot product:
//   sum(a_i * (d1 * q_i - m1)) = d1 * sum(a_i * q_i) - m1 * sum(a_i)
float dot_q4_k_blocks(const float* a, const ob_block_q4_k_t* blocks, uint32_t nb) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const uint8x16_t v_mask = vdupq_n_u8(0x0Fu);
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const float d = fp16_to_fp32_fast(load_u16_unaligned(&blocks[bi].d));
    const float dmin = fp16_to_fp32_fast(load_u16_unaligned(&blocks[bi].dmin));
    const uint8_t* q = blocks[bi].qs;
    const float* ap = a + static_cast<size_t>(bi) * OB_QK_K;
    uint32_t is = 0u;
    for (uint32_t j = 0u; j < OB_QK_K; j += 64u) {
      uint8_t sc1 = 0u, m1_u = 0u, sc2 = 0u, m2_u = 0u;
      get_scale_min_k4(is + 0u, blocks[bi].scales, &sc1, &m1_u);
      get_scale_min_k4(is + 1u, blocks[bi].scales, &sc2, &m2_u);
      const float32x4_t vd1 = vdupq_n_f32(d * static_cast<float>(sc1));
      const float32x4_t vm1 = vdupq_n_f32(dmin * static_cast<float>(m1_u));
      const float32x4_t vd2 = vdupq_n_f32(d * static_cast<float>(sc2));
      const float32x4_t vm2 = vdupq_n_f32(dmin * static_cast<float>(m2_u));

      const uint8x16_t raw0 = vld1q_u8(q);
      const uint8x16_t raw1 = vld1q_u8(q + 16u);

      // Sub-block 1 (low nibbles, 32 weights)
      {
        const uint8x16_t lo0 = vandq_u8(raw0, v_mask);
        const uint8x16_t lo1 = vandq_u8(raw1, v_mask);
        const uint16x8_t u0_0 = vmovl_u8(vget_low_u8(lo0));
        const uint16x8_t u0_1 = vmovl_high_u8(lo0);
        const uint16x8_t u1_0 = vmovl_u8(vget_low_u8(lo1));
        const uint16x8_t u1_1 = vmovl_high_u8(lo1);

        const float32x4_t a0 = vld1q_f32(ap + j +  0u);
        const float32x4_t a1 = vld1q_f32(ap + j +  4u);
        const float32x4_t a2 = vld1q_f32(ap + j +  8u);
        const float32x4_t a3 = vld1q_f32(ap + j + 12u);
        const float32x4_t a4 = vld1q_f32(ap + j + 16u);
        const float32x4_t a5 = vld1q_f32(ap + j + 20u);
        const float32x4_t a6 = vld1q_f32(ap + j + 24u);
        const float32x4_t a7 = vld1q_f32(ap + j + 28u);

        float32x4_t sa0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        float32x4_t sa1 = vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7));
        float32x4_t aq0 = vmulq_f32(a0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0_0))));
        float32x4_t aq1 = vmulq_f32(a1, vcvtq_f32_u32(vmovl_high_u16(u0_0)));
        aq0 = vfmaq_f32(aq0, a2, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0_1))));
        aq1 = vfmaq_f32(aq1, a3, vcvtq_f32_u32(vmovl_high_u16(u0_1)));
        aq0 = vfmaq_f32(aq0, a4, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1_0))));
        aq1 = vfmaq_f32(aq1, a5, vcvtq_f32_u32(vmovl_high_u16(u1_0)));
        aq0 = vfmaq_f32(aq0, a6, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1_1))));
        aq1 = vfmaq_f32(aq1, a7, vcvtq_f32_u32(vmovl_high_u16(u1_1)));

        acc0 = vfmsq_f32(vfmaq_f32(acc0, aq0, vd1), sa0, vm1);
        acc1 = vfmsq_f32(vfmaq_f32(acc1, aq1, vd1), sa1, vm1);
      }
      // Sub-block 2 (high nibbles, 32 weights)
      {
        const uint8x16_t hi0 = vshrq_n_u8(raw0, 4);
        const uint8x16_t hi1 = vshrq_n_u8(raw1, 4);
        const uint16x8_t u0_0 = vmovl_u8(vget_low_u8(hi0));
        const uint16x8_t u0_1 = vmovl_high_u8(hi0);
        const uint16x8_t u1_0 = vmovl_u8(vget_low_u8(hi1));
        const uint16x8_t u1_1 = vmovl_high_u8(hi1);

        const float32x4_t a0 = vld1q_f32(ap + j + 32u);
        const float32x4_t a1 = vld1q_f32(ap + j + 36u);
        const float32x4_t a2 = vld1q_f32(ap + j + 40u);
        const float32x4_t a3 = vld1q_f32(ap + j + 44u);
        const float32x4_t a4 = vld1q_f32(ap + j + 48u);
        const float32x4_t a5 = vld1q_f32(ap + j + 52u);
        const float32x4_t a6 = vld1q_f32(ap + j + 56u);
        const float32x4_t a7 = vld1q_f32(ap + j + 60u);

        float32x4_t sa0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        float32x4_t sa1 = vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7));
        float32x4_t aq0 = vmulq_f32(a0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0_0))));
        float32x4_t aq1 = vmulq_f32(a1, vcvtq_f32_u32(vmovl_high_u16(u0_0)));
        aq0 = vfmaq_f32(aq0, a2, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0_1))));
        aq1 = vfmaq_f32(aq1, a3, vcvtq_f32_u32(vmovl_high_u16(u0_1)));
        aq0 = vfmaq_f32(aq0, a4, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1_0))));
        aq1 = vfmaq_f32(aq1, a5, vcvtq_f32_u32(vmovl_high_u16(u1_0)));
        aq0 = vfmaq_f32(aq0, a6, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1_1))));
        aq1 = vfmaq_f32(aq1, a7, vcvtq_f32_u32(vmovl_high_u16(u1_1)));

        acc0 = vfmsq_f32(vfmaq_f32(acc0, aq0, vd2), sa0, vm2);
        acc1 = vfmsq_f32(vfmaq_f32(acc1, aq1, vd2), sa1, vm2);
      }
      q += 32u;
      is += 2u;
    }
  }
  return vaddvq_f32(vaddq_f32(acc0, acc1));
#else
  double sum = 0.0;
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const double d = static_cast<double>(fp16_to_fp32_scalar(load_u16_unaligned(&blocks[bi].d)));
    const double dmin = static_cast<double>(fp16_to_fp32_scalar(load_u16_unaligned(&blocks[bi].dmin)));
    const uint8_t* q = blocks[bi].qs;
    const float* ap = a + static_cast<size_t>(bi) * OB_QK_K;
    uint32_t is = 0u;
    for (uint32_t j = 0u; j < OB_QK_K; j += 64u) {
      uint8_t sc1 = 0u, m1_u = 0u, sc2 = 0u, m2_u = 0u;
      get_scale_min_k4(is + 0u, blocks[bi].scales, &sc1, &m1_u);
      get_scale_min_k4(is + 1u, blocks[bi].scales, &sc2, &m2_u);
      double aq1 = 0.0, sa1 = 0.0;
      double aq2 = 0.0, sa2 = 0.0;
      for (uint32_t l = 0u; l < 32u; ++l) {
        const double al1 = static_cast<double>(ap[j + l]);
        const double al2 = static_cast<double>(ap[j + 32u + l]);
        sa1 += al1;
        aq1 += al1 * static_cast<double>(q[l] & 0x0Fu);
        sa2 += al2;
        aq2 += al2 * static_cast<double>(q[l] >> 4);
      }
      sum += (d * static_cast<double>(sc1)) * aq1 - (dmin * static_cast<double>(m1_u)) * sa1;
      sum += (d * static_cast<double>(sc2)) * aq2 - (dmin * static_cast<double>(m2_u)) * sa2;
      q += 32u;
      is += 2u;
    }
  }
  return static_cast<float>(sum);
#endif
}

float dot_iq4_nl_blocks(const float* a, const ob_block_iq4_nl_t* blocks, uint32_t nb) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const int8x16_t v_lut = vld1q_s8(kValuesIq4Nl);
  const uint8x16_t v_mask = vdupq_n_u8(0x0Fu);
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const float32x4_t vd = vdupq_n_f32(fp16_to_fp32_fast(load_u16_unaligned(&blocks[bi].d)));
    const float* ap = a + static_cast<size_t>(bi) * OB_QK4_NL;
    const uint8x16_t raw = vld1q_u8(blocks[bi].qs);
    const int8x16_t lo8 = vqtbl1q_s8(v_lut, vandq_u8(raw, v_mask));
    const int8x16_t hi8 = vqtbl1q_s8(v_lut, vshrq_n_u8(raw, 4));
    const int16x8_t lo16_0 = vmovl_s8(vget_low_s8(lo8));
    const int16x8_t lo16_1 = vmovl_high_s8(lo8);
    const int16x8_t hi16_0 = vmovl_s8(vget_low_s8(hi8));
    const int16x8_t hi16_1 = vmovl_high_s8(hi8);
    float32x4_t b0 = vmulq_f32(vld1q_f32(ap +  0u), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0))));
    float32x4_t b1 = vmulq_f32(vld1q_f32(ap +  4u), vcvtq_f32_s32(vmovl_high_s16(lo16_0)));
    b0 = vfmaq_f32(b0, vld1q_f32(ap +  8u), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1))));
    b1 = vfmaq_f32(b1, vld1q_f32(ap + 12u), vcvtq_f32_s32(vmovl_high_s16(lo16_1)));
    b0 = vfmaq_f32(b0, vld1q_f32(ap + 16u), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0))));
    b1 = vfmaq_f32(b1, vld1q_f32(ap + 20u), vcvtq_f32_s32(vmovl_high_s16(hi16_0)));
    b0 = vfmaq_f32(b0, vld1q_f32(ap + 24u), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1))));
    b1 = vfmaq_f32(b1, vld1q_f32(ap + 28u), vcvtq_f32_s32(vmovl_high_s16(hi16_1)));
    acc0 = vfmaq_f32(acc0, b0, vd);
    acc1 = vfmaq_f32(acc1, b1, vd);
  }
  return vaddvq_f32(vaddq_f32(acc0, acc1));
#else
  double sum = 0.0;
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const double d = static_cast<double>(fp16_to_fp32_scalar(load_u16_unaligned(&blocks[bi].d)));
    const uint8_t* qs = blocks[bi].qs;
    const float* ap = a + static_cast<size_t>(bi) * OB_QK4_NL;
    double bsum = 0.0;
    for (uint32_t j = 0u; j < 16u; ++j) {
      const uint8_t byte = qs[j];
      bsum += static_cast<double>(ap[j]) * static_cast<double>(kValuesIq4Nl[byte & 0x0Fu]);
      bsum += static_cast<double>(ap[j + 16u]) * static_cast<double>(kValuesIq4Nl[byte >> 4]);
    }
    sum += d * bsum;
  }
  return static_cast<float>(sum);
#endif
}

// =========================================================================
// Requantization helper range (supports all 6 formats + Q4_K dmin centering)
// =========================================================================

void requant_rows_range(ob_quant_type_t type,
                        const void* b_nk,
                        uint32_t k_dim,
                        uint32_t n0,
                        uint32_t ni_start,
                        uint32_t ni_end,
                        uint32_t k0,
                        uint32_t tk,
                        uint32_t hw_k,
                        int8_t* out_bq_base,
                        float* out_sb_base,
                        float* out_row_offset,
                        RowNormStats* out_stats,
                        std::atomic<bool>* any_nonzero_offset) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  const uint32_t nb_k = k_dim / qk;

  // Zero-conversion fast path for single 32-element block symmetric formats (Q8_0, Q4_0, IQ4_NL).
  if (tk == 32u && (k0 % 32u) == 0u &&
      (type == OB_QUANT_Q8_0 || type == OB_QUANT_Q4_0 || type == OB_QUANT_IQ4_NL)) {
    const uint32_t blk0 = k0 / 32u;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const uint8x16_t v_mask = vdupq_n_u8(0x0Fu);
    const int8x16_t v_eight = vdupq_n_s8(8);
    const int8x16_t v_lut = vld1q_s8(kValuesIq4Nl);
#endif
    for (uint32_t ni = ni_start; ni < ni_end; ++ni) {
      const uint8_t* bp = static_cast<const uint8_t*>(b_nk) +
                          (static_cast<size_t>(n0 + ni) * nb_k + blk0) * blk_sz;
      int8_t* bq_row = out_bq_base + static_cast<size_t>(ni) * hw_k;
      if (type == OB_QUANT_Q8_0) {
        const auto& blk = *reinterpret_cast<const ob_block_q8_0_t*>(bp);
        out_sb_base[ni] = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
        std::memcpy(bq_row, blk.qs, 32u);
      } else if (type == OB_QUANT_Q4_0) {
        const auto& blk = *reinterpret_cast<const ob_block_q4_0_t*>(bp);
        out_sb_base[ni] = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
#if defined(__ARM_NEON) && defined(__aarch64__)
        const uint8x16_t raw = vld1q_u8(blk.qs);
        vst1q_s8(bq_row,       vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, v_mask)), v_eight));
        vst1q_s8(bq_row + 16u, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v_eight));
#else
        for (uint32_t j = 0u; j < 16u; ++j) {
          const uint8_t byte = blk.qs[j];
          bq_row[j] = static_cast<int8_t>(static_cast<int32_t>(byte & 0x0Fu) - 8);
          bq_row[j + 16u] = static_cast<int8_t>(static_cast<int32_t>(byte >> 4) - 8);
        }
#endif
      } else {
        const auto& blk = *reinterpret_cast<const ob_block_iq4_nl_t*>(bp);
        out_sb_base[ni] = fp16_to_fp32_fast(load_u16_unaligned(&blk.d));
#if defined(__ARM_NEON) && defined(__aarch64__)
        const uint8x16_t raw = vld1q_u8(blk.qs);
        vst1q_s8(bq_row,       vqtbl1q_s8(v_lut, vandq_u8(raw, v_mask)));
        vst1q_s8(bq_row + 16u, vqtbl1q_s8(v_lut, vshrq_n_u8(raw, 4)));
#else
        for (uint32_t j = 0u; j < 16u; ++j) {
          const uint8_t byte = blk.qs[j];
          bq_row[j] = kValuesIq4Nl[byte & 0x0Fu];
          bq_row[j + 16u] = kValuesIq4Nl[byte >> 4];
        }
#endif
      }
      if (out_row_offset) out_row_offset[ni] = 0.0f;
      if (tk < hw_k) std::memset(bq_row + tk, 0, hw_k - tk);
      if (out_stats) out_stats[ni] = compute_row_norm_stats(bq_row, tk);
    }
    return;
  }

  float stack_wrow[2048];
  std::vector<float> heap_wrow;
  float* wrow = stack_wrow;
  if (tk > 2048u) {
    heap_wrow.resize(tk);
    wrow = heap_wrow.data();
  }

  const bool center_q4_k = (type == OB_QUANT_Q4_K && out_row_offset != nullptr);
  bool local_nonzero_offset = false;

  for (uint32_t ni = ni_start; ni < ni_end; ++ni) {
    dequant_row_slice_f32(type, b_nk, k_dim, n0 + ni, k0, tk, wrow);
    int8_t* bq_row = out_bq_base + static_cast<size_t>(ni) * hw_k;

    if (center_q4_k) {
      float w_min = wrow[0];
      float w_max = wrow[0];
      uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
      if (tk >= 4u) {
        float32x4_t vmin0 = vld1q_f32(wrow);
        float32x4_t vmax0 = vmin0;
        ki = 4u;
        for (; ki + 16u <= tk; ki += 16u) {
          float32x4_t v0 = vld1q_f32(wrow + ki);
          float32x4_t v1 = vld1q_f32(wrow + ki + 4u);
          float32x4_t v2 = vld1q_f32(wrow + ki + 8u);
          float32x4_t v3 = vld1q_f32(wrow + ki + 12u);
          vmin0 = vminq_f32(vmin0, vminq_f32(vminq_f32(v0, v1), vminq_f32(v2, v3)));
          vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        }
        for (; ki + 4u <= tk; ki += 4u) {
          float32x4_t v0 = vld1q_f32(wrow + ki);
          vmin0 = vminq_f32(vmin0, v0);
          vmax0 = vmaxq_f32(vmax0, v0);
        }
        w_min = vminvq_f32(vmin0);
        w_max = vmaxvq_f32(vmax0);
      }
#endif
      for (; ki < tk; ++ki) {
        if (wrow[ki] < w_min) w_min = wrow[ki];
        if (wrow[ki] > w_max) w_max = wrow[ki];
      }

      const float raw_amax = std::max(std::fabs(w_min), std::fabs(w_max));
      if (raw_amax == 0.0f) {
        out_sb_base[ni] = 0.0f;
        out_row_offset[ni] = 0.0f;
        std::memset(bq_row, 0, hw_k);
        if (out_stats) out_stats[ni] = RowNormStats{};
        continue;
      }

      // Exact algebraic midrange centering: w_k = (w_k - mu) + mu.
      // Guarantees centered weights use 100% of symmetric [-127, +127] INT8 range
      // regardless of asymmetric Q4_K `dmin * m_j` shifts!
      const float mu = 0.5f * (w_min + w_max);
      const float half_range = 0.5f * (w_max - w_min);
      if (half_range <= 1e-12f * raw_amax) {
        // Constant non-zero row (e.g. uniform `-dmin * m`): exact via offset term alone!
        out_sb_base[ni] = 0.0f;
        out_row_offset[ni] = mu;
        local_nonzero_offset = true;
        std::memset(bq_row, 0, hw_k);
        if (out_stats) out_stats[ni] = RowNormStats{};
        continue;
      }

      if (std::fabs(mu) > 1e-6f * raw_amax) {
        const float sb = half_range / 127.0f;
        out_sb_base[ni] = sb;
        out_row_offset[ni] = mu;
        local_nonzero_offset = true;
        quant_sym_i8_row_centered(wrow, bq_row, tk, mu, 127.0f / half_range);
      } else {
        const float sb = raw_amax / 127.0f;
        out_sb_base[ni] = sb;
        out_row_offset[ni] = 0.0f;
        quant_sym_i8_row(wrow, bq_row, tk, 127.0f / raw_amax);
      }
      if (tk < hw_k) std::memset(bq_row + tk, 0, hw_k - tk);
      if (out_stats) out_stats[ni] = compute_row_norm_stats(bq_row, tk);
    } else {
      float amax = 0.0f;
      uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
      float32x4_t v_am0 = vdupq_n_f32(0.0f);
      float32x4_t v_am1 = vdupq_n_f32(0.0f);
      for (; ki + 8u <= tk; ki += 8u) {
        v_am0 = vmaxq_f32(v_am0, vabsq_f32(vld1q_f32(wrow + ki)));
        v_am1 = vmaxq_f32(v_am1, vabsq_f32(vld1q_f32(wrow + ki + 4u)));
      }
      for (; ki + 4u <= tk; ki += 4u) {
        v_am0 = vmaxq_f32(v_am0, vabsq_f32(vld1q_f32(wrow + ki)));
      }
      amax = vmaxvq_f32(vmaxq_f32(v_am0, v_am1));
#endif
      for (; ki < tk; ++ki) {
        const float av = std::fabs(wrow[ki]);
        if (av > amax) amax = av;
      }

      if (out_row_offset) out_row_offset[ni] = 0.0f;
      if (amax == 0.0f) {
        out_sb_base[ni] = 0.0f;
        std::memset(bq_row, 0, hw_k);
        if (out_stats) out_stats[ni] = RowNormStats{};
        continue;
      }
      const float sb = amax / 127.0f;
      out_sb_base[ni] = sb;
      quant_sym_i8_row(wrow, bq_row, tk, 127.0f / amax);
      if (tk < hw_k) std::memset(bq_row + tk, 0, hw_k - tk);
      if (out_stats) out_stats[ni] = compute_row_norm_stats(bq_row, tk);
    }
  }

  if (local_nonzero_offset && any_nonzero_offset) {
    any_nonzero_offset->store(true, std::memory_order_relaxed);
  }
}

}  // namespace

float fp16_to_fp32_scalar(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = static_cast<uint32_t>(h >> 10) & 0x1Fu;
  uint32_t mant = static_cast<uint32_t>(h & 0x03FFu);
  uint32_t bits = 0u;

  if (exp == 0u) {
    if (mant == 0u) {
      bits = sign;
    } else {
      exp = 1u;
      while ((mant & 0x0400u) == 0u) {
        mant <<= 1u;
        --exp;
      }
      mant &= 0x03FFu;
      const uint32_t fexp = exp + (127u - 15u);
      bits = sign | (fexp << 23) | (mant << 13);
    }
  } else if (exp == 31u) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    const uint32_t fexp = exp + (127u - 15u);
    bits = sign | (fexp << 23) | (mant << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

void get_scale_min_k4(uint32_t j, const uint8_t scales[OB_K_SCALE_SIZE],
                      uint8_t* out_sc, uint8_t* out_m) {
  if (j < 4u) {
    *out_sc = static_cast<uint8_t>(scales[j] & 63u);
    *out_m  = static_cast<uint8_t>(scales[j + 4u] & 63u);
  } else {
    *out_sc = static_cast<uint8_t>((scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] >> 6) << 4));
    *out_m  = static_cast<uint8_t>((scales[j + 4u] >> 4) | ((scales[j] >> 6) << 4));
  }
}

RowNormStats compute_row_norm_stats(const int8_t* row, uint32_t len) {
  uint64_t sq = 0ULL;
  uint64_t l1 = 0ULL;
  int64_t ssum = 0LL;
  uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  uint32x4_t v_sq = vdupq_n_u32(0u);
  uint32x4_t v_l1 = vdupq_n_u32(0u);
  int32x4_t v_ss = vdupq_n_s32(0);
  for (; ki + 16u <= len; ki += 16u) {
    int8x16_t v = vld1q_s8(row + ki);
    int16x8_t lo = vmull_s8(vget_low_s8(v), vget_low_s8(v));
    int16x8_t hi = vmull_high_s8(v, v);
    v_sq = vpadalq_u16(v_sq, vreinterpretq_u16_s16(lo));
    v_sq = vpadalq_u16(v_sq, vreinterpretq_u16_s16(hi));
    uint8x16_t av = vreinterpretq_u8_s8(vabsq_s8(v));
    v_l1 = vpadalq_u16(v_l1, vpaddlq_u8(av));
    v_ss = vpadalq_s16(v_ss, vpaddlq_s8(v));
  }
  sq = static_cast<uint64_t>(vaddvq_u32(v_sq));
  l1 = static_cast<uint64_t>(vaddvq_u32(v_l1));
  ssum = static_cast<int64_t>(vaddvq_s32(v_ss));
#endif
  for (; ki < len; ++ki) {
    const int32_t v = static_cast<int32_t>(row[ki]);
    sq += static_cast<uint64_t>(v * v);
    l1 += static_cast<uint64_t>(v < 0 ? -v : v);
    ssum += static_cast<int64_t>(v);
  }
  RowNormStats st{};
  st.l2 = std::sqrt(static_cast<double>(sq));
  st.l1 = l1;
  st.asum = static_cast<uint64_t>(ssum < 0 ? -ssum : ssum);
  return st;
}

void dequant_row_scalar_f32(ob_quant_type_t type,
                            const void* row_blocks,
                            uint32_t k_dim,
                            float* dst_f32) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  if (qk == 0u || blk_sz == 0u || !row_blocks || !dst_f32 || (k_dim % qk) != 0u) {
    return;
  }
  const uint32_t nb = k_dim / qk;
  const uint8_t* p = static_cast<const uint8_t*>(row_blocks);
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const uint8_t* bp = p + static_cast<size_t>(bi) * blk_sz;
    float* out = dst_f32 + static_cast<size_t>(bi) * qk;
    switch (type) {
      case OB_QUANT_Q8_0:
        dequant_block_q8_0_scalar(*reinterpret_cast<const ob_block_q8_0_t*>(bp), out);
        break;
      case OB_QUANT_Q4_0:
        dequant_block_q4_0_scalar(*reinterpret_cast<const ob_block_q4_0_t*>(bp), out);
        break;
      case OB_QUANT_Q4_K:
        dequant_block_q4_k_scalar(*reinterpret_cast<const ob_block_q4_k_t*>(bp), out);
        break;
      case OB_QUANT_Q6_K:
        dequant_block_q6_k_scalar(*reinterpret_cast<const ob_block_q6_k_t*>(bp), out);
        break;
      case OB_QUANT_IQ4_NL:
        dequant_block_iq4_nl_scalar(*reinterpret_cast<const ob_block_iq4_nl_t*>(bp), out);
        break;
      case OB_QUANT_IQ4_XS:
        dequant_block_iq4_xs_scalar(*reinterpret_cast<const ob_block_iq4_xs_t*>(bp), out);
        break;
    }
  }
}

void dequant_row_f32(ob_quant_type_t type,
                     const void* row_blocks,
                     uint32_t k_dim,
                     float* dst_f32) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  if (qk == 0u || blk_sz == 0u || !row_blocks || !dst_f32 || (k_dim % qk) != 0u) {
    return;
  }
  const uint32_t nb = k_dim / qk;
  const uint8_t* p = static_cast<const uint8_t*>(row_blocks);
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    dequant_block_fast(type, p + static_cast<size_t>(bi) * blk_sz,
                       dst_f32 + static_cast<size_t>(bi) * qk);
  }
}

void dequant_row_slice_f32(ob_quant_type_t type,
                           const void* b_nk,
                           uint32_t k_dim,
                           uint32_t row_idx,
                           uint32_t k0,
                           uint32_t tk,
                           float* dst_f32) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  if (qk == 0u || blk_sz == 0u || tk == 0u || !b_nk || !dst_f32 || (k_dim % qk) != 0u) {
    return;
  }
  const uint32_t nb_k = k_dim / qk;
  const uint8_t* row_base = static_cast<const uint8_t*>(b_nk) +
                            static_cast<size_t>(row_idx) * nb_k * blk_sz;

  if ((k0 % qk) == 0u && (tk % qk) == 0u) {
    const uint32_t blk0 = k0 / qk;
    const uint32_t blk_cnt = tk / qk;
    for (uint32_t bi = 0u; bi < blk_cnt; ++bi) {
      dequant_block_fast(type, row_base + static_cast<size_t>(blk0 + bi) * blk_sz,
                         dst_f32 + static_cast<size_t>(bi) * qk);
    }
    return;
  }

  // Sub-superblock slicing (e.g. tk=64 inside a 256-element Q4_K/Q6_K/IQ4_XS block).
  float tmp[OB_QK_K];
  const uint32_t blk_start = k0 / qk;
  const uint32_t blk_end = (k0 + tk + qk - 1u) / qk;
  uint32_t out_off = 0u;
  for (uint32_t bi = blk_start; bi < blk_end; ++bi) {
    dequant_block_fast(type, row_base + static_cast<size_t>(bi) * blk_sz, tmp);
    const uint32_t blk_k0 = bi * qk;
    const uint32_t lo = (k0 > blk_k0) ? (k0 - blk_k0) : 0u;
    const uint32_t hi = std::min<uint32_t>(qk, (k0 + tk) - blk_k0);
    const uint32_t cnt = hi - lo;
    std::memcpy(dst_f32 + out_off, tmp + lo, static_cast<size_t>(cnt) * sizeof(float));
    out_off += cnt;
  }
}

void extract_column_f32(ob_quant_type_t type,
                        const void* b_nk,
                        uint32_t k_dim,
                        uint32_t n0,
                        uint32_t tn,
                        uint32_t abs_k,
                        float* dst_col) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  if (qk == 0u || blk_sz == 0u || tn == 0u || !b_nk || !dst_col || (k_dim % qk) != 0u ||
      abs_k >= k_dim) {
    return;
  }
  const uint32_t nb_k = k_dim / qk;
  const uint32_t b_idx = abs_k / qk;
  const uint32_t rem = abs_k % qk;
  const size_t row_stride = static_cast<size_t>(nb_k) * blk_sz;
  const uint8_t* col_base = static_cast<const uint8_t*>(b_nk) +
                            static_cast<size_t>(n0) * row_stride +
                            static_cast<size_t>(b_idx) * blk_sz;

#if defined(__ARM_NEON) && defined(__aarch64__)
  if (type == OB_QUANT_Q8_0) {
    uint32_t ni = 0u;
    for (; ni + 4u <= tn; ni += 4u) {
      const auto& b0 = *reinterpret_cast<const ob_block_q8_0_t*>(col_base + static_cast<size_t>(ni + 0u) * row_stride);
      const auto& b1 = *reinterpret_cast<const ob_block_q8_0_t*>(col_base + static_cast<size_t>(ni + 1u) * row_stride);
      const auto& b2 = *reinterpret_cast<const ob_block_q8_0_t*>(col_base + static_cast<size_t>(ni + 2u) * row_stride);
      const auto& b3 = *reinterpret_cast<const ob_block_q8_0_t*>(col_base + static_cast<size_t>(ni + 3u) * row_stride);
      const uint16_t dh[4] = {
          load_u16_unaligned(&b0.d), load_u16_unaligned(&b1.d),
          load_u16_unaligned(&b2.d), load_u16_unaligned(&b3.d)
      };
      const int32_t qv[4] = {b0.qs[rem], b1.qs[rem], b2.qs[rem], b3.qs[rem]};
      float32x4_t vd = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(dh)));
      float32x4_t vq = vcvtq_f32_s32(vld1q_s32(qv));
      vst1q_f32(dst_col + ni, vmulq_f32(vd, vq));
    }
    for (; ni < tn; ++ni) {
      dst_col[ni] = extract_block_elem(type, col_base + static_cast<size_t>(ni) * row_stride, rem);
    }
    return;
  }
  if (type == OB_QUANT_Q4_0 || type == OB_QUANT_IQ4_NL) {
    const bool hi_nib = (rem >= 16u);
    const uint32_t byte_idx = rem & 15u;
    uint32_t ni = 0u;
    for (; ni + 4u <= tn; ni += 4u) {
      const uint8_t* p0 = col_base + static_cast<size_t>(ni + 0u) * row_stride;
      const uint8_t* p1 = col_base + static_cast<size_t>(ni + 1u) * row_stride;
      const uint8_t* p2 = col_base + static_cast<size_t>(ni + 2u) * row_stride;
      const uint8_t* p3 = col_base + static_cast<size_t>(ni + 3u) * row_stride;
      const uint16_t dh[4] = {
          load_u16_unaligned(p0), load_u16_unaligned(p1),
          load_u16_unaligned(p2), load_u16_unaligned(p3)
      };
      const uint32_t n0_v = hi_nib ? (p0[2u + byte_idx] >> 4) : (p0[2u + byte_idx] & 0x0Fu);
      const uint32_t n1_v = hi_nib ? (p1[2u + byte_idx] >> 4) : (p1[2u + byte_idx] & 0x0Fu);
      const uint32_t n2_v = hi_nib ? (p2[2u + byte_idx] >> 4) : (p2[2u + byte_idx] & 0x0Fu);
      const uint32_t n3_v = hi_nib ? (p3[2u + byte_idx] >> 4) : (p3[2u + byte_idx] & 0x0Fu);
      int32_t qv[4];
      if (type == OB_QUANT_Q4_0) {
        qv[0] = static_cast<int32_t>(n0_v) - 8;
        qv[1] = static_cast<int32_t>(n1_v) - 8;
        qv[2] = static_cast<int32_t>(n2_v) - 8;
        qv[3] = static_cast<int32_t>(n3_v) - 8;
      } else {
        qv[0] = kValuesIq4Nl[n0_v];
        qv[1] = kValuesIq4Nl[n1_v];
        qv[2] = kValuesIq4Nl[n2_v];
        qv[3] = kValuesIq4Nl[n3_v];
      }
      float32x4_t vd = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(dh)));
      float32x4_t vq = vcvtq_f32_s32(vld1q_s32(qv));
      vst1q_f32(dst_col + ni, vmulq_f32(vd, vq));
    }
    for (; ni < tn; ++ni) {
      dst_col[ni] = extract_block_elem(type, col_base + static_cast<size_t>(ni) * row_stride, rem);
    }
    return;
  }
#endif

  for (uint32_t ni = 0u; ni < tn; ++ni) {
    dst_col[ni] = extract_block_elem(type, col_base + static_cast<size_t>(ni) * row_stride, rem);
  }
}

float row_max_abs_f32(ob_quant_type_t type,
                      const void* b_nk,
                      uint32_t k_dim,
                      uint32_t row_idx,
                      uint32_t k0,
                      uint32_t tk) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  if (qk == 0u || blk_sz == 0u || tk == 0u || !b_nk || (k_dim % qk) != 0u) {
    return 0.0f;
  }
  const uint32_t nb_k = k_dim / qk;
  const uint8_t* row_base = static_cast<const uint8_t*>(b_nk) +
                            static_cast<size_t>(row_idx) * nb_k * blk_sz;

  float rm = 0.0f;
  if ((k0 % qk) == 0u && (tk % qk) == 0u) {
    const uint32_t blk0 = k0 / qk;
    const uint32_t blk_cnt = tk / qk;
    float tmp[OB_QK_K];
    for (uint32_t bi = 0u; bi < blk_cnt; ++bi) {
      dequant_block_fast(type, row_base + static_cast<size_t>(blk0 + bi) * blk_sz, tmp);
      uint32_t j = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
      float32x4_t vm0 = vdupq_n_f32(0.0f);
      float32x4_t vm1 = vdupq_n_f32(0.0f);
      for (; j + 8u <= qk; j += 8u) {
        vm0 = vmaxq_f32(vm0, vabsq_f32(vld1q_f32(tmp + j)));
        vm1 = vmaxq_f32(vm1, vabsq_f32(vld1q_f32(tmp + j + 4u)));
      }
      const float bm = vmaxvq_f32(vmaxq_f32(vm0, vm1));
      if (bm > rm) rm = bm;
#else
      for (; j < qk; ++j) {
        const float av = std::fabs(tmp[j]);
        if (av > rm) rm = av;
      }
#endif
    }
    return rm;
  }

  float stack_w[2048];
  std::vector<float> heap_w;
  float* w = stack_w;
  if (tk > 2048u) {
    heap_w.resize(tk);
    w = heap_w.data();
  }
  dequant_row_slice_f32(type, b_nk, k_dim, row_idx, k0, tk, w);
  for (uint32_t j = 0u; j < tk; ++j) {
    const float av = std::fabs(w[j]);
    if (av > rm) rm = av;
  }
  return rm;
}

float dot_row_f32(ob_quant_type_t type,
                  const float* a_row,
                  const void* b_row,
                  uint32_t k_dim) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  if (qk == 0u || blk_sz == 0u || !a_row || !b_row || (k_dim % qk) != 0u) {
    return 0.0f;
  }
  const uint32_t nb = k_dim / qk;
  if (type == OB_QUANT_Q4_K) {
    return dot_q4_k_blocks(a_row, static_cast<const ob_block_q4_k_t*>(b_row), nb);
  }
  if (type == OB_QUANT_IQ4_NL) {
    return dot_iq4_nl_blocks(a_row, static_cast<const ob_block_iq4_nl_t*>(b_row), nb);
  }

  // Block-buffered NEON/scalar dot for Q8_0, Q4_0, Q6_K, IQ4_XS (stays in L1).
  float tmp[OB_QK_K];
  const uint8_t* bp = static_cast<const uint8_t*>(b_row);
  double acc = 0.0;
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    dequant_block_fast(type, bp + static_cast<size_t>(bi) * blk_sz, tmp);
    acc += static_cast<double>(dot_f32_row(a_row + static_cast<size_t>(bi) * qk, tmp, qk));
  }
  return static_cast<float>(acc);
}

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
                         uint32_t tn) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  if (qk == 0u || blk_sz == 0u || !a_mk || !b_nk || !c_mn || (k_dim % qk) != 0u) {
    return;
  }

  // Fast full-block dot path when tile is block-aligned and tm == 1 (decode).
  if ((k0 % qk) == 0u && (tk % qk) == 0u && tm == 1u) {
    const uint32_t nb_k = k_dim / qk;
    const uint32_t blk0 = k0 / qk;
    const float* a_row = a_mk + static_cast<size_t>(m0) * k_dim + k0;
    float* c_row = c_mn + static_cast<size_t>(m0) * n_dim + n0;
    for (uint32_t ni = 0u; ni < tn; ++ni) {
      const uint8_t* b_slice = static_cast<const uint8_t*>(b_nk) +
                               (static_cast<size_t>(n0 + ni) * nb_k + blk0) * blk_sz;
      c_row[ni] += dot_row_f32(type, a_row, b_slice, tk);
    }
    return;
  }

  // General path (handles tm > 1 with weight reuse across rows + sub-superblock slices).
  float stack_w[2048];
  std::vector<float> heap_w;
  float* w_row = stack_w;
  if (tk > 2048u) {
    heap_w.resize(tk);
    w_row = heap_w.data();
  }
  for (uint32_t ni = 0u; ni < tn; ++ni) {
    dequant_row_slice_f32(type, b_nk, k_dim, n0 + ni, k0, tk, w_row);
    for (uint32_t mi = 0u; mi < tm; ++mi) {
      const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0;
      float* c_row = c_mn + static_cast<size_t>(m0 + mi) * n_dim + n0;
      c_row[ni] += dot_f32_row(a_row, w_row, tk);
    }
  }
}

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
                  RowNormStats* out_stats) {
  std::atomic<bool> any_nonzero_offset{false};
  if (pool && tn >= 64u && static_cast<size_t>(tn) * tk >= 32768u) {
    pool->parallel_for_fn(tn, 16u, [&](uint32_t s, uint32_t e, uint32_t) {
      requant_rows_range(type, b_nk, k_dim, n0, s, e, k0, tk, hw_k,
                         out_bq_base, out_sb_base, out_row_offset, out_stats,
                         &any_nonzero_offset);
    });
  } else {
    requant_rows_range(type, b_nk, k_dim, n0, 0u, tn, k0, tk, hw_k,
                       out_bq_base, out_sb_base, out_row_offset, out_stats,
                       &any_nonzero_offset);
  }
  return any_nonzero_offset.load(std::memory_order_relaxed);
}

}  // namespace quant
}  // namespace ob

extern "C" {

uint32_t ob_quant_block_elements(ob_quant_type_t type) {
  switch (type) {
    case OB_QUANT_Q8_0:   return OB_QK8_0;
    case OB_QUANT_Q4_0:   return OB_QK8_0;
    case OB_QUANT_Q4_K:   return OB_QK_K;
    case OB_QUANT_Q6_K:   return OB_QK_K;
    case OB_QUANT_IQ4_NL: return OB_QK4_NL;
    case OB_QUANT_IQ4_XS: return OB_QK_K;
    default:              return 0u;
  }
}

size_t ob_quant_block_bytes(ob_quant_type_t type) {
  switch (type) {
    case OB_QUANT_Q8_0:   return sizeof(ob_block_q8_0_t);
    case OB_QUANT_Q4_0:   return sizeof(ob_block_q4_0_t);
    case OB_QUANT_Q4_K:   return sizeof(ob_block_q4_k_t);
    case OB_QUANT_Q6_K:   return sizeof(ob_block_q6_k_t);
    case OB_QUANT_IQ4_NL: return sizeof(ob_block_iq4_nl_t);
    case OB_QUANT_IQ4_XS: return sizeof(ob_block_iq4_xs_t);
    default:              return 0u;
  }
}

const char* ob_quant_type_name(ob_quant_type_t type) {
  switch (type) {
    case OB_QUANT_Q8_0:   return "Q8_0";
    case OB_QUANT_Q4_0:   return "Q4_0";
    case OB_QUANT_Q4_K:   return "Q4_K";
    case OB_QUANT_Q6_K:   return "Q6_K";
    case OB_QUANT_IQ4_NL: return "IQ4_NL";
    case OB_QUANT_IQ4_XS: return "IQ4_XS";
    default:              return "UNKNOWN";
  }
}

ob_status_t ob_quant_dequantize_row_f32(ob_quant_type_t type,
                                        const void* src_blocks,
                                        uint32_t k_dim,
                                        float* dst_f32) {
  const uint32_t qk = ob_quant_block_elements(type);
  if (qk == 0u || !src_blocks || !dst_f32 || k_dim == 0u || (k_dim % qk) != 0u) {
    return OB_ERR_INVALID_ARG;
  }
  ob::quant::dequant_row_f32(type, src_blocks, k_dim, dst_f32);
  return OB_OK;
}

}  // extern "C"
