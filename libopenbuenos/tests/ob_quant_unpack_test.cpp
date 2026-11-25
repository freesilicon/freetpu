// SPDX-License-Identifier: Apache-2.0
// Dedicated unit & numerical accuracy test suite for modern GGUF quant formats
// (Q4_K, Q6_K, IQ4_NL, IQ4_XS) + Q8_0/Q4_0 regression in libopenbuenos.

#include "openbuenos.h"
#include "ob_internal.h"
#include "ob_quant_unpack.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
      ++g_failures;                                                             \
      return;                                                                   \
    }                                                                           \
  } while (0)

uint16_t f32_to_f16(float x) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &x, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exp_raw = (bits >> 23) & 0xFFu;
  const uint32_t mant = bits & 0x007FFFFFu;
  if (exp_raw == 0xFFu) {
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
  }
  int32_t exp = static_cast<int32_t>(exp_raw) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    const uint32_t m = mant | 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    return static_cast<uint16_t>(sign | ((m + (1u << (shift - 1u))) >> shift));
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  const uint32_t rem = mant & 0x1FFFu;
  uint32_t q = mant >> 13;
  if (rem > 0x1000u || (rem == 0x1000u && (q & 1u) != 0u)) {
    ++q;
    if (q == 0x0400u) {
      q = 0u;
      ++exp;
      if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | q);
}

float ref_fp16_to_fp32(uint16_t h) {
  return ob::quant::fp16_to_fp32_scalar(h);
}

// Independent reference ggml formulas (written directly from GGUF spec).
void ref_dequant_q4_k(const ob_block_q4_k_t* blocks, uint32_t nb, float* y) {
  for (uint32_t i = 0u; i < nb; ++i) {
    const uint8_t* q = blocks[i].qs;
    uint16_t dh = 0u, dmh = 0u;
    std::memcpy(&dh, &blocks[i].d, 2u);
    std::memcpy(&dmh, &blocks[i].dmin, 2u);
    const float d = ref_fp16_to_fp32(dh);
    const float min = ref_fp16_to_fp32(dmh);
    uint32_t is = 0u;
    for (uint32_t j = 0u; j < 256u; j += 64u) {
      auto get_sm = [&](uint32_t idx, uint8_t& sc, uint8_t& m) {
        if (idx < 4u) {
          sc = blocks[i].scales[idx] & 63u;
          m = blocks[i].scales[idx + 4u] & 63u;
        } else {
          sc = static_cast<uint8_t>((blocks[i].scales[idx + 4u] & 0x0Fu) |
                                    ((blocks[i].scales[idx - 4u] >> 6) << 4));
          m = static_cast<uint8_t>((blocks[i].scales[idx + 4u] >> 4) |
                                   ((blocks[i].scales[idx] >> 6) << 4));
        }
      };
      uint8_t sc1 = 0u, m1 = 0u, sc2 = 0u, m2 = 0u;
      get_sm(is + 0u, sc1, m1);
      get_sm(is + 1u, sc2, m2);
      const float d1 = d * sc1;
      const float min1 = min * m1;
      const float d2 = d * sc2;
      const float min2 = min * m2;
      for (uint32_t l = 0u; l < 32u; ++l) *y++ = d1 * static_cast<float>(q[l] & 0x0Fu) - min1;
      for (uint32_t l = 0u; l < 32u; ++l) *y++ = d2 * static_cast<float>(q[l] >> 4) - min2;
      q += 32u;
      is += 2u;
    }
  }
}

void ref_dequant_q6_k(const ob_block_q6_k_t* blocks, uint32_t nb, float* y) {
  for (uint32_t i = 0u; i < nb; ++i) {
    uint16_t dh = 0u;
    std::memcpy(&dh, &blocks[i].d, 2u);
    const float d = ref_fp16_to_fp32(dh);
    const uint8_t* ql = blocks[i].ql;
    const uint8_t* qh = blocks[i].qh;
    const int8_t* sc = blocks[i].scales;
    for (uint32_t n = 0u; n < 256u; n += 128u) {
      for (uint32_t l = 0u; l < 32u; ++l) {
        const uint32_t is = l / 16u;
        const int8_t q1 = static_cast<int8_t>(static_cast<int>((ql[l + 0u] & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32);
        const int8_t q2 = static_cast<int8_t>(static_cast<int>((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32);
        const int8_t q3 = static_cast<int8_t>(static_cast<int>((ql[l + 0u] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32);
        const int8_t q4 = static_cast<int8_t>(static_cast<int>((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32);
        y[l + 0u] = d * sc[is + 0u] * q1;
        y[l + 32u] = d * sc[is + 2u] * q2;
        y[l + 64u] = d * sc[is + 4u] * q3;
        y[l + 96u] = d * sc[is + 6u] * q4;
      }
      y += 128u;
      ql += 64u;
      qh += 32u;
      sc += 8u;
    }
  }
}

void ref_dequant_iq4_nl(const ob_block_iq4_nl_t* blocks, uint32_t nb, float* y) {
  static const int8_t kLut[16] = {
      -127, -104, -83, -65, -49, -35, -22, -10,
         1,   13,  25,  38,  53,  69,  89, 113
  };
  for (uint32_t i = 0u; i < nb; ++i) {
    uint16_t dh = 0u;
    std::memcpy(&dh, &blocks[i].d, 2u);
    const float d = ref_fp16_to_fp32(dh);
    for (uint32_t j = 0u; j < 16u; ++j) {
      y[j] = d * kLut[blocks[i].qs[j] & 0x0Fu];
      y[j + 16u] = d * kLut[blocks[i].qs[j] >> 4];
    }
    y += 32u;
  }
}

void ref_dequant_iq4_xs(const ob_block_iq4_xs_t* blocks, uint32_t nb, float* y) {
  static const int8_t kLut[16] = {
      -127, -104, -83, -65, -49, -35, -22, -10,
         1,   13,  25,  38,  53,  69,  89, 113
  };
  for (uint32_t i = 0u; i < nb; ++i) {
    uint16_t dh = 0u, sh = 0u;
    std::memcpy(&dh, &blocks[i].d, 2u);
    std::memcpy(&sh, &blocks[i].scales_h, 2u);
    const float d = ref_fp16_to_fp32(dh);
    const uint8_t* qs = blocks[i].qs;
    for (uint32_t ib = 0u; ib < 8u; ++ib) {
      const int ls = static_cast<int>(((blocks[i].scales_l[ib / 2u] >> (4u * (ib & 1u))) & 0x0Fu) |
                                      (((sh >> (2u * ib)) & 3u) << 4));
      const float dl = d * (ls - 32);
      for (uint32_t j = 0u; j < 16u; ++j) {
        y[j] = dl * kLut[qs[j] & 0x0Fu];
        y[j + 16u] = dl * kLut[qs[j] >> 4];
      }
      y += 32u;
      qs += 16u;
    }
  }
}

double compute_nmse(const std::vector<float>& got, const std::vector<float>& ref) {
  if (got.size() != ref.size() || got.empty()) return 1.0;
  double err_sq = 0.0;
  double ref_sq = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    err_sq += d * d;
    ref_sq += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return (ref_sq > 0.0) ? (err_sq / ref_sq) : err_sq;
}

std::vector<uint8_t> make_random_quant_matrix(ob_quant_type_t type,
                                              uint32_t n_rows,
                                              uint32_t k_dim,
                                              uint32_t seed,
                                              bool asymmetric_q4_k = false) {
  std::mt19937 rng(seed);
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t blk_sz = ob_quant_block_bytes(type);
  const uint32_t nb_k = k_dim / qk;
  std::vector<uint8_t> buf(static_cast<size_t>(n_rows) * nb_k * blk_sz, 0u);

  std::uniform_real_distribution<float> d_dist(0.002f, 0.04f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> sc6_dist(1, 63);
  std::uniform_int_distribution<int> sc8_dist(-60, 60);

  for (uint32_t r = 0u; r < n_rows; ++r) {
    for (uint32_t bi = 0u; bi < nb_k; ++bi) {
      uint8_t* bp = buf.data() + (static_cast<size_t>(r) * nb_k + bi) * blk_sz;
      if (type == OB_QUANT_Q8_0) {
        auto* blk = reinterpret_cast<ob_block_q8_0_t*>(bp);
        uint16_t dh = f32_to_f16(d_dist(rng));
        std::memcpy(&blk->d, &dh, 2u);
        for (uint32_t j = 0u; j < 32u; ++j) {
          blk->qs[j] = static_cast<int8_t>(sc8_dist(rng));
        }
      } else if (type == OB_QUANT_Q4_0) {
        auto* blk = reinterpret_cast<ob_block_q4_0_t*>(bp);
        const float dval = d_dist(rng) * ((rng() & 1u) ? -1.0f : 1.0f);
        uint16_t dh = f32_to_f16(dval);
        std::memcpy(&blk->d, &dh, 2u);
        for (uint32_t j = 0u; j < 16u; ++j) {
          blk->qs[j] = static_cast<uint8_t>(byte_dist(rng));
        }
      } else if (type == OB_QUANT_Q4_K) {
        auto* blk = reinterpret_cast<ob_block_q4_k_t*>(bp);
        uint16_t dh = f32_to_f16(asymmetric_q4_k ? 0.002f : (0.0005f + 0.003f * (rng() & 7u)));
        // In standard zero-centered Q4_K, min ~= 7.5 * sc * d; in asymmetric_q4_k, huge dmin shifts all weights negative!
        uint16_t dmh = f32_to_f16(asymmetric_q4_k ? 0.25f : (0.003f + 0.018f * (rng() & 7u)));
        std::memcpy(&blk->d, &dh, 2u);
        std::memcpy(&blk->dmin, &dmh, 2u);
        uint8_t sc[8], mn[8];
        for (uint32_t j = 0u; j < 8u; ++j) {
          sc[j] = static_cast<uint8_t>(asymmetric_q4_k ? (4 + (j % 4)) : sc6_dist(rng));
          mn[j] = static_cast<uint8_t>(asymmetric_q4_k ? (48 + (j % 8)) : (sc[j] / 2u + 1u));
        }
        std::memset(blk->scales, 0, OB_K_SCALE_SIZE);
        for (uint32_t j = 0u; j < 4u; ++j) {
          blk->scales[j] = static_cast<uint8_t>(sc[j] & 63u);
          blk->scales[j + 4u] = static_cast<uint8_t>(mn[j] & 63u);
        }
        for (uint32_t j = 4u; j < 8u; ++j) {
          blk->scales[j + 4u] = static_cast<uint8_t>((sc[j] & 0x0Fu) | ((mn[j] & 0x0Fu) << 4));
          blk->scales[j - 4u] |= static_cast<uint8_t>(((sc[j] >> 4) & 3u) << 6);
          blk->scales[j] |= static_cast<uint8_t>(((mn[j] >> 4) & 3u) << 6);
        }
        for (uint32_t j = 0u; j < 128u; ++j) {
          blk->qs[j] = static_cast<uint8_t>(byte_dist(rng));
        }
      } else if (type == OB_QUANT_Q6_K) {
        auto* blk = reinterpret_cast<ob_block_q6_k_t*>(bp);
        uint16_t dh = f32_to_f16(0.001f + 0.002f * (rng() & 7u));
        std::memcpy(&blk->d, &dh, 2u);
        for (uint32_t j = 0u; j < 128u; ++j) blk->ql[j] = static_cast<uint8_t>(byte_dist(rng));
        for (uint32_t j = 0u; j < 64u; ++j) blk->qh[j] = static_cast<uint8_t>(byte_dist(rng));
        for (uint32_t j = 0u; j < 16u; ++j) blk->scales[j] = static_cast<int8_t>(sc8_dist(rng));
      } else if (type == OB_QUANT_IQ4_NL) {
        auto* blk = reinterpret_cast<ob_block_iq4_nl_t*>(bp);
        const float dval = (0.001f + 0.004f * (rng() & 7u)) * ((rng() & 1u) ? -1.0f : 1.0f);
        uint16_t dh = f32_to_f16(dval);
        std::memcpy(&blk->d, &dh, 2u);
        for (uint32_t j = 0u; j < 16u; ++j) blk->qs[j] = static_cast<uint8_t>(byte_dist(rng));
      } else if (type == OB_QUANT_IQ4_XS) {
        auto* blk = reinterpret_cast<ob_block_iq4_xs_t*>(bp);
        const float dval = (0.0002f + 0.0005f * (rng() & 7u)) * ((rng() & 1u) ? -1.0f : 1.0f);
        uint16_t dh = f32_to_f16(dval);
        std::memcpy(&blk->d, &dh, 2u);
        uint16_t sh = static_cast<uint16_t>(rng() & 0xFFFFu);
        std::memcpy(&blk->scales_h, &sh, 2u);
        for (uint32_t j = 0u; j < 4u; ++j) blk->scales_l[j] = static_cast<uint8_t>(byte_dist(rng));
        for (uint32_t j = 0u; j < 128u; ++j) blk->qs[j] = static_cast<uint8_t>(byte_dist(rng));
      }
    }
  }
  return buf;
}

std::vector<float> dequant_matrix_ref(ob_quant_type_t type,
                                      const std::vector<uint8_t>& raw,
                                      uint32_t n_rows,
                                      uint32_t k_dim) {
  const uint32_t qk = ob_quant_block_elements(type);
  const size_t row_bytes = static_cast<size_t>(k_dim / qk) * ob_quant_block_bytes(type);
  std::vector<float> out(static_cast<size_t>(n_rows) * k_dim, 0.0f);
  for (uint32_t r = 0u; r < n_rows; ++r) {
    ob::quant::dequant_row_scalar_f32(type, raw.data() + static_cast<size_t>(r) * row_bytes,
                                      k_dim, out.data() + static_cast<size_t>(r) * k_dim);
  }
  return out;
}

class TempFile {
 public:
  ~TempFile() {
    if (!path_.empty()) ::unlink(path_.c_str());
  }
  bool write(const void* data, size_t n) {
    char tmpl[] = "/tmp/ob_quant_test_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd < 0) return false;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t off = 0u;
    while (off < n) {
      ssize_t w = ::write(fd, p + off, n - off);
      if (w <= 0) {
        ::close(fd);
        ::unlink(tmpl);
        return false;
      }
      off += static_cast<size_t>(w);
    }
    ::close(fd);
    path_ = tmpl;
    return true;
  }
  const char* path() const { return path_.c_str(); }
 private:
  std::string path_;
};

// Unpack a Buenos ext[2] tile back into [hw_n, hw_k] int8 weights for exact HW simulation.
void unpack_ext2_w8a8_tile(const uint8_t* ext2, uint32_t hw_n, uint32_t hw_k,
                           bool is_lt64, std::vector<int8_t>* out_nk) {
  out_nk->assign(static_cast<size_t>(hw_n) * hw_k, 0);
  if (is_lt64) {
    const uint32_t n_groups = hw_n / 8u;
    const uint32_t k_blocks = hw_k / 16u;
    const size_t group_stride = 128u + static_cast<size_t>(k_blocks) * 128u;
    for (uint32_t ng = 0u; ng < n_groups; ++ng) {
      const uint8_t* gbase = ext2 + static_cast<size_t>(ng) * group_stride + 128u;
      for (uint32_t kb = 0u; kb < k_blocks; ++kb) {
        const uint8_t* blk = gbase + static_cast<size_t>(kb) * 128u;
        for (uint32_t ni = 0u; ni < 8u; ++ni) {
          for (uint32_t ki = 0u; ki < 16u; ++ki) {
            const uint8_t u = blk[ni * 16u + ki];
            (*out_nk)[static_cast<size_t>(ng * 8u + ni) * hw_k + kb * 16u + ki] =
                static_cast<int8_t>(u ^ 0x80u);
          }
        }
      }
    }
    return;
  }
  // Standard / lt_1x* layout (include_bias=false):
  //   - When N >= 32, N % 32 == 0, K % 16 == 0: [N/32, K/4, 32, 4] xor 0x80
  //   - Otherwise: [K/4, N, 4] xor 0x80
  if (hw_n >= 32u && (hw_n % 32u) == 0u && (hw_k % 16u) == 0u) {
    const uint32_t n_groups = hw_n / 32u;
    const uint32_t k_groups = hw_k / 4u;
    for (uint32_t g = 0u; g < n_groups; ++g) {
      const uint8_t* gbase = ext2 + static_cast<size_t>(g) * hw_k * 32u;
      for (uint32_t k4 = 0u; k4 < k_groups; ++k4) {
        const uint8_t* kbase = gbase + static_cast<size_t>(k4) * 128u;
        for (uint32_t ni = 0u; ni < 32u; ++ni) {
          for (uint32_t lane = 0u; lane < 4u; ++lane) {
            (*out_nk)[static_cast<size_t>(g * 32u + ni) * hw_k + k4 * 4u + lane] =
                static_cast<int8_t>(kbase[ni * 4u + lane] ^ 0x80u);
          }
        }
      }
    }
    return;
  }
  for (uint32_t k4 = 0u; k4 < hw_k / 4u; ++k4) {
    for (uint32_t ni = 0u; ni < hw_n; ++ni) {
      const uint8_t* src = ext2 + (static_cast<size_t>(k4) * hw_n + ni) * 4u;
      for (uint32_t lane = 0u; lane < 4u; ++lane) {
        (*out_nk)[static_cast<size_t>(ni) * hw_k + k4 * 4u + lane] =
            static_cast<int8_t>(src[lane] ^ 0x80u);
      }
    }
  }
}

void unpack_ext2_w8a16_tile(const uint8_t* ext2, uint32_t hw_n, uint32_t hw_k,
                            std::vector<int8_t>* out_nk) {
  out_nk->assign(static_cast<size_t>(hw_n) * hw_k, 0);
  // [N/32, K/2, 32, 2] w_i8 ^ 0x80
  const uint32_t n_tiles = hw_n / 32u;
  const uint32_t k_pairs = hw_k / 2u;
  for (uint32_t nt = 0u; nt < n_tiles; ++nt) {
    for (uint32_t kp = 0u; kp < k_pairs; ++kp) {
      const uint8_t* base = ext2 + (static_cast<size_t>(nt) * k_pairs + kp) * 64u;
      for (uint32_t nr = 0u; nr < 32u; ++nr) {
        (*out_nk)[static_cast<size_t>(nt * 32u + nr) * hw_k + kp * 2u + 0u] =
            static_cast<int8_t>(base[nr * 2u + 0u] ^ 0x80u);
        (*out_nk)[static_cast<size_t>(nt * 32u + nr) * hw_k + kp * 2u + 1u] =
            static_cast<int8_t>(base[nr * 2u + 1u] ^ 0x80u);
      }
    }
  }
}

void test_abi_and_validation() {
  CHECK(OB_QUANT_Q8_0 == 0);
  CHECK(OB_QUANT_Q4_0 == 1);
  CHECK(OB_QUANT_Q4_K == 2);
  CHECK(OB_QUANT_Q6_K == 3);
  CHECK(OB_QUANT_IQ4_NL == 4);
  CHECK(OB_QUANT_IQ4_XS == 5);

  CHECK(ob_quant_block_elements(OB_QUANT_Q8_0) == 32u);
  CHECK(ob_quant_block_elements(OB_QUANT_Q4_0) == 32u);
  CHECK(ob_quant_block_elements(OB_QUANT_Q4_K) == 256u);
  CHECK(ob_quant_block_elements(OB_QUANT_Q6_K) == 256u);
  CHECK(ob_quant_block_elements(OB_QUANT_IQ4_NL) == 32u);
  CHECK(ob_quant_block_elements(OB_QUANT_IQ4_XS) == 256u);
  CHECK(ob_quant_block_elements(static_cast<ob_quant_type_t>(99)) == 0u);

  CHECK(ob_quant_block_bytes(OB_QUANT_Q8_0) == 34u);
  CHECK(ob_quant_block_bytes(OB_QUANT_Q4_0) == 18u);
  CHECK(ob_quant_block_bytes(OB_QUANT_Q4_K) == 144u);
  CHECK(ob_quant_block_bytes(OB_QUANT_Q6_K) == 210u);
  CHECK(ob_quant_block_bytes(OB_QUANT_IQ4_NL) == 18u);
  CHECK(ob_quant_block_bytes(OB_QUANT_IQ4_XS) == 136u);
  CHECK(ob_quant_block_bytes(static_cast<ob_quant_type_t>(99)) == 0u);

  CHECK(std::strcmp(ob_quant_type_name(OB_QUANT_Q4_K), "Q4_K") == 0);
  CHECK(std::strcmp(ob_quant_type_name(OB_QUANT_Q6_K), "Q6_K") == 0);
  CHECK(std::strcmp(ob_quant_type_name(OB_QUANT_IQ4_NL), "IQ4_NL") == 0);
  CHECK(std::strcmp(ob_quant_type_name(OB_QUANT_IQ4_XS), "IQ4_XS") == 0);

  ob_gemm_f32_quant_job_t job{};
  ob_gemm_f32_quant_job_set_type(&job, OB_QUANT_IQ4_XS);
  CHECK(ob_gemm_f32_quant_job_get_type(&job) == OB_QUANT_IQ4_XS);
  CHECK(job.is_q4_0 == 5u);

  // K % 256 != 0 must fail cleanly with OB_ERR_INVALID_ARG for 256-weight super-blocks.
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);

  std::vector<float> a(576u, 1.0f);
  std::vector<float> c(64u, 0.0f);
  std::vector<uint8_t> dummy_b(64u * 256u, 0u);
  const ob_quant_type_t super_types[3] = {OB_QUANT_Q4_K, OB_QUANT_Q6_K, OB_QUANT_IQ4_XS};
  const uint32_t bad_ks[5] = {32u, 64u, 128u, 192u, 576u};
  for (ob_quant_type_t qt : super_types) {
    for (uint32_t bad_k : bad_ks) {
      CHECK(ob_quant_dequantize_row_f32(qt, dummy_b.data(), bad_k, a.data()) == OB_ERR_INVALID_ARG);
      ob_gemm_f32_quant_job_t bj{};
      bj.b_nk = dummy_b.data();
      bj.c_mn = c.data();
      bj.n_dim = 16u;
      ob_gemm_f32_quant_job_set_type(&bj, qt);
      CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a.data(), 1u, bad_k, &bj, 1u, nullptr) == OB_ERR_INVALID_ARG);
    }
    ob_gemm_swiglu_quant_desc_t sd{};
    sd.w_gate_hk = dummy_b.data();
    sd.w_up_hk = dummy_b.data();
    sd.w_down_nh = dummy_b.data();
    sd.m_dim = 1u;
    sd.k_dim = 64u;
    sd.h_dim = 256u;
    sd.n_dim = 16u;
    ob_gemm_swiglu_quant_desc_set_types(&sd, qt, OB_QUANT_Q8_0, OB_QUANT_Q8_0);
    CHECK(ob_gemm_swiglu_f32_quant(ctx, a.data(), &sd, c.data(), nullptr) == OB_ERR_INVALID_ARG);

    sd.k_dim = 256u;
    sd.h_dim = 128u;
    ob_gemm_swiglu_quant_desc_set_types(&sd, OB_QUANT_Q8_0, OB_QUANT_Q8_0, qt);
    CHECK(ob_gemm_swiglu_f32_quant(ctx, a.data(), &sd, c.data(), nullptr) == OB_ERR_INVALID_ARG);
  }
  // Invalid enum value
  {
    ob_gemm_f32_quant_job_t bj{};
    bj.b_nk = dummy_b.data();
    bj.c_mn = c.data();
    bj.n_dim = 16u;
    bj.is_q4_0 = 6u;
    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a.data(), 1u, 256u, &bj, 1u, nullptr) == OB_ERR_INVALID_ARG);
  }

  ob_context_destroy(ctx);
  std::printf("[PASS] test_abi_and_validation\n");
}

void test_bit_exact_dequant_and_column_extract() {
  constexpr uint32_t kK = 512u;
  constexpr uint32_t kN = 7u;

  const ob_quant_type_t types[4] = {
      OB_QUANT_Q4_K, OB_QUANT_Q6_K, OB_QUANT_IQ4_NL, OB_QUANT_IQ4_XS
  };

  for (uint32_t ti = 0u; ti < 4u; ++ti) {
    const ob_quant_type_t qt = types[ti];
    const auto raw = make_random_quant_matrix(qt, kN, kK, 1000u + ti);
    const uint32_t qk = ob_quant_block_elements(qt);
    const uint32_t nb = kK / qk;
    const size_t row_bytes = static_cast<size_t>(nb) * ob_quant_block_bytes(qt);

    std::vector<float> ref_mat(static_cast<size_t>(kN) * kK, 0.0f);
    for (uint32_t r = 0u; r < kN; ++r) {
      const uint8_t* rp = raw.data() + static_cast<size_t>(r) * row_bytes;
      float* ref_row = ref_mat.data() + static_cast<size_t>(r) * kK;
      if (qt == OB_QUANT_Q4_K) {
        ref_dequant_q4_k(reinterpret_cast<const ob_block_q4_k_t*>(rp), nb, ref_row);
      } else if (qt == OB_QUANT_Q6_K) {
        ref_dequant_q6_k(reinterpret_cast<const ob_block_q6_k_t*>(rp), nb, ref_row);
      } else if (qt == OB_QUANT_IQ4_NL) {
        ref_dequant_iq4_nl(reinterpret_cast<const ob_block_iq4_nl_t*>(rp), nb, ref_row);
      } else {
        ref_dequant_iq4_xs(reinterpret_cast<const ob_block_iq4_xs_t*>(rp), nb, ref_row);
      }

      std::vector<float> got_fast(kK, 0.0f);
      std::vector<float> got_scalar(kK, 0.0f);
      CHECK(ob_quant_dequantize_row_f32(qt, rp, kK, got_fast.data()) == OB_OK);
      ob::quant::dequant_row_scalar_f32(qt, rp, kK, got_scalar.data());

      for (uint32_t k = 0u; k < kK; ++k) {
        uint32_t b_ref = 0u, b_fast = 0u, b_scal = 0u;
        std::memcpy(&b_ref, &ref_row[k], 4u);
        std::memcpy(&b_fast, &got_fast[k], 4u);
        std::memcpy(&b_scal, &got_scalar[k], 4u);
        if (b_ref != b_fast || b_ref != b_scal) {
          std::fprintf(stderr, "BIT MISMATCH %s row=%u k=%u ref=%g (0x%08x) fast=%g (0x%08x) scalar=%g (0x%08x)\n",
                       ob_quant_type_name(qt), r, k,
                       ref_row[k], b_ref, got_fast[k], b_fast, got_scalar[k], b_scal);
          CHECK(false);
        }
      }

      // Verify sub-superblock slicing at unaligned k0 / tk boundaries (e.g. 64, 96, 192).
      const uint32_t slices[][2] = {{0u, 64u}, {64u, 64u}, {96u, 96u}, {192u, 128u}, {224u, 64u}, {320u, 192u}};
      for (const auto& sl : slices) {
        std::vector<float> sub(sl[1], 0.0f);
        ob::quant::dequant_row_slice_f32(qt, raw.data(), kK, r, sl[0], sl[1], sub.data());
        for (uint32_t j = 0u; j < sl[1]; ++j) {
          CHECK(sub[j] == ref_row[sl[0] + j]);
        }
      }
    }

    // Verify extract_column_f32 bit-exact for ALL 512 column positions!
    std::vector<float> col(kN, 0.0f);
    for (uint32_t k = 0u; k < kK; ++k) {
      ob::quant::extract_column_f32(qt, raw.data(), kK, 0u, kN, k, col.data());
      for (uint32_t r = 0u; r < kN; ++r) {
        CHECK(col[r] == ref_mat[static_cast<size_t>(r) * kK + k]);
      }
    }
  }

  // Dedicated Q6_K -32 recentering boundary check: ql=0,qh=0 -> -32; ql=0,qh=0xAA -> 0; ql=0xFF,qh=0xFF -> +31.
  {
    ob_block_q6_k_t blk{};
    uint16_t one = f32_to_f16(1.0f);
    std::memcpy(&blk.d, &one, 2u);
    std::memset(blk.scales, 1, sizeof(blk.scales));
    std::vector<float> out(256u, 0.0f);
    CHECK(ob_quant_dequantize_row_f32(OB_QUANT_Q6_K, &blk, 256u, out.data()) == OB_OK);
    for (uint32_t i = 0u; i < 256u; ++i) CHECK(out[i] == -32.0f);

    std::memset(blk.qh, 0xAA, sizeof(blk.qh)); // 2-bit pattern '10' (=2) in all 4 planes -> 32 - 32 = 0
    CHECK(ob_quant_dequantize_row_f32(OB_QUANT_Q6_K, &blk, 256u, out.data()) == OB_OK);
    for (uint32_t i = 0u; i < 256u; ++i) CHECK(out[i] == 0.0f);

    std::memset(blk.ql, 0xFF, sizeof(blk.ql));
    std::memset(blk.qh, 0xFF, sizeof(blk.qh)); // 15 | (3<<4) = 63 -> 63 - 32 = +31
    CHECK(ob_quant_dequantize_row_f32(OB_QUANT_Q6_K, &blk, 256u, out.data()) == OB_OK);
    for (uint32_t i = 0u; i < 256u; ++i) CHECK(out[i] == 31.0f);
  }

  // Dedicated IQ4_NL exhaustive 16-codebook verification
  {
    ob_block_iq4_nl_t blk{};
    uint16_t two = f32_to_f16(2.0f);
    std::memcpy(&blk.d, &two, 2u);
    for (uint32_t j = 0u; j < 16u; ++j) {
      blk.qs[j] = static_cast<uint8_t>(j | ((15u - j) << 4));
    }
    std::vector<float> out(32u, 0.0f);
    CHECK(ob_quant_dequantize_row_f32(OB_QUANT_IQ4_NL, &blk, 32u, out.data()) == OB_OK);
    for (uint32_t j = 0u; j < 16u; ++j) {
      CHECK(out[j] == 2.0f * static_cast<float>(ob::quant::kValuesIq4Nl[j]));
      CHECK(out[j + 16u] == 2.0f * static_cast<float>(ob::quant::kValuesIq4Nl[15u - j]));
    }
  }

  std::printf("[PASS] test_bit_exact_dequant_and_column_extract (Q4_K/Q6_K/IQ4_NL/IQ4_XS 0-ULP)\n");
}

void test_q4_k_algebraic_dmin_centering() {
  constexpr uint32_t kN = 32u;
  constexpr uint32_t kK = 512u;

  // Build heavily asymmetric Q4_K weights where all entries are shifted far negative by dmin (~[-14, -12]).
  auto raw = make_random_quant_matrix(OB_QUANT_Q4_K, kN, kK, 4242u, /*asymmetric_q4_k=*/true);
  // Make row 0 a pure constant non-zero row (`q=0, dmin=0.5, m=10` -> every weight == -5.0f).
  {
    auto* r0 = reinterpret_cast<ob_block_q4_k_t*>(raw.data());
    for (uint32_t bi = 0u; bi < kK / 256u; ++bi) {
      std::memset(&r0[bi], 0, sizeof(ob_block_q4_k_t));
      uint16_t dmin_h = f32_to_f16(0.5f);
      std::memcpy(&r0[bi].dmin, &dmin_h, 2u);
      for (uint32_t j = 0u; j < 4u; ++j) r0[bi].scales[j + 4u] = 10u;
      for (uint32_t j = 4u; j < 8u; ++j) r0[bi].scales[j + 4u] = static_cast<uint8_t>(10u << 4);
    }
  }

  const auto w_ref = dequant_matrix_ref(OB_QUANT_Q4_K, raw, kN, kK);
  std::vector<int8_t> bq_centered(static_cast<size_t>(kN) * kK, 0);
  std::vector<float> sb_centered(kN, 0.0f);
  std::vector<float> mu_centered(kN, 0.0f);
  const bool any_off = ob::quant::requant_rows(
      OB_QUANT_Q4_K, raw.data(), kK, 0u, kN, 0u, kK, kK,
      bq_centered.data(), sb_centered.data(), mu_centered.data(), nullptr, nullptr);
  CHECK(any_off);
  // Row 0 is constant -5.0f -> exact offset -5.0f, zero scale, zero int8 residuals!
  CHECK(sb_centered[0] == 0.0f);
  CHECK(std::fabs(mu_centered[0] - (-5.0f)) < 1e-6f);

  // Compare naive uncentered INT8 vs algebraic dmin-centered INT8 on a random activation vector
  // WITH an extreme outlier channel at k=77 to prove outlier peeling algebra is exact.
  std::mt19937 rng(99u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> a(kK);
  for (uint32_t k = 0u; k < kK; ++k) a[k] = nd(rng);
  a[77] = 45.0f; // 45-sigma outlier!

  std::vector<int8_t> bq_naive(static_cast<size_t>(kN) * kK, 0);
  std::vector<float> sb_naive(kN, 0.0f);
  const bool naive_off = ob::quant::requant_rows(
      OB_QUANT_Q4_K, raw.data(), kK, 0u, kN, 0u, kK, kK,
      bq_naive.data(), sb_naive.data(), nullptr, nullptr, nullptr);
  CHECK(!naive_off);

  std::vector<float> y_ref(kN, 0.0f), y_cent(kN, 0.0f), y_naive(kN, 0.0f);
  // Quantize activation with outlier peel
  std::vector<int8_t> aq(kK, 0);
  float sa = 0.0f;
  std::vector<std::pair<uint32_t, float>> outliers;
  ob::quantize_activation_row_split(a.data(), kK, aq.data(), &sa, &outliers);
  CHECK(!outliers.empty());

  double a_sum_non_out = 0.0;
  for (uint32_t k = 0u; k < kK; ++k) {
    bool is_out = false;
    for (const auto& kv : outliers) {
      if (kv.first == k) { is_out = true; break; }
    }
    if (!is_out) a_sum_non_out += static_cast<double>(a[k]);
  }

  for (uint32_t n = 0u; n < kN; ++n) {
    double ref_acc = 0.0;
    for (uint32_t k = 0u; k < kK; ++k) {
      ref_acc += static_cast<double>(a[k]) * static_cast<double>(w_ref[static_cast<size_t>(n) * kK + k]);
    }
    y_ref[n] = static_cast<float>(ref_acc);

    int32_t dot_c = 0, dot_n = 0;
    for (uint32_t k = 0u; k < kK; ++k) {
      dot_c += static_cast<int32_t>(aq[k]) * static_cast<int32_t>(bq_centered[static_cast<size_t>(n) * kK + k]);
      dot_n += static_cast<int32_t>(aq[k]) * static_cast<int32_t>(bq_naive[static_cast<size_t>(n) * kK + k]);
    }
    double out_term = 0.0;
    for (const auto& kv : outliers) {
      float wcol = 0.0f;
      ob::quant::extract_column_f32(OB_QUANT_Q4_K, raw.data(), kK, n, 1u, kv.first, &wcol);
      out_term += static_cast<double>(kv.second) * static_cast<double>(wcol);
    }
    y_cent[n] = static_cast<float>(static_cast<double>(dot_c) * sa * sb_centered[n] +
                                   static_cast<double>(mu_centered[n]) * a_sum_non_out + out_term);
    y_naive[n] = static_cast<float>(static_cast<double>(dot_n) * sa * sb_naive[n] + out_term);
  }

  // Row 0 constant weight must be exact to FP32!
  CHECK(std::fabs(y_cent[0] - y_ref[0]) < 1e-4f);
  const double nmse_cent = compute_nmse(y_cent, y_ref);
  const double nmse_naive = compute_nmse(y_naive, y_ref);
  CHECK(nmse_cent < 1e-4);
  CHECK(nmse_cent < nmse_naive);
  std::printf("[PASS] test_q4_k_algebraic_dmin_centering (nmse_centered=%.3e vs naive=%.3e)\n",
              nmse_cent, nmse_naive);
}

void test_gemm_cache_ext2_w8a8_and_w8a16_accuracy() {
  std::vector<uint8_t> synth_interp(65536u, 0x3Cu);
  TempFile interp_file;
  CHECK(interp_file.write(synth_interp.data(), synth_interp.size()));

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);
  CHECK(ob_gemm_register_builtin_emitter(ctx, interp_file.path()) == OB_OK);

  const ob_quant_type_t all_types[6] = {
      OB_QUANT_Q8_0, OB_QUANT_Q4_0, OB_QUANT_Q4_K,
      OB_QUANT_Q6_K, OB_QUANT_IQ4_NL, OB_QUANT_IQ4_XS
  };

  std::mt19937 rng(2026u);
  std::normal_distribution<float> nd(0.0f, 1.0f);

  // Test 1: M=1 lt_1x* (K=512, N=512) + M=64 lt_64x* (K=512, N=128) + small tile M=4 (K=256, N=64, sub-superblock tk=64!)
  for (uint32_t ti = 0u; ti < 6u; ++ti) {
    const ob_quant_type_t qt = all_types[ti];
    const uint32_t K = 512u, N = 512u;
    const auto w_raw = make_random_quant_matrix(qt, N, K, 3000u + ti, qt == OB_QUANT_Q4_K);
    const auto w_ref = dequant_matrix_ref(qt, w_raw, N, K);

    std::vector<float> a(K);
    for (uint32_t k = 0u; k < K; ++k) a[k] = nd(rng);
    std::vector<float> y_ref(N, 0.0f);
    for (uint32_t n = 0u; n < N; ++n) {
      double acc = 0.0;
      for (uint32_t k = 0u; k < K; ++k) acc += static_cast<double>(a[k]) * w_ref[static_cast<size_t>(n) * K + k];
      y_ref[n] = static_cast<float>(acc);
    }

    // Register as immutable weight region to also test O(1) immutable hash path!
    CHECK(ob_gemm_register_immutable_weights(ctx, w_raw.data(), w_raw.size(), 1u) == OB_OK);

    std::vector<float> y_mock(N, 0.0f);
    ob_gemm_f32_quant_job_t job{};
    job.b_nk = w_raw.data();
    job.c_mn = y_mock.data();
    job.n_dim = N;
    job.activation = OB_ACT_NONE;
    ob_gemm_f32_quant_job_set_type(&job, qt);

    const uint64_t hits_before = ctx->weight_cache_hits;
    const uint64_t misses_before = ctx->weight_cache_misses;
    ob_invoke_stats_t st1{}, st2{};
    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a.data(), 1u, K, &job, 1u, &st1) == OB_OK);
    CHECK(ctx->weight_cache_misses == misses_before + 1u);
    CHECK(compute_nmse(y_mock, y_ref) < 1e-6);

    // Second call must hit the ext[2] DMA-BUF weight cache with ZERO re-unpacking!
    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a.data(), 1u, K, &job, 1u, &st2) == OB_OK);
    CHECK(ctx->weight_cache_hits == hits_before + 1u);

    // Verify actual cached ext[2] DMA-BUF contents via exact W8A8 hardware simulation!
    ob::CachedWeightTile* wt = ctx->weight_cache_lru.front();
    CHECK(wt != nullptr && wt->ext2_buf != nullptr);
    CHECK(wt->hw_k == K && wt->hw_n == N);
    std::vector<int8_t> bq_unpacked;
    unpack_ext2_w8a8_tile(static_cast<const uint8_t*>(ob_buffer_data(wt->ext2_buf)),
                          N, K, /*is_lt64=*/false, &bq_unpacked);

    std::vector<int8_t> aq(K, 0);
    float sa = 0.0f;
    ob::quantize_activation_row_split(a.data(), K, aq.data(), &sa, nullptr);
    const uint32_t rms_shift = ob::compute_activation_rms_shift(aq.data(), 1u, K, K);
    const uint32_t eff_d = (rms_shift > 0u)
        ? ob::quant_scale_D_grid(static_cast<double>(std::max<uint32_t>(1u, wt->scale_d >> rms_shift)))
        : wt->scale_d;
    double a_sum = 0.0;
    for (uint32_t k = 0u; k < K; ++k) a_sum += static_cast<double>(a[k]);

    const bool has_mu = (wt->row_s_b.size() >= 2u * N);
    if (qt == OB_QUANT_Q4_K) {
      CHECK(has_mu);
    } else {
      CHECK(wt->row_s_b.size() == N);
    }
    std::vector<float> y_hw_w8a8(N, 0.0f);
    std::vector<float> y_hw_w8a16(N, 0.0f);
    for (uint32_t n = 0u; n < N; ++n) {
      int32_t acc_i32 = 0;
      float acc_fp16 = 0.0f;
      for (uint32_t k = 0u; k < K; ++k) {
        const int8_t wq = bq_unpacked[static_cast<size_t>(n) * K + k];
        acc_i32 += static_cast<int32_t>(aq[k]) * static_cast<int32_t>(wq);
        acc_fp16 += ob::round_f32_to_fp16_f32(a[k]) * static_cast<float>(wq);
      }
      int32_t q_out = static_cast<int32_t>(std::lrint(static_cast<double>(acc_i32) / static_cast<double>(eff_d)));
      if (q_out < -128) q_out = -128;
      if (q_out > 127) q_out = 127;
      const float mu = has_mu ? wt->row_s_b[N + n] : 0.0f;
      y_hw_w8a8[n] = static_cast<float>(q_out) * (sa * static_cast<float>(eff_d) * wt->row_s_b[n]) +
                     mu * static_cast<float>(a_sum);
      y_hw_w8a16[n] = acc_fp16 * wt->row_s_b[n] + mu * static_cast<float>(a_sum);
    }
    const double nmse_w8a8 = compute_nmse(y_hw_w8a8, y_ref);
    const double nmse_w8a16 = compute_nmse(y_hw_w8a16, y_ref);
    std::printf("  [OK] %-6s W8A8_ext2_NMSE=%.3e W8A16_ext2_NMSE=%.3e (D=%u->%u)\n",
                ob_quant_type_name(qt), nmse_w8a8, nmse_w8a16, wt->scale_d, eff_d);
    CHECK(nmse_w8a8 < 1e-3);
    CHECK(nmse_w8a16 < 1e-3);

    // Also verify W8A16 ext[2] packing layout round-trip using pack_w8a16_ext2_into!
    std::vector<uint8_t> w8a16_ext2(ob::pack_w8a16_ext2_size(N, K), 0u);
    ob::pack_w8a16_ext2_into(w8a16_ext2.data(), N, K, bq_unpacked.data());
    std::vector<int8_t> bq_w8a16_rt;
    unpack_ext2_w8a16_tile(w8a16_ext2.data(), N, K, &bq_w8a16_rt);
    CHECK(bq_w8a16_rt == bq_unpacked);

    // Verify M=64 lt_64x* path (K=512, N=128)
    const uint32_t M64 = 64u, N64 = 128u;
    std::vector<float> a64(static_cast<size_t>(M64) * K);
    for (float& v : a64) v = nd(rng);
    std::vector<float> c64(static_cast<size_t>(M64) * N64, 0.0f);
    ob_gemm_f32_quant_job_t j64{};
    j64.b_nk = w_raw.data();
    j64.c_mn = c64.data();
    j64.n_dim = N64;
    ob_gemm_f32_quant_job_set_type(&j64, qt);
    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a64.data(), M64, K, &j64, 1u, nullptr) == OB_OK);
    // Inspect cached lt_64x ext[2] tile
    ob::CachedWeightTile* wt64 = ctx->weight_cache_lru.front();
    CHECK(wt64 != nullptr && wt64->ext2_buf != nullptr && wt64->hw_k == K && wt64->hw_n == N64);
    std::vector<int8_t> bq64;
    unpack_ext2_w8a8_tile(static_cast<const uint8_t*>(ob_buffer_data(wt64->ext2_buf)),
                          N64, K, /*is_lt64=*/true, &bq64);
    std::vector<float> y64_ref(static_cast<size_t>(M64) * N64, 0.0f);
    for (uint32_t m = 0u; m < M64; ++m) {
      for (uint32_t n = 0u; n < N64; ++n) {
        double acc = 0.0;
        for (uint32_t k = 0u; k < K; ++k) {
          acc += static_cast<double>(a64[static_cast<size_t>(m) * K + k]) *
                 static_cast<double>(w_ref[static_cast<size_t>(n) * K + k]);
        }
        y64_ref[static_cast<size_t>(m) * N64 + n] = static_cast<float>(acc);
      }
    }
    CHECK(compute_nmse(c64, y64_ref) < 1e-6);

    // Verify small-tile sub-superblock slicing (M=4, K=256, N=64 -> hw_m=8, tk=64 inside 256-weight super-blocks!)
    {
      const uint32_t Ms = 4u, Ks = 256u, Ns = 64u;
      const auto ws_raw = make_random_quant_matrix(qt, Ns, Ks, 7000u + ti, false);
      const auto ws_ref = dequant_matrix_ref(qt, ws_raw, Ns, Ks);
      std::vector<float> as(static_cast<size_t>(Ms) * Ks);
      for (float& v : as) v = nd(rng);
      std::vector<float> cs(static_cast<size_t>(Ms) * Ns, 0.0f);
      std::vector<float> cs_ref(static_cast<size_t>(Ms) * Ns, 0.0f);
      for (uint32_t m = 0u; m < Ms; ++m) {
        for (uint32_t n = 0u; n < Ns; ++n) {
          double acc = 0.0;
          for (uint32_t k = 0u; k < Ks; ++k) {
            acc += static_cast<double>(as[static_cast<size_t>(m) * Ks + k]) *
                   static_cast<double>(ws_ref[static_cast<size_t>(n) * Ks + k]);
          }
          cs_ref[static_cast<size_t>(m) * Ns + n] = static_cast<float>(acc);
        }
      }
      ob_gemm_f32_quant_job_t js{};
      js.b_nk = ws_raw.data();
      js.c_mn = cs.data();
      js.n_dim = Ns;
      ob_gemm_f32_quant_job_set_type(&js, qt);
      CHECK(ob_gemm_matmul_f32_quant_batch(ctx, as.data(), Ms, Ks, &js, 1u, nullptr) == OB_OK);
      CHECK(compute_nmse(cs, cs_ref) < 1e-6);
    }

    CHECK(ob_gemm_unregister_immutable_weights(ctx, w_raw.data(), w_raw.size()) == OB_OK);
  }

  // Test 2: Mixed-quant QKV horizontal fusion (Q=Q4_K, K=Q4_K, V=Q6_K -> single 1x1024x1536 fused tile!)
  {
    constexpr uint32_t K = 1024u, Nq = 1024u, Nk = 256u, Nv = 256u;
    auto wq = make_random_quant_matrix(OB_QUANT_Q4_K, Nq, K, 8101u, true);
    auto wk = make_random_quant_matrix(OB_QUANT_Q4_K, Nk, K, 8102u, true);
    auto wv = make_random_quant_matrix(OB_QUANT_Q6_K, Nv, K, 8103u, false);
    // Keep Q6_K super-block d and sub-block scale magnitudes tightly clustered
    // (like real LLM v_proj weights) so Q4_K+Q4_K+Q6_K have matched normalized L2
    // and satisfy the horizontal-fusion scale_d spread guard (<= 1.5x).
    for (uint32_t r = 0u; r < Nv; ++r) {
      const uint32_t nb = K / OB_QK_K;
      for (uint32_t b = 0u; b < nb; ++b) {
        auto* blk = reinterpret_cast<ob_block_q6_k_t*>(
            wv.data() + (static_cast<size_t>(r) * nb + b) * sizeof(ob_block_q6_k_t));
        const uint16_t dh = f32_to_f16(0.004f);
        std::memcpy(&blk->d, &dh, 2u);
        for (uint32_t j = 0u; j < 16u; ++j) {
          const int8_t sign = (blk->scales[j] < 0) ? -1 : 1;
          blk->scales[j] = static_cast<int8_t>(sign * (48 + static_cast<int>((r + b + j) & 7u)));
        }
      }
    }
    const auto wq_f = dequant_matrix_ref(OB_QUANT_Q4_K, wq, Nq, K);
    const auto wk_f = dequant_matrix_ref(OB_QUANT_Q4_K, wk, Nk, K);
    const auto wv_f = dequant_matrix_ref(OB_QUANT_Q6_K, wv, Nv, K);

    std::vector<float> a(K);
    for (float& v : a) v = nd(rng);
    std::vector<float> cq(Nq, 0.0f), ck(Nk, 0.0f), cv(Nv, 0.0f);
    ob_gemm_f32_quant_job_t jobs[3]{};
    jobs[0] = {wq.data(), cq.data(), Nq, OB_ACT_NONE, static_cast<uint8_t>(OB_QUANT_Q4_K), {0, 0, 0}};
    jobs[1] = {wk.data(), ck.data(), Nk, OB_ACT_NONE, static_cast<uint8_t>(OB_QUANT_Q4_K), {0, 0, 0}};
    jobs[2] = {wv.data(), cv.data(), Nv, OB_ACT_NONE, static_cast<uint8_t>(OB_QUANT_Q6_K), {0, 0, 0}};

    const uint64_t hf_before = ctx->prof.horiz_fusions;
    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a.data(), 1u, K, jobs, 3u, nullptr) == OB_OK);
    CHECK(ctx->prof.horiz_fusions > hf_before);

    // Check fused cached tile ext[2] simulated W8A8 accuracy across Q (asymmetric Q4_K), K (Q4_K), V (Q6_K)!
    ob::CachedWeightTile* wt = ctx->weight_cache_lru.front();
    const uint32_t Ntot = Nq + Nk + Nv;
    CHECK(wt != nullptr && wt->ext2_buf != nullptr && wt->hw_n == Ntot);
    CHECK(wt->row_s_b.size() == 2u * Ntot);
    // V slice (Q6_K) offsets must be strictly 0.0f
    for (uint32_t i = Nq + Nk; i < Ntot; ++i) CHECK(wt->row_s_b[Ntot + i] == 0.0f);

    std::vector<int8_t> bq_f;
    unpack_ext2_w8a8_tile(static_cast<const uint8_t*>(ob_buffer_data(wt->ext2_buf)),
                          Ntot, K, false, &bq_f);
    std::vector<int8_t> aq(K, 0);
    float sa = 0.0f;
    ob::quantize_activation_row_split(a.data(), K, aq.data(), &sa, nullptr);
    const uint32_t rms_shift = ob::compute_activation_rms_shift(aq.data(), 1u, K, K);
    const uint32_t eff_d = (rms_shift > 0u)
        ? ob::quant_scale_D_grid(static_cast<double>(std::max<uint32_t>(1u, wt->scale_d >> rms_shift)))
        : wt->scale_d;
    double a_sum = 0.0;
    for (float v : a) a_sum += static_cast<double>(v);

    std::vector<float> y_hw(Ntot, 0.0f), y_ref(Ntot, 0.0f);
    for (uint32_t n = 0u; n < Ntot; ++n) {
      const float* w_ptr = (n < Nq) ? (wq_f.data() + static_cast<size_t>(n) * K)
                                    : ((n < Nq + Nk) ? (wk_f.data() + static_cast<size_t>(n - Nq) * K)
                                                     : (wv_f.data() + static_cast<size_t>(n - Nq - Nk) * K));
      double ref_acc = 0.0;
      int32_t q_acc = 0;
      for (uint32_t k = 0u; k < K; ++k) {
        ref_acc += static_cast<double>(a[k]) * w_ptr[k];
        q_acc += static_cast<int32_t>(aq[k]) * static_cast<int32_t>(bq_f[static_cast<size_t>(n) * K + k]);
      }
      int32_t qo = static_cast<int32_t>(std::lrint(static_cast<double>(q_acc) / static_cast<double>(eff_d)));
      if (qo < -128) qo = -128;
      if (qo > 127) qo = 127;
      y_ref[n] = static_cast<float>(ref_acc);
      y_hw[n] = static_cast<float>(qo) * (sa * static_cast<float>(eff_d) * wt->row_s_b[n]) +
                wt->row_s_b[Ntot + n] * static_cast<float>(a_sum);
    }
    const double qkv_nmse = compute_nmse(y_hw, y_ref);
    std::printf("  [OK] Mixed QKV (Q4_K+Q4_K+Q6_K) fused ext2 NMSE=%.3e\n", qkv_nmse);
    CHECK(qkv_nmse < 1e-3);
  }

  ob_context_destroy(ctx);
  std::printf("[PASS] test_gemm_cache_ext2_w8a8_and_w8a16_accuracy\n");
}

void test_swiglu_modern_quants() {
  std::vector<uint8_t> synth_interp(65536u, 0x7Au);
  TempFile interp_file;
  CHECK(interp_file.write(synth_interp.data(), synth_interp.size()));

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);
  CHECK(ob_gemm_register_builtin_emitter(ctx, interp_file.path()) == OB_OK);

  std::mt19937 rng(5555u);
  std::normal_distribution<float> nd(0.0f, 0.5f);

  struct SwigluCase {
    ob_quant_type_t g;
    ob_quant_type_t u;
    ob_quant_type_t d;
    uint32_t M;
    uint32_t K;
    uint32_t H;
    uint32_t N;
    bool asym_down;
  };
  const SwigluCase cases[] = {
      {OB_QUANT_Q4_K,   OB_QUANT_Q4_K,   OB_QUANT_Q4_K,   1u,  512u,  512u,  512u, false},
      {OB_QUANT_Q6_K,   OB_QUANT_Q6_K,   OB_QUANT_Q6_K,   1u,  512u,  512u,  512u, false},
      {OB_QUANT_IQ4_NL, OB_QUANT_IQ4_NL, OB_QUANT_IQ4_NL, 1u,  512u,  512u,  512u, false},
      {OB_QUANT_IQ4_XS, OB_QUANT_IQ4_XS, OB_QUANT_IQ4_XS, 1u,  512u,  512u,  512u, false},
      // Mixed Q4_K_M realistic MLP (gate=Q4_K, up=Q4_K, down=Q6_K) at M=1 (chained-eligible K=1536, N=1536) and M=4!
      {OB_QUANT_Q4_K,   OB_QUANT_Q4_K,   OB_QUANT_Q6_K,   1u, 1536u, 1024u, 1536u, false},
      {OB_QUANT_Q4_K,   OB_QUANT_Q4_K,   OB_QUANT_Q6_K,   4u,  512u,  512u,  512u, false},
      // Skewed asymmetric Q4_K W_down at chained-eligible K=1536, N=1536 -> verifies DC guard + fallback accuracy!
      {OB_QUANT_Q4_K,   OB_QUANT_Q4_K,   OB_QUANT_Q4_K,   1u, 1536u,  512u, 1536u, true},
  };

  for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
    const SwigluCase& sc = cases[ci];
    const auto wg = make_random_quant_matrix(sc.g, sc.H, sc.K, 9000u + static_cast<uint32_t>(ci) * 3u + 0u, false);
    const auto wu = make_random_quant_matrix(sc.u, sc.H, sc.K, 9000u + static_cast<uint32_t>(ci) * 3u + 1u, false);
    const auto wd = make_random_quant_matrix(sc.d, sc.N, sc.H, 9000u + static_cast<uint32_t>(ci) * 3u + 2u, sc.asym_down);

    const auto wg_f = dequant_matrix_ref(sc.g, wg, sc.H, sc.K);
    const auto wu_f = dequant_matrix_ref(sc.u, wu, sc.H, sc.K);
    const auto wd_f = dequant_matrix_ref(sc.d, wd, sc.N, sc.H);

    std::vector<float> a(static_cast<size_t>(sc.M) * sc.K);
    for (float& v : a) v = nd(rng);
    std::vector<float> c_got(static_cast<size_t>(sc.M) * sc.N, 0.0f);
    std::vector<float> c_ref(static_cast<size_t>(sc.M) * sc.N, 0.0f);

    for (uint32_t m = 0u; m < sc.M; ++m) {
      std::vector<double> mid(sc.H, 0.0);
      for (uint32_t h = 0u; h < sc.H; ++h) {
        double g = 0.0, u = 0.0;
        for (uint32_t k = 0u; k < sc.K; ++k) {
          const double ak = static_cast<double>(a[static_cast<size_t>(m) * sc.K + k]);
          g += ak * static_cast<double>(wg_f[static_cast<size_t>(h) * sc.K + k]);
          u += ak * static_cast<double>(wu_f[static_cast<size_t>(h) * sc.K + k]);
        }
        mid[h] = (g / (1.0 + std::exp(-g))) * u;
      }
      for (uint32_t n = 0u; n < sc.N; ++n) {
        double acc = 0.0;
        for (uint32_t h = 0u; h < sc.H; ++h) {
          acc += mid[h] * static_cast<double>(wd_f[static_cast<size_t>(n) * sc.H + h]);
        }
        c_ref[static_cast<size_t>(m) * sc.N + n] = static_cast<float>(acc);
      }
    }

    ob_gemm_swiglu_quant_desc_t desc{};
    desc.w_gate_hk = wg.data();
    desc.w_up_hk = wu.data();
    desc.w_down_nh = wd.data();
    desc.m_dim = sc.M;
    desc.k_dim = sc.K;
    desc.h_dim = sc.H;
    desc.n_dim = sc.N;
    ob_gemm_swiglu_quant_desc_set_types(&desc, sc.g, sc.u, sc.d);

    ob_invoke_stats_t stats{};
    CHECK(ob_gemm_swiglu_f32_quant(ctx, a.data(), &desc, c_got.data(), &stats) == OB_OK);
    const double nmse = compute_nmse(c_got, c_ref);
    CHECK(nmse < 1e-3);
  }

  ob_context_destroy(ctx);
  std::printf("[PASS] test_swiglu_modern_quants (7/7 cases < 1e-3 NMSE)\n");
}

}  // namespace

int main() {
  test_abi_and_validation();
  test_bit_exact_dequant_and_column_extract();
  test_q4_k_algebraic_dmin_centering();
  test_gemm_cache_ext2_w8a8_and_w8a16_accuracy();
  test_swiglu_modern_quants();

  if (g_failures != 0) {
    std::fprintf(stderr, "FAILED: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("ALL OB_QUANT_UNPACK TESTS PASSED\n");
  return 0;
}
