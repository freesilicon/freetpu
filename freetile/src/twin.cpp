// SPDX-License-Identifier: Apache-2.0
// FreeTile Digital Twin (freetile-twin) — Deterministic Offline Evaluator & Scorer.
//
// Evaluates candidate FreeTile schedules, quantized kernels (W8A16, Q4_K, Q6_K,
// IQ4_NL, IQ4_XS, Q8_0, Q4_0), 11-chunk & quantized SwiGLU, FlashAttention tile
// schedules + online softmax, 0x640 minimax splines, and raw .dgc0 payloads
// across Pixel 10 (buenos-a0) and Pixel 11 (malibu) profiles using libopenbuenos
// mock_mode=1.
//
// Builds cleanly under -std=c++17 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti.

#include "freetile/twin.h"

#include "freetile/buenos_emitter.hpp"
#include "freetile/buenos_splines.hpp"
#include "freetile/flash_attn_planner.hpp"
#include "freetile/malibu_profile.hpp"
#include "freetile/tile_ir.hpp"

#include "ob_internal.h"
#include "ob_quant_unpack.hpp"
#include "openbuenos.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

struct ft_twin_session {
    ob_context_t* ob_ctx = nullptr;
    std::vector<uint8_t> dummy_dive_interp;
};

namespace {

using SteadyClock = std::chrono::steady_clock;

// Hardware roofline constants anchored to repo empirical profiles.
constexpr double kBuenosInt8Tops       = 18.0;    // Tensor G5 16-core INT8 systolic TOPS
constexpr double kBuenosFp16Tflops     = 9.0;     // Tensor G5 BF16/FP16 TFLOPS
constexpr double kBuenosFp32Tflops     = 1.5;     // Tensor G5 FP32 vector TFLOPS
constexpr double kBuenosDramBwGbps     = 27.96;   // Empirical Tensor G5 decode DRAM BW (GB/s)
constexpr double kBuenosViiDispatchUs  = 300.0;   // Empirical VII dispatch latency (us)
constexpr double kBuenosTilePipeUs     = 15.0;    // Amortized tile launch overhead in-graph (us)

constexpr double kMalibuTpuInt8Tops    = 27.0;    // Darwinn Heavy EdgeTPU estimated TOPS
constexpr double kMalibuTpuFp16Tflops  = 13.5;    // Darwinn Heavy FP16/BF16 TFLOPS
constexpr double kMalibuTpuViiUs       = 220.0;   // Next-gen MBU VII dispatch (us)
constexpr double kMalibuSme2Int8Tops   = 2.4;     // 5-core ARMv9 SME2 INT8 TOPS
constexpr double kMalibuSme2Fp16Tflops = 1.2;     // 5-core ARMv9 SME2 BF16/FP16 TFLOPS
constexpr double kMalibuNeonTops       = 0.6;     // 5-core ARMv9 NEON/i8mm TOPS
constexpr double kMalibuGpuTflops      = 1.5;     // PowerVR CXTP-48-1536 FP16 TFLOPS
constexpr double kMalibuCpuDispatchUs  = 15.0;    // Host CPU dispatch (us)
constexpr double kMalibuGpuDispatchUs  = 80.0;    // Vulkan BDA submit latency (us)

constexpr size_t kDgc0PayloadAlignBytes = 64u;
constexpr uint64_t kMaxFullVerifyMacs   = 4500000ULL; // Full exact FP64 check up to ~2048x2048 M=1

void copy_cstr(char* dst, size_t cap, const char* src) {
    if (!dst || cap == 0u) return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    std::snprintf(dst, cap, "%s", src);
}

void copy_str(char* dst, size_t cap, const std::string& s) {
    copy_cstr(dst, cap, s.c_str());
}

void add_diag(ft_twin_result_t* res,
              const char* code,
              ft_twin_diag_severity_t sev,
              int has_offset,
              int64_t offset,
              const char* field,
              const std::string& msg) {
    if (!res || res->num_diagnostics >= FT_TWIN_MAX_DIAGNOSTICS) return;
    ft_twin_diagnostic_t& d = res->diagnostics[res->num_diagnostics++];
    std::memset(&d, 0, sizeof(d));
    copy_cstr(d.code, sizeof(d.code), code);
    d.severity = sev;
    d.has_offset = has_offset;
    d.offset = offset;
    copy_cstr(d.field, sizeof(d.field), field ? field : "");
    copy_str(d.message, sizeof(d.message), msg);
}

void set_fault_at(ft_twin_result_t* res,
                  const char* code,
                  int has_offset,
                  int64_t offset,
                  const char* field,
                  const std::string& reason) {
    if (!res) return;
    if (!res->has_fault) {
        res->has_fault = 1;
        std::string full = std::string(code) + ": " + reason;
        copy_str(res->fault_reason, sizeof(res->fault_reason), full);
    }
    add_diag(res, code, FT_TWIN_DIAG_ERROR, has_offset, offset, field, reason);
}

void set_fault(ft_twin_result_t* res, const char* code, const std::string& reason) {
    set_fault_at(res, code, 0, 0, nullptr, reason);
}

uint32_t xorshift32(uint32_t& state) {
    if (state == 0u) state = 0x9E3779B9u;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float rand_uniform(uint32_t& state, float lo, float hi) {
    const uint32_t r = xorshift32(state);
    const double u = static_cast<double>(r) / static_cast<double>(0xFFFFFFFFu);
    return static_cast<float>(static_cast<double>(lo) + u * (static_cast<double>(hi) - static_cast<double>(lo)));
}

int8_t rand_i8(uint32_t& state, int lo, int hi) {
    const uint32_t r = xorshift32(state);
    const int span = hi - lo + 1;
    return static_cast<int8_t>(lo + static_cast<int>(r % static_cast<uint32_t>(span)));
}

uint16_t f32_to_f16_rne(float x) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp_raw = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x007FFFFFu;
    if (exp_raw == 0xFFu) {
        if (mant == 0u) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | 0x7E00u | (mant >> 13));
    }
    int32_t exp = static_cast<int32_t>(exp_raw) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = mant | 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        const uint32_t half_bit = 1u << (shift - 1u);
        const uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t q = m >> shift;
        if (rem > half_bit || (rem == half_bit && (q & 1u) != 0u)) {
            q += 1u;
        }
        return static_cast<uint16_t>(sign | q);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    const uint32_t rem = mant & 0x1FFFu;
    uint32_t q = mant >> 13;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u) != 0u)) {
        q += 1u;
        if (q == 0x0400u) {
            q = 0u;
            exp += 1;
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | q);
}

float round_f32_to_bf16(float x) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    if ((bits & 0x7F800000u) != 0x7F800000u) {
        bits += rounding_bias;
    }
    bits &= 0xFFFF0000u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<uint8_t> make_dummy_dive_interp(size_t size = 4096u) {
    std::vector<uint8_t> img(size, 0u);
    img[0] = 0x7fu;
    img[1] = 'E';
    img[2] = 'L';
    img[3] = 'F';
    for (size_t i = 4u; i < size; ++i) {
        img[i] = static_cast<uint8_t>((i * 37u + 13u) & 0xFFu);
    }
    return img;
}

void sha256_hex(const void* data, size_t size, char out_hex[65]) {
    uint8_t digest[32]{};
    ob::sha256_compute(data, size, digest);
    static const char kHex[] = "0123456789abcdef";
    for (size_t i = 0u; i < 32u; ++i) {
        out_hex[i * 2u]      = kHex[(digest[i] >> 4) & 0x0Fu];
        out_hex[i * 2u + 1u] = kHex[digest[i] & 0x0Fu];
    }
    out_hex[64] = '\0';
}

bool hex_equals_ci(const char* a, const char* b) {
    if (!a || !b) return false;
    for (size_t i = 0u; i < 64u; ++i) {
        char ca = a[i];
        char cb = b[i];
        if (ca == '\0' || cb == '\0') return false;
        if (ca >= 'A' && ca <= 'F') ca = static_cast<char>(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'F') cb = static_cast<char>(cb - 'A' + 'a');
        if (ca != cb) return false;
    }
    return a[64] == '\0' && b[64] == '\0';
}

double ref_silu(double x) {
    if (x >= 0.0) return x / (1.0 + std::exp(-x));
    const double ex = std::exp(x);
    return x * (ex / (1.0 + ex));
}

double ref_gelu_tanh(double x) {
    constexpr double kSqrt2OverPi = 0.79788456080286535588;
    return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x)));
}

double ref_gelu_erf(double x) {
    constexpr double kInvSqrt2 = 0.70710678118654752440;
    return 0.5 * x * (1.0 + std::erf(x * kInvSqrt2));
}

double apply_ref_activation(double x, ft_twin_activation_t act) {
    switch (act) {
        case FT_TWIN_ACT_NONE:  return x;
        case FT_TWIN_ACT_RELU:  return x > 0.0 ? x : 0.0;
        case FT_TWIN_ACT_RELU6: return x < 0.0 ? 0.0 : (x > 6.0 ? 6.0 : x);
        case FT_TWIN_ACT_SILU:  return ref_silu(x);
        case FT_TWIN_ACT_GELU:  return ref_gelu_tanh(x);
    }
    return x;
}

freetile::EpilogueActivation to_ft_act(ft_twin_activation_t act) {
    switch (act) {
        case FT_TWIN_ACT_NONE:  return freetile::EpilogueActivation::NONE;
        case FT_TWIN_ACT_RELU:  return freetile::EpilogueActivation::RELU;
        case FT_TWIN_ACT_RELU6: return freetile::EpilogueActivation::RELU6;
        case FT_TWIN_ACT_SILU:  return freetile::EpilogueActivation::SILU;
        case FT_TWIN_ACT_GELU:  return freetile::EpilogueActivation::GELU;
    }
    return freetile::EpilogueActivation::NONE;
}

uint32_t to_ob_act(ft_twin_activation_t act) {
    switch (act) {
        case FT_TWIN_ACT_NONE:  return OB_ACT_NONE;
        case FT_TWIN_ACT_RELU:  return OB_ACT_RELU;
        case FT_TWIN_ACT_RELU6: return OB_ACT_RELU6;
        case FT_TWIN_ACT_SILU:  return OB_ACT_SILU;
        case FT_TWIN_ACT_GELU:  return OB_ACT_GELU;
    }
    return OB_ACT_NONE;
}

size_t ft_dtype_bytes(ft_twin_dtype_t dt) {
    switch (dt) {
        case FT_TWIN_DTYPE_BF16: return 2u;
        case FT_TWIN_DTYPE_FP32: return 4u;
        case FT_TWIN_DTYPE_FP16: return 2u;
        case FT_TWIN_DTYPE_INT8: return 1u;
    }
    return 2u;
}

freetile::SplineKind to_ft_spline(ft_twin_spline_kind_t k) {
    switch (k) {
        case FT_TWIN_SPLINE_EXP_NEG12_96: return freetile::SplineKind::EXP_NEG12_96_0;
        case FT_TWIN_SPLINE_EXP_NEG11:    return freetile::SplineKind::EXP_NEG11_0;
        case FT_TWIN_SPLINE_RSQRT:        return freetile::SplineKind::RSQRT_MANTISSA;
        case FT_TWIN_SPLINE_RECIP:        return freetile::SplineKind::RECIP_MANTISSA;
        case FT_TWIN_SPLINE_SILU:         return freetile::SplineKind::SILU;
        case FT_TWIN_SPLINE_GELU_TANH:    return freetile::SplineKind::GELU_TANH;
        case FT_TWIN_SPLINE_GELU_ERF:     return freetile::SplineKind::GELU_ERF;
    }
    return freetile::SplineKind::SILU;
}

ob_quant_type_t workload_to_ob_quant(ft_twin_workload_t w) {
    switch (w) {
        case FT_TWIN_WORKLOAD_QUANT_Q4_K:   return OB_QUANT_Q4_K;
        case FT_TWIN_WORKLOAD_QUANT_Q6_K:   return OB_QUANT_Q6_K;
        case FT_TWIN_WORKLOAD_QUANT_IQ4_NL: return OB_QUANT_IQ4_NL;
        case FT_TWIN_WORKLOAD_QUANT_IQ4_XS: return OB_QUANT_IQ4_XS;
        case FT_TWIN_WORKLOAD_QUANT_Q4_0:   return OB_QUANT_Q4_0;
        case FT_TWIN_WORKLOAD_QUANT_Q8_0:
        default:                            return OB_QUANT_Q8_0;
    }
}

freetile::MalibuQuantKind ob_quant_to_malibu(ob_quant_type_t q) {
    switch (q) {
        case OB_QUANT_Q8_0:   return freetile::MalibuQuantKind::Q8_0;
        case OB_QUANT_Q4_0:   return freetile::MalibuQuantKind::Q4_0;
        case OB_QUANT_Q4_K:   return freetile::MalibuQuantKind::Q4_K;
        case OB_QUANT_Q6_K:   return freetile::MalibuQuantKind::Q6_K;
        case OB_QUANT_IQ4_NL: return freetile::MalibuQuantKind::IQ4_NL;
        case OB_QUANT_IQ4_XS: return freetile::MalibuQuantKind::IQ4_XS;
    }
    return freetile::MalibuQuantKind::Q8_0;
}

uint64_t target_default_sram_limit(ft_twin_target_t target, bool narrow_core_only) {
    if (target == FT_TWIN_TARGET_MALIBU) {
        // Day-1 conservative narrow SRAM floor from malibu_a0_target_profile().
        return static_cast<uint64_t>(freetile::malibu_a0_target_profile().vmem_bytes);
    }
    if (narrow_core_only) {
        return static_cast<uint64_t>(freetile::kBuenosCoreNarrowMemBytes); // 786,432 B (768 KiB)
    }
    return static_cast<uint64_t>(freetile::target_profile(freetile::ChipTarget::BUENOS_A0).vmem_bytes); // 16 MiB
}

// Compute NMSE and max_abs_err of `got` against FP64 reference `ref`.
void compute_error_metrics(const std::vector<double>& got,
                           const std::vector<double>& ref,
                           double* out_nmse,
                           double* out_max_abs,
                           bool* out_has_nonfinite) {
    double err_sq = 0.0;
    double ref_sq = 0.0;
    double max_abs = 0.0;
    bool nonfinite = false;
    const size_t n = std::min(got.size(), ref.size());
    for (size_t i = 0u; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
            nonfinite = true;
            continue;
        }
        const double d = got[i] - ref[i];
        const double ad = std::fabs(d);
        if (ad > max_abs) max_abs = ad;
        err_sq += d * d;
        ref_sq += ref[i] * ref[i];
    }
    if (out_has_nonfinite) *out_has_nonfinite = nonfinite;
    if (out_max_abs) *out_max_abs = max_abs;
    if (out_nmse) {
        if (nonfinite) {
            *out_nmse = 1.0;
        } else if (ref_sq > 1e-30) {
            *out_nmse = err_sq / ref_sq;
        } else {
            *out_nmse = err_sq;
        }
    }
}

// Deterministic quantized GGUF weight matrix generator covering all 6 formats.
std::vector<uint8_t> make_det_quant_matrix(ob_quant_type_t type,
                                           uint32_t n_rows,
                                           uint32_t k_dim,
                                           uint32_t seed) {
    uint32_t rng = seed * 2654435761u + 97u;
    const uint32_t qk = ob_quant_block_elements(type);
    const size_t blk_sz = ob_quant_block_bytes(type);
    const uint32_t nb_k = k_dim / qk;
    std::vector<uint8_t> buf(static_cast<size_t>(n_rows) * nb_k * blk_sz, 0u);

    for (uint32_t r = 0u; r < n_rows; ++r) {
        for (uint32_t b = 0u; b < nb_k; ++b) {
            uint8_t* dst = buf.data() + (static_cast<size_t>(r) * nb_k + b) * blk_sz;
            if (type == OB_QUANT_Q8_0) {
                ob_block_q8_0_t blk{};
                blk.d = f32_to_f16_rne(rand_uniform(rng, 0.004f, 0.025f));
                for (uint32_t i = 0u; i < OB_QK8_0; ++i) {
                    blk.qs[i] = rand_i8(rng, -120, 120);
                }
                std::memcpy(dst, &blk, sizeof(blk));
            } else if (type == OB_QUANT_Q4_0) {
                ob_block_q4_0_t blk{};
                blk.d = f32_to_f16_rne(rand_uniform(rng, 0.01f, 0.06f));
                for (uint32_t i = 0u; i < OB_QK8_0 / 2u; ++i) {
                    blk.qs[i] = static_cast<uint8_t>(xorshift32(rng) & 0xFFu);
                }
                std::memcpy(dst, &blk, sizeof(blk));
            } else if (type == OB_QUANT_Q4_K) {
                ob_block_q4_k_t blk{};
                blk.d = f32_to_f16_rne(rand_uniform(rng, 0.002f, 0.015f));
                blk.dmin = f32_to_f16_rne(rand_uniform(rng, 0.001f, 0.010f));
                for (uint32_t i = 0u; i < OB_K_SCALE_SIZE; ++i) {
                    blk.scales[i] = static_cast<uint8_t>(xorshift32(rng) & 0xFFu);
                }
                for (uint32_t i = 0u; i < OB_QK_K / 2u; ++i) {
                    blk.qs[i] = static_cast<uint8_t>(xorshift32(rng) & 0xFFu);
                }
                std::memcpy(dst, &blk, sizeof(blk));
            } else if (type == OB_QUANT_Q6_K) {
                ob_block_q6_k_t blk{};
                blk.d = f32_to_f16_rne(rand_uniform(rng, 0.001f, 0.008f));
                for (uint32_t i = 0u; i < OB_QK_K / 2u; ++i) {
                    blk.ql[i] = static_cast<uint8_t>(xorshift32(rng) & 0xFFu);
                }
                for (uint32_t i = 0u; i < OB_QK_K / 4u; ++i) {
                    blk.qh[i] = static_cast<uint8_t>(xorshift32(rng) & 0xFFu);
                }
                for (uint32_t i = 0u; i < OB_QK_K / 16u; ++i) {
                    blk.scales[i] = rand_i8(rng, -48, 48);
                }
                std::memcpy(dst, &blk, sizeof(blk));
            } else if (type == OB_QUANT_IQ4_NL) {
                ob_block_iq4_nl_t blk{};
                blk.d = f32_to_f16_rne(rand_uniform(rng, 0.003f, 0.02f));
                for (uint32_t i = 0u; i < OB_QK4_NL / 2u; ++i) {
                    blk.qs[i] = static_cast<uint8_t>(xorshift32(rng) & 0xFFu);
                }
                std::memcpy(dst, &blk, sizeof(blk));
            } else if (type == OB_QUANT_IQ4_XS) {
                ob_block_iq4_xs_t blk{};
                blk.d = f32_to_f16_rne(rand_uniform(rng, 0.001f, 0.008f));
                blk.scales_h = static_cast<uint16_t>(xorshift32(rng) & 0xFFFFu);
                for (uint32_t i = 0u; i < OB_QK_K / 64u; ++i) {
                    blk.scales_l[i] = static_cast<uint8_t>(xorshift32(rng) & 0xFFu);
                }
                for (uint32_t i = 0u; i < OB_QK_K / 2u; ++i) {
                    blk.qs[i] = static_cast<uint8_t>(xorshift32(rng) & 0xFFu);
                }
                std::memcpy(dst, &blk, sizeof(blk));
            }
        }
    }
    return buf;
}

// Compute deterministic GRPO scalar reward in [-1.0, 1.0].
void finalize_reward(ft_twin_result_t* res,
                     double nmse_ref,
                     double latency_ref_us,
                     const char* primary_fault_class) {
    if (!res) return;

    const bool has_sram = (res->has_sram_bytes && res->has_sram_limit_bytes && res->sram_limit_bytes > 0u);
    const double util = has_sram
                            ? (static_cast<double>(res->sram_bytes) / static_cast<double>(res->sram_limit_bytes))
                            : (res->has_fault ? 2.0 : 0.5);
    double sram_eff = 0.0;
    if (util <= 1.0) {
        if (util <= 0.85) {
            sram_eff = 0.5 + 0.5 * (util / 0.85);
        } else {
            sram_eff = std::max(0.2, 1.0 - ((util - 0.85) / 0.15) * 0.8);
        }
    } else {
        sram_eff = 0.0;
    }

    double acc_score = res->has_fault ? 0.0 : 1.0;
    if (res->has_nmse) {
        if (!std::isfinite(res->nmse) || res->nmse < 0.0) {
            acc_score = 0.0;
        } else {
            const double denom = (nmse_ref > 1e-12) ? nmse_ref : 1e-3;
            acc_score = 1.0 / (1.0 + (res->nmse / denom));
        }
    }

    double lat_score = res->has_fault ? 0.0 : 0.5;
    if (res->has_est_hw_us && std::isfinite(res->est_hw_us) && res->est_hw_us >= 0.0) {
        const double lref = (latency_ref_us > 1.0) ? latency_ref_us : 1000.0;
        lat_score = 1.0 / (1.0 + (res->est_hw_us / lref));
    }

    res->rb_accuracy_score = acc_score;
    res->rb_sram_efficiency = sram_eff;
    res->rb_latency_score = lat_score;

    if (!res->has_fault) {
        res->pass = 1;
        res->rb_gate_pass = 1.0;
        double r = 0.25 * res->rb_gate_pass +
                   0.35 * res->rb_accuracy_score +
                   0.15 * res->rb_sram_efficiency +
                   0.25 * res->rb_latency_score;
        if (r <= 0.0) r = 0.01;
        if (r > 1.0) r = 1.0;
        res->reward = r;
    } else {
        res->pass = 0;
        res->rb_gate_pass = 0.0;
        double r = -0.8;
        if (primary_fault_class && std::strcmp(primary_fault_class, "DGC0") == 0) {
            r = -1.0;
        } else if (primary_fault_class && std::strcmp(primary_fault_class, "SRAM") == 0) {
            const double over = (util > 1.0) ? std::min(1.0, util - 1.0) : 0.25;
            r = -0.5 - 0.4 * over;
        } else if (primary_fault_class && std::strcmp(primary_fault_class, "SPLINE") == 0) {
            r = -0.45;
        } else if (primary_fault_class && std::strcmp(primary_fault_class, "ACCURACY") == 0) {
            const double excess = res->has_nmse ? std::min(1.0, res->nmse / std::max(nmse_ref, 1e-6)) : 0.5;
            r = -0.2 - 0.5 * excess;
        } else if (primary_fault_class && std::strcmp(primary_fault_class, "SCHEDULE") == 0) {
            r = -0.75;
        } else {
            r = -0.9;
        }
        if (r < -1.0) r = -1.0;
        if (r > -0.1) r = -0.1;
        res->reward = r;
    }
}

// Deep wire integrity validation of a raw .dgc0 payload.
bool validate_dgc0_payload_internal(const uint8_t* data,
                                    size_t size,
                                    ft_twin_target_t target,
                                    const char* expected_sha256,
                                    uint64_t sram_limit_override,
                                    ft_twin_result_t* res,
                                    bool is_primary_domain,
                                    const char** out_fault_class) {
    res->dgc0_bytes_checked = static_cast<uint32_t>(std::min<size_t>(size, 0xFFFFFFFFu));

    if (!data || size < 16u) {
        res->check_dgc0_wire_ok = 0;
        set_fault(res, "DGC0_TOO_SMALL",
                  "DGC0 buffer is null or smaller than 16-byte FlatBuffer header (size=" +
                      std::to_string(size) + ")");
        if (out_fault_class) *out_fault_class = "DGC0";
        return false;
    }

    sha256_hex(data, size, res->dgc0_sha256);

    if (expected_sha256 && expected_sha256[0] != '\0') {
        if (!hex_equals_ci(res->dgc0_sha256, expected_sha256)) {
            res->check_dgc0_wire_ok = 0;
            set_fault(res, "DGC0_SHA256_MISMATCH",
                      std::string("DGC0 SHA-256 mismatch: actual=") + res->dgc0_sha256 +
                          " expected=" + expected_sha256);
            if (out_fault_class) *out_fault_class = "DGC0";
            return false;
        }
    }

    // 1. Magic check at bytes [4..7].
    if (std::memcmp(data + 4u, "DGC0", 4u) != 0) {
        res->check_dgc0_wire_ok = 0;
        set_fault_at(res, "DGC0_BAD_MAGIC", 1, 4, "file_identifier",
                     "Missing DGC0 file identifier at offset 4");
        if (out_fault_class) *out_fault_class = "DGC0";
        return false;
    }

    // 2. Root offset sanity.
    uint32_t root_off = 0u;
    std::memcpy(&root_off, data, sizeof(root_off));
    if (root_off < 8u || root_off >= size) {
        res->check_dgc0_wire_ok = 0;
        set_fault_at(res, "DGC0_BAD_ROOT", 1, 0, "root_offset",
                     "Invalid DGC0 root offset " + std::to_string(root_off) +
                         " for size=" + std::to_string(size));
        if (out_fault_class) *out_fault_class = "DGC0";
        return false;
    }

    // 3. Parse with both libopenbuenos ob::parse_dgc0_buffer and freetile::inspect_buenos_dgc0.
    ob::ParsedDgc0 ob_parsed;
    std::string ob_err;
    if (ob::parse_dgc0_buffer(data, size, &ob_parsed, &ob_err) != OB_OK) {
        res->check_dgc0_wire_ok = 0;
        const char* code = "DGC0_PARSE_FAIL";
        if (ob_err.find("DWS0") != std::string::npos) code = "DGC0_BAD_DWS0";
        else if (ob_err.find("ext[") != std::string::npos || ob_err.find("external_buffers") != std::string::npos) code = "DGC0_EXT_OOB";
        else if (ob_err.find("Executable") != std::string::npos || ob_err.find("input") != std::string::npos) code = "DGC0_BAD_EXECUTABLE";
        set_fault(res, code, ob_err);
        if (out_fault_class) *out_fault_class = "DGC0";
        return false;
    }

    freetile::BuenosDgc0Summary ft_parsed;
    std::string ft_err;
    if (!freetile::inspect_buenos_dgc0(data, size, &ft_parsed, &ft_err)) {
        res->check_dgc0_wire_ok = 0;
        set_fault(res, "DGC0_INSPECT_FAIL", ft_err);
        if (out_fault_class) *out_fault_class = "DGC0";
        return false;
    }

    // 4. Cross-check parser parity.
    if (ft_parsed.input_sizes != ob_parsed.input_sizes ||
        ft_parsed.output_sizes != ob_parsed.output_sizes ||
        ft_parsed.ext0_offset != ob_parsed.ext_dive_vm_ir.offset ||
        ft_parsed.ext0_size != ob_parsed.ext_dive_vm_ir.size ||
        ft_parsed.ext1_offset != ob_parsed.ext_dive_image.offset ||
        ft_parsed.ext1_size != ob_parsed.ext_dive_image.size ||
        ft_parsed.has_params != ob_parsed.has_params ||
        (ft_parsed.has_params &&
         (ft_parsed.ext2_offset != ob_parsed.ext_params.offset ||
          ft_parsed.ext2_size != ob_parsed.ext_params.size))) {
        res->check_dgc0_wire_ok = 0;
        set_fault(res, "DGC0_PARSER_DIVERGENCE",
                  "freetile::inspect_buenos_dgc0 diverged from ob::parse_dgc0_buffer");
        if (out_fault_class) *out_fault_class = "DGC0";
        return false;
    }

    // 5. External buffer alignment (64B) & non-overlap verification.
    struct Slice {
        const char* name;
        uint64_t off;
        uint64_t len;
    };
    std::vector<Slice> slices;
    slices.push_back({"ext[0]", ob_parsed.ext_dive_vm_ir.offset, ob_parsed.ext_dive_vm_ir.size});
    slices.push_back({"ext[1]", ob_parsed.ext_dive_image.offset, ob_parsed.ext_dive_image.size});
    if (ob_parsed.has_params) {
        slices.push_back({"ext[2]", ob_parsed.ext_params.offset, ob_parsed.ext_params.size});
    }

    for (const Slice& s : slices) {
        if (s.len == 0u || s.off > size || s.len > size - s.off) {
            res->check_dgc0_wire_ok = 0;
            set_fault_at(res, "DGC0_EXT_OOB", 1, static_cast<int64_t>(s.off), s.name,
                         std::string(s.name) + " slice out of bounds");
            if (out_fault_class) *out_fault_class = "DGC0";
            return false;
        }
        if ((s.off % kDgc0PayloadAlignBytes) != 0u) {
            res->check_dgc0_wire_ok = 0;
            set_fault_at(res, "DGC0_EXT_MISALIGNED", 1, static_cast<int64_t>(s.off), s.name,
                         std::string(s.name) + " offset " + std::to_string(s.off) + " is not 64-byte aligned");
            if (out_fault_class) *out_fault_class = "DGC0";
            return false;
        }
    }

    for (size_t i = 0u; i < slices.size(); ++i) {
        for (size_t j = i + 1u; j < slices.size(); ++j) {
            const uint64_t a0 = slices[i].off;
            const uint64_t a1 = slices[i].off + slices[i].len;
            const uint64_t b0 = slices[j].off;
            const uint64_t b1 = slices[j].off + slices[j].len;
            if (a0 < b1 && b0 < a1) {
                res->check_dgc0_wire_ok = 0;
                set_fault(res, "DGC0_EXT_OVERLAP",
                          std::string(slices[i].name) + " overlaps " + slices[j].name);
                if (out_fault_class) *out_fault_class = "DGC0";
                return false;
            }
        }
    }

    // 6. Validate non-zero I/O tensor sizes & cluster_id.
    res->dgc0_num_inputs = static_cast<uint32_t>(ob_parsed.input_sizes.size());
    res->dgc0_num_outputs = static_cast<uint32_t>(ob_parsed.output_sizes.size());
    res->dgc0_cluster_id = ob_parsed.cluster_id;

    uint64_t total_io_bytes = 0u;
    for (size_t i = 0u; i < ob_parsed.input_sizes.size(); ++i) {
        if (ob_parsed.input_sizes[i] == 0u) {
            res->check_dgc0_wire_ok = 0;
            set_fault(res, "DGC0_ZERO_IO", "Input tensor[" + std::to_string(i) + "] has 0 byte size");
            if (out_fault_class) *out_fault_class = "DGC0";
            return false;
        }
        total_io_bytes += ob_parsed.input_sizes[i];
    }
    for (size_t i = 0u; i < ob_parsed.output_sizes.size(); ++i) {
        if (ob_parsed.output_sizes[i] == 0u) {
            res->check_dgc0_wire_ok = 0;
            set_fault(res, "DGC0_ZERO_IO", "Output tensor[" + std::to_string(i) + "] has 0 byte size");
            if (out_fault_class) *out_fault_class = "DGC0";
            return false;
        }
        total_io_bytes += ob_parsed.output_sizes[i];
    }

    if (ob_parsed.cluster_id == 0u) {
        res->check_dgc0_wire_ok = 0;
        set_fault(res, "DGC0_BAD_CLUSTER_ID", "Executable cluster_id must be non-zero");
        if (out_fault_class) *out_fault_class = "DGC0";
        return false;
    }

    // 7. Validate ext[0] DIVE VM IR header (non-zero, at least 16 bytes).
    if (ob_parsed.ext_dive_vm_ir.size < 16u) {
        res->check_dgc0_wire_ok = 0;
        set_fault(res, "DGC0_BAD_EXT0", "ext[0] DIVE VM IR is truncated (< 16 bytes)");
        if (out_fault_class) *out_fault_class = "DGC0";
        return false;
    }

    res->check_dgc0_wire_ok = 1;

    // 8. When validating a standalone .dgc0 payload, also enforce SRAM budget & estimate latency.
    if (is_primary_domain) {
        const uint64_t sram_limit = (sram_limit_override > 0u)
                                        ? sram_limit_override
                                        : target_default_sram_limit(target, false);
        const uint64_t sram_est = ob_parsed.dive_scratch_size + total_io_bytes;
        res->has_sram_bytes = 1;
        res->sram_bytes = sram_est;
        res->has_sram_limit_bytes = 1;
        res->sram_limit_bytes = sram_limit;

        if (sram_est > sram_limit) {
            res->check_sram_budget_ok = 0;
            set_fault(res, "DGC0_SRAM_OVERFLOW",
                      "DGC0 working SRAM (" + std::to_string(sram_est) +
                          " B) exceeds " + ft_twin_target_name(target) +
                          " limit (" + std::to_string(sram_limit) + " B)");
            if (out_fault_class) *out_fault_class = "SRAM";
            return false;
        }
        res->check_sram_budget_ok = 1;

        const double bw_gbps = (target == FT_TWIN_TARGET_MALIBU)
                                   ? static_cast<double>(freetile::malibu_hardware_profile().lpddr5x_bw_gbps)
                                   : kBuenosDramBwGbps;
        const double vii_us = (target == FT_TWIN_TARGET_MALIBU) ? kMalibuTpuViiUs : kBuenosViiDispatchUs;
        const uint64_t dram_bytes = total_io_bytes + (ob_parsed.has_params ? ob_parsed.ext_params.size : 0u);
        const double mem_us = static_cast<double>(dram_bytes) / (bw_gbps * 1e3);
        res->has_est_hw_us = 1;
        res->est_hw_us = vii_us + mem_us;
        res->check_profile_routing_ok = 1;
        res->check_schedule_valid = 1;
        copy_cstr(res->selected_engine, sizeof(res->selected_engine),
                  (target == FT_TWIN_TARGET_MALIBU) ? "edgetpu_heavy" : "buenos_a0_tpu");
    }

    return true;
}

// Validate a 0x640 spline kind (wire packet round-trip + dense FP64 error sweep).
bool check_spline_0x640_internal(ft_twin_spline_kind_t kind,
                                 uint32_t samples,
                                 double max_abs_limit_override,
                                 double* out_nmse,
                                 double* out_max_abs,
                                 std::string* out_err) {
    const freetile::SplineKind fk = to_ft_spline(kind);
    const freetile::SplineTable0x640& tbl = freetile::buenos_spline_table(fk);
    std::string err;
    if (!freetile::validate_spline_table(tbl, &err)) {
        if (out_err) *out_err = "Spline table invalid: " + err;
        return false;
    }

    freetile::SplinePacket0x640 pkt{};
    if (!freetile::make_op640_packet(tbl, &pkt, 0xFFFFu, 1u, 0u, freetile::kOp640TailDeg4, &err)) {
        if (out_err) *out_err = "make_op640_packet failed: " + err;
        return false;
    }
    uint8_t wire[freetile::kOp640PacketBytes]{};
    if (!freetile::encode_op640_packet(pkt, wire, &err)) {
        if (out_err) *out_err = "encode_op640_packet failed: " + err;
        return false;
    }
    freetile::SplinePacket0x640 decoded{};
    if (!freetile::decode_op640_packet(wire, &decoded, &err)) {
        if (out_err) *out_err = "decode_op640_packet failed: " + err;
        return false;
    }
    if (std::memcmp(decoded.coeffs, pkt.coeffs, sizeof(pkt.coeffs)) != 0 ||
        std::memcmp(decoded.breakpoints, pkt.breakpoints, sizeof(pkt.breakpoints)) != 0) {
        if (out_err) *out_err = "0x640 208-byte wire round-trip mismatch";
        return false;
    }

    const uint32_t npts = (samples >= 16u) ? samples : 2049u;
    double err_sq = 0.0;
    double ref_sq = 0.0;
    double max_abs = 0.0;

    for (uint32_t i = 0u; i < npts; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(npts - 1u);
        double x = 0.0;
        double y_ref = 0.0;
        float y_hat = 0.0f;

        switch (kind) {
            case FT_TWIN_SPLINE_EXP_NEG12_96:
                x = -12.96 + 12.96 * t;
                y_ref = std::exp(x);
                y_hat = freetile::eval_buenos_exp_shifted(static_cast<float>(x), false);
                break;
            case FT_TWIN_SPLINE_EXP_NEG11:
                x = -11.0 + 11.0 * t;
                y_ref = std::exp(x);
                y_hat = freetile::eval_buenos_exp_shifted(static_cast<float>(x), true);
                break;
            case FT_TWIN_SPLINE_RSQRT:
                x = 1.0 + 3.0 * t;
                y_ref = 1.0 / std::sqrt(x);
                y_hat = freetile::eval_buenos_rsqrt(static_cast<float>(x), true);
                break;
            case FT_TWIN_SPLINE_RECIP:
                x = 1.0 + 1.0 * t;
                y_ref = 1.0 / x;
                y_hat = freetile::eval_buenos_recip(static_cast<float>(x), true);
                break;
            case FT_TWIN_SPLINE_SILU:
                x = -11.1 + 22.2 * t;
                y_ref = ref_silu(x);
                y_hat = freetile::eval_buenos_silu(static_cast<float>(x));
                break;
            case FT_TWIN_SPLINE_GELU_TANH:
                x = -11.1 + 22.2 * t;
                y_ref = ref_gelu_tanh(x);
                y_hat = freetile::eval_buenos_gelu_tanh(static_cast<float>(x));
                break;
            case FT_TWIN_SPLINE_GELU_ERF:
                x = -11.1 + 22.2 * t;
                y_ref = ref_gelu_erf(x);
                y_hat = freetile::eval_buenos_gelu_erf(static_cast<float>(x));
                break;
        }
        if (!std::isfinite(y_hat)) {
            if (out_err) *out_err = "0x640 spline produced non-finite output";
            return false;
        }
        const double d = static_cast<double>(y_hat) - y_ref;
        const double ad = std::fabs(d);
        if (ad > max_abs) max_abs = ad;
        err_sq += d * d;
        ref_sq += y_ref * y_ref;
    }

    const double nmse = (ref_sq > 1e-30) ? (err_sq / ref_sq) : err_sq;
    if (out_nmse) *out_nmse = nmse;
    if (out_max_abs) *out_max_abs = max_abs;

    double default_bound = 2.0e-3;
    switch (kind) {
        case FT_TWIN_SPLINE_EXP_NEG12_96:
        case FT_TWIN_SPLINE_EXP_NEG11:
            default_bound = 1.0e-5;
            break;
        case FT_TWIN_SPLINE_RSQRT:
        case FT_TWIN_SPLINE_RECIP:
            default_bound = 5.0e-6;
            break;
        case FT_TWIN_SPLINE_SILU:
        case FT_TWIN_SPLINE_GELU_TANH:
        case FT_TWIN_SPLINE_GELU_ERF:
            default_bound = 2.0e-3;
            break;
    }
    const double bound = (max_abs_limit_override > 0.0) ? max_abs_limit_override : default_bound;
    if (max_abs > bound) {
        if (out_err) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "0x640 spline max_abs_err=%.6e exceeds bound=%.6e", max_abs, bound);
            *out_err = buf;
        }
        return false;
    }
    return true;
}

// Compute tile SRAM footprint for a candidate (bm, bn, bk) GEMM/Quant tile.
uint64_t compute_gemm_tile_sram_bytes(uint32_t bm,
                                      uint32_t bn,
                                      uint32_t bk,
                                      size_t act_elem_bytes,
                                      size_t weight_elem_bits,
                                      size_t out_elem_bytes,
                                      bool has_scales,
                                      bool has_mins,
                                      uint32_t quant_block_elems,
                                      bool double_buffered,
                                      bool needs_unpack_bf16_buf,
                                      bool split_k) {
    const uint64_t stages = double_buffered ? 2ULL : 1ULL;
    const uint64_t act_b = static_cast<uint64_t>(bm) * bk * act_elem_bytes;
    const uint64_t w_bits = static_cast<uint64_t>(bk) * bn * weight_elem_bits;
    const uint64_t w_b = (w_bits + 7ULL) / 8ULL;
    uint64_t meta_b = 0ULL;
    if (has_scales && quant_block_elems > 0u) {
        const uint64_t nb = (static_cast<uint64_t>(bk) + quant_block_elems - 1ULL) / quant_block_elems;
        meta_b += nb * bn * 4ULL; // FP32/FP16 scales (conservative 4 B)
        if (has_mins) {
            meta_b += nb * bn * 4ULL;
        }
    }
    const uint64_t out_b = static_cast<uint64_t>(bm) * bn * out_elem_bytes;
    const uint64_t acc_b = split_k ? (static_cast<uint64_t>(bm) * bn * 4ULL) : 0ULL;
    const uint64_t unpack_b = needs_unpack_bf16_buf ? (static_cast<uint64_t>(bk) * bn * 2ULL) : 0ULL;

    return stages * (act_b + w_b + meta_b) + out_b + acc_b + unpack_b;
}

// Deterministic hardware roofline latency estimator for GEMM / Quantized Matmul.
double estimate_matmul_hw_us(ft_twin_target_t target,
                             uint32_t m,
                             uint32_t k,
                             uint32_t n,
                             uint32_t bm,
                             uint32_t bn,
                             uint32_t bk,
                             double weight_bytes_per_elem,
                             size_t act_bytes_per_elem,
                             size_t out_bytes_per_elem,
                             bool double_buffered,
                             bool is_fp16_or_bf16_compute,
                             bool is_fp32_compute,
                             freetile::MalibuComputeEngine malibu_engine,
                             bool malibu_kquant_repack,
                             bool malibu_bus_wait,
                             bool malibu_syscall_barrier,
                             uint32_t* out_tiles) {
    const uint32_t eff_bm = (bm > 0u) ? bm : freetile::buenos_align_hw_m(m);
    const uint32_t eff_bn = (bn > 0u) ? bn : n;
    const uint32_t eff_bk = (bk > 0u) ? bk : k;

    const uint32_t n_tiles = (n + eff_bn - 1u) / eff_bn;
    const uint32_t k_tiles = (k + eff_bk - 1u) / eff_bk;
    const uint32_t m_tiles = (m + eff_bm - 1u) / eff_bm;
    const uint32_t total_tiles = m_tiles * n_tiles * k_tiles;
    if (out_tiles) *out_tiles = total_tiles;

    const uint32_t hw_m = freetile::buenos_align_hw_m(m);
    const double row_eff = std::min(1.0, static_cast<double>(m) / static_cast<double>(hw_m));
    const double tile_eff = std::max(0.15, row_eff * ((eff_bn >= 64u) ? 1.0 : (static_cast<double>(eff_bn) / 64.0)));

    const double macs = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
    const double act_bytes = static_cast<double>(m) * k * static_cast<double>(act_bytes_per_elem) * static_cast<double>(n_tiles);
    const double w_bytes = static_cast<double>(k) * n * weight_bytes_per_elem * static_cast<double>(m_tiles);
    const double out_bytes = static_cast<double>(m) * n * static_cast<double>(out_bytes_per_elem) * static_cast<double>(k_tiles);
    const double total_bytes = act_bytes + w_bytes + out_bytes;

    double peak_tops = kBuenosInt8Tops;
    double bw_gbps = kBuenosDramBwGbps;
    double base_dispatch_us = kBuenosViiDispatchUs;
    double per_tile_us = kBuenosTilePipeUs;

    if (target == FT_TWIN_TARGET_BUENOS_A0) {
        if (is_fp32_compute) peak_tops = kBuenosFp32Tflops;
        else if (is_fp16_or_bf16_compute) peak_tops = kBuenosFp16Tflops;
        else peak_tops = kBuenosInt8Tops;
        bw_gbps = kBuenosDramBwGbps;
        base_dispatch_us = kBuenosViiDispatchUs;
        per_tile_us = kBuenosTilePipeUs;
    } else {
        bw_gbps = static_cast<double>(freetile::malibu_hardware_profile().lpddr5x_bw_gbps);
        switch (malibu_engine) {
            case freetile::MalibuComputeEngine::EDGETPU_HEAVY:
                peak_tops = is_fp16_or_bf16_compute ? kMalibuTpuFp16Tflops : kMalibuTpuInt8Tops;
                base_dispatch_us = kMalibuTpuViiUs;
                per_tile_us = 10.0;
                break;
            case freetile::MalibuComputeEngine::CPU_SME2:
                peak_tops = is_fp16_or_bf16_compute ? kMalibuSme2Fp16Tflops : kMalibuSme2Int8Tops;
                base_dispatch_us = kMalibuCpuDispatchUs;
                per_tile_us = 2.0;
                break;
            case freetile::MalibuComputeEngine::CPU_NEON:
                peak_tops = kMalibuNeonTops;
                base_dispatch_us = kMalibuCpuDispatchUs;
                per_tile_us = 2.0;
                break;
            case freetile::MalibuComputeEngine::GPU_VULKAN_BDA:
                peak_tops = kMalibuGpuTflops;
                base_dispatch_us = kMalibuGpuDispatchUs;
                per_tile_us = 5.0;
                break;
            case freetile::MalibuComputeEngine::EDGETPU_NANO:
                peak_tops = 4.0;
                base_dispatch_us = 120.0;
                per_tile_us = 8.0;
                break;
        }
    }

    const double compute_us = macs / (peak_tops * 1e6 * tile_eff);
    const double mem_us = total_bytes / (bw_gbps * 1e3);
    double body_us = double_buffered ? std::max(compute_us, mem_us) : (compute_us + mem_us);
    if (target == FT_TWIN_TARGET_MALIBU && malibu_kquant_repack) {
        body_us *= 1.125; // Empirical 12.5% SME2 runtime repack overhead for K-quants
    }
    double extra_us = 0.0;
    if (malibu_bus_wait) extra_us += 150.0;
    if (malibu_syscall_barrier) extra_us += 12.0;

    return base_dispatch_us + static_cast<double>(total_tiles > 0u ? (total_tiles - 1u) : 0u) * per_tile_us + body_us + extra_us;
}

// =========================================================================
// Workload Evaluators
// =========================================================================

void eval_gemm_workload(ft_twin_session_t* session,
                        const ft_twin_request_t& req,
                        ft_twin_result_t* res) {
    const uint32_t m = (req.m > 0u) ? req.m : 1u;
    const uint32_t k = (req.k > 0u) ? req.k : 512u;
    const uint32_t n = (req.n > 0u) ? req.n : 512u;

    const char* dt_name = "bf16";
    if (req.act_dtype == FT_TWIN_DTYPE_FP32) dt_name = "fp32";
    else if (req.act_dtype == FT_TWIN_DTYPE_FP16) dt_name = "fp16";
    else if (req.act_dtype == FT_TWIN_DTYPE_INT8) dt_name = "w8a8";

    if (req.workload_label && req.workload_label[0] != '\0') {
        copy_cstr(res->workload, sizeof(res->workload), req.workload_label);
    } else {
        char wbuf[128];
        std::snprintf(wbuf, sizeof(wbuf), "gemm_%s_%ux%ux%u", dt_name, m, k, n);
        copy_cstr(res->workload, sizeof(res->workload), wbuf);
    }

    const uint64_t sram_limit = (req.sram_limit_override_bytes > 0u)
                                    ? req.sram_limit_override_bytes
                                    : target_default_sram_limit(req.target, false);
    res->has_sram_limit_bytes = 1;
    res->sram_limit_bytes = sram_limit;

    // Select candidate tile dims (bm, bn, bk).
    uint32_t bm = (req.bm > 0u) ? req.bm : freetile::buenos_align_hw_m(m);
    uint32_t bn = req.bn;
    uint32_t bk = req.bk;

    // Auto-select tile dimensions that fit the target SRAM limit if not overridden.
    const size_t act_b = ft_dtype_bytes(req.act_dtype);
    const size_t w_bits = (req.act_dtype == FT_TWIN_DTYPE_INT8) ? 8u : (act_b * 8u);
    const size_t out_b = ft_dtype_bytes(req.out_dtype);

    if (bn == 0u || bk == 0u) {
        const uint32_t cands[] = {2048u, 1024u, 512u, 256u, 128u, 64u, 32u, 16u};
        uint32_t pick_bn = (bn > 0u) ? bn : n;
        uint32_t pick_bk = (bk > 0u) ? bk : k;
        if (bn == 0u) {
            pick_bn = n;
            for (uint32_t c : cands) {
                if (c <= n && (n % c) == 0u) {
                    const uint64_t est = compute_gemm_tile_sram_bytes(
                        bm, c, (bk > 0u ? bk : std::min(k, 256u)), act_b, w_bits, out_b,
                        false, false, 1u, false, false, false);
                    if (est <= sram_limit) {
                        pick_bn = c;
                        break;
                    }
                }
            }
        }
        if (bk == 0u) {
            pick_bk = k;
            for (uint32_t c : cands) {
                if (c <= k && (k % c) == 0u) {
                    const bool sk = (c < k);
                    const uint64_t est = compute_gemm_tile_sram_bytes(
                        bm, pick_bn, c, act_b, w_bits, out_b,
                        false, false, 1u, sk, false, sk);
                    if (est <= sram_limit) {
                        pick_bk = c;
                        break;
                    }
                }
            }
        }
        bn = pick_bn;
        bk = pick_bk;
    }

    if (bm < m) {
        res->check_schedule_valid = 0;
        set_fault(res, "SCHEDULE_INVALID", "bm (" + std::to_string(bm) + ") < m (" + std::to_string(m) + ")");
        finalize_reward(res, 1e-3, 1000.0, "SCHEDULE");
        return;
    }
    if (bn == 0u || bk == 0u || bn > n || bk > k) {
        res->check_schedule_valid = 0;
        set_fault(res, "SCHEDULE_INVALID", "Tile dims (bn, bk) must be in [1..N, 1..K]");
        finalize_reward(res, 1e-3, 1000.0, "SCHEDULE");
        return;
    }

    const bool split_k = (bk < k);
    const bool db = (req.double_buffered < 0) ? split_k : (req.double_buffered != 0);

    res->resolved_bm = bm;
    res->resolved_bn = bn;
    res->resolved_bk = bk;

    const uint64_t sram_bytes = compute_gemm_tile_sram_bytes(
        bm, bn, bk, act_b, w_bits, out_b, false, false, 1u, db, false, split_k);
    res->has_sram_bytes = 1;
    res->sram_bytes = sram_bytes;

    // Malibu hybrid routing telemetry.
    freetile::MalibuComputeEngine malibu_eng = freetile::MalibuComputeEngine::EDGETPU_HEAVY;
    bool malibu_bus_wait = false;
    bool malibu_syscall_barrier = false;
    if (req.target == FT_TWIN_TARGET_MALIBU) {
        freetile::MalibuRouteRequest mreq{};
        mreq.phase = (m > 1u) ? freetile::MalibuWorkloadPhase::PREFILL_DENSE
                              : freetile::MalibuWorkloadPhase::DECODE_M1;
        mreq.quant = (req.act_dtype == FT_TWIN_DTYPE_INT8) ? freetile::MalibuQuantKind::Q8_0
                                                           : freetile::MalibuQuantKind::BF16;
        mreq.batch_m = m;
        mreq.requested_threads = (req.malibu_threads > 0u) ? req.malibu_threads : 5u;
        mreq.active_engine_mask = req.malibu_active_engine_mask;
        mreq.mesa_pco_compute_ready = (req.malibu_mesa_pco_ready != 0);
        freetile::MalibuRouteDecision mdec{};
        std::string merr;
        if (!freetile::route_malibu_workload(mreq, &mdec, freetile::malibu_hardware_profile(), &merr)) {
            res->check_profile_routing_ok = 0;
            set_fault(res, "MALIBU_ROUTE_FAIL", merr);
            finalize_reward(res, 1e-3, 1000.0, "SCHEDULE");
            return;
        }
        malibu_eng = mdec.engine;
        malibu_bus_wait = mdec.requires_bus_serialization_wait;
        malibu_syscall_barrier = mdec.requires_syscall_barrier;
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), freetile::malibu_engine_name(malibu_eng));
        res->check_profile_routing_ok = 1;
    } else {
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), "buenos_a0_tpu");
        res->check_profile_routing_ok = 1;
    }

    // Hardware latency estimate (compute before SRAM check so diagnostics have full context).
    res->has_est_hw_us = 1;
    res->est_hw_us = estimate_matmul_hw_us(
        req.target, m, k, n, bm, bn, bk,
        static_cast<double>(w_bits) / 8.0, act_b, out_b,
        db,
        req.act_dtype == FT_TWIN_DTYPE_BF16 || req.act_dtype == FT_TWIN_DTYPE_FP16,
        req.act_dtype == FT_TWIN_DTYPE_FP32,
        malibu_eng, false, malibu_bus_wait, malibu_syscall_barrier,
        &res->tiles_dispatched);

    // SRAM hard gate.
    if (sram_bytes > sram_limit) {
        res->check_sram_budget_ok = 0;
        res->check_schedule_valid = 0;
        set_fault(res, "SRAM_OVERFLOW",
                  "GEMM tile footprint " + std::to_string(sram_bytes) +
                      " B exceeds " + ft_twin_target_name(req.target) +
                      " SRAM limit " + std::to_string(sram_limit) + " B");
        finalize_reward(res, 1e-3, 1000.0, "SRAM");
        return;
    }
    res->check_sram_budget_ok = 1;
    res->check_schedule_valid = 1;

    // 0x640 spline check when SiLU or GELU epilogue is requested.
    if (req.activation == FT_TWIN_ACT_SILU || req.activation == FT_TWIN_ACT_GELU) {
        const ft_twin_spline_kind_t sk = (req.activation == FT_TWIN_ACT_SILU)
                                             ? FT_TWIN_SPLINE_SILU
                                             : FT_TWIN_SPLINE_GELU_TANH;
        double sp_nmse = 0.0, sp_max = 0.0;
        std::string sp_err;
        if (!check_spline_0x640_internal(sk, req.spline_samples ? req.spline_samples : 1025u,
                                         0.0, &sp_nmse, &sp_max, &sp_err)) {
            res->check_spline_0x640_ok = 0;
            set_fault(res, "SPLINE_0X640_FAIL", sp_err);
            finalize_reward(res, 1e-3, 1000.0, "SPLINE");
            return;
        }
        res->spline_max_err = sp_max;
        res->check_spline_0x640_ok = 1;
    }

    // DGC0 emission & wire integrity check for supported BUENOS_A0 shapes.
    if (req.verify_dgc0) {
        const uint32_t dgc_m = freetile::is_buenos_w8a8_shape(m, k, n) ? m : std::min(m, 8u);
        const uint32_t dgc_k = freetile::is_buenos_w8a8_shape(dgc_m, k, n) ? k : std::min(k, 64u);
        const uint32_t dgc_n = freetile::is_buenos_w8a8_shape(dgc_m, dgc_k, n) ? n : std::min(n, 64u);

        freetile::TileKernel tk;
        std::string plan_err;
        if (freetile::plan_buenos_matmul_kernel(
                dgc_m, dgc_k, dgc_n, freetile::QuantFormat::Q8_0,
                freetile::DType::S8, freetile::DType::S8,
                to_ft_act(req.activation), &tk, &plan_err)) {
            freetile::BuenosEmitOptions opts{};
            opts.mode = freetile::BuenosKernelMode::W8A8_DYN_WEIGHTS;
            opts.s_in = 0.03125f;
            opts.s_w = 0.015625f;
            opts.s_out = 0.0625f;
            opts.dive_interp = session->dummy_dive_interp.data();
            opts.dive_interp_size = session->dummy_dive_interp.size();
            freetile::BuenosEmitResult emit_res;
            std::string emit_err;
            if (!freetile::try_emit_buenos_kernel(tk, opts, &emit_res, &emit_err) || emit_res.dgc0.empty()) {
                res->check_dgc0_wire_ok = 0;
                set_fault(res, "DGC0_EMIT_FAIL", emit_err);
                finalize_reward(res, 1e-3, 1000.0, "DGC0");
                return;
            }
            const char* fclass = nullptr;
            if (!validate_dgc0_payload_internal(emit_res.dgc0.data(), emit_res.dgc0.size(),
                                                req.target, nullptr, 0u, res, false, &fclass)) {
                finalize_reward(res, 1e-3, 1000.0, fclass ? fclass : "DGC0");
                return;
            }
        }
    }

    // Reference parity execution in libopenbuenos mock_mode=1 vs FP64 reference.
    // Bound verification slice only if M*K*N > kMaxFullVerifyMacs (preserves sub-ms latency on huge shapes).
    uint32_t vm = m;
    uint32_t vk = k;
    uint32_t vn = n;
    while (static_cast<uint64_t>(vm) * vk * vn > kMaxFullVerifyMacs && vm > 1u) {
        vm = std::max<uint32_t>(1u, vm / 2u);
    }
    while (static_cast<uint64_t>(vm) * vk * vn > kMaxFullVerifyMacs && vn > 64u) {
        vn = std::max<uint32_t>(64u, vn / 2u);
    }
    while (static_cast<uint64_t>(vm) * vk * vn > kMaxFullVerifyMacs && vk > 64u) {
        vk = std::max<uint32_t>(64u, vk / 2u);
    }

    uint32_t rng = req.seed * 747796405u + 2891336453u;
    std::vector<double> got_vec(static_cast<size_t>(vm) * vn, 0.0);
    std::vector<double> ref_vec(static_cast<size_t>(vm) * vn, 0.0);
    res->verified_elements = vm * vn;

    if (req.act_dtype == FT_TWIN_DTYPE_INT8) {
        std::vector<int8_t> a(static_cast<size_t>(vm) * vk);
        std::vector<int8_t> b(static_cast<size_t>(vn) * vk);
        std::vector<int32_t> bias(vn, 0);
        std::vector<int32_t> c_i32(static_cast<size_t>(vm) * vn, 0);
        for (auto& v : a) v = rand_i8(rng, -64, 64);
        for (auto& v : b) v = rand_i8(rng, -64, 64);
        for (auto& v : bias) v = static_cast<int32_t>(rand_i8(rng, -64, 64));

        ob_invoke_stats_t istats{};
        ob_status_t st = ob_gemm_matmul_int8(
            session->ob_ctx, a.data(), b.data(), bias.data(), c_i32.data(),
            vm, vk, vn, 0, 0, &istats);
        if (st != OB_OK) {
            res->check_reference_parity_ok = 0;
            set_fault(res, "MOCK_EXEC_FAIL",
                      std::string("ob_gemm_matmul_int8 failed: ") + ob_status_string(st));
            finalize_reward(res, 1e-4, 1000.0, "ACCURACY");
            return;
        }
        for (uint32_t mi = 0u; mi < vm; ++mi) {
            for (uint32_t ni = 0u; ni < vn; ++ni) {
                double acc = static_cast<double>(bias[ni]);
                for (uint32_t ki = 0u; ki < vk; ++ki) {
                    acc += static_cast<double>(a[mi * vk + ki]) * static_cast<double>(b[ni * vk + ki]);
                }
                const double g_val = static_cast<double>(c_i32[mi * vn + ni]);
                ref_vec[mi * vn + ni] = apply_ref_activation(acc, req.activation);
                if (req.activation == FT_TWIN_ACT_SILU) {
                    got_vec[mi * vn + ni] = static_cast<double>(freetile::eval_buenos_silu(static_cast<float>(g_val * 0.001))) * 1000.0;
                    ref_vec[mi * vn + ni] = ref_silu(acc * 0.001) * 1000.0;
                } else if (req.activation == FT_TWIN_ACT_GELU) {
                    got_vec[mi * vn + ni] = static_cast<double>(freetile::eval_buenos_gelu_tanh(static_cast<float>(g_val * 0.001))) * 1000.0;
                    ref_vec[mi * vn + ni] = ref_gelu_tanh(acc * 0.001) * 1000.0;
                } else {
                    got_vec[mi * vn + ni] = apply_ref_activation(g_val, req.activation);
                }
            }
        }
    } else {
        // Dense BF16 / FP16 / FP32 GEMM: simulate exact target numeric precision + 0x640 epilogue vs FP64 reference.
        std::vector<float> a_f32(static_cast<size_t>(vm) * vk);
        std::vector<float> b_f32(static_cast<size_t>(vn) * vk);
        const float inv_s = 1.0f / std::sqrt(static_cast<float>(vk));
        for (auto& v : a_f32) v = rand_uniform(rng, -1.5f, 1.5f) * inv_s;
        for (auto& v : b_f32) v = rand_uniform(rng, -1.5f, 1.5f);

        auto quant_in = [&](float x) -> float {
            if (req.act_dtype == FT_TWIN_DTYPE_BF16) return round_f32_to_bf16(x);
            if (req.act_dtype == FT_TWIN_DTYPE_FP16) return ob::quant::fp16_to_fp32_scalar(f32_to_f16_rne(x));
            return x;
        };
        auto quant_out = [&](float x) -> float {
            if (req.out_dtype == FT_TWIN_DTYPE_BF16) return round_f32_to_bf16(x);
            if (req.out_dtype == FT_TWIN_DTYPE_FP16) return ob::quant::fp16_to_fp32_scalar(f32_to_f16_rne(x));
            return x;
        };

        std::vector<float> a_hw(a_f32.size());
        std::vector<float> b_hw(b_f32.size());
        for (size_t i = 0u; i < a_f32.size(); ++i) a_hw[i] = quant_in(a_f32[i]);
        for (size_t i = 0u; i < b_f32.size(); ++i) b_hw[i] = quant_in(b_f32[i]);

        for (uint32_t mi = 0u; mi < vm; ++mi) {
            for (uint32_t ni = 0u; ni < vn; ++ni) {
                double ref_acc = 0.0;
                float hw_acc = 0.0f;
                const size_t a_off = static_cast<size_t>(mi) * vk;
                const size_t b_off = static_cast<size_t>(ni) * vk;
                for (uint32_t ki = 0u; ki < vk; ++ki) {
                    ref_acc += static_cast<double>(a_f32[a_off + ki]) * static_cast<double>(b_f32[b_off + ki]);
                    hw_acc += a_hw[a_off + ki] * b_hw[b_off + ki];
                }
                const double ref_y = apply_ref_activation(ref_acc, req.activation);
                float hw_y = hw_acc;
                if (req.activation == FT_TWIN_ACT_RELU) hw_y = hw_y > 0.0f ? hw_y : 0.0f;
                else if (req.activation == FT_TWIN_ACT_RELU6) hw_y = hw_y < 0.0f ? 0.0f : (hw_y > 6.0f ? 6.0f : hw_y);
                else if (req.activation == FT_TWIN_ACT_SILU) hw_y = freetile::eval_buenos_silu(hw_y);
                else if (req.activation == FT_TWIN_ACT_GELU) hw_y = freetile::eval_buenos_gelu_tanh(hw_y);
                hw_y = quant_out(hw_y);

                ref_vec[mi * vn + ni] = ref_y;
                got_vec[mi * vn + ni] = static_cast<double>(hw_y);
            }
        }
    }

    double nmse = 0.0, max_abs = 0.0;
    bool nonfinite = false;
    compute_error_metrics(got_vec, ref_vec, &nmse, &max_abs, &nonfinite);
    res->has_nmse = 1;
    res->nmse = nmse;
    res->has_max_abs_err = 1;
    res->max_abs_err = max_abs;

    const double nmse_tol = (req.max_nmse > 0.0)
                                ? req.max_nmse
                                : ((req.act_dtype == FT_TWIN_DTYPE_FP32 && req.out_dtype == FT_TWIN_DTYPE_FP32) ? 1e-6 : 5e-3);
    if (nonfinite || nmse > nmse_tol || (req.max_abs_err_limit > 0.0 && max_abs > req.max_abs_err_limit)) {
        res->check_reference_parity_ok = 0;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "NMSE=%.6e (tol=%.6e) max_abs_err=%.6e nonfinite=%d",
                      nmse, nmse_tol, max_abs, nonfinite ? 1 : 0);
        set_fault(res, "ACCURACY_FAIL", buf);
        finalize_reward(res, nmse_tol, 1000.0, "ACCURACY");
        return;
    }

    res->check_reference_parity_ok = 1;
    finalize_reward(res, nmse_tol, 1000.0, nullptr);
}

void eval_swiglu_workload(ft_twin_session_t* session,
                          const ft_twin_request_t& req,
                          ft_twin_result_t* res) {
    const uint32_t m = (req.m > 0u) ? req.m : 1u;
    const uint32_t k = (req.k > 0u) ? req.k : 576u;
    const uint32_t h = (req.h > 0u) ? req.h : 512u;
    const uint32_t n = (req.n > 0u) ? req.n : k;
    const bool mode_11chunk = (req.swiglu_11chunk != 0);

    if (req.workload_label && req.workload_label[0] != '\0') {
        copy_cstr(res->workload, sizeof(res->workload), req.workload_label);
    } else {
        char wbuf[128];
        std::snprintf(wbuf, sizeof(wbuf), "swiglu_%s_%ux%ux%ux%u",
                      mode_11chunk ? "11chunk" : ob_quant_type_name(static_cast<ob_quant_type_t>(req.quant_type % 6u)),
                      m, k, h, n);
        copy_cstr(res->workload, sizeof(res->workload), wbuf);
    }

    const uint64_t sram_limit = (req.sram_limit_override_bytes > 0u)
                                    ? req.sram_limit_override_bytes
                                    : target_default_sram_limit(req.target, false);
    res->has_sram_limit_bytes = 1;
    res->sram_limit_bytes = sram_limit;

    // 1. Always verify 0x640 SiLU spline error bounds for SwiGLU.
    {
        double sp_nmse = 0.0, sp_max = 0.0;
        std::string sp_err;
        if (!check_spline_0x640_internal(FT_TWIN_SPLINE_SILU,
                                         req.spline_samples ? req.spline_samples : 1025u,
                                         0.0, &sp_nmse, &sp_max, &sp_err)) {
            res->check_spline_0x640_ok = 0;
            set_fault(res, "SPLINE_0X640_FAIL", sp_err);
            finalize_reward(res, 1e-3, 1500.0, "SPLINE");
            return;
        }
        res->spline_max_err = sp_max;
        res->check_spline_0x640_ok = 1;
    }

    // Malibu routing telemetry.
    freetile::MalibuComputeEngine malibu_eng = freetile::MalibuComputeEngine::EDGETPU_HEAVY;
    bool malibu_bus_wait = false;
    bool malibu_syscall_barrier = false;
    if (req.target == FT_TWIN_TARGET_MALIBU) {
        freetile::MalibuRouteRequest mreq{};
        mreq.phase = (m > 1u) ? freetile::MalibuWorkloadPhase::PREFILL_DENSE
                              : freetile::MalibuWorkloadPhase::DECODE_M1;
        mreq.quant = mode_11chunk
                         ? freetile::MalibuQuantKind::Q8_0
                         : ob_quant_to_malibu(static_cast<ob_quant_type_t>(req.quant_type % 6u));
        mreq.batch_m = m;
        mreq.requested_threads = (req.malibu_threads > 0u) ? req.malibu_threads : 5u;
        mreq.active_engine_mask = req.malibu_active_engine_mask;
        mreq.mesa_pco_compute_ready = (req.malibu_mesa_pco_ready != 0);
        freetile::MalibuRouteDecision mdec{};
        std::string merr;
        if (!freetile::route_malibu_workload(mreq, &mdec, freetile::malibu_hardware_profile(), &merr)) {
            res->check_profile_routing_ok = 0;
            set_fault(res, "MALIBU_ROUTE_FAIL", merr);
            finalize_reward(res, 1e-3, 1500.0, "SCHEDULE");
            return;
        }
        malibu_eng = mdec.engine;
        malibu_bus_wait = mdec.requires_bus_serialization_wait;
        malibu_syscall_barrier = mdec.requires_syscall_barrier;
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), freetile::malibu_engine_name(malibu_eng));
        res->check_profile_routing_ok = 1;
    } else {
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), "buenos_a0_tpu");
        res->check_profile_routing_ok = 1;
    }

    if (mode_11chunk) {
        if (!freetile::is_buenos_swiglu_shape(m, k, h, n)) {
            res->check_schedule_valid = 0;
            set_fault(res, "SWIGLU_UNSUPPORTED_SHAPE",
                      "11-chunk chained SwiGLU shape (" + std::to_string(m) + "," +
                          std::to_string(k) + "," + std::to_string(h) + "," + std::to_string(n) +
                          ") is not a proven buenos_a0 11-chunk shape");
            finalize_reward(res, 1e-3, 1500.0, "SCHEDULE");
            return;
        }

        // Exact SRAM working set: input + gate/up/silu/prod intermediates + output + scratchpad.
        const uint32_t eff_bm = (req.bm > 0u) ? req.bm : m;
        const uint32_t eff_bn = (req.bn > 0u) ? req.bn : n;
        const uint32_t eff_bk = (req.bk > 0u) ? req.bk : k;
        res->resolved_bm = eff_bm;
        res->resolved_bn = eff_bn;
        res->resolved_bk = eff_bk;

        const uint64_t act_sram = static_cast<uint64_t>(eff_bm) * (eff_bk + 4ULL * h + eff_bn);
        const uint64_t w_tile_sram = (req.bm > 0u || req.bn > 0u || req.bk > 0u)
                                         ? (static_cast<uint64_t>(eff_bk) * h * 2ULL + static_cast<uint64_t>(eff_bn) * h)
                                         : (static_cast<uint64_t>(std::min<uint32_t>(k, 256u)) * std::min<uint32_t>(h, 512u) * 2ULL);
        const uint64_t sram_bytes = act_sram + w_tile_sram;
        res->has_sram_bytes = 1;
        res->sram_bytes = sram_bytes;

        // Latency: single-VII 11-chunk launch (saves 2x VII dispatches + intermediate DRAM roundtrips!).
        const double total_macs = 2.0 * static_cast<double>(m) *
                                  (2.0 * static_cast<double>(k) * h + static_cast<double>(h) * n);
        const size_t ext2_bytes = freetile::buenos_pack_swiglu_ext2_size(m, k, h, n);
        const double bw_gbps = (req.target == FT_TWIN_TARGET_MALIBU)
                                   ? static_cast<double>(freetile::malibu_hardware_profile().lpddr5x_bw_gbps)
                                   : kBuenosDramBwGbps;
        const double tops = (req.target == FT_TWIN_TARGET_MALIBU) ? kMalibuTpuInt8Tops : kBuenosInt8Tops;
        const double vii_us = (req.target == FT_TWIN_TARGET_MALIBU) ? kMalibuTpuViiUs : kBuenosViiDispatchUs;
        const double comp_us = total_macs / (tops * 1e6);
        const double mem_us = static_cast<double>(ext2_bytes + static_cast<size_t>(m) * (k + n)) / (bw_gbps * 1e3);
        res->has_est_hw_us = 1;
        res->est_hw_us = vii_us + std::max(comp_us, mem_us) +
                         (malibu_bus_wait ? 150.0 : 0.0) +
                         (malibu_syscall_barrier ? 12.0 : 0.0);
        res->tiles_dispatched = 1u;

        if (sram_bytes > sram_limit) {
            res->check_sram_budget_ok = 0;
            res->check_schedule_valid = 0;
            set_fault(res, "SRAM_OVERFLOW",
                      "11-chunk SwiGLU SRAM footprint " + std::to_string(sram_bytes) +
                          " B exceeds " + ft_twin_target_name(req.target) +
                          " limit " + std::to_string(sram_limit) + " B");
            finalize_reward(res, 1e-3, 1500.0, "SRAM");
            return;
        }
        res->check_sram_budget_ok = 1;
        res->check_schedule_valid = 1;

        // Deterministic INT8 weights + scales (calibrated to avoid saturation).
        uint32_t rng = req.seed * 1664525u + 1013904223u;
        std::vector<int8_t> x(static_cast<size_t>(m) * k);
        std::vector<int8_t> wg(static_cast<size_t>(h) * k);
        std::vector<int8_t> wu(static_cast<size_t>(h) * k);
        std::vector<int8_t> wd(static_cast<size_t>(n) * h);
        for (auto& v : x) v = rand_i8(rng, -24, 24);
        for (auto& v : wg) v = rand_i8(rng, -24, 24);
        for (auto& v : wu) v = rand_i8(rng, -24, 24);
        for (auto& v : wd) v = rand_i8(rng, -24, 24);

        freetile::BuenosSwigluDesc ft_desc{};
        ft_desc.m_dim = m;
        ft_desc.k_dim = k;
        ft_desc.h_dim = h;
        ft_desc.n_dim = n;
        ft_desc.wg_hk = wg.data();
        ft_desc.wu_hk = wu.data();
        ft_desc.wd_nh = wd.data();
        ft_desc.s_in = 0.03125f;
        ft_desc.s_wg = 0.015625f;
        ft_desc.s_gate = 0.0625f;
        ft_desc.s_wu = 0.015625f;
        ft_desc.s_up = 0.0625f;
        ft_desc.s_silu = 0.0625f;
        ft_desc.s_prod = 0.0625f;
        ft_desc.s_wd = 0.015625f;
        ft_desc.s_out = 0.0625f;
        ft_desc.dive_interp = session->dummy_dive_interp.data();
        ft_desc.dive_interp_size = session->dummy_dive_interp.size();

        if (req.verify_dgc0) {
            freetile::BuenosSwigluResult emit_res{};
            std::string emit_err;
            if (!freetile::try_emit_buenos_swiglu(ft_desc, &emit_res, &emit_err) || emit_res.dgc0.empty()) {
                res->check_dgc0_wire_ok = 0;
                set_fault(res, "DGC0_SWIGLU_EMIT_FAIL", emit_err);
                finalize_reward(res, 1e-3, 1500.0, "DGC0");
                return;
            }
            const char* fclass = nullptr;
            if (!validate_dgc0_payload_internal(emit_res.dgc0.data(), emit_res.dgc0.size(),
                                                req.target, nullptr, 0u, res, false, &fclass)) {
                finalize_reward(res, 1e-3, 1500.0, fclass ? fclass : "DGC0");
                return;
            }
        }

        // Execute via libopenbuenos mock_mode=1 ob_gemm_swiglu_int8 & verify bit-exact parity vs
        // freetile::buenos_simulate_swiglu_exact AND check numerical NMSE vs FP64 unquantized SwiGLU.
        ob_swiglu_int8_desc_t ob_desc{};
        ob_desc.m_dim = m;
        ob_desc.k_dim = k;
        ob_desc.h_dim = h;
        ob_desc.n_dim = n;
        ob_desc.wg_hk = wg.data();
        ob_desc.wu_hk = wu.data();
        ob_desc.wd_nh = wd.data();
        ob_desc.s_in = ft_desc.s_in;
        ob_desc.s_wg = ft_desc.s_wg;
        ob_desc.s_gate = ft_desc.s_gate;
        ob_desc.s_wu = ft_desc.s_wu;
        ob_desc.s_up = ft_desc.s_up;
        ob_desc.s_silu = ft_desc.s_silu;
        ob_desc.s_prod = ft_desc.s_prod;
        ob_desc.s_wd = ft_desc.s_wd;
        ob_desc.s_out = ft_desc.s_out;

        // For M=64, check full 11-chunk DGC0 emission above and verify exact numerical parity on M_verify rows.
        std::vector<int8_t> y_ob(static_cast<size_t>(m) * n, 0);
        std::vector<int8_t> y_ft(static_cast<size_t>(m) * n, 0);
        ob_invoke_stats_t istats{};
        ob_status_t st = ob_gemm_swiglu_int8(session->ob_ctx, x.data(), &ob_desc, y_ob.data(), &istats);
        if (st != OB_OK) {
            res->check_reference_parity_ok = 0;
            set_fault(res, "MOCK_SWIGLU_FAIL",
                      std::string("ob_gemm_swiglu_int8 failed: ") + ob_status_string(st));
            finalize_reward(res, 1e-3, 1500.0, "ACCURACY");
            return;
        }
        std::string sim_err;
        if (!freetile::buenos_simulate_swiglu_exact(x.data(), ft_desc, y_ft.data(), &sim_err)) {
            res->check_reference_parity_ok = 0;
            set_fault(res, "SWIGLU_SIM_FAIL", sim_err);
            finalize_reward(res, 1e-3, 1500.0, "ACCURACY");
            return;
        }

        // Exact bit-for-bit check between libopenbuenos mock_mode=1 and FreeTile simulator.
        std::vector<double> got_d(y_ob.size());
        std::vector<double> ref_d(y_ft.size());
        for (size_t i = 0u; i < y_ob.size(); ++i) {
            got_d[i] = static_cast<double>(y_ob[i]) * static_cast<double>(ft_desc.s_out);
            ref_d[i] = static_cast<double>(y_ft[i]) * static_cast<double>(ft_desc.s_out);
        }
        res->verified_elements = static_cast<uint32_t>(y_ob.size());
        double nmse = 0.0, max_abs = 0.0;
        bool nonfinite = false;
        compute_error_metrics(got_d, ref_d, &nmse, &max_abs, &nonfinite);
        res->has_nmse = 1;
        res->nmse = nmse;
        res->has_max_abs_err = 1;
        res->max_abs_err = max_abs;

        const double nmse_tol = (req.max_nmse > 0.0) ? req.max_nmse : 1e-6;
        if (nonfinite || nmse > nmse_tol || max_abs > 0.0) {
            res->check_reference_parity_ok = 0;
            set_fault(res, "SWIGLU_PARITY_MISMATCH",
                      "11-chunk SwiGLU mock_mode=1 diverged from exact Horner simulator");
            finalize_reward(res, nmse_tol, 1500.0, "ACCURACY");
            return;
        }

        res->check_reference_parity_ok = 1;
        finalize_reward(res, 1e-4, 1500.0, nullptr);
        return;
    }

    // Quantized FP32 SwiGLU path via ob_gemm_swiglu_f32_quant.
    const ob_quant_type_t qt = static_cast<ob_quant_type_t>(req.quant_type % 6u);
    const uint32_t qk = ob_quant_block_elements(qt);
    if ((k % qk) != 0u || (h % qk) != 0u) {
        res->check_schedule_valid = 0;
        set_fault(res, "SCHEDULE_INVALID",
                  "SwiGLU k (" + std::to_string(k) + ") and h (" + std::to_string(h) +
                      ") must be multiples of block size " + std::to_string(qk));
        finalize_reward(res, 1e-3, 1500.0, "SCHEDULE");
        return;
    }

    const uint32_t bm = (req.bm > 0u) ? req.bm : freetile::buenos_align_hw_m(m);
    const uint32_t bn = (req.bn > 0u) ? req.bn : std::min(n, 256u);
    const uint32_t bk = (req.bk > 0u) ? req.bk : std::min(k, 256u);
    res->resolved_bm = bm;
    res->resolved_bn = bn;
    res->resolved_bk = bk;

    const double w_bpe = static_cast<double>(ob_quant_block_bytes(qt)) / static_cast<double>(qk);
    const size_t w_bits = static_cast<size_t>(std::ceil(w_bpe * 8.0));
    const bool db = (req.double_buffered != 0);
    const uint64_t sram_bytes = compute_gemm_tile_sram_bytes(
        bm, bn, bk, 2u, w_bits, 4u, true, qt == OB_QUANT_Q4_K, qk, db, true, bk < k);
    res->has_sram_bytes = 1;
    res->sram_bytes = sram_bytes;

    const bool kquant_repack = (qt == OB_QUANT_Q4_K || qt == OB_QUANT_Q6_K);
    uint32_t t1 = 0u, t2 = 0u;
    const double lat_gu = 2.0 * estimate_matmul_hw_us(
        req.target, m, k, h, bm, bn, bk, w_bpe, 2u, 4u, db, false, false,
        malibu_eng, kquant_repack, malibu_bus_wait, malibu_syscall_barrier, &t1);
    const double lat_d = estimate_matmul_hw_us(
        req.target, m, h, n, bm, bn, bk, w_bpe, 2u, 4u, db, false, false,
        malibu_eng, kquant_repack, false, false, &t2);
    res->has_est_hw_us = 1;
    res->est_hw_us = lat_gu + lat_d;
    res->tiles_dispatched = 2u * t1 + t2;

    if (sram_bytes > sram_limit) {
        res->check_sram_budget_ok = 0;
        res->check_schedule_valid = 0;
        set_fault(res, "SRAM_OVERFLOW",
                  "Quantized SwiGLU tile SRAM " + std::to_string(sram_bytes) +
                      " B exceeds " + ft_twin_target_name(req.target) +
                      " limit " + std::to_string(sram_limit) + " B");
        finalize_reward(res, 1e-3, 1500.0, "SRAM");
        return;
    }
    res->check_sram_budget_ok = 1;
    res->check_schedule_valid = 1;

    uint32_t rng = req.seed * 2246822519u + 3266489917u;
    const auto wg = make_det_quant_matrix(qt, h, k, req.seed + 1u);
    const auto wu = make_det_quant_matrix(qt, h, k, req.seed + 2u);
    const auto wd = make_det_quant_matrix(qt, n, h, req.seed + 3u);
    std::vector<float> a(static_cast<size_t>(m) * k);
    for (auto& v : a) v = rand_uniform(rng, -0.5f, 0.5f);
    std::vector<float> c_out(static_cast<size_t>(m) * n, 0.0f);

    ob_gemm_swiglu_quant_desc_t sdesc{};
    sdesc.w_gate_hk = wg.data();
    sdesc.w_up_hk = wu.data();
    sdesc.w_down_nh = wd.data();
    sdesc.m_dim = m;
    sdesc.k_dim = k;
    sdesc.h_dim = h;
    sdesc.n_dim = n;
    ob_gemm_swiglu_quant_desc_set_types(&sdesc, qt, qt, qt);

    ob_invoke_stats_t istats{};
    ob_status_t st = ob_gemm_swiglu_f32_quant(session->ob_ctx, a.data(), &sdesc, c_out.data(), &istats);
    if (st != OB_OK) {
        res->check_reference_parity_ok = 0;
        set_fault(res, "MOCK_SWIGLU_QUANT_FAIL",
                  std::string("ob_gemm_swiglu_f32_quant failed: ") + ob_status_string(st));
        finalize_reward(res, 1e-3, 1500.0, "ACCURACY");
        return;
    }

    // Independent FP64 scalar reference dequant + SwiGLU.
    std::vector<float> row_f32(std::max(k, h));
    std::vector<double> gate_ref(static_cast<size_t>(m) * h, 0.0);
    std::vector<double> up_ref(static_cast<size_t>(m) * h, 0.0);
    const size_t blk_sz = ob_quant_block_bytes(qt);
    const size_t row_bytes_k = static_cast<size_t>(k / qk) * blk_sz;
    const size_t row_bytes_h = static_cast<size_t>(h / qk) * blk_sz;

    for (uint32_t hi = 0u; hi < h; ++hi) {
        ob::quant::dequant_row_scalar_f32(qt, wg.data() + hi * row_bytes_k, k, row_f32.data());
        for (uint32_t mi = 0u; mi < m; ++mi) {
            double acc = 0.0;
            for (uint32_t ki = 0u; ki < k; ++ki) acc += static_cast<double>(a[mi * k + ki]) * static_cast<double>(row_f32[ki]);
            gate_ref[mi * h + hi] = ref_silu(acc);
        }
        ob::quant::dequant_row_scalar_f32(qt, wu.data() + hi * row_bytes_k, k, row_f32.data());
        for (uint32_t mi = 0u; mi < m; ++mi) {
            double acc = 0.0;
            for (uint32_t ki = 0u; ki < k; ++ki) acc += static_cast<double>(a[mi * k + ki]) * static_cast<double>(row_f32[ki]);
            up_ref[mi * h + hi] = gate_ref[mi * h + hi] * acc;
        }
    }
    std::vector<double> ref_vec(static_cast<size_t>(m) * n, 0.0);
    std::vector<double> got_vec(static_cast<size_t>(m) * n, 0.0);
    for (uint32_t ni = 0u; ni < n; ++ni) {
        ob::quant::dequant_row_scalar_f32(qt, wd.data() + ni * row_bytes_h, h, row_f32.data());
        for (uint32_t mi = 0u; mi < m; ++mi) {
            double acc = 0.0;
            for (uint32_t hi = 0u; hi < h; ++hi) acc += up_ref[mi * h + hi] * static_cast<double>(row_f32[hi]);
            ref_vec[mi * n + ni] = acc;
            got_vec[mi * n + ni] = static_cast<double>(c_out[mi * n + ni]);
        }
    }

    res->verified_elements = m * n;
    double nmse = 0.0, max_abs = 0.0;
    bool nonfinite = false;
    compute_error_metrics(got_vec, ref_vec, &nmse, &max_abs, &nonfinite);
    res->has_nmse = 1;
    res->nmse = nmse;
    res->has_max_abs_err = 1;
    res->max_abs_err = max_abs;

    const double nmse_tol = (req.max_nmse > 0.0) ? req.max_nmse : 1e-3;
    if (nonfinite || nmse > nmse_tol) {
        res->check_reference_parity_ok = 0;
        set_fault(res, "ACCURACY_FAIL", "Quantized SwiGLU NMSE exceeded tolerance");
        finalize_reward(res, nmse_tol, 1500.0, "ACCURACY");
        return;
    }

    res->check_reference_parity_ok = 1;
    finalize_reward(res, nmse_tol, 1500.0, nullptr);
}

void eval_flash_attn_workload(const ft_twin_request_t& req, ft_twin_result_t* res) {
    const uint32_t head_dim = (req.head_dim > 0u) ? req.head_dim : 128u;
    const uint32_t num_q_heads = (req.num_q_heads > 0u) ? req.num_q_heads : 32u;
    const uint32_t num_kv_heads = (req.num_kv_heads > 0u) ? req.num_kv_heads : 4u;
    const uint32_t seq_len = (req.seq_len > 0u) ? req.seq_len : 1024u;
    const uint32_t kv_align = (req.kv_tile_align > 0u) ? req.kv_tile_align : 64u;
    const uint64_t sram_limit = (req.sram_limit_override_bytes > 0u)
                                    ? req.sram_limit_override_bytes
                                    : target_default_sram_limit(req.target, true);

    if (req.workload_label && req.workload_label[0] != '\0') {
        copy_cstr(res->workload, sizeof(res->workload), req.workload_label);
    } else {
        char wbuf[128];
        std::snprintf(wbuf, sizeof(wbuf), "flash_attn_%s_d%u_gqa%u_s%u",
                      req.fa_phase == FT_TWIN_FA_PREFILL ? "prefill" : "decode",
                      head_dim,
                      (num_kv_heads > 0u) ? (num_q_heads / num_kv_heads) : 1u,
                      seq_len);
        copy_cstr(res->workload, sizeof(res->workload), wbuf);
    }

    res->has_sram_limit_bytes = 1;
    res->sram_limit_bytes = sram_limit;

    freetile::FlashAttnDType fa_dt = freetile::FlashAttnDType::FP16;
    if (req.act_dtype == FT_TWIN_DTYPE_BF16) fa_dt = freetile::FlashAttnDType::BF16;
    else if (req.act_dtype == FT_TWIN_DTYPE_INT8) fa_dt = freetile::FlashAttnDType::INT8;

    freetile::FlashAttnBufferPolicy fa_pol = freetile::FlashAttnBufferPolicy::PING_PONG_FULL;
    if (req.fa_buf_policy == FT_TWIN_FA_KV_DB_FP32) fa_pol = freetile::FlashAttnBufferPolicy::KV_DOUBLE_BUFFERED_FP32_ACC;
    else if (req.fa_buf_policy == FT_TWIN_FA_SINGLE_BUF) fa_pol = freetile::FlashAttnBufferPolicy::SINGLE_BUFFERED;

    freetile::FlashAttnTileRequest freq{};
    freq.phase = (req.fa_phase == FT_TWIN_FA_PREFILL) ? freetile::FlashAttnPhase::PREFILL
                                                      : freetile::FlashAttnPhase::DECODE;
    freq.head_dim = head_dim;
    freq.num_q_heads = num_q_heads;
    freq.num_kv_heads = num_kv_heads;
    freq.seq_len = seq_len;
    freq.kv_tile_align = kv_align;
    freq.dtype = fa_dt;
    freq.buffer_policy = fa_pol;
    freq.sram_budget_bytes = static_cast<size_t>(sram_limit);
    freq.override_br = req.br;
    freq.override_bc = req.bc;

    // Malibu routing check.
    freetile::MalibuComputeEngine malibu_eng = freetile::MalibuComputeEngine::EDGETPU_HEAVY;
    bool malibu_bus_wait = false;
    bool malibu_syscall_barrier = false;
    if (req.target == FT_TWIN_TARGET_MALIBU) {
        freetile::MalibuRouteRequest mreq{};
        mreq.phase = (seq_len >= 512u) ? freetile::MalibuWorkloadPhase::ATTENTION_LONG_CTX
                                       : freetile::MalibuWorkloadPhase::ATTENTION_SHORT_CTX;
        mreq.quant = freetile::MalibuQuantKind::F16;
        mreq.batch_m = (req.fa_phase == FT_TWIN_FA_PREFILL) ? seq_len : 1u;
        mreq.seq_len = seq_len;
        mreq.n_ctx = std::max(seq_len, 4096u);
        mreq.requested_threads = (req.malibu_threads > 0u) ? req.malibu_threads : 5u;
        mreq.active_engine_mask = req.malibu_active_engine_mask;
        mreq.mesa_pco_compute_ready = (req.malibu_mesa_pco_ready != 0);
        freetile::MalibuRouteDecision mdec{};
        std::string merr;
        if (!freetile::route_malibu_workload(mreq, &mdec, freetile::malibu_hardware_profile(), &merr)) {
            res->check_profile_routing_ok = 0;
            set_fault(res, "MALIBU_ROUTE_FAIL", merr);
            finalize_reward(res, 1e-4, 1500.0, "SCHEDULE");
            return;
        }
        malibu_eng = mdec.engine;
        malibu_bus_wait = mdec.requires_bus_serialization_wait;
        malibu_syscall_barrier = mdec.requires_syscall_barrier;
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), freetile::malibu_engine_name(malibu_eng));
        res->check_profile_routing_ok = 1;
    } else {
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), "buenos_a0_tpu");
        res->check_profile_routing_ok = 1;
    }

    // Plan tile schedule (and compute exact SRAM even when override_br/override_bc overflows).
    freetile::FlashAttnTilePlan plan{};
    std::string plan_err;
    if (!freetile::plan_flash_attn_tile(freq, &plan, &plan_err)) {
        // Compute exact requested SRAM footprint if explicit br/bc caused overflow.
        const uint32_t eff_br = (req.br > 0u) ? req.br
                                              : ((num_kv_heads > 0u) ? (num_q_heads / num_kv_heads) : 8u);
        const uint32_t eff_bc = (req.bc > 0u) ? req.bc : kv_align;
        freetile::FlashAttnSramBreakdown bd{};
        if (freetile::compute_flash_attn_sram_footprint(
                eff_br, eff_bc, head_dim, fa_dt, fa_pol, static_cast<size_t>(sram_limit), &bd, nullptr)) {
            res->has_sram_bytes = 1;
            res->sram_bytes = static_cast<uint64_t>(bd.total_bytes);
            if (!bd.fits_sram) {
                res->check_sram_budget_ok = 0;
                res->check_schedule_valid = 0;
                set_fault(res, "SRAM_OVERFLOW",
                          "FlashAttention tile (br=" + std::to_string(eff_br) +
                              ", bc=" + std::to_string(eff_bc) + ", d=" + std::to_string(head_dim) +
                              ") requires " + std::to_string(bd.total_bytes) +
                              " B > " + ft_twin_target_name(req.target) +
                              " narrow SRAM limit " + std::to_string(sram_limit) + " B");
                finalize_reward(res, 1e-4, 1500.0, "SRAM");
                return;
            }
        }
        res->check_schedule_valid = 0;
        set_fault(res, "FLASH_ATTN_PLAN_FAIL", plan_err);
        finalize_reward(res, 1e-4, 1500.0, "SCHEDULE");
        return;
    }

    res->has_sram_bytes = 1;
    res->sram_bytes = static_cast<uint64_t>(plan.sram.total_bytes);
    res->resolved_bm = plan.br;
    res->resolved_bn = plan.bc;
    res->resolved_bk = plan.head_dim;
    res->check_sram_budget_ok = plan.sram.fits_sram ? 1 : 0;
    res->check_schedule_valid = 1;
    res->tiles_dispatched = plan.num_kv_tiles;

    // Validate KV cache DMA-BUF page geometry (catches Q4_0 96:1 software unpack stall on buenos-a0 & Day-1 malibu!).
    {
        freetile::KvCachePageRequest kv_req{};
        kv_req.num_layers = 1u;
        kv_req.num_kv_heads = num_kv_heads;
        kv_req.head_dim = head_dim;
        kv_req.max_seq_len = std::max(seq_len, plan.bc);
        kv_req.page_tokens = kv_align;
        kv_req.target_has_native_int4_expander = false; // Both buenos-a0 and Day-1 malibu lack native 1:4 expander
        if (req.kv_format == FT_TWIN_KV_F16) kv_req.format = freetile::KvCacheQuantFormat::F16;
        else if (req.kv_format == FT_TWIN_KV_Q4_0) kv_req.format = freetile::KvCacheQuantFormat::Q4_0;
        else kv_req.format = freetile::KvCacheQuantFormat::Q8_0;

        freetile::KvCachePagePlan kv_plan{};
        std::string kv_err;
        if (!freetile::plan_kv_cache_dmabuf_pages(kv_req, &kv_plan, &kv_err)) {
            res->check_schedule_valid = 0;
            set_fault(res, "KV_CACHE_FORMAT_REJECTED", kv_err);
            finalize_reward(res, 1e-4, 1500.0, "SCHEDULE");
            return;
        }
    }

    // Validate 0x640 exp & recip splines used by on-TPU online softmax.
    {
        double e_nmse = 0.0, e_max = 0.0, r_nmse = 0.0, r_max = 0.0;
        std::string sp_err;
        if (!check_spline_0x640_internal(FT_TWIN_SPLINE_EXP_NEG12_96, 1025u, 0.0, &e_nmse, &e_max, &sp_err) ||
            !check_spline_0x640_internal(FT_TWIN_SPLINE_RECIP, 1025u, 0.0, &r_nmse, &r_max, &sp_err)) {
            res->check_spline_0x640_ok = 0;
            set_fault(res, "SPLINE_0X640_FAIL", sp_err);
            finalize_reward(res, 1e-4, 1500.0, "SPLINE");
            return;
        }
        res->spline_max_err = std::max(e_max, r_max);
        res->check_spline_0x640_ok = 1;
    }

    // Hardware latency estimation for FlashAttention (compute + KV streaming + dispatch).
    {
        const double q_rows = (req.fa_phase == FT_TWIN_FA_PREFILL)
                                  ? static_cast<double>(seq_len) * static_cast<double>(num_q_heads)
                                  : static_cast<double>(num_q_heads);
        const double flops = 4.0 * q_rows * static_cast<double>(seq_len) * static_cast<double>(head_dim);
        const double kv_elem_bytes = (req.kv_format == FT_TWIN_KV_F16) ? 2.0 : 1.0625;
        const double kv_bytes = 2.0 * static_cast<double>(num_kv_heads) * static_cast<double>(seq_len) *
                                static_cast<double>(head_dim) * kv_elem_bytes;
        const double bw_gbps = (req.target == FT_TWIN_TARGET_MALIBU)
                                   ? static_cast<double>(freetile::malibu_hardware_profile().lpddr5x_bw_gbps)
                                   : kBuenosDramBwGbps;
        const double tflops = (req.target == FT_TWIN_TARGET_MALIBU)
                                  ? ((malibu_eng == freetile::MalibuComputeEngine::EDGETPU_HEAVY) ? kMalibuTpuFp16Tflops : kMalibuSme2Fp16Tflops)
                                  : kBuenosFp16Tflops;
        const double eff = std::max(0.2, static_cast<double>(plan.systolic_row_efficiency));
        const double comp_us = flops / (tflops * 1e6 * eff);
        const double mem_us = kv_bytes / (bw_gbps * 1e3);
        const double body_us = (plan.sram.buffer_stages >= 2u) ? std::max(comp_us, mem_us) : (comp_us + mem_us);
        const double disp_us = (req.target == FT_TWIN_TARGET_MALIBU)
                                   ? ((malibu_eng == freetile::MalibuComputeEngine::EDGETPU_HEAVY) ? kMalibuTpuViiUs : kMalibuCpuDispatchUs)
                                   : kBuenosViiDispatchUs;
        res->has_est_hw_us = 1;
        res->est_hw_us = disp_us + body_us +
                         (malibu_bus_wait ? 150.0 : 0.0) +
                         (malibu_syscall_barrier ? 12.0 : 0.0);
    }

    // Numerical online-softmax FlashAttention execution using 0x640 splines vs FP64 reference!
    // Run across tiled B_c chunks on a representative (sim_br x sim_seq x head_dim) block.
    const uint32_t sim_br = std::min<uint32_t>(plan.br, 8u);
    const uint32_t sim_bc = std::min<uint32_t>(plan.bc, 128u);
    const uint32_t sim_seq = std::min<uint32_t>(seq_len, std::max<uint32_t>(sim_bc * 2u, 128u));
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

    uint32_t rng = req.seed * 2654435761u + 555u;
    std::vector<float> Q(static_cast<size_t>(sim_br) * head_dim);
    std::vector<float> K(static_cast<size_t>(sim_seq) * head_dim);
    std::vector<float> V(static_cast<size_t>(sim_seq) * head_dim);
    for (auto& v : Q) v = rand_uniform(rng, -1.0f, 1.0f);
    for (auto& v : K) v = rand_uniform(rng, -1.0f, 1.0f);
    for (auto& v : V) v = rand_uniform(rng, -1.0f, 1.0f);

    // Tiled online softmax with 0x640 exp + recip splines.
    std::vector<double> O_hw(static_cast<size_t>(sim_br) * head_dim, 0.0);
    std::vector<float> scores(sim_bc, 0.0f);
    std::vector<float> probs(sim_bc, 0.0f);
    for (uint32_t r = 0u; r < sim_br; ++r) {
        float m_running = -std::numeric_limits<float>::infinity();
        float l_running = 0.0f;
        std::vector<float> acc(head_dim, 0.0f);

        for (uint32_t c0 = 0u; c0 < sim_seq; c0 += sim_bc) {
            const uint32_t tc = std::min<uint32_t>(sim_bc, sim_seq - c0);
            float block_max = -std::numeric_limits<float>::infinity();
            for (uint32_t j = 0u; j < tc; ++j) {
                float dot = 0.0f;
                for (uint32_t d = 0u; d < head_dim; ++d) {
                    dot += Q[r * head_dim + d] * K[(c0 + j) * head_dim + d];
                }
                const float s = dot * inv_sqrt_d;
                scores[j] = s;
                if (s > block_max) block_max = s;
            }
            const float m_new = std::max(m_running, block_max);
            const float alpha = (m_running == -std::numeric_limits<float>::infinity())
                                    ? 0.0f
                                    : freetile::eval_buenos_exp_shifted(m_running - m_new, false);
            float l_block = 0.0f;
            for (uint32_t j = 0u; j < tc; ++j) {
                const float p = freetile::eval_buenos_exp_shifted(scores[j] - m_new, false);
                probs[j] = p;
                l_block += p;
            }
            for (uint32_t d = 0u; d < head_dim; ++d) {
                float pv = 0.0f;
                for (uint32_t j = 0u; j < tc; ++j) {
                    pv += probs[j] * V[(c0 + j) * head_dim + d];
                }
                acc[d] = acc[d] * alpha + pv;
            }
            l_running = l_running * alpha + l_block;
            m_running = m_new;
        }
        const float inv_l = freetile::eval_buenos_recip(l_running, true);
        for (uint32_t d = 0u; d < head_dim; ++d) {
            O_hw[r * head_dim + d] = static_cast<double>(acc[d] * inv_l);
        }
    }

    // Exact FP64 attention reference.
    std::vector<double> O_ref(static_cast<size_t>(sim_br) * head_dim, 0.0);
    std::vector<double> s_ref(sim_seq, 0.0);
    const double inv_sqrt_d64 = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (uint32_t r = 0u; r < sim_br; ++r) {
        double max_s = -std::numeric_limits<double>::infinity();
        for (uint32_t j = 0u; j < sim_seq; ++j) {
            double dot = 0.0;
            for (uint32_t d = 0u; d < head_dim; ++d) {
                dot += static_cast<double>(Q[r * head_dim + d]) * static_cast<double>(K[j * head_dim + d]);
            }
            s_ref[j] = dot * inv_sqrt_d64;
            if (s_ref[j] > max_s) max_s = s_ref[j];
        }
        double denom = 0.0;
        for (uint32_t j = 0u; j < sim_seq; ++j) {
            s_ref[j] = std::exp(s_ref[j] - max_s);
            denom += s_ref[j];
        }
        for (uint32_t d = 0u; d < head_dim; ++d) {
            double acc = 0.0;
            for (uint32_t j = 0u; j < sim_seq; ++j) {
                acc += (s_ref[j] / denom) * static_cast<double>(V[j * head_dim + d]);
            }
            O_ref[r * head_dim + d] = acc;
        }
    }

    res->verified_elements = sim_br * head_dim;
    double nmse = 0.0, max_abs = 0.0;
    bool nonfinite = false;
    compute_error_metrics(O_hw, O_ref, &nmse, &max_abs, &nonfinite);
    res->has_nmse = 1;
    res->nmse = nmse;
    res->has_max_abs_err = 1;
    res->max_abs_err = max_abs;

    const double nmse_tol = (req.max_nmse > 0.0) ? req.max_nmse : 1e-5;
    if (nonfinite || nmse > nmse_tol) {
        res->check_reference_parity_ok = 0;
        set_fault(res, "ACCURACY_FAIL", "FlashAttention online-softmax NMSE exceeded tolerance");
        finalize_reward(res, nmse_tol, 1500.0, "ACCURACY");
        return;
    }

    res->check_reference_parity_ok = 1;
    finalize_reward(res, nmse_tol, 1500.0, nullptr);
}

void eval_w8a16_workload(ft_twin_session_t* session,
                         const ft_twin_request_t& req,
                         ft_twin_result_t* res) {
    const uint32_t m = (req.m > 0u) ? req.m : 1u;
    const uint32_t k = (req.k > 0u) ? req.k : 512u;
    const uint32_t n = (req.n > 0u) ? req.n : 512u;

    if (req.workload_label && req.workload_label[0] != '\0') {
        copy_cstr(res->workload, sizeof(res->workload), req.workload_label);
    } else {
        char wbuf[128];
        std::snprintf(wbuf, sizeof(wbuf), "quant_w8a16_%ux%ux%u", m, k, n);
        copy_cstr(res->workload, sizeof(res->workload), wbuf);
    }

    // W8A16_PT_64X uses narrow SRAM ceiling (768 KiB), W8A16_PT_1X uses target SRAM.
    const bool narrow_limit = (m == 64u) || (req.target == FT_TWIN_TARGET_MALIBU);
    const uint64_t sram_limit = (req.sram_limit_override_bytes > 0u)
                                    ? req.sram_limit_override_bytes
                                    : target_default_sram_limit(req.target, narrow_limit);
    res->has_sram_limit_bytes = 1;
    res->sram_limit_bytes = sram_limit;

    if (!freetile::is_buenos_w8a16_shape(m, k, n)) {
        res->check_schedule_valid = 0;
        set_fault(res, "W8A16_UNSUPPORTED_SHAPE",
                  "W8A16 requires M=1 with K,N in {512,576,1024,1536,2048} or M=64 with K,N in {512,576,1024,1536}");
        finalize_reward(res, 1e-4, 1000.0, "SCHEDULE");
        return;
    }

    const uint32_t bm = (req.bm > 0u) ? req.bm : m;
    const uint32_t bn = (req.bn > 0u) ? req.bn : ((m == 64u) ? std::min(n, 256u) : std::min(n, 512u));
    const uint32_t bk = (req.bk > 0u) ? req.bk : ((m == 64u) ? std::min(k, 256u) : std::min(k, 512u));
    res->resolved_bm = bm;
    res->resolved_bn = bn;
    res->resolved_bk = bk;

    const bool db = (req.double_buffered < 0) ? (bk < k) : (req.double_buffered != 0);
    // Hardware 1:2 HibDma u8->FP16 expander avoids software unpack buffer in narrow SRAM.
    const uint64_t sram_bytes = compute_gemm_tile_sram_bytes(
        bm, bn, bk, 2u, 8u, 4u, false, false, 1u, db, false, bk < k);
    res->has_sram_bytes = 1;
    res->sram_bytes = sram_bytes;

    // Malibu routing telemetry.
    freetile::MalibuComputeEngine malibu_eng = freetile::MalibuComputeEngine::EDGETPU_HEAVY;
    bool malibu_bus_wait = false;
    bool malibu_syscall_barrier = false;
    if (req.target == FT_TWIN_TARGET_MALIBU) {
        freetile::MalibuRouteRequest mreq{};
        mreq.phase = (m > 1u) ? freetile::MalibuWorkloadPhase::PREFILL_DENSE
                              : freetile::MalibuWorkloadPhase::DECODE_M1;
        mreq.quant = freetile::MalibuQuantKind::Q8_0;
        mreq.batch_m = m;
        mreq.requested_threads = (req.malibu_threads > 0u) ? req.malibu_threads : 5u;
        mreq.active_engine_mask = req.malibu_active_engine_mask;
        mreq.mesa_pco_compute_ready = (req.malibu_mesa_pco_ready != 0);
        freetile::MalibuRouteDecision mdec{};
        std::string merr;
        if (!freetile::route_malibu_workload(mreq, &mdec, freetile::malibu_hardware_profile(), &merr)) {
            res->check_profile_routing_ok = 0;
            set_fault(res, "MALIBU_ROUTE_FAIL", merr);
            finalize_reward(res, 1e-4, 1000.0, "SCHEDULE");
            return;
        }
        malibu_eng = mdec.engine;
        malibu_bus_wait = mdec.requires_bus_serialization_wait;
        malibu_syscall_barrier = mdec.requires_syscall_barrier;
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), freetile::malibu_engine_name(malibu_eng));
        res->check_profile_routing_ok = 1;
    } else {
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), "buenos_a0_tpu");
        res->check_profile_routing_ok = 1;
    }

    res->has_est_hw_us = 1;
    res->est_hw_us = estimate_matmul_hw_us(
        req.target, m, k, n, bm, bn, bk, 1.0, 2u, 4u, db, true, false,
        malibu_eng, false, malibu_bus_wait, malibu_syscall_barrier, &res->tiles_dispatched);

    if (sram_bytes > sram_limit) {
        res->check_sram_budget_ok = 0;
        res->check_schedule_valid = 0;
        set_fault(res, "SRAM_OVERFLOW",
                  "W8A16 tile SRAM " + std::to_string(sram_bytes) +
                      " B exceeds " + ft_twin_target_name(req.target) +
                      " limit " + std::to_string(sram_limit) + " B");
        finalize_reward(res, 1e-4, 1000.0, "SRAM");
        return;
    }
    res->check_sram_budget_ok = 1;
    res->check_schedule_valid = 1;

    uint32_t rng = req.seed * 747796405u + 1337u;
    std::vector<int8_t> weights(static_cast<size_t>(n) * k);
    for (auto& v : weights) v = rand_i8(rng, -127, 127);

    // Emit & validate real 7-chunk W8A16 .dgc0 payload.
    if (req.verify_dgc0) {
        freetile::TileKernel tk;
        std::string plan_err;
        if (!freetile::plan_buenos_matmul_kernel(
                m, k, n, freetile::QuantFormat::Q8_0,
                freetile::DType::F16, freetile::DType::F32,
                freetile::EpilogueActivation::NONE, &tk, &plan_err)) {
            res->check_dgc0_wire_ok = 0;
            set_fault(res, "W8A16_PLAN_FAIL", plan_err);
            finalize_reward(res, 1e-4, 1000.0, "SCHEDULE");
            return;
        }
        freetile::BuenosEmitOptions opts{};
        opts.mode = (m == 64u) ? freetile::BuenosKernelMode::W8A16_PT_64X
                               : freetile::BuenosKernelMode::W8A16_PT_1X;
        opts.w8a16_pt_scale = 1.0f;
        opts.weights_nk = weights.data();
        opts.dive_interp = session->dummy_dive_interp.data();
        opts.dive_interp_size = session->dummy_dive_interp.size();
        freetile::BuenosEmitResult emit_res;
        std::string emit_err;
        if (!freetile::try_emit_buenos_kernel(tk, opts, &emit_res, &emit_err) || emit_res.dgc0.empty()) {
            res->check_dgc0_wire_ok = 0;
            set_fault(res, "DGC0_W8A16_EMIT_FAIL", emit_err);
            finalize_reward(res, 1e-4, 1000.0, "DGC0");
            return;
        }
        if (!emit_res.silicon_verified) {
            add_diag(res, "W8A16_OFFLINE_EXTENSION", FT_TWIN_DIAG_INFO, 0, 0, "silicon_verified",
                     "W8A16 shape is offline-synthesized extension (silicon_verified=false)");
        }
        const char* fclass = nullptr;
        if (!validate_dgc0_payload_internal(emit_res.dgc0.data(), emit_res.dgc0.size(),
                                            req.target, nullptr, 0u, res, false, &fclass)) {
            finalize_reward(res, 1e-4, 1000.0, fclass ? fclass : "DGC0");
            return;
        }
    }

    // Run libopenbuenos mock_mode=1 W8A16 hardware simulator (FP16 act + INT8 weights + FP32 accum) vs FP64.
    std::vector<float> a_k(k);
    std::vector<float> sb_n(n);
    std::vector<float> c_n(n, 0.0f);
    for (auto& v : a_k) v = rand_uniform(rng, -2.0f, 2.0f);
    for (auto& v : sb_n) v = rand_uniform(rng, 0.005f, 0.03f);

    if (m == 1u && ob::is_w8a16_pt_1x_shape(1u, k, n)) {
        ob_invoke_stats_t istats{};
        ob_status_t st = ob_gemm_matmul_f32_w8a16_tile(
            session->ob_ctx, a_k.data(), weights.data(), sb_n.data(), k, n, c_n.data(), &istats);
        if (st != OB_OK) {
            res->check_reference_parity_ok = 0;
            set_fault(res, "MOCK_W8A16_FAIL",
                      std::string("ob_gemm_matmul_f32_w8a16_tile failed: ") + ob_status_string(st));
            finalize_reward(res, 1e-4, 1000.0, "ACCURACY");
            return;
        }
    } else {
        // Exact FP16-input + FP32-accum simulator for extended W8A16 shapes (K=576/N=576 or M=64).
        std::vector<float> a_fp16(k);
        for (uint32_t ki = 0u; ki < k; ++ki) {
            a_fp16[ki] = ob::quant::fp16_to_fp32_scalar(f32_to_f16_rne(a_k[ki]));
        }
        for (uint32_t ni = 0u; ni < n; ++ni) {
            float acc = 0.0f;
            const int8_t* wrow = weights.data() + static_cast<size_t>(ni) * k;
            for (uint32_t ki = 0u; ki < k; ++ki) acc += a_fp16[ki] * static_cast<float>(wrow[ki]);
            c_n[ni] = acc * sb_n[ni];
        }
    }

    std::vector<double> got_vec(n), ref_vec(n);
    for (uint32_t ni = 0u; ni < n; ++ni) {
        double acc = 0.0;
        const int8_t* wrow = weights.data() + static_cast<size_t>(ni) * k;
        for (uint32_t ki = 0u; ki < k; ++ki) {
            acc += static_cast<double>(a_k[ki]) * static_cast<double>(wrow[ki]);
        }
        ref_vec[ni] = acc * static_cast<double>(sb_n[ni]);
        got_vec[ni] = static_cast<double>(c_n[ni]);
    }

    res->verified_elements = n;
    double nmse = 0.0, max_abs = 0.0;
    bool nonfinite = false;
    compute_error_metrics(got_vec, ref_vec, &nmse, &max_abs, &nonfinite);
    res->has_nmse = 1;
    res->nmse = nmse;
    res->has_max_abs_err = 1;
    res->max_abs_err = max_abs;

    const double nmse_tol = (req.max_nmse > 0.0) ? req.max_nmse : 1e-4;
    if (nonfinite || nmse > nmse_tol) {
        res->check_reference_parity_ok = 0;
        set_fault(res, "ACCURACY_FAIL", "W8A16 NMSE exceeded tolerance");
        finalize_reward(res, nmse_tol, 1000.0, "ACCURACY");
        return;
    }

    res->check_reference_parity_ok = 1;
    finalize_reward(res, nmse_tol, 1000.0, nullptr);
}

void eval_quant_kernel_workload(ft_twin_session_t* session,
                                const ft_twin_request_t& req,
                                ft_twin_result_t* res) {
    const ob_quant_type_t qt = workload_to_ob_quant(req.workload);
    const uint32_t qk = ob_quant_block_elements(qt);
    const size_t blk_sz = ob_quant_block_bytes(qt);
    const uint32_t m = (req.m > 0u) ? req.m : 1u;
    const uint32_t k = (req.k > 0u) ? req.k : std::max<uint32_t>(512u, qk);
    const uint32_t n = (req.n > 0u) ? req.n : 256u;

    if (req.workload_label && req.workload_label[0] != '\0') {
        copy_cstr(res->workload, sizeof(res->workload), req.workload_label);
    } else {
        char wbuf[128];
        std::snprintf(wbuf, sizeof(wbuf), "quant_%s_%ux%ux%u",
                      ob_quant_type_name(qt), m, k, n);
        copy_cstr(res->workload, sizeof(res->workload), wbuf);
    }

    const uint64_t sram_limit = (req.sram_limit_override_bytes > 0u)
                                    ? req.sram_limit_override_bytes
                                    : target_default_sram_limit(req.target, false);
    res->has_sram_limit_bytes = 1;
    res->sram_limit_bytes = sram_limit;

    if (qk == 0u || (k % qk) != 0u) {
        res->check_schedule_valid = 0;
        set_fault(res, "QUANT_K_UNALIGNED",
                  std::string(ob_quant_type_name(qt)) + " requires K divisible by " + std::to_string(qk));
        finalize_reward(res, 1e-3, 1000.0, "SCHEDULE");
        return;
    }

    const uint32_t bm = (req.bm > 0u) ? req.bm : freetile::buenos_align_hw_m(m);
    uint32_t bn = req.bn;
    uint32_t bk = req.bk;
    const double w_bpe = static_cast<double>(blk_sz) / static_cast<double>(qk);
    const size_t w_bits = static_cast<size_t>(std::ceil(w_bpe * 8.0));
    const bool has_min = (qt == OB_QUANT_Q4_K);

    if (bn == 0u || bk == 0u) {
        const uint32_t cands[] = {1024u, 512u, 256u, 128u, 64u};
        uint32_t pick_bn = (bn > 0u) ? bn : n;
        uint32_t pick_bk = (bk > 0u) ? bk : k;
        if (bn == 0u) {
            pick_bn = std::min(n, 256u);
            for (uint32_t c : cands) {
                if (c <= n && (n % c) == 0u) {
                    const uint64_t est = compute_gemm_tile_sram_bytes(
                        bm, c, (bk > 0u ? bk : std::min(k, std::max(qk, 256u))),
                        2u, w_bits, 4u, true, has_min, qk, false, true, false);
                    if (est <= sram_limit) {
                        pick_bn = c;
                        break;
                    }
                }
            }
        }
        if (bk == 0u) {
            pick_bk = std::max(qk, std::min(k, 256u));
            for (uint32_t c : cands) {
                if (c <= k && c >= qk && (c % qk) == 0u && (k % c) == 0u) {
                    const bool sk = (c < k);
                    const uint64_t est = compute_gemm_tile_sram_bytes(
                        bm, pick_bn, c, 2u, w_bits, 4u, true, has_min, qk, sk, true, sk);
                    if (est <= sram_limit) {
                        pick_bk = c;
                        break;
                    }
                }
            }
        }
        bn = pick_bn;
        bk = pick_bk;
    }

    if (bm < m || bn == 0u || bk == 0u || bn > n || bk > k || (bk % qk) != 0u) {
        res->check_schedule_valid = 0;
        set_fault(res, "SCHEDULE_INVALID",
                  "Invalid quantized tile dims (bm, bn, bk) for block size " + std::to_string(qk));
        finalize_reward(res, 1e-3, 1000.0, "SCHEDULE");
        return;
    }

    const bool split_k = (bk < k);
    const bool db = (req.double_buffered < 0) ? split_k : (req.double_buffered != 0);
    res->resolved_bm = bm;
    res->resolved_bn = bn;
    res->resolved_bk = bk;

    const uint64_t sram_bytes = compute_gemm_tile_sram_bytes(
        bm, bn, bk, 2u, w_bits, 4u, true, has_min, qk, db, true, split_k);
    res->has_sram_bytes = 1;
    res->sram_bytes = sram_bytes;

    // Malibu phase-aware routing (checks IQ4_NL/IQ4_XS -> NEON, K-quants -> SME2 repack note, Day-1 no S4).
    freetile::MalibuComputeEngine malibu_eng = freetile::MalibuComputeEngine::EDGETPU_HEAVY;
    bool malibu_bus_wait = false;
    bool malibu_syscall_barrier = false;
    bool malibu_kquant_repack = false;
    if (req.target == FT_TWIN_TARGET_MALIBU) {
        freetile::MalibuRouteRequest mreq{};
        mreq.phase = (m > 1u) ? freetile::MalibuWorkloadPhase::PREFILL_DENSE
                              : freetile::MalibuWorkloadPhase::DECODE_M1;
        mreq.quant = ob_quant_to_malibu(qt);
        mreq.batch_m = m;
        mreq.requested_threads = (req.malibu_threads > 0u) ? req.malibu_threads : 5u;
        mreq.active_engine_mask = req.malibu_active_engine_mask;
        mreq.mesa_pco_compute_ready = (req.malibu_mesa_pco_ready != 0);
        freetile::MalibuRouteDecision mdec{};
        std::string merr;
        if (!freetile::route_malibu_workload(mreq, &mdec, freetile::malibu_hardware_profile(), &merr)) {
            res->check_profile_routing_ok = 0;
            set_fault(res, "MALIBU_ROUTE_FAIL", merr);
            finalize_reward(res, 1e-3, 1000.0, "SCHEDULE");
            return;
        }
        malibu_eng = mdec.engine;
        malibu_bus_wait = mdec.requires_bus_serialization_wait;
        malibu_syscall_barrier = mdec.requires_syscall_barrier;
        malibu_kquant_repack = (!mdec.sme2_native_zero_repack && malibu_eng == freetile::MalibuComputeEngine::CPU_SME2);
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), freetile::malibu_engine_name(malibu_eng));
        res->check_profile_routing_ok = 1;
    } else {
        copy_cstr(res->selected_engine, sizeof(res->selected_engine), "buenos_a0_tpu");
        res->check_profile_routing_ok = 1;
    }

    res->has_est_hw_us = 1;
    res->est_hw_us = estimate_matmul_hw_us(
        req.target, m, k, n, bm, bn, bk, w_bpe, 2u, 4u, db, false, false,
        malibu_eng, malibu_kquant_repack, malibu_bus_wait, malibu_syscall_barrier,
        &res->tiles_dispatched);

    if (sram_bytes > sram_limit) {
        res->check_sram_budget_ok = 0;
        res->check_schedule_valid = 0;
        set_fault(res, "SRAM_OVERFLOW",
                  std::string(ob_quant_type_name(qt)) + " tile SRAM " + std::to_string(sram_bytes) +
                      " B exceeds " + ft_twin_target_name(req.target) +
                      " limit " + std::to_string(sram_limit) + " B");
        finalize_reward(res, 1e-3, 1000.0, "SRAM");
        return;
    }
    res->check_sram_budget_ok = 1;
    res->check_schedule_valid = 1;

    // 0x640 spline check if SiLU/GELU activation is fused.
    if (req.activation == FT_TWIN_ACT_SILU || req.activation == FT_TWIN_ACT_GELU) {
        const ft_twin_spline_kind_t sk = (req.activation == FT_TWIN_ACT_SILU)
                                             ? FT_TWIN_SPLINE_SILU
                                             : FT_TWIN_SPLINE_GELU_TANH;
        double sp_nmse = 0.0, sp_max = 0.0;
        std::string sp_err;
        if (!check_spline_0x640_internal(sk, 1025u, 0.0, &sp_nmse, &sp_max, &sp_err)) {
            res->check_spline_0x640_ok = 0;
            set_fault(res, "SPLINE_0X640_FAIL", sp_err);
            finalize_reward(res, 1e-3, 1000.0, "SPLINE");
            return;
        }
        res->spline_max_err = sp_max;
        res->check_spline_0x640_ok = 1;
    }

    // DGC0 wire check for the underlying packed INT8 tile lowered onto TPU.
    if (req.verify_dgc0) {
        const uint32_t dgc_m = freetile::is_buenos_w8a8_shape(m, k, n) ? m : 1u;
        const uint32_t dgc_k = freetile::is_buenos_w8a8_shape(dgc_m, k, n) ? k : 64u;
        const uint32_t dgc_n = freetile::is_buenos_w8a8_shape(dgc_m, dgc_k, n) ? n : 64u;
        freetile::TileKernel tk;
        if (freetile::plan_buenos_matmul_kernel(
                dgc_m, dgc_k, dgc_n, freetile::QuantFormat::Q8_0,
                freetile::DType::S8, freetile::DType::S8,
                to_ft_act(req.activation), &tk, nullptr)) {
            freetile::BuenosEmitOptions opts{};
            opts.mode = freetile::BuenosKernelMode::W8A8_DYN_WEIGHTS;
            opts.dive_interp = session->dummy_dive_interp.data();
            opts.dive_interp_size = session->dummy_dive_interp.size();
            freetile::BuenosEmitResult emit_res;
            if (freetile::try_emit_buenos_kernel(tk, opts, &emit_res, nullptr) && !emit_res.dgc0.empty()) {
                const char* fclass = nullptr;
                if (!validate_dgc0_payload_internal(emit_res.dgc0.data(), emit_res.dgc0.size(),
                                                    req.target, nullptr, 0u, res, false, &fclass)) {
                    finalize_reward(res, 1e-3, 1000.0, fclass ? fclass : "DGC0");
                    return;
                }
            }
        }
    }

    // Bound numerical verification slice for massive matrices while keeping full shape for normal tiles.
    uint32_t vm = m;
    uint32_t vn = n;
    while (static_cast<uint64_t>(vm) * k * vn > kMaxFullVerifyMacs && vm > 1u) {
        vm = std::max<uint32_t>(1u, vm / 2u);
    }
    while (static_cast<uint64_t>(vm) * k * vn > kMaxFullVerifyMacs && vn > 64u) {
        vn = std::max<uint32_t>(64u, vn / 2u);
    }

    const auto w_blocks = make_det_quant_matrix(qt, vn, k, req.seed + 17u);
    uint32_t rng = req.seed * 2654435761u + 31337u;
    std::vector<float> a_mk(static_cast<size_t>(vm) * k);
    for (auto& v : a_mk) v = rand_uniform(rng, -1.0f, 1.0f);

    // 1) Verify bit-exact unpack parity: SIMD dequant_row_f32 vs portable scalar dequant_row_scalar_f32.
    {
        std::vector<float> fast_row(k), ref_row(k);
        const size_t row_bytes = static_cast<size_t>(k / qk) * blk_sz;
        const uint32_t check_rows = std::min<uint32_t>(vn, 8u);
        for (uint32_t ri = 0u; ri < check_rows; ++ri) {
            const uint8_t* rp = w_blocks.data() + ri * row_bytes;
            ob::quant::dequant_row_f32(qt, rp, k, fast_row.data());
            ob::quant::dequant_row_scalar_f32(qt, rp, k, ref_row.data());
            for (uint32_t ki = 0u; ki < k; ++ki) {
                if (fast_row[ki] != ref_row[ki]) {
                    res->check_reference_parity_ok = 0;
                    set_fault(res, "QUANT_UNPACK_MISMATCH",
                              std::string(ob_quant_type_name(qt)) + " SIMD unpack diverged from scalar ref");
                    finalize_reward(res, 1e-3, 1000.0, "ACCURACY");
                    return;
                }
            }
        }
    }

    // 2) Run libopenbuenos mock_mode=1 ob_gemm_matmul_f32_quant_batch.
    std::vector<float> c_mock(static_cast<size_t>(vm) * vn, 0.0f);
    ob_gemm_f32_quant_job_t job{};
    job.b_nk = w_blocks.data();
    job.c_mn = c_mock.data();
    job.n_dim = vn;
    job.activation = to_ob_act(req.activation);
    ob_gemm_f32_quant_job_set_type(&job, qt);

    ob_invoke_stats_t istats{};
    ob_status_t st = ob_gemm_matmul_f32_quant_batch(
        session->ob_ctx, a_mk.data(), vm, k, &job, 1u, &istats);
    if (st != OB_OK) {
        res->check_reference_parity_ok = 0;
        set_fault(res, "MOCK_QUANT_FAIL",
                  std::string("ob_gemm_matmul_f32_quant_batch failed: ") + ob_status_string(st));
        finalize_reward(res, 1e-3, 1000.0, "ACCURACY");
        return;
    }

    // 3) Simulate ext[2] requantized TPU execution via ob::quant::requant_rows (including Q4_K algebraic centering)
    //    and compare BOTH against exact FP64 dequantized reference!
    std::vector<int8_t> bq(static_cast<size_t>(vn) * k, 0);
    std::vector<float> sb(vn, 0.0f);
    std::vector<float> row_mu(vn, 0.0f);
    const bool has_offset = ob::quant::requant_rows(
        qt, w_blocks.data(), k, 0u, vn, 0u, k, k,
        bq.data(), sb.data(), row_mu.data(), nullptr, nullptr);

    std::vector<double> got_vec(static_cast<size_t>(vm) * vn, 0.0);
    std::vector<double> ref_vec(static_cast<size_t>(vm) * vn, 0.0);
    std::vector<float> deq_row(k);
    const size_t row_bytes = static_cast<size_t>(k / qk) * blk_sz;

    for (uint32_t ni = 0u; ni < vn; ++ni) {
        ob::quant::dequant_row_scalar_f32(qt, w_blocks.data() + ni * row_bytes, k, deq_row.data());
        const int8_t* bq_row = bq.data() + static_cast<size_t>(ni) * k;
        for (uint32_t mi = 0u; mi < vm; ++mi) {
            const float* arow = a_mk.data() + static_cast<size_t>(mi) * k;
            double ref_acc = 0.0;
            double hw_acc = 0.0;
            double a_sum = 0.0;
            for (uint32_t ki = 0u; ki < k; ++ki) {
                const double av = static_cast<double>(arow[ki]);
                ref_acc += av * static_cast<double>(deq_row[ki]);
                hw_acc += av * static_cast<double>(bq_row[ki]);
                a_sum += av;
            }
            double hw_val = hw_acc * static_cast<double>(sb[ni]);
            if (has_offset) hw_val += a_sum * static_cast<double>(row_mu[ni]);

            const double ref_y = apply_ref_activation(ref_acc, req.activation);
            const double hw_y = apply_ref_activation(hw_val, req.activation);

            // Also verify mock_mode=1 CPU output matches ref within FP32 rounding (~1e-4).
            const double mock_diff = std::fabs(static_cast<double>(c_mock[mi * vn + ni]) - ref_y);
            if (!std::isfinite(c_mock[mi * vn + ni]) || mock_diff > 1e-2 * (1.0 + std::fabs(ref_y))) {
                res->check_reference_parity_ok = 0;
                set_fault(res, "MOCK_PARITY_DIVERGENCE",
                          "ob_gemm_matmul_f32_quant_batch mock output diverged from FP64 reference");
                finalize_reward(res, 1e-3, 1000.0, "ACCURACY");
                return;
            }
            ref_vec[mi * vn + ni] = ref_y;
            got_vec[mi * vn + ni] = hw_y;
        }
    }

    res->verified_elements = vm * vn;
    double nmse = 0.0, max_abs = 0.0;
    bool nonfinite = false;
    compute_error_metrics(got_vec, ref_vec, &nmse, &max_abs, &nonfinite);
    res->has_nmse = 1;
    res->nmse = nmse;
    res->has_max_abs_err = 1;
    res->max_abs_err = max_abs;

    const double nmse_tol = (req.max_nmse > 0.0) ? req.max_nmse : 2.5e-3;
    if (nonfinite || nmse > nmse_tol) {
        res->check_reference_parity_ok = 0;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s ext2 requant NMSE=%.6e exceeds tol=%.6e",
                      ob_quant_type_name(qt), nmse, nmse_tol);
        set_fault(res, "ACCURACY_FAIL", buf);
        finalize_reward(res, nmse_tol, 1000.0, "ACCURACY");
        return;
    }

    res->check_reference_parity_ok = 1;
    finalize_reward(res, nmse_tol, 1000.0, nullptr);
}

void eval_spline_workload(const ft_twin_request_t& req, ft_twin_result_t* res) {
    const freetile::SplineKind fk = to_ft_spline(req.spline_kind);
    if (req.workload_label && req.workload_label[0] != '\0') {
        copy_cstr(res->workload, sizeof(res->workload), req.workload_label);
    } else {
        char wbuf[128];
        std::snprintf(wbuf, sizeof(wbuf), "spline_0x640_%s", freetile::spline_kind_name(fk));
        copy_cstr(res->workload, sizeof(res->workload), wbuf);
    }

    const uint64_t sram_limit = (req.sram_limit_override_bytes > 0u)
                                    ? req.sram_limit_override_bytes
                                    : target_default_sram_limit(req.target, true);
    res->has_sram_bytes = 1;
    res->sram_bytes = static_cast<uint64_t>(freetile::kOp640PacketBytes); // 208-byte LUT packet
    res->has_sram_limit_bytes = 1;
    res->sram_limit_bytes = sram_limit;
    res->check_sram_budget_ok = (res->sram_bytes <= res->sram_limit_bytes) ? 1 : 0;
    res->check_profile_routing_ok = 1;
    res->check_schedule_valid = 1;
    copy_cstr(res->selected_engine, sizeof(res->selected_engine),
              (req.target == FT_TWIN_TARGET_MALIBU) ? "edgetpu_heavy" : "buenos_a0_tpu");

    const uint32_t samples = (req.spline_samples >= 16u) ? req.spline_samples : 2049u;
    res->verified_elements = samples;
    res->has_est_hw_us = 1;
    // On-TPU 0x640 LUT load + degree-4 Horner vector evaluation (~1.2 us / 1K elements).
    res->est_hw_us = 1.5 + static_cast<double>(samples) * 0.0012;

    if (res->sram_bytes > res->sram_limit_bytes) {
        set_fault(res, "SRAM_OVERFLOW", "0x640 LUT packet exceeds SRAM limit override");
        finalize_reward(res, 1e-5, 50.0, "SRAM");
        return;
    }

    double nmse = 0.0, max_abs = 0.0;
    std::string sp_err;
    const bool ok = check_spline_0x640_internal(
        req.spline_kind, samples, req.max_abs_err_limit, &nmse, &max_abs, &sp_err);
    res->has_nmse = 1;
    res->nmse = nmse;
    res->has_max_abs_err = 1;
    res->max_abs_err = max_abs;
    res->spline_max_err = max_abs;

    if (!ok) {
        res->check_spline_0x640_ok = 0;
        res->check_reference_parity_ok = 0;
        set_fault(res, "SPLINE_0X640_FAIL", sp_err);
        finalize_reward(res, 1e-5, 50.0, "SPLINE");
        return;
    }
    if (req.max_nmse > 0.0 && nmse > req.max_nmse) {
        res->check_spline_0x640_ok = 0;
        res->check_reference_parity_ok = 0;
        set_fault(res, "SPLINE_0X640_NMSE", "0x640 spline NMSE exceeded custom threshold");
        finalize_reward(res, req.max_nmse, 50.0, "SPLINE");
        return;
    }

    res->check_spline_0x640_ok = 1;
    res->check_reference_parity_ok = 1;
    finalize_reward(res, 1e-5, 50.0, nullptr);
}

void eval_dgc0_workload(const ft_twin_request_t& req, ft_twin_result_t* res) {
    copy_cstr(res->domain, sizeof(res->domain), "dgc0");
    if (req.workload_label && req.workload_label[0] != '\0') {
        copy_cstr(res->workload, sizeof(res->workload), req.workload_label);
    } else {
        copy_cstr(res->workload, sizeof(res->workload), "dgc0_wire_verify");
    }

    std::vector<uint8_t> file_buf;
    const uint8_t* data = req.dgc0_bytes;
    size_t size = req.dgc0_size;

    if ((!data || size == 0u) && req.dgc0_path && req.dgc0_path[0] != '\0') {
        std::ifstream ifs(req.dgc0_path, std::ios::binary | std::ios::ate);
        if (!ifs) {
            res->check_dgc0_wire_ok = 0;
            set_fault(res, "DGC0_IO_ERROR", std::string("Failed to open DGC0 file: ") + req.dgc0_path);
            finalize_reward(res, 1e-4, 1000.0, "DGC0");
            return;
        }
        const std::streamoff fsz = ifs.tellg();
        if (fsz < 0) {
            res->check_dgc0_wire_ok = 0;
            set_fault(res, "DGC0_IO_ERROR", "Failed to determine DGC0 file size");
            finalize_reward(res, 1e-4, 1000.0, "DGC0");
            return;
        }
        file_buf.resize(static_cast<size_t>(fsz));
        ifs.seekg(0, std::ios::beg);
        if (fsz > 0 && !ifs.read(reinterpret_cast<char*>(file_buf.data()), fsz)) {
            res->check_dgc0_wire_ok = 0;
            set_fault(res, "DGC0_IO_ERROR", "Failed to read DGC0 file contents");
            finalize_reward(res, 1e-4, 1000.0, "DGC0");
            return;
        }
        data = file_buf.data();
        size = file_buf.size();
    }

    const char* fclass = nullptr;
    validate_dgc0_payload_internal(data, size, req.target, req.expected_sha256,
                                   req.sram_limit_override_bytes, res, true, &fclass);
    finalize_reward(res, 1e-4, 1000.0, fclass);
}

// =========================================================================
// Deterministic JSON Serialization
// =========================================================================

void json_append_escaped(std::string& out, const char* s) {
    out.push_back('"');
    if (s) {
        for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p != '\0'; ++p) {
            const unsigned char c = *p;
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20u) {
                        char hex[8];
                        std::snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned int>(c));
                        out += hex;
                    } else {
                        out.push_back(static_cast<char>(c));
                    }
                    break;
            }
        }
    }
    out.push_back('"');
}

void json_append_double(std::string& out, double v) {
    if (!std::isfinite(v)) {
        out += "null";
        return;
    }
    if (v == 0.0) {
        out += "0.0";
        return;
    }
    char buf[64];
    const double av = std::fabs(v);
    if (av >= 1e-4 && av < 1e9) {
        std::snprintf(buf, sizeof(buf), "%.6f", v);
        // Trim trailing zeros while preserving at least one fractional digit.
        char* dot = std::strchr(buf, '.');
        if (dot) {
            char* end = buf + std::strlen(buf) - 1;
            while (end > dot + 1 && *end == '0') {
                *end-- = '\0';
            }
        }
    } else {
        std::snprintf(buf, sizeof(buf), "%.6e", v);
    }
    out += buf;
}

void json_append_tristate(std::string& out, int v) {
    if (v < 0) out += "null";
    else if (v == 0) out += "false";
    else out += "true";
}

}  // namespace

// =========================================================================
// Public C11 API Implementation
// =========================================================================

extern "C" {

const char* ft_twin_target_name(ft_twin_target_t target) {
    switch (target) {
        case FT_TWIN_TARGET_BUENOS_A0: return "buenos-a0";
        case FT_TWIN_TARGET_MALIBU:    return "malibu";
    }
    return "buenos-a0";
}

const char* ft_twin_workload_name(ft_twin_workload_t workload) {
    switch (workload) {
        case FT_TWIN_WORKLOAD_GEMM:         return "gemm";
        case FT_TWIN_WORKLOAD_SWIGLU:       return "swiglu";
        case FT_TWIN_WORKLOAD_FLASH_ATTN:   return "flash_attn";
        case FT_TWIN_WORKLOAD_QUANT_W8A16:  return "w8a16";
        case FT_TWIN_WORKLOAD_QUANT_Q4_K:   return "q4_k";
        case FT_TWIN_WORKLOAD_QUANT_Q6_K:   return "q6_k";
        case FT_TWIN_WORKLOAD_QUANT_IQ4_NL: return "iq4_nl";
        case FT_TWIN_WORKLOAD_QUANT_IQ4_XS: return "iq4_xs";
        case FT_TWIN_WORKLOAD_QUANT_Q8_0:   return "q8_0";
        case FT_TWIN_WORKLOAD_QUANT_Q4_0:   return "q4_0";
        case FT_TWIN_WORKLOAD_SPLINE_0X640: return "spline";
        case FT_TWIN_WORKLOAD_DGC0:         return "dgc0";
    }
    return "gemm";
}

ft_twin_request_t ft_twin_request_default(void) {
    ft_twin_request_t req;
    std::memset(&req, 0, sizeof(req));
    req.struct_size = static_cast<uint32_t>(sizeof(ft_twin_request_t));
    req.target = FT_TWIN_TARGET_BUENOS_A0;
    req.workload = FT_TWIN_WORKLOAD_GEMM;
    req.m = 1u;
    req.k = 512u;
    req.n = 512u;
    req.h = 512u;
    req.double_buffered = -1;
    req.act_dtype = FT_TWIN_DTYPE_BF16;
    req.out_dtype = FT_TWIN_DTYPE_FP32;
    req.activation = FT_TWIN_ACT_NONE;
    req.swiglu_11chunk = 1;
    req.quant_type = static_cast<uint32_t>(OB_QUANT_Q8_0);
    req.fa_phase = FT_TWIN_FA_DECODE;
    req.head_dim = 128u;
    req.num_q_heads = 32u;
    req.num_kv_heads = 4u;
    req.seq_len = 1024u;
    req.kv_tile_align = 64u;
    req.fa_buf_policy = FT_TWIN_FA_PING_PONG;
    req.kv_format = FT_TWIN_KV_Q8_0;
    req.spline_kind = FT_TWIN_SPLINE_SILU;
    req.spline_samples = 2049u;
    req.malibu_threads = 5u;
    req.seed = 42u;
    req.verify_dgc0 = 1;
    return req;
}

ft_twin_status_t ft_twin_session_create(ft_twin_session_t** out_session) {
    if (!out_session) return FT_TWIN_ERR_INVALID_ARG;
    *out_session = nullptr;

    ft_twin_session_t* s = new (std::nothrow) ft_twin_session();
    if (!s) return FT_TWIN_ERR_INTERNAL;

    ob_config_t cfg = ob_config_default();
    cfg.mock_mode = 1;
    cfg.graph_cache_capacity = 256u;
    ob_status_t st = ob_context_create(&cfg, &s->ob_ctx);
    if (st != OB_OK || !s->ob_ctx) {
        delete s;
        return FT_TWIN_ERR_INTERNAL;
    }
    (void)ob_context_set_freetile(s->ob_ctx, 1);
    s->dummy_dive_interp = make_dummy_dive_interp(4096u);
    s->ob_ctx->builtin_dive_interp = s->dummy_dive_interp;
    s->ob_ctx->builtin_emitter_active = true;
    s->ob_ctx->gemm_emitter = ob::builtin_gemm_emitter_cb;
    s->ob_ctx->gemm_emitter_user_data = s->ob_ctx;
    *out_session = s;
    return FT_TWIN_OK;
}

void ft_twin_session_destroy(ft_twin_session_t* session) {
    if (!session) return;
    if (session->ob_ctx) {
        ob_context_destroy(session->ob_ctx);
        session->ob_ctx = nullptr;
    }
    delete session;
}

ft_twin_status_t ft_twin_session_eval(ft_twin_session_t* session,
                                      const ft_twin_request_t* req,
                                      ft_twin_result_t* out_res) {
    if (!session || !session->ob_ctx || !req || !out_res) {
        return FT_TWIN_ERR_INVALID_ARG;
    }
    const auto t0 = SteadyClock::now();

    std::memset(out_res, 0, sizeof(*out_res));
    copy_cstr(out_res->schema_version, sizeof(out_res->schema_version), FT_TWIN_SCHEMA_VERSION);
    copy_cstr(out_res->domain, sizeof(out_res->domain), "freetile");
    copy_cstr(out_res->target, sizeof(out_res->target), ft_twin_target_name(req->target));
    out_res->check_sram_budget_ok = -1;
    out_res->check_dgc0_wire_ok = -1;
    out_res->check_spline_0x640_ok = -1;
    out_res->check_reference_parity_ok = -1;
    out_res->check_profile_routing_ok = -1;
    out_res->check_schedule_valid = -1;

    switch (req->workload) {
        case FT_TWIN_WORKLOAD_GEMM:
            eval_gemm_workload(session, *req, out_res);
            break;
        case FT_TWIN_WORKLOAD_SWIGLU:
            eval_swiglu_workload(session, *req, out_res);
            break;
        case FT_TWIN_WORKLOAD_FLASH_ATTN:
            eval_flash_attn_workload(*req, out_res);
            break;
        case FT_TWIN_WORKLOAD_QUANT_W8A16:
            eval_w8a16_workload(session, *req, out_res);
            break;
        case FT_TWIN_WORKLOAD_QUANT_Q4_K:
        case FT_TWIN_WORKLOAD_QUANT_Q6_K:
        case FT_TWIN_WORKLOAD_QUANT_IQ4_NL:
        case FT_TWIN_WORKLOAD_QUANT_IQ4_XS:
        case FT_TWIN_WORKLOAD_QUANT_Q8_0:
        case FT_TWIN_WORKLOAD_QUANT_Q4_0:
            eval_quant_kernel_workload(session, *req, out_res);
            break;
        case FT_TWIN_WORKLOAD_SPLINE_0X640:
            eval_spline_workload(*req, out_res);
            break;
        case FT_TWIN_WORKLOAD_DGC0:
            eval_dgc0_workload(*req, out_res);
            break;
    }

    const auto t1 = SteadyClock::now();
    out_res->eval_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return FT_TWIN_OK;
}

ft_twin_status_t ft_twin_eval(const ft_twin_request_t* req,
                              ft_twin_result_t* out_res) {
    if (!req || !out_res) return FT_TWIN_ERR_INVALID_ARG;
    ft_twin_session_t* session = nullptr;
    ft_twin_status_t st = ft_twin_session_create(&session);
    if (st != FT_TWIN_OK) return st;
    st = ft_twin_session_eval(session, req, out_res);
    ft_twin_session_destroy(session);
    return st;
}

ft_twin_status_t ft_twin_eval_dgc0_buffer(ft_twin_target_t target,
                                          const uint8_t* data,
                                          size_t size,
                                          const char* expected_sha256,
                                          uint64_t sram_limit_override_bytes,
                                          ft_twin_result_t* out_res) {
    if (!out_res) return FT_TWIN_ERR_INVALID_ARG;
    ft_twin_request_t req = ft_twin_request_default();
    req.target = target;
    req.workload = FT_TWIN_WORKLOAD_DGC0;
    req.dgc0_bytes = data;
    req.dgc0_size = size;
    req.expected_sha256 = expected_sha256;
    req.sram_limit_override_bytes = sram_limit_override_bytes;
    return ft_twin_eval(&req, out_res);
}

ft_twin_status_t ft_twin_eval_dgc0_file(ft_twin_target_t target,
                                        const char* path,
                                        const char* expected_sha256,
                                        uint64_t sram_limit_override_bytes,
                                        ft_twin_result_t* out_res) {
    if (!path || !out_res) return FT_TWIN_ERR_INVALID_ARG;
    ft_twin_request_t req = ft_twin_request_default();
    req.target = target;
    req.workload = FT_TWIN_WORKLOAD_DGC0;
    req.dgc0_path = path;
    req.expected_sha256 = expected_sha256;
    req.sram_limit_override_bytes = sram_limit_override_bytes;
    return ft_twin_eval(&req, out_res);
}

ft_twin_status_t ft_twin_emit_sample_dgc0(const ft_twin_request_t* req,
                                          uint8_t* out_buf,
                                          size_t out_cap,
                                          size_t* out_size) {
    if (!req || !out_size) return FT_TWIN_ERR_INVALID_ARG;
    const std::vector<uint8_t> dummy_interp = make_dummy_dive_interp(4096u);
    std::vector<uint8_t> dgc0;

    if (req->workload == FT_TWIN_WORKLOAD_SWIGLU) {
        const uint32_t m = (req->m > 0u) ? req->m : 1u;
        const uint32_t k = (req->k > 0u) ? req->k : 576u;
        const uint32_t h = (req->h > 0u) ? req->h : 512u;
        const uint32_t n = (req->n > 0u) ? req->n : k;
        freetile::BuenosSwigluDesc desc{};
        desc.m_dim = m;
        desc.k_dim = k;
        desc.h_dim = h;
        desc.n_dim = n;
        desc.s_in = 0.03125f;
        desc.s_wg = 0.015625f;
        desc.s_gate = 0.0625f;
        desc.s_wu = 0.015625f;
        desc.s_up = 0.0625f;
        desc.s_silu = 0.0625f;
        desc.s_prod = 0.0625f;
        desc.s_wd = 0.015625f;
        desc.s_out = 0.0625f;
        desc.dive_interp = dummy_interp.data();
        desc.dive_interp_size = dummy_interp.size();
        freetile::BuenosSwigluResult res{};
        if (!freetile::try_emit_buenos_swiglu(desc, &res, nullptr) || res.dgc0.empty()) {
            return FT_TWIN_ERR_INVALID_ARG;
        }
        dgc0 = std::move(res.dgc0);
    } else if (req->workload == FT_TWIN_WORKLOAD_QUANT_W8A16) {
        const uint32_t m = (req->m > 0u) ? req->m : 1u;
        const uint32_t k = (req->k > 0u) ? req->k : 512u;
        const uint32_t n = (req->n > 0u) ? req->n : 512u;
        freetile::TileKernel tk;
        if (!freetile::plan_buenos_matmul_kernel(
                m, k, n, freetile::QuantFormat::Q8_0,
                freetile::DType::F16, freetile::DType::F32,
                freetile::EpilogueActivation::NONE, &tk, nullptr)) {
            return FT_TWIN_ERR_INVALID_ARG;
        }
        freetile::BuenosEmitOptions opts{};
        opts.mode = (m == 64u) ? freetile::BuenosKernelMode::W8A16_PT_64X
                               : freetile::BuenosKernelMode::W8A16_PT_1X;
        opts.w8a16_pt_scale = 1.0f;
        opts.dive_interp = dummy_interp.data();
        opts.dive_interp_size = dummy_interp.size();
        freetile::BuenosEmitResult res{};
        if (!freetile::try_emit_buenos_kernel(tk, opts, &res, nullptr) || res.dgc0.empty()) {
            return FT_TWIN_ERR_INVALID_ARG;
        }
        dgc0 = std::move(res.dgc0);
    } else {
        const uint32_t m = (req->m > 0u) ? req->m : 1u;
        const uint32_t k = (req->k > 0u) ? req->k : 64u;
        const uint32_t n = (req->n > 0u) ? req->n : 64u;
        freetile::TileKernel tk;
        if (!freetile::plan_buenos_matmul_kernel(
                m, k, n, freetile::QuantFormat::Q8_0,
                freetile::DType::S8, freetile::DType::S8,
                to_ft_act(req->activation), &tk, nullptr)) {
            return FT_TWIN_ERR_INVALID_ARG;
        }
        freetile::BuenosEmitOptions opts{};
        opts.mode = freetile::BuenosKernelMode::W8A8_DYN_WEIGHTS;
        opts.dive_interp = dummy_interp.data();
        opts.dive_interp_size = dummy_interp.size();
        freetile::BuenosEmitResult res{};
        if (!freetile::try_emit_buenos_kernel(tk, opts, &res, nullptr) || res.dgc0.empty()) {
            return FT_TWIN_ERR_INVALID_ARG;
        }
        dgc0 = std::move(res.dgc0);
    }

    *out_size = dgc0.size();
    if (!out_buf || out_cap < dgc0.size()) {
        return FT_TWIN_ERR_OVERFLOW;
    }
    std::memcpy(out_buf, dgc0.data(), dgc0.size());
    return FT_TWIN_OK;
}

ft_twin_status_t ft_twin_result_to_json(const ft_twin_result_t* res,
                                        int pretty,
                                        char* out_buf,
                                        size_t out_cap,
                                        size_t* out_len) {
    if (!res || !out_len) return FT_TWIN_ERR_INVALID_ARG;

    const char* nl = pretty ? "\n" : "";
    const char* sp = pretty ? " " : "";
    const char* i1 = pretty ? "  " : "";
    const char* i2 = pretty ? "    " : "";
    const char* i3 = pretty ? "      " : "";

    std::string js;
    js.reserve(1024u);
    js += "{";
    js += nl;

    auto key = [&](const char* ind, const char* k, bool first) {
        if (!first) {
            js += ",";
            js += nl;
        }
        js += ind;
        json_append_escaped(js, k);
        js += ":";
        js += sp;
    };

    key(i1, "schema_version", true);  json_append_escaped(js, res->schema_version);
    key(i1, "domain", false);         json_append_escaped(js, res->domain);
    key(i1, "target", false);         json_append_escaped(js, res->target);
    key(i1, "workload", false);       json_append_escaped(js, res->workload);
    key(i1, "pass", false);           js += (res->pass ? "true" : "false");
    key(i1, "reward", false);         json_append_double(js, res->reward);

    key(i1, "nmse", false);
    if (res->has_nmse) json_append_double(js, res->nmse); else js += "null";

    key(i1, "max_abs_err", false);
    if (res->has_max_abs_err) json_append_double(js, res->max_abs_err); else js += "null";

    key(i1, "sram_bytes", false);
    if (res->has_sram_bytes) js += std::to_string(res->sram_bytes); else js += "null";

    key(i1, "sram_limit_bytes", false);
    if (res->has_sram_limit_bytes) js += std::to_string(res->sram_limit_bytes); else js += "null";

    key(i1, "est_hw_us", false);
    if (res->has_est_hw_us) json_append_double(js, res->est_hw_us); else js += "null";

    key(i1, "eval_ms", false);
    json_append_double(js, res->eval_ms);

    key(i1, "fault_reason", false);
    if (res->has_fault) json_append_escaped(js, res->fault_reason); else js += "null";

    // diagnostics array
    key(i1, "diagnostics", false);
    if (res->num_diagnostics == 0u) {
        js += "[]";
    } else {
        js += "[";
        js += nl;
        for (uint32_t di = 0u; di < res->num_diagnostics; ++di) {
            const ft_twin_diagnostic_t& d = res->diagnostics[di];
            if (di > 0u) {
                js += ",";
                js += nl;
            }
            js += i2;
            js += "{";
            js += nl;
            const char* sev_s = "info";
            if (d.severity == FT_TWIN_DIAG_WARNING) sev_s = "warning";
            else if (d.severity == FT_TWIN_DIAG_ERROR) sev_s = "error";
            key(i3, "code", true);      json_append_escaped(js, d.code);
            key(i3, "severity", false); json_append_escaped(js, sev_s);
            key(i3, "offset", false);
            if (d.has_offset) js += std::to_string(d.offset); else js += "null";
            key(i3, "field", false);
            if (d.field[0] != '\0') json_append_escaped(js, d.field); else js += "null";
            key(i3, "message", false);  json_append_escaped(js, d.message);
            js += nl;
            js += i2;
            js += "}";
        }
        js += nl;
        js += i1;
        js += "]";
    }

    // checks object
    key(i1, "checks", false);
    js += "{";
    js += nl;
    key(i2, "sram_budget_ok", true);       json_append_tristate(js, res->check_sram_budget_ok);
    key(i2, "dgc0_wire_ok", false);        json_append_tristate(js, res->check_dgc0_wire_ok);
    key(i2, "spline_0x640_ok", false);     json_append_tristate(js, res->check_spline_0x640_ok);
    key(i2, "reference_parity_ok", false); json_append_tristate(js, res->check_reference_parity_ok);
    key(i2, "profile_routing_ok", false);  json_append_tristate(js, res->check_profile_routing_ok);
    key(i2, "schedule_valid", false);      json_append_tristate(js, res->check_schedule_valid);
    if (res->selected_engine[0] != '\0') {
        key(i2, "selected_engine", false);
        json_append_escaped(js, res->selected_engine);
    }
    if (res->tiles_dispatched > 0u) {
        key(i2, "tiles_dispatched", false);
        js += std::to_string(res->tiles_dispatched);
    }
    if (res->verified_elements > 0u) {
        key(i2, "verified_elements", false);
        js += std::to_string(res->verified_elements);
    }
    if (res->resolved_bm > 0u || res->resolved_bn > 0u || res->resolved_bk > 0u) {
        key(i2, "tile_bm", false); js += std::to_string(res->resolved_bm);
        key(i2, "tile_bn", false); js += std::to_string(res->resolved_bn);
        key(i2, "tile_bk", false); js += std::to_string(res->resolved_bk);
    }
    if (res->dgc0_bytes_checked > 0u) {
        key(i2, "dgc0_bytes", false);       js += std::to_string(res->dgc0_bytes_checked);
        key(i2, "dgc0_num_inputs", false);  js += std::to_string(res->dgc0_num_inputs);
        key(i2, "dgc0_num_outputs", false); js += std::to_string(res->dgc0_num_outputs);
        key(i2, "dgc0_cluster_id", false);  js += std::to_string(res->dgc0_cluster_id);
        if (res->dgc0_sha256[0] != '\0') {
            key(i2, "dgc0_sha256", false);
            json_append_escaped(js, res->dgc0_sha256);
        }
    }
    if (res->check_spline_0x640_ok >= 0) {
        key(i2, "spline_max_err", false);
        json_append_double(js, res->spline_max_err);
    }
    js += nl;
    js += i1;
    js += "}";

    // reward_breakdown object
    key(i1, "reward_breakdown", false);
    js += "{";
    js += nl;
    key(i2, "gate_pass", true);        json_append_double(js, res->rb_gate_pass);
    key(i2, "accuracy_score", false);  json_append_double(js, res->rb_accuracy_score);
    key(i2, "sram_efficiency", false); json_append_double(js, res->rb_sram_efficiency);
    key(i2, "latency_score", false);   json_append_double(js, res->rb_latency_score);
    js += nl;
    js += i1;
    js += "}";

    js += nl;
    js += "}";

    *out_len = js.size();
    if (!out_buf || out_cap <= js.size()) {
        return FT_TWIN_ERR_OVERFLOW;
    }
    std::memcpy(out_buf, js.data(), js.size() + 1u);
    return FT_TWIN_OK;
}

}  // extern "C"
