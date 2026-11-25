// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "freetile/tile_ir.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace freetile {

enum class BuenosKernelMode {
    AUTO = 0,
    W8A8_INT8,         // 7-chunk systolic INT8 GEMM (general multi-tile, lt_1x*, lt_64x*, Mode-A SiLU/GELU)
    W8A8_DYN_WEIGHTS,  // Weight-agnostic 2-input 0-param DGC0 (packed ext[2] passed dynamically via input_1)
    W8A16_PT_1X,       // 7-chunk per-tensor W8A16 (FP16 act + HibDma 1:2 u8->FP16 weights, FP32 I/O & accum, M=1)
    W8A16_PT_64X,      // 7-chunk per-tensor W8A16 batched prefill (M=64, FP16 act + FP32 accum, zero FP16 mid overflow)
};

const char* buenos_kernel_mode_name(BuenosKernelMode mode);

struct BuenosEmitOptions {
    BuenosKernelMode mode = BuenosKernelMode::AUTO;

    // W8A8_INT8 / W8A8_DYN_WEIGHTS quantization parameters.
    float s_in = 0.03125f;
    int32_t zp_in = 0;
    float s_w = 0.015625f;
    int32_t zp_w = 0;
    float s_out = 0.0625f;
    int32_t zp_out = 0;

    // Optional decoupled intermediate quantization for Mode-A 0x640 SiLU/GELU fusion.
    // When s_mid <= 0.0f, (s_mid, zp_mid) defaults to (s_out, zp_out).
    float s_mid = 0.0f;
    int32_t zp_mid = 0;

    // Per-tensor scale for W8A16_PT_1X mode.
    float w8a16_pt_scale = 1.0f;

    // Optional row-major [N, K] int8 weights and optional [N] int32 bias.
    const int8_t* weights_nk = nullptr;
    const int32_t* bias_n = nullptr;

    // Optional pre-packed ext[2] parameter buffer (used when weights_nk == nullptr).
    const uint8_t* packed_params = nullptr;
    size_t packed_params_bytes = 0;

    // Optional external RV64 DIVE interpreter image (ext[1]).
    // When provided (non-null, non-empty), emit_buenos_kernel() also serializes a
    // complete 64-byte-aligned DGC0 container in BuenosEmitResult::dgc0.
    const uint8_t* dive_interp = nullptr;
    size_t dive_interp_size = 0;
};

struct BuenosEmitResult {
    BuenosKernelMode resolved_mode = BuenosKernelMode::W8A8_INT8;
    std::array<std::vector<uint8_t>, 7> chunks;
    std::vector<uint8_t> ext0_dive;
    std::vector<uint8_t> ext2_params;
    std::vector<uint8_t> dgc0;
    uint32_t exec_field_14 = 0;
    uint32_t dive_scratch_size = 0;
    uint32_t ext2_bytes = 0;
    float mode_a_max_lsb_err = 0.0f;
    std::string kernel_name;
    std::string cache_key;
    // True when the exact emitted VLIW stream matches live silicon-proven rango oracles.
    // False for offline-synthesized extensions awaiting live silicon verification
    // (currently W8A16_PT_1X with K=576 or N=576, and W8A16_PT_64X).
    bool silicon_verified = true;
};

struct BuenosDgc0Summary {
    std::vector<uint64_t> input_sizes;
    std::vector<uint64_t> output_sizes;
    uint64_t ext0_offset = 0;
    uint64_t ext0_size = 0;
    uint64_t ext1_offset = 0;
    uint64_t ext1_size = 0;
    uint64_t ext2_offset = 0;
    uint64_t ext2_size = 0;
    uint64_t dive_scratch_size = 0;
    uint64_t compiler_cl = 0;
    uint32_t exec_field_14 = 0;
    uint8_t cluster_id = 0;
    bool has_params = false;
};

// Hardware shape predicates for Tensor G5 (buenos_a0).
bool is_buenos_lt_1x_dim(uint32_t d);
bool is_buenos_lt_64x_shape(uint32_t m, uint32_t k, uint32_t n);
bool is_buenos_w8a16_1x_silicon_verified_shape(uint32_t m, uint32_t k, uint32_t n);
bool is_buenos_w8a16_1x_shape(uint32_t m, uint32_t k, uint32_t n);
bool is_buenos_w8a16_64x_shape(uint32_t m, uint32_t k, uint32_t n);
bool is_buenos_w8a16_shape(uint32_t m, uint32_t k, uint32_t n);
bool is_buenos_w8a8_shape(uint32_t m, uint32_t k, uint32_t n, std::string* reason = nullptr);

// Selective mixed-precision W8A16(FP16) vs W8A8 auto policy for Tensor G5 (buenos_a0).
// W8A8 remains the fast-decode default (27.96 GB/s); 7-chunk PT W8A16(FP16) with FP32
// accumulator is selected for high-sensitivity projections (lm_head, down_proj, Layer 0/1,
// Q6_K tensors, or activation outlier crest >= crest_threshold).
enum class BuenosProjectionRole {
    GENERIC = 0,
    Q_PROJ,
    K_PROJ,
    V_PROJ,
    O_PROJ,
    GATE_PROJ,
    UP_PROJ,
    DOWN_PROJ,
    LM_HEAD,
};

struct BuenosW8A16PolicyInput {
    uint32_t m = 1u;
    uint32_t k = 0u;
    uint32_t n = 0u;
    BuenosProjectionRole role = BuenosProjectionRole::GENERIC;
    int32_t layer_index = -1;       // < 0 when unknown or non-layer (e.g. lm_head)
    int32_t total_layers = 0;       // > 0 enables last-2-layers sensitivity tagging
    QuantFormat weight_quant = QuantFormat::Q8_0;
    EpilogueActivation activation = EpilogueActivation::NONE;
    float act_crest_factor = 0.0f;  // max(|x|) / rms(x); >= crest_threshold triggers W8A16
    float crest_threshold = 16.0f;  // default 16.0x outlier guard
    float act_absmax = 0.0f;        // > 32768.0f sets needs_fp16_prescale guard
};

struct BuenosW8A16PolicyDecision {
    bool prefer_w8a16 = false;
    bool needs_fp16_prescale = false;
    BuenosKernelMode recommended_mode = BuenosKernelMode::W8A8_INT8;
    const char* reason = "default_w8a8";
};

bool buenos_should_prefer_w8a16(const BuenosW8A16PolicyInput& input,
                                BuenosW8A16PolicyDecision* out_decision = nullptr) noexcept;

// Plans a validated ChipTarget::BUENOS_A0 TileKernel for any supported buenos_a0
// W8A8 (general 81-shape multi-tile, lt_1x*, lt_64x*) or W8A16/BF16 matmul shape
// with optional fused epilogue activation (NONE, RELU, RELU6, SILU, GELU).
bool plan_buenos_matmul_kernel(int64_t m,
                               int64_t k,
                               int64_t n,
                               QuantFormat fmt,
                               DType act_dt,
                               DType out_dt,
                               EpilogueActivation activation,
                               TileKernel* out_kernel,
                               std::string* reason = nullptr);

// Parameter packing helpers matching buenos_a0 ext[2] hardware layouts.
size_t buenos_pack_ext2_size(uint32_t n, uint32_t k, bool include_bias);
std::vector<uint8_t> buenos_pack_ext2(uint32_t n,
                                      uint32_t k,
                                      const int8_t* weights_nk,
                                      const int32_t* bias_n,
                                      bool include_bias);
std::vector<uint8_t> buenos_pack_ext2_lt_64x(uint32_t n,
                                             uint32_t k,
                                             const int8_t* weights_nk,
                                             const int32_t* bias_n);
std::vector<uint8_t> buenos_pack_w8a16_ext2(uint32_t n,
                                            uint32_t k,
                                            const int8_t* weights_nk);

// Mode-A zero-overhead (+0 B) in-place chunk[5] 0x640 degree-4 Remez SiLU/GELU fusion
// and exact IEEE-754 RNE Horner hardware simulator.
bool buenos_apply_mode_a_fusion_c5(std::vector<uint8_t>* chunk5,
                                   EpilogueActivation activation,
                                   float s_mid,
                                   int32_t zp_mid,
                                   float s_out,
                                   int32_t zp_out,
                                   float* out_max_lsb_err = nullptr,
                                   std::string* err = nullptr);

bool buenos_simulate_c5_mode_a_fused(const std::vector<uint8_t>& chunk5,
                                     const uint8_t* u_mid,
                                     size_t count,
                                     int8_t* out_q8,
                                     std::string* err = nullptr);

// Offline zero-dependency DGC0 FlatBuffer container inspector.
bool inspect_buenos_dgc0(const uint8_t* data,
                         size_t size,
                         BuenosDgc0Summary* out,
                         std::string* err = nullptr);

// Lowers a validated ChipTarget::BUENOS_A0 TileKernel directly into Tensor G5
// (buenos_a0) 7-chunk VLIW descriptors, DIVE VM IR (ext[0]), packed parameters (ext[2]),
// and (when opts.dive_interp is supplied) a complete 64-byte-aligned DGC0 executable.
// Non-throwing variant safe for -fno-exceptions builds (e.g. libopenbuenos).
bool try_emit_buenos_kernel(const TileKernel& kernel,
                            const BuenosEmitOptions& opts,
                            BuenosEmitResult* out_result,
                            std::string* err = nullptr);

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
// Convenience wrapper that throws std::runtime_error on invalid kernel,
// unsupported target/shape, or invalid options.
BuenosEmitResult emit_buenos_kernel(const TileKernel& kernel,
                                    const BuenosEmitOptions& opts = {});
#endif

// ---------------------------------------------------------------------------
// 11-Chunk Zero-DRAM In-SRAM Chained SwiGLU Emitter
//   gate -> 0x640 SiLU * up [0x100 MUL] -> down (single VII call, sgs = 1)
//
// Silicon-proven shapes (0 / 138,880 byte diffs on rango):
//   M=1 : (576, 512, 576), (576, 1536, 576), (1536, 512, 1536), (1536, 1024, 1536)
//   M=64: (576, 512, 576), (1536, 512, 1536)
// ---------------------------------------------------------------------------

struct BuenosSwigluDesc {
    uint32_t m_dim = 0u;
    uint32_t k_dim = 0u;
    uint32_t h_dim = 0u;
    uint32_t n_dim = 0u;
    const int8_t* wg_hk = nullptr;  // [h_dim, k_dim] row-major int8
    const int32_t* bg_h = nullptr;  // optional [h_dim] int32 (nullptr -> 0)
    const int8_t* wu_hk = nullptr;  // [h_dim, k_dim] row-major int8
    const int32_t* bu_h = nullptr;  // optional [h_dim] int32 (nullptr -> 0)
    const int8_t* wd_nh = nullptr;  // [n_dim, h_dim] row-major int8
    const int32_t* bd_n = nullptr;  // optional [n_dim] int32 (nullptr -> 0)
    float s_in = 0.0f;
    float s_wg = 0.0f;
    float s_gate = 0.0f;
    float s_wu = 0.0f;
    float s_up = 0.0f;
    float s_silu = 0.0f;
    float s_prod = 0.0f;
    float s_wd = 0.0f;
    float s_out = 0.0f;
    // Optional external RV64 DIVE interpreter image (ext[1]).
    // When non-null and non-empty, try_emit_buenos_swiglu() also serializes a
    // complete 64-byte-aligned DGC0 executable into BuenosSwigluResult::dgc0.
    const uint8_t* dive_interp = nullptr;
    size_t dive_interp_size = 0u;
};

struct BuenosSwigluResult {
    std::array<std::vector<uint8_t>, 11> chunks;
    std::vector<uint8_t> ext0_dive;
    std::vector<uint8_t> ext2_params;
    std::vector<uint8_t> dgc0;
    uint32_t exec_field_14 = 0u;
    uint32_t dive_scratch_size = 0u;
    uint32_t ext2_bytes = 0u;
    std::string cache_key;
};

bool is_buenos_swiglu_shape(uint32_t m, uint32_t k, uint32_t h, uint32_t n) noexcept;
size_t buenos_pack_swiglu_ext2_size(uint32_t m, uint32_t k, uint32_t h, uint32_t n) noexcept;
std::vector<uint8_t> buenos_pack_swiglu_ext2(uint32_t m, uint32_t k, uint32_t h, uint32_t n,
                                             const int8_t* wg_hk, const int32_t* bg_h,
                                             const int8_t* wu_hk, const int32_t* bu_h,
                                             const int8_t* wd_nh, const int32_t* bd_n);
bool buenos_compute_swiglu_multipliers(const BuenosSwigluDesc& desc,
                                       float* out_m_gate = nullptr,
                                       float* out_m_up = nullptr,
                                       float* out_m_mul = nullptr,
                                       float* out_m_down = nullptr) noexcept;
bool buenos_simulate_swiglu_exact(const int8_t* x_mk,
                                  const BuenosSwigluDesc& desc,
                                  int8_t* y_mn,
                                  std::string* err = nullptr);

// Non-throwing 11-chunk chained SwiGLU emitter safe for -fno-exceptions builds.
bool try_emit_buenos_swiglu(const BuenosSwigluDesc& desc,
                            BuenosSwigluResult* out_result,
                            std::string* err = nullptr) noexcept;

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
BuenosSwigluResult emit_buenos_swiglu(const BuenosSwigluDesc& desc);
#endif

}  // namespace freetile
