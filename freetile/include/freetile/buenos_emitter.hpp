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
    W8A16_PT_1X,       // 7-chunk per-tensor W8A16 (16-bit act + HibDma 1:2 u8->16b weights, FP32 I/O & accum)
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
bool is_buenos_w8a16_shape(uint32_t m, uint32_t k, uint32_t n);
bool is_buenos_w8a8_shape(uint32_t m, uint32_t k, uint32_t n, std::string* reason = nullptr);

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
// Throws std::runtime_error on invalid kernel, unsupported target/shape, or invalid options.
BuenosEmitResult emit_buenos_kernel(const TileKernel& kernel,
                                    const BuenosEmitOptions& opts = {});

}  // namespace freetile
