// SPDX-License-Identifier: Apache-2.0
// Clean-room zero-dependency C++17 768 KiB Narrow-SRAM Double-Buffered FlashAttention /
// FlashDecoding Tile Planner, Block-Tiled DMA-BUF KV Geometry & Dispatch Break-Even Model
// for Tensor G5 (buenos_a0) and compatible mobile Edge TPU targets.
//
// Builds cleanly under -std=c++17 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace freetile {

constexpr size_t kBuenosCoreNarrowMemBytes = 786432u;  // 768 KiB narrow SRAM per core
constexpr uint32_t kBuenosAsyncQueueCredits = 16u;
constexpr uint32_t kBuenosQ4SoftwareUnpackStallRatio = 96u;  // 96 vector cycles / 1 systolic cycle
constexpr size_t kDmaBufPageAlignBytes = 4096u;
constexpr double kDefaultViiDispatchUs = 300.0;
constexpr double kDefaultNeonVectorOpUs = 20.0;
constexpr uint32_t kDefaultAttnTpuMinSeqLen = 512u;

enum class FlashAttnPhase : uint8_t {
    PREFILL = 0,
    DECODE = 1,
};

enum class FlashAttnDType : uint8_t {
    FP16 = 0,
    BF16 = 1,
    INT8 = 2,
};

enum class FlashAttnBufferPolicy : uint8_t {
    // Conservative full ping-pong double buffering across all working tiles:
    //   2 * (Q + K + V + S + O + m + l)
    // Guarantees zero DMA stalls and matches published 768 KiB ceiling tables exactly.
    PING_PONG_FULL = 0,

    // Resident Q + FP32 O/S accumulators with explicit 2xK and 2xV double-buffering:
    //   Q(act) + 2*K(act) + 2*V(act) + S(FP32) + O(FP32) + m(FP32) + l(FP32)
    KV_DOUBLE_BUFFERED_FP32_ACC = 1,

    // Single-buffered reference ceiling (diagnostic only; stalls on DMA).
    SINGLE_BUFFERED = 2,
};

enum class KvCacheQuantFormat : uint8_t {
    F16 = 0,
    Q8_0 = 1,
    Q4_0 = 2,
};

const char* flash_attn_phase_name(FlashAttnPhase phase);
const char* flash_attn_dtype_name(FlashAttnDType dtype);
const char* kv_cache_quant_format_name(KvCacheQuantFormat fmt);

size_t flash_attn_dtype_bytes(FlashAttnDType dtype);

// Aligns logical row count (B_r / GQA group) to buenos_a0 hardware systolic M:
//   1 -> 1; 2..8 -> 8; 9..16 -> 16; >16 -> next multiple of 16.
// Makes GQA 8:1 (B_r=8) a 1:1 hardware match with zero wasted systolic rows.
uint32_t buenos_align_hw_m(uint32_t logical_rows);

struct FlashAttnSramBreakdown {
    size_t q_bytes = 0u;
    size_t k_bytes = 0u;
    size_t v_bytes = 0u;
    size_t s_bytes = 0u;
    size_t o_bytes = 0u;
    size_t stats_ml_bytes = 0u;
    size_t total_bytes = 0u;
    size_t budget_bytes = kBuenosCoreNarrowMemBytes;
    size_t headroom_bytes = 0u;
    uint32_t buffer_stages = 2u;
    bool fits_sram = false;
};

struct FlashAttnTileRequest {
    FlashAttnPhase phase = FlashAttnPhase::DECODE;
    uint32_t head_dim = 128u;
    uint32_t num_q_heads = 32u;
    uint32_t num_kv_heads = 4u;
    uint32_t seq_len = 2048u;
    uint32_t kv_tile_align = 64u;
    FlashAttnDType dtype = FlashAttnDType::FP16;
    FlashAttnBufferPolicy buffer_policy = FlashAttnBufferPolicy::PING_PONG_FULL;
    size_t sram_budget_bytes = kBuenosCoreNarrowMemBytes;
    uint32_t override_br = 0u;  // 0 = auto-select
    uint32_t override_bc = 0u;  // 0 = auto-select
};

struct FlashAttnTilePlan {
    FlashAttnPhase phase = FlashAttnPhase::DECODE;
    uint32_t head_dim = 0u;
    uint32_t gqa_ratio = 1u;
    uint32_t br = 0u;
    uint32_t hw_m = 0u;
    uint32_t bc = 0u;
    uint32_t max_bc_unaligned = 0u;
    uint32_t max_bc_aligned = 0u;
    uint32_t num_kv_tiles = 0u;
    float systolic_row_efficiency = 0.0f;
    FlashAttnSramBreakdown sram;
    bool valid = false;
};

// Computes exact itemized SRAM byte footprint for (br, bc, head_dim) with overflow protection.
bool compute_flash_attn_sram_footprint(uint32_t br,
                                       uint32_t bc,
                                       uint32_t head_dim,
                                       FlashAttnDType dtype,
                                       FlashAttnBufferPolicy policy,
                                       size_t sram_budget_bytes,
                                       FlashAttnSramBreakdown* out_sram,
                                       std::string* err = nullptr);

// Computes exact maximum feasible B_c (unaligned and aligned) fitting <= sram_budget_bytes.
uint32_t flash_attn_max_bc(uint32_t br,
                           uint32_t head_dim,
                           FlashAttnDType dtype,
                           FlashAttnBufferPolicy policy,
                           size_t sram_budget_bytes,
                           uint32_t kv_tile_align = 1u);

// Plans an SRAM-feasible FlashAttention (prefill) or GQA-batched FlashDecoding (decode) tile.
bool plan_flash_attn_tile(const FlashAttnTileRequest& req,
                          FlashAttnTilePlan* out_plan,
                          std::string* err = nullptr);

// Block-tiled DMA-BUF KV cache page geometry planner (eliminates per-token SMMU remaps).
struct KvCachePageRequest {
    uint32_t num_layers = 32u;
    uint32_t num_kv_heads = 4u;
    uint32_t head_dim = 128u;
    uint32_t max_seq_len = 4096u;
    uint32_t page_tokens = 64u;  // B_c_page (e.g. 64 or 128)
    KvCacheQuantFormat format = KvCacheQuantFormat::Q8_0;
    bool target_has_native_int4_expander = false;  // false on buenos_a0 & Day-1 malibu_a0
};

struct KvCachePagePlan {
    KvCacheQuantFormat format = KvCacheQuantFormat::Q8_0;
    bool accepted = false;
    bool uses_hw_1to2_expander = false;
    uint32_t unpack_stall_ratio = 1u;
    uint32_t page_tokens = 0u;
    uint32_t pages_per_head = 0u;
    size_t payload_bytes_per_page = 0u;
    size_t scale_bytes_per_page = 0u;
    size_t dmabuf_bytes_per_kv_page_aligned = 0u;  // 4 KiB-aligned single K+V page pair
    size_t total_dmabuf_bytes = 0u;
    float bandwidth_savings_vs_f16 = 1.0f;
    std::string reason;
};

bool plan_kv_cache_dmabuf_pages(const KvCachePageRequest& req,
                                KvCachePagePlan* out_plan,
                                std::string* err = nullptr);

// Host-to-NPU VII dispatch break-even estimator (~300 us VII vs ~20 us ARM64 NEON).
struct DispatchBreakEvenRequest {
    uint32_t num_layers = 32u;
    uint32_t hybrid_dispatches_per_layer = 3u;
    uint32_t fused_dispatches_per_layer = 1u;
    uint32_t cpu_vector_ops_per_layer = 2u;
    double vii_dispatch_us = kDefaultViiDispatchUs;
    double neon_op_us = kDefaultNeonVectorOpUs;
    double target_token_budget_ms = 50.0;  // 20 tok/s real-time budget
    uint32_t seq_len = 512u;
};

struct DispatchBreakEvenResult {
    double hybrid_per_layer_non_compute_us = 0.0;
    double hybrid_total_overhead_ms = 0.0;
    double fused_per_layer_overhead_us = 0.0;
    double fused_total_overhead_ms = 0.0;
    double savings_ms = 0.0;
    double hybrid_budget_pct = 0.0;
    double fused_budget_pct = 0.0;
    bool hybrid_viable = false;
    bool fused_viable = false;
    bool should_offload_attention_to_tpu = false;
    bool should_offload_standalone_vector_op = false;
};

bool estimate_dispatch_break_even(const DispatchBreakEvenRequest& req,
                                  DispatchBreakEvenResult* out_res,
                                  std::string* err = nullptr);

}  // namespace freetile
