// SPDX-License-Identifier: Apache-2.0
// Clean-room zero-dependency C++17 768 KiB Narrow-SRAM Double-Buffered FlashAttention /
// FlashDecoding Tile Planner, Block-Tiled DMA-BUF KV Geometry & Dispatch Break-Even Model.

#include "freetile/flash_attn_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace freetile {

namespace {

bool set_err(std::string* err, const char* msg) {
    if (err != nullptr) {
        *err = msg;
    }
    return false;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t* out) {
    if (a != 0ULL && b > (std::numeric_limits<uint64_t>::max() / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t* out) {
    if (b > (std::numeric_limits<uint64_t>::max() - a)) {
        return false;
    }
    *out = a + b;
    return true;
}

bool checked_mul3(uint64_t a, uint64_t b, uint64_t c, uint64_t* out) {
    uint64_t ab = 0ULL;
    if (!checked_mul(a, b, &ab)) {
        return false;
    }
    return checked_mul(ab, c, out);
}

bool checked_align_up(uint64_t val, uint64_t align, uint64_t* out) {
    if (align == 0ULL) {
        return false;
    }
    const uint64_t rem = val % align;
    if (rem == 0ULL) {
        *out = val;
        return true;
    }
    return checked_add(val, align - rem, out);
}

bool compute_sram_terms(uint32_t br,
                        uint32_t bc,
                        uint32_t head_dim,
                        FlashAttnDType dtype,
                        FlashAttnBufferPolicy policy,
                        uint64_t* fixed_bytes,
                        uint64_t* per_bc_bytes,
                        FlashAttnSramBreakdown* out_detail) {
    const uint64_t elem = static_cast<uint64_t>(flash_attn_dtype_bytes(dtype));
    const uint64_t br64 = static_cast<uint64_t>(br);
    const uint64_t bc64 = static_cast<uint64_t>(bc);
    const uint64_t d64 = static_cast<uint64_t>(head_dim);

    uint64_t q_b = 0ULL;
    uint64_t k_b = 0ULL;
    uint64_t v_b = 0ULL;
    uint64_t s_b = 0ULL;
    uint64_t o_b = 0ULL;
    uint64_t ml_b = 0ULL;
    uint64_t per_bc = 0ULL;
    uint32_t stages = 2u;

    if (policy == FlashAttnBufferPolicy::PING_PONG_FULL) {
        stages = 2u;
        // 2 * (Q + K + V + S + O + m + l), where m, l are FP32 (4 B each).
        if (!checked_mul3(2ULL, br64 * d64, elem, &q_b) ||
            !checked_mul3(2ULL, bc64 * d64, elem, &k_b) ||
            !checked_mul3(2ULL, bc64 * d64, elem, &v_b) ||
            !checked_mul3(2ULL, br64 * bc64, elem, &s_b) ||
            !checked_mul3(2ULL, br64 * d64, elem, &o_b) ||
            !checked_mul(16ULL, br64, &ml_b)) {
            return false;
        }
        uint64_t kv_per_bc = 0ULL;
        uint64_t s_per_bc = 0ULL;
        if (!checked_mul3(4ULL, d64, elem, &kv_per_bc) ||
            !checked_mul3(2ULL, br64, elem, &s_per_bc) ||
            !checked_add(kv_per_bc, s_per_bc, &per_bc)) {
            return false;
        }
    } else if (policy == FlashAttnBufferPolicy::KV_DOUBLE_BUFFERED_FP32_ACC) {
        stages = 2u;
        // Resident Q(elem) + 2*K(elem) + 2*V(elem) + S(FP32) + O(FP32) + m(FP32) + l(FP32).
        if (!checked_mul3(1ULL, br64, d64 * elem, &q_b) ||
            !checked_mul3(2ULL, bc64 * d64, elem, &k_b) ||
            !checked_mul3(2ULL, bc64 * d64, elem, &v_b) ||
            !checked_mul3(4ULL, br64, bc64, &s_b) ||
            !checked_mul3(4ULL, br64, d64, &o_b) ||
            !checked_mul(8ULL, br64, &ml_b)) {
            return false;
        }
        uint64_t kv_per_bc = 0ULL;
        uint64_t s_per_bc = 0ULL;
        if (!checked_mul3(4ULL, d64, elem, &kv_per_bc) ||
            !checked_mul(4ULL, br64, &s_per_bc) ||
            !checked_add(kv_per_bc, s_per_bc, &per_bc)) {
            return false;
        }
    } else {
        stages = 1u;
        if (!checked_mul3(1ULL, br64 * d64, elem, &q_b) ||
            !checked_mul3(1ULL, bc64 * d64, elem, &k_b) ||
            !checked_mul3(1ULL, bc64 * d64, elem, &v_b) ||
            !checked_mul3(1ULL, br64 * bc64, elem, &s_b) ||
            !checked_mul3(1ULL, br64 * d64, elem, &o_b) ||
            !checked_mul(8ULL, br64, &ml_b)) {
            return false;
        }
        uint64_t kv_per_bc = 0ULL;
        uint64_t s_per_bc = 0ULL;
        if (!checked_mul3(2ULL, d64, elem, &kv_per_bc) ||
            !checked_mul(br64, elem, &s_per_bc) ||
            !checked_add(kv_per_bc, s_per_bc, &per_bc)) {
            return false;
        }
    }

    uint64_t fix = 0ULL;
    if (!checked_add(q_b, o_b, &fix) || !checked_add(fix, ml_b, &fix)) {
        return false;
    }

    if (fixed_bytes != nullptr) {
        *fixed_bytes = fix;
    }
    if (per_bc_bytes != nullptr) {
        *per_bc_bytes = per_bc;
    }
    if (out_detail != nullptr) {
        uint64_t total = 0ULL;
        uint64_t kv_s = 0ULL;
        if (!checked_add(k_b, v_b, &kv_s) ||
            !checked_add(kv_s, s_b, &kv_s) ||
            !checked_add(fix, kv_s, &total)) {
            return false;
        }
        if (total > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        out_detail->q_bytes = static_cast<size_t>(q_b);
        out_detail->k_bytes = static_cast<size_t>(k_b);
        out_detail->v_bytes = static_cast<size_t>(v_b);
        out_detail->s_bytes = static_cast<size_t>(s_b);
        out_detail->o_bytes = static_cast<size_t>(o_b);
        out_detail->stats_ml_bytes = static_cast<size_t>(ml_b);
        out_detail->total_bytes = static_cast<size_t>(total);
        out_detail->buffer_stages = stages;
    }
    return true;
}

}  // namespace

const char* flash_attn_phase_name(FlashAttnPhase phase) {
    switch (phase) {
        case FlashAttnPhase::PREFILL: return "prefill";
        case FlashAttnPhase::DECODE:  return "decode";
    }
    return "unknown";
}

const char* flash_attn_dtype_name(FlashAttnDType dtype) {
    switch (dtype) {
        case FlashAttnDType::FP16: return "fp16";
        case FlashAttnDType::BF16: return "bf16";
        case FlashAttnDType::INT8: return "int8";
    }
    return "unknown";
}

const char* kv_cache_quant_format_name(KvCacheQuantFormat fmt) {
    switch (fmt) {
        case KvCacheQuantFormat::F16:  return "f16";
        case KvCacheQuantFormat::Q8_0: return "q8_0";
        case KvCacheQuantFormat::Q4_0: return "q4_0";
    }
    return "unknown";
}

size_t flash_attn_dtype_bytes(FlashAttnDType dtype) {
    switch (dtype) {
        case FlashAttnDType::FP16: return 2u;
        case FlashAttnDType::BF16: return 2u;
        case FlashAttnDType::INT8: return 1u;
    }
    return 2u;
}

uint32_t buenos_align_hw_m(uint32_t logical_rows) {
    if (logical_rows <= 1u) {
        return 1u;
    }
    if (logical_rows <= 8u) {
        return 8u;
    }
    if (logical_rows <= 16u) {
        return 16u;
    }
    const uint64_t aligned = ((static_cast<uint64_t>(logical_rows) + 15ULL) / 16ULL) * 16ULL;
    if (aligned > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return std::numeric_limits<uint32_t>::max() & ~15u;
    }
    return static_cast<uint32_t>(aligned);
}

bool compute_flash_attn_sram_footprint(uint32_t br,
                                       uint32_t bc,
                                       uint32_t head_dim,
                                       FlashAttnDType dtype,
                                       FlashAttnBufferPolicy policy,
                                       size_t sram_budget_bytes,
                                       FlashAttnSramBreakdown* out_sram,
                                       std::string* err) {
    if (out_sram == nullptr) {
        return set_err(err, "out_sram must not be null");
    }
    if (br == 0u || bc == 0u || head_dim == 0u || sram_budget_bytes == 0u) {
        return set_err(err, "br, bc, head_dim, and sram_budget_bytes must be > 0");
    }
    FlashAttnSramBreakdown detail;
    if (!compute_sram_terms(br, bc, head_dim, dtype, policy, nullptr, nullptr, &detail)) {
        return set_err(err, "SRAM footprint calculation overflowed");
    }
    detail.budget_bytes = sram_budget_bytes;
    detail.fits_sram = (detail.total_bytes <= sram_budget_bytes);
    detail.headroom_bytes = detail.fits_sram ? (sram_budget_bytes - detail.total_bytes) : 0u;
    *out_sram = detail;
    return true;
}

uint32_t flash_attn_max_bc(uint32_t br,
                           uint32_t head_dim,
                           FlashAttnDType dtype,
                           FlashAttnBufferPolicy policy,
                           size_t sram_budget_bytes,
                           uint32_t kv_tile_align) {
    if (br == 0u || head_dim == 0u || sram_budget_bytes == 0u || kv_tile_align == 0u) {
        return 0u;
    }
    uint64_t fixed = 0ULL;
    uint64_t per_bc = 0ULL;
    if (!compute_sram_terms(br, 1u, head_dim, dtype, policy, &fixed, &per_bc, nullptr)) {
        return 0u;
    }
    const uint64_t budget = static_cast<uint64_t>(sram_budget_bytes);
    if (per_bc == 0ULL || fixed >= budget) {
        return 0u;
    }
    const uint64_t raw_max = (budget - fixed) / per_bc;
    const uint64_t aligned = (raw_max / static_cast<uint64_t>(kv_tile_align)) *
                             static_cast<uint64_t>(kv_tile_align);
    if (aligned > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(aligned);
}

bool plan_flash_attn_tile(const FlashAttnTileRequest& req,
                          FlashAttnTilePlan* out_plan,
                          std::string* err) {
    if (out_plan == nullptr) {
        return set_err(err, "out_plan must not be null");
    }
    *out_plan = FlashAttnTilePlan{};

    if (req.head_dim == 0u || (req.head_dim % 16u) != 0u) {
        return set_err(err, "head_dim must be a positive multiple of 16");
    }
    if (req.num_q_heads == 0u || req.num_kv_heads == 0u ||
        (req.num_q_heads % req.num_kv_heads) != 0u) {
        return set_err(err, "num_q_heads must be a positive multiple of num_kv_heads");
    }
    if (req.kv_tile_align == 0u) {
        return set_err(err, "kv_tile_align must be > 0");
    }
    if (req.sram_budget_bytes == 0u) {
        return set_err(err, "sram_budget_bytes must be > 0");
    }

    const uint32_t gqa_ratio = req.num_q_heads / req.num_kv_heads;
    uint32_t br = req.override_br;
    if (br == 0u) {
        if (req.phase == FlashAttnPhase::PREFILL) {
            if (req.head_dim <= 64u) {
                br = 128u;
            } else if (req.head_dim <= 128u) {
                br = 64u;
            } else {
                br = 32u;
            }
        } else {
            // Decode: batch all query heads sharing one KV head into a single GQA row block.
            br = gqa_ratio;
        }
    }

    const uint32_t max_bc_unaligned = flash_attn_max_bc(
        br, req.head_dim, req.dtype, req.buffer_policy, req.sram_budget_bytes, 1u);
    const uint32_t max_bc_aligned = flash_attn_max_bc(
        br, req.head_dim, req.dtype, req.buffer_policy, req.sram_budget_bytes, req.kv_tile_align);

    if (max_bc_aligned == 0u) {
        return set_err(err, "No aligned B_c fits within SRAM budget for requested (B_r, head_dim)");
    }

    uint32_t bc = req.override_bc;
    if (bc == 0u) {
        if (req.phase == FlashAttnPhase::PREFILL) {
            // Prefill prefers B_c = 512 when feasible to balance Q/KV reuse & DMA burst size.
            bc = (max_bc_aligned >= 512u) ? 512u : max_bc_aligned;
        } else {
            // Decode: GQA 8:1 d=128 prefers power-of-two B_c = 512 (leaving 237 KiB headroom
            // for online softmax/QKV scratch); GQA 8:1 d=64 uses 1408, GQA 4:1 d=128 uses 704.
            if (req.head_dim == 128u && br == 8u && max_bc_aligned >= 512u) {
                bc = 512u;
            } else {
                bc = max_bc_aligned;
            }
        }
    }

    FlashAttnSramBreakdown sram;
    if (!compute_flash_attn_sram_footprint(br, bc, req.head_dim, req.dtype, req.buffer_policy,
                                           req.sram_budget_bytes, &sram, err)) {
        return false;
    }
    if (!sram.fits_sram) {
        return set_err(err, "Requested FlashAttention tile exceeds narrow SRAM budget");
    }

    const uint32_t hw_m = buenos_align_hw_m(br);
    const uint64_t num_tiles64 =
        (req.seq_len == 0u)
            ? 0ULL
            : ((static_cast<uint64_t>(req.seq_len) + static_cast<uint64_t>(bc) - 1ULL) /
               static_cast<uint64_t>(bc));
    const uint32_t num_tiles = static_cast<uint32_t>(num_tiles64);

    out_plan->phase = req.phase;
    out_plan->head_dim = req.head_dim;
    out_plan->gqa_ratio = gqa_ratio;
    out_plan->br = br;
    out_plan->hw_m = hw_m;
    out_plan->bc = bc;
    out_plan->max_bc_unaligned = max_bc_unaligned;
    out_plan->max_bc_aligned = max_bc_aligned;
    out_plan->num_kv_tiles = num_tiles;
    out_plan->systolic_row_efficiency =
        static_cast<float>(br) / static_cast<float>(hw_m);
    out_plan->sram = sram;
    out_plan->valid = true;
    return true;
}

bool plan_kv_cache_dmabuf_pages(const KvCachePageRequest& req,
                                KvCachePagePlan* out_plan,
                                std::string* err) {
    if (out_plan == nullptr) {
        return set_err(err, "out_plan must not be null");
    }
    *out_plan = KvCachePagePlan{};
    out_plan->format = req.format;
    out_plan->page_tokens = req.page_tokens;

    if (req.num_layers == 0u || req.num_kv_heads == 0u || req.head_dim == 0u ||
        req.page_tokens == 0u || req.max_seq_len == 0u) {
        return set_err(err, "KV cache dimensions, page_tokens, and max_seq_len must be > 0");
    }
    if ((req.head_dim % 32u) != 0u) {
        return set_err(err, "head_dim must be a multiple of 32 for Q8_0/Q4_0 block alignment");
    }

    // Reject software-unpacked Q4_0 KV cache on NPUs lacking native 1:4 hardware expander
    // due to the proven 96:1 VLIW vector-unpack compute stall.
    if (req.format == KvCacheQuantFormat::Q4_0 && !req.target_has_native_int4_expander) {
        out_plan->accepted = false;
        out_plan->uses_hw_1to2_expander = false;
        out_plan->unpack_stall_ratio = kBuenosQ4SoftwareUnpackStallRatio;
        out_plan->bandwidth_savings_vs_f16 = 0.0f;
        out_plan->reason =
            "Rejected Q4_0 KV cache on NPU without native 1:4 expander (96:1 VLIW unpack stall); "
            "use Q8_0 for hardware 1:2 u8->16b DMA expansion";
        return set_err(err, out_plan->reason.c_str());
    }

    uint64_t elems_per_page = 0ULL;
    if (!checked_mul(static_cast<uint64_t>(req.page_tokens),
                     static_cast<uint64_t>(req.head_dim), &elems_per_page)) {
        return set_err(err, "KV page element count overflow");
    }

    uint64_t payload_single = 0ULL;
    uint64_t scales_single = 0ULL;
    if (req.format == KvCacheQuantFormat::F16) {
        if (!checked_mul(elems_per_page, 2ULL, &payload_single)) {
            return set_err(err, "F16 KV page size overflow");
        }
        scales_single = 0ULL;
        out_plan->uses_hw_1to2_expander = false;
        out_plan->unpack_stall_ratio = 1u;
        out_plan->bandwidth_savings_vs_f16 = 1.0f;
        out_plan->reason = "F16 block-tiled DMA-BUF KV pages (baseline 1.0x bandwidth)";
    } else if (req.format == KvCacheQuantFormat::Q8_0) {
        // Separated Q8_0 micro-block page: 1 byte/element packed u8 payload (hardware 1:2
        // DMA expander) + 1 FP16 scale (2 bytes) per 32 elements.
        payload_single = elems_per_page;
        scales_single = (elems_per_page / 32ULL) * 2ULL;
        out_plan->uses_hw_1to2_expander = true;
        out_plan->unpack_stall_ratio = 1u;
        const double f16_bytes = static_cast<double>(elems_per_page * 2ULL);
        const double q8_bytes = static_cast<double>(payload_single + scales_single);
        out_plan->bandwidth_savings_vs_f16 = static_cast<float>(f16_bytes / q8_bytes);
        out_plan->reason =
            "Q8_0 block-tiled DMA-BUF KV pages (optimal: hardware 1:2 DMA expander, 0 SMMU remap)";
    } else {
        // Q4_0 when target_has_native_int4_expander == true (future silicon).
        payload_single = elems_per_page / 2ULL;
        scales_single = (elems_per_page / 32ULL) * 2ULL;
        out_plan->uses_hw_1to2_expander = false;
        out_plan->unpack_stall_ratio = 1u;
        const double f16_bytes = static_cast<double>(elems_per_page * 2ULL);
        const double q4_bytes = static_cast<double>(payload_single + scales_single);
        out_plan->bandwidth_savings_vs_f16 = static_cast<float>(f16_bytes / q4_bytes);
        out_plan->reason = "Q4_0 block-tiled DMA-BUF KV pages (native 1:4 hardware expander enabled)";
    }

    uint64_t single_total = 0ULL;
    uint64_t kv_pair_raw = 0ULL;
    uint64_t kv_pair_aligned = 0ULL;
    if (!checked_add(payload_single, scales_single, &single_total) ||
        !checked_mul(single_total, 2ULL, &kv_pair_raw) ||
        !checked_align_up(kv_pair_raw, static_cast<uint64_t>(kDmaBufPageAlignBytes),
                          &kv_pair_aligned)) {
        return set_err(err, "KV page byte alignment overflow");
    }

    const uint64_t pages_per_head =
        (static_cast<uint64_t>(req.max_seq_len) + static_cast<uint64_t>(req.page_tokens) - 1ULL) /
        static_cast<uint64_t>(req.page_tokens);
    uint64_t total_pages = 0ULL;
    uint64_t total_bytes = 0ULL;
    if (!checked_mul3(static_cast<uint64_t>(req.num_layers),
                      static_cast<uint64_t>(req.num_kv_heads),
                      pages_per_head,
                      &total_pages) ||
        !checked_mul(total_pages, kv_pair_aligned, &total_bytes)) {
        return set_err(err, "Total KV DMA-BUF byte size overflow");
    }
    if (total_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        pages_per_head > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return set_err(err, "Total KV DMA-BUF size exceeds size_t");
    }

    out_plan->accepted = true;
    out_plan->pages_per_head = static_cast<uint32_t>(pages_per_head);
    out_plan->payload_bytes_per_page = static_cast<size_t>(payload_single);
    out_plan->scale_bytes_per_page = static_cast<size_t>(scales_single);
    out_plan->dmabuf_bytes_per_kv_page_aligned = static_cast<size_t>(kv_pair_aligned);
    out_plan->total_dmabuf_bytes = static_cast<size_t>(total_bytes);
    return true;
}

bool estimate_dispatch_break_even(const DispatchBreakEvenRequest& req,
                                  DispatchBreakEvenResult* out_res,
                                  std::string* err) {
    if (out_res == nullptr) {
        return set_err(err, "out_res must not be null");
    }
    if (req.num_layers == 0u ||
        !std::isfinite(req.vii_dispatch_us) || req.vii_dispatch_us < 0.0 ||
        !std::isfinite(req.neon_op_us) || req.neon_op_us < 0.0 ||
        !std::isfinite(req.target_token_budget_ms) || req.target_token_budget_ms <= 0.0) {
        return set_err(err, "Invalid DispatchBreakEvenRequest parameters");
    }

    const double hybrid_layer_us =
        static_cast<double>(req.hybrid_dispatches_per_layer) * req.vii_dispatch_us +
        static_cast<double>(req.cpu_vector_ops_per_layer) * req.neon_op_us;
    const double fused_layer_us =
        static_cast<double>(req.fused_dispatches_per_layer) * req.vii_dispatch_us;

    const double hybrid_total_ms =
        (hybrid_layer_us * static_cast<double>(req.num_layers)) / 1000.0;
    const double fused_total_ms =
        (fused_layer_us * static_cast<double>(req.num_layers)) / 1000.0;

    out_res->hybrid_per_layer_non_compute_us = hybrid_layer_us;
    out_res->hybrid_total_overhead_ms = hybrid_total_ms;
    out_res->fused_per_layer_overhead_us = fused_layer_us;
    out_res->fused_total_overhead_ms = fused_total_ms;
    out_res->savings_ms = hybrid_total_ms - fused_total_ms;
    out_res->hybrid_budget_pct = (hybrid_total_ms / req.target_token_budget_ms) * 100.0;
    out_res->fused_budget_pct = (fused_total_ms / req.target_token_budget_ms) * 100.0;

    // Exceeding 50% of total token latency budget purely in non-compute dispatch overhead
    // leaves insufficient budget for DRAM weight streaming and compute.
    out_res->hybrid_viable = (out_res->hybrid_budget_pct <= 50.0);
    out_res->fused_viable = (out_res->fused_budget_pct <= 50.0);
    out_res->should_offload_attention_to_tpu = (req.seq_len >= kDefaultAttnTpuMinSeqLen);
    out_res->should_offload_standalone_vector_op = (req.vii_dispatch_us < req.neon_op_us);
    return true;
}

}  // namespace freetile
