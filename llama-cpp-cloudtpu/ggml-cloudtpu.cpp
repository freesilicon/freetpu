#include "ggml-cloudtpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "pjrt_runtime.hpp"
#include "stablehlo_emitter.hpp"
#include "freetile/tile_ir.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t CLOUDTPU_BUFFER_ALIGNMENT = 64;

enum class CloudTpuQ4Mode {
    I4 = 0,       // Direct PJRT_Buffer_Type_S4 (or U4 for Q4_K) -> stablehlo.convert -> single dot
    BITCAST = 1,  // Packed U8 [..., 32, N/2] -> stablehlo.bitcast_convert -> i4 -> single dot
    SPLIT = 2,    // M4C split-X U8 [..., 16, N] two-dot
    CONCAT = 3,   // M4C U8 [..., 16, N] single concat dot
};

struct cloudtpu_env_config {
    bool mock_mode = false;
    int mock_devices = 1;
    int spmd_chips = 1;       // 1 = default per-chip devices, 0 = opt-in auto (all rt.device_count()), >1 = explicit SPMD chip count
    int spmd_min_rows = 256;  // Minimum weight rows/cols to shard across SPMD chips
    int max_fuse_ops = 4096;  // Max ops per fused StableHLO subgraph
    int verbose = 0;
    bool fuse_graphs = true;
    bool fuse_attn = true;          // Fuse FLASH_ATTN_EXT with following O_proj/FFN ops
    bool donate_kv = true;          // Donate KV cache HBM buffers in-place on SET_ROWS
    bool bf16_matmul = true;        // Cast split-quantized weights & activations to BF16 for MXU (dense F16/F32 stay F32)
    bool bf16_flash_attn = false;   // Opt-in BF16 Q/K/probs/V matmuls in FLASH_ATTN_EXT (default false: F32 parity)
    CloudTpuQ4Mode q4_mode = CloudTpuQ4Mode::I4;  // Single-read S4/U4 default (overridden by CLOUDTPU_Q4_MODE / CLOUDTPU_Q4_SPLIT_DOT)
    bool q4_split_dot = true;       // Q4_0/Q4_K SPLIT vs CONCAT selection when q4_mode is SPLIT/CONCAT
    bool q4k_u4 = false;            // Q4_K I4/BITCAST: false = signed S4 (q-8, unshifted m_eff), true = U4 (0..15)
    bool q4k_bf16_scales = false;   // Opt-in hardware BF16 Q4_K/Q6_K scales (default false: full F32 scale precision for GET_ROWS parity)
    int32_t mm_pad_m = 8;           // Pad small BF16 Q4_*_I4/Q6_K_SPLIT M < mm_pad_m up to mm_pad_m for MXU (<=1 disables)
    bool freetile_mosaic = false;   // CLOUDTPU_FREETILE_MOSAIC=1: emit FreeTile -> Mosaic tpu_custom_call
    int64_t freetile_mosaic_version = -1;  // CLOUDTPU_FREETILE_MOSAIC_VERSION: optional dialect version override
    bool async_collectives = false; // Opt-in LIBTPU_INIT_ARGS TPU async collective overlap flags before libtpu init
    uint32_t cache_cap = 4096;
    uint64_t min_elements = 0;
    std::vector<int64_t> batch_buckets = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};  // Padded n_tokens buckets
    std::string libtpu_path;
};

static cloudtpu_env_config load_env_config() {
    cloudtpu_env_config cfg;
#if defined(__linux__)
    cfg.mock_mode = false;
#else
    cfg.mock_mode = true;
#endif
    if (const char* v = std::getenv("CLOUDTPU_MOCK")) {
        cfg.mock_mode = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_MOCK_DEVICES")) {
        const int parsed = std::atoi(v);
        if (parsed >= 1 && parsed <= 16) {
            cfg.mock_devices = parsed;
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_SPMD_CHIPS")) {
        if (std::strcmp(v, "auto") == 0 || std::strcmp(v, "AUTO") == 0) {
            cfg.spmd_chips = 0;
        } else {
            const int parsed = std::atoi(v);
            if (parsed >= 1) {
                cfg.spmd_chips = parsed;
            }
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_SPMD_MIN_ROWS")) {
        const int parsed = std::atoi(v);
        if (parsed >= 1) {
            cfg.spmd_min_rows = parsed;
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_MAX_FUSE_OPS")) {
        const int parsed = std::atoi(v);
        if (parsed >= 1) {
            cfg.max_fuse_ops = std::min(parsed, 16384);
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_VERBOSE")) {
        cfg.verbose = std::atoi(v);
    }
    if (const char* v = std::getenv("CLOUDTPU_FUSE_GRAPHS")) {
        cfg.fuse_graphs = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_FUSE_ATTN")) {
        cfg.fuse_attn = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_DONATE_KV")) {
        cfg.donate_kv = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_MATMUL_BF16")) {
        cfg.bf16_matmul = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_FLASH_ATTN_BF16")) {
        cfg.bf16_flash_attn = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_Q4_SPLIT_DOT")) {
        cfg.q4_split_dot = (std::atoi(v) != 0);
        cfg.q4_mode = cfg.q4_split_dot ? CloudTpuQ4Mode::SPLIT : CloudTpuQ4Mode::CONCAT;
    }
    if (const char* v = std::getenv("CLOUDTPU_Q4_MODE")) {
        if (std::strcmp(v, "i4") == 0 || std::strcmp(v, "s4") == 0) {
            cfg.q4_mode = CloudTpuQ4Mode::I4;
            cfg.q4k_u4 = false;
        } else if (std::strcmp(v, "u4") == 0) {
            cfg.q4_mode = CloudTpuQ4Mode::I4;
            cfg.q4k_u4 = true;
        } else if (std::strcmp(v, "bitcast") == 0) {
            cfg.q4_mode = CloudTpuQ4Mode::BITCAST;
        } else if (std::strcmp(v, "split") == 0) {
            cfg.q4_mode = CloudTpuQ4Mode::SPLIT;
            cfg.q4_split_dot = true;
        } else if (std::strcmp(v, "concat") == 0) {
            cfg.q4_mode = CloudTpuQ4Mode::CONCAT;
            cfg.q4_split_dot = false;
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_Q4K_U4")) {
        cfg.q4k_u4 = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_Q4K_BF16_SCALES")) {
        cfg.q4k_bf16_scales = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_MM_PAD_M")) {
        const int parsed = std::atoi(v);
        cfg.mm_pad_m = (parsed <= 1) ? 1 : parsed;
    }
    if (const char* v = std::getenv("CLOUDTPU_FREETILE_MOSAIC")) {
        cfg.freetile_mosaic = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_FREETILE_MOSAIC_VERSION")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0) {
            cfg.freetile_mosaic_version = static_cast<int64_t>(parsed);
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_ASYNC_COLLECTIVES")) {
        cfg.async_collectives = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_CACHE_CAP")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0 && parsed <= 1048576LL) {
            cfg.cache_cap = static_cast<uint32_t>(parsed);
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_MIN_ELEMENTS")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0) {
            cfg.min_elements = static_cast<uint64_t>(parsed);
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_BATCH_BUCKETS")) {
        cfg.batch_buckets = cloudtpu::parse_batch_buckets(v);
    }
    if (const char* v = std::getenv("LIBTPU_PATH")) {
        cfg.libtpu_path = v;
    } else if (const char* v2 = std::getenv("TPU_LIBRARY_PATH")) {
        cfg.libtpu_path = v2;
    }
    return cfg;
}

static int resolve_spmd_chips(const cloudtpu_env_config& env, const cloudtpu::PjrtRuntime& rt) {
    const int avail = rt.device_count();
    if (avail <= 0) {
        throw std::runtime_error("no PJRT devices available");
    }
    if (env.spmd_chips <= 0) {
        // Explicit CLOUDTPU_SPMD_CHIPS=auto uses all available chips (up to 16).
        return std::min(avail, 16);
    }
    if (env.spmd_chips > 16) {
        throw std::runtime_error(
            "CLOUDTPU_SPMD_CHIPS=" + std::to_string(env.spmd_chips) +
            " exceeds supported SPMD maximum (16); available PJRT devices=" +
            std::to_string(avail));
    }
    if (env.spmd_chips > avail) {
        throw std::runtime_error(
            "CLOUDTPU_SPMD_CHIPS=" + std::to_string(env.spmd_chips) +
            " exceeds available PJRT devices (" + std::to_string(avail) + ")");
    }
    return env.spmd_chips;
}

struct cloudtpu_hot_env_cache {
    std::atomic<bool> initialized{false};
    std::atomic<bool> mock_mode{false};
    std::atomic<int> verbose{0};
    std::atomic<bool> fuse_graphs{true};
    std::atomic<uint64_t> min_elements{0};

    void store(const cloudtpu_env_config& cfg) {
        mock_mode.store(cfg.mock_mode, std::memory_order_relaxed);
        verbose.store(cfg.verbose, std::memory_order_relaxed);
        fuse_graphs.store(cfg.fuse_graphs, std::memory_order_relaxed);
        min_elements.store(cfg.min_elements, std::memory_order_relaxed);
        initialized.store(true, std::memory_order_release);
    }

    void ensure_initialized() {
        if (!initialized.load(std::memory_order_acquire)) {
            store(load_env_config());
        }
    }
};

static cloudtpu_hot_env_cache g_hot_env;

static cloudtpu::PjrtRuntime& ensure_runtime_initialized() {
    g_hot_env.ensure_initialized();
    const cloudtpu_env_config env = load_env_config();
    cloudtpu::RuntimeConfig rt_cfg;
    rt_cfg.mock_mode = env.mock_mode;
    rt_cfg.mock_devices = env.mock_devices;
    rt_cfg.verbose = env.verbose;
    rt_cfg.exec_cache_cap = env.cache_cap;
    rt_cfg.async_collectives = env.async_collectives;
    rt_cfg.libtpu_path = env.libtpu_path;
    return cloudtpu::shared_pjrt_runtime(&rt_cfg);
}

static cloudtpu::PjrtDType ggml_type_to_pjrt(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return cloudtpu::PjrtDType::F32;
        case GGML_TYPE_F16: return cloudtpu::PjrtDType::F16;
        case GGML_TYPE_BF16: return cloudtpu::PjrtDType::BF16;
        case GGML_TYPE_I8: return cloudtpu::PjrtDType::S8;
        case GGML_TYPE_I16: return cloudtpu::PjrtDType::S16;
        case GGML_TYPE_I32: return cloudtpu::PjrtDType::S32;
        case GGML_TYPE_I64: return cloudtpu::PjrtDType::S64;
        default: return cloudtpu::PjrtDType::INVALID;
    }
}

static const struct ggml_tensor* unwrap_view_src(const struct ggml_tensor* t) {
    while (t && t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

static bool is_dense_supported_type(enum ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_I32;
}

static bool is_quant_split_type(enum ggml_type t) {
    return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K;
}

static int64_t quant_split_k_alignment(enum ggml_type t) {
    if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K) {
        return QK_K;
    }
    if (t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0) {
        return QK8_0;
    }
    return 1;
}

struct ViewSlicePlan {
    std::vector<int64_t> reshape_dims;
    std::vector<int64_t> starts;
    std::vector<int64_t> limits;
};

static const struct ggml_tensor* view_parent(const struct ggml_tensor* t) {
    if (!t) return nullptr;
    if ((t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
         t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) &&
        t->src[0] != nullptr) {
        return t->src[0];
    }
    return t->view_src;
}

static bool try_build_view_slice(const struct ggml_tensor* t,
                                 ViewSlicePlan* plan,
                                 const struct ggml_tensor** out_base = nullptr) {
    const struct ggml_tensor* base = view_parent(t);
    if (!t || !base) {
        return false;
    }
    // Nested views (e.g. packed QKV -> strided K/V view -> 2D KV update view) can have a
    // strided immediate parent; fall back to the contiguous root tensor t->view_src where
    // GGML already accumulated the byte offset t->view_offs.
    if (!ggml_is_contiguous(base) && t->view_src != nullptr && t->view_src != base &&
        ggml_is_contiguous(t->view_src)) {
        base = t->view_src;
    }
    if (t->type != base->type || !is_dense_supported_type(t->type)) {
        return false;
    }
    if (!ggml_is_contiguous(base)) {
        return false;
    }
    const size_t el = ggml_type_size(t->type);
    const size_t rel_offs = (t->view_src == base) ? t->view_offs
                            : (t->view_offs >= base->view_offs ? t->view_offs - base->view_offs : SIZE_MAX);
    if (el == 0 || rel_offs == SIZE_MAX || (rel_offs % el) != 0) {
        return false;
    }
    const int64_t base_nels = ggml_nelements(base);
    const int64_t view_nels = ggml_nelements(t);
    const int64_t elem_off = static_cast<int64_t>(rel_offs / el);

    // Case 1: contiguous slice inside flat base tensor.
    if (ggml_is_contiguous(t) && elem_off >= 0 && elem_off + view_nels <= base_nels) {
        if (plan) {
            plan->reshape_dims = {base_nels};
            plan->starts = {elem_off};
            plan->limits = {elem_off + view_nels};
        }
        if (out_base) {
            *out_base = base;
        }
        return true;
    }

    // Case 2: nested strided 4D slice (covers llama_kv_cache::get_k / get_v and padded views).
    if (t->nb[0] != el) {
        return false;
    }
    if ((t->nb[1] % el) != 0 || (t->nb[2] % el) != 0 || (t->nb[3] % el) != 0) {
        return false;
    }
    const int64_t raw_s1 = static_cast<int64_t>(t->nb[1] / el);
    const int64_t raw_s2 = static_cast<int64_t>(t->nb[2] / el);
    const int64_t raw_s3 = static_cast<int64_t>(t->nb[3] / el);
    // Reject overlapping row/plane/batch strides that a non-overlapping slice cannot represent.
    if (t->ne[1] > 1 && raw_s1 < t->ne[0]) {
        return false;
    }
    const int64_t s1 = (t->ne[1] == 1) ? std::max<int64_t>(t->ne[0], raw_s1) : raw_s1;
    if (t->ne[2] > 1 && raw_s2 < s1 * t->ne[1]) {
        return false;
    }
    const int64_t s2 = (t->ne[2] == 1 && t->ne[3] == 1)
                           ? std::max<int64_t>(s1 * t->ne[1], raw_s2)
                           : raw_s2;
    if (t->ne[3] > 1 && raw_s3 < s2 * t->ne[2]) {
        return false;
    }
    const int64_t s3 = (t->ne[3] == 1)
                           ? std::max<int64_t>(s2 * t->ne[2], raw_s3)
                           : raw_s3;
    if (s1 <= 0 || s2 <= 0 || s3 <= 0 || (s2 % s1) != 0 || (s3 % s2) != 0) {
        return false;
    }
    const int64_t D0 = s1;
    const int64_t D1 = s2 / s1;
    const int64_t D2 = s3 / s2;
    if (D0 < t->ne[0] || D1 < t->ne[1] || D2 < t->ne[2]) {
        return false;
    }

    // If ne[3] == 1 and base_nels isn't divisible by s3, handle via 3D slice inside flat base.
    if (t->ne[3] == 1 && (base_nels % s3) != 0) {
        if ((base_nels % s2) == 0) {
            const int64_t D2_flat = base_nels / s2;
            const int64_t o2 = elem_off / s2;
            const int64_t rem = elem_off % s2;
            const int64_t o1 = rem / s1;
            const int64_t o0 = rem % s1;
            if (o2 + t->ne[2] <= D2_flat && o1 + t->ne[1] <= D1 && o0 + t->ne[0] <= D0) {
                if (plan) {
                    plan->reshape_dims = {D2_flat, D1, D0};
                    plan->starts = {o2, o1, o0};
                    plan->limits = {o2 + t->ne[2], o1 + t->ne[1], o0 + t->ne[0]};
                }
                if (out_base) {
                    *out_base = base;
                }
                return true;
            }
        }
        return false;
    }

    if ((base_nels % s3) != 0) {
        return false;
    }
    const int64_t D3 = base_nels / s3;
    const int64_t o3 = elem_off / s3;
    int64_t rem = elem_off % s3;
    const int64_t o2 = rem / s2;
    rem %= s2;
    const int64_t o1 = rem / s1;
    const int64_t o0 = rem % s1;
    if (o3 + t->ne[3] > D3 || o2 + t->ne[2] > D2 || o1 + t->ne[1] > D1 || o0 + t->ne[0] > D0) {
        return false;
    }
    if (plan) {
        plan->reshape_dims = {D3, D2, D1, D0};
        plan->starts = {o3, o2, o1, o0};
        plan->limits = {o3 + t->ne[3], o2 + t->ne[2], o1 + t->ne[1], o0 + t->ne[0]};
    }
    if (out_base) {
        *out_base = base;
    }
    return true;
}

static bool is_supported_output_layout(const struct ggml_tensor* t) {
    if (!t || !ggml_is_contiguous(t)) {
        return false;
    }
    if (t->view_src == nullptr) {
        return true;
    }
    if (t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
        return false;
    }
    if (t->view_offs != 0 || ggml_nelements(t) != ggml_nelements(t->view_src) ||
        t->type != t->view_src->type) {
        return false;
    }
    return is_supported_output_layout(t->view_src);
}

static bool is_valid_strided_dense_layout(const struct ggml_tensor* t) {
    if (!t || !is_dense_supported_type(t->type)) {
        return false;
    }
    const size_t elem_sz = ggml_type_size(t->type);
    const size_t total_bytes = ggml_nbytes(t);
    if (elem_sz == 0 || total_bytes < elem_sz) {
        return false;
    }
    size_t max_off = 0;
    for (int d = 0; d < 4; ++d) {
        if (t->ne[d] <= 0) {
            return false;
        }
        const size_t extent = static_cast<size_t>(t->ne[d] - 1);
        if (extent > 0 && t->nb[d] > (total_bytes - elem_sz - max_off) / extent) {
            return false;
        }
        max_off += extent * t->nb[d];
    }
    return max_off + elem_sz <= total_bytes;
}

static void gather_strided_dense_bytes(const struct ggml_tensor* t,
                                       const uint8_t* raw_src,
                                       size_t raw_size,
                                       std::vector<uint8_t>& out_dense) {
    const size_t elem_sz = ggml_type_size(t->type);
    const size_t n_elems = static_cast<size_t>(ggml_nelements(t));
    out_dense.resize(n_elems * elem_sz);
    uint8_t* dst = out_dense.data();
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    const size_t nb0 = t->nb[0];
    const size_t nb1 = t->nb[1];
    const size_t nb2 = t->nb[2];
    const size_t nb3 = t->nb[3];
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        const size_t off3 = static_cast<size_t>(i3) * nb3;
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            const size_t off2 = off3 + static_cast<size_t>(i2) * nb2;
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                const size_t off1 = off2 + static_cast<size_t>(i1) * nb1;
                if (nb0 == elem_sz) {
                    const size_t row_bytes = static_cast<size_t>(ne0) * elem_sz;
                    if (off1 + row_bytes > raw_size) {
                        throw std::runtime_error("ggml-cloudtpu: strided gather out of bounds");
                    }
                    std::memcpy(dst, raw_src + off1, row_bytes);
                    dst += row_bytes;
                } else {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const size_t off0 = off1 + static_cast<size_t>(i0) * nb0;
                        if (off0 + elem_sz > raw_size) {
                            throw std::runtime_error("ggml-cloudtpu: strided gather out of bounds");
                        }
                        std::memcpy(dst, raw_src + off0, elem_sz);
                        dst += elem_sz;
                    }
                }
            }
        }
    }
}

static bool is_supported_input_layout(const struct ggml_tensor* t, int depth = 0) {
    if (!t || depth > 8) {
        return false;
    }
    if (t->view_src == nullptr) {
        return ggml_is_contiguous(t) || is_valid_strided_dense_layout(t);
    }
    if (is_quant_split_type(t->type)) {
        if (!is_supported_output_layout(t)) {
            return false;
        }
        // Split quantized PJRT buffers encode the parent tensor's exact block shape;
        // reject reshapes that alter logical dimensions relative to the underlying slot.
        for (const struct ggml_tensor* cur = t; cur->view_src != nullptr; cur = cur->view_src) {
            if (!ggml_are_same_shape(cur, cur->view_src)) {
                return false;
            }
        }
        return true;
    }
    const struct ggml_tensor* parent = view_parent(t);
    if (!parent) {
        return false;
    }
    if (t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
        if (!is_dense_supported_type(t->type) || t->type != parent->type || t->view_offs != parent->view_offs) {
            return false;
        }
        return is_supported_input_layout(parent, depth + 1);
    }
    if (t->op == GGML_OP_RESHAPE || (t->view_offs == parent->view_offs && ggml_is_contiguous(t) &&
                                     ggml_is_contiguous(parent) &&
                                     ggml_nelements(t) == ggml_nelements(parent) &&
                                     t->type == parent->type)) {
        return is_supported_input_layout(parent, depth + 1);
    }
    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_NONE) {
        const struct ggml_tensor* slice_base = nullptr;
        if (!try_build_view_slice(t, nullptr, &slice_base)) {
            return false;
        }
        return is_supported_input_layout(slice_base, depth + 1);
    }
    return is_supported_output_layout(t);
}

// =========================================================================
// HBM Buffer Management (PJRT_Buffer* Tensor Slot Table)
// =========================================================================

struct CloudTpuAllocError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct CloudTpuTensorSlot {
    size_t offset = 0;
    size_t nbytes = 0;
    enum ggml_type ggml_dt = GGML_TYPE_F32;
    cloudtpu::TensorStorageKind storage = cloudtpu::TensorStorageKind::DENSE;
    cloudtpu::TensorShardingKind sharding = cloudtpu::TensorShardingKind::REPLICATED;
    cloudtpu::PjrtDType dtype = cloudtpu::PjrtDType::F32;
    cloudtpu::PjrtDType scale_dtype = cloudtpu::PjrtDType::INVALID;
    cloudtpu::PjrtDType bitcast_target = cloudtpu::PjrtDType::S4;
    int64_t dims[4] = {1, 1, 1, 1};      // Logical shape [ne3, ne2, ne1, ne0]
    int64_t hbm_dims[4] = {1, 1, 1, 1};  // Physical HBM shape (padded along hbm_token_axis when >= 0)
    int hbm_token_axis = -1;             // -1 = unpadded; 0..3 = token axis padded on HBM
    // Single-chip: size() == 1 (on device_index). Multi-chip SPMD: size() == spmd_chips (devices 0..P-1).
    std::vector<cloudtpu::PjrtBufferPtr> hbm_primary;
    std::vector<cloudtpu::PjrtBufferPtr> hbm_scales;
    std::vector<cloudtpu::PjrtBufferPtr> hbm_mins;
    std::vector<uint8_t> quant_meta;  // Compact 16B/18B per QK_K block header for lossless get_tensor roundtrip
    std::vector<uint8_t> staging;
    bool staging_dirty = false;
    bool upload_failed = false;
    std::string upload_error;

    bool has_hbm() const { return !hbm_primary.empty() && hbm_primary[0] != nullptr; }
};

static void mark_slot_upload_failed(CloudTpuTensorSlot& slot, const std::string& err) {
    slot.hbm_primary.clear();
    slot.hbm_scales.clear();
    slot.hbm_mins.clear();
    std::vector<uint8_t>().swap(slot.quant_meta);
    std::vector<uint8_t>().swap(slot.staging);
    slot.staging_dirty = false;
    slot.upload_failed = true;
    slot.upload_error = err;
}

struct ggml_backend_cloudtpu_buffer_context {
    int device_index = 0;
    int spmd_chips = 1;
    int spmd_min_rows = 256;
    bool q4k_bf16_scales = false;
    CloudTpuQ4Mode q4_mode = CloudTpuQ4Mode::I4;
    bool q4k_u4 = false;
    uintptr_t virtual_base = 0;
    size_t size = 0;
    uint8_t clear_value = 0;

    std::mutex mu;
    std::unordered_map<size_t, CloudTpuTensorSlot> slots;  // keyed by byte offset from virtual_base

    void evict_overlapping_locked(size_t offset, size_t nbytes, size_t keep_exact_offset = SIZE_MAX) {
        // O(1) fast path: if an exact-matching slot [offset, offset + nbytes) already exists,
        // no other slot can overlap it (maintained invariant).
        if (keep_exact_offset == offset) {
            auto exact = slots.find(offset);
            if (exact != slots.end() && exact->second.nbytes == nbytes) {
                return;
            }
        }
        const size_t end = offset + nbytes;
        for (auto it = slots.begin(); it != slots.end();) {
            if (it->first == keep_exact_offset) {
                ++it;
                continue;
            }
            const size_t s0 = it->second.offset;
            const size_t s1 = s0 + it->second.nbytes;
            if (s0 < end && offset < s1) {
                it = slots.erase(it);
            } else {
                ++it;
            }
        }
    }
};

static std::atomic<uintptr_t> g_next_virtual_base{0x200000000ULL};

static bool is_cloudtpu_buft(ggml_backend_buffer_type_t buft);

static bool is_cloudtpu_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && is_cloudtpu_buft(buffer->buft);
}

static size_t tensor_offset_in_buffer(const ggml_backend_cloudtpu_buffer_context* bctx,
                                      const struct ggml_tensor* tensor) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(tensor->data);
    GGML_ASSERT(addr >= bctx->virtual_base);
    return static_cast<size_t>(addr - bctx->virtual_base);
}

static void unpack_q8_0_to_split(const block_q8_0* src,
                                 size_t num_rows,
                                 size_t nb,
                                 std::vector<int8_t>& out_qs,
                                 std::vector<uint16_t>& out_d) {
    const size_t total_blocks = num_rows * nb;
    out_qs.resize(total_blocks * QK8_0);
    out_d.resize(total_blocks);
    for (size_t b = 0; b < total_blocks; ++b) {
        std::memcpy(&out_d[b], &src[b].d, sizeof(uint16_t));
        std::memcpy(out_qs.data() + b * QK8_0, src[b].qs, QK8_0);
    }
}

static void pack_split_to_q8_0(const int8_t* qs,
                               const uint16_t* d,
                               size_t total_blocks,
                               block_q8_0* dst) {
    for (size_t b = 0; b < total_blocks; ++b) {
        std::memcpy(&dst[b].d, &d[b], sizeof(uint16_t));
        std::memcpy(dst[b].qs, qs + b * QK8_0, QK8_0);
    }
}

static void unpack_q4_0_to_split(const block_q4_0* src,
                                 size_t num_rows,
                                 size_t nb,
                                 std::vector<uint8_t>& out_qs,
                                 std::vector<uint16_t>& out_d) {
    const size_t total_blocks = num_rows * nb;
    out_qs.resize(total_blocks * (QK4_0 / 2));
    out_d.resize(total_blocks);
    for (size_t b = 0; b < total_blocks; ++b) {
        std::memcpy(&out_d[b], &src[b].d, sizeof(uint16_t));
        std::memcpy(out_qs.data() + b * (QK4_0 / 2), src[b].qs, QK4_0 / 2);
    }
}

static void pack_split_to_q4_0(const uint8_t* qs,
                               const uint16_t* d,
                               size_t total_blocks,
                               block_q4_0* dst) {
    for (size_t b = 0; b < total_blocks; ++b) {
        std::memcpy(&dst[b].d, &d[b], sizeof(uint16_t));
        std::memcpy(dst[b].qs, qs + b * (QK4_0 / 2), QK4_0 / 2);
    }
}

static inline void get_scale_min_k4_local(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static void unpack_q4_K_to_split(const block_q4_K* src,
                                 size_t num_rows,
                                 size_t nb_k,
                                 std::vector<uint8_t>& out_qs,
                                 std::vector<float>& out_d,
                                 std::vector<float>& out_m) {
    const size_t total_k_blocks = num_rows * nb_k;
    const size_t total_sub = total_k_blocks * 8;  // 8 sub-blocks of 32 per QK_K=256
    out_qs.resize(total_sub * 16);
    out_d.resize(total_sub);
    out_m.resize(total_sub);
    for (size_t b = 0; b < total_k_blocks; ++b) {
        uint16_t d16 = 0, dmin16 = 0;
        const auto* raw_blk = reinterpret_cast<const uint8_t*>(&src[b]);
        std::memcpy(&d16, raw_blk + 0, sizeof(uint16_t));
        std::memcpy(&dmin16, raw_blk + 2, sizeof(uint16_t));
        const float d = cloudtpu::f16_to_f32(d16);
        const float dmin = cloudtpu::f16_to_f32(dmin16);
        const uint8_t* q = src[b].qs;
        for (int g = 0; g < 4; ++g) {
            const int is = 2 * g;
            uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
            get_scale_min_k4_local(is + 0, src[b].scales, &sc0, &m0);
            get_scale_min_k4_local(is + 1, src[b].scales, &sc1, &m1);
            const size_t sub0 = b * 8 + static_cast<size_t>(is + 0);
            const size_t sub1 = b * 8 + static_cast<size_t>(is + 1);
            out_d[sub0] = d * static_cast<float>(sc0);
            out_m[sub0] = -(dmin * static_cast<float>(m0));
            out_d[sub1] = d * static_cast<float>(sc1);
            out_m[sub1] = -(dmin * static_cast<float>(m1));
            uint8_t* dst0 = out_qs.data() + sub0 * 16;
            uint8_t* dst1 = out_qs.data() + sub1 * 16;
            for (int l = 0; l < 16; ++l) {
                const uint8_t lo0 = q[l] & 0x0F;
                const uint8_t hi0 = q[l + 16] & 0x0F;
                dst0[l] = static_cast<uint8_t>(lo0 | (hi0 << 4));
                const uint8_t lo1 = q[l] >> 4;
                const uint8_t hi1 = q[l + 16] >> 4;
                dst1[l] = static_cast<uint8_t>(lo1 | (hi1 << 4));
            }
            q += 32;
        }
    }
}

static void unpack_q6_K_to_split(const block_q6_K* src,
                                 size_t num_rows,
                                 size_t nb_k,
                                 std::vector<int8_t>& out_qs,
                                 std::vector<float>& out_d) {
    const size_t total_k_blocks = num_rows * nb_k;
    const size_t total_sub = total_k_blocks * 16;  // 16 sub-blocks of 16 per QK_K=256
    out_qs.resize(total_sub * 16);
    out_d.resize(total_sub);
    for (size_t b = 0; b < total_k_blocks; ++b) {
        uint16_t d16 = 0;
        std::memcpy(&d16, &src[b].d, sizeof(uint16_t));
        const float d = cloudtpu::f16_to_f32(d16);
        const uint8_t* ql = src[b].ql;
        const uint8_t* qh = src[b].qh;
        const int8_t* sc = src[b].scales;
        int8_t* q_dst = out_qs.data() + b * QK_K;
        float* d_dst = out_d.data() + b * 16;
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const int8_t q1 = static_cast<int8_t>(((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32);
                const int8_t q2 = static_cast<int8_t>(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32);
                const int8_t q3 = static_cast<int8_t>(((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                const int8_t q4 = static_cast<int8_t>(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                q_dst[l + 0] = q1;
                q_dst[l + 32] = q2;
                q_dst[l + 64] = q3;
                q_dst[l + 96] = q4;
            }
            for (int s = 0; s < 8; ++s) {
                d_dst[s] = d * static_cast<float>(sc[s]);
            }
            ql += 64;
            qh += 32;
            sc += 8;
            q_dst += 128;
            d_dst += 8;
        }
    }
}

static void pack_split_to_q4_K(const uint8_t* qs,
                               const uint8_t* meta,
                               size_t total_k_blocks,
                               block_q4_K* dst) {
    for (size_t b = 0; b < total_k_blocks; ++b) {
        auto* raw_blk = reinterpret_cast<uint8_t*>(&dst[b]);
        std::memcpy(raw_blk, meta + b * 16, 16);  // dm (4B) + scales[12] (12B)
        uint8_t* q = dst[b].qs;
        for (int g = 0; g < 4; ++g) {
            const size_t sub0 = b * 8 + static_cast<size_t>(2 * g + 0);
            const size_t sub1 = b * 8 + static_cast<size_t>(2 * g + 1);
            const uint8_t* src0 = qs + sub0 * 16;
            const uint8_t* src1 = qs + sub1 * 16;
            for (int l = 0; l < 16; ++l) {
                q[l] = static_cast<uint8_t>((src0[l] & 0x0F) | ((src1[l] & 0x0F) << 4));
                q[l + 16] = static_cast<uint8_t>((src0[l] >> 4) | ((src1[l] >> 4) << 4));
            }
            q += 32;
        }
    }
}

static void pack_split_to_q6_K(const int8_t* qs,
                               const uint8_t* meta,
                               size_t total_k_blocks,
                               block_q6_K* dst) {
    for (size_t b = 0; b < total_k_blocks; ++b) {
        const uint8_t* m = meta + b * 18;
        std::memcpy(&dst[b].d, m + 0, 2);
        std::memcpy(dst[b].scales, m + 2, 16);
        uint8_t* ql = dst[b].ql;
        uint8_t* qh = dst[b].qh;
        const int8_t* q_src = qs + b * QK_K;
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t u1 = static_cast<uint8_t>(static_cast<int>(q_src[l + 0]) + 32);
                const uint8_t u2 = static_cast<uint8_t>(static_cast<int>(q_src[l + 32]) + 32);
                const uint8_t u3 = static_cast<uint8_t>(static_cast<int>(q_src[l + 64]) + 32);
                const uint8_t u4 = static_cast<uint8_t>(static_cast<int>(q_src[l + 96]) + 32);
                ql[l + 0] = static_cast<uint8_t>((u1 & 0x0F) | ((u3 & 0x0F) << 4));
                ql[l + 32] = static_cast<uint8_t>((u2 & 0x0F) | ((u4 & 0x0F) << 4));
                qh[l] = static_cast<uint8_t>(((u1 >> 4) & 3) | (((u2 >> 4) & 3) << 2) |
                                             (((u3 >> 4) & 3) << 4) | (((u4 >> 4) & 3) << 6));
            }
            ql += 64;
            qh += 32;
            q_src += 128;
        }
    }
}

static bool is_replicated_cache_or_mask_name(const char* name) {
    if (!name || name[0] == '\0') {
        return false;
    }
    return std::strstr(name, "mask") != nullptr ||
           std::strncmp(name, "cache_", 6) == 0 ||
           std::strncmp(name, "moe_cache.", 10) == 0 ||
           std::strstr(name, "_state_") != nullptr;
}

struct Q4SlotPlan {
    cloudtpu::TensorStorageKind storage = cloudtpu::TensorStorageKind::DENSE;
    cloudtpu::PjrtDType q_dtype = cloudtpu::PjrtDType::U8;
    cloudtpu::PjrtDType bitcast_target = cloudtpu::PjrtDType::S4;
};

static inline void write_nibble_4bit(uint8_t* dst_packed, size_t idx, uint8_t val4) {
    const size_t bi = idx >> 1;
    const uint8_t nib = val4 & 0x0FU;
    if ((idx & 1U) == 0U) {
        dst_packed[bi] = static_cast<uint8_t>((dst_packed[bi] & 0xF0U) | nib);
    } else {
        dst_packed[bi] = static_cast<uint8_t>((dst_packed[bi] & 0x0FU) | (nib << 4));
    }
}

static inline uint8_t read_nibble_4bit(const uint8_t* src_packed, size_t idx) {
    const uint8_t b = src_packed[idx >> 1];
    return ((idx & 1U) == 0U) ? (b & 0x0FU) : ((b >> 4) & 0x0FU);
}

static Q4SlotPlan resolve_q4_slot_plan(enum ggml_type t,
                                        int64_t K,
                                        int64_t N,
                                        int spmd_chips,
                                        cloudtpu::TensorShardingKind sh,
                                        CloudTpuQ4Mode q4_mode,
                                        bool q4k_u4) {
    Q4SlotPlan plan;
    if (t == GGML_TYPE_Q4_0) {
        if ((K % QK4_0) != 0) {
            return plan;
        }
        const int64_t n_local = (spmd_chips > 1 && sh == cloudtpu::TensorShardingKind::SHARD_N) ? (N / spmd_chips) : N;
        if (q4_mode == CloudTpuQ4Mode::I4) {
            plan.storage = cloudtpu::TensorStorageKind::Q4_0_I4;
            plan.q_dtype = cloudtpu::PjrtDType::S4;
            plan.bitcast_target = cloudtpu::PjrtDType::S4;
            return plan;
        }
        if (q4_mode == CloudTpuQ4Mode::BITCAST && (n_local % 2) == 0) {
            plan.storage = cloudtpu::TensorStorageKind::Q4_0_I4;
            plan.q_dtype = cloudtpu::PjrtDType::U8;
            plan.bitcast_target = cloudtpu::PjrtDType::S4;
            return plan;
        }
        plan.storage = cloudtpu::TensorStorageKind::Q4_0_SPLIT;
        plan.q_dtype = cloudtpu::PjrtDType::U8;
        return plan;
    }
    if (t == GGML_TYPE_Q4_K) {
        if ((K % QK_K) != 0) {
            return plan;
        }
        const int64_t n_local = (spmd_chips > 1 && sh == cloudtpu::TensorShardingKind::SHARD_N) ? (N / spmd_chips) : N;
        const cloudtpu::PjrtDType target4 = q4k_u4 ? cloudtpu::PjrtDType::U4 : cloudtpu::PjrtDType::S4;
        if (q4_mode == CloudTpuQ4Mode::I4) {
            plan.storage = cloudtpu::TensorStorageKind::Q4_K_I4;
            plan.q_dtype = target4;
            plan.bitcast_target = target4;
            return plan;
        }
        if (q4_mode == CloudTpuQ4Mode::BITCAST && (n_local % 2) == 0) {
            plan.storage = cloudtpu::TensorStorageKind::Q4_K_I4;
            plan.q_dtype = cloudtpu::PjrtDType::U8;
            plan.bitcast_target = target4;
            return plan;
        }
        plan.storage = cloudtpu::TensorStorageKind::Q4_K_SPLIT;
        plan.q_dtype = cloudtpu::PjrtDType::U8;
        return plan;
    }
    return plan;
}

static inline uint8_t encode_q4_host_elem(uint8_t u4_nib, bool use_s4) {
    const uint8_t u = u4_nib & 0x0FU;
    return use_s4 ? static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>(u) - 8)) : u;
}

static void unpack_q4_0_to_i4(const block_q4_0* src,
                               size_t B,
                               size_t N,
                               size_t nb,
                               bool use_s4,
                               bool packed_nibbles,
                               std::vector<uint8_t>& out_qs,
                               std::vector<uint16_t>& out_d_km) {
    out_d_km.resize(B * nb * N);
    const size_t total_elems = B * nb * QK4_0 * N;
    out_qs.assign(packed_nibbles ? cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, total_elems) : total_elems, 0);
    const uint8_t xor_mask = use_s4 ? 0x08U : 0x00U;

    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            const block_q4_0* row = src + (b * N + n) * nb;
            for (size_t blk = 0; blk < nb; ++blk) {
                std::memcpy(&out_d_km[(b * nb + blk) * N + n], &row[blk].d, sizeof(uint16_t));
            }
        }
        if (!packed_nibbles) {
            for (size_t blk = 0; blk < nb; ++blk) {
                for (size_t n = 0; n < N; ++n) {
                    const uint8_t* sq = src[((b * N + n) * nb) + blk].qs;
                    for (size_t j = 0; j < QK4_0 / 2; ++j) {
                        const uint8_t byte = sq[j];
                        out_qs[((b * nb + blk) * QK4_0 + j) * N + n] = encode_q4_host_elem(byte & 0x0FU, use_s4);
                        out_qs[((b * nb + blk) * QK4_0 + (j + 16)) * N + n] = encode_q4_host_elem(byte >> 4, use_s4);
                    }
                }
            }
        } else if ((N & 1U) == 0U) {
            const size_t n_half = N / 2;
            for (size_t blk = 0; blk < nb; ++blk) {
                for (size_t j = 0; j < QK4_0 / 2; ++j) {
                    uint8_t* dst_lo = out_qs.data() + (((b * nb + blk) * QK4_0 + j) * n_half);
                    uint8_t* dst_hi = out_qs.data() + (((b * nb + blk) * QK4_0 + (j + 16)) * n_half);
                    for (size_t nh = 0; nh < n_half; ++nh) {
                        const uint8_t b0 = src[((b * N + (2 * nh + 0)) * nb) + blk].qs[j];
                        const uint8_t b1 = src[((b * N + (2 * nh + 1)) * nb) + blk].qs[j];
                        dst_lo[nh] = static_cast<uint8_t>(((b0 & 0x0FU) ^ xor_mask) | (((b1 & 0x0FU) ^ xor_mask) << 4));
                        dst_hi[nh] = static_cast<uint8_t>((((b0 >> 4) & 0x0FU) ^ xor_mask) | ((((b1 >> 4) & 0x0FU) ^ xor_mask) << 4));
                    }
                }
            }
        } else {
            for (size_t blk = 0; blk < nb; ++blk) {
                for (size_t n = 0; n < N; ++n) {
                    const uint8_t* sq = src[((b * N + n) * nb) + blk].qs;
                    for (size_t j = 0; j < QK4_0 / 2; ++j) {
                        const uint8_t byte = sq[j];
                        const size_t idx_lo = (((b * nb + blk) * QK4_0 + j) * N + n);
                        const size_t idx_hi = (((b * nb + blk) * QK4_0 + (j + 16)) * N + n);
                        write_nibble_4bit(out_qs.data(), idx_lo, (byte & 0x0FU) ^ xor_mask);
                        write_nibble_4bit(out_qs.data(), idx_hi, ((byte >> 4) & 0x0FU) ^ xor_mask);
                    }
                }
            }
        }
    }
}

static void pack_i4_to_q4_0(const uint8_t* qs_km,
                               const uint16_t* d_km,
                               size_t B,
                               size_t N,
                               size_t nb,
                               bool use_s4,
                               bool packed_nibbles,
                               block_q4_0* dst) {
    const uint8_t xor_mask = use_s4 ? 0x08U : 0x00U;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            block_q4_0* row = dst + (b * N + n) * nb;
            for (size_t blk = 0; blk < nb; ++blk) {
                std::memcpy(&row[blk].d, &d_km[(b * nb + blk) * N + n], sizeof(uint16_t));
                for (size_t j = 0; j < QK4_0 / 2; ++j) {
                    const size_t idx_lo = (((b * nb + blk) * QK4_0 + j) * N + n);
                    const size_t idx_hi = (((b * nb + blk) * QK4_0 + (j + 16)) * N + n);
                    const uint8_t raw_lo = packed_nibbles ? read_nibble_4bit(qs_km, idx_lo) : (qs_km[idx_lo] & 0x0FU);
                    const uint8_t raw_hi = packed_nibbles ? read_nibble_4bit(qs_km, idx_hi) : (qs_km[idx_hi] & 0x0FU);
                    const uint8_t lo = raw_lo ^ xor_mask;
                    const uint8_t hi = raw_hi ^ xor_mask;
                    row[blk].qs[j] = static_cast<uint8_t>((lo & 0x0FU) | ((hi & 0x0FU) << 4));
                }
            }
        }
    }
}

static void unpack_q4_K_to_i4(const block_q4_K* src,
                               size_t B,
                               size_t N,
                               size_t nb_k,
                               bool use_s4,
                               bool packed_nibbles,
                               std::vector<uint8_t>& out_qs,
                               std::vector<float>& out_d_km,
                               std::vector<float>& out_m_km) {
    const size_t nb32 = nb_k * 8;
    out_d_km.resize(B * nb32 * N);
    out_m_km.resize(B * nb32 * N);
    const size_t total_elems = B * nb32 * 32 * N;
    out_qs.assign(packed_nibbles ? cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, total_elems) : total_elems, 0);
    const uint8_t xor_mask = use_s4 ? 0x08U : 0x00U;

    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            const block_q4_K* row = src + (b * N + n) * nb_k;
            for (size_t bk = 0; bk < nb_k; ++bk) {
                uint16_t d16 = 0, dmin16 = 0;
                const auto* raw_blk = reinterpret_cast<const uint8_t*>(&row[bk]);
                std::memcpy(&d16, raw_blk + 0, sizeof(uint16_t));
                std::memcpy(&dmin16, raw_blk + 2, sizeof(uint16_t));
                const float d = cloudtpu::f16_to_f32(d16);
                const float dmin = cloudtpu::f16_to_f32(dmin16);
                for (int is = 0; is < 8; ++is) {
                    uint8_t sc = 0, m = 0;
                    get_scale_min_k4_local(is, row[bk].scales, &sc, &m);
                    const float de = d * static_cast<float>(sc);
                    const float me = -(dmin * static_cast<float>(m));
                    const size_t sub = bk * 8 + static_cast<size_t>(is);
                    out_d_km[(b * nb32 + sub) * N + n] = de;
                    out_m_km[(b * nb32 + sub) * N + n] = me;
                }
            }
        }
        if (!packed_nibbles) {
            for (size_t bk = 0; bk < nb_k; ++bk) {
                for (size_t n = 0; n < N; ++n) {
                    const uint8_t* sq = src[((b * N + n) * nb_k) + bk].qs;
                    for (int g = 0; g < 4; ++g) {
                        const size_t sub0 = bk * 8 + static_cast<size_t>(2 * g + 0);
                        const size_t sub1 = bk * 8 + static_cast<size_t>(2 * g + 1);
                        for (int l = 0; l < 32; ++l) {
                            const uint8_t q = sq[32 * g + l];
                            out_qs[((b * nb32 + sub0) * 32 + static_cast<size_t>(l)) * N + n] =
                                encode_q4_host_elem(q & 0x0FU, use_s4);
                            out_qs[((b * nb32 + sub1) * 32 + static_cast<size_t>(l)) * N + n] =
                                encode_q4_host_elem(q >> 4, use_s4);
                        }
                    }
                }
            }
        } else if ((N & 1U) == 0U) {
            const size_t n_half = N / 2;
            for (size_t bk = 0; bk < nb_k; ++bk) {
                for (int g = 0; g < 4; ++g) {
                    const size_t sub0 = bk * 8 + static_cast<size_t>(2 * g + 0);
                    const size_t sub1 = bk * 8 + static_cast<size_t>(2 * g + 1);
                    for (int l = 0; l < 32; ++l) {
                        uint8_t* dst0 = out_qs.data() + (((b * nb32 + sub0) * 32 + static_cast<size_t>(l)) * n_half);
                        uint8_t* dst1 = out_qs.data() + (((b * nb32 + sub1) * 32 + static_cast<size_t>(l)) * n_half);
                        for (size_t nh = 0; nh < n_half; ++nh) {
                            const uint8_t q0 = src[((b * N + (2 * nh + 0)) * nb_k) + bk].qs[32 * g + l];
                            const uint8_t q1 = src[((b * N + (2 * nh + 1)) * nb_k) + bk].qs[32 * g + l];
                            dst0[nh] = static_cast<uint8_t>(((q0 & 0x0FU) ^ xor_mask) | (((q1 & 0x0FU) ^ xor_mask) << 4));
                            dst1[nh] = static_cast<uint8_t>((((q0 >> 4) & 0x0FU) ^ xor_mask) | ((((q1 >> 4) & 0x0FU) ^ xor_mask) << 4));
                        }
                    }
                }
            }
        } else {
            for (size_t bk = 0; bk < nb_k; ++bk) {
                for (size_t n = 0; n < N; ++n) {
                    const uint8_t* sq = src[((b * N + n) * nb_k) + bk].qs;
                    for (int g = 0; g < 4; ++g) {
                        const size_t sub0 = bk * 8 + static_cast<size_t>(2 * g + 0);
                        const size_t sub1 = bk * 8 + static_cast<size_t>(2 * g + 1);
                        for (int l = 0; l < 32; ++l) {
                            const uint8_t q = sq[32 * g + l];
                            const size_t idx0 = (((b * nb32 + sub0) * 32 + static_cast<size_t>(l)) * N + n);
                            const size_t idx1 = (((b * nb32 + sub1) * 32 + static_cast<size_t>(l)) * N + n);
                            write_nibble_4bit(out_qs.data(), idx0, (q & 0x0FU) ^ xor_mask);
                            write_nibble_4bit(out_qs.data(), idx1, ((q >> 4) & 0x0FU) ^ xor_mask);
                        }
                    }
                }
            }
        }
    }
}

static void pack_i4_to_q4_K(const uint8_t* qs_km,
                               const uint8_t* meta,
                               size_t B,
                               size_t N,
                               size_t nb_k,
                               bool use_s4,
                               bool packed_nibbles,
                               block_q4_K* dst) {
    const size_t nb32 = nb_k * 8;
    const uint8_t xor_mask = use_s4 ? 0x08U : 0x00U;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            const size_t row_blk_base = (b * N + n) * nb_k;
            for (size_t bk = 0; bk < nb_k; ++bk) {
                const size_t bi = row_blk_base + bk;
                std::memcpy(reinterpret_cast<uint8_t*>(&dst[bi]), meta + bi * 16, 16);
                uint8_t* q = dst[bi].qs;
                for (int g = 0; g < 4; ++g) {
                    const size_t sub0 = bk * 8 + static_cast<size_t>(2 * g + 0);
                    const size_t sub1 = bk * 8 + static_cast<size_t>(2 * g + 1);
                    for (int l = 0; l < 32; ++l) {
                        const size_t idx0 = (((b * nb32 + sub0) * 32 + static_cast<size_t>(l)) * N + n);
                        const size_t idx1 = (((b * nb32 + sub1) * 32 + static_cast<size_t>(l)) * N + n);
                        const uint8_t raw0 = packed_nibbles ? read_nibble_4bit(qs_km, idx0) : (qs_km[idx0] & 0x0FU);
                        const uint8_t raw1 = packed_nibbles ? read_nibble_4bit(qs_km, idx1) : (qs_km[idx1] & 0x0FU);
                        const uint8_t v0 = raw0 ^ xor_mask;
                        const uint8_t v1 = raw1 ^ xor_mask;
                        q[32 * g + l] = static_cast<uint8_t>((v0 & 0x0FU) | ((v1 & 0x0FU) << 4));
                    }
                }
            }
        }
    }
}

static cloudtpu::TensorShardingKind choose_upload_sharding(const struct ggml_tensor* tensor,
                                                           int spmd_chips,
                                                           int spmd_min_rows,
                                                           bool allow_sharding,
                                                           CloudTpuQ4Mode q4_mode = CloudTpuQ4Mode::SPLIT) {
    if (!allow_sharding || spmd_chips <= 1 || !tensor || tensor->view_src != nullptr) {
        return cloudtpu::TensorShardingKind::REPLICATED;
    }
    if (tensor->ne[3] != 1 || tensor->ne[2] != 1) {
        return cloudtpu::TensorShardingKind::REPLICATED;
    }
    const bool is_q = is_quant_split_type(tensor->type);
    const bool is_fp = (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 ||
                        tensor->type == GGML_TYPE_BF16);
    if ((!is_q && !is_fp) || !ggml_is_contiguous(tensor)) {
        return cloudtpu::TensorShardingKind::REPLICATED;
    }
    const int64_t K = tensor->ne[0];
    const int64_t N = tensor->ne[1];
    // Never shard 1D vectors (e.g. norm weights, biases), attention masks, or KV/state caches;
    // only shard 2D weight matrices.
    if (N <= 1 || K <= 1) {
        return cloudtpu::TensorShardingKind::REPLICATED;
    }
    if (is_replicated_cache_or_mask_name(tensor->name)) {
        return cloudtpu::TensorShardingKind::REPLICATED;
    }
    const int64_t q_align = quant_split_k_alignment(tensor->type);
    if (is_q && (K % q_align) != 0) {
        return cloudtpu::TensorShardingKind::REPLICATED;
    }
    // Packed U8 bitcast mode stores two rows per byte and requires an even per-chip N;
    // native S4/U4 mode transfers 1 host byte/element and supports arbitrary odd per-chip N.
    const bool is_q4_bitcast =
        (tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K) &&
        (q4_mode == CloudTpuQ4Mode::BITCAST);
    const int64_t n_align = is_q4_bitcast ? (static_cast<int64_t>(spmd_chips) * 2) : static_cast<int64_t>(spmd_chips);
    // Prefer SHARD_K for row-parallel FFN down projections and embedding tables (enables 1-collective
    // Megatron-LM FFN and low-comm sharded GET_ROWS); use SHARD_N for all other large 2D weights.
    const bool prefer_k = (tensor->name[0] != '\0') &&
                          (std::strstr(tensor->name, "ffn_down") != nullptr ||
                           std::strstr(tensor->name, "embd") != nullptr);
    if (prefer_k && K >= spmd_min_rows && (K % (static_cast<int64_t>(spmd_chips) * q_align)) == 0) {
        return cloudtpu::TensorShardingKind::SHARD_K;
    }
    if (N >= spmd_min_rows && (N % n_align) == 0) {
        return cloudtpu::TensorShardingKind::SHARD_N;
    }
    if (K >= spmd_min_rows && (K % (static_cast<int64_t>(spmd_chips) * q_align)) == 0) {
        return cloudtpu::TensorShardingKind::SHARD_K;
    }
    return cloudtpu::TensorShardingKind::REPLICATED;
}

template <typename T>
static std::vector<T> transpose_n_to_minor(const T* src, size_t B, size_t N, size_t K_outer) {
    // [B, N, K_outer] -> [B, K_outer, N] (places N as the minor dimension for TPU (8,128) tiles)
    std::vector<T> dst(B * K_outer * N);
    for (size_t b = 0; b < B; ++b) {
        const T* sb = src + b * N * K_outer;
        T* db = dst.data() + b * K_outer * N;
        for (size_t n = 0; n < N; ++n) {
            const T* row = sb + n * K_outer;
            for (size_t k = 0; k < K_outer; ++k) {
                db[k * N + n] = row[k];
            }
        }
    }
    return dst;
}

template <typename T>
static std::vector<T> transpose_minor_to_n(const T* src, size_t B, size_t N, size_t K_outer) {
    // [B, K_outer, N] -> [B, N, K_outer]
    std::vector<T> dst(B * N * K_outer);
    for (size_t b = 0; b < B; ++b) {
        const T* sb = src + b * K_outer * N;
        T* db = dst.data() + b * N * K_outer;
        for (size_t k = 0; k < K_outer; ++k) {
            const T* row = sb + k * N;
            for (size_t n = 0; n < N; ++n) {
                db[n * K_outer + k] = row[n];
            }
        }
    }
    return dst;
}

// Uploads a row-major tensor part across 1 or P chips, sharding along shard_axis when P > 1.
// Default shard_axis == -1 uses DENSE layout (SHARD_N -> axis 2, SHARD_K -> axis 3).
static std::vector<cloudtpu::PjrtBufferPtr> upload_sharded_part(cloudtpu::PjrtRuntime& rt,
                                                                int device_index,
                                                                int spmd_chips,
                                                                cloudtpu::TensorShardingKind sharding,
                                                                const void* data,
                                                                cloudtpu::PjrtDType dtype,
                                                                const std::vector<int64_t>& global_dims,
                                                                int shard_axis = -1) {
    if (spmd_chips <= 1) {
        return {rt.buffer_from_host(device_index, data, dtype, global_dims)};
    }
    if (sharding == cloudtpu::TensorShardingKind::REPLICATED) {
        return rt.buffer_from_host_replicated(spmd_chips, data, dtype, global_dims);
    }
    const int ax = (shard_axis >= 0) ? shard_axis
                                     : ((sharding == cloudtpu::TensorShardingKind::SHARD_N) ? 2 : 3);
    const uint8_t* raw = static_cast<const uint8_t*>(data);
    // OpenXLA PJRT BufferFromHostBuffer(byte_strides=nullptr) reads 1 byte per element for S4/U4.
    const size_t elem_sz = cloudtpu::pjrt_host_nbytes(dtype, 1);
    std::vector<int64_t> local_dims = global_dims;
    const int64_t ax_full = global_dims[static_cast<size_t>(ax)];
    const int64_t ax_local = ax_full / spmd_chips;
    local_dims[static_cast<size_t>(ax)] = ax_local;

    size_t outer_elems = 1;
    for (int i = 0; i < ax; ++i) outer_elems *= static_cast<size_t>(global_dims[static_cast<size_t>(i)]);
    size_t inner_elems = 1;
    for (size_t i = static_cast<size_t>(ax) + 1; i < global_dims.size(); ++i) {
        inner_elems *= static_cast<size_t>(global_dims[i]);
    }
    const size_t full_chunk_bytes = static_cast<size_t>(ax_full) * inner_elems * elem_sz;
    const size_t local_chunk_bytes = static_cast<size_t>(ax_local) * inner_elems * elem_sz;

    std::vector<cloudtpu::PjrtBufferPtr> bufs;
    bufs.reserve(static_cast<size_t>(spmd_chips));
    if (outer_elems == 1) {
        for (int p = 0; p < spmd_chips; ++p) {
            bufs.push_back(rt.buffer_from_host(
                p, raw + static_cast<size_t>(p) * local_chunk_bytes, dtype, local_dims));
        }
        return bufs;
    }

    std::vector<uint8_t> tmp(outer_elems * local_chunk_bytes);
    for (int p = 0; p < spmd_chips; ++p) {
        const size_t col_off = static_cast<size_t>(p) * local_chunk_bytes;
        for (size_t o = 0; o < outer_elems; ++o) {
            std::memcpy(tmp.data() + o * local_chunk_bytes,
                        raw + o * full_chunk_bytes + col_off,
                        local_chunk_bytes);
        }
        bufs.push_back(rt.buffer_from_host(p, tmp.data(), dtype, local_dims));
    }
    return bufs;
}

static void download_sharded_part(cloudtpu::PjrtRuntime& rt,
                                  cloudtpu::TensorShardingKind sharding,
                                  const std::vector<cloudtpu::PjrtBufferPtr>& bufs,
                                  void* dst,
                                  size_t dst_bytes,
                                  int shard_axis = -1) {
    if (bufs.empty() || !bufs[0]) {
        return;
    }
    if (bufs.size() == 1 || sharding == cloudtpu::TensorShardingKind::REPLICATED) {
        rt.buffer_to_host(bufs[0]->handle.get(), dst, dst_bytes);
        return;
    }
    const int ax = (shard_axis >= 0) ? shard_axis
                                     : ((sharding == cloudtpu::TensorShardingKind::SHARD_N) ? 2 : 3);
    uint8_t* out = static_cast<uint8_t*>(dst);
    const size_t P = bufs.size();
    const auto& ldims = bufs[0]->dims;
    size_t outer_elems = 1;
    for (int i = 0; i < ax; ++i) outer_elems *= static_cast<size_t>(ldims[static_cast<size_t>(i)]);
    size_t inner_elems = 1;
    for (size_t i = static_cast<size_t>(ax) + 1; i < ldims.size(); ++i) {
        inner_elems *= static_cast<size_t>(ldims[i]);
    }
    const size_t elem_sz = cloudtpu::pjrt_host_nbytes(bufs[0]->dtype, 1);
    const size_t local_chunk_bytes = static_cast<size_t>(ldims[static_cast<size_t>(ax)]) * inner_elems * elem_sz;
    if (outer_elems == 1) {
        for (size_t p = 0; p < P; ++p) {
            rt.buffer_to_host(bufs[p]->handle.get(), out + p * local_chunk_bytes, local_chunk_bytes);
        }
        return;
    }
    const size_t full_chunk_bytes = local_chunk_bytes * P;
    std::vector<uint8_t> tmp(outer_elems * local_chunk_bytes);
    for (size_t p = 0; p < P; ++p) {
        rt.buffer_to_host(bufs[p]->handle.get(), tmp.data(), tmp.size());
        const size_t col_off = p * local_chunk_bytes;
        for (size_t o = 0; o < outer_elems; ++o) {
            std::memcpy(out + o * full_chunk_bytes + col_off,
                        tmp.data() + o * local_chunk_bytes,
                        local_chunk_bytes);
        }
    }
}

static void upload_raw_tensor_to_slot(cloudtpu::PjrtRuntime& rt,
                                      int device_index,
                                      int spmd_chips,
                                      int spmd_min_rows,
                                      CloudTpuTensorSlot& slot,
                                      const struct ggml_tensor* tensor,
                                      const void* raw_data,
                                      bool allow_sharding = true,
                                      const cloudtpu::TensorShardingKind* forced_sharding = nullptr,
                                      bool q4k_bf16_scales = false,
                                      CloudTpuQ4Mode q4_mode = CloudTpuQ4Mode::I4,
                                      bool q4k_u4 = false) {
    slot.ggml_dt = tensor->type;
    slot.dims[0] = tensor->ne[3];
    slot.dims[1] = tensor->ne[2];
    slot.dims[2] = tensor->ne[1];
    slot.dims[3] = tensor->ne[0];
    slot.hbm_dims[0] = slot.dims[0];
    slot.hbm_dims[1] = slot.dims[1];
    slot.hbm_dims[2] = slot.dims[2];
    slot.hbm_dims[3] = slot.dims[3];
    slot.hbm_token_axis = -1;

    const int64_t B3 = tensor->ne[3];
    const int64_t B2 = tensor->ne[2];
    const int64_t N = tensor->ne[1];
    const int64_t K = tensor->ne[0];
    const size_t B = static_cast<size_t>(B3 * B2);
    const cloudtpu::TensorShardingKind sh =
        forced_sharding ? *forced_sharding
                        : choose_upload_sharding(tensor, spmd_chips, spmd_min_rows, allow_sharding, q4_mode);
    slot.sharding = sh;
    slot.scale_dtype = cloudtpu::PjrtDType::INVALID;
    slot.bitcast_target = cloudtpu::PjrtDType::S4;
    const int q5_shard_ax = (sh == cloudtpu::TensorShardingKind::SHARD_N) ? 4 : 2;
    const int q4_shard_ax = (sh == cloudtpu::TensorShardingKind::SHARD_N) ? 3 : 2;

    const bool is_contig = ggml_is_contiguous(tensor);
    if (is_contig && tensor->type == GGML_TYPE_Q8_0 && (K % QK8_0) == 0) {
        const size_t nb = static_cast<size_t>(K / QK8_0);
        const size_t nrows = static_cast<size_t>(B3 * B2 * N);
        std::vector<int8_t> qs;
        std::vector<uint16_t> d;
        unpack_q8_0_to_split(static_cast<const block_q8_0*>(raw_data), nrows, nb, qs, d);
        auto qs_km = transpose_n_to_minor(qs.data(), B, static_cast<size_t>(N), static_cast<size_t>(K));
        auto d_km = transpose_n_to_minor(d.data(), B, static_cast<size_t>(N), nb);
        slot.storage = cloudtpu::TensorStorageKind::Q8_0_SPLIT;
        slot.dtype = cloudtpu::PjrtDType::S8;
        slot.scale_dtype = cloudtpu::PjrtDType::F16;
        slot.hbm_primary = upload_sharded_part(
            rt, device_index, spmd_chips, sh, qs_km.data(), cloudtpu::PjrtDType::S8,
            {B3, B2, static_cast<int64_t>(nb), QK8_0, N}, q5_shard_ax);
        slot.hbm_scales = upload_sharded_part(
            rt, device_index, spmd_chips, sh, d_km.data(), cloudtpu::PjrtDType::F16,
            {B3, B2, static_cast<int64_t>(nb), N}, q4_shard_ax);
        slot.hbm_mins.clear();
    } else if (is_contig && tensor->type == GGML_TYPE_Q4_0 && (K % QK4_0) == 0) {
        const size_t nb = static_cast<size_t>(K / QK4_0);
        const Q4SlotPlan q4p = resolve_q4_slot_plan(tensor->type, K, N, spmd_chips, sh, q4_mode, q4k_u4);
        slot.storage = q4p.storage;
        slot.dtype = q4p.q_dtype;
        slot.scale_dtype = cloudtpu::PjrtDType::F16;
        slot.bitcast_target = q4p.bitcast_target;
        if (q4p.storage == cloudtpu::TensorStorageKind::Q4_0_I4) {
            std::vector<uint8_t> qs_data;
            std::vector<uint16_t> d_km;
            const bool use_s4 = (q4p.bitcast_target == cloudtpu::PjrtDType::S4);
            const bool packed_nibbles = (q4p.q_dtype == cloudtpu::PjrtDType::U8);
            unpack_q4_0_to_i4(static_cast<const block_q4_0*>(raw_data), B, static_cast<size_t>(N), nb,
                              use_s4, packed_nibbles, qs_data, d_km);
            const std::vector<int64_t> qs_dims = packed_nibbles
                                                     ? std::vector<int64_t>{B3, B2, static_cast<int64_t>(nb), QK4_0, N / 2}
                                                     : std::vector<int64_t>{B3, B2, static_cast<int64_t>(nb), QK4_0, N};
            slot.hbm_primary = upload_sharded_part(
                rt, device_index, spmd_chips, sh, qs_data.data(), q4p.q_dtype, qs_dims, q5_shard_ax);
            slot.hbm_scales = upload_sharded_part(
                rt, device_index, spmd_chips, sh, d_km.data(), cloudtpu::PjrtDType::F16,
                {B3, B2, static_cast<int64_t>(nb), N}, q4_shard_ax);
            slot.hbm_mins.clear();
        } else {
            const size_t nrows = static_cast<size_t>(B3 * B2 * N);
            std::vector<uint8_t> qs;
            std::vector<uint16_t> d;
            unpack_q4_0_to_split(static_cast<const block_q4_0*>(raw_data), nrows, nb, qs, d);
            auto qs_km = transpose_n_to_minor(qs.data(), B, static_cast<size_t>(N), static_cast<size_t>(K / 2));
            auto d_km = transpose_n_to_minor(d.data(), B, static_cast<size_t>(N), nb);
            slot.hbm_primary = upload_sharded_part(
                rt, device_index, spmd_chips, sh, qs_km.data(), cloudtpu::PjrtDType::U8,
                {B3, B2, static_cast<int64_t>(nb), QK4_0 / 2, N}, q5_shard_ax);
            slot.hbm_scales = upload_sharded_part(
                rt, device_index, spmd_chips, sh, d_km.data(), cloudtpu::PjrtDType::F16,
                {B3, B2, static_cast<int64_t>(nb), N}, q4_shard_ax);
            slot.hbm_mins.clear();
        }
    } else if (is_contig && tensor->type == GGML_TYPE_Q4_K && (K % QK_K) == 0) {
        const size_t nb_k = static_cast<size_t>(K / QK_K);
        const int64_t nb32 = K / 32;
        const size_t nrows = static_cast<size_t>(B3 * B2 * N);
        const size_t total_k_blocks = nrows * nb_k;
        const auto* src_blk = static_cast<const block_q4_K*>(raw_data);
        slot.quant_meta.resize(total_k_blocks * 16);
        for (size_t b = 0; b < total_k_blocks; ++b) {
            std::memcpy(slot.quant_meta.data() + b * 16, &src_blk[b], 16);
        }
        const Q4SlotPlan q4p = resolve_q4_slot_plan(tensor->type, K, N, spmd_chips, sh, q4_mode, q4k_u4);
        slot.storage = q4p.storage;
        slot.dtype = q4p.q_dtype;
        slot.bitcast_target = q4p.bitcast_target;
        std::vector<float> d_km_f32;
        std::vector<float> m_km_f32;
        if (q4p.storage == cloudtpu::TensorStorageKind::Q4_K_I4) {
            std::vector<uint8_t> qs_data;
            const bool use_s4 = (q4p.bitcast_target == cloudtpu::PjrtDType::S4);
            const bool packed_nibbles = (q4p.q_dtype == cloudtpu::PjrtDType::U8);
            unpack_q4_K_to_i4(src_blk, B, static_cast<size_t>(N), nb_k, use_s4, packed_nibbles,
                              qs_data, d_km_f32, m_km_f32);
            const std::vector<int64_t> qs_dims = packed_nibbles
                                                     ? std::vector<int64_t>{B3, B2, nb32, 32, N / 2}
                                                     : std::vector<int64_t>{B3, B2, nb32, 32, N};
            slot.hbm_primary = upload_sharded_part(
                rt, device_index, spmd_chips, sh, qs_data.data(), q4p.q_dtype, qs_dims, q5_shard_ax);
        } else {
            std::vector<uint8_t> qs;
            std::vector<float> d_eff;
            std::vector<float> m_eff;
            unpack_q4_K_to_split(src_blk, nrows, nb_k, qs, d_eff, m_eff);
            auto qs_km = transpose_n_to_minor(qs.data(), B, static_cast<size_t>(N), static_cast<size_t>(K / 2));
            d_km_f32 = transpose_n_to_minor(d_eff.data(), B, static_cast<size_t>(N), static_cast<size_t>(nb32));
            m_km_f32 = transpose_n_to_minor(m_eff.data(), B, static_cast<size_t>(N), static_cast<size_t>(nb32));
            slot.hbm_primary = upload_sharded_part(
                rt, device_index, spmd_chips, sh, qs_km.data(), cloudtpu::PjrtDType::U8,
                {B3, B2, nb32, 16, N}, q5_shard_ax);
        }
        if (q4k_bf16_scales) {
            // Store compact effective scales in BF16 (full F32 8-bit exponent range) rather than
            // F16 so large finite products (e.g. d * sc > 65504) never overflow to Inf/NaN.
            std::vector<uint16_t> d_km_bf16(d_km_f32.size());
            std::vector<uint16_t> m_km_bf16(m_km_f32.size());
            for (size_t i = 0; i < d_km_f32.size(); ++i) {
                d_km_bf16[i] = cloudtpu::f32_to_bf16(d_km_f32[i]);
                m_km_bf16[i] = cloudtpu::f32_to_bf16(m_km_f32[i]);
            }
            slot.scale_dtype = cloudtpu::PjrtDType::BF16;
            slot.hbm_scales = upload_sharded_part(
                rt, device_index, spmd_chips, sh, d_km_bf16.data(), cloudtpu::PjrtDType::BF16,
                {B3, B2, nb32, N}, q4_shard_ax);
            slot.hbm_mins = upload_sharded_part(
                rt, device_index, spmd_chips, sh, m_km_bf16.data(), cloudtpu::PjrtDType::BF16,
                {B3, B2, nb32, N}, q4_shard_ax);
        } else {
            slot.scale_dtype = cloudtpu::PjrtDType::F32;
            slot.hbm_scales = upload_sharded_part(
                rt, device_index, spmd_chips, sh, d_km_f32.data(), cloudtpu::PjrtDType::F32,
                {B3, B2, nb32, N}, q4_shard_ax);
            slot.hbm_mins = upload_sharded_part(
                rt, device_index, spmd_chips, sh, m_km_f32.data(), cloudtpu::PjrtDType::F32,
                {B3, B2, nb32, N}, q4_shard_ax);
        }
    } else if (is_contig && tensor->type == GGML_TYPE_Q6_K && (K % QK_K) == 0) {
        const size_t nb_k = static_cast<size_t>(K / QK_K);
        const int64_t nb16 = K / 16;
        const size_t nrows = static_cast<size_t>(B3 * B2 * N);
        const size_t total_k_blocks = nrows * nb_k;
        const auto* src_blk = static_cast<const block_q6_K*>(raw_data);
        std::vector<int8_t> qs;
        std::vector<float> d_eff;
        unpack_q6_K_to_split(src_blk, nrows, nb_k, qs, d_eff);
        slot.quant_meta.resize(total_k_blocks * 18);
        for (size_t b = 0; b < total_k_blocks; ++b) {
            std::memcpy(slot.quant_meta.data() + b * 18 + 0, &src_blk[b].d, 2);
            std::memcpy(slot.quant_meta.data() + b * 18 + 2, src_blk[b].scales, 16);
        }
        auto qs_km = transpose_n_to_minor(qs.data(), B, static_cast<size_t>(N), static_cast<size_t>(K));
        slot.storage = cloudtpu::TensorStorageKind::Q6_K_SPLIT;
        slot.dtype = cloudtpu::PjrtDType::S8;
        slot.hbm_primary = upload_sharded_part(
            rt, device_index, spmd_chips, sh, qs_km.data(), cloudtpu::PjrtDType::S8,
            {B3, B2, nb16, 16, N}, q5_shard_ax);
        if (q4k_bf16_scales) {
            // Store compact effective scales in BF16 (full F32 8-bit exponent range) rather than
            // F16 so valid Q6_K products like d=1024 * sc=64 = 65536 stay finite without NaN.
            std::vector<uint16_t> d_bf16(d_eff.size());
            for (size_t i = 0; i < d_eff.size(); ++i) {
                d_bf16[i] = cloudtpu::f32_to_bf16(d_eff[i]);
            }
            auto d_km = transpose_n_to_minor(d_bf16.data(), B, static_cast<size_t>(N), static_cast<size_t>(nb16));
            slot.scale_dtype = cloudtpu::PjrtDType::BF16;
            slot.hbm_scales = upload_sharded_part(
                rt, device_index, spmd_chips, sh, d_km.data(), cloudtpu::PjrtDType::BF16,
                {B3, B2, nb16, N}, q4_shard_ax);
        } else {
            auto d_km = transpose_n_to_minor(d_eff.data(), B, static_cast<size_t>(N), static_cast<size_t>(nb16));
            slot.scale_dtype = cloudtpu::PjrtDType::F32;
            slot.hbm_scales = upload_sharded_part(
                rt, device_index, spmd_chips, sh, d_km.data(), cloudtpu::PjrtDType::F32,
                {B3, B2, nb16, N}, q4_shard_ax);
        }
        slot.hbm_mins.clear();
    } else {
        const cloudtpu::PjrtDType pdt = ggml_type_to_pjrt(tensor->type);
        if (pdt != cloudtpu::PjrtDType::INVALID && is_contig) {
            slot.storage = cloudtpu::TensorStorageKind::DENSE;
            slot.dtype = pdt;
            slot.hbm_primary = upload_sharded_part(
                rt, device_index, spmd_chips, sh, raw_data, pdt, {B3, B2, N, K});
            slot.hbm_scales.clear();
            slot.hbm_mins.clear();
        } else {
            // Raw byte fallback for unsupported types (keeps tensor data intact for get_tensor / CPU fallback).
            slot.storage = cloudtpu::TensorStorageKind::DENSE;
            slot.sharding = cloudtpu::TensorShardingKind::REPLICATED;
            slot.dtype = cloudtpu::PjrtDType::U8;
            slot.hbm_primary = upload_sharded_part(
                rt, device_index, spmd_chips, cloudtpu::TensorShardingKind::REPLICATED,
                raw_data, cloudtpu::PjrtDType::U8, {static_cast<int64_t>(slot.nbytes)});
            slot.hbm_scales.clear();
            slot.hbm_mins.clear();
        }
    }
    std::vector<uint8_t>().swap(slot.staging);
    slot.staging_dirty = false;
    slot.upload_failed = false;
    slot.upload_error.clear();
}

static void download_slot_to_host(cloudtpu::PjrtRuntime& rt,
                                  const CloudTpuTensorSlot& slot,
                                  void* dst_full) {
    if (slot.staging_dirty && !slot.staging.empty()) {
        std::memcpy(dst_full, slot.staging.data(), slot.nbytes);
        return;
    }
    if (!slot.has_hbm()) {
        return;
    }
    const size_t shard_mul =
        (slot.sharding == cloudtpu::TensorShardingKind::REPLICATED) ? 1 : slot.hbm_primary.size();
    const size_t B = static_cast<size_t>(slot.dims[0] * slot.dims[1]);
    const size_t N = static_cast<size_t>(slot.dims[2]);
    const size_t K = static_cast<size_t>(slot.dims[3]);
    const int q5_shard_ax = (slot.sharding == cloudtpu::TensorShardingKind::SHARD_N) ? 4 : 2;
    const int q4_shard_ax = (slot.sharding == cloudtpu::TensorShardingKind::SHARD_N) ? 3 : 2;

    if (slot.storage == cloudtpu::TensorStorageKind::Q8_0_SPLIT) {
        const size_t total_blocks = slot.hbm_scales[0]->num_elements() * shard_mul;
        const size_t nb = K / QK8_0;
        std::vector<int8_t> qs_km(total_blocks * QK8_0);
        std::vector<uint16_t> d_km(total_blocks);
        download_sharded_part(rt, slot.sharding, slot.hbm_primary, qs_km.data(), qs_km.size(), q5_shard_ax);
        download_sharded_part(rt, slot.sharding, slot.hbm_scales, d_km.data(), d_km.size() * sizeof(uint16_t), q4_shard_ax);
        auto qs = transpose_minor_to_n(qs_km.data(), B, N, K);
        auto d = transpose_minor_to_n(d_km.data(), B, N, nb);
        pack_split_to_q8_0(qs.data(), d.data(), total_blocks, static_cast<block_q8_0*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q4_0_SPLIT) {
        const size_t total_blocks = slot.hbm_scales[0]->num_elements() * shard_mul;
        const size_t nb = K / QK4_0;
        std::vector<uint8_t> qs_km(total_blocks * (QK4_0 / 2));
        std::vector<uint16_t> d_km(total_blocks);
        download_sharded_part(rt, slot.sharding, slot.hbm_primary, qs_km.data(), qs_km.size(), q5_shard_ax);
        download_sharded_part(rt, slot.sharding, slot.hbm_scales, d_km.data(), d_km.size() * sizeof(uint16_t), q4_shard_ax);
        auto qs = transpose_minor_to_n(qs_km.data(), B, N, K / 2);
        auto d = transpose_minor_to_n(d_km.data(), B, N, nb);
        pack_split_to_q4_0(qs.data(), d.data(), total_blocks, static_cast<block_q4_0*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q4_0_I4) {
        const size_t total_blocks = slot.hbm_scales[0]->num_elements() * shard_mul;
        const size_t nb = K / QK4_0;
        const size_t total_elems = total_blocks * QK4_0;
        const bool packed_nibbles = (slot.dtype == cloudtpu::PjrtDType::U8);
        std::vector<uint8_t> qs_data(
            packed_nibbles ? cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, total_elems) : total_elems);
        std::vector<uint16_t> d_km(total_blocks);
        download_sharded_part(rt, slot.sharding, slot.hbm_primary, qs_data.data(), qs_data.size(), q5_shard_ax);
        download_sharded_part(rt, slot.sharding, slot.hbm_scales, d_km.data(), d_km.size() * sizeof(uint16_t), q4_shard_ax);
        const bool is_s4 = (slot.dtype == cloudtpu::PjrtDType::S4) ||
                           (slot.dtype == cloudtpu::PjrtDType::U8 && slot.bitcast_target == cloudtpu::PjrtDType::S4);
        pack_i4_to_q4_0(qs_data.data(), d_km.data(), B, N, nb, is_s4, packed_nibbles, static_cast<block_q4_0*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q4_K_SPLIT) {
        const size_t total_sub = slot.hbm_scales[0]->num_elements() * shard_mul;
        const size_t total_k_blocks = total_sub / 8;
        std::vector<uint8_t> qs_km(total_sub * 16);
        download_sharded_part(rt, slot.sharding, slot.hbm_primary, qs_km.data(), qs_km.size(), q5_shard_ax);
        auto qs = transpose_minor_to_n(qs_km.data(), B, N, K / 2);
        pack_split_to_q4_K(qs.data(), slot.quant_meta.data(), total_k_blocks, static_cast<block_q4_K*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q4_K_I4) {
        const size_t total_sub = slot.hbm_scales[0]->num_elements() * shard_mul;
        const size_t nb_k = K / QK_K;
        const size_t total_elems = total_sub * 32;
        const bool packed_nibbles = (slot.dtype == cloudtpu::PjrtDType::U8);
        std::vector<uint8_t> qs_data(
            packed_nibbles ? cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, total_elems) : total_elems);
        download_sharded_part(rt, slot.sharding, slot.hbm_primary, qs_data.data(), qs_data.size(), q5_shard_ax);
        const bool is_s4 = (slot.dtype == cloudtpu::PjrtDType::S4) ||
                           (slot.dtype == cloudtpu::PjrtDType::U8 && slot.bitcast_target == cloudtpu::PjrtDType::S4);
        pack_i4_to_q4_K(qs_data.data(), slot.quant_meta.data(), B, N, nb_k, is_s4, packed_nibbles, static_cast<block_q4_K*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q6_K_SPLIT) {
        const size_t total_sub = slot.hbm_scales[0]->num_elements() * shard_mul;
        const size_t total_k_blocks = total_sub / 16;
        std::vector<int8_t> qs_km(total_sub * 16);
        download_sharded_part(rt, slot.sharding, slot.hbm_primary, qs_km.data(), qs_km.size(), q5_shard_ax);
        auto qs = transpose_minor_to_n(qs_km.data(), B, N, K);
        pack_split_to_q6_K(qs.data(), slot.quant_meta.data(), total_k_blocks, static_cast<block_q6_K*>(dst_full));
    } else if (slot.hbm_token_axis >= 0 && slot.hbm_token_axis < 4 &&
               slot.hbm_dims[slot.hbm_token_axis] > slot.dims[slot.hbm_token_axis] &&
               slot.storage == cloudtpu::TensorStorageKind::DENSE) {
        const size_t elem_sz = cloudtpu::pjrt_dtype_size(slot.dtype);
        size_t hbm_bytes = 0;
        if (!cloudtpu::checked_tensor_nbytes(slot.hbm_dims, elem_sz, &hbm_bytes) ||
            hbm_bytes < slot.nbytes) {
            throw std::overflow_error("ggml-cloudtpu: padded HBM slot byte size overflow");
        }
        std::vector<uint8_t> hbm_tmp(hbm_bytes);
        download_sharded_part(rt, slot.sharding, slot.hbm_primary, hbm_tmp.data(), hbm_bytes);
        cloudtpu::slice_dense_host_tensor(
            hbm_tmp.data(), dst_full, elem_sz, slot.hbm_dims, slot.dims, slot.hbm_token_axis);
    } else {
        download_sharded_part(rt, slot.sharding, slot.hbm_primary, dst_full, slot.nbytes);
    }
}

static void ggml_backend_cloudtpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    delete bctx;
}

static void* ggml_backend_cloudtpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    return reinterpret_cast<void*>(bctx->virtual_base);
}

// Patches a byte range belonging to a view into its canonical root tensor's slot,
// initializing the root slot with bctx->clear_value on the first view write so parent
// reads always observe the patched bytes and untouched regions stay preserved.
static bool patch_view_into_root_slot_locked(cloudtpu::PjrtRuntime& rt,
                                             ggml_backend_cloudtpu_buffer_context* bctx,
                                             ggml_backend_buffer_t buffer,
                                             const struct ggml_tensor* view_tensor,
                                             size_t t_off,
                                             size_t offset,
                                             size_t size,
                                             const void* src_data,
                                             uint8_t fill_value,
                                             bool flush_now) {
    if (!view_tensor || view_tensor->view_src == nullptr) {
        return false;
    }
    const struct ggml_tensor* root = unwrap_view_src(view_tensor);
    if (!root || root->buffer != buffer) {
        return false;
    }
    const size_t root_off = tensor_offset_in_buffer(bctx, root);
    const size_t root_bytes = ggml_nbytes(root);
    if (t_off < root_off || offset > SIZE_MAX - t_off || size > SIZE_MAX - (t_off + offset) ||
        t_off + offset + size > root_off + root_bytes) {
        return false;
    }
    auto pit = bctx->slots.find(root_off);
    if (pit == bctx->slots.end() || pit->second.nbytes != root_bytes) {
        const cloudtpu::TensorShardingKind root_sh = choose_upload_sharding(
            root, bctx->spmd_chips, bctx->spmd_min_rows, /*allow_sharding=*/true, bctx->q4_mode);
        const size_t rel_new = (t_off - root_off) + offset;
        if (rel_new == 0 && size == root_bytes) {
            CloudTpuTensorSlot next;
            next.offset = root_off;
            next.nbytes = root_bytes;
            if (src_data != nullptr) {
                upload_raw_tensor_to_slot(
                    rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                    next, root, src_data, /*allow_sharding=*/true, &root_sh,
                    bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
            } else {
                std::vector<uint8_t> tmp(root_bytes, fill_value);
                upload_raw_tensor_to_slot(
                    rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                    next, root, tmp.data(), /*allow_sharding=*/true, &root_sh,
                    bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
            }
            bctx->evict_overlapping_locked(root_off, root_bytes, root_off);
            bctx->slots[root_off] = std::move(next);
            return true;
        }
        std::vector<uint8_t> init_staging(root_bytes, bctx->clear_value);
        if (src_data != nullptr) {
            std::memcpy(init_staging.data() + rel_new, src_data, size);
        } else {
            std::memset(init_staging.data() + rel_new, fill_value, size);
        }
        bctx->evict_overlapping_locked(root_off, root_bytes, root_off);
        CloudTpuTensorSlot& parent = bctx->slots[root_off];
        parent.offset = root_off;
        parent.nbytes = root_bytes;
        parent.sharding = root_sh;
        parent.hbm_primary.clear();
        parent.hbm_scales.clear();
        parent.hbm_mins.clear();
        parent.staging = std::move(init_staging);
        parent.staging_dirty = true;
        parent.upload_failed = false;
        parent.upload_error.clear();
        if (flush_now) {
            CloudTpuTensorSlot next;
            next.offset = parent.offset;
            next.nbytes = parent.nbytes;
            upload_raw_tensor_to_slot(
                rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                next, root, parent.staging.data(), /*allow_sharding=*/true, &root_sh,
                bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
            parent = std::move(next);
        }
        return true;
    }
    CloudTpuTensorSlot& parent = pit->second;
    const cloudtpu::TensorShardingKind root_sh = parent.sharding;
    const size_t rel = (t_off - parent.offset) + offset;
    if (rel == 0 && size == root_bytes) {
        CloudTpuTensorSlot next;
        next.offset = parent.offset;
        next.nbytes = parent.nbytes;
        if (src_data != nullptr) {
            upload_raw_tensor_to_slot(
                rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                next, root, src_data, /*allow_sharding=*/true, &root_sh,
                bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
        } else {
            std::vector<uint8_t> tmp(root_bytes, fill_value);
            upload_raw_tensor_to_slot(
                rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                next, root, tmp.data(), /*allow_sharding=*/true, &root_sh,
                bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
        }
        parent = std::move(next);
        return true;
    }
    if (parent.upload_failed) {
        throw CloudTpuAllocError(parent.upload_error);
    }
    if (!parent.staging_dirty) {
        std::vector<uint8_t> tmp(parent.nbytes, bctx->clear_value);
        if (parent.has_hbm()) {
            download_slot_to_host(rt, parent, tmp.data());
        }
        parent.staging = std::move(tmp);
        parent.staging_dirty = true;
    }
    // Preserve the root slot's existing shard layout so a partial view patch never expands
    // a sharded 2D matrix into P replicated per-chip copies.
    if (src_data != nullptr) {
        std::memcpy(parent.staging.data() + rel, src_data, size);
    } else {
        std::memset(parent.staging.data() + rel, fill_value, size);
    }
    // Keep partial host writes batched in staging until device execution needs the tensor
    // (avoids repeated full-tensor D2H+H2D roundtrips during session KV restore).
    if (flush_now) {
        CloudTpuTensorSlot next;
        next.offset = parent.offset;
        next.nbytes = parent.nbytes;
        upload_raw_tensor_to_slot(
            rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
            next, root, parent.staging.data(), /*allow_sharding=*/true, &root_sh,
            bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
        parent = std::move(next);
    }
    return true;
}

static bool is_partial_view_write(ggml_backend_cloudtpu_buffer_context* bctx,
                                  ggml_backend_buffer_t buffer,
                                  const struct ggml_tensor* tensor,
                                  size_t offset,
                                  size_t size) {
    if (!bctx || !tensor || tensor->view_src == nullptr) {
        return false;
    }
    const struct ggml_tensor* root = unwrap_view_src(tensor);
    if (!root || root->buffer != buffer) {
        return false;
    }
    const size_t t_off = tensor_offset_in_buffer(bctx, tensor);
    const size_t root_off = tensor_offset_in_buffer(bctx, root);
    const size_t root_bytes = ggml_nbytes(root);
    if (t_off < root_off || offset > SIZE_MAX - t_off || size > SIZE_MAX - (t_off + offset) ||
        t_off + offset + size > root_off + root_bytes) {
        return false;
    }
    const size_t rel = (t_off - root_off) + offset;
    return !(rel == 0 && size == root_bytes);
}

static void record_buffer_upload_failure(ggml_backend_cloudtpu_buffer_context* bctx,
                                         ggml_backend_buffer_t buffer,
                                         const struct ggml_tensor* tensor,
                                         const char* op_name,
                                         const std::string& err) {
    GGML_LOG_ERROR("ggml-cloudtpu: %s failed for '%s': %s\n",
                   op_name,
                   (tensor && tensor->name[0] != '\0') ? tensor->name : "?",
                   err.c_str());
    if (!bctx || !tensor) {
        return;
    }
    const struct ggml_tensor* target = tensor;
    if (tensor->view_src != nullptr) {
        const struct ggml_tensor* root = unwrap_view_src(tensor);
        if (root && root->buffer == buffer) {
            target = root;
        }
    }
    const size_t err_off = tensor_offset_in_buffer(bctx, target);
    const size_t err_bytes = ggml_nbytes(target);
    std::lock_guard<std::mutex> lock(bctx->mu);
    bctx->evict_overlapping_locked(err_off, err_bytes, err_off);
    CloudTpuTensorSlot& err_slot = bctx->slots[err_off];
    err_slot.offset = err_off;
    err_slot.nbytes = err_bytes;
    mark_slot_upload_failed(err_slot, err);
}

static void ggml_backend_cloudtpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                       struct ggml_tensor* tensor,
                                                       uint8_t value,
                                                       size_t offset,
                                                       size_t size) {
    if (size == 0) {
        return;
    }
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    const size_t t_bytes = ggml_nbytes(tensor);
    const bool partial_view = is_partial_view_write(bctx, buffer, tensor, offset, size);
    const bool is_compute_buf =
        (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    try {
        cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
        const size_t t_off = tensor_offset_in_buffer(bctx, tensor);

        std::lock_guard<std::mutex> lock(bctx->mu);
        if (patch_view_into_root_slot_locked(
                rt, bctx, buffer, tensor, t_off, offset, size, nullptr, value, /*flush_now=*/false)) {
            return;
        }
        if (offset == 0 && size == t_bytes) {
            std::vector<uint8_t> tmp(t_bytes, value);
            CloudTpuTensorSlot next;
            next.offset = t_off;
            next.nbytes = t_bytes;
            upload_raw_tensor_to_slot(
                rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                next, tensor, tmp.data(), /*allow_sharding=*/true, nullptr,
                bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
            bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
            bctx->slots[t_off] = std::move(next);
        } else {
            auto it = bctx->slots.find(t_off);
            if (it != bctx->slots.end() && it->second.nbytes == t_bytes) {
                CloudTpuTensorSlot& slot = it->second;
                if (slot.upload_failed) {
                    throw CloudTpuAllocError(slot.upload_error);
                }
                if (!slot.staging_dirty) {
                    std::vector<uint8_t> tmp(t_bytes, bctx->clear_value);
                    if (slot.has_hbm()) {
                        download_slot_to_host(rt, slot, tmp.data());
                    }
                    slot.staging = std::move(tmp);
                    slot.staging_dirty = true;
                }
                std::memset(slot.staging.data() + offset, value, size);
            } else {
                std::vector<uint8_t> tmp(t_bytes, bctx->clear_value);
                std::memset(tmp.data() + offset, value, size);
                bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
                CloudTpuTensorSlot& slot = bctx->slots[t_off];
                slot.offset = t_off;
                slot.nbytes = t_bytes;
                slot.sharding = choose_upload_sharding(
                    tensor, bctx->spmd_chips, bctx->spmd_min_rows, /*allow_sharding=*/true, bctx->q4_mode);
                slot.hbm_primary.clear();
                slot.hbm_scales.clear();
                slot.hbm_mins.clear();
                slot.staging = std::move(tmp);
                slot.staging_dirty = true;
                slot.upload_failed = false;
                slot.upload_error.clear();
            }
        }
    } catch (const std::exception& e) {
        if (is_compute_buf && !partial_view && offset == 0 && size == t_bytes) {
            record_buffer_upload_failure(bctx, buffer, tensor, "memset_tensor", e.what());
            return;
        }
        // Preserve any existing valid weights / KV-cache slot intact and propagate to caller.
        GGML_LOG_ERROR("ggml-cloudtpu: memset_tensor failed for '%s': %s\n",
                       (tensor && tensor->name[0] != '\0') ? tensor->name : "?",
                       e.what());
        throw;
    }
}

static void ggml_backend_cloudtpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    struct ggml_tensor* tensor,
                                                    const void* data,
                                                    size_t offset,
                                                    size_t size) {
    if (size == 0) {
        return;
    }
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    const size_t t_bytes = ggml_nbytes(tensor);
    const bool partial_view = is_partial_view_write(bctx, buffer, tensor, offset, size);
    const bool is_compute_buf =
        (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    try {
        cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
        const size_t t_off = tensor_offset_in_buffer(bctx, tensor);

        std::lock_guard<std::mutex> lock(bctx->mu);
        if (patch_view_into_root_slot_locked(
                rt, bctx, buffer, tensor, t_off, offset, size, data, 0, /*flush_now=*/false)) {
            return;
        }
        if (offset == 0 && size == t_bytes) {
            CloudTpuTensorSlot next;
            next.offset = t_off;
            next.nbytes = t_bytes;
            upload_raw_tensor_to_slot(
                rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                next, tensor, data, /*allow_sharding=*/true, nullptr,
                bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
            bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
            bctx->slots[t_off] = std::move(next);
        } else {
            auto it = bctx->slots.find(t_off);
            CloudTpuTensorSlot* slot_ptr = nullptr;
            if (it != bctx->slots.end() && it->second.nbytes == t_bytes) {
                slot_ptr = &it->second;
                if (slot_ptr->upload_failed) {
                    throw CloudTpuAllocError(slot_ptr->upload_error);
                }
                if (!slot_ptr->staging_dirty) {
                    std::vector<uint8_t> tmp(t_bytes, bctx->clear_value);
                    if (slot_ptr->has_hbm()) {
                        download_slot_to_host(rt, *slot_ptr, tmp.data());
                    }
                    slot_ptr->staging = std::move(tmp);
                    slot_ptr->staging_dirty = true;
                }
                std::memcpy(slot_ptr->staging.data() + offset, data, size);
            } else {
                std::vector<uint8_t> tmp(t_bytes, bctx->clear_value);
                std::memcpy(tmp.data() + offset, data, size);
                bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
                CloudTpuTensorSlot& slot = bctx->slots[t_off];
                slot.offset = t_off;
                slot.nbytes = t_bytes;
                slot.sharding = choose_upload_sharding(
                    tensor, bctx->spmd_chips, bctx->spmd_min_rows, /*allow_sharding=*/true, bctx->q4_mode);
                slot.hbm_primary.clear();
                slot.hbm_scales.clear();
                slot.hbm_mins.clear();
                slot.staging = std::move(tmp);
                slot.staging_dirty = true;
                slot.upload_failed = false;
                slot.upload_error.clear();
                slot_ptr = &slot;
            }
            // If full buffer has been populated via tail write, flush to HBM immediately;
            // on compute buffers keep staging intact on OOM so graph_compute reports ALLOC_FAILED.
            if (offset + size == t_bytes) {
                try {
                    CloudTpuTensorSlot next;
                    next.offset = slot_ptr->offset;
                    next.nbytes = slot_ptr->nbytes;
                    const cloudtpu::TensorShardingKind slot_sh = slot_ptr->sharding;
                    upload_raw_tensor_to_slot(
                        rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                        next, tensor, slot_ptr->staging.data(), /*allow_sharding=*/true, &slot_sh,
                        bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
                    *slot_ptr = std::move(next);
                } catch (const std::exception& e) {
                    if (!is_compute_buf) {
                        throw;
                    }
                    GGML_LOG_ERROR("ggml-cloudtpu: deferred tail flush failed for '%s' (keeping staging): %s\n",
                                   (tensor && tensor->name[0] != '\0') ? tensor->name : "?",
                                   e.what());
                }
            }
        }
    } catch (const std::exception& e) {
        if (is_compute_buf && !partial_view && offset == 0 && size == t_bytes) {
            record_buffer_upload_failure(bctx, buffer, tensor, "set_tensor", e.what());
            return;
        }
        // Preserve any existing valid weights / KV-cache slot intact and propagate to caller.
        GGML_LOG_ERROR("ggml-cloudtpu: set_tensor failed for '%s': %s\n",
                       (tensor && tensor->name[0] != '\0') ? tensor->name : "?",
                       e.what());
        throw;
    }
}

static void ggml_backend_cloudtpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    const struct ggml_tensor* tensor,
                                                    void* data,
                                                    size_t offset,
                                                    size_t size) {
    if (size == 0) {
        return;
    }
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    const size_t t_off = tensor_offset_in_buffer(bctx, tensor);
    const size_t t_bytes = ggml_nbytes(tensor);

    std::lock_guard<std::mutex> lock(bctx->mu);
    auto it = bctx->slots.find(t_off);
    if (it != bctx->slots.end() && it->second.nbytes >= offset + size) {
        const CloudTpuTensorSlot& slot = it->second;
        if (slot.upload_failed) {
            throw std::runtime_error("ggml-cloudtpu: get_tensor on failed slot '" +
                                     std::string(tensor->name) + "': " + slot.upload_error);
        }
        const bool is_padded_slot =
            (slot.hbm_token_axis >= 0 && slot.hbm_token_axis < 4 &&
             slot.hbm_dims[slot.hbm_token_axis] > slot.dims[slot.hbm_token_axis]);
        if (offset == 0 && size == slot.nbytes && !is_padded_slot &&
            slot.storage == cloudtpu::TensorStorageKind::DENSE &&
            slot.sharding == cloudtpu::TensorShardingKind::REPLICATED && !slot.staging_dirty && slot.has_hbm()) {
            rt.buffer_to_host(slot.hbm_primary[0]->handle.get(), data, size);
            return;
        }
        if (offset == 0 && size == slot.nbytes && is_padded_slot &&
            slot.storage == cloudtpu::TensorStorageKind::DENSE && !slot.staging_dirty && slot.has_hbm()) {
            download_slot_to_host(rt, slot, data);
            return;
        }
        std::vector<uint8_t> tmp(slot.nbytes);
        download_slot_to_host(rt, slot, tmp.data());
        std::memcpy(data, tmp.data() + offset, size);
        return;
    }

    // Check if tensor is a sub-slice view inside a parent slot.
    for (const auto& kv : bctx->slots) {
        const CloudTpuTensorSlot& parent = kv.second;
        if (t_off >= parent.offset && t_off + t_bytes <= parent.offset + parent.nbytes) {
            if (parent.upload_failed) {
                throw std::runtime_error("ggml-cloudtpu: get_tensor on failed parent slot '" +
                                         std::string(tensor->name) + "': " + parent.upload_error);
            }
            const size_t rel = (t_off - parent.offset) + offset;
            if (rel + size <= parent.nbytes) {
                std::vector<uint8_t> tmp(parent.nbytes);
                download_slot_to_host(rt, parent, tmp.data());
                std::memcpy(data, tmp.data() + rel, size);
                return;
            }
        }
    }

    // Fill-byte fallback is reserved for unwritten tensors (honors buffer clear value).
    std::memset(data, bctx->clear_value, size);
}

static bool ggml_backend_cloudtpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    const struct ggml_tensor* src,
                                                    struct ggml_tensor* dst) {
    const size_t src_bytes = ggml_nbytes(src);
    const size_t dst_bytes = ggml_nbytes(dst);
    if (src_bytes != dst_bytes) {
        return false;
    }
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (!src->data) {
            return false;
        }
        ggml_backend_cloudtpu_buffer_set_tensor(buffer, dst, src->data, 0, src_bytes);
        return true;
    }
    // Restrict zero-copy HBM slot sharing to exact, non-view, contiguous allocations with
    // identical type and shape. View/subview copies fall back to GGML's byte-range get/set
    // path so parent slots are never clobbered or over-copied.
    if (is_cloudtpu_buffer(src->buffer) &&
        src->view_src == nullptr && dst->view_src == nullptr &&
        ggml_is_contiguous(src) && ggml_is_contiguous(dst) &&
        src->type == dst->type && ggml_are_same_shape(src, dst)) {
        auto* src_bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(src->buffer->context);
        auto* dst_bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
        if (src_bctx->device_index == dst_bctx->device_index && src_bctx->spmd_chips == dst_bctx->spmd_chips) {
            const size_t src_off = tensor_offset_in_buffer(src_bctx, src);
            const size_t dst_off = tensor_offset_in_buffer(dst_bctx, dst);
            auto slot_matches_exact = [&](const CloudTpuTensorSlot& slot) -> bool {
                return !slot.upload_failed && !slot.staging_dirty && slot.has_hbm() &&
                       slot.nbytes == src_bytes && slot.ggml_dt == src->type &&
                       slot.dims[0] == src->ne[3] && slot.dims[1] == src->ne[2] &&
                       slot.dims[2] == src->ne[1] && slot.dims[3] == src->ne[0];
            };
            auto has_enclosing_parent_locked = [&](const ggml_backend_cloudtpu_buffer_context* bctx,
                                                   size_t off, size_t nbytes) -> bool {
                for (const auto& kv : bctx->slots) {
                    const CloudTpuTensorSlot& s = kv.second;
                    if (s.nbytes > nbytes && off >= s.offset && off + nbytes <= s.offset + s.nbytes) {
                        return true;
                    }
                }
                return false;
            };
            if (src_bctx == dst_bctx) {
                std::lock_guard<std::mutex> lock(src_bctx->mu);
                auto sit = src_bctx->slots.find(src_off);
                if (sit != src_bctx->slots.end() &&
                    !has_enclosing_parent_locked(dst_bctx, dst_off, dst_bytes)) {
                    if (sit->second.upload_failed) {
                        const std::string err = sit->second.upload_error;
                        src_bctx->evict_overlapping_locked(dst_off, dst_bytes, dst_off);
                        CloudTpuTensorSlot& dslot = src_bctx->slots[dst_off];
                        dslot.offset = dst_off;
                        dslot.nbytes = dst_bytes;
                        mark_slot_upload_failed(dslot, err);
                        return true;
                    }
                    if (slot_matches_exact(sit->second)) {
                        CloudTpuTensorSlot copy = sit->second;
                        copy.offset = dst_off;
                        src_bctx->evict_overlapping_locked(dst_off, copy.nbytes, dst_off);
                        src_bctx->slots[dst_off] = std::move(copy);
                        return true;
                    }
                }
            } else {
                CloudTpuTensorSlot copy;
                bool ok = false;
                bool failed = false;
                std::string failed_err;
                {
                    std::lock_guard<std::mutex> slock(src_bctx->mu);
                    auto sit = src_bctx->slots.find(src_off);
                    if (sit != src_bctx->slots.end()) {
                        if (sit->second.upload_failed) {
                            failed = true;
                            failed_err = sit->second.upload_error;
                        } else if (slot_matches_exact(sit->second)) {
                            copy = sit->second;
                            ok = true;
                        }
                    }
                }
                if (failed) {
                    std::lock_guard<std::mutex> dlock(dst_bctx->mu);
                    if (!has_enclosing_parent_locked(dst_bctx, dst_off, dst_bytes)) {
                        dst_bctx->evict_overlapping_locked(dst_off, dst_bytes, dst_off);
                        CloudTpuTensorSlot& dslot = dst_bctx->slots[dst_off];
                        dslot.offset = dst_off;
                        dslot.nbytes = dst_bytes;
                        mark_slot_upload_failed(dslot, failed_err);
                        return true;
                    }
                }
                if (ok) {
                    std::lock_guard<std::mutex> dlock(dst_bctx->mu);
                    if (!has_enclosing_parent_locked(dst_bctx, dst_off, dst_bytes)) {
                        copy.offset = dst_off;
                        dst_bctx->evict_overlapping_locked(dst_off, copy.nbytes, dst_off);
                        dst_bctx->slots[dst_off] = std::move(copy);
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

static void ggml_backend_cloudtpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    std::lock_guard<std::mutex> lock(bctx->mu);
    bctx->clear_value = value;
    bctx->slots.clear();
}

static const struct ggml_backend_buffer_i ggml_backend_cloudtpu_buffer_i = {
    /* .free_buffer   = */ ggml_backend_cloudtpu_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_cloudtpu_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_cloudtpu_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_cloudtpu_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_cloudtpu_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_cloudtpu_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_cloudtpu_buffer_clear,
    /* .reset         = */ NULL,
    /* .set_tensor_n  = */ NULL,
    /* .get_tensor_n  = */ NULL,
    /* .cpy_tensor_n  = */ NULL,
};

struct cloudtpu_buft_context {
    int device_index = 0;
    int spmd_chips = 1;
    int spmd_min_rows = 256;
    bool q4k_bf16_scales = false;
    CloudTpuQ4Mode q4_mode = CloudTpuQ4Mode::I4;
    bool q4k_u4 = false;
    std::string name;
};

static const char* ggml_backend_cloudtpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto* btc = static_cast<cloudtpu_buft_context*>(buft->context);
    return btc->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cloudtpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t size) {
    auto* btc = static_cast<cloudtpu_buft_context*>(buft->context);
    try {
        ensure_runtime_initialized();

        const size_t aligned_span = std::max<size_t>(
            CLOUDTPU_BUFFER_ALIGNMENT,
            ((size + CLOUDTPU_BUFFER_ALIGNMENT - 1) & ~(CLOUDTPU_BUFFER_ALIGNMENT - 1)) + 4096);

        auto* bctx = new ggml_backend_cloudtpu_buffer_context();
        bctx->device_index = btc->device_index;
        bctx->spmd_chips = btc->spmd_chips;
        bctx->spmd_min_rows = btc->spmd_min_rows;
        bctx->q4k_bf16_scales = btc->q4k_bf16_scales;
        bctx->q4_mode = btc->q4_mode;
        bctx->q4k_u4 = btc->q4k_u4;
        bctx->virtual_base = g_next_virtual_base.fetch_add(aligned_span, std::memory_order_relaxed);
        bctx->size = size;

        return ggml_backend_buffer_init(buft, ggml_backend_cloudtpu_buffer_i, bctx, size);
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: alloc_buffer failed (size=%zu): %s\n", size, e.what());
        return NULL;
    }
}

static size_t ggml_backend_cloudtpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return CLOUDTPU_BUFFER_ALIGNMENT;
}

static size_t ggml_backend_cloudtpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               const struct ggml_tensor* tensor) {
    if (!tensor) {
        return 0;
    }
    auto* btc = buft ? static_cast<cloudtpu_buft_context*>(buft->context) : nullptr;
    const bool bf16_scales = (btc && btc->q4k_bf16_scales);
    size_t base_size = ggml_nbytes(tensor);
    const int64_t K = tensor->ne[0];
    if (K > 0 && (K % QK_K) == 0 && (tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q6_K)) {
        const size_t n_blocks = static_cast<size_t>(K / QK_K) *
                                static_cast<size_t>(std::max<int64_t>(1, tensor->ne[1])) *
                                static_cast<size_t>(std::max<int64_t>(1, tensor->ne[2])) *
                                static_cast<size_t>(std::max<int64_t>(1, tensor->ne[3]));
        // Split HBM representation expands Q4_K (128B u8 + 32B/16B d + 32B/16B m = 192B/160B/block)
        // and Q6_K (256B s8 + 64B/32B d = 320B/288B/block) relative to packed GGML blocks.
        const size_t bytes_per_block =
            (tensor->type == GGML_TYPE_Q4_K) ? (bf16_scales ? 160U : 192U) : (bf16_scales ? 288U : 320U);
        if (n_blocks <= SIZE_MAX / bytes_per_block) {
            base_size = std::max(base_size, n_blocks * bytes_per_block);
        }
    }
    // Under SPMD (P > 1), the unified device reports per-chip HBM (min_chip_hbm). Returning 1x
    // base_size here satisfies GGML's >= ggml_nbytes and TAG_ALLOC_SIZE_EXPAND invariants while
    // charging every replicated compute op (ADD, MUL, RMS_NORM, MUL_MAT, FLASH_ATTN_EXT, etc.)
    // its exact per-chip HBM footprint; sharded leaf weights are charged 1/P per chip in
    // get_alloc_size_n / alloc_buffer_n.
    return base_size;
}

static size_t cloudtpu_tensor_per_chip_alloc_size(ggml_backend_buffer_type_t buft,
                                                  const struct ggml_tensor* tensor) {
    size_t sz = ggml_backend_cloudtpu_buffer_type_get_alloc_size(buft, tensor);
    auto* btc = buft ? static_cast<cloudtpu_buft_context*>(buft->context) : nullptr;
    if (btc && btc->spmd_chips > 1 && tensor && tensor->op == GGML_OP_NONE && tensor->view_src == nullptr) {
        const cloudtpu::TensorShardingKind sh = choose_upload_sharding(
            tensor, btc->spmd_chips, btc->spmd_min_rows, /*allow_sharding=*/true, btc->q4_mode);
        if (sh != cloudtpu::TensorShardingKind::REPLICATED) {
            const size_t p = static_cast<size_t>(btc->spmd_chips);
            sz = (sz + p - 1) / p;
        }
    }
    return sz;
}

static size_t ggml_backend_cloudtpu_buffer_type_get_alloc_size_n(ggml_backend_buffer_type_t buft,
                                                                 struct ggml_tensor** tensors,
                                                                 int n_tensors) {
    const size_t align = CLOUDTPU_BUFFER_ALIGNMENT;
    size_t total = 0;
    for (int i = 0; i < n_tensors; ++i) {
        struct ggml_tensor* t = tensors[i];
        if (!t || t->data != nullptr || t->view_src != nullptr) {
            continue;
        }
        const size_t sz = cloudtpu_tensor_per_chip_alloc_size(buft, t);
        const size_t padded = (sz + align - 1) & ~(align - 1);
        total = (padded <= SIZE_MAX - total) ? (total + padded) : SIZE_MAX;
    }
    return total;
}

static ggml_backend_buffer_t ggml_backend_cloudtpu_buffer_type_alloc_buffer_n(
    ggml_backend_buffer_type_t buft,
    struct ggml_tensor** tensors,
    int n_tensors) {
    auto* btc = static_cast<cloudtpu_buft_context*>(buft->context);
    const size_t align = CLOUDTPU_BUFFER_ALIGNMENT;
    size_t virtual_span = 0;
    size_t per_chip_total = 0;
    bool has_unalloc_tensor = false;
    for (int i = 0; i < n_tensors; ++i) {
        struct ggml_tensor* t = tensors[i];
        if (!t || t->data != nullptr || t->view_src != nullptr) {
            continue;
        }
        has_unalloc_tensor = true;
        const size_t full_sz = ggml_backend_cloudtpu_buffer_type_get_alloc_size(buft, t);
        const size_t full_pad = (full_sz + align - 1) & ~(align - 1);
        const size_t chip_sz = cloudtpu_tensor_per_chip_alloc_size(buft, t);
        const size_t chip_pad = (chip_sz + align - 1) & ~(align - 1);
        if (full_pad > SIZE_MAX - virtual_span || chip_pad > SIZE_MAX - per_chip_total) {
            GGML_LOG_ERROR("ggml-cloudtpu: alloc_buffer_n size overflow\n");
            return NULL;
        }
        virtual_span += full_pad;
        per_chip_total += chip_pad;
    }
    if (!has_unalloc_tensor || virtual_span == 0) {
        return NULL;
    }
    try {
        ensure_runtime_initialized();
        const size_t aligned_span = std::max<size_t>(
            align,
            ((virtual_span + align - 1) & ~(align - 1)) + 4096);
        auto* bctx = new ggml_backend_cloudtpu_buffer_context();
        bctx->device_index = btc->device_index;
        bctx->spmd_chips = btc->spmd_chips;
        bctx->spmd_min_rows = btc->spmd_min_rows;
        bctx->q4k_bf16_scales = btc->q4k_bf16_scales;
        bctx->q4_mode = btc->q4_mode;
        bctx->q4k_u4 = btc->q4k_u4;
        bctx->virtual_base = g_next_virtual_base.fetch_add(aligned_span, std::memory_order_relaxed);
        bctx->size = per_chip_total;

        ggml_backend_buffer_t buffer =
            ggml_backend_buffer_init(buft, ggml_backend_cloudtpu_buffer_i, bctx, per_chip_total);
        if (!buffer) {
            delete bctx;
            return NULL;
        }
        // Assign strictly disjoint full-size virtual offsets so tensor_offset_in_buffer and
        // overlap tracking never collide, while buffer->size reports exact per-chip HBM.
        size_t cur_off = 0;
        for (int i = 0; i < n_tensors; ++i) {
            struct ggml_tensor* t = tensors[i];
            if (!t || t->data != nullptr || t->view_src != nullptr) {
                continue;
            }
            const size_t full_sz = ggml_backend_cloudtpu_buffer_type_get_alloc_size(buft, t);
            const size_t full_pad = (full_sz + align - 1) & ~(align - 1);
            t->buffer = buffer;
            t->data = reinterpret_cast<void*>(bctx->virtual_base + cur_off);
            cur_off += full_pad;
        }
        for (int i = 0; i < n_tensors; ++i) {
            struct ggml_tensor* t = tensors[i];
            if (!t || t->view_src == nullptr || t->buffer != nullptr) {
                continue;
            }
            if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(buffer);
                return NULL;
            }
        }
        return buffer;
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: alloc_buffer_n failed (per_chip=%zu): %s\n", per_chip_total, e.what());
        return NULL;
    }
}

static bool ggml_backend_cloudtpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static bool is_cloudtpu_buft(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_cloudtpu_buffer_type_get_name;
}

// =========================================================================
// Backend Stream Context & StableHLO Subgraph Dispatch
// =========================================================================

struct SubgraphSigKey {
    uint64_t hash = 0;
    std::vector<uint64_t> words;

    bool operator==(const SubgraphSigKey& o) const noexcept {
        return hash == o.hash && words == o.words;
    }
};

struct SubgraphSigHash {
    size_t operator()(const SubgraphSigKey& k) const noexcept {
        return static_cast<size_t>(k.hash);
    }
};

struct PlanInputBinding {
    uint32_t input_idx = 0;
    int part = 0;  // 0 = primary/qs, 1 = scales, 2 = mins
    bool donatable = false;
};

struct CachedSubgraphPlan {
    // Non-owning key into PjrtRuntime's shared LRU executable cache; dispatch acquires a live
    // CachedExecutablePtr via rt.get_cached_exec(exec_key) so evicted executables are never retained.
    std::string exec_key;
    size_t num_emitted_ops = 0;
    bool has_donation = false;
    std::vector<PlanInputBinding> input_bindings;
    std::vector<cloudtpu::PjrtDType> out_dtypes;
    std::vector<std::vector<int64_t>> out_dims;
};

struct CGraphLiveness {
    std::vector<int> first_consumer;   // smallest k > i that consumes node i (or n_nodes)
    std::vector<int> last_consumer;    // largest k > i that consumes node i (or -1)
    std::vector<int> first_overwrite;  // smallest k > i overlapping node i's HBM range (or n_nodes)
};

struct ResolvedTensorBuffers {
    cloudtpu::TensorShardingKind sharding = cloudtpu::TensorShardingKind::REPLICATED;
    std::vector<cloudtpu::PjrtBufferPtr> primary;
    std::vector<cloudtpu::PjrtBufferPtr> scales;
    std::vector<cloudtpu::PjrtBufferPtr> mins;
};

struct SigCacheEntry {
    uint32_t id = 0;
    uint64_t slot_epoch = 0;
    uint32_t root_leaf_idx = UINT32_MAX;
};

struct CloudSlotKey {
    ggml_backend_buffer_t buffer = nullptr;
    uintptr_t data_addr = 0;

    bool valid() const noexcept { return buffer != nullptr && data_addr != 0; }
    bool operator==(const CloudSlotKey& o) const noexcept {
        return buffer == o.buffer && data_addr == o.data_addr;
    }
};

struct CloudSlotKeyHash {
    size_t operator()(const CloudSlotKey& k) const noexcept {
        const size_t h1 = std::hash<const void*>{}(k.buffer);
        const size_t h2 = std::hash<uintptr_t>{}(k.data_addr);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct InSliceSlotSig {
    uint32_t id = 0;
    const struct ggml_tensor* root = nullptr;
    uint64_t epoch = 0;
};

struct ggml_backend_cloudtpu_context {
    int device_index = 0;
    int spmd_chips = 1;
    cloudtpu_env_config env;
    uint64_t total_graph_evals = 0;
    uint64_t total_subgraphs = 0;
    uint64_t total_ops_executed = 0;
    cloudtpu::PendingSyncLaunchQueue pending_sync_launches;
    uint64_t plan_hits = 0;
    uint64_t plan_misses = 0;
    uint64_t total_kv_donated = 0;
    uint64_t total_kv_donation_skipped = 0;
    uint64_t total_bucketed_subgraphs = 0;
    uint64_t mosaic_compiles = 0;
    uint64_t mosaic_fallbacks = 0;

    // Fast-path structural SubgraphPlanCache (skips MLIR emission & compile_or_get on decode hit)
    uint64_t plan_cache_epoch = 0;
    std::unordered_map<SubgraphSigKey, CachedSubgraphPlan, SubgraphSigHash> plan_cache;

    // Reusable per-call scratch storage (eliminates per-subgraph heap allocations on hot path)
    CGraphLiveness liveness;
    std::unordered_map<const struct ggml_tensor*, int> cgraph_node_idx;
    SubgraphSigKey scratch_key;
    std::unordered_map<const struct ggml_tensor*, SigCacheEntry> scratch_tensor_to_id;
    std::unordered_set<const struct ggml_tensor*> scratch_slice_computed;
    std::unordered_map<CloudSlotKey, InSliceSlotSig, CloudSlotKeyHash> scratch_slot_sig_state;
    std::vector<uint32_t> scratch_leaf_ssa_uses;
    std::vector<const struct ggml_tensor*> scratch_inputs;
    std::vector<cloudtpu::TensorShardingKind> scratch_input_sharding;
    std::vector<cloudtpu::PjrtDType> scratch_input_scale_dt;
    std::vector<Q4SlotPlan> scratch_input_q4_plans;
    std::vector<struct ggml_tensor*> scratch_outputs;
    std::vector<ResolvedTensorBuffers> scratch_resolved;
    std::vector<PJRT_Buffer*> scratch_raw_inputs;
    std::vector<int64_t> scratch_non_donatable;
    std::vector<std::vector<PJRT_Buffer*>> scratch_spmd_inputs;
    std::vector<PJRT_Buffer* const*> scratch_spmd_input_ptrs;
    std::vector<cloudtpu::PjrtBufferPtr> scratch_outputs_1d;
    std::vector<std::vector<cloudtpu::PjrtBufferPtr>> scratch_spmd_outputs;

    ~ggml_backend_cloudtpu_context() {
        if (env.verbose > 0) {
            cloudtpu::PjrtRuntime& rt = cloudtpu::shared_pjrt_runtime(nullptr);
            if (rt.is_initialized()) {
                const cloudtpu::CacheStats st = rt.cache_stats();
                GGML_LOG_INFO(
                    "ggml-cloudtpu: summary device=%d spmd_chips=%d mock=%d maps_libpython=%zu evals=%llu subgraphs=%llu ops=%llu "
                    "plan_hits=%llu plan_misses=%llu kv_donated=%llu kv_donation_skipped=%llu mosaic_compiles=%llu mosaic_fallbacks=%llu "
                    "cache_entries=%zu cache_hits=%llu cache_misses=%llu cache_evictions=%llu "
                    "compile_us=%llu execute_us=%llu bucketed=%llu\n",
                    device_index,
                    spmd_chips,
                    rt.is_mock() ? 1 : 0,
                    cloudtpu::count_libpython_mappings(),
                    static_cast<unsigned long long>(total_graph_evals),
                    static_cast<unsigned long long>(total_subgraphs),
                    static_cast<unsigned long long>(total_ops_executed),
                    static_cast<unsigned long long>(plan_hits),
                    static_cast<unsigned long long>(plan_misses),
                    static_cast<unsigned long long>(total_kv_donated),
                    static_cast<unsigned long long>(total_kv_donation_skipped),
                    static_cast<unsigned long long>(mosaic_compiles),
                    static_cast<unsigned long long>(mosaic_fallbacks),
                    st.entries,
                    static_cast<unsigned long long>(st.hits),
                    static_cast<unsigned long long>(st.misses),
                    static_cast<unsigned long long>(st.evictions),
                    static_cast<unsigned long long>(st.compile_us),
                    static_cast<unsigned long long>(st.execute_us),
                    static_cast<unsigned long long>(total_bucketed_subgraphs));
            }
        }
    }
};

// Cap materialized [B, H_Q, N_Q, N_KV] F32 score tensors on both host mock (32 MiB)
// and live Cloud TPU (256 MiB) so oversized long-context prefills fall back cleanly
// instead of exhausting host RAM or 16 GiB v5e HBM.
static constexpr uint64_t kMaxMockAttentionScoreElems = 8ULL * 1024ULL * 1024ULL;
static constexpr uint64_t kMaxLiveAttentionScoreElems = 64ULL * 1024ULL * 1024ULL;

static bool is_host_mock_active() {
    cloudtpu::PjrtRuntime& rt = cloudtpu::shared_pjrt_runtime(nullptr);
    if (rt.is_initialized()) {
        return rt.is_mock();
    }
    return load_env_config().mock_mode;
}

static bool is_view_noop_op(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool is_skipped_graph_node(const struct ggml_tensor* node) {
    return !node || is_view_noop_op(node->op) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
}

static bool ggml_backend_cloudtpu_device_supports_op_impl(const struct ggml_tensor* op,
                                                          bool allow_probe = false) {
    if (!op) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            if (!src0 || !src1 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (!is_supported_input_layout(src0) || !is_supported_input_layout(src1) ||
                !is_supported_output_layout(op)) {
                return false;
            }
            const bool src0_ok = is_dense_supported_type(src0->type) ||
                                 (is_quant_split_type(src0->type) &&
                                  (src0->ne[0] % quant_split_k_alignment(src0->type)) == 0);
            const bool src1_ok = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                                  src1->type == GGML_TYPE_BF16);
            if (!src0_ok || !src1_ok || src0->type == GGML_TYPE_I32) {
                return false;
            }
            const int64_t K = src0->ne[0];
            const int64_t N = src0->ne[1];
            const int64_t M = src1->ne[1];
            if (K <= 0 || N <= 0 || M <= 0 || src1->ne[0] != K) {
                return false;
            }
            if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                return false;
            }
            g_hot_env.ensure_initialized();
            const uint64_t min_elements = g_hot_env.min_elements.load(std::memory_order_relaxed);
            if (min_elements > 0) {
                const uint64_t work = static_cast<uint64_t>(M) * static_cast<uint64_t>(K) *
                                      static_cast<uint64_t>(N) * static_cast<uint64_t>(src1->ne[2]) *
                                      static_cast<uint64_t>(src1->ne[3]);
                if (work < min_elements) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_RMS_NORM: {
            const struct ggml_tensor* src0 = op->src[0];
            if (!src0 || op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            return is_supported_input_layout(src0) && is_supported_output_layout(op);
        }

        case GGML_OP_ROPE: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            const struct ggml_tensor* src2 = op->src[2];
            if (!src0 || !src1) {
                return false;
            }
            if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) ||
                op->type != src0->type || src1->type != GGML_TYPE_I32) {
                return false;
            }
            if (!is_supported_input_layout(src0) || !is_supported_input_layout(src1) ||
                !is_supported_output_layout(op)) {
                return false;
            }
            const int32_t n_dims = static_cast<const int32_t*>(op->op_params)[1];
            const int32_t mode = static_cast<const int32_t*>(op->op_params)[2];
            const int32_t n_offs = static_cast<const int32_t*>(op->op_params)[15];
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
                return false;
            }
            if (src1->ne[0] != src0->ne[2]) {
                return false;
            }
            // llama-model-loader probes rope_freqs.weight placement with a synthetic RoPE
            // op carrying n_dims == 0; accept that placement probe while requiring positive
            // even n_dims for executable graph nodes.
            if (allow_probe && n_dims == 0 && n_offs == 0 && mode == GGML_ROPE_TYPE_NORMAL &&
                src2 != nullptr && src2->type == GGML_TYPE_F32 && is_supported_input_layout(src2)) {
                return true;
            }
            if (n_dims <= 0 || (n_dims % 2) != 0 || n_offs < 0 || (n_offs % 2) != 0 ||
                n_offs + n_dims > src0->ne[0]) {
                return false;
            }
            if (src2 != nullptr &&
                (src2->type != GGML_TYPE_F32 || !is_supported_input_layout(src2) || src2->ne[0] < n_dims / 2)) {
                return false;
            }
            return true;
        }

        case GGML_OP_UNARY: {
            if (ggml_get_unary_op(op) != GGML_UNARY_OP_SILU) {
                return false;
            }
            const struct ggml_tensor* src0 = op->src[0];
            if (!src0 || op->type != src0->type) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
                return false;
            }
            return is_supported_input_layout(src0) && is_supported_output_layout(op);
        }

        case GGML_OP_GLU: {
            if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU) {
                return false;
            }
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            if (!src0 || op->type != src0->type) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
                return false;
            }
            if (!is_supported_input_layout(src0) || !is_supported_output_layout(op)) {
                return false;
            }
            if (src1) {
                if (src1->type != src0->type || !is_supported_input_layout(src1) ||
                    !ggml_are_same_shape(src0, src1)) {
                    return false;
                }
            } else if ((src0->ne[0] % 2) != 0) {
                return false;
            }
            return true;
        }

        case GGML_OP_SOFT_MAX: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            const struct ggml_tensor* src2 = op->src[2];
            if (!src0 || src2 != nullptr || op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            if (!is_supported_input_layout(src0) || !is_supported_output_layout(op)) {
                return false;
            }
            if (src1) {
                if ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                    !is_supported_input_layout(src1)) {
                    return false;
                }
                if (src1->ne[0] != src0->ne[0] || src1->ne[1] < src0->ne[1] ||
                    (src0->ne[2] % src1->ne[2]) != 0 || (src0->ne[3] % src1->ne[3]) != 0) {
                    return false;
                }
            }
            const uint64_t max_score_elems =
                is_host_mock_active() ? kMaxMockAttentionScoreElems : kMaxLiveAttentionScoreElems;
            if (static_cast<uint64_t>(ggml_nelements(op)) > max_score_elems) {
                return false;
            }
            return true;
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            const struct ggml_tensor* q = op->src[0];
            const struct ggml_tensor* k = op->src[1];
            const struct ggml_tensor* v = op->src[2];
            const struct ggml_tensor* mask = op->src[3];
            const struct ggml_tensor* sinks = op->src[4];
            const struct ggml_tensor* kv_rows = op->src[5];
            if (!q || !k || !v || sinks != nullptr || kv_rows != nullptr) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32) {
                return false;
            }
            if ((k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16) ||
                (v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16)) {
                return false;
            }
            if (!is_supported_input_layout(q) || !is_supported_input_layout(k) ||
                !is_supported_input_layout(v) || !is_supported_output_layout(op)) {
                return false;
            }
            if (q->ne[0] != k->ne[0] || k->ne[1] != v->ne[1] ||
                k->ne[2] != v->ne[2] || (q->ne[2] % k->ne[2]) != 0 ||
                q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
                return false;
            }
            if (mask) {
                if ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) ||
                    !is_supported_input_layout(mask)) {
                    return false;
                }
                if (mask->ne[0] != k->ne[1] || mask->ne[1] < q->ne[1] ||
                    (q->ne[2] % mask->ne[2]) != 0 || (q->ne[3] % mask->ne[3]) != 0) {
                    return false;
                }
            }
            {
                const uint64_t max_score_elems =
                    is_host_mock_active() ? kMaxMockAttentionScoreElems : kMaxLiveAttentionScoreElems;
                const uint64_t b = static_cast<uint64_t>(q->ne[3]);
                const uint64_t hq = static_cast<uint64_t>(q->ne[2]);
                const uint64_t nq = static_cast<uint64_t>(q->ne[1]);
                const uint64_t nkv = static_cast<uint64_t>(k->ne[1]);
                if (b == 0 || hq == 0 || nq == 0 || nkv == 0 ||
                    b > max_score_elems ||
                    hq > max_score_elems / b ||
                    nq > max_score_elems / (b * hq) ||
                    nkv > max_score_elems / (b * hq * nq)) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_GET_ROWS: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            if (!src0 || !src1 || src1->type != GGML_TYPE_I32 || src1->ne[3] != 1) {
                return false;
            }
            if (!is_supported_input_layout(src0) || !is_supported_input_layout(src1) ||
                !is_supported_output_layout(op)) {
                return false;
            }
            if (src0->ne[2] != src1->ne[1] || src0->ne[3] != src1->ne[2]) {
                return false;
            }
            const bool src0_ok = is_dense_supported_type(src0->type) ||
                                 (is_quant_split_type(src0->type) &&
                                  (src0->ne[0] % quant_split_k_alignment(src0->type)) == 0);
            if (!src0_ok) {
                return false;
            }
            if (src0->type == GGML_TYPE_I32) {
                return op->type == GGML_TYPE_I32;
            }
            return op->type == GGML_TYPE_F32;
        }

        case GGML_OP_SET_ROWS: {
            const struct ggml_tensor* src_upd = op->src[0];
            const struct ggml_tensor* src_idx = op->src[1];
            const struct ggml_tensor* src_tbl = op->src[2];
            if (!src_upd || !src_idx || !src_tbl) {
                return false;
            }
            if ((src_upd->type != GGML_TYPE_F32 && src_upd->type != GGML_TYPE_F16 && src_upd->type != GGML_TYPE_BF16) ||
                (src_idx->type != GGML_TYPE_I32 && src_idx->type != GGML_TYPE_I64) ||
                (src_tbl->type != GGML_TYPE_F32 && src_tbl->type != GGML_TYPE_F16 && src_tbl->type != GGML_TYPE_BF16) ||
                op->type != src_tbl->type) {
                return false;
            }
            if (!is_supported_input_layout(src_upd) || !is_supported_output_layout(src_idx) ||
                !is_supported_output_layout(src_tbl) || !is_supported_output_layout(op)) {
                return false;
            }
            if (src_tbl->ne[0] != src_upd->ne[0] || src_tbl->ne[2] != src_upd->ne[2] ||
                src_tbl->ne[3] != src_upd->ne[3] || src_upd->ne[1] != src_idx->ne[0] ||
                (src_upd->ne[2] % src_idx->ne[1]) != 0 || (src_upd->ne[3] % src_idx->ne[2]) != 0 ||
                src_idx->ne[3] != 1) {
                return false;
            }
            return true;
        }

        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            if (!src0 || !src1) {
                return false;
            }
            if (src0->type != op->type || src1->type != op->type) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) {
                return false;
            }
            if (!is_supported_input_layout(src0) || !is_supported_input_layout(src1) ||
                !is_supported_output_layout(op)) {
                return false;
            }
            for (int d = 0; d < 4; ++d) {
                if (src1->ne[d] <= 0 || (src0->ne[d] % src1->ne[d]) != 0 || op->ne[d] != src0->ne[d]) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_SCALE: {
            const struct ggml_tensor* src0 = op->src[0];
            if (!src0 || op->type != src0->type) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) {
                return false;
            }
            return is_supported_input_layout(src0) && is_supported_output_layout(op);
        }

        case GGML_OP_CPY:
        case GGML_OP_CONT: {
            const struct ggml_tensor* src0 = op->src[0];
            if (!src0 || !is_dense_supported_type(src0->type) || !is_dense_supported_type(op->type)) {
                return false;
            }
            if (!is_supported_input_layout(src0) || !is_supported_output_layout(op)) {
                return false;
            }
            return ggml_nelements(src0) == ggml_nelements(op);
        }

        default:
            return false;
    }
}

static cloudtpu::TensorSpec tensor_to_spec(const struct ggml_tensor* t,
                                           uint64_t id,
                                           cloudtpu::TensorShardingKind sharding = cloudtpu::TensorShardingKind::REPLICATED,
                                           const Q4SlotPlan* q4_plan = nullptr) {
    cloudtpu::TensorSpec ts;
    ts.id = id;
    ts.dims[0] = t->ne[3];
    ts.dims[1] = t->ne[2];
    ts.dims[2] = t->ne[1];
    ts.dims[3] = t->ne[0];
    ts.sharding = sharding;
    if (t->type == GGML_TYPE_Q8_0) {
        ts.storage = cloudtpu::TensorStorageKind::Q8_0_SPLIT;
        ts.dtype = cloudtpu::PjrtDType::S8;
    } else if (t->type == GGML_TYPE_Q4_0) {
        if (q4_plan && q4_plan->storage != cloudtpu::TensorStorageKind::DENSE) {
            ts.storage = q4_plan->storage;
            ts.dtype = q4_plan->q_dtype;
            ts.bitcast_target = q4_plan->bitcast_target;
        } else {
            ts.storage = cloudtpu::TensorStorageKind::Q4_0_SPLIT;
            ts.dtype = cloudtpu::PjrtDType::U8;
        }
    } else if (t->type == GGML_TYPE_Q4_K) {
        if (q4_plan && q4_plan->storage != cloudtpu::TensorStorageKind::DENSE) {
            ts.storage = q4_plan->storage;
            ts.dtype = q4_plan->q_dtype;
            ts.bitcast_target = q4_plan->bitcast_target;
        } else {
            ts.storage = cloudtpu::TensorStorageKind::Q4_K_SPLIT;
            ts.dtype = cloudtpu::PjrtDType::U8;
        }
    } else if (t->type == GGML_TYPE_Q6_K) {
        ts.storage = cloudtpu::TensorStorageKind::Q6_K_SPLIT;
        ts.dtype = cloudtpu::PjrtDType::S8;
    } else {
        ts.storage = cloudtpu::TensorStorageKind::DENSE;
        ts.dtype = ggml_type_to_pjrt(t->type);
    }
    return ts;
}

static cloudtpu::TensorShardingKind lookup_tensor_sharding(const struct ggml_tensor* t, int spmd_chips) {
    if (spmd_chips <= 1 || !t || !t->buffer || !is_cloudtpu_buffer(t->buffer)) {
        return cloudtpu::TensorShardingKind::REPLICATED;
    }
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
    const size_t t_off = tensor_offset_in_buffer(bctx, t);
    auto slot_matches_tensor = [](const CloudTpuTensorSlot& s, const struct ggml_tensor* tensor) -> bool {
        return s.ggml_dt == tensor->type &&
               s.dims[0] == tensor->ne[3] && s.dims[1] == tensor->ne[2] &&
               s.dims[2] == tensor->ne[1] && s.dims[3] == tensor->ne[0];
    };
    std::lock_guard<std::mutex> lock(bctx->mu);
    auto it = bctx->slots.find(t_off);
    if (it != bctx->slots.end() && slot_matches_tensor(it->second, t)) {
        // Slot records its canonical shard layout at staging/upload time so all matching
        // same-shape aliases compile and upload with one layout.
        return it->second.sharding;
    }
    if (t->view_src != nullptr && t->view_offs == 0 && ggml_is_contiguous(t)) {
        const struct ggml_tensor* base = unwrap_view_src(t);
        const size_t base_off = tensor_offset_in_buffer(bctx, base);
        auto bit = bctx->slots.find(base_off);
        if (bit != bctx->slots.end() && slot_matches_tensor(bit->second, t)) {
            return bit->second.sharding;
        }
    }
    return cloudtpu::TensorShardingKind::REPLICATED;
}

static Q4SlotPlan lookup_tensor_q4_plan(const struct ggml_tensor* t,
                                        int spmd_chips,
                                        cloudtpu::TensorShardingKind sh,
                                        CloudTpuQ4Mode default_q4_mode,
                                        bool default_q4k_u4) {
    if (!t || (t->type != GGML_TYPE_Q4_0 && t->type != GGML_TYPE_Q4_K)) {
        return {};
    }
    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        const size_t t_off = tensor_offset_in_buffer(bctx, t);
        auto slot_matches_q4_tensor = [](const CloudTpuTensorSlot& s, const struct ggml_tensor* tensor) -> bool {
            return s.has_hbm() && !s.staging_dirty &&
                   s.storage != cloudtpu::TensorStorageKind::DENSE &&
                   s.ggml_dt == tensor->type &&
                   s.dims[0] == tensor->ne[3] && s.dims[1] == tensor->ne[2] &&
                   s.dims[2] == tensor->ne[1] && s.dims[3] == tensor->ne[0];
        };
        std::lock_guard<std::mutex> lock(bctx->mu);
        auto it = bctx->slots.find(t_off);
        if (it != bctx->slots.end()) {
            if (slot_matches_q4_tensor(it->second, t)) {
                Q4SlotPlan plan;
                plan.storage = it->second.storage;
                plan.q_dtype = it->second.dtype;
                plan.bitcast_target = it->second.bitcast_target;
                return plan;
            }
            return resolve_q4_slot_plan(
                t->type, t->ne[0], t->ne[1], bctx->spmd_chips, sh, bctx->q4_mode, bctx->q4k_u4);
        }
        if (t->view_src != nullptr && t->view_offs == 0 && ggml_is_contiguous(t)) {
            const struct ggml_tensor* base = unwrap_view_src(t);
            const size_t base_off = tensor_offset_in_buffer(bctx, base);
            auto bit = bctx->slots.find(base_off);
            if (bit != bctx->slots.end() && slot_matches_q4_tensor(bit->second, t)) {
                Q4SlotPlan plan;
                plan.storage = bit->second.storage;
                plan.q_dtype = bit->second.dtype;
                plan.bitcast_target = bit->second.bitcast_target;
                return plan;
            }
        }
        return resolve_q4_slot_plan(
            t->type, t->ne[0], t->ne[1], bctx->spmd_chips, sh, bctx->q4_mode, bctx->q4k_u4);
    }
    return resolve_q4_slot_plan(t->type, t->ne[0], t->ne[1], spmd_chips, sh, default_q4_mode, default_q4k_u4);
}

static cloudtpu::PjrtDType lookup_tensor_scale_dtype(const struct ggml_tensor* t, bool default_q4k_bf16) {
    if (!t || !is_quant_split_type(t->type)) {
        return cloudtpu::PjrtDType::INVALID;
    }
    if (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) {
        return cloudtpu::PjrtDType::F16;
    }
    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        const size_t t_off = tensor_offset_in_buffer(bctx, t);
        std::lock_guard<std::mutex> lock(bctx->mu);
        auto it = bctx->slots.find(t_off);
        if (it != bctx->slots.end() && it->second.scale_dtype != cloudtpu::PjrtDType::INVALID) {
            return it->second.scale_dtype;
        }
        return bctx->q4k_bf16_scales ? cloudtpu::PjrtDType::BF16 : cloudtpu::PjrtDType::F32;
    }
    return default_q4k_bf16 ? cloudtpu::PjrtDType::BF16 : cloudtpu::PjrtDType::F32;
}

static const struct ggml_tensor* canonical_root(const struct ggml_tensor* t) {
    if (t && t->view_src != nullptr && t->view_offs == 0) {
        const struct ggml_tensor* root = unwrap_view_src(t);
        if (root && ggml_nelements(t) == ggml_nelements(root) && t->type == root->type) {
            return root;
        }
    }
    return t;
}

// Populates `out` from an existing HBM slot, slicing back to logical shape if `slot` was stored padded.
static void resolve_unpadded_slot_buffers(cloudtpu::PjrtRuntime& rt,
                                          int device_index,
                                          int spmd_chips,
                                          const CloudTpuTensorSlot& slot,
                                          ResolvedTensorBuffers& out) {
    out.sharding = slot.sharding;
    if (slot.hbm_token_axis >= 0 && slot.hbm_token_axis < 4 &&
        slot.hbm_dims[slot.hbm_token_axis] > slot.dims[slot.hbm_token_axis]) {
        std::vector<uint8_t> raw_host(slot.nbytes);
        download_slot_to_host(rt, slot, raw_host.data());
        out.primary = upload_sharded_part(
            rt, device_index, spmd_chips, slot.sharding,
            raw_host.data(), slot.dtype,
            {slot.dims[0], slot.dims[1], slot.dims[2], slot.dims[3]});
    } else {
        out.primary = slot.hbm_primary;
    }
    out.scales = slot.hbm_scales;
    out.mins = slot.hbm_mins;
}

// Resolves a GGML tensor to HBM buffers (fetching from CloudTpuBuffer or uploading transient host tensor).
static void resolve_tensor_hbm(cloudtpu::PjrtRuntime& rt,
                               int device_index,
                               int spmd_chips,
                               bool q4k_bf16_scales,
                               CloudTpuQ4Mode q4_mode,
                               bool q4k_u4,
                               const struct ggml_tensor* t,
                               ResolvedTensorBuffers& out) {
    // Standalone non-contiguous dense tensors (e.g. scheduler split copies of CPU strided views)
    // are stored as raw strided bytes and gathered into canonical dense order before dispatch.
    if (t->view_src == nullptr && !ggml_is_contiguous(t) && is_valid_strided_dense_layout(t)) {
        const size_t raw_bytes = ggml_nbytes(t);
        std::vector<uint8_t> dense_bytes;
        if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
            auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
            const size_t t_off = tensor_offset_in_buffer(bctx, t);
            std::vector<uint8_t> raw_host;
            bool found = false;
            uint8_t fill_byte = 0;
            {
                std::lock_guard<std::mutex> lock(bctx->mu);
                fill_byte = bctx->clear_value;
                auto it = bctx->slots.find(t_off);
                if (it != bctx->slots.end() && it->second.nbytes >= raw_bytes) {
                    const CloudTpuTensorSlot& slot = it->second;
                    if (slot.upload_failed) {
                        throw CloudTpuAllocError("ggml-cloudtpu: prior HBM upload failed for strided tensor '" +
                                                 std::string(t->name) + "': " + slot.upload_error);
                    }
                    raw_host.resize(slot.nbytes);
                    if (slot.staging_dirty && !slot.staging.empty()) {
                        std::memcpy(raw_host.data(), slot.staging.data(), slot.nbytes);
                        found = true;
                    } else if (slot.has_hbm()) {
                        download_slot_to_host(rt, slot, raw_host.data());
                        found = true;
                    }
                }
                if (!found) {
                    const size_t t_end = t_off + raw_bytes;
                    for (const auto& kv : bctx->slots) {
                        const CloudTpuTensorSlot& s = kv.second;
                        if (s.upload_failed && s.offset < t_end && t_off < s.offset + s.nbytes) {
                            throw CloudTpuAllocError("ggml-cloudtpu: prior HBM upload failed for strided tensor '" +
                                                     std::string(t->name) + "': " + s.upload_error);
                        }
                    }
                }
            }
            if (!found) {
                if (t->op == GGML_OP_NONE) {
                    raw_host.assign(raw_bytes, fill_byte);
                } else {
                    throw std::runtime_error("ggml-cloudtpu: missing HBM slot for strided input tensor '" +
                                             std::string(t->name) + "'");
                }
            }
            gather_strided_dense_bytes(t, raw_host.data(), raw_host.size(), dense_bytes);
        } else {
            if (!t->data) {
                throw std::runtime_error("ggml-cloudtpu: null host data pointer for strided tensor '" +
                                         std::string(t->name) + "'");
            }
            gather_strided_dense_bytes(
                t, static_cast<const uint8_t*>(t->data), raw_bytes, dense_bytes);
        }
        const cloudtpu::PjrtDType pdt = ggml_type_to_pjrt(t->type);
        out.sharding = cloudtpu::TensorShardingKind::REPLICATED;
        out.primary = upload_sharded_part(
            rt, device_index, spmd_chips, cloudtpu::TensorShardingKind::REPLICATED,
            dense_bytes.data(), pdt, {t->ne[3], t->ne[2], t->ne[1], t->ne[0]});
        out.scales.clear();
        out.mins.clear();
        return;
    }

    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        const size_t t_off = tensor_offset_in_buffer(bctx, t);
        std::lock_guard<std::mutex> lock(bctx->mu);
        auto it = bctx->slots.find(t_off);
        if (it != bctx->slots.end()) {
            CloudTpuTensorSlot& slot = it->second;
            if (slot.upload_failed) {
                throw CloudTpuAllocError("ggml-cloudtpu: prior HBM upload failed for tensor '" +
                                         std::string(t->name) + "': " + slot.upload_error);
            }
            const struct ggml_tensor* upload_t =
                (t->view_src != nullptr && t->view_offs == 0 && ggml_is_contiguous(t))
                    ? unwrap_view_src(t)
                    : t;
            if (slot.staging_dirty && !slot.staging.empty() && slot.nbytes == ggml_nbytes(upload_t)) {
                CloudTpuTensorSlot next;
                next.offset = slot.offset;
                next.nbytes = slot.nbytes;
                const cloudtpu::TensorShardingKind slot_sh = slot.sharding;
                upload_raw_tensor_to_slot(
                    rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                    next, upload_t, slot.staging.data(), /*allow_sharding=*/true, &slot_sh,
                    bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
                slot = std::move(next);
            }
            if (slot.has_hbm() && slot.ggml_dt == t->type &&
                slot.dims[0] == t->ne[3] && slot.dims[1] == t->ne[2] &&
                slot.dims[2] == t->ne[1] && slot.dims[3] == t->ne[0]) {
                resolve_unpadded_slot_buffers(rt, bctx->device_index, bctx->spmd_chips, slot, out);
                return;
            }
        }
        // Check if tensor is a contiguous view aliasing a parent slot at offset 0 with matching
        // type and exact shape (split quantized PJRT buffers must never be reused across reshapes).
        if (t->view_src != nullptr && t->view_offs == 0 && ggml_is_contiguous(t)) {
            const struct ggml_tensor* base = unwrap_view_src(t);
            const size_t base_off = tensor_offset_in_buffer(bctx, base);
            auto bit = bctx->slots.find(base_off);
            if (bit != bctx->slots.end()) {
                if (bit->second.upload_failed) {
                    throw CloudTpuAllocError("ggml-cloudtpu: prior HBM upload failed for tensor '" +
                                             std::string(t->name) + "': " + bit->second.upload_error);
                }
                if (bit->second.staging_dirty && !bit->second.staging.empty()) {
                    CloudTpuTensorSlot next;
                    next.offset = bit->second.offset;
                    next.nbytes = bit->second.nbytes;
                    const cloudtpu::TensorShardingKind slot_sh = bit->second.sharding;
                    upload_raw_tensor_to_slot(
                        rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                        next, base, bit->second.staging.data(), /*allow_sharding=*/true, &slot_sh,
                        bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
                    bit->second = std::move(next);
                }
                if (bit->second.has_hbm() && bit->second.ggml_dt == t->type &&
                    bit->second.dims[0] == t->ne[3] && bit->second.dims[1] == t->ne[2] &&
                    bit->second.dims[2] == t->ne[1] && bit->second.dims[3] == t->ne[0]) {
                    resolve_unpadded_slot_buffers(rt, bctx->device_index, bctx->spmd_chips, bit->second, out);
                    return;
                }
            }
        }
        {
            const size_t t_end = t_off + ggml_nbytes(t);
            for (const auto& kv : bctx->slots) {
                const CloudTpuTensorSlot& s = kv.second;
                if (s.upload_failed && s.offset < t_end && t_off < s.offset + s.nbytes) {
                    throw CloudTpuAllocError("ggml-cloudtpu: prior HBM upload failed for tensor '" +
                                             std::string(t->name) + "': " + s.upload_error);
                }
            }
        }
        // Extract a contiguous sub-range from an existing live parent slot (e.g. a 3-row Q4_K
        // alias over a 4-row BITCAST Q4_K slot) without evicting or zeroing the parent weights.
        const cloudtpu::PjrtDType pdt = ggml_type_to_pjrt(t->type);
        const bool quant_ok = is_quant_split_type(t->type) &&
                              (t->ne[0] % quant_split_k_alignment(t->type)) == 0;
        if (t->op == GGML_OP_NONE && (pdt != cloudtpu::PjrtDType::INVALID || quant_ok) && ggml_is_contiguous(t)) {
            const size_t t_bytes = ggml_nbytes(t);
            for (const auto& kv : bctx->slots) {
                const CloudTpuTensorSlot& parent = kv.second;
                if (t_bytes <= parent.nbytes && t_off >= parent.offset &&
                    (t_off - parent.offset) <= parent.nbytes - t_bytes) {
                    std::vector<uint8_t> slice_bytes;
                    const size_t rel = t_off - parent.offset;
                    if (parent.staging_dirty && !parent.staging.empty() && parent.staging.size() >= rel + t_bytes) {
                        slice_bytes.assign(parent.staging.begin() + static_cast<std::ptrdiff_t>(rel),
                                           parent.staging.begin() + static_cast<std::ptrdiff_t>(rel + t_bytes));
                    } else if (parent.has_hbm()) {
                        std::vector<uint8_t> parent_host(parent.nbytes);
                        download_slot_to_host(rt, parent, parent_host.data());
                        slice_bytes.assign(parent_host.begin() + static_cast<std::ptrdiff_t>(rel),
                                           parent_host.begin() + static_cast<std::ptrdiff_t>(rel + t_bytes));
                    }
                    if (slice_bytes.size() == t_bytes) {
                        CloudTpuTensorSlot tmp;
                        tmp.offset = t_off;
                        tmp.nbytes = t_bytes;
                        upload_raw_tensor_to_slot(
                            rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                            tmp, t, slice_bytes.data(), /*allow_sharding=*/false, nullptr,
                            bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
                        out.sharding = tmp.sharding;
                        out.primary = std::move(tmp.hbm_primary);
                        out.scales = std::move(tmp.hbm_scales);
                        out.mins = std::move(tmp.hbm_mins);
                        return;
                    }
                }
            }
            // Uninitialized fill-byte HBM buffer for newly allocated leaf/KV-cache buffers before first write.
            // Never silently materialize a computed graph activation node (t->op != GGML_OP_NONE).
            std::vector<uint8_t> init_bytes(t_bytes, bctx->clear_value);
            CloudTpuTensorSlot next;
            next.offset = t_off;
            next.nbytes = t_bytes;
            upload_raw_tensor_to_slot(
                rt, bctx->device_index, bctx->spmd_chips, bctx->spmd_min_rows,
                next, t, init_bytes.data(), /*allow_sharding=*/false, nullptr,
                bctx->q4k_bf16_scales, bctx->q4_mode, bctx->q4k_u4);
            bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
            CloudTpuTensorSlot& slot = (bctx->slots[t_off] = std::move(next));
            out.sharding = slot.sharding;
            out.primary = slot.hbm_primary;
            out.scales = slot.hbm_scales;
            out.mins = slot.hbm_mins;
            return;
        }
        throw std::runtime_error("ggml-cloudtpu: missing HBM slot for input tensor '" +
                                 std::string(t->name) + "' (op=" + std::string(ggml_op_desc(t)) + ")");
    }

    // Host tensor fallback: upload transient replicated copy to device HBM.
    if (!t->data) {
        throw std::runtime_error("ggml-cloudtpu: null host data pointer for tensor '" +
                                 std::string(t->name) + "'");
    }
    CloudTpuTensorSlot tmp_slot;
    tmp_slot.nbytes = ggml_nbytes(t);
    upload_raw_tensor_to_slot(
        rt, device_index, spmd_chips, 0, tmp_slot, t, t->data, false, nullptr,
        q4k_bf16_scales, q4_mode, q4k_u4);
    out.sharding = tmp_slot.sharding;
    out.primary = std::move(tmp_slot.hbm_primary);
    out.scales = std::move(tmp_slot.hbm_scales);
    out.mins = std::move(tmp_slot.hbm_mins);
}

// Pads a token-varying dense input along `token_axis` from T to `bucket_tokens` (B) with
// zero-corruption sentinel values (-1 for SET_ROWS scatter indices, 0 for GET_ROWS/ROPE/mask/dense).
static void upload_padded_token_input(cloudtpu::PjrtRuntime& rt,
                                      int device_index,
                                      int spmd_chips,
                                      const struct ggml_tensor* t,
                                      int token_axis,
                                      int64_t bucket_tokens,
                                      cloudtpu::TokenInputPadKind pad_kind,
                                      ResolvedTensorBuffers& out) {
    const cloudtpu::PjrtDType pdt = ggml_type_to_pjrt(t->type);
    if (pdt == cloudtpu::PjrtDType::INVALID || !ggml_is_contiguous(t) ||
        token_axis < 0 || token_axis >= 4) {
        throw std::runtime_error("ggml-cloudtpu: unsupported padded token input '" + std::string(t->name) + "'");
    }
    const int64_t orig_dims[4] = {t->ne[3], t->ne[2], t->ne[1], t->ne[0]};
    if (bucket_tokens < orig_dims[token_axis] || bucket_tokens > cloudtpu::kMaxBatchBucketTokens) {
        throw std::overflow_error("ggml-cloudtpu: invalid or oversized padded bucket for '" +
                                  std::string(t->name) + "'");
    }
    int64_t padded_dims[4] = {orig_dims[0], orig_dims[1], orig_dims[2], orig_dims[3]};
    padded_dims[token_axis] = bucket_tokens;
    const std::vector<int64_t> padded_vec = {padded_dims[0], padded_dims[1], padded_dims[2], padded_dims[3]};
    const size_t t_bytes = ggml_nbytes(t);
    size_t padded_bytes = 0;
    if (!cloudtpu::checked_tensor_nbytes(padded_dims, cloudtpu::pjrt_dtype_size(pdt), &padded_bytes) ||
        padded_bytes < t_bytes) {
        throw std::overflow_error("ggml-cloudtpu: padded host buffer byte size overflow for '" +
                                  std::string(t->name) + "'");
    }

    out.sharding = cloudtpu::TensorShardingKind::REPLICATED;
    out.scales.clear();
    out.mins.clear();

    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        const size_t t_off = tensor_offset_in_buffer(bctx, t);
        std::vector<uint8_t> raw_host;
        bool have_raw = false;
        bool fill_init = false;
        uint8_t fill_byte = 0;
        {
            std::lock_guard<std::mutex> lock(bctx->mu);
            fill_byte = bctx->clear_value;
            auto it = bctx->slots.find(t_off);
            const CloudTpuTensorSlot* slot_ptr = (it != bctx->slots.end()) ? &it->second : nullptr;
            if (!slot_ptr && t->view_src != nullptr && t->view_offs == 0 && ggml_is_contiguous(t)) {
                const struct ggml_tensor* base = unwrap_view_src(t);
                const size_t base_off = tensor_offset_in_buffer(bctx, base);
                auto bit = bctx->slots.find(base_off);
                if (bit != bctx->slots.end()) {
                    slot_ptr = &bit->second;
                }
            }
            if (slot_ptr) {
                const CloudTpuTensorSlot& slot = *slot_ptr;
                if (slot.upload_failed) {
                    throw CloudTpuAllocError("ggml-cloudtpu: prior HBM upload failed for padded tensor '" +
                                             std::string(t->name) + "': " + slot.upload_error);
                }
                if (!slot.staging_dirty && slot.has_hbm() && slot.ggml_dt == t->type &&
                    slot.hbm_token_axis == token_axis &&
                    slot.hbm_dims[0] == padded_dims[0] && slot.hbm_dims[1] == padded_dims[1] &&
                    slot.hbm_dims[2] == padded_dims[2] && slot.hbm_dims[3] == padded_dims[3] &&
                    slot.sharding == cloudtpu::TensorShardingKind::REPLICATED &&
                    pad_kind != cloudtpu::TokenInputPadKind::NEG_ONE) {
                    out.sharding = slot.sharding;
                    out.primary = slot.hbm_primary;
                    return;
                }
                if (slot.staging_dirty && !slot.staging.empty() && slot.nbytes == t_bytes) {
                    raw_host.assign(slot.staging.begin(), slot.staging.end());
                    have_raw = true;
                } else if (slot.has_hbm() && slot.ggml_dt == t->type &&
                           slot.dims[0] == orig_dims[0] && slot.dims[1] == orig_dims[1] &&
                           slot.dims[2] == orig_dims[2] && slot.dims[3] == orig_dims[3]) {
                    raw_host.resize(t_bytes);
                    download_slot_to_host(rt, slot, raw_host.data());
                    have_raw = true;
                }
            }
            if (!have_raw && t->view_src != nullptr) {
                for (const auto& kv : bctx->slots) {
                    const CloudTpuTensorSlot& parent = kv.second;
                    if (t_off >= parent.offset && t_off + t_bytes <= parent.offset + parent.nbytes) {
                        if (parent.upload_failed) {
                            throw CloudTpuAllocError("ggml-cloudtpu: prior HBM upload failed for padded parent '" +
                                                     std::string(t->name) + "': " + parent.upload_error);
                        }
                        std::vector<uint8_t> parent_host(parent.nbytes);
                        download_slot_to_host(rt, parent, parent_host.data());
                        const size_t rel = t_off - parent.offset;
                        raw_host.resize(t_bytes);
                        std::memcpy(raw_host.data(), parent_host.data() + rel, t_bytes);
                        have_raw = true;
                        break;
                    }
                }
            }
            if (!have_raw) {
                const size_t t_end = t_off + t_bytes;
                for (const auto& kv : bctx->slots) {
                    const CloudTpuTensorSlot& s = kv.second;
                    if (s.upload_failed && s.offset < t_end && t_off < s.offset + s.nbytes) {
                        throw CloudTpuAllocError("ggml-cloudtpu: prior HBM upload failed for padded tensor '" +
                                                 std::string(t->name) + "': " + s.upload_error);
                    }
                }
            }
            if (!have_raw && t->op == GGML_OP_NONE) {
                fill_init = true;
            }
        }
        if (!have_raw && !fill_init) {
            throw std::runtime_error("ggml-cloudtpu: missing HBM slot for padded input tensor '" +
                                     std::string(t->name) + "' (op=" + std::string(ggml_op_desc(t)) + ")");
        }
        if (!have_raw && fill_byte != 0) {
            raw_host.assign(t_bytes, fill_byte);
            have_raw = true;
        }
        std::vector<uint8_t> padded_host(padded_bytes, 0);
        cloudtpu::pad_dense_host_tensor(
            have_raw ? raw_host.data() : nullptr,
            padded_host.data(),
            pdt,
            orig_dims,
            padded_dims,
            token_axis,
            pad_kind);
        out.primary = upload_sharded_part(
            rt, device_index, spmd_chips, cloudtpu::TensorShardingKind::REPLICATED,
            padded_host.data(), pdt, padded_vec);
        return;
    }

    if (!t->data) {
        throw std::runtime_error("ggml-cloudtpu: null host data pointer for padded tensor '" +
                                 std::string(t->name) + "'");
    }
    std::vector<uint8_t> padded_host(padded_bytes, 0);
    cloudtpu::pad_dense_host_tensor(
        t->data, padded_host.data(), pdt, orig_dims, padded_dims, token_axis, pad_kind);
    out.primary = upload_sharded_part(
        rt, device_index, spmd_chips, cloudtpu::TensorShardingKind::REPLICATED,
        padded_host.data(), pdt, padded_vec);
}

// Records logical and physical (possibly bucket-padded) HBM shapes on `slot`.
static void set_slot_hbm_dims(CloudTpuTensorSlot& slot,
                              const struct ggml_tensor* target,
                              const cloudtpu::PjrtBufferPtr& buf,
                              int out_token_axis) {
    slot.dims[0] = target->ne[3];
    slot.dims[1] = target->ne[2];
    slot.dims[2] = target->ne[1];
    slot.dims[3] = target->ne[0];
    slot.hbm_token_axis = -1;
    for (int d = 0; d < 4; ++d) {
        slot.hbm_dims[d] = slot.dims[d];
    }
    if (out_token_axis >= 0 && out_token_axis < 4 && buf && buf->dims.size() == 4 &&
        buf->dims[static_cast<size_t>(out_token_axis)] > slot.dims[out_token_axis]) {
        slot.hbm_token_axis = out_token_axis;
        for (int d = 0; d < 4; ++d) {
            slot.hbm_dims[d] = buf->dims[static_cast<size_t>(d)];
        }
    }
}

// Copies a PJRT output buffer to a host destination buffer, slicing `[0..T)` if bucket-padded.
static void copy_output_buffer_to_host_ptr(cloudtpu::PjrtRuntime& rt,
                                           const struct ggml_tensor* t,
                                           const cloudtpu::PjrtBufferPtr& buf,
                                           int out_token_axis,
                                           void* dst) {
    if (out_token_axis >= 0 && out_token_axis < 4 && buf && buf->dims.size() == 4 &&
        buf->dims[static_cast<size_t>(out_token_axis)] > t->ne[3 - out_token_axis]) {
        const size_t elem_sz = ggml_type_size(t->type);
        const int64_t logical_dims[4] = {t->ne[3], t->ne[2], t->ne[1], t->ne[0]};
        const int64_t padded_dims[4] = {buf->dims[0], buf->dims[1], buf->dims[2], buf->dims[3]};
        size_t hbm_bytes = 0;
        if (!cloudtpu::checked_tensor_nbytes(padded_dims, elem_sz, &hbm_bytes) ||
            hbm_bytes < ggml_nbytes(t)) {
            throw std::overflow_error("ggml-cloudtpu: padded output buffer byte size overflow");
        }
        std::vector<uint8_t> hbm_tmp(hbm_bytes);
        rt.buffer_to_host(buf->handle.get(), hbm_tmp.data(), hbm_bytes);
        cloudtpu::slice_dense_host_tensor(
            hbm_tmp.data(), dst, elem_sz, padded_dims, logical_dims, out_token_axis);
    } else {
        rt.buffer_to_host(buf->handle.get(), dst, ggml_nbytes(t));
    }
}

static void store_tensor_hbm_single(cloudtpu::PjrtRuntime& rt,
                                    struct ggml_tensor* t,
                                    cloudtpu::PjrtBufferPtr buf,
                                    int out_token_axis = -1) {
    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        // For full-buffer in-place views (like SET_ROWS or ROPE inplace), store directly under
        // the canonical root tensor's offset; for partial sub-slice views, patch the affected
        // byte range inside the root slot without evicting the parent tensor.
        const struct ggml_tensor* target = t;
        if (t->view_src != nullptr) {
            const struct ggml_tensor* root = unwrap_view_src(t);
            if (t->view_offs == 0 && ggml_nelements(t) == ggml_nelements(root) && t->type == root->type) {
                target = root;
            } else if (root && root->buffer == t->buffer) {
                const size_t t_off = tensor_offset_in_buffer(bctx, t);
                const size_t t_bytes = ggml_nbytes(t);
                std::vector<uint8_t> slice_bytes(t_bytes);
                copy_output_buffer_to_host_ptr(rt, t, buf, out_token_axis, slice_bytes.data());
                std::lock_guard<std::mutex> lock(bctx->mu);
                if (patch_view_into_root_slot_locked(
                        rt, bctx, t->buffer, t, t_off, 0, t_bytes, slice_bytes.data(), 0,
                        /*flush_now=*/true)) {
                    return;
                }
            }
        }
        const size_t t_off = tensor_offset_in_buffer(bctx, target);
        const size_t t_bytes = ggml_nbytes(target);
        std::lock_guard<std::mutex> lock(bctx->mu);
        bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
        CloudTpuTensorSlot& slot = bctx->slots[t_off];
        slot.offset = t_off;
        slot.nbytes = t_bytes;
        slot.ggml_dt = target->type;
        slot.storage = cloudtpu::TensorStorageKind::DENSE;
        slot.sharding = cloudtpu::TensorShardingKind::REPLICATED;
        slot.dtype = buf->dtype;
        slot.scale_dtype = cloudtpu::PjrtDType::INVALID;
        set_slot_hbm_dims(slot, target, buf, out_token_axis);
        slot.hbm_primary.resize(1);
        slot.hbm_primary[0] = std::move(buf);
        slot.hbm_scales.clear();
        slot.hbm_mins.clear();
        if (!slot.staging.empty()) {
            std::vector<uint8_t>().swap(slot.staging);
        }
        slot.staging_dirty = false;
        slot.upload_failed = false;
        slot.upload_error.clear();
        return;
    }

    if (t->data) {
        const struct ggml_tensor* shape_t = canonical_root(t);
        copy_output_buffer_to_host_ptr(rt, shape_t, buf, out_token_axis, t->data);
    }
}

static void store_tensor_hbm_spmd(cloudtpu::PjrtRuntime& rt,
                                  struct ggml_tensor* t,
                                  const std::vector<std::vector<cloudtpu::PjrtBufferPtr>>& per_dev_outs,
                                  size_t out_idx,
                                  int out_token_axis = -1) {
    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        const struct ggml_tensor* target = t;
        if (t->view_src != nullptr) {
            const struct ggml_tensor* root = unwrap_view_src(t);
            if (t->view_offs == 0 && ggml_nelements(t) == ggml_nelements(root) && t->type == root->type) {
                target = root;
            } else if (root && root->buffer == t->buffer && !per_dev_outs.empty()) {
                const size_t t_off = tensor_offset_in_buffer(bctx, t);
                const size_t t_bytes = ggml_nbytes(t);
                std::vector<uint8_t> slice_bytes(t_bytes);
                copy_output_buffer_to_host_ptr(rt, t, per_dev_outs[0][out_idx], out_token_axis, slice_bytes.data());
                std::lock_guard<std::mutex> lock(bctx->mu);
                if (patch_view_into_root_slot_locked(
                        rt, bctx, t->buffer, t, t_off, 0, t_bytes, slice_bytes.data(), 0,
                        /*flush_now=*/true)) {
                    return;
                }
            }
        }
        const size_t t_off = tensor_offset_in_buffer(bctx, target);
        const size_t t_bytes = ggml_nbytes(target);
        const size_t P = per_dev_outs.size();
        const auto& ref_buf = per_dev_outs[0][out_idx];
        std::lock_guard<std::mutex> lock(bctx->mu);
        bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
        CloudTpuTensorSlot& slot = bctx->slots[t_off];
        slot.offset = t_off;
        slot.nbytes = t_bytes;
        slot.ggml_dt = target->type;
        slot.storage = cloudtpu::TensorStorageKind::DENSE;
        slot.sharding = cloudtpu::TensorShardingKind::REPLICATED;
        slot.dtype = ref_buf->dtype;
        slot.scale_dtype = cloudtpu::PjrtDType::INVALID;
        set_slot_hbm_dims(slot, target, ref_buf, out_token_axis);
        slot.hbm_primary.resize(P);
        for (size_t p = 0; p < P; ++p) {
            slot.hbm_primary[p] = per_dev_outs[p][out_idx];
        }
        slot.hbm_scales.clear();
        slot.hbm_mins.clear();
        if (!slot.staging.empty()) {
            std::vector<uint8_t>().swap(slot.staging);
        }
        slot.staging_dirty = false;
        slot.upload_failed = false;
        slot.upload_error.clear();
        return;
    }

    if (t->data && !per_dev_outs.empty()) {
        const struct ggml_tensor* shape_t = canonical_root(t);
        copy_output_buffer_to_host_ptr(rt, shape_t, per_dev_outs[0][out_idx], out_token_axis, t->data);
    }
}

static bool same_backing_address_space(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const bool a_tpu = (a->buffer != nullptr && is_cloudtpu_buffer(a->buffer));
    const bool b_tpu = (b->buffer != nullptr && is_cloudtpu_buffer(b->buffer));
    if (a_tpu || b_tpu) {
        return a_tpu && b_tpu && a->buffer == b->buffer;
    }
    // Both tensors live in host virtual memory (raw host pointers or non-CloudTPU host buffers),
    // so byte-range checks detect host-backed in-place write/read aliasing too.
    return true;
}

static bool tensor_ranges_overlap(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    if (!same_backing_address_space(a, b)) {
        return false;
    }
    const size_t a_bytes = ggml_nbytes(a);
    const size_t b_bytes = ggml_nbytes(b);
    if (a_bytes == 0 || b_bytes == 0) {
        return false;
    }
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t a1 = a0 + a_bytes;
    const uintptr_t b0 = reinterpret_cast<uintptr_t>(b->data);
    const uintptr_t b1 = b0 + b_bytes;
    return a0 < b1 && b0 < a1;
}

static bool tensor_range_covers(const struct ggml_tensor* outer, const struct ggml_tensor* inner) {
    if (!same_backing_address_space(outer, inner)) {
        return false;
    }
    const size_t o_bytes = ggml_nbytes(outer);
    const size_t i_bytes = ggml_nbytes(inner);
    if (o_bytes == 0 || i_bytes == 0) {
        return false;
    }
    const uintptr_t o0 = reinterpret_cast<uintptr_t>(outer->data);
    const uintptr_t o1 = o0 + o_bytes;
    const uintptr_t i0 = reinterpret_cast<uintptr_t>(inner->data);
    const uintptr_t i1 = i0 + i_bytes;
    return o0 <= i0 && o1 >= i1;
}

static CloudSlotKey make_slot_key(const struct ggml_tensor* t) {
    if (!t || !t->buffer || !is_cloudtpu_buffer(t->buffer) || !t->data) {
        return {};
    }
    return {t->buffer, reinterpret_cast<uintptr_t>(t->data)};
}

// Returns the canonical root tensor updated in-place by `node` iff `node` produces a complete,
// dense, contiguous representation of that root slot suitable for in-slice SSA forwarding.
static const struct ggml_tensor* get_forwardable_inplace_root(const struct ggml_tensor* node) {
    if (!node || is_skipped_graph_node(node)) {
        return nullptr;
    }
    if (node->op != GGML_OP_SET_ROWS && node->view_src == nullptr) {
        return nullptr;
    }
    if (!node->buffer || !is_cloudtpu_buffer(node->buffer) || !node->data) {
        return nullptr;
    }
    if (!is_supported_output_layout(node) || !is_dense_supported_type(node->type)) {
        return nullptr;
    }
    const struct ggml_tensor* root = canonical_root(node);
    if (!root || root->buffer != node->buffer || root->data != node->data ||
        root->type != node->type || !ggml_is_contiguous(root) ||
        ggml_nelements(root) != ggml_nelements(node)) {
        return nullptr;
    }
    if (node->op == GGML_OP_SET_ROWS) {
        if (!node->src[2]) {
            return nullptr;
        }
        const struct ggml_tensor* tbl_root = canonical_root(node->src[2]);
        if (!tbl_root || tbl_root->buffer != root->buffer || tbl_root->data != root->data ||
            tbl_root->type != root->type || ggml_nelements(tbl_root) != ggml_nelements(root)) {
            return nullptr;
        }
    }
    return root;
}

static bool can_collapse_view_chain_to_leaf(const struct ggml_tensor* t,
                                            const struct ggml_tensor* target_leaf,
                                            int depth = 0) {
    if (!t || depth > 8) {
        return false;
    }
    if (t == target_leaf) {
        return true;
    }
    const struct ggml_tensor* parent = view_parent(t);
    if (!t->view_src || !parent ||
        !(is_dense_supported_type(t->type) || t->type == GGML_TYPE_I64) ||
        t->type != parent->type) {
        return false;
    }
    if ((t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) && t->view_offs == parent->view_offs) {
        return can_collapse_view_chain_to_leaf(parent, target_leaf, depth + 1);
    }
    if (t->view_offs == parent->view_offs && ggml_is_contiguous(t) &&
        ggml_is_contiguous(parent) && ggml_nelements(t) == ggml_nelements(parent)) {
        return can_collapse_view_chain_to_leaf(parent, target_leaf, depth + 1);
    }
    const struct ggml_tensor* slice_base = nullptr;
    if (try_build_view_slice(t, nullptr, &slice_base) && slice_base != nullptr) {
        return can_collapse_view_chain_to_leaf(slice_base, target_leaf, depth + 1);
    }
    return false;
}

static bool is_alias_tracked_write(const struct ggml_tensor* t) {
    return !t || t->op == GGML_OP_SET_ROWS || t->view_src != nullptr ||
           !t->buffer || !is_cloudtpu_buffer(t->buffer);
}

// True when `node` at cgraph index `node_idx` is a pure ggml_gallocr scratch intermediate
// whose consumers have all finished before `before_idx` (and is not an output or alias write),
// so a later standalone CloudTPU write reusing part of its byte range safely retires it.
static bool is_dead_gallocr_scratch(const ggml_backend_cloudtpu_context* ctx,
                                    const struct ggml_tensor* node,
                                    int node_idx,
                                    int before_idx) {
    if (!node || node_idx < 0 || is_alias_tracked_write(node)) {
        return false;
    }
    if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    const int last_use = ctx->liveness.last_consumer[static_cast<size_t>(node_idx)];
    return last_use >= 0 && last_use < before_idx;
}

static void compute_cgraph_liveness(ggml_backend_cloudtpu_context* ctx, const struct ggml_cgraph* cgraph) {
    const int n = cgraph->n_nodes;
    ctx->liveness.first_consumer.assign(static_cast<size_t>(n), n);
    ctx->liveness.last_consumer.assign(static_cast<size_t>(n), -1);
    ctx->liveness.first_overwrite.assign(static_cast<size_t>(n), n);

    ctx->cgraph_node_idx.clear();
    if (ctx->cgraph_node_idx.bucket_count() < static_cast<size_t>(n * 2)) {
        ctx->cgraph_node_idx.reserve(static_cast<size_t>(n * 2));
    }
    for (int i = 0; i < n; ++i) {
        ctx->cgraph_node_idx[cgraph->nodes[i]] = i;
    }

    // Track consumers in one O(N) pass.
    for (int k = 0; k < n; ++k) {
        const struct ggml_tensor* nk = cgraph->nodes[k];
        if (!nk || (nk->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (!is_view_noop_op(nk->op)) {
            const int max_src = (nk->op == GGML_OP_CPY || nk->op == GGML_OP_CONT) ? 1 : GGML_MAX_SRC;
            for (int s = 0; s < max_src; ++s) {
                for (const struct ggml_tensor* cur = nk->src[s]; cur != nullptr; cur = view_parent(cur)) {
                    auto it = ctx->cgraph_node_idx.find(cur);
                    if (it != ctx->cgraph_node_idx.end() && it->second < k) {
                        const size_t idx = static_cast<size_t>(it->second);
                        if (k < ctx->liveness.first_consumer[idx]) ctx->liveness.first_consumer[idx] = k;
                        if (k > ctx->liveness.last_consumer[idx]) ctx->liveness.last_consumer[idx] = k;
                    }
                }
            }
        }
        for (const struct ggml_tensor* cur = view_parent(nk); cur != nullptr; cur = view_parent(cur)) {
            auto it = ctx->cgraph_node_idx.find(cur);
            if (it != ctx->cgraph_node_idx.end() && it->second < k) {
                const size_t idx = static_cast<size_t>(it->second);
                if (k > ctx->liveness.last_consumer[idx]) ctx->liveness.last_consumer[idx] = k;
            }
        }
    }

    // Track HBM buffer-slot reuse (ggml_gallocr liveness) in O(N * active_live_slots).
    struct LiveRange {
        int node_idx;
        uintptr_t start;
        uintptr_t end;
    };
    std::unordered_map<ggml_backend_buffer_t, std::vector<LiveRange>> active_by_buf;
    for (int k = 0; k < n; ++k) {
        const struct ggml_tensor* nk = cgraph->nodes[k];
        if (is_skipped_graph_node(nk) || !nk->buffer || !is_cloudtpu_buffer(nk->buffer)) {
            continue;
        }
        const uintptr_t b0 = reinterpret_cast<uintptr_t>(nk->data);
        const uintptr_t b1 = b0 + ggml_nbytes(nk);
        auto& active = active_by_buf[nk->buffer];
        for (size_t a = 0; a < active.size();) {
            // Only treat a later write as killing an earlier write when it completely covers
            // the earlier byte range, OR when a standalone compute op reuses freed ggml_gallocr
            // scratch bytes of a dead non-output intermediate whose consumers already finished;
            // partial subview overwrites of live/output buffers must preserve untouched bytes.
            const bool covers = (b0 <= active[a].start && b1 >= active[a].end);
            const bool overlaps = (active[a].start < b1 && b0 < active[a].end);
            const bool dead_scratch_reuse =
                overlaps && !is_alias_tracked_write(nk) &&
                is_dead_gallocr_scratch(ctx, cgraph->nodes[active[a].node_idx], active[a].node_idx, k);
            if (covers || dead_scratch_reuse) {
                ctx->liveness.first_overwrite[static_cast<size_t>(active[a].node_idx)] = k;
                active[a] = active.back();
                active.pop_back();
            } else {
                ++a;
            }
        }
        active.push_back({k, b0, b1});
    }
}

static inline void sig_push(std::vector<uint64_t>& sig, uint64_t v) {
    sig.push_back(v);
}

static inline void sig_push_f32(std::vector<uint64_t>& sig, float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    sig.push_back(static_cast<uint64_t>(u));
}

static uint64_t hash_sig_words(const std::vector<uint64_t>& words) {
    uint64_t h = 14695981039346656037ULL;
    for (uint64_t w : words) {
        h ^= w;
        h *= 1099511628211ULL;
    }
    return h;
}

static cloudtpu::OpNodeSpec build_op_node_spec(const struct ggml_tensor* node,
                                               uint64_t dst_id,
                                               const std::vector<uint64_t>& src_ids) {
    cloudtpu::OpNodeSpec spec;
    spec.src_ids = src_ids;
    spec.dst = tensor_to_spec(node, dst_id);

    switch (node->op) {
        case GGML_OP_MUL_MAT: {
            spec.kind = cloudtpu::CloudOpKind::MUL_MAT;
            spec.high_precision =
                (ggml_prec_get_acc(node) == GGML_PREC_F32) ||
                (ggml_prec_get_src(node, 0) == GGML_PREC_F32) ||
                (ggml_prec_get_src(node, 1) == GGML_PREC_F32);
            break;
        }
        case GGML_OP_RMS_NORM:
            spec.kind = cloudtpu::CloudOpKind::RMS_NORM;
            std::memcpy(&spec.eps, node->op_params, sizeof(float));
            break;
        case GGML_OP_ROPE: {
            spec.kind = cloudtpu::CloudOpKind::ROPE;
            const int32_t* p = node->op_params;
            spec.rope.n_dims = p[1];
            spec.rope.mode = p[2];
            spec.rope.n_ctx_orig = p[4];
            spec.rope.truncate = ggml_rope_get_truncate(node);
            std::memcpy(&spec.rope.freq_base, p + 5, sizeof(float));
            std::memcpy(&spec.rope.freq_scale, p + 6, sizeof(float));
            std::memcpy(&spec.rope.ext_factor, p + 7, sizeof(float));
            std::memcpy(&spec.rope.attn_factor, p + 8, sizeof(float));
            std::memcpy(&spec.rope.beta_fast, p + 9, sizeof(float));
            std::memcpy(&spec.rope.beta_slow, p + 10, sizeof(float));
            spec.rope.n_offs = p[15];
            break;
        }
        case GGML_OP_UNARY:
            spec.kind = cloudtpu::CloudOpKind::SILU;
            break;
        case GGML_OP_GLU:
            spec.kind = cloudtpu::CloudOpKind::SWIGLU;
            spec.glu_swapped = ggml_get_op_params_i32(node, 1);
            break;
        case GGML_OP_SOFT_MAX:
            spec.kind = cloudtpu::CloudOpKind::SOFT_MAX;
            std::memcpy(&spec.soft_max.scale, reinterpret_cast<const float*>(node->op_params) + 0, sizeof(float));
            std::memcpy(&spec.soft_max.max_bias, reinterpret_cast<const float*>(node->op_params) + 1, sizeof(float));
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            spec.kind = cloudtpu::CloudOpKind::FLASH_ATTN_EXT;
            std::memcpy(&spec.flash_attn.scale, reinterpret_cast<const float*>(node->op_params) + 0, sizeof(float));
            std::memcpy(&spec.flash_attn.max_bias, reinterpret_cast<const float*>(node->op_params) + 1, sizeof(float));
            std::memcpy(&spec.flash_attn.logit_softcap, reinterpret_cast<const float*>(node->op_params) + 2, sizeof(float));
            spec.high_precision = (ggml_flash_attn_ext_get_prec(node) == GGML_PREC_F32);
            break;
        case GGML_OP_GET_ROWS:
            spec.kind = cloudtpu::CloudOpKind::GET_ROWS;
            break;
        case GGML_OP_SET_ROWS:
            spec.kind = cloudtpu::CloudOpKind::SET_ROWS;
            break;
        case GGML_OP_ADD:
            spec.kind = cloudtpu::CloudOpKind::ADD;
            break;
        case GGML_OP_MUL:
            spec.kind = cloudtpu::CloudOpKind::MUL;
            break;
        case GGML_OP_SCALE:
            spec.kind = cloudtpu::CloudOpKind::SCALE;
            std::memcpy(&spec.scale, reinterpret_cast<const float*>(node->op_params) + 0, sizeof(float));
            std::memcpy(&spec.bias, reinterpret_cast<const float*>(node->op_params) + 1, sizeof(float));
            break;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            spec.kind = cloudtpu::CloudOpKind::CPY;
            break;
        default:
            GGML_ABORT("ggml-cloudtpu: unexpected op %s", ggml_op_desc(node));
    }
    return spec;
}

static void await_sync_launches(cloudtpu::PjrtRuntime& rt,
                                std::deque<std::vector<cloudtpu::PjrtEventHandle>>& launches) {
    while (!launches.empty()) {
        std::vector<cloudtpu::PjrtEventHandle> launch = std::move(launches.front());
        launches.pop_front();
        for (auto& ev : launch) {
            if (ev) {
                rt.await_event(std::move(ev));
            }
        }
    }
}

static void ggml_backend_cloudtpu_synchronize_ctx(ggml_backend_cloudtpu_context* ctx) {
    if (!ctx) {
        return;
    }
    std::deque<std::vector<cloudtpu::PjrtEventHandle>> to_wait = ctx->pending_sync_launches.drain_all();
    if (to_wait.empty()) {
        return;
    }
    cloudtpu::PjrtRuntime& rt = cloudtpu::shared_pjrt_runtime(nullptr);
    if (!rt.is_initialized()) {
        return;
    }
    await_sync_launches(rt, to_wait);
}

struct BuiltSubgraphSpec {
    cloudtpu::GraphProgramSpec prog_spec;
    std::unordered_map<uint64_t, uint32_t> id_to_input_idx;
    std::vector<const struct ggml_tensor*> input_nodes;
    std::vector<const struct ggml_tensor*> output_nodes;
};

static int64_t detect_slice_candidate_tokens(const struct ggml_cgraph* cgraph,
                                             int start_idx,
                                             int end_idx,
                                             int64_t max_bucket) {
    if (max_bucket <= 0) {
        return 0;
    }
    for (int i = start_idx; i < end_idx; ++i) {
        const struct ggml_tensor* node = cgraph->nodes[i];
        if (is_skipped_graph_node(node)) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_GET_ROWS: {
                const struct ggml_tensor* idx = node->src[1];
                if (idx && idx->ne[1] == 1 && idx->ne[2] == 1 && idx->ne[3] == 1 &&
                    idx->ne[0] >= 1 && idx->ne[0] <= max_bucket) {
                    return idx->ne[0];
                }
                break;
            }
            case GGML_OP_SET_ROWS: {
                const struct ggml_tensor* idx = node->src[1];
                if (idx && idx->ne[1] == 1 && idx->ne[2] == 1 && idx->ne[3] == 1 &&
                    idx->ne[0] >= 1 && idx->ne[0] <= max_bucket) {
                    return idx->ne[0];
                }
                break;
            }
            case GGML_OP_ROPE: {
                const struct ggml_tensor* x = node->src[0];
                const struct ggml_tensor* pos = node->src[1];
                if (x && pos && pos->ne[1] == 1 && pos->ne[2] == 1 && pos->ne[3] == 1 &&
                    x->ne[2] == pos->ne[0] && pos->ne[0] >= 1 && pos->ne[0] <= max_bucket) {
                    return pos->ne[0];
                }
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                const struct ggml_tensor* q = node->src[0];
                if (q && q->ne[1] >= 1 && q->ne[1] <= max_bucket) {
                    return q->ne[1];
                }
                break;
            }
            case GGML_OP_MUL_MAT: {
                const struct ggml_tensor* src1 = node->src[1];
                if (src1 && src1->ne[1] >= 1 && src1->ne[1] <= max_bucket) {
                    return src1->ne[1];
                }
                break;
            }
            default:
                break;
        }
    }
    return 0;
}

static BuiltSubgraphSpec build_subgraph_program_spec(const ggml_backend_cloudtpu_context* ctx,
                                                     const struct ggml_cgraph* cgraph,
                                                     int start_idx,
                                                     int end_idx,
                                                     int P,
                                                     const std::vector<uint8_t>& live_out_flags,
                                                     const std::vector<uint8_t>& keep_sharded_flags,
                                                     const std::vector<int32_t>* input_alias_out,
                                                     bool use_scratch_input_meta) {
    BuiltSubgraphSpec built;
    cloudtpu::GraphProgramSpec& prog_spec = built.prog_spec;
    prog_spec.spmd_chips = P;
    prog_spec.bf16_matmul = ctx->env.bf16_matmul;
    prog_spec.bf16_flash_attn = ctx->env.bf16_flash_attn;
    prog_spec.q4_split_dot = ctx->env.q4_split_dot;
    prog_spec.mm_pad_m = ctx->env.mm_pad_m;
    prog_spec.freetile_mosaic = ctx->env.freetile_mosaic;
    prog_spec.freetile_mosaic_version = ctx->env.freetile_mosaic_version;
    if (ctx->env.freetile_mosaic) {
        cloudtpu::PjrtRuntime& rt = cloudtpu::shared_pjrt_runtime(nullptr);
        if (rt.is_initialized() && ctx->device_index >= 0 && ctx->device_index < rt.device_count()) {
            const freetile::ChipTarget ct = freetile::detect_chip_target(rt.device_info(ctx->device_index).kind);
            prog_spec.freetile_chip_target = static_cast<int32_t>(ct);
        }
    }

    struct SpecCacheEntry {
        uint64_t id = 0;
        uint64_t slot_epoch = 0;
    };
    struct InSliceSlotSpec {
        uint64_t id = 0;
        const struct ggml_tensor* root = nullptr;
        uint64_t epoch = 0;
    };
    std::unordered_map<const struct ggml_tensor*, SpecCacheEntry> tensor_to_id;
    std::unordered_set<const struct ggml_tensor*> spec_computed_nodes;
    std::unordered_map<CloudSlotKey, InSliceSlotSpec, CloudSlotKeyHash> slot_spec_state;
    uint64_t spec_next_id = 1;
    uint64_t spec_slot_epoch = 0;
    uint32_t next_in_idx = 0;

    auto get_or_register_input = [&](auto& self, const struct ggml_tensor* t) -> uint64_t {
        if (!t) return 0;
        if (spec_computed_nodes.count(t)) {
            return tensor_to_id.at(t).id;
        }
        const struct ggml_tensor* root_t = unwrap_view_src(t);
        const CloudSlotKey root_sk = make_slot_key(root_t);
        uint64_t cur_epoch = 0;
        if (root_sk.valid()) {
            auto sit = slot_spec_state.find(root_sk);
            if (sit != slot_spec_state.end()) {
                cur_epoch = sit->second.epoch;
            }
        }
        auto it = tensor_to_id.find(t);
        if (it != tensor_to_id.end() && it->second.slot_epoch == cur_epoch) {
            return it->second.id;
        }
        const struct ggml_tensor* parent = view_parent(t);
        if (t->view_src != nullptr && parent != nullptr &&
            (is_dense_supported_type(t->type) || t->type == GGML_TYPE_I64) &&
            t->type == parent->type) {
            if (t->op == GGML_OP_PERMUTE && t->view_offs == parent->view_offs) {
                const uint64_t base_id = self(self, parent);
                const uint64_t tid = spec_next_id++;
                cloudtpu::OpNodeSpec p_op;
                p_op.kind = cloudtpu::CloudOpKind::TRANSPOSE;
                p_op.src_ids = {base_id};
                p_op.dst = tensor_to_spec(t, tid);
                const int32_t* ax = static_cast<const int32_t*>(t->op_params);
                int inv[4] = {0, 1, 2, 3};
                for (int g = 0; g < 4; ++g) inv[ax[g]] = g;
                for (int h = 0; h < 4; ++h) p_op.perm[h] = 3 - inv[3 - h];
                prog_spec.ops.push_back(std::move(p_op));
                tensor_to_id[t] = {tid, cur_epoch};
                return tid;
            }
            if (t->op == GGML_OP_TRANSPOSE && t->view_offs == parent->view_offs) {
                const uint64_t base_id = self(self, parent);
                const uint64_t tid = spec_next_id++;
                cloudtpu::OpNodeSpec p_op;
                p_op.kind = cloudtpu::CloudOpKind::TRANSPOSE;
                p_op.src_ids = {base_id};
                p_op.dst = tensor_to_spec(t, tid);
                p_op.perm[0] = 0;
                p_op.perm[1] = 1;
                p_op.perm[2] = 3;
                p_op.perm[3] = 2;
                prog_spec.ops.push_back(std::move(p_op));
                tensor_to_id[t] = {tid, cur_epoch};
                return tid;
            }
            if (t->view_offs == parent->view_offs && ggml_is_contiguous(t) &&
                ggml_is_contiguous(parent) &&
                ggml_nelements(t) == ggml_nelements(parent)) {
                const uint64_t base_id = self(self, parent);
                const uint64_t rid = spec_next_id++;
                cloudtpu::OpNodeSpec r_op;
                r_op.kind = cloudtpu::CloudOpKind::RESHAPE;
                r_op.src_ids = {base_id};
                r_op.dst = tensor_to_spec(t, rid);
                prog_spec.ops.push_back(std::move(r_op));
                tensor_to_id[t] = {rid, cur_epoch};
                return rid;
            }
            ViewSlicePlan vplan;
            const struct ggml_tensor* slice_base = nullptr;
            if (try_build_view_slice(t, &vplan, &slice_base)) {
                const uint64_t base_id = self(self, slice_base);
                const uint64_t sid = spec_next_id++;
                cloudtpu::OpNodeSpec s_op;
                s_op.kind = cloudtpu::CloudOpKind::SLICE;
                s_op.src_ids = {base_id};
                s_op.dst = tensor_to_spec(t, sid);
                s_op.slice_reshape = std::move(vplan.reshape_dims);
                s_op.slice_starts = std::move(vplan.starts);
                s_op.slice_limits = std::move(vplan.limits);
                prog_spec.ops.push_back(std::move(s_op));
                tensor_to_id[t] = {sid, cur_epoch};
                return sid;
            }
        }
        const CloudSlotKey leaf_sk = make_slot_key(t);
        if (leaf_sk.valid()) {
            auto sit = slot_spec_state.find(leaf_sk);
            if (sit != slot_spec_state.end() && sit->second.root != nullptr &&
                is_dense_supported_type(t->type) && t->type == sit->second.root->type &&
                ggml_is_contiguous(t) && ggml_nelements(t) == ggml_nelements(sit->second.root)) {
                const struct ggml_tensor* r = sit->second.root;
                if (t->ne[0] == r->ne[0] && t->ne[1] == r->ne[1] &&
                    t->ne[2] == r->ne[2] && t->ne[3] == r->ne[3]) {
                    tensor_to_id[t] = {sit->second.id, sit->second.epoch};
                    return sit->second.id;
                }
                const uint64_t rid = spec_next_id++;
                cloudtpu::OpNodeSpec r_op;
                r_op.kind = cloudtpu::CloudOpKind::RESHAPE;
                r_op.src_ids = {sit->second.id};
                r_op.dst = tensor_to_spec(t, rid);
                prog_spec.ops.push_back(std::move(r_op));
                tensor_to_id[t] = {rid, sit->second.epoch};
                return rid;
            }
        }
        const uint64_t id = spec_next_id++;
        const uint32_t in_idx = next_in_idx++;
        const cloudtpu::TensorShardingKind sh = use_scratch_input_meta
                                                    ? ctx->scratch_input_sharding[in_idx]
                                                    : lookup_tensor_sharding(t, P);
        const cloudtpu::PjrtDType sdt = use_scratch_input_meta
                                            ? ctx->scratch_input_scale_dt[in_idx]
                                            : lookup_tensor_scale_dtype(t, ctx->env.q4k_bf16_scales);
        const Q4SlotPlan q4p = use_scratch_input_meta
                                   ? ctx->scratch_input_q4_plans[in_idx]
                                   : lookup_tensor_q4_plan(t, P, sh, ctx->env.q4_mode, ctx->env.q4k_u4);
        tensor_to_id[t] = {id, 0};
        built.id_to_input_idx[id] = in_idx;
        built.input_nodes.push_back(t);
        cloudtpu::TensorSpec ts = tensor_to_spec(t, id, sh, &q4p);
        ts.scale_dtype = sdt;
        ts.alias_output_idx = (input_alias_out != nullptr) ? (*input_alias_out)[in_idx] : -1;
        prog_spec.inputs.push_back(std::move(ts));
        return id;
    };

    for (int i = start_idx; i < end_idx; ++i) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (is_skipped_graph_node(node)) {
            continue;
        }
        std::vector<uint64_t> src_ids;
        if (node->op == GGML_OP_ROPE || node->op == GGML_OP_SOFT_MAX) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[1]));
            if (node->src[2]) src_ids.push_back(get_or_register_input(get_or_register_input, node->src[2]));
        } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[1]));
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[2]));
            if (node->src[3]) src_ids.push_back(get_or_register_input(get_or_register_input, node->src[3]));
        } else if (node->op == GGML_OP_GLU) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
            if (node->src[1]) src_ids.push_back(get_or_register_input(get_or_register_input, node->src[1]));
        } else if (node->op == GGML_OP_CPY || node->op == GGML_OP_CONT) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
        } else {
            for (int s = 0; s < GGML_MAX_SRC && node->src[s] != nullptr; ++s) {
                src_ids.push_back(get_or_register_input(get_or_register_input, node->src[s]));
            }
        }

        const size_t rel_idx = static_cast<size_t>(i - start_idx);
        const uint64_t dst_id = spec_next_id++;
        cloudtpu::OpNodeSpec op_spec = build_op_node_spec(node, dst_id, src_ids);
        op_spec.keep_sharded_out = (keep_sharded_flags[rel_idx] != 0);
        prog_spec.ops.push_back(std::move(op_spec));
        spec_computed_nodes.insert(node);
        tensor_to_id[node] = {dst_id, 0};

        uint64_t store_id = dst_id;
        const struct ggml_tensor* root = canonical_root(node);
        if (root != node && (root->ne[0] != node->ne[0] || root->ne[1] != node->ne[1] ||
                             root->ne[2] != node->ne[2] || root->ne[3] != node->ne[3])) {
            const uint64_t rid = spec_next_id++;
            cloudtpu::OpNodeSpec r_op;
            r_op.kind = cloudtpu::CloudOpKind::RESHAPE;
            r_op.src_ids = {dst_id};
            r_op.dst = tensor_to_spec(root, rid);
            prog_spec.ops.push_back(std::move(r_op));
            store_id = rid;
        }

        const struct ggml_tensor* fwd_root = get_forwardable_inplace_root(node);
        if (fwd_root != nullptr) {
            const CloudSlotKey sk = make_slot_key(fwd_root);
            if (sk.valid()) {
                for (auto sit = slot_spec_state.begin(); sit != slot_spec_state.end();) {
                    if (!(sit->first == sk) && tensor_ranges_overlap(fwd_root, sit->second.root)) {
                        sit = slot_spec_state.erase(sit);
                    } else {
                        ++sit;
                    }
                }
                const uint64_t ep = ++spec_slot_epoch;
                slot_spec_state[sk] = {store_id, fwd_root, ep};
            }
        } else if (node->buffer && is_cloudtpu_buffer(node->buffer) && node->data) {
            for (auto sit = slot_spec_state.begin(); sit != slot_spec_state.end();) {
                if (tensor_ranges_overlap(node, sit->second.root)) {
                    sit = slot_spec_state.erase(sit);
                } else {
                    ++sit;
                }
            }
        }

        if (live_out_flags[rel_idx]) {
            prog_spec.output_ids.push_back(store_id);
            built.output_nodes.push_back(node);
        }
    }
    return built;
}

static enum ggml_status execute_subgraph(ggml_backend_cloudtpu_context* ctx,
                                         cloudtpu::PjrtRuntime& rt,
                                         struct ggml_cgraph* cgraph,
                                         int start_idx,
                                         int end_idx,
                                         bool is_final_subgraph) {
    const int P = ctx->spmd_chips;
    const int n_nodes = cgraph->n_nodes;
    const int slice_len = end_idx - start_idx;

    // 1. Pre-compute live-out flags for [start_idx, end_idx) in O(slice_len) using cgraph liveness.
    std::vector<uint8_t> live_out_flags(static_cast<size_t>(slice_len), 0);
    std::vector<uint8_t> keep_sharded_flags(static_cast<size_t>(slice_len), 0);
    int compute_ops = 0;
    for (int i = start_idx; i < end_idx; ++i) {
        const struct ggml_tensor* node = cgraph->nodes[i];
        if (is_skipped_graph_node(node)) {
            continue;
        }
        ++compute_ops;
        const size_t gi = static_cast<size_t>(i);
        const bool overwritten_in_slice = (ctx->liveness.first_overwrite[gi] < end_idx);
        const bool alive_past_cgraph = (ctx->liveness.first_overwrite[gi] >= n_nodes);
        const bool consumed_inside = (ctx->liveness.first_consumer[gi] < end_idx);
        const bool needed_outside = (ctx->liveness.last_consumer[gi] >= end_idx);
        const bool is_live_out =
            !overwritten_in_slice &&
            ((i == end_idx - 1) ||
             (node->op == GGML_OP_SET_ROWS) ||
             (node->view_src != nullptr) ||
             !consumed_inside ||
             (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
             needed_outside ||
             alive_past_cgraph ||
             (!node->buffer || !is_cloudtpu_buffer(node->buffer)));
        live_out_flags[static_cast<size_t>(i - start_idx)] = is_live_out ? 1 : 0;
    }
    if (compute_ops == 0) {
        return GGML_STATUS_SUCCESS;
    }

    // 2. When SPMD multi-chip (P > 1), detect Megatron-LM FFN patterns inside this slice:
    //    (Gate SHARD_N, Up SHARD_N) -> SwiGLU (or SiLU + Mul) -> Down SHARD_K.
    //    Keeping Gate/Up sharded eliminates 2 all_gathers per FFN (1 all_reduce total).
    if (P > 1) {
        auto in_slice_non_live = [&](const struct ggml_tensor* t, int* out_idx) -> bool {
            if (!t) return false;
            auto it = ctx->cgraph_node_idx.find(t);
            if (it == ctx->cgraph_node_idx.end()) return false;
            const int idx = it->second;
            if (idx < start_idx || idx >= end_idx) return false;
            if (live_out_flags[static_cast<size_t>(idx - start_idx)]) return false;
            if (ctx->liveness.last_consumer[static_cast<size_t>(idx)] >= end_idx) return false;
            *out_idx = idx;
            return true;
        };
        auto is_shard_n_mm = [&](const struct ggml_tensor* t, int* out_idx) -> bool {
            if (!in_slice_non_live(t, out_idx)) return false;
            if (t->op != GGML_OP_MUL_MAT || !t->src[0]) return false;
            if (ctx->cgraph_node_idx.find(t->src[0]) != ctx->cgraph_node_idx.end() &&
                ctx->cgraph_node_idx[t->src[0]] >= start_idx) {
                return false;
            }
            return lookup_tensor_sharding(t->src[0], P) == cloudtpu::TensorShardingKind::SHARD_N;
        };
        auto count_consumers_in_slice = [&](const struct ggml_tensor* target) -> int {
            int cnt = 0;
            for (int k = start_idx; k < end_idx; ++k) {
                const struct ggml_tensor* nk = cgraph->nodes[k];
                if (is_skipped_graph_node(nk)) continue;
                const int max_src = (nk->op == GGML_OP_CPY || nk->op == GGML_OP_CONT) ? 1 : GGML_MAX_SRC;
                for (int s = 0; s < max_src && nk->src[s] != nullptr; ++s) {
                    if (nk->src[s] == target) ++cnt;
                }
            }
            return cnt;
        };

        for (int i = start_idx; i < end_idx; ++i) {
            const struct ggml_tensor* dn = cgraph->nodes[i];
            if (is_skipped_graph_node(dn)) continue;
            if (dn->op != GGML_OP_MUL_MAT || !dn->src[0] || !dn->src[1]) continue;
            if (lookup_tensor_sharding(dn->src[0], P) != cloudtpu::TensorShardingKind::SHARD_K) continue;
            int act_idx = -1;
            const struct ggml_tensor* act = dn->src[1];
            if (!in_slice_non_live(act, &act_idx) || count_consumers_in_slice(act) != 1) continue;

            if (act->op == GGML_OP_GLU && act->src[0] && act->src[1]) {
                int g_idx = -1, u_idx = -1;
                if (is_shard_n_mm(act->src[0], &g_idx) && is_shard_n_mm(act->src[1], &u_idx) &&
                    g_idx != u_idx &&
                    count_consumers_in_slice(act->src[0]) == 1 &&
                    count_consumers_in_slice(act->src[1]) == 1) {
                    keep_sharded_flags[static_cast<size_t>(g_idx - start_idx)] = 1;
                    keep_sharded_flags[static_cast<size_t>(u_idx - start_idx)] = 1;
                }
            } else if (act->op == GGML_OP_MUL && act->src[0] && act->src[1]) {
                for (int side = 0; side < 2; ++side) {
                    const struct ggml_tensor* s_node = act->src[side];
                    const struct ggml_tensor* u_node = act->src[1 - side];
                    int s_idx = -1, g_idx = -1, u_idx = -1;
                    if (in_slice_non_live(s_node, &s_idx) &&
                        s_node->op == GGML_OP_UNARY && ggml_get_unary_op(s_node) == GGML_UNARY_OP_SILU &&
                        count_consumers_in_slice(s_node) == 1 &&
                        is_shard_n_mm(s_node->src[0], &g_idx) && is_shard_n_mm(u_node, &u_idx) &&
                        g_idx != u_idx &&
                        count_consumers_in_slice(s_node->src[0]) == 1 &&
                        count_consumers_in_slice(u_node) == 1) {
                        keep_sharded_flags[static_cast<size_t>(g_idx - start_idx)] = 1;
                        keep_sharded_flags[static_cast<size_t>(u_idx - start_idx)] = 1;
                        break;
                    }
                }
            }
        }
    }

    // 3. Selectively prune plan_cache entries whose executables were evicted by PJRT runtime,
    //    preserving plans whose executables survived eviction.
    auto prune_evicted_plans = [&]() {
        const uint64_t cur_epoch = rt.eviction_epoch();
        if (ctx->plan_cache_epoch != cur_epoch) {
            for (auto it = ctx->plan_cache.begin(); it != ctx->plan_cache.end();) {
                if (it->second.exec_key.empty() || !rt.has_cached_exec(it->second.exec_key)) {
                    it = ctx->plan_cache.erase(it);
                } else {
                    ++it;
                }
            }
            ctx->plan_cache_epoch = cur_epoch;
        }
    };
    prune_evicted_plans();

    // 3b. Continuous batching padded buckets: analyze dataflow token_axis only when 1 <= T < B.
    BuiltSubgraphSpec bucket_built;
    cloudtpu::BucketAnalysisResult bucket_plan;
    if (!ctx->env.batch_buckets.empty()) {
        const int64_t cand_T = detect_slice_candidate_tokens(
            cgraph, start_idx, end_idx, ctx->env.batch_buckets.back());
        const int64_t bucket_B = (cand_T >= 1) ? cloudtpu::select_batch_bucket(cand_T, ctx->env.batch_buckets) : 0;
        if (bucket_B > cand_T) {
            bucket_built = build_subgraph_program_spec(
                ctx, cgraph, start_idx, end_idx, P, live_out_flags, keep_sharded_flags, nullptr, false);
            bucket_plan = cloudtpu::analyze_program_token_axes(bucket_built.prog_spec, cand_T, bucket_B);
            if (bucket_plan.success && bucket_plan.id_token_axis.empty()) {
                bucket_plan.success = false;
                bucket_plan.fail_reason = "id_token_axis_empty";
            }
            if (bucket_plan.success) {
                for (size_t ii = 0; ii < bucket_built.input_nodes.size(); ++ii) {
                    if (bucket_plan.input_token_axis[ii] >= 0) {
                        const struct ggml_tensor* inp = bucket_built.input_nodes[ii];
                        if (!inp || !ggml_is_contiguous(inp) ||
                            ggml_type_to_pjrt(inp->type) == cloudtpu::PjrtDType::INVALID) {
                            bucket_plan.success = false;
                            bucket_plan.fail_reason = "non_contiguous_padded_input";
                            break;
                        }
                    }
                }
            }
            if (bucket_plan.success) {
                for (size_t oi = 0; oi < bucket_built.output_nodes.size(); ++oi) {
                    if (bucket_plan.output_token_axis[oi] >= 0) {
                        auto cit = ctx->cgraph_node_idx.find(bucket_built.output_nodes[oi]);
                        if (cit != ctx->cgraph_node_idx.end() &&
                            ctx->liveness.last_consumer[static_cast<size_t>(cit->second)] >= end_idx) {
                            bucket_plan.success = false;
                            bucket_plan.fail_reason = "downstream_cgraph_consumer";
                            break;
                        }
                    }
                }
            }
            if (!bucket_plan.success && ctx->env.verbose >= 2) {
                GGML_LOG_INFO("ggml-cloudtpu: batch bucket skipped [%d..%d) T=%lld B=%lld reason=%s\n",
                              start_idx, end_idx,
                              static_cast<long long>(cand_T),
                              static_cast<long long>(bucket_B),
                              bucket_plan.fail_reason.empty() ? "unknown" : bucket_plan.fail_reason.c_str());
            }
        }
    }

    // 4. Single-pass structural signature + ordered leaf input/output discovery (zero MLIR emission!).
    ctx->scratch_key.words.clear();
    ctx->scratch_tensor_to_id.clear();
    ctx->scratch_slice_computed.clear();
    ctx->scratch_slot_sig_state.clear();
    ctx->scratch_leaf_ssa_uses.clear();
    ctx->scratch_inputs.clear();
    ctx->scratch_input_sharding.clear();
    ctx->scratch_input_scale_dt.clear();
    ctx->scratch_input_q4_plans.clear();
    ctx->scratch_outputs.clear();

    auto& sig = ctx->scratch_key.words;
    sig_push(sig, static_cast<uint64_t>(P));
    sig_push(sig, static_cast<uint64_t>(compute_ops));
    sig_push(sig, (ctx->env.bf16_matmul ? 1ULL : 0ULL) |
                  (ctx->env.async_collectives ? 2ULL : 0ULL) |
                  (ctx->env.donate_kv ? 4ULL : 0ULL) |
                  (ctx->env.q4_split_dot ? 8ULL : 0ULL) |
                  (ctx->env.bf16_flash_attn ? 16ULL : 0ULL) |
                  (ctx->env.q4k_u4 ? 32ULL : 0ULL) |
                  (ctx->env.freetile_mosaic ? 64ULL : 0ULL) |
                  (static_cast<uint64_t>(ctx->env.q4_mode) << 8));
    sig_push(sig, static_cast<uint64_t>(ctx->env.mm_pad_m));
    if (ctx->env.freetile_mosaic) {
        sig_push(sig, static_cast<uint64_t>(ctx->env.freetile_mosaic_version));
    }
    uint32_t next_id = 1;
    uint64_t sig_slot_epoch = 0;

    auto sig_ne = [&](const struct ggml_tensor* t, uint32_t id, int ggml_dim) -> uint64_t {
        if (bucket_plan.success && static_cast<size_t>(id) < bucket_plan.id_token_axis.size()) {
            const int ax = bucket_plan.id_token_axis[static_cast<size_t>(id)];
            if (ax >= 0 && (3 - ax) == ggml_dim) {
                return static_cast<uint64_t>(bucket_plan.bucket_tokens);
            }
        }
        return static_cast<uint64_t>(t->ne[ggml_dim]);
    };

    auto register_input_sig = [&](auto& self, const struct ggml_tensor* t) -> SigCacheEntry {
        if (!t) return {};
        if (ctx->scratch_slice_computed.count(t)) {
            return ctx->scratch_tensor_to_id.at(t);
        }
        const struct ggml_tensor* root_t = unwrap_view_src(t);
        const CloudSlotKey root_sk = make_slot_key(root_t);
        uint64_t cur_epoch = 0;
        if (root_sk.valid()) {
            auto sit = ctx->scratch_slot_sig_state.find(root_sk);
            if (sit != ctx->scratch_slot_sig_state.end()) {
                cur_epoch = sit->second.epoch;
            }
        }
        auto it = ctx->scratch_tensor_to_id.find(t);
        if (it != ctx->scratch_tensor_to_id.end() && it->second.slot_epoch == cur_epoch) {
            return it->second;
        }
        const struct ggml_tensor* parent = view_parent(t);
        if (t->view_src != nullptr && parent != nullptr &&
            (is_dense_supported_type(t->type) || t->type == GGML_TYPE_I64) &&
            t->type == parent->type) {
            if (t->op == GGML_OP_PERMUTE && t->view_offs == parent->view_offs) {
                const SigCacheEntry base = self(self, parent);
                const uint32_t tid = next_id++;
                const int32_t* ax = static_cast<const int32_t*>(t->op_params);
                sig_push(sig, 2ULL);
                sig_push(sig, base.id);
                sig_push(sig, static_cast<uint64_t>(t->type));
                sig_push(sig, sig_ne(t, tid, 0));
                sig_push(sig, sig_ne(t, tid, 1));
                sig_push(sig, sig_ne(t, tid, 2));
                sig_push(sig, sig_ne(t, tid, 3));
                sig_push(sig, (static_cast<uint64_t>(ax[0] & 3) << 0) |
                              (static_cast<uint64_t>(ax[1] & 3) << 4) |
                              (static_cast<uint64_t>(ax[2] & 3) << 8) |
                              (static_cast<uint64_t>(ax[3] & 3) << 12));
                const SigCacheEntry entry{tid, cur_epoch, base.root_leaf_idx};
                ctx->scratch_tensor_to_id[t] = entry;
                return entry;
            }
            if (t->op == GGML_OP_TRANSPOSE && t->view_offs == parent->view_offs) {
                const SigCacheEntry base = self(self, parent);
                const uint32_t tid = next_id++;
                sig_push(sig, 3ULL);
                sig_push(sig, base.id);
                sig_push(sig, static_cast<uint64_t>(t->type));
                sig_push(sig, sig_ne(t, tid, 0));
                sig_push(sig, sig_ne(t, tid, 1));
                sig_push(sig, sig_ne(t, tid, 2));
                sig_push(sig, sig_ne(t, tid, 3));
                const SigCacheEntry entry{tid, cur_epoch, base.root_leaf_idx};
                ctx->scratch_tensor_to_id[t] = entry;
                return entry;
            }
            if (t->view_offs == parent->view_offs && ggml_is_contiguous(t) &&
                ggml_is_contiguous(parent) &&
                ggml_nelements(t) == ggml_nelements(parent)) {
                const SigCacheEntry base = self(self, parent);
                const uint32_t rid = next_id++;
                sig_push(sig, 4ULL);
                sig_push(sig, base.id);
                sig_push(sig, static_cast<uint64_t>(t->type));
                sig_push(sig, sig_ne(t, rid, 0));
                sig_push(sig, sig_ne(t, rid, 1));
                sig_push(sig, sig_ne(t, rid, 2));
                sig_push(sig, sig_ne(t, rid, 3));
                const SigCacheEntry entry{rid, cur_epoch, base.root_leaf_idx};
                ctx->scratch_tensor_to_id[t] = entry;
                return entry;
            }
            ViewSlicePlan vplan;
            const struct ggml_tensor* slice_base = nullptr;
            if (try_build_view_slice(t, &vplan, &slice_base)) {
                const SigCacheEntry base = self(self, slice_base);
                const uint32_t sid = next_id++;
                if (bucket_plan.success &&
                    static_cast<size_t>(base.id) < bucket_plan.id_token_axis.size() &&
                    bucket_plan.id_token_axis[static_cast<size_t>(base.id)] >= 0) {
                    const int64_t parent_dims[4] = {
                        slice_base->ne[3], slice_base->ne[2], slice_base->ne[1], slice_base->ne[0]};
                    if (!cloudtpu::bucket_view_slice_dims(
                            parent_dims,
                            bucket_plan.id_token_axis[static_cast<size_t>(base.id)],
                            bucket_plan.orig_tokens,
                            bucket_plan.bucket_tokens,
                            vplan.reshape_dims,
                            vplan.starts,
                            vplan.limits)) {
                        throw std::runtime_error("ggml-cloudtpu: SLICE bucketing mismatch");
                    }
                }
                sig_push(sig, 5ULL);
                sig_push(sig, base.id);
                sig_push(sig, static_cast<uint64_t>(t->type));
                sig_push(sig, sig_ne(t, sid, 0));
                sig_push(sig, sig_ne(t, sid, 1));
                sig_push(sig, sig_ne(t, sid, 2));
                sig_push(sig, sig_ne(t, sid, 3));
                sig_push(sig, vplan.reshape_dims.size());
                for (int64_t d : vplan.reshape_dims) sig_push(sig, static_cast<uint64_t>(d));
                sig_push(sig, vplan.starts.size());
                for (size_t k = 0; k < vplan.starts.size(); ++k) {
                    sig_push(sig, static_cast<uint64_t>(vplan.starts[k]));
                    sig_push(sig, static_cast<uint64_t>(vplan.limits[k]));
                }
                const SigCacheEntry entry{sid, cur_epoch, base.root_leaf_idx};
                ctx->scratch_tensor_to_id[t] = entry;
                return entry;
            }
        }
        // In-slice SSA forwarding: if this leaf aliases a canonical root already written in-slice
        // (e.g. KV cache after SET_ROWS), resolve to the post-update SSA tensor instead of adding a duplicate leaf.
        const CloudSlotKey leaf_sk = make_slot_key(t);
        if (leaf_sk.valid()) {
            auto sit = ctx->scratch_slot_sig_state.find(leaf_sk);
            if (sit != ctx->scratch_slot_sig_state.end() && sit->second.root != nullptr &&
                is_dense_supported_type(t->type) && t->type == sit->second.root->type &&
                ggml_is_contiguous(t) && ggml_nelements(t) == ggml_nelements(sit->second.root)) {
                const struct ggml_tensor* r = sit->second.root;
                if (t->ne[0] == r->ne[0] && t->ne[1] == r->ne[1] &&
                    t->ne[2] == r->ne[2] && t->ne[3] == r->ne[3]) {
                    const SigCacheEntry entry{sit->second.id, sit->second.epoch, UINT32_MAX};
                    ctx->scratch_tensor_to_id[t] = entry;
                    return entry;
                }
                const uint32_t rid = next_id++;
                sig_push(sig, 4ULL);
                sig_push(sig, sit->second.id);
                sig_push(sig, static_cast<uint64_t>(t->type));
                sig_push(sig, sig_ne(t, rid, 0));
                sig_push(sig, sig_ne(t, rid, 1));
                sig_push(sig, sig_ne(t, rid, 2));
                sig_push(sig, sig_ne(t, rid, 3));
                const SigCacheEntry entry{rid, sit->second.epoch, UINT32_MAX};
                ctx->scratch_tensor_to_id[t] = entry;
                return entry;
            }
        }
        const uint32_t id = next_id++;
        const uint32_t leaf_idx = static_cast<uint32_t>(ctx->scratch_inputs.size());
        const cloudtpu::TensorShardingKind sh = lookup_tensor_sharding(t, P);
        const cloudtpu::PjrtDType sdt = lookup_tensor_scale_dtype(t, ctx->env.q4k_bf16_scales);
        const Q4SlotPlan q4p = lookup_tensor_q4_plan(t, P, sh, ctx->env.q4_mode, ctx->env.q4k_u4);
        const SigCacheEntry entry{id, 0, leaf_idx};
        ctx->scratch_tensor_to_id[t] = entry;
        ctx->scratch_inputs.push_back(t);
        ctx->scratch_input_sharding.push_back(sh);
        ctx->scratch_input_scale_dt.push_back(sdt);
        ctx->scratch_input_q4_plans.push_back(q4p);
        ctx->scratch_leaf_ssa_uses.push_back(0);
        sig_push(sig, 1ULL);
        sig_push(sig, (static_cast<uint64_t>(t->type) << 40) |
                      (static_cast<uint64_t>(q4p.storage) << 32) |
                      (static_cast<uint64_t>(q4p.q_dtype) << 24) |
                      (static_cast<uint64_t>(q4p.bitcast_target) << 16) |
                      (static_cast<uint64_t>(sdt) << 8) |
                      static_cast<uint64_t>(sh));
        sig_push(sig, sig_ne(t, id, 0));
        sig_push(sig, sig_ne(t, id, 1));
        sig_push(sig, sig_ne(t, id, 2));
        sig_push(sig, sig_ne(t, id, 3));
        return entry;
    };

    for (int i = start_idx; i < end_idx; ++i) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (is_skipped_graph_node(node)) {
            continue;
        }
        const size_t rel_idx = static_cast<size_t>(i - start_idx);
        uint32_t sids[GGML_MAX_SRC];
        size_t n_src = 0;
        auto push_sig_src = [&](const struct ggml_tensor* s_ptr) {
            const SigCacheEntry se = register_input_sig(register_input_sig, s_ptr);
            sids[n_src++] = se.id;
            if (se.root_leaf_idx != UINT32_MAX) {
                ctx->scratch_leaf_ssa_uses[se.root_leaf_idx] += 1;
            }
        };
        if (node->op == GGML_OP_ROPE || node->op == GGML_OP_SOFT_MAX) {
            push_sig_src(node->src[0]);
            push_sig_src(node->src[1]);
            if (node->src[2]) push_sig_src(node->src[2]);
        } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            push_sig_src(node->src[0]);
            push_sig_src(node->src[1]);
            push_sig_src(node->src[2]);
            if (node->src[3]) push_sig_src(node->src[3]);
        } else if (node->op == GGML_OP_GLU) {
            push_sig_src(node->src[0]);
            if (node->src[1]) push_sig_src(node->src[1]);
        } else if (node->op == GGML_OP_CPY || node->op == GGML_OP_CONT) {
            // GGML_OP_CPY (including ggml_cast) places the uncomputed destination in src[1];
            // only src[0] is read as a PJRT input.
            push_sig_src(node->src[0]);
        } else {
            for (int s = 0; s < GGML_MAX_SRC && node->src[s] != nullptr; ++s) {
                push_sig_src(node->src[s]);
            }
        }

        const uint32_t dst_id = next_id++;
        ctx->scratch_slice_computed.insert(node);
        ctx->scratch_tensor_to_id[node] = {dst_id, 0, UINT32_MAX};

        bool needs_root_reshape = false;
        const struct ggml_tensor* root = canonical_root(node);
        uint32_t store_sig_id = dst_id;
        if (root != node && (root->ne[0] != node->ne[0] || root->ne[1] != node->ne[1] ||
                             root->ne[2] != node->ne[2] || root->ne[3] != node->ne[3])) {
            needs_root_reshape = true;
            store_sig_id = next_id++;
        }

        const struct ggml_tensor* fwd_root = get_forwardable_inplace_root(node);
        if (fwd_root != nullptr) {
            const CloudSlotKey sk = make_slot_key(fwd_root);
            if (sk.valid()) {
                for (auto sit = ctx->scratch_slot_sig_state.begin(); sit != ctx->scratch_slot_sig_state.end();) {
                    if (!(sit->first == sk) && tensor_ranges_overlap(fwd_root, sit->second.root)) {
                        sit = ctx->scratch_slot_sig_state.erase(sit);
                    } else {
                        ++sit;
                    }
                }
                const uint64_t ep = ++sig_slot_epoch;
                ctx->scratch_slot_sig_state[sk] = {store_sig_id, fwd_root, ep};
            }
        } else if (node->buffer && is_cloudtpu_buffer(node->buffer) && node->data) {
            for (auto sit = ctx->scratch_slot_sig_state.begin(); sit != ctx->scratch_slot_sig_state.end();) {
                if (tensor_ranges_overlap(node, sit->second.root)) {
                    sit = ctx->scratch_slot_sig_state.erase(sit);
                } else {
                    ++sit;
                }
            }
        }

        const uint8_t is_live_out = live_out_flags[rel_idx];
        const uint8_t keep_sh = keep_sharded_flags[rel_idx];
        sig_push(sig, 6ULL);
        sig_push(sig, (static_cast<uint64_t>(node->op) << 16) |
                      (static_cast<uint64_t>(node->type) << 8) |
                      (static_cast<uint64_t>(is_live_out) << 0) |
                      (static_cast<uint64_t>(keep_sh) << 1) |
                      (static_cast<uint64_t>(needs_root_reshape ? 1 : 0) << 2));
        sig_push(sig, sig_ne(node, dst_id, 0));
        sig_push(sig, sig_ne(node, dst_id, 1));
        sig_push(sig, sig_ne(node, dst_id, 2));
        sig_push(sig, sig_ne(node, dst_id, 3));
        sig_push(sig, n_src);
        for (size_t s = 0; s < n_src; ++s) sig_push(sig, sids[s]);
        if (needs_root_reshape && root) {
            sig_push(sig, sig_ne(root, store_sig_id, 0));
            sig_push(sig, sig_ne(root, store_sig_id, 1));
            sig_push(sig, sig_ne(root, store_sig_id, 2));
            sig_push(sig, sig_ne(root, store_sig_id, 3));
        }

        // Exact op-parameter bits
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                const bool high_prec =
                    (ggml_prec_get_acc(node) == GGML_PREC_F32) ||
                    (ggml_prec_get_src(node, 0) == GGML_PREC_F32) ||
                    (ggml_prec_get_src(node, 1) == GGML_PREC_F32);
                sig_push(sig, high_prec ? 1ULL : 0ULL);
                break;
            }
            case GGML_OP_RMS_NORM: {
                float eps = 0.0f;
                std::memcpy(&eps, node->op_params, sizeof(float));
                sig_push_f32(sig, eps);
                break;
            }
            case GGML_OP_ROPE: {
                const int32_t* p = node->op_params;
                sig_push(sig, static_cast<uint64_t>(static_cast<uint32_t>(p[1])));
                sig_push(sig, static_cast<uint64_t>(static_cast<uint32_t>(p[2])));
                sig_push(sig, static_cast<uint64_t>(static_cast<uint32_t>(p[4])));
                sig_push(sig, ggml_rope_get_truncate(node) ? 1ULL : 0ULL);
                for (int pi = 5; pi <= 10; ++pi) {
                    sig_push(sig, static_cast<uint64_t>(static_cast<uint32_t>(p[pi])));
                }
                sig_push(sig, static_cast<uint64_t>(static_cast<uint32_t>(p[15])));
                break;
            }
            case GGML_OP_UNARY:
                sig_push(sig, static_cast<uint64_t>(ggml_get_unary_op(node)));
                break;
            case GGML_OP_GLU:
                sig_push(sig, static_cast<uint64_t>(static_cast<uint32_t>(ggml_get_op_params_i32(node, 1))));
                break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SCALE: {
                const uint32_t* p = reinterpret_cast<const uint32_t*>(node->op_params);
                sig_push(sig, (static_cast<uint64_t>(p[0]) << 32) | static_cast<uint64_t>(p[1]));
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                const uint32_t* p = reinterpret_cast<const uint32_t*>(node->op_params);
                sig_push(sig, (static_cast<uint64_t>(p[0]) << 32) | static_cast<uint64_t>(p[1]));
                sig_push(sig, (static_cast<uint64_t>(p[2]) << 32) |
                              static_cast<uint64_t>(ggml_flash_attn_ext_get_prec(node)));
                break;
            }
            default:
                break;
        }

        if (is_live_out) {
            ctx->scratch_outputs.push_back(node);
        }
    }

    // Identify safe in-place KV-cache buffer donations (SET_ROWS leaf table -> live-out same HBM slot).
    std::vector<int32_t> input_alias_out(ctx->scratch_inputs.size(), -1);
    bool has_alias = false;
    if (ctx->env.donate_kv && !ctx->scratch_inputs.empty() && !ctx->scratch_outputs.empty()) {
        for (size_t oi = 0; oi < ctx->scratch_outputs.size(); ++oi) {
            const struct ggml_tensor* out_node = ctx->scratch_outputs[oi];
            if (!out_node || out_node->op != GGML_OP_SET_ROWS || !out_node->src[2]) {
                continue;
            }
            if (!out_node->buffer || !is_cloudtpu_buffer(out_node->buffer)) {
                continue;
            }
            const struct ggml_tensor* dst_root = canonical_root(out_node);
            const struct ggml_tensor* tbl_in = out_node->src[2];
            const struct ggml_tensor* tbl_root = canonical_root(tbl_in);
            if (!dst_root || !tbl_root || dst_root->buffer != tbl_root->buffer ||
                dst_root->data != tbl_root->data || dst_root->type != tbl_root->type ||
                out_node->type != tbl_in->type || dst_root->type != tbl_in->type ||
                !ggml_is_contiguous(dst_root) || !ggml_is_contiguous(tbl_in) ||
                dst_root->ne[0] != tbl_root->ne[0] || dst_root->ne[1] != tbl_root->ne[1] ||
                dst_root->ne[2] != tbl_root->ne[2] || dst_root->ne[3] != tbl_root->ne[3] ||
                ggml_nelements(tbl_in) != ggml_nelements(tbl_root)) {
                continue;
            }
            if (is_quant_split_type(tbl_in->type)) {
                continue;
            }
            size_t found_in = SIZE_MAX;
            bool dup_slot = false;
            for (size_t ii = 0; ii < ctx->scratch_inputs.size(); ++ii) {
                const struct ggml_tensor* cand = ctx->scratch_inputs[ii];
                const struct ggml_tensor* croot = canonical_root(cand);
                if (croot && croot->buffer == dst_root->buffer && croot->data == dst_root->data) {
                    if (found_in == SIZE_MAX &&
                        cand->type == dst_root->type &&
                        cand->ne[0] == dst_root->ne[0] && cand->ne[1] == dst_root->ne[1] &&
                        cand->ne[2] == dst_root->ne[2] && cand->ne[3] == dst_root->ne[3]) {
                        found_in = ii;
                    } else {
                        dup_slot = true;
                        break;
                    }
                } else if (tensor_ranges_overlap(cand, dst_root)) {
                    dup_slot = true;
                    break;
                }
            }
            for (size_t oj = 0; !dup_slot && oj < ctx->scratch_outputs.size(); ++oj) {
                if (oj != oi && tensor_ranges_overlap(ctx->scratch_outputs[oj], dst_root)) {
                    dup_slot = true;
                }
            }
            if (found_in == SIZE_MAX || dup_slot || input_alias_out[found_in] >= 0 ||
                ctx->scratch_input_sharding[found_in] != cloudtpu::TensorShardingKind::REPLICATED) {
                continue;
            }
            // Verify pre-SET_ROWS leaf input is consumed ONLY once in the lowered SSA program
            // (downstream VIEW/FLASH_ATTN_EXT reads in the same slice consume the forwarded post-SET_ROWS SSA value).
            // Never donate a token-varying non-KV buffer (e.g. mixed_embd) whose H2D upload is bucket-padded.
            const bool bucketed_input =
                (bucket_plan.success && found_in < bucket_plan.input_token_axis.size() &&
                 bucket_plan.input_token_axis[found_in] >= 0);
            if (!bucketed_input && ctx->scratch_leaf_ssa_uses[found_in] == 1) {
                input_alias_out[found_in] = static_cast<int32_t>(oi);
                has_alias = true;
            }
        }
    }
    if (has_alias) {
        sig_push(sig, 7ULL);
        for (size_t ii = 0; ii < input_alias_out.size(); ++ii) {
            if (input_alias_out[ii] >= 0) {
                sig_push(sig, (static_cast<uint64_t>(ii) << 32) | static_cast<uint64_t>(input_alias_out[ii]));
            }
        }
    }

    ctx->scratch_key.hash = hash_sig_words(sig);
    cloudtpu::CachedExecutablePtr exec;
    const CachedSubgraphPlan* plan_ptr = nullptr;
    auto pit = ctx->plan_cache.find(ctx->scratch_key);
    if (pit != ctx->plan_cache.end()) {
        exec = rt.get_cached_exec(pit->second.exec_key);
        if (exec) {
            ctx->plan_hits += 1;
            plan_ptr = &pit->second;
        } else {
            ctx->plan_cache.erase(pit);
        }
    }
    if (plan_ptr == nullptr) {
        ctx->plan_misses += 1;
        BuiltSubgraphSpec built_miss;
        BuiltSubgraphSpec* active_built = &bucket_built;
        if (bucket_plan.success) {
            for (size_t ii = 0; ii < bucket_built.prog_spec.inputs.size(); ++ii) {
                bucket_built.prog_spec.inputs[ii].alias_output_idx = input_alias_out[ii];
            }
            if (!cloudtpu::apply_program_batch_bucket(bucket_built.prog_spec, bucket_plan)) {
                throw std::runtime_error("ggml-cloudtpu: failed to apply batch bucket to program spec");
            }
        } else {
            built_miss = build_subgraph_program_spec(
                ctx, cgraph, start_idx, end_idx, P, live_out_flags, keep_sharded_flags, &input_alias_out, true);
            active_built = &built_miss;
        }
        cloudtpu::GraphProgramSpec& prog_spec = active_built->prog_spec;
        const auto& id_to_input_idx = active_built->id_to_input_idx;

        cloudtpu::EmittedProgram emitted = cloudtpu::emit_stablehlo_program(prog_spec);
        if (ctx->env.verbose >= 3) {
            GGML_LOG_INFO("ggml-cloudtpu: subgraph [%d..%d) spmd=%d inputs=%zu ops=%zu outs=%zu\n%s\n",
                          start_idx, end_idx, P,
                          emitted.input_bindings.size(), prog_spec.ops.size(), emitted.output_bindings.size(),
                          emitted.mlir_text.c_str());
        }
        // Use explicit {tf.aliasing_output} attributes alone (alias_passthrough_params=false) so XLA
        // never implicitly donates an unchecked passthrough copy's source buffer.
        const cloudtpu::PjrtCompileOptions copts =
            rt.make_spmd_compile_options(P, ctx->device_index, /*alias_passthrough=*/false);
        const int compile_dev = (P > 1) ? 0 : ctx->device_index;
        const bool used_mosaic =
            prog_spec.freetile_mosaic && emitted.mlir_text.find("@tpu_custom_call") != std::string::npos;
        if (used_mosaic) {
            try {
                bool compiled_now = false;
                exec = rt.compile_or_get(emitted.cache_key, emitted.mlir_text, compile_dev, copts, &compiled_now);
                if (compiled_now) {
                    ctx->mosaic_compiles += 1;
                }
            } catch (const std::exception& e) {
                ctx->mosaic_fallbacks += 1;
                GGML_LOG_WARN("ggml-cloudtpu: FreeTile Mosaic compile failed (%s); falling back to pure StableHLO\n",
                              e.what());
                prog_spec.freetile_mosaic = false;
                emitted = cloudtpu::emit_stablehlo_program(prog_spec);
                exec = rt.compile_or_get(emitted.cache_key, emitted.mlir_text, compile_dev, copts);
            }
        } else {
            exec = rt.compile_or_get(emitted.cache_key, emitted.mlir_text, compile_dev, copts);
        }

        CachedSubgraphPlan new_plan;
        new_plan.exec_key = exec->key;
        new_plan.num_emitted_ops = prog_spec.ops.size();
        new_plan.has_donation = has_alias;
        new_plan.input_bindings.reserve(emitted.input_bindings.size());
        for (const auto& ib : emitted.input_bindings) {
            new_plan.input_bindings.push_back({id_to_input_idx.at(ib.tensor_id), ib.part, ib.donatable});
        }
        new_plan.out_dtypes.reserve(emitted.output_bindings.size());
        new_plan.out_dims.reserve(emitted.output_bindings.size());
        for (const auto& ob : emitted.output_bindings) {
            new_plan.out_dtypes.push_back(ob.dtype);
            new_plan.out_dims.push_back(ob.dims);
        }

        // Selectively prune any entries evicted by compile_or_get, preserving surviving plans.
        prune_evicted_plans();
        if (ctx->plan_cache.size() >= ctx->env.cache_cap && !ctx->plan_cache.empty()) {
            ctx->plan_cache.erase(ctx->plan_cache.begin());
        }
        auto [ins_it, ok] = ctx->plan_cache.emplace(ctx->scratch_key, std::move(new_plan));
        (void) ok;
        plan_ptr = &ins_it->second;
    }

    const CachedSubgraphPlan& plan = *plan_ptr;

    // Release owning PJRT buffer references from reusable scratch containers on exit (normal
    // or exception) so freeing/resetting GGML buffers immediately reclaims HBM while preserving
    // container capacities across decode steps.
    struct ScratchBufferGuard {
        ggml_backend_cloudtpu_context* c;
        ~ScratchBufferGuard() {
            for (auto& rb : c->scratch_resolved) {
                rb.primary.clear();
                rb.scales.clear();
                rb.mins.clear();
            }
            c->scratch_outputs_1d.clear();
            for (auto& row : c->scratch_spmd_outputs) {
                row.clear();
            }
            c->scratch_raw_inputs.clear();
            c->scratch_non_donatable.clear();
            for (auto& row : c->scratch_spmd_inputs) {
                row.clear();
            }
            c->scratch_spmd_input_ptrs.clear();
        }
    } scratch_guard{ctx};

    // 5. Resolve each unique leaf input ONCE into flat scratch_resolved[].
    const size_t n_in = ctx->scratch_inputs.size();
    ctx->scratch_resolved.resize(n_in);
    for (size_t k = 0; k < n_in; ++k) {
        const int token_ax = bucket_plan.success ? bucket_plan.input_token_axis[k] : -1;
        if (token_ax >= 0) {
            upload_padded_token_input(
                rt, ctx->device_index, P, ctx->scratch_inputs[k],
                token_ax, bucket_plan.bucket_tokens, bucket_plan.input_pad_kind[k],
                ctx->scratch_resolved[k]);
        } else {
            resolve_tensor_hbm(
                rt, ctx->device_index, P, ctx->env.q4k_bf16_scales,
                ctx->env.q4_mode, ctx->env.q4k_u4,
                ctx->scratch_inputs[k], ctx->scratch_resolved[k]);
        }
    }

    // Keep TPU execution fully pipelined across intermediate subgraphs when fuse_graphs is enabled;
    // synchronize on the final subgraph, when fuse_graphs=0, or when ready events are unavailable.
    const bool sync_exec = is_final_subgraph || !ctx->env.fuse_graphs || !rt.has_ready_event();
    ctx->scratch_non_donatable.clear();
    std::vector<cloudtpu::PjrtEventHandle> new_events;
    if (P <= 1) {
        ctx->scratch_raw_inputs.clear();
        ctx->scratch_raw_inputs.reserve(plan.input_bindings.size());
        for (const auto& ib : plan.input_bindings) {
            const auto& rp = ctx->scratch_resolved[ib.input_idx];
            const auto& vec = (ib.part == 0) ? rp.primary : ((ib.part == 1) ? rp.scales : rp.mins);
            ctx->scratch_raw_inputs.push_back(vec[0]->handle.get());
        }
        if (plan.has_donation) {
            for (size_t bi = 0; bi < plan.input_bindings.size(); ++bi) {
                const auto& ib = plan.input_bindings[bi];
                if (!ib.donatable) {
                    ctx->scratch_non_donatable.push_back(static_cast<int64_t>(bi));
                    continue;
                }
                const auto& rp = ctx->scratch_resolved[ib.input_idx];
                // Exclusive ownership check: 1 ref in bctx->slots + 1 ref in scratch_resolved == 2,
                // and raw PJRT_Buffer* must not appear twice in argument list.
                bool safe = (rp.primary.size() == 1 && rp.primary[0] && rp.primary[0].use_count() == 2);
                if (safe) {
                    PJRT_Buffer* raw_ptr = ctx->scratch_raw_inputs[bi];
                    for (size_t bj = 0; bj < ctx->scratch_raw_inputs.size(); ++bj) {
                        if (bj != bi && ctx->scratch_raw_inputs[bj] == raw_ptr) {
                            safe = false;
                            break;
                        }
                    }
                }
                if (!safe) {
                    ++ctx->total_kv_donation_skipped;
                    ctx->scratch_non_donatable.push_back(static_cast<int64_t>(bi));
                } else {
                    ++ctx->total_kv_donated;
                }
            }
        }
        rt.execute(exec,
                   ctx->device_index,
                   ctx->scratch_raw_inputs.data(),
                   ctx->scratch_raw_inputs.size(),
                   ctx->scratch_outputs_1d,
                   plan.out_dtypes,
                   plan.out_dims,
                   sync_exec,
                   ctx->scratch_non_donatable.empty() ? nullptr : &ctx->scratch_non_donatable);

        if (!sync_exec) {
            new_events.reserve(ctx->scratch_outputs_1d.size());
            for (const auto& buf : ctx->scratch_outputs_1d) {
                if (buf && buf->handle) {
                    cloudtpu::PjrtEventHandle ev = rt.create_buffer_ready_event(buf->handle.get());
                    if (ev) {
                        new_events.push_back(std::move(ev));
                    }
                }
            }
        }
        for (size_t oi = 0; oi < ctx->scratch_outputs.size(); ++oi) {
            const int out_ax = bucket_plan.success ? bucket_plan.output_token_axis[oi] : -1;
            store_tensor_hbm_single(rt, ctx->scratch_outputs[oi], std::move(ctx->scratch_outputs_1d[oi]), out_ax);
        }
        if (plan.has_donation) {
            for (const auto& ib : plan.input_bindings) {
                if (ib.donatable) {
                    ctx->scratch_resolved[ib.input_idx].primary.clear();
                }
            }
        }
    } else {
        const size_t n_devs = static_cast<size_t>(P);
        ctx->scratch_spmd_inputs.resize(n_devs);
        ctx->scratch_spmd_input_ptrs.resize(n_devs);
        for (size_t d = 0; d < n_devs; ++d) {
            auto& row = ctx->scratch_spmd_inputs[d];
            row.clear();
            row.reserve(plan.input_bindings.size());
            for (const auto& ib : plan.input_bindings) {
                const auto& rp = ctx->scratch_resolved[ib.input_idx];
                const auto& vec = (ib.part == 0) ? rp.primary : ((ib.part == 1) ? rp.scales : rp.mins);
                row.push_back(vec[d]->handle.get());
            }
            ctx->scratch_spmd_input_ptrs[d] = row.data();
        }
        if (plan.has_donation) {
            for (size_t bi = 0; bi < plan.input_bindings.size(); ++bi) {
                const auto& ib = plan.input_bindings[bi];
                if (!ib.donatable) {
                    ctx->scratch_non_donatable.push_back(static_cast<int64_t>(bi));
                    continue;
                }
                const auto& rp = ctx->scratch_resolved[ib.input_idx];
                bool safe = (rp.primary.size() == n_devs);
                for (size_t d = 0; safe && d < n_devs; ++d) {
                    if (!rp.primary[d] || rp.primary[d].use_count() != 2) {
                        safe = false;
                    }
                    PJRT_Buffer* raw_ptr = ctx->scratch_spmd_inputs[d][bi];
                    for (size_t bj = 0; safe && bj < plan.input_bindings.size(); ++bj) {
                        if (bj != bi && ctx->scratch_spmd_inputs[d][bj] == raw_ptr) {
                            safe = false;
                        }
                    }
                }
                if (!safe) {
                    ++ctx->total_kv_donation_skipped;
                    ctx->scratch_non_donatable.push_back(static_cast<int64_t>(bi));
                } else {
                    ++ctx->total_kv_donated;
                }
            }
        }
        rt.execute_spmd(exec,
                        P,
                        ctx->scratch_spmd_input_ptrs.data(),
                        plan.input_bindings.size(),
                        ctx->scratch_spmd_outputs,
                        plan.out_dtypes,
                        plan.out_dims,
                        sync_exec,
                        ctx->scratch_non_donatable.empty() ? nullptr : &ctx->scratch_non_donatable);

        if (!sync_exec) {
            new_events.reserve(n_devs * ctx->scratch_outputs.size());
            for (size_t d = 0; d < n_devs; ++d) {
                for (const auto& buf : ctx->scratch_spmd_outputs[d]) {
                    if (buf && buf->handle) {
                        cloudtpu::PjrtEventHandle ev = rt.create_buffer_ready_event(buf->handle.get());
                        if (ev) {
                            new_events.push_back(std::move(ev));
                        }
                    }
                }
            }
        }
        for (size_t oi = 0; oi < ctx->scratch_outputs.size(); ++oi) {
            const int out_ax = bucket_plan.success ? bucket_plan.output_token_axis[oi] : -1;
            store_tensor_hbm_spmd(rt, ctx->scratch_outputs[oi], ctx->scratch_spmd_outputs, oi, out_ax);
        }
        if (plan.has_donation) {
            for (const auto& ib : plan.input_bindings) {
                if (ib.donatable) {
                    ctx->scratch_resolved[ib.input_idx].primary.clear();
                }
            }
        }
    }
    if (sync_exec) {
        ggml_backend_cloudtpu_synchronize_ctx(ctx);
    } else if (!new_events.empty()) {
        // Track lightweight PJRT ready events per launch instead of owning PJRT_Buffer references
        // so GGML slot reuse frees HBM immediately while capping outstanding launches (not raw
        // per-chip/per-output events) so a single SPMD launch never forces waiting on itself.
        std::deque<std::vector<cloudtpu::PjrtEventHandle>> overflow_launches =
            ctx->pending_sync_launches.enqueue(std::move(new_events));
        await_sync_launches(rt, overflow_launches);
    }

    ctx->total_subgraphs += 1;
    if (bucket_plan.success) {
        ctx->total_bucketed_subgraphs += 1;
    }
    ctx->total_ops_executed += plan.num_emitted_ops;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cloudtpu_synchronize(ggml_backend_t backend) {
    auto* ctx = static_cast<ggml_backend_cloudtpu_context*>(backend->context);
    try {
        ggml_backend_cloudtpu_synchronize_ctx(ctx);
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: synchronize failed: %s\n", e.what());
    }
}

static enum ggml_status ggml_backend_cloudtpu_graph_compute(ggml_backend_t backend,
                                                            struct ggml_cgraph* cgraph) {
    auto* ctx = static_cast<ggml_backend_cloudtpu_context*>(backend->context);
    ctx->total_graph_evals += 1;

    const int max_fuse = std::max(1, ctx->env.max_fuse_ops);
    int i = 0;
    try {
        compute_cgraph_liveness(ctx, cgraph);

        int last_exec_idx = -1;
        for (int k = cgraph->n_nodes - 1; k >= 0; --k) {
            if (!is_skipped_graph_node(cgraph->nodes[k])) {
                last_exec_idx = k;
                break;
            }
        }

        cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
        while (i < cgraph->n_nodes) {
            if (is_skipped_graph_node(cgraph->nodes[i])) {
                ++i;
                continue;
            }
            if (!ggml_backend_cloudtpu_device_supports_op_impl(cgraph->nodes[i])) {
                GGML_LOG_ERROR("ggml-cloudtpu: unsupported op %s in graph_compute\n",
                               ggml_op_desc(cgraph->nodes[i]));
                return GGML_STATUS_FAILED;
            }
            int j = i + 1;
            if (ctx->env.fuse_graphs) {
                // Isolate attention (FLASH_ATTN_EXT) at subgraph boundaries when CLOUDTPU_FUSE_ATTN=0.
                const bool start_is_attn = (!ctx->env.fuse_attn && cgraph->nodes[i]->op == GGML_OP_FLASH_ATTN_EXT);
                if (!start_is_attn) {
                    struct SliceWriteRecord {
                        const struct ggml_tensor* node = nullptr;
                        const struct ggml_tensor* root = nullptr;
                        CloudSlotKey slot_key;
                        int cgraph_idx = -1;
                        bool forwardable = false;
                        bool active_alias_target = true;
                    };
                    std::vector<SliceWriteRecord> inplace_writes;
                    std::unordered_map<CloudSlotKey, size_t, CloudSlotKeyHash> latest_slot_write;
                    std::vector<std::pair<const struct ggml_tensor*, int>> slice_writes;
                    auto record_slice_node = [&](const struct ggml_tensor* node, int idx) {
                        slice_writes.push_back({node, idx});
                        if (node->data) {
                            for (auto& prev : inplace_writes) {
                                if (prev.active_alias_target && tensor_ranges_overlap(node, prev.node)) {
                                    prev.active_alias_target = false;
                                    if (prev.forwardable && prev.slot_key.valid()) {
                                        auto lit = latest_slot_write.find(prev.slot_key);
                                        if (lit != latest_slot_write.end() && &inplace_writes[lit->second] == &prev) {
                                            latest_slot_write.erase(lit);
                                        }
                                    }
                                }
                            }
                        }
                        if (is_alias_tracked_write(node)) {
                            SliceWriteRecord rec;
                            rec.node = node;
                            rec.cgraph_idx = idx;
                            rec.root = get_forwardable_inplace_root(node);
                            rec.forwardable = (rec.root != nullptr);
                            if (rec.forwardable) {
                                rec.slot_key = make_slot_key(rec.root);
                            }
                            const size_t rec_idx = inplace_writes.size();
                            inplace_writes.push_back(rec);
                            if (rec.forwardable && rec.slot_key.valid()) {
                                latest_slot_write[rec.slot_key] = rec_idx;
                            }
                        }
                    };
                    record_slice_node(cgraph->nodes[i], i);
                    int fused_ops = 1;
                    while (j < cgraph->n_nodes && fused_ops < max_fuse) {
                        const struct ggml_tensor* nj = cgraph->nodes[j];
                        if (is_skipped_graph_node(nj)) {
                            ++j;
                            continue;
                        }
                        if (!ggml_backend_cloudtpu_device_supports_op_impl(nj)) {
                            break;
                        }
                        if (!ctx->env.fuse_attn && nj->op == GGML_OP_FLASH_ATTN_EXT) {
                            break;
                        }
                        // Allow fusing downstream VIEW/PERMUTE/RESHAPE/SLICE reads (e.g. FLASH_ATTN_EXT
                        // reading KV cache after SET_ROWS) via in-slice SSA forwarding when the prior
                        // write cleanly updates the full canonical root; split on unforwardable
                        // host-backed/partial/overlapping writes or stale pre-overwrite dependencies.
                        bool alias_conflict = false;
                        const bool nj_is_alias_write = is_alias_tracked_write(nj);
                        for (const auto& pw : slice_writes) {
                            if (tensor_ranges_overlap(nj, pw.first) && !tensor_range_covers(nj, pw.first)) {
                                if (nj_is_alias_write || !is_dead_gallocr_scratch(ctx, pw.first, pw.second, j)) {
                                    alias_conflict = true;
                                    break;
                                }
                            }
                        }
                        if (!alias_conflict && !inplace_writes.empty()) {
                            int max_check_src = GGML_MAX_SRC;
                            if (nj->op == GGML_OP_ROPE || nj->op == GGML_OP_SOFT_MAX) {
                                max_check_src = 3;
                            } else if (nj->op == GGML_OP_FLASH_ATTN_EXT) {
                                max_check_src = 4;
                            } else if (nj->op == GGML_OP_GLU) {
                                max_check_src = 2;
                            } else if (nj->op == GGML_OP_CPY || nj->op == GGML_OP_CONT) {
                                max_check_src = 1;
                            }
                            for (int s = 0; s < max_check_src && nj->src[s] != nullptr && !alias_conflict; ++s) {
                                const struct ggml_tensor* src = nj->src[s];
                                const struct ggml_tensor* leaf_base = nullptr;
                                bool resolved_direct_ssa = false;
                                for (const struct ggml_tensor* cur = src; cur != nullptr; cur = view_parent(cur)) {
                                    if (!is_skipped_graph_node(cur)) {
                                        auto nit = ctx->cgraph_node_idx.find(cur);
                                        if (nit != ctx->cgraph_node_idx.end() && nit->second >= i && nit->second < j) {
                                            const int cur_idx = nit->second;
                                            bool overwritten_by_later_iw = false;
                                            for (const auto& iw : inplace_writes) {
                                                if (iw.node == cur) {
                                                    if (!iw.active_alias_target) {
                                                        overwritten_by_later_iw = true;
                                                    }
                                                    break;
                                                }
                                                if (iw.cgraph_idx > cur_idx && iw.active_alias_target &&
                                                    tensor_ranges_overlap(cur, iw.node)) {
                                                    overwritten_by_later_iw = true;
                                                    break;
                                                }
                                            }
                                            if (overwritten_by_later_iw) {
                                                alias_conflict = true;
                                            }
                                            resolved_direct_ssa = true;
                                            break;
                                        }
                                    }
                                    if (view_parent(cur) == nullptr) {
                                        leaf_base = cur;
                                        break;
                                    }
                                }
                                if (!resolved_direct_ssa && !alias_conflict) {
                                    bool has_overlap = false;
                                    for (const auto& iw : inplace_writes) {
                                        if (iw.active_alias_target &&
                                            (tensor_ranges_overlap(src, iw.node) ||
                                             (leaf_base && tensor_ranges_overlap(leaf_base, iw.node)))) {
                                            has_overlap = true;
                                            break;
                                        }
                                    }
                                    if (has_overlap) {
                                        bool can_forward = false;
                                        const CloudSlotKey base_key = make_slot_key(leaf_base);
                                        auto lit = base_key.valid() ? latest_slot_write.find(base_key)
                                                                    : latest_slot_write.end();
                                        if (lit != latest_slot_write.end()) {
                                            const auto& latest_iw = inplace_writes[lit->second];
                                            if (latest_iw.active_alias_target && latest_iw.forwardable &&
                                                latest_iw.root != nullptr &&
                                                is_dense_supported_type(leaf_base->type) &&
                                                leaf_base->type == latest_iw.root->type &&
                                                ggml_is_contiguous(leaf_base) &&
                                                ggml_nelements(leaf_base) == ggml_nelements(latest_iw.root)) {
                                                bool other_active_overlap = false;
                                                for (size_t wi = 0; wi < inplace_writes.size(); ++wi) {
                                                    if (wi == lit->second) continue;
                                                    const auto& iw = inplace_writes[wi];
                                                    if (iw.active_alias_target &&
                                                        (tensor_ranges_overlap(src, iw.node) ||
                                                         tensor_ranges_overlap(leaf_base, iw.node))) {
                                                        other_active_overlap = true;
                                                        break;
                                                    }
                                                }
                                                if (!other_active_overlap &&
                                                    can_collapse_view_chain_to_leaf(src, leaf_base)) {
                                                    can_forward = true;
                                                }
                                            }
                                        }
                                        if (!can_forward) {
                                            alias_conflict = true;
                                        }
                                    }
                                }
                            }
                        }
                        if (alias_conflict) {
                            break;
                        }
                        record_slice_node(nj, j);
                        ++j;
                        ++fused_ops;
                    }
                    while (j > i + 1 && is_skipped_graph_node(cgraph->nodes[j - 1])) {
                        --j;
                    }
                }
            }
            const bool is_final_subgraph = (j > last_exec_idx);
            const enum ggml_status st = execute_subgraph(ctx, rt, cgraph, i, j, is_final_subgraph);
            if (st != GGML_STATUS_SUCCESS) {
                return st;
            }
            i = j;
        }
        ggml_backend_cloudtpu_synchronize_ctx(ctx);
        return GGML_STATUS_SUCCESS;
    } catch (const CloudTpuAllocError& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: graph_compute allocation error at node %d (%s '%s'): %s\n",
                       i,
                       (i < cgraph->n_nodes && cgraph->nodes[i]) ? ggml_op_desc(cgraph->nodes[i]) : "?",
                       (i < cgraph->n_nodes && cgraph->nodes[i]) ? cgraph->nodes[i]->name : "?",
                       e.what());
        return GGML_STATUS_ALLOC_FAILED;
    } catch (const cloudtpu::PjrtError& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: graph_compute PJRT error (code=%d) at node %d (%s '%s'): %s\n",
                       static_cast<int>(e.code()),
                       i,
                       (i < cgraph->n_nodes && cgraph->nodes[i]) ? ggml_op_desc(cgraph->nodes[i]) : "?",
                       (i < cgraph->n_nodes && cgraph->nodes[i]) ? cgraph->nodes[i]->name : "?",
                       e.what());
        return e.is_resource_exhausted() ? GGML_STATUS_ALLOC_FAILED : GGML_STATUS_FAILED;
    } catch (const std::bad_alloc& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: graph_compute host OOM at node %d (%s '%s'): %s\n",
                       i,
                       (i < cgraph->n_nodes && cgraph->nodes[i]) ? ggml_op_desc(cgraph->nodes[i]) : "?",
                       (i < cgraph->n_nodes && cgraph->nodes[i]) ? cgraph->nodes[i]->name : "?",
                       e.what());
        return GGML_STATUS_ALLOC_FAILED;
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: graph_compute error at node %d (%s '%s'): %s\n",
                       i,
                       (i < cgraph->n_nodes && cgraph->nodes[i]) ? ggml_op_desc(cgraph->nodes[i]) : "?",
                       (i < cgraph->n_nodes && cgraph->nodes[i]) ? cgraph->nodes[i]->name : "?",
                       e.what());
        return GGML_STATUS_FAILED;
    }
}

static const char* ggml_backend_cloudtpu_get_name(ggml_backend_t backend) {
    return ggml_backend_dev_name(backend->device);
}

static void ggml_backend_cloudtpu_free(ggml_backend_t backend) {
    ggml_backend_cloudtpu_synchronize(backend);
    auto* ctx = static_cast<ggml_backend_cloudtpu_context*>(backend->context);
    delete ctx;
    delete backend;
}

static struct ggml_backend_i ggml_backend_cloudtpu_i = {
    /* .get_name            = */ ggml_backend_cloudtpu_get_name,
    /* .free                = */ ggml_backend_cloudtpu_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ ggml_backend_cloudtpu_synchronize,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_cloudtpu_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid_t ggml_backend_cloudtpu_guid(void) {
    static const uint8_t guid[16] = {
        0x43, 0x4c, 0x4f, 0x55, 0x44, 0x54, 0x50, 0x55, // "CLOUDTPU"
        0x50, 0x4a, 0x52, 0x54, 0x00, 0x00, 0x00, 0x01  // "PJRT\0\0\0\1"
    };
    return const_cast<ggml_guid_t>(&guid);
}

// =========================================================================
// Device & Registry Interface
// =========================================================================

struct cloudtpu_dev_context {
    int device_index = 0;
    int spmd_chips = 1;
    std::string name;
    std::string description;
};

static const char* ggml_backend_cloudtpu_device_get_name(ggml_backend_dev_t dev) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return dc->name.c_str();
}

static const char* ggml_backend_cloudtpu_device_get_description(ggml_backend_dev_t dev) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return dc->description.c_str();
}

static void ggml_backend_cloudtpu_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    try {
        cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
        if (dc->spmd_chips > 1) {
            size_t min_free = SIZE_MAX;
            size_t min_total = SIZE_MAX;
            for (int i = 0; i < dc->spmd_chips; ++i) {
                size_t f = 0, t = 0;
                if (!rt.query_memory_stats(i, &f, &t)) {
                    *free = 0;
                    *total = 0;
                    return;
                }
                min_free = std::min(min_free, f);
                min_total = std::min(min_total, t);
            }
            // Symmetric SPMD allocations are bounded by the most constrained chip (min_free).
            // Reporting per-chip min_free pairs with get_alloc_size_n / alloc_buffer_n charging
            // sharded weights 1/P per chip and replicated KV/compute buffers 1x per chip (including
            // non-expandable compute ops like ADD/MUL/RMS_NORM and per-chip --fit-target margins).
            *free = min_free;
            *total = min_total;
            return;
        }
        if (!rt.query_memory_stats(dc->device_index, free, total)) {
            *free = 0;
            *total = 0;
        }
    } catch (const std::exception&) {
        *free = 0;
        *total = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_cloudtpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cloudtpu_device_get_props(ggml_backend_dev_t dev,
                                                   struct ggml_backend_dev_props* props) {
    props->name = ggml_backend_cloudtpu_device_get_name(dev);
    props->description = ggml_backend_cloudtpu_device_get_description(dev);
    props->type = ggml_backend_cloudtpu_device_get_type(dev);
    ggml_backend_cloudtpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_cloudtpu_device_init_backend(ggml_backend_dev_t dev,
                                                                const char* params) {
    GGML_UNUSED(params);
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return ggml_backend_cloudtpu_init_device(dc->device_index);
}

static ggml_backend_buffer_type_t ggml_backend_cloudtpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return ggml_backend_cloudtpu_buffer_type(dc->device_index);
}

static bool ggml_backend_cloudtpu_device_supports_op(ggml_backend_dev_t dev,
                                                     const struct ggml_tensor* op) {
    GGML_UNUSED(dev);
    return ggml_backend_cloudtpu_device_supports_op_impl(op, /*allow_probe=*/true);
}

static bool ggml_backend_cloudtpu_device_supports_buft(ggml_backend_dev_t dev,
                                                       ggml_backend_buffer_type_t buft) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return buft == ggml_backend_cloudtpu_buffer_type(dc->device_index);
}

static const struct ggml_backend_device_i ggml_backend_cloudtpu_device_i = {
    /* .get_name             = */ ggml_backend_cloudtpu_device_get_name,
    /* .get_description      = */ ggml_backend_cloudtpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cloudtpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cloudtpu_device_get_type,
    /* .get_props            = */ ggml_backend_cloudtpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cloudtpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cloudtpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_cloudtpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cloudtpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

struct cloudtpu_reg_state {
    std::mutex mu;
    bool initialized = false;
    std::vector<std::unique_ptr<cloudtpu_dev_context>> dev_ctxs;
    std::vector<ggml_backend_device> devices;
    std::vector<std::unique_ptr<cloudtpu_buft_context>> buft_ctxs;
    std::vector<ggml_backend_buffer_type> bufts;
};

static cloudtpu_reg_state& get_reg_state(ggml_backend_reg_t reg) {
    static cloudtpu_reg_state state;
    std::lock_guard<std::mutex> lock(state.mu);
    if (!state.initialized) {
        try {
            const cloudtpu_env_config env = load_env_config();
            cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
            const int spmd = resolve_spmd_chips(env, rt);
            // When SPMD spans multiple chips (e.g. v6e-8 or tpu7x-4t), expose one unified
            // logical CLOUDTPU device reporting per-chip HBM capacity (with sharded weights
            // charged 1/P per chip and replicated KV/compute buffers charged 1x per chip).
            const int count = (spmd > 1) ? 1 : std::max(1, rt.device_count());
            state.dev_ctxs.reserve(static_cast<size_t>(count));
            state.devices.resize(static_cast<size_t>(count));
            state.buft_ctxs.reserve(static_cast<size_t>(count));
            state.bufts.resize(static_cast<size_t>(count));

            for (int i = 0; i < count; ++i) {
                auto dc = std::make_unique<cloudtpu_dev_context>();
                dc->device_index = i;
                dc->spmd_chips = spmd;
                dc->name = (i == 0) ? "CLOUDTPU" : ("CLOUDTPU" + std::to_string(i));
                if (spmd > 1) {
                    dc->description = "SPMD " + std::to_string(spmd) + "x " + rt.device_info(0).kind;
                } else if (i < rt.device_count()) {
                    dc->description = rt.device_info(i).kind;
                } else {
                    dc->description = "Google Cloud TPU";
                }

                state.devices[static_cast<size_t>(i)] = {
                    /* .iface   = */ ggml_backend_cloudtpu_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ dc.get(),
                };
                state.dev_ctxs.push_back(std::move(dc));

                auto btc = std::make_unique<cloudtpu_buft_context>();
                btc->device_index = i;
                btc->spmd_chips = spmd;
                btc->spmd_min_rows = env.spmd_min_rows;
                btc->q4k_bf16_scales = env.q4k_bf16_scales;
                btc->q4_mode = env.q4_mode;
                btc->q4k_u4 = env.q4k_u4;
                btc->name = state.dev_ctxs[static_cast<size_t>(i)]->name;
                state.bufts[static_cast<size_t>(i)] = {
                    /* .iface   = */ {
                        /* .get_name         = */ ggml_backend_cloudtpu_buffer_type_get_name,
                        /* .alloc_buffer     = */ ggml_backend_cloudtpu_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n   = */ ggml_backend_cloudtpu_buffer_type_alloc_buffer_n,
                        /* .get_alignment    = */ ggml_backend_cloudtpu_buffer_type_get_alignment,
                        /* .get_max_size     = */ NULL,
                        /* .get_alloc_size   = */ ggml_backend_cloudtpu_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n = */ ggml_backend_cloudtpu_buffer_type_get_alloc_size_n,
                        /* .is_host          = */ ggml_backend_cloudtpu_buffer_type_is_host,
                    },
                    /* .device  = */ &state.devices[static_cast<size_t>(i)],
                    /* .context = */ btc.get(),
                };
                state.buft_ctxs.push_back(std::move(btc));
            }
            state.initialized = true;
        } catch (const std::exception& e) {
            GGML_LOG_ERROR("ggml-cloudtpu: failed to initialize PJRT runtime (%s); 0 devices exposed\n",
                           e.what());
            state.devices.clear();
            state.bufts.clear();
            state.dev_ctxs.clear();
            state.buft_ctxs.clear();
            state.initialized = false;
        }
    }
    return state;
}

static const char* ggml_backend_cloudtpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "CLOUDTPU";
}

static size_t ggml_backend_cloudtpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return get_reg_state(reg).devices.size();
}

static ggml_backend_dev_t ggml_backend_cloudtpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    cloudtpu_reg_state& st = get_reg_state(reg);
    GGML_ASSERT(index < st.devices.size());
    return &st.devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_cloudtpu_reg_i = {
    /* .get_name         = */ ggml_backend_cloudtpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cloudtpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cloudtpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

}  // namespace

// =========================================================================
// Public C API
// =========================================================================

int ggml_backend_cloudtpu_get_device_count(void) {
    return static_cast<int>(ggml_backend_reg_dev_count(ggml_backend_cloudtpu_reg()));
}

ggml_backend_buffer_type_t ggml_backend_cloudtpu_buffer_type(int device_index) {
    cloudtpu_reg_state& st = get_reg_state(ggml_backend_cloudtpu_reg());
    if (device_index < 0 || static_cast<size_t>(device_index) >= st.bufts.size()) {
        return nullptr;
    }
    return &st.bufts[static_cast<size_t>(device_index)];
}

ggml_backend_t ggml_backend_cloudtpu_init_device(int device_index) {
    const cloudtpu_env_config env = load_env_config();
    g_hot_env.store(env);

    cloudtpu::PjrtRuntime* rt_ptr = nullptr;
    try {
        rt_ptr = &ensure_runtime_initialized();
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: runtime init failed: %s\n", e.what());
        return nullptr;
    }
    cloudtpu::PjrtRuntime& rt = *rt_ptr;

    ggml_backend_reg_t reg = ggml_backend_cloudtpu_reg();
    const size_t reg_count = ggml_backend_reg_dev_count(reg);
    if (device_index < 0 || static_cast<size_t>(device_index) >= reg_count) {
        GGML_LOG_ERROR("ggml-cloudtpu: invalid device_index=%d (exposed_devices=%zu)\n",
                       device_index, reg_count);
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, static_cast<size_t>(device_index));
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    const int spmd = dc->spmd_chips;

    auto* ctx = new ggml_backend_cloudtpu_context();
    ctx->device_index = dc->device_index;
    ctx->spmd_chips = spmd;
    ctx->env = env;
    ctx->env.spmd_chips = spmd;

    if (env.verbose > 0) {
        const auto& info = rt.device_info(dc->device_index);
        GGML_LOG_INFO(
            "ggml-cloudtpu: initialized device %d (%s, platform=%s %s, spmd_chips=%d, mock=%d, fuse=%d, max_fuse=%d, hbm_limit=%lld)\n",
            dc->device_index,
            info.kind.c_str(),
            rt.platform_name().c_str(),
            rt.platform_version().c_str(),
            spmd,
            rt.is_mock() ? 1 : 0,
            env.fuse_graphs ? 1 : 0,
            env.max_fuse_ops,
            static_cast<long long>(info.hbm_bytes_limit) * static_cast<long long>(spmd));
    }

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_cloudtpu_guid(),
        /* .iface   = */ ggml_backend_cloudtpu_i,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    return backend;
}

ggml_backend_t ggml_backend_cloudtpu_init(void) {
    return ggml_backend_cloudtpu_init_device(0);
}

bool ggml_backend_is_cloudtpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cloudtpu_guid());
}

bool ggml_backend_cloudtpu_is_mock(ggml_backend_t backend) {
    if (!ggml_backend_is_cloudtpu(backend)) {
        return true;
    }
    cloudtpu::PjrtRuntime& rt = cloudtpu::shared_pjrt_runtime(nullptr);
    return !rt.is_initialized() || rt.is_mock();
}

ggml_backend_reg_t ggml_backend_cloudtpu_reg(void) {
    static struct ggml_backend_reg ggml_backend_cloudtpu_reg_s = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cloudtpu_reg_i,
        /* .context     = */ NULL,
    };
    return &ggml_backend_cloudtpu_reg_s;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cloudtpu_reg)
