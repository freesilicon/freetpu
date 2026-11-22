#include "ggml-cloudtpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "pjrt_runtime.hpp"
#include "stablehlo_emitter.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t CLOUDTPU_BUFFER_ALIGNMENT = 64;

struct cloudtpu_env_config {
    bool mock_mode = false;
    int verbose = 0;
    bool fuse_graphs = true;
    uint32_t cache_cap = 4096;
    uint64_t min_elements = 0;
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
    if (const char* v = std::getenv("CLOUDTPU_VERBOSE")) {
        cfg.verbose = std::atoi(v);
    }
    if (const char* v = std::getenv("CLOUDTPU_FUSE_GRAPHS")) {
        cfg.fuse_graphs = (std::atoi(v) != 0);
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
    if (const char* v = std::getenv("LIBTPU_PATH")) {
        cfg.libtpu_path = v;
    } else if (const char* v2 = std::getenv("TPU_LIBRARY_PATH")) {
        cfg.libtpu_path = v2;
    }
    return cfg;
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
    rt_cfg.verbose = env.verbose;
    rt_cfg.exec_cache_cap = env.cache_cap;
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

static bool try_build_view_slice(const struct ggml_tensor* t, ViewSlicePlan* plan) {
    const struct ggml_tensor* base = view_parent(t);
    if (!t || !base || t->type != base->type || !is_dense_supported_type(t->type)) {
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
        return true;
    }

    // Case 2: nested strided 4D slice (covers llama_kv_cache::get_k / get_v and padded views).
    if (t->nb[0] != el) {
        return false;
    }
    if ((t->nb[1] % el) != 0 || (t->nb[2] % el) != 0 || (t->nb[3] % el) != 0) {
        return false;
    }
    const int64_t s1 = std::max<int64_t>(t->ne[0], static_cast<int64_t>(t->nb[1] / el));
    const int64_t s2 = (t->ne[2] == 1 && t->ne[3] == 1)
                           ? std::max<int64_t>(s1 * t->ne[1], static_cast<int64_t>(t->nb[2] / el))
                           : static_cast<int64_t>(t->nb[2] / el);
    const int64_t s3 = (t->ne[3] == 1)
                           ? std::max<int64_t>(s2 * t->ne[2], static_cast<int64_t>(t->nb[3] / el))
                           : static_cast<int64_t>(t->nb[3] / el);
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

static bool is_supported_input_layout(const struct ggml_tensor* t, int depth = 0) {
    if (!t || depth > 8) {
        return false;
    }
    if (t->view_src == nullptr) {
        return ggml_is_contiguous(t);
    }
    if (is_quant_split_type(t->type)) {
        return is_supported_output_layout(t);
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
        if (!try_build_view_slice(t, nullptr)) {
            return false;
        }
        return is_supported_input_layout(parent, depth + 1);
    }
    return is_supported_output_layout(t);
}

// =========================================================================
// HBM Buffer Management (PJRT_Buffer* Tensor Slot Table)
// =========================================================================

struct CloudTpuTensorSlot {
    size_t offset = 0;
    size_t nbytes = 0;
    enum ggml_type ggml_dt = GGML_TYPE_F32;
    cloudtpu::TensorStorageKind storage = cloudtpu::TensorStorageKind::DENSE;
    cloudtpu::PjrtDType dtype = cloudtpu::PjrtDType::F32;
    int64_t dims[4] = {1, 1, 1, 1};
    cloudtpu::PjrtBufferPtr hbm_primary;
    cloudtpu::PjrtBufferPtr hbm_scales;
    cloudtpu::PjrtBufferPtr hbm_mins;
    std::vector<uint8_t> quant_meta;  // Compact 16B/18B per QK_K block header for lossless get_tensor roundtrip
    std::vector<uint8_t> staging;
    bool staging_dirty = false;
};

struct ggml_backend_cloudtpu_buffer_context {
    int device_index = 0;
    uintptr_t virtual_base = 0;
    size_t size = 0;

    std::mutex mu;
    std::unordered_map<size_t, CloudTpuTensorSlot> slots;  // keyed by byte offset from virtual_base

    void evict_overlapping_locked(size_t offset, size_t nbytes, size_t keep_exact_offset = SIZE_MAX) {
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

static void upload_raw_tensor_to_slot(cloudtpu::PjrtRuntime& rt,
                                      int device_index,
                                      CloudTpuTensorSlot& slot,
                                      const struct ggml_tensor* tensor,
                                      const void* raw_data) {
    slot.ggml_dt = tensor->type;
    slot.dims[0] = tensor->ne[3];
    slot.dims[1] = tensor->ne[2];
    slot.dims[2] = tensor->ne[1];
    slot.dims[3] = tensor->ne[0];

    const int64_t B3 = tensor->ne[3];
    const int64_t B2 = tensor->ne[2];
    const int64_t N = tensor->ne[1];
    const int64_t K = tensor->ne[0];

    if (tensor->type == GGML_TYPE_Q8_0 && (K % QK8_0) == 0) {
        const size_t nb = static_cast<size_t>(K / QK8_0);
        const size_t nrows = static_cast<size_t>(B3 * B2 * N);
        std::vector<int8_t> qs;
        std::vector<uint16_t> d;
        unpack_q8_0_to_split(static_cast<const block_q8_0*>(raw_data), nrows, nb, qs, d);
        slot.storage = cloudtpu::TensorStorageKind::Q8_0_SPLIT;
        slot.dtype = cloudtpu::PjrtDType::S8;
        slot.hbm_primary = rt.buffer_from_host(
            device_index, qs.data(), cloudtpu::PjrtDType::S8, {B3, B2, N, static_cast<int64_t>(nb), QK8_0});
        slot.hbm_scales = rt.buffer_from_host(
            device_index, d.data(), cloudtpu::PjrtDType::F16, {B3, B2, N, static_cast<int64_t>(nb)});
        slot.hbm_mins.reset();
    } else if (tensor->type == GGML_TYPE_Q4_0 && (K % QK4_0) == 0) {
        const size_t nb = static_cast<size_t>(K / QK4_0);
        const size_t nrows = static_cast<size_t>(B3 * B2 * N);
        std::vector<uint8_t> qs;
        std::vector<uint16_t> d;
        unpack_q4_0_to_split(static_cast<const block_q4_0*>(raw_data), nrows, nb, qs, d);
        slot.storage = cloudtpu::TensorStorageKind::Q4_0_SPLIT;
        slot.dtype = cloudtpu::PjrtDType::U8;
        slot.hbm_primary = rt.buffer_from_host(
            device_index, qs.data(), cloudtpu::PjrtDType::U8, {B3, B2, N, static_cast<int64_t>(nb), QK4_0 / 2});
        slot.hbm_scales = rt.buffer_from_host(
            device_index, d.data(), cloudtpu::PjrtDType::F16, {B3, B2, N, static_cast<int64_t>(nb)});
        slot.hbm_mins.reset();
    } else if (tensor->type == GGML_TYPE_Q4_K && (K % QK_K) == 0) {
        const size_t nb_k = static_cast<size_t>(K / QK_K);
        const int64_t nb32 = K / 32;
        const size_t nrows = static_cast<size_t>(B3 * B2 * N);
        const size_t total_k_blocks = nrows * nb_k;
        const auto* src_blk = static_cast<const block_q4_K*>(raw_data);
        std::vector<uint8_t> qs;
        std::vector<float> d_eff;
        std::vector<float> m_eff;
        unpack_q4_K_to_split(src_blk, nrows, nb_k, qs, d_eff, m_eff);
        slot.quant_meta.resize(total_k_blocks * 16);
        for (size_t b = 0; b < total_k_blocks; ++b) {
            std::memcpy(slot.quant_meta.data() + b * 16, &src_blk[b], 16);
        }
        slot.storage = cloudtpu::TensorStorageKind::Q4_K_SPLIT;
        slot.dtype = cloudtpu::PjrtDType::U8;
        slot.hbm_primary = rt.buffer_from_host(
            device_index, qs.data(), cloudtpu::PjrtDType::U8, {B3, B2, N, nb32, 16});
        slot.hbm_scales = rt.buffer_from_host(
            device_index, d_eff.data(), cloudtpu::PjrtDType::F32, {B3, B2, N, nb32});
        slot.hbm_mins = rt.buffer_from_host(
            device_index, m_eff.data(), cloudtpu::PjrtDType::F32, {B3, B2, N, nb32});
    } else if (tensor->type == GGML_TYPE_Q6_K && (K % QK_K) == 0) {
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
        slot.storage = cloudtpu::TensorStorageKind::Q6_K_SPLIT;
        slot.dtype = cloudtpu::PjrtDType::S8;
        slot.hbm_primary = rt.buffer_from_host(
            device_index, qs.data(), cloudtpu::PjrtDType::S8, {B3, B2, N, nb16, 16});
        slot.hbm_scales = rt.buffer_from_host(
            device_index, d_eff.data(), cloudtpu::PjrtDType::F32, {B3, B2, N, nb16});
        slot.hbm_mins.reset();
    } else {
        const cloudtpu::PjrtDType pdt = ggml_type_to_pjrt(tensor->type);
        if (pdt != cloudtpu::PjrtDType::INVALID && ggml_is_contiguous(tensor)) {
            slot.storage = cloudtpu::TensorStorageKind::DENSE;
            slot.dtype = pdt;
            slot.hbm_primary = rt.buffer_from_host(device_index, raw_data, pdt, {B3, B2, N, K});
            slot.hbm_scales.reset();
            slot.hbm_mins.reset();
        } else {
            // Raw byte fallback for unsupported types (keeps tensor data intact for get_tensor / CPU fallback).
            slot.storage = cloudtpu::TensorStorageKind::DENSE;
            slot.dtype = cloudtpu::PjrtDType::U8;
            slot.hbm_primary = rt.buffer_from_host(
                device_index, raw_data, cloudtpu::PjrtDType::U8, {static_cast<int64_t>(slot.nbytes)});
            slot.hbm_scales.reset();
            slot.hbm_mins.reset();
        }
    }
    std::vector<uint8_t>().swap(slot.staging);
    slot.staging_dirty = false;
}

static void download_slot_to_host(cloudtpu::PjrtRuntime& rt,
                                  const CloudTpuTensorSlot& slot,
                                  void* dst_full) {
    if (slot.staging_dirty && !slot.staging.empty()) {
        std::memcpy(dst_full, slot.staging.data(), slot.nbytes);
        return;
    }
    if (slot.storage == cloudtpu::TensorStorageKind::Q8_0_SPLIT) {
        const size_t total_blocks = slot.hbm_scales->num_elements();
        std::vector<int8_t> qs(total_blocks * QK8_0);
        std::vector<uint16_t> d(total_blocks);
        rt.buffer_to_host(slot.hbm_primary->handle.get(), qs.data(), qs.size());
        rt.buffer_to_host(slot.hbm_scales->handle.get(), d.data(), d.size() * sizeof(uint16_t));
        pack_split_to_q8_0(qs.data(), d.data(), total_blocks, static_cast<block_q8_0*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q4_0_SPLIT) {
        const size_t total_blocks = slot.hbm_scales->num_elements();
        std::vector<uint8_t> qs(total_blocks * (QK4_0 / 2));
        std::vector<uint16_t> d(total_blocks);
        rt.buffer_to_host(slot.hbm_primary->handle.get(), qs.data(), qs.size());
        rt.buffer_to_host(slot.hbm_scales->handle.get(), d.data(), d.size() * sizeof(uint16_t));
        pack_split_to_q4_0(qs.data(), d.data(), total_blocks, static_cast<block_q4_0*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q4_K_SPLIT) {
        const size_t total_sub = slot.hbm_scales->num_elements();
        const size_t total_k_blocks = total_sub / 8;
        std::vector<uint8_t> qs(total_sub * 16);
        rt.buffer_to_host(slot.hbm_primary->handle.get(), qs.data(), qs.size());
        pack_split_to_q4_K(qs.data(), slot.quant_meta.data(), total_k_blocks, static_cast<block_q4_K*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q6_K_SPLIT) {
        const size_t total_sub = slot.hbm_scales->num_elements();
        const size_t total_k_blocks = total_sub / 16;
        std::vector<int8_t> qs(total_sub * 16);
        rt.buffer_to_host(slot.hbm_primary->handle.get(), qs.data(), qs.size());
        pack_split_to_q6_K(qs.data(), slot.quant_meta.data(), total_k_blocks, static_cast<block_q6_K*>(dst_full));
    } else if (slot.hbm_primary) {
        rt.buffer_to_host(slot.hbm_primary->handle.get(), dst_full, slot.nbytes);
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

static void ggml_backend_cloudtpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                       struct ggml_tensor* tensor,
                                                       uint8_t value,
                                                       size_t offset,
                                                       size_t size) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    const size_t t_off = tensor_offset_in_buffer(bctx, tensor);
    const size_t t_bytes = ggml_nbytes(tensor);

    std::lock_guard<std::mutex> lock(bctx->mu);
    // If writing into a sub-slice view of an existing larger HBM slot (e.g. KV cache),
    // patch the parent slot instead of evicting it.
    if (tensor->view_src != nullptr) {
        const struct ggml_tensor* root = unwrap_view_src(tensor);
        if (root && root->buffer == buffer) {
            const size_t root_off = tensor_offset_in_buffer(bctx, root);
            auto pit = bctx->slots.find(root_off);
            if (pit != bctx->slots.end() && t_off >= pit->second.offset &&
                t_off + offset + size <= pit->second.offset + pit->second.nbytes) {
                CloudTpuTensorSlot& parent = pit->second;
                if (!parent.staging_dirty) {
                    parent.staging.assign(parent.nbytes, 0);
                    if (parent.hbm_primary) {
                        download_slot_to_host(rt, parent, parent.staging.data());
                    }
                    parent.staging_dirty = true;
                }
                const size_t rel = (t_off - parent.offset) + offset;
                std::memset(parent.staging.data() + rel, value, size);
                upload_raw_tensor_to_slot(rt, bctx->device_index, parent, root, parent.staging.data());
                return;
            }
        }
    }
    bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
    CloudTpuTensorSlot& slot = bctx->slots[t_off];
    slot.offset = t_off;
    slot.nbytes = t_bytes;

    if (offset == 0 && size == t_bytes) {
        std::vector<uint8_t> tmp(t_bytes, value);
        upload_raw_tensor_to_slot(rt, bctx->device_index, slot, tensor, tmp.data());
    } else {
        if (!slot.staging_dirty) {
            slot.staging.assign(t_bytes, 0);
            if (slot.hbm_primary) {
                download_slot_to_host(rt, slot, slot.staging.data());
            }
            slot.staging_dirty = true;
        }
        std::memset(slot.staging.data() + offset, value, size);
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
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    const size_t t_off = tensor_offset_in_buffer(bctx, tensor);
    const size_t t_bytes = ggml_nbytes(tensor);

    std::lock_guard<std::mutex> lock(bctx->mu);
    // If writing into a sub-slice view of an existing larger HBM slot (e.g. KV cache),
    // patch the parent slot instead of evicting it.
    if (tensor->view_src != nullptr) {
        const struct ggml_tensor* root = unwrap_view_src(tensor);
        if (root && root->buffer == buffer) {
            const size_t root_off = tensor_offset_in_buffer(bctx, root);
            auto pit = bctx->slots.find(root_off);
            if (pit != bctx->slots.end() && t_off >= pit->second.offset &&
                t_off + offset + size <= pit->second.offset + pit->second.nbytes) {
                CloudTpuTensorSlot& parent = pit->second;
                if (!parent.staging_dirty) {
                    parent.staging.assign(parent.nbytes, 0);
                    if (parent.hbm_primary) {
                        download_slot_to_host(rt, parent, parent.staging.data());
                    }
                    parent.staging_dirty = true;
                }
                const size_t rel = (t_off - parent.offset) + offset;
                std::memcpy(parent.staging.data() + rel, data, size);
                upload_raw_tensor_to_slot(rt, bctx->device_index, parent, root, parent.staging.data());
                return;
            }
        }
    }
    bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
    CloudTpuTensorSlot& slot = bctx->slots[t_off];
    slot.offset = t_off;
    slot.nbytes = t_bytes;

    if (offset == 0 && size == t_bytes) {
        upload_raw_tensor_to_slot(rt, bctx->device_index, slot, tensor, data);
    } else {
        if (!slot.staging_dirty) {
            slot.staging.assign(t_bytes, 0);
            if (slot.hbm_primary) {
                download_slot_to_host(rt, slot, slot.staging.data());
            }
            slot.staging_dirty = true;
        }
        std::memcpy(slot.staging.data() + offset, data, size);
        // If full buffer has been populated via tail write, flush to HBM immediately.
        if (offset + size == t_bytes) {
            upload_raw_tensor_to_slot(rt, bctx->device_index, slot, tensor, slot.staging.data());
        }
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
        if (offset == 0 && size == slot.nbytes && slot.storage == cloudtpu::TensorStorageKind::DENSE &&
            !slot.staging_dirty && slot.hbm_primary) {
            rt.buffer_to_host(slot.hbm_primary->handle.get(), data, size);
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
            std::vector<uint8_t> tmp(parent.nbytes);
            download_slot_to_host(rt, parent, tmp.data());
            const size_t rel = (t_off - parent.offset) + offset;
            std::memcpy(data, tmp.data() + rel, size);
            return;
        }
    }

    // Zero-initialized fallback if tensor was never explicitly written.
    std::memset(data, 0, size);
}

static bool ggml_backend_cloudtpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    const struct ggml_tensor* src,
                                                    struct ggml_tensor* dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_cloudtpu_buffer_set_tensor(buffer, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    if (is_cloudtpu_buffer(src->buffer) && src->type == dst->type && ggml_nbytes(src) == ggml_nbytes(dst)) {
        auto* src_bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(src->buffer->context);
        auto* dst_bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
        if (src_bctx->device_index == dst_bctx->device_index) {
            const size_t src_off = tensor_offset_in_buffer(src_bctx, src);
            const size_t dst_off = tensor_offset_in_buffer(dst_bctx, dst);
            if (src_bctx == dst_bctx) {
                std::lock_guard<std::mutex> lock(src_bctx->mu);
                auto sit = src_bctx->slots.find(src_off);
                if (sit != src_bctx->slots.end() && !sit->second.staging_dirty && sit->second.hbm_primary) {
                    CloudTpuTensorSlot copy = sit->second;
                    copy.offset = dst_off;
                    src_bctx->evict_overlapping_locked(dst_off, copy.nbytes, dst_off);
                    src_bctx->slots[dst_off] = std::move(copy);
                    return true;
                }
            } else {
                CloudTpuTensorSlot copy;
                bool ok = false;
                {
                    std::lock_guard<std::mutex> slock(src_bctx->mu);
                    auto sit = src_bctx->slots.find(src_off);
                    if (sit != src_bctx->slots.end() && !sit->second.staging_dirty && sit->second.hbm_primary) {
                        copy = sit->second;
                        ok = true;
                    }
                }
                if (ok) {
                    std::lock_guard<std::mutex> dlock(dst_bctx->mu);
                    copy.offset = dst_off;
                    dst_bctx->evict_overlapping_locked(dst_off, copy.nbytes, dst_off);
                    dst_bctx->slots[dst_off] = std::move(copy);
                    return true;
                }
            }
        }
    }
    return false;
}

static void ggml_backend_cloudtpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(value);
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    std::lock_guard<std::mutex> lock(bctx->mu);
    bctx->slots.clear();
}

static void ggml_backend_cloudtpu_buffer_reset(ggml_backend_buffer_t buffer) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    std::lock_guard<std::mutex> lock(bctx->mu);
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
    /* .reset         = */ ggml_backend_cloudtpu_buffer_reset,
    /* .set_tensor_n  = */ NULL,
    /* .get_tensor_n  = */ NULL,
    /* .cpy_tensor_n  = */ NULL,
};

struct cloudtpu_buft_context {
    int device_index = 0;
    std::string name;
};

static const char* ggml_backend_cloudtpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto* btc = static_cast<cloudtpu_buft_context*>(buft->context);
    return btc->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cloudtpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t size) {
    auto* btc = static_cast<cloudtpu_buft_context*>(buft->context);
    ensure_runtime_initialized();

    const size_t aligned_span = std::max<size_t>(
        CLOUDTPU_BUFFER_ALIGNMENT,
        ((size + CLOUDTPU_BUFFER_ALIGNMENT - 1) & ~(CLOUDTPU_BUFFER_ALIGNMENT - 1)) + 4096);

    auto* bctx = new ggml_backend_cloudtpu_buffer_context();
    bctx->device_index = btc->device_index;
    bctx->virtual_base = g_next_virtual_base.fetch_add(aligned_span, std::memory_order_relaxed);
    bctx->size = size;

    return ggml_backend_buffer_init(buft, ggml_backend_cloudtpu_buffer_i, bctx, size);
}

static size_t ggml_backend_cloudtpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return CLOUDTPU_BUFFER_ALIGNMENT;
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

struct ggml_backend_cloudtpu_context {
    int device_index = 0;
    cloudtpu_env_config env;
    uint64_t total_graph_evals = 0;
    uint64_t total_subgraphs = 0;
    uint64_t total_ops_executed = 0;

    ~ggml_backend_cloudtpu_context() {
        if (env.verbose > 0) {
            cloudtpu::PjrtRuntime& rt = cloudtpu::shared_pjrt_runtime(nullptr);
            if (rt.is_initialized()) {
                const cloudtpu::CacheStats st = rt.cache_stats();
                GGML_LOG_INFO(
                    "ggml-cloudtpu: summary device=%d mock=%d maps_libpython=%zu evals=%llu subgraphs=%llu ops=%llu "
                    "cache_entries=%zu cache_hits=%llu cache_misses=%llu cache_evictions=%llu "
                    "compile_us=%llu execute_us=%llu\n",
                    device_index,
                    rt.is_mock() ? 1 : 0,
                    cloudtpu::count_libpython_mappings(),
                    static_cast<unsigned long long>(total_graph_evals),
                    static_cast<unsigned long long>(total_subgraphs),
                    static_cast<unsigned long long>(total_ops_executed),
                    st.entries,
                    static_cast<unsigned long long>(st.hits),
                    static_cast<unsigned long long>(st.misses),
                    static_cast<unsigned long long>(st.evictions),
                    static_cast<unsigned long long>(st.compile_us),
                    static_cast<unsigned long long>(st.execute_us));
            }
        }
    }
};

static bool is_view_noop_op(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool ggml_backend_cloudtpu_device_supports_op_impl(const struct ggml_tensor* op) {
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
            if (n_dims <= 0 || (n_dims % 2) != 0 || n_offs < 0 || (n_offs % 2) != 0 ||
                n_offs + n_dims > src0->ne[0]) {
                return false;
            }
            if (src1->ne[0] != src0->ne[2]) {
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

static cloudtpu::TensorSpec tensor_to_spec(const struct ggml_tensor* t, uint64_t id) {
    cloudtpu::TensorSpec ts;
    ts.id = id;
    ts.dims[0] = t->ne[3];
    ts.dims[1] = t->ne[2];
    ts.dims[2] = t->ne[1];
    ts.dims[3] = t->ne[0];
    if (t->type == GGML_TYPE_Q8_0) {
        ts.storage = cloudtpu::TensorStorageKind::Q8_0_SPLIT;
        ts.dtype = cloudtpu::PjrtDType::S8;
    } else if (t->type == GGML_TYPE_Q4_0) {
        ts.storage = cloudtpu::TensorStorageKind::Q4_0_SPLIT;
        ts.dtype = cloudtpu::PjrtDType::U8;
    } else if (t->type == GGML_TYPE_Q4_K) {
        ts.storage = cloudtpu::TensorStorageKind::Q4_K_SPLIT;
        ts.dtype = cloudtpu::PjrtDType::U8;
    } else if (t->type == GGML_TYPE_Q6_K) {
        ts.storage = cloudtpu::TensorStorageKind::Q6_K_SPLIT;
        ts.dtype = cloudtpu::PjrtDType::S8;
    } else {
        ts.storage = cloudtpu::TensorStorageKind::DENSE;
        ts.dtype = ggml_type_to_pjrt(t->type);
    }
    return ts;
}

// Resolves a GGML tensor to HBM buffers (fetching from CloudTpuBuffer or uploading transient host tensor).
static void resolve_tensor_hbm(cloudtpu::PjrtRuntime& rt,
                               int device_index,
                               const struct ggml_tensor* t,
                               cloudtpu::PjrtBufferPtr& out_primary,
                               cloudtpu::PjrtBufferPtr& out_scales,
                               cloudtpu::PjrtBufferPtr& out_mins) {
    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        const size_t t_off = tensor_offset_in_buffer(bctx, t);
        std::lock_guard<std::mutex> lock(bctx->mu);
        auto it = bctx->slots.find(t_off);
        if (it != bctx->slots.end()) {
            CloudTpuTensorSlot& slot = it->second;
            if (slot.staging_dirty && !slot.staging.empty()) {
                upload_raw_tensor_to_slot(rt, bctx->device_index, slot, t, slot.staging.data());
            }
            if (slot.hbm_primary && slot.ggml_dt == t->type &&
                slot.dims[0] == t->ne[3] && slot.dims[1] == t->ne[2] &&
                slot.dims[2] == t->ne[1] && slot.dims[3] == t->ne[0]) {
                out_primary = slot.hbm_primary;
                out_scales = slot.hbm_scales;
                out_mins = slot.hbm_mins;
                return;
            }
        }
        // Check if tensor is a contiguous view aliasing a parent slot at offset 0 with matching element count.
        if (t->view_src != nullptr && t->view_offs == 0 && ggml_is_contiguous(t)) {
            const struct ggml_tensor* base = unwrap_view_src(t);
            const size_t base_off = tensor_offset_in_buffer(bctx, base);
            auto bit = bctx->slots.find(base_off);
            if (bit != bctx->slots.end() && bit->second.hbm_primary) {
                out_primary = bit->second.hbm_primary;
                out_scales = bit->second.hbm_scales;
                out_mins = bit->second.hbm_mins;
                return;
            }
        }
        // Uninitialized zero-filled HBM buffer for newly allocated leaf/KV-cache buffers before first write.
        // Never silently zero-init a computed graph activation node (t->op != GGML_OP_NONE).
        const cloudtpu::PjrtDType pdt = ggml_type_to_pjrt(t->type);
        if (t->op == GGML_OP_NONE && pdt != cloudtpu::PjrtDType::INVALID && ggml_is_contiguous(t)) {
            const size_t t_bytes = ggml_nbytes(t);
            std::vector<uint8_t> zeros(t_bytes, 0);
            bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
            CloudTpuTensorSlot& slot = bctx->slots[t_off];
            slot.offset = t_off;
            slot.nbytes = t_bytes;
            upload_raw_tensor_to_slot(rt, bctx->device_index, slot, t, zeros.data());
            out_primary = slot.hbm_primary;
            out_scales = slot.hbm_scales;
            out_mins = slot.hbm_mins;
            return;
        }
        throw std::runtime_error("ggml-cloudtpu: missing HBM slot for input tensor '" +
                                 std::string(t->name) + "' (op=" + std::string(ggml_op_desc(t)) + ")");
    }

    // Host tensor fallback: upload transient copy to device HBM.
    if (!t->data) {
        throw std::runtime_error("ggml-cloudtpu: null host data pointer for tensor '" +
                                 std::string(t->name) + "'");
    }
    CloudTpuTensorSlot tmp_slot;
    tmp_slot.nbytes = ggml_nbytes(t);
    upload_raw_tensor_to_slot(rt, device_index, tmp_slot, t, t->data);
    out_primary = std::move(tmp_slot.hbm_primary);
    out_scales = std::move(tmp_slot.hbm_scales);
    out_mins = std::move(tmp_slot.hbm_mins);
}

static void store_tensor_hbm(cloudtpu::PjrtRuntime& rt,
                             struct ggml_tensor* t,
                             const cloudtpu::PjrtBufferPtr& buf) {
    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        // For in-place views (like SET_ROWS or ROPE inplace), store under the canonical root tensor's offset.
        const struct ggml_tensor* target = t;
        if (t->view_src != nullptr && t->view_offs == 0 && ggml_nelements(t) == ggml_nelements(t->view_src)) {
            target = unwrap_view_src(t);
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
        slot.dtype = buf->dtype;
        slot.dims[0] = target->ne[3];
        slot.dims[1] = target->ne[2];
        slot.dims[2] = target->ne[1];
        slot.dims[3] = target->ne[0];
        slot.hbm_primary = buf;
        slot.hbm_scales.reset();
        slot.hbm_mins.reset();
        std::vector<uint8_t>().swap(slot.staging);
        slot.staging_dirty = false;
        return;
    }

    if (t->data) {
        rt.buffer_to_host(buf->handle.get(), t->data, ggml_nbytes(t));
    }
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
            const enum ggml_prec prec = ggml_prec_get_src(node, 1);
            spec.high_precision = (prec == GGML_PREC_F32);
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

static enum ggml_status execute_subgraph(ggml_backend_cloudtpu_context* ctx,
                                         cloudtpu::PjrtRuntime& rt,
                                         struct ggml_cgraph* cgraph,
                                         int start_idx,
                                         int end_idx) {
    // Identify which tensors produced inside [start_idx, end_idx) are consumed inside vs outside.
    std::unordered_set<const struct ggml_tensor*> produced_in_slice;
    std::unordered_set<const struct ggml_tensor*> consumed_inside;
    std::unordered_set<const struct ggml_tensor*> needed_outside;
    for (int i = start_idx; i < end_idx; ++i) {
        produced_in_slice.insert(cgraph->nodes[i]);
    }
    for (int i = start_idx; i < end_idx; ++i) {
        const struct ggml_tensor* n = cgraph->nodes[i];
        if (is_view_noop_op(n->op)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            for (const struct ggml_tensor* cur = n->src[s]; cur != nullptr; cur = view_parent(cur)) {
                if (produced_in_slice.count(cur)) {
                    consumed_inside.insert(cur);
                }
            }
        }
    }
    for (int i = end_idx; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor* n = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            for (const struct ggml_tensor* cur = n->src[s]; cur != nullptr; cur = view_parent(cur)) {
                if (produced_in_slice.count(cur)) {
                    needed_outside.insert(cur);
                }
            }
        }
        for (const struct ggml_tensor* cur = view_parent(n); cur != nullptr; cur = view_parent(cur)) {
            if (produced_in_slice.count(cur)) {
                needed_outside.insert(cur);
            }
        }
    }

    // Track buffer-slot reuse (ggml_gallocr liveness) so tensors still live at the end of a
    // ggml_backend_sched split cgraph are always materialized to HBM, while overwritten dead
    // intermediates remain fused in XLA VMEM and never clobber newer slots.
    auto ranges_overlap = [](const struct ggml_tensor* a, const struct ggml_tensor* b) -> bool {
        if (!a || !b || !a->buffer || a->buffer != b->buffer || !is_cloudtpu_buffer(a->buffer)) {
            return false;
        }
        const uintptr_t a0 = reinterpret_cast<uintptr_t>(a->data);
        const uintptr_t a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = reinterpret_cast<uintptr_t>(b->data);
        const uintptr_t b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };

    const int slice_len = end_idx - start_idx;
    std::vector<bool> overwritten_in_slice(static_cast<size_t>(slice_len), false);
    std::vector<bool> overwritten_after_slice(static_cast<size_t>(slice_len), false);
    for (int i = start_idx; i < end_idx; ++i) {
        const struct ggml_tensor* ni = cgraph->nodes[i];
        if (is_view_noop_op(ni->op)) {
            continue;
        }
        for (int k = i + 1; k < end_idx; ++k) {
            const struct ggml_tensor* nk = cgraph->nodes[k];
            if (!is_view_noop_op(nk->op) && ranges_overlap(ni, nk)) {
                overwritten_in_slice[static_cast<size_t>(i - start_idx)] = true;
                break;
            }
        }
        for (int k = end_idx; k < cgraph->n_nodes; ++k) {
            const struct ggml_tensor* nk = cgraph->nodes[k];
            if (!is_view_noop_op(nk->op) && ranges_overlap(ni, nk)) {
                overwritten_after_slice[static_cast<size_t>(i - start_idx)] = true;
                break;
            }
        }
    }

    cloudtpu::GraphProgramSpec prog_spec;
    std::unordered_map<const struct ggml_tensor*, uint64_t> tensor_to_id;
    std::unordered_map<uint64_t, const struct ggml_tensor*> id_to_input_tensor;
    std::unordered_map<uint64_t, struct ggml_tensor*> id_to_output_tensor;
    uint64_t next_id = 1;

    auto get_or_register_input = [&](auto& self, const struct ggml_tensor* t) -> uint64_t {
        if (!t) return 0;
        auto it = tensor_to_id.find(t);
        if (it != tensor_to_id.end()) {
            return it->second;
        }
        const struct ggml_tensor* parent = view_parent(t);
        if (t->view_src != nullptr && parent != nullptr &&
            is_dense_supported_type(t->type) && t->type == parent->type) {
            if (t->op == GGML_OP_PERMUTE && t->view_offs == parent->view_offs) {
                const uint64_t base_id = self(self, parent);
                const uint64_t tid = next_id++;
                cloudtpu::OpNodeSpec p_op;
                p_op.kind = cloudtpu::CloudOpKind::TRANSPOSE;
                p_op.src_ids = {base_id};
                p_op.dst = tensor_to_spec(t, tid);
                const int32_t* ax = static_cast<const int32_t*>(t->op_params);
                int inv[4] = {0, 1, 2, 3};
                for (int g = 0; g < 4; ++g) {
                    inv[ax[g]] = g;
                }
                for (int h = 0; h < 4; ++h) {
                    p_op.perm[h] = 3 - inv[3 - h];
                }
                prog_spec.ops.push_back(std::move(p_op));
                tensor_to_id[t] = tid;
                return tid;
            }
            if (t->op == GGML_OP_TRANSPOSE && t->view_offs == parent->view_offs) {
                const uint64_t base_id = self(self, parent);
                const uint64_t tid = next_id++;
                cloudtpu::OpNodeSpec p_op;
                p_op.kind = cloudtpu::CloudOpKind::TRANSPOSE;
                p_op.src_ids = {base_id};
                p_op.dst = tensor_to_spec(t, tid);
                p_op.perm[0] = 0;
                p_op.perm[1] = 1;
                p_op.perm[2] = 3;
                p_op.perm[3] = 2;
                prog_spec.ops.push_back(std::move(p_op));
                tensor_to_id[t] = tid;
                return tid;
            }
            if (t->view_offs == parent->view_offs && ggml_is_contiguous(t) &&
                ggml_is_contiguous(parent) &&
                ggml_nelements(t) == ggml_nelements(parent)) {
                const uint64_t base_id = self(self, parent);
                const uint64_t rid = next_id++;
                cloudtpu::OpNodeSpec r_op;
                r_op.kind = cloudtpu::CloudOpKind::RESHAPE;
                r_op.src_ids = {base_id};
                r_op.dst = tensor_to_spec(t, rid);
                prog_spec.ops.push_back(std::move(r_op));
                tensor_to_id[t] = rid;
                return rid;
            }
            ViewSlicePlan vplan;
            if (try_build_view_slice(t, &vplan)) {
                const uint64_t base_id = self(self, parent);
                const uint64_t sid = next_id++;
                cloudtpu::OpNodeSpec s_op;
                s_op.kind = cloudtpu::CloudOpKind::SLICE;
                s_op.src_ids = {base_id};
                s_op.dst = tensor_to_spec(t, sid);
                s_op.slice_reshape = std::move(vplan.reshape_dims);
                s_op.slice_starts = std::move(vplan.starts);
                s_op.slice_limits = std::move(vplan.limits);
                prog_spec.ops.push_back(std::move(s_op));
                tensor_to_id[t] = sid;
                return sid;
            }
        }
        const uint64_t id = next_id++;
        tensor_to_id[t] = id;
        id_to_input_tensor[id] = t;
        prog_spec.inputs.push_back(tensor_to_spec(t, id));
        return id;
    };

    for (int i = start_idx; i < end_idx; ++i) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (is_view_noop_op(node->op)) {
            continue;
        }

        std::vector<uint64_t> src_ids;
        if (node->op == GGML_OP_ROPE || node->op == GGML_OP_SOFT_MAX) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[1]));
            if (node->src[2]) {
                src_ids.push_back(get_or_register_input(get_or_register_input, node->src[2]));
            }
        } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[1]));
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[2]));
            if (node->src[3]) {
                src_ids.push_back(get_or_register_input(get_or_register_input, node->src[3]));
            }
        } else if (node->op == GGML_OP_GLU) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
            if (node->src[1]) {
                src_ids.push_back(get_or_register_input(get_or_register_input, node->src[1]));
            }
        } else {
            for (int s = 0; s < GGML_MAX_SRC && node->src[s] != nullptr; ++s) {
                src_ids.push_back(get_or_register_input(get_or_register_input, node->src[s]));
            }
        }

        uint64_t dst_id = next_id++;
        prog_spec.ops.push_back(build_op_node_spec(node, dst_id, src_ids));
        tensor_to_id[node] = dst_id;

        // If an in-place op (e.g. SET_ROWS on reshaped multi-stream KV cache) aliases a root tensor
        // with a different shape, reshape back to root shape before storing into root HBM slot.
        uint64_t store_id = dst_id;
        if (node->view_src != nullptr && node->view_offs == 0 &&
            ggml_nelements(node) == ggml_nelements(node->view_src)) {
            const struct ggml_tensor* root = unwrap_view_src(node);
            if (root && (root->ne[0] != node->ne[0] || root->ne[1] != node->ne[1] ||
                         root->ne[2] != node->ne[2] || root->ne[3] != node->ne[3])) {
                const uint64_t rid = next_id++;
                cloudtpu::OpNodeSpec r_op;
                r_op.kind = cloudtpu::CloudOpKind::RESHAPE;
                r_op.src_ids = {dst_id};
                r_op.dst = tensor_to_spec(root, rid);
                prog_spec.ops.push_back(std::move(r_op));
                store_id = rid;
            }
        }

        const size_t rel_idx = static_cast<size_t>(i - start_idx);
        const bool alive_past_cgraph = !overwritten_in_slice[rel_idx] && !overwritten_after_slice[rel_idx];
        const bool is_live_out =
            !overwritten_in_slice[rel_idx] &&
            ((i == end_idx - 1) ||
             (node->op == GGML_OP_SET_ROWS) ||
             (node->view_src != nullptr) ||
             (consumed_inside.count(node) == 0) ||
             (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
             (needed_outside.count(node) != 0) ||
             alive_past_cgraph ||
             (!node->buffer || !is_cloudtpu_buffer(node->buffer)));
        if (is_live_out) {
            prog_spec.output_ids.push_back(store_id);
            id_to_output_tensor[store_id] = node;
        }
    }

    if (prog_spec.ops.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    const cloudtpu::EmittedProgram emitted = cloudtpu::emit_stablehlo_program(prog_spec);
    if (ctx->env.verbose >= 3) {
        GGML_LOG_INFO("ggml-cloudtpu: subgraph [%d..%d) inputs=%zu ops=%zu outs=%zu\n%s\n",
                      start_idx, end_idx,
                      emitted.input_bindings.size(), prog_spec.ops.size(), emitted.output_bindings.size(),
                      emitted.mlir_text.c_str());
    }
    const cloudtpu::CachedExecutable* exec = rt.compile_or_get(emitted.cache_key, emitted.mlir_text);

    // Resolve HBM PJRT_Buffer* inputs (keeping transient buffers alive through execute).
    std::vector<cloudtpu::PjrtBufferPtr> held_inputs;
    std::vector<PJRT_Buffer*> raw_inputs;
    held_inputs.reserve(emitted.input_bindings.size());
    raw_inputs.reserve(emitted.input_bindings.size());

    struct ResolvedParts {
        cloudtpu::PjrtBufferPtr primary;
        cloudtpu::PjrtBufferPtr scales;
        cloudtpu::PjrtBufferPtr mins;
    };
    std::unordered_map<uint64_t, ResolvedParts> resolved_cache;
    for (const auto& ib : emitted.input_bindings) {
        auto rit = resolved_cache.find(ib.tensor_id);
        if (rit == resolved_cache.end()) {
            const struct ggml_tensor* t = id_to_input_tensor.at(ib.tensor_id);
            ResolvedParts rp;
            resolve_tensor_hbm(rt, ctx->device_index, t, rp.primary, rp.scales, rp.mins);
            rit = resolved_cache.emplace(ib.tensor_id, std::move(rp)).first;
        }
        cloudtpu::PjrtBufferPtr chosen =
            (ib.part == 0) ? rit->second.primary : ((ib.part == 1) ? rit->second.scales : rit->second.mins);
        raw_inputs.push_back(chosen->handle.get());
        held_inputs.push_back(std::move(chosen));
    }

    std::vector<cloudtpu::PjrtDType> out_dtypes;
    std::vector<std::vector<int64_t>> out_dims;
    out_dtypes.reserve(emitted.output_bindings.size());
    out_dims.reserve(emitted.output_bindings.size());
    for (const auto& ob : emitted.output_bindings) {
        out_dtypes.push_back(ob.dtype);
        out_dims.push_back(ob.dims);
    }

    std::vector<cloudtpu::PjrtBufferPtr> outputs;
    const bool sync_exec = (end_idx == cgraph->n_nodes) || !ctx->env.fuse_graphs;
    rt.execute(exec,
               ctx->device_index,
               raw_inputs.data(),
               raw_inputs.size(),
               outputs,
               out_dtypes,
               out_dims,
               sync_exec);

    for (size_t oi = 0; oi < emitted.output_bindings.size(); ++oi) {
        struct ggml_tensor* out_t = id_to_output_tensor.at(emitted.output_bindings[oi].tensor_id);
        store_tensor_hbm(rt, out_t, outputs[oi]);
    }

    ctx->total_subgraphs += 1;
    ctx->total_ops_executed += prog_spec.ops.size();
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_cloudtpu_graph_compute(ggml_backend_t backend,
                                                            struct ggml_cgraph* cgraph) {
    auto* ctx = static_cast<ggml_backend_cloudtpu_context*>(backend->context);
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    ctx->total_graph_evals += 1;

    int i = 0;
    try {
        while (i < cgraph->n_nodes) {
            if (is_view_noop_op(cgraph->nodes[i]->op)) {
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
                const bool start_breaks_after =
                    (cgraph->nodes[i]->op == GGML_OP_SET_ROWS ||
                     cgraph->nodes[i]->op == GGML_OP_FLASH_ATTN_EXT ||
                     cgraph->nodes[i]->view_src != nullptr);
                int fused_ops = 1;
                if (!start_breaks_after) {
                    while (j < cgraph->n_nodes && fused_ops < 16 &&
                           ggml_backend_cloudtpu_device_supports_op_impl(cgraph->nodes[j])) {
                        const struct ggml_tensor* nj = cgraph->nodes[j];
                        if (is_view_noop_op(nj->op)) {
                            ++j;
                            continue;
                        }
                        // Isolate KV-cache mutations (SET_ROWS), in-place views, and dynamic-n_kv
                        // attention (FLASH_ATTN_EXT) at subgraph boundaries so static-shape QKV/FFN
                        // subgraphs stay 100% cache-hit across all decode tokens.
                        if (nj->op == GGML_OP_SET_ROWS || nj->op == GGML_OP_FLASH_ATTN_EXT ||
                            nj->view_src != nullptr) {
                            break;
                        }
                        ++j;
                        ++fused_ops;
                    }
                    while (j > i + 1 && is_view_noop_op(cgraph->nodes[j - 1]->op)) {
                        --j;
                    }
                }
            }
            const enum ggml_status st = execute_subgraph(ctx, rt, cgraph, i, j);
            if (st != GGML_STATUS_SUCCESS) {
                return st;
            }
            i = j;
        }
        return GGML_STATUS_SUCCESS;
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
    /* .synchronize         = */ NULL,
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
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    if (!rt.query_memory_stats(dc->device_index, free, total)) {
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
    return ggml_backend_cloudtpu_device_supports_op_impl(op);
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
            cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
            const int count = std::max(1, rt.device_count());
            state.dev_ctxs.reserve(static_cast<size_t>(count));
            state.devices.resize(static_cast<size_t>(count));
            state.buft_ctxs.reserve(static_cast<size_t>(count));
            state.bufts.resize(static_cast<size_t>(count));

            for (int i = 0; i < count; ++i) {
                auto dc = std::make_unique<cloudtpu_dev_context>();
                dc->device_index = i;
                dc->name = (i == 0) ? "CLOUDTPU" : ("CLOUDTPU" + std::to_string(i));
                if (i < rt.device_count()) {
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
                btc->name = state.dev_ctxs[static_cast<size_t>(i)]->name;
                state.bufts[static_cast<size_t>(i)] = {
                    /* .iface   = */ {
                        /* .get_name         = */ ggml_backend_cloudtpu_buffer_type_get_name,
                        /* .alloc_buffer     = */ ggml_backend_cloudtpu_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n   = */ NULL,
                        /* .get_alignment    = */ ggml_backend_cloudtpu_buffer_type_get_alignment,
                        /* .get_max_size     = */ NULL,
                        /* .get_alloc_size   = */ NULL,
                        /* .get_alloc_size_n = */ NULL,
                        /* .is_host          = */ ggml_backend_cloudtpu_buffer_type_is_host,
                    },
                    /* .device  = */ &state.devices[static_cast<size_t>(i)],
                    /* .context = */ btc.get(),
                };
                state.buft_ctxs.push_back(std::move(btc));
            }
        } catch (const std::exception& e) {
            GGML_LOG_ERROR("ggml-cloudtpu: failed to initialize PJRT runtime (%s); 0 devices exposed\n",
                           e.what());
            state.devices.clear();
            state.bufts.clear();
        }
        state.initialized = true;
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
    if (device_index < 0 || device_index >= rt.device_count()) {
        GGML_LOG_ERROR("ggml-cloudtpu: invalid device_index=%d (device_count=%d)\n",
                       device_index, rt.device_count());
        return nullptr;
    }

    auto* ctx = new ggml_backend_cloudtpu_context();
    ctx->device_index = device_index;
    ctx->env = env;

    if (env.verbose > 0) {
        const auto& info = rt.device_info(device_index);
        GGML_LOG_INFO(
            "ggml-cloudtpu: initialized device %d (%s, platform=%s %s, mock=%d, fuse=%d, hbm_limit=%lld)\n",
            device_index,
            info.kind.c_str(),
            rt.platform_name().c_str(),
            rt.platform_version().c_str(),
            rt.is_mock() ? 1 : 0,
            env.fuse_graphs ? 1 : 0,
            static_cast<long long>(info.hbm_bytes_limit));
    }

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_cloudtpu_guid(),
        /* .iface   = */ ggml_backend_cloudtpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cloudtpu_reg(), static_cast<size_t>(device_index)),
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
