#include "ggml-edgetpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "openbuenos.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t       EDGETPU_BUFFER_ALIGNMENT         = 64;
constexpr uint32_t     EDGETPU_DEFAULT_CACHE_CAP        = 2048u;
constexpr uint32_t     EDGETPU_DEFAULT_POOL_MAX_BYTES   = 128u * 1024u * 1024u;
constexpr const char * EDGETPU_DEFAULT_DIVE_INTERP_PATH = "/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin";

struct edgetpu_env_config {
    bool         mock_mode             = false;
    int          verbose               = 0;
    bool         packed_q8             = false;
    bool         batch_siblings        = true;
    uint64_t     min_elements          = 0;
    uint32_t     cache_cap             = EDGETPU_DEFAULT_CACHE_CAP;
    uint32_t     pool_max_bytes        = EDGETPU_DEFAULT_POOL_MAX_BYTES;
    uint32_t     tile_max_m            = 0; // 0 -> inherit libopenbuenos tuned silicon defaults
    uint32_t     tile_max_k            = 0;
    uint32_t     tile_max_n            = 0;
    uint8_t      core_power_state      = 0; // 0 -> libopenbuenos default / perf_mode
    uint8_t      mem_power_state       = 0; // 0 -> libopenbuenos default / perf_mode
    uint8_t      qos_class             = 0; // 0 -> libopenbuenos default / perf_mode
    uint8_t      cluster_id            = 0; // 0 -> .dgc0 default
    uint8_t      priority              = 0;
    bool         has_priority          = false;
    uint8_t      perf_mode             = 0; // 0 -> libopenbuenos default, 1..5 = DVFS perf tier
    bool         has_perf_mode         = false;
    const char * dive_interp_path      = EDGETPU_DEFAULT_DIVE_INTERP_PATH;
};

static bool parse_env_u8(const char * name, uint8_t * out) {
    if (const char * v = std::getenv(name)) {
        if (v[0] != '\0') {
            char * end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 0);
            if (end != v && parsed <= 0xFFul) {
                *out = static_cast<uint8_t>(parsed);
                return true;
            }
        }
    }
    return false;
}

static edgetpu_env_config load_env_config() {
    edgetpu_env_config cfg;
#if defined(__ANDROID__)
    cfg.mock_mode = false;
#else
    cfg.mock_mode = true;
#endif

    if (const char * v = std::getenv("OPENBUENOS_MOCK")) {
        cfg.mock_mode = (std::atoi(v) != 0);
    }
    if (const char * v = std::getenv("OPENBUENOS_VERBOSE")) {
        cfg.verbose = std::atoi(v);
    }
    if (const char * v = std::getenv("OPENBUENOS_PACKED_Q8")) {
        cfg.packed_q8 = (std::atoi(v) != 0);
    }
    if (const char * v = std::getenv("OPENBUENOS_BATCH_SIBLINGS")) {
        cfg.batch_siblings = (std::atoi(v) != 0);
    }
    if (const char * v = std::getenv("OPENBUENOS_MIN_ELEMENTS")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0) {
            cfg.min_elements = static_cast<uint64_t>(parsed);
        }
    }
    if (const char * v = std::getenv("OPENBUENOS_CACHE_CAP")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0 && parsed <= 1048576LL) {
            cfg.cache_cap = static_cast<uint32_t>(parsed);
        }
    }
    if (const char * v = std::getenv("OPENBUENOS_POOL_MB")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0 && parsed <= 2048LL) {
            cfg.pool_max_bytes = static_cast<uint32_t>(parsed) * 1024u * 1024u;
        }
    }
    if (const char * v = std::getenv("OPENBUENOS_TILE_M")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0 && parsed <= 4096LL) {
            cfg.tile_max_m = static_cast<uint32_t>(parsed);
        }
    }
    if (const char * v = std::getenv("OPENBUENOS_TILE_K")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0 && parsed <= 4096LL) {
            cfg.tile_max_k = static_cast<uint32_t>(parsed);
        }
    }
    if (const char * v = std::getenv("OPENBUENOS_TILE_N")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0 && parsed <= 4096LL) {
            cfg.tile_max_n = static_cast<uint32_t>(parsed);
        }
    }
    parse_env_u8("OPENBUENOS_CORE_POWER_STATE", &cfg.core_power_state);
    parse_env_u8("OPENBUENOS_MEM_POWER_STATE",  &cfg.mem_power_state);
    parse_env_u8("OPENBUENOS_QOS_CLASS",        &cfg.qos_class);
    parse_env_u8("OPENBUENOS_CLUSTER_ID",       &cfg.cluster_id);
    cfg.has_priority  = parse_env_u8("OPENBUENOS_PRIORITY",  &cfg.priority);
    cfg.has_perf_mode = parse_env_u8("OPENBUENOS_PERF_MODE", &cfg.perf_mode);
    if (const char * v = std::getenv("OPENBUENOS_DIVE_INTERP_PATH")) {
        if (v[0] != '\0') {
            cfg.dive_interp_path = v;
        }
    }
    return cfg;
}

static inline int8_t clamp_i8_sym(float x) {
    const int v = static_cast<int>(std::lround(x));
    if (v > 127) {
        return 127;
    }
    if (v < -127) {
        return -127;
    }
    return static_cast<int8_t>(v);
}

static void quantize_f32_rows_to_i8(const float * src,
                                    uint32_t      rows,
                                    uint32_t      cols,
                                    int8_t *      dst_q,
                                    float *       dst_scales) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float * row = src + static_cast<size_t>(r) * cols;
        int8_t *      q   = dst_q + static_cast<size_t>(r) * cols;

        float amax = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            const float a = std::fabs(row[c]);
            if (a > amax) {
                amax = a;
            }
        }

        if (amax == 0.0f) {
            dst_scales[r] = 0.0f;
            std::memset(q, 0, cols);
            continue;
        }

        const float scale     = amax / 127.0f;
        const float inv_scale = 127.0f / amax;
        dst_scales[r] = scale;

        for (uint32_t c = 0; c < cols; ++c) {
            q[c] = clamp_i8_sym(row[c] * inv_scale);
        }
    }
}

static_assert(QK4_0 == QK8_0, "ggml-edgetpu requires QK4_0 == QK8_0 == 32");
static_assert(sizeof(block_q8_0) == sizeof(ob_block_q8_0_t), "block_q8_0 must match ob_block_q8_0_t");
static_assert(sizeof(block_q4_0) == sizeof(ob_block_q4_0_t), "block_q4_0 must match ob_block_q4_0_t");

struct packed_q8_weight {
    uint32_t            k_dim = 0;
    uint32_t            n_dim = 0;
    std::vector<float>  row_scales;    // [n_dim] per-row float32 scale
    std::vector<int8_t> weights_i8_nk; // [n_dim, k_dim] row-major symmetric INT8
};

static std::shared_ptr<packed_q8_weight> repack_q8_0_to_buenos(const block_q8_0 * src,
                                                               uint32_t           k_dim,
                                                               uint32_t           n_dim,
                                                               int                verbose) {
    if (!src || k_dim == 0 || n_dim == 0 || (k_dim % QK8_0) != 0) {
        return nullptr;
    }

    auto entry = std::make_shared<packed_q8_weight>();
    entry->k_dim = k_dim;
    entry->n_dim = n_dim;
    entry->row_scales.resize(n_dim);
    entry->weights_i8_nk.resize(static_cast<size_t>(n_dim) * k_dim);

    const uint32_t n_blocks = k_dim / QK8_0;
    std::vector<float> tmp_row(k_dim);

    for (uint32_t n = 0; n < n_dim; ++n) {
        const block_q8_0 * row_blocks = src + static_cast<size_t>(n) * n_blocks;
        float amax = 0.0f;

        for (uint32_t b = 0; b < n_blocks; ++b) {
            const float d = GGML_FP16_TO_FP32(row_blocks[b].d);
            const int8_t * qs = row_blocks[b].qs;
            float * out_blk = tmp_row.data() + static_cast<size_t>(b) * QK8_0;
            for (uint32_t j = 0; j < QK8_0; ++j) {
                const float v = d * static_cast<float>(qs[j]);
                out_blk[j] = v;
                const float a = std::fabs(v);
                if (a > amax) {
                    amax = a;
                }
            }
        }

        int8_t * dst_row = entry->weights_i8_nk.data() + static_cast<size_t>(n) * k_dim;
        if (amax == 0.0f) {
            entry->row_scales[n] = 0.0f;
            std::memset(dst_row, 0, k_dim);
        } else {
            entry->row_scales[n] = amax / 127.0f;
            const float inv_s = 127.0f / amax;
            for (uint32_t k = 0; k < k_dim; ++k) {
                dst_row[k] = clamp_i8_sym(tmp_row[k] * inv_s);
            }
        }
    }

    if (verbose >= 2) {
        GGML_LOG_INFO("ggml-edgetpu: repacked Q8_0 weight [N=%u, K=%u] -> row-scaled INT8 (%zu bytes)\n",
                      n_dim, k_dim, entry->weights_i8_nk.size());
    }

    return entry;
}

// =========================================================================
// EDGETPU Backend Buffer (64B-aligned host-accessible + Q8_0 repack cache)
// =========================================================================

struct ggml_backend_edgetpu_buffer_context {
    void * data        = nullptr;
    size_t size        = 0;
    bool   owns_memory = false;

    std::mutex cache_mu;
    std::unordered_map<const void *, std::shared_ptr<packed_q8_weight>> packed_q8_cache;

    ~ggml_backend_edgetpu_buffer_context() {
        if (owns_memory && data) {
            std::free(data);
        }
    }
};

static bool is_edgetpu_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->buft == ggml_backend_edgetpu_buffer_type();
}

static void ggml_backend_edgetpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_edgetpu_buffer_context *>(buffer->context);
    delete ctx;
}

static void * ggml_backend_edgetpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_edgetpu_buffer_context *>(buffer->context);
    return ctx->data;
}

static void ggml_backend_edgetpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                      struct ggml_tensor *  tensor,
                                                      uint8_t               value,
                                                      size_t                offset,
                                                      size_t                size) {
    auto * ctx = static_cast<ggml_backend_edgetpu_buffer_context *>(buffer->context);
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);

    std::lock_guard<std::mutex> lock(ctx->cache_mu);
    ctx->packed_q8_cache.erase(tensor->data);
}

static void ggml_backend_edgetpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   struct ggml_tensor *  tensor,
                                                   const void *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto * ctx = static_cast<ggml_backend_edgetpu_buffer_context *>(buffer->context);
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);

    // Invalidate any stale repacked weight for this tensor address.
    {
        std::lock_guard<std::mutex> lock(ctx->cache_mu);
        ctx->packed_q8_cache.erase(tensor->data);
    }

    // Eagerly repack full 2D Q8_0 weight tensors at upload time into buenos [K//4, n_pad, 4] + 128 layout.
    const bool is_full_write = (offset == 0 && size == ggml_nbytes(tensor));
    const bool is_2d_q8_0 =
        tensor->type == GGML_TYPE_Q8_0 &&
        tensor->ne[0] > 0 && (tensor->ne[0] % QK8_0) == 0 &&
        tensor->ne[1] > 0 &&
        tensor->ne[2] == 1 &&
        tensor->ne[3] == 1 &&
        ggml_is_contiguous(tensor);

    if (is_full_write && is_2d_q8_0 && buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        const edgetpu_env_config env = load_env_config();
        if (env.packed_q8) {
            auto packed = repack_q8_0_to_buenos(
                static_cast<const block_q8_0 *>(tensor->data),
                static_cast<uint32_t>(tensor->ne[0]),
                static_cast<uint32_t>(tensor->ne[1]),
                env.verbose);
            if (packed) {
                std::lock_guard<std::mutex> lock(ctx->cache_mu);
                ctx->packed_q8_cache[tensor->data] = std::move(packed);
            }
        }
    }
}

static void ggml_backend_edgetpu_buffer_get_tensor(ggml_backend_buffer_t      buffer,
                                                   const struct ggml_tensor * tensor,
                                                   void *                     data,
                                                   size_t                     offset,
                                                   size_t                     size) {
    GGML_UNUSED(buffer);
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static bool ggml_backend_edgetpu_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   const struct ggml_tensor * src,
                                                   struct ggml_tensor *       dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_edgetpu_buffer_set_tensor(buffer, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_edgetpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<ggml_backend_edgetpu_buffer_context *>(buffer->context);
    if (ctx->data && ctx->size > 0) {
        std::memset(ctx->data, value, ctx->size);
    }
    std::lock_guard<std::mutex> lock(ctx->cache_mu);
    ctx->packed_q8_cache.clear();
}

static void ggml_backend_edgetpu_buffer_reset(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_edgetpu_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lock(ctx->cache_mu);
    ctx->packed_q8_cache.clear();
}

static const struct ggml_backend_buffer_i ggml_backend_edgetpu_buffer_i = {
    /* .free_buffer   = */ ggml_backend_edgetpu_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_edgetpu_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_edgetpu_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_edgetpu_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_edgetpu_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_edgetpu_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_edgetpu_buffer_clear,
    /* .reset         = */ ggml_backend_edgetpu_buffer_reset,
    /* .set_tensor_n  = */ NULL,
    /* .get_tensor_n  = */ NULL,
    /* .cpy_tensor_n  = */ NULL,
};

static const char * ggml_backend_edgetpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "EDGETPU";
}

static ggml_backend_buffer_t ggml_backend_edgetpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t alloc_size = (size == 0) ? EDGETPU_BUFFER_ALIGNMENT : ((size + EDGETPU_BUFFER_ALIGNMENT - 1) & ~(EDGETPU_BUFFER_ALIGNMENT - 1));
    void * ptr = nullptr;
    if (posix_memalign(&ptr, EDGETPU_BUFFER_ALIGNMENT, alloc_size) != 0 || !ptr) {
        GGML_LOG_ERROR("ggml-edgetpu: failed to allocate %zu bytes\n", alloc_size);
        return nullptr;
    }

    auto * ctx = new ggml_backend_edgetpu_buffer_context();
    ctx->data        = ptr;
    ctx->size        = size;
    ctx->owns_memory = true;

    return ggml_backend_buffer_init(buft, ggml_backend_edgetpu_buffer_i, ctx, size);
}

static size_t ggml_backend_edgetpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return EDGETPU_BUFFER_ALIGNMENT;
}

static bool ggml_backend_edgetpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // Primary storage keeps canonical ggml tensor layout in 64B-aligned host RAM so
    // ARM64 CPU glue ops (RMS_NORM, ROPE, SOFT_MAX, GET_ROWS, etc.) read directly with zero copies,
    // while packed buenos weights live in the buffer's sidecar cache.
    return true;
}

static ggml_backend_buffer_t ggml_backend_edgetpu_buffer_from_host_ptr(void * ptr, size_t size) {
    auto * ctx = new ggml_backend_edgetpu_buffer_context();
    ctx->data        = ptr;
    ctx->size        = size;
    ctx->owns_memory = false;

    return ggml_backend_buffer_init(ggml_backend_edgetpu_buffer_type(), ggml_backend_edgetpu_buffer_i, ctx, size);
}

// =========================================================================
// EDGETPU Backend Stream Context & MUL_MAT Execution
// =========================================================================

struct ggml_backend_edgetpu_context {
    ob_context_t *     ob_ctx = nullptr;
    edgetpu_env_config env;

    // Reusable per-stream scratch buffers for dynamic quantization & dequantization
    std::vector<int8_t>  q_a;
    std::vector<int8_t>  q_b;
    std::vector<float>   scale_a;
    std::vector<float>   scale_b;
    std::vector<int32_t> c_i32;
    std::vector<float>   f32_a;

    // Fallback repack cache for weights residing in non-EDGETPU host buffers (e.g. mmap/CPU)
    std::mutex fallback_cache_mu;
    std::unordered_map<const void *, std::shared_ptr<packed_q8_weight>> fallback_packed_cache;

    // Cumulative execution telemetry for OPENBUENOS_VERBOSE=1 shutdown summary
    uint64_t total_mul_mat_calls       = 0;
    uint64_t total_slice_calls         = 0;
    uint64_t total_emitter_slices      = 0;
    uint64_t total_cpu_fallback_slices = 0;
    uint64_t total_tiles_dispatched    = 0;
    uint64_t total_tile_cache_hits     = 0;
    uint64_t total_hw_us               = 0;
    uint64_t total_wall_us             = 0;
    uint64_t total_batched_groups      = 0;
    uint64_t total_batched_ops         = 0;
    uint64_t total_bias_adds           = 0;

    ~ggml_backend_edgetpu_context() {
        if (ob_ctx) {
            if (env.verbose > 0) {
                size_t   cache_entries   = 0;
                uint64_t cache_hits      = 0;
                uint64_t cache_misses    = 0;
                uint64_t cache_evictions = 0;
                ob_context_graph_cache_stats(ob_ctx, &cache_entries, &cache_hits, &cache_misses, &cache_evictions);

                size_t   pool_bytes  = 0;
                size_t   pool_count  = 0;
                uint64_t pool_hits   = 0;
                uint64_t pool_misses = 0;
                ob_context_buffer_pool_stats(ob_ctx, &pool_bytes, &pool_count, &pool_hits, &pool_misses);

                const uint64_t cache_total = cache_hits + cache_misses;
                const double   hit_pct     = (cache_total > 0)
                                                 ? (100.0 * static_cast<double>(cache_hits) / static_cast<double>(cache_total))
                                                 : 0.0;
                GGML_LOG_INFO(
                    "ggml-edgetpu: summary mock=%d calls=%llu slices=%llu emitter_slices=%llu cpu_fallback_slices=%llu "
                    "batched_groups=%llu batched_ops=%llu bias_adds=%llu "
                    "tiles=%llu tile_hits=%llu hw_us=%llu wall_us=%llu "
                    "cache_entries=%zu cache_hits=%llu cache_misses=%llu cache_evictions=%llu cache_hit_rate=%.2f%% "
                    "pool_bytes=%zu pool_count=%zu pool_hits=%llu pool_misses=%llu\n",
                    ob_context_is_mock(ob_ctx),
                    static_cast<unsigned long long>(total_mul_mat_calls),
                    static_cast<unsigned long long>(total_slice_calls),
                    static_cast<unsigned long long>(total_emitter_slices),
                    static_cast<unsigned long long>(total_cpu_fallback_slices),
                    static_cast<unsigned long long>(total_batched_groups),
                    static_cast<unsigned long long>(total_batched_ops),
                    static_cast<unsigned long long>(total_bias_adds),
                    static_cast<unsigned long long>(total_tiles_dispatched),
                    static_cast<unsigned long long>(total_tile_cache_hits),
                    static_cast<unsigned long long>(total_hw_us),
                    static_cast<unsigned long long>(total_wall_us),
                    cache_entries,
                    static_cast<unsigned long long>(cache_hits),
                    static_cast<unsigned long long>(cache_misses),
                    static_cast<unsigned long long>(cache_evictions),
                    hit_pct,
                    pool_bytes,
                    pool_count,
                    static_cast<unsigned long long>(pool_hits),
                    static_cast<unsigned long long>(pool_misses));
            }
            ob_context_destroy(ob_ctx);
            ob_ctx = nullptr;
        }
    }
};

static std::shared_ptr<packed_q8_weight> get_or_repack_q8_weight(ggml_backend_edgetpu_context * ctx,
                                                                 const ggml_tensor *            src0,
                                                                 const void *                   slice_ptr,
                                                                 uint32_t                       k_dim,
                                                                 uint32_t                       n_dim) {
    if (is_edgetpu_buffer(src0->buffer)) {
        auto * bctx = static_cast<ggml_backend_edgetpu_buffer_context *>(src0->buffer->context);
        {
            std::lock_guard<std::mutex> lock(bctx->cache_mu);
            auto it = bctx->packed_q8_cache.find(slice_ptr);
            if (it != bctx->packed_q8_cache.end() && it->second->k_dim == k_dim && it->second->n_dim == n_dim) {
                return it->second;
            }
        }
        auto packed = repack_q8_0_to_buenos(static_cast<const block_q8_0 *>(slice_ptr), k_dim, n_dim, ctx->env.verbose);
        if (packed) {
            std::lock_guard<std::mutex> lock(bctx->cache_mu);
            bctx->packed_q8_cache[slice_ptr] = packed;
        }
        return packed;
    }

    // For mmap/CPU host weight buffers, cache only when marked WEIGHTS (so dynamic test buffers never go stale).
    if (src0->buffer && src0->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        std::lock_guard<std::mutex> lock(ctx->fallback_cache_mu);
        auto it = ctx->fallback_packed_cache.find(slice_ptr);
        if (it != ctx->fallback_packed_cache.end() && it->second->k_dim == k_dim && it->second->n_dim == n_dim) {
            return it->second;
        }
        auto packed = repack_q8_0_to_buenos(static_cast<const block_q8_0 *>(slice_ptr), k_dim, n_dim, ctx->env.verbose);
        if (packed) {
            ctx->fallback_packed_cache[slice_ptr] = packed;
        }
        return packed;
    }

    return repack_q8_0_to_buenos(static_cast<const block_q8_0 *>(slice_ptr), k_dim, n_dim, ctx->env.verbose);
}

static ggml_status ggml_edgetpu_compute_forward_mul_mat(ggml_backend_edgetpu_context * ctx,
                                                        struct ggml_tensor *           dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // weights [K, N, bs0, bs1]
    const struct ggml_tensor * src1 = dst->src[1]; // inputs  [K, M, bs0*r2, bs1*r3]

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);
    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const uint32_t M = static_cast<uint32_t>(ne11);
    const uint32_t K = static_cast<uint32_t>(ne10);
    const uint32_t N = static_cast<uint32_t>(ne01);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    uint64_t total_hw_us   = 0;
    uint64_t total_wall_us = 0;
    uint32_t total_tiles   = 0;
    uint32_t total_hits    = 0;
    uint16_t merged_flags  = 0;

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const char * src0_ptr = static_cast<const char *>(src0->data) + (i12 / r2) * nb02 + (i13 / r3) * nb03;
            const char * src1_ptr = static_cast<const char *>(src1->data) + i12 * nb12 + i13 * nb13;
            float *      dst_ptr  = reinterpret_cast<float *>(static_cast<char *>(dst->data) + i12 * nb2 + i13 * nb3);

            ob_invoke_stats_t stats {};
            ob_status_t st = OB_OK;

            if (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) {
                const float * a_f32 = nullptr;
                if (src1->type == GGML_TYPE_F32) {
                    a_f32 = reinterpret_cast<const float *>(src1_ptr);
                } else if (src1->type == GGML_TYPE_Q8_0) {
                    ctx->f32_a.resize(static_cast<size_t>(M) * K);
                    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
                    for (uint32_t m = 0; m < M; ++m) {
                        traits->to_float(src1_ptr + static_cast<size_t>(m) * nb11,
                                         ctx->f32_a.data() + static_cast<size_t>(m) * K,
                                         K);
                    }
                    a_f32 = ctx->f32_a.data();
                } else {
                    return GGML_STATUS_FAILED;
                }

                if (src0->type == GGML_TYPE_Q4_0) {
                    // Native zero-copy raw Q4_0 dispatch into libopenbuenos O(1) content-hashed ext[2] DMA-BUF LRU
                    // (unpacks nibbles once on cache miss; zero per-token CPU unpack, zero duplicate host RAM).
                    ob_gemm_f32_quant_job_t job {};
                    job.b_nk       = src0_ptr;
                    job.c_mn       = dst_ptr;
                    job.n_dim      = N;
                    job.activation = OB_ACT_NONE;
                    job.is_q4_0    = 1u;
                    st = ob_gemm_matmul_f32_quant_batch(ctx->ob_ctx, a_f32, M, K, &job, 1u, &stats);
                    if (st != OB_OK) {
                        GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_f32_quant_batch (Q4_0 M=%u,K=%u,N=%u) failed: %s\n",
                                       M, K, N, ob_status_string(st));
                        return GGML_STATUS_FAILED;
                    }
                } else if (ctx->env.packed_q8) {
                    // Per-row packed INT8 path using pre-packed buenos weights + ob_gemm_matmul_int8.
                    auto packed = get_or_repack_q8_weight(ctx, src0, src0_ptr, K, N);
                    if (!packed) {
                        return GGML_STATUS_FAILED;
                    }
                    ctx->q_a.resize(static_cast<size_t>(M) * K);
                    ctx->scale_a.resize(M);
                    ctx->c_i32.resize(static_cast<size_t>(M) * N);

                    quantize_f32_rows_to_i8(a_f32, M, K, ctx->q_a.data(), ctx->scale_a.data());
                    st = ob_gemm_matmul_int8(
                        ctx->ob_ctx,
                        ctx->q_a.data(),
                        packed->weights_i8_nk.data(),
                        nullptr,
                        ctx->c_i32.data(),
                        M, K, N,
                        0, 0,
                        &stats);
                    if (st != OB_OK) {
                        GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_int8 (packed Q8_0) failed: %s\n", ob_status_string(st));
                        return GGML_STATUS_FAILED;
                    }
                    for (uint32_t m = 0; m < M; ++m) {
                        const float sa = ctx->scale_a[m];
                        const int32_t * c_row = ctx->c_i32.data() + static_cast<size_t>(m) * N;
                        float * out_row = dst_ptr + static_cast<size_t>(m) * N;
                        for (uint32_t n = 0; n < N; ++n) {
                            out_row[n] = static_cast<float>(c_row[n]) * (sa * packed->row_scales[n]);
                        }
                    }
                } else {
                    // Direct libopenbuenos tiled F32 x Q8_0 path (preserves per-32-element block scales).
                    st = ob_gemm_matmul_f32_q8_0(
                        ctx->ob_ctx,
                        a_f32,
                        src0_ptr,
                        dst_ptr,
                        M, K, N,
                        &stats);
                    if (st != OB_OK) {
                        GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_f32_q8_0(M=%u,K=%u,N=%u) failed: %s\n",
                                       M, K, N, ob_status_string(st));
                        return GGML_STATUS_FAILED;
                    }
                }
            } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
                // Dynamic per-row symmetric INT8 quantization -> buenos tiled INT8 matmul -> F32 dequantize.
                ctx->q_a.resize(static_cast<size_t>(M) * K);
                ctx->q_b.resize(static_cast<size_t>(N) * K);
                ctx->scale_a.resize(M);
                ctx->scale_b.resize(N);
                ctx->c_i32.resize(static_cast<size_t>(M) * N);

                quantize_f32_rows_to_i8(reinterpret_cast<const float *>(src1_ptr), M, K, ctx->q_a.data(), ctx->scale_a.data());
                quantize_f32_rows_to_i8(reinterpret_cast<const float *>(src0_ptr), N, K, ctx->q_b.data(), ctx->scale_b.data());

                st = ob_gemm_matmul_int8(
                    ctx->ob_ctx,
                    ctx->q_a.data(),
                    ctx->q_b.data(),
                    nullptr,
                    ctx->c_i32.data(),
                    M, K, N,
                    0, 0,
                    &stats);
                if (st != OB_OK) {
                    GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_int8(M=%u,K=%u,N=%u) failed: %s\n",
                                   M, K, N, ob_status_string(st));
                    return GGML_STATUS_FAILED;
                }

                for (uint32_t m = 0; m < M; ++m) {
                    const float sa = ctx->scale_a[m];
                    const int32_t * c_row = ctx->c_i32.data() + static_cast<size_t>(m) * N;
                    float * out_row = dst_ptr + static_cast<size_t>(m) * N;
                    for (uint32_t n = 0; n < N; ++n) {
                        out_row[n] = static_cast<float>(c_row[n]) * (sa * ctx->scale_b[n]);
                    }
                }
            } else {
                return GGML_STATUS_FAILED;
            }

            total_hw_us   += stats.hw_execution_us;
            total_wall_us += stats.wall_time_us;
            total_tiles   += stats.tiles_dispatched;
            total_hits    += stats.cache_hits;
            merged_flags  |= stats.flags;

            ctx->total_slice_calls += 1;
            if (stats.flags & OB_STAT_FLAG_EMITTER_USED) {
                ctx->total_emitter_slices += 1;
            }
            if (stats.flags & OB_STAT_FLAG_CPU_FALLBACK) {
                ctx->total_cpu_fallback_slices += 1;
                if (!ctx->env.mock_mode && ctx->total_cpu_fallback_slices <= 8) {
                    GGML_LOG_WARN("ggml-edgetpu: CPU fallback triggered on MUL_MAT (%s x %s) M=%u K=%u N=%u (flags=0x%x)\n",
                                  ggml_type_name(src0->type), ggml_type_name(src1->type),
                                  M, K, N, static_cast<unsigned>(stats.flags));
                }
            }
        }
    }

    ctx->total_mul_mat_calls    += 1;
    ctx->total_tiles_dispatched += total_tiles;
    ctx->total_tile_cache_hits  += total_hits;
    ctx->total_hw_us            += total_hw_us;
    ctx->total_wall_us          += total_wall_us;

    if (ctx->env.verbose >= 2 || (ctx->env.verbose == 1 && ctx->total_mul_mat_calls <= 16)) {
        GGML_LOG_INFO("ggml-edgetpu: MUL_MAT (%s x %s) M=%u K=%u N=%u batch=[%lld,%lld] tiles=%u hits=%u hw_us=%llu wall_us=%llu flags=0x%x\n",
                      ggml_type_name(src0->type),
                      ggml_type_name(src1->type),
                      M, K, N,
                      static_cast<long long>(ne12),
                      static_cast<long long>(ne13),
                      total_tiles,
                      total_hits,
                      static_cast<unsigned long long>(total_hw_us),
                      static_cast<unsigned long long>(total_wall_us),
                      static_cast<unsigned>(merged_flags));
    }

    return GGML_STATUS_SUCCESS;
}

// =========================================================================
// Backend Interface
// =========================================================================

static const char * ggml_backend_edgetpu_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "EDGETPU";
}

static void ggml_backend_edgetpu_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_edgetpu_context *>(backend->context);
    delete ctx;
    delete backend;
}

static ggml_status ggml_edgetpu_compute_forward_add_bias(ggml_backend_edgetpu_context * ctx,
                                                         struct ggml_tensor *           dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[0] == dst->ne[0] && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t n_cols = dst->ne[0];
    const int64_t n_rows = ggml_nrows(dst);
    const float * x      = reinterpret_cast<const float *>(src0->data);
    const float * bias   = reinterpret_cast<const float *>(src1->data);
    float *       y      = reinterpret_cast<float *>(dst->data);

    for (int64_t r = 0; r < n_rows; ++r) {
        const float * xr = x + r * n_cols;
        float *       yr = y + r * n_cols;
        for (int64_t c = 0; c < n_cols; ++c) {
            yr[c] = xr[c] + bias[c];
        }
    }

    ctx->total_bias_adds += 1;
    return GGML_STATUS_SUCCESS;
}

static bool is_batchable_sibling_mul_mat(const struct ggml_tensor * node,
                                         const struct ggml_tensor * shared_src1) {
    if (!node || node->op != GGML_OP_MUL_MAT || node->type != GGML_TYPE_F32) {
        return false;
    }
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1) {
        return false;
    }
    if ((src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_0) || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (shared_src1) {
        if (src1->data != shared_src1->data || src1->ne[0] != shared_src1->ne[0] || src1->ne[1] != shared_src1->ne[1]) {
            return false;
        }
    }
    if (node->data == src1->data) {
        return false;
    }
    return true;
}

static ggml_status ggml_backend_edgetpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_edgetpu_context *>(backend->context);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                // Collect sibling Q/K/V or up/gate MUL_MATs sharing the same input activation src1,
                // deferring intervening post-MUL_MAT 1D bias ADDs until the sibling group completes.
                if (ctx->env.batch_siblings && !ctx->env.packed_q8 && is_batchable_sibling_mul_mat(node, nullptr)) {
                    constexpr int MAX_SIBLINGS = 8;
                    struct ggml_tensor * mm_nodes[MAX_SIBLINGS];
                    struct ggml_tensor * deferred_adds[MAX_SIBLINGS];
                    int n_mm  = 0;
                    int n_add = 0;
                    const struct ggml_tensor * shared_x = node->src[1];
                    mm_nodes[n_mm++] = node;

                    int j = i + 1;
                    while (j < cgraph->n_nodes && n_mm < MAX_SIBLINGS) {
                        struct ggml_tensor * next = cgraph->nodes[j];
                        if ((next->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                            next->op == GGML_OP_NONE ||
                            next->op == GGML_OP_RESHAPE ||
                            next->op == GGML_OP_VIEW ||
                            next->op == GGML_OP_PERMUTE ||
                            next->op == GGML_OP_TRANSPOSE) {
                            ++j;
                            continue;
                        }
                        if (next->op == GGML_OP_ADD && n_add < MAX_SIBLINGS) {
                            bool depends_on_group = false;
                            for (int k = 0; k < n_mm; ++k) {
                                if (next->src[0] == mm_nodes[k]) {
                                    depends_on_group = true;
                                    break;
                                }
                            }
                            if (depends_on_group && next->data != shared_x->data) {
                                deferred_adds[n_add++] = next;
                                ++j;
                                continue;
                            }
                            break;
                        }
                        if (is_batchable_sibling_mul_mat(next, shared_x)) {
                            bool hazard = false;
                            for (int k = 0; k < n_mm; ++k) {
                                if (next->src[0]->data == mm_nodes[k]->data ||
                                    next->data == mm_nodes[k]->src[0]->data) {
                                    hazard = true;
                                    break;
                                }
                            }
                            if (hazard) {
                                break;
                            }
                            mm_nodes[n_mm++] = next;
                            ++j;
                            continue;
                        }
                        break;
                    }

                    if (n_mm > 1) {
                        ctx->total_batched_groups += 1;
                        ctx->total_batched_ops    += static_cast<uint64_t>(n_mm);

                        ob_gemm_f32_quant_job_t jobs[MAX_SIBLINGS];
                        for (int k = 0; k < n_mm; ++k) {
                            jobs[k]            = {};
                            jobs[k].b_nk       = mm_nodes[k]->src[0]->data;
                            jobs[k].c_mn       = reinterpret_cast<float *>(mm_nodes[k]->data);
                            jobs[k].n_dim      = static_cast<uint32_t>(mm_nodes[k]->src[0]->ne[1]);
                            jobs[k].activation = OB_ACT_NONE;
                            jobs[k].is_q4_0    = (mm_nodes[k]->src[0]->type == GGML_TYPE_Q4_0) ? 1u : 0u;
                        }
                        ob_invoke_stats_t bstats {};
                        const ob_status_t bst = ob_gemm_matmul_f32_quant_batch(
                            ctx->ob_ctx,
                            reinterpret_cast<const float *>(shared_x->data),
                            static_cast<uint32_t>(shared_x->ne[1]),
                            static_cast<uint32_t>(shared_x->ne[0]),
                            jobs,
                            static_cast<uint32_t>(n_mm),
                            &bstats);
                        if (bst != OB_OK) {
                            GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_f32_quant_batch(n=%d,M=%lld,K=%lld) failed: %s\n",
                                           n_mm, (long long)shared_x->ne[1], (long long)shared_x->ne[0],
                                           ob_status_string(bst));
                            return GGML_STATUS_FAILED;
                        }
                        ctx->total_mul_mat_calls    += static_cast<uint64_t>(n_mm);
                        ctx->total_slice_calls      += static_cast<uint64_t>(n_mm);
                        ctx->total_tiles_dispatched += bstats.tiles_dispatched;
                        ctx->total_tile_cache_hits  += bstats.cache_hits;
                        ctx->total_hw_us            += bstats.hw_execution_us;
                        ctx->total_wall_us          += bstats.wall_time_us;
                        if (bstats.flags & OB_STAT_FLAG_EMITTER_USED) {
                            ctx->total_emitter_slices += static_cast<uint64_t>(n_mm);
                        }
                        if (bstats.flags & OB_STAT_FLAG_CPU_FALLBACK) {
                            ctx->total_cpu_fallback_slices += static_cast<uint64_t>(n_mm);
                            if (!ctx->env.mock_mode && ctx->total_cpu_fallback_slices <= 8) {
                                GGML_LOG_WARN("ggml-edgetpu: CPU fallback in sibling batch(n=%d,M=%lld,K=%lld,flags=0x%x)\n",
                                              n_mm, (long long)shared_x->ne[1], (long long)shared_x->ne[0],
                                              static_cast<unsigned>(bstats.flags));
                            }
                        }
                        for (int k = 0; k < n_add; ++k) {
                            const ggml_status st = ggml_edgetpu_compute_forward_add_bias(ctx, deferred_adds[k]);
                            if (st != GGML_STATUS_SUCCESS) {
                                return st;
                            }
                        }
                        i = j - 1;
                        break;
                    }
                }

                const ggml_status st = ggml_edgetpu_compute_forward_mul_mat(ctx, node);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                break;
            }
            case GGML_OP_ADD: {
                const ggml_status st = ggml_edgetpu_compute_forward_add_bias(ctx, node);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                break;
            }
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("ggml-edgetpu: unsupported op %s\n", ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_edgetpu_i = {
    /* .get_name            = */ ggml_backend_edgetpu_get_name,
    /* .free                = */ ggml_backend_edgetpu_free,
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
    /* .graph_compute       = */ ggml_backend_edgetpu_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid_t ggml_backend_edgetpu_guid(void) {
    static const uint8_t guid[16] = {
        0x4f, 0x42, 0x55, 0x45, 0x4e, 0x4f, 0x53, 0x54, // "OBUENOST"
        0x50, 0x55, 0x47, 0x35, 0x41, 0x30, 0x00, 0x01  // "PUG5A0\0\1"
    };
    return const_cast<ggml_guid_t>(&guid);
}

// =========================================================================
// Device Interface
// =========================================================================

static const char * ggml_backend_edgetpu_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "EDGETPU";
}

static const char * ggml_backend_edgetpu_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Pixel Tensor G5 (buenos_a0)";
}

static void ggml_backend_edgetpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_edgetpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_edgetpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_edgetpu_device_get_name(dev);
    props->description = ggml_backend_edgetpu_device_get_description(dev);
    props->type        = ggml_backend_edgetpu_device_get_type(dev);
    ggml_backend_edgetpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_edgetpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_edgetpu_init();
}

static ggml_backend_buffer_type_t ggml_backend_edgetpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_edgetpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_edgetpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                                               void *             ptr,
                                                                               size_t             size,
                                                                               size_t             max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_edgetpu_buffer_from_host_ptr(ptr, size);
}

static bool ggml_backend_edgetpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (!src0 || !src1) {
                return false;
            }
            if (op->type != GGML_TYPE_F32) {
                return false;
            }

            // Respect explicit unquantized FP32/BF16 precision requests & Hadamard hint.
            const enum ggml_prec prec = ggml_prec_get_src(op, 1);
            if (prec == GGML_PREC_F32 || prec == GGML_PREC_BF16) {
                return false;
            }
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }

            // Require contiguous non-permuted layouts.
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }
            if (src0->nb[0] != ggml_type_size(src0->type) ||
                src1->nb[0] != ggml_type_size(src1->type) ||
                src0->nb[1] != ggml_row_size(src0->type, src0->ne[0]) ||
                src1->nb[1] != ggml_row_size(src1->type, src1->ne[0])) {
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

            // Supported quant / precision regimes:
            //   - Q8_0 x F32 / Q8_0 (K % 32 == 0)
            //   - Q4_0 x F32 / Q8_0 (K % 32 == 0; lossless block-32 unpack into bounded scratch -> ext[2] DMA-BUF LRU)
            //   - F32  x F32        (dynamic per-row INT8 quantize -> buenos INT8 GEMM -> F32 dequantize;
            //                        requires K % 4 == 0 and N >= 8 matching Tensor G5's 8-channel n_pad width)
            bool dtype_ok = false;
            if ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) &&
                (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_Q8_0)) {
                dtype_ok = ((K % QK8_0) == 0);
            } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
                dtype_ok = ((K % 4) == 0 && N >= 8);
            }
            if (!dtype_ok) {
                return false;
            }

            const edgetpu_env_config env = load_env_config();
            if (env.min_elements > 0) {
                const uint64_t work =
                    static_cast<uint64_t>(M) *
                    static_cast<uint64_t>(K) *
                    static_cast<uint64_t>(N) *
                    static_cast<uint64_t>(src1->ne[2]) *
                    static_cast<uint64_t>(src1->ne[3]);
                if (work < env.min_elements) {
                    return false;
                }
            }

            return true;
        }

        case GGML_OP_ADD: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (!src0 || !src1) {
                return false;
            }
            // Accept ONLY post-MUL_MAT 1D F32 bias broadcast [N, 1, 1, 1] whose producer MUL_MAT runs on EDGETPU,
            // keeping Qwen2.5 Q/K/V projections in a single EDGETPU split for sibling async batching.
            if (op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->op != GGML_OP_MUL_MAT || !ggml_backend_edgetpu_device_supports_op(dev, src0)) {
                return false;
            }
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
                return false;
            }
            if (src1->ne[0] != src0->ne[0] || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
                return false;
            }
            return true;
        }

        // Explicitly return false for RMS_NORM, ROPE, SOFT_MAX, MUL_MAT_ID, GET_ROWS, etc.
        // so ggml-backend-sched automatically routes glue ops to ARM64 CPU.
        default:
            return false;
    }
}

static bool ggml_backend_edgetpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft == ggml_backend_edgetpu_buffer_type() || ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_edgetpu_device_i = {
    /* .get_name             = */ ggml_backend_edgetpu_device_get_name,
    /* .get_description      = */ ggml_backend_edgetpu_device_get_description,
    /* .get_memory           = */ ggml_backend_edgetpu_device_get_memory,
    /* .get_type             = */ ggml_backend_edgetpu_device_get_type,
    /* .get_props            = */ ggml_backend_edgetpu_device_get_props,
    /* .init_backend         = */ ggml_backend_edgetpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_edgetpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_edgetpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_edgetpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_edgetpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// =========================================================================
// Registry Interface
// =========================================================================

static const char * ggml_backend_edgetpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "EDGETPU";
}

static size_t ggml_backend_edgetpu_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_edgetpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static struct ggml_backend_device ggml_backend_edgetpu_device = {
        /* .iface   = */ ggml_backend_edgetpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };
    return &ggml_backend_edgetpu_device;
}

static const struct ggml_backend_reg_i ggml_backend_edgetpu_reg_i = {
    /* .get_name         = */ ggml_backend_edgetpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_edgetpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_edgetpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

} // namespace

// =========================================================================
// Public C API
// =========================================================================

ggml_backend_buffer_type_t ggml_backend_edgetpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_edgetpu_buft = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_edgetpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_edgetpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ ggml_backend_edgetpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .get_alloc_size_n = */ NULL,
            /* .is_host          = */ ggml_backend_edgetpu_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_edgetpu_reg(), 0),
        /* .context = */ NULL,
    };
    return &ggml_backend_edgetpu_buft;
}

ggml_backend_t ggml_backend_edgetpu_init(void) {
    const edgetpu_env_config env = load_env_config();

    ob_config_t ob_cfg = ob_config_default();
    ob_cfg.mock_mode             = env.mock_mode ? 1 : 0;
    ob_cfg.graph_cache_capacity  = env.cache_cap;
    ob_cfg.buffer_pool_max_bytes = env.pool_max_bytes;
    if (env.core_power_state != 0u) {
        ob_cfg.core_power_state = env.core_power_state;
    }
    if (env.mem_power_state != 0u) {
        ob_cfg.mem_power_state = env.mem_power_state;
    }
    if (env.qos_class != 0u) {
        ob_cfg.qos_class = env.qos_class;
    }
    if (env.cluster_id != 0u) {
        ob_cfg.cluster_id = env.cluster_id;
    }
    if (env.has_priority) {
        ob_cfg.priority = env.priority;
    }
    if (env.has_perf_mode) {
        ob_cfg.perf_mode = env.perf_mode;
    }

    ob_context_t * ob_ctx = nullptr;
    ob_status_t st = ob_context_create(&ob_cfg, &ob_ctx);
    if (st != OB_OK || !ob_ctx) {
        GGML_LOG_ERROR("ggml-edgetpu: ob_context_create(mock_mode=%d, cache_cap=%u, pool_bytes=%u) failed: %s\n",
                       ob_cfg.mock_mode, ob_cfg.graph_cache_capacity, ob_cfg.buffer_pool_max_bytes,
                       ob_status_string(st));
        return nullptr;
    }

    const char * interp_path = (env.dive_interp_path && env.dive_interp_path[0] != '\0')
                                   ? env.dive_interp_path
                                   : EDGETPU_DEFAULT_DIVE_INTERP_PATH;
    bool emitter_ok = false;
    if (!env.mock_mode) {
        st = ob_gemm_register_builtin_emitter(ob_ctx, interp_path);
        if (st != OB_OK) {
            GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_register_builtin_emitter('%s') failed (%s): %s\n",
                           interp_path, ob_status_string(st), ob_context_last_error(ob_ctx));
            GGML_LOG_ERROR("ggml-edgetpu: refusing silent CPU fallback in hardware mode (mock_mode=0); "
                           "ensure dive_interp_rv64.bin exists or set OPENBUENOS_DIVE_INTERP_PATH / OPENBUENOS_MOCK=1\n");
            ob_context_destroy(ob_ctx);
            return nullptr;
        }
        emitter_ok = true;
    }

    if (env.tile_max_m > 0u || env.tile_max_k > 0u || env.tile_max_n > 0u) {
        const uint32_t tm = env.tile_max_m > 0u ? env.tile_max_m : 16u;
        const uint32_t tk = env.tile_max_k > 0u ? env.tile_max_k : 512u;
        const uint32_t tn = env.tile_max_n > 0u ? env.tile_max_n : 128u;
        st = ob_gemm_set_tile_limits(ob_ctx, tm, tk, tn);
        if (st != OB_OK) {
            GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_set_tile_limits(%u, %u, %u) failed: %s\n",
                           tm, tk, tn, ob_status_string(st));
            ob_context_destroy(ob_ctx);
            return nullptr;
        }
    }

    auto * ctx = new ggml_backend_edgetpu_context();
    ctx->ob_ctx = ob_ctx;
    ctx->env    = env;

    if (env.verbose > 0) {
        GGML_LOG_INFO(
            "ggml-edgetpu: initialized Pixel Tensor G5 (buenos_a0) "
            "[mock=%d, emitter=%d, interp=%s, credits=%u, cache_cap=%u, pool_mb=%u, "
            "tile_limits=(%u,%u,%u), pwr=(perf=%u,core=0x%02x,mem=0x%02x,qos=0x%02x,prio=%u), "
            "batch_siblings=%d, min_elements=%llu, packed_q8=%d, libopenbuenos=%d]\n",
            ob_context_is_mock(ob_ctx),
            emitter_ok ? 1 : 0,
            env.mock_mode ? "(mock)" : interp_path,
            ob_context_credit_size(ob_ctx),
            env.cache_cap,
            env.pool_max_bytes / (1024u * 1024u),
            env.tile_max_m, env.tile_max_k, env.tile_max_n,
            static_cast<unsigned>(ob_cfg.perf_mode),
            static_cast<unsigned>(ob_cfg.core_power_state),
            static_cast<unsigned>(ob_cfg.mem_power_state),
            static_cast<unsigned>(ob_cfg.qos_class),
            static_cast<unsigned>(ob_cfg.priority),
            env.batch_siblings ? 1 : 0,
            static_cast<unsigned long long>(env.min_elements),
            env.packed_q8 ? 1 : 0,
            GGML_EDGETPU_LINKED_LIBOPENBUENOS);
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_edgetpu_guid(),
        /* .iface   = */ ggml_backend_edgetpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_edgetpu_reg(), 0),
        /* .context = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_edgetpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_edgetpu_guid());
}

bool ggml_backend_edgetpu_is_mock(ggml_backend_t backend) {
    if (!ggml_backend_is_edgetpu(backend)) {
        return true;
    }
    const auto * ctx = static_cast<const ggml_backend_edgetpu_context *>(backend->context);
    return ctx && ctx->ob_ctx ? (ob_context_is_mock(ctx->ob_ctx) != 0) : true;
}

ggml_backend_reg_t ggml_backend_edgetpu_reg(void) {
    static struct ggml_backend_reg ggml_backend_edgetpu_reg_s = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_edgetpu_reg_i,
        /* .context     = */ NULL,
    };
    return &ggml_backend_edgetpu_reg_s;
}

GGML_BACKEND_DL_IMPL(ggml_backend_edgetpu_reg)
