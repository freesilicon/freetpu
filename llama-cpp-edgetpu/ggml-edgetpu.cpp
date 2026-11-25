#include "ggml-edgetpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "openbuenos.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr size_t   EDGETPU_BUFFER_ALIGNMENT       = 64;
constexpr uint32_t EDGETPU_DEFAULT_CACHE_CAP      = 2048u;
constexpr uint32_t EDGETPU_DEFAULT_POOL_MAX_BYTES = 128u * 1024u * 1024u;
constexpr int64_t  EDGETPU_DEFAULT_GLU_MAX_M      = 64;

struct edgetpu_env_config {
    bool         mock_mode             = false;
    int          verbose               = 0;
    bool         packed_q8             = false; // Strictly off by default for zero-dup single-copy weight residency
    bool         batch_siblings        = true;
    bool         fuse_swiglu           = true;
    bool         enable_rope           = true;
    bool         enable_glu            = true;
    int64_t      glu_max_m             = EDGETPU_DEFAULT_GLU_MAX_M;
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
    bool         has_freetile          = false;
    bool         freetile              = false; // OPENBUENOS_FREETILE=1 enables FreeTile BUENOS_A0 emitter
    const char * dive_interp_path      = nullptr;
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
#elif defined(__linux__) && (defined(__aarch64__) || defined(__arm__))
    // Native ARM/ARM64 Linux (e.g. OmaMobile / Arch Linux ARM): run hardware
    // mode when /dev/edgetpu-soc exists, otherwise fall back to mock mode.
    cfg.mock_mode = (::access("/dev/edgetpu-soc", F_OK) != 0);
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
    if (const char * v = std::getenv("OPENBUENOS_FUSE_SWIGLU")) {
        cfg.fuse_swiglu = (std::atoi(v) != 0);
    }
    if (const char * v = std::getenv("OPENBUENOS_EDGETPU_ROPE")) {
        cfg.enable_rope = (std::atoi(v) != 0);
    }
    if (const char * v = std::getenv("OPENBUENOS_EDGETPU_GLU")) {
        cfg.enable_glu = (std::atoi(v) != 0);
    }
    if (const char * v = std::getenv("OPENBUENOS_EDGETPU_GLU_MAX_M")) {
        const long long parsed = std::atoll(v);
        cfg.glu_max_m = (parsed > 0) ? static_cast<int64_t>(parsed) : 0;
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
    if (const char * v = std::getenv("OPENBUENOS_FREETILE")) {
        if (v[0] != '\0') {
            cfg.has_freetile = true;
            cfg.freetile = (std::atoi(v) != 0);
        }
    }
    if (const char * v = std::getenv("OPENBUENOS_DIVE_INTERP_PATH")) {
        if (v[0] != '\0') {
            cfg.dive_interp_path = v;
        }
    }
    return cfg;
}

// Fast lock-free cached env flags for hot-path supports_op / buffer callbacks (eliminates per-op getenv scans).
struct edgetpu_hot_env_cache {
    std::atomic<bool>     initialized{false};
    std::atomic<uint64_t> min_elements{0};
    std::atomic<bool>     packed_q8{false};
    std::atomic<int>      verbose{0};
    std::atomic<bool>     enable_rope{true};
    std::atomic<bool>     enable_glu{true};
    std::atomic<bool>     fuse_swiglu{true};
    std::atomic<int64_t>  glu_max_m{EDGETPU_DEFAULT_GLU_MAX_M};

    void store(const edgetpu_env_config & cfg) {
        min_elements.store(cfg.min_elements, std::memory_order_relaxed);
        packed_q8.store(cfg.packed_q8, std::memory_order_relaxed);
        verbose.store(cfg.verbose, std::memory_order_relaxed);
        enable_rope.store(cfg.enable_rope, std::memory_order_relaxed);
        enable_glu.store(cfg.enable_glu, std::memory_order_relaxed);
        fuse_swiglu.store(cfg.fuse_swiglu, std::memory_order_relaxed);
        glu_max_m.store(cfg.glu_max_m, std::memory_order_relaxed);
        initialized.store(true, std::memory_order_release);
    }

    void ensure_initialized() {
        if (!initialized.load(std::memory_order_acquire)) {
            store(load_env_config());
        }
    }
};

static edgetpu_hot_env_cache g_hot_env;

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

    std::atomic<bool> has_packed_entries{false};
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

    if (ctx->has_packed_entries.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(ctx->cache_mu);
        ctx->packed_q8_cache.erase(tensor->data);
        if (ctx->packed_q8_cache.empty()) {
            ctx->has_packed_entries.store(false, std::memory_order_release);
        }
    }
}

static void ggml_backend_edgetpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   struct ggml_tensor *  tensor,
                                                   const void *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto * ctx = static_cast<ggml_backend_edgetpu_buffer_context *>(buffer->context);
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);

    // Invalidate any stale repacked weight for this tensor address (lock-free skip in default zero-dup mode).
    if (ctx->has_packed_entries.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(ctx->cache_mu);
        ctx->packed_q8_cache.erase(tensor->data);
        if (ctx->packed_q8_cache.empty()) {
            ctx->has_packed_entries.store(false, std::memory_order_release);
        }
    }

    // Strictly zero-dup single-copy residency by default: only repack host weights if explicitly opted in via OPENBUENOS_PACKED_Q8=1.
    g_hot_env.ensure_initialized();
    if (!g_hot_env.packed_q8.load(std::memory_order_relaxed)) {
        return;
    }

    const bool is_full_write = (offset == 0 && size == ggml_nbytes(tensor));
    const bool is_2d_q8_0 =
        tensor->type == GGML_TYPE_Q8_0 &&
        tensor->ne[0] > 0 && (tensor->ne[0] % QK8_0) == 0 &&
        tensor->ne[1] > 0 &&
        tensor->ne[2] == 1 &&
        tensor->ne[3] == 1 &&
        ggml_is_contiguous(tensor);

    if (is_full_write && is_2d_q8_0 && buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        auto packed = repack_q8_0_to_buenos(
            static_cast<const block_q8_0 *>(tensor->data),
            static_cast<uint32_t>(tensor->ne[0]),
            static_cast<uint32_t>(tensor->ne[1]),
            g_hot_env.verbose.load(std::memory_order_relaxed));
        if (packed) {
            std::lock_guard<std::mutex> lock(ctx->cache_mu);
            ctx->packed_q8_cache[tensor->data] = std::move(packed);
            ctx->has_packed_entries.store(true, std::memory_order_release);
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
    if (ctx->has_packed_entries.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(ctx->cache_mu);
        ctx->packed_q8_cache.clear();
        ctx->has_packed_entries.store(false, std::memory_order_release);
    }
}

static void ggml_backend_edgetpu_buffer_reset(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_edgetpu_buffer_context *>(buffer->context);
    if (ctx->has_packed_entries.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(ctx->cache_mu);
        ctx->packed_q8_cache.clear();
        ctx->has_packed_entries.store(false, std::memory_order_release);
    }
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

struct moe_row_ref {
    int32_t id;
    int32_t iid1;
};

struct rope_table_key {
    bool                 valid       = false;
    int64_t              ne2         = 0;
    uint64_t             pos_hash    = 0;
    int64_t              ne0         = 0;
    int                  n_dims      = 0;
    int                  n_ctx_orig  = 0;
    bool                 truncate    = false;
    float                freq_base   = 0.0f;
    float                freq_scale  = 0.0f;
    float                ext_factor  = 0.0f;
    float                attn_factor = 0.0f;
    float                beta_fast   = 0.0f;
    float                beta_slow   = 0.0f;
    std::vector<int32_t> pos_copy;

    static uint64_t hash_pos(const int32_t * pos, int64_t count) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (int64_t i = 0; i < count; ++i) {
            h ^= static_cast<uint64_t>(static_cast<uint32_t>(pos[i]));
            h *= 0x100000001b3ULL;
        }
        return h;
    }

    bool matches(const int32_t * pos, int64_t count, uint64_t ph, int64_t n0, int nd, int nco, bool tr,
                 float fb, float fs, float ef, float af, float bf, float bs) const {
        if (!valid || ne2 != count || pos_hash != ph || ne0 != n0 || n_dims != nd ||
            n_ctx_orig != nco || truncate != tr || freq_base != fb || freq_scale != fs ||
            ext_factor != ef || attn_factor != af || beta_fast != bf || beta_slow != bs) {
            return false;
        }
        return pos_copy.size() == static_cast<size_t>(count) &&
               std::memcmp(pos_copy.data(), pos, static_cast<size_t>(count) * sizeof(int32_t)) == 0;
    }
};

struct ggml_backend_edgetpu_context {
    ob_context_t *     ob_ctx = nullptr;
    edgetpu_env_config env;

    // Reusable per-stream scratch buffers (zero per-eval heap allocations after warmup)
    std::vector<int8_t>      q_a;
    std::vector<int8_t>      q_b;
    std::vector<float>       scale_a;
    std::vector<float>       scale_b;
    std::vector<int32_t>     c_i32;
    std::vector<float>       f32_a;
    std::vector<float>       sibling_scratch;
    std::vector<float>       rope_cache;
    rope_table_key           last_rope_key;

    // Reusable MoE (GGML_OP_MUL_MAT_ID) routing scratch buffers
    std::vector<int32_t>     moe_counts;
    std::vector<int32_t>     moe_offsets;
    std::vector<int32_t>     moe_cursor;
    std::vector<moe_row_ref> moe_rows;
    std::vector<float>       moe_in_f32;
    std::vector<float>       moe_out_f32;

    // Fallback repack cache for weights residing in non-EDGETPU host buffers (only when OPENBUENOS_PACKED_Q8=1)
    std::mutex fallback_cache_mu;
    std::unordered_map<const void *, std::shared_ptr<packed_q8_weight>> fallback_packed_cache;

    // Cumulative execution telemetry for OPENBUENOS_VERBOSE=1 shutdown summary
    uint64_t total_graph_evals         = 0;
    uint64_t total_mul_mat_calls       = 0;
    uint64_t total_mul_mat_id_calls    = 0;
    uint64_t total_slice_calls         = 0;
    uint64_t total_emitter_slices      = 0;
    uint64_t total_cpu_fallback_slices = 0;
    uint64_t total_tiles_dispatched    = 0;
    uint64_t total_tile_cache_hits     = 0;
    uint64_t total_hw_us               = 0;
    uint64_t total_wall_us             = 0;
    uint64_t total_batched_groups      = 0;
    uint64_t total_batched_ops         = 0;
    uint64_t total_batched_3_groups    = 0;
    uint64_t total_batched_2_groups    = 0;
    uint64_t total_bias_adds           = 0;
    uint64_t total_rope_ops            = 0;
    uint64_t total_glu_ops             = 0;
    uint64_t total_swiglu_fused        = 0;

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

                uint64_t freetile_emits     = 0;
                uint64_t freetile_fallbacks = 0;
                ob_context_freetile_stats(ob_ctx, &freetile_emits, &freetile_fallbacks);

                const uint64_t cache_total = cache_hits + cache_misses;
                const double   hit_pct     = (cache_total > 0)
                                                 ? (100.0 * static_cast<double>(cache_hits) / static_cast<double>(cache_total))
                                                 : 0.0;
                GGML_LOG_INFO(
                    "ggml-edgetpu: summary mock=%d freetile=%d freetile_emits=%llu freetile_fallbacks=%llu "
                    "evals=%llu calls=%llu mmid_calls=%llu slices=%llu emitter_slices=%llu cpu_fallback_slices=%llu "
                    "batched_groups=%llu batched_ops=%llu batched_3=%llu batched_2=%llu swiglu_fused=%llu bias_adds=%llu rope_ops=%llu glu_ops=%llu "
                    "tiles=%llu tile_hits=%llu hw_us=%llu wall_us=%llu "
                    "cache_entries=%zu cache_hits=%llu cache_misses=%llu cache_evictions=%llu cache_hit_rate=%.2f%% "
                    "pool_bytes=%zu pool_count=%zu pool_hits=%llu pool_misses=%llu\n",
                    ob_context_is_mock(ob_ctx),
                    ob_context_get_freetile(ob_ctx),
                    static_cast<unsigned long long>(freetile_emits),
                    static_cast<unsigned long long>(freetile_fallbacks),
                    static_cast<unsigned long long>(total_graph_evals),
                    static_cast<unsigned long long>(total_mul_mat_calls),
                    static_cast<unsigned long long>(total_mul_mat_id_calls),
                    static_cast<unsigned long long>(total_slice_calls),
                    static_cast<unsigned long long>(total_emitter_slices),
                    static_cast<unsigned long long>(total_cpu_fallback_slices),
                    static_cast<unsigned long long>(total_batched_groups),
                    static_cast<unsigned long long>(total_batched_ops),
                    static_cast<unsigned long long>(total_batched_3_groups),
                    static_cast<unsigned long long>(total_batched_2_groups),
                    static_cast<unsigned long long>(total_swiglu_fused),
                    static_cast<unsigned long long>(total_bias_adds),
                    static_cast<unsigned long long>(total_rope_ops),
                    static_cast<unsigned long long>(total_glu_ops),
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
            bctx->has_packed_entries.store(true, std::memory_order_release);
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
                        GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_f32_quant_batch (Q4_0 M=%u,K=%u,N=%u) failed: %s (%s)\n",
                                       M, K, N, ob_status_string(st), ob_context_last_error(ctx->ob_ctx));
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
                        GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_int8 (packed Q8_0) failed: %s (%s)\n",
                                       ob_status_string(st), ob_context_last_error(ctx->ob_ctx));
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
                        GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_f32_q8_0(M=%u,K=%u,N=%u) failed: %s (%s)\n",
                                       M, K, N, ob_status_string(st), ob_context_last_error(ctx->ob_ctx));
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
                    GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_int8(M=%u,K=%u,N=%u) failed: %s (%s)\n",
                                   M, K, N, ob_status_string(st), ob_context_last_error(ctx->ob_ctx));
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

static inline bool tensors_overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t b0 = reinterpret_cast<uintptr_t>(b->data);
    const size_t    sa = ggml_nbytes(a);
    const size_t    sb = ggml_nbytes(b);
    if (sa == 0 || sb == 0) {
        return a0 == b0;
    }
    return (a0 < b0 + sb) && (b0 < a0 + sa);
}

static inline const struct ggml_tensor * unwrap_view_src(const struct ggml_tensor * t) {
    for (int depth = 0; depth < 4 && t != nullptr; ++depth) {
        if (t->view_src != nullptr) {
            t = t->view_src;
            continue;
        }
        if ((t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
             t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) &&
            t->src[0] != nullptr) {
            t = t->src[0];
            continue;
        }
        break;
    }
    return t;
}

static inline bool view_preserves_2d_identity(const struct ggml_tensor * t,
                                              const struct ggml_tensor * producer) {
    if (!t || !producer) {
        return false;
    }
    const struct ggml_tensor * cur = t;
    for (int depth = 0; depth < 4 && cur != nullptr; ++depth) {
        if (cur == producer) {
            return ggml_is_contiguous(t) &&
                   t->type == producer->type &&
                   t->data == producer->data &&
                   t->ne[0] == producer->ne[0] &&
                   ggml_nrows(t) == ggml_nrows(producer);
        }
        if (cur->op == GGML_OP_PERMUTE || cur->op == GGML_OP_TRANSPOSE) {
            return false;
        }
        if (!ggml_is_contiguous(cur) || cur->view_offs != 0 ||
            cur->type != producer->type || cur->ne[0] != producer->ne[0] ||
            ggml_nrows(cur) != ggml_nrows(producer)) {
            return false;
        }
        if (cur->view_src != nullptr) {
            cur = cur->view_src;
            continue;
        }
        if ((cur->op == GGML_OP_RESHAPE || cur->op == GGML_OP_VIEW) && cur->src[0] != nullptr) {
            cur = cur->src[0];
            continue;
        }
        break;
    }
    return false;
}

static inline void add_bias_1d_row(float * y, const float * x, const float * bias, int64_t n_cols) {
    int64_t c = 0;
#if defined(__ARM_NEON)
    for (; c + 16 <= n_cols; c += 16) {
        float32x4_t x0 = vld1q_f32(x + c +  0);
        float32x4_t x1 = vld1q_f32(x + c +  4);
        float32x4_t x2 = vld1q_f32(x + c +  8);
        float32x4_t x3 = vld1q_f32(x + c + 12);
        float32x4_t b0 = vld1q_f32(bias + c +  0);
        float32x4_t b1 = vld1q_f32(bias + c +  4);
        float32x4_t b2 = vld1q_f32(bias + c +  8);
        float32x4_t b3 = vld1q_f32(bias + c + 12);
        vst1q_f32(y + c +  0, vaddq_f32(x0, b0));
        vst1q_f32(y + c +  4, vaddq_f32(x1, b1));
        vst1q_f32(y + c +  8, vaddq_f32(x2, b2));
        vst1q_f32(y + c + 12, vaddq_f32(x3, b3));
    }
    for (; c + 4 <= n_cols; c += 4) {
        vst1q_f32(y + c, vaddq_f32(vld1q_f32(x + c), vld1q_f32(bias + c)));
    }
#elif defined(__AVX2__)
    for (; c + 16 <= n_cols; c += 16) {
        __m256 x0 = _mm256_loadu_ps(x + c);
        __m256 x1 = _mm256_loadu_ps(x + c + 8);
        __m256 b0 = _mm256_loadu_ps(bias + c);
        __m256 b1 = _mm256_loadu_ps(bias + c + 8);
        _mm256_storeu_ps(y + c,     _mm256_add_ps(x0, b0));
        _mm256_storeu_ps(y + c + 8, _mm256_add_ps(x1, b1));
    }
    for (; c + 8 <= n_cols; c += 8) {
        _mm256_storeu_ps(y + c, _mm256_add_ps(_mm256_loadu_ps(x + c), _mm256_loadu_ps(bias + c)));
    }
#endif
    for (; c < n_cols; ++c) {
        y[c] = x[c] + bias[c];
    }
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
        add_bias_1d_row(y + r * n_cols, x + r * n_cols, bias, n_cols);
    }

    ctx->total_bias_adds += 1;
    return GGML_STATUS_SUCCESS;
}

static inline float edgetpu_rope_yarn_ramp(float low, float high, int64_t i0) {
    const float denom = (high - low) > 0.001f ? (high - low) : 0.001f;
    const float y = (static_cast<float>(i0 / 2) - low) / denom;
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

static void edgetpu_rope_cache_init(float         theta_base,
                                    float         freq_scale,
                                    const float * freq_factors,
                                    const float   corr_dims[2],
                                    int64_t       n_dims,
                                    float         ext_factor,
                                    float         mscale,
                                    float *       cache,
                                    float         theta_scale) {
    float theta = theta_base;
    const float yarn_mscale = (ext_factor != 0.0f) ? (mscale * (1.0f + 0.1f * std::log(1.0f / freq_scale))) : mscale;
    for (int64_t i0 = 0; i0 + 1 < n_dims; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        const float theta_extrap = theta / ff;
        const float theta_interp = freq_scale * theta_extrap;
        float t = theta_interp;
        if (ext_factor != 0.0f) {
            const float ramp_mix = edgetpu_rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
            t = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        }
        cache[i0 + 0] = std::cos(t) * yarn_mscale;
        cache[i0 + 1] = std::sin(t) * yarn_mscale;
        theta *= theta_scale;
    }
}

static inline void edgetpu_rotate_pairs_neox_f32(int64_t n_dims, const float * cache, const float * src, float * dst) {
    const int64_t half = n_dims / 2;
    int64_t ic = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    for (; ic + 4 <= half; ic += 4) {
        const float32x4x2_t cs = vld2q_f32(cache + 2 * ic);
        const float32x4_t   x0 = vld1q_f32(src + ic);
        const float32x4_t   x1 = vld1q_f32(src + ic + half);
        const float32x4_t   y0 = vfmsq_f32(vmulq_f32(x0, cs.val[0]), x1, cs.val[1]);
        const float32x4_t   y1 = vfmaq_f32(vmulq_f32(x0, cs.val[1]), x1, cs.val[0]);
        vst1q_f32(dst + ic,        y0);
        vst1q_f32(dst + ic + half, y1);
    }
#elif defined(__AVX2__)
    for (; ic + 4 <= half; ic += 4) {
        const __m128 x0 = _mm_loadu_ps(src + ic);
        const __m128 x1 = _mm_loadu_ps(src + ic + half);
        const __m128 c01 = _mm_loadu_ps(cache + 2 * ic);
        const __m128 c23 = _mm_loadu_ps(cache + 2 * ic + 4);
        const __m128 c   = _mm_shuffle_ps(c01, c23, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 s   = _mm_shuffle_ps(c01, c23, _MM_SHUFFLE(3, 1, 3, 1));
        const __m128 y0  = _mm_sub_ps(_mm_mul_ps(x0, c), _mm_mul_ps(x1, s));
        const __m128 y1  = _mm_add_ps(_mm_mul_ps(x0, s), _mm_mul_ps(x1, c));
        _mm_storeu_ps(dst + ic,        y0);
        _mm_storeu_ps(dst + ic + half, y1);
    }
#endif
    for (; ic < half; ++ic) {
        const float c  = cache[2 * ic + 0];
        const float s  = cache[2 * ic + 1];
        const float x0 = src[ic];
        const float x1 = src[ic + half];
        dst[ic]        = x0 * c - x1 * s;
        dst[ic + half] = x0 * s + x1 * c;
    }
}

static inline void edgetpu_rotate_pairs_normal_f32(int64_t n_dims, const float * cache, const float * src, float * dst) {
    int64_t i0 = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    for (; i0 + 8 <= n_dims; i0 += 8) {
        const float32x4x2_t cs = vld2q_f32(cache + i0);
        const float32x4x2_t x  = vld2q_f32(src + i0);
        float32x4x2_t       y;
        y.val[0] = vfmsq_f32(vmulq_f32(x.val[0], cs.val[0]), x.val[1], cs.val[1]);
        y.val[1] = vfmaq_f32(vmulq_f32(x.val[0], cs.val[1]), x.val[1], cs.val[0]);
        vst2q_f32(dst + i0, y);
    }
#endif
    for (; i0 < n_dims; i0 += 2) {
        const float c  = cache[i0 + 0];
        const float s  = cache[i0 + 1];
        const float x0 = src[i0 + 0];
        const float x1 = src[i0 + 1];
        dst[i0 + 0] = x0 * c - x1 * s;
        dst[i0 + 1] = x0 * s + x1 * c;
    }
}

static ggml_status ggml_edgetpu_compute_forward_rope(ggml_backend_edgetpu_context * ctx,
                                                     struct ggml_tensor *           dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32);

    const int  n_dims     = static_cast<const int32_t *>(dst->op_params)[1];
    const int  mode       = static_cast<const int32_t *>(dst->op_params)[2];
    const int  n_ctx_orig = static_cast<const int32_t *>(dst->op_params)[4];
    const bool truncate   = ggml_rope_get_truncate(dst);

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    std::memcpy(&freq_base,   static_cast<const int32_t *>(dst->op_params) +  5, sizeof(float));
    std::memcpy(&freq_scale,  static_cast<const int32_t *>(dst->op_params) +  6, sizeof(float));
    std::memcpy(&ext_factor,  static_cast<const int32_t *>(dst->op_params) +  7, sizeof(float));
    std::memcpy(&attn_factor, static_cast<const int32_t *>(dst->op_params) +  8, sizeof(float));
    std::memcpy(&beta_fast,   static_cast<const int32_t *>(dst->op_params) +  9, sizeof(float));
    std::memcpy(&beta_slow,   static_cast<const int32_t *>(dst->op_params) + 10, sizeof(float));
    const int n_offs = static_cast<const int32_t *>(dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t n_offs_64 = static_cast<int64_t>(n_offs);
    const int64_t n_dims_64 = static_cast<int64_t>(n_dims);
    if (n_dims_64 <= 0 || (n_dims_64 % 2) != 0 || n_offs_64 < 0 || (n_offs_64 % 2) != 0 ||
        n_offs_64 + n_dims_64 > ne0 || n_offs_64 + n_dims_64 > ne00) {
        return GGML_STATUS_FAILED;
    }

    const float * freq_factors = src2 ? reinterpret_cast<const float *>(src2->data) : nullptr;
    const int32_t * pos        = reinterpret_cast<const int32_t *>(src1->data);

    const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_dims));
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, truncate, corr_dims);

    // Cache full cos/sin table across Q/K and all layers for both decode (ne2==1) and prefill (ne2<=2048)!
    // Eliminates 55/56ths of trigonometric cos/sin evaluations on Qwen2.5 pp512 (28L x Q,K).
    const size_t stride_dims = static_cast<size_t>(n_dims_64);
    const bool can_cache_table = (freq_factors == nullptr && ne2 >= 1 && ne2 <= 2048 && n_dims_64 <= 512);
    if (can_cache_table) {
        const size_t table_elems = static_cast<size_t>(ne2) * stride_dims;
        if (ctx->rope_cache.size() < table_elems) {
            ctx->rope_cache.resize(table_elems);
        }
        const uint64_t ph = rope_table_key::hash_pos(pos, ne2);
        if (!ctx->last_rope_key.matches(pos, ne2, ph, ne0, n_dims, n_ctx_orig, truncate,
                                        freq_base, freq_scale, ext_factor, attn_factor,
                                        beta_fast, beta_slow)) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                edgetpu_rope_cache_init(
                    static_cast<float>(pos[i2]), freq_scale, nullptr, corr_dims,
                    n_dims, ext_factor, attn_factor,
                    ctx->rope_cache.data() + static_cast<size_t>(i2) * stride_dims,
                    theta_scale);
            }
            ctx->last_rope_key.valid       = true;
            ctx->last_rope_key.ne2         = ne2;
            ctx->last_rope_key.pos_hash    = ph;
            ctx->last_rope_key.ne0         = ne0;
            ctx->last_rope_key.n_dims      = n_dims;
            ctx->last_rope_key.n_ctx_orig  = n_ctx_orig;
            ctx->last_rope_key.truncate    = truncate;
            ctx->last_rope_key.freq_base   = freq_base;
            ctx->last_rope_key.freq_scale  = freq_scale;
            ctx->last_rope_key.ext_factor  = ext_factor;
            ctx->last_rope_key.attn_factor = attn_factor;
            ctx->last_rope_key.beta_fast   = beta_fast;
            ctx->last_rope_key.beta_slow   = beta_slow;
            ctx->last_rope_key.pos_copy.assign(pos, pos + ne2);
        }

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const float * cache = ctx->rope_cache.data() + static_cast<size_t>(i2) * stride_dims;
                const char * src_base = static_cast<const char *>(src0->data) + i3 * nb03 + i2 * nb02;
                char *       dst_base = static_cast<char *>(dst->data)        + i3 * nb3  + i2 * nb2;

                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const float * srow = reinterpret_cast<const float *>(src_base + i1 * nb01);
                    float *       drow = reinterpret_cast<float *>(dst_base + i1 * nb1);

                    if (mode == GGML_ROPE_TYPE_NEOX) {
                        edgetpu_rotate_pairs_neox_f32(n_dims, cache, srow + n_offs, drow + n_offs);
                    } else {
                        edgetpu_rotate_pairs_normal_f32(n_dims, cache, srow + n_offs, drow + n_offs);
                    }

                    if (drow != srow) {
                        if (n_offs_64 > 0) {
                            std::memcpy(drow, srow, static_cast<size_t>(n_offs_64) * sizeof(float));
                        }
                        const int64_t tail_start = n_offs_64 + n_dims_64;
                        if (tail_start < ne0) {
                            std::memcpy(drow + tail_start, srow + tail_start,
                                        static_cast<size_t>(ne0 - tail_start) * sizeof(float));
                        }
                    }
                }
            }
        }
    } else {
        if (ctx->rope_cache.size() < stride_dims) {
            ctx->rope_cache.resize(stride_dims);
        }
        float * cache = ctx->rope_cache.data();
        ctx->last_rope_key.valid = false;
        int64_t last_i2 = -1;

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                if (last_i2 != i2) {
                    edgetpu_rope_cache_init(
                        static_cast<float>(pos[i2]), freq_scale, freq_factors, corr_dims,
                        n_dims, ext_factor, attn_factor, cache, theta_scale);
                    last_i2 = i2;
                }

                const char * src_base = static_cast<const char *>(src0->data) + i3 * nb03 + i2 * nb02;
                char *       dst_base = static_cast<char *>(dst->data)        + i3 * nb3  + i2 * nb2;

                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const float * srow = reinterpret_cast<const float *>(src_base + i1 * nb01);
                    float *       drow = reinterpret_cast<float *>(dst_base + i1 * nb1);

                    if (mode == GGML_ROPE_TYPE_NEOX) {
                        edgetpu_rotate_pairs_neox_f32(n_dims, cache, srow + n_offs, drow + n_offs);
                    } else {
                        edgetpu_rotate_pairs_normal_f32(n_dims, cache, srow + n_offs, drow + n_offs);
                    }

                    if (drow != srow) {
                        if (n_offs_64 > 0) {
                            std::memcpy(drow, srow, static_cast<size_t>(n_offs_64) * sizeof(float));
                        }
                        const int64_t tail_start = n_offs_64 + n_dims_64;
                        if (tail_start < ne0) {
                            std::memcpy(drow + tail_start, srow + tail_start,
                                        static_cast<size_t>(ne0 - tail_start) * sizeof(float));
                        }
                    }
                }
            }
        }
    }

    ctx->total_rope_ops += 1;
    return GGML_STATUS_SUCCESS;
}

#if defined(__ARM_NEON) && defined(__aarch64__)
// High-accuracy (< 2 ULP) vectorized expf for ARM64 NEON (Cody-Waite + degree-5 minimax).
static inline float32x4_t edgetpu_vexpq_f32(float32x4_t x) {
    const float32x4_t v_min   = vdupq_n_f32(-87.3f);
    const float32x4_t v_max   = vdupq_n_f32( 88.0f);
    const float32x4_t log2e   = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t ln2_hi  = vdupq_n_f32(0.693359375f);
    const float32x4_t ln2_lo  = vdupq_n_f32(-2.12194440e-4f);
    const float32x4_t c5      = vdupq_n_f32(1.9875691500e-4f);
    const float32x4_t c4      = vdupq_n_f32(1.3981999507e-3f);
    const float32x4_t c3      = vdupq_n_f32(8.3334519073e-3f);
    const float32x4_t c2      = vdupq_n_f32(4.1665795894e-2f);
    const float32x4_t c1      = vdupq_n_f32(1.6666665459e-1f);
    const float32x4_t c0      = vdupq_n_f32(5.0000001201e-1f);
    const float32x4_t v_one   = vdupq_n_f32(1.0f);

    x = vmaxq_f32(vminq_f32(x, v_max), v_min);
    const float32x4_t fx = vrndnq_f32(vmulq_f32(x, log2e));
    float32x4_t z = vfmsq_f32(x, fx, ln2_hi);
    z = vfmsq_f32(z, fx, ln2_lo);
    const float32x4_t z2 = vmulq_f32(z, z);

    float32x4_t p = vfmaq_f32(c4, c5, z);
    p = vfmaq_f32(c3, p, z);
    p = vfmaq_f32(c2, p, z);
    p = vfmaq_f32(c1, p, z);
    p = vfmaq_f32(c0, p, z);
    p = vfmaq_f32(vaddq_f32(z, v_one), p, z2);

    const int32x4_t n     = vcvtq_s32_f32(fx);
    const int32x4_t pow2i = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(pow2i));
}
#endif

static inline void swiglu_f32_row(float * __restrict dst,
                                  const float * __restrict gate,
                                  const float * __restrict up,
                                  int64_t n) {
    int64_t i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    for (; i + 8 <= n; i += 8) {
        const float32x4_t g0 = vld1q_f32(gate + i);
        const float32x4_t g1 = vld1q_f32(gate + i + 4);
        const float32x4_t u0 = vld1q_f32(up + i);
        const float32x4_t u1 = vld1q_f32(up + i + 4);
        const float32x4_t e0 = edgetpu_vexpq_f32(vnegq_f32(g0));
        const float32x4_t e1 = edgetpu_vexpq_f32(vnegq_f32(g1));
        const float32x4_t s0 = vdivq_f32(g0, vaddq_f32(v_one, e0));
        const float32x4_t s1 = vdivq_f32(g1, vaddq_f32(v_one, e1));
        vst1q_f32(dst + i,     vmulq_f32(s0, u0));
        vst1q_f32(dst + i + 4, vmulq_f32(s1, u1));
    }
    for (; i + 4 <= n; i += 4) {
        const float32x4_t g0 = vld1q_f32(gate + i);
        const float32x4_t u0 = vld1q_f32(up + i);
        const float32x4_t e0 = edgetpu_vexpq_f32(vnegq_f32(g0));
        const float32x4_t s0 = vdivq_f32(g0, vaddq_f32(v_one, e0));
        vst1q_f32(dst + i, vmulq_f32(s0, u0));
    }
#endif
    for (; i < n; ++i) {
        const float g = gate[i];
        const float s = g / (1.0f + std::exp(-g));
        dst[i] = s * up[i];
    }
}

static ggml_status ggml_edgetpu_compute_forward_glu(ggml_backend_edgetpu_context * ctx,
                                                    struct ggml_tensor *           dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU);
    GGML_ASSERT(src0 && src1 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int64_t nc = dst->ne[0];
    const int64_t nr = ggml_nrows(dst);
    const char * s0_base = static_cast<const char *>(src0->data);
    const char * s1_base = static_cast<const char *>(src1->data);
    char *       d_base  = static_cast<char *>(dst->data);

    for (int64_t r = 0; r < nr; ++r) {
        const float * g = reinterpret_cast<const float *>(s0_base + r * src0->nb[1]);
        const float * u = reinterpret_cast<const float *>(s1_base + r * src1->nb[1]);
        float *       y = reinterpret_cast<float *>(d_base + r * dst->nb[1]);
        swiglu_f32_row(y, g, u, nc);
    }

    ctx->total_glu_ops += 1;
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_edgetpu_compute_forward_mul_mat_id(ggml_backend_edgetpu_context * ctx,
                                                           struct ggml_tensor *           dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // [K, N, n_experts, 1]
    const struct ggml_tensor * src1 = dst->src[1]; // [K, n_used or 1, n_tokens, 1]
    const struct ggml_tensor * ids  = dst->src[2]; // [n_used, n_tokens] I32

    GGML_TENSOR_BINARY_OP_LOCALS

    const uint32_t K        = static_cast<uint32_t>(ne00);
    const uint32_t N        = static_cast<uint32_t>(ne01);
    const int64_t  n_expert = ne02;
    const int64_t  n_used   = ids->ne[0];
    const int64_t  n_tokens = ids->ne[1];
    const uint8_t  is_q4_0  = (src0->type == GGML_TYPE_Q4_0) ? 1u : 0u;

    const auto * q8_traits = (src1->type == GGML_TYPE_Q8_0) ? ggml_get_type_traits(GGML_TYPE_Q8_0) : nullptr;

    uint64_t total_hw_us   = 0;
    uint64_t total_wall_us = 0;
    uint32_t total_tiles   = 0;
    uint32_t total_hits    = 0;
    uint64_t total_slices  = 0;
    uint16_t merged_flags  = 0;

    auto record_stats = [&](const ob_invoke_stats_t & st, uint32_t slices) {
        total_hw_us   += st.hw_execution_us;
        total_wall_us += st.wall_time_us;
        total_tiles   += st.tiles_dispatched;
        total_hits    += st.cache_hits;
        total_slices  += slices;
        merged_flags  |= st.flags;
        ctx->total_slice_calls += slices;
        if (st.flags & OB_STAT_FLAG_EMITTER_USED) {
            ctx->total_emitter_slices += slices;
        }
        if (st.flags & OB_STAT_FLAG_CPU_FALLBACK) {
            ctx->total_cpu_fallback_slices += slices;
            if (!ctx->env.mock_mode && ctx->total_cpu_fallback_slices <= 8) {
                GGML_LOG_WARN("ggml-edgetpu: CPU fallback in MUL_MAT_ID (%s x %s) K=%u N=%u (flags=0x%x)\n",
                              ggml_type_name(src0->type), ggml_type_name(src1->type),
                              K, N, static_cast<unsigned>(st.flags));
            }
        }
    };

    // Fast path: single token with shared broadcast activation (ne11 == 1) -> batch all active experts
    // directly into ob_gemm_matmul_f32_quant_batch across the 16-credit async ring (zero gather/scatter).
    if (n_tokens == 1 && ne11 == 1) {
        const float * a_row = nullptr;
        if (src1->type == GGML_TYPE_F32) {
            a_row = reinterpret_cast<const float *>(src1->data);
        } else if (q8_traits) {
            ctx->f32_a.resize(K);
            q8_traits->to_float(src1->data, ctx->f32_a.data(), K);
            a_row = ctx->f32_a.data();
        } else {
            return GGML_STATUS_FAILED;
        }

        constexpr uint32_t MAX_BATCH_JOBS = 16u;
        ob_gemm_f32_quant_job_t jobs[MAX_BATCH_JOBS];
        uint32_t n_jobs = 0;

        auto flush_jobs = [&]() -> ggml_status {
            if (n_jobs == 0) {
                return GGML_STATUS_SUCCESS;
            }
            ob_invoke_stats_t bstats {};
            const ob_status_t st = ob_gemm_matmul_f32_quant_batch(
                ctx->ob_ctx, a_row, 1u, K, jobs, n_jobs, &bstats);
            if (st != OB_OK) {
                GGML_LOG_ERROR("ggml-edgetpu: MUL_MAT_ID batch(n=%u,K=%u,N=%u) failed: %s\n",
                               n_jobs, K, N, ob_status_string(st));
                return GGML_STATUS_FAILED;
            }
            record_stats(bstats, n_jobs);
            if (n_jobs > 1u) {
                ctx->total_batched_groups += 1;
                ctx->total_batched_ops    += n_jobs;
            }
            n_jobs = 0;
            return GGML_STATUS_SUCCESS;
        };

        for (int64_t id = 0; id < n_used; ++id) {
            const int32_t expert = *reinterpret_cast<const int32_t *>(
                static_cast<const char *>(ids->data) + id * ids->nb[0]);
            float * out_row = reinterpret_cast<float *>(static_cast<char *>(dst->data) + id * nb1);

            if (expert == -1) {
                std::memset(out_row, 0, static_cast<size_t>(N) * sizeof(float));
                continue;
            }
            if (expert < 0 || expert >= n_expert) {
                GGML_LOG_ERROR("ggml-edgetpu: MUL_MAT_ID expert id %d out of range [-1, %lld)\n",
                               expert, static_cast<long long>(n_expert));
                return GGML_STATUS_FAILED;
            }

            jobs[n_jobs]            = {};
            jobs[n_jobs].b_nk       = static_cast<const char *>(src0->data) + static_cast<size_t>(expert) * nb02;
            jobs[n_jobs].c_mn       = out_row;
            jobs[n_jobs].n_dim      = N;
            jobs[n_jobs].activation = OB_ACT_NONE;
            jobs[n_jobs].is_q4_0    = is_q4_0;
            n_jobs++;

            if (n_jobs == MAX_BATCH_JOBS) {
                const ggml_status fst = flush_jobs();
                if (fst != GGML_STATUS_SUCCESS) {
                    return fst;
                }
            }
        }
        const ggml_status fst = flush_jobs();
        if (fst != GGML_STATUS_SUCCESS) {
            return fst;
        }
    } else {
        // General / multi-token path: group routed (id, iid1) pairs by expert_id so every active expert
        // executes as one contiguous batched M=count GEMM (amortizing weight bandwidth & VII launches).
        if (ctx->moe_counts.size() < static_cast<size_t>(n_expert)) {
            ctx->moe_counts.resize(static_cast<size_t>(n_expert));
            ctx->moe_offsets.resize(static_cast<size_t>(n_expert));
            ctx->moe_cursor.resize(static_cast<size_t>(n_expert));
        }
        std::memset(ctx->moe_counts.data(), 0, static_cast<size_t>(n_expert) * sizeof(int32_t));

        int64_t total_valid = 0;
        for (int64_t iid1 = 0; iid1 < n_tokens; ++iid1) {
            const char * ids_row = static_cast<const char *>(ids->data) + iid1 * ids->nb[1];
            for (int64_t id = 0; id < n_used; ++id) {
                const int32_t expert = *reinterpret_cast<const int32_t *>(ids_row + id * ids->nb[0]);
                if (expert == -1) {
                    std::memset(static_cast<char *>(dst->data) + id * nb1 + iid1 * nb2,
                                0, static_cast<size_t>(N) * sizeof(float));
                    continue;
                }
                if (expert < 0 || expert >= n_expert) {
                    GGML_LOG_ERROR("ggml-edgetpu: MUL_MAT_ID expert id %d out of range [-1, %lld)\n",
                                   expert, static_cast<long long>(n_expert));
                    return GGML_STATUS_FAILED;
                }
                ctx->moe_counts[expert] += 1;
                total_valid += 1;
            }
        }

        if (total_valid > 0) {
            if (ctx->moe_rows.size() < static_cast<size_t>(total_valid)) {
                ctx->moe_rows.resize(static_cast<size_t>(total_valid));
            }
            int32_t running = 0;
            for (int64_t e = 0; e < n_expert; ++e) {
                ctx->moe_offsets[e] = running;
                ctx->moe_cursor[e]  = running;
                running += ctx->moe_counts[e];
            }
            for (int64_t iid1 = 0; iid1 < n_tokens; ++iid1) {
                const char * ids_row = static_cast<const char *>(ids->data) + iid1 * ids->nb[1];
                for (int64_t id = 0; id < n_used; ++id) {
                    const int32_t expert = *reinterpret_cast<const int32_t *>(ids_row + id * ids->nb[0]);
                    if (expert >= 0) {
                        const int32_t pos = ctx->moe_cursor[expert]++;
                        ctx->moe_rows[pos] = { static_cast<int32_t>(id), static_cast<int32_t>(iid1) };
                    }
                }
            }

            for (int64_t e = 0; e < n_expert; ++e) {
                const int32_t count = ctx->moe_counts[e];
                if (count == 0) {
                    continue;
                }
                const int32_t base = ctx->moe_offsets[e];
                const void *  w_ptr = static_cast<const char *>(src0->data) + static_cast<size_t>(e) * nb02;

                if (count == 1 && src1->type == GGML_TYPE_F32) {
                    // Single routed row for this expert: zero-copy direct dispatch.
                    const moe_row_ref ref = ctx->moe_rows[base];
                    const float * a_ptr = reinterpret_cast<const float *>(
                        static_cast<const char *>(src1->data) + (ref.id % ne11) * nb11 + ref.iid1 * nb12);
                    float * c_ptr = reinterpret_cast<float *>(
                        static_cast<char *>(dst->data) + ref.id * nb1 + ref.iid1 * nb2);

                    ob_gemm_f32_quant_job_t job {};
                    job.b_nk       = w_ptr;
                    job.c_mn       = c_ptr;
                    job.n_dim      = N;
                    job.activation = OB_ACT_NONE;
                    job.is_q4_0    = is_q4_0;

                    ob_invoke_stats_t estats {};
                    const ob_status_t st = ob_gemm_matmul_f32_quant_batch(ctx->ob_ctx, a_ptr, 1u, K, &job, 1u, &estats);
                    if (st != OB_OK) {
                        GGML_LOG_ERROR("ggml-edgetpu: MUL_MAT_ID expert=%lld M=1 K=%u N=%u failed: %s\n",
                                       static_cast<long long>(e), K, N, ob_status_string(st));
                        return GGML_STATUS_FAILED;
                    }
                    record_stats(estats, 1u);
                } else {
                    const size_t need_in  = static_cast<size_t>(count) * K;
                    const size_t need_out = static_cast<size_t>(count) * N;
                    if (ctx->moe_in_f32.size() < need_in) {
                        ctx->moe_in_f32.resize(need_in);
                    }
                    if (ctx->moe_out_f32.size() < need_out) {
                        ctx->moe_out_f32.resize(need_out);
                    }

                    for (int32_t r = 0; r < count; ++r) {
                        const moe_row_ref ref = ctx->moe_rows[base + r];
                        const char * src_row = static_cast<const char *>(src1->data) + (ref.id % ne11) * nb11 + ref.iid1 * nb12;
                        float *      dst_in  = ctx->moe_in_f32.data() + static_cast<size_t>(r) * K;
                        if (src1->type == GGML_TYPE_F32) {
                            std::memcpy(dst_in, src_row, static_cast<size_t>(K) * sizeof(float));
                        } else if (q8_traits) {
                            q8_traits->to_float(src_row, dst_in, K);
                        } else {
                            return GGML_STATUS_FAILED;
                        }
                    }

                    ob_gemm_f32_quant_job_t job {};
                    job.b_nk       = w_ptr;
                    job.c_mn       = ctx->moe_out_f32.data();
                    job.n_dim      = N;
                    job.activation = OB_ACT_NONE;
                    job.is_q4_0    = is_q4_0;

                    ob_invoke_stats_t estats {};
                    const ob_status_t st = ob_gemm_matmul_f32_quant_batch(
                        ctx->ob_ctx, ctx->moe_in_f32.data(), static_cast<uint32_t>(count), K, &job, 1u, &estats);
                    if (st != OB_OK) {
                        GGML_LOG_ERROR("ggml-edgetpu: MUL_MAT_ID expert=%lld M=%d K=%u N=%u failed: %s\n",
                                       static_cast<long long>(e), count, K, N, ob_status_string(st));
                        return GGML_STATUS_FAILED;
                    }
                    record_stats(estats, 1u);

                    for (int32_t r = 0; r < count; ++r) {
                        const moe_row_ref ref = ctx->moe_rows[base + r];
                        float * dst_row = reinterpret_cast<float *>(
                            static_cast<char *>(dst->data) + ref.id * nb1 + ref.iid1 * nb2);
                        std::memcpy(dst_row, ctx->moe_out_f32.data() + static_cast<size_t>(r) * N,
                                    static_cast<size_t>(N) * sizeof(float));
                    }
                }
            }
        }
    }

    ctx->total_mul_mat_id_calls += 1;
    ctx->total_tiles_dispatched += total_tiles;
    ctx->total_tile_cache_hits  += total_hits;
    ctx->total_hw_us            += total_hw_us;
    ctx->total_wall_us          += total_wall_us;

    if (ctx->env.verbose >= 2) {
        GGML_LOG_INFO("ggml-edgetpu: MUL_MAT_ID (%s x %s) K=%u N=%u experts=%lld used=%lld tokens=%lld slices=%llu tiles=%u hw_us=%llu wall_us=%llu flags=0x%x\n",
                      ggml_type_name(src0->type), ggml_type_name(src1->type),
                      K, N,
                      static_cast<long long>(n_expert),
                      static_cast<long long>(n_used),
                      static_cast<long long>(n_tokens),
                      static_cast<unsigned long long>(total_slices),
                      total_tiles,
                      static_cast<unsigned long long>(total_hw_us),
                      static_cast<unsigned long long>(total_wall_us),
                      static_cast<unsigned>(merged_flags));
    }

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
    if (tensors_overlap(node, src1)) {
        return false;
    }
    return true;
}

static ggml_status ggml_edgetpu_compute_forward_silu(ggml_backend_edgetpu_context * ctx,
                                                     struct ggml_tensor *           dst) {
    GGML_UNUSED(ctx);
    const struct ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0 && src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int64_t n = ggml_nelements(dst);
    const float * x = reinterpret_cast<const float *>(src0->data);
    float *       y = reinterpret_cast<float *>(dst->data);

    int64_t i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t g = vld1q_f32(x + i);
        const float32x4_t e = edgetpu_vexpq_f32(vnegq_f32(g));
        vst1q_f32(y + i, vdivq_f32(g, vaddq_f32(v_one, e)));
    }
#endif
    for (; i < n; ++i) {
        const float g = x[i];
        y[i] = g / (1.0f + std::exp(-g));
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_edgetpu_compute_forward_mul(ggml_backend_edgetpu_context * ctx,
                                                    struct ggml_tensor *           dst) {
    GGML_UNUSED(ctx);
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(src0 && src1 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int64_t n = ggml_nelements(dst);
    const float * a = reinterpret_cast<const float *>(src0->data);
    const float * b = reinterpret_cast<const float *>(src1->data);
    float *       y = reinterpret_cast<float *>(dst->data);

    int64_t i = 0;
#if defined(__ARM_NEON)
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i,     vmulq_f32(vld1q_f32(a + i),     vld1q_f32(b + i)));
        vst1q_f32(y + i + 4, vmulq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
    return GGML_STATUS_SUCCESS;
}

static bool is_fusible_swiglu_mul_mat(const struct ggml_tensor * node) {
    if (!node || node->op != GGML_OP_MUL_MAT || node->type != GGML_TYPE_F32) {
        return false;
    }
    const struct ggml_tensor * w = node->src[0];
    const struct ggml_tensor * x = node->src[1];
    if (!w || !x) {
        return false;
    }
    if ((w->type != GGML_TYPE_Q8_0 && w->type != GGML_TYPE_Q4_0) || x->type != GGML_TYPE_F32) {
        return false;
    }
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(node)) {
        return false;
    }
    if ((w->ne[0] % QK8_0) != 0) {
        return false;
    }
    return true;
}

static bool is_view_or_noop(const struct ggml_tensor * t) {
    return (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
           t->op == GGML_OP_NONE ||
           t->op == GGML_OP_RESHAPE ||
           t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE;
}

// Attempts to match a whole-block SwiGLU FFN starting at cgraph->nodes[start_idx]:
//   Pattern A (GGML_OP_GLU): MUL_MAT(gate, x) + MUL_MAT(up, x) -> GLU(SWIGLU) -> MUL_MAT(down)
//   Pattern B (unfused):     MUL_MAT(gate, x) -> SILU -> MUL(up) -> MUL_MAT(down)
// On match, executes via ob_gemm_swiglu_f32_quant (in-SRAM chained on Tensor G5) and stores
// the final consumed node index in *out_end_idx, returning 1 (0 = no match, -1 = error).
static int try_compute_fused_swiglu(ggml_backend_edgetpu_context * ctx,
                                    struct ggml_cgraph *           cgraph,
                                    int                            start_idx,
                                    int *                          out_end_idx) {
    if (!ctx->env.fuse_swiglu || ctx->env.packed_q8) {
        return 0;
    }
    struct ggml_tensor * n0 = cgraph->nodes[start_idx];
    if (!is_fusible_swiglu_mul_mat(n0)) {
        return 0;
    }

    // Collect the next up to 5 compute ops starting from start_idx.
    int                  comp_idx[5];
    struct ggml_tensor * comp[5];
    int n_comp = 0;
    for (int k = start_idx; k < cgraph->n_nodes && n_comp < 5; ++k) {
        struct ggml_tensor * t = cgraph->nodes[k];
        if (k != start_idx && is_view_or_noop(t)) {
            continue;
        }
        comp_idx[n_comp] = k;
        comp[n_comp]     = t;
        n_comp++;
    }
    if (n_comp < 4) {
        return 0;
    }

    struct ggml_tensor * gate_mm   = nullptr;
    struct ggml_tensor * up_mm     = nullptr;
    struct ggml_tensor * down_mm   = nullptr;
    struct ggml_tensor * inter[4]  = { nullptr, nullptr, nullptr, nullptr };
    int                  n_inter   = 0;
    int                  end_idx   = -1;

    // Pattern A: [mm0, mm1, GLU(SWIGLU), down_mm]
    if (comp[1]->op == GGML_OP_MUL_MAT &&
        comp[2]->op == GGML_OP_GLU && ggml_get_glu_op(comp[2]) == GGML_GLU_OP_SWIGLU &&
        comp[3]->op == GGML_OP_MUL_MAT) {
        struct ggml_tensor * mm0 = comp[0];
        struct ggml_tensor * mm1 = comp[1];
        struct ggml_tensor * glu = comp[2];
        struct ggml_tensor * dmm = comp[3];
        const struct ggml_tensor * g0 = unwrap_view_src(glu->src[0]);
        const struct ggml_tensor * g1 = unwrap_view_src(glu->src[1]);
        if (g0 == mm0 && g1 == mm1 &&
            view_preserves_2d_identity(glu->src[0], mm0) &&
            view_preserves_2d_identity(glu->src[1], mm1)) {
            gate_mm = mm0;
            up_mm   = mm1;
        } else if (g0 == mm1 && g1 == mm0 &&
                   view_preserves_2d_identity(glu->src[0], mm1) &&
                   view_preserves_2d_identity(glu->src[1], mm0)) {
            gate_mm = mm1;
            up_mm   = mm0;
        }
        if (gate_mm && up_mm && view_preserves_2d_identity(dmm->src[1], glu)) {
            down_mm   = dmm;
            inter[0]  = gate_mm;
            inter[1]  = up_mm;
            inter[2]  = glu;
            n_inter   = 3;
            end_idx   = comp_idx[3];
        }
    }

    // Pattern B: unfused [gate_mm, silu, up_mm, mul, down_mm] or [mm0, mm1, silu, mul, down_mm]
    if (!down_mm && n_comp == 5 && comp[3]->op == GGML_OP_MUL && comp[4]->op == GGML_OP_MUL_MAT) {
        struct ggml_tensor * mm_a = nullptr;
        struct ggml_tensor * mm_b = nullptr;
        struct ggml_tensor * silu = nullptr;
        if (comp[1]->op == GGML_OP_UNARY && ggml_get_unary_op(comp[1]) == GGML_UNARY_OP_SILU &&
            comp[2]->op == GGML_OP_MUL_MAT) {
            mm_a = comp[0];
            silu = comp[1];
            mm_b = comp[2];
        } else if (comp[1]->op == GGML_OP_MUL_MAT &&
                   comp[2]->op == GGML_OP_UNARY && ggml_get_unary_op(comp[2]) == GGML_UNARY_OP_SILU) {
            mm_a = comp[0];
            mm_b = comp[1];
            silu = comp[2];
        }
        if (mm_a && mm_b && silu) {
            const struct ggml_tensor * s_in = unwrap_view_src(silu->src[0]);
            struct ggml_tensor * cand_gate = (s_in == mm_a) ? mm_a : ((s_in == mm_b) ? mm_b : nullptr);
            struct ggml_tensor * cand_up   = (cand_gate == mm_a) ? mm_b : mm_a;
            struct ggml_tensor * mul       = comp[3];
            struct ggml_tensor * dmm       = comp[4];
            const struct ggml_tensor * m0  = unwrap_view_src(mul->src[0]);
            const struct ggml_tensor * m1  = unwrap_view_src(mul->src[1]);
            const bool mul_ok =
                (m0 == silu && m1 == cand_up &&
                 view_preserves_2d_identity(mul->src[0], silu) &&
                 view_preserves_2d_identity(mul->src[1], cand_up)) ||
                (m0 == cand_up && m1 == silu &&
                 view_preserves_2d_identity(mul->src[0], cand_up) &&
                 view_preserves_2d_identity(mul->src[1], silu));
            if (cand_gate &&
                view_preserves_2d_identity(silu->src[0], cand_gate) &&
                mul_ok &&
                view_preserves_2d_identity(dmm->src[1], mul)) {
                gate_mm  = cand_gate;
                up_mm    = cand_up;
                down_mm  = dmm;
                inter[0] = gate_mm;
                inter[1] = up_mm;
                inter[2] = silu;
                inter[3] = mul;
                n_inter  = 4;
                end_idx  = comp_idx[4];
            }
        }
    }

    if (!down_mm || !gate_mm || !up_mm) {
        return 0;
    }

    // Validate all 3 projections & shared activation shapes.
    if (!is_fusible_swiglu_mul_mat(gate_mm) ||
        !is_fusible_swiglu_mul_mat(up_mm) ||
        !is_fusible_swiglu_mul_mat(down_mm)) {
        return 0;
    }

    const struct ggml_tensor * x = gate_mm->src[1];
    if (!x || !up_mm->src[1] ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(up_mm->src[1]) ||
        x->type != up_mm->src[1]->type ||
        up_mm->src[1]->data != x->data ||
        up_mm->src[1]->ne[0] != x->ne[0] ||
        up_mm->src[1]->ne[1] != x->ne[1] ||
        x->ne[2] != 1 || x->ne[3] != 1 ||
        up_mm->src[1]->ne[2] != 1 || up_mm->src[1]->ne[3] != 1) {
        return 0;
    }

    const uint32_t M = static_cast<uint32_t>(x->ne[1]);
    const uint32_t K = static_cast<uint32_t>(gate_mm->src[0]->ne[0]);
    const uint32_t H = static_cast<uint32_t>(gate_mm->src[0]->ne[1]);
    const uint32_t N = static_cast<uint32_t>(down_mm->src[0]->ne[1]);

    if (up_mm->src[0]->ne[0] != static_cast<int64_t>(K) ||
        up_mm->src[0]->ne[1] != static_cast<int64_t>(H) ||
        down_mm->src[0]->ne[0] != static_cast<int64_t>(H) ||
        down_mm->src[1]->ne[0] != static_cast<int64_t>(H) ||
        down_mm->src[1]->ne[1] != static_cast<int64_t>(M)) {
        return 0;
    }
    if ((K % QK8_0) != 0u || (H % QK8_0) != 0u || M == 0u || N == 0u) {
        return 0;
    }

    // Ensure no intermediate tensor is a graph output or consumed outside [start_idx, end_idx].
    // Checking cgraph->use_counts catches consumers in later backend splits that are absent
    // from this split's cgraph->nodes.
    if (!cgraph->use_counts || cgraph->visited_hash_set.size == 0 ||
        !cgraph->visited_hash_set.used || !cgraph->visited_hash_set.keys) {
        return 0;
    }
    for (int k = start_idx; k < end_idx; ++k) {
        const struct ggml_tensor * t = cgraph->nodes[k];
        if (!t || (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return 0;
        }
        const size_t hpos = ggml_hash_find(&cgraph->visited_hash_set, t);
        if (hpos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, hpos)) {
            return 0;
        }
        int32_t internal_uses = 0;
        for (int j = start_idx; j <= end_idx; ++j) {
            const struct ggml_tensor * consumer = cgraph->nodes[j];
            if (!consumer) {
                continue;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (consumer->src[s] == t) {
                    internal_uses++;
                }
            }
        }
        if (internal_uses <= 0 || cgraph->use_counts[hpos] != internal_uses) {
            return 0;
        }
    }
    auto tensor_aliases_inter = [&](const struct ggml_tensor * t) -> bool {
        if (!t) {
            return false;
        }
        const struct ggml_tensor * tv = t->view_src ? unwrap_view_src(t->view_src) : nullptr;
        for (int p = 0; p < n_inter; ++p) {
            if (tv == inter[p]) {
                return true;
            }
        }
        return false;
    };
    for (int k = 0; k < cgraph->n_nodes; ++k) {
        if (k >= start_idx && k <= end_idx) {
            continue;
        }
        const struct ggml_tensor * other = cgraph->nodes[k];
        if (!other) {
            continue;
        }
        if (tensor_aliases_inter(other)) {
            return 0;
        }
        for (int p = 0; p < n_inter; ++p) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (other->src[s] && unwrap_view_src(other->src[s]) == inter[p]) {
                    return 0;
                }
            }
        }
    }
    if (cgraph->leafs) {
        for (int k = 0; k < cgraph->n_leafs; ++k) {
            if (tensor_aliases_inter(cgraph->leafs[k])) {
                return 0;
            }
        }
    }
    for (size_t h = 0; h < cgraph->visited_hash_set.size; ++h) {
        if (!ggml_bitset_get(cgraph->visited_hash_set.used, h)) {
            continue;
        }
        const struct ggml_tensor * ht = cgraph->visited_hash_set.keys[h];
        bool is_internal_node = false;
        for (int j = start_idx; j <= end_idx; ++j) {
            if (cgraph->nodes[j] == ht) {
                is_internal_node = true;
                break;
            }
        }
        if (is_internal_node) {
            continue;
        }
        if (tensor_aliases_inter(ht)) {
            return 0;
        }
    }

    // Protect input activation if down_mm's output buffer aliases x.
    const float * a_ptr = reinterpret_cast<const float *>(x->data);
    if (tensors_overlap(down_mm, x)) {
        const size_t elems = static_cast<size_t>(M) * static_cast<size_t>(K);
        if (ctx->f32_a.size() < elems) {
            ctx->f32_a.resize(elems);
        }
        std::memcpy(ctx->f32_a.data(), a_ptr, elems * sizeof(float));
        a_ptr = ctx->f32_a.data();
    }

    ob_gemm_swiglu_quant_desc_t sdesc {};
    sdesc.w_gate_hk    = gate_mm->src[0]->data;
    sdesc.w_up_hk      = up_mm->src[0]->data;
    sdesc.w_down_nh    = down_mm->src[0]->data;
    sdesc.m_dim        = M;
    sdesc.k_dim        = K;
    sdesc.h_dim        = H;
    sdesc.n_dim        = N;
    sdesc.gate_is_q4_0 = (gate_mm->src[0]->type == GGML_TYPE_Q4_0) ? 1u : 0u;
    sdesc.up_is_q4_0   = (up_mm->src[0]->type   == GGML_TYPE_Q4_0) ? 1u : 0u;
    sdesc.down_is_q4_0 = (down_mm->src[0]->type == GGML_TYPE_Q4_0) ? 1u : 0u;

    ob_invoke_stats_t sstats {};
    const ob_status_t sst = ob_gemm_swiglu_f32_quant(
        ctx->ob_ctx, a_ptr, &sdesc, reinterpret_cast<float *>(down_mm->data), &sstats);
    if (sst != OB_OK) {
        GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_swiglu_f32_quant(M=%u,K=%u,H=%u,N=%u) failed: %s\n",
                       M, K, H, N, ob_status_string(sst));
        return -1;
    }

    ctx->total_swiglu_fused     += 1;
    ctx->total_mul_mat_calls    += 3;
    ctx->total_slice_calls      += 3;
    ctx->total_tiles_dispatched += sstats.tiles_dispatched;
    ctx->total_tile_cache_hits  += sstats.cache_hits;
    ctx->total_hw_us            += sstats.hw_execution_us;
    ctx->total_wall_us          += sstats.wall_time_us;
    if (sstats.flags & OB_STAT_FLAG_EMITTER_USED) {
        ctx->total_emitter_slices += 3;
    }
    if (sstats.flags & OB_STAT_FLAG_CPU_FALLBACK) {
        ctx->total_cpu_fallback_slices += 3;
        if (!ctx->env.mock_mode && ctx->total_cpu_fallback_slices <= 8) {
            GGML_LOG_WARN("ggml-edgetpu: CPU fallback in fused SwiGLU(M=%u,K=%u,H=%u,N=%u,flags=0x%x)\n",
                          M, K, H, N, static_cast<unsigned>(sstats.flags));
        }
    }

    *out_end_idx = end_idx;
    return 1;
}

static ggml_status ggml_backend_edgetpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_edgetpu_context *>(backend->context);
    ctx->total_graph_evals += 1;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                // First check for whole-block SwiGLU FFN: gate + up -> SiLU * up -> down
                // (chains in-SRAM on Tensor G5 via ob_gemm_swiglu_f32_quant with zero DRAM spill).
                {
                    int swiglu_end = -1;
                    const int swiglu_rc = try_compute_fused_swiglu(ctx, cgraph, i, &swiglu_end);
                    if (swiglu_rc < 0) {
                        return GGML_STATUS_FAILED;
                    }
                    if (swiglu_rc > 0) {
                        i = swiglu_end;
                        break;
                    }
                }

                // Collect sibling Q/K/V or gate/up MUL_MATs sharing the same input activation src1,
                // deferring intervening post-MUL_MAT 1D bias ADDs and post-Q/K ROPEs so ALL 3 [Q, K, V]
                // and [gate, up] hit ob_gemm_matmul_f32_quant_batch as single batched calls.
                // Exact byte-range hazard tracking + scratch-buffer commit preserves exact sequential
                // semantics even when ggml-alloc reuses a dead Q buffer for V or K.
                if (ctx->env.batch_siblings && !ctx->env.packed_q8 && is_batchable_sibling_mul_mat(node, nullptr)) {
                    constexpr int MAX_SIBLINGS = 8;
                    constexpr int MAX_PLAN     = 24;

                    struct plan_item {
                        bool                 is_mm;
                        int                  mm_idx; // valid when is_mm == true
                        struct ggml_tensor * op;     // valid when is_mm == false
                    };

                    struct ggml_tensor * mm_nodes[MAX_SIBLINGS];
                    bool                 mm_needs_scratch[MAX_SIBLINGS];
                    size_t               mm_scratch_offs[MAX_SIBLINGS];
                    plan_item            plan[MAX_PLAN];
                    int n_mm   = 0;
                    int n_plan = 0;

                    const struct ggml_tensor * shared_x = node->src[1];
                    mm_nodes[n_mm]         = node;
                    mm_needs_scratch[n_mm] = false;
                    mm_scratch_offs[n_mm]  = 0;
                    plan[n_plan++]         = { true, n_mm, nullptr };
                    n_mm++;

                    size_t total_scratch_elems = 0;
                    const uint32_t M = static_cast<uint32_t>(shared_x->ne[1]);
                    const uint32_t K = static_cast<uint32_t>(shared_x->ne[0]);

                    auto produced_by_group = [&](const struct ggml_tensor * t) -> bool {
                        const struct ggml_tensor * base = unwrap_view_src(t);
                        for (int k = 0; k < n_mm; ++k) {
                            if (base == mm_nodes[k]) {
                                return true;
                            }
                        }
                        for (int p = 0; p < n_plan; ++p) {
                            if (!plan[p].is_mm && base == plan[p].op) {
                                return true;
                            }
                        }
                        return false;
                    };

                    int j = i + 1;
                    while (j < cgraph->n_nodes && n_plan < MAX_PLAN) {
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
                        if (next->op == GGML_OP_ADD || next->op == GGML_OP_ROPE) {
                            if (produced_by_group(next->src[0]) && !tensors_overlap(next, shared_x)) {
                                // Ensure deferred op does not overwrite weights of any already-queued sibling.
                                bool w_hazard = false;
                                for (int k = 0; k < n_mm; ++k) {
                                    if (tensors_overlap(next, mm_nodes[k]->src[0])) {
                                        w_hazard = true;
                                        break;
                                    }
                                }
                                if (!w_hazard) {
                                    plan[n_plan++] = { false, -1, next };
                                    ++j;
                                    continue;
                                }
                            }
                            break;
                        }
                        if (n_mm < MAX_SIBLINGS && is_batchable_sibling_mul_mat(next, shared_x)) {
                            // Verify next's weights are untouched by all prior ops in plan.
                            bool w_hazard = false;
                            for (int p = 0; p < n_plan; ++p) {
                                const struct ggml_tensor * prior_dst = plan[p].is_mm ? mm_nodes[plan[p].mm_idx] : plan[p].op;
                                if (tensors_overlap(next->src[0], prior_dst)) {
                                    w_hazard = true;
                                    break;
                                }
                            }
                            if (w_hazard) {
                                break;
                            }
                            // Check if next's output buffer aliases ANY prior plan tensor (e.g. ggml-alloc
                            // reusing dead Q_mm/Q_bias storage for V_mm or K_mm). If so, stage in sibling_scratch
                            // and commit at next's exact original timestamp.
                            bool dst_aliases = false;
                            for (int p = 0; p < n_plan; ++p) {
                                if (plan[p].is_mm) {
                                    const struct ggml_tensor * pm = mm_nodes[plan[p].mm_idx];
                                    if (tensors_overlap(next, pm) || tensors_overlap(next, pm->src[0])) {
                                        dst_aliases = true;
                                        break;
                                    }
                                } else {
                                    const struct ggml_tensor * po = plan[p].op;
                                    if (tensors_overlap(next, po) ||
                                        tensors_overlap(next, po->src[0]) ||
                                        tensors_overlap(next, po->src[1]) ||
                                        tensors_overlap(next, po->src[2])) {
                                        dst_aliases = true;
                                        break;
                                    }
                                }
                            }
                            mm_nodes[n_mm]         = next;
                            mm_needs_scratch[n_mm] = dst_aliases;
                            if (dst_aliases) {
                                mm_scratch_offs[n_mm] = total_scratch_elems;
                                total_scratch_elems += static_cast<size_t>(M) * static_cast<size_t>(next->src[0]->ne[1]);
                            } else {
                                mm_scratch_offs[n_mm] = 0;
                            }
                            plan[n_plan++] = { true, n_mm, nullptr };
                            n_mm++;
                            ++j;
                            continue;
                        }
                        break;
                    }

                    if (n_mm > 1) {
                        ctx->total_batched_groups += 1;
                        ctx->total_batched_ops    += static_cast<uint64_t>(n_mm);
                        if (n_mm == 3) {
                            ctx->total_batched_3_groups += 1;
                        } else if (n_mm == 2) {
                            ctx->total_batched_2_groups += 1;
                        }

                        if (total_scratch_elems > 0 && ctx->sibling_scratch.size() < total_scratch_elems) {
                            ctx->sibling_scratch.resize(total_scratch_elems);
                        }

                        ob_gemm_f32_quant_job_t jobs[MAX_SIBLINGS];
                        for (int k = 0; k < n_mm; ++k) {
                            jobs[k]            = {};
                            jobs[k].b_nk       = mm_nodes[k]->src[0]->data;
                            jobs[k].c_mn       = mm_needs_scratch[k]
                                                     ? (ctx->sibling_scratch.data() + mm_scratch_offs[k])
                                                     : reinterpret_cast<float *>(mm_nodes[k]->data);
                            jobs[k].n_dim      = static_cast<uint32_t>(mm_nodes[k]->src[0]->ne[1]);
                            jobs[k].activation = OB_ACT_NONE;
                            jobs[k].is_q4_0    = (mm_nodes[k]->src[0]->type == GGML_TYPE_Q4_0) ? 1u : 0u;
                        }

                        ob_invoke_stats_t bstats {};
                        const ob_status_t bst = ob_gemm_matmul_f32_quant_batch(
                            ctx->ob_ctx,
                            reinterpret_cast<const float *>(shared_x->data),
                            M, K,
                            jobs,
                            static_cast<uint32_t>(n_mm),
                            &bstats);
                        if (bst != OB_OK) {
                            GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_matmul_f32_quant_batch(n=%d,M=%u,K=%u) failed: %s (%s)\n",
                                           n_mm, M, K, ob_status_string(bst), ob_context_last_error(ctx->ob_ctx));
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
                                GGML_LOG_WARN("ggml-edgetpu: CPU fallback in sibling batch(n=%d,M=%u,K=%u,flags=0x%x)\n",
                                              n_mm, M, K, static_cast<unsigned>(bstats.flags));
                            }
                        }

                        // Replay deferred ADDs / ROPEs and commit any scratch-staged sibling outputs in exact original order.
                        for (int p = 0; p < n_plan; ++p) {
                            if (plan[p].is_mm) {
                                const int k = plan[p].mm_idx;
                                if (mm_needs_scratch[k]) {
                                    const size_t elems = static_cast<size_t>(M) * static_cast<size_t>(mm_nodes[k]->src[0]->ne[1]);
                                    std::memcpy(mm_nodes[k]->data,
                                                ctx->sibling_scratch.data() + mm_scratch_offs[k],
                                                elems * sizeof(float));
                                }
                            } else if (plan[p].op->op == GGML_OP_ADD) {
                                const ggml_status st = ggml_edgetpu_compute_forward_add_bias(ctx, plan[p].op);
                                if (st != GGML_STATUS_SUCCESS) {
                                    return st;
                                }
                            } else if (plan[p].op->op == GGML_OP_ROPE) {
                                const ggml_status st = ggml_edgetpu_compute_forward_rope(ctx, plan[p].op);
                                if (st != GGML_STATUS_SUCCESS) {
                                    return st;
                                }
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
            case GGML_OP_MUL_MAT_ID: {
                const ggml_status st = ggml_edgetpu_compute_forward_mul_mat_id(ctx, node);
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
            case GGML_OP_ROPE: {
                const ggml_status st = ggml_edgetpu_compute_forward_rope(ctx, node);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                break;
            }
            case GGML_OP_GLU: {
                const ggml_status st = ggml_edgetpu_compute_forward_glu(ctx, node);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                break;
            }
            case GGML_OP_UNARY: {
                if (ggml_get_unary_op(node) != GGML_UNARY_OP_SILU) {
                    GGML_ABORT("ggml-edgetpu: unsupported unary op %s\n", ggml_op_desc(node));
                }
                const ggml_status st = ggml_edgetpu_compute_forward_silu(ctx, node);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                break;
            }
            case GGML_OP_MUL: {
                const ggml_status st = ggml_edgetpu_compute_forward_mul(ctx, node);
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
#if defined(__linux__) || defined(__ANDROID__)
    const long pages     = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    size_t total_bytes = (pages > 0 && page_size > 0)
        ? static_cast<size_t>(pages) * static_cast<size_t>(page_size)
        : 0;
    size_t avail_bytes = 0;
    bool   have_avail  = false;

    if (FILE * f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), f)) {
            if (total_bytes == 0 && std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
                total_bytes = static_cast<size_t>(kb) * 1024u;
            } else if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                avail_bytes = static_cast<size_t>(kb) * 1024u;
                have_avail  = true;
                break;
            }
        }
        std::fclose(f);
    }
    *total = total_bytes;
    *free  = have_avail ? avail_bytes : total_bytes;
#else
    *free  = 0;
    *total = 0;
#endif
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

            // Respect explicit unquantized FP32/BF16/F16 precision requests & Hadamard hint.
            const enum ggml_prec prec = ggml_prec_get_src(op, 1);
            if (prec == GGML_PREC_F32 || prec == GGML_PREC_BF16 || prec == GGML_PREC_F16) {
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

            g_hot_env.ensure_initialized();
            const uint64_t min_elements = g_hot_env.min_elements.load(std::memory_order_relaxed);
            if (min_elements > 0) {
                const uint64_t work =
                    static_cast<uint64_t>(M) *
                    static_cast<uint64_t>(K) *
                    static_cast<uint64_t>(N) *
                    static_cast<uint64_t>(src1->ne[2]) *
                    static_cast<uint64_t>(src1->ne[3]);
                if (work < min_elements) {
                    return false;
                }
            }

            return true;
        }

        case GGML_OP_MUL_MAT_ID: {
            const struct ggml_tensor * src0 = op->src[0]; // [K, N, n_experts, 1]
            const struct ggml_tensor * src1 = op->src[1]; // [K, n_used or 1, n_tokens, 1]
            const struct ggml_tensor * ids  = op->src[2]; // [n_used, n_tokens] I32
            if (!src0 || !src1 || !ids) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
                return false;
            }

            // Respect explicit unquantized FP32/BF16/F16 precision requests & Hadamard hint.
            const enum ggml_prec prec = ggml_prec_get_src(op, 1);
            if (prec == GGML_PREC_F32 || prec == GGML_PREC_BF16 || prec == GGML_PREC_F16) {
                return false;
            }
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }

            if (src0->ne[3] != 1 || src1->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) {
                return false;
            }

            // Each 2D expert slice in src0 must be contiguous; src0 dim 2 and ids dim 1 may be strided views.
            if (!ggml_is_contiguous_2(src0) || !ggml_is_contiguous_1(src1) || !ggml_is_contiguous_1(op)) {
                return false;
            }
            if (src0->nb[0] != ggml_type_size(src0->type) ||
                (src0->ne[1] > 1 && src0->nb[1] != ggml_row_size(src0->type, src0->ne[0])) ||
                src1->nb[0] != ggml_type_size(src1->type) ||
                ids->nb[0]  != sizeof(int32_t) ||
                op->nb[0]   != sizeof(float)) {
                return false;
            }

            const int64_t K        = src0->ne[0];
            const int64_t N        = src0->ne[1];
            const int64_t n_expert = src0->ne[2];
            const int64_t n_used   = ids->ne[0];
            const int64_t n_tokens = ids->ne[1];
            if (K <= 0 || N <= 0 || n_expert <= 0 || n_used <= 0 || n_tokens <= 0) {
                return false;
            }
            if (src1->ne[0] != K || src1->ne[2] != n_tokens) {
                return false;
            }
            if (src1->ne[1] != 1 && src1->ne[1] != n_used) {
                return false;
            }
            if (op->ne[0] != N || op->ne[1] != n_used || op->ne[2] != n_tokens) {
                return false;
            }
            if ((src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_0) ||
                (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_0)) {
                return false;
            }
            if ((K % QK8_0) != 0) {
                return false;
            }

            g_hot_env.ensure_initialized();
            const uint64_t min_elements = g_hot_env.min_elements.load(std::memory_order_relaxed);
            if (min_elements > 0) {
                const uint64_t work =
                    static_cast<uint64_t>(n_tokens) *
                    static_cast<uint64_t>(n_used) *
                    static_cast<uint64_t>(K) *
                    static_cast<uint64_t>(N);
                if (work < min_elements) {
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

        case GGML_OP_ROPE: {
            g_hot_env.ensure_initialized();
            if (!g_hot_env.enable_rope.load(std::memory_order_relaxed)) {
                return false;
            }
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            const struct ggml_tensor * src2 = op->src[2];
            if (!src0 || !src1) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
                return false;
            }
            if (op->nb[0] != sizeof(float) || src0->nb[0] != sizeof(float)) {
                return false;
            }
            const int64_t n_dims = static_cast<int64_t>(static_cast<const int32_t *>(op->op_params)[1]);
            const int     mode   = static_cast<const int32_t *>(op->op_params)[2];
            const int64_t n_offs = static_cast<int64_t>(static_cast<const int32_t *>(op->op_params)[15]);
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
                return false;
            }
            if (n_dims <= 0 || (n_dims % 2) != 0 || n_offs < 0 || (n_offs % 2) != 0 ||
                n_offs + n_dims > op->ne[0] || n_offs + n_dims > src0->ne[0]) {
                return false;
            }
            if (src2 != nullptr && (src2->type != GGML_TYPE_F32 || src2->ne[0] < n_dims / 2)) {
                return false;
            }
            // Accept post-MUL_MAT/ADD Q/K RoPE so [Q, K, V] stay in one EDGETPU split (num_jobs=3).
            const struct ggml_tensor * base = unwrap_view_src(src0);
            if (!base) {
                return false;
            }
            if (base->op == GGML_OP_MUL_MAT || base->op == GGML_OP_ADD) {
                return ggml_backend_edgetpu_device_supports_op(dev, base);
            }
            return false;
        }

        case GGML_OP_GLU: {
            g_hot_env.ensure_initialized();
            if (!g_hot_env.enable_glu.load(std::memory_order_relaxed)) {
                return false;
            }
            if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU) {
                return false;
            }
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (!src0 || !src1) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (!ggml_is_contiguous_1(src0) || !ggml_is_contiguous_1(src1) || !ggml_is_contiguous_1(op)) {
                return false;
            }
            if (src0->ne[0] != src1->ne[0] || op->ne[0] != src0->ne[0] ||
                ggml_nrows(src0) != ggml_nrows(src1) || ggml_nrows(op) != ggml_nrows(src0)) {
                return false;
            }
            const struct ggml_tensor * b0 = unwrap_view_src(src0);
            const struct ggml_tensor * b1 = unwrap_view_src(src1);
            const bool b0_ok = b0 && (b0->op == GGML_OP_MUL_MAT || b0->op == GGML_OP_ADD) &&
                               ggml_backend_edgetpu_device_supports_op(dev, b0);
            const bool b1_ok = b1 && (b1->op == GGML_OP_MUL_MAT || b1->op == GGML_OP_ADD) &&
                               ggml_backend_edgetpu_device_supports_op(dev, b1);
            if (!b0_ok || !b1_ok) {
                return false;
            }
            // Enforce OPENBUENOS_EDGETPU_GLU_MAX_M unconditionally during op support checks
            // (supports_op cannot observe downstream down-projection format to guarantee whole-block fusion).
            const int64_t glu_max_m = g_hot_env.glu_max_m.load(std::memory_order_relaxed);
            if (glu_max_m <= 0 || ggml_nrows(op) > glu_max_m) {
                return false;
            }
            return true;
        }

        case GGML_OP_UNARY: {
            g_hot_env.ensure_initialized();
            if (!g_hot_env.fuse_swiglu.load(std::memory_order_relaxed) ||
                g_hot_env.packed_q8.load(std::memory_order_relaxed)) {
                return false;
            }
            if (ggml_get_unary_op(op) != GGML_UNARY_OP_SILU) {
                return false;
            }
            const struct ggml_tensor * src0 = op->src[0];
            if (!src0 || op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
                return false;
            }
            const struct ggml_tensor * base = unwrap_view_src(src0);
            return base && is_fusible_swiglu_mul_mat(base) &&
                   view_preserves_2d_identity(src0, base) &&
                   (base->ne[0] % QK8_0) == 0 &&
                   ggml_backend_edgetpu_device_supports_op(dev, base);
        }

        case GGML_OP_MUL: {
            g_hot_env.ensure_initialized();
            if (!g_hot_env.fuse_swiglu.load(std::memory_order_relaxed) ||
                g_hot_env.packed_q8.load(std::memory_order_relaxed)) {
                return false;
            }
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (!src0 || !src1) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
                return false;
            }
            if (!ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, op)) {
                return false;
            }
            // Accept ONLY SwiGLU gate*up elementwise MUL (one input is SILU(gate_mm), other is up_mm sharing input x).
            const struct ggml_tensor * b0 = unwrap_view_src(src0);
            const struct ggml_tensor * b1 = unwrap_view_src(src1);
            auto is_silu_mul_pair = [&](const struct ggml_tensor * raw_silu,
                                        const struct ggml_tensor * raw_up,
                                        const struct ggml_tensor * silu_t,
                                        const struct ggml_tensor * up_t) -> bool {
                if (!silu_t || !up_t || silu_t->op != GGML_OP_UNARY || ggml_get_unary_op(silu_t) != GGML_UNARY_OP_SILU) {
                    return false;
                }
                if (!view_preserves_2d_identity(raw_silu, silu_t) ||
                    !view_preserves_2d_identity(raw_up, up_t)) {
                    return false;
                }
                const struct ggml_tensor * gate_t = unwrap_view_src(silu_t->src[0]);
                if (!gate_t || !is_fusible_swiglu_mul_mat(gate_t) || !is_fusible_swiglu_mul_mat(up_t)) {
                    return false;
                }
                if (!view_preserves_2d_identity(silu_t->src[0], gate_t)) {
                    return false;
                }
                if (unwrap_view_src(gate_t->src[1]) != unwrap_view_src(up_t->src[1]) || (gate_t->ne[0] % QK8_0) != 0) {
                    return false;
                }
                return ggml_backend_edgetpu_device_supports_op(dev, gate_t) &&
                       ggml_backend_edgetpu_device_supports_op(dev, up_t);
            };
            return is_silu_mul_pair(src0, src1, b0, b1) || is_silu_mul_pair(src1, src0, b1, b0);
        }

        // Explicitly return false for RMS_NORM, SOFT_MAX, FLASH_ATTN, GET_ROWS, etc.
        // so ggml-backend-sched automatically routes remaining glue ops to ARM64 CPU.
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
    g_hot_env.store(env);

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

    if (env.has_freetile) {
        st = ob_context_set_freetile(ob_ctx, env.freetile ? 1 : 0);
        if (st != OB_OK) {
            GGML_LOG_WARN("ggml-edgetpu: ob_context_set_freetile(%d) failed (%s): %s; using legacy emitter\n",
                          env.freetile ? 1 : 0, ob_status_string(st), ob_context_last_error(ob_ctx));
        }
    }

    const char * interp_path = (env.dive_interp_path && env.dive_interp_path[0] != '\0')
                                   ? env.dive_interp_path
                                   : nullptr;
    bool emitter_ok = false;
    if (!env.mock_mode) {
        st = ob_gemm_register_builtin_emitter(ob_ctx, interp_path);
        if (st != OB_OK) {
            GGML_LOG_ERROR("ggml-edgetpu: ob_gemm_register_builtin_emitter('%s') failed (%s): %s\n",
                           interp_path ? interp_path : "<auto>",
                           ob_status_string(st), ob_context_last_error(ob_ctx));
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
            "[mock=%d, emitter=%d, freetile=%d, interp=%s, credits=%u, cache_cap=%u, pool_mb=%u, "
            "tile_limits=(%u,%u,%u), pwr=(perf=%u,core=0x%02x,mem=0x%02x,qos=0x%02x,prio=%u), "
            "batch_siblings=%d, min_elements=%llu, packed_q8=%d, libopenbuenos=%d]\n",
            ob_context_is_mock(ob_ctx),
            emitter_ok ? 1 : 0,
            ob_context_get_freetile(ob_ctx),
            env.mock_mode ? "(mock)" : (interp_path ? interp_path : "<auto>"),
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
