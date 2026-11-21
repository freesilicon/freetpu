/*
 * libopenbuenos - Zero-Google-.so Bare-Metal Tensor G5 (Buenos A0 / rango) TPU Runtime
 *
 * Public C11 / C++17 ABI header.
 * Designed for direct consumption by llama.cpp (ggml-edgetpu) and standalone C/C++ tools.
 *
 * Transport:
 *   - Device node : /dev/edgetpu-soc (buenos.ko ioctl ABI 1.4, Litebuf VII)
 *   - Buffer heap : /dev/dma_heap/system + EDGETPU_MAP_DMABUF (36-bit IOVA)
 *   - Concurrency : Up to 16 in-flight VII credits (credit_size=16) over eventfd
 *   - Host testing: Set ob_config_t::mock_mode = 1 to run offline without /dev/edgetpu-soc
 */
#ifndef OPENBUENOS_H_
#define OPENBUENOS_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Version & Compile-Time Constants
 * ========================================================================= */

#define OB_API_VERSION_MAJOR 0u
#define OB_API_VERSION_MINOR 1u
#define OB_API_VERSION_PATCH 0u

#define OB_MAX_CREDITS       16u
#define OB_MAX_GRAPH_IO      8u
#define OB_QK8_0             32u

/* =========================================================================
 * Status Codes
 * ========================================================================= */

typedef enum ob_status {
  OB_OK              = 0,
  OB_ERR_INVALID_ARG = -1,
  OB_ERR_NO_MEMORY   = -2,
  OB_ERR_DEVICE      = -3,
  OB_ERR_IOCTL       = -4,
  OB_ERR_TIMEOUT     = -5,
  OB_ERR_BAD_DGC0    = -6,
  OB_ERR_BUSY        = -7,
  OB_ERR_NOT_FOUND   = -8,
  OB_ERR_UNSUPPORTED = -9,
  OB_ERR_FW_ERROR    = -10,
  OB_ERR_OVERFLOW    = -11,
  OB_ERR_IO          = -12
} ob_status_t;

/* Buffer allocation flags (bitmask) */
enum ob_buffer_flags {
  OB_BUF_BIDIRECTIONAL = 0u,        /* Default read/write DMA buffer */
  OB_BUF_TO_DEVICE     = (1u << 0), /* Hint: host -> TPU */
  OB_BUF_FROM_DEVICE   = (1u << 1), /* Hint: TPU -> host */
  OB_BUF_NO_POOL       = (1u << 2), /* Bypass slab/pool cache on alloc/free */
  OB_BUF_ZERO_INIT     = (1u << 3)  /* Zero-fill buffer on allocation */
};

/* Fused GEMM activation enum (matches DarwiNN / TFLite FC activation codes) */
enum ob_activation {
  OB_ACT_NONE  = 0u,
  OB_ACT_RELU  = 1u,
  OB_ACT_RELU6 = 3u
};

/* Execution stats flags */
enum ob_stat_flags {
  OB_STAT_FLAG_MOCK         = (1u << 0),
  OB_STAT_FLAG_CACHE_HIT    = (1u << 1),
  OB_STAT_FLAG_CPU_FALLBACK = (1u << 2),
  OB_STAT_FLAG_EMITTER_USED = (1u << 3)
};

/* =========================================================================
 * Opaque Handles & Public Structs
 * ========================================================================= */

typedef struct ob_context ob_context_t;
typedef struct ob_buffer  ob_buffer_t;
typedef struct ob_graph   ob_graph_t;

/* Opaque non-zero asynchronous execution handle (0 == invalid) */
typedef uint64_t ob_exec_handle_t;

/* ggml-compatible Q8_0 quantization block (34 bytes: IEEE-754 fp16 scale + 32 int8 quants) */
#if defined(__GNUC__) || defined(__clang__)
typedef struct __attribute__((packed)) ob_block_q8_0 {
  uint16_t d;
  int8_t   qs[OB_QK8_0];
} ob_block_q8_0_t;
#else
#pragma pack(push, 1)
typedef struct ob_block_q8_0 {
  uint16_t d;
  int8_t   qs[OB_QK8_0];
} ob_block_q8_0_t;
#pragma pack(pop)
#endif

/* Context creation configuration (zero-initialized struct is valid and uses defaults) */
typedef struct ob_config {
  uint32_t    struct_size;           /* Optional: sizeof(ob_config_t) or 0 */
  int         mock_mode;             /* 1 = offline mock device (no /dev/edgetpu-soc required) */
  const char* device_path;           /* NULL -> "/dev/edgetpu-soc" */
  const char* dma_heap_path;         /* NULL -> "/dev/dma_heap/system" */
  uint32_t    max_inflight;          /* 0 -> default 16 (clamped to [1, OB_MAX_CREDITS]) */
  uint32_t    graph_cache_capacity;  /* 0 -> default 64 cached registered graphs */
  uint32_t    buffer_pool_max_bytes; /* 0 -> default 64 MiB pooled DMA-BUF capacity */
  uint32_t    timeout_ms;            /* 0 -> default 5000 ms */
  uint8_t     core_power_state;      /* 0 -> default 0x0b */
  uint8_t     mem_power_state;       /* 0 -> default 0x03 */
  uint8_t     qos_class;             /* 0 -> default 0x01 */
  uint8_t     cluster_id;            /* 0 -> use .dgc0 cluster_id (default 0x02 on rango) */
  uint8_t     priority;              /* VII queue priority (default 0) */
  uint8_t     reserved[3];
} ob_config_t;

/* Execution telemetry returned by invoke / wait / matmul */
typedef struct ob_invoke_stats {
  uint64_t hw_execution_us; /* Firmware-reported hardware execution time (us) */
  uint64_t wall_time_us;    /* Host monotonic round-trip time (us) */
  uint64_t seq;             /* VII mailbox sequence number */
  uint32_t model_handle;    /* Firmware graph handle (RegisterModelResponse) */
  uint16_t vii_resp_code;   /* VII response status code (0 = OK) */
  uint16_t flags;           /* Bitmask of enum ob_stat_flags */
  uint32_t tiles_dispatched;/* Number of hardware/mock tile dispatches (matmul) */
  uint32_t cache_hits;      /* Number of graph cache hits during operation */
} ob_invoke_stats_t;

/* Parsed .dgc0 container metadata */
typedef struct ob_dgc0_info {
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint64_t input_sizes[OB_MAX_GRAPH_IO];
  uint64_t output_sizes[OB_MAX_GRAPH_IO];
  uint64_t dive_scratch_size;
  uint64_t ext_vm_ir_size;
  uint64_t ext_image_size;
  uint64_t ext_params_size;
  uint64_t compiler_cl;
  uint32_t model_handle;
  uint8_t  cluster_id;
  uint8_t  has_params;
  uint8_t  cache_hit;
  uint8_t  reserved;
} ob_dgc0_info_t;

/* Descriptor passed to a pluggable clean-room (M, K, N) .dgc0 GEMM emitter */
typedef struct ob_gemm_desc {
  uint32_t       m_dim;
  uint32_t       k_dim;
  uint32_t       n_dim;
  float          s_in;
  int32_t        zp_in;
  float          s_w;
  int32_t        zp_w;
  float          s_out;
  int32_t        zp_out;
  uint32_t       activation;          /* enum ob_activation */
  const int8_t*  weights_nk;          /* Optional [n_dim, k_dim] int8 weights */
  const int32_t* bias_n;              /* Optional [n_dim] int32 biases (NULL -> zeros) */
  const uint8_t* packed_params;       /* Optional pre-packed ext[2] bytes */
  size_t         packed_params_bytes; /* Byte size of packed_params */
  uint64_t       cache_key;           /* Optional non-zero key for persistent graph cache */
} ob_gemm_desc_t;

/*
 * Pluggable clean-room .dgc0 emitter callback for (M, K, N) GEMM tiles.
 *
 * Contract:
 *   - Writes a complete valid .dgc0 binary into `out_buf` (capacity `out_cap`)
 *     and stores the exact byte length in `*out_size`, returning OB_OK.
 *   - If `out_buf == NULL` or `out_cap` is too small, stores the required byte
 *     size in `*out_size` and returns OB_ERR_OVERFLOW (libopenbuenos will resize
 *     its scratch buffer and retry once).
 *   - Return OB_ERR_UNSUPPORTED if the emitter does not support this tile shape
 *     (libopenbuenos will fall back to CPU reference if allowed).
 */
typedef ob_status_t (*ob_gemm_emitter_fn)(const ob_gemm_desc_t* desc,
                                          uint8_t*              out_buf,
                                          size_t                out_cap,
                                          size_t*               out_size,
                                          void*                 user_data);

/* =========================================================================
 * Context Lifecycle & Diagnostics
 * ========================================================================= */

ob_config_t ob_config_default(void);

ob_status_t ob_context_create(const ob_config_t* config, ob_context_t** out_ctx);
void        ob_context_destroy(ob_context_t* ctx);

const char* ob_status_string(ob_status_t status);
const char* ob_context_last_error(const ob_context_t* ctx);
uint32_t    ob_context_credit_size(const ob_context_t* ctx);
int         ob_context_is_mock(const ob_context_t* ctx);

/* =========================================================================
 * DMA-BUF Slab / Pool Buffer API (/dev/dma_heap/system + EDGETPU_MAP_DMABUF)
 * ========================================================================= */

ob_status_t ob_buffer_alloc(ob_context_t* ctx,
                            size_t        size_bytes,
                            uint32_t      flags,
                            ob_buffer_t** out_buf);
void        ob_buffer_free(ob_buffer_t* buf);

void*       ob_buffer_data(ob_buffer_t* buf);
size_t      ob_buffer_size(const ob_buffer_t* buf);
uint64_t    ob_buffer_iova(const ob_buffer_t* buf);

ob_status_t ob_buffer_sync_to_device(ob_buffer_t* buf);
ob_status_t ob_buffer_sync_to_cpu(ob_buffer_t* buf);

/* Convenience host-copy helpers with automatic DMA-BUF cache sync */
ob_status_t ob_buffer_write(ob_buffer_t* buf, const void* src, size_t bytes, size_t offset);
ob_status_t ob_buffer_read(ob_buffer_t* buf, void* dst, size_t bytes, size_t offset);

ob_status_t ob_context_buffer_pool_stats(const ob_context_t* ctx,
                                         size_t*             out_pooled_bytes,
                                         size_t*             out_pooled_count,
                                         uint64_t*           out_pool_hits,
                                         uint64_t*           out_pool_misses);

/* =========================================================================
 * Graph Registration (Persistent LRU Handle Cache) & Sync/Async Execution
 * ========================================================================= */

/*
 * Registers a .dgc0 executable with the TPU firmware (or returns a cached handle).
 *
 * Caching:
 *   - If `cache_key != 0`, lookup/insert uses `cache_key`.
 *   - If `cache_key == 0`, lookup/insert hashes `(dgc0_bytes, dgc0_size)` with SHA-256.
 *   - Cache hits skip DMA upload and RegisterModel ioctl entirely (zero per-token
 *     registration overhead in llama.cpp decode loops).
 *   - Caller must release each returned handle with `ob_graph_release()`.
 */
ob_status_t ob_graph_register_dgc0(ob_context_t* ctx,
                                   const void*   dgc0_bytes,
                                   size_t        dgc0_size,
                                   uint64_t      cache_key,
                                   ob_graph_t**  out_graph);

ob_status_t ob_graph_register_dgc0_file(ob_context_t* ctx,
                                        const char*   path,
                                        uint64_t      cache_key,
                                        ob_graph_t**  out_graph);

void        ob_graph_release(ob_graph_t* graph);

ob_status_t ob_graph_get_info(const ob_graph_t* graph, ob_dgc0_info_t* out_info);

/* Parse a .dgc0 buffer offline without allocating device buffers or registering */
ob_status_t ob_dgc0_inspect(const void*     dgc0_bytes,
                            size_t          dgc0_size,
                            ob_dgc0_info_t* out_info);

ob_status_t ob_context_graph_cache_stats(const ob_context_t* ctx,
                                         size_t*             out_entries,
                                         uint64_t*           out_hits,
                                         uint64_t*           out_misses,
                                         uint64_t*           out_evictions);

/* Synchronous execution (submits + waits on a single VII credit) */
ob_status_t ob_graph_invoke(ob_graph_t*        graph,
                            ob_buffer_t* const* inputs,
                            uint32_t           num_inputs,
                            ob_buffer_t* const* outputs,
                            uint32_t           num_outputs,
                            ob_invoke_stats_t* out_stats);

/*
 * Asynchronous execution over the 16-credit Litebuf VII mailbox ring.
 *
 *   - Supports up to `ob_context_credit_size(ctx)` (16 on rango) concurrent
 *     in-flight commands.
 *   - Returns OB_ERR_BUSY if all credits are currently occupied.
 *   - Input/output ob_buffer_t instances must remain valid until `ob_graph_wait()`
 *     completes for the returned handle.
 */
ob_status_t ob_graph_submit_async(ob_graph_t*        graph,
                                  ob_buffer_t* const* inputs,
                                  uint32_t           num_inputs,
                                  ob_buffer_t* const* outputs,
                                  uint32_t           num_outputs,
                                  ob_exec_handle_t*  out_handle);

/* Wait for a specific in-flight async handle (timeout_ms == 0 uses context default) */
ob_status_t ob_graph_wait(ob_context_t*      ctx,
                          ob_exec_handle_t   handle,
                          uint32_t           timeout_ms,
                          ob_invoke_stats_t* out_stats);

/* Drain and wait for all currently in-flight async handles */
ob_status_t ob_graph_wait_all(ob_context_t* ctx, uint32_t timeout_ms);

/* =========================================================================
 * Tensor G5 (Buenos) Weight Packing & Tiled GEMM API for llama.cpp
 * ========================================================================= */

/*
 * Returns exact packed ext[2] parameter byte size for a (k_dim, n_dim) INT8 FC/GEMM:
 *   n_pad = ceil(n_dim / 8) * 8
 *   k_pad = ceil(k_dim / 4) * 4
 *   bytes = n_pad * sizeof(int32_t) + (k_pad / 4) * n_pad * 4
 */
size_t ob_gemm_packed_weight_bytes(uint32_t k_dim, uint32_t n_dim);

/*
 * Packs INT8 [N, K] row-major weights + optional INT32 [N] bias into Tensor G5
 * buenos_a0 ext[2] format:
 *   1. int32[n_pad] little-endian bias (0 for padded channels [n_dim..n_pad) or NULL bias_n)
 *   2. uint8[k_pad/4, n_pad, 4] weights encoded as (int8_w + 128) & 0xFF
 *      (raw byte 0x00 for padded channels n >= n_dim; 128 for padded k >= k_dim).
 */
ob_status_t ob_gemm_pack_weights_int8(const int8_t*  weights_nk,
                                      const int32_t* bias_n,
                                      uint32_t       k_dim,
                                      uint32_t       n_dim,
                                      uint8_t*       out_packed,
                                      size_t         out_cap);

/* Attach/detach a pluggable clean-room (M, K, N) .dgc0 tile emitter callback */
ob_status_t ob_gemm_register_emitter(ob_context_t*      ctx,
                                     ob_gemm_emitter_fn fn,
                                     void*              user_data);

/*
 * Attach the clean-room C++17 Tensor G5 multi-tile (M, K, N) .dgc0 emitter.
 *
 * `dive_interp_path` loads the external RV64 DIVE interpreter binary (ext[1],
 * 638,977 B raw image, or extracts ext[1] automatically if a .dgc0 file is
 * passed). If NULL or empty, checks env `OPENBUENOS_DIVE_INTERP_PATH` and
 * falls back to "/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin".
 */
ob_status_t ob_gemm_register_builtin_emitter(ob_context_t* ctx,
                                             const char*   dive_interp_path);

/*
 * Helper: extract the RV64 DIVE interpreter binary (ext[1]) from an existing
 * .dgc0 container file without embedding any proprietary binary inside libopenbuenos.
 * If `out_buf == NULL` or `out_cap` is too small, stores required byte count in
 * `*out_size` and returns OB_ERR_OVERFLOW.
 */
ob_status_t ob_gemm_extract_dive_interp_from_dgc0(const char* dgc0_path,
                                                  uint8_t*    out_buf,
                                                  size_t      out_cap,
                                                  size_t*     out_size);

/* Configure maximum single-tile (M, K, N) dimensions for automatic GEMM tiling */
ob_status_t ob_gemm_set_tile_limits(ob_context_t* ctx,
                                    uint32_t      max_m,
                                    uint32_t      max_k,
                                    uint32_t      max_n);

/*
 * Tiled INT8 matrix multiplication with INT32 accumulation:
 *   C[m, n] = bias[n] + sum_{k=0..K-1} (A[m, k] - zp_a) * (B[n, k] - zp_b)
 *
 * Layouts:
 *   a_mk   : [m_dim, k_dim] row-major int8
 *   b_nk   : [n_dim, k_dim] row-major int8 (transposed B, matching TFLite FC / ggml)
 *   bias_n : optional [n_dim] int32 (may be NULL)
 *   c_mn   : [m_dim, n_dim] row-major int32
 *
 * Automatically tiles arbitrary LLM shapes (e.g. M=1..64, K=2048..4096, N=2048..11008)
 * across the attached buenos .dgc0 emitter + graph cache, falling back to an exact
 * tiled CPU reference when mock_mode=1 or no emitter is registered.
 */
ob_status_t ob_gemm_matmul_int8(ob_context_t*      ctx,
                                const int8_t*      a_mk,
                                const int8_t*      b_nk,
                                const int32_t*     bias_n,
                                int32_t*           c_mn,
                                uint32_t           m_dim,
                                uint32_t           k_dim,
                                uint32_t           n_dim,
                                int32_t            zp_a,
                                int32_t            zp_b,
                                ob_invoke_stats_t* out_stats);

/*
 * Tiled FP32 x Q8_0 matrix multiplication for llama.cpp (ggml-edgetpu):
 *   C[m, n] = sum_{k=0..K-1} A[m, k] * dequant_q8_0(B[n, k])
 *
 * Layouts:
 *   a_mk      : [m_dim, k_dim] row-major float32
 *   b_nk_q8_0 : [n_dim, k_dim / 32] contiguous ob_block_q8_0_t blocks (k_dim % 32 == 0)
 *   c_mn      : [m_dim, n_dim] row-major float32
 */
ob_status_t ob_gemm_matmul_f32_q8_0(ob_context_t*      ctx,
                                    const float*       a_mk,
                                    const void*        b_nk_q8_0,
                                    float*             c_mn,
                                    uint32_t           m_dim,
                                    uint32_t           k_dim,
                                    uint32_t           n_dim,
                                    ob_invoke_stats_t* out_stats);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENBUENOS_H_ */
