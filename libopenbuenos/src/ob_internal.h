#ifndef OPENBUENOS_INTERNAL_H_
#define OPENBUENOS_INTERNAL_H_

#include "openbuenos.h"
#include "litebuf_vii_wire.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <list>
#include <mutex>
#include <pthread.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ob {

constexpr uint32_t kDefaultGraphCacheCapacity     = 512u;
constexpr size_t   kDefaultBufferPoolMaxBytes     = 64u * 1024u * 1024u;
constexpr size_t   kDefaultBufferPoolMaxCount     = 256u;
constexpr size_t   kDefaultWeightCacheMaxEntries  = 4096u;
constexpr uint32_t kDefaultTimeoutMs              = 5000u;
constexpr uint32_t kDefaultTileMaxM           = 64u;
constexpr uint32_t kDefaultTileMaxK           = 2048u;
constexpr uint32_t kDefaultTileMaxN           = 2048u;
constexpr uint32_t kDefaultWorkerThreads          = 1u;
constexpr uint32_t kDefaultMaxCachedOutlierCols   = 160u;

uint64_t monotonic_now_ns();
// Pin calling thread to non-LITTLE performance cores (Linux/Android cpufreq sysfs; no-op elsewhere).
void pin_thread_to_perf_cores();

// Lock-free 1-helper SPSC async dequant worker (zero locks/futexes during active pp512,
// bounded ~1.5 ms idle spin -> 0% CPU during tg64/idle, pinned to non-LITTLE perf cores).
struct AsyncDequantWorker {
  using JobFn = void (*)(void* arg);
  static constexpr uint32_t kRingCap = OB_MAX_CREDITS;
  static constexpr uint32_t kRingMask = kRingCap - 1u;

  struct Slot {
    JobFn fn = nullptr;
    void* arg = nullptr;
  };

  alignas(64) std::atomic<uint64_t> prod_idx{0u};
  alignas(64) std::atomic<uint64_t> cons_idx{0u};
  alignas(64) std::atomic<bool> helper_sleeping{false};
  alignas(64) std::atomic<bool> main_waiting{false};
  std::atomic<bool> stopping{false};

  std::array<Slot, kRingCap> ring{};
  bool started = false;
  pthread_t thread{};
  pthread_mutex_t wake_mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t wake_cv = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t done_mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t done_cv = PTHREAD_COND_INITIALIZER;

  AsyncDequantWorker() = default;
  ~AsyncDequantWorker();
  AsyncDequantWorker(const AsyncDequantWorker&) = delete;
  AsyncDequantWorker& operator=(const AsyncDequantWorker&) = delete;

  bool ensure_started();
  uint64_t push(JobFn fn, void* arg);
  uint64_t completed_count() const { return cons_idx.load(std::memory_order_acquire); }
  uint64_t pushed_count() const { return prod_idx.load(std::memory_order_relaxed); }
  void wait_until(uint64_t target_seq);
  void shutdown();
};

// Persistent pthread worker pool (zero extra Android .so dependencies; libc.so pthread only).
struct WorkerPool {
  using TaskFn = void (*)(void* arg, uint32_t start, uint32_t end, uint32_t tid);

  uint32_t num_threads = 1u;
  bool started = false;
  bool stopping = false;
  std::mutex dispatch_mu;
  std::vector<pthread_t> threads;
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t work_cv = PTHREAD_COND_INITIALIZER;
  pthread_cond_t done_cv = PTHREAD_COND_INITIALIZER;

  uint64_t job_gen = 0u;
  TaskFn cur_fn = nullptr;
  void* cur_arg = nullptr;
  uint32_t cur_total = 0u;
  uint32_t cur_step = 1u;
  uint32_t cur_active_workers = 0u;
  std::atomic<uint32_t> next_item{0u};
  std::atomic<uint32_t> workers_remaining{0u};

  WorkerPool() = default;
  ~WorkerPool();
  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  void init(uint32_t requested_threads);
  void shutdown();
  void parallel_for(uint32_t total, uint32_t min_chunk, TaskFn fn, void* arg);

  template <typename Fn>
  void parallel_for_fn(uint32_t total, uint32_t min_chunk, Fn&& fn) {
    if (total == 0u) return;
    if (num_threads <= 1u || total <= min_chunk) {
      fn(0u, total, 0u);
      return;
    }
    struct Trampoline {
      Fn* f;
      static void run(void* p, uint32_t s, uint32_t e, uint32_t tid) {
        (*static_cast<Trampoline*>(p)->f)(s, e, tid);
      }
    };
    Trampoline tr{&fn};
    parallel_for(total, min_chunk, &Trampoline::run, &tr);
  }
};

// Standalone SHA-256 (zero external crypto dependencies)
struct Sha256Digest {
  std::array<uint8_t, 32> bytes{};
  bool operator==(const Sha256Digest& o) const { return bytes == o.bytes; }
};

void sha256_compute(const void* data, size_t len, uint8_t out[32]);

struct CacheKey {
  bool is_u64 = false;
  uint64_t u64_key = 0;
  Sha256Digest sha256{};

  bool operator==(const CacheKey& o) const {
    if (is_u64 != o.is_u64) return false;
    return is_u64 ? (u64_key == o.u64_key) : (sha256 == o.sha256);
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& k) const noexcept {
    if (k.is_u64) {
      uint64_t x = k.u64_key ^ 0x9e3779b97f4a7c15ULL;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      return static_cast<size_t>(x ^ (x >> 31));
    }
    uint64_t h = 0;
    std::memcpy(&h, k.sha256.bytes.data(), sizeof(h));
    return static_cast<size_t>(h);
  }
};

struct Dgc0Slice {
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct ParsedDgc0 {
  std::vector<uint64_t> input_sizes;
  std::vector<uint64_t> output_sizes;
  uint64_t dive_scratch_size = 0;
  uint64_t compiler_cl = 0;
  int32_t dive_heap_size = 0;
  int32_t exec_field_29 = 0;
  uint32_t exec_field_14 = 0;
  uint8_t cluster_id = LITEBUF_DEFAULT_CLUSTER_ID;
  uint8_t exec_field_31 = 0;
  uint8_t exec_field_34 = 0;
  uint8_t subgraph_flag_18 = 0;
  Dgc0Slice ext_dive_vm_ir{};
  Dgc0Slice ext_dive_image{};
  bool has_params = false;
  Dgc0Slice ext_params{};
};

ob_status_t parse_dgc0_buffer(const uint8_t* data, size_t size,
                              ParsedDgc0* out, std::string* err);

void fill_dgc0_info(const ParsedDgc0& parsed, uint32_t model_handle,
                    bool cache_hit, ob_dgc0_info_t* out);

struct PooledSlab {
  int dmabuf_fd = -1;
  uint8_t* data = nullptr;
  size_t alloc_size = 0;
  uint64_t iova = 0;
  uint32_t map_flags = 0;
  uint32_t die_index = 0xffffffffu;
  bool iova_mapped = false;
};

struct CachedGraph {
  CacheKey key{};
  ob_context* ctx = nullptr;
  ParsedDgc0 parsed{};
  uint32_t model_handle = 0;
  ob_buffer* scratch_buf = nullptr;
  ob_buffer* image_buf = nullptr;
  ob_buffer* vm_ir_buf = nullptr;
  ob_buffer* params_buf = nullptr;
  ob_buffer* reg_cmd_buf = nullptr;
  bool owns_image_buf = true;
  bool owns_params_buf = true;
  uint32_t ref_count = 0;
  std::list<CachedGraph*>::iterator lru_it;
  bool in_cache = false;
};

struct CreditSlot {
  enum State {
    FREE = 0,
    IN_FLIGHT = 1,
    COMPLETED = 2,
    FAILED = 3
  };
  State state = FREE;
  bool abandoned = false;
  ob_exec_handle_t handle = 0;
  uint32_t seq = 0;
  CachedGraph* graph_entry = nullptr;
  ob_buffer* exec_cmd_buf = nullptr;
  uint64_t submit_time_ns = 0;
  uint64_t complete_time_ns = 0;
  uint64_t hw_execution_us = 0;
  uint16_t vii_resp_code = 0;
  ob_status_t error_status = OB_OK;
  std::string error_msg;
  std::vector<ob_buffer*> retired_bufs;
};

// Device & VII mailbox helpers (implemented in ob_device.cpp)
ob_status_t device_open(ob_context* ctx);
void        device_close(ob_context* ctx);

ob_status_t device_submit_control(ob_context* ctx,
                                  const void* outer_buf,
                                  size_t outer_len,
                                  struct litebuf_vii_parsed_response* parsed_out,
                                  uint16_t* resp_code_out,
                                  uint32_t timeout_ms);

ob_status_t device_submit_execute_async(ob_context* ctx,
                                        CachedGraph* entry,
                                        ob_buffer* const* inputs,
                                        uint32_t num_inputs,
                                        ob_buffer* const* outputs,
                                        uint32_t num_outputs,
                                        ob_exec_handle_t* out_handle,
                                        const ob_buffer* override_params = nullptr);

ob_status_t device_wait_handle(ob_context* ctx,
                               ob_exec_handle_t handle,
                               uint32_t timeout_ms,
                               ob_invoke_stats_t* out_stats);

// Non-blocking completion check + harvest for overlapped pipeline drain.
// Returns true if `handle` is finished (or errored/not-found), setting `*out_status` and `*out_stats`.
// Returns false immediately without blocking if `handle` is still in-flight on TPU.
bool        device_try_wait_handle(ob_context* ctx,
                                   ob_exec_handle_t handle,
                                   ob_status_t* out_status,
                                   ob_invoke_stats_t* out_stats);

ob_status_t device_wait_all(ob_context* ctx, uint32_t timeout_ms);
bool        device_attach_inflight_buffers(ob_context* ctx,
                                           ob_exec_handle_t handle,
                                           std::initializer_list<ob_buffer*> bufs);

// Buffer pool helpers (implemented in ob_buffer_pool.cpp)
size_t buffer_slab_size(size_t requested);
void buffer_retain(ob_buffer* buf);
void buffer_destroy_slab(ob_context* ctx, PooledSlab* slab);
void buffer_pool_clear(ob_context* ctx);
ob_status_t buffer_sync_begin_read(ob_buffer* buf);
ob_status_t buffer_sync_end_read(ob_buffer* buf);
ob_status_t buffer_sync_begin_write(ob_buffer* buf);
ob_status_t buffer_sync_end_write(ob_buffer* buf);
void graph_cache_clear(ob_context* ctx);
void shared_image_clear(ob_context* ctx);
void gemm_release_pipeline_buffers(ob_context* ctx);
void graph_entry_unref_locked(ob_context* ctx, CachedGraph* entry);

// Fast-path cache lookup by explicit u64 key without setting last_error on miss.
bool graph_cache_lookup_u64(ob_context* ctx, uint64_t cache_key, ob_graph_t** out_graph);

// Zero-alloc fast-path lookup returning CachedGraph* directly (+1 ref_count held by caller).
bool graph_cache_lookup_entry_u64(ob_context* ctx, uint64_t cache_key, CachedGraph** out_entry);

// Internal graph registration returning CachedGraph* (+1 ref_count), with optional
// shared read-only 8 MiB zero-filled dummy params_buf for Probe-A weight-agnostic graphs.
ob_status_t graph_register_dgc0_internal(ob_context* ctx,
                                         const void* dgc0_bytes,
                                         size_t dgc0_size,
                                         uint64_t cache_key,
                                         bool share_zero_params,
                                         CachedGraph** out_entry,
                                         bool* out_cache_hit);

// Compute Android LMKD-safe ext[2] weight-cache byte budget (ob_gemm.cpp).
size_t compute_weight_cache_budget_bytes(ob_context* ctx);

// Submit async Execute with an optional runtime param_slice DMA-BUF override
// (Probe-A weight-agnostic graph reuse: single model_handle per (M,K,N,D)).
ob_status_t graph_submit_async_with_params(ob_graph_t* graph,
                                           ob_buffer* const* inputs,
                                           uint32_t num_inputs,
                                           ob_buffer* const* outputs,
                                           uint32_t num_outputs,
                                           const ob_buffer* override_params,
                                           ob_exec_handle_t* out_handle);

// 128-bit weight tile key (eliminates 64-bit collision risk while holding O(1) lookup).
struct WeightTileKey {
  uint64_t h1 = 0;
  uint64_t h2 = 0;
  uint64_t bytes = 0;

  bool operator==(const WeightTileKey& o) const {
    return h1 == o.h1 && h2 == o.h2 && bytes == o.bytes;
  }
};

struct WeightTileKeyHash {
  size_t operator()(const WeightTileKey& k) const noexcept {
    uint64_t x = k.h1 ^ (k.h2 * 0x9e3779b97f4a7c15ULL) ^ (k.bytes * 0xbf58476d1ce4e5b9ULL);
    x ^= (x >> 30);
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= (x >> 27);
    return static_cast<size_t>(x);
  }
};

// Cached packed ext[2] weight tile (pinned in DMA-BUF, independent of model_handle).
struct CachedWeightTile {
  WeightTileKey key{};
  uint32_t scale_d = 1;
  uint32_t hw_k = 0;
  uint32_t hw_n = 0;
  std::vector<float> row_s_b;
  ob_buffer* ext2_buf = nullptr;
  size_t ext2_bytes = 0;
  // Lazy exact FP32 outlier-column cache (keyed by local k_off in [0, tk), length tn_total).
  // Eliminates strided Q8_0/Q4_0 DRAM walks for recurring LLM activation sink channels.
  std::unordered_map<uint32_t, std::vector<float>> outlier_cols;
  size_t outlier_cols_bytes = 0;
  // Chained SwiGLU chunk metadata (used when this tile stores packed [Wg|Wu|Wd_eff]).
  float swiglu_s_wg = 0.0f;
  float swiglu_l2_gate_max = 0.0f;
  float swiglu_l2_gate_mean = 0.0f;
  float swiglu_l2_up_max = 0.0f;
  float swiglu_l2_up_mean = 0.0f;
  float swiglu_l2_down_max = 0.0f;
  float swiglu_dc_gate_max = 0.0f;
  float swiglu_dc_up_max = 0.0f;
  float swiglu_l1_gate_max = 0.0f;
  float swiglu_linf_gate_max = 0.0f;
  float swiglu_l1_up_max = 0.0f;
  float swiglu_linf_up_max = 0.0f;
  uint32_t swiglu_d_gate = 0;
  uint32_t swiglu_d_up = 0;
  uint32_t swiglu_d_mul = 0;
  uint32_t swiglu_d_down = 0;
  std::list<CachedWeightTile*>::iterator lru_it;
  uint64_t in_use = 0;
  bool in_cache = false;
  bool rejected_fusion = false; // O(1) cached split decision when fused D spread > 1.5x
  bool rejected_swiglu = false; // O(1) cached fallback when W_gate row-scale spread is extreme
  bool swiglu_norms_only = false; // O(1) cached weight norms (0 B DMA) when activation guard rejects before packing
};

// Built-in nanosecond stage profiler accumulators (zero clock_gettime when profile_enabled=false).
struct ProfileCounters {
  uint64_t quant_ns = 0;
  uint64_t hash_ns = 0;
  uint64_t weight_pack_ns = 0;
  uint64_t dma_sync_ns = 0;
  uint64_t submit_ns = 0;
  uint64_t wait_ns = 0;
  uint64_t dequant_ns = 0;
  uint64_t hw_us = 0;
  uint64_t total_wall_ns = 0;
  uint64_t calls = 0;
  uint64_t tiles_dispatched = 0;
  uint64_t horiz_fusions = 0;
  uint64_t horiz_jobs_saved = 0;
  uint64_t sat_retries = 0;
  uint64_t spin_hits = 0;
  uint64_t poll_waits = 0;
  uint64_t swiglu_calls = 0;
  uint64_t swiglu_chunks_chained = 0;
  uint64_t swiglu_fallbacks = 0;
};

// Token-invariant weight-tile output scale ratio D and activation outlier/RMS helpers (ob_gemm.cpp).
uint32_t quant_scale_D_grid(double raw_d);
uint32_t compute_tile_scale_D(const int8_t* b_tile, uint32_t tn, uint32_t tk,
                              uint32_t stride_k, int32_t zp_a = 0);
void quantize_activation_row_split(const float* a_row,
                                   uint32_t tk,
                                   int8_t* aq_row,
                                   float* out_s_a,
                                   std::vector<std::pair<uint32_t, float>>* out_outliers);
uint32_t compute_activation_rms_shift(const int8_t* aq_tile,
                                      uint32_t tm,
                                      uint32_t tk,
                                      uint32_t hw_k);
bool tile_output_is_saturated(const int8_t* out_data, uint32_t tm, uint32_t tn, uint32_t hw_n);
void dequant_accum_row_f32(float* c_row, const int8_t* o_row,
                           const float* sb_ptr, uint32_t tn, float sa_d);
void dequant_outliers_row_f32(float* c_row,
                              const int8_t* o_row,
                              const float* sb_ptr,
                              uint32_t tn,
                              float sa_d,
                              bool store_first,
                              const float* const* outlier_cols,
                              const float* outlier_vals,
                              uint32_t num_outliers);
void apply_activation_inplace_f32(float* c_mn, size_t count, uint32_t act);

// Clean-room C++17 Tensor G5 GEMM emitter internals (implemented in ob_gemm_emitter.cpp)
struct GemmSynthResult {
  std::array<std::vector<uint8_t>, 7> chunks;
  std::vector<uint8_t> ext0_dive;
  std::vector<uint8_t> ext2_params;
  uint32_t exec_field_14 = 0;
  uint32_t dive_scratch_size = 0;
};

ob_status_t synthesize_gemm_parts(const ob_gemm_desc_t& desc,
                                  GemmSynthResult* out,
                                  std::string* err);

ob_status_t build_dgc0_from_parts(const ob_gemm_desc_t& desc,
                                  const GemmSynthResult& parts,
                                  const uint8_t* dive_interp,
                                  size_t dive_interp_size,
                                  std::vector<uint8_t>* out_dgc0,
                                  std::string* err);

// Emit a weight-agnostic 2-input 0-param .dgc0 where packed ext[2] weights are
// passed dynamically at Execute time via inputs[1] (zero RegisterModel per weight).
ob_status_t build_dgc0_dyn_weights(const ob_gemm_desc_t& desc,
                                   const uint8_t* dive_interp,
                                   size_t dive_interp_size,
                                   std::vector<uint8_t>* out_dgc0,
                                   uint32_t* out_ext2_bytes,
                                   std::string* err);

// Pack weights (+ optional bias) into exact buenos_a0 ext[2] layout.
std::vector<uint8_t> pack_ext2_for_tile(uint32_t N, uint32_t K,
                                        const int8_t* weights_nk,
                                        const int32_t* bias_n);
size_t pack_ext2_size_for_tile(uint32_t N, uint32_t K, bool include_bias);
void pack_ext2_for_tile_into(uint8_t* dst, uint32_t N, uint32_t K,
                             const int8_t* weights_nk,
                             const int32_t* bias_n,
                             bool include_bias,
                             WorkerPool* pool = nullptr);

// Pack weights (+ optional bias) into exact buenos_a0 M=64 large-tile ext[2] layout
// (8-channel groups: 128 B padded bias header + K/16 blocks of 8x16 weights ^ 0x80).
std::vector<uint8_t> pack_ext2_for_lt_64x(uint32_t N, uint32_t K,
                                          const int8_t* weights_nk,
                                          const int32_t* bias_n);
size_t pack_ext2_size_for_lt_64x(uint32_t N, uint32_t K);
void pack_ext2_for_lt_64x_into(uint8_t* dst, uint32_t N, uint32_t K,
                               const int8_t* weights_nk,
                               const int32_t* bias_n,
                               WorkerPool* pool = nullptr);

// Mode-A zero-overhead (+0 B) in-place chunk[5] 0x640 + HDR_DATA SiLU/GELU fusion.
// Allows decoupled (s_mid, zp_mid) -> (s_out, zp_out) for arbitrary quantization.
ob_status_t apply_mode_a_fusion_c5(std::vector<uint8_t>* chunk5,
                                   uint32_t activation,
                                   float s_mid,
                                   int32_t zp_mid,
                                   float s_out,
                                   int32_t zp_out,
                                   float* out_max_lsb_err,
                                   std::string* err);

// Exact FP32-RNE Horner simulator for chunk[5] Mode-A fused 0x640 + HDR_DATA.
ob_status_t simulate_c5_mode_a_fused(const std::vector<uint8_t>& chunk5,
                                     const uint8_t* u_mid,
                                     size_t count,
                                     int8_t* out_q8,
                                     std::string* err);

// Check whether a dimension is a proven M=1 large-tile dimension ({512,576,1024,1536,2048}).
bool is_lt_1x_dim(uint32_t d);

// Check whether (M, K, N) is a proven M=64 batched prefill large-tile shape
// (M==64, K in {512,576,1024,1536}, N in [128..1536] step 32, K+N <= 3072).
bool is_lt_64x_shape(uint32_t M, uint32_t K, uint32_t N);

// Clean-room 7-chunk per-tensor W8A16(FP16) (full FP32 accumulator, scale=1.0 default)
// weight packer + .dgc0 emitter (ob_gemm_emitter.cpp) & NEON scale epilogue (ob_gemm.cpp).
bool is_w8a16_pt_1x_shape(uint32_t M, uint32_t K, uint32_t N);
size_t pack_w8a16_ext2_size(uint32_t N, uint32_t K);
void pack_w8a16_ext2_into(uint8_t* dst, uint32_t N, uint32_t K, const int8_t* weights_nk);
std::vector<uint8_t> pack_w8a16_ext2(uint32_t N, uint32_t K, const int8_t* weights_nk);
ob_status_t synthesize_w8a16_pt_1x_parts(uint32_t K, uint32_t N, float pt_scale,
                                         const int8_t* weights_nk,
                                         GemmSynthResult* out, std::string* err);
ob_status_t build_dgc0_w8a16_pt_1x(uint32_t K, uint32_t N, float pt_scale,
                                   const int8_t* weights_nk,
                                   const uint8_t* dive_interp,
                                   size_t dive_interp_size,
                                   std::vector<uint8_t>* out_dgc0,
                                   std::string* err);
void w8a16_pt_epilogue_f32(float* c_row,
                           const float* y_raw,
                           const float* sb_ptr,
                           uint32_t tn,
                           float prescale_inv,
                           bool store_first);
float round_f32_to_fp16_f32(float x);

// Clean-room C++17 11-chunk zero-DRAM in-SRAM chained SwiGLU emitter (ob_gemm_emitter.cpp)
struct SwigluSynthResult {
  std::array<std::vector<uint8_t>, 11> chunks;
  std::vector<uint8_t> ext0_dive;
  std::vector<uint8_t> ext2_params;
  uint32_t exec_field_14 = 0;
  uint32_t dive_scratch_size = 0;
};

bool is_swiglu_shape(uint32_t M, uint32_t K, uint32_t H, uint32_t N);

std::vector<uint8_t> pack_ext2_for_swiglu(uint32_t M, uint32_t K, uint32_t H, uint32_t N,
                                          const int8_t* wg_hk, const int32_t* bg_h,
                                          const int8_t* wu_hk, const int32_t* bu_h,
                                          const int8_t* wd_nh, const int32_t* bd_n);
size_t pack_ext2_size_for_swiglu(uint32_t M, uint32_t K, uint32_t H, uint32_t N);
void pack_ext2_for_swiglu_into(uint8_t* dst,
                               uint32_t M, uint32_t K, uint32_t H, uint32_t N,
                               const int8_t* wg_hk, const int32_t* bg_h,
                               const int8_t* wu_hk, const int32_t* bu_h,
                               const int8_t* wd_nh, const int32_t* bd_n,
                               WorkerPool* pool = nullptr);

ob_status_t synthesize_swiglu_parts(const ob_swiglu_int8_desc_t& desc,
                                    SwigluSynthResult* out,
                                    std::string* err);

ob_status_t build_swiglu_dgc0(const ob_swiglu_int8_desc_t& desc,
                              const uint8_t* dive_interp,
                              size_t dive_interp_size,
                              std::vector<uint8_t>* out_dgc0,
                              SwigluSynthResult* out_parts,
                              std::string* err);

// Validate SwiGLU quantization scales and compute FP32 hardware multipliers using double
// intermediates to avoid intermediate overflow. Returns false if any scale, multiplier,
// or scaled SiLU spline coefficient is non-finite or non-positive.
bool compute_swiglu_hw_multipliers(const ob_swiglu_int8_desc_t& desc,
                                   float* out_m_gate,
                                   float* out_m_up,
                                   float* out_m_mul,
                                   float* out_m_down);

// Exact FP32-RNE Horner SwiGLU hardware simulator (matches rango silicon max_abs_err=0).
void simulate_swiglu_hw_exact(const int8_t* x_mk,
                              const ob_swiglu_int8_desc_t& desc,
                              int8_t* y_mn);

// Evaluate exact TPU piecewise-Horner SiLU on quantized gate code q_g in [-128, 127],
// returning quantized INT8 code q_silu in [-128, 127] matching simulate_swiglu_hw_exact.
int32_t eval_swiglu_silu_q(int32_t q_g, float s_gate);
void build_swiglu_silu_lut(float s_gate, int8_t out_lut[256]);

// RMS-calibrated 3-stage divider selection for chained SwiGLU (ob_gemm.cpp).
void compute_swiglu_stage_dividers(uint32_t m_dim,
                                   float rms_a,
                                   float l2_g_max, float l2_g_mean,
                                   float l2_u_max, float l2_u_mean,
                                   float l2_d_max,
                                   uint32_t* out_d_gate,
                                   uint32_t* out_d_up,
                                   uint32_t* out_d_mul,
                                   uint32_t* out_d_down);

ob_status_t builtin_gemm_emitter_cb(const ob_gemm_desc_t* desc,
                                    uint8_t* out_buf,
                                    size_t out_cap,
                                    size_t* out_size,
                                    void* user_data);

struct ImmutableWeightRegion {
  uintptr_t begin = 0u;
  uintptr_t end = 0u;
  uint64_t version = 0u;
  uint64_t generation = 0u;
};

void set_last_error(ob_context* ctx, const std::string& msg);

}  // namespace ob

struct ob_buffer {
  ob_context* ctx = nullptr;
  ob::PooledSlab slab{};
  size_t requested_size = 0;
  uint32_t flags = 0;
  std::atomic<uint32_t> refcnt{1u};
};

struct ob_graph {
  ob::CachedGraph* entry = nullptr;
  bool was_cache_hit = false;
};

struct ob_context {
  ob_config_t cfg{};
  std::string device_path;
  std::string dma_heap_path;

  bool mock_mode = false;
  int tpu_fd = -1;
  int dma_heap_fd = -1;
  int resp_efd = -1;
  int err_efd = -1;
  bool wake_locked = false;
  // Firmware VII Execute +0x1c fw_priority & Litebuf AcquireWakeLock (union_type=4) state
  uint8_t fw_priority = 0x01u;
  bool fw_wakelock_enabled = true;
  bool fw_wakelock_held = false;
  uint8_t fw_wakelock_core = 0u;
  uint8_t fw_wakelock_mem = 0u;
  uint8_t fw_wakelock_bitset = 0u;
  // Cached runtime env knobs (read once on ob_context_create to eliminate hot-path getenv scans)
  bool profile_enabled = false;
  bool verbose_enabled = false;
  bool strict_weight_hash = false;
  bool large_tiles_enabled = true;
  bool batched_large_tiles_enabled = true;
  bool horiz_fusion_enabled = true;
  bool swiglu_chain_enabled = true;
  bool swiglu_fast_chain_enabled = false;
  bool swiglu_prefill_enabled = false;
  float swiglu_max_crest = 16.0f;
  uint32_t param_swap_slots = 1u;
  uint32_t spin_us = 450u; // Adaptive spin window (us) before poll(resp_efd) when ema_hw_us < 600
  uint32_t num_threads = ob::kDefaultWorkerThreads; // OPENBUENOS_THREADS (default 1, opt-in >1)
  bool async_dequant_enabled = true; // OPENBUENOS_ASYNC_DEQUANT (default 1, opt-out 0)
  uint32_t w8a16_mode = 0u; // OPENBUENOS_W8A16: 0=off, 1=opt-in supported M=1
  uint32_t max_cached_outlier_cols = ob::kDefaultMaxCachedOutlierCols; // OPENBUENOS_MAX_OUTLIER_COLS (default 160)
  uint64_t ema_hw_us = 0u; // Exponential moving average of firmware hw_execution_us
  uint32_t credit_size = OB_MAX_CREDITS;
  uint32_t fw_major = 0;
  uint32_t fw_minor = 0;
  uint32_t fw_vii = 0;
  uint32_t fw_kci = 0;

  // Monotonic generators
  uint32_t next_seq = 0;
  uint64_t next_exec_handle = 1;
  uint32_t next_mock_model_handle = 0x1000u;
  uint64_t next_mock_iova = 0x100000000ULL; // 4 GiB base (36-bit IOVA range)

  // Buffer slab/pool
  std::mutex pool_mu;
  size_t pool_max_bytes = ob::kDefaultBufferPoolMaxBytes;
  size_t pool_max_count = ob::kDefaultBufferPoolMaxCount;
  size_t pool_current_bytes = 0;
  size_t pool_current_count = 0;
  uint64_t pool_hits = 0;
  uint64_t pool_misses = 0;
  // Ordered LRU of free slabs + bucket index by alloc_size
  std::list<ob::PooledSlab> pool_lru;
  std::unordered_map<size_t, std::vector<std::list<ob::PooledSlab>::iterator>> pool_buckets;

  // Persistent graph LRU cache
  std::mutex cache_mu;
  uint32_t cache_capacity = ob::kDefaultGraphCacheCapacity;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t cache_evictions = 0;
  std::list<ob::CachedGraph*> cache_lru;
  std::unordered_map<ob::CacheKey, ob::CachedGraph*, ob::CacheKeyHash> cache_map;

  // Shared read-only RV64 DIVE interpreter DMA-BUF (deduplicated by ext[1] SHA-256)
  bool has_shared_image = false;
  ob::Sha256Digest shared_image_sha256{};
  size_t shared_image_size = 0;
  ob_buffer* shared_image_buf = nullptr;
  // Shared 8 MiB zero-filled dummy ext[2] DMA-BUF for Probe-A weight-agnostic graphs
  // (eliminates 3-5 MiB of unused dummy params DMA-BUF per cached SwiGLU/GEMM model_handle).
  ob_buffer* shared_zero_params_buf = nullptr;

  // 16-credit async VII execution ring
  std::mutex vii_mu;
  std::array<ob::CreditSlot, OB_MAX_CREDITS> credits{};
  uint32_t inflight_count = 0;

  // Pluggable GEMM emitter, builtin emitter state & reusable async pipeline buffers
  std::mutex gemm_mu;
  ob_gemm_emitter_fn gemm_emitter = nullptr;
  void* gemm_emitter_user_data = nullptr;
  bool builtin_emitter_active = false;
  std::vector<uint8_t> builtin_dive_interp;
  uint32_t tile_max_m = ob::kDefaultTileMaxM;
  uint32_t tile_max_k = ob::kDefaultTileMaxK;
  uint32_t tile_max_n = ob::kDefaultTileMaxN;
  std::vector<uint8_t> emitter_scratch;
  std::array<ob_buffer*, OB_MAX_CREDITS> gemm_in_bufs{};
  std::array<ob_buffer*, OB_MAX_CREDITS> gemm_out_bufs{};
  std::array<ob_buffer*, OB_MAX_CREDITS> gemm_in_bufs_4k{};
  std::array<ob_buffer*, OB_MAX_CREDITS> gemm_out_bufs_4k{};

  // Persistent packed ext[2] weight tile DMA-BUF LRU cache (Probe-A dynamic param_slice)
  std::unordered_map<ob::WeightTileKey, ob::CachedWeightTile*, ob::WeightTileKeyHash> weight_cache_map;
  std::list<ob::CachedWeightTile*> weight_cache_lru;
  size_t weight_cache_bytes = 0;
  size_t weight_cache_peak_bytes = 0;
  size_t weight_cache_max_bytes = 0;   // lazily initialized (min(2048 MiB, MemAvailable/3) on Android)
  size_t weight_cache_default_max_entries = ob::kDefaultWeightCacheMaxEntries;
  size_t weight_cache_max_entries = 0; // FD-safe cap (default 4096, env OPENBUENOS_WEIGHT_CACHE_ENTRIES)
  size_t outlier_cache_bytes = 0;      // Separate tracked total of cached FP32 outlier columns
  std::vector<ob::ImmutableWeightRegion> immutable_weight_regions;
  uint64_t next_immutable_region_gen = 1u;
  size_t simulated_mem_available_bytes = 0; // Optional test override for LMKD budget verification
  bool simulated_mem_available_set = false; // Allows explicitly simulating MemAvailable == 0 kB
  uint64_t weight_cache_hits = 0;
  uint64_t weight_cache_misses = 0;
  uint64_t weight_cache_evictions = 0;

  // Persistent pthread worker pool (opt-in via OPENBUENOS_THREADS>1) +
  // lock-free 1-helper SPSC async dequant worker (OPENBUENOS_ASYNC_DEQUANT=1).
  ob::WorkerPool worker_pool;
  ob::AsyncDequantWorker async_dequant;

  // Reusable GEMM host scratch buffers (eliminates per-call malloc/free in hot loop)
  std::vector<int8_t> gemm_aq_scratch;
  std::vector<int8_t> gemm_bq_scratch;
  std::vector<int8_t> gemm_out_scratch;
  std::vector<float> gemm_wrow_scratch;
  std::vector<float> gemm_wcol_scratch;
  std::vector<float> gemm_sa_scratch;
  std::vector<std::vector<std::pair<uint32_t, float>>> gemm_outliers_scratch;

  // Persistent M7A SwiGLU fallback scratch (eliminates per-layer 36.7 MiB mmap/page-fault/munmap churn)
  std::mutex swiglu_fb_mu;
  std::vector<float> swiglu_gate_scratch;
  std::vector<float> swiglu_up_scratch;

  // Stage profiler counters
  ob::ProfileCounters prof{};

  // Diagnostics
  mutable std::mutex err_mu;
  std::string last_error;
};

#endif  // OPENBUENOS_INTERNAL_H_
