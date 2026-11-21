#ifndef OPENBUENOS_INTERNAL_H_
#define OPENBUENOS_INTERNAL_H_

#include "openbuenos.h"
#include "litebuf_vii_wire.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ob {

constexpr uint32_t kDefaultGraphCacheCapacity = 64u;
constexpr size_t   kDefaultBufferPoolMaxBytes = 64u * 1024u * 1024u;
constexpr uint32_t kDefaultTimeoutMs          = 5000u;
constexpr uint32_t kDefaultTileMaxM           = 64u;
constexpr uint32_t kDefaultTileMaxK           = 512u;
constexpr uint32_t kDefaultTileMaxN           = 256u;

uint64_t monotonic_now_ns();

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

ob_status_t device_wait_all(ob_context* ctx, uint32_t timeout_ms);

// Buffer pool helpers (implemented in ob_buffer_pool.cpp)
void buffer_destroy_slab(ob_context* ctx, PooledSlab* slab);
void buffer_pool_clear(ob_context* ctx);
void graph_cache_clear(ob_context* ctx);
void shared_image_clear(ob_context* ctx);
void gemm_release_pipeline_buffers(ob_context* ctx);
void graph_entry_unref_locked(ob_context* ctx, CachedGraph* entry);

// Fast-path cache lookup by explicit u64 key without setting last_error on miss.
bool graph_cache_lookup_u64(ob_context* ctx, uint64_t cache_key, ob_graph_t** out_graph);

// Submit async Execute with an optional runtime param_slice DMA-BUF override
// (Probe-A weight-agnostic graph reuse: single model_handle per (M,K,N,D)).
ob_status_t graph_submit_async_with_params(ob_graph_t* graph,
                                           ob_buffer* const* inputs,
                                           uint32_t num_inputs,
                                           ob_buffer* const* outputs,
                                           uint32_t num_outputs,
                                           const ob_buffer* override_params,
                                           ob_exec_handle_t* out_handle);

// Cached packed ext[2] weight tile (pinned in DMA-BUF, independent of model_handle).
struct CachedWeightTile {
  uint64_t key = 0;
  uint32_t scale_d = 1;
  uint32_t hw_k = 0;
  uint32_t hw_n = 0;
  std::vector<float> row_s_b;
  ob_buffer* ext2_buf = nullptr;
  size_t ext2_bytes = 0;
  std::list<CachedWeightTile*>::iterator lru_it;
  uint64_t in_use = 0;
};

// Token-invariant weight-tile output scale ratio D (implemented in ob_gemm.cpp).
uint32_t compute_tile_scale_D(const int8_t* b_tile, uint32_t tn, uint32_t tk,
                              uint32_t stride_k, int32_t zp_a = 0);

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

ob_status_t builtin_gemm_emitter_cb(const ob_gemm_desc_t* desc,
                                    uint8_t* out_buf,
                                    size_t out_cap,
                                    size_t* out_size,
                                    void* user_data);

void set_last_error(ob_context* ctx, const std::string& msg);

}  // namespace ob

struct ob_buffer {
  ob_context* ctx = nullptr;
  ob::PooledSlab slab{};
  size_t requested_size = 0;
  uint32_t flags = 0;
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
  int resp_efd = -1;
  int err_efd = -1;
  bool wake_locked = false;
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

  // Persistent packed ext[2] weight tile DMA-BUF LRU cache (Probe-A dynamic param_slice)
  std::unordered_map<uint64_t, ob::CachedWeightTile*> weight_cache_map;
  std::list<ob::CachedWeightTile*> weight_cache_lru;
  size_t weight_cache_bytes = 0;
  size_t weight_cache_max_bytes = 0; // lazily initialized on first use (256 MiB mock / 2048 MiB HW)
  uint64_t weight_cache_hits = 0;
  uint64_t weight_cache_misses = 0;

  // Diagnostics
  mutable std::mutex err_mu;
  std::string last_error;
};

#endif  // OPENBUENOS_INTERNAL_H_
