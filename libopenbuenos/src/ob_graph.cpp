#include "ob_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ob {

namespace {

void destroy_cached_graph_resources(ob_context* ctx, CachedGraph* entry) {
  if (!entry) return;
  if (ctx && entry->model_handle != 0u) {
    uint8_t outer_unreg[64]{};
    size_t len = litebuf_build_vii_unregister_model(
        outer_unreg, sizeof(outer_unreg), entry->model_handle);
    if (len > 0u) {
      struct litebuf_vii_parsed_response resp {};
      uint16_t code = 0;
      device_submit_control(ctx, outer_unreg, len, &resp, &code, ctx->cfg.timeout_ms);
    }
    entry->model_handle = 0u;
  }
  if (entry->reg_cmd_buf) ob_buffer_free(entry->reg_cmd_buf);
  if (entry->params_buf && entry->owns_params_buf) ob_buffer_free(entry->params_buf);
  if (entry->vm_ir_buf) ob_buffer_free(entry->vm_ir_buf);
  if (entry->image_buf && entry->owns_image_buf) ob_buffer_free(entry->image_buf);
  if (entry->scratch_buf) ob_buffer_free(entry->scratch_buf);
  delete entry;
}

// Evict up to `max_evict` unreferenced LRU entries (called with ctx->cache_mu held).
size_t evict_unreferenced_batch_locked(ob_context* ctx, size_t max_evict) {
  size_t evicted = 0u;
  auto it = ctx->cache_lru.end();
  while (evicted < max_evict && it != ctx->cache_lru.begin()) {
    --it;
    CachedGraph* victim = *it;
    if (victim->ref_count == 0u) {
      it = ctx->cache_lru.erase(it);
      ctx->cache_map.erase(victim->key);
      victim->in_cache = false;
      ctx->cache_evictions += 1u;
      destroy_cached_graph_resources(ctx, victim);
      evicted += 1u;
    }
  }
  return evicted;
}

// Must be called with ctx->cache_mu held. Evicts unreferenced LRU entries
// until cache_lru.size() <= cache_capacity.
void evict_lru_locked(ob_context* ctx) {
  if (ctx->cache_lru.size() > ctx->cache_capacity) {
    evict_unreferenced_batch_locked(ctx, ctx->cache_lru.size() - ctx->cache_capacity);
  }
}

// Acquire shared read-only DMA-BUF for ext[1] RV64 DIVE interpreter when identical.
// Returns OB_OK with `*out_buf` set and `*out_owned` indicating whether caller owns it.
ob_status_t acquire_image_buf(ob_context* ctx,
                              const uint8_t* image_bytes,
                              size_t image_size,
                              ob_buffer** out_buf,
                              bool* out_owned) {
  *out_buf = nullptr;
  *out_owned = true;
  if (!image_bytes || image_size == 0u) return OB_ERR_INVALID_ARG;

  const char* share_env = std::getenv("OPENBUENOS_SHARE_IMAGE");
  const bool allow_share = !(share_env && share_env[0] == '0' && share_env[1] == '\0');

  Sha256Digest digest{};
  if (allow_share) {
    sha256_compute(image_bytes, image_size, digest.bytes.data());

    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    if (ctx->has_shared_image && ctx->shared_image_buf &&
        ctx->shared_image_size == image_size &&
        ctx->shared_image_sha256 == digest) {
      *out_buf = ctx->shared_image_buf;
      *out_owned = false;
      return OB_OK;
    }
  }

  // Allocate & populate a new buffer outside cache_mu.
  // NOTE: Tensor G5 RV64 DIVE interpreter writes to its in-image .bss tail
  // (~0x9ac00..0x9c000) during RegisterModel, so image_buf MUST be mapped
  // OB_BUF_BIDIRECTIONAL (matching tpu_baremetal_runner.cpp).
  ob_buffer* buf = nullptr;
  ob_status_t st = ob_buffer_alloc(
      ctx, image_size, OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL | OB_BUF_ZERO_INIT, &buf);
  if (st != OB_OK) return st;
  st = ob_buffer_write(buf, image_bytes, image_size, 0u);
  if (st != OB_OK) {
    ob_buffer_free(buf);
    return st;
  }

  if (!allow_share) {
    *out_buf = buf;
    *out_owned = true;
    return OB_OK;
  }

  std::lock_guard<std::mutex> lk(ctx->cache_mu);
  if (!ctx->has_shared_image) {
    ctx->has_shared_image = true;
    ctx->shared_image_sha256 = digest;
    ctx->shared_image_size = image_size;
    ctx->shared_image_buf = buf;
    *out_buf = buf;
    *out_owned = false;
    return OB_OK;
  }
  if (ctx->shared_image_buf && ctx->shared_image_size == image_size &&
      ctx->shared_image_sha256 == digest) {
    // Another thread populated the identical shared buffer first.
    ob_buffer_free(buf);
    *out_buf = ctx->shared_image_buf;
    *out_owned = false;
    return OB_OK;
  }
  // Distinct ext[1] image: caller owns its private copy.
  *out_buf = buf;
  *out_owned = true;
  return OB_OK;
}

constexpr size_t kSharedZeroParamsBytes = 8u * 1024u * 1024u;

ob_status_t acquire_shared_zero_params_buf(ob_context* ctx,
                                           size_t needed_bytes,
                                           ob_buffer** out_buf,
                                           bool* out_owned) {
  *out_buf = nullptr;
  *out_owned = true;
  const char* env = std::getenv("OPENBUENOS_SHARE_ZERO_PARAMS");
  if ((env && env[0] == '0' && env[1] == '\0') || needed_bytes > kSharedZeroParamsBytes) {
    return OB_ERR_UNSUPPORTED;
  }
  {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    if (ctx->shared_zero_params_buf) {
      *out_buf = ctx->shared_zero_params_buf;
      *out_owned = false;
      return OB_OK;
    }
  }
  ob_buffer* buf = nullptr;
  ob_status_t st = ob_buffer_alloc(
      ctx, kSharedZeroParamsBytes,
      OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL | OB_BUF_ZERO_INIT, &buf);
  if (st != OB_OK) return st;
  {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    if (!ctx->shared_zero_params_buf) {
      ctx->shared_zero_params_buf = buf;
      *out_buf = buf;
      *out_owned = false;
      return OB_OK;
    }
    ob_buffer_free(buf);
    *out_buf = ctx->shared_zero_params_buf;
    *out_owned = false;
    return OB_OK;
  }
}

ob_status_t upload_and_register_graph(ob_context* ctx,
                                      const uint8_t* blob,
                                      const ParsedDgc0& parsed,
                                      bool share_zero_params,
                                      CachedGraph** out_entry) {
  *out_entry = nullptr;
  CachedGraph* e = new (std::nothrow) CachedGraph();
  if (!e) return OB_ERR_NO_MEMORY;
  e->ctx = ctx;
  e->parsed = parsed;

  ob_status_t st = OB_OK;
  do {
    size_t scratch_sz =
        parsed.dive_scratch_size > 0u ? static_cast<size_t>(parsed.dive_scratch_size) : 4096u;
    st = ob_buffer_alloc(ctx, scratch_sz, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &e->scratch_buf);
    if (st != OB_OK) break;

    st = acquire_image_buf(ctx,
                           blob + parsed.ext_dive_image.offset,
                           static_cast<size_t>(parsed.ext_dive_image.size),
                           &e->image_buf,
                           &e->owns_image_buf);
    if (st != OB_OK) break;

    st = ob_buffer_alloc(ctx, static_cast<size_t>(parsed.ext_dive_vm_ir.size),
                         OB_BUF_BIDIRECTIONAL, &e->vm_ir_buf);
    if (st != OB_OK) break;
    st = ob_buffer_write(e->vm_ir_buf, blob + parsed.ext_dive_vm_ir.offset,
                         static_cast<size_t>(parsed.ext_dive_vm_ir.size), 0u);
    if (st != OB_OK) break;

    struct litebuf_dive_buffer_slice param_slice {};
    uint32_t param_count = 0u;
    if (parsed.has_params) {
      bool shared_ok = false;
      if (share_zero_params) {
        ob_status_t sst = acquire_shared_zero_params_buf(
            ctx, static_cast<size_t>(parsed.ext_params.size),
            &e->params_buf, &e->owns_params_buf);
        if (sst == OB_OK) {
          shared_ok = true;
        }
      }
      if (!shared_ok) {
        e->owns_params_buf = true;
        st = ob_buffer_alloc(ctx, static_cast<size_t>(parsed.ext_params.size),
                             OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &e->params_buf);
        if (st != OB_OK) break;
        st = ob_buffer_write(e->params_buf, blob + parsed.ext_params.offset,
                             static_cast<size_t>(parsed.ext_params.size), 0u);
        if (st != OB_OK) break;
      }
      param_slice.address = e->params_buf->slab.iova;
      param_slice.size_bytes = parsed.ext_params.size;
      param_count = 1u;
    }

    uint8_t inner_reg[256]{};
    size_t inner_reg_len = litebuf_build_dive_register_interpreter(
        inner_reg, sizeof(inner_reg),
        e->vm_ir_buf->slab.iova, parsed.ext_dive_vm_ir.size,
        param_count ? &param_slice : nullptr, param_count, 0u, 0u);
    if (inner_reg_len == 0u) {
      set_last_error(ctx, "Failed building inner RegisterInterpreterModel Litebuf");
      st = OB_ERR_OVERFLOW;
      break;
    }

    st = ob_buffer_alloc(ctx, inner_reg_len, OB_BUF_BIDIRECTIONAL, &e->reg_cmd_buf);
    if (st != OB_OK) break;
    st = ob_buffer_write(e->reg_cmd_buf, inner_reg, inner_reg_len, 0u);
    if (st != OB_OK) break;

    uint8_t cluster = (ctx->cfg.cluster_id != 0u) ? ctx->cfg.cluster_id : parsed.cluster_id;
    struct litebuf_register_model_params rp {};
    rp.dive_cmd_address = e->reg_cmd_buf->slab.iova;
    rp.dive_cmd_size_bytes = static_cast<uint32_t>(inner_reg_len);
    rp.dive_image_address = e->image_buf->slab.iova;
    rp.dive_image_size_bytes = static_cast<uint32_t>(parsed.ext_dive_image.size);
    rp.dive_scratch_address = e->scratch_buf->slab.iova;
    rp.dive_scratch_size_bytes = static_cast<uint32_t>(parsed.dive_scratch_size);
    rp.dive_heap_size = parsed.dive_heap_size;
    rp.exec_field_29 = parsed.exec_field_29;
    rp.core_session_handle = 0u;
    rp.cluster_id = cluster;
    rp.exec_field_31 = parsed.exec_field_31;
    rp.exec_field_14 = parsed.exec_field_14;
    rp.compiler_cl = parsed.compiler_cl;
    rp.subgraph_flag_18 = parsed.subgraph_flag_18;
    rp.exec_field_34 = parsed.exec_field_34;
    rp.model_id = "Unknown";

    uint8_t outer_reg[256]{};
    size_t outer_reg_len = litebuf_build_vii_register_model(outer_reg, sizeof(outer_reg), &rp);
    if (outer_reg_len == 0u) {
      set_last_error(ctx, "Failed building outer RegisterModel Litebuf");
      st = OB_ERR_OVERFLOW;
      break;
    }

    struct litebuf_vii_parsed_response reg_resp {};
    uint16_t resp_code = 0;
    st = device_submit_control(ctx, outer_reg, outer_reg_len, &reg_resp, &resp_code,
                               ctx->cfg.timeout_ms);
    if (st != OB_OK) break;
    if (reg_resp.response_type != LITEBUF_VII_RESP_REGISTER_MODEL) {
      set_last_error(ctx, "Expected REGISTER_MODEL response type=2, got " +
                              std::to_string(reg_resp.response_type));
      st = OB_ERR_FW_ERROR;
      break;
    }
    e->model_handle = reg_resp.handle;
    // RegisterModel is synchronous; return the 4 KiB inner registration command
    // buffer to the slab pool immediately so subsequent registrations reuse it
    // (zero per-graph FD retention and immediate buffer-pool hits).
    if (e->reg_cmd_buf) {
      ob_buffer_free(e->reg_cmd_buf);
      e->reg_cmd_buf = nullptr;
    }
  } while (false);

  if (st != OB_OK) {
    destroy_cached_graph_resources(ctx, e);
    return st;
  }

  *out_entry = e;
  return OB_OK;
}

}  // namespace

void graph_entry_unref_locked(ob_context* ctx, CachedGraph* entry) {
  if (!ctx || !entry) return;
  if (entry->ref_count > 0u) entry->ref_count -= 1u;
  if (entry->ref_count == 0u) {
    if (!entry->in_cache) {
      destroy_cached_graph_resources(ctx, entry);
    } else if (ctx->cache_lru.size() > ctx->cache_capacity) {
      evict_lru_locked(ctx);
    }
  }
}

void graph_cache_clear(ob_context* ctx) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lk(ctx->cache_mu);
  for (CachedGraph* entry : ctx->cache_lru) {
    entry->in_cache = false;
    destroy_cached_graph_resources(ctx, entry);
  }
  ctx->cache_lru.clear();
  ctx->cache_map.clear();
}

void shared_image_clear(ob_context* ctx) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lk(ctx->cache_mu);
  if (ctx->shared_image_buf) {
    ob_buffer_free(ctx->shared_image_buf);
    ctx->shared_image_buf = nullptr;
  }
  if (ctx->shared_zero_params_buf) {
    ob_buffer_free(ctx->shared_zero_params_buf);
    ctx->shared_zero_params_buf = nullptr;
  }
  ctx->has_shared_image = false;
  ctx->shared_image_size = 0;
}

bool graph_cache_lookup_entry_u64(ob_context* ctx, uint64_t cache_key, CachedGraph** out_entry) {
  if (!ctx || cache_key == 0u || !out_entry) return false;
  *out_entry = nullptr;
  CacheKey key {};
  key.is_u64 = true;
  key.u64_key = cache_key;

  std::lock_guard<std::mutex> lk(ctx->cache_mu);
  auto it = ctx->cache_map.find(key);
  if (it == ctx->cache_map.end()) return false;

  CachedGraph* entry = it->second;
  if (entry->lru_it != ctx->cache_lru.begin()) {
    ctx->cache_lru.splice(ctx->cache_lru.begin(), ctx->cache_lru, entry->lru_it);
  }
  entry->ref_count += 1u;
  ctx->cache_hits += 1u;
  *out_entry = entry;
  return true;
}

bool graph_cache_lookup_u64(ob_context* ctx, uint64_t cache_key, ob_graph_t** out_graph) {
  if (!ctx || cache_key == 0u || !out_graph) return false;
  *out_graph = nullptr;
  CachedGraph* entry = nullptr;
  if (!graph_cache_lookup_entry_u64(ctx, cache_key, &entry)) return false;

  ob_graph* handle = new (std::nothrow) ob_graph();
  if (!handle) {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    entry->ref_count -= 1u;
    ctx->cache_hits -= 1u;
    return false;
  }
  handle->entry = entry;
  handle->was_cache_hit = true;
  *out_graph = handle;
  return true;
}

ob_status_t graph_submit_async_with_params(ob_graph_t* graph,
                                           ob_buffer* const* inputs,
                                           uint32_t num_inputs,
                                           ob_buffer* const* outputs,
                                           uint32_t num_outputs,
                                           const ob_buffer* override_params,
                                           ob_exec_handle_t* out_handle) {
  if (!graph || !graph->entry || !graph->entry->ctx || !out_handle) {
    return OB_ERR_INVALID_ARG;
  }
  ob_context* ctx = graph->entry->ctx;
  ob::CachedGraph* entry = graph->entry;

  {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    entry->ref_count += 1u;
  }

  ob_status_t st = device_submit_execute_async(
      ctx, entry, inputs, num_inputs, outputs, num_outputs, out_handle, override_params);
  if (st != OB_OK) {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    graph_entry_unref_locked(ctx, entry);
  }
  return st;
}

ob_status_t graph_register_dgc0_internal(ob_context* ctx,
                                         const void* dgc0_bytes,
                                         size_t dgc0_size,
                                         uint64_t cache_key,
                                         bool share_zero_params,
                                         CachedGraph** out_entry,
                                         bool* out_cache_hit) {
  if (!ctx || !out_entry) return OB_ERR_INVALID_ARG;
  *out_entry = nullptr;
  if (out_cache_hit) *out_cache_hit = false;

  CacheKey key {};
  if (cache_key != 0u) {
    key.is_u64 = true;
    key.u64_key = cache_key;
  } else {
    if (!dgc0_bytes || dgc0_size == 0u) return OB_ERR_INVALID_ARG;
    key.is_u64 = false;
    sha256_compute(dgc0_bytes, dgc0_size, key.sha256.bytes.data());
  }

  {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    auto it = ctx->cache_map.find(key);
    if (it != ctx->cache_map.end()) {
      CachedGraph* entry = it->second;
      ctx->cache_lru.splice(ctx->cache_lru.begin(), ctx->cache_lru, entry->lru_it);
      entry->ref_count += 1u;
      ctx->cache_hits += 1u;
      *out_entry = entry;
      if (out_cache_hit) *out_cache_hit = true;
      return OB_OK;
    }
  }

  if (!dgc0_bytes || dgc0_size == 0u) {
    set_last_error(ctx, "Cache miss requires non-empty dgc0_bytes");
    return OB_ERR_INVALID_ARG;
  }

  const uint8_t* bytes = static_cast<const uint8_t*>(dgc0_bytes);
  ParsedDgc0 parsed;
  std::string err;
  ob_status_t st = parse_dgc0_buffer(bytes, dgc0_size, &parsed, &err);
  if (st != OB_OK) {
    set_last_error(ctx, err);
    return st;
  }

  CachedGraph* new_entry = nullptr;
  st = upload_and_register_graph(ctx, bytes, parsed, share_zero_params, &new_entry);
  if (st != OB_OK &&
      (st == OB_ERR_FW_ERROR || st == OB_ERR_NO_MEMORY || st == OB_ERR_IOCTL)) {
    size_t evicted = 0u;
    {
      std::lock_guard<std::mutex> lk(ctx->cache_mu);
      if (!ctx->cache_lru.empty()) {
        size_t batch = std::max<size_t>(16u, ctx->cache_lru.size() / 4u);
        evicted = evict_unreferenced_batch_locked(ctx, batch);
      }
    }
    if (evicted > 0u) {
      st = upload_and_register_graph(ctx, bytes, parsed, share_zero_params, &new_entry);
    }
  }
  if (st != OB_OK) return st;

  new_entry->key = key;

  {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    auto it = ctx->cache_map.find(key);
    if (it != ctx->cache_map.end()) {
      CachedGraph* existing = it->second;
      ctx->cache_lru.splice(ctx->cache_lru.begin(), ctx->cache_lru, existing->lru_it);
      existing->ref_count += 1u;
      ctx->cache_hits += 1u;
      *out_entry = existing;
      if (out_cache_hit) *out_cache_hit = true;
      destroy_cached_graph_resources(ctx, new_entry);
    } else {
      ctx->cache_misses += 1u;
      new_entry->ref_count = 1u;
      new_entry->in_cache = true;
      ctx->cache_lru.push_front(new_entry);
      new_entry->lru_it = ctx->cache_lru.begin();
      ctx->cache_map[key] = new_entry;
      evict_lru_locked(ctx);
      *out_entry = new_entry;
      if (out_cache_hit) *out_cache_hit = false;
    }
  }
  return OB_OK;
}

}  // namespace ob

extern "C" {

ob_status_t ob_graph_register_dgc0(ob_context_t* ctx,
                                   const void* dgc0_bytes,
                                   size_t dgc0_size,
                                   uint64_t cache_key,
                                   ob_graph_t** out_graph) {
  if (!ctx || !out_graph) return OB_ERR_INVALID_ARG;
  *out_graph = nullptr;

  ob::CachedGraph* entry = nullptr;
  bool was_hit = false;
  ob_status_t st = ob::graph_register_dgc0_internal(
      ctx, dgc0_bytes, dgc0_size, cache_key, false, &entry, &was_hit);
  if (st != OB_OK) return st;

  ob_graph* handle = new (std::nothrow) ob_graph();
  if (!handle) {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    ob::graph_entry_unref_locked(ctx, entry);
    return OB_ERR_NO_MEMORY;
  }
  handle->entry = entry;
  handle->was_cache_hit = was_hit;
  *out_graph = handle;
  return OB_OK;
}

ob_status_t ob_graph_register_dgc0_file(ob_context_t* ctx,
                                        const char* path,
                                        uint64_t cache_key,
                                        ob_graph_t** out_graph) {
  if (!ctx || !path || !path[0] || !out_graph) return OB_ERR_INVALID_ARG;
  *out_graph = nullptr;

  int fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    ob::set_last_error(ctx, std::string("open(") + path + ") failed errno=" +
                                std::to_string(errno));
    return OB_ERR_IO;
  }
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    ob::set_last_error(ctx, std::string("fstat(") + path + ") failed or empty file");
    return OB_ERR_IO;
  }

  std::vector<uint8_t> blob(static_cast<size_t>(st.st_size));
  size_t got = 0;
  while (got < blob.size()) {
    ssize_t n = ::read(fd, blob.data() + got, blob.size() - got);
    if (n < 0) {
      if (errno == EINTR) continue;
      ::close(fd);
      ob::set_last_error(ctx, std::string("read(") + path + ") failed errno=" +
                                  std::to_string(errno));
      return OB_ERR_IO;
    }
    if (n == 0) break;
    got += static_cast<size_t>(n);
  }
  ::close(fd);
  blob.resize(got);

  return ob_graph_register_dgc0(ctx, blob.data(), blob.size(), cache_key, out_graph);
}

void ob_graph_release(ob_graph_t* graph) {
  if (!graph) return;
  ob::CachedGraph* entry = graph->entry;
  ob_context* ctx = entry ? entry->ctx : nullptr;
  delete graph;
  if (!ctx || !entry) return;
  std::lock_guard<std::mutex> lk(ctx->cache_mu);
  ob::graph_entry_unref_locked(ctx, entry);
}

ob_status_t ob_graph_get_info(const ob_graph_t* graph, ob_dgc0_info_t* out_info) {
  if (!graph || !graph->entry || !out_info) return OB_ERR_INVALID_ARG;
  ob::fill_dgc0_info(graph->entry->parsed, graph->entry->model_handle,
                     graph->was_cache_hit, out_info);
  return OB_OK;
}

ob_status_t ob_context_graph_cache_stats(const ob_context_t* ctx,
                                         size_t* out_entries,
                                         uint64_t* out_hits,
                                         uint64_t* out_misses,
                                         uint64_t* out_evictions) {
  if (!ctx) return OB_ERR_INVALID_ARG;
  std::lock_guard<std::mutex> lk(const_cast<ob_context*>(ctx)->cache_mu);
  if (out_entries) *out_entries = ctx->cache_lru.size();
  if (out_hits) *out_hits = ctx->cache_hits;
  if (out_misses) *out_misses = ctx->cache_misses;
  if (out_evictions) *out_evictions = ctx->cache_evictions;
  return OB_OK;
}

ob_status_t ob_graph_submit_async(ob_graph_t* graph,
                                  ob_buffer_t* const* inputs,
                                  uint32_t num_inputs,
                                  ob_buffer_t* const* outputs,
                                  uint32_t num_outputs,
                                  ob_exec_handle_t* out_handle) {
  if (!graph || !graph->entry || !graph->entry->ctx || !out_handle) {
    return OB_ERR_INVALID_ARG;
  }
  ob_context* ctx = graph->entry->ctx;
  ob::CachedGraph* entry = graph->entry;

  // Pin entry while in-flight so ob_graph_release() is safe before wait()
  {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    entry->ref_count += 1u;
  }

  ob_status_t st = ob::device_submit_execute_async(
      ctx, entry, inputs, num_inputs, outputs, num_outputs, out_handle);
  if (st != OB_OK) {
    std::lock_guard<std::mutex> lk(ctx->cache_mu);
    ob::graph_entry_unref_locked(ctx, entry);
  }
  return st;
}

ob_status_t ob_graph_wait(ob_context_t* ctx,
                          ob_exec_handle_t handle,
                          uint32_t timeout_ms,
                          ob_invoke_stats_t* out_stats) {
  return ob::device_wait_handle(ctx, handle, timeout_ms, out_stats);
}

ob_status_t ob_graph_wait_all(ob_context_t* ctx, uint32_t timeout_ms) {
  return ob::device_wait_all(ctx, timeout_ms);
}

ob_status_t ob_graph_invoke(ob_graph_t* graph,
                            ob_buffer_t* const* inputs,
                            uint32_t num_inputs,
                            ob_buffer_t* const* outputs,
                            uint32_t num_outputs,
                            ob_invoke_stats_t* out_stats) {
  if (!graph || !graph->entry || !graph->entry->ctx) return OB_ERR_INVALID_ARG;
  ob_context* ctx = graph->entry->ctx;

  ob_exec_handle_t h = 0;
  ob_status_t st = ob_graph_submit_async(graph, inputs, num_inputs, outputs, num_outputs, &h);
  if (st != OB_OK) return st;

  st = ob_graph_wait(ctx, h, ctx->cfg.timeout_ms, out_stats);
  if (st == OB_OK && out_stats && graph->was_cache_hit) {
    out_stats->flags |= OB_STAT_FLAG_CACHE_HIT;
    out_stats->cache_hits = 1u;
  }
  return st;
}

}  // extern "C"
