// SPDX-License-Identifier: Apache-2.0
#include "ot_internal.h"
#include "ot_vliw.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <unistd.h>
#include <vector>

namespace {

std::atomic<uint64_t> g_next_ctx_instance_id{1};

bool is_foreign_forked_buffer(const ot_buffer_t* buf) {
  return buf && !buf->is_mock && buf->owner_pid != 0 &&
         buf->owner_pid != static_cast<int64_t>(::getpid());
}

inline float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<uint16_t>((u >> 16) | 0x0040U);
  }
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7fffU + lsb;
  return static_cast<uint16_t>(u >> 16);
}

thread_local std::string g_last_create_error;

ot_status_t raw_h2d_locked(ot_context_t* ctx, uint64_t dst_hbm, size_t total_buf_bytes,
                           size_t dst_offset, const uint8_t* src_bytes, size_t size_bytes) {
  if (size_bytes == 0) return OT_OK;
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  const uint64_t write_hbm = dst_hbm + dst_offset;
  if (ctx->info.is_mock) {
    if (write_hbm > ctx->transport.mock_hbm.size() ||
        size_bytes > ctx->transport.mock_hbm.size() - write_hbm) {
      ctx->last_error = "Mock HBM H2D range out of bounds";
      return OT_ERR_DEVICE;
    }
    std::memcpy(ctx->transport.mock_hbm.data() + write_hbm, src_bytes, size_bytes);
    return OT_OK;
  }

  if ((dst_offset % 8u) != 0) {
    ctx->last_error = "Hardware H2D destination offset must be 8-byte aligned";
    return OT_ERR_INVALID_ARG;
  }
  if (!ctx->transport.host_stage_mem || ctx->transport.host_stage_size < 8u) {
    ctx->last_error = "DMA staging buffer not initialized";
    return OT_ERR_DEVICE;
  }

  const size_t max_chunk = ctx->transport.host_stage_size & ~static_cast<size_t>(7u);
  size_t done = 0;
  std::string err;
  while (done < size_bytes) {
    size_t chunk = std::min(size_bytes - done, max_chunk);
    size_t dma_len = (chunk + 7u) & ~static_cast<size_t>(7u);
    if (dma_len > chunk) {
      if (dst_offset + done + chunk < total_buf_bytes) {
        // Partial H2D write ending mid-qword inside a larger buffer: read-modify-write
        // the trailing 8-byte HBM word so untouched suffix bytes are preserved.
        size_t tail_off = chunk & ~static_cast<size_t>(7u);
        if (!ctx->transport.dma_d2h_sync(
                ctx->transport.host_stage_iova + tail_off,
                write_hbm + done + tail_off, 8u,
                0x60u, 0u, ctx->cfg.timeout_ms, &err)) {
          ctx->last_error = err.empty() ? "Hardware H2D tail preserve D2H failed" : err;
          return ctx->transport.dma_faulted ? OT_ERR_TIMEOUT : OT_ERR_IOCTL;
        }
      } else {
        std::memset(ctx->transport.host_stage_mem + chunk, 0, dma_len - chunk);
      }
    }
    std::memcpy(ctx->transport.host_stage_mem, src_bytes + done, chunk);
    if (!ctx->transport.dma_h2d_sync(
            ctx->transport.host_stage_iova, write_hbm + done, dma_len,
            0x10u, 0u, ctx->cfg.timeout_ms, &err)) {
      ctx->last_error = err.empty() ? "Hardware H2D DMA failed" : err;
      return ctx->transport.dma_faulted ? OT_ERR_TIMEOUT : OT_ERR_IOCTL;
    }
    done += chunk;
  }
  return OT_OK;
}

ot_status_t raw_d2h_locked(ot_context_t* ctx, uint64_t src_hbm,
                           uint8_t* dst_bytes, size_t size_bytes) {
  if (size_bytes == 0) return OT_OK;
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  if (ctx->info.is_mock) {
    if (src_hbm > ctx->transport.mock_hbm.size() ||
        size_bytes > ctx->transport.mock_hbm.size() - src_hbm) {
      ctx->last_error = "Mock HBM D2H range out of bounds";
      return OT_ERR_DEVICE;
    }
    std::memcpy(dst_bytes, ctx->transport.mock_hbm.data() + src_hbm, size_bytes);
    return OT_OK;
  }

  if (!ctx->transport.host_stage_mem || ctx->transport.host_stage_size < 8u) {
    ctx->last_error = "DMA staging buffer not initialized";
    return OT_ERR_DEVICE;
  }

  const size_t max_chunk = ctx->transport.host_stage_size & ~static_cast<size_t>(7u);
  size_t done = 0;
  std::string err;
  while (done < size_bytes) {
    size_t chunk = std::min(size_bytes - done, max_chunk);
    size_t dma_len = (chunk + 7u) & ~static_cast<size_t>(7u);
    if (!ctx->transport.dma_d2h_sync(
            ctx->transport.host_stage_iova, src_hbm + done, dma_len,
            0x60u, 0u, ctx->cfg.timeout_ms, &err)) {
      ctx->last_error = err.empty() ? "Hardware D2H DMA failed" : err;
      return ctx->transport.dma_faulted ? OT_ERR_TIMEOUT : OT_ERR_IOCTL;
    }
    std::memcpy(dst_bytes + done, ctx->transport.host_stage_mem, chunk);
    done += chunk;
  }
  return OT_OK;
}

ot_status_t read_buffer_row_major_locked(ot_context_t* ctx, const ot_buffer_t* src,
                                         uint8_t* dst_bytes, size_t size_bytes) {
  if (size_bytes == 0) return OT_OK;
  if (src->layout != ot::BufferLayout::kTiled2x128Bf16) {
    return raw_d2h_locked(ctx, src->hbm_addr, dst_bytes, size_bytes);
  }
  if (src->size_bytes < ot::kTile128x128Bf16Bytes) {
    ctx->last_error = "Invalid tiled buffer size";
    return OT_ERR_INVALID_ARG;
  }
  std::vector<uint16_t> tiled(128u * 128u);
  ot_status_t st = raw_d2h_locked(ctx, src->hbm_addr,
                                  reinterpret_cast<uint8_t*>(tiled.data()),
                                  ot::kTile128x128Bf16Bytes);
  if (st != OT_OK) return st;

  std::vector<uint16_t> rm(128u * 128u);
  ot::tile_unpack_128x128_bf16(tiled.data(), rm.data());
  const size_t first = std::min(size_bytes, ot::kTile128x128Bf16Bytes);
  std::memcpy(dst_bytes, rm.data(), first);
  if (size_bytes > ot::kTile128x128Bf16Bytes) {
    return raw_d2h_locked(ctx, src->hbm_addr + ot::kTile128x128Bf16Bytes,
                          dst_bytes + ot::kTile128x128Bf16Bytes,
                          size_bytes - ot::kTile128x128Bf16Bytes);
  }
  return OT_OK;
}

ot_status_t write_buffer_raw_locked(ot_context_t* ctx, ot_buffer_t* dst, size_t dst_offset,
                                    const uint8_t* src_bytes, size_t size_bytes) {
  // If overwriting a tiled buffer without replacing its entire 32 KiB tiled prefix,
  // materialize the tiled prefix to row-major first so untouched logical bytes survive.
  if (dst->layout == ot::BufferLayout::kTiled2x128Bf16 &&
      !(dst_offset == 0 && size_bytes >= ot::kTile128x128Bf16Bytes)) {
    std::vector<uint8_t> rm(ot::kTile128x128Bf16Bytes, 0);
    ot_status_t st = read_buffer_row_major_locked(ctx, dst, rm.data(), rm.size());
    if (st != OT_OK) return st;
    if (dst_offset < ot::kTile128x128Bf16Bytes) {
      size_t overlap = std::min(size_bytes, ot::kTile128x128Bf16Bytes - dst_offset);
      std::memcpy(rm.data() + dst_offset, src_bytes, overlap);
    }
    st = raw_h2d_locked(ctx, dst->hbm_addr, dst->size_bytes, 0u, rm.data(), rm.size());
    if (st != OT_OK) return st;
    dst->layout = ot::BufferLayout::kRaw;
    if (dst_offset + size_bytes > ot::kTile128x128Bf16Bytes) {
      size_t tail_start = std::max(dst_offset, ot::kTile128x128Bf16Bytes);
      size_t src_skip = tail_start - dst_offset;
      size_t tail_len = size_bytes - src_skip;
      return raw_h2d_locked(ctx, dst->hbm_addr, dst->size_bytes, tail_start,
                            src_bytes + src_skip, tail_len);
    }
    return OT_OK;
  }

  ot_status_t st = raw_h2d_locked(ctx, dst->hbm_addr, dst->size_bytes, dst_offset,
                                  src_bytes, size_bytes);
  if (st != OT_OK) return st;
  dst->layout = ot::BufferLayout::kRaw;
  return OT_OK;
}

}  // namespace

namespace ot {

ot_status_t buffer_h2d_tiled_128x128_bf16(ot_context_t* ctx, ot_buffer_t* dst,
                                          const void* host_src_row_major) {
  if (!ctx || !dst || !host_src_row_major) return OT_ERR_INVALID_ARG;
  if (is_foreign_forked_buffer(dst)) return OT_ERR_DEVICE;
  if (dst->ctx != ctx || dst->ctx_instance_id != ctx->instance_id ||
      dst->size_bytes < kTile128x128Bf16Bytes) {
    return OT_ERR_INVALID_ARG;
  }
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    return OT_ERR_DEVICE;
  }
  std::vector<uint16_t> rm(128u * 128u);
  std::memcpy(rm.data(), host_src_row_major, kTile128x128Bf16Bytes);
  std::vector<uint16_t> tiled(128u * 128u);
  tile_pack_128x128_bf16(rm.data(), tiled.data());

  std::lock_guard<std::mutex> lk(ctx->mu);
  ot_status_t st = raw_h2d_locked(ctx, dst->hbm_addr, dst->size_bytes, 0u,
                                  reinterpret_cast<const uint8_t*>(tiled.data()),
                                  kTile128x128Bf16Bytes);
  if (st != OT_OK) return st;
  dst->layout = BufferLayout::kTiled2x128Bf16;
  return OT_OK;
}

uint64_t context_hbm_used_bytes(const ot_context_t* ctx) {
  if (!ctx) return 0;
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) return 0;
  std::lock_guard<std::mutex> lk(const_cast<ot_context_t*>(ctx)->mu);
  return ctx->transport.hbm_alloc.used_bytes();
}

}  // namespace ot

extern "C" {

ot_config_t ot_config_default(void) {
  ot_config_t cfg = {};
  cfg.struct_size = sizeof(ot_config_t);
  cfg.mock_mode = 0;
  cfg.bdf = nullptr;
  cfg.hbm_budget_bytes = 0;
  cfg.timeout_ms = 5000u;
  return cfg;
}

const char* ot_status_string(ot_status_t status) {
  switch (status) {
    case OT_OK: return "OT_OK";
    case OT_ERR_INVALID_ARG: return "OT_ERR_INVALID_ARG";
    case OT_ERR_NO_MEMORY: return "OT_ERR_NO_MEMORY";
    case OT_ERR_DEVICE: return "OT_ERR_DEVICE";
    case OT_ERR_IOCTL: return "OT_ERR_IOCTL";
    case OT_ERR_TIMEOUT: return "OT_ERR_TIMEOUT";
    case OT_ERR_UNSUPPORTED: return "OT_ERR_UNSUPPORTED";
    case OT_ERR_NOT_FOUND: return "OT_ERR_NOT_FOUND";
    case OT_ERR_IO: return "OT_ERR_IO";
    case OT_ERR_BUSY: return "OT_ERR_BUSY";
  }
  return "OT_ERR_UNKNOWN";
}

ot_status_t ot_context_create(const ot_config_t* config, ot_context_t** out_ctx) {
  g_last_create_error.clear();
  if (!out_ctx) {
    g_last_create_error = "out_ctx must be non-null";
    return OT_ERR_INVALID_ARG;
  }
  *out_ctx = nullptr;

  ot_config_t cfg = ot_config_default();
  if (config) {
    if (config->struct_size != 0 && config->struct_size != sizeof(ot_config_t)) {
      g_last_create_error = "invalid ot_config_t struct_size";
      return OT_ERR_INVALID_ARG;
    }
    cfg = *config;
  }

  auto* ctx = new (std::nothrow) ot_context();
  if (!ctx) {
    g_last_create_error = "failed to allocate ot_context";
    return OT_ERR_NO_MEMORY;
  }
  ctx->instance_id = g_next_ctx_instance_id.fetch_add(1, std::memory_order_relaxed);
  ctx->cfg = cfg;

  std::string err;
  if (!ctx->transport.open_device(cfg, &ctx->info, &err)) {
    g_last_create_error = err.empty() ? "VFIO device open failed" : err;
    delete ctx;
    return OT_ERR_DEVICE;
  }

  if (!cfg.mock_mode) {
    ctx->info.is_prebooted = 0u;
    if (ctx->transport.bar4_mmio && ctx->transport.bar4_mapped_len >= 0x20) {
      size_t read_len = ctx->transport.bar4_mapped_len < 4096u ? ctx->transport.bar4_mapped_len : 4096u;
      read_len &= ~size_t{3u};
      std::vector<uint8_t> page(read_len, 0);
      for (size_t i = 0; i < page.size(); i += 4) {
        uint32_t w = ctx->transport.read32_bar4(i);
        std::memcpy(page.data() + i, &w, 4);
      }
      if (ot::parse_bar4_descriptor(page.data(), page.size(), &ctx->bar4_desc)) {
        ctx->info.is_prebooted = 1u;
      }
    }
  }

  *out_ctx = ctx;
  return OT_OK;
}

void ot_context_destroy(ot_context_t* ctx) {
  if (!ctx) return;
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    ctx->transport.close_device();
    delete ctx;
    return;
  }
  {
    std::lock_guard<std::mutex> lk(ctx->mu);
    for (ot_buffer_t* b : ctx->live_buffers) {
      if (b) {
        b->ctx = nullptr;
        b->hbm_addr = 0;
      }
    }
    ctx->live_buffers.clear();
  }
  ctx->transport.close_device();
  delete ctx;
}

const char* ot_context_last_error(const ot_context_t* ctx) {
  if (!ctx) return g_last_create_error.c_str();
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    return "VFIO context cannot be used in a forked child process";
  }
  return ctx->last_error.c_str();
}

int ot_context_is_mock(const ot_context_t* ctx) {
  if (!ctx) return 0;
  return ctx->info.is_mock ? 1 : 0;
}

ot_status_t ot_device_get_info(const ot_context_t* ctx, ot_device_info_t* out_info) {
  if (!ctx || !out_info) return OT_ERR_INVALID_ARG;
  *out_info = ctx->info;
  return OT_OK;
}

ot_status_t ot_buffer_alloc(ot_context_t* ctx, size_t size_bytes, ot_buffer_t** out_buf) {
  if (!ctx || !out_buf || size_bytes == 0 ||
      size_bytes > std::numeric_limits<uint64_t>::max() - 4095ULL) {
    return OT_ERR_INVALID_ARG;
  }
  *out_buf = nullptr;
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    return OT_ERR_DEVICE;
  }

  std::lock_guard<std::mutex> lk(ctx->mu);
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  uint64_t hbm_addr = 0;
  uint64_t aligned_sz = (static_cast<uint64_t>(size_bytes) + 4095ULL) & ~4095ULL;
  if (!ctx->transport.hbm_alloc.alloc(aligned_sz, 4096ULL, &hbm_addr)) {
    ctx->last_error = "HBM allocator out of space";
    return OT_ERR_NO_MEMORY;
  }

  auto* buf = new (std::nothrow) ot_buffer();
  if (!buf) {
    ctx->transport.hbm_alloc.free(hbm_addr);
    return OT_ERR_NO_MEMORY;
  }
  buf->ctx = ctx;
  buf->ctx_instance_id = ctx->instance_id;
  buf->size_bytes = size_bytes;
  buf->hbm_addr = hbm_addr;
  buf->is_mock = (ctx->info.is_mock != 0u);
  buf->owner_pid = ctx->transport.owner_pid;
  buf->layout = ot::BufferLayout::kRaw;
  if (ctx->info.is_mock) {
    if (hbm_addr > ctx->transport.mock_hbm.size() ||
        size_bytes > ctx->transport.mock_hbm.size() - hbm_addr) {
      ctx->transport.hbm_alloc.free(hbm_addr);
      delete buf;
      ctx->last_error = "Mock HBM backing store out of bounds";
      return OT_ERR_NO_MEMORY;
    }
    std::memset(ctx->transport.mock_hbm.data() + hbm_addr, 0, size_bytes);
  }
  ctx->live_buffers.push_back(buf);
  *out_buf = buf;
  return OT_OK;
}

void ot_buffer_free(ot_buffer_t* buf) {
  if (!buf) return;
  if (is_foreign_forked_buffer(buf)) {
    delete buf;
    return;
  }
  if (buf->ctx && buf->ctx->instance_id == buf->ctx_instance_id &&
      (buf->ctx->info.is_mock || buf->ctx->transport.is_owner_process())) {
    std::lock_guard<std::mutex> lk(buf->ctx->mu);
    auto& live = buf->ctx->live_buffers;
    auto it = std::find(live.begin(), live.end(), buf);
    if (it != live.end()) live.erase(it);
    // Keep HBM range quarantined if the context has a faulted/timed-out DMA.
    if (!buf->ctx->transport.dma_faulted) {
      buf->ctx->transport.hbm_alloc.free(buf->hbm_addr);
    }
  }
  delete buf;
}

size_t ot_buffer_size(const ot_buffer_t* buf) {
  return buf ? buf->size_bytes : 0;
}

uint64_t ot_buffer_hbm_addr(const ot_buffer_t* buf) {
  return buf ? buf->hbm_addr : 0;
}

}  // extern "C"

namespace ot {

ot_status_t buffer_h2d_offset(ot_context_t* ctx, ot_buffer_t* dst, size_t dst_offset,
                              const void* host_src, size_t size_bytes) {
  if (!ctx || !dst || !host_src) return OT_ERR_INVALID_ARG;
  if (is_foreign_forked_buffer(dst)) return OT_ERR_DEVICE;
  if (dst->ctx != ctx || dst->ctx_instance_id != ctx->instance_id ||
      dst_offset > dst->size_bytes || size_bytes > dst->size_bytes - dst_offset) {
    return OT_ERR_INVALID_ARG;
  }
  if (size_bytes == 0) return OT_OK;
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    return OT_ERR_DEVICE;
  }
  std::lock_guard<std::mutex> lk(ctx->mu);
  return write_buffer_raw_locked(ctx, dst, dst_offset,
                                 static_cast<const uint8_t*>(host_src), size_bytes);
}

}  // namespace ot

extern "C" {

ot_status_t ot_buffer_h2d(ot_context_t* ctx, ot_buffer_t* dst,
                          const void* host_src, size_t size_bytes) {
  return ot::buffer_h2d_offset(ctx, dst, 0, host_src, size_bytes);
}

ot_status_t ot_buffer_d2h(ot_context_t* ctx, const ot_buffer_t* src,
                          void* host_dst, size_t size_bytes) {
  if (!ctx || !src || !host_dst) return OT_ERR_INVALID_ARG;
  if (is_foreign_forked_buffer(src)) return OT_ERR_DEVICE;
  if (src->ctx != ctx || src->ctx_instance_id != ctx->instance_id ||
      size_bytes > src->size_bytes) {
    return OT_ERR_INVALID_ARG;
  }
  if (size_bytes == 0) return OT_OK;
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    return OT_ERR_DEVICE;
  }
  std::lock_guard<std::mutex> lk(ctx->mu);
  return read_buffer_row_major_locked(ctx, src, static_cast<uint8_t*>(host_dst), size_bytes);
}

ot_status_t ot_matmul_bf16(ot_context_t* ctx,
                           const ot_buffer_t* lhs_mk,
                           const ot_buffer_t* rhs_kn,
                           ot_buffer_t* out_mn,
                           uint32_t m_dim,
                           uint32_t k_dim,
                           uint32_t n_dim) {
  if (!ctx || !lhs_mk || !rhs_kn || !out_mn ||
      m_dim == 0 || k_dim == 0 || n_dim == 0) {
    return OT_ERR_INVALID_ARG;
  }
  if (is_foreign_forked_buffer(lhs_mk) || is_foreign_forked_buffer(rhs_kn) ||
      is_foreign_forked_buffer(out_mn)) {
    return OT_ERR_DEVICE;
  }
  if (lhs_mk->ctx != ctx || lhs_mk->ctx_instance_id != ctx->instance_id ||
      rhs_kn->ctx != ctx || rhs_kn->ctx_instance_id != ctx->instance_id ||
      out_mn->ctx != ctx || out_mn->ctx_instance_id != ctx->instance_id) {
    return OT_ERR_INVALID_ARG;
  }
  auto mul_bf16_bytes = [](uint32_t a, uint32_t b, uint64_t* out) -> bool {
    const uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    if (prod > std::numeric_limits<uint64_t>::max() / sizeof(uint16_t)) return false;
    *out = prod * sizeof(uint16_t);
    return true;
  };
  uint64_t need_a = 0, need_b = 0, need_c = 0;
  if (!mul_bf16_bytes(m_dim, k_dim, &need_a) ||
      !mul_bf16_bytes(k_dim, n_dim, &need_b) ||
      !mul_bf16_bytes(m_dim, n_dim, &need_c)) {
    return OT_ERR_INVALID_ARG;
  }
  if (lhs_mk->size_bytes < need_a || rhs_kn->size_bytes < need_b || out_mn->size_bytes < need_c) {
    return OT_ERR_INVALID_ARG;
  }
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    return OT_ERR_DEVICE;
  }

  std::lock_guard<std::mutex> lk(ctx->mu);
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  auto ranges_overlap = [](const ot_buffer_t* x, const ot_buffer_t* y) -> bool {
    if (x == y) return true;
    if (x->size_bytes == 0 || y->size_bytes == 0) return false;
    uint64_t x_end = x->hbm_addr + x->size_bytes;
    uint64_t y_end = y->hbm_addr + y->size_bytes;
    return x->hbm_addr < y_end && y->hbm_addr < x_end;
  };

  if (ctx->info.is_mock) {
    struct MallocGuard {
      uint16_t* p = nullptr;
      ~MallocGuard() { std::free(p); }
    };
    const size_t hbm_sz = ctx->transport.mock_hbm.size();
    auto in_mock_bounds = [&](uint64_t addr, uint64_t len) -> bool {
      return addr <= hbm_sz && len <= hbm_sz - addr;
    };
    if (!in_mock_bounds(lhs_mk->hbm_addr, need_a) ||
        !in_mock_bounds(rhs_kn->hbm_addr, need_b) ||
        !in_mock_bounds(out_mn->hbm_addr, need_c)) {
      ctx->last_error = "Mock HBM matmul range out of bounds";
      return OT_ERR_DEVICE;
    }

    MallocGuard a_tmp, b_tmp;
    const uint16_t* a_ptr = nullptr;
    const uint16_t* b_ptr = nullptr;
    const bool lhs_needs_copy =
        (lhs_mk->layout == ot::BufferLayout::kTiled2x128Bf16) || ranges_overlap(lhs_mk, out_mn);
    const bool rhs_needs_copy =
        (rhs_kn->layout == ot::BufferLayout::kTiled2x128Bf16) || ranges_overlap(rhs_kn, out_mn);

    if (lhs_needs_copy) {
      a_tmp.p = static_cast<uint16_t*>(std::malloc(static_cast<size_t>(need_a)));
      if (!a_tmp.p) return OT_ERR_NO_MEMORY;
      ot_status_t st = read_buffer_row_major_locked(
          ctx, lhs_mk, reinterpret_cast<uint8_t*>(a_tmp.p), static_cast<size_t>(need_a));
      if (st != OT_OK) return st;
      a_ptr = a_tmp.p;
    } else {
      a_ptr = reinterpret_cast<const uint16_t*>(ctx->transport.mock_hbm.data() + lhs_mk->hbm_addr);
    }

    if (rhs_needs_copy) {
      if (rhs_kn == lhs_mk && need_b == need_a && a_tmp.p) {
        b_ptr = a_tmp.p;
      } else {
        b_tmp.p = static_cast<uint16_t*>(std::malloc(static_cast<size_t>(need_b)));
        if (!b_tmp.p) return OT_ERR_NO_MEMORY;
        ot_status_t st = read_buffer_row_major_locked(
            ctx, rhs_kn, reinterpret_cast<uint8_t*>(b_tmp.p), static_cast<size_t>(need_b));
        if (st != OT_OK) return st;
        b_ptr = b_tmp.p;
      }
    } else {
      b_ptr = reinterpret_cast<const uint16_t*>(ctx->transport.mock_hbm.data() + rhs_kn->hbm_addr);
    }

    if (m_dim == 128u && k_dim == 128u && n_dim == 128u) {
      MallocGuard c_rm, c_tiled;
      c_rm.p = static_cast<uint16_t*>(std::malloc(ot::kTile128x128Bf16Bytes));
      c_tiled.p = static_cast<uint16_t*>(std::malloc(ot::kTile128x128Bf16Bytes));
      if (!c_rm.p || !c_tiled.p) return OT_ERR_NO_MEMORY;
      for (uint32_t m = 0; m < m_dim; ++m) {
        for (uint32_t n = 0; n < n_dim; ++n) {
          float acc = 0.0f;
          for (uint32_t k = 0; k < k_dim; ++k) {
            acc += bf16_to_f32(a_ptr[static_cast<size_t>(m) * k_dim + k]) *
                   bf16_to_f32(b_ptr[static_cast<size_t>(k) * n_dim + n]);
          }
          c_rm.p[static_cast<size_t>(m) * n_dim + n] = f32_to_bf16(acc);
        }
      }
      ot::tile_pack_128x128_bf16(c_rm.p, c_tiled.p);
      std::memcpy(ctx->transport.mock_hbm.data() + out_mn->hbm_addr,
                  c_tiled.p, ot::kTile128x128Bf16Bytes);
      out_mn->layout = ot::BufferLayout::kTiled2x128Bf16;
      return OT_OK;
    }

    // If reusing a tiled output buffer for a smaller raw matmul, unpack the 32 KiB
    // tile in-place first so unwritten suffix bytes remain valid row-major data.
    if (out_mn->layout == ot::BufferLayout::kTiled2x128Bf16 &&
        need_c < ot::kTile128x128Bf16Bytes) {
      MallocGuard prev_rm;
      prev_rm.p = static_cast<uint16_t*>(std::malloc(ot::kTile128x128Bf16Bytes));
      if (!prev_rm.p) return OT_ERR_NO_MEMORY;
      ot_status_t st = read_buffer_row_major_locked(
          ctx, out_mn, reinterpret_cast<uint8_t*>(prev_rm.p), ot::kTile128x128Bf16Bytes);
      if (st != OT_OK) return st;
      st = raw_h2d_locked(ctx, out_mn->hbm_addr, out_mn->size_bytes, 0u,
                          reinterpret_cast<const uint8_t*>(prev_rm.p),
                          ot::kTile128x128Bf16Bytes);
      if (st != OT_OK) return st;
      out_mn->layout = ot::BufferLayout::kRaw;
    }

    auto* c_ptr = reinterpret_cast<uint16_t*>(ctx->transport.mock_hbm.data() + out_mn->hbm_addr);
    for (uint32_t m = 0; m < m_dim; ++m) {
      for (uint32_t n = 0; n < n_dim; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < k_dim; ++k) {
          acc += bf16_to_f32(a_ptr[static_cast<size_t>(m) * k_dim + k]) *
                 bf16_to_f32(b_ptr[static_cast<size_t>(k) * n_dim + n]);
        }
        c_ptr[static_cast<size_t>(m) * n_dim + n] = f32_to_bf16(acc);
      }
    }
    out_mn->layout = ot::BufferLayout::kRaw;
    return OT_OK;
  }

  if (m_dim != 128u || k_dim != 128u || n_dim != 128u) {
    ctx->last_error = "Live hardware BF16 matmul currently supports 128x128x128 only";
    return OT_ERR_UNSUPPORTED;
  }
  if ((lhs_mk->hbm_addr & 0x1ffULL) != 0 || (rhs_kn->hbm_addr & 0x1ffULL) != 0 ||
      (out_mn->hbm_addr & 0x1ffULL) != 0) {
    ctx->last_error = "Hardware matmul HBM buffers must be 512-byte aligned";
    return OT_ERR_INVALID_ARG;
  }
  if (!ctx->transport.host_stage_mem ||
      ctx->transport.host_stage_size < ot::kTile128x128Bf16Bytes) {
    ctx->last_error = "Host DMA staging buffer unavailable for hardware matmul";
    return OT_ERR_DEVICE;
  }

  auto transport_status = [&](const std::string& e) -> ot_status_t {
    if (e.find("timeout") != std::string::npos) {
      return OT_ERR_TIMEOUT;
    }
    if (ctx->transport.dma_faulted) {
      return OT_ERR_DEVICE;
    }
    return OT_ERR_IOCTL;
  };

  std::string err;
  if (!ctx->transport.ensure_imem_booted(ctx->cfg.timeout_ms, &err)) {
    ctx->last_error = err.empty() ? "IMEM boot stub initialization failed" : err;
    return transport_status(err);
  }

  auto prepare_tiled_input = [&](const ot_buffer_t* src, uint64_t* scratch_slot,
                                 uint64_t* out_hbm) -> ot_status_t {
    if (src->layout == ot::BufferLayout::kTiled2x128Bf16 && !ranges_overlap(src, out_mn)) {
      *out_hbm = src->hbm_addr;
      return OT_OK;
    }
    if (!ctx->transport.ensure_matmul_scratch_slot(scratch_slot, &err)) {
      ctx->last_error = err.empty() ? "Failed to allocate matmul scratch HBM" : err;
      return OT_ERR_NO_MEMORY;
    }
    const uint64_t dst_scratch = *scratch_slot;
    if (src->layout == ot::BufferLayout::kTiled2x128Bf16) {
      // In-place aliasing (src overlaps out_mn) with an already tiled buffer: copy raw 32 KiB to scratch.
      if (!ctx->transport.dma_d2h_sync(ctx->transport.host_stage_iova, src->hbm_addr,
                                       ot::kTile128x128Bf16Bytes, 0x60u, 0u,
                                       ctx->cfg.timeout_ms, &err) ||
          !ctx->transport.dma_h2d_sync(ctx->transport.host_stage_iova, dst_scratch,
                                       ot::kTile128x128Bf16Bytes, 0x10u, 0u,
                                       ctx->cfg.timeout_ms, &err)) {
        ctx->last_error = err.empty() ? "Scratch copy DMA failed" : err;
        return transport_status(err);
      }
      *out_hbm = dst_scratch;
      return OT_OK;
    }

    // Raw row-major input in HBM: D2H -> pack {1,0:T(2,128)} -> H2D into scratch.
    std::vector<uint16_t> rm(128u * 128u);
    ot_status_t st = raw_d2h_locked(ctx, src->hbm_addr,
                                    reinterpret_cast<uint8_t*>(rm.data()),
                                    ot::kTile128x128Bf16Bytes);
    if (st != OT_OK) return st;
    std::vector<uint16_t> tiled(128u * 128u);
    ot::tile_pack_128x128_bf16(rm.data(), tiled.data());
    st = raw_h2d_locked(ctx, dst_scratch, ot::kTile128x128Bf16Bytes, 0u,
                        reinterpret_cast<const uint8_t*>(tiled.data()),
                        ot::kTile128x128Bf16Bytes);
    if (st != OT_OK) return st;
    *out_hbm = dst_scratch;
    return OT_OK;
  };

  uint64_t lhs_hbm = 0;
  uint64_t rhs_hbm = 0;
  ot_status_t st = prepare_tiled_input(lhs_mk, &ctx->transport.scratch_lhs_hbm, &lhs_hbm);
  if (st != OT_OK) return st;
  if (rhs_kn == lhs_mk) {
    rhs_hbm = lhs_hbm;
  } else {
    st = prepare_tiled_input(rhs_kn, &ctx->transport.scratch_rhs_hbm, &rhs_hbm);
    if (st != OT_OK) return st;
  }

  const uint32_t next_seq =
      (ctx->transport.exec_seq == 0xffffffffu) ? 1u : (ctx->transport.exec_seq + 1u);
  const ot::OtV5eLaunchDesc ldesc = ot::OtV5eLaunchDesc::build(
      lhs_hbm, rhs_hbm, out_mn->hbm_addr, /*prog_hbm=*/0ULL, next_seq);

  std::memcpy(ctx->transport.host_stage_mem, ldesc.bytes, sizeof(ldesc.bytes));
  if (!ctx->transport.dma_h2d_sync(ctx->transport.host_stage_iova,
                                   ot::kDefaultLaunchDescHbmAddr,
                                   sizeof(ldesc.bytes),
                                   ot::kLdescDmaW28Doorbell,
                                   ot::kLdescDmaW29,
                                   ctx->cfg.timeout_ms, &err)) {
    ctx->last_error = err.empty() ? "Launch descriptor doorbell DMA failed" : err;
    return transport_status(err);
  }
  ctx->transport.exec_seq = next_seq;

  uint32_t cq_out[8] = {};
  if (!ctx->transport.submit_exec_sync(next_seq, ctx->cfg.timeout_ms, cq_out, &err)) {
    ctx->last_error = err.empty() ? "Hardware MXU EXEC failed" : err;
    return transport_status(err);
  }

  out_mn->layout = ot::BufferLayout::kTiled2x128Bf16;
  return OT_OK;
}

}  // extern "C"
