// SPDX-License-Identifier: Apache-2.0
#include "ot_internal.h"
#include "ot_vliw.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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
          return ot::classify_transport_error(ctx->transport.last_op_timed_out,
                                              ctx->transport.dma_faulted);
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
      return ot::classify_transport_error(ctx->transport.last_op_timed_out,
                                          ctx->transport.dma_faulted);
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
      return ot::classify_transport_error(ctx->transport.last_op_timed_out,
                                          ctx->transport.dma_faulted);
    }
    std::memcpy(dst_bytes + done, ctx->transport.host_stage_mem, chunk);
    done += chunk;
  }
  return OT_OK;
}

struct ScopedBuf {
  uint8_t* ptr = nullptr;
  explicit ScopedBuf(size_t bytes) : ptr(static_cast<uint8_t*>(std::malloc(bytes))) {}
  ~ScopedBuf() { std::free(ptr); }
  ScopedBuf(const ScopedBuf&) = delete;
  ScopedBuf& operator=(const ScopedBuf&) = delete;
  uint16_t* u16() const { return reinterpret_cast<uint16_t*>(ptr); }
};

ot_status_t read_buffer_row_major_locked(ot_context_t* ctx, const ot_buffer_t* src,
                                         uint8_t* dst_bytes, size_t size_bytes) {
  if (size_bytes == 0) return OT_OK;
  if (src->layout != ot::BufferLayout::kTiled2x128Bf16) {
    return raw_d2h_locked(ctx, src->hbm_addr, dst_bytes, size_bytes);
  }
  const uint32_t rows =
      (src->tiled_rows >= 128u && (src->tiled_rows % 128u) == 0u) ? src->tiled_rows : 128u;
  const uint32_t cols =
      (src->tiled_cols >= 128u && (src->tiled_cols % 128u) == 0u) ? src->tiled_cols : 128u;
  const size_t tiled_bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
  if (src->size_bytes < tiled_bytes) {
    ctx->last_error = "Invalid tiled buffer size";
    return OT_ERR_INVALID_ARG;
  }
  ScopedBuf tiled(tiled_bytes);
  if (!tiled.ptr) {
    ctx->last_error = "Failed to allocate host tile readback workspace";
    return OT_ERR_NO_MEMORY;
  }
  ot_status_t st = raw_d2h_locked(ctx, src->hbm_addr, tiled.ptr, tiled_bytes);
  if (st != OT_OK) return st;

  if (rows == 128u && cols == 128u) {
    if (size_bytes >= ot::kTile128x128Bf16Bytes &&
        (reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) == 0) {
      ot::tile_unpack_128x128_bf16(tiled.u16(), reinterpret_cast<uint16_t*>(dst_bytes));
    } else {
      ScopedBuf rm(ot::kTile128x128Bf16Bytes);
      if (!rm.ptr) {
        ctx->last_error = "Failed to allocate host tile unpack workspace";
        return OT_ERR_NO_MEMORY;
      }
      ot::tile_unpack_128x128_bf16(tiled.u16(), rm.u16());
      std::memcpy(dst_bytes, rm.ptr, std::min(size_bytes, ot::kTile128x128Bf16Bytes));
    }
  } else {
    ScopedBuf rm(tiled_bytes);
    ScopedBuf tile_rm(ot::kTile128x128Bf16Bytes);
    if (!rm.ptr || !tile_rm.ptr) {
      ctx->last_error = "Failed to allocate multi-tile unpack workspace";
      return OT_ERR_NO_MEMORY;
    }
    const uint32_t m_tiles = rows / 128u;
    const uint32_t n_tiles = cols / 128u;
    uint16_t* dst_u16 = rm.u16();
    for (uint32_t mt = 0; mt < m_tiles; ++mt) {
      for (uint32_t nt = 0; nt < n_tiles; ++nt) {
        const size_t tile_idx = static_cast<size_t>(mt) * n_tiles + nt;
        ot::tile_unpack_128x128_bf16(tiled.u16() + tile_idx * (128u * 128u), tile_rm.u16());
        for (uint32_t r = 0; r < 128u; ++r) {
          const size_t dst_off =
              (static_cast<size_t>(mt * 128u + r) * cols + nt * 128u);
          std::memcpy(dst_u16 + dst_off, tile_rm.u16() + r * 128u, 128u * sizeof(uint16_t));
        }
      }
    }
    std::memcpy(dst_bytes, rm.ptr, std::min(size_bytes, tiled_bytes));
  }
  if (size_bytes > tiled_bytes) {
    return raw_d2h_locked(ctx, src->hbm_addr + tiled_bytes,
                          dst_bytes + tiled_bytes,
                          size_bytes - tiled_bytes);
  }
  return OT_OK;
}

size_t buffer_tiled_bytes(const ot_buffer_t* buf) {
  if (!buf || buf->layout != ot::BufferLayout::kTiled2x128Bf16) return 0u;
  const uint32_t rows =
      (buf->tiled_rows >= 128u && (buf->tiled_rows % 128u) == 0u) ? buf->tiled_rows : 128u;
  const uint32_t cols =
      (buf->tiled_cols >= 128u && (buf->tiled_cols % 128u) == 0u) ? buf->tiled_cols : 128u;
  return static_cast<size_t>(rows) * cols * sizeof(uint16_t);
}

// Materialize an existing tiled prefix to logical row-major before a smaller
// write overwrites the start of `dst`, preserving untouched logical suffix bytes.
// Call ONLY after any aliased inputs have already been read/staged out of `dst`.
ot_status_t preserve_tiled_suffix_locked(ot_context_t* ctx, ot_buffer_t* dst,
                                         size_t new_write_bytes) {
  const size_t old_tiled_bytes = buffer_tiled_bytes(dst);
  if (old_tiled_bytes == 0u || old_tiled_bytes <= new_write_bytes) {
    return OT_OK;
  }
  ScopedBuf prev_rm(old_tiled_bytes);
  if (!prev_rm.ptr) {
    ctx->last_error = "Failed to allocate host tile materialize workspace";
    return OT_ERR_NO_MEMORY;
  }
  ot_status_t st = read_buffer_row_major_locked(ctx, dst, prev_rm.ptr, old_tiled_bytes);
  if (st != OT_OK) return st;
  st = raw_h2d_locked(ctx, dst->hbm_addr, dst->size_bytes, 0u, prev_rm.ptr, old_tiled_bytes);
  if (st != OT_OK) return st;
  dst->layout = ot::BufferLayout::kRaw;
  dst->tiled_rows = 0u;
  dst->tiled_cols = 0u;
  return OT_OK;
}

ot_status_t write_buffer_raw_locked(ot_context_t* ctx, ot_buffer_t* dst, size_t dst_offset,
                                    const uint8_t* src_bytes, size_t size_bytes) {
  // If overwriting a tiled buffer without replacing its entire tiled prefix,
  // materialize the tiled prefix to row-major first so untouched logical bytes survive.
  const size_t tiled_bytes = buffer_tiled_bytes(dst);
  if (dst->layout == ot::BufferLayout::kTiled2x128Bf16 &&
      !(dst_offset == 0 && size_bytes >= tiled_bytes)) {
    ScopedBuf rm(tiled_bytes);
    if (!rm.ptr) {
      ctx->last_error = "Failed to allocate host tile materialize workspace";
      return OT_ERR_NO_MEMORY;
    }
    ot_status_t st = read_buffer_row_major_locked(ctx, dst, rm.ptr, tiled_bytes);
    if (st != OT_OK) return st;
    if (dst_offset < tiled_bytes) {
      size_t overlap = std::min(size_bytes, tiled_bytes - dst_offset);
      std::memcpy(rm.ptr + dst_offset, src_bytes, overlap);
    }
    st = raw_h2d_locked(ctx, dst->hbm_addr, dst->size_bytes, 0u, rm.ptr, tiled_bytes);
    if (st != OT_OK) return st;
    dst->layout = ot::BufferLayout::kRaw;
    dst->tiled_rows = 0;
    dst->tiled_cols = 0;
    if (dst_offset + size_bytes > tiled_bytes) {
      size_t tail_start = std::max(dst_offset, tiled_bytes);
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
  dst->tiled_rows = 0;
  dst->tiled_cols = 0;
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
  ScopedBuf tiled(kTile128x128Bf16Bytes);
  if (!tiled.ptr) return OT_ERR_NO_MEMORY;
  if ((reinterpret_cast<uintptr_t>(host_src_row_major) % alignof(uint16_t)) == 0) {
    tile_pack_128x128_bf16(static_cast<const uint16_t*>(host_src_row_major), tiled.u16());
  } else {
    ScopedBuf rm(kTile128x128Bf16Bytes);
    if (!rm.ptr) return OT_ERR_NO_MEMORY;
    std::memcpy(rm.ptr, host_src_row_major, kTile128x128Bf16Bytes);
    tile_pack_128x128_bf16(rm.u16(), tiled.u16());
  }

  std::lock_guard<std::mutex> lk(ctx->mu);
  ot_status_t st = preserve_tiled_suffix_locked(ctx, dst, kTile128x128Bf16Bytes);
  if (st != OT_OK) return st;
  st = raw_h2d_locked(ctx, dst->hbm_addr, dst->size_bytes, 0u,
                      tiled.ptr, kTile128x128Bf16Bytes);
  if (st != OT_OK) return st;
  dst->layout = BufferLayout::kTiled2x128Bf16;
  dst->tiled_rows = 128u;
  dst->tiled_cols = 128u;
  return OT_OK;
}

ot_status_t buffer_d2h_raw(ot_context_t* ctx, const ot_buffer_t* src,
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
  return raw_d2h_locked(ctx, src->hbm_addr, static_cast<uint8_t*>(host_dst), size_bytes);
}

ot_status_t buffer_d2h_tiled_128x128_bf16(ot_context_t* ctx, const ot_buffer_t* src,
                                          void* host_dst_tiled) {
  if (!ctx || !src || !host_dst_tiled) return OT_ERR_INVALID_ARG;
  if (is_foreign_forked_buffer(src)) return OT_ERR_DEVICE;
  if (src->ctx != ctx || src->ctx_instance_id != ctx->instance_id ||
      src->size_bytes < kTile128x128Bf16Bytes) {
    return OT_ERR_INVALID_ARG;
  }
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    return OT_ERR_DEVICE;
  }
  std::lock_guard<std::mutex> lk(ctx->mu);
  const bool exact_single_tile =
      (src->layout == BufferLayout::kTiled2x128Bf16) &&
      ((src->tiled_rows == 0u && src->tiled_cols == 0u) ||
       (src->tiled_rows == 128u && src->tiled_cols == 128u));
  if (exact_single_tile) {
    return raw_d2h_locked(ctx, src->hbm_addr, static_cast<uint8_t*>(host_dst_tiled),
                          kTile128x128Bf16Bytes);
  }
  ScopedBuf rm(kTile128x128Bf16Bytes);
  if (!rm.ptr) {
    ctx->last_error = "Failed to allocate host tile pack workspace";
    return OT_ERR_NO_MEMORY;
  }
  ot_status_t st = read_buffer_row_major_locked(ctx, src, rm.ptr, kTile128x128Bf16Bytes);
  if (st != OT_OK) return st;
  if ((reinterpret_cast<uintptr_t>(host_dst_tiled) % alignof(uint16_t)) == 0) {
    tile_pack_128x128_bf16(rm.u16(), static_cast<uint16_t*>(host_dst_tiled));
  } else {
    ScopedBuf tiled(kTile128x128Bf16Bytes);
    if (!tiled.ptr) {
      ctx->last_error = "Failed to allocate host tile pack output workspace";
      return OT_ERR_NO_MEMORY;
    }
    tile_pack_128x128_bf16(rm.u16(), tiled.u16());
    std::memcpy(host_dst_tiled, tiled.ptr, kTile128x128Bf16Bytes);
  }
  return OT_OK;
}

uint64_t context_hbm_used_bytes(const ot_context_t* ctx) {
  if (!ctx) return 0;
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) return 0;
  std::lock_guard<std::mutex> lk(const_cast<ot_context_t*>(ctx)->mu);
  return ctx->transport.hbm_alloc.used_bytes();
}

ot_status_t classify_transport_error(bool last_op_timed_out, bool dma_faulted) {
  if (last_op_timed_out) return OT_ERR_TIMEOUT;
  if (dma_faulted) return OT_ERR_DEVICE;
  return OT_ERR_IOCTL;
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
      uint8_t page[4096] = {};
      for (size_t i = 0; i < read_len; i += 4) {
        uint32_t w = ctx->transport.read32_bar4(i);
        std::memcpy(page + i, &w, 4);
      }
      if (ot::parse_bar4_descriptor(page, read_len, &ctx->bar4_desc)) {
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

}  // extern "C"

namespace {

struct HbmScratchGuard {
  ot_context_t* ctx = nullptr;
  uint64_t addr = 0;
  ~HbmScratchGuard() {
    if (ctx && addr != 0 && !ctx->transport.dma_faulted) {
      ctx->transport.hbm_alloc.free(addr);
    }
  }
};

ot_status_t transport_err_status(ot_context_t* ctx, const std::string& e) {
  (void)e;
  return ot::classify_transport_error(ctx->transport.last_op_timed_out,
                                      ctx->transport.dma_faulted);
}

void pack_bf16_matrix_tiles(const uint16_t* row_major, uint32_t rows, uint32_t cols,
                            uint16_t* tiled_out) {
  const uint32_t mt_cnt = rows / 128u;
  const uint32_t nt_cnt = cols / 128u;
  std::vector<uint16_t> tile_rm(128u * 128u);
  for (uint32_t mt = 0; mt < mt_cnt; ++mt) {
    for (uint32_t nt = 0; nt < nt_cnt; ++nt) {
      for (uint32_t r = 0; r < 128u; ++r) {
        const size_t src_off = static_cast<size_t>(mt * 128u + r) * cols + nt * 128u;
        std::memcpy(tile_rm.data() + r * 128u, row_major + src_off, 128u * sizeof(uint16_t));
      }
      const size_t tile_idx = static_cast<size_t>(mt) * nt_cnt + nt;
      ot::tile_pack_128x128_bf16(tile_rm.data(), tiled_out + tile_idx * (128u * 128u));
    }
  }
}

// Pack a row-major [dst_cols, dst_rows] (i.e. [N, K]) BF16 matrix transposed into
// tiled [dst_rows, dst_cols] (i.e. [K, N]) 128x128 MXU tiles.
void pack_bf16_matrix_tiles_transposed(const uint16_t* row_major_transposed,
                                       uint32_t dst_rows, uint32_t dst_cols,
                                       uint16_t* tiled_out) {
  const uint32_t rt_cnt = dst_rows / 128u;
  const uint32_t ct_cnt = dst_cols / 128u;
  std::vector<uint16_t> tile_rm(128u * 128u);
  for (uint32_t rt = 0; rt < rt_cnt; ++rt) {
    for (uint32_t ct = 0; ct < ct_cnt; ++ct) {
      for (uint32_t r = 0; r < 128u; ++r) {
        const size_t gr = static_cast<size_t>(rt) * 128u + r;
        for (uint32_t c = 0; c < 128u; ++c) {
          const size_t gc = static_cast<size_t>(ct) * 128u + c;
          tile_rm[r * 128u + c] = row_major_transposed[gc * dst_rows + gr];
        }
      }
      const size_t tile_idx = static_cast<size_t>(rt) * ct_cnt + ct;
      ot::tile_pack_128x128_bf16(tile_rm.data(), tiled_out + tile_idx * (128u * 128u));
    }
  }
}

bool ranges_overlap_buf(const ot_buffer_t* x, const ot_buffer_t* y) {
  if (x == y) return true;
  if (x->size_bytes == 0 || y->size_bytes == 0) return false;
  return x->hbm_addr < (y->hbm_addr + y->size_bytes) &&
         y->hbm_addr < (x->hbm_addr + x->size_bytes);
}

ot_status_t prepare_live_bf16_input_locked(ot_context_t* ctx,
                                           const ot_buffer_t* src,
                                           const ot_buffer_t* out_mn,
                                           uint32_t rows,
                                           uint32_t cols,
                                           bool transpose_src,
                                           uint64_t* single_tile_slot,
                                           HbmScratchGuard* dyn_guard,
                                           uint64_t* out_hbm,
                                           std::string* err) {
  const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
  const bool already_tiled =
      !transpose_src &&
      (src->layout == ot::BufferLayout::kTiled2x128Bf16) &&
      ((rows == 128u && cols == 128u && src->tiled_rows == 0u) ||
       (src->tiled_rows == rows && src->tiled_cols == cols));
  if (already_tiled && !ranges_overlap_buf(src, out_mn)) {
    *out_hbm = src->hbm_addr;
    return OT_OK;
  }

  uint64_t dst_scratch = 0;
  if (single_tile_slot && bytes == ot::kTile128x128Bf16Bytes) {
    if (!ctx->transport.ensure_matmul_scratch_slot(single_tile_slot, err)) {
      ctx->last_error = (err && !err->empty()) ? *err : "Failed to allocate matmul scratch HBM";
      return OT_ERR_NO_MEMORY;
    }
    dst_scratch = *single_tile_slot;
  } else {
    if (!ctx->transport.hbm_alloc.alloc(bytes, 4096u, &dyn_guard->addr)) {
      ctx->last_error = "Out of device HBM for multi-tile BF16 packing workspace";
      return OT_ERR_NO_MEMORY;
    }
    dst_scratch = dyn_guard->addr;
  }

  if (already_tiled) {
    ScopedBuf copy_buf(bytes);
    if (!copy_buf.ptr) return OT_ERR_NO_MEMORY;
    ot_status_t st = raw_d2h_locked(ctx, src->hbm_addr, copy_buf.ptr, bytes);
    if (st != OT_OK) return st;
    st = raw_h2d_locked(ctx, dst_scratch, bytes, 0u, copy_buf.ptr, bytes);
    if (st != OT_OK) return st;
    *out_hbm = dst_scratch;
    return OT_OK;
  }

  ScopedBuf rm(bytes);
  ScopedBuf tiled(bytes);
  if (!rm.ptr || !tiled.ptr) {
    ctx->last_error = "Failed to allocate host multi-tile pack workspace";
    return OT_ERR_NO_MEMORY;
  }
  ot_status_t st = read_buffer_row_major_locked(ctx, src, rm.ptr, bytes);
  if (st != OT_OK) return st;
  if (transpose_src) {
    pack_bf16_matrix_tiles_transposed(rm.u16(), rows, cols, tiled.u16());
  } else {
    pack_bf16_matrix_tiles(rm.u16(), rows, cols, tiled.u16());
  }
  st = raw_h2d_locked(ctx, dst_scratch, bytes, 0u, tiled.ptr, bytes);
  if (st != OT_OK) return st;
  *out_hbm = dst_scratch;
  return OT_OK;
}

ot_status_t validate_gemm_buffers(ot_context_t* ctx, const ot_buffer_t* a,
                                  const ot_buffer_t* b, const ot_buffer_t* c) {
  if (!ctx || !a || !b || !c) return OT_ERR_INVALID_ARG;
  if (is_foreign_forked_buffer(a) || is_foreign_forked_buffer(b) ||
      is_foreign_forked_buffer(c)) {
    return OT_ERR_DEVICE;
  }
  if (a->ctx != ctx || a->ctx_instance_id != ctx->instance_id ||
      b->ctx != ctx || b->ctx_instance_id != ctx->instance_id ||
      c->ctx != ctx || c->ctx_instance_id != ctx->instance_id) {
    return OT_ERR_INVALID_ARG;
  }
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    return OT_ERR_DEVICE;
  }
  return OT_OK;
}

ot_status_t launch_v5e_chunked_exec(ot_context_t* ctx,
                                  uint64_t lhs_hbm,
                                  uint64_t rhs_hbm,
                                  uint64_t out_hbm,
                                  uint64_t scales_hbm,
                                  uint64_t const_hbm,
                                  uint32_t m_tiles,
                                  uint32_t k_tiles,
                                  uint32_t n_tiles,
                                  uint32_t quant_mode,
                                  uint64_t rhs_tile_bytes,
                                  uint32_t max_iters_per_launch,
                                  uint64_t* out_hw_us,
                                  std::string* err) {
  const uint32_t m_step =
      (k_tiles >= max_iters_per_launch)
          ? 1u
          : std::min(m_tiles, std::max(1u, max_iters_per_launch / k_tiles));
  const uint32_t n_step_base =
      (k_tiles >= max_iters_per_launch) ? 1u : std::max(1u, max_iters_per_launch / k_tiles);

  uint64_t total_hw_us = 0u;
  for (uint32_t ms = 0u; ms < m_tiles;) {
    const uint32_t m_chunk = std::min(m_step, m_tiles - ms);
    const uint32_t n_max_for_m = std::max(1u, n_step_base / m_chunk);
    for (uint32_t ns = 0u; ns < n_tiles;) {
      const uint32_t n_chunk = std::min(n_max_for_m, n_tiles - ns);
      const uint64_t sub_lhs = lhs_hbm + static_cast<uint64_t>(ms) * k_tiles * ot::kTile128x128Bf16Bytes;
      const uint64_t sub_rhs = rhs_hbm + static_cast<uint64_t>(ns) * rhs_tile_bytes;
      const uint64_t sub_out = out_hbm +
          (static_cast<uint64_t>(ms) * n_tiles + ns) * ot::kTile128x128Bf16Bytes;
      const uint64_t sub_sc = (scales_hbm != 0ULL)
          ? (scales_hbm + static_cast<uint64_t>(ns) * 1024ULL)
          : 0ULL;
      const uint32_t n_stride =
          (m_tiles == 1u && k_tiles == 1u && n_tiles == 1u) ? 0u : n_tiles;

      const uint32_t next_seq =
          (ctx->transport.exec_seq == 0xffffffffu) ? 1u : (ctx->transport.exec_seq + 1u);
      const ot::OtV5eLaunchDesc ldesc = ot::OtV5eLaunchDesc::build_ex(
          sub_lhs, sub_rhs, sub_out, /*prog_hbm=*/0ULL, next_seq,
          m_chunk, k_tiles, n_chunk, quant_mode, sub_sc, const_hbm, n_stride);

      std::memcpy(ctx->transport.host_stage_mem, ldesc.bytes, sizeof(ldesc.bytes));
      const auto t0 = std::chrono::steady_clock::now();
      if (!ctx->transport.dma_h2d_sync(ctx->transport.host_stage_iova,
                                       ot::kDefaultLaunchDescHbmAddr,
                                       sizeof(ldesc.bytes),
                                       ot::kLdescDmaW28Doorbell,
                                       ot::kLdescDmaW29,
                                       ctx->cfg.timeout_ms, err)) {
        ctx->last_error = err->empty() ? "Launch descriptor doorbell DMA failed" : *err;
        return transport_err_status(ctx, *err);
      }
      ctx->transport.exec_seq = next_seq;

      uint32_t cq_out[8] = {};
      if (!ctx->transport.submit_exec_sync(next_seq, ctx->cfg.timeout_ms, cq_out, err)) {
        ctx->last_error = err->empty() ? "Hardware MXU EXEC failed" : *err;
        return transport_err_status(ctx, *err);
      }
      const auto t1 = std::chrono::steady_clock::now();
      total_hw_us += static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
      ns += n_chunk;
    }
    ms += m_chunk;
  }
  if (out_hw_us) *out_hw_us = total_hw_us;
  return OT_OK;
}

}  // namespace

extern "C" {

ot_freetile_gemm_desc_t ot_freetile_gemm_desc_default(uint32_t m_dim,
                                                      uint32_t k_dim,
                                                      uint32_t n_dim) {
  ot_freetile_gemm_desc_t d{};
  d.struct_size = sizeof(ot_freetile_gemm_desc_t);
  d.m_dim = m_dim;
  d.k_dim = k_dim;
  d.n_dim = n_dim;
  d.tile.block_m = 128u;
  d.tile.block_k = 128u;
  d.tile.block_n = 128u;
  d.tile.vmem_double_buffer = 1u;
  d.tile.k_loop_accum_f32 = 1u;
  d.tile.rhs_transposed = 0u;
  d.tile.reserved = 0u;
  d.lhs_dtype = OT_DTYPE_BF16;
  d.rhs_dtype = OT_DTYPE_BF16;
  d.out_dtype = OT_DTYPE_BF16;
  d.quant_mode = OT_QUANT_NONE;
  d.quant_group_size = 0u;
  d.quant_scale = 1.0f;
  d.scale_dtype = OT_DTYPE_F32;
  d.flags = 0u;
  return d;
}

ot_status_t ot_matmul_bf16(ot_context_t* ctx,
                           const ot_buffer_t* lhs_mk,
                           const ot_buffer_t* rhs_kn,
                           ot_buffer_t* out_mn,
                           uint32_t m_dim,
                           uint32_t k_dim,
                           uint32_t n_dim) {
  const ot_freetile_gemm_desc_t desc = ot_freetile_gemm_desc_default(m_dim, k_dim, n_dim);
  return ot_matmul_bf16_ex(ctx, lhs_mk, rhs_kn, out_mn, &desc, nullptr);
}

ot_status_t ot_matmul_bf16_ex(ot_context_t* ctx,
                              const ot_buffer_t* lhs_mk,
                              const ot_buffer_t* rhs_kn,
                              ot_buffer_t* out_mn,
                              const ot_freetile_gemm_desc_t* desc,
                              ot_exec_stats_t* out_stats) {
  const auto wall_t0 = std::chrono::steady_clock::now();
  if (!desc || desc->m_dim == 0 || desc->k_dim == 0 || desc->n_dim == 0) {
    return OT_ERR_INVALID_ARG;
  }
  if (desc->struct_size != 0u && desc->struct_size != sizeof(ot_freetile_gemm_desc_t)) {
    return OT_ERR_INVALID_ARG;
  }
  if (desc->lhs_dtype != OT_DTYPE_BF16 || desc->rhs_dtype != OT_DTYPE_BF16 ||
      desc->out_dtype != OT_DTYPE_BF16 || desc->quant_mode != OT_QUANT_NONE) {
    return OT_ERR_INVALID_ARG;
  }
  if ((desc->tile.block_m != 0u && desc->tile.block_m != 128u) ||
      (desc->tile.block_k != 0u && desc->tile.block_k != 128u) ||
      (desc->tile.block_n != 0u && desc->tile.block_n != 128u) ||
      desc->tile.rhs_transposed > 1u) {
    return OT_ERR_INVALID_ARG;
  }
  ot_status_t st_val = validate_gemm_buffers(ctx, lhs_mk, rhs_kn, out_mn);
  if (st_val != OT_OK) return st_val;

  const uint32_t m_dim = desc->m_dim;
  const uint32_t k_dim = desc->k_dim;
  const uint32_t n_dim = desc->n_dim;
  const bool rhs_transposed = (desc->tile.rhs_transposed != 0u);
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

  std::lock_guard<std::mutex> lk(ctx->mu);
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  if (out_stats) std::memset(out_stats, 0, sizeof(*out_stats));

  const bool is_tile_aligned =
      ((m_dim % 128u) == 0u) && ((k_dim % 128u) == 0u) && ((n_dim % 128u) == 0u);
  ot::freetile::LoweringPlan plan{};
  if (is_tile_aligned) {
    auto fdesc = ot::freetile::TileKernelDescriptor::from_c_desc(*desc);
    plan = ot::freetile::plan_v5e_tile_gemm(fdesc);
    if (!plan.valid) {
      ctx->last_error = plan.error;
      return OT_ERR_INVALID_ARG;
    }
  }

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
        (lhs_mk->layout == ot::BufferLayout::kTiled2x128Bf16) || ranges_overlap_buf(lhs_mk, out_mn);
    const bool rhs_needs_copy =
        (rhs_kn->layout == ot::BufferLayout::kTiled2x128Bf16) || ranges_overlap_buf(rhs_kn, out_mn);

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

    auto read_rhs_bf16 = [&](uint32_t k, uint32_t n) -> uint16_t {
      return rhs_transposed
                 ? b_ptr[static_cast<size_t>(n) * k_dim + k]
                 : b_ptr[static_cast<size_t>(k) * n_dim + n];
    };

    if (is_tile_aligned) {
      MallocGuard c_rm, c_tiled;
      c_rm.p = static_cast<uint16_t*>(std::malloc(static_cast<size_t>(need_c)));
      c_tiled.p = static_cast<uint16_t*>(std::malloc(static_cast<size_t>(need_c)));
      if (!c_rm.p || !c_tiled.p) return OT_ERR_NO_MEMORY;
      for (uint32_t m = 0; m < m_dim; ++m) {
        for (uint32_t n = 0; n < n_dim; ++n) {
          float acc = 0.0f;
          for (uint32_t k = 0; k < k_dim; ++k) {
            acc += bf16_to_f32(a_ptr[static_cast<size_t>(m) * k_dim + k]) *
                   bf16_to_f32(read_rhs_bf16(k, n));
          }
          c_rm.p[static_cast<size_t>(m) * n_dim + n] = f32_to_bf16(acc);
        }
      }
      pack_bf16_matrix_tiles(c_rm.p, m_dim, n_dim, c_tiled.p);
      ot_status_t st_suf = preserve_tiled_suffix_locked(ctx, out_mn, static_cast<size_t>(need_c));
      if (st_suf != OT_OK) return st_suf;
      std::memcpy(ctx->transport.mock_hbm.data() + out_mn->hbm_addr,
                  c_tiled.p, static_cast<size_t>(need_c));
      out_mn->layout = ot::BufferLayout::kTiled2x128Bf16;
      out_mn->tiled_rows = m_dim;
      out_mn->tiled_cols = n_dim;
      if (out_stats) {
        const auto wall_t1 = std::chrono::steady_clock::now();
        out_stats->hw_us = 0u;
        out_stats->wall_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(wall_t1 - wall_t0).count());
        out_stats->tiles_m = plan.tiles_m;
        out_stats->tiles_k = plan.tiles_k;
        out_stats->tiles_n = plan.tiles_n;
        out_stats->vmem_bytes_used = plan.vmem.total_vmem_bytes;
        out_stats->hbm_bytes_moved = plan.hbm_bytes_moved;
      }
      return OT_OK;
    }

    ot_status_t st_suf = preserve_tiled_suffix_locked(ctx, out_mn, static_cast<size_t>(need_c));
    if (st_suf != OT_OK) return st_suf;

    auto* c_ptr = reinterpret_cast<uint16_t*>(ctx->transport.mock_hbm.data() + out_mn->hbm_addr);
    for (uint32_t m = 0; m < m_dim; ++m) {
      for (uint32_t n = 0; n < n_dim; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < k_dim; ++k) {
          acc += bf16_to_f32(a_ptr[static_cast<size_t>(m) * k_dim + k]) *
                 bf16_to_f32(read_rhs_bf16(k, n));
        }
        c_ptr[static_cast<size_t>(m) * n_dim + n] = f32_to_bf16(acc);
      }
    }
    out_mn->layout = ot::BufferLayout::kRaw;
    out_mn->tiled_rows = 0;
    out_mn->tiled_cols = 0;
    return OT_OK;
  }

  if (!is_tile_aligned) {
    ctx->last_error = "Live hardware BF16 matmul requires M, K, N to be positive multiples of 128";
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

  std::string err;
  if (!ctx->transport.ensure_imem_program(/*bf16=*/1u, ctx->cfg.timeout_ms, &err)) {
    ctx->last_error = err.empty() ? "IMEM boot initialization failed" : err;
    return transport_err_status(ctx, err);
  }

  HbmScratchGuard temp_lhs{ctx, 0}, temp_rhs{ctx, 0};
  uint64_t lhs_hbm = 0;
  uint64_t rhs_hbm = 0;
  ot_status_t st = prepare_live_bf16_input_locked(
      ctx, lhs_mk, out_mn, m_dim, k_dim, /*transpose_src=*/false,
      &ctx->transport.scratch_lhs_hbm, &temp_lhs, &lhs_hbm, &err);
  if (st != OT_OK) return st;
  if (!rhs_transposed && rhs_kn == lhs_mk && k_dim == m_dim && n_dim == k_dim) {
    rhs_hbm = lhs_hbm;
  } else {
    st = prepare_live_bf16_input_locked(
        ctx, rhs_kn, out_mn, k_dim, n_dim, rhs_transposed,
        &ctx->transport.scratch_rhs_hbm, &temp_rhs, &rhs_hbm, &err);
    if (st != OT_OK) return st;
  }

  st = preserve_tiled_suffix_locked(ctx, out_mn, static_cast<size_t>(need_c));
  if (st != OT_OK) return st;

  uint64_t hw_us = 0u;
  st = launch_v5e_chunked_exec(ctx, lhs_hbm, rhs_hbm, out_mn->hbm_addr, 0ULL, 0ULL,
                               plan.tiles_m, plan.tiles_k, plan.tiles_n,
                               /*quant_mode=*/0u, ot::kTile128x128Bf16Bytes,
                               ot::freetile::kV5eMaxItersPerLaunch, &hw_us, &err);
  if (st != OT_OK) return st;
  const auto t1 = std::chrono::steady_clock::now();

  out_mn->layout = ot::BufferLayout::kTiled2x128Bf16;
  out_mn->tiled_rows = m_dim;
  out_mn->tiled_cols = n_dim;
  if (out_stats) {
    out_stats->hw_us = hw_us;
    out_stats->wall_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - wall_t0).count());
    out_stats->hbm_bytes_moved = plan.hbm_bytes_moved;
    out_stats->tiles_m = plan.tiles_m;
    out_stats->tiles_k = plan.tiles_k;
    out_stats->tiles_n = plan.tiles_n;
    out_stats->imem_bundles = static_cast<uint32_t>(ot::emit_v5e_imem_for_freetile(plan).size());
    out_stats->vmem_bytes_used = plan.vmem.total_vmem_bytes;
  }
  return OT_OK;
}

ot_status_t ot_matmul_quant(ot_context_t* ctx,
                            const ot_buffer_t* lhs_mk,
                            const ot_buffer_t* rhs_quant,
                            const ot_buffer_t* rhs_scales,
                            ot_buffer_t* out_mn,
                            const ot_freetile_gemm_desc_t* desc,
                            ot_exec_stats_t* out_stats) {
  const auto wall_t0 = std::chrono::steady_clock::now();
  if (!desc || desc->m_dim == 0 || desc->k_dim == 0 || desc->n_dim == 0) {
    return OT_ERR_INVALID_ARG;
  }
  if (desc->struct_size != 0u && desc->struct_size != sizeof(ot_freetile_gemm_desc_t)) {
    return OT_ERR_INVALID_ARG;
  }
  if (desc->tile.rhs_transposed > 1u) {
    return OT_ERR_INVALID_ARG;
  }
  ot_status_t st_val = validate_gemm_buffers(ctx, lhs_mk, rhs_quant, out_mn);
  if (st_val != OT_OK) return st_val;
  if (rhs_scales && (rhs_scales->ctx != ctx ||
                     rhs_scales->ctx_instance_id != ctx->instance_id ||
                     is_foreign_forked_buffer(rhs_scales))) {
    return OT_ERR_INVALID_ARG;
  }
  if (desc->rhs_dtype == OT_DTYPE_Q4_K || desc->quant_mode == OT_QUANT_BLOCK_256) {
    std::lock_guard<std::mutex> lk(ctx->mu);
    ctx->last_error = "Q4_K / BLOCK_256 is reserved and not yet supported";
    return OT_ERR_UNSUPPORTED;
  }

  ot::freetile::TileKernelDescriptor fdesc = ot::freetile::TileKernelDescriptor::from_c_desc(*desc);
  ot::freetile::LoweringPlan plan = ot::freetile::plan_v5e_tile_gemm(fdesc);
  if (!plan.valid) {
    std::lock_guard<std::mutex> lk(ctx->mu);
    ctx->last_error = plan.error;
    return OT_ERR_INVALID_ARG;
  }

  const uint32_t m_dim = desc->m_dim;
  const uint32_t k_dim = desc->k_dim;
  const uint32_t n_dim = desc->n_dim;
  const bool rhs_transposed = (desc->tile.rhs_transposed != 0u);
  const bool is_q4 = (desc->rhs_dtype == OT_DTYPE_Q4_0);
  const bool is_i8 = (desc->rhs_dtype == OT_DTYPE_INT8 || desc->rhs_dtype == OT_DTYPE_Q8_0);
  if (!is_q4 && !is_i8) {
    return OT_ERR_INVALID_ARG;
  }

  bool is_block32 = false;
  bool is_per_channel = false;
  bool is_per_tensor = false;
  if (desc->quant_mode == OT_QUANT_BLOCK_32) {
    if (desc->quant_group_size != 0u && desc->quant_group_size != OT_QK8_0) {
      return OT_ERR_INVALID_ARG;
    }
    is_block32 = true;
  } else if (desc->quant_mode == OT_QUANT_PER_CHANNEL) {
    if (desc->quant_group_size != 0u) return OT_ERR_INVALID_ARG;
    is_per_channel = true;
  } else if (desc->quant_mode == OT_QUANT_PER_TENSOR) {
    if (desc->quant_group_size != 0u) return OT_ERR_INVALID_ARG;
    is_per_tensor = true;
  } else if (desc->quant_mode == OT_QUANT_NONE &&
             (desc->rhs_dtype == OT_DTYPE_Q8_0 || desc->rhs_dtype == OT_DTYPE_Q4_0)) {
    if (desc->quant_group_size != 0u && desc->quant_group_size != OT_QK8_0) {
      return OT_ERR_INVALID_ARG;
    }
    is_block32 = true;
  } else {
    return OT_ERR_INVALID_ARG;
  }

  if (desc->scale_dtype != OT_DTYPE_BF16 && desc->scale_dtype != OT_DTYPE_F32) {
    return OT_ERR_INVALID_ARG;
  }

  const bool is_inline_blocks =
      is_block32 && !rhs_scales &&
      (desc->rhs_dtype == OT_DTYPE_Q8_0 || desc->rhs_dtype == OT_DTYPE_Q4_0);
  if (!is_per_tensor && !is_inline_blocks && !rhs_scales) {
    return OT_ERR_INVALID_ARG;
  }

  uint32_t qmode_word = 0u;
  if (is_q4) {
    qmode_word = is_block32
        ? static_cast<uint32_t>(ot::QuantKernelMode::kQ4_0)
        : static_cast<uint32_t>(ot::QuantKernelMode::kInt4PerChannel);
  } else {
    qmode_word = is_block32
        ? static_cast<uint32_t>(ot::QuantKernelMode::kQ8_0)
        : static_cast<uint32_t>(ot::QuantKernelMode::kInt8PerChannel);
  }

  auto checked_mul_u64 = [](uint64_t a, uint64_t b, uint64_t* out) -> bool {
    if (a != 0ULL && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
  };
  const uint64_t k_blocks32 = static_cast<uint64_t>(k_dim) / OT_QK8_0;
  uint64_t mk = 0, kn = 0, mn = 0, need_lhs_u64 = 0, need_q_u64 = 0, need_out_u64 = 0;
  if (!checked_mul_u64(m_dim, k_dim, &mk) ||
      !checked_mul_u64(mk, sizeof(uint16_t), &need_lhs_u64) ||
      !checked_mul_u64(k_dim, n_dim, &kn) ||
      !checked_mul_u64(m_dim, n_dim, &mn) ||
      !checked_mul_u64(mn, sizeof(uint16_t), &need_out_u64)) {
    return OT_ERR_INVALID_ARG;
  }
  if (is_inline_blocks) {
    uint64_t blks = 0;
    const uint64_t blk_sz = is_q4 ? sizeof(ot_block_q4_0_t) : sizeof(ot_block_q8_0_t);
    if (!checked_mul_u64(k_blocks32, n_dim, &blks) ||
        !checked_mul_u64(blks, blk_sz, &need_q_u64)) {
      return OT_ERR_INVALID_ARG;
    }
  } else {
    need_q_u64 = is_q4 ? (kn / 2ULL) : kn;
  }
  const uint64_t scale_groups = is_block32 ? k_blocks32 : 1ULL;
  uint64_t scale_elems = 1ULL;
  if (!is_per_tensor && !checked_mul_u64(scale_groups, n_dim, &scale_elems)) {
    return OT_ERR_INVALID_ARG;
  }
  const bool scales_are_f32 = (desc->scale_dtype == OT_DTYPE_F32);
  uint64_t need_scales_u64 = 0ULL;
  if (rhs_scales &&
      !checked_mul_u64(scale_elems, scales_are_f32 ? sizeof(float) : sizeof(uint16_t),
                       &need_scales_u64)) {
    return OT_ERR_INVALID_ARG;
  }
  const size_t need_lhs = static_cast<size_t>(need_lhs_u64);
  const size_t need_q = static_cast<size_t>(need_q_u64);
  const size_t need_scales = static_cast<size_t>(need_scales_u64);
  const size_t need_out = static_cast<size_t>(need_out_u64);
  if (lhs_mk->size_bytes < need_lhs_u64 || rhs_quant->size_bytes < need_q_u64 ||
      (rhs_scales && rhs_scales->size_bytes < need_scales_u64) ||
      out_mn->size_bytes < need_out_u64) {
    return OT_ERR_INVALID_ARG;
  }

  std::lock_guard<std::mutex> lk(ctx->mu);
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  if (out_stats) std::memset(out_stats, 0, sizeof(*out_stats));

  const size_t block_stride_bytes = is_q4 ? sizeof(ot_block_q4_0_t) : sizeof(ot_block_q8_0_t);
  auto inline_block_ptr = [&](const uint8_t* raw_q, size_t sg, size_t col) -> const uint8_t* {
    const size_t blk_idx = rhs_transposed
        ? (col * static_cast<size_t>(k_blocks32) + sg)
        : (sg * n_dim + col);
    return raw_q + blk_idx * block_stride_bytes;
  };

  auto decode_weight_i8 = [&](const uint8_t* raw_q, size_t k, size_t col) -> int8_t {
    if (is_inline_blocks) {
      const size_t sg = k / OT_QK8_0;
      const size_t ki = k % OT_QK8_0;
      const uint8_t* blk = inline_block_ptr(raw_q, sg, col);
      if (is_q4) {
        const uint8_t packed = blk[sizeof(uint16_t) + (ki >> 1)];
        const uint32_t nib = (ki & 1u) ? (packed >> 4) : (packed & 0xfu);
        return static_cast<int8_t>(static_cast<int32_t>(nib) - 8);
      }
      return static_cast<int8_t>(blk[sizeof(uint16_t) + ki]);
    }
    const size_t elem = rhs_transposed ? (col * k_dim + k) : (k * n_dim + col);
    if (is_q4) {
      const uint8_t packed = raw_q[elem >> 1];
      const uint32_t nib = (elem & 1u) ? (packed >> 4) : (packed & 0xfu);
      return static_cast<int8_t>(static_cast<int32_t>(nib) - 8);
    }
    return static_cast<int8_t>(raw_q[elem]);
  };

  auto decode_scale_bf16 = [&](const uint8_t* raw_q, const uint8_t* raw_sc,
                               size_t sg, size_t col) -> uint16_t {
    if (is_inline_blocks) {
      const uint8_t* blk = inline_block_ptr(raw_q, sg, col);
      uint16_t d_bf16 = 0;
      std::memcpy(&d_bf16, blk, sizeof(uint16_t));
      return d_bf16;
    }
    if (is_per_tensor) {
      if (!rhs_scales) return f32_to_bf16(desc->quant_scale);
      if (scales_are_f32) {
        float f = 0.0f;
        std::memcpy(&f, raw_sc, sizeof(float));
        return f32_to_bf16(f);
      }
      uint16_t b = 0;
      std::memcpy(&b, raw_sc, sizeof(uint16_t));
      return b;
    }
    const size_t idx = (is_per_channel ? 0u : sg) * n_dim + col;
    if (scales_are_f32) {
      float f = 0.0f;
      std::memcpy(&f, raw_sc + idx * sizeof(float), sizeof(float));
      return f32_to_bf16(f);
    }
    uint16_t b = 0;
    std::memcpy(&b, raw_sc + idx * sizeof(uint16_t), sizeof(uint16_t));
    return b;
  };

  if (ctx->info.is_mock) {
    const size_t hbm_sz = ctx->transport.mock_hbm.size();
    auto in_bounds = [&](uint64_t a, size_t len) {
      return a <= hbm_sz && len <= hbm_sz - a;
    };
    if (!in_bounds(lhs_mk->hbm_addr, need_lhs) ||
        !in_bounds(rhs_quant->hbm_addr, need_q) ||
        (rhs_scales && !in_bounds(rhs_scales->hbm_addr, need_scales)) ||
        !in_bounds(out_mn->hbm_addr, need_out)) {
      ctx->last_error = "Mock HBM quant matmul range out of bounds";
      return OT_ERR_DEVICE;
    }
    std::vector<uint16_t> lhs_rm(static_cast<size_t>(m_dim) * k_dim);
    ot_status_t st = read_buffer_row_major_locked(
        ctx, lhs_mk, reinterpret_cast<uint8_t*>(lhs_rm.data()), need_lhs);
    if (st != OT_OK) return st;

    std::vector<uint8_t> q_rm(need_q);
    st = read_buffer_row_major_locked(ctx, rhs_quant, q_rm.data(), need_q);
    if (st != OT_OK) return st;

    std::vector<uint8_t> sc_rm(rhs_scales ? need_scales : 0u);
    if (rhs_scales && need_scales > 0u) {
      st = read_buffer_row_major_locked(ctx, rhs_scales, sc_rm.data(), need_scales);
      if (st != OT_OK) return st;
    }
    const uint8_t* q_ptr = q_rm.data();
    const uint8_t* sc_ptr = rhs_scales ? sc_rm.data() : nullptr;

    std::vector<uint16_t> out_rm(static_cast<size_t>(m_dim) * n_dim);
    std::vector<uint16_t> rhs_deq_bf16(static_cast<size_t>(k_dim) * n_dim);
    for (uint32_t k = 0; k < k_dim; ++k) {
      const size_t sg = k / OT_QK8_0;
      for (uint32_t n = 0; n < n_dim; ++n) {
        const int32_t qv = decode_weight_i8(q_ptr, k, n);
        const float sc = bf16_to_f32(decode_scale_bf16(q_ptr, sc_ptr, sg, n));
        rhs_deq_bf16[static_cast<size_t>(k) * n_dim + n] =
            f32_to_bf16(static_cast<float>(qv) * sc);
      }
    }
    for (uint32_t m = 0; m < m_dim; ++m) {
      for (uint32_t n = 0; n < n_dim; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < k_dim; ++k) {
          acc += bf16_to_f32(lhs_rm[static_cast<size_t>(m) * k_dim + k]) *
                 bf16_to_f32(rhs_deq_bf16[static_cast<size_t>(k) * n_dim + n]);
        }
        out_rm[static_cast<size_t>(m) * n_dim + n] = f32_to_bf16(acc);
      }
    }
    std::vector<uint16_t> out_tiled(static_cast<size_t>(m_dim) * n_dim);
    pack_bf16_matrix_tiles(out_rm.data(), m_dim, n_dim, out_tiled.data());
    st = preserve_tiled_suffix_locked(ctx, out_mn, need_out);
    if (st != OT_OK) return st;
    std::memcpy(ctx->transport.mock_hbm.data() + out_mn->hbm_addr,
                out_tiled.data(), need_out);
    out_mn->layout = ot::BufferLayout::kTiled2x128Bf16;
    out_mn->tiled_rows = m_dim;
    out_mn->tiled_cols = n_dim;
    if (out_stats) {
      const auto wall_t1 = std::chrono::steady_clock::now();
      out_stats->hw_us = 0u;
      out_stats->wall_us = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(wall_t1 - wall_t0).count());
      out_stats->hbm_bytes_moved = plan.hbm_bytes_moved;
      out_stats->tiles_m = plan.tiles_m;
      out_stats->tiles_k = plan.tiles_k;
      out_stats->tiles_n = plan.tiles_n;
      out_stats->vmem_bytes_used = plan.vmem.total_vmem_bytes;
    }
    return OT_OK;
  }

  if ((lhs_mk->hbm_addr & 0x1ffULL) != 0 || (rhs_quant->hbm_addr & 0x1ffULL) != 0 ||
      (rhs_scales && (rhs_scales->hbm_addr & 0x1ffULL) != 0) ||
      (out_mn->hbm_addr & 0x1ffULL) != 0) {
    ctx->last_error = "Hardware quantized matmul HBM buffers must be 512-byte aligned";
    return OT_ERR_INVALID_ARG;
  }
  std::string err;
  if (!ctx->transport.ensure_imem_program(is_q4 ? 3u : 2u, ctx->cfg.timeout_ms, &err) ||
      !ctx->transport.ensure_quant_constants(ctx->cfg.timeout_ms, &err)) {
    ctx->last_error = err.empty() ? "Quantized IMEM setup failed" : err;
    return transport_err_status(ctx, err);
  }

  HbmScratchGuard g_lhs{ctx, 0}, g_q{ctx, 0}, g_sc{ctx, 0};
  uint64_t lhs_hbm = 0;
  ot_status_t st = prepare_live_bf16_input_locked(
      ctx, lhs_mk, out_mn, m_dim, k_dim, /*transpose_src=*/false,
      &ctx->transport.scratch_lhs_hbm, &g_lhs, &lhs_hbm, &err);
  if (st != OT_OK) return st;

  const size_t q_tile_bytes = is_q4 ? 8192u : 16384u;
  const size_t q_tiled_total = static_cast<size_t>(plan.tiles_k) * plan.tiles_n * q_tile_bytes;
  const size_t sc_tile_groups = is_block32 ? plan.tiles_k : 1u;
  const size_t sc_tiled_total = sc_tile_groups * plan.tiles_n * 1024u;
  if (!ctx->transport.hbm_alloc.alloc(q_tiled_total, 4096u, &g_q.addr) ||
      !ctx->transport.hbm_alloc.alloc(sc_tiled_total, 4096u, &g_sc.addr)) {
    return OT_ERR_NO_MEMORY;
  }

  {
    ScopedBuf q_rm(need_q), q_tiled(q_tiled_total);
    ScopedBuf sc_rm(rhs_scales ? need_scales : 4u), sc_tiled(sc_tiled_total);
    if (!q_rm.ptr || !q_tiled.ptr || !sc_rm.ptr || !sc_tiled.ptr) return OT_ERR_NO_MEMORY;
    st = read_buffer_row_major_locked(ctx, rhs_quant, q_rm.ptr, need_q);
    if (st != OT_OK) return st;
    if (rhs_scales && need_scales > 0u) {
      st = read_buffer_row_major_locked(ctx, rhs_scales, sc_rm.ptr, need_scales);
      if (st != OT_OK) return st;
    }

    std::vector<int8_t> tile_i8(128u * 128u);
    std::vector<uint16_t> tile_sc(4u * 128u);
    for (uint32_t kt = 0; kt < plan.tiles_k; ++kt) {
      for (uint32_t nt = 0; nt < plan.tiles_n; ++nt) {
        for (uint32_t r = 0; r < 128u; ++r) {
          const size_t gr = kt * 128u + r;
          for (uint32_t c = 0; c < 128u; ++c) {
            const size_t gc = nt * 128u + c;
            tile_i8[r * 128u + c] = decode_weight_i8(q_rm.ptr, gr, gc);
          }
        }
        const size_t q_off = (static_cast<size_t>(kt) * plan.tiles_n + nt) * q_tile_bytes;
        auto* dst_u32 = reinterpret_cast<uint32_t*>(q_tiled.ptr + q_off);
        if (is_q4) {
          ot::tile_pack_128x128_q4(tile_i8.data(), dst_u32);
        } else {
          ot::tile_pack_128x128_i8(tile_i8.data(), dst_u32);
        }
        if (is_block32 || kt == 0u) {
          for (uint32_t kb = 0; kb < 4u; ++kb) {
            const size_t sg = kt * 4u + kb;
            for (uint32_t c = 0; c < 128u; ++c) {
              tile_sc[kb * 128u + c] =
                  decode_scale_bf16(q_rm.ptr, sc_rm.ptr, sg, nt * 128u + c);
            }
          }
          const size_t sc_off =
              (static_cast<size_t>(is_block32 ? kt : 0u) * plan.tiles_n + nt) * 1024u;
          ot::tile_pack_scales_4x128_bf16(
              tile_sc.data(), reinterpret_cast<uint16_t*>(sc_tiled.ptr + sc_off));
        }
      }
    }
    st = raw_h2d_locked(ctx, g_q.addr, q_tiled_total, 0u, q_tiled.ptr, q_tiled_total);
    if (st != OT_OK) return st;
    st = raw_h2d_locked(ctx, g_sc.addr, sc_tiled_total, 0u, sc_tiled.ptr, sc_tiled_total);
    if (st != OT_OK) return st;
  }

  st = preserve_tiled_suffix_locked(ctx, out_mn, need_out);
  if (st != OT_OK) return st;

  uint64_t hw_us = 0u;
  ot_status_t st_exec = launch_v5e_chunked_exec(
      ctx, lhs_hbm, g_q.addr, out_mn->hbm_addr, g_sc.addr, ctx->transport.quant_const_hbm,
      plan.tiles_m, plan.tiles_k, plan.tiles_n, qmode_word, q_tile_bytes,
      ot::freetile::kV5eMaxItersPerLaunch, &hw_us, &err);
  if (st_exec != OT_OK) return st_exec;
  const auto t1 = std::chrono::steady_clock::now();

  out_mn->layout = ot::BufferLayout::kTiled2x128Bf16;
  out_mn->tiled_rows = m_dim;
  out_mn->tiled_cols = n_dim;
  if (out_stats) {
    out_stats->hw_us = hw_us;
    out_stats->wall_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - wall_t0).count());
    out_stats->hbm_bytes_moved = plan.hbm_bytes_moved;
    out_stats->tiles_m = plan.tiles_m;
    out_stats->tiles_k = plan.tiles_k;
    out_stats->tiles_n = plan.tiles_n;
    out_stats->imem_bundles = static_cast<uint32_t>(ot::emit_v5e_imem_for_freetile(plan).size());
    out_stats->vmem_bytes_used = plan.vmem.total_vmem_bytes;
  }
  return OT_OK;
}

}  // extern "C"
