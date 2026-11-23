// SPDX-License-Identifier: Apache-2.0
#include "ot_internal.h"
#include "ot_vliw.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

namespace {

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

}  // namespace

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
  ctx->cfg = cfg;

  std::string err;
  if (!ctx->transport.open_device(cfg, &ctx->info, &err)) {
    g_last_create_error = err.empty() ? "VFIO device open failed" : err;
    delete ctx;
    return OT_ERR_DEVICE;
  }

  if (!cfg.mock_mode && ctx->transport.bar4_mmio && ctx->transport.bar4_mapped_len >= 0x1000) {
    std::vector<uint8_t> page(4096, 0);
    for (size_t i = 0; i < page.size(); i += 4) {
      uint32_t w = ctx->transport.read32_bar4(i);
      std::memcpy(page.data() + i, &w, 4);
    }
    ot::parse_bar4_descriptor(page.data(), page.size(), &ctx->bar4_desc);
  }

  *out_ctx = ctx;
  return OT_OK;
}

void ot_context_destroy(ot_context_t* ctx) {
  if (!ctx) return;
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
  if (!ctx || !out_buf || size_bytes == 0) return OT_ERR_INVALID_ARG;
  if (size_bytes > UINT64_MAX - 4095ULL) return OT_ERR_INVALID_ARG;
  *out_buf = nullptr;

  std::lock_guard<std::mutex> lk(ctx->mu);
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    ctx->last_error = "VFIO context cannot be used in a forked child process";
    return OT_ERR_DEVICE;
  }
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
  buf->size_bytes = size_bytes;
  buf->hbm_addr = hbm_addr;
  if (ctx->info.is_mock) {
    buf->mock_data.assign(size_bytes, 0);
  }
  ctx->live_buffers.push_back(buf);
  *out_buf = buf;
  return OT_OK;
}

void ot_buffer_free(ot_buffer_t* buf) {
  if (!buf) return;
  if (buf->ctx) {
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
  if (!ctx || !dst || !host_src || dst->ctx != ctx || dst_offset > dst->size_bytes ||
      size_bytes > dst->size_bytes - dst_offset) {
    return OT_ERR_INVALID_ARG;
  }
  if (size_bytes == 0) return OT_OK;
  std::lock_guard<std::mutex> lk(ctx->mu);
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    ctx->last_error = "VFIO context cannot be used in a forked child process";
    return OT_ERR_DEVICE;
  }
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  if (ctx->info.is_mock) {
    std::memcpy(dst->mock_data.data() + dst_offset, host_src, size_bytes);
    if (dst->hbm_addr + dst_offset + size_bytes <= ctx->transport.mock_hbm.size()) {
      std::memcpy(ctx->transport.mock_hbm.data() + dst->hbm_addr + dst_offset, host_src, size_bytes);
    }
    return OT_OK;
  }

  if ((dst_offset % 8u) != 0) {
    ctx->last_error = "Hardware H2D destination offset must be 8-byte aligned";
    return OT_ERR_INVALID_ARG;
  }
  if (!ctx->transport.host_stage_mem || ctx->transport.host_stage_size == 0) {
    ctx->last_error = "DMA staging buffer not initialized";
    return OT_ERR_DEVICE;
  }

  const auto* src_bytes = static_cast<const uint8_t*>(host_src);
  size_t done = 0;
  std::string err;
  while (done < size_bytes) {
    size_t chunk = std::min(size_bytes - done, ctx->transport.host_stage_size);
    size_t dma_len = (chunk + 7u) & ~static_cast<size_t>(7u);
    if (dma_len > ctx->transport.host_stage_size) {
      chunk = ctx->transport.host_stage_size;
      dma_len = chunk & ~static_cast<size_t>(7u);
    }
    if (dma_len > chunk) {
      if (dst_offset + done + chunk < dst->size_bytes) {
        // Partial H2D write ending mid-qword inside a larger buffer: read-modify-write
        // the trailing 8-byte HBM word so untouched suffix bytes are preserved.
        size_t tail_off = chunk & ~static_cast<size_t>(7u);
        if (!ctx->transport.dma_transfer_sync(
                ctx->transport.host_stage_iova + tail_off,
                dst->hbm_addr + dst_offset + done + tail_off, 8u,
                /*is_h2d=*/false, 0x40000000u, 0x60u, 0u, ctx->cfg.timeout_ms, &err)) {
          ctx->last_error = err.empty() ? "Hardware H2D tail preserve D2H failed" : err;
          return ctx->transport.dma_faulted ? OT_ERR_TIMEOUT : OT_ERR_IOCTL;
        }
      } else {
        std::memset(ctx->transport.host_stage_mem + chunk, 0, dma_len - chunk);
      }
    }
    std::memcpy(ctx->transport.host_stage_mem, src_bytes + done, chunk);
    if (!ctx->transport.dma_transfer_sync(
            ctx->transport.host_stage_iova, dst->hbm_addr + dst_offset + done, dma_len,
            /*is_h2d=*/true, 0x40000000u, 0x10u, 0u, ctx->cfg.timeout_ms, &err)) {
      ctx->last_error = err.empty() ? "Hardware H2D DMA failed" : err;
      return ctx->transport.dma_faulted ? OT_ERR_TIMEOUT : OT_ERR_IOCTL;
    }
    done += chunk;
  }
  return OT_OK;
}

}  // namespace ot

extern "C" {

ot_status_t ot_buffer_h2d(ot_context_t* ctx, ot_buffer_t* dst,
                          const void* host_src, size_t size_bytes) {
  return ot::buffer_h2d_offset(ctx, dst, 0, host_src, size_bytes);
}

ot_status_t ot_buffer_d2h(ot_context_t* ctx, const ot_buffer_t* src,
                          void* host_dst, size_t size_bytes) {
  if (!ctx || !src || !host_dst || src->ctx != ctx || size_bytes > src->size_bytes) {
    return OT_ERR_INVALID_ARG;
  }
  if (size_bytes == 0) return OT_OK;
  std::lock_guard<std::mutex> lk(ctx->mu);
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    ctx->last_error = "VFIO context cannot be used in a forked child process";
    return OT_ERR_DEVICE;
  }
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  if (ctx->info.is_mock) {
    if (src->hbm_addr + size_bytes <= ctx->transport.mock_hbm.size()) {
      std::memcpy(host_dst, ctx->transport.mock_hbm.data() + src->hbm_addr, size_bytes);
    } else {
      std::memcpy(host_dst, src->mock_data.data(), size_bytes);
    }
    return OT_OK;
  }

  if (!ctx->transport.host_stage_mem || ctx->transport.host_stage_size == 0) {
    ctx->last_error = "DMA staging buffer not initialized";
    return OT_ERR_DEVICE;
  }

  auto* dst_bytes = static_cast<uint8_t*>(host_dst);
  size_t done = 0;
  std::string err;
  while (done < size_bytes) {
    size_t chunk = std::min(size_bytes - done, ctx->transport.host_stage_size);
    size_t dma_len = (chunk + 7u) & ~static_cast<size_t>(7u);
    if (dma_len > ctx->transport.host_stage_size) {
      chunk = ctx->transport.host_stage_size;
      dma_len = chunk & ~static_cast<size_t>(7u);
    }
    if (!ctx->transport.dma_transfer_sync(
            ctx->transport.host_stage_iova, src->hbm_addr + done, dma_len,
            /*is_h2d=*/false, 0x40000000u, 0x60u, 0u, ctx->cfg.timeout_ms, &err)) {
      ctx->last_error = err.empty() ? "Hardware D2H DMA failed" : err;
      return ctx->transport.dma_faulted ? OT_ERR_TIMEOUT : OT_ERR_IOCTL;
    }
    std::memcpy(dst_bytes + done, ctx->transport.host_stage_mem, chunk);
    done += chunk;
  }
  return OT_OK;
}

ot_status_t ot_matmul_bf16(ot_context_t* ctx,
                           const ot_buffer_t* lhs_mk,
                           const ot_buffer_t* rhs_kn,
                           ot_buffer_t* out_mn,
                           uint32_t m_dim,
                           uint32_t k_dim,
                           uint32_t n_dim) {
  if (!ctx || !lhs_mk || !rhs_kn || !out_mn ||
      lhs_mk->ctx != ctx || rhs_kn->ctx != ctx || out_mn->ctx != ctx ||
      m_dim == 0 || k_dim == 0 || n_dim == 0) {
    return OT_ERR_INVALID_ARG;
  }
  uint64_t need_a = static_cast<uint64_t>(m_dim) * k_dim * sizeof(uint16_t);
  uint64_t need_b = static_cast<uint64_t>(k_dim) * n_dim * sizeof(uint16_t);
  uint64_t need_c = static_cast<uint64_t>(m_dim) * n_dim * sizeof(uint16_t);
  if (lhs_mk->size_bytes < need_a || rhs_kn->size_bytes < need_b || out_mn->size_bytes < need_c) {
    return OT_ERR_INVALID_ARG;
  }
  auto ranges_overlap = [](const ot_buffer_t* x, const ot_buffer_t* y) -> bool {
    if (x == y) return true;
    if (x->size_bytes == 0 || y->size_bytes == 0) return false;
    uint64_t x_end = x->hbm_addr + x->size_bytes;
    uint64_t y_end = y->hbm_addr + y->size_bytes;
    return x->hbm_addr < y_end && y->hbm_addr < x_end;
  };
  if (ranges_overlap(out_mn, lhs_mk) || ranges_overlap(out_mn, rhs_kn)) {
    return OT_ERR_INVALID_ARG;
  }

  std::lock_guard<std::mutex> lk(ctx->mu);
  if (!ctx->info.is_mock && !ctx->transport.is_owner_process()) {
    ctx->last_error = "VFIO context cannot be used in a forked child process";
    return OT_ERR_DEVICE;
  }
  if (ctx->transport.dma_faulted) {
    ctx->last_error = "Context faulted after prior DMA timeout; recreate context";
    return OT_ERR_DEVICE;
  }
  if (ctx->info.is_mock) {
    const auto* a = reinterpret_cast<const uint16_t*>(ctx->transport.mock_hbm.data() + lhs_mk->hbm_addr);
    const auto* b = reinterpret_cast<const uint16_t*>(ctx->transport.mock_hbm.data() + rhs_kn->hbm_addr);
    auto* c = reinterpret_cast<uint16_t*>(ctx->transport.mock_hbm.data() + out_mn->hbm_addr);
    for (uint32_t m = 0; m < m_dim; ++m) {
      for (uint32_t n = 0; n < n_dim; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < k_dim; ++k) {
          acc += bf16_to_f32(a[static_cast<size_t>(m) * k_dim + k]) *
                 bf16_to_f32(b[static_cast<size_t>(k) * n_dim + n]);
        }
        c[static_cast<size_t>(m) * n_dim + n] = f32_to_bf16(acc);
      }
    }
    return OT_OK;
  }

  // Clean-room 58-bundle 128x128 BF16 MXU kernel body + 512B launch ABI live in ot_vliw.h;
  // full self-contained IMEM boot replacement (without vendor blobs) remains tracked for M3 completion.
  auto body = ot::emit_v5e_matmul_128x128_body();
  (void)body;
  ctx->last_error = "Hardware MXU launch requires standalone clean-room IMEM boot stub (M3 partial: DMA+kernel body ready)";
  return OT_ERR_UNSUPPORTED;
}

}  // extern "C"
