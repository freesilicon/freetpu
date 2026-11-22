#include "ob_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/ioctl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "edgetpu_buenos_uapi.h"
#endif

namespace ob {

namespace {

#if defined(__linux__)
struct LocalDmaHeapAllocationData {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};

struct LocalDmaBufSync {
  uint64_t flags;
};

#define OB_DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, LocalDmaHeapAllocationData)
#define OB_DMA_BUF_IOCTL_SYNC   _IOW('b', 0x0, LocalDmaBufSync)
#define OB_DMA_BUF_SYNC_READ    (1ULL << 0)
#define OB_DMA_BUF_SYNC_WRITE   (2ULL << 0)
#define OB_DMA_BUF_SYNC_START   (0ULL << 2)
#define OB_DMA_BUF_SYNC_END     (1ULL << 2)

int sync_dmabuf_fd(int dmabuf_fd, uint64_t flags) {
  LocalDmaBufSync req{flags};
  if (::ioctl(dmabuf_fd, OB_DMA_BUF_IOCTL_SYNC, &req) != 0) {
    return -errno;
  }
  return 0;
}
#endif

size_t compute_slab_size(size_t requested) {
  if (requested <= 4096u) return 4096u;
  // Power-of-two slab classes up to 64 KiB for high pool hit rates on small IO/cmd/param
  // buffers, and exact 4 KiB page-aligned slabs above 64 KiB (e.g. 192 KiB / 196 KiB
  // DIVE scratchpads) to avoid 64+ KiB waste per cached graph.
  constexpr size_t kPow2Max = 64u * 1024u;
  if (requested <= kPow2Max) {
    size_t s = 4096u;
    while (s < requested) s <<= 1u;
    return s;
  }
  return (requested + 4095u) & ~static_cast<size_t>(4095u);
}

uint32_t map_flags_from_ob_flags(uint32_t flags) {
#if defined(__linux__)
  uint32_t dir = EDGETPU_MAP_DMA_BIDIRECTIONAL;
  if ((flags & OB_BUF_TO_DEVICE) && !(flags & OB_BUF_FROM_DEVICE)) {
    dir = EDGETPU_MAP_DMA_TO_DEVICE;
  } else if ((flags & OB_BUF_FROM_DEVICE) && !(flags & OB_BUF_TO_DEVICE)) {
    dir = EDGETPU_MAP_DMA_FROM_DEVICE;
  }
  return dir | EDGETPU_MAP_36BIT_IOVA;
#else
  (void)flags;
  return 0x8u;
#endif
}

ob_status_t alloc_raw_slab(ob_context* ctx, size_t slab_size, uint32_t map_flags,
                           PooledSlab* out) {
  *out = PooledSlab{};
  out->alloc_size = slab_size;
  out->map_flags = map_flags;
  out->die_index = 0xffffffffu;

  if (ctx->mock_mode) {
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, 4096u, slab_size) != 0 || !ptr) {
      set_last_error(ctx, "posix_memalign failed in mock buffer alloc");
      return OB_ERR_NO_MEMORY;
    }
    std::memset(ptr, 0, slab_size);
    out->data = static_cast<uint8_t*>(ptr);
    out->dmabuf_fd = -1;
    // Assign unique 4K-aligned synthetic 36-bit IOVA
    out->iova = ctx->next_mock_iova;
    ctx->next_mock_iova += (slab_size + 4095u) & ~static_cast<size_t>(4095u);
    out->iova_mapped = true;
    return OB_OK;
  }

#if defined(__linux__)
  int heap_fd = ctx->dma_heap_fd;
  bool close_heap_fd = false;
  if (heap_fd < 0) {
    heap_fd = ::open(ctx->dma_heap_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (heap_fd < 0) {
      set_last_error(ctx, "open(" + ctx->dma_heap_path + ") failed errno=" +
                              std::to_string(errno));
      return OB_ERR_DEVICE;
    }
    close_heap_fd = true;
  }

  LocalDmaHeapAllocationData req{};
  req.len = slab_size;
  req.fd = 0;
  req.fd_flags = O_RDWR | O_CLOEXEC;
  req.heap_flags = 0;
  if (::ioctl(heap_fd, OB_DMA_HEAP_IOCTL_ALLOC, &req) != 0) {
    int e = errno;
    if (close_heap_fd) ::close(heap_fd);
    set_last_error(ctx, "DMA_HEAP_IOCTL_ALLOC(" + std::to_string(slab_size) +
                            ") failed errno=" + std::to_string(e));
    return OB_ERR_IOCTL;
  }
  if (close_heap_fd) ::close(heap_fd);
  out->dmabuf_fd = static_cast<int>(req.fd);

  void* ptr = ::mmap(nullptr, slab_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     out->dmabuf_fd, 0);
  if (ptr == MAP_FAILED) {
    int e = errno;
    ::close(out->dmabuf_fd);
    out->dmabuf_fd = -1;
    set_last_error(ctx, "mmap(dmabuf) failed errno=" + std::to_string(e));
    return OB_ERR_NO_MEMORY;
  }
  out->data = static_cast<uint8_t*>(ptr);

  struct edgetpu_map_dmabuf_ioctl m {};
  m.offset = 0;
  m.size = slab_size;
  m.dmabuf_fd = out->dmabuf_fd;
  m.flags = map_flags;
  m.die_index = out->die_index;
  if (::ioctl(ctx->tpu_fd, EDGETPU_MAP_DMABUF, &m) != 0) {
    int e = errno;
    ::munmap(out->data, slab_size);
    ::close(out->dmabuf_fd);
    out->data = nullptr;
    out->dmabuf_fd = -1;
    set_last_error(ctx, "EDGETPU_MAP_DMABUF failed errno=" + std::to_string(e));
    return OB_ERR_IOCTL;
  }
  out->iova = m.device_address;
  out->iova_mapped = true;
  return OB_OK;
#else
  set_last_error(ctx, "Non-mock DMA-BUF allocation requires Linux/Android");
  return OB_ERR_UNSUPPORTED;
#endif
}

}  // namespace

void buffer_destroy_slab(ob_context* ctx, PooledSlab* slab) {
  if (!slab) return;
  if (ctx && ctx->mock_mode) {
    if (slab->data) {
      std::free(slab->data);
      slab->data = nullptr;
    }
    slab->iova_mapped = false;
    slab->dmabuf_fd = -1;
    return;
  }
#if defined(__linux__)
  if (ctx && slab->iova_mapped && ctx->tpu_fd >= 0) {
    struct edgetpu_map_dmabuf_ioctl u {};
    u.offset = 0;
    u.size = slab->alloc_size;
    u.device_address = slab->iova;
    u.dmabuf_fd = slab->dmabuf_fd;
    u.flags = slab->map_flags;
    u.die_index = slab->die_index;
    ::ioctl(ctx->tpu_fd, EDGETPU_UNMAP_DMABUF, &u);
    slab->iova_mapped = false;
  }
  if (slab->data && slab->data != MAP_FAILED) {
    ::munmap(slab->data, slab->alloc_size);
    slab->data = nullptr;
  }
  if (slab->dmabuf_fd >= 0) {
    ::close(slab->dmabuf_fd);
    slab->dmabuf_fd = -1;
  }
#endif
}

void buffer_pool_clear(ob_context* ctx) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lk(ctx->pool_mu);
  for (auto& slab : ctx->pool_lru) {
    buffer_destroy_slab(ctx, &slab);
  }
  ctx->pool_lru.clear();
  ctx->pool_buckets.clear();
  ctx->pool_current_bytes = 0;
  ctx->pool_current_count = 0;
}

ob_status_t buffer_sync_begin_read(ob_buffer* buf) {
  if (!buf || !buf->ctx) return OB_ERR_INVALID_ARG;
  if (buf->ctx->mock_mode) return OB_OK;
#if defined(__linux__)
  if (sync_dmabuf_fd(buf->slab.dmabuf_fd,
                     OB_DMA_BUF_SYNC_START | OB_DMA_BUF_SYNC_READ) != 0) {
    set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(START|READ) failed");
    return OB_ERR_IOCTL;
  }
  return OB_OK;
#else
  return OB_ERR_UNSUPPORTED;
#endif
}

ob_status_t buffer_sync_end_read(ob_buffer* buf) {
  if (!buf || !buf->ctx) return OB_ERR_INVALID_ARG;
  if (buf->ctx->mock_mode) return OB_OK;
#if defined(__linux__)
  if (sync_dmabuf_fd(buf->slab.dmabuf_fd,
                     OB_DMA_BUF_SYNC_END | OB_DMA_BUF_SYNC_READ) != 0) {
    set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(END|READ) failed");
    return OB_ERR_IOCTL;
  }
  return OB_OK;
#else
  return OB_ERR_UNSUPPORTED;
#endif
}

ob_status_t buffer_sync_begin_write(ob_buffer* buf) {
  if (!buf || !buf->ctx) return OB_ERR_INVALID_ARG;
  if (buf->ctx->mock_mode) return OB_OK;
#if defined(__linux__)
  if (sync_dmabuf_fd(buf->slab.dmabuf_fd,
                     OB_DMA_BUF_SYNC_START | OB_DMA_BUF_SYNC_WRITE) != 0) {
    set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(START|WRITE) failed");
    return OB_ERR_IOCTL;
  }
  return OB_OK;
#else
  return OB_ERR_UNSUPPORTED;
#endif
}

ob_status_t buffer_sync_end_write(ob_buffer* buf) {
  if (!buf || !buf->ctx) return OB_ERR_INVALID_ARG;
  if (buf->ctx->mock_mode) return OB_OK;
#if defined(__linux__)
  if (sync_dmabuf_fd(buf->slab.dmabuf_fd,
                     OB_DMA_BUF_SYNC_END | OB_DMA_BUF_SYNC_WRITE) != 0) {
    set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(END|WRITE) failed");
    return OB_ERR_IOCTL;
  }
  return OB_OK;
#else
  return OB_ERR_UNSUPPORTED;
#endif
}

void buffer_retain(ob_buffer* buf) {
  if (!buf) return;
  buf->refcnt.fetch_add(1u, std::memory_order_relaxed);
}

}  // namespace ob

extern "C" {

ob_status_t ob_buffer_alloc(ob_context_t* ctx,
                            size_t size_bytes,
                            uint32_t flags,
                            ob_buffer_t** out_buf) {
  if (!ctx || !out_buf || size_bytes == 0) return OB_ERR_INVALID_ARG;
  *out_buf = nullptr;

  const size_t slab_size = ob::compute_slab_size(size_bytes);
  // Always map bidirectional when pooled so any future caller can reuse the slab safely.
  const uint32_t map_flags =
      (flags & OB_BUF_NO_POOL)
          ? ob::map_flags_from_ob_flags(flags)
          : ob::map_flags_from_ob_flags(OB_BUF_BIDIRECTIONAL);

  ob::PooledSlab slab{};
  bool got_pooled = false;

  if (!(flags & OB_BUF_NO_POOL)) {
    std::lock_guard<std::mutex> lk(ctx->pool_mu);
    auto it = ctx->pool_buckets.find(slab_size);
    if (it == ctx->pool_buckets.end() || it->second.empty()) {
      // Best-fit fallback: reuse a slightly larger pooled slab (<= 2x) to avoid DMA-BUF churn.
      size_t best_sz = 0u;
      auto best_it = ctx->pool_buckets.end();
      const size_t max_reuse = slab_size * 2u;
      for (auto bit = ctx->pool_buckets.begin(); bit != ctx->pool_buckets.end(); ++bit) {
        if (!bit->second.empty() && bit->first >= slab_size && bit->first <= max_reuse) {
          if (best_sz == 0u || bit->first < best_sz) {
            best_sz = bit->first;
            best_it = bit;
          }
        }
      }
      it = best_it;
    }
    if (it != ctx->pool_buckets.end() && !it->second.empty()) {
      auto lru_it = it->second.back();
      it->second.pop_back();
      slab = *lru_it;
      ctx->pool_lru.erase(lru_it);
      ctx->pool_current_bytes -= slab.alloc_size;
      ctx->pool_current_count -= 1u;
      ctx->pool_hits += 1u;
      got_pooled = true;
    } else {
      ctx->pool_misses += 1u;
    }
  }

  if (!got_pooled) {
    std::lock_guard<std::mutex> lk(ctx->pool_mu);
    ob_status_t st = ob::alloc_raw_slab(ctx, slab_size, map_flags, &slab);
    if (st != OB_OK) return st;
  }

  ob_buffer* buf = new (std::nothrow) ob_buffer();
  if (!buf) {
    ob::buffer_destroy_slab(ctx, &slab);
    return OB_ERR_NO_MEMORY;
  }
  buf->ctx = ctx;
  buf->slab = slab;
  buf->requested_size = size_bytes;
  buf->flags = flags;

  if (flags & OB_BUF_ZERO_INIT) {
#if defined(__linux__)
    if (!ctx->mock_mode) {
      ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                         OB_DMA_BUF_SYNC_START | OB_DMA_BUF_SYNC_WRITE);
    }
#endif
    std::memset(buf->slab.data, 0, buf->slab.alloc_size);
#if defined(__linux__)
    if (!ctx->mock_mode) {
      ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                         OB_DMA_BUF_SYNC_END | OB_DMA_BUF_SYNC_WRITE);
    }
#endif
  }

  *out_buf = buf;
  return OB_OK;
}

void ob_buffer_free(ob_buffer_t* buf) {
  if (!buf) return;
  if (buf->refcnt.fetch_sub(1u, std::memory_order_acq_rel) > 1u) {
    return;
  }
  ob_context* ctx = buf->ctx;
  ob::PooledSlab slab = buf->slab;
  uint32_t flags = buf->flags;
  delete buf;

  if (!ctx) return;
  if ((flags & OB_BUF_NO_POOL) || ctx->pool_max_bytes == 0 ||
      slab.alloc_size > ctx->pool_max_bytes) {
    ob::buffer_destroy_slab(ctx, &slab);
    return;
  }

  std::lock_guard<std::mutex> lk(ctx->pool_mu);
  // Evict oldest slabs until the returned slab fits within pool_max_bytes and FD-safe count cap
  while (!ctx->pool_lru.empty() &&
         (ctx->pool_current_bytes + slab.alloc_size > ctx->pool_max_bytes ||
          ctx->pool_current_count >= ob::kDefaultBufferPoolMaxCount)) {
    auto victim_it = std::prev(ctx->pool_lru.end());
    size_t vsz = victim_it->alloc_size;
    auto& bucket = ctx->pool_buckets[vsz];
    for (auto bit = bucket.begin(); bit != bucket.end(); ++bit) {
      if (*bit == victim_it) {
        bucket.erase(bit);
        break;
      }
    }
    ob::PooledSlab victim = *victim_it;
    ctx->pool_lru.erase(victim_it);
    ctx->pool_current_bytes -= victim.alloc_size;
    ctx->pool_current_count -= 1u;
    ob::buffer_destroy_slab(ctx, &victim);
  }

  ctx->pool_lru.push_front(slab);
  ctx->pool_buckets[slab.alloc_size].push_back(ctx->pool_lru.begin());
  ctx->pool_current_bytes += slab.alloc_size;
  ctx->pool_current_count += 1u;
}

void* ob_buffer_data(ob_buffer_t* buf) {
  return buf ? static_cast<void*>(buf->slab.data) : nullptr;
}

size_t ob_buffer_size(const ob_buffer_t* buf) {
  return buf ? buf->requested_size : 0u;
}

uint64_t ob_buffer_iova(const ob_buffer_t* buf) {
  return buf ? buf->slab.iova : 0u;
}

ob_status_t ob_buffer_sync_to_device(ob_buffer_t* buf) {
  if (!buf || !buf->ctx) return OB_ERR_INVALID_ARG;
  if (buf->ctx->mock_mode) return OB_OK;
#if defined(__linux__)
  if (ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                         OB_DMA_BUF_SYNC_START | OB_DMA_BUF_SYNC_WRITE) != 0 ||
      ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                         OB_DMA_BUF_SYNC_END | OB_DMA_BUF_SYNC_WRITE) != 0) {
    ob::set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(WRITE) failed");
    return OB_ERR_IOCTL;
  }
  return OB_OK;
#else
  return OB_ERR_UNSUPPORTED;
#endif
}

ob_status_t ob_buffer_sync_to_cpu(ob_buffer_t* buf) {
  if (!buf || !buf->ctx) return OB_ERR_INVALID_ARG;
  if (buf->ctx->mock_mode) return OB_OK;
#if defined(__linux__)
  if (ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                         OB_DMA_BUF_SYNC_START | OB_DMA_BUF_SYNC_READ) != 0 ||
      ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                         OB_DMA_BUF_SYNC_END | OB_DMA_BUF_SYNC_READ) != 0) {
    ob::set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(READ) failed");
    return OB_ERR_IOCTL;
  }
  return OB_OK;
#else
  return OB_ERR_UNSUPPORTED;
#endif
}

ob_status_t ob_buffer_write(ob_buffer_t* buf, const void* src, size_t bytes, size_t offset) {
  if (!buf || !buf->ctx) return OB_ERR_INVALID_ARG;
  if (offset > buf->slab.alloc_size || bytes > buf->slab.alloc_size - offset) {
    return OB_ERR_OVERFLOW;
  }
  if (bytes > 0 && !src) return OB_ERR_INVALID_ARG;

#if defined(__linux__)
  if (!buf->ctx->mock_mode) {
    if (ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                           OB_DMA_BUF_SYNC_START | OB_DMA_BUF_SYNC_WRITE) != 0) {
      ob::set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(START|WRITE) failed");
      return OB_ERR_IOCTL;
    }
  }
#endif
  if (bytes > 0) {
    std::memcpy(buf->slab.data + offset, src, bytes);
  }
#if defined(__linux__)
  if (!buf->ctx->mock_mode) {
    if (ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                           OB_DMA_BUF_SYNC_END | OB_DMA_BUF_SYNC_WRITE) != 0) {
      ob::set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(END|WRITE) failed");
      return OB_ERR_IOCTL;
    }
  }
#endif
  return OB_OK;
}

ob_status_t ob_buffer_read(ob_buffer_t* buf, void* dst, size_t bytes, size_t offset) {
  if (!buf || !buf->ctx) return OB_ERR_INVALID_ARG;
  if (offset > buf->slab.alloc_size || bytes > buf->slab.alloc_size - offset) {
    return OB_ERR_OVERFLOW;
  }
  if (bytes > 0 && !dst) return OB_ERR_INVALID_ARG;

#if defined(__linux__)
  if (!buf->ctx->mock_mode) {
    if (ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                           OB_DMA_BUF_SYNC_START | OB_DMA_BUF_SYNC_READ) != 0) {
      ob::set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(START|READ) failed");
      return OB_ERR_IOCTL;
    }
  }
#endif
  if (bytes > 0) {
    std::memcpy(dst, buf->slab.data + offset, bytes);
  }
#if defined(__linux__)
  if (!buf->ctx->mock_mode) {
    if (ob::sync_dmabuf_fd(buf->slab.dmabuf_fd,
                           OB_DMA_BUF_SYNC_END | OB_DMA_BUF_SYNC_READ) != 0) {
      ob::set_last_error(buf->ctx, "DMA_BUF_IOCTL_SYNC(END|READ) failed");
      return OB_ERR_IOCTL;
    }
  }
#endif
  return OB_OK;
}

ob_status_t ob_context_buffer_pool_stats(const ob_context_t* ctx,
                                         size_t* out_pooled_bytes,
                                         size_t* out_pooled_count,
                                         uint64_t* out_pool_hits,
                                         uint64_t* out_pool_misses) {
  if (!ctx) return OB_ERR_INVALID_ARG;
  std::lock_guard<std::mutex> lk(const_cast<ob_context*>(ctx)->pool_mu);
  if (out_pooled_bytes) *out_pooled_bytes = ctx->pool_current_bytes;
  if (out_pooled_count) *out_pooled_count = ctx->pool_current_count;
  if (out_pool_hits) *out_pool_hits = ctx->pool_hits;
  if (out_pool_misses) *out_pool_misses = ctx->pool_misses;
  return OB_OK;
}

}  // extern "C"
