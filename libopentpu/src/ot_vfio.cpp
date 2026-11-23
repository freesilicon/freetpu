// SPDX-License-Identifier: Apache-2.0
#include "ot_internal.h"
#include "ot_vliw.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __linux__
#include <dirent.h>
#include <fcntl.h>
#include <linux/vfio.h>
#include <sys/ioctl.h>
#endif

namespace ot {

bool parse_hex_u64(const std::string& text, uint64_t* out) {
  if (!out || text.empty()) return false;
  char* end = nullptr;
  errno = 0;
  unsigned long long v = std::strtoull(text.c_str(), &end, 0);
  if (errno != 0 || end == text.c_str()) return false;
  *out = static_cast<uint64_t>(v);
  return true;
}

void IovaAllocator::init(uint64_t base, uint64_t limit) {
  base_ = base;
  limit_ = limit;
  next_ = base;
  blocks_.clear();
}

bool IovaAllocator::alloc(uint64_t size, uint64_t align, uint64_t* out_iova) {
  if (!out_iova || size == 0) return false;
  if (align == 0) align = 4096;
  if ((align & (align - 1)) != 0) return false;

  // Reuse a free block if big enough, splitting unused leading/trailing space.
  for (size_t i = 0; i < blocks_.size(); ++i) {
    if (blocks_[i].in_use) continue;
    uint64_t b_start = blocks_[i].iova;
    uint64_t b_size = blocks_[i].size;
    if (b_size < size) continue;
    uint64_t aligned = (b_start + (align - 1)) & ~(align - 1);
    if (aligned < b_start) continue;
    uint64_t pad = aligned - b_start;
    if (pad > b_size || size > b_size - pad) continue;

    uint64_t rem = (b_size - pad) - size;
    if (pad > 0) {
      blocks_[i].size = pad;
      blocks_.insert(blocks_.begin() + static_cast<ptrdiff_t>(i + 1),
                     IovaBlock{aligned, size, true});
      ++i;
    } else {
      blocks_[i].iova = aligned;
      blocks_[i].size = size;
      blocks_[i].in_use = true;
    }
    if (rem > 0) {
      blocks_.insert(blocks_.begin() + static_cast<ptrdiff_t>(i + 1),
                     IovaBlock{aligned + size, rem, false});
    }
    *out_iova = aligned;
    return true;
  }

  uint64_t aligned = (next_ + (align - 1)) & ~(align - 1);
  if (aligned < next_ || aligned > limit_ || size > limit_ - aligned) {
    return false;
  }
  if (aligned > next_) {
    blocks_.push_back({next_, aligned - next_, false});
  }
  blocks_.push_back({aligned, size, true});
  next_ = aligned + size;
  *out_iova = aligned;
  return true;
}

void IovaAllocator::free(uint64_t iova) {
  for (size_t i = 0; i < blocks_.size(); ++i) {
    if (blocks_[i].in_use && blocks_[i].iova == iova) {
      blocks_[i].in_use = false;
      // Coalesce with next free block if contiguous.
      if (i + 1 < blocks_.size() && !blocks_[i + 1].in_use &&
          blocks_[i].iova + blocks_[i].size == blocks_[i + 1].iova) {
        blocks_[i].size += blocks_[i + 1].size;
        blocks_.erase(blocks_.begin() + static_cast<ptrdiff_t>(i + 1));
      }
      // Coalesce with previous free block if contiguous.
      if (i > 0 && !blocks_[i - 1].in_use &&
          blocks_[i - 1].iova + blocks_[i - 1].size == blocks_[i].iova) {
        blocks_[i - 1].size += blocks_[i].size;
        blocks_.erase(blocks_.begin() + static_cast<ptrdiff_t>(i));
      }
      // Trim trailing free blocks back into the bump frontier.
      while (!blocks_.empty() && !blocks_.back().in_use &&
             blocks_.back().iova + blocks_.back().size == next_) {
        next_ = blocks_.back().iova;
        blocks_.pop_back();
      }
      return;
    }
  }
}

uint64_t IovaAllocator::used_bytes() const {
  uint64_t total = 0;
  for (const auto& b : blocks_) {
    if (b.in_use) total += b.size;
  }
  return total;
}

static bool read_varint(const uint8_t* p, size_t len, size_t* pos, uint64_t* out) {
  uint64_t val = 0;
  int shift = 0;
  while (*pos < len && shift < 64) {
    uint8_t b = p[(*pos)++];
    val |= static_cast<uint64_t>(b & 0x7fu) << shift;
    if ((b & 0x80u) == 0) {
      *out = val;
      return true;
    }
    shift += 7;
  }
  return false;
}

bool parse_bar4_descriptor(const uint8_t* bar4_page, size_t page_len, Bar4Descriptor* out) {
  if (!bar4_page || !out || page_len < 0x20) return false;
  *out = Bar4Descriptor{};

  uint32_t proto_len = 0;
  uint32_t hdr_word = 0;
  std::memcpy(&proto_len, bar4_page + 0x08, 4);
  std::memcpy(&hdr_word, bar4_page + 0x0c, 4);
  if (proto_len == 0 || proto_len > page_len - 0x10) {
    return false;
  }

  out->proto_len = proto_len;
  out->header_word = hdr_word;

  const uint8_t* p = bar4_page + 0x10;
  size_t pos = 0;
  while (pos < proto_len) {
    uint64_t tag = 0;
    if (!read_varint(p, proto_len, &pos, &tag)) return false;
    uint32_t field_num = static_cast<uint32_t>(tag >> 3);
    uint32_t wire = static_cast<uint32_t>(tag & 7u);
    if (field_num == 0) return false;
    if (wire == 0) {
      uint64_t v = 0;
      if (!read_varint(p, proto_len, &pos, &v)) return false;
    } else if (wire == 1) {
      if (pos + 8 > proto_len) return false;
      pos += 8;
    } else if (wire == 2) {
      uint64_t slen = 0;
      if (!read_varint(p, proto_len, &pos, &slen)) return false;
      if (slen > proto_len - pos) return false;
      if (field_num == 2) {
        out->core_count_observed++;
      } else if (field_num == 3 && slen > 0) {
        out->instance_token.assign(reinterpret_cast<const char*>(p + pos),
                                   static_cast<size_t>(slen));
      }
      pos += static_cast<size_t>(slen);
    } else if (wire == 5) {
      if (pos + 4 > proto_len) return false;
      pos += 4;
    } else {
      return false;
    }
  }
  out->valid = true;
  return true;
}

void* alloc_aligned_anon_pages(size_t len, size_t align, std::string* err) {
  if (len == 0) {
    if (err) *err = "DMA arena length must be non-zero";
    return nullptr;
  }
  long host_ps_raw = ::sysconf(_SC_PAGESIZE);
  const size_t host_ps = (host_ps_raw > 0) ? static_cast<size_t>(host_ps_raw) : 4096u;
  if ((host_ps & (host_ps - 1u)) != 0) {
    if (err) *err = "Host page size is not a power of two";
    return nullptr;
  }
  if (align == 0) align = host_ps;
  if ((align & (align - 1u)) != 0) {
    if (err) *err = "DMA alignment must be a power of two";
    return nullptr;
  }
  if (align < host_ps) align = host_ps;
  if ((len % host_ps) != 0) {
    if (err) *err = "DMA arena length must be a multiple of host page size";
    return nullptr;
  }

  if (align <= host_ps) {
    void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
      if (err) *err = std::string("mmap failed: ") + std::strerror(errno);
      return nullptr;
    }
    if ((reinterpret_cast<uintptr_t>(p) & static_cast<uintptr_t>(align - 1u)) != 0) {
      ::munmap(p, len);
      if (err) *err = "mmap returned misaligned backing page";
      return nullptr;
    }
    return p;
  }

  if (len > std::numeric_limits<size_t>::max() - align) {
    if (err) *err = "DMA arena aligned allocation overflow";
    return nullptr;
  }
  const size_t over_len = len + align;
  void* raw = ::mmap(nullptr, over_len, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) {
    if (err) *err = std::string("mmap failed: ") + std::strerror(errno);
    return nullptr;
  }
  const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw);
  const uintptr_t mask = static_cast<uintptr_t>(align - 1u);
  const uintptr_t aligned_addr = (raw_addr + mask) & ~mask;
  if (aligned_addr < raw_addr) {
    ::munmap(raw, over_len);
    if (err) *err = "DMA arena aligned address wraparound";
    return nullptr;
  }
  const size_t lead = static_cast<size_t>(aligned_addr - raw_addr);
  if (lead > over_len || over_len - lead < len) {
    ::munmap(raw, over_len);
    if (err) *err = "DMA arena aligned subrange out of bounds";
    return nullptr;
  }
  const size_t trail = over_len - lead - len;
  if (lead > 0) (void)::munmap(raw, lead);
  if (trail > 0) (void)::munmap(reinterpret_cast<void*>(aligned_addr + len), trail);
  return reinterpret_cast<void*>(aligned_addr);
}

namespace {

pthread_mutex_t g_vfio_fork_mu = PTHREAD_MUTEX_INITIALIZER;
pthread_once_t g_vfio_atfork_once = PTHREAD_ONCE_INIT;
constexpr size_t kMaxTrackedVfioTransports = 64;
VfioTransport* g_tracked_transports[kMaxTrackedVfioTransports] = {};
bool g_vfio_atfork_registered = false;

void vfio_atfork_prepare() {
  pthread_mutex_lock(&g_vfio_fork_mu);
}

void vfio_atfork_parent() {
  pthread_mutex_unlock(&g_vfio_fork_mu);
}

void vfio_atfork_child() {
  for (size_t i = 0; i < kMaxTrackedVfioTransports; ++i) {
    VfioTransport* t = g_tracked_transports[i];
    if (!t) continue;
    if (t->device_fd >= 0) {
      ::close(t->device_fd);
      t->device_fd = -1;
    }
    if (t->group_fd >= 0) {
      ::close(t->group_fd);
      t->group_fd = -1;
    }
    if (t->container_fd >= 0) {
      ::close(t->container_fd);
      t->container_fd = -1;
    }
    // BAR0/BAR4 and DMA arenas are marked MADV_DONTFORK so they are not mapped
    // in the child; clear pointers so child cleanup touches no hardware or VMAs.
    t->bar0_mmio = nullptr;
    t->bar0_mapped_len = 0;
    t->bar4_mmio = nullptr;
    t->bar4_mapped_len = 0;
    t->ring_arena_mem = nullptr;
    t->ring_arena_bytes = 0;
    t->host_stage_mem = nullptr;
    t->host_stage_size = 0;
    t->sq_ring_mem = nullptr;
    t->cq_ring_mem = nullptr;
    t->status_page_mem = nullptr;
    t->pci_cmd_saved = false;
    t->ring_ready = false;
    t->imem_booted = false;
    t->dma_faulted = true;
    g_tracked_transports[i] = nullptr;
  }
  pthread_mutex_unlock(&g_vfio_fork_mu);
}

void init_vfio_atfork_once() {
  g_vfio_atfork_registered =
      (pthread_atfork(vfio_atfork_prepare, vfio_atfork_parent, vfio_atfork_child) == 0);
}

void unregister_transport_locked(VfioTransport* t) {
  for (size_t i = 0; i < kMaxTrackedVfioTransports; ++i) {
    if (g_tracked_transports[i] == t) {
      g_tracked_transports[i] = nullptr;
    }
  }
}

}  // namespace

VfioTransport::~VfioTransport() {
  close_device();
}

bool VfioTransport::register_fork_cleanup(std::string* err) {
  pthread_once(&g_vfio_atfork_once, init_vfio_atfork_once);
  if (!g_vfio_atfork_registered) {
    if (err) *err = "pthread_atfork registration failed for VFIO transport";
    return false;
  }
  pthread_mutex_lock(&g_vfio_fork_mu);
  size_t free_slot = kMaxTrackedVfioTransports;
  for (size_t i = 0; i < kMaxTrackedVfioTransports; ++i) {
    if (g_tracked_transports[i] == this) {
      pthread_mutex_unlock(&g_vfio_fork_mu);
      return true;
    }
    if (g_tracked_transports[i] == nullptr && free_slot == kMaxTrackedVfioTransports) {
      free_slot = i;
    }
  }
  if (free_slot == kMaxTrackedVfioTransports) {
    pthread_mutex_unlock(&g_vfio_fork_mu);
    if (err) *err = "Too many concurrent live VFIO transports";
    return false;
  }
  g_tracked_transports[free_slot] = this;
  pthread_mutex_unlock(&g_vfio_fork_mu);
  return true;
}

#ifdef __linux__
static inline void dma_wmb() {
#if defined(__aarch64__)
  asm volatile("dsb oshst" ::: "memory");
#elif defined(__arm__)
  asm volatile("dsb ost" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("sfence" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

static inline void dma_rmb() {
#if defined(__aarch64__)
  asm volatile("dmb oshld" ::: "memory");
#elif defined(__arm__)
  asm volatile("dmb osh" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("lfence" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
}

static bool is_supported_vfio_device_id(uint16_t device_id) {
  // Only TPU v5e (0x1ae0:0x0063) has a validated VFIO ring/DMA transport.
  // Documented TPU v6e (0x1ae0:0x006f) is rejected until its transport is validated.
  return device_id == 0x0063u;
}

static std::string sysfs_read_line(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return "";
  char buf[256];
  ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
  ::close(fd);
  if (n <= 0) return "";
  buf[n] = '\0';
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
    s.pop_back();
  }
  return s;
}

static std::string sysfs_read_link_base(const std::string& path) {
  char buf[512];
  ssize_t n = ::readlink(path.c_str(), buf, sizeof(buf) - 1);
  if (n <= 0) return "";
  buf[n] = '\0';
  std::string s(buf);
  auto pos = s.rfind('/');
  return (pos == std::string::npos) ? s : s.substr(pos + 1);
}

static bool discover_linux_tpu(const char* preferred_bdf, PciDiscoveryEntry* out, std::string* err) {
  DIR* dir = ::opendir("/sys/bus/pci/devices");
  if (!dir) {
    if (err) *err = "Failed to open /sys/bus/pci/devices";
    return false;
  }

  bool found = false;
  struct dirent* ent = nullptr;
  while ((ent = ::readdir(dir)) != nullptr) {
    if (ent->d_name[0] == '.') continue;
    std::string bdf(ent->d_name);
    if (preferred_bdf && preferred_bdf[0] != '\0' && bdf != preferred_bdf) {
      continue;
    }
    std::string base = "/sys/bus/pci/devices/" + bdf;
    uint64_t vid = 0, did = 0, cls = 0;
    parse_hex_u64(sysfs_read_line(base + "/vendor"), &vid);
    parse_hex_u64(sysfs_read_line(base + "/device"), &did);
    parse_hex_u64(sysfs_read_line(base + "/class"), &cls);
    std::string drv = sysfs_read_link_base(base + "/driver");
    std::string grp_str = sysfs_read_link_base(base + "/iommu_group");

    if (drv == "vfio-pci" && vid == 0x1ae0ULL && cls == 0xff0000ULL && !grp_str.empty()) {
      char* end = nullptr;
      errno = 0;
      unsigned long grp_val = std::strtoul(grp_str.c_str(), &end, 10);
      if (errno != 0 || end == grp_str.c_str() || *end != '\0') continue;
      if (!is_supported_vfio_device_id(static_cast<uint16_t>(did))) {
        if (preferred_bdf && preferred_bdf[0] != '\0') {
          if (err) {
            char msg[160];
            std::snprintf(msg, sizeof(msg),
                          "Unsupported TPU PCI device %s (0x%04x:0x%04x); only validated v5e (0x1ae0:0x0063) is supported",
                          bdf.c_str(), static_cast<unsigned>(vid), static_cast<unsigned>(did));
            *err = msg;
          }
          ::closedir(dir);
          return false;
        }
        continue;
      }
      out->bdf = bdf;
      out->vendor_id = static_cast<uint16_t>(vid);
      out->device_id = static_cast<uint16_t>(did);
      out->pci_class = static_cast<uint32_t>(cls);
      out->driver = drv;
      out->iommu_group = static_cast<uint32_t>(grp_val);
      found = true;
      break;
    }
  }
  ::closedir(dir);

  if (!found && err) {
    *err = "No supported vfio-pci Google Cloud TPU (0x1ae0:0x0063, class=0xff0000) discovered";
  }
  return found;
}
#endif

bool VfioTransport::is_owner_process() const {
  return owner_pid == 0 || owner_pid == static_cast<int64_t>(::getpid());
}

bool VfioTransport::open_device(const ot_config_t& cfg, ot_device_info_t* info, std::string* err) {
  if (!info) return false;
  std::memset(info, 0, sizeof(*info));
  owner_pid = static_cast<int64_t>(::getpid());

  static constexpr uint64_t kHbmAllocBase = 0x00100000ULL;

  if (cfg.mock_mode) {
    is_mock = true;
    std::snprintf(info->bdf, sizeof(info->bdf), "%s",
                  (cfg.bdf && cfg.bdf[0]) ? cfg.bdf : "0000:00:05.0");
    info->vendor_id = 0x1ae0u;
    info->device_id = 0x0063u;
    info->subsys_vendor_id = 0x1ae0u;
    info->subsys_device_id = 0x00afu;
    info->iommu_group = 0;
    info->chip_gen = OT_CHIP_V5E;
    info->bar0_size = 0x100000000ULL;
    info->bar4_size = 0x1000000ULL;
    info->msix_vectors = 128u;
    info->vmem_bytes = 128ULL * 1024ULL * 1024ULL;
    info->smem_bytes = 1024ULL * 1024ULL;
    info->num_mxus = 4u;
    info->is_mock = 1u;
    info->is_prebooted = 1u;

    mock_bar0.assign(128u * 1024u, 0);
    mock_bar4.assign(64u * 1024u, 0);
    // Mock 64 MiB HBM backing store for offline tests; honor smaller explicit budgets.
    mock_hbm.assign(64ULL * 1024ULL * 1024ULL, 0);
    uint64_t mock_max_usable = mock_hbm.size() - kHbmAllocBase;
    uint64_t usable_bytes = mock_max_usable;
    if (cfg.hbm_budget_bytes > 0 && cfg.hbm_budget_bytes < usable_bytes) {
      usable_bytes = cfg.hbm_budget_bytes;
    }
    iova_alloc.init(0x100000ULL, 0xf0000000ULL);
    hbm_alloc.init(kHbmAllocBase, kHbmAllocBase + usable_bytes);
    info->hbm_bytes_total = hbm_alloc.capacity_bytes();
    return true;
  }

#ifndef __linux__
  if (err) *err = "Real /dev/vfio/* transport requires Linux (set mock_mode=1 on this host)";
  return false;
#else
  PciDiscoveryEntry pci;
  if (!discover_linux_tpu(cfg.bdf, &pci, err)) {
    return false;
  }

  if (!register_fork_cleanup(err)) {
    return false;
  }

  // Acquire VFIO descriptors under g_vfio_fork_mu so a concurrent fork() cannot
  // observe an untracked descriptor in the child process.
  pthread_mutex_lock(&g_vfio_fork_mu);
  container_fd = ::open("/dev/vfio/vfio", O_RDWR | O_CLOEXEC);
  int open_err = errno;
  pthread_mutex_unlock(&g_vfio_fork_mu);
  if (container_fd < 0) {
    if (err) *err = std::string("open(/dev/vfio/vfio) failed: ") + std::strerror(open_err);
    close_device();
    return false;
  }
  if (::ioctl(container_fd, VFIO_GET_API_VERSION) != VFIO_API_VERSION) {
    if (err) *err = "Incompatible VFIO API version";
    close_device();
    return false;
  }

  std::string grp_path = "/dev/vfio/" + std::to_string(pci.iommu_group);
  pthread_mutex_lock(&g_vfio_fork_mu);
  group_fd = ::open(grp_path.c_str(), O_RDWR | O_CLOEXEC);
  open_err = errno;
  pthread_mutex_unlock(&g_vfio_fork_mu);
  if (group_fd < 0) {
    if (err) *err = "open(" + grp_path + ") failed: " + std::strerror(open_err);
    close_device();
    return false;
  }

  struct vfio_group_status gstat = {};
  gstat.argsz = sizeof(gstat);
  if (::ioctl(group_fd, VFIO_GROUP_GET_STATUS, &gstat) < 0 ||
      !(gstat.flags & VFIO_GROUP_FLAGS_VIABLE)) {
    if (err) *err = "VFIO group not viable";
    close_device();
    return false;
  }

  if (::ioctl(group_fd, VFIO_GROUP_SET_CONTAINER, &container_fd) < 0) {
    if (err) *err = std::string("VFIO_GROUP_SET_CONTAINER failed: ") + std::strerror(errno);
    close_device();
    return false;
  }

  if (::ioctl(container_fd, VFIO_SET_IOMMU, VFIO_TYPE1v2_IOMMU) < 0) {
    if (::ioctl(container_fd, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU) < 0) {
      if (err) *err = std::string("VFIO_SET_IOMMU failed: ") + std::strerror(errno);
      close_device();
      return false;
    }
  }

  pthread_mutex_lock(&g_vfio_fork_mu);
  device_fd = ::ioctl(group_fd, VFIO_GROUP_GET_DEVICE_FD, pci.bdf.c_str());
  open_err = errno;
  pthread_mutex_unlock(&g_vfio_fork_mu);
  if (device_fd < 0) {
    if (err) *err = std::string("VFIO_GROUP_GET_DEVICE_FD failed: ") + std::strerror(open_err);
    close_device();
    return false;
  }

  struct vfio_region_info r0 = {}, r4 = {}, rcfg = {};
  r0.argsz = sizeof(r0);
  r0.index = VFIO_PCI_BAR0_REGION_INDEX;
  r4.argsz = sizeof(r4);
  r4.index = VFIO_PCI_BAR4_REGION_INDEX;
  rcfg.argsz = sizeof(rcfg);
  rcfg.index = VFIO_PCI_CONFIG_REGION_INDEX;

  if (::ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO, &r0) < 0 ||
      ::ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO, &r4) < 0 ||
      ::ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO, &rcfg) < 0) {
    if (err) *err = std::string("VFIO_DEVICE_GET_REGION_INFO failed: ") + std::strerror(errno);
    close_device();
    return false;
  }

  bar0_offset = r0.offset;
  bar0_size = r0.size;
  bar4_offset = r4.offset;
  bar4_size = r4.size;
  cfg_offset = rcfg.offset;
  cfg_size = rcfg.size;

  // Re-verify live PCI vendor/device ID from config space before resetting or programming.
  uint16_t live_vid = 0, live_did = 0;
  if (!read_pci_config(0x00, &live_vid, sizeof(live_vid)) ||
      !read_pci_config(0x02, &live_did, sizeof(live_did)) ||
      live_vid != 0x1ae0u || !is_supported_vfio_device_id(live_did)) {
    if (err) {
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "Refusing to reset/program unsupported VFIO PCI device (0x%04x:0x%04x); only 0x1ae0:0x0063 is validated",
                    static_cast<unsigned>(live_vid), static_cast<unsigned>(live_did));
      *err = msg;
    }
    close_device();
    return false;
  }

  // Best-effort device function reset to ensure clean ring state across runs.
  (void)::ioctl(device_fd, VFIO_DEVICE_RESET);

  if (bar4_size > 0 && (r4.flags & VFIO_REGION_INFO_FLAG_MMAP)) {
    pthread_mutex_lock(&g_vfio_fork_mu);
    void* p4 = ::mmap(nullptr, static_cast<size_t>(bar4_size), PROT_READ | PROT_WRITE,
                      MAP_SHARED, device_fd, static_cast<off_t>(bar4_offset));
    int map_err = errno;
    if (p4 != MAP_FAILED) {
      if (::madvise(p4, static_cast<size_t>(bar4_size), MADV_DONTFORK) != 0) {
        map_err = errno;
        ::munmap(p4, static_cast<size_t>(bar4_size));
        p4 = MAP_FAILED;
        pthread_mutex_unlock(&g_vfio_fork_mu);
        if (err) *err = std::string("madvise(BAR4, MADV_DONTFORK) failed: ") + std::strerror(map_err);
        close_device();
        return false;
      }
      bar4_mmio = reinterpret_cast<volatile uint8_t*>(p4);
      bar4_mapped_len = static_cast<size_t>(bar4_size);
    }
    pthread_mutex_unlock(&g_vfio_fork_mu);
    if (p4 == MAP_FAILED) {
      if (err) *err = std::string("mmap(BAR4) failed: ") + std::strerror(map_err);
      close_device();
      return false;
    }
  }

  // Map first 1 MiB of BAR0 covering the 0x10000..0x12000 ring controller aperture.
  if (bar0_size >= 0x20000ULL && (r0.flags & VFIO_REGION_INFO_FLAG_MMAP)) {
    size_t map_len = static_cast<size_t>(bar0_size < 0x100000ULL ? bar0_size : 0x100000ULL);
    pthread_mutex_lock(&g_vfio_fork_mu);
    void* p0 = ::mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_SHARED,
                      device_fd, static_cast<off_t>(bar0_offset));
    int map_err = errno;
    if (p0 != MAP_FAILED) {
      if (::madvise(p0, map_len, MADV_DONTFORK) != 0) {
        map_err = errno;
        ::munmap(p0, map_len);
        p0 = MAP_FAILED;
        pthread_mutex_unlock(&g_vfio_fork_mu);
        if (err) *err = std::string("madvise(BAR0, MADV_DONTFORK) failed: ") + std::strerror(map_err);
        close_device();
        return false;
      }
      bar0_mmio = reinterpret_cast<volatile uint8_t*>(p0);
      bar0_mapped_len = map_len;
    }
    pthread_mutex_unlock(&g_vfio_fork_mu);
    if (p0 == MAP_FAILED) {
      if (err) *err = std::string("mmap(BAR0) failed: ") + std::strerror(map_err);
      close_device();
      return false;
    }
  }

  uint16_t svid = 0, sdid = 0;
  read_pci_config(0x2c, &svid, 2);
  read_pci_config(0x2e, &sdid, 2);

  if (!enable_bus_master(err)) {
    close_device();
    return false;
  }

  // Disable any active MSI-X interrupt bindings without enabling or reserving vectors
  // for polling-only transfers (count=0 + DATA_NONE|ACTION_TRIGGER per Linux VFIO).
  struct vfio_irq_info irqi = {};
  irqi.argsz = sizeof(irqi);
  irqi.index = VFIO_PCI_MSIX_IRQ_INDEX;
  uint32_t msix_cnt = 0;
  if (::ioctl(device_fd, VFIO_DEVICE_GET_IRQ_INFO, &irqi) == 0) {
    msix_cnt = irqi.count;
    struct vfio_irq_set irq_set = {};
    irq_set.argsz = sizeof(irq_set);
    irq_set.flags = VFIO_IRQ_SET_DATA_NONE | VFIO_IRQ_SET_ACTION_TRIGGER;
    irq_set.index = VFIO_PCI_MSIX_IRQ_INDEX;
    irq_set.start = 0;
    irq_set.count = 0;
    (void)::ioctl(device_fd, VFIO_DEVICE_SET_IRQS, &irq_set);
  }

  std::snprintf(info->bdf, sizeof(info->bdf), "%s", pci.bdf.c_str());
  info->vendor_id = pci.vendor_id;
  info->device_id = pci.device_id;
  info->subsys_vendor_id = svid;
  info->subsys_device_id = sdid;
  info->iommu_group = pci.iommu_group;
  if (pci.device_id == 0x0063u) {
    info->chip_gen = OT_CHIP_V5E;
  } else if (pci.device_id == 0x006fu) {
    info->chip_gen = OT_CHIP_V6E;
  } else {
    info->chip_gen = OT_CHIP_UNKNOWN;
  }
  info->bar0_size = bar0_size;
  info->bar4_size = bar4_size;
  info->msix_vectors = msix_cnt;
  info->vmem_bytes = 128ULL * 1024ULL * 1024ULL;
  info->smem_bytes = 1024ULL * 1024ULL;
  info->num_mxus = 4u;
  info->is_mock = 0u;
  // Derived in ot_context_create() only after parsing a valid BAR4 descriptor.
  info->is_prebooted = 0u;

  // Reserve 0x0000..0x00100000 for ring queues + 0x00100000..0x00500000 for staging.
  iova_alloc.init(0x00800000ULL, 0xf0000000ULL);
  // Usable HBM starts at kHbmAllocBase (0x00100000) and ends unconditionally at 0x3ef000000.
  static constexpr uint64_t kHbmUsableTop = 0x3ef000000ULL;
  static constexpr uint64_t kHbmUsableMaxBytes = kHbmUsableTop - kHbmAllocBase;
  uint64_t usable_bytes = kHbmUsableMaxBytes;
  if (cfg.hbm_budget_bytes > 0 && cfg.hbm_budget_bytes < usable_bytes) {
    usable_bytes = cfg.hbm_budget_bytes;
  }
  hbm_alloc.init(kHbmAllocBase, kHbmAllocBase + usable_bytes);
  info->hbm_bytes_total = hbm_alloc.capacity_bytes();

  if (!init_ring_queues(err)) {
    close_device();
    return false;
  }
  return true;
#endif
}

void VfioTransport::close_device() {
  teardown_imem_state();
  teardown_ring_queues();
#ifdef __linux__
  if (!is_mock && is_owner_process() && device_fd >= 0 && pci_cmd_saved) {
    write_pci_config(0x04, &saved_pci_cmd, sizeof(saved_pci_cmd));
    pci_cmd_saved = false;
  }
  if (is_owner_process()) {
    if (bar0_mmio && bar0_mapped_len > 0) {
      ::munmap(const_cast<uint8_t*>(bar0_mmio), bar0_mapped_len);
    }
    if (bar4_mmio && bar4_mapped_len > 0) {
      ::munmap(const_cast<uint8_t*>(bar4_mmio), bar4_mapped_len);
    }
  }
#endif
  pthread_mutex_lock(&g_vfio_fork_mu);
  unregister_transport_locked(this);
  if (device_fd >= 0) {
    ::close(device_fd);
    device_fd = -1;
  }
  if (group_fd >= 0) {
    ::close(group_fd);
    group_fd = -1;
  }
  if (container_fd >= 0) {
    ::close(container_fd);
    container_fd = -1;
  }
  bar0_mmio = nullptr;
  bar0_mapped_len = 0;
  bar4_mmio = nullptr;
  bar4_mapped_len = 0;
  owner_pid = 0;
  pthread_mutex_unlock(&g_vfio_fork_mu);
}

bool VfioTransport::map_dma_at(void* vaddr, uint64_t iova, uint64_t size, std::string* err) {
  if (!vaddr || size == 0) return false;
  if (is_mock) return true;
#ifdef __linux__
  struct vfio_iommu_type1_dma_map dmap = {};
  dmap.argsz = sizeof(dmap);
  dmap.flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE;
  dmap.vaddr = reinterpret_cast<uint64_t>(vaddr);
  dmap.iova = iova;
  dmap.size = size;
  if (::ioctl(container_fd, VFIO_IOMMU_MAP_DMA, &dmap) < 0) {
    if (err) *err = std::string("VFIO_IOMMU_MAP_DMA failed: ") + std::strerror(errno);
    return false;
  }
  return true;
#else
  (void)iova;
  (void)err;
  return false;
#endif
}

bool VfioTransport::map_dma(void* vaddr, uint64_t size, uint64_t* out_iova, std::string* err) {
  if (!vaddr || size == 0 || !out_iova) return false;
  uint64_t iova = 0;
  uint64_t align = iommu_min_page_size >= 4096ULL ? iommu_min_page_size : 4096ULL;
  if (!iova_alloc.alloc(size, align, &iova)) {
    if (err) *err = "IOVA arena exhausted";
    return false;
  }
  if (!map_dma_at(vaddr, iova, size, err)) {
    iova_alloc.free(iova);
    return false;
  }
  *out_iova = iova;
  return true;
}

bool VfioTransport::unmap_dma(uint64_t iova, uint64_t size) {
  iova_alloc.free(iova);
  if (is_mock) return true;
#ifdef __linux__
  struct vfio_iommu_type1_dma_unmap dunmap = {};
  dunmap.argsz = sizeof(dunmap);
  dunmap.iova = iova;
  dunmap.size = size;
  return ::ioctl(container_fd, VFIO_IOMMU_UNMAP_DMA, &dunmap) == 0;
#else
  (void)size;
  return false;
#endif
}

bool VfioTransport::read_pci_config(uint32_t offset, void* dst, size_t len) {
  if (is_mock) {
    std::memset(dst, 0, len);
    return true;
  }
#ifdef __linux__
  if (device_fd < 0 || offset + len > cfg_size) return false;
  return ::pread(device_fd, dst, len, static_cast<off_t>(cfg_offset + offset)) ==
         static_cast<ssize_t>(len);
#else
  (void)offset;
  (void)dst;
  (void)len;
  return false;
#endif
}

bool VfioTransport::write_pci_config(uint32_t offset, const void* src, size_t len) {
  if (is_mock) return true;
#ifdef __linux__
  if (device_fd < 0 || offset + len > cfg_size) return false;
  return ::pwrite(device_fd, src, len, static_cast<off_t>(cfg_offset + offset)) ==
         static_cast<ssize_t>(len);
#else
  (void)offset;
  (void)src;
  (void)len;
  return false;
#endif
}

bool VfioTransport::enable_bus_master(std::string* err) {
  if (is_mock) return true;
  uint16_t cmd = 0;
  if (!read_pci_config(0x04, &cmd, sizeof(cmd))) {
    if (err) *err = "Failed to read PCI command register";
    return false;
  }
  saved_pci_cmd = cmd;
  pci_cmd_saved = true;
  if ((cmd & 0x0006u) != 0x0006u) {
    cmd |= 0x0006u;  // PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER
    if (!write_pci_config(0x04, &cmd, sizeof(cmd))) {
      if (err) *err = "Failed to enable PCI Bus Master";
      return false;
    }
  }
  return true;
}

bool VfioTransport::init_ring_queues(std::string* err) {
  dma_faulted = false;
  if (is_mock) return true;
#ifndef __linux__
  (void)err;
  return false;
#else
  if (!bar0_mmio || bar0_mapped_len < 0x12000u) {
    if (err) *err = "BAR0 MMIO ring window unavailable";
    return false;
  }

  // Determine the effective VFIO DMA mapping alignment: max(host page size,
  // minimum IOMMU supported page size, 4 KiB). Mapping SQ, CQ, and the status
  // page inside one page-aligned contiguous arena preserves fixed hardware
  // offsets (0x0000, 0x8000, 0xa000) even on 16 KiB / 64 KiB page hosts.
  long host_ps = ::sysconf(_SC_PAGESIZE);
  uint64_t map_granule = (host_ps > 4096) ? static_cast<uint64_t>(host_ps) : 4096ULL;
  struct vfio_iommu_type1_info iommu_info = {};
  iommu_info.argsz = sizeof(iommu_info);
  if (::ioctl(container_fd, VFIO_IOMMU_GET_INFO, &iommu_info) == 0 &&
      (iommu_info.flags & VFIO_IOMMU_INFO_PGSIZES) && iommu_info.iova_pgsizes != 0) {
    uint64_t min_iommu_ps = iommu_info.iova_pgsizes & (~iommu_info.iova_pgsizes + 1ULL);
    if (min_iommu_ps > map_granule) map_granule = min_iommu_ps;
  }
  iommu_min_page_size = map_granule;

  const uint64_t ring_min_span = kStatusPageIova + kStatusPageBytes;
  const uint64_t aligned_ring_bytes =
      (ring_min_span + map_granule - 1ULL) & ~(map_granule - 1ULL);
  if (aligned_ring_bytes < ring_min_span || aligned_ring_bytes > kStageBaseIova) {
    if (err) *err = "IOMMU page size exceeds ring controller IOVA aperture";
    return false;
  }
  if ((kStageBaseIova % map_granule) != 0 || (kStageBytes % map_granule) != 0) {
    if (err) *err = "Staging DMA window is not aligned to IOMMU page size";
    return false;
  }

  if (map_granule > std::numeric_limits<size_t>::max()) {
    if (err) *err = "IOMMU page size exceeds host address width";
    return false;
  }

  auto map_anon_dma = [&](uint64_t iova, size_t len) -> uint8_t* {
    pthread_mutex_lock(&g_vfio_fork_mu);
    void* p = alloc_aligned_anon_pages(len, static_cast<size_t>(map_granule), err);
    if (!p) {
      pthread_mutex_unlock(&g_vfio_fork_mu);
      return nullptr;
    }
    // Prevent copy-on-write page replacement across fork() while pages are pinned by VFIO.
    if (::madvise(p, len, MADV_DONTFORK) != 0) {
      int madv_err = errno;
      ::munmap(p, len);
      pthread_mutex_unlock(&g_vfio_fork_mu);
      if (err) *err = std::string("madvise(MADV_DONTFORK) failed: ") + std::strerror(madv_err);
      return nullptr;
    }
    pthread_mutex_unlock(&g_vfio_fork_mu);
    std::memset(p, 0, len);
    if (!map_dma_at(p, iova, len, err)) {
      ::munmap(p, len);
      return nullptr;
    }
    return reinterpret_cast<uint8_t*>(p);
  };

  ring_arena_bytes = static_cast<size_t>(aligned_ring_bytes);
  ring_arena_mem = map_anon_dma(kSqRingIova, ring_arena_bytes);
  if (!ring_arena_mem) {
    if (err && err->empty()) *err = "Failed to map VFIO ring queue arena";
    teardown_ring_queues();
    return false;
  }
  host_stage_size = kStageBytes;
  host_stage_iova = kStageBaseIova;
  host_stage_mem = map_anon_dma(kStageBaseIova, kStageBytes);
  if (!host_stage_mem) {
    if (err && err->empty()) *err = "Failed to map VFIO host staging window";
    teardown_ring_queues();
    return false;
  }
  sq_ring_mem = ring_arena_mem + kSqRingIova;
  cq_ring_mem = ring_arena_mem + kCqRingIova;
  status_page_mem = reinterpret_cast<volatile uint32_t*>(ring_arena_mem + kStatusPageIova);

  // Disable queue before reprogramming base addresses.
  write32_bar0(0x10000u, 0u);
  write32_bar0(0x10040u, 0u);

  // Program SQ (0x0), CQ (0x8000), and Status Page (0xa000) IOVAs.
  write32_bar0(0x10010u, 0u);
  write32_bar0(0x10014u, static_cast<uint32_t>(kSqRingIova));
  write32_bar0(0x10020u, 0u);
  write32_bar0(0x10024u, static_cast<uint32_t>(kCqRingIova));
  write32_bar0(0x10030u, 0u);
  write32_bar0(0x10034u, static_cast<uint32_t>(kStatusPageIova));

  write32_bar0(0x10000u, 1u);
  write32_bar0(0x10104u, 0u);
  write32_bar0(0x10040u, 1u);

  // Assign polling MSI-X vector slots (matches clean-room bus survey).
  write32_bar0(0x10158u, 2u);
  write32_bar0(0x10118u, 3u);
  write32_bar0(0x10128u, 4u);
  write32_bar0(0x10138u, 5u);
  write32_bar0(0x10148u, 6u);
  for (uint32_t i = 0; i < 0x1cu; ++i) {
    write32_bar0(0x10208u + static_cast<uint64_t>(i) * 0x10u, 7u + i);
  }
  write32_bar0(0x10110u, 1u);
  write32_bar0(0x11f08u, 0x23u);
  write32_bar0(0x11f18u, 0x24u);
  write32_bar0(0x11f28u, 0x25u);
  write32_bar0(0x11f38u, 0x26u);
  write32_bar0(0x11f48u, 0x27u);
  write32_bar0(0x10150u, 1u);
  write32_bar0(0x10120u, 1u);
  write32_bar0(0x10130u, 1u);
  write32_bar0(0x11f10u, 1u);
  write32_bar0(0x11f20u, 1u);
  write32_bar0(0x11f30u, 1u);
  write32_bar0(0x11f40u, 1u);
  write32_bar0(0x11f00u, 1u);

  // Flush lingering hardware ring pointers back to index 0 so seq=0 executes cleanly.
  write32_bar0(0x10018u, 0u);
  write32_bar0(0x1002cu, 0u);
  write32_bar0(0x10044u, 0u);
  ::usleep(100);

  sq_seq = 0;
  ring_ready = true;
  return true;
#endif
}

void VfioTransport::teardown_ring_queues() {
  teardown_imem_state();
#ifdef __linux__
  // Only the process that opened the VFIO device owns hardware queues and MADV_DONTFORK arenas.
  // A forked child must never stop the parent's queues, unmap parent DMA, or munmap DONTFORK VMAs.
  if (is_owner_process()) {
    if (!is_mock && bar0_mmio && bar0_mapped_len >= 0x12000u && ring_ready) {
      write32_bar0(0x10150u, 0u);
      write32_bar0(0x10110u, 0u);
      write32_bar0(0x10120u, 0u);
      write32_bar0(0x10130u, 0u);
      write32_bar0(0x11f00u, 0u);
      write32_bar0(0x10100u, 0u);
      write32_bar0(0x10000u, 0u);
      write32_bar0(0x10010u, 0u);
      write32_bar0(0x10014u, 0u);
      write32_bar0(0x10020u, 0u);
      write32_bar0(0x10024u, 0u);
      write32_bar0(0x10030u, 0u);
      write32_bar0(0x10034u, 0u);
      write32_bar0(0x10040u, 0u);
    }
    auto unmap_anon_dma = [&](void* p, uint64_t iova, size_t len) {
      if (!p) return;
      if (container_fd >= 0) {
        struct vfio_iommu_type1_dma_unmap dunmap = {};
        dunmap.argsz = sizeof(dunmap);
        dunmap.iova = iova;
        dunmap.size = len;
        (void)::ioctl(container_fd, VFIO_IOMMU_UNMAP_DMA, &dunmap);
      }
      ::munmap(p, len);
    };
    unmap_anon_dma(host_stage_mem, kStageBaseIova, host_stage_size);
    unmap_anon_dma(ring_arena_mem, kSqRingIova, ring_arena_bytes);
  }
#endif
  ring_arena_mem = nullptr;
  ring_arena_bytes = 0;
  sq_ring_mem = nullptr;
  cq_ring_mem = nullptr;
  status_page_mem = nullptr;
  host_stage_mem = nullptr;
  host_stage_size = 0;
  host_stage_iova = 0;
  sq_seq = 0;
  ring_ready = false;
  dma_faulted = false;
}

bool validate_cq_completion(uint32_t sent_seq, const uint32_t cq[8], std::string* err) {
  if (!cq) {
    if (err) *err = "Null CQ completion buffer";
    return false;
  }
  if (cq[0] != sent_seq) {
    if (err) {
      char msg[128];
      std::snprintf(msg, sizeof(msg),
                    "CQ completion sequence mismatch: got=%u expected=%u",
                    cq[0], sent_seq);
      *err = msg;
    }
    return false;
  }
  if (cq[1] != 0u) {
    if (err) {
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "VFIO CQ completion error: seq=%u status=0x%08x w2=0x%08x w3=0x%08x",
                    cq[0], cq[1], cq[2], cq[3]);
      *err = msg;
    }
    return false;
  }
  return true;
}

bool VfioTransport::submit_sq_wait(const uint32_t desc[32], uint32_t timeout_ms,
                                   uint32_t cq_out[8], std::string* err) {
  last_op_timed_out = false;
  if (!is_mock && !is_owner_process()) {
    if (err) *err = "VFIO context cannot be used in a forked child process";
    return false;
  }
  if (dma_faulted) {
    if (err) *err = "VFIO DMA faulted after prior timeout; recreate context to reset device";
    return false;
  }
  if (is_mock) {
    if (cq_out) {
      std::memset(cq_out, 0, 32);
      cq_out[0] = sq_seq;
    }
    ++sq_seq;
    return true;
  }
#ifndef __linux__
  (void)desc; (void)timeout_ms; (void)cq_out; (void)err;
  return false;
#else
  if (!ring_ready || !sq_ring_mem || !cq_ring_mem || !status_page_mem) {
    if (err) *err = "VFIO ring queues not initialized";
    return false;
  }

  const uint32_t sent_seq = sq_seq;
  uint32_t idx = sent_seq % 256u;
  uint32_t* sq_slot = reinterpret_cast<uint32_t*>(sq_ring_mem + idx * 128u);
  uint32_t* cq_slot = reinterpret_cast<uint32_t*>(cq_ring_mem + idx * 32u);
  // Poison the CQ slot before ringing the doorbell so stale entries cannot match sent_seq.
  std::memset(cq_slot, 0, 32u);
  cq_slot[0] = ~sent_seq;
  std::memcpy(sq_slot, desc, 128u);
  sq_slot[0] = sent_seq;
  // Order DMA payload, CQ poison, and SQ descriptor writes in outer-shareable domain before MMIO doorbell.
  dma_wmb();

  const uint32_t target_ring = (sent_seq + 1u) % 256u;
  sq_seq = sent_seq + 1u;
  write32_bar0(0x10018u, target_ring);
  write32_bar0(0x10044u, 0u);

  auto quiesce_on_unverified_dma = [&]() {
    // Quiesce the device immediately so an in-flight descriptor cannot complete
    // late into reused staging memory or reallocated HBM.
    dma_faulted = true;
    ring_ready = false;
    if (bar0_mmio && bar0_mapped_len >= 0x12000u) {
      write32_bar0(0x10000u, 0u);
      write32_bar0(0x10040u, 0u);
    }
    uint16_t cmd = 0;
    if (read_pci_config(0x04, &cmd, sizeof(cmd)) && (cmd & 0x0004u)) {
      cmd &= static_cast<uint16_t>(~0x0004u);  // Clear PCI_COMMAND_MASTER
      (void)write_pci_config(0x04, &cmd, sizeof(cmd));
    }
    if (device_fd >= 0) {
      (void)::ioctl(device_fd, VFIO_DEVICE_RESET);
    }
  };

  const auto timeout = std::chrono::milliseconds(timeout_ms ? timeout_ms : 5000u);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    uint32_t cq_tail = __atomic_load_n(&status_page_mem[1], __ATOMIC_ACQUIRE);
    if (cq_tail == target_ring) {
      // Order completion entry and DMA payload reads after observing cq_tail from PCIe device.
      dma_rmb();
      uint32_t local_cq[8] = {};
      std::memcpy(local_cq, cq_slot, 32u);
      if (cq_out) {
        std::memcpy(cq_out, local_cq, 32u);
      }
      if (local_cq[0] != sent_seq) {
        // Submitted DMA has no verified completion: quiesce and quarantine context.
        quiesce_on_unverified_dma();
        (void)validate_cq_completion(sent_seq, local_cq, err);
        if (err) *err += " (device quiesced; recreate context)";
        return false;
      }
      dma_wmb();
      write32_bar0(0x1002cu, target_ring);
      return validate_cq_completion(sent_seq, local_cq, err);
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    ::usleep(10);
  }

  quiesce_on_unverified_dma();
  last_op_timed_out = true;
  if (err) {
    *err = "SQ doorbell timeout waiting for completion seq=" + std::to_string(sent_seq) +
           " (device quiesced; recreate context)";
  }
  return false;
#endif
}

bool VfioTransport::dma_transfer_sync(uint64_t host_iova, uint64_t dev_addr, size_t len_bytes,
                                      bool is_h2d, uint32_t dst_flags_hi, uint32_t w28,
                                      uint32_t w29, uint32_t timeout_ms, std::string* err) {
  if (len_bytes == 0 || (len_bytes % 8u) != 0) {
    if (err) *err = "DMA transfer length must be a positive multiple of 8 bytes";
    return false;
  }
  uint64_t qwords = len_bytes / 8u;
  if (qwords > 0xffffffffULL) {
    if (err) *err = "DMA transfer length exceeds 32-bit qword count";
    return false;
  }
  if ((dev_addr >> 36) != 0ULL) {
    if (err) *err = "Device HBM address exceeds 36-bit aperture";
    return false;
  }
  uint32_t d[32] = {};
  d[1] = kSqOpcodeDma;
  d[8] = (static_cast<uint32_t>(qwords & 0xffffu) << 16) | 0x1001u;
  d[9] = 0x00400000u | static_cast<uint32_t>((qwords >> 16) & 0xffffu);
  d[10] = static_cast<uint32_t>(host_iova >> 32);
  d[11] = 0x10u;
  d[12] = 1u; d[13] = 1u; d[14] = 1u; d[15] = 0x10u;
  d[16] = 1u; d[17] = 1u; d[18] = 1u; d[19] = 1u;
  d[20] = 1u; d[21] = 1u; d[22] = 0x10u; d[23] = 0u;

  if (is_h2d) {
    d[24] = static_cast<uint32_t>(host_iova & 0xffffffffULL);
    d[25] = 0x00604000u;
    d[26] = static_cast<uint32_t>(dev_addr & 0xffffffffULL);
    d[27] = dst_flags_hi | static_cast<uint32_t>((dev_addr >> 32) & 0xfu);
    d[28] = w28;
    d[29] = w29;
  } else {
    d[24] = static_cast<uint32_t>(dev_addr & 0xffffffffULL);
    d[25] = 0x00104000u | static_cast<uint32_t>((dev_addr >> 32) & 0xfu);
    d[26] = static_cast<uint32_t>(host_iova & 0xffffffffULL);
    d[27] = 0x40000000u;
    d[28] = w28;
    d[29] = w29;
  }

  uint32_t cq_out[8] = {};
  return submit_sq_wait(d, timeout_ms, cq_out, err);
}

bool VfioTransport::dma_h2d_sync(uint64_t host_iova, uint64_t dst_hbm, size_t len_bytes,
                                 uint32_t w28, uint32_t w29, uint32_t timeout_ms,
                                 std::string* err) {
  return dma_transfer_sync(host_iova, dst_hbm, len_bytes, /*is_h2d=*/true,
                           0x40000000u, w28, w29, timeout_ms, err);
}

bool VfioTransport::dma_d2h_sync(uint64_t host_iova, uint64_t src_hbm, size_t len_bytes,
                                 uint32_t w28, uint32_t w29, uint32_t timeout_ms,
                                 std::string* err) {
  return dma_transfer_sync(host_iova, src_hbm, len_bytes, /*is_h2d=*/false,
                           0x40000000u, w28, w29, timeout_ms, err);
}

bool VfioTransport::dma_host_to_imem(uint64_t host_iova, uint64_t dst_imem, size_t len_bytes,
                                     uint32_t timeout_ms, std::string* err) {
  if (len_bytes == 0 || (len_bytes % kImemDmaAlignBytes) != 0) {
    if (err) *err = "Host->IMEM DMA length must be a positive multiple of 512 bytes (8 bundles)";
    return false;
  }
  if ((dst_imem % kImemDmaAlignBytes) != 0 || (dst_imem >> 32) != 0ULL) {
    if (err) *err = "Host->IMEM DMA destination must be 512-byte aligned within 32-bit IMEM space";
    return false;
  }
  return dma_transfer_sync(host_iova, dst_imem, len_bytes, /*is_h2d=*/true,
                           kImemDmaDstFlagsHi, kImemDmaW28, kImemDmaW29, timeout_ms, err);
}

bool VfioTransport::submit_core_ctrl(uint32_t ctrl_word, uint32_t timeout_ms,
                                     uint32_t cq_out[8], std::string* err) {
  if (is_mock) {
    if (mock_core_ctrl_fail) {
      if (err) *err = "Mock core_ctrl command failed";
      return false;
    }
    if (cq_out) std::memset(cq_out, 0, 32);
    return true;
  }
  uint32_t d[32] = {};
  d[1] = kSqOpcodeCoreCtrl;
  d[8] = ctrl_word;
  return submit_sq_wait(d, timeout_ms, cq_out, err);
}

bool VfioTransport::submit_exec_sync(uint32_t expected_exec_seq, uint32_t timeout_ms,
                                     uint32_t cq_out[8], std::string* err) {
  last_op_timed_out = false;
  uint32_t local_cq[8] = {};
  bool poll_failed = false;
  bool timed_out = false;

  if (is_mock && !mock_exec_timeout && !mock_exec_poll_error) {
    if (cq_out) {
      std::memset(cq_out, 0, 32);
      cq_out[4] = 1u;
      cq_out[5] = expected_exec_seq;
    }
    return true;
  }
  if (is_mock) {
    if (mock_exec_poll_error) {
      poll_failed = true;
      if (err) *err = "Mock EXEC poll CQ failure";
    } else {
      timed_out = true;
    }
  } else {
    uint32_t d[32] = {};
    d[1] = kSqOpcodeExec;
    d[8] = kExecOpW8;
    d[10] = kExecOpW10;
    const auto timeout = std::chrono::milliseconds(timeout_ms ? timeout_ms : 5000u);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      std::memset(local_cq, 0, sizeof(local_cq));
      const auto now = std::chrono::steady_clock::now();
      const auto rem_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
      const uint32_t sq_timeout_ms = (rem_ms > 0) ? static_cast<uint32_t>(rem_ms) : 1u;
      if (!submit_sq_wait(d, sq_timeout_ms, local_cq, err)) {
        poll_failed = true;
        break;
      }
      if (local_cq[4] == 1u && local_cq[5] == expected_exec_seq) {
        if (cq_out) {
          std::memcpy(cq_out, local_cq, sizeof(local_cq));
        }
        return true;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        timed_out = true;
        break;
      }
#ifdef __linux__
      ::usleep(25);
#endif
    }
  }
  if (cq_out) {
    std::memcpy(cq_out, local_cq, sizeof(local_cq));
  }
  // Launch doorbell has already been rung; any unconfirmed exit (deadline timeout
  // OR SQ/CQ poll error) must fault and quiesce the device so scratch HBM is
  // never freed under an in-flight kernel.
  const bool was_timeout = timed_out || last_op_timed_out;
  const bool already_faulted = dma_faulted;
  dma_faulted = true;
  last_op_timed_out = was_timeout;
  ring_ready = false;
  imem_booted = false;
  loaded_prog_kind = 0u;
#ifdef __linux__
  if (!is_mock && !already_faulted) {
    if (bar0_mmio && bar0_mapped_len >= 0x12000u) {
      write32_bar0(0x10000u, 0u);
      write32_bar0(0x10040u, 0u);
    }
    uint16_t cmd = 0;
    if (read_pci_config(0x04, &cmd, sizeof(cmd)) && (cmd & 0x0004u)) {
      cmd &= static_cast<uint16_t>(~0x0004u);
      (void)write_pci_config(0x04, &cmd, sizeof(cmd));
    }
    if (device_fd >= 0) {
      (void)::ioctl(device_fd, VFIO_DEVICE_RESET);
    }
  }
#else
  (void)already_faulted;
#endif
  if (err) {
    if (poll_failed && !err->empty()) {
      if (err->find("device quiesced") == std::string::npos) {
        *err += " (EXEC unconfirmed; device quiesced; recreate context)";
      }
    } else if (was_timeout) {
      *err = "EXEC completion timeout: cq[4]=" + std::to_string(local_cq[4]) +
             " cq[5]=" + std::to_string(local_cq[5]) +
             " expected=" + std::to_string(expected_exec_seq) +
             " (device quiesced; recreate context)";
    } else {
      *err = "EXEC poll failed before completion (device quiesced; recreate context)";
    }
  }
  return false;
}

bool VfioTransport::ensure_matmul_scratch_slot(uint64_t* slot, std::string* err) {
  if (!slot) return false;
  if (*slot != 0) return true;
  if (!hbm_alloc.alloc(kTile128x128Bf16Bytes, 4096ULL, slot)) {
    if (err) *err = "Failed to allocate HBM matmul scratch buffer";
    return false;
  }
  return true;
}

bool VfioTransport::ensure_matmul_scratch(std::string* err) {
  const bool had_lhs = (scratch_lhs_hbm != 0);
  if (!ensure_matmul_scratch_slot(&scratch_lhs_hbm, err)) return false;
  if (!ensure_matmul_scratch_slot(&scratch_rhs_hbm, err)) {
    if (!had_lhs && scratch_lhs_hbm != 0) {
      hbm_alloc.free(scratch_lhs_hbm);
      scratch_lhs_hbm = 0;
    }
    return false;
  }
  return true;
}

bool VfioTransport::ensure_imem_booted(uint32_t timeout_ms, std::string* err) {
  return ensure_imem_program(/*bf16=*/1u, timeout_ms, err);
}

bool VfioTransport::ensure_imem_program(uint32_t prog_kind, uint32_t timeout_ms,
                                        std::string* err) {
  if (!is_mock && !is_owner_process()) {
    if (err) *err = "VFIO context cannot be used in a forked child process";
    return false;
  }
  if (dma_faulted) {
    if (err) *err = "VFIO DMA faulted after prior timeout; recreate context";
    return false;
  }
  if (imem_booted && loaded_prog_kind == prog_kind) return true;
  if (is_mock) {
    // Invalidate prior resident cache before reloading so failed core_ctrl start
    // never leaves stale resident metadata behind. Keep exec_seq monotonic.
    imem_booted = false;
    loaded_prog_kind = 0u;
    uint32_t cq_out[8] = {};
    if (!submit_core_ctrl(kCoreCtrlStart, timeout_ms, cq_out, err)) {
      return false;
    }
    imem_booted = true;
    loaded_prog_kind = prog_kind;
    return true;
  }
  if (!ring_ready || !host_stage_mem || host_stage_size < kResidentImemBytes) {
    if (err) *err = "VFIO ring queues or host DMA staging unavailable for IMEM boot";
    return false;
  }

  std::vector<VliwBundle> imem_prog;
  if (prog_kind == 2u) {
    imem_prog = emit_v5e_resident_freetile_quant_imem(/*is_q4=*/false);
  } else if (prog_kind == 3u) {
    imem_prog = emit_v5e_resident_freetile_quant_imem(/*is_q4=*/true);
  } else {
    imem_prog = emit_v5e_resident_freetile_bf16_imem();
  }
  const size_t prog_bytes = imem_prog.size() * sizeof(VliwBundle);
  if (prog_bytes == 0 || (prog_bytes % kImemDmaAlignBytes) != 0 ||
      prog_bytes > host_stage_size) {
    if (err) *err = "Resident IMEM program size or alignment invariant violated";
    return false;
  }

  // Invalidate cached resident program before modifying IMEM so a failed DMA
  // upload or core-start never leaves stale resident metadata behind.
  imem_booted = false;
  loaded_prog_kind = 0u;

  std::memcpy(host_stage_mem, imem_prog.data(), prog_bytes);
  if (!dma_host_to_imem(host_stage_iova, /*dst_imem=*/0ULL, prog_bytes, timeout_ms, err)) {
    return false;
  }
  uint32_t cq_out[8] = {};
  if (!submit_core_ctrl(kCoreCtrlStart, timeout_ms, cq_out, err)) {
    return false;
  }
  // Keep exec_seq monotonic across resident program switches so sync[0x1f9]
  // from a prior program's completion can never false-match the next launch.
  imem_booted = true;
  loaded_prog_kind = prog_kind;
  return true;
}

bool VfioTransport::ensure_quant_constants(uint32_t timeout_ms, std::string* err) {
  if (quant_const_hbm != 0) return true;
  if (!hbm_alloc.alloc(40960u, 4096u, &quant_const_hbm)) {
    if (err) *err = "Out of device HBM for quantized dequant constants (40 KiB)";
    return false;
  }
  if (is_mock) return true;
  if (!host_stage_mem || host_stage_size < 40960u) {
    hbm_alloc.free(quant_const_hbm);
    quant_const_hbm = 0;
    if (err) *err = "Host DMA staging too small for quantized dequant constants";
    return false;
  }
  init_quant_const_vregs_40kib(host_stage_mem);
  if (!dma_h2d_sync(host_stage_iova, quant_const_hbm, 40960u, 0x10u, 0u, timeout_ms, err)) {
    hbm_alloc.free(quant_const_hbm);
    quant_const_hbm = 0;
    return false;
  }
  return true;
}

void VfioTransport::teardown_imem_state() {
  if (is_owner_process() && !dma_faulted) {
    if (scratch_rhs_hbm != 0) {
      hbm_alloc.free(scratch_rhs_hbm);
    }
    if (scratch_lhs_hbm != 0) {
      hbm_alloc.free(scratch_lhs_hbm);
    }
    if (quant_const_hbm != 0) {
      hbm_alloc.free(quant_const_hbm);
    }
  }
  scratch_rhs_hbm = 0;
  scratch_lhs_hbm = 0;
  quant_const_hbm = 0;
  imem_booted = false;
  loaded_prog_kind = 0;
  exec_seq = 0;
}

uint32_t VfioTransport::read32_bar4(uint64_t off) const {
  if (is_mock) {
    if (off + 4 > mock_bar4.size()) return 0xffffffffu;
    uint32_t v = 0;
    std::memcpy(&v, mock_bar4.data() + off, 4);
    return v;
  }
  if (!bar4_mmio || off + 4 > bar4_mapped_len) return 0xffffffffu;
  return *reinterpret_cast<const volatile uint32_t*>(bar4_mmio + off);
}

void VfioTransport::write32_bar4(uint64_t off, uint32_t val) {
  if (is_mock) {
    if (off + 4 <= mock_bar4.size()) std::memcpy(mock_bar4.data() + off, &val, 4);
    return;
  }
  if (bar4_mmio && off + 4 <= bar4_mapped_len) {
    *reinterpret_cast<volatile uint32_t*>(bar4_mmio + off) = val;
  }
}

uint32_t VfioTransport::read32_bar0(uint64_t off) const {
  if (is_mock) {
    if (off + 4 > mock_bar0.size()) return 0xffffffffu;
    uint32_t v = 0;
    std::memcpy(&v, mock_bar0.data() + off, 4);
    return v;
  }
  if (!bar0_mmio || off + 4 > bar0_mapped_len) return 0xffffffffu;
  return *reinterpret_cast<const volatile uint32_t*>(bar0_mmio + off);
}

void VfioTransport::write32_bar0(uint64_t off, uint32_t val) {
  if (is_mock) {
    if (off + 4 <= mock_bar0.size()) std::memcpy(mock_bar0.data() + off, &val, 4);
    return;
  }
  if (bar0_mmio && off + 4 <= bar0_mapped_len) {
    *reinterpret_cast<volatile uint32_t*>(bar0_mmio + off) = val;
  }
}

}  // namespace ot
