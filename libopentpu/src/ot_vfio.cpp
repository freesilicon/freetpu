// SPDX-License-Identifier: Apache-2.0
#include "ot_internal.h"
#include "ot_vliw.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __linux__
#include <dirent.h>
#include <fcntl.h>
#include <linux/vfio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
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

  // Reuse a free block if big enough
  for (auto& b : blocks_) {
    if (!b.in_use && b.size >= size && (b.iova % align) == 0) {
      b.in_use = true;
      *out_iova = b.iova;
      return true;
    }
  }

  uint64_t aligned = (next_ + (align - 1)) & ~(align - 1);
  if (aligned < next_ || aligned > limit_ || size > limit_ - aligned) {
    return false;
  }
  blocks_.push_back({aligned, size, true});
  next_ = aligned + size;
  *out_iova = aligned;
  return true;
}

void IovaAllocator::free(uint64_t iova) {
  for (auto& b : blocks_) {
    if (b.in_use && b.iova == iova) {
      b.in_use = false;
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

#ifdef __linux__
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

    if (drv == "vfio-pci" && vid == 0x1ae0ULL && cls == 0xff0000ULL) {
      out->bdf = bdf;
      out->vendor_id = static_cast<uint16_t>(vid);
      out->device_id = static_cast<uint16_t>(did);
      out->pci_class = static_cast<uint32_t>(cls);
      out->driver = drv;
      out->iommu_group = static_cast<uint32_t>(std::strtoul(grp_str.c_str(), nullptr, 10));
      found = true;
      break;
    }
  }
  ::closedir(dir);

  if (!found && err) {
    *err = "No vfio-pci Google Cloud TPU (vendor=0x1ae0, class=0xff0000) discovered";
  }
  return found;
}
#endif

bool VfioTransport::open_device(const ot_config_t& cfg, ot_device_info_t* info, std::string* err) {
  if (!info) return false;
  std::memset(info, 0, sizeof(*info));

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
    info->hbm_bytes_total = cfg.hbm_budget_bytes ? cfg.hbm_budget_bytes : 16909336064ULL;
    info->vmem_bytes = 128ULL * 1024ULL * 1024ULL;
    info->smem_bytes = 1024ULL * 1024ULL;
    info->num_mxus = 4u;
    info->is_mock = 1u;
    info->is_prebooted = 1u;

    mock_bar0.assign(128u * 1024u, 0);
    mock_bar4.assign(64u * 1024u, 0);
    // Mock 64 MiB HBM backing store for offline tests
    mock_hbm.assign(64ULL * 1024ULL * 1024ULL, 0);
    iova_alloc.init(0x100000ULL, 0xf0000000ULL);
    hbm_alloc.init(0x00100000ULL, mock_hbm.size());
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

  container_fd = ::open("/dev/vfio/vfio", O_RDWR | O_CLOEXEC);
  if (container_fd < 0) {
    if (err) *err = std::string("open(/dev/vfio/vfio) failed: ") + std::strerror(errno);
    return false;
  }
  if (::ioctl(container_fd, VFIO_GET_API_VERSION) != VFIO_API_VERSION) {
    if (err) *err = "Incompatible VFIO API version";
    close_device();
    return false;
  }

  std::string grp_path = "/dev/vfio/" + std::to_string(pci.iommu_group);
  group_fd = ::open(grp_path.c_str(), O_RDWR | O_CLOEXEC);
  if (group_fd < 0) {
    if (err) *err = "open(" + grp_path + ") failed: " + std::strerror(errno);
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

  device_fd = ::ioctl(group_fd, VFIO_GROUP_GET_DEVICE_FD, pci.bdf.c_str());
  if (device_fd < 0) {
    if (err) *err = std::string("VFIO_GROUP_GET_DEVICE_FD failed: ") + std::strerror(errno);
    close_device();
    return false;
  }

  // Best-effort device function reset to ensure clean ring state across runs.
  (void)::ioctl(device_fd, VFIO_DEVICE_RESET);

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

  if (bar4_size > 0 && (r4.flags & VFIO_REGION_INFO_FLAG_MMAP)) {
    void* p4 = ::mmap(nullptr, static_cast<size_t>(bar4_size), PROT_READ | PROT_WRITE,
                      MAP_SHARED, device_fd, static_cast<off_t>(bar4_offset));
    if (p4 == MAP_FAILED) {
      if (err) *err = std::string("mmap(BAR4) failed: ") + std::strerror(errno);
      close_device();
      return false;
    }
    bar4_mmio = reinterpret_cast<volatile uint8_t*>(p4);
    bar4_mapped_len = static_cast<size_t>(bar4_size);
  }

  // Map first 1 MiB of BAR0 covering the 0x10000..0x12000 ring controller aperture.
  if (bar0_size >= 0x20000ULL && (r0.flags & VFIO_REGION_INFO_FLAG_MMAP)) {
    size_t map_len = static_cast<size_t>(bar0_size < 0x100000ULL ? bar0_size : 0x100000ULL);
    void* p0 = ::mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_SHARED,
                      device_fd, static_cast<off_t>(bar0_offset));
    if (p0 == MAP_FAILED) {
      if (err) *err = std::string("mmap(BAR0) failed: ") + std::strerror(errno);
      close_device();
      return false;
    }
    bar0_mmio = reinterpret_cast<volatile uint8_t*>(p0);
    bar0_mapped_len = map_len;
  }

  uint16_t svid = 0, sdid = 0;
  read_pci_config(0x2c, &svid, 2);
  read_pci_config(0x2e, &sdid, 2);

  if (!enable_bus_master(err)) {
    close_device();
    return false;
  }

  // Clear any stale MSI-X eventfd bindings so polling mode starts clean.
  struct vfio_irq_info irqi = {};
  irqi.argsz = sizeof(irqi);
  irqi.index = VFIO_PCI_MSIX_IRQ_INDEX;
  uint32_t msix_cnt = 0;
  if (::ioctl(device_fd, VFIO_DEVICE_GET_IRQ_INFO, &irqi) == 0) {
    msix_cnt = irqi.count;
    if (msix_cnt > 0 && msix_cnt <= 256) {
      std::vector<uint8_t> set_buf(sizeof(struct vfio_irq_set) + msix_cnt * sizeof(int), 0);
      auto* irq_set = reinterpret_cast<struct vfio_irq_set*>(set_buf.data());
      irq_set->argsz = static_cast<uint32_t>(set_buf.size());
      irq_set->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
      irq_set->index = VFIO_PCI_MSIX_IRQ_INDEX;
      irq_set->start = 0;
      irq_set->count = msix_cnt;
      int* fds = reinterpret_cast<int*>(irq_set->data);
      for (uint32_t i = 0; i < msix_cnt; ++i) fds[i] = -1;
      ::ioctl(device_fd, VFIO_DEVICE_SET_IRQS, irq_set);
    }
  }

  std::snprintf(info->bdf, sizeof(info->bdf), "%s", pci.bdf.c_str());
  info->vendor_id = pci.vendor_id;
  info->device_id = pci.device_id;
  info->subsys_vendor_id = svid;
  info->subsys_device_id = sdid;
  info->iommu_group = pci.iommu_group;
  if (pci.device_id == 0x0063u) {
    info->chip_gen = OT_CHIP_V5E;
  } else if (pci.device_id == 0x0066u || pci.device_id == 0x0065u) {
    info->chip_gen = OT_CHIP_V6E;
  } else {
    info->chip_gen = OT_CHIP_UNKNOWN;
  }
  info->bar0_size = bar0_size;
  info->bar4_size = bar4_size;
  info->msix_vectors = msix_cnt;
  info->hbm_bytes_total = cfg.hbm_budget_bytes ? cfg.hbm_budget_bytes : 16909336064ULL;
  info->vmem_bytes = 128ULL * 1024ULL * 1024ULL;
  info->smem_bytes = 1024ULL * 1024ULL;
  info->num_mxus = 4u;
  info->is_mock = 0u;
  info->is_prebooted = (read32_bar4(0x08) != 0) ? 1u : 0u;

  // Reserve 0x0000..0x00100000 for ring queues + 0x00100000..0x00500000 for staging.
  iova_alloc.init(0x00800000ULL, 0xf0000000ULL);
  // 15.748 GiB usable HBM starts at 0x00100000; usable top ends before 0x3ef000000.
  uint64_t hbm_limit = info->hbm_bytes_total < 0x3f0000000ULL ? info->hbm_bytes_total : 0x3ef000000ULL;
  hbm_alloc.init(0x00100000ULL, hbm_limit);

  if (!init_ring_queues(err)) {
    close_device();
    return false;
  }
  return true;
#endif
}

void VfioTransport::close_device() {
  teardown_ring_queues();
#ifdef __linux__
  if (!is_mock && device_fd >= 0 && pci_cmd_saved) {
    write_pci_config(0x04, &saved_pci_cmd, sizeof(saved_pci_cmd));
    pci_cmd_saved = false;
  }
  if (bar0_mmio && bar0_mapped_len > 0) {
    ::munmap(const_cast<uint8_t*>(bar0_mmio), bar0_mapped_len);
  }
  if (bar4_mmio && bar4_mapped_len > 0) {
    ::munmap(const_cast<uint8_t*>(bar4_mmio), bar4_mapped_len);
  }
  if (device_fd >= 0) ::close(device_fd);
  if (group_fd >= 0) ::close(group_fd);
  if (container_fd >= 0) ::close(container_fd);
#endif
  bar0_mmio = nullptr;
  bar0_mapped_len = 0;
  bar4_mmio = nullptr;
  bar4_mapped_len = 0;
  device_fd = -1;
  group_fd = -1;
  container_fd = -1;
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
  if (!iova_alloc.alloc(size, 4096, &iova)) {
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
  if (is_mock) return true;
#ifndef __linux__
  (void)err;
  return false;
#else
  if (!bar0_mmio || bar0_mapped_len < 0x12000u) {
    if (err) *err = "BAR0 MMIO ring window unavailable";
    return false;
  }

  auto map_anon_dma = [&](uint64_t iova, size_t len) -> uint8_t* {
    void* p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    std::memset(p, 0, len);
    if (!map_dma_at(p, iova, len, err)) {
      ::munmap(p, len);
      return nullptr;
    }
    return reinterpret_cast<uint8_t*>(p);
  };

  sq_ring_mem = map_anon_dma(kSqRingIova, kSqRingBytes);
  cq_ring_mem = map_anon_dma(kCqRingIova, kCqRingBytes);
  status_page_mem = reinterpret_cast<volatile uint32_t*>(
      map_anon_dma(kStatusPageIova, kStatusPageBytes));
  host_stage_mem = map_anon_dma(kStageBaseIova, kStageBytes);
  if (!sq_ring_mem || !cq_ring_mem || !status_page_mem || !host_stage_mem) {
    if (err && err->empty()) *err = "Failed to map VFIO ring queue buffers";
    teardown_ring_queues();
    return false;
  }
  host_stage_size = kStageBytes;
  host_stage_iova = kStageBaseIova;

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
#ifdef __linux__
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
  unmap_anon_dma(const_cast<uint32_t*>(status_page_mem), kStatusPageIova, kStatusPageBytes);
  unmap_anon_dma(cq_ring_mem, kCqRingIova, kCqRingBytes);
  unmap_anon_dma(sq_ring_mem, kSqRingIova, kSqRingBytes);
#endif
  sq_ring_mem = nullptr;
  cq_ring_mem = nullptr;
  status_page_mem = nullptr;
  host_stage_mem = nullptr;
  host_stage_size = 0;
  host_stage_iova = 0;
  sq_seq = 0;
  ring_ready = false;
}

bool VfioTransport::submit_sq_wait(const uint32_t desc[32], uint32_t timeout_ms,
                                   uint32_t cq_out[8], std::string* err) {
  if (is_mock) {
    if (cq_out) std::memset(cq_out, 0, 32);
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

  uint32_t idx = sq_seq % 256u;
  uint32_t* sq_slot = reinterpret_cast<uint32_t*>(sq_ring_mem + idx * 128u);
  std::memcpy(sq_slot, desc, 128u);
  sq_slot[0] = sq_seq;
  __sync_synchronize();

  uint32_t target_seq = sq_seq + 1u;
  sq_seq = target_seq;
  write32_bar0(0x10018u, target_seq);
  write32_bar0(0x10044u, 0u);

  uint32_t budget_us = (timeout_ms ? timeout_ms : 5000u) * 1000u;
  uint32_t waited_us = 0;
  while (waited_us <= budget_us) {
    __sync_synchronize();
    if (status_page_mem[1] >= target_seq) {
      if (cq_out) {
        const uint32_t* cq_slot = reinterpret_cast<const uint32_t*>(cq_ring_mem + idx * 32u);
        std::memcpy(cq_out, cq_slot, 32u);
      }
      write32_bar0(0x1002cu, target_seq);
      return true;
    }
    ::usleep(10);
    waited_us += 10u;
  }

  if (err) {
    *err = "SQ doorbell timeout waiting for completion seq=" + std::to_string(target_seq - 1u);
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
