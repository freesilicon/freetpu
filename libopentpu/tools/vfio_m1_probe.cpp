// SPDX-License-Identifier: Apache-2.0
// Clean-room M1 VFIO / PCIe / BAR Feasibility Probe for Cloud TPU (/dev/vfio/*).
// Links only standard C/C++ runtime (zero libtpu.so, zero libpython).

#include <errno.h>
#include <fcntl.h>
#include <linux/vfio.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace {

sigjmp_buf g_fault_jmp;
volatile sig_atomic_t g_fault_armed = 0;
volatile sig_atomic_t g_last_sig = 0;

void fault_handler(int sig) {
  if (g_fault_armed) {
    g_last_sig = sig;
    g_fault_armed = 0;
    siglongjmp(g_fault_jmp, 1);
  }
  signal(sig, SIG_DFL);
  raise(sig);
}

template <typename T>
bool safe_mmio_read(const volatile void* addr, T* out) {
  g_last_sig = 0;
  if (sigsetjmp(g_fault_jmp, 1) != 0) {
    return false;
  }
  g_fault_armed = 1;
  __sync_synchronize();
  T val = *reinterpret_cast<const volatile T*>(addr);
  __sync_synchronize();
  g_fault_armed = 0;
  *out = val;
  return true;
}

std::string read_sysfs_line(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) return "";
  std::string s;
  std::getline(ifs, s);
  return s;
}

std::string read_sysfs_link_base(const std::string& path) {
  char buf[512];
  ssize_t n = readlink(path.c_str(), buf, sizeof(buf) - 1);
  if (n <= 0) return "";
  buf[n] = '\0';
  std::string s(buf);
  auto pos = s.rfind('/');
  return (pos == std::string::npos) ? s : s.substr(pos + 1);
}

struct PciTarget {
  std::string bdf;
  std::string vendor;
  std::string device;
  std::string pci_class;
  std::string driver;
  std::string iommu_group;
};

std::vector<PciTarget> discover_vfio_tpus() {
  std::vector<PciTarget> out;
  DIR* dir = opendir("/sys/bus/pci/devices");
  if (!dir) return out;
  struct dirent* ent;
  while ((ent = readdir(dir)) != nullptr) {
    if (ent->d_name[0] == '.') continue;
    std::string base = std::string("/sys/bus/pci/devices/") + ent->d_name;
    PciTarget t;
    t.bdf = ent->d_name;
    t.vendor = read_sysfs_line(base + "/vendor");
    t.device = read_sysfs_line(base + "/device");
    t.pci_class = read_sysfs_line(base + "/class");
    t.driver = read_sysfs_link_base(base + "/driver");
    t.iommu_group = read_sysfs_link_base(base + "/iommu_group");
    if (t.driver == "vfio-pci" || (t.vendor == "0x1ae0" && t.pci_class == "0xff0000")) {
      out.push_back(t);
    }
  }
  closedir(dir);
  std::sort(out.begin(), out.end(), [](const PciTarget& a, const PciTarget& b) {
    return a.bdf < b.bdf;
  });
  return out;
}

void check_proc_maps() {
  std::ifstream ifs("/proc/self/maps");
  std::string line;
  int libtpu_cnt = 0;
  int libpython_cnt = 0;
  while (std::getline(ifs, line)) {
    if (line.find("libtpu") != std::string::npos) libtpu_cnt++;
    if (line.find("libpython") != std::string::npos) libpython_cnt++;
  }
  printf("MAPS_LIBTPU=%d MAPS_LIBPYTHON=%d\n", libtpu_cnt, libpython_cnt);
}

void print_pci_config(int dev_fd, uint64_t cfg_offset, uint64_t cfg_size) {
  size_t read_sz = std::min<uint64_t>(cfg_size, 4096);
  std::vector<uint8_t> cfg(read_sz, 0);
  ssize_t got = pread(dev_fd, cfg.data(), read_sz, static_cast<off_t>(cfg_offset));
  printf("\n=== PCI CONFIG SPACE (size=%llu, pread=%zd, errno=%d) ===\n",
         (unsigned long long)cfg_size, got, errno);
  if (got < 64) return;

  auto r16 = [&](size_t off) -> uint16_t {
    uint16_t v = 0;
    memcpy(&v, cfg.data() + off, 2);
    return v;
  };
  auto r32 = [&](size_t off) -> uint32_t {
    uint32_t v = 0;
    memcpy(&v, cfg.data() + off, 4);
    return v;
  };

  uint16_t vid = r16(0x00), did = r16(0x02);
  uint16_t cmd = r16(0x04), sts = r16(0x06);
  uint8_t rev = cfg[0x08];
  uint32_t cls = r32(0x08) >> 8;
  uint16_t svid = r16(0x2c), sdid = r16(0x2e);
  uint8_t irq_line = cfg[0x3c], irq_pin = cfg[0x3d];
  printf("vid=0x%04x did=0x%04x svid=0x%04x sdid=0x%04x rev=0x%02x class=0x%06x\n",
         vid, did, svid, sdid, rev, cls);
  printf("cmd=0x%04x (IO=%d MEM=%d MASTER=%d INTxDis=%d) sts=0x%04x (CAPLIST=%d) irq_line=%u irq_pin=%u\n",
         cmd, !!(cmd & 1), !!(cmd & 2), !!(cmd & 4), !!(cmd & 0x400),
         sts, !!(sts & 0x10), irq_line, irq_pin);
  for (int b = 0; b < 6; ++b) {
    printf("  cfg.BAR%d = 0x%08x\n", b, r32(0x10 + b * 4));
  }

  // Dump non-zero 256B standard config header
  printf("Standard 256B config hex:\n");
  for (size_t off = 0; off < std::min<size_t>(got, 256); off += 16) {
    bool any = false;
    for (size_t k = 0; k < 16; ++k) if (cfg[off + k] != 0) any = true;
    if (!any) continue;
    printf("  %03zx:", off);
    for (size_t k = 0; k < 16; ++k) printf(" %02x", cfg[off + k]);
    printf("\n");
  }

  // Walk PCI capabilities
  if (sts & 0x10) {
    uint8_t ptr = cfg[0x34] & 0xfc;
    int guard = 0;
    printf("PCI Capabilities:\n");
    while (ptr >= 0x40 && static_cast<size_t>(ptr) + 4 <= static_cast<size_t>(got) && guard++ < 48) {
      uint8_t cap_id = cfg[ptr];
      uint8_t next = cfg[ptr + 1] & 0xfc;
      printf("  cap@0x%02x id=0x%02x next=0x%02x", ptr, cap_id, next);
      if (cap_id == 0x11 && static_cast<size_t>(ptr) + 12 <= static_cast<size_t>(got)) {  // MSI-X
        uint16_t msgctl = r16(ptr + 2);
        uint32_t tbl = r32(ptr + 4);
        uint32_t pba = r32(ptr + 8);
        printf(" [MSI-X] tbl_size=%u enable=%d func_mask=%d tbl_bir=%u tbl_off=0x%x pba_bir=%u pba_off=0x%x",
               (msgctl & 0x7ff) + 1, !!(msgctl & 0x8000), !!(msgctl & 0x4000),
               tbl & 7, tbl & ~7u, pba & 7, pba & ~7u);
      } else if (cap_id == 0x10 && static_cast<size_t>(ptr) + 20 <= static_cast<size_t>(got)) {  // PCIe
        uint16_t pcie_caps = r16(ptr + 2);
        uint32_t devcap = r32(ptr + 4);
        uint16_t devctl = r16(ptr + 8);
        uint16_t devsts = r16(ptr + 10);
        uint32_t lnkcap = r32(ptr + 12);
        uint16_t lnksts = r16(ptr + 18);
        printf(" [PCIe] caps=0x%04x devcap=0x%08x devctl=0x%04x(MPS=%u,MRRS=%u) devsts=0x%04x lnkcap=0x%08x lnksts=0x%04x",
               pcie_caps, devcap, devctl, 128u << ((devctl >> 5) & 7), 128u << ((devctl >> 12) & 7),
               devsts, lnkcap, lnksts);
      } else if (cap_id == 0x09 && static_cast<size_t>(ptr) + 4 <= static_cast<size_t>(got)) {  // Vendor-Specific
        uint8_t vlen = cfg[ptr + 2];
        printf(" [VendorSpec len=%u]", vlen);
        for (size_t j = 0; j < vlen && static_cast<size_t>(ptr) + j < static_cast<size_t>(got); ++j) {
          printf(" %02x", cfg[ptr + j]);
        }
      }
      printf("\n");
      ptr = next;
    }
  }

  // Walk PCIe Extended Capabilities (offset 0x100+)
  if (static_cast<size_t>(got) >= 0x104) {
    uint16_t eptr = 0x100;
    int guard = 0;
    printf("PCIe Extended Capabilities:\n");
    while (eptr >= 0x100 && static_cast<size_t>(eptr) + 4 <= static_cast<size_t>(got) && guard++ < 64) {
      uint32_t hdr = r32(eptr);
      if (hdr == 0 || hdr == 0xffffffffu) break;
      uint16_t eid = hdr & 0xffffu;
      uint8_t ever = (hdr >> 16) & 0xfu;
      uint16_t enext = (hdr >> 20) & 0xfffu;
      printf("  ecap@0x%03x id=0x%04x ver=%u next=0x%03x", eptr, eid, ever, enext);
      if (eid == 0x000b && static_cast<size_t>(eptr) + 16 <= static_cast<size_t>(got)) {
        printf(" [VSEC hdr1=0x%08x]", r32(eptr + 4));
      }
      printf("\n");
      if (enext == 0 || enext == eptr) break;
      eptr = enext;
    }
  }
}

void probe_bar_reads(int dev_fd, const char* name, uint32_t region_idx,
                     uint64_t offset, uint64_t size, uint32_t flags) {
  printf("\n=== PROBE %s (region=%u size=0x%llx (%llu MiB) offset=0x%llx flags=0x%x) ===\n",
         name, region_idx, (unsigned long long)size,
         (unsigned long long)(size / (1024ULL * 1024ULL)),
         (unsigned long long)offset, flags);

  // 1) Test pread on device_fd if READ flag is set
  if (flags & VFIO_REGION_INFO_FLAG_READ) {
    uint32_t v32 = 0;
    uint64_t v64 = 0;
    ssize_t r4 = pread(dev_fd, &v32, 4, static_cast<off_t>(offset));
    int e4 = errno;
    ssize_t r8 = pread(dev_fd, &v64, 8, static_cast<off_t>(offset));
    int e8 = errno;
    printf("  pread(0): 4B=%zd(0x%08x, err=%d) 8B=%zd(0x%016llx, err=%d)\n",
           r4, v32, (r4 < 0 ? e4 : 0), r8, (unsigned long long)v64, (r8 < 0 ? e8 : 0));
  }

  // 2) Test mmap if MMAP flag is set
  if (!(flags & VFIO_REGION_INFO_FLAG_MMAP)) {
    printf("  MMAP flag NOT set\n");
    return;
  }

  // Try mapping up to 16 MiB window at offset 0 (or full BAR4, or multiple windows for BAR0)
  size_t win_sz = static_cast<size_t>(std::min<uint64_t>(size, 16ULL * 1024ULL * 1024ULL));
  void* map_rw = mmap(nullptr, win_sz, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd,
                      static_cast<off_t>(offset));
  if (map_rw == MAP_FAILED) {
    printf("  mmap(RW, %zu bytes) FAILED errno=%d (%s)\n", win_sz, errno, strerror(errno));
    // Fallback to PROT_READ only
    map_rw = mmap(nullptr, win_sz, PROT_READ, MAP_SHARED, dev_fd, static_cast<off_t>(offset));
    if (map_rw == MAP_FAILED) {
      printf("  mmap(RO, %zu bytes) FAILED errno=%d (%s)\n", win_sz, errno, strerror(errno));
      return;
    }
    printf("  mmap(RO, %zu bytes) OK -> %p\n", win_sz, map_rw);
  } else {
    printf("  mmap(RW, %zu bytes) OK -> %p\n", win_sz, map_rw);
  }

  // Probe first 64 KiB at 32-bit stride (print all non-zero and non-0xffffffff words, capped)
  const volatile uint8_t* base = reinterpret_cast<const volatile uint8_t*>(map_rw);
  int printed = 0;
  int fault_cnt = 0;
  int ffff_cnt = 0;
  int zero_cnt = 0;
  printf("  Scanning first 64 KiB (0x0000..0x10000) in 4B words:\n");
  for (size_t off = 0; off < std::min<size_t>(win_sz, 65536); off += 4) {
    uint32_t v = 0;
    if (!safe_mmio_read<uint32_t>(base + off, &v)) {
      fault_cnt++;
      if (fault_cnt <= 8) printf("    [FAULT sig=%d at +0x%04zx]\n", (int)g_last_sig, off);
      continue;
    }
    if (v == 0xffffffffu) {
      ffff_cnt++;
    } else if (v == 0) {
      zero_cnt++;
    } else {
      if (printed < 120) {
        printf("    +0x%05zx: 0x%08x\n", off, v);
      }
      printed++;
    }
  }
  printf("  First 64 KiB summary: nonzero=%d, zero=%d, 0xffffffff=%d, faults=%d\n",
         printed, zero_cnt, ffff_cnt, fault_cnt);

  // Sparse 4 KiB page scan across win_sz (sample +0x0, +0x4, +0x8, +0x10, +0x100)
  printf("  Sparse 4 KiB page scan across 0..%zu MiB (offsets 0x0,0x4,0x8,0x10,0x100):\n",
         win_sz / (1024 * 1024));
  int page_hits = 0;
  for (size_t pg = 0; pg < win_sz; pg += 4096) {
    static const size_t kSub[] = {0x0, 0x4, 0x8, 0x10, 0x100};
    uint32_t vals[5] = {0};
    bool any_interesting = false;
    bool faulted = false;
    for (int i = 0; i < 5; ++i) {
      if (!safe_mmio_read<uint32_t>(base + pg + kSub[i], &vals[i])) {
        faulted = true;
        break;
      }
      if (vals[i] != 0 && vals[i] != 0xffffffffu) any_interesting = true;
    }
    if (faulted) {
      if (page_hits < 128) printf("    page 0x%07zx: FAULT\n", pg);
      page_hits++;
    } else if (any_interesting) {
      if (page_hits < 128) {
        printf("    page 0x%07zx: [0]=0x%08x [4]=0x%08x [8]=0x%08x [10]=0x%08x [100]=0x%08x\n",
               pg, vals[0], vals[1], vals[2], vals[3], vals[4]);
      }
      page_hits++;
    }
  }
  printf("  Total interesting/faulted 4 KiB pages in window: %d\n", page_hits);

  munmap(map_rw, win_sz);

  // If BAR is larger than 16 MiB (e.g. BAR0 = 4 GiB), sample 4 KiB pages at 16 MiB strides
  if (size > win_sz) {
    printf("  Sampling large BAR (>16 MiB) at 64 MiB strides up to %llu MiB:\n",
           (unsigned long long)(size / (1024ULL * 1024ULL)));
    for (uint64_t off = win_sz; off < size; off += 64ULL * 1024ULL * 1024ULL) {
      void* pg = mmap(nullptr, 4096, PROT_READ, MAP_SHARED, dev_fd,
                      static_cast<off_t>(offset + off));
      if (pg == MAP_FAILED) {
        printf("    +0x%09llx: mmap failed err=%d\n", (unsigned long long)off, errno);
        continue;
      }
      uint32_t v0 = 0, v4 = 0;
      bool ok0 = safe_mmio_read<uint32_t>(pg, &v0);
      bool ok4 = safe_mmio_read<uint32_t>(reinterpret_cast<const uint8_t*>(pg) + 4, &v4);
      printf("    +0x%09llx: ok=(%d,%d) [0]=0x%08x [4]=0x%08x\n",
             (unsigned long long)off, ok0, ok4, v0, v4);
      munmap(pg, 4096);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGBUS, fault_handler);
  signal(SIGSEGV, fault_handler);

  printf("=== LIBOPENTPU M1 VFIO FEASIBILITY PROBE ===\n");
  check_proc_maps();

  auto targets = discover_vfio_tpus();
  if (targets.empty()) {
    fprintf(stderr, "ERROR: No vfio-pci / Google TPU PCI device found\n");
    return 1;
  }
  const PciTarget& tgt = (argc > 1) ? [&]() -> const PciTarget& {
    for (const auto& t : targets) if (t.bdf == argv[1]) return t;
    return targets[0];
  }() : targets[0];

  printf("Selected target: bdf=%s vendor=%s device=%s class=%s driver=%s iommu_group=%s\n",
         tgt.bdf.c_str(), tgt.vendor.c_str(), tgt.device.c_str(),
         tgt.pci_class.c_str(), tgt.driver.c_str(), tgt.iommu_group.c_str());

  int container = open("/dev/vfio/vfio", O_RDWR);
  if (container < 0) {
    perror("open(/dev/vfio/vfio)");
    return 1;
  }
  int api_ver = ioctl(container, VFIO_GET_API_VERSION);
  printf("VFIO_GET_API_VERSION = %d (expected %d)\n", api_ver, VFIO_API_VERSION);
  printf("IOMMU extensions: TYPE1=%d TYPE1v2=%d DMA_CC=%d NOIOMMU=%d\n",
         ioctl(container, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU),
         ioctl(container, VFIO_CHECK_EXTENSION, VFIO_TYPE1v2_IOMMU),
         ioctl(container, VFIO_CHECK_EXTENSION, VFIO_DMA_CC_IOMMU),
         ioctl(container, VFIO_CHECK_EXTENSION, VFIO_NOIOMMU_IOMMU));

  std::string grp_path = "/dev/vfio/" + tgt.iommu_group;
  int group = open(grp_path.c_str(), O_RDWR);
  if (group < 0) {
    perror(("open(" + grp_path + ")").c_str());
    return 1;
  }

  struct vfio_group_status gstat = {};
  gstat.argsz = sizeof(gstat);
  if (ioctl(group, VFIO_GROUP_GET_STATUS, &gstat) < 0) {
    perror("VFIO_GROUP_GET_STATUS");
    return 1;
  }
  printf("VFIO_GROUP_GET_STATUS flags=0x%x (VIABLE=%d CONTAINER_SET=%d)\n",
         gstat.flags, !!(gstat.flags & VFIO_GROUP_FLAGS_VIABLE),
         !!(gstat.flags & VFIO_GROUP_FLAGS_CONTAINER_SET));

  if (ioctl(group, VFIO_GROUP_SET_CONTAINER, &container) < 0) {
    perror("VFIO_GROUP_SET_CONTAINER");
    return 1;
  }

  int iommu_type = VFIO_TYPE1v2_IOMMU;
  if (ioctl(container, VFIO_SET_IOMMU, iommu_type) < 0) {
    perror("VFIO_SET_IOMMU(TYPE1v2)");
    iommu_type = VFIO_TYPE1_IOMMU;
    if (ioctl(container, VFIO_SET_IOMMU, iommu_type) < 0) {
      perror("VFIO_SET_IOMMU(TYPE1)");
      return 1;
    }
  }
  printf("VFIO_SET_IOMMU succeeded: type=%d\n", iommu_type);

  // Query IOMMU info + capabilities (IOVA ranges)
  std::vector<uint8_t> iommu_buf(4096, 0);
  auto* iinfo = reinterpret_cast<struct vfio_iommu_type1_info*>(iommu_buf.data());
  iinfo->argsz = static_cast<uint32_t>(iommu_buf.size());
  if (ioctl(container, VFIO_IOMMU_GET_INFO, iinfo) == 0) {
    printf("VFIO_IOMMU_GET_INFO flags=0x%x iova_pgsizes=0x%llx cap_offset=%u argsz=%u\n",
           iinfo->flags, (unsigned long long)iinfo->iova_pgsizes,
           iinfo->cap_offset, iinfo->argsz);
    if ((iinfo->flags & VFIO_IOMMU_INFO_CAPS) && iinfo->cap_offset > 0) {
      uint32_t off = iinfo->cap_offset;
      while (off >= sizeof(*iinfo) && off + sizeof(struct vfio_info_cap_header) <= iommu_buf.size()) {
        auto* hdr = reinterpret_cast<struct vfio_info_cap_header*>(iommu_buf.data() + off);
        printf("  IOMMU cap id=%u ver=%u next=%u\n", hdr->id, hdr->version, hdr->next);
#ifdef VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE
        if (hdr->id == VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE &&
            off + sizeof(struct vfio_iommu_type1_info_cap_iova_range) <= iommu_buf.size()) {
          auto* r = reinterpret_cast<struct vfio_iommu_type1_info_cap_iova_range*>(iommu_buf.data() + off);
          for (uint32_t i = 0; i < r->nr_iovas; ++i) {
            printf("    IOVA range[%u]: [0x%llx .. 0x%llx]\n", i,
                   (unsigned long long)r->iova_ranges[i].start,
                   (unsigned long long)r->iova_ranges[i].end);
          }
        }
#endif
        if (hdr->next == 0 || hdr->next <= off) break;
        off = hdr->next;
      }
    }
  } else {
    perror("VFIO_IOMMU_GET_INFO");
  }

  // Test DMA mapping at 32-bit and >32-bit IOVAs
  const size_t dma_len = 2 * 1024 * 1024;
  void* host_dma = mmap(nullptr, dma_len, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_dma != MAP_FAILED) {
    madvise(host_dma, dma_len, MADV_DONTFORK);
    memset(host_dma, 0xA5, dma_len);
    const uint64_t test_iovas[] = {0x00100000ULL, 0x80000000ULL, 0x100000000ULL, 0x1000000000ULL};
    for (uint64_t iova : test_iovas) {
      struct vfio_iommu_type1_dma_map dmap = {};
      dmap.argsz = sizeof(dmap);
      dmap.flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE;
      dmap.vaddr = reinterpret_cast<uint64_t>(host_dma);
      dmap.iova = iova;
      dmap.size = dma_len;
      int rc = ioctl(container, VFIO_IOMMU_MAP_DMA, &dmap);
      int err = errno;
      printf("VFIO_IOMMU_MAP_DMA(iova=0x%llx, size=0x%zx): rc=%d err=%d (%s)\n",
             (unsigned long long)iova, dma_len, rc, (rc < 0 ? err : 0),
             (rc < 0 ? strerror(err) : "OK"));
      if (rc == 0) {
        struct vfio_iommu_type1_dma_unmap dunmap = {};
        dunmap.argsz = sizeof(dunmap);
        dunmap.iova = iova;
        dunmap.size = dma_len;
        int urc = ioctl(container, VFIO_IOMMU_UNMAP_DMA, &dunmap);
        printf("  UNMAP rc=%d unmapped_size=0x%llx\n", urc, (unsigned long long)dunmap.size);
      }
    }
    munmap(host_dma, dma_len);
  }

  int dev = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, tgt.bdf.c_str());
  if (dev < 0) {
    perror("VFIO_GROUP_GET_DEVICE_FD");
    return 1;
  }
  printf("VFIO_GROUP_GET_DEVICE_FD(%s) = %d\n", tgt.bdf.c_str(), dev);

  struct vfio_device_info dinfo = {};
  dinfo.argsz = sizeof(dinfo);
  if (ioctl(dev, VFIO_DEVICE_GET_INFO, &dinfo) < 0) {
    perror("VFIO_DEVICE_GET_INFO");
    return 1;
  }
  printf("VFIO_DEVICE_GET_INFO flags=0x%x num_regions=%u num_irqs=%u\n",
         dinfo.flags, dinfo.num_regions, dinfo.num_irqs);

  struct vfio_region_info bar0_info = {};
  struct vfio_region_info bar4_info = {};
  struct vfio_region_info cfg_info = {};

  for (uint32_t i = 0; i < dinfo.num_regions; ++i) {
    std::vector<uint8_t> rbuf(4096, 0);
    auto* rinfo = reinterpret_cast<struct vfio_region_info*>(rbuf.data());
    rinfo->argsz = static_cast<uint32_t>(rbuf.size());
    rinfo->index = i;
    if (ioctl(dev, VFIO_DEVICE_GET_REGION_INFO, rinfo) < 0) {
      printf("  region[%u]: ioctl err=%d (%s)\n", i, errno, strerror(errno));
      continue;
    }
    printf("  region[%u]: size=0x%llx (%llu KiB) offset=0x%llx flags=0x%x (R=%d W=%d MMAP=%d CAPS=%d)\n",
           i, (unsigned long long)rinfo->size,
           (unsigned long long)(rinfo->size / 1024ULL),
           (unsigned long long)rinfo->offset, rinfo->flags,
           !!(rinfo->flags & VFIO_REGION_INFO_FLAG_READ),
           !!(rinfo->flags & VFIO_REGION_INFO_FLAG_WRITE),
           !!(rinfo->flags & VFIO_REGION_INFO_FLAG_MMAP),
           !!(rinfo->flags & VFIO_REGION_INFO_FLAG_CAPS));
    if ((rinfo->flags & VFIO_REGION_INFO_FLAG_CAPS) && rinfo->cap_offset > 0) {
      uint32_t off = rinfo->cap_offset;
      while (off >= sizeof(*rinfo) && off + sizeof(struct vfio_info_cap_header) <= rbuf.size()) {
        auto* hdr = reinterpret_cast<struct vfio_info_cap_header*>(rbuf.data() + off);
        printf("    cap id=%u ver=%u next=%u\n", hdr->id, hdr->version, hdr->next);
        if (hdr->id == VFIO_REGION_INFO_CAP_SPARSE_MMAP) {
          auto* sm = reinterpret_cast<struct vfio_region_info_cap_sparse_mmap*>(rbuf.data() + off);
          for (uint32_t a = 0; a < sm->nr_areas; ++a) {
            printf("      sparse_area[%u]: off=0x%llx size=0x%llx\n", a,
                   (unsigned long long)sm->areas[a].offset,
                   (unsigned long long)sm->areas[a].size);
          }
        }
        if (hdr->next == 0 || hdr->next <= off) break;
        off = hdr->next;
      }
    }
    if (i == VFIO_PCI_BAR0_REGION_INDEX) bar0_info = *rinfo;
    if (i == VFIO_PCI_BAR4_REGION_INDEX) bar4_info = *rinfo;
    if (i == VFIO_PCI_CONFIG_REGION_INDEX) cfg_info = *rinfo;
  }

  for (uint32_t i = 0; i < dinfo.num_irqs; ++i) {
    struct vfio_irq_info iinfo_q = {};
    iinfo_q.argsz = sizeof(iinfo_q);
    iinfo_q.index = i;
    if (ioctl(dev, VFIO_DEVICE_GET_IRQ_INFO, &iinfo_q) < 0) {
      printf("  irq[%u]: ioctl err=%d (%s)\n", i, errno, strerror(errno));
      continue;
    }
    printf("  irq[%u]: count=%u flags=0x%x (EVFD=%d MASK=%d AUTO=%d NORESIZE=%d)\n",
           i, iinfo_q.count, iinfo_q.flags,
           !!(iinfo_q.flags & VFIO_IRQ_INFO_EVENTFD),
           !!(iinfo_q.flags & VFIO_IRQ_INFO_MASKABLE),
           !!(iinfo_q.flags & VFIO_IRQ_INFO_AUTOMASKED),
           !!(iinfo_q.flags & VFIO_IRQ_INFO_NORESIZE));
  }

  if (cfg_info.size > 0) {
    print_pci_config(dev, cfg_info.offset, cfg_info.size);
  }
  if (bar4_info.size > 0) {
    probe_bar_reads(dev, "BAR4", VFIO_PCI_BAR4_REGION_INDEX, bar4_info.offset, bar4_info.size, bar4_info.flags);
  }
  if (bar0_info.size > 0) {
    probe_bar_reads(dev, "BAR0", VFIO_PCI_BAR0_REGION_INDEX, bar0_info.offset, bar0_info.size, bar0_info.flags);
  }

  close(dev);
  close(group);
  close(container);
  printf("\n=== M1_PROBE_COMPLETE ===\n");
  return 0;
}
