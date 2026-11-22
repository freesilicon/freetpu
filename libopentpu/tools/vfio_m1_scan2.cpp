// SPDX-License-Identifier: Apache-2.0
// M1 Deep Read-Only BAR0/BAR4 Layout Scan on /dev/vfio/0 (Zero libtpu.so).

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

sigjmp_buf g_jmp;
volatile sig_atomic_t g_armed = 0;

void fault_handler(int sig) {
  if (g_armed) {
    g_armed = 0;
    siglongjmp(g_jmp, 1);
  }
  signal(sig, SIG_DFL);
  raise(sig);
}

bool safe_r32(const volatile void* p, uint32_t* out) {
  if (sigsetjmp(g_jmp, 1) != 0) return false;
  g_armed = 1;
  uint32_t v = *reinterpret_cast<const volatile uint32_t*>(p);
  g_armed = 0;
  *out = v;
  return true;
}

std::string read_line(const std::string& p) {
  std::ifstream ifs(p);
  std::string s;
  if (ifs.is_open()) std::getline(ifs, s);
  return s;
}

std::string read_link_base(const std::string& p) {
  char buf[512];
  ssize_t n = readlink(p.c_str(), buf, sizeof(buf) - 1);
  if (n <= 0) return "";
  buf[n] = '\0';
  std::string s(buf);
  auto pos = s.rfind('/');
  return pos == std::string::npos ? s : s.substr(pos + 1);
}

void dump_hex_ascii(const volatile uint8_t* base, size_t start, size_t len) {
  for (size_t off = 0; off < len; off += 16) {
    uint8_t row[16];
    bool any = false;
    for (size_t i = 0; i < 16; i += 4) {
      uint32_t w = 0xffffffffu;
      safe_r32(base + start + off + i, &w);
      memcpy(row + i, &w, 4);
      if (w != 0 && w != 0xffffffffu) any = true;
    }
    if (!any) continue;
    printf("  +0x%06zx:", start + off);
    for (size_t i = 0; i < 16; ++i) printf(" %02x", row[i]);
    printf("  |");
    for (size_t i = 0; i < 16; ++i) {
      uint8_t c = row[i];
      printf("%c", (c >= 0x20 && c < 0x7f) ? c : '.');
    }
    printf("|\n");
  }
}

}  // namespace

int main() {
  signal(SIGBUS, fault_handler);
  signal(SIGSEGV, fault_handler);

  std::string bdf, grp;
  DIR* d = opendir("/sys/bus/pci/devices");
  if (!d) return 1;
  struct dirent* e;
  while ((e = readdir(d)) != nullptr) {
    if (e->d_name[0] == '.') continue;
    std::string base = std::string("/sys/bus/pci/devices/") + e->d_name;
    if (read_link_base(base + "/driver") == "vfio-pci" &&
        read_line(base + "/vendor") == "0x1ae0") {
      bdf = e->d_name;
      grp = read_link_base(base + "/iommu_group");
      break;
    }
  }
  closedir(d);
  if (bdf.empty()) {
    fprintf(stderr, "No TPU found\n");
    return 1;
  }

  int container = open("/dev/vfio/vfio", O_RDWR | O_CLOEXEC);
  if (container < 0) {
    fprintf(stderr, "open(/dev/vfio/vfio) failed: %s\n", strerror(errno));
    return 1;
  }
  int group = open(("/dev/vfio/" + grp).c_str(), O_RDWR | O_CLOEXEC);
  if (group < 0) {
    fprintf(stderr, "open(/dev/vfio/%s) failed: %s\n", grp.c_str(), strerror(errno));
    close(container);
    return 1;
  }
  if (ioctl(group, VFIO_GROUP_SET_CONTAINER, &container) < 0) {
    fprintf(stderr, "VFIO_GROUP_SET_CONTAINER failed: %s\n", strerror(errno));
    close(group);
    close(container);
    return 1;
  }
  if (ioctl(container, VFIO_SET_IOMMU, VFIO_TYPE1v2_IOMMU) < 0 &&
      ioctl(container, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU) < 0) {
    fprintf(stderr, "VFIO_SET_IOMMU failed: %s\n", strerror(errno));
    close(group);
    close(container);
    return 1;
  }
  int dev = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, bdf.c_str());
  if (dev < 0) {
    fprintf(stderr, "VFIO_GROUP_GET_DEVICE_FD(%s) failed: %s\n", bdf.c_str(), strerror(errno));
    close(group);
    close(container);
    return 1;
  }

  struct vfio_region_info r0 = {}, r4 = {};
  r0.argsz = sizeof(r0); r0.index = VFIO_PCI_BAR0_REGION_INDEX;
  r4.argsz = sizeof(r4); r4.index = VFIO_PCI_BAR4_REGION_INDEX;
  if (ioctl(dev, VFIO_DEVICE_GET_REGION_INFO, &r0) < 0 ||
      ioctl(dev, VFIO_DEVICE_GET_REGION_INFO, &r4) < 0) {
    fprintf(stderr, "VFIO_DEVICE_GET_REGION_INFO failed: %s\n", strerror(errno));
    close(dev);
    close(group);
    close(container);
    return 1;
  }

  printf("=== BAR4 FULL FIRST 4 KiB NONZERO DUMP ===\n");
  void* m4 = mmap(nullptr, r4.size, PROT_READ | PROT_WRITE, MAP_SHARED, dev, r4.offset);
  if (m4 != MAP_FAILED) {
    const volatile uint8_t* b4 = reinterpret_cast<const volatile uint8_t*>(m4);
    dump_hex_ascii(b4, 0, 4096);

    printf("\n=== BAR4 FULL 16 MiB SCAN (every 64 bytes) ===\n");
    int hits4 = 0;
    for (size_t off = 0; off < r4.size; off += 64) {
      bool interesting = false;
      uint32_t vals[4] = {};
      for (int i = 0; i < 4; ++i) {
        if (!safe_r32(b4 + off + i * 4, &vals[i])) {
          interesting = true;
          break;
        }
        if (vals[i] != 0 && vals[i] != 0xffffffffu) interesting = true;
      }
      if (interesting && off >= 4096) {
        if (hits4 < 128) {
          printf("  BAR4+0x%06zx: %08x %08x %08x %08x\n",
                 off, vals[0], vals[1], vals[2], vals[3]);
        }
        hits4++;
      }
    }
    printf("BAR4 >=4KiB interesting 64B-aligned hits: %d\n", hits4);
    munmap(m4, r4.size);
  }

  printf("\n=== BAR0 DENSE FIRST 256 KiB SCAN + 64 KiB STRIDE ACROSS 4 GiB ===\n");
  // Map BAR0 in 64 MiB chunks to scan 4 GiB at 64 KiB stride (65536 reads total)
  const size_t chunk_sz = 64ULL * 1024ULL * 1024ULL;
  int bar0_hits = 0;
  for (uint64_t base_off = 0; base_off < r0.size; base_off += chunk_sz) {
    void* m0 = mmap(nullptr, chunk_sz, PROT_READ, MAP_SHARED, dev,
                    static_cast<off_t>(r0.offset + base_off));
    if (m0 == MAP_FAILED) {
      printf("  mmap BAR0+0x%llx failed err=%d\n", (unsigned long long)base_off, errno);
      break;
    }
    const volatile uint8_t* b0 = reinterpret_cast<const volatile uint8_t*>(m0);
    if (base_off == 0) {
      printf("  BAR0 [0x00000..0x40000] non-zero/non-ff 4KiB pages:\n");
      for (size_t pg = 0; pg < 256 * 1024; pg += 4096) {
        int nz = 0, zr = 0, ff = 0;
        uint32_t first_nz_off = 0, first_nz_val = 0;
        for (size_t w = 0; w < 4096; w += 4) {
          uint32_t v = 0xffffffffu;
          if (!safe_r32(b0 + pg + w, &v)) continue;
          if (v == 0xffffffffu) ff++;
          else if (v == 0) zr++;
          else {
            if (nz == 0) { first_nz_off = (uint32_t)w; first_nz_val = v; }
            nz++;
          }
        }
        if (nz > 0 || zr > 0) {
          printf("    page 0x%05zx: nz=%d zero=%d ff=%d first_nz=(+0x%x: 0x%08x)\n",
                 pg, nz, zr, ff, first_nz_off, first_nz_val);
        }
      }
    }
    // Check every 64 KiB across this 64 MiB chunk at offsets +0x0, +0x4, +0x8, +0x100
    for (size_t off = 0; off < chunk_sz; off += 65536) {
      uint32_t v0 = 0xffffffffu, v4 = 0xffffffffu, v8 = 0xffffffffu, v100 = 0xffffffffu;
      bool ok = safe_r32(b0 + off, &v0) && safe_r32(b0 + off + 4, &v4) &&
                safe_r32(b0 + off + 8, &v8) && safe_r32(b0 + off + 0x100, &v100);
      if (!ok || (v0 != 0xffffffffu) || (v4 != 0xffffffffu) ||
          (v8 != 0xffffffffu) || (v100 != 0xffffffffu)) {
        if (bar0_hits < 200) {
          printf("  BAR0+0x%09llx: ok=%d [0]=0x%08x [4]=0x%08x [8]=0x%08x [100]=0x%08x\n",
                 (unsigned long long)(base_off + off), ok, v0, v4, v8, v100);
        }
        bar0_hits++;
      }
    }
    munmap(m0, chunk_sz);
  }
  printf("Total BAR0 64KiB non-0xffffffff hits across 4 GiB: %d\n", bar0_hits);

  close(dev);
  close(group);
  close(container);
  printf("=== M1_SCAN2_COMPLETE ===\n");
  return 0;
}
