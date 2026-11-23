// SPDX-License-Identifier: Apache-2.0
// Standalone C++17 M1/M2/M3/M4 hardware verification smoke binary for libopentpu.so.
// Verifies zero libtpu.so and zero libpython in /proc/self/maps.

#include "opentpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string mapped_basename(const std::string& maps_line) {
  // /proc/self/maps format: addr perms offset dev inode [pathname]
  size_t pos = 0;
  for (int field = 0; field < 5; ++field) {
    while (pos < maps_line.size() && maps_line[pos] == ' ') ++pos;
    while (pos < maps_line.size() && maps_line[pos] != ' ') ++pos;
  }
  while (pos < maps_line.size() && maps_line[pos] == ' ') ++pos;
  if (pos >= maps_line.size()) return "";
  std::string path = maps_line.substr(pos);
  const std::string deleted_suffix = " (deleted)";
  if (path.size() >= deleted_suffix.size() &&
      path.compare(path.size() - deleted_suffix.size(), deleted_suffix.size(), deleted_suffix) == 0) {
    path.erase(path.size() - deleted_suffix.size());
  }
  size_t slash = path.rfind('/');
  return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

bool check_maps() {
  std::ifstream ifs("/proc/self/maps");
  if (!ifs.is_open()) {
    printf("MAPS_CHECK=SKIPPED (/proc/self/maps unavailable on this host)\n");
    return true;
  }
  std::string line;
  int libtpu_cnt = 0;
  int libpython_cnt = 0;
  while (std::getline(ifs, line)) {
    std::string base = mapped_basename(line);
    if (base.compare(0, 6, "libtpu") == 0) libtpu_cnt++;
    if (base.compare(0, 9, "libpython") == 0) libpython_cnt++;
  }
  printf("MAPS_LIBTPU=%d MAPS_LIBPYTHON=%d\n", libtpu_cnt, libpython_cnt);
  if (libtpu_cnt > 0 || libpython_cnt > 0) {
    fprintf(stderr, "ERROR: forbidden dependency mapped (MAPS_LIBTPU=%d MAPS_LIBPYTHON=%d)\n",
            libtpu_cnt, libpython_cnt);
    return false;
  }
  return true;
}

void print_usage(FILE* out, const char* prog) {
  fprintf(out, "Usage: %s [--mock] [--bdf <PCI_BDF>] [--help]\n", prog ? prog : "opentpu_vfio_smoke");
}

}  // namespace

int main(int argc, char** argv) {
  ot_config_t cfg = ot_config_default();
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(stdout, argv[0]);
      return 0;
    } else if (std::strcmp(argv[i], "--mock") == 0) {
      cfg.mock_mode = 1;
    } else if (std::strcmp(argv[i], "--bdf") == 0) {
      if (i + 1 >= argc || argv[i + 1][0] == '\0') {
        fprintf(stderr, "ERROR: --bdf requires a valid PCI BDF argument\n");
        print_usage(stderr, argv[0]);
        return 2;
      }
      cfg.bdf = argv[++i];
    } else {
      fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
      print_usage(stderr, argv[0]);
      return 2;
    }
  }

  if (!check_maps()) {
    return 3;
  }

  ot_context_t* ctx = nullptr;
  ot_status_t st = ot_context_create(&cfg, &ctx);
  if (st != OT_OK) {
    const char* detail = ot_context_last_error(nullptr);
    fprintf(stderr, "ot_context_create failed: %s%s%s\n",
            ot_status_string(st),
            (detail && detail[0]) ? " — " : "",
            (detail && detail[0]) ? detail : "");
    return 1;
  }

  ot_device_info_t info = {};
  ot_device_get_info(ctx, &info);
  printf("OpenTPU device: bdf=%s vid=0x%04x did=0x%04x subsys=0x%04x:0x%04x "
         "group=%u bar0=%lluMiB bar4=%lluMiB msix=%u prebooted=%u mock=%u\n",
         info.bdf, info.vendor_id, info.device_id, info.subsys_vendor_id,
         info.subsys_device_id, info.iommu_group,
         (unsigned long long)(info.bar0_size / (1024ULL * 1024ULL)),
         (unsigned long long)(info.bar4_size / (1024ULL * 1024ULL)),
         info.msix_vectors, info.is_prebooted, info.is_mock);

  // M2 H2D -> D2H DMA round-trip across 4 KiB, 64 KiB, 1 MiB
  const size_t dma_sizes[] = {4096u, 65536u, 1048576u};
  int dma_failures = 0;
  for (size_t sz : dma_sizes) {
    ot_buffer_t* buf = nullptr;
    if (ot_buffer_alloc(ctx, sz, &buf) != OT_OK) {
      printf("M2_DMA_ROUNDTRIP size=%zu: alloc_failed (%s)\n", sz, ot_context_last_error(ctx));
      ++dma_failures;
      continue;
    }
    const size_t nwords = sz / sizeof(uint32_t);
    std::vector<uint32_t> src(nwords), dst(nwords, 0);
    for (size_t i = 0; i < nwords; ++i) {
      src[i] = 0xCAFE0000u ^ static_cast<uint32_t>(i * 2654435761u);
    }
    ot_status_t h2d = ot_buffer_h2d(ctx, buf, src.data(), sz);
    ot_status_t clobber_st = OT_OK;
    if (h2d == OT_OK) {
      // Overwrite the DMA staging window with an inverted pattern via a second HBM buffer
      // so a stale staging window cannot masquerade as a successful D2H transfer.
      ot_buffer_t* clobber_buf = nullptr;
      clobber_st = ot_buffer_alloc(ctx, sz, &clobber_buf);
      if (clobber_st == OT_OK) {
        std::vector<uint32_t> inverted(nwords);
        for (size_t i = 0; i < nwords; ++i) inverted[i] = ~src[i];
        clobber_st = ot_buffer_h2d(ctx, clobber_buf, inverted.data(), sz);
        ot_buffer_free(clobber_buf);
      }
    }
    ot_status_t d2h = (h2d == OT_OK && clobber_st == OT_OK)
                          ? ot_buffer_d2h(ctx, buf, dst.data(), sz)
                          : (h2d != OT_OK ? h2d : clobber_st);
    if (h2d == OT_OK && d2h == OT_OK) {
      bool match = (std::memcmp(src.data(), dst.data(), sz) == 0);
      printf("M2_DMA_ROUNDTRIP size=%zu: match=%d first=0x%08x last=0x%08x\n",
             sz, match ? 1 : 0, dst[0], dst[nwords - 1]);
      if (!match) ++dma_failures;
    } else {
      printf("M2_DMA_ROUNDTRIP size=%zu: pending (%s)\n", sz, ot_context_last_error(ctx));
      ++dma_failures;
    }
    ot_buffer_free(buf);
  }

  ot_context_destroy(ctx);
  return dma_failures == 0 ? 0 : 2;
}
