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

void check_maps() {
  std::ifstream ifs("/proc/self/maps");
  if (!ifs.is_open()) {
    printf("MAPS_LIBTPU=0 MAPS_LIBPYTHON=0 (non-proc host)\n");
    return;
  }
  std::string line;
  int libtpu_cnt = 0;
  int libpython_cnt = 0;
  while (std::getline(ifs, line)) {
    if (line.find("libtpu") != std::string::npos) libtpu_cnt++;
    if (line.find("libpython") != std::string::npos) libpython_cnt++;
  }
  printf("MAPS_LIBTPU=%d MAPS_LIBPYTHON=%d\n", libtpu_cnt, libpython_cnt);
}

}  // namespace

int main(int argc, char** argv) {
  check_maps();

  ot_config_t cfg = ot_config_default();
  if (argc > 1 && std::strcmp(argv[1], "--mock") == 0) {
    cfg.mock_mode = 1;
  }

  ot_context_t* ctx = nullptr;
  ot_status_t st = ot_context_create(&cfg, &ctx);
  if (st != OT_OK) {
    fprintf(stderr, "ot_context_create failed: %s\n", ot_status_string(st));
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
    ot_status_t d2h = (h2d == OT_OK) ? ot_buffer_d2h(ctx, buf, dst.data(), sz) : h2d;
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
