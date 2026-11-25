// Clean-room /dev/edgetpu-soc live UAPI + SMMUv3 DMA-BUF + minimal INT8 GEMM probe.
// Designed for standalone Linux ARM64 (OmaMobile / Arch Linux ARM) and host --mock CI.
#include "openbuenos.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/ioctl.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "edgetpu_buenos_uapi.h"
#endif

namespace {

#if defined(__linux__)
struct LocalDmaHeapAllocationData {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};

#define PROBE_DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, LocalDmaHeapAllocationData)

struct ScopedFd {
  int fd = -1;
  ~ScopedFd() { reset(); }
  ScopedFd() = default;
  explicit ScopedFd(int f) : fd(f) {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  void reset(int next = -1) {
    if (fd >= 0) ::close(fd);
    fd = next;
  }
};

bool run_uapi_probe(const char* dev_path, const char* dma_heap_path) {
  ScopedFd tpu(::open(dev_path, O_RDWR | O_CLOEXEC));
  if (tpu.fd < 0) {
    std::fprintf(stderr, "uapi: open(%s) failed: errno=%d (%s)\n",
                 dev_path, errno, std::strerror(errno));
    return false;
  }

  struct edgetpu_interface_version_ioctl iface {};
  if (::ioctl(tpu.fd, EDGETPU_GET_INTERFACE_VERSION, &iface) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_GET_INTERFACE_VERSION failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    return false;
  }
  iface.version_build[sizeof(iface.version_build) - 1] = '\0';

  uint64_t credit_size = 0;
  bool credit_probed = (::ioctl(tpu.fd, EDGETPU_GET_CREDIT_SIZE, &credit_size) == 0);
  if (!credit_probed) {
    if (errno != ENOTTY) {
      std::fprintf(stderr, "uapi: EDGETPU_GET_CREDIT_SIZE failed: errno=%d (%s)\n",
                   errno, std::strerror(errno));
      return false;
    }
    credit_size = 8u;
  }

  struct edgetpu_mailbox_attr attr {};
  attr.cmd_queue_size = 24;
  attr.resp_queue_size = 12;
  attr.sizeof_cmd = 48;
  attr.sizeof_resp = 24;
  attr.priority = 8;
  attr.cmdq_tail_doorbell = 1;
  attr.partition_type = 0;
  attr.client_priv = 1;
  if (::ioctl(tpu.fd, EDGETPU_CREATE_GROUP, &attr) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_CREATE_GROUP failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    return false;
  }

  ScopedFd resp_efd(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK));
  ScopedFd err_efd(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK));
  if (resp_efd.fd < 0 || err_efd.fd < 0) {
    std::fprintf(stderr, "uapi: eventfd() failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    return false;
  }

  struct edgetpu_event_register ev0 { EDGETPU_EVENT_RESPDATA, static_cast<__u32>(resp_efd.fd) };
  if (::ioctl(tpu.fd, EDGETPU_SET_EVENTFD, &ev0) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_SET_EVENTFD(RESPDATA) failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    return false;
  }
  struct edgetpu_event_register ev1 { EDGETPU_EVENT_FATAL_ERROR, static_cast<__u32>(err_efd.fd) };
  if (::ioctl(tpu.fd, EDGETPU_SET_EVENTFD, &ev1) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_SET_EVENTFD(FATAL_ERROR) failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    return false;
  }

  if (::ioctl(tpu.fd, EDGETPU_ACQUIRE_WAKE_LOCK) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_ACQUIRE_WAKE_LOCK failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    return false;
  }

  struct edgetpu_fw_version fw {};
  if (::ioctl(tpu.fd, EDGETPU_FIRMWARE_VERSION, &fw) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_FIRMWARE_VERSION failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }

  __u32 fatal_mask = 0;
  if (::ioctl(tpu.fd, EDGETPU_GET_FATAL_ERRORS, &fatal_mask) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_GET_FATAL_ERRORS failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }

  ScopedFd heap(::open(dma_heap_path, O_RDONLY | O_CLOEXEC));
  if (heap.fd < 0) {
    std::fprintf(stderr, "uapi: open(%s) failed: errno=%d (%s)\n",
                 dma_heap_path, errno, std::strerror(errno));
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }

  LocalDmaHeapAllocationData alloc {};
  alloc.len = 4096u;
  alloc.fd_flags = O_RDWR | O_CLOEXEC;
  alloc.heap_flags = 0;
  if (::ioctl(heap.fd, PROBE_DMA_HEAP_IOCTL_ALLOC, &alloc) != 0) {
    std::fprintf(stderr, "uapi: DMA_HEAP_IOCTL_ALLOC(4096) failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }
  ScopedFd dmabuf(static_cast<int>(alloc.fd));

  void* mapped = ::mmap(nullptr, 4096u, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf.fd, 0);
  if (mapped == MAP_FAILED) {
    std::fprintf(stderr, "uapi: mmap(dmabuf) failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }
  auto* words = static_cast<volatile uint32_t*>(mapped);
  words[0] = 0x54505531u;
  const uint32_t cpu_check = words[0];
  ::munmap(mapped, 4096u);
  if (cpu_check != 0x54505531u) {
    std::fprintf(stderr, "uapi: dmabuf CPU readback mismatch (0x%08x)\n", cpu_check);
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }

  struct edgetpu_map_dmabuf_ioctl map_req {};
  map_req.offset = 0;
  map_req.size = 4096u;
  map_req.dmabuf_fd = dmabuf.fd;
  map_req.flags = EDGETPU_MAP_DMA_BIDIRECTIONAL | EDGETPU_MAP_36BIT_IOVA;
  map_req.die_index = 0;
  if (::ioctl(tpu.fd, EDGETPU_MAP_DMABUF, &map_req) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_MAP_DMABUF failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }
  if (map_req.device_address == 0 || (map_req.device_address >> 36) != 0) {
    std::fprintf(stderr, "uapi: invalid SMMU device_address=0x%llx\n",
                 static_cast<unsigned long long>(map_req.device_address));
    ::ioctl(tpu.fd, EDGETPU_UNMAP_DMABUF, &map_req);
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }

  if (::ioctl(tpu.fd, EDGETPU_UNMAP_DMABUF, &map_req) != 0) {
    std::fprintf(stderr, "uapi: EDGETPU_UNMAP_DMABUF failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }

  if (::ioctl(tpu.fd, EDGETPU_FINALIZE_GROUP) != 0 && errno != ENOTTY) {
    std::fprintf(stderr, "uapi: EDGETPU_FINALIZE_GROUP failed: errno=%d (%s)\n",
                 errno, std::strerror(errno));
    ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);
    return false;
  }
  ::ioctl(tpu.fd, EDGETPU_RELEASE_WAKE_LOCK);

  std::printf("uapi_probe: PASS dev=%s iface=%u.%u build=\"%s\" credits=%llu(%s) "
              "fw=%u.%u vii=%u kci=%u fatal=0x%x dmabuf_iova=0x%llx\n",
              dev_path,
              static_cast<unsigned>(iface.version_major),
              static_cast<unsigned>(iface.version_minor),
              iface.version_build,
              static_cast<unsigned long long>(credit_size),
              credit_probed ? "ioctl" : "fallback",
              static_cast<unsigned>(fw.major_version),
              static_cast<unsigned>(fw.minor_version),
              static_cast<unsigned>(fw.vii_version),
              static_cast<unsigned>(fw.kci_version),
              static_cast<unsigned>(fatal_mask),
              static_cast<unsigned long long>(map_req.device_address));
  return fatal_mask == 0u;
}
#endif

bool run_libopenbuenos_gemm(const char* dev_path,
                            const char* dma_heap_path,
                            const char* dive_interp_path,
                            int perf_mode,
                            bool mock,
                            bool require_tpu_gemm) {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = mock ? 1 : 0;
  cfg.device_path = dev_path;
  cfg.dma_heap_path = dma_heap_path;
  cfg.max_inflight = 16u;
  if (perf_mode >= 0) cfg.perf_mode = static_cast<uint8_t>(perf_mode);

  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "libopenbuenos: ob_context_create failed: %s\n",
                 ob_status_string(st));
    return false;
  }

  ob_buffer_t* buf = nullptr;
  st = ob_buffer_alloc(ctx, 4096u, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &buf);
  if (st != OB_OK || !buf) {
    std::fprintf(stderr, "libopenbuenos: ob_buffer_alloc failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return false;
  }

  const uint32_t tx_pat[4] = {0x11223344u, 0x55667788u, 0xCAFEBABEu, 0x0BADF00Du};
  uint32_t rx_pat[4] = {};
  if (ob_buffer_write(buf, tx_pat, sizeof(tx_pat), 0u) != OB_OK ||
      ob_buffer_read(buf, rx_pat, sizeof(rx_pat), 0u) != OB_OK ||
      std::memcmp(tx_pat, rx_pat, sizeof(tx_pat)) != 0) {
    std::fprintf(stderr, "libopenbuenos: buffer write/read mismatch: %s\n",
                 ob_context_last_error(ctx));
    ob_buffer_free(buf);
    ob_context_destroy(ctx);
    return false;
  }
  const uint64_t buf_iova = ob_buffer_iova(buf);
  ob_buffer_free(buf);

  bool emitter_attached = false;
  st = ob_gemm_register_builtin_emitter(ctx, dive_interp_path);
  if (st == OB_OK) {
    emitter_attached = true;
  } else if (require_tpu_gemm) {
    std::fprintf(stderr, "libopenbuenos: builtin emitter required but unavailable: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return false;
  }

  constexpr uint32_t M = 1u, K = 64u, N = 64u;
  std::vector<int8_t> a_mk(M * K, 0);
  std::vector<int8_t> b_nk(N * K, 0);
  std::vector<int32_t> bias_n(N, 0);
  std::vector<int32_t> c_out(M * N, 0);
  std::vector<int32_t> c_ref(M * N, 0);

  // Multiples of 6 ensure exact divisibility by compute_tile_scale_D() (=3 for this row norm)
  // so hardware INT8 requantization yields bit-exact worst_err=0.
  for (uint32_t k = 0; k < K; ++k) {
    a_mk[k] = static_cast<int8_t>((static_cast<int32_t>(k % 7u) - 3) * 6);
  }
  for (uint32_t n = 0; n < N; ++n) {
    bias_n[n] = static_cast<int32_t>((n * 5u) % 19u) - 9;
    b_nk[n * K + ((n * 3u) & 63u)] = 1;
    b_nk[n * K + ((n * 3u + 4u) & 63u)] = -1;
  }
  for (uint32_t n = 0; n < N; ++n) {
    int32_t acc = bias_n[n];
    for (uint32_t k = 0; k < K; ++k) {
      acc += static_cast<int32_t>(a_mk[k]) * static_cast<int32_t>(b_nk[n * K + k]);
    }
    c_ref[n] = acc;
  }

  ob_invoke_stats_t stats {};
  st = ob_gemm_matmul_int8(ctx, a_mk.data(), b_nk.data(), bias_n.data(), c_out.data(),
                           M, K, N, 0, 0, &stats);
  if (st != OB_OK) {
    std::fprintf(stderr, "libopenbuenos: ob_gemm_matmul_int8 failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return false;
  }

  int64_t worst_err = 0;
  for (size_t i = 0; i < c_ref.size(); ++i) {
    int64_t d = static_cast<int64_t>(c_out[i]) - static_cast<int64_t>(c_ref[i]);
    if (d < 0) d = -d;
    worst_err = std::max(worst_err, d);
  }
  if (worst_err != 0) {
    std::fprintf(stderr, "libopenbuenos: GEMM mismatch worst_err=%lld\n",
                 static_cast<long long>(worst_err));
    ob_context_destroy(ctx);
    return false;
  }

  const bool emitter_used = (stats.flags & OB_STAT_FLAG_EMITTER_USED) != 0u;
  const bool freetile_used = (stats.flags & OB_STAT_FLAG_FREETILE) != 0u;
  if (require_tpu_gemm && (!emitter_used || (!mock && stats.hw_execution_us == 0u))) {
    std::fprintf(stderr, "libopenbuenos: TPU GEMM check failed (emitter_used=%d hw_us=%llu)\n",
                 emitter_used ? 1 : 0,
                 static_cast<unsigned long long>(stats.hw_execution_us));
    ob_context_destroy(ctx);
    return false;
  }

  std::printf("gemm_probe: PASS shape=1x64x64 mock=%d credits=%u buf_iova=0x%llx "
              "emitter=%s freetile=%d flags=0x%x hw_us=%llu wall_us=%llu worst_err=%lld\n",
              ob_context_is_mock(ctx),
              static_cast<unsigned>(ob_context_credit_size(ctx)),
              static_cast<unsigned long long>(buf_iova),
              emitter_attached ? (emitter_used ? "attached+used" : "attached") : "none(cpu_fallback)",
              (ob_context_get_freetile(ctx) || freetile_used) ? 1 : 0,
              static_cast<unsigned>(stats.flags),
              static_cast<unsigned long long>(stats.hw_execution_us),
              static_cast<unsigned long long>(stats.wall_time_us),
              static_cast<long long>(worst_err));

  ob_context_destroy(ctx);
  return true;
}

void print_usage(const char* argv0) {
  std::fprintf(stderr,
      "Usage: %s [--dev /dev/edgetpu-soc] [--dma-heap /dev/dma_heap/system] "
      "[--dive-interp <path>] [--perf-mode <0..5>] [--mock] [--no-gemm] [--require-tpu-gemm]\n",
      argv0);
}

}  // namespace

int main(int argc, char** argv) {
  const char* dev_path = "/dev/edgetpu-soc";
  const char* dma_heap_path = "/dev/dma_heap/system";
  const char* dive_interp_path = nullptr;
  int perf_mode = -1;
  bool mock = false;
  bool run_gemm = true;
  bool require_tpu_gemm = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dev") == 0 && i + 1 < argc) {
      dev_path = argv[++i];
    } else if (std::strcmp(argv[i], "--dma-heap") == 0 && i + 1 < argc) {
      dma_heap_path = argv[++i];
    } else if (std::strcmp(argv[i], "--dive-interp") == 0 && i + 1 < argc) {
      dive_interp_path = argv[++i];
    } else if (std::strcmp(argv[i], "--perf-mode") == 0 && i + 1 < argc) {
      perf_mode = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--mock") == 0) {
      mock = true;
    } else if (std::strcmp(argv[i], "--no-gemm") == 0) {
      run_gemm = false;
    } else if (std::strcmp(argv[i], "--require-tpu-gemm") == 0) {
      require_tpu_gemm = true;
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      print_usage(argv[0]);
      return 2;
    }
  }

  if (mock) {
    std::printf("uapi_probe: SKIP (mock mode enabled; dev=%s heap=%s)\n", dev_path, dma_heap_path);
  } else {
#if defined(__linux__)
    if (!run_uapi_probe(dev_path, dma_heap_path)) return 1;
#else
    std::fprintf(stderr, "uapi_probe: live /dev/edgetpu-soc probe requires Linux (pass --mock on host)\n");
    return 1;
#endif
  }

  if (run_gemm) {
    if (!run_libopenbuenos_gemm(dev_path, dma_heap_path, dive_interp_path,
                                perf_mode, mock, require_tpu_gemm)) {
      return 1;
    }
  }
  return 0;
}
