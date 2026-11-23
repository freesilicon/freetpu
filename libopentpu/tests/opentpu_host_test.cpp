// SPDX-License-Identifier: Apache-2.0
#include "opentpu.h"
#include "ot_internal.h"
#include "ot_vliw.h"
#include "pjrt_c_api.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern "C" const PJRT_Api* GetPjrtApi(void);

#define OT_CHECK(expr)                                                   \
  do {                                                                   \
    if (!(expr)) {                                                       \
      std::fprintf(stderr, "OT_CHECK FAILED (%s:%d): %s\n", __FILE__,    \
                   __LINE__, #expr);                                     \
      std::abort();                                                      \
    }                                                                    \
  } while (0)

static void destroy_pjrt_error(const PJRT_Api* api, PJRT_Error* err) {
  if (!err) return;
  PJRT_Error_Destroy_Args d = {};
  d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  d.error = err;
  api->PJRT_Error_Destroy(&d);
}

namespace {

inline float test_bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline uint16_t test_f32_to_bf16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<uint16_t>((u >> 16) | 0x0040U);
  }
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7fffU + lsb;
  return static_cast<uint16_t>(u >> 16);
}

inline uint64_t splitmix64(uint64_t* state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

void fill_random_bf16(std::vector<uint16_t>* dst, uint64_t seed, int half_steps, float scale) {
  const uint64_t span = static_cast<uint64_t>(2 * half_steps + 1);
  uint64_t rng = seed;
  for (uint16_t& v : *dst) {
    int q = static_cast<int>(splitmix64(&rng) % span) - half_steps;
    v = test_f32_to_bf16(static_cast<float>(q) * scale);
  }
}

std::vector<uint16_t> ref_matmul(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                                 uint32_t m, uint32_t k, uint32_t n) {
  std::vector<uint16_t> out(static_cast<size_t>(m) * n, 0);
  for (uint32_t r = 0; r < m; ++r) {
    for (uint32_t c = 0; c < n; ++c) {
      float acc = 0.0f;
      for (uint32_t j = 0; j < k; ++j) {
        acc += test_bf16_to_f32(a[static_cast<size_t>(r) * k + j]) *
               test_bf16_to_f32(b[static_cast<size_t>(j) * n + c]);
      }
      out[static_cast<size_t>(r) * n + c] = test_f32_to_bf16(acc);
    }
  }
  return out;
}

}  // namespace

static void test_iova_allocator() {
  ot::IovaAllocator alloc;
  alloc.init(0x100000ULL, 0x200000ULL);
  uint64_t a = 0, b = 0, c = 0;
  OT_CHECK(alloc.alloc(4096, 4096, &a) && a == 0x100000ULL);
  OT_CHECK(alloc.alloc(8192, 4096, &b) && b == 0x101000ULL);
  OT_CHECK(alloc.used_bytes() == 12288);
  alloc.free(a);
  OT_CHECK(alloc.alloc(4096, 4096, &c) && c == 0x100000ULL);
  alloc.free(c);
  alloc.free(b);
  OT_CHECK(alloc.used_bytes() == 0);

  // Full-arena alloc -> free -> split into two 4 KiB blocks without losing capacity.
  uint64_t full = 0, s1 = 0, s2 = 0;
  OT_CHECK(alloc.alloc(0x100000ULL, 4096, &full) && full == 0x100000ULL);
  OT_CHECK(!alloc.alloc(4096, 4096, &s1));
  alloc.free(full);
  OT_CHECK(alloc.alloc(4096, 4096, &s1) && s1 == 0x100000ULL);
  OT_CHECK(alloc.alloc(4096, 4096, &s2) && s2 == 0x101000ULL);
  OT_CHECK(alloc.used_bytes() == 8192);
  printf("[PASS] test_iova_allocator\n");
}

static void test_bar4_descriptor_parser() {
  std::vector<uint8_t> page(4096, 0);
  // Synthetic protobuf:
  // field 2 (tag 0x12), len 2, bytes {0x08, 0x01}
  // field 3 (tag 0x1a), len 5, "TPUv5"
  const uint8_t proto[] = {0x12, 0x02, 0x08, 0x01, 0x1a, 0x05, 'T', 'P', 'U', 'v', '5'};
  uint32_t plen = sizeof(proto);
  uint32_t hdr = 0x00140001u;
  std::memcpy(page.data() + 0x08, &plen, 4);
  std::memcpy(page.data() + 0x0c, &hdr, 4);
  std::memcpy(page.data() + 0x10, proto, plen);

  ot::Bar4Descriptor desc;
  OT_CHECK(ot::parse_bar4_descriptor(page.data(), page.size(), &desc));
  OT_CHECK(desc.valid);
  OT_CHECK(desc.proto_len == plen);
  OT_CHECK(desc.header_word == hdr);
  OT_CHECK(desc.core_count_observed == 1);
  OT_CHECK(desc.instance_token == "TPUv5");

  // Unmapped/error BAR4 reads (all 0xff) and blank/zero-field pages must be rejected.
  std::vector<uint8_t> ff_page(4096, 0xff);
  OT_CHECK(!ot::parse_bar4_descriptor(ff_page.data(), ff_page.size(), &desc));
  OT_CHECK(!desc.valid);
  std::vector<uint8_t> zero_page(4096, 0x00);
  OT_CHECK(!ot::parse_bar4_descriptor(zero_page.data(), zero_page.size(), &desc));
  uint32_t bad_len = 4;
  std::memcpy(zero_page.data() + 0x08, &bad_len, 4);
  OT_CHECK(!ot::parse_bar4_descriptor(zero_page.data(), zero_page.size(), &desc));
  printf("[PASS] test_bar4_descriptor_parser\n");
}

static void test_aligned_dma_arena_alloc() {
  std::string err;
  constexpr size_t kTwoMiB = 2u * 1024u * 1024u;
  void* p = ot::alloc_aligned_anon_pages(kTwoMiB, kTwoMiB, &err);
  OT_CHECK(p != nullptr);
  OT_CHECK((reinterpret_cast<uintptr_t>(p) & (kTwoMiB - 1u)) == 0);
  auto* bytes = static_cast<uint8_t*>(p);
  bytes[0] = 0x5au;
  bytes[kTwoMiB - 1u] = 0xa5u;
  OT_CHECK(bytes[0] == 0x5au && bytes[kTwoMiB - 1u] == 0xa5u);
  OT_CHECK(::munmap(p, kTwoMiB) == 0);

  // Non-power-of-two alignment and zero length must fail.
  OT_CHECK(ot::alloc_aligned_anon_pages(kTwoMiB, 3u * 4096u, &err) == nullptr);
  OT_CHECK(ot::alloc_aligned_anon_pages(0, kTwoMiB, &err) == nullptr);
  printf("[PASS] test_aligned_dma_arena_alloc\n");
}

static void test_fork_closes_vfio_fds_and_static_pjrt_err() {
  // Verify pthread_atfork releases inherited VFIO descriptors in a forked child
  // without closing the parent's descriptors.
  int p0[2] = {-1, -1};
  int p1[2] = {-1, -1};
  int p2[2] = {-1, -1};
  OT_CHECK(::pipe(p0) == 0 && ::pipe(p1) == 0 && ::pipe(p2) == 0);
  {
    ot::VfioTransport tr;
    std::string err;
    OT_CHECK(tr.register_fork_cleanup(&err));
    tr.is_mock = false;
    tr.owner_pid = static_cast<int64_t>(::getpid());
    tr.container_fd = p0[0];
    tr.group_fd = p1[0];
    tr.device_fd = p2[0];
    int tracked_fds[3] = {p0[0], p1[0], p2[0]};

    pid_t pid = ::fork();
    OT_CHECK(pid >= 0);
    if (pid == 0) {
      bool ok = true;
      for (int fd : tracked_fds) {
        errno = 0;
        if (::fcntl(fd, F_GETFD) != -1 || errno != EBADF) ok = false;
      }
      if (tr.container_fd != -1 || tr.group_fd != -1 || tr.device_fd != -1) ok = false;
      if (!tr.dma_faulted || tr.is_owner_process()) ok = false;
      ::_exit(ok ? 0 : 1);
    }
    int status = 0;
    OT_CHECK(::waitpid(pid, &status, 0) == pid);
    OT_CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    // Parent descriptors remain open until parent closes the transport.
    for (int fd : tracked_fds) {
      OT_CHECK(::fcntl(fd, F_GETFD) != -1);
    }
    tr.close_device();
    for (int fd : tracked_fds) {
      errno = 0;
      OT_CHECK(::fcntl(fd, F_GETFD) == -1 && errno == EBADF);
    }
  }
  ::close(p0[1]);
  ::close(p1[1]);
  ::close(p2[1]);

  // Verify allocation-free static fallback PJRT_Error behaves cleanly across code/msg/destroy.
  const PJRT_Api* api = GetPjrtApi();
  auto* fallback = reinterpret_cast<PJRT_Error*>(
      ot::pjrt_static_fallback_error_raw(PJRT_Error_Code_RESOURCE_EXHAUSTED));
  OT_CHECK(fallback != nullptr);
  PJRT_Error_GetCode_Args gc = {};
  gc.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  gc.error = fallback;
  OT_CHECK(api->PJRT_Error_GetCode(&gc) == nullptr);
  OT_CHECK(gc.code == PJRT_Error_Code_RESOURCE_EXHAUSTED);
  PJRT_Error_Message_Args gm = {};
  gm.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  gm.error = fallback;
  api->PJRT_Error_Message(&gm);
  OT_CHECK(gm.message != nullptr && gm.message_size > 0);
  // Destroying the static fallback error repeatedly must be a safe no-op.
  destroy_pjrt_error(api, fallback);
  destroy_pjrt_error(api, fallback);

  // Verify forked-child VFIO API calls return OT_ERR_DEVICE without deadlocking
  // even when another thread in the parent held ctx->mu across fork().
  ot_config_t cfg = ot_config_default();
  cfg.mock_mode = 1;
  ot_context_t* ctx = nullptr;
  OT_CHECK(ot_context_create(&cfg, &ctx) == OT_OK);
  ot_buffer_t *ba = nullptr, *bb = nullptr, *bc = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx, 16, &ba) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, 16, &bb) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, 16, &bc) == OT_OK);
  ctx->info.is_mock = 0u;
  ctx->transport.is_mock = false;
  ctx->transport.owner_pid = static_cast<int64_t>(::getpid());
  for (ot_buffer_t* b : {ba, bb, bc}) {
    b->is_mock = false;
    b->owner_pid = ctx->transport.owner_pid;
  }
  ctx->mu.lock();
  pid_t cpid = ::fork();
  OT_CHECK(cpid >= 0);
  if (cpid == 0) {
    ot_buffer_t* out = nullptr;
    uint8_t scratch[16] = {};
    bool ok = true;
    if (ot_buffer_alloc(ctx, 16, &out) != OT_ERR_DEVICE || out != nullptr) ok = false;
    if (ot_buffer_h2d(ctx, ba, scratch, sizeof(scratch)) != OT_ERR_DEVICE) ok = false;
    if (ot_buffer_d2h(ctx, ba, scratch, sizeof(scratch)) != OT_ERR_DEVICE) ok = false;
    if (ot_matmul_bf16(ctx, ba, bb, bc, 2, 2, 2) != OT_ERR_DEVICE) ok = false;
    const char* em = ot_context_last_error(ctx);
    if (!em || std::strstr(em, "forked") == nullptr) ok = false;
    ot_buffer_free(ba);
    ot_context_destroy(ctx);
    // Even if a new context reuses the freed address, inherited buffers must be rejected.
    ot_context_t* child_ctx = nullptr;
    if (ot_context_create(&cfg, &child_ctx) != OT_OK || !child_ctx) {
      ok = false;
    } else {
      bb->ctx = child_ctx;  // Simulate heap address reuse
      if (ot_buffer_h2d(child_ctx, bb, scratch, sizeof(scratch)) != OT_ERR_DEVICE) ok = false;
      if (ot_buffer_d2h(child_ctx, bb, scratch, sizeof(scratch)) != OT_ERR_DEVICE) ok = false;
      if (ot_matmul_bf16(child_ctx, bb, bc, bb, 2, 2, 2) != OT_ERR_DEVICE) ok = false;
      ot_context_destroy(child_ctx);
    }
    // Free remaining buffers AFTER destroying ctx in the forked child (must not touch ctx).
    ot_buffer_free(bb);
    ot_buffer_free(bc);
    ::_exit(ok ? 0 : 1);
  }
  ctx->mu.unlock();
  int cstatus = 0;
  OT_CHECK(::waitpid(cpid, &cstatus, 0) == cpid);
  OT_CHECK(WIFEXITED(cstatus) && WEXITSTATUS(cstatus) == 0);
  ctx->info.is_mock = 1u;
  ctx->transport.is_mock = true;
  for (ot_buffer_t* b : {ba, bb, bc}) {
    b->is_mock = true;
  }
  ot_buffer_free(ba);
  ot_buffer_free(bb);
  ot_buffer_free(bc);
  ot_context_destroy(ctx);
  printf("[PASS] test_fork_closes_vfio_fds_and_static_pjrt_err\n");
}

static void test_mock_dma_and_matmul() {
  ot_config_t cfg = ot_config_default();
  cfg.mock_mode = 1;
  ot_context_t* ctx = nullptr;
  OT_CHECK(ot_context_create(&cfg, &ctx) == OT_OK);
  OT_CHECK(ot_context_is_mock(ctx) == 1);

  ot_device_info_t info = {};
  OT_CHECK(ot_device_get_info(ctx, &info) == OT_OK);
  OT_CHECK(info.vendor_id == 0x1ae0u && info.device_id == 0x0063u);

  ot_buffer_t* ba = nullptr;
  ot_buffer_t* bb = nullptr;
  ot_buffer_t* bc = nullptr;
  const uint32_t M = 128, K = 128, N = 128;
  OT_CHECK(ot_buffer_alloc(ctx, M * K * 2, &ba) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, K * N * 2, &bb) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, M * N * 2, &bc) == OT_OK);

  std::vector<uint16_t> ones(M * K, 0x3f80u);  // 1.0 bf16
  OT_CHECK(ot_buffer_h2d(ctx, ba, ones.data(), ones.size() * 2) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bb, ones.data(), ones.size() * 2) == OT_OK);
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, bc, M, K, N) == OT_OK);

  std::vector<uint16_t> out(M * N, 0);
  OT_CHECK(ot_buffer_d2h(ctx, bc, out.data(), out.size() * 2) == OT_OK);
  for (uint16_t v : out) {
    OT_CHECK(v == 0x4300u);  // 128.0 bf16
  }

  // Cross-context buffer ownership must be rejected for H2D, D2H, and matmul.
  ot_context_t* ctx2 = nullptr;
  OT_CHECK(ot_context_create(&cfg, &ctx2) == OT_OK);
  ot_buffer_t* foreign = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx2, M * K * 2, &foreign) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, foreign, ones.data(), ones.size() * 2) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_buffer_d2h(ctx, foreign, out.data(), out.size() * 2) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_matmul_bf16(ctx, foreign, bb, bc, M, K, N) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_matmul_bf16(ctx, ba, foreign, bc, M, K, N) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, foreign, M, K, N) == OT_ERR_INVALID_ARG);

  // Identical lhs and rhs into a disjoint output and in-place aliasing remain valid.
  OT_CHECK(ot_matmul_bf16(ctx, ba, ba, bc, M, K, N) == OT_OK);
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, ba, M, K, N) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, ba, ones.data(), ones.size() * 2) == OT_OK);

  // Explicit smaller HBM budget in mock mode clamps capacity and enforces OT_ERR_NOMEM.
  ot_config_t cfg_small = ot_config_default();
  cfg_small.mock_mode = 1;
  cfg_small.hbm_budget_bytes = 256ULL * 1024ULL;
  ot_context_t* ctx_small = nullptr;
  OT_CHECK(ot_context_create(&cfg_small, &ctx_small) == OT_OK);
  ot_device_info_t small_info = {};
  OT_CHECK(ot_device_get_info(ctx_small, &small_info) == OT_OK);
  OT_CHECK(small_info.hbm_bytes_total == 256ULL * 1024ULL);
  ot_buffer_t* too_big = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx_small, 300ULL * 1024ULL, &too_big) == OT_ERR_NO_MEMORY);
  ot_buffer_t* fits = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx_small, 128ULL * 1024ULL, &fits) == OT_OK);
  // Destroying context before freeing an outstanding buffer must remain safe.
  ot_context_destroy(ctx_small);
  ot_buffer_free(fits);

  // Failed ot_context_create must expose diagnostic via ot_context_last_error(nullptr).
  ot_config_t bad_cfg = ot_config_default();
  bad_cfg.struct_size = 1;
  ot_context_t* bad_ctx = nullptr;
  OT_CHECK(ot_context_create(&bad_cfg, &bad_ctx) == OT_ERR_INVALID_ARG);
  OT_CHECK(std::strlen(ot_context_last_error(nullptr)) > 0);

  // Faulted DMA context must refuse subsequent transfers and allocations.
  ctx2->transport.dma_faulted = true;
  ot_buffer_t* blocked = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx2, 4096, &blocked) == OT_ERR_DEVICE);
  OT_CHECK(ot_buffer_h2d(ctx2, foreign, ones.data(), 16) == OT_ERR_DEVICE);
  OT_CHECK(ot_buffer_d2h(ctx2, foreign, out.data(), 16) == OT_ERR_DEVICE);
  ot_buffer_free(foreign);
  ot_context_destroy(ctx2);

  // Verify CQ completion validation rejects mismatched sequence and non-zero status.
  {
    std::string cq_err;
    uint32_t ok_cq[8] = {7u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    OT_CHECK(ot::validate_cq_completion(7u, ok_cq, &cq_err));
    uint32_t bad_seq_cq[8] = {6u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    OT_CHECK(!ot::validate_cq_completion(7u, bad_seq_cq, &cq_err));
    OT_CHECK(!cq_err.empty());
    uint32_t bad_stat_cq[8] = {7u, 0x1u, 0u, 0u, 0u, 0u, 0u, 0u};
    cq_err.clear();
    OT_CHECK(!ot::validate_cq_completion(7u, bad_stat_cq, &cq_err));
    OT_CHECK(!cq_err.empty());
  }

  // Verify owner_pid enforcement rejects live-VFIO context use from another PID.
  {
    ot_config_t cfg_pid = ot_config_default();
    cfg_pid.mock_mode = 1;
    ot_context_t* ctx_pid = nullptr;
    OT_CHECK(ot_context_create(&cfg_pid, &ctx_pid) == OT_OK);
    OT_CHECK(ctx_pid->transport.is_owner_process());
    ctx_pid->info.is_mock = 0;
    ctx_pid->transport.is_mock = false;
    ctx_pid->transport.owner_pid = -1;
    OT_CHECK(!ctx_pid->transport.is_owner_process());
    ot_buffer_t* child_buf = nullptr;
    OT_CHECK(ot_buffer_alloc(ctx_pid, 4096, &child_buf) == OT_ERR_DEVICE);
    ctx_pid->info.is_mock = 1;
    ctx_pid->transport.is_mock = true;
    ot_context_destroy(ctx_pid);
  }

  ot_buffer_free(bc);
  ot_buffer_free(bb);
  ot_buffer_free(ba);
  ot_context_destroy(ctx);
  printf("[PASS] test_mock_dma_and_matmul\n");
}

static void test_pjrt_api_mock() {
  setenv("OPENTPU_MOCK", "1", 1);
  const PJRT_Api* api = GetPjrtApi();
  OT_CHECK(api != nullptr);
  OT_CHECK(api->struct_size == PJRT_Api_STRUCT_SIZE);
  OT_CHECK(api->pjrt_api_version.major_version == 0);
  OT_CHECK(api->pjrt_api_version.minor_version == 116);
  OT_CHECK(api->PJRT_Client_TopologyDescription != nullptr);
  OT_CHECK(api->PJRT_TopologyDescription_PlatformName != nullptr);
  OT_CHECK(api->PJRT_TopologyDescription_PlatformVersion != nullptr);
  OT_CHECK(api->PJRT_TopologyDescription_GetDeviceDescriptions != nullptr);
  OT_CHECK(api->PJRT_TopologyDescription_Serialize != nullptr);
  OT_CHECK(api->PJRT_TopologyDescription_Attributes != nullptr);

  PJRT_Plugin_Initialize_Args pi = {};
  pi.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  OT_CHECK(api->PJRT_Plugin_Initialize(&pi) == nullptr);

  // Distributed client create_options (node_id != 0 or num_nodes != 1) must be rejected.
  {
    PJRT_NamedValue bad_node = {};
    bad_node.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    bad_node.name = "node_id";
    bad_node.name_size = 7;
    bad_node.type = PJRT_NamedValue_kInt64;
    bad_node.int64_value = 1;
    bad_node.value_size = 1;
    PJRT_Client_Create_Args bad_cc = {};
    bad_cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    bad_cc.create_options = &bad_node;
    bad_cc.num_options = 1;
    PJRT_Error* ne = api->PJRT_Client_Create(&bad_cc);
    OT_CHECK(ne != nullptr);
    destroy_pjrt_error(api, ne);

    PJRT_NamedValue bad_count = {};
    bad_count.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    bad_count.name = "num_nodes";
    bad_count.name_size = 9;
    bad_count.type = PJRT_NamedValue_kInt64;
    bad_count.int64_value = 4;
    bad_count.value_size = 1;
    bad_cc.create_options = &bad_count;
    PJRT_Error* ce = api->PJRT_Client_Create(&bad_cc);
    OT_CHECK(ce != nullptr);
    destroy_pjrt_error(api, ce);
  }

  PJRT_Client_Create_Args cc = {};
  cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  OT_CHECK(api->PJRT_Client_Create(&cc) == nullptr);
  OT_CHECK(cc.client != nullptr);

  // OpenXLA calls PJRT_Client_TopologyDescription immediately during client init.
  PJRT_Client_TopologyDescription_Args ctd = {};
  ctd.struct_size = PJRT_Client_TopologyDescription_Args_STRUCT_SIZE;
  ctd.client = cc.client;
  OT_CHECK(api->PJRT_Client_TopologyDescription(&ctd) == nullptr);
  OT_CHECK(ctd.topology != nullptr);

  PJRT_TopologyDescription_PlatformName_Args tpn = {};
  tpn.struct_size = PJRT_TopologyDescription_PlatformName_Args_STRUCT_SIZE;
  tpn.topology = ctd.topology;
  OT_CHECK(api->PJRT_TopologyDescription_PlatformName(&tpn) == nullptr);
  OT_CHECK(tpn.platform_name_size == 3 && std::memcmp(tpn.platform_name, "tpu", 3) == 0);

  PJRT_TopologyDescription_GetDeviceDescriptions_Args tdd = {};
  tdd.struct_size = PJRT_TopologyDescription_GetDeviceDescriptions_Args_STRUCT_SIZE;
  tdd.topology = ctd.topology;
  OT_CHECK(api->PJRT_TopologyDescription_GetDeviceDescriptions(&tdd) == nullptr);
  OT_CHECK(tdd.num_descriptions == 1 && tdd.descriptions[0] != nullptr);

  PJRT_TopologyDescription_Serialize_Args tser = {};
  tser.struct_size = PJRT_TopologyDescription_Serialize_Args_STRUCT_SIZE;
  tser.topology = ctd.topology;
  OT_CHECK(api->PJRT_TopologyDescription_Serialize(&tser) == nullptr);
  OT_CHECK(tser.serialized_bytes != nullptr && tser.serialized_bytes_size > 0);
  tser.serialized_topology_deleter(tser.serialized_topology);

  PJRT_Client_AddressableDevices_Args ad = {};
  ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  ad.client = cc.client;
  OT_CHECK(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
  OT_CHECK(ad.num_addressable_devices == 1);

  // MemoryStats must initialize all *_is_set booleans even when caller passes uninitialized struct.
  PJRT_Device_MemoryStats_Args ms;
  std::memset(&ms, 0xaa, sizeof(ms));
  ms.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
  ms.extension_start = nullptr;
  ms.device = ad.addressable_devices[0];
  OT_CHECK(api->PJRT_Device_MemoryStats(&ms) == nullptr);
  OT_CHECK(ms.bytes_in_use == 0);
  // Mock arena is 64 MiB minus 1 MiB reserved prefix = 63 MiB allocatable.
  OT_CHECK(ms.bytes_limit == static_cast<int64_t>(63ULL * 1024ULL * 1024ULL));
  OT_CHECK(ms.bytes_limit_is_set == true);
  OT_CHECK(ms.peak_bytes_in_use_is_set == false);
  OT_CHECK(ms.num_allocs_is_set == false);
  OT_CHECK(ms.largest_alloc_size_is_set == false);
  OT_CHECK(ms.bytes_reserved_is_set == false);
  OT_CHECK(ms.peak_bytes_reserved_is_set == false);
  OT_CHECK(ms.bytes_reservable_limit_is_set == false);
  OT_CHECK(ms.largest_free_block_bytes_is_set == false);
  OT_CHECK(ms.pool_bytes_is_set == false);
  OT_CHECK(ms.peak_pool_bytes_is_set == false);
  OT_CHECK(ms.peak_allocated_bytes_is_set == false);

  // Reject StableHLO programs with extra/unsupported post-dot operations (e.g. negate).
  const char* bad_mlir =
      "module @jit_neg_mm {\n"
      "  func.func public @main(%a: tensor<4x4xbf16>, %b: tensor<4x4xbf16>) -> tensor<4x4xbf16> {\n"
      "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>\n"
      "    %1 = stablehlo.negate %0 : tensor<4x4xbf16>\n"
      "    return %1 : tensor<4x4xbf16>\n"
      "  }\n"
      "}\n";
  PJRT_Program bad_prog = {};
  bad_prog.struct_size = PJRT_Program_STRUCT_SIZE;
  bad_prog.code = const_cast<char*>(bad_mlir);
  bad_prog.code_size = std::strlen(bad_mlir);
  bad_prog.format = "mlir";
  bad_prog.format_size = 4;
  PJRT_Client_Compile_Args bad_ca = {};
  bad_ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  bad_ca.client = cc.client;
  bad_ca.program = &bad_prog;
  PJRT_Error* bad_err = api->PJRT_Client_Compile(&bad_ca);
  OT_CHECK(bad_err != nullptr);
  destroy_pjrt_error(api, bad_err);

  // Binary MLIR bytecode (ML\xefR) must return UNIMPLEMENTED explicitly.
  const char bc_bytes[8] = {'M', 'L', static_cast<char>(0xef), 'R', 0, 0, 0, 0};
  PJRT_Program bc_prog = {};
  bc_prog.struct_size = PJRT_Program_STRUCT_SIZE;
  bc_prog.code = const_cast<char*>(bc_bytes);
  bc_prog.code_size = sizeof(bc_bytes);
  bc_prog.format = "mlir";
  bc_prog.format_size = 4;
  PJRT_Client_Compile_Args bc_ca = {};
  bc_ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  bc_ca.client = cc.client;
  bc_ca.program = &bc_prog;
  PJRT_Error* bc_err = api->PJRT_Client_Compile(&bc_ca);
  OT_CHECK(bc_err != nullptr);
  PJRT_Error_GetCode_Args bc_code = {};
  bc_code.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  bc_code.error = bc_err;
  OT_CHECK(api->PJRT_Error_GetCode(&bc_code) == nullptr);
  OT_CHECK(bc_code.code == PJRT_Error_Code_UNIMPLEMENTED);
  destroy_pjrt_error(api, bc_err);

  const char* mlir =
      "module @jit_mm {\n"
      "  func.func public @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {\n"
      "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>\n"
      "    return %0 : tensor<128x128xbf16>\n"
      "  }\n"
      "}\n";
  PJRT_Program prog = {};
  prog.struct_size = PJRT_Program_STRUCT_SIZE;
  prog.code = const_cast<char*>(mlir);
  prog.code_size = std::strlen(mlir);
  prog.format = "mlir";
  prog.format_size = 4;

  PJRT_Client_Compile_Args ca = {};
  ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  ca.client = cc.client;
  ca.program = &prog;
  // Unsupported compile_options (multi-replica, multi-partition, non-zero ordinal, tupled, malformed) must fail.
  {
    // field 3 (exec_build_options): field 4 (num_replicas=2)
    const char opt_rep2[] = {0x1a, 0x02, 0x20, 0x02};
    // field 3 (exec_build_options): field 5 (num_partitions=2)
    const char opt_part2[] = {0x1a, 0x02, 0x28, 0x02};
    // field 3 (exec_build_options): field 1 (device_ordinal=1)
    const char opt_ord1[] = {0x1a, 0x02, 0x08, 0x01};
    // field 3 (exec_build_options): field 2 (result_layout = non-empty ShapeProto)
    const char opt_res_layout[] = {0x1a, 0x04, 0x12, 0x02, 0x08, 0x01};
    // field 3 (exec_build_options): field 9 (device_assignment: 1x1 targeting device 1)
    const char opt_da_dev1[] = {0x1a, 0x0a, 0x4a, 0x08, 0x08, 0x01, 0x10, 0x01, 0x1a, 0x02, 0x08, 0x01};
    // field 2 (parameter_is_tupled_arguments=1)
    const char opt_tupled[] = {0x10, 0x01};
    // field 4 (compile_portable_executable=1)
    const char opt_portable[] = {0x20, 0x01};
    // field 10 (matrix_unit_operand_precision=HIGH(2))
    const char opt_mat_prec[] = {0x50, 0x02};
    // malformed truncated varint
    const char opt_bad[] = {static_cast<char>(0x80)};
    auto expect_compile_fail = [&](const char* opts, size_t sz, PJRT_Error_Code want) {
      PJRT_Client_Compile_Args bad_ca = ca;
      bad_ca.compile_options = opts;
      bad_ca.compile_options_size = sz;
      bad_ca.executable = nullptr;
      PJRT_Error* e = api->PJRT_Client_Compile(&bad_ca);
      OT_CHECK(e != nullptr);
      PJRT_Error_GetCode_Args gc = {};
      gc.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
      gc.error = e;
      OT_CHECK(api->PJRT_Error_GetCode(&gc) == nullptr);
      OT_CHECK(gc.code == want);
      destroy_pjrt_error(api, e);
    };
    expect_compile_fail(opt_rep2, sizeof(opt_rep2), PJRT_Error_Code_UNIMPLEMENTED);
    expect_compile_fail(opt_part2, sizeof(opt_part2), PJRT_Error_Code_UNIMPLEMENTED);
    expect_compile_fail(opt_ord1, sizeof(opt_ord1), PJRT_Error_Code_INVALID_ARGUMENT);
    expect_compile_fail(opt_res_layout, sizeof(opt_res_layout), PJRT_Error_Code_UNIMPLEMENTED);
    expect_compile_fail(opt_da_dev1, sizeof(opt_da_dev1), PJRT_Error_Code_UNIMPLEMENTED);
    expect_compile_fail(opt_tupled, sizeof(opt_tupled), PJRT_Error_Code_UNIMPLEMENTED);
    expect_compile_fail(opt_portable, sizeof(opt_portable), PJRT_Error_Code_UNIMPLEMENTED);
    expect_compile_fail(opt_mat_prec, sizeof(opt_mat_prec), PJRT_Error_Code_UNIMPLEMENTED);
    expect_compile_fail(opt_bad, sizeof(opt_bad), PJRT_Error_Code_INVALID_ARGUMENT);
  }

  // Explicit precision = [DEFAULT, DEFAULT] and line-wrapped dot_general must be accepted; HIGH precision rejected.
  {
    const char* mlir_default_prec =
        "module {\n"
        "  func.func public @main(%a: tensor<128 x 128 x bf16> {jax.arg_info = \"inputs//left\"},\n"
        "                         %b: tensor<128x128xbf16> {mhlo.layout_mode = \"default\"})\n"
        "      -> (tensor<128 x 128 x bf16> {jax.result_info = \"\"}) {\n"
        "    %0 = stablehlo.dot_general %a, %b,\n"
        "      contracting_dims = [1] x [0],\n"
        "      precision = [DEFAULT, DEFAULT]\n"
        "      : (tensor<128 x 128 x bf16>, tensor<128x128xbf16>) -> tensor<128 x 128 x bf16>\n"
        "    return %0 : tensor<128 x 128 x bf16>\n"
        "  }\n"
        "}\n";
    PJRT_Program p_def = prog;
    p_def.code = const_cast<char*>(mlir_default_prec);
    p_def.code_size = std::strlen(mlir_default_prec);
    PJRT_Client_Compile_Args ca_def = {};
    ca_def.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca_def.client = cc.client;
    ca_def.program = &p_def;
    OT_CHECK(api->PJRT_Client_Compile(&ca_def) == nullptr);
    OT_CHECK(ca_def.executable != nullptr);
    PJRT_LoadedExecutable_Destroy_Args led = {};
    led.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    led.executable = ca_def.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_Destroy(&led) == nullptr);

    const char* mlir_high_prec =
        "module {\n"
        "  func.func @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {\n"
        "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0], precision = [HIGH, HIGH] : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>\n"
        "    return %0 : tensor<128x128xbf16>\n"
        "  }\n"
        "}\n";
    PJRT_Program p_hi = prog;
    p_hi.code = const_cast<char*>(mlir_high_prec);
    p_hi.code_size = std::strlen(mlir_high_prec);
    PJRT_Client_Compile_Args ca_hi = {};
    ca_hi.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca_hi.client = cc.client;
    ca_hi.program = &p_hi;
    PJRT_Error* hi_err = api->PJRT_Client_Compile(&ca_hi);
    OT_CHECK(hi_err != nullptr);
    destroy_pjrt_error(api, hi_err);

    // Non-@main helper function must be rejected even if body is a valid matmul.
    const char* mlir_helper =
        "module {\n"
        "  func.func @helper(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {\n"
        "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>\n"
        "    return %0 : tensor<128x128xbf16>\n"
        "  }\n"
        "}\n";
    PJRT_Program p_hlp = prog;
    p_hlp.code = const_cast<char*>(mlir_helper);
    p_hlp.code_size = std::strlen(mlir_helper);
    PJRT_Client_Compile_Args ca_hlp = {};
    ca_hlp.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca_hlp.client = cc.client;
    ca_hlp.program = &p_hlp;
    PJRT_Error* hlp_err = api->PJRT_Client_Compile(&ca_hlp);
    OT_CHECK(hlp_err != nullptr);
    destroy_pjrt_error(api, hlp_err);

    // Unterminated enclosing module (missing final '}') must be rejected.
    const char* mlir_unterm =
        "module {\n"
        "  func.func @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {\n"
        "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>\n"
        "    return %0 : tensor<128x128xbf16>\n"
        "  }\n";
    PJRT_Program p_unt = prog;
    p_unt.code = const_cast<char*>(mlir_unterm);
    p_unt.code_size = std::strlen(mlir_unterm);
    PJRT_Client_Compile_Args ca_unt = {};
    ca_unt.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca_unt.client = cc.client;
    ca_unt.program = &p_unt;
    PJRT_Error* unt_err = api->PJRT_Client_Compile(&ca_unt);
    OT_CHECK(unt_err != nullptr);
    destroy_pjrt_error(api, unt_err);

    // Module-level explicit result layout attribute must be rejected.
    const char* mlir_mod_layout =
        "module attributes {mhlo.xla_entry_computation_result_layout = [dense<[0, 1]> : tensor<2xindex>]} {\n"
        "  func.func @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {\n"
        "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>\n"
        "    return %0 : tensor<128x128xbf16>\n"
        "  }\n"
        "}\n";
    PJRT_Program p_ml = prog;
    p_ml.code = const_cast<char*>(mlir_mod_layout);
    p_ml.code_size = std::strlen(mlir_mod_layout);
    PJRT_Client_Compile_Args ca_ml = {};
    ca_ml.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca_ml.client = cc.client;
    ca_ml.program = &p_ml;
    PJRT_Error* ml_err = api->PJRT_Client_Compile(&ca_ml);
    OT_CHECK(ml_err != nullptr);
    destroy_pjrt_error(api, ml_err);

    // Unsupported result memory_kind ("pinned_host") must be rejected; "device" accepted.
    const char* mlir_pinned =
        "module {\n"
        "  func.func @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> (tensor<128x128xbf16> {mhlo.memory_kind = \"pinned_host\"}) {\n"
        "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>\n"
        "    return %0 : tensor<128x128xbf16>\n"
        "  }\n"
        "}\n";
    PJRT_Program p_pin = prog;
    p_pin.code = const_cast<char*>(mlir_pinned);
    p_pin.code_size = std::strlen(mlir_pinned);
    PJRT_Client_Compile_Args ca_pin = {};
    ca_pin.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca_pin.client = cc.client;
    ca_pin.program = &p_pin;
    PJRT_Error* pin_err = api->PJRT_Client_Compile(&ca_pin);
    OT_CHECK(pin_err != nullptr);
    destroy_pjrt_error(api, pin_err);
  }

  // Valid 1x1 compile_options (including field 9 device_assignment for device 0) must succeed.
  const char opt_1x1[] = {0x1a, 0x10, 0x20, 0x01, 0x28, 0x01, 0x30, 0x01,
                          0x4a, 0x08, 0x08, 0x01, 0x10, 0x01, 0x1a, 0x02, 0x08, 0x00};
  ca.compile_options = opt_1x1;
  ca.compile_options_size = sizeof(opt_1x1);
  OT_CHECK(api->PJRT_Client_Compile(&ca) == nullptr);
  OT_CHECK(ca.executable != nullptr);

  // PJRT_LoadedExecutable_GetDeviceAssignment must return a valid 1x1 proto with deleter.
  OT_CHECK(api->PJRT_LoadedExecutable_GetDeviceAssignment != nullptr);
  PJRT_LoadedExecutable_GetDeviceAssignment_Args gda = {};
  gda.struct_size = PJRT_LoadedExecutable_GetDeviceAssignment_Args_STRUCT_SIZE;
  gda.executable = ca.executable;
  OT_CHECK(api->PJRT_LoadedExecutable_GetDeviceAssignment(&gda) == nullptr);
  OT_CHECK(gda.serialized_bytes != nullptr && gda.serialized_bytes_size == 8);
  OT_CHECK(gda.serialized_device_assignment != nullptr &&
           gda.serialized_device_assignment_deleter != nullptr);
  gda.serialized_device_assignment_deleter(gda.serialized_device_assignment);

  // PJRT_LoadedExecutable_GetExecutable must return an independently owned handle.
  PJRT_LoadedExecutable_GetExecutable_Args ge = {};
  ge.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  ge.loaded_executable = ca.executable;
  OT_CHECK(api->PJRT_LoadedExecutable_GetExecutable(&ge) == nullptr);
  OT_CHECK(ge.executable != nullptr);

  // Directly-invoked optional PJRT callbacks must be populated and return UNIMPLEMENTED.
  OT_CHECK(api->PJRT_Executable_OptimizedProgram != nullptr);
  PJRT_Program opt_prog = {};
  opt_prog.struct_size = PJRT_Program_STRUCT_SIZE;
  PJRT_Executable_OptimizedProgram_Args opt_args = {};
  opt_args.struct_size = PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE;
  opt_args.executable = ge.executable;
  opt_args.program = &opt_prog;
  PJRT_Error* opt_err = api->PJRT_Executable_OptimizedProgram(&opt_args);
  OT_CHECK(opt_err != nullptr);
  {
    PJRT_Error_GetCode_Args gc = {};
    gc.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    gc.error = opt_err;
    OT_CHECK(api->PJRT_Error_GetCode(&gc) == nullptr);
    OT_CHECK(gc.code == PJRT_Error_Code_UNIMPLEMENTED);
  }
  destroy_pjrt_error(api, opt_err);

  OT_CHECK(api->PJRT_Buffer_CopyRawToHost != nullptr);
  PJRT_Buffer_CopyRawToHost_Args cr_args = {};
  cr_args.struct_size = PJRT_Buffer_CopyRawToHost_Args_STRUCT_SIZE;
  PJRT_Error* cr_err = api->PJRT_Buffer_CopyRawToHost(&cr_args);
  OT_CHECK(cr_err != nullptr);
  {
    PJRT_Error_GetCode_Args gc = {};
    gc.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    gc.error = cr_err;
    OT_CHECK(api->PJRT_Error_GetCode(&gc) == nullptr);
    OT_CHECK(gc.code == PJRT_Error_Code_UNIMPLEMENTED);
  }
  destroy_pjrt_error(api, cr_err);

  OT_CHECK(api->PJRT_Device_CreateAsyncTrackingEvent != nullptr);
  PJRT_Device_CreateAsyncTrackingEvent_Args te_args = {};
  te_args.struct_size = PJRT_Device_CreateAsyncTrackingEvent_Args_STRUCT_SIZE;
  te_args.device = ad.addressable_devices[0];
  te_args.event = reinterpret_cast<PJRT_AsyncTrackingEvent*>(0x1);
  OT_CHECK(api->PJRT_Device_CreateAsyncTrackingEvent(&te_args) == nullptr);
  OT_CHECK(te_args.event == nullptr);

  // PJRT_Device_GetAttributes must supply a non-null callable deleter even when empty.
  OT_CHECK(api->PJRT_Device_GetAttributes != nullptr);
  PJRT_Device_GetAttributes_Args ga_args = {};
  ga_args.struct_size = PJRT_Device_GetAttributes_Args_STRUCT_SIZE;
  ga_args.device = ad.addressable_devices[0];
  OT_CHECK(api->PJRT_Device_GetAttributes(&ga_args) == nullptr);
  OT_CHECK(ga_args.attributes_deleter != nullptr);
  ga_args.attributes_deleter(ga_args.device_attributes);

  OT_CHECK(api->PJRT_TopologyDescription_Create != nullptr);
  PJRT_TopologyDescription_Create_Args tc_args = {};
  tc_args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
  PJRT_Error* tc_err = api->PJRT_TopologyDescription_Create(&tc_args);
  OT_CHECK(tc_err != nullptr);
  {
    PJRT_Error_GetCode_Args gc = {};
    gc.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    gc.error = tc_err;
    OT_CHECK(api->PJRT_Error_GetCode(&gc) == nullptr);
    OT_CHECK(gc.code == PJRT_Error_Code_UNIMPLEMENTED);
  }
  destroy_pjrt_error(api, tc_err);

  // Oversized strided upload must fail on HBM capacity before reading/packing host memory.
  {
    uint16_t tiny_dummy = 0;
    int64_t huge_dims[2] = {8192, 8192};  // 128 MiB > 63 MiB mock HBM
    int64_t huge_strides[2] = {2, 16384};
    PJRT_Client_BufferFromHostBuffer_Args huge_up = {};
    huge_up.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    huge_up.client = cc.client;
    huge_up.data = &tiny_dummy;
    huge_up.type = PJRT_Buffer_Type_BF16;
    huge_up.dims = huge_dims;
    huge_up.num_dims = 2;
    huge_up.byte_strides = huge_strides;
    huge_up.num_byte_strides = 2;
    huge_up.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    huge_up.device = ad.addressable_devices[0];
    PJRT_Error* huge_err = api->PJRT_Client_BufferFromHostBuffer(&huge_up);
    OT_CHECK(huge_err != nullptr);
    PJRT_Error_GetCode_Args gc = {};
    gc.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    gc.error = huge_err;
    OT_CHECK(api->PJRT_Error_GetCode(&gc) == nullptr);
    OT_CHECK(gc.code == PJRT_Error_Code_RESOURCE_EXHAUSTED);
    destroy_pjrt_error(api, huge_err);
  }

  // Strided & reversed host uploads must reorder elements accurately; unsupported layouts rejected.
  uint16_t raw_2x2[4] = {0x3f80u, 0x4000u, 0x4040u, 0x4080u};  // [1, 2; 3, 4]
  int64_t sdims[2] = {2, 2};
  int64_t transposed_strides[2] = {2, 4};  // transpose -> [1, 3; 2, 4]
  PJRT_Client_BufferFromHostBuffer_Args st_up = {};
  st_up.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  st_up.client = cc.client;
  st_up.data = raw_2x2;
  st_up.type = PJRT_Buffer_Type_BF16;
  st_up.dims = sdims;
  st_up.num_dims = 2;
  st_up.byte_strides = transposed_strides;
  st_up.num_byte_strides = 2;
  st_up.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  st_up.device = ad.addressable_devices[0];
  OT_CHECK(api->PJRT_Client_BufferFromHostBuffer(&st_up) == nullptr);
  if (st_up.done_with_host_buffer) {
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = st_up.done_with_host_buffer;
    OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
  }
  uint16_t out_2x2[4] = {};
  PJRT_Buffer_ToHostBuffer_Args st_dl = {};
  st_dl.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  st_dl.src = st_up.buffer;
  st_dl.dst = out_2x2;
  st_dl.dst_size = sizeof(out_2x2);
  OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&st_dl) == nullptr);
  if (st_dl.event) {
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = st_dl.event;
    OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
  }
  OT_CHECK(out_2x2[0] == 0x3f80u && out_2x2[1] == 0x4040u &&
           out_2x2[2] == 0x4000u && out_2x2[3] == 0x4080u);

  // Unsupported host_layout on ToHostBuffer must be rejected.
  int64_t col_major[2] = {0, 1};
  PJRT_Buffer_MemoryLayout bad_layout = {};
  bad_layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
  bad_layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
  bad_layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
  bad_layout.tiled.minor_to_major = col_major;
  bad_layout.tiled.minor_to_major_size = 2;
  st_dl.host_layout = &bad_layout;
  PJRT_Error* hl_err = api->PJRT_Buffer_ToHostBuffer(&st_dl);
  OT_CHECK(hl_err != nullptr);
  destroy_pjrt_error(api, hl_err);

  // Negative/reversed byte_strides pointing to the end of host allocation must read in-bounds.
  int64_t rev_dim[1] = {4};
  int64_t rev_stride[1] = {-2};
  PJRT_Client_BufferFromHostBuffer_Args rev_up = {};
  rev_up.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  rev_up.client = cc.client;
  rev_up.data = &raw_2x2[3];
  rev_up.type = PJRT_Buffer_Type_BF16;
  rev_up.dims = rev_dim;
  rev_up.num_dims = 1;
  rev_up.byte_strides = rev_stride;
  rev_up.num_byte_strides = 1;
  rev_up.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  rev_up.device = ad.addressable_devices[0];
  OT_CHECK(api->PJRT_Client_BufferFromHostBuffer(&rev_up) == nullptr);
  if (rev_up.done_with_host_buffer) {
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = rev_up.done_with_host_buffer;
    OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
  }
  uint16_t rev_out[4] = {};
  PJRT_Buffer_ToHostBuffer_Args rev_dl = {};
  rev_dl.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  rev_dl.src = rev_up.buffer;
  rev_dl.dst = rev_out;
  rev_dl.dst_size = sizeof(rev_out);
  OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&rev_dl) == nullptr);
  if (rev_dl.event) {
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = rev_dl.event;
    OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
  }
  OT_CHECK(rev_out[0] == 0x4080u && rev_out[1] == 0x4040u &&
           rev_out[2] == 0x4000u && rev_out[3] == 0x3f80u);

  // PJRT_Buffer_Delete must immediately release device HBM before PJRT_Buffer_Destroy.
  for (PJRT_Buffer* b : {st_up.buffer, rev_up.buffer}) {
    PJRT_Buffer_Delete_Args del = {};
    del.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
    del.buffer = b;
    OT_CHECK(api->PJRT_Buffer_Delete(&del) == nullptr);
  }
  OT_CHECK(api->PJRT_Device_MemoryStats(&ms) == nullptr);
  OT_CHECK(ms.bytes_in_use == 0);

  // ReadyEvent on a deleted buffer must report an immediate error through Event_Error/Await/OnReady.
  {
    PJRT_Buffer_ReadyEvent_Args re = {};
    re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    re.buffer = st_up.buffer;
    OT_CHECK(api->PJRT_Buffer_ReadyEvent(&re) == nullptr);
    OT_CHECK(re.event != nullptr);
    PJRT_Event_Error_Args ee = {};
    ee.struct_size = PJRT_Event_Error_Args_STRUCT_SIZE;
    ee.event = re.event;
    PJRT_Error* ev_err = api->PJRT_Event_Error(&ee);
    OT_CHECK(ev_err != nullptr);
    destroy_pjrt_error(api, ev_err);
    PJRT_Event_Await_Args ea = {};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event = re.event;
    PJRT_Error* aw_err = api->PJRT_Event_Await(&ea);
    OT_CHECK(aw_err != nullptr);
    destroy_pjrt_error(api, aw_err);
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = re.event;
    OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
  }

  // Non-null `memory` takes precedence over `device`, and foreign device/memory must be rejected.
  {
    PJRT_Client_Create_Args cc2 = {};
    cc2.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    OT_CHECK(api->PJRT_Client_Create(&cc2) == nullptr);
    PJRT_Client_AddressableDevices_Args ad2 = {};
    ad2.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    ad2.client = cc2.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&ad2) == nullptr);
    PJRT_Device_DefaultMemory_Args dm1 = {};
    dm1.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    dm1.device = ad.addressable_devices[0];
    OT_CHECK(api->PJRT_Device_DefaultMemory(&dm1) == nullptr);

    // Foreign device without memory -> rejected.
    PJRT_Client_BufferFromHostBuffer_Args fdev = {};
    fdev.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    fdev.client = cc.client;
    fdev.data = raw_2x2;
    fdev.type = PJRT_Buffer_Type_BF16;
    fdev.dims = sdims;
    fdev.num_dims = 2;
    fdev.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    fdev.device = ad2.addressable_devices[0];
    PJRT_Error* fd_err = api->PJRT_Client_BufferFromHostBuffer(&fdev);
    OT_CHECK(fd_err != nullptr);
    destroy_pjrt_error(api, fd_err);

    // Client's own memory + foreign device -> memory wins and buffer records client 1's device.
    fdev.memory = dm1.memory;
    OT_CHECK(api->PJRT_Client_BufferFromHostBuffer(&fdev) == nullptr);
    PJRT_Buffer_Device_Args bdev = {};
    bdev.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
    bdev.buffer = fdev.buffer;
    OT_CHECK(api->PJRT_Buffer_Device(&bdev) == nullptr);
    OT_CHECK(bdev.device == ad.addressable_devices[0]);
    if (fdev.done_with_host_buffer) {
      PJRT_Event_Destroy_Args ed = {};
      ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      ed.event = fdev.done_with_host_buffer;
      OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
    }
    PJRT_Buffer_Destroy_Args bd = {};
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = fdev.buffer;
    OT_CHECK(api->PJRT_Buffer_Destroy(&bd) == nullptr);

    PJRT_Client_Destroy_Args cd2 = {};
    cd2.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd2.client = cc2.client;
    // Verify PJRT_Memory vtable get/set_user_data and destruction cleanup callback.
    PJRT_Device_DefaultMemory_Args dm2 = {};
    dm2.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    dm2.device = ad2.addressable_devices[0];
    OT_CHECK(api->PJRT_Device_DefaultMemory(&dm2) == nullptr);
    OT_CHECK(dm2.memory != nullptr && dm2.memory->vtable != nullptr);
    OT_CHECK(dm2.memory->vtable->get_user_data != nullptr &&
             dm2.memory->vtable->set_user_data != nullptr);
    static int g_mem_dtor_calls = 0;
    g_mem_dtor_calls = 0;
    int key_a = 1;
    int val_a = 42;
    auto inc_dtor = [](void* p) {
      if (p) ++(*static_cast<int*>(p));
    };
    dm2.memory->vtable->set_user_data(dm2.memory, &key_a, &g_mem_dtor_calls, inc_dtor);
    OT_CHECK(dm2.memory->vtable->get_user_data(dm2.memory, &key_a) == &g_mem_dtor_calls);
    // Replacing same key invokes previous destructor.
    dm2.memory->vtable->set_user_data(dm2.memory, &key_a, &val_a, nullptr);
    OT_CHECK(g_mem_dtor_calls == 1);
    OT_CHECK(dm2.memory->vtable->get_user_data(dm2.memory, &key_a) == &val_a);
    // Re-attach destructor and verify PJRT_Client_Destroy invokes it.
    dm2.memory->vtable->set_user_data(dm2.memory, &key_a, &g_mem_dtor_calls, inc_dtor);
    OT_CHECK(api->PJRT_Client_Destroy(&cd2) == nullptr);
    OT_CHECK(g_mem_dtor_calls == 2);
  }

  for (PJRT_Buffer* b : {st_up.buffer, rev_up.buffer}) {
    PJRT_Buffer_Destroy_Args bd = {};
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = b;
    OT_CHECK(api->PJRT_Buffer_Destroy(&bd) == nullptr);
  }

  std::vector<uint16_t> ones(128 * 128, 0x3f80u);
  int64_t dims[2] = {128, 128};
  auto upload = [&](const void* data) -> PJRT_Buffer* {
    PJRT_Client_BufferFromHostBuffer_Args a = {};
    a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    a.client = cc.client;
    a.data = data;
    a.type = PJRT_Buffer_Type_BF16;
    a.dims = dims;
    a.num_dims = 2;
    a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    a.device = ad.addressable_devices[0];
    OT_CHECK(api->PJRT_Client_BufferFromHostBuffer(&a) == nullptr);
    if (a.done_with_host_buffer) {
      PJRT_Event_Destroy_Args ed = {};
      ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      ed.event = a.done_with_host_buffer;
      OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
    }
    return a.buffer;
  };

  PJRT_Buffer* ba = upload(ones.data());
  PJRT_Buffer* bb = upload(ones.data());
  PJRT_Buffer* ins[2] = {ba, bb};
  PJRT_Buffer* outs[1] = {nullptr};
  PJRT_Buffer* const* arg_lists[1] = {ins};
  PJRT_Buffer** const out_lists[1] = {outs};

  // Executing client 1's executable with live valid 128x128 inputs but client 2's execute_device must fail.
  {
    PJRT_Client_Create_Args cc3 = {};
    cc3.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    OT_CHECK(api->PJRT_Client_Create(&cc3) == nullptr);
    PJRT_Client_AddressableDevices_Args ad3 = {};
    ad3.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    ad3.client = cc3.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&ad3) == nullptr);

    PJRT_LoadedExecutable_Execute_Args bad_ex = {};
    bad_ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    bad_ex.executable = ca.executable;
    bad_ex.num_devices = 1;
    bad_ex.num_args = 2;
    bad_ex.argument_lists = arg_lists;
    bad_ex.output_lists = out_lists;
    bad_ex.execute_device = ad3.addressable_devices[0];
    PJRT_Error* ex_dev_err = api->PJRT_LoadedExecutable_Execute(&bad_ex);
    OT_CHECK(ex_dev_err != nullptr);
    PJRT_Error_Message_Args em = {};
    em.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    em.error = ex_dev_err;
    api->PJRT_Error_Message(&em);
    OT_CHECK(em.message != nullptr &&
             std::string(em.message, em.message_size).find("execute_device") != std::string::npos);
    destroy_pjrt_error(api, ex_dev_err);
    OT_CHECK(outs[0] == nullptr);

    PJRT_Client_Destroy_Args cd3 = {};
    cd3.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd3.client = cc3.client;
    OT_CHECK(api->PJRT_Client_Destroy(&cd3) == nullptr);
  }
  PJRT_ExecuteOptions opts = {};
  opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  PJRT_Event* dev_ev[1] = {nullptr};

  PJRT_LoadedExecutable_Execute_Args ex = {};
  ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  ex.executable = ca.executable;
  ex.options = &opts;
  ex.argument_lists = arg_lists;
  ex.num_devices = 1;
  ex.num_args = 2;
  ex.output_lists = out_lists;
  ex.device_complete_events = dev_ev;
  OT_CHECK(api->PJRT_LoadedExecutable_Execute(&ex) == nullptr);
  OT_CHECK(outs[0] != nullptr);
  if (dev_ev[0]) {
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = dev_ev[0];
    OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
  }

  std::vector<uint16_t> out_h(128 * 128, 0);
  const int64_t dense_m2m[2] = {1, 0};
  PJRT_Buffer_MemoryLayout dense_host_layout = {};
  dense_host_layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
  dense_host_layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
  dense_host_layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
  dense_host_layout.tiled.minor_to_major = dense_m2m;
  dense_host_layout.tiled.minor_to_major_size = 2;
  PJRT_Buffer_ToHostBuffer_Args th = {};
  th.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  th.src = outs[0];
  th.host_layout = &dense_host_layout;
  th.dst = out_h.data();
  th.dst_size = out_h.size() * 2;
  OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&th) == nullptr);
  if (th.event) {
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = th.event;
    OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
  }
  for (uint16_t v : out_h) OT_CHECK(v == 0x4300u);

  for (PJRT_Buffer* b : {ba, bb, outs[0]}) {
    PJRT_Buffer_Destroy_Args bd = {};
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = b;
    OT_CHECK(api->PJRT_Buffer_Destroy(&bd) == nullptr);
  }
  PJRT_LoadedExecutable_Destroy_Args led = {};
  led.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  led.executable = ca.executable;
  OT_CHECK(api->PJRT_LoadedExecutable_Destroy(&led) == nullptr);

  // Independently owned PJRT_Executable handle must stay valid after LoadedExecutable destruction.
  PJRT_Executable_OutputDimensions_Args od = {};
  od.struct_size = PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE;
  od.executable = ge.executable;
  OT_CHECK(api->PJRT_Executable_OutputDimensions(&od) == nullptr);
  OT_CHECK(od.num_outputs == 1 && od.dim_sizes[0] == 2 && od.dims[0] == 128 && od.dims[1] == 128);
  PJRT_Executable_Destroy_Args edest = {};
  edest.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  edest.executable = ge.executable;
  OT_CHECK(api->PJRT_Executable_Destroy(&edest) == nullptr);

  // Verify default (null device_layout) 128x128 BF16 upload remains dense while explicit
  // {1,0:T(2,128)(2,1)} stores native tiled bytes and reports matching repeated tiling.
  {
    const int64_t d2[2] = {128, 128};
    const int64_t m2m[2] = {1, 0};
    const int64_t valid_tiles[4] = {2, 128, 2, 1};
    const size_t valid_tile_sizes[2] = {2, 2};
    const int64_t bad_tiles[2] = {2, 128};
    const size_t bad_tile_sizes[1] = {2};
    std::vector<uint16_t> tile_src(128u * 128u, 0x3f80u);
    tile_src[128] = 0x4000u;

    PJRT_Client_BufferFromHostBuffer_Args up_dense = {};
    up_dense.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    up_dense.client = cc.client;
    up_dense.data = tile_src.data();
    up_dense.type = PJRT_Buffer_Type_BF16;
    up_dense.dims = d2;
    up_dense.num_dims = 2;
    up_dense.device = ad.addressable_devices[0];
    OT_CHECK(api->PJRT_Client_BufferFromHostBuffer(&up_dense) == nullptr);
    PJRT_Buffer_GetMemoryLayout_Args ml_dense = {};
    ml_dense.struct_size = PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE;
    ml_dense.buffer = up_dense.buffer;
    OT_CHECK(api->PJRT_Buffer_GetMemoryLayout(&ml_dense) == nullptr);
    OT_CHECK(ml_dense.layout.tiled.num_tiles == 0);
    PJRT_Event_Destroy_Args evd = {};
    evd.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    evd.event = up_dense.done_with_host_buffer;
    OT_CHECK(api->PJRT_Event_Destroy(&evd) == nullptr);
    PJRT_Buffer_Destroy_Args bd = {};
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = up_dense.buffer;
    OT_CHECK(api->PJRT_Buffer_Destroy(&bd) == nullptr);

    PJRT_Buffer_MemoryLayout good_layout = {};
    good_layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
    good_layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
    good_layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
    good_layout.tiled.minor_to_major = m2m;
    good_layout.tiled.minor_to_major_size = 2;
    good_layout.tiled.tile_dims = valid_tiles;
    good_layout.tiled.tile_dim_sizes = valid_tile_sizes;
    good_layout.tiled.num_tiles = 2;

    PJRT_Client_BufferFromHostBuffer_Args up_tiled = {};
    up_tiled.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    up_tiled.client = cc.client;
    up_tiled.data = tile_src.data();
    up_tiled.type = PJRT_Buffer_Type_BF16;
    up_tiled.dims = d2;
    up_tiled.num_dims = 2;
    up_tiled.device = ad.addressable_devices[0];
    up_tiled.device_layout = &good_layout;
    OT_CHECK(api->PJRT_Client_BufferFromHostBuffer(&up_tiled) == nullptr);
    PJRT_Buffer_GetMemoryLayout_Args ml_tiled = {};
    ml_tiled.struct_size = PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE;
    ml_tiled.buffer = up_tiled.buffer;
    OT_CHECK(api->PJRT_Buffer_GetMemoryLayout(&ml_tiled) == nullptr);
    OT_CHECK(ml_tiled.layout.tiled.num_tiles == 2 &&
             ml_tiled.layout.tiled.tile_dim_sizes[0] == 2 &&
             ml_tiled.layout.tiled.tile_dim_sizes[1] == 2 &&
             ml_tiled.layout.tiled.tile_dims[0] == 2 &&
             ml_tiled.layout.tiled.tile_dims[1] == 128 &&
             ml_tiled.layout.tiled.tile_dims[2] == 2 &&
             ml_tiled.layout.tiled.tile_dims[3] == 1);

    // Null host_layout preserves source {1,0:T(2,128)(2,1)} layout: (r=1,c=0) -> index 1.
    std::vector<uint16_t> out_native(128u * 128u, 0);
    PJRT_Buffer_ToHostBuffer_Args th_native = {};
    th_native.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    th_native.src = up_tiled.buffer;
    th_native.dst = out_native.data();
    th_native.dst_size = out_native.size() * sizeof(uint16_t);
    OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&th_native) == nullptr);
    evd.event = th_native.event;
    OT_CHECK(api->PJRT_Event_Destroy(&evd) == nullptr);
    OT_CHECK(out_native[1] == 0x4000u && out_native[128] == 0x3f80u);

    // Explicit dense host_layout unpacks {1,0:T(2,128)(2,1)} to row-major: (r=1,c=0) -> index 128.
    std::vector<uint16_t> out_unpacked(128u * 128u, 0);
    PJRT_Buffer_ToHostBuffer_Args th_dense = {};
    th_dense.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    th_dense.src = up_tiled.buffer;
    th_dense.host_layout = &dense_host_layout;
    th_dense.dst = out_unpacked.data();
    th_dense.dst_size = out_unpacked.size() * sizeof(uint16_t);
    OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&th_dense) == nullptr);
    evd.event = th_dense.event;
    OT_CHECK(api->PJRT_Event_Destroy(&evd) == nullptr);
    OT_CHECK(out_unpacked[128] == 0x4000u && out_unpacked[1] == 0x3f80u);

    evd.event = up_tiled.done_with_host_buffer;
    OT_CHECK(api->PJRT_Event_Destroy(&evd) == nullptr);
    bd.buffer = up_tiled.buffer;
    OT_CHECK(api->PJRT_Buffer_Destroy(&bd) == nullptr);

    // Reject single-tile {1,0:T(2,128)} and reject explicit tiled + strided host.
    PJRT_Buffer_MemoryLayout wrong_layout = good_layout;
    wrong_layout.tiled.tile_dims = bad_tiles;
    wrong_layout.tiled.tile_dim_sizes = bad_tile_sizes;
    wrong_layout.tiled.num_tiles = 1;
    PJRT_Client_BufferFromHostBuffer_Args up_wrong = up_tiled;
    up_wrong.device_layout = &wrong_layout;
    PJRT_Error* wrong_err = api->PJRT_Client_BufferFromHostBuffer(&up_wrong);
    OT_CHECK(wrong_err != nullptr);
    destroy_pjrt_error(api, wrong_err);

    const int64_t transposed_strides[2] = {2, 256};
    PJRT_Client_BufferFromHostBuffer_Args up_strided = up_tiled;
    up_strided.byte_strides = transposed_strides;
    up_strided.num_byte_strides = 2;
    PJRT_Error* strided_err = api->PJRT_Client_BufferFromHostBuffer(&up_strided);
    OT_CHECK(strided_err != nullptr);
    destroy_pjrt_error(api, strided_err);
  }

  PJRT_Client_Destroy_Args cd = {};
  cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  cd.client = cc.client;
  OT_CHECK(api->PJRT_Client_Destroy(&cd) == nullptr);
  printf("[PASS] test_pjrt_api_mock\n");
}

static void test_vliw_and_tiling() {
  ot::VliwBundle nop = ot::VliwBundle::nop();
  OT_CHECK(sizeof(nop) == 64);
  OT_CHECK(nop.w[15] == 0x55780000u);

  std::vector<ot::VliwBundle> prog = ot::emit_v5e_matmul_128x128_body();
  OT_CHECK(prog.size() == 58u);
  OT_CHECK(prog[6].w[0] == 0xaf003c00u);
  OT_CHECK(prog[prog.size() - 4u].w[15] == 0x55700db1u);
  OT_CHECK(prog.back().w[15] == 0x55700900u);

  ot::OtV5eLaunchDesc desc = ot::OtV5eLaunchDesc::build(
      0x00100000ULL, 0x00200000ULL, 0x00300000ULL, 0x00400000ULL, 42u);
  OT_CHECK(sizeof(desc.bytes) == 512u);
  auto rd32 = [&](size_t off) {
    uint32_t v = 0;
    std::memcpy(&v, desc.bytes + off, sizeof(v));
    return v;
  };
  OT_CHECK(rd32(0x170u) == static_cast<uint32_t>(0x00200000ULL >> 9));
  OT_CHECK(rd32(0x174u) == static_cast<uint32_t>(0x00100000ULL >> 9));
  OT_CHECK(rd32(0x17cu) == 0xffffffffu);
  OT_CHECK(rd32(0x190u) == static_cast<uint32_t>(0x00300000ULL >> 9));
  OT_CHECK(rd32(0x1a0u) == 0x20u);
  OT_CHECK(rd32(0x1a4u) == static_cast<uint32_t>(0x00400000ULL >> 9));
  OT_CHECK(rd32(0x1e4u) == static_cast<uint32_t>(0x00400000ULL >> 9));
  OT_CHECK(rd32(0x1e8u) == 42u);

  std::vector<uint16_t> src(128u * 128u), tiled(128u * 128u), dst(128u * 128u);
  for (uint32_t r = 0; r < 128u; ++r) {
    for (uint32_t c = 0; c < 128u; ++c) {
      src[r * 128u + c] = static_cast<uint16_t>(r * 131u + c);
    }
  }
  ot::tile_pack_128x128_bf16(src.data(), tiled.data());
  OT_CHECK(tiled[0] == src[0]);
  OT_CHECK(tiled[1] == src[128]);
  OT_CHECK(tiled[2] == src[1]);
  OT_CHECK(tiled[255] == src[128 + 127]);
  ot::tile_unpack_128x128_bf16(tiled.data(), dst.data());
  OT_CHECK(std::memcmp(src.data(), dst.data(), src.size() * sizeof(uint16_t)) == 0);
  printf("[PASS] test_vliw_and_tiling\n");
}

static void test_layout_tiling_and_aliasing() {
  ot_config_t cfg = ot_config_default();
  cfg.mock_mode = 1;
  ot_context_t* ctx = nullptr;
  ot_context_t* other_ctx = nullptr;
  OT_CHECK(ot_context_create(&cfg, &ctx) == OT_OK);
  OT_CHECK(ot_context_create(&cfg, &other_ctx) == OT_OK);

  const uint32_t M = 128u, K = 128u, N = 128u;
  const size_t nbytes = static_cast<size_t>(M) * N * sizeof(uint16_t);
  ot_buffer_t *ba = nullptr, *bb = nullptr, *bc = nullptr, *bd = nullptr, *be = nullptr;
  ot_buffer_t* foreign = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx, nbytes, &ba) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, nbytes, &bb) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, nbytes, &bc) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, nbytes, &bd) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, nbytes, &be) == OT_OK);
  OT_CHECK(ot_buffer_alloc(other_ctx, nbytes, &foreign) == OT_OK);

  std::vector<uint16_t> a(M * K), b(K * N), d(K * N), got(M * N);
  fill_random_bf16(&a, 0x1111ULL, 16, 0.125f);
  fill_random_bf16(&b, 0x2222ULL, 16, 0.125f);
  fill_random_bf16(&d, 0x3333ULL, 2, 0.25f);
  OT_CHECK(ot_buffer_h2d(ctx, foreign, a.data(), nbytes) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_buffer_d2h(ctx, foreign, got.data(), nbytes) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_matmul_bf16(ctx, foreign, bb, bc, M, K, N) == OT_ERR_INVALID_ARG);

  std::vector<uint16_t> ref_ab = ref_matmul(a, b, M, K, N);
  OT_CHECK(ot_buffer_h2d(ctx, ba, a.data(), nbytes) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bb, b.data(), nbytes) == OT_OK);
  OT_CHECK(ba->layout == ot::BufferLayout::kRaw);
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, bc, M, K, N) == OT_OK);
  OT_CHECK(bc->layout == ot::BufferLayout::kTiled2x128Bf16);

  std::vector<uint16_t> expected_tiled(M * N);
  ot::tile_pack_128x128_bf16(ref_ab.data(), expected_tiled.data());
  OT_CHECK(std::memcmp(ctx->transport.mock_hbm.data() + bc->hbm_addr,
                       expected_tiled.data(), nbytes) == 0);

  OT_CHECK(ot_buffer_d2h(ctx, bc, got.data(), nbytes) == OT_OK);
  OT_CHECK(got == ref_ab);
  std::vector<uint16_t> partial(150, 0);
  OT_CHECK(ot_buffer_d2h(ctx, bc, partial.data(), partial.size() * 2) == OT_OK);
  OT_CHECK(std::memcmp(partial.data(), ref_ab.data(), partial.size() * 2) == 0);

  std::vector<uint16_t> ref_abd = ref_matmul(ref_ab, d, M, K, N);
  OT_CHECK(ot::buffer_h2d_tiled_128x128_bf16(ctx, bd, d.data()) == OT_OK);
  OT_CHECK(bd->layout == ot::BufferLayout::kTiled2x128Bf16);
  OT_CHECK(ot_matmul_bf16(ctx, bc, bd, be, M, K, N) == OT_OK);
  OT_CHECK(ot_buffer_d2h(ctx, be, got.data(), nbytes) == OT_OK);
  OT_CHECK(got == ref_abd);

  std::vector<uint16_t> patch(64, 0x4000u);
  OT_CHECK(ot_buffer_h2d(ctx, bc, patch.data(), patch.size() * 2) == OT_OK);
  OT_CHECK(bc->layout == ot::BufferLayout::kRaw);
  OT_CHECK(ot_buffer_d2h(ctx, bc, got.data(), nbytes) == OT_OK);
  OT_CHECK(std::memcmp(got.data(), patch.data(), patch.size() * 2) == 0);
  OT_CHECK(std::memcmp(got.data() + 64, ref_ab.data() + 64, (M * N - 64) * 2) == 0);

  std::vector<uint16_t> ref_aa = ref_matmul(a, a, M, K, N);
  OT_CHECK(ot_buffer_h2d(ctx, ba, a.data(), nbytes) == OT_OK);
  OT_CHECK(ot_matmul_bf16(ctx, ba, ba, ba, M, K, N) == OT_OK);
  OT_CHECK(ot_buffer_d2h(ctx, ba, got.data(), nbytes) == OT_OK);
  OT_CHECK(got == ref_aa);

  // Smaller mock matmul reusing a tiled 128x128 output buffer must preserve the unwritten suffix.
  OT_CHECK(ot_buffer_h2d(ctx, ba, a.data(), nbytes) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bb, b.data(), nbytes) == OT_OK);
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, bc, M, K, N) == OT_OK);
  OT_CHECK(bc->layout == ot::BufferLayout::kTiled2x128Bf16);
  std::vector<uint16_t> sm_a(16u * 32u), sm_b(32u * 24u), sm_got(16u * 24u);
  fill_random_bf16(&sm_a, 0x4444ULL, 8, 0.25f);
  fill_random_bf16(&sm_b, 0x5555ULL, 8, 0.25f);
  std::vector<uint16_t> sm_ref = ref_matmul(sm_a, sm_b, 16u, 32u, 24u);
  OT_CHECK(ot_buffer_h2d(ctx, ba, sm_a.data(), sm_a.size() * 2) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bb, sm_b.data(), sm_b.size() * 2) == OT_OK);
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, bc, 16u, 32u, 24u) == OT_OK);
  OT_CHECK(bc->layout == ot::BufferLayout::kRaw);
  OT_CHECK(ot_buffer_d2h(ctx, bc, sm_got.data(), sm_got.size() * 2) == OT_OK);
  OT_CHECK(sm_got == sm_ref);
  OT_CHECK(ot_buffer_d2h(ctx, bc, got.data(), nbytes) == OT_OK);
  OT_CHECK(std::memcmp(got.data(), sm_ref.data(), sm_ref.size() * 2) == 0);
  OT_CHECK(std::memcmp(got.data() + sm_ref.size(), ref_ab.data() + sm_ref.size(),
                       (M * N - sm_ref.size()) * 2) == 0);

  ot_buffer_free(foreign);
  ot_buffer_free(be);
  ot_buffer_free(bd);
  ot_buffer_free(bc);
  ot_buffer_free(bb);
  ot_buffer_free(ba);
  ot_context_destroy(other_ctx);
  ot_context_destroy(ctx);
  printf("[PASS] test_layout_tiling_and_aliasing\n");
}

static void test_vfio_imem_lifecycle_mock() {
  ot_config_t cfg = ot_config_default();
  cfg.mock_mode = 1;
  ot_context_t* ctx = nullptr;
  OT_CHECK(ot_context_create(&cfg, &ctx) == OT_OK);

  std::string err;
  OT_CHECK(!ctx->transport.imem_booted);
  OT_CHECK(ctx->transport.ensure_imem_booted(1000u, &err));
  OT_CHECK(ctx->transport.imem_booted);
  OT_CHECK(ctx->transport.ensure_imem_booted(1000u, &err));

  OT_CHECK(ctx->transport.ensure_matmul_scratch_slot(&ctx->transport.scratch_lhs_hbm, &err));
  OT_CHECK(ctx->transport.scratch_lhs_hbm != 0 && ctx->transport.scratch_rhs_hbm == 0);
  OT_CHECK(ctx->transport.hbm_alloc.used_bytes() == ot::kTile128x128Bf16Bytes);
  OT_CHECK(ctx->transport.ensure_matmul_scratch(&err));
  OT_CHECK(ctx->transport.scratch_lhs_hbm != 0 && ctx->transport.scratch_rhs_hbm != 0);
  OT_CHECK(ctx->transport.hbm_alloc.used_bytes() == 2u * ot::kTile128x128Bf16Bytes);

  uint32_t cq[8] = {};
  OT_CHECK(ctx->transport.submit_exec_sync(7u, 1000u, cq, &err));
  OT_CHECK(cq[4] == 1u && cq[5] == 7u);

  ctx->transport.teardown_imem_state();
  OT_CHECK(!ctx->transport.imem_booted);
  OT_CHECK(ctx->transport.scratch_lhs_hbm == 0 && ctx->transport.scratch_rhs_hbm == 0);
  OT_CHECK(ctx->transport.hbm_alloc.used_bytes() == 0);

  ot_context_destroy(ctx);
  printf("[PASS] test_vfio_imem_lifecycle_mock\n");
}

static void test_resident_imem_stub_structure() {
  auto bundle_eq = [](const ot::VliwBundle& a, const ot::VliwBundle& b) -> bool {
    return std::memcmp(a.w, b.w, sizeof(a.w)) == 0;
  };
  const ot::VliwBundle nop = ot::VliwBundle::nop();

  {
    const ot::VliwBundle w1ff = ot::VliwBundle::make_sync_wait_imm(0x1ffu, 512u);
    OT_CHECK(w1ff.w[2] == 0x3a8ca30fu && w1ff.w[12] == 0xfc000000u && w1ff.w[13] == 0x00040007u);
    const ot::VliwBundle s1ff = ot::VliwBundle::make_sync_sub_imm(0x1ffu, 512u);
    OT_CHECK(s1ff.w[2] == 0x3a0ca10fu && s1ff.w[12] == 0xfc000000u &&
             s1ff.w[13] == 0xfffc0007u && s1ff.w[14] == 0x0f000003u);

    const ot::VliwBundle w33 = ot::VliwBundle::make_sync_wait_imm(0x33u, 32768u);
    OT_CHECK(w33.w[2] == 0x3a8ca30fu && w33.w[12] == 0xcc000000u && w33.w[13] == 0x01000000u);
    const ot::VliwBundle s33 = ot::VliwBundle::make_sync_sub_imm(0x33u, 32768u);
    OT_CHECK(s33.w[2] == 0x3a0ca10fu && s33.w[12] == 0xcc000000u &&
             s33.w[13] == 0xff000000u && s33.w[14] == 0x0f000003u);
    const ot::VliwBundle c33 = ot::VliwBundle::make_sync_clear(0x33u);
    OT_CHECK(c33.w[2] == 0x3a0ca70fu && c33.w[12] == 0xcc000000u && c33.w[13] == 0x00004000u);

    const ot::VliwBundle l_out = ot::VliwBundle::make_smem_load_ldesc(2u, 0x190u);
    OT_CHECK(l_out.w[13] == 0xfff90000u && l_out.w[14] == 0x0e041008u);
    const ot::VliwBundle l_lhs = ot::VliwBundle::make_smem_load_ldesc(18u, 0x174u);
    OT_CHECK(l_lhs.w[13] == 0xfff74000u && l_lhs.w[14] == 0x0e041048u);
    const ot::VliwBundle l_rhs = ot::VliwBundle::make_smem_load_ldesc(1u, 0x170u);
    OT_CHECK(l_rhs.w[13] == 0xfff70000u && l_rhs.w[14] == 0x0e041004u);

    const ot::VliwBundle f = ot::VliwBundle::make_fence();
    OT_CHECK(f.w[14] == 0x0f000000u && f.w[15] == 0x55700900u);
    const ot::VliwBundle br0 = ot::VliwBundle::make_br_abs(0u);
    OT_CHECK(br0.w[13] == 0u && br0.w[14] == 0x0f000000u && br0.w[15] == 0x55700400u);
  }

  const std::vector<ot::VliwBundle> imem = ot::emit_v5e_resident_matmul_imem();
  OT_CHECK(imem.size() == ot::kResidentImemBundles);
  OT_CHECK(imem.size() * sizeof(ot::VliwBundle) == ot::kResidentImemBytes);
  OT_CHECK((ot::kResidentImemBytes % ot::kImemDmaAlignBytes) == 0u);

  OT_CHECK(imem[0].w[2] == 0x3b00060fu && imem[0].w[15] == 0x55700dc4u);
  OT_CHECK(imem[3].w[13] == 0xfffe8007u && imem[3].w[14] == 0x8e04100cu);
  OT_CHECK(imem[14].w[2] == 0x3a0a200fu && imem[14].w[13] == (ot::kSyncFlagExecDone << 14));
  OT_CHECK(bundle_eq(imem[2], nop) && bundle_eq(imem[4], nop) && bundle_eq(imem[6], nop));
  for (size_t pc = 8; pc <= 12; ++pc) OT_CHECK(bundle_eq(imem[pc], nop));
  OT_CHECK(bundle_eq(imem[15], nop) && bundle_eq(imem[16], nop));

  OT_CHECK(imem[17].w[15] == 0x55701101u);
  OT_CHECK(imem[18].w[2] == 0x3a82a30fu && imem[18].w[13] == 0x00040000u);
  OT_CHECK(imem[19].w[2] == 0x3a02a10fu && imem[19].w[13] == 0xfffc0000u);
  OT_CHECK(bundle_eq(imem[20], nop));

  OT_CHECK(imem[24].w[13] == (ot::kSyncFlagLdescDma << 14) && imem[24].w[14] == 0x0211f000u);
  OT_CHECK(bundle_eq(imem[25], ot::VliwBundle::make_sync_wait_imm(0x1ffu, 512u)));
  OT_CHECK(bundle_eq(imem[26], ot::VliwBundle::make_sync_sub_imm(0x1ffu, 512u)));
  OT_CHECK(bundle_eq(imem[27], ot::VliwBundle::make_fence()));

  OT_CHECK(bundle_eq(imem[28], ot::VliwBundle::make_smem_load_ldesc(2u, 0x190u)));
  OT_CHECK(bundle_eq(imem[29], ot::VliwBundle::make_smem_load_ldesc(18u, 0x174u)));
  OT_CHECK(bundle_eq(imem[30], ot::VliwBundle::make_smem_load_ldesc(1u, 0x170u)));
  for (size_t pc = 31; pc <= 36; ++pc) OT_CHECK(bundle_eq(imem[pc], nop));
  OT_CHECK(imem[37].w[14] == 0x4e01bee8u && imem[37].w[15] == 0x55780000u);
  for (size_t pc = 40; pc <= 44; ++pc) OT_CHECK(bundle_eq(imem[pc], nop));
  OT_CHECK(imem[45].w[13] == (ot::kSyncFlagLhsDma << 14) && imem[45].w[14] == 0x0013d000u);
  OT_CHECK(bundle_eq(imem[46], ot::VliwBundle::make_sync_wait_imm(0x33u, 32768u)));
  OT_CHECK(bundle_eq(imem[47], ot::VliwBundle::make_sync_sub_imm(0x33u, 32768u)));
  OT_CHECK(bundle_eq(imem[48], ot::VliwBundle::make_sync_clear(0x33u)));
  OT_CHECK(imem[49].w[14] == 0x0f000000u && imem[49].w[15] == 0x55700dc0u);
  OT_CHECK(bundle_eq(imem[50], nop));

  const std::vector<ot::VliwBundle> body = ot::emit_v5e_matmul_128x128_body();
  OT_CHECK(body.size() == 58u);
  for (size_t i = 0; i < body.size(); ++i) {
    OT_CHECK(bundle_eq(imem[51u + i], body[i]));
  }

  OT_CHECK(bundle_eq(imem[109], nop));
  OT_CHECK(bundle_eq(imem[110], ot::VliwBundle::make_br_abs(0u)));
  for (size_t pc = 111; pc < ot::kResidentImemBundles; ++pc) {
    OT_CHECK(bundle_eq(imem[pc], nop));
  }
  printf("[PASS] test_resident_imem_stub_structure\n");
}

static void test_multigen_hal_geometry_and_rebar() {
  ot_device_geometry_t g5 = {}, g6 = {}, g7 = {};
  OT_CHECK(ot_chip_geometry_for_device_id(0x0063u, &g5) == OT_OK);
  OT_CHECK(g5.chip_gen == OT_CHIP_V5E);
  OT_CHECK(g5.mxu_rows == 128u && g5.mxu_cols == 128u);
  OT_CHECK(g5.logical_chiplets_per_pci_func == 1u);
  OT_CHECK(g5.hbm_bytes_per_logical_chiplet == (16ULL << 30));

  OT_CHECK(ot_chip_geometry_for_device_id(0x006fu, &g6) == OT_OK);
  OT_CHECK(g6.chip_gen == OT_CHIP_V6E);
  OT_CHECK(g6.mxu_rows == 256u && g6.mxu_cols == 256u);
  OT_CHECK(g6.num_mxus_per_logical_chiplet == 2u);
  OT_CHECK(g6.logical_chiplets_per_pci_func == 1u);
  OT_CHECK(g6.hbm_bytes_per_logical_chiplet == (32ULL << 30));
  OT_CHECK(ot_chip_geometry_for_device_id(0x006eu, &g6) == OT_OK);
  OT_CHECK(g6.chip_gen == OT_CHIP_V6E && g6.num_mxus_per_logical_chiplet == 2u);

  for (uint16_t did7 : {static_cast<uint16_t>(0x0075u), static_cast<uint16_t>(0x0076u)}) {
    OT_CHECK(ot_chip_geometry_for_device_id(did7, &g7) == OT_OK);
    OT_CHECK(g7.chip_gen == OT_CHIP_TPU7X);
    OT_CHECK(g7.pci_device_id == did7);
    OT_CHECK(g7.mxu_rows == 256u && g7.mxu_cols == 256u);
    OT_CHECK(g7.fp8_inner_rows == 4u && g7.fp8_inner_cols == 1u);
    OT_CHECK(g7.supports_native_fp8 == 1u);
    OT_CHECK(g7.logical_chiplets_per_pci_func == 2u);
    OT_CHECK(g7.num_mxus_per_logical_chiplet == 2u);
    OT_CHECK(g7.hbm_bytes_per_logical_chiplet == (96ULL << 30));
    OT_CHECK(g7.vmem_bytes_per_logical_chiplet == (64ULL << 20));
  }

  ot_device_geometry_t gbad = {};
  OT_CHECK(ot_chip_geometry_for_device_id(0x9999u, &gbad) == OT_ERR_UNSUPPORTED);

  // 64-bit Resizable BAR mmap safety clamping.
  OT_CHECK(ot::clamp_bar_mmap_len(96ULL << 30, 16ULL << 20) == (16ULL << 20));
  OT_CHECK(ot::clamp_bar_mmap_len(16ULL << 20, 64ULL << 20) == (16ULL << 20));
  OT_CHECK(ot::clamp_bar_mmap_len(0ULL, 16ULL << 20) == 0u);

  // Synthetic PCIe Extended Config Space (0x100 = dummy 0x0001 -> next 0x180,
  // 0x180 = Resizable BAR 0x0015 with 1 BAR entry: BAR2 size_pow=17 -> 128 GiB).
  uint8_t cfg_space[512];
  std::memset(cfg_space, 0, sizeof(cfg_space));
  auto write_le32 = [&](size_t off, uint32_t v) {
    std::memcpy(cfg_space + off, &v, sizeof(v));
  };
  write_le32(0x100u, (0x180u << 20) | 0x0001u);
  write_le32(0x180u, 0x0015u);
  write_le32(0x184u, 0x00020000u);  // Cap register
  write_le32(0x188u, (1u << 5) | (17u << 8) | 2u);  // 1 BAR, BAR2, 2^(20+17)=128 GiB

  ot::RebarCapabilityInfo rebar;
  OT_CHECK(ot::parse_pcie_rebar_cap(cfg_space, sizeof(cfg_space), &rebar));
  OT_CHECK(rebar.found && rebar.cap_offset == 0x180u && rebar.num_bars == 1u);
  OT_CHECK(rebar.bar_bytes[2] == (128ULL << 30));

  // Mock chip selection for tpu7x via OPENTPU_MOCK_CHIP.
  setenv("OPENTPU_MOCK_CHIP", "tpu7x", 1);
  ot_config_t cfg = ot_config_default();
  cfg.mock_mode = 1;
  ot_context_t* ctx = nullptr;
  OT_CHECK(ot_context_create(&cfg, &ctx) == OT_OK && ctx != nullptr);
  ot_device_info_t info = {};
  OT_CHECK(ot_device_get_info(ctx, &info) == OT_OK);
  OT_CHECK(info.chip_gen == OT_CHIP_TPU7X && info.device_id == 0x0075u);
  ot_device_geometry_t ctx_geom = {};
  OT_CHECK(ot_device_get_geometry(ctx, &ctx_geom) == OT_OK);
  OT_CHECK(ctx_geom.logical_chiplets_per_pci_func == 2u && ctx_geom.mxu_rows == 256u &&
           ctx_geom.vmem_bytes_per_logical_chiplet == (64ULL << 20));
  ot_context_destroy(ctx);
  unsetenv("OPENTPU_MOCK_CHIP");

  // Guest VM VFIO Ironwood PCI Device ID (0x0076) via OPENTPU_MOCK_DEVICE_ID.
  setenv("OPENTPU_MOCK_DEVICE_ID", "0x0076", 1);
  ctx = nullptr;
  OT_CHECK(ot_context_create(&cfg, &ctx) == OT_OK && ctx != nullptr);
  OT_CHECK(ot_device_get_info(ctx, &info) == OT_OK);
  OT_CHECK(info.chip_gen == OT_CHIP_TPU7X && info.device_id == 0x0076u &&
           info.vmem_bytes == (64ULL << 20) && info.num_mxus == 2u);
  ot_context_destroy(ctx);
  unsetenv("OPENTPU_MOCK_DEVICE_ID");

  printf("[PASS] test_multigen_hal_geometry_and_rebar\n");
}

static void test_pjrt_dual_chiplet_tpu7x_enumeration() {
  const PJRT_Api* api = GetPjrtApi();
  OT_CHECK(api != nullptr);

  auto check_topology = [&](const char* topo_name, size_t expected_logical) {
    setenv("OPENTPU_MOCK", "1", 1);
    setenv("OPENTPU_MOCK_CHIP", "tpu7x", 1);
    setenv("OPENTPU_MOCK_TOPOLOGY", topo_name, 1);

    PJRT_Client_Create_Args c_args = {};
    c_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    OT_CHECK(api->PJRT_Client_Create(&c_args) == nullptr && c_args.client != nullptr);

    PJRT_Client_AddressableDevices_Args d_args = {};
    d_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    d_args.client = c_args.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&d_args) == nullptr);
    OT_CHECK(d_args.num_addressable_devices == expected_logical);

    for (size_t i = 0; i < expected_logical; ++i) {
      PJRT_Device_GetDescription_Args gd = {};
      gd.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
      gd.device = d_args.addressable_devices[i];
      OT_CHECK(api->PJRT_Device_GetDescription(&gd) == nullptr);

      PJRT_DeviceDescription_Id_Args id_args = {};
      id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
      id_args.device_description = gd.device_description;
      OT_CHECK(api->PJRT_DeviceDescription_Id(&id_args) == nullptr);
      OT_CHECK(id_args.id == static_cast<int>(i));

      PJRT_DeviceDescription_Kind_Args k_args = {};
      k_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
      k_args.device_description = gd.device_description;
      OT_CHECK(api->PJRT_DeviceDescription_Kind(&k_args) == nullptr);
      OT_CHECK(std::string(k_args.device_kind, k_args.device_kind_size) == "TPU v7x");

      PJRT_DeviceDescription_Attributes_Args attr_args = {};
      attr_args.struct_size = PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
      attr_args.device_description = gd.device_description;
      OT_CHECK(api->PJRT_DeviceDescription_Attributes(&attr_args) == nullptr);
      OT_CHECK(attr_args.num_attributes == 2u);
      bool saw_coords = false, saw_core = false;
      for (size_t ai = 0; ai < attr_args.num_attributes; ++ai) {
        const PJRT_NamedValue& a = attr_args.attributes[ai];
        std::string aname(a.name, a.name_size);
        if (aname == "coords") {
          OT_CHECK(a.type == PJRT_NamedValue_kInt64List && a.value_size == 3u);
          saw_coords = true;
        } else if (aname == "core_on_chip") {
          OT_CHECK(a.type == PJRT_NamedValue_kInt64 &&
                   a.int64_value == static_cast<int64_t>(i % 2));
          saw_core = true;
        }
      }
      OT_CHECK(saw_coords && saw_core);
    }

    std::vector<int> def_assign(expected_logical, -1);
    PJRT_Client_DefaultDeviceAssignment_Args da_args = {};
    da_args.struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE;
    da_args.client = c_args.client;
    da_args.num_replicas = static_cast<int>(expected_logical);
    da_args.num_partitions = 1;
    da_args.default_assignment = def_assign.data();
    da_args.default_assignment_size = def_assign.size();
    OT_CHECK(api->PJRT_Client_DefaultDeviceAssignment(&da_args) == nullptr);
    for (size_t i = 0; i < expected_logical; ++i) {
      OT_CHECK(def_assign[i] == static_cast<int>(i));
    }

    PJRT_Client_Destroy_Args cd_args = {};
    cd_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd_args.client = c_args.client;
    OT_CHECK(api->PJRT_Client_Destroy(&cd_args) == nullptr);
  };

  check_topology("tpu7x-1t", 2u);
  check_topology("tpu7x-4t", 8u);

  unsetenv("OPENTPU_MOCK_TOPOLOGY");
  unsetenv("OPENTPU_MOCK_CHIP");
  unsetenv("OPENTPU_MOCK");
  printf("[PASS] test_pjrt_dual_chiplet_tpu7x_enumeration\n");
}

static void test_pjrt_multidevice_isolation_and_topology() {
  const PJRT_Api* api = GetPjrtApi();
  OT_CHECK(api != nullptr);

  const char* mlir_4x4 =
      "module {\n"
      "  func.func public @main(%a: tensor<4x4xbf16>, %b: tensor<4x4xbf16>) -> tensor<4x4xbf16> {\n"
      "    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : "
      "(tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>\n"
      "    return %0 : tensor<4x4xbf16>\n"
      "  }\n"
      "}\n";
  PJRT_Program prog = {};
  prog.struct_size = PJRT_Program_STRUCT_SIZE;
  prog.code = const_cast<char*>(mlir_4x4);
  prog.code_size = std::strlen(mlir_4x4);
  prog.format = "mlir";
  prog.format_size = 4;

  auto get_mem_used = [&](PJRT_Device* dev) -> int64_t {
    PJRT_Device_MemoryStats_Args ms = {};
    ms.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    ms.device = dev;
    OT_CHECK(api->PJRT_Device_MemoryStats(&ms) == nullptr);
    return ms.bytes_in_use;
  };

  auto upload_bf16 = [&](PJRT_Client* client, PJRT_Device* dev,
                         const std::vector<uint16_t>& data,
                         int64_t rows, int64_t cols) -> PJRT_Buffer* {
    int64_t dims[2] = {rows, cols};
    PJRT_Client_BufferFromHostBuffer_Args up = {};
    up.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    up.client = client;
    up.data = data.data();
    up.type = PJRT_Buffer_Type_BF16;
    up.dims = dims;
    up.num_dims = 2;
    up.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    up.device = dev;
    OT_CHECK(api->PJRT_Client_BufferFromHostBuffer(&up) == nullptr);
    if (up.done_with_host_buffer) {
      PJRT_Event_Destroy_Args ed = {};
      ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      ed.event = up.done_with_host_buffer;
      OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
    }
    return up.buffer;
  };

  auto destroy_buf = [&](PJRT_Buffer* b) {
    if (!b) return;
    PJRT_Buffer_Destroy_Args bd = {};
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = b;
    OT_CHECK(api->PJRT_Buffer_Destroy(&bd) == nullptr);
  };

  auto get_err_code = [&](PJRT_Error* err) -> PJRT_Error_Code {
    PJRT_Error_GetCode_Args gc = {};
    gc.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    gc.error = err;
    OT_CHECK(api->PJRT_Error_GetCode(&gc) == nullptr);
    return gc.code;
  };

  // 1. Topology serialization & fingerprint must distinguish 2-chiplet vs 8-chiplet layouts.
  {
    auto capture_topo = [&](const char* topo_name, std::string* out_ser, uint64_t* out_fp) {
      setenv("OPENTPU_MOCK", "1", 1);
      setenv("OPENTPU_MOCK_CHIP", "tpu7x", 1);
      setenv("OPENTPU_MOCK_TOPOLOGY", topo_name, 1);

      PJRT_Client_Create_Args cc = {};
      cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
      OT_CHECK(api->PJRT_Client_Create(&cc) == nullptr && cc.client != nullptr);

      PJRT_Client_TopologyDescription_Args td = {};
      td.struct_size = PJRT_Client_TopologyDescription_Args_STRUCT_SIZE;
      td.client = cc.client;
      OT_CHECK(api->PJRT_Client_TopologyDescription(&td) == nullptr && td.topology != nullptr);

      PJRT_TopologyDescription_Serialize_Args ser = {};
      ser.struct_size = PJRT_TopologyDescription_Serialize_Args_STRUCT_SIZE;
      ser.topology = td.topology;
      OT_CHECK(api->PJRT_TopologyDescription_Serialize(&ser) == nullptr);
      OT_CHECK(ser.serialized_bytes != nullptr && ser.serialized_bytes_size > 0);
      out_ser->assign(ser.serialized_bytes, ser.serialized_bytes_size);
      ser.serialized_topology_deleter(ser.serialized_topology);

      PJRT_TopologyDescription_Fingerprint_Args fp = {};
      fp.struct_size = PJRT_TopologyDescription_Fingerprint_Args_STRUCT_SIZE;
      fp.topology = td.topology;
      OT_CHECK(api->PJRT_TopologyDescription_Fingerprint(&fp) == nullptr);
      *out_fp = fp.fingerprint;

      PJRT_Client_Destroy_Args cd = {};
      cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
      cd.client = cc.client;
      OT_CHECK(api->PJRT_Client_Destroy(&cd) == nullptr);
    };

    std::string ser_2a, ser_2b, ser_8;
    uint64_t fp_2a = 0, fp_2b = 0, fp_8 = 0;
    capture_topo("tpu7x-1t", &ser_2a, &fp_2a);
    capture_topo("tpu7x-1t", &ser_2b, &fp_2b);
    capture_topo("tpu7x-4t", &ser_8, &fp_8);

    OT_CHECK(ser_2a == ser_2b && fp_2a == fp_2b);
    OT_CHECK(ser_2a != ser_8);
    OT_CHECK(fp_2a != fp_8);
    OT_CHECK(ser_2a.find("n=2") != std::string::npos);
    OT_CHECK(ser_8.find("n=8") != std::string::npos);
  }

  // 2. visible_devices=[1] on 2-chiplet mock must restrict devices, memories, lookup, and default assignment.
  {
    setenv("OPENTPU_MOCK", "1", 1);
    setenv("OPENTPU_MOCK_CHIP", "tpu7x", 1);
    setenv("OPENTPU_MOCK_TOPOLOGY", "tpu7x-1t", 1);

    // Reject empty, duplicate, negative, and out-of-range visible_devices with INVALID_ARGUMENT.
    auto check_bad_visible = [&](const int64_t* vals, size_t cnt) {
      PJRT_NamedValue opt = {};
      opt.struct_size = PJRT_NamedValue_STRUCT_SIZE;
      opt.name = "visible_devices";
      opt.name_size = 15;
      opt.type = PJRT_NamedValue_kInt64List;
      opt.int64_array_value = vals;
      opt.value_size = cnt;
      PJRT_Client_Create_Args cc = {};
      cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
      cc.create_options = &opt;
      cc.num_options = 1;
      PJRT_Error* e = api->PJRT_Client_Create(&cc);
      OT_CHECK(e != nullptr);
      OT_CHECK(get_err_code(e) == PJRT_Error_Code_INVALID_ARGUMENT);
      destroy_pjrt_error(api, e);
    };
    const int64_t dup_vals[2] = {1, 1};
    const int64_t oor_vals[1] = {2};
    const int64_t neg_vals[1] = {-1};
    check_bad_visible(nullptr, 0);
    check_bad_visible(dup_vals, 2);
    check_bad_visible(oor_vals, 1);
    check_bad_visible(neg_vals, 1);

    // Valid visible_devices=[1] -> exposes ONLY device 1 and memory 1, assigns device 1 by default.
    const int64_t vis1[1] = {1};
    PJRT_NamedValue opt = {};
    opt.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    opt.name = "visible_devices";
    opt.name_size = 15;
    opt.type = PJRT_NamedValue_kInt64List;
    opt.int64_array_value = vis1;
    opt.value_size = 1;
    PJRT_Client_Create_Args cc = {};
    cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    cc.create_options = &opt;
    cc.num_options = 1;
    OT_CHECK(api->PJRT_Client_Create(&cc) == nullptr && cc.client != nullptr);

    PJRT_Client_Devices_Args da = {};
    da.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    da.client = cc.client;
    OT_CHECK(api->PJRT_Client_Devices(&da) == nullptr);
    OT_CHECK(da.num_devices == 1u);

    PJRT_Client_AddressableDevices_Args ad = {};
    ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    ad.client = cc.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
    OT_CHECK(ad.num_addressable_devices == 1u && ad.addressable_devices[0] == da.devices[0]);

    PJRT_Device_GetDescription_Args gd = {};
    gd.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
    gd.device = ad.addressable_devices[0];
    OT_CHECK(api->PJRT_Device_GetDescription(&gd) == nullptr);
    PJRT_DeviceDescription_Id_Args did = {};
    did.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
    did.device_description = gd.device_description;
    OT_CHECK(api->PJRT_DeviceDescription_Id(&did) == nullptr);
    OT_CHECK(did.id == 1);

    PJRT_Client_AddressableMemories_Args am = {};
    am.struct_size = PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
    am.client = cc.client;
    OT_CHECK(api->PJRT_Client_AddressableMemories(&am) == nullptr);
    OT_CHECK(am.num_addressable_memories == 1u);
    PJRT_Memory_Id_Args mid = {};
    mid.struct_size = PJRT_Memory_Id_Args_STRUCT_SIZE;
    mid.memory = am.addressable_memories[0];
    OT_CHECK(api->PJRT_Memory_Id(&mid) == nullptr);
    OT_CHECK(mid.id == 1);

    PJRT_Client_LookupDevice_Args ld0 = {};
    ld0.struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE;
    ld0.client = cc.client;
    ld0.id = 0;
    PJRT_Error* ld0_err = api->PJRT_Client_LookupDevice(&ld0);
    OT_CHECK(ld0_err != nullptr && get_err_code(ld0_err) == PJRT_Error_Code_NOT_FOUND);
    destroy_pjrt_error(api, ld0_err);

    PJRT_Client_LookupDevice_Args ld1 = {};
    ld1.struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE;
    ld1.client = cc.client;
    ld1.id = 1;
    OT_CHECK(api->PJRT_Client_LookupDevice(&ld1) == nullptr &&
             ld1.device == ad.addressable_devices[0]);

    int def_assign[1] = {-1};
    PJRT_Client_DefaultDeviceAssignment_Args dda = {};
    dda.struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE;
    dda.client = cc.client;
    dda.num_replicas = 1;
    dda.num_partitions = 1;
    dda.default_assignment = def_assign;
    dda.default_assignment_size = 1;
    OT_CHECK(api->PJRT_Client_DefaultDeviceAssignment(&dda) == nullptr);
    OT_CHECK(def_assign[0] == 1);

    // Default compile on visible_devices=[1] assigns device 1 and executes cleanly.
    PJRT_Client_Compile_Args ca = {};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client = cc.client;
    ca.program = &prog;
    OT_CHECK(api->PJRT_Client_Compile(&ca) == nullptr && ca.executable != nullptr);
    PJRT_LoadedExecutable_AddressableDevices_Args lead = {};
    lead.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
    lead.executable = ca.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_AddressableDevices(&lead) == nullptr);
    OT_CHECK(lead.num_addressable_devices == 1u &&
             lead.addressable_devices[0] == ad.addressable_devices[0]);

    std::vector<uint16_t> ones(16, 0x3f80u);
    PJRT_Buffer* b0 = upload_bf16(cc.client, ad.addressable_devices[0], ones, 4, 4);
    PJRT_Buffer* b1 = upload_bf16(cc.client, ad.addressable_devices[0], ones, 4, 4);
    PJRT_Buffer* ins[2] = {b0, b1};
    PJRT_Buffer* outs[1] = {nullptr};
    PJRT_Buffer* const* arg_lists[1] = {ins};
    PJRT_Buffer** const out_lists[1] = {outs};
    PJRT_ExecuteOptions eopts = {};
    eopts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    PJRT_LoadedExecutable_Execute_Args ex = {};
    ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex.executable = ca.executable;
    ex.options = &eopts;
    ex.argument_lists = arg_lists;
    ex.num_devices = 1;
    ex.num_args = 2;
    ex.output_lists = out_lists;
    OT_CHECK(api->PJRT_LoadedExecutable_Execute(&ex) == nullptr && outs[0] != nullptr);
    PJRT_Buffer_Device_Args obd = {};
    obd.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
    obd.buffer = outs[0];
    OT_CHECK(api->PJRT_Buffer_Device(&obd) == nullptr && obd.device == ad.addressable_devices[0]);

    destroy_buf(b0);
    destroy_buf(b1);
    destroy_buf(outs[0]);
    PJRT_LoadedExecutable_Destroy_Args led = {};
    led.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    led.executable = ca.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_Destroy(&led) == nullptr);
    PJRT_Client_Destroy_Args cd = {};
    cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd.client = cc.client;
    OT_CHECK(api->PJRT_Client_Destroy(&cd) == nullptr);
  }

  // 3. 2-chiplet mock: per-device isolated HBM, 1-replica executable returns 1 device,
  //    and 2-replica Execute failure on replica 1 leaks zero buffers/events from replica 0.
  {
    setenv("OPENTPU_MOCK", "1", 1);
    setenv("OPENTPU_MOCK_CHIP", "tpu7x", 1);
    setenv("OPENTPU_MOCK_TOPOLOGY", "tpu7x-1t", 1);

    PJRT_Client_Create_Args cc = {};
    cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    OT_CHECK(api->PJRT_Client_Create(&cc) == nullptr && cc.client != nullptr);

    PJRT_Client_AddressableDevices_Args ad = {};
    ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    ad.client = cc.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
    OT_CHECK(ad.num_addressable_devices == 2u);
    PJRT_Device* dev0 = ad.addressable_devices[0];
    PJRT_Device* dev1 = ad.addressable_devices[1];

    // Isolated mock contexts: uploading to dev0 changes dev0 HBM only, leaving dev1 at 0.
    std::vector<uint16_t> ones(16, 0x3f80u);
    std::vector<uint16_t> twos(16, 0x4000u);
    PJRT_Buffer* d0_a = upload_bf16(cc.client, dev0, ones, 4, 4);
    PJRT_Buffer* d0_b = upload_bf16(cc.client, dev0, ones, 4, 4);
    OT_CHECK(get_mem_used(dev0) > 0);
    OT_CHECK(get_mem_used(dev1) == 0);

    // 1-replica executable on 2-chiplet client must advertise ONLY 1 assigned device and execute with num_devices=1.
    PJRT_Client_Compile_Args ca1 = {};
    ca1.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca1.client = cc.client;
    ca1.program = &prog;
    OT_CHECK(api->PJRT_Client_Compile(&ca1) == nullptr && ca1.executable != nullptr);

    PJRT_LoadedExecutable_AddressableDevices_Args lead1 = {};
    lead1.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
    lead1.executable = ca1.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_AddressableDevices(&lead1) == nullptr);
    OT_CHECK(lead1.num_addressable_devices == 1u && lead1.addressable_devices[0] == dev0);

    PJRT_Buffer* ins1[2] = {d0_a, d0_b};
    PJRT_Buffer* outs1[1] = {nullptr};
    PJRT_Buffer* const* arg_lists1[1] = {ins1};
    PJRT_Buffer** const out_lists1[1] = {outs1};
    PJRT_ExecuteOptions eopts = {};
    eopts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    PJRT_LoadedExecutable_Execute_Args ex1 = {};
    ex1.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex1.executable = ca1.executable;
    ex1.options = &eopts;
    ex1.argument_lists = arg_lists1;
    ex1.num_devices = 1;
    ex1.num_args = 2;
    ex1.output_lists = out_lists1;
    OT_CHECK(api->PJRT_LoadedExecutable_Execute(&ex1) == nullptr && outs1[0] != nullptr);
    destroy_buf(outs1[0]);

    PJRT_LoadedExecutable_Destroy_Args led1 = {};
    led1.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    led1.executable = ca1.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_Destroy(&led1) == nullptr);

    // Reject mismatched num_replicas vs explicit device_assignment with INVALID_ARGUMENT.
    {
      // field 3 (exec_build_options): field 4 (num_replicas=2) + field 9 (1x1 device_assignment=[0])
      const char opt_rep2_da1[] = {
          0x1a, 0x0c,
          0x20, 0x02,
          0x4a, 0x08, 0x08, 0x01, 0x10, 0x01, 0x1a, 0x02, 0x08, 0x00};
      PJRT_Client_Compile_Args bad_m = {};
      bad_m.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
      bad_m.client = cc.client;
      bad_m.program = &prog;
      bad_m.compile_options = opt_rep2_da1;
      bad_m.compile_options_size = sizeof(opt_rep2_da1);
      PJRT_Error* me = api->PJRT_Client_Compile(&bad_m);
      OT_CHECK(me != nullptr && get_err_code(me) == PJRT_Error_Code_INVALID_ARGUMENT);
      destroy_pjrt_error(api, me);

      // field 3 (exec_build_options): field 4 (num_replicas=1) + field 9 (2x1 device_assignment=[0, 1])
      const char opt_rep1_da2[] = {
          0x1a, 0x0e,
          0x20, 0x01,
          0x4a, 0x0a, 0x08, 0x02, 0x10, 0x01, 0x1a, 0x04, 0x08, 0x00, 0x08, 0x01};
      bad_m.compile_options = opt_rep1_da2;
      bad_m.compile_options_size = sizeof(opt_rep1_da2);
      bad_m.executable = nullptr;
      me = api->PJRT_Client_Compile(&bad_m);
      OT_CHECK(me != nullptr && get_err_code(me) == PJRT_Error_Code_INVALID_ARGUMENT);
      destroy_pjrt_error(api, me);
    }

    // 2-replica executable: replica 0 valid, replica 1 has shape mismatch -> must fail with zero leaks.
    const char opt_rep2[] = {0x1a, 0x02, 0x20, 0x02};
    PJRT_Client_Compile_Args ca2 = {};
    ca2.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca2.client = cc.client;
    ca2.program = &prog;
    ca2.compile_options = opt_rep2;
    ca2.compile_options_size = sizeof(opt_rep2);
    OT_CHECK(api->PJRT_Client_Compile(&ca2) == nullptr && ca2.executable != nullptr);

    PJRT_LoadedExecutable_AddressableDevices_Args lead2 = {};
    lead2.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
    lead2.executable = ca2.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_AddressableDevices(&lead2) == nullptr);
    OT_CHECK(lead2.num_addressable_devices == 2u &&
             lead2.addressable_devices[0] == dev0 &&
             lead2.addressable_devices[1] == dev1);

    PJRT_Buffer* d1_a = upload_bf16(cc.client, dev1, twos, 4, 4);
    std::vector<uint16_t> bad_rhs_data(8, 0x4000u);
    PJRT_Buffer* d1_bad = upload_bf16(cc.client, dev1, bad_rhs_data, 2, 4);

    const int64_t dev0_used_before = get_mem_used(dev0);
    const int64_t dev1_used_before = get_mem_used(dev1);

    PJRT_Buffer* r0_in[2] = {d0_a, d0_b};
    PJRT_Buffer* r1_bad_in[2] = {d1_a, d1_bad};
    PJRT_Buffer* const* bad_arg_lists[2] = {r0_in, r1_bad_in};
    PJRT_Buffer* r0_out[1] = {nullptr};
    PJRT_Buffer* r1_out[1] = {nullptr};
    PJRT_Buffer** const rep_out_lists[2] = {r0_out, r1_out};
    PJRT_Event* rep_evs[2] = {nullptr, nullptr};

    PJRT_LoadedExecutable_Execute_Args ex_fail = {};
    ex_fail.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex_fail.executable = ca2.executable;
    ex_fail.options = &eopts;
    ex_fail.argument_lists = bad_arg_lists;
    ex_fail.num_devices = 2;
    ex_fail.num_args = 2;
    ex_fail.output_lists = rep_out_lists;
    ex_fail.device_complete_events = rep_evs;
    PJRT_Error* fail_err = api->PJRT_LoadedExecutable_Execute(&ex_fail);
    OT_CHECK(fail_err != nullptr);
    OT_CHECK(get_err_code(fail_err) == PJRT_Error_Code_INVALID_ARGUMENT);
    destroy_pjrt_error(api, fail_err);

    // Caller outputs/events untouched and dev0 HBM released completely (zero leaked buffers/events).
    OT_CHECK(r0_out[0] == nullptr && r1_out[0] == nullptr);
    OT_CHECK(rep_evs[0] == nullptr && rep_evs[1] == nullptr);
    OT_CHECK(get_mem_used(dev0) == dev0_used_before);
    OT_CHECK(get_mem_used(dev1) == dev1_used_before);

    // Valid 2-replica Execute succeeds on both devices independently.
    PJRT_Buffer* d1_b = upload_bf16(cc.client, dev1, twos, 4, 4);
    PJRT_Buffer* r1_ok_in[2] = {d1_a, d1_b};
    PJRT_Buffer* const* ok_arg_lists[2] = {r0_in, r1_ok_in};
    PJRT_LoadedExecutable_Execute_Args ex_ok = ex_fail;
    ex_ok.argument_lists = ok_arg_lists;
    OT_CHECK(api->PJRT_LoadedExecutable_Execute(&ex_ok) == nullptr);
    OT_CHECK(r0_out[0] != nullptr && r1_out[0] != nullptr);
    OT_CHECK(rep_evs[0] != nullptr && rep_evs[1] != nullptr);

    std::vector<uint16_t> out0_h(16, 0), out1_h(16, 0);
    auto read_back = [&](PJRT_Buffer* b, std::vector<uint16_t>& dst) {
      PJRT_Buffer_ToHostBuffer_Args th = {};
      th.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
      th.src = b;
      th.dst = dst.data();
      th.dst_size = dst.size() * sizeof(uint16_t);
      OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&th) == nullptr);
      if (th.event) {
        PJRT_Event_Destroy_Args ed = {};
        ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        ed.event = th.event;
        OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
      }
    };
    read_back(r0_out[0], out0_h);
    read_back(r1_out[0], out1_h);
    for (uint16_t v : out0_h) OT_CHECK(v == 0x4080u);  // 4 * (1 * 1) = 4.0
    for (uint16_t v : out1_h) OT_CHECK(v == 0x4180u);  // 4 * (2 * 2) = 16.0

    for (PJRT_Event* ev : rep_evs) {
      PJRT_Event_Destroy_Args ed = {};
      ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      ed.event = ev;
      OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
    }
    for (PJRT_Buffer* b : {d0_a, d0_b, d1_a, d1_b, d1_bad, r0_out[0], r1_out[0]}) {
      destroy_buf(b);
    }
    PJRT_LoadedExecutable_Destroy_Args led2 = {};
    led2.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    led2.executable = ca2.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_Destroy(&led2) == nullptr);
    PJRT_Client_Destroy_Args cd = {};
    cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd.client = cc.client;
    OT_CHECK(api->PJRT_Client_Destroy(&cd) == nullptr);
  }

  // 4. Live mode (OPENTPU_MOCK unset): invalid visible_devices must be rejected with INVALID_ARGUMENT
  //    BEFORE attempting to open or reset any VFIO device (which would fail with INTERNAL on host).
  {
    unsetenv("OPENTPU_MOCK");
    unsetenv("OPENTPU_MOCK_CHIP");
    unsetenv("OPENTPU_MOCK_TOPOLOGY");
    auto check_live_visible_rejected = [&](const int64_t* vals, size_t cnt) {
      PJRT_NamedValue opt = {};
      opt.struct_size = PJRT_NamedValue_STRUCT_SIZE;
      opt.name = "visible_devices";
      opt.name_size = 15;
      opt.type = PJRT_NamedValue_kInt64List;
      opt.int64_array_value = vals;
      opt.value_size = cnt;
      PJRT_Client_Create_Args cc = {};
      cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
      cc.create_options = &opt;
      cc.num_options = 1;
      PJRT_Error* e = api->PJRT_Client_Create(&cc);
      OT_CHECK(e != nullptr);
      OT_CHECK(get_err_code(e) == PJRT_Error_Code_INVALID_ARGUMENT);
      destroy_pjrt_error(api, e);
    };
    const int64_t dev1[1] = {1};
    const int64_t dup0[2] = {0, 0};
    const int64_t neg1[1] = {-1};
    check_live_visible_rejected(dev1, 1);
    check_live_visible_rejected(dup0, 2);
    check_live_visible_rejected(neg1, 1);
  }

  // 5. Setting ONLY OPENTPU_MOCK_TOPOLOGY=tpu7x-1t (with OPENTPU_MOCK_CHIP unset) must configure
  //    every created device HAL context as TPU v7x (0x0075, 256x256 MXU, native FP8).
  {
    unsetenv("OPENTPU_MOCK_CHIP");
    unsetenv("OPENTPU_MOCK_DEVICE_ID");
    setenv("OPENTPU_MOCK", "1", 1);
    setenv("OPENTPU_MOCK_TOPOLOGY", "tpu7x-1t", 1);

    PJRT_Client_Create_Args cc = {};
    cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    OT_CHECK(api->PJRT_Client_Create(&cc) == nullptr && cc.client != nullptr);

    PJRT_Client_AddressableDevices_Args ad = {};
    ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    ad.client = cc.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
    OT_CHECK(ad.num_addressable_devices == 2u);

    for (size_t i = 0; i < ad.num_addressable_devices; ++i) {
      const ot_context_t* dev_ctx = ot::pjrt_device_context_raw(ad.addressable_devices[i]);
      OT_CHECK(dev_ctx != nullptr);
      ot_device_info_t dinfo = {};
      ot_device_geometry_t dgeom = {};
      OT_CHECK(ot_device_get_info(dev_ctx, &dinfo) == OT_OK);
      OT_CHECK(ot_device_get_geometry(dev_ctx, &dgeom) == OT_OK);
      OT_CHECK(dinfo.chip_gen == OT_CHIP_TPU7X);
      OT_CHECK(dinfo.device_id == 0x0075u);
      OT_CHECK(dgeom.mxu_rows == 256u && dgeom.mxu_cols == 256u && dgeom.supports_native_fp8 == 1u);
    }

    PJRT_Client_Destroy_Args cd = {};
    cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd.client = cc.client;
    OT_CHECK(api->PJRT_Client_Destroy(&cd) == nullptr);
  }

  // 6. On an 8-device client, execute_device outside the executable's assigned_devices ([0, 1])
  //    must be rejected with INVALID_ARGUMENT, while an assigned device succeeds.
  {
    unsetenv("OPENTPU_MOCK_CHIP");
    unsetenv("OPENTPU_MOCK_DEVICE_ID");
    setenv("OPENTPU_MOCK", "1", 1);
    setenv("OPENTPU_MOCK_TOPOLOGY", "tpu7x-4t", 1);

    PJRT_Client_Create_Args cc = {};
    cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    OT_CHECK(api->PJRT_Client_Create(&cc) == nullptr && cc.client != nullptr);

    PJRT_Client_AddressableDevices_Args ad = {};
    ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    ad.client = cc.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
    OT_CHECK(ad.num_addressable_devices == 8u);

    const char opt_rep2[] = {0x1a, 0x02, 0x20, 0x02};
    PJRT_Client_Compile_Args ca = {};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client = cc.client;
    ca.program = &prog;
    ca.compile_options = opt_rep2;
    ca.compile_options_size = sizeof(opt_rep2);
    OT_CHECK(api->PJRT_Client_Compile(&ca) == nullptr && ca.executable != nullptr);

    std::vector<uint16_t> ones(16, 0x3f80u);
    PJRT_Buffer* d2_a = upload_bf16(cc.client, ad.addressable_devices[2], ones, 4, 4);
    PJRT_Buffer* d2_b = upload_bf16(cc.client, ad.addressable_devices[2], ones, 4, 4);
    PJRT_Buffer* ins2[2] = {d2_a, d2_b};
    PJRT_Buffer* outs2[1] = {nullptr};
    PJRT_Buffer* const* arg_lists2[1] = {ins2};
    PJRT_Buffer** const out_lists2[1] = {outs2};
    PJRT_ExecuteOptions eopts = {};
    eopts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    PJRT_LoadedExecutable_Execute_Args ex_unassigned = {};
    ex_unassigned.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex_unassigned.executable = ca.executable;
    ex_unassigned.options = &eopts;
    ex_unassigned.argument_lists = arg_lists2;
    ex_unassigned.num_devices = 1;
    ex_unassigned.num_args = 2;
    ex_unassigned.output_lists = out_lists2;
    ex_unassigned.execute_device = ad.addressable_devices[2];
    PJRT_Error* ue = api->PJRT_LoadedExecutable_Execute(&ex_unassigned);
    OT_CHECK(ue != nullptr && get_err_code(ue) == PJRT_Error_Code_INVALID_ARGUMENT);
    OT_CHECK(outs2[0] == nullptr);
    destroy_pjrt_error(api, ue);

    // Launching on assigned device 1 with execute_device=dev[1] succeeds.
    PJRT_Buffer* d1_a = upload_bf16(cc.client, ad.addressable_devices[1], ones, 4, 4);
    PJRT_Buffer* d1_b = upload_bf16(cc.client, ad.addressable_devices[1], ones, 4, 4);
    PJRT_Buffer* ins1[2] = {d1_a, d1_b};
    PJRT_Buffer* outs1[1] = {nullptr};
    PJRT_Buffer* const* arg_lists1[1] = {ins1};
    PJRT_Buffer** const out_lists1[1] = {outs1};
    PJRT_LoadedExecutable_Execute_Args ex_assigned = ex_unassigned;
    ex_assigned.argument_lists = arg_lists1;
    ex_assigned.output_lists = out_lists1;
    ex_assigned.execute_device = ad.addressable_devices[1];
    OT_CHECK(api->PJRT_LoadedExecutable_Execute(&ex_assigned) == nullptr && outs1[0] != nullptr);

    destroy_buf(d2_a);
    destroy_buf(d2_b);
    destroy_buf(d1_a);
    destroy_buf(d1_b);
    destroy_buf(outs1[0]);
    PJRT_LoadedExecutable_Destroy_Args led = {};
    led.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    led.executable = ca.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_Destroy(&led) == nullptr);
    PJRT_Client_Destroy_Args cd = {};
    cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd.client = cc.client;
    OT_CHECK(api->PJRT_Client_Destroy(&cd) == nullptr);
  }

  // 7. With visible_devices=[1, 3], compile device_ordinal=1 must resolve to the 2nd visible
  //    device (global ID 3), NOT global device ID 1, and device_ordinal=2 must be rejected.
  {
    unsetenv("OPENTPU_MOCK_CHIP");
    unsetenv("OPENTPU_MOCK_DEVICE_ID");
    setenv("OPENTPU_MOCK", "1", 1);
    setenv("OPENTPU_MOCK_TOPOLOGY", "tpu7x-4t", 1);

    const int64_t vis13[2] = {1, 3};
    PJRT_NamedValue nv = {};
    nv.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    nv.name = "visible_devices";
    nv.name_size = std::strlen("visible_devices");
    nv.type = PJRT_NamedValue_kInt64List;
    nv.int64_array_value = vis13;
    nv.value_size = 2;

    PJRT_Client_Create_Args cc = {};
    cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    cc.create_options = &nv;
    cc.num_options = 1;
    OT_CHECK(api->PJRT_Client_Create(&cc) == nullptr && cc.client != nullptr);

    PJRT_Client_AddressableDevices_Args ad = {};
    ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    ad.client = cc.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
    OT_CHECK(ad.num_addressable_devices == 2u);

    auto get_dev_id = [&](PJRT_Device* d) -> int {
      PJRT_Device_GetDescription_Args gd = {};
      gd.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
      gd.device = d;
      OT_CHECK(api->PJRT_Device_GetDescription(&gd) == nullptr);
      PJRT_DeviceDescription_Id_Args id = {};
      id.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
      id.device_description = gd.device_description;
      OT_CHECK(api->PJRT_DeviceDescription_Id(&id) == nullptr);
      return id.id;
    };
    OT_CHECK(get_dev_id(ad.addressable_devices[0]) == 1);
    OT_CHECK(get_dev_id(ad.addressable_devices[1]) == 3);

    // CompileOptions { executable_build_options (field 3) { device_ordinal (field 1) = 1 } }
    const char opt_ord1[] = {0x1a, 0x02, 0x08, 0x01};
    PJRT_Client_Compile_Args ca = {};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client = cc.client;
    ca.program = &prog;
    ca.compile_options = opt_ord1;
    ca.compile_options_size = sizeof(opt_ord1);
    OT_CHECK(api->PJRT_Client_Compile(&ca) == nullptr && ca.executable != nullptr);

    PJRT_LoadedExecutable_AddressableDevices_Args ead = {};
    ead.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
    ead.executable = ca.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_AddressableDevices(&ead) == nullptr);
    OT_CHECK(ead.num_addressable_devices == 1u);
    OT_CHECK(ead.addressable_devices[0] == ad.addressable_devices[1]);
    OT_CHECK(get_dev_id(ead.addressable_devices[0]) == 3);

    // Out-of-range ordinal (device_ordinal=2 >= visible_count=2) must fail with INVALID_ARGUMENT.
    const char opt_ord2[] = {0x1a, 0x02, 0x08, 0x02};
    PJRT_Client_Compile_Args ca_bad = ca;
    ca_bad.compile_options = opt_ord2;
    ca_bad.compile_options_size = sizeof(opt_ord2);
    ca_bad.executable = nullptr;
    PJRT_Error* oob_err = api->PJRT_Client_Compile(&ca_bad);
    OT_CHECK(oob_err != nullptr && get_err_code(oob_err) == PJRT_Error_Code_INVALID_ARGUMENT);
    OT_CHECK(ca_bad.executable == nullptr);
    destroy_pjrt_error(api, oob_err);

    PJRT_LoadedExecutable_Destroy_Args led = {};
    led.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    led.executable = ca.executable;
    OT_CHECK(api->PJRT_LoadedExecutable_Destroy(&led) == nullptr);
    PJRT_Client_Destroy_Args cd = {};
    cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd.client = cc.client;
    OT_CHECK(api->PJRT_Client_Destroy(&cd) == nullptr);
  }

  // 8. Conflicting mock topology + device ID / chip selectors must fail with INVALID_ARGUMENT,
  //    while matching guest Ironwood VFIO ID (0x0076) + tpu7x-4t succeeds consistently.
  {
    setenv("OPENTPU_MOCK", "1", 1);
    unsetenv("OPENTPU_MOCK_CHIP");
    setenv("OPENTPU_MOCK_TOPOLOGY", "tpu7x-4t", 1);
    setenv("OPENTPU_MOCK_DEVICE_ID", "0x0063", 1);

    PJRT_Client_Create_Args cc_bad_did = {};
    cc_bad_did.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* err_did = api->PJRT_Client_Create(&cc_bad_did);
    OT_CHECK(err_did != nullptr && get_err_code(err_did) == PJRT_Error_Code_INVALID_ARGUMENT);
    OT_CHECK(cc_bad_did.client == nullptr);
    destroy_pjrt_error(api, err_did);

    unsetenv("OPENTPU_MOCK_DEVICE_ID");
    setenv("OPENTPU_MOCK_CHIP", "v5e", 1);
    PJRT_Client_Create_Args cc_bad_chip = {};
    cc_bad_chip.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* err_chip = api->PJRT_Client_Create(&cc_bad_chip);
    OT_CHECK(err_chip != nullptr && get_err_code(err_chip) == PJRT_Error_Code_INVALID_ARGUMENT);
    OT_CHECK(cc_bad_chip.client == nullptr);
    destroy_pjrt_error(api, err_chip);

    unsetenv("OPENTPU_MOCK_CHIP");
    setenv("OPENTPU_MOCK_DEVICE_ID", "0x0076", 1);
    PJRT_Client_Create_Args cc_ok = {};
    cc_ok.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    OT_CHECK(api->PJRT_Client_Create(&cc_ok) == nullptr && cc_ok.client != nullptr);
    PJRT_Client_AddressableDevices_Args ad = {};
    ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    ad.client = cc_ok.client;
    OT_CHECK(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
    OT_CHECK(ad.num_addressable_devices == 8u);
    ot_device_info_t dinfo = {};
    OT_CHECK(ot_device_get_info(ot::pjrt_device_context_raw(ad.addressable_devices[0]), &dinfo) == OT_OK);
    OT_CHECK(dinfo.chip_gen == OT_CHIP_TPU7X && dinfo.device_id == 0x0076u);

    PJRT_Client_Destroy_Args cd = {};
    cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd.client = cc_ok.client;
    OT_CHECK(api->PJRT_Client_Destroy(&cd) == nullptr);
    unsetenv("OPENTPU_MOCK_DEVICE_ID");
  }

  unsetenv("OPENTPU_MOCK_TOPOLOGY");
  unsetenv("OPENTPU_MOCK_CHIP");
  unsetenv("OPENTPU_MOCK");
  printf("[PASS] test_pjrt_multidevice_isolation_and_topology\n");
}

int main() {
  test_iova_allocator();
  test_bar4_descriptor_parser();
  test_aligned_dma_arena_alloc();
  test_fork_closes_vfio_fds_and_static_pjrt_err();
  test_mock_dma_and_matmul();
  test_layout_tiling_and_aliasing();
  test_vfio_imem_lifecycle_mock();
  test_pjrt_api_mock();
  test_vliw_and_tiling();
  test_resident_imem_stub_structure();
  test_multigen_hal_geometry_and_rebar();
  test_pjrt_dual_chiplet_tpu7x_enumeration();
  test_pjrt_multidevice_isolation_and_topology();
  printf("ALL_HOST_TESTS_PASSED\n");
  return 0;
}
