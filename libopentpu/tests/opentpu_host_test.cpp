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

  // 1) Monotonic exec_seq & cached loaded_imem_bundles across BF16 -> Q8 -> Q4 -> BF16 switches.
  OT_CHECK(ctx->transport.loaded_prog_kind == 1u);
  OT_CHECK(ctx->transport.loaded_imem_bundles == ot::freetile::kV5eImemBundlesBf16);
  ctx->transport.exec_seq = ctx->transport.next_exec_seq();
  OT_CHECK(ctx->transport.exec_seq == 1u);
  OT_CHECK(ctx->transport.ensure_imem_program(2u, 1000u, &err));
  OT_CHECK(ctx->transport.imem_booted && ctx->transport.loaded_prog_kind == 2u);
  OT_CHECK(ctx->transport.loaded_imem_bundles == ot::freetile::kV5eImemBundlesQuantQ8);
  OT_CHECK(ctx->transport.exec_seq == 1u && ctx->transport.next_exec_seq() == 2u);
  ctx->transport.exec_seq = ctx->transport.next_exec_seq();
  OT_CHECK(ctx->transport.ensure_imem_program(3u, 1000u, &err));
  OT_CHECK(ctx->transport.imem_booted && ctx->transport.loaded_prog_kind == 3u);
  OT_CHECK(ctx->transport.loaded_imem_bundles == ot::freetile::kV5eImemBundlesQuantQ4);
  OT_CHECK(ctx->transport.exec_seq == 2u && ctx->transport.next_exec_seq() == 3u);
  OT_CHECK(ctx->transport.ensure_imem_program(1u, 1000u, &err));
  OT_CHECK(ctx->transport.imem_booted && ctx->transport.loaded_prog_kind == 1u);
  OT_CHECK(ctx->transport.loaded_imem_bundles == ot::freetile::kV5eImemBundlesBf16);
  OT_CHECK(ctx->transport.exec_seq == 2u);
  ctx->transport.exec_seq = 0xffffffffu;
  OT_CHECK(ctx->transport.next_exec_seq() == 1u);
  ctx->transport.exec_seq = 2u;

  // 2) Failed core-start during IMEM program switch must invalidate the old cache.
  ctx->transport.mock_core_ctrl_fail = true;
  err.clear();
  OT_CHECK(!ctx->transport.ensure_imem_program(2u, 1000u, &err));
  OT_CHECK(!ctx->transport.imem_booted && ctx->transport.loaded_prog_kind == 0u &&
           ctx->transport.loaded_imem_bundles == 0u);
  ctx->transport.mock_core_ctrl_fail = false;
  OT_CHECK(ctx->transport.ensure_imem_program(2u, 1000u, &err));
  OT_CHECK(ctx->transport.imem_booted && ctx->transport.loaded_prog_kind == 2u);

  // 2b) ensure_quant_constants HBM OOM must set last_op_oom so transport maps to OT_ERR_NO_MEMORY.
  {
    const uint64_t avail =
        ctx->transport.hbm_alloc.capacity_bytes() - ctx->transport.hbm_alloc.used_bytes();
    uint64_t filler_addr = 0;
    const uint64_t fill_sz = (avail > 16384ULL) ? ((avail - 16384ULL) & ~4095ULL) : 0ULL;
    OT_CHECK(fill_sz > 0ULL && ctx->transport.hbm_alloc.alloc(fill_sz, 4096ULL, &filler_addr));
    err.clear();
    OT_CHECK(!ctx->transport.ensure_quant_constants(1000u, &err));
    OT_CHECK(ctx->transport.last_op_oom);
    OT_CHECK(ctx->transport.quant_const_hbm == 0ULL);
    ctx->transport.hbm_alloc.free(filler_addr);
    OT_CHECK(ctx->transport.ensure_quant_constants(1000u, &err));
    OT_CHECK(!ctx->transport.last_op_oom && ctx->transport.quant_const_hbm != 0ULL);
  }

  uint32_t cq[8] = {};
  OT_CHECK(ctx->transport.submit_exec_sync(7u, 1000u, cq, &err));
  OT_CHECK(cq[4] == 1u && cq[5] == 7u);
  OT_CHECK(!ctx->transport.last_op_timed_out);

  // 3) EXEC polling deadline expiration must report timeout (OT_ERR_TIMEOUT) & quiesce,
  //    while non-timeout EXEC poll failures must ALSO fault & quiesce (OT_ERR_DEVICE).
  OT_CHECK(ot::classify_transport_error(true, true) == OT_ERR_TIMEOUT);
  OT_CHECK(ot::classify_transport_error(false, true) == OT_ERR_DEVICE);
  OT_CHECK(ot::classify_transport_error(false, false) == OT_ERR_IOCTL);
  ctx->transport.mock_exec_timeout = true;
  err.clear();
  OT_CHECK(!ctx->transport.submit_exec_sync(8u, 10u, cq, &err));
  OT_CHECK(ctx->transport.dma_faulted);
  OT_CHECK(ctx->transport.last_op_timed_out);
  OT_CHECK(!ctx->transport.imem_booted && ctx->transport.loaded_prog_kind == 0u);
  OT_CHECK(err.find("EXEC completion timeout") != std::string::npos);
  OT_CHECK(ot::classify_transport_error(ctx->transport.last_op_timed_out,
                                        ctx->transport.dma_faulted) == OT_ERR_TIMEOUT);
  ctx->transport.dma_faulted = false;
  ctx->transport.last_op_timed_out = false;
  ctx->transport.mock_exec_timeout = false;

  OT_CHECK(ctx->transport.ensure_imem_program(1u, 1000u, &err));
  ctx->transport.mock_exec_poll_error = true;
  err.clear();
  OT_CHECK(!ctx->transport.submit_exec_sync(9u, 10u, cq, &err));
  OT_CHECK(ctx->transport.dma_faulted);
  OT_CHECK(!ctx->transport.last_op_timed_out);
  OT_CHECK(!ctx->transport.imem_booted && ctx->transport.loaded_prog_kind == 0u);
  OT_CHECK(err.find("device quiesced") != std::string::npos);
  OT_CHECK(ot::classify_transport_error(ctx->transport.last_op_timed_out,
                                        ctx->transport.dma_faulted) == OT_ERR_DEVICE);
  ctx->transport.dma_faulted = false;
  ctx->transport.mock_exec_poll_error = false;

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

static void test_freetile_descriptor_and_planner() {
  const ot_freetile_gemm_desc_t cdesc = ot_freetile_gemm_desc_default(128u, 4096u, 4096u);
  OT_CHECK(cdesc.struct_size == sizeof(ot_freetile_gemm_desc_t));
  OT_CHECK(cdesc.tile.block_m == 128u && cdesc.tile.block_k == 128u && cdesc.tile.block_n == 128u);
  OT_CHECK(cdesc.tile.vmem_double_buffer == 1u && cdesc.tile.k_loop_accum_f32 == 1u);

  const auto fdesc = ot::freetile::TileKernelDescriptor::from_c_desc(cdesc);
  const ot::freetile::LoweringPlan bf16_plan = ot::freetile::plan_v5e_tile_gemm(fdesc);
  OT_CHECK(bf16_plan.valid);
  OT_CHECK(bf16_plan.tiles_m == 1u && bf16_plan.tiles_k == 32u && bf16_plan.tiles_n == 32u);
  OT_CHECK(bf16_plan.vmem.stages == 2u);
  OT_CHECK(bf16_plan.desc.block_spec.vmem_double_buffer);
  OT_CHECK(bf16_plan.vmem.acc_vmem_bytes == 0u);  // Accumulators live in v48..v63 registers
  OT_CHECK(bf16_plan.vmem.total_vmem_bytes == 2u * (32768u + 32768u) + 32768u);
  OT_CHECK(bf16_plan.vmem.total_vmem_bytes <= 16u * 1024u * 1024u);

  // Multi-K BF16 always emits 2-stage ping-pong VMEM and FP32 K-accumulation even if hints are 0;
  // single-K BF16 uses only stage 0 (1 stage).
  ot_freetile_gemm_desc_t bf16_hint0 = ot_freetile_gemm_desc_default(128u, 512u, 256u);
  bf16_hint0.tile.vmem_double_buffer = 0u;
  bf16_hint0.tile.k_loop_accum_f32 = 0u;
  const ot::freetile::LoweringPlan bf16_hint0_plan =
      ot::freetile::plan_v5e_tile_gemm(ot::freetile::TileKernelDescriptor::from_c_desc(bf16_hint0));
  OT_CHECK(bf16_hint0_plan.valid);
  OT_CHECK(bf16_hint0_plan.vmem.stages == 2u);
  OT_CHECK(bf16_hint0_plan.desc.block_spec.vmem_double_buffer);
  OT_CHECK(bf16_hint0_plan.desc.block_spec.k_loop_accum_f32);
  OT_CHECK(bf16_hint0_plan.vmem.total_vmem_bytes == bf16_plan.vmem.total_vmem_bytes);

  const ot_freetile_gemm_desc_t bf16_k1 = ot_freetile_gemm_desc_default(256u, 128u, 256u);
  const ot::freetile::LoweringPlan bf16_k1_plan =
      ot::freetile::plan_v5e_tile_gemm(ot::freetile::TileKernelDescriptor::from_c_desc(bf16_k1));
  OT_CHECK(bf16_k1_plan.valid);
  OT_CHECK(bf16_k1_plan.vmem.stages == 1u);
  OT_CHECK(!bf16_k1_plan.desc.block_spec.vmem_double_buffer);
  OT_CHECK(bf16_k1_plan.vmem.acc_vmem_bytes == 0u);
  OT_CHECK(bf16_k1_plan.vmem.total_vmem_bytes == 1u * (32768u + 32768u) + 32768u);

  ot_freetile_gemm_desc_t q4 = ot_freetile_gemm_desc_default(128u, 512u, 512u);
  q4.rhs_dtype = OT_DTYPE_Q4_0;
  q4.quant_mode = OT_QUANT_BLOCK_32;
  q4.quant_group_size = OT_QK8_0;
  const ot::freetile::LoweringPlan q4_plan =
      ot::freetile::plan_v5e_tile_gemm(ot::freetile::TileKernelDescriptor::from_c_desc(q4));
  OT_CHECK(q4_plan.valid);
  // Quantized IMEM kernel is single-stage: 32 KiB LHS + 8 KiB Q4 + 32 KiB cleared scale window
  // + 32 KiB dequant BF16 staging + 36 KiB constants + 32 KiB OUT (0 VMEM for register acc).
  OT_CHECK(q4_plan.vmem.stages == 1u);
  OT_CHECK(!q4_plan.desc.block_spec.vmem_double_buffer);
  OT_CHECK(q4_plan.vmem.rhs_vmem_bytes == 8192u);
  OT_CHECK(q4_plan.vmem.scale_vmem_bytes == 32768u);
  OT_CHECK(q4_plan.vmem.dequant_vmem_bytes == 32768u);
  OT_CHECK(q4_plan.vmem.const_vmem_bytes == 36864u);
  OT_CHECK(q4_plan.vmem.acc_vmem_bytes == 0u);
  OT_CHECK(q4_plan.vmem.total_vmem_bytes ==
           32768u + 8192u + 32768u + 32768u + 36864u + 32768u);
  const uint64_t expected_q4_512_hbm =
      4ULL * (128ULL * 512ULL * 2ULL) +       // tiles_n * A
      1ULL * (512ULL * 512ULL / 2ULL) +       // tiles_m * B_q4
      (128ULL * 512ULL * 2ULL) +              // C
      (1ULL * 4ULL * 4ULL) * 1024ULL +        // 1 KiB scale tile per (m,n,k) iter
      1ULL * 36864ULL;                        // 36 KiB constants per launch chunk
  OT_CHECK(q4_plan.hbm_bytes_moved == expected_q4_512_hbm);
  OT_CHECK(q4_plan.hbm_bytes_moved < bf16_plan.hbm_bytes_moved);

  // Verify O(1) closed-form count_v5e_launch_chunks matches the reference chunking loop
  // across an exhaustive grid including remainder m_chunks and k_tiles >= max_iters.
  auto ref_launch_chunks = [](uint32_t mt, uint32_t kt, uint32_t nt, uint32_t lim) -> uint64_t {
    const uint32_t m_step = (kt >= lim) ? 1u : std::min(mt, std::max(1u, lim / kt));
    const uint32_t n_base = (kt >= lim) ? 1u : std::max(1u, lim / kt);
    uint64_t c = 0ULL;
    for (uint32_t ms = 0u; ms < mt;) {
      const uint32_t mc = std::min(m_step, mt - ms);
      const uint32_t n_max = std::max(1u, n_base / mc);
      for (uint32_t ns = 0u; ns < nt;) {
        ns += std::min(n_max, nt - ns);
        ++c;
      }
      ms += mc;
    }
    return c;
  };
  const uint32_t ms_grid[] = {1u, 2u, 3u, 5u, 7u, 13u, 32u};
  const uint32_t ks_grid[] = {1u, 2u, 3u, 7u, 16u, 32u, 64u, 128u, 513u, 1023u, 1024u, 1025u, 2048u};
  const uint32_t ns_grid[] = {1u, 2u, 3u, 5u, 9u, 32u, 65u};
  for (uint32_t mt : ms_grid) {
    for (uint32_t kt : ks_grid) {
      for (uint32_t nt : ns_grid) {
        OT_CHECK(ot::freetile::count_v5e_launch_chunks(mt, kt, nt) ==
                 ref_launch_chunks(mt, kt, nt, ot::freetile::kV5eMaxItersPerLaunch));
      }
    }
  }
  OT_CHECK(ot::freetile::count_v5e_launch_chunks(8192u, 1024u, 8192u) == 67108864ULL);

  // Multi-chunk quantized shape (2*32*32 = 2048 iters -> 2 launch chunks of 36 KiB constants).
  ot_freetile_gemm_desc_t q8_2c = ot_freetile_gemm_desc_default(256u, 4096u, 4096u);
  q8_2c.rhs_dtype = OT_DTYPE_Q8_0;
  q8_2c.quant_mode = OT_QUANT_BLOCK_32;
  q8_2c.quant_group_size = OT_QK8_0;
  const ot::freetile::LoweringPlan q8_2c_plan =
      ot::freetile::plan_v5e_tile_gemm(ot::freetile::TileKernelDescriptor::from_c_desc(q8_2c));
  OT_CHECK(q8_2c_plan.valid);
  OT_CHECK(ot::freetile::count_v5e_launch_chunks(2u, 32u, 32u) == 2u);
  const uint64_t expected_q8_2c_hbm =
      32ULL * (256ULL * 4096ULL * 2ULL) +
      2ULL * (4096ULL * 4096ULL) +
      (256ULL * 4096ULL * 2ULL) +
      (2ULL * 32ULL * 32ULL) * 1024ULL +
      2ULL * 36864ULL;
  OT_CHECK(q8_2c_plan.hbm_bytes_moved == expected_q8_2c_hbm);

  // Reserved Q4_K / BLOCK_256 and unsupported quantization descriptors must be rejected by planner.
  ot_freetile_gemm_desc_t q4k = ot_freetile_gemm_desc_default(128u, 256u, 256u);
  q4k.rhs_dtype = OT_DTYPE_Q4_K;
  q4k.quant_mode = OT_QUANT_BLOCK_256;
  q4k.quant_group_size = 256u;
  OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(
                ot::freetile::TileKernelDescriptor::from_c_desc(q4k))
                .valid);

  ot_freetile_gemm_desc_t i8_none = ot_freetile_gemm_desc_default(128u, 256u, 256u);
  i8_none.rhs_dtype = OT_DTYPE_INT8;
  i8_none.quant_mode = OT_QUANT_NONE;
  OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(
                ot::freetile::TileKernelDescriptor::from_c_desc(i8_none))
                .valid);

  ot_freetile_gemm_desc_t bf16_badq = ot_freetile_gemm_desc_default(128u, 256u, 256u);
  bf16_badq.quant_mode = OT_QUANT_BLOCK_32;
  OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(
                ot::freetile::TileKernelDescriptor::from_c_desc(bf16_badq))
                .valid);

  ot_freetile_gemm_desc_t q8_bad_sd = q8_2c;
  q8_bad_sd.scale_dtype = OT_DTYPE_INT8;
  OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(
                ot::freetile::TileKernelDescriptor::from_c_desc(q8_bad_sd))
                .valid);

  // High-bit-polluted uint32_t descriptor fields must NOT silently narrow to valid 8-bit enums.
  {
    ot_freetile_gemm_desc_t trunc_sd = q8_2c;
    trunc_sd.scale_dtype = 257u;  // (uint8_t)257 == OT_DTYPE_F32
    OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(
                  ot::freetile::TileKernelDescriptor::from_c_desc(trunc_sd))
                  .valid);
    ot_freetile_gemm_desc_t trunc_rhs = q8_2c;
    trunc_rhs.rhs_dtype = 257u;
    OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(
                  ot::freetile::TileKernelDescriptor::from_c_desc(trunc_rhs))
                  .valid);
    ot_freetile_gemm_desc_t trunc_lhs = q8_2c;
    trunc_lhs.lhs_dtype = 256u;  // (uint8_t)256 == OT_DTYPE_BF16
    OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(
                  ot::freetile::TileKernelDescriptor::from_c_desc(trunc_lhs))
                  .valid);
    ot_freetile_gemm_desc_t trunc_qm = q8_2c;
    trunc_qm.quant_mode = 259u;  // (uint8_t)259 == OT_QUANT_BLOCK_32
    const auto trunc_qm_desc = ot::freetile::TileKernelDescriptor::from_c_desc(trunc_qm);
    OT_CHECK(trunc_qm_desc.quant.mode == OT_QUANT_NONE && trunc_qm_desc.has_invalid_c_field);
    OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(trunc_qm_desc).valid);
  }

  // Valid lowering plans must populate imem_bundles matching the exact emitted IMEM kernel size.
  OT_CHECK(bf16_plan.imem_bundles > 0u &&
           bf16_plan.imem_bundles == ot::emit_v5e_imem_for_freetile(bf16_plan).size());
  OT_CHECK(bf16_k1_plan.imem_bundles > 0u &&
           bf16_k1_plan.imem_bundles == ot::emit_v5e_imem_for_freetile(bf16_k1_plan).size());
  OT_CHECK(q8_2c_plan.imem_bundles > 0u &&
           q8_2c_plan.imem_bundles == ot::emit_v5e_imem_for_freetile(q8_2c_plan).size());
  OT_CHECK(q4_plan.imem_bundles > 0u &&
           q4_plan.imem_bundles == ot::emit_v5e_imem_for_freetile(q4_plan).size());

  // Quantized planner must also normalize k_loop_accum_f32 to true (matching emitted FP32 accum).
  {
    ot_freetile_gemm_desc_t q4_acc0 = q4;
    q4_acc0.tile.k_loop_accum_f32 = 0u;
    const ot::freetile::LoweringPlan q4_acc0_plan =
        ot::freetile::plan_v5e_tile_gemm(ot::freetile::TileKernelDescriptor::from_c_desc(q4_acc0));
    OT_CHECK(q4_acc0_plan.valid);
    OT_CHECK(q4_acc0_plan.desc.block_spec.k_loop_accum_f32);
  }

  OT_CHECK(q8_2c_plan.vmem.total_vmem_bytes ==
           32768u + 16384u + 32768u + 32768u + 36864u + 32768u);

  ot_freetile_gemm_desc_t bad = ot_freetile_gemm_desc_default(128u, 130u, 128u);
  const ot::freetile::LoweringPlan bad_plan =
      ot::freetile::plan_v5e_tile_gemm(ot::freetile::TileKernelDescriptor::from_c_desc(bad));
  OT_CHECK(!bad_plan.valid);
  printf("[PASS] test_freetile_descriptor_and_planner\n");
}

static void test_freetile_vliw_emitters_and_packing() {
  const auto bf16_prog = ot::emit_v5e_resident_freetile_bf16_imem();
  const auto q8_prog = ot::emit_v5e_resident_freetile_quant_imem(/*is_q4=*/false);
  const auto q4_prog = ot::emit_v5e_resident_freetile_quant_imem(/*is_q4=*/true);

  for (const auto* prog : {&bf16_prog, &q8_prog, &q4_prog}) {
    OT_CHECK(!prog->empty());
    OT_CHECK((prog->size() * sizeof(ot::VliwBundle)) % ot::kImemDmaAlignBytes == 0u);
    OT_CHECK(prog->size() <= 1024u);  // Strictly fits within 64 KiB resident IMEM window
    OT_CHECK((*prog)[0].w[2] == 0x3b00060fu);
    OT_CHECK((*prog)[25].w[2] == 0x3a8ca30fu);
  }

  ot_freetile_gemm_desc_t cq8 = ot_freetile_gemm_desc_default(128u, 256u, 256u);
  cq8.rhs_dtype = OT_DTYPE_Q8_0;
  cq8.quant_mode = OT_QUANT_BLOCK_32;
  const ot::freetile::LoweringPlan p =
      ot::freetile::plan_v5e_tile_gemm(ot::freetile::TileKernelDescriptor::from_c_desc(cq8));
  OT_CHECK(ot::emit_v5e_imem_for_freetile(p).size() == q8_prog.size());

  // Verify T(8,128)(4,1) int8 and q4 packing byte lanes.
  std::vector<int8_t> i8_rm(128u * 128u, 0);
  i8_rm[0u * 128u + 3u] = 17;     // r=0  -> chunk=0, sublane=0, byte 0
  i8_rm[8u * 128u + 3u] = -5;     // r=8  -> chunk=0, sublane=0, byte 1
  i8_rm[16u * 128u + 3u] = 42;    // r=16 -> chunk=0, sublane=0, byte 2
  i8_rm[24u * 128u + 3u] = -128;  // r=24 -> chunk=0, sublane=0, byte 3
  std::vector<uint32_t> i8_tiled(4096u, 0u);
  ot::tile_pack_128x128_i8(i8_rm.data(), i8_tiled.data());
  OT_CHECK((i8_tiled[3] & 0xffu) == 17u);
  OT_CHECK(static_cast<int8_t>((i8_tiled[3] >> 8) & 0xffu) == -5);
  OT_CHECK(((i8_tiled[3] >> 16) & 0xffu) == 42u);
  OT_CHECK(static_cast<int8_t>((i8_tiled[3] >> 24) & 0xffu) == -128);

  std::vector<int8_t> q4_rm(128u * 128u, 0);
  q4_rm[0u * 128u + 5u] = -3;   // chunk4=0, lo_nib -> (-3 + 8) = 5
  q4_rm[32u * 128u + 5u] = 6;   // chunk4=0, hi_nib -> (6 + 8) = 14
  std::vector<uint32_t> q4_tiled(2048u, 0u);
  ot::tile_pack_128x128_q4(q4_rm.data(), q4_tiled.data());
  OT_CHECK((q4_tiled[5] & 0xfu) == 5u);
  OT_CHECK(((q4_tiled[5] >> 4) & 0xfu) == 14u);

  const auto ldesc = ot::OtV5eLaunchDesc::build_ex(
      0x1000ULL, 0x2000ULL, 0x3000ULL, 0ULL, 7u, 2u, 4u, 3u, 1u, 0x4000ULL, 0x5000ULL, 32u);
  auto read32 = [&](size_t off) {
    uint32_t v = 0;
    std::memcpy(&v, ldesc.bytes + off, 4);
    return v;
  };
  OT_CHECK(read32(0x178u) == 2u && read32(0x180u) == 4u && read32(0x184u) == 3u);
  OT_CHECK(read32(0x188u) == 1u && read32(0x18cu) == (0x4000u >> 9) &&
           read32(0x194u) == (0x5000u >> 9) && read32(0x198u) == 32u);
  printf("[PASS] test_freetile_vliw_emitters_and_packing\n");
}

static uint16_t host_f32_to_bf16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

static float host_bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, 4);
  return f;
}

static void test_mock_multitile_and_quant_matmul() {
  ot_config_t cfg = ot_config_default();
  cfg.mock_mode = 1u;
  ot_context_t* ctx = nullptr;
  OT_CHECK(ot_context_create(&cfg, &ctx) == OT_OK && ctx != nullptr);

  // 1) Multi-tile 256 x 256 x 256 BF16 GEMM via ot_matmul_bf16_ex
  const uint32_t M = 256u, K = 256u, N = 256u;
  std::vector<uint16_t> a_bf16(M * K), b_bf16(K * N), c_got(M * N, 0u);
  for (size_t i = 0; i < a_bf16.size(); ++i) {
    a_bf16[i] = host_f32_to_bf16(static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f);
  }
  for (size_t i = 0; i < b_bf16.size(); ++i) {
    b_bf16[i] = host_f32_to_bf16(static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25f);
  }
  ot_buffer_t* ba = nullptr;
  ot_buffer_t* bb = nullptr;
  ot_buffer_t* bc = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx, a_bf16.size() * 2u, &ba) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, b_bf16.size() * 2u, &bb) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, c_got.size() * 2u, &bc) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, ba, a_bf16.data(), a_bf16.size() * 2u) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bb, b_bf16.data(), b_bf16.size() * 2u) == OT_OK);

  const ot_freetile_gemm_desc_t bf_desc = ot_freetile_gemm_desc_default(M, K, N);
  ot_exec_stats_t stats{};
  OT_CHECK(ot_matmul_bf16_ex(ctx, ba, bb, bc, &bf_desc, &stats) == OT_OK);
  OT_CHECK(stats.tiles_m == 2u && stats.tiles_k == 2u && stats.tiles_n == 2u);
  OT_CHECK(ot_buffer_d2h(ctx, bc, c_got.data(), c_got.size() * 2u) == OT_OK);

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        acc += host_bf16_to_f32(a_bf16[m * K + k]) * host_bf16_to_f32(b_bf16[k * N + n]);
      }
      OT_CHECK(c_got[m * N + n] == host_f32_to_bf16(acc));
    }
  }
  ot_buffer_free(bc);
  ot_buffer_free(bb);
  ot_buffer_free(ba);

  // 2) Quantized 128 x 256 x 256 Q8_0 (block-32) & Q4_0 (block-32) & INT8_PER_CHANNEL
  const uint32_t QM = 128u, QK = 256u, QN = 256u;
  std::vector<uint16_t> qa(QM * QK);
  for (size_t i = 0; i < qa.size(); ++i) {
    qa[i] = host_f32_to_bf16(static_cast<float>(static_cast<int>(i % 9) - 4) * 0.125f);
  }
  std::vector<int8_t> w_i8(QK * QN);
  std::vector<uint8_t> w_q4(QK * QN / 2u, 0u);
  std::vector<float> sc_blk((QK / 32u) * QN);
  for (size_t i = 0; i < sc_blk.size(); ++i) {
    sc_blk[i] = 0.0625f * static_cast<float>((i % 7u) + 1u);
  }
  for (uint32_t k = 0; k < QK; ++k) {
    for (uint32_t n = 0; n < QN; ++n) {
      const int8_t v = static_cast<int8_t>(static_cast<int>((k * 13u + n * 7u) % 15u) - 7);
      w_i8[k * QN + n] = v;
      const size_t lin = static_cast<size_t>(k) * QN + n;
      const uint32_t nib = static_cast<uint32_t>(v + 8) & 0xfu;
      w_q4[lin >> 1] |= static_cast<uint8_t>((lin & 1u) ? (nib << 4) : nib);
    }
  }

  ot_buffer_t *bqa = nullptr, *bqi8 = nullptr, *bqq4 = nullptr, *bqsc = nullptr, *bqout = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx, qa.size() * 2u, &bqa) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, w_i8.size(), &bqi8) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, w_q4.size(), &bqq4) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, sc_blk.size() * sizeof(float), &bqsc) == OT_OK);
  OT_CHECK(ot_buffer_alloc(ctx, QM * QN * 2u, &bqout) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bqa, qa.data(), qa.size() * 2u) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bqi8, w_i8.data(), w_i8.size()) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bqq4, w_q4.data(), w_q4.size()) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bqsc, sc_blk.data(), sc_blk.size() * sizeof(float)) == OT_OK);

  ot_freetile_gemm_desc_t q8d = ot_freetile_gemm_desc_default(QM, QK, QN);
  q8d.rhs_dtype = OT_DTYPE_Q8_0;
  q8d.quant_mode = OT_QUANT_BLOCK_32;
  q8d.quant_group_size = OT_QK8_0;
  q8d.scale_dtype = OT_DTYPE_F32;
  OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, bqsc, bqout, &q8d, &stats) == OT_OK);
  const uint64_t expected_q8_hbm =
      2ULL * (static_cast<uint64_t>(QM) * QK * 2ULL) +
      1ULL * (static_cast<uint64_t>(QK) * QN) +
      (static_cast<uint64_t>(QM) * QN * 2ULL) +
      (1ULL * 2ULL * 2ULL) * 1024ULL +
      36864ULL;
  OT_CHECK(stats.hbm_bytes_moved == expected_q8_hbm);
  std::vector<uint16_t> out_q8(QM * QN, 0u);
  OT_CHECK(ot_buffer_d2h(ctx, bqout, out_q8.data(), out_q8.size() * 2u) == OT_OK);

  ot_freetile_gemm_desc_t q4d = ot_freetile_gemm_desc_default(QM, QK, QN);
  q4d.rhs_dtype = OT_DTYPE_Q4_0;
  q4d.quant_mode = OT_QUANT_BLOCK_32;
  q4d.quant_group_size = OT_QK8_0;
  q4d.scale_dtype = OT_DTYPE_F32;
  OT_CHECK(ot_matmul_quant(ctx, bqa, bqq4, bqsc, bqout, &q4d, &stats) == OT_OK);
  const uint64_t expected_q4_hbm =
      2ULL * (static_cast<uint64_t>(QM) * QK * 2ULL) +
      1ULL * (static_cast<uint64_t>(QK) * QN / 2ULL) +
      (static_cast<uint64_t>(QM) * QN * 2ULL) +
      (1ULL * 2ULL * 2ULL) * 1024ULL +
      36864ULL;
  OT_CHECK(stats.hbm_bytes_moved == expected_q4_hbm);
  std::vector<uint16_t> out_q4(QM * QN, 0u);
  OT_CHECK(ot_buffer_d2h(ctx, bqout, out_q4.data(), out_q4.size() * 2u) == OT_OK);

  // Q4 values stay within [-7, 7], matching w_i8 above cleanly!
  std::vector<uint16_t> ref_q(QM * QN, 0u);
  for (uint32_t m = 0; m < QM; ++m) {
    for (uint32_t n = 0; n < QN; ++n) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < QK; ++k) {
        const float sc = host_bf16_to_f32(host_f32_to_bf16(sc_blk[(k / 32u) * QN + n]));
        const float wf = host_bf16_to_f32(host_f32_to_bf16(static_cast<float>(w_i8[k * QN + n]) * sc));
        acc += host_bf16_to_f32(qa[m * QK + k]) * wf;
      }
      ref_q[m * QN + n] = host_f32_to_bf16(acc);
      OT_CHECK(out_q8[m * QN + n] == ref_q[m * QN + n]);
      OT_CHECK(out_q4[m * QN + n] == ref_q[m * QN + n]);
    }
  }

  // 3) Scale dtype must depend strictly on desc->scale_dtype, never buffer capacity:
  //    store BF16 scales inside a 2x-oversized (float-sized) buffer.
  {
    std::vector<uint16_t> sc_bf16(sc_blk.size());
    for (size_t i = 0; i < sc_blk.size(); ++i) sc_bf16[i] = host_f32_to_bf16(sc_blk[i]);
    ot_buffer_t* bqsc_big = nullptr;
    OT_CHECK(ot_buffer_alloc(ctx, sc_blk.size() * sizeof(float), &bqsc_big) == OT_OK);
    // Fill upper half with poison so accidental FP32 decoding would corrupt scales.
    std::vector<uint8_t> sc_bytes(sc_blk.size() * sizeof(float), 0x7fu);
    std::memcpy(sc_bytes.data(), sc_bf16.data(), sc_bf16.size() * sizeof(uint16_t));
    OT_CHECK(ot_buffer_h2d(ctx, bqsc_big, sc_bytes.data(), sc_bytes.size()) == OT_OK);

    ot_freetile_gemm_desc_t q8_bf16sc = q8d;
    q8_bf16sc.scale_dtype = OT_DTYPE_BF16;
    std::vector<uint16_t> out_bf16sc(QM * QN, 0u);
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, bqsc_big, bqout, &q8_bf16sc, nullptr) == OT_OK);
    OT_CHECK(ot_buffer_d2h(ctx, bqout, out_bf16sc.data(), out_bf16sc.size() * 2u) == OT_OK);
    OT_CHECK(out_bf16sc == ref_q);

    ot_freetile_gemm_desc_t bad_sd = q8d;
    bad_sd.scale_dtype = OT_DTYPE_INT8;
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, bqsc_big, bqout, &bad_sd, nullptr) ==
             OT_ERR_INVALID_ARG);
    bad_sd.scale_dtype = 257u;  // Must reject non-narrowed uint32_t enum value
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, bqsc_big, bqout, &bad_sd, nullptr) ==
             OT_ERR_INVALID_ARG);
    ot_freetile_gemm_desc_t bad_rhs = q8d;
    bad_rhs.rhs_dtype = 257u;
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, bqsc_big, bqout, &bad_rhs, nullptr) ==
             OT_ERR_INVALID_ARG);
    ot_buffer_free(bqsc_big);
  }

  // 4) Inline interleaved ot_block_q8_0_t and ot_block_q4_0_t (rhs_scales == NULL),
  //    both non-transposed [K/32, N] and transposed [N, K/32].
  for (uint8_t trans = 0u; trans <= 1u; ++trans) {
    const size_t k_blks = QK / OT_QK8_0;
    std::vector<ot_block_q8_0_t> blk_q8(k_blks * QN);
    std::vector<ot_block_q4_0_t> blk_q4(k_blks * QN);
    std::memset(blk_q8.data(), 0, blk_q8.size() * sizeof(ot_block_q8_0_t));
    std::memset(blk_q4.data(), 0, blk_q4.size() * sizeof(ot_block_q4_0_t));
    for (size_t sg = 0; sg < k_blks; ++sg) {
      for (size_t n = 0; n < QN; ++n) {
        const size_t bidx = trans ? (n * k_blks + sg) : (sg * QN + n);
        const uint16_t d = host_f32_to_bf16(sc_blk[sg * QN + n]);
        blk_q8[bidx].d = d;
        blk_q4[bidx].d = d;
        for (size_t ki = 0; ki < OT_QK8_0; ++ki) {
          const int8_t v = w_i8[(sg * OT_QK8_0 + ki) * QN + n];
          blk_q8[bidx].qs[ki] = v;
          const uint32_t nib = static_cast<uint32_t>(v + 8) & 0xfu;
          blk_q4[bidx].qs[ki >> 1] |= static_cast<uint8_t>((ki & 1u) ? (nib << 4) : nib);
        }
      }
    }
    ot_buffer_t *bbq8 = nullptr, *bbq4 = nullptr;
    OT_CHECK(ot_buffer_alloc(ctx, blk_q8.size() * sizeof(ot_block_q8_0_t), &bbq8) == OT_OK);
    OT_CHECK(ot_buffer_alloc(ctx, blk_q4.size() * sizeof(ot_block_q4_0_t), &bbq4) == OT_OK);
    OT_CHECK(ot_buffer_h2d(ctx, bbq8, blk_q8.data(), blk_q8.size() * sizeof(ot_block_q8_0_t)) ==
             OT_OK);
    OT_CHECK(ot_buffer_h2d(ctx, bbq4, blk_q4.data(), blk_q4.size() * sizeof(ot_block_q4_0_t)) ==
             OT_OK);

    ot_freetile_gemm_desc_t d_q8 = q8d;
    d_q8.tile.rhs_transposed = trans;
    std::vector<uint16_t> got_blk(QM * QN, 0u);
    OT_CHECK(ot_matmul_quant(ctx, bqa, bbq8, nullptr, bqout, &d_q8, nullptr) == OT_OK);
    OT_CHECK(ot_buffer_d2h(ctx, bqout, got_blk.data(), got_blk.size() * 2u) == OT_OK);
    OT_CHECK(got_blk == ref_q);

    ot_freetile_gemm_desc_t d_q4 = q4d;
    d_q4.tile.rhs_transposed = trans;
    std::fill(got_blk.begin(), got_blk.end(), 0u);
    OT_CHECK(ot_matmul_quant(ctx, bqa, bbq4, nullptr, bqout, &d_q4, nullptr) == OT_OK);
    OT_CHECK(ot_buffer_d2h(ctx, bqout, got_blk.data(), got_blk.size() * 2u) == OT_OK);
    OT_CHECK(got_blk == ref_q);

    ot_buffer_free(bbq4);
    ot_buffer_free(bbq8);
  }

  // Reserved Q4_K / BLOCK_256 must return OT_ERR_UNSUPPORTED; INT8 BLOCK_32 without scales invalid.
  {
    ot_freetile_gemm_desc_t q4kd = q4d;
    q4kd.rhs_dtype = OT_DTYPE_Q4_K;
    q4kd.quant_mode = OT_QUANT_BLOCK_256;
    q4kd.quant_group_size = 256u;
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqq4, bqsc, bqout, &q4kd, nullptr) == OT_ERR_UNSUPPORTED);

    ot_freetile_gemm_desc_t i8_nosc = q8d;
    i8_nosc.rhs_dtype = OT_DTYPE_INT8;
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, nullptr, bqout, &i8_nosc, nullptr) ==
             OT_ERR_INVALID_ARG);

    // INT8 with OT_QUANT_NONE must be rejected by both planner and ot_matmul_quant.
    ot_freetile_gemm_desc_t i8_none = q8d;
    i8_none.rhs_dtype = OT_DTYPE_INT8;
    i8_none.quant_mode = OT_QUANT_NONE;
    i8_none.quant_group_size = 0u;
    OT_CHECK(!ot::freetile::plan_v5e_tile_gemm(
                  ot::freetile::TileKernelDescriptor::from_c_desc(i8_none))
                  .valid);
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, bqsc, bqout, &i8_none, nullptr) ==
             OT_ERR_INVALID_ARG);

    // Huge dimensions with small buffers must fail fast on buffer-size validation
    // (O(1), never spinning in launch planning).
    ot_freetile_gemm_desc_t huge_q = ot_freetile_gemm_desc_default(1048576u, 131072u, 1048576u);
    huge_q.rhs_dtype = OT_DTYPE_Q8_0;
    huge_q.quant_mode = OT_QUANT_BLOCK_32;
    huge_q.quant_group_size = OT_QK8_0;
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, bqsc, bqout, &huge_q, nullptr) ==
             OT_ERR_INVALID_ARG);
  }

  // 5) Transposed RHS in BF16 GEMM (rhs_transposed = 1, [N, K] storage) + validation.
  {
    const uint32_t TM = 128u, TK = 256u, TN = 128u;
    std::vector<uint16_t> ta(TM * TK), tb_nk(TN * TK), tc_got(TM * TN, 0u), tc_ref(TM * TN, 0u);
    for (uint32_t m = 0; m < TM; ++m) {
      for (uint32_t k = 0; k < TK; ++k) {
        ta[m * TK + k] =
            host_f32_to_bf16(static_cast<float>(static_cast<int>((m * 7u + k * 3u) % 13u) - 6) * 0.125f);
      }
    }
    for (uint32_t n = 0; n < TN; ++n) {
      for (uint32_t k = 0; k < TK; ++k) {
        tb_nk[n * TK + k] =
            host_f32_to_bf16(static_cast<float>(static_cast<int>((n * 11u + k * 5u) % 11u) - 5) * 0.25f);
      }
    }
    for (uint32_t m = 0; m < TM; ++m) {
      for (uint32_t n = 0; n < TN; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < TK; ++k) {
          acc += host_bf16_to_f32(ta[m * TK + k]) * host_bf16_to_f32(tb_nk[n * TK + k]);
        }
        tc_ref[m * TN + n] = host_f32_to_bf16(acc);
      }
    }
    ot_buffer_t *bta = nullptr, *btb = nullptr, *btc = nullptr;
    OT_CHECK(ot_buffer_alloc(ctx, ta.size() * 2u, &bta) == OT_OK);
    OT_CHECK(ot_buffer_alloc(ctx, tb_nk.size() * 2u, &btb) == OT_OK);
    OT_CHECK(ot_buffer_alloc(ctx, tc_got.size() * 2u, &btc) == OT_OK);
    OT_CHECK(ot_buffer_h2d(ctx, bta, ta.data(), ta.size() * 2u) == OT_OK);
    OT_CHECK(ot_buffer_h2d(ctx, btb, tb_nk.data(), tb_nk.size() * 2u) == OT_OK);

    ot_freetile_gemm_desc_t tdesc = ot_freetile_gemm_desc_default(TM, TK, TN);
    tdesc.tile.rhs_transposed = 1u;
    OT_CHECK(ot_matmul_bf16_ex(ctx, bta, btb, btc, &tdesc, nullptr) == OT_OK);
    OT_CHECK(ot_buffer_d2h(ctx, btc, tc_got.data(), tc_got.size() * 2u) == OT_OK);
    OT_CHECK(tc_got == tc_ref);

    // Invalid rhs_transposed > 1 must fail.
    tdesc.tile.rhs_transposed = 2u;
    OT_CHECK(ot_matmul_bf16_ex(ctx, bta, btb, btc, &tdesc, nullptr) == OT_ERR_INVALID_ARG);
    ot_freetile_gemm_desc_t bad_qt = q8d;
    bad_qt.tile.rhs_transposed = 2u;
    OT_CHECK(ot_matmul_quant(ctx, bqa, bqi8, bqsc, bqout, &bad_qt, nullptr) == OT_ERR_INVALID_ARG);

    ot_buffer_free(btc);
    ot_buffer_free(btb);
    ot_buffer_free(bta);
  }

  // 6) Reusing a 256x256 tiled output buffer for smaller tiled (128x128) and
  //    non-tile-aligned (64x64) GEMMs must preserve the entire logical suffix,
  //    and feeding a tiled output back as an aliased / mismatched-shape input
  //    must read logical row-major elements without double-tiling.
  {
    const uint32_t BM = 256u, BK = 256u, BN = 256u;
    std::vector<uint16_t> ma(BM * BK), mb(BK * BN);
    for (size_t i = 0; i < ma.size(); ++i) {
      ma[i] = host_f32_to_bf16(static_cast<float>(static_cast<int>(i % 13u) - 6) * 0.125f);
    }
    for (size_t i = 0; i < mb.size(); ++i) {
      mb[i] = host_f32_to_bf16(static_cast<float>(static_cast<int>(i % 11u) - 5) * 0.125f);
    }
    ot_buffer_t *bma = nullptr, *bmb = nullptr, *bmc = nullptr;
    OT_CHECK(ot_buffer_alloc(ctx, ma.size() * 2u, &bma) == OT_OK);
    OT_CHECK(ot_buffer_alloc(ctx, mb.size() * 2u, &bmb) == OT_OK);
    OT_CHECK(ot_buffer_alloc(ctx, BM * BN * 2u, &bmc) == OT_OK);
    OT_CHECK(ot_buffer_h2d(ctx, bma, ma.data(), ma.size() * 2u) == OT_OK);
    OT_CHECK(ot_buffer_h2d(ctx, bmb, mb.data(), mb.size() * 2u) == OT_OK);

    const ot_freetile_gemm_desc_t d256 = ot_freetile_gemm_desc_default(BM, BK, BN);
    OT_CHECK(ot_matmul_bf16_ex(ctx, bma, bmb, bmc, &d256, nullptr) == OT_OK);
    std::vector<uint16_t> ref_256(BM * BN, 0u);
    OT_CHECK(ot_buffer_d2h(ctx, bmc, ref_256.data(), ref_256.size() * 2u) == OT_OK);

    // 6a) Reuse 256x256 tiled `bmc` as output for 128x128x128 GEMM (with `bmc` also aliased as LHS!).
    std::vector<uint16_t> ref_128(128u * 128u, 0u);
    for (uint32_t m = 0; m < 128u; ++m) {
      for (uint32_t n = 0; n < 128u; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < 128u; ++k) {
          acc += host_bf16_to_f32(ref_256[m * 128u + k]) * host_bf16_to_f32(mb[k * 128u + n]);
        }
        ref_128[m * 128u + n] = host_f32_to_bf16(acc);
      }
    }
    const ot_freetile_gemm_desc_t d128 = ot_freetile_gemm_desc_default(128u, 128u, 128u);
    OT_CHECK(ot_matmul_bf16_ex(ctx, bmc, bmb, bmc, &d128, nullptr) == OT_OK);
    std::vector<uint16_t> after_128(BM * BN, 0u);
    OT_CHECK(ot_buffer_d2h(ctx, bmc, after_128.data(), after_128.size() * 2u) == OT_OK);
    OT_CHECK(std::memcmp(after_128.data(), ref_128.data(), ref_128.size() * 2u) == 0);
    OT_CHECK(std::memcmp(after_128.data() + ref_128.size(),
                         ref_256.data() + ref_128.size(),
                         (BM * BN - ref_128.size()) * 2u) == 0);

    // 6b) Re-populate 256x256 tiled `bmc`, then overwrite prefix with non-tile-aligned 64x64x64 GEMM,
    //     and verify wall_us and hbm_bytes_moved are populated on non-tile-aligned mock stats.
    OT_CHECK(ot_matmul_bf16_ex(ctx, bma, bmb, bmc, &d256, nullptr) == OT_OK);
    const ot_freetile_gemm_desc_t d64 = ot_freetile_gemm_desc_default(64u, 64u, 64u);
    ot_exec_stats_t stats_64{};
    OT_CHECK(ot_matmul_bf16_ex(ctx, bma, bmb, bmc, &d64, &stats_64) == OT_OK);
    OT_CHECK(stats_64.wall_us > 0u);
    OT_CHECK(stats_64.hbm_bytes_moved == 3ULL * 64ULL * 64ULL * sizeof(uint16_t));
    std::vector<uint16_t> after_64(BM * BN, 0u);
    OT_CHECK(ot_buffer_d2h(ctx, bmc, after_64.data(), after_64.size() * 2u) == OT_OK);
    OT_CHECK(std::memcmp(after_64.data() + (64u * 64u),
                         ref_256.data() + (64u * 64u),
                         (BM * BN - 64u * 64u) * 2u) == 0);

    // 6c) Feed a tiled 256x128 buffer as mismatched-shape 128x256 LHS (and aliased output) to ot_matmul_quant.
    const ot_freetile_gemm_desc_t d256x128 = ot_freetile_gemm_desc_default(256u, 128u, 128u);
    OT_CHECK(ot_matmul_bf16_ex(ctx, bma, bmb, bmc, &d256x128, nullptr) == OT_OK);
    std::vector<uint16_t> lhs_logical(128u * 256u, 0u);
    OT_CHECK(ot_buffer_d2h(ctx, bmc, lhs_logical.data(), lhs_logical.size() * 2u) == OT_OK);
    std::vector<uint16_t> ref_q_alias(QM * QN, 0u);
    for (uint32_t m = 0; m < QM; ++m) {
      for (uint32_t n = 0; n < QN; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < QK; ++k) {
          const float sc = host_bf16_to_f32(host_f32_to_bf16(sc_blk[(k / 32u) * QN + n]));
          const float wf = host_bf16_to_f32(host_f32_to_bf16(static_cast<float>(w_i8[k * QN + n]) * sc));
          acc += host_bf16_to_f32(lhs_logical[m * QK + k]) * wf;
        }
        ref_q_alias[m * QN + n] = host_f32_to_bf16(acc);
      }
    }
    OT_CHECK(ot_matmul_quant(ctx, bmc, bqi8, bqsc, bmc, &q8d, nullptr) == OT_OK);
    std::vector<uint16_t> got_q_alias(QM * QN, 0u);
    OT_CHECK(ot_buffer_d2h(ctx, bmc, got_q_alias.data(), got_q_alias.size() * 2u) == OT_OK);
    OT_CHECK(got_q_alias == ref_q_alias);

    ot_buffer_free(bmc);
    ot_buffer_free(bmb);
    ot_buffer_free(bma);
  }

  ot_buffer_free(bqout);
  ot_buffer_free(bqsc);
  ot_buffer_free(bqq4);
  ot_buffer_free(bqi8);
  ot_buffer_free(bqa);
  ot_context_destroy(ctx);
  printf("[PASS] test_mock_multitile_and_quant_matmul\n");
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
  test_freetile_descriptor_and_planner();
  test_freetile_vliw_emitters_and_packing();
  test_mock_multitile_and_quant_matmul();
  printf("ALL_HOST_TESTS_PASSED\n");
  return 0;
}
