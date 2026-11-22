// SPDX-License-Identifier: Apache-2.0
#include "opentpu.h"
#include "ot_internal.h"
#include "ot_vliw.h"
#include "pjrt_c_api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" const PJRT_Api* GetPjrtApi(void);

#define OT_CHECK(cond)                                                               \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::fprintf(stderr, "CHECK FAILED (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
      std::abort();                                                                  \
    }                                                                                \
  } while (0)

namespace {

inline float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline uint16_t f32_to_bf16(float f) {
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
    v = f32_to_bf16(static_cast<float>(q) * scale);
  }
}

std::vector<uint16_t> ref_matmul(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                                 uint32_t m, uint32_t k, uint32_t n) {
  std::vector<uint16_t> out(static_cast<size_t>(m) * n, 0);
  for (uint32_t r = 0; r < m; ++r) {
    for (uint32_t c = 0; c < n; ++c) {
      float acc = 0.0f;
      for (uint32_t j = 0; j < k; ++j) {
        acc += bf16_to_f32(a[static_cast<size_t>(r) * k + j]) *
               bf16_to_f32(b[static_cast<size_t>(j) * n + c]);
      }
      out[static_cast<size_t>(r) * n + c] = f32_to_bf16(acc);
    }
  }
  return out;
}

void consume_expected_err(const PJRT_Api* api, PJRT_Error* err, PJRT_Error_Code expected_code) {
  OT_CHECK(err != nullptr);
  PJRT_Error_GetCode_Args gc = {};
  gc.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  gc.error = err;
  OT_CHECK(api->PJRT_Error_GetCode(&gc) == nullptr);
  OT_CHECK(gc.code == expected_code);
  PJRT_Error_Destroy_Args ed = {};
  ed.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  ed.error = err;
  api->PJRT_Error_Destroy(&ed);
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
  printf("[PASS] test_iova_allocator\n");
}

static void test_bar4_descriptor_parser() {
  std::vector<uint8_t> page(4096, 0);
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
  printf("[PASS] test_bar4_descriptor_parser\n");
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

  ot_buffer_free(bc);
  ot_buffer_free(bb);
  ot_buffer_free(ba);
  ot_context_destroy(ctx);
  printf("[PASS] test_mock_dma_and_matmul\n");
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

  // Cross-context rejection.
  std::vector<uint16_t> a(M * K), b(K * N), d(K * N), got(M * N);
  fill_random_bf16(&a, 0x1111ULL, 16, 0.125f);
  fill_random_bf16(&b, 0x2222ULL, 16, 0.125f);
  fill_random_bf16(&d, 0x3333ULL, 2, 0.25f);
  OT_CHECK(ot_buffer_h2d(ctx, foreign, a.data(), nbytes) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_buffer_d2h(ctx, foreign, got.data(), nbytes) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_matmul_bf16(ctx, foreign, bb, bc, M, K, N) == OT_ERR_INVALID_ARG);

  // Raw H2D -> 128x128 matmul writes {1,0:T(2,128)} tiled HBM -> D2H returns exact row-major.
  std::vector<uint16_t> ref_ab = ref_matmul(a, b, M, K, N);
  OT_CHECK(ot_buffer_h2d(ctx, ba, a.data(), nbytes) == OT_OK);
  OT_CHECK(ot_buffer_h2d(ctx, bb, b.data(), nbytes) == OT_OK);
  OT_CHECK(ba->layout == ot::BufferLayout::kRaw);
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, bc, M, K, N) == OT_OK);
  OT_CHECK(bc->layout == ot::BufferLayout::kTiled2x128Bf16);

  // Verify on-device mock_hbm actually stores {1,0:T(2,128)} tiled layout!
  std::vector<uint16_t> expected_tiled(M * N);
  ot::tile_pack_128x128_bf16(ref_ab.data(), expected_tiled.data());
  OT_CHECK(std::memcmp(ctx->transport.mock_hbm.data() + bc->hbm_addr,
                       expected_tiled.data(), nbytes) == 0);

  // Full & partial D2H from tiled buffer return exact row-major bytes.
  OT_CHECK(ot_buffer_d2h(ctx, bc, got.data(), nbytes) == OT_OK);
  OT_CHECK(got == ref_ab);
  std::vector<uint16_t> partial(150, 0);
  OT_CHECK(ot_buffer_d2h(ctx, bc, partial.data(), partial.size() * 2) == OT_OK);
  OT_CHECK(std::memcmp(partial.data(), ref_ab.data(), partial.size() * 2) == 0);

  // Chained matmul using tiled output bc directly as LHS input: (A @ B) @ D.
  std::vector<uint16_t> ref_abd = ref_matmul(ref_ab, d, M, K, N);
  OT_CHECK(ot::buffer_h2d_tiled_128x128_bf16(ctx, bd, d.data()) == OT_OK);
  OT_CHECK(bd->layout == ot::BufferLayout::kTiled2x128Bf16);
  OT_CHECK(ot_matmul_bf16(ctx, bc, bd, be, M, K, N) == OT_OK);
  OT_CHECK(ot_buffer_d2h(ctx, be, got.data(), nbytes) == OT_OK);
  OT_CHECK(got == ref_abd);

  // Partial H2D overwrite on a tiled buffer preserves trailing logical row-major bytes.
  std::vector<uint16_t> patch(64, 0x4000u);  // 2.0 bf16
  OT_CHECK(ot_buffer_h2d(ctx, bc, patch.data(), patch.size() * 2) == OT_OK);
  OT_CHECK(bc->layout == ot::BufferLayout::kRaw);
  OT_CHECK(ot_buffer_d2h(ctx, bc, got.data(), nbytes) == OT_OK);
  OT_CHECK(std::memcmp(got.data(), patch.data(), patch.size() * 2) == 0);
  OT_CHECK(std::memcmp(got.data() + 64, ref_ab.data() + 64, (M * N - 64) * 2) == 0);

  // In-place aliasing: A = A @ A.
  std::vector<uint16_t> ref_aa = ref_matmul(a, a, M, K, N);
  OT_CHECK(ot_buffer_h2d(ctx, ba, a.data(), nbytes) == OT_OK);
  OT_CHECK(ot_matmul_bf16(ctx, ba, ba, ba, M, K, N) == OT_OK);
  OT_CHECK(ot_buffer_d2h(ctx, ba, got.data(), nbytes) == OT_OK);
  OT_CHECK(got == ref_aa);

  // Non-128 shape works cleanly in mock mode (kRaw output).
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
  // Idempotent second call succeeds without side effects.
  OT_CHECK(ctx->transport.ensure_imem_booted(1000u, &err));

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

static void test_pjrt_api_mock() {
  setenv("OPENTPU_MOCK", "1", 1);
  const PJRT_Api* api = GetPjrtApi();
  OT_CHECK(api != nullptr);
  OT_CHECK(api->struct_size == PJRT_Api_STRUCT_SIZE);
  OT_CHECK(api->pjrt_api_version.major_version == 0);
  OT_CHECK(api->pjrt_api_version.minor_version == 116);

  PJRT_Plugin_Initialize_Args pi = {};
  pi.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  OT_CHECK(api->PJRT_Plugin_Initialize(&pi) == nullptr);

  PJRT_Client_Create_Args cc = {};
  cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  OT_CHECK(api->PJRT_Client_Create(&cc) == nullptr);
  OT_CHECK(cc.client != nullptr);

  PJRT_Client_AddressableDevices_Args ad = {};
  ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  ad.client = cc.client;
  OT_CHECK(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
  OT_CHECK(ad.num_addressable_devices == 1);

  // Reject malformed / 3D / wrong-contracting-dim StableHLO programs cleanly.
  auto check_compile_reject = [&](const char* bad_mlir) {
    PJRT_Program p = {};
    p.struct_size = PJRT_Program_STRUCT_SIZE;
    p.code = const_cast<char*>(bad_mlir);
    p.code_size = std::strlen(bad_mlir);
    p.format = "mlir";
    p.format_size = 4;
    PJRT_Client_Compile_Args ca = {};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client = cc.client;
    ca.program = &p;
    consume_expected_err(api, api->PJRT_Client_Compile(&ca), PJRT_Error_Code_UNIMPLEMENTED);
  };
  check_compile_reject(
      "func.func @main(%a: tensor<2x128x128xbf16>, %b: tensor<128x128xbf16>) { stablehlo.dot_general }");
  check_compile_reject(
      "func.func @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) { "
      "stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] }");
  check_compile_reject(
      "func.func @main(%a: tensor<128x64xbf16>, %b: tensor<128x128xbf16>) { stablehlo.dot_general }");

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
  OT_CHECK(api->PJRT_Client_Compile(&ca) == nullptr);
  OT_CHECK(ca.executable != nullptr);

  int64_t dims[2] = {128, 128};
  auto upload = [&](const void* data, PJRT_Buffer_Type dtype, const int64_t* d,
                    size_t nd) -> PJRT_Buffer* {
    PJRT_Client_BufferFromHostBuffer_Args a = {};
    a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    a.client = cc.client;
    a.data = data;
    a.type = dtype;
    a.dims = d;
    a.num_dims = nd;
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

  // Reject non-contiguous byte_strides cleanly.
  {
    std::vector<uint16_t> tmp(128 * 128, 0x3f80u);
    int64_t bad_strides[2] = {2, 256};  // column-major instead of row-major
    PJRT_Client_BufferFromHostBuffer_Args a = {};
    a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    a.client = cc.client;
    a.data = tmp.data();
    a.type = PJRT_Buffer_Type_BF16;
    a.dims = dims;
    a.num_dims = 2;
    a.byte_strides = bad_strides;
    a.num_byte_strides = 2;
    a.device = ad.addressable_devices[0];
    consume_expected_err(api, api->PJRT_Client_BufferFromHostBuffer(&a),
                         PJRT_Error_Code_UNIMPLEMENTED);
  }

  // Random 128x128 BF16 round-trip (BufferFromHostBuffer -> ToHostBuffer without execute)
  // AND 3 consecutive random PJRT_LoadedExecutable_Execute runs.
  std::vector<uint16_t> ha(128 * 128), hb(128 * 128), out_h(128 * 128);
  for (int run = 0; run < 3; ++run) {
    fill_random_bf16(&ha, 0x9001ULL + static_cast<uint64_t>(run) * 31u, 32, 1.0f / 16.0f);
    fill_random_bf16(&hb, 0xa003ULL + static_cast<uint64_t>(run) * 47u, 32, 1.0f / 16.0f);
    std::vector<uint16_t> ref = ref_matmul(ha, hb, 128u, 128u, 128u);

    PJRT_Buffer* ba = upload(ha.data(), PJRT_Buffer_Type_BF16, dims, 2);
    PJRT_Buffer* bb = upload(hb.data(), PJRT_Buffer_Type_BF16, dims, 2);

    if (run == 0) {
      // Verify direct ToHostBuffer round-trip on tiled 128x128 BF16 buffer.
      std::vector<uint16_t> rt(128 * 128, 0);
      PJRT_Buffer_ToHostBuffer_Args th0 = {};
      th0.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
      th0.src = ba;
      th0.dst = rt.data();
      th0.dst_size = rt.size() * 2;
      OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&th0) == nullptr);
      PJRT_Event_Destroy_Args ed = {};
      ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      ed.event = th0.event;
      OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
      OT_CHECK(rt == ha);

      // Verify Device_MemoryStats reflects active HBM usage.
      PJRT_Device_MemoryStats_Args ms = {};
      ms.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
      ms.device = ad.addressable_devices[0];
      OT_CHECK(api->PJRT_Device_MemoryStats(&ms) == nullptr);
      OT_CHECK(ms.bytes_in_use >= 2 * 32768);
    }

    PJRT_Buffer* ins[2] = {ba, bb};
    PJRT_Buffer* outs[1] = {nullptr};
    PJRT_Buffer* const* arg_lists[1] = {ins};
    PJRT_Buffer** const out_lists[1] = {outs};
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

    // Query required size first (dst == nullptr), then fetch.
    PJRT_Buffer_ToHostBuffer_Args thq = {};
    thq.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    thq.src = outs[0];
    OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&thq) == nullptr);
    OT_CHECK(thq.dst_size == 128u * 128u * 2u);

    std::fill(out_h.begin(), out_h.end(), 0u);
    PJRT_Buffer_ToHostBuffer_Args th = {};
    th.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    th.src = outs[0];
    th.dst = out_h.data();
    th.dst_size = out_h.size() * 2;
    OT_CHECK(api->PJRT_Buffer_ToHostBuffer(&th) == nullptr);
    if (th.event) {
      PJRT_Event_Destroy_Args ed = {};
      ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      ed.event = th.event;
      OT_CHECK(api->PJRT_Event_Destroy(&ed) == nullptr);
    }
    OT_CHECK(out_h == ref);

    for (PJRT_Buffer* b : {ba, bb, outs[0]}) {
      PJRT_Buffer_Destroy_Args bd = {};
      bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      bd.buffer = b;
      OT_CHECK(api->PJRT_Buffer_Destroy(&bd) == nullptr);
    }
  }

  // Reject shape mismatch & wrong dtype in PJRT_LoadedExecutable_Execute.
  {
    int64_t dims64[2] = {64, 64};
    std::vector<uint16_t> small_data(64 * 64, 0x3f80u);
    std::vector<float> f32_data(128 * 128, 1.0f);
    PJRT_Buffer* b_small = upload(small_data.data(), PJRT_Buffer_Type_BF16, dims64, 2);
    PJRT_Buffer* b_f32 = upload(f32_data.data(), PJRT_Buffer_Type_F32, dims, 2);
    PJRT_Buffer* b_ok = upload(ha.data(), PJRT_Buffer_Type_BF16, dims, 2);

    PJRT_Buffer* outs[1] = {nullptr};
    PJRT_Buffer** const out_lists[1] = {outs};
    PJRT_ExecuteOptions opts = {};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

    PJRT_Buffer* bad_shape_ins[2] = {b_small, b_ok};
    PJRT_Buffer* const* bad_shape_lists[1] = {bad_shape_ins};
    PJRT_LoadedExecutable_Execute_Args ex = {};
    ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex.executable = ca.executable;
    ex.options = &opts;
    ex.argument_lists = bad_shape_lists;
    ex.num_devices = 1;
    ex.num_args = 2;
    ex.output_lists = out_lists;
    consume_expected_err(api, api->PJRT_LoadedExecutable_Execute(&ex),
                         PJRT_Error_Code_INVALID_ARGUMENT);

    PJRT_Buffer* bad_dtype_ins[2] = {b_f32, b_ok};
    PJRT_Buffer* const* bad_dtype_lists[1] = {bad_dtype_ins};
    ex.argument_lists = bad_dtype_lists;
    consume_expected_err(api, api->PJRT_LoadedExecutable_Execute(&ex),
                         PJRT_Error_Code_INVALID_ARGUMENT);

    for (PJRT_Buffer* b : {b_small, b_f32, b_ok}) {
      PJRT_Buffer_Destroy_Args bd = {};
      bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      bd.buffer = b;
      OT_CHECK(api->PJRT_Buffer_Destroy(&bd) == nullptr);
    }
  }

  PJRT_LoadedExecutable_Destroy_Args led = {};
  led.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  led.executable = ca.executable;
  OT_CHECK(api->PJRT_LoadedExecutable_Destroy(&led) == nullptr);

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

static void test_resident_imem_stub_structure() {
  auto bundle_eq = [](const ot::VliwBundle& a, const ot::VliwBundle& b) -> bool {
    return std::memcmp(a.w, b.w, sizeof(a.w)) == 0;
  };
  const ot::VliwBundle nop = ot::VliwBundle::nop();

  // Verify decoded VLIW ISA primitives.
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

  // Verify 120-bundle (7,680 B, 512B-aligned) direct-in-IMEM resident loop layout.
  const std::vector<ot::VliwBundle> imem = ot::emit_v5e_resident_matmul_imem();
  OT_CHECK(imem.size() == ot::kResidentImemBundles);
  OT_CHECK(imem.size() * sizeof(ot::VliwBundle) == ot::kResidentImemBytes);
  OT_CHECK((ot::kResidentImemBytes % ot::kImemDmaAlignBytes) == 0u);

  // Stage 1 (PC 0..16): signal exec_seq -> sync[0x1f9].
  OT_CHECK(imem[0].w[2] == 0x3b00060fu && imem[0].w[15] == 0x55700dc4u);
  OT_CHECK(imem[3].w[13] == 0xfffe8007u && imem[3].w[14] == 0x8e04100cu);
  OT_CHECK(imem[14].w[2] == 0x3a0a200fu && imem[14].w[13] == (ot::kSyncFlagExecDone << 14));
  OT_CHECK(bundle_eq(imem[2], nop) && bundle_eq(imem[4], nop) && bundle_eq(imem[6], nop));
  for (size_t pc = 8; pc <= 12; ++pc) OT_CHECK(bundle_eq(imem[pc], nop));
  OT_CHECK(bundle_eq(imem[15], nop) && bundle_eq(imem[16], nop));

  // Stage 2 (PC 17..20): doorbell sync[0x1fb] wait & consume.
  OT_CHECK(imem[17].w[15] == 0x55701101u);
  OT_CHECK(imem[18].w[2] == 0x3a82a30fu && imem[18].w[13] == 0x00040000u);
  OT_CHECK(imem[19].w[2] == 0x3a02a10fu && imem[19].w[13] == 0xfffc0000u);
  OT_CHECK(bundle_eq(imem[20], nop));

  // Stage 3 (PC 21..27): OtV5eLaunchDesc HBM 0x3fff00000 -> SMEM + sync[0x1ff] + fence.
  OT_CHECK(imem[24].w[13] == (ot::kSyncFlagLdescDma << 14) && imem[24].w[14] == 0x0211f000u);
  OT_CHECK(bundle_eq(imem[25], ot::VliwBundle::make_sync_wait_imm(0x1ffu, 512u)));
  OT_CHECK(bundle_eq(imem[26], ot::VliwBundle::make_sync_sub_imm(0x1ffu, 512u)));
  OT_CHECK(bundle_eq(imem[27], ot::VliwBundle::make_fence()));

  // Stage 4 (PC 28..50): SMEM loads + unconditional LHS DMA + sync[0x33] + s0=0.
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

  // Stage 5 (PC 51..108): exact embedded 58-bundle 128x128 BF16 MXU kernel body.
  const std::vector<ot::VliwBundle> body = ot::emit_v5e_matmul_128x128_body();
  OT_CHECK(body.size() == 58u);
  for (size_t i = 0; i < body.size(); ++i) {
    OT_CHECK(bundle_eq(imem[51u + i], body[i]));
  }

  // Stage 6 (PC 109..119): NOP + br_abs(0) + 4 delay-slot NOPs + 5 pad NOPs.
  OT_CHECK(bundle_eq(imem[109], nop));
  OT_CHECK(bundle_eq(imem[110], ot::VliwBundle::make_br_abs(0u)));
  for (size_t pc = 111; pc < ot::kResidentImemBundles; ++pc) {
    OT_CHECK(bundle_eq(imem[pc], nop));
  }
  printf("[PASS] test_resident_imem_stub_structure\n");
}

int main() {
  test_iova_allocator();
  test_bar4_descriptor_parser();
  test_mock_dma_and_matmul();
  test_layout_tiling_and_aliasing();
  test_vfio_imem_lifecycle_mock();
  test_pjrt_api_mock();
  test_vliw_and_tiling();
  test_resident_imem_stub_structure();
  printf("ALL_HOST_TESTS_PASSED\n");
  return 0;
}
