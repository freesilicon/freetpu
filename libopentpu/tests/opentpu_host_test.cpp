// SPDX-License-Identifier: Apache-2.0
#include "opentpu.h"
#include "ot_internal.h"
#include "pjrt_c_api.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" const PJRT_Api* GetPjrtApi(void);

static void test_iova_allocator() {
  ot::IovaAllocator alloc;
  alloc.init(0x100000ULL, 0x200000ULL);
  uint64_t a = 0, b = 0, c = 0;
  assert(alloc.alloc(4096, 4096, &a) && a == 0x100000ULL);
  assert(alloc.alloc(8192, 4096, &b) && b == 0x101000ULL);
  assert(alloc.used_bytes() == 12288);
  alloc.free(a);
  assert(alloc.alloc(4096, 4096, &c) && c == 0x100000ULL);
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
  assert(ot::parse_bar4_descriptor(page.data(), page.size(), &desc));
  assert(desc.valid);
  assert(desc.proto_len == plen);
  assert(desc.header_word == hdr);
  assert(desc.core_count_observed == 1);
  assert(desc.instance_token == "TPUv5");
  printf("[PASS] test_bar4_descriptor_parser\n");
}

static void test_mock_dma_and_matmul() {
  ot_config_t cfg = ot_config_default();
  cfg.mock_mode = 1;
  ot_context_t* ctx = nullptr;
  assert(ot_context_create(&cfg, &ctx) == OT_OK);
  assert(ot_context_is_mock(ctx) == 1);

  ot_device_info_t info = {};
  assert(ot_device_get_info(ctx, &info) == OT_OK);
  assert(info.vendor_id == 0x1ae0u && info.device_id == 0x0063u);

  ot_buffer_t* ba = nullptr;
  ot_buffer_t* bb = nullptr;
  ot_buffer_t* bc = nullptr;
  const uint32_t M = 128, K = 128, N = 128;
  assert(ot_buffer_alloc(ctx, M * K * 2, &ba) == OT_OK);
  assert(ot_buffer_alloc(ctx, K * N * 2, &bb) == OT_OK);
  assert(ot_buffer_alloc(ctx, M * N * 2, &bc) == OT_OK);

  std::vector<uint16_t> ones(M * K, 0x3f80u);  // 1.0 bf16
  assert(ot_buffer_h2d(ctx, ba, ones.data(), ones.size() * 2) == OT_OK);
  assert(ot_buffer_h2d(ctx, bb, ones.data(), ones.size() * 2) == OT_OK);
  assert(ot_matmul_bf16(ctx, ba, bb, bc, M, K, N) == OT_OK);

  std::vector<uint16_t> out(M * N, 0);
  assert(ot_buffer_d2h(ctx, bc, out.data(), out.size() * 2) == OT_OK);
  for (uint16_t v : out) {
    assert(v == 0x4300u);  // 128.0 bf16
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
  assert(api != nullptr);
  assert(api->struct_size == PJRT_Api_STRUCT_SIZE);
  assert(api->pjrt_api_version.major_version == 0);
  assert(api->pjrt_api_version.minor_version == 116);

  PJRT_Plugin_Initialize_Args pi = {};
  pi.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  assert(api->PJRT_Plugin_Initialize(&pi) == nullptr);

  PJRT_Client_Create_Args cc = {};
  cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  assert(api->PJRT_Client_Create(&cc) == nullptr);
  assert(cc.client != nullptr);

  PJRT_Client_AddressableDevices_Args ad = {};
  ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  ad.client = cc.client;
  assert(api->PJRT_Client_AddressableDevices(&ad) == nullptr);
  assert(ad.num_addressable_devices == 1);

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
  assert(api->PJRT_Client_Compile(&ca) == nullptr);
  assert(ca.executable != nullptr);

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
    assert(api->PJRT_Client_BufferFromHostBuffer(&a) == nullptr);
    if (a.done_with_host_buffer) {
      PJRT_Event_Destroy_Args ed = {};
      ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      ed.event = a.done_with_host_buffer;
      assert(api->PJRT_Event_Destroy(&ed) == nullptr);
    }
    return a.buffer;
  };

  PJRT_Buffer* ba = upload(ones.data());
  PJRT_Buffer* bb = upload(ones.data());
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
  assert(api->PJRT_LoadedExecutable_Execute(&ex) == nullptr);
  assert(outs[0] != nullptr);
  if (dev_ev[0]) {
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = dev_ev[0];
    assert(api->PJRT_Event_Destroy(&ed) == nullptr);
  }

  std::vector<uint16_t> out_h(128 * 128, 0);
  PJRT_Buffer_ToHostBuffer_Args th = {};
  th.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  th.src = outs[0];
  th.dst = out_h.data();
  th.dst_size = out_h.size() * 2;
  assert(api->PJRT_Buffer_ToHostBuffer(&th) == nullptr);
  if (th.event) {
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = th.event;
    assert(api->PJRT_Event_Destroy(&ed) == nullptr);
  }
  for (uint16_t v : out_h) assert(v == 0x4300u);

  for (PJRT_Buffer* b : {ba, bb, outs[0]}) {
    PJRT_Buffer_Destroy_Args bd = {};
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = b;
    assert(api->PJRT_Buffer_Destroy(&bd) == nullptr);
  }
  PJRT_LoadedExecutable_Destroy_Args led = {};
  led.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  led.executable = ca.executable;
  assert(api->PJRT_LoadedExecutable_Destroy(&led) == nullptr);

  PJRT_Client_Destroy_Args cd = {};
  cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  cd.client = cc.client;
  assert(api->PJRT_Client_Destroy(&cd) == nullptr);
  printf("[PASS] test_pjrt_api_mock\n");
}

#include "../src/ot_vliw.h"

static void test_vliw_and_tiling() {
  ot::VliwBundle nop = ot::VliwBundle::nop();
  assert(sizeof(nop) == 64);
  assert(nop.w[15] == 0x55780000u);

  std::vector<ot::VliwBundle> prog = ot::emit_v5e_matmul_128x128_body();
  assert(prog.size() == 58u);
  assert(prog[6].w[0] == 0xaf003c00u);
  assert(prog[prog.size() - 4u].w[15] == 0x55700db1u);
  assert(prog.back().w[15] == 0x55700900u);

  ot::OtV5eLaunchDesc desc = ot::OtV5eLaunchDesc::build(
      0x00100000ULL, 0x00200000ULL, 0x00300000ULL, 0x00400000ULL, 42u);
  assert(sizeof(desc.bytes) == 512u);
  auto rd32 = [&](size_t off) {
    uint32_t v = 0;
    std::memcpy(&v, desc.bytes + off, sizeof(v));
    return v;
  };
  assert(rd32(0x170u) == static_cast<uint32_t>(0x00200000ULL >> 9));
  assert(rd32(0x174u) == static_cast<uint32_t>(0x00100000ULL >> 9));
  assert(rd32(0x17cu) == 0xffffffffu);
  assert(rd32(0x190u) == static_cast<uint32_t>(0x00300000ULL >> 9));
  assert(rd32(0x1a0u) == 0x20u);
  assert(rd32(0x1a4u) == static_cast<uint32_t>(0x00400000ULL >> 9));
  assert(rd32(0x1e4u) == static_cast<uint32_t>(0x00400000ULL >> 9));
  assert(rd32(0x1e8u) == 42u);

  std::vector<uint16_t> src(128u * 128u), tiled(128u * 128u), dst(128u * 128u);
  for (uint32_t r = 0; r < 128u; ++r) {
    for (uint32_t c = 0; c < 128u; ++c) {
      src[r * 128u + c] = static_cast<uint16_t>(r * 131u + c);
    }
  }
  ot::tile_pack_128x128_bf16(src.data(), tiled.data());
  assert(tiled[0] == src[0]);
  assert(tiled[1] == src[128]);
  assert(tiled[2] == src[1]);
  assert(tiled[255] == src[128 + 127]);
  ot::tile_unpack_128x128_bf16(tiled.data(), dst.data());
  assert(std::memcmp(src.data(), dst.data(), src.size() * sizeof(uint16_t)) == 0);
  printf("[PASS] test_vliw_and_tiling\n");
}

int main() {
  test_iova_allocator();
  test_bar4_descriptor_parser();
  test_mock_dma_and_matmul();
  test_pjrt_api_mock();
  test_vliw_and_tiling();
  printf("ALL_HOST_TESTS_PASSED\n");
  return 0;
}
