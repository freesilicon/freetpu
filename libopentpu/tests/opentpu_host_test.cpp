// SPDX-License-Identifier: Apache-2.0
#include "opentpu.h"
#include "ot_internal.h"
#include "ot_vliw.h"
#include "pjrt_c_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

  // Overlapping output and input buffers must be rejected without mutating inputs.
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, ba, M, K, N) == OT_ERR_INVALID_ARG);
  OT_CHECK(ot_matmul_bf16(ctx, ba, bb, bb, M, K, N) == OT_ERR_INVALID_ARG);
  // Identical lhs and rhs into a disjoint output remains valid.
  OT_CHECK(ot_matmul_bf16(ctx, ba, ba, bc, M, K, N) == OT_OK);

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

  // Faulted DMA context must refuse subsequent transfers and allocations.
  ctx2->transport.dma_faulted = true;
  ot_buffer_t* blocked = nullptr;
  OT_CHECK(ot_buffer_alloc(ctx2, 4096, &blocked) == OT_ERR_DEVICE);
  OT_CHECK(ot_buffer_h2d(ctx2, foreign, ones.data(), 16) == OT_ERR_DEVICE);
  OT_CHECK(ot_buffer_d2h(ctx2, foreign, out.data(), 16) == OT_ERR_DEVICE);
  ot_buffer_free(foreign);
  ot_context_destroy(ctx2);

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
    expect_compile_fail(opt_bad, sizeof(opt_bad), PJRT_Error_Code_INVALID_ARGUMENT);
  }

  // Explicit precision = [DEFAULT, DEFAULT] and line-wrapped dot_general must be accepted; HIGH precision rejected.
  {
    const char* mlir_default_prec =
        "module {\n"
        "  func.func @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {\n"
        "    %0 = stablehlo.dot_general %a, %b,\n"
        "      contracting_dims = [1] x [0],\n"
        "      precision = [DEFAULT, DEFAULT]\n"
        "      : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>\n"
        "    return %0 : tensor<128x128xbf16>\n"
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
    OT_CHECK(api->PJRT_Client_Destroy(&cd2) == nullptr);
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

int main() {
  test_iova_allocator();
  test_bar4_descriptor_parser();
  test_mock_dma_and_matmul();
  test_pjrt_api_mock();
  test_vliw_and_tiling();
  printf("ALL_HOST_TESTS_PASSED\n");
  return 0;
}
