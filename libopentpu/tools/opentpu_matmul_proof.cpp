// SPDX-License-Identifier: Apache-2.0
// Standalone clean-room live BF16 128x128 matmul proof binary for libopentpu.so.
// Exercises BOTH:
//   (A) C API ot_matmul_bf16 across >=3 consecutive random 128x128 BF16 runs (+ chained run)
//   (B) dlopen("./libopentpu.so") -> GetPjrtApi() -> BufferFromHostBuffer ->
//       PJRT_Client_Compile -> PJRT_LoadedExecutable_Execute -> ToHostBuffer
//       across >=3 consecutive random 128x128 BF16 runs
// Asserts 0/16384 mismatches (max_diff == 0.0), MAPS_LIBTPU=0, MAPS_LIBPYTHON=0.

#include "opentpu.h"
#include "pjrt_c_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

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

// Fill 128x128 row-major BF16 matrix with exact binary fractions in [-range, +range]
// at resolution 1/denom (power of two). Because K=128 sums of products stay well under
// FP32's 24-bit mantissa, FP32 accumulation is 100% associative-exact while still
// exercising sign, exponent, mantissa, and RNE BF16 rounding across all 16,384 outputs.
void fill_random_bf16(std::vector<uint16_t>* dst, uint64_t seed, int half_steps, float scale) {
  const uint64_t span = static_cast<uint64_t>(2 * half_steps + 1);
  uint64_t rng = seed;
  for (uint16_t& v : *dst) {
    int q = static_cast<int>(splitmix64(&rng) % span) - half_steps;
    float f = static_cast<float>(q) * scale;
    v = f32_to_bf16(f);
  }
}

void ref_matmul_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                     std::vector<uint16_t>* out, uint32_t m, uint32_t k, uint32_t n) {
  out->assign(static_cast<size_t>(m) * n, 0);
  for (uint32_t r = 0; r < m; ++r) {
    for (uint32_t c = 0; c < n; ++c) {
      float acc = 0.0f;
      for (uint32_t j = 0; j < k; ++j) {
        acc += bf16_to_f32(a[static_cast<size_t>(r) * k + j]) *
               bf16_to_f32(b[static_cast<size_t>(j) * n + c]);
      }
      (*out)[static_cast<size_t>(r) * n + c] = f32_to_bf16(acc);
    }
  }
}

struct CompareResult {
  size_t mismatches = 0;
  float max_diff = 0.0f;
  size_t first_bad_idx = 0;
  uint16_t first_got = 0;
  uint16_t first_ref = 0;
};

CompareResult compare_bf16(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  CompareResult res{};
  const size_t n = std::min(got.size(), ref.size());
  for (size_t i = 0; i < n; ++i) {
    float fg = bf16_to_f32(got[i]);
    float fr = bf16_to_f32(ref[i]);
    float diff = std::fabs(fg - fr);
    if (diff > res.max_diff) res.max_diff = diff;
    if (got[i] != ref[i] || diff != 0.0f) {
      if (res.mismatches == 0) {
        res.first_bad_idx = i;
        res.first_got = got[i];
        res.first_ref = ref[i];
      }
      ++res.mismatches;
    }
  }
  return res;
}

template <typename Fn>
bool load_sym(void* handle, const char* name, Fn* out_fn) {
  dlerror();
  void* sym = dlsym(handle, name);
  const char* err = dlerror();
  if (err != nullptr || sym == nullptr) {
    fprintf(stderr, "dlsym(%s) failed: %s\n", name, err ? err : "null symbol");
    return false;
  }
  std::memcpy(out_fn, &sym, sizeof(*out_fn));
  return true;
}

std::string executable_dir(const char* argv0) {
#ifdef __APPLE__
  uint32_t sz = 0;
  _NSGetExecutablePath(nullptr, &sz);
  if (sz > 0) {
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(&exe[0], &sz) == 0) {
      size_t nul = exe.find('\0');
      if (nul != std::string::npos) exe.resize(nul);
      auto slash = exe.rfind('/');
      if (slash != std::string::npos) return exe.substr(0, slash + 1);
    }
  }
#else
  char buf[4096];
  ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) {
    std::string exe(buf, static_cast<size_t>(n));
    auto slash = exe.rfind('/');
    if (slash != std::string::npos) return exe.substr(0, slash + 1);
  }
#endif
  if (argv0) {
    std::string exe(argv0);
    auto slash = exe.rfind('/');
    if (slash != std::string::npos) return exe.substr(0, slash + 1);
  }
  return "";
}

void* open_opentpu_library(const std::string& explicit_path, const char* argv0,
                           std::string* resolved_out) {
  std::string path;
  if (!explicit_path.empty()) {
    path = explicit_path;
  } else {
#ifdef __APPLE__
    const char* lib_name = "libopentpu.dylib";
#else
    const char* lib_name = "libopentpu.so";
#endif
    std::string dir = executable_dir(argv0);
    if (dir.empty()) {
      fprintf(stderr, "Unable to determine executable directory; pass --lib <path>\n");
      return nullptr;
    }
    path = dir + lib_name;
  }

  void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (h != nullptr) {
    if (resolved_out) *resolved_out = path;
    return h;
  }
  fprintf(stderr, "dlopen(%s) failed: %s\n", path.c_str(), dlerror());
  return nullptr;
}

std::string mapped_basename(const std::string& maps_line) {
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

bool check_maps_clean(const char* tag, bool mock_mode) {
  std::ifstream ifs("/proc/self/maps");
  if (!ifs.is_open()) {
    if (!mock_mode) {
      fprintf(stderr,
              "[%s] ERROR: /proc/self/maps unreadable in live mode; cannot verify MAPS_LIBTPU=0 MAPS_LIBPYTHON=0\n",
              tag);
      return false;
    }
    printf("[%s] MAPS_CHECK=SKIPPED (/proc/self/maps unavailable on this host)\n", tag);
    return true;
  }
  std::string line;
  int libtpu_cnt = 0;
  int libpython_cnt = 0;
  while (std::getline(ifs, line)) {
    std::string base = mapped_basename(line);
    if (base.compare(0, 6, "libtpu") == 0) ++libtpu_cnt;
    if (base.compare(0, 9, "libpython") == 0) ++libpython_cnt;
  }
  printf("[%s] MAPS_LIBTPU=%d MAPS_LIBPYTHON=%d\n", tag, libtpu_cnt, libpython_cnt);
  return (libtpu_cnt == 0 && libpython_cnt == 0);
}

void print_pjrt_error(const PJRT_Api* api, const char* step, PJRT_Error* err) {
  if (!err) return;
  PJRT_Error_Message_Args ma = {};
  ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  ma.error = err;
  api->PJRT_Error_Message(&ma);
  fprintf(stderr, "PJRT %s failed: %.*s\n", step,
          static_cast<int>(ma.message_size), ma.message ? ma.message : "");
  PJRT_Error_Destroy_Args da = {};
  da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  da.error = err;
  api->PJRT_Error_Destroy(&da);
}

bool run_c_api_proof(void* lib, bool mock_mode, int num_runs) {
  using fn_config_default_t  = ot_config_t (*)(void);
  using fn_context_create_t  = ot_status_t (*)(const ot_config_t*, ot_context_t**);
  using fn_context_destroy_t = void (*)(ot_context_t*);
  using fn_status_string_t   = const char* (*)(ot_status_t);
  using fn_last_error_t      = const char* (*)(const ot_context_t*);
  using fn_device_get_info_t = ot_status_t (*)(const ot_context_t*, ot_device_info_t*);
  using fn_buffer_alloc_t    = ot_status_t (*)(ot_context_t*, size_t, ot_buffer_t**);
  using fn_buffer_free_t     = void (*)(ot_buffer_t*);
  using fn_buffer_h2d_t      = ot_status_t (*)(ot_context_t*, ot_buffer_t*, const void*, size_t);
  using fn_buffer_d2h_t      = ot_status_t (*)(ot_context_t*, const ot_buffer_t*, void*, size_t);
  using fn_matmul_bf16_t     = ot_status_t (*)(ot_context_t*, const ot_buffer_t*,
                                               const ot_buffer_t*, ot_buffer_t*,
                                               uint32_t, uint32_t, uint32_t);

  fn_config_default_t  p_config_default  = nullptr;
  fn_context_create_t  p_context_create  = nullptr;
  fn_context_destroy_t p_context_destroy = nullptr;
  fn_status_string_t   p_status_string   = nullptr;
  fn_last_error_t      p_last_error      = nullptr;
  fn_device_get_info_t p_device_get_info = nullptr;
  fn_buffer_alloc_t    p_buffer_alloc    = nullptr;
  fn_buffer_free_t     p_buffer_free     = nullptr;
  fn_buffer_h2d_t      p_buffer_h2d      = nullptr;
  fn_buffer_d2h_t      p_buffer_d2h      = nullptr;
  fn_matmul_bf16_t     p_matmul_bf16     = nullptr;

  if (!load_sym(lib, "ot_config_default", &p_config_default) ||
      !load_sym(lib, "ot_context_create", &p_context_create) ||
      !load_sym(lib, "ot_context_destroy", &p_context_destroy) ||
      !load_sym(lib, "ot_status_string", &p_status_string) ||
      !load_sym(lib, "ot_context_last_error", &p_last_error) ||
      !load_sym(lib, "ot_device_get_info", &p_device_get_info) ||
      !load_sym(lib, "ot_buffer_alloc", &p_buffer_alloc) ||
      !load_sym(lib, "ot_buffer_free", &p_buffer_free) ||
      !load_sym(lib, "ot_buffer_h2d", &p_buffer_h2d) ||
      !load_sym(lib, "ot_buffer_d2h", &p_buffer_d2h) ||
      !load_sym(lib, "ot_matmul_bf16", &p_matmul_bf16)) {
    return false;
  }

  ot_config_t cfg = p_config_default();
  cfg.mock_mode = mock_mode ? 1 : 0;

  ot_context_t* ctx = nullptr;
  ot_status_t st = p_context_create(&cfg, &ctx);
  if (st != OT_OK || !ctx) {
    fprintf(stderr, "[C-API] ot_context_create failed: %s\n", p_status_string(st));
    return false;
  }

  ot_device_info_t info = {};
  p_device_get_info(ctx, &info);
  printf("[C-API] device bdf=%s vid=0x%04x did=0x%04x mock=%u prebooted=%u\n",
         info.bdf, info.vendor_id, info.device_id, info.is_mock, info.is_prebooted);

  const uint32_t M = 128u, K = 128u, N = 128u;
  const size_t nbytes = static_cast<size_t>(M) * N * sizeof(uint16_t);
  ot_buffer_t* ba = nullptr;
  ot_buffer_t* bb = nullptr;
  ot_buffer_t* bc = nullptr;
  ot_buffer_t* bd = nullptr;
  ot_buffer_t* be = nullptr;

  auto cleanup = [&]() {
    p_buffer_free(be);
    p_buffer_free(bd);
    p_buffer_free(bc);
    p_buffer_free(bb);
    p_buffer_free(ba);
    p_context_destroy(ctx);
  };

  if (p_buffer_alloc(ctx, nbytes, &ba) != OT_OK ||
      p_buffer_alloc(ctx, nbytes, &bb) != OT_OK ||
      p_buffer_alloc(ctx, nbytes, &bc) != OT_OK ||
      p_buffer_alloc(ctx, nbytes, &bd) != OT_OK ||
      p_buffer_alloc(ctx, nbytes, &be) != OT_OK) {
    fprintf(stderr, "[C-API] ot_buffer_alloc failed: %s\n", p_last_error(ctx));
    cleanup();
    return false;
  }

  std::vector<uint16_t> host_a(M * K);
  std::vector<uint16_t> host_b(K * N);
  std::vector<uint16_t> host_c(M * N);
  std::vector<uint16_t> ref_c(M * N);

  bool ok = true;
  for (int run = 0; run < num_runs; ++run) {
    fill_random_bf16(&host_a, 0x10001ULL + static_cast<uint64_t>(run) * 97ULL, 16, 0.125f);
    fill_random_bf16(&host_b, 0x20003ULL + static_cast<uint64_t>(run) * 131ULL, 16, 0.125f);
    ref_matmul_bf16(host_a, host_b, &ref_c, M, K, N);

    auto t0 = std::chrono::steady_clock::now();
    st = p_buffer_h2d(ctx, ba, host_a.data(), nbytes);
    if (st == OT_OK) st = p_buffer_h2d(ctx, bb, host_b.data(), nbytes);
    if (st == OT_OK) st = p_matmul_bf16(ctx, ba, bb, bc, M, K, N);
    if (st == OT_OK) st = p_buffer_d2h(ctx, bc, host_c.data(), nbytes);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    if (st != OT_OK) {
      fprintf(stderr, "[C-API] run %d failed: %s (%s)\n",
              run + 1, p_status_string(st), p_last_error(ctx));
      ok = false;
      break;
    }

    CompareResult cmp = compare_bf16(host_c, ref_c);
    printf("[C-API] run %d/%d: mismatches=%zu/16384 max_diff=%.6f wall_us=%.1f\n",
           run + 1, num_runs, cmp.mismatches, cmp.max_diff, us);
    if (cmp.mismatches != 0 || cmp.max_diff != 0.0f) {
      fprintf(stderr, "[C-API] run %d mismatch at idx=%zu got=0x%04x ref=0x%04x\n",
              run + 1, cmp.first_bad_idx, cmp.first_got, cmp.first_ref);
      ok = false;
      break;
    }
  }

  // Chained device-resident matmul check: E = (A @ B) @ D without re-uploading C.
  if (ok) {
    std::vector<uint16_t> host_d(K * N);
    std::vector<uint16_t> host_e(M * N);
    std::vector<uint16_t> ref_ab(M * N);
    std::vector<uint16_t> ref_e(M * N);
    fill_random_bf16(&host_a, 0x55551ULL, 2, 0.25f);
    fill_random_bf16(&host_b, 0x66663ULL, 2, 0.25f);
    fill_random_bf16(&host_d, 0x77779ULL, 2, 0.25f);
    ref_matmul_bf16(host_a, host_b, &ref_ab, M, K, N);
    ref_matmul_bf16(ref_ab, host_d, &ref_e, M, K, N);

    st = p_buffer_h2d(ctx, ba, host_a.data(), nbytes);
    if (st == OT_OK) st = p_buffer_h2d(ctx, bb, host_b.data(), nbytes);
    if (st == OT_OK) st = p_buffer_h2d(ctx, bd, host_d.data(), nbytes);
    if (st == OT_OK) st = p_matmul_bf16(ctx, ba, bb, bc, M, K, N);
    if (st == OT_OK) st = p_matmul_bf16(ctx, bc, bd, be, M, K, N);
    if (st == OT_OK) st = p_buffer_d2h(ctx, be, host_e.data(), nbytes);
    if (st != OT_OK) {
      fprintf(stderr, "[C-API] chained run failed: %s (%s)\n",
              p_status_string(st), p_last_error(ctx));
      ok = false;
    } else {
      CompareResult cmp = compare_bf16(host_e, ref_e);
      printf("[C-API] chained (A@B)@D: mismatches=%zu/16384 max_diff=%.6f\n",
             cmp.mismatches, cmp.max_diff);
      if (cmp.mismatches != 0 || cmp.max_diff != 0.0f) {
        fprintf(stderr, "[C-API] chained mismatch at idx=%zu got=0x%04x ref=0x%04x\n",
                cmp.first_bad_idx, cmp.first_got, cmp.first_ref);
        ok = false;
      }
    }
  }

  bool maps_ok = check_maps_clean("C_API_ACTIVE", mock_mode);
  cleanup();
  return ok && maps_ok;
}

bool run_pjrt_api_proof(void* lib, bool mock_mode, int num_runs) {
  if (mock_mode) {
    setenv("OPENTPU_MOCK", "1", 1);
  } else {
    unsetenv("OPENTPU_MOCK");
  }

  using fn_get_pjrt_api_t = const PJRT_Api* (*)(void);
  fn_get_pjrt_api_t p_get_pjrt_api = nullptr;
  if (!load_sym(lib, "GetPjrtApi", &p_get_pjrt_api)) {
    return false;
  }

  const PJRT_Api* api = p_get_pjrt_api();
  if (!api) {
    fprintf(stderr, "[PJRT] GetPjrtApi() returned null\n");
    return false;
  }
  printf("[PJRT] GetPjrtApi() version=%d.%d struct_size=%zu\n",
         api->pjrt_api_version.major_version,
         api->pjrt_api_version.minor_version,
         api->struct_size);

  PJRT_Plugin_Initialize_Args pi = {};
  pi.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  if (PJRT_Error* e = api->PJRT_Plugin_Initialize(&pi)) {
    print_pjrt_error(api, "Plugin_Initialize", e);
    return false;
  }

  PJRT_Client_Create_Args cc = {};
  cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  if (PJRT_Error* e = api->PJRT_Client_Create(&cc)) {
    print_pjrt_error(api, "Client_Create", e);
    return false;
  }
  PJRT_Client* client = cc.client;
  PJRT_LoadedExecutable* exec = nullptr;

  auto cleanup_client = [&]() {
    if (exec) {
      PJRT_LoadedExecutable_Destroy_Args led = {};
      led.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
      led.executable = exec;
      if (PJRT_Error* e = api->PJRT_LoadedExecutable_Destroy(&led)) {
        print_pjrt_error(api, "LoadedExecutable_Destroy", e);
      }
    }
    if (client) {
      PJRT_Client_Destroy_Args cd = {};
      cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
      cd.client = client;
      if (PJRT_Error* e = api->PJRT_Client_Destroy(&cd)) {
        print_pjrt_error(api, "Client_Destroy", e);
      }
    }
  };

  PJRT_Client_PlatformName_Args pn = {};
  pn.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  pn.client = client;
  if (PJRT_Error* e = api->PJRT_Client_PlatformName(&pn)) {
    print_pjrt_error(api, "Client_PlatformName", e);
    cleanup_client();
    return false;
  }

  PJRT_Client_AddressableDevices_Args ad = {};
  ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  ad.client = client;
  if (PJRT_Error* e = api->PJRT_Client_AddressableDevices(&ad)) {
    print_pjrt_error(api, "Client_AddressableDevices", e);
    cleanup_client();
    return false;
  }
  if (ad.num_addressable_devices < 1) {
    fprintf(stderr, "[PJRT] no addressable devices\n");
    cleanup_client();
    return false;
  }
  printf("[PJRT] platform=%.*s devices=%zu\n",
         static_cast<int>(pn.platform_name_size), pn.platform_name,
         ad.num_addressable_devices);

  const char* mlir =
      "module @jit_mm_proof {\n"
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
  ca.client = client;
  ca.program = &prog;
  if (PJRT_Error* e = api->PJRT_Client_Compile(&ca)) {
    print_pjrt_error(api, "Client_Compile", e);
    cleanup_client();
    return false;
  }
  exec = ca.executable;

  const uint32_t M = 128u, K = 128u, N = 128u;
  const int64_t dims[2] = {128, 128};
  std::vector<uint16_t> host_a(M * K);
  std::vector<uint16_t> host_b(K * N);
  std::vector<uint16_t> host_c(M * N);
  std::vector<uint16_t> ref_c(M * N);

  auto destroy_event = [&](PJRT_Event* ev) {
    if (!ev) return;
    PJRT_Event_Destroy_Args ed = {};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = ev;
    if (PJRT_Error* e = api->PJRT_Event_Destroy(&ed)) {
      print_pjrt_error(api, "Event_Destroy", e);
    }
  };
  auto destroy_buf = [&](PJRT_Buffer* b) {
    if (!b) return;
    PJRT_Buffer_Destroy_Args bd = {};
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = b;
    if (PJRT_Error* e = api->PJRT_Buffer_Destroy(&bd)) {
      print_pjrt_error(api, "Buffer_Destroy", e);
    }
  };

  bool ok = true;
  for (int run = 0; run < num_runs; ++run) {
    fill_random_bf16(&host_a, 0x30007ULL + static_cast<uint64_t>(run) * 193ULL, 16, 0.125f);
    fill_random_bf16(&host_b, 0x40009ULL + static_cast<uint64_t>(run) * 257ULL, 16, 0.125f);
    ref_matmul_bf16(host_a, host_b, &ref_c, M, K, N);

    PJRT_Buffer* ba = nullptr;
    PJRT_Buffer* bb = nullptr;
    PJRT_Buffer* outs[1] = {nullptr};

    auto upload = [&](const void* data, PJRT_Buffer** out_b) -> bool {
      static const int64_t kMinorToMajor[2] = {1, 0};
      static const int64_t kTileDims[4] = {2, 128, 2, 1};
      static const size_t kTileDimSizes[2] = {2, 2};
      PJRT_Buffer_MemoryLayout native_layout = {};
      native_layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
      native_layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
      native_layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
      native_layout.tiled.minor_to_major = kMinorToMajor;
      native_layout.tiled.minor_to_major_size = 2;
      native_layout.tiled.tile_dims = kTileDims;
      native_layout.tiled.tile_dim_sizes = kTileDimSizes;
      native_layout.tiled.num_tiles = 2;

      PJRT_Client_BufferFromHostBuffer_Args a = {};
      a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
      a.client = client;
      a.data = data;
      a.type = PJRT_Buffer_Type_BF16;
      a.dims = dims;
      a.num_dims = 2;
      a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
      a.device = ad.addressable_devices[0];
      a.device_layout = &native_layout;
      if (PJRT_Error* e = api->PJRT_Client_BufferFromHostBuffer(&a)) {
        print_pjrt_error(api, "BufferFromHostBuffer", e);
        return false;
      }
      destroy_event(a.done_with_host_buffer);
      *out_b = a.buffer;
      return true;
    };

    auto t0 = std::chrono::steady_clock::now();
    if (!upload(host_a.data(), &ba) || !upload(host_b.data(), &bb)) {
      destroy_buf(bb);
      destroy_buf(ba);
      ok = false;
      break;
    }

    PJRT_Buffer* ins[2] = {ba, bb};
    PJRT_Buffer* const* arg_lists[1] = {ins};
    PJRT_Buffer** const out_lists[1] = {outs};
    PJRT_ExecuteOptions opts = {};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    PJRT_Event* dev_ev[1] = {nullptr};

    PJRT_LoadedExecutable_Execute_Args ex = {};
    ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex.executable = exec;
    ex.options = &opts;
    ex.argument_lists = arg_lists;
    ex.num_devices = 1;
    ex.num_args = 2;
    ex.output_lists = out_lists;
    ex.device_complete_events = dev_ev;
    if (PJRT_Error* e = api->PJRT_LoadedExecutable_Execute(&ex)) {
      print_pjrt_error(api, "LoadedExecutable_Execute", e);
      destroy_buf(bb);
      destroy_buf(ba);
      ok = false;
      break;
    }
    destroy_event(dev_ev[0]);

    std::fill(host_c.begin(), host_c.end(), 0u);
    PJRT_Buffer_ToHostBuffer_Args th = {};
    th.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    th.src = outs[0];
    th.dst = host_c.data();
    th.dst_size = host_c.size() * sizeof(uint16_t);
    if (PJRT_Error* e = api->PJRT_Buffer_ToHostBuffer(&th)) {
      print_pjrt_error(api, "Buffer_ToHostBuffer", e);
      destroy_buf(outs[0]);
      destroy_buf(bb);
      destroy_buf(ba);
      ok = false;
      break;
    }
    destroy_event(th.event);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    destroy_buf(outs[0]);
    destroy_buf(bb);
    destroy_buf(ba);

    CompareResult cmp = compare_bf16(host_c, ref_c);
    printf("[PJRT] run %d/%d: mismatches=%zu/16384 max_diff=%.6f wall_us=%.1f\n",
           run + 1, num_runs, cmp.mismatches, cmp.max_diff, us);
    if (cmp.mismatches != 0 || cmp.max_diff != 0.0f) {
      fprintf(stderr, "[PJRT] run %d mismatch at idx=%zu got=0x%04x ref=0x%04x\n",
              run + 1, cmp.first_bad_idx, cmp.first_got, cmp.first_ref);
      ok = false;
      break;
    }
  }

  bool maps_ok = check_maps_clean("PJRT_ACTIVE", mock_mode);
  cleanup_client();
  return ok && maps_ok;
}

}  // namespace

int main(int argc, char** argv) {
  bool mock_mode = false;
  int num_runs = 3;
  std::string lib_path;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      printf("Usage: %s [--mock] [--lib <path>] [--runs <N>=3] [--help]\n", argv[0]);
      return 0;
    } else if (std::strcmp(argv[i], "--mock") == 0) {
      mock_mode = true;
    } else if (std::strcmp(argv[i], "--lib") == 0 && i + 1 < argc) {
      lib_path = argv[++i];
    } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
      num_runs = std::atoi(argv[++i]);
      if (num_runs < 3) num_runs = 3;
    } else {
      fprintf(stderr, "Usage: %s [--mock] [--lib <path>] [--runs <N>=3] [--help]\n", argv[0]);
      return 1;
    }
  }

  std::string resolved;
  void* lib_a = open_opentpu_library(lib_path, argv[0], &resolved);
  if (!lib_a) return 2;
  printf("Loaded libopentpu via dlopen: %s\n", resolved.c_str());

  bool c_ok = run_c_api_proof(lib_a, mock_mode, num_runs);
  bool maps_a_ok = check_maps_clean("C_API_DONE", mock_mode);
  dlclose(lib_a);
  if (!c_ok || !maps_a_ok) return 3;

  // Fresh dlopen for Part B: GetPjrtApi() end-to-end proof.
  void* lib_b = open_opentpu_library(resolved, argv[0], nullptr);
  if (!lib_b) return 4;
  bool pjrt_ok = run_pjrt_api_proof(lib_b, mock_mode, num_runs);
  dlclose(lib_b);
  if (!pjrt_ok) return 5;

  if (!check_maps_clean("FINAL", mock_mode)) return 6;
  printf("OPENTPU_MATMUL_PROOF_PASSED (mock=%d runs=%d)\n", mock_mode ? 1 : 0, num_runs);
  return 0;
}
