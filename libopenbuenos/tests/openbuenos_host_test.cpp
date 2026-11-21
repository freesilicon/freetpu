// Comprehensive host unit test suite for libopenbuenos (100% offline, mock_mode=1).
#include "openbuenos.h"
#include "ob_internal.h"
#include "litebuf_vii_wire.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t kSyntheticDiveInterpBytes = 65536u;

std::vector<uint8_t> make_synthetic_dive_interp(uint8_t seed = 0x5Au) {
  std::vector<uint8_t> img(kSyntheticDiveInterpBytes, 0u);
  for (size_t i = 0; i < img.size(); ++i) {
    img[i] = static_cast<uint8_t>((i * 17u + seed) & 0xFFu);
  }
  return img;
}

std::vector<uint8_t> make_synthetic_gemm_dgc0(uint32_t M,
                                              uint32_t K,
                                              uint32_t N,
                                              uint8_t interp_seed = 0x5Au) {
  std::vector<int8_t> w(static_cast<size_t>(N) * K);
  std::vector<int32_t> bias(N);
  for (uint32_t o = 0; o < N; ++o) {
    int32_t b_mod = ((static_cast<int32_t>(o) * 29 - 100) % 250 + 250) % 250;
    bias[o] = b_mod - 125;
    for (uint32_t i = 0; i < K; ++i) {
      int32_t v = static_cast<int32_t>(o * 7u + i * 11u + 3u) % 63;
      w[static_cast<size_t>(o) * K + i] = static_cast<int8_t>(v - 31);
    }
  }
  ob_gemm_desc_t desc{};
  desc.m_dim = M;
  desc.k_dim = K;
  desc.n_dim = N;
  desc.s_in = 0.03125f;
  desc.zp_in = 0;
  desc.s_w = 0.015625f;
  desc.zp_w = 0;
  desc.s_out = 0.0625f;
  desc.zp_out = 0;
  desc.activation = OB_ACT_NONE;
  desc.weights_nk = w.data();
  desc.bias_n = bias.data();

  ob::GemmSynthResult syn{};
  std::string err;
  if (ob::synthesize_gemm_parts(desc, &syn, &err) != OB_OK) {
    return {};
  }
  std::vector<uint8_t> interp = make_synthetic_dive_interp(interp_seed);
  std::vector<uint8_t> dgc0;
  if (ob::build_dgc0_from_parts(desc, syn, interp.data(), interp.size(), &dgc0, &err) != OB_OK) {
    return {};
  }
  return dgc0;
}

class TempBinaryFile {
 public:
  TempBinaryFile() = default;
  ~TempBinaryFile() {
    if (!path_.empty()) {
      ::unlink(path_.c_str());
    }
  }
  bool write(const void* data, size_t size) {
    char tmpl[] = "/tmp/openbuenos_test_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd < 0) return false;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t off = 0;
    while (off < size) {
      ssize_t n = ::write(fd, p + off, size - off);
      if (n <= 0) {
        ::close(fd);
        ::unlink(tmpl);
        return false;
      }
      off += static_cast<size_t>(n);
    }
    ::close(fd);
    path_ = tmpl;
    return true;
  }
  const char* path() const { return path_.c_str(); }

 private:
  std::string path_;
};

int g_failures = 0;

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
      ++g_failures;                                                             \
      return;                                                                   \
    }                                                                           \
  } while (0)

bool read_file(const char* path, std::vector<uint8_t>* out) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz <= 0) {
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(sz));
  size_t got = std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);
  return got == out->size();
}

// IEEE-754 fp32 -> fp16 encoder for constructing Q8_0 test tensors
uint16_t fp32_to_fp16(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x007fffffu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float fp16_to_fp32_ref(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t bits = 0;
  if (exp == 0u) {
    bits = sign;
  } else if (exp == 31u) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// ---------------------------------------------------------------------------
// Test 1: Buffer Slab/Pool Allocator, Reuse, Capacity Eviction & Sync
// ---------------------------------------------------------------------------
void test_buffer_pool() {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  cfg.buffer_pool_max_bytes = 16384u;  // 4 x 4KiB slabs max
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);
  CHECK(ob_context_is_mock(ctx) == 1);
  CHECK(ob_context_credit_size(ctx) == 16u);

  ob_buffer_t* b1 = nullptr;
  CHECK(ob_buffer_alloc(ctx, 1024u, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &b1) == OB_OK);
  CHECK(ob_buffer_size(b1) == 1024u);
  CHECK(ob_buffer_data(b1) != nullptr);
  uint64_t iova1 = ob_buffer_iova(b1);
  CHECK(iova1 != 0u && (iova1 & 4095u) == 0u);

  uint8_t payload[64];
  for (size_t i = 0; i < sizeof(payload); ++i) payload[i] = static_cast<uint8_t>(i * 3 + 7);
  CHECK(ob_buffer_write(b1, payload, sizeof(payload), 16u) == OB_OK);
  uint8_t readback[64]{};
  CHECK(ob_buffer_read(b1, readback, sizeof(readback), 16u) == OB_OK);
  CHECK(std::memcmp(payload, readback, sizeof(payload)) == 0);
  CHECK(ob_buffer_sync_to_device(b1) == OB_OK);
  CHECK(ob_buffer_sync_to_cpu(b1) == OB_OK);

  // Free returns slab to pool
  ob_buffer_free(b1);
  size_t pbytes = 0, pcount = 0;
  uint64_t phits = 0, pmisses = 0;
  CHECK(ob_context_buffer_pool_stats(ctx, &pbytes, &pcount, &phits, &pmisses) == OB_OK);
  CHECK(pcount == 1u && pbytes == 4096u && phits == 0u && pmisses == 1u);

  // Next allocation of <=4KiB reuses exact same IOVA (pool hit!)
  ob_buffer_t* b2 = nullptr;
  CHECK(ob_buffer_alloc(ctx, 2048u, OB_BUF_BIDIRECTIONAL, &b2) == OB_OK);
  CHECK(ob_buffer_iova(b2) == iova1);
  CHECK(ob_buffer_size(b2) == 2048u);
  CHECK(ob_context_buffer_pool_stats(ctx, &pbytes, &pcount, &phits, &pmisses) == OB_OK);
  CHECK(phits == 1u && pcount == 0u);
  ob_buffer_free(b2);

  // Pool eviction check: allocate 6 distinct 4KiB slabs simultaneously, then free all -> capped at 16KiB
  ob_buffer_t* batch[6]{};
  for (int i = 0; i < 6; ++i) {
    CHECK(ob_buffer_alloc(ctx, 4000u, OB_BUF_BIDIRECTIONAL, &batch[i]) == OB_OK);
  }
  for (int i = 0; i < 6; ++i) {
    ob_buffer_free(batch[i]);
  }
  CHECK(ob_context_buffer_pool_stats(ctx, &pbytes, &pcount, &phits, &pmisses) == OB_OK);
  CHECK(pbytes <= 16384u && pcount == 4u);

  ob_context_destroy(ctx);
  std::printf("[PASS] test_buffer_pool (hits=%llu misses=%llu pooled=%zuB)\n",
              static_cast<unsigned long long>(phits),
              static_cast<unsigned long long>(pmisses), pbytes);
}

// ---------------------------------------------------------------------------
// Test 2: Native .dgc0 Container Parser (1-in/params + 2-in/dyn_weights + corrupt)
// ---------------------------------------------------------------------------
void test_dgc0_parser() {
  std::vector<uint8_t> fc = make_synthetic_gemm_dgc0(1u, 16u, 16u, 0x5Au);
  std::vector<uint8_t> custom = make_synthetic_gemm_dgc0(1u, 16u, 32u, 0x5Au);
  CHECK(!fc.empty());
  CHECK(!custom.empty());

  // Also synthesize a 2-input weight-agnostic container (has_params == 0)
  std::vector<uint8_t> interp = make_synthetic_dive_interp(0x5Au);
  ob_gemm_desc_t dyn_desc{};
  dyn_desc.m_dim = 1u;
  dyn_desc.k_dim = 64u;
  dyn_desc.n_dim = 32u;
  dyn_desc.s_in = 0.03125f;
  dyn_desc.s_w = 0.015625f;
  dyn_desc.s_out = 0.0625f;
  std::vector<uint8_t> dyn;
  uint32_t dyn_ext2_bytes = 0u;
  std::string derr;
  CHECK(ob::build_dgc0_dyn_weights(dyn_desc, interp.data(), interp.size(),
                                   &dyn, &dyn_ext2_bytes, &derr) == OB_OK);

  ob_dgc0_info_t info {};
  CHECK(ob_dgc0_inspect(fc.data(), fc.size(), &info) == OB_OK);
  CHECK(info.num_inputs == 1u && info.input_sizes[0] == 16u);
  CHECK(info.num_outputs == 1u && info.output_sizes[0] == 16u);
  CHECK(info.ext_vm_ir_size > 0u);
  CHECK(info.ext_image_size == kSyntheticDiveInterpBytes);
  CHECK(info.has_params == 1u && info.ext_params_size == 320u);
  CHECK(info.dive_scratch_size == 192512u);
  CHECK(info.cluster_id == 2u);

  CHECK(ob_dgc0_inspect(dyn.data(), dyn.size(), &info) == OB_OK);
  CHECK(info.num_inputs == 2u && info.input_sizes[0] == 64u && info.input_sizes[1] == dyn_ext2_bytes);
  CHECK(info.num_outputs == 1u && info.output_sizes[0] == 32u);
  CHECK(info.has_params == 0u && info.ext_params_size == 0u);

  CHECK(ob_dgc0_inspect(custom.data(), custom.size(), &info) == OB_OK);
  CHECK(info.num_inputs == 1u && info.input_sizes[0] == 16u);
  CHECK(info.num_outputs == 1u && info.output_sizes[0] == 32u);
  CHECK(info.ext_vm_ir_size > 0u);
  CHECK(info.ext_image_size == kSyntheticDiveInterpBytes);
  CHECK(info.has_params == 1u && info.ext_params_size == 640u);

  // Corrupted container must fail cleanly
  uint8_t bad[32]{};
  CHECK(ob_dgc0_inspect(bad, sizeof(bad), &info) == OB_ERR_BAD_DGC0);

  std::printf("[PASS] test_dgc0_parser (fc=16->16/320B, dyn=2in->32, custom=16->32/640B)\n");
}

// ---------------------------------------------------------------------------
// Test 3: Persistent Graph LRU Cache (SHA-256 + Explicit Key + Eviction)
// ---------------------------------------------------------------------------
void test_graph_cache() {
  std::vector<uint8_t> fc = make_synthetic_gemm_dgc0(1u, 16u, 16u, 0x5Au);
  std::vector<uint8_t> custom = make_synthetic_gemm_dgc0(1u, 16u, 32u, 0x5Au);
  std::vector<uint8_t> third = make_synthetic_gemm_dgc0(4u, 32u, 32u, 0x5Au);
  CHECK(!fc.empty() && !custom.empty() && !third.empty());

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  cfg.graph_cache_capacity = 2u;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);

  // 1. Register fc with cache_key=0 (auto SHA-256) -> miss
  ob_graph_t* g1 = nullptr;
  CHECK(ob_graph_register_dgc0(ctx, fc.data(), fc.size(), 0u, &g1) == OB_OK);
  ob_dgc0_info_t i1 {};
  CHECK(ob_graph_get_info(g1, &i1) == OB_OK);
  CHECK(i1.cache_hit == 0u && i1.model_handle != 0u);
  uint32_t fc_handle = i1.model_handle;

  // Release g1: handle stays warm in LRU cache!
  ob_graph_release(g1);

  // 2. Register fc again (cache_key=0) -> SHA-256 cache HIT, same model_handle!
  ob_graph_t* g2 = nullptr;
  CHECK(ob_graph_register_dgc0(ctx, fc.data(), fc.size(), 0u, &g2) == OB_OK);
  ob_dgc0_info_t i2 {};
  CHECK(ob_graph_get_info(g2, &i2) == OB_OK);
  CHECK(i2.cache_hit == 1u && i2.model_handle == fc_handle);
  ob_graph_release(g2);

  // 3. Explicit cache_key hit & LRU eviction when capacity=2 exceeded
  ob_graph_t* gc = nullptr;
  CHECK(ob_graph_register_dgc0(ctx, custom.data(), custom.size(), 0x1111ULL, &gc) == OB_OK);
  ob_dgc0_info_t ic {};
  CHECK(ob_graph_get_info(gc, &ic) == OB_OK);
  CHECK(ic.cache_hit == 0u);
  ob_graph_release(gc);

  // Re-fetch by explicit key (even with NULL bytes if already resident!)
  ob_graph_t* gc_hit = nullptr;
  CHECK(ob_graph_register_dgc0(ctx, nullptr, 0u, 0x1111ULL, &gc_hit) == OB_OK);
  ob_dgc0_info_t ic2 {};
  CHECK(ob_graph_get_info(gc_hit, &ic2) == OB_OK);
  CHECK(ic2.cache_hit == 1u && ic2.model_handle == ic.model_handle);
  ob_graph_release(gc_hit);

  // Insert 3rd distinct graph -> evicts LRU (fc)
  ob_graph_t* ga = nullptr;
  CHECK(ob_graph_register_dgc0(ctx, third.data(), third.size(), 0x2222ULL, &ga) == OB_OK);
  ob_graph_release(ga);

  size_t entries = 0;
  uint64_t hits = 0, misses = 0, evictions = 0;
  CHECK(ob_context_graph_cache_stats(ctx, &entries, &hits, &misses, &evictions) == OB_OK);
  CHECK(entries == 2u && hits == 2u && misses == 3u && evictions == 1u);

  // Re-register fc -> must now be a miss (evicted) with a new handle
  ob_graph_t* g3 = nullptr;
  CHECK(ob_graph_register_dgc0(ctx, fc.data(), fc.size(), 0u, &g3) == OB_OK);
  ob_dgc0_info_t i3 {};
  CHECK(ob_graph_get_info(g3, &i3) == OB_OK);
  CHECK(i3.cache_hit == 0u && i3.model_handle != fc_handle);
  ob_graph_release(g3);

  ob_context_destroy(ctx);
  std::printf("[PASS] test_graph_cache (entries=%zu hits=%llu misses=%llu evictions=%llu)\n",
              entries, static_cast<unsigned long long>(hits),
              static_cast<unsigned long long>(misses),
              static_cast<unsigned long long>(evictions));
}

// ---------------------------------------------------------------------------
// Test 4: 16-Credit Async Ring State Machine (Full Saturation + Out-of-Order Wait)
// ---------------------------------------------------------------------------
void test_async_16_credit_ring() {
  std::vector<uint8_t> fc = make_synthetic_gemm_dgc0(1u, 16u, 16u, 0x5Au);
  CHECK(!fc.empty());
  TempBinaryFile fc_file;
  CHECK(fc_file.write(fc.data(), fc.size()));

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  cfg.max_inflight = 16u;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);

  ob_graph_t* g = nullptr;
  CHECK(ob_graph_register_dgc0_file(ctx, fc_file.path(), 0xFC01ULL, &g) == OB_OK);

  ob_buffer_t* in_buf = nullptr;
  ob_buffer_t* out_buf = nullptr;
  CHECK(ob_buffer_alloc(ctx, 16u, OB_BUF_TO_DEVICE | OB_BUF_ZERO_INIT, &in_buf) == OB_OK);
  CHECK(ob_buffer_alloc(ctx, 16u, OB_BUF_FROM_DEVICE | OB_BUF_ZERO_INIT, &out_buf) == OB_OK);

  // Fill all 16 VII credits concurrently
  ob_exec_handle_t handles[16]{};
  for (int i = 0; i < 16; ++i) {
    CHECK(ob_graph_submit_async(g, &in_buf, 1u, &out_buf, 1u, &handles[i]) == OB_OK);
    CHECK(handles[i] != 0u);
  }

  // 17th concurrent submission MUST return OB_ERR_BUSY
  ob_exec_handle_t extra = 0;
  CHECK(ob_graph_submit_async(g, &in_buf, 1u, &out_buf, 1u, &extra) == OB_ERR_BUSY);
  CHECK(extra == 0u);

  // Wait out-of-order on slot #7 -> frees 1 credit
  ob_invoke_stats_t st7 {};
  CHECK(ob_graph_wait(ctx, handles[7], 1000u, &st7) == OB_OK);
  CHECK((st7.flags & OB_STAT_FLAG_MOCK) != 0u);
  CHECK(st7.model_handle != 0u);

  // Double-wait on completed handle returns OB_ERR_NOT_FOUND
  CHECK(ob_graph_wait(ctx, handles[7], 1000u, nullptr) == OB_ERR_NOT_FOUND);

  // Now 17th submission succeeds into freed credit slot!
  CHECK(ob_graph_submit_async(g, &in_buf, 1u, &out_buf, 1u, &extra) == OB_OK);
  CHECK(extra != 0u);

  // Release graph handle while 16 ops are still in-flight (refcount safety check)
  ob_graph_release(g);

  // Drain all remaining 16 in-flight ops
  CHECK(ob_graph_wait_all(ctx, 2000u) == OB_OK);

  // Verify synchronous invoke still works cleanly after full drain
  ob_graph_t* g_again = nullptr;
  CHECK(ob_graph_register_dgc0_file(ctx, fc_file.path(), 0xFC01ULL, &g_again) == OB_OK);
  ob_invoke_stats_t sync_st {};
  CHECK(ob_graph_invoke(g_again, &in_buf, 1u, &out_buf, 1u, &sync_st) == OB_OK);
  CHECK((sync_st.flags & OB_STAT_FLAG_CACHE_HIT) != 0u);

  ob_graph_release(g_again);
  ob_buffer_free(out_buf);
  ob_buffer_free(in_buf);
  ob_context_destroy(ctx);
  std::printf("[PASS] test_async_16_credit_ring (16-credit saturation + OOO wait + drain)\n");
}

// ---------------------------------------------------------------------------
// Test 5: Weight Packing Exact Byte Parity + Channel/K Padding Rules
// ---------------------------------------------------------------------------
void test_weight_packing_parity() {
  constexpr uint32_t K = 16u;
  constexpr uint32_t N = 16u;
  int8_t w[N * K];
  int32_t bias[N];
  for (uint32_t o = 0; o < N; ++o) {
    int32_t b_mod = ((static_cast<int32_t>(o) * 29 - 100) % 250 + 250) % 250;
    bias[o] = b_mod - 125;
    for (uint32_t i = 0; i < K; ++i) {
      int32_t v = static_cast<int32_t>(o * 7u + i * 11u + 3u) % 63;
      w[o * K + i] = static_cast<int8_t>(v - 31);
    }
  }

  size_t packed_sz = ob_gemm_packed_weight_bytes(K, N);
  CHECK(packed_sz == 320u);
  std::vector<uint8_t> packed(packed_sz, 0);
  CHECK(ob_gemm_pack_weights_int8(w, bias, K, N, packed.data(), packed.size()) == OB_OK);

  // Verify exact byte parity with synthesized ext[2] payload inside DGC0
  std::vector<uint8_t> fc = make_synthetic_gemm_dgc0(1u, K, N, 0x5Au);
  ob::ParsedDgc0 parsed{};
  std::string err;
  CHECK(ob::parse_dgc0_buffer(fc.data(), fc.size(), &parsed, &err) == OB_OK);
  CHECK(parsed.has_params && parsed.ext_params.size == 320u);
  CHECK(std::memcmp(packed.data(), fc.data() + parsed.ext_params.offset, 320u) == 0);

  // Verify non-multiple-of-8 channel padding (N=20 -> n_pad=24, 480B, padded lanes=0x00)
  CHECK(ob_gemm_packed_weight_bytes(16u, 20u) == 480u);
  CHECK(ob_gemm_packed_weight_bytes(16u, 8u) == 160u);
  CHECK(ob_gemm_packed_weight_bytes(16u, 32u) == 640u);
  CHECK(ob_gemm_packed_weight_bytes(32u, 16u) == 576u);
  CHECK(ob_gemm_packed_weight_bytes(64u, 16u) == 1088u);

  std::vector<int8_t> w20(20u * 16u, 5);
  std::vector<uint8_t> p20(480u, 0xffu);
  CHECK(ob_gemm_pack_weights_int8(w20.data(), nullptr, 16u, 20u, p20.data(), p20.size()) == OB_OK);
  // Biases (first 24*4 = 96B) all zero when bias_n == NULL
  for (size_t i = 0; i < 96u; ++i) CHECK(p20[i] == 0u);
  // Active channels n < 20 have (5 + 128) = 133; padded channels n in [20..24) have 0x00
  for (uint32_t k4 = 0; k4 < 4u; ++k4) {
    for (uint32_t ni = 0; ni < 24u; ++ni) {
      for (uint32_t lane = 0; lane < 4u; ++lane) {
        uint8_t got = p20[96u + (k4 * 24u + ni) * 4u + lane];
        uint8_t exp = (ni < 20u) ? 133u : 0u;
        CHECK(got == exp);
      }
    }
  }

  std::printf("[PASS] test_weight_packing_parity (320/320 exact ext[2] match + channel padding)\n");
}

// ---------------------------------------------------------------------------
// Test 6: Pluggable Clean-Room Emitter Callback + Persistent Cache Reuse
// ---------------------------------------------------------------------------
struct MockEmitterState {
  std::vector<uint8_t> template_dgc0;
  uint32_t calls = 0;
  uint32_t last_m = 0, last_k = 0, last_n = 0;
};

ob_status_t mock_gemm_emitter(const ob_gemm_desc_t* desc,
                              uint8_t* out_buf,
                              size_t out_cap,
                              size_t* out_size,
                              void* user_data) {
  auto* st = static_cast<MockEmitterState*>(user_data);
  st->calls += 1u;
  st->last_m = desc->m_dim;
  st->last_k = desc->k_dim;
  st->last_n = desc->n_dim;
  *out_size = st->template_dgc0.size();
  if (!out_buf || out_cap < st->template_dgc0.size()) {
    return OB_ERR_OVERFLOW;
  }
  std::memcpy(out_buf, st->template_dgc0.data(), st->template_dgc0.size());
  return OB_OK;
}

void test_pluggable_emitter_tiling() {
  MockEmitterState est {};
  est.template_dgc0 = make_synthetic_gemm_dgc0(1u, 16u, 16u, 0x5Au);
  CHECK(!est.template_dgc0.empty());

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  cfg.graph_cache_capacity = 32u;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);
  CHECK(ob_gemm_register_emitter(ctx, mock_gemm_emitter, &est) == OB_OK);
  CHECK(ob_gemm_set_tile_limits(ctx, 2u, 16u, 16u) == OB_OK);

  // Shape M=2, K=32, N=32 -> 1 x 2 x 2 = 4 tiles
  constexpr uint32_t M = 2u, K = 32u, N = 32u;
  std::vector<int8_t> a(M * K), b(N * K);
  std::vector<int32_t> bias(N), c(M * N, 0);
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int8_t>((i * 5) % 21 - 10);
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int8_t>((i * 7) % 23 - 11);
  for (uint32_t n = 0; n < N; ++n) bias[n] = static_cast<int32_t>(n) - 10;

  ob_invoke_stats_t s1 {}, s2 {};
  CHECK(ob_gemm_matmul_int8(ctx, a.data(), b.data(), bias.data(), c.data(),
                            M, K, N, -3, 2, &s1) == OB_OK);
  CHECK(s1.tiles_dispatched == 4u);
  CHECK((s1.flags & OB_STAT_FLAG_EMITTER_USED) != 0u);
  CHECK(est.calls == 4u);
  CHECK(s1.cache_hits == 0u);

  // Second token with same weights: all 4 tiles MUST hit persistent graph cache!
  CHECK(ob_gemm_matmul_int8(ctx, a.data(), b.data(), bias.data(), c.data(),
                            M, K, N, -3, 2, &s2) == OB_OK);
  CHECK(s2.tiles_dispatched == 4u);
  CHECK(s2.cache_hits == 4u);
  CHECK((s2.flags & OB_STAT_FLAG_CACHE_HIT) != 0u);

  // Verify exact numerical match against naive reference
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      int32_t ref = bias[n];
      for (uint32_t k = 0; k < K; ++k) {
        ref += (static_cast<int32_t>(a[m * K + k]) + 3) *
               (static_cast<int32_t>(b[n * K + k]) - 2);
      }
      CHECK(c[m * N + n] == ref);
    }
  }

  ob_context_destroy(ctx);
  std::printf("[PASS] test_pluggable_emitter_tiling (4 tiles, repeat-token cache_hits=4/4)\n");
}

// ---------------------------------------------------------------------------
// Test 7: Tiled LLM-Size Matmul Accuracy (2048x2048 INT8 & 4096x11008 F32xQ8_0)
// ---------------------------------------------------------------------------
void test_llm_size_matmuls() {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);

  // 7A. INT8 2048x2048 (M=2, K=2048, N=2048)
  {
    constexpr uint32_t M = 2u, K = 2048u, N = 2048u;
    std::vector<int8_t> a(M * K);
    std::vector<int8_t> b(N * K);
    std::vector<int32_t> bias(N);
    std::vector<int32_t> c(M * N, 0);
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int8_t>((i * 13u + 5u) % 31u - 15);
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int8_t>((i * 17u + 3u) % 29u - 14);
    for (uint32_t n = 0; n < N; ++n) bias[n] = static_cast<int32_t>((n * 19u) % 100u) - 50;

    ob_invoke_stats_t stats {};
    CHECK(ob_gemm_matmul_int8(ctx, a.data(), b.data(), bias.data(), c.data(),
                              M, K, N, 0, 0, &stats) == OB_OK);
    // 1 M-tile * (2048/512=4 K-tiles) * (2048/256=8 N-tiles) = 32 tiles
    CHECK(stats.tiles_dispatched == 32u);

    // Spot-check 64 distributed outputs against exact reference
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; n += 64u) {
        int32_t ref = bias[n];
        for (uint32_t k = 0; k < K; ++k) {
          ref += static_cast<int32_t>(a[m * K + k]) * static_cast<int32_t>(b[n * K + k]);
        }
        CHECK(c[m * N + n] == ref);
      }
    }
    std::printf("[PASS] test_llm_int8_2048x2048 (tiles=%u wall_us=%llu exact=0 err)\n",
                stats.tiles_dispatched, static_cast<unsigned long long>(stats.wall_time_us));
  }

  // 7B. F32 x Q8_0 2048x2048 AND 4096x11008 (LLaMA-7B MLP gate/up shape!)
  const uint32_t shapes[2][3] = {
      {1u, 2048u, 2048u},
      {1u, 4096u, 11008u},
  };
  for (int s = 0; s < 2; ++s) {
    const uint32_t M = shapes[s][0];
    const uint32_t K = shapes[s][1];
    const uint32_t N = shapes[s][2];
    const uint32_t nb_k = K / OB_QK8_0;

    std::vector<float> a(M * K);
    std::vector<ob_block_q8_0_t> b(static_cast<size_t>(N) * nb_k);
    std::vector<float> c(M * N, 0.0f);

    for (uint32_t k = 0; k < K; ++k) {
      a[k] = static_cast<float>(static_cast<int32_t>((k * 23u + 7u) % 41u) - 20) * 0.025f;
    }
    const uint16_t d_scales[4] = {
        fp32_to_fp16(0.015625f),
        fp32_to_fp16(0.03125f),
        fp32_to_fp16(0.0078125f),
        fp32_to_fp16(0.0234375f),
    };
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        ob_block_q8_0_t& blk = b[static_cast<size_t>(n) * nb_k + bi];
        blk.d = d_scales[(n + bi) & 3u];
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          blk.qs[j] = static_cast<int8_t>(
              static_cast<int32_t>((n * 3u + bi * 5u + j * 11u) % 61u) - 30);
        }
      }
    }

    ob_invoke_stats_t stats {};
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a.data(), b.data(), c.data(),
                                  M, K, N, &stats) == OB_OK);
    uint32_t expected_tiles = ((K + 511u) / 512u) * ((N + 255u) / 256u);
    CHECK(stats.tiles_dispatched == expected_tiles);

    // Verify distributed rows against independent fp64 reference
    double max_abs_err = 0.0;
    uint32_t step = (N > 256u) ? (N / 64u) : 1u;
    for (uint32_t n = 0; n < N; n += step) {
      double ref = 0.0;
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        const ob_block_q8_0_t& blk = b[static_cast<size_t>(n) * nb_k + bi];
        double d = static_cast<double>(fp16_to_fp32_ref(blk.d));
        double bsum = 0.0;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          bsum += static_cast<double>(a[bi * OB_QK8_0 + j]) *
                  static_cast<double>(blk.qs[j]);
        }
        ref += d * bsum;
      }
      double err = std::fabs(static_cast<double>(c[n]) - ref);
      if (err > max_abs_err) max_abs_err = err;
    }
    CHECK(max_abs_err < 1e-4);
    std::printf("[PASS] test_llm_f32_q8_0_%ux%u (tiles=%u wall_us=%llu max_abs_err=%.2e)\n",
                K, N, stats.tiles_dispatched,
                static_cast<unsigned long long>(stats.wall_time_us), max_abs_err);
  }

  ob_context_destroy(ctx);
}

// ---------------------------------------------------------------------------
// Minimal FlatBuffer helper for Test 8 (reads TFLite FC + reference DGC0 DIVE)
// ---------------------------------------------------------------------------
struct TestFbView {
  const uint8_t* data = nullptr;
  size_t size = 0;

  bool in_bounds(size_t off, size_t n) const {
    return off <= size && n <= size - off;
  }

  template <typename T>
  bool read(size_t off, T* out) const {
    if (!in_bounds(off, sizeof(T))) return false;
    std::memcpy(out, data + off, sizeof(T));
    return true;
  }

  bool follow(size_t fp, size_t* target) const {
    uint32_t rel = 0;
    if (!read(fp, &rel) || rel == 0 || fp > size - rel) return false;
    *target = fp + rel;
    return *target <= size;
  }

  bool field_pos(size_t tbl, uint16_t slot, size_t* out) const {
    int32_t soff = 0;
    if (!read(tbl, &soff)) return false;
    int64_t vt_s = static_cast<int64_t>(tbl) - static_cast<int64_t>(soff);
    if (vt_s < 0) return false;
    size_t vt = static_cast<size_t>(vt_s);
    uint16_t vsz = 0, osz = 0;
    if (!read(vt, &vsz) || !read(vt + 2u, &osz)) return false;
    size_t so = 4u + static_cast<size_t>(slot) * 2u;
    if (so + 2u > vsz) return false;
    uint16_t fo = 0;
    if (!read(vt + so, &fo) || fo == 0 || fo >= osz || tbl + fo > size) return false;
    *out = tbl + fo;
    return true;
  }

  bool uoffset_vec(size_t tbl, uint16_t slot, std::vector<size_t>* elems) const {
    elems->clear();
    size_t fp = 0, vp = 0;
    if (!field_pos(tbl, slot, &fp) || !follow(fp, &vp)) return false;
    uint32_t len = 0;
    if (!read(vp, &len) || !in_bounds(vp + 4u, static_cast<size_t>(len) * 4u)) return false;
    elems->reserve(len);
    for (uint32_t i = 0; i < len; ++i) {
      size_t e = 0;
      if (!follow(vp + 4u + static_cast<size_t>(i) * 4u, &e)) return false;
      elems->push_back(e);
    }
    return true;
  }

  bool byte_vec(size_t tbl, uint16_t slot, const uint8_t** ptr, size_t* len) const {
    size_t fp = 0, vp = 0;
    if (!field_pos(tbl, slot, &fp) || !follow(fp, &vp)) return false;
    uint32_t n = 0;
    if (!read(vp, &n) || !in_bounds(vp + 4u, n)) return false;
    *ptr = data + vp + 4u;
    *len = n;
    return true;
  }

  template <typename T>
  bool scalar_vec(size_t tbl, uint16_t slot, std::vector<T>* out) const {
    out->clear();
    size_t fp = 0, vp = 0;
    if (!field_pos(tbl, slot, &fp) || !follow(fp, &vp)) return false;
    uint32_t n = 0;
    if (!read(vp, &n) || !in_bounds(vp + 4u, static_cast<size_t>(n) * sizeof(T))) return false;
    out->resize(n);
    for (uint32_t i = 0; i < n; ++i) {
      read(vp + 4u + static_cast<size_t>(i) * sizeof(T), &(*out)[i]);
    }
    return true;
  }
};

struct TfliteFcSpec {
  uint32_t M = 0, K = 0, N = 0;
  std::vector<int8_t> weights_nk;
  std::vector<int32_t> bias_n;
  float s_in = 1.0f, s_w = 1.0f, s_out = 1.0f;
  int32_t zp_in = 0, zp_out = 0;
  uint32_t act = OB_ACT_NONE;
};

bool parse_tflite_fc(const std::vector<uint8_t>& raw, TfliteFcSpec* out) {
  TestFbView fb{raw.data(), raw.size()};
  uint32_t root = 0;
  if (!fb.read(0u, &root)) return false;

  std::vector<size_t> buf_tbls;
  if (!fb.uoffset_vec(root, 4u, &buf_tbls)) return false;
  struct BufSpan { const uint8_t* ptr = nullptr; size_t len = 0; };
  std::vector<BufSpan> bufs(buf_tbls.size());
  for (size_t i = 0; i < buf_tbls.size(); ++i) {
    fb.byte_vec(buf_tbls[i], 0u, &bufs[i].ptr, &bufs[i].len);
  }

  std::vector<size_t> sgs;
  if (!fb.uoffset_vec(root, 2u, &sgs) || sgs.empty()) return false;
  std::vector<size_t> tensors, ops;
  if (!fb.uoffset_vec(sgs[0], 0u, &tensors) || !fb.uoffset_vec(sgs[0], 3u, &ops) || ops.empty()) {
    return false;
  }

  std::vector<int32_t> op_in, op_out;
  if (!fb.scalar_vec(ops[0], 1u, &op_in) || !fb.scalar_vec(ops[0], 2u, &op_out) ||
      op_in.size() < 2u || op_out.empty()) {
    return false;
  }

  // Fused activation inside BuiltinOptions (op slot 4 -> table slot 0)
  out->act = OB_ACT_NONE;
  size_t ofp = 0, ot = 0, afp = 0;
  if (fb.field_pos(ops[0], 4u, &ofp) && fb.follow(ofp, &ot) && fb.field_pos(ot, 0u, &afp)) {
    uint8_t a = 0;
    fb.read(afp, &a);
    out->act = a;
  }

  auto read_quant = [&](size_t t, float* scale, int32_t* zp) -> bool {
    size_t qfp = 0, qt = 0;
    if (!fb.field_pos(t, 4u, &qfp) || !fb.follow(qfp, &qt)) return false;
    std::vector<float> scs;
    std::vector<int64_t> zps;
    if (!fb.scalar_vec(qt, 2u, &scs) || scs.empty()) return false;
    *scale = scs[0];
    *zp = (fb.scalar_vec(qt, 3u, &zps) && !zps.empty()) ? static_cast<int32_t>(zps[0]) : 0;
    return true;
  };

  auto read_buf_idx = [&](size_t t) -> uint32_t {
    size_t fp = 0;
    uint32_t b = 0;
    if (fb.field_pos(t, 2u, &fp)) fb.read(fp, &b);
    return b;
  };

  size_t t_in = tensors[static_cast<size_t>(op_in[0])];
  size_t t_w = tensors[static_cast<size_t>(op_in[1])];
  size_t t_out = tensors[static_cast<size_t>(op_out[0])];

  std::vector<int32_t> in_shape, w_shape;
  if (!fb.scalar_vec(t_in, 0u, &in_shape) || in_shape.size() < 2u ||
      !fb.scalar_vec(t_w, 0u, &w_shape) || w_shape.size() < 2u) {
    return false;
  }
  out->M = static_cast<uint32_t>(in_shape[0]);
  out->K = static_cast<uint32_t>(in_shape[1]);
  out->N = static_cast<uint32_t>(w_shape[0]);

  int32_t zp_w_unused = 0;
  if (!read_quant(t_in, &out->s_in, &out->zp_in) ||
      !read_quant(t_w, &out->s_w, &zp_w_unused) ||
      !read_quant(t_out, &out->s_out, &out->zp_out)) {
    return false;
  }

  uint32_t wb_idx = read_buf_idx(t_w);
  if (wb_idx >= bufs.size() || bufs[wb_idx].len != static_cast<size_t>(out->N) * out->K) {
    return false;
  }
  out->weights_nk.resize(bufs[wb_idx].len);
  std::memcpy(out->weights_nk.data(), bufs[wb_idx].ptr, bufs[wb_idx].len);

  out->bias_n.assign(out->N, 0);
  if (op_in.size() >= 3u && op_in[2] >= 0) {
    size_t t_b = tensors[static_cast<size_t>(op_in[2])];
    uint32_t bb_idx = read_buf_idx(t_b);
    if (bb_idx < bufs.size() && bufs[bb_idx].len == static_cast<size_t>(out->N) * 4u) {
      std::memcpy(out->bias_n.data(), bufs[bb_idx].ptr, bufs[bb_idx].len);
    }
  }
  return true;
}

bool extract_dive_chunks(const uint8_t* dive, size_t dive_len,
                         std::array<std::vector<uint8_t>, 7>* out_chunks,
                         size_t* out_c6_off_in_dive) {
  TestFbView fb{dive, dive_len};
  uint32_t root = 0;
  if (!fb.read(0u, &root)) return false;
  std::vector<size_t> c_tbls;
  if (!fb.uoffset_vec(root, 3u, &c_tbls) || c_tbls.size() != 7u) return false;
  for (size_t i = 0; i < 7u; ++i) {
    const uint8_t* ptr = nullptr;
    size_t len = 0;
    if (!fb.byte_vec(c_tbls[i], 0u, &ptr, &len)) return false;
    (*out_chunks)[i].assign(ptr, ptr + len);
    if (i == 6u && out_c6_off_in_dive) {
      *out_c6_off_in_dive = static_cast<size_t>(ptr - dive);
    }
  }
  return true;
}

void mask_c6_cyclic_tag(std::vector<uint8_t>& c6) {
  if (c6.size() >= 80u) {
    c6[c6.size() - 80u + 1u] &= 0x0Fu;
  }
}

// ---------------------------------------------------------------------------
// Test 8: Built-in C++17 Emitter Synthesis & DGC0 Round-Trip (+ Optional Corpus Parity)
// ---------------------------------------------------------------------------
void test_builtin_emitter_81_parity() {
  // Always run self-contained clean-room synthesis & round-trip verification
  std::vector<uint8_t> fc = make_synthetic_gemm_dgc0(1u, 16u, 16u, 0x5Au);
  CHECK(!fc.empty());
  TempBinaryFile fc_file;
  CHECK(fc_file.write(fc.data(), fc.size()));

  size_t interp_sz = 0;
  CHECK(ob_gemm_extract_dive_interp_from_dgc0(fc_file.path(), nullptr, 0u, &interp_sz) == OB_ERR_OVERFLOW);
  CHECK(interp_sz == kSyntheticDiveInterpBytes);
  std::vector<uint8_t> synth_interp(interp_sz, 0u);
  CHECK(ob_gemm_extract_dive_interp_from_dgc0(
            fc_file.path(), synth_interp.data(), synth_interp.size(), &interp_sz) == OB_OK);

  struct ShapeSpec { uint32_t M, K, N, act; };
  const ShapeSpec specs[] = {
      {1u, 16u, 16u, OB_ACT_NONE},
      {1u, 16u, 32u, OB_ACT_RELU},
      {4u, 32u, 32u, OB_ACT_RELU6},
      {8u, 64u, 64u, OB_ACT_NONE},
      {16u, 64u, 64u, OB_ACT_RELU},
      {6u, 48u, 48u, OB_ACT_RELU6},
      {32u, 32u, 32u, OB_ACT_NONE},
      {1u, 512u, 128u, OB_ACT_NONE},
  };
  for (const ShapeSpec& sp : specs) {
    std::vector<int8_t> w(static_cast<size_t>(sp.N) * sp.K, 3);
    std::vector<int32_t> b(sp.N, 7);
    ob_gemm_desc_t desc{};
    desc.m_dim = sp.M;
    desc.k_dim = sp.K;
    desc.n_dim = sp.N;
    desc.s_in = 0.03125f;
    desc.zp_in = -5;
    desc.s_w = 0.015625f;
    desc.s_out = 0.05f;
    desc.zp_out = 11;
    desc.activation = sp.act;
    desc.weights_nk = w.data();
    desc.bias_n = b.data();

    ob::GemmSynthResult syn{};
    std::string serr;
    CHECK(ob::synthesize_gemm_parts(desc, &syn, &serr) == OB_OK);

    std::vector<uint8_t> built_dgc0;
    CHECK(ob::build_dgc0_from_parts(desc, syn, synth_interp.data(), synth_interp.size(),
                                    &built_dgc0, &serr) == OB_OK);
    CHECK((built_dgc0.size() % 64u) == 0u);

    ob_dgc0_info_t info{};
    CHECK(ob_dgc0_inspect(built_dgc0.data(), built_dgc0.size(), &info) == OB_OK);
    CHECK(info.num_inputs == 1u && info.input_sizes[0] == static_cast<uint64_t>(sp.M) * sp.K);
    CHECK(info.num_outputs == 1u && info.output_sizes[0] == static_cast<uint64_t>(sp.M) * sp.N);
    CHECK(info.ext_image_size == kSyntheticDiveInterpBytes);
    CHECK(info.has_params == 1u && info.ext_params_size == syn.ext2_params.size());
    CHECK(info.cluster_id == 2u);
  }

  // Optional: run full 81-model external corpus parity if OPENBUENOS_CORPUS_DIR is set
  const char* corpus_root = std::getenv("OPENBUENOS_CORPUS_DIR");
  if (!corpus_root || !corpus_root[0]) {
    std::printf("[PASS] test_builtin_emitter_synth_roundtrip (8/8 multi-tile shapes + DGC0 roundtrip; external 81-model parity skipped without OPENBUENOS_CORPUS_DIR)\n");
    return;
  }

  std::string kFcPath = std::string(corpus_root) + "/work/device-runner/fc.dgc0";
  CHECK(ob_gemm_extract_dive_interp_from_dgc0(kFcPath.c_str(), nullptr, 0u, &interp_sz) == OB_ERR_OVERFLOW);
  CHECK(interp_sz == 638977u);
  std::vector<uint8_t> dive_interp(interp_sz, 0u);
  CHECK(ob_gemm_extract_dive_interp_from_dgc0(
            kFcPath.c_str(), dive_interp.data(), dive_interp.size(), &interp_sz) == OB_OK);

  struct CorpusDir {
    const char* dgc0_dir;
    const char* tflite_dir;
    const char* required_prefix;
  };
  const CorpusDir dirs[4] = {
      {"/work/device-runner/phase3_gemm_sweep1", "/work/isa-gemm/sweep1", ""},
      {"/work/device-runner/phase3_batch0",      "/work/dgc0-dissector/sweep_models", "gemm_fc_"},
      {"/work/device-runner/phase4_gemm_sweep",  "/work/isa-gemm/phase4_sweep", ""},
      {"/work/device-runner/phase4_edge_sweep",  "/work/isa-gemm/phase4_edge_sweep", ""},
  };

  struct Pair {
    std::string stem;
    std::string dgc0_path;
    std::string tflite_path;
  };
  std::vector<Pair> pairs;
  for (const CorpusDir& cd : dirs) {
    std::string ddir = std::string(corpus_root) + cd.dgc0_dir;
    std::string tdir = std::string(corpus_root) + cd.tflite_dir;
    DIR* dp = ::opendir(ddir.c_str());
    CHECK(dp != nullptr);
    while (struct dirent* ent = ::readdir(dp)) {
      std::string name(ent->d_name);
      if (name.size() <= 5u || name.substr(name.size() - 5u) != ".dgc0") continue;
      std::string stem = name.substr(0u, name.size() - 5u);
      if (cd.required_prefix[0] && stem.rfind(cd.required_prefix, 0u) != 0u) continue;
      pairs.push_back({stem, ddir + "/" + name, tdir + "/" + stem + ".tflite"});
    }
    ::closedir(dp);
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const Pair& a, const Pair& b) { return a.stem < b.stem; });
  CHECK(pairs.size() == 81u);

  int passed_models = 0;
  int exact_chunks = 0;
  int exact_ext0 = 0;
  int exact_ext2 = 0;
  int exact_parse = 0;

  for (const Pair& pr : pairs) {
    std::vector<uint8_t> ref_dgc0, ref_tflite;
    CHECK(read_file(pr.dgc0_path.c_str(), &ref_dgc0));
    CHECK(read_file(pr.tflite_path.c_str(), &ref_tflite));

    TfliteFcSpec tf{};
    CHECK(parse_tflite_fc(ref_tflite, &tf));

    ob::ParsedDgc0 ref_parsed{};
    std::string perr;
    CHECK(ob::parse_dgc0_buffer(ref_dgc0.data(), ref_dgc0.size(), &ref_parsed, &perr) == OB_OK);

    const uint8_t* ref_ext0_ptr = ref_dgc0.data() + ref_parsed.ext_dive_vm_ir.offset;
    const size_t ref_ext0_sz = static_cast<size_t>(ref_parsed.ext_dive_vm_ir.size);
    const uint8_t* ref_ext2_ptr = ref_dgc0.data() + ref_parsed.ext_params.offset;
    const size_t ref_ext2_sz = static_cast<size_t>(ref_parsed.ext_params.size);

    std::array<std::vector<uint8_t>, 7> ref_chunks;
    size_t ref_c6_off = 0;
    CHECK(extract_dive_chunks(ref_ext0_ptr, ref_ext0_sz, &ref_chunks, &ref_c6_off));

    ob_gemm_desc_t desc {};
    desc.m_dim = tf.M;
    desc.k_dim = tf.K;
    desc.n_dim = tf.N;
    desc.s_in = tf.s_in;
    desc.zp_in = tf.zp_in;
    desc.s_w = tf.s_w;
    desc.zp_w = 0;
    desc.s_out = tf.s_out;
    desc.zp_out = tf.zp_out;
    desc.activation = tf.act;
    desc.weights_nk = tf.weights_nk.data();
    desc.bias_n = tf.bias_n.data();

    ob::GemmSynthResult syn{};
    std::string serr;
    ob_status_t st = ob::synthesize_gemm_parts(desc, &syn, &serr);
    if (st != OB_OK) {
      std::fprintf(stderr, "FAIL synth %s (%u,%u,%u): %s\n",
                   pr.stem.c_str(), tf.M, tf.K, tf.N, serr.c_str());
      ++g_failures;
      return;
    }

    // 1. Exact ext[2] byte parity
    bool ext2_ok = (syn.ext2_params.size() == ref_ext2_sz &&
                    std::memcmp(syn.ext2_params.data(), ref_ext2_ptr, ref_ext2_sz) == 0);
    if (!ext2_ok) {
      std::fprintf(stderr, "DIFF ext2 %s (%u,%u,%u): syn=%zu ref=%zu\n",
                   pr.stem.c_str(), tf.M, tf.K, tf.N, syn.ext2_params.size(), ref_ext2_sz);
    } else {
      ++exact_ext2;
    }

    // 2. Exact chunk[0..6] byte parity (masking only 4-bit cyclic tag in c6[-5])
    bool chunks_ok = true;
    for (size_t ci = 0; ci < 7u; ++ci) {
      std::vector<uint8_t> sc = syn.chunks[ci];
      std::vector<uint8_t> rc = ref_chunks[ci];
      if (ci == 6u) {
        mask_c6_cyclic_tag(sc);
        mask_c6_cyclic_tag(rc);
      }
      if (sc.size() != rc.size() || std::memcmp(sc.data(), rc.data(), sc.size()) != 0) {
        chunks_ok = false;
        std::fprintf(stderr, "DIFF c%zu %s (%u,%u,%u): syn=%zu ref=%zu\n",
                     ci, pr.stem.c_str(), tf.M, tf.K, tf.N, sc.size(), rc.size());
        break;
      }
    }
    if (chunks_ok) ++exact_chunks;

    // 3. Exact ext[0] (DIVE FlatBuffer) byte parity (masking only c6[-5] cyclic tag nibble)
    std::vector<uint8_t> syn_ext0_m = syn.ext0_dive;
    std::vector<uint8_t> ref_ext0_m(ref_ext0_ptr, ref_ext0_ptr + ref_ext0_sz);
    size_t syn_c6_off = 0;
    std::array<std::vector<uint8_t>, 7> syn_ext0_chunks;
    CHECK(extract_dive_chunks(syn_ext0_m.data(), syn_ext0_m.size(), &syn_ext0_chunks, &syn_c6_off));
    if (syn_c6_off + syn.chunks[6].size() <= syn_ext0_m.size() && syn.chunks[6].size() >= 80u) {
      syn_ext0_m[syn_c6_off + syn.chunks[6].size() - 80u + 1u] &= 0x0Fu;
    }
    if (ref_c6_off + ref_chunks[6].size() <= ref_ext0_m.size() && ref_chunks[6].size() >= 80u) {
      ref_ext0_m[ref_c6_off + ref_chunks[6].size() - 80u + 1u] &= 0x0Fu;
    }
    bool ext0_ok = (syn_ext0_m.size() == ref_ext0_m.size() &&
                    std::memcmp(syn_ext0_m.data(), ref_ext0_m.data(), syn_ext0_m.size()) == 0);
    if (!ext0_ok) {
      std::fprintf(stderr, "DIFF ext0 %s (%u,%u,%u): syn=%zu ref=%zu\n",
                   pr.stem.c_str(), tf.M, tf.K, tf.N, syn_ext0_m.size(), ref_ext0_m.size());
    } else {
      ++exact_ext0;
    }

    // 4. Full native DGC0 container build + parse_dgc0_buffer exact metadata verification
    std::vector<uint8_t> built_dgc0;
    CHECK(ob::build_dgc0_from_parts(desc, syn, dive_interp.data(), dive_interp.size(),
                                    &built_dgc0, &serr) == OB_OK);
    CHECK((built_dgc0.size() % 64u) == 0u);

    ob::ParsedDgc0 built_parsed{};
    CHECK(ob::parse_dgc0_buffer(built_dgc0.data(), built_dgc0.size(), &built_parsed, &perr) == OB_OK);
    bool parse_ok =
        built_parsed.input_sizes.size() == 1u &&
        built_parsed.output_sizes.size() == 1u &&
        built_parsed.input_sizes[0] == ref_parsed.input_sizes[0] &&
        built_parsed.output_sizes[0] == ref_parsed.output_sizes[0] &&
        built_parsed.dive_scratch_size == ref_parsed.dive_scratch_size &&
        built_parsed.exec_field_14 == ref_parsed.exec_field_14 &&
        built_parsed.dive_heap_size == ref_parsed.dive_heap_size &&
        built_parsed.exec_field_29 == ref_parsed.exec_field_29 &&
        built_parsed.cluster_id == ref_parsed.cluster_id &&
        built_parsed.compiler_cl == ref_parsed.compiler_cl &&
        built_parsed.has_params &&
        built_parsed.ext_dive_vm_ir.size == ref_parsed.ext_dive_vm_ir.size &&
        built_parsed.ext_dive_image.size == ref_parsed.ext_dive_image.size &&
        built_parsed.ext_params.size == ref_parsed.ext_params.size &&
        (built_parsed.ext_dive_vm_ir.offset % 64u) == 0u &&
        (built_parsed.ext_dive_image.offset % 64u) == 0u &&
        (built_parsed.ext_params.offset % 64u) == 0u &&
        std::memcmp(built_dgc0.data() + built_parsed.ext_dive_image.offset,
                    dive_interp.data(), dive_interp.size()) == 0 &&
        std::memcmp(built_dgc0.data() + built_parsed.ext_params.offset,
                    ref_ext2_ptr, ref_ext2_sz) == 0;
    if (!parse_ok) {
      std::fprintf(stderr, "DIFF parse %s (%u,%u,%u): f14=%u/%u scr=%llu/%llu\n",
                   pr.stem.c_str(), tf.M, tf.K, tf.N,
                   built_parsed.exec_field_14, ref_parsed.exec_field_14,
                   static_cast<unsigned long long>(built_parsed.dive_scratch_size),
                   static_cast<unsigned long long>(ref_parsed.dive_scratch_size));
    } else {
      ++exact_parse;
    }

    if (ext2_ok && chunks_ok && ext0_ok && parse_ok) {
      ++passed_models;
    }
  }

  CHECK(exact_ext2 == 81);
  CHECK(exact_chunks == 81);
  CHECK(exact_ext0 == 81);
  CHECK(exact_parse == 81);
  CHECK(passed_models == 81);
  std::printf("[PASS] test_builtin_emitter_81_parity (81/81 BIT-EXACT: c0..c6=81/81 ext0_DIVE=81/81 ext2=81/81 dgc0_parse=81/81)\n");
}

// ---------------------------------------------------------------------------
// Test 9: Token-Invariant Graph Caching + 16-Credit Async Pipeline (Builtin Emitter)
// ---------------------------------------------------------------------------
void test_token_invariant_caching_and_async_pipeline() {
  std::vector<uint8_t> synth_interp = make_synthetic_dive_interp(0x5Au);
  TempBinaryFile interp_file;
  CHECK(interp_file.write(synth_interp.data(), synth_interp.size()));

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 256u;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);

  // Dynamically load ext[1] image via builtin emitter registration (no vendor binaries in repo)
  CHECK(ob_gemm_register_builtin_emitter(ctx, interp_file.path()) == OB_OK);
  CHECK(ob_gemm_set_tile_limits(ctx, 16u, 512u, 128u) == OB_OK);

  // SmolLM2-135M FFN shape: M=1, K=576 (512+64), N=1536 -> 12x(512,128) + 24x(64,64) = 36 tiles (>16 credits!)
  // Probe-A weight-agnostic graph caching registers ONLY <=6 shape+scale model_handles (not 36!)!
  constexpr uint32_t M = 1u, K = 576u, N = 1536u;
  const uint32_t nb_k = K / OB_QK8_0;
  std::vector<float> a1(M * K), a2(M * K), c1(M * N, 0.0f), c2(M * N, 0.0f);
  std::vector<ob_block_q8_0_t> b(static_cast<size_t>(N) * nb_k);

  for (uint32_t k = 0; k < K; ++k) {
    a1[k] = static_cast<float>(static_cast<int32_t>((k * 17u + 3u) % 29u) - 14) * 0.03f;
    // Token 2 has ~25x larger activation scale (different s_in!)
    a2[k] = static_cast<float>(static_cast<int32_t>((k * 31u + 11u) % 41u) - 20) * 0.75f;
  }
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t bi = 0; bi < nb_k; ++bi) {
      ob_block_q8_0_t& blk = b[static_cast<size_t>(n) * nb_k + bi];
      blk.d = fp32_to_fp16(0.015625f + 0.004f * static_cast<float>((n + bi) & 3u));
      for (uint32_t j = 0; j < OB_QK8_0; ++j) {
        blk.qs[j] = static_cast<int8_t>(static_cast<int32_t>((n * 5u + bi * 7u + j * 13u) % 81u) - 40);
      }
    }
  }

  ob_invoke_stats_t s1 {}, s2 {};
  CHECK(ob_gemm_matmul_f32_q8_0(ctx, a1.data(), b.data(), c1.data(), M, K, N, &s1) == OB_OK);
  CHECK(s1.tiles_dispatched == 36u);
  CHECK((s1.flags & OB_STAT_FLAG_EMITTER_USED) != 0u);
  CHECK(s1.hw_execution_us > 0u);
  // Even on cold token 1, weight-agnostic Probe-A caching hits >80% across the 36 tiles!
  CHECK(s1.cache_hits >= 30u);

  // Token 2 with completely different s_in MUST achieve 36/36 (100%) graph + weight cache hits!
  CHECK(ob_gemm_matmul_f32_q8_0(ctx, a2.data(), b.data(), c2.data(), M, K, N, &s2) == OB_OK);
  CHECK(s2.tiles_dispatched == 36u);
  CHECK(s2.cache_hits == 36u);
  CHECK((s2.flags & OB_STAT_FLAG_CACHE_HIT) != 0u);

  size_t entries = 0;
  uint64_t hits = 0, misses = 0, evictions = 0;
  CHECK(ob_context_graph_cache_stats(ctx, &entries, &hits, &misses, &evictions) == OB_OK);
  CHECK(entries <= 6u && misses <= 6u && hits >= 66u && evictions == 0u);
  CHECK(ctx->weight_cache_misses == 36u && ctx->weight_cache_hits == 36u);

  // Edge shapes: M in {2, 4, 9, 29} padded to silicon-proven hw_m in {8, 16} (zero M=2/4 K=64 tiles!)
  // M=29 = 16 + 13(->hw_m=16) MUST reuse all 4 graphs from slice 0 -> cache_hits == 4 on a single call!
  for (uint32_t EM : {2u, 4u, 9u, 29u}) {
    constexpr uint32_t EK = 128u, EN = 100u;
    std::vector<float> ea(static_cast<size_t>(EM) * EK, 0.25f), ec(static_cast<size_t>(EM) * EN, 0.0f);
    std::vector<ob_block_q8_0_t> eb(static_cast<size_t>(EN) * (EK / OB_QK8_0));
    for (size_t i = 0; i < eb.size(); ++i) {
      eb[i].d = fp32_to_fp16(0.02f);
      for (uint32_t j = 0; j < OB_QK8_0; ++j) eb[i].qs[j] = static_cast<int8_t>((i + j) % 21 - 10);
    }
    ob_invoke_stats_t es {};
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, ea.data(), eb.data(), ec.data(), EM, EK, EN, &es) == OB_OK);
    CHECK((es.flags & OB_STAT_FLAG_EMITTER_USED) != 0u);
    if (EM <= 16u) {
      CHECK(es.tiles_dispatched == 4u);
    } else {
      CHECK(es.tiles_dispatched == 8u);
      CHECK(es.cache_hits >= 4u);
    }
  }

  ob_context_destroy(ctx);
  std::printf("[PASS] test_token_invariant_caching_and_async_pipeline (36 tiles -> %zu models via Probe-A param_slice, tok2 hits=36/36, w_hits=36/36, hw_m pad OK)\n", entries);
}

// ---------------------------------------------------------------------------
// Test 10: Realistic Quantized Tensor G5 TPU NMSE Simulator (IEEE-754 RNE, real_mul=1/D)
// ---------------------------------------------------------------------------
void test_quantized_tpu_nmse() {
  struct ShapeCase { uint32_t M, K, N; };
  const ShapeCase cases[] = {
      {4u,   32u,   64u},
      {4u,   64u,   64u},
      {4u,  128u,  128u},
      {4u,  256u,  192u},
      {2u,  512u,  256u},
      {2u,  576u,  576u},
      {1u,  576u, 1536u},
      {1u, 1536u,  576u},
      {1u, 2048u,  256u},
  };

  double worst_uniform_nmse = 0.0;
  double worst_gauss_nmse = 0.0;
  uint64_t total_clips = 0ULL;

  for (int dist = 0; dist < 2; ++dist) {  // 0 = Uniform[-1,1], 1 = Q8_0 Gaussian(0,1)
    std::mt19937 rng(dist == 0 ? 20261121u : 20261122u);
    std::uniform_real_distribution<float> udist(-1.0f, 1.0f);
    std::normal_distribution<float> gdist(0.0f, 1.0f);

    for (const ShapeCase& sc : cases) {
      const uint32_t M = sc.M, K = sc.K, N = sc.N;
      const uint32_t nb_k = K / OB_QK8_0;
      std::vector<float> A(static_cast<size_t>(M) * K);
      std::vector<ob_block_q8_0_t> Bq8(static_cast<size_t>(N) * nb_k);
      std::vector<float> B_deq(static_cast<size_t>(N) * K);

      for (float& v : A) v = (dist == 0) ? udist(rng) : gdist(rng);
      for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t bi = 0; bi < nb_k; ++bi) {
          float blk_f[OB_QK8_0];
          float am = 0.0f;
          for (uint32_t j = 0; j < OB_QK8_0; ++j) {
            blk_f[j] = (dist == 0) ? udist(rng) : gdist(rng);
            float av = std::fabs(blk_f[j]);
            if (av > am) am = av;
          }
          float d_f = (am > 0.0f) ? (am / 127.0f) : 1.0f;
          uint16_t d_h = fp32_to_fp16(d_f);
          float d_exact = fp16_to_fp32_ref(d_h);
          ob_block_q8_0_t& blk = Bq8[static_cast<size_t>(n) * nb_k + bi];
          blk.d = d_h;
          for (uint32_t j = 0; j < OB_QK8_0; ++j) {
            long q = (d_exact > 0.0f) ? std::lround(blk_f[j] / d_exact) : 0L;
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            blk.qs[j] = static_cast<int8_t>(q);
            B_deq[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j] = d_exact * static_cast<float>(blk.qs[j]);
          }
        }
      }

      // Exact FP64 reference of F32 x Q8_0
      std::vector<double> C_ref(static_cast<size_t>(M) * N, 0.0);
      for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
          double acc = 0.0;
          for (uint32_t k = 0; k < K; ++k) {
            acc += static_cast<double>(A[static_cast<size_t>(m) * K + k]) *
                   static_cast<double>(B_deq[static_cast<size_t>(n) * K + k]);
          }
          C_ref[static_cast<size_t>(m) * N + n] = acc;
        }
      }

      // Faithful Tensor G5 tiled INT8 + IEEE-754 RNE 1/D requantization simulator
      std::vector<float> C_sim(static_cast<size_t>(M) * N, 0.0f);
      const uint32_t k_step = (M == 1u) ? 512u : 64u;
      const uint32_t n_step = 64u;

      for (uint32_t k0 = 0; k0 < K; k0 += k_step) {
        const uint32_t tk = std::min(k_step, K - k0);
        std::vector<int8_t> aq(static_cast<size_t>(M) * tk);
        std::vector<float> sa(M, 0.0f);
        for (uint32_t m = 0; m < M; ++m) {
          float am = 0.0f;
          for (uint32_t ki = 0; ki < tk; ++ki) {
            float av = std::fabs(A[static_cast<size_t>(m) * K + k0 + ki]);
            if (av > am) am = av;
          }
          sa[m] = (am > 0.0f) ? (am / 127.0f) : 0.0f;
          float inv = (sa[m] > 0.0f) ? (1.0f / sa[m]) : 0.0f;
          for (uint32_t ki = 0; ki < tk; ++ki) {
            long q = std::lround(A[static_cast<size_t>(m) * K + k0 + ki] * inv);
            aq[static_cast<size_t>(m) * tk + ki] = static_cast<int8_t>(std::max(-127L, std::min(127L, q)));
          }
        }

        for (uint32_t n0 = 0; n0 < N; n0 += n_step) {
          const uint32_t tn = std::min(n_step, N - n0);
          std::vector<int8_t> bq(static_cast<size_t>(tn) * tk);
          std::vector<float> sb(tn, 0.0f);
          for (uint32_t ni = 0; ni < tn; ++ni) {
            float am = 0.0f;
            for (uint32_t ki = 0; ki < tk; ++ki) {
              float av = std::fabs(B_deq[static_cast<size_t>(n0 + ni) * K + k0 + ki]);
              if (av > am) am = av;
            }
            sb[ni] = (am > 0.0f) ? (am / 127.0f) : 0.0f;
            float inv = (sb[ni] > 0.0f) ? (1.0f / sb[ni]) : 0.0f;
            for (uint32_t ki = 0; ki < tk; ++ki) {
              long q = std::lround(B_deq[static_cast<size_t>(n0 + ni) * K + k0 + ki] * inv);
              bq[static_cast<size_t>(ni) * tk + ki] = static_cast<int8_t>(std::max(-127L, std::min(127L, q)));
            }
          }

          const uint32_t D = ob::compute_tile_scale_D(bq.data(), tn, tk, tk, 0);
          const float real_mul = 1.0f / static_cast<float>(D);
          for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t ni = 0; ni < tn; ++ni) {
              int32_t acc = 0;
              for (uint32_t ki = 0; ki < tk; ++ki) {
                acc += static_cast<int32_t>(aq[static_cast<size_t>(m) * tk + ki]) *
                       static_cast<int32_t>(bq[static_cast<size_t>(ni) * tk + ki]);
              }
              float scaled = static_cast<float>(acc) * real_mul;
              int32_t q_raw = static_cast<int32_t>(std::nearbyint(scaled));
              if (q_raw < -128 || q_raw > 127) total_clips += 1ULL;
              int32_t q_clamped = std::max(-128, std::min(127, q_raw));
              C_sim[static_cast<size_t>(m) * N + n0 + ni] +=
                  static_cast<float>(q_clamped) * static_cast<float>(D) * sa[m] * sb[ni];
            }
          }
        }
      }

      double mse = 0.0, ref_e = 0.0;
      for (size_t i = 0; i < C_ref.size(); ++i) {
        double diff = static_cast<double>(C_sim[i]) - C_ref[i];
        mse += diff * diff;
        ref_e += C_ref[i] * C_ref[i];
      }
      double nmse = mse / std::max(ref_e, 1e-30);
      if (dist == 0) {
        if (nmse > worst_uniform_nmse) worst_uniform_nmse = nmse;
        if (!(nmse < 5e-4)) {
          std::fprintf(stderr, "  [diag] dist=uniform shape=%ux%ux%u nmse=%.3e clips=%llu\n",
                       M, K, N, nmse, static_cast<unsigned long long>(total_clips));
        }
        CHECK(nmse < 5e-4);
      } else {
        if (nmse > worst_gauss_nmse) worst_gauss_nmse = nmse;
        if (!(nmse < 1e-3)) {
          std::fprintf(stderr, "  [diag] dist=gauss shape=%ux%ux%u nmse=%.3e clips=%llu\n",
                       M, K, N, nmse, static_cast<unsigned long long>(total_clips));
        }
        CHECK(nmse < 1e-3);
      }
    }
  }

  CHECK(total_clips == 0ULL);
  std::printf("[PASS] test_quantized_tpu_nmse (worst_uniform=%.2e <5e-4, worst_q8_gauss=%.2e <1e-3, clips=%llu)\n",
              worst_uniform_nmse, worst_gauss_nmse, static_cast<unsigned long long>(total_clips));
}

}  // namespace

int main() {
  test_buffer_pool();
  test_dgc0_parser();
  test_graph_cache();
  test_async_16_credit_ring();
  test_weight_packing_parity();
  test_pluggable_emitter_tiling();
  test_llm_size_matmuls();
  test_builtin_emitter_81_parity();
  test_token_invariant_caching_and_async_pipeline();
  test_quantized_tpu_nmse();

  if (g_failures == 0) {
    std::printf("ALL LIBOPENBUENOS HOST TESTS PASSED (10/10 suites)\n");
    return 0;
  }
  std::fprintf(stderr, "LIBOPENBUENOS HOST TESTS FAILED (%d failures)\n", g_failures);
  return 1;
}
