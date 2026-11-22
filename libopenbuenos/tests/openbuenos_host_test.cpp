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
  // Explicitly test multi-tile CPU accumulation (512x256 tiles) independent of default 2048x2048 lt_1x* limits
  CHECK(ob_gemm_set_tile_limits(ctx, 16u, 512u, 256u) == OB_OK);

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
      {1u, 1536u, 1536u},
      {1u, 2048u, 2048u},
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
      // Exercises real lt_1x* tile sizes (up to 2048x2048) for M==1 and 64x64 for M>1.
      std::vector<float> C_sim(static_cast<size_t>(M) * N, 0.0f);
      const uint32_t k_step = (M == 1u) ? 2048u : 64u;
      const uint32_t n_step = (M == 1u) ? 2048u : 64u;

      for (uint32_t k0 = 0; k0 < K; k0 += k_step) {
        const uint32_t tk = std::min(k_step, K - k0);
        std::vector<int8_t> aq(static_cast<size_t>(M) * tk);
        std::vector<float> sa(M, 0.0f);
        std::vector<std::vector<std::pair<uint32_t, float>>> a_outliers(M);
        for (uint32_t m = 0; m < M; ++m) {
          ob::quantize_activation_row_split(
              A.data() + static_cast<size_t>(m) * K + k0, tk,
              aq.data() + static_cast<size_t>(m) * tk, &sa[m], &a_outliers[m]);
        }
        const uint32_t a_shift = ob::compute_activation_rms_shift(aq.data(), M, tk, tk);

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

          uint32_t D = ob::compute_tile_scale_D(bq.data(), tn, tk, tk, 0);
          if (a_shift > 0u) {
            D = ob::quant_scale_D_grid(static_cast<double>(std::max<uint32_t>(1u, D >> a_shift)));
          }
          std::vector<int32_t> acc_buf(static_cast<size_t>(M) * tn, 0);
          for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t ni = 0; ni < tn; ++ni) {
              int32_t acc = 0;
              for (uint32_t ki = 0; ki < tk; ++ki) {
                acc += static_cast<int32_t>(aq[static_cast<size_t>(m) * tk + ki]) *
                       static_cast<int32_t>(bq[static_cast<size_t>(ni) * tk + ki]);
              }
              acc_buf[static_cast<size_t>(m) * tn + ni] = acc;
            }
          }
          for (uint32_t retry = 0; retry < 4u; ++retry) {
            const float real_mul = 1.0f / static_cast<float>(D);
            bool sat = false;
            for (int32_t acc : acc_buf) {
              int32_t q_raw = static_cast<int32_t>(std::nearbyint(static_cast<float>(acc) * real_mul));
              if (q_raw <= -128 || q_raw >= 127) { sat = true; break; }
            }
            if (sat && D < (1u << 24)) {
              D = ob::quant_scale_D_grid(std::min<double>(static_cast<double>(D) * 4.0, static_cast<double>(1u << 24)));
              continue;
            }
            break;
          }
          const float real_mul = 1.0f / static_cast<float>(D);
          for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t ni = 0; ni < tn; ++ni) {
              int32_t q_raw = static_cast<int32_t>(
                  std::nearbyint(static_cast<float>(acc_buf[static_cast<size_t>(m) * tn + ni]) * real_mul));
              if (q_raw < -128 || q_raw > 127) total_clips += 1ULL;
              int32_t q_clamped = std::max(-128, std::min(127, q_raw));
              float v = static_cast<float>(q_clamped) * static_cast<float>(D) * sa[m] * sb[ni];
              for (const auto& ov : a_outliers[m]) {
                v += ov.second * B_deq[static_cast<size_t>(n0 + ni) * K + k0 + ov.first];
              }
              C_sim[static_cast<size_t>(m) * N + n0 + ni] += v;
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

// ---------------------------------------------------------------------------
// Test 11: Clean-Room C++17 16-Core / 9-Core Hardware-Looped lt_1x* Parity
// ---------------------------------------------------------------------------
void test_lt_1x_6_parity() {
  std::vector<uint8_t> synth_interp = make_synthetic_dive_interp(0x5Au);

  // Always verify all 6 lt_1x* canonical shapes + 8 cross-product LLM shapes
  // synthesize, build valid 64B-aligned .dgc0 containers, and roundtrip-parse offline.
  const uint32_t all_lt_shapes[][2] = {
      {512u, 512u}, {576u, 576u}, {576u, 1536u},
      {1024u, 1024u}, {1536u, 576u}, {2048u, 2048u},
      {512u, 1536u}, {1536u, 1536u}, {1536u, 1024u}, {1024u, 1536u},
      {1536u, 2048u}, {2048u, 1536u}, {576u, 2048u}, {1536u, 512u},
  };
  for (const auto& sp : all_lt_shapes) {
    ob_gemm_desc_t desc {};
    desc.m_dim = 1u;
    desc.k_dim = sp[0];
    desc.n_dim = sp[1];
    desc.s_in = 1.0f;
    desc.s_w = 1.0f;
    desc.s_out = 64.0f;
    desc.activation = OB_ACT_NONE;
    desc.weights_nk = nullptr;
    desc.bias_n = nullptr;
    ob::GemmSynthResult syn{};
    std::string serr;
    CHECK(ob::synthesize_gemm_parts(desc, &syn, &serr) == OB_OK);
    std::vector<uint8_t> dgc0;
    CHECK(ob::build_dgc0_from_parts(desc, syn, synth_interp.data(), synth_interp.size(),
                                    &dgc0, &serr) == OB_OK);
    CHECK((dgc0.size() % 64u) == 0u);
    ob::ParsedDgc0 parsed{};
    CHECK(ob::parse_dgc0_buffer(dgc0.data(), dgc0.size(), &parsed, &serr) == OB_OK);
    CHECK(parsed.input_sizes[0] == sp[0]);
    CHECK(parsed.output_sizes[0] == sp[1]);
    // bias_n == nullptr omits the 4*N bias prefix -> ext[2] is exactly K*N bytes!
    CHECK(parsed.ext_params.size == static_cast<uint64_t>(sp[0]) * sp[1]);
  }

  // Optional: run full 6/6 external corpus bit-exact parity if OPENBUENOS_CORPUS_DIR is set
  const char* corpus_root = std::getenv("OPENBUENOS_CORPUS_DIR");
  if (!corpus_root || corpus_root[0] == '\0') {
    std::printf("[PASS] test_lt_1x_6_parity (14/14 lt_1x* + LLM cross-shapes DGC0 roundtrip; external 6-model parity skipped without OPENBUENOS_CORPUS_DIR)\n");
    return;
  }

  std::string kFcPath = std::string(corpus_root) + "/work/device-runner/fc.dgc0";
  size_t interp_sz = 0;
  CHECK(ob_gemm_extract_dive_interp_from_dgc0(kFcPath.c_str(), nullptr, 0u, &interp_sz) == OB_ERR_OVERFLOW);
  CHECK(interp_sz == 638977u);
  std::vector<uint8_t> dive_interp(interp_sz, 0u);
  CHECK(ob_gemm_extract_dive_interp_from_dgc0(
            kFcPath.c_str(), dive_interp.data(), dive_interp.size(), &interp_sz) == OB_OK);

  std::string lt_dir = std::string(corpus_root) + "/work/device-runner/phase5b_live/large_tile_probes";
  const char* stems[6] = {
      "lt_1x512x512",
      "lt_1x576x576",
      "lt_1x576x1536",
      "lt_1x1024x1024",
      "lt_1x1536x576",
      "lt_1x2048x2048",
  };

  int exact_chunks = 0;
  int exact_ext2 = 0;
  int exact_parse = 0;
  int passed = 0;

  for (const char* stem : stems) {
    std::string dgc0_path = lt_dir + "/" + stem + ".dgc0";
    std::string tflite_path = lt_dir + "/" + stem + ".tflite";

    std::vector<uint8_t> ref_dgc0, ref_tflite;
    CHECK(read_file(dgc0_path.c_str(), &ref_dgc0));
    CHECK(read_file(tflite_path.c_str(), &ref_tflite));

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
    CHECK(extract_dive_chunks(ref_ext0_ptr, ref_ext0_sz, &ref_chunks, nullptr));

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
      std::fprintf(stderr, "FAIL lt synth %s (%u,%u,%u): %s\n",
                   stem, tf.M, tf.K, tf.N, serr.c_str());
      ++g_failures;
      return;
    }

    bool ext2_ok = (syn.ext2_params.size() == ref_ext2_sz &&
                    std::memcmp(syn.ext2_params.data(), ref_ext2_ptr, ref_ext2_sz) == 0);
    if (!ext2_ok) {
      std::fprintf(stderr, "DIFF lt ext2 %s: syn=%zu ref=%zu\n",
                   stem, syn.ext2_params.size(), ref_ext2_sz);
    } else {
      ++exact_ext2;
    }

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
        size_t first_diff = 0;
        size_t lim = std::min(sc.size(), rc.size());
        while (first_diff < lim && sc[first_diff] == rc[first_diff]) ++first_diff;
        std::fprintf(stderr, "DIFF lt c%zu %s (%u,%u,%u): syn=%zu ref=%zu first_diff=%zu (syn=0x%02x ref=0x%02x)\n",
                     ci, stem, tf.M, tf.K, tf.N, sc.size(), rc.size(), first_diff,
                     first_diff < sc.size() ? sc[first_diff] : 0u,
                     first_diff < rc.size() ? rc[first_diff] : 0u);
        break;
      }
    }
    if (chunks_ok) ++exact_chunks;

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
        built_parsed.has_params &&
        built_parsed.ext_dive_image.size == ref_parsed.ext_dive_image.size &&
        built_parsed.ext_params.size == ref_parsed.ext_params.size &&
        (built_parsed.ext_dive_vm_ir.offset % 64u) == 0u &&
        (built_parsed.ext_dive_image.offset % 64u) == 0u &&
        (built_parsed.ext_params.offset % 64u) == 0u;
    if (!parse_ok) {
      std::fprintf(stderr, "DIFF lt parse %s: f14=%u/%u scr=%llu/%llu\n",
                   stem, built_parsed.exec_field_14, ref_parsed.exec_field_14,
                   static_cast<unsigned long long>(built_parsed.dive_scratch_size),
                   static_cast<unsigned long long>(ref_parsed.dive_scratch_size));
    } else {
      ++exact_parse;
    }

    if (ext2_ok && chunks_ok && parse_ok) ++passed;
  }

  CHECK(exact_ext2 == 6);
  CHECK(exact_chunks == 6);
  CHECK(exact_parse == 6);
  CHECK(passed == 6);
  std::printf("[PASS] test_lt_1x_6_parity (6/6 BIT-EXACT: c0..c6=6/6 ext2=6/6 dgc0_parse=6/6 + 8 LLM cross-shapes OK)\n");
}

// ---------------------------------------------------------------------------
// Test 12: Mode-A Zero-Overhead (+0 B) Fused GEMM+SiLU / GEMM+GELU Parity
// ---------------------------------------------------------------------------
void test_mode_a_fused_silu_gelu() {
  std::vector<uint8_t> synth_interp = make_synthetic_dive_interp(0x5Au);

  // Always verify direct synthesize_gemm_parts(desc.activation = OB_ACT_SILU / OB_ACT_GELU)
  // on BOTH standard (1x16x16) AND large-tile (lt_1x576x1536) shapes + full 256-codeword sweep!
  const uint32_t direct_shapes[][3] = {{1u, 16u, 16u}, {1u, 576u, 1536u}};
  const uint32_t direct_acts[2] = {OB_ACT_SILU, OB_ACT_GELU};
  double worst_synth_lsb_err = 0.0;
  for (const auto& ds : direct_shapes) {
    for (uint32_t act : direct_acts) {
      ob_gemm_desc_t base_desc {};
      base_desc.m_dim = ds[0];
      base_desc.k_dim = ds[1];
      base_desc.n_dim = ds[2];
      base_desc.s_in = 0.0625f;
      base_desc.s_w = 0.0625f;
      base_desc.s_out = 0.0625f;
      base_desc.zp_out = 0;
      base_desc.activation = OB_ACT_NONE;

      ob_gemm_desc_t fused_desc = base_desc;
      fused_desc.activation = act;

      ob::GemmSynthResult base_syn{}, fused_syn{};
      std::string err;
      CHECK(ob::synthesize_gemm_parts(base_desc, &base_syn, &err) == OB_OK);
      CHECK(ob::synthesize_gemm_parts(fused_desc, &fused_syn, &err) == OB_OK);
      CHECK(fused_syn.chunks[5].size() == base_syn.chunks[5].size());

      // Also verify explicit apply_mode_a_fusion_c5 in-place patch reports < 0.035 LSB error
      std::vector<uint8_t> c5_copy = base_syn.chunks[5];
      float max_lsb_err = 1.0f;
      CHECK(ob::apply_mode_a_fusion_c5(
                &c5_copy, act, 0.0625f, 0, 0.0625f, 0, &max_lsb_err, &err) == OB_OK);
      CHECK(c5_copy.size() == base_syn.chunks[5].size());
      CHECK(max_lsb_err < 0.035f);
      if (static_cast<double>(max_lsb_err) > worst_synth_lsb_err) {
        worst_synth_lsb_err = static_cast<double>(max_lsb_err);
      }

      std::vector<uint8_t> dgc0;
      CHECK(ob::build_dgc0_from_parts(fused_desc, fused_syn, synth_interp.data(),
                                      synth_interp.size(), &dgc0, &err) == OB_OK);
      ob::ParsedDgc0 parsed{};
      CHECK(ob::parse_dgc0_buffer(dgc0.data(), dgc0.size(), &parsed, &err) == OB_OK);
      CHECK(parsed.input_sizes[0] == ds[0] * ds[1]);
      CHECK(parsed.output_sizes[0] == ds[0] * ds[2]);

      // Verify full 256-codeword sweep through simulate_c5_mode_a_fused vs fp64 SiLU/GELU
      std::vector<uint8_t> u_all(256u);
      for (uint32_t i = 0; i < 256u; ++i) u_all[i] = static_cast<uint8_t>(i);
      std::vector<int8_t> y_sim(256u, 0);
      CHECK(ob::simulate_c5_mode_a_fused(fused_syn.chunks[5], u_all.data(), 256u, y_sim.data(), &err) == OB_OK);
      for (uint32_t i = 0; i < 256u; ++i) {
        double x = (static_cast<double>(i) - 128.0) * 0.0625;
        double fx = (act == OB_ACT_SILU)
            ? (x / (1.0 + std::exp(-x)))
            : (0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))));
        int32_t ref_q = static_cast<int32_t>(std::nearbyint(fx / 0.0625));
        if (ref_q < -128) ref_q = -128;
        if (ref_q > 127) ref_q = 127;
        CHECK(std::abs(static_cast<int32_t>(y_sim[i]) - ref_q) <= 1);
      }
    }
  }

  const char* corpus_root = std::getenv("OPENBUENOS_CORPUS_DIR");
  if (!corpus_root || corpus_root[0] == '\0') {
    std::printf("[PASS] test_mode_a_fused_silu_gelu (synthetic 256-codeword SiLU/GELU +0 B on 1x16x16 & lt_1x576x1536, worst_lsb=%.4f; external 4-probe parity skipped without OPENBUENOS_CORPUS_DIR)\n",
                worst_synth_lsb_err);
    return;
  }

  std::string probe_dir = std::string(corpus_root) + "/work/isa-gemm/int4_and_fusion/probes";
  struct FusedProbe {
    const char* name;
    uint32_t act;
    uint32_t M, K, N;
    float s_in;
    int32_t zp_in;
    float s_w;
    float s_mid;
    int32_t zp_mid;
    float s_out;
    int32_t zp_out;
  };
  const FusedProbe probes[4] = {
      {"fused_gemm_silu_1x16x16_zp0",    OB_ACT_SILU, 1u, 16u, 16u, 0.0625f,  0, 0.0625f, 0.0625f,   0, 0.0625f,    0},
      {"fused_gemm_gelu_1x16x16_zp0",    OB_ACT_GELU, 1u, 16u, 16u, 0.0625f,  0, 0.0625f, 0.0625f,   0, 0.0625f,    0},
      {"fused_gemm_silu_1x16x16_zpm112", OB_ACT_SILU, 1u, 16u, 16u, 0.0625f, -5, 0.0625f, 0.03125f, -10, 0.02f,  -112},
      {"fused_gemm_silu_4x32x32_zp0",    OB_ACT_SILU, 4u, 32u, 32u, 0.0625f,  3, 0.0625f, 0.0625f,   0, 0.0625f,    0},
  };

  int total_elems = 0;
  int exact_elems = 0;
  double worst_lsb_err = worst_synth_lsb_err;

  for (const FusedProbe& fp : probes) {
    std::string p_dgc0 = probe_dir + "/" + fp.name + ".dgc0";
    std::string p_in   = probe_dir + "/" + fp.name + ".in.bin";
    std::string p_ref  = probe_dir + "/" + fp.name + ".ref.bin";

    std::vector<uint8_t> ref_dgc0, in_raw, ref_raw;
    CHECK(read_file(p_dgc0.c_str(), &ref_dgc0));
    CHECK(read_file(p_in.c_str(), &in_raw));
    CHECK(read_file(p_ref.c_str(), &ref_raw));
    CHECK(in_raw.size() == static_cast<size_t>(fp.M) * fp.K);
    CHECK(ref_raw.size() == static_cast<size_t>(fp.M) * fp.N);

    ob::ParsedDgc0 ref_parsed{};
    std::string err;
    CHECK(ob::parse_dgc0_buffer(ref_dgc0.data(), ref_dgc0.size(), &ref_parsed, &err) == OB_OK);
    const uint8_t* ref_ext2 = ref_dgc0.data() + ref_parsed.ext_params.offset;

    const uint32_t n_pad = (fp.N + 7u) & ~7u;
    std::vector<int32_t> bias(fp.N, 0);
    for (uint32_t n = 0; n < fp.N; ++n) {
      uint32_t u = static_cast<uint32_t>(ref_ext2[4u * n]) |
                   (static_cast<uint32_t>(ref_ext2[4u * n + 1u]) << 8) |
                   (static_cast<uint32_t>(ref_ext2[4u * n + 2u]) << 16) |
                   (static_cast<uint32_t>(ref_ext2[4u * n + 3u]) << 24);
      bias[n] = static_cast<int32_t>(u);
    }
    std::vector<int8_t> weights(static_cast<size_t>(fp.N) * fp.K, 0);
    const uint8_t* wbase = ref_ext2 + 4u * n_pad;
    for (uint32_t k = 0; k < fp.K; ++k) {
      for (uint32_t n = 0; n < fp.N; ++n) {
        uint8_t b = wbase[(k / 4u) * n_pad * 4u + n * 4u + (k & 3u)];
        weights[static_cast<size_t>(n) * fp.K + k] = static_cast<int8_t>(static_cast<int32_t>(b) - 128);
      }
    }

    ob_gemm_desc_t desc {};
    desc.m_dim = fp.M;
    desc.k_dim = fp.K;
    desc.n_dim = fp.N;
    desc.s_in = fp.s_in;
    desc.zp_in = fp.zp_in;
    desc.s_w = fp.s_w;
    desc.zp_w = 0;
    desc.s_out = fp.s_mid;
    desc.zp_out = fp.zp_mid;
    desc.activation = OB_ACT_NONE;
    desc.weights_nk = weights.data();
    desc.bias_n = bias.data();

    ob::GemmSynthResult syn{};
    CHECK(ob::synthesize_gemm_parts(desc, &syn, &err) == OB_OK);
    const size_t c5_orig_sz = syn.chunks[5].size();

    float max_lsb_err = 1.0f;
    CHECK(ob::apply_mode_a_fusion_c5(
              &syn.chunks[5], fp.act, fp.s_mid, fp.zp_mid, fp.s_out, fp.zp_out,
              &max_lsb_err, &err) == OB_OK);
    CHECK(syn.chunks[5].size() == c5_orig_sz);
    CHECK(max_lsb_err < 0.035f);
    if (static_cast<double>(max_lsb_err) > worst_lsb_err) worst_lsb_err = static_cast<double>(max_lsb_err);

    std::array<std::vector<uint8_t>, 7> ref_chunks;
    CHECK(extract_dive_chunks(ref_dgc0.data() + ref_parsed.ext_dive_vm_ir.offset,
                              static_cast<size_t>(ref_parsed.ext_dive_vm_ir.size),
                              &ref_chunks, nullptr));
    for (size_t ci = 0; ci < 7u; ++ci) {
      if (ci == 5u) {
        CHECK(syn.chunks[5].size() == ref_chunks[5].size());
        continue;
      }
      std::vector<uint8_t> sc = syn.chunks[ci];
      std::vector<uint8_t> rc = ref_chunks[ci];
      if (ci == 6u) {
        mask_c6_cyclic_tag(sc);
        mask_c6_cyclic_tag(rc);
      }
      CHECK(sc.size() == rc.size() && std::memcmp(sc.data(), rc.data(), sc.size()) == 0);
    }

    const float real_mul_mid = (fp.s_in * fp.s_w) / fp.s_mid;
    const size_t n_out = static_cast<size_t>(fp.M) * fp.N;
    std::vector<uint8_t> u_mid(n_out, 0u);
    for (uint32_t m = 0; m < fp.M; ++m) {
      for (uint32_t n = 0; n < fp.N; ++n) {
        int32_t acc = bias[n];
        for (uint32_t k = 0; k < fp.K; ++k) {
          int32_t xv = static_cast<int32_t>(static_cast<int8_t>(in_raw[static_cast<size_t>(m) * fp.K + k]));
          int32_t wv = static_cast<int32_t>(weights[static_cast<size_t>(n) * fp.K + k]);
          acc += (xv - fp.zp_in) * wv;
        }
        float pre = static_cast<float>(acc) * real_mul_mid + static_cast<float>(fp.zp_mid + 128);
        int32_t q = static_cast<int32_t>(std::nearbyint(pre));
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        u_mid[static_cast<size_t>(m) * fp.N + n] = static_cast<uint8_t>(q);
      }
    }

    std::vector<int8_t> sim_out(n_out, 0);
    CHECK(ob::simulate_c5_mode_a_fused(syn.chunks[5], u_mid.data(), n_out, sim_out.data(), &err) == OB_OK);
    for (size_t i = 0; i < n_out; ++i) {
      int8_t ref_v = static_cast<int8_t>(ref_raw[i]);
      ++total_elems;
      if (sim_out[i] == ref_v) {
        ++exact_elems;
      } else {
        std::fprintf(stderr, "DIFF mode-a %s[%zu]: sim=%d ref=%d\n",
                     fp.name, i, static_cast<int>(sim_out[i]), static_cast<int>(ref_v));
      }
      CHECK(sim_out[i] == ref_v);
    }
  }

  CHECK(exact_elems == total_elems && total_elems == 176);
  std::printf("[PASS] test_mode_a_fused_silu_gelu (4/4 probes %d/%d EXACT max_abs_diff=0, worst_lsb=%.4f, +0 B on 1x16x16 & lt_1x576x1536)\n",
              exact_elems, total_elems, worst_lsb_err);
}

// ---------------------------------------------------------------------------
// Test 13: Default 2048x2048 lt_1x* Tiling & Probe-A Cache Reduction (SmolLM2 + Qwen2.5)
// ---------------------------------------------------------------------------
void test_lt_1x_tiling_and_cache() {
  std::vector<uint8_t> synth_interp = make_synthetic_dive_interp(0x5Au);
  TempBinaryFile interp_file;
  CHECK(interp_file.write(synth_interp.data(), synth_interp.size()));

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 64u;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);
  CHECK(ob_gemm_register_builtin_emitter(ctx, interp_file.path()) == OB_OK);

  auto run_f32_q8 = [&](uint32_t K, uint32_t N, uint32_t expect_tiles, uint32_t expect_tok2_hits) {
    const uint32_t nb_k = K / OB_QK8_0;
    std::vector<float> a1(K), a2(K), c1(N, 0.0f), c2(N, 0.0f);
    std::vector<ob_block_q8_0_t> b(static_cast<size_t>(N) * nb_k);
    for (uint32_t k = 0; k < K; ++k) {
      a1[k] = static_cast<float>(static_cast<int32_t>((k * 13u + 7u) % 29u) - 14) * 0.02f;
      a2[k] = static_cast<float>(static_cast<int32_t>((k * 23u + 5u) % 37u) - 18) * 0.50f;
    }
    const uint16_t d_half = fp32_to_fp16(0.015625f);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        ob_block_q8_0_t& blk = b[static_cast<size_t>(n) * nb_k + bi];
        blk.d = d_half;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          blk.qs[j] = static_cast<int8_t>(static_cast<int32_t>((n * 3u + bi * 5u + j * 11u) % 61u) - 30);
        }
      }
    }
    ob_invoke_stats_t s1 {}, s2 {};
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a1.data(), b.data(), c1.data(), 1u, K, N, &s1) == OB_OK);
    CHECK((s1.flags & OB_STAT_FLAG_EMITTER_USED) != 0u);
    CHECK(s1.tiles_dispatched == expect_tiles);

    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a2.data(), b.data(), c2.data(), 1u, K, N, &s2) == OB_OK);
    CHECK(s2.tiles_dispatched == expect_tiles);
    CHECK(s2.cache_hits == expect_tok2_hits);
  };

  // SmolLM2-135M projections: 1 tile each! (576x192 pads -> 576x576!)
  run_f32_q8(576u,  576u,  1u, 1u);
  run_f32_q8(576u,  192u,  1u, 1u);
  run_f32_q8(576u, 1536u,  1u, 1u);
  run_f32_q8(1536u, 576u,  1u, 1u);
  // Qwen2.5-1.5B projections: 1536x256->1, 1536x1536->1, 1536x8960->5, 8960x1536->5!
  run_f32_q8(1536u, 256u,  1u, 1u);
  run_f32_q8(1536u, 1536u, 1u, 1u);
  run_f32_q8(1536u, 8960u, 5u, 5u);
  run_f32_q8(8960u, 1536u, 5u, 5u);

  // Mixed lt_1x* + small-tile tail (<192): K=2112 (2048+64), N=600 (->1024)
  {
    constexpr uint32_t K = 2112u, N = 600u;
    const uint32_t nb_k = K / OB_QK8_0;
    std::vector<float> a(K, 0.1f), c(N, 0.0f);
    std::vector<ob_block_q8_0_t> b(static_cast<size_t>(N) * nb_k);
    const uint16_t d_half = fp32_to_fp16(0.015625f);
    for (auto& blk : b) {
      blk.d = d_half;
      for (uint32_t j = 0; j < OB_QK8_0; ++j) blk.qs[j] = static_cast<int8_t>((j % 17u) - 8);
    }
    ob_invoke_stats_t s {};
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a.data(), b.data(), c.data(), 1u, K, N, &s) == OB_OK);
    CHECK((s.flags & OB_STAT_FLAG_EMITTER_USED) != 0u);
    CHECK(s.tiles_dispatched == 11u);
  }

  // Reused scratch-pointer collision immunity
  {
    constexpr uint32_t K = 576u, N = 1536u;
    const uint32_t nb_k = K / OB_QK8_0;
    std::vector<float> a(K, 0.2f), c(N, 0.0f);
    std::vector<ob_block_q8_0_t> scratch_b(static_cast<size_t>(N) * nb_k);
    auto fill_layer = [&](uint32_t layer_id) {
      const uint16_t dh = fp32_to_fp16(0.015625f + 0.002f * static_cast<float>(layer_id));
      for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t bi = 0; bi < nb_k; ++bi) {
          ob_block_q8_0_t& blk = scratch_b[static_cast<size_t>(n) * nb_k + bi];
          blk.d = dh;
          for (uint32_t j = 0; j < OB_QK8_0; ++j) {
            blk.qs[j] = static_cast<int8_t>(
                static_cast<int32_t>((layer_id * 19u + n * 7u + bi * 13u + j * 3u) % 63u) - 31);
          }
        }
      }
    };
    const uint64_t m0 = ctx->weight_cache_misses;
    const uint64_t h0 = ctx->weight_cache_hits;
    ob_invoke_stats_t s {};

    fill_layer(10u);
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a.data(), scratch_b.data(), c.data(), 1u, K, N, &s) == OB_OK);
    CHECK(ctx->weight_cache_misses == m0 + 1u && ctx->weight_cache_hits == h0);

    fill_layer(11u);
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a.data(), scratch_b.data(), c.data(), 1u, K, N, &s) == OB_OK);
    CHECK(ctx->weight_cache_misses == m0 + 2u && ctx->weight_cache_hits == h0);

    fill_layer(10u);
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a.data(), scratch_b.data(), c.data(), 1u, K, N, &s) == OB_OK);
    CHECK(ctx->weight_cache_misses == m0 + 2u && ctx->weight_cache_hits == h0 + 1u);
  }

  size_t entries = 0;
  uint64_t hits = 0, misses = 0, evictions = 0;
  CHECK(ob_context_graph_cache_stats(ctx, &entries, &hits, &misses, &evictions) == OB_OK);
  CHECK(evictions == 0u);
  CHECK(entries <= 20u);

  ob_context_destroy(ctx);
  std::printf("[PASS] test_lt_1x_tiling_and_cache (SmolLM2 576x1536->1t, Qwen2.5 1536x8960->5t, mixed tail->11t, models=%zu, evict=0)\n",
              entries);
}

// ---------------------------------------------------------------------------
// Test 14: Qwen2.5-Style 40-Sigma Activation Outlier Peel + RMS D-Shift + TPU Retry
// ---------------------------------------------------------------------------
void test_outlier_spike_and_rms_shift() {
  std::mt19937 rng(20261121u);
  std::normal_distribution<float> gdist(0.0f, 1.0f);

  {
    constexpr uint32_t K = 1536u, N = 256u;
    std::vector<float> A(K), B(static_cast<size_t>(N) * K);
    for (float& v : A) v = gdist(rng);
    A[17] = 50.0f;
    A[513] = -42.0f;
    for (float& v : B) v = gdist(rng);
    for (uint32_t n = 0; n < N; ++n) {
      B[static_cast<size_t>(n) * K + 17] *= 0.02f;
      B[static_cast<size_t>(n) * K + 513] *= 0.02f;
    }

    std::vector<int8_t> aq(K);
    float sa = 0.0f;
    std::vector<std::pair<uint32_t, float>> outliers;
    ob::quantize_activation_row_split(A.data(), K, aq.data(), &sa, &outliers);
    CHECK(outliers.size() == 2u);
    CHECK(aq[17] == 0 && aq[513] == 0);
    uint32_t shift = ob::compute_activation_rms_shift(aq.data(), 1u, K, K);

    std::vector<int8_t> bq(static_cast<size_t>(N) * K);
    std::vector<float> sb(N);
    for (uint32_t n = 0; n < N; ++n) {
      float am = 0.0f;
      for (uint32_t k = 0; k < K; ++k) am = std::max(am, std::fabs(B[static_cast<size_t>(n) * K + k]));
      sb[n] = am / 127.0f;
      float inv = 127.0f / am;
      for (uint32_t k = 0; k < K; ++k) {
        bq[static_cast<size_t>(n) * K + k] = static_cast<int8_t>(
            std::max(-127L, std::min(127L, std::lround(B[static_cast<size_t>(n) * K + k] * inv))));
      }
    }
    uint32_t D = ob::quant_scale_D_grid(
        static_cast<double>(std::max<uint32_t>(1u, ob::compute_tile_scale_D(bq.data(), N, K, K, 0) >> shift)));

    double mse = 0.0, ref_e = 0.0;
    uint32_t clips = 0u;
    for (uint32_t n = 0; n < N; ++n) {
      double ref = 0.0;
      int32_t acc = 0;
      for (uint32_t k = 0; k < K; ++k) {
        ref += static_cast<double>(A[k]) * static_cast<double>(B[static_cast<size_t>(n) * K + k]);
        acc += static_cast<int32_t>(aq[k]) * static_cast<int32_t>(bq[static_cast<size_t>(n) * K + k]);
      }
      int32_t qr = static_cast<int32_t>(std::nearbyint(static_cast<float>(acc) / static_cast<float>(D)));
      if (qr < -128 || qr > 127) ++clips;
      double got = static_cast<double>(std::max(-128, std::min(127, qr))) *
                   static_cast<double>(D) * static_cast<double>(sa) * static_cast<double>(sb[n]);
      for (const auto& ov : outliers) {
        got += static_cast<double>(ov.second) * static_cast<double>(B[static_cast<size_t>(n) * K + ov.first]);
      }
      mse += (got - ref) * (got - ref);
      ref_e += ref * ref;
    }
    double nmse = mse / std::max(ref_e, 1e-30);
    CHECK(clips == 0u);
    CHECK(nmse < 1e-3);

    std::vector<float> A2(K), B2(static_cast<size_t>(N) * K);
    for (float& v : A2) v = gdist(rng) * 0.1f;
    for (float& v : B2) v = gdist(rng) * 0.1f;
    for (uint32_t s = 0; s < 20u; ++s) {
      A2[s * 32u] = 40.0f;
      for (uint32_t n = 0; n < 4u; ++n) B2[static_cast<size_t>(n) * K + s * 32u] = 2.0f;
    }
    std::vector<int8_t> aq2(K);
    float sa2 = 0.0f;
    std::vector<std::pair<uint32_t, float>> out2;
    ob::quantize_activation_row_split(A2.data(), K, aq2.data(), &sa2, &out2);
    CHECK(out2.size() == 8u);
    uint32_t sh2 = ob::compute_activation_rms_shift(aq2.data(), 1u, K, K);
    CHECK(sh2 >= 1u);
    std::vector<int8_t> bq2(static_cast<size_t>(N) * K);
    std::vector<float> sb2(N);
    for (uint32_t n = 0; n < N; ++n) {
      float am = 0.0f;
      for (uint32_t k = 0; k < K; ++k) am = std::max(am, std::fabs(B2[static_cast<size_t>(n) * K + k]));
      sb2[n] = am / 127.0f;
      float inv = 127.0f / am;
      for (uint32_t k = 0; k < K; ++k) {
        bq2[static_cast<size_t>(n) * K + k] = static_cast<int8_t>(
            std::max(-127L, std::min(127L, std::lround(B2[static_cast<size_t>(n) * K + k] * inv))));
      }
    }
    uint32_t D2 = ob::quant_scale_D_grid(
        static_cast<double>(std::max<uint32_t>(1u, ob::compute_tile_scale_D(bq2.data(), N, K, K, 0) >> sh2)));
    std::vector<int32_t> acc2(N, 0);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t k = 0; k < K; ++k) acc2[n] += static_cast<int32_t>(aq2[k]) * static_cast<int32_t>(bq2[static_cast<size_t>(n) * K + k]);
    }
    uint32_t retries = 0u;
    for (uint32_t r = 0; r < 4u; ++r) {
      bool sat = false;
      for (int32_t a : acc2) {
        int32_t qr = static_cast<int32_t>(std::nearbyint(static_cast<float>(a) / static_cast<float>(D2)));
        if (qr <= -128 || qr >= 127) { sat = true; break; }
      }
      if (sat) {
        ++retries;
        D2 = ob::quant_scale_D_grid(static_cast<double>(D2) * 4.0);
      } else {
        break;
      }
    }
    CHECK(retries >= 1u);
    double mse2 = 0.0, ref2 = 0.0;
    uint32_t clips2 = 0u;
    for (uint32_t n = 0; n < N; ++n) {
      double ref = 0.0;
      for (uint32_t k = 0; k < K; ++k) ref += static_cast<double>(A2[k]) * static_cast<double>(B2[static_cast<size_t>(n) * K + k]);
      int32_t qr = static_cast<int32_t>(std::nearbyint(static_cast<float>(acc2[n]) / static_cast<float>(D2)));
      if (qr < -128 || qr > 127) ++clips2;
      double got = static_cast<double>(std::max(-128, std::min(127, qr))) *
                   static_cast<double>(D2) * static_cast<double>(sa2) * static_cast<double>(sb2[n]);
      for (const auto& ov : out2) got += static_cast<double>(ov.second) * static_cast<double>(B2[static_cast<size_t>(n) * K + ov.first]);
      mse2 += (got - ref) * (got - ref);
      ref2 += ref * ref;
    }
    double nmse2 = mse2 / std::max(ref2, 1e-30);
    CHECK(clips2 == 0u);
    CHECK(nmse2 < 2e-3);

    std::printf("[PASS] test_outlier_spike_and_rms_shift (40-sigma peel nmse=%.2e clips=0, 12-spike retry=%u nmse=%.2e clips=0)\n",
                nmse, retries, nmse2);
  }
}

// ---------------------------------------------------------------------------
// Test 15: M=64 Batched Prefill Large-Tile ISA (5/5 Bit-Exact -> 92/92 Total!)
//          + Generalized LLM Shapes + Prefill Tiling + pack_fmt Cache Separation
// ---------------------------------------------------------------------------
void test_lt_64x_batched_prefill_parity() {
  std::vector<uint8_t> synth_interp = make_synthetic_dive_interp(0x5Au);
  TempBinaryFile interp_file;
  CHECK(interp_file.write(synth_interp.data(), synth_interp.size()));

  // Always verify all 5 canonical M=64 shapes + 6 generalized LLM cross-shapes offline
  const uint32_t gen_shapes[][2] = {
      {512u,  512u}, {576u,  576u}, {576u, 1536u}, {1024u, 1024u}, {1536u, 1536u},
      {512u,  128u}, {576u,  192u}, {1536u, 256u},
      {1536u, 576u}, {1536u, 1280u}, {1024u, 1536u},
  };
  for (const auto& sp : gen_shapes) {
    CHECK(ob::is_lt_64x_shape(64u, sp[0], sp[1]));
    ob_gemm_desc_t desc {};
    desc.m_dim = 64u;
    desc.k_dim = sp[0];
    desc.n_dim = sp[1];
    desc.s_in = 1.0f;
    desc.s_w = 1.0f;
    desc.s_out = 64.0f;
    desc.activation = OB_ACT_NONE;
    desc.weights_nk = nullptr;
    desc.bias_n = nullptr;
    ob::GemmSynthResult syn{};
    std::string serr;
    CHECK(ob::synthesize_gemm_parts(desc, &syn, &serr) == OB_OK);
    CHECK(syn.chunks[5].size() == 3440u);
    CHECK(syn.chunks[6].size() == 3616u);
    std::vector<uint8_t> dgc0;
    CHECK(ob::build_dgc0_from_parts(desc, syn, synth_interp.data(), synth_interp.size(),
                                    &dgc0, &serr) == OB_OK);
    ob::ParsedDgc0 parsed{};
    CHECK(ob::parse_dgc0_buffer(dgc0.data(), dgc0.size(), &parsed, &serr) == OB_OK);
    CHECK(parsed.input_sizes[0] == 64u * sp[0]);
    CHECK(parsed.output_sizes[0] == 64u * sp[1]);
    CHECK(parsed.ext_params.size == static_cast<uint64_t>(sp[0]) * sp[1] + 16u * sp[1]);
    CHECK(parsed.dive_scratch_size == 192512u + 64u * (sp[0] + sp[1]));
    CHECK(parsed.exec_field_14 == 1024u * (sp[0] + sp[1]) + 320u);
  }

  // Prefill tiling tile-count reduction + M=1 (pack_fmt=0) vs M=64 (pack_fmt=1) cache separation
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 128u;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);
  CHECK(ob_gemm_register_builtin_emitter(ctx, interp_file.path()) == OB_OK);

  auto check_prefill_tiles = [&](uint32_t M, uint32_t K, uint32_t N, uint32_t expect_tiles) {
    const uint32_t nb_k = K / OB_QK8_0;
    std::vector<float> a(static_cast<size_t>(M) * K, 0.05f);
    std::vector<float> c(static_cast<size_t>(M) * N, 0.0f);
    std::vector<ob_block_q8_0_t> b(static_cast<size_t>(N) * nb_k);
    const uint16_t dh = fp32_to_fp16(0.015625f);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        ob_block_q8_0_t& blk = b[static_cast<size_t>(n) * nb_k + bi];
        blk.d = dh;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          blk.qs[j] = static_cast<int8_t>(static_cast<int32_t>((n + bi * 3u + j * 7u) % 41u) - 20);
        }
      }
    }
    ob_invoke_stats_t s {};
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a.data(), b.data(), c.data(), M, K, N, &s) == OB_OK);
    CHECK((s.flags & OB_STAT_FLAG_EMITTER_USED) != 0u);
    CHECK(s.tiles_dispatched == expect_tiles);
  };

  check_prefill_tiles(64u,  1536u, 1536u, 1u);
  check_prefill_tiles(128u,  576u, 1536u, 2u);
  check_prefill_tiles(5u,    576u, 1536u, 1u);
  check_prefill_tiles(65u,   576u, 1536u, 2u);
  check_prefill_tiles(64u,  8960u, 1536u, 6u);
  check_prefill_tiles(64u,  1536u, 8960u, 6u);

  // pack_fmt=0 (M=1) vs pack_fmt=1 (M=64) on the EXACT SAME weight buffer must NOT alias!
  {
    constexpr uint32_t K = 576u, N = 1536u, nb_k = K / OB_QK8_0;
    std::vector<float> a64(static_cast<size_t>(64u) * K, 0.12f);
    std::vector<float> c64(static_cast<size_t>(64u) * N, 0.0f);
    std::vector<ob_block_q8_0_t> b(static_cast<size_t>(N) * nb_k);
    const uint16_t dh = fp32_to_fp16(0.02f);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        ob_block_q8_0_t& blk = b[static_cast<size_t>(n) * nb_k + bi];
        blk.d = dh;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          blk.qs[j] = static_cast<int8_t>(static_cast<int32_t>((n * 5u + bi * 11u + j) % 51u) - 25);
        }
      }
    }
    const uint64_t m0 = ctx->weight_cache_misses;
    const uint64_t h0 = ctx->weight_cache_hits;
    ob_invoke_stats_t s {};

    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a64.data(), b.data(), c64.data(), 1u, K, N, &s) == OB_OK);
    CHECK(ctx->weight_cache_misses == m0 + 1u && ctx->weight_cache_hits == h0);
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a64.data(), b.data(), c64.data(), 64u, K, N, &s) == OB_OK);
    CHECK(ctx->weight_cache_misses == m0 + 2u && ctx->weight_cache_hits == h0);
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a64.data(), b.data(), c64.data(), 1u, K, N, &s) == OB_OK);
    CHECK(ctx->weight_cache_hits == h0 + 1u);
    CHECK(ob_gemm_matmul_f32_q8_0(ctx, a64.data(), b.data(), c64.data(), 64u, K, N, &s) == OB_OK);
    CHECK(ctx->weight_cache_hits == h0 + 2u);

    bool saw_fmt0 = false, saw_fmt1 = false;
    for (const auto& kv : ctx->weight_cache_map) {
      if (kv.second->ext2_bytes == static_cast<size_t>(K) * N) saw_fmt0 = true;
      if (kv.second->ext2_bytes == static_cast<size_t>(K) * N + 16u * N) saw_fmt1 = true;
    }
    CHECK(saw_fmt0 && saw_fmt1);
  }
  ob_context_destroy(ctx);

  // Optional: run full 5/5 external corpus bit-exact parity if OPENBUENOS_CORPUS_DIR is set
  const char* corpus_root = std::getenv("OPENBUENOS_CORPUS_DIR");
  if (!corpus_root || corpus_root[0] == '\0') {
    std::printf("[PASS] test_lt_64x_batched_prefill_parity (11/11 M=64 shapes + prefill tiling + pack_fmt separation; external 5-model parity skipped without OPENBUENOS_CORPUS_DIR)\n");
    return;
  }

  std::string kFcPath = std::string(corpus_root) + "/work/device-runner/fc.dgc0";
  size_t interp_sz = 0;
  CHECK(ob_gemm_extract_dive_interp_from_dgc0(kFcPath.c_str(), nullptr, 0u, &interp_sz) == OB_ERR_OVERFLOW);
  CHECK(interp_sz == 638977u);
  std::vector<uint8_t> dive_interp(interp_sz, 0u);
  CHECK(ob_gemm_extract_dive_interp_from_dgc0(
            kFcPath.c_str(), dive_interp.data(), dive_interp.size(), &interp_sz) == OB_OK);

  struct BpCase {
    const char* name;
    const char* rel_dir;
  };
  const BpCase cases[5] = {
      {"bp_64x512x512",   "/work/device-runner/phase6_perf/batched_probes"},
      {"bp_64x576x576",   "/work/device-runner/phase6_perf/batched_probes"},
      {"lt_64x576x1536",  "/work/device-runner/phase5b_live/large_tile_probes"},
      {"bp_64x1024x1024", "/work/device-runner/phase6_perf/batched_probes"},
      {"bp_64x1536x1536", "/work/device-runner/phase6_perf/batched_probes"},
  };

  int exact_chunks = 0;
  int exact_ext2 = 0;
  int exact_parse = 0;
  int passed = 0;

  for (const BpCase& bc : cases) {
    std::string dgc0_path = std::string(corpus_root) + bc.rel_dir + "/" + bc.name + ".dgc0";
    std::string tflite_path = std::string(corpus_root) + bc.rel_dir + "/" + bc.name + ".tflite";

    std::vector<uint8_t> ref_dgc0, ref_tflite;
    CHECK(read_file(dgc0_path.c_str(), &ref_dgc0));
    CHECK(read_file(tflite_path.c_str(), &ref_tflite));

    TfliteFcSpec tf{};
    CHECK(parse_tflite_fc(ref_tflite, &tf));
    CHECK(ob::is_lt_64x_shape(tf.M, tf.K, tf.N));

    ob::ParsedDgc0 ref_parsed{};
    std::string perr;
    CHECK(ob::parse_dgc0_buffer(ref_dgc0.data(), ref_dgc0.size(), &ref_parsed, &perr) == OB_OK);

    const uint8_t* ref_ext0_ptr = ref_dgc0.data() + ref_parsed.ext_dive_vm_ir.offset;
    const size_t ref_ext0_sz = static_cast<size_t>(ref_parsed.ext_dive_vm_ir.size);
    const uint8_t* ref_ext2_ptr = ref_dgc0.data() + ref_parsed.ext_params.offset;
    const size_t ref_ext2_sz = static_cast<size_t>(ref_parsed.ext_params.size);

    std::array<std::vector<uint8_t>, 7> ref_chunks;
    CHECK(extract_dive_chunks(ref_ext0_ptr, ref_ext0_sz, &ref_chunks, nullptr));

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
      std::fprintf(stderr, "FAIL lt64 synth %s (%u,%u,%u): %s\n",
                   bc.name, tf.M, tf.K, tf.N, serr.c_str());
      ++g_failures;
      return;
    }

    bool ext2_ok = (syn.ext2_params.size() == ref_ext2_sz &&
                    std::memcmp(syn.ext2_params.data(), ref_ext2_ptr, ref_ext2_sz) == 0);
    if (!ext2_ok) {
      std::fprintf(stderr, "DIFF lt64 ext2 %s: syn=%zu ref=%zu\n",
                   bc.name, syn.ext2_params.size(), ref_ext2_sz);
    } else {
      ++exact_ext2;
    }

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
        size_t first_diff = 0;
        size_t lim = std::min(sc.size(), rc.size());
        while (first_diff < lim && sc[first_diff] == rc[first_diff]) ++first_diff;
        std::fprintf(stderr, "DIFF lt64 c%zu %s (%u,%u,%u): syn=%zu ref=%zu first_diff=%zu (syn=0x%02x ref=0x%02x)\n",
                     ci, bc.name, tf.M, tf.K, tf.N, sc.size(), rc.size(), first_diff,
                     first_diff < sc.size() ? sc[first_diff] : 0u,
                     first_diff < rc.size() ? rc[first_diff] : 0u);
        break;
      }
    }
    if (chunks_ok) ++exact_chunks;

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
        built_parsed.has_params &&
        built_parsed.ext_dive_image.size == ref_parsed.ext_dive_image.size &&
        built_parsed.ext_params.size == ref_parsed.ext_params.size &&
        (built_parsed.ext_dive_vm_ir.offset % 64u) == 0u &&
        (built_parsed.ext_dive_image.offset % 64u) == 0u &&
        (built_parsed.ext_params.offset % 64u) == 0u;
    if (!parse_ok) {
      std::fprintf(stderr, "DIFF lt64 parse %s: f14=%u/%u scr=%llu/%llu\n",
                   bc.name, built_parsed.exec_field_14, ref_parsed.exec_field_14,
                   static_cast<unsigned long long>(built_parsed.dive_scratch_size),
                   static_cast<unsigned long long>(ref_parsed.dive_scratch_size));
    } else {
      ++exact_parse;
    }

    if (ext2_ok && chunks_ok && parse_ok) ++passed;
  }

  CHECK(exact_ext2 == 5);
  CHECK(exact_chunks == 5);
  CHECK(exact_parse == 5);
  CHECK(passed == 5);
  std::printf("[PASS] test_lt_64x_batched_prefill_parity (5/5 BIT-EXACT -> 92/92 TOTAL! + 6 gen shapes + prefill tiling + pack_fmt separation)\n");
}

// ---------------------------------------------------------------------------
// Test 16: Native Q4_0 + Shared-A Batched GEMM + DVFS/WakeLock + FD-Safe Caps
// ---------------------------------------------------------------------------
void test_quant_batch_q4_0_dvfs_and_fd_caps() {
  std::vector<uint8_t> synth_interp = make_synthetic_dive_interp(0x5Au);
  TempBinaryFile interp_file;
  CHECK(interp_file.write(synth_interp.data(), synth_interp.size()));

  // 1. DVFS power profiles & Litebuf FW WakeLock state machine
  ::unsetenv("OPENBUENOS_PERF_MODE");
  ::unsetenv("OPENBUENOS_WAKELOCK");
  ::unsetenv("OPENBUENOS_FW_PRIORITY");
  ::unsetenv("OPENBUENOS_CORE_POWER_STATE");
  ::unsetenv("OPENBUENOS_MEM_POWER_STATE");
  ::unsetenv("OPENBUENOS_QOS_CLASS");

  ob_config_t cfg = ob_config_default();
  CHECK(cfg.perf_mode == 5u);
  cfg.mock_mode = 1;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 128u;
  ob_context_t* ctx = nullptr;
  CHECK(ob_context_create(&cfg, &ctx) == OB_OK);

  CHECK(ctx->cfg.perf_mode == 5u);
  CHECK(ctx->fw_priority == 0x01u);
  CHECK(ctx->cfg.core_power_state == 0x19u);
  CHECK(ctx->cfg.mem_power_state == 0x15u);
  CHECK(ctx->cfg.qos_class == 0x02u);
  CHECK(ctx->fw_wakelock_held == true);
  CHECK(ctx->fw_wakelock_core == 0x19u);
  CHECK(ctx->fw_wakelock_mem == 0x15u);
  CHECK(ctx->fw_wakelock_bitset == 0x04u);

  CHECK(ob_context_set_perf_mode(ctx, 0u) == OB_OK);
  CHECK(ctx->cfg.perf_mode == 0u);
  CHECK(ctx->fw_priority == 0x0bu);
  CHECK(ctx->cfg.core_power_state == 0x03u);
  CHECK(ctx->cfg.mem_power_state == 0x01u);
  CHECK(ctx->cfg.qos_class == 0x01u);
  CHECK(ctx->fw_wakelock_held == false);

  CHECK(ob_context_set_perf_mode(ctx, 4u) == OB_OK);
  CHECK(ctx->cfg.perf_mode == 4u);
  CHECK(ctx->fw_priority == 0x00u);
  CHECK(ctx->cfg.core_power_state == 0x19u);
  CHECK(ctx->cfg.mem_power_state == 0x0du);
  CHECK(ctx->cfg.qos_class == 0x01u);
  CHECK(ctx->fw_wakelock_held == true);
  CHECK(ctx->fw_wakelock_core == 0x19u);
  CHECK(ctx->fw_wakelock_mem == 0x0du);

  CHECK(ob_context_set_perf_mode(ctx, 6u) == OB_ERR_INVALID_ARG);
  CHECK(ctx->cfg.perf_mode == 4u);
  CHECK(ob_context_set_perf_mode(ctx, 5u) == OB_OK);
  CHECK(ctx->cfg.perf_mode == 5u && ctx->cfg.mem_power_state == 0x15u);

  ::setenv("OPENBUENOS_PERF_MODE", "2", 1);
  {
    ob_config_t env_cfg = ob_config_default();
    env_cfg.mock_mode = 1;
    ob_context_t* env_ctx = nullptr;
    CHECK(ob_context_create(&env_cfg, &env_ctx) == OB_OK);
    CHECK(env_ctx->cfg.perf_mode == 2u);
    CHECK(env_ctx->fw_priority == 0x0bu);
    CHECK(env_ctx->cfg.core_power_state == 0x16u);
    CHECK(env_ctx->cfg.mem_power_state == 0x0du);
    CHECK(env_ctx->fw_wakelock_held == true);
    ob_context_destroy(env_ctx);
  }
  ::unsetenv("OPENBUENOS_PERF_MODE");

  CHECK(ob_gemm_register_builtin_emitter(ctx, interp_file.path()) == OB_OK);

  // 2. Native Q4_0 + Shared-A Batched GEMM (Q+K+V and gate(SiLU)+up)
  std::mt19937 rng(20261121u);
  std::normal_distribution<float> gdist(0.0f, 1.0f);

  auto make_q4_matrix = [&](uint32_t K, uint32_t N,
                            std::vector<ob_block_q4_0_t>* out_q4,
                            std::vector<float>* out_deq) {
    const uint32_t nb_k = K / OB_QK8_0;
    out_q4->resize(static_cast<size_t>(N) * nb_k);
    out_deq->resize(static_cast<size_t>(N) * K);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        float vals[OB_QK8_0];
        float am = 0.0f;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          vals[j] = gdist(rng);
          float av = std::fabs(vals[j]);
          if (av > am) am = av;
        }
        float d_f = (am > 0.0f) ? (am / 7.0f) : 1.0f;
        uint16_t dh = fp32_to_fp16(d_f);
        float d_exact = fp16_to_fp32_ref(dh);
        ob_block_q4_0_t& blk = (*out_q4)[static_cast<size_t>(n) * nb_k + bi];
        blk.d = dh;
        for (uint32_t j = 0; j < 16u; ++j) {
          long q0 = (d_exact > 0.0f) ? std::lround(vals[j] / d_exact) : 0L;
          long q1 = (d_exact > 0.0f) ? std::lround(vals[j + 16u] / d_exact) : 0L;
          if (q0 < -8) q0 = -8;
          if (q0 > 7) q0 = 7;
          if (q1 < -8) q1 = -8;
          if (q1 > 7) q1 = 7;
          blk.qs[j] = static_cast<uint8_t>((q0 + 8) | ((q1 + 8) << 4));
          (*out_deq)[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j] = d_exact * static_cast<float>(q0);
          (*out_deq)[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j + 16u] = d_exact * static_cast<float>(q1);
        }
      }
    }
  };

  auto check_f32_close = [&](const std::vector<float>& got,
                             const std::vector<float>& ref,
                             double max_nmse) -> bool {
    double mse = 0.0, re = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
      mse += d * d;
      re += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (mse / std::max(re, 1e-30)) < max_nmse;
  };

  // 2a. Q + K + V shared-A batch at M=1 (Q4_0) + verify exact unpacked ext[2] DMA-BUF NMSE < 1e-3
  {
    constexpr uint32_t M = 1u, K = 576u, Nq = 576u, Nkv = 192u;
    std::vector<float> a1(K), a2(K);
    for (float& v : a1) v = gdist(rng);
    for (float& v : a2) v = gdist(rng);

    std::vector<ob_block_q4_0_t> bq, bk, bv;
    std::vector<float> dq, dk, dv;
    make_q4_matrix(K, Nq, &bq, &dq);
    make_q4_matrix(K, Nkv, &bk, &dk);
    make_q4_matrix(K, Nkv, &bv, &dv);

    std::vector<float> cq(Nq, 0.0f), ck(Nkv, 0.0f), cv(Nkv, 0.0f);
    ob_gemm_f32_quant_job_t jobs[3] {};
    jobs[0] = {bq.data(), cq.data(), Nq,  OB_ACT_NONE, 1u, {0, 0, 0}};
    jobs[1] = {bk.data(), ck.data(), Nkv, OB_ACT_NONE, 1u, {0, 0, 0}};
    jobs[2] = {bv.data(), cv.data(), Nkv, OB_ACT_NONE, 1u, {0, 0, 0}};

    const uint64_t wm0 = ctx->weight_cache_misses;
    const uint64_t wh0 = ctx->weight_cache_hits;
    ob_invoke_stats_t s1 {}, s2 {};
    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a1.data(), M, K, jobs, 3u, &s1) == OB_OK);
    CHECK((s1.flags & OB_STAT_FLAG_EMITTER_USED) != 0u);
    CHECK(s1.tiles_dispatched == 3u);
    CHECK(ctx->weight_cache_misses == wm0 + 3u && ctx->weight_cache_hits == wh0);

    auto compute_ref = [&](const std::vector<float>& a_in, const std::vector<float>& b_d,
                           uint32_t mm, uint32_t kk, uint32_t nn, uint32_t act) {
      std::vector<float> out(static_cast<size_t>(mm) * nn, 0.0f);
      for (uint32_t m = 0; m < mm; ++m) {
        for (uint32_t n = 0; n < nn; ++n) {
          double acc = 0.0;
          for (uint32_t k = 0; k < kk; ++k) {
            acc += static_cast<double>(a_in[static_cast<size_t>(m) * kk + k]) *
                   static_cast<double>(b_d[static_cast<size_t>(n) * kk + k]);
          }
          if (act == OB_ACT_SILU) {
            acc = acc / (1.0 + std::exp(-acc));
          } else if (act == OB_ACT_GELU) {
            acc = 0.5 * acc * (1.0 + std::tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)));
          }
          out[static_cast<size_t>(m) * nn + n] = static_cast<float>(acc);
        }
      }
      return out;
    };
    CHECK(check_f32_close(cq, compute_ref(a1, dq, M, K, Nq, OB_ACT_NONE), 1e-10));
    CHECK(check_f32_close(ck, compute_ref(a1, dk, M, K, Nkv, OB_ACT_NONE), 1e-10));
    CHECK(check_f32_close(cv, compute_ref(a1, dv, M, K, Nkv, OB_ACT_NONE), 1e-10));

    ob::CachedWeightTile* q_tile = nullptr;
    for (const auto& kv : ctx->weight_cache_map) {
      if (kv.second->hw_k == K && kv.second->hw_n == Nq &&
          kv.second->row_s_b.size() == Nq && kv.second->ext2_bytes == static_cast<size_t>(K) * Nq) {
        q_tile = kv.second;
        break;
      }
    }
    CHECK(q_tile != nullptr);
    std::vector<uint8_t> raw_ext2(q_tile->ext2_bytes, 0u);
    CHECK(ob_buffer_read(q_tile->ext2_buf, raw_ext2.data(), raw_ext2.size(), 0u) == OB_OK);
    std::vector<int8_t> aq(K);
    float sa = 0.0f;
    std::vector<std::pair<uint32_t, float>> outliers;
    ob::quantize_activation_row_split(a1.data(), K, aq.data(), &sa, &outliers);
    const uint32_t D = q_tile->scale_d;
    std::vector<float> tpu_sim_q(Nq, 0.0f);
    for (uint32_t g = 0; g < Nq / 32u; ++g) {
      const uint8_t* gbase = raw_ext2.data() + static_cast<size_t>(g) * (K / 4u) * 128u;
      for (uint32_t r = 0; r < 32u; ++r) {
        const uint32_t n = g * 32u + r;
        int32_t acc = 0;
        for (uint32_t kb = 0; kb < K / 4u; ++kb) {
          for (uint32_t j = 0; j < 4u; ++j) {
            int8_t w = static_cast<int8_t>(static_cast<int32_t>(gbase[kb * 128u + r * 4u + j]) - 128);
            acc += static_cast<int32_t>(aq[kb * 4u + j]) * static_cast<int32_t>(w);
          }
        }
        int32_t qr = static_cast<int32_t>(std::nearbyint(static_cast<float>(acc) / static_cast<float>(D)));
        qr = std::max(-128, std::min(127, qr));
        float val = static_cast<float>(qr) * static_cast<float>(D) * sa * q_tile->row_s_b[n];
        for (const auto& ov : outliers) {
          val += ov.second * dq[static_cast<size_t>(n) * K + ov.first];
        }
        tpu_sim_q[n] = val;
      }
    }
    CHECK(check_f32_close(tpu_sim_q, compute_ref(a1, dq, M, K, Nq, OB_ACT_NONE), 1e-3));

    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a2.data(), M, K, jobs, 3u, &s2) == OB_OK);
    CHECK(s2.tiles_dispatched == 3u && s2.cache_hits == 3u);
    CHECK(ctx->weight_cache_misses == wm0 + 3u && ctx->weight_cache_hits == wh0 + 3u);
  }

  // 2b. Gate (Q4_0 + OB_ACT_SILU) + Up (Q4_0 + OB_ACT_NONE) shared-A batch at M=64 (lt_64x!)
  {
    constexpr uint32_t M = 64u, K = 576u, N = 1536u;
    std::vector<float> a(static_cast<size_t>(M) * K);
    for (float& v : a) v = gdist(rng) * 0.25f;
    std::vector<ob_block_q4_0_t> bg, bu;
    std::vector<float> dg, du;
    make_q4_matrix(K, N, &bg, &dg);
    make_q4_matrix(K, N, &bu, &du);
    std::vector<float> cg(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> cu(static_cast<size_t>(M) * N, 0.0f);

    ob_gemm_f32_quant_job_t jobs[2] {};
    jobs[0] = {bg.data(), cg.data(), N, OB_ACT_SILU, 1u, {0, 0, 0}};
    jobs[1] = {bu.data(), cu.data(), N, OB_ACT_NONE, 1u, {0, 0, 0}};

    ob_invoke_stats_t s1 {}, s2 {};
    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 2u, &s1) == OB_OK);
    CHECK(s1.tiles_dispatched == 2u);
    for (uint32_t m : {0u, 31u, 63u}) {
      for (uint32_t n = 0; n < 32u; ++n) {
        double acc_g = 0.0, acc_u = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
          double av = static_cast<double>(a[static_cast<size_t>(m) * K + k]);
          acc_g += av * static_cast<double>(dg[static_cast<size_t>(n) * K + k]);
          acc_u += av * static_cast<double>(du[static_cast<size_t>(n) * K + k]);
        }
        double ref_g = acc_g / (1.0 + std::exp(-acc_g));
        CHECK(std::fabs(static_cast<double>(cg[static_cast<size_t>(m) * N + n]) - ref_g) < 1e-4);
        CHECK(std::fabs(static_cast<double>(cu[static_cast<size_t>(m) * N + n]) - acc_u) < 1e-4);
      }
    }
    const uint64_t wh_before = ctx->weight_cache_hits;
    CHECK(ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 2u, &s2) == OB_OK);
    CHECK(s2.tiles_dispatched == 2u && s2.cache_hits == 2u);
    CHECK(ctx->weight_cache_hits == wh_before + 2u);
  }

  // 3. FD-safe weight_cache_max_entries & buffer pool cap enforcement
  {
    ob_config_t cap_cfg = ob_config_default();
    cap_cfg.mock_mode = 1;
    cap_cfg.max_inflight = 16u;
    cap_cfg.graph_cache_capacity = 32u;
    ob_context_t* cap_ctx = nullptr;
    CHECK(ob_context_create(&cap_cfg, &cap_ctx) == OB_OK);
    CHECK(ob_gemm_register_builtin_emitter(cap_ctx, interp_file.path()) == OB_OK);
    cap_ctx->weight_cache_max_entries = 12u;

    constexpr uint32_t K = 64u, N = 64u, nb_k = K / OB_QK8_0;
    std::vector<float> a(K, 0.25f), c(N, 0.0f);
    std::vector<ob_block_q4_0_t> b(static_cast<size_t>(N) * nb_k);
    for (uint32_t iter = 0; iter < 30u; ++iter) {
      const uint16_t dh = fp32_to_fp16(0.05f + 0.003f * static_cast<float>(iter));
      for (size_t i = 0; i < b.size(); ++i) {
        b[i].d = dh;
        for (uint32_t j = 0; j < 16u; ++j) {
          b[i].qs[j] = static_cast<uint8_t>(((iter + i + j) & 0x0Fu) | ((((iter * 3u + j) & 0x0Fu)) << 4));
        }
      }
      ob_gemm_f32_quant_job_t job {b.data(), c.data(), N, OB_ACT_NONE, 1u, {0, 0, 0}};
      ob_invoke_stats_t s {};
      CHECK(ob_gemm_matmul_f32_quant_batch(cap_ctx, a.data(), 1u, K, &job, 1u, &s) == OB_OK);
      CHECK(cap_ctx->weight_cache_map.size() <= 12u);
      CHECK(cap_ctx->weight_cache_lru.size() == cap_ctx->weight_cache_map.size());
    }
    CHECK(cap_ctx->weight_cache_map.size() == 12u);
    CHECK(cap_ctx->weight_cache_misses == 30u);

    std::vector<ob_buffer_t*> slabs;
    for (uint32_t i = 0; i < 280u; ++i) {
      ob_buffer_t* sb = nullptr;
      CHECK(ob_buffer_alloc(cap_ctx, 4096u, OB_BUF_TO_DEVICE, &sb) == OB_OK);
      slabs.push_back(sb);
    }
    for (ob_buffer_t* sb : slabs) ob_buffer_free(sb);
    CHECK(cap_ctx->pool_current_count <= ob::kDefaultBufferPoolMaxCount);
    ob_context_destroy(cap_ctx);
  }

  ob_context_destroy(ctx);
  std::printf("[PASS] test_quant_batch_q4_0_dvfs_and_fd_caps (PM=0..5+FW-WL OK, Q4_0 ext2 NMSE<1e-3, QKV/Gate+Up shared-A M=1&64 OK, FD caps<=12/256)\n");
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
  test_lt_1x_6_parity();
  test_mode_a_fused_silu_gelu();
  test_lt_1x_tiling_and_cache();
  test_outlier_spike_and_rms_shift();
  test_lt_64x_batched_prefill_parity();
  test_quant_batch_q4_0_dvfs_and_fd_caps();

  if (g_failures == 0) {
    std::printf("ALL LIBOPENBUENOS HOST TESTS PASSED (16/16 suites)\n");
    return 0;
  }
  std::fprintf(stderr, "LIBOPENBUENOS HOST TESTS FAILED (%d failures)\n", g_failures);
  return 1;
}
