// Standalone Android ARM64 PIE self-test binary for libopenbuenos.
// Links ONLY against libopenbuenos.a + libc/libm (zero Google/vendor .so).
#include "openbuenos.h"
#include "ob_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
  std::fprintf(stderr,
      "Usage:\n"
      "  %s --mock\n"
      "  %s --dgc0 <model.dgc0> [--in <in.bin>] [--ref <ref.bin>] [--iters <N>]\n"
      "  %s --emit-gemm <M> <K> <N> [--dive-interp <path>] [--iters <N>] [--mock]\n"
      "  %s --probe-dyn-weights [--dive-interp <path>] [--mock]\n",
      argv0, argv0, argv0, argv0);
}

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

uint16_t fp32_to_fp16(float f) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = bits & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float fp16_to_fp32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x03FFu;
  uint32_t bits = 0u;
  if (exp == 0u) {
    bits = sign;
  } else if (exp == 31u) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint32_t selftest_scale_D_for_tile(uint32_t tk) {
  uint64_t l1 = (tk >= 8u) ? static_cast<uint64_t>(2u * (tk / 8u)) : 1ULL;
  double l2 = std::sqrt(static_cast<double>(l1));
  double hard_cap = static_cast<double>(l1) * (128.0 / 127.0);
  double stat_est = 2.6 * l2;
  double raw_d = std::min(hard_cap, stat_est);
  if (raw_d <= 1.0) return 1u;
  if (raw_d <= 8.0) {
    uint32_t d = static_cast<uint32_t>(std::ceil(raw_d));
    return d < 1u ? 1u : d;
  }
  int p = static_cast<int>(std::floor(std::log2(raw_d)));
  if (p < 3) p = 3;
  if (p > 24) return 1u << 24;
  uint32_t step = 1u << static_cast<uint32_t>(p - 3);
  uint64_t d = static_cast<uint64_t>(std::ceil(raw_d / static_cast<double>(step))) *
               static_cast<uint64_t>(step);
  return static_cast<uint32_t>(d);
}

int run_mock_smoke() {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = 1;
  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "mock ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }

  ob_buffer_t* buf = nullptr;
  st = ob_buffer_alloc(ctx, 4096u, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &buf);
  if (st != OB_OK || !buf || ob_buffer_iova(buf) == 0u) {
    std::fprintf(stderr, "mock ob_buffer_alloc failed: %s\n", ob_status_string(st));
    ob_context_destroy(ctx);
    return 1;
  }
  ob_buffer_free(buf);

  // Verify weight packing + tiled int8 matmul in mock mode
  std::vector<int8_t> a(16, 2);
  std::vector<int8_t> b(16 * 16, 3);
  std::vector<int32_t> bias(16, 5);
  std::vector<int32_t> c(16, 0);
  ob_invoke_stats_t stats {};
  st = ob_gemm_matmul_int8(ctx, a.data(), b.data(), bias.data(), c.data(),
                           1u, 16u, 16u, 0, 0, &stats);
  if (st != OB_OK) {
    std::fprintf(stderr, "mock ob_gemm_matmul_int8 failed: %s\n", ob_status_string(st));
    ob_context_destroy(ctx);
    return 1;
  }
  for (int32_t v : c) {
    if (v != 16 * 2 * 3 + 5) {
      std::fprintf(stderr, "mock matmul mismatch: got %d expected %d\n", v, 16 * 2 * 3 + 5);
      ob_context_destroy(ctx);
      return 1;
    }
  }

  std::printf("openbuenos_selftest: mock OK (credits=%u tiles=%u)\n",
              ob_context_credit_size(ctx), stats.tiles_dispatched);
  ob_context_destroy(ctx);
  return 0;
}

int run_emit_gemm(uint32_t M, uint32_t K, uint32_t N,
                  int iters, const char* dive_interp_path, bool mock) {
  if (M == 0u || K == 0u || N == 0u) {
    std::fprintf(stderr, "--emit-gemm requires M, K, N > 0\n");
    return 2;
  }

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = mock ? 1 : 0;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 512u;

  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }

  st = ob_gemm_register_builtin_emitter(ctx, dive_interp_path);
  if (st != OB_OK) {
    std::fprintf(stderr, "ob_gemm_register_builtin_emitter failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  st = ob_gemm_set_tile_limits(ctx, 16u, 512u, 128u);
  if (st != OB_OK) {
    std::fprintf(stderr, "ob_gemm_set_tile_limits failed: %s\n", ob_status_string(st));
    ob_context_destroy(ctx);
    return 1;
  }

  // Construct deterministic non-trivial B[N, K] where every 8-element K-block
  // has one +1 and one -1 at distinct row- and block-dependent positions.
  // This makes every row in any K-tile share the exact same L1/L2/D while
  // exercising non-trivial MACs and non-power-of-two real_mul = 1/D on silicon.
  std::vector<int8_t> b_nk(static_cast<size_t>(N) * K, 0);
  const uint32_t n_blk8 = K / 8u;
  for (uint32_t n = 0; n < N; ++n) {
    int8_t* brow = b_nk.data() + static_cast<size_t>(n) * K;
    if (n_blk8 > 0u) {
      for (uint32_t blk = 0; blk < n_blk8; ++blk) {
        uint32_t pos_p = blk * 8u + ((n + blk * 3u) & 7u);
        uint32_t pos_m = blk * 8u + ((n + blk * 3u + 4u) & 7u);
        brow[pos_p] = 1;
        brow[pos_m] = -1;
      }
    } else {
      brow[n % K] = (n & 1u) ? -1 : 1;
    }
  }

  // Build A[M, K] with exact multiples of each (m_slice, k_tile)'s scale D so
  // RNE(acc_tile / D) * D == acc_tile identically with zero rounding loss and
  // |acc_tile / D| << 127 (zero clipping).
  std::vector<int8_t> a_mk(static_cast<size_t>(M) * K, 0);
  {
    uint32_t m0 = 0u;
    while (m0 < M) {
      uint32_t tm = std::min<uint32_t>(M - m0, 16u);
      const uint32_t k_step = (tm == 1u) ? 512u : 64u;
      for (uint32_t k0 = 0; k0 < K; k0 += k_step) {
        const uint32_t tk = std::min(k_step, K - k0);
        const uint32_t D = selftest_scale_D_for_tile(tk);
        const int32_t max_q = (D * 2u <= 120u && tk <= 64u) ? 2 : 1;
        for (uint32_t mi = 0; mi < tm; ++mi) {
          const uint32_t m = m0 + mi;
          int8_t* arow = a_mk.data() + static_cast<size_t>(m) * K;
          for (uint32_t ki = 0; ki < tk; ++ki) {
            const uint32_t k = k0 + ki;
            int32_t span = 2 * max_q + 1;
            uint32_t h = (m * 2654435761u) ^ (k * 2246822519u) ^ 0x9E3779B9u;
            h = (h ^ (h >> 16)) * 0x85EBCA6Bu;
            h ^= (h >> 13);
            int32_t q = static_cast<int32_t>(h % static_cast<uint32_t>(span)) - max_q;
            arow[k] = static_cast<int8_t>(q * static_cast<int32_t>(D));
          }
        }
      }
      m0 += tm;
    }
  }

  std::vector<int32_t> bias_n(N, 0);
  for (uint32_t n = 0; n < N; ++n) {
    bias_n[n] = static_cast<int32_t>((n * 13u) % 41u) - 20;
  }

  // Exact CPU reference
  std::vector<int32_t> c_ref(static_cast<size_t>(M) * N, 0);
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      int32_t acc = bias_n[n];
      const int8_t* arow = a_mk.data() + static_cast<size_t>(m) * K;
      const int8_t* brow = b_nk.data() + static_cast<size_t>(n) * K;
      for (uint32_t k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(arow[k]) * static_cast<int32_t>(brow[k]);
      }
      c_ref[static_cast<size_t>(m) * N + n] = acc;
    }
  }

  std::vector<int32_t> c_out(static_cast<size_t>(M) * N, 0);
  ob_invoke_stats_t last_stats {};
  int64_t worst_err = 0;
  bool ok = true;

  for (int it = 0; it < iters; ++it) {
    std::fill(c_out.begin(), c_out.end(), 0);
    st = ob_gemm_matmul_int8(ctx, a_mk.data(), b_nk.data(), bias_n.data(), c_out.data(),
                             M, K, N, 0, 0, &last_stats);
    if (st != OB_OK) {
      std::fprintf(stderr, "ob_gemm_matmul_int8 iter=%d failed: %s (%s)\n",
                   it, ob_status_string(st), ob_context_last_error(ctx));
      ok = false;
      break;
    }
    for (size_t i = 0; i < c_ref.size(); ++i) {
      int64_t d = static_cast<int64_t>(c_out[i]) - static_cast<int64_t>(c_ref[i]);
      if (d < 0) d = -d;
      if (d > worst_err) worst_err = d;
    }
  }

  if (ok) {
    if ((last_stats.flags & OB_STAT_FLAG_EMITTER_USED) == 0u) {
      std::fprintf(stderr, "Expected OB_STAT_FLAG_EMITTER_USED to be set\n");
      ok = false;
    }
    if (!mock && last_stats.hw_execution_us == 0u) {
      std::fprintf(stderr, "Expected non-zero TPU hw_execution_us on hardware\n");
      ok = false;
    }
    if (iters > 1 && last_stats.cache_hits != last_stats.tiles_dispatched) {
      std::fprintf(stderr, "Expected 100%% cache hits on iter %d, got %u/%u\n",
                   iters, last_stats.cache_hits, last_stats.tiles_dispatched);
      ok = false;
    }
    if (worst_err != 0) {
      std::fprintf(stderr, "INT8 GEMM mismatch: max_abs_err=%lld (expected 0)\n",
                   static_cast<long long>(worst_err));
      ok = false;
    }
  }

  // Also run F32 x Q8_0 live NMSE + token-invariant cache check when K % 32 == 0
  double f32_nmse = -1.0;
  uint32_t f32_tok2_hits = 0u;
  uint32_t f32_tok2_tiles = 0u;
  if (ok && (K % OB_QK8_0) == 0u) {
    const uint32_t nb_k = K / OB_QK8_0;
    std::vector<float> af1(static_cast<size_t>(M) * K);
    std::vector<float> af2(static_cast<size_t>(M) * K);
    std::vector<ob_block_q8_0_t> bq8(static_cast<size_t>(N) * nb_k);
    std::vector<float> b_deq(static_cast<size_t>(N) * K);

    uint32_t prng = 0x12345678u;
    auto next_u = [&]() -> float {
      prng = prng * 1664525u + 1013904223u;
      return (static_cast<float>(prng >> 8) / static_cast<float>(1u << 23)) - 1.0f;
    };

    for (size_t i = 0; i < af1.size(); ++i) {
      af1[i] = next_u();
      af2[i] = next_u() * 17.0f;  // Different token scale s_in!
    }
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        float tmp[OB_QK8_0];
        float am = 0.0f;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          tmp[j] = next_u();
          float av = std::fabs(tmp[j]);
          if (av > am) am = av;
        }
        uint16_t dh = fp32_to_fp16(am > 0.0f ? (am / 127.0f) : 1.0f);
        float d = fp16_to_fp32(dh);
        ob_block_q8_0_t& blk = bq8[static_cast<size_t>(n) * nb_k + bi];
        blk.d = dh;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          long q = (d > 0.0f) ? std::lround(tmp[j] / d) : 0L;
          if (q < -127) q = -127;
          if (q > 127) q = 127;
          blk.qs[j] = static_cast<int8_t>(q);
          b_deq[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j] = d * static_cast<float>(blk.qs[j]);
        }
      }
    }

    std::vector<float> cf1(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> cf2(static_cast<size_t>(M) * N, 0.0f);
    ob_invoke_stats_t fs1 {}, fs2 {};
    ob_status_t st1 = ob_gemm_matmul_f32_q8_0(ctx, af1.data(), bq8.data(), cf1.data(), M, K, N, &fs1);
    ob_status_t st2 = (st1 == OB_OK)
        ? ob_gemm_matmul_f32_q8_0(ctx, af2.data(), bq8.data(), cf2.data(), M, K, N, &fs2)
        : st1;
    if (st1 != OB_OK || st2 != OB_OK) {
      std::fprintf(stderr, "ob_gemm_matmul_f32_q8_0 failed: %s (%s)\n",
                   ob_status_string(st1 != OB_OK ? st1 : st2), ob_context_last_error(ctx));
      ok = false;
    } else {
      f32_tok2_hits = fs2.cache_hits;
      f32_tok2_tiles = fs2.tiles_dispatched;
      double mse = 0.0, ref_e = 0.0;
      for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
          double acc = 0.0;
          for (uint32_t k = 0; k < K; ++k) {
            acc += static_cast<double>(af2[static_cast<size_t>(m) * K + k]) *
                   static_cast<double>(b_deq[static_cast<size_t>(n) * K + k]);
          }
          double diff = static_cast<double>(cf2[static_cast<size_t>(m) * N + n]) - acc;
          mse += diff * diff;
          ref_e += acc * acc;
        }
      }
      f32_nmse = mse / std::max(ref_e, 1e-30);
      if (f32_tok2_hits != f32_tok2_tiles) {
        std::fprintf(stderr, "Expected token-invariant cache hits %u/%u on f32_q8_0, got %u/%u\n",
                     f32_tok2_tiles, f32_tok2_tiles, f32_tok2_hits, f32_tok2_tiles);
        ok = false;
      }
      const double nmse_thresh = (static_cast<size_t>(M) * N < 64u) ? 1.0e-3 : 5.0e-4;
      if (!(f32_nmse < nmse_thresh)) {
        std::fprintf(stderr, "F32xQ8_0 NMSE too high: %.3e (threshold %.1e)\n", f32_nmse, nmse_thresh);
        ok = false;
      }
    }
  }

  size_t cache_entries = 0;
  uint64_t cache_hits_tot = 0, cache_misses_tot = 0, cache_evict_tot = 0;
  ob_context_graph_cache_stats(ctx, &cache_entries, &cache_hits_tot, &cache_misses_tot, &cache_evict_tot);

  std::printf("openbuenos_selftest: ok=%d emit_gemm=%ux%ux%u iters=%d tiles=%u cache_hits=%u "
              "models=%zu hw_us=%llu wall_us=%llu max_abs_err=%lld f32_nmse=%.2e f32_tok2_hits=%u/%u\n",
              ok ? 1 : 0, M, K, N, iters,
              last_stats.tiles_dispatched, last_stats.cache_hits,
              cache_entries,
              static_cast<unsigned long long>(last_stats.hw_execution_us),
              static_cast<unsigned long long>(last_stats.wall_time_us),
              static_cast<long long>(worst_err),
              f32_nmse, f32_tok2_hits, f32_tok2_tiles);

  ob_context_destroy(ctx);
  return ok ? 0 : 1;
}

int run_probe_dyn_weights(const char* dive_interp_path, bool mock) {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = mock ? 1 : 0;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 64u;
  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "probe ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }
  st = ob_gemm_register_builtin_emitter(ctx, dive_interp_path);
  if (st != OB_OK) {
    std::fprintf(stderr, "probe ob_gemm_register_builtin_emitter failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  auto make_w = [](uint32_t N, uint32_t K, uint32_t seed) -> std::vector<int8_t> {
    std::vector<int8_t> w(static_cast<size_t>(N) * K, 0);
    uint32_t n_blk8 = K / 8u;
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t blk = 0; blk < n_blk8; ++blk) {
        uint32_t p = blk * 8u + ((n + blk * 3u + seed) & 7u);
        uint32_t m = blk * 8u + ((n + blk * 3u + seed + 4u) & 7u);
        w[static_cast<size_t>(n) * K + p] = 1;
        w[static_cast<size_t>(n) * K + m] = -1;
      }
    }
    return w;
  };
  auto make_a = [](uint32_t K, uint32_t D, uint32_t seed) -> std::vector<int8_t> {
    std::vector<int8_t> a(K, 0);
    for (uint32_t k = 0; k < K; ++k) {
      uint32_t h = (k * 2246822519u) ^ (seed * 2654435761u) ^ 0x9E3779B9u;
      h = (h ^ (h >> 16)) * 0x85EBCA6Bu;
      h ^= (h >> 13);
      int32_t q = static_cast<int32_t>(h % 3u) - 1;
      a[k] = static_cast<int8_t>(q * static_cast<int32_t>(D));
    }
    return a;
  };
  auto cpu_ref_q = [](const std::vector<int8_t>& a, const std::vector<int8_t>& w,
                      uint32_t N, uint32_t K, uint32_t D) -> std::vector<int8_t> {
    std::vector<int8_t> out(N, 0);
    float rm = 1.0f / static_cast<float>(D);
    for (uint32_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (uint32_t k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(a[k]) * static_cast<int32_t>(w[static_cast<size_t>(n) * K + k]);
      }
      int32_t q = static_cast<int32_t>(std::nearbyint(static_cast<float>(acc) * rm));
      out[n] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }
    return out;
  };
  auto max_err_i8 = [](const std::vector<int8_t>& a, const std::vector<int8_t>& b) -> int {
    int mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
      if (d < 0) d = -d;
      if (d > mx) mx = d;
    }
    return mx;
  };

  // --- Probe A & B: standard 1-in 1-param 1x16x16 graph with Execute param_slice swap + in-place write ---
  {
    constexpr uint32_t M = 1u, K = 16u, N = 16u;
    const uint32_t D = selftest_scale_D_for_tile(K);
    std::vector<int8_t> w1 = make_w(N, K, 1u);
    std::vector<int8_t> w2 = make_w(N, K, 5u);
    std::vector<int8_t> a = make_a(K, D, 7u);
    std::vector<int8_t> ref1 = cpu_ref_q(a, w1, N, K, D);
    std::vector<int8_t> ref2 = cpu_ref_q(a, w2, N, K, D);

    ob_gemm_desc_t desc {};
    desc.m_dim = M; desc.k_dim = K; desc.n_dim = N;
    desc.s_in = 1.0f; desc.s_w = 1.0f; desc.s_out = static_cast<float>(D);
    desc.weights_nk = w1.data();

    ob::GemmSynthResult parts{};
    std::string err;
    std::vector<uint8_t> dgc0;
    ob_status_t pst = ob::synthesize_gemm_parts(desc, &parts, &err);
    if (pst == OB_OK) {
      pst = ob::build_dgc0_from_parts(desc, parts,
                                      ctx->builtin_dive_interp.data(),
                                      ctx->builtin_dive_interp.size(),
                                      &dgc0, &err);
    }
    ob_graph_t* g = nullptr;
    if (pst == OB_OK) pst = ob_graph_register_dgc0(ctx, dgc0.data(), dgc0.size(), 0xA001u, &g);

    std::vector<uint8_t> ext2_w2 = ob::pack_ext2_for_tile(N, K, w2.data(), nullptr);
    ob_buffer_t *in_b = nullptr, *out_b = nullptr, *w2_b = nullptr;
    if (pst == OB_OK) pst = ob_buffer_alloc(ctx, K, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &in_b);
    if (pst == OB_OK) pst = ob_buffer_alloc(ctx, N, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &out_b);
    if (pst == OB_OK) pst = ob_buffer_alloc(ctx, ext2_w2.size(), OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &w2_b);
    if (pst == OB_OK) pst = ob_buffer_write(in_b, a.data(), a.size(), 0u);
    if (pst == OB_OK) pst = ob_buffer_write(w2_b, ext2_w2.data(), ext2_w2.size(), 0u);

    if (pst == OB_OK) {
      std::vector<int8_t> out(N, 0);
      ob_invoke_stats_t is {};
      // Baseline W1
      pst = ob_graph_invoke(g, &in_b, 1u, &out_b, 1u, &is);
      ob_buffer_read(out_b, out.data(), N, 0u);
      int base_err1 = max_err_i8(out, ref1);

      // Probe A: Execute param_slice override -> W2 on SAME model_handle
      ob_exec_handle_t eh = 0u;
      {
        std::lock_guard<std::mutex> lk(ctx->cache_mu);
        g->entry->ref_count += 1u;
      }
      ob_status_t ast = ob::device_submit_execute_async(ctx, g->entry, &in_b, 1u, &out_b, 1u, &eh, w2_b);
      if (ast != OB_OK) {
        std::lock_guard<std::mutex> lk(ctx->cache_mu);
        ob::graph_entry_unref_locked(ctx, g->entry);
      } else {
        ast = ob_graph_wait(ctx, eh, ctx->cfg.timeout_ms, &is);
      }
      ob_buffer_read(out_b, out.data(), N, 0u);
      int a_err2 = max_err_i8(out, ref2);
      int a_err1 = max_err_i8(out, ref1);
      std::printf("probe_A_exec_param_swap: st=%s base_err1=%d vs_W2_err=%d vs_W1_err=%d verdict=%s\n",
                  ob_status_string(ast), base_err1, a_err2, a_err1,
                  (ast == OB_OK && a_err2 == 0) ? "SWAP_WORKS" : "IGNORED_OR_FAIL");

      // Probe A2: 16 concurrent in-flight async param_slice swaps on ONE model_handle!
      {
        constexpr uint32_t kBurst = 16u;
        std::vector<ob_buffer_t*> wbufs(kBurst, nullptr), obufs(kBurst, nullptr);
        std::vector<std::vector<int8_t>> refs(kBurst);
        std::vector<ob_exec_handle_t> hs(kBurst, 0u);
        for (uint32_t i = 0; i < kBurst; ++i) {
          std::vector<int8_t> wi = make_w(N, K, 10u + i);
          std::vector<uint8_t> pi = ob::pack_ext2_for_tile(N, K, wi.data(), nullptr);
          refs[i] = cpu_ref_q(a, wi, N, K, D);
          ob_buffer_alloc(ctx, pi.size(), OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &wbufs[i]);
          ob_buffer_alloc(ctx, N, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &obufs[i]);
          ob_buffer_write(wbufs[i], pi.data(), pi.size(), 0u);
        }
        ob_status_t a16_st = OB_OK;
        for (uint32_t i = 0; i < kBurst && a16_st == OB_OK; ++i) {
          a16_st = ob::graph_submit_async_with_params(g, &in_b, 1u, &obufs[i], 1u, wbufs[i], &hs[i]);
        }
        int a16_err = 0;
        std::vector<int8_t> ob_tmp(N, 0);
        for (uint32_t i = 0; i < kBurst; ++i) {
          if (hs[i] != 0u) {
            ob_status_t wst = ob_graph_wait(ctx, hs[i], ctx->cfg.timeout_ms, nullptr);
            if (wst != OB_OK && a16_st == OB_OK) a16_st = wst;
            if (wst == OB_OK) {
              ob_buffer_read(obufs[i], ob_tmp.data(), N, 0u);
              a16_err = std::max(a16_err, max_err_i8(ob_tmp, refs[i]));
            }
          }
          ob_buffer_free(wbufs[i]);
          ob_buffer_free(obufs[i]);
        }
        std::printf("probe_A2_async16_param_swap: st=%s max_err=%d verdict=%s\n",
                    ob_status_string(a16_st), a16_err,
                    (a16_st == OB_OK && a16_err == 0) ? "ASYNC16_SWAP_WORKS" : "FAIL");
      }

      // Probe B: Overwrite g->entry->params_buf in-place with W2
      ob_status_t bst = ob_buffer_write(g->entry->params_buf, ext2_w2.data(), ext2_w2.size(), 0u);
      if (bst == OB_OK) bst = ob_graph_invoke(g, &in_b, 1u, &out_b, 1u, &is);
      ob_buffer_read(out_b, out.data(), N, 0u);
      int b_err2 = max_err_i8(out, ref2);
      int b_err1 = max_err_i8(out, ref1);
      std::printf("probe_B_inplace_param_write: st=%s vs_W2_err=%d vs_W1_err=%d verdict=%s\n",
                  ob_status_string(bst), b_err2, b_err1,
                  (bst == OB_OK && b_err2 == 0) ? "INPLACE_WORKS" : "STALE_OR_FAIL");
    } else {
      std::printf("probe_A_B_setup_failed: %s (%s)\n", ob_status_string(pst), ob_context_last_error(ctx));
    }
    ob_buffer_free(w2_b);
    ob_buffer_free(out_b);
    ob_buffer_free(in_b);
    ob_graph_release(g);
  }

  // --- Probe C / C2 / D_dyn: 2-input 0-param DIVE input_1 dynamic weights ---
  auto run_dyn_shape = [&](const char* label, uint32_t K, uint32_t N, bool test_async16) {
    constexpr uint32_t M = 1u;
    const uint32_t D = selftest_scale_D_for_tile(K);
    ob_gemm_desc_t desc {};
    desc.m_dim = M; desc.k_dim = K; desc.n_dim = N;
    desc.s_in = 1.0f; desc.s_w = 1.0f; desc.s_out = static_cast<float>(D);

    std::vector<uint8_t> dgc0;
    uint32_t ext2_bytes = 0u;
    std::string err;
    ob_status_t pst = ob::build_dgc0_dyn_weights(
        desc, ctx->builtin_dive_interp.data(), ctx->builtin_dive_interp.size(),
        &dgc0, &ext2_bytes, &err);
    ob_graph_t* g = nullptr;
    if (pst == OB_OK) {
      pst = ob_graph_register_dgc0(ctx, dgc0.data(), dgc0.size(),
                                   0xC000u ^ (K << 8) ^ N, &g);
    }
    if (pst != OB_OK) {
      std::printf("%s: register failed st=%s (%s %s)\n", label, ob_status_string(pst),
                  err.c_str(), ob_context_last_error(ctx));
      return;
    }

    std::vector<int8_t> a = make_a(K, D, 11u);
    std::vector<int8_t> w1 = make_w(N, K, 1u);
    std::vector<int8_t> w2 = make_w(N, K, 5u);
    std::vector<uint8_t> p1 = ob::pack_ext2_for_tile(N, K, w1.data(), nullptr);
    std::vector<uint8_t> p2 = ob::pack_ext2_for_tile(N, K, w2.data(), nullptr);
    std::vector<int8_t> r1 = cpu_ref_q(a, w1, N, K, D);
    std::vector<int8_t> r2 = cpu_ref_q(a, w2, N, K, D);

    ob_buffer_t *in0 = nullptr, *in_w1 = nullptr, *in_w2 = nullptr, *out0 = nullptr;
    ob_buffer_alloc(ctx, K, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &in0);
    ob_buffer_alloc(ctx, ext2_bytes, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &in_w1);
    ob_buffer_alloc(ctx, ext2_bytes, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &in_w2);
    ob_buffer_alloc(ctx, N, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &out0);
    ob_buffer_write(in0, a.data(), a.size(), 0u);
    ob_buffer_write(in_w1, p1.data(), p1.size(), 0u);
    ob_buffer_write(in_w2, p2.data(), p2.size(), 0u);

    ob_buffer_t* ins1[2] = {in0, in_w1};
    ob_buffer_t* ins2[2] = {in0, in_w2};
    ob_invoke_stats_t is1 {}, is2 {};
    std::vector<int8_t> o1(N, 0), o2(N, 0);
    ob_status_t st1 = ob_graph_invoke(g, ins1, 2u, &out0, 1u, &is1);
    if (st1 == OB_OK) ob_buffer_read(out0, o1.data(), N, 0u);
    ob_status_t st2 = (st1 == OB_OK) ? ob_graph_invoke(g, ins2, 2u, &out0, 1u, &is2) : st1;
    if (st2 == OB_OK) ob_buffer_read(out0, o2.data(), N, 0u);

    int e1 = max_err_i8(o1, r1);
    int e2 = max_err_i8(o2, r2);
    int async16_err = -1;
    ob_status_t ast = OB_OK;
    if (test_async16 && st1 == OB_OK && st2 == OB_OK) {
      constexpr uint32_t kBurst = 16u;
      std::vector<ob_buffer_t*> wbufs(kBurst, nullptr), obufs(kBurst, nullptr);
      std::vector<std::vector<int8_t>> refs(kBurst);
      std::vector<ob_exec_handle_t> hs(kBurst, 0u);
      for (uint32_t i = 0; i < kBurst; ++i) {
        std::vector<int8_t> wi = make_w(N, K, 10u + i);
        std::vector<uint8_t> pi = ob::pack_ext2_for_tile(N, K, wi.data(), nullptr);
        refs[i] = cpu_ref_q(a, wi, N, K, D);
        ob_buffer_alloc(ctx, ext2_bytes, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &wbufs[i]);
        ob_buffer_alloc(ctx, N, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &obufs[i]);
        ob_buffer_write(wbufs[i], pi.data(), pi.size(), 0u);
      }
      for (uint32_t i = 0; i < kBurst && ast == OB_OK; ++i) {
        ob_buffer_t* ins[2] = {in0, wbufs[i]};
        ast = ob_graph_submit_async(g, ins, 2u, &obufs[i], 1u, &hs[i]);
      }
      async16_err = 0;
      std::vector<int8_t> ob_tmp(N, 0);
      for (uint32_t i = 0; i < kBurst; ++i) {
        if (hs[i] != 0u) {
          ob_status_t wst = ob_graph_wait(ctx, hs[i], ctx->cfg.timeout_ms, nullptr);
          if (wst != OB_OK && ast == OB_OK) ast = wst;
          if (wst == OB_OK) {
            ob_buffer_read(obufs[i], ob_tmp.data(), N, 0u);
            async16_err = std::max(async16_err, max_err_i8(ob_tmp, refs[i]));
          }
        }
        ob_buffer_free(wbufs[i]);
        ob_buffer_free(obufs[i]);
      }
    }

    std::printf("%s (%ux%ux%u): st=%s/%s err_W1=%d err_W2=%d hw_us=%llu/%llu async16_st=%s async16_err=%d\n",
                label, M, K, N, ob_status_string(st1), ob_status_string(st2),
                e1, e2,
                static_cast<unsigned long long>(is1.hw_execution_us),
                static_cast<unsigned long long>(is2.hw_execution_us),
                ob_status_string(ast), async16_err);

    ob_buffer_free(out0);
    ob_buffer_free(in_w2);
    ob_buffer_free(in_w1);
    ob_buffer_free(in0);
    ob_graph_release(g);
  };

  run_dyn_shape("probe_C_input1_dyn", 16u, 16u, true);
  run_dyn_shape("probe_C2_input1_dyn_stream", 512u, 64u, false);
  run_dyn_shape("probe_D_input1_dyn_n128", 512u, 128u, false);

  // Also test standard 1-in 1-param 1x512x128 directly
  {
    constexpr uint32_t M = 1u, K = 512u, N = 128u;
    const uint32_t D = selftest_scale_D_for_tile(K);
    std::vector<int8_t> a = make_a(K, D, 13u);
    std::vector<int8_t> w = make_w(N, K, 3u);
    std::vector<int8_t> ref = cpu_ref_q(a, w, N, K, D);
    ob_gemm_desc_t desc {};
    desc.m_dim = M; desc.k_dim = K; desc.n_dim = N;
    desc.s_in = 1.0f; desc.s_w = 1.0f; desc.s_out = static_cast<float>(D);
    desc.weights_nk = w.data();
    ob::GemmSynthResult parts{};
    std::string err;
    std::vector<uint8_t> dgc0;
    ob_status_t pst = ob::synthesize_gemm_parts(desc, &parts, &err);
    if (pst == OB_OK) {
      pst = ob::build_dgc0_from_parts(desc, parts,
                                      ctx->builtin_dive_interp.data(),
                                      ctx->builtin_dive_interp.size(),
                                      &dgc0, &err);
    }
    ob_graph_t* g = nullptr;
    if (pst == OB_OK) pst = ob_graph_register_dgc0(ctx, dgc0.data(), dgc0.size(), 0xD128u, &g);
    ob_buffer_t *in_b = nullptr, *out_b = nullptr;
    if (pst == OB_OK) pst = ob_buffer_alloc(ctx, K, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &in_b);
    if (pst == OB_OK) pst = ob_buffer_alloc(ctx, N, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, &out_b);
    if (pst == OB_OK) pst = ob_buffer_write(in_b, a.data(), a.size(), 0u);
    ob_invoke_stats_t is {};
    if (pst == OB_OK) pst = ob_graph_invoke(g, &in_b, 1u, &out_b, 1u, &is);
    std::vector<int8_t> out(N, 0);
    if (pst == OB_OK) ob_buffer_read(out_b, out.data(), N, 0u);
    int e = (pst == OB_OK) ? max_err_i8(out, ref) : -1;
    std::printf("probe_D_std_1x512x128: st=%s err=%d hw_us=%llu\n",
                ob_status_string(pst), e, static_cast<unsigned long long>(is.hw_execution_us));
    ob_buffer_free(out_b);
    ob_buffer_free(in_b);
    ob_graph_release(g);
  }

  ob_context_destroy(ctx);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool mock = false;
  bool emit_gemm = false;
  bool probe_dyn = false;
  uint32_t gemm_m = 0u, gemm_k = 0u, gemm_n = 0u;
  const char* dive_interp_path = nullptr;
  const char* dgc0_path = nullptr;
  const char* in_path = nullptr;
  const char* ref_path = nullptr;
  int iters = 1;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mock") == 0) {
      mock = true;
    } else if (std::strcmp(argv[i], "--probe-dyn-weights") == 0) {
      probe_dyn = true;
    } else if (std::strcmp(argv[i], "--emit-gemm") == 0 && i + 3 < argc) {
      emit_gemm = true;
      gemm_m = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
      gemm_k = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
      gemm_n = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--dive-interp") == 0 && i + 1 < argc) {
      dive_interp_path = argv[++i];
    } else if (std::strcmp(argv[i], "--dgc0") == 0 && i + 1 < argc) {
      dgc0_path = argv[++i];
    } else if (std::strcmp(argv[i], "--in") == 0 && i + 1 < argc) {
      in_path = argv[++i];
    } else if (std::strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
      ref_path = argv[++i];
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
      if (iters < 1) iters = 1;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      print_usage(argv[0]);
      return 2;
    }
  }

  if (probe_dyn) {
    return run_probe_dyn_weights(dive_interp_path, mock);
  }

  if (emit_gemm) {
    return run_emit_gemm(gemm_m, gemm_k, gemm_n, iters, dive_interp_path, mock);
  }

  if (mock || !dgc0_path) {
    return run_mock_smoke();
  }

  ob_config_t cfg = ob_config_default();
  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }

  ob_graph_t* graph = nullptr;
  st = ob_graph_register_dgc0_file(ctx, dgc0_path, 0u, &graph);
  if (st != OB_OK || !graph) {
    std::fprintf(stderr, "ob_graph_register_dgc0_file failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  ob_dgc0_info_t info {};
  ob_graph_get_info(graph, &info);

  std::vector<ob_buffer_t*> in_bufs(info.num_inputs, nullptr);
  std::vector<ob_buffer_t*> out_bufs(info.num_outputs, nullptr);

  std::vector<uint8_t> in_blob;
  if (in_path && !read_file(in_path, &in_blob)) {
    std::fprintf(stderr, "Failed reading --in %s\n", in_path);
    ob_graph_release(graph);
    ob_context_destroy(ctx);
    return 1;
  }

  size_t cursor = 0;
  for (uint32_t i = 0; i < info.num_inputs; ++i) {
    size_t sz = static_cast<size_t>(info.input_sizes[i]);
    st = ob_buffer_alloc(ctx, sz, OB_BUF_TO_DEVICE | OB_BUF_ZERO_INIT, &in_bufs[i]);
    if (st != OB_OK) break;
    if (!in_blob.empty() && cursor + sz <= in_blob.size()) {
      st = ob_buffer_write(in_bufs[i], in_blob.data() + cursor, sz, 0u);
      cursor += sz;
      if (st != OB_OK) break;
    }
  }
  for (uint32_t i = 0; st == OB_OK && i < info.num_outputs; ++i) {
    st = ob_buffer_alloc(ctx, static_cast<size_t>(info.output_sizes[i]),
                         OB_BUF_FROM_DEVICE | OB_BUF_ZERO_INIT, &out_bufs[i]);
  }

  ob_invoke_stats_t last_stats {};
  for (int it = 0; st == OB_OK && it < iters; ++it) {
    st = ob_graph_invoke(graph, in_bufs.data(), info.num_inputs,
                         out_bufs.data(), info.num_outputs, &last_stats);
  }

  int rc = (st == OB_OK) ? 0 : 1;
  if (st != OB_OK) {
    std::fprintf(stderr, "ob_graph_invoke failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
  } else {
    int max_err = -1;
    if (ref_path && info.num_outputs > 0u) {
      std::vector<uint8_t> ref_blob;
      std::vector<uint8_t> out0(static_cast<size_t>(info.output_sizes[0]), 0);
      if (read_file(ref_path, &ref_blob) &&
          ob_buffer_read(out_bufs[0], out0.data(), out0.size(), 0u) == OB_OK &&
          ref_blob.size() == out0.size()) {
        max_err = 0;
        for (size_t i = 0; i < out0.size(); ++i) {
          int a = static_cast<int8_t>(out0[i]);
          int b = static_cast<int8_t>(ref_blob[i]);
          int d = a > b ? (a - b) : (b - a);
          if (d > max_err) max_err = d;
        }
        if (max_err != 0) rc = 1;
      } else {
        rc = 1;
      }
    }
    std::printf("openbuenos_selftest: ok=%d handle=0x%x hw_us=%llu wall_us=%llu max_abs_err=%d\n",
                rc == 0 ? 1 : 0, last_stats.model_handle,
                static_cast<unsigned long long>(last_stats.hw_execution_us),
                static_cast<unsigned long long>(last_stats.wall_time_us),
                max_err);
  }

  for (ob_buffer_t* b : out_bufs) ob_buffer_free(b);
  for (ob_buffer_t* b : in_bufs) ob_buffer_free(b);
  ob_graph_release(graph);
  ob_context_destroy(ctx);
  return rc;
}
