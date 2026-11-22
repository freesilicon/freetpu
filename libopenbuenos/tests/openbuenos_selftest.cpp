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
      "  %s --dgc0 <model.dgc0> [--in <in.bin>] [--ref <ref.bin>] [--iters <N>] [--perf-mode <0..5>] [--profile]\n"
      "  %s --emit-gemm <M> <K> <N> [--dive-interp <path>] [--iters <N>] [--perf-mode <0..5>] [--profile] [--mock]\n"
      "  %s --test-quant-batch [--dive-interp <path>] [--iters <N>] [--perf-mode <0..5>] [--profile] [--mock]\n"
      "  %s --test-swiglu [--dive-interp <path>] [--iters <N>] [--perf-mode <0..5>] [--profile] [--mock]\n"
      "  %s --probe-dyn-weights [--dive-interp <path>] [--mock]\n"
      "  %s --test-fused-act [--dive-interp <path>] [--mock]\n"
      "  %s --test-outlier-spike [--dive-interp <path>] [--mock]\n",
      argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

void print_profile_line(const char* tag, const ob_context_t* ctx) {
  ob_profile_stats_t ps {};
  if (ob_context_get_profile(ctx, &ps) != OB_OK) return;
  std::printf(
      "profile[%s]: calls=%llu tiles=%llu horiz_fusions=%llu horiz_saved=%llu "
      "swiglu_calls=%llu swiglu_chained=%llu swiglu_fb=%llu "
      "hw_us=%llu wall_us=%llu quant_us=%llu hash_us=%llu pack_us=%llu "
      "dma_sync_us=%llu submit_us=%llu wait_us=%llu dequant_us=%llu "
      "sat_retries=%llu spin_hits=%llu poll_waits=%llu pool_hits=%llu pool_misses=%llu "
      "w_hits=%llu w_misses=%llu w_evict=%llu w_bytes=%llu w_peak_bytes=%llu w_budget_bytes=%llu\n",
      tag,
      static_cast<unsigned long long>(ps.calls),
      static_cast<unsigned long long>(ps.tiles_dispatched),
      static_cast<unsigned long long>(ps.horiz_fusions),
      static_cast<unsigned long long>(ps.horiz_jobs_saved),
      static_cast<unsigned long long>(ps.swiglu_calls),
      static_cast<unsigned long long>(ps.swiglu_chunks_chained),
      static_cast<unsigned long long>(ps.swiglu_fallbacks),
      static_cast<unsigned long long>(ps.hw_us),
      static_cast<unsigned long long>(ps.total_wall_us),
      static_cast<unsigned long long>(ps.quant_us),
      static_cast<unsigned long long>(ps.hash_us),
      static_cast<unsigned long long>(ps.weight_pack_us),
      static_cast<unsigned long long>(ps.dma_sync_us),
      static_cast<unsigned long long>(ps.submit_us),
      static_cast<unsigned long long>(ps.wait_us),
      static_cast<unsigned long long>(ps.dequant_us),
      static_cast<unsigned long long>(ps.sat_retries),
      static_cast<unsigned long long>(ps.spin_hits),
      static_cast<unsigned long long>(ps.poll_waits),
      static_cast<unsigned long long>(ps.pool_hits),
      static_cast<unsigned long long>(ps.pool_misses),
      static_cast<unsigned long long>(ps.weight_cache_hits),
      static_cast<unsigned long long>(ps.weight_cache_misses),
      static_cast<unsigned long long>(ps.weight_cache_evictions),
      static_cast<unsigned long long>(ps.weight_cache_bytes),
      static_cast<unsigned long long>(ps.weight_cache_peak_bytes),
      static_cast<unsigned long long>(ps.weight_cache_budget_bytes));
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
                  int iters, int perf_mode, const char* dive_interp_path,
                  bool mock, bool profile) {
  if (M == 0u || K == 0u || N == 0u) {
    std::fprintf(stderr, "--emit-gemm requires M, K, N > 0\n");
    return 2;
  }

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = mock ? 1 : 0;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 512u;
  if (perf_mode >= 0) cfg.perf_mode = static_cast<uint32_t>(perf_mode);

  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }
  if (profile) ob_context_set_profile(ctx, 1);

  st = ob_gemm_register_builtin_emitter(ctx, dive_interp_path);
  if (st != OB_OK) {
    std::fprintf(stderr, "ob_gemm_register_builtin_emitter failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  // Mirror ob_gemm.cpp tile limits (M=1 lt_1x* up to 2048, M>=2 lt_64x up to 1536)
  const char* lt_dis = std::getenv("OPENBUENOS_DISABLE_LARGE_TILES");
  const char* blt_dis = std::getenv("OPENBUENOS_DISABLE_BATCHED_LARGE_TILES");
  const bool lt_enabled = !(lt_dis && lt_dis[0] == '1');
  const bool blt_enabled = lt_enabled && !(blt_dis && blt_dis[0] == '1');
  const bool use_lt1 = lt_enabled && K >= 192u && N >= 192u && (K >= 512u || N >= 512u);
  const bool use_lt64 = blt_enabled && M >= 2u && K >= 256u && N >= 128u;

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
    auto build_k_slices = [&](uint32_t tm_cur) -> std::vector<std::pair<uint32_t, uint32_t>> {
      std::vector<std::pair<uint32_t, uint32_t>> ks;
      if (tm_cur >= 2u && use_lt64) {
        static constexpr uint32_t kSteps[4] = {1536u, 1024u, 576u, 512u};
        uint32_t off = 0u;
        while (off < K) {
          uint32_t rem = K - off;
          if (rem <= 1536u) {
            ks.push_back({off, rem});
            break;
          }
          uint32_t take = 512u;
          for (uint32_t s : kSteps) {
            if (s <= rem) { take = s; break; }
          }
          ks.push_back({off, take});
          off += take;
        }
      } else if (tm_cur == 1u && use_lt1) {
        static constexpr uint32_t kSteps[4] = {2048u, 1536u, 1024u, 512u};
        uint32_t off = 0u;
        while (off < K) {
          uint32_t rem = K - off;
          if (rem <= 2048u) {
            ks.push_back({off, rem});
            break;
          }
          uint32_t take = 512u;
          for (uint32_t s : kSteps) {
            if (s <= rem) { take = s; break; }
          }
          ks.push_back({off, take});
          off += take;
        }
      } else {
        const uint32_t k_step = (tm_cur == 1u) ? 512u : 64u;
        for (uint32_t k0 = 0u; k0 < K; k0 += k_step) {
          ks.push_back({k0, std::min(k_step, K - k0)});
        }
      }
      return ks;
    };

    uint32_t m0 = 0u;
    while (m0 < M) {
      const uint32_t rem_m = M - m0;
      const uint32_t tm = use_lt64
          ? ((rem_m == 1u) ? 1u : std::min<uint32_t>(rem_m, 64u))
          : std::min<uint32_t>(rem_m, 16u);
      auto ks = build_k_slices(tm);
      for (const auto& kv : ks) {
        const uint32_t k0 = kv.first;
        const uint32_t tk = kv.second;
        const uint32_t D = ob::compute_tile_scale_D(b_nk.data() + k0, N, tk, K, 0);
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
      if (f32_tok2_hits < last_stats.tiles_dispatched) {
        std::fprintf(stderr, "Expected token-invariant cache hits >=%u on f32_q8_0, got %u/%u\n",
                     last_stats.tiles_dispatched, f32_tok2_hits, f32_tok2_tiles);
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

  std::printf("openbuenos_selftest: ok=%d emit_gemm=%ux%ux%u iters=%d pm=%u core=0x%02x mem=0x%02x wl=%d "
              "tiles=%u cache_hits=%u models=%zu hw_us=%llu wall_us=%llu max_abs_err=%lld "
              "f32_nmse=%.2e f32_tok2_hits=%u/%u\n",
              ok ? 1 : 0, M, K, N, iters,
              ctx->cfg.perf_mode, ctx->cfg.core_power_state, ctx->cfg.mem_power_state,
              ctx->fw_wakelock_held ? 1 : 0,
              last_stats.tiles_dispatched, last_stats.cache_hits,
              cache_entries,
              static_cast<unsigned long long>(last_stats.hw_execution_us),
              static_cast<unsigned long long>(last_stats.wall_time_us),
              static_cast<long long>(worst_err),
              f32_nmse, f32_tok2_hits, f32_tok2_tiles);

  if (profile) {
    char tag[64];
    std::snprintf(tag, sizeof(tag), "emit_%ux%ux%u", M, K, N);
    print_profile_line(tag, ctx);
  }

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

int run_test_fused_act(const char* dive_interp_path, bool mock) {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = mock ? 1 : 0;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 64u;
  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "fused-act ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }
  st = ob_gemm_register_builtin_emitter(ctx, dive_interp_path);
  if (st != OB_OK) {
    std::fprintf(stderr, "fused-act ob_gemm_register_builtin_emitter failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  struct CaseSpec {
    const char* name;
    uint32_t act;
    uint32_t M, K, N;
    float s_in, s_w, s_out;
    int32_t zp_out;
  };
  const CaseSpec cases[4] = {
      {"silu_1x16x16",    OB_ACT_SILU, 1u,  16u,   16u, 0.0625f, 0.0625f, 0.0625f, 0},
      {"gelu_1x16x16",    OB_ACT_GELU, 1u,  16u,   16u, 0.0625f, 0.0625f, 0.0625f, 0},
      {"silu_1x576x1536", OB_ACT_SILU, 1u, 576u, 1536u, 0.0625f, 0.0625f, 0.0625f, 0},
      {"gelu_1x576x1536", OB_ACT_GELU, 1u, 576u, 1536u, 0.0625f, 0.0625f, 0.0625f, 0},
  };

  bool all_ok = true;
  for (const CaseSpec& cs : cases) {
    // Build sparse deterministic weights so acc spans [-8.0, +7.9] cleanly
    std::vector<int8_t> a(static_cast<size_t>(cs.M) * cs.K, 0);
    std::vector<int8_t> w(static_cast<size_t>(cs.N) * cs.K, 0);
    std::vector<int32_t> bias(cs.N, 0);
    for (uint32_t k = 0; k < cs.K; ++k) {
      a[k] = static_cast<int8_t>(static_cast<int32_t>((k * 17u + 5u) % 33u) - 16);
    }
    for (uint32_t n = 0; n < cs.N; ++n) {
      // 4 active taps per row so acc * real_mul sweeps full [-120..+120] range
      for (uint32_t t = 0; t < 4u; ++t) {
        uint32_t k = (n * 7u + t * (cs.K / 4u)) % cs.K;
        w[static_cast<size_t>(n) * cs.K + k] = static_cast<int8_t>(((n + t) & 1u) ? -12 : 12);
      }
      bias[n] = (static_cast<int32_t>(n % 240u) - 120) * 16;
    }

    ob_gemm_desc_t desc {};
    desc.m_dim = cs.M;
    desc.k_dim = cs.K;
    desc.n_dim = cs.N;
    desc.s_in = cs.s_in;
    desc.zp_in = 0;
    desc.s_w = cs.s_w;
    desc.zp_w = 0;
    desc.s_out = cs.s_out;
    desc.zp_out = cs.zp_out;
    desc.activation = cs.act;
    desc.weights_nk = w.data();
    desc.bias_n = bias.data();

    ob::GemmSynthResult syn{};
    std::string err;
    st = ob::synthesize_gemm_parts(desc, &syn, &err);
    std::vector<uint8_t> dgc0;
    if (st == OB_OK) {
      st = ob::build_dgc0_from_parts(desc, syn, ctx->builtin_dive_interp.data(),
                                     ctx->builtin_dive_interp.size(), &dgc0, &err);
    }
    if (st != OB_OK) {
      std::fprintf(stderr, "fused-act %s synth failed: %s (%s)\n",
                   cs.name, ob_status_string(st), err.c_str());
      all_ok = false;
      continue;
    }

    // Compute simulated expected output from exact u_mid + Mode-A 0x640 spline
    const float rm = (cs.s_in * cs.s_w) / cs.s_out;
    std::vector<uint8_t> u_mid(cs.N, 0u);
    for (uint32_t n = 0; n < cs.N; ++n) {
      int32_t acc = bias[n];
      for (uint32_t k = 0; k < cs.K; ++k) {
        acc += static_cast<int32_t>(a[k]) * static_cast<int32_t>(w[static_cast<size_t>(n) * cs.K + k]);
      }
      int32_t q = static_cast<int32_t>(std::nearbyint(static_cast<float>(acc) * rm + static_cast<float>(cs.zp_out + 128)));
      if (q < 0) q = 0;
      if (q > 255) q = 255;
      u_mid[n] = static_cast<uint8_t>(q);
    }
    std::vector<int8_t> ref_out(cs.N, 0);
    st = ob::simulate_c5_mode_a_fused(syn.chunks[5], u_mid.data(), cs.N, ref_out.data(), &err);
    if (st != OB_OK) {
      std::fprintf(stderr, "fused-act %s sim failed: %s\n", cs.name, err.c_str());
      all_ok = false;
      continue;
    }

    ob_graph_t* g = nullptr;
    ob_buffer_t* in_b = nullptr;
    ob_buffer_t* out_b = nullptr;
    st = ob_graph_register_dgc0(ctx, dgc0.data(), dgc0.size(), 0u, &g);
    if (st == OB_OK) st = ob_buffer_alloc(ctx, std::max<size_t>(a.size(), 4096u), OB_BUF_TO_DEVICE | OB_BUF_ZERO_INIT, &in_b);
    if (st == OB_OK) st = ob_buffer_alloc(ctx, std::max<size_t>(cs.N, 4096u), OB_BUF_FROM_DEVICE | OB_BUF_ZERO_INIT, &out_b);
    if (st == OB_OK) st = ob_buffer_write(in_b, a.data(), a.size(), 0u);
    ob_invoke_stats_t ist {};
    if (st == OB_OK) st = ob_graph_invoke(g, &in_b, 1u, &out_b, 1u, &ist);

    int max_err = 0;
    if (st == OB_OK && !mock) {
      std::vector<int8_t> got(cs.N, 0);
      st = ob_buffer_read(out_b, got.data(), cs.N, 0u);
      if (st == OB_OK) {
        for (uint32_t n = 0; n < cs.N; ++n) {
          int d = std::abs(static_cast<int>(got[n]) - static_cast<int>(ref_out[n]));
          if (d > max_err) max_err = d;
        }
      }
    }
    const bool ok = (st == OB_OK) && (mock || max_err == 0);
    if (!ok) all_ok = false;
    std::printf("fused_act[%s]: ok=%d c5_bytes=%zu hw_us=%llu wall_us=%llu max_abs_err=%d\n",
                cs.name, ok ? 1 : 0, syn.chunks[5].size(),
                static_cast<unsigned long long>(ist.hw_execution_us),
                static_cast<unsigned long long>(ist.wall_time_us),
                mock ? -1 : max_err);

    if (out_b) ob_buffer_free(out_b);
    if (in_b) ob_buffer_free(in_b);
    if (g) ob_graph_release(g);
  }

  ob_context_destroy(ctx);
  return all_ok ? 0 : 1;
}

int run_test_outlier_spike(const char* dive_interp_path, bool mock) {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = mock ? 1 : 0;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 256u;
  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) return 1;
  st = ob_gemm_register_builtin_emitter(ctx, dive_interp_path);
  if (st != OB_OK) {
    std::fprintf(stderr, "outlier-spike emitter failed: %s\n", ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  constexpr uint32_t M = 1u, K = 1536u, N = 256u, nb_k = K / OB_QK8_0;
  uint32_t prng = 0x51A7BEEFu;
  auto next_g = [&]() -> float {
    float s = 0.0f;
    for (int i = 0; i < 6; ++i) {
      prng = prng * 1664525u + 1013904223u;
      s += (static_cast<float>(prng >> 8) / static_cast<float>(1u << 23)) - 1.0f;
    }
    return s * 0.7071f;
  };

  auto run_case = [&](const char* label, bool force_retry) -> bool {
    std::vector<float> a(K), b_f(static_cast<size_t>(N) * K);
    for (float& v : a) v = force_retry ? (next_g() * 0.1f) : next_g();
    for (float& v : b_f) v = force_retry ? (next_g() * 0.1f) : next_g();
    if (!force_retry) {
      // 40-sigma Qwen2.5 attention-sink spike with tiny weight columns (bulk dominates!)
      a[17] = 50.0f;
      a[513] = -42.0f;
      for (uint32_t n = 0; n < N; ++n) {
        b_f[static_cast<size_t>(n) * K + 17] *= 0.02f;
        b_f[static_cast<size_t>(n) * K + 513] *= 0.02f;
      }
    } else {
      // 20 aligned spikes (>8 peel cap) + aligned super-weights -> triggers live TPU retry!
      for (uint32_t s = 0; s < 20u; ++s) {
        a[s * 32u] = 40.0f;
        for (uint32_t n = 0; n < 4u; ++n) b_f[static_cast<size_t>(n) * K + s * 32u] = 2.0f;
      }
    }

    std::vector<ob_block_q8_0_t> bq8(static_cast<size_t>(N) * nb_k);
    std::vector<float> b_deq(static_cast<size_t>(N) * K);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        float am = 0.0f;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          float av = std::fabs(b_f[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j]);
          if (av > am) am = av;
        }
        uint16_t dh = fp32_to_fp16(am > 0.0f ? (am / 127.0f) : 1.0f);
        float d = fp16_to_fp32(dh);
        ob_block_q8_0_t& blk = bq8[static_cast<size_t>(n) * nb_k + bi];
        blk.d = dh;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          long q = (d > 0.0f) ? std::lround(b_f[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j] / d) : 0L;
          if (q < -127) q = -127;
          if (q > 127) q = 127;
          blk.qs[j] = static_cast<int8_t>(q);
          b_deq[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j] = d * static_cast<float>(blk.qs[j]);
        }
      }
    }

    std::vector<float> c(N, 0.0f);
    ob_invoke_stats_t stats {};
    st = ob_gemm_matmul_f32_q8_0(ctx, a.data(), bq8.data(), c.data(), M, K, N, &stats);
    double mse = 0.0, ref_e = 0.0;
    for (uint32_t n = 0; n < N; ++n) {
      double ref = 0.0;
      for (uint32_t k = 0; k < K; ++k) ref += static_cast<double>(a[k]) * static_cast<double>(b_deq[static_cast<size_t>(n) * K + k]);
      double diff = static_cast<double>(c[n]) - ref;
      mse += diff * diff;
      ref_e += ref * ref;
    }
    double nmse = mse / std::max(ref_e, 1e-30);
    bool ok = (st == OB_OK) &&
              ((stats.flags & OB_STAT_FLAG_EMITTER_USED) != 0u) &&
              (mock || (stats.flags & OB_STAT_FLAG_CPU_FALLBACK) == 0u) &&
              (nmse < (force_retry ? 2.0e-3 : 1.5e-3)) &&
              (mock || !force_retry || stats.tiles_dispatched >= 2u);
    std::printf("outlier_spike[%s]: ok=%d tiles=%u hw_us=%llu wall_us=%llu nmse=%.2e cpu_fb=%d\n",
                label, ok ? 1 : 0, stats.tiles_dispatched,
                static_cast<unsigned long long>(stats.hw_execution_us),
                static_cast<unsigned long long>(stats.wall_time_us),
                nmse, (stats.flags & OB_STAT_FLAG_CPU_FALLBACK) ? 1 : 0);
    return ok;
  };

  bool ok1 = run_case("40sigma_peel_1x1536x256", false);
  bool ok2 = run_case("12spike_tpu_retry_1x1536x256", true);
  ob_context_destroy(ctx);
  return (ok1 && ok2) ? 0 : 1;
}

int run_test_quant_batch(int iters, int perf_mode, const char* dive_interp_path,
                         bool mock, bool profile) {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = mock ? 1 : 0;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 256u;
  if (perf_mode >= 0) cfg.perf_mode = static_cast<uint32_t>(perf_mode);

  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "quant-batch ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }
  if (profile) ob_context_set_profile(ctx, 1);
  st = ob_gemm_register_builtin_emitter(ctx, dive_interp_path);
  if (st != OB_OK) {
    std::fprintf(stderr, "quant-batch emitter failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  uint32_t prng = 0x91B2C3D4u;
  auto next_g = [&]() -> float {
    float s = 0.0f;
    for (int i = 0; i < 6; ++i) {
      prng = prng * 1664525u + 1013904223u;
      s += (static_cast<float>(prng >> 8) / static_cast<float>(1u << 23)) - 1.0f;
    }
    return s * 0.7071f;
  };

  auto make_q4 = [&](uint32_t K, uint32_t N,
                     std::vector<ob_block_q4_0_t>* q4,
                     std::vector<float>* deq) {
    const uint32_t nb_k = K / OB_QK8_0;
    q4->resize(static_cast<size_t>(N) * nb_k);
    deq->resize(static_cast<size_t>(N) * K);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        float tmp[OB_QK8_0];
        float am = 0.0f;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          tmp[j] = next_g();
          float av = std::fabs(tmp[j]);
          if (av > am) am = av;
        }
        uint16_t dh = fp32_to_fp16(am > 0.0f ? (am / 7.0f) : 1.0f);
        float d = fp16_to_fp32(dh);
        ob_block_q4_0_t& blk = (*q4)[static_cast<size_t>(n) * nb_k + bi];
        blk.d = dh;
        for (uint32_t j = 0; j < 16u; ++j) {
          long q0 = (d > 0.0f) ? std::lround(tmp[j] / d) : 0L;
          long q1 = (d > 0.0f) ? std::lround(tmp[j + 16u] / d) : 0L;
          if (q0 < -8) q0 = -8;
          if (q0 > 7) q0 = 7;
          if (q1 < -8) q1 = -8;
          if (q1 > 7) q1 = 7;
          blk.qs[j] = static_cast<uint8_t>((q0 + 8) | ((q1 + 8) << 4));
          (*deq)[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j] = d * static_cast<float>(q0);
          (*deq)[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j + 16u] = d * static_cast<float>(q1);
        }
      }
    }
  };

  auto make_q8 = [&](uint32_t K, uint32_t N,
                     std::vector<ob_block_q8_0_t>* q8,
                     std::vector<float>* deq) {
    const uint32_t nb_k = K / OB_QK8_0;
    q8->resize(static_cast<size_t>(N) * nb_k);
    deq->resize(static_cast<size_t>(N) * K);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t bi = 0; bi < nb_k; ++bi) {
        float tmp[OB_QK8_0];
        float am = 0.0f;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          tmp[j] = next_g();
          float av = std::fabs(tmp[j]);
          if (av > am) am = av;
        }
        uint16_t dh = fp32_to_fp16(am > 0.0f ? (am / 127.0f) : 1.0f);
        float d = fp16_to_fp32(dh);
        ob_block_q8_0_t& blk = (*q8)[static_cast<size_t>(n) * nb_k + bi];
        blk.d = dh;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          long q = (d > 0.0f) ? std::lround(tmp[j] / d) : 0L;
          if (q < -127) q = -127;
          if (q > 127) q = 127;
          blk.qs[j] = static_cast<int8_t>(q);
          (*deq)[static_cast<size_t>(n) * K + bi * OB_QK8_0 + j] = d * static_cast<float>(q);
        }
      }
    }
  };

  auto nmse_vs_ref = [&](const std::vector<float>& a, const std::vector<float>& b_deq,
                         const std::vector<float>& got,
                         uint32_t M, uint32_t K, uint32_t N, uint32_t act) -> double {
    double mse = 0.0, re = 0.0;
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        double acc = 0.0;
        const float* arow = a.data() + static_cast<size_t>(m) * K;
        const float* brow = b_deq.data() + static_cast<size_t>(n) * K;
        for (uint32_t k = 0; k < K; ++k) acc += static_cast<double>(arow[k]) * static_cast<double>(brow[k]);
        if (act == OB_ACT_SILU) {
          acc = acc / (1.0 + std::exp(-acc));
        } else if (act == OB_ACT_GELU) {
          acc = 0.5 * acc * (1.0 + std::tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)));
        }
        double diff = static_cast<double>(got[static_cast<size_t>(m) * N + n]) - acc;
        mse += diff * diff;
        re += acc * acc;
      }
    }
    return mse / std::max(re, 1e-30);
  };

  bool all_ok = true;
  const uint32_t exp_qkv_tiles = ctx->horiz_fusion_enabled ? 1u : 3u;

  // Case 1: SmolLM2-135M M=1 Q4_0 shared-A Q+K+V (576 -> 576+192+192 = 960 -> 1x576x1024 fused!)
  {
    constexpr uint32_t M = 1u, K = 576u, Nq = 576u, Nkv = 192u;
    std::vector<float> a(K);
    for (float& v : a) v = next_g();
    std::vector<ob_block_q4_0_t> bq, bk, bv;
    std::vector<float> dq, dk, dv;
    make_q4(K, Nq, &bq, &dq);
    make_q4(K, Nkv, &bk, &dk);
    make_q4(K, Nkv, &bv, &dv);
    std::vector<float> cq(Nq, 0.0f), ck(Nkv, 0.0f), cv(Nkv, 0.0f);
    ob_gemm_f32_quant_job_t jobs[3] {};
    jobs[0] = {bq.data(), cq.data(), Nq,  OB_ACT_NONE, 1u, {0, 0, 0}};
    jobs[1] = {bk.data(), ck.data(), Nkv, OB_ACT_NONE, 1u, {0, 0, 0}};
    jobs[2] = {bv.data(), cv.data(), Nkv, OB_ACT_NONE, 1u, {0, 0, 0}};

    ob_invoke_stats_t s_cold {}, s_warm {};
    st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 3u, &s_cold);
    if (profile) ob_context_reset_profile(ctx);
    for (int it = 0; st == OB_OK && it < std::max(1, iters); ++it) {
      st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 3u, &s_warm);
    }
    double e_q = nmse_vs_ref(a, dq, cq, M, K, Nq, OB_ACT_NONE);
    double e_k = nmse_vs_ref(a, dk, ck, M, K, Nkv, OB_ACT_NONE);
    double e_v = nmse_vs_ref(a, dv, cv, M, K, Nkv, OB_ACT_NONE);
    double max_e = std::max(e_q, std::max(e_k, e_v));
    bool ok = (st == OB_OK) && (s_warm.tiles_dispatched == exp_qkv_tiles) &&
              (s_warm.cache_hits == exp_qkv_tiles) &&
              std::isfinite(e_q) && (e_q < 1.5e-3) &&
              std::isfinite(e_k) && (e_k < 1.5e-3) &&
              std::isfinite(e_v) && (e_v < 1.5e-3);
    if (!ok) all_ok = false;
    std::printf("quant_batch[q4_qkv_1x576]: ok=%d pm=%u wl=%d tiles=%u hits=%u cold_us=%llu warm_hw_us=%llu warm_wall_us=%llu max_nmse=%.2e\n",
                ok ? 1 : 0, ctx->cfg.perf_mode, ctx->fw_wakelock_held ? 1 : 0,
                s_warm.tiles_dispatched, s_warm.cache_hits,
                static_cast<unsigned long long>(s_cold.wall_time_us),
                static_cast<unsigned long long>(s_warm.hw_execution_us),
                static_cast<unsigned long long>(s_warm.wall_time_us),
                max_e);
    if (profile) print_profile_line("q4_qkv_1x576", ctx);
  }

  // Case 2: Qwen2.5-1.5B M=1 Q8_0 shared-A Q+K+V (1536 -> 1536+256+256 = 2048 -> 1x1536x2048 fused!)
  {
    constexpr uint32_t M = 1u, K = 1536u, Nq = 1536u, Nkv = 256u;
    std::vector<float> a(K);
    for (float& v : a) v = next_g();
    std::vector<ob_block_q8_0_t> bq, bk, bv;
    std::vector<float> dq, dk, dv;
    make_q8(K, Nq, &bq, &dq);
    make_q8(K, Nkv, &bk, &dk);
    make_q8(K, Nkv, &bv, &dv);
    std::vector<float> cq(Nq, 0.0f), ck(Nkv, 0.0f), cv(Nkv, 0.0f);
    ob_gemm_f32_quant_job_t jobs[3] {};
    jobs[0] = {bq.data(), cq.data(), Nq,  OB_ACT_NONE, 0u, {0, 0, 0}};
    jobs[1] = {bk.data(), ck.data(), Nkv, OB_ACT_NONE, 0u, {0, 0, 0}};
    jobs[2] = {bv.data(), cv.data(), Nkv, OB_ACT_NONE, 0u, {0, 0, 0}};

    ob_invoke_stats_t s_cold {}, s_warm {};
    st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 3u, &s_cold);
    if (profile) ob_context_reset_profile(ctx);
    for (int it = 0; st == OB_OK && it < std::max(1, iters); ++it) {
      st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 3u, &s_warm);
    }
    double e_q = nmse_vs_ref(a, dq, cq, M, K, Nq, OB_ACT_NONE);
    double e_k = nmse_vs_ref(a, dk, ck, M, K, Nkv, OB_ACT_NONE);
    double e_v = nmse_vs_ref(a, dv, cv, M, K, Nkv, OB_ACT_NONE);
    double max_e = std::max(e_q, std::max(e_k, e_v));
    bool ok = (st == OB_OK) && (s_warm.tiles_dispatched == exp_qkv_tiles) &&
              (s_warm.cache_hits == exp_qkv_tiles) &&
              std::isfinite(e_q) && (e_q < 1.5e-3) &&
              std::isfinite(e_k) && (e_k < 1.5e-3) &&
              std::isfinite(e_v) && (e_v < 1.5e-3);
    if (!ok) all_ok = false;
    std::printf("quant_batch[q8_qkv_1x1536]: ok=%d pm=%u wl=%d tiles=%u hits=%u cold_us=%llu warm_hw_us=%llu warm_wall_us=%llu max_nmse=%.2e\n",
                ok ? 1 : 0, ctx->cfg.perf_mode, ctx->fw_wakelock_held ? 1 : 0,
                s_warm.tiles_dispatched, s_warm.cache_hits,
                static_cast<unsigned long long>(s_cold.wall_time_us),
                static_cast<unsigned long long>(s_warm.hw_execution_us),
                static_cast<unsigned long long>(s_warm.wall_time_us),
                max_e);
    if (profile) print_profile_line("q8_qkv_1x1536", ctx);
  }

  // Case 3: SmolLM2-135M M=64 batched prefill Q4_0 shared-A Q+K+V (576 -> 960 -> 64x576x960 fused!)
  {
    constexpr uint32_t M = 64u, K = 576u, Nq = 576u, Nkv = 192u;
    std::vector<float> a(static_cast<size_t>(M) * K);
    for (float& v : a) v = next_g() * 0.25f;
    std::vector<ob_block_q4_0_t> bq, bk, bv;
    std::vector<float> dq, dk, dv;
    make_q4(K, Nq, &bq, &dq);
    make_q4(K, Nkv, &bk, &dk);
    make_q4(K, Nkv, &bv, &dv);
    std::vector<float> cq(static_cast<size_t>(M) * Nq, 0.0f);
    std::vector<float> ck(static_cast<size_t>(M) * Nkv, 0.0f);
    std::vector<float> cv(static_cast<size_t>(M) * Nkv, 0.0f);
    ob_gemm_f32_quant_job_t jobs[3] {};
    jobs[0] = {bq.data(), cq.data(), Nq,  OB_ACT_NONE, 1u, {0, 0, 0}};
    jobs[1] = {bk.data(), ck.data(), Nkv, OB_ACT_NONE, 1u, {0, 0, 0}};
    jobs[2] = {bv.data(), cv.data(), Nkv, OB_ACT_NONE, 1u, {0, 0, 0}};

    ob_invoke_stats_t s_cold {}, s_warm {};
    st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 3u, &s_cold);
    if (profile) ob_context_reset_profile(ctx);
    for (int it = 0; st == OB_OK && it < std::max(1, iters); ++it) {
      st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 3u, &s_warm);
    }
    double e_q = nmse_vs_ref(a, dq, cq, M, K, Nq, OB_ACT_NONE);
    double e_k = nmse_vs_ref(a, dk, ck, M, K, Nkv, OB_ACT_NONE);
    double e_v = nmse_vs_ref(a, dv, cv, M, K, Nkv, OB_ACT_NONE);
    double max_e = std::max(e_q, std::max(e_k, e_v));
    bool ok = (st == OB_OK) && (s_warm.tiles_dispatched == exp_qkv_tiles) &&
              (s_warm.cache_hits == exp_qkv_tiles) &&
              std::isfinite(e_q) && (e_q < 2.0e-3) &&
              std::isfinite(e_k) && (e_k < 2.0e-3) &&
              std::isfinite(e_v) && (e_v < 2.0e-3);
    if (!ok) all_ok = false;
    std::printf("quant_batch[q4_qkv_64x576]: ok=%d pm=%u wl=%d tiles=%u hits=%u cold_us=%llu warm_hw_us=%llu warm_wall_us=%llu max_nmse=%.2e\n",
                ok ? 1 : 0, ctx->cfg.perf_mode, ctx->fw_wakelock_held ? 1 : 0,
                s_warm.tiles_dispatched, s_warm.cache_hits,
                static_cast<unsigned long long>(s_cold.wall_time_us),
                static_cast<unsigned long long>(s_warm.hw_execution_us),
                static_cast<unsigned long long>(s_warm.wall_time_us),
                max_e);
    if (profile) print_profile_line("q4_qkv_64x576", ctx);
  }

  // Case 4: SmolLM2-135M M=64 batched prefill Q4_0 shared-A Gate(SiLU) + Up (64x576x1536)
  {
    constexpr uint32_t M = 64u, K = 576u, N = 1536u;
    std::vector<float> a(static_cast<size_t>(M) * K);
    for (float& v : a) v = next_g() * 0.25f;
    std::vector<ob_block_q4_0_t> bg, bu;
    std::vector<float> dg, du;
    make_q4(K, N, &bg, &dg);
    make_q4(K, N, &bu, &du);
    std::vector<float> cg(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> cu(static_cast<size_t>(M) * N, 0.0f);
    ob_gemm_f32_quant_job_t jobs[2] {};
    jobs[0] = {bg.data(), cg.data(), N, OB_ACT_SILU, 1u, {0, 0, 0}};
    jobs[1] = {bu.data(), cu.data(), N, OB_ACT_NONE, 1u, {0, 0, 0}};

    ob_invoke_stats_t s_cold {}, s_warm {};
    st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 2u, &s_cold);
    if (profile) ob_context_reset_profile(ctx);
    for (int it = 0; st == OB_OK && it < std::max(1, iters); ++it) {
      st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, jobs, 2u, &s_warm);
    }
    double e_g = nmse_vs_ref(a, dg, cg, M, K, N, OB_ACT_SILU);
    double e_u = nmse_vs_ref(a, du, cu, M, K, N, OB_ACT_NONE);
    double max_e = std::max(e_g, e_u);
    bool ok = (st == OB_OK) && (s_warm.tiles_dispatched == 2u) &&
              (s_warm.cache_hits == 2u) &&
              std::isfinite(e_g) && (e_g < 2.0e-3) &&
              std::isfinite(e_u) && (e_u < 2.0e-3);
    if (!ok) all_ok = false;
    std::printf("quant_batch[q4_gate_up_64x576x1536]: ok=%d pm=%u wl=%d tiles=%u hits=%u cold_us=%llu warm_hw_us=%llu warm_wall_us=%llu max_nmse=%.2e\n",
                ok ? 1 : 0, ctx->cfg.perf_mode, ctx->fw_wakelock_held ? 1 : 0,
                s_warm.tiles_dispatched, s_warm.cache_hits,
                static_cast<unsigned long long>(s_cold.wall_time_us),
                static_cast<unsigned long long>(s_warm.hw_execution_us),
                static_cast<unsigned long long>(s_warm.wall_time_us),
                max_e);
    if (profile) print_profile_line("q4_gate_up_64x576x1536", ctx);
  }

  // Case 5: Qwen2.5-1.5B M=64 batched prefill Q4_0 64x1536x1536 (1 lt_64x tile!)
  {
    constexpr uint32_t M = 64u, K = 1536u, N = 1536u;
    std::vector<float> a(static_cast<size_t>(M) * K);
    for (float& v : a) v = next_g() * 0.25f;
    std::vector<ob_block_q4_0_t> bw;
    std::vector<float> dw;
    make_q4(K, N, &bw, &dw);
    std::vector<float> cw(static_cast<size_t>(M) * N, 0.0f);
    ob_gemm_f32_quant_job_t job {bw.data(), cw.data(), N, OB_ACT_NONE, 1u, {0, 0, 0}};

    ob_invoke_stats_t s_cold {}, s_warm {};
    st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, &job, 1u, &s_cold);
    if (profile) ob_context_reset_profile(ctx);
    for (int it = 0; st == OB_OK && it < std::max(1, iters); ++it) {
      st = ob_gemm_matmul_f32_quant_batch(ctx, a.data(), M, K, &job, 1u, &s_warm);
    }
    double e_w = nmse_vs_ref(a, dw, cw, M, K, N, OB_ACT_NONE);
    bool ok = (st == OB_OK) && (s_warm.tiles_dispatched == 1u) &&
              (s_warm.cache_hits == 1u) && (e_w < 2.0e-3);
    if (!ok) all_ok = false;
    std::printf("quant_batch[q4_prefill_64x1536x1536]: ok=%d pm=%u wl=%d tiles=%u hits=%u cold_us=%llu warm_hw_us=%llu warm_wall_us=%llu nmse=%.2e\n",
                ok ? 1 : 0, ctx->cfg.perf_mode, ctx->fw_wakelock_held ? 1 : 0,
                s_warm.tiles_dispatched, s_warm.cache_hits,
                static_cast<unsigned long long>(s_cold.wall_time_us),
                static_cast<unsigned long long>(s_warm.hw_execution_us),
                static_cast<unsigned long long>(s_warm.wall_time_us),
                e_w);
    if (profile) print_profile_line("q4_prefill_64x1536x1536", ctx);
  }

  ob_context_destroy(ctx);
  return all_ok ? 0 : 1;
}

int run_test_swiglu(int iters, int perf_mode, const char* dive_interp_path,
                    bool mock, bool profile) {
  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = mock ? 1 : 0;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 256u;
  if (perf_mode >= 0) cfg.perf_mode = static_cast<uint32_t>(perf_mode);

  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "swiglu ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }
  if (profile) ob_context_set_profile(ctx, 1);
  // Enable opt-in fast SwiGLU chaining + M=64 prefill inside --test-swiglu so M=1 and M=64 shapes are verified
  ctx->swiglu_fast_chain_enabled = true;
  ctx->swiglu_prefill_enabled = true;
  st = ob_gemm_register_builtin_emitter(ctx, dive_interp_path);
  if (st != OB_OK) {
    std::fprintf(stderr, "swiglu emitter failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  uint32_t prng = 0x7B516101u;
  auto next_i8 = [&]() -> int8_t {
    prng = prng * 1664525u + 1013904223u;
    int32_t v = static_cast<int32_t>((prng >> 16) % 33u) - 16;
    return static_cast<int8_t>(v);
  };
  auto next_g = [&]() -> float {
    float s = 0.0f;
    for (int i = 0; i < 6; ++i) {
      prng = prng * 1664525u + 1013904223u;
      s += (static_cast<float>(prng >> 8) / static_cast<float>(1u << 23)) - 1.0f;
    }
    return s * 0.7071f;
  };

  bool all_ok = true;

  // Part A: Exact INT8 single-VII parity (max_abs_err == 0) across all 4 core LLM shapes
  struct Int8Case {
    const char* label;
    uint32_t M, K, H, N;
    float s_in, s_wg, s_gate, s_wu, s_up, s_silu, s_prod, s_wd, s_out;
  };
  const Int8Case i8_cases[4] = {
      {"i8_1x576x1536x576",  1u,  576u, 1536u,  576u, 0.025f, 0.017f, 0.11f,  0.013f, 0.086f, 0.094f, 0.356f, 0.019f, 0.69f},
      {"i8_1x1536x1024x1536", 1u, 1536u, 1024u, 1536u, 0.025f, 0.017f, 0.165f, 0.013f, 0.129f, 0.141f, 0.712f, 0.019f, 0.69f},
      {"i8_64x576x512x576", 64u,  576u,  512u,  576u, 0.025f, 0.017f, 0.11f,  0.013f, 0.086f, 0.094f, 0.356f, 0.019f, 0.46f},
      {"i8_64x1536x512x1536", 64u, 1536u, 512u, 1536u, 0.025f, 0.017f, 0.165f, 0.013f, 0.129f, 0.141f, 0.712f, 0.019f, 0.46f},
  };
  for (const Int8Case& c : i8_cases) {
    std::vector<int8_t> x(static_cast<size_t>(c.M) * c.K);
    std::vector<int8_t> wg(static_cast<size_t>(c.H) * c.K);
    std::vector<int8_t> wu(static_cast<size_t>(c.H) * c.K);
    std::vector<int8_t> wd(static_cast<size_t>(c.N) * c.H);
    for (int8_t& v : x) v = next_i8();
    for (int8_t& v : wg) v = next_i8();
    for (int8_t& v : wu) v = next_i8();
    for (int8_t& v : wd) v = next_i8();
    std::vector<int8_t> y(static_cast<size_t>(c.M) * c.N, 0);
    std::vector<int8_t> y_ref(static_cast<size_t>(c.M) * c.N, 0);

    ob_swiglu_int8_desc_t d{};
    d.m_dim = c.M; d.k_dim = c.K; d.h_dim = c.H; d.n_dim = c.N;
    d.s_in = c.s_in; d.s_wg = c.s_wg; d.s_gate = c.s_gate;
    d.s_wu = c.s_wu; d.s_up = c.s_up; d.s_silu = c.s_silu;
    d.s_prod = c.s_prod; d.s_wd = c.s_wd; d.s_out = c.s_out;
    d.wg_hk = wg.data(); d.wu_hk = wu.data(); d.wd_nh = wd.data();

    ob_invoke_stats_t sc{}, sw{};
    st = ob_gemm_swiglu_int8(ctx, x.data(), &d, y.data(), &sc);
    if (profile) ob_context_reset_profile(ctx);
    for (int it = 0; st == OB_OK && it < std::max(1, iters); ++it) {
      st = ob_gemm_swiglu_int8(ctx, x.data(), &d, y.data(), &sw);
    }
    ob::simulate_swiglu_hw_exact(x.data(), d, y_ref.data());
    int max_err = 0;
    for (size_t i = 0u; i < y.size(); ++i) {
      int e = std::abs(static_cast<int>(y[i]) - static_cast<int>(y_ref[i]));
      if (e > max_err) max_err = e;
    }
    bool ok = (st == OB_OK) && (max_err == 0);
    if (!ok) all_ok = false;
    std::printf("swiglu[%s]: ok=%d pm=%u wl=%d max_abs_err=%d cold_us=%llu warm_hw_us=%llu warm_wall_us=%llu\n",
                c.label, ok ? 1 : 0, ctx->cfg.perf_mode, ctx->fw_wakelock_held ? 1 : 0,
                max_err,
                static_cast<unsigned long long>(sc.wall_time_us),
                static_cast<unsigned long long>(sw.hw_execution_us),
                static_cast<unsigned long long>(sw.wall_time_us));
    if (profile) print_profile_line(c.label, ctx);
  }

  // Part B: End-to-end FP32 Q8_0 / Q4_0 ob_gemm_swiglu_f32_quant live NMSE & latency
  auto make_q8 = [&](uint32_t rows, uint32_t cols, float scale,
                     std::vector<ob_block_q8_0_t>* q8, std::vector<float>* deq) {
    const uint32_t nb = cols / OB_QK8_0;
    q8->resize(static_cast<size_t>(rows) * nb);
    deq->resize(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0u; r < rows; ++r) {
      for (uint32_t bi = 0u; bi < nb; ++bi) {
        float tmp[OB_QK8_0];
        float am = 0.0f;
        for (uint32_t j = 0u; j < OB_QK8_0; ++j) {
          tmp[j] = next_g() * scale;
          float av = std::fabs(tmp[j]);
          if (av > am) am = av;
        }
        uint16_t dh = fp32_to_fp16(am > 0.0f ? (am / 127.0f) : 1.0f);
        float d = fp16_to_fp32(dh);
        ob_block_q8_0_t& blk = (*q8)[static_cast<size_t>(r) * nb + bi];
        blk.d = dh;
        for (uint32_t j = 0u; j < OB_QK8_0; ++j) {
          long q = (d > 0.0f) ? std::lround(tmp[j] / d) : 0L;
          if (q < -127) q = -127;
          if (q > 127) q = 127;
          blk.qs[j] = static_cast<int8_t>(q);
          (*deq)[static_cast<size_t>(r) * cols + bi * OB_QK8_0 + j] = d * static_cast<float>(q);
        }
      }
    }
  };
  auto make_q4 = [&](uint32_t rows, uint32_t cols, float scale,
                     std::vector<ob_block_q4_0_t>* q4, std::vector<float>* deq) {
    const uint32_t nb = cols / OB_QK8_0;
    q4->resize(static_cast<size_t>(rows) * nb);
    deq->resize(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0u; r < rows; ++r) {
      for (uint32_t bi = 0u; bi < nb; ++bi) {
        float tmp[OB_QK8_0];
        float am = 0.0f;
        for (uint32_t j = 0u; j < OB_QK8_0; ++j) {
          tmp[j] = next_g() * scale;
          float av = std::fabs(tmp[j]);
          if (av > am) am = av;
        }
        uint16_t dh = fp32_to_fp16(am > 0.0f ? (am / 7.0f) : 1.0f);
        float d = fp16_to_fp32(dh);
        ob_block_q4_0_t& blk = (*q4)[static_cast<size_t>(r) * nb + bi];
        blk.d = dh;
        for (uint32_t j = 0u; j < 16u; ++j) {
          long q0 = (d > 0.0f) ? std::lround(tmp[j] / d) : 0L;
          long q1 = (d > 0.0f) ? std::lround(tmp[j + 16u] / d) : 0L;
          if (q0 < -8) q0 = -8;
          if (q0 > 7) q0 = 7;
          if (q1 < -8) q1 = -8;
          if (q1 > 7) q1 = 7;
          blk.qs[j] = static_cast<uint8_t>((q0 + 8) | ((q1 + 8) << 4));
          (*deq)[static_cast<size_t>(r) * cols + bi * OB_QK8_0 + j] = d * static_cast<float>(q0);
          (*deq)[static_cast<size_t>(r) * cols + bi * OB_QK8_0 + j + 16u] = d * static_cast<float>(q1);
        }
      }
    }
  };
  auto swiglu_ref_nmse = [&](const std::vector<float>& a,
                             const std::vector<float>& wg_d,
                             const std::vector<float>& wu_d,
                             const std::vector<float>& wd_d,
                             const std::vector<float>& got,
                             uint32_t M, uint32_t K, uint32_t H, uint32_t N) -> double {
    std::vector<double> mid(H, 0.0);
    double mse = 0.0, re = 0.0;
    for (uint32_t m = 0u; m < M; ++m) {
      const float* arow = a.data() + static_cast<size_t>(m) * K;
      for (uint32_t h = 0u; h < H; ++h) {
        double g = 0.0, u = 0.0;
        const float* gr = wg_d.data() + static_cast<size_t>(h) * K;
        const float* ur = wu_d.data() + static_cast<size_t>(h) * K;
        for (uint32_t k = 0u; k < K; ++k) {
          double av = static_cast<double>(arow[k]);
          g += av * static_cast<double>(gr[k]);
          u += av * static_cast<double>(ur[k]);
        }
        mid[h] = (g / (1.0 + std::exp(-g))) * u;
      }
      for (uint32_t n = 0u; n < N; ++n) {
        double acc = 0.0;
        const float* dr = wd_d.data() + static_cast<size_t>(n) * H;
        for (uint32_t h = 0u; h < H; ++h) acc += mid[h] * static_cast<double>(dr[h]);
        double diff = static_cast<double>(got[static_cast<size_t>(m) * N + n]) - acc;
        mse += diff * diff;
        re += acc * acc;
      }
    }
    return mse / std::max(re, 1e-30);
  };

  struct F32Case {
    const char* label;
    uint32_t M, K, H, N;
    uint8_t is_q4;
    uint32_t exp_chunks;
  };
  const F32Case f32_cases[4] = {
      {"smollm2_q8_1x576x1536",   1u,  576u, 1536u,  576u, 0u,  1u},
      {"smollm2_q8_64x576x1536", 64u,  576u, 1536u,  576u, 0u,  3u},
      {"qwen25_q4_1x1536x8960",   1u, 1536u, 8960u, 1536u, 1u,  9u},
      {"qwen25_q4_64x1536x8960", 64u, 1536u, 8960u, 1536u, 1u, 18u},
  };
  for (const F32Case& fc : f32_cases) {
    std::vector<float> a(static_cast<size_t>(fc.M) * fc.K);
    for (float& v : a) v = next_g() * 0.8f;
    std::vector<ob_block_q8_0_t> g8, u8, d8;
    std::vector<ob_block_q4_0_t> g4, u4, d4;
    std::vector<float> gd, ud, dd;
    if (fc.is_q4) {
      make_q4(fc.H, fc.K, 0.04f, &g4, &gd);
      make_q4(fc.H, fc.K, 0.04f, &u4, &ud);
      make_q4(fc.N, fc.H, 0.03f, &d4, &dd);
    } else {
      make_q8(fc.H, fc.K, 0.06f, &g8, &gd);
      make_q8(fc.H, fc.K, 0.06f, &u8, &ud);
      make_q8(fc.N, fc.H, 0.05f, &d8, &dd);
    }
    std::vector<float> c_out(static_cast<size_t>(fc.M) * fc.N, 0.0f);
    ob_gemm_swiglu_quant_desc_t sd{};
    sd.w_gate_hk = fc.is_q4 ? static_cast<const void*>(g4.data()) : static_cast<const void*>(g8.data());
    sd.w_up_hk   = fc.is_q4 ? static_cast<const void*>(u4.data()) : static_cast<const void*>(u8.data());
    sd.w_down_nh = fc.is_q4 ? static_cast<const void*>(d4.data()) : static_cast<const void*>(d8.data());
    sd.m_dim = fc.M; sd.k_dim = fc.K; sd.h_dim = fc.H; sd.n_dim = fc.N;
    sd.gate_is_q4_0 = fc.is_q4; sd.up_is_q4_0 = fc.is_q4; sd.down_is_q4_0 = fc.is_q4;

    ob_invoke_stats_t sc{}, sw{};
    st = ob_gemm_swiglu_f32_quant(ctx, a.data(), &sd, c_out.data(), &sc);
    if (profile) ob_context_reset_profile(ctx);
    ob_profile_stats_t p0{}, p1{};
    (void)ob_context_get_profile(ctx, &p0);
    const int warm_iters = std::max(1, iters);
    for (int it = 0; st == OB_OK && it < warm_iters; ++it) {
      st = ob_gemm_swiglu_f32_quant(ctx, a.data(), &sd, c_out.data(), &sw);
    }
    (void)ob_context_get_profile(ctx, &p1);
    const uint64_t d_chained = p1.swiglu_chunks_chained - p0.swiglu_chunks_chained;
    const uint64_t d_fb = p1.swiglu_fallbacks - p0.swiglu_fallbacks;
    double nmse = swiglu_ref_nmse(a, gd, ud, dd, c_out, fc.M, fc.K, fc.H, fc.N);
    const bool expect_chained = !(fc.is_q4 != 0u && fc.M > 8u);
    bool ok = (st == OB_OK) &&
              (expect_chained
                   ? (sw.tiles_dispatched == fc.exp_chunks &&
                      sw.cache_hits == fc.exp_chunks &&
                      d_chained == static_cast<uint64_t>(warm_iters) * fc.exp_chunks &&
                      d_fb == 0u)
                   : (sw.tiles_dispatched > 0u &&
                      sw.cache_hits == sw.tiles_dispatched &&
                      d_chained == 0u &&
                      d_fb == static_cast<uint64_t>(warm_iters))) &&
              (nmse < 3.5e-3);
    if (!ok) all_ok = false;
    std::printf("swiglu[%s]: ok=%d pm=%u wl=%d chunks=%u hits=%u cold_us=%llu warm_hw_us=%llu warm_wall_us=%llu nmse=%.2e\n",
                fc.label, ok ? 1 : 0, ctx->cfg.perf_mode, ctx->fw_wakelock_held ? 1 : 0,
                sw.tiles_dispatched, sw.cache_hits,
                static_cast<unsigned long long>(sc.wall_time_us),
                static_cast<unsigned long long>(sw.hw_execution_us),
                static_cast<unsigned long long>(sw.wall_time_us),
                nmse);
    if (profile) print_profile_line(fc.label, ctx);
  }

  ob_context_destroy(ctx);
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  bool mock = false;
  bool profile = false;
  bool emit_gemm = false;
  bool probe_dyn = false;
  bool test_fused_act = false;
  bool test_outlier_spike = false;
  bool test_quant_batch = false;
  bool test_swiglu = false;
  uint32_t gemm_m = 0u, gemm_k = 0u, gemm_n = 0u;
  const char* dive_interp_path = nullptr;
  const char* dgc0_path = nullptr;
  const char* in_path = nullptr;
  const char* ref_path = nullptr;
  int iters = 1;
  int perf_mode = -1;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mock") == 0) {
      mock = true;
    } else if (std::strcmp(argv[i], "--profile") == 0) {
      profile = true;
    } else if (std::strcmp(argv[i], "--probe-dyn-weights") == 0) {
      probe_dyn = true;
    } else if (std::strcmp(argv[i], "--test-fused-act") == 0) {
      test_fused_act = true;
    } else if (std::strcmp(argv[i], "--test-outlier-spike") == 0) {
      test_outlier_spike = true;
    } else if (std::strcmp(argv[i], "--test-quant-batch") == 0) {
      test_quant_batch = true;
    } else if (std::strcmp(argv[i], "--test-swiglu") == 0) {
      test_swiglu = true;
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
    } else if (std::strcmp(argv[i], "--perf-mode") == 0 && i + 1 < argc) {
      perf_mode = std::atoi(argv[++i]);
      if (perf_mode < 0 || perf_mode > 5) {
        std::fprintf(stderr, "--perf-mode must be 0..5\n");
        return 2;
      }
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      print_usage(argv[0]);
      return 2;
    }
  }

  if (test_swiglu) {
    return run_test_swiglu(iters, perf_mode, dive_interp_path, mock, profile);
  }

  if (test_quant_batch) {
    return run_test_quant_batch(iters, perf_mode, dive_interp_path, mock, profile);
  }

  if (test_outlier_spike) {
    return run_test_outlier_spike(dive_interp_path, mock);
  }

  if (test_fused_act) {
    return run_test_fused_act(dive_interp_path, mock);
  }

  if (probe_dyn) {
    return run_probe_dyn_weights(dive_interp_path, mock);
  }

  if (emit_gemm) {
    return run_emit_gemm(gemm_m, gemm_k, gemm_n, iters, perf_mode, dive_interp_path, mock, profile);
  }

  if (mock || !dgc0_path) {
    return run_mock_smoke();
  }

  ob_config_t cfg = ob_config_default();
  if (perf_mode >= 0) cfg.perf_mode = static_cast<uint32_t>(perf_mode);
  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }
  if (profile) ob_context_set_profile(ctx, 1);

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
    if (profile) print_profile_line("dgc0", ctx);
  }

  for (ob_buffer_t* b : out_bufs) ob_buffer_free(b);
  for (ob_buffer_t* b : in_bufs) ob_buffer_free(b);
  ob_graph_release(graph);
  ob_context_destroy(ctx);
  return rc;
}
