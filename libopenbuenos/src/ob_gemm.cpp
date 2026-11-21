#include "ob_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ob {

namespace {

inline uint32_t align_up_u32(uint32_t v, uint32_t a) {
  return ((v + a - 1u) / a) * a;
}

inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t bits = 0;

  if (exp == 0u) {
    if (mant == 0u) {
      bits = sign;
    } else {
      exp = 1u;
      while ((mant & 0x0400u) == 0u) {
        mant <<= 1u;
        --exp;
      }
      mant &= 0x03ffu;
      uint32_t fexp = exp + (127u - 15u);
      bits = sign | (fexp << 23) | (mant << 13);
    }
  } else if (exp == 31u) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    uint32_t fexp = exp + (127u - 15u);
    bits = sign | (fexp << 23) | (mant << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline int8_t quant_sym_i8(float x) {
  long q = std::lround(x);
  if (q < -127) q = -127;
  if (q > 127) q = 127;
  return static_cast<int8_t>(q);
}

inline uint64_t fast_hash64_update(uint64_t h, const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  size_t i = 0;
  for (; i + 8u <= len; i += 8u) {
    uint64_t w = 0ULL;
    std::memcpy(&w, p + i, 8u);
    h ^= w;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= (h >> 31);
  }
  if (i < len) {
    uint64_t tail = 0ULL;
    std::memcpy(&tail, p + i, len - i);
    h ^= tail ^ (static_cast<uint64_t>(len - i) << 56);
    h *= 0x94d049bb133111ebULL;
    h ^= (h >> 31);
  }
  return h;
}

uint64_t hash_tile_key(uint32_t m, uint32_t k, uint32_t n,
                       int32_t zp_a, int32_t zp_b, uint32_t scale_d,
                       const void* weight_data, size_t weight_len) {
  uint64_t h = 14695981039346656037ULL;
  uint32_t hdr[6] = {
      m, k, n,
      static_cast<uint32_t>(zp_a),
      static_cast<uint32_t>(zp_b),
      scale_d
  };
  h = fast_hash64_update(h, hdr, sizeof(hdr));
  if (weight_data && weight_len > 0u) {
    h = fast_hash64_update(h, weight_data, weight_len);
  }
  return h == 0u ? 1u : h;
}

// Weight-agnostic graph cache key for Probe-A dynamic param_slice swapping:
// depends ONLY on (hw_m, hw_k, hw_n, zp_a, zp_b, scale_d, slot_bucket).
uint64_t hash_shape_key(uint32_t hw_m, uint32_t hw_k, uint32_t hw_n,
                        int32_t zp_a, int32_t zp_b, uint32_t scale_d,
                        uint32_t slot_bucket) {
  uint64_t h = 0x50524F4245415F31ULL; // "PROBEA_1"
  uint32_t hdr[7] = {
      hw_m, hw_k, hw_n,
      static_cast<uint32_t>(zp_a),
      static_cast<uint32_t>(zp_b),
      scale_d,
      slot_bucket
  };
  h = fast_hash64_update(h, hdr, sizeof(hdr));
  return h == 0u ? 1u : h;
}

uint32_t get_param_swap_slots() {
  const char* env = std::getenv("OPENBUENOS_PARAM_SWAP_SLOTS");
  if (!env || !env[0]) return 1u;
  unsigned long v = std::strtoul(env, nullptr, 10);
  if (v < 1ul) v = 1ul;
  if (v > OB_MAX_CREDITS) v = OB_MAX_CREDITS;
  return static_cast<uint32_t>(v);
}

void ensure_weight_cache_budget(ob_context* ctx) {
  if (ctx->weight_cache_max_bytes > 0u) return;
  const char* env = std::getenv("OPENBUENOS_WEIGHT_CACHE_MB");
  if (env && env[0]) {
    unsigned long mb = std::strtoul(env, nullptr, 10);
    if (mb < 16ul) mb = 16ul;
    if (mb > 8192ul) mb = 8192ul;
    ctx->weight_cache_max_bytes = static_cast<size_t>(mb) * 1024u * 1024u;
    return;
  }
  // 256 MiB on host mock; 2048 MiB on live 16 GiB Android device (pins Qwen2.5-1.5B Q8_0!).
  ctx->weight_cache_max_bytes = ctx->mock_mode
      ? (256u * 1024u * 1024u)
      : (2048u * 1024u * 1024u);
}

CachedWeightTile* weight_cache_find_locked(ob_context* ctx, uint64_t key) {
  auto it = ctx->weight_cache_map.find(key);
  if (it == ctx->weight_cache_map.end()) return nullptr;
  CachedWeightTile* t = it->second;
  ctx->weight_cache_lru.splice(ctx->weight_cache_lru.begin(), ctx->weight_cache_lru, t->lru_it);
  ctx->weight_cache_hits += 1u;
  return t;
}

void weight_cache_evict_locked(ob_context* ctx, size_t incoming_bytes) {
  ensure_weight_cache_budget(ctx);
  auto it = ctx->weight_cache_lru.end();
  while (ctx->weight_cache_bytes + incoming_bytes > ctx->weight_cache_max_bytes &&
         it != ctx->weight_cache_lru.begin()) {
    --it;
    CachedWeightTile* victim = *it;
    if (victim->in_use > 0u) continue;
    auto erase_it = it;
    ctx->weight_cache_map.erase(victim->key);
    ctx->weight_cache_bytes -= std::min(ctx->weight_cache_bytes, victim->ext2_bytes);
    it = ctx->weight_cache_lru.erase(erase_it);
    if (victim->ext2_buf) ob_buffer_free(victim->ext2_buf);
    delete victim;
  }
}

void weight_cache_insert_locked(ob_context* ctx, CachedWeightTile* tile) {
  weight_cache_evict_locked(ctx, tile->ext2_bytes);
  ctx->weight_cache_lru.push_front(tile);
  tile->lru_it = ctx->weight_cache_lru.begin();
  ctx->weight_cache_map[tile->key] = tile;
  ctx->weight_cache_bytes += tile->ext2_bytes;
  ctx->weight_cache_misses += 1u;
}

// Compute token-invariant integer scale ratio D (1/8-octave ceiling) solely from
// the static quantized weight tile b_tile[tn, tk] so Tensor G5 real_mul = 1/D
// avoids int8 clipping ([-128, 127]) while keeping quantized NMSE < 5e-4 (uniform)
// and < 1e-3 (Gaussian).
uint32_t compute_tile_scale_D_impl(const int8_t* b_tile, uint32_t tn, uint32_t tk,
                                   uint32_t stride_k, int32_t zp_a) {
  double max_l2 = 0.0;
  uint64_t max_l1 = 0ULL;
  uint64_t max_abs_sum = 0ULL;

  for (uint32_t ni = 0; ni < tn; ++ni) {
    const int8_t* row = b_tile + static_cast<size_t>(ni) * stride_k;
    uint64_t sq = 0ULL;
    uint64_t l1 = 0ULL;
    int64_t ssum = 0LL;
    for (uint32_t ki = 0; ki < tk; ++ki) {
      int32_t v = static_cast<int32_t>(row[ki]);
      sq += static_cast<uint64_t>(v * v);
      l1 += static_cast<uint64_t>(v < 0 ? -v : v);
      ssum += static_cast<int64_t>(v);
    }
    double l2 = std::sqrt(static_cast<double>(sq));
    uint64_t asum = static_cast<uint64_t>(ssum < 0 ? -ssum : ssum);
    if (l2 > max_l2) max_l2 = l2;
    if (l1 > max_l1) max_l1 = l1;
    if (asum > max_abs_sum) max_abs_sum = asum;
  }

  if (max_l1 == 0ULL) return 1u;

  double zp_scale = 1.0 + static_cast<double>(zp_a < 0 ? -zp_a : zp_a) / 128.0;
  double hard_cap = static_cast<double>(max_l1) * (128.0 / 127.0) * zp_scale;
  double stat_est = std::max(2.6 * max_l2, static_cast<double>(max_abs_sum) * 1.05) * zp_scale;
  double raw_d = std::min(hard_cap, stat_est);
  if (raw_d <= 1.0) return 1u;

  // Quantize upward to 1/8-octave exact integer D = m * 2^(p-3) (m in [8..16]).
  // Never rounds down below raw_d (>= 4.5 sigma -> zero clipping) while bounding
  // overshoot to <= 12.5% (4x lower MSE than power-of-two ceiling).
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
  if (d < 1ULL) d = 1ULL;
  if (d > (1ULL << 24)) d = (1ULL << 24);
  return static_cast<uint32_t>(d);
}

ob_status_t ensure_pipeline_buf(ob_context* ctx,
                                ob_buffer_t** slot_buf,
                                size_t needed_bytes,
                                uint32_t flags) {
  size_t req = std::max<size_t>(needed_bytes, 4096u);
  if (*slot_buf && ob_buffer_size(*slot_buf) >= req) {
    return OB_OK;
  }
  if (*slot_buf) {
    ob_buffer_free(*slot_buf);
    *slot_buf = nullptr;
  }
  return ob_buffer_alloc(ctx, req, flags, slot_buf);
}

// Acquire cached graph by key, invoking emitter ONLY on cache miss.
ob_status_t acquire_tile_graph(ob_context* ctx,
                               ob_gemm_emitter_fn emitter,
                               void* user_data,
                               const ob_gemm_desc_t& desc,
                               ob_graph_t** out_graph,
                               bool* out_cache_hit) {
  *out_graph = nullptr;
  *out_cache_hit = false;

  if (desc.cache_key != 0u && graph_cache_lookup_u64(ctx, desc.cache_key, out_graph)) {
    *out_cache_hit = true;
    return OB_OK;
  }

  if (ctx->emitter_scratch.size() < 1024u * 1024u) {
    ctx->emitter_scratch.resize(1024u * 1024u);
  }
  size_t dgc0_size = 0u;
  ob_status_t st = emitter(&desc, ctx->emitter_scratch.data(),
                           ctx->emitter_scratch.size(), &dgc0_size, user_data);
  if (st == OB_ERR_OVERFLOW && dgc0_size > ctx->emitter_scratch.size()) {
    ctx->emitter_scratch.resize(dgc0_size);
    st = emitter(&desc, ctx->emitter_scratch.data(),
                 ctx->emitter_scratch.size(), &dgc0_size, user_data);
  }
  if (st != OB_OK) return st;

  st = ob_graph_register_dgc0(ctx, ctx->emitter_scratch.data(), dgc0_size,
                              desc.cache_key, out_graph);
  if (st != OB_OK) return st;

  ob_dgc0_info_t ginfo {};
  ob_graph_get_info(*out_graph, &ginfo);
  *out_cache_hit = (ginfo.cache_hit != 0u);
  return OB_OK;
}

struct MSlice {
  uint32_t m0;
  uint32_t tm;
  uint32_t hw_m;
};

std::vector<MSlice> build_m_slices(uint32_t m_dim, uint32_t max_m,
                                   bool builtin_active, bool allow_m_pad) {
  std::vector<MSlice> out;
  if (!builtin_active) {
    for (uint32_t m0 = 0; m0 < m_dim; m0 += max_m) {
      uint32_t tm = std::min(max_m, m_dim - m0);
      out.push_back({m0, tm, tm});
    }
    return out;
  }
  if (allow_m_pad) {
    // Builtin Tensor G5 emitter (zp_a == 0): pad M in [2..7] -> hw_m=8 and
    // M in [9..15] -> hw_m=16 so every multi-row tile runs on silicon-verified
    // hw_m in {1, 8, 16} (avoiding buenos_a0's M in {2,4}, K=64 schedule bug
    // and maximizing graph cache reuse across tail slices like M=509).
    uint32_t m_cap = (max_m >= 16u) ? 16u : ((max_m >= 8u) ? 8u : 1u);
    for (uint32_t m0 = 0u; m0 < m_dim; ) {
      uint32_t tm = std::min(m_cap, m_dim - m0);
      uint32_t hw_m = (tm == 1u) ? 1u : ((tm <= 8u) ? 8u : 16u);
      out.push_back({m0, tm, hw_m});
      m0 += tm;
    }
    return out;
  }
  // Asymmetric zp_a != 0 fallback: exact power-of-two slices <= 16 (caller clamps K<=32 for tm<8).
  uint32_t m_cap = std::min<uint32_t>(max_m, 16u);
  uint32_t m0 = 0u;
  while (m0 < m_dim) {
    uint32_t rem = std::min<uint32_t>(m_dim - m0, m_cap);
    uint32_t tm = 1u;
    if (rem >= 16u) tm = 16u;
    else if (rem >= 8u) tm = 8u;
    else if (rem >= 4u) tm = 4u;
    else if (rem >= 2u) tm = 2u;
    out.push_back({m0, tm, tm});
    m0 += tm;
  }
  return out;
}

void compute_builtin_hw_dims(uint32_t hw_m, uint32_t tk, uint32_t tn,
                             uint32_t* out_hw_k, uint32_t* out_hw_n) {
  uint32_t hk = std::max<uint32_t>(4u, align_up_u32(tk, 4u));
  uint32_t hn = std::max<uint32_t>(4u, align_up_u32(tn, 4u));
  if (hw_m == 1u) {
    if (hk > 64u && (hk % 32u) != 0u) {
      hk = align_up_u32(hk, 32u);
      if (hk > 512u) hk = 512u;
    }
    if (hk == 512u && hn > 64u) {
      // Silicon-proven 1x512x128 streaming path (sw=4, n_groups=4).
      hn = 128u;
    } else if (hn > 32u && hk > 48u && hk <= 64u) {
      hk = 64u;
      hn = 64u;
    }
  } else {
    // M >= 2: pin to silicon-verified 1D scatter/gather tile shapes
    // (and never emit K=64 for hw_m < 8).
    hn = (hn > 32u) ? 64u : 32u;
    if (hk <= 16u) {
      hk = 16u;
    } else if (hk <= 32u || hw_m < 8u) {
      hk = 32u;
    } else {
      hk = 64u;
      hn = 64u;
    }
  }
  *out_hw_k = hk;
  *out_hw_n = hn;
}

void cpu_matmul_int8_tile(const int8_t* a_mk,
                          const int8_t* b_nk,
                          int32_t* c_mn,
                          uint32_t k_dim,
                          uint32_t n_dim,
                          uint32_t m0, uint32_t tm,
                          uint32_t k0, uint32_t tk,
                          uint32_t n0, uint32_t tn,
                          int32_t zp_a, int32_t zp_b) {
  for (uint32_t mi = 0; mi < tm; ++mi) {
    const int8_t* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0;
    int32_t* c_row = c_mn + static_cast<size_t>(m0 + mi) * n_dim + n0;
    for (uint32_t ni = 0; ni < tn; ++ni) {
      const int8_t* b_row = b_nk + static_cast<size_t>(n0 + ni) * k_dim + k0;
      int32_t acc = 0;
      if (zp_a == 0 && zp_b == 0) {
        for (uint32_t ki = 0; ki < tk; ++ki) {
          acc += static_cast<int32_t>(a_row[ki]) * static_cast<int32_t>(b_row[ki]);
        }
      } else {
        for (uint32_t ki = 0; ki < tk; ++ki) {
          acc += (static_cast<int32_t>(a_row[ki]) - zp_a) *
                 (static_cast<int32_t>(b_row[ki]) - zp_b);
        }
      }
      c_row[ni] += acc;
    }
  }
}

void cpu_matmul_f32_q8_0_tile(const float* a_mk,
                              const ob_block_q8_0_t* b_blocks,
                              float* c_mn,
                              uint32_t k_dim,
                              uint32_t n_dim,
                              uint32_t m0, uint32_t tm,
                              uint32_t k0, uint32_t tk,
                              uint32_t n0, uint32_t tn) {
  const uint32_t nb_k = k_dim / OB_QK8_0;
  const uint32_t blk0 = k0 / OB_QK8_0;
  const uint32_t blk_cnt = tk / OB_QK8_0;

  for (uint32_t mi = 0; mi < tm; ++mi) {
    const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim;
    float* c_row = c_mn + static_cast<size_t>(m0 + mi) * n_dim + n0;
    for (uint32_t ni = 0; ni < tn; ++ni) {
      const ob_block_q8_0_t* b_row =
          b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
      float acc = 0.0f;
      for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
        const float d = fp16_to_fp32(b_row[bi].d);
        const int8_t* qs = b_row[bi].qs;
        const float* a_blk = a_row + static_cast<size_t>(blk0 + bi) * OB_QK8_0;
        float bsum = 0.0f;
        for (uint32_t j = 0; j < OB_QK8_0; ++j) {
          bsum += a_blk[j] * static_cast<float>(qs[j]);
        }
        acc += d * bsum;
      }
      c_row[ni] += acc;
    }
  }
}

}  // namespace

uint32_t compute_tile_scale_D(const int8_t* b_tile, uint32_t tn, uint32_t tk,
                              uint32_t stride_k, int32_t zp_a) {
  return compute_tile_scale_D_impl(b_tile, tn, tk, stride_k, zp_a);
}

void gemm_release_pipeline_buffers(ob_context* ctx) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);
  for (uint32_t i = 0; i < OB_MAX_CREDITS; ++i) {
    if (ctx->gemm_in_bufs[i]) {
      ob_buffer_free(ctx->gemm_in_bufs[i]);
      ctx->gemm_in_bufs[i] = nullptr;
    }
    if (ctx->gemm_out_bufs[i]) {
      ob_buffer_free(ctx->gemm_out_bufs[i]);
      ctx->gemm_out_bufs[i] = nullptr;
    }
  }
  for (CachedWeightTile* t : ctx->weight_cache_lru) {
    if (t->ext2_buf) ob_buffer_free(t->ext2_buf);
    delete t;
  }
  ctx->weight_cache_lru.clear();
  ctx->weight_cache_map.clear();
  ctx->weight_cache_bytes = 0;
}

}  // namespace ob

extern "C" {

size_t ob_gemm_packed_weight_bytes(uint32_t k_dim, uint32_t n_dim) {
  if (k_dim == 0u || n_dim == 0u) return 0u;
  const size_t n_pad = static_cast<size_t>(ob::align_up_u32(n_dim, 8u));
  const size_t k_pad = static_cast<size_t>(ob::align_up_u32(k_dim, 4u));
  return n_pad * sizeof(int32_t) + (k_pad / 4u) * n_pad * 4u;
}

ob_status_t ob_gemm_pack_weights_int8(const int8_t* weights_nk,
                                      const int32_t* bias_n,
                                      uint32_t k_dim,
                                      uint32_t n_dim,
                                      uint8_t* out_packed,
                                      size_t out_cap) {
  if (!weights_nk || !out_packed || k_dim == 0u || n_dim == 0u) {
    return OB_ERR_INVALID_ARG;
  }
  const size_t needed = ob_gemm_packed_weight_bytes(k_dim, n_dim);
  if (out_cap < needed) return OB_ERR_OVERFLOW;

  const uint32_t n_pad = ob::align_up_u32(n_dim, 8u);
  const uint32_t k_pad = ob::align_up_u32(k_dim, 4u);
  uint8_t* p = out_packed;

  for (uint32_t ni = 0; ni < n_pad; ++ni) {
    int32_t b = (bias_n && ni < n_dim) ? bias_n[ni] : 0;
    uint32_t u = static_cast<uint32_t>(b);
    p[0] = static_cast<uint8_t>(u & 0xffu);
    p[1] = static_cast<uint8_t>((u >> 8) & 0xffu);
    p[2] = static_cast<uint8_t>((u >> 16) & 0xffu);
    p[3] = static_cast<uint8_t>((u >> 24) & 0xffu);
    p += 4;
  }

  const uint32_t k_groups = k_pad / 4u;
  for (uint32_t k4 = 0; k4 < k_groups; ++k4) {
    for (uint32_t ni = 0; ni < n_pad; ++ni) {
      if (ni < n_dim) {
        const int8_t* w_row = weights_nk + static_cast<size_t>(ni) * k_dim;
        for (uint32_t lane = 0; lane < 4u; ++lane) {
          uint32_t ki = k4 * 4u + lane;
          int32_t w = (ki < k_dim) ? static_cast<int32_t>(w_row[ki]) : 0;
          *p++ = static_cast<uint8_t>((w + 128) & 0xff);
        }
      } else {
        p[0] = 0u;
        p[1] = 0u;
        p[2] = 0u;
        p[3] = 0u;
        p += 4;
      }
    }
  }
  return OB_OK;
}

ob_status_t ob_gemm_register_emitter(ob_context_t* ctx,
                                     ob_gemm_emitter_fn fn,
                                     void* user_data) {
  if (!ctx) return OB_ERR_INVALID_ARG;
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);
  ctx->gemm_emitter = fn;
  ctx->gemm_emitter_user_data = user_data;
  ctx->builtin_emitter_active = false;
  return OB_OK;
}

ob_status_t ob_gemm_set_tile_limits(ob_context_t* ctx,
                                    uint32_t max_m,
                                    uint32_t max_k,
                                    uint32_t max_n) {
  if (!ctx || max_m == 0u || max_k == 0u || max_n == 0u) {
    return OB_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);
  ctx->tile_max_m = max_m;
  ctx->tile_max_k = max_k;
  ctx->tile_max_n = max_n;
  return OB_OK;
}

ob_status_t ob_gemm_matmul_int8(ob_context_t* ctx,
                                const int8_t* a_mk,
                                const int8_t* b_nk,
                                const int32_t* bias_n,
                                int32_t* c_mn,
                                uint32_t m_dim,
                                uint32_t k_dim,
                                uint32_t n_dim,
                                int32_t zp_a,
                                int32_t zp_b,
                                ob_invoke_stats_t* out_stats) {
  if (!ctx || !a_mk || !b_nk || !c_mn || m_dim == 0u || k_dim == 0u || n_dim == 0u) {
    return OB_ERR_INVALID_ARG;
  }

  const uint64_t t0_ns = ob::monotonic_now_ns();
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);

  const uint32_t max_m = ctx->tile_max_m;
  const uint32_t max_k = ctx->tile_max_k;
  const uint32_t max_n = ctx->tile_max_n;
  const ob_gemm_emitter_fn emitter = ctx->gemm_emitter;
  void* const emitter_ud = ctx->gemm_emitter_user_data;
  const bool builtin_active = ctx->builtin_emitter_active;
  const uint32_t ring_depth = std::max<uint32_t>(1u, std::min<uint32_t>(ctx->credit_size, OB_MAX_CREDITS));

  // Initialize output accumulator with optional bias
  for (uint32_t m = 0; m < m_dim; ++m) {
    int32_t* c_row = c_mn + static_cast<size_t>(m) * n_dim;
    if (bias_n) {
      std::memcpy(c_row, bias_n, static_cast<size_t>(n_dim) * sizeof(int32_t));
    } else {
      std::memset(c_row, 0, static_cast<size_t>(n_dim) * sizeof(int32_t));
    }
  }

  uint32_t tiles_dispatched = 0u;
  uint32_t cache_hits = 0u;
  uint64_t total_hw_us = 0u;
  bool used_emitter = false;
  bool used_cpu_fallback = false;

  struct PendingInt8Tile {
    ob_exec_handle_t handle = 0u;
    ob_graph_t* graph = nullptr;
    ob::CachedWeightTile* w_tile = nullptr;
    uint32_t slot = 0u;
    uint32_t m0 = 0u, tm = 0u, hw_m = 0u;
    uint32_t n0 = 0u, tn = 0u;
    uint32_t hw_n = 0u;
    uint32_t scale_d = 1u;
  };
  std::array<PendingInt8Tile, OB_MAX_CREDITS> ring{};
  uint32_t ring_head = 0u;
  uint32_t ring_count = 0u;
  std::vector<int8_t> out_tile;

  auto drain_one = [&]() -> ob_status_t {
    PendingInt8Tile& p = ring[ring_head];
    ob_invoke_stats_t istats {};
    ob_status_t wst = ob_graph_wait(ctx, p.handle, ctx->cfg.timeout_ms, &istats);
    ob_status_t rst = OB_OK;
    if (wst == OB_OK) {
      total_hw_us += istats.hw_execution_us;
      if (!ctx->mock_mode) {
        size_t out_bytes = static_cast<size_t>(p.hw_m) * p.hw_n;
        out_tile.resize(out_bytes);
        rst = ob_buffer_read(ctx->gemm_out_bufs[p.slot], out_tile.data(), out_bytes, 0u);
        if (rst == OB_OK) {
          const int32_t d_i32 = static_cast<int32_t>(p.scale_d);
          for (uint32_t mi = 0; mi < p.tm; ++mi) {
            int32_t* c_row = c_mn + static_cast<size_t>(p.m0 + mi) * n_dim + p.n0;
            const int8_t* o_row = out_tile.data() + static_cast<size_t>(mi) * p.hw_n;
            for (uint32_t ni = 0; ni < p.tn; ++ni) {
              c_row[ni] += static_cast<int32_t>(o_row[ni]) * d_i32;
            }
          }
        }
      }
    }
    if (p.w_tile && p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
    ob_graph_release(p.graph);
    p = PendingInt8Tile{};
    ring_head = (ring_head + 1u) % ring_depth;
    ring_count -= 1u;
    return (wst != OB_OK) ? wst : rst;
  };

  auto drain_all_cleanup = [&]() {
    while (ring_count > 0u) {
      PendingInt8Tile& p = ring[ring_head];
      ob_graph_wait(ctx, p.handle, ctx->cfg.timeout_ms, nullptr);
      if (p.w_tile && p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
      ob_graph_release(p.graph);
      p = PendingInt8Tile{};
      ring_head = (ring_head + 1u) % ring_depth;
      ring_count -= 1u;
    }
  };

  std::vector<int8_t> a_tile;
  std::vector<int8_t> b_tile;
  std::vector<uint8_t> packed_tile;
  const uint32_t param_swap_slots = ob::get_param_swap_slots();

  const bool allow_m_pad = (zp_a == 0 && zp_b == 0);
  std::vector<ob::MSlice> m_slices = ob::build_m_slices(m_dim, max_m, builtin_active, allow_m_pad);

  for (const ob::MSlice& ms : m_slices) {
    const uint32_t m0 = ms.m0;
    const uint32_t tm = ms.tm;
    const uint32_t hw_m = ms.hw_m;
    const uint32_t k_cap = (hw_m == 1u) ? 512u : ((hw_m >= 8u) ? 64u : 32u);
    const uint32_t eff_max_k = builtin_active ? std::min<uint32_t>(max_k, k_cap) : max_k;

    for (uint32_t k0 = 0; k0 < k_dim; k0 += eff_max_k) {
      const uint32_t tk = std::min(eff_max_k, k_dim - k0);
      const uint32_t n_cap = (hw_m == 1u && tk == 512u) ? 128u : 64u;
      const uint32_t eff_max_n = builtin_active ? std::min<uint32_t>(max_n, n_cap) : max_n;

      for (uint32_t n0 = 0; n0 < n_dim; n0 += eff_max_n) {
        const uint32_t tn = std::min(eff_max_n, n_dim - n0);
        tiles_dispatched += 1u;

        bool tile_handled_by_hw = false;
        if (emitter != nullptr) {
          uint32_t hw_k = tk;
          uint32_t hw_n = tn;
          if (builtin_active) {
            ob::compute_builtin_hw_dims(hw_m, tk, tn, &hw_k, &hw_n);
          }

          // Zero-padding only preserves exact semantics when zp_a == 0 and zp_b == 0.
          const bool needs_pad = (hw_m != tm || hw_k != tk || hw_n != tn);
          const bool can_hw = !(builtin_active && (zp_b != 0 || (needs_pad && zp_a != 0)));

          if (can_hw) {
            a_tile.assign(static_cast<size_t>(hw_m) * hw_k, 0);
            for (uint32_t mi = 0; mi < tm; ++mi) {
              std::memcpy(a_tile.data() + static_cast<size_t>(mi) * hw_k,
                          a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0, tk);
            }

            ob::CachedWeightTile* w_tile = nullptr;
            uint32_t scale_d = 1u;
            if (builtin_active) {
              // Fast content hash of raw int8 weight tile rows [n0..n0+tn) x [k0..k0+tk)
              uint64_t wh = 0x49385F5754494C45ULL; // "I8_WTILE"
              uint32_t whdr[5] = {tn, tk, hw_k, hw_n, static_cast<uint32_t>(zp_a)};
              wh = ob::fast_hash64_update(wh, whdr, sizeof(whdr));
              for (uint32_t ni = 0; ni < tn; ++ni) {
                wh = ob::fast_hash64_update(
                    wh, b_nk + static_cast<size_t>(n0 + ni) * k_dim + k0, tk);
              }
              if (wh == 0u) wh = 1u;

              w_tile = ob::weight_cache_find_locked(ctx, wh);
              if (!w_tile) {
                b_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
                for (uint32_t ni = 0; ni < tn; ++ni) {
                  std::memcpy(b_tile.data() + static_cast<size_t>(ni) * hw_k,
                              b_nk + static_cast<size_t>(n0 + ni) * k_dim + k0, tk);
                }
                scale_d = ob::compute_tile_scale_D(b_tile.data(), tn, tk, hw_k, zp_a);
                std::vector<uint8_t> ext2 = ob::pack_ext2_for_tile(hw_n, hw_k, b_tile.data(), nullptr);
                ob_buffer* wbuf = nullptr;
                ob_status_t wst = ob_buffer_alloc(
                    ctx, ext2.size(), OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL | OB_BUF_ZERO_INIT, &wbuf);
                if (wst == OB_OK) {
                  wst = ob_buffer_write(wbuf, ext2.data(), ext2.size(), 0u);
                }
                if (wst != OB_OK) {
                  if (wbuf) ob_buffer_free(wbuf);
                  drain_all_cleanup();
                  return wst;
                }
                w_tile = new (std::nothrow) ob::CachedWeightTile();
                if (!w_tile) {
                  ob_buffer_free(wbuf);
                  drain_all_cleanup();
                  return OB_ERR_NO_MEMORY;
                }
                w_tile->key = wh;
                w_tile->scale_d = scale_d;
                w_tile->hw_k = hw_k;
                w_tile->hw_n = hw_n;
                w_tile->ext2_buf = wbuf;
                w_tile->ext2_bytes = ext2.size();
                ob::weight_cache_insert_locked(ctx, w_tile);
              } else {
                scale_d = w_tile->scale_d;
              }
            } else {
              b_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
              for (uint32_t ni = 0; ni < tn; ++ni) {
                std::memcpy(b_tile.data() + static_cast<size_t>(ni) * hw_k,
                            b_nk + static_cast<size_t>(n0 + ni) * k_dim + k0, tk);
              }
              scale_d = ob::compute_tile_scale_D(b_tile.data(), tn, tk, hw_k, zp_a);
            }

            // Ensure a free credit slot BEFORE acquiring graph so slot_bucket is known.
            if (ring_count >= ring_depth) {
              ob_status_t dst = drain_one();
              if (dst != OB_OK) {
                drain_all_cleanup();
                return dst;
              }
            }
            const uint32_t slot = (ring_head + ring_count) % ring_depth;

            ob_gemm_desc_t desc {};
            desc.m_dim = hw_m;
            desc.k_dim = hw_k;
            desc.n_dim = hw_n;
            desc.s_in = 1.0f;
            desc.zp_in = zp_a;
            desc.s_w = 1.0f;
            desc.zp_w = zp_b;
            desc.s_out = static_cast<float>(scale_d);
            desc.zp_out = 0;
            desc.activation = OB_ACT_NONE;
            desc.bias_n = nullptr;
            if (builtin_active) {
              // Weight-agnostic graph keyed ONLY by (hw_m, hw_k, hw_n, zp_a, zp_b, scale_d, slot_bucket)!
              desc.weights_nk = nullptr;
              desc.cache_key = ob::hash_shape_key(
                  hw_m, hw_k, hw_n, zp_a, zp_b, scale_d, slot % param_swap_slots);
            } else {
              desc.weights_nk = b_tile.data();
              const size_t packed_sz = ob_gemm_packed_weight_bytes(hw_k, hw_n);
              packed_tile.resize(packed_sz);
              ob_status_t pst = ob_gemm_pack_weights_int8(
                  b_tile.data(), nullptr, hw_k, hw_n, packed_tile.data(), packed_tile.size());
              if (pst != OB_OK) {
                drain_all_cleanup();
                return pst;
              }
              desc.packed_params = packed_tile.data();
              desc.packed_params_bytes = packed_tile.size();
              desc.cache_key = ob::hash_tile_key(
                  hw_m, hw_k, hw_n, zp_a, zp_b, scale_d, b_tile.data(), b_tile.size());
            }

            ob_graph_t* graph = nullptr;
            bool was_hit = false;
            ob_status_t gst = ob::acquire_tile_graph(
                ctx, emitter, emitter_ud, desc, &graph, &was_hit);
            if (gst == OB_OK) {
              used_emitter = true;
              if (was_hit) cache_hits += 1u;

              ob_dgc0_info_t ginfo {};
              ob_graph_get_info(graph, &ginfo);
              size_t in_alloc = std::max<size_t>(
                  a_tile.size(),
                  ginfo.num_inputs > 0u ? static_cast<size_t>(ginfo.input_sizes[0]) : 1u);
              size_t out_alloc = std::max<size_t>(
                  static_cast<size_t>(hw_m) * ob::align_up_u32(hw_n, 4u),
                  ginfo.num_outputs > 0u ? static_cast<size_t>(ginfo.output_sizes[0]) : 1u);

              ob_status_t bst = ob::ensure_pipeline_buf(
                  ctx, &ctx->gemm_in_bufs[slot], in_alloc, OB_BUF_TO_DEVICE | OB_BUF_ZERO_INIT);
              if (bst == OB_OK) {
                bst = ob::ensure_pipeline_buf(
                    ctx, &ctx->gemm_out_bufs[slot], out_alloc, OB_BUF_FROM_DEVICE | OB_BUF_ZERO_INIT);
              }
              if (bst == OB_OK) {
                bst = ob_buffer_write(ctx->gemm_in_bufs[slot], a_tile.data(), a_tile.size(), 0u);
              }

              ob_exec_handle_t exec_h = 0u;
              if (bst == OB_OK) {
                if (w_tile != nullptr) {
                  bst = ob::graph_submit_async_with_params(
                      graph, &ctx->gemm_in_bufs[slot], 1u, &ctx->gemm_out_bufs[slot], 1u,
                      w_tile->ext2_buf, &exec_h);
                } else {
                  bst = ob_graph_submit_async(
                      graph, &ctx->gemm_in_bufs[slot], 1u, &ctx->gemm_out_bufs[slot], 1u, &exec_h);
                }
              }
              if (bst != OB_OK) {
                ob_graph_release(graph);
                drain_all_cleanup();
                return bst;
              }

              PendingInt8Tile& p = ring[slot];
              p.handle = exec_h;
              p.graph = graph;
              p.w_tile = w_tile;
              if (w_tile) w_tile->in_use += 1u;
              p.slot = slot;
              p.m0 = m0;
              p.tm = tm;
              p.hw_m = hw_m;
              p.n0 = n0;
              p.tn = tn;
              p.hw_n = hw_n;
              p.scale_d = scale_d;
              ring_count += 1u;

              if (!ctx->mock_mode) {
                tile_handled_by_hw = true;
              }
            } else if (gst != OB_ERR_UNSUPPORTED) {
              drain_all_cleanup();
              return gst;
            }
          }
        }

        if (!tile_handled_by_hw) {
          used_cpu_fallback = true;
          ob::cpu_matmul_int8_tile(a_mk, b_nk, c_mn, k_dim, n_dim,
                                   m0, tm, k0, tk, n0, tn, zp_a, zp_b);
        }
      }
    }
  }

  // Drain all remaining in-flight TPU dispatches
  while (ring_count > 0u) {
    ob_status_t dst = drain_one();
    if (dst != OB_OK) {
      drain_all_cleanup();
      return dst;
    }
  }

  if (out_stats) {
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->hw_execution_us = total_hw_us;
    out_stats->wall_time_us = (ob::monotonic_now_ns() - t0_ns) / 1000ULL;
    out_stats->tiles_dispatched = tiles_dispatched;
    out_stats->cache_hits = cache_hits;
    if (ctx->mock_mode) out_stats->flags |= OB_STAT_FLAG_MOCK;
    if (used_emitter) out_stats->flags |= OB_STAT_FLAG_EMITTER_USED;
    if (used_cpu_fallback) out_stats->flags |= OB_STAT_FLAG_CPU_FALLBACK;
    if (cache_hits > 0u) out_stats->flags |= OB_STAT_FLAG_CACHE_HIT;
  }
  return OB_OK;
}

ob_status_t ob_gemm_matmul_f32_q8_0(ob_context_t* ctx,
                                    const float* a_mk,
                                    const void* b_nk_q8_0,
                                    float* c_mn,
                                    uint32_t m_dim,
                                    uint32_t k_dim,
                                    uint32_t n_dim,
                                    ob_invoke_stats_t* out_stats) {
  if (!ctx || !a_mk || !b_nk_q8_0 || !c_mn ||
      m_dim == 0u || k_dim == 0u || n_dim == 0u || (k_dim % OB_QK8_0) != 0u) {
    return OB_ERR_INVALID_ARG;
  }

  const uint64_t t0_ns = ob::monotonic_now_ns();
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);

  const uint32_t max_m = ctx->tile_max_m;
  const uint32_t max_k = std::max<uint32_t>(OB_QK8_0, (ctx->tile_max_k / OB_QK8_0) * OB_QK8_0);
  const uint32_t max_n = ctx->tile_max_n;
  const ob_gemm_emitter_fn emitter = ctx->gemm_emitter;
  void* const emitter_ud = ctx->gemm_emitter_user_data;
  const bool builtin_active = ctx->builtin_emitter_active;
  const uint32_t ring_depth = std::max<uint32_t>(1u, std::min<uint32_t>(ctx->credit_size, OB_MAX_CREDITS));

  const ob_block_q8_0_t* b_blocks = static_cast<const ob_block_q8_0_t*>(b_nk_q8_0);
  std::memset(c_mn, 0, static_cast<size_t>(m_dim) * n_dim * sizeof(float));

  uint32_t tiles_dispatched = 0u;
  uint32_t cache_hits = 0u;
  uint64_t total_hw_us = 0u;
  bool used_emitter = false;
  bool used_cpu_fallback = false;

  struct PendingF32Tile {
    ob_exec_handle_t handle = 0u;
    ob_graph_t* graph = nullptr;
    ob::CachedWeightTile* w_tile = nullptr;
    uint32_t slot = 0u;
    uint32_t m0 = 0u, tm = 0u, hw_m = 0u;
    uint32_t n0 = 0u, tn = 0u;
    uint32_t hw_n = 0u;
    float scale_d = 1.0f;
    std::vector<float> s_a;
    std::vector<float> s_b;
  };
  std::array<PendingF32Tile, OB_MAX_CREDITS> ring{};
  uint32_t ring_head = 0u;
  uint32_t ring_count = 0u;
  std::vector<int8_t> out_tile;

  auto drain_one = [&]() -> ob_status_t {
    PendingF32Tile& p = ring[ring_head];
    ob_invoke_stats_t istats {};
    ob_status_t wst = ob_graph_wait(ctx, p.handle, ctx->cfg.timeout_ms, &istats);
    ob_status_t rst = OB_OK;
    if (wst == OB_OK) {
      total_hw_us += istats.hw_execution_us;
      if (!ctx->mock_mode) {
        size_t out_bytes = static_cast<size_t>(p.hw_m) * p.hw_n;
        out_tile.resize(out_bytes);
        rst = ob_buffer_read(ctx->gemm_out_bufs[p.slot], out_tile.data(), out_bytes, 0u);
        if (rst == OB_OK) {
          const float* sb_ptr = p.w_tile ? p.w_tile->row_s_b.data() : p.s_b.data();
          for (uint32_t mi = 0; mi < p.tm; ++mi) {
            const float sa_d = p.s_a[mi] * p.scale_d;
            float* c_row = c_mn + static_cast<size_t>(p.m0 + mi) * n_dim + p.n0;
            const int8_t* o_row = out_tile.data() + static_cast<size_t>(mi) * p.hw_n;
            for (uint32_t ni = 0; ni < p.tn; ++ni) {
              c_row[ni] += static_cast<float>(o_row[ni]) * (sa_d * sb_ptr[ni]);
            }
          }
        }
      }
    }
    if (p.w_tile && p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
    ob_graph_release(p.graph);
    p.handle = 0u;
    p.graph = nullptr;
    p.w_tile = nullptr;
    ring_head = (ring_head + 1u) % ring_depth;
    ring_count -= 1u;
    return (wst != OB_OK) ? wst : rst;
  };

  auto drain_all_cleanup = [&]() {
    while (ring_count > 0u) {
      PendingF32Tile& p = ring[ring_head];
      ob_graph_wait(ctx, p.handle, ctx->cfg.timeout_ms, nullptr);
      if (p.w_tile && p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
      ob_graph_release(p.graph);
      p.handle = 0u;
      p.graph = nullptr;
      p.w_tile = nullptr;
      ring_head = (ring_head + 1u) % ring_depth;
      ring_count -= 1u;
    }
  };

  const uint32_t nb_k = k_dim / OB_QK8_0;
  std::vector<int8_t> a_q_tile;
  std::vector<int8_t> b_q_tile;
  std::vector<float> tmp_w_row;
  std::vector<float> row_s_a;
  std::vector<float> row_s_b;
  std::vector<uint8_t> packed_tile;
  const uint32_t param_swap_slots = ob::get_param_swap_slots();

  std::vector<ob::MSlice> m_slices = ob::build_m_slices(m_dim, max_m, builtin_active, true);

  for (const ob::MSlice& ms : m_slices) {
    const uint32_t m0 = ms.m0;
    const uint32_t tm = ms.tm;
    const uint32_t hw_m = ms.hw_m;
    const uint32_t k_cap = (hw_m == 1u) ? 512u : ((hw_m >= 8u) ? 64u : 32u);
    uint32_t eff_max_k = builtin_active ? std::min<uint32_t>(max_k, k_cap) : max_k;
    eff_max_k = std::max<uint32_t>(OB_QK8_0, (eff_max_k / OB_QK8_0) * OB_QK8_0);

    for (uint32_t k0 = 0; k0 < k_dim; k0 += eff_max_k) {
      const uint32_t tk = std::min(eff_max_k, k_dim - k0);
      const uint32_t blk0 = k0 / OB_QK8_0;
      const uint32_t blk_cnt = tk / OB_QK8_0;
      const uint32_t n_cap = (hw_m == 1u && tk == 512u) ? 128u : 64u;
      const uint32_t eff_max_n = builtin_active ? std::min<uint32_t>(max_n, n_cap) : max_n;

      // Quantize A rows for this (m0, k0) slice once and reuse across all N-tiles!
      row_s_a.resize(tm);
      uint32_t cached_aq_hw_k = 0u;
      if (emitter != nullptr) {
        for (uint32_t mi = 0; mi < tm; ++mi) {
          const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0;
          float amax = 0.0f;
          for (uint32_t ki = 0; ki < tk; ++ki) {
            float v = std::fabs(a_row[ki]);
            if (v > amax) amax = v;
          }
          row_s_a[mi] = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        }
      }

      for (uint32_t n0 = 0; n0 < n_dim; n0 += eff_max_n) {
        const uint32_t tn = std::min(eff_max_n, n_dim - n0);
        tiles_dispatched += 1u;

        bool tile_handled_by_hw = false;
        if (emitter != nullptr) {
          uint32_t hw_k = tk;
          uint32_t hw_n = tn;
          if (builtin_active) {
            ob::compute_builtin_hw_dims(hw_m, tk, tn, &hw_k, &hw_n);
          }

          // Pack quantized A tile [hw_m, hw_k] only when hw_k changes inside (m0, k0)
          if (cached_aq_hw_k != hw_k) {
            a_q_tile.assign(static_cast<size_t>(hw_m) * hw_k, 0);
            for (uint32_t mi = 0; mi < tm; ++mi) {
              if (row_s_a[mi] == 0.0f) continue;
              const float inv_sa = 1.0f / row_s_a[mi];
              const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0;
              int8_t* aq_row = a_q_tile.data() + static_cast<size_t>(mi) * hw_k;
              for (uint32_t ki = 0; ki < tk; ++ki) {
                aq_row[ki] = ob::quant_sym_i8(a_row[ki] * inv_sa);
              }
            }
            cached_aq_hw_k = hw_k;
          }

          ob::CachedWeightTile* w_tile = nullptr;
          uint32_t scale_d = 1u;
          if (builtin_active) {
            // Fast content hash of raw Q8_0 blocks [n0..n0+tn) x [blk0..blk0+blk_cnt).
            // On cache hit, skips ALL Q8_0 dequant + requant + compute_tile_scale_D + pack_ext2!
            uint64_t wh = 0x51385F5754494C45ULL; // "Q8_WTILE"
            uint32_t whdr[4] = {tn, tk, hw_k, hw_n};
            wh = ob::fast_hash64_update(wh, whdr, sizeof(whdr));
            const size_t row_blk_bytes = static_cast<size_t>(blk_cnt) * sizeof(ob_block_q8_0_t);
            for (uint32_t ni = 0; ni < tn; ++ni) {
              wh = ob::fast_hash64_update(
                  wh, b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0, row_blk_bytes);
            }
            if (wh == 0u) wh = 1u;

            w_tile = ob::weight_cache_find_locked(ctx, wh);
            if (!w_tile) {
              b_q_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
              std::vector<float> sb_vec(tn, 0.0f);
              tmp_w_row.resize(tk);

              for (uint32_t ni = 0; ni < tn; ++ni) {
                const ob_block_q8_0_t* b_row =
                    b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
                int8_t* bq_row = b_q_tile.data() + static_cast<size_t>(ni) * hw_k;

                if (blk_cnt == 1u) {
                  sb_vec[ni] = ob::fp16_to_fp32(b_row[0].d);
                  std::memcpy(bq_row, b_row[0].qs, OB_QK8_0);
                } else {
                  float amax = 0.0f;
                  for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
                    const float d = ob::fp16_to_fp32(b_row[bi].d);
                    const int8_t* qs = b_row[bi].qs;
                    float* dst = tmp_w_row.data() + static_cast<size_t>(bi) * OB_QK8_0;
                    for (uint32_t j = 0; j < OB_QK8_0; ++j) {
                      float v = d * static_cast<float>(qs[j]);
                      dst[j] = v;
                      float av = std::fabs(v);
                      if (av > amax) amax = av;
                    }
                  }
                  if (amax == 0.0f) {
                    sb_vec[ni] = 0.0f;
                  } else {
                    sb_vec[ni] = amax / 127.0f;
                    const float inv_sb = 127.0f / amax;
                    for (uint32_t ki = 0; ki < tk; ++ki) {
                      bq_row[ki] = ob::quant_sym_i8(tmp_w_row[ki] * inv_sb);
                    }
                  }
                }
              }

              scale_d = ob::compute_tile_scale_D(b_q_tile.data(), tn, tk, hw_k);
              std::vector<uint8_t> ext2 = ob::pack_ext2_for_tile(hw_n, hw_k, b_q_tile.data(), nullptr);
              ob_buffer* wbuf = nullptr;
              ob_status_t wst = ob_buffer_alloc(
                  ctx, ext2.size(), OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL | OB_BUF_ZERO_INIT, &wbuf);
              if (wst == OB_OK) {
                wst = ob_buffer_write(wbuf, ext2.data(), ext2.size(), 0u);
              }
              if (wst != OB_OK) {
                if (wbuf) ob_buffer_free(wbuf);
                drain_all_cleanup();
                return wst;
              }
              w_tile = new (std::nothrow) ob::CachedWeightTile();
              if (!w_tile) {
                ob_buffer_free(wbuf);
                drain_all_cleanup();
                return OB_ERR_NO_MEMORY;
              }
              w_tile->key = wh;
              w_tile->scale_d = scale_d;
              w_tile->hw_k = hw_k;
              w_tile->hw_n = hw_n;
              w_tile->row_s_b = std::move(sb_vec);
              w_tile->ext2_buf = wbuf;
              w_tile->ext2_bytes = ext2.size();
              ob::weight_cache_insert_locked(ctx, w_tile);
            } else {
              scale_d = w_tile->scale_d;
            }
          } else {
            // Non-builtin custom user emitter: preserve original per-weight-tile contract.
            b_q_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
            row_s_b.resize(tn);
            tmp_w_row.resize(tk);

            for (uint32_t ni = 0; ni < tn; ++ni) {
              const ob_block_q8_0_t* b_row =
                  b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
              int8_t* bq_row = b_q_tile.data() + static_cast<size_t>(ni) * hw_k;

              if (blk_cnt == 1u) {
                row_s_b[ni] = ob::fp16_to_fp32(b_row[0].d);
                std::memcpy(bq_row, b_row[0].qs, OB_QK8_0);
              } else {
                float amax = 0.0f;
                for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
                  const float d = ob::fp16_to_fp32(b_row[bi].d);
                  const int8_t* qs = b_row[bi].qs;
                  float* dst = tmp_w_row.data() + static_cast<size_t>(bi) * OB_QK8_0;
                  for (uint32_t j = 0; j < OB_QK8_0; ++j) {
                    float v = d * static_cast<float>(qs[j]);
                    dst[j] = v;
                    float av = std::fabs(v);
                    if (av > amax) amax = av;
                  }
                }
                if (amax == 0.0f) {
                  row_s_b[ni] = 0.0f;
                } else {
                  row_s_b[ni] = amax / 127.0f;
                  const float inv_sb = 127.0f / amax;
                  for (uint32_t ki = 0; ki < tk; ++ki) {
                    bq_row[ki] = ob::quant_sym_i8(tmp_w_row[ki] * inv_sb);
                  }
                }
              }
            }
            scale_d = ob::compute_tile_scale_D(b_q_tile.data(), tn, tk, hw_k);
          }

          if (ring_count >= ring_depth) {
            ob_status_t dst = drain_one();
            if (dst != OB_OK) {
              drain_all_cleanup();
              return dst;
            }
          }
          const uint32_t slot = (ring_head + ring_count) % ring_depth;

          ob_gemm_desc_t desc {};
          desc.m_dim = hw_m;
          desc.k_dim = hw_k;
          desc.n_dim = hw_n;
          desc.s_in = 1.0f;
          desc.zp_in = 0;
          desc.s_w = 1.0f;
          desc.zp_w = 0;
          desc.s_out = static_cast<float>(scale_d);
          desc.zp_out = 0;
          desc.activation = OB_ACT_NONE;
          desc.bias_n = nullptr;
          if (builtin_active) {
            // Weight-agnostic graph keyed ONLY by (hw_m, hw_k, hw_n, scale_d, slot_bucket)!
            desc.weights_nk = nullptr;
            desc.cache_key = ob::hash_shape_key(
                hw_m, hw_k, hw_n, 0, 0, scale_d, slot % param_swap_slots);
          } else {
            desc.weights_nk = b_q_tile.data();
            const size_t packed_sz = ob_gemm_packed_weight_bytes(hw_k, hw_n);
            packed_tile.resize(packed_sz);
            ob_status_t pst = ob_gemm_pack_weights_int8(
                b_q_tile.data(), nullptr, hw_k, hw_n, packed_tile.data(), packed_tile.size());
            if (pst != OB_OK) {
              drain_all_cleanup();
              return pst;
            }
            desc.packed_params = packed_tile.data();
            desc.packed_params_bytes = packed_tile.size();
            desc.cache_key = ob::hash_tile_key(
                hw_m, hw_k, hw_n, 0, 0, scale_d, b_q_tile.data(), b_q_tile.size());
          }

          ob_graph_t* graph = nullptr;
          bool was_hit = false;
          ob_status_t gst = ob::acquire_tile_graph(
              ctx, emitter, emitter_ud, desc, &graph, &was_hit);
          if (gst == OB_OK) {
            used_emitter = true;
            if (was_hit) cache_hits += 1u;

            ob_dgc0_info_t ginfo {};
            ob_graph_get_info(graph, &ginfo);
            size_t in_alloc = std::max<size_t>(
                a_q_tile.size(),
                ginfo.num_inputs > 0u ? static_cast<size_t>(ginfo.input_sizes[0]) : 1u);
            size_t out_alloc = std::max<size_t>(
                static_cast<size_t>(hw_m) * ob::align_up_u32(hw_n, 4u),
                ginfo.num_outputs > 0u ? static_cast<size_t>(ginfo.output_sizes[0]) : 1u);

            ob_status_t bst = ob::ensure_pipeline_buf(
                ctx, &ctx->gemm_in_bufs[slot], in_alloc, OB_BUF_TO_DEVICE | OB_BUF_ZERO_INIT);
            if (bst == OB_OK) {
              bst = ob::ensure_pipeline_buf(
                  ctx, &ctx->gemm_out_bufs[slot], out_alloc, OB_BUF_FROM_DEVICE | OB_BUF_ZERO_INIT);
            }
            if (bst == OB_OK) {
              bst = ob_buffer_write(ctx->gemm_in_bufs[slot], a_q_tile.data(), a_q_tile.size(), 0u);
            }

            ob_exec_handle_t exec_h = 0u;
            if (bst == OB_OK) {
              if (w_tile != nullptr) {
                bst = ob::graph_submit_async_with_params(
                    graph, &ctx->gemm_in_bufs[slot], 1u, &ctx->gemm_out_bufs[slot], 1u,
                    w_tile->ext2_buf, &exec_h);
              } else {
                bst = ob_graph_submit_async(
                    graph, &ctx->gemm_in_bufs[slot], 1u, &ctx->gemm_out_bufs[slot], 1u, &exec_h);
              }
            }
            if (bst != OB_OK) {
              ob_graph_release(graph);
              drain_all_cleanup();
              return bst;
            }

            PendingF32Tile& p = ring[slot];
            p.handle = exec_h;
            p.graph = graph;
            p.w_tile = w_tile;
            if (w_tile) w_tile->in_use += 1u;
            p.slot = slot;
            p.m0 = m0;
            p.tm = tm;
            p.hw_m = hw_m;
            p.n0 = n0;
            p.tn = tn;
            p.hw_n = hw_n;
            p.scale_d = static_cast<float>(scale_d);
            p.s_a = row_s_a;
            if (!w_tile) p.s_b = row_s_b;
            ring_count += 1u;

            if (!ctx->mock_mode) {
              tile_handled_by_hw = true;
            }
          } else if (gst != OB_ERR_UNSUPPORTED) {
            drain_all_cleanup();
            return gst;
          }
        }

        if (!tile_handled_by_hw) {
          used_cpu_fallback = true;
          ob::cpu_matmul_f32_q8_0_tile(a_mk, b_blocks, c_mn, k_dim, n_dim,
                                       m0, tm, k0, tk, n0, tn);
        }
      }
    }
  }

  while (ring_count > 0u) {
    ob_status_t dst = drain_one();
    if (dst != OB_OK) {
      drain_all_cleanup();
      return dst;
    }
  }

  if (out_stats) {
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->hw_execution_us = total_hw_us;
    out_stats->wall_time_us = (ob::monotonic_now_ns() - t0_ns) / 1000ULL;
    out_stats->tiles_dispatched = tiles_dispatched;
    out_stats->cache_hits = cache_hits;
    if (ctx->mock_mode) out_stats->flags |= OB_STAT_FLAG_MOCK;
    if (used_emitter) out_stats->flags |= OB_STAT_FLAG_EMITTER_USED;
    if (used_cpu_fallback) out_stats->flags |= OB_STAT_FLAG_CPU_FALLBACK;
    if (cache_hits > 0u) out_stats->flags |= OB_STAT_FLAG_CACHE_HIT;
  }
  return OB_OK;
}

}  // extern "C"
