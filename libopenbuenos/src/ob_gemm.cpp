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

bool strict_weight_hash_enabled() {
  const char* env = std::getenv("OPENBUENOS_STRICT_WEIGHT_HASH");
  return (env && env[0] == '1');
}

// Fast content-only weight-tile hash for ext[2] DMA-BUF LRU cache lookup.
// - Small tiles (<= 16 KiB) or OPENBUENOS_STRICT_WEIGHT_HASH=1: hashes every byte.
// - Large tiles (e.g. 256 KiB..4 MiB lt_1x* weights): O(1) ~1.1 KiB strided
//   content fingerprint (64 rows x 16 B + head/tail 64 B). Avoids reading 150 MB..1.7 GB
//   of cold DRAM per token on CPU while remaining pointer-agnostic so reused Q4_0->Q8_0
//   scratch buffers hit cached ext[2] DMA-BUFs across tokens with zero false hits.
uint64_t hash_weight_tile_fast(uint64_t h,
                               const uint8_t* base,
                               size_t row_stride_bytes,
                               uint32_t tn,
                               size_t row_bytes) {
  if (tn == 0u || row_bytes == 0u) return h == 0u ? 1u : h;
  const size_t total_bytes = static_cast<size_t>(tn) * row_bytes;
  if (total_bytes <= 16384u || strict_weight_hash_enabled()) {
    for (uint32_t ni = 0; ni < tn; ++ni) {
      h = fast_hash64_update(h, base + static_cast<size_t>(ni) * row_stride_bytes, row_bytes);
    }
    return h == 0u ? 1u : h;
  }

  const size_t edge_len = std::min<size_t>(row_bytes, 64u);
  h = fast_hash64_update(h, base, edge_len);
  const uint8_t* last_row = base + static_cast<size_t>(tn - 1u) * row_stride_bytes;
  h = fast_hash64_update(h, last_row + (row_bytes - edge_len), edge_len);

  constexpr uint32_t kNumProbes = 64u;
  const size_t probe_len = std::min<size_t>(row_bytes, 16u);
  const size_t max_off = row_bytes - probe_len;
  for (uint32_t s = 0; s < kNumProbes; ++s) {
    const uint32_t ni = (tn > 1u)
        ? static_cast<uint32_t>((static_cast<uint64_t>(s) * (tn - 1u)) / (kNumProbes - 1u))
        : 0u;
    const size_t off = (max_off > 0u)
        ? static_cast<size_t>((static_cast<uint64_t>(s) * 2654435761ULL) % (max_off + 1u))
        : 0u;
    h = fast_hash64_update(h, base + static_cast<size_t>(ni) * row_stride_bytes + off, probe_len);
  }
  return h == 0u ? 1u : h;
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
  if (ctx->weight_cache_max_bytes == 0u) {
    const char* env = std::getenv("OPENBUENOS_WEIGHT_CACHE_MB");
    if (env && env[0]) {
      unsigned long mb = std::strtoul(env, nullptr, 10);
      if (mb < 16ul) mb = 16ul;
      if (mb > 8192ul) mb = 8192ul;
      ctx->weight_cache_max_bytes = static_cast<size_t>(mb) * 1024u * 1024u;
    } else {
      // 256 MiB on host mock; 2048 MiB on live 16 GiB Android device (pins Qwen2.5-1.5B Q8_0!).
      ctx->weight_cache_max_bytes = ctx->mock_mode
          ? (256u * 1024u * 1024u)
          : (2048u * 1024u * 1024u);
    }
  }
  if (ctx->weight_cache_max_entries == 0u) {
    const char* env_e = std::getenv("OPENBUENOS_WEIGHT_CACHE_ENTRIES");
    if (env_e && env_e[0]) {
      unsigned long lim = std::strtoul(env_e, nullptr, 10);
      if (lim < 64ul) lim = 64ul;
      if (lim > 16384ul) lim = 16384ul;
      ctx->weight_cache_max_entries = static_cast<size_t>(lim);
    } else {
      ctx->weight_cache_max_entries = kDefaultWeightCacheMaxEntries;
    }
  }
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
  while ((ctx->weight_cache_bytes + incoming_bytes > ctx->weight_cache_max_bytes ||
          ctx->weight_cache_map.size() >= ctx->weight_cache_max_entries) &&
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

uint32_t quant_scale_D_grid_impl(double raw_d) {
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
  if (d < 1ULL) d = 1ULL;
  if (d > (1ULL << 24)) d = (1ULL << 24);
  return static_cast<uint32_t>(d);
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
  // Engage DC-sum guard only for tiny K or statistically non-zero-mean weight rows
  // (|sum(w)| > 4.2 * ||w||_2), avoiding false D inflation from max_n |sum| random-walk
  // tails across large lt_1x* tiles (N = 576..2048).
  double dc_guard = (tk <= 32u || static_cast<double>(max_abs_sum) > 4.2 * max_l2)
                        ? (static_cast<double>(max_abs_sum) * 1.05)
                        : 0.0;
  double stat_est = std::max(2.55 * max_l2, dc_guard) * zp_scale;
  double raw_d = std::min(hard_cap, stat_est);
  return quant_scale_D_grid_impl(raw_d);
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

bool large_tiles_enabled() {
  const char* env = std::getenv("OPENBUENOS_DISABLE_LARGE_TILES");
  return !(env && env[0] == '1');
}

bool batched_large_tiles_enabled() {
  if (!large_tiles_enabled()) return false;
  const char* env = std::getenv("OPENBUENOS_DISABLE_BATCHED_LARGE_TILES");
  return !(env && env[0] == '1');
}

std::vector<MSlice> build_m_slices(uint32_t m_dim, uint32_t max_m,
                                   bool builtin_active, bool allow_m_pad,
                                   bool allow_lt_64x = false) {
  std::vector<MSlice> out;
  if (!builtin_active) {
    for (uint32_t m0 = 0; m0 < m_dim; m0 += max_m) {
      uint32_t tm = std::min(max_m, m_dim - m0);
      out.push_back({m0, tm, tm});
    }
    return out;
  }
  if (allow_m_pad && allow_lt_64x && batched_large_tiles_enabled() &&
      m_dim >= 2u && max_m >= 64u) {
    // M=64 batched prefill large-tile mode (up to 983.6 INT8 GOPS on rango):
    // slice M in blocks of 64; any multi-row tail [2..63] zero-pads to hw_m=64
    // (1 tile instead of dozens), while a singleton tail tm==1 uses lt_1x (hw_m=1).
    for (uint32_t m0 = 0u; m0 < m_dim; ) {
      uint32_t tm = std::min<uint32_t>(64u, m_dim - m0);
      uint32_t hw_m = (tm == 1u) ? 1u : 64u;
      out.push_back({m0, tm, hw_m});
      m0 += tm;
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

struct DimSlice {
  uint32_t off;
  uint32_t len;
};

struct TileRect {
  uint32_t k0;
  uint32_t tk;
  uint32_t n0;
  uint32_t tn;
  uint32_t hw_k;
  uint32_t hw_n;
};

uint32_t ceil_lt_1x_dim(uint32_t d, bool prefer_576_for_le_512) {
  if (d <= 512u) return prefer_576_for_le_512 ? 576u : 512u;
  if (d <= 576u) return 576u;
  if (d <= 1024u) return 1024u;
  if (d <= 1536u) return 1536u;
  if (d <= 2048u) return 2048u;
  return 0u;
}

std::vector<DimSlice> build_lt_dim_slices(uint32_t dim, uint32_t cap) {
  std::vector<DimSlice> out;
  if (dim == 0u) return out;
  static constexpr uint32_t kSteps[4] = {2048u, 1536u, 1024u, 512u};
  uint32_t off = 0u;
  while (off < dim) {
    uint32_t rem = dim - off;
    uint32_t padded_rem = ceil_lt_1x_dim(rem, false);
    if (rem <= cap && (rem < 512u || (padded_rem != 0u && padded_rem <= cap))) {
      out.push_back({off, rem});
      break;
    }
    uint32_t take = 0u;
    for (uint32_t s : kSteps) {
      if (s <= cap && s <= rem) {
        take = s;
        break;
      }
    }
    if (take == 0u) {
      out.push_back({off, rem});
      break;
    }
    out.push_back({off, take});
    off += take;
  }
  return out;
}

bool try_lt_1x_hw_dims(uint32_t tk, uint32_t tn, uint32_t max_k, uint32_t max_n,
                       uint32_t* out_hw_k, uint32_t* out_hw_n) {
  if (tk < 192u || tn < 192u || (tk < 512u && tn < 512u)) return false;
  if (max_k < 512u || max_n < 512u) return false;
  bool pref_n_576 = (tk == 576u && tn <= 512u && max_n >= 576u);
  bool pref_k_576 = (tn == 576u && tk <= 512u && max_k >= 576u);
  uint32_t hk = ceil_lt_1x_dim(tk, pref_k_576);
  uint32_t hn = ceil_lt_1x_dim(tn, pref_n_576);
  if (hk == 0u || hn == 0u || hk > max_k || hn > max_n) return false;
  *out_hw_k = hk;
  *out_hw_n = hn;
  return true;
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

void append_small_tile_rects(uint32_t hw_m,
                             uint32_t k_start, uint32_t k_len,
                             uint32_t n_start, uint32_t n_len,
                             uint32_t max_k, uint32_t max_n,
                             uint32_t k_align,
                             bool builtin_active,
                             std::vector<TileRect>* out) {
  const uint32_t k_cap = (hw_m == 1u) ? 512u : ((hw_m >= 8u) ? 64u : 32u);
  uint32_t eff_max_k = builtin_active ? std::min<uint32_t>(max_k, k_cap) : max_k;
  if (k_align > 1u) {
    eff_max_k = std::max<uint32_t>(k_align, (eff_max_k / k_align) * k_align);
  }
  for (uint32_t dk = 0u; dk < k_len; dk += eff_max_k) {
    const uint32_t k0 = k_start + dk;
    const uint32_t tk = std::min<uint32_t>(eff_max_k, k_len - dk);
    const uint32_t n_cap = (hw_m == 1u && tk == 512u) ? 128u : 64u;
    const uint32_t eff_max_n = builtin_active ? std::min<uint32_t>(max_n, n_cap) : max_n;
    for (uint32_t dn = 0u; dn < n_len; dn += eff_max_n) {
      const uint32_t n0 = n_start + dn;
      const uint32_t tn = std::min<uint32_t>(eff_max_n, n_len - dn);
      uint32_t hk = tk;
      uint32_t hn = tn;
      if (builtin_active) {
        compute_builtin_hw_dims(hw_m, tk, tn, &hk, &hn);
      }
      out->push_back({k0, tk, n0, tn, hk, hn});
    }
  }
}

uint32_t ceil_lt_64x_k_dim(uint32_t tk) {
  if (tk <= 512u) return 512u;
  if (tk <= 576u) return 576u;
  if (tk <= 1024u) return 1024u;
  if (tk <= 1536u) return 1536u;
  return 0u;
}

std::vector<DimSlice> build_lt_64x_k_slices(uint32_t k_dim, uint32_t k_cap) {
  std::vector<DimSlice> out;
  if (k_dim == 0u) return out;
  static constexpr uint32_t kSteps[4] = {1536u, 1024u, 576u, 512u};
  uint32_t off = 0u;
  while (off < k_dim) {
    uint32_t rem = k_dim - off;
    uint32_t padded = ceil_lt_64x_k_dim(rem);
    if (padded != 0u && padded <= k_cap) {
      out.push_back({off, rem});
      break;
    }
    uint32_t take = 0u;
    for (uint32_t s : kSteps) {
      if (s <= k_cap && s <= rem) {
        take = s;
        break;
      }
    }
    if (take == 0u) {
      out.push_back({off, rem});
      break;
    }
    out.push_back({off, take});
    off += take;
  }
  return out;
}

std::vector<TileRect> build_tile_rects(uint32_t hw_m,
                                       uint32_t k_dim, uint32_t n_dim,
                                       uint32_t max_k, uint32_t max_n,
                                       uint32_t k_align,
                                       bool builtin_active,
                                       bool allow_lt) {
  std::vector<TileRect> out;
  if (builtin_active && allow_lt && batched_large_tiles_enabled() &&
      hw_m == 64u && max_k >= 512u && max_n >= 128u &&
      k_dim >= 256u && n_dim >= 128u) {
    const uint32_t k_cap = std::min<uint32_t>(max_k, 1536u);
    const uint32_t n_cap = std::max<uint32_t>(128u, (std::min<uint32_t>(max_n, 1536u) / 32u) * 32u);
    std::vector<DimSlice> ks = build_lt_64x_k_slices(k_dim, k_cap);
    for (const DimSlice& ks_i : ks) {
      const uint32_t hk = ceil_lt_64x_k_dim(ks_i.len);
      if (hk == 0u || hk > k_cap) continue;
      for (uint32_t n0 = 0u; n0 < n_dim; n0 += n_cap) {
        const uint32_t tn = std::min<uint32_t>(n_cap, n_dim - n0);
        const uint32_t hn = std::max<uint32_t>(128u, align_up_u32(tn, 32u));
        out.push_back({ks_i.off, ks_i.len, n0, tn, hk, hn});
      }
    }
    if (!out.empty()) return out;
  }
  if (builtin_active && allow_lt && large_tiles_enabled() &&
      hw_m == 1u && max_k >= 512u && max_n >= 512u &&
      k_dim >= 192u && n_dim >= 192u && (k_dim >= 512u || n_dim >= 512u)) {
    const uint32_t k_cap = std::min<uint32_t>(max_k, 2048u);
    const uint32_t n_cap = std::min<uint32_t>(max_n, 2048u);
    std::vector<DimSlice> ks = build_lt_dim_slices(k_dim, k_cap);
    std::vector<DimSlice> ns = build_lt_dim_slices(n_dim, n_cap);
    for (const DimSlice& ks_i : ks) {
      for (const DimSlice& ns_j : ns) {
        uint32_t hk = 0u, hn = 0u;
        if (try_lt_1x_hw_dims(ks_i.len, ns_j.len, k_cap, n_cap, &hk, &hn)) {
          out.push_back({ks_i.off, ks_i.len, ns_j.off, ns_j.len, hk, hn});
        } else {
          append_small_tile_rects(hw_m, ks_i.off, ks_i.len, ns_j.off, ns_j.len,
                                  max_k, max_n, k_align, builtin_active, &out);
        }
      }
    }
    return out;
  }
  append_small_tile_rects(hw_m, 0u, k_dim, 0u, n_dim, max_k, max_n,
                          k_align, builtin_active, &out);
  return out;
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

void requant_q8_0_tile(const ob_block_q8_0_t* b_blocks,
                       uint32_t nb_k,
                       uint32_t n0, uint32_t tn,
                       uint32_t blk0, uint32_t blk_cnt,
                       uint32_t tk, uint32_t hw_k, uint32_t hw_n,
                       std::vector<int8_t>* out_bq_tile,
                       std::vector<float>* out_sb_vec,
                       std::vector<float>* scratch_w_row) {
  out_bq_tile->assign(static_cast<size_t>(hw_n) * hw_k, 0);
  out_sb_vec->assign(tn, 0.0f);

  if (blk_cnt == 1u) {
    for (uint32_t ni = 0; ni < tn; ++ni) {
      const ob_block_q8_0_t* b_row = b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
      (*out_sb_vec)[ni] = fp16_to_fp32(b_row[0].d);
      std::memcpy(out_bq_tile->data() + static_cast<size_t>(ni) * hw_k, b_row[0].qs, OB_QK8_0);
    }
    return;
  }

  scratch_w_row->resize(tk);
  for (uint32_t ni = 0; ni < tn; ++ni) {
    const ob_block_q8_0_t* b_row = b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
    float amax = 0.0f;
    for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
      const float d = fp16_to_fp32(b_row[bi].d);
      const int8_t* qs = b_row[bi].qs;
      float* dst = scratch_w_row->data() + static_cast<size_t>(bi) * OB_QK8_0;
      for (uint32_t j = 0; j < OB_QK8_0; ++j) {
        float v = d * static_cast<float>(qs[j]);
        dst[j] = v;
        float av = std::fabs(v);
        if (av > amax) amax = av;
      }
    }
    if (amax == 0.0f) {
      (*out_sb_vec)[ni] = 0.0f;
      continue;
    }
    const float sb = amax / 127.0f;
    (*out_sb_vec)[ni] = sb;
    const float inv_sb = 127.0f / amax;
    int8_t* bq_row = out_bq_tile->data() + static_cast<size_t>(ni) * hw_k;
    for (uint32_t ki = 0; ki < tk; ++ki) {
      bq_row[ki] = quant_sym_i8((*scratch_w_row)[ki] * inv_sb);
    }
  }
}

void requant_q4_0_tile(const ob_block_q4_0_t* b_blocks,
                       uint32_t nb_k,
                       uint32_t n0, uint32_t tn,
                       uint32_t blk0, uint32_t blk_cnt,
                       uint32_t tk, uint32_t hw_k, uint32_t hw_n,
                       std::vector<int8_t>* out_bq_tile,
                       std::vector<float>* out_sb_vec,
                       std::vector<float>* scratch_w_row) {
  out_bq_tile->assign(static_cast<size_t>(hw_n) * hw_k, 0);
  out_sb_vec->assign(tn, 0.0f);

  if (blk_cnt == 1u) {
    for (uint32_t ni = 0; ni < tn; ++ni) {
      const ob_block_q4_0_t* b_row = b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
      (*out_sb_vec)[ni] = fp16_to_fp32(b_row[0].d);
      int8_t* bq_row = out_bq_tile->data() + static_cast<size_t>(ni) * hw_k;
      const uint8_t* qs = b_row[0].qs;
      for (uint32_t j = 0; j < OB_QK8_0 / 2u; ++j) {
        const uint8_t byte = qs[j];
        bq_row[j] = static_cast<int8_t>(static_cast<int32_t>(byte & 0x0Fu) - 8);
        bq_row[j + OB_QK8_0 / 2u] = static_cast<int8_t>(static_cast<int32_t>(byte >> 4) - 8);
      }
    }
    return;
  }

  scratch_w_row->resize(tk);
  for (uint32_t ni = 0; ni < tn; ++ni) {
    const ob_block_q4_0_t* b_row = b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
    float amax = 0.0f;
    for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
      const float d = fp16_to_fp32(b_row[bi].d);
      const uint8_t* qs = b_row[bi].qs;
      float* dst = scratch_w_row->data() + static_cast<size_t>(bi) * OB_QK8_0;
      for (uint32_t j = 0; j < OB_QK8_0 / 2u; ++j) {
        const uint8_t byte = qs[j];
        float v0 = d * static_cast<float>(static_cast<int32_t>(byte & 0x0Fu) - 8);
        float v1 = d * static_cast<float>(static_cast<int32_t>(byte >> 4) - 8);
        dst[j] = v0;
        dst[j + OB_QK8_0 / 2u] = v1;
        float av0 = std::fabs(v0);
        float av1 = std::fabs(v1);
        if (av0 > amax) amax = av0;
        if (av1 > amax) amax = av1;
      }
    }
    if (amax == 0.0f) {
      (*out_sb_vec)[ni] = 0.0f;
      continue;
    }
    const float sb = amax / 127.0f;
    (*out_sb_vec)[ni] = sb;
    const float inv_sb = 127.0f / amax;
    int8_t* bq_row = out_bq_tile->data() + static_cast<size_t>(ni) * hw_k;
    for (uint32_t ki = 0; ki < tk; ++ki) {
      bq_row[ki] = quant_sym_i8((*scratch_w_row)[ki] * inv_sb);
    }
  }
}

void cpu_matmul_f32_q4_0_tile(const float* a_mk,
                              const ob_block_q4_0_t* b_blocks,
                              float* c_mn,
                              uint32_t k_dim,
                              uint32_t n_dim,
                              uint32_t m0, uint32_t tm,
                              uint32_t k0, uint32_t tk,
                              uint32_t n0, uint32_t tn) {
  const uint32_t nb_k = k_dim / OB_QK8_0;
  const uint32_t blk0 = k0 / OB_QK8_0;
  const uint32_t blk_cnt = tk / OB_QK8_0;
  constexpr uint32_t kHalf = OB_QK8_0 / 2u;

  for (uint32_t mi = 0; mi < tm; ++mi) {
    const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim;
    float* c_row = c_mn + static_cast<size_t>(m0 + mi) * n_dim + n0;
    for (uint32_t ni = 0; ni < tn; ++ni) {
      const ob_block_q4_0_t* b_row =
          b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
      float acc = 0.0f;
      for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
        const float d = fp16_to_fp32(b_row[bi].d);
        const uint8_t* qs = b_row[bi].qs;
        const float* a_blk = a_row + static_cast<size_t>(blk0 + bi) * OB_QK8_0;
        float bsum = 0.0f;
        for (uint32_t j = 0; j < kHalf; ++j) {
          const uint8_t byte = qs[j];
          bsum += a_blk[j] * static_cast<float>(static_cast<int32_t>(byte & 0x0Fu) - 8);
          bsum += a_blk[j + kHalf] * static_cast<float>(static_cast<int32_t>(byte >> 4) - 8);
        }
        acc += d * bsum;
      }
      c_row[ni] += acc;
    }
  }
}

void apply_activation_inplace(float* c_mn, size_t count, uint32_t act) {
  if (!c_mn || act == OB_ACT_NONE) return;
  if (act == OB_ACT_RELU) {
    for (size_t i = 0; i < count; ++i) {
      if (c_mn[i] < 0.0f) c_mn[i] = 0.0f;
    }
  } else if (act == OB_ACT_RELU6) {
    for (size_t i = 0; i < count; ++i) {
      float v = c_mn[i];
      c_mn[i] = (v < 0.0f) ? 0.0f : ((v > 6.0f) ? 6.0f : v);
    }
  } else if (act == OB_ACT_SILU) {
    for (size_t i = 0; i < count; ++i) {
      float x = c_mn[i];
      c_mn[i] = x / (1.0f + std::exp(-x));
    }
  } else if (act == OB_ACT_GELU) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    for (size_t i = 0; i < count; ++i) {
      float x = c_mn[i];
      float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
      c_mn[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
  }
}

void quantize_activation_row_split_impl(const float* a_row,
                                        uint32_t tk,
                                        int8_t* aq_row,
                                        float* out_s_a,
                                        std::vector<std::pair<uint32_t, float>>* out_outliers) {
  if (out_outliers) out_outliers->clear();
  float amax = 0.0f;
  double sq_sum = 0.0;
  for (uint32_t ki = 0; ki < tk; ++ki) {
    float v = a_row[ki];
    float av = std::fabs(v);
    if (av > amax) amax = av;
    sq_sum += static_cast<double>(v) * static_cast<double>(v);
  }
  if (amax == 0.0f) {
    std::memset(aq_row, 0, tk);
    *out_s_a = 0.0f;
    return;
  }

  // LLM.int8()-style sparse outlier peel: if a vector contains extreme super-activation
  // channels (|x[k]| > 8 * rms, e.g. Qwen2.5 30..40 sigma attention sinks), peel <= 8
  // channels (<0.5% of K) into an exact FP32 residual so the remaining 99.5% of channels
  // retain full 8-bit dynamic range on the TPU instead of collapsing to 1..2 bits!
  constexpr uint32_t kMaxOutliers = 8u;
  constexpr double kOutlierSigma = 8.0;
  uint32_t peeled_idx[kMaxOutliers]{};
  uint32_t n_peeled = 0u;

  if (out_outliers && tk >= 128u) {
    double rms = std::sqrt(sq_sum / static_cast<double>(tk));
    if (static_cast<double>(amax) > kOutlierSigma * std::max(rms, 1e-12)) {
      double rem_sq = sq_sum;
      for (uint32_t iter = 0; iter < kMaxOutliers; ++iter) {
        uint32_t best_k = 0u;
        float best_av = -1.0f;
        for (uint32_t ki = 0; ki < tk; ++ki) {
          bool skip = false;
          for (uint32_t p = 0; p < n_peeled; ++p) {
            if (peeled_idx[p] == ki) { skip = true; break; }
          }
          if (skip) continue;
          float av = std::fabs(a_row[ki]);
          if (av > best_av) {
            best_av = av;
            best_k = ki;
          }
        }
        if (best_av <= 0.0f) break;
        double denom = static_cast<double>(tk > n_peeled + 1u ? (tk - n_peeled - 1u) : 1u);
        double next_sq = std::max(0.0, rem_sq - static_cast<double>(best_av) * static_cast<double>(best_av));
        double rem_rms = std::sqrt(next_sq / denom);
        if (static_cast<double>(best_av) <= kOutlierSigma * std::max(rem_rms, 1e-12)) {
          break;
        }
        peeled_idx[n_peeled++] = best_k;
        out_outliers->push_back({best_k, a_row[best_k]});
        rem_sq = next_sq;
      }
      if (n_peeled > 0u) {
        amax = 0.0f;
        for (uint32_t ki = 0; ki < tk; ++ki) {
          bool skip = false;
          for (uint32_t p = 0; p < n_peeled; ++p) {
            if (peeled_idx[p] == ki) { skip = true; break; }
          }
          if (skip) continue;
          float av = std::fabs(a_row[ki]);
          if (av > amax) amax = av;
        }
      }
    }
  }

  if (amax == 0.0f) {
    std::memset(aq_row, 0, tk);
    *out_s_a = 0.0f;
    return;
  }
  *out_s_a = amax / 127.0f;
  const float inv_sa = 127.0f / amax;
  for (uint32_t ki = 0; ki < tk; ++ki) {
    aq_row[ki] = quant_sym_i8(a_row[ki] * inv_sa);
  }
  for (uint32_t p = 0; p < n_peeled; ++p) {
    aq_row[peeled_idx[p]] = 0;
  }
}

uint32_t compute_activation_rms_shift_impl(const int8_t* aq_tile,
                                           uint32_t tm,
                                           uint32_t tk,
                                           uint32_t hw_k) {
  if (!aq_tile || tm == 0u || tk == 0u) return 0u;
  double max_rms = 0.0;
  for (uint32_t mi = 0; mi < tm; ++mi) {
    const int8_t* row = aq_tile + static_cast<size_t>(mi) * hw_k;
    uint64_t sq = 0ULL;
    for (uint32_t ki = 0; ki < tk; ++ki) {
      int32_t v = static_cast<int32_t>(row[ki]);
      sq += static_cast<uint64_t>(v * v);
    }
    double rms = std::sqrt(static_cast<double>(sq) / static_cast<double>(tk));
    if (rms > max_rms) max_rms = rms;
  }
  if (max_rms <= 0.0) return 0u;
  // Base D assumes Uniform[-127,127] activation RMS = 73.3. For lower-RMS Gaussian
  // or LLM activations, step D down by coarse powers of two (shift in [0..4]) with
  // 1.35x safety margin (paired with on-TPU saturation retry if any output hits +-127).
  double ratio = (max_rms * 1.35) / 73.3;
  uint32_t shift = 0u;
  while (ratio <= 0.5 && shift < 4u) {
    ratio *= 2.0;
    shift += 1u;
  }
  return shift;
}

}  // namespace

uint32_t quant_scale_D_grid(double raw_d) {
  return quant_scale_D_grid_impl(raw_d);
}

uint32_t compute_tile_scale_D(const int8_t* b_tile, uint32_t tn, uint32_t tk,
                              uint32_t stride_k, int32_t zp_a) {
  return compute_tile_scale_D_impl(b_tile, tn, tk, stride_k, zp_a);
}

void quantize_activation_row_split(const float* a_row,
                                   uint32_t tk,
                                   int8_t* aq_row,
                                   float* out_s_a,
                                   std::vector<std::pair<uint32_t, float>>* out_outliers) {
  quantize_activation_row_split_impl(a_row, tk, aq_row, out_s_a, out_outliers);
}

uint32_t compute_activation_rms_shift(const int8_t* aq_tile,
                                      uint32_t tm,
                                      uint32_t tk,
                                      uint32_t hw_k) {
  return compute_activation_rms_shift_impl(aq_tile, tm, tk, hw_k);
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
  const bool allow_lt_64x = allow_m_pad && k_dim >= 256u && n_dim >= 128u &&
                            max_k >= 512u && max_n >= 128u;
  std::vector<ob::MSlice> m_slices = ob::build_m_slices(
      m_dim, max_m, builtin_active, allow_m_pad, allow_lt_64x);

  for (const ob::MSlice& ms : m_slices) {
    const uint32_t m0 = ms.m0;
    const uint32_t tm = ms.tm;
    const uint32_t hw_m = ms.hw_m;
    std::vector<ob::TileRect> tiles = ob::build_tile_rects(
        hw_m, k_dim, n_dim, max_k, max_n, 1u, builtin_active, allow_m_pad);

    for (const ob::TileRect& tr : tiles) {
      const uint32_t k0 = tr.k0;
      const uint32_t tk = tr.tk;
      const uint32_t n0 = tr.n0;
      const uint32_t tn = tr.tn;
      const uint32_t hw_k = tr.hw_k;
      const uint32_t hw_n = tr.hw_n;
      tiles_dispatched += 1u;

      bool tile_handled_by_hw = false;
      if (emitter != nullptr) {
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
              const bool is_lt64_tile = ob::is_lt_64x_shape(hw_m, hw_k, hw_n);
              const uint32_t pack_fmt = is_lt64_tile ? 1u : 0u;
              // O(1) strided content hash of raw int8 weight tile [n0..n0+tn) x [k0..k0+tk)
              uint64_t wh = 0x49385F5754494C45ULL; // "I8_WTILE"
              uint32_t whdr[6] = {tn, tk, hw_k, hw_n, static_cast<uint32_t>(zp_a), pack_fmt};
              wh = ob::fast_hash64_update(wh, whdr, sizeof(whdr));
              wh = ob::hash_weight_tile_fast(
                  wh,
                  reinterpret_cast<const uint8_t*>(b_nk + static_cast<size_t>(n0) * k_dim + k0),
                  static_cast<size_t>(k_dim), tn, static_cast<size_t>(tk));

              w_tile = ob::weight_cache_find_locked(ctx, wh);
              if (!w_tile) {
                b_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
                for (uint32_t ni = 0; ni < tn; ++ni) {
                  std::memcpy(b_tile.data() + static_cast<size_t>(ni) * hw_k,
                              b_nk + static_cast<size_t>(n0 + ni) * k_dim + k0, tk);
                }
                scale_d = ob::compute_tile_scale_D(b_tile.data(), tn, tk, hw_k, zp_a);
                std::vector<uint8_t> ext2 = is_lt64_tile
                    ? ob::pack_ext2_for_lt_64x(hw_n, hw_k, b_tile.data(), nullptr)
                    : ob::pack_ext2_for_tile(hw_n, hw_k, b_tile.data(), nullptr);
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

ob_status_t ob_gemm_matmul_f32_quant_batch(ob_context_t* ctx,
                                           const float* a_mk,
                                           uint32_t m_dim,
                                           uint32_t k_dim,
                                           const ob_gemm_f32_quant_job_t* jobs,
                                           uint32_t num_jobs,
                                           ob_invoke_stats_t* out_stats) {
  if (!ctx || !a_mk || !jobs || num_jobs == 0u ||
      m_dim == 0u || k_dim == 0u || (k_dim % OB_QK8_0) != 0u) {
    return OB_ERR_INVALID_ARG;
  }
  for (uint32_t j = 0; j < num_jobs; ++j) {
    if (!jobs[j].b_nk || !jobs[j].c_mn || jobs[j].n_dim == 0u) {
      return OB_ERR_INVALID_ARG;
    }
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

  for (uint32_t j = 0; j < num_jobs; ++j) {
    std::memset(jobs[j].c_mn, 0, static_cast<size_t>(m_dim) * jobs[j].n_dim * sizeof(float));
  }

  uint32_t tiles_dispatched = 0u;
  uint32_t cache_hits = 0u;
  uint64_t total_hw_us = 0u;
  bool used_emitter = false;
  bool used_cpu_fallback = false;

  const uint32_t param_swap_slots = ob::get_param_swap_slots();

  struct PendingF32Tile {
    ob_exec_handle_t handle = 0u;
    ob_graph_t* graph = nullptr;
    ob::CachedWeightTile* w_tile = nullptr;
    uint32_t job_idx = 0u;
    uint32_t slot = 0u;
    uint32_t m0 = 0u, tm = 0u, hw_m = 0u;
    uint32_t n0 = 0u, tn = 0u;
    uint32_t hw_k = 0u, hw_n = 0u;
    uint32_t scale_d_u32 = 1u;
    float scale_d = 1.0f;
    std::array<float, 64> s_a_inline{};
    std::vector<float> s_a_overflow;
    std::vector<float> s_b;
  };
  std::array<PendingF32Tile, OB_MAX_CREDITS> ring{};
  uint32_t ring_head = 0u;
  uint32_t ring_count = 0u;
  std::vector<int8_t>& out_tile = ctx->gemm_out_scratch;

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
        uint32_t eff_d = p.scale_d_u32;

        // Saturation-proof on-TPU retry: if any active output byte hit int8 limits
        // (-128 or +127, e.g. from aligned outlier channels), immediately re-run THIS
        // tile on TPU hardware with 4x larger D (zero CPU fallback; gemm_in_bufs[slot]
        // and w_tile->ext2_buf are still populated and pinned!).
        if (rst == OB_OK && builtin_active && p.w_tile != nullptr) {
          auto is_saturated = [&]() -> bool {
            for (uint32_t mi = 0; mi < p.tm; ++mi) {
              const int8_t* o_row = out_tile.data() + static_cast<size_t>(mi) * p.hw_n;
              for (uint32_t ni = 0; ni < p.tn; ++ni) {
                int8_t qv = o_row[ni];
                if (qv == 127 || qv == -128) return true;
              }
            }
            return false;
          };
          static constexpr double kRetryMults[5] = {1.25, 1.5, 2.0, 4.0, 4.0};
          for (uint32_t retry = 0u; retry < 5u && eff_d < (1u << 24) && is_saturated(); ++retry) {
            eff_d = ob::quant_scale_D_grid(
                std::min<double>(static_cast<double>(eff_d) * kRetryMults[retry],
                                 static_cast<double>(1u << 24)));
            ob_gemm_desc_t rdesc {};
            rdesc.m_dim = p.hw_m;
            rdesc.k_dim = p.hw_k;
            rdesc.n_dim = p.hw_n;
            rdesc.s_in = 1.0f;
            rdesc.s_w = 1.0f;
            rdesc.s_out = static_cast<float>(eff_d);
            rdesc.activation = OB_ACT_NONE;
            rdesc.cache_key = ob::hash_shape_key(
                p.hw_m, p.hw_k, p.hw_n, 0, 0, eff_d, p.slot % param_swap_slots);
            ob_graph_t* rgraph = nullptr;
            bool rhit = false;
            rst = ob::acquire_tile_graph(ctx, emitter, emitter_ud, rdesc, &rgraph, &rhit);
            if (rst != OB_OK) break;
            if (rhit) cache_hits += 1u;
            tiles_dispatched += 1u;
            ob_exec_handle_t rh = 0u;
            rst = ob::graph_submit_async_with_params(
                rgraph, &ctx->gemm_in_bufs[p.slot], 1u, &ctx->gemm_out_bufs[p.slot], 1u,
                p.w_tile->ext2_buf, &rh);
            if (rst == OB_OK) {
              ob_invoke_stats_t rstats {};
              rst = ob_graph_wait(ctx, rh, ctx->cfg.timeout_ms, &rstats);
              if (rst == OB_OK) {
                total_hw_us += rstats.hw_execution_us;
                rst = ob_buffer_read(ctx->gemm_out_bufs[p.slot], out_tile.data(), out_bytes, 0u);
              }
            }
            ob_graph_release(rgraph);
            if (rst != OB_OK) break;
          }
        }

        if (rst == OB_OK) {
          const ob_gemm_f32_quant_job_t& job = jobs[p.job_idx];
          const float eff_scale_d = static_cast<float>(eff_d);
          const float* sa_ptr = (p.tm <= p.s_a_inline.size()) ? p.s_a_inline.data() : p.s_a_overflow.data();
          const float* sb_ptr = p.w_tile ? p.w_tile->row_s_b.data() : p.s_b.data();
          for (uint32_t mi = 0; mi < p.tm; ++mi) {
            const float sa_d = sa_ptr[mi] * eff_scale_d;
            float* c_row = job.c_mn + static_cast<size_t>(p.m0 + mi) * job.n_dim + p.n0;
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
  std::vector<int8_t>& a_q_tile = ctx->gemm_aq_scratch;
  std::vector<int8_t>& b_q_tile = ctx->gemm_bq_scratch;
  std::vector<float>& tmp_w_row = ctx->gemm_wrow_scratch;
  std::vector<float> row_s_a;
  std::vector<float> row_s_b;
  std::vector<std::vector<std::pair<uint32_t, float>>> row_outliers;
  std::vector<uint8_t> packed_tile;

  struct ScheduledTile {
    uint32_t job_idx;
    ob::TileRect tr;
  };
  std::vector<ScheduledTile> sched;

  bool allow_lt_64x = (k_dim >= 256u && max_k >= 512u && max_n >= 128u);
  for (uint32_t j = 0; j < num_jobs && allow_lt_64x; ++j) {
    if (jobs[j].n_dim < 128u) allow_lt_64x = false;
  }
  std::vector<ob::MSlice> m_slices = ob::build_m_slices(
      m_dim, max_m, builtin_active, true, allow_lt_64x);

  for (const ob::MSlice& ms : m_slices) {
    const uint32_t m0 = ms.m0;
    const uint32_t tm = ms.tm;
    const uint32_t hw_m = ms.hw_m;

    sched.clear();
    for (uint32_t j = 0; j < num_jobs; ++j) {
      std::vector<ob::TileRect> jtiles = ob::build_tile_rects(
          hw_m, k_dim, jobs[j].n_dim, max_k, max_n, OB_QK8_0, builtin_active, true);
      for (const ob::TileRect& tr : jtiles) {
        sched.push_back({j, tr});
      }
    }
    if (num_jobs > 1u) {
      // Group sibling projections sharing (k0, tk, hw_k) so A is quantized/peeled ONCE!
      std::stable_sort(sched.begin(), sched.end(),
                       [](const ScheduledTile& a, const ScheduledTile& b) {
                         if (a.tr.k0 != b.tr.k0) return a.tr.k0 < b.tr.k0;
                         if (a.tr.tk != b.tr.tk) return a.tr.tk < b.tr.tk;
                         return a.tr.hw_k < b.tr.hw_k;
                       });
    }

    row_s_a.resize(tm);
    row_outliers.resize(tm);
    uint32_t cached_a_k0 = 0xFFFFFFFFu;
    uint32_t cached_a_tk = 0u;
    uint32_t cached_aq_hw_k = 0u;
    uint32_t cached_a_rms_shift = 0u;

    for (const ScheduledTile& st_item : sched) {
      const uint32_t job_idx = st_item.job_idx;
      const ob_gemm_f32_quant_job_t& job = jobs[job_idx];
      const bool is_q4 = (job.is_q4_0 != 0u);
      const ob_block_q8_0_t* b_q8 = is_q4 ? nullptr : static_cast<const ob_block_q8_0_t*>(job.b_nk);
      const ob_block_q4_0_t* b_q4 = is_q4 ? static_cast<const ob_block_q4_0_t*>(job.b_nk) : nullptr;
      const uint32_t n_dim = job.n_dim;
      float* c_mn = job.c_mn;

      const uint32_t k0 = st_item.tr.k0;
      const uint32_t tk = st_item.tr.tk;
      const uint32_t n0 = st_item.tr.n0;
      const uint32_t tn = st_item.tr.tn;
      const uint32_t hw_k = st_item.tr.hw_k;
      const uint32_t hw_n = st_item.tr.hw_n;
      const uint32_t blk0 = k0 / OB_QK8_0;
      const uint32_t blk_cnt = tk / OB_QK8_0;
      tiles_dispatched += 1u;

      bool tile_handled_by_hw = false;
      if (emitter != nullptr) {
        // Quantize A tile [hw_m, hw_k] (+ peel <=8 extreme outlier channels and compute
        // coarse power-of-two RMS shift) ONCE per shared (k0, tk, hw_k) across all jobs!
        if (cached_a_k0 != k0 || cached_a_tk != tk || cached_aq_hw_k != hw_k) {
          a_q_tile.assign(static_cast<size_t>(hw_m) * hw_k, 0);
          for (uint32_t mi = 0; mi < tm; ++mi) {
            const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0;
            int8_t* aq_row = a_q_tile.data() + static_cast<size_t>(mi) * hw_k;
            ob::quantize_activation_row_split(
                a_row, tk, aq_row, &row_s_a[mi],
                builtin_active ? &row_outliers[mi] : nullptr);
          }
          cached_a_rms_shift = builtin_active
              ? ob::compute_activation_rms_shift(a_q_tile.data(), tm, tk, hw_k)
              : 0u;
          cached_a_k0 = k0;
          cached_a_tk = tk;
          cached_aq_hw_k = hw_k;
        }

        ob::CachedWeightTile* w_tile = nullptr;
        uint32_t scale_d = 1u;
        if (builtin_active) {
          const bool is_lt64_tile = ob::is_lt_64x_shape(hw_m, hw_k, hw_n);
          const uint32_t pack_fmt = is_lt64_tile ? 1u : 0u;
          // O(1) strided content hash of raw Q8_0 or Q4_0 blocks [n0..n0+tn) x [blk0..blk0+blk_cnt).
          // On cache hit, skips ALL Q4_0/Q8_0 unpack + dequant + requant + compute_tile_scale_D + pack_ext2!
          uint64_t wh = is_q4 ? 0x51345F5754494C45ULL : 0x51385F5754494C45ULL; // "Q4_WTILE" / "Q8_WTILE"
          uint32_t whdr[5] = {tn, tk, hw_k, hw_n, pack_fmt};
          wh = ob::fast_hash64_update(wh, whdr, sizeof(whdr));
          const size_t blk_sz = is_q4 ? sizeof(ob_block_q4_0_t) : sizeof(ob_block_q8_0_t);
          const size_t row_blk_bytes = static_cast<size_t>(blk_cnt) * blk_sz;
          const size_t row_stride_bytes = static_cast<size_t>(nb_k) * blk_sz;
          const uint8_t* raw_base = static_cast<const uint8_t*>(job.b_nk) +
                                    (static_cast<size_t>(n0) * nb_k + blk0) * blk_sz;
          wh = ob::hash_weight_tile_fast(wh, raw_base, row_stride_bytes, tn, row_blk_bytes);

          w_tile = ob::weight_cache_find_locked(ctx, wh);
          if (!w_tile) {
            std::vector<float> sb_vec;
            if (is_q4) {
              ob::requant_q4_0_tile(b_q4, nb_k, n0, tn, blk0, blk_cnt,
                                    tk, hw_k, hw_n, &b_q_tile, &sb_vec, &tmp_w_row);
            } else {
              ob::requant_q8_0_tile(b_q8, nb_k, n0, tn, blk0, blk_cnt,
                                    tk, hw_k, hw_n, &b_q_tile, &sb_vec, &tmp_w_row);
            }

            scale_d = ob::compute_tile_scale_D(b_q_tile.data(), tn, tk, hw_k);
            std::vector<uint8_t> ext2 = is_lt64_tile
                ? ob::pack_ext2_for_lt_64x(hw_n, hw_k, b_q_tile.data(), nullptr)
                : ob::pack_ext2_for_tile(hw_n, hw_k, b_q_tile.data(), nullptr);
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
          if (is_q4) {
            ob::requant_q4_0_tile(b_q4, nb_k, n0, tn, blk0, blk_cnt,
                                  tk, hw_k, hw_n, &b_q_tile, &row_s_b, &tmp_w_row);
          } else {
            ob::requant_q8_0_tile(b_q8, nb_k, n0, tn, blk0, blk_cnt,
                                  tk, hw_k, hw_n, &b_q_tile, &row_s_b, &tmp_w_row);
          }
          scale_d = ob::compute_tile_scale_D(b_q_tile.data(), tn, tk, hw_k);
        }

        if (builtin_active && cached_a_rms_shift > 0u) {
          scale_d = ob::quant_scale_D_grid(
              static_cast<double>(std::max<uint32_t>(1u, scale_d >> cached_a_rms_shift)));
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
          p.job_idx = job_idx;
          p.slot = slot;
          p.m0 = m0;
          p.tm = tm;
          p.hw_m = hw_m;
          p.n0 = n0;
          p.tn = tn;
          p.hw_k = hw_k;
          p.hw_n = hw_n;
          p.scale_d_u32 = scale_d;
          p.scale_d = static_cast<float>(scale_d);
          if (tm <= p.s_a_inline.size()) {
            for (uint32_t mi = 0; mi < tm; ++mi) p.s_a_inline[mi] = row_s_a[mi];
          } else {
            p.s_a_overflow = row_s_a;
          }
          if (!w_tile) p.s_b = row_s_b;
          ring_count += 1u;

          if (!ctx->mock_mode) {
            tile_handled_by_hw = true;
            // Overlapped with in-flight TPU tile: add sparse exact FP32 residual for <=8 peeled outliers.
            for (uint32_t mi = 0; mi < tm; ++mi) {
              if (row_outliers[mi].empty()) continue;
              float* c_row = c_mn + static_cast<size_t>(m0 + mi) * n_dim + n0;
              for (const auto& ov : row_outliers[mi]) {
                const uint32_t abs_k = k0 + ov.first;
                const uint32_t b_idx = abs_k / OB_QK8_0;
                const uint32_t j_idx = abs_k % OB_QK8_0;
                const float x_val = ov.second;
                if (is_q4) {
                  constexpr uint32_t kHalf = OB_QK8_0 / 2u;
                  const bool hi_nib = (j_idx >= kHalf);
                  const uint32_t byte_idx = hi_nib ? (j_idx - kHalf) : j_idx;
                  for (uint32_t ni = 0; ni < tn; ++ni) {
                    const ob_block_q4_0_t& blk =
                        b_q4[static_cast<size_t>(n0 + ni) * nb_k + b_idx];
                    const uint8_t byte = blk.qs[byte_idx];
                    const int32_t q = static_cast<int32_t>(hi_nib ? (byte >> 4) : (byte & 0x0Fu)) - 8;
                    c_row[ni] += x_val * (ob::fp16_to_fp32(blk.d) * static_cast<float>(q));
                  }
                } else {
                  for (uint32_t ni = 0; ni < tn; ++ni) {
                    const ob_block_q8_0_t& blk =
                        b_q8[static_cast<size_t>(n0 + ni) * nb_k + b_idx];
                    c_row[ni] += x_val * (ob::fp16_to_fp32(blk.d) * static_cast<float>(blk.qs[j_idx]));
                  }
                }
              }
            }
          }
        } else if (gst != OB_ERR_UNSUPPORTED) {
          drain_all_cleanup();
          return gst;
        }
      }

      if (!tile_handled_by_hw) {
        used_cpu_fallback = true;
        if (is_q4) {
          ob::cpu_matmul_f32_q4_0_tile(a_mk, b_q4, c_mn, k_dim, n_dim,
                                       m0, tm, k0, tk, n0, tn);
        } else {
          ob::cpu_matmul_f32_q8_0_tile(a_mk, b_q8, c_mn, k_dim, n_dim,
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

  for (uint32_t j = 0; j < num_jobs; ++j) {
    if (jobs[j].activation != OB_ACT_NONE) {
      ob::apply_activation_inplace(
          jobs[j].c_mn, static_cast<size_t>(m_dim) * jobs[j].n_dim, jobs[j].activation);
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
  ob_gemm_f32_quant_job_t job {};
  job.b_nk = b_nk_q8_0;
  job.c_mn = c_mn;
  job.n_dim = n_dim;
  job.activation = OB_ACT_NONE;
  job.is_q4_0 = 0u;
  return ob_gemm_matmul_f32_quant_batch(ctx, a_mk, m_dim, k_dim, &job, 1u, out_stats);
}

}  // extern "C"
