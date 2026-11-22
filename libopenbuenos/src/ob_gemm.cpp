#include "ob_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ob {

namespace {

constexpr size_t kMinPipelineBufBytes = 131072u; // 128 KiB covers 64x2048 without resize churn
constexpr uint32_t kMaxFusedSubs = 8u;

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

inline void quant_sym_i8_row(const float* src, int8_t* dst, uint32_t len, float inv_scale) {
  uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t v_inv = vdupq_n_f32(inv_scale);
  const int32x4_t v_min = vdupq_n_s32(-127);
  const int32x4_t v_max = vdupq_n_s32(127);
  for (; ki + 16u <= len; ki += 16u) {
    int32x4_t q0 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + ki), v_inv)), v_min), v_max);
    int32x4_t q1 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + ki + 4u), v_inv)), v_min), v_max);
    int32x4_t q2 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + ki + 8u), v_inv)), v_min), v_max);
    int32x4_t q3 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + ki + 12u), v_inv)), v_min), v_max);
    int16x8_t s01 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
    int16x8_t s23 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
    vst1q_s8(dst + ki, vcombine_s8(vmovn_s16(s01), vmovn_s16(s23)));
  }
#endif
  for (; ki < len; ++ki) {
    dst[ki] = quant_sym_i8(src[ki] * inv_scale);
  }
}

#if defined(__ARM_NEON) && defined(__aarch64__)
// High-accuracy (< 2e-7 relative error) ARM64 NEON expf(x) using Cody-Waite
// range reduction + degree-6 Horner polynomial on [-ln2/2, +ln2/2].
inline float32x4_t fast_exp_f32x4(float32x4_t x) {
  const float32x4_t v_hi = vdupq_n_f32(88.3762626647949f);
  const float32x4_t v_lo = vdupq_n_f32(-87.3365447504f);
  x = vmaxq_f32(vminq_f32(x, v_hi), v_lo);

  const float32x4_t v_inv_ln2 = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t v_ln2_hi = vdupq_n_f32(0.693145751953125f);
  const float32x4_t v_ln2_lo = vdupq_n_f32(1.428606765330187045e-6f);

  float32x4_t nf = vrndnq_f32(vmulq_f32(x, v_inv_ln2));
  float32x4_t r = vfmsq_f32(vfmsq_f32(x, nf, v_ln2_hi), nf, v_ln2_lo);

  float32x4_t p = vdupq_n_f32(0.00138888892f);
  p = vfmaq_f32(vdupq_n_f32(0.00833333377f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.04166666791f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.16666667163f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);

  int32x4_t ni = vcvtnq_s32_f32(nf);
  float32x4_t pow2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
  return vmulq_f32(p, pow2);
}
#endif

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

struct WeightHash128 {
  uint64_t h1 = 0x14650fb0739d0383ULL;
  uint64_t h2 = 0x517cc1b727220a95ULL;
  uint64_t bytes = 0ULL;
};

inline void hash128_update(WeightHash128* st, const void* data, size_t len) {
  const uint64_t next_h1 = fast_hash64_update(st->h1, data, len);
  uint64_t h2 = st->h2 ^ (static_cast<uint64_t>(len) * 0x9e3779b97f4a7c15ULL);
  const uint8_t* p = static_cast<const uint8_t*>(data);
  size_t i = 0;
  for (; i + 8u <= len; i += 8u) {
    uint64_t w = 0ULL;
    std::memcpy(&w, p + i, 8u);
    h2 ^= (w << 19) | (w >> 45);
    h2 *= 0x94d049bb133111ebULL;
    h2 ^= (h2 >> 29);
  }
  if (i < len) {
    uint64_t tail = 0ULL;
    std::memcpy(&tail, p + i, len - i);
    h2 ^= tail ^ (static_cast<uint64_t>(len - i) << 51);
    h2 *= 0xbf58476d1ce4e5b9ULL;
    h2 ^= (h2 >> 33);
  }
  st->h1 = next_h1;
  st->h2 = h2 ^ (next_h1 >> 17);
  st->bytes += static_cast<uint64_t>(len);
}

inline WeightTileKey finalize_weight_key(const WeightHash128& st) {
  WeightTileKey k;
  k.h1 = (st.h1 == 0ULL) ? 1ULL : st.h1;
  k.h2 = (st.h2 == 0ULL) ? 1ULL : st.h2;
  k.bytes = st.bytes;
  return k;
}

// Content weight-tile hash for ext[2] DMA-BUF LRU cache lookup (128-bit key).
// - Default (strict_hash=true) or small tiles (<= 16 KiB): hashes every byte.
// - Explicit opt-out strict_hash=false on large tiles (> 16 KiB): strided 64-row fingerprint + edges.
void hash_weight_tile_fast(WeightHash128* st,
                           const uint8_t* base,
                           size_t row_stride_bytes,
                           uint32_t tn,
                           size_t row_bytes,
                           bool strict_hash) {
  if (tn == 0u || row_bytes == 0u) return;
  const size_t total_bytes = static_cast<size_t>(tn) * row_bytes;
  if (total_bytes <= 16384u || strict_hash) {
    for (uint32_t ni = 0; ni < tn; ++ni) {
      hash128_update(st, base + static_cast<size_t>(ni) * row_stride_bytes, row_bytes);
    }
    return;
  }

  const size_t edge_len = std::min<size_t>(row_bytes, 64u);
  hash128_update(st, base, edge_len);
  const uint8_t* last_row = base + static_cast<size_t>(tn - 1u) * row_stride_bytes;
  hash128_update(st, last_row + (row_bytes - edge_len), edge_len);

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
    hash128_update(st, base + static_cast<size_t>(ni) * row_stride_bytes + off, probe_len);
  }
}

bool lookup_immutable_region_locked(const ob_context* ctx,
                                    const void* ptr,
                                    size_t row_stride_bytes,
                                    uint32_t tn,
                                    size_t row_bytes,
                                    uint64_t* out_version,
                                    uint64_t* out_offset) {
  if (!ctx || !ptr || tn == 0u || row_bytes == 0u || ctx->immutable_weight_regions.empty()) {
    return false;
  }
  const size_t max_bytes = static_cast<size_t>(-1);
  if (tn > 1u && row_stride_bytes > 0u &&
      static_cast<size_t>(tn - 1u) > (max_bytes - row_bytes) / row_stride_bytes) {
    return false;
  }
  const size_t span_bytes =
      static_cast<size_t>(tn - 1u) * row_stride_bytes + row_bytes;
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  for (const ImmutableWeightRegion& r : ctx->immutable_weight_regions) {
    if (addr >= r.begin && addr <= r.end &&
        span_bytes <= static_cast<size_t>(r.end - addr)) {
      *out_version = r.version;
      *out_offset = static_cast<uint64_t>(addr - r.begin);
      return true;
    }
  }
  return false;
}

void hash_weight_tile_for_ctx(const ob_context* ctx,
                              WeightHash128* st,
                              const uint8_t* base,
                              size_t row_stride_bytes,
                              uint32_t tn,
                              size_t row_bytes,
                              bool strict_hash) {
  uint64_t reg_ver = 0u;
  uint64_t reg_off = 0u;
  if (lookup_immutable_region_locked(
          ctx, base, row_stride_bytes, tn, row_bytes, &reg_ver, &reg_off)) {
    const uint64_t imm_hdr[7] = {
        0x494D4D5554575431ULL, // "IMMUTWT1"
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)),
        reg_ver,
        reg_off,
        static_cast<uint64_t>(row_stride_bytes),
        static_cast<uint64_t>(tn),
        static_cast<uint64_t>(row_bytes),
    };
    hash128_update(st, imm_hdr, sizeof(imm_hdr));
    return;
  }
  hash_weight_tile_fast(st, base, row_stride_bytes, tn, row_bytes, strict_hash);
}

struct ScopedBufferRef {
  ob_buffer* buf = nullptr;
  ScopedBufferRef() = default;
  explicit ScopedBufferRef(ob_buffer* b) : buf(b) {
    if (buf) buffer_retain(buf);
  }
  ScopedBufferRef(const ScopedBufferRef&) = delete;
  ScopedBufferRef& operator=(const ScopedBufferRef&) = delete;
  void reset(ob_buffer* b = nullptr) {
    if (buf == b) return;
    if (buf) ob_buffer_free(buf);
    buf = b;
    if (buf) buffer_retain(buf);
  }
  ~ScopedBufferRef() {
    if (buf) ob_buffer_free(buf);
  }
};

void detach_timed_out_buffers_locked(ob_context* ctx,
                                     ob_exec_handle_t handle,
                                     CachedWeightTile* w_tile,
                                     ob_buffer* in_buf,
                                     ob_buffer* out_buf) {
  if (!ctx || handle == 0u) return;
  if (device_attach_inflight_buffers(
          ctx, handle, {w_tile ? w_tile->ext2_buf : nullptr, in_buf, out_buf})) {
    if (w_tile && w_tile->in_cache) {
      ctx->weight_cache_map.erase(w_tile->key);
      const size_t total_b = w_tile->ext2_bytes + w_tile->outlier_cols_bytes;
      ctx->weight_cache_bytes -= std::min(ctx->weight_cache_bytes, total_b);
      ctx->weight_cache_lru.erase(w_tile->lru_it);
      w_tile->in_cache = false;
    }
    auto drop_match = [](ob_buffer*& slot_ptr, ob_buffer* match) {
      if (slot_ptr == match && match != nullptr) {
        ob_buffer_free(slot_ptr);
        slot_ptr = nullptr;
      }
    };
    for (uint32_t i = 0u; i < OB_MAX_CREDITS; ++i) {
      drop_match(ctx->gemm_in_bufs[i], in_buf);
      drop_match(ctx->gemm_in_bufs_4k[i], in_buf);
      drop_match(ctx->gemm_out_bufs[i], out_buf);
      drop_match(ctx->gemm_out_bufs_4k[i], out_buf);
    }
  }
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

size_t read_system_mem_available_bytes() {
#if defined(__linux__)
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0u;
  char line[256];
  size_t avail_bytes = 0u;
  while (std::fgets(line, sizeof(line), f)) {
    unsigned long long kb = 0ULL;
    if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
      avail_bytes = static_cast<size_t>(kb) * 1024u;
      break;
    }
  }
  std::fclose(f);
  return avail_bytes;
#else
  return 0u;
#endif
}

size_t compute_weight_cache_budget_bytes_impl(ob_context* ctx) {
  if (!ctx) return 0u;
  size_t base_cap = 0u;
  const char* env = std::getenv("OPENBUENOS_WEIGHT_CACHE_MB");
  if (env && env[0]) {
    unsigned long mb = std::strtoul(env, nullptr, 10);
    if (mb < 16ul) mb = 16ul;
    if (mb > 8192ul) mb = 8192ul;
    base_cap = static_cast<size_t>(mb) * 1024u * 1024u;
  } else {
    // 256 MiB on host mock; up to 2048 MiB on live Android device (bounded below MemAvailable/3).
    base_cap = ctx->mock_mode
        ? (256u * 1024u * 1024u)
        : (2048u * 1024u * 1024u);
  }
  size_t mem_avail = ctx->simulated_mem_available_bytes;
  if (mem_avail == 0u && !ctx->mock_mode) {
    mem_avail = read_system_mem_available_bytes();
    if (mem_avail > 0u) {
      // /proc/meminfo MemAvailable excludes our own unswappable DMA-BUF cache;
      // add weight_cache_bytes back so our own cache cannot self-ratchet downward.
      mem_avail += ctx->weight_cache_bytes;
    }
  }
  if (mem_avail > 0u) {
    // Android LMKD-safe ceiling: never let cached ext[2] DMA-BUFs exceed MemAvailable / 3.
    const size_t lmkd_cap = mem_avail / 3u;
    if (lmkd_cap < base_cap) {
      base_cap = lmkd_cap;
    }
  }
  return std::max<size_t>(base_cap, 1u * 1024u * 1024u);
}

void ensure_weight_cache_budget(ob_context* ctx, bool refresh_meminfo = false) {
  if (ctx->weight_cache_max_bytes == 0u || refresh_meminfo ||
      ctx->simulated_mem_available_bytes > 0u) {
    ctx->weight_cache_max_bytes = compute_weight_cache_budget_bytes_impl(ctx);
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

CachedWeightTile* weight_cache_find_locked(ob_context* ctx, const WeightTileKey& key) {
  auto it = ctx->weight_cache_map.find(key);
  if (it == ctx->weight_cache_map.end()) return nullptr;
  CachedWeightTile* t = it->second;
  if (t->lru_it != ctx->weight_cache_lru.begin()) {
    ctx->weight_cache_lru.splice(ctx->weight_cache_lru.begin(), ctx->weight_cache_lru, t->lru_it);
  }
  ctx->weight_cache_hits += 1u;
  return t;
}

void weight_cache_evict_locked(ob_context* ctx, size_t incoming_bytes) {
  const bool periodic_refresh =
      (ctx->simulated_mem_available_bytes > 0u) ||
      (!ctx->mock_mode && ((ctx->weight_cache_misses & 31u) == 0u ||
                           ctx->weight_cache_bytes + incoming_bytes > ctx->weight_cache_max_bytes));
  ensure_weight_cache_budget(ctx, periodic_refresh);
  auto it = ctx->weight_cache_lru.end();
  while ((ctx->weight_cache_bytes + incoming_bytes > ctx->weight_cache_max_bytes ||
          ctx->weight_cache_map.size() >= ctx->weight_cache_max_entries) &&
         it != ctx->weight_cache_lru.begin()) {
    --it;
    CachedWeightTile* victim = *it;
    if (victim->in_use > 0u) continue;
    auto erase_it = it;
    ctx->weight_cache_map.erase(victim->key);
    const size_t victim_total = victim->ext2_bytes + victim->outlier_cols_bytes;
    ctx->weight_cache_bytes -= std::min(ctx->weight_cache_bytes, victim_total);
    victim->in_cache = false;
    it = ctx->weight_cache_lru.erase(erase_it);
    if (victim->ext2_buf) ob_buffer_free(victim->ext2_buf);
    delete victim;
    ctx->weight_cache_evictions += 1u;
  }
}

void weight_cache_insert_locked(ob_context* ctx, CachedWeightTile* tile) {
  weight_cache_evict_locked(ctx, tile->ext2_bytes);
  ctx->weight_cache_lru.push_front(tile);
  tile->lru_it = ctx->weight_cache_lru.begin();
  tile->in_cache = true;
  ctx->weight_cache_map[tile->key] = tile;
  ctx->weight_cache_bytes += tile->ext2_bytes;
  if (ctx->weight_cache_bytes > ctx->weight_cache_peak_bytes) {
    ctx->weight_cache_peak_bytes = ctx->weight_cache_bytes;
  }
  ctx->weight_cache_misses += 1u;
}

template <typename DrainOneFn>
ob_status_t weight_cache_prepare_space_locked(ob_context* ctx,
                                              size_t incoming_bytes,
                                              uint32_t& ring_count,
                                              DrainOneFn& drain_one,
                                              bool* out_cacheable) {
  ensure_weight_cache_budget(ctx, true);
  if (incoming_bytes > ctx->weight_cache_max_bytes) {
    weight_cache_evict_locked(ctx, ctx->weight_cache_max_bytes);
    *out_cacheable = false;
    return OB_OK;
  }
  weight_cache_evict_locked(ctx, incoming_bytes);
  while ((ctx->weight_cache_bytes + incoming_bytes > ctx->weight_cache_max_bytes ||
          ctx->weight_cache_map.size() >= ctx->weight_cache_max_entries) &&
         ring_count > 0u) {
    ob_status_t dst = drain_one();
    if (dst != OB_OK) return dst;
    weight_cache_evict_locked(ctx, incoming_bytes);
  }
  *out_cacheable =
      (ctx->weight_cache_bytes + incoming_bytes <= ctx->weight_cache_max_bytes) &&
      (ctx->weight_cache_map.size() < ctx->weight_cache_max_entries);
  return OB_OK;
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
    uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
    uint32x4_t v_sq = vdupq_n_u32(0u);
    uint32x4_t v_l1 = vdupq_n_u32(0u);
    int32x4_t v_ss = vdupq_n_s32(0);
    for (; ki + 16u <= tk; ki += 16u) {
      int8x16_t v = vld1q_s8(row + ki);
      int16x8_t lo = vmull_s8(vget_low_s8(v), vget_low_s8(v));
      int16x8_t hi = vmull_high_s8(v, v);
      v_sq = vpadalq_u16(v_sq, vreinterpretq_u16_s16(lo));
      v_sq = vpadalq_u16(v_sq, vreinterpretq_u16_s16(hi));

      // ABS(-128) = 0x80 == 128u when reinterpreted as uint8: exact |v| for all int8!
      uint8x16_t av = vreinterpretq_u8_s8(vabsq_s8(v));
      v_l1 = vpadalq_u16(v_l1, vpaddlq_u8(av));
      v_ss = vpadalq_s16(v_ss, vpaddlq_s8(v));
    }
    sq = static_cast<uint64_t>(vaddvq_u32(v_sq));
    l1 = static_cast<uint64_t>(vaddvq_u32(v_l1));
    ssum = static_cast<int64_t>(vaddvq_s32(v_ss));
#endif
    for (; ki < tk; ++ki) {
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

// Persistent pre-mapped bidirectional pipeline DMA-BUF allocator:
// checks true slab.alloc_size (not requested_size) and sizes >= 128 KiB once so
// shape changes across M=1/M=64 never free or reallocate during inference.
ob_status_t ensure_pipeline_buf(ob_context* ctx,
                                ob_buffer_t** slot_buf,
                                size_t needed_bytes) {
  const size_t req = std::max<size_t>(needed_bytes, 4096u);
  if (*slot_buf && (*slot_buf)->slab.alloc_size >= req) {
    if ((*slot_buf)->requested_size < req) {
      (*slot_buf)->requested_size = req;
    }
    return OB_OK;
  }
  if (*slot_buf) {
    ob_buffer_free(*slot_buf);
    *slot_buf = nullptr;
  }
  const size_t floor_sz = (ctx->pool_max_bytes >= 4u * 1024u * 1024u)
                              ? kMinPipelineBufBytes
                              : 4096u;
  const size_t alloc_req = std::max<size_t>(req, floor_sz);
  ob_status_t st = ob_buffer_alloc(
      ctx, alloc_req, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, slot_buf);
  if (st == OB_OK && *slot_buf && (*slot_buf)->requested_size < req) {
    (*slot_buf)->requested_size = req;
  }
  return st;
}

// 2-tier persistent pre-mapped bidirectional pipeline DMA-BUF allocator:
//   - Small tier (needed_bytes <= 4096): exact 4 KiB slab in `small_buf` (eliminates
//     32-page 128 KiB DMA_BUF_IOCTL_SYNC cache flush/invalidate on M=1 decode tiles!).
//   - Large tier (needed_bytes > 4096): >= 128 KiB slab in `large_buf` (zero churn
//     across M=64 prefill shapes).
ob_status_t ensure_pipeline_buf_tiered(ob_context* ctx,
                                       ob_buffer_t** small_buf,
                                       ob_buffer_t** large_buf,
                                       size_t needed_bytes,
                                       ob_buffer_t** out_buf) {
  if (needed_bytes <= 4096u) {
    if (*small_buf && (*small_buf)->slab.alloc_size >= 4096u) {
      *out_buf = *small_buf;
      return OB_OK;
    }
    if (*small_buf) {
      ob_buffer_free(*small_buf);
      *small_buf = nullptr;
    }
    ob_status_t st = ob_buffer_alloc(
        ctx, 4096u, OB_BUF_BIDIRECTIONAL | OB_BUF_ZERO_INIT, small_buf);
    if (st == OB_OK) *out_buf = *small_buf;
    return st;
  }
  ob_status_t st = ensure_pipeline_buf(ctx, large_buf, needed_bytes);
  if (st == OB_OK) *out_buf = *large_buf;
  return st;
}

inline void neon_fma_row_f32(float* dst, const float* src, uint32_t count, float scale) {
  uint32_t i = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 16u <= count; i += 16u) {
    vst1q_f32(dst + i,       vfmaq_f32(vld1q_f32(dst + i),       vld1q_f32(src + i),       vs));
    vst1q_f32(dst + i + 4u,  vfmaq_f32(vld1q_f32(dst + i + 4u),  vld1q_f32(src + i + 4u),  vs));
    vst1q_f32(dst + i + 8u,  vfmaq_f32(vld1q_f32(dst + i + 8u),  vld1q_f32(src + i + 8u),  vs));
    vst1q_f32(dst + i + 12u, vfmaq_f32(vld1q_f32(dst + i + 12u), vld1q_f32(src + i + 12u), vs));
  }
  for (; i + 4u <= count; i += 4u) {
    vst1q_f32(dst + i, vfmaq_f32(vld1q_f32(dst + i), vld1q_f32(src + i), vs));
  }
#endif
  for (; i < count; ++i) {
    dst[i] += scale * src[i];
  }
}

// Half-octave scale grid (2^(k/2), step sqrt(2) ~= 1.414x): guarantees s_gate = s_a * s_wg * d_gate
// lands EXACT on 2^(k/2) (0.000% SiLU scale mismatch) with only ~5-8 distinct s_gate values per model!
inline float quantize_scale_pow2_2nd_up(float raw_s) {
  if (raw_s <= 0.0f) return 0.0f;
  double e = std::ceil(2.0 * std::log2(static_cast<double>(raw_s)));
  return static_cast<float>(std::exp2(e * 0.5));
}

inline uint32_t round_pow2_clamped(double val, uint32_t lo, uint32_t hi, double thresh = 1.45) {
  if (!(val > 0.0)) return lo;
  uint32_t p = lo;
  while (p < hi && static_cast<double>(p) * thresh < val) {
    p <<= 1u;
  }
  return p;
}

void compute_matrix_row_l2_stats(const int8_t* mat, uint32_t rows, uint32_t cols, uint32_t stride,
                                 float* out_max_l2, float* out_mean_l2,
                                 float* out_max_abs_sum = nullptr,
                                 float* out_max_l1 = nullptr,
                                 float* out_max_linf = nullptr) {
  double max_l2 = 0.0;
  double sum_l2 = 0.0;
  double max_abs_sum = 0.0;
  double max_l1 = 0.0;
  double max_linf = 0.0;
  for (uint32_t r = 0u; r < rows; ++r) {
    const int8_t* row = mat + static_cast<size_t>(r) * stride;
    uint64_t sq = 0ULL;
    int64_t row_sum = 0LL;
    uint64_t row_l1 = 0ULL;
    uint32_t row_linf = 0u;
    uint32_t c = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
    uint32x4_t v_sq = vdupq_n_u32(0u);
    int32x4_t v_sum = vdupq_n_s32(0);
    uint32x4_t v_l1 = vdupq_n_u32(0u);
    int16x8_t v_linf = vdupq_n_s16(0);
    for (; c + 16u <= cols; c += 16u) {
      int8x16_t v = vld1q_s8(row + c);
      int16x8_t lo = vmull_s8(vget_low_s8(v), vget_low_s8(v));
      int16x8_t hi = vmull_high_s8(v, v);
      v_sq = vpadalq_u16(v_sq, vreinterpretq_u16_s16(lo));
      v_sq = vpadalq_u16(v_sq, vreinterpretq_u16_s16(hi));
      v_sum = vaddq_s32(v_sum, vpaddlq_s16(vpaddlq_s8(v)));
      int16x8_t a_lo = vabsq_s16(vmovl_s8(vget_low_s8(v)));
      int16x8_t a_hi = vabsq_s16(vmovl_high_s8(v));
      v_l1 = vpadalq_u16(v_l1, vreinterpretq_u16_s16(a_lo));
      v_l1 = vpadalq_u16(v_l1, vreinterpretq_u16_s16(a_hi));
      v_linf = vmaxq_s16(v_linf, vmaxq_s16(a_lo, a_hi));
    }
    sq = static_cast<uint64_t>(vaddvq_u32(v_sq));
    row_sum = static_cast<int64_t>(vaddvq_s32(v_sum));
    row_l1 = static_cast<uint64_t>(vaddvq_u32(v_l1));
    row_linf = static_cast<uint32_t>(vmaxvq_s16(v_linf));
#endif
    for (; c < cols; ++c) {
      const int32_t v = static_cast<int32_t>(row[c]);
      const uint32_t av = static_cast<uint32_t>(v < 0 ? -v : v);
      sq += static_cast<uint64_t>(v * v);
      row_sum += static_cast<int64_t>(v);
      row_l1 += static_cast<uint64_t>(av);
      if (av > row_linf) row_linf = av;
    }
    const double l2 = std::sqrt(static_cast<double>(sq));
    const double abs_sum = std::fabs(static_cast<double>(row_sum));
    const double l1_d = static_cast<double>(row_l1);
    const double linf_d = static_cast<double>(row_linf);
    if (l2 > max_l2) max_l2 = l2;
    if (abs_sum > max_abs_sum) max_abs_sum = abs_sum;
    if (l1_d > max_l1) max_l1 = l1_d;
    if (linf_d > max_linf) max_linf = linf_d;
    sum_l2 += l2;
  }
  *out_max_l2 = static_cast<float>(max_l2);
  *out_mean_l2 = (rows > 0u) ? static_cast<float>(sum_l2 / static_cast<double>(rows)) : 0.0f;
  if (out_max_abs_sum) *out_max_abs_sum = static_cast<float>(max_abs_sum);
  if (out_max_l1) *out_max_l1 = static_cast<float>(max_l1);
  if (out_max_linf) *out_max_linf = static_cast<float>(max_linf);
}

void compute_swiglu_stage_dividers_impl(uint32_t m_dim,
                                        float rms_a,
                                        float l2_g_max, float l2_g_mean,
                                        float l2_u_max, float l2_u_mean,
                                        float l2_d_max,
                                        uint32_t* out_d_gate,
                                        uint32_t* out_d_up,
                                        uint32_t* out_d_mul,
                                        uint32_t* out_d_down) {
  const bool is_batched = (m_dim > 1u);
  // Collapse continuous M=1 token rms_a into 3 geometric buckets calibrated for 2^(1/2) s_a
  // and pin all 4 dividers to pure powers of 2 -> cuts Probe-A SwiGLU graphs across 30L x 64T
  // from ~1,344 down to ~45-85 (well below kDefaultGraphCacheCapacity=512, zero evictions!)
  // while keeping NMSE < 3e-3 across Gaussian & real heavy-tailed LLM activation crests.
  double ra = 0.0;
  if (is_batched) {
    ra = std::max<double>(static_cast<double>(rms_a), 14.0);
  } else if (rms_a < 14.0f) {
    ra = 11.0;
  } else if (rms_a < 30.0f) {
    ra = 24.0;
  } else if (rms_a <= 42.0f) {
    ra = 36.0;
  } else {
    ra = static_cast<double>(rms_a);
  }
  const double sig_gu = is_batched ? 4.0 : 3.65;
  const uint32_t d_gate = round_pow2_clamped((sig_gu * ra * static_cast<double>(l2_g_max)) / 127.0, 64u, 65536u, 1.42);
  const uint32_t d_up = round_pow2_clamped((sig_gu * ra * static_cast<double>(l2_u_max)) / 127.0, 64u, 65536u, 1.42);
  const double rms_qg = (ra * static_cast<double>(l2_g_mean)) / static_cast<double>(d_gate);
  const double rms_qu = (ra * static_cast<double>(l2_u_mean)) / static_cast<double>(d_up);
  // Deterministic per-(layer, ra_bucket) d_mul in {16, 32, 64} -> zero q_prod clip, zero per-token fan-out!
  const double dm_div = is_batched ? 13.0 : 18.0;
  const uint32_t d_mul = round_pow2_clamped((rms_qg * rms_qu) / dm_div, 16u, 64u, 1.38);
  const double rms_qprod = std::max<double>((0.68 * rms_qg * rms_qu) / static_cast<double>(d_mul), 2.0);
  // Heavy-tailed real LLM down-projection headroom -> eliminates live sat_retries!
  const double sig_down = is_batched ? 5.8 : 5.6;
  const double dd_thresh = is_batched ? 1.40 : 1.38;
  const uint32_t d_down = round_pow2_clamped(
      (sig_down * rms_qprod * static_cast<double>(l2_d_max)) / 127.0, 256u, 4096u, dd_thresh);
  *out_d_gate = d_gate;
  *out_d_up = d_up;
  *out_d_mul = d_mul;
  *out_d_down = d_down;
}

uint64_t hash_swiglu_graph_key(uint32_t hw_m, uint32_t hw_k, uint32_t hw_h, uint32_t hw_n,
                               uint32_t d_gate, uint32_t d_up, uint32_t d_mul, uint32_t d_down,
                               float s_gate, uint32_t slot_bucket) {
  uint32_t sg_bits = 0u;
  std::memcpy(&sg_bits, &s_gate, sizeof(sg_bits));
  uint64_t h = 0x535749474C555F31ULL; // "SWIGLU_1"
  uint32_t hdr[10] = {
      hw_m, hw_k, hw_h, hw_n, d_gate, d_up, d_mul, d_down, sg_bits, slot_bucket
  };
  h = fast_hash64_update(h, hdr, sizeof(hdr));
  return h == 0u ? 1u : h;
}

ob_status_t acquire_swiglu_entry(ob_context* ctx,
                                 const ob_swiglu_int8_desc_t& desc,
                                 uint64_t cache_key,
                                 CachedGraph** out_entry,
                                 bool* out_cache_hit) {
  *out_entry = nullptr;
  *out_cache_hit = false;
  if (cache_key != 0u && graph_cache_lookup_entry_u64(ctx, cache_key, out_entry)) {
    *out_cache_hit = true;
    return OB_OK;
  }
  if (ctx->builtin_dive_interp.empty()) {
    set_last_error(ctx, "Builtin emitter has no loaded RV64 DIVE interpreter for SwiGLU");
    return OB_ERR_UNSUPPORTED;
  }
  std::vector<uint8_t> dgc0;
  std::string err;
  ob_status_t st = build_swiglu_dgc0(
      desc, ctx->builtin_dive_interp.data(), ctx->builtin_dive_interp.size(),
      &dgc0, nullptr, &err);
  if (st != OB_OK) {
    if (!err.empty()) set_last_error(ctx, err);
    return st;
  }
  return graph_register_dgc0_internal(
      ctx, dgc0.data(), dgc0.size(), cache_key, true, out_entry, out_cache_hit);
}

// Zero-alloc fast-path cached graph acquisition returning CachedGraph* (+1 ref_count).
// Caller transfers that single +1 ref_count straight into device_submit_execute_async,
// eliminating per-tile `new/delete ob_graph` and halving cache_mu lock/unlock hops!
ob_status_t acquire_tile_entry(ob_context* ctx,
                               ob_gemm_emitter_fn emitter,
                               void* user_data,
                               const ob_gemm_desc_t& desc,
                               CachedGraph** out_entry,
                               bool* out_cache_hit) {
  *out_entry = nullptr;
  *out_cache_hit = false;

  if (desc.cache_key != 0u && graph_cache_lookup_entry_u64(ctx, desc.cache_key, out_entry)) {
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

  const bool share_zero = (desc.weights_nk == nullptr && desc.packed_params == nullptr);
  return graph_register_dgc0_internal(
      ctx, ctx->emitter_scratch.data(), dgc0_size, desc.cache_key,
      share_zero, out_entry, out_cache_hit);
}

inline void release_tile_entry(ob_context* ctx, CachedGraph* entry) {
  if (!ctx || !entry) return;
  std::lock_guard<std::mutex> lk(ctx->cache_mu);
  graph_entry_unref_locked(ctx, entry);
}

struct MSlice {
  uint32_t m0;
  uint32_t tm;
  uint32_t hw_m;
};

std::vector<MSlice> build_m_slices(uint32_t m_dim, uint32_t max_m,
                                   bool builtin_active, bool allow_m_pad,
                                   bool allow_lt_64x, bool batched_lt) {
  std::vector<MSlice> out;
  if (!builtin_active) {
    for (uint32_t m0 = 0; m0 < m_dim; m0 += max_m) {
      uint32_t tm = std::min(max_m, m_dim - m0);
      out.push_back({m0, tm, tm});
    }
    return out;
  }
  if (allow_m_pad && allow_lt_64x && batched_lt && m_dim >= 2u && max_m >= 64u) {
    for (uint32_t m0 = 0u; m0 < m_dim; ) {
      uint32_t tm = std::min<uint32_t>(64u, m_dim - m0);
      uint32_t hw_m = (tm == 1u) ? 1u : 64u;
      out.push_back({m0, tm, hw_m});
      m0 += tm;
    }
    return out;
  }
  if (allow_m_pad) {
    uint32_t m_cap = (max_m >= 16u) ? 16u : ((max_m >= 8u) ? 8u : 1u);
    for (uint32_t m0 = 0u; m0 < m_dim; ) {
      uint32_t tm = std::min(m_cap, m_dim - m0);
      uint32_t hw_m = (tm == 1u) ? 1u : ((tm <= 8u) ? 8u : 16u);
      out.push_back({m0, tm, hw_m});
      m0 += tm;
    }
    return out;
  }
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

struct FusedSubSlice {
  uint32_t job_idx = 0u;
  uint32_t n0 = 0u;
  uint32_t tn = 0u;
  uint32_t out_off = 0u;
};

struct ScheduledGroup {
  uint32_t k0 = 0u;
  uint32_t tk = 0u;
  uint32_t hw_k = 0u;
  uint32_t hw_n = 0u;
  uint32_t tn_total = 0u;
  uint32_t num_subs = 0u;
  std::array<FusedSubSlice, kMaxFusedSubs> subs{};
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
      hn = 128u;
    } else if (hn > 32u && hk > 48u && hk <= 64u) {
      hk = 64u;
      hn = 64u;
    }
  } else {
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
                                       bool allow_lt,
                                       bool lt_enabled,
                                       bool blt_enabled) {
  std::vector<TileRect> out;
  if (builtin_active && allow_lt && blt_enabled &&
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
  if (builtin_active && allow_lt && lt_enabled &&
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

// Build scheduled tile groups for ob_gemm_matmul_f32_quant_batch, performing
// horizontal weight-concat fusion across jobs sharing (k0, tk, hw_k) whenever
// concatenated slices fit into a single lt_1x* (N_sum <= 2048) or lt_64x*
// (N_sum <= 1536) large tile:
//   - Qwen2.5 Q+K+V (M=1): 1536+256+256 = 2048 -> 1x1536x2048 (3 -> 1 VII call!)
//   - SmolLM2 Q+K+V (M=1):  576+192+192 =  960 -> 1x576x1024  (3 -> 1 VII call!)
//   - Qwen2.5 Gate+Up tails (M=1): 768+768 = 1536 -> 1x1536x1536 (10 -> 9 VII calls!)
//   - SmolLM2 Q+K+V (M=64): 576+192+192 =  960 -> 64x576x960  (3 -> 1 VII call!)
//   - Qwen2.5 Q+K+V (M=64): 1536 + (256+256=512) -> 64x1536x1536 + 64x1536x512 (3 -> 2!)
void build_scheduled_groups(uint32_t hw_m,
                            uint32_t k_dim,
                            const ob_gemm_f32_quant_job_t* jobs,
                            uint32_t num_jobs,
                            uint32_t max_k,
                            uint32_t max_n,
                            bool builtin_active,
                            bool lt_enabled,
                            bool blt_enabled,
                            bool horiz_fusion,
                            std::vector<ScheduledGroup>* out) {
  out->clear();

  auto append_unfused_range = [&](uint32_t job_idx, uint32_t k0, uint32_t tk,
                                  uint32_t n0, uint32_t tn) {
    std::vector<TileRect> rects;
    append_small_tile_rects(hw_m, k0, tk, n0, tn, max_k, max_n, OB_QK8_0,
                            builtin_active, &rects);
    for (const TileRect& r : rects) {
      ScheduledGroup g {};
      g.k0 = r.k0;
      g.tk = r.tk;
      g.hw_k = r.hw_k;
      g.hw_n = r.hw_n;
      g.tn_total = r.tn;
      g.num_subs = 1u;
      g.subs[0] = {job_idx, r.n0, r.tn, 0u};
      out->push_back(g);
    }
  };

  // Horizontal fusion fast path for M=64 batched prefill (lt_64x*)
  if (builtin_active && horiz_fusion && blt_enabled && num_jobs >= 2u &&
      hw_m == 64u && max_k >= 512u && max_n >= 128u && k_dim >= 256u) {
    const uint32_t k_cap = std::min<uint32_t>(max_k, 1536u);
    const uint32_t n_cap = std::max<uint32_t>(128u, (std::min<uint32_t>(max_n, 1536u) / 32u) * 32u);
    std::vector<DimSlice> ks = build_lt_64x_k_slices(k_dim, k_cap);
    for (const DimSlice& ks_i : ks) {
      const uint32_t hk = ceil_lt_64x_k_dim(ks_i.len);
      if (hk == 0u || hk > k_cap) continue;

      std::vector<ScheduledGroup> bins;
      for (uint32_t j = 0u; j < num_jobs; ++j) {
        const uint32_t nj = jobs[j].n_dim;
        for (uint32_t n0 = 0u; n0 < nj; n0 += n_cap) {
          const uint32_t tn = std::min<uint32_t>(n_cap, nj - n0);
          bool placed = false;
          for (ScheduledGroup& b : bins) {
            if (b.num_subs < kMaxFusedSubs && b.tn_total + tn <= n_cap) {
              b.subs[b.num_subs++] = {j, n0, tn, b.tn_total};
              b.tn_total += tn;
              placed = true;
              break;
            }
          }
          if (!placed) {
            ScheduledGroup b {};
            b.k0 = ks_i.off;
            b.tk = ks_i.len;
            b.hw_k = hk;
            b.tn_total = tn;
            b.num_subs = 1u;
            b.subs[0] = {j, n0, tn, 0u};
            bins.push_back(b);
          }
        }
      }
      for (ScheduledGroup& b : bins) {
        const uint32_t hn = std::max<uint32_t>(128u, align_up_u32(b.tn_total, 32u));
        if (b.tn_total >= 128u && is_lt_64x_shape(64u, hk, hn)) {
          b.hw_n = hn;
          out->push_back(b);
        } else {
          for (uint32_t si = 0u; si < b.num_subs; ++si) {
            append_unfused_range(b.subs[si].job_idx, ks_i.off, ks_i.len,
                                 b.subs[si].n0, b.subs[si].tn);
          }
        }
      }
    }
    if (!out->empty()) return;
  }

  // Horizontal fusion fast path for M=1 decode (lt_1x*)
  if (builtin_active && horiz_fusion && lt_enabled && num_jobs >= 2u &&
      hw_m == 1u && max_k >= 512u && max_n >= 512u && k_dim >= 192u) {
    const uint32_t k_cap = std::min<uint32_t>(max_k, 2048u);
    const uint32_t n_cap = std::min<uint32_t>(max_n, 2048u);
    std::vector<DimSlice> ks = build_lt_dim_slices(k_dim, k_cap);
    for (const DimSlice& ks_i : ks) {
      if (ks_i.len < 192u) {
        for (uint32_t j = 0u; j < num_jobs; ++j) {
          append_unfused_range(j, ks_i.off, ks_i.len, 0u, jobs[j].n_dim);
        }
        continue;
      }
      std::vector<ScheduledGroup> bins;
      for (uint32_t j = 0u; j < num_jobs; ++j) {
        std::vector<DimSlice> ns = build_lt_dim_slices(jobs[j].n_dim, n_cap);
        for (const DimSlice& ns_j : ns) {
          bool placed = false;
          for (ScheduledGroup& b : bins) {
            if (b.num_subs < kMaxFusedSubs && b.tn_total + ns_j.len <= n_cap) {
              uint32_t test_hk = 0u, test_hn = 0u;
              if (try_lt_1x_hw_dims(ks_i.len, b.tn_total + ns_j.len,
                                    k_cap, n_cap, &test_hk, &test_hn)) {
                b.subs[b.num_subs++] = {j, ns_j.off, ns_j.len, b.tn_total};
                b.tn_total += ns_j.len;
                b.hw_k = test_hk;
                b.hw_n = test_hn;
                placed = true;
                break;
              }
            }
          }
          if (!placed) {
            ScheduledGroup b {};
            b.k0 = ks_i.off;
            b.tk = ks_i.len;
            b.tn_total = ns_j.len;
            b.num_subs = 1u;
            b.subs[0] = {j, ns_j.off, ns_j.len, 0u};
            uint32_t hk = 0u, hn = 0u;
            if (try_lt_1x_hw_dims(ks_i.len, ns_j.len, k_cap, n_cap, &hk, &hn)) {
              b.hw_k = hk;
              b.hw_n = hn;
            }
            bins.push_back(b);
          }
        }
      }
      for (const ScheduledGroup& b : bins) {
        if (b.hw_k > 0u && b.hw_n > 0u) {
          out->push_back(b);
        } else {
          for (uint32_t si = 0u; si < b.num_subs; ++si) {
            append_unfused_range(b.subs[si].job_idx, ks_i.off, ks_i.len,
                                 b.subs[si].n0, b.subs[si].tn);
          }
        }
      }
    }
    if (!out->empty()) {
      std::stable_sort(out->begin(), out->end(),
                       [](const ScheduledGroup& a, const ScheduledGroup& b) {
                         if (a.k0 != b.k0) return a.k0 < b.k0;
                         if (a.tk != b.tk) return a.tk < b.tk;
                         return a.hw_k < b.hw_k;
                       });
      return;
    }
  }

  // Standard unfused per-job tiling (grouped by shared (k0, tk, hw_k))
  for (uint32_t j = 0u; j < num_jobs; ++j) {
    std::vector<TileRect> jtiles = build_tile_rects(
        hw_m, k_dim, jobs[j].n_dim, max_k, max_n, OB_QK8_0,
        builtin_active, true, lt_enabled, blt_enabled);
    for (const TileRect& tr : jtiles) {
      ScheduledGroup g {};
      g.k0 = tr.k0;
      g.tk = tr.tk;
      g.hw_k = tr.hw_k;
      g.hw_n = tr.hw_n;
      g.tn_total = tr.tn;
      g.num_subs = 1u;
      g.subs[0] = {j, tr.n0, tr.tn, 0u};
      out->push_back(g);
    }
  }
  if (num_jobs > 1u) {
    std::stable_sort(out->begin(), out->end(),
                     [](const ScheduledGroup& a, const ScheduledGroup& b) {
                       if (a.k0 != b.k0) return a.k0 < b.k0;
                       if (a.tk != b.tk) return a.tk < b.tk;
                       return a.hw_k < b.hw_k;
                     });
  }
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

void requant_q8_0_rows(const ob_block_q8_0_t* b_blocks,
                       uint32_t nb_k,
                       uint32_t n0, uint32_t tn,
                       uint32_t blk0, uint32_t blk_cnt,
                       uint32_t tk, uint32_t hw_k,
                       int8_t* out_bq_base,
                       float* out_sb_base,
                       std::vector<float>* scratch_w_row) {
  if (blk_cnt == 1u) {
    for (uint32_t ni = 0; ni < tn; ++ni) {
      const ob_block_q8_0_t* b_row = b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
      out_sb_base[ni] = fp16_to_fp32(b_row[0].d);
      std::memcpy(out_bq_base + static_cast<size_t>(ni) * hw_k, b_row[0].qs, OB_QK8_0);
    }
    return;
  }

  scratch_w_row->resize(tk);
  float* const wrow = scratch_w_row->data();
  for (uint32_t ni = 0; ni < tn; ++ni) {
    const ob_block_q8_0_t* b_row = b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
    float amax = 0.0f;
#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t v_amax = vdupq_n_f32(0.0f);
    for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
      const float32x4_t vd = vdupq_n_f32(fp16_to_fp32(b_row[bi].d));
      const int8_t* qs = b_row[bi].qs;
      float* dst = wrow + static_cast<size_t>(bi) * OB_QK8_0;
      for (uint32_t half = 0u; half < 2u; ++half) {
        int8x16_t q8 = vld1q_s8(qs + half * 16u);
        int16x8_t q16_lo = vmovl_s8(vget_low_s8(q8));
        int16x8_t q16_hi = vmovl_high_s8(q8);
        float32x4_t f0 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo))));
        float32x4_t f1 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(q16_lo)));
        float32x4_t f2 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi))));
        float32x4_t f3 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(q16_hi)));
        vst1q_f32(dst + half * 16u,       f0);
        vst1q_f32(dst + half * 16u + 4u,  f1);
        vst1q_f32(dst + half * 16u + 8u,  f2);
        vst1q_f32(dst + half * 16u + 12u, f3);
        v_amax = vmaxq_f32(v_amax, vmaxq_f32(vmaxq_f32(vabsq_f32(f0), vabsq_f32(f1)),
                                             vmaxq_f32(vabsq_f32(f2), vabsq_f32(f3))));
      }
    }
    amax = vmaxvq_f32(v_amax);
#else
    for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
      const float d = fp16_to_fp32(b_row[bi].d);
      const int8_t* qs = b_row[bi].qs;
      float* dst = wrow + static_cast<size_t>(bi) * OB_QK8_0;
      for (uint32_t j = 0; j < OB_QK8_0; ++j) {
        float v = d * static_cast<float>(qs[j]);
        dst[j] = v;
        float av = std::fabs(v);
        if (av > amax) amax = av;
      }
    }
#endif
    if (amax == 0.0f) {
      out_sb_base[ni] = 0.0f;
      continue;
    }
    const float sb = amax / 127.0f;
    out_sb_base[ni] = sb;
    const float inv_sb = 127.0f / amax;
    int8_t* bq_row = out_bq_base + static_cast<size_t>(ni) * hw_k;
    quant_sym_i8_row(wrow, bq_row, tk, inv_sb);
  }
}

void requant_q4_0_rows(const ob_block_q4_0_t* b_blocks,
                       uint32_t nb_k,
                       uint32_t n0, uint32_t tn,
                       uint32_t blk0, uint32_t blk_cnt,
                       uint32_t tk, uint32_t hw_k,
                       int8_t* out_bq_base,
                       float* out_sb_base,
                       std::vector<float>* scratch_w_row) {
  if (blk_cnt == 1u) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const uint8x16_t v_mask = vdupq_n_u8(0x0Fu);
    const int8x16_t v_eight = vdupq_n_s8(8);
#endif
    for (uint32_t ni = 0; ni < tn; ++ni) {
      const ob_block_q4_0_t* b_row = b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
      out_sb_base[ni] = fp16_to_fp32(b_row[0].d);
      int8_t* bq_row = out_bq_base + static_cast<size_t>(ni) * hw_k;
      const uint8_t* qs = b_row[0].qs;
#if defined(__ARM_NEON) && defined(__aarch64__)
      uint8x16_t raw = vld1q_u8(qs);
      int8x16_t q_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, v_mask)), v_eight);
      int8x16_t q_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v_eight);
      vst1q_s8(bq_row, q_lo);
      vst1q_s8(bq_row + 16u, q_hi);
#else
      for (uint32_t j = 0; j < OB_QK8_0 / 2u; ++j) {
        const uint8_t byte = qs[j];
        bq_row[j] = static_cast<int8_t>(static_cast<int32_t>(byte & 0x0Fu) - 8);
        bq_row[j + OB_QK8_0 / 2u] = static_cast<int8_t>(static_cast<int32_t>(byte >> 4) - 8);
      }
#endif
    }
    return;
  }

  scratch_w_row->resize(tk);
  float* const wrow = scratch_w_row->data();
#if defined(__ARM_NEON) && defined(__aarch64__)
  const uint8x16_t v_mask = vdupq_n_u8(0x0Fu);
  const int8x16_t v_eight = vdupq_n_s8(8);
#endif
  for (uint32_t ni = 0; ni < tn; ++ni) {
    const ob_block_q4_0_t* b_row = b_blocks + static_cast<size_t>(n0 + ni) * nb_k + blk0;
    float amax = 0.0f;
#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t v_amax = vdupq_n_f32(0.0f);
    for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
      const float32x4_t vd = vdupq_n_f32(fp16_to_fp32(b_row[bi].d));
      uint8x16_t raw = vld1q_u8(b_row[bi].qs);
      int8x16_t q_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, v_mask)), v_eight);
      int8x16_t q_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v_eight);
      float* dst = wrow + static_cast<size_t>(bi) * OB_QK8_0;

      // Low 16 nibbles -> dst[0..15]
      {
        int16x8_t s0 = vmovl_s8(vget_low_s8(q_lo));
        int16x8_t s1 = vmovl_high_s8(q_lo);
        float32x4_t f0 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))));
        float32x4_t f1 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(s0)));
        float32x4_t f2 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))));
        float32x4_t f3 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(s1)));
        vst1q_f32(dst,       f0);
        vst1q_f32(dst + 4u,  f1);
        vst1q_f32(dst + 8u,  f2);
        vst1q_f32(dst + 12u, f3);
        v_amax = vmaxq_f32(v_amax, vmaxq_f32(vmaxq_f32(vabsq_f32(f0), vabsq_f32(f1)),
                                             vmaxq_f32(vabsq_f32(f2), vabsq_f32(f3))));
      }
      // High 16 nibbles -> dst[16..31]
      {
        int16x8_t s0 = vmovl_s8(vget_low_s8(q_hi));
        int16x8_t s1 = vmovl_high_s8(q_hi);
        float32x4_t f0 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))));
        float32x4_t f1 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(s0)));
        float32x4_t f2 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))));
        float32x4_t f3 = vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(s1)));
        vst1q_f32(dst + 16u, f0);
        vst1q_f32(dst + 20u, f1);
        vst1q_f32(dst + 24u, f2);
        vst1q_f32(dst + 28u, f3);
        v_amax = vmaxq_f32(v_amax, vmaxq_f32(vmaxq_f32(vabsq_f32(f0), vabsq_f32(f1)),
                                             vmaxq_f32(vabsq_f32(f2), vabsq_f32(f3))));
      }
    }
    amax = vmaxvq_f32(v_amax);
#else
    for (uint32_t bi = 0; bi < blk_cnt; ++bi) {
      const float d = fp16_to_fp32(b_row[bi].d);
      const uint8_t* qs = b_row[bi].qs;
      float* dst = wrow + static_cast<size_t>(bi) * OB_QK8_0;
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
#endif
    if (amax == 0.0f) {
      out_sb_base[ni] = 0.0f;
      continue;
    }
    const float sb = amax / 127.0f;
    out_sb_base[ni] = sb;
    const float inv_sb = 127.0f / amax;
    int8_t* bq_row = out_bq_base + static_cast<size_t>(ni) * hw_k;
    quant_sym_i8_row(wrow, bq_row, tk, inv_sb);
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

void apply_activation_inplace_impl(float* c_mn, size_t count, uint32_t act) {
  if (!c_mn || act == OB_ACT_NONE) return;
  size_t i = 0u;
  if (act == OB_ACT_RELU) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    for (; i + 4u <= count; i += 4u) {
      vst1q_f32(c_mn + i, vmaxq_f32(vld1q_f32(c_mn + i), v_zero));
    }
#endif
    for (; i < count; ++i) {
      if (c_mn[i] < 0.0f) c_mn[i] = 0.0f;
    }
  } else if (act == OB_ACT_RELU6) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_six = vdupq_n_f32(6.0f);
    for (; i + 4u <= count; i += 4u) {
      vst1q_f32(c_mn + i, vminq_f32(vmaxq_f32(vld1q_f32(c_mn + i), v_zero), v_six));
    }
#endif
    for (; i < count; ++i) {
      float v = c_mn[i];
      c_mn[i] = (v < 0.0f) ? 0.0f : ((v > 6.0f) ? 6.0f : v);
    }
  } else if (act == OB_ACT_SILU) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    for (; i + 4u <= count; i += 4u) {
      float32x4_t x = vld1q_f32(c_mn + i);
      float32x4_t denom = vaddq_f32(v_one, fast_exp_f32x4(vnegq_f32(x)));
      vst1q_f32(c_mn + i, vdivq_f32(x, denom));
    }
#endif
    for (; i < count; ++i) {
      float x = c_mn[i];
      c_mn[i] = x / (1.0f + std::exp(-x));
    }
  } else if (act == OB_ACT_GELU) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
#if defined(__ARM_NEON) && defined(__aarch64__)
    // Exact algebraic identity: 0.5 * x * (1 + tanh(u)) == x / (1 + exp(-2 * u))
    const float32x4_t v_neg2_k = vdupq_n_f32(-2.0f * kSqrt2OverPi);
    const float32x4_t v_coeff = vdupq_n_f32(0.044715f);
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    for (; i + 4u <= count; i += 4u) {
      float32x4_t x = vld1q_f32(c_mn + i);
      float32x4_t x2 = vmulq_f32(x, x);
      float32x4_t inner = vfmaq_f32(x, v_coeff, vmulq_f32(x2, x));
      float32x4_t neg2u = vmulq_f32(v_neg2_k, inner);
      float32x4_t denom = vaddq_f32(v_one, fast_exp_f32x4(neg2u));
      vst1q_f32(c_mn + i, vdivq_f32(x, denom));
    }
#endif
    for (; i < count; ++i) {
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
  uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  float32x4_t v_amax0 = vdupq_n_f32(0.0f);
  float32x4_t v_amax1 = vdupq_n_f32(0.0f);
  float64x2_t v_sq0 = vdupq_n_f64(0.0);
  float64x2_t v_sq1 = vdupq_n_f64(0.0);
  float64x2_t v_sq2 = vdupq_n_f64(0.0);
  float64x2_t v_sq3 = vdupq_n_f64(0.0);
  for (; ki + 8u <= tk; ki += 8u) {
    float32x4_t v0 = vld1q_f32(a_row + ki);
    float32x4_t v1 = vld1q_f32(a_row + ki + 4u);
    v_amax0 = vmaxq_f32(v_amax0, vabsq_f32(v0));
    v_amax1 = vmaxq_f32(v_amax1, vabsq_f32(v1));
    float64x2_t d0 = vcvt_f64_f32(vget_low_f32(v0));
    float64x2_t d1 = vcvt_high_f64_f32(v0);
    float64x2_t d2 = vcvt_f64_f32(vget_low_f32(v1));
    float64x2_t d3 = vcvt_high_f64_f32(v1);
    v_sq0 = vfmaq_f64(v_sq0, d0, d0);
    v_sq1 = vfmaq_f64(v_sq1, d1, d1);
    v_sq2 = vfmaq_f64(v_sq2, d2, d2);
    v_sq3 = vfmaq_f64(v_sq3, d3, d3);
  }
  amax = vmaxvq_f32(vmaxq_f32(v_amax0, v_amax1));
  sq_sum = vaddvq_f64(vaddq_f64(vaddq_f64(v_sq0, v_sq1), vaddq_f64(v_sq2, v_sq3)));
#endif
  for (; ki < tk; ++ki) {
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

  // LLM.int8()-style sparse outlier peel: single-pass top-9 extraction (replaces 9 scalar passes
  // with 1 NEON-filtered pass while remaining 100% bit-exact including smallest-index tie breaks).
  constexpr uint32_t kMaxOutliers = 8u;
  constexpr uint32_t kTopK = kMaxOutliers + 1u;
  constexpr double kOutlierSigma = 8.0;
  uint32_t peeled_idx[kMaxOutliers]{};
  uint32_t n_peeled = 0u;

  if (out_outliers && tk >= 128u) {
    double rms = std::sqrt(sq_sum / static_cast<double>(tk));
    if (static_cast<double>(amax) > kOutlierSigma * std::max(rms, 1e-12)) {
      struct Cand {
        float av;
        uint32_t idx;
      };
      Cand top[kTopK];
      for (uint32_t t = 0u; t < kTopK; ++t) top[t] = {-1.0f, 0u};

      auto insert_cand = [&](float av, uint32_t idx) {
        uint32_t pos = kTopK - 1u;
        while (pos > 0u && av > top[pos - 1u].av) {
          top[pos] = top[pos - 1u];
          --pos;
        }
        top[pos] = {av, idx};
      };

      uint32_t oi = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
      float32x4_t v_cut = vdupq_n_f32(top[kTopK - 1u].av);
      for (; oi + 4u <= tk; oi += 4u) {
        float32x4_t av4 = vabsq_f32(vld1q_f32(a_row + oi));
        if (vmaxvq_u32(vcgtq_f32(av4, v_cut)) != 0u) {
          float vals[4];
          vst1q_f32(vals, av4);
          for (uint32_t j = 0u; j < 4u; ++j) {
            if (vals[j] > top[kTopK - 1u].av) {
              insert_cand(vals[j], oi + j);
            }
          }
          v_cut = vdupq_n_f32(top[kTopK - 1u].av);
        }
      }
#endif
      for (; oi < tk; ++oi) {
        float av = std::fabs(a_row[oi]);
        if (av > top[kTopK - 1u].av) {
          insert_cand(av, oi);
        }
      }

      double rem_sq = sq_sum;
      for (uint32_t iter = 0u; iter < kMaxOutliers; ++iter) {
        if (top[iter].av <= 0.0f) break;
        double denom = static_cast<double>(tk > n_peeled + 1u ? (tk - n_peeled - 1u) : 1u);
        double next_sq = std::max(
            0.0, rem_sq - static_cast<double>(top[iter].av) * static_cast<double>(top[iter].av));
        double rem_rms = std::sqrt(next_sq / denom);
        if (static_cast<double>(top[iter].av) <= kOutlierSigma * std::max(rem_rms, 1e-12)) {
          break;
        }
        peeled_idx[n_peeled++] = top[iter].idx;
        out_outliers->push_back({top[iter].idx, a_row[top[iter].idx]});
        rem_sq = next_sq;
      }
      if (n_peeled > 0u) {
        amax = (top[n_peeled].av > 0.0f) ? top[n_peeled].av : 0.0f;
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
  quant_sym_i8_row(a_row, aq_row, tk, inv_sa);
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
    uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
    uint32x4_t v_sq = vdupq_n_u32(0u);
    for (; ki + 16u <= tk; ki += 16u) {
      int8x16_t v = vld1q_s8(row + ki);
      int16x8_t lo = vmull_s8(vget_low_s8(v), vget_low_s8(v));
      int16x8_t hi = vmull_high_s8(v, v);
      v_sq = vpadalq_u16(v_sq, vreinterpretq_u16_s16(lo));
      v_sq = vpadalq_u16(v_sq, vreinterpretq_u16_s16(hi));
    }
    sq = static_cast<uint64_t>(vaddvq_u32(v_sq));
#endif
    for (; ki < tk; ++ki) {
      int32_t v = static_cast<int32_t>(row[ki]);
      sq += static_cast<uint64_t>(v * v);
    }
    double rms = std::sqrt(static_cast<double>(sq) / static_cast<double>(tk));
    if (rms > max_rms) max_rms = rms;
  }
  if (max_rms <= 0.0) return 0u;
  double ratio = (max_rms * 1.35) / 73.3;
  uint32_t shift = 0u;
  while (ratio <= 0.5 && shift < 4u) {
    ratio *= 2.0;
    shift += 1u;
  }
  return shift;
}

// Exact check whether any active int8 output byte equals +127 (0x7F) or -128 (0x80).
// Algebraic identity: uint8(x - 0x7F) <= 1 iff x == 0x7F (+127) or x == 0x80 (-128).
bool tile_output_is_saturated_impl(const int8_t* out_data,
                                   uint32_t tm, uint32_t tn, uint32_t hw_n) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const uint8x16_t v_sub = vdupq_n_u8(0x7Fu);
  uint8x16_t v_min = vdupq_n_u8(0xFFu);
  for (uint32_t mi = 0; mi < tm; ++mi) {
    const uint8_t* row =
        reinterpret_cast<const uint8_t*>(out_data) + static_cast<size_t>(mi) * hw_n;
    uint32_t ni = 0u;
    for (; ni + 16u <= tn; ni += 16u) {
      v_min = vminq_u8(v_min, vsubq_u8(vld1q_u8(row + ni), v_sub));
    }
    for (; ni < tn; ++ni) {
      int8_t qv = static_cast<int8_t>(row[ni]);
      if (qv == 127 || qv == -128) return true;
    }
  }
  return vminvq_u8(v_min) <= 1u;
#else
  for (uint32_t mi = 0; mi < tm; ++mi) {
    const int8_t* o_row = out_data + static_cast<size_t>(mi) * hw_n;
    for (uint32_t ni = 0; ni < tn; ++ni) {
      int8_t qv = o_row[ni];
      if (qv == 127 || qv == -128) return true;
    }
  }
  return false;
#endif
}

inline void dequant_accum_row_f32_impl(float* c_row,
                                       const int8_t* o_row,
                                       const float* sb_ptr,
                                       uint32_t tn,
                                       float sa_d) {
  uint32_t ni = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t v_sad = vdupq_n_f32(sa_d);
  for (; ni + 16u <= tn; ni += 16u) {
    int8x16_t q8 = vld1q_s8(o_row + ni);
    int16x8_t q16_lo = vmovl_s8(vget_low_s8(q8));
    int16x8_t q16_hi = vmovl_high_s8(q8);

    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo)));
    float32x4_t f1 = vcvtq_f32_s32(vmovl_high_s16(q16_lo));
    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi)));
    float32x4_t f3 = vcvtq_f32_s32(vmovl_high_s16(q16_hi));

    float32x4_t s0 = vmulq_f32(v_sad, vld1q_f32(sb_ptr + ni));
    float32x4_t s1 = vmulq_f32(v_sad, vld1q_f32(sb_ptr + ni + 4u));
    float32x4_t s2 = vmulq_f32(v_sad, vld1q_f32(sb_ptr + ni + 8u));
    float32x4_t s3 = vmulq_f32(v_sad, vld1q_f32(sb_ptr + ni + 12u));

    vst1q_f32(c_row + ni,       vfmaq_f32(vld1q_f32(c_row + ni),       f0, s0));
    vst1q_f32(c_row + ni + 4u,  vfmaq_f32(vld1q_f32(c_row + ni + 4u),  f1, s1));
    vst1q_f32(c_row + ni + 8u,  vfmaq_f32(vld1q_f32(c_row + ni + 8u),  f2, s2));
    vst1q_f32(c_row + ni + 12u, vfmaq_f32(vld1q_f32(c_row + ni + 12u), f3, s3));
  }
#endif
  for (; ni < tn; ++ni) {
    c_row[ni] += static_cast<float>(o_row[ni]) * (sa_d * sb_ptr[ni]);
  }
}

inline void dequant_accum_row_i32_impl(int32_t* c_row,
                                       const int8_t* o_row,
                                       uint32_t tn,
                                       int32_t d_i32) {
  uint32_t ni = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const int32x4_t v_d = vdupq_n_s32(d_i32);
  for (; ni + 16u <= tn; ni += 16u) {
    int8x16_t q8 = vld1q_s8(o_row + ni);
    int16x8_t q16_lo = vmovl_s8(vget_low_s8(q8));
    int16x8_t q16_hi = vmovl_high_s8(q8);
    vst1q_s32(c_row + ni,       vmlaq_s32(vld1q_s32(c_row + ni),       vmovl_s16(vget_low_s16(q16_lo)), v_d));
    vst1q_s32(c_row + ni + 4u,  vmlaq_s32(vld1q_s32(c_row + ni + 4u),  vmovl_high_s16(q16_lo), v_d));
    vst1q_s32(c_row + ni + 8u,  vmlaq_s32(vld1q_s32(c_row + ni + 8u),  vmovl_s16(vget_low_s16(q16_hi)), v_d));
    vst1q_s32(c_row + ni + 12u, vmlaq_s32(vld1q_s32(c_row + ni + 12u), vmovl_high_s16(q16_hi), v_d));
  }
#endif
  for (; ni < tn; ++ni) {
    c_row[ni] += static_cast<int32_t>(o_row[ni]) * d_i32;
  }
}

}  // namespace

size_t compute_weight_cache_budget_bytes(ob_context* ctx) {
  return compute_weight_cache_budget_bytes_impl(ctx);
}

uint32_t quant_scale_D_grid(double raw_d) {
  return quant_scale_D_grid_impl(raw_d);
}

uint32_t compute_tile_scale_D(const int8_t* b_tile, uint32_t tn, uint32_t tk,
                              uint32_t stride_k, int32_t zp_a) {
  return compute_tile_scale_D_impl(b_tile, tn, tk, stride_k, zp_a);
}

void compute_swiglu_stage_dividers(uint32_t m_dim,
                                   float rms_a,
                                   float l2_g_max, float l2_g_mean,
                                   float l2_u_max, float l2_u_mean,
                                   float l2_d_max,
                                   uint32_t* out_d_gate,
                                   uint32_t* out_d_up,
                                   uint32_t* out_d_mul,
                                   uint32_t* out_d_down) {
  compute_swiglu_stage_dividers_impl(m_dim, rms_a, l2_g_max, l2_g_mean, l2_u_max, l2_u_mean, l2_d_max,
                                     out_d_gate, out_d_up, out_d_mul, out_d_down);
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

bool tile_output_is_saturated(const int8_t* out_data, uint32_t tm, uint32_t tn, uint32_t hw_n) {
  return tile_output_is_saturated_impl(out_data, tm, tn, hw_n);
}

void dequant_accum_row_f32(float* c_row, const int8_t* o_row,
                           const float* sb_ptr, uint32_t tn, float sa_d) {
  dequant_accum_row_f32_impl(c_row, o_row, sb_ptr, tn, sa_d);
}

void apply_activation_inplace_f32(float* c_mn, size_t count, uint32_t act) {
  apply_activation_inplace_impl(c_mn, count, act);
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
    if (ctx->gemm_in_bufs_4k[i]) {
      ob_buffer_free(ctx->gemm_in_bufs_4k[i]);
      ctx->gemm_in_bufs_4k[i] = nullptr;
    }
    if (ctx->gemm_out_bufs_4k[i]) {
      ob_buffer_free(ctx->gemm_out_bufs_4k[i]);
      ctx->gemm_out_bufs_4k[i] = nullptr;
    }
  }
  for (CachedWeightTile* t : ctx->weight_cache_lru) {
    if (t->ext2_buf) ob_buffer_free(t->ext2_buf);
    delete t;
  }
  ctx->weight_cache_lru.clear();
  ctx->weight_cache_map.clear();
  ctx->weight_cache_bytes = 0;
  ctx->immutable_weight_regions.clear();
}

}  // namespace ob

extern "C" {

ob_status_t ob_gemm_register_immutable_weights(ob_context_t* ctx,
                                               const void*   base,
                                               size_t        size,
                                               uint64_t      version) {
  if (!ctx || !base || size == 0u) return OB_ERR_INVALID_ARG;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
  const uintptr_t max_ptr = static_cast<uintptr_t>(-1);
  if (size > static_cast<size_t>(max_ptr - begin)) {
    return OB_ERR_INVALID_ARG;
  }
  const uintptr_t end = begin + static_cast<uintptr_t>(size);

  std::lock_guard<std::mutex> lk(ctx->gemm_mu);
  auto& regions = ctx->immutable_weight_regions;
  regions.erase(
      std::remove_if(regions.begin(), regions.end(),
                     [&](const ob::ImmutableWeightRegion& r) {
                       return r.begin < end && begin < r.end;
                     }),
      regions.end());
  regions.push_back(ob::ImmutableWeightRegion{begin, end, version});
  return OB_OK;
}

ob_status_t ob_gemm_unregister_immutable_weights(ob_context_t* ctx,
                                                 const void*   base,
                                                 size_t        size) {
  if (!ctx || (!base && size != 0u)) return OB_ERR_INVALID_ARG;
  if (!base) return OB_OK;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
  const uintptr_t max_ptr = static_cast<uintptr_t>(-1);
  if (size > static_cast<size_t>(max_ptr - begin)) {
    return OB_ERR_INVALID_ARG;
  }
  const uintptr_t end = (size > 0u) ? (begin + static_cast<uintptr_t>(size)) : begin;

  std::lock_guard<std::mutex> lk(ctx->gemm_mu);
  auto& regions = ctx->immutable_weight_regions;
  regions.erase(
      std::remove_if(regions.begin(), regions.end(),
                     [&](const ob::ImmutableWeightRegion& r) {
                       if (size == 0u) return r.begin == begin;
                       return r.begin < end && begin < r.end;
                     }),
      regions.end());
  return OB_OK;
}

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
  const bool prof_on = ctx->profile_enabled;
  const bool strict_hash = ctx->strict_weight_hash;
  const bool lt_enabled = ctx->large_tiles_enabled;
  const bool blt_enabled = ctx->batched_large_tiles_enabled;
  const uint32_t param_swap_slots = ctx->param_swap_slots;
  const uint32_t ring_depth =
      std::max<uint32_t>(1u, std::min<uint32_t>(ctx->credit_size, OB_MAX_CREDITS));

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
    ob::CachedWeightTile* w_tile = nullptr;
    ob_buffer* in_buf = nullptr;
    ob_buffer* out_buf = nullptr;
    uint32_t slot = 0u;
    uint32_t in_slot = 0u;
    uint32_t m0 = 0u, tm = 0u, hw_m = 0u;
    uint32_t n0 = 0u, tn = 0u;
    uint32_t hw_n = 0u;
    uint32_t scale_d = 1u;
  };
  std::array<PendingInt8Tile, OB_MAX_CREDITS> ring{};
  std::array<uint32_t, OB_MAX_CREDITS> in_refcnt{};
  uint32_t ring_head = 0u;
  uint32_t ring_count = 0u;

  auto drain_one = [&]() -> ob_status_t {
    PendingInt8Tile& p = ring[ring_head];
    ob_invoke_stats_t istats {};
    const uint64_t tw0 = prof_on ? ob::monotonic_now_ns() : 0u;
    ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, &istats);
    if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - tw0);
    ob_status_t rst = OB_OK;
    if (wst == OB_OK) {
      total_hw_us += istats.hw_execution_us;
      if (!ctx->mock_mode) {
        const uint64_t td0 = prof_on ? ob::monotonic_now_ns() : 0u;
        rst = ob::buffer_sync_begin_read(p.out_buf);
        if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td0);
        if (rst == OB_OK) {
          const uint64_t tq0 = prof_on ? ob::monotonic_now_ns() : 0u;
          const int8_t* raw_out =
              static_cast<const int8_t*>(ob_buffer_data(p.out_buf));
          const int32_t d_i32 = static_cast<int32_t>(p.scale_d);
          for (uint32_t mi = 0; mi < p.tm; ++mi) {
            int32_t* c_row = c_mn + static_cast<size_t>(p.m0 + mi) * n_dim + p.n0;
            const int8_t* o_row = raw_out + static_cast<size_t>(mi) * p.hw_n;
            ob::dequant_accum_row_i32_impl(c_row, o_row, p.tn, d_i32);
          }
          if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tq0);
          const uint64_t td1 = prof_on ? ob::monotonic_now_ns() : 0u;
          rst = ob::buffer_sync_end_read(p.out_buf);
          if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td1);
        }
      }
    } else {
      ob::detach_timed_out_buffers_locked(ctx, p.handle, p.w_tile, p.in_buf, p.out_buf);
    }
    if (p.w_tile) {
      if (p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
      if (!p.w_tile->in_cache && p.w_tile->in_use == 0u) {
        if (p.w_tile->ext2_buf) ob_buffer_free(p.w_tile->ext2_buf);
        delete p.w_tile;
      }
    }
    if (p.in_buf) ob_buffer_free(p.in_buf);
    if (p.out_buf) ob_buffer_free(p.out_buf);
    if (in_refcnt[p.in_slot] > 0u) in_refcnt[p.in_slot] -= 1u;
    p = PendingInt8Tile{};
    ring_head = (ring_head + 1u) % ring_depth;
    ring_count -= 1u;
    return (wst != OB_OK) ? wst : rst;
  };

  auto drain_all_cleanup = [&]() {
    while (ring_count > 0u) {
      PendingInt8Tile& p = ring[ring_head];
      ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, nullptr);
      if (wst != OB_OK) {
        ob::detach_timed_out_buffers_locked(ctx, p.handle, p.w_tile, p.in_buf, p.out_buf);
      }
      if (p.w_tile) {
        if (p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
        if (!p.w_tile->in_cache && p.w_tile->in_use == 0u) {
          if (p.w_tile->ext2_buf) ob_buffer_free(p.w_tile->ext2_buf);
          delete p.w_tile;
        }
      }
      if (p.in_buf) ob_buffer_free(p.in_buf);
      if (p.out_buf) ob_buffer_free(p.out_buf);
      if (in_refcnt[p.in_slot] > 0u) in_refcnt[p.in_slot] -= 1u;
      p = PendingInt8Tile{};
      ring_head = (ring_head + 1u) % ring_depth;
      ring_count -= 1u;
    }
  };

  std::vector<int8_t>& a_tile = ctx->gemm_aq_scratch;
  std::vector<int8_t>& b_tile = ctx->gemm_bq_scratch;
  std::vector<uint8_t> packed_tile;

  const bool allow_m_pad = (zp_a == 0 && zp_b == 0);
  const bool allow_lt_64x = allow_m_pad && k_dim >= 256u && n_dim >= 128u &&
                            max_k >= 512u && max_n >= 128u;
  std::vector<ob::MSlice> m_slices = ob::build_m_slices(
      m_dim, max_m, builtin_active, allow_m_pad, allow_lt_64x, blt_enabled);

  for (const ob::MSlice& ms : m_slices) {
    const uint32_t m0 = ms.m0;
    const uint32_t tm = ms.tm;
    const uint32_t hw_m = ms.hw_m;
    std::vector<ob::TileRect> tiles = ob::build_tile_rects(
        hw_m, k_dim, n_dim, max_k, max_n, 1u, builtin_active, allow_m_pad,
        lt_enabled, blt_enabled);

    uint32_t cached_k0 = 0xFFFFFFFFu;
    uint32_t cached_tk = 0u;
    uint32_t cached_hw_k = 0u;
    uint32_t active_in_slot = OB_MAX_CREDITS;
    ob_buffer* active_in_buf = nullptr;

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
        const bool needs_pad = (hw_m != tm || hw_k != tk || hw_n != tn);
        const bool can_hw = !(builtin_active && (zp_b != 0 || (needs_pad && zp_a != 0)));

        if (can_hw) {
          if (cached_k0 != k0 || cached_tk != tk || cached_hw_k != hw_k) {
            const uint64_t tq0 = prof_on ? ob::monotonic_now_ns() : 0u;
            a_tile.assign(static_cast<size_t>(hw_m) * hw_k, 0);
            for (uint32_t mi = 0; mi < tm; ++mi) {
              std::memcpy(a_tile.data() + static_cast<size_t>(mi) * hw_k,
                          a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0, tk);
            }
            if (prof_on) ctx->prof.quant_ns += (ob::monotonic_now_ns() - tq0);
            cached_k0 = k0;
            cached_tk = tk;
            cached_hw_k = hw_k;
            active_in_slot = OB_MAX_CREDITS;
            active_in_buf = nullptr;
          }

          ob::CachedWeightTile* w_tile = nullptr;
          uint32_t scale_d = 1u;
          if (builtin_active) {
            const bool is_lt64_tile = ob::is_lt_64x_shape(hw_m, hw_k, hw_n);
            const uint32_t pack_fmt = is_lt64_tile ? 1u : 0u;
            const uint64_t th0 = prof_on ? ob::monotonic_now_ns() : 0u;
            ob::WeightHash128 wh_st{};
            uint32_t whdr[6] = {tn, tk, hw_k, hw_n, static_cast<uint32_t>(zp_a), pack_fmt};
            ob::hash128_update(&wh_st, whdr, sizeof(whdr));
            ob::hash_weight_tile_for_ctx(
                ctx, &wh_st,
                reinterpret_cast<const uint8_t*>(b_nk + static_cast<size_t>(n0) * k_dim + k0),
                static_cast<size_t>(k_dim), tn, static_cast<size_t>(tk), strict_hash);
            const ob::WeightTileKey wh = ob::finalize_weight_key(wh_st);
            w_tile = ob::weight_cache_find_locked(ctx, wh);
            if (prof_on) ctx->prof.hash_ns += (ob::monotonic_now_ns() - th0);

            if (!w_tile) {
              const uint64_t tp0 = prof_on ? ob::monotonic_now_ns() : 0u;
              b_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
              for (uint32_t ni = 0; ni < tn; ++ni) {
                std::memcpy(b_tile.data() + static_cast<size_t>(ni) * hw_k,
                            b_nk + static_cast<size_t>(n0 + ni) * k_dim + k0, tk);
              }
              scale_d = ob::compute_tile_scale_D(b_tile.data(), tn, tk, hw_k, zp_a);
              std::vector<uint8_t> ext2 = is_lt64_tile
                  ? ob::pack_ext2_for_lt_64x(hw_n, hw_k, b_tile.data(), nullptr)
                  : ob::pack_ext2_for_tile(hw_n, hw_k, b_tile.data(), nullptr);
              if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);

              bool cacheable = false;
              ob_status_t pst = ob::weight_cache_prepare_space_locked(
                  ctx, ext2.size(), ring_count, drain_one, &cacheable);
              if (pst != OB_OK) {
                drain_all_cleanup();
                return pst;
              }

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
              if (cacheable) {
                ob::weight_cache_insert_locked(ctx, w_tile);
              } else {
                w_tile->in_cache = false;
              }
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

          auto discard_unsubmitted_wtile = [&]() {
            if (w_tile && !w_tile->in_cache && w_tile->in_use == 0u) {
              if (w_tile->ext2_buf) ob_buffer_free(w_tile->ext2_buf);
              delete w_tile;
              w_tile = nullptr;
            }
          };

          if (ring_count >= ring_depth) {
            ob_status_t dst = drain_one();
            if (dst != OB_OK) {
              discard_unsubmitted_wtile();
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
              discard_unsubmitted_wtile();
              drain_all_cleanup();
              return pst;
            }
            desc.packed_params = packed_tile.data();
            desc.packed_params_bytes = packed_tile.size();
            desc.cache_key = ob::hash_tile_key(
                hw_m, hw_k, hw_n, zp_a, zp_b, scale_d, b_tile.data(), b_tile.size());
          }

          const uint64_t ts0 = prof_on ? ob::monotonic_now_ns() : 0u;
          ob::CachedGraph* entry = nullptr;
          bool was_hit = false;
          ob_status_t gst = ob::acquire_tile_entry(
              ctx, emitter, emitter_ud, desc, &entry, &was_hit);
          if (gst == OB_OK) {
            used_emitter = true;
            if (was_hit) cache_hits += 1u;

            size_t in_alloc = std::max<size_t>(
                a_tile.size(),
                !entry->parsed.input_sizes.empty()
                    ? static_cast<size_t>(entry->parsed.input_sizes[0]) : 1u);
            size_t out_alloc = std::max<size_t>(
                static_cast<size_t>(hw_m) * ob::align_up_u32(hw_n, 4u),
                !entry->parsed.output_sizes.empty()
                    ? static_cast<size_t>(entry->parsed.output_sizes[0]) : 1u);

            ob_buffer* out_buf = nullptr;
            ob_status_t bst = ob::ensure_pipeline_buf_tiered(
                ctx, &ctx->gemm_out_bufs_4k[slot], &ctx->gemm_out_bufs[slot], out_alloc, &out_buf);
            if (bst == OB_OK) {
              if (active_in_slot == OB_MAX_CREDITS ||
                  !active_in_buf ||
                  active_in_buf->slab.alloc_size < in_alloc) {
                uint32_t chosen = 0u;
                for (uint32_t c = 0u; c < ring_depth; ++c) {
                  if (in_refcnt[c] == 0u) { chosen = c; break; }
                }
                ob_buffer* in_buf = nullptr;
                bst = ob::ensure_pipeline_buf_tiered(
                    ctx, &ctx->gemm_in_bufs_4k[chosen], &ctx->gemm_in_bufs[chosen], in_alloc, &in_buf);
                if (bst == OB_OK) {
                  const uint64_t td0 = prof_on ? ob::monotonic_now_ns() : 0u;
                  bst = ob_buffer_write(in_buf, a_tile.data(), a_tile.size(), 0u);
                  if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td0);
                }
                if (bst == OB_OK) {
                  active_in_slot = chosen;
                  active_in_buf = in_buf;
                }
              }
            }

            ob_exec_handle_t exec_h = 0u;
            if (bst == OB_OK) {
              bst = ob::device_submit_execute_async(
                  ctx, entry,
                  &active_in_buf, 1u,
                  &out_buf, 1u,
                  &exec_h,
                  w_tile ? w_tile->ext2_buf : nullptr);
            }
            if (prof_on) ctx->prof.submit_ns += (ob::monotonic_now_ns() - ts0);
            if (bst != OB_OK) {
              ob::release_tile_entry(ctx, entry);
              discard_unsubmitted_wtile();
              drain_all_cleanup();
              return bst;
            }

            PendingInt8Tile& p = ring[slot];
            p.handle = exec_h;
            p.w_tile = w_tile;
            if (w_tile) w_tile->in_use += 1u;
            p.in_buf = active_in_buf;
            if (p.in_buf) ob::buffer_retain(p.in_buf);
            p.out_buf = out_buf;
            if (p.out_buf) ob::buffer_retain(p.out_buf);
            p.slot = slot;
            p.in_slot = active_in_slot;
            in_refcnt[active_in_slot] += 1u;
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
          } else {
            discard_unsubmitted_wtile();
            if (gst != OB_ERR_UNSUPPORTED) {
              drain_all_cleanup();
              return gst;
            }
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

  while (ring_count > 0u) {
    ob_status_t dst = drain_one();
    if (dst != OB_OK) {
      drain_all_cleanup();
      return dst;
    }
  }

  const uint64_t wall_ns = ob::monotonic_now_ns() - t0_ns;
  ctx->prof.calls += 1u;
  ctx->prof.tiles_dispatched += tiles_dispatched;
  ctx->prof.hw_us += total_hw_us;
  ctx->prof.total_wall_ns += wall_ns;

  if (out_stats) {
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->hw_execution_us = total_hw_us;
    out_stats->wall_time_us = wall_ns / 1000ULL;
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
  const bool prof_on = ctx->profile_enabled;
  const bool strict_hash = ctx->strict_weight_hash;
  const bool lt_enabled = ctx->large_tiles_enabled;
  const bool blt_enabled = ctx->batched_large_tiles_enabled;
  const bool horiz_fusion = ctx->horiz_fusion_enabled;
  const uint32_t param_swap_slots = ctx->param_swap_slots;
  const uint32_t ring_depth =
      std::max<uint32_t>(1u, std::min<uint32_t>(ctx->credit_size, OB_MAX_CREDITS));

  for (uint32_t j = 0; j < num_jobs; ++j) {
    std::memset(jobs[j].c_mn, 0, static_cast<size_t>(m_dim) * jobs[j].n_dim * sizeof(float));
  }

  uint32_t tiles_dispatched = 0u;
  uint32_t cache_hits = 0u;
  uint64_t total_hw_us = 0u;
  bool used_emitter = false;
  bool used_cpu_fallback = false;

  struct PendingF32Tile {
    ob_exec_handle_t handle = 0u;
    ob::CachedWeightTile* w_tile = nullptr;
    ob_buffer* in_buf = nullptr;
    ob_buffer* out_buf = nullptr;
    uint32_t slot = 0u;
    uint32_t in_slot = 0u;
    uint32_t m0 = 0u, tm = 0u, hw_m = 0u;
    uint32_t hw_k = 0u, hw_n = 0u;
    uint32_t tn_total = 0u;
    uint32_t num_subs = 0u;
    std::array<ob::FusedSubSlice, ob::kMaxFusedSubs> subs{};
    uint32_t scale_d_u32 = 1u;
    std::array<float, 64> s_a_inline{};
    std::vector<float> s_a_overflow;
    std::vector<float> s_b;
  };
  std::array<PendingF32Tile, OB_MAX_CREDITS> ring{};
  std::array<uint32_t, OB_MAX_CREDITS> in_refcnt{};
  uint32_t ring_head = 0u;
  uint32_t ring_count = 0u;

  auto drain_one = [&]() -> ob_status_t {
    PendingF32Tile& p = ring[ring_head];
    ob_invoke_stats_t istats {};
    const uint64_t tw0 = prof_on ? ob::monotonic_now_ns() : 0u;
    ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, &istats);
    if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - tw0);
    ob_status_t rst = OB_OK;
    if (wst == OB_OK) {
      total_hw_us += istats.hw_execution_us;
      if (!ctx->mock_mode) {
        uint32_t eff_d = p.scale_d_u32;
        const uint64_t td0 = prof_on ? ob::monotonic_now_ns() : 0u;
        rst = ob::buffer_sync_begin_read(p.out_buf);
        if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td0);

        if (rst == OB_OK) {
          const int8_t* raw_out =
              static_cast<const int8_t*>(ob_buffer_data(p.out_buf));
          bool read_active = true;

          // Saturation-proof on-TPU retry: direct NEON check on mapped DMA-BUF (zero memcpy!).
          if (builtin_active && p.w_tile != nullptr) {
            static constexpr double kRetryMults[5] = {1.25, 1.5, 2.0, 4.0, 4.0};
            for (uint32_t retry = 0u; retry < 5u && eff_d < (1u << 24); ++retry) {
              const uint64_t tsat0 = prof_on ? ob::monotonic_now_ns() : 0u;
              const bool sat = ob::tile_output_is_saturated_impl(raw_out, p.tm, p.tn_total, p.hw_n);
              if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tsat0);
              if (!sat) break;

              const uint64_t tde0 = prof_on ? ob::monotonic_now_ns() : 0u;
              ob::buffer_sync_end_read(p.out_buf);
              if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - tde0);
              read_active = false;

              ctx->prof.sat_retries += 1u;
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

              const uint64_t trs0 = prof_on ? ob::monotonic_now_ns() : 0u;
              ob::CachedGraph* rentry = nullptr;
              bool rhit = false;
              rst = ob::acquire_tile_entry(ctx, emitter, emitter_ud, rdesc, &rentry, &rhit);
              if (rst != OB_OK) break;
              if (rhit) cache_hits += 1u;
              tiles_dispatched += 1u;
              ob_exec_handle_t rh = 0u;
              rst = ob::device_submit_execute_async(
                  ctx, rentry,
                  &p.in_buf, 1u,
                  &p.out_buf, 1u,
                  &rh, p.w_tile->ext2_buf);
              if (prof_on) ctx->prof.submit_ns += (ob::monotonic_now_ns() - trs0);
              if (rst != OB_OK) {
                ob::release_tile_entry(ctx, rentry);
                break;
              }
              ob_invoke_stats_t rstats {};
              const uint64_t trw0 = prof_on ? ob::monotonic_now_ns() : 0u;
              rst = ob::device_wait_handle(ctx, rh, ctx->cfg.timeout_ms, &rstats);
              if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - trw0);
              if (rst != OB_OK) {
                ob::detach_timed_out_buffers_locked(ctx, rh, p.w_tile, p.in_buf, p.out_buf);
                break;
              }
              total_hw_us += rstats.hw_execution_us;

              const uint64_t tdr0 = prof_on ? ob::monotonic_now_ns() : 0u;
              rst = ob::buffer_sync_begin_read(p.out_buf);
              if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - tdr0);
              if (rst != OB_OK) break;
              raw_out = static_cast<const int8_t*>(ob_buffer_data(p.out_buf));
              read_active = true;
            }
          }

          if (rst == OB_OK && read_active) {
            const uint64_t tq0 = prof_on ? ob::monotonic_now_ns() : 0u;
            const float eff_scale_d = static_cast<float>(eff_d);
            const float* sa_ptr =
                (p.tm <= p.s_a_inline.size()) ? p.s_a_inline.data() : p.s_a_overflow.data();
            const float* sb_base = p.w_tile ? p.w_tile->row_s_b.data() : p.s_b.data();

            // Zero-copy NEON dequant-accum straight from mapped DMA-BUF into each sub-slice output!
            for (uint32_t si = 0u; si < p.num_subs; ++si) {
              const ob::FusedSubSlice& sub = p.subs[si];
              const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
              const float* sb_ptr = sb_base + sub.out_off;
              for (uint32_t mi = 0; mi < p.tm; ++mi) {
                const float sa_d = sa_ptr[mi] * eff_scale_d;
                float* c_row = job.c_mn + static_cast<size_t>(p.m0 + mi) * job.n_dim + sub.n0;
                const int8_t* o_row = raw_out + static_cast<size_t>(mi) * p.hw_n + sub.out_off;
                ob::dequant_accum_row_f32_impl(c_row, o_row, sb_ptr, sub.tn, sa_d);
              }
            }
            if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tq0);
          }
          if (read_active) {
            const uint64_t td1 = prof_on ? ob::monotonic_now_ns() : 0u;
            ob_status_t est = ob::buffer_sync_end_read(p.out_buf);
            if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td1);
            if (rst == OB_OK) rst = est;
          }
        }
      }
    } else {
      ob::detach_timed_out_buffers_locked(ctx, p.handle, p.w_tile, p.in_buf, p.out_buf);
    }
    if (p.w_tile) {
      if (p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
      if (!p.w_tile->in_cache && p.w_tile->in_use == 0u) {
        if (p.w_tile->ext2_buf) ob_buffer_free(p.w_tile->ext2_buf);
        delete p.w_tile;
      }
    }
    if (p.in_buf) ob_buffer_free(p.in_buf);
    if (p.out_buf) ob_buffer_free(p.out_buf);
    if (in_refcnt[p.in_slot] > 0u) in_refcnt[p.in_slot] -= 1u;
    p.handle = 0u;
    p.w_tile = nullptr;
    p.in_buf = nullptr;
    p.out_buf = nullptr;
    p.num_subs = 0u;
    ring_head = (ring_head + 1u) % ring_depth;
    ring_count -= 1u;
    return (wst != OB_OK) ? wst : rst;
  };

  auto drain_all_cleanup = [&]() {
    while (ring_count > 0u) {
      PendingF32Tile& p = ring[ring_head];
      ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, nullptr);
      if (wst != OB_OK) {
        ob::detach_timed_out_buffers_locked(ctx, p.handle, p.w_tile, p.in_buf, p.out_buf);
      }
      if (p.w_tile) {
        if (p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
        if (!p.w_tile->in_cache && p.w_tile->in_use == 0u) {
          if (p.w_tile->ext2_buf) ob_buffer_free(p.w_tile->ext2_buf);
          delete p.w_tile;
        }
      }
      if (p.in_buf) ob_buffer_free(p.in_buf);
      if (p.out_buf) ob_buffer_free(p.out_buf);
      if (in_refcnt[p.in_slot] > 0u) in_refcnt[p.in_slot] -= 1u;
      p.handle = 0u;
      p.w_tile = nullptr;
      p.in_buf = nullptr;
      p.out_buf = nullptr;
      p.num_subs = 0u;
      ring_head = (ring_head + 1u) % ring_depth;
      ring_count -= 1u;
    }
  };

  const uint32_t nb_k = k_dim / OB_QK8_0;
  std::vector<int8_t>& a_q_tile = ctx->gemm_aq_scratch;
  std::vector<int8_t>& b_q_tile = ctx->gemm_bq_scratch;
  std::vector<float>& tmp_w_row = ctx->gemm_wrow_scratch;
  std::vector<float>& row_s_a = ctx->gemm_sa_scratch;
  std::vector<std::vector<std::pair<uint32_t, float>>>& row_outliers = ctx->gemm_outliers_scratch;
  std::vector<float> row_s_b;
  std::vector<uint8_t> packed_tile;
  std::vector<ob::ScheduledGroup> groups;

  bool allow_lt_64x = (k_dim >= 256u && max_k >= 512u && max_n >= 128u);
  if (allow_lt_64x) {
    bool all_ge_128 = true;
    uint64_t total_n = 0ULL;
    for (uint32_t j = 0u; j < num_jobs; ++j) {
      if (jobs[j].n_dim < 128u) all_ge_128 = false;
      total_n += jobs[j].n_dim;
    }
    if (!all_ge_128 && !(horiz_fusion && num_jobs >= 2u && total_n >= 128ULL)) {
      allow_lt_64x = false;
    }
  }

  std::vector<ob::MSlice> m_slices = ob::build_m_slices(
      m_dim, max_m, builtin_active, true, allow_lt_64x, blt_enabled);

  for (const ob::MSlice& ms : m_slices) {
    const uint32_t m0 = ms.m0;
    const uint32_t tm = ms.tm;
    const uint32_t hw_m = ms.hw_m;

    ob::build_scheduled_groups(hw_m, k_dim, jobs, num_jobs, max_k, max_n,
                               builtin_active, lt_enabled, blt_enabled,
                               horiz_fusion, &groups);

    row_s_a.resize(tm);
    row_outliers.resize(tm);
    uint32_t cached_a_k0 = 0xFFFFFFFFu;
    uint32_t cached_a_tk = 0u;
    uint32_t cached_aq_hw_k = 0u;
    uint32_t cached_a_rms_shift = 0u;
    uint32_t active_in_slot = OB_MAX_CREDITS;
    ob_buffer* active_in_buf = nullptr;

    // Dispatch one ScheduledGroup (1-sub single tile or multi-sub horizontally fused tile).
    // Returns OB_OK on success. If a multi-sub fused group's D spread exceeds 1.5x,
    // caches `rejected_fusion=true` and recursively dispatches each sub-slice unfused.
    auto dispatch_group = [&](const ob::ScheduledGroup& grp, auto& self_ref) -> ob_status_t {
      const uint32_t k0 = grp.k0;
      const uint32_t tk = grp.tk;
      const uint32_t hw_k = grp.hw_k;
      const uint32_t hw_n = grp.hw_n;
      const uint32_t tn_total = grp.tn_total;
      const uint32_t num_subs = grp.num_subs;
      const uint32_t blk0 = k0 / OB_QK8_0;
      const uint32_t blk_cnt = tk / OB_QK8_0;

      // Helper to split a rejected multi-sub horizontal fusion cleanly into 1-sub groups.
      auto fallback_split_subs = [&]() -> ob_status_t {
        for (uint32_t si = 0u; si < num_subs; ++si) {
          const ob::FusedSubSlice& sub = grp.subs[si];
          std::vector<ob::TileRect> rects;
          const uint32_t k_cap = (hw_m == 64u) ? std::min<uint32_t>(max_k, 1536u)
                                               : std::min<uint32_t>(max_k, 2048u);
          const uint32_t n_cap = (hw_m == 64u)
              ? std::max<uint32_t>(128u, (std::min<uint32_t>(max_n, 1536u) / 32u) * 32u)
              : std::min<uint32_t>(max_n, 2048u);
          uint32_t sh_k = 0u, sh_n = 0u;
          if (hw_m == 64u && sub.tn >= 128u) {
            sh_k = ob::ceil_lt_64x_k_dim(tk);
            sh_n = std::max<uint32_t>(128u, ob::align_up_u32(sub.tn, 32u));
            if (sh_k > 0u && sh_k <= k_cap && ob::is_lt_64x_shape(64u, sh_k, sh_n)) {
              rects.push_back({k0, tk, sub.n0, sub.tn, sh_k, sh_n});
            }
          } else if (hw_m == 1u && ob::try_lt_1x_hw_dims(tk, sub.tn, k_cap, n_cap, &sh_k, &sh_n)) {
            rects.push_back({k0, tk, sub.n0, sub.tn, sh_k, sh_n});
          }
          if (rects.empty()) {
            ob::append_small_tile_rects(hw_m, k0, tk, sub.n0, sub.tn, max_k, max_n,
                                        OB_QK8_0, builtin_active, &rects);
          }
          for (const ob::TileRect& r : rects) {
            ob::ScheduledGroup sg {};
            sg.k0 = r.k0;
            sg.tk = r.tk;
            sg.hw_k = r.hw_k;
            sg.hw_n = r.hw_n;
            sg.tn_total = r.tn;
            sg.num_subs = 1u;
            sg.subs[0] = {sub.job_idx, r.n0, r.tn, 0u};
            ob_status_t sst = self_ref(sg, self_ref);
            if (sst != OB_OK) return sst;
          }
        }
        return OB_OK;
      };

      bool tile_handled_by_hw = false;
      if (emitter != nullptr) {
        // 1. Quantize shared A [hw_m, hw_k] ONCE per (k0, tk, hw_k) across all jobs & tiles!
        if (cached_a_k0 != k0 || cached_a_tk != tk || cached_aq_hw_k != hw_k) {
          const uint64_t tq0 = prof_on ? ob::monotonic_now_ns() : 0u;
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
          if (prof_on) ctx->prof.quant_ns += (ob::monotonic_now_ns() - tq0);
          cached_a_k0 = k0;
          cached_a_tk = tk;
          cached_aq_hw_k = hw_k;
          active_in_slot = OB_MAX_CREDITS;
          active_in_buf = nullptr;
        }

        // 2. Hash & lookup (or cold-pack) ext[2] weight tile (single or horizontally fused!)
        ob::CachedWeightTile* w_tile = nullptr;
        uint32_t scale_d = 1u;
        if (builtin_active) {
          const bool is_lt64_tile = ob::is_lt_64x_shape(hw_m, hw_k, hw_n);
          const uint32_t pack_fmt = is_lt64_tile ? 1u : 0u;
          const uint64_t th0 = prof_on ? ob::monotonic_now_ns() : 0u;
          ob::WeightHash128 wh_st{};
          if (num_subs == 1u) {
            const ob::FusedSubSlice& sub = grp.subs[0];
            const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
            const bool is_q4 = (job.is_q4_0 != 0u);
            const uint64_t tag = is_q4 ? 0x51345F5754494C45ULL : 0x51385F5754494C45ULL;
            uint32_t whdr[5] = {sub.tn, tk, hw_k, hw_n, pack_fmt};
            ob::hash128_update(&wh_st, &tag, sizeof(tag));
            ob::hash128_update(&wh_st, whdr, sizeof(whdr));
            const size_t blk_sz = is_q4 ? sizeof(ob_block_q4_0_t) : sizeof(ob_block_q8_0_t);
            const size_t row_blk_bytes = static_cast<size_t>(blk_cnt) * blk_sz;
            const size_t row_stride_bytes = static_cast<size_t>(nb_k) * blk_sz;
            const uint8_t* raw_base = static_cast<const uint8_t*>(job.b_nk) +
                                      (static_cast<size_t>(sub.n0) * nb_k + blk0) * blk_sz;
            ob::hash_weight_tile_for_ctx(ctx, &wh_st, raw_base, row_stride_bytes, sub.tn,
                                         row_blk_bytes, strict_hash);
          } else {
            const uint64_t tag = 0x465553455F575431ULL;
            uint32_t fhdr[5] = {num_subs, tk, hw_k, hw_n, pack_fmt};
            ob::hash128_update(&wh_st, &tag, sizeof(tag));
            ob::hash128_update(&wh_st, fhdr, sizeof(fhdr));
            for (uint32_t si = 0u; si < num_subs; ++si) {
              const ob::FusedSubSlice& sub = grp.subs[si];
              const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
              const bool is_q4 = (job.is_q4_0 != 0u);
              uint32_t shdr[3] = {sub.tn, sub.out_off, is_q4 ? 1u : 0u};
              ob::hash128_update(&wh_st, shdr, sizeof(shdr));
              const size_t blk_sz = is_q4 ? sizeof(ob_block_q4_0_t) : sizeof(ob_block_q8_0_t);
              const size_t row_blk_bytes = static_cast<size_t>(blk_cnt) * blk_sz;
              const size_t row_stride_bytes = static_cast<size_t>(nb_k) * blk_sz;
              const uint8_t* raw_base = static_cast<const uint8_t*>(job.b_nk) +
                                        (static_cast<size_t>(sub.n0) * nb_k + blk0) * blk_sz;
              ob::hash_weight_tile_for_ctx(ctx, &wh_st, raw_base, row_stride_bytes, sub.tn,
                                           row_blk_bytes, strict_hash);
            }
          }
          const ob::WeightTileKey wh = ob::finalize_weight_key(wh_st);
          w_tile = ob::weight_cache_find_locked(ctx, wh);
          if (prof_on) ctx->prof.hash_ns += (ob::monotonic_now_ns() - th0);

          if (w_tile && w_tile->rejected_fusion) {
            return fallback_split_subs();
          }

          if (!w_tile) {
            const uint64_t tp0 = prof_on ? ob::monotonic_now_ns() : 0u;
            b_q_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
            std::vector<float> sb_vec(tn_total, 0.0f);
            uint32_t min_sub_d = 0xFFFFFFFFu;

            for (uint32_t si = 0u; si < num_subs; ++si) {
              const ob::FusedSubSlice& sub = grp.subs[si];
              const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
              int8_t* bq_dst = b_q_tile.data() + static_cast<size_t>(sub.out_off) * hw_k;
              float* sb_dst = sb_vec.data() + sub.out_off;
              if (job.is_q4_0 != 0u) {
                ob::requant_q4_0_rows(static_cast<const ob_block_q4_0_t*>(job.b_nk),
                                      nb_k, sub.n0, sub.tn, blk0, blk_cnt,
                                      tk, hw_k, bq_dst, sb_dst, &tmp_w_row);
              } else {
                ob::requant_q8_0_rows(static_cast<const ob_block_q8_0_t*>(job.b_nk),
                                      nb_k, sub.n0, sub.tn, blk0, blk_cnt,
                                      tk, hw_k, bq_dst, sb_dst, &tmp_w_row);
              }
              if (num_subs >= 2u) {
                uint32_t sd = ob::compute_tile_scale_D(bq_dst, sub.tn, tk, hw_k);
                if (sd < min_sub_d) min_sub_d = sd;
              }
            }

            scale_d = ob::compute_tile_scale_D(b_q_tile.data(), tn_total, tk, hw_k);

            // Precision guard for horizontal fusion: if any sub-job's optimal D is
            // > 1.5x smaller than the fused D (e.g. mixed sparse+dense synthetic test),
            // cache a zero-byte split sentinel and run sub-jobs independently with exact D.
            if (num_subs >= 2u && min_sub_d > 0u && (scale_d * 2u > min_sub_d * 3u)) {
              bool sentinel_ok = false;
              ob_status_t spst = ob::weight_cache_prepare_space_locked(
                  ctx, 0u, ring_count, drain_one, &sentinel_ok);
              if (spst != OB_OK) {
                drain_all_cleanup();
                return spst;
              }
              if (sentinel_ok) {
                ob::CachedWeightTile* sentinel = new (std::nothrow) ob::CachedWeightTile();
                if (sentinel) {
                  sentinel->key = wh;
                  sentinel->scale_d = scale_d;
                  sentinel->hw_k = hw_k;
                  sentinel->hw_n = hw_n;
                  sentinel->ext2_buf = nullptr;
                  sentinel->ext2_bytes = 0u;
                  sentinel->rejected_fusion = true;
                  ob::weight_cache_insert_locked(ctx, sentinel);
                }
              }
              if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
              return fallback_split_subs();
            }

            std::vector<uint8_t> ext2 = is_lt64_tile
                ? ob::pack_ext2_for_lt_64x(hw_n, hw_k, b_q_tile.data(), nullptr)
                : ob::pack_ext2_for_tile(hw_n, hw_k, b_q_tile.data(), nullptr);
            if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);

            bool cacheable = false;
            ob_status_t pst = ob::weight_cache_prepare_space_locked(
                ctx, ext2.size(), ring_count, drain_one, &cacheable);
            if (pst != OB_OK) {
              drain_all_cleanup();
              return pst;
            }

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
            if (cacheable) {
              ob::weight_cache_insert_locked(ctx, w_tile);
            } else {
              w_tile->in_cache = false;
            }
          } else {
            scale_d = w_tile->scale_d;
          }
        } else {
          // Non-builtin pluggable emitter path (always 1-sub)
          const ob::FusedSubSlice& sub = grp.subs[0];
          const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
          b_q_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
          row_s_b.assign(sub.tn, 0.0f);
          if (job.is_q4_0 != 0u) {
            ob::requant_q4_0_rows(static_cast<const ob_block_q4_0_t*>(job.b_nk),
                                  nb_k, sub.n0, sub.tn, blk0, blk_cnt,
                                  tk, hw_k, b_q_tile.data(), row_s_b.data(), &tmp_w_row);
          } else {
            ob::requant_q8_0_rows(static_cast<const ob_block_q8_0_t*>(job.b_nk),
                                  nb_k, sub.n0, sub.tn, blk0, blk_cnt,
                                  tk, hw_k, b_q_tile.data(), row_s_b.data(), &tmp_w_row);
          }
          scale_d = ob::compute_tile_scale_D(b_q_tile.data(), sub.tn, tk, hw_k);
        }

        if (builtin_active && cached_a_rms_shift > 0u) {
          scale_d = ob::quant_scale_D_grid(
              static_cast<double>(std::max<uint32_t>(1u, scale_d >> cached_a_rms_shift)));
        }

        auto discard_unsubmitted_wtile = [&]() {
          if (w_tile && !w_tile->in_cache && w_tile->in_use == 0u) {
            if (w_tile->ext2_buf) ob_buffer_free(w_tile->ext2_buf);
            delete w_tile;
            w_tile = nullptr;
          }
        };

        if (ring_count >= ring_depth) {
          ob_status_t dst = drain_one();
          if (dst != OB_OK) {
            discard_unsubmitted_wtile();
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
            discard_unsubmitted_wtile();
            drain_all_cleanup();
            return pst;
          }
          desc.packed_params = packed_tile.data();
          desc.packed_params_bytes = packed_tile.size();
          desc.cache_key = ob::hash_tile_key(
              hw_m, hw_k, hw_n, 0, 0, scale_d, b_q_tile.data(), b_q_tile.size());
        }

        const uint64_t ts0 = prof_on ? ob::monotonic_now_ns() : 0u;
        ob::CachedGraph* entry = nullptr;
        bool was_hit = false;
        ob_status_t gst = ob::acquire_tile_entry(
            ctx, emitter, emitter_ud, desc, &entry, &was_hit);
        if (gst == OB_OK) {
          tiles_dispatched += 1u;
          used_emitter = true;
          if (was_hit) cache_hits += 1u;
          if (num_subs >= 2u) {
            ctx->prof.horiz_fusions += 1u;
            ctx->prof.horiz_jobs_saved += (num_subs - 1u);
          }

          size_t in_alloc = std::max<size_t>(
              a_q_tile.size(),
              !entry->parsed.input_sizes.empty()
                  ? static_cast<size_t>(entry->parsed.input_sizes[0]) : 1u);
          size_t out_alloc = std::max<size_t>(
              static_cast<size_t>(hw_m) * ob::align_up_u32(hw_n, 4u),
              !entry->parsed.output_sizes.empty()
                  ? static_cast<size_t>(entry->parsed.output_sizes[0]) : 1u);

          ob_buffer* out_buf = nullptr;
          ob_status_t bst = ob::ensure_pipeline_buf_tiered(
              ctx, &ctx->gemm_out_bufs_4k[slot], &ctx->gemm_out_bufs[slot], out_alloc, &out_buf);
          if (bst == OB_OK) {
            // Share ONE synced input DMA-BUF across all sibling tiles sharing (k0, tk, hw_k)!
            if (active_in_slot == OB_MAX_CREDITS ||
                !active_in_buf ||
                active_in_buf->slab.alloc_size < in_alloc) {
              uint32_t chosen = 0u;
              for (uint32_t c = 0u; c < ring_depth; ++c) {
                if (in_refcnt[c] == 0u) { chosen = c; break; }
              }
              ob_buffer* in_buf = nullptr;
              bst = ob::ensure_pipeline_buf_tiered(
                  ctx, &ctx->gemm_in_bufs_4k[chosen], &ctx->gemm_in_bufs[chosen], in_alloc, &in_buf);
              if (bst == OB_OK) {
                const uint64_t td0 = prof_on ? ob::monotonic_now_ns() : 0u;
                bst = ob_buffer_write(in_buf, a_q_tile.data(), a_q_tile.size(), 0u);
                if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td0);
              }
              if (bst == OB_OK) {
                active_in_slot = chosen;
                active_in_buf = in_buf;
              }
            }
          }

          ob_exec_handle_t exec_h = 0u;
          if (bst == OB_OK) {
            bst = ob::device_submit_execute_async(
                ctx, entry,
                &active_in_buf, 1u,
                &out_buf, 1u,
                &exec_h,
                w_tile ? w_tile->ext2_buf : nullptr);
          }
          if (prof_on) ctx->prof.submit_ns += (ob::monotonic_now_ns() - ts0);
          if (bst != OB_OK) {
            ob::release_tile_entry(ctx, entry);
            discard_unsubmitted_wtile();
            drain_all_cleanup();
            return bst;
          }

          PendingF32Tile& p = ring[slot];
          p.handle = exec_h;
          p.w_tile = w_tile;
          if (w_tile) w_tile->in_use += 1u;
          p.in_buf = active_in_buf;
          if (p.in_buf) ob::buffer_retain(p.in_buf);
          p.out_buf = out_buf;
          if (p.out_buf) ob::buffer_retain(p.out_buf);
          p.slot = slot;
          p.in_slot = active_in_slot;
          in_refcnt[active_in_slot] += 1u;
          p.m0 = m0;
          p.tm = tm;
          p.hw_m = hw_m;
          p.hw_k = hw_k;
          p.hw_n = hw_n;
          p.tn_total = tn_total;
          p.num_subs = num_subs;
          for (uint32_t si = 0u; si < num_subs; ++si) p.subs[si] = grp.subs[si];
          p.scale_d_u32 = scale_d;
          if (tm <= p.s_a_inline.size()) {
            for (uint32_t mi = 0; mi < tm; ++mi) p.s_a_inline[mi] = row_s_a[mi];
          } else {
            p.s_a_overflow = row_s_a;
          }
          if (!w_tile) p.s_b = row_s_b;
          ring_count += 1u;

          if (!ctx->mock_mode) {
            tile_handled_by_hw = true;
            // Overlapped with in-flight TPU tile: add sparse exact FP32 residual for <=8 peeled outliers
            // using deduplicated k_off + lazy per-tile FP32 outlier-column cache + ARM64 NEON FMA!
            bool any_outliers = false;
            for (uint32_t mi = 0u; mi < tm; ++mi) {
              if (!row_outliers[mi].empty()) {
                any_outliers = true;
                break;
              }
            }
            if (any_outliers) {
              const uint64_t to0 = prof_on ? ob::monotonic_now_ns() : 0u;
              uint32_t uniq_k_stack[512]{};
              std::vector<uint32_t> uniq_k_vec;
              uint32_t n_uniq = 0u;
              const uint32_t* uniq_ptr = uniq_k_stack;
              for (uint32_t mi = 0u; mi < tm; ++mi) {
                for (const auto& ov : row_outliers[mi]) {
                  bool seen = false;
                  for (uint32_t u = 0u; u < n_uniq; ++u) {
                    if (uniq_ptr[u] == ov.first) { seen = true; break; }
                  }
                  if (!seen) {
                    if (n_uniq < 512u && uniq_k_vec.empty()) {
                      uniq_k_stack[n_uniq++] = ov.first;
                    } else {
                      if (uniq_k_vec.empty()) {
                        uniq_k_vec.assign(uniq_k_stack, uniq_k_stack + n_uniq);
                      }
                      uniq_k_vec.push_back(ov.first);
                      uniq_ptr = uniq_k_vec.data();
                      n_uniq = static_cast<uint32_t>(uniq_k_vec.size());
                    }
                  }
                }
              }

              auto extract_col = [&](uint32_t k_off, float* dst_col) {
                const uint32_t abs_k = k0 + k_off;
                const uint32_t b_idx = abs_k / OB_QK8_0;
                const uint32_t j_idx = abs_k % OB_QK8_0;
                for (uint32_t si = 0u; si < num_subs; ++si) {
                  const ob::FusedSubSlice& sub = grp.subs[si];
                  const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
                  float* out_p = dst_col + sub.out_off;
                  if (job.is_q4_0 != 0u) {
                    const ob_block_q4_0_t* b_q4 = static_cast<const ob_block_q4_0_t*>(job.b_nk);
                    constexpr uint32_t kHalf = OB_QK8_0 / 2u;
                    const bool hi_nib = (j_idx >= kHalf);
                    const uint32_t byte_idx = hi_nib ? (j_idx - kHalf) : j_idx;
                    for (uint32_t ni = 0u; ni < sub.tn; ++ni) {
                      if (ni + 4u < sub.tn) {
                        __builtin_prefetch(
                            &b_q4[static_cast<size_t>(sub.n0 + ni + 4u) * nb_k + b_idx], 0, 1);
                      }
                      const ob_block_q4_0_t& blk =
                          b_q4[static_cast<size_t>(sub.n0 + ni) * nb_k + b_idx];
                      const uint8_t byte = blk.qs[byte_idx];
                      const int32_t q =
                          static_cast<int32_t>(hi_nib ? (byte >> 4) : (byte & 0x0Fu)) - 8;
                      out_p[ni] = ob::fp16_to_fp32(blk.d) * static_cast<float>(q);
                    }
                  } else {
                    const ob_block_q8_0_t* b_q8 = static_cast<const ob_block_q8_0_t*>(job.b_nk);
                    for (uint32_t ni = 0u; ni < sub.tn; ++ni) {
                      if (ni + 4u < sub.tn) {
                        __builtin_prefetch(
                            &b_q8[static_cast<size_t>(sub.n0 + ni + 4u) * nb_k + b_idx], 0, 1);
                      }
                      const ob_block_q8_0_t& blk =
                          b_q8[static_cast<size_t>(sub.n0 + ni) * nb_k + b_idx];
                      out_p[ni] = ob::fp16_to_fp32(blk.d) * static_cast<float>(blk.qs[j_idx]);
                    }
                  }
                }
              };

              constexpr size_t kMaxCachedOutlierCols = 24u;
              const size_t w_budget = ob::compute_weight_cache_budget_bytes(ctx);
              for (uint32_t u = 0u; u < n_uniq; ++u) {
                const uint32_t k_off = uniq_ptr[u];
                const float* w_col = nullptr;
                if (w_tile) {
                  auto it = w_tile->outlier_cols.find(k_off);
                  if (it != w_tile->outlier_cols.end() && it->second.size() >= tn_total) {
                    w_col = it->second.data();
                  } else if (w_tile->in_cache &&
                             w_tile->outlier_cols.size() < kMaxCachedOutlierCols &&
                             ctx->weight_cache_bytes + static_cast<size_t>(tn_total) * sizeof(float) <= w_budget) {
                    std::vector<float> col(tn_total);
                    extract_col(k_off, col.data());
                    const size_t col_bytes = col.size() * sizeof(float);
                    auto res = w_tile->outlier_cols.emplace(k_off, std::move(col));
                    if (res.second) {
                      w_tile->outlier_cols_bytes += col_bytes;
                      ctx->weight_cache_bytes += col_bytes;
                      if (ctx->weight_cache_bytes > ctx->weight_cache_peak_bytes) {
                        ctx->weight_cache_peak_bytes = ctx->weight_cache_bytes;
                      }
                    }
                    w_col = res.first->second.data();
                  }
                }
                if (!w_col) {
                  ctx->gemm_wcol_scratch.resize(tn_total);
                  extract_col(k_off, ctx->gemm_wcol_scratch.data());
                  w_col = ctx->gemm_wcol_scratch.data();
                }
                for (uint32_t mi = 0u; mi < tm; ++mi) {
                  for (const auto& ov : row_outliers[mi]) {
                    if (ov.first != k_off) continue;
                    const float x_val = ov.second;
                    for (uint32_t si = 0u; si < num_subs; ++si) {
                      const ob::FusedSubSlice& sub = grp.subs[si];
                      const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
                      float* c_row = job.c_mn + static_cast<size_t>(m0 + mi) * job.n_dim + sub.n0;
                      ob::neon_fma_row_f32(c_row, w_col + sub.out_off, sub.tn, x_val);
                    }
                  }
                }
              }
              if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - to0);
            }
          }
        } else {
          discard_unsubmitted_wtile();
          if (gst != OB_ERR_UNSUPPORTED) {
            drain_all_cleanup();
            return gst;
          }
        }
      } else {
        tiles_dispatched += 1u;
      }

      if (!tile_handled_by_hw) {
        used_cpu_fallback = true;
        for (uint32_t si = 0u; si < num_subs; ++si) {
          const ob::FusedSubSlice& sub = grp.subs[si];
          const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
          if (job.is_q4_0 != 0u) {
            ob::cpu_matmul_f32_q4_0_tile(a_mk, static_cast<const ob_block_q4_0_t*>(job.b_nk),
                                         job.c_mn, k_dim, job.n_dim,
                                         m0, tm, k0, tk, sub.n0, sub.tn);
          } else {
            ob::cpu_matmul_f32_q8_0_tile(a_mk, static_cast<const ob_block_q8_0_t*>(job.b_nk),
                                         job.c_mn, k_dim, job.n_dim,
                                         m0, tm, k0, tk, sub.n0, sub.tn);
          }
        }
      }
      return OB_OK;
    };

    for (const ob::ScheduledGroup& grp : groups) {
      ob_status_t dst = dispatch_group(grp, dispatch_group);
      if (dst != OB_OK) {
        drain_all_cleanup();
        return dst;
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

  const uint64_t tact0 = prof_on ? ob::monotonic_now_ns() : 0u;
  for (uint32_t j = 0; j < num_jobs; ++j) {
    if (jobs[j].activation != OB_ACT_NONE) {
      ob::apply_activation_inplace_impl(
          jobs[j].c_mn, static_cast<size_t>(m_dim) * jobs[j].n_dim, jobs[j].activation);
    }
  }
  if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tact0);

  const uint64_t wall_ns = ob::monotonic_now_ns() - t0_ns;
  ctx->prof.calls += 1u;
  ctx->prof.tiles_dispatched += tiles_dispatched;
  ctx->prof.hw_us += total_hw_us;
  ctx->prof.total_wall_ns += wall_ns;

  if (out_stats) {
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->hw_execution_us = total_hw_us;
    out_stats->wall_time_us = wall_ns / 1000ULL;
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

ob_status_t ob_gemm_swiglu_int8(ob_context_t* ctx,
                                const int8_t* x_mk,
                                const ob_swiglu_int8_desc_t* desc,
                                int8_t* y_mn,
                                ob_invoke_stats_t* out_stats) {
  if (!ctx || !x_mk || !desc || !y_mn) return OB_ERR_INVALID_ARG;
  if (!ob::is_swiglu_shape(desc->m_dim, desc->k_dim, desc->h_dim, desc->n_dim)) {
    ob::set_last_error(ctx, "Unsupported exact INT8 SwiGLU shape");
    return OB_ERR_UNSUPPORTED;
  }

  const uint64_t t0_ns = ob::monotonic_now_ns();
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);
  const bool prof_on = ctx->profile_enabled;

  const uint32_t M = desc->m_dim;
  const uint32_t K = desc->k_dim;
  const uint32_t H = desc->h_dim;
  const uint32_t N = desc->n_dim;

  // Pack ext[2] for the 3 weight matrices
  std::vector<uint8_t> ext2 = ob::pack_ext2_for_swiglu(
      M, K, H, N,
      desc->wg_hk, desc->bg_h,
      desc->wu_hk, desc->bu_h,
      desc->wd_nh, desc->bd_n);

  ob_buffer* wbuf = nullptr;
  ob_status_t st = ob_buffer_alloc(
      ctx, ext2.size(), OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL | OB_BUF_ZERO_INIT, &wbuf);
  if (st != OB_OK) return st;
  st = ob_buffer_write(wbuf, ext2.data(), ext2.size(), 0u);
  if (st != OB_OK) {
    ob_buffer_free(wbuf);
    return st;
  }

  // Unique graph cache key binding shape + all 9 exact float scales (weight-agnostic Probe-A!)
  uint64_t gkey = 0x535749385F303031ULL; // "SWI8_001"
  uint32_t shdr[4] = {M, K, H, N};
  float sf[9] = {
      desc->s_in, desc->s_wg, desc->s_gate,
      desc->s_wu, desc->s_up, desc->s_silu,
      desc->s_prod, desc->s_wd, desc->s_out
  };
  gkey = ob::fast_hash64_update(gkey, shdr, sizeof(shdr));
  gkey = ob::fast_hash64_update(gkey, sf, sizeof(sf));
  if (gkey == 0u) gkey = 1u;

  ob_swiglu_int8_desc_t graph_desc = *desc;
  // Pass zero weights into graph template since live weights ride in wbuf via Probe-A!
  graph_desc.wg_hk = nullptr;
  graph_desc.bg_h = nullptr;
  graph_desc.wu_hk = nullptr;
  graph_desc.bu_h = nullptr;
  graph_desc.wd_nh = nullptr;
  graph_desc.bd_n = nullptr;

  ob::CachedGraph* entry = nullptr;
  bool was_hit = false;
  st = ob::acquire_swiglu_entry(ctx, graph_desc, gkey, &entry, &was_hit);
  if (st != OB_OK) {
    ob_buffer_free(wbuf);
    return st;
  }

  const size_t in_bytes = static_cast<size_t>(M) * K;
  const size_t out_bytes = static_cast<size_t>(M) * N;
  ob_buffer* in_buf = nullptr;
  ob_buffer* out_buf = nullptr;
  st = ob::ensure_pipeline_buf_tiered(
      ctx, &ctx->gemm_in_bufs_4k[0], &ctx->gemm_in_bufs[0], in_bytes, &in_buf);
  if (st == OB_OK) {
    st = ob::ensure_pipeline_buf_tiered(
        ctx, &ctx->gemm_out_bufs_4k[0], &ctx->gemm_out_bufs[0], out_bytes, &out_buf);
  }
  if (st == OB_OK) {
    st = ob_buffer_write(in_buf, x_mk, in_bytes, 0u);
  }

  ob_exec_handle_t exec_h = 0u;
  if (st == OB_OK) {
    st = ob::device_submit_execute_async(ctx, entry, &in_buf, 1u, &out_buf, 1u, &exec_h, wbuf);
  }
  if (st != OB_OK) {
    ob::release_tile_entry(ctx, entry);
    ob_buffer_free(wbuf);
    return st;
  }

  ob_invoke_stats_t istats {};
  st = ob::device_wait_handle(ctx, exec_h, ctx->cfg.timeout_ms, &istats);
  if (st != OB_OK) {
    if (ob::device_attach_inflight_buffers(ctx, exec_h, {wbuf, in_buf, out_buf})) {
      if (ctx->gemm_in_bufs_4k[0] == in_buf) {
        ob_buffer_free(ctx->gemm_in_bufs_4k[0]);
        ctx->gemm_in_bufs_4k[0] = nullptr;
      }
      if (ctx->gemm_in_bufs[0] == in_buf) {
        ob_buffer_free(ctx->gemm_in_bufs[0]);
        ctx->gemm_in_bufs[0] = nullptr;
      }
      if (ctx->gemm_out_bufs_4k[0] == out_buf) {
        ob_buffer_free(ctx->gemm_out_bufs_4k[0]);
        ctx->gemm_out_bufs_4k[0] = nullptr;
      }
      if (ctx->gemm_out_bufs[0] == out_buf) {
        ob_buffer_free(ctx->gemm_out_bufs[0]);
        ctx->gemm_out_bufs[0] = nullptr;
      }
    }
    ob_buffer_free(wbuf);
    return st;
  }
  ob_buffer_free(wbuf);

  if (ctx->mock_mode) {
    ob::simulate_swiglu_hw_exact(x_mk, *desc, y_mn);
  } else {
    st = ob_buffer_read(out_buf, y_mn, out_bytes, 0u);
    if (st != OB_OK) return st;
  }

  const uint64_t wall_ns = ob::monotonic_now_ns() - t0_ns;
  ctx->prof.swiglu_calls += 1u;
  ctx->prof.swiglu_chunks_chained += 1u;
  ctx->prof.tiles_dispatched += 1u;
  ctx->prof.hw_us += istats.hw_execution_us;
  ctx->prof.total_wall_ns += wall_ns;
  (void)prof_on;

  if (out_stats) {
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->hw_execution_us = istats.hw_execution_us;
    out_stats->wall_time_us = wall_ns / 1000ULL;
    out_stats->tiles_dispatched = 1u;
    out_stats->cache_hits = was_hit ? 1u : 0u;
    if (ctx->mock_mode) out_stats->flags |= OB_STAT_FLAG_MOCK;
    out_stats->flags |= OB_STAT_FLAG_EMITTER_USED;
    if (was_hit) out_stats->flags |= OB_STAT_FLAG_CACHE_HIT;
  }
  return OB_OK;
}

namespace {

// Dequantize one quantized row [0..len) into float dst[0..len).
void dequant_quant_row_f32(const void* blocks_base, uint8_t is_q4_0,
                           uint32_t nb_stride, uint32_t row_idx,
                           uint32_t blk0, uint32_t blk_cnt, float* dst) {
  if (is_q4_0 != 0u) {
    const ob_block_q4_0_t* b_row =
        static_cast<const ob_block_q4_0_t*>(blocks_base) +
        static_cast<size_t>(row_idx) * nb_stride + blk0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const uint8x16_t mask_lo = vdupq_n_u8(0x0Fu);
    const int8x16_t bias_8 = vdupq_n_s8(-8);
    for (uint32_t bi = 0u; bi < blk_cnt; ++bi) {
      const float32x4_t vd = vdupq_n_f32(ob::fp16_to_fp32(b_row[bi].d));
      float* out = dst + static_cast<size_t>(bi) * OB_QK8_0;
      uint8x16_t raw = vld1q_u8(b_row[bi].qs);
      int8x16_t lo8 = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), bias_8);
      int8x16_t hi8 = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias_8);
      int16x8_t lo16_0 = vmovl_s8(vget_low_s8(lo8));
      int16x8_t lo16_1 = vmovl_high_s8(lo8);
      int16x8_t hi16_0 = vmovl_s8(vget_low_s8(hi8));
      int16x8_t hi16_1 = vmovl_high_s8(hi8);
      vst1q_f32(out +  0u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0)))));
      vst1q_f32(out +  4u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(lo16_0))));
      vst1q_f32(out +  8u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1)))));
      vst1q_f32(out + 12u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(lo16_1))));
      vst1q_f32(out + 16u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0)))));
      vst1q_f32(out + 20u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(hi16_0))));
      vst1q_f32(out + 24u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1)))));
      vst1q_f32(out + 28u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(hi16_1))));
    }
#else
    constexpr uint32_t kHalf = OB_QK8_0 / 2u;
    for (uint32_t bi = 0u; bi < blk_cnt; ++bi) {
      const float d = ob::fp16_to_fp32(b_row[bi].d);
      const uint8_t* qs = b_row[bi].qs;
      float* out = dst + static_cast<size_t>(bi) * OB_QK8_0;
      for (uint32_t j = 0u; j < kHalf; ++j) {
        const uint8_t byte = qs[j];
        out[j] = d * static_cast<float>(static_cast<int32_t>(byte & 0x0Fu) - 8);
        out[j + kHalf] = d * static_cast<float>(static_cast<int32_t>(byte >> 4) - 8);
      }
    }
#endif
  } else {
    const ob_block_q8_0_t* b_row =
        static_cast<const ob_block_q8_0_t*>(blocks_base) +
        static_cast<size_t>(row_idx) * nb_stride + blk0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    for (uint32_t bi = 0u; bi < blk_cnt; ++bi) {
      const float32x4_t vd = vdupq_n_f32(ob::fp16_to_fp32(b_row[bi].d));
      const int8_t* qs = b_row[bi].qs;
      float* out = dst + static_cast<size_t>(bi) * OB_QK8_0;
      for (uint32_t off = 0u; off < OB_QK8_0; off += 16u) {
        int8x16_t q8 = vld1q_s8(qs + off);
        int16x8_t q16_lo = vmovl_s8(vget_low_s8(q8));
        int16x8_t q16_hi = vmovl_high_s8(q8);
        vst1q_f32(out + off +  0u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo)))));
        vst1q_f32(out + off +  4u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(q16_lo))));
        vst1q_f32(out + off +  8u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi)))));
        vst1q_f32(out + off + 12u, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(q16_hi))));
      }
    }
#else
    for (uint32_t bi = 0u; bi < blk_cnt; ++bi) {
      const float d = ob::fp16_to_fp32(b_row[bi].d);
      const int8_t* qs = b_row[bi].qs;
      float* out = dst + static_cast<size_t>(bi) * OB_QK8_0;
      for (uint32_t j = 0u; j < OB_QK8_0; ++j) {
        out[j] = d * static_cast<float>(qs[j]);
      }
    }
#endif
  }
}

// Fast ARM64 NEON Q8_0 / Q4_0 dot product for M=1 exact-FP32 transient outlier SwiGLU fallback
// (zero weight-cache pollution + exact FP32 precision on rare attention-sink tokens).
float neon_dot_f32_q8_row(const float* a, const ob_block_q8_0_t* row, uint32_t nb) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const float d = ob::fp16_to_fp32(row[bi].d);
    const int8_t* qs = row[bi].qs;
    const float* ap = a + static_cast<size_t>(bi) * OB_QK8_0;
    float32x4_t bsum0 = vdupq_n_f32(0.0f);
    float32x4_t bsum1 = vdupq_n_f32(0.0f);
    for (uint32_t off = 0u; off < OB_QK8_0; off += 16u) {
      int8x16_t q8 = vld1q_s8(qs + off);
      int16x8_t q16_lo = vmovl_s8(vget_low_s8(q8));
      int16x8_t q16_hi = vmovl_high_s8(q8);
      bsum0 = vfmaq_f32(bsum0, vld1q_f32(ap + off),       vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo))));
      bsum1 = vfmaq_f32(bsum1, vld1q_f32(ap + off + 4u),  vcvtq_f32_s32(vmovl_high_s16(q16_lo)));
      bsum0 = vfmaq_f32(bsum0, vld1q_f32(ap + off + 8u),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi))));
      bsum1 = vfmaq_f32(bsum1, vld1q_f32(ap + off + 12u), vcvtq_f32_s32(vmovl_high_s16(q16_hi)));
    }
    const float32x4_t vd = vdupq_n_f32(d);
    acc0 = vfmaq_f32(acc0, bsum0, vd);
    acc1 = vfmaq_f32(acc1, bsum1, vd);
  }
  return vaddvq_f32(vaddq_f32(acc0, acc1));
#else
  double sum = 0.0;
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const double d = static_cast<double>(ob::fp16_to_fp32(row[bi].d));
    const int8_t* qs = row[bi].qs;
    const float* ap = a + static_cast<size_t>(bi) * OB_QK8_0;
    double bsum = 0.0;
    for (uint32_t j = 0u; j < OB_QK8_0; ++j) bsum += static_cast<double>(ap[j]) * static_cast<double>(qs[j]);
    sum += d * bsum;
  }
  return static_cast<float>(sum);
#endif
}

float neon_dot_f32_q4_row(const float* a, const ob_block_q4_0_t* row, uint32_t nb) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const uint8x16_t mask_lo = vdupq_n_u8(0x0Fu);
  const int8x16_t bias_8 = vdupq_n_s8(-8);
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const float d = ob::fp16_to_fp32(row[bi].d);
    const float* ap = a + static_cast<size_t>(bi) * OB_QK8_0;
    uint8x16_t raw = vld1q_u8(row[bi].qs);
    int8x16_t lo8 = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), bias_8);
    int8x16_t hi8 = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias_8);
    int16x8_t lo16_0 = vmovl_s8(vget_low_s8(lo8));
    int16x8_t lo16_1 = vmovl_high_s8(lo8);
    int16x8_t hi16_0 = vmovl_s8(vget_low_s8(hi8));
    int16x8_t hi16_1 = vmovl_high_s8(hi8);
    float32x4_t bsum0 = vmulq_f32(vld1q_f32(ap + 0u),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0))));
    float32x4_t bsum1 = vmulq_f32(vld1q_f32(ap + 4u),  vcvtq_f32_s32(vmovl_high_s16(lo16_0)));
    bsum0 = vfmaq_f32(bsum0, vld1q_f32(ap + 8u),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1))));
    bsum1 = vfmaq_f32(bsum1, vld1q_f32(ap + 12u), vcvtq_f32_s32(vmovl_high_s16(lo16_1)));
    bsum0 = vfmaq_f32(bsum0, vld1q_f32(ap + 16u), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0))));
    bsum1 = vfmaq_f32(bsum1, vld1q_f32(ap + 20u), vcvtq_f32_s32(vmovl_high_s16(hi16_0)));
    bsum0 = vfmaq_f32(bsum0, vld1q_f32(ap + 24u), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1))));
    bsum1 = vfmaq_f32(bsum1, vld1q_f32(ap + 28u), vcvtq_f32_s32(vmovl_high_s16(hi16_1)));
    const float32x4_t vd = vdupq_n_f32(d);
    acc0 = vfmaq_f32(acc0, bsum0, vd);
    acc1 = vfmaq_f32(acc1, bsum1, vd);
  }
  return vaddvq_f32(vaddq_f32(acc0, acc1));
#else
  double sum = 0.0;
  constexpr uint32_t kHalf = OB_QK8_0 / 2u;
  for (uint32_t bi = 0u; bi < nb; ++bi) {
    const double d = static_cast<double>(ob::fp16_to_fp32(row[bi].d));
    const uint8_t* qs = row[bi].qs;
    const float* ap = a + static_cast<size_t>(bi) * OB_QK8_0;
    double bsum = 0.0;
    for (uint32_t j = 0u; j < kHalf; ++j) {
      const uint8_t byte = qs[j];
      bsum += static_cast<double>(ap[j]) * static_cast<double>(static_cast<int32_t>(byte & 0x0Fu) - 8);
      bsum += static_cast<double>(ap[j + kHalf]) * static_cast<double>(static_cast<int32_t>(byte >> 4) - 8);
    }
    sum += d * bsum;
  }
  return static_cast<float>(sum);
#endif
}

inline int64_t dot_i8_row_exact(const int8_t* a, const int8_t* b, uint32_t K) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  uint32_t k = 0u;
  for (; k + 16u <= K; k += 16u) {
    int8x16_t va = vld1q_s8(a + k);
    int8x16_t vb = vld1q_s8(b + k);
    acc0 = vaddq_s32(acc0, vpaddlq_s16(vmull_s8(vget_low_s8(va), vget_low_s8(vb))));
    acc1 = vaddq_s32(acc1, vpaddlq_s16(vmull_high_s8(va, vb)));
  }
  int64_t s = static_cast<int64_t>(vaddvq_s32(vaddq_s32(acc0, acc1)));
  for (; k < K; ++k) s += static_cast<int64_t>(a[k]) * static_cast<int64_t>(b[k]);
  return s;
#else
  int64_t s = 0LL;
  for (uint32_t k = 0u; k < K; ++k) s += static_cast<int64_t>(a[k]) * static_cast<int64_t>(b[k]);
  return s;
#endif
}

// Verify intermediate gate, up, and product ranges across all rows of a SwiGLU chunk.
// Allows at most a 2x power-of-2 divider adjustment when borderline Gaussian tail channels
// exceed 127 (up to 254), and rejects to M7A fallback whenever any intermediate stage
// would still clip (eliminating silent correlated saturation).
bool swiglu_chunk_clips_or_tunes(ob_context* ctx,
                                 const ob::CachedWeightTile* w_tile,
                                 const int8_t* cold_wg_q,
                                 const int8_t* cold_wu_q,
                                 const int8_t* a_q,
                                 uint32_t tm,
                                 uint32_t hw_m,
                                 uint32_t K,
                                 uint32_t hw_h,
                                 uint32_t th,
                                 float s_a,
                                 float s_wg,
                                 uint32_t* io_d_gate,
                                 uint32_t* io_d_up,
                                 uint32_t* io_d_mul) {
  const bool have_cold = (cold_wg_q != nullptr && cold_wu_q != nullptr);
  const size_t p_gate = (hw_m == 64u)
      ? (static_cast<size_t>(hw_h) * (K + 16u))
      : (static_cast<size_t>(hw_h) * (K + 4u));

  struct Ext2ReadGuard {
    ob_buffer* buf = nullptr;
    bool active = false;
    ~Ext2ReadGuard() {
      if (active && buf) ob::buffer_sync_end_read(buf);
    }
  } ext2_guard;

  const uint8_t* e2_g = nullptr;
  const uint8_t* e2_u = nullptr;
  std::vector<int8_t> wg_row_buf;
  std::vector<int8_t> wu_row_buf;

  if (!have_cold) {
    if (!w_tile || !w_tile->ext2_buf || w_tile->ext2_bytes < 2u * p_gate) {
      return true;
    }
    if (ctx && !ctx->mock_mode) {
      if (ob::buffer_sync_begin_read(w_tile->ext2_buf) != OB_OK) {
        return true;
      }
      ext2_guard.buf = w_tile->ext2_buf;
      ext2_guard.active = true;
    }
    const uint8_t* e2_base = static_cast<const uint8_t*>(ob_buffer_data(w_tile->ext2_buf));
    if (!e2_base) return true;
    e2_g = e2_base;
    e2_u = e2_base + p_gate;
    wg_row_buf.resize(K);
    wu_row_buf.resize(K);
  }

  for (int attempt = 0; attempt < 2; ++attempt) {
    const uint32_t dg = *io_d_gate;
    const uint32_t du = *io_d_up;
    const uint32_t dm = *io_d_mul;
    const float s_gate = s_a * s_wg * static_cast<float>(dg);

    int64_t worst_g = 0;
    int64_t worst_u = 0;
    int64_t worst_prod = 0;
    for (uint32_t hi = 0u; hi < th; ++hi) {
      const int8_t* wg_row = nullptr;
      const int8_t* wu_row = nullptr;
      if (have_cold) {
        wg_row = cold_wg_q + static_cast<size_t>(hi) * K;
        wu_row = cold_wu_q + static_cast<size_t>(hi) * K;
      } else {
        if (hw_m == 64u) {
          const uint32_t ng = hi >> 3u;
          const uint32_t c = hi & 7u;
          const size_t grp_off = static_cast<size_t>(ng) * (128u + 8u * K) + 128u;
          for (uint32_t kb = 0u; kb < K / 16u; ++kb) {
            const size_t blk_off = grp_off + (static_cast<size_t>(kb) * 8u + c) * 16u;
            for (uint32_t l = 0u; l < 16u; ++l) {
              wg_row_buf[kb * 16u + l] = static_cast<int8_t>(e2_g[blk_off + l] ^ 0x80u);
              wu_row_buf[kb * 16u + l] = static_cast<int8_t>(e2_u[blk_off + l] ^ 0x80u);
            }
          }
        } else {
          const uint32_t g = hi >> 5u;
          const uint32_t r = hi & 31u;
          const size_t grp_off = static_cast<size_t>(g) * (128u + 32u * K) + 128u;
          for (uint32_t kb = 0u; kb < K / 16u; ++kb) {
            const size_t blk_off = grp_off + static_cast<size_t>(kb) * 512u;
            for (uint32_t q = 0u; q < 4u; ++q) {
              const size_t q_off = blk_off + static_cast<size_t>(q) * 128u + static_cast<size_t>(r) * 4u;
              for (uint32_t l = 0u; l < 4u; ++l) {
                wg_row_buf[kb * 16u + q * 4u + l] = static_cast<int8_t>(e2_g[q_off + l] ^ 0x80u);
                wu_row_buf[kb * 16u + q * 4u + l] = static_cast<int8_t>(e2_u[q_off + l] ^ 0x80u);
              }
            }
          }
        }
        wg_row = wg_row_buf.data();
        wu_row = wu_row_buf.data();
      }
      for (uint32_t mi = 0u; mi < tm; ++mi) {
        const int8_t* aq = a_q + static_cast<size_t>(mi) * K;
        const int64_t acc_g = dot_i8_row_exact(aq, wg_row, K);
        const int64_t acc_u = dot_i8_row_exact(aq, wu_row, K);
        const int64_t ag = acc_g < 0 ? -acc_g : acc_g;
        const int64_t au = acc_u < 0 ? -acc_u : acc_u;
        if (ag > worst_g) worst_g = ag;
        if (au > worst_u) worst_u = au;
        if (ag > 254LL * static_cast<int64_t>(dg) || au > 254LL * static_cast<int64_t>(du)) {
          return true;
        }
        const float gr = std::nearbyint(static_cast<float>(acc_g) / static_cast<float>(dg));
        const float ur = std::nearbyint(static_cast<float>(acc_u) / static_cast<float>(du));
        const int32_t q_g = (gr < -128.0f) ? -128 : ((gr > 127.0f) ? 127 : static_cast<int32_t>(gr));
        const int32_t q_u = (ur < -128.0f) ? -128 : ((ur > 127.0f) ? 127 : static_cast<int32_t>(ur));
        const int32_t q_silu = ob::eval_swiglu_silu_q(q_g, s_gate);
        const int64_t prod = static_cast<int64_t>(q_silu) * static_cast<int64_t>(q_u);
        const int64_t ap = prod < 0 ? -prod : prod;
        if (ap > worst_prod) worst_prod = ap;
        if (ap > 254LL * 64LL) {
          return true;
        }
      }
    }

    const bool g_ok = (worst_g <= 127LL * static_cast<int64_t>(dg));
    const bool u_ok = (worst_u <= 127LL * static_cast<int64_t>(du));
    const bool m_ok = (worst_prod <= 127LL * static_cast<int64_t>(dm));
    if (g_ok && u_ok && m_ok) return false;
    if (attempt > 0) return true;

    bool changed = false;
    if (!g_ok) {
      if (dg >= 65536u || worst_g > 127LL * static_cast<int64_t>(dg << 1u)) return true;
      *io_d_gate = dg << 1u;
      changed = true;
    }
    if (!u_ok) {
      if (du >= 65536u || worst_u > 127LL * static_cast<int64_t>(du << 1u)) return true;
      *io_d_up = du << 1u;
      changed = true;
    }
    if (!m_ok) {
      uint32_t next_dm = dm;
      while (next_dm < 64u && worst_prod > 127LL * static_cast<int64_t>(next_dm)) {
        next_dm <<= 1u;
      }
      if (worst_prod > 127LL * static_cast<int64_t>(next_dm) && !changed) return true;
      if (next_dm != dm) {
        *io_d_mul = next_dm;
        changed = true;
      }
    }
    if (!changed) return true;
  }
  return true;
}

void cpu_swiglu_f32_quant_slice(const float* a_slice,
                                const ob_gemm_swiglu_quant_desc_t* d,
                                uint32_t tm,
                                float* c_slice) {
  const uint32_t K = d->k_dim, H = d->h_dim, N = d->n_dim;
  const uint32_t nb_k = K / OB_QK8_0;
  const uint32_t nb_h = H / OB_QK8_0;
  std::vector<float> mid(H, 0.0f);
  for (uint32_t mi = 0u; mi < tm; ++mi) {
    const float* a_row = a_slice + static_cast<size_t>(mi) * K;
    for (uint32_t h = 0u; h < H; ++h) {
      const float g = d->gate_is_q4_0
          ? neon_dot_f32_q4_row(a_row, static_cast<const ob_block_q4_0_t*>(d->w_gate_hk) + static_cast<size_t>(h) * nb_k, nb_k)
          : neon_dot_f32_q8_row(a_row, static_cast<const ob_block_q8_0_t*>(d->w_gate_hk) + static_cast<size_t>(h) * nb_k, nb_k);
      const float u = d->up_is_q4_0
          ? neon_dot_f32_q4_row(a_row, static_cast<const ob_block_q4_0_t*>(d->w_up_hk) + static_cast<size_t>(h) * nb_k, nb_k)
          : neon_dot_f32_q8_row(a_row, static_cast<const ob_block_q8_0_t*>(d->w_up_hk) + static_cast<size_t>(h) * nb_k, nb_k);
      mid[h] = (g / (1.0f + std::exp(-g))) * u;
    }
    float* c_row = c_slice + static_cast<size_t>(mi) * N;
    for (uint32_t n = 0u; n < N; ++n) {
      c_row[n] = d->down_is_q4_0
          ? neon_dot_f32_q4_row(mid.data(), static_cast<const ob_block_q4_0_t*>(d->w_down_nh) + static_cast<size_t>(n) * nb_h, nb_h)
          : neon_dot_f32_q8_row(mid.data(), static_cast<const ob_block_q8_0_t*>(d->w_down_nh) + static_cast<size_t>(n) * nb_h, nb_h);
    }
  }
}

ob_status_t run_swiglu_m7a_fallback(ob_context_t* ctx,
                                    const float* a_slice,
                                    const ob_gemm_swiglu_quant_desc_t* d,
                                    uint32_t tm,
                                    float* c_slice,
                                    ob_invoke_stats_t* out_stats) {
  const uint32_t K = d->k_dim, H = d->h_dim, N = d->n_dim;
  std::vector<float> gate_buf(static_cast<size_t>(tm) * H, 0.0f);
  std::vector<float> up_buf(static_cast<size_t>(tm) * H, 0.0f);

  ob_gemm_f32_quant_job_t gu_jobs[2]{};
  gu_jobs[0].b_nk = d->w_gate_hk;
  gu_jobs[0].c_mn = gate_buf.data();
  gu_jobs[0].n_dim = H;
  gu_jobs[0].activation = OB_ACT_SILU;
  gu_jobs[0].is_q4_0 = d->gate_is_q4_0;
  gu_jobs[1].b_nk = d->w_up_hk;
  gu_jobs[1].c_mn = up_buf.data();
  gu_jobs[1].n_dim = H;
  gu_jobs[1].activation = OB_ACT_NONE;
  gu_jobs[1].is_q4_0 = d->up_is_q4_0;

  ob_invoke_stats_t s1{};
  ob_status_t st = ob_gemm_matmul_f32_quant_batch(ctx, a_slice, tm, K, gu_jobs, 2u, &s1);
  if (st != OB_OK) return st;

  // NEON elementwise gate_silu *= up
  const size_t cnt = gate_buf.size();
  size_t i = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  for (; i + 16u <= cnt; i += 16u) {
    vst1q_f32(gate_buf.data() + i,       vmulq_f32(vld1q_f32(gate_buf.data() + i),       vld1q_f32(up_buf.data() + i)));
    vst1q_f32(gate_buf.data() + i + 4u,  vmulq_f32(vld1q_f32(gate_buf.data() + i + 4u),  vld1q_f32(up_buf.data() + i + 4u)));
    vst1q_f32(gate_buf.data() + i + 8u,  vmulq_f32(vld1q_f32(gate_buf.data() + i + 8u),  vld1q_f32(up_buf.data() + i + 8u)));
    vst1q_f32(gate_buf.data() + i + 12u, vmulq_f32(vld1q_f32(gate_buf.data() + i + 12u), vld1q_f32(up_buf.data() + i + 12u)));
  }
  for (; i + 4u <= cnt; i += 4u) {
    vst1q_f32(gate_buf.data() + i, vmulq_f32(vld1q_f32(gate_buf.data() + i), vld1q_f32(up_buf.data() + i)));
  }
#endif
  for (; i < cnt; ++i) {
    gate_buf[i] *= up_buf[i];
  }

  ob_gemm_f32_quant_job_t d_job{};
  d_job.b_nk = d->w_down_nh;
  d_job.c_mn = c_slice;
  d_job.n_dim = N;
  d_job.activation = OB_ACT_NONE;
  d_job.is_q4_0 = d->down_is_q4_0;

  ob_invoke_stats_t s2{};
  st = ob_gemm_matmul_f32_quant_batch(ctx, gate_buf.data(), tm, H, &d_job, 1u, &s2);
  if (st != OB_OK) return st;

  if (out_stats) {
    out_stats->hw_execution_us += s1.hw_execution_us + s2.hw_execution_us;
    out_stats->tiles_dispatched += s1.tiles_dispatched + s2.tiles_dispatched;
    out_stats->cache_hits += s1.cache_hits + s2.cache_hits;
    out_stats->flags |= (s1.flags | s2.flags);
  }
  return OB_OK;
}

}  // namespace

ob_status_t ob_gemm_swiglu_f32_quant(ob_context_t* ctx,
                                     const float* a_mk,
                                     const ob_gemm_swiglu_quant_desc_t* desc,
                                     float* c_mn,
                                     ob_invoke_stats_t* out_stats) {
  if (!ctx || !a_mk || !desc || !c_mn ||
      !desc->w_gate_hk || !desc->w_up_hk || !desc->w_down_nh ||
      desc->m_dim == 0u || desc->k_dim == 0u || desc->h_dim == 0u || desc->n_dim == 0u ||
      (desc->k_dim % OB_QK8_0) != 0u || (desc->h_dim % OB_QK8_0) != 0u) {
    return OB_ERR_INVALID_ARG;
  }

  const uint64_t t0_ns = ob::monotonic_now_ns();
  const uint32_t M = desc->m_dim;
  const uint32_t K = desc->k_dim;
  const uint32_t H = desc->h_dim;
  const uint32_t N = desc->n_dim;

  bool can_chain_shape = false;
  bool prefill_chain_on = false;
  bool builtin_active = false;
  bool prof_on = false;
  bool strict_hash = false;
  float max_crest = 20.0f;
  uint32_t param_swap_slots = 1u;
  uint32_t ring_depth = 1u;
  {
    std::lock_guard<std::mutex> lk(ctx->gemm_mu);
    builtin_active = ctx->builtin_emitter_active && !ctx->builtin_dive_interp.empty();
    prof_on = ctx->profile_enabled;
    strict_hash = ctx->strict_weight_hash;
    max_crest = ctx->swiglu_max_crest;
    prefill_chain_on = ctx->swiglu_prefill_enabled && ctx->batched_large_tiles_enabled;
    param_swap_slots = ctx->param_swap_slots;
    ring_depth = std::max<uint32_t>(1u, std::min<uint32_t>(ctx->credit_size, OB_MAX_CREDITS));
    can_chain_shape = ctx->swiglu_chain_enabled && builtin_active &&
                      ((K == 576u && N == 576u) || (K == 1536u && N == 1536u));
  }

  // Default: chain M==1 decode; route tiny prompts (2<=M<=8) to exact NEON FP32 CPU (preserves
  // exact BOS/attention-sink KV cache precision with 0 B weight-cache allocation!); keep M>8
  // prefill on faster lt_64x* GEMM unless opt-in OPENBUENOS_SWIGLU_PREFILL=1 is enabled.
  const bool any_q4 =
      (desc->gate_is_q4_0 != 0u) || (desc->up_is_q4_0 != 0u) || (desc->down_is_q4_0 != 0u);
  if (!can_chain_shape || (M > 8u && (!prefill_chain_on || any_q4))) {
    ob_invoke_stats_t fb{};
    ob_status_t st = run_swiglu_m7a_fallback(ctx, a_mk, desc, M, c_mn, &fb);
    if (st == OB_OK) {
      std::lock_guard<std::mutex> lk(ctx->gemm_mu);
      ctx->prof.swiglu_calls += 1u;
      ctx->prof.swiglu_fallbacks += 1u;
    }
    if (out_stats) {
      *out_stats = fb;
      out_stats->wall_time_us = (ob::monotonic_now_ns() - t0_ns) / 1000ULL;
    }
    return st;
  }

  if (!prefill_chain_on && M >= 2u && M <= 8u) {
    cpu_swiglu_f32_quant_slice(a_mk, desc, M, c_mn);
    const uint64_t wall_ns = ob::monotonic_now_ns() - t0_ns;
    {
      std::lock_guard<std::mutex> lk(ctx->gemm_mu);
      ctx->prof.swiglu_calls += 1u;
      ctx->prof.swiglu_fallbacks += 1u;
      ctx->prof.total_wall_ns += wall_ns;
    }
    if (out_stats) {
      std::memset(out_stats, 0, sizeof(*out_stats));
      out_stats->wall_time_us = wall_ns / 1000ULL;
      out_stats->flags = static_cast<uint16_t>(
          OB_STAT_FLAG_CPU_FALLBACK |
          (ctx->mock_mode ? static_cast<uint32_t>(OB_STAT_FLAG_MOCK) : 0u));
    }
    return OB_OK;
  }

  // Partition M into supported chained slices:
  //   - M == 1: single 1-row chained decode slice
  //   - prefill_chain_on == true: 64-row chained blocks + 1-row tail or M7A tail
  struct SwigluMSlice {
    uint32_t m0;
    uint32_t tm;
    uint32_t hw_m; // 1 or 64 if chained, 0 if M7A fallback
  };
  std::vector<SwigluMSlice> m_slices;
  for (uint32_t m0 = 0u; m0 < M; ) {
    const uint32_t rem = M - m0;
    if (rem == 1u) {
      m_slices.push_back({m0, 1u, 1u});
      m0 += 1u;
    } else if (rem >= 16u && prefill_chain_on && !any_q4) {
      const uint32_t tm = std::min<uint32_t>(64u, rem);
      m_slices.push_back({m0, tm, 64u});
      m0 += tm;
    } else {
      m_slices.push_back({m0, rem, 0u});
      m0 += rem;
    }
  }

  uint64_t total_hw_us = 0u;
  uint64_t chained_hw_us = 0u;
  uint64_t fb_wall_ns = 0u;
  bool ran_cpu_fb_slice = false;
  uint32_t total_tiles = 0u;
  uint32_t total_hits = 0u;
  uint64_t chained_chunks = 0u;
  uint64_t fb_slices = 0u;
  uint16_t agg_flags = OB_STAT_FLAG_EMITTER_USED;
  if (ctx->mock_mode) agg_flags |= OB_STAT_FLAG_MOCK;

  const uint32_t nb_k = K / OB_QK8_0;
  const uint32_t nb_h = H / OB_QK8_0;
  const size_t g_row_bytes = static_cast<size_t>(nb_k) *
      (desc->gate_is_q4_0 ? sizeof(ob_block_q4_0_t) : sizeof(ob_block_q8_0_t));
  const size_t u_row_bytes = static_cast<size_t>(nb_k) *
      (desc->up_is_q4_0 ? sizeof(ob_block_q4_0_t) : sizeof(ob_block_q8_0_t));
  const size_t d_row_bytes = static_cast<size_t>(nb_h) *
      (desc->down_is_q4_0 ? sizeof(ob_block_q4_0_t) : sizeof(ob_block_q8_0_t));

  for (const SwigluMSlice& ms : m_slices) {
    const float* a_slice = a_mk + static_cast<size_t>(ms.m0) * K;
    float* c_slice = c_mn + static_cast<size_t>(ms.m0) * N;

    // Inspect activation dynamic range & crest factor before entering chained path
    bool need_fallback = (ms.hw_m == 0u);
    bool transient_m1_outlier = false;
    float batch_amax = 0.0f;
    double max_row_rms = 0.0;
    double max_row_abs_mean = 0.0;
    float max_row_crest = 0.0f;
    if (!need_fallback) {
      float min_row_amax = 1e30f;
      for (uint32_t mi = 0u; mi < ms.tm; ++mi) {
        const float* a_row = a_slice + static_cast<size_t>(mi) * K;
        float r_amax = 0.0f;
        double r_sq = 0.0;
        double r_sum = 0.0;
        uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t v_am = vdupq_n_f32(0.0f);
        float64x2_t v_sq = vdupq_n_f64(0.0);
        float64x2_t v_sm = vdupq_n_f64(0.0);
        for (; ki + 4u <= K; ki += 4u) {
          float32x4_t v = vld1q_f32(a_row + ki);
          v_am = vmaxq_f32(v_am, vabsq_f32(v));
          float64x2_t lo = vcvt_f64_f32(vget_low_f32(v));
          float64x2_t hi = vcvt_high_f64_f32(v);
          v_sq = vfmaq_f64(v_sq, lo, lo);
          v_sq = vfmaq_f64(v_sq, hi, hi);
          v_sm = vaddq_f64(v_sm, vaddq_f64(lo, hi));
        }
        r_amax = vmaxvq_f32(v_am);
        r_sq = vaddvq_f64(v_sq);
        r_sum = vaddvq_f64(v_sm);
#endif
        for (; ki < K; ++ki) {
          float v = a_row[ki];
          float av = std::fabs(v);
          if (av > r_amax) r_amax = av;
          r_sq += static_cast<double>(v) * static_cast<double>(v);
          r_sum += static_cast<double>(v);
        }
        if (r_amax > batch_amax) batch_amax = r_amax;
        if (r_amax > 0.0f && r_amax < min_row_amax) min_row_amax = r_amax;
        if (r_amax > 0.0f) {
          double rms = std::sqrt(r_sq / static_cast<double>(K));
          if (rms > max_row_rms) max_row_rms = rms;
          double abs_mean = std::fabs(r_sum) / static_cast<double>(K);
          if (abs_mean > max_row_abs_mean) max_row_abs_mean = abs_mean;
          float crest = static_cast<float>(static_cast<double>(r_amax) / std::max(rms, 1e-12));
          if (crest > max_row_crest) max_row_crest = crest;
        }
      }
      if (max_crest > 0.0f && max_row_crest > max_crest) {
        need_fallback = true;
        if (ms.tm == 1u) transient_m1_outlier = true;
      }
      if (ms.tm > 1u && batch_amax > 0.0f && min_row_amax < 1e29f &&
          batch_amax > 6.0f * min_row_amax) {
        need_fallback = true;
      }
    }

    if (need_fallback) {
      if (transient_m1_outlier) {
        // Rare M=1 attention-sink outlier token: run exact NEON FP32 on CPU so we do NOT
        // allocate 41 MiB of duplicate M7A ext[2] tiles into weight_cache (prevents Qwen2.5
        // LRU thrashing while delivering 0-error FP32 precision on sink tokens!).
        cpu_swiglu_f32_quant_slice(a_slice, desc, 1u, c_slice);
        agg_flags |= OB_STAT_FLAG_CPU_FALLBACK;
        fb_slices += 1u;
        ran_cpu_fb_slice = true;
        continue;
      }
      ob_invoke_stats_t fb{};
      const uint64_t fbt0 = ob::monotonic_now_ns();
      ob_status_t st = run_swiglu_m7a_fallback(ctx, a_slice, desc, ms.tm, c_slice, &fb);
      fb_wall_ns += (ob::monotonic_now_ns() - fbt0);
      if (st != OB_OK) return st;
      total_hw_us += fb.hw_execution_us;
      total_tiles += fb.tiles_dispatched;
      total_hits += fb.cache_hits;
      agg_flags |= fb.flags;
      fb_slices += 1u;
      continue;
    }

    if (batch_amax == 0.0f) {
      std::memset(c_slice, 0, static_cast<size_t>(ms.tm) * N * sizeof(float));
      continue;
    }

    // Build H-chunks for this (hw_m, K, N)
    struct HChunk {
      uint32_t h0;
      uint32_t th;
      uint32_t hw_h;
    };
    std::vector<HChunk> h_chunks;
    if (ms.hw_m == 1u && K == 576u && H == 1536u) {
      // SmolLM2-135M FULL decode MLP in 1 single VII call!
      h_chunks.push_back({0u, 1536u, 1536u});
    } else if (ms.hw_m == 1u && K == 1536u) {
      for (uint32_t h0 = 0u; h0 < H; ) {
        const uint32_t rem = H - h0;
        if (rem <= 512u) {
          h_chunks.push_back({h0, rem, 512u});
          h0 += rem;
        } else {
          const uint32_t th = std::min<uint32_t>(1024u, rem);
          h_chunks.push_back({h0, th, 1024u});
          h0 += th;
        }
      }
    } else {
      for (uint32_t h0 = 0u; h0 < H; h0 += 512u) {
        const uint32_t th = std::min<uint32_t>(512u, H - h0);
        h_chunks.push_back({h0, th, 512u});
      }
    }

    bool slice_rejected = false;
    {
      std::lock_guard<std::mutex> lk(ctx->gemm_mu);

      // Quantize shared A [hw_m, K] ONCE on the exact 2^(1/2) scale grid!
      const uint64_t tq0 = prof_on ? ob::monotonic_now_ns() : 0u;
      const float s_a = ob::quantize_scale_pow2_2nd_up(batch_amax / 127.0f);
      const float inv_sa = 1.0f / s_a;
      std::vector<int8_t>& a_q = ctx->gemm_aq_scratch;
      a_q.assign(static_cast<size_t>(ms.hw_m) * K, 0);
      for (uint32_t mi = 0u; mi < ms.tm; ++mi) {
        ob::quant_sym_i8_row(a_slice + static_cast<size_t>(mi) * K,
                             a_q.data() + static_cast<size_t>(mi) * K,
                             K, inv_sa);
      }
      uint32_t amax_aq = 0u;
      uint64_t max_l1_aq = 0ULL;
      for (uint32_t mi = 0u; mi < ms.tm; ++mi) {
        const int8_t* aq_row = a_q.data() + static_cast<size_t>(mi) * K;
        uint64_t r_l1 = 0ULL;
        for (uint32_t ki = 0u; ki < K; ++ki) {
          const int32_t qv = static_cast<int32_t>(aq_row[ki]);
          const uint32_t av = static_cast<uint32_t>(qv < 0 ? -qv : qv);
          r_l1 += av;
          if (av > amax_aq) amax_aq = av;
        }
        if (r_l1 > max_l1_aq) max_l1_aq = r_l1;
      }
      if (prof_on) ctx->prof.quant_ns += (ob::monotonic_now_ns() - tq0);

      // Upload shared input DMA-BUF ONCE for all H-chunks in this M-slice!
      ob_buffer* shared_in_buf = nullptr;
      ob_status_t st = ob::ensure_pipeline_buf_tiered(
          ctx, &ctx->gemm_in_bufs_4k[0], &ctx->gemm_in_bufs[0], a_q.size(), &shared_in_buf);
      if (st != OB_OK) return st;
      ob::ScopedBufferRef shared_in_guard(shared_in_buf);
      {
        const uint64_t td0 = prof_on ? ob::monotonic_now_ns() : 0u;
        st = ob_buffer_write(shared_in_buf, a_q.data(), a_q.size(), 0u);
        if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td0);
        if (st != OB_OK) return st;
      }

      std::memset(c_slice, 0, static_cast<size_t>(ms.tm) * N * sizeof(float));

      struct PendingSwigluChunk {
        ob_exec_handle_t handle = 0u;
        ob::CachedWeightTile* w_tile = nullptr;
        ob_buffer* in_buf = nullptr;
        ob_buffer* out_buf = nullptr;
        uint32_t slot = 0u;
        uint32_t hw_h = 0u;
        uint32_t d_gate = 0u, d_up = 0u, d_mul = 0u, d_down = 0u;
        float s_gate = 0.0f;
      };
      std::array<PendingSwigluChunk, OB_MAX_CREDITS> ring{};
      uint32_t ring_head = 0u;
      uint32_t ring_count = 0u;

      auto drain_all_cleanup = [&]() {
        while (ring_count > 0u) {
          PendingSwigluChunk& p = ring[ring_head];
          ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, nullptr);
          if (wst != OB_OK) {
            ob::detach_timed_out_buffers_locked(ctx, p.handle, p.w_tile, p.in_buf, p.out_buf);
          }
          if (p.w_tile) {
            if (p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
            if (!p.w_tile->in_cache && p.w_tile->in_use == 0u) {
              if (p.w_tile->ext2_buf) ob_buffer_free(p.w_tile->ext2_buf);
              delete p.w_tile;
            }
          }
          if (p.in_buf) ob_buffer_free(p.in_buf);
          if (p.out_buf) ob_buffer_free(p.out_buf);
          p = PendingSwigluChunk{};
          ring_head = (ring_head + 1u) % ring_depth;
          ring_count -= 1u;
        }
      };

      auto make_graph_desc = [&](uint32_t hw_h, uint32_t d_gate, uint32_t d_up,
                                 uint32_t d_mul, uint32_t d_down, float s_gate) {
        ob_swiglu_int8_desc_t gd{};
        gd.m_dim = ms.hw_m;
        gd.k_dim = K;
        gd.h_dim = hw_h;
        gd.n_dim = N;
        gd.s_in = 1.0f;
        gd.s_gate = s_gate;
        gd.s_wg = s_gate / static_cast<float>(d_gate);
        gd.s_wu = 1.0f;
        gd.s_up = static_cast<float>(d_up);
        gd.s_silu = s_gate;
        gd.s_prod = gd.s_silu * gd.s_up * static_cast<float>(d_mul);
        gd.s_wd = 1.0f / gd.s_prod;
        gd.s_out = static_cast<float>(d_down);
        return gd;
      };

      auto drain_one = [&]() -> ob_status_t {
        PendingSwigluChunk& p = ring[ring_head];
        ob_invoke_stats_t istats{};
        const uint64_t tw0 = prof_on ? ob::monotonic_now_ns() : 0u;
        ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, &istats);
        if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - tw0);
        ob_status_t rst = OB_OK;
        if (wst == OB_OK) {
          total_hw_us += istats.hw_execution_us;
          chained_hw_us += istats.hw_execution_us;
          if (!ctx->mock_mode) {
            uint32_t eff_d_down = p.d_down;
            const uint64_t td0 = prof_on ? ob::monotonic_now_ns() : 0u;
            rst = ob::buffer_sync_begin_read(p.out_buf);
            if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td0);
            if (rst == OB_OK) {
              const int8_t* raw_out = static_cast<const int8_t*>(ob_buffer_data(p.out_buf));
              bool read_active = true;
              for (uint32_t retry = 0u; retry < 4u && eff_d_down < (1u << 23); ++retry) {
                const uint64_t tsat0 = prof_on ? ob::monotonic_now_ns() : 0u;
                const bool sat = ob::tile_output_is_saturated_impl(raw_out, ms.tm, N, N);
                if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tsat0);
                if (!sat) break;

                const uint64_t tde0 = prof_on ? ob::monotonic_now_ns() : 0u;
                ob::buffer_sync_end_read(p.out_buf);
                if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - tde0);
                read_active = false;

                ctx->prof.sat_retries += 1u;
                eff_d_down <<= 1u;
                const uint64_t rkey = ob::hash_swiglu_graph_key(
                    ms.hw_m, K, p.hw_h, N, p.d_gate, p.d_up, p.d_mul, eff_d_down,
                    p.s_gate, p.slot % param_swap_slots);
                ob_swiglu_int8_desc_t rdesc =
                    make_graph_desc(p.hw_h, p.d_gate, p.d_up, p.d_mul, eff_d_down, p.s_gate);
                const uint64_t trs0 = prof_on ? ob::monotonic_now_ns() : 0u;
                ob::CachedGraph* rentry = nullptr;
                bool rhit = false;
                rst = ob::acquire_swiglu_entry(ctx, rdesc, rkey, &rentry, &rhit);
                if (rst != OB_OK) break;
                if (rhit) total_hits += 1u;
                total_tiles += 1u;
                chained_chunks += 1u;
                ob_exec_handle_t rh = 0u;
                rst = ob::device_submit_execute_async(
                    ctx, rentry, &p.in_buf, 1u, &p.out_buf, 1u, &rh, p.w_tile->ext2_buf);
                if (prof_on) ctx->prof.submit_ns += (ob::monotonic_now_ns() - trs0);
                if (rst != OB_OK) {
                  ob::release_tile_entry(ctx, rentry);
                  break;
                }
                ob_invoke_stats_t rstats{};
                const uint64_t trw0 = prof_on ? ob::monotonic_now_ns() : 0u;
                rst = ob::device_wait_handle(ctx, rh, ctx->cfg.timeout_ms, &rstats);
                if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - trw0);
                if (rst != OB_OK) {
                  ob::detach_timed_out_buffers_locked(ctx, rh, p.w_tile, p.in_buf, p.out_buf);
                  break;
                }
                total_hw_us += rstats.hw_execution_us;
                chained_hw_us += rstats.hw_execution_us;

                const uint64_t tdr0 = prof_on ? ob::monotonic_now_ns() : 0u;
                rst = ob::buffer_sync_begin_read(p.out_buf);
                if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - tdr0);
                if (rst != OB_OK) break;
                raw_out = static_cast<const int8_t*>(ob_buffer_data(p.out_buf));
                read_active = true;
              }
              if (rst == OB_OK && read_active) {
                const uint64_t tq1 = prof_on ? ob::monotonic_now_ns() : 0u;
                // Y_f32[m, n] += q_out[m, n] * (s_silu * s_a * d_up * d_mul * eff_d_down) * row_s_d[n]
                const float out_mult = p.s_gate * s_a * static_cast<float>(p.d_up) *
                                       static_cast<float>(p.d_mul) * static_cast<float>(eff_d_down);
                const float* sd_ptr = p.w_tile->row_s_b.data();
                for (uint32_t mi = 0u; mi < ms.tm; ++mi) {
                  float* c_row = c_slice + static_cast<size_t>(mi) * N;
                  const int8_t* o_row = raw_out + static_cast<size_t>(mi) * N;
                  ob::dequant_accum_row_f32_impl(c_row, o_row, sd_ptr, N, out_mult);
                }
                if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tq1);
              }
              if (read_active) {
                const uint64_t td1 = prof_on ? ob::monotonic_now_ns() : 0u;
                ob_status_t est = ob::buffer_sync_end_read(p.out_buf);
                if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td1);
                if (rst == OB_OK) rst = est;
              }
            }
          }
        } else {
          ob::detach_timed_out_buffers_locked(ctx, p.handle, p.w_tile, p.in_buf, p.out_buf);
        }
        if (p.w_tile) {
          if (p.w_tile->in_use > 0u) p.w_tile->in_use -= 1u;
          if (!p.w_tile->in_cache && p.w_tile->in_use == 0u) {
            if (p.w_tile->ext2_buf) ob_buffer_free(p.w_tile->ext2_buf);
            delete p.w_tile;
          }
        }
        if (p.in_buf) ob_buffer_free(p.in_buf);
        if (p.out_buf) ob_buffer_free(p.out_buf);
        p = PendingSwigluChunk{};
        ring_head = (ring_head + 1u) % ring_depth;
        ring_count -= 1u;
        return (wst != OB_OK) ? wst : rst;
      };

      for (const HChunk& hc : h_chunks) {
        const uint32_t h0 = hc.h0;
        const uint32_t th = hc.th;
        const uint32_t hw_h = hc.hw_h;
        const uint32_t h_blk0 = h0 / OB_QK8_0;
        const uint32_t h_blk_cnt = th / OB_QK8_0;

        // 1. O(1) content hash of (W_gate[h0..], W_up[h0..], W_down[:, h0..])
        const uint64_t th0 = prof_on ? ob::monotonic_now_ns() : 0u;
        ob::WeightHash128 wh_st{};
        const uint64_t wh_tag = 0x5357474C555F5731ULL; // "SWGLU_W1"
        uint32_t whdr[8] = {
            ms.hw_m, K, th, hw_h, N,
            desc->gate_is_q4_0, desc->up_is_q4_0, desc->down_is_q4_0
        };
        ob::hash128_update(&wh_st, &wh_tag, sizeof(wh_tag));
        ob::hash128_update(&wh_st, whdr, sizeof(whdr));
        const uint8_t* g_ptr = static_cast<const uint8_t*>(desc->w_gate_hk) +
                               static_cast<size_t>(h0) * g_row_bytes;
        const uint8_t* u_ptr = static_cast<const uint8_t*>(desc->w_up_hk) +
                               static_cast<size_t>(h0) * u_row_bytes;
        const size_t d_blk_sz =
            desc->down_is_q4_0 ? sizeof(ob_block_q4_0_t) : sizeof(ob_block_q8_0_t);
        const uint8_t* d_ptr = static_cast<const uint8_t*>(desc->w_down_nh) +
                               static_cast<size_t>(h_blk0) * d_blk_sz;
        ob::hash_weight_tile_for_ctx(ctx, &wh_st, g_ptr, g_row_bytes, th, g_row_bytes, strict_hash);
        ob::hash_weight_tile_for_ctx(ctx, &wh_st, u_ptr, u_row_bytes, th, u_row_bytes, strict_hash);
        ob::hash_weight_tile_for_ctx(ctx, &wh_st, d_ptr, d_row_bytes, N,
                                     static_cast<size_t>(h_blk_cnt) * d_blk_sz, strict_hash);
        const ob::WeightTileKey wh = ob::finalize_weight_key(wh_st);
        ob::CachedWeightTile* w_tile = ob::weight_cache_find_locked(ctx, wh);
        const bool was_cache_hit = (w_tile != nullptr);
        if (prof_on) ctx->prof.hash_ns += (ob::monotonic_now_ns() - th0);

        if (w_tile && w_tile->rejected_swiglu) {
          drain_all_cleanup();
          slice_rejected = true;
          break;
        }

        std::vector<int8_t> cold_wg_q;
        std::vector<int8_t> cold_wu_q;
        if (!w_tile) {
          const uint64_t tp0 = prof_on ? ob::monotonic_now_ns() : 0u;
          // Cold pack:
          //   a) W_up per-row -> wu_q[hw_h, K] + row_s_u[th]
          std::vector<int8_t> wu_q(static_cast<size_t>(hw_h) * K, 0);
          std::vector<float> row_s_u(th, 0.0f);
          if (desc->up_is_q4_0 != 0u) {
            ob::requant_q4_0_rows(static_cast<const ob_block_q4_0_t*>(desc->w_up_hk),
                                  nb_k, h0, th, 0u, nb_k, K, K,
                                  wu_q.data(), row_s_u.data(), &ctx->gemm_wrow_scratch);
          } else {
            ob::requant_q8_0_rows(static_cast<const ob_block_q8_0_t*>(desc->w_up_hk),
                                  nb_k, h0, th, 0u, nb_k, K, K,
                                  wu_q.data(), row_s_u.data(), &ctx->gemm_wrow_scratch);
          }

          //   b) W_down with row_s_u[h] absorbed -> wd_q[N, hw_h] + row_s_d[N]
          std::vector<int8_t> wd_q(static_cast<size_t>(N) * hw_h, 0);
          std::vector<float> row_s_d(N, 0.0f);
          ctx->gemm_wrow_scratch.resize(th);
          float* wrow_d = ctx->gemm_wrow_scratch.data();
          for (uint32_t ni = 0u; ni < N; ++ni) {
            dequant_quant_row_f32(desc->w_down_nh, desc->down_is_q4_0,
                                  nb_h, ni, h_blk0, h_blk_cnt, wrow_d);
            float amax_d = 0.0f;
            uint32_t hi = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t vam = vdupq_n_f32(0.0f);
            for (; hi + 4u <= th; hi += 4u) {
              float32x4_t v = vmulq_f32(vld1q_f32(wrow_d + hi), vld1q_f32(row_s_u.data() + hi));
              vst1q_f32(wrow_d + hi, v);
              vam = vmaxq_f32(vam, vabsq_f32(v));
            }
            amax_d = vmaxvq_f32(vam);
#endif
            for (; hi < th; ++hi) {
              float v = wrow_d[hi] * row_s_u[hi];
              wrow_d[hi] = v;
              float av = std::fabs(v);
              if (av > amax_d) amax_d = av;
            }
            if (amax_d > 0.0f) {
              row_s_d[ni] = amax_d / 127.0f;
              ob::quant_sym_i8_row(wrow_d, wd_q.data() + static_cast<size_t>(ni) * hw_h,
                                   th, 127.0f / amax_d);
            }
          }

          //   c) W_gate with common 2^(1/2)-aligned per-chunk scale s_wg -> wg_q[hw_h, K]
          std::vector<float> wg_f32(static_cast<size_t>(th) * K, 0.0f);
          float max_wg = 0.0f;
          double sum_row_max_wg = 0.0;
          for (uint32_t hi = 0u; hi < th; ++hi) {
            float* dst_row = wg_f32.data() + static_cast<size_t>(hi) * K;
            dequant_quant_row_f32(desc->w_gate_hk, desc->gate_is_q4_0,
                                  nb_k, h0 + hi, 0u, nb_k, dst_row);
            float rm = 0.0f;
            uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t vam = vdupq_n_f32(0.0f);
            for (; ki + 4u <= K; ki += 4u) {
              vam = vmaxq_f32(vam, vabsq_f32(vld1q_f32(dst_row + ki)));
            }
            rm = vmaxvq_f32(vam);
#endif
            for (; ki < K; ++ki) {
              float av = std::fabs(dst_row[ki]);
              if (av > rm) rm = av;
            }
            if (rm > max_wg) max_wg = rm;
            sum_row_max_wg += static_cast<double>(rm);
          }
          const double mean_row_max_wg = sum_row_max_wg / static_cast<double>(std::max<uint32_t>(1u, th));
          if (max_wg > 0.0f && static_cast<double>(max_wg) > 16.0 * std::max(mean_row_max_wg, 1e-12)) {
            bool sentinel_ok = false;
            if (ob::weight_cache_prepare_space_locked(ctx, 0u, ring_count, drain_one, &sentinel_ok) == OB_OK &&
                sentinel_ok) {
              ob::CachedWeightTile* sentinel = new (std::nothrow) ob::CachedWeightTile();
              if (sentinel) {
                sentinel->key = wh;
                sentinel->ext2_bytes = 0u;
                sentinel->rejected_swiglu = true;
                ob::weight_cache_insert_locked(ctx, sentinel);
              }
            }
            if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
            drain_all_cleanup();
            slice_rejected = true;
            break;
          }

          const float s_wg = (max_wg > 0.0f) ? ob::quantize_scale_pow2_2nd_up(max_wg / 127.0f) : 1.0f;
          const float inv_swg = (max_wg > 0.0f) ? (1.0f / s_wg) : 0.0f;
          std::vector<int8_t> wg_q(static_cast<size_t>(hw_h) * K, 0);
          if (max_wg > 0.0f) {
            for (uint32_t hi = 0u; hi < th; ++hi) {
              ob::quant_sym_i8_row(wg_f32.data() + static_cast<size_t>(hi) * K,
                                   wg_q.data() + static_cast<size_t>(hi) * K,
                                   K, inv_swg);
            }
          }

          //   d) Record per-stage L2, DC-sum, L1, and Linf weight norms once for runtime divider selection & exact saturation guards!
          float l2_g_max = 0.0f, l2_g_mean = 0.0f, dc_g_max = 0.0f, l1_g_max = 0.0f, linf_g_max = 0.0f;
          float l2_u_max = 0.0f, l2_u_mean = 0.0f, dc_u_max = 0.0f, l1_u_max = 0.0f, linf_u_max = 0.0f;
          float l2_d_max = 0.0f, l2_d_mean = 0.0f;
          ob::compute_matrix_row_l2_stats(wg_q.data(), th, K, K, &l2_g_max, &l2_g_mean, &dc_g_max, &l1_g_max, &linf_g_max);
          ob::compute_matrix_row_l2_stats(wu_q.data(), th, K, K, &l2_u_max, &l2_u_mean, &dc_u_max, &l1_u_max, &linf_u_max);
          ob::compute_matrix_row_l2_stats(wd_q.data(), N, th, hw_h, &l2_d_max, &l2_d_mean);

          // Reject statistically non-zero-mean gate/up weight tiles where correlated inputs
          // could saturate intermediate INT8 stages silently despite unsaturated down output.
          if ((l2_g_max > 0.0f && dc_g_max > 5.5f * l2_g_max) ||
              (l2_u_max > 0.0f && dc_u_max > 5.5f * l2_u_max)) {
            bool sentinel_ok = false;
            if (ob::weight_cache_prepare_space_locked(ctx, 0u, ring_count, drain_one, &sentinel_ok) == OB_OK &&
                sentinel_ok) {
              ob::CachedWeightTile* sentinel = new (std::nothrow) ob::CachedWeightTile();
              if (sentinel) {
                sentinel->key = wh;
                sentinel->ext2_bytes = 0u;
                sentinel->rejected_swiglu = true;
                ob::weight_cache_insert_locked(ctx, sentinel);
              }
            }
            if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
            drain_all_cleanup();
            slice_rejected = true;
            break;
          }

          std::vector<uint8_t> ext2 = ob::pack_ext2_for_swiglu(
              ms.hw_m, K, hw_h, N,
              wg_q.data(), nullptr,
              wu_q.data(), nullptr,
              wd_q.data(), nullptr);
          if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);

          bool cacheable = false;
          ob_status_t pst = ob::weight_cache_prepare_space_locked(
              ctx, ext2.size(), ring_count, drain_one, &cacheable);
          if (pst != OB_OK) {
            drain_all_cleanup();
            return pst;
          }

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
          w_tile->hw_k = K;
          w_tile->hw_n = N;
          w_tile->row_s_b = std::move(row_s_d);
          w_tile->ext2_buf = wbuf;
          w_tile->ext2_bytes = ext2.size();
          w_tile->swiglu_s_wg = s_wg;
          w_tile->swiglu_l2_gate_max = l2_g_max;
          w_tile->swiglu_l2_gate_mean = l2_g_mean;
          w_tile->swiglu_l2_up_max = l2_u_max;
          w_tile->swiglu_l2_up_mean = l2_u_mean;
          w_tile->swiglu_l2_down_max = l2_d_max;
          w_tile->swiglu_dc_gate_max = dc_g_max;
          w_tile->swiglu_dc_up_max = dc_u_max;
          w_tile->swiglu_l1_gate_max = l1_g_max;
          w_tile->swiglu_linf_gate_max = linf_g_max;
          w_tile->swiglu_l1_up_max = l1_u_max;
          w_tile->swiglu_linf_up_max = linf_u_max;
          if (cacheable) {
            ob::weight_cache_insert_locked(ctx, w_tile);
          } else {
            w_tile->in_cache = false;
          }
          cold_wg_q = std::move(wg_q);
          cold_wu_q = std::move(wu_q);
        }

        auto discard_unsubmitted_wtile = [&]() {
          if (w_tile && !w_tile->in_cache && w_tile->in_use == 0u) {
            if (w_tile->ext2_buf) ob_buffer_free(w_tile->ext2_buf);
            delete w_tile;
            w_tile = nullptr;
          }
        };

        const float rms_aq = static_cast<float>(max_row_rms / static_cast<double>(s_a));
        uint32_t d_gate = 0u, d_up = 0u, d_mul = 0u, d_down = 0u;
        ob::compute_swiglu_stage_dividers(
            ms.tm, rms_aq,
            w_tile->swiglu_l2_gate_max, w_tile->swiglu_l2_gate_mean,
            w_tile->swiglu_l2_up_max, w_tile->swiglu_l2_up_mean,
            w_tile->swiglu_l2_down_max,
            &d_gate, &d_up, &d_mul, &d_down);

        // Guard against silent intermediate INT8 clipping in gate/up/mul stages on correlated
        // or low-crest dense activations before submitting any VII chunks.
        {
          const double mean_aq = max_row_abs_mean / static_cast<double>(s_a);
          const double dc_gate_term = mean_aq * static_cast<double>(w_tile->swiglu_dc_gate_max);
          const double dc_up_term = mean_aq * static_cast<double>(w_tile->swiglu_dc_up_max);
          const double l2_aq = static_cast<double>(rms_aq) * std::sqrt(static_cast<double>(K));
          const double ub_g = std::min({
              l2_aq * static_cast<double>(w_tile->swiglu_l2_gate_max),
              static_cast<double>(max_l1_aq) * static_cast<double>(w_tile->swiglu_linf_gate_max),
              static_cast<double>(amax_aq) * static_cast<double>(w_tile->swiglu_l1_gate_max),
          });
          const double ub_u = std::min({
              l2_aq * static_cast<double>(w_tile->swiglu_l2_up_max),
              static_cast<double>(max_l1_aq) * static_cast<double>(w_tile->swiglu_linf_up_max),
              static_cast<double>(amax_aq) * static_cast<double>(w_tile->swiglu_l1_up_max),
          });
          bool reject_clip = (rms_aq > 56.0f) ||
                             (dc_gate_term > 64.0 * static_cast<double>(d_gate)) ||
                             (dc_up_term > 64.0 * static_cast<double>(d_up));
          if (!reject_clip) {
            bool holder_proven = false;
            if (ub_g <= 127.0 * static_cast<double>(d_gate) &&
                ub_u <= 127.0 * static_cast<double>(d_up)) {
              const int64_t max_qg = std::min<int64_t>(
                  127LL, static_cast<int64_t>(std::ceil(ub_g / static_cast<double>(d_gate))));
              const int64_t max_qu = std::min<int64_t>(
                  127LL, static_cast<int64_t>(std::ceil(ub_u / static_cast<double>(d_up))));
              const float s_gate_init = s_a * w_tile->swiglu_s_wg * static_cast<float>(d_gate);
              int64_t max_qsilu = std::abs(
                  ob::eval_swiglu_silu_q(static_cast<int32_t>(max_qg), s_gate_init));
              max_qsilu = std::max<int64_t>(
                  max_qsilu,
                  std::abs(ob::eval_swiglu_silu_q(static_cast<int32_t>(-max_qg), s_gate_init)));
              if (s_gate_init > 0.0f && max_qg > 0LL) {
                const int64_t q_min_cand = std::max<int64_t>(
                    -max_qg,
                    std::min<int64_t>(0LL, -static_cast<int64_t>(
                                               std::llround(1.2784645f / s_gate_init))));
                max_qsilu = std::max<int64_t>(
                    max_qsilu,
                    std::abs(ob::eval_swiglu_silu_q(static_cast<int32_t>(q_min_cand), s_gate_init)));
              }
              if (max_qg > 0LL) {
                max_qsilu = std::min<int64_t>(127LL, std::max<int64_t>(max_qsilu, max_qg + 1LL));
              }
              const int64_t max_prod = max_qsilu * max_qu;
              uint32_t next_dm = d_mul;
              while (next_dm < 64u && max_prod > 127LL * static_cast<int64_t>(next_dm)) {
                next_dm <<= 1u;
              }
              if (max_prod <= 127LL * static_cast<int64_t>(next_dm)) {
                d_mul = next_dm;
                holder_proven = true;
              }
            }
            const bool need_exact_check =
                !holder_proven &&
                (!was_cache_hit ||
                 ctx->swiglu_exact_clip_check ||
                 w_tile->swiglu_needs_exact_check ||
                 rms_aq > 42.0f ||
                 max_row_crest > 10.0f ||
                 dc_gate_term > 16.0 * static_cast<double>(d_gate) ||
                 dc_up_term > 16.0 * static_cast<double>(d_up));
            if (need_exact_check) {
              const uint32_t orig_dg = d_gate;
              const uint32_t orig_du = d_up;
              const uint32_t orig_dm = d_mul;
              const int8_t* cold_wg_ptr = cold_wg_q.empty() ? nullptr : cold_wg_q.data();
              const int8_t* cold_wu_ptr = cold_wu_q.empty() ? nullptr : cold_wu_q.data();
              reject_clip = swiglu_chunk_clips_or_tunes(
                  ctx, w_tile, cold_wg_ptr, cold_wu_ptr, a_q.data(),
                  ms.tm, ms.hw_m, K, hw_h, th, s_a, w_tile->swiglu_s_wg,
                  &d_gate, &d_up, &d_mul);
              if (!reject_clip &&
                  (d_gate != orig_dg || d_up != orig_du || d_mul != orig_dm)) {
                w_tile->swiglu_needs_exact_check = true;
              }
            }
          }
          if (reject_clip) {
            discard_unsubmitted_wtile();
            drain_all_cleanup();
            slice_rejected = true;
            break;
          }
        }
        const float s_gate_cand = s_a * w_tile->swiglu_s_wg * static_cast<float>(d_gate);

        if (ring_count >= ring_depth) {
          ob_status_t dst = drain_one();
          if (dst != OB_OK) {
            discard_unsubmitted_wtile();
            drain_all_cleanup();
            return dst;
          }
        }
        const uint32_t slot = (ring_head + ring_count) % ring_depth;
        const float s_gate = s_gate_cand;
        const uint64_t gkey = ob::hash_swiglu_graph_key(
            ms.hw_m, K, hw_h, N, d_gate, d_up, d_mul, d_down,
            s_gate, slot % param_swap_slots);
        ob_swiglu_int8_desc_t gd = make_graph_desc(hw_h, d_gate, d_up, d_mul, d_down, s_gate);

        const uint64_t ts0 = prof_on ? ob::monotonic_now_ns() : 0u;
        ob::CachedGraph* entry = nullptr;
        bool was_hit = false;
        ob_status_t gst = ob::acquire_swiglu_entry(ctx, gd, gkey, &entry, &was_hit);
        if (gst != OB_OK) {
          discard_unsubmitted_wtile();
          drain_all_cleanup();
          return gst;
        }
        total_tiles += 1u;
        chained_chunks += 1u;
        if (was_hit) total_hits += 1u;

        ob_buffer* out_buf = nullptr;
        const size_t out_alloc = static_cast<size_t>(ms.hw_m) * N;
        ob_status_t bst = ob::ensure_pipeline_buf_tiered(
            ctx, &ctx->gemm_out_bufs_4k[slot], &ctx->gemm_out_bufs[slot], out_alloc, &out_buf);
        ob_exec_handle_t exec_h = 0u;
        if (bst == OB_OK) {
          bst = ob::device_submit_execute_async(
              ctx, entry, &shared_in_buf, 1u, &out_buf, 1u, &exec_h, w_tile->ext2_buf);
        }
        if (prof_on) ctx->prof.submit_ns += (ob::monotonic_now_ns() - ts0);
        if (bst != OB_OK) {
          ob::release_tile_entry(ctx, entry);
          discard_unsubmitted_wtile();
          drain_all_cleanup();
          return bst;
        }

        PendingSwigluChunk& p = ring[slot];
        p.handle = exec_h;
        p.w_tile = w_tile;
        w_tile->in_use += 1u;
        p.in_buf = shared_in_buf;
        ob::buffer_retain(p.in_buf);
        p.out_buf = out_buf;
        ob::buffer_retain(p.out_buf);
        p.slot = slot;
        p.hw_h = hw_h;
        p.d_gate = d_gate;
        p.d_up = d_up;
        p.d_mul = d_mul;
        p.d_down = d_down;
        p.s_gate = s_gate;
        ring_count += 1u;
      }

      if (!slice_rejected) {
        while (ring_count > 0u) {
          ob_status_t dst = drain_one();
          if (dst != OB_OK) {
            drain_all_cleanup();
            return dst;
          }
        }
        if (ctx->mock_mode) {
          cpu_swiglu_f32_quant_slice(a_slice, desc, ms.tm, c_slice);
        }
      }
    }

    if (slice_rejected) {
      ob_invoke_stats_t fb{};
      const uint64_t fbt0 = ob::monotonic_now_ns();
      ob_status_t st = run_swiglu_m7a_fallback(ctx, a_slice, desc, ms.tm, c_slice, &fb);
      fb_wall_ns += (ob::monotonic_now_ns() - fbt0);
      if (st != OB_OK) return st;
      total_hw_us += fb.hw_execution_us;
      total_tiles += fb.tiles_dispatched;
      total_hits += fb.cache_hits;
      agg_flags |= fb.flags;
      fb_slices += 1u;
    }
  }

  const uint64_t wall_ns = ob::monotonic_now_ns() - t0_ns;
  {
    std::lock_guard<std::mutex> lk(ctx->gemm_mu);
    ctx->prof.swiglu_calls += 1u;
    ctx->prof.swiglu_chunks_chained += chained_chunks;
    ctx->prof.swiglu_fallbacks += fb_slices;
    if (chained_chunks > 0u) {
      ctx->prof.tiles_dispatched += chained_chunks;
      ctx->prof.hw_us += chained_hw_us;
    }
    if (chained_chunks > 0u || ran_cpu_fb_slice) {
      ctx->prof.total_wall_ns += (wall_ns > fb_wall_ns) ? (wall_ns - fb_wall_ns) : 0ULL;
    }
  }

  if (out_stats) {
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->hw_execution_us = total_hw_us;
    out_stats->wall_time_us = wall_ns / 1000ULL;
    out_stats->tiles_dispatched = total_tiles;
    out_stats->cache_hits = total_hits;
    out_stats->flags = agg_flags;
    if (total_hits > 0u) out_stats->flags |= OB_STAT_FLAG_CACHE_HIT;
  }
  return OB_OK;
}

}  // extern "C"
