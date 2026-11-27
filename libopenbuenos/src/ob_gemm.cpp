#include "ob_internal.h"
#include "ob_quant_unpack.hpp"

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
#if defined(__ARM_NEON) && defined(__aarch64__)
  return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
#else
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
#endif
}

inline uint16_t fp32_to_fp16_rne(float x) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &x, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exp_raw = (bits >> 23) & 0xFFu;
  const uint32_t mant = bits & 0x007FFFFFu;
  if (exp_raw == 0xFFu) {
    if (mant == 0u) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | 0x7E00u | (mant >> 13));
  }
  int32_t exp = static_cast<int32_t>(exp_raw) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    const uint32_t m = mant | 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    const uint32_t half_bit = 1u << (shift - 1u);
    const uint32_t rem = m & ((1u << shift) - 1u);
    uint32_t q = m >> shift;
    if (rem > half_bit || (rem == half_bit && (q & 1u) != 0u)) {
      q += 1u;
    }
    return static_cast<uint16_t>(sign | q);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  const uint32_t rem = mant & 0x1FFFu;
  uint32_t q = mant >> 13;
  if (rem > 0x1000u || (rem == 0x1000u && (q & 1u) != 0u)) {
    q += 1u;
    if (q == 0x0400u) {
      q = 0u;
      exp += 1;
      if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | q);
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
                                    uint64_t* out_generation,
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
      *out_generation = r.generation;
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
  uint64_t reg_gen = 0u;
  uint64_t reg_off = 0u;
  if (lookup_immutable_region_locked(
          ctx, base, row_stride_bytes, tn, row_bytes, &reg_ver, &reg_gen, &reg_off)) {
    const uint64_t imm_hdr[8] = {
        0x494D4D5554575431ULL, // "IMMUTWT1"
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)),
        reg_ver,
        reg_gen,
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

struct PinnedWeightTile {
  CachedWeightTile* tile = nullptr;
  PinnedWeightTile() = default;
  explicit PinnedWeightTile(CachedWeightTile* t) : tile(t) {
    if (tile) tile->in_use += 1u;
  }
  PinnedWeightTile(const PinnedWeightTile&) = delete;
  PinnedWeightTile& operator=(const PinnedWeightTile&) = delete;
  void reset(CachedWeightTile* t = nullptr) {
    if (tile == t) return;
    if (tile) {
      if (tile->in_use > 0u) tile->in_use -= 1u;
      if (!tile->in_cache && tile->in_use == 0u) {
        if (tile->ext2_buf) ob_buffer_free(tile->ext2_buf);
        delete tile;
      }
    }
    tile = t;
    if (tile) tile->in_use += 1u;
  }
  CachedWeightTile* release() {
    CachedWeightTile* t = tile;
    tile = nullptr;
    return t;
  }
  ~PinnedWeightTile() { reset(); }
};

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
// depends ONLY on (hw_m, hw_k, hw_n, zp_a, zp_b, scale_d, slot_bucket, freetile).
uint64_t hash_shape_key(uint32_t hw_m, uint32_t hw_k, uint32_t hw_n,
                        int32_t zp_a, int32_t zp_b, uint32_t scale_d,
                        uint32_t slot_bucket, bool freetile = false) {
  uint64_t h = freetile ? 0x4652454554494C45ULL : 0x50524F4245415F31ULL;
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

bool read_system_mem_available_bytes(size_t* out_bytes) {
  if (!out_bytes) return false;
  *out_bytes = 0u;
#if defined(__linux__)
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return false;
  char line[256];
  bool found = false;
  while (std::fgets(line, sizeof(line), f)) {
    unsigned long long kb = 0ULL;
    if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
      *out_bytes = static_cast<size_t>(kb) * 1024u;
      found = true;
      break;
    }
  }
  std::fclose(f);
  return found;
#else
  return false;
#endif
}

size_t compute_weight_cache_budget_bytes_impl(ob_context* ctx) {
  if (!ctx) return 0u;
  size_t base_cap = 0u;
  bool explicit_cap = false;
  const char* env = std::getenv("OPENBUENOS_WEIGHT_CACHE_MB");
  if (env && env[0]) {
    unsigned long mb = std::strtoul(env, nullptr, 10);
    if (mb < 16ul) mb = 16ul;
    if (mb > 8192ul) mb = 8192ul;
    base_cap = static_cast<size_t>(mb) * 1024u * 1024u;
    explicit_cap = true;
  } else {
    // 256 MiB on host mock; up to 2048 MiB on live Android device (bounded below MemAvailable/3).
    base_cap = ctx->mock_mode
        ? (256u * 1024u * 1024u)
        : (2048u * 1024u * 1024u);
  }
  bool have_mem_avail = false;
  size_t mem_avail = 0u;
  if (ctx->simulated_mem_available_bytes > 0u || ctx->simulated_mem_available_set) {
    have_mem_avail = true;
    mem_avail = ctx->simulated_mem_available_bytes;
  } else if (!ctx->mock_mode) {
    if (read_system_mem_available_bytes(&mem_avail)) {
      have_mem_avail = true;
      // /proc/meminfo MemAvailable excludes our own unswappable DMA-BUF cache;
      // add weight_cache_bytes back so our own cache cannot self-ratchet downward.
      mem_avail += ctx->weight_cache_bytes;
    }
  }
  if (have_mem_avail) {
    // Auto mode uses LMKD-safe MemAvailable / 3 ceiling; explicit OPENBUENOS_WEIGHT_CACHE_MB
    // honors higher budgets on Linux hosts while reserving >= 384 MiB OS/KV headroom.
    const size_t lmkd_cap = mem_avail / 3u;
    size_t safe_cap = lmkd_cap;
    if (explicit_cap) {
      constexpr size_t kReserveBytes = 384u * 1024u * 1024u;
      const size_t headroom_cap = (mem_avail > 512u * 1024u * 1024u)
          ? (mem_avail - kReserveBytes)
          : (mem_avail / 2u);
      safe_cap = std::max<size_t>(lmkd_cap, headroom_cap);
    }
    if (safe_cap < base_cap) {
      base_cap = safe_cap;
    }
  }
  return std::max<size_t>(base_cap, 1u * 1024u * 1024u);
}

void ensure_weight_cache_budget(ob_context* ctx, bool refresh_meminfo = false) {
  if (ctx->weight_cache_max_bytes == 0u || refresh_meminfo ||
      ctx->simulated_mem_available_bytes > 0u || ctx->simulated_mem_available_set) {
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
      ctx->weight_cache_max_entries = ctx->weight_cache_default_max_entries;
    }
  }
}

void weight_cache_evict_locked(ob_context* ctx,
                               size_t incoming_bytes,
                               const CachedWeightTile* keep = nullptr,
                               size_t incoming_entries = 1u) {
  const bool periodic_refresh =
      (ctx->simulated_mem_available_bytes > 0u || ctx->simulated_mem_available_set) ||
      (!ctx->mock_mode && ((ctx->weight_cache_misses & 31u) == 0u ||
                           ctx->weight_cache_bytes + incoming_bytes > ctx->weight_cache_max_bytes));
  ensure_weight_cache_budget(ctx, periodic_refresh);
  auto it = ctx->weight_cache_lru.end();
  while ((ctx->weight_cache_bytes + incoming_bytes > ctx->weight_cache_max_bytes ||
          ctx->weight_cache_map.size() + incoming_entries > ctx->weight_cache_max_entries) &&
         it != ctx->weight_cache_lru.begin()) {
    --it;
    CachedWeightTile* victim = *it;
    if (victim == keep || victim->in_use > 0u) continue;
    auto erase_it = it;
    ctx->weight_cache_map.erase(victim->key);
    const size_t victim_total = victim->ext2_bytes + victim->outlier_cols_bytes;
    ctx->weight_cache_bytes -= std::min(ctx->weight_cache_bytes, victim_total);
    ctx->outlier_cache_bytes -= std::min(ctx->outlier_cache_bytes, victim->outlier_cols_bytes);
    victim->in_cache = false;
    it = ctx->weight_cache_lru.erase(erase_it);
    if (victim->ext2_buf) ob_buffer_free(victim->ext2_buf);
    delete victim;
    ctx->weight_cache_evictions += 1u;
  }
}

CachedWeightTile* weight_cache_find_locked(ob_context* ctx, const WeightTileKey& key) {
  auto it = ctx->weight_cache_map.find(key);
  if (it == ctx->weight_cache_map.end()) {
    ctx->weight_cache_misses += 1u;
    return nullptr;
  }
  CachedWeightTile* t = it->second;
  if (t->lru_it != ctx->weight_cache_lru.begin()) {
    ctx->weight_cache_lru.splice(ctx->weight_cache_lru.begin(), ctx->weight_cache_lru, t->lru_it);
  }
  ctx->weight_cache_hits += 1u;
  if ((ctx->simulated_mem_available_bytes > 0u || ctx->simulated_mem_available_set) ||
      (!ctx->mock_mode && (ctx->weight_cache_hits & 63u) == 0u)) {
    ensure_weight_cache_budget(ctx, true);
    if (ctx->weight_cache_bytes > ctx->weight_cache_max_bytes ||
        ctx->weight_cache_map.size() > ctx->weight_cache_max_entries) {
      weight_cache_evict_locked(ctx, 0u, t, 0u);
    }
    const size_t t_total = t->ext2_bytes + t->outlier_cols_bytes;
    if (t->in_cache && t_total > ctx->weight_cache_max_bytes) {
      ctx->weight_cache_map.erase(t->key);
      ctx->weight_cache_lru.erase(t->lru_it);
      ctx->weight_cache_bytes -= std::min(ctx->weight_cache_bytes, t_total);
      t->in_cache = false;
      ctx->weight_cache_evictions += 1u;
    }
  }
  return t;
}

bool weight_cache_insert_locked(ob_context* ctx, CachedWeightTile* tile) {
  weight_cache_evict_locked(ctx, tile->ext2_bytes);
  if (ctx->weight_cache_bytes + tile->ext2_bytes > ctx->weight_cache_max_bytes ||
      ctx->weight_cache_map.size() >= ctx->weight_cache_max_entries) {
    tile->in_cache = false;
    return false;
  }
  ctx->weight_cache_lru.push_front(tile);
  tile->lru_it = ctx->weight_cache_lru.begin();
  tile->in_cache = true;
  ctx->weight_cache_map[tile->key] = tile;
  ctx->weight_cache_bytes += tile->ext2_bytes;
  if (ctx->weight_cache_bytes > ctx->weight_cache_peak_bytes) {
    ctx->weight_cache_peak_bytes = ctx->weight_cache_bytes;
  }
  return true;
}

template <typename DrainOneFn>
ob_status_t weight_cache_prepare_space_locked(ob_context* ctx,
                                              size_t incoming_bytes,
                                              uint32_t& ring_count,
                                              DrainOneFn& drain_one,
                                              bool* out_cacheable) {
  ensure_weight_cache_budget(ctx, true);
  if (incoming_bytes > ctx->weight_cache_max_bytes) {
    while (ring_count > 0u) {
      ob_status_t dst = drain_one();
      if (dst != OB_OK) return dst;
    }
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

using RowNormStats = quant::RowNormStats;

inline RowNormStats compute_row_norm_stats(const int8_t* row, uint32_t tk) {
  return quant::compute_row_norm_stats(row, tk);
}

inline uint32_t compute_scale_D_from_maxima(double max_l2, uint64_t max_l1,
                                            uint64_t max_abs_sum, uint32_t tk,
                                            int32_t zp_a) {
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
    const RowNormStats st = compute_row_norm_stats(
        b_tile + static_cast<size_t>(ni) * stride_k, tk);
    if (st.l2 > max_l2) max_l2 = st.l2;
    if (st.l1 > max_l1) max_l1 = st.l1;
    if (st.asum > max_abs_sum) max_abs_sum = st.asum;
  }

  return compute_scale_D_from_maxima(max_l2, max_l1, max_abs_sum, tk, zp_a);
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

// Half-octave scale grid (2^(k/2), step sqrt(2) ~= 1.414x): guarantees s_gate = s_a * s_wg * d_gate
// lands EXACT on 2^(k/2) (0.000% SiLU scale mismatch) with only ~5-8 distinct s_gate values per model!
inline float quantize_scale_pow2_2nd_up(float raw_s) {
  if (raw_s <= 0.0f) return 0.0f;
  double e = std::ceil(2.0 * std::log2(static_cast<double>(raw_s)));
  return static_cast<float>(std::exp2(e * 0.5));
}

// Canonicalize a product of 2^(1/2)-grid scales (e.g. s_a * s_wg * d_gate) to the
// exact nearest IEEE-754 float on the 2^(1/2) lattice so (odd, odd) and (even, even)
// exponents hash to the identical Probe-A SwiGLU graph key (no 1-ULP cache split).
inline float snap_scale_pow2_2nd_nearest(float raw_s) {
  if (raw_s <= 0.0f) return 0.0f;
  double e = std::round(2.0 * std::log2(static_cast<double>(raw_s)));
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

// 1.5x-step geometric grid {lo, 1.5*lo, 2*lo, 3*lo, ...} for batched M=64 SwiGLU d_mul/d_down
// (eliminates 2x octave over-division quantization loss while preserving zero-clip headroom).
inline uint32_t round_1p5_grid_clamped(double val, uint32_t lo, uint32_t hi) {
  if (!(val > 0.0)) return lo;
  uint32_t p = lo;
  while (p < hi) {
    if (val <= static_cast<double>(p) * 1.18) return p;
    const uint32_t mid = p + (p >> 1u);
    if (mid <= hi && val <= static_cast<double>(mid) * 1.14) return mid;
    p <<= 1u;
  }
  return hi;
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
  // and pin M=1 dividers to pure powers of 2 -> cuts Probe-A SwiGLU graphs across 30L x 64T
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
  const double sig_gu = is_batched ? 3.85 : 3.65;
  const double gu_thresh = is_batched ? 1.45 : 1.42;
  const uint32_t d_gate = round_pow2_clamped((sig_gu * ra * static_cast<double>(l2_g_max)) / 127.0, 64u, 65536u, gu_thresh);
  const uint32_t d_up = round_pow2_clamped((sig_gu * ra * static_cast<double>(l2_u_max)) / 127.0, 64u, 65536u, gu_thresh);
  const double rms_qg = (ra * static_cast<double>(l2_g_mean)) / static_cast<double>(d_gate);
  const double rms_qu = (ra * static_cast<double>(l2_u_mean)) / static_cast<double>(d_up);
  // M=1 uses pure PoT d_mul/d_down for minimal Probe-A graph cache footprint;
  // M=64 batched prefill uses 1.5x-step d_mul/d_down so q_prod/q_out avoid 2x over-division
  // and stay well under 3.5e-3 NMSE (8/8 selftest --test-swiglu PASS) with zero sat_retries!
  uint32_t d_mul = 0u;
  if (is_batched) {
    d_mul = round_1p5_grid_clamped((rms_qg * rms_qu) / 16.0, 16u, 128u);
  } else {
    d_mul = round_pow2_clamped((rms_qg * rms_qu) / 18.0, 16u, 64u, 1.38);
  }
  const double rms_qprod = std::max<double>((0.68 * rms_qg * rms_qu) / static_cast<double>(d_mul), 2.0);
  uint32_t d_down = 0u;
  if (is_batched) {
    d_down = round_1p5_grid_clamped(
        (4.6 * rms_qprod * static_cast<double>(l2_d_max)) / 127.0, 16u, 4096u);
  } else {
    // Heavy-tailed real LLM M=1 decode down-projection headroom -> eliminates live sat_retries!
    d_down = round_pow2_clamped(
        (5.6 * rms_qprod * static_cast<double>(l2_d_max)) / 127.0, 16u, 4096u, 1.38);
  }
  *out_d_gate = d_gate;
  *out_d_up = d_up;
  *out_d_mul = d_mul;
  *out_d_down = d_down;
}

uint64_t hash_swiglu_graph_key(uint32_t hw_m, uint32_t hw_k, uint32_t hw_h, uint32_t hw_n,
                               uint32_t d_gate, uint32_t d_up, uint32_t d_mul, uint32_t d_down,
                               float s_gate, uint32_t slot_bucket, bool freetile = false) {
  uint32_t sg_bits = 0u;
  std::memcpy(&sg_bits, &s_gate, sizeof(sg_bits));
  uint64_t h = freetile ? 0x46545F5357494731ULL : 0x535749474C555F31ULL; // "FT_SWIG1" / "SWIGLU_1"
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
  bool emitted_by_freetile = false;
  ob_status_t st = OB_ERR_UNSUPPORTED;
  if (ctx->freetile_enabled) {
    std::string ft_err;
    st = freetile_try_emit_swiglu_dgc0(
        desc, ctx->builtin_dive_interp.data(), ctx->builtin_dive_interp.size(),
        &dgc0, &ft_err);
    if (st == OB_OK && !dgc0.empty()) {
      ctx->freetile_emits.fetch_add(1u, std::memory_order_relaxed);
      emitted_by_freetile = true;
    } else {
      ctx->freetile_fallbacks.fetch_add(1u, std::memory_order_relaxed);
      dgc0.clear();
    }
  }
  if (!emitted_by_freetile) {
    st = build_swiglu_dgc0(
        desc, ctx->builtin_dive_interp.data(), ctx->builtin_dive_interp.size(),
        &dgc0, nullptr, &err);
  }
  if (st != OB_OK) {
    if (!err.empty()) set_last_error(ctx, err);
    return st;
  }
  st = graph_register_dgc0_internal(
      ctx, dgc0.data(), dgc0.size(), cache_key, true, out_entry, out_cache_hit);
  if (st == OB_OK && *out_entry && !(*out_cache_hit) && emitted_by_freetile) {
    (*out_entry)->emitted_by_freetile = true;
  }
  return st;
}

uint64_t hash_w8a16_shape_key(uint32_t hw_k, uint32_t hw_n, float pt_scale,
                              uint32_t slot_id, bool freetile = false) {
  uint32_t sc_bits = 0u;
  std::memcpy(&sc_bits, &pt_scale, sizeof(sc_bits));
  uint32_t hdr[5] = {0x57384136u, hw_k, hw_n, sc_bits, slot_id};
  const uint64_t seed = freetile ? 0x46545F5738413136ULL : 0x57384131365F5054ULL;
  uint64_t h = fast_hash64_update(seed, hdr, sizeof(hdr));
  return h ? h : 1ULL;
}

ob_status_t acquire_w8a16_pt_1x_entry(ob_context* ctx,
                                      uint32_t hw_k,
                                      uint32_t hw_n,
                                      float pt_scale,
                                      uint32_t slot_id,
                                      CachedGraph** out_entry,
                                      bool* out_cache_hit) {
  *out_entry = nullptr;
  *out_cache_hit = false;
  if (!ctx->builtin_emitter_active || ctx->builtin_dive_interp.empty()) {
    return OB_ERR_UNSUPPORTED;
  }
  const uint64_t cache_key =
      hash_w8a16_shape_key(hw_k, hw_n, pt_scale, slot_id, ctx->freetile_enabled);
  if (graph_cache_lookup_entry_u64(ctx, cache_key, out_entry)) {
    *out_cache_hit = true;
    return OB_OK;
  }
  std::vector<uint8_t> dgc0;
  std::string err;
  bool emitted_by_freetile = false;
  ob_status_t st = OB_ERR_UNSUPPORTED;
  if (ctx->freetile_enabled) {
    std::string ft_err;
    st = freetile_try_emit_w8a16_dgc0(
        hw_k, hw_n, pt_scale, nullptr,
        ctx->builtin_dive_interp.data(), ctx->builtin_dive_interp.size(),
        &dgc0, &ft_err);
    if (st == OB_OK && !dgc0.empty()) {
      ctx->freetile_emits.fetch_add(1u, std::memory_order_relaxed);
      emitted_by_freetile = true;
    } else {
      ctx->freetile_fallbacks.fetch_add(1u, std::memory_order_relaxed);
      dgc0.clear();
    }
  }
  if (!emitted_by_freetile) {
    st = build_dgc0_w8a16_pt_1x(
        hw_k, hw_n, pt_scale, nullptr,
        ctx->builtin_dive_interp.data(), ctx->builtin_dive_interp.size(),
        &dgc0, &err);
  }
  if (st != OB_OK) {
    if (!err.empty()) set_last_error(ctx, err);
    return st;
  }
  st = graph_register_dgc0_internal(
      ctx, dgc0.data(), dgc0.size(), cache_key, true, out_entry, out_cache_hit);
  if (st == OB_OK && *out_entry && !(*out_cache_hit) && emitted_by_freetile) {
    (*out_entry)->emitted_by_freetile = true;
  }
  return st;
}

// Prepare FP32 input row for native W8A16(FP16) with power-of-2 prescale guard
// preventing IEEE FP16 input saturation (max finite 65504.0f -> guard threshold 32768.0f).
// Returns prescale_inv (= 1.0f in the common case when max(|a|) <= 32768.0f).
float prepare_w8a16_input_row_f32(const float* a_row,
                                  uint32_t tk,
                                  uint32_t hw_k,
                                  float* dst_row) {
  float amax = 0.0f;
  uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  float32x4_t v_am0 = vdupq_n_f32(0.0f);
  float32x4_t v_am1 = vdupq_n_f32(0.0f);
  for (; ki + 8u <= tk; ki += 8u) {
    v_am0 = vmaxq_f32(v_am0, vabsq_f32(vld1q_f32(a_row + ki)));
    v_am1 = vmaxq_f32(v_am1, vabsq_f32(vld1q_f32(a_row + ki + 4u)));
  }
  amax = vmaxvq_f32(vmaxq_f32(v_am0, v_am1));
#endif
  for (; ki < tk; ++ki) {
    const float av = std::fabs(a_row[ki]);
    if (av > amax) amax = av;
  }

  float prescale_inv = 1.0f;
  if (std::isfinite(amax) && amax > 32768.0f) {
    int exp2 = 0;
    (void)std::frexp(amax, &exp2);
    const int exp2_shift = exp2 - 14;
    const float prescale = std::ldexp(1.0f, -exp2_shift);
    prescale_inv = std::ldexp(1.0f, exp2_shift);
    uint32_t i = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t v_ps = vdupq_n_f32(prescale);
    for (; i + 8u <= tk; i += 8u) {
      vst1q_f32(dst_row + i, vmulq_f32(vld1q_f32(a_row + i), v_ps));
      vst1q_f32(dst_row + i + 4u, vmulq_f32(vld1q_f32(a_row + i + 4u), v_ps));
    }
#endif
    for (; i < tk; ++i) dst_row[i] = a_row[i] * prescale;
  } else {
    std::memcpy(dst_row, a_row, static_cast<size_t>(tk) * sizeof(float));
  }
  if (tk < hw_k) {
    std::memset(dst_row + tk, 0, static_cast<size_t>(hw_k - tk) * sizeof(float));
  }
  return prescale_inv;
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
  ctx->last_emit_was_freetile = false;
  size_t dgc0_size = 0u;
  ob_status_t st = emitter(&desc, ctx->emitter_scratch.data(),
                           ctx->emitter_scratch.size(), &dgc0_size, user_data);
  if (st == OB_ERR_OVERFLOW && dgc0_size > ctx->emitter_scratch.size()) {
    ctx->emitter_scratch.resize(dgc0_size);
    ctx->last_emit_was_freetile = false;
    st = emitter(&desc, ctx->emitter_scratch.data(),
                 ctx->emitter_scratch.size(), &dgc0_size, user_data);
  }
  if (st != OB_OK) return st;
  const bool emitted_by_freetile = ctx->last_emit_was_freetile;

  const bool share_zero = (desc.weights_nk == nullptr && desc.packed_params == nullptr);
  st = graph_register_dgc0_internal(
      ctx, ctx->emitter_scratch.data(), dgc0_size, desc.cache_key,
      share_zero, out_entry, out_cache_hit);
  if (st == OB_OK && *out_entry && !(*out_cache_hit) && emitted_by_freetile) {
    (*out_entry)->emitted_by_freetile = true;
  }
  return st;
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
        if (is_lt_64x_shape(64u, hk, hn)) {
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

inline uint64_t quant_tile_hash_tag(ob_quant_type_t qtype) {
  switch (qtype) {
    case OB_QUANT_Q8_0:   return 0x51385F5754494C45ULL; // "Q8_WTILE"
    case OB_QUANT_Q4_0:   return 0x51345F5754494C45ULL; // "Q4_WTILE"
    case OB_QUANT_Q4_K:   return 0x51344B5F5754494CULL; // "Q4K_WTIL"
    case OB_QUANT_Q6_K:   return 0x51364B5F5754494CULL; // "Q6K_WTIL"
    case OB_QUANT_IQ4_NL: return 0x4951344E5F57544CULL; // "IQ4N_WTL"
    case OB_QUANT_IQ4_XS: return 0x495134585F57544CULL; // "IQ4X_WTL"
    default:              return 0x51554E4B5F57544CULL;
  }
}

inline void add_scaled_offset_row_f32(float* c_row,
                                      const float* mu_ptr,
                                      uint32_t tn,
                                      float a_sum) {
  if (a_sum == 0.0f) return;
  uint32_t ni = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t vas = vdupq_n_f32(a_sum);
  for (; ni + 16u <= tn; ni += 16u) {
    vst1q_f32(c_row + ni,       vfmaq_f32(vld1q_f32(c_row + ni),       vld1q_f32(mu_ptr + ni),       vas));
    vst1q_f32(c_row + ni + 4u,  vfmaq_f32(vld1q_f32(c_row + ni + 4u),  vld1q_f32(mu_ptr + ni + 4u),  vas));
    vst1q_f32(c_row + ni + 8u,  vfmaq_f32(vld1q_f32(c_row + ni + 8u),  vld1q_f32(mu_ptr + ni + 8u),  vas));
    vst1q_f32(c_row + ni + 12u, vfmaq_f32(vld1q_f32(c_row + ni + 12u), vld1q_f32(mu_ptr + ni + 12u), vas));
  }
  for (; ni + 4u <= tn; ni += 4u) {
    vst1q_f32(c_row + ni, vfmaq_f32(vld1q_f32(c_row + ni), vld1q_f32(mu_ptr + ni), vas));
  }
#endif
  for (; ni < tn; ++ni) {
    c_row[ni] += mu_ptr[ni] * a_sum;
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

void apply_activation_inplace_parallel(float* c_mn, size_t count, uint32_t act, WorkerPool* pool) {
  if (!c_mn || act == OB_ACT_NONE || count == 0u) return;
  if (pool && count >= 16384u) {
    const uint32_t chunks = static_cast<uint32_t>((count + 4095u) / 4096u);
    pool->parallel_for_fn(chunks, 1u, [&](uint32_t s, uint32_t e, uint32_t) {
      const size_t lo = static_cast<size_t>(s) * 4096u;
      const size_t hi = std::min<size_t>(count, static_cast<size_t>(e) * 4096u);
      if (lo < hi) apply_activation_inplace_impl(c_mn + lo, hi - lo, act);
    });
  } else {
    apply_activation_inplace_impl(c_mn, count, act);
  }
}

void silu_mul_inplace_impl(float* gate, const float* up, size_t count) {
  size_t i = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t v_one = vdupq_n_f32(1.0f);
  // 2x-unrolled (8 floats/iter) with pipelined 2-step NR reciprocal (24-bit IEEE exact),
  // avoiding non-pipelined fdiv stalls on the 18.35 MB per-layer Gate*Up buffer.
  for (; i + 8u <= count; i += 8u) {
    float32x4_t g0 = vld1q_f32(gate + i);
    float32x4_t g1 = vld1q_f32(gate + i + 4u);
    float32x4_t u0 = vld1q_f32(up + i);
    float32x4_t u1 = vld1q_f32(up + i + 4u);
    float32x4_t d0 = vaddq_f32(v_one, fast_exp_f32x4(vnegq_f32(g0)));
    float32x4_t d1 = vaddq_f32(v_one, fast_exp_f32x4(vnegq_f32(g1)));
    float32x4_t r0 = vrecpeq_f32(d0);
    float32x4_t r1 = vrecpeq_f32(d1);
    r0 = vmulq_f32(r0, vrecpsq_f32(d0, r0));
    r1 = vmulq_f32(r1, vrecpsq_f32(d1, r1));
    r0 = vmulq_f32(r0, vrecpsq_f32(d0, r0));
    r1 = vmulq_f32(r1, vrecpsq_f32(d1, r1));
    vst1q_f32(gate + i, vmulq_f32(vmulq_f32(g0, r0), u0));
    vst1q_f32(gate + i + 4u, vmulq_f32(vmulq_f32(g1, r1), u1));
  }
  for (; i + 4u <= count; i += 4u) {
    float32x4_t g = vld1q_f32(gate + i);
    float32x4_t u = vld1q_f32(up + i);
    float32x4_t d = vaddq_f32(v_one, fast_exp_f32x4(vnegq_f32(g)));
    float32x4_t r = vrecpeq_f32(d);
    r = vmulq_f32(r, vrecpsq_f32(d, r));
    r = vmulq_f32(r, vrecpsq_f32(d, r));
    vst1q_f32(gate + i, vmulq_f32(vmulq_f32(g, r), u));
  }
#endif
  for (; i < count; ++i) {
    const float g = gate[i];
    gate[i] = (g / (1.0f + std::exp(-g))) * up[i];
  }
}

void silu_mul_inplace_parallel(float* gate, const float* up, size_t count, WorkerPool* pool) {
  if (!gate || !up || count == 0u) return;
  if (pool && pool->num_threads > 1u && count >= 16384u) {
    const uint32_t chunks = static_cast<uint32_t>((count + 4095u) / 4096u);
    pool->parallel_for_fn(chunks, 1u, [&](uint32_t s, uint32_t e, uint32_t) {
      const size_t lo = static_cast<size_t>(s) * 4096u;
      const size_t hi = std::min<size_t>(count, static_cast<size_t>(e) * 4096u);
      if (lo < hi) silu_mul_inplace_impl(gate + lo, up + lo, hi - lo);
    });
  } else {
    silu_mul_inplace_impl(gate, up, count);
  }
}

constexpr uint32_t kMaxRowOutliers = 8u;

struct RowOutliers {
  uint32_t count = 0u;
  uint32_t k_off[kMaxRowOutliers]{};
  uint16_t uniq_idx[kMaxRowOutliers]{};
  float val[kMaxRowOutliers]{};
};

inline uint64_t compute_i8_row_sq(const int8_t* row, uint32_t tk) {
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
  return sq;
}

inline uint32_t compute_rms_shift_from_max_sq(uint64_t max_sq, uint32_t tk) {
  if (max_sq == 0ULL || tk == 0u) return 0u;
  double max_rms = std::sqrt(static_cast<double>(max_sq) / static_cast<double>(tk));
  if (max_rms <= 0.0) return 0u;
  double ratio = (max_rms * 1.35) / 73.3;
  uint32_t shift = 0u;
  while (ratio <= 0.5 && shift < 4u) {
    ratio *= 2.0;
    shift += 1u;
  }
  return shift;
}

void quantize_activation_row_split_fast(const float* a_row,
                                        uint32_t tk,
                                        int8_t* aq_row,
                                        float* out_s_a,
                                        RowOutliers* out_outliers,
                                        uint64_t* out_aq_sq) {
  if (out_outliers) out_outliers->count = 0u;
  if (out_aq_sq) *out_aq_sq = 0ULL;
  float amax = 0.0f;
  double sq_sum = 0.0;
  uint32_t ki = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  // 16-wide pure FP32 NEON pass (2x throughput vs FP64 widen, <1e-5 rel error for tk<=2048).
  float32x4_t v_amax0 = vdupq_n_f32(0.0f);
  float32x4_t v_amax1 = vdupq_n_f32(0.0f);
  float32x4_t v_amax2 = vdupq_n_f32(0.0f);
  float32x4_t v_amax3 = vdupq_n_f32(0.0f);
  float32x4_t v_sq0 = vdupq_n_f32(0.0f);
  float32x4_t v_sq1 = vdupq_n_f32(0.0f);
  float32x4_t v_sq2 = vdupq_n_f32(0.0f);
  float32x4_t v_sq3 = vdupq_n_f32(0.0f);
  for (; ki + 16u <= tk; ki += 16u) {
    float32x4_t v0 = vld1q_f32(a_row + ki);
    float32x4_t v1 = vld1q_f32(a_row + ki + 4u);
    float32x4_t v2 = vld1q_f32(a_row + ki + 8u);
    float32x4_t v3 = vld1q_f32(a_row + ki + 12u);
    v_amax0 = vmaxq_f32(v_amax0, vabsq_f32(v0));
    v_amax1 = vmaxq_f32(v_amax1, vabsq_f32(v1));
    v_amax2 = vmaxq_f32(v_amax2, vabsq_f32(v2));
    v_amax3 = vmaxq_f32(v_amax3, vabsq_f32(v3));
    v_sq0 = vfmaq_f32(v_sq0, v0, v0);
    v_sq1 = vfmaq_f32(v_sq1, v1, v1);
    v_sq2 = vfmaq_f32(v_sq2, v2, v2);
    v_sq3 = vfmaq_f32(v_sq3, v3, v3);
  }
  for (; ki + 4u <= tk; ki += 4u) {
    float32x4_t v0 = vld1q_f32(a_row + ki);
    v_amax0 = vmaxq_f32(v_amax0, vabsq_f32(v0));
    v_sq0 = vfmaq_f32(v_sq0, v0, v0);
  }
  amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(v_amax0, v_amax1), vmaxq_f32(v_amax2, v_amax3)));
  sq_sum = static_cast<double>(vaddvq_f32(v_sq0)) + static_cast<double>(vaddvq_f32(v_sq1)) +
           static_cast<double>(vaddvq_f32(v_sq2)) + static_cast<double>(vaddvq_f32(v_sq3));
#endif
  for (; ki < tk; ++ki) {
    float v = a_row[ki];
    float av = std::fabs(v);
    if (av > amax) amax = av;
    sq_sum += static_cast<double>(v) * static_cast<double>(v);
  }
  auto recompute_sq_fp64 = [&]() -> double {
    double s = 0.0;
    for (uint32_t i = 0u; i < tk; ++i) {
      const double dv = static_cast<double>(a_row[i]);
      s += dv * dv;
    }
    return s;
  };
  bool sq_is_fp64 = false;
#if !(defined(__ARM_NEON) && defined(__aarch64__))
  sq_is_fp64 = true;
#endif
  if ((!std::isfinite(sq_sum) || sq_sum == 0.0) && std::isfinite(amax) && amax > 0.0f) {
    // Recompute activation energy in FP64 when large or tiny finite inputs overflow or
    // underflow FP32 squaring in the NEON fast path.
    sq_sum = recompute_sq_fp64();
    sq_is_fp64 = true;
  }
  if (amax == 0.0f) {
    std::memset(aq_row, 0, tk);
    *out_s_a = 0.0f;
    return;
  }

  // LLM.int8()-style sparse outlier peel: single-pass top-9 extraction (100% bit-exact).
  constexpr uint32_t kMaxOutliers = kMaxRowOutliers;
  constexpr uint32_t kTopK = kMaxOutliers + 1u;
  constexpr double kOutlierSigma = 8.0;
  uint32_t peeled_idx[kMaxOutliers]{};
  uint32_t n_peeled = 0u;
  double rem_sq = sq_sum;

  if (out_outliers && tk >= 128u) {
    double rms = std::sqrt(sq_sum / static_cast<double>(tk));
    if (static_cast<double>(amax) > kOutlierSigma * std::max(rms, 1e-12)) {
      if (!sq_is_fp64) {
        // Recompute exact FP64 energy before peeling so subtracting a dominant outlier's
        // square cannot lose smaller activations to FP32 rounding error.
        sq_sum = recompute_sq_fp64();
        rem_sq = sq_sum;
        sq_is_fp64 = true;
      }
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

      for (uint32_t iter = 0u; iter < kMaxOutliers; ++iter) {
        if (top[iter].av <= 0.0f) break;
        double denom = static_cast<double>(tk > n_peeled + 1u ? (tk - n_peeled - 1u) : 1u);
        double next_sq = std::max(
            0.0, rem_sq - static_cast<double>(top[iter].av) * static_cast<double>(top[iter].av));
        double rem_rms = std::sqrt(next_sq / denom);
        if (static_cast<double>(top[iter].av) <= kOutlierSigma * std::max(rem_rms, 1e-12)) {
          break;
        }
        peeled_idx[n_peeled] = top[iter].idx;
        out_outliers->k_off[n_peeled] = top[iter].idx;
        out_outliers->val[n_peeled] = a_row[top[iter].idx];
        ++n_peeled;
        rem_sq = next_sq;
      }
      out_outliers->count = n_peeled;
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
  if (out_aq_sq) {
    // Analytical sum(aq^2) directly from remaining FP32 energy rem_sq (eliminates Pass 4 read!)
    const uint64_t max_aq_sq = static_cast<uint64_t>(tk) * 16129ULL;
    const double d_inv = static_cast<double>(inv_sa);
    const double est_sq = rem_sq * d_inv * d_inv;
    if (!std::isfinite(est_sq) || est_sq >= static_cast<double>(max_aq_sq)) {
      *out_aq_sq = max_aq_sq;
    } else if (est_sq > 0.0) {
      *out_aq_sq = static_cast<uint64_t>(est_sq + 0.5);
    } else {
      *out_aq_sq = 0ULL;
    }
  }
}

void quantize_activation_row_split_impl(const float* a_row,
                                        uint32_t tk,
                                        int8_t* aq_row,
                                        float* out_s_a,
                                        std::vector<std::pair<uint32_t, float>>* out_outliers) {
  if (out_outliers) {
    out_outliers->clear();
    RowOutliers ro{};
    quantize_activation_row_split_fast(a_row, tk, aq_row, out_s_a, &ro, nullptr);
    for (uint32_t i = 0u; i < ro.count; ++i) {
      out_outliers->push_back({ro.k_off[i], ro.val[i]});
    }
  } else {
    quantize_activation_row_split_fast(a_row, tk, aq_row, out_s_a, nullptr, nullptr);
  }
}

uint32_t compute_activation_rms_shift_impl(const int8_t* aq_tile,
                                           uint32_t tm,
                                           uint32_t tk,
                                           uint32_t hw_k) {
  if (!aq_tile || tm == 0u || tk == 0u) return 0u;
  uint64_t max_sq = 0ULL;
  for (uint32_t mi = 0; mi < tm; ++mi) {
    const uint64_t sq = compute_i8_row_sq(aq_tile + static_cast<size_t>(mi) * hw_k, tk);
    if (sq > max_sq) max_sq = sq;
  }
  return compute_rms_shift_from_max_sq(max_sq, tk);
}

// Exact check whether any active int8 output byte equals +127 (0x7F) or -128 (0x80).
// Algebraic identity: uint8(x - 0x7F) <= 1 iff x == 0x7F (+127) or x == 0x80 (-128).
bool tile_output_is_saturated_impl(const int8_t* out_data,
                                   uint32_t tm, uint32_t tn, uint32_t hw_n) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const uint8x16_t v_sub = vdupq_n_u8(0x7Fu);
  uint8x16_t v_min0 = vdupq_n_u8(0xFFu);
  uint8x16_t v_min1 = vdupq_n_u8(0xFFu);
  uint8x16_t v_min2 = vdupq_n_u8(0xFFu);
  uint8x16_t v_min3 = vdupq_n_u8(0xFFu);
  for (uint32_t mi = 0; mi < tm; ++mi) {
    const uint8_t* row =
        reinterpret_cast<const uint8_t*>(out_data) + static_cast<size_t>(mi) * hw_n;
    uint32_t ni = 0u;
    for (; ni + 64u <= tn; ni += 64u) {
      v_min0 = vminq_u8(v_min0, vsubq_u8(vld1q_u8(row + ni), v_sub));
      v_min1 = vminq_u8(v_min1, vsubq_u8(vld1q_u8(row + ni + 16u), v_sub));
      v_min2 = vminq_u8(v_min2, vsubq_u8(vld1q_u8(row + ni + 32u), v_sub));
      v_min3 = vminq_u8(v_min3, vsubq_u8(vld1q_u8(row + ni + 48u), v_sub));
    }
    for (; ni + 16u <= tn; ni += 16u) {
      v_min0 = vminq_u8(v_min0, vsubq_u8(vld1q_u8(row + ni), v_sub));
    }
    for (; ni < tn; ++ni) {
      int8_t qv = static_cast<int8_t>(row[ni]);
      if (qv == 127 || qv == -128) return true;
    }
  }
  uint8x16_t v_min = vminq_u8(vminq_u8(v_min0, v_min1), vminq_u8(v_min2, v_min3));
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

// Single-pass fused NEON dequant (+ optional store_first avoiding RMW) + peeled outlier FMA!
// Keeps c_row[ni..ni+15] in NEON registers across all <=8 outlier columns, writing c_row ONCE.
inline void dequant_outliers_row_f32_impl(float* c_row,
                                          const int8_t* o_row,
                                          const float* sb_ptr,
                                          uint32_t tn,
                                          float sa_d,
                                          bool store_first,
                                          const float* const* outlier_cols,
                                          const float* outlier_vals,
                                          uint32_t num_outliers) {
  if (num_outliers == 0u && !store_first) {
    dequant_accum_row_f32_impl(c_row, o_row, sb_ptr, tn, sa_d);
    return;
  }
  uint32_t ni = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t v_sad = vdupq_n_f32(sa_d);
  float32x4_t v_ox[kMaxRowOutliers];
  const uint32_t fast_out = (num_outliers <= kMaxRowOutliers) ? num_outliers : kMaxRowOutliers;
  for (uint32_t oi = 0u; oi < fast_out; ++oi) {
    v_ox[oi] = vdupq_n_f32(outlier_vals[oi]);
  }
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

    float32x4_t a0, a1, a2, a3;
    if (store_first) {
      a0 = vmulq_f32(f0, s0);
      a1 = vmulq_f32(f1, s1);
      a2 = vmulq_f32(f2, s2);
      a3 = vmulq_f32(f3, s3);
    } else {
      a0 = vfmaq_f32(vld1q_f32(c_row + ni),       f0, s0);
      a1 = vfmaq_f32(vld1q_f32(c_row + ni + 4u),  f1, s1);
      a2 = vfmaq_f32(vld1q_f32(c_row + ni + 8u),  f2, s2);
      a3 = vfmaq_f32(vld1q_f32(c_row + ni + 12u), f3, s3);
    }

    for (uint32_t oi = 0u; oi < fast_out; ++oi) {
      const float* wc = outlier_cols[oi] + ni;
      const float32x4_t vx = v_ox[oi];
      a0 = vfmaq_f32(a0, vld1q_f32(wc),       vx);
      a1 = vfmaq_f32(a1, vld1q_f32(wc + 4u),  vx);
      a2 = vfmaq_f32(a2, vld1q_f32(wc + 8u),  vx);
      a3 = vfmaq_f32(a3, vld1q_f32(wc + 12u), vx);
    }
    for (uint32_t oi = fast_out; oi < num_outliers; ++oi) {
      const float* wc = outlier_cols[oi] + ni;
      const float32x4_t vx = vdupq_n_f32(outlier_vals[oi]);
      a0 = vfmaq_f32(a0, vld1q_f32(wc),       vx);
      a1 = vfmaq_f32(a1, vld1q_f32(wc + 4u),  vx);
      a2 = vfmaq_f32(a2, vld1q_f32(wc + 8u),  vx);
      a3 = vfmaq_f32(a3, vld1q_f32(wc + 12u), vx);
    }

    vst1q_f32(c_row + ni,       a0);
    vst1q_f32(c_row + ni + 4u,  a1);
    vst1q_f32(c_row + ni + 8u,  a2);
    vst1q_f32(c_row + ni + 12u, a3);
  }
#endif
  for (; ni < tn; ++ni) {
    float v = static_cast<float>(o_row[ni]) * (sa_d * sb_ptr[ni]);
    if (!store_first) v += c_row[ni];
    for (uint32_t oi = 0u; oi < num_outliers; ++oi) {
      v += outlier_vals[oi] * outlier_cols[oi][ni];
    }
    c_row[ni] = v;
  }
}

// Compute exact sum of non-outlier activations in FP64 (for Q4_K algebraic dmin offset correction).
inline float compute_row_sum_excluding_outliers(const float* a_row,
                                                uint32_t tk,
                                                const RowOutliers* ro) {
  double s = 0.0;
  if (!ro || ro->count == 0u) {
    for (uint32_t ki = 0u; ki < tk; ++ki) {
      s += static_cast<double>(a_row[ki]);
    }
  } else {
    for (uint32_t ki = 0u; ki < tk; ++ki) {
      bool peeled = false;
      for (uint32_t oi = 0u; oi < ro->count; ++oi) {
        if (ro->k_off[oi] == ki) {
          peeled = true;
          break;
        }
      }
      if (!peeled) s += static_cast<double>(a_row[ki]);
    }
  }
  return static_cast<float>(s);
}

// Extract exact FP32 outlier column across any mix of supported GGUF quant formats.
void extract_outlier_col_impl(const ob_gemm_f32_quant_job_t* jobs,
                              const FusedSubSlice* subs,
                              uint32_t num_subs,
                              uint32_t k_dim,
                              uint32_t k0,
                              uint32_t k_off,
                              float* dst_col) {
  const uint32_t abs_k = k0 + k_off;
  for (uint32_t si = 0u; si < num_subs; ++si) {
    const FusedSubSlice& sub = subs[si];
    const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
    quant::extract_column_f32(static_cast<ob_quant_type_t>(job.is_q4_0),
                              job.b_nk, k_dim, sub.n0, sub.tn, abs_k,
                              dst_col + sub.out_off);
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

void dequant_outliers_row_f32(float* c_row,
                              const int8_t* o_row,
                              const float* sb_ptr,
                              uint32_t tn,
                              float sa_d,
                              bool store_first,
                              const float* const* outlier_cols,
                              const float* outlier_vals,
                              uint32_t num_outliers) {
  dequant_outliers_row_f32_impl(c_row, o_row, sb_ptr, tn, sa_d, store_first,
                                outlier_cols, outlier_vals, num_outliers);
}

void apply_activation_inplace_f32(float* c_mn, size_t count, uint32_t act) {
  apply_activation_inplace_impl(c_mn, count, act);
}

float round_f32_to_fp16_f32(float x) {
  return fp16_to_fp32(fp32_to_fp16_rne(x));
}

void w8a16_pt_epilogue_f32(float* c_row,
                           const float* y_raw,
                           const float* sb_ptr,
                           uint32_t tn,
                           float prescale_inv,
                           bool store_first) {
  if (prescale_inv != 1.0f) {
    // Use FP64 intermediate so prescale_inv * sb_ptr[ni] never overflows FP32 when
    // the combined output y_raw[ni] * prescale_inv * sb_ptr[ni] is finite.
    const double d_ps = static_cast<double>(prescale_inv);
    if (store_first) {
      for (uint32_t ni = 0u; ni < tn; ++ni) {
        c_row[ni] = static_cast<float>(
            static_cast<double>(y_raw[ni]) * (d_ps * static_cast<double>(sb_ptr[ni])));
      }
    } else {
      for (uint32_t ni = 0u; ni < tn; ++ni) {
        c_row[ni] = static_cast<float>(
            static_cast<double>(c_row[ni]) +
            static_cast<double>(y_raw[ni]) * (d_ps * static_cast<double>(sb_ptr[ni])));
      }
    }
    return;
  }

  uint32_t ni = 0u;
#if defined(__ARM_NEON) && defined(__aarch64__)
  if (store_first) {
    for (; ni + 16u <= tn; ni += 16u) {
      float32x4_t y0 = vld1q_f32(y_raw + ni);
      float32x4_t y1 = vld1q_f32(y_raw + ni + 4u);
      float32x4_t y2 = vld1q_f32(y_raw + ni + 8u);
      float32x4_t y3 = vld1q_f32(y_raw + ni + 12u);
      float32x4_t s0 = vld1q_f32(sb_ptr + ni);
      float32x4_t s1 = vld1q_f32(sb_ptr + ni + 4u);
      float32x4_t s2 = vld1q_f32(sb_ptr + ni + 8u);
      float32x4_t s3 = vld1q_f32(sb_ptr + ni + 12u);
      vst1q_f32(c_row + ni,       vmulq_f32(y0, s0));
      vst1q_f32(c_row + ni + 4u,  vmulq_f32(y1, s1));
      vst1q_f32(c_row + ni + 8u,  vmulq_f32(y2, s2));
      vst1q_f32(c_row + ni + 12u, vmulq_f32(y3, s3));
    }
  } else {
    for (; ni + 16u <= tn; ni += 16u) {
      float32x4_t y0 = vld1q_f32(y_raw + ni);
      float32x4_t y1 = vld1q_f32(y_raw + ni + 4u);
      float32x4_t y2 = vld1q_f32(y_raw + ni + 8u);
      float32x4_t y3 = vld1q_f32(y_raw + ni + 12u);
      float32x4_t s0 = vld1q_f32(sb_ptr + ni);
      float32x4_t s1 = vld1q_f32(sb_ptr + ni + 4u);
      float32x4_t s2 = vld1q_f32(sb_ptr + ni + 8u);
      float32x4_t s3 = vld1q_f32(sb_ptr + ni + 12u);
      vst1q_f32(c_row + ni,       vfmaq_f32(vld1q_f32(c_row + ni),       y0, s0));
      vst1q_f32(c_row + ni + 4u,  vfmaq_f32(vld1q_f32(c_row + ni + 4u),  y1, s1));
      vst1q_f32(c_row + ni + 8u,  vfmaq_f32(vld1q_f32(c_row + ni + 8u),  y2, s2));
      vst1q_f32(c_row + ni + 12u, vfmaq_f32(vld1q_f32(c_row + ni + 12u), y3, s3));
    }
  }
#endif
  if (store_first) {
    for (; ni < tn; ++ni) c_row[ni] = y_raw[ni] * sb_ptr[ni];
  } else {
    for (; ni < tn; ++ni) c_row[ni] += y_raw[ni] * sb_ptr[ni];
  }
}

void gemm_release_pipeline_buffers(ob_context* ctx) {
  if (!ctx) return;
  {
    std::lock_guard<std::mutex> flk(ctx->swiglu_fb_mu);
    std::vector<float>().swap(ctx->swiglu_gate_scratch);
    std::vector<float>().swap(ctx->swiglu_up_scratch);
  }
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
  ctx->outlier_cache_bytes = 0;
  ctx->immutable_weight_regions.clear();
}

bool compute_allow_lt_64x(uint32_t k_dim,
                          const ob_gemm_f32_quant_job_t* jobs,
                          uint32_t num_jobs,
                          uint32_t max_k,
                          uint32_t max_n,
                          bool builtin_active,
                          bool lt_enabled,
                          bool blt_enabled,
                          bool horiz_fusion,
                          std::vector<ScheduledGroup>* scratch_groups) {
  bool allow_lt_64x = (k_dim >= 256u && max_k >= 512u && max_n >= 128u);
  if (!allow_lt_64x) return false;

  bool all_ge_128 = true;
  for (uint32_t j = 0u; j < num_jobs; ++j) {
    if (jobs[j].n_dim < 128u) {
      all_ge_128 = false;
      break;
    }
  }
  if (all_ge_128) return true;
  if (!horiz_fusion || num_jobs < 2u || !builtin_active || !blt_enabled) {
    return false;
  }
  build_scheduled_groups(64u, k_dim, jobs, num_jobs, max_k, max_n,
                         builtin_active, lt_enabled, blt_enabled,
                         horiz_fusion, scratch_groups);
  if (scratch_groups->empty()) return false;
  for (const ScheduledGroup& g : *scratch_groups) {
    if (!is_lt_64x_shape(64u, g.hw_k, g.hw_n)) {
      return false;
    }
  }
  return true;
}

void split_scheduled_group_subs(const ScheduledGroup& grp,
                                uint32_t hw_m,
                                uint32_t max_k,
                                uint32_t max_n,
                                bool builtin_active,
                                std::vector<ScheduledGroup>* out) {
  out->clear();
  const uint32_t k0 = grp.k0;
  const uint32_t tk = grp.tk;
  const uint32_t k_cap = (hw_m == 64u) ? std::min<uint32_t>(max_k, 1536u)
                                       : std::min<uint32_t>(max_k, 2048u);
  const uint32_t n_cap = (hw_m == 64u)
      ? std::max<uint32_t>(128u, (std::min<uint32_t>(max_n, 1536u) / 32u) * 32u)
      : std::min<uint32_t>(max_n, 2048u);

  for (uint32_t si = 0u; si < grp.num_subs; ++si) {
    const FusedSubSlice& sub = grp.subs[si];
    std::vector<TileRect> rects;
    uint32_t sh_k = 0u, sh_n = 0u;
    if (hw_m == 64u) {
      sh_k = ceil_lt_64x_k_dim(tk);
      sh_n = std::max<uint32_t>(128u, align_up_u32(sub.tn, 32u));
      if (sh_k > 0u && sh_k <= k_cap && sh_n <= n_cap && is_lt_64x_shape(64u, sh_k, sh_n)) {
        rects.push_back({k0, tk, sub.n0, sub.tn, sh_k, sh_n});
      }
    } else if (hw_m == 1u && try_lt_1x_hw_dims(tk, sub.tn, k_cap, n_cap, &sh_k, &sh_n)) {
      rects.push_back({k0, tk, sub.n0, sub.tn, sh_k, sh_n});
    }
    if (rects.empty()) {
      append_small_tile_rects(hw_m, k0, tk, sub.n0, sub.tn, max_k, max_n,
                              OB_QK8_0, builtin_active, &rects);
    }
    for (const TileRect& r : rects) {
      ScheduledGroup sg {};
      sg.k0 = r.k0;
      sg.tk = r.tk;
      sg.hw_k = r.hw_k;
      sg.hw_n = r.hw_n;
      sg.tn_total = r.tn;
      sg.num_subs = 1u;
      sg.subs[0] = {sub.job_idx, r.n0, r.tn, 0u};
      out->push_back(sg);
    }
  }
}

WeightTileKey compute_quant_group_weight_key(const ob_context* ctx,
                                             uint32_t k_dim,
                                             const ob_gemm_f32_quant_job_t* jobs,
                                             const ScheduledGroup& grp,
                                             uint32_t pack_fmt,
                                             bool strict_hash) {
  WeightHash128 wh_st{};
  const uint32_t num_subs = grp.num_subs;
  const uint32_t k0 = grp.k0;
  const uint32_t tk = grp.tk;
  const uint32_t hw_k = grp.hw_k;
  const uint32_t hw_n = grp.hw_n;

  if (num_subs == 1u) {
    const FusedSubSlice& sub = grp.subs[0];
    const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
    const ob_quant_type_t qtype = static_cast<ob_quant_type_t>(job.is_q4_0);
    const uint64_t tag = quant_tile_hash_tag(qtype);
    uint32_t whdr[5] = {sub.tn, tk, hw_k, hw_n, pack_fmt};
    hash128_update(&wh_st, &tag, sizeof(tag));
    hash128_update(&wh_st, whdr, sizeof(whdr));
    const uint32_t qk = ob_quant_block_elements(qtype);
    const size_t blk_sz = ob_quant_block_bytes(qtype);
    const uint32_t nb_k_q = k_dim / qk;
    const uint32_t blk_start = k0 / qk;
    const uint32_t blk_end = (k0 + tk + qk - 1u) / qk;
    const uint32_t blk_cnt_q = blk_end - blk_start;
    if ((k0 % qk) != 0u || (tk % qk) != 0u) {
      uint32_t sub_k[2] = {k0 % qk, tk};
      hash128_update(&wh_st, sub_k, sizeof(sub_k));
    }
    const size_t row_blk_bytes = static_cast<size_t>(blk_cnt_q) * blk_sz;
    const size_t row_stride_bytes = static_cast<size_t>(nb_k_q) * blk_sz;
    const uint8_t* raw_base = static_cast<const uint8_t*>(job.b_nk) +
                              (static_cast<size_t>(sub.n0) * nb_k_q + blk_start) * blk_sz;
    hash_weight_tile_for_ctx(ctx, &wh_st, raw_base, row_stride_bytes, sub.tn,
                             row_blk_bytes, strict_hash);
  } else {
    const uint64_t tag = 0x465553455F575431ULL;
    uint32_t fhdr[5] = {num_subs, tk, hw_k, hw_n, pack_fmt};
    hash128_update(&wh_st, &tag, sizeof(tag));
    hash128_update(&wh_st, fhdr, sizeof(fhdr));
    for (uint32_t si = 0u; si < num_subs; ++si) {
      const FusedSubSlice& sub = grp.subs[si];
      const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
      const ob_quant_type_t qtype = static_cast<ob_quant_type_t>(job.is_q4_0);
      uint32_t shdr[3] = {sub.tn, sub.out_off, static_cast<uint32_t>(qtype)};
      hash128_update(&wh_st, shdr, sizeof(shdr));
      const uint32_t qk = ob_quant_block_elements(qtype);
      const size_t blk_sz = ob_quant_block_bytes(qtype);
      const uint32_t nb_k_q = k_dim / qk;
      const uint32_t blk_start = k0 / qk;
      const uint32_t blk_end = (k0 + tk + qk - 1u) / qk;
      const uint32_t blk_cnt_q = blk_end - blk_start;
      if ((k0 % qk) != 0u || (tk % qk) != 0u) {
        uint32_t sub_k[2] = {k0 % qk, tk};
        hash128_update(&wh_st, sub_k, sizeof(sub_k));
      }
      const size_t row_blk_bytes = static_cast<size_t>(blk_cnt_q) * blk_sz;
      const size_t row_stride_bytes = static_cast<size_t>(nb_k_q) * blk_sz;
      const uint8_t* raw_base = static_cast<const uint8_t*>(job.b_nk) +
                                (static_cast<size_t>(sub.n0) * nb_k_q + blk_start) * blk_sz;
      hash_weight_tile_for_ctx(ctx, &wh_st, raw_base, row_stride_bytes, sub.tn,
                               row_blk_bytes, strict_hash);
    }
  }
  return finalize_weight_key(wh_st);
}

template <typename PrepareSpaceFn>
ob_status_t pack_quant_group_tile_locked(ob_context* ctx,
                                         uint32_t k_dim,
                                         const ob_gemm_f32_quant_job_t* jobs,
                                         const ScheduledGroup& grp,
                                         bool tile_use_w8a16,
                                         bool is_lt64_tile,
                                         bool grp_has_q4_k,
                                         const WeightTileKey& wh,
                                         bool abort_if_uncacheable,
                                         WorkerPool* pack_pool,
                                         std::vector<int8_t>* b_q_tile,
                                         PrepareSpaceFn& prepare_space,
                                         bool* out_rejected_fusion,
                                         bool* out_skipped_budget,
                                         uint32_t* out_scale_d,
                                         CachedWeightTile** out_tile) {
  *out_rejected_fusion = false;
  *out_skipped_budget = false;
  *out_tile = nullptr;

  const uint32_t k0 = grp.k0;
  const uint32_t tk = grp.tk;
  const uint32_t hw_k = grp.hw_k;
  const uint32_t hw_n = grp.hw_n;
  const uint32_t tn_total = grp.tn_total;
  const uint32_t num_subs = grp.num_subs;

  const size_t ext2_sz = tile_use_w8a16
      ? pack_w8a16_ext2_size(hw_n, hw_k)
      : (is_lt64_tile ? pack_ext2_size_for_lt_64x(hw_n, hw_k)
                      : pack_ext2_size_for_tile(hw_n, hw_k, false));
  const size_t ext2_slab_bytes = buffer_slab_size(ext2_sz);

  // When prepacking at load time (abort_if_uncacheable && single-sub), bail BEFORE unpacking if budget is full.
  if (abort_if_uncacheable && num_subs == 1u) {
    bool fit_precheck = false;
    ob_status_t chk_st = prepare_space(ext2_slab_bytes, &fit_precheck);
    if (chk_st != OB_OK) return chk_st;
    if (!fit_precheck) {
      *out_skipped_budget = true;
      return OB_OK;
    }
  }

  b_q_tile->resize(static_cast<size_t>(hw_n) * hw_k);
  if (tn_total < hw_n) {
    std::memset(b_q_tile->data() + static_cast<size_t>(tn_total) * hw_k, 0,
                static_cast<size_t>(hw_n - tn_total) * hw_k);
  }
  std::vector<float> sb_vec(tn_total, 0.0f);
  std::vector<float> offset_vec(grp_has_q4_k ? tn_total : 0u, 0.0f);
  bool any_nonzero_offset = false;
  std::vector<RowNormStats> row_stats(tile_use_w8a16 ? 0u : tn_total);
  uint32_t min_sub_d = 0xFFFFFFFFu;
  double all_max_l2 = 0.0;
  uint64_t all_max_l1 = 0ULL;
  uint64_t all_max_asum = 0ULL;

  for (uint32_t si = 0u; si < num_subs; ++si) {
    const FusedSubSlice& sub = grp.subs[si];
    const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
    const ob_quant_type_t qtype = static_cast<ob_quant_type_t>(job.is_q4_0);
    int8_t* bq_dst = b_q_tile->data() + static_cast<size_t>(sub.out_off) * hw_k;
    float* sb_dst = sb_vec.data() + sub.out_off;
    float* off_dst = grp_has_q4_k ? (offset_vec.data() + sub.out_off) : nullptr;
    RowNormStats* st_dst = tile_use_w8a16 ? nullptr : (row_stats.data() + sub.out_off);
    if (quant::requant_rows(qtype, job.b_nk, k_dim, sub.n0, sub.tn, k0, tk, hw_k,
                            bq_dst, sb_dst, off_dst, pack_pool, st_dst)) {
      any_nonzero_offset = true;
    }
    if (!tile_use_w8a16) {
      double sub_max_l2 = 0.0;
      uint64_t sub_max_l1 = 0ULL;
      uint64_t sub_max_asum = 0ULL;
      for (uint32_t ni = 0u; ni < sub.tn; ++ni) {
        if (st_dst[ni].l2 > sub_max_l2) sub_max_l2 = st_dst[ni].l2;
        if (st_dst[ni].l1 > sub_max_l1) sub_max_l1 = st_dst[ni].l1;
        if (st_dst[ni].asum > sub_max_asum) sub_max_asum = st_dst[ni].asum;
      }
      if (sub_max_l2 > all_max_l2) all_max_l2 = sub_max_l2;
      if (sub_max_l1 > all_max_l1) all_max_l1 = sub_max_l1;
      if (sub_max_asum > all_max_asum) all_max_asum = sub_max_asum;
      if (num_subs >= 2u) {
        uint32_t sd = compute_scale_D_from_maxima(
            sub_max_l2, sub_max_l1, sub_max_asum, tk, 0);
        if (sd < min_sub_d) min_sub_d = sd;
      }
    }
  }
  if (any_nonzero_offset) {
    sb_vec.insert(sb_vec.end(), offset_vec.begin(), offset_vec.end());
  }

  uint32_t scale_d = 1u;
  if (!tile_use_w8a16) {
    scale_d = compute_scale_D_from_maxima(
        all_max_l2, all_max_l1, all_max_asum, tk, 0);

    // Precision guard for horizontal fusion: if any sub-job's optimal D is
    // > 1.5x smaller than the fused D (e.g. mixed sparse+dense synthetic test),
    // cache a zero-byte split sentinel and run sub-jobs independently with exact D.
    if (num_subs >= 2u && min_sub_d > 0u && (scale_d * 2u > min_sub_d * 3u)) {
      bool sentinel_ok = false;
      ob_status_t spst = prepare_space(0u, &sentinel_ok);
      if (spst != OB_OK) return spst;
      if (sentinel_ok) {
        CachedWeightTile* sentinel = new (std::nothrow) CachedWeightTile();
        if (sentinel) {
          sentinel->key = wh;
          sentinel->scale_d = scale_d;
          sentinel->hw_k = hw_k;
          sentinel->hw_n = hw_n;
          sentinel->ext2_buf = nullptr;
          sentinel->ext2_bytes = 0u;
          sentinel->rejected_fusion = true;
          if (!weight_cache_insert_locked(ctx, sentinel)) {
            delete sentinel;
          }
        }
      }
      *out_scale_d = scale_d;
      *out_rejected_fusion = true;
      return OB_OK;
    }
  }

  bool cacheable = false;
  ob_status_t pst = prepare_space(ext2_slab_bytes, &cacheable);
  if (pst != OB_OK) return pst;
  if (abort_if_uncacheable && !cacheable) {
    *out_skipped_budget = true;
    return OB_OK;
  }

  ob_buffer* wbuf = nullptr;
  ob_status_t wst = ob_buffer_alloc(
      ctx, ext2_sz, OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL, &wbuf);
  if (wst == OB_OK) {
    wst = buffer_sync_begin_write(wbuf);
    if (wst == OB_OK) {
      uint8_t* dst_ext2 = static_cast<uint8_t*>(ob_buffer_data(wbuf));
      if (tile_use_w8a16) {
        pack_w8a16_ext2_into(dst_ext2, hw_n, hw_k, b_q_tile->data());
      } else if (is_lt64_tile) {
        pack_ext2_for_lt_64x_into(
            dst_ext2, hw_n, hw_k, b_q_tile->data(), nullptr, pack_pool);
      } else {
        pack_ext2_for_tile_into(
            dst_ext2, hw_n, hw_k, b_q_tile->data(), nullptr, false, pack_pool);
      }
      wst = buffer_sync_end_write(wbuf);
    }
  }
  if (wst != OB_OK) {
    if (wbuf) ob_buffer_free(wbuf);
    return wst;
  }

  CachedWeightTile* tile = new (std::nothrow) CachedWeightTile();
  if (!tile) {
    ob_buffer_free(wbuf);
    return OB_ERR_NO_MEMORY;
  }
  tile->key = wh;
  tile->scale_d = scale_d;
  tile->hw_k = hw_k;
  tile->hw_n = hw_n;
  tile->row_s_b = std::move(sb_vec);
  tile->ext2_buf = wbuf;
  tile->ext2_bytes = wbuf->slab.alloc_size;
  if (cacheable) {
    weight_cache_insert_locked(ctx, tile);
  } else {
    tile->in_cache = false;
  }
  *out_scale_d = scale_d;
  *out_tile = tile;
  return OB_OK;
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
  for (const ob::ImmutableWeightRegion& r : regions) {
    if (r.begin == begin && r.end == end && r.version == version) {
      return OB_OK;
    }
  }
  regions.erase(
      std::remove_if(regions.begin(), regions.end(),
                     [&](const ob::ImmutableWeightRegion& r) {
                       return r.begin < end && begin < r.end;
                     }),
      regions.end());
  const uint64_t gen = ctx->next_immutable_region_gen++;
  regions.push_back(ob::ImmutableWeightRegion{begin, end, version, gen});
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
  bool used_freetile = false;
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

  auto drain_one_harvested = [&](ob_status_t wst, const ob_invoke_stats_t& istats) -> ob_status_t {
    PendingInt8Tile& p = ring[ring_head];
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
          if (ctx->worker_pool.num_threads > 1u &&
              p.tm >= 8u && static_cast<size_t>(p.tm) * p.tn >= 16384u) {
            ctx->worker_pool.parallel_for_fn(p.tm, 4u, [&](uint32_t ms_lo, uint32_t ms_hi, uint32_t) {
              for (uint32_t mi = ms_lo; mi < ms_hi; ++mi) {
                int32_t* c_row = c_mn + static_cast<size_t>(p.m0 + mi) * n_dim + p.n0;
                const int8_t* o_row = raw_out + static_cast<size_t>(mi) * p.hw_n;
                ob::dequant_accum_row_i32_impl(c_row, o_row, p.tn, d_i32);
              }
            });
          } else {
            for (uint32_t mi = 0; mi < p.tm; ++mi) {
              int32_t* c_row = c_mn + static_cast<size_t>(p.m0 + mi) * n_dim + p.n0;
              const int8_t* o_row = raw_out + static_cast<size_t>(mi) * p.hw_n;
              ob::dequant_accum_row_i32_impl(c_row, o_row, p.tn, d_i32);
            }
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

  auto drain_one = [&]() -> ob_status_t {
    PendingInt8Tile& p = ring[ring_head];
    ob_invoke_stats_t istats {};
    const uint64_t tw0 = prof_on ? ob::monotonic_now_ns() : 0u;
    ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, &istats);
    if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - tw0);
    return drain_one_harvested(wst, istats);
  };

  auto try_drain_ready = [&]() -> ob_status_t {
    while (ring_count >= 2u) {
      ob_status_t wst = OB_OK;
      ob_invoke_stats_t istats {};
      if (!ob::device_try_wait_handle(ctx, ring[ring_head].handle, &wst, &istats)) {
        break;
      }
      ob_status_t dst = drain_one_harvested(wst, istats);
      if (dst != OB_OK) return dst;
    }
    return OB_OK;
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
              const size_t ext2_sz = is_lt64_tile
                  ? ob::pack_ext2_size_for_lt_64x(hw_n, hw_k)
                  : ob::pack_ext2_size_for_tile(hw_n, hw_k, false);
              bool cacheable = false;
              const size_t ext2_slab_bytes = ob::buffer_slab_size(ext2_sz);
              ob_status_t pst = ob::weight_cache_prepare_space_locked(
                  ctx, ext2_slab_bytes, ring_count, drain_one, &cacheable);
              if (pst != OB_OK) {
                drain_all_cleanup();
                return pst;
              }
              ob_buffer* wbuf = nullptr;
              ob_status_t wst = ob_buffer_alloc(
                  ctx, ext2_sz, OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL, &wbuf);
              if (wst == OB_OK) {
                wst = ob::buffer_sync_begin_write(wbuf);
                if (wst == OB_OK) {
                  uint8_t* dst_ext2 = static_cast<uint8_t*>(ob_buffer_data(wbuf));
                  if (is_lt64_tile) {
                    ob::pack_ext2_for_lt_64x_into(
                        dst_ext2, hw_n, hw_k, b_tile.data(), nullptr, &ctx->worker_pool);
                  } else {
                    ob::pack_ext2_for_tile_into(
                        dst_ext2, hw_n, hw_k, b_tile.data(), nullptr, false, &ctx->worker_pool);
                  }
                  wst = ob::buffer_sync_end_write(wbuf);
                }
              }
              if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
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
              w_tile->ext2_bytes = wbuf->slab.alloc_size;
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
            // Preserve exact unit output scale (s_out = 1.0f) for custom pluggable emitters.
            scale_d = 1u;
          }

          ob::PinnedWeightTile pinned_w(w_tile);

          if (ring_count >= ring_depth) {
            ob_status_t dst = drain_one();
            if (dst != OB_OK) {
              pinned_w.reset();
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
                hw_m, hw_k, hw_n, zp_a, zp_b, scale_d, slot % param_swap_slots,
                ctx->freetile_enabled);
          } else {
            desc.weights_nk = b_tile.data();
            const size_t packed_sz = ob_gemm_packed_weight_bytes(hw_k, hw_n);
            packed_tile.resize(packed_sz);
            ob_status_t pst = ob_gemm_pack_weights_int8(
                b_tile.data(), nullptr, hw_k, hw_n, packed_tile.data(), packed_tile.size());
            if (pst != OB_OK) {
              pinned_w.reset();
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
            if (entry && entry->emitted_by_freetile) used_freetile = true;
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
            uint64_t in_sync_ns = 0u;
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
                  if (prof_on) {
                    in_sync_ns = ob::monotonic_now_ns() - td0;
                    ctx->prof.dma_sync_ns += in_sync_ns;
                  }
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
            if (prof_on) {
              const uint64_t submit_total = ob::monotonic_now_ns() - ts0;
              ctx->prof.submit_ns += (submit_total > in_sync_ns) ? (submit_total - in_sync_ns) : 0u;
            }
            if (bst != OB_OK) {
              ob::release_tile_entry(ctx, entry);
              pinned_w.reset();
              drain_all_cleanup();
              return bst;
            }

            PendingInt8Tile& p = ring[slot];
            p.handle = exec_h;
            p.w_tile = pinned_w.release();
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
            tiles_dispatched += 1u;
            ring_count += 1u;

            if (!ctx->mock_mode) {
              tile_handled_by_hw = true;
              ob_status_t od_st = try_drain_ready();
              if (od_st != OB_OK) {
                drain_all_cleanup();
                return od_st;
              }
            }
          } else {
            pinned_w.reset();
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
    if (used_freetile) out_stats->flags |= OB_STAT_FLAG_FREETILE;
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
  bool has_any_q4_k = false;
  for (uint32_t j = 0; j < num_jobs; ++j) {
    if (!jobs[j].b_nk || !jobs[j].c_mn || jobs[j].n_dim == 0u ||
        !ob::quant::is_valid_type(jobs[j].is_q4_0)) {
      return OB_ERR_INVALID_ARG;
    }
    const ob_quant_type_t qt = static_cast<ob_quant_type_t>(jobs[j].is_q4_0);
    const uint32_t qk = ob_quant_block_elements(qt);
    if (qk == 0u || (k_dim % qk) != 0u) {
      return OB_ERR_INVALID_ARG;
    }
    if (qt == OB_QUANT_Q4_K) has_any_q4_k = true;
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
  const uint32_t w8a16_mode = ctx->w8a16_mode;
  const uint32_t ring_depth =
      std::max<uint32_t>(1u, std::min<uint32_t>(ctx->credit_size, OB_MAX_CREDITS));
  // Engage lock-free 1-helper SPSC async dequant ONLY for batched prefill (m_dim > 8);
  // keep M=1 decode 100% inline/lock-free with zero helper wakeups.
  const bool use_async_dq =
      ctx->async_dequant_enabled && !ctx->mock_mode && (m_dim > 8u) && (ring_depth >= 2u);

  // Skip upfront full-output memset when HW path is active: first K-slice (k0==0) uses
  // store-first fused NEON dequant (halving output DRAM bandwidth on single-K projections!).
  const bool skip_zero_init = (emitter != nullptr && !ctx->mock_mode);
  if (!skip_zero_init) {
    for (uint32_t j = 0; j < num_jobs; ++j) {
      std::memset(jobs[j].c_mn, 0, static_cast<size_t>(m_dim) * jobs[j].n_dim * sizeof(float));
    }
  }

  uint32_t tiles_dispatched = 0u;
  uint32_t cache_hits = 0u;
  uint64_t total_hw_us = 0u;
  bool used_emitter = false;
  bool used_freetile = false;
  bool used_cpu_fallback = false;

  struct InputSlotMeta {
    uint32_t refcnt = 0u;
    uint32_t k0 = 0u;
    uint32_t tk = 0u;
    uint32_t tm = 0u;
    bool has_outliers = false;
    ob::RowOutliers row0{};
    std::vector<ob::RowOutliers> rows_vec;
    uint32_t n_uniq = 0u;
    std::array<uint32_t, 16> uniq_k_inline{};
    std::vector<uint32_t> uniq_k_overflow;
    std::array<float, 64> a_sum_inline{};
    std::vector<float> a_sum_overflow;
  };

  struct OutlierMissItem {
    uint32_t u_idx;
    uint32_t k_off;
    float* dst;
  };

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
    bool is_first_k = false;
    bool is_w8a16 = false;
    float w8a16_prescale_inv = 1.0f;
    std::array<ob::FusedSubSlice, ob::kMaxFusedSubs> subs{};
    uint32_t scale_d_u32 = 1u;
    std::array<float, 64> s_a_inline{};
    std::vector<float> s_a_overflow;
    std::vector<float> s_b;

    // Dequant handoff payload (populated by main thread after TPU harvest, executed
    // either inline or FIFO-serially on the lock-free SPSC helper thread):
    uint64_t dq_seq = 0u;
    const ob_gemm_f32_quant_job_t* jobs_ptr = nullptr;
    const InputSlotMeta* im_ptr = nullptr;
    const void* raw_out = nullptr;
    uint32_t k_dim = 0u;
    float eff_scale_d = 1.0f;
    bool store_first = false;
    bool has_outliers = false;
    bool prof_on = false;
    std::array<bool, ob::kMaxFusedSubs> sub_done{};
    uint32_t dq_n_miss = 0u;
    std::array<const float*, 256> dq_col_ptrs_stack{};
    std::vector<const float*> dq_col_ptrs_vec;
    std::array<OutlierMissItem, 64> dq_misses_stack{};
    std::vector<OutlierMissItem> dq_misses_vec;
    std::vector<float> spill_scratch;
    uint64_t dq_ns = 0u;
    uint64_t dq_sync_ns = 0u;
    ob_status_t dq_status = OB_OK;

    static void run_dequant_job(void* raw) {
      PendingF32Tile* p = static_cast<PendingF32Tile*>(raw);
      const uint64_t tq0 = p->prof_on ? ob::monotonic_now_ns() : 0u;
      const float* sb_base = p->w_tile ? p->w_tile->row_s_b.data() : p->s_b.data();
      const size_t sb_len = p->w_tile ? p->w_tile->row_s_b.size() : p->s_b.size();
      const float* mu_base = (sb_len >= static_cast<size_t>(p->tn_total) * 2u)
                                 ? (sb_base + p->tn_total)
                                 : nullptr;
      const ob_gemm_f32_quant_job_t* jobs_p = p->jobs_ptr;
      const InputSlotMeta* im = p->im_ptr;
      const float* a_sum_ptr =
          (mu_base && im)
              ? ((p->tm <= im->a_sum_inline.size()) ? im->a_sum_inline.data() : im->a_sum_overflow.data())
              : nullptr;

      if (p->is_w8a16) {
        const float* y_f32 = static_cast<const float*>(p->raw_out);
        const float a_sum_0 = a_sum_ptr ? a_sum_ptr[0] : 0.0f;
        for (uint32_t si = 0u; si < p->num_subs; ++si) {
          const ob::FusedSubSlice& sub = p->subs[si];
          const ob_gemm_f32_quant_job_t& job = jobs_p[sub.job_idx];
          const float* sb_ptr = sb_base + sub.out_off;
          float* c_row = job.c_mn + static_cast<size_t>(p->m0) * job.n_dim + sub.n0;
          ob::w8a16_pt_epilogue_f32(
              c_row, y_f32 + sub.out_off, sb_ptr, sub.tn,
              p->w8a16_prescale_inv, p->store_first);
          if (mu_base && a_sum_0 != 0.0f) {
            ob::add_scaled_offset_row_f32(c_row, mu_base + sub.out_off, sub.tn, a_sum_0);
          }
        }
      } else {
        const int8_t* raw_i8 = static_cast<const int8_t*>(p->raw_out);
        const float* sa_ptr =
            (p->tm <= p->s_a_inline.size()) ? p->s_a_inline.data() : p->s_a_overflow.data();
        const bool has_out = p->has_outliers && im != nullptr;
        const ob::RowOutliers* rows_ptr =
            has_out ? ((p->tm <= 1u) ? &im->row0 : im->rows_vec.data()) : nullptr;

        if (has_out && p->dq_n_miss > 0u) {
          const OutlierMissItem* misses_ptr =
              p->dq_misses_vec.empty() ? p->dq_misses_stack.data() : p->dq_misses_vec.data();
          for (uint32_t i = 0u; i < p->dq_n_miss; ++i) {
            ob::extract_outlier_col_impl(
                jobs_p, p->subs.data(), p->num_subs, p->k_dim, im->k0,
                misses_ptr[i].k_off, misses_ptr[i].dst);
          }
        }

        const float* const* col_ptrs =
            p->dq_col_ptrs_vec.empty() ? p->dq_col_ptrs_stack.data() : p->dq_col_ptrs_vec.data();
        for (uint32_t mi = 0u; mi < p->tm; ++mi) {
          const float sa_d = sa_ptr[mi] * p->eff_scale_d;
          const uint32_t n_out = has_out ? rows_ptr[mi].count : 0u;
          const float* row_cols[ob::kMaxRowOutliers];
          for (uint32_t oi = 0u; oi < n_out; ++oi) {
            row_cols[oi] = col_ptrs[rows_ptr[mi].uniq_idx[oi]];
          }
          for (uint32_t si = 0u; si < p->num_subs; ++si) {
            const ob::FusedSubSlice& sub = p->subs[si];
            const ob_gemm_f32_quant_job_t& job = jobs_p[sub.job_idx];
            const float* sb_ptr = sb_base + sub.out_off;
            float* c_row = job.c_mn + static_cast<size_t>(p->m0 + mi) * job.n_dim + sub.n0;
            if (p->sub_done[si]) {
              for (uint32_t oi = 0u; oi < n_out; ++oi) {
                const float* col_p = row_cols[oi] + sub.out_off;
                const float x_val = rows_ptr[mi].val[oi];
                for (uint32_t ni = 0u; ni < sub.tn; ++ni) {
                  c_row[ni] += col_p[ni] * x_val;
                }
              }
              continue;
            }
            const int8_t* o_row = raw_i8 + static_cast<size_t>(mi) * p->hw_n + sub.out_off;
            if (n_out > 0u) {
              const float* sub_cols[ob::kMaxRowOutliers];
              for (uint32_t oi = 0u; oi < n_out; ++oi) {
                sub_cols[oi] = row_cols[oi] + sub.out_off;
              }
              ob::dequant_outliers_row_f32_impl(
                  c_row, o_row, sb_ptr, sub.tn, sa_d, p->store_first,
                  sub_cols, rows_ptr[mi].val, n_out);
            } else {
              ob::dequant_outliers_row_f32_impl(
                  c_row, o_row, sb_ptr, sub.tn, sa_d, p->store_first,
                  nullptr, nullptr, 0u);
            }
            if (mu_base && a_sum_ptr) {
              ob::add_scaled_offset_row_f32(c_row, mu_base + sub.out_off, sub.tn, a_sum_ptr[mi]);
            }
          }
        }
      }
      if (p->prof_on) p->dq_ns = ob::monotonic_now_ns() - tq0;

      const uint64_t td1 = p->prof_on ? ob::monotonic_now_ns() : 0u;
      p->dq_status = ob::buffer_sync_end_read(p->out_buf);
      if (p->prof_on) p->dq_sync_ns = ob::monotonic_now_ns() - td1;
    }
  };

  std::array<PendingF32Tile, OB_MAX_CREDITS> ring{};
  std::array<InputSlotMeta, OB_MAX_CREDITS> in_slots{};
  uint64_t sub_seq = 0u;
  uint64_t tpu_seq = 0u;
  uint64_t reap_seq = 0u;
  uint64_t last_dq_seq = 0u;
  ob_status_t first_dq_err = OB_OK;

  auto release_reaped_slot = [&](PendingF32Tile& rp) {
    if (prof_on) {
      ctx->prof.dequant_ns += rp.dq_ns;
      ctx->prof.dma_sync_ns += rp.dq_sync_ns;
    }
    if (rp.dq_status != OB_OK && first_dq_err == OB_OK) {
      first_dq_err = rp.dq_status;
    }
    if (rp.w_tile) {
      if (rp.w_tile->in_use > 0u) rp.w_tile->in_use -= 1u;
      if (!rp.w_tile->in_cache && rp.w_tile->in_use == 0u) {
        if (rp.w_tile->ext2_buf) ob_buffer_free(rp.w_tile->ext2_buf);
        delete rp.w_tile;
      }
    }
    if (rp.in_buf) ob_buffer_free(rp.in_buf);
    if (rp.out_buf) ob_buffer_free(rp.out_buf);
    if (in_slots[rp.in_slot].refcnt > 0u) in_slots[rp.in_slot].refcnt -= 1u;
    rp.handle = 0u;
    rp.dq_seq = 0u;
    rp.w_tile = nullptr;
    rp.in_buf = nullptr;
    rp.out_buf = nullptr;
    rp.num_subs = 0u;
    rp.is_first_k = false;
    rp.is_w8a16 = false;
    rp.dq_ns = 0u;
    rp.dq_sync_ns = 0u;
    rp.dq_status = OB_OK;
  };

  auto reap_ready = [&]() {
    while (reap_seq < tpu_seq) {
      PendingF32Tile& rp = ring[static_cast<uint32_t>(reap_seq % ring_depth)];
      if (rp.dq_seq != 0u && ctx->async_dequant.completed_count() < rp.dq_seq) {
        break;
      }
      release_reaped_slot(rp);
      ++reap_seq;
    }
  };

  auto wait_and_reap_one_dq = [&]() {
    if (reap_seq >= tpu_seq) return;
    PendingF32Tile& rp = ring[static_cast<uint32_t>(reap_seq % ring_depth)];
    if (rp.dq_seq != 0u) {
      ctx->async_dequant.wait_until(rp.dq_seq);
    }
    release_reaped_slot(rp);
    ++reap_seq;
  };

  auto harvest_one_with_result = [&](ob_status_t wst, const ob_invoke_stats_t& istats) -> ob_status_t {
    PendingF32Tile& p = ring[static_cast<uint32_t>(tpu_seq % ring_depth)];
    p.dq_seq = 0u;
    p.dq_ns = 0u;
    p.dq_sync_ns = 0u;
    p.dq_status = OB_OK;
    ob_status_t rst = OB_OK;
    bool dispatched_async = false;

    if (wst == OB_OK) {
      total_hw_us += istats.hw_execution_us;
      if (!ctx->mock_mode) {
        uint32_t eff_d = p.scale_d_u32;
        const uint64_t td0 = prof_on ? ob::monotonic_now_ns() : 0u;
        rst = ob::buffer_sync_begin_read(p.out_buf);
        if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - td0);

        if (rst == OB_OK) {
          const void* raw_out_ptr = ob_buffer_data(p.out_buf);
          bool read_active = true;
          p.sub_done.fill(false);
          const float* sa_ptr =
              (p.tm <= p.s_a_inline.size()) ? p.s_a_inline.data() : p.s_a_overflow.data();
          const float* sb_base = p.w_tile ? p.w_tile->row_s_b.data() : p.s_b.data();
          const size_t sb_len = p.w_tile ? p.w_tile->row_s_b.size() : p.s_b.size();
          const float* mu_base = (sb_len >= static_cast<size_t>(p.tn_total) * 2u)
                                     ? (sb_base + p.tn_total)
                                     : nullptr;
          const InputSlotMeta& im_cur = in_slots[p.in_slot];
          const float* a_sum_cur =
              mu_base ? ((p.tm <= im_cur.a_sum_inline.size()) ? im_cur.a_sum_inline.data()
                                                              : im_cur.a_sum_overflow.data())
                      : nullptr;
          const bool store_first_now = p.is_first_k && skip_zero_init;
          auto dequant_sub = [&](const int8_t* raw_i8, uint32_t si, uint32_t scale_d_cur) {
            if (use_async_dq && last_dq_seq > 0u) {
              // Ensure all earlier K-slice SPSC dequant jobs complete before writing on main thread.
              ctx->async_dequant.wait_until(last_dq_seq);
            }
            const uint64_t tq0 = prof_on ? ob::monotonic_now_ns() : 0u;
            const float eff_scale_d = static_cast<float>(scale_d_cur);
            const ob::FusedSubSlice& sub = p.subs[si];
            const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
            const float* sb_ptr = sb_base + sub.out_off;
            for (uint32_t mi = 0; mi < p.tm; ++mi) {
              const float sa_d = sa_ptr[mi] * eff_scale_d;
              float* c_row = job.c_mn + static_cast<size_t>(p.m0 + mi) * job.n_dim + sub.n0;
              const int8_t* o_row = raw_i8 + static_cast<size_t>(mi) * p.hw_n + sub.out_off;
              ob::dequant_outliers_row_f32_impl(
                  c_row, o_row, sb_ptr, sub.tn, sa_d, store_first_now,
                  nullptr, nullptr, 0u);
              if (mu_base && a_sum_cur) {
                ob::add_scaled_offset_row_f32(c_row, mu_base + sub.out_off, sub.tn, a_sum_cur[mi]);
              }
            }
            p.sub_done[si] = true;
            if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tq0);
          };

          // Saturation-proof on-TPU retry (W8A8 only; W8A16 has full FP32 accumulator!).
          // For fused tiles, preserve unsaturated sibling projections at their original scale
          // before raising eff_d for any saturated sibling.
          if (!p.is_w8a16 && builtin_active && p.w_tile != nullptr) {
            const int8_t* raw_out = static_cast<const int8_t*>(raw_out_ptr);
            static constexpr double kRetryMults[5] = {1.25, 1.5, 2.0, 4.0, 4.0};
            for (uint32_t retry = 0u; retry < 5u && eff_d < (1u << 24); ++retry) {
              bool sat = false;
              if (p.num_subs == 1u) {
                const uint64_t tsat0 = prof_on ? ob::monotonic_now_ns() : 0u;
                sat = ob::tile_output_is_saturated_impl(raw_out, p.tm, p.tn_total, p.hw_n);
                if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tsat0);
              } else {
                std::array<bool, ob::kMaxFusedSubs> sub_is_sat {};
                for (uint32_t si = 0u; si < p.num_subs; ++si) {
                  if (p.sub_done[si]) continue;
                  const ob::FusedSubSlice& sub = p.subs[si];
                  const uint64_t tsat0 = prof_on ? ob::monotonic_now_ns() : 0u;
                  const bool sub_sat = ob::tile_output_is_saturated_impl(
                      raw_out + sub.out_off, p.tm, sub.tn, p.hw_n);
                  if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tsat0);
                  sub_is_sat[si] = sub_sat;
                  if (sub_sat) sat = true;
                }
                if (sat) {
                  // Only inline-dequant unsaturated siblings when a saturation retry will
                  // overwrite p.out_buf at a higher eff_d; otherwise keep them on async dequant.
                  for (uint32_t si = 0u; si < p.num_subs; ++si) {
                    if (!p.sub_done[si] && !sub_is_sat[si]) {
                      dequant_sub(raw_out, si, eff_d);
                    }
                  }
                }
              }
              if (!sat) break;

              const uint64_t tde0 = prof_on ? ob::monotonic_now_ns() : 0u;
              rst = ob::buffer_sync_end_read(p.out_buf);
              if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - tde0);
              read_active = false;
              if (rst != OB_OK) break;

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
                  p.hw_m, p.hw_k, p.hw_n, 0, 0, eff_d, p.slot % param_swap_slots,
                  ctx->freetile_enabled);

              const uint64_t trs0 = prof_on ? ob::monotonic_now_ns() : 0u;
              ob::CachedGraph* rentry = nullptr;
              bool rhit = false;
              rst = ob::acquire_tile_entry(ctx, emitter, emitter_ud, rdesc, &rentry, &rhit);
              if (rst != OB_OK) break;
              if (rentry && rentry->emitted_by_freetile) used_freetile = true;
              if (rhit) cache_hits += 1u;
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
              tiles_dispatched += 1u;
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
              raw_out_ptr = ob_buffer_data(p.out_buf);
              raw_out = static_cast<const int8_t*>(raw_out_ptr);
              read_active = true;
            }
          }

          if (rst == OB_OK && read_active) {
            p.jobs_ptr = jobs;
            p.im_ptr = &in_slots[p.in_slot];
            p.raw_out = raw_out_ptr;
            p.k_dim = k_dim;
            p.eff_scale_d = static_cast<float>(eff_d);
            p.store_first = p.is_first_k && skip_zero_init;
            p.prof_on = prof_on;
            p.dq_n_miss = 0u;
            p.dq_col_ptrs_vec.clear();
            p.dq_misses_vec.clear();

            const InputSlotMeta& im = in_slots[p.in_slot];
            p.has_outliers = (!p.is_w8a16) && im.has_outliers && (im.n_uniq > 0u);

            // Resolve outlier_cols map lookups/emplacements on main thread (single writer to map!)
            // so the SPSC helper only touches stable heap float* buffers with zero contention.
            if (p.has_outliers) {
              const uint64_t tmap0 = prof_on ? ob::monotonic_now_ns() : 0u;
              const uint32_t n_uniq = im.n_uniq;
              const uint32_t* uniq_ptr =
                  im.uniq_k_overflow.empty() ? im.uniq_k_inline.data() : im.uniq_k_overflow.data();
              const float** col_ptrs = p.dq_col_ptrs_stack.data();
              if (n_uniq > p.dq_col_ptrs_stack.size()) {
                p.dq_col_ptrs_vec.resize(n_uniq, nullptr);
                col_ptrs = p.dq_col_ptrs_vec.data();
              }

              bool w_budget_refreshed = false;
              size_t w_budget = ctx->weight_cache_max_bytes;
              const size_t max_cols = ctx->max_cached_outlier_cols;
              const size_t col_bytes = static_cast<size_t>(p.tn_total) * sizeof(float);

              uint32_t n_miss = 0u;
              uint32_t n_spill = 0u;
              auto push_miss = [&](uint32_t u_idx, uint32_t k_off, float* dst) {
                if (n_miss < p.dq_misses_stack.size() && p.dq_misses_vec.empty()) {
                  p.dq_misses_stack[n_miss++] = {u_idx, k_off, dst};
                } else {
                  if (p.dq_misses_vec.empty()) {
                    p.dq_misses_vec.assign(p.dq_misses_stack.data(),
                                           p.dq_misses_stack.data() + n_miss);
                  }
                  p.dq_misses_vec.push_back({u_idx, k_off, dst});
                  n_miss = static_cast<uint32_t>(p.dq_misses_vec.size());
                }
              };

              for (uint32_t u = 0u; u < n_uniq; ++u) {
                const uint32_t k_off = uniq_ptr[u];
                const float* w_col = nullptr;
                if (p.w_tile) {
                  auto it = p.w_tile->outlier_cols.find(k_off);
                  if (it != p.w_tile->outlier_cols.end() && it->second.size() >= p.tn_total) {
                    w_col = it->second.data();
                  } else if (p.w_tile->in_cache &&
                             p.w_tile->outlier_cols.size() < max_cols) {
                    if (!w_budget_refreshed) {
                      w_budget = ob::compute_weight_cache_budget_bytes(ctx);
                      ctx->weight_cache_max_bytes = w_budget;
                      w_budget_refreshed = true;
                    }
                    const size_t outlier_sub_cap =
                        std::min<size_t>(256ULL * 1024ULL * 1024ULL, w_budget / 6u);
                    const size_t total_soft_cap = (w_budget / 8u) * 7u;
                    if (ctx->outlier_cache_bytes + col_bytes <= outlier_sub_cap &&
                        ctx->weight_cache_bytes + col_bytes <= total_soft_cap) {
                      std::vector<float> col(p.tn_total);
                      auto res = p.w_tile->outlier_cols.emplace(k_off, std::move(col));
                      if (res.second) {
                        p.w_tile->outlier_cols_bytes += col_bytes;
                        ctx->outlier_cache_bytes += col_bytes;
                        ctx->weight_cache_bytes += col_bytes;
                        if (ctx->weight_cache_bytes > ctx->weight_cache_peak_bytes) {
                          ctx->weight_cache_peak_bytes = ctx->weight_cache_bytes;
                        }
                      }
                      float* dst_p = res.first->second.data();
                      w_col = dst_p;
                      push_miss(u, k_off, dst_p);
                    }
                  }
                }
                if (!w_col) {
                  push_miss(u, k_off, nullptr);
                  ++n_spill;
                } else {
                  col_ptrs[u] = w_col;
                }
              }

              OutlierMissItem* misses_ptr =
                  p.dq_misses_vec.empty() ? p.dq_misses_stack.data() : p.dq_misses_vec.data();
              if (n_spill > 0u) {
                p.spill_scratch.resize(static_cast<size_t>(n_spill) * p.tn_total);
                uint32_t sp = 0u;
                for (uint32_t i = 0u; i < n_miss; ++i) {
                  if (!misses_ptr[i].dst) {
                    float* dst_p = p.spill_scratch.data() +
                                   static_cast<size_t>(sp++) * p.tn_total;
                    misses_ptr[i].dst = dst_p;
                    col_ptrs[misses_ptr[i].u_idx] = dst_p;
                  }
                }
              }
              p.dq_n_miss = n_miss;
              if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tmap0);
            }

            if (use_async_dq) {
              // Hand off NEON dequant + outlier FMA + buffer_sync_end_read to pinned SPSC helper!
              p.dq_seq = ctx->async_dequant.push(&PendingF32Tile::run_dequant_job, &p);
              last_dq_seq = p.dq_seq;
              dispatched_async = true;
            } else {
              PendingF32Tile::run_dequant_job(&p);
              rst = p.dq_status;
            }
          } else if (read_active) {
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

    p.handle = 0u;
    ++tpu_seq;
    if (!dispatched_async) {
      // Completed inline (or mock/error): reap immediately when all earlier slots are reaped
      reap_ready();
    }
    if (wst != OB_OK) return wst;
    if (rst != OB_OK) return rst;
    return first_dq_err;
  };

  auto harvest_one_blocking = [&]() -> ob_status_t {
    PendingF32Tile& p = ring[static_cast<uint32_t>(tpu_seq % ring_depth)];
    ob_invoke_stats_t istats {};
    const uint64_t tw0 = prof_on ? ob::monotonic_now_ns() : 0u;
    ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, &istats);
    if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - tw0);
    return harvest_one_with_result(wst, istats);
  };

  // Non-blocking harvest: feed the SPSC helper immediately as soon as any TPU tile finishes!
  auto try_harvest_ready = [&]() -> ob_status_t {
    const uint64_t min_inflight = use_async_dq ? 1u : 2u;
    while (sub_seq - tpu_seq >= min_inflight) {
      PendingF32Tile& p = ring[static_cast<uint32_t>(tpu_seq % ring_depth)];
      ob_status_t wst = OB_OK;
      ob_invoke_stats_t istats {};
      if (!ob::device_try_wait_handle(ctx, p.handle, &wst, &istats)) {
        break;
      }
      ob_status_t hst = harvest_one_with_result(wst, istats);
      if (hst != OB_OK) return hst;
    }
    reap_ready();
    return first_dq_err;
  };

  // Ensure at least one free ring slot (sub_seq - reap_seq < ring_depth).
  auto ensure_ring_slot = [&]() -> ob_status_t {
    while (sub_seq - reap_seq >= ring_depth) {
      reap_ready();
      if (sub_seq - reap_seq < ring_depth) break;
      if (reap_seq < tpu_seq) {
        wait_and_reap_one_dq();
      } else {
        ob_status_t hst = harvest_one_blocking();
        if (hst != OB_OK) return hst;
      }
    }
    return first_dq_err;
  };

  auto drain_all_normal = [&]() -> ob_status_t {
    while (tpu_seq < sub_seq) {
      ob_status_t hst = harvest_one_blocking();
      if (hst != OB_OK) return hst;
      reap_ready();
    }
    if (use_async_dq && last_dq_seq > 0u) {
      ctx->async_dequant.wait_until(last_dq_seq);
    }
    reap_ready();
    return first_dq_err;
  };

  auto drain_all_cleanup = [&]() {
    if (use_async_dq && last_dq_seq > 0u) {
      ctx->async_dequant.wait_until(last_dq_seq);
    }
    reap_ready();
    while (tpu_seq < sub_seq) {
      PendingF32Tile& p = ring[static_cast<uint32_t>(tpu_seq % ring_depth)];
      if (p.handle != 0u) {
        ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, nullptr);
        if (wst != OB_OK) {
          ob::detach_timed_out_buffers_locked(ctx, p.handle, p.w_tile, p.in_buf, p.out_buf);
        }
      }
      p.dq_seq = 0u;
      p.dq_status = OB_OK;
      ++tpu_seq;
    }
    reap_ready();
  };

  auto prepare_cache_space = [&](size_t incoming_bytes, bool* out_cacheable) -> ob_status_t {
    uint32_t inflight = static_cast<uint32_t>(sub_seq - reap_seq);
    auto drain_one_slot = [&]() -> ob_status_t {
      reap_ready();
      if (reap_seq < tpu_seq) {
        wait_and_reap_one_dq();
        inflight = static_cast<uint32_t>(sub_seq - reap_seq);
        return first_dq_err;
      }
      if (tpu_seq < sub_seq) {
        ob_status_t hst = harvest_one_blocking();
        if (hst != OB_OK) return hst;
        if (reap_seq < tpu_seq) wait_and_reap_one_dq();
        inflight = static_cast<uint32_t>(sub_seq - reap_seq);
        return first_dq_err;
      }
      inflight = static_cast<uint32_t>(sub_seq - reap_seq);
      return first_dq_err;
    };
    return ob::weight_cache_prepare_space_locked(
        ctx, incoming_bytes, inflight, drain_one_slot, out_cacheable);
  };

  std::vector<int8_t>& a_q_tile = ctx->gemm_aq_scratch;
  std::vector<int8_t>& b_q_tile = ctx->gemm_bq_scratch;
  std::vector<float>& row_s_a = ctx->gemm_sa_scratch;
  std::vector<float> a_f32_tile;
  std::vector<float> row_s_b;
  std::vector<uint8_t> packed_tile;
  std::vector<ob::ScheduledGroup> groups;

  const bool allow_lt_64x = ob::compute_allow_lt_64x(
      k_dim, jobs, num_jobs, max_k, max_n, builtin_active, lt_enabled, blt_enabled,
      horiz_fusion, &groups);

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
    uint32_t cached_a_k0 = 0xFFFFFFFFu;
    uint32_t cached_a_tk = 0u;
    uint32_t cached_aq_hw_k = 0u;
    bool cached_a_is_w8a16 = false;
    float cached_a_w8a16_prescale_inv = 1.0f;
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

      // Helper to split a rejected multi-sub horizontal fusion cleanly into 1-sub groups.
      auto fallback_split_subs = [&]() -> ob_status_t {
        std::vector<ob::ScheduledGroup> split_groups;
        ob::split_scheduled_group_subs(grp, hw_m, max_k, max_n, builtin_active, &split_groups);
        for (const ob::ScheduledGroup& sg : split_groups) {
          ob_status_t sst = self_ref(sg, self_ref);
          if (sst != OB_OK) return sst;
        }
        return OB_OK;
      };

      bool tile_handled_by_hw = false;
      if (emitter != nullptr) {
        // Decide whether this tile uses opt-in native 7-chunk PT W8A16(FP16)
        // (full FP32 accumulator, zero activation quantization).
        const bool tile_use_w8a16 =
            builtin_active && !ctx->mock_mode && w8a16_mode == 1u &&
            hw_m == 1u && ob::is_w8a16_pt_1x_shape(1u, hw_k, hw_n);

        // 1. Quantize/prepare shared A ONCE per (k0, tk, hw_k, tile_use_w8a16) across sibling tiles
        if (cached_a_k0 != k0 || cached_a_tk != tk || cached_aq_hw_k != hw_k ||
            cached_a_is_w8a16 != tile_use_w8a16) {
          ob_status_t rslot_st = ensure_ring_slot();
          if (rslot_st != OB_OK) {
            drain_all_cleanup();
            return rslot_st;
          }
          uint32_t chosen = 0u;
          for (uint32_t c = 0u; c < ring_depth; ++c) {
            if (in_slots[c].refcnt == 0u) { chosen = c; break; }
          }

          const uint64_t tq0 = prof_on ? ob::monotonic_now_ns() : 0u;
          InputSlotMeta& im = in_slots[chosen];
          im.k0 = k0;
          im.tk = tk;
          im.tm = tm;
          im.has_outliers = false;
          im.n_uniq = 0u;
          im.uniq_k_overflow.clear();

          if (tile_use_w8a16) {
            a_f32_tile.resize(hw_k);
            const float* a_row = a_mk + static_cast<size_t>(m0) * k_dim + k0;
            cached_a_w8a16_prescale_inv =
                ob::prepare_w8a16_input_row_f32(a_row, tk, hw_k, a_f32_tile.data());
            if (has_any_q4_k) {
              im.a_sum_inline[0] = ob::compute_row_sum_excluding_outliers(a_row, tk, nullptr);
            }
            cached_a_rms_shift = 0u;
          } else {
            ob::RowOutliers* rows_dst = &im.row0;
            if (tm > 1u) {
              im.rows_vec.resize(tm);
              rows_dst = im.rows_vec.data();
            }

            a_q_tile.resize(static_cast<size_t>(hw_m) * hw_k);
            if (tm < hw_m) {
              std::memset(a_q_tile.data() + static_cast<size_t>(tm) * hw_k, 0,
                          static_cast<size_t>(hw_m - tm) * hw_k);
            }

            uint64_t sq_stack[64]{};
            std::vector<uint64_t> sq_vec;
            uint64_t* sq_ptr = sq_stack;
            if (tm > 64u) {
              sq_vec.assign(tm, 0ULL);
              sq_ptr = sq_vec.data();
            }

            if (ctx->worker_pool.num_threads > 1u &&
                tm >= 8u && static_cast<size_t>(tm) * tk >= 16384u) {
              ctx->worker_pool.parallel_for_fn(tm, 4u, [&](uint32_t ms_lo, uint32_t ms_hi, uint32_t) {
                for (uint32_t mi = ms_lo; mi < ms_hi; ++mi) {
                  const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0;
                  int8_t* aq_row = a_q_tile.data() + static_cast<size_t>(mi) * hw_k;
                  ob::quantize_activation_row_split_fast(
                      a_row, tk, aq_row, &row_s_a[mi],
                      builtin_active ? &rows_dst[mi] : nullptr,
                      builtin_active ? &sq_ptr[mi] : nullptr);
                  if (tk < hw_k) std::memset(aq_row + tk, 0, hw_k - tk);
                }
              });
            } else {
              for (uint32_t mi = 0u; mi < tm; ++mi) {
                const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0;
                int8_t* aq_row = a_q_tile.data() + static_cast<size_t>(mi) * hw_k;
                ob::quantize_activation_row_split_fast(
                    a_row, tk, aq_row, &row_s_a[mi],
                    builtin_active ? &rows_dst[mi] : nullptr,
                    builtin_active ? &sq_ptr[mi] : nullptr);
                if (tk < hw_k) std::memset(aq_row + tk, 0, hw_k - tk);
              }
            }

            if (builtin_active) {
              uint64_t max_sq = 0ULL;
              uint32_t n_uniq = 0u;
              for (uint32_t mi = 0u; mi < tm; ++mi) {
                if (sq_ptr[mi] > max_sq) max_sq = sq_ptr[mi];
                ob::RowOutliers& ro = rows_dst[mi];
                if (ro.count == 0u) continue;
                im.has_outliers = true;
                for (uint32_t oi = 0u; oi < ro.count; ++oi) {
                  const uint32_t ko = ro.k_off[oi];
                  const uint32_t* up =
                      im.uniq_k_overflow.empty() ? im.uniq_k_inline.data() : im.uniq_k_overflow.data();
                  uint32_t found = n_uniq;
                  for (uint32_t u = 0u; u < n_uniq; ++u) {
                    if (up[u] == ko) { found = u; break; }
                  }
                  if (found == n_uniq) {
                    if (n_uniq < im.uniq_k_inline.size() && im.uniq_k_overflow.empty()) {
                      im.uniq_k_inline[n_uniq] = ko;
                    } else {
                      if (im.uniq_k_overflow.empty()) {
                        im.uniq_k_overflow.assign(im.uniq_k_inline.data(),
                                                  im.uniq_k_inline.data() + n_uniq);
                      }
                      im.uniq_k_overflow.push_back(ko);
                    }
                    ++n_uniq;
                  }
                  ro.uniq_idx[oi] = static_cast<uint16_t>(found);
                }
              }
              im.n_uniq = n_uniq;
              cached_a_rms_shift = ob::compute_rms_shift_from_max_sq(max_sq, tk);
            } else {
              cached_a_rms_shift = 0u;
            }
            if (has_any_q4_k) {
              float* a_sum_dst = im.a_sum_inline.data();
              if (tm > im.a_sum_inline.size()) {
                im.a_sum_overflow.resize(tm);
                a_sum_dst = im.a_sum_overflow.data();
              }
              for (uint32_t mi = 0u; mi < tm; ++mi) {
                const float* a_row = a_mk + static_cast<size_t>(m0 + mi) * k_dim + k0;
                a_sum_dst[mi] = ob::compute_row_sum_excluding_outliers(
                    a_row, tk, builtin_active ? &rows_dst[mi] : nullptr);
              }
            }
          }

          if (prof_on) ctx->prof.quant_ns += (ob::monotonic_now_ns() - tq0);
          cached_a_k0 = k0;
          cached_a_tk = tk;
          cached_aq_hw_k = hw_k;
          cached_a_is_w8a16 = tile_use_w8a16;
          active_in_slot = chosen;
          active_in_buf = nullptr;
        }

        // 2. Hash & lookup (or cold-pack) ext[2] weight tile (single or horizontally fused!)
        ob::CachedWeightTile* w_tile = nullptr;
        uint32_t scale_d = 1u;
        bool grp_has_q4_k = false;
        for (uint32_t si = 0u; si < num_subs; ++si) {
          if (jobs[grp.subs[si].job_idx].is_q4_0 == static_cast<uint8_t>(OB_QUANT_Q4_K)) {
            grp_has_q4_k = true;
            break;
          }
        }
        if (builtin_active) {
          const bool is_lt64_tile = (!tile_use_w8a16) && ob::is_lt_64x_shape(hw_m, hw_k, hw_n);
          const uint32_t pack_fmt = tile_use_w8a16 ? 2u : (is_lt64_tile ? 1u : 0u);
          const uint64_t th0 = prof_on ? ob::monotonic_now_ns() : 0u;
          const ob::WeightTileKey wh = ob::compute_quant_group_weight_key(
              ctx, k_dim, jobs, grp, pack_fmt, strict_hash);
          w_tile = ob::weight_cache_find_locked(ctx, wh);
          if (prof_on) ctx->prof.hash_ns += (ob::monotonic_now_ns() - th0);

          if (w_tile && w_tile->rejected_fusion) {
            return fallback_split_subs();
          }

          if (!w_tile) {
            const uint64_t tp0 = prof_on ? ob::monotonic_now_ns() : 0u;
            bool rejected_fusion = false;
            bool skipped_budget = false;
            ob_status_t pst = ob::pack_quant_group_tile_locked(
                ctx, k_dim, jobs, grp, tile_use_w8a16, is_lt64_tile, grp_has_q4_k, wh,
                false, &ctx->worker_pool, &b_q_tile, prepare_cache_space,
                &rejected_fusion, &skipped_budget, &scale_d, &w_tile);
            if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
            if (pst != OB_OK) {
              drain_all_cleanup();
              return pst;
            }
            if (rejected_fusion) {
              return fallback_split_subs();
            }
          } else {
            scale_d = w_tile->scale_d;
          }
        } else {
          // Non-builtin pluggable emitter path (always 1-sub)
          const ob::FusedSubSlice& sub = grp.subs[0];
          const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
          const ob_quant_type_t qtype = static_cast<ob_quant_type_t>(job.is_q4_0);
          b_q_tile.assign(static_cast<size_t>(hw_n) * hw_k, 0);
          row_s_b.assign(sub.tn, 0.0f);
          std::vector<float> offset_vec(grp_has_q4_k ? sub.tn : 0u, 0.0f);
          if (ob::quant::requant_rows(qtype, job.b_nk, k_dim, sub.n0, sub.tn, k0, tk, hw_k,
                                      b_q_tile.data(), row_s_b.data(),
                                      grp_has_q4_k ? offset_vec.data() : nullptr,
                                      &ctx->worker_pool, nullptr)) {
            row_s_b.insert(row_s_b.end(), offset_vec.begin(), offset_vec.end());
          }
          scale_d = ob::compute_tile_scale_D(b_q_tile.data(), sub.tn, tk, hw_k);
        }

        if (!tile_use_w8a16 && builtin_active && cached_a_rms_shift > 0u) {
          scale_d = ob::quant_scale_D_grid(
              static_cast<double>(std::max<uint32_t>(1u, scale_d >> cached_a_rms_shift)));
        }

        ob::PinnedWeightTile pinned_w(w_tile);

        ob_status_t rslot_st = ensure_ring_slot();
        if (rslot_st != OB_OK) {
          pinned_w.reset();
          drain_all_cleanup();
          return rslot_st;
        }
        const uint32_t slot = static_cast<uint32_t>(sub_seq % ring_depth);

        const uint64_t ts0 = prof_on ? ob::monotonic_now_ns() : 0u;
        ob::CachedGraph* entry = nullptr;
        bool was_hit = false;
        ob_status_t gst = OB_OK;
        if (tile_use_w8a16) {
          gst = ob::acquire_w8a16_pt_1x_entry(
              ctx, hw_k, hw_n, 1.0f, slot % param_swap_slots, &entry, &was_hit);
        } else {
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
                hw_m, hw_k, hw_n, 0, 0, scale_d, slot % param_swap_slots,
                ctx->freetile_enabled);
          } else {
            desc.weights_nk = b_q_tile.data();
            const size_t packed_sz = ob_gemm_packed_weight_bytes(hw_k, hw_n);
            packed_tile.resize(packed_sz);
            ob_status_t pst = ob_gemm_pack_weights_int8(
                b_q_tile.data(), nullptr, hw_k, hw_n, packed_tile.data(), packed_tile.size());
            if (pst != OB_OK) {
              pinned_w.reset();
              drain_all_cleanup();
              return pst;
            }
            desc.packed_params = packed_tile.data();
            desc.packed_params_bytes = packed_tile.size();
            desc.cache_key = ob::hash_tile_key(
                hw_m, hw_k, hw_n, 0, 0, scale_d, b_q_tile.data(), b_q_tile.size());
          }
          gst = ob::acquire_tile_entry(ctx, emitter, emitter_ud, desc, &entry, &was_hit);
        }

        if (gst == OB_OK) {
          used_emitter = true;
          if (entry && entry->emitted_by_freetile) used_freetile = true;
          if (was_hit) cache_hits += 1u;
          if (num_subs >= 2u) {
            ctx->prof.horiz_fusions += 1u;
            ctx->prof.horiz_jobs_saved += (num_subs - 1u);
          }

          const size_t in_bytes = tile_use_w8a16
              ? (a_f32_tile.size() * sizeof(float))
              : a_q_tile.size();
          const void* in_src = tile_use_w8a16
              ? static_cast<const void*>(a_f32_tile.data())
              : static_cast<const void*>(a_q_tile.data());
          size_t in_alloc = std::max<size_t>(
              in_bytes,
              !entry->parsed.input_sizes.empty()
                  ? static_cast<size_t>(entry->parsed.input_sizes[0]) : 1u);
          size_t out_alloc = std::max<size_t>(
              tile_use_w8a16 ? (static_cast<size_t>(hw_n) * sizeof(float))
                             : (static_cast<size_t>(hw_m) * ob::align_up_u32(hw_n, 4u)),
              !entry->parsed.output_sizes.empty()
                  ? static_cast<size_t>(entry->parsed.output_sizes[0]) : 1u);

          ob_buffer* out_buf = nullptr;
          ob_status_t bst = ob::ensure_pipeline_buf_tiered(
              ctx, &ctx->gemm_out_bufs_4k[slot], &ctx->gemm_out_bufs[slot], out_alloc, &out_buf);
          uint64_t in_sync_ns = 0u;
          if (bst == OB_OK) {
            // Share ONE synced input DMA-BUF across all sibling tiles sharing (k0, tk, hw_k)!
            if (!active_in_buf || active_in_buf->slab.alloc_size < in_alloc) {
              while (in_slots[active_in_slot].refcnt > 0u) {
                reap_ready();
                if (in_slots[active_in_slot].refcnt == 0u) break;
                if (reap_seq < tpu_seq) {
                  wait_and_reap_one_dq();
                } else if (tpu_seq < sub_seq) {
                  bst = harvest_one_blocking();
                  if (bst != OB_OK) break;
                } else {
                  break;
                }
              }
              ob_buffer* in_buf = nullptr;
              if (bst == OB_OK) {
                bst = ob::ensure_pipeline_buf_tiered(
                    ctx, &ctx->gemm_in_bufs_4k[active_in_slot],
                    &ctx->gemm_in_bufs[active_in_slot], in_alloc, &in_buf);
              }
              if (bst == OB_OK) {
                const uint64_t td0 = prof_on ? ob::monotonic_now_ns() : 0u;
                bst = ob_buffer_write(in_buf, in_src, in_bytes, 0u);
                if (prof_on) {
                  in_sync_ns = ob::monotonic_now_ns() - td0;
                  ctx->prof.dma_sync_ns += in_sync_ns;
                }
              }
              if (bst == OB_OK) {
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
          if (prof_on) {
            const uint64_t submit_total = ob::monotonic_now_ns() - ts0;
            ctx->prof.submit_ns += (submit_total > in_sync_ns) ? (submit_total - in_sync_ns) : 0u;
          }
          if (bst != OB_OK) {
            ob::release_tile_entry(ctx, entry);
            pinned_w.reset();
            drain_all_cleanup();
            return bst;
          }

          PendingF32Tile& p = ring[slot];
          p.handle = exec_h;
          p.w_tile = pinned_w.release();
          p.in_buf = active_in_buf;
          if (p.in_buf) ob::buffer_retain(p.in_buf);
          p.out_buf = out_buf;
          if (p.out_buf) ob::buffer_retain(p.out_buf);
          p.slot = slot;
          p.in_slot = active_in_slot;
          in_slots[active_in_slot].refcnt += 1u;
          p.m0 = m0;
          p.tm = tm;
          p.hw_m = hw_m;
          p.hw_k = hw_k;
          p.hw_n = hw_n;
          p.tn_total = tn_total;
          p.num_subs = num_subs;
          p.is_first_k = (k0 == 0u);
          p.is_w8a16 = tile_use_w8a16;
          p.w8a16_prescale_inv = tile_use_w8a16 ? cached_a_w8a16_prescale_inv : 1.0f;
          for (uint32_t si = 0u; si < num_subs; ++si) p.subs[si] = grp.subs[si];
          p.scale_d_u32 = scale_d;
          if (!tile_use_w8a16) {
            if (tm <= p.s_a_inline.size()) {
              for (uint32_t mi = 0; mi < tm; ++mi) p.s_a_inline[mi] = row_s_a[mi];
            } else {
              p.s_a_overflow = row_s_a;
            }
          }
          if (!w_tile) p.s_b = row_s_b;
          tiles_dispatched += 1u;
          ++sub_seq;

          if (!ctx->mock_mode) {
            tile_handled_by_hw = true;
            ob_status_t od_st = try_harvest_ready();
            if (od_st != OB_OK) {
              drain_all_cleanup();
              return od_st;
            }
          }
        } else {
          pinned_w.reset();
          if (gst != OB_ERR_UNSUPPORTED) {
            drain_all_cleanup();
            return gst;
          }
        }
      }

      if (!tile_handled_by_hw) {
        if (skip_zero_init) {
          ob_status_t dst = drain_all_normal();
          if (dst != OB_OK) {
            drain_all_cleanup();
            return dst;
          }
          if (k0 == 0u) {
            for (uint32_t si = 0u; si < num_subs; ++si) {
              const ob::FusedSubSlice& sub = grp.subs[si];
              const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
              for (uint32_t mi = 0u; mi < tm; ++mi) {
                std::memset(job.c_mn + static_cast<size_t>(m0 + mi) * job.n_dim + sub.n0, 0,
                            static_cast<size_t>(sub.tn) * sizeof(float));
              }
            }
          }
        }
        used_cpu_fallback = true;
        for (uint32_t si = 0u; si < num_subs; ++si) {
          const ob::FusedSubSlice& sub = grp.subs[si];
          const ob_gemm_f32_quant_job_t& job = jobs[sub.job_idx];
          ob::quant::cpu_matmul_f32_tile(
              static_cast<ob_quant_type_t>(job.is_q4_0),
              a_mk, job.b_nk, job.c_mn, k_dim, job.n_dim,
              m0, tm, k0, tk, sub.n0, sub.tn);
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

  {
    ob_status_t dst = drain_all_normal();
    if (dst != OB_OK) {
      drain_all_cleanup();
      return dst;
    }
  }

  const uint64_t tact0 = prof_on ? ob::monotonic_now_ns() : 0u;
  for (uint32_t j = 0; j < num_jobs; ++j) {
    if (jobs[j].activation != OB_ACT_NONE) {
      ob::apply_activation_inplace_parallel(
          jobs[j].c_mn, static_cast<size_t>(m_dim) * jobs[j].n_dim,
          jobs[j].activation, &ctx->worker_pool);
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
    if (used_freetile) out_stats->flags |= OB_STAT_FLAG_FREETILE;
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

ob_status_t ob_gemm_prepack_quant_batch(ob_context_t*                  ctx,
                                        uint32_t                       m_hint,
                                        uint32_t                       k_dim,
                                        const ob_gemm_f32_quant_job_t* jobs,
                                        uint32_t                       num_jobs,
                                        ob_prepack_stats_t*            out_stats) {
  if (!ctx || !jobs || m_hint == 0u || k_dim == 0u || num_jobs == 0u || (k_dim % OB_QK8_0) != 0u) {
    return OB_ERR_INVALID_ARG;
  }
  for (uint32_t j = 0u; j < num_jobs; ++j) {
    if (!jobs[j].b_nk || jobs[j].n_dim == 0u || !ob::quant::is_valid_type(jobs[j].is_q4_0)) {
      return OB_ERR_INVALID_ARG;
    }
    const ob_quant_type_t qt = static_cast<ob_quant_type_t>(jobs[j].is_q4_0);
    const uint32_t qk = ob_quant_block_elements(qt);
    if (qk == 0u || (k_dim % qk) != 0u) {
      return OB_ERR_INVALID_ARG;
    }
  }

  ob_prepack_stats_t local_stats{};
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);

  const uint32_t max_m = ctx->tile_max_m;
  const uint32_t max_k = std::max<uint32_t>(OB_QK8_0, (ctx->tile_max_k / OB_QK8_0) * OB_QK8_0);
  const uint32_t max_n = ctx->tile_max_n;
  const ob_gemm_emitter_fn emitter = ctx->gemm_emitter;
  void* const emitter_ud = ctx->gemm_emitter_user_data;
  const bool builtin_active = ctx->builtin_emitter_active;
  const bool strict_hash = ctx->strict_weight_hash;
  const bool lt_enabled = ctx->large_tiles_enabled;
  const bool blt_enabled = ctx->batched_large_tiles_enabled;
  const bool horiz_fusion = ctx->horiz_fusion_enabled;
  const uint32_t w8a16_mode = ctx->w8a16_mode;

  if (!builtin_active || emitter == nullptr) {
    local_stats.bytes_resident = static_cast<uint64_t>(ctx->weight_cache_bytes);
    if (out_stats) *out_stats = local_stats;
    return OB_OK;
  }

  ob::ensure_weight_cache_budget(ctx, true);

  ob::WorkerPool local_pool;
  ob::WorkerPool* pack_pool = &ctx->worker_pool;
  if (ctx->num_threads <= 1u) {
    local_pool.init(4u);
    pack_pool = &local_pool;
  }

  auto check_budget_no_evict = [&](size_t incoming_bytes, bool* out_cacheable) -> ob_status_t {
    *out_cacheable =
        (ctx->weight_cache_bytes + incoming_bytes <= ctx->weight_cache_max_bytes) &&
        (ctx->weight_cache_map.size() < ctx->weight_cache_max_entries);
    return OB_OK;
  };

  std::vector<int8_t>& b_q_tile = ctx->gemm_bq_scratch;
  std::vector<ob::ScheduledGroup> groups;

  const bool allow_lt_64x = ob::compute_allow_lt_64x(
      k_dim, jobs, num_jobs, max_k, max_n, builtin_active, lt_enabled, blt_enabled,
      horiz_fusion, &groups);
  std::vector<ob::MSlice> m_slices = ob::build_m_slices(
      m_hint, max_m, builtin_active, true, allow_lt_64x, blt_enabled);

  std::vector<uint32_t> seen_hw_m;
  seen_hw_m.reserve(m_slices.size());

  for (const ob::MSlice& ms : m_slices) {
    const uint32_t hw_m = ms.hw_m;
    if (std::find(seen_hw_m.begin(), seen_hw_m.end(), hw_m) != seen_hw_m.end()) {
      continue;
    }
    seen_hw_m.push_back(hw_m);

    ob::build_scheduled_groups(hw_m, k_dim, jobs, num_jobs, max_k, max_n,
                               builtin_active, lt_enabled, blt_enabled,
                               horiz_fusion, &groups);

    auto prepack_group = [&](const ob::ScheduledGroup& grp, auto& self_ref) -> ob_status_t {
      const uint32_t hw_k = grp.hw_k;
      const uint32_t hw_n = grp.hw_n;
      const uint32_t num_subs = grp.num_subs;
      const bool tile_use_w8a16 =
          !ctx->mock_mode && w8a16_mode == 1u &&
          hw_m == 1u && ob::is_w8a16_pt_1x_shape(1u, hw_k, hw_n);
      const bool is_lt64_tile = (!tile_use_w8a16) && ob::is_lt_64x_shape(hw_m, hw_k, hw_n);
      const uint32_t pack_fmt = tile_use_w8a16 ? 2u : (is_lt64_tile ? 1u : 0u);

      bool grp_has_q4_k = false;
      for (uint32_t si = 0u; si < num_subs; ++si) {
        if (jobs[grp.subs[si].job_idx].is_q4_0 == static_cast<uint8_t>(OB_QUANT_Q4_K)) {
          grp_has_q4_k = true;
          break;
        }
      }

      const ob::WeightTileKey wh = ob::compute_quant_group_weight_key(
          ctx, k_dim, jobs, grp, pack_fmt, strict_hash);
      local_stats.tiles_checked += 1u;

      auto it = ctx->weight_cache_map.find(wh);
      ob::CachedWeightTile* w_tile = (it != ctx->weight_cache_map.end()) ? it->second : nullptr;
      if (w_tile && w_tile->rejected_fusion) {
        std::vector<ob::ScheduledGroup> split_groups;
        ob::split_scheduled_group_subs(grp, hw_m, max_k, max_n, builtin_active, &split_groups);
        for (const ob::ScheduledGroup& sg : split_groups) {
          ob_status_t sst = self_ref(sg, self_ref);
          if (sst != OB_OK) return sst;
        }
        return OB_OK;
      }

      uint32_t scale_d = 1u;
      if (w_tile) {
        local_stats.tiles_already_cached += 1u;
        scale_d = w_tile->scale_d;
      } else {
        bool rejected_fusion = false;
        bool skipped_budget = false;
        ob_status_t pst = ob::pack_quant_group_tile_locked(
            ctx, k_dim, jobs, grp, tile_use_w8a16, is_lt64_tile, grp_has_q4_k, wh,
            true, pack_pool, &b_q_tile, check_budget_no_evict,
            &rejected_fusion, &skipped_budget, &scale_d, &w_tile);
        if (pst != OB_OK) return pst;
        if (rejected_fusion) {
          std::vector<ob::ScheduledGroup> split_groups;
          ob::split_scheduled_group_subs(grp, hw_m, max_k, max_n, builtin_active, &split_groups);
          for (const ob::ScheduledGroup& sg : split_groups) {
            ob_status_t sst = self_ref(sg, self_ref);
            if (sst != OB_OK) return sst;
          }
          return OB_OK;
        }
        if (skipped_budget || !w_tile) {
          local_stats.tiles_skipped_budget += 1u;
          return OB_OK;
        }
        local_stats.tiles_prepacked += 1u;
      }

      if (!tile_use_w8a16) {
        ob_gemm_desc_t desc {};
        desc.m_dim = hw_m;
        desc.k_dim = hw_k;
        desc.n_dim = hw_n;
        desc.s_in = 1.0f;
        desc.s_w = 1.0f;
        desc.s_out = static_cast<float>(scale_d);
        desc.activation = OB_ACT_NONE;
        desc.cache_key = ob::hash_shape_key(
            hw_m, hw_k, hw_n, 0, 0, scale_d, 0u, ctx->freetile_enabled);
        ob::CachedGraph* entry = nullptr;
        bool was_hit = false;
        if (ob::acquire_tile_entry(ctx, emitter, emitter_ud, desc, &entry, &was_hit) == OB_OK) {
          if (!was_hit) local_stats.graphs_precompiled += 1u;
          ob::release_tile_entry(ctx, entry);
        }
      }
      return OB_OK;
    };

    for (const ob::ScheduledGroup& grp : groups) {
      ob_status_t pst = prepack_group(grp, prepack_group);
      if (pst != OB_OK) {
        local_stats.bytes_resident = static_cast<uint64_t>(ctx->weight_cache_bytes);
        if (out_stats) *out_stats = local_stats;
        return pst;
      }
    }
  }

  local_stats.bytes_resident = static_cast<uint64_t>(ctx->weight_cache_bytes);
  if (out_stats) *out_stats = local_stats;
  return OB_OK;
}

ob_status_t ob_gemm_swiglu_int8(ob_context_t* ctx,
                                const int8_t* x_mk,
                                const ob_swiglu_int8_desc_t* desc,
                                int8_t* y_mn,
                                ob_invoke_stats_t* out_stats) {
  if (!ctx || !x_mk || !desc || !y_mn ||
      !desc->wg_hk || !desc->wu_hk || !desc->wd_nh) {
    return OB_ERR_INVALID_ARG;
  }
  if (!ob::compute_swiglu_hw_multipliers(*desc, nullptr, nullptr, nullptr, nullptr)) {
    ob::set_last_error(ctx, "All SwiGLU quantization scales, derived multipliers, and spline coefficients must be finite and strictly positive");
    return OB_ERR_INVALID_ARG;
  }
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

  // Pack ext[2] for the 3 weight matrices directly into DMA-BUF via parallel NEON packer
  const size_t ext2_sz = ob::pack_ext2_size_for_swiglu(M, K, H, N);
  ob_buffer* wbuf = nullptr;
  ob_status_t st = ob_buffer_alloc(
      ctx, ext2_sz, OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL, &wbuf);
  if (st != OB_OK) return st;
  const uint64_t tdw0 = prof_on ? ob::monotonic_now_ns() : 0u;
  st = ob::buffer_sync_begin_write(wbuf);
  if (st == OB_OK) {
    const uint64_t tp_pack0 = prof_on ? ob::monotonic_now_ns() : 0u;
    ob::pack_ext2_for_swiglu_into(
        static_cast<uint8_t*>(ob_buffer_data(wbuf)),
        M, K, H, N,
        desc->wg_hk, desc->bg_h,
        desc->wu_hk, desc->bu_h,
        desc->wd_nh, desc->bd_n,
        &ctx->worker_pool);
    const uint64_t tp_pack_ns = prof_on ? (ob::monotonic_now_ns() - tp_pack0) : 0u;
    st = ob::buffer_sync_end_write(wbuf);
    if (prof_on) {
      const uint64_t total_dw_ns = ob::monotonic_now_ns() - tdw0;
      ctx->prof.weight_pack_ns += tp_pack_ns;
      ctx->prof.dma_sync_ns += (total_dw_ns > tp_pack_ns) ? (total_dw_ns - tp_pack_ns) : 0u;
    }
  } else if (prof_on) {
    ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - tdw0);
  }
  if (st != OB_OK) {
    ob_buffer_free(wbuf);
    return st;
  }

  // Unique graph cache key binding shape + all 9 exact float scales (weight-agnostic Probe-A!)
  uint64_t gkey = ctx->freetile_enabled ? 0x46545F5357493831ULL : 0x535749385F303031ULL;
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

  const uint64_t ts0 = prof_on ? ob::monotonic_now_ns() : 0u;
  ob::CachedGraph* entry = nullptr;
  bool was_hit = false;
  st = ob::acquire_swiglu_entry(ctx, graph_desc, gkey, &entry, &was_hit);
  if (st != OB_OK) {
    if (prof_on) ctx->prof.submit_ns += (ob::monotonic_now_ns() - ts0);
    ob_buffer_free(wbuf);
    return st;
  }
  const bool entry_freetile = (entry && entry->emitted_by_freetile);

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
  uint64_t in_sync_ns = 0u;
  if (st == OB_OK) {
    const uint64_t tdi0 = prof_on ? ob::monotonic_now_ns() : 0u;
    st = ob_buffer_write(in_buf, x_mk, in_bytes, 0u);
    if (prof_on) {
      in_sync_ns = ob::monotonic_now_ns() - tdi0;
      ctx->prof.dma_sync_ns += in_sync_ns;
    }
  }

  ob_exec_handle_t exec_h = 0u;
  if (st == OB_OK) {
    st = ob::device_submit_execute_async(ctx, entry, &in_buf, 1u, &out_buf, 1u, &exec_h, wbuf);
  }
  if (prof_on) {
    const uint64_t submit_total = ob::monotonic_now_ns() - ts0;
    ctx->prof.submit_ns += (submit_total > in_sync_ns) ? (submit_total - in_sync_ns) : 0u;
  }
  if (st != OB_OK) {
    ob::release_tile_entry(ctx, entry);
    ob_buffer_free(wbuf);
    return st;
  }

  ob_invoke_stats_t istats {};
  const uint64_t tw0 = prof_on ? ob::monotonic_now_ns() : 0u;
  st = ob::device_wait_handle(ctx, exec_h, ctx->cfg.timeout_ms, &istats);
  if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - tw0);
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
    const uint64_t tdo0 = prof_on ? ob::monotonic_now_ns() : 0u;
    st = ob_buffer_read(out_buf, y_mn, out_bytes, 0u);
    if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - tdo0);
    if (st != OB_OK) return st;
  }

  const uint64_t wall_ns = ob::monotonic_now_ns() - t0_ns;
  ctx->prof.swiglu_calls += 1u;
  ctx->prof.swiglu_chunks_chained += 1u;
  ctx->prof.tiles_dispatched += 1u;
  ctx->prof.hw_us += istats.hw_execution_us;
  ctx->prof.total_wall_ns += wall_ns;

  if (out_stats) {
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->hw_execution_us = istats.hw_execution_us;
    out_stats->wall_time_us = wall_ns / 1000ULL;
    out_stats->tiles_dispatched = 1u;
    out_stats->cache_hits = was_hit ? 1u : 0u;
    if (ctx->mock_mode) out_stats->flags |= OB_STAT_FLAG_MOCK;
    out_stats->flags |= OB_STAT_FLAG_EMITTER_USED;
    if (entry_freetile) out_stats->flags |= OB_STAT_FLAG_FREETILE;
    if (was_hit) out_stats->flags |= OB_STAT_FLAG_CACHE_HIT;
  }
  return OB_OK;
}

namespace {

void cpu_swiglu_f32_quant_slice(const float* a_slice,
                                const ob_gemm_swiglu_quant_desc_t* d,
                                uint32_t tm,
                                float* c_slice,
                                ob::WorkerPool* pool) {
  const uint32_t K = d->k_dim, H = d->h_dim, N = d->n_dim;
  const ob_quant_type_t gate_qt = static_cast<ob_quant_type_t>(d->gate_is_q4_0);
  const ob_quant_type_t up_qt   = static_cast<ob_quant_type_t>(d->up_is_q4_0);
  const ob_quant_type_t down_qt = static_cast<ob_quant_type_t>(d->down_is_q4_0);
  const size_t g_row_bytes = static_cast<size_t>(K / ob_quant_block_elements(gate_qt)) * ob_quant_block_bytes(gate_qt);
  const size_t u_row_bytes = static_cast<size_t>(K / ob_quant_block_elements(up_qt))   * ob_quant_block_bytes(up_qt);
  const size_t d_row_bytes = static_cast<size_t>(H / ob_quant_block_elements(down_qt)) * ob_quant_block_bytes(down_qt);
  const uint8_t* wg_bytes = static_cast<const uint8_t*>(d->w_gate_hk);
  const uint8_t* wu_bytes = static_cast<const uint8_t*>(d->w_up_hk);
  const uint8_t* wd_bytes = static_cast<const uint8_t*>(d->w_down_nh);
  std::vector<float> mid(H, 0.0f);
  const bool par_h = (pool && pool->num_threads > 1u && H >= 512u);
  const bool par_n = (pool && pool->num_threads > 1u && N >= 256u && H >= 512u);
  for (uint32_t mi = 0u; mi < tm; ++mi) {
    const float* a_row = a_slice + static_cast<size_t>(mi) * K;
    auto run_h = [&](uint32_t hs, uint32_t he) {
      for (uint32_t h = hs; h < he; ++h) {
        const float g = ob::quant::dot_row_f32(gate_qt, a_row, wg_bytes + static_cast<size_t>(h) * g_row_bytes, K);
        const float u = ob::quant::dot_row_f32(up_qt,   a_row, wu_bytes + static_cast<size_t>(h) * u_row_bytes, K);
        mid[h] = (g / (1.0f + std::exp(-g))) * u;
      }
    };
    if (par_h) {
      pool->parallel_for_fn(H, 64u, [&](uint32_t s, uint32_t e, uint32_t) { run_h(s, e); });
    } else {
      run_h(0u, H);
    }
    float* c_row = c_slice + static_cast<size_t>(mi) * N;
    auto run_n = [&](uint32_t ns, uint32_t ne) {
      for (uint32_t n = ns; n < ne; ++n) {
        c_row[n] = ob::quant::dot_row_f32(down_qt, mid.data(), wd_bytes + static_cast<size_t>(n) * d_row_bytes, H);
      }
    };
    if (par_n) {
      pool->parallel_for_fn(N, 32u, [&](uint32_t s, uint32_t e, uint32_t) { run_n(s, e); });
    } else {
      run_n(0u, N);
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
  constexpr uint32_t kMaxFallbackM = 64u;
  const uint32_t buf_m = std::min<uint32_t>(tm, kMaxFallbackM);
  const size_t buf_cnt = static_cast<size_t>(buf_m) * H;
  // Persistent bounded scratch avoids per-layer mmap/page-fault/munmap on Qwen2.5 pp512!
  std::lock_guard<std::mutex> flk(ctx->swiglu_fb_mu);
  if (ctx->swiglu_gate_scratch.size() < buf_cnt) ctx->swiglu_gate_scratch.resize(buf_cnt);
  if (ctx->swiglu_up_scratch.size() < buf_cnt) ctx->swiglu_up_scratch.resize(buf_cnt);
  float* gate_ptr = ctx->swiglu_gate_scratch.data();
  float* up_ptr = ctx->swiglu_up_scratch.data();

  for (uint32_t m0 = 0u; m0 < tm; m0 += kMaxFallbackM) {
    const uint32_t sm = std::min<uint32_t>(kMaxFallbackM, tm - m0);
    const float* a_sub = a_slice + static_cast<size_t>(m0) * K;
    float* c_sub = c_slice + static_cast<size_t>(m0) * N;

    ob_gemm_f32_quant_job_t gu_jobs[2]{};
    gu_jobs[0].b_nk = d->w_gate_hk;
    gu_jobs[0].c_mn = gate_ptr;
    gu_jobs[0].n_dim = H;
    gu_jobs[0].activation = OB_ACT_NONE;
    gu_jobs[0].is_q4_0 = d->gate_is_q4_0;
    gu_jobs[1].b_nk = d->w_up_hk;
    gu_jobs[1].c_mn = up_ptr;
    gu_jobs[1].n_dim = H;
    gu_jobs[1].activation = OB_ACT_NONE;
    gu_jobs[1].is_q4_0 = d->up_is_q4_0;

    ob_invoke_stats_t s1{};
    ob_status_t st = ob_gemm_matmul_f32_quant_batch(ctx, a_sub, sm, K, gu_jobs, 2u, &s1);
    if (st != OB_OK) return st;

    // Single fused parallel NEON pass: gate[i] = silu(gate[i]) * up[i]
    const size_t cnt = static_cast<size_t>(sm) * H;
    const uint64_t tsm0 = ctx->profile_enabled ? ob::monotonic_now_ns() : 0u;
    ob::silu_mul_inplace_parallel(gate_ptr, up_ptr, cnt, &ctx->worker_pool);
    if (ctx->profile_enabled) {
      std::lock_guard<std::mutex> lk(ctx->gemm_mu);
      ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tsm0);
    }

    ob_gemm_f32_quant_job_t d_job{};
    d_job.b_nk = d->w_down_nh;
    d_job.c_mn = c_sub;
    d_job.n_dim = N;
    d_job.activation = OB_ACT_NONE;
    d_job.is_q4_0 = d->down_is_q4_0;

    ob_invoke_stats_t s2{};
    st = ob_gemm_matmul_f32_quant_batch(ctx, gate_ptr, sm, H, &d_job, 1u, &s2);
    if (st != OB_OK) return st;

    if (out_stats) {
      out_stats->hw_execution_us += s1.hw_execution_us + s2.hw_execution_us;
      out_stats->tiles_dispatched += s1.tiles_dispatched + s2.tiles_dispatched;
      out_stats->cache_hits += s1.cache_hits + s2.cache_hits;
      out_stats->flags |= (s1.flags | s2.flags);
    }
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
      !ob::quant::is_valid_type(desc->gate_is_q4_0) ||
      !ob::quant::is_valid_type(desc->up_is_q4_0) ||
      !ob::quant::is_valid_type(desc->down_is_q4_0)) {
    return OB_ERR_INVALID_ARG;
  }
  const ob_quant_type_t gate_qt = static_cast<ob_quant_type_t>(desc->gate_is_q4_0);
  const ob_quant_type_t up_qt   = static_cast<ob_quant_type_t>(desc->up_is_q4_0);
  const ob_quant_type_t down_qt = static_cast<ob_quant_type_t>(desc->down_is_q4_0);
  if ((desc->k_dim % ob_quant_block_elements(gate_qt)) != 0u ||
      (desc->k_dim % ob_quant_block_elements(up_qt))   != 0u ||
      (desc->h_dim % ob_quant_block_elements(down_qt)) != 0u) {
    return OB_ERR_INVALID_ARG;
  }

  const uint64_t t0_ns = ob::monotonic_now_ns();
  const uint32_t M = desc->m_dim;
  const uint32_t K = desc->k_dim;
  const uint32_t H = desc->h_dim;
  const uint32_t N = desc->n_dim;

  bool can_chain_shape = false;
  bool fast_chain_on = false;
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
    fast_chain_on = ctx->swiglu_fast_chain_enabled;
    prefill_chain_on = ctx->swiglu_prefill_enabled && ctx->batched_large_tiles_enabled;
    param_swap_slots = ctx->param_swap_slots;
    ring_depth = std::max<uint32_t>(1u, std::min<uint32_t>(ctx->credit_size, OB_MAX_CREDITS));
    can_chain_shape = ctx->swiglu_chain_enabled && builtin_active &&
                      ((K == 576u && N == 576u) || (K == 1536u && N == 1536u));
  }

  // Default: chain M==1 decode; route tiny prompts (2<=M<=8) to exact NEON FP32 CPU (preserves
  // exact BOS/attention-sink KV cache precision with 0 B weight-cache allocation!); keep M>8
  // prefill on faster lt_64x* GEMM unless opt-in OPENBUENOS_SWIGLU_PREFILL=1 is enabled.
  if (!can_chain_shape || (M > 8u && !prefill_chain_on)) {
    ob_invoke_stats_t fb{};
    ob_status_t st = run_swiglu_m7a_fallback(ctx, a_mk, desc, M, c_mn, &fb);
    if (st == OB_OK) {
      std::lock_guard<std::mutex> lk(ctx->gemm_mu);
      ctx->prof.swiglu_calls += 1u;
      if ((fb.flags & OB_STAT_FLAG_EMITTER_USED) != 0u) {
        ctx->prof.swiglu_fallbacks += 1u;
      }
    }
    if (out_stats) {
      *out_stats = fb;
      out_stats->wall_time_us = (ob::monotonic_now_ns() - t0_ns) / 1000ULL;
    }
    return st;
  }

  if (!prefill_chain_on && M >= 2u && M <= 8u) {
    cpu_swiglu_f32_quant_slice(a_mk, desc, M, c_mn, &ctx->worker_pool);
    const uint64_t wall_ns = ob::monotonic_now_ns() - t0_ns;
    {
      std::lock_guard<std::mutex> lk(ctx->gemm_mu);
      ctx->prof.swiglu_calls += 1u;
      ctx->prof.total_wall_ns += wall_ns;
    }
    if (out_stats) {
      std::memset(out_stats, 0, sizeof(*out_stats));
      out_stats->wall_time_us = wall_ns / 1000ULL;
      out_stats->flags = OB_STAT_FLAG_CPU_FALLBACK;
      if (ctx->mock_mode) out_stats->flags |= OB_STAT_FLAG_MOCK;
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
    } else if (rem >= 16u && prefill_chain_on) {
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
  uint16_t agg_flags = ctx->mock_mode ? static_cast<uint16_t>(OB_STAT_FLAG_MOCK) : static_cast<uint16_t>(0u);

  const uint32_t d_qk = ob_quant_block_elements(down_qt);
  const size_t d_blk_sz = ob_quant_block_bytes(down_qt);
  const size_t g_row_bytes = static_cast<size_t>(K / ob_quant_block_elements(gate_qt)) *
                             ob_quant_block_bytes(gate_qt);
  const size_t u_row_bytes = static_cast<size_t>(K / ob_quant_block_elements(up_qt)) *
                             ob_quant_block_bytes(up_qt);
  const size_t d_row_bytes = static_cast<size_t>(H / d_qk) * d_blk_sz;

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
      if (!need_fallback && batch_amax > 0.0f) {
        const float s_a_pre = ob::quantize_scale_pow2_2nd_up(batch_amax / 127.0f);
        const float rms_aq_pre = static_cast<float>(max_row_rms / static_cast<double>(s_a_pre));
        if (rms_aq_pre > 56.0f) {
          need_fallback = true;
        }
      }
    }

    if (need_fallback) {
      if (transient_m1_outlier) {
        // Rare M=1 attention-sink outlier token: run exact NEON FP32 on CPU so we do NOT
        // allocate 41 MiB of duplicate M7A ext[2] tiles into weight_cache (prevents Qwen2.5
        // LRU thrashing while delivering 0-error FP32 precision on sink tokens!).
        cpu_swiglu_f32_quant_slice(a_slice, desc, 1u, c_slice, &ctx->worker_pool);
        agg_flags |= OB_STAT_FLAG_CPU_FALLBACK;
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
      if ((fb.flags & OB_STAT_FLAG_EMITTER_USED) != 0u) {
        fb_slices += 1u;
      }
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
      a_q.resize(static_cast<size_t>(ms.hw_m) * K);
      if (ms.tm < ms.hw_m) {
        std::memset(a_q.data() + static_cast<size_t>(ms.tm) * K, 0,
                    static_cast<size_t>(ms.hw_m - ms.tm) * K);
      }
      if (ctx->worker_pool.num_threads > 1u &&
          ms.tm >= 8u && static_cast<size_t>(ms.tm) * K >= 16384u) {
        ctx->worker_pool.parallel_for_fn(ms.tm, 4u, [&](uint32_t ms_lo, uint32_t ms_hi, uint32_t) {
          for (uint32_t mi = ms_lo; mi < ms_hi; ++mi) {
            ob::quant_sym_i8_row(a_slice + static_cast<size_t>(mi) * K,
                                 a_q.data() + static_cast<size_t>(mi) * K,
                                 K, inv_sa);
          }
        });
      } else {
        for (uint32_t mi = 0u; mi < ms.tm; ++mi) {
          ob::quant_sym_i8_row(a_slice + static_cast<size_t>(mi) * K,
                               a_q.data() + static_cast<size_t>(mi) * K,
                               K, inv_sa);
        }
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

      struct PendingSwigluChunk {
        ob_exec_handle_t handle = 0u;
        ob::CachedWeightTile* w_tile = nullptr;
        ob_buffer* in_buf = nullptr;
        ob_buffer* out_buf = nullptr;
        uint32_t slot = 0u;
        uint32_t hw_h = 0u;
        uint32_t d_gate = 0u, d_up = 0u, d_mul = 0u, d_down = 0u;
        float s_gate = 0.0f;
        bool is_first_chunk = false;
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

      auto drain_one_harvested = [&](ob_status_t wst, const ob_invoke_stats_t& istats) -> ob_status_t {
        PendingSwigluChunk& p = ring[ring_head];
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
                rst = ob::buffer_sync_end_read(p.out_buf);
                if (prof_on) ctx->prof.dma_sync_ns += (ob::monotonic_now_ns() - tde0);
                read_active = false;
                if (rst != OB_OK) break;

                ctx->prof.sat_retries += 1u;
                eff_d_down <<= 1u;
                const uint64_t rkey = ob::hash_swiglu_graph_key(
                    ms.hw_m, K, p.hw_h, N, p.d_gate, p.d_up, p.d_mul, eff_d_down,
                    p.s_gate, p.slot % param_swap_slots, ctx->freetile_enabled);
                ob_swiglu_int8_desc_t rdesc =
                    make_graph_desc(p.hw_h, p.d_gate, p.d_up, p.d_mul, eff_d_down, p.s_gate);
                const uint64_t trs0 = prof_on ? ob::monotonic_now_ns() : 0u;
                ob::CachedGraph* rentry = nullptr;
                bool rhit = false;
                rst = ob::acquire_swiglu_entry(ctx, rdesc, rkey, &rentry, &rhit);
                if (rst != OB_OK) break;
                if (rentry && rentry->emitted_by_freetile) agg_flags |= OB_STAT_FLAG_FREETILE;
                if (rhit) total_hits += 1u;
                ob_exec_handle_t rh = 0u;
                rst = ob::device_submit_execute_async(
                    ctx, rentry, &p.in_buf, 1u, &p.out_buf, 1u, &rh, p.w_tile->ext2_buf);
                if (prof_on) ctx->prof.submit_ns += (ob::monotonic_now_ns() - trs0);
                if (rst != OB_OK) {
                  ob::release_tile_entry(ctx, rentry);
                  break;
                }
                total_tiles += 1u;
                chained_chunks += 1u;
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
                const uint64_t tsatf0 = prof_on ? ob::monotonic_now_ns() : 0u;
                const bool still_sat = ob::tile_output_is_saturated_impl(raw_out, ms.tm, N, N);
                if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tsatf0);
                if (still_sat) {
                  slice_rejected = true;
                } else {
                  const uint64_t tq1 = prof_on ? ob::monotonic_now_ns() : 0u;
                  // Y_f32[m, n] (+)= q_out[m, n] * (s_silu * s_a * d_up * d_mul * eff_d_down) * row_s_d[n]
                  const float out_mult = p.s_gate * s_a * static_cast<float>(p.d_up) *
                                         static_cast<float>(p.d_mul) * static_cast<float>(eff_d_down);
                  const float* sd_ptr = p.w_tile->row_s_b.data();
                  const bool store_first = p.is_first_chunk;
                  if (ctx->worker_pool.num_threads > 1u &&
                      ms.tm >= 8u && static_cast<size_t>(ms.tm) * N >= 16384u) {
                    ctx->worker_pool.parallel_for_fn(ms.tm, 4u, [&](uint32_t ms_lo, uint32_t ms_hi, uint32_t) {
                      for (uint32_t mi = ms_lo; mi < ms_hi; ++mi) {
                        float* c_row = c_slice + static_cast<size_t>(mi) * N;
                        const int8_t* o_row = raw_out + static_cast<size_t>(mi) * N;
                        ob::dequant_outliers_row_f32_impl(
                            c_row, o_row, sd_ptr, N, out_mult, store_first, nullptr, nullptr, 0u);
                      }
                    });
                  } else {
                    for (uint32_t mi = 0u; mi < ms.tm; ++mi) {
                      float* c_row = c_slice + static_cast<size_t>(mi) * N;
                      const int8_t* o_row = raw_out + static_cast<size_t>(mi) * N;
                      ob::dequant_outliers_row_f32_impl(
                          c_row, o_row, sd_ptr, N, out_mult, store_first, nullptr, nullptr, 0u);
                    }
                  }
                  if (prof_on) ctx->prof.dequant_ns += (ob::monotonic_now_ns() - tq1);
                }
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

      auto drain_one = [&]() -> ob_status_t {
        PendingSwigluChunk& p = ring[ring_head];
        ob_invoke_stats_t istats{};
        const uint64_t tw0 = prof_on ? ob::monotonic_now_ns() : 0u;
        ob_status_t wst = ob::device_wait_handle(ctx, p.handle, ctx->cfg.timeout_ms, &istats);
        if (prof_on) ctx->prof.wait_ns += (ob::monotonic_now_ns() - tw0);
        return drain_one_harvested(wst, istats);
      };

      auto try_drain_ready = [&]() -> ob_status_t {
        while (ring_count >= 2u) {
          ob_status_t wst = OB_OK;
          ob_invoke_stats_t istats{};
          if (!ob::device_try_wait_handle(ctx, ring[ring_head].handle, &wst, &istats)) {
            break;
          }
          ob_status_t dst = drain_one_harvested(wst, istats);
          if (dst != OB_OK) return dst;
        }
        return OB_OK;
      };

      const float slice_rms_aq = static_cast<float>(max_row_rms / static_cast<double>(s_a));
      const double slice_mean_aq = max_row_abs_mean / static_cast<double>(s_a);
      const double slice_l2_aq = static_cast<double>(slice_rms_aq) * std::sqrt(static_cast<double>(K));
      auto eval_swiglu_eligibility = [&](float swg,
                                         float lg_max, float lg_mean,
                                         float lu_max, float lu_mean,
                                         float ld_max,
                                         float dcg_max, float dcu_max,
                                         float l1g_max, float linfg_max,
                                         float l1u_max, float linfu_max,
                                         uint32_t* out_dg, uint32_t* out_du,
                                         uint32_t* out_dm, uint32_t* out_dd,
                                         float* out_sg) -> bool {
        uint32_t dg = 0u, du = 0u, dm = 0u, dd = 0u;
        ob::compute_swiglu_stage_dividers(
            ms.tm, slice_rms_aq, lg_max, lg_mean, lu_max, lu_mean, ld_max,
            &dg, &du, &dm, &dd);
        const double dc_g_t = slice_mean_aq * static_cast<double>(dcg_max);
        const double dc_u_t = slice_mean_aq * static_cast<double>(dcu_max);
        const float sg_cand = ob::snap_scale_pow2_2nd_nearest(
            s_a * swg * static_cast<float>(dg));
        if (slice_rms_aq > 56.0f ||
            dc_g_t > 64.0 * static_cast<double>(dg) ||
            dc_u_t > 64.0 * static_cast<double>(du) ||
            !(sg_cand >= 5.0e-4f)) {
          return false;
        }
        const double ub_g = std::min({
            slice_l2_aq * static_cast<double>(lg_max),
            static_cast<double>(max_l1_aq) * static_cast<double>(linfg_max),
            static_cast<double>(amax_aq) * static_cast<double>(l1g_max),
        });
        const double ub_u = std::min({
            slice_l2_aq * static_cast<double>(lu_max),
            static_cast<double>(max_l1_aq) * static_cast<double>(linfu_max),
            static_cast<double>(amax_aq) * static_cast<double>(l1u_max),
        });
        bool holder_proven = false;
        if (ub_g <= 127.0 * static_cast<double>(dg) &&
            ub_u <= 127.0 * static_cast<double>(du)) {
          const int64_t mqg = std::min<int64_t>(
              127LL, static_cast<int64_t>(std::ceil(ub_g / static_cast<double>(dg))));
          const int64_t mqu = std::min<int64_t>(
              127LL, static_cast<int64_t>(std::ceil(ub_u / static_cast<double>(du))));
          int64_t mqsilu = std::abs(ob::eval_swiglu_silu_q(static_cast<int32_t>(mqg), sg_cand));
          mqsilu = std::max<int64_t>(
              mqsilu, std::abs(ob::eval_swiglu_silu_q(static_cast<int32_t>(-mqg), sg_cand)));
          if (sg_cand > 0.0f && mqg > 0LL) {
            const int64_t qmc = std::max<int64_t>(
                -mqg, std::min<int64_t>(0LL, -static_cast<int64_t>(
                                                 std::llround(1.2784645f / sg_cand))));
            mqsilu = std::max<int64_t>(
                mqsilu, std::abs(ob::eval_swiglu_silu_q(static_cast<int32_t>(qmc), sg_cand)));
          }
          if (mqg > 0LL) {
            mqsilu = std::min<int64_t>(127LL, std::max<int64_t>(mqsilu, mqg + 1LL));
          }
          const int64_t mprod = mqsilu * mqu;
          uint32_t ndm = dm;
          while (ndm < 64u && mprod > 127LL * static_cast<int64_t>(ndm)) ndm <<= 1u;
          if (mprod <= 127LL * static_cast<int64_t>(ndm)) {
            dm = ndm;
            holder_proven = true;
          }
        }
        if (!holder_proven && !fast_chain_on) {
          return false;
        }
        *out_dg = dg;
        *out_du = du;
        *out_dm = dm;
        *out_dd = dd;
        *out_sg = sg_cand;
        return true;
      };

      for (const HChunk& hc : h_chunks) {
        const uint32_t h0 = hc.h0;
        const uint32_t th = hc.th;
        const uint32_t hw_h = hc.hw_h;
        const uint32_t h_blk0 = h0 / d_qk;
        const uint32_t h_blk_end = (h0 + th + d_qk - 1u) / d_qk;
        const uint32_t h_blk_cnt = h_blk_end - h_blk0;

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
        if ((h0 % d_qk) != 0u || (th % d_qk) != 0u) {
          uint32_t sub_h[2] = {h0 % d_qk, th};
          ob::hash128_update(&wh_st, sub_h, sizeof(sub_h));
        }
        const uint8_t* g_ptr = static_cast<const uint8_t*>(desc->w_gate_hk) +
                               static_cast<size_t>(h0) * g_row_bytes;
        const uint8_t* u_ptr = static_cast<const uint8_t*>(desc->w_up_hk) +
                               static_cast<size_t>(h0) * u_row_bytes;
        const uint8_t* d_ptr = static_cast<const uint8_t*>(desc->w_down_nh) +
                               static_cast<size_t>(h_blk0) * d_blk_sz;
        ob::hash_weight_tile_for_ctx(ctx, &wh_st, g_ptr, g_row_bytes, th, g_row_bytes, strict_hash);
        ob::hash_weight_tile_for_ctx(ctx, &wh_st, u_ptr, u_row_bytes, th, u_row_bytes, strict_hash);
        ob::hash_weight_tile_for_ctx(ctx, &wh_st, d_ptr, d_row_bytes, N,
                                     static_cast<size_t>(h_blk_cnt) * d_blk_sz, strict_hash);
        const ob::WeightTileKey wh = ob::finalize_weight_key(wh_st);
        ob::CachedWeightTile* w_tile = ob::weight_cache_find_locked(ctx, wh);
        if (prof_on) ctx->prof.hash_ns += (ob::monotonic_now_ns() - th0);

        if (w_tile && w_tile->rejected_swiglu) {
          drain_all_cleanup();
          slice_rejected = true;
          break;
        }

        uint32_t d_gate = 0u, d_up = 0u, d_mul = 0u, d_down = 0u;
        float s_gate_cand = 0.0f;
        bool eligibility_checked = false;
        if (w_tile && w_tile->swiglu_norms_only) {
          if (!eval_swiglu_eligibility(
                  w_tile->swiglu_s_wg,
                  w_tile->swiglu_l2_gate_max, w_tile->swiglu_l2_gate_mean,
                  w_tile->swiglu_l2_up_max, w_tile->swiglu_l2_up_mean,
                  w_tile->swiglu_l2_down_max,
                  w_tile->swiglu_dc_gate_max, w_tile->swiglu_dc_up_max,
                  w_tile->swiglu_l1_gate_max, w_tile->swiglu_linf_gate_max,
                  w_tile->swiglu_l1_up_max, w_tile->swiglu_linf_up_max,
                  &d_gate, &d_up, &d_mul, &d_down, &s_gate_cand)) {
            drain_all_cleanup();
            slice_rejected = true;
            break;
          }
          // Activation passed using cached norms; drop the 0-byte norms-only entry and pack ext[2].
          ctx->weight_cache_map.erase(w_tile->key);
          ctx->weight_cache_lru.erase(w_tile->lru_it);
          delete w_tile;
          w_tile = nullptr;
        }

        if (!w_tile) {
          const uint64_t tp0 = prof_on ? ob::monotonic_now_ns() : 0u;
          // Cold pack (parallel across ctx->worker_pool + direct-to-DMA-BUF NEON ext[2] pack!):
          //   a) W_up per-row -> wu_q[hw_h, K] + row_s_u[th]
          std::vector<int8_t> wu_q(static_cast<size_t>(hw_h) * K);
          if (th < hw_h) {
            std::memset(wu_q.data() + static_cast<size_t>(th) * K, 0,
                        static_cast<size_t>(hw_h - th) * K);
          }
          std::vector<float> row_s_u(th, 0.0f);
          ob::quant::requant_rows(up_qt, desc->w_up_hk, K, h0, th, 0u, K, K,
                                  wu_q.data(), row_s_u.data(), nullptr,
                                  &ctx->worker_pool, nullptr);

          //   b) W_down with row_s_u[h] absorbed -> wd_q[N, hw_h] + row_s_d[N]
          std::vector<int8_t> wd_q(static_cast<size_t>(N) * hw_h);
          std::vector<float> row_s_d(N, 0.0f);
          auto pack_wdown_range = [&](uint32_t ns, uint32_t ne) {
            float stack_wrow[2048];
            std::vector<float> heap_wrow;
            float* wrow_d = stack_wrow;
            if (th > 2048u) {
              heap_wrow.resize(th);
              wrow_d = heap_wrow.data();
            }
            for (uint32_t ni = ns; ni < ne; ++ni) {
              ob::quant::dequant_row_slice_f32(down_qt, desc->w_down_nh, H, ni, h0, th, wrow_d);
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
              int8_t* wd_row = wd_q.data() + static_cast<size_t>(ni) * hw_h;
              if (amax_d > 0.0f) {
                row_s_d[ni] = amax_d / 127.0f;
                ob::quant_sym_i8_row(wrow_d, wd_row, th, 127.0f / amax_d);
                if (th < hw_h) std::memset(wd_row + th, 0, hw_h - th);
              } else {
                std::memset(wd_row, 0, hw_h);
              }
            }
          };
          if (ctx->worker_pool.num_threads > 1u &&
              N >= 64u && static_cast<size_t>(N) * th >= 32768u) {
            ctx->worker_pool.parallel_for_fn(N, 16u, [&](uint32_t s, uint32_t e, uint32_t) {
              pack_wdown_range(s, e);
            });
          } else {
            pack_wdown_range(0u, N);
          }

          //   c) W_gate Pass 1: exact per-row max directly from quantized blocks
          std::vector<float> wg_row_max(th, 0.0f);
          auto scan_wgate_max = [&](uint32_t hs, uint32_t he) {
            for (uint32_t hi = hs; hi < he; ++hi) {
              wg_row_max[hi] = ob::quant::row_max_abs_f32(
                  gate_qt, desc->w_gate_hk, K, h0 + hi, 0u, K);
            }
          };
          if (ctx->worker_pool.num_threads > 1u &&
              th >= 64u && static_cast<size_t>(th) * K >= 32768u) {
            ctx->worker_pool.parallel_for_fn(th, 16u, [&](uint32_t s, uint32_t e, uint32_t) {
              scan_wgate_max(s, e);
            });
          } else {
            scan_wgate_max(0u, th);
          }
          float max_wg = 0.0f;
          double sum_row_max_wg = 0.0;
          for (uint32_t hi = 0u; hi < th; ++hi) {
            if (wg_row_max[hi] > max_wg) max_wg = wg_row_max[hi];
            sum_row_max_wg += static_cast<double>(wg_row_max[hi]);
          }
          const double mean_row_max_wg =
              sum_row_max_wg / static_cast<double>(std::max<uint32_t>(1u, th));
          if (max_wg > 0.0f && static_cast<double>(max_wg) > 16.0 * std::max(mean_row_max_wg, 1e-12)) {
            bool sentinel_ok = false;
            if (ob::weight_cache_prepare_space_locked(ctx, 0u, ring_count, drain_one, &sentinel_ok) == OB_OK &&
                sentinel_ok) {
              ob::CachedWeightTile* sentinel = new (std::nothrow) ob::CachedWeightTile();
              if (sentinel) {
                sentinel->key = wh;
                sentinel->ext2_bytes = 0u;
                sentinel->rejected_swiglu = true;
                if (!ob::weight_cache_insert_locked(ctx, sentinel)) {
                  delete sentinel;
                }
              }
            }
            if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
            drain_all_cleanup();
            slice_rejected = true;
            break;
          }

          //      W_gate Pass 2: parallel dequant + common-scale quant in L1 stack buffer!
          const float s_wg = (max_wg > 0.0f) ? ob::quantize_scale_pow2_2nd_up(max_wg / 127.0f) : 1.0f;
          const float inv_swg = (max_wg > 0.0f) ? (1.0f / s_wg) : 0.0f;
          std::vector<int8_t> wg_q(static_cast<size_t>(hw_h) * K);
          if (th < hw_h || max_wg == 0.0f) {
            const size_t z_off = (max_wg == 0.0f) ? 0u : static_cast<size_t>(th) * K;
            std::memset(wg_q.data() + z_off, 0, wg_q.size() - z_off);
          }
          if (max_wg > 0.0f) {
            auto pack_wgate_range = [&](uint32_t hs, uint32_t he) {
              float stack_wrow[2048];
              std::vector<float> heap_wrow;
              float* wrow_g = stack_wrow;
              if (K > 2048u) {
                heap_wrow.resize(K);
                wrow_g = heap_wrow.data();
              }
              for (uint32_t hi = hs; hi < he; ++hi) {
                ob::quant::dequant_row_slice_f32(gate_qt, desc->w_gate_hk, K, h0 + hi, 0u, K, wrow_g);
                int8_t* wg_row = wg_q.data() + static_cast<size_t>(hi) * K;
                ob::quant_sym_i8_row(wrow_g, wg_row, K, inv_swg);
              }
            };
            if (ctx->worker_pool.num_threads > 1u &&
                th >= 64u && static_cast<size_t>(th) * K >= 32768u) {
              ctx->worker_pool.parallel_for_fn(th, 16u, [&](uint32_t s, uint32_t e, uint32_t) {
                pack_wgate_range(s, e);
              });
            } else {
              pack_wgate_range(0u, th);
            }
          }

          //   d) Record per-stage L2, DC-sum, L1, and Linf weight norms once for runtime divider selection & exact saturation guards!
          float l2_g_max = 0.0f, l2_g_mean = 0.0f, dc_g_max = 0.0f, l1_g_max = 0.0f, linf_g_max = 0.0f;
          float l2_u_max = 0.0f, l2_u_mean = 0.0f, dc_u_max = 0.0f, l1_u_max = 0.0f, linf_u_max = 0.0f;
          float l2_d_max = 0.0f, l2_d_mean = 0.0f, dc_d_max = 0.0f;
          ob::compute_matrix_row_l2_stats(wg_q.data(), th, K, K, &l2_g_max, &l2_g_mean, &dc_g_max, &l1_g_max, &linf_g_max);
          ob::compute_matrix_row_l2_stats(wu_q.data(), th, K, K, &l2_u_max, &l2_u_mean, &dc_u_max, &l1_u_max, &linf_u_max);
          ob::compute_matrix_row_l2_stats(wd_q.data(), N, th, hw_h, &l2_d_max, &l2_d_mean, &dc_d_max);

          // Reject statistically non-zero-mean gate/up (or skewed Q4_K down) weight tiles so
          // correlated inputs route cleanly to M7A fallback with exact per-row algebraic centering.
          if ((l2_g_max > 0.0f && dc_g_max > 5.5f * l2_g_max) ||
              (l2_u_max > 0.0f && dc_u_max > 5.5f * l2_u_max) ||
              (down_qt == OB_QUANT_Q4_K && l2_d_max > 0.0f && dc_d_max > 5.5f * l2_d_max)) {
            bool sentinel_ok = false;
            if (ob::weight_cache_prepare_space_locked(ctx, 0u, ring_count, drain_one, &sentinel_ok) == OB_OK &&
                sentinel_ok) {
              ob::CachedWeightTile* sentinel = new (std::nothrow) ob::CachedWeightTile();
              if (sentinel) {
                sentinel->key = wh;
                sentinel->ext2_bytes = 0u;
                sentinel->rejected_swiglu = true;
                if (!ob::weight_cache_insert_locked(ctx, sentinel)) {
                  delete sentinel;
                }
              }
            }
            if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
            drain_all_cleanup();
            slice_rejected = true;
            break;
          }

          // Check activation + weight-norm eligibility BEFORE packing/allocating the chained DMA-BUF.
          // If rejected, cache the 0-byte activation-independent norms so subsequent fallback tokens
          // skip all weight conversions in O(1) without touching DMA memory.
          if (!eval_swiglu_eligibility(
                  s_wg,
                  l2_g_max, l2_g_mean, l2_u_max, l2_u_mean, l2_d_max,
                  dc_g_max, dc_u_max, l1_g_max, linf_g_max, l1_u_max, linf_u_max,
                  &d_gate, &d_up, &d_mul, &d_down, &s_gate_cand)) {
            bool norms_ok = false;
            if (ob::weight_cache_prepare_space_locked(ctx, 0u, ring_count, drain_one, &norms_ok) == OB_OK &&
                norms_ok) {
              ob::CachedWeightTile* norms_tile = new (std::nothrow) ob::CachedWeightTile();
              if (norms_tile) {
                norms_tile->key = wh;
                norms_tile->hw_k = K;
                norms_tile->hw_n = N;
                norms_tile->ext2_bytes = 0u;
                norms_tile->swiglu_norms_only = true;
                norms_tile->swiglu_s_wg = s_wg;
                norms_tile->swiglu_l2_gate_max = l2_g_max;
                norms_tile->swiglu_l2_gate_mean = l2_g_mean;
                norms_tile->swiglu_l2_up_max = l2_u_max;
                norms_tile->swiglu_l2_up_mean = l2_u_mean;
                norms_tile->swiglu_l2_down_max = l2_d_max;
                norms_tile->swiglu_dc_gate_max = dc_g_max;
                norms_tile->swiglu_dc_up_max = dc_u_max;
                norms_tile->swiglu_l1_gate_max = l1_g_max;
                norms_tile->swiglu_linf_gate_max = linf_g_max;
                norms_tile->swiglu_l1_up_max = l1_u_max;
                norms_tile->swiglu_linf_up_max = linf_u_max;
                if (!ob::weight_cache_insert_locked(ctx, norms_tile)) {
                  delete norms_tile;
                }
              }
            }
            if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
            drain_all_cleanup();
            slice_rejected = true;
            break;
          }
          eligibility_checked = true;

          const size_t ext2_sz = ob::pack_ext2_size_for_swiglu(ms.hw_m, K, hw_h, N);
          bool cacheable = false;
          const size_t ext2_slab_bytes = ob::buffer_slab_size(ext2_sz);
          ob_status_t pst = ob::weight_cache_prepare_space_locked(
              ctx, ext2_slab_bytes, ring_count, drain_one, &cacheable);
          if (pst != OB_OK) {
            drain_all_cleanup();
            return pst;
          }
          if (slice_rejected) {
            drain_all_cleanup();
            break;
          }

          ob_buffer* wbuf = nullptr;
          ob_status_t wst = ob_buffer_alloc(
              ctx, ext2_sz, OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL, &wbuf);
          if (wst == OB_OK) {
            wst = ob::buffer_sync_begin_write(wbuf);
            if (wst == OB_OK) {
              ob::pack_ext2_for_swiglu_into(
                  static_cast<uint8_t*>(ob_buffer_data(wbuf)),
                  ms.hw_m, K, hw_h, N,
                  wg_q.data(), nullptr,
                  wu_q.data(), nullptr,
                  wd_q.data(), nullptr,
                  &ctx->worker_pool);
              wst = ob::buffer_sync_end_write(wbuf);
            }
          }
          if (prof_on) ctx->prof.weight_pack_ns += (ob::monotonic_now_ns() - tp0);
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
          w_tile->ext2_bytes = wbuf->slab.alloc_size;
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
        }

        ob::PinnedWeightTile pinned_w(w_tile);

        if (!eligibility_checked &&
            !eval_swiglu_eligibility(
                w_tile->swiglu_s_wg,
                w_tile->swiglu_l2_gate_max, w_tile->swiglu_l2_gate_mean,
                w_tile->swiglu_l2_up_max, w_tile->swiglu_l2_up_mean,
                w_tile->swiglu_l2_down_max,
                w_tile->swiglu_dc_gate_max, w_tile->swiglu_dc_up_max,
                w_tile->swiglu_l1_gate_max, w_tile->swiglu_linf_gate_max,
                w_tile->swiglu_l1_up_max, w_tile->swiglu_linf_up_max,
                &d_gate, &d_up, &d_mul, &d_down, &s_gate_cand)) {
          pinned_w.reset();
          drain_all_cleanup();
          slice_rejected = true;
          break;
        }

        if (ring_count >= ring_depth) {
          ob_status_t dst = drain_one();
          if (dst != OB_OK) {
            pinned_w.reset();
            drain_all_cleanup();
            return dst;
          }
          if (slice_rejected) {
            pinned_w.reset();
            drain_all_cleanup();
            break;
          }
        }
        const uint32_t slot = (ring_head + ring_count) % ring_depth;
        const float s_gate = s_gate_cand;
        const uint64_t gkey = ob::hash_swiglu_graph_key(
            ms.hw_m, K, hw_h, N, d_gate, d_up, d_mul, d_down,
            s_gate, slot % param_swap_slots, ctx->freetile_enabled);
        ob_swiglu_int8_desc_t gd = make_graph_desc(hw_h, d_gate, d_up, d_mul, d_down, s_gate);

        const uint64_t ts0 = prof_on ? ob::monotonic_now_ns() : 0u;
        ob::CachedGraph* entry = nullptr;
        bool was_hit = false;
        ob_status_t gst = ob::acquire_swiglu_entry(ctx, gd, gkey, &entry, &was_hit);
        if (gst != OB_OK) {
          pinned_w.reset();
          drain_all_cleanup();
          return gst;
        }
        agg_flags |= OB_STAT_FLAG_EMITTER_USED;
        if (entry && entry->emitted_by_freetile) agg_flags |= OB_STAT_FLAG_FREETILE;
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
          pinned_w.reset();
          drain_all_cleanup();
          return bst;
        }
        total_tiles += 1u;
        chained_chunks += 1u;

        PendingSwigluChunk& p = ring[slot];
        p.handle = exec_h;
        p.w_tile = pinned_w.release();
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
        p.is_first_chunk = (h0 == 0u);
        ring_count += 1u;

        if (!ctx->mock_mode) {
          ob_status_t od_st = try_drain_ready();
          if (od_st != OB_OK) {
            drain_all_cleanup();
            return od_st;
          }
        }
      }

      if (slice_rejected) {
        drain_all_cleanup();
      } else {
        while (ring_count > 0u) {
          ob_status_t dst = drain_one();
          if (dst != OB_OK) {
            drain_all_cleanup();
            return dst;
          }
          if (slice_rejected) {
            drain_all_cleanup();
            break;
          }
        }
        if (!slice_rejected && ctx->mock_mode) {
          cpu_swiglu_f32_quant_slice(a_slice, desc, ms.tm, c_slice, &ctx->worker_pool);
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
      if ((fb.flags & OB_STAT_FLAG_EMITTER_USED) != 0u) {
        fb_slices += 1u;
      }
    }
  }

  const uint64_t wall_ns = ob::monotonic_now_ns() - t0_ns;
  {
    std::lock_guard<std::mutex> lk(ctx->gemm_mu);
    ctx->prof.swiglu_calls += 1u;
    ctx->prof.swiglu_chunks_chained += chained_chunks;
    if (fb_slices > 0u) {
      ctx->prof.swiglu_fallbacks += 1u;
    }
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

ob_status_t ob_gemm_matmul_f32_w8a16_tile(ob_context_t* ctx,
                                          const float* a_k,
                                          const int8_t* w_nk,
                                          const float* sb_n,
                                          uint32_t k_dim,
                                          uint32_t n_dim,
                                          float* c_n,
                                          ob_invoke_stats_t* out_stats) {
  if (!ctx || !a_k || !w_nk || !sb_n || !c_n) {
    return OB_ERR_INVALID_ARG;
  }
  if (!ob::is_w8a16_pt_1x_shape(1u, k_dim, n_dim)) {
    return OB_ERR_UNSUPPORTED;
  }

  const uint64_t t0_ns = ob::monotonic_now_ns();
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);

  std::vector<float> in_f32(k_dim);
  const float prescale_inv = ob::prepare_w8a16_input_row_f32(a_k, k_dim, k_dim, in_f32.data());

  if (ctx->mock_mode || !ctx->builtin_emitter_active || ctx->builtin_dive_interp.empty()) {
    // Exact FP16-input + FP32-accumulator hardware simulator for host/mock verification
    std::vector<float> a_fp16_f32(k_dim);
    for (uint32_t ki = 0u; ki < k_dim; ++ki) {
      // Exact IEEE-754 RNE FP16 round trip on every architecture (ARM64 and portable x86/host)
#if defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
      a_fp16_f32[ki] = static_cast<float>(static_cast<__fp16>(in_f32[ki]));
#else
      a_fp16_f32[ki] = ob::round_f32_to_fp16_f32(in_f32[ki]);
#endif
    }
    std::vector<float> y_raw(n_dim, 0.0f);
    for (uint32_t ni = 0u; ni < n_dim; ++ni) {
      const int8_t* wrow = w_nk + static_cast<size_t>(ni) * k_dim;
      float acc = 0.0f;
      for (uint32_t ki = 0u; ki < k_dim; ++ki) {
        acc += a_fp16_f32[ki] * static_cast<float>(wrow[ki]);
      }
      y_raw[ni] = acc;
    }
    ob::w8a16_pt_epilogue_f32(c_n, y_raw.data(), sb_n, n_dim, prescale_inv, true);
    if (out_stats) {
      std::memset(out_stats, 0, sizeof(*out_stats));
      out_stats->wall_time_us = (ob::monotonic_now_ns() - t0_ns) / 1000ULL;
      out_stats->tiles_dispatched = 0u;
      if (ctx->mock_mode) out_stats->flags |= OB_STAT_FLAG_MOCK;
      out_stats->flags |= OB_STAT_FLAG_CPU_FALLBACK;
    }
    return OB_OK;
  }

  // Acquire cached Probe-A weight-agnostic 7-chunk PT W8A16(FP16) (scale=1.0f) graph
  ob::CachedGraph* entry = nullptr;
  bool was_hit = false;
  ob_status_t st = ob::acquire_w8a16_pt_1x_entry(ctx, k_dim, n_dim, 1.0f, 0u, &entry, &was_hit);
  if (st != OB_OK) return st;
  const bool entry_freetile = (entry && entry->emitted_by_freetile);

  // Pack [N/32, K/2, 32, 2] w_i8 ^ 0x80 weights into a DMA-BUF (cached by 128-bit content hash!)
  ob::WeightHash128 wh_st{};
  const uint64_t wh_tag = 0x57384131365F5431ULL;
  uint32_t whdr[4] = {0x57384136u, k_dim, n_dim, 2u};
  ob::hash128_update(&wh_st, &wh_tag, sizeof(wh_tag));
  ob::hash128_update(&wh_st, whdr, sizeof(whdr));
  ob::hash_weight_tile_for_ctx(ctx, &wh_st, reinterpret_cast<const uint8_t*>(w_nk),
                               k_dim, n_dim, k_dim, ctx->strict_weight_hash);
  const ob::WeightTileKey wh = ob::finalize_weight_key(wh_st);
  ob::CachedWeightTile* w_tile = ob::weight_cache_find_locked(ctx, wh);
  if (!w_tile) {
    const size_t ext2_sz = ob::pack_w8a16_ext2_size(n_dim, k_dim);
    bool cacheable = false;
    uint32_t zero_inflight = 0u;
    auto noop_drain = [&]() -> ob_status_t { return OB_OK; };
    st = ob::weight_cache_prepare_space_locked(
        ctx, ob::buffer_slab_size(ext2_sz), zero_inflight, noop_drain, &cacheable);
    if (st != OB_OK) {
      ob::release_tile_entry(ctx, entry);
      return st;
    }
    ob_buffer* wbuf = nullptr;
    st = ob_buffer_alloc(ctx, ext2_sz, OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL, &wbuf);
    if (st == OB_OK) {
      st = ob::buffer_sync_begin_write(wbuf);
      if (st == OB_OK) {
        ob::pack_w8a16_ext2_into(static_cast<uint8_t*>(ob_buffer_data(wbuf)), n_dim, k_dim, w_nk);
        st = ob::buffer_sync_end_write(wbuf);
      }
    }
    if (st != OB_OK) {
      if (wbuf) ob_buffer_free(wbuf);
      ob::release_tile_entry(ctx, entry);
      return st;
    }
    w_tile = new (std::nothrow) ob::CachedWeightTile();
    if (!w_tile) {
      ob_buffer_free(wbuf);
      ob::release_tile_entry(ctx, entry);
      return OB_ERR_NO_MEMORY;
    }
    w_tile->key = wh;
    w_tile->scale_d = 1u;
    w_tile->hw_k = k_dim;
    w_tile->hw_n = n_dim;
    w_tile->ext2_buf = wbuf;
    w_tile->ext2_bytes = wbuf->slab.alloc_size;
    if (cacheable) {
      ob::weight_cache_insert_locked(ctx, w_tile);
    } else {
      w_tile->in_cache = false;
    }
  }

  ob::PinnedWeightTile pinned_w(w_tile);
  ob_buffer* in_buf = nullptr;
  ob_buffer* out_buf = nullptr;
  st = ob::ensure_pipeline_buf_tiered(
      ctx, &ctx->gemm_in_bufs_4k[0], &ctx->gemm_in_bufs[0],
      static_cast<size_t>(k_dim) * sizeof(float), &in_buf);
  if (st == OB_OK) {
    st = ob::ensure_pipeline_buf_tiered(
        ctx, &ctx->gemm_out_bufs_4k[0], &ctx->gemm_out_bufs[0],
        static_cast<size_t>(n_dim) * sizeof(float), &out_buf);
  }
  if (st == OB_OK) {
    st = ob_buffer_write(in_buf, in_f32.data(), static_cast<size_t>(k_dim) * sizeof(float), 0u);
  }
  ob_exec_handle_t exec_h = 0u;
  if (st == OB_OK) {
    st = ob::device_submit_execute_async(
        ctx, entry, &in_buf, 1u, &out_buf, 1u, &exec_h, w_tile->ext2_buf);
  }
  if (st != OB_OK) {
    ob::release_tile_entry(ctx, entry);
    return st;
  }

  ob::buffer_retain(in_buf);
  ob::buffer_retain(out_buf);
  ob_invoke_stats_t istats {};
  st = ob::device_wait_handle(ctx, exec_h, ctx->cfg.timeout_ms, &istats);
  if (st != OB_OK) {
    ob::detach_timed_out_buffers_locked(ctx, exec_h, w_tile, in_buf, out_buf);
  } else {
    st = ob::buffer_sync_begin_read(out_buf);
    if (st == OB_OK) {
      const float* y_raw = static_cast<const float*>(ob_buffer_data(out_buf));
      ob::w8a16_pt_epilogue_f32(c_n, y_raw, sb_n, n_dim, prescale_inv, true);
      st = ob::buffer_sync_end_read(out_buf);
    }
  }
  ob_buffer_free(in_buf);
  ob_buffer_free(out_buf);
  if (st == OB_OK && out_stats) {
    *out_stats = istats;
    out_stats->wall_time_us = (ob::monotonic_now_ns() - t0_ns) / 1000ULL;
    out_stats->tiles_dispatched = 1u;
    out_stats->cache_hits = was_hit ? 1u : 0u;
    out_stats->flags |= OB_STAT_FLAG_EMITTER_USED;
    if (entry_freetile) out_stats->flags |= OB_STAT_FLAG_FREETILE;
    if (was_hit) out_stats->flags |= OB_STAT_FLAG_CACHE_HIT;
  }
  return st;
}

}  // extern "C"
