// Clean-room C++17 Tensor G5 (buenos_a0) Multi-Tile (M, K, N) INT8 GEMM Emitter
// & Native FlatBuffer Container Builder (DIVE / DGD0 / MultiExecutable / DWS0 / DGC0).
//
// Zero embedded Google/vendor binaries: ext[1] (RV64 DIVE interpreter) is loaded
// dynamically at runtime from an external file path.
//
// Synthesizes chunk[0..6], ext[0] (DIVE), and ext[2] (packed bias + weights) in
// closed form matching the 81-model reference corpus bit-for-bit.

#include "ob_internal.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace ob {

namespace {

constexpr uint32_t kActNone = OB_ACT_NONE;
constexpr uint32_t kActRelu = OB_ACT_RELU;
constexpr uint32_t kActRelu6 = OB_ACT_RELU6;
constexpr uint64_t kCompilerCl = 925724637ULL;
constexpr const char* kDefaultDiveInterpPath =
    "/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin";

// 4x4 physical tile grid in serpentine (snake) traversal order
constexpr uint32_t kSnake16[16] = {
    0u, 1u, 2u, 3u, 7u, 6u, 5u, 4u, 8u, 9u, 10u, 11u, 15u, 14u, 13u, 12u
};

struct Flit {
  uint64_t lo;
  uint64_t hi;
};

using Pkt = std::vector<Flit>;

// ---------------------------------------------------------------------------
// Low-level bit / IEEE-754 float helpers
// ---------------------------------------------------------------------------

inline uint32_t pad4(uint32_t x) {
  return (x + 3u) & ~3u;
}

inline uint64_t align_up_u64(uint64_t v, uint64_t a) {
  return (v + a - 1ULL) & ~(a - 1ULL);
}

inline uint32_t next_pow2_u32(uint32_t v) {
  if (v <= 1u) return 1u;
  uint32_t p = 1u;
  while (p < v) p <<= 1u;
  return p;
}

inline uint32_t fp32_bits_f(float val) {
  uint32_t u = 0;
  std::memcpy(&u, &val, sizeof(u));
  return u;
}

inline uint32_t fp32_bits_d(double val) {
  float f = static_cast<float>(val);
  return fp32_bits_f(f);
}

inline double f32_round(double val) {
  return static_cast<double>(static_cast<float>(val));
}

inline uint32_t q_factor_m(uint32_t m, bool* even_flag) {
  if ((m % 4u) == 0u) {
    *even_flag = true;
    return m / 4u;
  }
  if ((m % 2u) == 0u) {
    *even_flag = true;
    return m / 2u;
  }
  *even_flag = false;
  return m;
}

std::vector<uint8_t> flits_to_bytes(const std::vector<Flit>& flits) {
  std::vector<uint8_t> out(flits.size() * 16u);
  for (size_t i = 0; i < flits.size(); ++i) {
    std::memcpy(out.data() + i * 16u, &flits[i].lo, 8u);
    std::memcpy(out.data() + i * 16u + 8u, &flits[i].hi, 8u);
  }
  return out;
}

std::vector<Flit> bytes_to_flits(const std::vector<uint8_t>& raw) {
  size_t n = raw.size() / 16u;
  std::vector<Flit> out(n);
  for (size_t i = 0; i < n; ++i) {
    std::memcpy(&out[i].lo, raw.data() + i * 16u, 8u);
    std::memcpy(&out[i].hi, raw.data() + i * 16u + 8u, 8u);
  }
  return out;
}

void set_bits_le(uint8_t* buf, size_t buf_len, uint32_t bit_off, uint32_t width, uint64_t val) {
  for (uint32_t i = 0; i < width; ++i) {
    uint32_t pos = bit_off + i;
    size_t byte_idx = pos >> 3u;
    uint32_t bit_idx = pos & 7u;
    if (byte_idx >= buf_len) break;
    if ((val >> i) & 1ULL) {
      buf[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
    } else {
      buf[byte_idx] &= static_cast<uint8_t>(~(1u << bit_idx));
    }
  }
}

uint32_t get_bits_le(const uint8_t* buf, size_t buf_len, uint32_t bit_off, uint32_t width) {
  uint32_t out = 0u;
  for (uint32_t i = 0; i < width; ++i) {
    uint32_t pos = bit_off + i;
    size_t byte_idx = pos >> 3u;
    uint32_t bit_idx = pos & 7u;
    if (byte_idx >= buf_len) break;
    if ((buf[byte_idx] >> bit_idx) & 1u) {
      out |= (1u << i);
    }
  }
  return out;
}

inline uint64_t mask_s(int64_t v, uint32_t bits) {
  if (bits >= 64u) return static_cast<uint64_t>(v);
  return static_cast<uint64_t>(v) & ((1ULL << bits) - 1ULL);
}

inline uint64_t stamp_pid(uint64_t lo, uint32_t pid) {
  uint32_t op = static_cast<uint32_t>(lo & 0xFFFu);
  uint64_t p = static_cast<uint64_t>(pid & 0xFFFFu);
  if (op == 0x680u || op == 0x6C0u) {
    return (lo & ~(0xFFFFULL << 28)) | (p << 28);
  }
  if (op == 0x900u || op == 0x940u || op == 0x980u || op == 0x9C0u || op == 0xA40u) {
    return (lo & ~(0xFFFFULL << 12)) | (p << 12);
  }
  return (lo & ~(0xFFFFULL << 48)) | (p << 48);
}

struct PacketStream {
  uint32_t pid = 0;
  uint32_t n_pkts = 0;
  std::vector<Flit> flits;

  explicit PacketStream(uint32_t start_pid) : pid(start_pid) {}

  uint32_t emit(const Pkt& pkt, bool stamp = true) {
    uint32_t cur_pid = pid;
    if (!pkt.empty()) {
      if (stamp) {
        flits.push_back(Flit{stamp_pid(pkt[0].lo, cur_pid), pkt[0].hi});
        for (size_t i = 1; i < pkt.size(); ++i) flits.push_back(pkt[i]);
      } else {
        for (const Flit& f : pkt) flits.push_back(f);
      }
    }
    pid += 1u;
    n_pkts += 1u;
    return cur_pid;
  }
};

// ---------------------------------------------------------------------------
// Constant baseline VLIW templates (from buenos_a0 GEMM ISA spec)
// ---------------------------------------------------------------------------

constexpr Flit kChunk0Flits[20] = {
    {0x000000000000F680ULL, 0x0001003000000040ULL},
    {0x0000000009108000ULL, 0x0000000000000000ULL},
    {0x00000000100F0680ULL, 0x0001003000000040ULL},
    {0x0000000009208000ULL, 0x0000000000000000ULL},
    {0x0000000020F00680ULL, 0x0001003000000040ULL},
    {0x0000000009408000ULL, 0x0000000000000000ULL},
    {0x000000003F000680ULL, 0x0001003000000040ULL},
    {0x0000000009808000ULL, 0x0000000000000000ULL},
    {0x0000FFFF00004900ULL, 0xFF80000040000000ULL},
    {0x000000000000007FULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000400100ULL},
    {0xFFFF00005FFFF680ULL, 0x000FFFF8000003FFULL},
    {0x0000000000008000ULL, 0x0000000000000000ULL},
    {0xFFF0FFFF00006900ULL, 0xFF8000004000001FULL},
    {0x000000000000007FULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x00000000000003E0ULL},
};

constexpr Flit kBaseOp5c0[13] = {
    {0x000000000FFFF5C0ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0200008000000000ULL, 0x001FFFC40000F000ULL},
    {0xFF1000003FFF8800ULL, 0x0000FFFE2000007FULL},
    {0x0000000000000010ULL, 0xC000006000060000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0900000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
};

constexpr Flit kBaseOp4c0[10] = {
    {0x000000000FFFF4C0ULL, 0x0000000200020000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x000000FC000FC000ULL, 0x0000000000000004ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x000024533F7E0000ULL, 0x0000000080000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x000F000000000000ULL, 0x0007E00000000000ULL},
};

constexpr Flit kBaseOp640[13] = {
    {0x0000000000001640ULL, 0x0000043000000000ULL},
    {0x0000000000000420ULL, 0x0000000000000000ULL},
    {0x0000042000000C30ULL, 0x0000000000000000ULL},
    {0x00000C3000000000ULL, 0x0000000000000420ULL},
    {0x0000000000000000ULL, 0x0000042000000C30ULL},
    {0x0000000000000000ULL, 0x00000C3000000000ULL},
    {0x0000000000000420ULL, 0x0000000000000000ULL},
    {0x0000042000000C30ULL, 0x0000000000000000ULL},
    {0x00000C3000000000ULL, 0x0000000000000420ULL},
    {0x0000000000000000ULL, 0x0000042000000C30ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x000007F8000007F8ULL, 0x000007F8000007F8ULL},
    {0x078017F8000007F8ULL, 0x0000000000000000ULL},
};

constexpr Flit kBaseHdrData[20] = {
    {0x0000000000001200ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000004000000000ULL},
    {0x0000000000008000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x00000C0000000000ULL, 0xE020000000000000ULL},
    {0x000000000200001FULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x00F0000000000000ULL, 0x0001000000000F00ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000008000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000009B40018C00ULL, 0x0000000002000001ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x7000214000000000ULL, 0xC000000000800000ULL},
    {0x008000000000007BULL, 0x103F800007B00000ULL},
    {0x0000000608000000ULL, 0x0000000000008000ULL},
};

constexpr Flit kBaseOp500[9] = {
    {0x0000000000001500ULL, 0x0000000080030000ULL},
    {0x0000000000000000ULL, 0x0288000000000000ULL},
    {0x0040000000000000ULL, 0x0000120003800010ULL},
    {0xFFC80001FFFE4000ULL, 0x200007FFF900003FULL},
    {0x0000420C0000FFFFULL, 0x0000000000000000ULL},
    {0x9394000000000000ULL, 0xD402000001000000ULL},
    {0x0009000000FFFFFFULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0109000000000000ULL, 0x0000000000000000ULL},
};

// ---------------------------------------------------------------------------
// Common barrier / sync / CSR packet constructors
// ---------------------------------------------------------------------------

Pkt pkt_680(uint32_t mask, uint64_t hi0, uint64_t lo1 = 0ULL, uint64_t top16 = 0ULL) {
  return {
      {(top16 << 48) | ((static_cast<uint64_t>(mask) & 0xFFFFULL) << 12) | 0x680ULL, hi0},
      {lo1, 0ULL},
  };
}

Pkt pkt_680_3f9(uint64_t lo1 = 0ULL) {
  return pkt_680(0xFFFFu, 0x000FFFF8000003F9ULL, lo1, 0xFFFFULL);
}

Pkt pkt_680_3c0(uint64_t lo1 = 0x8000ULL) {
  return pkt_680(0xFFFFu, 0x000FFFF8000003C0ULL, lo1, 0ULL);
}

Pkt pkt_680_3e0() {
  return pkt_680(0xFFFFu, 0x000F0008000003E0ULL, 0ULL, 0x0007ULL);
}

Pkt pkt_900_bar_all() {
  return {
      {0x0000FFFF00000900ULL, 0xFF80000040000000ULL},
      {0x7FULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0x180ULL},
  };
}

Pkt pkt_900_zero() {
  return {
      {0x0000000000000900ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0x180ULL},
  };
}

Pkt pkt_900_mask(uint32_t mask) {
  uint64_t m = static_cast<uint64_t>(mask & 0xFFFFu);
  return {
      {(m << 32) | 0x900ULL, (m << 55) | 0x40000000ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0x400100ULL},
  };
}

Pkt pkt_940_params(uint32_t ext2_bytes) {
  uint64_t u_neg = mask_s(-static_cast<int64_t>(ext2_bytes) + 4LL, 16u);
  uint64_t u_cnt = mask_s(static_cast<int64_t>(ext2_bytes / 4u) - 8LL, 16u);
  return {
      {0x0001000000000940ULL, 0x3C00400078008FFFULL},
      {0x0F0010001E002000ULL, (u_neg << 48) | (u_cnt << 32) | 0x80010800ULL},
      {0x000003FFE0040311ULL, 0x0000000000000004ULL},
  };
}

Pkt pkt_980_params(uint32_t g, uint32_t n_groups, uint32_t grp_bytes, bool m_ge_48) {
  uint64_t u_neg = mask_s(-2LL * static_cast<int64_t>(grp_bytes) + 8LL, 16u);
  uint64_t u_cnt = mask_s(static_cast<int64_t>(grp_bytes / 2u) - 15LL, 16u);
  uint64_t off_w = static_cast<uint64_t>(g * (grp_bytes / 8u)) & 0xFFFFULL;
  uint64_t dst_hi = static_cast<uint64_t>(0x02u + g * (grp_bytes / 16u)) & 0xFFFFULL;
  uint64_t f0l = (0x0002ULL << 48) | (off_w << 32) | 0x980ULL;
  uint64_t f2h = (dst_hi << 40) | 0x0221FF8C00ULL;
  Flit f3;
  if (m_ge_48) {
    f3 = {0x0000000458000004ULL, 0x0101000000C00000ULL};
  } else {
    uint64_t b48 = (n_groups > 1u) ? 0x01ULL : 0x81ULL;
    f3 = {0x0000000008000004ULL, ((1ULL << g) << 56) | (b48 << 48) | 0x00800000ULL};
  }
  return {
      {f0l, 0x78008000F0011FFEULL},
      {0x1E0020003C004000ULL, (u_neg << 48) | (u_cnt << 32) | 0x00021000ULL},
      {0x0000000000000003ULL, f2h},
      f3,
      {0ULL, 0ULL},
  };
}

using CsrBank = std::array<uint32_t, 0x62>;

Pkt pack_csr_packet(uint32_t op, uint32_t mask,
                    const std::vector<std::pair<uint32_t, uint32_t>>& items,
                    bool ef_flag = false) {
  std::vector<uint32_t> u32s;
  u32s.reserve(items.size());
  for (const auto& kv : items) {
    u32s.push_back(((kv.second & 0x1FFFFFFu) << 7) | (kv.first & 0x7Fu));
  }
  uint32_t n_csrs = static_cast<uint32_t>(u32s.size());
  uint32_t tw = 2u * n_csrs + 1u;
  uint64_t first_u32 = u32s.empty() ? 0ULL : static_cast<uint64_t>(u32s[0]);
  uint64_t lo0 = ((static_cast<uint64_t>(mask) & 0xFFFFULL) << 12) | (op & 0xFFFu);
  uint64_t hi0 = (first_u32 << 32) | ((static_cast<uint64_t>(tw) & 0xFFFULL) << 12) |
                 (ef_flag ? 0x50ULL : 0ULL);
  Pkt flits;
  flits.push_back({lo0, hi0});
  for (size_t i = 1; i < u32s.size(); i += 4u) {
    uint64_t w0 = (i + 0u < u32s.size()) ? u32s[i + 0u] : 0u;
    uint64_t w1 = (i + 1u < u32s.size()) ? u32s[i + 1u] : 0u;
    uint64_t w2 = (i + 2u < u32s.size()) ? u32s[i + 2u] : 0u;
    uint64_t w3 = (i + 3u < u32s.size()) ? u32s[i + 3u] : 0u;
    flits.push_back({w0 | (w1 << 32), w2 | (w3 << 32)});
  }
  return flits;
}

// ---------------------------------------------------------------------------
// 1. Parameter buffer packing (ext[2])
// ---------------------------------------------------------------------------

std::vector<uint8_t> pack_ext2(uint32_t N, uint32_t K,
                               const int8_t* weights_nk,
                               const int32_t* bias_n,
                               bool include_bias) {
  uint32_t k4 = pad4(K);
  uint32_t n4 = pad4(N);
  uint32_t n_blk = (N <= 32u) ? std::max<uint32_t>(8u, ((N + 7u) / 8u) * 8u) : 32u;
  uint32_t n_groups = (N <= 32u) ? 1u : ((N + 31u) / 32u);

  size_t grp_sz = (include_bias ? static_cast<size_t>(n_blk) * 4u : 0u) +
                  static_cast<size_t>(k4) * static_cast<size_t>(n_blk);
  std::vector<uint8_t> out;
  out.reserve(grp_sz * n_groups);

  for (uint32_t g = 0; g < n_groups; ++g) {
    uint32_t n0 = g * 32u;
    if (include_bias) {
      for (uint32_t ni = 0; ni < n_blk; ++ni) {
        uint32_t nn = n0 + ni;
        int32_t b = (bias_n && nn < N) ? bias_n[nn] : 0;
        uint32_t u = static_cast<uint32_t>(b);
        out.push_back(static_cast<uint8_t>(u & 0xFFu));
        out.push_back(static_cast<uint8_t>((u >> 8) & 0xFFu));
        out.push_back(static_cast<uint8_t>((u >> 16) & 0xFFu));
        out.push_back(static_cast<uint8_t>((u >> 24) & 0xFFu));
      }
    }
    for (uint32_t kg = 0; kg < k4 / 4u; ++kg) {
      for (uint32_t ni = 0; ni < n_blk; ++ni) {
        uint32_t nn = n0 + ni;
        for (uint32_t kk = 0; kk < 4u; ++kk) {
          uint32_t kkk = kg * 4u + kk;
          if (nn < N && kkk < K) {
            int32_t w = weights_nk ? static_cast<int32_t>(weights_nk[static_cast<size_t>(nn) * K + kkk]) : 0;
            out.push_back(static_cast<uint8_t>((w + 128) & 0xFF));
          } else if (nn < n4 && kkk < k4) {
            out.push_back(0x80u);
          } else {
            out.push_back(0x00u);
          }
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// 2. Static & DMA chunks: chunk[0], chunk[1], chunk[2], chunk[4]
// ---------------------------------------------------------------------------

std::vector<uint8_t> build_chunk0() {
  std::vector<Flit> fl(std::begin(kChunk0Flits), std::end(kChunk0Flits));
  return flits_to_bytes(fl);
}

std::vector<uint8_t> build_chunk1(uint32_t M, uint32_t K) {
  std::vector<Flit> fl = {
      {0x000000007FFFF680ULL, 0x000FFFF8000003C0ULL},
      {0x0000000000000000ULL, 0x0000000000000000ULL},
      {0x0001000100008940ULL, 0x0000000000000000ULL},
      {0x0000000000000000ULL, 0x0000000010010800ULL},
      {0x000003FFFC000210ULL | (static_cast<uint64_t>(M) << 18), 0x0000000000000004ULL},
  };
  uint32_t in_chunks = (M * K + 31u) / 32u;
  if (in_chunks > 1u) {
    uint64_t c = static_cast<uint64_t>(in_chunks - 1u) * 8ULL;
    fl[2].hi = 0x3F8040007F0081FFULL;
    fl[3].lo = 0x0FE010001FC02000ULL;
    uint64_t top = (0x3FE4ULL - 4ULL * c) & 0xFFFFULL;
    fl[3].hi = (((top << 16) | c) << 32) | 0x10010800ULL;
  }
  return flits_to_bytes(fl);
}

std::vector<uint8_t> build_dma_a40_chunk(uint32_t pid, uint32_t n_bytes, uint64_t f0_hi_word) {
  uint64_t dma_enc = static_cast<uint64_t>(n_bytes - 1u) * 8ULL;
  uint64_t f3_lo = (dma_enc << 48) | (1ULL << 47);
  uint64_t f3_hi = (dma_enc >> 16) | (dma_enc << 16);
  std::vector<Flit> fl = {
      {(f0_hi_word << 32) | ((static_cast<uint64_t>(pid) & 0xFFFFULL) << 12) | 0xA40ULL,
       0x0000002000000000ULL},
      {0ULL, 0ULL},
      {0ULL, 0x0000020000000000ULL},
      {f3_lo, f3_hi},
  };
  return flits_to_bytes(fl);
}

std::vector<uint8_t> build_chunk2(uint32_t M, uint32_t K) {
  return build_dma_a40_chunk(9u, M * K, 0x00000001ULL);
}

std::vector<uint8_t> build_chunk4(uint32_t c4_pid, uint32_t ext2_bytes) {
  return build_dma_a40_chunk(c4_pid, ext2_bytes, 0x00000022ULL);
}

// ---------------------------------------------------------------------------
// 3. Shared mid-chunk helpers & SRAM/streaming metrics
// ---------------------------------------------------------------------------

bool is_weight_streaming(uint32_t K, uint32_t N) {
  uint32_t n_pad_w = (N <= 32u) ? std::max<uint32_t>(8u, ((N + 7u) / 8u) * 8u)
                                : (((N + 31u) / 32u) * 32u);
  uint32_t k4 = pad4(K);
  return K > 64u && (k4 * n_pad_w >= 4096u);
}

uint32_t compute_sw_unit(uint32_t M, uint32_t K, uint32_t N) {
  if (!is_weight_streaming(K, N)) return 0u;
  uint32_t k4 = pad4(K);
  uint32_t last_980 = 0u;
  if ((k4 % 32u) == 0u) {
    last_980 = ((M * k4) / 32u) * 0x200u;
  } else {
    last_980 = (((M - 1u) * k4) / 32u + 1u) * 0x200u;
  }
  uint32_t np2 = next_pow2_u32(last_980);
  return np2 / 0x800u;
}

uint32_t compute_exec_field14(uint32_t M, uint32_t K, uint32_t N) {
  uint32_t k4 = pad4(K);
  uint32_t sw = compute_sw_unit(M, K, N);
  if (sw > 0u) {
    uint32_t extra = (N >= 32u && (K % 32u) == 0u) ? 320u : 256u;
    return 16u * (sw * 128u + M * std::min<uint32_t>(32u, k4)) + extra;
  }
  if (M >= 48u) {
    return 16u * (2u * M * std::min<uint32_t>(32u, k4)) + 320u;
  }
  uint32_t base = std::max<uint32_t>(128u, ((M * std::min<uint32_t>(32u, k4) + 127u) / 128u) * 128u);
  uint32_t extra = (M >= 4u && ((M * std::min<uint32_t>(32u, k4)) % 128u) == 0u) ? 320u : 256u;
  return 16u * (base + M * k4) + extra;
}

uint32_t compute_dive_scratch_size(uint32_t M, uint32_t N) {
  uint32_t n4 = pad4(N);
  return (M * n4 >= 1024u) ? 196608u : 192512u;
}

Flit dma_2d_rewind_f1(uint32_t step) {
  uint64_t s = static_cast<uint64_t>(step);
  uint64_t neg8 = mask_s(-static_cast<int64_t>(s) * 8LL, 16u);
  uint64_t f1l = (neg8 << 48) | ((0x80000000ULL - s * 0x400ULL) & 0xFFFFFFFFULL);
  uint64_t f1h = (0x0002000000000000ULL - (s << 28)) | 0xFFULL;
  return {f1l, f1h};
}

Pkt build_op5c0(uint32_t M, uint32_t d_pad4, bool zero_f0hi) {
  uint32_t d16 = std::min<uint32_t>(16u, d_pad4);
  uint32_t d32 = std::min<uint32_t>(32u, d_pad4);
  Pkt fl(std::begin(kBaseOp5c0), std::end(kBaseOp5c0));
  uint64_t val4c = zero_f0hi ? 0ULL : static_cast<uint64_t>(M * d32);
  fl[3].lo = (fl[3].lo & 0xFFFF0000FFFFFFFFULL) | (val4c << 35);
  fl[3].hi = 0x001FFF0400000000ULL | (static_cast<uint64_t>(16u - (d16 / 4u)) << 36) |
             (static_cast<uint64_t>(d16 - 1u) << 12);
  if (d16 == 4u) {
    fl[4] = {0xFFD000003FFFE800ULL, 0x0000FFFFA000007FULL};
  } else if (d16 == 8u) {
    fl[4] = {0xFF9000003FFFC800ULL, 0x0000FFFF2000007FULL};
  } else if (d16 == 12u) {
    fl[4] = {0xFF5000003FFFA800ULL, 0x0000FFFEA000007FULL};
  } else {
    fl[4] = {0xFF1000003FFF8800ULL, 0x0000FFFE2000007FULL};
  }
  return fl;
}

Pkt build_op4c0(uint32_t M, uint32_t d_pad4, bool zero_f0hi) {
  uint32_t d16 = std::min<uint32_t>(16u, d_pad4);
  uint32_t d32 = std::min<uint32_t>(32u, d_pad4);
  Pkt fl(std::begin(kBaseOp4c0), std::end(kBaseOp4c0));
  uint64_t val = static_cast<uint64_t>(d16 * 4u - 1u);
  fl[0].hi = zero_f0hi ? 0x000200000000ULL
                       : (0x000200000000ULL | (static_cast<uint64_t>(M * d32) << 13));
  fl[4].lo = (val << 34) | (val << 14);
  fl[6].lo = 0x0000245300000000ULL | (((val << 8) | (val * 2ULL)) << 16);
  fl[9].lo = static_cast<uint64_t>(d16 - 1u) << 48;
  return fl;
}

Pkt build_op640(uint32_t d, float scale, int32_t zp) {
  Pkt fl(std::begin(kBaseOp640), std::end(kBaseOp640));
  uint32_t n_c32 = (d + 31u) / 32u;
  if (n_c32 > 1u) {
    uint64_t mask = (1ULL << n_c32) - 1ULL;
    fl[0].lo = (fl[0].lo & 0xFFFFFFFFFFFF0FFFULL) | (mask << 12);
    fl[0].hi = 0x0000043000000000ULL;
  }
  std::vector<uint8_t> raw = flits_to_bytes(fl);
  double scale_f = f32_round(static_cast<double>(scale));
  uint32_t inv_bits = fp32_bits_d(1.0 / scale_f);
  for (uint32_t lane = 0; lane < 8u; ++lane) {
    set_bits_le(raw.data(), raw.size(), 108u + 160u * lane, 32u, inv_bits);
  }
  uint32_t zp_bits = (zp != 0) ? fp32_bits_d(static_cast<double>(-zp) * scale_f) : 0u;
  set_bits_le(raw.data(), raw.size(), 10u * 128u + 76u, 32u, zp_bits);
  return bytes_to_flits(raw);
}

std::vector<Pkt> build_op6c0_pkts(uint32_t M, uint32_t d_pad4, uint32_t d_mask,
                                  uint32_t base_off = 0u,
                                  bool has_out_off = false, uint32_t out_off = 0u) {
  bool ef = false;
  uint32_t q = q_factor_m(M, &ef);
  if (!ef) return {};
  uint32_t m_grp = M / q;
  uint32_t d_sub = std::min<uint32_t>(32u, d_pad4);
  std::vector<Pkt> pkts;
  if (m_grp == 4u) {
    uint64_t lo_base = ((static_cast<uint64_t>(d_mask) & 0xFFFFULL) << 12) | 0x6C0ULL;
    uint32_t stride = (M / 4u) * d_sub;
    uint32_t b_out = has_out_off ? out_off : base_off;
    for (uint32_t g = 0; g < 4u; ++g) {
      uint64_t a_in = static_cast<uint64_t>(base_off + g * stride);
      uint64_t a_out = static_cast<uint64_t>(b_out + g * stride);
      uint64_t lane = (1ULL << g) << 56;
      pkts.push_back({{lo_base, lane | (a_in << 16) | 0x8000ULL}});
      pkts.push_back({{lo_base, lane | (a_out << 16) | 0x8400ULL}});
    }
  } else {
    uint32_t s_hi = (M / 2u) * d_sub;
    struct Entry {
      uint32_t tmask;
      uint32_t lane_bit;
      uint32_t addr;
    };
    std::vector<Entry> entries;
    if (d_pad4 > 32u && (d_pad4 % 32u) != 0u) {
      uint32_t n_full = d_pad4 / 32u;
      uint32_t m_full = (1u << n_full) - 1u;
      uint32_t m_rem = 1u << n_full;
      entries.push_back({d_mask, 1u, base_off});
      entries.push_back({m_full, 2u, base_off + s_hi});
      entries.push_back({m_rem, 2u, base_off + (M / 2u) * (d_pad4 % 32u)});
    } else {
      entries.push_back({d_mask, 1u, base_off});
      entries.push_back({d_mask, 2u, base_off + s_hi});
    }
    for (const Entry& e : entries) {
      uint64_t lo_base = ((static_cast<uint64_t>(e.tmask) & 0xFFFFULL) << 12) | 0x6C0ULL;
      uint64_t hi_base = (static_cast<uint64_t>(e.lane_bit) << 56) |
                         (static_cast<uint64_t>(e.addr) << 16);
      pkts.push_back({{lo_base, hi_base | 0x8000ULL}});
      pkts.push_back({{lo_base, hi_base | 0x8400ULL}});
    }
  }
  return pkts;
}

Pkt build_hdr_data(uint32_t M, uint32_t d, float scale, int32_t zp, uint32_t sw_unit = 0u) {
  Pkt fl(std::begin(kBaseHdrData), std::end(kBaseHdrData));
  uint32_t d_pad4 = pad4(d);
  uint32_t d_sub = std::min<uint32_t>(32u, d_pad4);
  uint32_t d16 = std::min<uint32_t>(16u, d_sub);
  uint32_t d_rem = (d_sub > 16u) ? (d_sub - 16u) : 0u;
  uint32_t d_tail = (d_sub > 16u) ? std::min<uint32_t>(d_rem, d_sub & (~d_sub + 1u)) : 0u;
  bool ef = false;
  uint32_t q = q_factor_m(M, &ef);
  uint32_t m_grp = M / q;
  uint32_t qm1 = q - 1u;
  uint32_t n_full32 = std::max<uint32_t>(1u, d / 32u);
  bool sym32 = (q == 1u && d_sub == 32u);
  bool two_sub = (d_sub > 16u);

  fl[0].lo = 0x0000000000000200ULL | (((1ULL << n_full32) - 1ULL) << 12);
  fl[0].hi = sym32 ? (0x2000ULL | (ef ? 0x50ULL : 0ULL))
                   : ((two_sub ? (1ULL << 27) : 0ULL) |
                      (static_cast<uint64_t>(qm1) << 13) | (ef ? 0x50ULL : 0ULL));
  fl[1].hi = (static_cast<uint64_t>(qm1) << 55) |
             (static_cast<uint64_t>(d_sub / 4u) << 36) |
             (static_cast<uint64_t>(sw_unit * 2u) << 20);

  if (!two_sub && q == 1u) {
    fl[2] = {static_cast<uint64_t>(d16 / 4u) << 13, 0ULL};
    fl[3] = {0ULL, 0ULL};
    fl[4] = {0x00000C0000000000ULL,
             0x6020000000000000ULL | (((d16 % 8u) == 0u) ? (1ULL << 63) : 0ULL)};
    uint64_t d16_tag = (d16 == 4u) ? 6ULL : ((d16 == 8u) ? 0xEULL : ((d16 == 12u) ? 0x17ULL : 0x1FULL));
    fl[5] = {0x0000000002000000ULL | d16_tag | (static_cast<uint64_t>(sw_unit) << 12), 0ULL};
    fl[6] = {0ULL, 0ULL};
    fl[7] = {0ULL, 0ULL};
    fl[8] = {static_cast<uint64_t>(d16 - 1u) << 52,
             0x0001000000000000ULL | (static_cast<uint64_t>(d16 - 1u) << 8)};
    fl[9] = {0ULL, 0ULL};
    fl[12] = {0x8000ULL, 0ULL};
    fl[13] = {0ULL, 0ULL};
    fl[14].lo = 0x0000009B40018C00ULL;
  } else {
    int64_t S = static_cast<int64_t>((two_sub ? 16u : 0u) + qm1 * d_sub);
    int64_t T = 0;
    if (two_sub && d_sub < 32u) {
      T = static_cast<int64_t>(48u - (16u + d_tail) + qm1 * d_sub);
    } else if (two_sub) {
      T = static_cast<int64_t>(48u - d_sub + qm1 * d_sub);
    } else {
      T = static_cast<int64_t>(d_sub * qm1);
    }
    uint64_t low_f2l = (static_cast<uint64_t>(static_cast<uint32_t>(16 - static_cast<int32_t>(qm1 * d_sub))) << 11) & 0xFFFFFFFFULL;
    fl[2].lo = (mask_s(-S, 12u) << 52) | (two_sub ? (1ULL << 32) : 0ULL) | low_f2l;
    fl[2].hi = (mask_s(-S, 21u) << 29) | 0x1FFULL;
    // S is always a multiple of 4, so (-S) / 2 == floor(-S / 2)
    fl[3].lo = (mask_s((-S) / 2LL, 16u) << 48) | (mask_s(0x200000LL - S, 21u) << 6);
    fl[3].hi = (mask_s(0x200000LL - S, 21u) << 24) | 0xFULL;
    uint64_t f4h = 0x6000000000000000ULL |
                   (((d16 % 8u) == 0u) ? (1ULL << 63) : 0ULL) |
                   (sym32 ? 0x0020000000000000ULL : 0x0040000000000000ULL);
    fl[4] = {0x00000C0000000000ULL | (mask_s(0x200000LL - S, 21u) << 1), f4h};
    uint32_t step_eff = sym32 ? 16u : d_sub;
    uint32_t lb = step_eff & (~step_eff + 1u);
    uint64_t f5_upper = (lb == 4u) ? 3ULL : ((lb == 8u) ? 1ULL : static_cast<uint64_t>(std::max<uint32_t>(1u, qm1)));
    fl[5].lo = (f5_upper << 46) | (static_cast<uint64_t>(lb) << 25) |
               (static_cast<uint64_t>(sw_unit) << 12) | 0x1FULL;
    int64_t neg45 = 0;
    uint64_t lo23 = 0;
    if (d_sub == 8u || d_sub == 16u || d_sub == 32u) {
      neg45 = S / 4LL;
      int64_t tag23 = (d_sub == 32u && q >= 2u) ? 2LL : 1LL;
      lo23 = mask_s((tag23 << 23) - (S - 32LL * (tag23 - 1LL)) * 4LL, 32u);
    } else {
      neg45 = (q == 1u) ? static_cast<int64_t>(8u - d_tail / 4u)
                        : ((S - static_cast<int64_t>(d_sub)) / 4LL);
      uint64_t tail_part = (q == 1u) ? (static_cast<uint64_t>(d_tail) << 2)
                                     : (static_cast<uint64_t>(std::min<uint32_t>(2u, q) * d_rem) << 2);
      lo23 = (static_cast<uint64_t>(std::max<uint32_t>(1u, q - 1u)) << 23) | tail_part;
    }
    fl[5].hi = (mask_s(-neg45, 19u) << 45) | lo23;
    int64_t t4 = T / 4LL;
    fl[6].lo = (static_cast<uint64_t>(t4 & 1LL) << 63) |
               (mask_s(0x200000LL - T, 21u) << 20) |
               ((two_sub && d_rem < 16u && q >= 2u) ? 1ULL : 0ULL);
    fl[6].hi = (mask_s(-t4, 19u) << 40) | mask_s(0x40000LL - ((T + 7LL) / 8LL), 18u);
    fl[7].lo = (mask_s(-T, 8u) << 56) | (mask_s(0x100000LL - (T / 2LL), 20u) << 16);
    uint64_t f7h_top = (!two_sub || sym32) ? 0x2ULL
                                           : ((d_sub == 32u || q == 1u) ? 0x6ULL : 0xEULL);
    fl[7].hi = (f7h_top << 32) | ((S >= 480LL && two_sub) ? 0x1FFEULL : 0x1FFFULL);
    uint64_t d_tail_m1 = (two_sub && d_sub < 32u && q == 1u)
                             ? static_cast<uint64_t>(d_tail - 1u)
                             : (two_sub ? static_cast<uint64_t>(d_rem - 1u)
                                        : static_cast<uint64_t>(d16 - 1u));
    uint64_t f8l_lo = 0ULL;
    if (two_sub && d_sub < 32u) {
      uint64_t sh = (q >= 2u) ? 0x400000ULL : 0x200000ULL;
      f8l_lo = (static_cast<uint64_t>(16u - d_rem) << 24) | sh | ((d_sub == 28u) ? 4ULL : 0ULL);
    }
    uint64_t f8l_m = 0ULL;
    if (two_sub && d_sub < 32u && q >= 2u) {
      f8l_m = 0x4000ULL;
    } else if (two_sub && (d_sub < 32u || q >= 2u)) {
      f8l_m = 0x2000ULL;
    } else if (sym32) {
      f8l_m = 0x1000ULL;
    }
    fl[8] = {(d_tail_m1 << 52) | (f8l_m << 32) | f8l_lo, d_tail_m1 << 8};
    fl[9] = {((two_sub && q >= 2u) ? (0x4000ULL << 32) : 0ULL) |
                 (static_cast<uint64_t>(std::max<uint32_t>(1u, qm1)) << 5),
             0ULL};
    fl[12] = {((two_sub && q >= 2u) ? (0x00200000ULL << 32) : 0ULL) |
                  (static_cast<uint64_t>(std::max<uint32_t>(1u, qm1)) << 28),
              0ULL};
    fl[13] = {0ULL, 0x0000001000000000ULL};
    if (sym32) {
      fl[14].lo = 0x0000009B40008400ULL;
    } else if (q == 1u) {
      fl[14].lo = 0x000000AB40008C00ULL;
    } else if (two_sub && d_rem < 16u) {
      fl[14].lo = 0x000000AB40019400ULL;
    } else {
      fl[14].lo = 0x000000AB40018C00ULL;
    }
  }

  double scale_f = f32_round(static_cast<double>(scale));
  std::vector<uint8_t> raw = flits_to_bytes(fl);
  const uint32_t b17 = 17u * 128u;
  set_bits_le(raw.data(), raw.size(), b17 + 45u, 4u, (1u << m_grp) - 1u);
  set_bits_le(raw.data(), raw.size(), b17 + 80u, 8u, static_cast<uint32_t>((zp + 128) & 0xFF));
  set_bits_le(raw.data(), raw.size(), b17 + 189u, 32u, fp32_bits_d(scale_f));
  set_bits_le(raw.data(), raw.size(), b17 + 222u, 32u,
              fp32_bits_d(static_cast<double>(127 - zp) * scale_f));
  set_bits_le(raw.data(), raw.size(), b17 + 259u, 32u,
              fp32_bits_d(static_cast<double>(-128 - zp) * scale_f));
  set_bits_le(raw.data(), raw.size(), b17 + 328u, 8u, static_cast<uint32_t>((zp + 128) & 0xFF));
  if (q > 1u || (two_sub && d_rem < 16u)) {
    set_bits_le(raw.data(), raw.size(), b17 + 115u, 1u, 1u);
  }
  uint32_t v_lo = (d16 / 4u) - 1u;
  uint32_t v_hi = two_sub ? (d_rem / 4u - 1u) : v_lo;
  set_bits_le(raw.data(), raw.size(), b17 + 128u, 8u, (v_lo << 5) | 0x18u | v_hi);
  return bytes_to_flits(raw);
}

CsrBank unpack_hdr_csrs(const Pkt& hdr_flits) {
  std::vector<uint8_t> raw = flits_to_bytes(hdr_flits);
  CsrBank csrs{};
  for (uint32_t r = 0; r < 0x62u; ++r) {
    csrs[r] = get_bits_le(raw.data(), raw.size(), 77u + 25u * r, 25u);
  }
  return csrs;
}

CsrBank build_vica_csr_state(uint32_t M, uint32_t d_rem, uint32_t sw_unit,
                             const CsrBank& h32_csrs, int32_t zp) {
  bool ef = false;
  uint32_t q = q_factor_m(M, &ef);
  CsrBank hrm = unpack_hdr_csrs(build_hdr_data(M, d_rem, 0.0625f, zp, sw_unit));
  if (q > 1u) {
    hrm[0x57] = h32_csrs[0x57];
    if (d_rem < 16u) {
      hrm[0x18] = h32_csrs[0x18];
    }
  } else if (d_rem < 16u) {
    hrm[0x07] = h32_csrs[0x07];
    hrm[0x16] = h32_csrs[0x16];
  }
  // Keep 0x5a..0x61 from h32_csrs (VICA_CSR only diffs 0x00..0x59)
  for (uint32_t r = 0x5Au; r < 0x62u; ++r) {
    hrm[r] = h32_csrs[r];
  }
  return hrm;
}

Pkt build_vica_csr_pkt(uint32_t M, uint32_t d, uint32_t sw_unit,
                       const CsrBank& h32_csrs, int32_t zp, CsrBank* out_state) {
  uint32_t d_rem = pad4(d % 32u);
  bool ef = false;
  (void)q_factor_m(M, &ef);
  CsrBank v_state = build_vica_csr_state(M, d_rem, sw_unit, h32_csrs, zp);
  std::vector<std::pair<uint32_t, uint32_t>> diff;
  for (uint32_t r = 0; r < 0x5Au; ++r) {
    if (v_state[r] != h32_csrs[r]) {
      diff.push_back({r, v_state[r]});
    }
  }
  if (out_state) *out_state = v_state;
  uint32_t mask = 1u << (d / 32u);
  return pack_csr_packet(0x200u, mask, diff, ef);
}

// ---------------------------------------------------------------------------
// 4. Output tile partition (shared by chunk[5] scatter & chunk[6] gather)
// ---------------------------------------------------------------------------

struct TileSpec {
  uint32_t tid;
  uint32_t nbytes;
};

std::vector<TileSpec> compute_output_tiles(uint32_t M, uint32_t N) {
  uint32_t n4_bytes = pad4(N);
  std::vector<TileSpec> out;
  if (M == 1u) {
    uint32_t n_groups = (N + 31u) / 32u;
    for (uint32_t g = 0; g < n_groups; ++g) {
      uint32_t nb = std::min<uint32_t>(32u, n4_bytes - 32u * g);
      out.push_back({g, nb});
    }
    return out;
  }
  if (M <= 16u) {
    uint32_t u = n4_bytes / 4u;
    uint32_t row_stride = 16u / M;
    uint32_t t_row = std::min<uint32_t>(u, row_stride);
    uint32_t base_u = u / t_row;
    uint32_t rem_u = u % t_row;
    for (uint32_t r = 0; r < M; ++r) {
      for (uint32_t j = 0; j < t_row; ++j) {
        uint32_t nb = (base_u + (j < rem_u ? 1u : 0u)) * 4u;
        out.push_back({r * row_stride + j, nb});
      }
    }
    return out;
  }
  uint32_t per_tile = (M * n4_bytes) / 16u;
  for (uint32_t k = 0; k < 16u; ++k) {
    out.push_back({k, per_tile});
  }
  return out;
}

// ---------------------------------------------------------------------------
// 5. Compute CSR state (0x00..0x61) for chunk[3] CSR_040
// ---------------------------------------------------------------------------

void pack_stream_a(CsrBank& csrs, uint32_t ah,
                   const std::array<std::pair<int64_t, int64_t>, 7>& entries) {
  uint8_t bits[64]{};
  set_bits_le(bits, sizeof(bits), 0u, 20u, ah & ((1u << 20) - 1u));
  for (uint32_t k = 0; k < 7u; ++k) {
    uint64_t v = mask_s(entries[k].first, 22u) | (mask_s(entries[k].second, 19u) << 22);
    set_bits_le(bits, sizeof(bits), 20u + 41u * k, 41u, v);
  }
  for (uint32_t i = 0; i < 13u; ++i) {
    csrs[0x06u + i] = get_bits_le(bits, sizeof(bits), 25u * i, 25u);
  }
}

void pack_stream_h(CsrBank& csrs, uint32_t hh,
                   const std::array<std::pair<int64_t, int64_t>, 8>& entries) {
  uint8_t bits[64]{};
  set_bits_le(bits, sizeof(bits), 0u, 15u, hh & ((1u << 15) - 1u));
  for (uint32_t k = 0; k < 8u; ++k) {
    uint64_t v = mask_s(entries[k].first, 19u) | (mask_s(entries[k].second, 22u) << 19);
    set_bits_le(bits, sizeof(bits), 15u + 41u * k, 41u, v);
  }
  for (uint32_t i = 0; i < 14u; ++i) {
    csrs[0x17u + i] = get_bits_le(bits, sizeof(bits), 25u * i, 25u);
  }
}

void pack_stream_b(CsrBank& csrs, uint32_t bh,
                   const std::array<std::pair<int64_t, int64_t>, 7>& entries) {
  uint8_t bits[64]{};
  set_bits_le(bits, sizeof(bits), 0u, 30u, bh & ((1u << 30) - 1u));
  for (uint32_t k = 0; k < 7u; ++k) {
    uint64_t v = mask_s(entries[k].first, 20u) | (mask_s(entries[k].second, 21u) << 20);
    set_bits_le(bits, sizeof(bits), 30u + 41u * k, 41u, v);
  }
  for (uint32_t i = 0; i < 13u; ++i) {
    csrs[0x2Au + i] = get_bits_le(bits, sizeof(bits), 25u * i, 25u);
  }
}

CsrBank build_compute_csr_state(uint32_t M, uint32_t K, uint32_t n_tile,
                                float s_in, int32_t zp_in, float s_w,
                                float s_out, int32_t zp_out, uint32_t act,
                                bool has_bias, uint32_t sw_unit,
                                const CsrBank& h3_csrs, uint32_t N) {
  uint32_t k4 = pad4(K);
  uint32_t k_unit = k4 / 4u;
  uint32_t k_inner = std::min<uint32_t>(12u, k_unit);
  uint32_t k_passes = (k_unit + 11u) / 12u;
  uint32_t n4 = n_tile / 4u;
  bool m_ge_48 = (M >= 48u);
  uint32_t m_eff = m_ge_48 ? (M / 4u) : M;
  bool multi_k = (k_passes >= 2u);
  bool multi_m = (M >= 2u);

  CsrBank csrs{};

  // 0x00, 0x01
  uint32_t outer_cnt = multi_m ? (m_eff - 1u) : (k_passes - 1u);
  csrs[0x00] = (outer_cnt << 14) | (k_inner - 1u);
  csrs[0x01] = (multi_m && multi_k) ? 8u : 0u;

  // 0x04, 0x05, 0x13..0x16, 0x38, 0x39, 0x43, 0x44
  uint32_t k_rem12 = k_unit % 12u;
  bool partial_k = multi_k && (k_rem12 != 0u);
  if (partial_k) {
    csrs[0x04] = (k_rem12 - 1u) << 12;
    csrs[0x13] = 0x1Cu;
    csrs[0x14] = (2u * k_rem12 - 2u) << 8;
    csrs[0x15] = ((64u / k_rem12) << 8) | 0x20u;
    csrs[0x16] = multi_m ? 0xCC10u : 0xCC08u;
    csrs[0x38] = ((k_rem12 - 1u) << 19) | 0x0Cu;
    csrs[0x39] = ((16u / k_rem12) << 20) | (multi_m ? 0x8000u : 0x4000u);
    csrs[0x43] = 0x1800000u | (multi_m ? 1u : 0u);
    csrs[0x44] = (multi_m ? 0x800u : 0x400u) | ((k_rem12 == 8u) ? 1u : 0u);
  } else {
    csrs[0x13] = 0x0Cu;
    csrs[0x16] = multi_k ? 0xCC08u : 0xCC00u;
    csrs[0x38] = (m_ge_48 && k4 <= 16u) ? 4u : (m_ge_48 ? 8u : 0x0Cu);
    csrs[0x43] = m_ge_48 ? 0x1080u : (multi_m ? 1u : 0u);
  }

  // 0x05
  if (m_ge_48) {
    csrs[0x05] = 0u;
  } else if (sw_unit > 0u) {
    csrs[0x05] = partial_k ? (multi_m ? 4u : 2u) : 0u;
  } else {
    uint32_t base_05 = std::max<uint32_t>(1u, (M * std::min<uint32_t>(32u, k4) + 127u) / 128u);
    csrs[0x05] = (base_05 << 11) | (partial_k ? (multi_m ? 4u : 2u) : 0u);
  }

  // Stream A (0x06..0x12)
  std::array<std::pair<int64_t, int64_t>, 7> a_ents{};
  if (k_unit == 1u) {
    a_ents[0] = {0, 1};
    for (size_t i = 1; i < 7; ++i) a_ents[i] = {0, 0};
  } else if (!multi_k) {
    if (m_ge_48) {
      int64_t back = -(static_cast<int64_t>(m_eff) * k_unit - 1LL);
      a_ents[0] = {static_cast<int64_t>(k_unit) - 1LL, 1LL};
      a_ents[1] = {static_cast<int64_t>(m_eff) - 1LL, back};
      for (size_t i = 2; i < 7; ++i) a_ents[i] = {0, back};
    } else if (!multi_m) {
      int64_t back = -(static_cast<int64_t>(k_unit) - 1LL);
      a_ents[0] = {static_cast<int64_t>(k_unit) - 1LL, 1LL};
      for (size_t i = 1; i < 7; ++i) a_ents[i] = {0, back};
    } else {
      int64_t tot_a = static_cast<int64_t>(M) * k_unit - 1LL;
      int64_t st0 = -static_cast<int64_t>(M - 1u) * k_unit + 1LL;
      a_ents[0] = {tot_a, st0};
      for (size_t i = 1; i < 7; ++i) a_ents[i] = {0, -tot_a};
    }
  } else if (!partial_k) {
    int64_t back = -(static_cast<int64_t>(k_unit) - 1LL);
    a_ents[0] = {11LL, 1LL};
    a_ents[1] = {static_cast<int64_t>(k_passes) - 1LL, back};
    for (size_t i = 2; i < 7; ++i) a_ents[i] = {0, back};
  } else {
    int64_t st0 = static_cast<int64_t>(k_unit) - 11LL;
    int64_t st1 = -static_cast<int64_t>(M - 1u) * k_unit + 1LL;
    int64_t st2 = -(static_cast<int64_t>(M - 1u) * k_unit + 12LL * k_passes - 1LL);
    a_ents[0] = {11LL, st0};
    a_ents[1] = {static_cast<int64_t>(m_eff) - 1LL, st1};
    a_ents[2] = {static_cast<int64_t>(k_passes) - 1LL, st2};
    for (size_t i = 3; i < 7; ++i) a_ents[i] = {0, st2};
  }
  pack_stream_a(csrs, 2u, a_ents);

  // Stream H (0x17..0x24)
  std::array<std::pair<int64_t, int64_t>, 8> h_ents{};
  if (!multi_m) {
    pack_stream_h(csrs, 0x2000u | sw_unit, h_ents);
  } else {
    int64_t st_back = -static_cast<int64_t>(n4) * static_cast<int64_t>(m_eff - 1u);
    h_ents[0] = {static_cast<int64_t>(n4), static_cast<int64_t>(m_eff - 1u)};
    for (size_t i = 1; i < 8; ++i) h_ents[i] = {st_back, 0LL};
    pack_stream_h(csrs, sw_unit, h_ents);
    csrs[0x24] |= 0x10000u;
  }

  // Stream B (0x2a..0x36)
  std::array<std::pair<int64_t, int64_t>, 7> b_ents{};
  if (k_unit == 1u) {
    pack_stream_b(csrs, 0x200u, b_ents);
  } else {
    int64_t b_st = -4LL * static_cast<int64_t>(k_inner - 1u);
    int64_t b0 = static_cast<int64_t>(k_inner - 1u);
    int64_t b1 = multi_m ? static_cast<int64_t>(m_eff - 1u) : static_cast<int64_t>(k_passes - 1u);
    int64_t b2 = (multi_m && multi_k) ? static_cast<int64_t>(k_passes - 1u) : 0LL;
    b_ents[0] = {b0, b_st};
    b_ents[1] = {b1, b_st};
    b_ents[2] = {b2, b_st};
    for (size_t i = 3; i < 7; ++i) b_ents[i] = {0, b_st};
    pack_stream_b(csrs, 0x800u, b_ents);
  }

  // Stream C (0x3c..0x42)
  if (multi_m) {
    csrs[0x3C] = ((m_eff - 1u) << 12) | (m_ge_48 ? 0u : 2u);
    if (!m_ge_48) {
      uint32_t c_st = static_cast<uint32_t>(mask_s(-2LL * static_cast<int64_t>(M - 1u), 12u));
      for (uint32_t r = 0x3Du; r <= 0x42u; ++r) csrs[r] = c_st;
      if (multi_k) csrs[0x3D] |= (k_passes - 1u) << 12;
    }
  } else if (multi_k) {
    csrs[0x3C] = (k_passes - 1u) << 12;
  }

  // Misc scalars
  csrs[0x27] = 0x1000000u;
  csrs[0x28] = ((n_tile - 1u) << 19) | ((n_tile / 2u) - 1u);
  csrs[0x29] = (M == 32u) ? 0x200000u : 0u;
  csrs[0x37] = (multi_m ? 0x200000u : 0x100000u) | ((k_unit > 1u) ? 0x1000u : 0u);
  if (k_unit == 1u) {
    csrs[0x3A] = 0x1000000u;
  } else if (M == 2u) {
    csrs[0x3A] = 0x68000u;
  } else if (multi_m && M < 32u) {
    csrs[0x3A] = 0x60000u;
  } else {
    csrs[0x3A] = 0u;
  }
  csrs[0x3B] = (k_inner - 1u) << 12;
  csrs[0x45] = multi_m ? 0xE000A3u : 0xE00063u;
  csrs[0x46] = (multi_m ? 0x14u : 0x12u) | ((K == 64u && n_tile == 32u && N == 64u) ? 0x20u : 0u);
  csrs[0x47] = 0x10u;
  if (!has_bias) {
    csrs[0x48] = 0x10u;
  } else {
    csrs[0x48] = 0x13610u + (multi_m ? 0x2000u : 0u) + (multi_k ? 0x2000u : 0u);
    csrs[0x49] = 0x10000u;
    csrs[0x4A] = 0x10000u;
  }
  csrs[0x55] = m_ge_48 ? 0x785800u : (multi_m ? 0x81800u : 0x85800u);
  csrs[0x56] = 0x100000u;
  csrs[0x57] = h3_csrs[0x57];
  if (k_unit == 1u) {
    csrs[0x58] = 0x4000u;
  } else if (multi_m && multi_k) {
    csrs[0x58] = 0xAC000u;
  } else if (multi_k) {
    csrs[0x58] = 0x68000u;
  } else if (multi_m) {
    csrs[0x58] = 0x28000u;
  } else {
    csrs[0x58] = 0x24000u;
  }
  csrs[0x59] = ((n_tile - 1u) << 5) | (n_tile - 1u);

  // Quantization & activation clamp CSRs (0x5a..0x61) — exact FP32 ULP model
  double s_in_f = f32_round(static_cast<double>(s_in));
  double s_w_f = f32_round(static_cast<double>(s_w));
  double s_out_f = f32_round(static_cast<double>(s_out));
  double inv_s_out = f32_round(1.0 / s_out_f);
  uint32_t u_rm = fp32_bits_d(f32_round(f32_round(s_in_f * s_w_f) / s_out_f));

  int32_t qmin = -128;
  int32_t qmax = 127;
  if (act == kActRelu) {
    qmin = std::max<int32_t>(-128, std::min<int32_t>(127, zp_out));
  } else if (act == kActRelu6) {
    qmin = std::max<int32_t>(-128, std::min<int32_t>(127, zp_out));
    // Match Python round() (IEEE-754 RNE via std::nearbyint)
    int32_t r6 = static_cast<int32_t>(std::nearbyint(6.0 / s_out_f));
    qmax = std::max<int32_t>(-128, std::min<int32_t>(127, r6 + zp_out));
  }

  uint32_t u_max = fp32_bits_d(f32_round(f32_round(static_cast<double>(qmax - zp_out) * s_out_f) * inv_s_out));
  uint32_t u_min = fp32_bits_d(static_cast<double>(qmin - zp_out));
  uint32_t bf_min = (u_min >> 16) & 0xFFFFu;

  csrs[0x5A] = 0u;
  csrs[0x5B] = (u_rm & 0xFFFu) << 13;
  csrs[0x5C] = ((u_rm >> 12) & 0xFFFFFu) | ((u_max & 0xFu) << 21);
  csrs[0x5D] = (u_max >> 4) & 0x1FFFFFFu;
  csrs[0x5E] = ((bf_min & 1u) << 24) | ((u_max >> 29) & 0x7u);
  csrs[0x5F] = bf_min >> 1;
  csrs[0x60] = 0u;
  csrs[0x61] = ((has_bias ? 1u : 0u) << 18) | (static_cast<uint32_t>((zp_out + 128) & 0xFF) << 2);

  (void)zp_in;
  return csrs;
}

// ---------------------------------------------------------------------------
// 6. Build chunk[3]
// ---------------------------------------------------------------------------

struct C3SyncInfo {
  uint32_t c3_440_ch1 = 0;
  uint32_t c3_44c = 0;
  uint32_t c3_dst_cnt = 0;
  uint32_t c3_ch2 = 0;
  uint32_t c3_48c = 0;
};

Pkt build_pre3_3c0(uint32_t mask, uint32_t k_slice_bytes, uint32_t sram_off,
                   uint32_t row_off_in, uint32_t row_idx, int32_t zp_in, bool stream_w) {
  uint64_t km1_4 = static_cast<uint64_t>(k_slice_bytes - 1u) * 4ULL;
  uint64_t f0h = (static_cast<uint64_t>(k_slice_bytes) * 2ULL << 32) |
                 (static_cast<uint64_t>(sram_off) << 13);
  uint64_t f2h_top = (static_cast<uint64_t>(row_off_in) * 0x40ULL) | (stream_w ? 0x4000ULL : 0ULL);
  uint64_t f2h = ((f2h_top & 0xFFFFULL) << 48) | (km1_4 << 32) | (km1_4 << 12);
  uint64_t f3l = (static_cast<uint64_t>(k_slice_bytes) * 0x400ULL) | (f2h_top >> 16);
  uint64_t f4h = (km1_4 * 2ULL) << 52;
  uint64_t f5l = (static_cast<uint64_t>(row_idx + 1u) << 48) | (0x4EULL << 40) |
                 ((km1_4 * 2ULL) << 8);
  uint64_t zp_b = static_cast<uint64_t>(zp_in & 0xFF);
  uint64_t f12h = ((zp_b & 0xFULL) << 60) | 0x0100000000000000ULL;
  uint64_t f13l = ((zp_b >> 4) & 0xFULL) | (zp_b << 4);
  return {
      {((static_cast<uint64_t>(mask) & 0xFFFFULL) << 12) | 0x3C0ULL, f0h},
      {0ULL, 0ULL},
      {0x20ULL, f2h},
      {f3l, 0ULL},
      {0x0000400000000000ULL, f4h},
      {f5l, 0x0000000400000200ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, f12h},
      {f13l, 0ULL},
  };
}

Pkt build_post3_3c0(uint32_t M, uint32_t k_sl, uint32_t k_full,
                    uint32_t src_off, bool dst_zero) {
  uint32_t kp = pad4(k_sl);
  uint32_t nb = M * kp;
  uint64_t bm1_4 = static_cast<uint64_t>(nb - 1u) * 4ULL;
  uint32_t dst = 0u;
  if (!dst_zero) {
    uint32_t eff_k = std::min<uint32_t>(32u, k_full ? k_full : kp);
    dst = (M == 1u) ? 128u : std::max<uint32_t>(128u, ((M * eff_k + 127u) / 128u) * 128u);
  }
  uint64_t dst_enc = static_cast<uint64_t>(dst) * 0x40ULL;
  if (k_full > 32u && M >= 2u) {
    uint64_t sl_m1_4 = static_cast<uint64_t>(kp - 1u) * 4ULL;
    uint64_t x = sl_m1_4 * 2ULL;
    uint64_t f3l = ((static_cast<uint64_t>(k_full) * 8ULL) << 48) |
                   (static_cast<uint64_t>(kp) * 0x400ULL) | (dst_enc >> 16);
    uint64_t f3h = ((0x20000ULL - static_cast<uint64_t>(k_full / 16u)) << 32) |
                   (static_cast<uint64_t>(M - 1u) << 8);
    uint64_t f4l = 0x0000C00000000000ULL |
                   ((0x4000000ULL - static_cast<uint64_t>(k_full) * 32ULL) & 0x3FFFFFFULL);
    uint64_t f4h = x << 52;
    uint64_t f5l = (x << 8) | (x >> 12);
    return {
        {0x00000000000013C0ULL, (static_cast<uint64_t>(nb) * 2ULL << 32) | (static_cast<uint64_t>(src_off) << 13)},
        {0ULL, 0ULL},
        {0x20ULL, ((dst_enc & 0xFFFFULL) << 48) | (bm1_4 << 32) | (bm1_4 << 12)},
        {f3l, f3h},
        {f4l, f4h},
        {f5l, 0ULL},
        {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL},
        {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL},
    };
  }
  uint64_t x = bm1_4 * 2ULL;
  uint64_t f4h = x << 52;
  uint64_t f5l = (x << 8) | (x >> 12);
  return {
      {0x00000000000013C0ULL, (static_cast<uint64_t>(nb) * 2ULL << 32) | (static_cast<uint64_t>(src_off) << 13)},
      {0ULL, 0ULL},
      {0x20ULL, ((dst_enc & 0xFFFFULL) << 48) | (bm1_4 << 32) | (bm1_4 << 12)},
      {(static_cast<uint64_t>(nb) * 0x400ULL) | (dst_enc >> 16), 0ULL},
      {0x0000400000000000ULL, f4h},
      {f5l, 0ULL},
      {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL},
      {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL},
  };
}

Pkt build_post3_440_k64n64(uint32_t mask, uint32_t M, uint32_t dst_sram) {
  if (M == 1u) {
    Flit f1 = dma_2d_rewind_f1(32u);
    return {
        {(static_cast<uint64_t>(mask) << 12) | 0x440ULL,
         0x0040004000000000ULL | (static_cast<uint64_t>(dst_sram) << 13)},
        f1,
        {0x20ULL, 0ULL},
        {0x03E0000000000000ULL, 0x40003E00ULL},
        {0ULL, 0ULL},
        {0ULL, 0ULL},
        {0ULL, 0x20000000ULL},
        {0x0000800000000000ULL, 0ULL},
    };
  }
  uint64_t ym1 = static_cast<uint64_t>(M - 1u);
  uint64_t step = static_cast<uint64_t>((2u * M - 1u) * 32u);
  uint64_t neg8 = mask_s(-static_cast<int64_t>(step) * 8LL, 16u);
  uint64_t f1l = (neg8 << 48) | ((0x100000000ULL - (step - 64ULL) * 0x400ULL) & 0xFFFFFFFFULL);
  uint64_t f1h = (0x0002000000000000ULL - (step << 28)) | 0xFFULL;
  return {
      {(static_cast<uint64_t>(mask) << 12) | 0x440ULL,
       (ym1 << 54) | (0x80ULL << 32) | (static_cast<uint64_t>(dst_sram) << 13)},
      {f1l, f1h},
      {0x60ULL, 0ULL},
      {0x03E0000000000000ULL, 0xC0003E00ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0x20000000ULL},
      {0x0000800000000000ULL, 0ULL},
  };
}

Pkt build_post3_440_recv(uint32_t mask, uint32_t M, uint32_t k_sl, uint32_t dst_sram,
                         uint32_t op = 0x440u, uint32_t k_stride = 0u,
                         uint64_t f3h_hi = 0xC0000000ULL, uint64_t f6h = 0x00400000ULL) {
  uint64_t b_enc = static_cast<uint64_t>(k_sl - 1u) * 2ULL;
  if (M == 1u) {
    return {
        {(static_cast<uint64_t>(mask) << 12) | op,
         0x0000000200000000ULL | (static_cast<uint64_t>(dst_sram) << 13)},
        {0ULL, 0ULL},
        {0ULL, 0ULL},
        {b_enc << 52, f3h_hi | (b_enc << 8)},
        {0ULL, 0ULL},
        {0ULL, 0ULL},
        {0ULL, f6h},
        {0x0000800000000000ULL, 0ULL},
    };
  }
  uint32_t y_m1 = M - 1u;
  uint32_t stride_b = k_stride ? k_stride : k_sl;
  uint64_t f0h = (static_cast<uint64_t>(y_m1) << 54) |
                 ((static_cast<uint64_t>(stride_b) * 2ULL) << 32) |
                 (static_cast<uint64_t>(dst_sram) << 13);
  Flit f1 = dma_2d_rewind_f1(stride_b * y_m1);
  uint64_t f3l_flag = k_stride ? 0ULL : (0x2ULL << 48);
  return {
      {(static_cast<uint64_t>(mask) << 12) | op, f0h},
      f1,
      {0x20ULL, 0ULL},
      {(b_enc << 52) | f3l_flag, f3h_hi | (b_enc << 8)},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, f6h},
      {0x0000800000000000ULL, 0ULL},
  };
}

uint64_t snake_mask_bits(uint32_t logical_mask) {
  uint64_t dst_bits = 0ULL;
  for (uint32_t i = 0; i < 16u; ++i) {
    if ((logical_mask >> i) & 1u) {
      dst_bits |= (1ULL << kSnake16[i]);
    }
  }
  return dst_bits;
}

Pkt build_post3_400_bcast(uint32_t src_mask, uint32_t dst_mask, uint32_t nbytes,
                          uint64_t ch_val, uint32_t src_off = 0u,
                          uint32_t f4_mul = 1u, uint64_t f6_flag = 0x04000000ULL) {
  uint64_t b_enc = static_cast<uint64_t>(nbytes - 1u) * 2ULL;
  uint64_t dst_bits = snake_mask_bits(dst_mask);
  return {
      {(static_cast<uint64_t>(src_mask) << 12) | 0x400ULL,
       0x0000000200000000ULL | (static_cast<uint64_t>(src_off) << 13)},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {b_enc << 52, (ch_val << 32) | (b_enc << 8)},
      {0x0000000100000000ULL | (0x80ULL * static_cast<uint64_t>(f4_mul)), 0ULL},
      {0ULL, 0ULL},
      {0ULL, (dst_bits << 34) | f6_flag},
  };
}

Pkt build_post3_400_gather(uint32_t src_tile, uint32_t dst_mask, uint32_t M,
                           uint32_t k_sl, uint64_t ch_val,
                           uint32_t src_off = 0u, uint64_t f6_flag = 0x04000000ULL) {
  if (M == 1u) {
    return build_post3_400_bcast(1u << src_tile, dst_mask, k_sl, ch_val, src_off, 1u, f6_flag);
  }
  uint32_t ym1 = M - 1u;
  uint64_t b_enc = static_cast<uint64_t>(k_sl - 1u) * 2ULL;
  uint64_t dst_bits = snake_mask_bits(dst_mask);
  uint64_t f0h = (static_cast<uint64_t>(ym1) << 54) |
                 ((static_cast<uint64_t>(k_sl) * 2ULL) << 32) |
                 (static_cast<uint64_t>(src_off) << 13);
  Flit f1 = dma_2d_rewind_f1(k_sl * ym1);
  return {
      {(static_cast<uint64_t>(1u << src_tile) << 12) | 0x400ULL, f0h},
      f1,
      {0x20ULL, 0ULL},
      {b_enc << 52, (ch_val << 32) | (b_enc << 8)},
      {0x0000000100000080ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, (dst_bits << 34) | f6_flag},
  };
}

std::vector<uint8_t> build_chunk3(uint32_t M, uint32_t K, uint32_t N,
                                  float s_in, int32_t zp_in, float s_w,
                                  float s_out, int32_t zp_out, uint32_t act,
                                  bool has_bias, uint32_t ext2_bytes,
                                  uint32_t* out_n_pkts, C3SyncInfo* out_sync) {
  PacketStream ps(10u);
  uint32_t k4 = pad4(K);
  uint32_t n4 = pad4(N);
  uint32_t n_groups = (N + 31u) / 32u;
  uint32_t k_tiles = (K + 31u) / 32u;
  uint32_t k_mask = (1u << k_tiles) - 1u;
  uint32_t n_mask = (1u << n_groups) - 1u;
  bool stream_w = is_weight_streaming(K, N);
  uint32_t sw_unit = compute_sw_unit(M, K, N);
  bool m_ge_48 = (M >= 48u);
  uint32_t k_unit = k4 / 4u;
  uint32_t k_passes = (k_unit + 11u) / 12u;

  // A. pre3: input staging
  if ((K % 32u) == 0u) {
    ps.emit(pkt_680_3c0(0x8000ULL));
    ps.emit(pkt_900_bar_all());
    for (uint32_t r = 0; r < M; ++r) {
      uint32_t sram_off = sw_unit * 128u + r * 32u;
      ps.emit({
          {(static_cast<uint64_t>(k_mask) << 12) | 0x440ULL,
           0x0000000200000000ULL | (static_cast<uint64_t>(sram_off) << 13)},
          {0ULL, 0ULL},
          {0ULL, 0ULL},
          {0x03E0000000000000ULL, 0x0000000000003E00ULL},
          {0ULL, 0ULL},
          {0ULL, 0ULL},
          {0ULL, 0x00400000ULL},
          {0x0000800000000000ULL, 0ULL},
      });
      for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        uint64_t idx = static_cast<uint64_t>(r) * k_tiles + kt;
        uint64_t f0l = (0x0002ULL << 48) | ((1ULL + 4ULL * idx) << 32) | 0x980ULL;
        uint64_t f2h = ((0x02ULL + 2ULL * idx) << 40) | 0x021FF8C00ULL;
        uint32_t tile_bit = kSnake16[kt];
        uint64_t f3h = 0x0001000000800000ULL | ((tile_bit < 8u) ? (1ULL << (56u + tile_bit)) : 0ULL);
        uint64_t f4l = (tile_bit >= 8u) ? (1ULL << (tile_bit - 8u)) : 0ULL;
        ps.emit({
            {f0l, 0ULL},
            {0ULL, 0x20021000ULL},
            {0ULL, f2h},
            {0x08000004ULL, f3h},
            {f4l, 0ULL},
        });
      }
    }
  } else {
    uint32_t buf_base = stream_w ? 0u : (((M * std::min<uint32_t>(32u, k4) + 31u) / 32u) * 32u);
    for (uint32_t r = 0; r < M; ++r) {
      uint32_t row_byte = r * K;
      uint32_t off32 = row_byte % 32u;
      for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        uint32_t k_sl = std::min<uint32_t>(32u, K - 32u * kt);
        uint32_t sram_off = buf_base + kt * 32u + off32;
        uint32_t row_off_in = r * ((kt < k_tiles - 1u) ? 32u : pad4(k_sl));
        ps.emit(build_pre3_3c0(1u << kt, k_sl, sram_off, row_off_in, r, zp_in, stream_w));
      }
      if (r == 0u && k_tiles > 1u) {
        ps.emit(pkt_680(k_mask, 0x0001003000000040ULL, 0x0D108000ULL));
        ps.emit(pkt_900_mask(k_mask));
      }
      uint32_t c_start = row_byte / 32u;
      uint32_t c_end = (row_byte + K - 1u) / 32u;
      uint32_t n_c = c_end - c_start + 1u;
      uint32_t end_in_32 = ((row_byte + K - 1u) % 32u) + 1u;
      if (n_c == 1u) {
        uint32_t b_cnt = (row_byte + K) - c_start * 32u;
        uint64_t b_enc = static_cast<uint64_t>(b_cnt - 1u) * 2ULL;
        ps.emit({
            {(static_cast<uint64_t>(k_mask) << 12) | 0x440ULL,
             0x0000000200000000ULL | (static_cast<uint64_t>(buf_base) << 13)},
            {0ULL, 0ULL},
            {0ULL, 0ULL},
            {b_enc << 52, 0xC0000000ULL | (b_enc << 8)},
            {0ULL, 0ULL},
            {0ULL, 0ULL},
            {0ULL, 0x00400000ULL},
            {0x0000800000000000ULL, 0ULL},
        });
        uint64_t f0l_980 = (0x0002ULL << 48) | ((1ULL + 4ULL * c_start) << 32) | 0x980ULL;
        uint64_t bm1 = static_cast<uint64_t>(b_cnt - 1u);
        uint64_t f2h_980 = ((0x02ULL + 2ULL * c_start) << 40) | 0x20000000ULL |
                           (bm1 << 20) | (bm1 << 15) | 0x0C00ULL;
        ps.emit({
            {f0l_980, 0ULL},
            {0ULL, 0x20021000ULL},
            {0ULL, f2h_980},
            {0x08000004ULL, 0x0101000000800000ULL},
            {0ULL, 0ULL},
        });
      } else {
        uint32_t y_m1 = n_c - 1u;
        uint64_t b_last_enc = static_cast<uint64_t>(end_in_32 - 1u) * 2ULL;
        uint64_t f0h_440 = (static_cast<uint64_t>(y_m1) << 54) | (0x40ULL << 32) |
                           (static_cast<uint64_t>(buf_base) << 13);
        Flit f1_440 = dma_2d_rewind_f1(32u * y_m1);
        uint64_t f6h_440 = (k_tiles > 1u) ? 0x20000000ULL : 0x00400000ULL;
        ps.emit({
            {(static_cast<uint64_t>(k_mask) << 12) | 0x440ULL, f0h_440},
            f1_440,
            {0x20ULL, 0ULL},
            {(b_last_enc << 52) | (0x2ULL << 48), 0xC0003E00ULL},
            {0ULL, 0ULL},
            {0ULL, 0ULL},
            {0ULL, f6h_440},
            {0x0000800000000000ULL, 0ULL},
        });
        uint64_t f0l_980 = (0x0002ULL << 48) | ((1ULL + 4ULL * c_start) << 32) | 0x980ULL;
        uint64_t f1h_980 = (y_m1 == 1u) ? 0x7F88001020021000ULL : 0x7F48002020021000ULL;
        uint64_t bm1 = static_cast<uint64_t>(end_in_32 - 1u);
        uint64_t f2h_980 = ((0x02ULL + 2ULL * c_start) << 40) | 0x20000000ULL |
                           (31ULL << 20) | (bm1 << 15) | 0x0C00ULL;
        uint64_t f3h_980 = (k_tiles == 1u)
                               ? 0x0101000000800000ULL
                               : ((static_cast<uint64_t>(k_mask) << 56) | 0x0080000000800000ULL);
        ps.emit({
            {f0l_980, 0x7F008000FE0103FEULL},
            {0x1FC020003F804000ULL, f1h_980},
            {0ULL, f2h_980},
            {0x08000004ULL, f3h_980},
            {0ULL, 0ULL},
        });
      }
    }
  }

  ps.emit(pkt_680_3f9());
  ps.emit(pkt_680_3f9());

  // B. mid3
  ps.emit(build_op5c0(M, k4, stream_w));
  ps.emit(build_op4c0(M, k4, stream_w));
  ps.emit(pkt_680_3e0());
  ps.emit(build_op640(K, s_in, zp_in));
  for (const Pkt& p6 : build_op6c0_pkts(M, k4, k_mask, 0u)) {
    ps.emit(p6);
  }
  Pkt hdr3 = build_hdr_data(M, K, s_in, zp_in, sw_unit);
  ps.emit(hdr3);
  CsrBank h3_csrs = unpack_hdr_csrs(hdr3);
  CsrBank h3_vica_csrs = h3_csrs;
  if (K > 32u && (K % 32u) != 0u) {
    CsrBank v_state{};
    ps.emit(build_vica_csr_pkt(M, K, sw_unit, h3_csrs, zp_in, &v_state));
    h3_vica_csrs = v_state;
  }

  // C. post3
  C3SyncInfo sync{};
  if (m_ge_48) {
    ps.emit(pkt_680_3f9());
    ps.emit(pkt_680_3c0(0x8000ULL));
    ps.emit(pkt_900_bar_all());
  } else {
    ps.emit(pkt_680_3f9());
    ps.emit(pkt_680_3f9());
    if (K <= 32u && n_groups == 1u) {
      ps.emit(build_post3_3c0(M, K, 0u, 0u, false));
      ps.emit(pkt_680_3c0(0x8000ULL));
      ps.emit(pkt_900_bar_all());
      ps.emit(pkt_680_3f9());
      ps.emit(pkt_900_zero());
      ps.emit(pkt_680_3f9());
      ps.emit(pkt_680_3c0(0ULL));
    } else if (n_groups > 1u && K <= 32u) {
      sync.c3_44c = 1u;
      sync.c3_dst_cnt = n_groups;
      ps.emit(pkt_680(n_mask, 0x0001003000000040ULL, 0x0D108000ULL));
      ps.emit(pkt_900_mask(n_mask));
      if (n_groups < 4u) {
        ps.emit(pkt_680(n_mask, 0x0001003000000040ULL, 0x0F108000ULL));
        ps.emit(pkt_900_mask(n_mask));
      }
      uint32_t dst_sram = std::max<uint32_t>(128u, next_pow2_u32(M * k4));
      uint64_t f6_r = (n_groups == 4u) ? 0x00800000ULL : 0x10000000ULL;
      uint64_t f6_s = (n_groups == 4u) ? 0x08000000ULL : 0x100000000ULL;
      ps.emit(build_post3_440_recv(n_mask, M, k4, dst_sram, 0x440u, 0u, 0xC0000000ULL, f6_r));
      ps.emit(build_post3_400_bcast(1u, n_mask, M * k4, 0x144CULL, 0u, 1u, f6_s));
      ps.emit(pkt_680_3c0(0x8000ULL));
      ps.emit(pkt_900_bar_all());
      ps.emit(pkt_680_3f9());
      ps.emit(pkt_900_zero());
      ps.emit(pkt_680_3f9());
      ps.emit(pkt_680_3c0(0ULL));
    } else if (n_groups == 1u && K > 32u) {
      if ((K % 32u) != 0u) {
        ps.emit(pkt_680(k_mask, 0x0001003000000040ULL, 0x09108000ULL));
        ps.emit(pkt_900_mask(k_mask));
      }
      ps.emit(build_post3_3c0(M, 32u, k4, sw_unit ? (sw_unit * 128u) : 0u, sw_unit != 0u));
      uint32_t ch1_tok = 0u;
      uint32_t ch2_tok = 0u;
      for (uint32_t kt = 1u; kt < k_tiles; ++kt) {
        uint32_t k_sl = std::min<uint32_t>(32u, k4 - 32u * kt);
        bool use_ch2 = ((kt % 2u) == 1u);
        uint64_t ch_hi = 0ULL;
        uint32_t op_recv = 0x440u;
        uint64_t bar_hi = 0ULL, bar_lo = 0ULL, f6_r = 0ULL, f6_s = 0ULL;
        if (use_ch2) {
          ch_hi = ((4ULL * ch2_tok + 1ULL) << 12) | 0x480ULL;
          ch2_tok += 1u;
          sync.c3_ch2 += 2u;
          op_recv = 0x480u;
          bar_hi = 0x0001002000000040ULL;
          bar_lo = 0x20000000ULL;
          f6_r = 0x20000000ULL;
          f6_s = 0x200000000ULL;
        } else {
          ch_hi = ((4ULL * ch1_tok + 1ULL) << 12) | 0x440ULL;
          ch1_tok += 1u;
          sync.c3_440_ch1 += 1u;
          op_recv = 0x440u;
          bar_hi = 0x0001001000000040ULL;
          bar_lo = 0x10000000ULL;
          f6_r = 0x00400000ULL;
          f6_s = 0x04000000ULL;
        }
        uint32_t dst_base = sw_unit ? 0u : ((M == 1u) ? 128u : (((M * std::min<uint32_t>(32u, k4) + 127u) / 128u) * 128u));
        uint32_t dst_off = dst_base + kt * 32u;
        ps.emit(build_post3_440_recv(1u, M, k_sl, dst_off, op_recv, k4, 0ULL, f6_r));
        ps.emit(build_post3_400_gather(kt, 1u, M, k_sl, ch_hi, sw_unit ? (sw_unit * 128u) : 0u, f6_s));
        ps.emit(pkt_680(1u, bar_hi, bar_lo));
      }
      ps.emit(pkt_680_3c0(0x8000ULL));
      ps.emit(pkt_900_bar_all());
      ps.emit(pkt_680_3f9());
      ps.emit(pkt_900_zero());
      ps.emit(pkt_680_3f9());
      ps.emit(pkt_680_3c0(0ULL));
    } else {
      if (K == 64u && N == 64u) {
        sync.c3_44c = 2u;
        sync.c3_dst_cnt = 2u;
        ps.emit(pkt_680(n_mask, 0x0001003000000040ULL, 0x0D108000ULL));
        ps.emit(pkt_900_mask(n_mask));
        uint32_t dst_sram = (M == 1u) ? 128u : std::max<uint32_t>(256u, next_pow2_u32(M * 32u));
        ps.emit(build_post3_440_k64n64(n_mask, M, dst_sram));
        ps.emit(build_post3_400_bcast(1u, n_mask, M * 32u, 0x144CULL, 0u, 2u, 0x200000000ULL));
        ps.emit(build_post3_400_bcast(2u, n_mask, M * 32u, 0x544CULL, 0u, 2u, 0x200000000ULL));
        ps.emit(pkt_680_3c0(0ULL));
      } else if (K <= 48u) {
        sync.c3_44c = 1u;
        sync.c3_48c = 1u;
        sync.c3_dst_cnt = n_groups;
        ps.emit(pkt_680(k_mask, 0x0001003000000040ULL, 0x09108000ULL));
        ps.emit(pkt_900_mask(k_mask));
        ps.emit(pkt_680(n_mask, 0x0001003000000040ULL, 0x0D108000ULL));
        ps.emit(pkt_900_mask(n_mask));
        ps.emit(pkt_680(n_mask, 0x0001003000000040ULL, 0x0F108000ULL));
        ps.emit(pkt_900_mask(n_mask));
        uint32_t dst_sram = (M == 1u) ? 128u : std::max<uint32_t>(256u, next_pow2_u32(M * 32u));
        ps.emit(build_post3_440_recv(n_mask, M, 32u, dst_sram, 0x440u, k4, 0xC0000000ULL, 0x10000000ULL));
        ps.emit(build_post3_400_bcast(1u, n_mask, M * 32u, 0x144CULL, 0u, 1u, 0x100000000ULL));
        uint32_t k_sl1 = k4 - 32u;
        ps.emit(build_post3_440_recv(n_mask, M, k_sl1, dst_sram + 32u, 0x480u, k4, 0xC0000000ULL, 0x20000000ULL));
        ps.emit(build_post3_400_bcast(2u, n_mask, M * k_sl1, 0x148CULL, 0u, 1u, 0x200000000ULL));
        ps.emit(pkt_680_3c0(0x8000ULL));
        ps.emit(pkt_900_bar_all());
        ps.emit(pkt_680_3f9());
        ps.emit(pkt_900_zero());
        ps.emit(pkt_680_3f9());
        ps.emit(pkt_680_3c0(0ULL));
      } else {
        if ((K % 32u) != 0u) {
          ps.emit(pkt_680(k_mask, 0x0001003000000040ULL, 0x09108000ULL));
          ps.emit(pkt_900_mask(k_mask));
        }
        ps.emit(pkt_680(n_mask, 0x0001003000000040ULL, 0x0D108000ULL));
        ps.emit(pkt_900_mask(n_mask));
        ps.emit(pkt_680(n_mask, 0x0001003000000040ULL, 0x0F108000ULL));
        ps.emit(pkt_900_mask(n_mask));
        uint32_t ch1_tok = 0u;
        uint32_t ch2_tok = 0u;
        for (uint32_t kt = 0u; kt < k_tiles; ++kt) {
          uint32_t k_sl = std::min<uint32_t>(32u, k4 - 32u * kt);
          bool use_ch2 = ((kt % 2u) == 1u);
          uint64_t ch_hi = 0ULL;
          uint32_t op_recv = 0x440u;
          uint64_t bar_hi = 0ULL, bar_lo = 0ULL, f6_r = 0ULL, f6_s = 0ULL;
          if (use_ch2) {
            ch_hi = ((4ULL * ch2_tok + 1ULL) << 12) | 0x480ULL;
            ch2_tok += 1u;
            sync.c3_ch2 += 2u;
            op_recv = 0x480u;
            bar_hi = 0x0001002000000040ULL;
            bar_lo = 0x20000000ULL;
            f6_r = 0x20000000ULL;
            f6_s = 0x200000000ULL;
          } else {
            ch_hi = ((4ULL * ch1_tok + 1ULL) << 12) | 0x440ULL;
            ch1_tok += 1u;
            sync.c3_440_ch1 += 1u;
            op_recv = 0x440u;
            bar_hi = 0x0001001000000040ULL;
            bar_lo = 0x10000000ULL;
            f6_r = 0x10000000ULL;
            f6_s = 0x100000000ULL;
          }
          uint32_t dst_off = kt * 32u;
          ps.emit(build_post3_440_recv(n_mask, M, k_sl, dst_off, op_recv, k4, 0xC0000000ULL, f6_r));
          ps.emit(build_post3_400_bcast(1u << kt, n_mask, M * k_sl, ch_hi, sw_unit * 128u, 1u, f6_s));
          uint32_t bar_mask = (kt == 0u) ? 1u : (1u << (n_groups - 1u));
          ps.emit(pkt_680(bar_mask, bar_hi, bar_lo));
        }
        ps.emit(pkt_680_3c0(0x8000ULL));
        ps.emit(pkt_900_bar_all());
        ps.emit(pkt_680_3f9());
        ps.emit(pkt_900_zero());
        ps.emit(pkt_680_3f9());
        ps.emit(pkt_680_3c0(0ULL));
      }
    }
  }

  // Param SRAM OP_440 (one per N-group)
  uint32_t n_blk = (N > 32u) ? 32u : std::max<uint32_t>(8u, ((N + 7u) / 8u) * 8u);
  uint64_t p_field = static_cast<uint64_t>((n_blk / 8u) * 0x400u - 0x20u);
  uint32_t k_rem12 = k_unit % 12u;
  uint64_t p440_f0h = 0ULL;
  Flit p440_f1{0ULL, 0ULL};
  uint64_t p440_f2l = 0ULL;
  uint64_t p440_f2h = 0ULL;
  if (K <= 4u) {
    p440_f0h = 0x0000000200000000ULL;
    p440_f1 = {0ULL, 0ULL};
    p440_f2l = 0x0000000000006220ULL;
    p440_f2h = has_bias ? 0x0320000000031400ULL : 0ULL;
  } else if (m_ge_48) {
    uint64_t ym1 = static_cast<uint64_t>(k_unit - 1u);
    p440_f0h = (ym1 << 54) | 0x0000000200000000ULL;
    uint64_t f1l = (mask_s(-8LL * static_cast<int64_t>(ym1), 16u) << 48) |
                   ((0x80000000ULL - ym1 * 0x400ULL) & 0xFFFFFFFFULL);
    uint64_t f1h = ((0x200000ULL - ym1) << 28) | 0xFFULL;
    p440_f1 = {f1l, f1h};
    p440_f2l = (static_cast<uint64_t>(k_unit / 2u) << 12) | 0x220ULL;
    p440_f2h = has_bias ? (((static_cast<uint64_t>(k_unit) * 0x20ULL + 0x20ULL) << 48) |
                           ((static_cast<uint64_t>(k_unit) * 0x20ULL + 0x0CULL) << 8))
                        : 0ULL;
  } else {
    p440_f0h = 0x02C0000200000000ULL | ((M == 32u) ? 0x40000ULL : 0ULL);
    uint64_t f1l = 0xFFA800007FFFD400ULL + (static_cast<uint64_t>(k_passes - 1u) << 31);
    p440_f1 = {f1l, 0x0001FFFF500000FFULL};
    if (k_rem12 == 0u) {
      p440_f2l = 0x0000000000006220ULL;
    } else {
      p440_f2l = ((static_cast<uint64_t>((12u - k_rem12) * 0x40u + 0x10u)) << 48) |
                 (static_cast<uint64_t>(k_rem12 - 1u) << 32) | 0x6220ULL;
    }
    if (has_bias) {
      uint64_t m_off = 0x320ULL + ((M >= 3u) ? (static_cast<uint64_t>(M) * 0x20ULL) : 0ULL);
      uint64_t sub_k = (K == 48u) ? 0x14ULL : 0x0CULL;
      p440_f2h = (m_off << 48) | ((m_off - sub_k) << 8);
    } else {
      p440_f2h = 0ULL;
    }
  }
  uint64_t p440_f3l = ((p_field + 2ULL) << 48) | (has_bias ? ((p_field << 24) | 0x200ULL) : 0ULL);
  uint64_t p440_f3h = 0xFFFFFD84C0000000ULL | (p_field << 4);
  Flit p440_f4 = m_ge_48 ? Flit{0xFFFFFF419000000FULL, 0xFFFFFFD06C000003ULL}
                         : Flit{0x9000000FULL, 0ULL};
  Flit p440_f5 = m_ge_48 ? Flit{0x3FFFFFF41D000000ULL, 0x0000000003C00000ULL}
                         : Flit{0ULL, 0ULL};
  uint64_t p440_f6h = (n_groups > 1u || m_ge_48) ? 0x40400000ULL : 0x60400000ULL;
  for (uint32_t g = 0; g < n_groups; ++g) {
    ps.emit({
        {(static_cast<uint64_t>(1u << g) << 12) | 0x440ULL, p440_f0h},
        p440_f1,
        {p440_f2l, p440_f2h},
        {p440_f3l, p440_f3h},
        p440_f4,
        p440_f5,
        {0ULL, p440_f6h},
        {0x0007800000000000ULL, 0ULL},
    });
  }

  // OP_500 (bias loader, omitted when !has_bias)
  if (has_bias) {
    Pkt f500(std::begin(kBaseOp500), std::end(kBaseOp500));
    f500[0].lo = (static_cast<uint64_t>(n_mask) << 12) | 0x500ULL;
    if (m_ge_48) {
      f500[0].hi = 0x80000000ULL | (static_cast<uint64_t>(n_blk) * 0x800ULL);
      f500[8].lo = 0x0F09000000000000ULL;
      if (n_blk <= 16u) {
        f500[2].hi = 0x00001A0001800010ULL;
        f500[3] = {0xFFE80001FFFF4000ULL, 0xA00007FFFD00003FULL};
      }
    } else {
      f500[0].hi = 0x80030000ULL + ((M >= 3u) ? (static_cast<uint64_t>(M) * 0x2000ULL) : 0ULL);
    }
    f500[5].hi = 0xD402000000000000ULL | (static_cast<uint64_t>(k_passes) << 24) |
                 ((K == 64u && N > 32u) ? 1ULL : 0ULL);
    ps.emit(f500);
  }

  if (m_ge_48) {
    for (const Pkt& p6 : build_op6c0_pkts(M, k4, n_mask, 0u, true, M * std::min<uint32_t>(32u, k4))) {
      ps.emit(p6);
    }
  }

  // CSR_040[0] (+ optional CSR_040[1] when N > 32 and N % 32 != 0)
  uint32_t n_tile0 = std::min<uint32_t>(32u, n4);
  CsrBank tgt0 = build_compute_csr_state(M, K, n_tile0, s_in, zp_in, s_w, s_out, zp_out,
                                         act, has_bias, sw_unit, h3_csrs, N);
  std::vector<std::pair<uint32_t, uint32_t>> diff0;
  for (uint32_t r = 0; r < 0x62u; ++r) {
    if (tgt0[r] != h3_vica_csrs[r]) diff0.push_back({r, tgt0[r]});
  }
  uint32_t n_full_n = std::max<uint32_t>(1u, N / 32u);
  ps.emit(pack_csr_packet(0x040u, (1u << n_full_n) - 1u, diff0, m_ge_48));
  if (N > 32u && (N % 32u) != 0u) {
    uint32_t n_rem = pad4(N % 32u);
    CsrBank tgt1 = build_compute_csr_state(M, K, n_rem, s_in, zp_in, s_w, s_out, zp_out,
                                           act, has_bias, sw_unit, h3_csrs, N);
    std::vector<std::pair<uint32_t, uint32_t>> diff1;
    for (uint32_t r = 0; r < 0x62u; ++r) {
      if (tgt1[r] != tgt0[r]) diff1.push_back({r, tgt1[r]});
    }
    ps.emit(pack_csr_packet(0x040u, 1u << n_full_n, diff1, false));
  }

  if (m_ge_48) {
    ps.emit(pkt_940_params(ext2_bytes));
  }

  if (out_n_pkts) *out_n_pkts = ps.n_pkts;
  if (out_sync) *out_sync = sync;
  return flits_to_bytes(ps.flits);
}

// ---------------------------------------------------------------------------
// 7. Build chunk[5]
// ---------------------------------------------------------------------------

struct C5SyncInfo {
  uint32_t c5_ch1 = 0;
  uint32_t c5_ch2 = 0;
};

bool is_2d_scatter(uint32_t nb) {
  return nb != 4u && nb != 8u && nb != 16u && nb != 32u;
}

std::pair<uint32_t, uint32_t> split_2d(uint32_t nb) {
  if ((nb % 32u) == 0u && nb > 32u) return {32u, nb / 32u};
  if ((nb % 3u) == 0u) return {nb / 3u, 3u};
  return {nb, 1u};
}

Pkt build_c5_3c0(uint32_t mask, uint32_t src_off, uint32_t dst_off, uint32_t nb) {
  uint64_t bm1_4 = static_cast<uint64_t>(nb - 1u) * 4ULL;
  uint64_t f0h = (static_cast<uint64_t>(nb) * 2ULL << 32) | (static_cast<uint64_t>(src_off) << 13);
  uint64_t dst_enc = static_cast<uint64_t>(dst_off) * 0x40ULL;
  uint64_t f2h = ((dst_enc & 0xFFFFULL) << 48) | (bm1_4 << 32) | (bm1_4 << 12);
  uint64_t f3l = (static_cast<uint64_t>(nb) * 0x400ULL) | (dst_enc >> 16);
  uint64_t f4h = (bm1_4 * 2ULL) << 52;
  uint64_t f5l = (bm1_4 * 2ULL) << 8;
  return {
      {(static_cast<uint64_t>(mask) << 12) | 0x3C0ULL, f0h},
      {0ULL, 0ULL},
      {0x20ULL, f2h},
      {f3l, 0ULL},
      {0x0000400000000000ULL, f4h},
      {f5l, 0ULL},
      {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL},
      {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL}, {0ULL, 0ULL},
  };
}

Pkt build_c5_scatter_recv(uint32_t mask, uint32_t dst_sram, uint32_t nb, uint32_t op = 0x440u) {
  uint64_t f6h = (op == 0x480u) ? 0x20000000ULL : 0x00400000ULL;
  if (!is_2d_scatter(nb)) {
    uint64_t b_enc = static_cast<uint64_t>(nb - 1u) * 2ULL;
    return {
        {(static_cast<uint64_t>(mask) << 12) | op,
         0x0000000200000000ULL | (static_cast<uint64_t>(dst_sram) << 13)},
        {0ULL, 0ULL},
        {0ULL, 0ULL},
        {b_enc << 52, b_enc << 8},
        {0ULL, 0ULL},
        {0ULL, 0ULL},
        {0ULL, f6h},
        {0x0000800000000000ULL, 0ULL},
    };
  }
  auto [inner, y] = split_2d(nb);
  uint32_t ym1 = y - 1u;
  uint64_t b_enc = static_cast<uint64_t>(inner - 1u) * 2ULL;
  uint64_t f0h = (static_cast<uint64_t>(ym1) << 54) |
                 ((static_cast<uint64_t>(inner) * 2ULL) << 32) |
                 (static_cast<uint64_t>(dst_sram) << 13);
  Flit f1 = dma_2d_rewind_f1(inner * ym1);
  return {
      {(static_cast<uint64_t>(mask) << 12) | op, f0h},
      f1,
      {0x20ULL, 0ULL},
      {(b_enc << 52) | (0x2ULL << 48), b_enc << 8},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, f6h},
      {0x0000800000000000ULL, 0ULL},
  };
}

Pkt build_c5_scatter_400(uint32_t src_mask, uint32_t dst_phys, uint32_t src_off,
                         uint32_t nb, uint64_t ch_val, bool ch2) {
  uint64_t f6h = (1ULL << (34u + dst_phys)) | (ch2 ? 0x200000000ULL : 0x04000000ULL);
  if (!is_2d_scatter(nb)) {
    uint64_t b_enc = static_cast<uint64_t>(nb - 1u) * 2ULL;
    return {
        {(static_cast<uint64_t>(src_mask) << 12) | 0x400ULL,
         0x0000000200000000ULL | (static_cast<uint64_t>(src_off) << 13)},
        {0ULL, 0ULL},
        {0ULL, 0ULL},
        {b_enc << 52, (ch_val << 32) | (b_enc << 8)},
        {0x0000000100000080ULL, 0ULL},
        {0ULL, 0ULL},
        {0ULL, f6h},
    };
  }
  auto [inner, y] = split_2d(nb);
  uint32_t ym1 = y - 1u;
  uint64_t b_enc = static_cast<uint64_t>(inner - 1u) * 2ULL;
  uint64_t f0h = (static_cast<uint64_t>(ym1) << 54) |
                 ((static_cast<uint64_t>(inner) * 2ULL) << 32) |
                 (static_cast<uint64_t>(src_off) << 13);
  Flit f1 = dma_2d_rewind_f1(inner * ym1);
  return {
      {(static_cast<uint64_t>(src_mask) << 12) | 0x400ULL, f0h},
      f1,
      {0x20ULL, 0ULL},
      {b_enc << 52, (ch_val << 32) | (b_enc << 8)},
      {0x0000000100000080ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, f6h},
  };
}

std::vector<uint8_t> build_chunk5(uint32_t c5_pid0, uint32_t M, uint32_t K, uint32_t N,
                                  float s_out, int32_t zp_out, uint32_t ext2_bytes,
                                  const C3SyncInfo& c3_sync,
                                  uint32_t* out_n_pkts, C5SyncInfo* out_sync) {
  PacketStream ps(c5_pid0);
  uint32_t n4 = pad4(N);
  uint32_t n_groups = (N + 31u) / 32u;
  uint32_t grp_bytes = ext2_bytes / n_groups;
  bool stream_w = is_weight_streaming(K, N);
  uint32_t sw_unit = compute_sw_unit(M, K, N);
  bool m_ge_48 = (M >= 48u);

  // A. pre5
  if (m_ge_48) {
    ps.emit(pkt_980_params(0u, 1u, grp_bytes, true));
    ps.emit(pkt_680_3f9());
  } else {
    ps.emit(pkt_940_params(ext2_bytes));
    for (uint32_t g = 0; g < n_groups; ++g) {
      ps.emit(pkt_980_params(g, n_groups, grp_bytes, false));
    }
    ps.emit(pkt_680_3f9());
    ps.emit(pkt_900_zero());
    ps.emit(pkt_680_3f9(0x8000ULL));
    ps.emit(pkt_900_bar_all());
    ps.emit(pkt_680_3f9());
  }

  // B. mid5
  bool zero_f0_c5 = stream_w || m_ge_48;
  ps.emit(build_op5c0(M, n4, zero_f0_c5));
  ps.emit(build_op4c0(M, n4, zero_f0_c5));
  ps.emit(pkt_680_3e0());
  ps.emit(build_op640(N, s_out, zp_out));
  uint32_t c5_6c0_base = m_ge_48 ? (M * std::min<uint32_t>(32u, n4)) : 0u;
  uint32_t n_mask_c5 = (1u << n_groups) - 1u;
  for (const Pkt& p6 : build_op6c0_pkts(M, n4, n_mask_c5, c5_6c0_base)) {
    ps.emit(p6);
  }
  Pkt hdr5 = build_hdr_data(M, N, s_out, zp_out, sw_unit);
  ps.emit(hdr5);
  CsrBank h5_csrs = unpack_hdr_csrs(hdr5);
  if (N > 32u && (N % 32u) != 0u) {
    ps.emit(build_vica_csr_pkt(M, N, sw_unit, h5_csrs, zp_out, nullptr));
  }

  // C. post5
  C5SyncInfo sync{};
  if (M == 1u) {
    ps.emit(pkt_680_3f9());
    ps.emit(pkt_680_3c0(0x8000ULL));
    ps.emit(pkt_900_bar_all());
  } else {
    ps.emit(pkt_680_3f9());
    ps.emit(pkt_680_3f9());
    ps.emit(pkt_680_3c0(0x8000ULL));
    ps.emit(pkt_900_bar_all());

    std::vector<TileSpec> tile_specs = compute_output_tiles(M, N);
    struct TileRange {
      uint32_t tid;
      uint32_t tb0;
      uint32_t tb1;
    };
    std::vector<TileRange> tile_ranges;
    uint32_t pos = 0u;
    for (const TileSpec& ts : tile_specs) {
      tile_ranges.push_back({ts.tid, pos, pos + ts.nbytes});
      pos += ts.nbytes;
    }

    uint32_t sram_dst_base = (stream_w || m_ge_48) ? 0u : (M * std::min<uint32_t>(32u, n4));
    uint32_t sram_src_base = stream_w ? (sw_unit * 128u)
                                      : (m_ge_48 ? (M * std::min<uint32_t>(32u, n4)) : 0u);

    if (n_groups > 1u) {
      uint32_t n_mask = (1u << n_groups) - 1u;
      uint64_t f1l_sync = (K == 64u && N == 64u) ? 0x09108000ULL : 0x0B108000ULL;
      ps.emit(pkt_680(n_mask, 0x0001003000000040ULL, f1l_sync));
      ps.emit(pkt_900_mask(n_mask));
    }

    uint32_t ch1_tok = c3_sync.c3_440_ch1 + c3_sync.c3_44c;
    for (uint32_t g = 0; g < n_groups; ++g) {
      uint32_t w_g = std::min<uint32_t>(32u, n4 - 32u * g);
      struct Slice {
        uint32_t tid;
        uint32_t dst_in_tile;
        uint32_t src_in_grp;
        uint32_t nbytes;
      };
      std::vector<Slice> slices;
      for (const TileRange& tr : tile_ranges) {
        if (M > 16u) {
          slices.push_back({tr.tid, 0u, tr.tb0, tr.tb1 - tr.tb0});
        } else {
          uint32_t row = tr.tb0 / n4;
          uint32_t c0 = tr.tb0 % n4;
          uint32_t c1 = c0 + (tr.tb1 - tr.tb0);
          uint32_t g0 = 32u * g;
          uint32_t g1 = g0 + w_g;
          uint32_t ov0 = std::max(c0, g0);
          uint32_t ov1 = std::min(c1, g1);
          if (ov0 < ov1) {
            slices.push_back({tr.tid, ov0 - c0, row * w_g + (ov0 - g0), ov1 - ov0});
          }
        }
      }

      bool has_local = false;
      Slice local_slice{};
      std::vector<Slice> remote_slices;
      for (const Slice& sl : slices) {
        if (!has_local && sl.tid == g) {
          has_local = true;
          local_slice = sl;
        } else {
          remote_slices.push_back(sl);
        }
      }

      if (has_local) {
        if (g == 1u) {
          for (const Slice& sl : remote_slices) {
            if (sl.tid == 0u) {
              ps.emit(build_c5_scatter_recv(1u, sram_dst_base + sl.dst_in_tile, sl.nbytes, 0x480u));
              break;
            }
          }
        }
        ps.emit(build_c5_3c0(1u << g, sram_src_base + local_slice.src_in_grp,
                             sram_dst_base + local_slice.dst_in_tile, local_slice.nbytes));
      }

      uint32_t op_recv = (g == 0u) ? 0x440u : 0x480u;
      struct RecvGrp {
        uint32_t dst_in_tile;
        uint32_t nbytes;
        uint32_t bitmask;
      };
      std::vector<RecvGrp> recv_groups;
      for (const Slice& rs : remote_slices) {
        if (g == 1u && has_local && rs.tid == 0u) continue;
        bool found = false;
        for (RecvGrp& rg : recv_groups) {
          if (rg.dst_in_tile == rs.dst_in_tile && rg.nbytes == rs.nbytes) {
            rg.bitmask |= (1u << rs.tid);
            found = true;
            break;
          }
        }
        if (!found) {
          recv_groups.push_back({rs.dst_in_tile, rs.nbytes, 1u << rs.tid});
        }
      }
      for (const RecvGrp& rg : recv_groups) {
        ps.emit(build_c5_scatter_recv(rg.bitmask, sram_dst_base + rg.dst_in_tile, rg.nbytes, op_recv));
      }

      uint64_t ch_val = (g == 0u)
                            ? (((4ULL * ch1_tok + 1ULL) << 12) | 0x440ULL)
                            : (((4ULL * (c3_sync.c3_ch2 / 2u + c3_sync.c3_48c) + 1ULL) << 12) | 0x480ULL);
      for (const Slice& rs : remote_slices) {
        ps.emit(build_c5_scatter_400(1u << g, kSnake16[rs.tid],
                                     sram_src_base + rs.src_in_grp, rs.nbytes, ch_val, g > 0u));
        if (g == 0u) {
          sync.c5_ch1 += 1u;
        } else {
          sync.c5_ch2 += 2u;
        }
      }

      uint32_t all_rmask = 0u;
      for (const Slice& rs : remote_slices) all_rmask |= (1u << rs.tid);
      if (g == 0u) {
        ps.emit(pkt_680(all_rmask, 0x0001001000000040ULL, 0x10000000ULL));
      } else {
        ps.emit(pkt_680(all_rmask, 0x0001002000000040ULL, 0x20000000ULL));
      }
    }
    ps.emit(pkt_680_3c0(0ULL));
  }

  if (out_n_pkts) *out_n_pkts = ps.n_pkts;
  if (out_sync) *out_sync = sync;
  return flits_to_bytes(ps.flits);
}

// ---------------------------------------------------------------------------
// 8. Build chunk[6]
// ---------------------------------------------------------------------------

std::vector<uint8_t> build_chunk6(uint32_t c6_pid0, uint32_t M, uint32_t K, uint32_t N,
                                  const C3SyncInfo& c3_sync, const C5SyncInfo& c5_sync) {
  PacketStream ps(c6_pid0);
  uint32_t n4 = pad4(N);
  uint32_t n_groups = (N + 31u) / 32u;
  bool stream_w = is_weight_streaming(K, N);
  uint32_t sw_unit = compute_sw_unit(M, K, N);
  bool m_ge_48 = (M >= 48u);

  // 1. OP_a40 (output host DMA)
  uint32_t out_bytes = M * n4;
  uint64_t dma_enc = static_cast<uint64_t>(out_bytes - 1u) * 8ULL;
  uint64_t f3_lo = (dma_enc << 48) | (1ULL << 47);
  uint64_t f3_hi = (dma_enc >> 16) | (dma_enc << 16);
  ps.emit({
      {0x0000004300000A40ULL, 0x0000002000000000ULL},
      {0ULL, 0ULL},
      {0ULL, 0x0000020000000000ULL},
      {f3_lo, f3_hi},
  });

  // 2. OP_9c0
  uint32_t B = M * n4;
  uint32_t n32 = (B + 31u) / 32u;
  uint32_t first_c = std::min<uint32_t>(32u, B);
  uint32_t last_c = ((B - 1u) % 32u) + 1u;
  uint64_t f4l_hi = (n32 == 1u) ? 0x00040000ULL : 0ULL;
  uint64_t f4h = (n32 == 1u) ? 0ULL : (static_cast<uint64_t>(n32 - 1u) * 0x20ULL);
  uint64_t f6l = (static_cast<uint64_t>(first_c - 1u) << 55) |
                 (static_cast<uint64_t>(last_c - 1u) << 50) |
                 ((n32 > 1u) ? (1ULL << 45) : 0ULL);
  uint64_t f7l = (static_cast<uint64_t>(B / 2u) << 16) | 0x4040ULL;
  ps.emit({
      {(f4l_hi << 32) | 0x9C0ULL, f4h},
      {0ULL, 0ULL},
      {f6l, 0ULL},
      {f7l, 0ULL},
  });

  // 3. b7: coalesced OP_400 output tiles
  std::vector<TileSpec> tile_specs = compute_output_tiles(M, N);
  uint32_t n5 = static_cast<uint32_t>(tile_specs.size());
  uint32_t c3_440 = c3_sync.c3_440_ch1;
  uint32_t n_44c = c3_sync.c3_44c;
  uint32_t n_dst = c3_sync.c3_dst_cnt;
  uint32_t c5_ch1 = c5_sync.c5_ch1;
  uint32_t c2_tot = c3_sync.c3_ch2 + c5_sync.c5_ch2;

  uint32_t sram_off = 0u;
  if (M == 1u) {
    sram_off = sw_unit * 128u;
  } else if (stream_w || m_ge_48) {
    sram_off = 0u;
  } else {
    sram_off = M * std::min<uint32_t>(32u, n4);
  }

  uint64_t f4l_b7 = 0x0000000100000000ULL | (0x80ULL * ((M == 1u) ? n_groups : 1u));

  struct GatherGrp {
    uint32_t tok;
    uint32_t nb;
    uint32_t bitmask;
  };
  std::vector<GatherGrp> groups;
  for (uint32_t seq_idx = 0; seq_idx < n5; ++seq_idx) {
    uint32_t tid = tile_specs[seq_idx].tid;
    uint32_t nb = tile_specs[seq_idx].nbytes;
    uint32_t tok = c3_440 + c5_ch1 + seq_idx + ((tid < n_dst) ? n_44c : 0u);
    bool found = false;
    for (GatherGrp& g : groups) {
      if (g.tok == tok && g.nb == nb) {
        g.bitmask |= (1u << tid);
        found = true;
        break;
      }
    }
    if (!found) {
      groups.push_back({tok, nb, 1u << tid});
    }
  }

  for (const GatherGrp& g : groups) {
    uint64_t ch_hi = ((4ULL * g.tok + 1ULL) << 12) | 0x44CULL;
    if (g.nb == 48u) {
      ps.emit({
          {(static_cast<uint64_t>(g.bitmask) << 12) | 0x400ULL,
           0x0040004000000000ULL | (static_cast<uint64_t>(sram_off) << 13)},
          {0xFF0000007FFF8000ULL, 0x0001FFFE000000FFULL},
          {0x20ULL, 0ULL},
          {0x01E2000000000000ULL, (ch_hi << 32) | 0x00003E00ULL},
          {f4l_b7, 0ULL},
          {0ULL, 0ULL},
          {0ULL, 0x0004000004000000ULL},
      });
    } else {
      uint64_t b_enc = static_cast<uint64_t>(g.nb - 1u) * 2ULL;
      ps.emit({
          {(static_cast<uint64_t>(g.bitmask) << 12) | 0x400ULL,
           0x0000000200000000ULL | (static_cast<uint64_t>(sram_off) << 13)},
          {0ULL, 0ULL},
          {0ULL, 0ULL},
          {b_enc << 52, (ch_hi << 32) | (b_enc << 8)},
          {f4l_b7, 0ULL},
          {0ULL, 0ULL},
          {0ULL, 0x0004000004000000ULL},
      });
    }
  }

  // 4. b5: n5 OP_900 sync increments
  uint32_t b5_0 = c3_440 + c5_ch1 + 1u;
  for (uint32_t k = 0; k < n5; ++k) {
    uint64_t w2 = static_cast<uint64_t>(b5_0 + 2u * k) << 8;
    ps.emit({
        {0x0000000000000900ULL, 0ULL},
        {0ULL, 0ULL},
        {w2, 0ULL},
        {0ULL, 0ULL},
        {0x0200000000000000ULL, 0x400ULL},
    });
  }

  // 5. tail[0..25]
  uint32_t c1_tot = b5_0 + 2u * n5 - 1u;
  bool has_ch2 = (c2_tot > 0u) || (K > 32u && N > 32u && K != 64u);
  uint64_t ch2_byte = has_ch2 ? 0x06ULL : 0x02ULL;
  ps.emit(pkt_680_3f9());
  ps.emit({
      {(ch2_byte << 56) | 0x0000FFFF00000900ULL, 0ULL},
      {0ULL, 0ULL},
      {(static_cast<uint64_t>(c2_tot) << 32) | (static_cast<uint64_t>(c1_tot) << 8), 0ULL},
      {0ULL, 0ULL},
      {ch2_byte << 56, 0x180ULL},
  });
  ps.emit(pkt_680(0xFFFFu, 0x000FFFF8000003C0ULL | ch2_byte, 0x8000ULL));
  ps.emit(pkt_900_bar_all());
  ps.emit(pkt_680_3c0(0x8000ULL));
  ps.emit(pkt_900_bar_all());
  ps.emit({
      {0x000FFFFF40000900ULL, 0xFF80000000000000ULL},
      {0x7FFULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0ULL},
      {0ULL, 0x008003E0ULL},
  }, false);

  return flits_to_bytes(ps.flits);
}

// ---------------------------------------------------------------------------
// 9. Clean-room Bottom-Up FlatBuffer Builder (DIVE / DGD0 / MultiExecutable / DWS0 / DGC0)
// ---------------------------------------------------------------------------

class FbBuilder {
 public:
  FbBuilder() { buf_.reserve(16384u); }

  uint32_t pos() const { return static_cast<uint32_t>(buf_.size()); }

  void pad(size_t n) {
    buf_.insert(buf_.end(), n, 0u);
  }

  void prep(size_t align, size_t additional_bytes = 0u) {
    if (align > minalign_) minalign_ = align;
    size_t align_size = (~(buf_.size() + additional_bytes) + 1u) & (align - 1u);
    pad(align_size);
  }

  void push_bytes_rev(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = len; i > 0u; --i) {
      buf_.push_back(p[i - 1u]);
    }
  }

  void push_u8(uint8_t v) {
    prep(1u);
    buf_.push_back(v);
  }

  void push_u16(uint16_t v) {
    prep(2u);
    push_bytes_rev(&v, sizeof(v));
  }

  void push_u32(uint32_t v) {
    prep(4u);
    push_bytes_rev(&v, sizeof(v));
  }

  void push_i32(int32_t v) {
    prep(4u);
    push_bytes_rev(&v, sizeof(v));
  }

  void push_u64(uint64_t v) {
    prep(8u);
    push_bytes_rev(&v, sizeof(v));
  }

  void push_i64(int64_t v) {
    prep(8u);
    push_bytes_rev(&v, sizeof(v));
  }

  void push_uoffset(uint32_t target_off) {
    prep(4u);
    uint32_t rel = pos() - target_off + 4u;
    push_bytes_rev(&rel, sizeof(rel));
  }

  uint32_t create_string(const char* s) {
    size_t len = s ? std::strlen(s) : 0u;
    prep(4u, len + 1u);
    push_u8(0u);
    if (len > 0u) push_bytes_rev(s, len);
    push_u32(static_cast<uint32_t>(len));
    return pos();
  }

  uint32_t create_vector_bytes(const uint8_t* data, size_t len, size_t align = 1u) {
    size_t eff_align = std::max<size_t>(4u, align);
    prep(eff_align, len);
    if (len > 0u && data) push_bytes_rev(data, len);
    push_u32(static_cast<uint32_t>(len));
    return pos();
  }

  uint32_t create_vector_i32(const int32_t* vals, size_t count) {
    prep(4u, count * 4u);
    for (size_t i = count; i > 0u; --i) push_i32(vals[i - 1u]);
    push_u32(static_cast<uint32_t>(count));
    return pos();
  }

  uint32_t create_vector_u32(const uint32_t* vals, size_t count) {
    prep(4u, count * 4u);
    for (size_t i = count; i > 0u; --i) push_u32(vals[i - 1u]);
    push_u32(static_cast<uint32_t>(count));
    return pos();
  }

  uint32_t create_vector_i64(const int64_t* vals, size_t count) {
    prep(4u, count * 8u);
    if (count > 0u) prep(8u, count * 8u);
    for (size_t i = count; i > 0u; --i) push_i64(vals[i - 1u]);
    push_u32(static_cast<uint32_t>(count));
    return pos();
  }

  uint32_t create_vector_uoffsets(const uint32_t* offs, size_t count) {
    prep(4u, count * 4u);
    for (size_t i = count; i > 0u; --i) push_uoffset(offs[i - 1u]);
    push_u32(static_cast<uint32_t>(count));
    return pos();
  }

  void start_table() {
    cur_fields_.clear();
    table_start_ = pos();
  }

  void add_u8(uint16_t slot, uint8_t v, uint8_t def = 0u, bool force = false) {
    if (v != def || force) {
      push_u8(v);
      cur_fields_.push_back({slot, pos()});
    }
  }

  void add_u32(uint16_t slot, uint32_t v, uint32_t def = 0u, bool force = false) {
    if (v != def || force) {
      push_u32(v);
      cur_fields_.push_back({slot, pos()});
    }
  }

  void add_i32(uint16_t slot, int32_t v, int32_t def = 0, bool force = false) {
    if (v != def || force) {
      push_i32(v);
      cur_fields_.push_back({slot, pos()});
    }
  }

  void add_u64(uint16_t slot, uint64_t v, uint64_t def = 0ULL, bool force = false,
               uint32_t* out_from_end = nullptr) {
    if (v != def || force) {
      push_u64(v);
      uint32_t p = pos();
      cur_fields_.push_back({slot, p});
      if (out_from_end) *out_from_end = p;
    }
  }

  void add_uoffset(uint16_t slot, uint32_t off) {
    if (off != 0u) {
      push_uoffset(off);
      cur_fields_.push_back({slot, pos()});
    }
  }

  uint32_t end_table() {
    push_i32(0);  // placeholder for soffset
    uint32_t table_off = pos();
    int32_t max_slot = -1;
    for (const auto& f : cur_fields_) {
      if (static_cast<int32_t>(f.first) > max_slot) {
        max_slot = static_cast<int32_t>(f.first);
      }
    }
    size_t nslots = (max_slot >= 0) ? static_cast<size_t>(max_slot + 1) : 0u;
    uint16_t obj_size = static_cast<uint16_t>(table_off - table_start_);
    uint16_t vt_size = static_cast<uint16_t>(4u + 2u * nslots);
    std::vector<uint8_t> vt_bytes(vt_size, 0u);
    std::memcpy(vt_bytes.data(), &vt_size, 2u);
    std::memcpy(vt_bytes.data() + 2u, &obj_size, 2u);
    for (const auto& f : cur_fields_) {
      uint16_t rel = static_cast<uint16_t>(table_off - f.second);
      std::memcpy(vt_bytes.data() + 4u + static_cast<size_t>(f.first) * 2u, &rel, 2u);
    }

    int32_t soff = 0;
    bool found = false;
    for (const auto& vt : vtables_) {
      if (vt.second == vt_bytes) {
        soff = static_cast<int32_t>(vt.first) - static_cast<int32_t>(table_off);
        found = true;
        break;
      }
    }
    if (!found) {
      prep(2u, vt_bytes.size());
      push_bytes_rev(vt_bytes.data(), vt_bytes.size());
      uint32_t vt_off = pos();
      vtables_.push_back({vt_off, std::move(vt_bytes)});
      soff = static_cast<int32_t>(vt_off) - static_cast<int32_t>(table_off);
    }
    const uint8_t* sp = reinterpret_cast<const uint8_t*>(&soff);
    buf_[table_off - 4u] = sp[3];
    buf_[table_off - 3u] = sp[2];
    buf_[table_off - 2u] = sp[1];
    buf_[table_off - 1u] = sp[0];
    return table_off;
  }

  std::vector<uint8_t> finish(uint32_t root_off, const char* file_id_4 = nullptr) {
    if (file_id_4) {
      prep(minalign_, 8u);
      push_bytes_rev(file_id_4, 4u);
      push_uoffset(root_off);
    } else {
      prep(minalign_, 4u);
      push_uoffset(root_off);
    }
    std::vector<uint8_t> out(buf_.size());
    for (size_t i = 0; i < buf_.size(); ++i) {
      out[i] = buf_[buf_.size() - 1u - i];
    }
    return out;
  }

 private:
  std::vector<uint8_t> buf_;
  size_t minalign_ = 1u;
  std::vector<std::pair<uint32_t, std::vector<uint8_t>>> vtables_;
  std::vector<std::pair<uint16_t, uint32_t>> cur_fields_;
  uint32_t table_start_ = 0u;
};

// Build bit-exact DIVE FlatBuffer (ext[0]) cleanly from scratch (zero raw templates).
std::vector<uint8_t> build_dive_ext0_clean(const std::array<std::vector<uint8_t>, 7>& chunks,
                                           uint32_t M, uint32_t K, uint32_t N) {
  FbBuilder fb;

  struct ValSpec {
    int64_t dims[2];
    size_t ndims;
    uint8_t dt;
    const char* name;
  };
  const ValSpec val_specs[14] = {
      {{-2, 0}, 1u, 2u, "parameter_0"},
      {{-2, 0}, 1u, 2u, "scratchpad"},
      {{static_cast<int64_t>(M), static_cast<int64_t>(K)}, 2u, 1u, "input_0"},
      {{0, 0}, 0u, 11u, "main"},
      {{0, 0}, 0u, 11u, "main.1"},
      {{0, 0}, 0u, 11u, "dive_vm_tensor_address_of.1"},
      {{static_cast<int64_t>(M), static_cast<int64_t>(N)}, 2u, 1u, "output_0"},
      {{0, 0}, 0u, 11u, "dive_vm_tensor_address_of.3"},
      {{0, 0}, 0u, 11u, "variable_4097"},
      {{0, 0}, 0u, 11u, "variable_4099"},
      {{0, 0}, 0u, 11u, "variable_1"},
      {{0, 0}, 0u, 11u, "variable_5"},
      {{0, 0}, 0u, 11u, "variable_5.1"},
      {{0, 0}, 0u, 11u, "variable_5.2"},
  };

  uint32_t val_offs[14]{};
  for (size_t i = 0; i < 14u; ++i) {
    uint32_t d_off = fb.create_vector_i64(val_specs[i].dims, val_specs[i].ndims);
    uint32_t n_off = fb.create_string(val_specs[i].name);
    fb.start_table();
    fb.add_i32(3u, -1, 0);
    fb.add_uoffset(2u, n_off);
    fb.add_uoffset(0u, d_off);
    fb.add_u8(1u, val_specs[i].dt, 0u);
    val_offs[i] = fb.end_table();
  }

  const int32_t fn_in_idx[1] = {2};
  const int32_t fn_out_idx[1] = {6};
  uint32_t fn_vals = fb.create_vector_uoffsets(val_offs, 14u);
  uint32_t fn_ins = fb.create_vector_i32(fn_in_idx, 1u);
  uint32_t fn_outs = fb.create_vector_i32(fn_out_idx, 1u);

  enum DetKind { DET_NONE = 0, DET_DISP = 1, DET_PUT = 2 };
  struct OpSpec {
    uint8_t ot;
    DetKind det;
    uint32_t a;
    uint64_t b;
    uint32_t c;
    int32_t in_val;
    bool has_in;
    int32_t out_val;
    bool has_out;
    const char* name;
  };
  const OpSpec op_specs[24] = {
      {59u, DET_NONE, 0u, 0ULL, 0u, 0, true, 3, true, "main"},
      {59u, DET_NONE, 0u, 0ULL, 0u, 1, true, 4, true, "main"},
      {61u, DET_NONE, 0u, 0ULL, 0u, 2, true, 0, false, "dive_vm_tensor_cache_clean_invalidate"},
      {59u, DET_NONE, 0u, 0ULL, 0u, 2, true, 5, true, "dive_vm_tensor_address_of"},
      {59u, DET_NONE, 0u, 0ULL, 0u, 6, true, 7, true, "dive_vm_tensor_address_of.2"},
      {61u, DET_NONE, 0u, 0ULL, 0u, 6, true, 0, false, "dive_vm_tensor_cache_clean_invalidate.1"},
      {47u, DET_DISP, 0u, 0ULL, 0u, 0, false, 0, false, "program_prolog_hw_instructions_0"},
      {25u, DET_NONE, 0u, 0ULL, 0u, 5, true, 8, true, "_tpu_input_from_input_0::0"},
      {25u, DET_NONE, 0u, 0ULL, 0u, 7, true, 9, true, "_tpu_output_from_output_0::0"},
      {25u, DET_NONE, 0u, 0ULL, 0u, 3, true, 10, true, "parameter_0"},
      {25u, DET_NONE, 0u, 0ULL, 0u, 8, true, 11, true, "4097->5"},
      {47u, DET_DISP, 1u, 1ULL, 0u, 0, false, 0, false, "before_flexible_dma__tpu_input_from_input_0_hw_instructions_1"},
      {50u, DET_PUT, 2u, 37ULL, 64u, 11, true, 0, false, "_tpu_input_from_input_0"},
      {47u, DET_DISP, 2u, 2ULL, 0u, 0, false, 0, false, "after_request_flexible_dma__tpu_input_from_input_0_hw_instructions_2"},
      {25u, DET_NONE, 0u, 0ULL, 0u, 10, true, 12, true, "1->5"},
      {47u, DET_DISP, 3u, 3ULL, 0u, 0, false, 0, false, "before_flexible_dma_output_0::parameter_hw_instructions_3"},
      {50u, DET_PUT, 4u, 37ULL, 64u, 12, true, 0, false, "output_0::parameter"},
      {47u, DET_DISP, 4u, 4ULL, 0u, 0, false, 0, false, "after_request_flexible_dma_output_0::parameter_hw_instructions_4"},
      {25u, DET_NONE, 0u, 0ULL, 0u, 9, true, 13, true, "4099->5"},
      {47u, DET_DISP, 5u, 5ULL, 0u, 0, false, 0, false, "before_flexible_dma__tpu_output_from_output_0_hw_instructions_5"},
      {50u, DET_PUT, 6u, 37ULL, 64u, 13, true, 0, false, "_tpu_output_from_output_0"},
      {47u, DET_DISP, 6u, 6ULL, 0u, 0, false, 0, false, "after_request_flexible_dma__tpu_output_from_output_0_hw_instructions_6;finish_and_sleep_hw_instructions_7"},
      {51u, DET_NONE, 0u, 0ULL, 0u, 0, false, 0, false, "halt"},
      {61u, DET_NONE, 0u, 0ULL, 0u, 6, true, 0, false, "dive_vm_tensor_cache_clean_invalidate.2"},
  };

  uint32_t op_offs[24]{};
  for (size_t i = 0; i < 24u; ++i) {
    const OpSpec& op = op_specs[i];
    fb.start_table();
    if (op.det == DET_DISP) {
      fb.add_u32(1u, static_cast<uint32_t>(op.b), 0u);
      fb.add_u32(0u, op.a, 0u);
    } else if (op.det == DET_PUT) {
      fb.add_u64(1u, op.b, 0ULL);
      fb.add_u32(2u, op.c, 0u);
      fb.add_u32(0u, op.a, 0u);
    }
    uint32_t det_off = fb.end_table();
    uint32_t in_off = fb.create_vector_i32(&op.in_val, op.has_in ? 1u : 0u);
    uint32_t out_off = fb.create_vector_i32(&op.out_val, op.has_out ? 1u : 0u);
    uint32_t nm_off = fb.create_string(op.name);
    fb.start_table();
    fb.add_uoffset(4u, nm_off);
    fb.add_uoffset(3u, out_off);
    fb.add_uoffset(2u, in_off);
    fb.add_uoffset(1u, det_off);
    fb.add_u8(0u, op.ot, 0u);
    op_offs[i] = fb.end_table();
  }

  uint32_t fn_ops = fb.create_vector_uoffsets(op_offs, 24u);
  uint32_t fn_nm = fb.create_string("main");
  fb.start_table();
  fb.add_uoffset(4u, fn_nm);
  fb.add_uoffset(3u, fn_ops);
  fb.add_uoffset(2u, fn_outs);
  fb.add_uoffset(1u, fn_ins);
  fb.add_uoffset(0u, fn_vals);
  uint32_t fn_off = fb.end_table();

  uint32_t funcs_vec = fb.create_vector_uoffsets(&fn_off, 1u);
  uint32_t mod_name = fb.create_string("");

  uint32_t chunk_offs[7]{};
  for (size_t i = 0; i < 7u; ++i) {
    uint32_t b_off = fb.create_vector_bytes(chunks[i].data(), chunks[i].size(), 32u);
    fb.start_table();
    fb.add_uoffset(0u, b_off);
    chunk_offs[i] = fb.end_table();
  }

  uint32_t chunks_vec = fb.create_vector_uoffsets(chunk_offs, 7u);
  const int32_t port_in = 2;
  const int32_t port_out = 6;
  const int32_t port_param = 0;
  const int32_t port_scr = 1;
  uint32_t p_in = fb.create_vector_i32(&port_in, 1u);
  uint32_t p_out = fb.create_vector_i32(&port_out, 1u);
  uint32_t p_param = fb.create_vector_i32(&port_param, 1u);
  uint32_t p_scr = fb.create_vector_i32(&port_scr, 1u);
  uint32_t p_var = fb.create_vector_i32(nullptr, 0u);

  fb.start_table();
  fb.add_uoffset(8u, p_var);
  fb.add_uoffset(7u, p_scr);
  fb.add_uoffset(6u, p_param);
  fb.add_uoffset(5u, p_out);
  fb.add_uoffset(4u, p_in);
  fb.add_uoffset(3u, chunks_vec);
  fb.add_uoffset(2u, mod_name);
  fb.add_uoffset(1u, funcs_vec);
  uint32_t root = fb.end_table();
  return fb.finish(root, "DIVE");
}

std::vector<uint8_t> build_dgd0_clean() {
  FbBuilder fb;
  uint32_t empty_tensors = fb.create_vector_uoffsets(nullptr, 0u);
  uint32_t empty_children = fb.create_vector_uoffsets(nullptr, 0u);
  fb.start_table();
  fb.add_uoffset(1u, empty_children);
  fb.add_uoffset(0u, empty_tensors);
  uint32_t root = fb.end_table();
  return fb.finish(root, "DGD0");
}

std::vector<uint8_t> build_multi_executable_clean(uint32_t M, uint32_t K, uint32_t N,
                                                  uint32_t ext2_size,
                                                  uint32_t exec_field_14,
                                                  uint64_t exec_hash) {
  FbBuilder fb;
  auto make_layer = [&](uint64_t sz, const char* name) -> uint32_t {
    uint32_t nm = fb.create_string(name);
    fb.start_table();
    fb.add_u64(3u, sz, 0ULL, true);
    fb.add_uoffset(2u, nm);
    fb.add_u32(1u, static_cast<uint32_t>(sz), 0u, true);
    return fb.end_table();
  };

  uint32_t in_layer = make_layer(static_cast<uint64_t>(M) * K, "input_0");
  uint32_t out_layer = make_layer(static_cast<uint64_t>(M) * N, "output_0");
  uint32_t param_layer = make_layer(ext2_size, "main_STAND_ALONE_params_Group_0_0_2_0_0");

  uint32_t ins_vec = fb.create_vector_uoffsets(&in_layer, 1u);
  uint32_t outs_vec = fb.create_vector_uoffsets(&out_layer, 1u);
  uint32_t params_vec = fb.create_vector_uoffsets(&param_layer, 1u);
  uint32_t empty_meta = fb.create_vector_bytes(nullptr, 0u);
  uint32_t empty_bb = fb.create_vector_uoffsets(nullptr, 0u);
  uint32_t empty_dma = fb.create_vector_uoffsets(nullptr, 0u);
  uint32_t arch_str = fb.create_string("buenos_a0");
  uint32_t name_str = fb.create_string("main_STAND_ALONE_cluster2");

  fb.start_table();
  fb.add_uoffset(32u, empty_dma);
  fb.add_u64(29u, 81920ULL, 0ULL, true);
  fb.add_u64(28u, 32768ULL, 0ULL, true);
  fb.add_uoffset(25u, empty_bb);
  fb.add_u32(24u, 2u, 0u, true);
  fb.add_u32(23u, 311296u, 0u, true);
  fb.add_u64(22u, exec_hash, 0ULL, true);
  fb.add_uoffset(20u, params_vec);
  fb.add_u8(18u, 1u, 0u, true);
  fb.add_u8(17u, 1u, 0u, true);
  fb.add_u32(14u, exec_field_14, 0u, true);
  fb.add_u32(13u, 16384u, 0u, true);
  fb.add_u32(12u, 262144u, 0u, true);
  fb.add_u32(11u, 32768u, 0u, true);
  fb.add_u32(8u, 2u, 0u, true);
  fb.add_uoffset(4u, outs_vec);
  fb.add_uoffset(3u, ins_vec);
  fb.add_uoffset(2u, empty_meta);
  fb.add_uoffset(1u, arch_str);
  fb.add_uoffset(0u, name_str);
  uint32_t exec0 = fb.end_table();

  uint32_t execs_vec = fb.create_vector_uoffsets(&exec0, 1u);
  fb.start_table();
  fb.add_uoffset(1u, execs_vec);
  uint32_t root = fb.end_table();
  return fb.finish(root, nullptr);
}

std::vector<uint8_t> build_dws0_clean(uint32_t M, uint32_t K, uint32_t N,
                                      uint32_t ext0_size, uint32_t ext1_size, uint32_t ext2_size,
                                      uint32_t scratch_sz, uint32_t exec_field_14,
                                      uint64_t exec_hash, const char* token_str) {
  std::vector<uint8_t> mexec =
      build_multi_executable_clean(M, K, N, ext2_size, exec_field_14, exec_hash);

  FbBuilder fb;
  auto make_tref = [&](uint32_t idx) -> uint32_t {
    fb.start_table();
    fb.add_u32(0u, idx, 0u);
    return fb.end_table();
  };

  // 6 Tensors in SubGraph[0]
  auto make_tensor = [&](const uint32_t* dims, size_t ndims, uint8_t dt,
                         const char* name, uint8_t storage,
                         bool is_ext, uint64_t ext_idx) -> uint32_t {
    uint32_t d_off = fb.create_vector_u32(dims, ndims);
    uint32_t n_off = fb.create_string(name);
    fb.start_table();
    if (is_ext && ext_idx != 0ULL) fb.add_u64(13u, ext_idx, 0ULL);
    if (is_ext) fb.add_u8(12u, 1u, 0u);
    fb.add_u8(5u, storage, 0u);
    fb.add_i32(3u, -1, 0);
    fb.add_uoffset(2u, n_off);
    fb.add_u8(1u, dt, 0u);
    fb.add_uoffset(0u, d_off);
    return fb.end_table();
  };

  char scratch_name[48]{};
  std::snprintf(scratch_name, sizeof(scratch_name), "dive_scratch_%u", scratch_sz);

  uint32_t in_dims[2] = {M, K};
  uint32_t out_dims[2] = {M, N};
  uint32_t ext0_dims[1] = {ext0_size};
  uint32_t ext1_dims[1] = {ext1_size};
  uint32_t ext2_dims[1] = {ext2_size};
  uint32_t scr_dims[1] = {scratch_sz};

  uint32_t t_offs[6]{};
  t_offs[0] = make_tensor(in_dims, 2u, 1u, "input_0", 1u, false, 0ULL);
  t_offs[1] = make_tensor(out_dims, 2u, 1u, "output_0", 1u, false, 0ULL);
  t_offs[2] = make_tensor(ext0_dims, 1u, 2u, "main_STAND_ALONE_dive_vm_ir_Group_0_0_2_0", 5u, true, 0ULL);
  t_offs[3] = make_tensor(ext1_dims, 1u, 2u, "main_STAND_ALONE_interpreter_Group_0_0_2_0", 5u, true, 1ULL);
  t_offs[4] = make_tensor(ext2_dims, 1u, 2u, "main_STAND_ALONE_params_Group_0_0_2_0_0", 5u, true, 2ULL);
  t_offs[5] = make_tensor(scr_dims, 1u, 2u, scratch_name, 3u, false, 0ULL);

  uint32_t tensors_vec = fb.create_vector_uoffsets(t_offs, 6u);
  uint32_t sg_in_ref = make_tref(0u);
  uint32_t sg_out_ref = make_tref(1u);
  uint32_t sg_ins = fb.create_vector_uoffsets(&sg_in_ref, 1u);
  uint32_t sg_outs = fb.create_vector_uoffsets(&sg_out_ref, 1u);

  // ExecutableBinding
  uint32_t b_res_refs[2] = {make_tref(2u), make_tref(3u)};
  uint32_t b_res_vec = fb.create_vector_uoffsets(b_res_refs, 2u);
  uint32_t b_param_ref = make_tref(4u);
  uint32_t b_param_vec = fb.create_vector_uoffsets(&b_param_ref, 1u);
  uint32_t b_scr_ref = make_tref(5u);
  uint32_t b_aux_ref = make_tref(0u);
  uint32_t b_interp_ref = make_tref(3u);
  uint32_t b_ir_ref = make_tref(2u);
  fb.start_table();
  fb.add_uoffset(8u, b_ir_ref);
  fb.add_uoffset(7u, b_interp_ref);
  fb.add_uoffset(6u, b_aux_ref);
  fb.add_uoffset(5u, b_scr_ref);
  fb.add_uoffset(4u, b_param_vec);
  fb.add_uoffset(0u, b_res_vec);
  uint32_t binding0 = fb.end_table();
  uint32_t bindings_vec = fb.create_vector_uoffsets(&binding0, 1u);

  // OpExecutableCall (detail_type=2)
  uint32_t me_vec = fb.create_vector_bytes(mexec.data(), mexec.size());
  uint32_t empty_b3 = fb.create_vector_bytes(nullptr, 0u);
  uint32_t empty_b4 = fb.create_vector_bytes(nullptr, 0u);
  fb.start_table();
  fb.add_uoffset(4u, empty_b4);
  fb.add_uoffset(3u, empty_b3);
  fb.add_uoffset(1u, bindings_vec);
  fb.add_uoffset(0u, me_vec);
  uint32_t op0_detail = fb.end_table();

  uint32_t op0_in_ref = make_tref(0u);
  uint32_t op0_out_ref = make_tref(1u);
  uint32_t op0_ins = fb.create_vector_uoffsets(&op0_in_ref, 1u);
  uint32_t op0_outs = fb.create_vector_uoffsets(&op0_out_ref, 1u);
  uint32_t op0_nm = fb.create_string("main");
  fb.start_table();
  fb.add_uoffset(5u, op0_nm);
  fb.add_uoffset(4u, op0_outs);
  fb.add_uoffset(3u, op0_ins);
  fb.add_uoffset(1u, op0_detail);
  fb.add_u8(0u, 2u, 0u);
  uint32_t op0 = fb.end_table();

  // OpTerminator (detail_type=7)
  fb.start_table();
  uint32_t op1_detail = fb.end_table();
  uint32_t op1_ins = fb.create_vector_uoffsets(nullptr, 0u);
  uint32_t op1_outs = fb.create_vector_uoffsets(nullptr, 0u);
  uint32_t op1_nm = fb.create_string("main.1");
  fb.start_table();
  fb.add_uoffset(5u, op1_nm);
  fb.add_uoffset(4u, op1_outs);
  fb.add_uoffset(3u, op1_ins);
  fb.add_uoffset(1u, op1_detail);
  fb.add_u8(0u, 7u, 0u);
  uint32_t op1 = fb.end_table();

  uint32_t ops_arr[2] = {op0, op1};
  uint32_t ops_vec = fb.create_vector_uoffsets(ops_arr, 2u);
  uint32_t sg_nm = fb.create_string("main");
  fb.start_table();
  fb.add_uoffset(4u, sg_nm);
  fb.add_uoffset(3u, ops_vec);
  fb.add_uoffset(2u, sg_outs);
  fb.add_uoffset(1u, sg_ins);
  fb.add_uoffset(0u, tensors_vec);
  uint32_t sg0 = fb.end_table();
  uint32_t sgs_vec = fb.create_vector_uoffsets(&sg0, 1u);

  // PackageHeaderV1
  uint32_t hdr_arch = fb.create_string("buenos_a0");
  fb.start_table();
  fb.add_uoffset(0u, hdr_arch);
  uint32_t pkg_hdr = fb.end_table();

  uint32_t model_id = fb.create_string(token_str);
  uint32_t empty_emb = fb.create_vector_uoffsets(nullptr, 0u);
  uint32_t empty_exec_refs = fb.create_vector_uoffsets(nullptr, 0u);
  uint32_t empty_s = fb.create_string("");
  uint32_t f11_vec = fb.create_vector_uoffsets(&empty_s, 1u);
  uint32_t f12_s = fb.create_string("");
  uint32_t f14_s = fb.create_string("");
  uint32_t f17_s = fb.create_string("");
  uint32_t rel_s = fb.create_string("lga_tpu_20260603.0.0");

  fb.start_table();
  fb.add_uoffset(18u, rel_s);
  fb.add_uoffset(17u, f17_s);
  fb.add_uoffset(14u, f14_s);
  fb.add_u8(13u, 2u, 0u);
  fb.add_uoffset(12u, f12_s);
  fb.add_uoffset(11u, f11_vec);
  fb.add_u64(10u, kCompilerCl, 0ULL);
  fb.add_u8(9u, 1u, 0u);
  fb.add_uoffset(8u, empty_exec_refs);
  fb.add_u32(6u, 1u, 0u);
  fb.add_uoffset(5u, empty_emb);
  fb.add_uoffset(4u, model_id);
  fb.add_uoffset(3u, sgs_vec);
  fb.add_uoffset(2u, pkg_hdr);
  fb.add_u8(1u, 1u, 0u);
  fb.add_u32(0u, 6u, 0u);
  uint32_t root = fb.end_table();
  return fb.finish(root, "DWS0");
}

bool read_entire_file(const char* path, std::vector<uint8_t>* out, std::string* err) {
  out->clear();
  if (!path || !path[0]) {
    if (err) *err = "Empty file path";
    return false;
  }
  int fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (err) *err = std::string("open(") + path + ") failed errno=" + std::to_string(errno);
    return false;
  }
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    if (err) *err = std::string("fstat(") + path + ") failed or empty";
    return false;
  }
  out->resize(static_cast<size_t>(st.st_size));
  size_t got = 0u;
  while (got < out->size()) {
    ssize_t n = ::read(fd, out->data() + got, out->size() - got);
    if (n < 0) {
      if (errno == EINTR) continue;
      int e = errno;
      ::close(fd);
      out->clear();
      if (err) *err = std::string("read(") + path + ") failed errno=" + std::to_string(e);
      return false;
    }
    if (n == 0) break;
    got += static_cast<size_t>(n);
  }
  ::close(fd);
  out->resize(got);
  return got > 0u;
}

bool validate_gemm_shape(const ob_gemm_desc_t& d, std::string* err) {
  if (d.m_dim == 0u || d.k_dim == 0u || d.n_dim == 0u) {
    if (err) *err = "GEMM dimensions must be non-zero";
    return false;
  }
  if (d.m_dim > 64u || d.k_dim > 512u || d.n_dim > 128u) {
    if (err) *err = "GEMM tile exceeds buenos_a0 single-tile bounds (M<=64, K<=512, N<=128)";
    return false;
  }
  if (d.zp_w != 0) {
    if (err) *err = "buenos_a0 FC ISA requires symmetric weights (zp_w == 0)";
    return false;
  }
  if (d.s_in <= 0.0f || d.s_w <= 0.0f || d.s_out <= 0.0f) {
    if (err) *err = "Quantization scales must be strictly positive";
    return false;
  }
  if (d.activation != kActNone && d.activation != kActRelu && d.activation != kActRelu6) {
    if (err) *err = "Unsupported fused activation enum";
    return false;
  }
  if (d.m_dim > 16u) {
    if (d.n_dim > 32u) {
      if (err) *err = "M > 16 requires N <= 32 (n_groups == 1)";
      return false;
    }
    if (((d.m_dim * pad4(d.n_dim)) % 16u) != 0u) {
      if (err) *err = "M > 16 requires (M * pad4(N)) % 16 == 0";
      return false;
    }
  }
  if (d.m_dim >= 48u && d.k_dim > 32u) {
    if (err) *err = "M >= 48 requires K <= 32";
    return false;
  }
  if (d.m_dim >= 2u && d.n_dim > 64u) {
    if (err) *err = "M >= 2 requires N <= 64 (n_groups <= 2)";
    return false;
  }
  return true;
}

}  // namespace

ob_status_t synthesize_gemm_parts(const ob_gemm_desc_t& desc,
                                  GemmSynthResult* out,
                                  std::string* err) {
  if (!out) return OB_ERR_INVALID_ARG;
  if (!validate_gemm_shape(desc, err)) return OB_ERR_UNSUPPORTED;

  const uint32_t M = desc.m_dim;
  const uint32_t K = desc.k_dim;
  const uint32_t N = desc.n_dim;

  bool include_bias = false;
  if (desc.bias_n != nullptr) {
    for (uint32_t i = 0; i < N; ++i) {
      if (desc.bias_n[i] != 0) {
        include_bias = true;
        break;
      }
    }
  }

  if (desc.weights_nk != nullptr) {
    out->ext2_params = pack_ext2(N, K, desc.weights_nk, desc.bias_n, include_bias);
  } else if (desc.packed_params != nullptr && desc.packed_params_bytes > 0u) {
    out->ext2_params.assign(desc.packed_params, desc.packed_params + desc.packed_params_bytes);
    // Infer bias presence from size when only pre-packed params are provided
    uint32_t k4 = pad4(K);
    uint32_t n_blk = (N <= 32u) ? std::max<uint32_t>(8u, ((N + 7u) / 8u) * 8u) : 32u;
    uint32_t n_groups = (N <= 32u) ? 1u : ((N + 31u) / 32u);
    size_t no_bias_sz = static_cast<size_t>(k4) * n_blk * n_groups;
    include_bias = (desc.packed_params_bytes > no_bias_sz);
  } else {
    out->ext2_params = pack_ext2(N, K, nullptr, desc.bias_n, include_bias);
  }

  const uint32_t ext2_bytes = static_cast<uint32_t>(out->ext2_params.size());
  out->chunks[0] = build_chunk0();
  out->chunks[1] = build_chunk1(M, K);
  out->chunks[2] = build_chunk2(M, K);

  uint32_t n_pk3 = 0u;
  C3SyncInfo c3_sync{};
  out->chunks[3] = build_chunk3(M, K, N, desc.s_in, desc.zp_in, desc.s_w,
                                desc.s_out, desc.zp_out, desc.activation,
                                include_bias, ext2_bytes, &n_pk3, &c3_sync);

  uint32_t c4_pid = 10u + n_pk3;
  out->chunks[4] = build_chunk4(c4_pid, ext2_bytes);

  uint32_t n_pk5 = 0u;
  C5SyncInfo c5_sync{};
  out->chunks[5] = build_chunk5(c4_pid + 1u, M, K, N, desc.s_out, desc.zp_out,
                                ext2_bytes, c3_sync, &n_pk5, &c5_sync);

  out->chunks[6] = build_chunk6(c4_pid + 1u + n_pk5, M, K, N, c3_sync, c5_sync);

  out->ext0_dive = build_dive_ext0_clean(out->chunks, M, K, N);
  out->exec_field_14 = compute_exec_field14(M, K, N);
  out->dive_scratch_size = compute_dive_scratch_size(M, N);
  return OB_OK;
}

ob_status_t build_dgc0_from_parts(const ob_gemm_desc_t& desc,
                                  const GemmSynthResult& parts,
                                  const uint8_t* dive_interp,
                                  size_t dive_interp_size,
                                  std::vector<uint8_t>* out_dgc0,
                                  std::string* err) {
  if (!dive_interp || dive_interp_size == 0u || !out_dgc0) {
    if (err) *err = "Missing external RV64 DIVE interpreter image (ext[1])";
    return OB_ERR_INVALID_ARG;
  }

  const uint32_t M = desc.m_dim;
  const uint32_t K = desc.k_dim;
  const uint32_t N = desc.n_dim;

  // Deterministic executable_hash & 19-digit Token_ identifier from SHA-256(M,K,N,ext0,ext2)
  std::vector<uint8_t> hash_in;
  hash_in.reserve(12u + parts.ext0_dive.size() + parts.ext2_params.size());
  uint32_t dims[3] = {M, K, N};
  const uint8_t* dp = reinterpret_cast<const uint8_t*>(dims);
  hash_in.insert(hash_in.end(), dp, dp + sizeof(dims));
  hash_in.insert(hash_in.end(), parts.ext0_dive.begin(), parts.ext0_dive.end());
  hash_in.insert(hash_in.end(), parts.ext2_params.begin(), parts.ext2_params.end());
  uint8_t digest[32]{};
  sha256_compute(hash_in.data(), hash_in.size(), digest);

  uint64_t exec_hash = 0ULL;
  uint64_t model_hash = 0ULL;
  uint64_t tok_raw = 0ULL;
  std::memcpy(&exec_hash, digest, 8u);
  std::memcpy(&tok_raw, digest + 8u, 8u);
  std::memcpy(&model_hash, digest + 16u, 8u);

  constexpr uint64_t kPow10_18 = 1000000000000000000ULL;
  constexpr uint64_t kPow10_19 = 10000000000000000000ULL;
  uint64_t tok_num = tok_raw % kPow10_19;
  if (tok_num < kPow10_18) tok_num += kPow10_18;
  char token_str[32]{};
  std::snprintf(token_str, sizeof(token_str), "Token_%019llu",
                static_cast<unsigned long long>(tok_num));

  std::vector<uint8_t> dgd0 = build_dgd0_clean();
  std::vector<uint8_t> dws0 = build_dws0_clean(
      M, K, N,
      static_cast<uint32_t>(parts.ext0_dive.size()),
      static_cast<uint32_t>(dive_interp_size),
      static_cast<uint32_t>(parts.ext2_params.size()),
      parts.dive_scratch_size,
      parts.exec_field_14,
      exec_hash,
      token_str);

  // Build DGC0 FlatBuffer header (DGD0 4096B aligned, DWS0 64B aligned)
  FbBuilder fb;
  uint32_t dws_vec = fb.create_vector_bytes(dws0.data(), dws0.size(), 64u);
  uint32_t dgd_vec = fb.create_vector_bytes(dgd0.data(), dgd0.size(), 4096u);

  uint32_t eb_offs[3]{};
  uint32_t eb_off_from_end[3]{};
  const uint64_t eb_sizes[3] = {
      static_cast<uint64_t>(parts.ext0_dive.size()),
      static_cast<uint64_t>(dive_interp_size),
      static_cast<uint64_t>(parts.ext2_params.size()),
  };
  for (size_t i = 0; i < 3u; ++i) {
    fb.start_table();
    fb.add_u64(1u, eb_sizes[i], 0ULL, true);
    fb.add_u64(0u, 0ULL, 0ULL, true, &eb_off_from_end[i]);
    eb_offs[i] = fb.end_table();
  }
  uint32_t ebs_vec = fb.create_vector_uoffsets(eb_offs, 3u);

  fb.start_table();
  fb.add_uoffset(7u, ebs_vec);
  fb.add_u32(6u, 64u, 0u, true);
  fb.add_u64(5u, kCompilerCl, 0ULL, true);
  fb.add_u64(4u, model_hash, 0ULL, true);
  fb.add_u32(3u, 6u, 0u, true);
  fb.add_uoffset(1u, dgd_vec);
  fb.add_uoffset(0u, dws_vec);
  uint32_t root = fb.end_table();

  std::vector<uint8_t> hdr = fb.finish(root, "DGC0");
  const uint64_t ext0_off = align_up_u64(static_cast<uint64_t>(hdr.size()), 64ULL);
  const uint64_t ext1_off = align_up_u64(ext0_off + eb_sizes[0], 64ULL);
  const uint64_t ext2_off = align_up_u64(ext1_off + eb_sizes[1], 64ULL);
  const uint64_t total_sz = align_up_u64(ext2_off + eb_sizes[2], 64ULL);

  const uint64_t abs_offs[3] = {ext0_off, ext1_off, ext2_off};
  for (size_t i = 0; i < 3u; ++i) {
    size_t pos = hdr.size() - static_cast<size_t>(eb_off_from_end[i]);
    std::memcpy(hdr.data() + pos, &abs_offs[i], 8u);
  }

  out_dgc0->clear();
  out_dgc0->reserve(static_cast<size_t>(total_sz));
  out_dgc0->insert(out_dgc0->end(), hdr.begin(), hdr.end());

  auto append_aligned = [&](uint64_t target_off, const uint8_t* data, size_t len) {
    if (target_off > out_dgc0->size()) {
      out_dgc0->insert(out_dgc0->end(), static_cast<size_t>(target_off - out_dgc0->size()), 0x0Au);
    }
    if (len > 0u) {
      out_dgc0->insert(out_dgc0->end(), data, data + len);
    }
  };

  append_aligned(ext0_off, parts.ext0_dive.data(), parts.ext0_dive.size());
  append_aligned(ext1_off, dive_interp, dive_interp_size);
  append_aligned(ext2_off, parts.ext2_params.data(), parts.ext2_params.size());
  if (total_sz > out_dgc0->size()) {
    out_dgc0->insert(out_dgc0->end(), static_cast<size_t>(total_sz - out_dgc0->size()), 0x0Au);
  }
  return OB_OK;
}

std::vector<uint8_t> pack_ext2_for_tile(uint32_t N, uint32_t K,
                                        const int8_t* weights_nk,
                                        const int32_t* bias_n) {
  bool include_bias = false;
  if (bias_n != nullptr) {
    for (uint32_t i = 0; i < N; ++i) {
      if (bias_n[i] != 0) {
        include_bias = true;
        break;
      }
    }
  }
  return pack_ext2(N, K, weights_nk, bias_n, include_bias);
}

ob_status_t build_dgc0_dyn_weights(const ob_gemm_desc_t& desc,
                                   const uint8_t* dive_interp,
                                   size_t dive_interp_size,
                                   std::vector<uint8_t>* out_dgc0,
                                   uint32_t* out_ext2_bytes,
                                   std::string* err) {
  if (!dive_interp || dive_interp_size == 0u || !out_dgc0) {
    if (err) *err = "Missing external RV64 DIVE interpreter image (ext[1])";
    return OB_ERR_INVALID_ARG;
  }

  ob_gemm_desc_t synth_desc = desc;
  synth_desc.weights_nk = nullptr;
  synth_desc.packed_params = nullptr;
  synth_desc.packed_params_bytes = 0u;
  GemmSynthResult parts{};
  ob_status_t st = synthesize_gemm_parts(synth_desc, &parts, err);
  if (st != OB_OK) return st;

  const uint32_t M = desc.m_dim;
  const uint32_t K = desc.k_dim;
  const uint32_t N = desc.n_dim;
  const uint32_t ext2_bytes = static_cast<uint32_t>(parts.ext2_params.size());
  if (out_ext2_bytes) *out_ext2_bytes = ext2_bytes;

  // Retarget chunk[4] 0xa40 f0_hi_word from 0x22 (param) to 0x21 (input_1).
  if (parts.chunks[4].size() >= 8u) {
    uint64_t f0_lo = 0ULL;
    std::memcpy(&f0_lo, parts.chunks[4].data(), 8u);
    f0_lo = (f0_lo & ~(0xFFFFFFFFULL << 32)) | (0x00000021ULL << 32);
    std::memcpy(parts.chunks[4].data(), &f0_lo, 8u);
  }

  // Build 2-input 0-param DIVE FlatBuffer (matching add.dgc0 layout: input_0=[M,K], input_1=[ext2_bytes]).
  std::vector<uint8_t> ext0_dive;
  {
    FbBuilder fb;
    struct ValSpec { int64_t dims[2]; size_t ndims; uint8_t dt; const char* name; };
    const ValSpec val_specs[14] = {
        {{-2, 0}, 1u, 2u, "scratchpad"},
        {{static_cast<int64_t>(M), static_cast<int64_t>(K)}, 2u, 1u, "input_0"},
        {{static_cast<int64_t>(ext2_bytes), 0}, 1u, 1u, "input_1"},
        {{0, 0}, 0u, 11u, "main"},
        {{0, 0}, 0u, 11u, "dive_vm_tensor_address_of.1"},
        {{0, 0}, 0u, 11u, "dive_vm_tensor_address_of.3"},
        {{static_cast<int64_t>(M), static_cast<int64_t>(N)}, 2u, 1u, "output_0"},
        {{0, 0}, 0u, 11u, "dive_vm_tensor_address_of.5"},
        {{0, 0}, 0u, 11u, "variable_4097"},
        {{0, 0}, 0u, 11u, "variable_4099"},
        {{0, 0}, 0u, 11u, "variable_4101"},
        {{0, 0}, 0u, 11u, "variable_5"},
        {{0, 0}, 0u, 11u, "variable_5.1"},
        {{0, 0}, 0u, 11u, "variable_5.2"},
    };
    uint32_t val_offs[14]{};
    for (size_t i = 0; i < 14u; ++i) {
      uint32_t d_off = fb.create_vector_i64(val_specs[i].dims, val_specs[i].ndims);
      uint32_t n_off = fb.create_string(val_specs[i].name);
      fb.start_table();
      fb.add_i32(3u, -1, 0);
      fb.add_uoffset(2u, n_off);
      fb.add_uoffset(0u, d_off);
      fb.add_u8(1u, val_specs[i].dt, 0u);
      val_offs[i] = fb.end_table();
    }
    const int32_t fn_in_idx[2] = {1, 2};
    const int32_t fn_out_idx[1] = {6};
    uint32_t fn_vals = fb.create_vector_uoffsets(val_offs, 14u);
    uint32_t fn_ins = fb.create_vector_i32(fn_in_idx, 2u);
    uint32_t fn_outs = fb.create_vector_i32(fn_out_idx, 1u);

    enum DetKind { DET_NONE = 0, DET_DISP = 1, DET_PUT = 2 };
    struct OpSpec {
      uint8_t ot; DetKind det; uint32_t a; uint64_t b; uint32_t c;
      int32_t in_val; bool has_in; int32_t out_val; bool has_out; const char* name;
    };
    const OpSpec op_specs[25] = {
        {59u, DET_NONE, 0u, 0ULL, 0u, 0, true, 3, true, "main"},
        {61u, DET_NONE, 0u, 0ULL, 0u, 1, true, 0, false, "dive_vm_tensor_cache_clean_invalidate"},
        {61u, DET_NONE, 0u, 0ULL, 0u, 2, true, 0, false, "dive_vm_tensor_cache_clean_invalidate.1"},
        {59u, DET_NONE, 0u, 0ULL, 0u, 1, true, 4, true, "dive_vm_tensor_address_of"},
        {59u, DET_NONE, 0u, 0ULL, 0u, 2, true, 5, true, "dive_vm_tensor_address_of.2"},
        {59u, DET_NONE, 0u, 0ULL, 0u, 6, true, 7, true, "dive_vm_tensor_address_of.4"},
        {61u, DET_NONE, 0u, 0ULL, 0u, 6, true, 0, false, "dive_vm_tensor_cache_clean_invalidate.2"},
        {47u, DET_DISP, 0u, 0ULL, 0u, 0, false, 0, false, "program_prolog_hw_instructions_0"},
        {25u, DET_NONE, 0u, 0ULL, 0u, 4, true, 8, true, "_tpu_input_from_input_0::0"},
        {25u, DET_NONE, 0u, 0ULL, 0u, 5, true, 9, true, "_tpu_input1_from_input_1::0"},
        {25u, DET_NONE, 0u, 0ULL, 0u, 7, true, 10, true, "_tpu_output_from_output_0::0"},
        {25u, DET_NONE, 0u, 0ULL, 0u, 8, true, 11, true, "4097->5"},
        {47u, DET_DISP, 1u, 1ULL, 0u, 0, false, 0, false, "before_flexible_dma__tpu_input_from_input_0_hw_instructions_1"},
        {50u, DET_PUT, 2u, 37ULL, 64u, 11, true, 0, false, "_tpu_input_from_input_0"},
        {47u, DET_DISP, 2u, 2ULL, 0u, 0, false, 0, false, "after_request_flexible_dma__tpu_input_from_input_0_hw_instructions_2"},
        {25u, DET_NONE, 0u, 0ULL, 0u, 9, true, 12, true, "4099->5"},
        {47u, DET_DISP, 3u, 3ULL, 0u, 0, false, 0, false, "before_flexible_dma__tpu_input1_from_input_1_hw_instructions_3"},
        {50u, DET_PUT, 4u, 37ULL, 64u, 12, true, 0, false, "_tpu_input1_from_input_1"},
        {47u, DET_DISP, 4u, 4ULL, 0u, 0, false, 0, false, "after_request_flexible_dma__tpu_input1_from_input_1_hw_instructions_4"},
        {25u, DET_NONE, 0u, 0ULL, 0u, 10, true, 13, true, "4101->5"},
        {47u, DET_DISP, 5u, 5ULL, 0u, 0, false, 0, false, "before_flexible_dma__tpu_output_from_output_0_hw_instructions_5"},
        {50u, DET_PUT, 6u, 37ULL, 64u, 13, true, 0, false, "_tpu_output_from_output_0"},
        {47u, DET_DISP, 6u, 6ULL, 0u, 0, false, 0, false, "after_request_flexible_dma__tpu_output_from_output_0_hw_instructions_6;finish_and_sleep_hw_instructions_7"},
        {51u, DET_NONE, 0u, 0ULL, 0u, 0, false, 0, false, "halt"},
        {61u, DET_NONE, 0u, 0ULL, 0u, 6, true, 0, false, "dive_vm_tensor_cache_clean_invalidate.3"},
    };
    uint32_t op_offs[25]{};
    for (size_t i = 0; i < 25u; ++i) {
      const OpSpec& op = op_specs[i];
      fb.start_table();
      if (op.det == DET_DISP) {
        fb.add_u32(1u, static_cast<uint32_t>(op.b), 0u);
        fb.add_u32(0u, op.a, 0u);
      } else if (op.det == DET_PUT) {
        fb.add_u64(1u, op.b, 0ULL);
        fb.add_u32(2u, op.c, 0u);
        fb.add_u32(0u, op.a, 0u);
      }
      uint32_t det_off = fb.end_table();
      uint32_t in_off = fb.create_vector_i32(&op.in_val, op.has_in ? 1u : 0u);
      uint32_t out_off = fb.create_vector_i32(&op.out_val, op.has_out ? 1u : 0u);
      uint32_t nm_off = fb.create_string(op.name);
      fb.start_table();
      fb.add_uoffset(4u, nm_off);
      fb.add_uoffset(3u, out_off);
      fb.add_uoffset(2u, in_off);
      fb.add_uoffset(1u, det_off);
      fb.add_u8(0u, op.ot, 0u);
      op_offs[i] = fb.end_table();
    }
    uint32_t fn_ops = fb.create_vector_uoffsets(op_offs, 25u);
    uint32_t fn_nm = fb.create_string("main");
    fb.start_table();
    fb.add_uoffset(4u, fn_nm);
    fb.add_uoffset(3u, fn_ops);
    fb.add_uoffset(2u, fn_outs);
    fb.add_uoffset(1u, fn_ins);
    fb.add_uoffset(0u, fn_vals);
    uint32_t fn_off = fb.end_table();

    uint32_t funcs_vec = fb.create_vector_uoffsets(&fn_off, 1u);
    uint32_t mod_name = fb.create_string("");
    uint32_t chunk_offs[7]{};
    for (size_t i = 0; i < 7u; ++i) {
      uint32_t b_off = fb.create_vector_bytes(parts.chunks[i].data(), parts.chunks[i].size(), 32u);
      fb.start_table();
      fb.add_uoffset(0u, b_off);
      chunk_offs[i] = fb.end_table();
    }
    uint32_t chunks_vec = fb.create_vector_uoffsets(chunk_offs, 7u);
    const int32_t port_in[2] = {1, 2};
    const int32_t port_out = 6;
    const int32_t port_scr = 0;
    uint32_t p_in = fb.create_vector_i32(port_in, 2u);
    uint32_t p_out = fb.create_vector_i32(&port_out, 1u);
    uint32_t p_param = fb.create_vector_i32(nullptr, 0u);
    uint32_t p_scr = fb.create_vector_i32(&port_scr, 1u);
    uint32_t p_var = fb.create_vector_i32(nullptr, 0u);

    fb.start_table();
    fb.add_uoffset(8u, p_var);
    fb.add_uoffset(7u, p_scr);
    fb.add_uoffset(6u, p_param);
    fb.add_uoffset(5u, p_out);
    fb.add_uoffset(4u, p_in);
    fb.add_uoffset(3u, chunks_vec);
    fb.add_uoffset(2u, mod_name);
    fb.add_uoffset(1u, funcs_vec);
    uint32_t root = fb.end_table();
    ext0_dive = fb.finish(root, "DIVE");
  }

  // Deterministic hashes
  uint8_t digest[32]{};
  sha256_compute(ext0_dive.data(), ext0_dive.size(), digest);
  uint64_t exec_hash = 0ULL, tok_raw = 0ULL, model_hash = 0ULL;
  std::memcpy(&exec_hash, digest, 8u);
  std::memcpy(&tok_raw, digest + 8u, 8u);
  std::memcpy(&model_hash, digest + 16u, 8u);
  constexpr uint64_t kPow10_18 = 1000000000000000000ULL;
  constexpr uint64_t kPow10_19 = 10000000000000000000ULL;
  uint64_t tok_num = tok_raw % kPow10_19;
  if (tok_num < kPow10_18) tok_num += kPow10_18;
  char token_str[32]{};
  std::snprintf(token_str, sizeof(token_str), "Token_%019llu",
                static_cast<unsigned long long>(tok_num));

  // Build 2-input 0-param MultiExecutable
  std::vector<uint8_t> mexec;
  {
    FbBuilder fb;
    auto make_layer = [&](uint64_t sz, const char* name) -> uint32_t {
      uint32_t nm = fb.create_string(name);
      fb.start_table();
      fb.add_u64(3u, sz, 0ULL, true);
      fb.add_uoffset(2u, nm);
      fb.add_u32(1u, static_cast<uint32_t>(sz), 0u, true);
      return fb.end_table();
    };
    uint32_t in_layers[2] = {
        make_layer(static_cast<uint64_t>(M) * K, "input_0"),
        make_layer(static_cast<uint64_t>(ext2_bytes), "input_1"),
    };
    uint32_t out_layer = make_layer(static_cast<uint64_t>(M) * N, "output_0");
    uint32_t ins_vec = fb.create_vector_uoffsets(in_layers, 2u);
    uint32_t outs_vec = fb.create_vector_uoffsets(&out_layer, 1u);
    uint32_t params_vec = fb.create_vector_uoffsets(nullptr, 0u);
    uint32_t empty_meta = fb.create_vector_bytes(nullptr, 0u);
    uint32_t empty_bb = fb.create_vector_uoffsets(nullptr, 0u);
    uint32_t empty_dma = fb.create_vector_uoffsets(nullptr, 0u);
    uint32_t arch_str = fb.create_string("buenos_a0");
    uint32_t name_str = fb.create_string("main_STAND_ALONE_cluster2");

    fb.start_table();
    fb.add_uoffset(32u, empty_dma);
    fb.add_u64(29u, 81920ULL, 0ULL, true);
    fb.add_u64(28u, 32768ULL, 0ULL, true);
    fb.add_uoffset(25u, empty_bb);
    fb.add_u32(24u, 2u, 0u, true);
    fb.add_u32(23u, 311296u, 0u, true);
    fb.add_u64(22u, exec_hash, 0ULL, true);
    fb.add_uoffset(20u, params_vec);
    fb.add_u8(18u, 1u, 0u, true);
    fb.add_u8(17u, 1u, 0u, true);
    fb.add_u32(14u, parts.exec_field_14, 0u, true);
    fb.add_u32(13u, 16384u, 0u, true);
    fb.add_u32(12u, 262144u, 0u, true);
    fb.add_u32(11u, 32768u, 0u, true);
    fb.add_u32(8u, 2u, 0u, true);
    fb.add_uoffset(4u, outs_vec);
    fb.add_uoffset(3u, ins_vec);
    fb.add_uoffset(2u, empty_meta);
    fb.add_uoffset(1u, arch_str);
    fb.add_uoffset(0u, name_str);
    uint32_t exec0 = fb.end_table();
    uint32_t execs_vec = fb.create_vector_uoffsets(&exec0, 1u);
    fb.start_table();
    fb.add_uoffset(1u, execs_vec);
    uint32_t root = fb.end_table();
    mexec = fb.finish(root, nullptr);
  }

  // Build 2-input 0-param DWS0 (matching add.dgc0 layout)
  std::vector<uint8_t> dws0;
  {
    FbBuilder fb;
    auto make_tref = [&](uint32_t idx) -> uint32_t {
      fb.start_table();
      fb.add_u32(0u, idx, 0u);
      return fb.end_table();
    };
    auto make_tensor = [&](const uint32_t* dims, size_t ndims, uint8_t dt,
                           const char* name, uint8_t storage,
                           bool is_ext, uint64_t ext_idx) -> uint32_t {
      uint32_t d_off = fb.create_vector_u32(dims, ndims);
      uint32_t n_off = fb.create_string(name);
      fb.start_table();
      if (is_ext && ext_idx != 0ULL) fb.add_u64(13u, ext_idx, 0ULL);
      if (is_ext) fb.add_u8(12u, 1u, 0u);
      fb.add_u8(5u, storage, 0u);
      fb.add_i32(3u, -1, 0);
      fb.add_uoffset(2u, n_off);
      fb.add_u8(1u, dt, 0u);
      fb.add_uoffset(0u, d_off);
      return fb.end_table();
    };
    char scratch_name[48]{};
    std::snprintf(scratch_name, sizeof(scratch_name), "dive_scratch_%u", parts.dive_scratch_size);
    uint32_t in0_dims[2] = {M, K};
    uint32_t in1_dims[1] = {ext2_bytes};
    uint32_t out_dims[2] = {M, N};
    uint32_t ext0_dims[1] = {static_cast<uint32_t>(ext0_dive.size())};
    uint32_t ext1_dims[1] = {static_cast<uint32_t>(dive_interp_size)};
    uint32_t scr_dims[1] = {parts.dive_scratch_size};

    uint32_t t_offs[6]{};
    t_offs[0] = make_tensor(in0_dims, 2u, 1u, "input_0", 1u, false, 0ULL);
    t_offs[1] = make_tensor(in1_dims, 1u, 1u, "input_1", 1u, false, 0ULL);
    t_offs[2] = make_tensor(out_dims, 2u, 1u, "output_0", 1u, false, 0ULL);
    t_offs[3] = make_tensor(ext0_dims, 1u, 2u, "main_STAND_ALONE_dive_vm_ir_Group_0_0_2_0", 5u, true, 0ULL);
    t_offs[4] = make_tensor(ext1_dims, 1u, 2u, "main_STAND_ALONE_interpreter_Group_0_0_2_0", 5u, true, 1ULL);
    t_offs[5] = make_tensor(scr_dims, 1u, 2u, scratch_name, 3u, false, 0ULL);

    uint32_t tensors_vec = fb.create_vector_uoffsets(t_offs, 6u);
    uint32_t sg_in_refs[2] = {make_tref(0u), make_tref(1u)};
    uint32_t sg_out_ref = make_tref(2u);
    uint32_t sg_ins = fb.create_vector_uoffsets(sg_in_refs, 2u);
    uint32_t sg_outs = fb.create_vector_uoffsets(&sg_out_ref, 1u);

    uint32_t b_res_refs[2] = {make_tref(3u), make_tref(4u)};
    uint32_t b_res_vec = fb.create_vector_uoffsets(b_res_refs, 2u);
    uint32_t b_param_vec = fb.create_vector_uoffsets(nullptr, 0u);
    uint32_t b_scr_ref = make_tref(5u);
    uint32_t b_aux_ref = make_tref(0u);
    uint32_t b_interp_ref = make_tref(4u);
    uint32_t b_ir_ref = make_tref(3u);
    fb.start_table();
    fb.add_uoffset(8u, b_ir_ref);
    fb.add_uoffset(7u, b_interp_ref);
    fb.add_uoffset(6u, b_aux_ref);
    fb.add_uoffset(5u, b_scr_ref);
    fb.add_uoffset(4u, b_param_vec);
    fb.add_uoffset(0u, b_res_vec);
    uint32_t binding0 = fb.end_table();
    uint32_t bindings_vec = fb.create_vector_uoffsets(&binding0, 1u);

    uint32_t me_vec = fb.create_vector_bytes(mexec.data(), mexec.size());
    uint32_t empty_b3 = fb.create_vector_bytes(nullptr, 0u);
    uint32_t empty_b4 = fb.create_vector_bytes(nullptr, 0u);
    fb.start_table();
    fb.add_uoffset(4u, empty_b4);
    fb.add_uoffset(3u, empty_b3);
    fb.add_uoffset(1u, bindings_vec);
    fb.add_uoffset(0u, me_vec);
    uint32_t op0_detail = fb.end_table();

    uint32_t op0_in_refs[2] = {make_tref(0u), make_tref(1u)};
    uint32_t op0_out_ref = make_tref(2u);
    uint32_t op0_ins = fb.create_vector_uoffsets(op0_in_refs, 2u);
    uint32_t op0_outs = fb.create_vector_uoffsets(&op0_out_ref, 1u);
    uint32_t op0_nm = fb.create_string("main");
    fb.start_table();
    fb.add_uoffset(5u, op0_nm);
    fb.add_uoffset(4u, op0_outs);
    fb.add_uoffset(3u, op0_ins);
    fb.add_uoffset(1u, op0_detail);
    fb.add_u8(0u, 2u, 0u);
    uint32_t op0 = fb.end_table();

    fb.start_table();
    uint32_t op1_detail = fb.end_table();
    uint32_t op1_ins = fb.create_vector_uoffsets(nullptr, 0u);
    uint32_t op1_outs = fb.create_vector_uoffsets(nullptr, 0u);
    uint32_t op1_nm = fb.create_string("main.1");
    fb.start_table();
    fb.add_uoffset(5u, op1_nm);
    fb.add_uoffset(4u, op1_outs);
    fb.add_uoffset(3u, op1_ins);
    fb.add_uoffset(1u, op1_detail);
    fb.add_u8(0u, 7u, 0u);
    uint32_t op1 = fb.end_table();

    uint32_t ops_arr[2] = {op0, op1};
    uint32_t ops_vec = fb.create_vector_uoffsets(ops_arr, 2u);
    uint32_t sg_nm = fb.create_string("main");
    fb.start_table();
    fb.add_uoffset(4u, sg_nm);
    fb.add_uoffset(3u, ops_vec);
    fb.add_uoffset(2u, sg_outs);
    fb.add_uoffset(1u, sg_ins);
    fb.add_uoffset(0u, tensors_vec);
    uint32_t sg0 = fb.end_table();
    uint32_t sgs_vec = fb.create_vector_uoffsets(&sg0, 1u);

    uint32_t hdr_arch = fb.create_string("buenos_a0");
    fb.start_table();
    fb.add_uoffset(0u, hdr_arch);
    uint32_t pkg_hdr = fb.end_table();

    uint32_t model_id = fb.create_string(token_str);
    uint32_t empty_emb = fb.create_vector_uoffsets(nullptr, 0u);
    uint32_t empty_exec_refs = fb.create_vector_uoffsets(nullptr, 0u);
    uint32_t empty_s = fb.create_string("");
    uint32_t f11_vec = fb.create_vector_uoffsets(&empty_s, 1u);
    uint32_t f12_s = fb.create_string("");
    uint32_t f14_s = fb.create_string("");
    uint32_t f17_s = fb.create_string("");
    uint32_t rel_s = fb.create_string("lga_tpu_20260603.0.0");

    fb.start_table();
    fb.add_uoffset(18u, rel_s);
    fb.add_uoffset(17u, f17_s);
    fb.add_uoffset(14u, f14_s);
    fb.add_u8(13u, 2u, 0u);
    fb.add_uoffset(12u, f12_s);
    fb.add_uoffset(11u, f11_vec);
    fb.add_u64(10u, kCompilerCl, 0ULL);
    fb.add_u8(9u, 1u, 0u);
    fb.add_uoffset(8u, empty_exec_refs);
    fb.add_u32(6u, 1u, 0u);
    fb.add_uoffset(5u, empty_emb);
    fb.add_uoffset(4u, model_id);
    fb.add_uoffset(3u, sgs_vec);
    fb.add_uoffset(2u, pkg_hdr);
    fb.add_u8(1u, 1u, 0u);
    fb.add_u32(0u, 6u, 0u);
    uint32_t root = fb.end_table();
    dws0 = fb.finish(root, "DWS0");
  }

  std::vector<uint8_t> dgd0 = build_dgd0_clean();
  FbBuilder fb;
  uint32_t dws_vec = fb.create_vector_bytes(dws0.data(), dws0.size(), 64u);
  uint32_t dgd_vec = fb.create_vector_bytes(dgd0.data(), dgd0.size(), 4096u);
  uint32_t eb_offs[2]{};
  uint32_t eb_off_from_end[2]{};
  const uint64_t eb_sizes[2] = {
      static_cast<uint64_t>(ext0_dive.size()),
      static_cast<uint64_t>(dive_interp_size),
  };
  for (size_t i = 0; i < 2u; ++i) {
    fb.start_table();
    fb.add_u64(1u, eb_sizes[i], 0ULL, true);
    fb.add_u64(0u, 0ULL, 0ULL, true, &eb_off_from_end[i]);
    eb_offs[i] = fb.end_table();
  }
  uint32_t ebs_vec = fb.create_vector_uoffsets(eb_offs, 2u);

  fb.start_table();
  fb.add_uoffset(7u, ebs_vec);
  fb.add_u32(6u, 64u, 0u, true);
  fb.add_u64(5u, kCompilerCl, 0ULL, true);
  fb.add_u64(4u, model_hash, 0ULL, true);
  fb.add_u32(3u, 6u, 0u, true);
  fb.add_uoffset(1u, dgd_vec);
  fb.add_uoffset(0u, dws_vec);
  uint32_t root = fb.end_table();

  std::vector<uint8_t> hdr = fb.finish(root, "DGC0");
  const uint64_t ext0_off = align_up_u64(static_cast<uint64_t>(hdr.size()), 64ULL);
  const uint64_t ext1_off = align_up_u64(ext0_off + eb_sizes[0], 64ULL);
  const uint64_t total_sz = align_up_u64(ext1_off + eb_sizes[1], 64ULL);
  const uint64_t abs_offs[2] = {ext0_off, ext1_off};
  for (size_t i = 0; i < 2u; ++i) {
    size_t pos = hdr.size() - static_cast<size_t>(eb_off_from_end[i]);
    std::memcpy(hdr.data() + pos, &abs_offs[i], 8u);
  }

  out_dgc0->clear();
  out_dgc0->reserve(static_cast<size_t>(total_sz));
  out_dgc0->insert(out_dgc0->end(), hdr.begin(), hdr.end());
  auto append_aligned = [&](uint64_t target_off, const uint8_t* data, size_t len) {
    if (target_off > out_dgc0->size()) {
      out_dgc0->insert(out_dgc0->end(), static_cast<size_t>(target_off - out_dgc0->size()), 0x0Au);
    }
    if (len > 0u) out_dgc0->insert(out_dgc0->end(), data, data + len);
  };
  append_aligned(ext0_off, ext0_dive.data(), ext0_dive.size());
  append_aligned(ext1_off, dive_interp, dive_interp_size);
  if (total_sz > out_dgc0->size()) {
    out_dgc0->insert(out_dgc0->end(), static_cast<size_t>(total_sz - out_dgc0->size()), 0x0Au);
  }
  return OB_OK;
}

ob_status_t builtin_gemm_emitter_cb(const ob_gemm_desc_t* desc,
                                    uint8_t* out_buf,
                                    size_t out_cap,
                                    size_t* out_size,
                                    void* user_data) {
  if (!desc || !out_size || !user_data) return OB_ERR_INVALID_ARG;
  ob_context* ctx = static_cast<ob_context*>(user_data);
  if (ctx->builtin_dive_interp.empty()) {
    set_last_error(ctx, "Builtin GEMM emitter has no loaded RV64 DIVE interpreter image");
    return OB_ERR_INVALID_ARG;
  }

  GemmSynthResult parts{};
  std::string err;
  ob_status_t st = synthesize_gemm_parts(*desc, &parts, &err);
  if (st != OB_OK) {
    if (!err.empty()) set_last_error(ctx, err);
    return st;
  }

  std::vector<uint8_t> dgc0;
  st = build_dgc0_from_parts(*desc, parts,
                             ctx->builtin_dive_interp.data(),
                             ctx->builtin_dive_interp.size(),
                             &dgc0, &err);
  if (st != OB_OK) {
    if (!err.empty()) set_last_error(ctx, err);
    return st;
  }

  *out_size = dgc0.size();
  if (!out_buf || out_cap < dgc0.size()) {
    return OB_ERR_OVERFLOW;
  }
  std::memcpy(out_buf, dgc0.data(), dgc0.size());
  return OB_OK;
}

}  // namespace ob

extern "C" {

ob_status_t ob_gemm_extract_dive_interp_from_dgc0(const char* dgc0_path,
                                                  uint8_t* out_buf,
                                                  size_t out_cap,
                                                  size_t* out_size) {
  if (!dgc0_path || !dgc0_path[0] || !out_size) return OB_ERR_INVALID_ARG;
  std::vector<uint8_t> raw;
  std::string err;
  if (!ob::read_entire_file(dgc0_path, &raw, &err)) {
    return OB_ERR_IO;
  }
  ob::ParsedDgc0 parsed{};
  ob_status_t st = ob::parse_dgc0_buffer(raw.data(), raw.size(), &parsed, &err);
  if (st != OB_OK) return st;

  const size_t img_sz = static_cast<size_t>(parsed.ext_dive_image.size);
  const size_t img_off = static_cast<size_t>(parsed.ext_dive_image.offset);
  *out_size = img_sz;
  if (!out_buf || out_cap < img_sz) {
    return OB_ERR_OVERFLOW;
  }
  std::memcpy(out_buf, raw.data() + img_off, img_sz);
  return OB_OK;
}

ob_status_t ob_gemm_register_builtin_emitter(ob_context_t* ctx,
                                             const char* dive_interp_path) {
  if (!ctx) return OB_ERR_INVALID_ARG;

  const char* path = dive_interp_path;
  if (!path || !path[0]) {
    const char* env_p = std::getenv("OPENBUENOS_DIVE_INTERP_PATH");
    path = (env_p && env_p[0]) ? env_p : ob::kDefaultDiveInterpPath;
  }

  std::vector<uint8_t> raw;
  std::string err;
  if (!ob::read_entire_file(path, &raw, &err)) {
    ob::set_last_error(ctx, "Failed loading RV64 DIVE interpreter (" + std::string(path) + "): " + err);
    return OB_ERR_IO;
  }

  std::vector<uint8_t> interp;
  if (raw.size() >= 16u && std::memcmp(raw.data() + 4u, "DGC0", 4u) == 0) {
    ob::ParsedDgc0 parsed{};
    ob_status_t pst = ob::parse_dgc0_buffer(raw.data(), raw.size(), &parsed, &err);
    if (pst != OB_OK) {
      ob::set_last_error(ctx, "Failed extracting ext[1] from DGC0 (" + std::string(path) + "): " + err);
      return pst;
    }
    const size_t off = static_cast<size_t>(parsed.ext_dive_image.offset);
    const size_t sz = static_cast<size_t>(parsed.ext_dive_image.size);
    interp.assign(raw.data() + off, raw.data() + off + sz);
  } else {
    interp = std::move(raw);
  }

  if (interp.size() < 65536u) {
    ob::set_last_error(ctx, "RV64 DIVE interpreter image too small (" +
                                std::to_string(interp.size()) + " B)");
    return OB_ERR_BAD_DGC0;
  }

  std::lock_guard<std::mutex> lk(ctx->gemm_mu);
  ctx->builtin_dive_interp = std::move(interp);
  ctx->gemm_emitter = ob::builtin_gemm_emitter_cb;
  ctx->gemm_emitter_user_data = ctx;
  ctx->builtin_emitter_active = true;
  return OB_OK;
}

}  // extern "C"
