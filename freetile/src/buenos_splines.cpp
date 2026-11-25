// SPDX-License-Identifier: Apache-2.0
// Clean-room zero-dependency C++17 Tensor G5 (buenos_a0) 0x640 Minimax Spline Library.

#include "freetile/buenos_splines.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace freetile {

namespace {

bool set_err(std::string* err, const char* msg) {
    if (err != nullptr) {
        *err = msg;
    }
    return false;
}

inline uint32_t f32_to_u32(float v) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

inline float u32_to_f32(uint32_t bits) {
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

void write_bits_le(uint8_t* buf, size_t buf_len, uint32_t bit_off, uint32_t width, uint64_t val) {
    for (uint32_t i = 0u; i < width; ++i) {
        const uint32_t pos = bit_off + i;
        const size_t byte_idx = static_cast<size_t>(pos >> 3u);
        const uint32_t bit_idx = pos & 7u;
        if (byte_idx >= buf_len) {
            break;
        }
        if (((val >> i) & 1ULL) != 0ULL) {
            buf[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
        } else {
            buf[byte_idx] &= static_cast<uint8_t>(~(1u << bit_idx));
        }
    }
}

uint64_t read_bits_le(const uint8_t* buf, size_t buf_len, uint32_t bit_off, uint32_t width) {
    uint64_t out = 0ULL;
    for (uint32_t i = 0u; i < width; ++i) {
        const uint32_t pos = bit_off + i;
        const size_t byte_idx = static_cast<size_t>(pos >> 3u);
        const uint32_t bit_idx = pos & 7u;
        if (byte_idx >= buf_len) {
            break;
        }
        if (((buf[byte_idx] >> bit_idx) & 1u) != 0u) {
            out |= (1ULL << i);
        }
    }
    return out;
}

// Strict IEEE-754 FP32 Horner kernel (separate mul + add rounds, no NaN swallow).
inline float eval_horner_core_f32(float x_in,
                                  const float coeffs[kOp640NumSegments][kOp640CoeffsPerSegment],
                                  const float bps[kOp640NumBreakpoints],
                                  float fclip_min,
                                  float fclip_max) {
    if (std::isnan(x_in) || std::isnan(fclip_min) || std::isnan(fclip_max) ||
        fclip_min > fclip_max) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float x = x_in;
    if (x < fclip_min) {
        x = fclip_min;
    } else if (x > fclip_max) {
        x = fclip_max;
    }

    uint32_t seg = kOp640NumSegments - 1u;
    for (int s = static_cast<int>(kOp640NumBreakpoints) - 1; s >= 0; --s) {
        if (x < bps[s]) {
            seg = static_cast<uint32_t>(s);
        }
    }

    const float* c = coeffs[seg];
    float y = c[4];
    float p = y * x;
    y = p + c[3];
    p = y * x;
    y = p + c[2];
    p = y * x;
    y = p + c[1];
    p = y * x;
    y = p + c[0];
    return y;
}

// Clean-room degree-4 Remez minimax tables (synthesized from scratch in float64, rounded to FP32).

const SplineTable0x640 kExpNeg12_96Table = {
    SplineMode::NORMAL,
    -12.96f,
    0.0f,
    {
        {0.0191395339f, 0.00630242843f, 0.000782865332f, 4.34177236e-05f, 9.06106379e-07f},
        {0.118923187f, 0.0518533252f, 0.00861685909f, 0.000644828833f, 1.82892236e-05f},
        {0.32562387f, 0.178105161f, 0.0376598798f, 0.00362652028f, 0.000133533467f},
        {0.597417593f, 0.401288688f, 0.106734417f, 0.013174871f, 0.00063083024f},
        {0.832630873f, 0.669221401f, 0.221957728f, 0.0353442356f, 0.00224069296f},
        {0.95949626f, 0.881079733f, 0.355942398f, 0.0733803436f, 0.00632958347f},
        {0.996421874f, 0.980760455f, 0.458602011f, 0.121234328f, 0.0148496395f},
        {0.999998569f, 0.999889433f, 0.498614341f, 0.160524666f, 0.0302277021f},
    },
    {-8.96734524f, -6.650383f, -4.92254686f, -3.51585174f, -2.36831903f, -1.42645741f, -0.654367089f},
    1.983773e-06f,
};

const SplineTable0x640 kExpNeg11Table = {
    SplineMode::NORMAL,
    -11.0f,
    0.0f,
    {
        {0.0476641022f, 0.0178173762f, 0.00252240873f, 0.000159972667f, 3.82872258e-06f},
        {0.192271873f, 0.0924371555f, 0.0170257259f, 0.00141810463f, 4.49186773e-05f},
        {0.42441082f, 0.251347244f, 0.0579896495f, 0.00613035029f, 0.000248984492f},
        {0.678991616f, 0.484026998f, 0.138126045f, 0.0184552651f, 0.000963098486f},
        {0.872743905f, 0.727523685f, 0.253608108f, 0.0429503359f, 0.00292345393f},
        {0.970060825f, 0.905056298f, 0.376198679f, 0.0809309483f, 0.00737748202f},
        {0.997399271f, 0.984963298f, 0.465253532f, 0.12582776f, 0.0160183311f},
        {0.999998987f, 0.999915361f, 0.498862237f, 0.161260709f, 0.0308961812f},
    },
    {-7.87157679f, -5.90575123f, -4.40028095f, -3.17781901f, -2.16587782f, -1.31762588f, -0.608947694f},
    1.410793e-06f,
};

const SplineTable0x640 kRsqrtMantissaTable = {
    SplineMode::RSQRT,
    1.0f,
    4.0f,
    {
        {2.34678411f, -2.84304857f, 2.32221484f, -1.00244963f, 0.176499158f},
        {2.12686586f, -2.1161201f, 1.41934419f, -0.50305742f, 0.0727123171f},
        {1.93392384f, -1.59123528f, 0.882886708f, -0.258932173f, 0.0309781171f},
        {1.77143443f, -1.22294414f, 0.569355786f, -0.140117139f, 0.0140671711f},
        {1.62230515f, -0.939314365f, 0.366742164f, -0.075686276f, 0.00637173094f},
        {1.49097311f, -0.729271889f, 0.240585625f, -0.0419609323f, 0.0029860225f},
        {1.37441194f, -0.571222603f, 0.160110697f, -0.0237244777f, 0.00143420638f},
        {1.27369022f, -0.454712272f, 0.109510794f, -0.0139463069f, 0.000724803831f},
    },
    {1.21080089f, 1.4825176f, 1.77064002f, 2.10588312f, 2.51725769f, 2.95110345f, 3.4864037f},
    3.665169e-07f,
};

const SplineTable0x640 kRecipMantissaTable = {
    SplineMode::RECIP,
    1.0f,
    2.0f,
    {
        {4.78107738f, -9.14116859f, 8.73650742f, -4.17382479f, 0.797408402f},
        {4.36040068f, -7.60312366f, 6.62684822f, -2.8871634f, 0.503008306f},
        {3.95247602f, -6.24678564f, 4.93482924f, -1.94856942f, 0.307664871f},
        {3.60577846f, -5.19958401f, 3.74816513f, -1.35067034f, 0.194648296f},
        {3.27876258f, -4.29851055f, 2.81665802f, -0.922483027f, 0.120803893f},
        {2.98769426f, -3.56987524f, 2.13236094f, -0.636735678f, 0.076039359f},
        {2.7744801f, -3.07860208f, 1.70775604f, -0.473585606f, 0.0525244512f},
        {2.58705401f, -2.67674994f, 1.38457644f, -0.358041137f, 0.0370293185f},
    },
    {1.094118f, 1.20230925f, 1.33171809f, 1.44435108f, 1.61104012f, 1.73895419f, 1.86809993f},
    1.059185e-06f,
};

const SplineTable0x640 kSiluTable = {
    SplineMode::NORMAL,
    -11.1f,
    11.1f,
    {
        {-0.420764923f, -0.164861679f, -0.0245584957f, -0.0016421855f, -4.1464089e-05f},
        {-0.67491585f, -0.283502579f, -0.0420540087f, -0.00226688059f, -1.28515485e-05f},
        {0.00498309452f, 0.548501492f, 0.347209364f, 0.0802294239f, 0.00666073151f},
        {4.31725093e-05f, 0.501607776f, 0.259173572f, 0.0175480079f, -0.0090315165f},
        {4.90780367e-05f, 0.498320937f, 0.259398162f, -0.0178037472f, -0.00893626455f},
        {0.00449135946f, 0.452420652f, 0.346575856f, -0.080040291f, 0.00664000167f},
        {-0.676708996f, 1.28511178f, -0.0425887294f, 0.0023448607f, -1.70638305e-05f},
        {-0.419247448f, 1.16415524f, -0.024436662f, 0.00163295423f, -4.1204672e-05f},
    },
    {-6.14638805f, -3.24068832f, -1.35916698f, 0.00174465729f, 1.36345744f, 3.24618793f, 6.16077852f},
    4.623088e-05f,
};

const SplineTable0x640 kGeluTanhTable = {
    SplineMode::NORMAL,
    -11.1f,
    11.1f,
    {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {-0.737986624f, -0.746289194f, -0.284569442f, -0.0484482348f, -0.00310468581f},
        {0.0272876304f, 0.637874067f, 0.665122509f, 0.244518563f, 0.0311695207f},
        {3.29272771e-05f, 0.501699209f, 0.41235432f, 0.0353746973f, -0.0342364609f},
        {3.75464442e-05f, 0.498223156f, 0.412695855f, -0.0359185115f, -0.033952944f},
        {0.0270343255f, 0.362802029f, 0.664459407f, -0.244235188f, 0.0311249532f},
        {-0.738193572f, 1.74656129f, -0.284701765f, 0.0484764352f, -0.00310691004f},
        {0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
    },
    {-4.23674393f, -2.21483922f, -0.966763556f, 0.00128772273f, 0.969663322f, 2.21699429f, 4.23493052f},
    3.531375e-05f,
};

const SplineTable0x640 kGeluErfTable = {
    SplineMode::NORMAL,
    -11.1f,
    11.1f,
    {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {-0.705597639f, -0.707284391f, -0.267569363f, -0.0452168807f, -0.00287665916f},
        {0.0188986156f, 0.616971612f, 0.646397531f, 0.237089947f, 0.0300433654f},
        {4.48575738e-05f, 0.502154827f, 0.415008575f, 0.0401286446f, -0.0314284079f},
        {4.83976837e-05f, 0.497790396f, 0.41523236f, -0.0404611938f, -0.0312660746f},
        {0.0186871663f, 0.38356337f, 0.645899951f, -0.236888096f, 0.0300132018f},
        {-0.705673695f, 1.70738983f, -0.267622828f, 0.0452286638f, -0.0028776126f},
        {0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
    },
    {-4.33125305f, -2.30122232f, -1.02266407f, 0.000801948074f, 1.02420223f, 2.30219698f, 4.3302536f},
    4.668011e-05f,
};

}  // namespace

const char* spline_mode_name(SplineMode mode) {
    switch (mode) {
        case SplineMode::NORMAL: return "normal";
        case SplineMode::SQRT:   return "sqrt";
        case SplineMode::RSQRT:  return "rsqrt";
        case SplineMode::RECIP:  return "recip";
    }
    return "unknown";
}

const char* spline_kind_name(SplineKind kind) {
    switch (kind) {
        case SplineKind::EXP_NEG12_96_0: return "exp_neg12_96_0";
        case SplineKind::EXP_NEG11_0:    return "exp_neg11_0";
        case SplineKind::RSQRT_MANTISSA: return "rsqrt_mantissa";
        case SplineKind::RECIP_MANTISSA: return "recip_mantissa";
        case SplineKind::SILU:           return "silu";
        case SplineKind::GELU_TANH:      return "gelu_tanh";
        case SplineKind::GELU_ERF:       return "gelu_erf";
    }
    return "unknown";
}

SplineHeader76 make_op640_header(uint16_t core_mask,
                                 uint16_t seq,
                                 uint16_t flags12,
                                 uint32_t mid20) {
    SplineHeader76 hdr;
    hdr.lo64 = static_cast<uint64_t>(kOp640Opcode) |
               (static_cast<uint64_t>(core_mask) << 12) |
               (static_cast<uint64_t>(mid20 & 0xFFFFFu) << 28) |
               (static_cast<uint64_t>(seq) << 48);
    hdr.hi12 = static_cast<uint16_t>(flags12 & 0xFFFu);
    return hdr;
}

uint16_t op640_header_opcode(const SplineHeader76& hdr) {
    return static_cast<uint16_t>(hdr.lo64 & 0xFFFu);
}

uint16_t op640_header_core_mask(const SplineHeader76& hdr) {
    return static_cast<uint16_t>((hdr.lo64 >> 12) & 0xFFFFu);
}

uint16_t op640_header_seq(const SplineHeader76& hdr) {
    return static_cast<uint16_t>((hdr.lo64 >> 48) & 0xFFFFu);
}

uint16_t op640_header_flags(const SplineHeader76& hdr) {
    return static_cast<uint16_t>(hdr.hi12 & 0xFFFu);
}

const SplineTable0x640& buenos_spline_exp_neg12_96() { return kExpNeg12_96Table; }
const SplineTable0x640& buenos_spline_exp_neg11() { return kExpNeg11Table; }
const SplineTable0x640& buenos_spline_rsqrt_mantissa() { return kRsqrtMantissaTable; }
const SplineTable0x640& buenos_spline_recip_mantissa() { return kRecipMantissaTable; }
const SplineTable0x640& buenos_spline_silu() { return kSiluTable; }
const SplineTable0x640& buenos_spline_gelu_tanh() { return kGeluTanhTable; }
const SplineTable0x640& buenos_spline_gelu_erf() { return kGeluErfTable; }

const SplineTable0x640& buenos_spline_table(SplineKind kind) {
    switch (kind) {
        case SplineKind::EXP_NEG12_96_0: return kExpNeg12_96Table;
        case SplineKind::EXP_NEG11_0:    return kExpNeg11Table;
        case SplineKind::RSQRT_MANTISSA: return kRsqrtMantissaTable;
        case SplineKind::RECIP_MANTISSA: return kRecipMantissaTable;
        case SplineKind::SILU:           return kSiluTable;
        case SplineKind::GELU_TANH:      return kGeluTanhTable;
        case SplineKind::GELU_ERF:       return kGeluErfTable;
    }
    return kExpNeg12_96Table;
}

bool validate_spline_table(const SplineTable0x640& table, std::string* err) {
    if (!std::isfinite(table.domain_min) || !std::isfinite(table.domain_max) ||
        !(table.domain_min < table.domain_max)) {
        return set_err(err, "SplineTable0x640 domain must be finite with domain_min < domain_max");
    }
    for (uint32_t s = 0u; s < kOp640NumSegments; ++s) {
        for (uint32_t k = 0u; k < kOp640CoeffsPerSegment; ++k) {
            if (!std::isfinite(table.coeffs[s][k])) {
                return set_err(err, "SplineTable0x640 contains non-finite polynomial coefficient");
            }
        }
    }
    for (uint32_t i = 0u; i < kOp640NumBreakpoints; ++i) {
        if (!std::isfinite(table.breakpoints[i])) {
            return set_err(err, "SplineTable0x640 contains non-finite breakpoint");
        }
        if (i > 0u && !(table.breakpoints[i - 1u] < table.breakpoints[i])) {
            return set_err(err, "SplineTable0x640 breakpoints must be strictly ascending");
        }
    }
    if (!(table.domain_min < table.breakpoints[0] &&
          table.breakpoints[kOp640NumBreakpoints - 1u] < table.domain_max)) {
        return set_err(err, "SplineTable0x640 interior breakpoints must lie strictly inside domain");
    }
    return true;
}

SplineTable0x640 affine_fuse_spline_table(const SplineTable0x640& table,
                                          float scale,
                                          float offset) {
    SplineTable0x640 out = table;
    for (uint32_t s = 0u; s < kOp640NumSegments; ++s) {
        out.coeffs[s][0] = static_cast<float>(static_cast<double>(table.coeffs[s][0]) *
                                                  static_cast<double>(scale) +
                                              static_cast<double>(offset));
        for (uint32_t k = 1u; k < kOp640CoeffsPerSegment; ++k) {
            out.coeffs[s][k] = static_cast<float>(static_cast<double>(table.coeffs[s][k]) *
                                                  static_cast<double>(scale));
        }
    }
    out.max_abs_err = std::fabs(scale) * table.max_abs_err;
    return out;
}

bool make_op640_packet(const SplineTable0x640& table,
                       SplinePacket0x640* out_pkt,
                       uint16_t core_mask,
                       uint16_t seq,
                       uint16_t flags12,
                       uint64_t tail_lo64,
                       std::string* err) {
    if (out_pkt == nullptr) {
        return set_err(err, "out_pkt must not be null");
    }
    if (!validate_spline_table(table, err)) {
        return false;
    }
    out_pkt->header = make_op640_header(core_mask, seq, flags12);
    std::memcpy(out_pkt->coeffs, table.coeffs, sizeof(out_pkt->coeffs));
    std::memcpy(out_pkt->breakpoints, table.breakpoints, sizeof(out_pkt->breakpoints));
    out_pkt->tail.lo64 = tail_lo64;
    out_pkt->tail.hi20 = 0u;
    return true;
}

bool encode_op640_packet(const SplinePacket0x640& pkt,
                         uint8_t out_bytes[kOp640PacketBytes],
                         std::string* err) {
    if (out_bytes == nullptr) {
        return set_err(err, "out_bytes must not be null");
    }
    if (op640_header_opcode(pkt.header) != kOp640Opcode) {
        return set_err(err, "SplinePacket0x640 header opcode must be 0x640");
    }
    if (pkt.header.hi12 > 0xFFFu) {
        return set_err(err, "SplinePacket0x640 header.hi12 exceeds 12 bits");
    }
    if (pkt.tail.hi20 > 0xFFFFFu) {
        return set_err(err, "SplinePacket0x640 tail.hi20 exceeds 20 bits");
    }

    std::memset(out_bytes, 0, kOp640PacketBytes);

    // [63:0] and [75:64] header (76 bits)
    write_bits_le(out_bytes, kOp640PacketBytes, 0u, 64u, pkt.header.lo64);
    write_bits_le(out_bytes, kOp640PacketBytes, 64u, 12u, static_cast<uint64_t>(pkt.header.hi12));

    // [1355:76] coeffs[8][5] (40 x 32b = 1280 bits)
    uint32_t bit_pos = 76u;
    for (uint32_t s = 0u; s < kOp640NumSegments; ++s) {
        for (uint32_t k = 0u; k < kOp640CoeffsPerSegment; ++k) {
            write_bits_le(out_bytes, kOp640PacketBytes, bit_pos, 32u,
                          static_cast<uint64_t>(f32_to_u32(pkt.coeffs[s][k])));
            bit_pos += 32u;
        }
    }

    // [1579:1356] breakpoints[7] (7 x 32b = 224 bits)
    for (uint32_t i = 0u; i < kOp640NumBreakpoints; ++i) {
        write_bits_le(out_bytes, kOp640PacketBytes, bit_pos, 32u,
                      static_cast<uint64_t>(f32_to_u32(pkt.breakpoints[i])));
        bit_pos += 32u;
    }

    // [1663:1580] tail (84 bits = 64 + 20)
    write_bits_le(out_bytes, kOp640PacketBytes, 1580u, 64u, pkt.tail.lo64);
    write_bits_le(out_bytes, kOp640PacketBytes, 1644u, 20u, static_cast<uint64_t>(pkt.tail.hi20));
    return true;
}

bool decode_op640_packet(const uint8_t in_bytes[kOp640PacketBytes],
                         SplinePacket0x640* out_pkt,
                         std::string* err) {
    if (in_bytes == nullptr || out_pkt == nullptr) {
        return set_err(err, "in_bytes and out_pkt must not be null");
    }

    const uint64_t hdr_lo = read_bits_le(in_bytes, kOp640PacketBytes, 0u, 64u);
    if ((hdr_lo & 0xFFFULL) != static_cast<uint64_t>(kOp640Opcode)) {
        return set_err(err, "0x640 packet opcode mismatch (expected 0x640)");
    }

    out_pkt->header.lo64 = hdr_lo;
    out_pkt->header.hi12 = static_cast<uint16_t>(read_bits_le(in_bytes, kOp640PacketBytes, 64u, 12u));

    uint32_t bit_pos = 76u;
    for (uint32_t s = 0u; s < kOp640NumSegments; ++s) {
        for (uint32_t k = 0u; k < kOp640CoeffsPerSegment; ++k) {
            const uint32_t u = static_cast<uint32_t>(
                read_bits_le(in_bytes, kOp640PacketBytes, bit_pos, 32u));
            out_pkt->coeffs[s][k] = u32_to_f32(u);
            bit_pos += 32u;
        }
    }

    for (uint32_t i = 0u; i < kOp640NumBreakpoints; ++i) {
        const uint32_t u = static_cast<uint32_t>(
            read_bits_le(in_bytes, kOp640PacketBytes, bit_pos, 32u));
        out_pkt->breakpoints[i] = u32_to_f32(u);
        bit_pos += 32u;
    }

    out_pkt->tail.lo64 = read_bits_le(in_bytes, kOp640PacketBytes, 1580u, 64u);
    out_pkt->tail.hi20 = static_cast<uint32_t>(read_bits_le(in_bytes, kOp640PacketBytes, 1644u, 20u));
    return true;
}

float eval_op640_horner_f32(float x_in,
                            const SplineTable0x640& table,
                            float fclip_min,
                            float fclip_max) {
    return eval_horner_core_f32(x_in, table.coeffs, table.breakpoints, fclip_min, fclip_max);
}

float eval_op640_horner_f32(float x_in,
                            const SplinePacket0x640& pkt,
                            float fclip_min,
                            float fclip_max) {
    return eval_horner_core_f32(x_in, pkt.coeffs, pkt.breakpoints, fclip_min, fclip_max);
}

void eval_op640_horner_batch_f32(const float* in,
                                 float* out,
                                 size_t count,
                                 const SplineTable0x640& table,
                                 float fclip_min,
                                 float fclip_max) {
    if (in == nullptr || out == nullptr) {
        return;
    }
    for (size_t i = 0u; i < count; ++i) {
        out[i] = eval_horner_core_f32(in[i], table.coeffs, table.breakpoints, fclip_min, fclip_max);
    }
}

float rsqrt_newton_raphson_step(float x, float y_seed) {
    if (std::isnan(x) || std::isnan(y_seed)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    // Order as (x * y) * y to minimize intermediate overflow at extreme magnitudes.
    const float xy = x * y_seed;
    const float half_xy2 = 0.5f * (xy * y_seed);
    return y_seed * (1.5f - half_xy2);
}

float recip_newton_raphson_step(float x, float y_seed) {
    if (std::isnan(x) || std::isnan(y_seed)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return y_seed * (2.0f - x * y_seed);
}

float eval_buenos_exp_shifted(float x, bool use_neg11_domain) {
    if (std::isnan(x)) {
        return x;
    }
    const SplineTable0x640& tbl = use_neg11_domain ? kExpNeg11Table : kExpNeg12_96Table;
    if (x < tbl.domain_min) {
        return 0.0f;
    }
    const float y = eval_op640_horner_f32(x, tbl, tbl.domain_min, tbl.domain_max);
    return (y < 0.0f) ? 0.0f : y;
}

float eval_buenos_rsqrt(float x, bool nr_refine) {
    if (std::isnan(x) || x < 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (x == 0.0f) {
        return std::signbit(x) ? -std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::infinity();
    }
    if (std::isinf(x)) {
        return 0.0f;
    }

    // Range-reduce x = u * 2^(2k) with u in [1.0, 4.0).
    // std::frexpf returns m in [0.5, 1.0) and e such that x = m * 2^e.
    int e = 0;
    float m = std::frexp(x, &e);
    // Shift to [1.0, 2.0): x = (2m) * 2^(e - 1).
    float u = m * 2.0f;
    int exp_unbiased = e - 1;
    // Make exponent even: if odd, multiply u by 2 -> u in [2.0, 4.0) and decrement exp.
    if ((exp_unbiased & 1) != 0) {
        u *= 2.0f;
        exp_unbiased -= 1;
    }
    const int half_exp = exp_unbiased / 2;

    float y_u = eval_op640_horner_f32(u, kRsqrtMantissaTable, 1.0f, 4.0f);
    if (nr_refine) {
        y_u = rsqrt_newton_raphson_step(u, y_u);
    }
    return std::ldexp(y_u, -half_exp);
}

float eval_buenos_recip(float x, bool nr_refine) {
    if (std::isnan(x)) {
        return x;
    }
    if (x == 0.0f) {
        return std::copysign(std::numeric_limits<float>::infinity(), x);
    }
    if (std::isinf(x)) {
        return std::copysign(0.0f, x);
    }

    const float sign = std::signbit(x) ? -1.0f : 1.0f;
    const float ax = std::fabs(x);

    // Range-reduce |x| = u * 2^exp_unbiased with u in [1.0, 2.0).
    int e = 0;
    float m = std::frexp(ax, &e);
    const float u = m * 2.0f;
    const int exp_unbiased = e - 1;

    float y_u = eval_op640_horner_f32(u, kRecipMantissaTable, 1.0f, 2.0f);
    if (nr_refine) {
        y_u = recip_newton_raphson_step(u, y_u);
    }
    return sign * std::ldexp(y_u, -exp_unbiased);
}

float eval_buenos_silu(float x) {
    if (std::isnan(x)) {
        return x;
    }
    if (x < kSiluTable.domain_min) {
        return 0.0f;
    }
    if (x > kSiluTable.domain_max) {
        return x;
    }
    return eval_op640_horner_f32(x, kSiluTable, kSiluTable.domain_min, kSiluTable.domain_max);
}

float eval_buenos_gelu_tanh(float x) {
    if (std::isnan(x)) {
        return x;
    }
    if (x < kGeluTanhTable.domain_min) {
        return 0.0f;
    }
    if (x > kGeluTanhTable.domain_max) {
        return x;
    }
    return eval_op640_horner_f32(x, kGeluTanhTable, kGeluTanhTable.domain_min,
                                  kGeluTanhTable.domain_max);
}

float eval_buenos_gelu_erf(float x) {
    if (std::isnan(x)) {
        return x;
    }
    if (x < kGeluErfTable.domain_min) {
        return 0.0f;
    }
    if (x > kGeluErfTable.domain_max) {
        return x;
    }
    return eval_op640_horner_f32(x, kGeluErfTable, kGeluErfTable.domain_min,
                                 kGeluErfTable.domain_max);
}

}  // namespace freetile
