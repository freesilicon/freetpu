// SPDX-License-Identifier: Apache-2.0
// Exhaustive offline verification for Slice S4:
//   1. 0x640 Minimax Degree-4 Horner Spline Library (buenos_splines.*)
//      - >= 100,000-point sweeps for exp [-12.96, 0], exp [-11, 0], rsqrt, recip, SiLU, GELU
//      - Bit-exact 208-byte (1664-bit) 0x640 pack/unpack round-trip & invalid input checks
//      - NaN, +/-Inf, +/-0, subnormal, breakpoint-boundary, and domain-clamp edge cases
//   2. 768 KiB Narrow-SRAM Double-Buffered FlashAttention / FlashDecoding Planner
//      - Exact byte-count verification for all canonical prefill & GQA decode geometries
//      - Off-by-one ceiling proof (max_bc fits <= 786,432 B; max_bc + 1 exceeds 786,432 B)
//      - Block-tiled DMA-BUF KV page geometry (Q8_0 optimal vs Q4_0 96:1 stall rejection)
//      - Host-to-NPU VII dispatch break-even estimator (~300 us VII vs ~20 us NEON)
//   3. Pixel 11 (malibu / Tensor G6 santafe) Hardware & Phase-Aware Hybrid Profile
//      - Empirical hardware descriptor assertions & Day-1 safe supports_s4_native = false
//      - SME2 thread cap clamping, PSTATE.SM syscall barrier, BDA & workgroup serialization,
//        and zero concurrent LPDDR5X saturation enforcement.

#include "freetile/buenos_splines.hpp"
#include "freetile/flash_attn_planner.hpp"
#include "freetile/malibu_profile.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define CHECK_TRUE(cond, msg)                                                                  \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));             \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

namespace {

constexpr size_t kDensePoints = 100001u;

double ref_silu(double x) {
    if (x >= 0.0) {
        return x / (1.0 + std::exp(-x));
    }
    const double ex = std::exp(x);
    return x * (ex / (1.0 + ex));
}

double ref_gelu_tanh(double x) {
    constexpr double kSqrt2OverPi = 0.79788456080286535588;
    return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x)));
}

double ref_gelu_erf(double x) {
    constexpr double kInvSqrt2 = 0.70710678118654752440;
    return 0.5 * x * (1.0 + std::erf(x * kInvSqrt2));
}

int test_0x640_codec_and_validation() {
    using namespace freetile;

    const SplineKind all_kinds[] = {
        SplineKind::EXP_NEG12_96_0,
        SplineKind::EXP_NEG11_0,
        SplineKind::RSQRT_MANTISSA,
        SplineKind::RECIP_MANTISSA,
        SplineKind::SILU,
        SplineKind::GELU_TANH,
        SplineKind::GELU_ERF,
    };

    for (const SplineKind kind : all_kinds) {
        const SplineTable0x640& tbl = buenos_spline_table(kind);
        std::string err;
        CHECK_TRUE(validate_spline_table(tbl, &err), err.c_str());

        SplinePacket0x640 pkt;
        const uint16_t core_mask = 0xFFFFu;
        const uint16_t seq = static_cast<uint16_t>(51u + static_cast<uint16_t>(kind));
        const uint16_t flags12 = 0xA5Au;
        CHECK_TRUE(make_op640_packet(tbl, &pkt, core_mask, seq, flags12, kOp640TailDeg4, &err),
                   err.c_str());
        pkt.tail.hi20 = 0xABCDEu;

        uint8_t wire[kOp640PacketBytes];
        CHECK_TRUE(encode_op640_packet(pkt, wire, &err), err.c_str());

        // Verify raw bit positions: opcode [11:0] == 0x640, core_mask [27:12] == 0xFFFF
        const uint16_t lo16 = static_cast<uint16_t>(wire[0]) |
                              static_cast<uint16_t>(static_cast<uint16_t>(wire[1]) << 8);
        CHECK_TRUE((lo16 & 0xFFFu) == kOp640Opcode, "Wire opcode [11:0] == 0x640");

        SplinePacket0x640 decoded;
        CHECK_TRUE(decode_op640_packet(wire, &decoded, &err), err.c_str());
        CHECK_TRUE(decoded.header.lo64 == pkt.header.lo64, "Header lo64 round-trip exact");
        CHECK_TRUE(decoded.header.hi12 == pkt.header.hi12, "Header hi12 round-trip exact");
        CHECK_TRUE(op640_header_opcode(decoded.header) == kOp640Opcode, "Decoded opcode");
        CHECK_TRUE(op640_header_core_mask(decoded.header) == core_mask, "Decoded core_mask");
        CHECK_TRUE(op640_header_seq(decoded.header) == seq, "Decoded seq");
        CHECK_TRUE(op640_header_flags(decoded.header) == flags12, "Decoded flags12");
        CHECK_TRUE(decoded.tail.lo64 == pkt.tail.lo64, "Tail lo64 round-trip exact");
        CHECK_TRUE(decoded.tail.hi20 == pkt.tail.hi20, "Tail hi20 round-trip exact");

        uint8_t reencoded[kOp640PacketBytes];
        CHECK_TRUE(encode_op640_packet(decoded, reencoded, &err), err.c_str());
        CHECK_TRUE(std::memcmp(wire, reencoded, kOp640PacketBytes) == 0,
                   "208-byte 0x640 encode->decode->encode bit-exact");

        // Verify every float32 coefficient and breakpoint bit-exact
        CHECK_TRUE(std::memcmp(decoded.coeffs, pkt.coeffs, sizeof(pkt.coeffs)) == 0,
                   "All 40 float32 coeffs bit-exact");
        CHECK_TRUE(std::memcmp(decoded.breakpoints, pkt.breakpoints, sizeof(pkt.breakpoints)) == 0,
                   "All 7 float32 breakpoints bit-exact");
    }

    // Negative / invalid packet checks
    {
        uint8_t bad_wire[kOp640PacketBytes];
        std::memset(bad_wire, 0, sizeof(bad_wire));
        bad_wire[0] = 0x00u;
        bad_wire[1] = 0x02u;  // 0x200, not 0x640
        SplinePacket0x640 out;
        std::string err;
        CHECK_TRUE(!decode_op640_packet(bad_wire, &out, &err), "Reject non-0x640 opcode on decode");

        SplinePacket0x640 bad_pkt;
        bad_pkt.header.lo64 = 0x200ULL;
        CHECK_TRUE(!encode_op640_packet(bad_pkt, bad_wire, &err), "Reject non-0x640 opcode on encode");

        bad_pkt.header = make_op640_header();
        bad_pkt.header.hi12 = 0x1000u;  // > 12 bits
        CHECK_TRUE(!encode_op640_packet(bad_pkt, bad_wire, &err), "Reject >12-bit header flags");

        bad_pkt.header.hi12 = 0u;
        bad_pkt.tail.hi20 = 0x100000u;  // > 20 bits
        CHECK_TRUE(!encode_op640_packet(bad_pkt, bad_wire, &err), "Reject >20-bit tail hi20");

        SplineTable0x640 bad_tbl = buenos_spline_exp_neg12_96();
        bad_tbl.breakpoints[3] = bad_tbl.breakpoints[2];  // non-strictly ascending
        CHECK_TRUE(!validate_spline_table(bad_tbl, &err), "Reject duplicate breakpoints");
        bad_tbl = buenos_spline_exp_neg12_96();
        bad_tbl.coeffs[2][1] = std::numeric_limits<float>::quiet_NaN();
        CHECK_TRUE(!validate_spline_table(bad_tbl, &err), "Reject NaN coefficient");
    }

    // Affine quantization fusion test for exp(x) -> u = 256*exp(x) in [0, 255]
    {
        const SplineTable0x640 fused =
            affine_fuse_spline_table(buenos_spline_exp_neg12_96(), 256.0f, 0.0f);
        double max_lsb_err = 0.0;
        for (int q = -128; q <= 127; ++q) {
            const float x = static_cast<float>(q) * 0.125f;
            const float u_hat = eval_op640_horner_f32(x, fused, -12.96f, 0.0f);
            const double xc = (x < -12.96) ? -12.96 : ((x > 0.0) ? 0.0 : static_cast<double>(x));
            const double u_ref = 256.0 * std::exp(xc);
            const double err = std::fabs(static_cast<double>(u_hat) - u_ref);
            if (err > max_lsb_err) {
                max_lsb_err = err;
            }
        }
        CHECK_TRUE(max_lsb_err < 0.002, "Affine-fused exp(x) quantized LSB error < 0.002 LSB");
    }

    return 0;
}

int test_exp_splines_100k() {
    using namespace freetile;

    // 1. exp(x) on [-12.96, 0.0] (>= 100K points)
    {
        const SplineTable0x640& tbl = buenos_spline_exp_neg12_96();
        double max_err = 0.0;
        for (size_t i = 0u; i < kDensePoints; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(kDensePoints - 1u);
            const float x = static_cast<float>(-12.96 + 12.96 * t);
            const float y = eval_buenos_exp_shifted(x, false);
            const double err = std::fabs(static_cast<double>(y) - std::exp(static_cast<double>(x)));
            if (err > max_err) {
                max_err = err;
            }
        }
        std::printf("  [exp -12.96..0] 100,001 pts max_abs_err = %.6e\n", max_err);
        CHECK_TRUE(max_err <= 1.0e-5, "exp [-12.96, 0] max_abs_err <= 1e-5");
        CHECK_TRUE(max_err <= 3.815e-6, "exp [-12.96, 0] matches buenos_a0 exp_spline.pb <= 3.815e-6");

        // Breakpoint exact boundary & left/right neighbor continuity
        for (uint32_t b = 0u; b < kOp640NumBreakpoints; ++b) {
            const float bp = tbl.breakpoints[b];
            const float bp_lo = std::nextafter(bp, -13.0f);
            const float bp_hi = std::nextafter(bp, 1.0f);
            for (const float x : {bp_lo, bp, bp_hi}) {
                const double err =
                    std::fabs(static_cast<double>(eval_buenos_exp_shifted(x, false)) -
                              std::exp(static_cast<double>(x)));
                CHECK_TRUE(err <= 1.0e-5, "exp breakpoint boundary accuracy");
            }
        }

        // Edge cases: NaN, -inf, +inf, < -12.96, > 0
        CHECK_TRUE(std::isnan(eval_buenos_exp_shifted(std::numeric_limits<float>::quiet_NaN())),
                   "exp(NaN) == NaN");
        CHECK_TRUE(eval_buenos_exp_shifted(-std::numeric_limits<float>::infinity()) == 0.0f,
                   "exp(-inf) == 0");
        CHECK_TRUE(eval_buenos_exp_shifted(-100.0f) == 0.0f, "exp(-100) clamps to 0");
        CHECK_TRUE(std::fabs(eval_buenos_exp_shifted(0.0f) - 1.0f) <= 3.0e-6f, "exp(0) ~= 1");
        CHECK_TRUE(std::fabs(eval_buenos_exp_shifted(5.0f) - 1.0f) <= 3.0e-6f,
                   "shifted exp(x > 0) clamps to 0 -> ~= 1");
        CHECK_TRUE(std::fabs(eval_buenos_exp_shifted(std::numeric_limits<float>::infinity()) - 1.0f) <=
                       3.0e-6f,
                   "shifted exp(+inf) clamps to 0 -> ~= 1");
    }

    // 2. exp(x) on [-11.0, 0.0] (>= 100K points)
    {
        double max_err = 0.0;
        for (size_t i = 0u; i < kDensePoints; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(kDensePoints - 1u);
            const float x = static_cast<float>(-11.0 + 11.0 * t);
            const float y = eval_buenos_exp_shifted(x, true);
            const double err = std::fabs(static_cast<double>(y) - std::exp(static_cast<double>(x)));
            if (err > max_err) {
                max_err = err;
            }
        }
        std::printf("  [exp -11.0..0]  100,001 pts max_abs_err = %.6e\n", max_err);
        CHECK_TRUE(max_err <= 1.0e-5, "exp [-11, 0] max_abs_err <= 1e-5");
        CHECK_TRUE(eval_buenos_exp_shifted(-12.0f, true) == 0.0f, "exp_neg11(-12) == 0");
    }

    return 0;
}

int test_rsqrt_and_recip_100k() {
    using namespace freetile;

    // 1. rsqrt(x): >= 100K log-spaced points spanning 1e-30 .. 1e+30 + mantissa [1, 4]
    {
        double max_rel_seed = 0.0;
        double max_rel_nr = 0.0;
        for (size_t i = 0u; i < kDensePoints; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(kDensePoints - 1u);
            const double log10_x = -30.0 + 60.0 * t;
            const float x = static_cast<float>(std::pow(10.0, log10_x));
            const double exact = 1.0 / std::sqrt(static_cast<double>(x));
            const float y_seed = eval_buenos_rsqrt(x, false);
            const float y_nr = eval_buenos_rsqrt(x, true);
            const double rel_seed = std::fabs(static_cast<double>(y_seed) - exact) / exact;
            const double rel_nr = std::fabs(static_cast<double>(y_nr) - exact) / exact;
            if (rel_seed > max_rel_seed) max_rel_seed = rel_seed;
            if (rel_nr > max_rel_nr) max_rel_nr = rel_nr;
        }
        std::printf("  [rsqrt 1e-30..1e30] 100,001 pts max_rel_err: seed=%.6e nr=%.6e\n",
                    max_rel_seed, max_rel_nr);
        CHECK_TRUE(max_rel_seed <= 1.0e-6, "rsqrt seed max_rel_err <= 1e-6");
        CHECK_TRUE(max_rel_nr <= 2.5e-7, "rsqrt + 1x NR max_rel_err <= 2.5e-7");

        // Subnormal, FLT_MIN, FLT_MAX, and special values
        const float sub = std::numeric_limits<float>::denorm_min();
        const float y_sub = eval_buenos_rsqrt(sub, true);
        const double sub_exact = 1.0 / std::sqrt(static_cast<double>(sub));
        CHECK_TRUE(std::isfinite(y_sub) &&
                       (std::fabs(static_cast<double>(y_sub) - sub_exact) / sub_exact) < 5.0e-7,
                   "rsqrt(denorm_min) accurate and finite");

        const float y_fmax = eval_buenos_rsqrt(std::numeric_limits<float>::max(), true);
        const double fmax_exact = 1.0 / std::sqrt(static_cast<double>(std::numeric_limits<float>::max()));
        CHECK_TRUE(std::isfinite(y_fmax) &&
                       (std::fabs(static_cast<double>(y_fmax) - fmax_exact) / fmax_exact) < 5.0e-7,
                   "rsqrt(FLT_MAX) accurate and finite");

        CHECK_TRUE(std::isnan(eval_buenos_rsqrt(std::numeric_limits<float>::quiet_NaN())),
                   "rsqrt(NaN) == NaN");
        CHECK_TRUE(std::isnan(eval_buenos_rsqrt(-1.0f)), "rsqrt(-1) == NaN");
        CHECK_TRUE(std::isnan(eval_buenos_rsqrt(-std::numeric_limits<float>::infinity())),
                   "rsqrt(-inf) == NaN");
        CHECK_TRUE(std::isinf(eval_buenos_rsqrt(0.0f)) && !std::signbit(eval_buenos_rsqrt(0.0f)),
                   "rsqrt(+0) == +inf");
        CHECK_TRUE(eval_buenos_rsqrt(std::numeric_limits<float>::infinity()) == 0.0f,
                   "rsqrt(+inf) == 0");

        // Direct hardware NR helper check
        const float y0 = 0.24f;  // ~4% perturbed seed for rsqrt(16) = 0.25
        const float y1 = rsqrt_newton_raphson_step(16.0f, y0);
        CHECK_TRUE(std::fabs(y1 - 0.25f) < 0.001f, "rsqrt_newton_raphson_step converges quadratically");
    }

    // 2. recip(x): >= 100K points spanning FlashAttention l_new [1..2048] + signed 1e-30..1e30
    {
        double max_rel_seed = 0.0;
        double max_rel_nr = 0.0;
        for (size_t i = 0u; i < kDensePoints; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(kDensePoints - 1u);
            const double mag = std::pow(10.0, -30.0 + 60.0 * t);
            const float x = static_cast<float>(((i & 1u) == 0u) ? mag : -mag);
            const double exact = 1.0 / static_cast<double>(x);
            const float y_seed = eval_buenos_recip(x, false);
            const float y_nr = eval_buenos_recip(x, true);
            const double rel_seed = std::fabs(static_cast<double>(y_seed) - exact) / std::fabs(exact);
            const double rel_nr = std::fabs(static_cast<double>(y_nr) - exact) / std::fabs(exact);
            if (rel_seed > max_rel_seed) max_rel_seed = rel_seed;
            if (rel_nr > max_rel_nr) max_rel_nr = rel_nr;
        }
        std::printf("  [recip +-1e-30..1e30] 100,001 pts max_rel_err: seed=%.6e nr=%.6e\n",
                    max_rel_seed, max_rel_nr);
        CHECK_TRUE(max_rel_seed <= 2.0e-6, "recip seed max_rel_err <= 2e-6");
        CHECK_TRUE(max_rel_nr <= 2.5e-7, "recip + 1x NR max_rel_err <= 2.5e-7");

        // FlashAttention softmax denominator range l_new in [1.0, 1408.0]
        double max_fa_rel = 0.0;
        for (size_t i = 0u; i < 10001u; ++i) {
            const float l_new = 1.0f + 1407.0f * static_cast<float>(i) / 10000.0f;
            const float inv_l = eval_buenos_recip(l_new, true);
            const double exact = 1.0 / static_cast<double>(l_new);
            const double rel = std::fabs(static_cast<double>(inv_l) - exact) / exact;
            if (rel > max_fa_rel) max_fa_rel = rel;
        }
        CHECK_TRUE(max_fa_rel <= 2.5e-7, "FlashAttention 1/l_new max_rel_err <= 2.5e-7");

        // Edge cases
        CHECK_TRUE(std::isnan(eval_buenos_recip(std::numeric_limits<float>::quiet_NaN())),
                   "recip(NaN) == NaN");
        CHECK_TRUE(std::isinf(eval_buenos_recip(0.0f)) && !std::signbit(eval_buenos_recip(0.0f)),
                   "recip(+0) == +inf");
        CHECK_TRUE(std::isinf(eval_buenos_recip(-0.0f)) && std::signbit(eval_buenos_recip(-0.0f)),
                   "recip(-0) == -inf");
        CHECK_TRUE(eval_buenos_recip(std::numeric_limits<float>::infinity()) == 0.0f &&
                       !std::signbit(eval_buenos_recip(std::numeric_limits<float>::infinity())),
                   "recip(+inf) == +0");
        CHECK_TRUE(eval_buenos_recip(-std::numeric_limits<float>::infinity()) == 0.0f &&
                       std::signbit(eval_buenos_recip(-std::numeric_limits<float>::infinity())),
                   "recip(-inf) == -0");

        const float y0 = 0.12f;  // perturbed seed for 1/8 = 0.125
        const float y1 = recip_newton_raphson_step(8.0f, y0);
        CHECK_TRUE(std::fabs(y1 - 0.125f) < 0.0005f, "recip_newton_raphson_step converges");
    }

    return 0;
}

int test_silu_and_gelu_100k() {
    using namespace freetile;

    double max_silu = 0.0;
    double max_gelu_t = 0.0;
    double max_gelu_e = 0.0;
    std::vector<float> xs(kDensePoints);
    std::vector<float> ys_batch(kDensePoints);
    for (size_t i = 0u; i < kDensePoints; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(kDensePoints - 1u);
        xs[i] = static_cast<float>(-11.1 + 22.2 * t);
    }
    eval_op640_horner_batch_f32(xs.data(), ys_batch.data(), xs.size(),
                                buenos_spline_silu(), -11.1f, 11.1f);

    for (size_t i = 0u; i < kDensePoints; ++i) {
        const float x = xs[i];
        const double xd = static_cast<double>(x);
        const float ys = eval_buenos_silu(x);
        CHECK_TRUE(ys == ys_batch[i], "Batch Horner matches scalar Horner bit-for-bit");
        const double es = std::fabs(static_cast<double>(ys) - ref_silu(xd));
        const double egt = std::fabs(static_cast<double>(eval_buenos_gelu_tanh(x)) - ref_gelu_tanh(xd));
        const double ege = std::fabs(static_cast<double>(eval_buenos_gelu_erf(x)) - ref_gelu_erf(xd));
        if (es > max_silu) max_silu = es;
        if (egt > max_gelu_t) max_gelu_t = egt;
        if (ege > max_gelu_e) max_gelu_e = ege;
    }
    std::printf("  [SiLU/GELU -11.1..11.1] 100,001 pts max_abs_err: silu=%.6e gelu_tanh=%.6e gelu_erf=%.6e\n",
                max_silu, max_gelu_t, max_gelu_e);
    CHECK_TRUE(max_silu <= 6.104e-5, "SiLU tighter than vendor swish_spline.pb");
    CHECK_TRUE(max_gelu_t <= 5.0e-5, "GELU(tanh) max_abs_err <= 5e-5");
    CHECK_TRUE(max_gelu_e <= 6.0e-5, "GELU(erf) max_abs_err <= 6e-5");

    // NaN & asymptotic tails outside [-11.1, +11.1]
    CHECK_TRUE(std::isnan(eval_buenos_silu(std::numeric_limits<float>::quiet_NaN())), "silu(NaN)");
    CHECK_TRUE(std::isnan(eval_buenos_gelu_tanh(std::numeric_limits<float>::quiet_NaN())), "gelu_t(NaN)");
    CHECK_TRUE(std::isnan(eval_buenos_gelu_erf(std::numeric_limits<float>::quiet_NaN())), "gelu_e(NaN)");
    CHECK_TRUE(eval_buenos_silu(-25.0f) == 0.0f && eval_buenos_silu(25.0f) == 25.0f, "silu tails");
    CHECK_TRUE(eval_buenos_gelu_tanh(-25.0f) == 0.0f && eval_buenos_gelu_tanh(25.0f) == 25.0f, "gelu tails");
    CHECK_TRUE(std::isnan(eval_op640_horner_f32(0.0f, buenos_spline_silu(), 5.0f, -5.0f)),
               "Inverted clip bounds return NaN");
    return 0;
}

int test_flash_attn_sram_planner() {
    using namespace freetile;

    // 1. Hardware M alignment (hw_m in {1, 8, 16, ...})
    CHECK_TRUE(buenos_align_hw_m(1u) == 1u, "hw_m(1) == 1");
    CHECK_TRUE(buenos_align_hw_m(4u) == 8u, "hw_m(4) == 8");
    CHECK_TRUE(buenos_align_hw_m(8u) == 8u, "hw_m(8) == 8 (GQA 8:1 exact match)");
    CHECK_TRUE(buenos_align_hw_m(9u) == 16u, "hw_m(9) == 16");
    CHECK_TRUE(buenos_align_hw_m(16u) == 16u, "hw_m(16) == 16");
    CHECK_TRUE(buenos_align_hw_m(64u) == 64u, "hw_m(64) == 64");
    CHECK_TRUE(buenos_align_hw_m(128u) == 128u, "hw_m(128) == 128");

    // 2. Canonical Prefill & Decode geometries matching exact 768 KiB byte footprints
    struct Case {
        const char* label;
        FlashAttnPhase phase;
        uint32_t d;
        uint32_t qh;
        uint32_t kvh;
        uint32_t exp_br;
        uint32_t exp_bc;
        size_t exp_bytes;
        uint32_t exp_hw_m;
    };
    const Case cases[] = {
        {"Prefill d=64",       FlashAttnPhase::PREFILL, 64u,  32u, 4u, 128u, 512u,  591872u, 128u},
        {"Prefill d=128",      FlashAttnPhase::PREFILL, 128u, 32u, 4u, 64u,  512u,  721920u, 64u},
        {"Decode GQA 8:1 d=64",  FlashAttnPhase::DECODE, 64u,  32u, 4u, 8u,  1408u, 770176u, 8u},
        {"Decode GQA 8:1 d=128", FlashAttnPhase::DECODE, 128u, 32u, 4u, 8u,  512u,  548992u, 8u},
        {"Decode GQA 4:1 d=128", FlashAttnPhase::DECODE, 128u, 32u, 8u, 4u,  704u,  736320u, 8u},
        {"Decode GQA 4:1 d=64",  FlashAttnPhase::DECODE, 64u,  32u, 8u, 4u,  1472u, 779328u, 8u},
    };

    for (const Case& c : cases) {
        FlashAttnTileRequest req;
        req.phase = c.phase;
        req.head_dim = c.d;
        req.num_q_heads = c.qh;
        req.num_kv_heads = c.kvh;
        req.seq_len = 4096u;
        req.kv_tile_align = 64u;
        req.dtype = FlashAttnDType::FP16;
        req.buffer_policy = FlashAttnBufferPolicy::PING_PONG_FULL;
        req.sram_budget_bytes = kBuenosCoreNarrowMemBytes;

        FlashAttnTilePlan plan;
        std::string err;
        CHECK_TRUE(plan_flash_attn_tile(req, &plan, &err), err.c_str());
        CHECK_TRUE(plan.valid, c.label);
        CHECK_TRUE(plan.br == c.exp_br, c.label);
        CHECK_TRUE(plan.bc == c.exp_bc, c.label);
        CHECK_TRUE(plan.hw_m == c.exp_hw_m, c.label);
        CHECK_TRUE(plan.sram.total_bytes == c.exp_bytes, c.label);
        CHECK_TRUE(plan.sram.total_bytes <= kBuenosCoreNarrowMemBytes, "SRAM ceiling <= 786,432 B");
        CHECK_TRUE(plan.sram.fits_sram, "fits_sram == true");
        CHECK_TRUE(plan.sram.headroom_bytes == (kBuenosCoreNarrowMemBytes - plan.sram.total_bytes),
                   "Exact SRAM headroom");

        // Off-by-one ceiling proof: max_bc_unaligned fits <= 786,432 B, while +1 strictly exceeds!
        FlashAttnSramBreakdown at_max, over_max;
        CHECK_TRUE(compute_flash_attn_sram_footprint(
                       plan.br, plan.max_bc_unaligned, plan.head_dim, req.dtype,
                       req.buffer_policy, kBuenosCoreNarrowMemBytes, &at_max, &err),
                   err.c_str());
        CHECK_TRUE(at_max.fits_sram && at_max.total_bytes <= kBuenosCoreNarrowMemBytes,
                   "max_bc_unaligned fits <= 786,432 B");

        CHECK_TRUE(compute_flash_attn_sram_footprint(
                       plan.br, plan.max_bc_unaligned + 1u, plan.head_dim, req.dtype,
                       req.buffer_policy, kBuenosCoreNarrowMemBytes, &over_max, &err),
                   err.c_str());
        CHECK_TRUE(!over_max.fits_sram && over_max.total_bytes > kBuenosCoreNarrowMemBytes,
                   "max_bc_unaligned + 1 strictly exceeds 786,432 B (zero off-by-one)");
    }

    // Verify exact DR2 mathematical ceilings (1437 for GQA 8:1 d=64, 1485 for GQA 4:1 d=64, 752 for GQA 4:1 d=128)
    CHECK_TRUE(flash_attn_max_bc(8u, 64u, FlashAttnDType::FP16, FlashAttnBufferPolicy::PING_PONG_FULL,
                                 kBuenosCoreNarrowMemBytes, 1u) == 1437u,
               "Exact ceiling B_c=1437 for GQA 8:1 d=64");
    CHECK_TRUE(flash_attn_max_bc(4u, 64u, FlashAttnDType::FP16, FlashAttnBufferPolicy::PING_PONG_FULL,
                                 kBuenosCoreNarrowMemBytes, 1u) == 1485u,
               "Exact ceiling B_c=1485 for GQA 4:1 d=64");
    CHECK_TRUE(flash_attn_max_bc(4u, 128u, FlashAttnDType::FP16, FlashAttnBufferPolicy::PING_PONG_FULL,
                                 kBuenosCoreNarrowMemBytes, 1u) == 752u,
               "Exact ceiling B_c=752 for GQA 4:1 d=128");
    CHECK_TRUE(flash_attn_max_bc(8u, 64u, FlashAttnDType::FP16, FlashAttnBufferPolicy::SINGLE_BUFFERED,
                                 kBuenosCoreNarrowMemBytes, 1u) == 2883u,
               "Single-buffered ceiling B_c=2883 for GQA 8:1 d=64");

    // Verify explicit override exceeding 768 KiB is cleanly rejected
    {
        FlashAttnTileRequest bad_req;
        bad_req.phase = FlashAttnPhase::DECODE;
        bad_req.head_dim = 128u;
        bad_req.num_q_heads = 32u;
        bad_req.num_kv_heads = 8u;
        bad_req.override_br = 4u;
        bad_req.override_bc = 753u;  // 1 past ceiling 752
        FlashAttnTilePlan bad_plan;
        std::string err;
        CHECK_TRUE(!plan_flash_attn_tile(bad_req, &bad_plan, &err),
                   "Reject B_c=753 exceeding 768 KiB ceiling");
        CHECK_TRUE(!bad_plan.valid && !err.empty(), "Rejection provides diagnostic");
    }

    // Verify KV_DOUBLE_BUFFERED_FP32_ACC is strictly <= PING_PONG_FULL for canonical shapes
    {
        FlashAttnSramBreakdown pp, kvdb;
        CHECK_TRUE(compute_flash_attn_sram_footprint(
                       8u, 512u, 128u, FlashAttnDType::FP16, FlashAttnBufferPolicy::PING_PONG_FULL,
                       kBuenosCoreNarrowMemBytes, &pp),
                   "PING_PONG_FULL breakdown");
        CHECK_TRUE(compute_flash_attn_sram_footprint(
                       8u, 512u, 128u, FlashAttnDType::FP16,
                       FlashAttnBufferPolicy::KV_DOUBLE_BUFFERED_FP32_ACC,
                       kBuenosCoreNarrowMemBytes, &kvdb),
                   "KV_DOUBLE_BUFFERED_FP32_ACC breakdown");
        CHECK_TRUE(kvdb.total_bytes <= pp.total_bytes && kvdb.fits_sram,
                   "Resident-Q/FP32-acc KV double-buffering fits within PING_PONG_FULL");
    }

    // 3. Block-tiled DMA-BUF KV page geometry: Q8_0 optimal, Q4_0 rejected (96:1 stall)
    {
        KvCachePageRequest q8_req;
        q8_req.num_layers = 32u;
        q8_req.num_kv_heads = 4u;
        q8_req.head_dim = 128u;
        q8_req.max_seq_len = 4096u;
        q8_req.page_tokens = 64u;
        q8_req.format = KvCacheQuantFormat::Q8_0;
        q8_req.target_has_native_int4_expander = false;

        KvCachePagePlan q8_plan;
        std::string err;
        CHECK_TRUE(plan_kv_cache_dmabuf_pages(q8_req, &q8_plan, &err), err.c_str());
        CHECK_TRUE(q8_plan.accepted && q8_plan.uses_hw_1to2_expander, "Q8_0 uses hardware 1:2 expander");
        CHECK_TRUE(q8_plan.unpack_stall_ratio == 1u, "Q8_0 zero unpack stall");
        CHECK_TRUE(q8_plan.pages_per_head == 64u, "4096 / 64 = 64 pages/head");
        CHECK_TRUE((q8_plan.dmabuf_bytes_per_kv_page_aligned % kDmaBufPageAlignBytes) == 0u,
                   "DMA-BUF page is 4 KiB aligned");
        CHECK_TRUE(q8_plan.bandwidth_savings_vs_f16 > 1.85f && q8_plan.bandwidth_savings_vs_f16 <= 2.0f,
                   "Q8_0 ~1.88x bandwidth savings vs F16");

        KvCachePageRequest q4_req = q8_req;
        q4_req.format = KvCacheQuantFormat::Q4_0;
        KvCachePagePlan q4_plan;
        CHECK_TRUE(!plan_kv_cache_dmabuf_pages(q4_req, &q4_plan, &err),
                   "Q4_0 KV inside NPU without native 1:4 expander rejected");
        CHECK_TRUE(!q4_plan.accepted && q4_plan.unpack_stall_ratio == 96u,
                   "Q4_0 rejection reports 96:1 VLIW unpack stall");
    }

    // 4. Host-to-NPU VII dispatch break-even estimator (~300 us VII vs ~20 us NEON)
    {
        DispatchBreakEvenRequest dreq;
        dreq.num_layers = 32u;
        dreq.hybrid_dispatches_per_layer = 3u;
        dreq.fused_dispatches_per_layer = 1u;
        dreq.cpu_vector_ops_per_layer = 2u;
        dreq.vii_dispatch_us = 300.0;
        dreq.neon_op_us = 20.0;
        dreq.target_token_budget_ms = 50.0;
        dreq.seq_len = 512u;

        DispatchBreakEvenResult dres;
        std::string err;
        CHECK_TRUE(estimate_dispatch_break_even(dreq, &dres, &err), err.c_str());
        CHECK_TRUE(std::fabs(dres.hybrid_per_layer_non_compute_us - 940.0) < 1e-6, "940 us/layer hybrid");
        CHECK_TRUE(std::fabs(dres.hybrid_total_overhead_ms - 30.08) < 1e-6, "30.08 ms total hybrid");
        CHECK_TRUE(std::fabs(dres.fused_total_overhead_ms - 9.60) < 1e-6, "9.60 ms total fused");
        CHECK_TRUE(!dres.hybrid_viable && dres.fused_viable, "Hybrid unviable, fused viable");
        CHECK_TRUE(dres.should_offload_attention_to_tpu, "seq_len=512 offloads attention to TPU");
        CHECK_TRUE(!dres.should_offload_standalone_vector_op,
                   "Never offload standalone 20 us NEON op across 300 us VII");

        dreq.seq_len = 128u;
        CHECK_TRUE(estimate_dispatch_break_even(dreq, &dres, &err), err.c_str());
        CHECK_TRUE(!dres.should_offload_attention_to_tpu,
                   "Short seq_len=128 stays on CPU NEON/SME2");
    }

    return 0;
}

int test_malibu_profile_and_router() {
    using namespace freetile;

    const MalibuHardwareProfile& prof = malibu_hardware_profile();
    CHECK_TRUE(std::strcmp(prof.soc_codename, "malibu") == 0, "soc_codename == malibu");
    CHECK_TRUE(std::strcmp(prof.tpu.codename, "santafe") == 0, "tpu codename == santafe");
    CHECK_TRUE(std::strcmp(prof.tpu.fw_family, "gchips-mbu") == 0, "fw_family == gchips-mbu");
    CHECK_TRUE(prof.cpu.total_cores == 7u && prof.cpu.prime_cores == 1u &&
                   (prof.cpu.mid_cores + prof.cpu.eff_cores) == 6u,
               "7-core 1x prime + 6x pro topology");
    CHECK_TRUE(prof.cpu.prime_part_id == 0x0D8Cu && prof.cpu.pro_part_id == 0x0D8Bu,
               "CPU MIDR part IDs 0xd8c + 0xd8b");
    CHECK_TRUE(prof.cpu.has_sme && prof.cpu.has_sme2 && prof.cpu.syscall_clears_pstate_sm,
               "ARMv9 SME/SME2 + Linux 6.12+ PSTATE.SM syscall-boundary flag");
    CHECK_TRUE(prof.cpu.sme_perf_cpu_mask == 0x7Cu && prof.cpu.default_sme_thread_cap == 5u,
               "SME2 perf mask CPUs 2..6 + thread cap 5");
    CHECK_TRUE(prof.gpu.vendor_id == 0x1010u && prof.gpu.device_id == 0x70058002u &&
                   prof.gpu.subgroup_size == 128u &&
                   prof.gpu.max_storage_buffer_range_bytes == (128ULL * 1024ULL * 1024ULL) &&
                   prof.gpu.supports_bda,
               "PowerVR CXTP-48-1536 MC1 wave128 + 128 MiB + BDA");
    CHECK_TRUE(prof.tpu.dual_heavy_nano && prof.tpu.has_heavy_core && prof.tpu.has_nano_core,
               "Dual Heavy+Nano TPU");
    CHECK_TRUE(!prof.tpu.supports_s4_native && !prof.tpu.supports_hw_huffman &&
                   !prof.tpu.silicon_abi_verified,
               "Day-1 safe supports_s4_native=false & silicon_abi_verified=false");
    CHECK_TRUE(std::fabs(prof.lpddr5x_bw_gbps - 68.2f) < 1e-4f, "68.2 GB/s unified LPDDR5X");

    // TargetProfile skeleton
    const TargetProfile tp = malibu_a0_target_profile();
    CHECK_TRUE(std::strcmp(tp.name, "malibu_a0") == 0 && !tp.supports_mosaic &&
                   !tp.supports_s4_native && tp.vmem_bytes == kBuenosCoreNarrowMemBytes,
               "malibu_a0 TargetProfile skeleton");

    // Thread cap guard against KleidiAI sme_thread_cap silent drop to NEON
    CHECK_TRUE(malibu_clamp_sme2_threads(7u, prof) == 5u, "Clamp 7 -> 5");
    CHECK_TRUE(malibu_clamp_sme2_threads(0u, prof) == 5u, "Default 0 -> 5");
    CHECK_TRUE(malibu_clamp_sme2_threads(4u, prof) == 4u, "Preserve 4 <= 5");

    // Zero concurrent LPDDR5X saturation checks
    CHECK_TRUE(!malibu_allows_concurrent_engines(MalibuComputeEngine::CPU_SME2,
                                                 MalibuComputeEngine::GPU_VULKAN_BDA),
               "Disallow concurrent SME2 + GPU Vulkan");
    CHECK_TRUE(!malibu_allows_concurrent_engines(MalibuComputeEngine::CPU_SME2,
                                                 MalibuComputeEngine::EDGETPU_HEAVY),
               "Disallow concurrent SME2 + Heavy EdgeTPU");
    CHECK_TRUE(!malibu_allows_concurrent_engines(MalibuComputeEngine::EDGETPU_HEAVY,
                                                 MalibuComputeEngine::GPU_VULKAN_BDA),
               "Disallow concurrent Heavy EdgeTPU + GPU Vulkan");
    CHECK_TRUE(malibu_allows_concurrent_engines(MalibuComputeEngine::EDGETPU_NANO,
                                                MalibuComputeEngine::CPU_NEON),
               "Allow Nano-TPU alongside lightweight CPU control");

    // Phase-aware routing tests
    std::string err;
    MalibuRouteDecision dec;

    // 1) pp512 dense prefill -> Darwinn Heavy EdgeTPU
    {
        MalibuRouteRequest req;
        req.phase = MalibuWorkloadPhase::PREFILL_DENSE;
        req.quant = MalibuQuantKind::Q4_0;
        req.batch_m = 512u;
        req.seq_len = 512u;
        req.n_ctx = 4096u;
        req.mesa_pco_compute_ready = false;
        CHECK_TRUE(route_malibu_workload(req, &dec, prof, &err), err.c_str());
        CHECK_TRUE(dec.engine == MalibuComputeEngine::EDGETPU_HEAVY, "pp512 -> EdgeTPU Heavy");
        CHECK_TRUE(!dec.requires_bus_serialization_wait, "Idle bus requires no wait");
    }

    // 2) Long-context prefill when Mesa pco ready -> PowerVR Vulkan BDA + n_ctx > 7168 serialization
    {
        MalibuRouteRequest req;
        req.phase = MalibuWorkloadPhase::PREFILL_DENSE;
        req.batch_m = 512u;
        req.n_ctx = 8192u;
        req.max_tensor_bytes = 256ULL * 1024ULL * 1024ULL;
        req.mesa_pco_compute_ready = true;
        req.active_engine_mask = malibu_engine_bit(MalibuComputeEngine::CPU_SME2);
        CHECK_TRUE(route_malibu_workload(req, &dec, prof, &err), err.c_str());
        CHECK_TRUE(dec.engine == MalibuComputeEngine::GPU_VULKAN_BDA, "Long prefill -> GPU BDA");
        CHECK_TRUE(dec.requires_gpu_bda, "Tensor > 128 MiB enables BDA");
        CHECK_TRUE(dec.requires_gpu_workgroup_serialization, "n_ctx > 7168 serializes workgroups");
        CHECK_TRUE(dec.requires_bus_serialization_wait, "Active SME2 forces LPDDR5X serialization");
        CHECK_TRUE(dec.requires_syscall_barrier, "Transition from SME2 sets PSTATE.SM barrier");
    }

    // 3) M=1 Q4_0 / Q8_0 decode -> SME2 CPU (CPUs 2..6, threads clamped <= 5, zero-repack)
    for (const MalibuQuantKind q : {MalibuQuantKind::Q4_0, MalibuQuantKind::Q8_0}) {
        MalibuRouteRequest req;
        req.phase = MalibuWorkloadPhase::DECODE_M1;
        req.quant = q;
        req.batch_m = 1u;
        req.requested_threads = 7u;
        CHECK_TRUE(route_malibu_workload(req, &dec, prof, &err), err.c_str());
        CHECK_TRUE(dec.engine == MalibuComputeEngine::CPU_SME2, "M=1 Q4_0/Q8_0 -> CPU_SME2");
        CHECK_TRUE(dec.recommended_threads == 5u, "Threads clamped to sme_thread_cap=5");
        CHECK_TRUE(dec.cpu_affinity_mask == 0x7Cu, "Pinned to CPUs 2..6");
        CHECK_TRUE(dec.sme2_native_zero_repack, "Zero-repack KleidiAI SME2 kernel");
        CHECK_TRUE(dec.requires_syscall_barrier, "Linux 6.12+ PSTATE.SM syscall barrier");
    }

    // 4) M=1 IQ4_NL decode -> CPU_NEON (no native SME2 microkernel)
    {
        MalibuRouteRequest req;
        req.phase = MalibuWorkloadPhase::DECODE_M1;
        req.quant = MalibuQuantKind::IQ4_NL;
        req.batch_m = 1u;
        CHECK_TRUE(route_malibu_workload(req, &dec, prof, &err), err.c_str());
        CHECK_TRUE(dec.engine == MalibuComputeEngine::CPU_NEON && !dec.sme2_native_zero_repack,
                   "M=1 IQ4_NL routes to CPU_NEON");
    }

    // 5) Vision embeddings -> Nano-TPU
    {
        MalibuRouteRequest req;
        req.phase = MalibuWorkloadPhase::VISION_EMBEDDING;
        req.batch_m = 1u;
        CHECK_TRUE(route_malibu_workload(req, &dec, prof, &err), err.c_str());
        CHECK_TRUE(dec.engine == MalibuComputeEngine::EDGETPU_NANO, "Vision -> Nano-TPU");
    }

    return 0;
}

}  // namespace

int main() {
    if (test_0x640_codec_and_validation() != 0) return 1;
    if (test_exp_splines_100k() != 0) return 1;
    if (test_rsqrt_and_recip_100k() != 0) return 1;
    if (test_silu_and_gelu_100k() != 0) return 1;
    if (test_flash_attn_sram_planner() != 0) return 1;
    if (test_malibu_profile_and_router() != 0) return 1;

    std::printf("PASS: all S4 splines, FlashAttention SRAM planner, and Malibu profile tests succeeded!\n");
    return 0;
}
