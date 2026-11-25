// SPDX-License-Identifier: Apache-2.0
// Clean-room zero-dependency C++17 Tensor G5 (buenos_a0) 0x640 Minimax Spline Library.
//
// Provides:
//   - Exact 208-byte (1664-bit) LoadCoefficientTables (0x640) wire encoder/decoder
//     ([75:0] header, [1355:76] coeffs[8][5], [1579:1356] bp[7], [1663:1580] tail).
//   - Exact IEEE-754 FP32 RNE Horner evaluator with NaN/Inf/domain-clamp safety.
//   - Clean-room 8-segment degree-4 minimax coefficient tables for:
//       * exp(x) on [-12.96, 0] and [-11.0, 0] (shifted softmax domain)
//       * rsqrt(x) on [1.0, 4.0] + IEEE-754 exponent range reduction & 1x NR
//       * recip(x) on [1.0, 2.0] + IEEE-754 exponent range reduction & 1x NR
//       * SiLU(x) and GELU(x) (tanh & erf) on [-11.1, +11.1]
//   - Affine output quantization fusion helper.
//
// Builds cleanly under -std=c++17 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace freetile {

constexpr size_t kOp640PacketBytes = 208u;
constexpr uint32_t kOp640Opcode = 0x640u;
constexpr uint32_t kOp640NumSegments = 8u;
constexpr uint32_t kOp640CoeffsPerSegment = 5u;
constexpr uint32_t kOp640NumBreakpoints = 7u;
constexpr uint32_t kOp640TailDeg1 = 0x7801u;
constexpr uint32_t kOp640TailDeg4 = 0x7804u;

enum class SplineMode : uint8_t {
    NORMAL = 0,
    SQRT = 1,
    RSQRT = 2,
    RECIP = 3,
};

enum class SplineKind : uint8_t {
    EXP_NEG12_96_0 = 0,  // exp(x) on [-12.96, 0.0], shifted softmax numerator
    EXP_NEG11_0,         // exp(x) on [-11.0, 0.0], compact softmax numerator
    RSQRT_MANTISSA,      // 1/sqrt(u) on [1.0, 4.0], paired with IEEE-754 exponent reduction
    RECIP_MANTISSA,      // 1/u on [1.0, 2.0], paired with IEEE-754 exponent reduction
    SILU,                // x / (1 + exp(-x)) on [-11.1, +11.1]
    GELU_TANH,           // 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))) on [-11.1, +11.1]
    GELU_ERF,            // 0.5*x*(1 + erf(x/sqrt(2))) on [-11.1, +11.1]
};

const char* spline_mode_name(SplineMode mode);
const char* spline_kind_name(SplineKind kind);

// Exact 76-bit 0x640 header (bits [75:0]):
//   [11:0]  opcode (0x640)
//   [27:12] core_mask (e.g. 0xFFFF)
//   [47:28] reserved / lane control bits
//   [63:48] sequence counter
//   [75:64] 12-bit flags (stored in hi12)
struct SplineHeader76 {
    uint64_t lo64 = (0xFFFFULL << 12) | kOp640Opcode;
    uint16_t hi12 = 0u;
};

// Exact 84-bit 0x640 tail (bits [1663:1580]):
//   [1583:1580] polynomial degree (1 or 4)
//   [1594:1591] enable flags (0b1111 -> 0x7801 / 0x7804)
struct SplineTail84 {
    uint64_t lo64 = kOp640TailDeg4;
    uint32_t hi20 = 0u;
};

struct SplineTable0x640 {
    SplineMode mode = SplineMode::NORMAL;
    float domain_min = 0.0f;
    float domain_max = 0.0f;
    float coeffs[kOp640NumSegments][kOp640CoeffsPerSegment] = {};
    float breakpoints[kOp640NumBreakpoints] = {};
    float max_abs_err = 0.0f;
};

struct SplinePacket0x640 {
    SplineHeader76 header;
    float coeffs[kOp640NumSegments][kOp640CoeffsPerSegment] = {};
    float breakpoints[kOp640NumBreakpoints] = {};
    SplineTail84 tail;
};

// Header constructors and field extractors.
SplineHeader76 make_op640_header(uint16_t core_mask = 0xFFFFu,
                                 uint16_t seq = 0u,
                                 uint16_t flags12 = 0u,
                                 uint32_t mid20 = 0u);
uint16_t op640_header_opcode(const SplineHeader76& hdr);
uint16_t op640_header_core_mask(const SplineHeader76& hdr);
uint16_t op640_header_seq(const SplineHeader76& hdr);
uint16_t op640_header_flags(const SplineHeader76& hdr);

// Clean-room 8-segment degree-4 minimax coefficient tables.
const SplineTable0x640& buenos_spline_exp_neg12_96();
const SplineTable0x640& buenos_spline_exp_neg11();
const SplineTable0x640& buenos_spline_rsqrt_mantissa();
const SplineTable0x640& buenos_spline_recip_mantissa();
const SplineTable0x640& buenos_spline_silu();
const SplineTable0x640& buenos_spline_gelu_tanh();
const SplineTable0x640& buenos_spline_gelu_erf();
const SplineTable0x640& buenos_spline_table(SplineKind kind);

// Validates finiteness, domain ordering, and strictly ascending breakpoints.
bool validate_spline_table(const SplineTable0x640& table, std::string* err = nullptr);

// Affine output fusion: transforms f(x) -> scale * f(x) + offset across all 8 segments.
SplineTable0x640 affine_fuse_spline_table(const SplineTable0x640& table,
                                          float scale,
                                          float offset);

// Builds a validated SplinePacket0x640 from a SplineTable0x640.
bool make_op640_packet(const SplineTable0x640& table,
                       SplinePacket0x640* out_pkt,
                       uint16_t core_mask = 0xFFFFu,
                       uint16_t seq = 0u,
                       uint16_t flags12 = 0u,
                       uint64_t tail_lo64 = kOp640TailDeg4,
                       std::string* err = nullptr);

// Bit-exact 208-byte (1664-bit) 0x640 packet codec.
bool encode_op640_packet(const SplinePacket0x640& pkt,
                         uint8_t out_bytes[kOp640PacketBytes],
                         std::string* err = nullptr);

bool decode_op640_packet(const uint8_t in_bytes[kOp640PacketBytes],
                         SplinePacket0x640* out_pkt,
                         std::string* err = nullptr);

// Exact IEEE-754 FP32 RNE Horner evaluation with explicit NaN propagation and domain clamp.
float eval_op640_horner_f32(float x_in,
                            const SplineTable0x640& table,
                            float fclip_min,
                            float fclip_max);

float eval_op640_horner_f32(float x_in,
                            const SplinePacket0x640& pkt,
                            float fclip_min,
                            float fclip_max);

void eval_op640_horner_batch_f32(const float* in,
                                 float* out,
                                 size_t count,
                                 const SplineTable0x640& table,
                                 float fclip_min,
                                 float fclip_max);

// 1x Newton-Raphson hardware refinement steps (matching 0x100 MUL/ADD chaining).
float rsqrt_newton_raphson_step(float x, float y_seed);
float recip_newton_raphson_step(float x, float y_seed);

// Full-domain hardware-faithful evaluators:
//   - eval_buenos_exp_shifted: exp(x) for shifted softmax x <= 0 (returns 0 for x < domain_min,
//     clamps x > 0 to 0, propagates NaN).
//   - eval_buenos_rsqrt: range-reduced 0x640 mode=2 rsqrt over all positive FP32 (normal + subnormal),
//     with optional 1x NR refinement in reduced domain (overflow/underflow safe).
//   - eval_buenos_recip: range-reduced 0x640 mode=3 recip over all nonzero finite FP32,
//     with optional 1x NR refinement in reduced domain (overflow/underflow safe).
//   - eval_buenos_silu / eval_buenos_gelu_tanh / eval_buenos_gelu_erf: full real-line helpers
//     with asymptotic tail clamping/extension and NaN propagation.
float eval_buenos_exp_shifted(float x, bool use_neg11_domain = false);
float eval_buenos_rsqrt(float x, bool nr_refine = true);
float eval_buenos_recip(float x, bool nr_refine = true);
float eval_buenos_silu(float x);
float eval_buenos_gelu_tanh(float x);
float eval_buenos_gelu_erf(float x);

}  // namespace freetile
