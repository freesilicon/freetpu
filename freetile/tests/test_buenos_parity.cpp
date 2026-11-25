// SPDX-License-Identifier: Apache-2.0
#include "freetile/buenos_emitter.hpp"
#include "freetile/tile_ir.hpp"

#include "ob_internal.h"
#include "openbuenos.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

#define CHECK_TRUE(cond, msg)                                                                  \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));             \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

std::vector<uint8_t> make_dummy_dive_interp(size_t size = 4096u) {
    std::vector<uint8_t> img(size, 0u);
    img[0] = 0x7fu;
    img[1] = 'E';
    img[2] = 'L';
    img[3] = 'F';
    for (size_t i = 4; i < size; ++i) {
        img[i] = static_cast<uint8_t>((i * 37u + 13u) & 0xffu);
    }
    return img;
}

std::vector<int8_t> make_weights(uint32_t n, uint32_t k, uint32_t seed) {
    std::vector<int8_t> w(static_cast<size_t>(n) * static_cast<size_t>(k));
    uint32_t x = seed * 2654435761u + 17u;
    for (size_t i = 0; i < w.size(); ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        w[i] = static_cast<int8_t>(static_cast<int32_t>(x % 255u) - 127);
    }
    return w;
}

std::vector<int32_t> make_bias(uint32_t n, uint32_t seed) {
    std::vector<int32_t> b(n);
    uint32_t x = seed * 1597334677u + 31u;
    for (uint32_t i = 0; i < n; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        b[i] = static_cast<int32_t>(x % 1024u) - 512;
    }
    return b;
}

uint32_t to_ob_act(freetile::EpilogueActivation act) {
    switch (act) {
        case freetile::EpilogueActivation::NONE: return OB_ACT_NONE;
        case freetile::EpilogueActivation::RELU: return OB_ACT_RELU;
        case freetile::EpilogueActivation::RELU6: return OB_ACT_RELU6;
        case freetile::EpilogueActivation::SILU: return OB_ACT_SILU;
        case freetile::EpilogueActivation::GELU: return OB_ACT_GELU;
    }
    return OB_ACT_NONE;
}

bool compare_bytes(const char* label,
                   const std::vector<uint8_t>& actual,
                   const std::vector<uint8_t>& expected,
                   std::string* err) {
    if (actual.size() != expected.size()) {
        *err = std::string(label) + " size mismatch: actual=" + std::to_string(actual.size()) +
               " expected=" + std::to_string(expected.size());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "%s byte mismatch at offset 0x%zx: actual=0x%02x expected=0x%02x",
                          label, i, actual[i], expected[i]);
            *err = buf;
            return false;
        }
    }
    return true;
}

bool verify_dgc0_inspector_parity(const std::vector<uint8_t>& dgc0, std::string* err) {
    ob::ParsedDgc0 ob_parsed;
    std::string ob_err;
    if (ob::parse_dgc0_buffer(dgc0.data(), dgc0.size(), &ob_parsed, &ob_err) != OB_OK) {
        *err = "ob::parse_dgc0_buffer failed: " + ob_err;
        return false;
    }
    freetile::BuenosDgc0Summary ft_parsed;
    std::string ft_err;
    if (!freetile::inspect_buenos_dgc0(dgc0.data(), dgc0.size(), &ft_parsed, &ft_err)) {
        *err = "inspect_buenos_dgc0 failed: " + ft_err;
        return false;
    }
    if (ft_parsed.input_sizes != ob_parsed.input_sizes ||
        ft_parsed.output_sizes != ob_parsed.output_sizes ||
        ft_parsed.ext0_offset != ob_parsed.ext_dive_vm_ir.offset ||
        ft_parsed.ext0_size != ob_parsed.ext_dive_vm_ir.size ||
        ft_parsed.ext1_offset != ob_parsed.ext_dive_image.offset ||
        ft_parsed.ext1_size != ob_parsed.ext_dive_image.size ||
        ft_parsed.has_params != ob_parsed.has_params ||
        ft_parsed.ext2_offset != ob_parsed.ext_params.offset ||
        ft_parsed.ext2_size != ob_parsed.ext_params.size ||
        ft_parsed.dive_scratch_size != ob_parsed.dive_scratch_size ||
        ft_parsed.compiler_cl != ob_parsed.compiler_cl ||
        ft_parsed.exec_field_14 != ob_parsed.exec_field_14 ||
        ft_parsed.cluster_id != ob_parsed.cluster_id) {
        *err = "inspect_buenos_dgc0 summary diverged from ob::parse_dgc0_buffer";
        return false;
    }
    return true;
}

struct W8A8Case {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    float s_in;
    int32_t zp_in;
    float s_w;
    float s_out;
    int32_t zp_out;
    freetile::EpilogueActivation act;
    bool with_bias;
};

bool verify_w8a8_case(const W8A8Case& tc,
                      const std::vector<uint8_t>& dummy_interp,
                      uint32_t seed,
                      std::string* err) {
    const auto weights = make_weights(tc.n, tc.k, seed);
    std::vector<int32_t> bias;
    if (tc.with_bias) {
        bias = make_bias(tc.n, seed + 1000u);
    }

    // Plan and emit via FreeTile zero-dep buenos emitter.
    freetile::TileKernel kernel;
    std::string plan_err;
    if (!freetile::plan_buenos_matmul_kernel(
            tc.m, tc.k, tc.n, freetile::QuantFormat::Q8_0,
            freetile::DType::S8, freetile::DType::S8, tc.act, &kernel, &plan_err)) {
        *err = "plan_buenos_matmul_kernel failed: " + plan_err;
        return false;
    }

    freetile::BuenosEmitOptions opts;
    opts.mode = freetile::BuenosKernelMode::W8A8_INT8;
    opts.s_in = tc.s_in;
    opts.zp_in = tc.zp_in;
    opts.s_w = tc.s_w;
    opts.zp_w = 0;
    opts.s_out = tc.s_out;
    opts.zp_out = tc.zp_out;
    opts.weights_nk = weights.data();
    opts.bias_n = tc.with_bias ? bias.data() : nullptr;
    opts.dive_interp = dummy_interp.data();
    opts.dive_interp_size = dummy_interp.size();

    const freetile::BuenosEmitResult ft_res = freetile::emit_buenos_kernel(kernel, opts);

    // Emit reference via libopenbuenos.
    ob_gemm_desc_t desc{};
    desc.m_dim = tc.m;
    desc.k_dim = tc.k;
    desc.n_dim = tc.n;
    desc.weights_nk = weights.data();
    desc.bias_n = tc.with_bias ? bias.data() : nullptr;
    desc.s_in = tc.s_in;
    desc.zp_in = tc.zp_in;
    desc.s_w = tc.s_w;
    desc.zp_w = 0;
    desc.s_out = tc.s_out;
    desc.zp_out = tc.zp_out;
    desc.activation = to_ob_act(tc.act);

    ob::GemmSynthResult ob_parts;
    std::string ob_err;
    if (ob::synthesize_gemm_parts(desc, &ob_parts, &ob_err) != OB_OK) {
        *err = "ob::synthesize_gemm_parts failed: " + ob_err;
        return false;
    }
    std::vector<uint8_t> ob_dgc0;
    if (ob::build_dgc0_from_parts(desc, ob_parts, dummy_interp.data(), dummy_interp.size(),
                                  &ob_dgc0, &ob_err) != OB_OK) {
        *err = "ob::build_dgc0_from_parts failed: " + ob_err;
        return false;
    }

    if (ft_res.exec_field_14 != ob_parts.exec_field_14) {
        *err = "exec_field_14 mismatch";
        return false;
    }
    if (ft_res.dive_scratch_size != ob_parts.dive_scratch_size) {
        *err = "dive_scratch_size mismatch";
        return false;
    }
    for (int ci = 0; ci < 7; ++ci) {
        const std::string tag = "chunk[" + std::to_string(ci) + "]";
        if (!compare_bytes(tag.c_str(), ft_res.chunks[ci], ob_parts.chunks[ci], err)) {
            return false;
        }
    }
    if (!compare_bytes("ext0_dive", ft_res.ext0_dive, ob_parts.ext0_dive, err)) return false;
    if (!compare_bytes("ext2_params", ft_res.ext2_params, ob_parts.ext2_params, err)) return false;
    if (!compare_bytes("dgc0", ft_res.dgc0, ob_dgc0, err)) return false;
    if (!verify_dgc0_inspector_parity(ft_res.dgc0, err)) return false;
    return true;
}

}  // namespace

int main() {
    const std::vector<uint8_t> dummy_interp = make_dummy_dive_interp(4096u);

    // -------------------------------------------------------------------------
    // 1. 81-shape W8A8 multi-tile offline parity suite (Sweep 1 + Batch 0 +
    //    Phase 4 multi-tile, non-mult-of-4 edges, weight-streaming K > 64,
    //    asymmetric zp, PoT & non-PoT scales, NONE/RELU/RELU6, bias/no-bias).
    // -------------------------------------------------------------------------
    const std::vector<W8A8Case> matrix_81 = {
        // Sweep 1 core & quantization/activation variants (30 cases)
        {1, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, false},
        {1, 16, 16, 0.03125f, -12, 0.015625f, 0.0625f, 5, freetile::EpilogueActivation::NONE, true},
        {1, 16, 16, 0.03125f, 17, 0.015625f, 0.0625f, -33, freetile::EpilogueActivation::NONE, false},
        {1, 16, 16, 0.03125f, 0, 0.015625f, 0.015625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 16, 16, 0.03125f, 0, 0.015625f, 0.00390625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 16, 16, 0.03125f, 0, 0.015625f, 0.05f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 16, 16, 0.03125f, -7, 0.015625f, 0.03f, 11, freetile::EpilogueActivation::NONE, true},
        {1, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::RELU, true},
        {1, 16, 16, 0.03125f, -9, 0.015625f, 0.0625f, -128, freetile::EpilogueActivation::RELU, true},
        {1, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::RELU6, true},
        {1, 16, 16, 0.03125f, 4, 0.015625f, 0.05f, -100, freetile::EpilogueActivation::RELU6, true},
        {1, 16, 8, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 16, 20, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 32, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 32, 20, 0.03125f, -5, 0.015625f, 0.03f, 8, freetile::EpilogueActivation::RELU, true},
        {1, 64, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 64, 20, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::RELU6, false},
        {2, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {2, 32, 8, 0.03125f, 3, 0.015625f, 0.05f, -15, freetile::EpilogueActivation::NONE, true},
        {2, 64, 20, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::RELU, true},
        {3, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {3, 32, 20, 0.03125f, -10, 0.015625f, 0.0625f, 10, freetile::EpilogueActivation::NONE, true},
        {4, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {4, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::RELU, false},
        {4, 32, 16, 0.03125f, 0, 0.015625f, 0.03f, -20, freetile::EpilogueActivation::RELU6, true},
        {4, 64, 8, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {8, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {8, 32, 20, 0.03125f, -3, 0.015625f, 0.0625f, 2, freetile::EpilogueActivation::NONE, true},
        {8, 64, 16, 0.03125f, 0, 0.015625f, 0.05f, 0, freetile::EpilogueActivation::RELU, true},

        // Batch 0 & small edge/non-mult-of-4 shapes (20 cases)
        {1, 16, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 64, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 128, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {4, 32, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 4, 4, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 5, 5, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 7, 13, 0.03125f, -2, 0.015625f, 0.0625f, 3, freetile::EpilogueActivation::NONE, true},
        {1, 17, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 16, 17, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 19, 23, 0.03125f, 0, 0.015625f, 0.03f, -5, freetile::EpilogueActivation::RELU, false},
        {1, 24, 24, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 31, 31, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 33, 33, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 48, 48, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 63, 63, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 65, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 32, 65, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {2, 17, 19, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {3, 25, 29, 0.03125f, 1, 0.015625f, 0.05f, -4, freetile::EpilogueActivation::RELU6, true},
        {5, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},

        // Phase 4 multi-tile M/K/N sweeps, weight-streaming K > 64, large M/K/N (31 cases)
        {6, 48, 48, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {7, 32, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {9, 16, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {10, 32, 32, 0.03125f, -6, 0.015625f, 0.0625f, 6, freetile::EpilogueActivation::NONE, true},
        {12, 64, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::RELU, true},
        {15, 32, 48, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, false},
        {16, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {16, 32, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {16, 64, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {16, 128, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {17, 32, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {24, 64, 32, 0.03125f, 0, 0.015625f, 0.03f, -12, freetile::EpilogueActivation::RELU6, true},
        {31, 32, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {32, 32, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {32, 64, 32, 0.03125f, -4, 0.015625f, 0.0625f, 4, freetile::EpilogueActivation::NONE, true},
        {32, 128, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {33, 64, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {48, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {48, 32, 32, 0.03125f, 0, 0.015625f, 0.05f, 0, freetile::EpilogueActivation::RELU, true},
        {64, 16, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {64, 32, 16, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, false},
        {64, 32, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {64, 24, 24, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 80, 32, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 96, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 192, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 256, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 256, 128, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {1, 512, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
        {4, 128, 64, 0.03125f, -8, 0.015625f, 0.0625f, 12, freetile::EpilogueActivation::RELU, true},
        {8, 256, 64, 0.03125f, 0, 0.015625f, 0.0625f, 0, freetile::EpilogueActivation::NONE, true},
    };
    CHECK_TRUE(matrix_81.size() == 81u, "81-shape parity matrix must have exactly 81 entries");

    for (size_t i = 0; i < matrix_81.size(); ++i) {
        std::string err;
        if (!verify_w8a8_case(matrix_81[i], dummy_interp, static_cast<uint32_t>(i + 1u), &err)) {
            std::fprintf(stderr, "FAIL 81-shape[%zu] (%ux%ux%u): %s\n",
                         i, matrix_81[i].m, matrix_81[i].k, matrix_81[i].n, err.c_str());
            return 1;
        }
    }
    std::printf("PASS [1/7]: 81/81 W8A8 multi-tile shapes bit-exact vs libopenbuenos (0 byte diffs)\n");

    // -------------------------------------------------------------------------
    // 2. 14+ M=1 large-tile (lt_1x*) shapes (16-core & 9-core, bias & no-bias).
    // -------------------------------------------------------------------------
    const std::vector<std::pair<uint32_t, uint32_t>> lt_1x_shapes = {
        {512, 512},   {512, 1024},  {512, 2048},  {576, 576},   {576, 1536},
        {1024, 512},  {1024, 1024}, {1024, 1536}, {1024, 2048}, {1536, 576},
        {1536, 1024}, {1536, 1536}, {2048, 512},  {2048, 1024}, {2048, 2048},
    };
    for (size_t i = 0; i < lt_1x_shapes.size(); ++i) {
        const W8A8Case tc{
            1u, lt_1x_shapes[i].first, lt_1x_shapes[i].second,
            0.03125f, (i % 3u == 0u) ? -5 : 0, 0.015625f,
            (i % 2u == 0u) ? 0.0625f : 0.05f, (i % 3u == 1u) ? 7 : 0,
            (i % 4u == 0u) ? freetile::EpilogueActivation::RELU : freetile::EpilogueActivation::NONE,
            (i % 2u == 0u)};
        std::string err;
        if (!verify_w8a8_case(tc, dummy_interp, static_cast<uint32_t>(200u + i), &err)) {
            std::fprintf(stderr, "FAIL lt_1x[%zu] (1x%ux%u): %s\n",
                         i, tc.k, tc.n, err.c_str());
            return 1;
        }
    }
    std::printf("PASS [2/7]: %zu/%zu lt_1x* shapes bit-exact vs libopenbuenos (0 byte diffs)\n",
                lt_1x_shapes.size(), lt_1x_shapes.size());

    // -------------------------------------------------------------------------
    // 3. 11 M=64 batched prefill large-tile (lt_64x*) shapes.
    // -------------------------------------------------------------------------
    const std::vector<std::pair<uint32_t, uint32_t>> lt_64x_shapes = {
        {512, 128},  {512, 512},  {512, 1024}, {576, 576},   {576, 1536},
        {1024, 512}, {1024, 1024}, {1024, 1536}, {1536, 512}, {1536, 576},
        {1536, 1536},
    };
    for (size_t i = 0; i < lt_64x_shapes.size(); ++i) {
        const W8A8Case tc{
            64u, lt_64x_shapes[i].first, lt_64x_shapes[i].second,
            0.03125f, (i % 2u == 0u) ? 0 : -4, 0.015625f,
            0.0625f, (i % 3u == 0u) ? 9 : 0,
            (i % 3u == 1u) ? freetile::EpilogueActivation::RELU6 : freetile::EpilogueActivation::NONE,
            (i % 2u == 0u)};
        std::string err;
        if (!verify_w8a8_case(tc, dummy_interp, static_cast<uint32_t>(300u + i), &err)) {
            std::fprintf(stderr, "FAIL lt_64x[%zu] (64x%ux%u): %s\n",
                         i, tc.k, tc.n, err.c_str());
            return 1;
        }
    }
    std::printf("PASS [3/7]: %zu/%zu lt_64x* batched prefill shapes bit-exact vs libopenbuenos (0 byte diffs)\n",
                lt_64x_shapes.size(), lt_64x_shapes.size());

    // -------------------------------------------------------------------------
    // 4. All 16 W8A16_PT_1X (BF16/FP16 act + INT8 weights -> FP32) shapes.
    // -------------------------------------------------------------------------
    const uint32_t w8a16_dims[4] = {512u, 1024u, 1536u, 2048u};
    size_t w8a16_count = 0;
    for (uint32_t ki = 0; ki < 4u; ++ki) {
        for (uint32_t ni = 0; ni < 4u; ++ni) {
            const uint32_t k = w8a16_dims[ki];
            const uint32_t n = w8a16_dims[ni];
            const float pt_scale = ((ki + ni) % 2u == 0u) ? 1.0f : 0.015625f;
            const auto weights = make_weights(n, k, static_cast<uint32_t>(400u + w8a16_count));

            // Exercise both plan_buenos_matmul_kernel (BF16->F32) and plan_quant_matmul_kernel (Q8_0 BF16->BF16).
            freetile::TileKernel kernel;
            std::string plan_err;
            if (k == 1024u && n == 1024u) {
                CHECK_TRUE(freetile::plan_quant_matmul_kernel(
                               freetile::ChipTarget::BUENOS_A0, 1, k, n, freetile::QuantFormat::Q8_0,
                               freetile::DType::BF16, freetile::DType::BF16, false, &kernel, &plan_err),
                           ("plan_quant_matmul_kernel BUENOS_A0 failed: " + plan_err).c_str());
            } else {
                const freetile::DType act_dt = (ki % 2u == 0u) ? freetile::DType::BF16 : freetile::DType::F16;
                CHECK_TRUE(freetile::plan_buenos_matmul_kernel(
                               1, k, n, freetile::QuantFormat::Q8_0, act_dt, freetile::DType::F32,
                               freetile::EpilogueActivation::NONE, &kernel, &plan_err),
                           ("plan_buenos_matmul_kernel W8A16 failed: " + plan_err).c_str());
            }

            freetile::BuenosEmitOptions opts;
            opts.mode = freetile::BuenosKernelMode::AUTO;
            opts.w8a16_pt_scale = pt_scale;
            opts.weights_nk = weights.data();
            opts.dive_interp = dummy_interp.data();
            opts.dive_interp_size = dummy_interp.size();

            const freetile::BuenosEmitResult ft_res = freetile::emit_buenos_kernel(kernel, opts);
            CHECK_TRUE(ft_res.resolved_mode == freetile::BuenosKernelMode::W8A16_PT_1X,
                       "AUTO mode resolves BF16/FP16 1xKxN to W8A16_PT_1X");

            ob::GemmSynthResult ob_parts;
            std::vector<uint8_t> ob_dgc0;
            std::string ob_err;
            CHECK_TRUE(ob::synthesize_w8a16_pt_1x_parts(k, n, pt_scale, weights.data(), &ob_parts, &ob_err) == OB_OK,
                       ("ob::synthesize_w8a16_pt_1x_parts failed: " + ob_err).c_str());
            CHECK_TRUE(ob::build_dgc0_w8a16_pt_1x(k, n, pt_scale, weights.data(),
                                                  dummy_interp.data(), dummy_interp.size(),
                                                  &ob_dgc0, &ob_err) == OB_OK,
                       ("ob::build_dgc0_w8a16_pt_1x failed: " + ob_err).c_str());

            std::string cmp_err;
            CHECK_TRUE(ft_res.exec_field_14 == ob_parts.exec_field_14, "W8A16 exec_field_14 parity");
            CHECK_TRUE(ft_res.dive_scratch_size == ob_parts.dive_scratch_size, "W8A16 dive_scratch_size parity");
            for (int ci = 0; ci < 7; ++ci) {
                const std::string tag = "w8a16 chunk[" + std::to_string(ci) + "]";
                CHECK_TRUE(compare_bytes(tag.c_str(), ft_res.chunks[ci], ob_parts.chunks[ci], &cmp_err),
                           cmp_err.c_str());
            }
            CHECK_TRUE(compare_bytes("w8a16 ext0_dive", ft_res.ext0_dive, ob_parts.ext0_dive, &cmp_err),
                       cmp_err.c_str());
            CHECK_TRUE(compare_bytes("w8a16 ext2_params", ft_res.ext2_params, ob_parts.ext2_params, &cmp_err),
                       cmp_err.c_str());
            CHECK_TRUE(compare_bytes("w8a16 dgc0", ft_res.dgc0, ob_dgc0, &cmp_err), cmp_err.c_str());
            CHECK_TRUE(verify_dgc0_inspector_parity(ft_res.dgc0, &cmp_err), cmp_err.c_str());
            ++w8a16_count;
        }
    }
    std::printf("PASS [4/7]: %zu/%zu W8A16/BF16 shapes bit-exact vs libopenbuenos (0 byte diffs)\n",
                w8a16_count, w8a16_count);

    // -------------------------------------------------------------------------
    // 5. Mode-A zero-overhead (+0 B) fused SiLU & GELU epilogues (standard,
    //    decoupled asymmetric zp, lt_1x*, lt_64x*) + 256-codeword Horner parity.
    // -------------------------------------------------------------------------
    struct FusionCase {
        uint32_t m;
        uint32_t k;
        uint32_t n;
        freetile::EpilogueActivation act;
        float s_out;
        int32_t zp_out;
        float s_mid;
        int32_t zp_mid;
    };
    const std::vector<FusionCase> fusion_cases = {
        {1, 16, 16, freetile::EpilogueActivation::SILU, 0.0625f, 0, 0.0f, 0},
        {1, 16, 16, freetile::EpilogueActivation::GELU, 0.0625f, 0, 0.0f, 0},
        {4, 32, 32, freetile::EpilogueActivation::SILU, 0.05f, -112, 0.0f, 0},
        {4, 32, 32, freetile::EpilogueActivation::GELU, 0.04f, -116, 0.0f, 0},
        {1, 16, 16, freetile::EpilogueActivation::SILU, 0.03125f, -112, 0.0625f, 0},
        {1, 16, 16, freetile::EpilogueActivation::GELU, 0.03125f, -116, 0.0625f, 8},
        {1, 576, 1536, freetile::EpilogueActivation::SILU, 0.0625f, 0, 0.0f, 0},
        {1, 1024, 1024, freetile::EpilogueActivation::GELU, 0.0625f, -64, 0.0f, 0},
        {64, 576, 576, freetile::EpilogueActivation::SILU, 0.0625f, 0, 0.0f, 0},
        {64, 576, 576, freetile::EpilogueActivation::GELU, 0.05f, -100, 0.0f, 0},
    };
    uint8_t all_u_mid[256];
    for (int i = 0; i < 256; ++i) all_u_mid[i] = static_cast<uint8_t>(i);

    for (size_t i = 0; i < fusion_cases.size(); ++i) {
        const auto& fc = fusion_cases[i];
        const auto weights = make_weights(fc.n, fc.k, static_cast<uint32_t>(500u + i));
        const auto bias = make_bias(fc.n, static_cast<uint32_t>(600u + i));

        freetile::TileKernel kernel;
        std::string plan_err;
        CHECK_TRUE(freetile::plan_buenos_matmul_kernel(
                       fc.m, fc.k, fc.n, freetile::QuantFormat::Q8_0,
                       freetile::DType::S8, freetile::DType::S8, fc.act, &kernel, &plan_err),
                   ("plan_buenos_matmul_kernel fusion failed: " + plan_err).c_str());

        freetile::BuenosEmitOptions opts;
        opts.mode = freetile::BuenosKernelMode::W8A8_INT8;
        opts.s_in = 0.03125f;
        opts.zp_in = 0;
        opts.s_w = 0.015625f;
        opts.zp_w = 0;
        opts.s_out = fc.s_out;
        opts.zp_out = fc.zp_out;
        opts.s_mid = fc.s_mid;
        opts.zp_mid = fc.zp_mid;
        opts.weights_nk = weights.data();
        opts.bias_n = bias.data();
        opts.dive_interp = dummy_interp.data();
        opts.dive_interp_size = dummy_interp.size();

        const freetile::BuenosEmitResult ft_res = freetile::emit_buenos_kernel(kernel, opts);
        const float lsb_bound = (fc.s_out >= 0.0625f) ? 0.035f : 0.25f;
        if (!(ft_res.mode_a_max_lsb_err >= 0.0f && ft_res.mode_a_max_lsb_err < lsb_bound)) {
            std::fprintf(stderr, "FAIL fusion[%zu] (%ux%ux%u s_out=%g zp_out=%d): mode_a_max_lsb_err=%g (bound=%g)\n",
                         i, fc.m, fc.k, fc.n, fc.s_out, fc.zp_out, ft_res.mode_a_max_lsb_err, lsb_bound);
            return 1;
        }

        // Compare against libopenbuenos. For decoupled (s_mid > 0), synthesize base NONE
        // with (s_mid, zp_mid) and apply ob::apply_mode_a_fusion_c5 identically.
        ob_gemm_desc_t desc{};
        desc.m_dim = fc.m;
        desc.k_dim = fc.k;
        desc.n_dim = fc.n;
        desc.weights_nk = weights.data();
        desc.bias_n = bias.data();
        desc.s_in = 0.03125f;
        desc.zp_in = 0;
        desc.s_w = 0.015625f;
        desc.zp_w = 0;
        desc.s_out = (fc.s_mid > 0.0f) ? fc.s_mid : fc.s_out;
        desc.zp_out = (fc.s_mid > 0.0f) ? fc.zp_mid : fc.zp_out;
        desc.activation = (fc.s_mid > 0.0f) ? static_cast<uint32_t>(OB_ACT_NONE) : to_ob_act(fc.act);

        ob::GemmSynthResult ob_parts;
        std::string ob_err;
        CHECK_TRUE(ob::synthesize_gemm_parts(desc, &ob_parts, &ob_err) == OB_OK,
                   ("ob::synthesize_gemm_parts fusion failed: " + ob_err).c_str());

        if (fc.s_mid > 0.0f) {
            float ob_err_lsb = 0.0f;
            CHECK_TRUE(ob::apply_mode_a_fusion_c5(&ob_parts.chunks[5], to_ob_act(fc.act),
                                                  fc.s_mid, fc.zp_mid, fc.s_out, fc.zp_out,
                                                  &ob_err_lsb, &ob_err) == OB_OK,
                       ("ob::apply_mode_a_fusion_c5 failed: " + ob_err).c_str());
            // Rebuild ext0_dive? In decoupled test verify chunk[5] & simulator parity directly.
            std::string cmp_err;
            CHECK_TRUE(compare_bytes("decoupled fusion chunk[5]", ft_res.chunks[5], ob_parts.chunks[5], &cmp_err),
                       cmp_err.c_str());
            CHECK_TRUE(std::fabs(ft_res.mode_a_max_lsb_err - ob_err_lsb) < 1e-6f,
                       "decoupled mode_a_max_lsb_err exact match");
        } else {
            std::vector<uint8_t> ob_dgc0;
            CHECK_TRUE(ob::build_dgc0_from_parts(desc, ob_parts, dummy_interp.data(), dummy_interp.size(),
                                                 &ob_dgc0, &ob_err) == OB_OK,
                       ("ob::build_dgc0_from_parts fusion failed: " + ob_err).c_str());
            std::string cmp_err;
            for (int ci = 0; ci < 7; ++ci) {
                const std::string tag = "fusion chunk[" + std::to_string(ci) + "]";
                CHECK_TRUE(compare_bytes(tag.c_str(), ft_res.chunks[ci], ob_parts.chunks[ci], &cmp_err),
                           cmp_err.c_str());
            }
            CHECK_TRUE(compare_bytes("fusion ext0_dive", ft_res.ext0_dive, ob_parts.ext0_dive, &cmp_err),
                       cmp_err.c_str());
            CHECK_TRUE(compare_bytes("fusion dgc0", ft_res.dgc0, ob_dgc0, &cmp_err), cmp_err.c_str());
        }

        // 256-codeword exact Horner simulator parity.
        int8_t ft_sim[256];
        int8_t ob_sim[256];
        std::string sim_err;
        CHECK_TRUE(freetile::buenos_simulate_c5_mode_a_fused(ft_res.chunks[5], all_u_mid, 256, ft_sim, &sim_err),
                   ("buenos_simulate_c5_mode_a_fused failed: " + sim_err).c_str());
        CHECK_TRUE(ob::simulate_c5_mode_a_fused(ob_parts.chunks[5], all_u_mid, 256, ob_sim, &ob_err) == OB_OK,
                   ("ob::simulate_c5_mode_a_fused failed: " + ob_err).c_str());
        CHECK_TRUE(std::memcmp(ft_sim, ob_sim, 256) == 0,
                   "256-codeword Horner simulator matches libopenbuenos byte-for-byte");
    }
    std::printf("PASS [5/7]: %zu/%zu Mode-A fused SiLU/GELU epilogues + 256-codeword Horner parity bit-exact\n",
                fusion_cases.size(), fusion_cases.size());

    // -------------------------------------------------------------------------
    // 6. W8A8_DYN_WEIGHTS 2-input 0-param DGC0 bit-exact parity.
    // -------------------------------------------------------------------------
    const std::vector<std::array<uint32_t, 3>> dyn_shapes = {
        {1u, 16u, 16u}, {4u, 32u, 32u}, {8u, 64u, 64u}, {1u, 1024u, 1024u}, {64u, 576u, 576u},
    };
    for (size_t i = 0; i < dyn_shapes.size(); ++i) {
        const uint32_t m = dyn_shapes[i][0];
        const uint32_t k = dyn_shapes[i][1];
        const uint32_t n = dyn_shapes[i][2];
        freetile::TileKernel kernel;
        std::string plan_err;
        CHECK_TRUE(freetile::plan_buenos_matmul_kernel(
                       m, k, n, freetile::QuantFormat::Q8_0,
                       freetile::DType::S8, freetile::DType::S8,
                       freetile::EpilogueActivation::NONE, &kernel, &plan_err),
                   ("plan_buenos_matmul_kernel dyn failed: " + plan_err).c_str());

        freetile::BuenosEmitOptions opts;
        opts.mode = freetile::BuenosKernelMode::W8A8_DYN_WEIGHTS;
        opts.s_in = 0.03125f;
        opts.s_w = 0.015625f;
        opts.s_out = 0.0625f;
        opts.dive_interp = dummy_interp.data();
        opts.dive_interp_size = dummy_interp.size();
        const freetile::BuenosEmitResult ft_res = freetile::emit_buenos_kernel(kernel, opts);

        ob_gemm_desc_t desc{};
        desc.m_dim = m;
        desc.k_dim = k;
        desc.n_dim = n;
        desc.s_in = 0.03125f;
        desc.s_w = 0.015625f;
        desc.s_out = 0.0625f;
        desc.activation = OB_ACT_NONE;
        std::vector<uint8_t> ob_dgc0;
        uint32_t ob_ext2_bytes = 0;
        std::string ob_err;
        CHECK_TRUE(ob::build_dgc0_dyn_weights(desc, dummy_interp.data(), dummy_interp.size(),
                                              &ob_dgc0, &ob_ext2_bytes, &ob_err) == OB_OK,
                   ("ob::build_dgc0_dyn_weights failed: " + ob_err).c_str());

        std::string cmp_err;
        CHECK_TRUE(ft_res.ext2_bytes == ob_ext2_bytes, "dyn_weights ext2_bytes parity");
        CHECK_TRUE(ft_res.ext2_params.empty(), "dyn_weights has empty embedded ext2_params");
        CHECK_TRUE(compare_bytes("dyn_weights dgc0", ft_res.dgc0, ob_dgc0, &cmp_err), cmp_err.c_str());
        CHECK_TRUE(verify_dgc0_inspector_parity(ft_res.dgc0, &cmp_err), cmp_err.c_str());
    }
    std::printf("PASS [6/7]: %zu/%zu W8A8_DYN_WEIGHTS 2-input DGC0 executables bit-exact vs libopenbuenos\n",
                dyn_shapes.size(), dyn_shapes.size());

    // -------------------------------------------------------------------------
    // 7. Pre-packed params, weight-packing helpers & error handling parity.
    // -------------------------------------------------------------------------
    {
        const auto w = make_weights(32, 64, 777u);
        const auto b = make_bias(32, 778u);
        const auto ft_p = freetile::buenos_pack_ext2(32, 64, w.data(), b.data(), true);
        const auto ob_p = ob::pack_ext2_for_tile(32, 64, w.data(), b.data());
        std::string cmp_err;
        CHECK_TRUE(compare_bytes("pack_ext2", ft_p, ob_p, &cmp_err), cmp_err.c_str());

        const auto w64 = make_weights(576, 576, 779u);
        const auto b64 = make_bias(576, 780u);
        const auto ft_p64 = freetile::buenos_pack_ext2_lt_64x(576, 576, w64.data(), b64.data());
        const auto ob_p64 = ob::pack_ext2_for_lt_64x(576, 576, w64.data(), b64.data());
        CHECK_TRUE(compare_bytes("pack_ext2_lt_64x", ft_p64, ob_p64, &cmp_err), cmp_err.c_str());

        const auto w16 = make_weights(512, 512, 781u);
        const auto ft_w16 = freetile::buenos_pack_w8a16_ext2(512, 512, w16.data());
        const auto ob_w16 = ob::pack_w8a16_ext2(512, 512, w16.data());
        CHECK_TRUE(compare_bytes("pack_w8a16_ext2", ft_w16, ob_w16, &cmp_err), cmp_err.c_str());

        // Pre-packed ext[2] passed directly via opts.packed_params produces identical DGC0.
        freetile::TileKernel k_pre;
        CHECK_TRUE(freetile::plan_buenos_matmul_kernel(
                       4, 64, 32, freetile::QuantFormat::Q8_0,
                       freetile::DType::S8, freetile::DType::S8,
                       freetile::EpilogueActivation::NONE, &k_pre),
                   "plan pre-packed kernel");
        freetile::BuenosEmitOptions o_raw;
        o_raw.weights_nk = w.data();
        o_raw.bias_n = b.data();
        o_raw.dive_interp = dummy_interp.data();
        o_raw.dive_interp_size = dummy_interp.size();
        const auto r_raw = freetile::emit_buenos_kernel(k_pre, o_raw);

        freetile::BuenosEmitOptions o_pre;
        o_pre.packed_params = ft_p.data();
        o_pre.packed_params_bytes = ft_p.size();
        o_pre.dive_interp = dummy_interp.data();
        o_pre.dive_interp_size = dummy_interp.size();
        const auto r_pre = freetile::emit_buenos_kernel(k_pre, o_pre);
        CHECK_TRUE(compare_bytes("prepacked dgc0", r_pre.dgc0, r_raw.dgc0, &cmp_err), cmp_err.c_str());

        // Out-of-envelope shapes must be cleanly rejected identically to libopenbuenos.
        const std::array<std::array<uint32_t, 3>, 4> bad_shapes = {{
            {24u, 64u, 48u},
            {48u, 64u, 32u},
            {4u, 128u, 128u},
            {17u, 32u, 12u},
        }};
        for (const auto& bs : bad_shapes) {
            freetile::TileKernel bad_k;
            std::string bad_r;
            CHECK_TRUE(!freetile::plan_buenos_matmul_kernel(
                           bs[0], bs[1], bs[2], freetile::QuantFormat::Q8_0,
                           freetile::DType::S8, freetile::DType::S8,
                           freetile::EpilogueActivation::NONE, &bad_k, &bad_r),
                       "Out-of-envelope shape rejected by plan_buenos_matmul_kernel");
            ob_gemm_desc_t bad_desc{};
            bad_desc.m_dim = bs[0];
            bad_desc.k_dim = bs[1];
            bad_desc.n_dim = bs[2];
            bad_desc.s_in = 0.03125f;
            bad_desc.s_w = 0.015625f;
            bad_desc.s_out = 0.0625f;
            ob::GemmSynthResult bad_ob;
            std::string bad_ob_err;
            CHECK_TRUE(ob::synthesize_gemm_parts(bad_desc, &bad_ob, &bad_ob_err) != OB_OK,
                       "Out-of-envelope shape rejected identically by libopenbuenos");
        }
    }
    std::printf("PASS [7/7]: weight packers, pre-packed ext[2], and shape rejection parity verified\n");
    std::printf("PASS: ALL freetile BUENOS_A0 bit-exact parity checks succeeded!\n");
    return 0;
}
