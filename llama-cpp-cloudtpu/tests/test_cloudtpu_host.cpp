#include "pjrt_runtime.hpp"
#include "stablehlo_emitter.hpp"

#ifdef CLOUDTPU_HAS_GGML
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cloudtpu.h"
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK_TRUE(cond, msg)                                                              \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));         \
            ++g_failures;                                                                  \
            return;                                                                        \
        }                                                                                  \
    } while (0)

#define CHECK_NEAR(got, exp, tol, msg)                                                     \
    do {                                                                                   \
        const double _g = static_cast<double>(got);                                        \
        const double _e = static_cast<double>(exp);                                        \
        if (std::fabs(_g - _e) > (tol) || !std::isfinite(_g)) {                            \
            std::fprintf(stderr, "FAIL [%s:%d]: %s (got=%.6g exp=%.6g tol=%.3g)\n",        \
                         __FILE__, __LINE__, (msg), _g, _e, static_cast<double>(tol));     \
            ++g_failures;                                                                  \
            return;                                                                        \
        }                                                                                  \
    } while (0)

void test_compile_options_proto() {
    cloudtpu::PjrtCompileOptions o1;
    const std::string p1 = cloudtpu::encode_compile_options_proto(o1);
    const uint8_t exp1[] = {0x1a, 0x09, 0x20, 0x01, 0x28, 0x01, 0x30, 0x01, 0x98, 0x01, 0x01};
    CHECK_TRUE(p1.size() == sizeof(exp1), "shardy proto length mismatch");
    CHECK_TRUE(std::memcmp(p1.data(), exp1, sizeof(exp1)) == 0, "shardy proto bytes mismatch");

    cloudtpu::PjrtCompileOptions o2;
    o2.use_shardy_partitioner = false;
    const std::string p2 = cloudtpu::encode_compile_options_proto(o2);
    const uint8_t exp2[] = {0x1a, 0x06, 0x20, 0x01, 0x28, 0x01, 0x30, 0x01};
    CHECK_TRUE(p2.size() == sizeof(exp2), "no-shardy proto length mismatch");
    CHECK_TRUE(std::memcmp(p2.data(), exp2, sizeof(exp2)) == 0, "no-shardy proto bytes mismatch");

    cloudtpu::PjrtCompileOptions o8;
    o8.num_partitions = 8;
    const std::string p8 = cloudtpu::encode_compile_options_proto(o8);
    CHECK_TRUE(p8.size() == sizeof(exp1) && static_cast<uint8_t>(p8[5]) == 0x08, "8-partition proto mismatch");

    // Multi-chip replica-SPMD DeviceAssignmentProto (field 9 of ExecutableBuildOptionsProto)
    cloudtpu::PjrtCompileOptions da_opts;
    da_opts.num_replicas = 8;
    da_opts.num_partitions = 1;
    da_opts.use_spmd_partitioning = false;
    da_opts.use_shardy_partitioner = false;
    da_opts.device_assignment_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::string p_da = cloudtpu::encode_compile_options_proto(da_opts);
    // Expected EBO: 20 08 (replicas=8) 28 01 (parts=1) 4a 10 (field 9 len=16: 08 08 10 01 1a 0a 0a 08 00..07)
    const uint8_t exp_da[] = {
        0x1a, 0x16,
        0x20, 0x08,
        0x28, 0x01,
        0x4a, 0x10,
        0x08, 0x08, 0x10, 0x01,
        0x1a, 0x0a, 0x0a, 0x08, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
    };
    CHECK_TRUE(p_da.size() == sizeof(exp_da), "device_assignment proto length mismatch");
    CHECK_TRUE(std::memcmp(p_da.data(), exp_da, sizeof(exp_da)) == 0, "device_assignment proto bytes mismatch");

    // Alias passthrough + DebugOptions.xla_backend_extra_options (field 3 -> repeated field 500)
    cloudtpu::PjrtCompileOptions opt_extra;
    opt_extra.alias_passthrough_params = true;
    opt_extra.backend_extra_options = {
        {"xla_tpu_enable_async_collective_fusion", "true"},
        {"xla_enable_async_all_reduce", "true"},
    };
    const std::string p_extra = cloudtpu::encode_compile_options_proto(opt_extra);
    CHECK_TRUE(p_extra.find("xla_tpu_enable_async_collective_fusion") != std::string::npos,
               "backend_extra_options fusion key missing");
    CHECK_TRUE(p_extra.find("xla_enable_async_all_reduce") != std::string::npos,
               "backend_extra_options all_reduce key missing");
    // Field 10 tag = (10<<3)|0 = 0x50 followed by 0x01
    bool found_alias_tag = false;
    for (size_t i = 0; i + 1 < p_extra.size(); ++i) {
        if (static_cast<uint8_t>(p_extra[i]) == 0x50 && static_cast<uint8_t>(p_extra[i + 1]) == 0x01) {
            found_alias_tag = true;
            break;
        }
    }
    CHECK_TRUE(found_alias_tag, "alias_passthrough_params (field 10) missing");
    std::printf("  [PASS] compile_options_proto (incl. DeviceAssignmentProto, alias_passthrough, DebugOptions)\n");
}

void test_half_bf16_conversions() {
    const float vals[] = {0.0f, 1.0f, -2.5f, 0.125f, 64.5f, -0.0078125f};
    for (float v : vals) {
        CHECK_NEAR(cloudtpu::bf16_to_f32(cloudtpu::f32_to_bf16(v)), v, 1e-2f, "bf16 roundtrip");
        CHECK_NEAR(cloudtpu::f16_to_f32(cloudtpu::f32_to_f16(v)), v, 1e-2f, "f16 roundtrip");
    }
    std::printf("  [PASS] fp16/bf16 scalar roundtrip\n");
}

void test_lru_cache_and_hbm_roundtrip(cloudtpu::PjrtRuntime& rt) {
    size_t free_b = 0, total_b = 0;
    CHECK_TRUE(rt.query_memory_stats(0, &free_b, &total_b), "query_memory_stats");
    CHECK_TRUE(total_b > (1ULL << 30), "mock HBM limit > 1 GiB");

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    auto buf = rt.buffer_from_host(0, a.data(), cloudtpu::PjrtDType::F32, {1, 1, 2, 2});
    std::vector<float> back(4, 0.0f);
    rt.buffer_to_host(buf->handle.get(), back.data(), back.size() * sizeof(float));
    for (size_t i = 0; i < 4; ++i) {
        CHECK_NEAR(back[i], a[i], 1e-6f, "H2D/D2H roundtrip");
    }

    // Build 3 distinct programs with cache_cap = 2 -> verify hit and eviction
    cloudtpu::RuntimeConfig lru_cfg;
    lru_cfg.mock_mode = true;
    lru_cfg.exec_cache_cap = 2;
    cloudtpu::PjrtRuntime lru_rt;
    lru_rt.init(lru_cfg);

    auto make_add = [](int64_t n) {
        cloudtpu::GraphProgramSpec spec;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, n}});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, n}});
        cloudtpu::OpNodeSpec op;
        op.kind = cloudtpu::CloudOpKind::ADD;
        op.src_ids = {1, 2};
        op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, n}};
        spec.ops.push_back(op);
        spec.output_ids = {3};
        return cloudtpu::emit_stablehlo_program(spec);
    };

    const auto p4 = make_add(4);
    const auto p8 = make_add(8);
    const auto p16 = make_add(16);

    lru_rt.compile_or_get(p4.cache_key, p4.mlir_text);
    lru_rt.compile_or_get(p4.cache_key, p4.mlir_text);  // hit
    lru_rt.compile_or_get(p8.cache_key, p8.mlir_text);
    lru_rt.compile_or_get(p16.cache_key, p16.mlir_text); // evicts p4

    const auto st = lru_rt.cache_stats();
    CHECK_TRUE(st.entries == 2, "LRU cache entries == 2");
    CHECK_TRUE(st.hits == 1, "LRU cache hits == 1");
    CHECK_TRUE(st.misses == 3, "LRU cache misses == 3");
    CHECK_TRUE(st.evictions == 1, "LRU cache evictions == 1");
    std::printf("  [PASS] H2D/D2H roundtrip & executable LRU cache\n");
}

std::vector<float> run_emitted(cloudtpu::PjrtRuntime& rt,
                               const cloudtpu::EmittedProgram& prog,
                               const std::vector<cloudtpu::PjrtBufferPtr>& inputs) {
    std::vector<PJRT_Buffer*> raw_in;
    for (const auto& b : inputs) raw_in.push_back(b->handle.get());
    const auto* exec = rt.compile_or_get(prog.cache_key, prog.mlir_text);
    std::vector<cloudtpu::PjrtDType> out_dts;
    std::vector<std::vector<int64_t>> out_dims;
    for (const auto& ob : prog.output_bindings) {
        out_dts.push_back(ob.dtype);
        out_dims.push_back(ob.dims);
    }
    std::vector<cloudtpu::PjrtBufferPtr> outs;
    rt.execute(exec, 0, raw_in.data(), raw_in.size(), outs, out_dts, out_dims, true);
    std::vector<float> host(outs[0]->num_elements());
    rt.buffer_to_host(outs[0]->handle.get(), host.data(), host.size() * sizeof(float));
    return host;
}

void test_mul_mat_dense_and_quant(cloudtpu::PjrtRuntime& rt) {
    // 1) F32 x F32 with GQA batch ratio (src0 B2=1, src1 B2=2)
    constexpr int64_t M = 2, K = 32, N = 4;
    std::vector<float> w(N * K), x(2 * M * K);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            w[n * K + k] = static_cast<float>((n + 1) * (k % 7 - 3)) * 0.125f;
    for (int64_t b = 0; b < 2; ++b)
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < K; ++k)
                x[(b * M + m) * K + k] = static_cast<float>(b + m + (k % 5) - 2) * 0.25f;

    cloudtpu::GraphProgramSpec spec;
    spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, N, K}});
    spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 2, M, K}});
    cloudtpu::OpNodeSpec mm;
    mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
    mm.src_ids = {1, 2};
    mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 2, M, N}};
    mm.high_precision = true;
    spec.ops.push_back(mm);
    spec.output_ids = {3};

    const auto prog = cloudtpu::emit_stablehlo_program(spec);
    auto bw = rt.buffer_from_host(0, w.data(), cloudtpu::PjrtDType::F32, {1, 1, N, K});
    auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 2, M, K});
    auto got = run_emitted(rt, prog, {bw, bx});

    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float ref = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    ref += x[(b * M + m) * K + k] * w[n * K + k];
                }
                CHECK_NEAR(got[(b * M + m) * N + n], ref, 1e-4f, "MUL_MAT GQA F32");
            }
        }
    }

    // 2) Q4_0_SPLIT x F32 (K-major N-minor: qs [1, 1, NB=1, 16, N], d [1, 1, NB=1, N])
    std::vector<uint8_t> qs(N * 16);
    std::vector<uint16_t> d(N);
    std::vector<uint8_t> qs_km(16 * N);
    for (int64_t n = 0; n < N; ++n) {
        d[n] = cloudtpu::f32_to_f16(0.5f * static_cast<float>(n + 1));
        for (int64_t j = 0; j < 16; ++j) {
            const uint8_t lo = static_cast<uint8_t>((j + n + 5) & 0x0F);
            const uint8_t hi = static_cast<uint8_t>((j * 3 + 9) & 0x0F);
            const uint8_t packed = static_cast<uint8_t>(lo | (hi << 4));
            qs[n * 16 + j] = packed;
            qs_km[j * N + n] = packed;
        }
    }
    std::vector<float> x1(K);
    for (int64_t k = 0; k < K; ++k) x1[k] = static_cast<float>((k % 9) - 4) * 0.25f;

    cloudtpu::GraphProgramSpec qspec;
    qspec.inputs.push_back({1, cloudtpu::PjrtDType::U8, cloudtpu::TensorStorageKind::Q4_0_SPLIT, {1, 1, N, K}});
    qspec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, K}});
    cloudtpu::OpNodeSpec qmm;
    qmm.kind = cloudtpu::CloudOpKind::MUL_MAT;
    qmm.src_ids = {1, 2};
    qmm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, N}};
    qspec.ops.push_back(qmm);
    qspec.output_ids = {3};

    const auto qprog = cloudtpu::emit_stablehlo_program(qspec);
    auto bqs = rt.buffer_from_host(0, qs_km.data(), cloudtpu::PjrtDType::U8, {1, 1, 1, 16, N});
    auto bd = rt.buffer_from_host(0, d.data(), cloudtpu::PjrtDType::F16, {1, 1, 1, N});
    auto bx1 = rt.buffer_from_host(0, x1.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, K});
    auto qgot = run_emitted(rt, qprog, {bqs, bd, bx1});

    for (int64_t n = 0; n < N; ++n) {
        const float scale = cloudtpu::f16_to_f32(d[n]);
        float ref = 0.0f;
        for (int64_t j = 0; j < 16; ++j) {
            const float w_lo = static_cast<float>(static_cast<int>(qs[n * 16 + j] & 0x0F) - 8) * scale;
            const float w_hi = static_cast<float>(static_cast<int>(qs[n * 16 + j] >> 4) - 8) * scale;
            ref += x1[j] * w_lo + x1[j + 16] * w_hi;
        }
        CHECK_NEAR(qgot[n], ref, 0.25f, "MUL_MAT Q4_0_SPLIT");
    }
    std::printf("  [PASS] MUL_MAT (F32 GQA + Q4_0_SPLIT)\n");
}

void test_rms_rope_softmax_swiglu_getrows_fused(cloudtpu::PjrtRuntime& rt) {
    // 1) RMS_NORM
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
        cloudtpu::GraphProgramSpec spec;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
        cloudtpu::OpNodeSpec op;
        op.kind = cloudtpu::CloudOpKind::RMS_NORM;
        op.src_ids = {1};
        op.eps = 1e-5f;
        op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
        spec.ops.push_back(op);
        spec.output_ids = {2};
        auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec),
                               {rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4})});
        const float mean = (1 + 4 + 9 + 16) / 4.0f;
        const float r = 1.0f / std::sqrt(mean + 1e-5f);
        for (size_t i = 0; i < 4; ++i) CHECK_NEAR(got[i], x[i] * r, 1e-5f, "RMS_NORM");
    }

    // 2) ROPE (Normal & NeoX)
    for (int mode : {0, 2}) {
        std::vector<float> x = {1.0f, 0.5f, -0.5f, 2.0f};
        int32_t pos = 3;
        cloudtpu::GraphProgramSpec spec;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1}});
        cloudtpu::OpNodeSpec op;
        op.kind = cloudtpu::CloudOpKind::ROPE;
        op.src_ids = {1, 2};
        op.rope.n_dims = 4;
        op.rope.mode = mode;
        op.rope.freq_base = 10000.0f;
        op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
        spec.ops.push_back(op);
        spec.output_ids = {3};
        auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec),
                               {rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4}),
                                rt.buffer_from_host(0, &pos, cloudtpu::PjrtDType::S32, {1, 1, 1, 1})});
        const float t0 = 3.0f;
        const float t1 = 3.0f * std::pow(10000.0f, -0.5f);
        if (mode == 0) {
            CHECK_NEAR(got[0], x[0] * std::cos(t0) - x[1] * std::sin(t0), 1e-5f, "ROPE normal y0");
            CHECK_NEAR(got[1], x[0] * std::sin(t0) + x[1] * std::cos(t0), 1e-5f, "ROPE normal y1");
            CHECK_NEAR(got[2], x[2] * std::cos(t1) - x[3] * std::sin(t1), 1e-5f, "ROPE normal y2");
            CHECK_NEAR(got[3], x[2] * std::sin(t1) + x[3] * std::cos(t1), 1e-5f, "ROPE normal y3");
        } else {
            CHECK_NEAR(got[0], x[0] * std::cos(t0) - x[2] * std::sin(t0), 1e-5f, "ROPE neox y0");
            CHECK_NEAR(got[2], x[0] * std::sin(t0) + x[2] * std::cos(t0), 1e-5f, "ROPE neox y2");
            CHECK_NEAR(got[1], x[1] * std::cos(t1) - x[3] * std::sin(t1), 1e-5f, "ROPE neox y1");
            CHECK_NEAR(got[3], x[1] * std::sin(t1) + x[3] * std::cos(t1), 1e-5f, "ROPE neox y3");
        }
    }

    // 3) SOFT_MAX with mask + scale
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> mask = {0.0f, 0.0f, -1e9f, -1e9f};
        cloudtpu::GraphProgramSpec spec;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
        cloudtpu::OpNodeSpec op;
        op.kind = cloudtpu::CloudOpKind::SOFT_MAX;
        op.src_ids = {1, 2};
        op.soft_max.scale = 0.5f;
        op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
        spec.ops.push_back(op);
        spec.output_ids = {3};
        auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec),
                               {rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4}),
                                rt.buffer_from_host(0, mask.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4})});
        const float e0 = std::exp(0.5f), e1 = std::exp(1.0f);
        CHECK_NEAR(got[0], e0 / (e0 + e1), 1e-5f, "SOFT_MAX p0");
        CHECK_NEAR(got[1], e1 / (e0 + e1), 1e-5f, "SOFT_MAX p1");
        CHECK_NEAR(got[2], 0.0f, 1e-5f, "SOFT_MAX masked p2");
    }

    // 4) GET_ROWS
    {
        std::vector<float> tbl = {10, 11, 20, 21, 30, 31};
        std::vector<int32_t> idx = {2, 0};
        cloudtpu::GraphProgramSpec spec;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 3, 2}});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}});
        cloudtpu::OpNodeSpec op;
        op.kind = cloudtpu::CloudOpKind::GET_ROWS;
        op.src_ids = {1, 2};
        op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, 2}};
        spec.ops.push_back(op);
        spec.output_ids = {3};
        auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec),
                               {rt.buffer_from_host(0, tbl.data(), cloudtpu::PjrtDType::F32, {1, 1, 3, 2}),
                                rt.buffer_from_host(0, idx.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, 2})});
        CHECK_NEAR(got[0], 30.0f, 1e-5f, "GET_ROWS [0]");
        CHECK_NEAR(got[1], 31.0f, 1e-5f, "GET_ROWS [1]");
        CHECK_NEAR(got[2], 10.0f, 1e-5f, "GET_ROWS [2]");
        CHECK_NEAR(got[3], 11.0f, 1e-5f, "GET_ROWS [3]");
    }

    // 5) Fused Subgraph: RMS_NORM -> MUL(scale) -> SILU -> SWIGLU -> CPY(F32->BF16->F32)
    {
        std::vector<float> x = {1.0f, -2.0f, 0.5f, 1.5f};
        std::vector<float> w = {2.0f, 2.0f, 2.0f, 2.0f};
        std::vector<float> up = {1.0f, 1.0f, 1.0f, 1.0f};
        cloudtpu::GraphProgramSpec spec;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
        spec.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});

        cloudtpu::OpNodeSpec rms;
        rms.kind = cloudtpu::CloudOpKind::RMS_NORM;
        rms.src_ids = {1};
        rms.eps = 1e-5f;
        rms.dst = {10, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};

        cloudtpu::OpNodeSpec mul;
        mul.kind = cloudtpu::CloudOpKind::MUL;
        mul.src_ids = {10, 2};
        mul.dst = {11, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};

        cloudtpu::OpNodeSpec glu;
        glu.kind = cloudtpu::CloudOpKind::SWIGLU;
        glu.src_ids = {11, 3};
        glu.dst = {12, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};

        spec.ops = {rms, mul, glu};
        spec.output_ids = {12};

        auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec),
                               {rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4}),
                                rt.buffer_from_host(0, w.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4}),
                                rt.buffer_from_host(0, up.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4})});
        float sq_sum = 0.0f;
        for (float v : x) sq_sum += v * v;
        const float inv = 1.0f / std::sqrt(sq_sum / 4.0f + 1e-5f);
        for (size_t i = 0; i < 4; ++i) {
            const float g = x[i] * inv * w[i];
            const float exp = g / (1.0f + std::exp(-g));
            CHECK_NEAR(got[i], exp, 1e-5f, "Fused RMS_NORM+MUL+SWIGLU");
        }
    }

    std::printf("  [PASS] RMS_NORM, ROPE (normal+neox), SOFT_MAX, GET_ROWS, Fused Subgraph\n");
}

void test_silicon_regressions(cloudtpu::PjrtRuntime& rt) {
    // 1) ROPE with non-multiple-of-8 dim (M=52, B2=2, D=128) -> verifies explicit row-major host_layout on TPU.
    {
        const int64_t B2 = 2, M = 52, D = 128;
        std::vector<float> x(B2 * M * D);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = std::sin(static_cast<float>(i) * 0.13f);
        }
        std::vector<int32_t> pos = {7, 19};
        cloudtpu::GraphProgramSpec spec;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, B2, M, D}});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, B2}});
        cloudtpu::OpNodeSpec op;
        op.kind = cloudtpu::CloudOpKind::ROPE;
        op.src_ids = {1, 2};
        op.rope.n_dims = static_cast<int>(D);
        op.rope.mode = 0;
        op.rope.freq_base = 10000.0f;
        op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, B2, M, D}};
        spec.ops.push_back(op);
        spec.output_ids = {3};
        auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec),
                               {rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, B2, M, D}),
                                rt.buffer_from_host(0, pos.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, B2})});
        for (int64_t b = 0; b < B2; ++b) {
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t i = 0; i < D / 2; ++i) {
                    const float theta = static_cast<float>(pos[static_cast<size_t>(b)]) *
                                        std::pow(10000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(D));
                    const float c = std::cos(theta);
                    const float s = std::sin(theta);
                    const size_t base = static_cast<size_t>((b * M + m) * D + 2 * i);
                    const float r0 = x[base] * c - x[base + 1] * s;
                    const float r1 = x[base] * s + x[base + 1] * c;
                    if (std::fabs(got[base] - r0) > 1e-4f || std::fabs(got[base + 1] - r1) > 1e-4f) {
                        std::fprintf(stderr, "FAIL [ROPE M=52 b=%lld m=%lld i=%lld]: got=(%.5f,%.5f) ref=(%.5f,%.5f)\n",
                                     (long long) b, (long long) m, (long long) i, got[base], got[base + 1], r0, r1);
                        ++g_failures;
                        goto done_rope52;
                    }
                }
            }
        }
    done_rope52:;
    }

    // 2) Q4_0_SPLIT (N=16, NB=8, K=256): dequant-only CPY + MUL_MAT (M=4, both split-dot and single-dot)
    {
        const int64_t N = 16, NB = 8, K = NB * 32, M = 4;
        std::vector<uint8_t> qs_km(static_cast<size_t>(NB * 16 * N));
        std::vector<uint16_t> d_km(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t b = 0; b < NB; ++b) {
                const float sc = 0.05f * static_cast<float>((n + b) % 7 + 1);
                const uint16_t d16 = cloudtpu::f32_to_f16(sc);
                d_km[static_cast<size_t>(b * N + n)] = d16;
                const float scale = cloudtpu::f16_to_f32(d16);
                for (int64_t j = 0; j < 16; ++j) {
                    const uint8_t lo = static_cast<uint8_t>((n * 3 + b * 5 + j) & 0x0F);
                    const uint8_t hi = static_cast<uint8_t>((n * 7 + b * 11 + j * 3 + 2) & 0x0F);
                    qs_km[static_cast<size_t>((b * 16 + j) * N + n)] = static_cast<uint8_t>(lo | (hi << 4));
                    w_ref[static_cast<size_t>(n * K + b * 32 + j)] = static_cast<float>(static_cast<int>(lo) - 8) * scale;
                    w_ref[static_cast<size_t>(n * K + b * 32 + j + 16)] = static_cast<float>(static_cast<int>(hi) - 8) * scale;
                }
            }
        }
        auto bqs = rt.buffer_from_host(0, qs_km.data(), cloudtpu::PjrtDType::U8, {1, 1, NB, 16, N});
        auto bd = rt.buffer_from_host(0, d_km.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});

        // Dequant-only check
        {
            cloudtpu::GraphProgramSpec cspec;
            cspec.inputs.push_back({1, cloudtpu::PjrtDType::U8, cloudtpu::TensorStorageKind::Q4_0_SPLIT, {1, 1, N, K}});
            cloudtpu::OpNodeSpec cpy;
            cpy.kind = cloudtpu::CloudOpKind::CPY;
            cpy.src_ids = {1};
            cpy.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, N, K}};
            cspec.ops.push_back(cpy);
            cspec.output_ids = {2};
            auto wgot = run_emitted(rt, cloudtpu::emit_stablehlo_program(cspec), {bqs, bd});
            int bad = 0;
            for (size_t i = 0; i < w_ref.size(); ++i) {
                if (std::fabs(wgot[i] - w_ref[i]) > 1e-4f) {
                    if (bad < 4) {
                        std::fprintf(stderr, "FAIL [Q4_0 dequant idx=%zu]: got=%.6f ref=%.6f\n", i, wgot[i], w_ref[i]);
                    }
                    ++bad;
                }
            }
            if (bad > 0) {
                std::fprintf(stderr, "FAIL [Q4_0 dequant total_bad=%d/%zu]\n", bad, w_ref.size());
                ++g_failures;
            }
        }

        // MUL_MAT Q4_0 K=256 check (both split-dot and single-dot)
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = std::cos(static_cast<float>(i) * 0.17f);
        }
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});
        for (bool split_dot : {true, false}) {
            cloudtpu::GraphProgramSpec mspec;
            mspec.q4_split_dot = split_dot;
            mspec.inputs.push_back({1, cloudtpu::PjrtDType::U8, cloudtpu::TensorStorageKind::Q4_0_SPLIT, {1, 1, N, K}});
            mspec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
            cloudtpu::OpNodeSpec mm;
            mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
            mm.src_ids = {1, 2};
            mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
            mspec.ops.push_back(mm);
            mspec.output_ids = {3};
            auto ygot = run_emitted(rt, cloudtpu::emit_stablehlo_program(mspec), {bqs, bd, bx});
            double se = 0.0, sref = 0.0;
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n = 0; n < N; ++n) {
                    float ref = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        ref += x[static_cast<size_t>(m * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
                    }
                    const double diff = static_cast<double>(ygot[static_cast<size_t>(m * N + n)] - ref);
                    se += diff * diff;
                    sref += static_cast<double>(ref) * static_cast<double>(ref);
                }
            }
            const double nmse = se / std::max(1e-12, sref);
            if (nmse > 5e-4) {
                std::fprintf(stderr, "FAIL [Q4_0 MUL_MAT K=256 split=%d]: nmse=%.6f got[0]=%.5f\n",
                             split_dot ? 1 : 0, nmse, ygot[0]);
                ++g_failures;
            }
        }
    }

    // 3) Q8_0_SPLIT (N=16, NB=8, K=256, M=4)
    {
        const int64_t N = 16, NB = 8, K = NB * 32, M = 4;
        std::vector<int8_t> qs_km(static_cast<size_t>(NB * 32 * N));
        std::vector<uint16_t> d_km(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t b = 0; b < NB; ++b) {
                const uint16_t d16 = cloudtpu::f32_to_f16(0.01f * static_cast<float>((n + b) % 5 + 1));
                d_km[static_cast<size_t>(b * N + n)] = d16;
                const float scale = cloudtpu::f16_to_f32(d16);
                for (int64_t j = 0; j < 32; ++j) {
                    const int8_t q = static_cast<int8_t>(((n * 13 + b * 29 + j * 7) % 251) - 125);
                    qs_km[static_cast<size_t>((b * 32 + j) * N + n)] = q;
                    w_ref[static_cast<size_t>(n * K + b * 32 + j)] = static_cast<float>(q) * scale;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.23f);

        auto bqs = rt.buffer_from_host(0, qs_km.data(), cloudtpu::PjrtDType::S8, {1, 1, NB, 32, N});
        auto bd = rt.buffer_from_host(0, d_km.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});
        cloudtpu::GraphProgramSpec mspec;
        mspec.inputs.push_back({1, cloudtpu::PjrtDType::S8, cloudtpu::TensorStorageKind::Q8_0_SPLIT, {1, 1, N, K}});
        mspec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {1, 2};
        mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
        mspec.ops.push_back(mm);
        mspec.output_ids = {3};
        auto ygot = run_emitted(rt, cloudtpu::emit_stablehlo_program(mspec), {bqs, bd, bx});
        double se = 0.0, sref = 0.0;
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float ref = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    ref += x[static_cast<size_t>(m * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
                }
                const double diff = static_cast<double>(ygot[static_cast<size_t>(m * N + n)] - ref);
                se += diff * diff;
                sref += static_cast<double>(ref) * static_cast<double>(ref);
            }
        }
        const double nmse = se / std::max(1e-12, sref);
        if (nmse > 5e-4) {
            std::fprintf(stderr, "FAIL [Q8_0 MUL_MAT K=256]: nmse=%.6f got[0]=%.5f\n", nmse, ygot[0]);
            ++g_failures;
        }
    }

    // 4) Single-token SET_ROWS -> dynamic_update_slice + {tf.aliasing_output = 0 : i32}
    {
        const int64_t R = 8, D = 4;
        std::vector<float> tbl(static_cast<size_t>(R * D), 0.0f);
        for (size_t i = 0; i < tbl.size(); ++i) tbl[i] = static_cast<float>(i);
        std::vector<float> upd = {101.0f, 102.0f, 103.0f, 104.0f};
        int32_t row_idx = 5;

        cloudtpu::GraphProgramSpec spec;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, D}});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1}});
        cloudtpu::TensorSpec kv_in{3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, R, D}};
        kv_in.alias_output_idx = 0;
        spec.inputs.push_back(kv_in);
        cloudtpu::OpNodeSpec sr;
        sr.kind = cloudtpu::CloudOpKind::SET_ROWS;
        sr.src_ids = {1, 2, 3};
        sr.dst = {4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, R, D}};
        spec.ops.push_back(sr);
        spec.output_ids = {4};

        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.dynamic_update_slice") != std::string::npos,
                   "single-row SET_ROWS must emit dynamic_update_slice");
        CHECK_TRUE(prog.mlir_text.find("stablehlo.scatter") == std::string::npos,
                   "single-row SET_ROWS must not emit scatter");
        CHECK_TRUE(prog.mlir_text.find("{tf.aliasing_output = 0 : i32}") != std::string::npos,
                   "donated KV arg must emit tf.aliasing_output = 0");
        CHECK_TRUE(prog.input_bindings.size() == 3 && prog.input_bindings[2].donatable,
                   "donated KV binding flag must be set");

        auto bupd = rt.buffer_from_host(0, upd.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, D});
        auto bidx = rt.buffer_from_host(0, &row_idx, cloudtpu::PjrtDType::S32, {1, 1, 1, 1});
        auto btbl = rt.buffer_from_host(0, tbl.data(), cloudtpu::PjrtDType::F32, {1, 1, R, D});
        const auto copts = rt.make_spmd_compile_options(1, 0, true, true);
        const auto* exec = rt.compile_or_get(prog.cache_key + "|alias_test", prog.mlir_text, copts);
        std::vector<PJRT_Buffer*> raw_in = {bupd->handle.get(), bidx->handle.get(), btbl->handle.get()};
        std::vector<cloudtpu::PjrtBufferPtr> outs;
        rt.execute(exec, 0, raw_in.data(), raw_in.size(), outs,
                   {cloudtpu::PjrtDType::F32}, {{1, 1, R, D}}, true);
        std::vector<float> got(static_cast<size_t>(R * D));
        rt.buffer_to_host(outs[0]->handle.get(), got.data(), got.size() * sizeof(float));
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t d = 0; d < D; ++d) {
                const float exp_v = (r == row_idx) ? upd[static_cast<size_t>(d)] : tbl[static_cast<size_t>(r * D + d)];
                CHECK_NEAR(got[static_cast<size_t>(r * D + d)], exp_v, 1e-5f, "SET_ROWS DUS KV update");
            }
        }
    }

    std::printf("  [PASS] Silicon regressions (ROPE M=52, Q4_0/Q8_0 K=256, DUS+KV donation)\n");
}

void test_collectives_p1_live_compatible(cloudtpu::PjrtRuntime& rt) {
    // Runs on BOTH live v5e-1 (P=1) and Host Mock to verify StableHLO v1.0 collective MLIR syntax
    // (all_reduce, all_gather, reduce_scatter) compiles and executes cleanly on libtpu.so.
    std::vector<float> x = {1.5f, -2.0f, 3.25f, 4.75f};
    auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});

    for (cloudtpu::CloudOpKind kind :
         {cloudtpu::CloudOpKind::ALL_REDUCE_SUM,
          cloudtpu::CloudOpKind::ALL_GATHER,
          cloudtpu::CloudOpKind::REDUCE_SCATTER_SUM}) {
        cloudtpu::GraphProgramSpec spec;
        spec.spmd_chips = 1;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
        cloudtpu::OpNodeSpec op;
        op.kind = kind;
        op.collective_dim = 3;
        op.src_ids = {1};
        op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
        spec.ops.push_back(op);
        spec.output_ids = {2};
        auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec), {bx});
        for (size_t i = 0; i < 4; ++i) {
            CHECK_NEAR(got[i], x[i], 1e-5f, "P=1 collective identity");
        }
    }
    std::printf("  [PASS] StableHLO collectives P=1 syntax & execution (all_reduce/all_gather/reduce_scatter)\n");
}

void test_spmd_multichip_mesh(int num_chips) {
    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = true;
    cfg.mock_devices = num_chips;
    cfg.exec_cache_cap = 128;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);

    CHECK_TRUE(rt.device_count() == num_chips, "SPMD mock device_count");
    size_t free0 = 0, total0 = 0;
    CHECK_TRUE(rt.query_memory_stats(0, &free0, &total0), "SPMD device 0 memory stats");
    if (num_chips == 8) {
        CHECK_TRUE(total0 == (32ULL << 30), "v6e-8 32 GiB/chip HBM");
    } else if (num_chips == 4) {
        CHECK_TRUE(total0 == (192ULL << 30), "tpu7x-4t 192 GiB/chip HBM3e");
    }

    const auto copts = rt.make_spmd_compile_options(num_chips);
    CHECK_TRUE(copts.num_replicas == num_chips && copts.num_partitions == 1, "SPMD compile options");
    CHECK_TRUE(static_cast<int>(copts.device_assignment_ids.size()) == num_chips, "SPMD DeviceAssignment ids");
    CHECK_TRUE(copts.backend_extra_options.size() == 7, "SPMD async collective overlap flags (7)");

    auto run_spmd = [&](const cloudtpu::EmittedProgram& prog,
                        const std::vector<std::vector<cloudtpu::PjrtBufferPtr>>& per_dev_in) {
        const auto* exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, copts);
        std::vector<std::vector<PJRT_Buffer*>> raw_in(static_cast<size_t>(num_chips));
        std::vector<PJRT_Buffer* const*> in_ptrs(static_cast<size_t>(num_chips));
        for (int d = 0; d < num_chips; ++d) {
            for (const auto& b : per_dev_in[static_cast<size_t>(d)]) raw_in[static_cast<size_t>(d)].push_back(b->handle.get());
            in_ptrs[static_cast<size_t>(d)] = raw_in[static_cast<size_t>(d)].data();
        }
        std::vector<cloudtpu::PjrtDType> out_dts;
        std::vector<std::vector<int64_t>> out_dims;
        for (const auto& ob : prog.output_bindings) {
            out_dts.push_back(ob.dtype);
            out_dims.push_back(ob.dims);
        }
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
        rt.execute_spmd(exec, num_chips, in_ptrs.data(), prog.input_bindings.size(), outs, out_dts, out_dims, true);
        return outs;
    };

    // 1) Direct ALL_REDUCE_SUM, ALL_GATHER, REDUCE_SCATTER_SUM across P chips
    {
        const int64_t D = static_cast<int64_t>(num_chips) * 2;
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> in_ar(static_cast<size_t>(num_chips));
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> in_ag(static_cast<size_t>(num_chips));
        for (int d = 0; d < num_chips; ++d) {
            std::vector<float> v_ar(static_cast<size_t>(D), static_cast<float>(d + 1));
            std::vector<float> v_ag = {static_cast<float>(d * 10 + 1), static_cast<float>(d * 10 + 2)};
            in_ar[static_cast<size_t>(d)] = {rt.buffer_from_host(d, v_ar.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, D})};
            in_ag[static_cast<size_t>(d)] = {rt.buffer_from_host(d, v_ag.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 2})};
        }
        // ALL_REDUCE_SUM: sum(1..P) = P*(P+1)/2
        {
            cloudtpu::GraphProgramSpec spec;
            spec.spmd_chips = num_chips;
            spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, D}});
            cloudtpu::OpNodeSpec op;
            op.kind = cloudtpu::CloudOpKind::ALL_REDUCE_SUM;
            op.src_ids = {1};
            op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, D}};
            spec.ops.push_back(op);
            spec.output_ids = {2};
            auto outs = run_spmd(cloudtpu::emit_stablehlo_program(spec), in_ar);
            const float exp_sum = static_cast<float>(num_chips * (num_chips + 1)) * 0.5f;
            for (int d = 0; d < num_chips; ++d) {
                std::vector<float> host(static_cast<size_t>(D));
                rt.buffer_to_host(outs[static_cast<size_t>(d)][0]->handle.get(), host.data(), host.size() * 4);
                CHECK_NEAR(host[0], exp_sum, 1e-4f, "SPMD ALL_REDUCE_SUM");
            }
        }
        // ALL_GATHER
        {
            cloudtpu::GraphProgramSpec spec;
            spec.spmd_chips = num_chips;
            spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}});
            cloudtpu::OpNodeSpec op;
            op.kind = cloudtpu::CloudOpKind::ALL_GATHER;
            op.collective_dim = 3;
            op.src_ids = {1};
            op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, D}};
            spec.ops.push_back(op);
            spec.output_ids = {2};
            auto outs = run_spmd(cloudtpu::emit_stablehlo_program(spec), in_ag);
            std::vector<float> host(static_cast<size_t>(D));
            rt.buffer_to_host(outs[0][0]->handle.get(), host.data(), host.size() * 4);
            for (int d = 0; d < num_chips; ++d) {
                CHECK_NEAR(host[static_cast<size_t>(d * 2 + 0)], static_cast<float>(d * 10 + 1), 1e-4f, "SPMD ALL_GATHER[0]");
                CHECK_NEAR(host[static_cast<size_t>(d * 2 + 1)], static_cast<float>(d * 10 + 2), 1e-4f, "SPMD ALL_GATHER[1]");
            }
        }
        // REDUCE_SCATTER_SUM
        {
            cloudtpu::GraphProgramSpec spec;
            spec.spmd_chips = num_chips;
            spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, D}});
            cloudtpu::OpNodeSpec op;
            op.kind = cloudtpu::CloudOpKind::REDUCE_SCATTER_SUM;
            op.collective_dim = 3;
            op.src_ids = {1};
            op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}};
            spec.ops.push_back(op);
            spec.output_ids = {2};
            auto outs = run_spmd(cloudtpu::emit_stablehlo_program(spec), in_ar);
            const float exp_sum = static_cast<float>(num_chips * (num_chips + 1)) * 0.5f;
            for (int d = 0; d < num_chips; ++d) {
                std::vector<float> host(2);
                rt.buffer_to_host(outs[static_cast<size_t>(d)][0]->handle.get(), host.data(), host.size() * 4);
                CHECK_NEAR(host[0], exp_sum, 1e-4f, "SPMD REDUCE_SCATTER_SUM[0]");
                CHECK_NEAR(host[1], exp_sum, 1e-4f, "SPMD REDUCE_SCATTER_SUM[1]");
            }
        }
    }

    // 2) Fused Megatron-LM FFN across P chips:
    //    Gate (SHARD_N, keep_sharded_out=true) + Up (SHARD_N, keep_sharded_out=true) -> local SWIGLU
    //    -> Down (SHARD_K) -> SINGLE all_reduce(sum), ZERO all_gather!
    {
        const int64_t M = 2, K_embd = 16, H_ff = static_cast<int64_t>(num_chips) * 4;
        const int64_t H_local = H_ff / num_chips;
        std::vector<float> x(static_cast<size_t>(M * K_embd));
        std::vector<float> w_gate(static_cast<size_t>(H_ff * K_embd));
        std::vector<float> w_up(static_cast<size_t>(H_ff * K_embd));
        std::vector<float> w_down(static_cast<size_t>(K_embd * H_ff));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.19f);
        for (size_t i = 0; i < w_gate.size(); ++i) w_gate[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.25f;
        for (size_t i = 0; i < w_up.size(); ++i) w_up[i] = std::sin(static_cast<float>(i) * 0.07f) * 0.25f;
        for (size_t i = 0; i < w_down.size(); ++i) w_down[i] = std::cos(static_cast<float>(i) * 0.13f) * 0.25f;

        // Reference scalar FFN
        std::vector<float> ref(static_cast<size_t>(M * K_embd), 0.0f);
        for (int64_t m = 0; m < M; ++m) {
            std::vector<float> h(static_cast<size_t>(H_ff), 0.0f);
            for (int64_t j = 0; j < H_ff; ++j) {
                float g = 0.0f, u = 0.0f;
                for (int64_t k = 0; k < K_embd; ++k) {
                    g += x[static_cast<size_t>(m * K_embd + k)] * w_gate[static_cast<size_t>(j * K_embd + k)];
                    u += x[static_cast<size_t>(m * K_embd + k)] * w_up[static_cast<size_t>(j * K_embd + k)];
                }
                h[static_cast<size_t>(j)] = (g / (1.0f + std::exp(-g))) * u;
            }
            for (int64_t n = 0; n < K_embd; ++n) {
                float acc = 0.0f;
                for (int64_t j = 0; j < H_ff; ++j) {
                    acc += h[static_cast<size_t>(j)] * w_down[static_cast<size_t>(n * H_ff + j)];
                }
                ref[static_cast<size_t>(m * K_embd + n)] = acc;
            }
        }

        cloudtpu::GraphProgramSpec spec;
        spec.spmd_chips = num_chips;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K_embd}, cloudtpu::TensorShardingKind::REPLICATED});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, H_ff, K_embd}, cloudtpu::TensorShardingKind::SHARD_N});
        spec.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, H_ff, K_embd}, cloudtpu::TensorShardingKind::SHARD_N});
        spec.inputs.push_back({4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, K_embd, H_ff}, cloudtpu::TensorShardingKind::SHARD_K});

        cloudtpu::OpNodeSpec op_g;
        op_g.kind = cloudtpu::CloudOpKind::MUL_MAT;
        op_g.src_ids = {2, 1};
        op_g.dst = {10, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, H_ff}};
        op_g.high_precision = true;
        op_g.keep_sharded_out = true;

        cloudtpu::OpNodeSpec op_u;
        op_u.kind = cloudtpu::CloudOpKind::MUL_MAT;
        op_u.src_ids = {3, 1};
        op_u.dst = {11, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, H_ff}};
        op_u.high_precision = true;
        op_u.keep_sharded_out = true;

        cloudtpu::OpNodeSpec op_glu;
        op_glu.kind = cloudtpu::CloudOpKind::SWIGLU;
        op_glu.src_ids = {10, 11};
        op_glu.dst = {12, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, H_ff}};

        cloudtpu::OpNodeSpec op_d;
        op_d.kind = cloudtpu::CloudOpKind::MUL_MAT;
        op_d.src_ids = {4, 12};
        op_d.dst = {13, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K_embd}};
        op_d.high_precision = true;

        spec.ops = {op_g, op_u, op_glu, op_d};
        spec.output_ids = {13};

        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.all_gather") == std::string::npos,
                   "Megatron FFN must emit zero all_gather");
        const size_t ar_pos = prog.mlir_text.find("stablehlo.all_reduce");
        CHECK_TRUE(ar_pos != std::string::npos &&
                       prog.mlir_text.find("stablehlo.all_reduce", ar_pos + 1) == std::string::npos,
                   "Megatron FFN must emit exactly one all_reduce");

        // Prepare per-chip sharded buffers
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> per_dev(static_cast<size_t>(num_chips));
        for (int p = 0; p < num_chips; ++p) {
            auto bx = rt.buffer_from_host(p, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K_embd});
            auto bg = rt.buffer_from_host(
                p, w_gate.data() + static_cast<size_t>(p * H_local * K_embd),
                cloudtpu::PjrtDType::F32, {1, 1, H_local, K_embd});
            auto bu = rt.buffer_from_host(
                p, w_up.data() + static_cast<size_t>(p * H_local * K_embd),
                cloudtpu::PjrtDType::F32, {1, 1, H_local, K_embd});
            std::vector<float> wd_local(static_cast<size_t>(K_embd * H_local));
            for (int64_t r = 0; r < K_embd; ++r) {
                std::memcpy(wd_local.data() + static_cast<size_t>(r * H_local),
                            w_down.data() + static_cast<size_t>(r * H_ff + p * H_local),
                            static_cast<size_t>(H_local) * sizeof(float));
            }
            auto bd = rt.buffer_from_host(p, wd_local.data(), cloudtpu::PjrtDType::F32, {1, 1, K_embd, H_local});
            per_dev[static_cast<size_t>(p)] = {bx, bg, bu, bd};
        }

        // Verify per-chip HBM weight memory scales down by 1/P!
        size_t f_used = 0, t_lim = 0;
        CHECK_TRUE(rt.query_memory_stats(0, &f_used, &t_lim), "SPMD mem stats after shard upload");
        const size_t used_chip0 = t_lim - f_used;
        const size_t full_weights_bytes = (w_gate.size() + w_up.size() + w_down.size()) * sizeof(float);
        const size_t expected_chip0 = (x.size() * sizeof(float)) + full_weights_bytes / static_cast<size_t>(num_chips);
        CHECK_TRUE(used_chip0 == expected_chip0, "SPMD per-chip HBM scales by 1/P");

        auto outs = run_spmd(prog, per_dev);
        for (int p = 0; p < num_chips; ++p) {
            std::vector<float> got(ref.size());
            rt.buffer_to_host(outs[static_cast<size_t>(p)][0]->handle.get(), got.data(), got.size() * sizeof(float));
            for (size_t i = 0; i < ref.size(); ++i) {
                CHECK_NEAR(got[i], ref[i], 1e-4f, "SPMD Megatron FFN parity");
            }
        }
    }

    rt.shutdown();
    std::printf("  [PASS] Multi-chip SPMD mesh (%d chips: collectives + 1/P HBM + Megatron FFN)\n", num_chips);
}

struct DecodeTestLayer {
    std::vector<float> attn_norm;
    std::vector<float> wq;
    std::vector<float> wk;
    std::vector<float> wv;
    std::vector<float> wo;
    std::vector<float> ffn_norm;
    std::vector<float> wgate;
    std::vector<float> wup;
    std::vector<float> wdown;
};

struct DecodeTestModel {
    int64_t n_layers = 2;
    int64_t E = 32;
    int64_t H_Q = 4;
    int64_t H_KV = 2;
    int64_t D = 8;
    int64_t H_ff = 64;
    int64_t KV_CAP = 8;
    int64_t N_KV = 6;  // Static bucket (< KV_CAP) mirroring llama.cpp padded KV views
    std::vector<DecodeTestLayer> layers;
};

DecodeTestModel make_decode_test_model() {
    DecodeTestModel m;
    m.layers.resize(static_cast<size_t>(m.n_layers));
    const int64_t D_Q = m.H_Q * m.D;
    const int64_t D_KV = m.H_KV * m.D;
    auto fill = [](std::vector<float>& v, int64_t n, float scale, float phase) {
        v.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            v[static_cast<size_t>(i)] = scale * std::sin(static_cast<float>(i + 1) * 0.17f + phase);
        }
    };
    for (int64_t l = 0; l < m.n_layers; ++l) {
        const float lp = static_cast<float>(l) * 0.7f;
        DecodeTestLayer& ly = m.layers[static_cast<size_t>(l)];
        ly.attn_norm.resize(static_cast<size_t>(m.E));
        ly.ffn_norm.resize(static_cast<size_t>(m.E));
        for (int64_t i = 0; i < m.E; ++i) {
            ly.attn_norm[static_cast<size_t>(i)] = 0.8f + 0.1f * std::cos(static_cast<float>(i) * 0.3f + lp);
            ly.ffn_norm[static_cast<size_t>(i)] = 0.85f + 0.1f * std::sin(static_cast<float>(i) * 0.25f + lp);
        }
        fill(ly.wq, D_Q * m.E, 0.18f, 0.1f + lp);
        fill(ly.wk, D_KV * m.E, 0.17f, 0.3f + lp);
        fill(ly.wv, D_KV * m.E, 0.16f, 0.5f + lp);
        fill(ly.wo, m.E * D_Q, 0.15f, 0.7f + lp);
        fill(ly.wgate, m.H_ff * m.E, 0.14f, 0.9f + lp);
        fill(ly.wup, m.H_ff * m.E, 0.14f, 1.1f + lp);
        fill(ly.wdown, m.E * m.H_ff, 0.12f, 1.3f + lp);
    }
    return m;
}

std::vector<float> scalar_decode_step(const DecodeTestModel& m,
                                      const std::vector<float>& x_in,
                                      int32_t pos,
                                      std::vector<std::vector<float>>& k_caches,
                                      std::vector<std::vector<float>>& v_caches) {
    const int64_t D_Q = m.H_Q * m.D;
    const int64_t D_KV = m.H_KV * m.D;
    const float eps = 1e-5f;
    auto rms_mul = [&](const std::vector<float>& x, const std::vector<float>& w) {
        double ss = 0.0;
        for (float v : x) ss += static_cast<double>(v) * static_cast<double>(v);
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(m.E)) + eps);
        std::vector<float> out(static_cast<size_t>(m.E));
        for (int64_t i = 0; i < m.E; ++i) out[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] * inv * w[static_cast<size_t>(i)];
        return out;
    };
    auto matvec = [](const std::vector<float>& w, const std::vector<float>& x, int64_t rows, int64_t cols) {
        std::vector<float> out(static_cast<size_t>(rows), 0.0f);
        for (int64_t r = 0; r < rows; ++r) {
            float acc = 0.0f;
            for (int64_t c = 0; c < cols; ++c) {
                acc += w[static_cast<size_t>(r * cols + c)] * x[static_cast<size_t>(c)];
            }
            out[static_cast<size_t>(r)] = acc;
        }
        return out;
    };
    auto rope_neox = [&](std::vector<float>& t, int64_t n_heads) {
        const int64_t half = m.D / 2;
        for (int64_t h = 0; h < n_heads; ++h) {
            float* p = t.data() + static_cast<size_t>(h * m.D);
            for (int64_t i = 0; i < half; ++i) {
                const float inv_f = std::pow(10000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(m.D));
                const float ang = static_cast<float>(pos) * inv_f;
                const float c = std::cos(ang);
                const float s = std::sin(ang);
                const float x0 = p[static_cast<size_t>(i)];
                const float x1 = p[static_cast<size_t>(i + half)];
                p[static_cast<size_t>(i)] = x0 * c - x1 * s;
                p[static_cast<size_t>(i + half)] = x0 * s + x1 * c;
            }
        }
    };

    std::vector<float> h = x_in;
    for (int64_t l = 0; l < m.n_layers; ++l) {
        const DecodeTestLayer& ly = m.layers[static_cast<size_t>(l)];
        const std::vector<float> xn = rms_mul(h, ly.attn_norm);
        std::vector<float> q = matvec(ly.wq, xn, D_Q, m.E);
        std::vector<float> k = matvec(ly.wk, xn, D_KV, m.E);
        std::vector<float> v = matvec(ly.wv, xn, D_KV, m.E);
        rope_neox(q, m.H_Q);
        rope_neox(k, m.H_KV);

        for (int64_t d = 0; d < D_KV; ++d) {
            k_caches[static_cast<size_t>(l)][static_cast<size_t>(pos * D_KV + d)] =
                cloudtpu::f16_to_f32(cloudtpu::f32_to_f16(k[static_cast<size_t>(d)]));
            v_caches[static_cast<size_t>(l)][static_cast<size_t>(pos * D_KV + d)] =
                cloudtpu::f16_to_f32(cloudtpu::f32_to_f16(v[static_cast<size_t>(d)]));
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(m.D));
        const int64_t gqa = m.H_Q / m.H_KV;
        std::vector<float> attn_out(static_cast<size_t>(D_Q), 0.0f);
        for (int64_t hq = 0; hq < m.H_Q; ++hq) {
            const int64_t hk = hq / gqa;
            std::vector<float> sc(static_cast<size_t>(m.N_KV), 0.0f);
            float smax = -std::numeric_limits<float>::infinity();
            for (int64_t t = 0; t < m.N_KV; ++t) {
                float dot = 0.0f;
                for (int64_t d = 0; d < m.D; ++d) {
                    dot += q[static_cast<size_t>(hq * m.D + d)] *
                           k_caches[static_cast<size_t>(l)][static_cast<size_t>(t * D_KV + hk * m.D + d)];
                }
                const float mask_v = (t <= pos) ? 0.0f : -std::numeric_limits<float>::infinity();
                sc[static_cast<size_t>(t)] = dot * scale + mask_v;
                if (sc[static_cast<size_t>(t)] > smax) smax = sc[static_cast<size_t>(t)];
            }
            float ssum = 0.0f;
            for (int64_t t = 0; t < m.N_KV; ++t) {
                sc[static_cast<size_t>(t)] = std::exp(sc[static_cast<size_t>(t)] - smax);
                ssum += sc[static_cast<size_t>(t)];
            }
            for (int64_t t = 0; t < m.N_KV; ++t) {
                const float p = sc[static_cast<size_t>(t)] / ssum;
                for (int64_t d = 0; d < m.D; ++d) {
                    attn_out[static_cast<size_t>(hq * m.D + d)] +=
                        p * v_caches[static_cast<size_t>(l)][static_cast<size_t>(t * D_KV + hk * m.D + d)];
                }
            }
        }
        const std::vector<float> oproj = matvec(ly.wo, attn_out, m.E, D_Q);
        for (int64_t i = 0; i < m.E; ++i) h[static_cast<size_t>(i)] += oproj[static_cast<size_t>(i)];

        const std::vector<float> fn = rms_mul(h, ly.ffn_norm);
        const std::vector<float> gf = matvec(ly.wgate, fn, m.H_ff, m.E);
        const std::vector<float> uf = matvec(ly.wup, fn, m.H_ff, m.E);
        std::vector<float> sw(static_cast<size_t>(m.H_ff));
        for (int64_t j = 0; j < m.H_ff; ++j) {
            const float gv = gf[static_cast<size_t>(j)];
            sw[static_cast<size_t>(j)] = (gv / (1.0f + std::exp(-gv))) * uf[static_cast<size_t>(j)];
        }
        const std::vector<float> dn = matvec(ly.wdown, sw, m.E, m.H_ff);
        for (int64_t i = 0; i < m.E; ++i) h[static_cast<size_t>(i)] += dn[static_cast<size_t>(i)];
    }
    return h;
}

std::vector<uint16_t> make_causal_mask_f16(int32_t pos, int64_t n_kv) {
    std::vector<uint16_t> mask(static_cast<size_t>(n_kv));
    for (int64_t t = 0; t < n_kv; ++t) {
        const float v = (t <= pos) ? 0.0f : -std::numeric_limits<float>::infinity();
        mask[static_cast<size_t>(t)] = cloudtpu::f32_to_f16(v);
    }
    return mask;
}

cloudtpu::GraphProgramSpec build_one_graph_decode_spec(const DecodeTestModel& m, int32_t spmd_chips) {
    const int64_t D_Q = m.H_Q * m.D;
    const int64_t D_KV = m.H_KV * m.D;
    const auto DENSE = cloudtpu::TensorStorageKind::DENSE;
    const auto F32 = cloudtpu::PjrtDType::F32;
    const auto F16 = cloudtpu::PjrtDType::F16;
    const auto S32 = cloudtpu::PjrtDType::S32;
    const auto REP = cloudtpu::TensorShardingKind::REPLICATED;

    cloudtpu::GraphProgramSpec spec;
    spec.spmd_chips = spmd_chips;
    spec.bf16_matmul = false;
    uint64_t next_id = 1;

    const uint64_t id_x0 = next_id++;
    const uint64_t id_pos = next_id++;
    const uint64_t id_mask = next_id++;
    spec.inputs.push_back({id_x0, F32, DENSE, {1, 1, 1, m.E}, REP});
    spec.inputs.push_back({id_pos, S32, DENSE, {1, 1, 1, 1}, REP});
    spec.inputs.push_back({id_mask, F16, DENSE, {1, 1, 1, m.N_KV}, REP});

    std::vector<uint64_t> kv_out_ids;
    uint64_t cur_x = id_x0;
    for (int64_t l = 0; l < m.n_layers; ++l) {
        auto add_in = [&](cloudtpu::PjrtDType dt, int64_t d2, int64_t d3, int32_t alias_idx = -1) {
            const uint64_t tid = next_id++;
            cloudtpu::TensorSpec ts{tid, dt, DENSE, {1, 1, d2, d3}, REP, alias_idx};
            spec.inputs.push_back(ts);
            return tid;
        };
        const uint64_t id_an = add_in(F32, 1, m.E);
        const uint64_t id_wq = add_in(F32, D_Q, m.E);
        const uint64_t id_wk = add_in(F32, D_KV, m.E);
        const uint64_t id_wv = add_in(F32, D_KV, m.E);
        const uint64_t id_kcache = add_in(F16, m.KV_CAP, D_KV, static_cast<int32_t>(2 * l));
        const uint64_t id_vcache = add_in(F16, m.KV_CAP, D_KV, static_cast<int32_t>(2 * l + 1));
        const uint64_t id_wo = add_in(F32, m.E, D_Q);
        const uint64_t id_fn = add_in(F32, 1, m.E);
        const uint64_t id_wg = add_in(F32, m.H_ff, m.E);
        const uint64_t id_wu = add_in(F32, m.H_ff, m.E);
        const uint64_t id_wd = add_in(F32, m.E, m.H_ff);

        auto add_op = [&](cloudtpu::CloudOpKind kind,
                          std::vector<uint64_t> srcs,
                          cloudtpu::PjrtDType dt,
                          int64_t d0,
                          int64_t d1,
                          int64_t d2,
                          int64_t d3) {
            const uint64_t oid = next_id++;
            cloudtpu::OpNodeSpec op;
            op.kind = kind;
            op.src_ids = std::move(srcs);
            op.dst = {oid, dt, DENSE, {d0, d1, d2, d3}};
            spec.ops.push_back(std::move(op));
            return spec.ops.back();
        };

        const uint64_t id_rn = [&]() {
            auto op = add_op(cloudtpu::CloudOpKind::RMS_NORM, {cur_x}, F32, 1, 1, 1, m.E);
            spec.ops.back().eps = 1e-5f;
            return op.dst.id;
        }();
        const uint64_t id_xin = add_op(cloudtpu::CloudOpKind::MUL, {id_rn, id_an}, F32, 1, 1, 1, m.E).dst.id;

        auto emit_mm = [&](uint64_t w, uint64_t act, int64_t out_dim) {
            auto op = add_op(cloudtpu::CloudOpKind::MUL_MAT, {w, act}, F32, 1, 1, 1, out_dim);
            spec.ops.back().high_precision = true;
            return op.dst.id;
        };
        const uint64_t id_q2d = emit_mm(id_wq, id_xin, D_Q);
        const uint64_t id_k2d = emit_mm(id_wk, id_xin, D_KV);
        const uint64_t id_v2d = emit_mm(id_wv, id_xin, D_KV);

        const uint64_t id_q3d = add_op(cloudtpu::CloudOpKind::RESHAPE, {id_q2d}, F32, 1, 1, m.H_Q, m.D).dst.id;
        const uint64_t id_k3d = add_op(cloudtpu::CloudOpKind::RESHAPE, {id_k2d}, F32, 1, 1, m.H_KV, m.D).dst.id;

        auto emit_rope = [&](uint64_t src, int64_t n_heads) {
            auto op = add_op(cloudtpu::CloudOpKind::ROPE, {src, id_pos}, F32, 1, 1, n_heads, m.D);
            spec.ops.back().rope.n_dims = static_cast<int32_t>(m.D);
            spec.ops.back().rope.mode = 2;
            spec.ops.back().rope.freq_base = 10000.0f;
            spec.ops.back().rope.freq_scale = 1.0f;
            spec.ops.back().rope.attn_factor = 1.0f;
            return op.dst.id;
        };
        const uint64_t id_qr = emit_rope(id_q3d, m.H_Q);
        const uint64_t id_kr = emit_rope(id_k3d, m.H_KV);
        const uint64_t id_kflat = add_op(cloudtpu::CloudOpKind::RESHAPE, {id_kr}, F32, 1, 1, 1, D_KV).dst.id;

        const uint64_t id_kset =
            add_op(cloudtpu::CloudOpKind::SET_ROWS, {id_kflat, id_pos, id_kcache}, F16, 1, 1, m.KV_CAP, D_KV).dst.id;
        const uint64_t id_vset =
            add_op(cloudtpu::CloudOpKind::SET_ROWS, {id_v2d, id_pos, id_vcache}, F16, 1, 1, m.KV_CAP, D_KV).dst.id;
        kv_out_ids.push_back(id_kset);
        kv_out_ids.push_back(id_vset);

        auto emit_kv_view_perm = [&](uint64_t src_set) {
            add_op(cloudtpu::CloudOpKind::SLICE, {src_set}, F16, 1, m.N_KV, m.H_KV, m.D);
            spec.ops.back().slice_starts = {0, 0, 0, 0};
            spec.ops.back().slice_limits = {1, 1, m.N_KV, D_KV};
            const uint64_t sl_id = spec.ops.back().dst.id;

            add_op(cloudtpu::CloudOpKind::TRANSPOSE, {sl_id}, F16, 1, m.H_KV, m.N_KV, m.D);
            spec.ops.back().perm[0] = 0;
            spec.ops.back().perm[1] = 2;
            spec.ops.back().perm[2] = 1;
            spec.ops.back().perm[3] = 3;
            return spec.ops.back().dst.id;
        };
        const uint64_t id_kp = emit_kv_view_perm(id_kset);
        const uint64_t id_vp = emit_kv_view_perm(id_vset);

        add_op(cloudtpu::CloudOpKind::TRANSPOSE, {id_qr}, F32, 1, m.H_Q, 1, m.D);
        spec.ops.back().perm[0] = 0;
        spec.ops.back().perm[1] = 2;
        spec.ops.back().perm[2] = 1;
        spec.ops.back().perm[3] = 3;
        const uint64_t id_qp = spec.ops.back().dst.id;

        add_op(cloudtpu::CloudOpKind::FLASH_ATTN_EXT, {id_qp, id_kp, id_vp, id_mask}, F32, 1, 1, m.H_Q, m.D);
        spec.ops.back().high_precision = true;
        spec.ops.back().flash_attn.scale = 1.0f / std::sqrt(static_cast<float>(m.D));
        const uint64_t id_attn = spec.ops.back().dst.id;

        const uint64_t id_aflat = add_op(cloudtpu::CloudOpKind::RESHAPE, {id_attn}, F32, 1, 1, 1, D_Q).dst.id;
        const uint64_t id_oproj = emit_mm(id_wo, id_aflat, m.E);
        const uint64_t id_hmid = add_op(cloudtpu::CloudOpKind::ADD, {cur_x, id_oproj}, F32, 1, 1, 1, m.E).dst.id;

        const uint64_t id_fnorm = [&]() {
            auto op = add_op(cloudtpu::CloudOpKind::RMS_NORM, {id_hmid}, F32, 1, 1, 1, m.E);
            spec.ops.back().eps = 1e-5f;
            return op.dst.id;
        }();
        const uint64_t id_fin = add_op(cloudtpu::CloudOpKind::MUL, {id_fnorm, id_fn}, F32, 1, 1, 1, m.E).dst.id;
        const uint64_t id_gf = emit_mm(id_wg, id_fin, m.H_ff);
        const uint64_t id_uf = emit_mm(id_wu, id_fin, m.H_ff);
        const uint64_t id_sw = add_op(cloudtpu::CloudOpKind::SWIGLU, {id_gf, id_uf}, F32, 1, 1, 1, m.H_ff).dst.id;
        const uint64_t id_dn = emit_mm(id_wd, id_sw, m.E);
        cur_x = add_op(cloudtpu::CloudOpKind::ADD, {id_hmid, id_dn}, F32, 1, 1, 1, m.E).dst.id;
    }

    spec.output_ids = kv_out_ids;
    spec.output_ids.push_back(cur_x);
    return spec;
}

void test_one_graph_multilayer_decode_and_donation() {
    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = true;
    cfg.exec_cache_cap = 64;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);

    const DecodeTestModel m = make_decode_test_model();
    const int64_t D_Q = m.H_Q * m.D;
    const int64_t D_KV = m.H_KV * m.D;

    const cloudtpu::GraphProgramSpec spec = build_one_graph_decode_spec(m, 1);
    const cloudtpu::EmittedProgram prog = cloudtpu::emit_stablehlo_program(spec);

    // Verify all 4 KV caches emit {tf.aliasing_output = 0..3 : i32} and donatable bindings
    int donatable_count = 0;
    for (const auto& b : prog.input_bindings) {
        if (b.donatable) ++donatable_count;
    }
    CHECK_TRUE(donatable_count == 4, "all 4 KV cache buffers must be donatable in 1-graph spec");
    for (int32_t k = 0; k < 4; ++k) {
        const std::string tag = "{tf.aliasing_output = " + std::to_string(k) + " : i32}";
        CHECK_TRUE(prog.mlir_text.find(tag) != std::string::npos, "missing aliasing_output attr in 1-graph MLIR");
    }

    struct LayerBufs {
        cloudtpu::PjrtBufferPtr an, wq, wk, wv, kc, vc, wo, fn, wg, wu, wd;
    };
    std::vector<LayerBufs> lbufs(static_cast<size_t>(m.n_layers));
    const std::vector<uint16_t> zeros_kv(static_cast<size_t>(m.KV_CAP * D_KV), 0);
    for (int64_t l = 0; l < m.n_layers; ++l) {
        const DecodeTestLayer& ly = m.layers[static_cast<size_t>(l)];
        LayerBufs& b = lbufs[static_cast<size_t>(l)];
        b.an = rt.buffer_from_host(0, ly.attn_norm.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, m.E});
        b.wq = rt.buffer_from_host(0, ly.wq.data(), cloudtpu::PjrtDType::F32, {1, 1, D_Q, m.E});
        b.wk = rt.buffer_from_host(0, ly.wk.data(), cloudtpu::PjrtDType::F32, {1, 1, D_KV, m.E});
        b.wv = rt.buffer_from_host(0, ly.wv.data(), cloudtpu::PjrtDType::F32, {1, 1, D_KV, m.E});
        b.kc = rt.buffer_from_host(0, zeros_kv.data(), cloudtpu::PjrtDType::F16, {1, 1, m.KV_CAP, D_KV});
        b.vc = rt.buffer_from_host(0, zeros_kv.data(), cloudtpu::PjrtDType::F16, {1, 1, m.KV_CAP, D_KV});
        b.wo = rt.buffer_from_host(0, ly.wo.data(), cloudtpu::PjrtDType::F32, {1, 1, m.E, D_Q});
        b.fn = rt.buffer_from_host(0, ly.ffn_norm.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, m.E});
        b.wg = rt.buffer_from_host(0, ly.wgate.data(), cloudtpu::PjrtDType::F32, {1, 1, m.H_ff, m.E});
        b.wu = rt.buffer_from_host(0, ly.wup.data(), cloudtpu::PjrtDType::F32, {1, 1, m.H_ff, m.E});
        b.wd = rt.buffer_from_host(0, ly.wdown.data(), cloudtpu::PjrtDType::F32, {1, 1, m.E, m.H_ff});
    }

    std::vector<cloudtpu::PjrtDType> out_dts;
    std::vector<std::vector<int64_t>> out_dims;
    for (const auto& ob : prog.output_bindings) {
        out_dts.push_back(ob.dtype);
        out_dims.push_back(ob.dims);
    }

    std::vector<std::vector<float>> ref_k(static_cast<size_t>(m.n_layers),
                                          std::vector<float>(static_cast<size_t>(m.KV_CAP * D_KV), 0.0f));
    std::vector<std::vector<float>> ref_v(static_cast<size_t>(m.n_layers),
                                          std::vector<float>(static_cast<size_t>(m.KV_CAP * D_KV), 0.0f));

    uint64_t misses_after_0 = 0;
    uint64_t hits_after_0 = 0;
    const auto copts = rt.make_spmd_compile_options(1, 0, true, true);
    for (int32_t pos = 0; pos < 4; ++pos) {
        std::vector<float> x_step(static_cast<size_t>(m.E));
        for (int64_t i = 0; i < m.E; ++i) {
            x_step[static_cast<size_t>(i)] =
                0.2f * std::cos(static_cast<float>(pos * 13 + i) * 0.21f);
        }
        const std::vector<uint16_t> mask_f16 = make_causal_mask_f16(pos, m.N_KV);

        auto bx = rt.buffer_from_host(0, x_step.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, m.E});
        auto bp = rt.buffer_from_host(0, &pos, cloudtpu::PjrtDType::S32, {1, 1, 1, 1});
        auto bm = rt.buffer_from_host(0, mask_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, 1, m.N_KV});

        std::vector<PJRT_Buffer*> raw_in = {bx->handle.get(), bp->handle.get(), bm->handle.get()};
        for (int64_t l = 0; l < m.n_layers; ++l) {
            const LayerBufs& b = lbufs[static_cast<size_t>(l)];
            raw_in.push_back(b.an->handle.get());
            raw_in.push_back(b.wq->handle.get());
            raw_in.push_back(b.wk->handle.get());
            raw_in.push_back(b.wv->handle.get());
            raw_in.push_back(b.kc->handle.get());
            raw_in.push_back(b.vc->handle.get());
            raw_in.push_back(b.wo->handle.get());
            raw_in.push_back(b.fn->handle.get());
            raw_in.push_back(b.wg->handle.get());
            raw_in.push_back(b.wu->handle.get());
            raw_in.push_back(b.wd->handle.get());
        }

        const auto* exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, copts);
        std::vector<cloudtpu::PjrtBufferPtr> outs;
        rt.execute(exec, 0, raw_in.data(), raw_in.size(), outs, out_dts, out_dims, true);
        CHECK_TRUE(outs.size() == 5, "expected 4 donated KV outputs + 1 hidden output");

        // Feed donated KV buffers into next decode position without H2D re-upload
        for (int64_t l = 0; l < m.n_layers; ++l) {
            lbufs[static_cast<size_t>(l)].kc = outs[static_cast<size_t>(2 * l)];
            lbufs[static_cast<size_t>(l)].vc = outs[static_cast<size_t>(2 * l + 1)];
        }

        const cloudtpu::CacheStats st = rt.cache_stats();
        if (pos == 0) {
            misses_after_0 = st.misses;
            hits_after_0 = st.hits;
        } else {
            CHECK_TRUE(st.misses == misses_after_0, "zero recompilations expected on advancing decode pos");
            CHECK_TRUE(st.hits == hits_after_0 + static_cast<uint64_t>(pos), "plan/cache hits must track pos");
        }

        std::vector<float> got(static_cast<size_t>(m.E));
        rt.buffer_to_host(outs[4]->handle.get(), got.data(), got.size() * sizeof(float));
        const std::vector<float> exp = scalar_decode_step(m, x_step, pos, ref_k, ref_v);
        for (int64_t i = 0; i < m.E; ++i) {
            CHECK_NEAR(got[static_cast<size_t>(i)], exp[static_cast<size_t>(i)], 1e-4f,
                       "1-graph 2-layer decode parity across pos");
        }
    }

    rt.shutdown();
    std::printf("  [PASS] 1-graph 2-layer fused decode + 4x KV donation (0 recompiles across pos=0..3, parity < 1e-4)\n");
}

void test_one_graph_spmd_kv_donation(int num_chips) {
    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = true;
    cfg.mock_devices = num_chips;
    cfg.exec_cache_cap = 64;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);

    const DecodeTestModel m = make_decode_test_model();
    const int64_t D_Q = m.H_Q * m.D;
    const int64_t D_KV = m.H_KV * m.D;
    const cloudtpu::GraphProgramSpec spec = build_one_graph_decode_spec(m, num_chips);
    const cloudtpu::EmittedProgram prog = cloudtpu::emit_stablehlo_program(spec);
    const auto copts = rt.make_spmd_compile_options(num_chips);
    const auto* exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, copts);

    std::vector<cloudtpu::PjrtDType> out_dts;
    std::vector<std::vector<int64_t>> out_dims;
    for (const auto& ob : prog.output_bindings) {
        out_dts.push_back(ob.dtype);
        out_dims.push_back(ob.dims);
    }

    const std::vector<uint16_t> zeros_kv(static_cast<size_t>(m.KV_CAP * D_KV), 0);
    struct PerChipLayer {
        cloudtpu::PjrtBufferPtr an, wq, wk, wv, kc, vc, wo, fn, wg, wu, wd;
    };
    std::vector<std::vector<PerChipLayer>> dev_layers(static_cast<size_t>(num_chips));
    for (int d = 0; d < num_chips; ++d) {
        dev_layers[static_cast<size_t>(d)].resize(static_cast<size_t>(m.n_layers));
        for (int64_t l = 0; l < m.n_layers; ++l) {
            const DecodeTestLayer& ly = m.layers[static_cast<size_t>(l)];
            PerChipLayer& b = dev_layers[static_cast<size_t>(d)][static_cast<size_t>(l)];
            b.an = rt.buffer_from_host(d, ly.attn_norm.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, m.E});
            b.wq = rt.buffer_from_host(d, ly.wq.data(), cloudtpu::PjrtDType::F32, {1, 1, D_Q, m.E});
            b.wk = rt.buffer_from_host(d, ly.wk.data(), cloudtpu::PjrtDType::F32, {1, 1, D_KV, m.E});
            b.wv = rt.buffer_from_host(d, ly.wv.data(), cloudtpu::PjrtDType::F32, {1, 1, D_KV, m.E});
            b.kc = rt.buffer_from_host(d, zeros_kv.data(), cloudtpu::PjrtDType::F16, {1, 1, m.KV_CAP, D_KV});
            b.vc = rt.buffer_from_host(d, zeros_kv.data(), cloudtpu::PjrtDType::F16, {1, 1, m.KV_CAP, D_KV});
            b.wo = rt.buffer_from_host(d, ly.wo.data(), cloudtpu::PjrtDType::F32, {1, 1, m.E, D_Q});
            b.fn = rt.buffer_from_host(d, ly.ffn_norm.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, m.E});
            b.wg = rt.buffer_from_host(d, ly.wgate.data(), cloudtpu::PjrtDType::F32, {1, 1, m.H_ff, m.E});
            b.wu = rt.buffer_from_host(d, ly.wup.data(), cloudtpu::PjrtDType::F32, {1, 1, m.H_ff, m.E});
            b.wd = rt.buffer_from_host(d, ly.wdown.data(), cloudtpu::PjrtDType::F32, {1, 1, m.E, m.H_ff});
        }
    }

    std::vector<std::vector<float>> ref_k(static_cast<size_t>(m.n_layers),
                                          std::vector<float>(static_cast<size_t>(m.KV_CAP * D_KV), 0.0f));
    std::vector<std::vector<float>> ref_v(static_cast<size_t>(m.n_layers),
                                          std::vector<float>(static_cast<size_t>(m.KV_CAP * D_KV), 0.0f));

    for (int32_t pos = 0; pos < 3; ++pos) {
        std::vector<float> x_step(static_cast<size_t>(m.E));
        for (int64_t i = 0; i < m.E; ++i) x_step[static_cast<size_t>(i)] = 0.25f * std::sin(static_cast<float>(pos * 7 + i) * 0.19f);
        const std::vector<uint16_t> mask_f16 = make_causal_mask_f16(pos, m.N_KV);

        std::vector<cloudtpu::PjrtBufferPtr> bx(static_cast<size_t>(num_chips)), bp(static_cast<size_t>(num_chips)),
            bm(static_cast<size_t>(num_chips));
        std::vector<std::vector<PJRT_Buffer*>> raw_in(static_cast<size_t>(num_chips));
        std::vector<PJRT_Buffer* const*> in_ptrs(static_cast<size_t>(num_chips));
        for (int d = 0; d < num_chips; ++d) {
            bx[static_cast<size_t>(d)] = rt.buffer_from_host(d, x_step.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, m.E});
            bp[static_cast<size_t>(d)] = rt.buffer_from_host(d, &pos, cloudtpu::PjrtDType::S32, {1, 1, 1, 1});
            bm[static_cast<size_t>(d)] = rt.buffer_from_host(d, mask_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, 1, m.N_KV});
            raw_in[static_cast<size_t>(d)] = {
                bx[static_cast<size_t>(d)]->handle.get(),
                bp[static_cast<size_t>(d)]->handle.get(),
                bm[static_cast<size_t>(d)]->handle.get(),
            };
            for (int64_t l = 0; l < m.n_layers; ++l) {
                const PerChipLayer& b = dev_layers[static_cast<size_t>(d)][static_cast<size_t>(l)];
                raw_in[static_cast<size_t>(d)].push_back(b.an->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.wq->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.wk->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.wv->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.kc->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.vc->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.wo->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.fn->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.wg->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.wu->handle.get());
                raw_in[static_cast<size_t>(d)].push_back(b.wd->handle.get());
            }
            in_ptrs[static_cast<size_t>(d)] = raw_in[static_cast<size_t>(d)].data();
        }

        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
        rt.execute_spmd(exec, num_chips, in_ptrs.data(), prog.input_bindings.size(), outs, out_dts, out_dims, true);
        const std::vector<float> exp = scalar_decode_step(m, x_step, pos, ref_k, ref_v);
        for (int d = 0; d < num_chips; ++d) {
            for (int64_t l = 0; l < m.n_layers; ++l) {
                dev_layers[static_cast<size_t>(d)][static_cast<size_t>(l)].kc = outs[static_cast<size_t>(d)][static_cast<size_t>(2 * l)];
                dev_layers[static_cast<size_t>(d)][static_cast<size_t>(l)].vc = outs[static_cast<size_t>(d)][static_cast<size_t>(2 * l + 1)];
            }
            std::vector<float> got(static_cast<size_t>(m.E));
            rt.buffer_to_host(outs[static_cast<size_t>(d)][4]->handle.get(), got.data(), got.size() * sizeof(float));
            for (int64_t i = 0; i < m.E; ++i) {
                CHECK_NEAR(got[static_cast<size_t>(i)], exp[static_cast<size_t>(i)], 1e-4f, "SPMD multi-step KV donation parity");
            }
        }
    }
    rt.shutdown();
    std::printf("  [PASS] Multi-chip SPMD (%d chips) 1-graph 2-layer decode + repeated KV donation\n", num_chips);
}

#ifdef CLOUDTPU_HAS_GGML
struct ScopedEnv {
    std::string key;
    bool had = false;
    std::string old_val;
    explicit ScopedEnv(const char* k, const char* v) : key(k) {
        const char* cur = std::getenv(k);
        if (cur) {
            had = true;
            old_val = cur;
        }
        if (v) {
            setenv(k, v, 1);
        } else {
            unsetenv(k);
        }
    }
    ~ScopedEnv() {
        if (had) {
            setenv(key.c_str(), old_val.c_str(), 1);
        } else {
            unsetenv(key.c_str());
        }
    }
};

struct GgmlLogCapture {
    std::string buf;
    ggml_log_callback prev_cb = nullptr;
    void* prev_ud = nullptr;
    static void callback(enum ggml_log_level level, const char* text, void* user_data) {
        (void) level;
        if (text && user_data) {
            static_cast<GgmlLogCapture*>(user_data)->buf += text;
        }
    }
    GgmlLogCapture() {
        ggml_log_get(&prev_cb, &prev_ud);
        ggml_log_set(&GgmlLogCapture::callback, this);
    }
    ~GgmlLogCapture() {
        ggml_log_set(prev_cb, prev_ud);
    }
};

uint64_t parse_log_u64(const std::string& log, const char* field) {
    const std::string key = std::string(field) + "=";
    const size_t pos = log.rfind(key);
    if (pos == std::string::npos) return UINT64_MAX;
    return std::strtoull(log.c_str() + pos + key.size(), nullptr, 10);
}

struct GgmlDecodeExecResult {
    std::vector<std::vector<float>> step_outputs;
    std::vector<uint16_t> k0_after;
    std::vector<uint16_t> v0_after;
    std::string logs;
};

GgmlDecodeExecResult run_ggml_decode_sequence(const DecodeTestModel& m,
                                              ggml_backend_t backend,
                                              int num_steps,
                                              bool* ok_out) {
    GgmlDecodeExecResult res;
    *ok_out = false;
    const int64_t D_Q = m.H_Q * m.D;
    const int64_t D_KV = m.H_KV * m.D;

    const size_t w_mem = ggml_tensor_overhead() * 64 + 1024;
    struct ggml_init_params wp = {w_mem, nullptr, true};
    struct ggml_context* ctx_w = ggml_init(wp);
    struct ggml_context* ctx_kv = ggml_init(wp);
    if (!ctx_w || !ctx_kv) return res;

    struct WLayer {
        ggml_tensor *an, *wq, *wk, *wv, *wo, *fn, *wg, *wu, *wd;
        ggml_tensor *kc, *vc;
    };
    std::vector<WLayer> wl(static_cast<size_t>(m.n_layers));
    for (int64_t l = 0; l < m.n_layers; ++l) {
        WLayer& w = wl[static_cast<size_t>(l)];
        w.an = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, m.E);
        w.wq = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, m.E, D_Q);
        w.wk = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, m.E, D_KV);
        w.wv = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, m.E, D_KV);
        w.wo = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, D_Q, m.E);
        w.fn = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, m.E);
        w.wg = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, m.E, m.H_ff);
        w.wu = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, m.E, m.H_ff);
        w.wd = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, m.H_ff, m.E);
        w.kc = ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F16, D_KV, m.KV_CAP);
        w.vc = ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F16, D_KV, m.KV_CAP);
    }

    ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_t kv_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, backend);
    if (!w_buf || !kv_buf) {
        if (kv_buf) ggml_backend_buffer_free(kv_buf);
        if (w_buf) ggml_backend_buffer_free(w_buf);
        ggml_free(ctx_kv);
        ggml_free(ctx_w);
        return res;
    }

    const std::vector<uint16_t> zeros_kv(static_cast<size_t>(m.KV_CAP * D_KV), 0);
    for (int64_t l = 0; l < m.n_layers; ++l) {
        const DecodeTestLayer& ly = m.layers[static_cast<size_t>(l)];
        const WLayer& w = wl[static_cast<size_t>(l)];
        ggml_backend_tensor_set(w.an, ly.attn_norm.data(), 0, ly.attn_norm.size() * sizeof(float));
        ggml_backend_tensor_set(w.wq, ly.wq.data(), 0, ly.wq.size() * sizeof(float));
        ggml_backend_tensor_set(w.wk, ly.wk.data(), 0, ly.wk.size() * sizeof(float));
        ggml_backend_tensor_set(w.wv, ly.wv.data(), 0, ly.wv.size() * sizeof(float));
        ggml_backend_tensor_set(w.wo, ly.wo.data(), 0, ly.wo.size() * sizeof(float));
        ggml_backend_tensor_set(w.fn, ly.ffn_norm.data(), 0, ly.ffn_norm.size() * sizeof(float));
        ggml_backend_tensor_set(w.wg, ly.wgate.data(), 0, ly.wgate.size() * sizeof(float));
        ggml_backend_tensor_set(w.wu, ly.wup.data(), 0, ly.wup.size() * sizeof(float));
        ggml_backend_tensor_set(w.wd, ly.wdown.data(), 0, ly.wdown.size() * sizeof(float));
        ggml_backend_tensor_set(w.kc, zeros_kv.data(), 0, zeros_kv.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(w.vc, zeros_kv.data(), 0, zeros_kv.size() * sizeof(uint16_t));
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    for (int32_t pos = 0; pos < num_steps; ++pos) {
        const size_t g_mem = ggml_tensor_overhead() * 256 + ggml_graph_overhead() + 4096;
        struct ggml_init_params gp = {g_mem, nullptr, true};
        struct ggml_context* ctx_g = ggml_init(gp);
        struct ggml_cgraph* gf = ggml_new_graph(ctx_g);

        ggml_tensor* inp_x = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, m.E, 1);
        ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_g, GGML_TYPE_I32, 1);
        ggml_tensor* inp_row_idx = ggml_new_tensor_1d(ctx_g, GGML_TYPE_I64, 1);
        ggml_tensor* inp_mask = ggml_new_tensor_4d(ctx_g, GGML_TYPE_F16, m.N_KV, 1, 1, 1);
        ggml_set_input(inp_x);
        ggml_set_input(inp_pos);
        ggml_set_input(inp_row_idx);
        ggml_set_input(inp_mask);

        ggml_tensor* cur = inp_x;
        for (int64_t l = 0; l < m.n_layers; ++l) {
            const WLayer& w = wl[static_cast<size_t>(l)];
            ggml_tensor* rn = ggml_rms_norm(ctx_g, cur, 1e-5f);
            ggml_tensor* xin = ggml_mul(ctx_g, rn, w.an);
            ggml_tensor* q = ggml_mul_mat(ctx_g, w.wq, xin);
            ggml_tensor* k = ggml_mul_mat(ctx_g, w.wk, xin);
            ggml_tensor* v = ggml_mul_mat(ctx_g, w.wv, xin);

            ggml_tensor* q3 = ggml_reshape_3d(ctx_g, q, m.D, m.H_Q, 1);
            ggml_tensor* k3 = ggml_reshape_3d(ctx_g, k, m.D, m.H_KV, 1);
            ggml_tensor* qr = ggml_rope_ext(
                ctx_g, q3, inp_pos, nullptr, static_cast<int>(m.D), GGML_ROPE_TYPE_NEOX, 0,
                10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor* kr = ggml_rope_ext(
                ctx_g, k3, inp_pos, nullptr, static_cast<int>(m.D), GGML_ROPE_TYPE_NEOX, 0,
                10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor* kflat = ggml_reshape_2d(ctx_g, kr, D_KV, 1);

            ggml_tensor* kset = ggml_set_rows(ctx_g, w.kc, kflat, inp_row_idx);
            ggml_tensor* vset = ggml_set_rows(ctx_g, w.vc, v, inp_row_idx);
            ggml_build_forward_expand(gf, kset);
            ggml_build_forward_expand(gf, vset);

            ggml_tensor* kview = ggml_view_4d(
                ctx_g, w.kc, m.D, m.H_KV, m.N_KV, 1,
                static_cast<size_t>(m.D) * ggml_element_size(w.kc), w.kc->nb[1], w.kc->nb[2], 0);
            ggml_tensor* vview = ggml_view_4d(
                ctx_g, w.vc, m.D, m.H_KV, m.N_KV, 1,
                static_cast<size_t>(m.D) * ggml_element_size(w.vc), w.vc->nb[1], w.vc->nb[2], 0);
            ggml_tensor* qp = ggml_permute(ctx_g, qr, 0, 2, 1, 3);
            ggml_tensor* kp = ggml_permute(ctx_g, kview, 0, 2, 1, 3);
            ggml_tensor* vp = ggml_permute(ctx_g, vview, 0, 2, 1, 3);

            ggml_tensor* attn = ggml_flash_attn_ext(
                ctx_g, qp, kp, vp, inp_mask, 1.0f / std::sqrt(static_cast<float>(m.D)), 0.0f, 0.0f);
            ggml_prec_set_acc(attn, GGML_PREC_F32);
            ggml_tensor* aflat = ggml_reshape_2d(ctx_g, attn, D_Q, 1);
            ggml_tensor* oproj = ggml_mul_mat(ctx_g, w.wo, aflat);
            ggml_tensor* hmid = ggml_add(ctx_g, cur, oproj);

            ggml_tensor* fn = ggml_rms_norm(ctx_g, hmid, 1e-5f);
            ggml_tensor* fin = ggml_mul(ctx_g, fn, w.fn);
            ggml_tensor* fg = ggml_mul_mat(ctx_g, w.wg, fin);
            ggml_tensor* fu = ggml_mul_mat(ctx_g, w.wu, fin);
            ggml_tensor* sw = ggml_swiglu_split(ctx_g, fg, fu);
            ggml_tensor* dn = ggml_mul_mat(ctx_g, w.wd, sw);
            cur = ggml_add(ctx_g, hmid, dn);
        }
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            ggml_free(ctx_g);
            ggml_gallocr_free(galloc);
            ggml_backend_buffer_free(kv_buf);
            ggml_backend_buffer_free(w_buf);
            ggml_free(ctx_kv);
            ggml_free(ctx_w);
            return res;
        }

        std::vector<float> x_step(static_cast<size_t>(m.E));
        for (int64_t i = 0; i < m.E; ++i) {
            x_step[static_cast<size_t>(i)] =
                0.2f * std::cos(static_cast<float>(pos * 13 + i) * 0.21f);
        }
        const int64_t row_idx64 = pos;
        const std::vector<uint16_t> mask_f16 = make_causal_mask_f16(pos, m.N_KV);
        ggml_backend_tensor_set(inp_x, x_step.data(), 0, x_step.size() * sizeof(float));
        ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(pos));
        ggml_backend_tensor_set(inp_row_idx, &row_idx64, 0, sizeof(row_idx64));
        ggml_backend_tensor_set(inp_mask, mask_f16.data(), 0, mask_f16.size() * sizeof(uint16_t));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx_g);
            ggml_gallocr_free(galloc);
            ggml_backend_buffer_free(kv_buf);
            ggml_backend_buffer_free(w_buf);
            ggml_free(ctx_kv);
            ggml_free(ctx_w);
            return res;
        }
        std::vector<float> y(static_cast<size_t>(m.E), 0.0f);
        ggml_backend_tensor_get(cur, y.data(), 0, y.size() * sizeof(float));
        res.step_outputs.push_back(std::move(y));
        ggml_free(ctx_g);
    }

    res.k0_after.resize(static_cast<size_t>(m.KV_CAP * D_KV));
    res.v0_after.resize(static_cast<size_t>(m.KV_CAP * D_KV));
    ggml_backend_tensor_get(wl[0].kc, res.k0_after.data(), 0, res.k0_after.size() * sizeof(uint16_t));
    ggml_backend_tensor_get(wl[0].vc, res.v0_after.data(), 0, res.v0_after.size() * sizeof(uint16_t));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(kv_buf);
    ggml_backend_buffer_free(w_buf);
    ggml_free(ctx_kv);
    ggml_free(ctx_w);
    *ok_out = true;
    return res;
}

void test_ggml_cgraph_one_graph_decode() {
    ScopedEnv env_mock("CLOUDTPU_MOCK", "1");
    ScopedEnv env_verb("CLOUDTPU_VERBOSE", "1");
    ScopedEnv env_fuse_cap("CLOUDTPU_MAX_FUSE_OPS", nullptr);   // Verify default 4096
    ScopedEnv env_fuse_attn("CLOUDTPU_FUSE_ATTN", nullptr);     // Verify default true
    ScopedEnv env_fuse_graphs("CLOUDTPU_FUSE_GRAPHS", nullptr); // Verify default true
    ScopedEnv env_bf16("CLOUDTPU_BF16_MATMUL", "0");            // Isolate fusion/donation parity (< 1e-4)

    const DecodeTestModel m = make_decode_test_model();

    // CPU reference across pos=0..3
    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    CHECK_TRUE(cpu_be != nullptr, "cpu backend init");
    bool ok_cpu = false;
    const auto cpu_res = run_ggml_decode_sequence(m, cpu_be, 4, &ok_cpu);
    ggml_backend_free(cpu_be);
    CHECK_TRUE(ok_cpu && cpu_res.step_outputs.size() == 4, "cpu 2-layer decode reference");

    // 1) Single step under default fusion caps/flags: MUST collapse to subgraphs == 1
    {
        GgmlLogCapture cap;
        ggml_backend_t tpu_be = ggml_backend_cloudtpu_init();
        CHECK_TRUE(tpu_be != nullptr, "cloudtpu init (1-step)");
        bool ok_tpu = false;
        const auto tpu_res = run_ggml_decode_sequence(m, tpu_be, 1, &ok_tpu);
        ggml_backend_free(tpu_be);
        CHECK_TRUE(ok_tpu && tpu_res.step_outputs.size() == 1, "cloudtpu 1-step 2-layer decode");
        const uint64_t evals = parse_log_u64(cap.buf, "evals");
        const uint64_t subgraphs = parse_log_u64(cap.buf, "subgraphs");
        const uint64_t misses = parse_log_u64(cap.buf, "plan_misses");
        const uint64_t donated = parse_log_u64(cap.buf, "kv_donated");
        const uint64_t skipped = parse_log_u64(cap.buf, "kv_donation_skipped");
        CHECK_TRUE(evals == 1 && subgraphs == 1 && misses == 1,
                   "2-layer RMS_NORM->QKV->ROPE->SET_ROWS->VIEW->FLASH_ATTN_EXT->O_PROJ->FFN must fuse into 1 subgraph");
        CHECK_TRUE(donated == 4 && skipped == 0,
                   "all 4 KV cache buffers must donate cleanly with zero non_donatable conflicts");
        for (int64_t i = 0; i < m.E; ++i) {
            CHECK_NEAR(tpu_res.step_outputs[0][static_cast<size_t>(i)],
                       cpu_res.step_outputs[0][static_cast<size_t>(i)],
                       1e-4f, "ggml 1-step parity vs CPU");
        }
    }

    // 2) Repeated pos=0..3 calls: subgraphs == 4 (1/token), plan_hits == 3, plan_misses == 1, KV readback matches
    GgmlDecodeExecResult tpu_4res;
    {
        GgmlLogCapture cap;
        ggml_backend_t tpu_be = ggml_backend_cloudtpu_init();
        CHECK_TRUE(tpu_be != nullptr, "cloudtpu init (4-step)");
        bool ok_tpu = false;
        tpu_4res = run_ggml_decode_sequence(m, tpu_be, 4, &ok_tpu);
        ggml_backend_free(tpu_be);
        CHECK_TRUE(ok_tpu && tpu_4res.step_outputs.size() == 4, "cloudtpu 4-step 2-layer decode");
        const uint64_t evals = parse_log_u64(cap.buf, "evals");
        const uint64_t subgraphs = parse_log_u64(cap.buf, "subgraphs");
        const uint64_t hits = parse_log_u64(cap.buf, "plan_hits");
        const uint64_t misses = parse_log_u64(cap.buf, "plan_misses");
        const uint64_t donated = parse_log_u64(cap.buf, "kv_donated");
        const uint64_t skipped = parse_log_u64(cap.buf, "kv_donation_skipped");
        CHECK_TRUE(evals == 4 && subgraphs == 4 && hits == 3 && misses == 1,
                   "4 decode steps must yield 4 subgraphs, 3 plan_hits, 1 plan_miss");
        CHECK_TRUE(donated == 16 && skipped == 0,
                   "all 4 KV cache buffers must donate cleanly across pos=0..3");
        for (int s = 0; s < 4; ++s) {
            for (int64_t i = 0; i < m.E; ++i) {
                CHECK_NEAR(tpu_4res.step_outputs[static_cast<size_t>(s)][static_cast<size_t>(i)],
                           cpu_res.step_outputs[static_cast<size_t>(s)][static_cast<size_t>(i)],
                           1e-4f, "ggml 4-step 1-graph parity vs CPU");
            }
        }
        for (size_t i = 0; i < cpu_res.k0_after.size(); ++i) {
            CHECK_NEAR(cloudtpu::f16_to_f32(tpu_4res.k0_after[i]),
                       cloudtpu::f16_to_f32(cpu_res.k0_after[i]),
                       2e-3f, "donated K0 cache readback parity vs CPU");
            CHECK_NEAR(cloudtpu::f16_to_f32(tpu_4res.v0_after[i]),
                       cloudtpu::f16_to_f32(cpu_res.v0_after[i]),
                       2e-3f, "donated V0 cache readback parity vs CPU");
        }
    }

    // 3) CLOUDTPU_FUSE_ATTN=0 splits at FLASH_ATTN_EXT (2+ pass reference) and matches 1-graph identically
    {
        ScopedEnv env_no_attn("CLOUDTPU_FUSE_ATTN", "0");
        GgmlLogCapture cap;
        ggml_backend_t tpu_split = ggml_backend_cloudtpu_init();
        CHECK_TRUE(tpu_split != nullptr, "cloudtpu init (CLOUDTPU_FUSE_ATTN=0)");
        bool ok_split = false;
        const auto split_res = run_ggml_decode_sequence(m, tpu_split, 4, &ok_split);
        ggml_backend_free(tpu_split);
        CHECK_TRUE(ok_split && split_res.step_outputs.size() == 4, "cloudtpu 2-pass split reference");
        const uint64_t subgraphs = parse_log_u64(cap.buf, "subgraphs");
        CHECK_TRUE(subgraphs > 4, "CLOUDTPU_FUSE_ATTN=0 must split attention subgraphs");
        for (int s = 0; s < 4; ++s) {
            for (int64_t i = 0; i < m.E; ++i) {
                CHECK_NEAR(tpu_4res.step_outputs[static_cast<size_t>(s)][static_cast<size_t>(i)],
                           split_res.step_outputs[static_cast<size_t>(s)][static_cast<size_t>(i)],
                           1e-5f, "1-graph fused vs CLOUDTPU_FUSE_ATTN=0 2-pass reference");
            }
        }
    }

    // 4) CLOUDTPU_MAX_FUSE_OPS=25 env override splits cleanly while preserving parity
    {
        ScopedEnv env_cap("CLOUDTPU_MAX_FUSE_OPS", "25");
        GgmlLogCapture cap;
        ggml_backend_t tpu_cap = ggml_backend_cloudtpu_init();
        CHECK_TRUE(tpu_cap != nullptr, "cloudtpu init (CLOUDTPU_MAX_FUSE_OPS=25)");
        bool ok_cap = false;
        const auto cap_res = run_ggml_decode_sequence(m, tpu_cap, 1, &ok_cap);
        ggml_backend_free(tpu_cap);
        CHECK_TRUE(ok_cap && parse_log_u64(cap.buf, "subgraphs") > 1,
                   "CLOUDTPU_MAX_FUSE_OPS=25 override must split 2-layer decode");
        for (int64_t i = 0; i < m.E; ++i) {
            CHECK_NEAR(cap_res.step_outputs[0][static_cast<size_t>(i)],
                       cpu_res.step_outputs[0][static_cast<size_t>(i)],
                       1e-4f, "CLOUDTPU_MAX_FUSE_OPS=25 parity vs CPU");
        }
    }

    // 5) Default CLOUDTPU_BF16_MATMUL (unset) still fuses into subgraphs == 1 & passes bounded parity
    {
        ScopedEnv env_bf16_default("CLOUDTPU_BF16_MATMUL", nullptr);
        GgmlLogCapture cap;
        ggml_backend_t tpu_def = ggml_backend_cloudtpu_init();
        CHECK_TRUE(tpu_def != nullptr, "cloudtpu init (default env)");
        bool ok_def = false;
        const auto def_res = run_ggml_decode_sequence(m, tpu_def, 1, &ok_def);
        ggml_backend_free(tpu_def);
        CHECK_TRUE(ok_def && parse_log_u64(cap.buf, "subgraphs") == 1,
                   "default CLOUDTPU env must execute 2-layer decode in 1 subgraph");
        for (int64_t i = 0; i < m.E; ++i) {
            CHECK_NEAR(def_res.step_outputs[0][static_cast<size_t>(i)],
                       cpu_res.step_outputs[0][static_cast<size_t>(i)],
                       1e-2f, "default BF16 1-graph bounded parity vs CPU");
        }
    }

    std::printf("  [PASS] ggml_cgraph 2-layer 1-graph decode + KV donation + plan_hits=3 parity\n");
}
#endif

}  // namespace

int main() {
    bool mock_mode = true;
    if (const char* v = std::getenv("CLOUDTPU_MOCK")) {
        mock_mode = (std::atoi(v) != 0);
    }
    std::printf("Running ggml-cloudtpu standalone tests (mock_mode=%d)...\n", mock_mode ? 1 : 0);

    test_compile_options_proto();
    test_half_bf16_conversions();

    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = mock_mode;
    cfg.exec_cache_cap = 256;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);

    std::printf("  Platform: %s (%s), device[0]: %s\n",
                rt.platform_name().c_str(),
                rt.platform_version().c_str(),
                rt.device_info(0).kind.c_str());

    test_lru_cache_and_hbm_roundtrip(rt);
    test_mul_mat_dense_and_quant(rt);
    test_rms_rope_softmax_swiglu_getrows_fused(rt);
    test_silicon_regressions(rt);
    test_collectives_p1_live_compatible(rt);
    test_spmd_multichip_mesh(4);  // tpu7x-4t (4x 192 GiB HBM3e)
    test_spmd_multichip_mesh(8);  // v6e-8 (8x 32 GiB HBM)
    test_one_graph_multilayer_decode_and_donation();
    test_one_graph_spmd_kv_donation(4);
#ifdef CLOUDTPU_HAS_GGML
    test_ggml_cgraph_one_graph_decode();
#endif

    const size_t py_maps = cloudtpu::count_libpython_mappings();
    std::printf("  MAPS_LIBPYTHON=%zu\n", py_maps);
    if (py_maps != 0) {
        std::fprintf(stderr, "FAIL: expected MAPS_LIBPYTHON=0, got %zu\n", py_maps);
        ++g_failures;
    }

    rt.shutdown();
    std::printf("OVERALL: %s (%d failures)\n", (g_failures == 0) ? "ALL PASSED" : "FAILED", g_failures);
    return (g_failures == 0) ? 0 : 1;
}
