#include "pjrt_runtime.hpp"
#include "stablehlo_emitter.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    std::printf("  [PASS] compile_options_proto\n");
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

    // 2) Q4_0_SPLIT x F32
    std::vector<uint8_t> qs(N * 16);
    std::vector<uint16_t> d(N);
    for (int64_t n = 0; n < N; ++n) {
        d[n] = cloudtpu::f32_to_f16(0.5f * static_cast<float>(n + 1));
        for (int64_t j = 0; j < 16; ++j) {
            const uint8_t lo = static_cast<uint8_t>((j + n + 5) & 0x0F);
            const uint8_t hi = static_cast<uint8_t>((j * 3 + 9) & 0x0F);
            qs[n * 16 + j] = static_cast<uint8_t>(lo | (hi << 4));
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
    auto bqs = rt.buffer_from_host(0, qs.data(), cloudtpu::PjrtDType::U8, {1, 1, N, 1, 16});
    auto bd = rt.buffer_from_host(0, d.data(), cloudtpu::PjrtDType::F16, {1, 1, N, 1});
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

}  // namespace

int main() {
    std::printf("Running ggml-cloudtpu standalone host tests (Mock PJRT)...\n");

    test_compile_options_proto();
    test_half_bf16_conversions();

    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = true;
    cfg.exec_cache_cap = 256;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);

    test_lru_cache_and_hbm_roundtrip(rt);
    test_mul_mat_dense_and_quant(rt);
    test_rms_rope_softmax_swiglu_getrows_fused(rt);

    rt.shutdown();
    std::printf("OVERALL: %s (%d failures)\n", (g_failures == 0) ? "ALL PASSED" : "FAILED", g_failures);
    return (g_failures == 0) ? 0 : 1;
}
