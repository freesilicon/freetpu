#include "pjrt_runtime.hpp"
#include "stablehlo_emitter.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

double compute_nmse(const std::vector<float>& got, const std::vector<float>& ref) {
    double se = 0.0, sref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        se += diff * diff;
        sref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return se / std::max(1e-12, sref);
}

void test_subbyte_s4_u4_packing_and_hbm(cloudtpu::PjrtRuntime& rt) {
    CHECK_TRUE(cloudtpu::pjrt_dtype_bits(cloudtpu::PjrtDType::S4) == 4, "S4 bits == 4");
    CHECK_TRUE(cloudtpu::pjrt_dtype_bits(cloudtpu::PjrtDType::U4) == 4, "U4 bits == 4");
    CHECK_TRUE(cloudtpu::pjrt_dtype_size(cloudtpu::PjrtDType::S4) == 0, "S4 sub-byte dtype_size == 0");
    CHECK_TRUE(cloudtpu::pjrt_dtype_size(cloudtpu::PjrtDType::S8) == 1, "S8 dtype_size == 1");
    CHECK_TRUE(cloudtpu::pjrt_dtype_size(cloudtpu::PjrtDType::F32) == 4, "F32 dtype_size == 4");
    CHECK_TRUE(cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, 0) == 0, "S4 0 elems == 0 B");
    CHECK_TRUE(cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, 1) == 1, "S4 1 elem == 1 B");
    CHECK_TRUE(cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, 2) == 1, "S4 2 elems == 1 B");
    CHECK_TRUE(cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::U4, 3) == 2, "U4 3 elems == 2 B");
    CHECK_TRUE(cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, 512) == 256, "S4 512 elems == 256 B");
    CHECK_TRUE(cloudtpu::pjrt_host_nbytes(cloudtpu::PjrtDType::S4, 512) == 512, "S4 host ABI 1 B/elem");
    CHECK_TRUE(cloudtpu::pjrt_host_nbytes(cloudtpu::PjrtDType::F16, 10) == 20, "F16 host ABI 2 B/elem");

    // Odd-length S4 pack/unpack roundtrip & little-endian nibble verification
    std::vector<int8_t> s4_src = {-8, -7, -1, 0, 1, 7, -5, 3, -2, 6, -4, 2, 5, -3, 4, -6, 7};
    std::vector<uint8_t> s4_packed(cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, s4_src.size()), 0);
    std::vector<int8_t> s4_back(s4_src.size(), 0);
    cloudtpu::pack_s4_from_i8(s4_src.data(), s4_src.size(), s4_packed.data());
    CHECK_TRUE(s4_packed[0] == static_cast<uint8_t>(((-8) & 0x0F) | (((-7) & 0x0F) << 4)),
               "pack_s4 little-endian nibble layout");
    cloudtpu::unpack_s4_to_i8(s4_packed.data(), s4_src.size(), s4_back.data());
    CHECK_TRUE(s4_back == s4_src, "pack/unpack S4 odd-length roundtrip");

    std::vector<uint8_t> u4_src = {0, 15, 7, 8, 1, 14, 9, 2, 13, 3, 12, 4, 11, 5, 10, 6, 15};
    std::vector<uint8_t> u4_packed(cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::U4, u4_src.size()), 0);
    std::vector<uint8_t> u4_back(u4_src.size(), 0);
    cloudtpu::pack_u4_from_u8(u4_src.data(), u4_src.size(), u4_packed.data());
    CHECK_TRUE(u4_packed[0] == static_cast<uint8_t>(0x00U | (0x0FU << 4)),
               "pack_u4 little-endian nibble layout");
    cloudtpu::unpack_u4_to_u8(u4_packed.data(), u4_src.size(), u4_back.data());
    CHECK_TRUE(u4_back == u4_src, "pack/unpack U4 odd-length roundtrip");

    // Host Mock HBM 0.5 B/weight accounting & H2D/D2H roundtrip (1 B/elem host ABI)
    size_t free_before = 0, free_after = 0, total_b = 0;
    CHECK_TRUE(rt.query_memory_stats(0, &free_before, &total_b), "query_memory_stats before S4");
    {
        std::vector<int8_t> s4_host(4096);
        for (size_t i = 0; i < s4_host.size(); ++i) {
            s4_host[i] = static_cast<int8_t>(static_cast<int>(i % 16) - 8);
        }
        auto bs4 = rt.buffer_from_host(0, s4_host.data(), cloudtpu::PjrtDType::S4, {1, 1, 64, 64});
        CHECK_TRUE(bs4->num_elements() == 4096, "S4 num_elements == 4096");
        CHECK_TRUE(bs4->num_bytes() == 2048, "S4 num_bytes == 0.5 B/elem (2048)");
        CHECK_TRUE(rt.query_memory_stats(0, &free_after, &total_b), "query_memory_stats after S4");
        if (rt.platform_name() == "tpu-mock") {
            CHECK_TRUE((free_before - free_after) == 2048, "Mock HBM tracks 0.5 B/weight for S4");
        }

        std::vector<int8_t> s4_roundtrip(4096, 0);
        rt.buffer_to_host(bs4->handle.get(), s4_roundtrip.data(), s4_roundtrip.size());
        CHECK_TRUE(s4_roundtrip == s4_host, "S4 H2D/D2H 1-B/elem roundtrip");

        auto bu4 = rt.buffer_from_host(0, u4_src.data(), cloudtpu::PjrtDType::U4, {1, 1, 1, 17});
        CHECK_TRUE(bu4->num_bytes() == 9, "odd U4 num_bytes == 9");
        std::vector<uint8_t> u4_roundtrip(17, 0);
        rt.buffer_to_host(bu4->handle.get(), u4_roundtrip.data(), u4_roundtrip.size());
        CHECK_TRUE(u4_roundtrip == u4_src, "odd U4 H2D/D2H roundtrip");
    }
    std::printf("  [PASS] Sub-byte S4/U4 packing, 0.5 B/weight HBM, and H2D/D2H roundtrip\n");
}

void test_q4_i4_single_read_matmul_and_bitcast(cloudtpu::PjrtRuntime& rt) {
    // 1) Q4_0_I4: Direct S4 single-read + bitcast U8 + M=1, M=16 (factored), M=40 (in-loop reshape)
    {
        const int64_t N = 16, NB = 4, K = NB * 32;  // K=128
        std::vector<int8_t> qs_s4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint8_t> qs_u8_bc(static_cast<size_t>(NB * 32 * (N / 2)));
        std::vector<uint16_t> d_km(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));

        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t d16 = cloudtpu::f32_to_f16(0.04f * static_cast<float>((n + b) % 7 + 1));
                d_km[static_cast<size_t>(b * N + n)] = d16;
                const float scale = cloudtpu::f16_to_f32(d16);
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 5 + b * 7 + k * 3 + 1) & 0x0F);
                    const int8_t s4 = static_cast<int8_t>(static_cast<int>(u4) - 8);
                    const size_t idx = static_cast<size_t>((b * 32 + k) * N + n);
                    qs_s4[idx] = s4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(s4) * scale;
                }
            }
            for (int64_t k = 0; k < 32; ++k) {
                for (int64_t nb_pair = 0; nb_pair < N / 2; ++nb_pair) {
                    const size_t base_s = static_cast<size_t>((b * 32 + k) * N + nb_pair * 2);
                    const uint8_t lo = static_cast<uint8_t>(qs_s4[base_s]) & 0x0FU;
                    const uint8_t hi = static_cast<uint8_t>(qs_s4[base_s + 1]) & 0x0FU;
                    qs_u8_bc[static_cast<size_t>((b * 32 + k) * (N / 2) + nb_pair)] = static_cast<uint8_t>(lo | (hi << 4));
                }
            }
        }

        auto bqs_s4 = rt.buffer_from_host(0, qs_s4.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
        auto bqs_bc = rt.buffer_from_host(0, qs_u8_bc.data(), cloudtpu::PjrtDType::U8, {1, 1, NB, 32, N / 2});
        auto bd = rt.buffer_from_host(0, d_km.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});

        for (int64_t M : {int64_t(1), int64_t(16), int64_t(40)}) {
            std::vector<float> x(static_cast<size_t>(M * K));
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(static_cast<float>(i) * 0.17f);
            auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

            std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n = 0; n < N; ++n) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += x[static_cast<size_t>(m * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
                    }
                    ref[static_cast<size_t>(m * N + n)] = acc;
                }
            }

            // Direct S4
            {
                cloudtpu::GraphProgramSpec spec;
                spec.bf16_matmul = (M != 40);
                spec.inputs.push_back({1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_0_I4, {1, 1, N, K}});
                spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
                cloudtpu::OpNodeSpec mm;
                mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
                mm.src_ids = {1, 2};
                mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
                spec.ops.push_back(mm);
                spec.output_ids = {3};
                const auto prog = cloudtpu::emit_stablehlo_program(spec);
                CHECK_TRUE(prog.mlir_text.find("i4>") != std::string::npos, "Q4_0_I4 MLIR uses i4 tensor");
                CHECK_TRUE(prog.mlir_text.find("shift_right_logical") == std::string::npos,
                           "Q4_0_I4 MLIR must not nibble-shift");
                auto got = run_emitted(rt, prog, {bqs_s4, bd, bx});
                const double nmse = compute_nmse(got, ref);
                CHECK_TRUE(nmse < 5e-4, "Q4_0_I4 direct S4 MUL_MAT NMSE < 5e-4");
            }

            // U8 -> bitcast_convert -> S4
            if (M == 16 || M == 40) {
                cloudtpu::GraphProgramSpec spec;
                spec.bf16_matmul = true;
                cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::U8, cloudtpu::TensorStorageKind::Q4_0_I4, {1, 1, N, K}};
                w_spec.bitcast_target = cloudtpu::PjrtDType::S4;
                spec.inputs.push_back(w_spec);
                spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
                cloudtpu::OpNodeSpec mm;
                mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
                mm.src_ids = {1, 2};
                mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
                spec.ops.push_back(mm);
                spec.output_ids = {3};
                const auto prog = cloudtpu::emit_stablehlo_program(spec);
                CHECK_TRUE(prog.mlir_text.find("stablehlo.bitcast_convert") != std::string::npos,
                           "Q4_0_I4 bitcast mode emits bitcast_convert");
                auto got = run_emitted(rt, prog, {bqs_bc, bd, bx});
                const double nmse = compute_nmse(got, ref);
                CHECK_TRUE(nmse < 5e-4, "Q4_0_I4 bitcast_convert MUL_MAT NMSE < 5e-4");
            }
        }
    }

    // 2) Q4_K_I4: Unified shifted S4 (m_s4 = m + 8*d) with F32 & F16 scales, plus U4 (0..15) & bitcast
    {
        const int64_t N = 16, NB = 8, K = NB * 32;  // K=256 (1 QK_K block)
        std::vector<int8_t> qs_s4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint8_t> qs_u4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint8_t> qs_u8_bc(static_cast<size_t>(NB * 32 * (N / 2)));
        std::vector<float> d_f32(static_cast<size_t>(NB * N));
        std::vector<float> m_orig_f32(static_cast<size_t>(NB * N));
        std::vector<float> m_s4_f32(static_cast<size_t>(NB * N));
        std::vector<uint16_t> d_f16(static_cast<size_t>(NB * N));
        std::vector<uint16_t> m_s4_f16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));

        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const float d = 0.03f * static_cast<float>((n * 2 + b) % 9 + 1);
                const float m = -0.15f + 0.025f * static_cast<float>((n + b * 3) % 11);
                const size_t bn = static_cast<size_t>(b * N + n);
                d_f32[bn] = d;
                m_orig_f32[bn] = m;
                m_s4_f32[bn] = m + 8.0f * d;
                d_f16[bn] = cloudtpu::f32_to_f16(d);
                m_s4_f16[bn] = cloudtpu::f32_to_f16(m_s4_f32[bn]);
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 7 + b * 3 + k * 5) & 0x0F);
                    const int8_t s4 = static_cast<int8_t>(static_cast<int>(u4) - 8);
                    const size_t idx = static_cast<size_t>((b * 32 + k) * N + n);
                    qs_s4[idx] = s4;
                    qs_u4[idx] = u4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(u4) * d + m;
                }
            }
            for (int64_t k = 0; k < 32; ++k) {
                for (int64_t nb_pair = 0; nb_pair < N / 2; ++nb_pair) {
                    const size_t base_s = static_cast<size_t>((b * 32 + k) * N + nb_pair * 2);
                    const uint8_t lo = static_cast<uint8_t>(qs_s4[base_s]) & 0x0FU;
                    const uint8_t hi = static_cast<uint8_t>(qs_s4[base_s + 1]) & 0x0FU;
                    qs_u8_bc[static_cast<size_t>((b * 32 + k) * (N / 2) + nb_pair)] = static_cast<uint8_t>(lo | (hi << 4));
                }
            }
        }

        auto bqs_s4 = rt.buffer_from_host(0, qs_s4.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
        auto bqs_u4 = rt.buffer_from_host(0, qs_u4.data(), cloudtpu::PjrtDType::U4, {1, 1, NB, 32, N});
        auto bqs_bc = rt.buffer_from_host(0, qs_u8_bc.data(), cloudtpu::PjrtDType::U8, {1, 1, NB, 32, N / 2});
        auto bd_f32 = rt.buffer_from_host(0, d_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
        auto bm_s4_f32 = rt.buffer_from_host(0, m_s4_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
        auto bm_orig_f32 = rt.buffer_from_host(0, m_orig_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
        auto bd_f16 = rt.buffer_from_host(0, d_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});
        auto bm_s4_f16 = rt.buffer_from_host(0, m_s4_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});

        for (int64_t M : {int64_t(1), int64_t(16), int64_t(40)}) {
            std::vector<float> x(static_cast<size_t>(M * K));
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.11f) + 0.25f;
            auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});
            std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n = 0; n < N; ++n) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += x[static_cast<size_t>(m * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
                    }
                    ref[static_cast<size_t>(m * N + n)] = acc;
                }
            }

            // Unified S4 + F32 scales
            {
                cloudtpu::GraphProgramSpec spec;
                spec.bf16_matmul = true;
                cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
                w_spec.scale_dtype = cloudtpu::PjrtDType::F32;
                spec.inputs.push_back(w_spec);
                spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
                cloudtpu::OpNodeSpec mm;
                mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
                mm.src_ids = {1, 2};
                mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
                spec.ops.push_back(mm);
                spec.output_ids = {3};
                auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec), {bqs_s4, bd_f32, bm_s4_f32, bx});
                CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q4_K_I4 unified S4 F32-scales NMSE < 5e-4");
            }

            // Unified S4 + F16 scales (M=16)
            if (M == 16) {
                cloudtpu::GraphProgramSpec spec;
                spec.bf16_matmul = true;
                cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
                w_spec.scale_dtype = cloudtpu::PjrtDType::F16;
                spec.inputs.push_back(w_spec);
                spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
                cloudtpu::OpNodeSpec mm;
                mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
                mm.src_ids = {1, 2};
                mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
                spec.ops.push_back(mm);
                spec.output_ids = {3};
                auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec), {bqs_s4, bd_f16, bm_s4_f16, bx});
                CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q4_K_I4 unified S4 F16-scales NMSE < 5e-4");

                // Optional U4 (ui4) + orig m
                cloudtpu::GraphProgramSpec u4_spec = spec;
                u4_spec.inputs[0].dtype = cloudtpu::PjrtDType::U4;
                u4_spec.inputs[0].scale_dtype = cloudtpu::PjrtDType::F32;
                auto got_u4 = run_emitted(rt, cloudtpu::emit_stablehlo_program(u4_spec), {bqs_u4, bd_f32, bm_orig_f32, bx});
                CHECK_TRUE(compute_nmse(got_u4, ref) < 5e-4, "Q4_K_I4 U4 NMSE < 5e-4");

                // Bitcast U8 -> S4
                cloudtpu::GraphProgramSpec bc_spec = spec;
                bc_spec.inputs[0].dtype = cloudtpu::PjrtDType::U8;
                bc_spec.inputs[0].bitcast_target = cloudtpu::PjrtDType::S4;
                bc_spec.inputs[0].scale_dtype = cloudtpu::PjrtDType::F32;
                const auto bc_prog = cloudtpu::emit_stablehlo_program(bc_spec);
                CHECK_TRUE(bc_prog.mlir_text.find("stablehlo.bitcast_convert") != std::string::npos,
                           "Q4_K_I4 bitcast emits bitcast_convert");
                auto got_bc = run_emitted(rt, bc_prog, {bqs_bc, bd_f32, bm_s4_f32, bx});
                CHECK_TRUE(compute_nmse(got_bc, ref) < 5e-4, "Q4_K_I4 bitcast NMSE < 5e-4");
            }
        }
    }
    std::printf("  [PASS] Q4_0_I4 & Q4_K_I4 single-read S4/U4/bitcast MUL_MAT (M=1, M=16, M=40)\n");
}

void test_q4_i4_get_rows_coexistence(cloudtpu::PjrtRuntime& rt) {
    // Shared Q4_K_I4 table used simultaneously in GET_ROWS (incl. odd+even indices) and MUL_MAT
    const int64_t N = 16, NB = 8, K = NB * 32;
    std::vector<int8_t> qs_s4(static_cast<size_t>(NB * 32 * N));
    std::vector<uint8_t> qs_u8_bc(static_cast<size_t>(NB * 32 * (N / 2)));
    std::vector<float> d_f32(static_cast<size_t>(NB * N));
    std::vector<float> m_s4_f32(static_cast<size_t>(NB * N));
    std::vector<float> w_ref(static_cast<size_t>(N * K));
    for (int64_t b = 0; b < NB; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            const float d = 0.04f * static_cast<float>((n + b * 3) % 7 + 1);
            const float m = -0.1f + 0.02f * static_cast<float>(n % 5);
            d_f32[static_cast<size_t>(b * N + n)] = d;
            m_s4_f32[static_cast<size_t>(b * N + n)] = m + 8.0f * d;
            for (int64_t k = 0; k < 32; ++k) {
                const uint8_t u4 = static_cast<uint8_t>((n * 3 + b * 11 + k * 7) & 0x0F);
                const int8_t s4 = static_cast<int8_t>(static_cast<int>(u4) - 8);
                qs_s4[static_cast<size_t>((b * 32 + k) * N + n)] = s4;
                w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(u4) * d + m;
            }
        }
        for (int64_t k = 0; k < 32; ++k) {
            for (int64_t p = 0; p < N / 2; ++p) {
                const size_t s = static_cast<size_t>((b * 32 + k) * N + p * 2);
                qs_u8_bc[static_cast<size_t>((b * 32 + k) * (N / 2) + p)] = static_cast<uint8_t>(
                    (static_cast<uint8_t>(qs_s4[s]) & 0x0FU) |
                    ((static_cast<uint8_t>(qs_s4[s + 1]) & 0x0FU) << 4));
            }
        }
    }
    auto bqs_s4 = rt.buffer_from_host(0, qs_s4.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
    auto bqs_bc = rt.buffer_from_host(0, qs_u8_bc.data(), cloudtpu::PjrtDType::U8, {1, 1, NB, 32, N / 2});
    auto bd = rt.buffer_from_host(0, d_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
    auto bm = rt.buffer_from_host(0, m_s4_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});

    const std::vector<int32_t> row_idx = {3, 0, 15, 2};
    auto bidx = rt.buffer_from_host(0, row_idx.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, 4});

    // Build combined graph: rows = GET_ROWS(tbl, idx), y = MUL_MAT(tbl, rows)
    for (bool use_bitcast : {false, true}) {
        cloudtpu::GraphProgramSpec spec;
        spec.bf16_matmul = true;
        cloudtpu::TensorSpec tbl_spec{
            1,
            use_bitcast ? cloudtpu::PjrtDType::U8 : cloudtpu::PjrtDType::S4,
            cloudtpu::TensorStorageKind::Q4_K_I4,
            {1, 1, N, K}};
        tbl_spec.scale_dtype = cloudtpu::PjrtDType::F32;
        tbl_spec.bitcast_target = cloudtpu::PjrtDType::S4;
        spec.inputs.push_back(tbl_spec);
        spec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});

        cloudtpu::OpNodeSpec gr;
        gr.kind = cloudtpu::CloudOpKind::GET_ROWS;
        gr.src_ids = {1, 2};
        gr.dst = {10, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 4, K}};

        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {1, 10};
        mm.dst = {11, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 4, N}};

        spec.ops = {gr, mm};
        spec.output_ids = {10, 11};
        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        const auto* exec = rt.compile_or_get(prog.cache_key, prog.mlir_text);
        std::vector<PJRT_Buffer*> raw_in = {
            (use_bitcast ? bqs_bc : bqs_s4)->handle.get(),
            bd->handle.get(),
            bm->handle.get(),
            bidx->handle.get()};
        std::vector<cloudtpu::PjrtBufferPtr> outs;
        rt.execute(exec, 0, raw_in.data(), raw_in.size(), outs,
                   {cloudtpu::PjrtDType::F32, cloudtpu::PjrtDType::F32},
                   {{1, 1, 4, K}, {1, 1, 4, N}}, true);

        std::vector<float> got_rows(static_cast<size_t>(4 * K));
        std::vector<float> got_mm(static_cast<size_t>(4 * N));
        rt.buffer_to_host(outs[0]->handle.get(), got_rows.data(), got_rows.size() * sizeof(float));
        rt.buffer_to_host(outs[1]->handle.get(), got_mm.data(), got_mm.size() * sizeof(float));

        for (size_t r = 0; r < 4; ++r) {
            const int64_t row = row_idx[r];
            for (int64_t k = 0; k < K; ++k) {
                CHECK_NEAR(got_rows[r * static_cast<size_t>(K) + static_cast<size_t>(k)],
                           w_ref[static_cast<size_t>(row * K + k)], 1e-4f, "GET_ROWS on Q4_K_I4 table");
            }
        }
        std::vector<float> ref_mm(static_cast<size_t>(4 * N), 0.0f);
        for (size_t r = 0; r < 4; ++r) {
            const int64_t row = row_idx[r];
            for (int64_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    acc += w_ref[static_cast<size_t>(row * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
                }
                ref_mm[r * static_cast<size_t>(N) + static_cast<size_t>(n)] = acc;
            }
        }
        CHECK_TRUE(compute_nmse(got_mm, ref_mm) < 5e-4, "GET_ROWS + MUL_MAT Q4_K_I4 coexistence NMSE < 5e-4");
    }
    std::printf("  [PASS] GET_ROWS + MUL_MAT coexistence on Q4_K_I4 (direct S4 & U8 bitcast)\n");
}

void test_q4_i4_spmd_sharding() {
    constexpr int P = 4;
    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = true;
    cfg.mock_devices = P;
    cfg.exec_cache_cap = 64;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);
    const auto copts = rt.make_spmd_compile_options(P, 0, false, true);

    // 1) SHARD_N across P=4 chips (N=32 -> N_local=8 even, K=256) + 0.5 B/weight 1/P HBM check
    {
        const int64_t N = 32, N_local = N / P, NB = 8, K = NB * 32, M = 8;
        std::vector<int8_t> qs_km(static_cast<size_t>(NB * 32 * N));
        std::vector<float> d_km(static_cast<size_t>(NB * N));
        std::vector<float> m_s4_km(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const float d = 0.03f * static_cast<float>((n + b) % 7 + 1);
                const float m = -0.1f + 0.02f * static_cast<float>(n % 9);
                d_km[static_cast<size_t>(b * N + n)] = d;
                m_s4_km[static_cast<size_t>(b * N + n)] = m + 8.0f * d;
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 5 + b * 3 + k) & 0x0F);
                    const int8_t s4 = static_cast<int8_t>(static_cast<int>(u4) - 8);
                    qs_km[static_cast<size_t>((b * 32 + k) * N + n)] = s4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(u4) * d + m;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.19f);
        std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    acc += x[static_cast<size_t>(m * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
                }
                ref[static_cast<size_t>(m * N + n)] = acc;
            }
        }

        cloudtpu::GraphProgramSpec spec;
        spec.spmd_chips = P;
        spec.bf16_matmul = true;
        cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}, cloudtpu::TensorShardingKind::SHARD_N};
        w_spec.scale_dtype = cloudtpu::PjrtDType::F32;
        spec.inputs.push_back(w_spec);
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}, cloudtpu::TensorShardingKind::REPLICATED});
        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {1, 2};
        mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
        spec.ops.push_back(mm);
        spec.output_ids = {3};
        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.all_gather") != std::string::npos,
                   "SHARD_N Q4_K_I4 emits all_gather");

        size_t free_before = 0, free_after = 0, total_b = 0;
        CHECK_TRUE(rt.query_memory_stats(0, &free_before, &total_b), "SPMD S4 mem before");

        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> per_dev(static_cast<size_t>(P));
        for (int p = 0; p < P; ++p) {
            std::vector<int8_t> qs_p(static_cast<size_t>(NB * 32 * N_local));
            std::vector<float> d_p(static_cast<size_t>(NB * N_local));
            std::vector<float> m_p(static_cast<size_t>(NB * N_local));
            for (int64_t b = 0; b < NB; ++b) {
                std::memcpy(d_p.data() + static_cast<size_t>(b * N_local),
                            d_km.data() + static_cast<size_t>(b * N + p * N_local),
                            static_cast<size_t>(N_local) * sizeof(float));
                std::memcpy(m_p.data() + static_cast<size_t>(b * N_local),
                            m_s4_km.data() + static_cast<size_t>(b * N + p * N_local),
                            static_cast<size_t>(N_local) * sizeof(float));
                for (int64_t k = 0; k < 32; ++k) {
                    std::memcpy(qs_p.data() + static_cast<size_t>((b * 32 + k) * N_local),
                                qs_km.data() + static_cast<size_t>((b * 32 + k) * N + p * N_local),
                                static_cast<size_t>(N_local));
                }
            }
            per_dev[static_cast<size_t>(p)] = {
                rt.buffer_from_host(p, qs_p.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N_local}),
                rt.buffer_from_host(p, d_p.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N_local}),
                rt.buffer_from_host(p, m_p.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N_local}),
                rt.buffer_from_host(p, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K})};
        }

        CHECK_TRUE(rt.query_memory_stats(0, &free_after, &total_b), "SPMD S4 mem after");
        const size_t expected_chip0 =
            cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, static_cast<size_t>(NB * 32 * N_local)) +
            static_cast<size_t>(NB * N_local) * 2 * sizeof(float) +
            x.size() * sizeof(float);
        CHECK_TRUE((free_before - free_after) == expected_chip0, "SPMD Q4_K_I4 chip0 uses 0.5B/weight / P");

        const auto* exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, copts);
        std::vector<std::vector<PJRT_Buffer*>> raw_in(static_cast<size_t>(P));
        std::vector<PJRT_Buffer* const*> in_ptrs(static_cast<size_t>(P));
        for (int p = 0; p < P; ++p) {
            for (const auto& b : per_dev[static_cast<size_t>(p)]) raw_in[static_cast<size_t>(p)].push_back(b->handle.get());
            in_ptrs[static_cast<size_t>(p)] = raw_in[static_cast<size_t>(p)].data();
        }
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
        rt.execute_spmd(exec, P, in_ptrs.data(), prog.input_bindings.size(), outs,
                        {cloudtpu::PjrtDType::F32}, {{1, 1, M, N}}, true);
        for (int p = 0; p < P; ++p) {
            std::vector<float> got(ref.size());
            rt.buffer_to_host(outs[static_cast<size_t>(p)][0]->handle.get(), got.data(), got.size() * sizeof(float));
            CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "SPMD SHARD_N Q4_K_I4 P=4 NMSE < 5e-4");
        }
    }

    // 2) SHARD_K across P=4 chips (N=16, K=512 -> K_local=128 (NB_local=4), M=8 -> all_reduce)
    {
        const int64_t N = 16, NB = 16, NB_local = NB / P, K = NB * 32, K_local = NB_local * 32, M = 8;
        std::vector<int8_t> qs_km(static_cast<size_t>(NB * 32 * N));
        std::vector<float> d_km(static_cast<size_t>(NB * N));
        std::vector<float> m_s4_km(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const float d = 0.025f * static_cast<float>((n + b * 2) % 7 + 1);
                const float m = -0.08f + 0.015f * static_cast<float>(n % 7);
                d_km[static_cast<size_t>(b * N + n)] = d;
                m_s4_km[static_cast<size_t>(b * N + n)] = m + 8.0f * d;
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 3 + b * 5 + k * 2) & 0x0F);
                    const int8_t s4 = static_cast<int8_t>(static_cast<int>(u4) - 8);
                    qs_km[static_cast<size_t>((b * 32 + k) * N + n)] = s4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(u4) * d + m;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(static_cast<float>(i) * 0.13f);
        std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    acc += x[static_cast<size_t>(m * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
                }
                ref[static_cast<size_t>(m * N + n)] = acc;
            }
        }

        cloudtpu::GraphProgramSpec spec;
        spec.spmd_chips = P;
        spec.bf16_matmul = true;
        cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}, cloudtpu::TensorShardingKind::SHARD_K};
        w_spec.scale_dtype = cloudtpu::PjrtDType::F32;
        spec.inputs.push_back(w_spec);
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}, cloudtpu::TensorShardingKind::REPLICATED});
        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {1, 2};
        mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
        spec.ops.push_back(mm);
        spec.output_ids = {3};
        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.all_reduce") != std::string::npos,
                   "SHARD_K Q4_K_I4 emits all_reduce");

        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> per_dev(static_cast<size_t>(P));
        for (int p = 0; p < P; ++p) {
            const size_t qs_off = static_cast<size_t>(p * NB_local * 32 * N);
            const size_t d_off = static_cast<size_t>(p * NB_local * N);
            per_dev[static_cast<size_t>(p)] = {
                rt.buffer_from_host(p, qs_km.data() + qs_off, cloudtpu::PjrtDType::S4, {1, 1, NB_local, 32, N}),
                rt.buffer_from_host(p, d_km.data() + d_off, cloudtpu::PjrtDType::F32, {1, 1, NB_local, N}),
                rt.buffer_from_host(p, m_s4_km.data() + d_off, cloudtpu::PjrtDType::F32, {1, 1, NB_local, N}),
                rt.buffer_from_host(p, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K})};
            (void) K_local;
        }
        const auto* exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, copts);
        std::vector<std::vector<PJRT_Buffer*>> raw_in(static_cast<size_t>(P));
        std::vector<PJRT_Buffer* const*> in_ptrs(static_cast<size_t>(P));
        for (int p = 0; p < P; ++p) {
            for (const auto& b : per_dev[static_cast<size_t>(p)]) raw_in[static_cast<size_t>(p)].push_back(b->handle.get());
            in_ptrs[static_cast<size_t>(p)] = raw_in[static_cast<size_t>(p)].data();
        }
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
        rt.execute_spmd(exec, P, in_ptrs.data(), prog.input_bindings.size(), outs,
                        {cloudtpu::PjrtDType::F32}, {{1, 1, M, N}}, true);
        for (int p = 0; p < P; ++p) {
            std::vector<float> got(ref.size());
            rt.buffer_to_host(outs[static_cast<size_t>(p)][0]->handle.get(), got.data(), got.size() * sizeof(float));
            CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "SPMD SHARD_K Q4_K_I4 P=4 NMSE < 5e-4");
        }
    }

    rt.shutdown();
    std::printf("  [PASS] SPMD SHARD_N & SHARD_K Q4_K_I4 (P=4, 0.5 B/weight HBM parity)\n");
}

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
    test_subbyte_s4_u4_packing_and_hbm(rt);
    test_mul_mat_dense_and_quant(rt);
    test_q4_i4_single_read_matmul_and_bitcast(rt);
    test_q4_i4_get_rows_coexistence(rt);
    test_rms_rope_softmax_swiglu_getrows_fused(rt);
    test_silicon_regressions(rt);
    test_collectives_p1_live_compatible(rt);
    test_spmd_multichip_mesh(4);  // tpu7x-4t (4x 192 GiB HBM3e)
    test_spmd_multichip_mesh(8);  // v6e-8 (8x 32 GiB HBM)
    test_q4_i4_spmd_sharding();

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
