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

void test_bf16_scales_zero_f32_dequant_and_pad_m(cloudtpu::PjrtRuntime& rt) {
    // 1) Q6_K_SPLIT with true BF16 scales: zero-F32 dequant + M<8 pad8 (M=1, M=3) + M=16 + mm_pad_m=1
    {
        const int64_t N = 16, NB = 8, K = NB * 16;  // K=128 (NB16 = 8)
        std::vector<int8_t> qs_km(static_cast<size_t>(NB * 16 * N));
        std::vector<uint16_t> d_bf16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));

        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const float raw_d = 0.02f * static_cast<float>((n + b * 2) % 9 + 1) * ((n & 1) ? -1.0f : 1.0f);
                const uint16_t db = cloudtpu::f32_to_bf16(raw_d);
                d_bf16[static_cast<size_t>(b * N + n)] = db;
                const float d = cloudtpu::bf16_to_f32(db);
                for (int64_t k = 0; k < 16; ++k) {
                    const int8_t q = static_cast<int8_t>((static_cast<int>(n * 5 + b * 3 + k * 7) % 63) - 31);
                    qs_km[static_cast<size_t>((b * 16 + k) * N + n)] = q;
                    w_ref[static_cast<size_t>(n * K + b * 16 + k)] = static_cast<float>(q) * d;
                }
            }
        }

        auto bqs = rt.buffer_from_host(0, qs_km.data(), cloudtpu::PjrtDType::S8, {1, 1, NB, 16, N});
        auto bd = rt.buffer_from_host(0, d_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});

        for (int64_t M : {int64_t(1), int64_t(3), int64_t(16)}) {
            std::vector<float> x(static_cast<size_t>(M * K));
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.13f);
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

            cloudtpu::GraphProgramSpec spec;
            spec.bf16_matmul = true;
            spec.mm_pad_m = 8;
            cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S8, cloudtpu::TensorStorageKind::Q6_K_SPLIT, {1, 1, N, K}};
            w_spec.scale_dtype = cloudtpu::PjrtDType::BF16;
            spec.inputs.push_back(w_spec);
            spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
            cloudtpu::OpNodeSpec mm;
            mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
            mm.src_ids = {1, 2};
            mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
            spec.ops.push_back(mm);
            spec.output_ids = {3};

            const auto prog = cloudtpu::emit_stablehlo_program(spec);
            // Zero-F32-temp BF16 dequant: qs converts directly to tensor<1x1x8x16x16xbf16> with zero xf32 intermediate
            CHECK_TRUE(prog.mlir_text.find("-> tensor<1x1x8x16x16xbf16>") != std::string::npos,
                       "Q6_K BF16 scales dequantize qs directly to BF16");
            CHECK_TRUE(prog.mlir_text.find("tensor<1x1x8x16x16xf32>") == std::string::npos,
                       "Q6_K BF16 scales must emit zero F32 intermediate");

            if (M < 8) {
                CHECK_TRUE(prog.mlir_text.find("stablehlo.concatenate") != std::string::npos,
                           "Q6_K M<8 pad8 must emit concatenate");
                CHECK_TRUE(prog.mlir_text.find("stablehlo.slice") != std::string::npos,
                           "Q6_K M<8 pad8 must emit slice back");
                CHECK_TRUE(prog.mlir_text.find("tensor<1x1x8x" + std::to_string(N) + "x") != std::string::npos,
                           "Q6_K M<8 pad8 dot output must have M=8");
            } else {
                CHECK_TRUE(prog.mlir_text.find("stablehlo.concatenate") == std::string::npos,
                           "Q6_K M>=8 must not pad M");
            }

            auto got = run_emitted(rt, prog, {bqs, bd, bx});
            CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q6_K BF16 scales + pad8 NMSE < 5e-4");

            // Verify mm_pad_m=1 disables padding and produces distinct cache key
            cloudtpu::GraphProgramSpec nopad = spec;
            nopad.mm_pad_m = 1;
            const auto np_prog = cloudtpu::emit_stablehlo_program(nopad);
            CHECK_TRUE(np_prog.mlir_text.find("stablehlo.concatenate") == std::string::npos,
                       "Q6_K mm_pad_m=1 disables concatenate pad");
            CHECK_TRUE(np_prog.cache_key != prog.cache_key,
                       "StableHLO cache_key tracks non-default mm_pad_m");
            CHECK_TRUE(np_prog.cache_key.find("_pm1") != std::string::npos,
                       "StableHLO cache_key embeds _pm1");
            auto got_np = run_emitted(rt, np_prog, {bqs, bd, bx});
            CHECK_TRUE(compute_nmse(got_np, ref) < 5e-4, "Q6_K mm_pad_m=1 NMSE < 5e-4");
        }
    }

    // 2) Q4_K_I4 with true BF16 d/m scales: factored M<8 pad8 (M=1, M=3) & M=40 zero-F32-temp BF16 prefill
    {
        const int64_t N = 16, NB = 8, K = NB * 32;  // K=256
        std::vector<int8_t> qs_s4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint16_t> d_bf16(static_cast<size_t>(NB * N));
        std::vector<uint16_t> m_s4_bf16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));

        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t db = cloudtpu::f32_to_bf16(0.03f * static_cast<float>((n * 2 + b) % 9 + 1));
                const float d = cloudtpu::bf16_to_f32(db);
                const float m_raw = -0.15f + 0.025f * static_cast<float>((n + b * 3) % 11);
                const uint16_t mb = cloudtpu::f32_to_bf16(m_raw + 8.0f * d);
                const float m_s4 = cloudtpu::bf16_to_f32(mb);
                d_bf16[static_cast<size_t>(b * N + n)] = db;
                m_s4_bf16[static_cast<size_t>(b * N + n)] = mb;
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 7 + b * 3 + k * 5) & 0x0F);
                    const int8_t s4 = static_cast<int8_t>(static_cast<int>(u4) - 8);
                    qs_s4[static_cast<size_t>((b * 32 + k) * N + n)] = s4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(s4) * d + m_s4;
                }
            }
        }

        auto bqs = rt.buffer_from_host(0, qs_s4.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
        auto bd = rt.buffer_from_host(0, d_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
        auto bm = rt.buffer_from_host(0, m_s4_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});

        for (int64_t M : {int64_t(1), int64_t(3), int64_t(40)}) {
            std::vector<float> x(static_cast<size_t>(M * K));
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(static_cast<float>(i) * 0.11f) + 0.1f;
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

            cloudtpu::GraphProgramSpec spec;
            spec.bf16_matmul = true;
            spec.mm_pad_m = 8;
            cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
            w_spec.scale_dtype = cloudtpu::PjrtDType::BF16;
            spec.inputs.push_back(w_spec);
            spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
            cloudtpu::OpNodeSpec mm;
            mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
            mm.src_ids = {1, 2};
            mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
            spec.ops.push_back(mm);
            spec.output_ids = {3};

            const auto prog = cloudtpu::emit_stablehlo_program(spec);
            if (M < 8) {
                CHECK_TRUE(prog.mlir_text.find("stablehlo.concatenate") != std::string::npos,
                           "Q4_K_I4 M<8 pad8 emits concatenate");
                CHECK_TRUE(prog.mlir_text.find("stablehlo.slice") != std::string::npos,
                           "Q4_K_I4 M<8 pad8 emits slice back");
                CHECK_TRUE(prog.mlir_text.find("tensor<1x1x8x8x16xf32>") != std::string::npos,
                           "Q4_K_I4 M<8 pad8 factored dot_b has M=8");
            } else {
                // M=40 in-loop dequant with BF16 scales uses zero-F32-temp fast_bf16 path
                CHECK_TRUE(prog.mlir_text.find("tensor<1x1x8x32x16xf32>") == std::string::npos,
                           "Q4_K_I4 BF16 scales prefill must emit zero F32 dequant temp");
                CHECK_TRUE(prog.mlir_text.find("-> tensor<1x1x8x32x16xbf16>") != std::string::npos,
                           "Q4_K_I4 BF16 scales prefill converts i4 directly to BF16");
            }

            auto got = run_emitted(rt, prog, {bqs, bd, bm, bx});
            CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q4_K_I4 BF16 scales + pad8 NMSE < 5e-4");
        }
    }

    // 3) Q4_0_I4 factored M=3 pad8
    {
        const int64_t N = 16, NB = 4, K = NB * 32, M = 3;
        std::vector<int8_t> qs_s4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint16_t> d_f16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t dh = cloudtpu::f32_to_f16(0.05f * static_cast<float>((n + b) % 7 + 1));
                d_f16[static_cast<size_t>(b * N + n)] = dh;
                const float d = cloudtpu::f16_to_f32(dh);
                for (int64_t k = 0; k < 32; ++k) {
                    const int8_t s4 = static_cast<int8_t>(static_cast<int>((n * 3 + b * 5 + k) & 0x0F) - 8);
                    qs_s4[static_cast<size_t>((b * 32 + k) * N + n)] = s4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(s4) * d;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.21f);
        std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; ++k) acc += x[static_cast<size_t>(m * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
                ref[static_cast<size_t>(m * N + n)] = acc;
            }
        }
        auto bqs = rt.buffer_from_host(0, qs_s4.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
        auto bd = rt.buffer_from_host(0, d_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});
        cloudtpu::GraphProgramSpec spec;
        spec.bf16_matmul = true;
        spec.mm_pad_m = 8;
        spec.inputs.push_back({1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_0_I4, {1, 1, N, K}});
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {1, 2};
        mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
        spec.ops.push_back(mm);
        spec.output_ids = {3};
        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.concatenate") != std::string::npos, "Q4_0_I4 M=3 pad8 emits concatenate");
        CHECK_TRUE(prog.mlir_text.find("stablehlo.slice") != std::string::npos, "Q4_0_I4 M=3 pad8 emits slice");
        auto got = run_emitted(rt, prog, {bqs, bd, bx});
        CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q4_0_I4 M=3 pad8 NMSE < 5e-4");
    }

    // 4) SPMD P=4 Q6_K_SPLIT with BF16 scales + M=1 pad8 (SHARD_N & SHARD_K)
    {
        constexpr int P = 4;
        cloudtpu::RuntimeConfig scfg;
        scfg.mock_mode = true;
        scfg.mock_devices = P;
        scfg.exec_cache_cap = 32;
        cloudtpu::PjrtRuntime srt;
        srt.init(scfg);
        const auto copts = srt.make_spmd_compile_options(P, 0, false, true);

        const int64_t N = 32, NB = 16, K = NB * 16, M = 1;  // K=256
        std::vector<int8_t> qs_km(static_cast<size_t>(NB * 16 * N));
        std::vector<uint16_t> d_bf16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t db = cloudtpu::f32_to_bf16(0.02f * static_cast<float>((n + b) % 7 + 1));
                d_bf16[static_cast<size_t>(b * N + n)] = db;
                const float d = cloudtpu::bf16_to_f32(db);
                for (int64_t k = 0; k < 16; ++k) {
                    const int8_t q = static_cast<int8_t>((static_cast<int>(n * 3 + b * 7 + k * 5) % 63) - 31);
                    qs_km[static_cast<size_t>((b * 16 + k) * N + n)] = q;
                    w_ref[static_cast<size_t>(n * K + b * 16 + k)] = static_cast<float>(q) * d;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(static_cast<float>(i) * 0.17f);
        std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += x[static_cast<size_t>(k)] * w_ref[static_cast<size_t>(n * K + k)];
            ref[static_cast<size_t>(n)] = acc;
        }

        // SHARD_N
        {
            const int64_t N_local = N / P;
            cloudtpu::GraphProgramSpec spec;
            spec.spmd_chips = P;
            spec.bf16_matmul = true;
            spec.mm_pad_m = 8;
            cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S8, cloudtpu::TensorStorageKind::Q6_K_SPLIT, {1, 1, N, K}, cloudtpu::TensorShardingKind::SHARD_N};
            w_spec.scale_dtype = cloudtpu::PjrtDType::BF16;
            spec.inputs.push_back(w_spec);
            spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}, cloudtpu::TensorShardingKind::REPLICATED});
            cloudtpu::OpNodeSpec mm;
            mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
            mm.src_ids = {1, 2};
            mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
            spec.ops.push_back(mm);
            spec.output_ids = {3};
            const auto prog = cloudtpu::emit_stablehlo_program(spec);
            CHECK_TRUE(prog.mlir_text.find("stablehlo.all_gather") != std::string::npos, "SPMD Q6_K SHARD_N emits all_gather");
            std::vector<std::vector<cloudtpu::PjrtBufferPtr>> per_dev(static_cast<size_t>(P));
            for (int p = 0; p < P; ++p) {
                std::vector<int8_t> q_p(static_cast<size_t>(NB * 16 * N_local));
                std::vector<uint16_t> d_p(static_cast<size_t>(NB * N_local));
                for (int64_t b = 0; b < NB; ++b) {
                    std::memcpy(d_p.data() + static_cast<size_t>(b * N_local),
                                d_bf16.data() + static_cast<size_t>(b * N + p * N_local),
                                static_cast<size_t>(N_local) * sizeof(uint16_t));
                    for (int64_t k = 0; k < 16; ++k) {
                        std::memcpy(q_p.data() + static_cast<size_t>((b * 16 + k) * N_local),
                                    qs_km.data() + static_cast<size_t>((b * 16 + k) * N + p * N_local),
                                    static_cast<size_t>(N_local));
                    }
                }
                per_dev[static_cast<size_t>(p)] = {
                    srt.buffer_from_host(p, q_p.data(), cloudtpu::PjrtDType::S8, {1, 1, NB, 16, N_local}),
                    srt.buffer_from_host(p, d_p.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N_local}),
                    srt.buffer_from_host(p, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K})};
            }
            const auto* exec = srt.compile_or_get(prog.cache_key, prog.mlir_text, copts);
            std::vector<std::vector<PJRT_Buffer*>> raw_in(static_cast<size_t>(P));
            std::vector<PJRT_Buffer* const*> in_ptrs(static_cast<size_t>(P));
            for (int p = 0; p < P; ++p) {
                for (const auto& b : per_dev[static_cast<size_t>(p)]) raw_in[static_cast<size_t>(p)].push_back(b->handle.get());
                in_ptrs[static_cast<size_t>(p)] = raw_in[static_cast<size_t>(p)].data();
            }
            std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
            srt.execute_spmd(exec, P, in_ptrs.data(), prog.input_bindings.size(), outs, {cloudtpu::PjrtDType::F32}, {{1, 1, M, N}}, true);
            for (int p = 0; p < P; ++p) {
                std::vector<float> got(ref.size());
                srt.buffer_to_host(outs[static_cast<size_t>(p)][0]->handle.get(), got.data(), got.size() * sizeof(float));
                CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "SPMD SHARD_N Q6_K BF16 M=1 pad8 NMSE < 5e-4");
            }
        }

        // SHARD_K
        {
            const int64_t NB_local = NB / P;
            cloudtpu::GraphProgramSpec spec;
            spec.spmd_chips = P;
            spec.bf16_matmul = true;
            spec.mm_pad_m = 8;
            cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S8, cloudtpu::TensorStorageKind::Q6_K_SPLIT, {1, 1, N, K}, cloudtpu::TensorShardingKind::SHARD_K};
            w_spec.scale_dtype = cloudtpu::PjrtDType::BF16;
            spec.inputs.push_back(w_spec);
            spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}, cloudtpu::TensorShardingKind::REPLICATED});
            cloudtpu::OpNodeSpec mm;
            mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
            mm.src_ids = {1, 2};
            mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
            spec.ops.push_back(mm);
            spec.output_ids = {3};
            const auto prog = cloudtpu::emit_stablehlo_program(spec);
            CHECK_TRUE(prog.mlir_text.find("stablehlo.all_reduce") != std::string::npos, "SPMD Q6_K SHARD_K emits all_reduce");
            std::vector<std::vector<cloudtpu::PjrtBufferPtr>> per_dev(static_cast<size_t>(P));
            for (int p = 0; p < P; ++p) {
                per_dev[static_cast<size_t>(p)] = {
                    srt.buffer_from_host(p, qs_km.data() + static_cast<size_t>(p * NB_local * 16 * N), cloudtpu::PjrtDType::S8, {1, 1, NB_local, 16, N}),
                    srt.buffer_from_host(p, d_bf16.data() + static_cast<size_t>(p * NB_local * N), cloudtpu::PjrtDType::BF16, {1, 1, NB_local, N}),
                    srt.buffer_from_host(p, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K})};
            }
            const auto* exec = srt.compile_or_get(prog.cache_key, prog.mlir_text, copts);
            std::vector<std::vector<PJRT_Buffer*>> raw_in(static_cast<size_t>(P));
            std::vector<PJRT_Buffer* const*> in_ptrs(static_cast<size_t>(P));
            for (int p = 0; p < P; ++p) {
                for (const auto& b : per_dev[static_cast<size_t>(p)]) raw_in[static_cast<size_t>(p)].push_back(b->handle.get());
                in_ptrs[static_cast<size_t>(p)] = raw_in[static_cast<size_t>(p)].data();
            }
            std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
            srt.execute_spmd(exec, P, in_ptrs.data(), prog.input_bindings.size(), outs, {cloudtpu::PjrtDType::F32}, {{1, 1, M, N}}, true);
            for (int p = 0; p < P; ++p) {
                std::vector<float> got(ref.size());
                srt.buffer_to_host(outs[static_cast<size_t>(p)][0]->handle.get(), got.data(), got.size() * sizeof(float));
                CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "SPMD SHARD_K Q6_K BF16 M=1 pad8 NMSE < 5e-4");
            }
        }
        srt.shutdown();
    }

    std::printf("  [PASS] BF16 Q4_K/Q6_K scales (zero-F32 dequant + M<8 pad8 + SPMD P=4)\n");
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

struct CbQuantWeights {
    cloudtpu::PjrtBufferPtr bqs;
    cloudtpu::PjrtBufferPtr bd;
    cloudtpu::PjrtBufferPtr bm;
};

CbQuantWeights make_cb_q4k_weights(cloudtpu::PjrtRuntime& rt, int64_t N, int64_t K, float phase) {
    const int64_t NB = K / 32;
    std::vector<int8_t> qs(static_cast<size_t>(NB * 32 * N));
    std::vector<uint16_t> d_bf16(static_cast<size_t>(NB * N));
    std::vector<uint16_t> m_bf16(static_cast<size_t>(NB * N));
    for (int64_t b = 0; b < NB; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            const float raw_d = 0.025f * (1.0f + 0.35f * std::sin(static_cast<float>(n * 3 + b) + phase));
            const uint16_t db = cloudtpu::f32_to_bf16(raw_d);
            const float d = cloudtpu::bf16_to_f32(db);
            const float m_raw = 0.04f * std::cos(static_cast<float>(n + b * 5) + phase);
            const uint16_t mb = cloudtpu::f32_to_bf16(m_raw + 8.0f * d);
            d_bf16[static_cast<size_t>(b * N + n)] = db;
            m_bf16[static_cast<size_t>(b * N + n)] = mb;
            for (int64_t k = 0; k < 32; ++k) {
                const int s4 = static_cast<int>((n * 7 + b * 3 + k * 5 + static_cast<int>(phase * 10.0f)) & 0x0F) - 8;
                qs[static_cast<size_t>((b * 32 + k) * N + n)] = static_cast<int8_t>(s4);
            }
        }
    }
    CbQuantWeights w;
    w.bqs = rt.buffer_from_host(0, qs.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
    w.bd = rt.buffer_from_host(0, d_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
    w.bm = rt.buffer_from_host(0, m_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
    return w;
}

CbQuantWeights make_cb_q6k_weights(cloudtpu::PjrtRuntime& rt, int64_t N, int64_t K, float phase) {
    const int64_t NB = K / 16;
    std::vector<int8_t> qs(static_cast<size_t>(NB * 16 * N));
    std::vector<uint16_t> d_bf16(static_cast<size_t>(NB * N));
    for (int64_t b = 0; b < NB; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            const float raw_d = 0.02f * (1.0f + 0.4f * std::cos(static_cast<float>(n * 2 + b) + phase));
            const uint16_t db = cloudtpu::f32_to_bf16(raw_d);
            d_bf16[static_cast<size_t>(b * N + n)] = db;
            for (int64_t k = 0; k < 16; ++k) {
                const int q = (static_cast<int>(n * 5 + b * 3 + k * 7 + static_cast<int>(phase * 10.0f)) % 63) - 31;
                qs[static_cast<size_t>((b * 16 + k) * N + n)] = static_cast<int8_t>(q);
            }
        }
    }
    CbQuantWeights w;
    w.bqs = rt.buffer_from_host(0, qs.data(), cloudtpu::PjrtDType::S8, {1, 1, NB, 16, N});
    w.bd = rt.buffer_from_host(0, d_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
    return w;
}

struct CbModelDims {
    int64_t n_layers = 2;
    int64_t E = 128;
    int64_t H_Q = 4;
    int64_t H_KV = 2;
    int64_t D = 32;
    int64_t H_ff = 128;
    int64_t KV_CAP = 32;
    int64_t N_KV = 24;
    int64_t V_emb = 32;
    int64_t V_out = 16;
};

cloudtpu::GraphProgramSpec build_cb_decode_spec(const CbModelDims& m,
                                                int64_t T,
                                                int64_t mask_rows,
                                                bool use_s64_kv_idx,
                                                int64_t n_out = -1,
                                                bool dual_out_res_norm = false) {
    const int64_t eff_n_out = (n_out >= 0) ? n_out : T;
    const int64_t D_Q = m.H_Q * m.D;
    const int64_t D_KV = m.H_KV * m.D;
    const auto DENSE = cloudtpu::TensorStorageKind::DENSE;
    const auto Q4K = cloudtpu::TensorStorageKind::Q4_K_I4;
    const auto Q6K = cloudtpu::TensorStorageKind::Q6_K_SPLIT;
    const auto F32 = cloudtpu::PjrtDType::F32;
    const auto F16 = cloudtpu::PjrtDType::F16;
    const auto BF16 = cloudtpu::PjrtDType::BF16;
    const auto S32 = cloudtpu::PjrtDType::S32;
    const auto S64 = cloudtpu::PjrtDType::S64;
    const auto S4 = cloudtpu::PjrtDType::S4;
    const auto S8 = cloudtpu::PjrtDType::S8;
    const auto REP = cloudtpu::TensorShardingKind::REPLICATED;

    cloudtpu::GraphProgramSpec spec;
    spec.bf16_matmul = true;
    spec.mm_pad_m = 8;
    uint64_t next_id = 1;

    const uint64_t id_emb_tbl = next_id++;
    const uint64_t id_tokens = next_id++;
    const uint64_t id_pos = next_id++;
    const uint64_t id_row_idx = next_id++;
    const uint64_t id_mask = next_id++;
    const uint64_t id_out_ids = next_id++;
    spec.inputs.push_back({id_emb_tbl, F32, DENSE, {1, 1, m.V_emb, m.E}, REP});
    spec.inputs.push_back({id_tokens, S32, DENSE, {1, 1, 1, T}, REP});
    spec.inputs.push_back({id_pos, S32, DENSE, {1, 1, 1, T}, REP});
    spec.inputs.push_back({id_row_idx, use_s64_kv_idx ? S64 : S32, DENSE, {1, 1, 1, T}, REP});
    spec.inputs.push_back({id_mask, F16, DENSE, {1, 1, mask_rows, m.N_KV}, REP});
    spec.inputs.push_back({id_out_ids, S32, DENSE, {1, 1, 1, eff_n_out}, REP});

    auto add_op = [&](cloudtpu::CloudOpKind kind,
                      std::vector<uint64_t> srcs,
                      cloudtpu::PjrtDType dt,
                      int64_t d0,
                      int64_t d1,
                      int64_t d2,
                      int64_t d3) -> cloudtpu::OpNodeSpec& {
        const uint64_t oid = next_id++;
        cloudtpu::OpNodeSpec op;
        op.kind = kind;
        op.src_ids = std::move(srcs);
        op.dst = {oid, dt, DENSE, {d0, d1, d2, d3}};
        spec.ops.push_back(std::move(op));
        return spec.ops.back();
    };

    uint64_t cur_x = add_op(cloudtpu::CloudOpKind::GET_ROWS, {id_emb_tbl, id_tokens}, F32, 1, 1, T, m.E).dst.id;
    uint64_t last_hmid = cur_x;
    std::vector<uint64_t> kv_out_ids;

    for (int64_t l = 0; l < m.n_layers; ++l) {
        auto add_dense = [&](cloudtpu::PjrtDType dt, int64_t d2, int64_t d3, int32_t alias_idx = -1) {
            const uint64_t tid = next_id++;
            spec.inputs.push_back({tid, dt, DENSE, {1, 1, d2, d3}, REP, alias_idx});
            return tid;
        };
        auto add_q4k = [&](int64_t rows, int64_t cols) {
            const uint64_t tid = next_id++;
            cloudtpu::TensorSpec ts{tid, S4, Q4K, {1, 1, rows, cols}, REP};
            ts.scale_dtype = BF16;
            spec.inputs.push_back(ts);
            return tid;
        };

        const uint64_t id_an = add_dense(F32, 1, m.E);
        const uint64_t id_wq = add_q4k(D_Q, m.E);
        const uint64_t id_wk = add_q4k(D_KV, m.E);
        const uint64_t id_wv = add_q4k(D_KV, m.E);
        const uint64_t id_kcache = add_dense(F16, m.KV_CAP, D_KV, static_cast<int32_t>(2 * l));
        const uint64_t id_vcache = add_dense(F16, m.KV_CAP, D_KV, static_cast<int32_t>(2 * l + 1));
        const uint64_t id_wo = add_q4k(m.E, D_Q);
        const uint64_t id_fn = add_dense(F32, 1, m.E);
        const uint64_t id_wg = add_q4k(m.H_ff, m.E);
        const uint64_t id_wu = add_q4k(m.H_ff, m.E);
        const uint64_t id_wd = add_q4k(m.E, m.H_ff);

        uint64_t id_rn = 0;
        {
            auto& op = add_op(cloudtpu::CloudOpKind::RMS_NORM, {cur_x}, F32, 1, 1, T, m.E);
            op.eps = 1e-5f;
            id_rn = op.dst.id;
        }
        const uint64_t id_xin = add_op(cloudtpu::CloudOpKind::MUL, {id_rn, id_an}, F32, 1, 1, T, m.E).dst.id;

        auto emit_mm = [&](uint64_t w, uint64_t act, int64_t out_dim) {
            auto& op = add_op(cloudtpu::CloudOpKind::MUL_MAT, {w, act}, F32, 1, 1, T, out_dim);
            op.high_precision = true;
            return op.dst.id;
        };
        const uint64_t id_q2d = emit_mm(id_wq, id_xin, D_Q);
        const uint64_t id_k2d = emit_mm(id_wk, id_xin, D_KV);
        const uint64_t id_v2d = emit_mm(id_wv, id_xin, D_KV);

        const uint64_t id_q3d = add_op(cloudtpu::CloudOpKind::RESHAPE, {id_q2d}, F32, 1, T, m.H_Q, m.D).dst.id;
        const uint64_t id_k3d = add_op(cloudtpu::CloudOpKind::RESHAPE, {id_k2d}, F32, 1, T, m.H_KV, m.D).dst.id;

        auto emit_rope = [&](uint64_t src, int64_t n_heads) {
            auto& op = add_op(cloudtpu::CloudOpKind::ROPE, {src, id_pos}, F32, 1, T, n_heads, m.D);
            op.rope.n_dims = static_cast<int32_t>(m.D);
            op.rope.mode = 2;
            op.rope.freq_base = 10000.0f;
            op.rope.freq_scale = 1.0f;
            op.rope.attn_factor = 1.0f;
            return op.dst.id;
        };
        const uint64_t id_qr = emit_rope(id_q3d, m.H_Q);
        const uint64_t id_kr = emit_rope(id_k3d, m.H_KV);
        const uint64_t id_kflat = add_op(cloudtpu::CloudOpKind::RESHAPE, {id_kr}, F32, 1, 1, T, D_KV).dst.id;

        const uint64_t id_kset =
            add_op(cloudtpu::CloudOpKind::SET_ROWS, {id_kflat, id_row_idx, id_kcache}, F16, 1, 1, m.KV_CAP, D_KV).dst.id;
        const uint64_t id_vset =
            add_op(cloudtpu::CloudOpKind::SET_ROWS, {id_v2d, id_row_idx, id_vcache}, F16, 1, 1, m.KV_CAP, D_KV).dst.id;
        kv_out_ids.push_back(id_kset);
        kv_out_ids.push_back(id_vset);

        auto emit_kv_view_perm = [&](uint64_t src_set) {
            uint64_t sl_id = 0;
            {
                auto& op = add_op(cloudtpu::CloudOpKind::SLICE, {src_set}, F16, 1, m.N_KV, m.H_KV, m.D);
                op.slice_starts = {0, 0, 0, 0};
                op.slice_limits = {1, 1, m.N_KV, D_KV};
                sl_id = op.dst.id;
            }
            auto& tp = add_op(cloudtpu::CloudOpKind::TRANSPOSE, {sl_id}, F16, 1, m.H_KV, m.N_KV, m.D);
            tp.perm[0] = 0;
            tp.perm[1] = 2;
            tp.perm[2] = 1;
            tp.perm[3] = 3;
            return tp.dst.id;
        };
        const uint64_t id_kp = emit_kv_view_perm(id_kset);
        const uint64_t id_vp = emit_kv_view_perm(id_vset);

        uint64_t id_qp = 0;
        {
            auto& tp = add_op(cloudtpu::CloudOpKind::TRANSPOSE, {id_qr}, F32, 1, m.H_Q, T, m.D);
            tp.perm[0] = 0;
            tp.perm[1] = 2;
            tp.perm[2] = 1;
            tp.perm[3] = 3;
            id_qp = tp.dst.id;
        }
        uint64_t id_attn = 0;
        {
            auto& fa = add_op(cloudtpu::CloudOpKind::FLASH_ATTN_EXT, {id_qp, id_kp, id_vp, id_mask}, F32, 1, T, m.H_Q, m.D);
            fa.high_precision = true;
            fa.flash_attn.scale = 1.0f / std::sqrt(static_cast<float>(m.D));
            id_attn = fa.dst.id;
        }

        const uint64_t id_aflat = add_op(cloudtpu::CloudOpKind::RESHAPE, {id_attn}, F32, 1, 1, T, D_Q).dst.id;
        const uint64_t id_oproj = emit_mm(id_wo, id_aflat, m.E);
        const uint64_t id_hmid = add_op(cloudtpu::CloudOpKind::ADD, {cur_x, id_oproj}, F32, 1, 1, T, m.E).dst.id;

        uint64_t id_fnorm = 0;
        {
            auto& op = add_op(cloudtpu::CloudOpKind::RMS_NORM, {id_hmid}, F32, 1, 1, T, m.E);
            op.eps = 1e-5f;
            id_fnorm = op.dst.id;
        }
        const uint64_t id_fin = add_op(cloudtpu::CloudOpKind::MUL, {id_fnorm, id_fn}, F32, 1, 1, T, m.E).dst.id;
        const uint64_t id_gf = emit_mm(id_wg, id_fin, m.H_ff);
        const uint64_t id_uf = emit_mm(id_wu, id_fin, m.H_ff);
        const uint64_t id_sw = add_op(cloudtpu::CloudOpKind::SWIGLU, {id_gf, id_uf}, F32, 1, 1, T, m.H_ff).dst.id;
        const uint64_t id_dn = emit_mm(id_wd, id_sw, m.E);
        last_hmid = id_hmid;
        cur_x = add_op(cloudtpu::CloudOpKind::ADD, {id_hmid, id_dn}, F32, 1, 1, T, m.E).dst.id;
    }

    const uint64_t id_wout = next_id++;
    {
        cloudtpu::TensorSpec ts{id_wout, S8, Q6K, {1, 1, m.V_out, m.E}, REP};
        ts.scale_dtype = BF16;
        spec.inputs.push_back(ts);
    }
    uint64_t id_hsel = add_op(cloudtpu::CloudOpKind::GET_ROWS, {cur_x, id_out_ids}, F32, 1, 1, eff_n_out, m.E).dst.id;
    if (dual_out_res_norm) {
        const uint64_t id_sa_sel =
            add_op(cloudtpu::CloudOpKind::GET_ROWS, {last_hmid, id_out_ids}, F32, 1, 1, eff_n_out, m.E).dst.id;
        const uint64_t id_res =
            add_op(cloudtpu::CloudOpKind::ADD, {id_hsel, id_sa_sel}, F32, 1, 1, eff_n_out, m.E).dst.id;
        auto& rn = add_op(cloudtpu::CloudOpKind::RMS_NORM, {id_res}, F32, 1, 1, eff_n_out, m.E);
        rn.eps = 1e-5f;
        id_hsel = rn.dst.id;
    }
    uint64_t id_logits = 0;
    {
        auto& op = add_op(cloudtpu::CloudOpKind::MUL_MAT, {id_wout, id_hsel}, F32, 1, 1, eff_n_out, m.V_out);
        op.high_precision = true;
        id_logits = op.dst.id;
    }

    spec.output_ids = kv_out_ids;
    spec.output_ids.push_back(id_logits);
    return spec;
}

void test_continuous_batching_padded_buckets() {
    // 1) Env parsing and select_batch_bucket contract
    {
        const auto def_b = cloudtpu::parse_batch_buckets(nullptr);
        const std::vector<int64_t> exp_def = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
        CHECK_TRUE(def_b == exp_def, "default batch buckets must be {1,2,4,8,16,32,64,128,256,512}");
        CHECK_TRUE(cloudtpu::parse_batch_buckets("0").empty(), "\"0\" must disable batch bucketing");
        CHECK_TRUE(cloudtpu::parse_batch_buckets("").empty(), "\"\" must disable batch bucketing");
        const auto custom_b = cloudtpu::parse_batch_buckets("16, 4, 8, 4, 1, 2");
        CHECK_TRUE(custom_b == std::vector<int64_t>({1, 2, 4, 8, 16}), "custom batch buckets must sort and deduplicate");

        CHECK_TRUE(cloudtpu::select_batch_bucket(0, def_b) == 0, "n=0 selects 0");
        CHECK_TRUE(cloudtpu::select_batch_bucket(1, def_b) == 1, "n=1 selects 1");
        CHECK_TRUE(cloudtpu::select_batch_bucket(2, def_b) == 2, "n=2 selects 2");
        CHECK_TRUE(cloudtpu::select_batch_bucket(3, def_b) == 4, "n=3 selects 4");
        CHECK_TRUE(cloudtpu::select_batch_bucket(5, def_b) == 8, "n=5 selects 8");
        CHECK_TRUE(cloudtpu::select_batch_bucket(8, def_b) == 8, "n=8 selects 8");
        CHECK_TRUE(cloudtpu::select_batch_bucket(9, def_b) == 16, "n=9 selects 16");
        CHECK_TRUE(cloudtpu::select_batch_bucket(17, def_b) == 32, "n=17 selects 32");
        CHECK_TRUE(cloudtpu::select_batch_bucket(33, def_b) == 64, "n=33 selects 64");
        CHECK_TRUE(cloudtpu::select_batch_bucket(129, def_b) == 256, "n=129 selects 256");
        CHECK_TRUE(cloudtpu::select_batch_bucket(512, def_b) == 512, "n=512 selects 512");
        CHECK_TRUE(cloudtpu::select_batch_bucket(513, def_b) == 0, "n=513 > max returns 0");
        CHECK_TRUE(cloudtpu::select_batch_bucket(5, {}) == 0, "empty buckets disables bucketing");
    }

    // 2) Reshape/view/pad/slice helpers + fail-closed dataflow invariants
    {
        const int64_t d_src1[4] = {1, 1, 5, 32};
        const int64_t d_dst1[4] = {1, 5, 4, 8};
        CHECK_TRUE(cloudtpu::map_token_axis_reshape(d_src1, 4, 2, d_dst1, 4) == 1, "reshape [1,1,5,32]->[1,5,4,8]");
        CHECK_TRUE(cloudtpu::map_token_axis_reshape(d_dst1, 4, 1, d_src1, 4) == 2, "reshape [1,5,4,8]->[1,1,5,32]");
        const int64_t d_amb_src[4] = {1, 1, 1, 8};
        const int64_t d_amb_dst[4] = {1, 1, 1, 8};
        CHECK_TRUE(cloudtpu::map_token_axis_reshape(d_amb_src, 4, 2, d_amb_dst, 4) == -1,
                   "ambiguous T=1 reshape must fail closed");

        std::vector<int64_t> rdims, starts = {0, 0, 0, 0}, limits = {1, 1, 5, 16};
        CHECK_TRUE(cloudtpu::bucket_view_slice_dims(d_src1, 2, 5, 8, rdims, starts, limits), "bucket_view_slice_dims");
        CHECK_TRUE(rdims.empty() && starts == std::vector<int64_t>({0, 0, 0, 0}) &&
                       limits == std::vector<int64_t>({1, 1, 8, 16}),
                   "bucket_view_slice_dims widens token limit [0..T)->[0..B)");

        std::vector<int64_t> rdims2 = {1, 5, 4, 8}, starts2 = {0, 0, 0, 0}, limits2 = {1, 5, 2, 8};
        CHECK_TRUE(cloudtpu::bucket_view_slice_dims(d_src1, 2, 5, 8, rdims2, starts2, limits2),
                   "bucket_view_slice_dims reshaped head view");
        CHECK_TRUE(rdims2 == std::vector<int64_t>({1, 8, 4, 8}) &&
                       limits2 == std::vector<int64_t>({1, 8, 2, 8}),
                   "bucket_view_slice_dims widens reshaped token axis");

        // Strided pad & slice with outer=2 > 1
        const int64_t od[4] = {1, 2, 5, 3};
        const int64_t pd[4] = {1, 2, 8, 3};
        std::vector<float> src(2 * 5 * 3);
        for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i + 1) * 0.5f;
        std::vector<float> padded(2 * 8 * 3, -99.0f);
        cloudtpu::pad_dense_host_tensor(src.data(), padded.data(), cloudtpu::PjrtDType::F32,
                                        od, pd, 2, cloudtpu::TokenInputPadKind::ZERO);
        for (int b1 = 0; b1 < 2; ++b1) {
            for (int t = 5; t < 8; ++t) {
                for (int k = 0; k < 3; ++k) {
                    CHECK_NEAR(padded[static_cast<size_t>((b1 * 8 + t) * 3 + k)], 0.0f, 1e-6f,
                               "strided ZERO pad must clear dummy rows [T..B)");
                }
            }
        }
        std::vector<float> sliced(2 * 5 * 3, 0.0f);
        cloudtpu::slice_dense_host_tensor(padded.data(), sliced.data(), sizeof(float), pd, od, 2);
        for (size_t i = 0; i < src.size(); ++i) {
            CHECK_NEAR(sliced[i], src[i], 1e-6f, "strided slice_dense_host_tensor roundtrip");
        }

        // Fail-closed vs partial-output GET_ROWS(hidden, out_ids):
        // - n_outputs == 2 (< T=5): SUCCEEDS with idx=-1 and dst=-1 (prefill last-token / qwen2 pattern)
        // - n_outputs == 6 (> T=5): FAILS CLOSED with non-empty fail_reason
        {
            cloudtpu::GraphProgramSpec part_gr;
            part_gr.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 16}});
            part_gr.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}});
            part_gr.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 8, 16}});
            part_gr.inputs.push_back({4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 8}});
            cloudtpu::OpNodeSpec mm;
            mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
            mm.src_ids = {3, 1};
            mm.dst = {5, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 8}};
            cloudtpu::OpNodeSpec gr;
            gr.kind = cloudtpu::CloudOpKind::GET_ROWS;
            gr.src_ids = {5, 2};
            gr.dst = {6, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, 8}};
            cloudtpu::OpNodeSpec mul_lhs;
            mul_lhs.kind = cloudtpu::CloudOpKind::MUL;
            mul_lhs.src_ids = {4, 6};
            mul_lhs.dst = {7, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, 8}};
            part_gr.ops = {mm, gr, mul_lhs};
            part_gr.output_ids = {7};
            const auto part_plan = cloudtpu::analyze_program_token_axes(part_gr, 5, 8);
            CHECK_TRUE(part_plan.success && part_plan.input_token_axis[0] == 2 &&
                           part_plan.input_token_axis[1] == -1 && part_plan.output_token_axis[0] == -1,
                       "GET_ROWS(hidden, out_ids) with N_out < T must succeed keeping out invariant");

            // Also verify N_out > T fails closed with non-empty fail_reason
            cloudtpu::GraphProgramSpec over_gr = part_gr;
            over_gr.inputs[1].dims[3] = 6;
            over_gr.ops[1].dst.dims[2] = 6;
            over_gr.ops[2].dst.dims[2] = 6;
            const auto over_plan = cloudtpu::analyze_program_token_axes(over_gr, 5, 8);
            CHECK_TRUE(!over_plan.success && !over_plan.fail_reason.empty(),
                       "GET_ROWS(hidden, out_ids) with N_out > T must fail closed with fail_reason");
        }

        // Fail-closed: shared idx between SET_ROWS (NEG_ONE) and GET_ROWS (ZERO)
        {
            cloudtpu::GraphProgramSpec bad_idx;
            bad_idx.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 16, 8}});
            bad_idx.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 5}});
            bad_idx.inputs.push_back({3, cloudtpu::PjrtDType::F16, cloudtpu::TensorStorageKind::DENSE, {1, 1, 16, 8}});
            cloudtpu::OpNodeSpec gr;
            gr.kind = cloudtpu::CloudOpKind::GET_ROWS;
            gr.src_ids = {1, 2};
            gr.dst = {4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 8}};
            cloudtpu::OpNodeSpec sr;
            sr.kind = cloudtpu::CloudOpKind::SET_ROWS;
            sr.src_ids = {4, 2, 3};
            sr.dst = {5, cloudtpu::PjrtDType::F16, cloudtpu::TensorStorageKind::DENSE, {1, 1, 16, 8}};
            bad_idx.ops = {gr, sr};
            bad_idx.output_ids = {5};
            const auto bad_idx_plan = cloudtpu::analyze_program_token_axes(bad_idx, 5, 8);
            CHECK_TRUE(!bad_idx_plan.success && !bad_idx_plan.fail_reason.empty(),
                       "shared index tensor between GET_ROWS and SET_ROWS must fail closed with fail_reason");
        }

        // Fail-closed: SLICE along token_axis
        {
            cloudtpu::GraphProgramSpec bad_sl;
            bad_sl.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 8, 8}});
            bad_sl.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 5}});
            cloudtpu::OpNodeSpec gr;
            gr.kind = cloudtpu::CloudOpKind::GET_ROWS;
            gr.src_ids = {1, 2};
            gr.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 8}};
            cloudtpu::OpNodeSpec sl;
            sl.kind = cloudtpu::CloudOpKind::SLICE;
            sl.src_ids = {3};
            sl.slice_starts = {0, 0, 1, 0};
            sl.slice_limits = {1, 1, 4, 8};
            sl.dst = {4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 3, 8}};
            bad_sl.ops = {gr, sl};
            bad_sl.output_ids = {4};
            const auto bad_sl_plan = cloudtpu::analyze_program_token_axes(bad_sl, 5, 8);
            CHECK_TRUE(!bad_sl_plan.success && !bad_sl_plan.fail_reason.empty(),
                       "SLICE along token_axis must fail closed with fail_reason");
        }

        // FLASH_ATTN_EXT mask rules:
        // - 1 < mask[2] < T fails closed with fail_reason
        // - T < mask[2] < B (GGML_KQ_MASK_PAD between T and B) pads mask[2] -> B cleanly
        // - mask[2] >= B (64) stays unpadded (-1)
        const CbModelDims dims;
        const auto small_mask = build_cb_decode_spec(dims, 5, 3, true);
        const auto small_mask_plan = cloudtpu::analyze_program_token_axes(small_mask, 5, 8);
        CHECK_TRUE(!small_mask_plan.success && !small_mask_plan.fail_reason.empty(),
                   "mask with 1 < mask[2] < T must fail closed with fail_reason");
        const auto prepad_mask = build_cb_decode_spec(dims, 5, 6, true);
        const auto prepad_plan = cloudtpu::analyze_program_token_axes(prepad_mask, 5, 8);
        CHECK_TRUE(prepad_plan.success && prepad_plan.input_token_axis[4] == 2 &&
                       prepad_plan.input_pad_kind[4] == cloudtpu::TokenInputPadKind::ZERO,
                   "mask with T < mask[2] < B (GGML_PAD) must pad cleanly to B");
        const auto pad64_mask = build_cb_decode_spec(dims, 5, 64, true);
        const auto plan64 = cloudtpu::analyze_program_token_axes(pad64_mask, 5, 8);
        CHECK_TRUE(plan64.success && plan64.input_token_axis[4] == -1,
                   "mask with dims[2] >= B (64) must remain unpadded (-1)");

        // ROPE [1,1,T,D] with scalar pos [1,1,1,1] keeps pos unpadded (-1)
        {
            cloudtpu::GraphProgramSpec rspec;
            rspec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 16}});
            rspec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1}});
            rspec.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 8, 16}});
            cloudtpu::OpNodeSpec mm;
            mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
            mm.src_ids = {3, 1};
            mm.dst = {4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 8}};
            cloudtpu::OpNodeSpec rp;
            rp.kind = cloudtpu::CloudOpKind::ROPE;
            rp.src_ids = {4, 2};
            rp.dst = {5, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 8}};
            rp.rope.n_dims = 8;
            rspec.ops = {mm, rp};
            rspec.output_ids = {5};
            const auto rplan = cloudtpu::analyze_program_token_axes(rspec, 5, 8);
            CHECK_TRUE(rplan.success && rplan.input_token_axis[0] == 2 &&
                           rplan.input_token_axis[1] == -1 && rplan.output_token_axis[0] == 2,
                       "ROPE on [1,1,T,D] with scalar pos must preserve pos=-1 and dst ax=2");
        }
    }

    // 3) End-to-end T=5,6,7,8 continuous-batching decode graph (GET_ROWS + Q4_K_I4/BF16 + ROPE +
    //    SET_ROWS(scatter -1) + FLASH_ATTN_EXT + FFN + GET_ROWS(out_ids) + lm_head(Q6_K/BF16))
    const CbModelDims m;
    const int64_t D_Q = m.H_Q * m.D;
    const int64_t D_KV = m.H_KV * m.D;

    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = true;
    cfg.exec_cache_cap = 64;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);

    std::vector<float> emb_tbl(static_cast<size_t>(m.V_emb * m.E));
    for (size_t i = 0; i < emb_tbl.size(); ++i) {
        emb_tbl[i] = 0.25f * std::sin(static_cast<float>(i + 1) * 0.19f);
    }
    auto b_emb = rt.buffer_from_host(0, emb_tbl.data(), cloudtpu::PjrtDType::F32, {1, 1, m.V_emb, m.E});

    struct LayerWeightBufs {
        cloudtpu::PjrtBufferPtr an, fn;
        CbQuantWeights wq, wk, wv, wo, wg, wu, wd;
    };
    std::vector<LayerWeightBufs> wbufs(static_cast<size_t>(m.n_layers));
    for (int64_t l = 0; l < m.n_layers; ++l) {
        const float lp = static_cast<float>(l) * 0.6f;
        LayerWeightBufs& lb = wbufs[static_cast<size_t>(l)];
        std::vector<float> an(static_cast<size_t>(m.E)), fn(static_cast<size_t>(m.E));
        for (int64_t i = 0; i < m.E; ++i) {
            an[static_cast<size_t>(i)] = 0.85f + 0.1f * std::cos(static_cast<float>(i) * 0.23f + lp);
            fn[static_cast<size_t>(i)] = 0.9f + 0.1f * std::sin(static_cast<float>(i) * 0.29f + lp);
        }
        lb.an = rt.buffer_from_host(0, an.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, m.E});
        lb.fn = rt.buffer_from_host(0, fn.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, m.E});
        lb.wq = make_cb_q4k_weights(rt, D_Q, m.E, 0.1f + lp);
        lb.wk = make_cb_q4k_weights(rt, D_KV, m.E, 0.2f + lp);
        lb.wv = make_cb_q4k_weights(rt, D_KV, m.E, 0.3f + lp);
        lb.wo = make_cb_q4k_weights(rt, m.E, D_Q, 0.4f + lp);
        lb.wg = make_cb_q4k_weights(rt, m.H_ff, m.E, 0.5f + lp);
        lb.wu = make_cb_q4k_weights(rt, m.H_ff, m.E, 0.6f + lp);
        lb.wd = make_cb_q4k_weights(rt, m.E, m.H_ff, 0.7f + lp);
    }
    const CbQuantWeights lm_head = make_cb_q6k_weights(rt, m.V_out, m.E, 0.9f);

    // Non-zero sentinel KV pattern to detect any corruption of untouched slots
    std::vector<uint16_t> sentinel_kv(static_cast<size_t>(m.KV_CAP * D_KV));
    for (size_t i = 0; i < sentinel_kv.size(); ++i) {
        sentinel_kv[i] = cloudtpu::f32_to_f16(0.22f * std::sin(static_cast<float>(i + 3) * 0.37f));
    }

    auto make_step_buffers = [&](int64_t T,
                                 int64_t n_out,
                                 int64_t mask_rows,
                                 int64_t bucket_B,
                                 const cloudtpu::BucketAnalysisResult* plan,
                                 bool use_s64,
                                 std::vector<int64_t>& active_slots_out,
                                 std::vector<cloudtpu::PjrtBufferPtr>& owned_bufs_out) {
        owned_bufs_out.clear();
        active_slots_out.resize(static_cast<size_t>(T));
        std::vector<int32_t> tokens(static_cast<size_t>(T));
        std::vector<int32_t> pos(static_cast<size_t>(T));
        std::vector<int32_t> out_ids(static_cast<size_t>(n_out));
        std::vector<int64_t> row_idx64(static_cast<size_t>(T));
        std::vector<int32_t> row_idx32(static_cast<size_t>(T));
        std::vector<uint16_t> mask(static_cast<size_t>(mask_rows * m.N_KV),
                                   cloudtpu::f32_to_f16(-std::numeric_limits<float>::infinity()));

        const int64_t fixed_slots[8] = {2, 5, 8, 11, 14, 17, 20, 22};
        for (int64_t t = 0; t < T; ++t) {
            tokens[static_cast<size_t>(t)] = static_cast<int32_t>((t * 5 + 3) % m.V_emb);
            pos[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);
            row_idx64[static_cast<size_t>(t)] = fixed_slots[static_cast<size_t>(t)];
            row_idx32[static_cast<size_t>(t)] = static_cast<int32_t>(fixed_slots[static_cast<size_t>(t)]);
            active_slots_out[static_cast<size_t>(t)] = fixed_slots[static_cast<size_t>(t)];
            mask[static_cast<size_t>(t * m.N_KV + 0)] = cloudtpu::f32_to_f16(0.0f);
            mask[static_cast<size_t>(t * m.N_KV + fixed_slots[static_cast<size_t>(t)])] = cloudtpu::f32_to_f16(0.0f);
        }
        for (int64_t t = T; t < mask_rows; ++t) {
            mask[static_cast<size_t>(t * m.N_KV + 0)] = cloudtpu::f32_to_f16(0.0f);
        }
        for (int64_t oi = 0; oi < n_out; ++oi) {
            out_ids[static_cast<size_t>(oi)] = (n_out == T)
                                                   ? static_cast<int32_t>(T - 1 - oi)
                                                   : static_cast<int32_t>(T - n_out + oi);
        }

        auto push_padded = [&](const void* src_ptr, cloudtpu::PjrtDType dt,
                               const int64_t od[4], size_t input_idx) {
            if (plan && plan->success && plan->input_token_axis[input_idx] >= 0) {
                int64_t pd[4] = {od[0], od[1], od[2], od[3]};
                pd[plan->input_token_axis[input_idx]] = bucket_B;
                const size_t nbytes = cloudtpu::pjrt_dtype_nbytes(dt, pd[0] * pd[1] * pd[2] * pd[3]);
                std::vector<uint8_t> tmp(nbytes);
                cloudtpu::pad_dense_host_tensor(src_ptr, tmp.data(), dt, od, pd,
                                                plan->input_token_axis[input_idx],
                                                plan->input_pad_kind[input_idx]);
                owned_bufs_out.push_back(rt.buffer_from_host(0, tmp.data(), dt, {pd[0], pd[1], pd[2], pd[3]}));
            } else {
                owned_bufs_out.push_back(rt.buffer_from_host(0, src_ptr, dt, {od[0], od[1], od[2], od[3]}));
            }
        };
        const int64_t d_1d[4] = {1, 1, 1, T};
        const int64_t d_out_1d[4] = {1, 1, 1, n_out};
        const int64_t d_mk[4] = {1, 1, mask_rows, m.N_KV};
        push_padded(tokens.data(), cloudtpu::PjrtDType::S32, d_1d, 1);
        push_padded(pos.data(), cloudtpu::PjrtDType::S32, d_1d, 2);
        if (use_s64) {
            push_padded(row_idx64.data(), cloudtpu::PjrtDType::S64, d_1d, 3);
        } else {
            push_padded(row_idx32.data(), cloudtpu::PjrtDType::S32, d_1d, 3);
        }
        push_padded(mask.data(), cloudtpu::PjrtDType::F16, d_mk, 4);
        push_padded(out_ids.data(), cloudtpu::PjrtDType::S32, d_out_1d, 5);
    };

    auto run_step = [&](const cloudtpu::GraphProgramSpec& run_spec,
                        const cloudtpu::EmittedProgram& prog,
                        int64_t T,
                        int64_t n_out,
                        int64_t mask_rows,
                        int64_t bucket_B,
                        const cloudtpu::BucketAnalysisResult* plan,
                        bool use_s64,
                        std::vector<std::vector<cloudtpu::PjrtBufferPtr>>& kv_bufs,
                        std::vector<int64_t>& active_slots_out) {
        std::vector<cloudtpu::PjrtBufferPtr> step_in;
        make_step_buffers(T, n_out, mask_rows, bucket_B, plan, use_s64, active_slots_out, step_in);
        std::vector<PJRT_Buffer*> raw_in;
        raw_in.push_back(b_emb->handle.get());
        for (const auto& p : step_in) raw_in.push_back(p->handle.get());
        for (int64_t l = 0; l < m.n_layers; ++l) {
            const LayerWeightBufs& lb = wbufs[static_cast<size_t>(l)];
            auto push_q4k = [&](const CbQuantWeights& w) {
                raw_in.push_back(w.bqs->handle.get());
                raw_in.push_back(w.bd->handle.get());
                raw_in.push_back(w.bm->handle.get());
            };
            raw_in.push_back(lb.an->handle.get());
            push_q4k(lb.wq);
            push_q4k(lb.wk);
            push_q4k(lb.wv);
            raw_in.push_back(kv_bufs[static_cast<size_t>(l)][0]->handle.get());
            raw_in.push_back(kv_bufs[static_cast<size_t>(l)][1]->handle.get());
            push_q4k(lb.wo);
            raw_in.push_back(lb.fn->handle.get());
            push_q4k(lb.wg);
            push_q4k(lb.wu);
            push_q4k(lb.wd);
        }
        raw_in.push_back(lm_head.bqs->handle.get());
        raw_in.push_back(lm_head.bd->handle.get());

        std::vector<cloudtpu::PjrtDType> out_dts;
        std::vector<std::vector<int64_t>> out_dims;
        for (const auto& ob : prog.output_bindings) {
            out_dts.push_back(ob.dtype);
            out_dims.push_back(ob.dims);
        }
        (void) run_spec;
        const auto copts = rt.make_spmd_compile_options(1, 0, true, true);
        const auto* exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, copts);
        std::vector<cloudtpu::PjrtBufferPtr> outs;
        rt.execute(exec, 0, raw_in.data(), raw_in.size(), outs, out_dts, out_dims, true);
        for (int64_t l = 0; l < m.n_layers; ++l) {
            kv_bufs[static_cast<size_t>(l)][0] = outs[static_cast<size_t>(2 * l)];
            kv_bufs[static_cast<size_t>(l)][1] = outs[static_cast<size_t>(2 * l + 1)];
        }
        const bool out_bucketed = (plan && plan->success && plan->output_token_axis.back() >= 0);
        const int64_t eff_out_rows = out_bucketed ? bucket_B : n_out;
        std::vector<float> raw_logits(static_cast<size_t>(eff_out_rows * m.V_out));
        rt.buffer_to_host(outs.back()->handle.get(), raw_logits.data(), raw_logits.size() * sizeof(float));
        if (out_bucketed) {
            const int64_t pd[4] = {1, 1, bucket_B, m.V_out};
            const int64_t ld[4] = {1, 1, n_out, m.V_out};
            std::vector<float> sliced(static_cast<size_t>(n_out * m.V_out));
            cloudtpu::slice_dense_host_tensor(raw_logits.data(), sliced.data(), sizeof(float),
                                              pd, ld, plan->output_token_axis.back());
            return sliced;
        }
        return raw_logits;
    };

    auto init_kv_pair = [&]() {
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> kv(static_cast<size_t>(m.n_layers));
        for (int64_t l = 0; l < m.n_layers; ++l) {
            kv[static_cast<size_t>(l)].push_back(
                rt.buffer_from_host(0, sentinel_kv.data(), cloudtpu::PjrtDType::F16, {1, 1, m.KV_CAP, D_KV}));
            kv[static_cast<size_t>(l)].push_back(
                rt.buffer_from_host(0, sentinel_kv.data(), cloudtpu::PjrtDType::F16, {1, 1, m.KV_CAP, D_KV}));
        }
        return kv;
    };

    auto verify_kv_preservation = [&](const std::vector<std::vector<cloudtpu::PjrtBufferPtr>>& kv_buck,
                                      const std::vector<std::vector<cloudtpu::PjrtBufferPtr>>& kv_unb,
                                      const std::vector<int64_t>& slots_b) {
        for (int64_t l = 0; l < m.n_layers; ++l) {
            for (int kv_idx = 0; kv_idx < 2; ++kv_idx) {
                std::vector<uint16_t> got_b(static_cast<size_t>(m.KV_CAP * D_KV));
                std::vector<uint16_t> got_u(static_cast<size_t>(m.KV_CAP * D_KV));
                rt.buffer_to_host(kv_buck[static_cast<size_t>(l)][static_cast<size_t>(kv_idx)]->handle.get(),
                                  got_b.data(), got_b.size() * sizeof(uint16_t));
                rt.buffer_to_host(kv_unb[static_cast<size_t>(l)][static_cast<size_t>(kv_idx)]->handle.get(),
                                  got_u.data(), got_u.size() * sizeof(uint16_t));
                for (int64_t slot = 0; slot < m.KV_CAP; ++slot) {
                    bool is_active = false;
                    for (int64_t s : slots_b) {
                        if (s == slot) {
                            is_active = true;
                            break;
                        }
                    }
                    for (int64_t d = 0; d < D_KV; ++d) {
                        const size_t off = static_cast<size_t>(slot * D_KV + d);
                        if (!is_active) {
                            CHECK_TRUE(got_b[off] == sentinel_kv[off],
                                       "padded -1 SET_ROWS indices must never corrupt inactive KV slot");
                        } else {
                            CHECK_TRUE(got_b[off] == got_u[off],
                                       "active KV slot values must match unbucketed bit-for-bit");
                        }
                    }
                }
            }
        }
    };

    // 3a) Sequential T=5,6,7,8 must share the exact cache_key and yield misses==1, hits==3
    {
        auto kv_seq = init_kv_pair();
        const cloudtpu::CacheStats st_before = rt.cache_stats();
        std::string key_ref;
        for (int64_t T : {int64_t(5), int64_t(6), int64_t(7), int64_t(8)}) {
            const int64_t B = cloudtpu::select_batch_bucket(T, {1, 2, 4, 8, 16});
            CHECK_TRUE(B == 8, "T in 5..8 must select bucket 8");
            cloudtpu::GraphProgramSpec spec = build_cb_decode_spec(m, T, T, true);
            cloudtpu::BucketAnalysisResult plan;
            if (B > T) {
                plan = cloudtpu::analyze_program_token_axes(spec, T, B);
                CHECK_TRUE(plan.success, "CB decode graph analysis must succeed");
                CHECK_TRUE(cloudtpu::apply_program_batch_bucket(spec, plan), "apply_program_batch_bucket must succeed");
            }
            const cloudtpu::EmittedProgram prog = cloudtpu::emit_stablehlo_program(spec);
            if (key_ref.empty()) {
                key_ref = prog.cache_key;
            } else {
                CHECK_TRUE(prog.cache_key == key_ref, "T=5,6,7,8 bucketed specs must share identical cache_key");
            }
            std::vector<int64_t> active_slots;
            run_step(spec, prog, T, T, T, B, (B > T) ? &plan : nullptr, true, kv_seq, active_slots);
        }
        const cloudtpu::CacheStats st_after = rt.cache_stats();
        CHECK_TRUE(st_after.misses - st_before.misses == 1,
                   "T=5,6,7,8 continuous batching must have misses == 1");
        CHECK_TRUE(st_after.hits - st_before.hits == 3,
                   "T=5,6,7,8 continuous batching must have hits == 3");
    }

    // 3b) Parity (< 1e-4) vs unbucketed and bit-exact KV slot non-corruption (both S64 and S32 indices)
    for (bool use_s64 : {true, false}) {
        for (int64_t T : {int64_t(5), int64_t(6), int64_t(7)}) {
            auto kv_buck = init_kv_pair();
            auto kv_unb = init_kv_pair();

            const int64_t B = 8;
            cloudtpu::GraphProgramSpec spec_u = build_cb_decode_spec(m, T, T, use_s64);
            const cloudtpu::EmittedProgram prog_u = cloudtpu::emit_stablehlo_program(spec_u);
            std::vector<int64_t> slots_u;
            const std::vector<float> logits_u =
                run_step(spec_u, prog_u, T, T, T, T, nullptr, use_s64, kv_unb, slots_u);

            cloudtpu::GraphProgramSpec spec_b = spec_u;
            const cloudtpu::BucketAnalysisResult plan = cloudtpu::analyze_program_token_axes(spec_b, T, B);
            CHECK_TRUE(plan.success && plan.input_pad_kind[3] == cloudtpu::TokenInputPadKind::NEG_ONE,
                       "SET_ROWS index input must require NEG_ONE padding");
            CHECK_TRUE(cloudtpu::apply_program_batch_bucket(spec_b, plan), "apply bucket for parity");
            const cloudtpu::EmittedProgram prog_b = cloudtpu::emit_stablehlo_program(spec_b);
            std::vector<int64_t> slots_b;
            const std::vector<float> logits_b =
                run_step(spec_b, prog_b, T, T, T, B, &plan, use_s64, kv_buck, slots_b);

            CHECK_TRUE(logits_b.size() == static_cast<size_t>(T * m.V_out) &&
                           logits_u.size() == logits_b.size(),
                       "logits output length must match logical [T, V_out]");
            for (size_t i = 0; i < logits_b.size(); ++i) {
                CHECK_NEAR(logits_b[i], logits_u[i], 1e-4f,
                           "bucketed CB decode logits parity < 1e-4 vs unbucketed");
            }
            verify_kv_preservation(kv_buck, kv_unb, slots_b);
        }
    }

    // 3c) Partial prefill/qwen2 inp_out_ids[N_out=1] (< T) + dual GET_ROWS + ADD + RMS_NORM + Q6_K lm_head
    //     across T=5,6,7,8 -> B=8: misses==1, hits==3, < 1e-4 parity vs unbucketed, 0 KV corruption
    {
        auto kv_seq = init_kv_pair();
        const cloudtpu::CacheStats st_before = rt.cache_stats();
        std::string key_ref;
        for (int64_t T : {int64_t(5), int64_t(6), int64_t(7), int64_t(8)}) {
            const int64_t B = cloudtpu::select_batch_bucket(T, {1, 2, 4, 8, 16});
            cloudtpu::GraphProgramSpec spec = build_cb_decode_spec(m, T, T, true, 1, true);
            cloudtpu::BucketAnalysisResult plan;
            if (B > T) {
                plan = cloudtpu::analyze_program_token_axes(spec, T, B);
                CHECK_TRUE(plan.success && plan.input_token_axis[5] == -1 &&
                               plan.output_token_axis.back() == -1,
                           "N_out=1 prefill GET_ROWS must keep out_ids and logits invariant (-1)");
                CHECK_TRUE(cloudtpu::apply_program_batch_bucket(spec, plan), "apply N_out=1 bucket");
            }
            const cloudtpu::EmittedProgram prog = cloudtpu::emit_stablehlo_program(spec);
            if (key_ref.empty()) {
                key_ref = prog.cache_key;
            } else {
                CHECK_TRUE(prog.cache_key == key_ref,
                           "N_out=1 prefill T=5,6,7,8 bucketed specs must share identical cache_key");
            }
            std::vector<int64_t> active_slots;
            run_step(spec, prog, T, 1, T, B, (B > T) ? &plan : nullptr, true, kv_seq, active_slots);
        }
        const cloudtpu::CacheStats st_after = rt.cache_stats();
        CHECK_TRUE(st_after.misses - st_before.misses == 1,
                   "N_out=1 prefill T=5,6,7,8 must have misses == 1");
        CHECK_TRUE(st_after.hits - st_before.hits == 3,
                   "N_out=1 prefill T=5,6,7,8 must have hits == 3");

        for (int64_t T : {int64_t(5), int64_t(6), int64_t(7)}) {
            const int64_t mask_rows = (T == 5) ? 6 : T;  // Also verify T < mask_rows < B runtime parity
            auto kv_buck = init_kv_pair();
            auto kv_unb = init_kv_pair();
            cloudtpu::GraphProgramSpec spec_u = build_cb_decode_spec(m, T, mask_rows, true, 1, true);
            const cloudtpu::EmittedProgram prog_u = cloudtpu::emit_stablehlo_program(spec_u);
            std::vector<int64_t> slots_u;
            const std::vector<float> logits_u =
                run_step(spec_u, prog_u, T, 1, mask_rows, T, nullptr, true, kv_unb, slots_u);

            cloudtpu::GraphProgramSpec spec_b = spec_u;
            const cloudtpu::BucketAnalysisResult plan = cloudtpu::analyze_program_token_axes(spec_b, T, 8);
            CHECK_TRUE(plan.success && cloudtpu::apply_program_batch_bucket(spec_b, plan),
                       "N_out=1 bucket analysis/apply for parity");
            const cloudtpu::EmittedProgram prog_b = cloudtpu::emit_stablehlo_program(spec_b);
            std::vector<int64_t> slots_b;
            const std::vector<float> logits_b =
                run_step(spec_b, prog_b, T, 1, mask_rows, 8, &plan, true, kv_buck, slots_b);

            CHECK_TRUE(logits_b.size() == static_cast<size_t>(m.V_out) && logits_u.size() == logits_b.size(),
                       "N_out=1 logits length must equal [1, V_out]");
            for (size_t i = 0; i < logits_b.size(); ++i) {
                CHECK_NEAR(logits_b[i], logits_u[i], 1e-4f,
                           "N_out=1 prefill bucketed logits parity < 1e-4 vs unbucketed");
            }
            verify_kv_preservation(kv_buck, kv_unb, slots_b);
        }
    }

    rt.shutdown();
    std::printf("  [PASS] Continuous batching padded buckets (T=5..8 -> B=8 + N_out<T prefill: misses=1, hits=3, parity < 1e-4, 0 KV corruption)\n");
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

void test_ggml_cgraph_continuous_batching_buckets() {
    ScopedEnv env_mock("CLOUDTPU_MOCK", "1");
    ScopedEnv env_verb("CLOUDTPU_VERBOSE", "1");
    ScopedEnv env_fuse_cap("CLOUDTPU_MAX_FUSE_OPS", nullptr);
    ScopedEnv env_fuse_attn("CLOUDTPU_FUSE_ATTN", nullptr);
    ScopedEnv env_fuse_graphs("CLOUDTPU_FUSE_GRAPHS", nullptr);

    const int64_t n_layers = 2;
    const int64_t E = 256;
    const int64_t H_Q = 4;
    const int64_t H_KV = 2;
    const int64_t D = 64;
    const int64_t D_Q = H_Q * D;
    const int64_t D_KV = H_KV * D;
    const int64_t H_ff = 256;
    const int64_t KV_CAP = 32;
    const int64_t N_KV = 24;
    const int64_t V_emb = 32;
    const int64_t V_out = 16;

    std::vector<float> emb_data(static_cast<size_t>(V_emb * E));
    for (size_t i = 0; i < emb_data.size(); ++i) {
        emb_data[i] = 0.25f * std::sin(static_cast<float>(i + 1) * 0.17f);
    }
    std::vector<uint16_t> sentinel_kv(static_cast<size_t>(KV_CAP * D_KV));
    for (size_t i = 0; i < sentinel_kv.size(); ++i) {
        sentinel_kv[i] = cloudtpu::f32_to_f16(0.22f * std::sin(static_cast<float>(i + 5) * 0.31f));
    }

    auto fill_q4k = [](size_t nrows, size_t nbytes, float phase) {
        std::vector<uint8_t> buf(nbytes, 0);
        const size_t bs = 144;
        const size_t nblocks = nbytes / bs;
        for (size_t b = 0; b < nblocks; ++b) {
            uint8_t* p = buf.data() + b * bs;
            const uint16_t d = cloudtpu::f32_to_f16(0.02f);
            const uint16_t dm = cloudtpu::f32_to_f16(0.005f);
            std::memcpy(p, &d, 2);
            std::memcpy(p + 2, &dm, 2);
            for (size_t i = 0; i < 12; ++i) {
                p[4 + i] = static_cast<uint8_t>((b * 5 + i * 3 + static_cast<size_t>(phase * 10.0f) + nrows) & 0x1F);
            }
            for (size_t i = 0; i < 128; ++i) {
                p[16 + i] = static_cast<uint8_t>((b * 7 + i * 11 + static_cast<size_t>(phase * 13.0f)) & 0xFF);
            }
        }
        return buf;
    };
    auto fill_q6k = [](size_t nrows, size_t nbytes) {
        std::vector<uint8_t> buf(nbytes, 0);
        const size_t bs = 210;
        const size_t nblocks = nbytes / bs;
        for (size_t b = 0; b < nblocks; ++b) {
            uint8_t* p = buf.data() + b * bs;
            for (size_t i = 0; i < 128; ++i) p[i] = static_cast<uint8_t>((b * 3 + i * 5) & 0xFF);
            for (size_t i = 0; i < 64; ++i) p[128 + i] = static_cast<uint8_t>((b * 7 + i * 9) & 0xFF);
            for (size_t i = 0; i < 16; ++i) p[192 + i] = static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>((b + i) % 7) - 3));
            const uint16_t d = cloudtpu::f32_to_f16(0.015f);
            std::memcpy(p + 208, &d, 2);
        }
        return buf;
    };

    struct CbRunResult {
        std::vector<std::vector<float>> logits_per_step;
        std::vector<std::vector<uint16_t>> kc_after;
        std::vector<std::vector<uint16_t>> vc_after;
        std::string logs;
        bool ok = false;
    };

    auto run_steps = [&](const std::vector<int64_t>& batch_sizes, const char* buckets_env) -> CbRunResult {
        CbRunResult out;
        ScopedEnv env_b("CLOUDTPU_BATCH_BUCKETS", buckets_env);
        GgmlLogCapture cap;
        ggml_backend_t backend = ggml_backend_cloudtpu_init();
        if (!backend) return out;

        const size_t w_mem = ggml_tensor_overhead() * 96 + 4096;
        struct ggml_init_params wp = {w_mem, nullptr, true};
        struct ggml_context* ctx_w = ggml_init(wp);
        struct ggml_context* ctx_kv = ggml_init(wp);
        if (!ctx_w || !ctx_kv) {
            ggml_backend_free(backend);
            return out;
        }

        ggml_tensor* tok_embd = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, E, V_emb);
        ggml_tensor* w_out = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q6_K, E, V_out);
        struct WLayer {
            ggml_tensor *an, *wq, *wk, *wv, *wo, *fn, *wg, *wu, *wd;
            ggml_tensor *kc, *vc;
        };
        std::vector<WLayer> wl(static_cast<size_t>(n_layers));
        for (int64_t l = 0; l < n_layers; ++l) {
            WLayer& w = wl[static_cast<size_t>(l)];
            w.an = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, E);
            w.wq = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, E, D_Q);
            w.wk = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, E, D_KV);
            w.wv = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, E, D_KV);
            w.wo = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, D_Q, E);
            w.fn = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, E);
            w.wg = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, E, H_ff);
            w.wu = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, E, H_ff);
            w.wd = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, H_ff, E);
            w.kc = ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F16, D_KV, KV_CAP);
            w.vc = ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F16, D_KV, KV_CAP);
        }

        ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
        ggml_backend_buffer_t kv_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, backend);
        if (!w_buf || !kv_buf) {
            if (kv_buf) ggml_backend_buffer_free(kv_buf);
            if (w_buf) ggml_backend_buffer_free(w_buf);
            ggml_free(ctx_kv);
            ggml_free(ctx_w);
            ggml_backend_free(backend);
            return out;
        }

        ggml_backend_tensor_set(tok_embd, emb_data.data(), 0, emb_data.size() * sizeof(float));
        const auto q6_bytes = fill_q6k(static_cast<size_t>(V_out), ggml_nbytes(w_out));
        ggml_backend_tensor_set(w_out, q6_bytes.data(), 0, q6_bytes.size());
        for (int64_t l = 0; l < n_layers; ++l) {
            const float lp = static_cast<float>(l) * 0.5f;
            const WLayer& w = wl[static_cast<size_t>(l)];
            std::vector<float> an(static_cast<size_t>(E)), fn(static_cast<size_t>(E));
            for (int64_t i = 0; i < E; ++i) {
                an[static_cast<size_t>(i)] = 0.85f + 0.1f * std::cos(static_cast<float>(i) * 0.2f + lp);
                fn[static_cast<size_t>(i)] = 0.9f + 0.1f * std::sin(static_cast<float>(i) * 0.25f + lp);
            }
            ggml_backend_tensor_set(w.an, an.data(), 0, an.size() * sizeof(float));
            ggml_backend_tensor_set(w.fn, fn.data(), 0, fn.size() * sizeof(float));
            auto set_q4k = [&](ggml_tensor* t, float ph) {
                const auto b = fill_q4k(static_cast<size_t>(t->ne[1]), ggml_nbytes(t), ph + lp);
                ggml_backend_tensor_set(t, b.data(), 0, b.size());
            };
            set_q4k(w.wq, 0.1f);
            set_q4k(w.wk, 0.2f);
            set_q4k(w.wv, 0.3f);
            set_q4k(w.wo, 0.4f);
            set_q4k(w.wg, 0.5f);
            set_q4k(w.wu, 0.6f);
            set_q4k(w.wd, 0.7f);
            ggml_backend_tensor_set(w.kc, sentinel_kv.data(), 0, sentinel_kv.size() * sizeof(uint16_t));
            ggml_backend_tensor_set(w.vc, sentinel_kv.data(), 0, sentinel_kv.size() * sizeof(uint16_t));
        }

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        const int64_t fixed_slots[8] = {2, 5, 8, 11, 14, 17, 20, 22};
        for (int64_t T : batch_sizes) {
            const size_t g_mem = ggml_tensor_overhead() * 256 + ggml_graph_overhead() + 4096;
            struct ggml_init_params gp = {g_mem, nullptr, true};
            struct ggml_context* ctx_g = ggml_init(gp);
            struct ggml_cgraph* gf = ggml_new_graph(ctx_g);

            ggml_tensor* inp_tokens = ggml_new_tensor_1d(ctx_g, GGML_TYPE_I32, T);
            ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_g, GGML_TYPE_I32, T);
            ggml_tensor* inp_row_idx = ggml_new_tensor_1d(ctx_g, GGML_TYPE_I64, T);
            ggml_tensor* inp_mask = ggml_new_tensor_4d(ctx_g, GGML_TYPE_F16, N_KV, T, 1, 1);
            ggml_tensor* inp_out_ids = ggml_new_tensor_1d(ctx_g, GGML_TYPE_I32, T);
            ggml_set_input(inp_tokens);
            ggml_set_input(inp_pos);
            ggml_set_input(inp_row_idx);
            ggml_set_input(inp_mask);
            ggml_set_input(inp_out_ids);

            ggml_tensor* cur = ggml_get_rows(ctx_g, tok_embd, inp_tokens);
            for (int64_t l = 0; l < n_layers; ++l) {
                const WLayer& w = wl[static_cast<size_t>(l)];
                ggml_tensor* rn = ggml_rms_norm(ctx_g, cur, 1e-5f);
                ggml_tensor* xin = ggml_mul(ctx_g, rn, w.an);
                ggml_tensor* q = ggml_mul_mat(ctx_g, w.wq, xin);
                ggml_tensor* k = ggml_mul_mat(ctx_g, w.wk, xin);
                ggml_tensor* v = ggml_mul_mat(ctx_g, w.wv, xin);

                ggml_tensor* q3 = ggml_reshape_3d(ctx_g, q, D, H_Q, T);
                ggml_tensor* k3 = ggml_reshape_3d(ctx_g, k, D, H_KV, T);
                ggml_tensor* qr = ggml_rope_ext(
                    ctx_g, q3, inp_pos, nullptr, static_cast<int>(D), GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                ggml_tensor* kr = ggml_rope_ext(
                    ctx_g, k3, inp_pos, nullptr, static_cast<int>(D), GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                ggml_tensor* kflat = ggml_reshape_2d(ctx_g, kr, D_KV, T);

                ggml_tensor* kset = ggml_set_rows(ctx_g, w.kc, kflat, inp_row_idx);
                ggml_tensor* vset = ggml_set_rows(ctx_g, w.vc, v, inp_row_idx);
                ggml_build_forward_expand(gf, kset);
                ggml_build_forward_expand(gf, vset);

                ggml_tensor* kview = ggml_view_4d(
                    ctx_g, w.kc, D, H_KV, N_KV, 1,
                    static_cast<size_t>(D) * ggml_element_size(w.kc), w.kc->nb[1], w.kc->nb[2], 0);
                ggml_tensor* vview = ggml_view_4d(
                    ctx_g, w.vc, D, H_KV, N_KV, 1,
                    static_cast<size_t>(D) * ggml_element_size(w.vc), w.vc->nb[1], w.vc->nb[2], 0);
                ggml_tensor* qp = ggml_permute(ctx_g, qr, 0, 2, 1, 3);
                ggml_tensor* kp = ggml_permute(ctx_g, kview, 0, 2, 1, 3);
                ggml_tensor* vp = ggml_permute(ctx_g, vview, 0, 2, 1, 3);

                ggml_tensor* attn = ggml_flash_attn_ext(
                    ctx_g, qp, kp, vp, inp_mask, 1.0f / std::sqrt(static_cast<float>(D)), 0.0f, 0.0f);
                ggml_prec_set_acc(attn, GGML_PREC_F32);
                ggml_tensor* aflat = ggml_reshape_2d(ctx_g, attn, D_Q, T);
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
            ggml_tensor* hsel = ggml_get_rows(ctx_g, cur, inp_out_ids);
            ggml_tensor* logits = ggml_mul_mat(ctx_g, w_out, hsel);
            ggml_set_output(logits);
            ggml_build_forward_expand(gf, logits);

            if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                ggml_free(ctx_g);
                break;
            }

            std::vector<int32_t> tokens(static_cast<size_t>(T));
            std::vector<int32_t> pos(static_cast<size_t>(T));
            std::vector<int64_t> row_idx64(static_cast<size_t>(T));
            std::vector<int32_t> out_ids(static_cast<size_t>(T));
            std::vector<uint16_t> mask_f16(static_cast<size_t>(T * N_KV),
                                           cloudtpu::f32_to_f16(-std::numeric_limits<float>::infinity()));
            for (int64_t t = 0; t < T; ++t) {
                tokens[static_cast<size_t>(t)] = static_cast<int32_t>((t * 5 + 3) % V_emb);
                pos[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);
                row_idx64[static_cast<size_t>(t)] = fixed_slots[static_cast<size_t>(t)];
                out_ids[static_cast<size_t>(t)] = static_cast<int32_t>(T - 1 - t);
                mask_f16[static_cast<size_t>(t * N_KV + 0)] = cloudtpu::f32_to_f16(0.0f);
                mask_f16[static_cast<size_t>(t * N_KV + fixed_slots[static_cast<size_t>(t)])] = cloudtpu::f32_to_f16(0.0f);
            }
            ggml_backend_tensor_set(inp_tokens, tokens.data(), 0, tokens.size() * sizeof(int32_t));
            ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
            ggml_backend_tensor_set(inp_row_idx, row_idx64.data(), 0, row_idx64.size() * sizeof(int64_t));
            ggml_backend_tensor_set(inp_mask, mask_f16.data(), 0, mask_f16.size() * sizeof(uint16_t));
            ggml_backend_tensor_set(inp_out_ids, out_ids.data(), 0, out_ids.size() * sizeof(int32_t));

            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx_g);
                break;
            }
            std::vector<float> y(static_cast<size_t>(T * V_out), 0.0f);
            ggml_backend_tensor_get(logits, y.data(), 0, y.size() * sizeof(float));
            // Also verify partial offset/size tensor_get on bucketed HBM slots
            if (T >= 2) {
                std::vector<float> y_row1(static_cast<size_t>(V_out), 0.0f);
                ggml_backend_tensor_get(logits, y_row1.data(),
                                        static_cast<size_t>(V_out) * sizeof(float),
                                        y_row1.size() * sizeof(float));
                for (int64_t v = 0; v < V_out; ++v) {
                    if (std::fabs(y_row1[static_cast<size_t>(v)] - y[static_cast<size_t>(V_out + v)]) > 1e-6f) {
                        ggml_free(ctx_g);
                        ggml_gallocr_free(galloc);
                        ggml_backend_buffer_free(kv_buf);
                        ggml_backend_buffer_free(w_buf);
                        ggml_free(ctx_kv);
                        ggml_free(ctx_w);
                        ggml_backend_free(backend);
                        return out;
                    }
                }
            }
            out.logits_per_step.push_back(std::move(y));
            ggml_free(ctx_g);
        }

        if (out.logits_per_step.size() == batch_sizes.size()) {
            out.kc_after.resize(static_cast<size_t>(n_layers));
            out.vc_after.resize(static_cast<size_t>(n_layers));
            for (int64_t l = 0; l < n_layers; ++l) {
                out.kc_after[static_cast<size_t>(l)].resize(static_cast<size_t>(KV_CAP * D_KV));
                out.vc_after[static_cast<size_t>(l)].resize(static_cast<size_t>(KV_CAP * D_KV));
                ggml_backend_tensor_get(wl[static_cast<size_t>(l)].kc,
                                        out.kc_after[static_cast<size_t>(l)].data(), 0,
                                        static_cast<size_t>(KV_CAP * D_KV) * sizeof(uint16_t));
                ggml_backend_tensor_get(wl[static_cast<size_t>(l)].vc,
                                        out.vc_after[static_cast<size_t>(l)].data(), 0,
                                        static_cast<size_t>(KV_CAP * D_KV) * sizeof(uint16_t));
            }
            out.ok = true;
        }
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(kv_buf);
        ggml_backend_buffer_free(w_buf);
        ggml_free(ctx_kv);
        ggml_free(ctx_w);
        ggml_backend_free(backend);
        out.logs = cap.buf;
        return out;
    };

    // 1) T=5,6,7,8 sequence with default CLOUDTPU_BATCH_BUCKETS -> plan_misses=1, plan_hits=3
    const CbRunResult seq_res = run_steps({5, 6, 7, 8}, nullptr);
    CHECK_TRUE(seq_res.ok, "ggml CB bucket sequence T=5,6,7,8 succeeded");
    const uint64_t misses = parse_log_u64(seq_res.logs, "plan_misses");
    const uint64_t hits = parse_log_u64(seq_res.logs, "plan_hits");
    const uint64_t bucketed = parse_log_u64(seq_res.logs, "bucketed");
    CHECK_TRUE(misses == 1 && hits == 3, "ggml CB T=5,6,7,8 must yield plan_misses==1, plan_hits==3");
    CHECK_TRUE(bucketed == 3, "ggml CB T=5,6,7 must increment bucketed==3");

    // 2) Single-step parity (< 1e-4) vs unbucketed (CLOUDTPU_BATCH_BUCKETS=0) and zero KV corruption
    const int64_t fixed_slots[8] = {2, 5, 8, 11, 14, 17, 20, 22};
    for (int64_t T : {int64_t(5), int64_t(6), int64_t(7)}) {
        const CbRunResult b_res = run_steps({T}, nullptr);
        const CbRunResult u_res = run_steps({T}, "0");
        CHECK_TRUE(b_res.ok && u_res.ok, "ggml CB bucketed vs unbucketed runs succeeded");
        for (size_t i = 0; i < b_res.logits_per_step[0].size(); ++i) {
            CHECK_NEAR(b_res.logits_per_step[0][i], u_res.logits_per_step[0][i], 1e-4f,
                       "ggml CB bucketed logits parity < 1e-4 vs unbucketed");
        }
        for (int64_t l = 0; l < n_layers; ++l) {
            for (int64_t slot = 0; slot < KV_CAP; ++slot) {
                bool active = false;
                for (int64_t t = 0; t < T; ++t) {
                    if (fixed_slots[static_cast<size_t>(t)] == slot) {
                        active = true;
                        break;
                    }
                }
                for (int64_t d = 0; d < D_KV; ++d) {
                    const size_t off = static_cast<size_t>(slot * D_KV + d);
                    if (!active) {
                        CHECK_TRUE(b_res.kc_after[static_cast<size_t>(l)][off] == sentinel_kv[off] &&
                                       b_res.vc_after[static_cast<size_t>(l)][off] == sentinel_kv[off],
                                   "ggml CB unused KV slots must remain untouched");
                    } else {
                        CHECK_TRUE(b_res.kc_after[static_cast<size_t>(l)][off] ==
                                       u_res.kc_after[static_cast<size_t>(l)][off] &&
                                       b_res.vc_after[static_cast<size_t>(l)][off] ==
                                           u_res.vc_after[static_cast<size_t>(l)][off],
                                   "ggml CB active KV slots must match unbucketed");
                    }
                }
            }
        }
    }

    std::printf("  [PASS] ggml_cgraph CB buckets (T=5..8: plan_misses=1, plan_hits=3, parity < 1e-4, 0 KV corruption)\n");
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
    test_bf16_scales_zero_f32_dequant_and_pad_m(rt);
    test_one_graph_multilayer_decode_and_donation();
    test_one_graph_spmd_kv_donation(4);
    test_continuous_batching_padded_buckets();
#ifdef CLOUDTPU_HAS_GGML
    test_ggml_cgraph_one_graph_decode();
    test_ggml_cgraph_continuous_batching_buckets();
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
