#include "pjrt_runtime.hpp"
#include "stablehlo_emitter.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <locale>
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

void check_vec_close(const std::vector<float>& got,
                     const std::vector<float>& exp,
                     float tol,
                     const char* msg) {
    CHECK_TRUE(got.size() == exp.size(), msg);
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK_NEAR(got[i], exp[i], tol, msg);
    }
}

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
    std::printf("  [PASS] compile_options_proto (incl. DeviceAssignmentProto 8-chip)\n");
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

    // Verify touch_exec_cache refreshes runtime LRU recency (cap=3: 2 recurring subgraphs +
    // 2 alternating dynamic attention shapes must evict the older attention shape, never the
    // recurring subgraphs).
    {
        cloudtpu::RuntimeConfig cap3;
        cap3.mock_mode = true;
        cap3.exec_cache_cap = 3;
        cloudtpu::PjrtRuntime rt3;
        rt3.init(cap3);

        const auto p_rec0 = make_add(4);
        const auto p_rec1 = make_add(8);
        const auto p_attn0 = make_add(16);
        const auto p_attn1 = make_add(32);

        const auto ex_rec0 = rt3.compile_or_get(p_rec0.cache_key, p_rec0.mlir_text);
        const auto ex_rec1 = rt3.compile_or_get(p_rec1.cache_key, p_rec1.mlir_text);
        const auto ex_attn0 = rt3.compile_or_get(p_attn0.cache_key, p_attn0.mlir_text);
        // Simulate decode step plan-cache hits refreshing recurring executables in runtime LRU.
        CHECK_TRUE(rt3.touch_exec_cache(ex_rec0->key), "touch_exec_cache finds rec0");
        CHECK_TRUE(rt3.touch_exec_cache(ex_rec1->key), "touch_exec_cache finds rec1");
        // Compiling a new attention shape should evict attn0, NOT rec0 or rec1.
        const auto ex_attn1 = rt3.compile_or_get(p_attn1.cache_key, p_attn1.mlir_text);
        CHECK_TRUE(rt3.has_cached_exec(ex_rec0->key), "rec0 survives eviction after touch_exec_cache");
        CHECK_TRUE(rt3.has_cached_exec(ex_rec1->key), "rec1 survives eviction after touch_exec_cache");
        CHECK_TRUE(rt3.has_cached_exec(ex_attn1->key), "attn1 present in cache");
        CHECK_TRUE(!rt3.has_cached_exec(ex_attn0->key), "stale attn0 evicted first");
        CHECK_TRUE(!rt3.touch_exec_cache(ex_attn0->key), "touch_exec_cache returns false for evicted entry");
        // Next step: touch rec0/rec1 again and verify alternating back to attn0 evicts attn1.
        CHECK_TRUE(rt3.touch_exec_cache(ex_rec0->key), "rec0 still hits after attn1 insert");
        CHECK_TRUE(rt3.touch_exec_cache(ex_rec1->key), "rec1 still hits after attn1 insert");
        const auto ex_attn0_b = rt3.compile_or_get(p_attn0.cache_key, p_attn0.mlir_text);
        CHECK_TRUE(rt3.has_cached_exec(ex_rec0->key) && rt3.has_cached_exec(ex_rec1->key),
                   "recurring subgraphs never evicted across alternating attention shapes");
        CHECK_TRUE(rt3.has_cached_exec(ex_attn0_b->key) && !rt3.has_cached_exec(ex_attn1->key),
                   "alternating attention shape evicts previous attention shape");
        rt3.shutdown();
    }

    std::printf("  [PASS] H2D/D2H roundtrip & executable LRU cache (incl. touch_exec_cache recency)\n");
}

std::vector<float> run_emitted(cloudtpu::PjrtRuntime& rt,
                               const cloudtpu::EmittedProgram& prog,
                               const std::vector<cloudtpu::PjrtBufferPtr>& inputs) {
    std::vector<PJRT_Buffer*> raw_in;
    for (const auto& b : inputs) raw_in.push_back(b->handle.get());
    const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0);
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
                    if (!std::isfinite(got[base]) || !std::isfinite(got[base + 1]) ||
                        std::fabs(got[base] - r0) > 1e-4f || std::fabs(got[base + 1] - r1) > 1e-4f) {
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

    // 2) Q4_0_SPLIT (N=16, NB=8, K=256): dequant-only CPY + MUL_MAT (M=4)
    {
        const int64_t N = 16, NB = 8, K = NB * 32, M = 4;
        std::vector<uint8_t> qs(static_cast<size_t>(N * NB * 16));
        std::vector<uint16_t> d(static_cast<size_t>(N * NB));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t b = 0; b < NB; ++b) {
                const float sc = 0.05f * static_cast<float>((n + b) % 7 + 1);
                const uint16_t d16 = cloudtpu::f32_to_f16(sc);
                d[static_cast<size_t>(n * NB + b)] = d16;
                const float scale = cloudtpu::f16_to_f32(d16);
                for (int64_t j = 0; j < 16; ++j) {
                    const uint8_t lo = static_cast<uint8_t>((n * 3 + b * 5 + j) & 0x0F);
                    const uint8_t hi = static_cast<uint8_t>((n * 7 + b * 11 + j * 3 + 2) & 0x0F);
                    qs[static_cast<size_t>((n * NB + b) * 16 + j)] = static_cast<uint8_t>(lo | (hi << 4));
                    w_ref[static_cast<size_t>(n * K + b * 32 + j)] = static_cast<float>(static_cast<int>(lo) - 8) * scale;
                    w_ref[static_cast<size_t>(n * K + b * 32 + j + 16)] = static_cast<float>(static_cast<int>(hi) - 8) * scale;
                }
            }
        }
        auto bqs = rt.buffer_from_host(0, qs.data(), cloudtpu::PjrtDType::U8, {1, 1, N, NB, 16});
        auto bd = rt.buffer_from_host(0, d.data(), cloudtpu::PjrtDType::F16, {1, 1, N, NB});

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
                if (!std::isfinite(wgot[i]) || std::fabs(wgot[i] - w_ref[i]) > 1e-4f) {
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

        // MUL_MAT Q4_0 K=256 check
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = std::cos(static_cast<float>(i) * 0.17f);
        }
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});
        {
            cloudtpu::GraphProgramSpec mspec;
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
            if (!std::isfinite(nmse) || nmse > 5e-4) {
                std::fprintf(stderr, "FAIL [Q4_0 MUL_MAT K=256]: nmse=%.6f got[0]=%.5f\n", nmse, ygot[0]);
                ++g_failures;
            }
        }
    }

    // 3) Q8_0_SPLIT (N=16, NB=8, K=256, M=4)
    {
        const int64_t N = 16, NB = 8, K = NB * 32, M = 4;
        std::vector<int8_t> qs(static_cast<size_t>(N * NB * 32));
        std::vector<uint16_t> d(static_cast<size_t>(N * NB));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t b = 0; b < NB; ++b) {
                const uint16_t d16 = cloudtpu::f32_to_f16(0.01f * static_cast<float>((n + b) % 5 + 1));
                d[static_cast<size_t>(n * NB + b)] = d16;
                const float scale = cloudtpu::f16_to_f32(d16);
                for (int64_t j = 0; j < 32; ++j) {
                    const int8_t q = static_cast<int8_t>(((n * 13 + b * 29 + j * 7) % 251) - 125);
                    qs[static_cast<size_t>((n * NB + b) * 32 + j)] = q;
                    w_ref[static_cast<size_t>(n * K + b * 32 + j)] = static_cast<float>(q) * scale;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.23f);

        auto bqs = rt.buffer_from_host(0, qs.data(), cloudtpu::PjrtDType::S8, {1, 1, N, NB, 32});
        auto bd = rt.buffer_from_host(0, d.data(), cloudtpu::PjrtDType::F16, {1, 1, N, NB});
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
        if (!std::isfinite(nmse) || nmse > 5e-4) {
            std::fprintf(stderr, "FAIL [Q8_0 MUL_MAT K=256]: nmse=%.6f got[0]=%.5f\n", nmse, ygot[0]);
            ++g_failures;
        }
    }

    std::printf("  [PASS] Silicon regressions (ROPE M=52, Q4_0/Q8_0 K=256)\n");
}

void test_multichip_device_assignment_and_eviction_lifetime() {
    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = true;
    cfg.exec_cache_cap = 1;
    cfg.mock_devices = 2;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);
    CHECK_TRUE(rt.device_count() == 2, "mock 2-chip device_count == 2");

    cloudtpu::GraphProgramSpec spec;
    spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
    spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
    cloudtpu::OpNodeSpec op;
    op.kind = cloudtpu::CloudOpKind::ADD;
    op.src_ids = {1, 2};
    op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
    spec.ops.push_back(op);
    spec.output_ids = {3};
    const auto prog = cloudtpu::emit_stablehlo_program(spec);

    {
        // Compile for chip 0, then compile same program for chip 1 (evicting chip 0 because cap=1).
        const auto exec0 = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0);
        const auto exec1 = rt.compile_or_get(prog.cache_key, prog.mlir_text, 1);
        CHECK_TRUE(exec0 != exec1, "distinct executables compiled per device");
        CHECK_TRUE(rt.cache_stats().evictions == 1, "cap=1 evicted exec0 when exec1 inserted");

        // Held shared_ptr keeps evicted exec0 alive through dispatch on chip 0.
        const std::vector<float> a0 = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> b0 = {10.0f, 20.0f, 30.0f, 40.0f};
        auto ba0 = rt.buffer_from_host(0, a0.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
        auto bb0 = rt.buffer_from_host(0, b0.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
        PJRT_Buffer* in0[2] = {ba0->handle.get(), bb0->handle.get()};
        std::vector<cloudtpu::PjrtBufferPtr> out0;
        rt.execute(exec0, 0, in0, 2, out0, {cloudtpu::PjrtDType::F32}, {{1, 1, 1, 4}}, true);
        std::vector<float> h0(4, 0.0f);
        rt.buffer_to_host(out0[0]->handle.get(), h0.data(), h0.size() * sizeof(float));
        check_vec_close(h0, {11.0f, 22.0f, 33.0f, 44.0f}, 1e-6f, "evicted exec0 dispatch");

        // Execute exec1 on chip 1.
        const std::vector<float> a1 = {2.0f, 4.0f, 6.0f, 8.0f};
        const std::vector<float> b1 = {1.0f, 1.0f, 1.0f, 1.0f};
        auto ba1 = rt.buffer_from_host(1, a1.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
        auto bb1 = rt.buffer_from_host(1, b1.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
        PJRT_Buffer* in1[2] = {ba1->handle.get(), bb1->handle.get()};
        std::vector<cloudtpu::PjrtBufferPtr> out1;
        rt.execute(exec1, 1, in1, 2, out1, {cloudtpu::PjrtDType::F32}, {{1, 1, 1, 4}}, true);
        std::vector<float> h1(4, 0.0f);
        rt.buffer_to_host(out1[0]->handle.get(), h1.data(), h1.size() * sizeof(float));
        check_vec_close(h1, {3.0f, 5.0f, 7.0f, 9.0f}, 1e-6f, "chip 1 dispatch");

        // Mismatched device execution must fail cleanly.
        bool threw = false;
        try {
            std::vector<cloudtpu::PjrtBufferPtr> bad_out;
            rt.execute(exec0, 1, in1, 2, bad_out, {cloudtpu::PjrtDType::F32}, {{1, 1, 1, 4}}, true);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK_TRUE(threw, "mismatched device execution rejected");

        // Repeated SSA value in return list must remain valid in every output buffer.
        cloudtpu::GraphProgramSpec dup_spec = spec;
        dup_spec.output_ids = {3, 3};
        const auto dup_prog = cloudtpu::emit_stablehlo_program(dup_spec);
        const auto dup_exec = rt.compile_or_get(dup_prog.cache_key, dup_prog.mlir_text, 0);
        std::vector<cloudtpu::PjrtBufferPtr> dup_outs;
        rt.execute(dup_exec, 0, in0, 2, dup_outs,
                   {cloudtpu::PjrtDType::F32, cloudtpu::PjrtDType::F32},
                   {{1, 1, 1, 4}, {1, 1, 1, 4}}, true);
        std::vector<float> d0(4, 0.0f), d1(4, 0.0f);
        rt.buffer_to_host(dup_outs[0]->handle.get(), d0.data(), d0.size() * sizeof(float));
        rt.buffer_to_host(dup_outs[1]->handle.get(), d1.data(), d1.size() * sizeof(float));
        check_vec_close(d0, {11.0f, 22.0f, 33.0f, 44.0f}, 1e-6f, "repeated return output[0]");
        check_vec_close(d1, {11.0f, 22.0f, 33.0f, 44.0f}, 1e-6f, "repeated return output[1]");

        // Distinct partitioner settings on the same program & device must not alias in cache.
        cloudtpu::PjrtCompileOptions opt_shardy_on;
        opt_shardy_on.use_shardy_partitioner = true;
        cloudtpu::PjrtCompileOptions opt_shardy_off;
        opt_shardy_off.use_shardy_partitioner = false;
        const auto ex_on = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, opt_shardy_on);
        const auto ex_off = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, opt_shardy_off);
        CHECK_TRUE(ex_on != ex_off, "different partitioner options must use distinct cache entries");
    }

    rt.shutdown();
    std::printf("  [PASS] Multi-chip device assignment & cache eviction lifetime\n");
}

void test_flash_attn_fully_masked_row(cloudtpu::PjrtRuntime& rt) {
    // B=1, H_Q=1, H_KV=1, N_Q=2, N_KV=4, D=4:
    // Query row 0 is unmasked; query row 1 is fully masked with -INFINITY and must return 0.0f.
    constexpr int64_t B = 1, H = 1, NQ = 2, NKV = 4, D = 4;
    const std::vector<float> q = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.5f,
    };
    const std::vector<float> k = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const std::vector<float> v = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
    };
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const std::vector<float> mask = {
        0.0f,    0.0f,    0.0f,    0.0f,
        neg_inf, neg_inf, neg_inf, neg_inf,
    };

    auto bq = rt.buffer_from_host(0, q.data(), cloudtpu::PjrtDType::F32, {B, H, NQ, D});
    auto bk = rt.buffer_from_host(0, k.data(), cloudtpu::PjrtDType::F32, {B, H, NKV, D});
    auto bv = rt.buffer_from_host(0, v.data(), cloudtpu::PjrtDType::F32, {B, H, NKV, D});
    auto bm = rt.buffer_from_host(0, mask.data(), cloudtpu::PjrtDType::F32, {B, H, NQ, NKV});

    cloudtpu::GraphProgramSpec spec;
    spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {B, H, NQ, D}});
    spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {B, H, NKV, D}});
    spec.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {B, H, NKV, D}});
    spec.inputs.push_back({4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {B, H, NQ, NKV}});
    cloudtpu::OpNodeSpec fa;
    fa.kind = cloudtpu::CloudOpKind::FLASH_ATTN_EXT;
    fa.src_ids = {1, 2, 3, 4};
    fa.dst = {5, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {B, NQ, H, D}};
    fa.flash_attn.scale = 1.0f;
    fa.high_precision = true;
    spec.ops.push_back(fa);
    spec.output_ids = {5};

    const auto out = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec), {bq, bk, bv, bm});
    // Row 0 scores = [1, 0, 0, 0] -> softmax probs
    const float e1 = std::exp(1.0f);
    const float denom = e1 + 3.0f;
    const float p[4] = {e1 / denom, 1.0f / denom, 1.0f / denom, 1.0f / denom};
    std::vector<float> expected(static_cast<size_t>(NQ * D), 0.0f);
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < NKV; ++j) {
            acc += p[j] * v[static_cast<size_t>(j * D + d)];
        }
        expected[static_cast<size_t>(d)] = acc;
        expected[static_cast<size_t>(D + d)] = 0.0f;  // Fully masked row must be exact zero.
    }
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK_TRUE(std::isfinite(out[i]), "FLASH_ATTN_EXT output must be finite");
    }
    check_vec_close(out, expected, 1e-5f, "FLASH_ATTN_EXT fully masked row");
    std::printf("  [PASS] FLASH_ATTN_EXT fully -INF masked row returns zero\n");

    // Verify masked NaN/Inf K and V rows cannot contaminate another query that masks them,
    // even when another query in the same batch attends to the poisoned row.
    const float nan_f = std::numeric_limits<float>::quiet_NaN();
    const float pos_inf = std::numeric_limits<float>::infinity();
    std::vector<float> k_poison = k;
    std::vector<float> v_poison = v;
    for (int d = 0; d < D; ++d) {
        k_poison[static_cast<size_t>(2 * D + d)] = nan_f;    // masked for all queries
        v_poison[static_cast<size_t>(2 * D + d)] = pos_inf;  // masked for all queries
    }
    v_poison[static_cast<size_t>(3 * D + 0)] = nan_f;  // only channel 0 poisoned; channels 1..D-1 remain finite
    const std::vector<float> mask_iso = {
        0.0f,    0.0f,    neg_inf, neg_inf,  // q0 sees rows 0, 1 only
        neg_inf, neg_inf, neg_inf, 0.0f,     // q1 sees row 3 only (ch0=NaN, ch1..3=finite)
    };
    auto bkp = rt.buffer_from_host(0, k_poison.data(), cloudtpu::PjrtDType::F32, {B, H, NKV, D});
    auto bvp = rt.buffer_from_host(0, v_poison.data(), cloudtpu::PjrtDType::F32, {B, H, NKV, D});
    auto bmi = rt.buffer_from_host(0, mask_iso.data(), cloudtpu::PjrtDType::F32, {B, H, NQ, NKV});
    const auto out_iso = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec), {bq, bkp, bvp, bmi});
    const float p0 = e1 / (e1 + 1.0f);
    const float p1 = 1.0f / (e1 + 1.0f);
    std::vector<float> q0_got(out_iso.begin(), out_iso.begin() + D);
    std::vector<float> q0_exp(static_cast<size_t>(D));
    for (int d = 0; d < D; ++d) {
        q0_exp[static_cast<size_t>(d)] = p0 * v[static_cast<size_t>(d)] + p1 * v[static_cast<size_t>(D + d)];
    }
    CHECK_TRUE(std::isnan(out_iso[static_cast<size_t>(D + 0)]),
               "query attending to NaN V channel should surface NaN on that channel");
    for (int d = 1; d < D; ++d) {
        CHECK_TRUE(std::fabs(out_iso[static_cast<size_t>(D + d)] - v[static_cast<size_t>(3 * D + d)]) < 1e-5f,
                   "unaffected finite V channels must remain preserved");
    }
    check_vec_close(q0_got, q0_exp, 1e-5f, "FLASH_ATTN_EXT masked NaN/Inf KV isolation");
    std::printf("  [PASS] FLASH_ATTN_EXT isolates masked NaN/Inf K/V rows\n");
}

void test_q4_k_split_mul_mat(cloudtpu::PjrtRuntime& rt) {
    // Verify Q4_K_SPLIT emission & execution preserves rank-4 dims for both scales (d) and mins (m).
    constexpr int64_t M = 2, K = 32, N = 3, NB = K / 32;
    std::vector<uint8_t> qs(static_cast<size_t>(N * NB * 16), 0);
    std::vector<float> d(static_cast<size_t>(N * NB), 0.5f);
    std::vector<float> m(static_cast<size_t>(N * NB), -1.0f);
    std::vector<float> w_ref(static_cast<size_t>(N * K), 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        d[static_cast<size_t>(n)] = 0.25f * static_cast<float>(n + 1);
        m[static_cast<size_t>(n)] = -0.5f * static_cast<float>(n + 1);
        for (int64_t i = 0; i < 16; ++i) {
            const uint8_t lo = static_cast<uint8_t>((n + i) & 0x0F);
            const uint8_t hi = static_cast<uint8_t>((n + i + 3) & 0x0F);
            qs[static_cast<size_t>(n * 16 + i)] = static_cast<uint8_t>(lo | (hi << 4));
            w_ref[static_cast<size_t>(n * K + i)] = static_cast<float>(lo) * d[static_cast<size_t>(n)] + m[static_cast<size_t>(n)];
            w_ref[static_cast<size_t>(n * K + 16 + i)] = static_cast<float>(hi) * d[static_cast<size_t>(n)] + m[static_cast<size_t>(n)];
        }
    }
    std::vector<float> x(static_cast<size_t>(M * K));
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * static_cast<float>(static_cast<int>(i % 7) - 3);

    auto bqs = rt.buffer_from_host(0, qs.data(), cloudtpu::PjrtDType::U8, {1, 1, N, NB, 16});
    auto bd = rt.buffer_from_host(0, d.data(), cloudtpu::PjrtDType::F32, {1, 1, N, NB});
    auto bm = rt.buffer_from_host(0, m.data(), cloudtpu::PjrtDType::F32, {1, 1, N, NB});
    auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

    cloudtpu::GraphProgramSpec spec;
    spec.inputs.push_back({1, cloudtpu::PjrtDType::U8, cloudtpu::TensorStorageKind::Q4_K_SPLIT, {1, 1, N, K}});
    spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
    cloudtpu::OpNodeSpec mm;
    mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
    mm.src_ids = {1, 2};
    mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
    mm.high_precision = true;
    spec.ops.push_back(mm);
    spec.output_ids = {3};

    const auto prog = cloudtpu::emit_stablehlo_program(spec);
    CHECK_TRUE(prog.input_bindings.size() == 4, "Q4_K_SPLIT MUL_MAT has 4 input bindings (qs, d, m, x)");
    CHECK_TRUE(prog.input_bindings[2].dims == std::vector<int64_t>({1, 1, N, NB}), "Q4_K mins binding rank-4");

    const auto got = run_emitted(rt, prog, {bqs, bd, bm, bx});
    std::vector<float> expected(static_cast<size_t>(M * N), 0.0f);
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += x[static_cast<size_t>(r * K + k)] * w_ref[static_cast<size_t>(n * K + k)];
            }
            expected[static_cast<size_t>(r * N + n)] = acc;
        }
    }
    check_vec_close(got, expected, 1e-4f, "Q4_K_SPLIT MUL_MAT");
    std::printf("  [PASS] Q4_K_SPLIT MUL_MAT (rank-4 scales & mins)\n");

    // Verify SET_ROWS preserves 64-bit scatter indices (i64, not narrowed to i32).
    const std::vector<float> tbl_init = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> upd_vals = {9.0f, 8.0f};
    const std::vector<int64_t> row_idx64 = {1};
    auto b_upd = rt.buffer_from_host(0, upd_vals.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 2});
    auto b_idx = rt.buffer_from_host(0, row_idx64.data(), cloudtpu::PjrtDType::S64, {1, 1, 1, 1});
    auto b_tbl = rt.buffer_from_host(0, tbl_init.data(), cloudtpu::PjrtDType::F32, {1, 1, 3, 2});
    cloudtpu::GraphProgramSpec sr_spec;
    sr_spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}});
    sr_spec.inputs.push_back({2, cloudtpu::PjrtDType::S64, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1}});
    sr_spec.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 3, 2}});
    cloudtpu::OpNodeSpec sr_op;
    sr_op.kind = cloudtpu::CloudOpKind::SET_ROWS;
    sr_op.src_ids = {1, 2, 3};
    sr_op.dst = {4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 3, 2}};
    sr_spec.ops.push_back(sr_op);
    sr_spec.output_ids = {4};
    const auto sr_prog = cloudtpu::emit_stablehlo_program(sr_spec);
    CHECK_TRUE(sr_prog.mlir_text.find("tensor<1x1x1x3xi64>") != std::string::npos,
               "SET_ROWS with S64 indices must preserve i64 scatter coordinates");
    const auto sr_out = run_emitted(rt, sr_prog, {b_upd, b_idx, b_tbl});
    check_vec_close(sr_out, {1.0f, 2.0f, 9.0f, 8.0f, 5.0f, 6.0f}, 1e-6f, "SET_ROWS S64");
    std::printf("  [PASS] SET_ROWS preserves 64-bit scatter indices\n");

    // 7. FP16 subnormal rounding preserves sticky bits above 2^-25
    const float half_sub = std::ldexp(1.0f, -25);
    const float just_above_half_sub = std::nextafter(half_sub, 1.0f);
    CHECK_TRUE(cloudtpu::f32_to_f16(half_sub) == 0x0000U, "exact 2^-25 rounds to even 0");
    CHECK_TRUE(cloudtpu::f32_to_f16(just_above_half_sub) == 0x0001U,
               "nextafter(2^-25, +inf) rounds to smallest positive FP16 subnormal");
    CHECK_TRUE(cloudtpu::f32_to_f16(-just_above_half_sub) == 0x8001U,
               "-nextafter(2^-25, +inf) rounds to smallest negative FP16 subnormal");
    std::printf("  [PASS] FP16 subnormal sticky-bit rounding\n");

    // 8. SCALE with non-finite bias preserves IEEE -inf literal in StableHLO and mock execution
    std::vector<float> sc_in = {1.0f, 2.0f, 3.0f, 4.0f};
    auto b_sc = rt.buffer_from_host(0, sc_in.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
    cloudtpu::GraphProgramSpec sc_spec;
    sc_spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
    cloudtpu::OpNodeSpec sc_op;
    sc_op.kind = cloudtpu::CloudOpKind::SCALE;
    sc_op.src_ids = {1};
    sc_op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
    sc_op.scale = 1.0f;
    sc_op.bias = -std::numeric_limits<float>::infinity();
    sc_spec.ops.push_back(sc_op);
    sc_spec.output_ids = {2};
    const auto sc_prog = cloudtpu::emit_stablehlo_program(sc_spec);
    CHECK_TRUE(sc_prog.mlir_text.find("0xFF800000") != std::string::npos,
               "non-finite -inf constant must emit exact IEEE hex literal");
    const auto sc_out = run_emitted(rt, sc_prog, {b_sc});
    CHECK_TRUE(sc_out.size() == 4 && std::isinf(sc_out[0]) && sc_out[0] < 0.0f,
               "SCALE with -inf bias must produce -inf output");
    std::printf("  [PASS] SCALE preserves non-finite -inf constant\n");

    // 7) Mock buffer surviving runtime shutdown must destroy safely without UAF.
    {
        cloudtpu::PjrtBufferPtr orphan_buf;
        {
            cloudtpu::PjrtRuntime tmp_rt;
            cloudtpu::RuntimeConfig tmp_cfg;
            tmp_cfg.mock_mode = true;
            tmp_rt.init(tmp_cfg);
            const std::vector<float> payload = {1.0f, 2.0f, 3.0f, 4.0f};
            orphan_buf = tmp_rt.buffer_from_host(0, payload.data(), cloudtpu::PjrtDType::F32, {4});
            tmp_rt.shutdown();
        }
        orphan_buf.reset();
        std::printf("  [PASS] Mock PjrtBufferPtr outliving runtime shutdown destroys cleanly\n");
    }

    // 8) StableHLO emission must be independent of global C++ decimal-comma locales.
    {
        struct CommaPunct : std::numpunct<char> {
            char do_decimal_point() const override { return ','; }
            char do_thousands_sep() const override { return '.'; }
            std::string do_grouping() const override { return "\3"; }
        };
        const std::locale prev_loc = std::locale::global(std::locale(std::locale::classic(), new CommaPunct()));
        cloudtpu::GraphProgramSpec loc_spec;
        loc_spec.inputs = {{1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1024}}};
        cloudtpu::OpNodeSpec rms;
        rms.kind = cloudtpu::CloudOpKind::RMS_NORM;
        rms.src_ids = {1};
        rms.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1024}};
        rms.eps = 0.125f;
        loc_spec.ops.push_back(rms);
        loc_spec.output_ids = {2};
        const auto loc_prog = cloudtpu::emit_stablehlo_program(loc_spec);
        // Also compile and execute under the decimal-comma locale to prove mock parser independence.
        std::vector<float> loc_in(1024, 1.0f);
        auto b_loc = rt.buffer_from_host(0, loc_in.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 1024});
        const auto loc_out = run_emitted(rt, loc_prog, {b_loc});
        std::locale::global(prev_loc);
        CHECK_TRUE(loc_prog.mlir_text.find("1024xf32") != std::string::npos,
                   "dimension formatting must not insert locale thousands separators");
        CHECK_TRUE(loc_prog.mlir_text.find("1.25000000e-01") != std::string::npos,
                   "float formatting must use classic decimal point under comma locale");
        CHECK_TRUE(loc_prog.mlir_text.find("1,25") == std::string::npos,
                   "float formatting must not emit comma decimal separator");
        const float expected_rms = 1.0f / std::sqrt(1.0f + 0.125f);
        CHECK_TRUE(loc_out.size() == 1024 && std::fabs(loc_out[0] - expected_rms) < 1e-5f,
                   "mock StableHLO parser must parse float constants with eps=0.125 under comma locale");
        std::printf("  [PASS] StableHLO emission & mock parsing ignore global C++ locale\n");
    }
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

    auto run_spmd = [&](const cloudtpu::EmittedProgram& prog,
                        const std::vector<std::vector<cloudtpu::PjrtBufferPtr>>& per_dev_in) {
        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
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

void test_spmd_unequal_k_shard_broadcast() {
    // Regression test: with P=2 and SHARD_K inputs shaped [1024, 32] and [512, 32] in GGML order
    // (HLO [1, 1, 32, 1024] and [1, 1, 32, 512]), ADD and MUL must gather before repeat-broadcasting
    // along K so columns 256..511 match b[256..511] rather than repeating b[0..255].
    constexpr int P = 2;
    constexpr int64_t N = 32;
    constexpr int64_t KA = 1024;
    constexpr int64_t KB = 512;
    constexpr int64_t KA_LOCAL = KA / P;
    constexpr int64_t KB_LOCAL = KB / P;

    cloudtpu::RuntimeConfig cfg;
    cfg.mock_mode = true;
    cfg.mock_devices = P;
    cfg.exec_cache_cap = 64;
    cloudtpu::PjrtRuntime rt;
    rt.init(cfg);

    std::vector<float> a_full(static_cast<size_t>(N * KA));
    std::vector<float> b_full(static_cast<size_t>(N * KB));
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t k = 0; k < KA; ++k) {
            a_full[static_cast<size_t>(r * KA + k)] = static_cast<float>(r * 1000 + k) * 0.01f;
        }
        for (int64_t k = 0; k < KB; ++k) {
            // Distinct values across first half (0..255) vs second half (256..511)
            b_full[static_cast<size_t>(r * KB + k)] = static_cast<float>(k + 1) + static_cast<float>(r) * 0.25f;
        }
    }

    std::vector<std::vector<cloudtpu::PjrtBufferPtr>> per_dev(P);
    for (int p = 0; p < P; ++p) {
        std::vector<float> a_sh(static_cast<size_t>(N * KA_LOCAL));
        std::vector<float> b_sh(static_cast<size_t>(N * KB_LOCAL));
        for (int64_t r = 0; r < N; ++r) {
            std::memcpy(a_sh.data() + static_cast<size_t>(r * KA_LOCAL),
                        a_full.data() + static_cast<size_t>(r * KA + p * KA_LOCAL),
                        static_cast<size_t>(KA_LOCAL) * sizeof(float));
            std::memcpy(b_sh.data() + static_cast<size_t>(r * KB_LOCAL),
                        b_full.data() + static_cast<size_t>(r * KB + p * KB_LOCAL),
                        static_cast<size_t>(KB_LOCAL) * sizeof(float));
        }
        per_dev[static_cast<size_t>(p)] = {
            rt.buffer_from_host(p, a_sh.data(), cloudtpu::PjrtDType::F32, {1, 1, N, KA_LOCAL}),
            rt.buffer_from_host(p, b_sh.data(), cloudtpu::PjrtDType::F32, {1, 1, N, KB_LOCAL}),
        };
    }

    const auto copts = rt.make_spmd_compile_options(P);
    for (cloudtpu::CloudOpKind kind : {cloudtpu::CloudOpKind::ADD, cloudtpu::CloudOpKind::MUL}) {
        cloudtpu::GraphProgramSpec spec;
        spec.spmd_chips = P;
        spec.inputs.push_back(
            {1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, N, KA}, cloudtpu::TensorShardingKind::SHARD_K});
        spec.inputs.push_back(
            {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, N, KB}, cloudtpu::TensorShardingKind::SHARD_K});
        cloudtpu::OpNodeSpec op;
        op.kind = kind;
        op.src_ids = {1, 2};
        op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, N, KA}};
        spec.ops.push_back(op);
        spec.output_ids = {3};

        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.all_gather") != std::string::npos,
                   "unequal-width SHARD_K ADD/MUL must all_gather before broadcast");
        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
        std::vector<std::vector<PJRT_Buffer*>> raw_in(P);
        std::vector<PJRT_Buffer* const*> in_ptrs(P);
        for (int p = 0; p < P; ++p) {
            for (const auto& b : per_dev[static_cast<size_t>(p)]) raw_in[static_cast<size_t>(p)].push_back(b->handle.get());
            in_ptrs[static_cast<size_t>(p)] = raw_in[static_cast<size_t>(p)].data();
        }
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
        rt.execute_spmd(exec, P, in_ptrs.data(), prog.input_bindings.size(), outs,
                        {cloudtpu::PjrtDType::F32}, {{1, 1, N, KA}}, true);

        for (int p = 0; p < P; ++p) {
            std::vector<float> got(static_cast<size_t>(N * KA));
            rt.buffer_to_host(outs[static_cast<size_t>(p)][0]->handle.get(), got.data(), got.size() * sizeof(float));
            for (int64_t r = 0; r < N; ++r) {
                for (int64_t k = 0; k < KA; ++k) {
                    const float av = a_full[static_cast<size_t>(r * KA + k)];
                    const float bv = b_full[static_cast<size_t>(r * KB + (k % KB))];
                    const float exp_v = (kind == cloudtpu::CloudOpKind::ADD) ? (av + bv) : (av * bv);
                    const float diff = std::fabs(got[static_cast<size_t>(r * KA + k)] - exp_v);
                    if (!std::isfinite(diff) || diff > 1e-3f) {
                        std::fprintf(stderr, "FAIL [SPMD unequal SHARD_K %s p=%d r=%lld k=%lld]: got=%.4f exp=%.4f\n",
                                     (kind == cloudtpu::CloudOpKind::ADD) ? "ADD" : "MUL",
                                     p, static_cast<long long>(r), static_cast<long long>(k),
                                     got[static_cast<size_t>(r * KA + k)], exp_v);
                        ++g_failures;
                        break;
                    }
                }
            }
        }
    }

    rt.shutdown();
    std::printf("  [PASS] SPMD P=2 unequal-width SHARD_K broadcast ([1024, 32] + [512, 32] ADD/MUL)\n");
}

void test_spmd_shard_n_get_rows_and_collectives() {
    // 1) P=4 SHARD_N GET_ROWS (both DENSE F32 and Q8_0_SPLIT): must NOT all_gather the full table;
    //    must gather only requested rows on each replica and all_reduce the small lookup result.
    {
        constexpr int P = 4;
        constexpr int64_t N = 16;  // 4 rows per chip
        constexpr int64_t N_LOCAL = N / P;
        constexpr int64_t K = 32;
        constexpr int64_t N_IDX = 5;

        cloudtpu::RuntimeConfig cfg;
        cfg.mock_mode = true;
        cfg.mock_devices = P;
        cfg.exec_cache_cap = 64;
        cloudtpu::PjrtRuntime rt;
        rt.init(cfg);
        const auto copts = rt.make_spmd_compile_options(P);

        const std::vector<int32_t> indices = {13, 0, 7, 15, 4};

        // Dense F32 SHARD_N table
        std::vector<float> tbl(static_cast<size_t>(N * K));
        for (int64_t r = 0; r < N; ++r) {
            for (int64_t c = 0; c < K; ++c) {
                tbl[static_cast<size_t>(r * K + c)] = static_cast<float>(r * 100 + c) * 0.25f;
            }
        }
        cloudtpu::GraphProgramSpec spec;
        spec.spmd_chips = P;
        spec.inputs.push_back(
            {1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, N, K}, cloudtpu::TensorShardingKind::SHARD_N});
        spec.inputs.push_back(
            {2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, N_IDX}, cloudtpu::TensorShardingKind::REPLICATED});
        cloudtpu::OpNodeSpec op;
        op.kind = cloudtpu::CloudOpKind::GET_ROWS;
        op.src_ids = {1, 2};
        op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, N_IDX, K}};
        spec.ops.push_back(op);
        spec.output_ids = {3};

        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.all_gather") == std::string::npos,
                   "SHARD_N GET_ROWS must not all_gather the full embedding table");
        CHECK_TRUE(prog.mlir_text.find("stablehlo.replica_id") != std::string::npos &&
                       prog.mlir_text.find("stablehlo.all_reduce") != std::string::npos,
                   "SHARD_N GET_ROWS must use replica_id local gather + all_reduce");

        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> per_dev(P);
        std::vector<std::vector<PJRT_Buffer*>> raw_in(P);
        std::vector<PJRT_Buffer* const*> in_ptrs(P);
        for (int p = 0; p < P; ++p) {
            auto bt = rt.buffer_from_host(
                p, tbl.data() + static_cast<size_t>(p * N_LOCAL * K), cloudtpu::PjrtDType::F32, {1, 1, N_LOCAL, K});
            auto bi = rt.buffer_from_host(p, indices.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, N_IDX});
            per_dev[static_cast<size_t>(p)] = {bt, bi};
            raw_in[static_cast<size_t>(p)] = {bt->handle.get(), bi->handle.get()};
            in_ptrs[static_cast<size_t>(p)] = raw_in[static_cast<size_t>(p)].data();
        }
        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
        rt.execute_spmd(exec, P, in_ptrs.data(), prog.input_bindings.size(), outs,
                        {cloudtpu::PjrtDType::F32}, {{1, 1, N_IDX, K}}, true);
        for (int p = 0; p < P; ++p) {
            std::vector<float> got(static_cast<size_t>(N_IDX * K));
            rt.buffer_to_host(outs[static_cast<size_t>(p)][0]->handle.get(), got.data(), got.size() * sizeof(float));
            for (int64_t ii = 0; ii < N_IDX; ++ii) {
                const int64_t row = indices[static_cast<size_t>(ii)];
                for (int64_t c = 0; c < K; ++c) {
                    CHECK_NEAR(got[static_cast<size_t>(ii * K + c)],
                               tbl[static_cast<size_t>(row * K + c)], 1e-5f, "SHARD_N GET_ROWS dense");
                }
            }
        }

        // Quantized Q8_0_SPLIT SHARD_N table (K=32, 1 block per row)
        std::vector<int8_t> qs(static_cast<size_t>(N * K));
        std::vector<uint16_t> ds(static_cast<size_t>(N));
        for (int64_t r = 0; r < N; ++r) {
            ds[static_cast<size_t>(r)] = cloudtpu::f32_to_f16(0.5f * static_cast<float>(r + 1));
            for (int64_t c = 0; c < K; ++c) {
                qs[static_cast<size_t>(r * K + c)] = static_cast<int8_t>((r + c) % 19 - 9);
            }
        }
        cloudtpu::GraphProgramSpec qspec;
        qspec.spmd_chips = P;
        qspec.inputs.push_back(
            {1, cloudtpu::PjrtDType::S8, cloudtpu::TensorStorageKind::Q8_0_SPLIT, {1, 1, N, K}, cloudtpu::TensorShardingKind::SHARD_N});
        qspec.inputs.push_back(
            {2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, N_IDX}, cloudtpu::TensorShardingKind::REPLICATED});
        qspec.ops.push_back(op);
        qspec.output_ids = {3};
        const auto qprog = cloudtpu::emit_stablehlo_program(qspec);
        CHECK_TRUE(qprog.mlir_text.find("stablehlo.all_gather") == std::string::npos,
                   "Q8_0_SPLIT SHARD_N GET_ROWS must not all_gather the full table");
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> q_per_dev(P);
        std::vector<std::vector<PJRT_Buffer*>> q_raw_in(P);
        std::vector<PJRT_Buffer* const*> q_in_ptrs(P);
        for (int p = 0; p < P; ++p) {
            auto bq = rt.buffer_from_host(
                p, qs.data() + static_cast<size_t>(p * N_LOCAL * K), cloudtpu::PjrtDType::S8, {1, 1, N_LOCAL, 1, K});
            auto bd = rt.buffer_from_host(
                p, ds.data() + static_cast<size_t>(p * N_LOCAL), cloudtpu::PjrtDType::F16, {1, 1, N_LOCAL, 1});
            auto bi = rt.buffer_from_host(p, indices.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, N_IDX});
            q_per_dev[static_cast<size_t>(p)] = {bq, bd, bi};
            q_raw_in[static_cast<size_t>(p)] = {bq->handle.get(), bd->handle.get(), bi->handle.get()};
            q_in_ptrs[static_cast<size_t>(p)] = q_raw_in[static_cast<size_t>(p)].data();
        }
        const auto qexec = rt.compile_or_get(qprog.cache_key, qprog.mlir_text, 0, copts);
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> qouts;
        rt.execute_spmd(qexec, P, q_in_ptrs.data(), qprog.input_bindings.size(), qouts,
                        {cloudtpu::PjrtDType::F32}, {{1, 1, N_IDX, K}}, true);
        std::vector<float> qgot(static_cast<size_t>(N_IDX * K));
        rt.buffer_to_host(qouts[0][0]->handle.get(), qgot.data(), qgot.size() * sizeof(float));
        for (int64_t ii = 0; ii < N_IDX; ++ii) {
            const int64_t row = indices[static_cast<size_t>(ii)];
            const float scale = cloudtpu::f16_to_f32(ds[static_cast<size_t>(row)]);
            for (int64_t c = 0; c < K; ++c) {
                const float exp_v = static_cast<float>(qs[static_cast<size_t>(row * K + c)]) * scale;
                CHECK_NEAR(qgot[static_cast<size_t>(ii * K + c)], exp_v, 1e-4f, "SHARD_N GET_ROWS Q8_0");
            }
        }
        rt.shutdown();
        std::printf("  [PASS] SPMD P=4 SHARD_N GET_ROWS (dense + Q8_0_SPLIT, zero all_gather)\n");
    }

    // 2) P=2 REDUCE_SCATTER_SUM feeding downstream CPY preserves replica-local shape without all_gather.
    // 3) Integer S32 precision preservation (16777217) in MockPJRT ALL_REDUCE_SUM & REDUCE_SCATTER_SUM.
    {
        constexpr int P = 2;
        cloudtpu::RuntimeConfig cfg;
        cfg.mock_mode = true;
        cfg.mock_devices = P;
        cfg.exec_cache_cap = 64;
        cloudtpu::PjrtRuntime rt;
        rt.init(cfg);
        const auto copts = rt.make_spmd_compile_options(P);

        // REDUCE_SCATTER_SUM [1,1,1,4] -> [1,1,1,2] -> CPY [1,1,1,2]
        cloudtpu::GraphProgramSpec rs_cpy_spec;
        rs_cpy_spec.spmd_chips = P;
        rs_cpy_spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
        cloudtpu::OpNodeSpec rs_op;
        rs_op.kind = cloudtpu::CloudOpKind::REDUCE_SCATTER_SUM;
        rs_op.collective_dim = 3;
        rs_op.src_ids = {1};
        rs_op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}};
        cloudtpu::OpNodeSpec cpy_op;
        cpy_op.kind = cloudtpu::CloudOpKind::CPY;
        cpy_op.src_ids = {2};
        cpy_op.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}};
        rs_cpy_spec.ops = {rs_op, cpy_op};
        rs_cpy_spec.output_ids = {3};

        const auto rs_prog = cloudtpu::emit_stablehlo_program(rs_cpy_spec);
        CHECK_TRUE(rs_prog.mlir_text.find("stablehlo.all_gather") == std::string::npos,
                   "REDUCE_SCATTER_SUM -> CPY must stay replica-local (no all_gather)");
        const std::vector<float> r0 = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> r1 = {10.0f, 20.0f, 30.0f, 40.0f};
        auto b0 = rt.buffer_from_host(0, r0.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
        auto b1 = rt.buffer_from_host(1, r1.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
        PJRT_Buffer* in0[1] = {b0->handle.get()};
        PJRT_Buffer* in1[1] = {b1->handle.get()};
        PJRT_Buffer* const* in_ptrs[2] = {in0, in1};
        const auto rs_exec = rt.compile_or_get(rs_prog.cache_key, rs_prog.mlir_text, 0, copts);
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> rs_outs;
        rt.execute_spmd(rs_exec, P, in_ptrs, 1, rs_outs, {cloudtpu::PjrtDType::F32}, {{1, 1, 1, 2}}, true);
        std::vector<float> h0(2), h1(2);
        rt.buffer_to_host(rs_outs[0][0]->handle.get(), h0.data(), h0.size() * sizeof(float));
        rt.buffer_to_host(rs_outs[1][0]->handle.get(), h1.data(), h1.size() * sizeof(float));
        check_vec_close(h0, {11.0f, 22.0f}, 1e-5f, "REDUCE_SCATTER -> CPY chip 0");
        check_vec_close(h1, {33.0f, 44.0f}, 1e-5f, "REDUCE_SCATTER -> CPY chip 1");

        // Exact S32 precision preservation in ALL_REDUCE_SUM (P=1 & P=2) and REDUCE_SCATTER_SUM (P=2)
        constexpr int32_t kOddInt = 16777217;  // 2^24 + 1 (not exactly representable in F32)
        const std::vector<int32_t> i0 = {kOddInt, 0, kOddInt, 1};
        const std::vector<int32_t> i1 = {0, kOddInt, kOddInt, 2};
        auto bi0 = rt.buffer_from_host(0, i0.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, 4});
        auto bi1 = rt.buffer_from_host(1, i1.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, 4});

        // P=1 S32 ALL_REDUCE_SUM identity on 16777217
        {
            cloudtpu::GraphProgramSpec s1;
            s1.spmd_chips = 1;
            s1.inputs.push_back({1, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
            cloudtpu::OpNodeSpec aop;
            aop.kind = cloudtpu::CloudOpKind::ALL_REDUCE_SUM;
            aop.src_ids = {1};
            aop.dst = {2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
            s1.ops.push_back(aop);
            s1.output_ids = {2};
            const auto p1 = cloudtpu::emit_stablehlo_program(s1);
            const auto e1 = rt.compile_or_get(p1.cache_key, p1.mlir_text, 0);
            PJRT_Buffer* raw1[1] = {bi0->handle.get()};
            std::vector<cloudtpu::PjrtBufferPtr> o1;
            rt.execute(e1, 0, raw1, 1, o1, {cloudtpu::PjrtDType::S32}, {{1, 1, 1, 4}}, true);
            std::vector<int32_t> got1(4, 0);
            rt.buffer_to_host(o1[0]->handle.get(), got1.data(), got1.size() * sizeof(int32_t));
            CHECK_TRUE(got1[0] == kOddInt && got1[2] == kOddInt,
                       "P=1 S32 ALL_REDUCE_SUM must preserve 16777217 exactly");
        }

        // P=2 S32 ALL_REDUCE_SUM + REDUCE_SCATTER_SUM on 16777217
        {
            cloudtpu::GraphProgramSpec s2;
            s2.spmd_chips = P;
            s2.inputs.push_back({1, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
            cloudtpu::OpNodeSpec aop;
            aop.kind = cloudtpu::CloudOpKind::ALL_REDUCE_SUM;
            aop.src_ids = {1};
            aop.dst = {2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
            s2.ops.push_back(aop);
            s2.output_ids = {2};
            const auto p2 = cloudtpu::emit_stablehlo_program(s2);
            const auto e2 = rt.compile_or_get(p2.cache_key, p2.mlir_text, 0, copts);
            PJRT_Buffer* raw_a[1] = {bi0->handle.get()};
            PJRT_Buffer* raw_b[1] = {bi1->handle.get()};
            PJRT_Buffer* const* raw_ptrs[2] = {raw_a, raw_b};
            std::vector<std::vector<cloudtpu::PjrtBufferPtr>> o2;
            rt.execute_spmd(e2, P, raw_ptrs, 1, o2, {cloudtpu::PjrtDType::S32}, {{1, 1, 1, 4}}, true);
            std::vector<int32_t> got2(4, 0);
            rt.buffer_to_host(o2[0][0]->handle.get(), got2.data(), got2.size() * sizeof(int32_t));
            CHECK_TRUE(got2[0] == kOddInt && got2[1] == kOddInt &&
                           got2[2] == (kOddInt * 2) && got2[3] == 3,
                       "P=2 S32 ALL_REDUCE_SUM must preserve 16777217 exactly");

            cloudtpu::GraphProgramSpec rs2;
            rs2.spmd_chips = P;
            rs2.inputs.push_back({1, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}});
            cloudtpu::OpNodeSpec rop;
            rop.kind = cloudtpu::CloudOpKind::REDUCE_SCATTER_SUM;
            rop.collective_dim = 3;
            rop.src_ids = {1};
            rop.dst = {2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}};
            rs2.ops.push_back(rop);
            rs2.output_ids = {2};
            const auto pr2 = cloudtpu::emit_stablehlo_program(rs2);
            const auto er2 = rt.compile_or_get(pr2.cache_key, pr2.mlir_text, 0, copts);
            std::vector<std::vector<cloudtpu::PjrtBufferPtr>> or2;
            rt.execute_spmd(er2, P, raw_ptrs, 1, or2, {cloudtpu::PjrtDType::S32}, {{1, 1, 1, 2}}, true);
            std::vector<int32_t> gr0(2, 0), gr1(2, 0);
            rt.buffer_to_host(or2[0][0]->handle.get(), gr0.data(), gr0.size() * sizeof(int32_t));
            rt.buffer_to_host(or2[1][0]->handle.get(), gr1.data(), gr1.size() * sizeof(int32_t));
            CHECK_TRUE(gr0[0] == kOddInt && gr0[1] == kOddInt &&
                           gr1[0] == (kOddInt * 2) && gr1[1] == 3,
                       "P=2 S32 REDUCE_SCATTER_SUM must preserve 16777217 exactly");
        }

        rt.shutdown();
        std::printf("  [PASS] SPMD REDUCE_SCATTER_SUM -> CPY & exact S32 collective precision (16777217)\n");
    }
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
    test_multichip_device_assignment_and_eviction_lifetime();
    test_mul_mat_dense_and_quant(rt);
    test_rms_rope_softmax_swiglu_getrows_fused(rt);
    test_flash_attn_fully_masked_row(rt);
    test_q4_k_split_mul_mat(rt);
    test_silicon_regressions(rt);
    test_collectives_p1_live_compatible(rt);
    test_spmd_unequal_k_shard_broadcast();
    test_spmd_shard_n_get_rows_and_collectives();
    test_spmd_multichip_mesh(4);  // tpu7x-4t (4x 192 GiB HBM3e)
    test_spmd_multichip_mesh(8);  // v6e-8 (8x 32 GiB HBM)

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
